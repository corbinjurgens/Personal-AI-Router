// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"sync"

	"nvpair-shared/cors"
	"nvpair-shared/nodepolicy"
)

var modelListClient = &http.Client{
	Transport: &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   proxyDialTimeout,
			KeepAlive: proxyKeepAlive,
		}).DialContext,
		ResponseHeaderTimeout: proxyDialTimeout,
		MaxIdleConns:          proxyMaxIdleConns,
		IdleConnTimeout:       proxyIdleConnTimeout,
	},
	Timeout: proxyDialTimeout,
}

type modelListItem struct {
	key    string
	digest string
	raw    json.RawMessage
}

type modelListResult struct {
	headers http.Header
	denied  bool
	// unavailable marks a candidate that never answered — unreachable or a
	// server error — as opposed to one that answered unusably. Only the former
	// is excluded from the CORS intersection.
	unavailable bool
	items       []modelListItem
	ok          bool
	err         error
}

// serveModelList queries every candidate concurrently and returns the engine's
// model-list envelope with duplicate model records removed. Results are merged
// in candidate order, not completion order, so duplicate metadata is
// deterministic while an unavailable peer cannot hide healthy inventories.
//
// matchedRoute selects the wire dialect and the optional upstream path. The
// client-facing path remains unchanged in telemetry and in the merged response.
func (f *facade) serveModelList(w http.ResponseWriter, r *http.Request, matchedRoute route, candidates []candidate) (int, error) {
	p := f.host
	openAI := matchedRoute.Role == roleModelListOpenAIGET
	writeJSON := func(status int, body []byte) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.WriteHeader(status)
		_, _ = w.Write(body)
	}
	results := make([]modelListResult, len(candidates))
	var wg sync.WaitGroup
	for i, cand := range candidates {
		target := *cand.url
		target.Path = matchedRoute.upstreamPath()
		if matchedRoute.UpstreamPath == "" {
			target.RawPath = r.URL.RawPath
		} else {
			target.RawPath = ""
		}
		target.RawQuery = r.URL.RawQuery
		upstream, err := http.NewRequestWithContext(r.Context(), http.MethodGet, target.String(), nil)
		if err != nil {
			results[i].err = err
			continue
		}
		upstream.Header = cors.FanoutHeaders(r.Header)
		if cand.peerUUID == "" {
			stripInternalRoutingHeaders(upstream.Header)
			upstream.Header.Del(peerStreamFramingHeader)
		}
		upstream.Header.Del("Content-Length")
		if upstream.Header.Get("Accept") == "" {
			upstream.Header.Set("Accept", "application/json")
		}
		if cand.self {
			fac := cand.fac
			if fac == nil {
				fac = f
			}
			withLocalCredential(upstream.Header, fac.localCredentialFor(cand.url))
		}

		// A cluster-peer candidate is queried over mTLS to its promoted proxy;
		// self/manual candidates use the shared plain client.
		clientCopy := *modelListClient
		clientCopy.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
		client := &clientCopy
		if cand.peerUUID != "" {
			client.Transport = p.candidateTransport(cand)
		}

		wg.Add(1)
		go func(i int, cand candidate, req *http.Request, client *http.Client) {
			defer wg.Done()
			resp, err := client.Do(req)
			if err != nil {
				f.targets.Forget(cand.id)
				results[i] = modelListResult{unavailable: true, err: err}
				return
			}
			defer resp.Body.Close()
			responseHeaders := cors.EndToEndHeaders(resp.Header)
			results[i].headers = responseHeaders
			if r.Header.Get("Origin") != "" && (resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden || (resp.StatusCode == http.StatusOK && !cors.AllowsOrigin(responseHeaders, r.Header.Get("Origin")))) {
				results[i].denied = true
				return
			}
			if resp.StatusCode >= 500 {
				results[i] = modelListResult{unavailable: true, err: fmt.Errorf("upstream returned %s", resp.Status)}
				return
			}
			if resp.StatusCode != http.StatusOK {
				results[i].err = fmt.Errorf("upstream returned %s", resp.Status)
				return
			}
			body, err := io.ReadAll(io.LimitReader(resp.Body, maxModelListBytes+1))
			if err != nil {
				results[i].err = err
				return
			}
			if len(body) > maxModelListBytes {
				results[i].err = fmt.Errorf("model list exceeds %d bytes", maxModelListBytes)
				return
			}
			var envelope struct {
				Models *[]json.RawMessage `json:"models"`
				Data   *[]json.RawMessage `json:"data"`
			}
			if err := json.Unmarshal(body, &envelope); err != nil {
				results[i].err = err
				return
			}
			models := envelope.Models
			if openAI {
				models = envelope.Data
			}
			if models == nil {
				results[i].err = fmt.Errorf("upstream response has no model array")
				return
			}
			items := make([]modelListItem, 0, len(*models))
			for _, raw := range *models {
				if !isJSONObject(raw) {
					results[i].err = fmt.Errorf("invalid model record")
					return
				}
				invalid, hasIdentity := false, false
				visitObject(raw, func(field []byte, start, end int) {
					if (openAI && jsonKeyEqual(field, "id", true)) || (!openAI && (jsonKeyEqual(field, "model", true) || jsonKeyEqual(field, "name", true))) {
						hasIdentity = true
					}
					if (jsonKeyEqual(field, "id", true) || jsonKeyEqual(field, "model", true) || jsonKeyEqual(field, "name", true)) && nodepolicy.ValidateModelIdentifierJSON(raw[start:end]) != nil {
						invalid = true
					}
				})
				if !hasIdentity {
					results[i].err = fmt.Errorf("model record has no identity")
					return
				}
				if invalid {
					continue
				}
				var identity struct {
					ID     string `json:"id"`
					Model  string `json:"model"`
					Name   string `json:"name"`
					Digest string `json:"digest"`
				}
				if err := json.Unmarshal(raw, &identity); err != nil {
					results[i].err = fmt.Errorf("invalid model record: %w", err)
					return
				}
				key := identity.ID
				if !openAI {
					key = identity.Model
					if key == "" {
						key = identity.Name
					}
					if nodepolicy.ValidateModelIdentifier(key) != nil {
						continue
					}
					key = f.profile.normalizeModel(key)
				}
				if nodepolicy.ValidateModelIdentifier(key) != nil {
					continue
				}
				items = append(items, modelListItem{key: key, digest: identity.Digest, raw: raw})
			}
			results[i] = modelListResult{items: items, ok: true, headers: responseHeaders}
		}(i, cand, upstream, client)
	}
	wg.Wait()

	var combinedHeaders http.Header
	if r.Header.Get("Origin") != "" {
		headers := make([]http.Header, 0, len(results))
		invalidInventory := false
		for _, result := range results {
			// An engine that answered and refused the origin is a real denial;
			// its peers must not vote it away.
			if result.denied {
				writeJSON(http.StatusForbidden, []byte(`{"error":"an engine denied access to the model list"}`))
				return http.StatusForbidden, fmt.Errorf("engine denied model list")
			}
			// An engine that never answered expressed no opinion. Its models
			// are dropped from the merged list below, so intersecting its
			// absent headers would let one unreachable node deny the browser
			// access to every reachable one.
			if result.unavailable {
				continue
			}
			// An engine that answered unusably is reachable and broken. Do not
			// quietly hand a browser a list that silently omits it.
			if !result.ok {
				invalidInventory = true
			}
			headers = append(headers, result.headers)
		}
		if len(headers) == 0 {
			writeJSON(http.StatusBadGateway, []byte(`{"error":"model inventory unavailable"}`))
			return http.StatusBadGateway, fmt.Errorf("model inventory unavailable")
		}
		var allowed bool
		combinedHeaders, allowed = cors.Combine(r, headers)
		if invalidInventory {
			if allowed {
				for key, values := range combinedHeaders {
					w.Header()[key] = values
				}
			}
			writeJSON(http.StatusBadGateway, []byte(`{"error":"model inventory unavailable"}`))
			return http.StatusBadGateway, fmt.Errorf("model inventory unavailable")
		}
		if !allowed {
			writeJSON(http.StatusForbidden, []byte(`{"error":"engines do not all permit access to the model list"}`))
			return http.StatusForbidden, fmt.Errorf("engines denied model list")
		}
	}

	success := false
	models := make([]json.RawMessage, 0)
	seen := make(map[string]modelListItem)
	for i, result := range results {
		if !result.ok {
			slog.Debug("model list candidate unavailable",
				"node_id", candidates[i].id, "target", candidates[i].url.Host, "err", result.err)
			continue
		}
		success = true
		for _, item := range result.items {
			if previous, duplicate := seen[item.key]; duplicate {
				if previous.digest != "" && item.digest != "" && previous.digest != item.digest {
					slog.Warn("conflicting model digests across candidates",
						"model", item.key, "first_digest", previous.digest,
						"other_digest", item.digest, "node_id", candidates[i].id)
				}
				continue
			}
			seen[item.key] = item
			models = append(models, item.raw)
		}
	}
	var aliasModels []json.RawMessage
	if openAI {
		aliasModels = f.tierModelEntries(seen)
		models = append(models, aliasModels...)
	}
	// Configured aliases are served without any upstream. The Origin checks
	// above still apply: they speak for the engines' own CORS policy.
	if !success && len(aliasModels) == 0 {
		err := fmt.Errorf("no valid model list from %d candidate(s)", len(candidates))
		writeJSON(http.StatusServiceUnavailable, []byte(`{"error":"model inventory unavailable"}`))
		return http.StatusServiceUnavailable, err
	}
	var body []byte
	var err error
	if openAI {
		body, err = json.Marshal(struct {
			Object string            `json:"object"`
			Data   []json.RawMessage `json:"data"`
		}{Object: "list", Data: models})
	} else {
		body, err = json.Marshal(struct {
			Models []json.RawMessage `json:"models"`
		}{Models: models})
	}
	if err != nil {
		writeJSON(http.StatusInternalServerError, []byte(`{"error":"failed to encode model inventory"}`))
		return http.StatusInternalServerError, err
	}
	for key, values := range combinedHeaders {
		w.Header()[key] = values
	}
	writeJSON(http.StatusOK, body)
	return http.StatusOK, nil
}
