// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"

	"nvpair-shared/cors"
)

// serveModelList serves the same policy-derived catalog as the routing listener.
// Engine inventories remain available through the engine inventory views.
func (f *facade) serveModelList(w http.ResponseWriter, r *http.Request, matchedRoute route, candidates []candidate) (int, error) {
	if r.Header.Get("Origin") != "" {
		status, headers, err := f.catalogPermissions(r, matchedRoute, candidates)
		for key, values := range headers {
			w.Header()[key] = values
		}
		if err != nil {
			http.Error(w, `{"error":"model catalog browser permission refused"}`, status)
			return status, err
		}
	}
	pol := f.host.admission.currentPolicy()
	models := f.host.profileModelEntries(pol, f)
	body, err := json.Marshal(struct {
		Object string            `json:"object"`
		Data   []json.RawMessage `json:"data"`
	}{Object: "list", Data: models})
	if err != nil {
		http.Error(w, `{"error":"failed to encode model catalog"}`, http.StatusInternalServerError)
		return http.StatusInternalServerError, err
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
	return http.StatusOK, nil
}

// catalogPermissions retains the facade's engine-owned browser policy. Only
// response headers participate; native catalog contents never supply entries.
func (f *facade) catalogPermissions(r *http.Request, matchedRoute route, candidates []candidate) (int, http.Header, error) {
	type result struct {
		headers http.Header
		denied  bool
		invalid bool
	}
	results := make([]result, len(candidates))
	var wg sync.WaitGroup
	for i, cand := range candidates {
		wg.Add(1)
		go func(i int, cand candidate) {
			defer wg.Done()
			target := *cand.url
			target.Path = matchedRoute.upstreamPath()
			target.RawQuery = r.URL.RawQuery
			req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, target.String(), nil)
			if err != nil {
				return
			}
			req.Header = cors.FanoutHeaders(r.Header)
			if cand.peerUUID == "" {
				stripInternalRoutingHeaders(req.Header)
				req.Header.Del(peerStreamFramingHeader)
			}
			if cand.self {
				g := cand.fac
				if g == nil {
					g = f
				}
				withLocalCredential(req.Header, g.localCredentialFor(cand.url))
			}
			client := &http.Client{Transport: f.host.candidateTransport(cand), Timeout: proxyDialTimeout,
				CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
			resp, err := client.Do(req)
			if err != nil {
				f.targets.Forget(cand.id)
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode >= http.StatusInternalServerError {
				return
			}
			results[i].headers = cors.EndToEndHeaders(resp.Header)
			results[i].denied = resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices || !cors.AllowsOrigin(resp.Header, r.Header.Get("Origin"))
			if results[i].denied {
				return
			}
			// Retain the existing browser response-validity gate without using
			// any native model record in the policy-derived catalog.
			body, err := io.ReadAll(io.LimitReader(resp.Body, maxModelListBytes+1))
			var envelope struct {
				Data *[]json.RawMessage `json:"data"`
			}
			results[i].invalid = resp.StatusCode != http.StatusOK || err != nil || len(body) > maxModelListBytes || json.Unmarshal(body, &envelope) != nil || envelope.Data == nil
		}(i, cand)
	}
	wg.Wait()
	var responding []http.Header
	invalid := false
	for _, result := range results {
		if result.denied {
			return http.StatusForbidden, nil, fmt.Errorf("engine denied catalog origin")
		}
		if result.headers != nil {
			responding = append(responding, result.headers)
		}
		invalid = invalid || result.invalid
	}
	if len(responding) == 0 {
		return http.StatusBadGateway, nil, fmt.Errorf("no engine answered catalog origin check")
	}
	headers, allowed := cors.Combine(r, responding)
	if !allowed {
		return http.StatusForbidden, nil, fmt.Errorf("engines disagree on catalog origin")
	}
	if invalid {
		return http.StatusBadGateway, headers, fmt.Errorf("invalid engine catalog permission response")
	}
	return http.StatusOK, headers, nil
}
