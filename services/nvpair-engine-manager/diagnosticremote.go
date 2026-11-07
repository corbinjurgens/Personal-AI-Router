// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"nvpair-shared/modelconfig"
)

const diagnosticPath = "/v1/model-diagnostic/"

var diagnosticLocalPaths = map[string]string{
	"engine:diagnostic-preview": diagnosticPath + "preview",
	"engine:diagnostic-start":   diagnosticPath + "start",
	"engine:diagnostic-status":  diagnosticPath + "status",
	"engine:diagnostic-cancel":  diagnosticPath + "cancel",
}

// Decode before replacing the transport alias. Inbound requests are always
// local at the target; an authenticated peer cannot request a third hop.
func normalizeDiagnosticParams(method string, raw []byte, target string) (json.RawMessage, string, error) {
	var value any
	var nodeID string
	var err error
	switch method {
	case "engine:diagnostic-preview":
		var p modelconfig.DiagnosticPreviewRequest
		err = modelconfig.DecodeStrict(raw, &p)
		if err == nil {
			err = p.Validate()
		}
		nodeID, p.NodeID = p.NodeID, target
		value = p
	case "engine:diagnostic-start":
		var p modelconfig.DiagnosticStartRequest
		err = modelconfig.DecodeStrict(raw, &p)
		if err == nil {
			err = p.Validate()
		}
		nodeID, p.NodeID = p.NodeID, target
		value = p
	case "engine:diagnostic-status", "engine:diagnostic-cancel":
		var p modelconfig.DiagnosticRunRef
		err = modelconfig.DecodeStrict(raw, &p)
		validation := p
		if method == "engine:diagnostic-status" && validation.RunID == "" {
			validation.RunID = "active-lookup"
		}
		if err == nil {
			err = validation.Validate()
		}
		nodeID, p.NodeID = p.NodeID, target
		value = p
	default:
		return nil, "", &codedError{code: -32601, message: "unknown diagnostic method"}
	}
	if err != nil {
		return nil, "", &codedError{code: -32602, message: err.Error()}
	}
	params, err := json.Marshal(value)
	return params, nodeID, err
}

func (s *controlServer) diagnosticRoutes() []controlRoute {
	routes := make([]controlRoute, 0, len(diagnosticLocalPaths))
	for method, path := range diagnosticLocalPaths {
		class := controlMutation
		if method == "engine:diagnostic-status" {
			class = controlRead
		}
		routes = append(routes, controlRoute{path, http.MethodPost, class, "model-diagnostic-" + strings.TrimPrefix(method, "engine:diagnostic-"), s.diagnosticHandler(method)})
	}
	return routes
}

func (s *controlServer) diagnosticHandler(method string) pinnedPeerHandler {
	return func(w http.ResponseWriter, r *http.Request, caller string) {
		if s.exec == nil || s.exec.diagnosticParent == nil {
			http.Error(w, "diagnostic coordinator unavailable", http.StatusServiceUnavailable)
			return
		}
		raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, modelconfig.MaxPayloadBytes))
		if err != nil {
			http.Error(w, "invalid diagnostic request", http.StatusBadRequest)
			return
		}
		params, _, err := normalizeDiagnosticParams(method, raw, "")
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		// Halt/maintenance and a fresh permission read are target broker
		// acceptance gates. Do not use remoteVeto: configured pause is allowed.
		result, err := s.exec.diagnosticParent(r.Context(), method, caller, params)
		if err != nil {
			var coded *codedError
			if errors.As(err, &coded) && coded.code == -32602 {
				http.Error(w, err.Error(), http.StatusBadRequest)
			} else if errors.As(err, &coded) && (wireCodes[coded.code] != "" || coded.code == codeOperationError) {
				writeCodedError(w, coded.code, err.Error())
			} else {
				http.Error(w, "diagnostic request failed; read status to reconcile", http.StatusBadGateway)
			}
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(result)
	}
}

func (m *Manager) remoteDiagnostic(ctx context.Context, msg *Message) {
	if m.mesh == nil {
		m.respondOrErr(msg, nil, errors.New("diagnostic target unavailable: node is not clustered"))
		return
	}
	method := strings.Replace(msg.Method, "engine:remote-", "engine:", 1)
	result, err := callRemoteDiagnostic(ctx, method, msg.Params, m.peers.lookup, m.remoteClient)
	m.respondOrErr(msg, result, err)
}

func callRemoteDiagnostic(ctx context.Context, method string, raw json.RawMessage,
	lookup func(string) (ecPeer, bool), clientFor func(context.Context, ecPeer) (*remoteClient, error),
) (json.RawMessage, error) {
	path, ok := diagnosticLocalPaths[method]
	if !ok {
		return nil, &codedError{code: -32601, message: "unknown remote diagnostic method"}
	}
	params, nodeID, err := normalizeDiagnosticParams(method, raw, "")
	if err != nil {
		return nil, err
	}
	if nodeID == "" {
		return nil, &codedError{code: -32602, message: "diagnostic nodeId required"}
	}
	peer, ok := lookup(nodeID)
	if !ok {
		return nil, errors.New("diagnostic target is not a discovered engine-control peer")
	}
	client, err := clientFor(ctx, peer)
	if err != nil {
		return nil, err
	}
	budget := 29 * time.Second
	if method == "engine:diagnostic-cancel" {
		budget = 205 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	result, err := client.postDiagnostic(ctx, path, params)
	if err != nil {
		return nil, err
	}
	return projectRemoteDiagnostic(method, raw, result, peer.nodeID)
}

func (c *remoteClient) postDiagnostic(ctx context.Context, path string, params json.RawMessage) (json.RawMessage, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+path, bytes.NewReader(params))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	// Cancellation can join physical cleanup beyond the ordinary header
	// deadline. Both adapters use the same NewPeerServiceClient transport.
	response, err := c.controlClient(path == diagnosticPath+"cancel").Do(req)
	if err != nil {
		c.forgetAddress()
		return nil, err
	}
	defer response.Body.Close()
	data, err := readRemoteResponse(response.Body, modelconfig.MaxPayloadBytes)
	if err != nil {
		return nil, fmt.Errorf("remote diagnostic response uncertain: %w", err)
	}
	if response.StatusCode == http.StatusBadRequest {
		return nil, &codedError{code: -32602, message: "invalid remote diagnostic request"}
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, remoteStatusError("remote diagnostic", response.StatusCode, data)
	}
	return data, nil
}

type diagnosticPeerRun struct {
	modelconfig.DiagnosticRun
	NodeID string `json:"nodeId,omitempty"`
}

// Re-project the public contract at the origin. Private manager tokens,
// markers, identities and observations cannot cross this boundary, even in a
// malformed successful response. Never adopt a different run after cancellation.
func projectRemoteDiagnostic(method string, request, raw json.RawMessage, nodeID string) (json.RawMessage, error) {
	if method == "engine:diagnostic-preview" {
		var p modelconfig.DiagnosticPreview
		var expected modelconfig.DiagnosticPreviewRequest
		if err := modelconfig.DecodeStrict(request, &expected); err != nil {
			return nil, err
		}
		if err := modelconfig.DecodeStrict(raw, &p); err != nil {
			return nil, err
		}
		if err := p.Validate(); err != nil {
			return nil, err
		}
		if p.NodeID != nodeID || p.Engine != expected.Engine || p.ModelID != expected.ModelID || p.ProfileKey != expected.ProfileKey {
			return nil, errors.New("remote diagnostic preview target mismatch")
		}
		return json.Marshal(p.Clone())
	}
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		if method == "engine:diagnostic-start" {
			return nil, errors.New("remote diagnostic acceptance missing")
		}
		return json.RawMessage("null"), nil
	}
	var run diagnosticPeerRun
	if err := modelconfig.DecodeStrict(raw, &run); err != nil {
		return nil, err
	}
	if err := run.Validate(); err != nil {
		return nil, err
	}
	if run.NodeID != "" && run.NodeID != nodeID {
		return nil, errors.New("remote diagnostic run target mismatch")
	}
	if method == "engine:diagnostic-start" {
		var expected modelconfig.DiagnosticStartRequest
		if err := modelconfig.DecodeStrict(request, &expected); err != nil {
			return nil, err
		}
		if run.Engine != expected.Engine || run.ModelID != expected.ModelID || run.ProfileKey != expected.ProfileKey {
			return nil, errors.New("remote diagnostic acceptance target mismatch")
		}
	} else {
		var expected modelconfig.DiagnosticRunRef
		if err := modelconfig.DecodeStrict(request, &expected); err != nil {
			return nil, err
		}
		if expected.RunID != "" && run.RunID != expected.RunID {
			return nil, errors.New("remote diagnostic run mismatch")
		}
	}
	return json.Marshal(diagnosticPeerRun{DiagnosticRun: run.DiagnosticRun.Clone(), NodeID: nodeID})
}
