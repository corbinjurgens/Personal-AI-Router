// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// remoteclient.go is the outbound half of remote engine management: dialing a
// peer's ec surface over pin-based mTLS. It presents this node's cluster leaf
// and pins the peer's exact server cert (nvpair-shared/clustertrust), so a call
// only succeeds against a peer we hold a pin for — the same trust decision the
// server side enforces with a 403.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"nvpair-shared/modeldownload"
	"nvpair-shared/noderec"
)

const (
	remoteResponseHeaderTimeout      = 30 * time.Second
	remoteReadyResponseHeaderTimeout = 11 * time.Minute
)

// Room for the existing 1 MiB native result, the bounded lifecycle snapshot
// and its envelope. Pull frames are bounded before any JSON decoding.
const maxPullStreamFrameBytes = 2 << 20

// Typed JSON-RPC error codes both remote families carry end to end: the node
// policy relay and the engine routes.
const (
	// codeOperationError preserves an authoritative RPC operation failure. Its
	// message, rather than this generic code alone, distinguishes receipt absence.
	codeOperationError = -32000
	operationErrorWire = "operation_error"
	// codePolicyConflict is a policy set whose expected revision is not the
	// persisted one.
	codePolicyConflict = -32010
	// codeLocalVeto is a start, load, resume or other management change the
	// target node's own pause forbids, or accepted peer work that a local Stop
	// on the target cancelled. A saved Off alone does not raise it.
	codeLocalVeto = -32011
	// codeRemoteManagementDenied is a pinned peer's management operation the
	// target's local remote-management permission refused. It is distinct from
	// an unpinned 403 and from codeLocalVeto.
	codeRemoteManagementDenied = -32012
	// codeEngineLocked is a start, load or switch into an engine locked on the
	// target node. It is distinct from codeLocalVeto.
	codeEngineLocked = -32013
)

// wireCodes names the typed codes in an ec 409 body and a stream error frame.
var wireCodes = map[int]string{
	codePolicyConflict:         "policy_conflict",
	codeLocalVeto:              "local_veto",
	codeRemoteManagementDenied: remoteManagementWireCode,
	codeEngineLocked:           "engine_locked",
}

// codedError reaches the JSON-RPC client with its own code: the broker's
// classification of a relayed policy call, or a typed code a peer answered.
type codedError struct {
	code    int
	message string
}

func (e *codedError) Error() string { return e.message }

// codedBody is the body of an ec 409 that carries a typed code.
type codedBody struct {
	Error string `json:"error"`
	Code  string `json:"code"`
}

// writeCodedError refuses an ec request with a typed code: HTTP 409 and
// {"error": message, "code": <wire name>}.
func writeCodedError(w http.ResponseWriter, code int, message string) {
	wire := wireCodes[code]
	if code == codeOperationError {
		wire = operationErrorWire
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusConflict)
	_ = json.NewEncoder(w).Encode(codedBody{Error: message, Code: wire})
}

// remoteStatusError is the error for an ec response with a non-2xx status. A
// 409 whose body names a typed code keeps that code; any other is plain text.
func remoteStatusError(prefix string, status int, body []byte) error {
	var coded codedBody
	if status == http.StatusConflict && json.Unmarshal(body, &coded) == nil {
		if code, ok := codeOfWire(coded.Code); ok {
			return &codedError{code: code, message: fmt.Sprintf("%s: HTTP %d: %s", prefix, status, coded.Error)}
		}
	}
	return fmt.Errorf("%s: HTTP %d: %s", prefix, status, strings.TrimSpace(string(body)))
}

// codeOfWire is the typed code a wire name stands for.
func codeOfWire(name string) (int, bool) {
	// Generic RPC failure envelopes are emitted only by owners that explicitly
	// call writeCodedError for them. Keep wireCodes' existing family membership
	// unchanged for stream codes and policy/settings relay classification.
	if name == operationErrorWire {
		return codeOperationError, true
	}
	for code, wire := range wireCodes {
		if name != "" && wire == name {
			return code, true
		}
	}
	return 0, false
}

// rpcCode is the JSON-RPC error code err reaches a client with.
func rpcCode(err error) int {
	var veto *localVetoError
	if errors.As(err, &veto) && veto.code != 0 {
		return veto.code
	}
	var coded *codedError
	if errors.As(err, &coded) && coded.code != 0 {
		return coded.code
	}
	return codeOperationError
}

// remoteClient talks to one peer's ec surface.
type remoteClient struct {
	controlHTTP *http.Client
	// controlReadyHTTP covers engine readiness, model loading and settings apply.
	controlReadyHTTP *http.Client
	base             string // https://<ip>:<port>
	forget           func()
	// loadBudget returns an engine's model-load response budget as this node's
	// manifest binds it (Executor.loadResponseBudget). Nil means ordinary.
	loadBudget func(engine string) time.Duration
}

func (c *remoteClient) controlClient(ready bool) *http.Client {
	if ready {
		return c.controlReadyHTTP
	}
	return c.controlHTTP
}

// waitsForEngineReadiness reports whether a peer may only answer after an
// engine is healthy or a model is loaded, which the ordinary 30s
// response-header budget cannot cover. loadBudget is the engine's model-load
// response budget; a load waits on the readiness budget when that exceeds the
// ordinary one (Ollama answers a load only once the model is resident).
//
// Cutting such a call off is worse than slow. The initiator's cancellation
// propagates into the peer's in-flight handler, so bringUpCommand tears the
// engine back down — for a delete that means the files are gone, the engine is
// left stopped, and the caller is told the delete failed.
func waitsForEngineReadiness(path string, loadBudget time.Duration) bool {
	if path == settingsPath+"apply" {
		return true
	}
	// controlDeletePath: LM Studio's delete_model declares restart_after, so the
	// peer replies only after the post-delete restart is ready.
	if path == controlLoadPath {
		return loadBudget > remoteResponseHeaderTimeout
	}
	return path == controlStartPath || path == controlDeletePath
}

func newRemoteHTTPClient(base *http.Transport, responseHeaderTimeout time.Duration) *http.Client {
	tr := base.Clone()
	tr.ResponseHeaderTimeout = responseHeaderTimeout
	return &http.Client{Transport: tr}
}

// remoteClient builds a pinned-mTLS client for a target peer. It fails when this
// node isn't clustered or the peer isn't a pinned cluster member. Membership is
// re-derived per call, so remote management becomes available as soon as this
// node joins a cluster and stops the moment it leaves.
func (m *Manager) remoteClient(ctx context.Context, peer ecPeer) (*remoteClient, error) {
	m.mesh.Refresh()
	if !m.mesh.Clustered() {
		return nil, fmt.Errorf("this node is not clustered; remote engine management is unavailable")
	}
	if _, ok := m.mesh.PeerServiceIncarnation(peer.clusterUUID); !ok {
		return nil, fmt.Errorf("node %q is not a pinned cluster peer", peer.nodeID)
	}
	// No total client timeout: install/pull run for minutes, bounded on the
	// server by the executor's action ceiling. ResponseHeaderTimeout still
	// guards a peer that accepts the connection but never replies (ordinary
	// vs readiness budgets live on the two adapters).
	candidates := make([]string, 0, len(peer.addresses))
	for _, address := range peer.addresses {
		candidates = append(candidates, net.JoinHostPort(address, strconv.Itoa(peer.port)))
	}
	hostport := m.addrs.ChooseWithin(ctx, peer.nodeID, candidates)
	if !m.peers.selectAddress(peer, hostport) {
		return nil, fmt.Errorf("node %q ec endpoint changed before dispatch", peer.nodeID)
	}
	control := m.controlHTTP.Client(peer.clusterUUID, noderec.ServiceEngineControl)
	ready := m.controlReadyHTTP.Client(peer.clusterUUID, noderec.ServiceEngineControl)
	return &remoteClient{
		controlHTTP:      control,
		controlReadyHTTP: ready,
		base:             "https://" + hostport,
		forget:           func() { m.addrs.Forget(peer.nodeID) },
		loadBudget:       m.exec.loadResponseBudget,
	}, nil
}

func (c *remoteClient) forgetAddress() {
	if c.forget != nil {
		c.forget()
	}
}

// Read one byte past the cap so a complete prefix cannot hide overflow or an
// incomplete HTTP body. Only the real body's EOF establishes completeness.
func readRemoteResponse(body io.Reader, limit int64) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("response exceeds %d-byte limit", limit)
	}
	return data, nil
}

// getEngines fetches the peer's engine status list (GET /v1/engines), returning
// the raw {"engines":[...]} object.
func (c *remoteClient) getEngines(ctx context.Context) (json.RawMessage, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+controlEnginesPath, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.controlClient(false).Do(req)
	if err != nil {
		c.forgetAddress()
		return nil, err
	}
	defer resp.Body.Close()
	data, err := readRemoteResponse(resp.Body, 8<<20)
	if err != nil {
		return nil, fmt.Errorf("remote engines: read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("remote engines: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	return json.RawMessage(data), nil
}

// getRemoteManagement reads the target's effective remote-management permission
// for this node (GET /v1/remote-management). It is a caller-only read: the
// target answers for the authenticated certificate, never for a peer selector.
// A target that predates the route answers 404 (or 405 on an unknown method) and
// is reported unsupported — never as an allowed peer.
func (c *remoteClient) getRemoteManagement(ctx context.Context) (remoteManagementCapability, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+controlRemoteManagementPath, nil)
	if err != nil {
		return remoteManagementCapability{}, err
	}
	resp, err := c.controlClient(false).Do(req)
	if err != nil {
		c.forgetAddress()
		return remoteManagementCapability{}, err
	}
	defer resp.Body.Close()
	data, err := readRemoteResponse(resp.Body, 64<<10)
	if err != nil {
		return remoteManagementCapability{}, fmt.Errorf("remote management: read response: %w", err)
	}
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMethodNotAllowed {
		return remoteManagementCapability{Version: remoteManagementVersion, State: remoteManagementStateUnsupported}, nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return remoteManagementCapability{}, remoteStatusError("remote management", resp.StatusCode, data)
	}
	var view remoteManagementView
	if err := json.Unmarshal(data, &view); err != nil {
		return remoteManagementCapability{}, fmt.Errorf("remote management: decode response: %w", err)
	}
	if !view.valid() {
		return remoteManagementCapability{}, fmt.Errorf("remote management: unrecognized response")
	}
	// An inconsistent target answer never becomes an allow.
	return remoteManagementCapability{
		Version:  view.Version,
		Allowed:  view.Allowed && view.State == remoteManagementStateAllowed,
		Revision: view.Revision,
		State:    view.State,
	}, nil
}

// postJSON POSTs body to a non-streaming ec endpoint and returns the raw JSON
// response (e.g. an EngineStatus from start/stop).
func (c *remoteClient) postJSON(ctx context.Context, path, engine string, body any) (json.RawMessage, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+path, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	var loadBudget time.Duration
	if path == controlLoadPath && c.loadBudget != nil {
		loadBudget = c.loadBudget(engine)
	}
	client := c.controlClient(waitsForEngineReadiness(path, loadBudget))
	resp, err := client.Do(req)
	if err != nil {
		c.forgetAddress()
		return nil, err
	}
	defer resp.Body.Close()
	data, err := readRemoteResponse(resp.Body, 8<<20)
	if err != nil {
		return nil, fmt.Errorf("remote %s: read response; operation outcome may be uncertain: %w", path, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, remoteStatusError("remote "+path, resp.StatusCode, data)
	}
	return json.RawMessage(data), nil
}

// stream POSTs body to a streaming ec endpoint and consumes its NDJSON frames,
// calling onProgress for each progress frame and returning the terminal result
// frame. An error frame (or a stream that ends without a result) is an error.
func (c *remoteClient) stream(ctx context.Context, path string, body any, onProgress func(streamFrame)) (streamFrame, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return streamFrame{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+path, bytes.NewReader(b))
	if err != nil {
		return streamFrame{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.controlClient(false).Do(req)
	if err != nil {
		c.forgetAddress()
		return streamFrame{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return streamFrame{}, remoteStatusError("remote "+path, resp.StatusCode, data)
	}

	dec := json.NewDecoder(resp.Body)
	decodeFrame := func(frame *streamFrame) error { return dec.Decode(frame) }
	var pullIdentity *pullStreamIdentity
	if path == controlPullPath {
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 64*1024), maxPullStreamFrameBytes+1)
		decodeFrame = func(frame *streamFrame) error {
			for scanner.Scan() {
				line := scanner.Bytes()
				if len(line) > maxPullStreamFrameBytes {
					return fmt.Errorf("pull frame exceeds %d-byte limit", maxPullStreamFrameBytes)
				}
				if len(bytes.TrimSpace(line)) == 0 {
					continue
				}
				return json.Unmarshal(line, frame)
			}
			if errors.Is(scanner.Err(), bufio.ErrTooLong) {
				return fmt.Errorf("pull frame exceeds %d-byte limit", maxPullStreamFrameBytes)
			}
			if err := scanner.Err(); err != nil {
				return err
			}
			return io.EOF
		}
		var expected pullRequest
		if err := json.Unmarshal(b, &expected); err != nil {
			return streamFrame{}, fmt.Errorf("remote pull: invalid request identity")
		}
		if expected.Model == "" {
			expected.Model = modelFromParams(expected.Params)
		}
		pullIdentity = &pullStreamIdentity{request: expected}
	}
	var result streamFrame
	haveResult := false
	for {
		var f streamFrame
		if derr := decodeFrame(&f); derr != nil {
			if derr == io.EOF {
				break
			}
			return streamFrame{}, fmt.Errorf("remote %s: decode frame: %w", path, derr)
		}
		if pullIdentity != nil {
			if haveResult {
				return streamFrame{}, fmt.Errorf("remote pull: frame after terminal result")
			}
			if err := pullIdentity.validate(f); err != nil {
				return streamFrame{}, err
			}
		}
		switch f.Type {
		case "progress":
			if onProgress != nil {
				onProgress(f)
			}
		case "result":
			result = f
			haveResult = true
		case "error":
			if code, ok := codeOfWire(f.Code); ok {
				return f, &codedError{code: code, message: f.Message}
			}
			return f, fmt.Errorf("%s", f.Message)
		}
	}
	if !haveResult {
		return streamFrame{}, fmt.Errorf("remote %s: stream ended without a result", path)
	}
	return result, nil
}

// pullStreamIdentity binds the caller's correlation to one target-owned
// download. Installs and file hashing retain their separate stream contracts.
type pullStreamIdentity struct {
	request pullRequest
	id      string
	last    *modeldownload.Snapshot
}

func (v *pullStreamIdentity) validate(f streamFrame) error {
	if v.request.OpID == "" || v.request.Model == "" || f.OpID != v.request.OpID ||
		f.Engine != v.request.Engine || f.Op != "pull" || f.Model != v.request.Model {
		return errors.New("remote pull: mismatched request identity")
	}
	if f.Type != "progress" && f.Type != "result" && f.Type != "error" {
		return errors.New("remote pull: unsupported frame type")
	}
	if f.DownloadState == nil || f.DownloadState.Validate() != nil {
		return errors.New("remote pull: missing or invalid download state")
	}
	state := f.DownloadState
	var entry *modeldownload.Entry
	for i := range state.Entries {
		if state.Entries[i].ID == f.ID && state.Entries[i].Engine == f.Engine && state.Entries[i].Model == f.Model {
			entry = &state.Entries[i]
			break
		}
	}
	if entry == nil || v.id != "" && f.ID != v.id {
		return errors.New("remote pull: mismatched target download identity")
	}
	if v.last != nil {
		if state.SourceID != v.last.SourceID {
			return errors.New("remote pull: target source changed")
		}
		// Validate established canonical decimal uint64s, so conversion cannot fail.
		revision, _ := strconv.ParseUint(state.Revision, 10, 64)
		previous, _ := strconv.ParseUint(v.last.Revision, 10, 64)
		if revision < previous || revision == previous && !state.Equal(*v.last) {
			return errors.New("remote pull: inconsistent download revision")
		}
	}
	if f.Type == "result" && entry.Phase != modeldownload.PhaseComplete ||
		f.Type == "error" && entry.Phase != modeldownload.PhaseFailed && entry.Phase != modeldownload.PhaseUnknown {
		return errors.New("remote pull: terminal outcome does not match download state")
	}
	copy := state.Clone()
	v.last, v.id = &copy, f.ID
	return nil
}
