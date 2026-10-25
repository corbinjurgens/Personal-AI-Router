// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// controlstream.go holds the long-running ec endpoints that stream live
// progress: remote install and remote model pull. Both return a chunked
// application/x-ndjson body — zero or more {"type":"progress",...} frames as the
// operation advances, terminated by exactly one {"type":"result",...} or
// {"type":"error",...} frame. The initiating node relays each progress frame up
// to its UI as engine:remote-progress and settles the RPC on the terminal frame.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"nvpair-shared/modeldownload"
)

// controlInstallPath streams a remote engine install.
const controlInstallPath = "/v1/engines/install"

// controlPullPath streams a remote model pull.
const controlPullPath = "/v1/models/pull"

// maxControlBody caps a request body on the ec surface. Requests are small
// JSON control messages; the large payloads are downloads the target fetches
// itself, never carried over this wire.
const maxControlBody = 1 << 20 // 1 MiB

// streamFrame is one NDJSON frame on the ec streaming endpoints. Type is
// "progress", "result", or "error". Percent is omitted when indeterminate (0).
type streamFrame struct {
	Type              string                  `json:"type"`
	OpID              string                  `json:"opId,omitempty"`
	Engine            string                  `json:"engine,omitempty"`
	Op                string                  `json:"op,omitempty"`
	Stage             string                  `json:"stage,omitempty"`
	Percent           int                     `json:"percent,omitempty"`
	Message           string                  `json:"message,omitempty"`
	Status            *EngineStatus           `json:"status,omitempty"`
	Result            json.RawMessage         `json:"result,omitempty"`
	Node              string                  `json:"node,omitempty"`
	SourceNodeID      string                  `json:"sourceNodeId,omitempty"`
	DestinationNodeID string                  `json:"destinationNodeId,omitempty"`
	Model             string                  `json:"model,omitempty"`
	ExportID          string                  `json:"exportId,omitempty"`
	File              string                  `json:"file,omitempty"`
	BytesDone         int64                   `json:"bytesDone,omitempty"`
	BytesTotal        int64                   `json:"bytesTotal,omitempty"`
	ID                string                  `json:"id,omitempty"`
	DownloadState     *modeldownload.Snapshot `json:"downloadState,omitempty"`
	// Code is the wire name of a typed error code on an error frame, such as
	// "local_veto"; empty for an untyped failure.
	Code string `json:"code,omitempty"`
}

// installRequest is the POST /v1/engines/install body.
type installRequest struct {
	OpID   string `json:"opId"`
	Engine string `json:"engine"`
	Start  bool   `json:"start"`
}

// pullRequest is the POST /v1/models/pull body. Params, when set, is passed
// through to the manifest pull_model action verbatim (same as engine:action);
// otherwise Model is used to build a default body.
type pullRequest struct {
	OpID   string          `json:"opId"`
	Engine string          `json:"engine"`
	Model  string          `json:"model,omitempty"`
	Params json.RawMessage `json:"params,omitempty"`
}

// handleInstall runs an install (optionally starting the engine after) and
// streams its progress, ending with the resulting EngineStatus.
func (s *controlServer) handleInstall(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req installRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, maxControlBody)).Decode(&req); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	if req.Engine == "" {
		http.Error(w, `"engine" is required`, http.StatusBadRequest)
		return
	}
	// Install with start is refused up front, as a 409, when a local veto
	// forbids the start; a veto arriving during the install cancels only the
	// start, and the start rechecks it under the lifecycle lock.
	startCtx := r.Context()
	if req.Start {
		ctx, done, err := s.exec.beginRemote(r.Context(), req.Engine)
		if err != nil {
			writeRemoteError(w, err)
			return
		}
		defer done()
		startCtx = ctx
	}
	s.streamOp(w, r, req.OpID, req.Engine, "install", func(ctx context.Context) (streamFrame, error) {
		if req.Start {
			if err := s.exec.InstallWithStart(ctx, startCtx, req.Engine, startOpts{Origin: intentPeer}); err != nil {
				return streamFrame{}, vetoCause(startCtx, err)
			}
		} else if err := s.exec.Install(ctx, req.Engine); err != nil {
			return streamFrame{}, err
		}
		st, err := s.exec.Status(req.Engine)
		if err != nil {
			return streamFrame{}, err
		}
		return streamFrame{Type: "result", OpID: req.OpID, Engine: req.Engine, Op: "install", Status: &st}, nil
	})
}

// handlePull runs a model pull and streams its download progress, ending with
// the pull_model action's final result.
func (s *controlServer) handlePull(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req pullRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, maxControlBody)).Decode(&req); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	if req.Engine == "" {
		http.Error(w, `"engine" is required`, http.StatusBadRequest)
		return
	}
	if req.Model == "" && len(req.Params) == 0 {
		http.Error(w, `"model" or "params" is required`, http.StatusBadRequest)
		return
	}
	manifest, ok := s.exec.reg.Get(req.Engine)
	if !ok {
		http.Error(w, "unknown engine", http.StatusBadRequest)
		return
	}
	params := req.Params
	if len(params) == 0 || string(params) == "null" {
		var err error
		params, err = s.exec.pullParams(req.Engine, req.Model)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}
	if _, err := manifest.pullProgressModel(params); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(streamFrame{Type: "error", OpID: req.OpID, Engine: req.Engine, Op: "pull", Message: err.Error()})
		return
	}
	token, err := s.exec.reserveModelPull(r.Context(), req.Engine, req.Model, req.Params)
	if err != nil {
		writeRemoteError(w, err)
		return
	}
	defer token.cancelUnstarted()
	events, cancelSub := s.exec.progress.subscribeDownload(token.id, token.engine, token.model)
	defer cancelSub()
	initial := token.progressEvent("queued", 0)
	s.streamEvents(w, r, req.OpID, token.engine, "pull", events, &initial, func(ctx context.Context) (streamFrame, error) {
		res, err := token.run(ctx)
		terminal := token.progressEvent("success", 100)
		frame := streamFrame{Type: "result", OpID: req.OpID, Engine: token.engine, Op: "pull", ID: token.id, Model: token.model, Stage: "success", DownloadState: terminal.DownloadState, Result: res}
		if err != nil {
			frame.Stage = "error"
			frame.Code = wireCodes[rpcCode(err)]
			msg := s.exec.reportPullFailed(token.engine, token.model, err)
			return frame, fmt.Errorf("%s", msg)
		}
		return frame, nil
	})
}

// streamOp is the shared engine of the streaming ec endpoints. It subscribes to
// the engine's progress before launching run (so no early frame is missed),
// forwards each progress event as an NDJSON frame, and writes run's terminal
// frame (or an error frame) last. run executes on the request context, so a
// disconnected initiator cancels the operation.
func (s *controlServer) streamOp(w http.ResponseWriter, r *http.Request, opID, engine, op string, run func(ctx context.Context) (streamFrame, error)) {
	events, cancelSub := s.exec.progress.subscribe(engine)
	defer cancelSub()
	s.streamEvents(w, r, opID, engine, op, events, nil, run)
}

func (s *controlServer) streamEvents(w http.ResponseWriter, r *http.Request, opID, engine, op string, events <-chan ProgressEvent, initial *ProgressEvent, run func(context.Context) (streamFrame, error)) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	enc := json.NewEncoder(w)
	writeProgress := func(ev ProgressEvent) error {
		if ev.Op != op {
			return nil
		}
		f := streamFrame{
			Type: "progress", OpID: opID, Engine: ev.Engine, Op: ev.Op,
			Stage: ev.Stage, Message: ev.Message,
			ID: ev.ID, Model: ev.Model, DownloadState: ev.DownloadState,
		}
		if wirePercentIncluded(ev.Percent) {
			f.Percent = ev.Percent
		}
		if err := enc.Encode(f); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}
	// A reservation publishes queued before subscribing. Send its retained
	// snapshot explicitly, and do not launch work if that initial write fails.
	if initial != nil {
		if r.Context().Err() != nil || writeProgress(*initial) != nil {
			return
		}
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	type opResult struct {
		frame streamFrame
		err   error
	}
	resCh := make(chan opResult, 1)
	go func() {
		f, err := run(ctx)
		resCh <- opResult{frame: f, err: err}
	}()

	for {
		select {
		case ev, ok := <-events:
			if !ok || writeProgress(ev) != nil {
				return
			}
		case res := <-resCh:
			// Drain any progress buffered before the terminal frame.
			for drained := false; !drained; {
				select {
				case ev, ok := <-events:
					if !ok || writeProgress(ev) != nil {
						return
					}
				default:
					drained = true
				}
			}
			frame := res.frame
			if res.err != nil {
				frame.Type, frame.OpID, frame.Engine, frame.Op = "error", opID, engine, op
				frame.Message, frame.Result = res.err.Error(), nil
				var veto *localVetoError
				if errors.As(res.err, &veto) {
					frame.Code = wireCodes[veto.code]
				}
			}
			_ = enc.Encode(frame)
			flusher.Flush()
			return
		case <-r.Context().Done():
			return
		}
	}
}
