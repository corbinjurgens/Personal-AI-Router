// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

// postControl drives one ec handler as a paired node's request does.
func postControl(t *testing.T, handler func(http.ResponseWriter, *http.Request), path, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
	return rec
}

// requireLocalVeto fails unless rec is the typed 409 of a local veto.
func requireLocalVeto(t *testing.T, what string, rec *httptest.ResponseRecorder) {
	t.Helper()
	var body clustertrust.CodedRefusal
	if rec.Code != http.StatusConflict || json.Unmarshal(rec.Body.Bytes(), &body) != nil || body.Code != "local_veto" || body.Error == "" {
		t.Fatalf("%s = %d %q, want a 409 local_veto refusal", what, rec.Code, rec.Body.String())
	}
}

// vetoTestServer is a control server on a fake-engine executor with its own
// desired-state directory.
func vetoTestServer(t *testing.T, manifest *Manifest) (*controlServer, *Executor) {
	t.Helper()
	exec := newTestExecutor(t, manifest)
	t.Cleanup(exec.StopAll)
	return &controlServer{exec: exec}, exec
}

func TestRemoteLoadFailsClosedBeforeTheFirstProjection(t *testing.T) {
	s, exec := vetoTestServer(t, testEngineManifest(fakeEngineBin))
	for _, c := range []struct {
		what    string
		handler func(http.ResponseWriter, *http.Request)
		path    string
		body    string
	}{
		{"remote load", s.handleLoad, controlLoadPath, `{"engine":"fake","model":"tiny"}`},
	} {
		if rec := postControl(t, c.handler, c.path, c.body); rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("%s before the first projection = %d %q, want 503", c.what, rec.Code, rec.Body.String())
		}
	}
	if st, err := exec.Status("fake"); err != nil || st.Running {
		t.Fatalf("a refused remote start ran the engine: %+v %v", st, err)
	}
}

func TestRemoteLoadIsVetoedWhileLocallyPaused(t *testing.T) {
	s, exec := vetoTestServer(t, testEngineManifest(fakeEngineBin))
	exec.nodeProjection.set(nodeProjection{PausedLocally: true, ManagementBlocked: true})
	requireLocalVeto(t, "remote load", postControl(t, s.handleLoad, controlLoadPath, `{"engine":"fake","model":"tiny"}`))
	if st, err := exec.Status("fake"); err != nil || st.Running {
		t.Fatalf("a vetoed remote start ran the engine: %+v %v", st, err)
	}
	if rec := postControl(t, s.handleUnload, controlUnloadPath, `{"engine":"fake","model":"tiny"}`); rec.Code == http.StatusConflict {
		t.Fatalf("remote unload while paused was vetoed: %q", rec.Body.String())
	}
	if err := exec.Start(context.Background(), "fake"); rpcCode(err) != codeLocalVeto {
		t.Fatalf("local start while paused = %v, want -32011", err)
	}
}

func TestRemoteLoadIgnoresRetiredEngineIntent(t *testing.T) {
	for _, stored := range []struct{ name, document string }{
		{"local", `{"engines":{"fake":false},"offBy":{"fake":"local"}}`},
		{"peer", `{"engines":{"fake":false},"offBy":{"fake":"peer"}}`},
		{"legacy", `{"engines":{"fake":false}}`},
		{"missing", `{"engines":{}}`},
	} {
		t.Run(stored.name, func(t *testing.T) {
			s, exec := vetoTestServer(t, lockLoadManifest())
			exec.nodeProjection.set(nodeProjection{ManagementBlocked: false})
			exec.admittedLoadParent = exec.ModelLoad
			if err := os.WriteFile(exec.desired.path, []byte(stored.document), 0o600); err != nil {
				t.Fatal(err)
			}
			// An explicit load does not start the stopped engine. Its runtime
			// failure must remain an ordinary error, rather than an Off veto.
			load := postControl(t, s.handleLoad, controlLoadPath, `{"engine":"fake","model":"tiny"}`)
			if load.Code != http.StatusInternalServerError {
				t.Fatalf("stopped load = %d %s, want runtime failure", load.Code, load.Body.String())
			}
			requireInstallTerminal(t, postControl(t, s.handleInstall, controlInstallPath, `{"engine":"fake"}`), "result", "")
			if rec := postControl(t, s.handleLoad, controlLoadPath, `{"engine":"fake","model":"tiny"}`); rec.Code != http.StatusOK {
				t.Fatalf("peer load = %d %s", rec.Code, rec.Body.String())
			}
		})
	}
}

// slowStartManifest is a fake engine that never becomes ready, so a start
// waits in its readiness probe until it is canceled.
func slowStartManifest() *Manifest {
	m := testEngineManifest(fakeEngineBin)
	key := runtime.GOOS + "/" + runtime.GOARCH
	p := m.Platforms[key]
	p.Runtime.Args = []string{"noserve"}
	p.Runtime.Ready.TimeoutS = 60
	m.Platforms[key] = p
	return m
}

// installRemotely waits for installation's automatic start to own a process
// inside its readiness wait, then returns the channel its response arrives on.
func installRemotely(t *testing.T, s *controlServer) <-chan *httptest.ResponseRecorder {
	t.Helper()
	st, err := s.exec.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		rec := httptest.NewRecorder()
		s.handleInstall(rec, httptest.NewRequest(http.MethodPost, controlInstallPath, strings.NewReader(`{"engine":"fake"}`)))
		done <- rec
	}()
	awaitLockCondition(t, "remote installation's owned startup", func() bool {
		st.mu.Lock()
		defer st.mu.Unlock()
		return st.proc != nil && st.startCancel != nil
	})
	return done
}

func requireInstallTerminal(t *testing.T, rec *httptest.ResponseRecorder, wantType, wantCode string) {
	t.Helper()
	if rec.Code != http.StatusOK {
		t.Fatalf("install stream status = %d, body=%s", rec.Code, rec.Body.String())
	}
	dec := json.NewDecoder(rec.Body)
	var terminal streamFrame
	for {
		var frame streamFrame
		if err := dec.Decode(&frame); err == io.EOF {
			break
		} else if err != nil {
			t.Fatalf("decode install frame: %v", err)
		}
		if terminal.Type != "" {
			t.Fatal("install frame after terminal")
		}
		if frame.Type == "result" || frame.Type == "error" {
			terminal = frame
		}
	}
	if terminal.Type != wantType || terminal.Code != wantCode {
		t.Fatalf("install terminal = %+v, want %s/%s", terminal, wantType, wantCode)
	}
}

func awaitResponse(t *testing.T, done <-chan *httptest.ResponseRecorder) *httptest.ResponseRecorder {
	t.Helper()
	select {
	case rec := <-done:
		return rec
	case <-time.After(20 * time.Second):
		t.Fatal("the accepted remote start was not canceled")
		return nil
	}
}

func TestLocalVetoCancelsAnAcceptedRemoteInstallStart(t *testing.T) {
	t.Run("local pause", func(t *testing.T) {
		s, exec := vetoTestServer(t, slowStartManifest())
		m := &Manager{exec: exec}
		m.applyNodeProjection(&Message{Params: json.RawMessage(`{"policy":{"revision":"1","roles":[]},"pausedLocally":false,"managementBlocked":false}`)})
		done := installRemotely(t, s)
		m.applyNodeProjection(&Message{Params: json.RawMessage(`{"policy":{"revision":"2","roles":[],"pausedBy":"local"},"pausedLocally":true,"managementBlocked":true}`)})
		requireInstallTerminal(t, awaitResponse(t, done), "error", "local_veto")
	})
	t.Run("a peer pause cancels", func(t *testing.T) {
		_, exec := vetoTestServer(t, testEngineManifest(fakeEngineBin))
		m := &Manager{exec: exec}
		exec.nodeProjection.set(nodeProjection{ManagementBlocked: false})
		ctx, done, err := exec.beginRemote(context.Background(), "fake")
		if err != nil {
			t.Fatalf("begin remote work: %v", err)
		}
		defer done()
		m.applyNodeProjection(&Message{Params: json.RawMessage(`{"policy":{"revision":"2","roles":[],"pausedBy":"peer"},"pausedLocally":false,"managementBlocked":true}`)})
		if rpcCode(context.Cause(ctx)) != codeLocalVeto {
			t.Fatalf("peer pause cause = %v, want -32011", context.Cause(ctx))
		}
	})
}

func TestRemoteClientMapsTypedRefusalsToRPCCodes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case controlLoadPath:
			writeCodedError(w, codeLocalVeto, nodePausedVeto)
		case controlUnloadPath:
			http.Error(w, "plain conflict", http.StatusConflict)
		case controlInstallPath:
			w.Header().Set("Content-Type", "application/x-ndjson")
			_ = json.NewEncoder(w).Encode(streamFrame{Type: "error", Message: nodePausedVeto, Code: "local_veto"})
		}
	}))
	defer srv.Close()
	base := http.DefaultTransport.(*http.Transport)
	c := &remoteClient{controlHTTP: newRemoteHTTPClient(base, time.Second), controlReadyHTTP: newRemoteHTTPClient(base, time.Second), base: srv.URL}

	_, err := c.postJSON(context.Background(), controlLoadPath, "fake", modelActionRequest{Engine: "fake", Model: "tiny"})
	if rpcCode(err) != codeLocalVeto || !strings.HasSuffix(err.Error(), nodePausedVeto) {
		t.Fatalf("vetoed remote load = %v (code %d), want %d with the target's reason", err, rpcCode(err), codeLocalVeto)
	}
	_, err = c.postJSON(context.Background(), controlUnloadPath, "fake", modelActionRequest{Engine: "fake", Model: "tiny"})
	if err == nil || rpcCode(err) != -32000 {
		t.Fatalf("untyped 409 = %v (code %d), want an ordinary -32000 failure", err, rpcCode(err))
	}
	_, err = c.stream(context.Background(), controlInstallPath, installRequest{Engine: "fake"}, nil)
	if rpcCode(err) != codeLocalVeto || err.Error() != nodePausedVeto {
		t.Fatalf("vetoed streamed install = %v (code %d), want %d", err, rpcCode(err), codeLocalVeto)
	}
}
