// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

// This file covers the engine-manager half of the model-mutation handshake
// (model_mutation.go): the guard every content mutation takes, the correlated
// relay, and the reply's route through the read loop. No test starts an engine,
// a CLI, a listener or a sleep-synchronized watcher: an engine HTTP call is a
// RoundTripper or an httptest recorder, and every ordering proof is a channel.

// acceptModelMutations installs an explicit successful acknowledger, so an
// executor behaves as one attached to a broker that confirms the withdrawal.
// Fixtures use it wherever the mutation is incidental; a test that needs a
// missing parent builds an unwired executor, and one that needs a blocking or
// refusing parent replaces this seam.
func acceptModelMutations(ex *Executor) {
	ex.modelMutationParent = func(context.Context, string) error { return nil }
}

// No process is launched: only the owned-run fields and fake HTTP transport
// participate in inventory and identity production.
func mutationIdentityFixture(t *testing.T) (*Executor, *mutationTransport) {
	t.Helper()
	ex := newTestExecutor(t, digestManifest())
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.running, st.installed = true, true
	st.proc = &managedProc{}
	st.runID, st.launchDigest = "run-mutation", nameDigest("launch")
	transport := &mutationTransport{respond: func(*http.Request) (*http.Response, error) {
		return mutationJSON(http.StatusOK, `{"models":[{"name":"m","digest":"`+nameDigest("original")+`"}]}`)
	}}
	ex.client = &http.Client{Transport: transport}
	return ex, transport
}

func TestModelMutationFrozenSweepCannotRestoreIdentity(t *testing.T) {
	test := func(name string, mutationErr error, finalDigest string) {
		t.Run(name, func(t *testing.T) {
			ex, transport := mutationIdentityFixture(t)
			ctx := context.Background()
			ex.modelsSweep(ctx, true)
			var published []modelIdentityParams
			ex.emit = func(method string, params any) {
				if method == modelIdentityMethod {
					id, ok := params.(modelIdentityParams)
					if !ok {
						t.Errorf("identity params = %T", params)
						return
					}
					published = append(published, id)
				}
			}
			last := ex.publishModelIdentity(nil)
			started, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
			transport.respond = func(*http.Request) (*http.Response, error) {
				close(started)
				<-release
				return mutationJSON(http.StatusOK, `{"models":[{"name":"m","digest":"`+nameDigest("original")+`"}]}`)
			}
			go func() { ex.modelsSweep(ctx, true); close(done) }()
			<-started
			err := ex.withModelMutation(ctx, "fake", func() error { return mutationErr })
			if !errors.Is(err, mutationErr) {
				t.Fatalf("mutation error = %v, want %v", err, mutationErr)
			}
			close(release)
			<-done
			if got := ex.modelIdentity().Engines["fake"].ModelDigests; len(got) != 0 {
				t.Fatalf("frozen sweep restored stale identity: %v", got)
			}
			// Do not publish the intermediate empty snapshot: dedup must still
			// remember the boundary when the final content equals the original.
			transport.respond = func(*http.Request) (*http.Response, error) {
				return mutationJSON(http.StatusOK, `{"models":[{"name":"m","digest":"`+finalDigest+`"}]}`)
			}
			ex.modelsSweep(ctx, true)
			ex.publishModelIdentity(last)
			if len(published) != 2 || published[1].Engines["fake"].ModelDigests["m"] != finalDigest {
				t.Fatalf("fresh identity not republished: %+v", published)
			}
		})
	}
	test("successful replacement", nil, nameDigest("replacement"))
	test("partial failure", errors.New("partial mutation"), nameDigest("partial"))
	test("unchanged final digest", nil, nameDigest("original"))
}

func TestModelMutationSuppressionRejectsConcurrentSnapshots(t *testing.T) {
	ex, transport := mutationIdentityFixture(t)
	ctx := context.Background()
	ex.modelsSweep(ctx, true)
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	ex.modelMutationParent = func(context.Context, string) error {
		close(entered)
		<-release
		return nil
	}
	go func() { done <- ex.withModelMutation(ctx, "fake", func() error { return nil }) }()
	<-entered
	ex.emit = func(method string, params any) {
		if method == modelIdentityMethod {
			id, ok := params.(modelIdentityParams)
			if !ok {
				t.Errorf("identity params = %T", params)
				return
			}
			if _, ok := id.Engines["fake"]; ok {
				t.Error("suppressed engine was published")
			}
		}
	}
	// An observation started during suppression must remain unusable after end.
	started, finish, swept := make(chan struct{}), make(chan struct{}), make(chan struct{})
	transport.respond = func(*http.Request) (*http.Response, error) {
		close(started)
		<-finish
		return mutationJSON(http.StatusOK, `{"models":[{"name":"m","digest":"`+nameDigest("during")+`"}]}`)
	}
	go func() { ex.modelsSweep(ctx, true); close(swept) }()
	<-started
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() { defer wg.Done(); ex.publishModelIdentity(nil) }()
	}
	wg.Wait()
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	close(finish)
	<-swept
	if got := ex.modelIdentity().Engines["fake"].ModelDigests; len(got) != 0 {
		t.Fatalf("during-mutation observation restored identity: %v", got)
	}
}

// mutationManifest is one engine whose delete operation is bound to a custom
// action name ("purge"), so a test also proves the guard follows the semantic
// binding rather than an action's spelling. list_models stays a read-only
// action, and the standard pull_model action carries the pull.
func mutationManifest(t *testing.T, engine string) *Manifest {
	t.Helper()
	m := testEngineManifest(fakeEngineBin)
	m.Engine = engine
	m.DisplayName = engine
	m.Actions = map[string]Action{
		"list_models":   {HTTP: &ActionHTTP{Method: "GET", Path: "/api/tags"}},
		"purge":         {HTTP: &ActionHTTP{Method: "DELETE", Path: "/api/delete"}},
		pullModelAction: {HTTP: &ActionHTTP{Method: "POST", Path: "/api/pull"}},
	}
	m.Operations = map[string]OperationBinding{
		engines.OperationDelete: {Action: "purge"},
	}
	if err := m.Validate(); err != nil {
		t.Fatalf("mutation manifest: %v", err)
	}
	return m
}

// mutationTransport is the engine's loopback API as a fake: it records every
// call and answers from respond, so no listener and no engine are involved.
type mutationTransport struct {
	mu       sync.Mutex
	requests []string
	respond  func(*http.Request) (*http.Response, error)
}

func (t *mutationTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	t.mu.Lock()
	t.requests = append(t.requests, r.Method+" "+r.URL.Path)
	t.mu.Unlock()
	return t.respond(r)
}

func (t *mutationTransport) seen() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.requests...)
}

func mutationJSON(status int, body string) (*http.Response, error) {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}, nil
}

// mutationEngineFixture is one engine whose delete and pull actions answer
// through the returned fake transport.
func mutationEngineFixture(t *testing.T, engine string) (*Executor, *mutationTransport) {
	t.Helper()
	transport := &mutationTransport{respond: func(r *http.Request) (*http.Response, error) {
		if r.URL.Path == "/api/pull" {
			return mutationJSON(http.StatusOK, `{"status":"success"}`+"\n")
		}
		return mutationJSON(http.StatusOK, `{"ok":true}`)
	}}
	ex := newTestExecutor(t, mutationManifest(t, engine))
	ex.client = &http.Client{Transport: transport}
	st, err := ex.state(engine)
	if err != nil {
		t.Fatalf("resolve engine state: %v", err)
	}
	st.running = true
	return ex, transport
}

// mutationBroker is the broker end of the handshake as a test sees it: it
// records every request the manager sends and answers them only when the test
// says so, through the manager's own notification path.
type mutationBroker struct {
	mu       sync.Mutex
	seen     []nodepolicy.ModelMutationRequest
	sent     chan nodepolicy.ModelMutationRequest
	failSend error
}

func newMutationBroker() *mutationBroker {
	return &mutationBroker{sent: make(chan nodepolicy.ModelMutationRequest, 16)}
}

func (b *mutationBroker) send(method string, params any) error {
	if method != nodepolicy.MethodModelMutationRequest {
		return fmt.Errorf("unexpected relay notification %q", method)
	}
	if b.failSend != nil {
		return b.failSend
	}
	req, ok := params.(nodepolicy.ModelMutationRequest)
	if !ok {
		return fmt.Errorf("relay params are %T, want a model mutation request", params)
	}
	b.mu.Lock()
	b.seen = append(b.seen, req)
	b.mu.Unlock()
	b.sent <- req
	return nil
}

func (b *mutationBroker) requests() []nodepolicy.ModelMutationRequest {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]nodepolicy.ModelMutationRequest(nil), b.seen...)
}

// awaitRequest returns the next request the manager sent, failing rather than
// hanging the suite if the manager never asked.
func (b *mutationBroker) awaitRequest(t *testing.T) nodepolicy.ModelMutationRequest {
	t.Helper()
	select {
	case req := <-b.sent:
		return req
	case <-time.After(5 * time.Second):
		t.Fatal("the engine manager never asked the broker to withdraw an engine")
		return nodepolicy.ModelMutationRequest{}
	}
}

// replyTo delivers one reply through the manager's read-loop routing, the way
// the broker's notification arrives in production.
func (b *mutationBroker) replyTo(t *testing.T, m *Manager, req nodepolicy.ModelMutationRequest, ok bool, failure string, code int) {
	t.Helper()
	reply := nodepolicy.ModelMutationReply{RequestID: req.RequestID, Engine: req.Engine, OK: ok, Error: failure, Code: code}
	raw, err := json.Marshal(reply)
	if err != nil {
		t.Fatalf("encode reply: %v", err)
	}
	m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", Method: nodepolicy.MethodModelMutationReply, Params: raw})
}

// managerFrames decodes every frame a manager has written.
func managerFrames(t *testing.T, s string) []Message {
	t.Helper()
	var out []Message
	sc := bufio.NewScanner(strings.NewReader(s))
	for sc.Scan() {
		var m Message
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("decode manager frame: %v", err)
		}
		out = append(out, m)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan manager frames: %v", err)
	}
	return out
}

// replyMessage is the read-loop notification a broker reply arrives as.
func replyMessage(t *testing.T, reply nodepolicy.ModelMutationReply) *Message {
	t.Helper()
	raw, err := json.Marshal(reply)
	if err != nil {
		t.Fatalf("encode reply: %v", err)
	}
	return &Message{JSONRPC: "2.0", Method: nodepolicy.MethodModelMutationReply, Params: raw}
}

func (b *mutationBroker) pendingWait(m *Manager) int {
	m.modelMutation.mu.Lock()
	defer m.modelMutation.mu.Unlock()
	return len(m.modelMutation.pending)
}

// ---- the guard on the mutation entry paths ----

// A typed delete RPC whose binding names a custom action must not touch the
// engine until the broker confirms the withdrawal, and must touch it once it
// has. This is also the first ordering regression: without the pre-mutation
// guard in Action the engine call happens while the confirmation is pending.
func TestModelMutationGuardBlocksTypedDeleteUntilConfirmed(t *testing.T) {
	ex, engine := mutationEngineFixture(t, "fake")
	var out lockedBuffer
	m := NewManager(NewCodec(&out), ex, nil, "")
	broker := newMutationBroker()
	m.modelMutation.send = broker.send

	id := json.RawMessage("11")
	done := make(chan struct{})
	go func() {
		defer close(done)
		m.runModelOperation(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: deleteModelMethod,
			Params: json.RawMessage(`{"engine":"fake","model":"m:1"}`)})
	}()

	req := broker.awaitRequest(t)
	if req.Engine != "fake" || req.RequestID == "" {
		t.Fatalf("broker request = %+v, want the engine and a correlation id", req)
	}
	if got := engine.seen(); len(got) != 0 {
		t.Fatalf("engine saw %v before the withdrawal was confirmed", got)
	}

	broker.replyTo(t, m, req, true, "", 0)
	<-done

	frames := managerFrames(t, out.String())
	if len(frames) != 1 || frames[0].Error != nil {
		t.Fatalf("typed delete response = %s, want the engine's result", out.String())
	}
	if got := engine.seen(); len(got) != 1 || got[0] != "DELETE /api/delete" {
		t.Fatalf("engine saw %v, want one DELETE of the bound action", got)
	}
}

// The same guard applies to a generic engine:action call, because the manifest
// binds the delete operation to "purge": the action's name is not the reason.
func TestModelMutationGuardBlocksGenericActionUntilConfirmed(t *testing.T) {
	ex, engine := mutationEngineFixture(t, "fake")
	var out lockedBuffer
	m := NewManager(NewCodec(&out), ex, nil, "")
	broker := newMutationBroker()
	m.modelMutation.send = broker.send

	id := json.RawMessage("12")
	done := make(chan struct{})
	go func() {
		defer close(done)
		m.runAction(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:action",
			Params: json.RawMessage(`{"engine":"fake","action":"purge","params":{"model":"m:1"}}`)})
	}()

	req := broker.awaitRequest(t)
	if got := engine.seen(); len(got) != 0 {
		t.Fatalf("engine saw %v before the withdrawal was confirmed", got)
	}
	broker.replyTo(t, m, req, true, "", 0)
	<-done

	if got := engine.seen(); len(got) != 1 || got[0] != "DELETE /api/delete" {
		t.Fatalf("engine saw %v, want one DELETE of the bound action", got)
	}
}

// The streaming pull owns its own dispatch, so it takes the guard itself.
func TestModelMutationGuardBlocksStreamingPullUntilConfirmed(t *testing.T) {
	ex, engine := mutationEngineFixture(t, "fake")
	var out lockedBuffer
	m := NewManager(NewCodec(&out), ex, nil, "")
	broker := newMutationBroker()
	m.modelMutation.send = broker.send

	id := json.RawMessage("13")
	done := make(chan struct{})
	go func() {
		defer close(done)
		m.runAction(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:action",
			Params: json.RawMessage(`{"engine":"fake","action":"pull_model","params":{"model":"m:1"}}`)})
	}()

	req := broker.awaitRequest(t)
	if got := engine.seen(); len(got) != 0 {
		t.Fatalf("engine saw %v before the withdrawal was confirmed", got)
	}
	broker.replyTo(t, m, req, true, "", 0)
	<-done

	if got := engine.seen(); len(got) != 1 || got[0] != "POST /api/pull" {
		t.Fatalf("engine saw %v, want one POST of the pull action", got)
	}
	if !strings.Contains(out.String(), "success") {
		t.Fatalf("pull response = %s, want the action's terminal result", out.String())
	}
}

// The direct ec endpoints are not a bypass: they enter the same owners, so they
// inherit the guard without a second barrier of their own.
func TestModelMutationGuardBlocksDirectECEndpointsUntilConfirmed(t *testing.T) {
	cases := []struct {
		name string
		path string
		body string
		want string
	}{
		{"delete", controlDeletePath, `{"engine":"fake","model":"m:1"}`, "DELETE /api/delete"},
		{"pull", controlPullPath, `{"engine":"fake","model":"m:1"}`, "POST /api/pull"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ex, engine := mutationEngineFixture(t, "fake")
			var out lockedBuffer
			m := NewManager(NewCodec(&out), ex, nil, "")
			broker := newMutationBroker()
			m.modelMutation.send = broker.send

			server := &controlServer{exec: ex}
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			done := make(chan struct{})
			go func() {
				defer close(done)
				if tc.name == "delete" {
					server.handleDelete(rec, req)
					return
				}
				server.handlePull(rec, req)
			}()

			reply := broker.awaitRequest(t)
			if got := engine.seen(); len(got) != 0 {
				t.Fatalf("engine saw %v before the withdrawal was confirmed", got)
			}
			broker.replyTo(t, m, reply, true, "", 0)
			<-done

			if got := engine.seen(); len(got) != 1 || got[0] != tc.want {
				t.Fatalf("engine saw %v, want one %s", got, tc.want)
			}
			if rec.Code != http.StatusOK {
				t.Fatalf("%s status = %d, want 200", tc.name, rec.Code)
			}
		})
	}
}

// A destination copy lists and plans before the guard, and writes nothing until
// the withdrawal is confirmed.
func TestModelMutationGuardBlocksCopyUntilConfirmed(t *testing.T) {
	plentyOfDisk(t)
	srcDir, dstDir := t.TempDir(), t.TempDir()
	srcExec := copyTestExecutor(t, "ollama", srcDir, true)
	writeOllamaModel(t, filepath.Join(srcDir, "models"), "tiny:1b", 4096, 4)
	source := sourceFromExecutor(t, srcExec, "ollama", "tiny:1b")
	dst := copyTestExecutor(t, "ollama", dstDir, true)

	var out lockedBuffer
	m := NewManager(NewCodec(&out), dst, nil, "")
	broker := newMutationBroker()
	m.modelMutation.send = broker.send

	type result struct {
		res copyModelResult
		err error
	}
	results := make(chan result, 1)
	go func() {
		res, err := dst.CopyModelFrom(context.Background(), source, "ollama", "tiny:1b", nil)
		results <- result{res, err}
	}()

	req := broker.awaitRequest(t)
	if fetched := source.fetchedPaths(); len(fetched) != 0 {
		t.Fatalf("the copy fetched %v before the withdrawal was confirmed", fetched)
	}
	root := filepath.Join(dstDir, "models")
	if _, err := os.Stat(root); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("the copy wrote %s before the withdrawal was confirmed (stat err %v)", root, err)
	}

	broker.replyTo(t, m, req, true, "", 0)
	got := <-results
	if got.err != nil {
		t.Fatalf("copy: %v", got.err)
	}
	if got.res.Files != 3 {
		t.Fatalf("copy result = %+v, want 3 files", got.res)
	}
	if fetched := source.fetchedPaths(); len(fetched) == 0 {
		t.Fatal("the copy fetched nothing after the withdrawal was confirmed")
	}
	entries, err := os.ReadDir(root)
	if err != nil || len(entries) == 0 {
		t.Fatalf("destination %s holds %d entries, %v", root, len(entries), err)
	}
}

// The guard follows the semantic binding, not a name: an action the manifest
// binds to no model operation runs on an executor with no broker at all, and so
// do load and unload, which never change installed content.
func TestModelMutationReadOnlyAndLoadPathsTakeNoGuard(t *testing.T) {
	reg := NewRegistry()
	manifest := mutationManifest(t, "fake")
	manifest.Actions["warm"] = Action{HTTP: &ActionHTTP{Method: "POST", Path: "/v1/load"}}
	manifest.Actions["cool"] = Action{HTTP: &ActionHTTP{Method: "POST", Path: "/v1/unload"}}
	manifest.Operations = map[string]OperationBinding{
		engines.OperationDelete: {Action: "purge"},
		engines.OperationLoad:   {Action: "warm"},
		engines.OperationUnload: {Action: "cool"},
	}
	if err := manifest.Validate(); err != nil {
		t.Fatalf("manifest: %v", err)
	}
	reg.engines["fake"] = manifest
	// Deliberately unwired: nothing here may need a broker.
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
	transport := &mutationTransport{respond: func(*http.Request) (*http.Response, error) {
		return mutationJSON(http.StatusOK, `{"ok":true}`)
	}}
	ex.client = &http.Client{Transport: transport}
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.running = true

	if _, err := ex.Action(context.Background(), "fake", "list_models", nil); err != nil {
		t.Fatalf("a read-only action needed a broker: %v", err)
	}
	if _, err := ex.ModelLoad(context.Background(), "fake", "m:1"); err != nil {
		t.Fatalf("a load needed a broker: %v", err)
	}
	if _, err := ex.ModelUnload(context.Background(), "fake", "m:1"); err != nil {
		t.Fatalf("an unload needed a broker: %v", err)
	}
	if got := transport.seen(); len(got) != 3 {
		t.Fatalf("engine saw %v, want the three unguarded calls", got)
	}
}

// ---- failures authorize nothing ----

// Every failure path leaves the engine untouched and no pending entry behind.
func TestModelMutationFailuresAuthorizeNothing(t *testing.T) {
	cases := []struct {
		name string
		// wire prepares the manager and returns the context for one delete.
		wire    func(t *testing.T, m *Manager, broker *mutationBroker) context.Context
		want    string
		wantErr error
	}{
		{
			name: "refusal",
			wire: func(t *testing.T, m *Manager, broker *mutationBroker) context.Context {
				go func() {
					req := <-broker.sent
					broker.replyTo(t, m, req, false, "the destination is busy", nodepolicy.ModelMutationWithdrawalFailed)
				}()
				return context.Background()
			},
			want: "the destination did not confirm the withdrawal: the destination is busy",
		},
		{
			name: "timeout",
			wire: func(t *testing.T, m *Manager, broker *mutationBroker) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			want:    "the withdrawal was not confirmed before the deadline",
			wantErr: context.DeadlineExceeded,
		},
		{
			name: "cancellation",
			wire: func(t *testing.T, m *Manager, broker *mutationBroker) context.Context {
				ctx, cancel := context.WithCancel(context.Background())
				t.Cleanup(cancel)
				go func() {
					<-broker.sent
					cancel()
				}()
				return ctx
			},
			want:    "the withdrawal confirmation was cancelled",
			wantErr: context.Canceled,
		},
		{
			name: "send error",
			wire: func(t *testing.T, m *Manager, broker *mutationBroker) context.Context {
				broker.failSend = errors.New("the parent codec is closed")
				return context.Background()
			},
			want: "the withdrawal request could not be sent: the parent codec is closed",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ex, engine := mutationEngineFixture(t, "fake")
			var out lockedBuffer
			m := NewManager(NewCodec(&out), ex, nil, "")
			broker := newMutationBroker()
			m.modelMutation.send = broker.send
			ctx := tc.wire(t, m, broker)

			_, err := ex.ModelDelete(ctx, "fake", "m:1")
			if err == nil {
				t.Fatalf("%s: the delete was authorized", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("%s error = %v, want %q", tc.name, err, tc.want)
			}
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Fatalf("%s error = %v, want it to wrap %v", tc.name, err, tc.wantErr)
			}
			if got := engine.seen(); len(got) != 0 {
				t.Fatalf("%s: engine saw %v", tc.name, got)
			}
			if n := broker.pendingWait(m); n != 0 {
				t.Fatalf("%s: %d pending requests survived the failure", tc.name, n)
			}
		})
	}
}

// A refused confirmation must not reach process launch for a CLI action. The
// diagnostic must be the withdrawal refusal, before command resolution or launch.
func TestModelMutationCLIRefusalNeverLaunchesTheCommand(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "must-not-run")
	manifest := mutationManifest(t, "fake")
	manifest.Actions["purge"] = Action{Cmd: []string{marker, "--model", "{model}"}}
	if err := manifest.Validate(); err != nil {
		t.Fatalf("manifest: %v", err)
	}
	ex := newTestExecutor(t, manifest)

	var out lockedBuffer
	m := NewManager(NewCodec(&out), ex, nil, "")
	broker := newMutationBroker()
	m.modelMutation.send = broker.send

	go func() {
		req := <-broker.sent
		broker.replyTo(t, m, req, false, "the destination is busy", nodepolicy.ModelMutationWithdrawalFailed)
	}()
	_, err := ex.ModelDelete(context.Background(), "fake", "m:1")
	if err == nil {
		t.Fatal("a refused confirmation authorized a CLI delete")
	}
	if !strings.Contains(err.Error(), "the destination did not confirm the withdrawal") {
		t.Fatalf("refused CLI delete error = %v, want the confirmation failure", err)
	}
	if strings.Contains(err.Error(), marker) {
		t.Fatalf("refused CLI delete reached process launch: %v", err)
	}

	assertNoPending(t, &m.modelMutation)
}

func TestModelMutationAbsentParentAuthorizesNothing(t *testing.T) {
	reg := NewRegistry()
	reg.engines["fake"] = mutationManifest(t, "fake")
	ex := NewExecutor(reg, NewReporter(nil), nil, t.TempDir())
	transport := &mutationTransport{respond: func(*http.Request) (*http.Response, error) {
		return mutationJSON(http.StatusOK, `{"ok":true}`)
	}}
	ex.client = &http.Client{Transport: transport}
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.running = true
	if _, err := ex.ModelDelete(context.Background(), "fake", "m:1"); err == nil || !strings.Contains(err.Error(), "not attached to a broker") {
		t.Fatalf("unwired delete error = %v", err)
	}
	if got := transport.seen(); len(got) != 0 {
		t.Fatalf("unwired delete reached the engine: %v", got)
	}
	var relay modelMutationRelay
	assertNoPending(t, &relay)
}

// ---- serialization ----

// Same-engine mutations serialize on the content mutex while a different
// engine's mutation proceeds independently. The acknowledger is the barrier:
// a call reports its engine on entered and waits for one outcome.
func TestModelMutationSameEngineSerializesAndOtherEnginesProceed(t *testing.T) {
	reg := NewRegistry()
	for _, name := range []string{"first", "second"} {
		reg.engines[name] = mutationManifest(t, name)
	}
	ex := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
	transport := &mutationTransport{respond: func(*http.Request) (*http.Response, error) {
		return mutationJSON(http.StatusOK, `{"ok":true}`)
	}}
	ex.client = &http.Client{Transport: transport}
	for _, name := range []string{"first", "second"} {
		st, err := ex.state(name)
		if err != nil {
			t.Fatal(err)
		}
		st.running = true
	}
	entered := make(chan string, 8)
	firstRelease := make(chan error)
	secondRelease := make(chan error)
	ex.modelMutationParent = func(ctx context.Context, engine string) error {
		entered <- engine
		release := firstRelease
		if engine == "second" {
			release = secondRelease
		}
		select {
		case err := <-release:
			return err
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	delete := func(engine string) <-chan error {
		done := make(chan error, 1)
		go func() {
			_, err := ex.ModelDelete(context.Background(), engine, "m:1")
			done <- err
		}()
		return done
	}

	firstA := delete("first")
	if got := <-entered; got != "first" {
		t.Fatalf("first acknowledger entry = %q", got)
	}
	// A different engine must not wait behind the held guard.
	second := delete("second")
	if got := <-entered; got != "second" {
		t.Fatalf("second-engine acknowledger entry = %q, want it to proceed while first is held", got)
	}
	// A second mutation of the held engine must not reach its acknowledger.
	firstB := delete("first")
	st, err := ex.state("first")
	if err != nil {
		t.Fatal(err)
	}
	if st.contentMu.TryLock() {
		st.contentMu.Unlock()
		t.Fatal("first mutation did not hold the content mutex across confirmation")
	}

	firstRelease <- nil
	if err := <-firstA; err != nil {
		t.Fatalf("first mutation: %v", err)
	}
	select {
	case got := <-entered:
		if got != "first" {
			t.Fatalf("serialized second mutation reported engine %q", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the serialized second mutation never entered the handshake")
	}
	firstRelease <- nil
	if err := <-firstB; err != nil {
		t.Fatalf("serialized mutation: %v", err)
	}
	secondRelease <- nil
	if err := <-second; err != nil {
		t.Fatalf("second-engine mutation: %v", err)
	}
	if got := transport.seen(); len(got) != 3 {
		t.Fatalf("engine saw %v, want one delete per mutation", got)
	}
}

// ---- the correlated relay ----

func TestModelMutationRelayAuthorizesOnlyAnAcceptedReply(t *testing.T) {
	t.Run("accepted", func(t *testing.T) {
		relay := &modelMutationRelay{}
		sent := make(chan nodepolicy.ModelMutationRequest, 1)
		relay.send = func(_ string, params any) error {
			sent <- params.(nodepolicy.ModelMutationRequest)
			return nil
		}
		go func() {
			req := <-sent
			raw := replyMessage(t, nodepolicy.ModelMutationReply{RequestID: req.RequestID, Engine: req.Engine, OK: true}).Params
			relay.reply(raw)
		}()
		if err := relay.call(context.Background(), "fake"); err != nil {
			t.Fatalf("accepted reply = %v", err)
		}
		assertNoPending(t, relay)
	})

	t.Run("an unknown request id is not this waiter's reply", func(t *testing.T) {
		relay := &modelMutationRelay{}
		sent := make(chan nodepolicy.ModelMutationRequest, 1)
		relay.send = func(_ string, params any) error {
			stray := replyMessage(t, nodepolicy.ModelMutationReply{RequestID: "someone-elses", Engine: "fake", OK: true}).Params
			relay.reply(stray)
			sent <- params.(nodepolicy.ModelMutationRequest)
			return nil
		}
		// The waiter is still pending: only its own id can release it.
		go func() {
			req := <-sent
			raw := replyMessage(t, nodepolicy.ModelMutationReply{RequestID: req.RequestID, Engine: "fake", OK: true}).Params
			relay.reply(raw)
		}()
		if err := relay.call(context.Background(), "fake"); err != nil {
			t.Fatalf("call = %v", err)
		}
		assertNoPending(t, relay)
	})

	t.Run("a reply for another engine does not authorize", func(t *testing.T) {
		relay := &modelMutationRelay{}
		relay.send = func(_ string, params any) error {
			req := params.(nodepolicy.ModelMutationRequest)
			raw := replyMessage(t, nodepolicy.ModelMutationReply{RequestID: req.RequestID, Engine: "other", OK: true}).Params
			relay.reply(raw)
			return nil
		}
		err := relay.call(context.Background(), "fake")
		if err == nil || !strings.Contains(err.Error(), `confirmed a withdrawal for engine "other"`) {
			t.Fatalf("call error = %v, want the mismatched engine named", err)
		}
		assertNoPending(t, relay)
	})

	t.Run("a malformed reply authorizes nothing", func(t *testing.T) {
		relay := &modelMutationRelay{}
		relay.send = func(_ string, _ any) error {
			relay.reply(json.RawMessage(`{"requestId":7,"engine":"fake","ok":true}`))
			relay.reply(json.RawMessage(`{"requestId":"x","engine":"fake","ok":true,"extra":1}`))
			return nil
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
		defer cancel()
		err := relay.call(ctx, "fake")
		if err == nil || !strings.Contains(err.Error(), "not confirmed before the deadline") {
			t.Fatalf("call error = %v, want the deadline", err)
		}
		assertNoPending(t, relay)
	})

	t.Run("a reply after the waiter left is dropped", func(t *testing.T) {
		relay := &modelMutationRelay{}
		var req nodepolicy.ModelMutationRequest
		relay.send = func(_ string, params any) error {
			req = params.(nodepolicy.ModelMutationRequest)
			return nil
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if err := relay.call(ctx, "fake"); err == nil {
			t.Fatal("a cancelled call succeeded")
		}
		assertNoPending(t, relay)
		// A late reply, and a duplicate for the same id, must not block or panic.
		raw := replyMessage(t, nodepolicy.ModelMutationReply{RequestID: req.RequestID, Engine: "fake", OK: true}).Params
		relay.reply(raw)
		relay.reply(raw)
		assertNoPending(t, relay)
	})

	t.Run("the pending bound refuses more work", func(t *testing.T) {
		relay := &modelMutationRelay{pending: make(map[string]chan nodepolicy.ModelMutationReply, maxPendingModelMutations)}
		for i := 0; i < maxPendingModelMutations; i++ {
			relay.pending[fmt.Sprintf("m%d", i)] = make(chan nodepolicy.ModelMutationReply, 1)
		}
		relay.send = func(string, any) error { return nil }
		err := relay.call(context.Background(), "fake")
		if err == nil || !strings.Contains(err.Error(), "too many model changes") {
			t.Fatalf("call error = %v, want the pending bound", err)
		}
		if n := len(relay.pending); n != maxPendingModelMutations {
			t.Fatalf("pending entries = %d, want the bound to survive", n)
		}
	})

	t.Run("no attached broker fails closed", func(t *testing.T) {
		var relay modelMutationRelay
		err := relay.call(context.Background(), "fake")
		if err == nil || !strings.Contains(err.Error(), "not attached to a broker") {
			t.Fatalf("call error = %v, want a fail-closed relay", err)
		}
		assertNoPending(t, &relay)
	})
}

// The relay's reply is routed by the manager's read loop, and only as an
// internal notification: a public request with the same method cannot complete
// a pending wait.
func TestModelMutationReplyRouteIsNotificationOnly(t *testing.T) {
	ex, _ := mutationEngineFixture(t, "fake")
	var out lockedBuffer
	m := NewManager(NewCodec(&out), ex, nil, "")
	broker := newMutationBroker()
	m.modelMutation.send = broker.send

	done := make(chan error, 1)
	go func() {
		_, err := ex.ModelDelete(context.Background(), "fake", "m:1")
		done <- err
	}()
	req := broker.awaitRequest(t)

	// A client request with the reply's method name is answered as an unknown
	// request and leaves the waiter pending.
	id := json.RawMessage("21")
	raw, err := json.Marshal(nodepolicy.ModelMutationReply{RequestID: req.RequestID, Engine: "fake", OK: true})
	if err != nil {
		t.Fatal(err)
	}
	m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: nodepolicy.MethodModelMutationReply, Params: raw})
	frames := managerFrames(t, out.String())
	if len(frames) != 1 || frames[0].Error == nil || frames[0].Error.Code != -32601 {
		t.Fatalf("client request response = %s, want method not found", out.String())
	}
	if n := broker.pendingWait(m); n != 1 {
		t.Fatalf("pending entries after a client request = %d, want the waiter untouched", n)
	}

	// The notification form completes it.
	m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", Method: nodepolicy.MethodModelMutationReply, Params: raw})
	if err := <-done; err != nil {
		t.Fatalf("delete after the notification reply: %v", err)
	}
}

func assertNoPending(t *testing.T, relay *modelMutationRelay) {
	t.Helper()
	relay.mu.Lock()
	defer relay.mu.Unlock()
	if len(relay.pending) != 0 {
		t.Fatalf("%d pending model mutation entries survived", len(relay.pending))
	}
}

func TestModelMutationEndsOnEveryExit(t *testing.T) {
	for _, outcome := range []string{"refused", "mutation failed", "success"} {
		t.Run(outcome, func(t *testing.T) {
			ex, _ := mutationEngineFixture(t, "fake")
			failure := errors.New(outcome)
			ex.modelMutationParent = func(context.Context, string) error {
				if outcome == "refused" {
					return failure
				}
				return nil
			}
			ran := false
			err := ex.withModelMutation(context.Background(), "fake", func() error {
				ran = true
				if outcome == "mutation failed" {
					return failure
				}
				return nil
			})
			if outcome == "success" && err != nil || outcome != "success" && !errors.Is(err, failure) {
				t.Fatalf("mutation error = %v for %s", err, outcome)
			}
			if ran != (outcome != "refused") {
				t.Fatalf("mutation ran = %v for %s", ran, outcome)
			}
			select {
			case <-ex.loadedPoke:
			default:
				t.Fatal("endModelMutation did not request an inventory sweep")
			}
		})
	}
}
