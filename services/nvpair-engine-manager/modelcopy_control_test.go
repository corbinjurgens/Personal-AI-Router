// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

func TestModelCopyControlStrictIngress(t *testing.T) {
	valid := `{"opId":"0123456789abcdef0123456789abcdef","sourceNodeId":"source-host","destinationNodeId":"destination-host","engine":"ollama","model":"tiny"}`
	if _, err := decodeCopyRequest([]byte(valid), false); err != nil {
		t.Fatalf("valid request: %v", err)
	}
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) {
			if _, err := decodeCopyRequest([]byte(body), false); err == nil {
				t.Fatal("accepted invalid copy request")
			}
		})
	}
	test("duplicate op", strings.Replace(valid, `"opId":`, `"opId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","opId":`, 1))
	test("case variant", strings.Replace(valid, `"model":`, `"Model":`, 1))
	test("forged caller", strings.TrimSuffix(valid, "}")+`,"caller":"trusted"}`)
	test("trailing JSON", valid+`{}`)
	test("invalid op", strings.Replace(valid, "0123456789abcdef0123456789abcdef", "OP", 1))
	test("equal endpoints", strings.Replace(valid, "destination-host", "source-host", 1))
	test("unpaired surrogate", strings.Replace(valid, `"tiny"`, `"\ud800"`, 1))
	test("invalid raw UTF-8", strings.Replace(valid, "tiny", string([]byte{0xff}), 1))
	test("invalid model control", strings.Replace(valid, `"tiny"`, `"bad\u0000model"`, 1))
	test("empty expected identity", strings.TrimSuffix(valid, "}")+`,"expectedExportId":""}`)
	test("escaped duplicate field", strings.TrimSuffix(valid, "}")+`,"mo\u0064el":"other"}`)
	test("oversize", valid+strings.Repeat(" ", maxCopyControlBody))
	for _, field := range []string{"caller", "destinationNodeId", "url", "path", "port"} {
		t.Run("inbound rejects "+field, func(t *testing.T) {
			body := `{"opId":"0123456789abcdef0123456789abcdef","sourceNodeId":"source-host","engine":"ollama","model":"tiny","` + field + `":"x"}`
			if _, err := decodeCopyRequest([]byte(body), true); err == nil {
				t.Fatal("accepted forwarding/authority field")
			}
		})
	}
}

func TestModelCopyControlRejectsBeforeLookup(t *testing.T) {
	output := &syncBuffer{}
	m := &Manager{codec: NewCodec(output)} // No peer directory or executor may be consulted.
	id := json.RawMessage(`1`)
	m.runCopy(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Params: json.RawMessage(`{"opId":"invalid","sourceNodeId":"source","destinationNodeId":"destination","engine":"ollama","model":"tiny"}`)})
	frames := output.frames()
	if len(frames) != 1 || frames[0].Error == nil || frames[0].Error.Code != -32602 || len(m.activeCopies()) != 0 {
		t.Fatalf("invalid request response=%+v", frames)
	}
}

func copyControlManager(t *testing.T, exec *Executor, dir string) *Manager {
	t.Helper()
	m := NewManager(NewCodec(&syncBuffer{}), exec, clustertrust.Open(dir), dir)
	exec.switchRecoveryPending.Store(false)
	exec.nodeProjection.set(nodeProjection{})
	acceptModelMutations(exec)
	t.Cleanup(func() { m.stopCopies(); m.controlHTTP.CloseIdle(); m.controlReadyHTTP.CloseIdle() })
	return m
}

func copyControlNode(t *testing.T, host, principal string, server *httptest.Server) noderec.DirectoryNode {
	t.Helper()
	node := noderec.DirectoryNode{HostUUID: host, ClusterUUID: principal, IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: 14323}}}
	if server != nil {
		ip, portString, err := net.SplitHostPort(server.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		port, err := strconv.Atoi(portString)
		if err != nil {
			t.Fatal(err)
		}
		node.IP = ip
		node.Services = map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: port}}
	}
	return node
}

func copyControlDirectory(m *Manager, nodes ...noderec.DirectoryNode) {
	m.peers.set(nodes)
}

type copyControlFixture struct {
	authority                         *rmFixture
	controller, destination           *Manager
	source                            *Executor
	sourceServer                      *httptest.Server
	destinationServer                 *httptest.Server
	controllerStore, destinationStore string
	reads                             atomic.Int32
}

func newCopyControlFixture(t *testing.T) *copyControlFixture {
	t.Helper()
	return newCopyControlSourceFixture(t, nil)
}

func newCopyControlSourceFixture(t *testing.T, wrap func(http.Handler) http.Handler) *copyControlFixture {
	t.Helper()
	plentyOfDisk(t)
	f := &copyControlFixture{authority: rmNewFixture(t, "destination-certificate", "controller-certificate", "source-certificate")}
	f.authority.writePermissions("1", map[string]bool{"controller-certificate": true, "source-certificate": false})
	sourceDir, controllerDir := f.authority.peer("source-certificate").dir, f.authority.peer("controller-certificate").dir
	grantRemoteManagement(t, sourceDir, "1", map[string]bool{"destination-certificate": false})
	grantRemoteManagement(t, controllerDir, "1", map[string]bool{"destination-certificate": false})
	f.source = copyTestExecutor(t, "ollama", t.TempDir(), true)
	_, sourceStore, err := f.source.modelStoreFor("ollama")
	if err != nil {
		t.Fatal(err)
	}
	writeOllamaModel(t, sourceStore.root, "tiny", 100_000, 2)
	f.source.nodeProjection.set(nodeProjection{})
	sourceMesh := clustertrust.Open(sourceDir)
	sourceControl := &controlServer{exec: f.source, mesh: sourceMesh}
	var sourceHandler http.Handler = sourceControl.mux()
	if wrap != nil {
		sourceHandler = wrap(sourceHandler)
	}
	f.sourceServer = rmTLSServer(t, sourceMesh, sourceControl.requirePinCaller(func(w http.ResponseWriter, r *http.Request, caller string) {
		if caller != "destination-certificate" {
			t.Errorf("source reader = %q, want destination", caller)
		}
		if r.URL.Path == controlModelFilePath {
			f.reads.Add(1)
		}
		sourceHandler.ServeHTTP(w, r)
	}))
	destinationExecutor := copyTestExecutor(t, "ollama", t.TempDir(), true)
	_, destinationStore, err := destinationExecutor.modelStoreFor("ollama")
	if err != nil {
		t.Fatal(err)
	}
	f.destinationStore = destinationStore.root
	f.destination = copyControlManager(t, destinationExecutor, f.authority.dir)
	copyControlDirectory(f.destination, copyControlNode(t, "destination-host", "destination-certificate", nil), copyControlNode(t, "source-host", "source-certificate", f.sourceServer))
	f.destinationServer = rmTLSServer(t, f.authority.mesh, (&controlServer{exec: destinationExecutor, mesh: f.authority.mesh, copyFromPeer: f.destination.copyFromPeer}).mux())
	f.controllerStore = filepath.Join(t.TempDir(), "controller-store-must-never-exist")
	f.controller = copyControlManager(t, copyTestExecutor(t, "ollama", f.controllerStore, true), controllerDir)
	copyControlDirectory(f.controller, copyControlNode(t, "controller-host", "controller-certificate", nil), copyControlNode(t, "destination-host", "destination-certificate", f.destinationServer))
	return f
}

func controlCopyRequest() copyRequest {
	return copyRequest{OpID: "0123456789abcdef0123456789abcdef", SourceNodeID: "source-host", DestinationNodeID: "destination-host", Engine: "ollama", Model: "tiny"}
}

func TestModelCopyControlDestinationOwnsBytes(t *testing.T) {
	f := newCopyControlFixture(t)
	f.controller.exec = nil // Controller artifact/executor access would fail this operation.
	if _, err := f.destination.exec.desired.setLocked("ollama", true); err != nil {
		t.Fatal(err)
	}
	var progress []copyProgress
	res, err := f.controller.executeCopy(remoteFixtureContext(t), controlCopyRequest(), false, nil, func(cp copyProgress) { progress = append(progress, cp) })
	if err != nil {
		t.Fatal(err)
	}
	if res.BytesCopied <= 0 || res.BytesTotal != res.BytesCopied || f.reads.Load() == 0 {
		t.Fatalf("copy=%+v source reads=%d", res, f.reads.Load())
	}
	if _, err := os.Stat(f.controllerStore); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("controller store accessed/created: %v", err)
	}
	terminal := 0
	for _, cp := range progress {
		if cp.OpID != controlCopyRequest().OpID || cp.Node != "source-host" || cp.SourceNodeID != "source-host" || cp.DestinationNodeID != "destination-host" || cp.Model != "tiny" {
			t.Fatalf("progress lost tuple: %+v", cp)
		}
		if cp.Stage == "done" || cp.Stage == "error" || cp.Stage == "cancelled" {
			terminal++
			if cp.Stage != "done" || cp.BytesDone != res.BytesTotal {
				t.Fatalf("terminal = %+v", cp)
			}
		}
	}
	if terminal != 1 {
		t.Fatalf("terminal outcomes = %d", terminal)
	}
	if len(f.controller.activeCopies()) != 0 || len(f.destination.activeCopies()) != 0 {
		t.Fatal("completed operations leaked")
	}
	st, err := f.destination.exec.Status("ollama")
	if err != nil || st.Running || st.Locked == nil || !*st.Locked {
		t.Fatalf("copy activated engine: %+v, %v", st, err)
	}
	enabled, known, err := f.destination.exec.desired.get("ollama")
	if err != nil || !known || enabled {
		t.Fatalf("copy changed saved Off: enabled=%v known=%v err=%v", enabled, known, err)
	}
	list, err := f.source.modelSourceFiles(context.Background(), "ollama", "tiny")
	if err != nil {
		t.Fatal(err)
	}
	if err := hashModelFiles(context.Background(), &list, func(string, int) {}); err != nil {
		t.Fatal(err)
	}
	for _, file := range list.Files {
		data, err := os.ReadFile(filepath.Join(f.destinationStore, filepath.FromSlash(file.Path)))
		if err != nil || int64(len(data)) != file.Size || sha256Hex(data) != file.SHA256 {
			t.Fatalf("destination file %s differs: %v", file.Path, err)
		}
	}
}

func TestModelCopyControlInboundNotifiesDestinationAndController(t *testing.T) {
	controllerSawProgress := make(chan struct{})
	f := newCopyControlSourceFixture(t, func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == controlModelFilePath {
				select {
				case <-controllerSawProgress:
				case <-r.Context().Done():
					return
				}
			}
			next.ServeHTTP(w, r)
		})
	})
	destinationOutput, controllerOutput := &syncBuffer{}, &syncBuffer{}
	f.destination.codec, f.controller.codec = NewCodec(destinationOutput), NewCodec(controllerOutput)
	p := controlCopyRequest()
	params, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	id := json.RawMessage(`1`)
	ctx := remoteFixtureContext(t)
	finished := make(chan struct{})
	go func() {
		f.controller.runCopy(ctx, &Message{JSONRPC: "2.0", ID: &id, Method: "engine:copy-model", Params: params})
		close(finished)
	}()
	// Hold source bytes until an actual controller worker-codec notification
	// arrives. Both endpoint streams must expose nonterminal work, even when
	// no GUI subscriber is attached to either worker's broker.
	deadline := time.Now().Add(2 * time.Second)
	for {
		seen := false
		for _, frame := range controllerOutput.frames() {
			if frame.Method == "engine:remote-progress" {
				seen = true
				break
			}
		}
		if seen {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("controller received no copy notification")
		}
		time.Sleep(time.Millisecond)
	}
	close(controllerSawProgress)
	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("copy did not finish after progress observation")
	}
	var result struct {
		OpID   string          `json:"opId"`
		Result copyModelResult `json:"result"`
	}
	responses := 0
	for _, frame := range controllerOutput.frames() {
		if frame.ID == nil {
			continue
		}
		responses++
		if frame.Error != nil {
			t.Fatalf("controller copy failed: %+v", frame.Error)
		}
		if err := json.Unmarshal(frame.Result, &result); err != nil {
			t.Fatal(err)
		}
	}
	if responses != 1 || result.OpID != p.OpID || result.Result.BytesTotal <= 0 || result.Result.BytesCopied != result.Result.BytesTotal || f.reads.Load() == 0 {
		t.Fatalf("result=%+v responses=%d source reads=%d", result, responses, f.reads.Load())
	}
	check := func(name string, output *syncBuffer) {
		t.Run(name, func(t *testing.T) {
			nonterminal, terminal := 0, 0
			for _, frame := range output.frames() {
				if frame.Method != "engine:remote-progress" {
					continue
				}
				var cp copyProgress
				if err := json.Unmarshal(frame.Params, &cp); err != nil {
					t.Fatal(err)
				}
				if cp.OpID != p.OpID || cp.Node != p.SourceNodeID || cp.SourceNodeID != p.SourceNodeID || cp.DestinationNodeID != p.DestinationNodeID || cp.Engine != p.Engine || cp.Model != p.Model || cp.Op != "copy" {
					t.Fatalf("notification lost tuple: %+v", cp)
				}
				if cp.Stage == "done" || cp.Stage == "error" || cp.Stage == "cancelled" {
					terminal++
					if cp.Stage != "done" || cp.BytesDone != result.Result.BytesTotal || cp.BytesTotal != result.Result.BytesTotal {
						t.Fatalf("terminal=%+v", cp)
					}
				} else {
					nonterminal++
				}
			}
			if nonterminal == 0 || terminal != 1 {
				t.Fatalf("nonterminal=%d terminal=%d", nonterminal, terminal)
			}
		})
	}
	check("destination worker notifications", destinationOutput)
	check("controller worker notifications", controllerOutput)
}

func TestModelCopyControlDestinationWorkerFailureCancelsCopy(t *testing.T) {
	f := newCopyControlFixture(t)
	f.destination.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), &copyFailWriter{}})
	if _, err := f.controller.executeCopy(remoteFixtureContext(t), controlCopyRequest(), false, nil, nil); err == nil {
		t.Fatal("copy survived destination worker transport failure")
	}
	if f.reads.Load() != 0 || len(f.controller.activeCopies()) != 0 || len(f.destination.activeCopies()) != 0 {
		t.Fatalf("failed worker retained work or read artifacts: reads=%d", f.reads.Load())
	}
}

func TestModelCopyControlLegacyUsesCanonicalExecutor(t *testing.T) {
	f := newCopyControlFixture(t)
	output := &syncBuffer{}
	f.destination.codec = NewCodec(output)
	id := json.RawMessage(`1`)
	f.destination.runRemoteCopy(remoteFixtureContext(t), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:remote-copy-model", Params: json.RawMessage(`{"node":"source-host","engine":"ollama","model":"tiny"}`)})
	var result struct {
		OpID   string          `json:"opId"`
		Result copyModelResult `json:"result"`
	}
	terminal, responses := 0, 0
	for _, frame := range output.frames() {
		if frame.ID != nil {
			responses++
			if frame.Error != nil {
				t.Fatalf("legacy copy refusal=%+v", frame.Error)
			}
			if err := json.Unmarshal(frame.Result, &result); err != nil {
				t.Fatal(err)
			}
			continue
		}
		var progress copyProgress
		if err := json.Unmarshal(frame.Params, &progress); err != nil {
			t.Fatal(err)
		}
		if progress.SourceNodeID != "source-host" || progress.DestinationNodeID != "destination-host" || progress.Model != "tiny" {
			t.Fatalf("legacy lost canonical tuple=%+v", progress)
		}
		if progress.Stage == "done" || progress.Stage == "error" || progress.Stage == "cancelled" {
			terminal++
		}
	}
	if responses != 1 || terminal != 1 || !lowerHex(result.OpID, 32) || result.Result.BytesCopied <= 0 || len(f.destination.activeCopies()) != 0 {
		t.Fatalf("legacy response=%+v responses=%d terminal=%d", result, responses, terminal)
	}
}

func TestModelCopyControlCancellationClosesSourceAndRetainsPartial(t *testing.T) {
	started, closed := make(chan struct{}), make(chan struct{})
	weights := testBytes(100_000, 2)
	weightPath := "blobs/sha256-" + sha256Hex(weights)
	f := newCopyControlSourceFixture(t, func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != controlModelFilePath || r.URL.Query().Get("path") != weightPath {
				next.ServeHTTP(w, r)
				return
			}
			w.Header().Set("Content-Length", strconv.Itoa(len(weights)))
			if _, err := w.Write(weights[:4096]); err != nil {
				t.Errorf("source partial: %v", err)
				return
			}
			if err := http.NewResponseController(w).Flush(); err != nil {
				t.Errorf("source flush: %v", err)
				return
			}
			close(started)
			<-r.Context().Done()
			close(closed)
		})
	})
	p := controlCopyRequest()
	result := make(chan error, 1)
	terminal := make(chan copyProgress, 2)
	go func() {
		_, err := f.controller.executeCopy(remoteFixtureContext(t), p, false, nil, func(cp copyProgress) {
			if cp.Stage == "done" || cp.Stage == "error" || cp.Stage == "cancelled" {
				terminal <- cp
			}
		})
		result <- err
	}()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("source did not begin streaming")
	}
	part := filepath.Join(f.destinationStore, filepath.FromSlash(weightPath)+".part")
	deadline := time.Now().Add(2 * time.Second)
	for {
		info, err := os.Stat(part)
		if err == nil && info.Size() > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("destination did not retain positive partial bytes: %v", err)
		}
		time.Sleep(time.Millisecond)
	}
	if f.controller.cancelCopy(strings.Repeat("f", 32)) {
		t.Fatal("unknown opId cancelled active copy")
	}
	if !f.controller.cancelCopy(p.OpID) {
		t.Fatal("exact copy was not active")
	}
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled copy=%v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("controller cancellation did not settle")
	}
	if len(terminal) != 1 {
		t.Fatalf("cancel terminal outcomes=%d", len(terminal))
	}
	if cp := <-terminal; cp.Stage != "cancelled" || cp.OpID != p.OpID {
		t.Fatalf("cancel terminal=%+v", cp)
	}
	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("source body remained open")
	}
	deadline = time.Now().Add(2 * time.Second)
	for len(f.destination.activeCopies()) != 0 {
		if time.Now().After(deadline) {
			t.Fatal("destination retained cancelled reservation")
		}
		time.Sleep(time.Millisecond)
	}
	info, err := os.Stat(part)
	if err != nil || info.Size() <= 0 {
		t.Fatalf("cancel discarded resumable partial: %v", err)
	}
	rel, err := ollamaManifestRel("tiny")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(f.destinationStore, filepath.FromSlash(rel))); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("cancelled copy published model: %v", err)
	}
}

func TestModelCopyControlVerifiedRetryPrecedesSourceLookup(t *testing.T) {
	plentyOfDisk(t)
	source := newCustomCopyFixture(t)
	src := sourceFromExecutor(t, source.source, "llamacpp", reportedCopyAlias)
	dst, store := customDestination(t)
	if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err != nil {
		t.Fatal(err)
	}
	f := rmNewFixture(t, "destination-cert", "controller-cert")
	m := copyControlManager(t, dst, f.dir)
	copyControlDirectory(m, copyControlNode(t, "destination-host", "destination-cert", nil))
	p := copyRequest{OpID: strings.Repeat("a", 32), SourceNodeID: "offline-source-not-in-directory", DestinationNodeID: "destination-host", Engine: "llamacpp", Model: reportedCopyAlias, ExpectedExportID: src.list.ExportID}
	if _, err := m.copyFromPeer(context.Background(), p, func() error {
		return &codedError{code: codeRemoteManagementDenied, message: remoteManagementDeniedMessage}
	}, nil); rpcCode(err) != codeRemoteManagementDenied {
		t.Fatalf("receipt bypassed destination authorization: %v", err)
	}
	res, err := m.executeCopy(context.Background(), p, false, nil, nil)
	if err != nil || res.ExportID != src.list.ExportID || res.FilesSkipped != len(src.list.Files) || res.BytesCopied != 0 || res.Durability != copyDurabilityUncertain {
		t.Fatalf("offline receipt retry=%+v, %v", res, err)
	}
	assertCustomReceipt(t, store, src.list)
	p.ExpectedExportID = strings.Repeat("b", 64)
	if _, err := m.executeCopy(context.Background(), p, false, nil, nil); rpcCode(err) != codePolicyConflict {
		t.Fatalf("changed identity error=%v", err)
	}
	// A hash cached by the original copy must not substitute for re-reading
	// receipt bytes, even when a changed artifact preserves size and mtime.
	file := src.list.Files[0]
	path := filepath.Join(copyAliasDir(store.root, reportedCopyAlias), src.list.ExportID, filepath.FromSlash(file.Path))
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, path, testBytes(int(file.Size), 17))
	if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	p.ExpectedExportID = src.list.ExportID
	if _, err := m.executeCopy(context.Background(), p, false, nil, nil); rpcCode(err) != codePolicyConflict {
		t.Fatalf("tampered receipt bytes error=%v", err)
	}
}

func TestModelCopyControlRemoteVerifiedRetryHasNoSourceContact(t *testing.T) {
	f := newCopyControlFixture(t)
	source := newCustomCopyFixture(t)
	src := sourceFromExecutor(t, source.source, "llamacpp", reportedCopyAlias)
	dst, _ := customDestination(t)
	if _, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", reportedCopyAlias, nil); err != nil {
		t.Fatal(err)
	}
	dst.nodeProjection.set(nodeProjection{})
	f.destination.exec = dst
	copyControlDirectory(f.destination, copyControlNode(t, "destination-host", "destination-certificate", nil))
	server := rmTLSServer(t, f.authority.mesh, (&controlServer{exec: dst, mesh: f.authority.mesh, copyFromPeer: f.destination.copyFromPeer}).mux())
	copyControlDirectory(f.controller, copyControlNode(t, "controller-host", "controller-certificate", nil), copyControlNode(t, "destination-host", "destination-certificate", server))
	p := controlCopyRequest()
	p.Engine, p.Model, p.ExpectedExportID = "llamacpp", reportedCopyAlias, src.list.ExportID
	res, err := f.controller.executeCopy(remoteFixtureContext(t), p, false, nil, nil)
	if err != nil || res.ExportID != p.ExpectedExportID || res.BytesCopied != 0 || res.Durability != copyDurabilityUncertain {
		t.Fatalf("remote offline retry=%+v, %v", res, err)
	}
	if f.reads.Load() != 0 {
		t.Fatal("verified retry contacted source bytes")
	}
	if _, err := os.Stat(f.controllerStore); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("controller touched store: %v", err)
	}
}

func TestModelCopyControlReceiptLockCancellation(t *testing.T) {
	dst, store := customDestination(t)
	unlock, err := copyDestLocks.lock(context.Background(), store.root+"\x00"+reportedCopyAlias)
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()
	ctx, cancel := context.WithCancel(context.Background())
	finished := make(chan error, 1)
	go func() {
		_, _, err := dst.completedCopy(ctx, "llamacpp", reportedCopyAlias, strings.Repeat("b", 64), nil)
		finished <- err
	}()
	cancel()
	select {
	case err := <-finished:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled lock=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancelled retry waited on destination lock")
	}
}

func TestModelCopyControlExpectedExportAndLayoutRefuseBeforeWrites(t *testing.T) {
	test := func(name, expected string, exports, imports []string, wantCode int) {
		t.Run(name, func(t *testing.T) {
			f := newCustomCopyFixture(t)
			source := sourceFromExecutor(t, f.source, "llamacpp", reportedCopyAlias)
			dst, store := customDestination(t)
			_, err := dst.copyModelFrom(context.Background(), source, "llamacpp", reportedCopyAlias, expected, exports, imports, nil)
			if err == nil || rpcCode(err) != wantCode {
				t.Fatalf("refusal=%v", err)
			}
			if len(source.fetchedPaths()) != 0 {
				t.Fatal("refused copy fetched bytes")
			}
			if entries, err := os.ReadDir(store.root); err != nil || len(entries) != 0 {
				t.Fatalf("refused copy wrote destination: %v", err)
			}
		})
	}
	test("identity changed", strings.Repeat("b", 64), []string{llamaArtifactsLayout}, []string{llamaArtifactsLayout}, codePolicyConflict)
	test("actual layout absent from source capability", "", []string{"llamacpp"}, []string{llamaArtifactsLayout}, -32000)
	test("actual layout absent from destination capability", "", []string{llamaArtifactsLayout}, []string{"llamacpp"}, -32000)
}

func TestModelCopyControlPermissionAndSourceAuthority(t *testing.T) {
	t.Run("controller grant belongs to destination", func(t *testing.T) {
		f := newCopyControlFixture(t)
		f.authority.writePermissions("2", map[string]bool{"controller-certificate": false, "source-certificate": false})
		if _, err := f.controller.executeCopy(remoteFixtureContext(t), controlCopyRequest(), false, nil, nil); rpcCode(err) != codeRemoteManagementDenied {
			t.Fatalf("denial=%v", err)
		}
		if f.reads.Load() != 0 {
			t.Fatal("denied controller reached source bytes")
		}
	})
	t.Run("destination must itself pin source", func(t *testing.T) {
		f := newCopyControlFixture(t)
		if err := os.Remove(filepath.Join(f.authority.dir, "trusted", "source-certificate.json")); err != nil {
			t.Fatal(err)
		}
		if _, err := f.controller.executeCopy(remoteFixtureContext(t), controlCopyRequest(), false, nil, nil); err == nil {
			t.Fatal("controller substituted its source authority")
		}
		if f.reads.Load() != 0 {
			t.Fatal("unpinned source served bytes")
		}
	})
	t.Run("source must pin the actual destination reader", func(t *testing.T) {
		f := newCopyControlFixture(t)
		if err := os.Remove(filepath.Join(f.authority.peer("source-certificate").dir, "trusted", "destination-certificate.json")); err != nil {
			t.Fatal(err)
		}
		if _, err := f.controller.executeCopy(remoteFixtureContext(t), controlCopyRequest(), false, nil, nil); err == nil {
			t.Fatal("source accepted an unpinned destination")
		}
		if f.reads.Load() != 0 {
			t.Fatal("source served bytes to unpinned destination")
		}
	})
	t.Run("pause remains a typed veto", func(t *testing.T) {
		f := newCopyControlFixture(t)
		f.destination.exec.nodeProjection.set(nodeProjection{ManagementBlocked: true})
		if _, err := f.controller.executeCopy(remoteFixtureContext(t), controlCopyRequest(), false, nil, nil); rpcCode(err) != codeLocalVeto {
			t.Fatalf("pause refusal=%v", err)
		}
	})
	t.Run("fresh authorization occurs after reservation", func(t *testing.T) {
		f := newCopyControlFixture(t)
		p := controlCopyRequest()
		calls := 0
		_, err := f.destination.executeCopy(context.Background(), p, true, func() error {
			calls++
			if len(f.destination.activeCopies()) != 1 {
				t.Error("authorization did not hold reservation")
			}
			return &codedError{code: codeRemoteManagementDenied, message: remoteManagementDeniedMessage}
		}, nil)
		if rpcCode(err) != codeRemoteManagementDenied || calls != 1 || f.reads.Load() != 0 || len(f.destination.activeCopies()) != 0 {
			t.Fatalf("fresh authorization=%v calls=%d reads=%d", err, calls, f.reads.Load())
		}
	})
}

type copyFailWriter struct {
	writes   int
	flushErr bool
}

func (w *copyFailWriter) Header() http.Header { return http.Header{} }
func (w *copyFailWriter) WriteHeader(int)     {}
func (w *copyFailWriter) Write(b []byte) (int, error) {
	w.writes++
	if w.flushErr {
		return len(b), nil
	}
	return 0, io.ErrClosedPipe
}
func (w *copyFailWriter) FlushError() error { return io.ErrClosedPipe }

func TestModelCopyControlFailedStreamCancelsWork(t *testing.T) {
	test := func(name string, flush bool) {
		t.Run(name, func(t *testing.T) {
			cancelled := make(chan struct{})
			s := &controlServer{copyFromPeer: func(ctx context.Context, p copyRequest, _ func() error, emit func(copyProgress)) (copyModelResult, error) {
				emit(copyProgress{remoteProgress: remoteProgress{OpID: p.OpID, Engine: p.Engine, Op: "copy", Stage: "downloading"}})
				<-ctx.Done()
				close(cancelled)
				return copyModelResult{}, ctx.Err()
			}}
			body := `{"opId":"0123456789abcdef0123456789abcdef","sourceNodeId":"source-host","engine":"ollama","model":"tiny"}`
			req := httptest.NewRequest(http.MethodPost, controlCopyPath, strings.NewReader(body))
			done := make(chan struct{})
			go func() { s.handleCopy(&copyFailWriter{flushErr: flush}, req, "controller"); close(done) }()
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("failed stream did not stop")
			}
			select {
			case <-cancelled:
			default:
				t.Fatal("failed write/flush retained work")
			}
		})
	}
	test("failed write cancels", false)
	test("failed flush cancels", true)
}

func TestModelCopyControlDispatchedMutationIsNotReplayed(t *testing.T) {
	var accepted atomic.Int32
	f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		return (&controlServer{mesh: mesh}).requirePin(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			accepted.Add(1)
			if r.Method != http.MethodPost || r.URL.Path != controlCopyPath || !r.Close {
				t.Errorf("copy dispatch=%s %s close=%v", r.Method, r.URL.Path, r.Close)
			}
			w.Header().Set("Content-Type", "application/x-ndjson")
			cp := copyProgress{remoteProgress: remoteProgress{OpID: controlCopyRequest().OpID, Node: "source-host", Engine: "ollama", Op: "copy", Stage: "checking"}, SourceNodeID: "source-host", DestinationNodeID: "host-peer", Model: "tiny"}
			remoteFixtureEncode(t, w, copyProgressFrame(cp))
			if err := http.NewResponseController(w).Flush(); err != nil {
				t.Errorf("flush dispatched progress: %v", err)
			}
			panic(http.ErrAbortHandler) // Accepted mutation, lost terminal acknowledgement.
		}))
	})
	f.directHealthy.Store(true)
	ctx := remoteFixtureContext(t)
	p := controlCopyRequest()
	p.DestinationNodeID = "host-peer"
	_, err := f.client(t, ctx).copyModel(ctx, p, nil)
	if err == nil || !strings.Contains(err.Error(), "unknown") {
		t.Fatalf("lost response=%v", err)
	}
	if accepted.Load() != 1 || f.directCalls.Load() != 1 || f.reverseCalls.Load() != 0 {
		t.Fatalf("accepted=%d direct=%d reverse=%d", accepted.Load(), f.directCalls.Load(), f.reverseCalls.Load())
	}
}

func TestModelCopyControlReverseStreamsExactTuple(t *testing.T) {
	f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
		return (&controlServer{mesh: mesh}).requirePin(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			data, err := readRemoteResponse(r.Body, maxCopyControlBody)
			if err != nil {
				t.Error(err)
				return
			}
			p, err := decodeCopyRequest(data, true)
			if err != nil {
				t.Error(err)
				return
			}
			if p.SourceNodeID != "source-host" || p.DestinationNodeID != "" {
				t.Errorf("forwarded copy=%+v", p)
			}
			cp := copyProgress{remoteProgress: remoteProgress{OpID: p.OpID, Node: p.SourceNodeID, Engine: p.Engine, Op: "copy", Stage: "downloading", Percent: 50}, SourceNodeID: p.SourceNodeID, DestinationNodeID: "host-peer", Model: p.Model, BytesDone: 5, BytesTotal: 10}
			remoteFixtureEncode(t, w, copyProgressFrame(cp))
			result, err := json.Marshal(copyModelResult{Engine: p.Engine, Model: p.Model, Files: 1, BytesTotal: 10, BytesCopied: 10})
			if err != nil {
				t.Error(err)
				return
			}
			remoteFixtureEncode(t, w, streamFrame{Type: "result", OpID: p.OpID, Engine: p.Engine, Op: "copy", Result: result})
		}))
	})
	ctx := remoteFixtureContext(t)
	p := controlCopyRequest()
	p.DestinationNodeID = "host-peer"
	var progress []copyProgress
	result, err := f.client(t, ctx).copyModel(ctx, p, func(cp copyProgress) { progress = append(progress, cp) })
	if err != nil || result.BytesTotal != 10 || len(progress) != 1 || progress[0].BytesDone != 5 || progress[0].DestinationNodeID != "host-peer" {
		t.Fatalf("result=%+v progress=%+v err=%v", result, progress, err)
	}
	if f.reverseCalls.Load() != 1 {
		t.Fatalf("reverse dispatches=%d", f.reverseCalls.Load())
	}
}

func TestModelCopyControlConcurrentDestinationsRemainIndependent(t *testing.T) {
	f := rmNewFixture(t, "controller-cert", "first-cert", "second-cert")
	controller := copyControlManager(t, copyTestExecutor(t, "ollama", t.TempDir(), true), f.dir)
	release := make(chan struct{})
	t.Cleanup(func() { controller.stopCopies() })
	makeDestination := func(host, principal string) noderec.DirectoryNode {
		dir := f.peer(principal).dir
		grantRemoteManagement(t, dir, "1", map[string]bool{"controller-cert": true})
		mesh := clustertrust.Open(dir)
		ex := copyTestExecutor(t, "ollama", t.TempDir(), true)
		ex.nodeProjection.set(nodeProjection{})
		s := &controlServer{exec: ex, mesh: mesh, copyFromPeer: func(ctx context.Context, p copyRequest, authorize func() error, emit func(copyProgress)) (copyModelResult, error) {
			if err := authorize(); err != nil {
				return copyModelResult{}, err
			}
			emit(copyProgress{remoteProgress: remoteProgress{OpID: p.OpID, Node: p.SourceNodeID, Engine: p.Engine, Op: "copy", Stage: "downloading", Percent: 50}, SourceNodeID: p.SourceNodeID, DestinationNodeID: host, Model: p.Model, BytesDone: 5, BytesTotal: 10})
			select {
			case <-ctx.Done():
				return copyModelResult{}, ctx.Err()
			case <-release:
				return copyModelResult{Engine: p.Engine, Model: p.Model, Files: 1, BytesTotal: 10, BytesCopied: 10}, nil
			}
		}}
		server := rmTLSServer(t, mesh, s.mux())
		return copyControlNode(t, host, principal, server)
	}
	first := makeDestination("first-host", "first-cert")
	second := makeDestination("second-host", "second-cert")
	copyControlDirectory(controller, copyControlNode(t, "controller-host", "controller-cert", nil), first, second)
	ctx := remoteFixtureContext(t)
	p1, p2 := controlCopyRequest(), controlCopyRequest()
	p1.DestinationNodeID, p2.DestinationNodeID = first.HostUUID, second.HostUUID
	p2.OpID = strings.Repeat("b", 32)
	observed := make(chan string, 4)
	firstDone, secondDone := make(chan error, 1), make(chan error, 1)
	start := func(p copyRequest, done chan error) {
		_, err := controller.executeCopy(ctx, p, false, nil, func(cp copyProgress) {
			if cp.Stage == "downloading" {
				observed <- cp.OpID
			}
		})
		done <- err
	}
	go start(p1, firstDone)
	go start(p2, secondDone)
	seen := map[string]bool{}
	for len(seen) < 2 {
		select {
		case id := <-observed:
			seen[id] = true
		case <-time.After(2 * time.Second):
			t.Fatal("both destination streams did not arrive")
		}
	}
	active := controller.activeCopies()
	if len(active) != 2 {
		t.Fatalf("active copies=%+v", active)
	}
	for _, entry := range active {
		if entry.Progress == nil || entry.Progress.OpID != entry.OpID || entry.Progress.DestinationNodeID != entry.DestinationNodeID || entry.SourceNodeID != p1.SourceNodeID || entry.Model != p1.Model {
			t.Fatalf("snapshot mixed concurrent tuples: %+v", entry)
		}
	}
	if !controller.cancelCopy(p1.OpID) {
		t.Fatal("first operation was not active")
	}
	select {
	case err := <-firstDone:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("first cancellation=%v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("first cancellation did not settle")
	}
	if active := controller.activeCopies(); len(active) != 1 || active[0].OpID != p2.OpID {
		t.Fatalf("cancellation affected another destination: %+v", active)
	}
	close(release)
	select {
	case err := <-secondDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("second destination did not complete")
	}
	if len(controller.activeCopies()) != 0 {
		t.Fatal("terminal operations accumulated")
	}
}

func TestModelCopyControlCapabilitiesReadiness(t *testing.T) {
	t.Run("installed stopped locked engine is copy writable", func(t *testing.T) {
		ex := copyTestExecutor(t, "llamacpp", t.TempDir(), true)
		ex.nodeProjection.set(nodeProjection{})
		if _, err := ex.desired.setLocked("llamacpp", true); err != nil {
			t.Fatal(err)
		}
		capability := ex.copyCapabilities("llamacpp")
		if !capability.Available || !capability.Installed || !capability.Writable || capability.ModelStore != "llamacpp" || len(capability.ImportLayouts) != 2 || !capability.IdentityBoundRetry || !capability.ReceiptDurability {
			t.Fatalf("capability=%+v", capability)
		}
		st, err := ex.Status("llamacpp")
		if err != nil || st.Running || st.Locked == nil || !*st.Locked {
			t.Fatalf("copy capability activated engine: %+v, %v", st, err)
		}
	})
	t.Run("not installed is distinct from unavailable", func(t *testing.T) {
		ex := copyTestExecutor(t, "ollama", t.TempDir(), false)
		ex.nodeProjection.set(nodeProjection{})
		capability := ex.copyCapabilities("ollama")
		if !capability.Available || capability.Installed || capability.Writable || capability.Reason == "" {
			t.Fatalf("capability=%+v", capability)
		}
	})
	t.Run("unsupported adapter has no layout", func(t *testing.T) {
		ex := copyTestExecutor(t, "unrecognized-engine", t.TempDir(), true)
		capability := ex.copyCapabilities("unrecognized-engine")
		if capability.Available || capability.Writable || len(capability.ImportLayouts) != 0 {
			t.Fatalf("capability=%+v", capability)
		}
	})
	t.Run("unknown projection and recovery fail closed", func(t *testing.T) {
		ex := copyTestExecutor(t, "ollama", t.TempDir(), true)
		if ex.copyCapabilities("ollama").Writable {
			t.Fatal("unknown policy became writable")
		}
		ex.nodeProjection.set(nodeProjection{})
		ex.switchRecoveryPending.Store(true)
		if ex.copyCapabilities("ollama").Writable {
			t.Fatal("pending recovery became writable")
		}
	})
	t.Run("unsupported protocol never becomes ready", func(t *testing.T) {
		server := httptest.NewServer(http.NotFoundHandler())
		defer server.Close()
		client := &remoteClient{base: server.URL, controlHTTP: server.Client()}
		if _, err := client.copyCapabilities(context.Background(), "ollama"); err == nil {
			t.Fatal("old peer became copy capable")
		}
	})
	t.Run("no layout intersection refuses", func(t *testing.T) {
		if copyLayoutsIntersect(copyCapabilities{ExportLayouts: []string{"source-only"}}, copyCapabilities{ImportLayouts: []string{"destination-only"}}) {
			t.Fatal("unsupported layout accepted")
		}
	})
}

func TestModelCopyControlReservationBoundAndExactCancel(t *testing.T) {
	m := &Manager{}
	var releases []func()
	var contexts []context.Context
	for i := range 8 {
		p := copyRequest{OpID: fmt.Sprintf("%032x", i), SourceNodeID: "s", DestinationNodeID: "d", Engine: "ollama", Model: "tiny"}
		ctx, release, err := m.reserveCopy(context.Background(), p)
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, release)
		contexts = append(contexts, ctx)
	}
	defer func() {
		for _, release := range releases {
			release()
		}
	}()
	if _, _, err := m.reserveCopy(context.Background(), copyRequest{OpID: fmt.Sprintf("%032x", 8)}); rpcCode(err) != codePolicyConflict {
		t.Fatalf("capacity error = %v", err)
	}
	if _, _, err := m.reserveCopy(context.Background(), copyRequest{OpID: fmt.Sprintf("%032x", 0)}); rpcCode(err) != codePolicyConflict || contexts[0].Err() != nil {
		t.Fatalf("duplicate op admission=%v", err)
	}
	if m.cancelCopy("ffffffffffffffffffffffffffffffff") {
		t.Fatal("unknown cancellation matched an operation")
	}
	if !m.cancelCopy(fmt.Sprintf("%032x", 3)) || contexts[3].Err() != context.Canceled {
		t.Fatal("exact operation was not cancelled")
	}
	for i, ctx := range contexts {
		if i != 3 && ctx.Err() != nil {
			t.Fatalf("cancelled sibling %d", i)
		}
	}
	m.stopCopies()
	if got := m.activeCopies(); len(got) != 0 {
		t.Fatalf("shutdown retained %d operations", len(got))
	}
	for _, ctx := range contexts {
		if ctx.Err() != context.Canceled {
			t.Fatal("shutdown left a live context")
		}
	}
}

func TestModelCopyControlSelfUsesDirectoryIdentity(t *testing.T) {
	f := rmNewFixture(t, "self-certificate", "source-certificate")
	m := &Manager{mesh: f.mesh, peers: newPeerDirectory()}
	set := func(nodes ...noderec.DirectoryNode) {
		body, err := json.Marshal(noderec.GetNodesResult{Nodes: nodes})
		if err != nil {
			t.Fatal(err)
		}
		m.applyDiscovery(&Message{Params: body})
	}
	self := copyControlNode(t, "actual-self-host", "self-certificate", nil)
	set(self)
	if got, err := m.copySelfID(); err != nil || got != self.HostUUID {
		t.Fatalf("self = %q, %v", got, err)
	}
	set(self, copyControlNode(t, "duplicate-host", "self-certificate", nil))
	if _, err := m.copySelfID(); err == nil {
		t.Fatal("ambiguous self accepted")
	}
	set(copyControlNode(t, "self-certificate", "unrelated", nil))
	if _, err := m.copySelfID(); err == nil {
		t.Fatal("certificate UUID guessed as host UUID")
	}
}
