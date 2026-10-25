// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

type historyRunTransport struct {
	input  *io.PipeReader
	output *historyBoundaryWriter
}

func (t *historyRunTransport) Read(p []byte) (int, error)  { return t.input.Read(p) }
func (t *historyRunTransport) Write(p []byte) (int, error) { return t.output.Write(p) }
func (t *historyRunTransport) Close() error {
	_ = t.output.Close()
	return t.input.Close()
}

func TestRoutingHistoryProductionRunReplaysEmptyWindowUntilAcknowledged(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		input, feed := io.Pipe()
		output := newHistoryBoundaryWriter(0)
		transport := &historyRunTransport{input: input, output: output}
		root, cancel := context.WithCancel(context.Background())
		fail, join := newTerminalSession(root, cancel, transport)
		defer join()
		defer feed.Close()
		p := NewProxy(newOwnedCodec(root, transport, fail, notificationWriteBudget))
		done := make(chan error, 1)
		go func() { done <- p.Run(root) }()
		policy := nodepolicy.Default()
		policy.Revision = "1"
		policy.Routing.Listener.Enabled = false
		params, err := json.Marshal(nodepolicy.SetPolicyParams{OriginNodeID: "origin", Policy: policy})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.WriteString(feed, `{"jsonrpc":"2.0","id":1,"method":"node/set-policy","params":`+string(params)+"}\n"); err != nil {
			t.Fatal(err)
		}
		synctest.Wait()
		cycles := func() []Message {
			t.Helper()
			frames, _ := output.snapshot()
			var history []Message
			for _, msg := range historyDecodeFrames(t, frames) {
				if msg.Method == nodepolicy.NotifyRoutingHistoryWindow || msg.Method == nodepolicy.NotifyRoutingHistoryWindowEnd {
					history = append(history, msg)
				}
			}
			return history
		}
		historyCheckCycle(t, cycles(), nil, "1")
		frames, _ := output.snapshot()
		var acknowledged bool
		for _, msg := range historyDecodeFrames(t, frames) {
			if msg.ID != nil && string(*msg.ID) == "1" {
				var ack nodepolicy.PolicyAck
				if msg.Error != nil {
					t.Fatalf("policy rejected: %+v", msg.Error)
				}
				if err := json.Unmarshal(msg.Result, &ack); err != nil {
					t.Fatal(err)
				}
				acknowledged = ack.OriginNodeID == "origin" && ack.RunID == p.runID && ack.Revision == "1"
			}
		}
		if !acknowledged {
			t.Fatal("entry read loop did not acknowledge the exact policy binding")
		}
		source := p.routingFactSource()
		if source == nil || string(source.process.origin[:source.process.originLen]) != "origin" || string(source.process.run[:source.process.runLen]) != p.runID {
			t.Fatal("valid policy did not initialize the empty process source")
		}
		// Both a pre-bind drop and a lost accepted reply leave this same window
		// unacknowledged. The sole production timer replays its generation.
		time.Sleep(routingHistoryReconcile)
		synctest.Wait()
		messages := cycles()
		if len(messages) != 4 {
			t.Fatalf("unacknowledged empty window frames = %d, want 4", len(messages))
		}
		historyCheckCycle(t, messages[2:], nil, "1")
		if _, err := io.WriteString(feed, "{\"jsonrpc\":\"2.0\",\"method\":\"routing/history-accepted\",\"params\":{\"schema\":1,\"generation\":\"1\"}}\n"); err != nil {
			t.Fatal(err)
		}
		synctest.Wait()
		if _, err := io.WriteString(feed, `{"jsonrpc":"2.0","id":2,"method":"node/set-policy","params":`+string(params)+"}\n"); err != nil {
			t.Fatal(err)
		}
		synctest.Wait()
		time.Sleep(2 * routingHistoryReconcile)
		synctest.Wait()
		if len(cycles()) != 4 || p.routingFactSource() != source {
			t.Fatal("accepted empty source replayed or was replaced")
		}
		if err := feed.Close(); err != nil {
			t.Fatal(err)
		}
		if err := <-done; err != nil {
			t.Fatal(err)
		}
		p.routingHistory.mu.Lock()
		active := p.routingHistory.active
		p.routingHistory.mu.Unlock()
		if active {
			t.Fatal("Run returned before its history writer completed")
		}
		if err := p.codec.Notify("ordinary-after-run", struct{}{}); err != nil || root.Err() != nil {
			t.Fatalf("history shutdown changed the ordinary terminal-session lifetime: %v / %v", err, root.Err())
		}
	})
}

func TestRoutingHistoryAcknowledgementDispatch(t *testing.T) {
	test := func(name, method, params string, request bool, want bool) {
		t.Run(name, func(t *testing.T) {
			p := newBoundProxy(NewCodec(&recordingWriter{}))
			nativeRegistered(t, p, time.UnixMilli(1000))
			h := p.newRoutingHistory()
			historyTestScan(t, h)
			historyTestCycle(t, h)
			msg := &Message{JSONRPC: "2.0", Method: method, Params: json.RawMessage(params)}
			if request {
				id := json.RawMessage("7")
				msg.ID = &id
			}
			p.handleMessage(msg)
			if h.acknowledged != want {
				t.Fatalf("acknowledged = %v, want %v", h.acknowledged, want)
			}
		})
	}
	test("exact parent notification", nodepolicy.NotifyRoutingHistoryAccepted, `{"schema":1,"generation":"1"}`, false, true)
	test("old generation", nodepolicy.NotifyRoutingHistoryAccepted, `{"schema":1,"generation":"2"}`, false, false)
	test("addressed alias", engines.AddressMethod("ollama", nodepolicy.NotifyRoutingHistoryAccepted), `{"schema":1,"generation":"1"}`, false, false)
	test("request is not an acknowledgement", nodepolicy.NotifyRoutingHistoryAccepted, `{"schema":1,"generation":"1"}`, true, false)
	test("duplicate field", nodepolicy.NotifyRoutingHistoryAccepted, `{"schema":1,"generation":"2","generation":"1"}`, false, false)
	test("missing schema", nodepolicy.NotifyRoutingHistoryAccepted, `{"generation":"1"}`, false, false)
}

func TestRoutingHistoryParentIngressRawGuard(t *testing.T) {
	test := func(name, raw string, accepted bool, refusalID string, refusalCode int) {
		t.Run(name, func(t *testing.T) {
			logs := &bytes.Buffer{}
			previous := slog.Default()
			slog.SetDefault(slog.New(slog.NewTextHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
			defer slog.SetDefault(previous)
			out := &recordingWriter{}
			// The ordinary response is a synchronous barrier after the guarded
			// frame. No direct dispatch or last-wins JSON parsing precedes it.
			ordinary := `{"jsonrpc":"2.0","id":"ordinary","method":"node/set-priority","params":{"generation":1,"nodes":[],"ranks":[]}}`
			rw := struct {
				io.Reader
				io.Writer
			}{strings.NewReader(raw + "\n" + ordinary + "\n"), out}
			p := newBoundProxy(NewCodec(rw))
			nativeRegistered(t, p, time.UnixMilli(1000))
			h := p.newRoutingHistory()
			historyTestScan(t, h)
			historyTestCycle(t, h)
			before := len(out.lines())
			if err := p.readLoop(t.Context()); err != nil {
				t.Fatal(err)
			}
			if h.acknowledged != accepted {
				t.Fatalf("acknowledged = %v, want %v", h.acknowledged, accepted)
			}
			messages := historyDecodeFrames(t, out.lines()[before:])
			want := 1
			if refusalID != "" {
				want++
			}
			if len(messages) != want {
				t.Fatalf("control/ordinary replies = %d, want %d", len(messages), want)
			}
			if refusalID != "" {
				reply := messages[0]
				id := "null"
				if reply.ID != nil {
					id = string(*reply.ID)
				}
				if id != refusalID || reply.Error == nil || reply.Error.Code != refusalCode || reply.Error.Data != nil {
					t.Fatalf("private refusal = %+v / ID %s", reply, id)
				}
			}
			reply := messages[len(messages)-1]
			var result struct {
				Count *int `json:"count"`
			}
			if reply.ID == nil || string(*reply.ID) != `"ordinary"` || reply.Error != nil {
				t.Fatalf("ordinary RPC did not survive: %+v", reply)
			}
			if err := json.Unmarshal(reply.Result, &result); err != nil || result.Count == nil || *result.Count != 0 {
				t.Fatalf("ordinary result = %+v / %v", result, err)
			}
			if strings.Contains(logs.String(), "private-raw-sentinel") {
				t.Fatal("raw history input reached diagnostics")
			}
		})
	}
	test("duplicate method before history", `{"jsonrpc":"2.0","method":"private-raw-sentinel","method":"routing/history-accepted","params":{"schema":1,"generation":"1"}}`, false, "", 0)
	test("duplicate method after history", `{"jsonrpc":"2.0","method":"routing/history-accepted","method":"private-raw-sentinel","params":{"schema":1,"generation":"1"}}`, false, "", 0)
	test("escaped duplicate method", `{"jsonrpc":"2.0","method":"private-raw-sentinel","\u006dethod":"routing/history-accepted","params":{"schema":1,"generation":"1"}}`, false, "", 0)
	test("duplicate params", `{"jsonrpc":"2.0","method":"routing/history-accepted","params":{"private-raw-sentinel":true},"params":{"schema":1,"generation":"1"}}`, false, "", 0)
	test("duplicate unknown subtree", `{"jsonrpc":"2.0","method":"routing/history-accepted","params":{"schema":1,"generation":"1"},"unknown":{"private-raw-sentinel":1,"private-raw-sentinel":2}}`, false, "", 0)
	test("invalid ACK then valid same-generation ACK", `{"jsonrpc":"2.0","method":"private-raw-sentinel","method":"routing/history-accepted","params":{"schema":1,"generation":"1"}}`+"\n"+
		`{"jsonrpc":"2.0","method":"routing/history-accepted","params":{"schema":1,"generation":"1"}}`, true, "", 0)
	test("escaped acknowledgement", `{"jsonrpc":"2.0","\u006dethod":"routing/history-\u0061ccepted","params":{"schema":1,"generation":"1"}}`, true, "", 0)
	test("known acknowledgement strips unknown fields", `{"jsonrpc":"2.0","method":"routing/history-accepted","params":{"schema":1,"generation":"1","unknown":"private-raw-sentinel"}}`, true, "", 0)
	test("null acknowledgement params", `{"jsonrpc":"2.0","method":"routing/history-accepted","params":null}`, false, "", 0)
	test("wrong acknowledgement schema", `{"jsonrpc":"2.0","method":"routing/history-accepted","params":{"schema":2,"generation":"1"}}`, false, "", 0)
	test("old generation", `{"jsonrpc":"2.0","method":"routing/history-accepted","params":{"schema":1,"generation":"2"}}`, false, "", 0)
	test("explicit null ID is not an ACK", `{"jsonrpc":"2.0","id":null,"method":"routing/history-accepted","params":{"schema":1,"generation":"1"}}`, false, "", 0)
	test("overlong private request ID", `{"jsonrpc":"2.0","id":"`+strings.Repeat("x", jsonrpc.HistoryIDBytes)+`","method":"routing/history-accepted","params":{"schema":1,"generation":"1"}}`, false, "null", jsonrpc.HistoryInvalidCode)
	test("private request", `{"jsonrpc":"2.0","id":"private\u0031","method":"routing/history-accepted","params":{"schema":1,"generation":"1"}}`, false, `"private\u0031"`, -32601)
	test("public history request has no proxy owner", `{"jsonrpc":"2.0","id":"public","method":"routing:subscribe","params":{}}`, false, `"public"`, -32601)
	test("wrong notification direction", `{"jsonrpc":"2.0","method":"routing/history-window-end","params":{"schema":1,"generation":"1"}}`, false, "", 0)
}

func TestRoutingHistoryParentIngressRetainsOrdinaryCapacity(t *testing.T) {
	blob := strings.Repeat("x", jsonrpc.HistoryRecordFrameBytes+1)
	raw := `{"jsonrpc":"2.0","id":1,"method":"ordinary","params":{"blob":"` + blob + `"}}` + "\n"
	rw := struct {
		io.Reader
		io.Writer
	}{strings.NewReader(raw), io.Discard}
	p := NewProxy(NewCodec(rw))
	if cap(p.codec.raw) != 1<<20 || p.codec.transport != nil || !reflect.ValueOf(p.codec.Codec).Elem().FieldByName("scanner").IsNil() {
		t.Fatal("history construction retained an old scanner or replaced ordinary capacity with P")
	}
	msg, err := p.codec.Read()
	if err != nil {
		t.Fatal(err)
	}
	var params struct {
		Blob string `json:"blob"`
	}
	if err := json.Unmarshal(msg.Params, &params); err != nil || params.Blob != blob || msg.Method != "ordinary" {
		t.Fatalf("ordinary frame between P and 1MiB changed: %v", err)
	}
}

func TestRoutingHistoryParentIngressKeepsOrdinaryLimit(t *testing.T) {
	raw := `{"jsonrpc":"2.0","method":"ordinary","params":{"blob":"` + strings.Repeat("x", 1<<20) + `"}}` + "\n" +
		`{"jsonrpc":"2.0","method":"next-ordinary","params":{}}` + "\n"
	rw := struct {
		io.Reader
		io.Writer
	}{strings.NewReader(raw), io.Discard}
	p := NewProxy(NewCodec(rw))
	msg, err := p.codec.Read()
	var failure *jsonrpc.HistoryFailure
	if msg != nil || !errors.As(err, &failure) {
		t.Fatalf("over-cap ordinary input escaped its raw allowance: %v / %v", msg, err)
	}
	msg, err = p.codec.Read()
	if err != nil || msg == nil || msg.Method != "next-ordinary" {
		t.Fatalf("next ordinary frame did not survive bounded discard: %v / %v", msg, err)
	}
	if cap(p.codec.raw) != 1<<20 {
		t.Fatal("ordinary raw input grew after overflow")
	}
}

func TestRoutingHistoryProductionRunJoinsBlockedWriter(t *testing.T) {
	test := func(name string, eof bool) {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				input, feed := io.Pipe()
				defer feed.Close()
				output := newHistoryBoundaryWriter(0)
				output.holdAfterClose = true
				defer output.unblock()
				root, cancel := context.WithCancel(context.Background())
				fail, join := newTerminalSession(root, cancel, &historyRunTransport{input: input, output: output})
				defer join()
				p := NewProxy(newOwnedCodec(root, &historyRunTransport{input: input, output: output}, fail, notificationWriteBudget))
				policy := nodepolicy.Default()
				policy.Revision = "1"
				policy.Routing.Listener.Enabled = false
				params, err := json.Marshal(nodepolicy.SetPolicyParams{OriginNodeID: "origin", Policy: policy})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := p.applySetPolicy(params); err != nil {
					t.Fatal(err)
				}
				output.block = output.attempt + 1
				done := make(chan error, 1)
				go func() { done <- p.Run(root) }()
				<-output.entered
				_, pinned := output.snapshot()
				if eof {
					if err := feed.Close(); err != nil {
						t.Fatal(err)
					}
				} else {
					cancel()
				}
				select {
				case <-output.closed:
				case <-time.After(time.Second):
					t.Fatal("process shutdown did not interrupt the actual Write")
				}
				synctest.Wait()
				select {
				case err := <-done:
					t.Fatalf("Run returned while its actual Write still borrowed frame storage: %v", err)
				default:
				}
				_, after := output.snapshot()
				if len(pinned) == 0 || !bytes.Equal(pinned, after) {
					t.Fatal("cancellation reused the active frame")
				}
				if root.Err() == nil {
					t.Fatal("active write interruption bypassed the existing terminal-session owner")
				}
				output.unblock()
				if err := <-done; err != nil {
					t.Fatal(err)
				}
			})
		})
	}
	test("parent cancellation", false)
	test("input EOF with a healthy parent root", true)
}

// Unlike the ordinary notification fixture, exactly one frame blocks. Its
// borrowed slice is intentionally not copied until the actual Write returns.
type historyBoundaryWriter struct {
	mu             sync.Mutex
	frames         [][]byte
	borrowed       []byte
	attempt, block int
	holdAfterClose bool
	entered        chan struct{}
	resume, closed chan struct{}
	resumeOnce     sync.Once
	closeOnce      sync.Once
}

func newHistoryBoundaryWriter(block int) *historyBoundaryWriter {
	return &historyBoundaryWriter{block: block, entered: make(chan struct{}, 1), resume: make(chan struct{}), closed: make(chan struct{})}
}

func (w *historyBoundaryWriter) Read([]byte) (int, error) {
	<-w.closed
	return 0, io.ErrClosedPipe
}

func (w *historyBoundaryWriter) Write(frame []byte) (int, error) {
	w.mu.Lock()
	w.attempt++
	block := w.attempt == w.block
	if block {
		w.borrowed = frame
	}
	w.mu.Unlock()
	if block {
		w.entered <- struct{}{}
		select {
		case <-w.resume:
		case <-w.closed:
			if w.holdAfterClose {
				<-w.resume
			}
		}
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.borrowed = nil
	select {
	case <-w.closed:
		return 0, io.ErrClosedPipe
	default:
	}
	w.frames = append(w.frames, bytes.Clone(frame))
	return len(frame), nil
}

func (w *historyBoundaryWriter) Close() error {
	w.closeOnce.Do(func() { close(w.closed) })
	return nil
}

func (w *historyBoundaryWriter) unblock() { w.resumeOnce.Do(func() { close(w.resume) }) }

func (w *historyBoundaryWriter) snapshot() (frames [][]byte, borrowed []byte) {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, f := range w.frames {
		frames = append(frames, bytes.Clone(f))
	}
	return frames, bytes.Clone(w.borrowed)
}

func historyOwnedCodec(t *testing.T, w *historyBoundaryWriter) (context.Context, context.CancelFunc, *Codec) {
	t.Helper()
	root, cancel := context.WithCancel(context.Background())
	fail, join := newTerminalSession(root, cancel, w)
	t.Cleanup(func() { w.unblock(); join() })
	return root, cancel, newOwnedCodec(root, w, fail, time.Millisecond)
}

func historyDecodeFrames(t *testing.T, frames [][]byte) []Message {
	t.Helper()
	messages := make([]Message, 0, len(frames))
	for _, frame := range frames {
		var msg Message
		if err := json.Unmarshal(frame, &msg); err != nil {
			t.Fatal(err)
		}
		if msg.JSONRPC != "2.0" {
			t.Fatal("invalid JSON-RPC envelope")
		}
		messages = append(messages, msg)
	}
	return messages
}

func historyCheckCycle(t *testing.T, messages []Message, ids []uint64, generation string) {
	t.Helper()
	if len(messages) != len(ids)+2 || messages[0].Method != "routing/history-window" || messages[len(messages)-1].Method != "routing/history-window-end" {
		t.Fatal("window frame boundaries/count differ")
	}
	var manifest nodepolicy.RoutingHistoryWindow
	if err := json.Unmarshal(messages[0].Params, &manifest); err != nil {
		t.Fatal(err)
	}
	var end nodepolicy.RoutingHistoryWindowEnd
	if err := json.Unmarshal(messages[len(messages)-1].Params, &end); err != nil {
		t.Fatal(err)
	}
	if manifest.Generation != generation || end.Generation != generation || len(manifest.Keys) != len(ids) {
		t.Fatal("mixed generations or missing manifest keys")
	}
	for i, id := range ids {
		if messages[i+1].Method != "routing/request" || messages[i+1].ID != nil {
			t.Fatal("record is not a private notification")
		}
		var record nodepolicy.RoutingHistoryRecord
		if err := json.Unmarshal(messages[i+1].Params, &record); err != nil {
			t.Fatal(err)
		}
		key := manifest.Keys[i]
		if record.Request != key.Request || record.Request.RequestID != strconv.FormatUint(id, 10) || record.CreatedAt != key.CreatedAt || record.Seq != key.Seq {
			t.Fatal("manifest does not describe exactly the emitted sorted records")
		}
	}
	if len(ids) == 256 {
		if manifest.Cutoff == nil || manifest.Cutoff.Request != manifest.Keys[0].Request || manifest.Cutoff.CreatedAt != manifest.Keys[0].CreatedAt {
			t.Fatal("wrong retained source cutoff")
		}
	} else if manifest.Cutoff != nil {
		t.Fatal("premature source cutoff")
	}
}

func TestRoutingHistoryBlockedWrite(t *testing.T) {
	// Stop at begin, every record, and end. Every schedule uses the real shared
	// write gate and synchronous writer, with no sleeps for producer ordering.
	for _, boundary := range []struct {
		name  string
		frame int
	}{{"begin", 1}, {"first record", 2}, {"middle record", 3}, {"last record", 4}, {"end", 5}} {
		t.Run(boundary.name, func(t *testing.T) {
			w := newHistoryBoundaryWriter(boundary.frame)
			root, _, codec := historyOwnedCodec(t, w)
			profile, ok := profileFor("ollama")
			if !ok {
				t.Fatal("missing fixture profile")
			}
			p := newTestProxy(profile, codec, NewDiscovery(), 0)
			old := nativeRegistered(t, p, time.UnixMilli(10000))
			nativeRegistered(t, p, time.UnixMilli(10000))
			nativeRegistered(t, p, time.UnixMilli(10000))
			old.owner.mu.Lock()
			old.owner.reason = "raw-error-prompt-credential-URL-sentinel"
			old.owner.mu.Unlock()
			h := p.newRoutingHistory()
			historyTestScan(t, h)
			frozen := h.frozen.cells[0].facts
			select {
			case <-h.writeWake:
			default:
				t.Fatal("initial cycle not scheduled")
			}
			done := make(chan struct{})
			go func() { h.writeCycle(); close(done) }()
			lifetimeReceive(t, w.entered)
			_, pinned := w.snapshot()
			if len(pinned) == 0 {
				t.Fatal("test missed actual write borrower")
			}
			// More than a full source window completes while output is stopped.
			// No producer calls the history owner or awaits its gates/ack.
			progress := make(chan error, 1)
			go func() {
				for i := 0; i < 260; i++ {
					r, err := p.registerRoutingRequest(newRoutingRequest(nil))
					if err != nil {
						progress <- err
						return
					}
					p.attachRoutingFacts(r, time.UnixMilli(int64(10001+i)))
					r.owner.observeIngress(routingIngressResult{factFailed, factRequestInvalid, http.StatusBadRequest})
					p.unregisterRoutingRequest(r)
				}
				progress <- nil
			}()
			if err := lifetimeReceive(t, progress); err != nil {
				t.Fatal(err)
			}
			historyTestScan(t, h)
			// A backward clock registration is immediately evicted; a later old
			// live update must not reinsert that owner into either source window.
			rollback := nativeRegistered(t, p, time.UnixMilli(1000))
			rollback.owner.observeIngress(routingIngressResult{factFailed, factRequestInvalid, 400})
			p.unregisterRoutingRequest(rollback)
			if old.owner.cancelRouting(false) != nodepolicy.RoutingCancelCancelled {
				t.Fatal("history pressure blocked canonical request cancellation")
			}
			historyTestScan(t, h)
			if h.current.count != 256 || h.current.find(1) >= 0 || h.current.find(264) >= 0 || h.frozen.count != 3 || h.frozen.cells[0].facts != frozen || h.generation.Low != 1 {
				t.Fatal("source eviction or frozen ownership changed under active Write")
			}
			for id := uint64(8); id <= 263; id++ {
				if h.current.find(id) < 0 {
					t.Fatalf("retained source lost permitted key %d", id)
				}
			}
			// This proves only nonwaiting admission refusal and early authenticated
			// ingress validation during a physically blocked Write. Valid inference
			// uses ordinary notifications and retains their existing pipe coupling;
			// the separate between-frame test covers its progress guarantee.
			decision := p.admission.admit(context.Background(), admissionRequest{engine: "ollama", model: "m", waitCap: 0})
			if decision.reject != nodepolicy.RejectOff || decision.ticket != nil || decision.err != nil {
				t.Fatalf("admission changed under history pressure: %+v", decision)
			}
			strict := strictRequest()
			strict.Request.OriginNodeID = "claimed"
			header, err := json.Marshal(strict)
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m"}`))
			req.Header.Set(nodepolicy.StrictAdmissionHeader, string(header))
			rec := httptest.NewRecorder()
			p.soleFacade().serveIngress(rec, req, "authenticated-other")
			if rec.Code != http.StatusForbidden {
				t.Fatalf("ordinary ingress validation blocked/changed: %d", rec.Code)
			}
			_, after := w.snapshot()
			if root.Err() != nil || !h.active || !bytes.Equal(after, pinned) {
				t.Fatal("pressure killed parent or reused active frame")
			}
			w.unblock()
			lifetimeReceive(t, done)
			historyTestScan(t, h)
			if h.generation.Low != 2 {
				t.Fatal("coalesced content did not get a new generation after completion")
			}
			historyTestCycle(t, h)
			frames, _ := w.snapshot()
			messages := historyDecodeFrames(t, frames)
			historyCheckCycle(t, messages[:5], []uint64{1, 2, 3}, "1")
			ids := make([]uint64, 256)
			for i := range ids {
				ids[i] = uint64(i + 8)
			}
			historyCheckCycle(t, messages[5:], ids, "2")
			if bytes.Contains(bytes.Join(frames, nil), []byte("raw-error-prompt-credential-URL-sentinel")) {
				t.Fatal("projection reconstructed native reasons from raw error text")
			}
			if !h.accepted(nodepolicy.RoutingHistoryAccepted{Schema: 1, Generation: "2"}) {
				t.Fatal("latest generation ack refused")
			}
			h.reconcile(true)
			if h.active || len(h.writeWake) != 0 || root.Err() != nil {
				t.Fatal("clean accepted window repeated IPC or retired parent")
			}
		})
	}
}

func TestRoutingHistoryBetweenFramesAllowsInference(t *testing.T) {
	for _, boundary := range []struct {
		name       string
		afterFrame int
		accepted   bool
	}{
		{"before begin", 0, false},
		{"after begin", 1, false},
		{"after first record", 2, false},
		{"after middle record", 3, false},
		{"before end", 4, false},
		{"awaiting ack", 5, false},
		{"awaiting next copier turn after ack", 5, true},
	} {
		t.Run(boundary.name, func(t *testing.T) {
			w := newHistoryBoundaryWriter(0)
			root, _, codec := historyOwnedCodec(t, w)
			codec.budget = time.Second // A leaked gate fails promptly under the ordinary Notify owner.
			profile := ollamaCase(t).profile
			const model = "history-boundary:tag"
			engine := newRecordingEngine(t, http.StatusOK)
			discovery := NewDiscovery()
			discovery.AddManual(selfNode("self", profile.FacadePort, model))
			p := newTestProxy(profile, codec, discovery, profile.FacadePort)
			setBackend(t, p.soleFacade(), engine.URL, true)
			candidates := p.soleFacade().resolveCandidates(model)
			if len(candidates) != 1 || !candidates[0].self || candidates[0].url.Host != strings.TrimPrefix(engine.URL, "http://") {
				t.Fatal("fixture must route only to its private loopback engine")
			}
			p.admission.mu.Lock()
			p.admission.stateSentAt = time.Time{} // Make the actual grant-state publication observable.
			p.admission.mu.Unlock()
			for i := 0; i < 3; i++ {
				nativeRegistered(t, p, time.UnixMilli(int64(1000+i)))
			}
			h := p.newRoutingHistory()
			historyTestScan(t, h)
			select {
			case <-h.writeWake:
			default:
				t.Fatal("coherent cycle was not scheduled")
			}
			// Schedule the SAME step and completion owner used by writeCycle.
			// A pause follows actual frame/callback return, never a transport
			// callback, reconstructed method list or manual active-state change.
			framesInCycle := h.frozen.count + 2
			cursor, sent := 0, 0
			more := true
			advance := func(to int) {
				t.Helper()
				for sent < to {
					if !more {
						t.Fatal("canonical step completed before the requested boundary")
					}
					more = h.writeCycleStep(&cursor)
					sent++
					if h.failed || more != (sent < framesInCycle) || h.active != more {
						t.Fatal("canonical frame/completion handoff did not match cycle progress")
					}
				}
			}
			advance(boundary.afterFrame)
			if boundary.accepted && !h.accepted(nodepolicy.RoutingHistoryAccepted{Schema: 1, Generation: "1"}) {
				t.Fatal("clean-cycle acknowledgement was refused")
			}
			if h.active != (sent < framesInCycle) || h.acknowledged != boundary.accepted {
				t.Fatal("writer pause does not represent the named cycle boundary")
			}
			before, borrowed := w.snapshot()
			if len(borrowed) != 0 {
				t.Fatal("fixture paused an active Write instead of a frame boundary")
			}
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"history-boundary:tag","stream":false}`)).WithContext(root)
			done := make(chan struct{})
			go func() { defer close(done); p.soleFacade().handleHTTP(rec, req) }()
			lifetimeReceive(t, done) // History remains paused until AFTER the entire handler returns.
			var response struct {
				Done bool `json:"done"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if rec.Code != http.StatusOK || !response.Done || engine.hits() != 1 || p.admission.currentActive() != 0 || root.Err() != nil {
				t.Fatalf("valid inference did not finish normally at idle history boundary: status=%d done=%v hits=%d active=%d parent=%v", rec.Code, response.Done, engine.hits(), p.admission.currentActive(), root.Err())
			}
			after, borrowed := w.snapshot()
			if len(borrowed) != 0 {
				t.Fatal("ordinary completion left a codec borrower")
			}
			var submitted, running, terminal, started, completed, activity int
			var sequence int64
			var sawAdmission bool
			var lastAdmission int
			for _, msg := range historyDecodeFrames(t, after[len(before):]) {
				switch msg.Method {
				case workloadSubmittedMethod, workloadStartedMethod, workloadCompletedMethod, workloadErroredMethod:
					var params workloadParams
					if err := json.Unmarshal(msg.Params, &params); err != nil {
						t.Fatal(err)
					}
					workload := params.WorkloadInfo
					if terminal != 0 || workload.ID != "1" || workload.Engine != profile.Name || workload.RunID != p.runID || workload.Model != model || workload.Seq <= sequence {
						t.Fatal("workload order/identity changed or a terminal workload resurrected")
					}
					sequence = workload.Seq
					if msg.Method == workloadSubmittedMethod && workload.State == "queued" {
						submitted++
					} else if msg.Method == workloadStartedMethod && workload.State == "running" && workload.StartedAt != nil {
						running++
					} else if msg.Method == workloadCompletedMethod && workload.State == "completed" && workload.CompletedAt != nil {
						terminal++
					} else {
						t.Fatal("ordinary workload lifecycle payload/method changed")
					}
				case engines.AddressMethod(profile.Name, "proxy/request-started"):
					var event RequestStartedEvent
					if err := json.Unmarshal(msg.Params, &event); err != nil {
						t.Fatal(err)
					}
					if event.ID != "1" || event.NodeID != "self" || event.Path != "/api/chat" {
						t.Fatal("ordinary request-started identity changed")
					}
					started++
				case engines.AddressMethod(profile.Name, "proxy/request"):
					var event RequestEvent
					if err := json.Unmarshal(msg.Params, &event); err != nil {
						t.Fatal(err)
					}
					if event.ID != "1" || event.NodeID != "self" || event.Status != http.StatusOK || event.Error != "" {
						t.Fatal("ordinary request completion changed")
					}
					completed++
				case noderec.NotifyNodeActivity:
					var event noderec.NodeActivityParams
					if err := json.Unmarshal(msg.Params, &event); err != nil {
						t.Fatal(err)
					}
					if event.HostUUID != "self" {
						t.Fatal("ordinary node activity attribution changed")
					}
					activity++
				case nodepolicy.NotifyAdmissionState:
					var state nodepolicy.AdmissionState
					if err := json.Unmarshal(msg.Params, &state); err != nil {
						t.Fatal(err)
					}
					sawAdmission = sawAdmission || state.Active == 1
					lastAdmission = state.Active
				case nodepolicy.NotifyRoutingHistoryWindow, nodepolicy.NotifyRoutingRequest, nodepolicy.NotifyRoutingHistoryWindowEnd:
					t.Fatal("history advanced while its sole writer was paused")
				}
			}
			if submitted == 0 || running != 1 || terminal != 1 || started != 1 || completed != 1 || activity != 1 || !sawAdmission || lastAdmission != 0 {
				t.Fatalf("missing ordinary publication while history idle: submitted=%d running=%d terminal=%d started=%d completed=%d activity=%d admission=%v/%d", submitted, running, terminal, started, completed, activity, sawAdmission, lastAdmission)
			}
			pausedInCycle := more
			if pausedInCycle {
				// Coalesce the just-completed real request into current while the
				// original frozen window is still owned by this writer. Completion
				// must wake the copier to publish that newer complete window.
				historyTestScan(t, h)
				if !h.dirty || h.current.find(4) < 0 || h.frozen.count != 3 || h.generation.Low != 1 {
					t.Fatal("copier did not coalesce independently of the paused writer")
				}
			}
			advance(framesInCycle)
			if h.active || h.failed || cursor >= 0 || h.dirty != pausedInCycle {
				t.Fatal("canonical completion did not release exactly the completed cycle")
			}
			if pausedInCycle {
				select {
				case <-h.wake:
				default:
					t.Fatal("canonical completion lost the coalesced dirty wake")
				}
			}
			if h.writeCycleStep(&cursor) {
				t.Fatal("completed cursor admitted another frame")
			}
			select {
			case <-h.wake:
				t.Fatal("completed cursor repeated its completion/dirty-wake handoff")
			default:
			}
			frames, _ := w.snapshot()
			var history []Message
			for _, msg := range historyDecodeFrames(t, frames) {
				if msg.Method == nodepolicy.NotifyRoutingHistoryWindow || msg.Method == nodepolicy.NotifyRoutingRequest || msg.Method == nodepolicy.NotifyRoutingHistoryWindowEnd {
					history = append(history, msg)
				}
			}
			historyCheckCycle(t, history, []uint64{1, 2, 3}, "1")
			beforeNextCycle := len(frames)
			historyTestScan(t, h) // Actual copier turn after the canonical handoff.
			if !h.active || h.dirty || h.generation.Low != 2 || h.frozen.count != 4 {
				t.Fatal("completion did not permit the coalesced replacement cycle")
			}
			if h.writeCycleStep(&cursor) || !h.active {
				t.Fatal("old completed cursor released a later cycle's frozen bank")
			}
			historyTestCycle(t, h) // Production wrapper uses the same canonical step.
			frames, _ = w.snapshot()
			historyCheckCycle(t, historyDecodeFrames(t, frames[beforeNextCycle:]), []uint64{1, 2, 3, 4}, "2")
			if h.active || h.failed || h.dirty {
				t.Fatal("production wrapper failed its canonical completion handoff")
			}
			select {
			case <-h.wake:
				t.Fatal("clean canonical completion emitted a dirty wake")
			default:
			}
			if root.Err() != nil {
				t.Fatal("ordinary completion or resumed history retired the healthy parent")
			}
		})
	}
}

func TestParentNotificationHistoryGateDoesNotWaitOrFailParent(t *testing.T) {
	w := newHistoryBoundaryWriter(1)
	root, _, codec := historyOwnedCodec(t, w)
	p := newBoundProxy(codec) // Complete production codec construction before any IO.
	id := json.RawMessage(`1`)
	ordinary := make(chan error, 1)
	go func() { ordinary <- codec.Respond(&id, struct{}{}) }()
	lifetimeReceive(t, w.entered)
	nativeRegistered(t, p, time.UnixMilli(1000))
	h := p.newRoutingHistory()
	historyTestScan(t, h)
	select {
	case <-h.writeWake:
	default:
		t.Fatal("initial cycle not scheduled")
	}
	done := make(chan struct{})
	go func() { h.writeCycle(); close(done) }()
	lifetimeReceive(t, done)
	if h.active || h.failed || root.Err() != nil {
		t.Fatal("failed history gate acquisition waited, failed history or killed parent")
	}
	frames, pinned := w.snapshot()
	if len(frames) != 0 || len(pinned) == 0 {
		t.Fatal("history gate failure touched the ordinary active writer")
	}
	w.unblock()
	if err := lifetimeReceive(t, ordinary); err != nil {
		t.Fatal(err)
	}
	h.reconcile(true)
	historyTestCycle(t, h)
	frames, _ = w.snapshot()
	messages := historyDecodeFrames(t, frames)
	if messages[0].ID == nil || !bytes.Equal(*messages[0].ID, id) {
		t.Fatal("ordinary response was displaced")
	}
	historyCheckCycle(t, messages[1:], []uint64{1}, "1")
}

func TestRoutingHistoryReconciliationSingleWorker(t *testing.T) {
	w := newHistoryBoundaryWriter(1)
	root, cancel, codec := historyOwnedCodec(t, w)
	p := newBoundProxy(codec)
	nativeRegistered(t, p, time.UnixMilli(1000))
	h := p.newRoutingHistory()
	done := make(chan struct{})
	go func() { h.run(); close(done) }()
	t.Cleanup(func() {
		cancel()
		w.unblock()
		lifetimeReceive(t, done)
	})
	lifetimeReceive(t, w.entered)
	duplicate := make(chan struct{})
	go func() { h.run(); close(duplicate) }()
	lifetimeReceive(t, duplicate)
	w.mu.Lock()
	attempts := w.attempt
	w.mu.Unlock()
	if attempts != 1 || root.Err() != nil {
		t.Fatal("duplicate start admitted another writer/timer or failed parent")
	}
	cancel()
	lifetimeReceive(t, done)
	h.mu.Lock()
	active := h.active
	h.mu.Unlock()
	if active {
		t.Fatal("copier returned before its real writer joined")
	}
}

func TestParentNotificationHistoryActiveWriteUsesParentLifetime(t *testing.T) {
	w := newHistoryBoundaryWriter(1)
	root, _, codec := historyOwnedCodec(t, w)
	done := make(chan error, 1)
	frame := []byte("{\"jsonrpc\":\"2.0\",\"method\":\"routing/history-window-end\",\"params\":{\"schema\":1,\"generation\":\"1\"}}\n")
	go func() { done <- codec.writeRoutingFrame(root, frame) }()
	lifetimeReceive(t, w.entered)
	// The standard parent owns exactly the single J Write registration while
	// this borrower is blocked. No timer/callback is minted per record/key.
	children := reflect.ValueOf(root).Elem().FieldByName("children")
	if !children.IsValid() || children.Len() != 1 {
		t.Fatal("active history writer exceeded its one callback association")
	}
	// The ordinary 1ms notification budget expires many times while the
	// history borrower remains active on the healthy parent's lifetime.
	timer := time.NewTimer(20 * time.Millisecond)
	defer timer.Stop()
	select {
	case err := <-done:
		t.Fatalf("history write inherited ordinary notification timeout: %v", err)
	case <-timer.C:
	}
	if root.Err() != nil {
		t.Fatal("history pressure retired the parent")
	}
	w.unblock()
	if err := lifetimeReceive(t, done); err != nil {
		t.Fatal(err)
	}
	if children.Len() != 0 || children.IsNil() {
		t.Fatal("completed Write retained its callback or hid the charged map backing")
	}
	if err := codec.Notify("ordinary-after-history", struct{}{}); err != nil {
		t.Fatal(err)
	}
}

func TestParentNotificationHistoryPinsUntilActualCompletion(t *testing.T) {
	w := newHistoryBoundaryWriter(2) // Begin succeeds; first record never completes.
	w.holdAfterClose = true
	root, cancel, codec := historyOwnedCodec(t, w)
	p := newBoundProxy(codec)
	r := nativeRegistered(t, p, time.UnixMilli(1000))
	h := p.newRoutingHistory()
	historyTestScan(t, h)
	before := h.frozen.cells[0].facts
	select {
	case <-h.writeWake:
	default:
		t.Fatal("cycle not scheduled")
	}
	done := make(chan struct{})
	go func() { h.writeCycle(); close(done) }()
	lifetimeReceive(t, w.entered)
	_, pinned := w.snapshot()
	cancel()
	lifetimeReceive(t, w.closed)
	r.owner.observeRequested(requestedDescriptor("new", "", "/api/chat"))
	historyTestScan(t, h)
	_, after := w.snapshot()
	if !h.active || root.Err() == nil || !reflect.DeepEqual(before, h.frozen.cells[0].facts) || !bytes.Equal(pinned, after) || h.generation.Low != 1 {
		t.Fatal("parent cancellation released storage before actual borrower completion")
	}
	select {
	case <-done:
		t.Fatal("active write completed without physical release")
	default:
	}
	w.unblock()
	lifetimeReceive(t, done)
	frames, _ := w.snapshot()
	messages := historyDecodeFrames(t, frames)
	if len(messages) != 1 || messages[0].Method != "routing/history-window" || h.active || !h.failed {
		t.Fatal("source death fabricated a complete end or retained active borrower")
	}
}

func TestParentNotificationHistoryBrokenPipeUsesExistingFailureOwner(t *testing.T) {
	w := newHistoryBoundaryWriter(0)
	root, _, codec := historyOwnedCodec(t, w)
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	err := codec.writeRoutingFrame(root, []byte("{}\n"))
	if err == nil || root.Err() == nil || (!errors.Is(err, io.ErrClosedPipe) && !errors.Is(err, context.Canceled)) {
		t.Fatalf("broken pipe bypassed existing parent owner: err=%v parent=%v", err, root.Err())
	}
	if err := codec.WritePreencoded(nil, context.Background(), []byte("{}\n")); !errors.Is(err, jsonrpc.ErrPeerClosed) {
		t.Fatalf("failed transport remained usable: %v", err)
	}
}
