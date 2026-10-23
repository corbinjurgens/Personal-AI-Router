// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hashicorp/yamux"
	"nvpair-shared/noderec"
)

// recordedLog keeps each record's message and attributes for assertions.
type recordedLog struct {
	mu      sync.Mutex
	records []map[string]string
}

func (r *recordedLog) Enabled(context.Context, slog.Level) bool { return true }
func (r *recordedLog) WithAttrs([]slog.Attr) slog.Handler       { return r }
func (r *recordedLog) WithGroup(string) slog.Handler            { return r }
func (r *recordedLog) Handle(_ context.Context, record slog.Record) error {
	fields := map[string]string{"msg": record.Message, "level": record.Level.String()}
	record.Attrs(func(a slog.Attr) bool {
		fields[a.Key] = a.Value.String()
		return true
	})
	r.mu.Lock()
	r.records = append(r.records, fields)
	r.mu.Unlock()
	return nil
}

// find returns the records with msg whose fields include every want pair.
func (r *recordedLog) find(msg string, want map[string]string) []map[string]string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var found []map[string]string
	for _, fields := range r.records {
		if fields["msg"] != msg {
			continue
		}
		match := true
		for key, value := range want {
			match = match && fields[key] == value
		}
		if match {
			found = append(found, fields)
		}
	}
	return found
}

func (r *recordedLog) await(t *testing.T, msg string, want map[string]string) map[string]string {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		if found := r.find(msg, want); len(found) > 0 {
			return found[0]
		}
		if time.Now().After(deadline) {
			r.mu.Lock()
			defer r.mu.Unlock()
			t.Fatalf("no %q record with %v; logged %v", msg, want, r.records)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func newRecordedLog() (*slog.Logger, *recordedLog) {
	r := &recordedLog{}
	return slog.New(r), r
}

func TestReverseSessionCloseLogsItsReason(t *testing.T) {
	test := func(name string, end func(a, b *ReverseManager, admitted *atomic.Bool), wantReason string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			logger, records := newRecordedLog()
			var admitted atomic.Bool
			admitted.Store(true)
			a := reverseTestManager(t, ReverseOptions{LocalUUID: "a", Log: logger,
				Validate: func(peer ReversePeer) bool { return peer.Incarnation == "v1" && admitted.Load() }})
			b := reverseTestManager(t, ReverseOptions{LocalUUID: "b"})
			reverseTestConnect(t, a, b, "a", reverseTestID)
			end(a, b, &admitted)
			closed := records.await(t, "reverse session closed", map[string]string{"peer": "b", "direction": "outbound"})
			if wantReason == "" {
				if !strings.HasPrefix(closed["reason"], "session ended: ") && !strings.HasPrefix(closed["reason"], "control stream read failed: ") {
					t.Fatalf("close reason = %q, want the transport ending", closed["reason"])
				}
			} else if closed["reason"] != wantReason {
				t.Fatalf("close reason = %q, want %q", closed["reason"], wantReason)
			}
			if closed["uptime"] == "" {
				t.Fatalf("close record lacks uptime: %v", closed)
			}
		})
	}
	test("revoked locally", func(a, _ *ReverseManager, _ *atomic.Bool) { a.Revoke("b") }, "revoked")
	test("peer no longer admitted", func(a, _ *ReverseManager, admitted *atomic.Bool) {
		admitted.Store(false)
		a.Revalidate()
	}, "peer no longer admitted")
	// The remote close is seen first by the control stream's reader or by the
	// stream acceptor, depending on timing; both name the transport ending.
	test("remote ended the session", func(_, b *ReverseManager, _ *atomic.Bool) { b.Revoke("a") }, "")
	test("manager closed", func(a, _ *ReverseManager, _ *atomic.Bool) { a.Close() }, "reverse manager closed")
}

func TestReverseDuplicateResolutionIsLogged(t *testing.T) {
	logger, records := newRecordedLog()
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a", Log: logger})
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b"})
	// The session initiated by b exists first; a's own session wins the tie.
	reverseTestConnect(t, a, b, "b", reverseTestID)
	reverseTestConnect(t, a, b, "a", "22222222222222222222222222222222")
	replaced := records.await(t, "reverse session duplicate resolved", map[string]string{"peer": "b", "kept": "new"})
	if replaced["keptDirection"] != "outbound" || replaced["replacedDirection"] != "inbound" {
		t.Fatalf("replacement record = %v", replaced)
	}
	records.await(t, "reverse session closed", map[string]string{"peer": "b", "direction": "inbound", "reason": "replaced by duplicate session"})

	left, right := net.Pipe()
	defer right.Close()
	err := a.Register(ReversePeer{"b", "v1"}, left, "b", "00000000000000000000000000000000")
	if !errors.Is(err, ErrReverseDuplicate) {
		t.Fatalf("losing duplicate = %v, want ErrReverseDuplicate", err)
	}
	kept := records.await(t, "reverse session duplicate resolved", map[string]string{"peer": "b", "kept": "existing"})
	if kept["existingDirection"] != "outbound" || kept["rejectedDirection"] != "inbound" {
		t.Fatalf("rejection record = %v", kept)
	}
}

func TestReverseSessionLogsNameOnlyPeerAndDirection(t *testing.T) {
	logger, records := newRecordedLog()
	a := reverseTestManager(t, ReverseOptions{LocalUUID: "a", Log: logger})
	b := reverseTestManager(t, ReverseOptions{LocalUUID: "b"})
	reverseTestConnect(t, a, b, "a", reverseTestID)
	a.Revoke("b")
	records.await(t, "reverse session closed", map[string]string{"peer": "b"})
	records.mu.Lock()
	defer records.mu.Unlock()
	allowed := map[string]bool{"msg": true, "level": true, "peer": true, "direction": true, "reason": true, "uptime": true,
		"kept": true, "keptDirection": true, "replacedDirection": true, "existingDirection": true, "rejectedDirection": true, "detail": true}
	for _, fields := range records.records {
		for key := range fields {
			if !allowed[key] {
				t.Errorf("record %q carries unexpected field %q", fields["msg"], key)
			}
		}
		if strings.Contains(fmt.Sprint(fields), reverseTestID) {
			t.Errorf("record %q carries the session ID", fields["msg"])
		}
	}
}

func TestReverseOpenFailuresNameTheirReason(t *testing.T) {
	t.Run("no reverse session", func(t *testing.T) {
		a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
		_, err := a.Open(context.Background(), ReversePeer{"nobody", "v1"}, noderec.ServiceNodeInfo)
		requireOpenReason(t, err, ErrReverseNoSession, ReverseRefusedNoSession)
	})
	t.Run("remote service refused", func(t *testing.T) {
		a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
		b := reverseTestManager(t, ReverseOptions{LocalUUID: "b"}) // resolves no service
		reverseTestConnect(t, a, b, "a", reverseTestID)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_, err := a.Open(ctx, ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo)
		requireOpenReason(t, err, ErrReverseRemoteRefused, ReverseRefusedRemote)
	})
	t.Run("stream capacity", func(t *testing.T) {
		a := reverseTestManager(t, ReverseOptions{LocalUUID: "a", PerSessionStreams: 1})
		left, right := net.Pipe()
		cfg := yamux.DefaultConfig()
		cfg.LogOutput = io.Discard
		raw, err := yamux.Server(right, cfg)
		if err != nil {
			t.Fatalf("raw server: %v", err)
		}
		defer raw.Close()
		if err := a.Register(ReversePeer{"b", "v1"}, left, "a", reverseTestID); err != nil {
			t.Fatalf("register: %v", err)
		}
		reverseTestRawControl(t, raw, false)
		first, cancelFirst := context.WithCancel(context.Background())
		defer cancelFirst()
		go func() {
			if conn, err := a.Open(first, ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo); err == nil {
				conn.Close()
			}
		}()
		// The first request holds the only stream slot while it waits for an
		// acknowledgment the raw peer never sends.
		stream, err := raw.AcceptStream()
		if err != nil {
			t.Fatalf("accept the held stream: %v", err)
		}
		defer abortReverseStream(stream)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_, err = a.Open(ctx, ReversePeer{"b", "v1"}, noderec.ServiceNodeInfo)
		requireOpenReason(t, err, ErrReverseCapacity, ReverseRefusedCapacity)
	})
	t.Run("admission failed at registration", func(t *testing.T) {
		a := reverseTestManager(t, ReverseOptions{LocalUUID: "a"})
		left, right := net.Pipe()
		defer left.Close()
		err := a.Register(ReversePeer{"b", "stale"}, right, "b", reverseTestID)
		requireOpenReason(t, err, ErrReverseAdmission, ReverseRefusedAdmission)
	})
	t.Run("other failure", func(t *testing.T) {
		if got := ReverseRefusalFor(errors.New("stream reset")); got != ReverseRefusedFailed {
			t.Fatalf("unclassified failure = %d, want %d", got, ReverseRefusedFailed)
		}
	})
}

func requireOpenReason(t *testing.T, err error, cause error, code ReverseRefusal) {
	t.Helper()
	var unavailable *ReverseUnavailableError
	if !errors.As(err, &unavailable) || unavailable.Route != RouteReverse {
		t.Fatalf("error = %v, want a reverse-route ReverseUnavailableError", err)
	}
	if !errors.Is(err, cause) {
		t.Fatalf("error = %v, want cause %v", err, cause)
	}
	if got := ReverseRefusalFor(err); got != code {
		t.Fatalf("refusal code = %d, want %d", got, code)
	}
}

func TestPeerServiceClientErrorNamesTheFailedRoute(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { writeServiceResponse(t, w, "ok") })

	t.Run("direct after the connection was established", func(t *testing.T) {
		f := newServiceClientFixture(t, http.HandlerFunc(func(http.ResponseWriter, *http.Request) { panic(http.ErrAbortHandler) }))
		_, err := f.adapter.Client("peer", noderec.ServiceCluster).Get(f.url)
		var unavailable *ReverseUnavailableError
		if !errors.As(err, &unavailable) || unavailable.Route != RouteDirect || unavailable.PeerUUID != "peer" {
			t.Fatalf("error = %v, want route %q", err, RouteDirect)
		}
		if !strings.Contains(unavailable.Error(), "(direct)") || errors.Unwrap(unavailable) == nil {
			t.Fatalf("message %q does not name the direct route with its cause", unavailable.Error())
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		if len(f.reverseCalls) != 0 {
			t.Fatalf("a failed TLS exchange was retried over the reverse session: %v", f.reverseCalls)
		}
	})
	t.Run("reverse only", func(t *testing.T) {
		f := newServiceClientFixture(t, handler)
		directErr, reverseErr := errors.New("direct blocked"), errors.New("reverse session lost")
		f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, directErr }
		client := f.adapter.Client("peer", noderec.ServiceCluster)
		if got := serviceClientGet(t, client, f.url); got != "ok" {
			t.Fatalf("fallback body = %q", got)
		}
		// The failed direct attempt is cached, so the next request uses only the
		// reverse session.
		f.mu.Lock()
		f.reverse = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, reverseErr }
		f.mu.Unlock()
		_, err := client.Get(f.url)
		var unavailable *ReverseUnavailableError
		if !errors.As(err, &unavailable) || unavailable.Route != RouteReverse || !errors.Is(err, reverseErr) {
			t.Fatalf("error = %v, want route %q with %v", err, RouteReverse, reverseErr)
		}
		if errors.Is(err, directErr) {
			t.Fatalf("reverse-only failure reports the earlier direct error: %v", err)
		}
	})
	t.Run("direct then reverse", func(t *testing.T) {
		f := newServiceClientFixture(t, handler)
		directErr, reverseErr := errors.New("direct blocked"), errors.New("reverse session lost")
		f.direct = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, directErr }
		f.reverse = func(context.Context, noderec.ServiceKey) (net.Conn, error) { return nil, reverseErr }
		_, err := f.adapter.Client("peer", noderec.ServiceCluster).Get(f.url)
		var unavailable *ReverseUnavailableError
		if !errors.As(err, &unavailable) || unavailable.Route != RouteDirectThenReverse ||
			!errors.Is(err, directErr) || !errors.Is(err, reverseErr) {
			t.Fatalf("error = %v, want route %q with both causes", err, RouteDirectThenReverse)
		}
		for _, want := range []string{"(direct, then reverse)", "direct: direct blocked", "reverse: reverse session lost"} {
			if !strings.Contains(unavailable.Error(), want) {
				t.Fatalf("message %q lacks %q", unavailable.Error(), want)
			}
		}
		if strings.Count(unavailable.Error(), "unavailable") != 1 {
			t.Fatalf("message %q wraps the failure more than once", unavailable.Error())
		}
	})
	t.Run("no route attempted", func(t *testing.T) {
		f := newServiceClientFixture(t, handler)
		f.mu.Lock()
		delete(f.routes, noderec.ServiceCluster)
		f.mu.Unlock()
		_, err := f.adapter.Client("peer", noderec.ServiceCluster).Get(f.url)
		var unavailable *ReverseUnavailableError
		if !errors.As(err, &unavailable) || unavailable.Route != "" || strings.Contains(unavailable.Error(), "(") {
			t.Fatalf("error = %v, want no route named", err)
		}
	})
}

func TestReverseControlFailureNamesItsReason(t *testing.T) {
	test := func(name string, frame []byte, wantReason string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			logger, records := newRecordedLog()
			_, raw, control := reverseTestRawWithControl(t, ReverseOptions{SetupTimeout: 120 * time.Millisecond, Log: logger})
			if _, err := control.Write(frame); err != nil && !errors.Is(err, yamux.ErrSessionShutdown) {
				t.Fatalf("write control frame: %v", err)
			}
			select {
			case <-raw.CloseChan():
			case <-time.After(time.Second):
				t.Fatal("control frame did not close the session")
			}
			closed := records.await(t, "reverse session closed", map[string]string{"peer": "a", "direction": "inbound"})
			if !strings.HasPrefix(closed["reason"], wantReason) {
				t.Fatalf("close reason = %q, want prefix %q", closed["reason"], wantReason)
			}
		})
	}
	test("cancellation of the control stream", []byte{0, 0, 0, 1}, "peer sent an invalid stream cancellation")
	test("cancellation of a stream the peer does not own", []byte{0, 0, 0, 2}, "peer sent an invalid stream cancellation")
	test("partial frame", []byte{0}, "control stream read failed: ")
}
