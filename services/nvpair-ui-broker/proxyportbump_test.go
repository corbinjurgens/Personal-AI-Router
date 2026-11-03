// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Coverage for automatic facade preparation, bind recovery, and ownership gates.

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"
)

// observeErrors attaches an errors process and returns a channel of the
// notifications the broker pushes into it.
func observeErrors(t *testing.T, b *Broker) <-chan *Message {
	t.Helper()
	client, server := net.Pipe()
	t.Cleanup(func() {
		_ = client.Close()
		_ = server.Close()
	})
	b.setErrors(&errorsProcess{peer: NewPeer(NewCodec(client))})

	seen := make(chan *Message, 4)
	go func() {
		codec := NewCodec(server)
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			select {
			case seen <- msg:
			default:
			}
		}
	}()
	return seen
}

func awaitErrorsNotification(t *testing.T, seen <-chan *Message, method string) *Message {
	t.Helper()
	deadline := time.After(2 * time.Second)
	for {
		select {
		case msg := <-seen:
			if msg != nil && msg.Method == method {
				return msg
			}
		case <-deadline:
			t.Fatalf("no %s notification arrived", method)
			return nil
		}
	}
}

// serveFacadeEnable answers facade/enable frames on a pipe, recording the port
// each attempt asked for and rejecting the ones in reject as bind races.
//
// It replies with codeFacadeBindFailed rather than a generic error because that
// code is the entire basis on which the broker decides a retry is worthwhile.
func serveFacadeEnable(t *testing.T, conn net.Conn, reject map[int]bool, attempts chan<- enableFacadeRequest) {
	t.Helper()
	go func() {
		codec := NewCodec(conn)
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			if msg.Method != "facade/enable" {
				continue
			}
			var spec enableFacadeRequest
			if err := json.Unmarshal(msg.Params, &spec); err != nil {
				return
			}
			attempts <- spec
			if reject[spec.Port] {
				_ = codec.RespondError(msg.ID, codeFacadeBindFailed,
					fmt.Sprintf("facade bind failed: port %d: address already in use", spec.Port))
				continue
			}
			_ = codec.Respond(msg.ID, enableFacadeReply{Engine: spec.Engine, Port: spec.Port})
		}
	}()
}

// A facade that loses a bind race is retried on a fallback port inside the same
// process. It used to be a process exit: the child log.Fatalf'd and the
// supervisor respawned it on a corrected port. That is only acceptable while a
// process hosts one engine — once it hosts several, one engine losing a race
// would take working listeners down with it.
func TestFacadeBindRaceRetriesInProcess(t *testing.T) {
	proxyClient, proxyServer := net.Pipe()
	t.Cleanup(func() {
		_ = proxyClient.Close()
		_ = proxyServer.Close()
	})
	proxy := &proxyProcess{peer: NewPeer(NewCodec(proxyClient))}
	go proxy.peer.Serve(nil, nil)

	attempts := make(chan enableFacadeRequest, 4)
	serveFacadeEnable(t, proxyServer, map[int]bool{24080: true}, attempts)

	b := &Broker{}

	const fallback = 12000
	err := b.enableProxyFacadeWithFallback(context.Background(), proxy,
		enableFacadeRequest{
			Engine:              mustEngineProxyProfile("llamacpp").Name,
			Port:                24080,
			IgnorePersistedPort: true,
		},
		func(failed int) int {
			if failed != 24080 {
				t.Errorf("fallback asked for port %d, want the port that failed (%d)", failed, 24080)
			}
			return fallback
		})
	if err != nil {
		t.Fatalf("enable with fallback = %v, want success on the fallback port", err)
	}

	first := <-attempts
	if first.Port != 24080 {
		t.Fatalf("first attempt port = %d, want %d", first.Port, 24080)
	}
	second := <-attempts
	if second.Port != fallback {
		t.Fatalf("retry port = %d, want the fallback %d", second.Port, fallback)
	}
	// The persisted port is the one that just failed to bind, so a retry that
	// let the child restore it would land straight back on the taken port.
	if !second.IgnorePersistedPort {
		t.Error("retry did not set ignorePersistedPort, so the child could restore the port that just failed")
	}
	if second.Engine != first.Engine {
		t.Errorf("retry engine = %q, want %q", second.Engine, first.Engine)
	}
}

// A rejection no other port would fix must not be retried. Retrying an
// unknown-engine enable would burn a port change and hide
// the real cause behind a second identical failure.
func TestFacadeEnableRejectionIsNotRetried(t *testing.T) {
	proxyClient, proxyServer := net.Pipe()
	t.Cleanup(func() {
		_ = proxyClient.Close()
		_ = proxyServer.Close()
	})
	proxy := &proxyProcess{peer: NewPeer(NewCodec(proxyClient))}
	go proxy.peer.Serve(nil, nil)

	attempts := make(chan enableFacadeRequest, 4)
	go func() {
		codec := NewCodec(proxyServer)
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			var spec enableFacadeRequest
			if json.Unmarshal(msg.Params, &spec) != nil {
				return
			}
			attempts <- spec
			// -32602 is what an unknown engine returns.
			_ = codec.RespondError(msg.ID, -32602, "unknown engine \"nope\"")
		}
	}()

	b := &Broker{}
	fallbackCalls := 0
	err := b.enableProxyFacadeWithFallback(context.Background(), proxy,
		enableFacadeRequest{Engine: "nope", Port: 11434},
		func(int) int {
			fallbackCalls++
			return 12000
		})
	if err == nil {
		t.Fatal("enable with fallback = nil, want the rejection surfaced")
	}
	if fallbackCalls != 0 {
		t.Errorf("fallback consulted %d times, want 0: only a bind race is retryable", fallbackCalls)
	}
	<-attempts
	select {
	case extra := <-attempts:
		t.Fatalf("a second enable was sent for %+v; a rejection must not be retried", extra)
	case <-time.After(200 * time.Millisecond):
	}
}

// The two ways an engine ends up without a proxy need different treatment. An
// unresolved binary is a packaging or --proxy-path problem the user should see;
// an engine deliberately left out of --proxy-engines is not, and reporting it
// would be noise for something they asked for.
//
// Gating preparation on proxyEnabled made this worth pinning: the disabled
// branches used to reach the blocked-ownership report through managedFacade,
// which is now necessarily false there, so the report had to move.
func TestUnavailableProxyIsReportedButDeselectionIsNot(t *testing.T) {
	reason, report := proxyDisabledReason("")
	if !report {
		t.Error("an unresolved proxy binary is not reported; a packaging problem would be log-only")
	}
	if !strings.Contains(reason, "not resolved") {
		t.Errorf("reason for a missing binary = %q", reason)
	}

	reason, report = proxyDisabledReason("/path/to/nvpair-proxy")
	if report {
		t.Error("a deliberately deselected engine is reported as an error; that is noise for a requested state")
	}
	if !strings.Contains(reason, "--proxy-engines") {
		t.Errorf("reason for a deselected engine = %q", reason)
	}
}
