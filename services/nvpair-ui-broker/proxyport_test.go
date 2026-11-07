// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net"
	"path/filepath"
	"testing"
	"time"
)

func TestEnabledEngineRestoreWaitsForFacadeStartup(t *testing.T) {
	worker, codec := newTestRPCWorkerPipe(t)
	broker := &Broker{
		clusterDir:   filepath.Join(t.TempDir(), "cluster"),
		facadesReady: make(chan struct{}),
	}
	broker.setEngineMgr(worker)
	method := recordEnabledEngineRestoration(t, codec)
	done := make(chan bool, 1)
	go func() { done <- broker.runEngineAvailability(context.Background()) }()

	select {
	case got := <-method:
		t.Fatalf("restore %q was sent before facade startup", got)
	case <-time.After(100 * time.Millisecond):
	}
	close(broker.facadesReady)
	for _, want := range []string{engineNodeProjectionMethod, "engine:settings-projection", restoreInstalledEnginesMethod} {
		select {
		case got := <-method:
			if got != want {
				t.Fatalf("restoration method = %q, want %q", got, want)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("%s was not sent after facade startup", want)
		}
	}
	select {
	case completed := <-done:
		if !completed {
			t.Fatal("restore wait reported cancellation")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("enabled-engine restoration did not return")
	}
}

func TestNextAvailablePortExcludingCustomBackend(t *testing.T) {
	available := func(port int) bool { return port == 11435 || port == 11436 }
	if got := nextAvailablePortExcluding(11435, []int{11435}, available); got != 11436 {
		t.Fatalf("fallback = %d, want 11436 (11435 is the configured backend)", got)
	}
}

func TestEnginePortAssignmentRequest(t *testing.T) {
	engine, port, ok := enginePortAssignmentRequest("engine:set-port", []byte(`{"engine":"llamacpp","port":8081}`))
	if !ok || engine != "llamacpp" || port != 8081 {
		t.Fatalf("valid request = (%q, %d, %v), want (llamacpp, 8081, true)", engine, port, ok)
	}
	for _, tc := range []struct {
		method string
		params string
	}{
		{"engine:status", `{"engine":"llamacpp","port":8081}`},
		{"engine:set-port", `{"engine":"","port":11434}`},
		{"engine:set-port", `{"engine":"llamacpp","port":0}`},
		{"engine:install", `{"engine":"llamacpp","port":8082}`},
		{"engine:restart", `{"engine":"llamacpp","port":8082}`},
		{"engine:set-port", `{`},
	} {
		if _, _, ok := enginePortAssignmentRequest(tc.method, []byte(tc.params)); ok {
			t.Fatalf("unexpected engine set-port match for %s %s", tc.method, tc.params)
		}
	}
}

func TestBrokerDoesNotExposeInternalReservationSetter(t *testing.T) {
	brokerConn, clientConn := net.Pipe()
	defer brokerConn.Close()
	defer clientConn.Close()
	b := &Broker{codec: NewCodec(brokerConn)}
	id := json.RawMessage(`10`)
	response := make(chan *Message, 1)
	go func() {
		msg, _ := NewCodec(clientConn).Read()
		response <- msg
	}()

	b.handleMessage(&Message{
		JSONRPC: "2.0",
		ID:      &id,
		Method:  "engine:set-reserved-port",
		Params:  json.RawMessage(`{"port":0}`),
	})

	select {
	case msg := <-response:
		if msg == nil || msg.Error == nil || msg.Error.Code != -32601 {
			t.Fatalf("response = %+v, want method-not-found", msg)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for private-method rejection")
	}
}

func TestNextFreeProxyPort(t *testing.T) {
	for _, tc := range []struct {
		name  string
		want  int
		req   int
		taken []int
	}{
		{"free returns requested", 11435, 11435, nil},
		{"free with others taken", 11435, 11435, []int{11434, 1234}},
		{"single collision bumps by one", 11435, 11434, []int{11434}},
		{"consecutive collisions skip", 11436, 11434, []int{11434, 11435}},
		{"gap above collision", 11435, 11434, []int{11434, 11436}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			taken := map[int]bool{}
			for _, p := range tc.taken {
				taken[p] = true
			}
			if got := nextFreeProxyPort(tc.req, taken); got != tc.want {
				t.Errorf("nextFreeProxyPort(%d, %v) = %d, want %d", tc.req, tc.taken, got, tc.want)
			}
		})
	}
}
