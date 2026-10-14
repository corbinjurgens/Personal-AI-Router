// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net"
	"reflect"
	"sync"
	"testing"
	"time"

	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

func observeLocalServices(t *testing.T, b *Broker) <-chan localServicesSnapshot {
	t.Helper()
	client, server := net.Pipe()
	cm := &clusterManagerProcess{stdin: client, peer: NewPeer(NewCodec(client))}
	seen := make(chan localServicesSnapshot, 256)
	t.Cleanup(func() {
		cm.peer.Close()
		_ = client.Close()
		_ = server.Close()
	})
	go func() {
		codec := NewCodec(server)
		for {
			msg, err := codec.Read()
			if err != nil {
				return
			}
			if msg.Method != "cluster:set-local-services" || !msg.IsNotification() {
				t.Errorf("unexpected parent frame: %+v", msg)
				return
			}
			var snapshot localServicesSnapshot
			if err := json.Unmarshal(msg.Params, &snapshot); err != nil {
				t.Errorf("decode local service snapshot: %v", err)
				return
			}
			if snapshot.Services == nil {
				t.Error("services must be an array, including the empty snapshot")
				return
			}
			seen <- snapshot
		}
	}()
	b.setClusterMgr(cm)
	return seen
}

func awaitLocalServices(t *testing.T, seen <-chan localServicesSnapshot, generation uint64) localServicesSnapshot {
	t.Helper()
	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()
	for {
		select {
		case snapshot := <-seen:
			if snapshot.Generation == generation {
				return snapshot
			}
			if snapshot.Generation > generation {
				t.Fatalf("snapshot generation %d passed expected %d", snapshot.Generation, generation)
			}
		case <-timer.C:
			t.Fatalf("no local service snapshot for generation %d", generation)
		}
	}
}

func TestLocalServicesIncludesEmptyBaselineAndRemoval(t *testing.T) {
	b := &Broker{regCache: relay.NewRegistrationCache()}
	seen := observeLocalServices(t, b)
	if got := awaitLocalServices(t, seen, 0); len(got.Services) != 0 {
		t.Fatalf("initial services = %+v, want empty", got.Services)
	}
	b.registerService(noderec.RegisterParams{Service: noderec.ServiceOllama, Port: 44001})
	if got := awaitLocalServices(t, seen, 1); !reflect.DeepEqual(got.Services,
		[]noderec.RegisterParams{{Service: noderec.ServiceOllama, Port: 44001}}) {
		t.Fatalf("registered snapshot = %+v, want ol on 44001", got)
	}
	b.registerService(noderec.RegisterParams{Service: noderec.ServiceOllama, Port: 44001})
	b.unregisterService(noderec.ServiceLMStudio)
	b.unregisterService(noderec.ServiceOllama)
	if got := awaitLocalServices(t, seen, 2); len(got.Services) != 0 {
		t.Fatalf("removed services = %+v, want empty", got.Services)
	}
}

func TestLocalServicesConcurrentChangesConvergeInGenerationOrder(t *testing.T) {
	b := &Broker{regCache: relay.NewRegistrationCache()}
	seen := observeLocalServices(t, b)
	awaitLocalServices(t, seen, 0)
	var writers sync.WaitGroup
	for _, service := range []noderec.ServiceKey{noderec.ServiceOllama, noderec.ServiceLMStudio, noderec.ServiceLlamaCPP} {
		writers.Add(1)
		go func(service noderec.ServiceKey) {
			defer writers.Done()
			for port := 44000; port < 44020; port++ {
				b.registerService(noderec.RegisterParams{Service: service, Port: port})
				b.unregisterService(service)
			}
		}(service)
	}
	writers.Wait()
	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()
	var previous uint64
	for {
		select {
		case snapshot := <-seen:
			if snapshot.Generation <= previous {
				t.Fatalf("generation regressed from %d to %d", previous, snapshot.Generation)
			}
			previous = snapshot.Generation
			if snapshot.Generation == 120 {
				if len(snapshot.Services) != 0 {
					t.Fatalf("final snapshot = %+v, want all concurrent services removed", snapshot)
				}
				return
			}
		case <-timer.C:
			t.Fatalf("concurrent snapshot stopped at generation %d, want 120", previous)
		}
	}
}

type observedLocalServicesWrite struct {
	net.Conn
	entered chan struct{}
	exited  chan struct{}
	once    sync.Once
}

func (w *observedLocalServicesWrite) Write(p []byte) (int, error) {
	w.once.Do(func() { close(w.entered) })
	n, err := w.Conn.Write(p)
	close(w.exited)
	return n, err
}

func blockedLocalServicesChild(t *testing.T, b *Broker) *observedLocalServicesWrite {
	t.Helper()
	client, server := net.Pipe()
	w := &observedLocalServicesWrite{Conn: client, entered: make(chan struct{}), exited: make(chan struct{})}
	cm := &clusterManagerProcess{stdin: client, peer: NewPeer(NewCodec(w))}
	t.Cleanup(func() {
		cm.peer.Close()
		_ = client.Close()
		_ = server.Close()
	})
	b.setClusterMgr(cm)
	select {
	case <-w.entered:
	case <-time.After(2 * time.Second):
		t.Fatal("old child's baseline did not reach the blocked writer")
	}
	return w
}

func TestLocalServicesReplacementBypassesPendingOldChildWrite(t *testing.T) {
	b := &Broker{regCache: relay.NewRegistrationCache()}
	old := blockedLocalServicesChild(t, b)
	updated := make(chan struct{})
	go func() {
		b.registerService(noderec.RegisterParams{Service: noderec.ServiceOllama, Port: 44001})
		b.registerService(noderec.RegisterParams{Service: noderec.ServiceLMStudio, Port: 44002})
		b.unregisterService(noderec.ServiceOllama)
		close(updated)
	}()
	select {
	case <-updated:
	case <-time.After(time.Second):
		t.Fatal("registrations stalled behind old child's pending write")
	}
	seen := observeLocalServices(t, b)
	if got := awaitLocalServices(t, seen, 3); !reflect.DeepEqual(got.Services,
		[]noderec.RegisterParams{{Service: noderec.ServiceLMStudio, Port: 44002}}) {
		t.Fatalf("replacement baseline = %+v, want only lm on 44002", got)
	}
	select {
	case <-old.exited:
	case <-time.After(time.Second):
		t.Fatal("replacement did not release old child's pending write")
	}
	b.registerService(noderec.RegisterParams{Service: noderec.ServiceLMStudio, Port: 44003})
	if got := awaitLocalServices(t, seen, 4); got.Services[0].Port != 44003 {
		t.Fatalf("current child update = %+v, want lm on 44003", got)
	}
}

func TestLocalServicesShutdownReleasesPendingWrite(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	b := &Broker{regCache: relay.NewRegistrationCache(), serveCtx: ctx}
	blocked := blockedLocalServicesChild(t, b)
	cancel()
	select {
	case <-blocked.exited:
	case <-time.After(time.Second):
		t.Fatal("broker cancellation did not release pending metadata write")
	}
}

func TestLocalServicesWriteTimeoutClosesUnresponsiveChild(t *testing.T) {
	b := &Broker{regCache: relay.NewRegistrationCache()}
	blocked := blockedLocalServicesChild(t, b)
	select {
	case <-blocked.exited:
	case <-time.After(7 * time.Second):
		t.Fatal("unresponsive child retained its blocked metadata writer beyond the write budget")
	}
	select {
	case <-b.getClusterMgr().peer.Done():
	case <-time.After(time.Second):
		t.Fatal("write timeout did not close the unresponsive peer")
	}
}

func TestLocalServicesRespawnReplaysCurrentRegistrationsWithoutMutation(t *testing.T) {
	b := &Broker{regCache: relay.NewRegistrationCache()}
	b.registerService(noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 14318})
	b.registerService(noderec.RegisterParams{Service: noderec.ServiceCluster, Port: 14321})
	first := observeLocalServices(t, b)
	initial := awaitLocalServices(t, first, 2)
	want := []noderec.RegisterParams{
		{Service: noderec.ServiceCluster, Port: 14321},
		{Service: noderec.ServiceNodeInfo, Port: 14318},
	}
	if !reflect.DeepEqual(initial.Services, want) {
		t.Fatalf("first child baseline = %+v, want %+v", initial.Services, want)
	}
	b.setClusterMgr(nil)
	replacement := observeLocalServices(t, b)
	if got := awaitLocalServices(t, replacement, 2); !reflect.DeepEqual(got.Services, want) {
		t.Fatalf("respawn baseline = %+v, want unchanged registrations %+v", got, want)
	}
}

func TestLocalServicesClientRequestCannotReachChild(t *testing.T) {
	client, server := net.Pipe()
	t.Cleanup(func() {
		_ = client.Close()
		_ = server.Close()
	})
	b := &Broker{codec: NewCodec(client), regCache: relay.NewRegistrationCache()}
	seen := observeLocalServices(t, b)
	awaitLocalServices(t, seen, 0)
	id := json.RawMessage("17")
	done := make(chan struct{})
	go func() {
		b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: "cluster:set-local-services",
			Params: json.RawMessage(`{"generation":999,"services":[{"service":"ol","port":1}]}`)})
		close(done)
	}()
	if err := server.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatalf("set response deadline: %v", err)
	}
	response, err := NewCodec(server).Read()
	if err != nil {
		t.Fatalf("read private method rejection: %v", err)
	}
	if response.Error == nil || response.Error.Code != -32601 || response.ID == nil || string(*response.ID) != "17" {
		t.Fatalf("private method response = %+v, want method-not-found for request 17", response)
	}
	<-done
	b.handleMessage(&Message{JSONRPC: "2.0", Method: "cluster:set-local-services",
		Params: json.RawMessage(`{"generation":999,"services":[]}`)})
	select {
	case snapshot := <-seen:
		t.Fatalf("client metadata reached cluster-manager: %+v", snapshot)
	case <-time.After(50 * time.Millisecond):
	}
}
