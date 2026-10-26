// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// These are process proofs using real policy IPC and loopback HTTP. A known
// Content-Length retains one 50 MiB request representation per accepted stream:
// three fit in 192 MiB, four fit in 256 MiB, and four cannot fit in 192 MiB.
// The additional parser and transport allowances do not approach either gap.
func TestUnifiedDispatchProcess(t *testing.T) {
	t.Run("concrete and hybrid share the broker controlled lifecycle", i7HybridLifecycle)
	t.Run("active hybrid streams survive listener rebind", i7HybridRebind)
	t.Run("broker routing control orders nine before ten and fences committed regenerate", i7HybridCancellation)
	t.Run("disabled public hybrid selection sends nothing to the synthetic provider", i7HybridDisabledPublic)

	t.Run("request under storage pressure completes after owner release", func(t *testing.T) {
		f, _, _ := i7BuffersBroker(t, 192, 30)
		u := i7BuffersUpstream(t)
		egressProcessAddConcrete(t, f.call, "llamacpp-proxy:", u.port)
		body := i7BuffersBody(50 << 20)
		held := f.hold(t, u, body, 3)
		waitingSlot := u.slot(t, "waiting")
		waiting := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "waiting", body)
		// WroteHeaders fences the client send only. There is no process-visible
		// buffer-waiter entry event; this proves recovery after release, not that
		// this request had already entered the proxy's storage queue.
		held[0].slot.release()
		i7BuffersJoin(t, held[0].request, http.StatusOK)
		egressProcessBarrier(t, waitingSlot.entered, "request reaches fake engine after capacity release")
		waitingSlot.release()
		i7BuffersJoin(t, waiting, http.StatusOK)
	})

	t.Run("zero wait refuses aggregate pressure with busy 503", func(t *testing.T) {
		f, _, _ := i7BuffersBroker(t, 192, 0)
		u := i7BuffersUpstream(t)
		egressProcessAddConcrete(t, f.call, "llamacpp-proxy:", u.port)
		body := i7BuffersBody(50 << 20)
		f.hold(t, u, body, 3)
		slot := u.slot(t, "busy")
		request := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "busy", body)
		i7BuffersBusy(t, request, slot)
	})

	t.Run("live cap decrease retains existing leases and blocks new storage", func(t *testing.T) {
		f, _, _ := i7BuffersBroker(t, 256, 0)
		u := i7BuffersUpstream(t)
		egressProcessAddConcrete(t, f.call, "llamacpp-proxy:", u.port)
		held := f.hold(t, u, i7BuffersBody(50<<20), 4)
		// Preserve terminal-frame headroom here.
		f.set(t, 224, 0)
		for _, h := range held {
			select {
			case <-h.request.done:
				t.Fatal("live decrease ended an existing stream")
			case <-h.slot.cancelled:
				t.Fatal("live decrease cancelled an existing upstream")
			default:
			}
		}
		probe := u.slot(t, "over-new-cap")
		i7BuffersBusy(t, i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "over-new-cap", i7BuffersBody(50<<20)), probe)
		held[0].slot.release()
		i7BuffersJoin(t, held[0].request, http.StatusOK)
		recovered := u.slot(t, "below-new-cap")
		r := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "below-new-cap", i7BuffersBody(50<<20))
		egressProcessBarrier(t, recovered.entered, "new storage after old lease release")
		recovered.release()
		i7BuffersJoin(t, r, http.StatusOK)
		// Every lease predating the decrease must remain usable through a
		// successful terminal stream, rather than merely survive one snapshot.
		for _, h := range held {
			h.slot.release()
			i7BuffersCompletedStream(t, h.request)
		}
	})

	t.Run("live cap decrease below usage drains existing streams before admitting new storage", func(t *testing.T) {
		f, _, _ := i7BuffersBroker(t, 256, 0)
		u := i7BuffersUpstream(t)
		egressProcessAddConcrete(t, f.call, "llamacpp-proxy:", u.port)
		body := i7BuffersBody(50 << 20)
		preOutputSlot := u.slot(t, "pre-output-owner")
		u.mu.Lock()
		preOutputSlot.preOutput = true
		u.mu.Unlock()
		preOutputRequest := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "pre-output-owner", body)
		egressProcessBarrier(t, preOutputSlot.entered, "accepted storage owner held before first output")
		held := []i7BuffersHold{{preOutputSlot, preOutputRequest}}
		held = append(held, f.hold(t, u, body, 3)...)
		// The entered barriers fence admission of roughly 200 MiB of request
		// storage. set waits for the broker's new policy to become active.
		f.set(t, 192, 0)
		probe := u.slot(t, "above-cap")
		i7BuffersBusy(t, i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "above-cap", body), probe)

		// The first owner must initialize its stream frame reader and emit its
		// normal terminal frames while usage still exceeds the new cap.
		// One drained owner leaves roughly 150 MiB: a
		// fresh 50 MiB request still cannot fit in 192 MiB.
		held[0].slot.release()
		i7BuffersCompletedStream(t, held[0].request)
		probe = u.slot(t, "insufficient-drain")
		i7BuffersBusy(t, i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "insufficient-drain", body), probe)

		held[1].slot.release()
		i7BuffersCompletedStream(t, held[1].request)
		recovered := u.slot(t, "enough-drained")
		request := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "enough-drained", body)
		egressProcessBarrier(t, recovered.entered, "new storage after enough old owners drain")
		for _, h := range held[2:] {
			h.slot.release()
			i7BuffersCompletedStream(t, h.request)
		}
		recovered.release()
		i7BuffersCompletedStream(t, request)
	})

	t.Run("client cancelled under storage pressure cannot dispatch after release", func(t *testing.T) {
		f, _, _ := i7BuffersBroker(t, 192, 30)
		u := i7BuffersUpstream(t)
		egressProcessAddConcrete(t, f.call, "llamacpp-proxy:", u.port)
		body := i7BuffersBody(50 << 20)
		held := f.hold(t, u, body, 3)
		cancelledSlot := u.slot(t, "cancelled-waiter")
		cancelled := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "cancelled-waiter", body)
		// Cancellation is fenced by sent headers, not an unobservable storage
		// queue entry. Workload cancellation below has an actual queued event.
		cancelled.cancel()
		i7BuffersAwait(t, cancelled)
		if cancelled.result.err == nil {
			t.Fatal("cancelled waiting HTTP client returned without a context error")
		}
		held[0].slot.release()
		i7BuffersJoin(t, held[0].request, http.StatusOK)
		replacementSlot := u.slot(t, "replacement-waiter")
		replacement := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "replacement-waiter", body)
		egressProcessBarrier(t, replacementSlot.entered, "replacement after queued cancellation")
		replacementSlot.release()
		i7BuffersJoin(t, replacement, http.StatusOK)
		select {
		case <-cancelledSlot.entered:
			t.Fatal("cancelled request reached the engine after capacity returned")
		default:
		}
	})

	t.Run("replacement proxy enforces replayed persisted buffers", func(t *testing.T) {
		f, service, policyPath := i7BuffersBroker(t, 192, 0)
		stored := readPolicyBytes(t, policyPath)
		upstream := newEgressProcessUpstream(t)
		egressProcessAddConcrete(t, f.call, "llamacpp-proxy:", upstream.localPort)
		egressProcessAssertConcrete(t, f.client, f.port)
		var old serviceProcessIdentity
		for _, identity := range service.scan() {
			if identity.Path == service.proxyPath {
				if old.PID != 0 {
					t.Fatal("fixture owns more than one proxy")
				}
				old = identity
			}
		}
		if old.PID == 0 {
			t.Fatal("owned fixture proxy not found")
		}
		if err := service.inventory.kill(old); err != nil {
			t.Fatalf("restart owned proxy: %v", err)
		}
		deadline := time.Now().Add(20 * time.Second)
		for {
			replaced := false
			for _, identity := range service.scan() {
				if identity.Path == service.proxyPath && identity.PID != old.PID {
					replaced = true
				}
			}
			if replaced {
				break
			}
			if time.Now().After(deadline) {
				t.Fatal("fixture proxy was not replaced")
			}
			time.Sleep(25 * time.Millisecond)
		}
		// No policy:set is sent to the replacement. This proves replay and
		// enforcement after acknowledgement; it does not inject a request into
		// the pre-replay interval. Missing-binding refusal is component evidence
		// in proxy TestRoutingIdentityMissingBindingHasZeroEffects.
		view := egressProcessActivePolicy(t, f.call, f.revision)
		if view.Policy.Routing.Buffers != (nodepolicy.BufferLimits{CapacityMiB: 192, WaitSeconds: 0}) || !bytes.Equal(stored, readPolicyBytes(t, policyPath)) {
			t.Fatal("proxy restart changed the persisted buffer policy")
		}
		for {
			reply := f.call(t, "llamacpp-proxy:get-status", map[string]any{})
			egressProcessOK(t, reply)
			var status struct {
				Ready bool `json:"ready"`
				Port  int  `json:"port"`
			}
			egressProcessDecode(t, reply.Result, &status)
			if status.Ready && status.Port == f.port {
				break
			}
			if time.Now().After(deadline) {
				t.Fatal("replacement facade did not become ready on its persisted port")
			}
			time.Sleep(25 * time.Millisecond)
		}
		u := i7BuffersUpstream(t)
		egressProcessAddConcrete(t, f.call, "llamacpp-proxy:", u.port)
		body := i7BuffersBody(50 << 20)
		f.hold(t, u, body, 3)
		probe := u.slot(t, "replayed-busy")
		i7BuffersBusy(t, i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "replayed-busy", body), probe)
		upstream.assertNoPublic(t)
	})

	t.Run("broker cancels queued workload without ending a running stream", func(t *testing.T) {
		// startDepartedRunFixture fixes its own private config directory but
		// does not accept a facade port plan. Use the service harness's persisted
		// ephemeral facade instead, retaining the departed-run wire assertions.
		f, _, _ := i7BuffersBroker(t, 192, 0)
		u := i7BuffersUpstream(t)
		egressProcessAddConcrete(t, f.call, "llamacpp-proxy:", u.port)
		reply := f.call(t, "workloads:subscribe", map[string]any{})
		egressProcessOK(t, reply)
		var subscribed struct {
			Subscribed bool `json:"subscribed"`
		}
		egressProcessDecode(t, reply.Result, &subscribed)
		if !subscribed.Subscribed {
			t.Fatal("broker did not acknowledge workload subscription")
		}
		runningSlot := u.slot(t, "running-control")
		runningRequest := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "running-control", i7BuffersBody(1024))
		egressProcessBarrier(t, runningSlot.entered, "running cancellation control")
		running := i7BuffersBrokerWorkload(t, f.rpc, func(w departedRunInfo) bool { return w.State == "running" })
		queuedSlot := u.slot(t, "queued-control")
		u.mu.Lock()
		queuedSlot.preOutput = true
		u.mu.Unlock()
		queuedRequest := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "queued-control", i7BuffersBody(1024))
		egressProcessBarrier(t, queuedSlot.entered, "queued request reaches fake engine without output")
		queued := i7BuffersBrokerWorkload(t, f.rpc, func(w departedRunInfo) bool {
			return w.ID != running.ID && w.State == "queued" && w.ScheduledOn != ""
		})
		if queued.StartedAt != nil || queued.CompletedAt != nil || queued.RunID == "" || queued.RunID != running.RunID {
			t.Fatalf("request did not remain queued in the current run: %+v", queued)
		}
		cancel := func(w departedRunInfo, want bool) {
			t.Helper()
			assertDepartedRunCancelResult(t, f.call(t, "workloads:cancel", nodepolicy.WorkloadCancelParams{
				OriginatedFrom: w.OriginatedFrom, WorkloadID: w.ID, Engine: w.Engine, RunID: w.RunID,
			}), want)
		}
		cancel(queued, true)
		terminal := i7BuffersBrokerWorkload(t, f.rpc, func(w departedRunInfo) bool {
			return w.key() == queued.key() && w.State == "cancelled"
		})
		assertDepartedRunTerminal(t, terminal, queued, "cancelled", "cancelled by request")
		egressProcessBarrier(t, queuedSlot.cancelled, "queued upstream cancelled by broker")
		i7BuffersAwait(t, queuedRequest)
		if err := queuedRequest.ctx.Err(); err != nil {
			t.Fatalf("client context caused queued cancellation: %v", err)
		}
		select {
		case <-runningRequest.done:
			t.Fatal("queued cancellation ended the other stream")
		case <-runningSlot.cancelled:
			t.Fatal("queued cancellation cancelled the other upstream")
		default:
		}
		cancel(queued, false)
		cancel(running, true)
		i7BuffersAwait(t, runningRequest)
	})

	t.Run("broker stdin EOF emits terminals within shutdown grace", func(t *testing.T) {
		p := startEgressProcessProxy(t, t.TempDir())
		policy := nodepolicy.Default()
		policy.Revision = "17"
		p.activate(t, policy)
		port := p.enable(t)
		u := i7BuffersUpstream(t)
		egressProcessAddConcrete(t, p.call, "llamacpp:", u.port)
		client := egressProcessHTTPClient(t)
		client.Timeout = 0
		started := make(map[string]departedRunInfo)
		var streams []i7BuffersHold
		for _, id := range []string{"eof-first", "eof-second"} {
			slot := u.slot(t, id)
			request := i7BuffersStart(t, client, port, "/v1/chat/completions", id, i7BuffersBody(1024))
			egressProcessBarrier(t, slot.entered, "EOF stream reaches fake engine")
			workload := i7BuffersProxyWorkload(t, p, "workload:started", time.Now().Add(10*time.Second))
			if workload.State != "running" || workload.RunID == "" || workload.StartedAt == nil || workload.CompletedAt != nil || started[workload.ID].ID != "" {
				t.Fatalf("EOF stream did not start with a unique identity: %+v", workload)
			}
			started[workload.ID] = workload
			streams = append(streams, i7BuffersHold{slot, request})
		}
		for _, stream := range streams {
			select {
			case <-stream.request.done:
				t.Fatal("stream ended before parent stdin EOF")
			case <-stream.slot.cancelled:
				t.Fatal("upstream cancelled before parent stdin EOF")
			default:
			}
		}
		// Keep the HTTP client context and stdout reader alive. The existing EOF
		// driver closes only trusted-parent stdin; no workload cancel is sent.
		grace := time.Now().Add(3 * time.Second)
		if err := p.stdio.closeInput(); err != nil {
			t.Fatalf("close parent stdin: %v", err)
		}
		retired := make(map[string]bool)
		for range streams {
			terminal := i7BuffersProxyWorkload(t, p, "workload:errored", grace)
			prior, exists := started[terminal.ID]
			if !exists || retired[terminal.ID] || terminal.Error == nil || (*terminal.Error != "client disconnected before completion" && *terminal.Error != "request cancelled before completion") {
				t.Fatalf("unexpected or duplicate EOF terminal: %+v", terminal)
			}
			assertDepartedRunTerminal(t, terminal, prior, "cancelled", *terminal.Error)
			retired[terminal.ID] = true
		}
		for _, stream := range streams {
			egressProcessBarrier(t, stream.slot.cancelled, "upstream cancellation after parent EOF")
			i7BuffersAwait(t, stream.request)
			if err := stream.request.ctx.Err(); err != nil {
				t.Fatalf("client context caused EOF termination: %v", err)
			}
		}
		p.awaitExit(t)
		p.client.mu.Lock()
		defer p.client.mu.Unlock()
		for _, frame := range p.client.frames {
			if frame.Method == "workload:errored" || frame.Method == "workload:completed" {
				var params departedRunParams
				egressProcessDecode(t, frame.Params, &params)
				if prior, exists := started[params.WorkloadInfo.ID]; exists && params.WorkloadInfo.RunID == prior.RunID {
					t.Fatal("EOF emitted a duplicate terminal for the same workload")
				}
			}
		}
	})
}

func i7HybridIdentity() nodepolicy.StrictModelIdentity {
	return nodepolicy.StrictModelIdentity{EngineRunID: "i7-synthetic-engine", SettingsFingerprint: strings.Repeat("a", 64), ModelDigest: strings.Repeat("b", 64)}
}

// Publish through the existing private scanner fixture, not a product hook.
// The pinned loopback peer validates the synthetic strict envelope and rewritten
// model. It simulates qualified peer ingress, not native engine qualification.
func i7HybridFixture(t *testing.T, edits ...func(*nodepolicy.Policy)) (*i7BuffersFixture, *serviceFixture, *i7BuffersFake, int) {
	t.Helper()
	return i7HybridFixturePeers(t, []string{"i7-synthetic-node"}, edits...)
}

func i7HybridFixturePeers(t *testing.T, peerIDs []string, edits ...func(*nodepolicy.Policy)) (*i7BuffersFixture, *serviceFixture, *i7BuffersFake, int) {
	t.Helper()
	var u *i7BuffersFake
	f, service, _ := i7BuffersBroker(t, 256, 0, func(service *serviceFixture) {
		const origin = "i7-source-node"
		cert := mintClusterIdentity(t, service.clusterDir, origin)
		writeActiveAdmission(t, service.clusterDir)
		jointWriteJSON(t, filepath.Join(service.clusterDir, "identity.json"), map[string]any{"node_uuid": origin})
		members := append([]string{origin}, peerIDs...)
		writeMembers(t, service.clusterDir, members...)
		var peers []*clustertrust.Mesh
		for _, id := range peerIDs {
			dir := t.TempDir()
			peerCert := mintClusterIdentity(t, dir, id)
			writeActiveAdmission(t, dir)
			writePin(t, dir, origin, cert)
			writeMembers(t, dir, id, origin)
			writePin(t, service.clusterDir, id, peerCert)
			mesh := clustertrust.Open(dir)
			if _, ok := mesh.PeerServiceIncarnation(origin); !ok {
				t.Fatalf("fixture peer %q lacks current source admission", id)
			}
			peers = append(peers, mesh)
		}
		mesh := clustertrust.Open(service.clusterDir)
		for _, id := range peerIDs {
			if _, ok := mesh.PeerServiceIncarnation(id); !ok {
				t.Fatalf("fixture source lacks current peer admission for %q", id)
			}
		}
		u = i7BuffersUpstream(t, peers...)
	})
	listener := freePort(t)
	i7HybridPolicy(t, f, func(p *nodepolicy.Policy) {
		p.Tiers["weak"] = []nodepolicy.TierEntry{{Kind: nodepolicy.EntryLocal, Engine: "llamacpp", Model: egressProcessModel}}
		p.Routing.Listener = nodepolicy.Listener{Enabled: true, Port: listener}
		for _, edit := range edits {
			edit(p)
		}
	})
	advertisement := &nodepolicy.Advertisement{Revision: "1", Roles: map[string]nodepolicy.AdvertisedRole{
		"weak": {Engine: "llamacpp", Model: egressProcessModel, ContextTokens: 4096, MaxOutputTokens: 1024,
			Qualification: &nodepolicy.QualificationRef{ID: "i7-fixture-only", Revision: "1"}},
	}, StrictAdmission: &nodepolicy.StrictAdmissionCapability{Version: nodepolicy.StrictAdmissionVersion,
		Models: []nodepolicy.StrictAdmissionModel{{Engine: "llamacpp", Model: egressProcessModel, MaxConcurrent: 16, Identity: i7HybridIdentity()}}}}
	if err := advertisement.StrictAdmission.Validate(); err != nil {
		t.Fatal(err)
	}
	for i, id := range peerIDs {
		event := noderec.NodeEvent{Node: noderec.DirectoryNode{HostUUID: id, ClusterUUID: id, Trusted: true, Name: id, IP: "127.0.0.1",
			Services:       map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceLlamaCPP: {Port: u.ports[i]}},
			ModelsByEngine: map[string][]string{"llamacpp": {egressProcessModel}}, Policy: advertisement}}
		frame := jsonrpc.Message{JSONRPC: "2.0", Method: noderec.NotifyNodeDiscovered, Params: egressProcessJSON(t, event)}
		egressProcessHTTP(t, egressProcessHTTPClient(t), http.MethodPost, f.scannerPort, "/", egressProcessJSON(t, frame), http.StatusOK)
		i7HybridDestinationReady(t, f, id)
	}
	i7HybridListenerReady(t, f.client, listener)
	return f, service, u, listener
}

func i7HybridDestinationReady(t *testing.T, f *i7BuffersFixture, id string) {
	t.Helper()
	neEventually(t, "pinned destination forwarded by the scanner and broker", 10*time.Second, func() bool {
		reply := f.call(t, "llamacpp-proxy:nodes/list", map[string]any{})
		egressProcessOK(t, reply)
		var result struct {
			Nodes []struct {
				ID     string   `json:"id"`
				Models []string `json:"models"`
			} `json:"nodes"`
		}
		egressProcessDecode(t, reply.Result, &result)
		for _, node := range result.Nodes {
			if node.ID == id && slices.Contains(node.Models, egressProcessModel) {
				return true
			}
		}
		return false
	})
}

func i7HybridPolicy(t *testing.T, f *i7BuffersFixture, edit func(*nodepolicy.Policy)) {
	t.Helper()
	reply := f.call(t, "policy:get", map[string]any{})
	before := decodeServicePolicySnapshot(t, &reply)
	policy := before.Policy
	edit(&policy)
	if err := policy.Validate(); err != nil {
		t.Fatal(err)
	}
	egressProcessOK(t, f.call(t, "policy:set", map[string]any{"policy": policy, "expectedPersistedRevision": before.Policy.Revision}))
	reply = f.call(t, "policy:get", map[string]any{})
	stored := decodeServicePolicySnapshot(t, &reply)
	egressProcessActivePolicy(t, f.call, stored.Policy.Revision)
}

func i7HybridListenerReady(t *testing.T, client *http.Client, port int) {
	t.Helper()
	neEventually(t, "hybrid listener model endpoint ready", 10*time.Second, func() bool {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/v1/models", port), nil)
		if err != nil {
			t.Fatal(err)
		}
		response, err := client.Do(request)
		if err != nil {
			return false
		}
		_, readErr := io.Copy(io.Discard, response.Body)
		closeErr := response.Body.Close()
		return response.StatusCode == http.StatusOK && readErr == nil && closeErr == nil
	})
}

func i7HybridActive(t *testing.T, f *i7BuffersFixture) []nodepolicy.RoutingRequestKey {
	t.Helper()
	reply := f.call(t, "routing:active", nodepolicy.RoutingActiveParams{})
	egressProcessOK(t, reply)
	var active nodepolicy.RoutingActiveResult
	egressProcessDecode(t, reply.Result, &active)
	return active.Requests
}

func i7HybridActiveCount(t *testing.T, f *i7BuffersFixture, count int) []nodepolicy.RoutingRequestKey {
	t.Helper()
	var keys []nodepolicy.RoutingRequestKey
	neEventually(t, "routing registry reaches expected owner count", 10*time.Second, func() bool {
		keys = i7HybridActive(t, f)
		return len(keys) == count
	})
	return keys
}

func i7HybridCancel(t *testing.T, f *i7BuffersFixture, key nodepolicy.RoutingRequestKey, regenerate bool, want nodepolicy.RoutingCancelStatus) {
	t.Helper()
	reply := f.call(t, "routing:cancel", nodepolicy.RoutingCancelParams{Request: key, Regenerate: regenerate})
	egressProcessOK(t, reply)
	var result nodepolicy.RoutingCancelResult
	egressProcessDecode(t, reply.Result, &result)
	if result.Status != want {
		t.Fatalf("routing cancellation=%q, want %q", result.Status, want)
	}
}

func i7HybridSlot(t *testing.T, u *i7BuffersFake, id string) *i7BuffersSlot {
	t.Helper()
	slot := u.slot(t, id)
	u.mu.Lock()
	slot.hybrid = true
	u.mu.Unlock()
	return slot
}

func i7HybridBody() []byte {
	return []byte(`{"model":"weak-local","messages":[],"stream":true,"max_tokens":16}`)
}

func i7HybridAccepted(t *testing.T, slot *i7BuffersSlot, request *i7BuffersRequest) {
	t.Helper()
	select {
	case <-slot.entered:
	case <-request.done:
		t.Fatalf("hybrid fixture request ended before destination acceptance: status=%d error=%v", request.result.status, request.result.err)
	case <-time.After(10 * time.Second):
		t.Fatal("hybrid destination did not accept the request")
	}
}

func i7HybridLifecycle(t *testing.T) {
	f, _, u, listener := i7HybridFixture(t)
	egressProcessOK(t, f.call(t, "workloads:subscribe", map[string]any{}))
	var run nodepolicy.RoutingRequestKey
	for i, route := range []struct {
		name   string
		port   int
		body   []byte
		hybrid bool
	}{
		{"concrete facade", f.port, i7BuffersBody(1024), false},
		{"hybrid listener", listener, i7HybridBody(), true},
	} {
		t.Run(route.name, func(t *testing.T) {
			slot := u.slot(t, route.name)
			u.mu.Lock()
			slot.hybrid = route.hybrid
			u.mu.Unlock()
			request := i7BuffersStart(t, f.client, route.port, "/v1/chat/completions", route.name, route.body)
			i7HybridAccepted(t, slot, request)
			egressProcessBarrier(t, request.firstByte, "client observed committed stream bytes")
			keys := i7HybridActiveCount(t, f, 1)
			key := keys[0]
			if key.RequestID != strconv.Itoa(i+1) {
				t.Fatalf("request ID=%q, want sequential shared owner %d", key.RequestID, i+1)
			}
			if i == 0 {
				run = key
			} else if key.OriginNodeID != run.OriginNodeID || key.ProxyRunID != run.ProxyRunID || slot.strict.Request != key {
				t.Fatalf("hybrid/concrete identity or strict linkage differs: first=%+v current=%+v strict=%+v", run, key, slot.strict.Request)
			}
			started := i7BuffersBrokerWorkload(t, f.rpc, func(w departedRunInfo) bool { return w.State == "running" && w.ID == key.RequestID })
			if started.RunID != key.ProxyRunID || started.OriginatedFrom != key.OriginNodeID || started.Engine != "llamacpp" {
				t.Fatalf("workload and routing owner differ: workload=%+v routing=%+v", started, key)
			}
			slot.release()
			i7BuffersCompletedStream(t, request)
			i7BuffersBrokerWorkload(t, f.rpc, func(w departedRunInfo) bool { return w.key() == started.key() && w.State == "completed" })
			i7HybridActiveCount(t, f, 0)
		})
	}
}

func i7HybridProxyIdentity(t *testing.T, service *serviceFixture) serviceProcessIdentity {
	t.Helper()
	owned, err := service.inventory.scan()
	if err != nil {
		t.Fatal(err)
	}
	var proxies []serviceProcessIdentity
	for _, identity := range owned {
		if identity.Path == service.proxyPath {
			proxies = append(proxies, identity)
		}
	}
	if len(proxies) != 1 {
		t.Fatalf("proxy identities=%+v, want exactly one", proxies)
	}
	return proxies[0]
}

func i7HybridRebind(t *testing.T) {
	f, service, u, listener := i7HybridFixture(t)
	identity := i7HybridProxyIdentity(t, service)
	var held []i7BuffersHold
	var retired []int
	for i := range 3 {
		id := fmt.Sprintf("rebind-%d", i)
		slot := i7HybridSlot(t, u, id)
		request := i7BuffersStart(t, f.client, listener, "/v1/chat/completions", id, i7HybridBody())
		i7HybridAccepted(t, slot, request)
		egressProcessBarrier(t, request.firstByte, "hybrid stream committed before rebind")
		held = append(held, i7BuffersHold{slot, request})
		before := i7HybridActiveCount(t, f, len(held))
		retired = append(retired, listener)
		listener = freePort(t)
		i7HybridPolicy(t, f, func(p *nodepolicy.Policy) { p.Routing.Listener.Port = listener })
		i7HybridListenerReady(t, f.client, listener)
		for _, port := range retired {
			conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), time.Second)
			if err == nil {
				if closeErr := conn.Close(); closeErr != nil {
					t.Error(closeErr)
				}
				t.Fatalf("retired listener %d still accepts", port)
			}
			if !neConnectionRefused(err) {
				t.Fatalf("retired listener %d did not refuse a fresh connection: %v", port, err)
			}
		}
		if after := i7HybridActiveCount(t, f, len(held)); !slices.Equal(before, after) {
			t.Fatalf("rebind changed owners: before=%+v after=%+v", before, after)
		}
		if got := i7HybridProxyIdentity(t, service); got != identity {
			t.Fatalf("rebind replaced proxy: before=%+v after=%+v", identity, got)
		}
	}
	// The source owner reuses one http.Server. Black-box evidence here is one
	// unchanged proxy, one accepting routing port, and successful terminal output
	// on every connection accepted before the preceding rebinds.
	for _, h := range held {
		h.slot.release()
		i7BuffersCompletedStream(t, h.request)
	}
	i7HybridActiveCount(t, f, 0)
}

func i7HybridCancellation(t *testing.T) {
	// Ten committed owners prove numeric ordering of nine before ten. Spread
	// them across pinned peers so the per-peer transport slot cap still applies.
	f, _, u, listener := i7HybridFixturePeers(t, []string{"i7-synthetic-node", "i7-second-node"})
	var held []i7BuffersHold
	for i := range 10 {
		id := fmt.Sprintf("control-%d", i+1)
		slot := i7HybridSlot(t, u, id)
		request := i7BuffersStart(t, f.client, listener, "/v1/chat/completions", id, i7HybridBody())
		i7HybridAccepted(t, slot, request)
		egressProcessBarrier(t, request.firstByte, "routing cancellation stream committed")
		held = append(held, i7BuffersHold{slot, request})
	}
	keys := i7HybridActiveCount(t, f, len(held))
	for i, key := range keys {
		if key.RequestID != strconv.Itoa(i+1) || key.OriginNodeID != keys[0].OriginNodeID || key.ProxyRunID != keys[0].ProxyRunID {
			t.Fatalf("routing active order/identity at %d: %+v", i, keys)
		}
		if held[i].slot.strict.Request != key {
			t.Fatalf("pinned destination strict request=%+v, active owner=%+v", held[i].slot.strict.Request, key)
		}
	}
	i7HybridCancel(t, f, keys[8], true, nodepolicy.RoutingCancelConflictCommitted)
	if after := i7HybridActive(t, f); !slices.Equal(keys, after) {
		t.Fatalf("committed regenerate changed active owners: before=%+v after=%+v", keys, after)
	}
	i7HybridCancel(t, f, keys[9], false, nodepolicy.RoutingCancelCancelled)
	egressProcessBarrier(t, held[9].slot.cancelled, "broker cancellation reached the live destination")
	i7BuffersAwait(t, held[9].request)
	remaining := i7HybridActiveCount(t, f, 9)
	if !slices.Equal(remaining, keys[:9]) {
		t.Fatalf("cancellation removed a different owner: %+v", remaining)
	}
	i7HybridCancel(t, f, keys[9], false, nodepolicy.RoutingCancelNotFound)
	// Completing nine, after cancellation of ten, positively fences the conflict's
	// lack of cancellation rather than treating a quiet interval as evidence.
	for _, h := range held[:9] {
		h.slot.release()
		i7BuffersCompletedStream(t, h.request)
	}
	i7HybridActiveCount(t, f, 0)
	pre := i7HybridSlot(t, u, "pre-output-cancel")
	u.mu.Lock()
	pre.preOutput = true
	u.mu.Unlock()
	request := i7BuffersStart(t, f.client, listener, "/v1/chat/completions", "pre-output-cancel", i7HybridBody())
	i7HybridAccepted(t, pre, request)
	key := i7HybridActiveCount(t, f, 1)[0]
	i7HybridCancel(t, f, key, false, nodepolicy.RoutingCancelCancelled)
	egressProcessBarrier(t, pre.cancelled, "pre-output cancellation reached destination")
	i7BuffersAwait(t, request)
	i7HybridActiveCount(t, f, 0)
}

func i7HybridDisabledPublic(t *testing.T) {
	public := newEgressProcessUpstream(t)
	staged := egressProcessPolicy(t, "1", public.publicPort)
	// A changed credential-reference set intentionally restarts the proxy.
	// Configure it before installing the fixture's ephemeral manual overlay.
	f, _, u, listener := i7HybridFixture(t, func(p *nodepolicy.Policy) {
		p.Tiers["weak"] = staged.Tiers["weak"]
		p.Routing.Accounts, p.Routing.Deployments = staged.Routing.Accounts, staged.Routing.Deployments
	})
	egressProcessHTTP(t, egressProcessHTTPClient(t), http.MethodPost, listener, "/v1/chat/completions", []byte(`{"model":"weak-api","messages":[],"stream":false}`), http.StatusNotFound)
	i7HybridActiveCount(t, f, 0)
	public.assertNoPublic(t)
	// Positive local completion after the refusal shows that an API-only miss
	// neither fell through locally nor disabled subsequent hybrid dispatch.
	slot := i7HybridSlot(t, u, "after-disabled-public")
	request := i7BuffersStart(t, f.client, listener, "/v1/chat/completions", "after-disabled-public", i7HybridBody())
	i7HybridAccepted(t, slot, request)
	slot.release()
	i7BuffersCompletedStream(t, request)
	public.assertNoPublic(t)
	i7HybridActiveCount(t, f, 0)
}

type i7BuffersFixture struct {
	call        egressProcessCall
	rpc         *serviceRPCClient
	client      *http.Client
	port        int
	scannerPort int
	revision    string
}

func i7BuffersBroker(t *testing.T, capacity, wait int, setup ...func(*serviceFixture)) (*i7BuffersFixture, *serviceFixture, string) {
	t.Helper()
	f := newProxyServiceFixture(t)
	self, err := os.Executable()
	if err != nil {
		t.Fatalf("locate registered empty scanner fixture: %v", err)
	}
	jointCopyExecutable(t, self, f.scannerPath)
	scannerPort := freePort(t)
	f.env = append(f.env, "NVPAIR_TEST_SUBSCRIPTION_SCANNER="+f.scannerPath,
		"NVPAIR_TEST_SUBSCRIPTION_PORT="+strconv.Itoa(scannerPort), "USERPROFILE="+f.configDir, "OLLAMA_HOST=")
	for _, prepare := range setup {
		prepare(f)
	}
	f.start()
	c := f.readyClient()
	nextID := 7000
	fixture := &i7BuffersFixture{rpc: c, client: egressProcessHTTPClient(t), port: f.facadePort, scannerPort: scannerPort}
	fixture.client.Timeout = 0 // Each HTTP request has its own bounded context.
	fixture.call = func(t *testing.T, method string, params any) jsonrpc.Message {
		t.Helper()
		nextID++
		reply, err := c.call(strconv.Itoa(nextID), method, egressProcessJSON(t, params), serviceCallTimeout)
		if err != nil {
			t.Fatalf("broker %s: %v", method, err)
		}
		return *reply
	}
	fixture.set(t, capacity, wait)
	return fixture, f, f.policyPath()
}

func (f *i7BuffersFixture) set(t *testing.T, capacity, wait int) {
	t.Helper()
	reply := f.call(t, "policy:get", map[string]any{})
	before := decodeServicePolicySnapshot(t, &reply)
	policy := before.Policy
	policy.Routing.Buffers = nodepolicy.BufferLimits{CapacityMiB: capacity, WaitSeconds: wait}
	egressProcessOK(t, f.call(t, "policy:set", map[string]any{"policy": policy, "expectedPersistedRevision": before.Policy.Revision}))
	reply = f.call(t, "policy:get", map[string]any{})
	stored := decodeServicePolicySnapshot(t, &reply)
	if stored.Policy.Revision == before.Policy.Revision {
		t.Fatal("buffer edit did not advance persisted revision")
	}
	f.revision = stored.Policy.Revision
	active := egressProcessActivePolicy(t, f.call, f.revision)
	if active.Policy.Routing.Buffers != policy.Routing.Buffers {
		t.Fatalf("activated buffers = %+v, want %+v", active.Policy.Routing.Buffers, policy.Routing.Buffers)
	}
}

type i7BuffersHold struct {
	slot    *i7BuffersSlot
	request *i7BuffersRequest
}

func (f *i7BuffersFixture) hold(t *testing.T, u *i7BuffersFake, body []byte, count int) []i7BuffersHold {
	t.Helper()
	held := make([]i7BuffersHold, 0, count)
	for i := range count {
		id := fmt.Sprintf("holder-%d", i)
		slot := u.slot(t, id)
		request := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", id, body)
		egressProcessBarrier(t, slot.entered, "accepted storage holder "+id)
		held = append(held, i7BuffersHold{slot, request})
	}
	return held
}

type i7BuffersSlot struct {
	entered, cancelled, released chan struct{}
	once                         sync.Once
	preOutput                    bool
	hybrid                       bool
	strict                       nodepolicy.StrictAdmissionRequest
}

func (s *i7BuffersSlot) release() { s.once.Do(func() { close(s.released) }) }

type i7BuffersFake struct {
	port  int
	ports []int
	mu    sync.Mutex
	slots map[string]*i7BuffersSlot
}

func i7BuffersUpstream(t *testing.T, peers ...*clustertrust.Mesh) *i7BuffersFake {
	t.Helper()
	u := &i7BuffersFake{slots: make(map[string]*i7BuffersSlot)}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(peers) == 0 {
			for name := range r.Header {
				if strings.HasPrefix(strings.ToLower(name), "x-pair-") || strings.EqualFold(name, "X-NVPAIR-Engine-Identity-Probe") {
					t.Errorf("private header %q reached a manual engine fixture", name)
					http.Error(w, "private engine header", http.StatusBadRequest)
					return
				}
			}
		}
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			if _, err := io.WriteString(w, `{"data":[{"id":"g-t4-concrete"}]}`); err != nil {
				t.Errorf("write fake inventory: %v", err)
			}
			return
		}
		u.mu.Lock()
		slot := u.slots[r.Header.Get("X-I7-Buffers")]
		preOutput := slot != nil && slot.preOutput
		hybrid := slot != nil && slot.hybrid
		u.mu.Unlock()
		if slot == nil || r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			t.Error("unexpected inference reached buffer fixture")
			http.Error(w, "unexpected fixture route", http.StatusBadRequest)
			return
		}
		if hybrid {
			var body struct {
				Model string `json:"model"`
			}
			if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&body); err != nil || body.Model != egressProcessModel {
				t.Errorf("hybrid destination model=%q decode=%v", body.Model, err)
				http.Error(w, "invalid hybrid fixture body", http.StatusBadRequest)
				return
			}
			if err := json.Unmarshal([]byte(r.Header.Get(nodepolicy.StrictAdmissionHeader)), &slot.strict); err != nil || slot.strict.Identity != i7HybridIdentity() {
				t.Errorf("hybrid strict envelope identity=%+v decode=%v", slot.strict.Identity, err)
				http.Error(w, "invalid hybrid fixture envelope", http.StatusBadRequest)
				return
			}
			if r.TLS == nil || slot.strict.Request.OriginNodeID != clustertrust.UUIDFromCert(r.TLS.PeerCertificates[0]) {
				t.Error("hybrid strict request origin differs from the pinned caller")
				http.Error(w, "invalid hybrid fixture origin", http.StatusForbidden)
				return
			}
		}
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Errorf("read held fake request: %v", err)
			return
		}
		close(slot.entered)
		if preOutput {
			select {
			case <-r.Context().Done():
				close(slot.cancelled)
				return
			case <-slot.released:
			}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if len(peers) != 0 {
			// Simulate the pinned destination's framing and terminal attestation.
			w.Header().Set("X-PAIR-Stream-Framing", "v1;declared;1")
			w.Header().Add("Trailer", "X-PAIR-Stream-Framing")
		}
		if _, err := io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ready\"}}]}\n\n"); err != nil {
			t.Errorf("write held stream: %v", err)
			return
		}
		if err := http.NewResponseController(w).Flush(); err != nil {
			t.Errorf("flush held stream: %v", err)
			return
		}
		select {
		case <-r.Context().Done():
			close(slot.cancelled)
		case <-slot.released:
			if _, err := io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"); err != nil {
				t.Errorf("finish held stream: %v", err)
				return
			}
			if len(peers) != 0 {
				w.Header().Set("X-PAIR-Stream-Framing", "v1;completed")
			}
		}
	})
	var servers []*httptest.Server
	if len(peers) == 0 {
		servers = append(servers, newFixtureHTTPServer(t, handler))
	}
	for _, mesh := range peers {
		server := &httptest.Server{Listener: listenFixtureLoopback(t), Config: &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mesh.Refresh()
			principal, pinned := mesh.VerifyClientPin(r)
			incarnation, current := mesh.PeerServiceIncarnation(principal)
			if !pinned || principal != "i7-source-node" || !current {
				t.Error("hybrid fixture peer lacks an authenticated source relationship")
				http.Error(w, "unpinned fixture caller", http.StatusForbidden)
				return
			}
			mesh.Refresh()
			verified, pinned := mesh.VerifyClientPin(r)
			latest, current := mesh.PeerServiceIncarnation(principal)
			if !pinned || verified != principal || !current || latest != incarnation {
				t.Error("hybrid fixture source relationship changed at ingress")
				http.Error(w, "changed fixture caller", http.StatusForbidden)
				return
			}
			handler.ServeHTTP(w, r)
		})}}
		server.TLS = mesh.ServerTLSConfig()
		server.StartTLS()
		servers = append(servers, server)
	}
	for _, server := range servers {
		u.ports = append(u.ports, portOfURL(t, server.URL))
	}
	u.port = u.ports[0]
	t.Cleanup(func() {
		u.mu.Lock()
		for _, slot := range u.slots {
			slot.release()
		}
		u.mu.Unlock()
		for _, server := range servers {
			server.Close()
		}
	})
	return u
}

func (u *i7BuffersFake) slot(t *testing.T, id string) *i7BuffersSlot {
	t.Helper()
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.slots[id] != nil {
		t.Fatalf("duplicate fake request slot %q", id)
	}
	slot := &i7BuffersSlot{entered: make(chan struct{}), cancelled: make(chan struct{}), released: make(chan struct{})}
	u.slots[id] = slot
	return slot
}

func i7BuffersBody(size int) []byte {
	body := bytes.Repeat([]byte(" "), size)
	copy(body, `{"model":"g-t4-concrete","messages":[],"stream":true}`)
	return body
}

type i7BuffersHTTPResult struct {
	status int
	body   []byte
	err    error
}

type i7BuffersRequest struct {
	ctx       context.Context
	cancel    context.CancelFunc
	done      chan struct{}
	firstByte chan struct{}
	result    i7BuffersHTTPResult
}

func i7BuffersStart(t *testing.T, client *http.Client, port int, path, id string, body []byte) *i7BuffersRequest {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	r := &i7BuffersRequest{ctx: ctx, cancel: cancel, done: make(chan struct{}), firstByte: make(chan struct{})}
	sent := make(chan struct{})
	var sentOnce sync.Once
	ctx = httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{WroteHeaders: func() { sentOnce.Do(func() { close(sent) }) }})
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d%s", port, path), bytes.NewReader(body))
	if err != nil {
		cancel()
		t.Fatalf("create buffer fixture request: %v", err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-I7-Buffers", id)
	go func() {
		defer close(r.done)
		response, err := client.Do(request)
		if err != nil {
			r.result.err = err
			return
		}
		r.result.status = response.StatusCode
		var first [1]byte
		n, firstErr := io.ReadFull(response.Body, first[:])
		if n != 0 {
			close(r.firstByte)
		}
		data, readErr := io.ReadAll(io.LimitReader(response.Body, (64<<10)-int64(n)))
		closeErr := response.Body.Close()
		r.result.body = append(first[:n], data...)
		if firstErr != nil && firstErr != io.EOF {
			r.result.err = firstErr
		} else if readErr != nil {
			r.result.err = readErr
		} else {
			r.result.err = closeErr
		}
	}()
	t.Cleanup(func() {
		cancel()
		i7BuffersAwait(t, r)
	})
	select {
	case <-sent:
	case <-r.done:
		select {
		case <-sent:
		default:
			t.Fatalf("request ended before headers were sent: %v", r.result.err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("request headers were not sent")
	}
	return r
}

func i7BuffersAwait(t *testing.T, r *i7BuffersRequest) {
	t.Helper()
	select {
	case <-r.done:
	case <-time.After(10 * time.Second):
		t.Fatal("buffer fixture HTTP request did not join")
	}
}

func i7BuffersJoin(t *testing.T, r *i7BuffersRequest, status int) {
	t.Helper()
	i7BuffersAwait(t, r)
	if r.result.err != nil || r.result.status != status {
		t.Fatalf("buffer HTTP result: status=%d error=%v, want %d", r.result.status, r.result.err, status)
	}
}

func i7BuffersCompletedStream(t *testing.T, r *i7BuffersRequest) {
	t.Helper()
	i7BuffersJoin(t, r, http.StatusOK)
	frames := bytes.Split(r.result.body, []byte("\n\n"))
	if len(frames) != 4 || len(frames[3]) != 0 || !bytes.Equal(frames[2], []byte("data: [DONE]")) {
		t.Fatalf("held stream did not finish with exactly one terminal marker: bytes=%d frames=%d", len(r.result.body), len(frames))
	}
	data, ok := bytes.CutPrefix(frames[0], []byte("data: "))
	if !ok {
		t.Fatal("held stream first frame is not SSE data")
	}
	var event struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	egressProcessDecode(t, data, &event)
	if len(event.Choices) != 1 || event.Choices[0].Delta.Content != "ready" {
		t.Fatalf("held stream lost its initial content: choices=%d", len(event.Choices))
	}
	terminal, ok := bytes.CutPrefix(frames[1], []byte("data: "))
	if !ok {
		t.Fatal("held stream finish frame is not SSE data")
	}
	var finish struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	egressProcessDecode(t, terminal, &finish)
	if len(finish.Choices) != 1 || finish.Choices[0].FinishReason != "stop" {
		t.Fatalf("held stream has no successful finish: choices=%d", len(finish.Choices))
	}
}

func i7BuffersBusy(t *testing.T, r *i7BuffersRequest, slot *i7BuffersSlot) {
	t.Helper()
	i7BuffersJoin(t, r, http.StatusServiceUnavailable)
	var refusal struct {
		Code string `json:"code"`
	}
	egressProcessDecode(t, r.result.body, &refusal)
	if refusal.Code != "request-body" && refusal.Code != "inference-buffer" {
		t.Fatalf("busy response code = %q, want owned-buffer refusal", refusal.Code)
	}
	select {
	case <-slot.entered:
		t.Fatal("busy request dispatched upstream")
	default:
	}
}

func i7BuffersProxyWorkload(t *testing.T, p *egressProcessProxy, method string, deadline time.Time) departedRunInfo {
	t.Helper()
	remaining := time.Until(deadline)
	if remaining <= 0 {
		t.Fatalf("deadline expired before %s", method)
	}
	frame, err := p.client.next(remaining, func(m *jsonrpc.Message) bool { return m.IsNotification() && m.Method == method })
	if err != nil {
		t.Fatalf("proxy %s: %v", method, err)
	}
	var params departedRunParams
	if err := json.Unmarshal(frame.Params, &params); err != nil {
		t.Fatalf("decode proxy %s: %v", method, err)
	}
	return params.WorkloadInfo
}

func i7BuffersBrokerWorkload(t *testing.T, c *serviceRPCClient, matches func(departedRunInfo) bool) departedRunInfo {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			t.Fatal("broker did not publish the expected workload state")
		}
		frame, err := c.next(remaining, func(m *jsonrpc.Message) bool { return m.IsNotification() && m.Method == "workloads:upsert" })
		if err != nil {
			t.Fatalf("broker workload event: %v", err)
		}
		var params departedRunParams
		egressProcessDecode(t, frame.Params, &params)
		if matches(params.WorkloadInfo) {
			return params.WorkloadInfo
		}
	}
}
