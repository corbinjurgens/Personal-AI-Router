// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestPeerSizingActiveRequestPolicyRequiresExactLiveActivation(t *testing.T) {
	cases := []struct {
		name     string
		revision string
		change   func(*Broker, *policyRuntime, *proxyProcess)
		want     bool
	}{
		{"active live run", "22", func(*Broker, *policyRuntime, *proxyProcess) {}, true},
		{"pending activation", "22", func(_ *Broker, r *policyRuntime, _ *proxyProcess) { r.activeRevision = "21" }, false},
		{"activation error", "22", func(_ *Broker, r *policyRuntime, _ *proxyProcess) { r.activationErr = "refused" }, false},
		{"availability error", "22", func(_ *Broker, r *policyRuntime, _ *proxyProcess) { r.availabilityErr = "refused" }, false},
		{"recovery", "22", func(_ *Broker, r *policyRuntime, _ *proxyProcess) { r.recoveryRequired = true }, false},
		{"no acknowledgement", "22", func(_ *Broker, r *policyRuntime, _ *proxyProcess) { r.runID = "" }, false},
		{"exited proxy", "22", func(_ *Broker, _ *policyRuntime, p *proxyProcess) { close(p.done) }, false},
		{"replaced proxy", "22", func(b *Broker, _ *policyRuntime, _ *proxyProcess) {
			b.setProxyProcessHandle(&proxyProcess{done: make(chan struct{})})
		}, false},
		{"older captured policy", "21", func(_ *Broker, r *policyRuntime, _ *proxyProcess) {}, false},
		{"newer persisted policy", "22", func(_ *Broker, r *policyRuntime, _ *proxyProcess) { r.current.Revision = "23" }, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			proxy := &proxyProcess{done: make(chan struct{})}
			b := &Broker{}
			b.setProxyProcessHandle(proxy)
			r := b.nodePolicyRuntime()
			r.mu.Lock()
			r.current.Revision, r.activeRevision, r.runID, r.incarnation = "22", "22", "proxy-run", proxy
			tc.change(b, r, proxy)
			r.mu.Unlock()
			got := b.activeRequestPolicy(tc.revision)
			if (got != nil) != tc.want {
				t.Fatalf("attestation = %+v, want present %v", got, tc.want)
			}
			if got != nil && (got.Revision != "22" || got.ProxyRunID != "proxy-run") {
				t.Fatalf("attestation = %+v", got)
			}
		})
	}
	t.Run("unavailable encodes explicit null", func(t *testing.T) {
		raw, err := json.Marshal(peerPolicyGetResult{})
		if err != nil || !bytes.Contains(raw, []byte(`"activeRequestPolicy":null`)) {
			t.Fatalf("peer read = %s, %v", raw, err)
		}
	})
}

type fakeSizingAuthority struct {
	mu       sync.Mutex
	identity map[string]string
	epoch    map[string]uint64
}

func (a *fakeSizingAuthority) Refresh() {}

func (a *fakeSizingAuthority) PeerServiceIncarnation(peer string) (string, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	v, ok := a.identity[peer]
	return v, ok
}

func (a *fakeSizingAuthority) PeerAdmissionEpoch(peer string) (uint64, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	v, ok := a.epoch[peer]
	return v, ok
}

func (a *fakeSizingAuthority) set(peer, identity string, epoch uint64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if identity == "" {
		delete(a.identity, peer)
		delete(a.epoch, peer)
		return
	}
	a.identity[peer], a.epoch[peer] = identity, epoch
}

type sizingFixture struct {
	r         *peerSizingRuntime
	authority *fakeSizingAuthority
	clock     time.Time
	worker    *rpcWorker
	mu        sync.Mutex
	replies   map[string]func(ctx context.Context) (json.RawMessage, error)
	fetches   map[string]int
	sent      []nodepolicy.PeerSizingSnapshot
}

func newSizingFixture(t *testing.T) *sizingFixture {
	t.Helper()
	f := &sizingFixture{authority: &fakeSizingAuthority{identity: map[string]string{}, epoch: map[string]uint64{}}, clock: time.Unix(1000, 0), worker: &rpcWorker{}, replies: map[string]func(context.Context) (json.RawMessage, error){}, fetches: map[string]int{}}
	f.r = newPeerSizingRuntime("self", f.authority)
	f.r.now = func() time.Time {
		f.mu.Lock()
		defer f.mu.Unlock()
		return f.clock
	}
	f.r.source = func() *rpcWorker {
		f.mu.Lock()
		defer f.mu.Unlock()
		return f.worker
	}
	f.r.fetch = func(ctx context.Context, w *rpcWorker, host string) (json.RawMessage, error) {
		f.mu.Lock()
		f.fetches[host]++
		reply := f.replies[host]
		f.mu.Unlock()
		if reply == nil {
			return nil, errors.New("unavailable")
		}
		return reply(ctx)
	}
	f.r.deliver = func(_ context.Context, params json.RawMessage) {
		var s nodepolicy.PeerSizingSnapshot
		if err := json.Unmarshal(params, &s); err != nil {
			t.Fatalf("delivered invalid snapshot %s: %v", params, err)
		}
		f.mu.Lock()
		f.sent = append(f.sent, s)
		f.mu.Unlock()
	}
	return f
}

func (f *sizingFixture) advance(d time.Duration) {
	f.mu.Lock()
	f.clock = f.clock.Add(d)
	f.mu.Unlock()
}

func (f *sizingFixture) reply(host string, fn func(ctx context.Context) (json.RawMessage, error)) {
	f.mu.Lock()
	f.replies[host] = fn
	f.mu.Unlock()
}

func (f *sizingFixture) last(t *testing.T) nodepolicy.PeerSizingSnapshot {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.sent) == 0 {
		t.Fatal("nothing delivered")
	}
	return f.sent[len(f.sent)-1]
}

func (f *sizingFixture) sends() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.sent)
}

func sizingNode(host, revision string, profiles ...nodepolicy.AdvertisedProfile) noderec.DirectoryNode {
	return noderec.DirectoryNode{HostUUID: host, ClusterUUID: "cluster-" + host, Policy: &nodepolicy.Advertisement{Revision: revision, Profiles: profiles}}
}

func advertised(key, engine, model string) nodepolicy.AdvertisedProfile {
	return nodepolicy.AdvertisedProfile{Key: key, Engine: engine, Model: model}
}

func policyRead(revision string, active *nodepolicy.ActiveRequestPolicy, profiles ...nodepolicy.Profile) func(context.Context) (json.RawMessage, error) {
	return func(context.Context) (json.RawMessage, error) {
		raw, err := json.Marshal(map[string]any{"policy": map[string]any{"revision": revision, "profiles": profiles}, "activeRequestPolicy": active})
		return raw, err
	}
}

func active(revision string) *nodepolicy.ActiveRequestPolicy {
	return &nodepolicy.ActiveRequestPolicy{Revision: revision, ProxyRunID: "run-" + revision}
}

func TestPeerSizingProjectsOnlyActivatedAdvertisedSharedProfiles(t *testing.T) {
	f := newSizingFixture(t)
	f.authority.set("cluster-b", "incarnation-b", 4)
	f.r.setDirectory([]noderec.DirectoryNode{
		sizingNode("b", "7", advertised("medium", "llamacpp", "gemma"), advertised("fast", "llamacpp", "qwen"), advertised("moved", "llamacpp", "old-name")),
		sizingNode("self", "1", advertised("medium", "llamacpp", "gemma")),
		{HostUUID: "unpaired", Policy: &nodepolicy.Advertisement{Revision: "3"}},
	})
	f.reply("b", policyRead("7", active("7"),
		nodepolicy.Profile{Key: "medium", Engine: "llamacpp", Model: "gemma", RequestOptions: json.RawMessage(`{"max_tokens":64}`), RequestOptionsMode: nodepolicy.RequestOptionsEnforced},
		nodepolicy.Profile{Key: "fast", Engine: "llamacpp", Model: "qwen"},
		nodepolicy.Profile{Key: "moved", Engine: "llamacpp", Model: "new-name"},
		nodepolicy.Profile{Key: "secret", Engine: "llamacpp", Model: "private", LocalOnly: true, RequestOptions: json.RawMessage(`{"temperature":0}`)},
	))
	f.r.reconcile(context.Background())
	got := f.last(t)
	if len(got.Peers) != 1 || got.Peers[0].Host != "b" {
		t.Fatalf("peers = %+v", got.Peers)
	}
	peer := got.Peers[0]
	if peer.AdmissionIdentity != "incarnation-b" || peer.AdmissionIncarnation != "4" || peer.PolicyRevision != "7" || peer.ProxyRunID != "run-7" {
		t.Fatalf("peer authority = %+v", peer)
	}
	keys := []string{}
	for _, p := range peer.Profiles {
		keys = append(keys, p.Key)
	}
	if strings.Join(keys, ",") != "medium,fast" {
		t.Fatalf("recipes = %v, want only matching advertised shared profiles", keys)
	}
	if string(peer.Profiles[0].RequestOptions) != `{"max_tokens":64}` || peer.Profiles[0].RequestOptionsMode != nodepolicy.RequestOptionsEnforced || len(peer.Profiles[1].RequestOptions) != 0 {
		t.Fatalf("recipe contents = %+v", peer.Profiles)
	}
	if f.fetches["self"] != 0 || f.fetches["unpaired"] != 0 {
		t.Fatalf("fetched self or unadvertised peers: %v", f.fetches)
	}
}

func TestPeerSizingRejectedReadsWithdrawImmediately(t *testing.T) {
	good := policyRead("7", active("7"), nodepolicy.Profile{Key: "medium", Engine: "llamacpp", Model: "gemma"})
	cases := []struct {
		name  string
		reply func(context.Context) (json.RawMessage, error)
	}{
		{"null activation", policyRead("7", nil, nodepolicy.Profile{Key: "medium", Engine: "llamacpp", Model: "gemma"})},
		{"activation of another revision", policyRead("7", active("6"), nodepolicy.Profile{Key: "medium", Engine: "llamacpp", Model: "gemma"})},
		{"stale advertisement", policyRead("8", active("8"), nodepolicy.Profile{Key: "medium", Engine: "llamacpp", Model: "gemma"})},
		{"malformed", func(context.Context) (json.RawMessage, error) { return json.RawMessage(`{"policy":`), nil }},
		{"read failure", func(context.Context) (json.RawMessage, error) { return nil, errors.New("timeout") }},
		{"oversize response", func(context.Context) (json.RawMessage, error) {
			return nil, errors.New("remote node policy: read response: body exceeds limit")
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newSizingFixture(t)
			f.authority.set("cluster-b", "incarnation-b", 4)
			f.r.setDirectory([]noderec.DirectoryNode{sizingNode("b", "7", advertised("medium", "llamacpp", "gemma"))})
			f.reply("b", good)
			f.r.reconcile(context.Background())
			first := f.last(t)
			if len(first.Peers) != 1 {
				t.Fatalf("initial snapshot = %+v", first)
			}
			f.reply("b", tc.reply)
			f.advance(peerSizingRefreshInterval)
			f.r.reconcile(context.Background())
			got := f.last(t)
			if len(got.Peers) != 0 || got.Revision <= first.Revision {
				t.Fatalf("rejected read left %+v (first revision %d)", got, first.Revision)
			}
		})
	}
}

func TestPeerSizingDelayedReplyAfterAuthorityChangeIsDiscarded(t *testing.T) {
	cases := []struct {
		name   string
		change func(*sizingFixture)
	}{
		{"unpaired", func(f *sizingFixture) { f.authority.set("cluster-b", "", 0) }},
		{"re-paired with a new epoch", func(f *sizingFixture) { f.authority.set("cluster-b", "incarnation-b", 5) }},
		{"re-paired with a new identity", func(f *sizingFixture) { f.authority.set("cluster-b", "incarnation-b2", 4) }},
		{"engine manager replaced", func(f *sizingFixture) {
			f.mu.Lock()
			f.worker = &rpcWorker{}
			f.mu.Unlock()
		}},
		{"advertisement changed", func(f *sizingFixture) {
			f.r.setDirectory([]noderec.DirectoryNode{sizingNode("b", "8", advertised("medium", "llamacpp", "gemma"))})
		}},
		{"directory removal", func(f *sizingFixture) { f.r.setDirectory(nil) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newSizingFixture(t)
			f.authority.set("cluster-b", "incarnation-b", 4)
			f.r.setDirectory([]noderec.DirectoryNode{sizingNode("b", "7", advertised("medium", "llamacpp", "gemma"))})
			release := make(chan struct{})
			entered := make(chan struct{})
			read := policyRead("7", active("7"), nodepolicy.Profile{Key: "medium", Engine: "llamacpp", Model: "gemma"})
			f.reply("b", func(ctx context.Context) (json.RawMessage, error) {
				close(entered)
				select {
				case <-release:
				case <-ctx.Done():
				}
				return read(ctx)
			})
			done := make(chan struct{})
			go func() {
				defer close(done)
				f.r.reconcile(context.Background())
			}()
			<-entered
			tc.change(f)
			close(release)
			<-done
			f.mu.Lock()
			defer f.mu.Unlock()
			for _, s := range f.sent {
				if len(s.Peers) != 0 {
					t.Fatalf("delayed reply under changed authority was projected: %+v", s)
				}
			}
		})
	}
}

func TestPeerSizingHeldObservationInvalidatesBeforeRefresh(t *testing.T) {
	cases := []struct {
		name   string
		change func(*sizingFixture)
	}{
		{"directory revision change", func(f *sizingFixture) {
			f.r.setDirectory([]noderec.DirectoryNode{sizingNode("b", "8", advertised("medium", "llamacpp", "gemma"))})
		}},
		{"profile identity change", func(f *sizingFixture) {
			f.r.setDirectory([]noderec.DirectoryNode{sizingNode("b", "7", advertised("medium", "llamacpp", "gemma-2"))})
		}},
		{"removal", func(f *sizingFixture) { f.r.setDirectory(nil) }},
		{"trust change", func(f *sizingFixture) { f.authority.set("cluster-b", "incarnation-b", 9) }},
		{"manager replacement", func(f *sizingFixture) {
			f.mu.Lock()
			f.worker = &rpcWorker{}
			f.mu.Unlock()
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newSizingFixture(t)
			f.authority.set("cluster-b", "incarnation-b", 4)
			f.r.setDirectory([]noderec.DirectoryNode{sizingNode("b", "7", advertised("medium", "llamacpp", "gemma"))})
			f.reply("b", policyRead("7", active("7"), nodepolicy.Profile{Key: "medium", Engine: "llamacpp", Model: "gemma"}))
			f.r.reconcile(context.Background())
			f.reply("b", nil)
			tc.change(f)
			f.r.invalidate()
			f.r.send(context.Background(), false)
			if got := f.last(t); len(got.Peers) != 0 {
				t.Fatalf("held observation survived %s: %+v", tc.name, got)
			}
		})
	}
}

func TestPeerSizingOnlyFreshReadsRenewObservations(t *testing.T) {
	f := newSizingFixture(t)
	f.authority.set("cluster-b", "incarnation-b", 4)
	node := sizingNode("b", "7", advertised("medium", "llamacpp", "gemma"))
	f.r.setDirectory([]noderec.DirectoryNode{node})
	f.reply("b", policyRead("7", active("7"), nodepolicy.Profile{Key: "medium", Engine: "llamacpp", Model: "gemma"}))
	f.r.reconcile(context.Background())
	first := f.last(t)
	sent := f.sends()

	// Unchanged inventory heartbeats inside the refresh interval neither fetch
	// nor renew, and an unchanged projection is not delivered again.
	for range 5 {
		f.advance(5 * time.Second)
		f.r.setDirectory([]noderec.DirectoryNode{node})
		f.r.reconcile(context.Background())
	}
	if f.fetches["b"] != 1 || f.sends() != sent {
		t.Fatalf("heartbeats fetched %d times and sent %d snapshots", f.fetches["b"], f.sends())
	}
	// Replay keeps the revision while ages advance from the original fetch.
	f.r.send(context.Background(), true)
	replay := f.last(t)
	if replay.Revision != first.Revision || replay.Peers[0].AgeMs != 25000 {
		t.Fatalf("replay = %+v, first revision %d", replay, first.Revision)
	}
	// A fresh successful read with equal content renews the age and is a new revision.
	f.advance(5 * time.Second)
	f.r.reconcile(context.Background())
	renewed := f.last(t)
	if f.fetches["b"] != 2 || renewed.Revision != first.Revision+1 || renewed.Peers[0].AgeMs != 0 {
		t.Fatalf("fresh read = %+v after %d fetches", renewed, f.fetches["b"])
	}
	// Failing refreshes plus heartbeats cannot hold it past the freshness window.
	f.reply("b", nil)
	f.advance(peerSizingRefreshInterval)
	f.r.reconcile(context.Background())
	if got := f.last(t); len(got.Peers) != 0 {
		t.Fatalf("failed refresh kept the observation: %+v", got)
	}
}

func TestPeerSizingExpiresAtFreshnessWithoutRead(t *testing.T) {
	f := newSizingFixture(t)
	f.authority.set("cluster-b", "incarnation-b", 4)
	f.r.setDirectory([]noderec.DirectoryNode{sizingNode("b", "7", advertised("medium", "llamacpp", "gemma"))})
	f.reply("b", policyRead("7", active("7"), nodepolicy.Profile{Key: "medium", Engine: "llamacpp", Model: "gemma"}))
	f.r.reconcile(context.Background())
	f.advance(nodepolicy.PeerSizingFreshnessMs * time.Millisecond)
	f.r.invalidate()
	f.r.send(context.Background(), false)
	if got := f.last(t); len(got.Peers) != 0 {
		t.Fatalf("expired observation delivered: %+v", got)
	}
}

func TestPeerSizingSnapshotBudgetOmitsWholePeers(t *testing.T) {
	f := newSizingFixture(t)
	big := json.RawMessage(fmt.Sprintf(`{"pad":%q}`, strings.Repeat("x", 150<<10)))
	var nodes []noderec.DirectoryNode
	for _, host := range []string{"a", "b", "c"} {
		f.authority.set("cluster-"+host, "incarnation-"+host, 1)
		nodes = append(nodes, sizingNode(host, "2", advertised("medium", "llamacpp", "gemma")))
		options := json.RawMessage(`{"max_tokens":8}`)
		if host != "c" {
			options = big
		}
		f.reply(host, policyRead("2", active("2"), nodepolicy.Profile{Key: "medium", Engine: "llamacpp", Model: "gemma", RequestOptions: options}))
	}
	f.r.setDirectory(nodes)
	f.r.reconcile(context.Background())
	got := f.last(t)
	hosts := []string{}
	for _, p := range got.Peers {
		hosts = append(hosts, p.Host)
	}
	if strings.Join(hosts, ",") != "a,c" {
		t.Fatalf("packed hosts = %v, want a whole, b omitted, c whole", hosts)
	}
}

func TestPeerSizingClientCannotSetProxyCache(t *testing.T) {
	for _, prefix := range []string{"", engines.ProxyComponent + ":", "llamacpp-proxy:"} {
		t.Run(prefix, func(t *testing.T) {
			peer, worker := newFakeWorkerPipe(t)
			var output bytes.Buffer
			b := &Broker{codec: NewCodec(readWriter{Reader: strings.NewReader(""), Writer: &output})}
			b.setEngineProxyHandle(mustEngineProxyProfile("llamacpp"), &proxyProcess{peer: peer})
			id := json.RawMessage(`1`)
			b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: prefix + nodepolicy.MethodSetPeerSizing, Params: json.RawMessage(`{"revision":18446744073709551615,"peers":[]}`)})
			var response Message
			if err := json.Unmarshal(bytes.TrimSpace(output.Bytes()), &response); err != nil {
				t.Fatal(err)
			}
			if response.Error == nil || response.Error.Code != -32601 || len(worker.snapshot()) != 0 {
				t.Fatalf("client request was not privately refused: %+v, calls %+v", response, worker.snapshot())
			}
		})
	}
}
