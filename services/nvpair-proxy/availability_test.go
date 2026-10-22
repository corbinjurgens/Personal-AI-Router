// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestSubscribedToNodeCarriesPausedBy(t *testing.T) {
	for _, profile := range append(buildProfiles(), strataProfile(t)) {
		t.Run(profile.Name, func(t *testing.T) {
			test := func(name string, policy *nodepolicy.Advertisement, revision string, pausedBy nodepolicy.PauseOrigin) {
				t.Run(name, func(t *testing.T) {
					capability := strictTestCapability("m")
					if policy != nil {
						policy.Roles = map[string]nodepolicy.AdvertisedRole{
							nodepolicy.TierStrong: {Engine: profile.Name, Model: "m", ContextTokens: 32768},
						}
						policy.ServedModels = nodepolicy.ServedModels{{Engine: profile.Name, Model: "m"}}
						policy.StrictAdmission = capability
					}
					n := noderec.DirectoryNode{
						HostUUID:    "host-uuid",
						Name:        "host-a",
						IP:          "10.0.0.5",
						IPs:         []string{"10.0.0.5", "10.0.0.6"},
						ClusterUUID: "cluster-uuid",
						Services: map[noderec.ServiceKey]noderec.ServiceStatus{
							profile.DiscoveryService: {Port: 19001},
						},
						Models:         []string{"m", "other-engine-model"},
						ModelsByEngine: map[string][]string{profile.Name: {"m"}},
						LoadedByEngine: map[string][]string{profile.Name: {"m"}},
						Policy:         policy,
					}
					got, ok := subscribedToNode(profile, n)
					if !ok {
						t.Fatal("node advertising this engine and IP should project")
					}
					want := Node{
						ID:             "host-uuid",
						Host:           "host-a",
						Port:           19001,
						Addresses:      []string{"10.0.0.5", "10.0.0.6"},
						TXT:            []string{"ip=10.0.0.5", "ips=10.0.0.5,10.0.0.6"},
						Models:         []string{"m"},
						Loaded:         []string{"m"},
						IP:             "10.0.0.5",
						ClusterUUID:    "cluster-uuid",
						PolicyRevision: revision,
						PausedBy:       pausedBy,
					}
					if policy != nil {
						want.Roles = map[string]nodepolicy.AdvertisedRole{
							nodepolicy.TierStrong: {Engine: profile.Name, Model: "m", ContextTokens: 32768},
						}
						want.ServedModels = nodepolicy.ServedModels{{Engine: profile.Name, Model: "m"}}
						want.StrictAdmission = capability
					}
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("projected node = %+v, want %+v", got, want)
					}

					data, err := json.Marshal(got)
					if err != nil {
						t.Fatalf("encode projected node: %v", err)
					}
					var wire map[string]json.RawMessage
					if err := json.Unmarshal(data, &wire); err != nil {
						t.Fatalf("decode projected node: %v", err)
					}
					for _, key := range []string{"pausedBy", "PausedBy", "policyRevision", "PolicyRevision"} {
						if value, exists := wire[key]; exists {
							t.Errorf("internal routing metadata %q serialized as %s", key, value)
						}
					}
				})
			}
			test("nil policy", nil, "", "")
			test("revision without pause", &nodepolicy.Advertisement{Revision: "41"}, "41", "")
			test("local pause", &nodepolicy.Advertisement{Revision: "42", PausedBy: nodepolicy.PausedLocally}, "42", nodepolicy.PausedLocally)
			test("peer pause", &nodepolicy.Advertisement{Revision: "43", PausedBy: nodepolicy.PausedByPeer}, "43", nodepolicy.PausedByPeer)
		})
	}
}

func TestPauseHintAnnouncesNoNodeUpdate(t *testing.T) {
	d := NewDiscovery()
	initial := Node{
		ID:             "host-uuid",
		Host:           "host-a",
		Port:           19001,
		Addresses:      []string{"10.0.0.5"},
		TXT:            []string{"ip=10.0.0.5"},
		Models:         []string{"m"},
		IP:             "10.0.0.5",
		PolicyRevision: "40",
	}
	discovered, updated, removed := d.SetSubscribed([]Node{initial})
	if !reflect.DeepEqual(discovered, []Node{initial}) || len(updated) != 0 || len(removed) != 0 {
		t.Fatalf("initial changes: discovered=%+v updated=%+v removed=%+v", discovered, updated, removed)
	}
	if got := d.Nodes(); !reflect.DeepEqual(got, []Node{initial}) {
		t.Fatalf("initial snapshot = %+v, want %+v", got, initial)
	}
	for _, step := range []struct {
		name     string
		revision string
		pausedBy nodepolicy.PauseOrigin
	}{
		{"revision only", "41", ""},
		{"local pause", "42", nodepolicy.PausedLocally},
		{"provenance only", "42", nodepolicy.PausedByPeer},
		{"available again", "43", ""},
	} {
		t.Run(step.name, func(t *testing.T) {
			next := initial
			next.PolicyRevision = step.revision
			next.PausedBy = step.pausedBy
			discovered, updated, removed := d.SetSubscribed([]Node{next})
			if len(discovered) != 0 || len(updated) != 0 || len(removed) != 0 {
				t.Errorf("metadata replacement announced changes: discovered=%+v updated=%+v removed=%+v", discovered, updated, removed)
			}
			if got := d.Nodes(); !reflect.DeepEqual(got, []Node{next}) {
				t.Fatalf("stored snapshot = %+v, want %+v", got, next)
			}
		})
	}
}

const paModel = "pa-model"

// PA fixtures own every listener, including the engine behind each real peer
// ingress. None can occupy an installed engine or PAIR service port.
func paListen(t *testing.T) net.Listener {
	t.Helper()
	for range 32 {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("listen on checked loopback: %v", err)
		}
		port := ln.Addr().(*net.TCPAddr).Port
		if port != 8080 && port != 11434 && port != 1234 && (port < 14318 || port > 14323) {
			return ln
		}
		if err := ln.Close(); err != nil {
			t.Fatalf("close excluded listener: %v", err)
		}
	}
	t.Fatal("could not allocate a permitted loopback port")
	return nil
}

func paServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	s := &httptest.Server{Listener: paListen(t), Config: &http.Server{Handler: handler}}
	s.Start()
	t.Cleanup(s.Close)
	return s
}

type paEndpoint struct {
	p          *Proxy
	node       noderec.DirectoryNode
	inference  atomic.Int32
	list       atomic.Int32
	control    atomic.Int32
	ingress    atomic.Int32
	beforeMu   sync.Mutex
	before     func()
	rejections chan paRejection
}

type paRejection struct {
	status int
	header http.Header
}

type paFixture struct {
	p        *Proxy
	f        *facade
	dir      string
	origin   *httptest.Server
	returned chan struct{}
	w        *recordingWriter
}

// Install the actual certificates of the joined endpoints, rather than the
// independently minted pins Join can supply for resolver-only tests.
func paPinPeers(t *testing.T, destination string, sources map[string]string) {
	t.Helper()
	trusted := filepath.Join(destination, "trusted")
	if err := os.MkdirAll(trusted, 0o700); err != nil {
		t.Fatalf("create fixture pins: %v", err)
	}
	var members []map[string]any
	for id, source := range sources {
		cert, err := os.ReadFile(filepath.Join(source, "node.crt"))
		if err != nil {
			t.Fatalf("read fixture certificate: %v", err)
		}
		pin, err := json.Marshal(map[string]any{"nodeUuid": id, "certPem": string(cert), "clusterId": "fixture-cluster", "admissionEpoch": 1})
		if err != nil {
			t.Fatalf("encode fixture pin: %v", err)
		}
		if err := os.WriteFile(filepath.Join(trusted, id+".json"), pin, 0o600); err != nil {
			t.Fatalf("write fixture pin: %v", err)
		}
		members = append(members, map[string]any{"nodeUuid": id, "state": "member", "clusterId": "fixture-cluster", "admissionEpoch": 1})
	}
	encoded, err := json.Marshal(members)
	if err != nil {
		t.Fatalf("encode fixture members: %v", err)
	}
	if err := os.WriteFile(filepath.Join(destination, "members.json"), encoded, 0o600); err != nil {
		t.Fatalf("write fixture members: %v", err)
	}
}

func paNewFixture(t *testing.T, profile engineProfile) *paFixture {
	t.Helper()
	ln := paListen(t)
	w := &recordingWriter{}
	p := newTestProxy(profile, NewCodec(w), NewDiscovery(), ln.Addr().(*net.TCPAddr).Port)
	dir := t.TempDir()
	clustertrusttest.Join(t, dir, "fixture-cluster", "test-origin")
	p.mesh = clustertrust.Open(dir)
	t.Cleanup(p.closeIdleTransports)
	fx := &paFixture{p: p, f: p.soleFacade(), dir: dir, w: w, returned: make(chan struct{}, 32)}
	fx.origin = &httptest.Server{Listener: ln, Config: &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() { fx.returned <- struct{}{} }()
		fx.f.handlePlain(w, r)
	})}}
	fx.origin.Start()
	fx.origin.Client().Timeout = 5 * time.Second
	t.Cleanup(fx.origin.Close)
	return fx
}

func paEngine(t *testing.T, endpoint *paEndpoint, profile engineProfile, id string) *httptest.Server {
	t.Helper()
	return paServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if isInferenceRequest(profile, r.Method, r.URL.Path) {
			endpoint.inference.Add(1)
			var body struct {
				Model string `json:"model"`
			}
			err := json.NewDecoder(r.Body).Decode(&body)
			closeErr := r.Body.Close()
			if err != nil || closeErr != nil || profile.normalizeModel(body.Model) != profile.normalizeModel(paModel) {
				t.Errorf("engine request model/read/close = %q/%v/%v", body.Model, err, closeErr)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(map[string]any{"id": id, "object": "chat.completion", "choices": []map[string]any{{"index": 0, "message": map[string]string{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}}}); err != nil {
				t.Errorf("write engine completion: %v", err)
			}
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet && (r.URL.Path == "/v1/models" || r.URL.Path == "/models") {
			endpoint.list.Add(1)
			if err := json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []map[string]string{{"id": paModel, "object": "model"}}}); err != nil {
				t.Errorf("write engine inventory: %v", err)
			}
			return
		}
		endpoint.control.Add(1)
		if err := json.NewEncoder(w).Encode(map[string]string{"id": id}); err != nil {
			t.Errorf("write engine control response: %v", err)
		}
	}))
}

const paModelLessBody = `{"messages":[{"role":"user","content":"fixture request"}],"stream":false}`

func paModelLessEngine(t *testing.T, endpoint *paEndpoint, id string) *httptest.Server {
	t.Helper()
	return paServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		endpoint.inference.Add(1)
		body, readErr := io.ReadAll(r.Body)
		closeErr := r.Body.Close()
		if readErr != nil || closeErr != nil {
			t.Errorf("model-less engine request read/close = %v/%v", readErr, closeErr)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if r.Method != http.MethodPost || r.URL.Path != routeChatCompletions || string(body) != paModelLessBody {
			t.Errorf("model-less engine method/path/body preserved = %s/%s/%t", r.Method, r.URL.Path, string(body) == paModelLessBody)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		var request map[string]json.RawMessage
		if err := json.Unmarshal(body, &request); err != nil {
			t.Errorf("decode model-less engine request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if _, exists := request["model"]; exists {
			t.Error("model-less engine request acquired a model member")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{"id": id, "object": "chat.completion", "choices": []map[string]any{{"index": 0, "message": map[string]string{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}}}); err != nil {
			t.Errorf("write model-less engine completion: %v", err)
		}
	}))
}

func paModelLessServed(t *testing.T, response paResponse, profile engineProfile, id string) {
	t.Helper()
	paServed(t, response, id)
	if response.header.Get(nodepolicy.NodeHeader) != id || response.header.Get(nodepolicy.EngineHeader) != profile.Name || response.header.Get(nodepolicy.ModelHeader) != "" || response.header.Get(nodepolicy.TierHeader) != "" {
		t.Fatalf("model-less routing node/engine/model/tier = %q/%q/%q/%q, want %q/%q/empty/empty", response.header.Get(nodepolicy.NodeHeader), response.header.Get(nodepolicy.EngineHeader), response.header.Get(nodepolicy.ModelHeader), response.header.Get(nodepolicy.TierHeader), id, profile.Name)
	}
}

func (fx *paFixture) peers(t *testing.T, ids ...string) []*paEndpoint {
	t.Helper()
	var peers []*paEndpoint
	sources := map[string]string{}
	for _, id := range ids {
		dir := t.TempDir()
		clustertrusttest.Join(t, dir, "fixture-cluster", id)
		paPinPeers(t, dir, map[string]string{"test-origin": fx.dir})
		e := &paEndpoint{rejections: make(chan paRejection, 8)}
		engine := paEngine(t, e, fx.f.profile, id)
		ln := paListen(t)
		port := ln.Addr().(*net.TCPAddr).Port
		e.p = newTestProxy(fx.f.profile, NewCodec(rwNop{}), NewDiscovery(), port)
		e.p.originNodeID = id
		e.p.mesh = clustertrust.Open(dir)
		setBackend(t, e.p.soleFacade(), engine.URL, true)
		pol := nodepolicy.Default()
		pol.ServedModels = nodepolicy.ServedModels{{Engine: fx.f.profile.Name, Model: paModel}}
		e.p.admission.setPolicy(pol)
		s := &httptest.Server{Listener: ln, Config: &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			e.ingress.Add(1)
			e.beforeMu.Lock()
			before := e.before
			e.beforeMu.Unlock()
			if before != nil {
				before()
			}
			sc := &statusCapture{ResponseWriter: w, status: http.StatusOK}
			e.p.soleFacade().handleClusterIngress(sc, r)
			if sc.status == http.StatusServiceUnavailable {
				e.rejections <- paRejection{status: sc.status, header: w.Header().Clone()}
			}
		})}}
		s.TLS = e.p.mesh.ServerTLSConfig()
		s.StartTLS()
		t.Cleanup(s.Close)
		t.Cleanup(e.p.closeIdleTransports)
		e.node = noderec.DirectoryNode{
			HostUUID: id, Name: id, IP: "127.0.0.1", IPs: []string{"127.0.0.1"}, ClusterUUID: id,
			Services:       map[noderec.ServiceKey]noderec.ServiceStatus{fx.f.profile.DiscoveryService: {Port: port}},
			ModelsByEngine: map[string][]string{fx.f.profile.Name: {paModel}},
			Policy: &nodepolicy.Advertisement{Revision: "1", ServedModels: pol.ServedModels,
				Roles: map[string]nodepolicy.AdvertisedRole{nodepolicy.TierStrong: {
					Engine: fx.f.profile.Name, Model: paModel, ContextTokens: 4096, MaxOutputTokens: 1024,
					Qualification: &nodepolicy.QualificationRef{ID: "q", Revision: "1"},
				}}, StrictAdmission: strictCapabilityFor(fx.f.profile.Name, paModel)},
		}
		if state := e.node.Policy.StrictAdmission.State(); state != nodepolicy.StrictCapabilityValid {
			t.Fatalf("peer strict capability = %s, want valid", state)
		}
		peers = append(peers, e)
		sources[id] = dir
	}
	paPinPeers(t, fx.dir, sources)
	fx.p.mesh.Refresh()
	return peers
}

func (fx *paFixture) replace(t *testing.T, nodes ...noderec.DirectoryNode) {
	t.Helper()
	encoded, err := json.Marshal(noderec.GetNodesResult{Nodes: nodes})
	if err != nil {
		t.Fatalf("encode discovery replacement: %v", err)
	}
	fx.f.replaceSubscribed(encoded)
	want := make([]Node, 0, len(nodes))
	for _, n := range nodes {
		projected, ok := subscribedToNode(fx.f.profile, n)
		if !ok {
			t.Fatalf("replacement node %s did not project", n.HostUUID)
		}
		want = append(want, projected)
	}
	got := fx.f.discovery.Nodes()
	slices.SortFunc(got, func(a, b Node) int { return strings.Compare(a.ID, b.ID) })
	slices.SortFunc(want, func(a, b Node) int { return strings.Compare(a.ID, b.ID) })
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("discovery replacement = %+v, want %+v", got, want)
	}
}

type paResponse struct {
	status int
	header http.Header
	body   []byte
	err    error
}

func (fx *paFixture) request(method, path, body string) paResponse {
	r, err := http.NewRequest(method, fx.origin.URL+path, strings.NewReader(body))
	if err != nil {
		return paResponse{err: err}
	}
	r.Header.Set("Content-Type", "application/json")
	response, err := fx.origin.Client().Do(r)
	if err != nil {
		return paResponse{err: err}
	}
	data, readErr := io.ReadAll(response.Body)
	closeErr := response.Body.Close()
	if readErr != nil {
		err = readErr
	} else {
		err = closeErr
	}
	return paResponse{status: response.StatusCode, header: response.Header.Clone(), body: data, err: err}
}

func (fx *paFixture) finish(t *testing.T, response paResponse, status int) paResponse {
	t.Helper()
	if response.err != nil {
		t.Fatalf("HTTP request/read/close: %v", response.err)
	}
	waitChan(t, fx.returned, "PA origin handler return")
	if response.status != status {
		t.Fatalf("HTTP status = %d, want %d", response.status, status)
	}
	return response
}

func (fx *paFixture) infer(t *testing.T, model string, status int) paResponse {
	t.Helper()
	return fx.finish(t, fx.request(http.MethodPost, routeChatCompletions, fmt.Sprintf(`{"model":%q,"stream":false}`, model)), status)
}

func paServed(t *testing.T, response paResponse, id string) {
	t.Helper()
	var completion struct {
		ID      string `json:"id"`
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(response.body, &completion); err != nil {
		t.Fatalf("decode completion: %v", err)
	}
	if completion.ID != id || len(completion.Choices) != 1 || completion.Choices[0].FinishReason != "stop" {
		t.Fatalf("completion identity/terminal = %+v, want %s/stop", completion, id)
	}
}

func paCandidates(t *testing.T, f *facade, ids ...string) {
	t.Helper()
	check := func(name string, candidates []candidate) {
		t.Helper()
		got := make([]string, 0, len(candidates))
		for _, c := range candidates {
			got = append(got, c.id)
			if !c.self && (c.peerUUID == "" || c.url.Scheme != "https") {
				t.Fatalf("%s candidate %s lacks pinned transport", name, c.id)
			}
		}
		if !slices.Equal(got, ids) {
			t.Errorf("%s candidates = %v, want %v", name, got, ids)
		}
	}
	check("concrete", f.resolveCandidates(paModel))
	check("tier", f.resolveTierCandidates(nodepolicy.TierStrong, requestNeeds{}, true))
}

func paPaused(t *testing.T, response paResponse) {
	t.Helper()
	var rejection struct {
		Code   string `json:"code"`
		Reason string `json:"reason"`
		Error  string `json:"error"`
	}
	if err := json.Unmarshal(response.body, &rejection); err != nil {
		t.Fatalf("decode pause rejection: %v", err)
	}
	if response.status != http.StatusServiceUnavailable || rejection.Code != "admission" || rejection.Reason != "paused" || rejection.Error != "this node is not accepting the request (paused)" || response.header.Get(nodepolicy.AdmissionHeader) != "paused" || response.header.Get("Retry-After") != "30" {
		t.Fatalf("pause rejection status/code/reason/headers = %d/%s/%s/%v", response.status, rejection.Code, rejection.Reason, response.header)
	}
}

func TestPAPausedPinnedPeerEligibility(t *testing.T) {
	for _, profile := range append(buildProfiles(), strataProfile(t)) {
		t.Run(profile.Name, func(t *testing.T) {
			test := func(name string, hint nodepolicy.PauseOrigin, selected bool) {
				t.Run(name, func(t *testing.T) {
					fx := paNewFixture(t, profile)
					peers := fx.peers(t, "a-paused", "z-available")
					a, b := peers[0], peers[1]
					fx.replace(t, a.node, b.node)
					fx.p.SetPriority([]string{a.node.HostUUID, b.node.HostUUID})
					if selected {
						fx.f.SetSelected(a.node.HostUUID)
					}
					paCandidates(t, fx.f, "a-paused", "z-available")
					paServed(t, fx.infer(t, paModel, http.StatusOK), "a-paused")
					paServed(t, fx.infer(t, nodepolicy.TierStrong, http.StatusOK), "a-paused")
					a.node.Policy.PausedBy, a.node.Policy.Revision = hint, "2"
					fx.replace(t, a.node, b.node)
					paCandidates(t, fx.f, "z-available")
					paServed(t, fx.infer(t, paModel, http.StatusOK), "z-available")
					paServed(t, fx.infer(t, nodepolicy.TierStrong, http.StatusOK), "z-available")
					if a.ingress.Load() != 2 || a.inference.Load() != 2 || b.ingress.Load() != 2 || b.inference.Load() != 2 {
						t.Fatalf("paused/eligible ingress and engine counts = %d/%d and %d/%d, want 2/2 each", a.ingress.Load(), a.inference.Load(), b.ingress.Load(), b.inference.Load())
					}
				})
			}
			test("selected local pause", nodepolicy.PausedLocally, true)
			test("priority peer pause", nodepolicy.PausedByPeer, false)
			test("selected future provenance", nodepolicy.PauseOrigin("future-origin"), true)
		})
	}
}

func TestPAModelLessPausedPeerEligibility(t *testing.T) {
	for _, profile := range append(buildProfiles(), strataProfile(t)) {
		t.Run(profile.Name, func(t *testing.T) {
			test := func(name string, hint nodepolicy.PauseOrigin, selected bool) {
				t.Run(name, func(t *testing.T) {
					fx := paNewFixture(t, profile)
					peers := fx.peers(t, "a-paused", "z-available")
					a, b := peers[0], peers[1]
					for _, peer := range peers {
						engine := paModelLessEngine(t, peer, peer.node.HostUUID)
						setBackend(t, peer.p.soleFacade(), engine.URL, true)
						policy := peer.p.admission.currentPolicy()
						policy.ServedModels = nil
						peer.p.admission.setPolicy(policy)
					}
					// Keep destination admission available: a leaked origin dispatch
					// succeeds visibly instead of being hidden by a destination veto.
					// Unrestricted serving above also accepts the omitted model.
					a.node.Policy.PausedBy = hint
					fx.replace(t, a.node, b.node)
					fx.p.SetPriority([]string{"a-paused", "z-available"})
					if selected {
						fx.f.SetSelected("a-paused")
					}
					response := fx.finish(t, fx.request(http.MethodPost, routeChatCompletions, paModelLessBody), http.StatusOK)
					if a.ingress.Load() != 0 || a.inference.Load() != 0 {
						t.Errorf("paused peer ingress/engine = %d/%d, want 0/0", a.ingress.Load(), a.inference.Load())
					}
					if b.ingress.Load() != 1 || b.inference.Load() != 1 {
						t.Errorf("available peer ingress/engine = %d/%d, want 1/1", b.ingress.Load(), b.inference.Load())
					}
					paModelLessServed(t, response, profile, "z-available")
				})
			}
			test("selected local pause", nodepolicy.PausedLocally, true)
			test("priority peer pause", nodepolicy.PausedByPeer, false)
			test("selected future provenance", nodepolicy.PauseOrigin("future-origin"), true)
		})
	}
}

func TestPAModelLessRouteExceptions(t *testing.T) {
	for _, profile := range append(buildProfiles(), strataProfile(t)) {
		t.Run(profile.Name, func(t *testing.T) {
			t.Run("available peer without inventory", func(t *testing.T) {
				fx := paNewFixture(t, profile)
				peer := fx.peers(t, "available-peer")[0]
				engine := paModelLessEngine(t, peer, "available-peer")
				setBackend(t, peer.p.soleFacade(), engine.URL, true)
				policy := peer.p.admission.currentPolicy()
				policy.ServedModels = nil
				peer.p.admission.setPolicy(policy)
				peer.node.ModelsByEngine = nil
				fx.replace(t, peer.node)
				response := fx.finish(t, fx.request(http.MethodPost, routeChatCompletions, paModelLessBody), http.StatusOK)
				paModelLessServed(t, response, profile, "available-peer")
				if peer.ingress.Load() != 1 || peer.inference.Load() != 1 {
					t.Fatalf("available peer ingress/engine = %d/%d, want 1/1", peer.ingress.Load(), peer.inference.Load())
				}
			})
			t.Run("manual without hint or inventory", func(t *testing.T) {
				fx := paNewFixture(t, profile)
				e := &paEndpoint{}
				engine := paModelLessEngine(t, e, "manual")
				fx.f.discovery.AddManual(nodeFor(t, "manual", engine.URL))
				response := fx.finish(t, fx.request(http.MethodPost, routeChatCompletions, paModelLessBody), http.StatusOK)
				paModelLessServed(t, response, profile, "manual")
				if e.inference.Load() != 1 {
					t.Fatalf("manual engine requests = %d, want 1", e.inference.Load())
				}
			})
			t.Run("self ignores advertised pause hint", func(t *testing.T) {
				fx := paNewFixture(t, profile)
				e := &paEndpoint{}
				engine := paModelLessEngine(t, e, "test-origin")
				setBackend(t, fx.f, engine.URL, true)
				n := selfNode("test-origin", fx.f.port)
				n.PausedBy = nodepolicy.PausedLocally
				fx.f.discovery.SetSubscribed([]Node{n})
				response := fx.finish(t, fx.request(http.MethodPost, routeChatCompletions, paModelLessBody), http.StatusOK)
				paModelLessServed(t, response, profile, "test-origin")
				if e.inference.Load() != 1 {
					t.Fatalf("self engine requests = %d, want 1", e.inference.Load())
				}
			})
			t.Run("paused peer inventory and model-less passthrough", func(t *testing.T) {
				fx := paNewFixture(t, profile)
				peer := fx.peers(t, "paused-peer")[0]
				peer.node.Policy.PausedBy = nodepolicy.PausedLocally
				fx.replace(t, peer.node)
				response := fx.finish(t, fx.request(http.MethodGet, "/v1/models", ""), http.StatusOK)
				var inventory struct {
					Object string `json:"object"`
					Data   []struct {
						ID string `json:"id"`
					} `json:"data"`
				}
				if err := json.Unmarshal(response.body, &inventory); err != nil {
					t.Fatalf("decode paused peer inventory: %v", err)
				}
				var ids []string
				for _, item := range inventory.Data {
					ids = append(ids, item.ID)
				}
				if inventory.Object != "list" || !slices.Contains(ids, paModel) {
					t.Fatalf("paused peer inventory object/ids = %q/%v, want list including %s", inventory.Object, ids, paModel)
				}
				response = fx.finish(t, fx.request(http.MethodPost, "/pa-control", paModelLessBody), http.StatusOK)
				var control struct {
					ID string `json:"id"`
				}
				if err := json.Unmarshal(response.body, &control); err != nil {
					t.Fatalf("decode paused peer control: %v", err)
				}
				if control.ID != "paused-peer" || peer.inference.Load() != 0 || peer.list.Load() != 1 || peer.control.Load() != 1 || peer.ingress.Load() != 2 {
					t.Fatalf("paused peer control/inference/list/passthrough/ingress = %s/%d/%d/%d/%d, want paused-peer/0/1/1/2", control.ID, peer.inference.Load(), peer.list.Load(), peer.control.Load(), peer.ingress.Load())
				}
			})
		})
	}
}

func TestPAModelLessPausedDuplicateDoesNotClaimHost(t *testing.T) {
	for _, profile := range append(buildProfiles(), strataProfile(t)) {
		t.Run(profile.Name, func(t *testing.T) {
			fx := paNewFixture(t, profile)
			peers := fx.peers(t, "a-paused", "z-available")
			peer := peers[1]
			engine := paModelLessEngine(t, peer, "z-available")
			setBackend(t, peer.p.soleFacade(), engine.URL, true)
			serving := peer.p.admission.currentPolicy()
			serving.ServedModels = nil
			peer.p.admission.setPolicy(serving)
			// Distinct pinned principals share one advertised target. The paused
			// record must not claim that host before the available peer is tried.
			paused := peers[0].node
			paused.Services = peer.node.Services
			paused.Policy.PausedBy = nodepolicy.PausedLocally
			fx.replace(t, paused, peer.node)
			fx.f.SetSelected("a-paused")
			response := fx.finish(t, fx.request(http.MethodPost, routeChatCompletions, paModelLessBody), http.StatusOK)
			paModelLessServed(t, response, profile, "z-available")
			if peer.ingress.Load() != 1 || peer.inference.Load() != 1 {
				t.Fatalf("deduplicated peer ingress/engine = %d/%d, want 1/1", peer.ingress.Load(), peer.inference.Load())
			}
		})
	}
}

func TestPAModelLessAllPausedResponse(t *testing.T) {
	for _, profile := range append(buildProfiles(), strataProfile(t)) {
		t.Run(profile.Name, func(t *testing.T) {
			fx := paNewFixture(t, profile)
			peer := fx.peers(t, "paused-peer")[0]
			engine := paModelLessEngine(t, peer, "paused-peer")
			setBackend(t, peer.p.soleFacade(), engine.URL, true)
			policy := peer.p.admission.currentPolicy()
			policy.ServedModels = nil
			peer.p.admission.setPolicy(policy)
			peer.node.Policy.PausedBy = nodepolicy.PausedByPeer
			fx.replace(t, peer.node)
			response := fx.finish(t, fx.request(http.MethodPost, routeChatCompletions, paModelLessBody), http.StatusBadGateway)
			var rejection struct {
				Error string `json:"error"`
			}
			if err := json.Unmarshal(response.body, &rejection); err != nil {
				t.Fatalf("decode model-less empty-candidate rejection: %v", err)
			}
			if rejection.Error != "no active node selected or available" || peer.ingress.Load() != 0 || peer.inference.Load() != 0 {
				t.Fatalf("model-less rejection/paused ingress/engine = %q/%d/%d, want no active node selected or available/0/0", rejection.Error, peer.ingress.Load(), peer.inference.Load())
			}
		})
	}
}

func TestPAPausedPeerDynamicResume(t *testing.T) {
	for _, profile := range append(buildProfiles(), strataProfile(t)) {
		t.Run(profile.Name, func(t *testing.T) {
			fx := paNewFixture(t, profile)
			peer := fx.peers(t, "same-peer")[0]
			fx.replace(t, peer.node)
			roles := fx.f.declaredRoles(fx.p.admission.currentPolicy())
			catalogue := func(t *testing.T) []string {
				t.Helper()
				response := fx.finish(t, fx.request(http.MethodGet, "/v1/models", ""), http.StatusOK)
				var inventory struct {
					Data []struct {
						ID string `json:"id"`
					} `json:"data"`
				}
				if err := json.Unmarshal(response.body, &inventory); err != nil {
					t.Fatalf("decode catalogue: %v", err)
				}
				var ids []string
				for _, item := range inventory.Data {
					ids = append(ids, item.ID)
				}
				slices.Sort(ids)
				return ids
			}
			wantCatalogue := []string{paModel, "strong", "strong-api", "strong-local"}
			if got := catalogue(t); !slices.Equal(got, wantCatalogue) {
				t.Fatalf("initial catalogue = %v, want %v", got, wantCatalogue)
			}
			paCandidates(t, fx.f, "same-peer")
			paServed(t, fx.infer(t, paModel, http.StatusOK), "same-peer")
			paServed(t, fx.infer(t, nodepolicy.TierStrong, http.StatusOK), "same-peer")
			for _, step := range []struct {
				name, revision string
				hint           nodepolicy.PauseOrigin
				hits           int32
			}{
				{"paused", "2", nodepolicy.PausedByPeer, 2},
				{"resumed", "3", "", 4},
			} {
				t.Run(step.name, func(t *testing.T) {
					before := len(frames(t, fx.w))
					peer.node.Policy.Revision, peer.node.Policy.PausedBy = step.revision, step.hint
					fx.replace(t, peer.node)
					if got := len(frames(t, fx.w)); got != before {
						t.Fatalf("metadata-only replacement emitted %d notifications", got-before)
					}
					if step.hint != "" {
						paCandidates(t, fx.f)
						fx.infer(t, paModel, http.StatusBadGateway)
						fx.infer(t, nodepolicy.TierStrong, http.StatusNotFound)
					} else {
						paCandidates(t, fx.f, "same-peer")
						paServed(t, fx.infer(t, paModel, http.StatusOK), "same-peer")
						paServed(t, fx.infer(t, nodepolicy.TierStrong, http.StatusOK), "same-peer")
					}
					if got := catalogue(t); !slices.Equal(got, wantCatalogue) {
						t.Errorf("catalogue = %v, want %v", got, wantCatalogue)
					}
					if got := fx.f.declaredRoles(fx.p.admission.currentPolicy()); !reflect.DeepEqual(got, roles) {
						t.Errorf("declared roles changed: %+v, want %+v", got, roles)
					}
					if peer.inference.Load() != step.hits {
						t.Errorf("engine inference requests = %d, want %d", peer.inference.Load(), step.hits)
					}
					if peer.ingress.Load() != step.hits+peer.list.Load() {
						t.Errorf("peer ingress = %d, want inference %d + inventory %d", peer.ingress.Load(), step.hits, peer.list.Load())
					}
				})
			}
			if peer.list.Load() != 3 {
				t.Fatalf("catalogue HTTP calls = %d, want 3", peer.list.Load())
			}
		})
	}
}

func TestPAPauseRouteExceptions(t *testing.T) {
	for _, profile := range append(buildProfiles(), strataProfile(t)) {
		t.Run(profile.Name, func(t *testing.T) {
			t.Run("manual without hint", func(t *testing.T) {
				fx := paNewFixture(t, profile)
				e := &paEndpoint{}
				engine := paEngine(t, e, profile, "manual")
				fx.f.discovery.AddManual(nodeForModel(t, "manual", engine.URL, paModel))
				paServed(t, fx.infer(t, paModel, http.StatusOK), "manual")
				if e.inference.Load() != 1 {
					t.Fatalf("manual engine requests = %d, want 1", e.inference.Load())
				}
			})
			t.Run("paused self reaches local admission", func(t *testing.T) {
				fx := paNewFixture(t, profile)
				e := &paEndpoint{}
				engine := paEngine(t, e, profile, "test-origin")
				setBackend(t, fx.f, engine.URL, true)
				n := selfNode("test-origin", fx.f.port, paModel)
				n.PausedBy = nodepolicy.PausedLocally
				fx.f.discovery.SetSubscribed([]Node{n})
				paServed(t, fx.infer(t, paModel, http.StatusOK), "test-origin")
				fx.p.admission.setAvailability(nodepolicy.Paused, false)
				candidates := fx.f.resolveCandidates(paModel)
				if len(candidates) != 1 || !candidates[0].self {
					t.Fatalf("paused self candidates = %+v", candidates)
				}
				paPaused(t, fx.infer(t, paModel, http.StatusServiceUnavailable))
				if e.inference.Load() != 1 {
					t.Fatalf("paused self engine requests = %d, want positive control only", e.inference.Load())
				}
			})
			dormant := func(name string, pauseWhileWaking bool) {
				t.Run(name, func(t *testing.T) {
					fx := paNewFixture(t, profile)
					e := &paEndpoint{}
					engine := paEngine(t, e, profile, "test-origin")
					setBackend(t, fx.f, engine.URL, false)
					backend := fx.f.currentLocalBackend()
					backend.Healthy = true
					n := selfNode("test-origin", fx.f.port, paModel)
					n.PausedBy = nodepolicy.PausedByPeer
					fx.f.discovery.SetSubscribed([]Node{n})
					fx.f.noteSelfFromSnapshot([]Node{n})
					fx.f.discovery.SetSubscribed(nil)
					fx.p.admission.setIntent(map[string]bool{profile.Name: true})
					pol := nodepolicy.Default()
					pol.Idle.StartOnDemand, pol.Idle.WakeTimeoutSeconds = true, 2
					fx.p.admission.setPolicy(pol)
					var wakes atomic.Int32
					fx.p.admission.notify = func(method string, _ any) {
						if method != nodepolicy.NotifyAdmissionWake {
							return
						}
						wakes.Add(1)
						if pauseWhileWaking {
							fx.p.admission.setAvailability(nodepolicy.Paused, false)
						} else if err := fx.f.setLocalBackend(backend); err != nil {
							t.Errorf("wake fixture engine: %v", err)
						}
					}
					candidates := fx.f.resolveCandidates(paModel)
					if len(candidates) != 1 || !candidates[0].self || !candidates[0].dormant {
						t.Fatalf("dormant self candidates = %+v", candidates)
					}
					if pauseWhileWaking {
						paPaused(t, fx.infer(t, paModel, http.StatusServiceUnavailable))
						if e.inference.Load() != 0 {
							t.Fatal("paused dormant self reached engine")
						}
					} else {
						paServed(t, fx.infer(t, paModel, http.StatusOK), "test-origin")
						if e.inference.Load() != 1 {
							t.Fatalf("woken engine requests = %d, want 1", e.inference.Load())
						}
					}
					if wakes.Load() != 1 {
						t.Fatalf("local admission wake requests = %d, want 1", wakes.Load())
					}
				})
			}
			dormant("dormant self wakes through local admission", false)
			dormant("dormant self pause during admission", true)
			t.Run("paused peer inventory and passthrough", func(t *testing.T) {
				fx := paNewFixture(t, profile)
				e := fx.peers(t, "paused-peer")[0]
				fx.replace(t, e.node)
				paServed(t, fx.infer(t, paModel, http.StatusOK), "paused-peer")
				e.node.Policy.PausedBy = nodepolicy.PausedLocally
				fx.replace(t, e.node)
				fx.finish(t, fx.request(http.MethodGet, "/v1/models", ""), http.StatusOK)
				response := fx.finish(t, fx.request(http.MethodPost, "/pa-control", `{"model":"pa-model"}`), http.StatusOK)
				var control struct {
					ID string `json:"id"`
				}
				if err := json.Unmarshal(response.body, &control); err != nil {
					t.Fatalf("decode control response: %v", err)
				}
				if control.ID != "paused-peer" || e.inference.Load() != 1 || e.list.Load() != 1 || e.control.Load() != 1 || e.ingress.Load() != 3 {
					t.Fatalf("paused peer control/inference/list/passthrough/ingress = %s/%d/%d/%d/%d", control.ID, e.inference.Load(), e.list.Load(), e.control.Load(), e.ingress.Load())
				}
			})
			t.Run("paused duplicate does not claim host", func(t *testing.T) {
				fx := paNewFixture(t, profile)
				peers := fx.peers(t, "a-paused", "z-available")
				a, b := peers[0], peers[1]
				fx.replace(t, a.node, b.node)
				fx.f.SetSelected("a-paused")
				paServed(t, fx.infer(t, paModel, http.StatusOK), "a-paused")
				fx.f.SetSelected("z-available")
				paServed(t, fx.infer(t, paModel, http.StatusOK), "z-available")
				// A stale endpoint on the earlier paused record now names the
				// available peer's destination. Its distinct real pin remains.
				a.node.Services = b.node.Services
				a.node.Policy.PausedBy = nodepolicy.PausedLocally
				fx.replace(t, a.node, b.node)
				fx.f.SetSelected("a-paused")
				paCandidates(t, fx.f, "z-available")
				paServed(t, fx.infer(t, paModel, http.StatusOK), "z-available")
				paServed(t, fx.infer(t, nodepolicy.TierStrong, http.StatusOK), "z-available")
				if a.ingress.Load() != 1 || a.inference.Load() != 1 || b.ingress.Load() != 3 || b.inference.Load() != 3 {
					t.Fatalf("duplicate ingress/engine counts = %d/%d and %d/%d", a.ingress.Load(), a.inference.Load(), b.ingress.Load(), b.inference.Load())
				}
			})
		})
	}
}

func TestPAAllPausedResponses(t *testing.T) {
	for _, profile := range append(buildProfiles(), strataProfile(t)) {
		t.Run(profile.Name, func(t *testing.T) {
			fx := paNewFixture(t, profile)
			peers := fx.peers(t, "a-paused", "b-paused")
			fx.replace(t, peers[0].node, peers[1].node)
			for _, e := range peers {
				fx.f.SetSelected(e.node.HostUUID)
				paServed(t, fx.infer(t, paModel, http.StatusOK), e.node.HostUUID)
			}
			peers[0].node.Policy.PausedBy = nodepolicy.PausedLocally
			peers[1].node.Policy.PausedBy = nodepolicy.PausedByPeer
			fx.replace(t, peers[0].node, peers[1].node)
			paCandidates(t, fx.f)
			test := func(name, model string, status int, wantError string, wantCapabilities bool) {
				t.Run(name, func(t *testing.T) {
					response := fx.infer(t, model, status)
					var rejection struct {
						Error string          `json:"error"`
						Needs json.RawMessage `json:"needs"`
					}
					if err := json.Unmarshal(response.body, &rejection); err != nil {
						t.Fatalf("decode empty-owner rejection: %v", err)
					}
					if rejection.Error != wantError || (rejection.Needs != nil) != wantCapabilities || response.header.Get(nodepolicy.AdmissionHeader) != "" || response.header.Get("Retry-After") != "" {
						t.Fatalf("empty-owner rejection = %+v, headers=%v", rejection, response.header)
					}
					for _, e := range peers {
						if e.ingress.Load() != 1 || e.inference.Load() != 1 {
							t.Errorf("paused peer %s ingress/engine = %d/%d, want positive control only", e.node.HostUUID, e.ingress.Load(), e.inference.Load())
						}
					}
				})
			}
			test("concrete retains 502", paModel, http.StatusBadGateway, "no available node advertises the requested model", false)
			test("tier retains 404", nodepolicy.TierStrong, http.StatusNotFound, "no eligible model for tier strong", true)
		})
	}
}

func TestPAStaleHintDestinationRejection(t *testing.T) {
	for _, profile := range append(buildProfiles(), strataProfile(t)) {
		t.Run(profile.Name, func(t *testing.T) {
			for _, model := range []string{paModel, nodepolicy.TierStrong} {
				t.Run(model, func(t *testing.T) {
					fx := paNewFixture(t, profile)
					peers := fx.peers(t, "a-stale", "z-available")
					a, b := peers[0], peers[1]
					fx.replace(t, a.node, b.node)
					fx.f.SetSelected("a-stale")
					paServed(t, fx.infer(t, model, http.StatusOK), "a-stale")
					a.p.admission.setAvailability(nodepolicy.Paused, false)
					paCandidates(t, fx.f, "a-stale", "z-available")
					entered, release := make(chan struct{}), make(chan struct{})
					a.beforeMu.Lock()
					a.before = func() { close(entered); <-release }
					a.beforeMu.Unlock()
					defer func() {
						select {
						case <-release:
						default:
							close(release)
						}
					}()
					result := make(chan paResponse, 1)
					go func() {
						result <- fx.request(http.MethodPost, routeChatCompletions, fmt.Sprintf(`{"model":%q,"stream":false}`, model))
					}()
					waitChan(t, entered, "stale available snapshot selected for HTTP dispatch")
					if a.ingress.Load() != 2 || a.inference.Load() != 1 || b.ingress.Load() != 0 {
						t.Fatalf("before destination admission ingress/engine/fallback = %d/%d/%d", a.ingress.Load(), a.inference.Load(), b.ingress.Load())
					}
					stored, ok := discoveredNode(fx.f, "a-stale")
					if !ok || stored.PolicyRevision != "1" || stored.PausedBy != "" {
						t.Fatalf("stale available snapshot changed: %+v", stored)
					}
					close(release)
					select {
					case response := <-result:
						paServed(t, fx.finish(t, response, http.StatusOK), "z-available")
					case <-time.After(5 * time.Second):
						t.Fatal("stale hint failover did not finish")
					}
					select {
					case rejection := <-a.rejections:
						if rejection.status != http.StatusServiceUnavailable || rejection.header.Get(nodepolicy.AdmissionHeader) != "paused" || rejection.header.Get("Retry-After") != "30" {
							t.Fatalf("destination rejection = %+v", rejection)
						}
					default:
						t.Fatal("no actual paused destination admission rejection")
					}
					if a.ingress.Load() != 2 || a.inference.Load() != 1 || b.ingress.Load() != 1 || b.inference.Load() != 1 {
						t.Fatalf("stale rejection/failover ingress/engine = %d/%d and %d/%d", a.ingress.Load(), a.inference.Load(), b.ingress.Load(), b.inference.Load())
					}
				})
			}
		})
	}
}
