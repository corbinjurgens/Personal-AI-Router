// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptrace"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// These cases exercise production service, broker, proxy and manager processes.
// Only native engine HTTP and multicast discovery are fixtures. Discovery copies
// the real manager inventory/projection; a binding never manufactures inventory.
type engineBindingFixture struct {
	service *remoteServiceFixture
	engine  *switchProcessFixture
	client  *serviceRPCClient
	args    []string
	nodeID  string
	started time.Time
}

func newEngineBindingFixture(t *testing.T, configure func(*switchProcessFixture)) *engineBindingFixture {
	t.Helper()
	s := newRemoteServiceFixture(t)
	return newEngineBindingServiceFixture(t, s, configure)
}

func newEngineBindingServiceFixture(t *testing.T, s *remoteServiceFixture, configure func(*switchProcessFixture)) *engineBindingFixture {
	t.Helper()
	e := prepareSwitchProcessFixture(t, s.configDir, s.ports, s.scannerPath)
	if configure != nil {
		configure(e)
	}
	// The service owns a separate config root and cluster root. Its broker's
	// settings journal is beside clusterDir, rather than inside configDir.
	var journal map[string]json.RawMessage
	task2Decode(t, readPolicyBytes(t, filepath.Join(s.configDir, "engine-settings-operations.json")), &journal)
	jointWriteJSON(t, filepath.Join(filepath.Dir(s.clusterDir), "engine-settings-operations.json"), journal)
	facades := map[string]int{"ollama": e.facadePort, "llamacpp": e.targetFacadePort}
	for _, name := range engines.Names() {
		if _, exists := facades[name]; !exists {
			facades[name] = freePort(t)
		}
	}
	writePrivateBrokerEngineSettings(t, s.configDir, s.clusterDir, facades)
	s.proxyPath = s.copyBinary(proxyBin)
	s.inventory.paths = append(s.inventory.paths, s.proxyPath)
	s.facadePort = e.targetFacadePort
	s.env = append(s.env, "NVPAIR_TEST_SUBSCRIPTION_SCANNER="+s.scannerPath,
		"NVPAIR_TEST_SUBSCRIPTION_PORT="+strconv.Itoa(e.scannerPort))
	f := &engineBindingFixture{service: s, engine: e, args: s.args(), nodeID: clustertrust.Open(s.clusterDir).NodeUUID()}
	if f.args[len(f.args)-2] != "--proxy-engines" || f.args[len(f.args)-1] != "llamacpp" {
		t.Fatal("service fixture no longer supplies the expected proxy argument")
	}
	f.args[len(f.args)-1] = "ollama,llamacpp"
	f.start(t)
	t.Cleanup(func() {
		e.unblock()
		err := s.cleanupChecked()
		if t.Failed() {
			// The existing owner has joined the stderr pumps; capture before
			// removeBundle deletes the private log directory.
			logBindingRefusalDecisions(t, s.serviceFixture)
		}
		if err != nil {
			t.Errorf("binding service cleanup: %v", err)
			return
		}
		s.assertStopped()
		e.mu.Lock()
		identities := append([]serviceProcessIdentity(nil), e.identities...)
		e.mu.Unlock()
		for _, identity := range identities {
			assertSwitchProcessIdentityStopped(t, identity)
		}
		for _, port := range []int{e.sourcePort, e.targetPort, e.facadePort, e.targetFacadePort, e.scannerPort} {
			assertInteropListenerClosed(t, port)
		}
	})
	return f
}

func logBindingRefusalDecisions(t *testing.T, s *serviceFixture) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(s.dir, "logs", "broker.log"))
	if err != nil {
		t.Logf("binding permission decision log read: %v", err)
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.Contains(line, "remote management decision") || strings.Contains(line, "bound engine management refused") {
			t.Logf("binding permission decision: %s", line)
		}
	}
}

func (f *engineBindingFixture) grant(t *testing.T, peer string, allowed bool) {
	t.Helper()
	var before managementSnapshot
	task2Decode(t, f.call(t, "cluster:get-remote-management", map[string]any{}), &before)
	var after managementSnapshot
	task2Decode(t, f.call(t, "cluster:set-remote-management", map[string]any{"peerUuid": peer, "allowed": allowed, "expectedRevision": before.Revision}), &after)
	wantState := "denied"
	if allowed {
		wantState = "allowed"
	}
	for _, grant := range after.Grants {
		if grant.PeerUUID == peer && grant.Allowed == allowed && grant.State == wantState && after.Revision != before.Revision {
			return
		}
	}
	t.Fatal("service did not acknowledge the target-local management grant")
}

func (f *engineBindingFixture) start(t *testing.T) {
	t.Helper()
	f.started = time.Now()
	f.service.service = f.service.startChild(f.args...)
	f.client = f.service.readyClient()
	for _, name := range engines.Names() {
		f.call(t, "engine:get-settings", map[string]string{"engine": name})
	}
}

func (f *engineBindingFixture) call(t *testing.T, method string, params any) json.RawMessage {
	t.Helper()
	return f.client.mustCall(t, strconv.FormatInt(newRPCID(), 10), method, mustMarshal(t, params)).Result
}

func (f *engineBindingFixture) bind(t *testing.T, bindings nodepolicy.EngineBindings) nodepolicy.Policy {
	t.Helper()
	want := fetchServicePolicy(t, f.client).Policy
	want.EngineBindings = bindings
	want = setServicePolicy(t, f.client, want)
	task1Eventually(t, "binding policy acknowledged by the real proxy", 15*time.Second, func() bool {
		snapshot := fetchServicePolicy(t, f.client)
		return snapshot.ActiveRevision == want.Revision && snapshot.Activation == "active"
	})
	return want
}

func (f *engineBindingFixture) targetBinding(t *testing.T) nodepolicy.Policy {
	t.Helper()
	return f.bind(t, nodepolicy.EngineBindings{{Engine: "llamacpp", Model: "switch-model"}})
}

// Relay a genuine inventory read through the existing scanner fixture. This is
// deliberately not independent production-scanner enrichment coverage.
func (f *engineBindingFixture) publish(t *testing.T, requireOffer bool) noderec.DirectoryNode {
	t.Helper()
	var inventory struct {
		Models   []string                  `json:"models"`
		ByEngine map[string][]string       `json:"modelsByEngine"`
		Loaded   map[string][]string       `json:"loadedByEngine"`
		Policy   *nodepolicy.Advertisement `json:"policy"`
	}
	task1Eventually(t, "target-owned inventory and binding offer", 15*time.Second, func() bool {
		task2Decode(t, f.call(t, "engine:models", nil), &inventory)
		return inventory.Policy != nil && (!requireOffer || inventory.Policy.EngineBindingOffers != nil)
	})
	node := noderec.DirectoryNode{
		HostUUID: f.nodeID, ClusterUUID: f.nodeID, Name: "binding-target-" + inventory.Policy.Revision, IP: "127.0.0.1", Trusted: true,
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{
			noderec.ServiceOllama: {Port: f.engine.facadePort}, noderec.ServiceLlamaCPP: {Port: f.engine.targetFacadePort},
		}, Models: inventory.Models, ModelsByEngine: inventory.ByEngine, LoadedByEngine: inventory.Loaded, Policy: inventory.Policy,
	}
	f.publishDirectory(t, node)
	for _, engine := range []string{"ollama", "llamacpp"} {
		if len(inventory.ByEngine[engine]) != 0 {
			f.waitHealthy(t, engine)
		}
	}
	return node
}

func (f *engineBindingFixture) waitHealthy(t *testing.T, engine string) {
	t.Helper()
	port, path := f.engine.targetFacadePort, "/v1/models"
	if engine == "ollama" {
		port, path = f.engine.facadePort, "/api/tags"
	}
	if f.service.peer == nil {
		t.Fatal("health barrier requires a pinned peer client")
	}
	task1Eventually(t, "broker publishes healthy local "+engine+" to pinned ingress", 15*time.Second, func() bool {
		response, err := f.service.peer.Get(fmt.Sprintf("https://127.0.0.1:%d%s", port, path))
		if err != nil {
			t.Fatalf("pinned health barrier: %v", err)
		}
		if _, err := io.Copy(io.Discard, response.Body); err != nil {
			t.Fatal(err)
		}
		if err := response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		return response.StatusCode == http.StatusOK
	})
}

func (f *engineBindingFixture) publishDirectory(t *testing.T, node noderec.DirectoryNode) {
	t.Helper()
	frame := jsonrpc.Message{JSONRPC: "2.0", Method: noderec.NotifyNodeUpdated, Params: mustMarshal(t, noderec.NodeEvent{Node: node})}
	client := &http.Client{Timeout: 5 * time.Second}
	defer client.CloseIdleConnections()
	response, err := client.Post(fmt.Sprintf("http://127.0.0.1:%d", f.engine.scannerPort), "application/json", bytes.NewReader(mustMarshal(t, frame)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(io.Discard, response.Body); err != nil {
		t.Fatal(err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("scanner publication HTTP %d", response.StatusCode)
	}
	for _, engine := range []string{"ollama", "llamacpp"} {
		identity, ok := engines.ByName(engine)
		if !ok {
			t.Fatalf("unknown fixture engine %s", engine)
		}
		task1Eventually(t, "proxy observed target inventory for "+engine, 10*time.Second, func() bool {
			var result struct {
				Nodes []struct {
					ID     string   `json:"id"`
					Host   string   `json:"host"`
					Port   int      `json:"port"`
					Models []string `json:"models"`
				} `json:"nodes"`
			}
			task2Decode(t, f.call(t, engine+"-proxy:nodes/list", nil), &result)
			for _, got := range result.Nodes {
				if got.ID == node.HostUUID && got.Host == node.Name && got.Port == node.Services[identity.DiscoveryService].Port && slices.Equal(got.Models, node.ModelsByEngine[engine]) {
					return true
				}
			}
			return false
		})
	}
}

type engineBindingResponse struct {
	status int
	header http.Header
	body   struct {
		Model   string `json:"model"`
		Code    string `json:"code"`
		Reason  string `json:"reason"`
		Error   string `json:"error"`
		Done    bool   `json:"done"`
		Choices []struct {
			Index   int `json:"index"`
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	err error
}

func engineBindingHTTP(ctx context.Context, client *http.Client, endpoint string, budget int) engineBindingResponse {
	var result engineBindingResponse
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(`{"model":"switch-model","stream":false,"messages":[]}`))
	if err != nil {
		result.err = err
		return result
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(nodepolicy.EnginePrepareBudgetHeader, strconv.Itoa(budget))
	request.Header.Set("X-PAIR-Peer-Id", "management-service-allowed")
	request.Header.Set("X-PAIR-Activation-Identity", "forged-fixture-authority")
	response, err := client.Do(request)
	if err != nil {
		result.err = err
		return result
	}
	result.status, result.header = response.StatusCode, response.Header.Clone()
	result.err = json.NewDecoder(io.LimitReader(response.Body, 64<<10)).Decode(&result.body)
	if err := response.Body.Close(); result.err == nil {
		result.err = err
	}
	return result
}

func (f *engineBindingFixture) request(t *testing.T, client *http.Client, port int, pinned bool, budget int) <-chan engineBindingResponse {
	t.Helper()
	done, _ := f.requestObserved(t, client, port, pinned, budget)
	return done
}

func (f *engineBindingFixture) requestObserved(t *testing.T, client *http.Client, port int, pinned bool, budget int) (<-chan engineBindingResponse, <-chan struct{}) {
	t.Helper()
	scheme := "http"
	if pinned {
		scheme = "https"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Second)
	t.Cleanup(cancel)
	firstByte := make(chan struct{}, 1)
	ctx = httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{GotFirstResponseByte: func() {
		select {
		case firstByte <- struct{}{}:
		default:
		}
	}})
	done := make(chan engineBindingResponse, 1)
	go func() {
		done <- engineBindingHTTP(ctx, client, fmt.Sprintf("%s://127.0.0.1:%d/v1/chat/completions", scheme, port), budget)
	}()
	return done, firstByte
}

func engineBindingAwait(t *testing.T, done <-chan engineBindingResponse) engineBindingResponse {
	t.Helper()
	select {
	case response := <-done:
		if response.err != nil {
			t.Fatalf("binding inference transport/decode: %v", response.err)
		}
		return response
	case <-time.After(100 * time.Second):
		t.Fatal("binding inference exceeded HTTP request bound")
	}
	return engineBindingResponse{}
}

func assertEngineBindingSuccess(t *testing.T, response engineBindingResponse, engine, node string) {
	t.Helper()
	if response.status != http.StatusOK || response.body.Error != "" || response.body.Model != "switch-model" || len(response.body.Choices) != 1 || response.body.Choices[0].Index != 0 || response.body.Choices[0].Message.Role != "assistant" || response.body.Choices[0].Message.Content != "finished" || response.body.Choices[0].FinishReason != "stop" || response.header.Get(nodepolicy.EngineHeader) != engine || response.header.Get(nodepolicy.NodeHeader) != node || response.header.Get(nodepolicy.ModelHeader) != "switch-model" {
		t.Fatalf("binding success HTTP=%d engine=%q node=%q model=%q choices=%d admission=%q code=%q reason=%q error=%q", response.status, response.header.Get(nodepolicy.EngineHeader), response.header.Get(nodepolicy.NodeHeader), response.body.Model, len(response.body.Choices), response.header.Get(nodepolicy.AdmissionHeader), response.body.Code, response.body.Reason, response.body.Error)
	}
}

func assertEngineBindingRefusal(t *testing.T, response engineBindingResponse, reason string) {
	t.Helper()
	if response.status != http.StatusServiceUnavailable || response.header.Get(nodepolicy.AdmissionHeader) != reason || response.body.Code != "admission" || response.body.Reason != reason || response.body.Error != "Could not prepare the engine for this model within the request budget." || len(response.body.Choices) != 0 {
		t.Fatalf("binding refusal HTTP=%d header=%q code=%q reason=%q choices=%d, want %q", response.status, response.header.Get(nodepolicy.AdmissionHeader), response.body.Code, response.body.Reason, len(response.body.Choices), reason)
	}
}

func (f *engineBindingFixture) status(t *testing.T, id string) settings.SwitchStatus {
	t.Helper()
	var result settings.SwitchStatus
	task2Decode(t, f.call(t, settings.SwitchStatusMethod, settings.SwitchStatusRequest{RequestID: id}), &result)
	// An unqualified status returns only the active ID. Read that receipt by
	// its actual ID before asserting a phase or retaining it for recovery.
	if id == "" && result.ActiveRequestID != "" {
		var active settings.SwitchStatus
		task2Decode(t, f.call(t, settings.SwitchStatusMethod, settings.SwitchStatusRequest{RequestID: result.ActiveRequestID}), &active)
		return active
	}
	return result
}

func (f *engineBindingFixture) journal(t *testing.T) settings.SwitchJournal {
	t.Helper()
	journal, err := settings.DecodeSwitchJournal(readLiveDurableFile(t, filepath.Join(filepath.Dir(f.service.clusterDir), "engine-switch-operations.json")))
	if err != nil {
		t.Fatalf("decode real switch journal: %v", err)
	}
	return journal
}

func (f *engineBindingFixture) assertEffects(t *testing.T, sourceStarts, targetStarts, sourceInference, targetInference int) {
	t.Helper()
	e := f.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.effectError != nil {
		t.Fatalf("fixture effect observation: %v", e.effectError)
	}
	if e.checkSourceExit && e.sourceAliveAtTarget {
		t.Fatal("target started before actual source process exit")
	}
	if len(e.starts["ollama"]) != sourceStarts || len(e.starts["llamacpp"]) != targetStarts || e.inferences["ollama"] != sourceInference || e.inferences["llamacpp"] != targetInference {
		t.Fatalf("engine effects starts=%v inference=%v, want starts=%d/%d inference=%d/%d", e.starts, e.inferences, sourceStarts, targetStarts, sourceInference, targetInference)
	}
}

// Uncovered process seams, intentionally not simulated as product owners:
//   - Legacy-target readback requires an actual older target binary.
//   - No observable barrier separates a private preparation notification's
//     receipt from its authority acceptance for removal/readmission injection.
//   - Delayed scoped cleanup cannot be retained and delivered after newer
//     manual work without intercepting the manager's private product channel.
//   - No public barrier proves a second same-target waiter joined preparation
//     before readiness release. Competing-target refusal and later ready reuse
//     are covered, but process-level coalescing remains unproven.
//
// Restart below proves recovery/no replay, not that last delayed-delivery edge.
// The scanner fixture relays real inventory, but does not prove independent
// production-scanner enrichment or multicast convergence.
func TestEngineBindingProcess(t *testing.T) {
	xaPlatform(t)
	t.Run("local and granted pinned CAS survive service restart", testEngineBindingPolicy)
	t.Run("ordinary demand drains old inference and proves source exit before target start", testEngineBindingDrain)
	t.Run("ready bound engine excludes the wrong engine for ordinary demand", testEngineBindingWrongEngine)
	t.Run("ready peer outranks a scheduler-preferred replacement", testEngineBindingReadyPeer)
	t.Run("inference-only peer cannot activate but can use a ready engine", testEngineBindingPermission)
	t.Run("cold target lock and pause prevent native effects", testEngineBindingVeto)
	t.Run("protected source resident prevents replacement", testEngineBindingProtectedResident)
	t.Run("cold preparation refuses missing model without inference", testEngineBindingMissingModel)
	t.Run("cold startup failure refuses without inference", testEngineBindingStartFailure)
	t.Run("accepted cold work refuses competing target", testEngineBindingCompeting)
	t.Run("settled automatic receipts cycle while a manual ID survives", testEngineBindingRing)
	t.Run("broker restart interrupts accepted replacement without replay", testEngineBindingRestart)
	t.Run("stopped offered facade is allowlisted for pinned direct inference", func(t *testing.T) { testEngineBindingPinned(t, false) })
	t.Run("stopped offered facade is allowlisted for pinned reverse inference", func(t *testing.T) { testEngineBindingPinned(t, true) })
}

func testEngineBindingPolicy(t *testing.T) {
	f := newEngineBindingFixture(t, nil)
	before := fetchServicePolicy(t, f.client).Policy
	want := f.targetBinding(t)
	stale := f.client.mustPolicyCallBased(t, strconv.FormatInt(newRPCID(), 10), before, before.Revision)
	if stale.Error == nil || stale.Error.Code != -32010 || len(stale.Result) != 0 {
		t.Fatalf("stale binding CAS = %+v", stale.Error)
	}
	assertStoredPolicy(t, f.service.policyPath(), want)
	view := decodeResult[managementSnapshot](t, *f.client.mustCall(t, strconv.FormatInt(newRPCID(), 10), "cluster:get-remote-management", mustMarshal(t, map[string]any{})))
	f.call(t, "cluster:set-remote-management", map[string]any{"peerUuid": "management-service-allowed", "allowed": true, "expectedRevision": view.Revision})
	peer := want.PeerProjection()
	peer.EngineBindings = nodepolicy.EngineBindings{{Engine: "ollama", Model: "other-model"}}
	data := managementHTTP(t, f.service.allowed, f.service.ports.engineControl, http.MethodPost, "/v1/node-policy/set", map[string]any{"policy": peer, "expectedPersistedRevision": want.Revision}, http.StatusOK)
	var accepted struct {
		Policy nodepolicy.PeerPolicy `json:"policy"`
	}
	task2Decode(t, data, &accepted)
	if accepted.Policy.Revision == want.Revision || !reflect.DeepEqual(accepted.Policy.EngineBindings, peer.EngineBindings) {
		t.Fatal("granted remote binding CAS did not persist the exact assignment")
	}
	want.EngineBindings, want.Revision = peer.EngineBindings, accepted.Policy.Revision
	assertPolicySnapshot(t, fetchServicePolicy(t, f.client), want, nodepolicy.Available)
	status, _, data := managementHTTPResponse(t, f.service.allowed, f.service.ports.engineControl, http.MethodPost, "/v1/node-policy/set", map[string]any{"policy": peer, "expectedPersistedRevision": before.Revision})
	var conflict struct {
		Code string `json:"code"`
	}
	task2Decode(t, data, &conflict)
	if status != http.StatusConflict || conflict.Code != "policy_conflict" {
		t.Fatalf("remote stale CAS HTTP=%d code=%q", status, conflict.Code)
	}
	f.grant(t, "management-service-peer", false)
	stored := assertStoredPolicy(t, f.service.policyPath(), want)
	managementHTTP(t, f.service.peer, f.service.ports.engineControl, http.MethodPost, "/v1/node-policy/set", map[string]any{"policy": peer, "expectedPersistedRevision": want.Revision}, http.StatusConflict)
	if !bytes.Equal(stored, readPolicyBytes(t, f.service.policyPath())) {
		t.Fatal("denied peer altered binding bytes")
	}
	if err := f.service.cleanupChecked(); err != nil {
		t.Fatal(err)
	}
	f.service.assertStopped()
	f.start(t)
	assertPolicySnapshot(t, fetchServicePolicy(t, f.client), want, nodepolicy.Available)
	assertStoredPolicy(t, f.service.policyPath(), want)
	f.assertEffects(t, 0, 0, 0, 0)
}

func testEngineBindingDrain(t *testing.T) {
	f := newEngineBindingFixture(t, func(e *switchProcessFixture) { e.checkSourceExit = true; e.holdTarget = true })
	e := f.engine
	f.call(t, "engine:start", map[string]string{"engine": "ollama"})
	sourcePID := e.pid(t, "ollama")
	f.publish(t, false)
	client := &http.Client{Timeout: 100 * time.Second}
	defer client.CloseIdleConnections()
	old := f.request(t, client, e.facadePort, false, nodepolicy.MaxBoundEngineBudgetMs)
	select {
	case <-e.inflight:
	case <-time.After(15 * time.Second):
		t.Fatal("old inference did not reach the source image")
	}
	f.targetBinding(t)
	f.publish(t, true)
	demand, output := f.requestObserved(t, client, e.targetFacadePort, false, nodepolicy.MaxBoundEngineBudgetMs)
	var active settings.SwitchStatus
	task1Eventually(t, "automatic replacement reaches the real drain after startup dwell", 80*time.Second, func() bool {
		active = f.status(t, "")
		return active.Receipt != nil && active.Receipt.Phase == settings.SwitchDraining
	})
	if !active.Receipt.Automatic || active.ActiveRequestID != active.Receipt.RequestID || active.Receipt.Request.SourceEngine != "ollama" || active.Receipt.Request.TargetEngine != "llamacpp" || active.Receipt.Caller != settings.SwitchOriginLocal {
		t.Fatalf("automatic drain identity=%+v", active)
	}
	if active.Receipt.AcceptedAt.Before(f.started.Add(time.Minute)) {
		t.Fatal("replacement bypassed the production startup dwell")
	}
	f.assertEffects(t, 1, 0, 1, 0)
	stopped, err := serviceProcessStopped(context.Background(), sourcePID)
	if err != nil || stopped {
		t.Fatalf("old inference source stopped before release: stopped=%t err=%v", stopped, err)
	}
	select {
	case response := <-demand:
		t.Fatalf("new inference responded before old work drained: HTTP=%d err=%v", response.status, response.err)
	default:
	}
	select {
	case <-output:
		t.Fatal("new inference emitted HTTP bytes before source drain")
	default:
	}
	e.unblockSource()
	assertEngineBindingSuccess(t, engineBindingAwait(t, old), "ollama", "management-service-target")
	select {
	case <-e.targetStarting:
	case <-time.After(20 * time.Second):
		t.Fatal("target did not start after source release")
	}
	assertSwitchProcessStopped(t, sourcePID)
	e.mu.Lock()
	sourceAliveAtTarget := e.sourceAliveAtTarget
	e.mu.Unlock()
	if sourceAliveAtTarget {
		t.Fatal("target started before actual source process exit")
	}
	f.assertEffects(t, 1, 1, 1, 0)
	select {
	case response := <-demand:
		t.Fatalf("new inference responded while target readiness was held: HTTP=%d err=%v", response.status, response.err)
	default:
	}
	select {
	case <-output:
		t.Fatal("new inference emitted HTTP bytes before target readiness")
	default:
	}
	e.unblockTarget()
	response := engineBindingAwait(t, demand)
	assertEngineBindingSuccess(t, response, "llamacpp", "management-service-target")
	// Successful inference exposes the final engine/model/node headers. The
	// switch ID is a refusal header; correlate completion with the exact active
	// receipt observed at the drain barrier instead.
	terminal := f.status(t, active.Receipt.RequestID)
	if terminal.Receipt == nil || !terminal.Receipt.Terminal || terminal.Receipt.Phase != settings.SwitchCompleted || terminal.Receipt.Error != nil || !terminal.Receipt.Source.TerminationVerified || terminal.Receipt.RecoveryRequired {
		t.Fatalf("automatic completion=%+v", terminal)
	}
	f.assertEffects(t, 1, 1, 1, 1)
	// Replacement dwell consumes the caller's allowance and must not start a
	// second process merely because another concrete model is now requested.
	f.bind(t, nodepolicy.EngineBindings{{Engine: "ollama", Model: "switch-model"}})
	f.publish(t, true)
	assertEngineBindingRefusal(t, engineBindingAwait(t, f.request(t, client, e.facadePort, false, 1000)), "engine-switch-cooldown")
	f.assertEffects(t, 1, 1, 1, 1)
}

func testEngineBindingWrongEngine(t *testing.T) {
	f := newEngineBindingFixture(t, nil)
	f.engine.unblock()
	for _, name := range []string{"ollama", "llamacpp"} {
		f.call(t, "engine:start", map[string]string{"engine": name})
	}
	f.targetBinding(t)
	f.publish(t, true)
	client := &http.Client{Timeout: 20 * time.Second}
	defer client.CloseIdleConnections()
	response := engineBindingAwait(t, f.request(t, client, f.engine.facadePort, false, 5000))
	assertEngineBindingSuccess(t, response, "llamacpp", "management-service-target")
	if response.header.Get(nodepolicy.SwitchRequestIDHeader) != "" {
		t.Fatal("ready bound engine unexpectedly created a replacement")
	}
	f.assertEffects(t, 1, 1, 0, 1)
}

func testEngineBindingReadyPeer(t *testing.T) {
	// One service bundle is available on the fixed-path manifest. The other
	// node reuses E4's registered broker stack, with its own pinned identity.
	s := newRemoteServiceFixture(t)
	base := shortClusterTempDir(t)
	cluster := filepath.Join(base, "cluster")
	if err := os.MkdirAll(cluster, 0o700); err != nil {
		t.Fatal(err)
	}
	const peerID = "zz-binding-ready"
	peerCert := mintClusterIdentity(t, cluster, peerID)
	targetCert, err := os.ReadFile(filepath.Join(s.clusterDir, "node.crt"))
	if err != nil {
		t.Fatal(err)
	}
	writeActiveAdmission(t, cluster)
	writePin(t, cluster, "management-service-target", string(targetCert))
	writeMembers(t, cluster, "management-service-target", peerID)
	jointWriteJSON(t, filepath.Join(cluster, "identity.json"), map[string]string{"node_uuid": peerID})
	writePin(t, s.clusterDir, peerID, peerCert)
	writeMembers(t, s.clusterDir, "management-service-target", "management-service-peer", "management-service-allowed", peerID)
	f := newEngineBindingServiceFixture(t, s, nil)
	peer := newSwitchProcessFixtureAt(t, base, reserveFixtureServicePorts(t))
	peer.unblock()
	peer.rpcOK(t, "engine:start", map[string]string{"engine": "llamacpp"})
	f.call(t, "engine:start", map[string]string{"engine": "ollama"})
	f.targetBinding(t)
	f.publish(t, true)
	var inventory struct {
		Models   []string                  `json:"models"`
		ByEngine map[string][]string       `json:"modelsByEngine"`
		Loaded   map[string][]string       `json:"loadedByEngine"`
		Policy   *nodepolicy.Advertisement `json:"policy"`
	}
	task2Decode(t, peer.rpcOK(t, "engine:models", nil), &inventory)
	if !slices.Contains(inventory.ByEngine["llamacpp"], "switch-model") || inventory.Policy == nil {
		t.Fatal("ready peer lacks real target inventory")
	}
	cfg, ok := clustertrust.Open(s.clusterDir).ClientTLSConfig(peerID)
	if !ok {
		t.Fatal("origin lacks the ready peer pin")
	}
	pinned := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
	defer pinned.CloseIdleConnections()
	task1Eventually(t, "ready peer broker publishes healthy ingress", 15*time.Second, func() bool {
		response, err := pinned.Get(fmt.Sprintf("https://127.0.0.1:%d/v1/models", peer.targetFacadePort))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.Copy(io.Discard, response.Body); err != nil {
			t.Fatal(err)
		}
		if err := response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		return response.StatusCode == http.StatusOK
	})
	f.publishDirectory(t, noderec.DirectoryNode{HostUUID: peerID, ClusterUUID: peerID, Name: peerID, IP: "127.0.0.1", Trusted: true,
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceOllama: {Port: peer.facadePort}, noderec.ServiceLlamaCPP: {Port: peer.targetFacadePort}},
		Models:   inventory.Models, ModelsByEngine: inventory.ByEngine, LoadedByEngine: inventory.Loaded, Policy: inventory.Policy,
	})
	f.call(t, "llamacpp-proxy:node/set-priority", map[string]any{"generation": 1, "nodes": []string{f.nodeID, peerID}})
	// Use the real service history owner, including its strict decoder and
	// transient busy contract. This fixture originates exactly one inference.
	history := func(method string) nodepolicy.RoutingHistoryInitial {
		t.Helper()
		var initial nodepolicy.RoutingHistoryInitial
		neEventually(t, "ready-peer history returns an admitted baseline", 30*time.Second, func() bool {
			id := strconv.FormatInt(newRPCID(), 10)
			m, err := f.client.call(id, method, json.RawMessage(`{}`), serviceCallTimeout)
			if err != nil {
				t.Fatalf("%s: %v", method, err)
			}
			if m.ID == nil || string(*m.ID) != id {
				t.Fatal("ready-peer history reply changed its request ID")
			}
			if m.Error != nil {
				if m.Error.Code == jsonrpc.HistoryBusyCode {
					return false
				}
				t.Fatalf("%s: %+v", method, m.Error)
			}
			var diagnostic nodepolicy.RoutingHistoryDiagnostic
			initial, diagnostic, err = nodepolicy.DecodeRoutingHistoryInitial(m.Result)
			if err != nil || diagnostic.Count != 0 {
				t.Fatalf("ready-peer history baseline: diagnostic=%+v error=%v", diagnostic, err)
			}
			return true
		})
		return initial
	}
	if initial := history(nodepolicy.MethodRoutingSubscribe); len(initial.Requests) != 0 {
		t.Fatal("fresh ready-peer fixture already has inference history")
	}
	client := &http.Client{Timeout: 15 * time.Second}
	defer client.CloseIdleConnections()
	assertEngineBindingSuccess(t, engineBindingAwait(t, f.request(t, client, f.engine.targetFacadePort, false, 5000)), "llamacpp", peerID)
	var completed nodepolicy.RoutingHistoryRecord
	neEventually(t, "ready-peer inference reaches terminal production history", 30*time.Second, func() bool {
		initial := history(nodepolicy.MethodRoutingGetInitial)
		if len(initial.Requests) > 1 {
			t.Fatalf("unique inference produced %d history records", len(initial.Requests))
		}
		if len(initial.Requests) == 0 || initial.Requests[0].EndedAt == nil {
			return false
		}
		completed = initial.Requests[0]
		return true
	})
	// Dispatch attempts are created before bound preparation. A cold attempt
	// that refuses cooldown and then falls back to the peer must fail this
	// oracle even when no engine was stopped or started.
	if completed.Request.OriginNodeID != f.nodeID || completed.Requested.Model != "switch-model" || completed.State != "completed" || completed.Status == nil || *completed.Status != http.StatusOK || completed.Served == nil || completed.DroppedAttempts != 0 || len(completed.Attempts) != 1 {
		t.Fatalf("ready-peer terminal history did not retain exactly one successful attempt: %+v", completed)
	}
	attempt := completed.Attempts[0]
	destination := attempt.Destination
	if attempt.Index != 0 || attempt.EndedAt == nil || attempt.Outcome != "committed" || attempt.Status == nil || *attempt.Status != http.StatusOK || destination.Kind != "local" || destination.NodeID == nil || *destination.NodeID != peerID || destination.Engine == nil || *destination.Engine != "llamacpp" || destination.Model == nil || *destination.Model != "switch-model" || !reflect.DeepEqual(completed.Served, &destination) {
		t.Fatalf("ready-peer history includes a cold or different destination: %+v", completed)
	}
	f.assertEffects(t, 1, 0, 0, 0)
	peer.mu.Lock()
	defer peer.mu.Unlock()
	if peer.inferences["llamacpp"] != 1 || peer.inferences["ollama"] != 0 || len(peer.starts["llamacpp"]) != 1 {
		t.Fatalf("ready peer effects starts=%v inference=%v", peer.starts, peer.inferences)
	}
}

func testEngineBindingPermission(t *testing.T) {
	f := newEngineBindingFixture(t, nil)
	f.targetBinding(t)
	f.publish(t, true)
	// These pre-existing pins receive legacy migration grants. Establish the
	// inference-only peer explicitly through the real local CAS writer.
	f.grant(t, "management-service-peer", false)
	// Reach the stopped offered facade with a valid pinned inference identity.
	// Caller/body headers cannot turn this inference-only peer into a manager.
	assertEngineBindingRefusal(t, engineBindingAwait(t, f.request(t, f.service.peer, f.engine.targetFacadePort, true, 5000)), "management-denied")
	f.assertEffects(t, 0, 0, 0, 0)
	f.grant(t, "management-service-peer", true)
	f.engine.unblock()
	assertEngineBindingSuccess(t, engineBindingAwait(t, f.request(t, f.service.peer, f.engine.targetFacadePort, true, 5000)), "llamacpp", f.nodeID)
	f.grant(t, "management-service-peer", false)
	f.publish(t, true)
	assertEngineBindingSuccess(t, engineBindingAwait(t, f.request(t, f.service.peer, f.engine.targetFacadePort, true, 5000)), "llamacpp", "management-service-target")
	f.assertEffects(t, 0, 1, 0, 2)
}

func testEngineBindingVeto(t *testing.T) {
	test := func(name, method string, params any, reason string) {
		t.Run(name, func(t *testing.T) {
			f := newEngineBindingFixture(t, nil)
			f.targetBinding(t)
			f.publish(t, true)
			f.grant(t, "management-service-peer", true)
			f.call(t, method, params)
			// Pinned ingress exercises the target guard even when veto withdraws
			// its offer from ordinary routing.
			assertEngineBindingRefusal(t, engineBindingAwait(t, f.request(t, f.service.peer, f.engine.targetFacadePort, true, 5000)), reason)
			f.assertEffects(t, 0, 0, 0, 0)
		})
	}
	test("local lock", "engine:set-lock", map[string]any{"engine": "llamacpp", "locked": true}, "engine-locked")
	test("node pause", "node:set-availability", map[string]string{"state": "paused"}, "paused")
}

func testEngineBindingProtectedResident(t *testing.T) {
	f := newEngineBindingFixture(t, nil)
	f.call(t, "engine:start", map[string]string{"engine": "ollama"})
	profile := fixtureResidentProfile()
	// Desired resident protection is sufficient for the stop guard. This
	// declaration supplies no live qualification and claims no loaded model.
	profile.Engine, profile.Model, profile.Name = "ollama", "switch-model", "binding-protected-source"
	want := fetchServicePolicy(t, f.client).Policy
	want.Profiles = []nodepolicy.Profile{profile}
	want.ResidentModels = nodepolicy.ResidentModels{{Engine: "ollama", Model: "switch-model"}}
	want.EngineBindings = nodepolicy.EngineBindings{{Engine: "llamacpp", Model: "switch-model"}}
	setServicePolicy(t, f.client, want)
	f.publish(t, true)
	client := &http.Client{Timeout: 20 * time.Second}
	defer client.CloseIdleConnections()
	assertEngineBindingRefusal(t, engineBindingAwait(t, f.request(t, client, f.engine.targetFacadePort, false, 5000)), "no-fit")
	f.assertEffects(t, 1, 0, 0, 0)
	stopped, err := serviceProcessStopped(context.Background(), f.engine.pid(t, "ollama"))
	if err != nil || stopped {
		t.Fatalf("protected source stopped: stopped=%t err=%v", stopped, err)
	}
}

func testEngineBindingMissingModel(t *testing.T) {
	f := newEngineBindingFixture(t, func(e *switchProcessFixture) {
		path := filepath.Join(jointAppDir(e.base), "engines", "llamacpp.json")
		var manifest map[string]json.RawMessage
		task2Decode(t, readPolicyBytes(t, path), &manifest)
		// The native fake reports empty canonical inventory on both manager
		// and proxy list routes. No stored model receipt is manufactured.
		var rt map[string]json.RawMessage
		task2Decode(t, manifest["runtime"], &rt)
		var env map[string]string
		task2Decode(t, rt["env"], &env)
		env["NVPAIR_TEST_SWITCH_EMPTY_MODELS"] = "1"
		rt["env"] = mustMarshal(t, env)
		manifest["runtime"] = mustMarshal(t, rt)
		var platforms map[string]map[string]json.RawMessage
		task2Decode(t, manifest["platforms"], &platforms)
		for _, platform := range platforms {
			platform["runtime"] = mustMarshal(t, rt)
		}
		manifest["platforms"] = mustMarshal(t, platforms)
		jointWriteJSON(t, path, manifest)
	})
	f.engine.unblock()
	f.targetBinding(t)
	f.publish(t, true)
	client := &http.Client{Timeout: 30 * time.Second}
	defer client.CloseIdleConnections()
	assertEngineBindingRefusal(t, engineBindingAwait(t, f.request(t, client, f.engine.targetFacadePort, false, 15000)), "model-unavailable")
	f.assertEffects(t, 0, 1, 0, 0)
}

func testEngineBindingStartFailure(t *testing.T) {
	f := newEngineBindingFixture(t, func(e *switchProcessFixture) { e.failTarget = true })
	f.targetBinding(t)
	f.publish(t, true)
	client := &http.Client{Timeout: 30 * time.Second}
	defer client.CloseIdleConnections()
	assertEngineBindingRefusal(t, engineBindingAwait(t, f.request(t, client, f.engine.targetFacadePort, false, 15000)), "model-unavailable")
	f.assertEffects(t, 0, 1, 0, 0)
	assertSwitchProcessStopped(t, f.engine.pid(t, "llamacpp"))
}

func testEngineBindingCompeting(t *testing.T) {
	f := newEngineBindingFixture(t, func(e *switchProcessFixture) { e.holdTarget = true })
	f.bind(t, nodepolicy.EngineBindings{{Engine: "llamacpp", Model: "switch-model"}, {Engine: "ollama", Model: "other-model"}})
	f.publish(t, true)
	client := &http.Client{Timeout: 40 * time.Second}
	defer client.CloseIdleConnections()
	one := f.request(t, client, f.engine.targetFacadePort, false, 30000)
	select {
	case <-f.engine.targetStarting:
	case <-time.After(15 * time.Second):
		t.Fatal("cold operation did not reach accepted startup")
	}
	// The competing model belongs to a different engine. Its public request
	// remains ordinary inference; only the native name differs.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", f.engine.facadePort), strings.NewReader(`{"model":"other-model","messages":[],"stream":false}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	var refused engineBindingResponse
	refused.status, refused.header = response.StatusCode, response.Header.Clone()
	if err := json.NewDecoder(response.Body).Decode(&refused.body); err != nil {
		t.Fatal(err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	assertEngineBindingRefusal(t, refused, "engine-switch-busy")
	f.assertEffects(t, 0, 1, 0, 0)
	f.engine.unblockTarget()
	assertEngineBindingSuccess(t, engineBindingAwait(t, one), "llamacpp", "management-service-target")
	f.assertEffects(t, 0, 1, 0, 1)
	// This request intentionally begins after readiness. It proves reuse of
	// the ready engine, not that another waiter joined the held preparation.
	two := f.request(t, client, f.engine.targetFacadePort, false, 30000)
	assertEngineBindingSuccess(t, engineBindingAwait(t, two), "llamacpp", "management-service-target")
	f.assertEffects(t, 0, 1, 0, 2)
}

func testEngineBindingRing(t *testing.T) {
	f := newEngineBindingFixture(t, func(e *switchProcessFixture) { e.failTarget = true; e.checkSourceExit = true })
	f.engine.unblockSource()
	f.call(t, "engine:start", map[string]string{"engine": "ollama"})
	manual := settings.SwitchRequest{RequestID: "binding-manual-retained", SourceEngine: "ollama", TargetEngine: "llamacpp", ExpectedRevision: "e3v1:" + strings.Repeat("0", 64)}
	m, err := f.client.call(strconv.FormatInt(newRPCID(), 10), settings.SwitchMethod, mustMarshal(t, manual), serviceCallTimeout)
	if err != nil || m.Error == nil || m.Error.Code != settings.SwitchConfigurationConflictCode {
		t.Fatalf("manual receipt setup: transport=%v", err)
	}
	manualReceipt := f.status(t, manual.RequestID).Receipt
	if manualReceipt == nil || manualReceipt.Automatic || !manualReceipt.Terminal || manualReceipt.Phase != settings.SwitchRejected {
		t.Fatal("manual validation receipt was not retained")
	}
	f.targetBinding(t)
	f.publish(t, true)
	client := &http.Client{Timeout: 100 * time.Second}
	defer client.CloseIdleConnections()
	var first settings.SwitchReceipt
	for i := 0; i < settings.MaxAutomaticSwitchReceipts+2; i++ {
		if i > 0 {
			f.call(t, "engine:start", map[string]string{"engine": "ollama"})
			f.publish(t, true)
		}
		response := engineBindingAwait(t, f.request(t, client, f.engine.targetFacadePort, false, nodepolicy.MaxBoundEngineBudgetMs))
		assertEngineBindingRefusal(t, response, "model-unavailable")
		id := response.header.Get(nodepolicy.SwitchRequestIDHeader)
		if id == "" {
			t.Fatalf("cycle %d did not reach a real automatic transaction", i)
		}
		var settled settings.SwitchJournal
		task1Eventually(t, "automatic failure cleanup settled before next cycle", 15*time.Second, func() bool {
			settled = f.journal(t)
			return settled.ActiveRequestID == "" && settled.AutomaticSettled[id] != 0
		})
		receipt := settled.Receipts[id]
		if !receipt.Automatic || !receipt.Terminal || receipt.Phase != settings.SwitchFailedTargetStart || receipt.RecoveryRequired || receipt.Error == nil || !receipt.Source.TerminationVerified {
			t.Fatalf("cycle %d terminal receipt=%+v", i, receipt)
		}
		if i == 0 {
			first = receipt
		}
		if !reflect.DeepEqual(settled.Receipts[manual.RequestID], *manualReceipt) {
			t.Fatal("automatic cycle changed the manual ID")
		}
	}
	journal := f.journal(t)
	if len(journal.AutomaticSettled) != settings.MaxAutomaticSwitchReceipts || len(journal.Receipts) != settings.MaxAutomaticSwitchReceipts+1 {
		t.Fatalf("automatic retention settled=%d receipts=%d", len(journal.AutomaticSettled), len(journal.Receipts))
	}
	if _, exists := journal.Receipts[first.RequestID]; exists {
		t.Fatal("oldest settled automatic receipt was not pruned")
	}
	m, err = f.client.call(strconv.FormatInt(newRPCID(), 10), settings.SwitchMethod, mustMarshal(t, manual), serviceCallTimeout)
	if err != nil {
		t.Fatalf("retained manual retry transport: %v", err)
	}
	if m.Error != nil {
		t.Fatalf("retained manual retry RPC error: %+v", m.Error)
	}
	var retry settings.SwitchResult
	task2Decode(t, m.Result, &retry)
	if !reflect.DeepEqual(retry.Receipt, *manualReceipt) || !reflect.DeepEqual(f.status(t, manual.RequestID).Receipt, manualReceipt) {
		t.Fatal("automatic cycling changed manual-ID retry/readback behavior")
	}
	// A pruned automatic ID cannot be replayed through public manual E4.
	m, err = f.client.call(strconv.FormatInt(newRPCID(), 10), settings.SwitchMethod, mustMarshal(t, first.Request), serviceCallTimeout)
	if err != nil {
		t.Fatalf("pruned automatic replay transport: %v", err)
	}
	if m.Error == nil || m.Error.Code != settings.SwitchInvalidParamsCode {
		t.Fatalf("pruned automatic replay error=%v", m.Error)
	}
	f.assertEffects(t, settings.MaxAutomaticSwitchReceipts+2, settings.MaxAutomaticSwitchReceipts+2, 0, 0)
}

func testEngineBindingRestart(t *testing.T) {
	f := newEngineBindingFixture(t, func(e *switchProcessFixture) { e.holdTarget = true; e.checkSourceExit = true })
	f.call(t, "engine:start", map[string]string{"engine": "ollama"})
	f.targetBinding(t)
	f.publish(t, true)
	client := &http.Client{Timeout: 100 * time.Second}
	defer client.CloseIdleConnections()
	demand, output := f.requestObserved(t, client, f.engine.targetFacadePort, false, nodepolicy.MaxBoundEngineBudgetMs)
	select {
	case <-f.engine.targetStarting:
	case <-time.After(80 * time.Second):
		t.Fatal("automatic replacement never reached accepted target startup")
	}
	active := f.status(t, "")
	if active.Receipt == nil || !active.Receipt.Automatic || active.Receipt.Terminal || active.Receipt.Phase != settings.SwitchTargetStarting || !active.Receipt.Source.TerminationVerified {
		t.Fatalf("restart acceptance barrier=%+v", active)
	}
	select {
	case <-output:
		t.Fatal("accepted replacement emitted inference before startup")
	default:
	}
	var broker serviceProcessIdentity
	for _, identity := range f.service.scan() {
		if identity.Path == f.service.brokerPath {
			broker = identity
		}
	}
	if broker.PID == 0 {
		t.Fatal("service-owned broker identity missing")
	}
	if err := f.service.inventory.kill(broker); err != nil {
		t.Fatal(err)
	}
	assertSwitchProcessStopped(t, broker.PID)
	assertSwitchProcessStopped(t, f.engine.pid(t, "ollama"))
	assertSwitchProcessStopped(t, f.engine.pid(t, "llamacpp"))
	f.engine.unblock()
	select {
	case result := <-demand:
		if result.err == nil && result.status == http.StatusOK {
			t.Fatal("interrupted inference returned success")
		}
	case <-time.After(15 * time.Second):
		t.Fatal("retired proxy did not release interrupted HTTP waiter")
	}
	f.client.notification(t, "service/broker-restarted")
	f.client = f.service.readyClient()
	var recovered settings.SwitchStatus
	task1Eventually(t, "replacement broker reconciles the original automatic ID", 30*time.Second, func() bool {
		recovered = f.status(t, active.Receipt.RequestID)
		return recovered.Receipt != nil && recovered.Receipt.Terminal && !recovered.Receipt.RecoveryRequired && recovered.ActiveRequestID == ""
	})
	// The receipt retains its last durable observations. Once the checkpoint
	// is released, query current state through the ordinary manager owner.
	if recovered.Receipt.Phase != settings.SwitchFailedInterrupted || !recovered.Receipt.Automatic || recovered.Receipt.RequestID != active.Receipt.RequestID || recovered.Receipt.Error == nil || recovered.Receipt.Error.Kind != settings.SwitchErrorInterrupted || !reflect.DeepEqual(recovered.Receipt.Source, active.Receipt.Source) || !reflect.DeepEqual(recovered.Receipt.Target, active.Receipt.Target) {
		t.Fatalf("reconciled automatic receipt=%+v", *recovered.Receipt)
	}
	var intent struct {
		Engines map[string]bool `json:"engines"`
	}
	task2Decode(t, readLiveDurableFile(t, filepath.Join(jointAppDir(f.service.configDir), "engine-bin", "engine-state.json")), &intent)
	for _, engine := range []string{"ollama", "llamacpp"} {
		if on, known := intent.Engines[engine]; !known || on {
			t.Fatalf("post-recovery %s intent known=%t on=%t", engine, known, on)
		}
		var status struct {
			Engine  string `json:"engine"`
			Running *bool  `json:"running"`
		}
		task2Decode(t, f.call(t, "engine:status", map[string]string{"engine": engine}), &status)
		if status.Engine != engine || status.Running == nil || *status.Running {
			t.Fatalf("post-recovery manager state=%+v", status)
		}
	}
	f.assertEffects(t, 1, 1, 0, 0)
	// A new manual start must survive old recovery. The request is new work,
	// not replay of the interrupted inference or its automatic submission.
	f.call(t, "engine:start", map[string]string{"engine": "llamacpp"})
	f.publish(t, true)
	assertEngineBindingSuccess(t, engineBindingAwait(t, f.request(t, client, f.engine.targetFacadePort, false, 5000)), "llamacpp", f.nodeID)
	f.assertEffects(t, 1, 2, 0, 1)
	if got := f.status(t, active.Receipt.RequestID); !reflect.DeepEqual(got.Receipt, recovered.Receipt) {
		t.Fatal("new manual work changed the interrupted receipt")
	}
}

func testEngineBindingPinned(t *testing.T, reverse bool) {
	mesh := newTask2Fixture(t, task2WithoutScanner)
	previousSessions := mesh.a.observers[noderec.ServiceCluster].established(mesh.b.uuid)
	mesh.engines[mesh.b.uuid].stop()
	mesh.b.cm.stop()
	s := newServiceFixture(t)
	// Unlike newRemoteServiceFixture, this constructor has not replaced the
	// production scanner with the registered empty-directory test personality.
	image, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	jointCopyExecutable(t, image, s.scannerPath)
	s.clusterDir = filepath.Join(mesh.b.base, "cluster")
	s.ports.clusterManager = mesh.b.ports[noderec.ServiceCluster]
	s.ports.engineManagerHTTP = mesh.b.ports[noderec.ServiceEngineManager]
	s.ports.engineControl = mesh.b.ports[noderec.ServiceEngineControl]
	for _, path := range []string{engineMgrBin, clusterMgrBin} {
		s.inventory.paths = append(s.inventory.paths, s.copyBinary(path))
	}
	cfg, ok := mesh.a.mesh.ClientTLSConfig(mesh.b.uuid)
	if !ok {
		t.Fatal("mesh caller lacks the service target pin")
	}
	pinned := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
	t.Cleanup(pinned.CloseIdleConnections)
	f := newEngineBindingServiceFixture(t, &remoteServiceFixture{serviceFixture: s, peer: pinned}, nil)
	t.Cleanup(func() {
		if t.Failed() {
			logSwitchPeerAuthority(t, "binding failure before fixture teardown", mesh.a, mesh.b)
			logSwitchPeerAuthority(t, "binding failure before fixture teardown", mesh.b, mesh.a)
		}
	})
	f.grant(t, mesh.a.uuid, true)
	f.targetBinding(t)
	node := f.publish(t, true)
	if len(node.ModelsByEngine["llamacpp"]) != 0 || node.Policy.EngineBindingOffers == nil {
		t.Fatal("stopped facade fixture advertised actual target inventory")
	}
	// A deliberately selected ready onward route makes forwarding observable.
	// Authenticated ingress must still prepare and execute only its local target.
	var onward atomic.Int32
	trap := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			onward.Add(1)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"model":"switch-model","choices":[]}`)
	}))
	t.Cleanup(trap.Close)
	f.call(t, "llamacpp-proxy:node/add-manual", map[string]any{"id": "binding-onward", "host": "127.0.0.1", "addresses": []string{"127.0.0.1"}, "port": portOfURL(t, trap.URL), "models": []string{"switch-model"}})
	f.call(t, "llamacpp-proxy:node/select", map[string]string{"id": "binding-onward"})
	port := f.engine.targetFacadePort
	if reverse {
		port = mesh.blockedPort
		task1Eventually(t, "service-owned target establishes a new pinned reverse session", 20*time.Second, func() bool {
			return mesh.a.observers[noderec.ServiceCluster].established(mesh.b.uuid) > previousSessions
		})
	} else {
		// Reject the outer route so direct success cannot borrow reverse fallback.
		mesh.a.observers[noderec.ServiceCluster].interrupt(mesh.b.uuid, true)
	}
	var refusedAuthority sync.Once
	adapter := clustertrust.NewPeerServiceClient(mesh.a.mesh, clustertrust.PeerServiceClientOptions{
		BaseConfigDir:     mesh.a.base,
		PeerClientOptions: clustertrust.PeerClientOptions{Timeout: 100 * time.Second},
		ResolveRoute: func(peer string, service noderec.ServiceKey) (clustertrust.PeerServiceRoute, bool) {
			mesh.a.mesh.Refresh()
			incarnation, admitted := mesh.a.mesh.PeerServiceIncarnation(peer)
			if peer == mesh.b.uuid && !admitted {
				refusedAuthority.Do(func() {
					logSwitchPeerAuthority(t, "route resolver PeerServiceIncarnation returned false", mesh.a, mesh.b)
				})
			}
			if peer != mesh.b.uuid || service != noderec.ServiceLlamaCPP || !admitted {
				return clustertrust.PeerServiceRoute{}, false
			}
			return clustertrust.PeerServiceRoute{Incarnation: incarnation, Address: "127.0.0.1:" + strconv.Itoa(port), Protocol: clustertrust.ReverseTLS, Generation: 1}, true
		},
	})
	t.Cleanup(adapter.CloseIdle)
	client := adapter.Client(mesh.b.uuid, noderec.ServiceLlamaCPP)
	task1Eventually(t, "stopped offered facade reaches authenticated local ingress through the service allowlist", 15*time.Second, func() bool {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("https://127.0.0.1:%d/v1/models", port), nil)
		if err != nil {
			t.Fatal(err)
		}
		response, err := client.Do(request)
		if err != nil {
			return false
		}
		var body struct {
			Code  string `json:"code"`
			Error string `json:"error"`
		}
		decodeErr := json.NewDecoder(response.Body).Decode(&body)
		if err := response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		if decodeErr != nil {
			t.Fatalf("decode stopped ingress: %v", decodeErr)
		}
		return response.StatusCode == http.StatusServiceUnavailable && body.Code == "no-local-backend" && body.Error == "no local inference backend is available on this node"
	})
	f.assertEffects(t, 0, 0, 0, 0)
	f.engine.unblock()
	// This is a successful cold-start case, not a shortened-budget refusal.
	// Leave the production preparation allowance intact across the pinned hop.
	response := engineBindingAwait(t, f.request(t, client, port, true, nodepolicy.MaxBoundEngineBudgetMs))
	assertEngineBindingSuccess(t, response, "llamacpp", mesh.b.uuid)
	if onward.Load() != 0 {
		t.Fatalf("terminal pinned ingress forwarded onward %d times", onward.Load())
	}
	f.assertEffects(t, 0, 1, 0, 1)
}
