// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

// Complete the older pin-only fixtures without replacing their certificates or
// local identity. Subsequent calls change only the permission document.
func grantRemoteManagement(t *testing.T, dir, cluster, self, peer string, allowed bool) {
	t.Helper()
	mesh := clustertrust.Open(dir)
	if mesh.NodeUUID() != self {
		t.Fatalf("fixture identity = %q, want %q", mesh.NodeUUID(), self)
	}
	docPath := filepath.Join(dir, clustertrust.RemoteManagementFile)
	var doc clustertrust.RemoteManagementDocument
	data, err := os.ReadFile(docPath)
	if errors.Is(err, os.ErrNotExist) {
		var admission struct {
			Epoch uint64 `json:"epoch"`
		}
		rmReadJSON(t, filepath.Join(dir, "admission.json"), &admission)
		pinPath := filepath.Join(dir, "trusted", peer+".json")
		var pin map[string]json.RawMessage
		rmReadJSON(t, pinPath, &pin)
		pin["clusterId"] = rmJSON(t, cluster)
		pin["admissionEpoch"] = rmJSON(t, admission.Epoch)
		rmWriteJSON(t, pinPath, pin)
		member := map[string]any{"nodeUuid": peer, "state": "member", "clusterId": cluster, "admissionEpoch": admission.Epoch}
		rmWriteJSON(t, filepath.Join(dir, "members.json"), []any{member})
		cert, readErr := os.ReadFile(filepath.Join(dir, "node.crt"))
		if readErr != nil {
			t.Fatal(readErr)
		}
		snapshot := clustertrust.RemoteManagementMigrationSnapshot{
			Local: clustertrust.RemoteManagementMigrationAuthority{PeerUUID: self, CertFingerprint: fmt.Sprintf("sha256:%x", sha256.Sum256(cert)), ClusterID: cluster, AdmissionEpoch: admission.Epoch},
			Peers: []clustertrust.RemoteManagementMigrationAuthority{},
		}
		id, idErr := clustertrust.RemoteManagementMigrationID(snapshot)
		if idErr != nil {
			t.Fatal(idErr)
		}
		rmWriteJSON(t, filepath.Join(dir, clustertrust.RemoteManagementMigrationFile), clustertrust.RemoteManagementMigration{Version: 1, ID: id, State: "complete", Snapshot: snapshot})
		doc = clustertrust.RemoteManagementDocument{Version: 1, Revision: "0", MigrationID: id, Grants: []clustertrust.RemoteManagementGrant{}}
	} else if err != nil {
		t.Fatal(err)
	} else if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	incarnation, ok := mesh.PeerServiceIncarnation(peer)
	if !ok {
		t.Fatal("fixture peer has no service incarnation")
	}
	revision, err := strconv.ParseUint(doc.Revision, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	doc.Revision = strconv.FormatUint(revision+1, 10)
	grant := clustertrust.RemoteManagementGrant{PeerUUID: peer, Incarnation: incarnation, Allowed: allowed, DecisionRevision: doc.Revision}
	found := false
	for i := range doc.Grants {
		if doc.Grants[i].PeerUUID == peer {
			doc.Grants[i], found = grant, true
		}
	}
	if !found {
		doc.Grants = append(doc.Grants, grant)
	}
	rmWriteJSON(t, docPath, doc)
	if got := mesh.RemoteManagement(peer); got.Allowed != allowed || (got.State != "allowed" && got.State != "denied") {
		t.Fatalf("permission fixture = %+v, want allowed %t", got, allowed)
	}
}

func rmJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func rmWriteJSON(t *testing.T, path string, value any) {
	t.Helper()
	if err := os.WriteFile(path, rmJSON(t, value), 0600); err != nil {
		t.Fatal(err)
	}
}

func rmReadJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, value); err != nil {
		t.Fatal(err)
	}
}

func bindRemoteManagementTestSource(b *Broker) *engineManagerSource {
	source := b.activateEngineManagerSource()
	b.bindEngineMgr(b.getEngineMgr(), source)
	return source
}

func rmDeny(t *testing.T, err error) {
	t.Helper()
	var denied *remoteManagementDeniedError
	if !errors.As(err, &denied) || policyErrorCode(err) != -32012 || err.Error() != remoteManagementDeniedMessage {
		t.Fatalf("permission refusal = %v (code %d), want typed -32012", err, policyErrorCode(err))
	}
}

func rmPolicyRequest(t *testing.T, rb *routedBroker) policyRelayRequest {
	t.Helper()
	p := rb.b.peerPolicy().Policy
	p.Idle.UnloadAfterMinutes++
	return policyRelayRequest{ID: "policy-write", Method: "set", Caller: "pinned-peer", Params: rmJSON(t, policyParams{Policy: rmJSON(t, p), ExpectedPersistedRevision: rmJSON(t, p.Revision)})}
}

func TestRemoteManagementPolicyAcceptance(t *testing.T) {
	rb := newRoutedBroker(t)
	waitPushesSettled(t, rb.b)
	p := rmPolicyRequest(t, rb)
	before := rb.baseline(t)
	engineCalls := len(rb.engine.snapshot())
	grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", false)
	_, err := rb.b.servePolicyRelay(context.Background(), p)
	rmDeny(t, err)
	before.requireUnchanged(t)
	if len(rb.engine.snapshot()) != engineCalls {
		t.Fatal("denied policy called engine manager")
	}
	grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", true)
	if _, err := rb.b.servePolicyRelay(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	if rb.b.localPolicy().Policy.Revision == before.revision {
		t.Fatal("granted policy did not reach writer")
	}
}

func TestRemoteManagementAvailabilityAcceptance(t *testing.T) {
	rb := newRoutedBroker(t)
	waitPushesSettled(t, rb.b)
	before := rb.baseline(t)
	engineCalls := len(rb.engine.snapshot())
	p := policyRelayRequest{ID: "pause", Method: "set-availability", Caller: "pinned-peer", Params: rmJSON(t, policyParams{State: string(nodepolicy.Paused)})}
	grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", false)
	_, err := rb.b.servePolicyRelay(context.Background(), p)
	rmDeny(t, err)
	before.requireUnchanged(t)
	if len(rb.engine.snapshot()) != engineCalls {
		t.Fatal("denied availability called engine manager")
	}
	grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", true)
	if _, err := rb.b.servePolicyRelay(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	if rb.b.localPolicy().Policy.Availability != nodepolicy.Paused {
		t.Fatal("granted pause did not reach writer")
	}
}

func rmSettings(t *testing.T) (*settingsHarness, settings.Request) {
	t.Helper()
	h := newSettingsHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
	return h, h.changedRequest(t, "--pg2")
}

func rmSettingsUnchanged(t *testing.T, h *settingsHarness, before settings.Snapshot, requestID string, journal []byte) {
	t.Helper()
	if got := h.snapshot(); !reflect.DeepEqual(got, before) || h.applies.Load() != 0 || h.proxyRebinds.Load() != 0 {
		t.Fatalf("denied settings changed runtime: %+v", got)
	}
	path, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, journal) {
		t.Fatal("denied settings changed journal")
	}
	if _, ok := h.b.engineSettings["ollama"].Receipts[requestID]; ok {
		t.Fatal("denied settings created receipt")
	}
}

func rmSettingsJournal(t *testing.T, h *settingsHarness) []byte {
	t.Helper()
	path, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestRemoteManagementSettingsAcceptance(t *testing.T) {
	h, p := rmSettings(t)
	before, journal := h.snapshot(), rmSettingsJournal(t, h)
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
	_, err := h.b.applyEngineSettings(context.Background(), p, "peer")
	rmDeny(t, err)
	rmSettingsUnchanged(t, h, before, p.RequestID, journal)
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "peer")
	if err != nil || receipt.Phase != "succeeded" || h.applies.Load() != 1 {
		t.Fatalf("granted settings = %+v, %v, calls %d", receipt, err, h.applies.Load())
	}
}

func TestRemoteManagementDeniedSettingsCreatesNoJournal(t *testing.T) {
	h := newSettingsHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
	_, err := h.b.applyEngineSettings(context.Background(), settings.Request{Engine: "ollama", RequestID: "unaccepted"}, "peer")
	rmDeny(t, err)
	path, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("denied request journal stat = %v", err)
	}
	if h.b.engineSettingsLoaded || h.applies.Load() != 0 {
		t.Fatal("denied request loaded journal or applied settings")
	}
}

func rmSwitch(t *testing.T, id string) (*switchHarness, settings.SwitchRequest) {
	t.Helper()
	h := newSwitchHarness(t)
	clustertrusttest.Join(t, h.b.clusterDir, "cluster", "target", "peer")
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
	return h, h.request(t, id)
}

func rmSwitchUnchanged(t *testing.T, h *switchHarness, id string, calls int) {
	t.Helper()
	h.b.engineConfigMu.Lock()
	_, found := h.b.engineSwitch.store.Lookup(id)
	h.b.engineConfigMu.Unlock()
	if found || len(h.engine.snapshot()) != calls || len(switchSteps(t, h.engine)) != 0 {
		t.Fatal("denied switch created receipt or called worker")
	}
}

func TestRemoteManagementSwitchAcceptance(t *testing.T) {
	h, p := rmSwitch(t, "permission-switch")
	calls := len(h.engine.snapshot())
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
	_, err := h.b.acceptEngineSwitch(context.Background(), p, "peer")
	rmDeny(t, err)
	rmSwitchUnchanged(t, h, p.RequestID, calls)
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", true)
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	if r := h.receipt(t, p.RequestID); r.Phase != settings.SwitchCompleted {
		t.Fatalf("granted switch = %+v", r)
	}
}

func TestRemoteManagementSwitchRetryRequiresGrant(t *testing.T) {
	h, p := rmSwitch(t, "retry-requires-grant")
	if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
		t.Fatal(err)
	}
	h.receipt(t, p.RequestID)
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
	_, err := h.b.acceptEngineSwitch(context.Background(), p, "peer")
	rmDeny(t, err)
}

func TestRemoteManagementDeniedPeerReads(t *testing.T) {
	t.Run("policy", func(t *testing.T) {
		rb := newRoutedBroker(t)
		grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", false)
		if _, err := rb.b.servePolicyRelay(context.Background(), policyRelayRequest{Method: "get", Caller: "pinned-peer"}); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("settings get and preview", func(t *testing.T) {
		h, p := rmSettings(t)
		grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
		if _, err := h.b.getEngineSettings(context.Background(), p, "peer"); err != nil {
			t.Fatal(err)
		}
		if _, err := h.b.previewEngineSettings(context.Background(), p, "peer"); err != nil {
			t.Fatal(err)
		}
		if h.applies.Load() != 0 {
			t.Fatal("read applied settings")
		}
	})
	t.Run("switch status and receipt", func(t *testing.T) {
		h, p := rmSwitch(t, "read-receipt")
		if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
			t.Fatal(err)
		}
		want := h.receipt(t, p.RequestID)
		grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
		result, err := h.b.dispatchEngineSwitch(context.Background(), settings.SwitchStatusMethod, rmJSON(t, settings.SwitchStatusRequest{RequestID: p.RequestID}), "peer")
		if err != nil {
			t.Fatal(err)
		}
		status, ok := result.(settings.SwitchStatus)
		if !ok || status.Receipt == nil || !reflect.DeepEqual(*status.Receipt, want) {
			t.Fatalf("denied peer status = %+v", result)
		}
	})
}

// Observe the submitted goroutine blocked in the real owning mutex. The channel
// identifies that goroutine, avoiding sleeps or production-only test hooks.
func rmBlockedCall(t *testing.T, function string, call func() error) <-chan error {
	t.Helper()
	started, done := make(chan string, 1), make(chan error, 1)
	go func() {
		stack := make([]byte, 512)
		n := runtime.Stack(stack, false)
		started <- strings.Split(string(stack[:n]), "\n")[0]
		done <- call()
	}()
	header := <-started
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		stack := make([]byte, 1<<20)
		n := runtime.Stack(stack, true)
		for _, frame := range strings.Split(string(stack[:n]), "\n\n") {
			ownsCall := strings.HasPrefix(frame, strings.Split(header, " [")[0]+" [sync.Mutex.Lock]")
			// Availability hands execution to its target-owned transition goroutine.
			if function == "runAvailabilityTransition" {
				ownsCall = strings.Contains(strings.Split(frame, "\n")[0], "[sync.Mutex.Lock]")
			}
			if ownsCall && strings.Contains(frame, function) {
				return done
			}
		}
		runtime.Gosched()
	}
	t.Fatalf("request did not block in %s", function)
	return done
}

func rmHoldLock(t *testing.T, mu *sync.Mutex) func() {
	t.Helper()
	mu.Lock()
	var once sync.Once
	unlock := func() { once.Do(mu.Unlock) }
	t.Cleanup(unlock)
	return unlock
}

func rmResult(t *testing.T, done <-chan error) error {
	t.Helper()
	select {
	case err := <-done:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("request did not finish after barrier")
		return nil
	}
}

func TestRemoteManagementPolicyRevokeBarrier(t *testing.T) {
	rb := newRoutedBroker(t)
	waitPushesSettled(t, rb.b)
	p, before := rmPolicyRequest(t, rb), rb.baseline(t)
	r := rb.b.nodePolicyRuntime()
	unlock := rmHoldLock(t, &r.mu)
	done := rmBlockedCall(t, "setPeerPolicyAs", func() error { _, err := rb.b.servePolicyRelay(context.Background(), p); return err })
	grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", false)
	unlock()
	rmDeny(t, rmResult(t, done))
	before.requireUnchanged(t)
}

func TestRemoteManagementAvailabilityRevokeBarrier(t *testing.T) {
	rb := newRoutedBroker(t)
	waitPushesSettled(t, rb.b)
	before := rb.baseline(t)
	r := rb.b.nodePolicyRuntime()
	previousCancelled := make(chan struct{})
	r.mu.Lock()
	r.transitionCancel = func() { close(previousCancelled) }
	r.mu.Unlock()
	unlock := rmHoldLock(t, &r.mu)
	done := rmBlockedCall(t, "setAvailabilityAs", func() error {
		_, err := rb.b.setAvailabilityAs(context.Background(), nodepolicy.Paused, nodepolicy.PausedByPeer, "pinned-peer")
		return err
	})
	grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", false)
	unlock()
	rmDeny(t, rmResult(t, done))
	before.requireUnchanged(t)
	select {
	case <-previousCancelled:
		t.Fatal("denied registration cancelled prior accepted work")
	default:
	}
}

func TestRemoteManagementAcceptedAvailabilityFinishesAfterRevoke(t *testing.T) {
	rb := newRoutedBroker(t)
	waitPushesSettled(t, rb.b)
	r := rb.b.nodePolicyRuntime()
	unlock := rmHoldLock(t, &r.transitionMu)
	done := rmBlockedCall(t, "runAvailabilityTransition", func() error {
		_, err := rb.b.setAvailabilityAs(context.Background(), nodepolicy.Paused, nodepolicy.PausedByPeer, "pinned-peer")
		return err
	})
	// Registration has already accepted and superseded under r.mu. The
	// execution mutex is downstream of that boundary, like a manager op lock.
	grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", false)
	unlock()
	if err := rmResult(t, done); err != nil {
		t.Fatalf("accepted availability abandoned after revoke: %v", err)
	}
	if got := rb.b.localPolicy().Policy.Availability; got != nodepolicy.Paused {
		t.Fatalf("accepted pause = %q", got)
	}
}

func TestRemoteManagementSettingsRevokeBarrier(t *testing.T) {
	for _, lock := range []string{"apply", "journal"} {
		t.Run(lock, func(t *testing.T) {
			h, p := rmSettings(t)
			before, journal := h.snapshot(), rmSettingsJournal(t, h)
			mu := &h.b.settingsApplyMu
			if lock == "journal" {
				mu = &h.b.engineConfigMu
			}
			unlock := rmHoldLock(t, mu)
			done := rmBlockedCall(t, "applyEngineSettings", func() error { _, err := h.b.applyEngineSettings(context.Background(), p, "peer"); return err })
			grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
			unlock()
			rmDeny(t, rmResult(t, done))
			rmSettingsUnchanged(t, h, before, p.RequestID, journal)
		})
	}
}

func TestRemoteManagementSwitchRevokeBarrier(t *testing.T) {
	h, p := rmSwitch(t, "switch-barrier")
	calls := len(h.engine.snapshot())
	unlock := rmHoldLock(t, &h.b.engineConfigMu)
	done := rmBlockedCall(t, "acceptEngineSwitch", func() error { _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); return err })
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
	unlock()
	rmDeny(t, rmResult(t, done))
	rmSwitchUnchanged(t, h, p.RequestID, calls)
}

type rmDecisionBarrier struct {
	slog.Handler
	entered, release chan struct{}
	once             sync.Once
}

func (h *rmDecisionBarrier) Handle(ctx context.Context, record slog.Record) error {
	family, outcome := "", ""
	record.Attrs(func(a slog.Attr) bool {
		if a.Key == "family" {
			family = a.Value.String()
		}
		if a.Key == "outcome" {
			outcome = a.Value.String()
		}
		return true
	})
	if record.Message == "remote management decision" && family == "switch" && outcome == "allowed" {
		h.once.Do(func() { close(h.entered); <-h.release })
	}
	return h.Handler.Handle(ctx, record)
}

func TestRemoteManagementSwitchSecondAcceptanceRecheck(t *testing.T) {
	h, p := rmSwitch(t, "second-acceptance")
	calls := len(h.engine.snapshot())
	barrier := &rmDecisionBarrier{Handler: slog.NewTextHandler(io.Discard, nil), entered: make(chan struct{}), release: make(chan struct{})}
	previous := slog.Default()
	previousWriter, previousFlags := log.Writer(), log.Flags()
	slog.SetDefault(slog.New(barrier))
	t.Cleanup(func() { slog.SetDefault(previous); log.SetOutput(previousWriter); log.SetFlags(previousFlags) })
	var once sync.Once
	release := func() { once.Do(func() { close(barrier.release) }) }
	t.Cleanup(release)
	done := make(chan error, 1)
	go func() { _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); done <- err }()
	requireGateOpens(t, barrier.entered, "first switch permission snapshot")
	// The first lock has read an allow, but receipt acceptance still lies
	// behind the second acquisition. Revocation must be re-read there.
	grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
	release()
	rmDeny(t, rmResult(t, done))
	rmSwitchUnchanged(t, h, p.RequestID, calls)
}

type rmReply struct {
	ID        string          `json:"id"`
	Result    json.RawMessage `json:"result"`
	Error     string          `json:"error"`
	ErrorCode int             `json:"errorCode"`
}

func rmWaitReply(t *testing.T, engine *fakeWorker, method, id string) rmReply {
	t.Helper()
	var reply rmReply
	engine.waitFor(method+" "+id, func(calls []policyCall) bool {
		for _, call := range calls {
			if call.method == method {
				if err := json.Unmarshal(call.params, &reply); err != nil {
					t.Fatal(err)
				}
				if reply.ID == id {
					return true
				}
			}
		}
		return false
	})
	return reply
}

func TestRemoteManagementBlankRelayCaller(t *testing.T) {
	for _, caller := range []string{"", " \t\n"} {
		t.Run(fmt.Sprintf("caller %q", caller), func(t *testing.T) {
			rb := newRoutedBroker(t)
			for _, family := range []struct {
				method, reply string
				payload       any
			}{
				{policyRelayRequestMethod, policyRelayReplyMethod, policyRelayRequest{ID: "blank-policy", Method: "get", Caller: caller}},
				{"engine:settings-request", "engine:settings-reply", settings.Relay{ID: "blank-settings", Method: "get", Caller: caller, Request: settings.Request{Engine: "ollama"}}},
				{"engine:switch-request", "engine:switch-reply", map[string]any{"id": "blank-switch", "method": settings.SwitchStatusMethod, "caller": caller, "params": map[string]any{}}},
			} {
				rb.b.forwardEngineNotificationFrom(rb.source, family.method, rmJSON(t, family.payload))
				var request struct {
					ID string `json:"id"`
				}
				if err := json.Unmarshal(rmJSON(t, family.payload), &request); err != nil {
					t.Fatal(err)
				}
				reply := rmWaitReply(t, rb.engine, family.reply, request.ID)
				if reply.ErrorCode != -32602 || reply.Error != errRemoteCallerRequired.Error() {
					t.Fatalf("%s blank caller reply = %+v", family.method, reply)
				}
			}
		})
	}
}

func TestRemoteManagementStaleWorkerDropped(t *testing.T) {
	rb := newRoutedBroker(t)
	old := rb.source
	current := bindRemoteManagementTestSource(rb.b)
	waitPushesSettled(t, rb.b)
	before := rb.baseline(t)
	for _, notification := range []struct {
		method string
		params any
	}{
		{policyRelayRequestMethod, rmPolicyRequest(t, rb)},
		{"engine:settings-request", settings.Relay{ID: "stale-settings", Method: "apply", Caller: "pinned-peer", Request: settings.Request{RequestID: "bad", Engine: "ollama"}}},
		{"engine:switch-request", map[string]any{"id": "stale-switch", "method": settings.SwitchMethod, "caller": "pinned-peer", "params": map[string]any{}}},
	} {
		rb.b.forwardEngineNotificationFrom(old, notification.method, rmJSON(t, notification.params))
	}
	// A valid request on the current source is a FIFO read-pump fence. Stale
	// callbacks must not even register asynchronous work or cancellation entries.
	r := rb.b.nodePolicyRuntime()
	r.mu.Lock()
	queued := len(r.relayCancels)
	r.mu.Unlock()
	rb.b.settingsRelayMu.Lock()
	queued += len(rb.b.settingsCancels)
	rb.b.settingsRelayMu.Unlock()
	c := rb.b.switchCoordinator()
	c.waitMu.Lock()
	queued += len(c.waiters)
	c.waitMu.Unlock()
	if queued != 0 {
		t.Fatalf("stale source registered %d requests", queued)
	}
	rb.b.forwardEngineNotificationFrom(current, policyRelayRequestMethod, rmJSON(t, policyRelayRequest{ID: "fence", Method: "get", Caller: "pinned-peer"}))
	rmWaitReply(t, rb.engine, policyRelayReplyMethod, "fence")
	for _, call := range rb.engine.snapshot() {
		if call.method == policyRelayReplyMethod || call.method == "engine:settings-reply" || call.method == "engine:switch-reply" {
			var reply rmReply
			if err := json.Unmarshal(call.params, &reply); err != nil {
				t.Fatal(err)
			}
			if reply.ID != "fence" {
				t.Fatalf("retired source got reply %+v", reply)
			}
		}
	}
	before.requireUnchanged(t)
}

func TestRemoteManagementRetiredWorkerStillBoundIsDropped(t *testing.T) {
	rb := newRoutedBroker(t)
	waitPushesSettled(t, rb.b)
	before := rb.baseline(t)
	rb.b.retireEngineManagerSource(rb.source)
	waitPushesSettled(t, rb.b)
	// Retirement precedes handle clearing on real worker exit. Pair identity
	// alone is insufficient in that window: the source must still be current.
	rb.b.forwardEngineNotificationFrom(rb.source, policyRelayRequestMethod, rmJSON(t, rmPolicyRequest(t, rb)))
	rb.b.forwardEngineNotificationFrom(rb.source, "engine:settings-request", rmJSON(t, settings.Relay{ID: "retired-settings", Method: "apply", Caller: "pinned-peer"}))
	rb.b.forwardEngineNotificationFrom(rb.source, "engine:switch-request", rmJSON(t, map[string]any{"id": "retired-switch", "method": settings.SwitchMethod, "caller": "pinned-peer", "params": map[string]any{}}))
	r := rb.b.nodePolicyRuntime()
	r.mu.Lock()
	queued := len(r.relayCancels)
	r.mu.Unlock()
	rb.b.settingsRelayMu.Lock()
	queued += len(rb.b.settingsCancels)
	rb.b.settingsRelayMu.Unlock()
	c := rb.b.switchCoordinator()
	c.waitMu.Lock()
	queued += len(c.waiters)
	c.waitMu.Unlock()
	if queued != 0 {
		t.Fatalf("retired source registered %d requests", queued)
	}
	for _, call := range rb.engine.snapshot() {
		if call.method == policyRelayReplyMethod || call.method == "engine:settings-reply" || call.method == "engine:switch-reply" {
			t.Fatalf("retired source got %s", call.method)
		}
	}
	before.requireUnchanged(t)
}

func TestRemoteManagementRetiredWorkerGetsNoAcceptedReply(t *testing.T) {
	h, p := rmSettings(t)
	source := bindRemoteManagementTestSource(h.b)
	h.entered, h.release = make(chan struct{}, 1), make(chan struct{})
	var once sync.Once
	release := func() { once.Do(func() { close(h.release) }) }
	t.Cleanup(release)
	h.b.forwardEngineNotificationFrom(source, "engine:settings-request", rmJSON(t, settings.Relay{ID: "accepted-before-retirement", Method: "apply", Caller: "peer", Request: p}))
	requireGateOpens(t, h.entered, "accepted settings configure")
	h.b.retireEngineManagerSource(source)
	bindRemoteManagementTestSource(h.b)
	release()
	deadline := time.Now().Add(5 * time.Second)
	for {
		h.b.settingsRelayMu.Lock()
		pending := len(h.b.settingsCancels)
		h.b.settingsRelayMu.Unlock()
		if pending == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("accepted relay did not settle")
		}
		runtime.Gosched()
	}
	select {
	case reply := <-h.settingsReplies:
		t.Fatalf("reply escaped through retired/replacement pair: %s", reply)
	default:
	}
	if h.applies.Load() != 1 || h.snapshot().Phase != "succeeded" {
		t.Fatal("retirement abandoned accepted work")
	}
}

func TestRemoteManagementPublicInternalMethods(t *testing.T) {
	// Independent inventory: never derive this from the production rejection map.
	methods := []string{
		"policy:request", "policy:reply", "policy:cancel",
		"engine:settings-request", "engine:settings-reply", "engine:settings-cancel", "engine:settings-projection",
		"engine:switch-request", "engine:switch-reply", "engine:switch-cancel", "engine:switch-changed", "engine:switch-recovery-ready",
		settings.SwitchPrepareMethod, settings.SwitchStepMethod, settings.SwitchReconcileMethod,
		"engine:set-reserved-port", "internal:set-reserved-port", "engine:configure-launch",
		nodepolicy.MethodModelMutationRequest, nodepolicy.MethodModelMutationReply, nodepolicy.MethodReconcileResident,
		"engine:dispatch-resident-request", "engine:dispatch-resident-reply", "engine:node-projection", "engine:restore-enabled",
		"engine:prepare-shutdown", "engine:get-launch", "engine:preview-launch", "engine:configured-ports",
	}
	for _, method := range methods {
		t.Run(method, func(t *testing.T) {
			b, out := newPolicyBroker(t)
			engine := attachFakeEngine(t, b)
			peer, cluster := newFakeWorkerPipe(t)
			b.clusterMgr = &clusterManagerProcess{peer: peer}
			b.handleMessage(clientRequest(1, method, map[string]any{"caller": "pinned-peer", "local": true}))
			response := out.waitFrame(t, method, isResponse(1))
			if response.Error == nil || response.Error.Code != -32601 {
				t.Fatalf("public %s response = %+v", method, response)
			}
			if len(engine.snapshot()) != 0 || len(cluster.snapshot()) != 0 {
				t.Fatal("internal client method reached worker")
			}
		})
	}
}

func TestRemoteManagementReplyCodes(t *testing.T) {
	t.Run("policy and availability", func(t *testing.T) {
		rb := newRoutedBroker(t)
		grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", false)
		for _, p := range []policyRelayRequest{rmPolicyRequest(t, rb), {ID: "availability-code", Method: "set-availability", Caller: "pinned-peer", Params: rmJSON(t, policyParams{State: string(nodepolicy.Paused)})}} {
			rb.b.forwardEngineNotificationFrom(rb.source, policyRelayRequestMethod, rmJSON(t, p))
			reply := rmWaitReply(t, rb.engine, policyRelayReplyMethod, p.ID)
			if reply.ErrorCode != -32012 || reply.Error != remoteManagementDeniedMessage {
				t.Fatalf("policy denial reply = %+v", reply)
			}
		}
	})
	t.Run("settings", func(t *testing.T) {
		h, p := rmSettings(t)
		source := bindRemoteManagementTestSource(h.b)
		grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
		h.b.forwardEngineNotificationFrom(source, "engine:settings-request", rmJSON(t, settings.Relay{ID: "settings-code", Method: "apply", Caller: "peer", Request: p}))
		select {
		case raw := <-h.settingsReplies:
			var reply rmReply
			if err := json.Unmarshal(raw, &reply); err != nil {
				t.Fatal(err)
			}
			if reply.ErrorCode != -32012 || reply.Error != remoteManagementDeniedMessage {
				t.Fatalf("settings denial reply = %+v", reply)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("no settings denial reply")
		}
	})
	t.Run("switch", func(t *testing.T) {
		h, p := rmSwitch(t, "switch-code")
		source := bindRemoteManagementTestSource(h.b)
		grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
		h.b.forwardEngineNotificationFrom(source, "engine:switch-request", rmJSON(t, map[string]any{"id": "switch-code", "method": settings.SwitchMethod, "caller": "peer", "params": p}))
		reply := rmWaitReply(t, h.engine, "engine:switch-reply", "switch-code")
		if reply.ErrorCode != -32012 || reply.Error != remoteManagementDeniedMessage {
			t.Fatalf("switch denial reply = %+v", reply)
		}
		detail := switchErrorDetail(&remoteManagementDeniedError{}, settings.SwitchErrorStorage)
		if detail.Code != -32012 || remoteManagementDeniedWire != "remote_management_denied" {
			t.Fatalf("switch error mapping = %+v", detail)
		}
	})
}

func TestRemoteManagementOriginRelaysPreserveCode(t *testing.T) {
	for _, method := range []string{methodPolicySet, methodSetAvailability, "engine:apply-settings", settings.SwitchMethod, "engine:remote-management-get"} {
		t.Run(method, func(t *testing.T) {
			b, out := newPolicyBroker(t)
			worker, codec := newTestRPCWorkerPipe(t)
			b.setEngineMgr(worker)
			go func() {
				for {
					request, err := codec.Read()
					if err != nil {
						return
					}
					if request.IsRequest() {
						if err := codec.RespondError(request.ID, -32012, remoteManagementDeniedMessage); err != nil {
							t.Error(err)
						}
						return
					}
				}
			}()
			b.handleMessage(clientRequest(1, method, map[string]any{"nodeId": "peer", "engine": "ollama", "requestId": "origin"}))
			response := out.waitFrame(t, method+" origin denial", isResponse(1))
			if response.Error == nil || response.Error.Code != -32012 || response.Error.Message != remoteManagementDeniedMessage {
				t.Fatalf("origin %s = %+v", method, response)
			}
		})
	}
}

func TestRemoteManagementGrantedLocalVeto(t *testing.T) {
	t.Run("availability resume", func(t *testing.T) {
		rb := newRoutedBroker(t)
		r := rb.b.nodePolicyRuntime()
		r.mu.Lock()
		r.current.Availability, r.current.PausedBy = nodepolicy.Paused, nodepolicy.PausedLocally
		r.mu.Unlock()
		_, err := rb.b.servePolicyRelay(context.Background(), policyRelayRequest{Method: "set-availability", Caller: "pinned-peer", Params: rmJSON(t, policyParams{State: string(nodepolicy.Available)})})
		if policyErrorCode(err) != -32011 {
			t.Fatalf("granted resume veto = %v (%d)", err, policyErrorCode(err))
		}
	})
	t.Run("settings pending resume", func(t *testing.T) {
		h, _ := rmSettings(t)
		h.pendingResume(t)
		h.pause(nodepolicy.PausedLocally)
		_, err := h.b.applyEngineSettings(context.Background(), h.changedRequest(t, "--veto"), "peer")
		if policyErrorCode(err) != -32011 {
			t.Fatalf("granted settings veto = %v (%d)", err, policyErrorCode(err))
		}
	})
	t.Run("switch pause", func(t *testing.T) {
		h, p := rmSwitch(t, "veto-switch")
		r := h.b.nodePolicyRuntime()
		r.mu.Lock()
		r.current.Availability, r.current.PausedBy = nodepolicy.Paused, nodepolicy.PausedLocally
		r.mu.Unlock()
		_, err := h.b.acceptEngineSwitch(context.Background(), p, "peer")
		if err == nil || switchErrorDetail(err, "").Code != -32011 {
			t.Fatalf("granted switch veto = %v", err)
		}
	})
	t.Run("switch target Off", func(t *testing.T) {
		h, p := rmSwitch(t, "off-switch")
		h.failStep = settings.SwitchStepTargetStart
		if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
			t.Fatal(err)
		}
		r := h.receipt(t, p.RequestID)
		if r.Error == nil || r.Error.Code != -32011 {
			t.Fatalf("granted target Off veto = %+v", r)
		}
	})
}

func TestRemoteManagementAcceptedCleanupSurvivesRevocation(t *testing.T) {
	t.Run("settings rebind has no caller", func(t *testing.T) {
		h, p := rmSettings(t)
		source := bindRemoteManagementTestSource(h.b)
		grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
		h.b.settingsRelayMu.Lock()
		h.b.activeSettings = map[string]activeSettingsOperation{p.RequestID: {Engine: p.Engine, Port: p.Settings.ProxyPort}}
		h.b.settingsRelayMu.Unlock()
		h.b.forwardEngineNotificationFrom(source, "engine:settings-request", rmJSON(t, settings.Relay{ID: "rebind", Method: "rebind", Request: p}))
		select {
		case raw := <-h.settingsReplies:
			var reply rmReply
			if err := json.Unmarshal(raw, &reply); err != nil {
				t.Fatal(err)
			}
			if reply.ErrorCode != 0 || reply.Error != "" || h.proxyRebinds.Load() != 1 {
				t.Fatalf("accepted rebind = %+v", reply)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("no rebind reply")
		}
	})
	t.Run("accepted switch completes and releases", func(t *testing.T) {
		h, p := rmSwitch(t, "cleanup-switch")
		reached, release := holdSwitchCompletion(t, h)
		if _, err := h.b.acceptEngineSwitch(context.Background(), p, "peer"); err != nil {
			t.Fatal(err)
		}
		requireGateOpens(t, reached, "accepted switch completion")
		grantRemoteManagement(t, h.b.clusterDir, "cluster", "target", "peer", false)
		release()
		if r := h.receipt(t, p.RequestID); !r.Terminal || r.Phase != settings.SwitchCompleted {
			t.Fatalf("accepted switch abandoned: %+v", r)
		}
		steps := switchSteps(t, h.engine)
		if len(steps) == 0 || steps[len(steps)-1] != settings.SwitchStepRelease {
			t.Fatalf("accepted switch cleanup = %v", steps)
		}
	})
}

// Audit handlers are swapped only in serial tests; the broker does not retain
// them. The fields asserted here are the complete decision contract.
func TestRemoteManagementDecisionAudit(t *testing.T) {
	b, _ := newPolicyBroker(t)
	clustertrusttest.Join(t, b.clusterDir, "cluster", "self-node", "peer")
	grantRemoteManagement(t, b.clusterDir, "cluster", "self-node", "peer", false)
	var output syncBuffer
	previous := slog.Default()
	previousWriter, previousFlags := log.Writer(), log.Flags()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&output, nil)))
	defer func() { slog.SetDefault(previous); log.SetOutput(previousWriter); log.SetFlags(previousFlags) }()
	rmDeny(t, b.authorizeRemoteManagement("peer", "settings", "receipt-1"))
	var record map[string]any
	output.mu.Lock()
	data := append([]byte(nil), output.buf.Bytes()...)
	output.mu.Unlock()
	if err := json.Unmarshal(data, &record); err != nil {
		t.Fatal(err)
	}
	if record["caller"] != "peer" || record["target"] != "self-node" || record["family"] != "settings" || record["outcome"] != "denied" || record["operation"] != "receipt-1" || record["revision"] != "1" || record["reason"] == "" {
		t.Fatalf("decision fields = %+v", record)
	}
}

func TestRemoteManagementUnavailablePermissionFailsClosed(t *testing.T) {
	b, _ := newPolicyBroker(t)
	clustertrusttest.Join(t, b.clusterDir, "cluster", "self-node", "peer")
	grantRemoteManagement(t, b.clusterDir, "cluster", "self-node", "peer", true)
	if err := b.authorizeRemoteManagement("peer", "policy", ""); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(b.clusterDir, clustertrust.RemoteManagementFile), []byte(`{"version":999}`), 0600); err != nil {
		t.Fatal(err)
	}
	rmDeny(t, b.authorizeRemoteManagement("peer", "policy", ""))
	if err := b.authorizeRemoteManagement("", "policy", ""); err != nil {
		t.Fatalf("local caller = %v", err)
	}
}
