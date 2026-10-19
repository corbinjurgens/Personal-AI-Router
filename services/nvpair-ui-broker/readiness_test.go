// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// strictFixture is an activated broker whose proxy can exit (close done), with
// a fake engine-manager that records every node projection.
type strictFixture struct {
	*activatedBroker
	done   chan struct{}
	engine *fakeWorker
}

func newStrictFixture(t *testing.T) *strictFixture {
	t.Helper()
	return newStrictFixtureWithFile(t, nil)
}

// newStrictFixtureWithFile starts from a policy file holding data, or from no
// file when data is nil.
func newStrictFixtureWithFile(t *testing.T, data []byte) *strictFixture {
	t.Helper()
	b, out := newPolicyBroker(t)
	path := filepath.Join(t.TempDir(), nodepolicy.FileName)
	if data != nil {
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatalf("write policy file: %v", err)
		}
	}
	b.loadNodePolicy(path)
	setClock(b, 5000)
	pp, proxy, acks := attachAckingProxy(t, b)
	done := make(chan struct{})
	pp.done = done
	engine := attachFakeEngine(t, b)
	b.replayPolicyToProxy(context.Background(), pp)
	ab := &activatedBroker{b: b, out: out, pp: pp, proxy: proxy, acks: acks, path: path}
	ab.wantActivation(t, "0", activationActive, "")
	seedStrictIdentity(b, "ollama", "lmstudio")
	return &strictFixture{activatedBroker: ab, done: done, engine: engine}
}

// seedStrictIdentity gives each engine the identity strictCapability names, so
// the identity fence (model_identity.go) passes its reports.
func seedStrictIdentity(b *Broker, engines ...string) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.engineIdentity == nil {
		r.engineIdentity = map[string]engineManagerIdentity{}
	}
	if r.settingsIdentity == nil {
		r.settingsIdentity = map[string]string{}
	}
	for _, engine := range engines {
		r.engineIdentity[engine] = engineManagerIdentity{
			EngineRunID:  "engine-run-1",
			LaunchDigest: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
			ModelDigests: map[string]string{"m:latest": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
		}
		r.settingsIdentity[engine] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	}
}

// strictCapability is a valid version 1 capability, one model per name, each
// on engine.
func strictCapability(t *testing.T, engine string, models ...string) *nodepolicy.StrictAdmissionCapability {
	t.Helper()
	c := &nodepolicy.StrictAdmissionCapability{Version: nodepolicy.StrictAdmissionVersion}
	for _, model := range models {
		c.Models = append(c.Models, nodepolicy.StrictAdmissionModel{
			Engine:        engine,
			Model:         model,
			MaxConcurrent: 1,
			Identity: nodepolicy.StrictModelIdentity{
				EngineRunID:         "engine-run-1",
				SettingsFingerprint: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
				ModelDigest:         "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
			},
		})
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("test capability is invalid: %v", err)
	}
	return c
}

// reportStrict delivers a readiness report the way the proxy process does.
func reportStrict(t *testing.T, b *Broker, revision, runID string, capability *nodepolicy.StrictAdmissionCapability) {
	t.Helper()
	params, err := json.Marshal(nodepolicy.StrictAdmissionReadiness{PolicyRevision: revision, ProxyRunID: runID, Capability: capability})
	if err != nil {
		t.Fatalf("encode readiness: %v", err)
	}
	if !b.routeProcessScopedProxyNotification(nodepolicy.NotifyStrictReadiness, params) {
		t.Fatalf("%s was not taken as a process-scoped proxy notification", nodepolicy.NotifyStrictReadiness)
	}
}

// waitLastProjection waits until the newest node projection satisfies want and
// returns it.
func waitLastProjection(t *testing.T, engine *fakeWorker, what string, want func(nodepolicy.Advertisement) bool) nodepolicy.Advertisement {
	t.Helper()
	var last nodepolicy.Advertisement
	engine.waitFor(what, func([]policyCall) bool {
		got := nodeProjections(t, engine)
		if len(got) == 0 {
			return false
		}
		last = got[len(got)-1]
		return want(last)
	})
	return last
}

func waitAdvertised(t *testing.T, f *strictFixture, want *nodepolicy.StrictAdmissionCapability) nodepolicy.Advertisement {
	t.Helper()
	return waitLastProjection(t, f.engine, "the strict capability to be advertised", func(a nodepolicy.Advertisement) bool {
		return reflect.DeepEqual(a.StrictAdmission, want)
	})
}

func waitWithdrawn(t *testing.T, f *strictFixture) nodepolicy.Advertisement {
	t.Helper()
	return waitLastProjection(t, f.engine, "the strict capability to be withdrawn", func(a nodepolicy.Advertisement) bool {
		return a.StrictAdmission == nil
	})
}

// requireNotReady fails when the broker would advertise a capability now.
func requireNotReady(t *testing.T, b *Broker, why string) {
	t.Helper()
	if got := b.StrictReadiness(); got.Capability != nil {
		t.Fatalf("%s: the broker advertises %+v", why, got)
	}
}

// advertise reports readiness for the current revision from the acknowledged
// run and waits for the projection to carry it.
func (f *strictFixture) advertise(t *testing.T, capability *nodepolicy.StrictAdmissionCapability) nodepolicy.Advertisement {
	t.Helper()
	reportStrict(t, f.b, f.b.localPolicy().Policy.Revision, f.acks.runID, capability)
	return waitAdvertised(t, f, capability)
}

func TestStrictReadinessAdvertisedOnlyWithMatchingRevisionAndRun(t *testing.T) {
	capability := strictCapability(t, "ollama", "m:latest")

	t.Run("report from another proxy run is discarded", func(t *testing.T) {
		f := newStrictFixture(t)
		reportStrict(t, f.b, "0", "run-other", capability)
		requireNotReady(t, f.b, "another run's report")
	})
	t.Run("report for another revision is not advertised", func(t *testing.T) {
		f := newStrictFixture(t)
		reportStrict(t, f.b, "4999", f.acks.runID, capability)
		requireNotReady(t, f.b, "another revision's report")
	})
	t.Run("matching report is advertised", func(t *testing.T) {
		f := newStrictFixture(t)
		got := f.advertise(t, capability)
		if got.Revision != "0" {
			t.Fatalf("projection revision = %q, want the reported 0", got.Revision)
		}
		readiness := f.b.StrictReadiness()
		if readiness.PolicyRevision != "0" || readiness.ProxyRunID != f.acks.runID {
			t.Fatalf("StrictReadiness fences = %q/%q, want 0/%s", readiness.PolicyRevision, readiness.ProxyRunID, f.acks.runID)
		}
		if stored := f.b.localPolicy().Policy.Advertisement(); stored.StrictAdmission != nil {
			t.Fatalf("Policy.Advertisement set the capability by itself: %+v", stored.StrictAdmission)
		}
	})
	t.Run("another run's late report does not displace the current one", func(t *testing.T) {
		f := newStrictFixture(t)
		f.advertise(t, capability)
		reportStrict(t, f.b, "0", "run-other", nil)
		if f.b.StrictReadiness().Capability == nil {
			t.Fatal("a discarded report withdrew the current run's readiness")
		}
	})
}

func TestStrictReadinessWithdrawn(t *testing.T) {
	capability := strictCapability(t, "ollama", "m:latest")
	withdrawn := func(name string, act func(t *testing.T, f *strictFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newStrictFixture(t)
			f.advertise(t, capability)
			act(t, f)
			waitWithdrawn(t, f)
			requireNotReady(t, f.b, name)
		})
	}

	withdrawn("by a report without a capability", func(t *testing.T, f *strictFixture) {
		reportStrict(t, f.b, "0", f.acks.runID, nil)
	})
	withdrawn("by an invalid report", func(t *testing.T, f *strictFixture) {
		params := json.RawMessage(`{"policyRevision":"0","proxyRunId":"` + f.acks.runID + `","capability":{"version":1,"models":[]}}`)
		if !f.b.routeProcessScopedProxyNotification(nodepolicy.NotifyStrictReadiness, params) {
			t.Fatalf("%s was not taken", nodepolicy.NotifyStrictReadiness)
		}
	})
	withdrawn("on proxy exit", func(t *testing.T, f *strictFixture) {
		close(f.done)
	})
	withdrawn("on proxy run replacement", func(t *testing.T, f *strictFixture) {
		next, _, _ := attachAckingProxy(t, f.b)
		f.b.replayPolicyToProxy(context.Background(), next)
	})
	withdrawn("on a policy change", func(t *testing.T, f *strictFixture) {
		set := f.setIdle(t, 3)
		if set.Activation != activationActive {
			t.Fatalf("policy:set activation = %+v, want active", set.policyActivation)
		}
	})
	withdrawn("on a local pause", func(t *testing.T, f *strictFixture) {
		if _, err := f.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil {
			t.Fatalf("pause: %v", err)
		}
	})
	withdrawn("on a peer pause", func(t *testing.T, f *strictFixture) {
		if _, err := f.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedByPeer); err != nil {
			t.Fatalf("pause: %v", err)
		}
	})
	withdrawn("when its only engine is saved Off", func(t *testing.T, f *strictFixture) {
		f.b.observeIntent(map[string]bool{"ollama": false})
	})
}

func TestStrictReadinessNotAdvertisedWhileActivationIsPending(t *testing.T) {
	f := newStrictFixture(t)
	f.advertise(t, strictCapability(t, "ollama", "m:latest"))
	f.b.nodePolicy.mu.Lock()
	f.b.nodePolicy.activationTimeout = 100 * time.Millisecond
	f.b.nodePolicy.mu.Unlock()
	// The proxy activates the revision but its acknowledgement never arrives.
	f.proxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
		if method == nodepolicy.MethodSetPolicy {
			if _, errText := f.acks.answer(method, params); errText != "" {
				return nil, errText
			}
			return nil, holdReply
		}
		return f.acks.answer(method, params)
	})
	set := f.setIdle(t, 3)
	if set.Activation == activationActive {
		t.Fatalf("policy:set reported %+v without an acknowledgement", set.policyActivation)
	}
	reportStrict(t, f.b, set.Policy.Revision, f.acks.runID, strictCapability(t, "ollama", "m:latest"))
	requireNotReady(t, f.b, "the new revision is not acknowledged")
	if got := waitWithdrawn(t, f); got.Revision != set.Policy.Revision {
		t.Fatalf("withdrawal projection revision = %q, want %s", got.Revision, set.Policy.Revision)
	}
}

// During policy recovery availability changes take no revision, so only the
// availability itself withdraws the capability.
func TestStrictReadinessWithdrawnOnPauseDuringRecovery(t *testing.T) {
	f := newStrictFixtureWithFile(t, []byte(`{"version":1,"availability":"sideways"}`))
	if !f.b.localPolicy().RecoveryRequired {
		t.Fatal("the invalid policy file did not put the node in recovery")
	}
	if _, err := f.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally); err != nil {
		t.Fatalf("resume: %v", err)
	}
	capability := strictCapability(t, "ollama", "m:latest")
	revision := f.advertise(t, capability).Revision
	if _, err := f.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil {
		t.Fatalf("pause: %v", err)
	}
	if got := waitWithdrawn(t, f); got.Revision != revision {
		t.Fatalf("the recovery pause took revision %q; the test needs it to keep %s", got.Revision, revision)
	}
	requireNotReady(t, f.b, "paused during recovery")
}

// A report can reach the broker before the acknowledgement it follows, while
// the new incarnation has acknowledged nothing yet.
func TestStrictReadinessReportedBeforeTheFirstAcknowledgement(t *testing.T) {
	capability := strictCapability(t, "ollama", "m:latest")
	restart := func(t *testing.T, f *strictFixture, reportRun func(next *proxyAcks) string) *proxyAcks {
		t.Helper()
		close(f.done)
		waitWithdrawn(t, f)
		next, nextProxy, nextAcks := attachAckingProxy(t, f.b)
		nextProxy.setAnswer(func(method string, params json.RawMessage) (any, string) {
			if method == nodepolicy.MethodSetPolicy {
				reportStrict(t, f.b, "0", reportRun(nextAcks), capability)
			}
			return nextAcks.answer(method, params)
		})
		f.b.replayPolicyToProxy(context.Background(), next)
		if got := f.b.localPolicy(); got.Activation != activationActive {
			t.Fatalf("replay left activation %+v", got.policyActivation)
		}
		return nextAcks
	}

	t.Run("the new run's report is advertised once acknowledged", func(t *testing.T) {
		f := newStrictFixture(t)
		f.advertise(t, capability)
		nextAcks := restart(t, f, func(next *proxyAcks) string { return next.runID })
		waitAdvertised(t, f, capability)
		if got := f.b.StrictReadiness(); got.ProxyRunID != nextAcks.runID {
			t.Fatalf("advertised under run %q, want %s", got.ProxyRunID, nextAcks.runID)
		}
	})
	t.Run("the exited run's report is never advertised", func(t *testing.T) {
		f := newStrictFixture(t)
		f.advertise(t, capability)
		exited := f.acks.runID
		restart(t, f, func(*proxyAcks) string { return exited })
		requireNotReady(t, f.b, "the exited run's report held across the new acknowledgement")
	})
}

func TestStrictReadinessReadvertisedAfterResume(t *testing.T) {
	f := newStrictFixture(t)
	capability := strictCapability(t, "ollama", "m:latest")
	f.advertise(t, capability)
	if _, err := f.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally); err != nil {
		t.Fatalf("pause: %v", err)
	}
	if _, err := f.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedLocally); err != nil {
		t.Fatalf("resume: %v", err)
	}
	requireNotReady(t, f.b, "the held report names the revision before the pause")
	got := f.advertise(t, capability)
	if got.Revision != f.b.localPolicy().Policy.Revision {
		t.Fatalf("readvertised at revision %q, want the resumed %s", got.Revision, f.b.localPolicy().Policy.Revision)
	}
}

func TestStrictReadinessDropsModelsOfEnginesSavedOff(t *testing.T) {
	f := newStrictFixture(t)
	both := strictCapability(t, "ollama", "m:latest")
	both.Models = append(both.Models, strictCapability(t, "lmstudio", "n").Models...)
	f.advertise(t, both)

	f.b.observeIntent(map[string]bool{"ollama": false, "lmstudio": true})
	waitAdvertised(t, f, strictCapability(t, "lmstudio", "n"))

	f.b.observeIntent(map[string]bool{"ollama": true, "lmstudio": true})
	waitAdvertised(t, f, both)
}

func TestStrictReadinessReplayedAfterProxyRestart(t *testing.T) {
	f := newStrictFixture(t)
	capability := strictCapability(t, "ollama", "m:latest")
	f.advertise(t, capability)
	close(f.done)
	waitWithdrawn(t, f)

	next, _, nextAcks := attachAckingProxy(t, f.b)
	f.b.replayPolicyToProxy(context.Background(), next)
	if got := f.b.localPolicy(); got.Activation != activationActive {
		t.Fatalf("replay left activation %+v", got.policyActivation)
	}
	reportStrict(t, f.b, "0", f.acks.runID, capability)
	requireNotReady(t, f.b, "the exited run's late report")

	reportStrict(t, f.b, "0", nextAcks.runID, capability)
	waitAdvertised(t, f, capability)
	if got := f.b.StrictReadiness(); got.ProxyRunID != nextAcks.runID {
		t.Fatalf("advertised under run %q, want the restarted %s", got.ProxyRunID, nextAcks.runID)
	}
}

// Saves race readiness reports for current and earlier revisions. Every
// projection that carries a capability must carry the one reported for its
// own revision, and the last one the report for the final revision.
func TestStrictReadinessConcurrentSavesNeverAdvertiseStale(t *testing.T) {
	f := newStrictFixture(t)
	forRevision := func(t *testing.T, revision string) *nodepolicy.StrictAdmissionCapability {
		return strictCapability(t, "ollama", "m-"+revision)
	}
	const writers, saves = 4, 4
	stop := make(chan struct{})
	reporterDone := make(chan struct{})
	go func() {
		defer close(reporterDone)
		var seen []string
		for {
			select {
			case <-stop:
				return
			default:
			}
			revision := f.b.localPolicy().Policy.Revision
			seen = append(seen, revision)
			reportStrict(t, f.b, revision, f.acks.runID, forRevision(t, revision))
			stale := seen[len(seen)/2]
			reportStrict(t, f.b, stale, f.acks.runID, forRevision(t, stale))
		}
	}()
	var wg sync.WaitGroup
	for w := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for s := 0; s < saves; {
				_, err := f.b.dispatchPolicy(context.Background(), methodPolicySet, policyParams{
					Policy:                    mustJSON(map[string]any{"version": nodepolicy.SchemaVersion, "idle": map[string]any{"unloadAfterMinutes": 1 + w*saves + s}}),
					ExpectedPersistedRevision: mustJSON(f.b.localPolicy().Policy.Revision),
				})
				var conflict *policyConflictError
				switch {
				case err == nil:
					s++
				case errors.As(err, &conflict):
				default:
					t.Errorf("policy:set: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(stop)
	<-reporterDone

	final := f.b.localPolicy().Policy.Revision
	f.advertise(t, forRevision(t, final))
	for i, p := range nodeProjections(t, f.engine) {
		if p.StrictAdmission == nil {
			continue
		}
		if want := forRevision(t, p.Revision); !reflect.DeepEqual(p.StrictAdmission, want) {
			t.Fatalf("projection %d at revision %s advertised %+v, the readiness of another revision", i, p.Revision, p.StrictAdmission.Models)
		}
	}
}
