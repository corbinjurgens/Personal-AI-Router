// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"nvpair-shared/durablefile"
	"nvpair-shared/nodepolicy"
)

func TestLocalPolicyGetCarriesCredentialReferencesWithoutResolvedSecrets(t *testing.T) {
	const freeSecret = "test-only-free-secret-value-6f7239"
	const paidSecret = "test-only-paid-secret-value-8a1042"
	t.Setenv("NVPAIR_PROVIDER_SECRET_FREE", freeSecret)
	t.Setenv("NVPAIR_PROVIDER_SECRET_PAID", paidSecret)
	rb := newRoutedBroker(t)

	rb.b.handleMessage(clientRequest(1, methodPolicyGet, nil))
	response := rb.out.waitFrame(t, "local policy:get", isResponse(1))
	if response.Error != nil {
		t.Fatalf("policy:get: %+v", response.Error)
	}
	var got policyGetResult
	if err := json.Unmarshal(response.Result, &got); err != nil {
		t.Fatalf("decode local policy:get: %v", err)
	}
	want := map[string]nodepolicy.CredentialRef{
		"free-acct": "NVPAIR_PROVIDER_SECRET_FREE",
		"paid-acct": "NVPAIR_PROVIDER_SECRET_PAID",
	}
	if len(got.Policy.Routing.Accounts) != len(want) {
		t.Fatalf("policy:get returned %d accounts, want %d", len(got.Policy.Routing.Accounts), len(want))
	}
	for _, account := range got.Policy.Routing.Accounts {
		credential, ok := want[account.ID]
		if !ok || account.Credential != credential {
			t.Errorf("account %q did not retain its credential reference", account.ID)
		}
		delete(want, account.ID)
	}
	if len(want) != 0 {
		t.Errorf("policy:get omitted expected accounts: %v", want)
	}
	for _, secret := range []string{freeSecret, paidSecret} {
		if bytes.Contains(response.Result, []byte(secret)) {
			t.Error("local policy:get exposed a resolved secret canary")
		}
	}
}

func newRoutedActivatedBroker(t *testing.T) *activatedBroker {
	t.Helper()
	rb := newRoutedBroker(t)
	setClock(rb.b, 5)
	pp, proxy, acks := attachAckingProxy(t, rb.b)
	rb.b.replayPolicyToProxy(context.Background(), pp)
	ab := &activatedBroker{b: rb.b, out: rb.out, pp: pp, proxy: proxy, acks: acks, path: rb.path}
	ab.wantActivation(t, "12", activationActive, "")
	return ab
}

func TestCeilingReductionFailedPushReportsPersistedAndActiveRevisions(t *testing.T) {
	ab := newRoutedActivatedBroker(t)
	ab.refuseSetPolicy()
	ab.b.handleMessage(clientRequest(1, methodPolicySet, map[string]any{
		"policy": paidCeilingDoc(t, ab.b, "5"), "expectedPersistedRevision": "12",
	}))
	response := ab.out.waitFrame(t, "ceiling reduction response", isResponse(1))
	if response.Error != nil {
		t.Fatalf("durably saved ceiling reduction returned an RPC error: %+v", response.Error)
	}
	var result policySetResult
	if err := json.Unmarshal(response.Result, &result); err != nil {
		t.Fatalf("decode ceiling reduction response: %v", err)
	}
	if result.Policy.Revision != "13" || result.ActiveRevision != "12" || result.Activation != activationError {
		t.Fatalf("ceiling reduction response = revision %q, %+v; want persisted 13, active 12 and error", result.Policy.Revision, result.policyActivation)
	}
	if result.Policy.Routing.Accounts[1].MonthlyCeiling != "5" {
		t.Fatal("ceiling reduction response did not contain monthly ceiling 5")
	}
	if !strings.Contains(result.ActivationError, "13") || !strings.Contains(result.ActivationError, "refused") {
		t.Fatalf("activation diagnostic = %q, want the failed revision and refusal", result.ActivationError)
	}
	get := ab.wantActivation(t, "12", activationError, "refused")
	if !reflect.DeepEqual(get.Policy, result.Policy) {
		t.Error("policy:get does not report the persisted ceiling reduction")
	}
	announced, activation := lastPolicyChanged(t, ab.out)
	if !reflect.DeepEqual(announced, result.Policy) || activation.ActiveRevision != "12" || activation.Activation != activationError {
		t.Errorf("policy:changed did not report the reduced document, active revision 12 and activation error: %+v", activation)
	}
	if held := ab.acks.heldRevision(); held != "12" {
		t.Errorf("proxy holds revision %q after refusal, want 12", held)
	}
	var pushed nodepolicy.Policy
	for _, call := range ab.proxy.snapshot() {
		if call.method == nodepolicy.MethodSetPolicy {
			if err := json.Unmarshal(call.params, &pushed); err != nil {
				t.Fatalf("decode pushed policy: %v", err)
			}
		}
	}
	if !reflect.DeepEqual(pushed, result.Policy) {
		t.Error("the failed push did not attempt the reduced document at revision 13")
	}
	data, err := os.ReadFile(ab.path)
	if err != nil {
		t.Fatalf("read persisted reduction: %v", err)
	}
	var stored nodepolicy.Policy
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatalf("decode persisted reduction: %v", err)
	}
	if !reflect.DeepEqual(stored, result.Policy) {
		t.Error("failed activation lost the persisted reduced document")
	}
}

func TestPeerPolicySetRejectsDeploymentAndExistingAccountCeilingChanges(t *testing.T) {
	test := func(name string, edit func(map[string]any)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			baseline := rb.baseline(t)
			doc := rb.peerSnapshot(t)
			var routing map[string]any
			if err := json.Unmarshal(mustJSON(rb.b.localPolicy().Policy.Routing), &routing); err != nil {
				t.Fatalf("decode stored routing: %v", err)
			}
			edit(routing)
			doc["routing"] = routing
			runtime := rb.b.nodePolicyRuntime()
			runtime.acquirePush(context.Background())
			defer runtime.releasePush()

			if result, errText := rb.relaySet(t, doc); errText == "" {
				t.Fatalf("peer changed a protected routing definition: %s", result)
			}
			baseline.requireUnchanged(t)
			fresh := freshBroker(t, rb.path).localPolicy()
			if fresh.RecoveryRequired || !reflect.DeepEqual(fresh.Policy, baseline.policy) {
				t.Error("restart did not retain the full original policy after the rejected peer edit")
			}
		})
	}
	test("deployment definition", func(routing map[string]any) {
		routing["deployments"].([]any)[0].(map[string]any)["model"] = "peer-replacement-model"
	})
	test("existing paid account ceiling", func(routing map[string]any) {
		routing["accounts"].([]any)[1].(map[string]any)["monthlyCeiling"] = "5"
	})
}

func TestCeilingReductionPublicationFaultBoundariesRestartWholeDocument(t *testing.T) {
	boundaries := append(publicationBoundaries(),
		publicationBoundary{"after file sync", faultingTemp(faultTemp{afterSyncErr: errPublish}), false},
		publicationBoundary{"after directory or platform flush", durablefile.Replacer{SyncDir: failAfterPolicyDirectoryFlush}, true},
	)
	for _, boundary := range boundaries {
		t.Run(boundary.name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			setClock(rb.b, 5)
			old := rb.b.localPolicy().Policy
			oldBytes := rb.fileBytes(t)
			reduction := paidCeilingDoc(t, rb.b, "5")
			wantNew, err := nodepolicy.Parse(reduction)
			if err != nil {
				t.Fatalf("parse proposed reduction: %v", err)
			}
			wantNew.Revision = "13"
			setReplacer(rb.b, boundary.rep)
			// Holding the push slot keeps a push that a failed publication wrongly
			// scheduled marked pending, so it is visible without waiting for it.
			r := rb.b.nodePolicyRuntime()
			r.acquirePush(context.Background())
			_, err = setCurrent(rb.b, reduction)
			r.mu.Lock()
			pending := len(r.pendingPush)
			r.mu.Unlock()
			r.releasePush()
			setReplacer(rb.b, durablefile.Replacer{})
			if !errors.Is(err, errPublish) {
				t.Fatalf("ceiling reduction error = %v, want the injected publication failure", err)
			}
			if got := rb.b.localPolicy().Policy; !reflect.DeepEqual(got, old) {
				t.Error("failed publication changed the running policy")
			}
			if pending != 0 || policyPushes(rb.proxy) != 0 || len(policyChangedRevisions(t, rb.out)) != 0 {
				t.Errorf("failed publication scheduled %d pushes, sent %d or announced the unacknowledged reduction", pending, policyPushes(rb.proxy))
			}
			fresh := freshBroker(t, rb.path).localPolicy()
			if fresh.RecoveryRequired {
				t.Fatal("restart required recovery instead of loading a complete document")
			}
			want := old
			if boundary.landed {
				want = wantNew
			} else if !bytes.Equal(rb.fileBytes(t), oldBytes) {
				t.Error("failure before replacement changed the original file bytes")
			}
			if !reflect.DeepEqual(fresh.Policy, want) {
				t.Errorf("restart did not load the complete expected document at revision %s", want.Revision)
			}
			if revisionNumber(t, fresh.Policy) < 12 {
				t.Errorf("restart regressed revision to %q from 12", fresh.Policy.Revision)
			}
		})
	}
}

// Windows' default replacement has already written through when SyncDir is
// called. Elsewhere, perform the directory sync before injecting the failure.
func failAfterPolicyDirectoryFlush(dir string) error {
	if runtime.GOOS == "windows" {
		return errPublish
	}
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	syncErr := f.Sync()
	closeErr := f.Close()
	if syncErr != nil {
		return syncErr
	}
	if closeErr != nil {
		return closeErr
	}
	return errPublish
}

func TestAcknowledgedCeilingReductionSurvivesRestart(t *testing.T) {
	test := func(name string, refusePush bool) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			ab := newRoutedActivatedBroker(t)
			if refusePush {
				ab.refuseSetPolicy()
			}
			ab.b.handleMessage(clientRequest(1, methodPolicySet, map[string]any{
				"policy": paidCeilingDoc(t, ab.b, "5"), "expectedPersistedRevision": "12",
			}))
			response := ab.out.waitFrame(t, "acknowledged ceiling reduction", isResponse(1))
			if response.Error != nil {
				t.Fatalf("ceiling reduction was not acknowledged: %+v", response.Error)
			}
			var result policySetResult
			if err := json.Unmarshal(response.Result, &result); err != nil {
				t.Fatalf("decode acknowledged reduction: %v", err)
			}
			if result.Policy.Revision != "13" || result.Policy.Routing.Accounts[1].MonthlyCeiling != "5" {
				t.Fatal("success response did not acknowledge ceiling 5 at revision 13")
			}
			if refusePush {
				if result.Activation != activationError || result.ActiveRevision != "12" {
					t.Fatalf("post-publication push fault did not leave revision 12 active: %+v", result.policyActivation)
				}
			} else if result.Activation != activationActive || result.ActiveRevision != "13" {
				t.Fatalf("acknowledged proxy did not activate revision 13: %+v", result.policyActivation)
			}
			fresh := freshBroker(t, ab.path).localPolicy()
			if fresh.RecoveryRequired || !reflect.DeepEqual(fresh.Policy, result.Policy) {
				t.Fatal("restart did not recover the complete success-returned reduction at revision 13")
			}
			if got := fresh.Policy.Routing.Accounts[1].MonthlyCeiling; got != "5" {
				t.Errorf("restart restored monthly ceiling %q, want the acknowledged reduction 5", got)
			}
		})
	}
	test("proxy acknowledges", false)
	test("proxy refuses after durable publication", true)
}
