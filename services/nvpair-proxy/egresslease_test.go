// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func egressJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// The eligible fixture bypasses only policy parsing, inside tests. There is no
// production enable switch. D/E remain real, backed by a disposable ledger and
// an in-memory HTTP transport that cannot reach a provider.
func egressFixture(t *testing.T) (*Proxy, *chatAccountFixture, *atomic.Int64, chatAccountSnapshot) {
	t.Helper()
	f := newChatAccountFixture(t)
	for i := range f.routing.Deployments {
		f.routing.Deployments[i].Enabled = true
	}
	pol := policyAt("7")
	pol.Routing = f.routing
	pol.Routing.Limits = nodepolicy.Default().Routing.Limits
	pol.Routing.Listener = nodepolicy.Default().Routing.Listener
	pol.Routing.Tiers = map[string]nodepolicy.TierRouting{"weak": {RemoteAllowed: true, PaidAllowed: true}}
	pol.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryAPI, Deployment: "chat-a"}}}
	p := NewProxy(NewCodec(&recordingWriter{}))
	p.accounts = f.ledger
	clock := &atomic.Int64{}
	clock.Store(int64(time.Second))
	p.egress = newEgressLease(func() time.Duration { return time.Duration(clock.Load()) })
	token, err := p.beginPublicActivation(pol)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.ledger.applyRouting(pol.Routing, "7"); err != nil {
		t.Fatal(err)
	}
	p.completePublicActivation(token, nil)
	snapshot, err := f.ledger.prepareChatAccount("chat-a")
	if err != nil {
		t.Fatal(err)
	}
	return p, f, clock, snapshot
}

func egressGrant(t *testing.T, p *Proxy, sequence int64, deadline time.Duration) nodepolicy.RenewEgressLeaseResult {
	t.Helper()
	ack, err := p.renewEgressLease(egressJSON(t, nodepolicy.RenewEgressLeaseParams{
		RunID: p.runID, Revision: p.admission.currentPolicy().Revision,
		Sequence: strconv.FormatInt(sequence, 10), NotAfterNanos: strconv.FormatInt(int64(deadline), 10),
	}))
	if err != nil {
		t.Fatal(err)
	}
	return ack
}

func egressRevoke(t *testing.T, p *Proxy, sequence int64) {
	t.Helper()
	ack, err := p.revokeEgressLease(egressJSON(t, nodepolicy.RevokeEgressLeaseParams{RunID: p.runID, Sequence: strconv.FormatInt(sequence, 10)}))
	if err != nil || ack.RunID != p.runID || ack.Sequence != strconv.FormatInt(sequence, 10) {
		t.Fatalf("withdrawal acknowledgement = %+v, err %v", ack, err)
	}
}

func egressUnsentHold(t *testing.T, f *chatAccountFixture) {
	t.Helper()
	if f.calls.Load() != 0 {
		t.Fatalf("public Do calls = %d, want zero", f.calls.Load())
	}
	s, err := readAccountStore(f.ledger.storePath)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Holds) != 1 {
		t.Fatalf("retained durable holds = %d, want one", len(s.Holds))
	}
	h := s.Holds[0]
	if h.State != holdReserved || h.Outcome != "" || h.Usage != nil || h.Actual != "" || h.QualificationChecked {
		t.Fatalf("pre-send hold was classified or released: %+v", h)
	}
	if len(s.Accounts) != 1 || s.Accounts[0].Cooldown != nil {
		t.Fatal("pre-send refusal changed account cooldown")
	}
}

func TestEgressLeaseStartup(t *testing.T) {
	p := NewProxy(NewCodec(&recordingWriter{}))
	if p.egress == nil || p.egress.notAfter != 0 || p.egress.sequence != 0 || p.egress.ready || p.egress.now() < 0 {
		t.Fatal("new proxy inherited public authority or lacks its monotonic epoch")
	}
	if p.publicPermission("weak", chatAccountSnapshot{}) == nil {
		t.Fatal("startup accepted a public attempt")
	}
}

func TestEgressLeaseExactExpiry(t *testing.T) {
	p, _, clock, snapshot := egressFixture(t)
	ack := egressGrant(t, p, 1, 6*time.Second)
	if ack.NowNanos != "1000000000" || ack.Sequence != "1" || ack.Revision != "7" || ack.RunID != p.runID {
		t.Fatalf("grant acknowledgement = %+v", ack)
	}
	clock.Store(int64(6*time.Second - 1))
	if err := p.publicPermission("weak", snapshot); err != nil {
		t.Fatalf("last live nanosecond refused: %v", err)
	}
	clock.Store(int64(6 * time.Second))
	if p.publicPermission("weak", snapshot) == nil {
		t.Fatal("exact expiry accepted a public attempt")
	}
}

func TestEgressLeaseAnchorAndDelayedGrant(t *testing.T) {
	p, f, clock, snapshot := egressFixture(t)
	if ack := egressGrant(t, p, 1, 0); ack.NowNanos != "1000000000" {
		t.Fatalf("anchor sample = %q", ack.NowNanos)
	}
	if p.publicPermission("weak", snapshot) == nil {
		t.Fatal("anchor-only renewal granted authority")
	}
	egressGrant(t, p, 2, 6*time.Second)
	clock.Store(int64(4 * time.Second))
	if ack := egressGrant(t, p, 3, 0); ack.NowNanos != "4000000000" {
		t.Fatalf("second anchor sample = %q", ack.NowNanos)
	}
	if p.egress.notAfter != int64(6*time.Second) || p.publicPermission("weak", snapshot) != nil {
		t.Fatal("anchor-only renewal changed existing authority")
	}
	// Delayed, in-order messages keep their receiver-domain deadline: receipt
	// never restarts a five-second TTL. A ledger wall clock is unrelated.
	egressGrant(t, p, 4, 6*time.Second)
	f.clock.advance(-24 * time.Hour)
	clock.Store(int64(6 * time.Second))
	if p.publicPermission("weak", snapshot) == nil {
		t.Fatal("delayed renewal or wall-clock rollback extended the lease")
	}
}

func TestEgressLeaseBindingAndHighWater(t *testing.T) {
	tests := []struct {
		name string
		edit func(*nodepolicy.RenewEgressLeaseParams)
	}{
		{"old run", func(v *nodepolicy.RenewEgressLeaseParams) { v.RunID = "old-run" }},
		{"old revision", func(v *nodepolicy.RenewEgressLeaseParams) { v.Revision = "6" }},
		{"future revision", func(v *nodepolicy.RenewEgressLeaseParams) { v.Revision = "8" }},
		{"duplicate sequence", func(v *nodepolicy.RenewEgressLeaseParams) { v.Sequence = "2" }},
		{"reordered sequence", func(v *nodepolicy.RenewEgressLeaseParams) { v.Sequence = "1" }},
		{"expired deadline", func(v *nodepolicy.RenewEgressLeaseParams) { v.NotAfterNanos = "1000000000" }},
		{"too long deadline", func(v *nodepolicy.RenewEgressLeaseParams) { v.NotAfterNanos = "6000000001" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, _, _, snapshot := egressFixture(t)
			egressGrant(t, p, 2, 5*time.Second)
			in := nodepolicy.RenewEgressLeaseParams{RunID: p.runID, Revision: "7", Sequence: "3", NotAfterNanos: "6000000000"}
			tt.edit(&in)
			if _, err := p.renewEgressLease(egressJSON(t, in)); err == nil {
				t.Fatal("invalid renewal accepted")
			}
			if p.egress.sequence != 2 || p.egress.notAfter != int64(5*time.Second) || p.publicPermission("weak", snapshot) != nil {
				t.Fatal("invalid renewal changed effective authority")
			}
		})
	}
	t.Run("withdrawal and policy changes preserve high-water", func(t *testing.T) {
		p, _, _, snapshot := egressFixture(t)
		egressGrant(t, p, 3, 6*time.Second)
		egressRevoke(t, p, 5)
		if p.publicPermission("weak", snapshot) == nil {
			t.Fatal("withdrawal left public authority")
		}
		pol := p.admission.currentPolicy()
		pol.Revision = "8"
		token, err := p.beginPublicActivation(pol)
		if err != nil {
			t.Fatal(err)
		}
		p.completePublicActivation(token, nil)
		for _, seq := range []string{"3", "4", "5"} {
			if _, err := p.renewEgressLease(egressJSON(t, nodepolicy.RenewEgressLeaseParams{RunID: p.runID, Revision: "8", Sequence: seq, NotAfterNanos: "6000000000"})); err == nil {
				t.Fatalf("sequence %s resurrected revoked authority after policy change", seq)
			}
		}
		if p.egress.sequence != 5 || p.egress.notAfter != 0 {
			t.Fatal("activation reset sequence or renewed permission")
		}
		egressGrant(t, p, 6, 6*time.Second)
	})
	t.Run("replacement refuses prior run", func(t *testing.T) {
		p, _, _, _ := egressFixture(t)
		replacement, _, _, _ := egressFixture(t)
		if _, err := replacement.renewEgressLease(egressJSON(t, nodepolicy.RenewEgressLeaseParams{RunID: p.runID, Revision: "7", Sequence: "99", NotAfterNanos: "6000000000"})); err == nil || replacement.egress.notAfter != 0 {
			t.Fatal("prior run acquired replacement authority")
		}
	})
}

func TestEgressLeaseMalformed(t *testing.T) {
	p, _, _, _ := egressFixture(t)
	for _, raw := range []string{
		`{}`, `{"runId":"run","revision":"7","sequence":1,"notAfterNanos":"0"}`,
		`{"runId":"run","revision":"7","sequence":"01","notAfterNanos":"0"}`,
		`{"runId":"run","revision":"7","sequence":"1","sequence":"2","notAfterNanos":"0"}`,
		`{"runId":"run","revision":"7","sequence":"1","notAfterNanos":"0","extra":true}`,
		`{"runId":"run","revision":"7","sequence":"9223372036854775808","notAfterNanos":"0"}`,
	} {
		if _, err := p.renewEgressLease(json.RawMessage(raw)); err == nil {
			t.Fatalf("malformed renewal accepted: %s", raw)
		}
	}
	if p.egress.sequence != 0 || p.egress.notAfter != 0 {
		t.Fatal("malformed renewal changed authority")
	}
}

func TestEgressLeasePermissions(t *testing.T) {
	test := func(name string, edit func(*nodepolicy.Policy, *chatAccountSnapshot)) {
		t.Run(name, func(t *testing.T) {
			p, _, _, snapshot := egressFixture(t)
			pol := p.admission.currentPolicy()
			edit(&pol, &snapshot)
			if err := p.admission.activatePolicy(pol); err != nil {
				t.Fatal(err)
			}
			egressGrant(t, p, 1, 6*time.Second)
			if p.publicPermission("weak", snapshot) == nil {
				t.Fatal("restricted public candidate accepted")
			}
		})
	}
	test("disabled deployment", func(_ *nodepolicy.Policy, s *chatAccountSnapshot) { s.deployment.Enabled = false })
	test("stale snapshot revision", func(_ *nodepolicy.Policy, s *chatAccountSnapshot) { s.revision = "6" })
	test("changed account", func(_ *nodepolicy.Policy, s *chatAccountSnapshot) { s.account.Credential = otherCredential })
	test("changed deployment", func(_ *nodepolicy.Policy, s *chatAccountSnapshot) { s.deployment.Model = "other" })
	test("unknown billing", func(_ *nodepolicy.Policy, s *chatAccountSnapshot) { s.account.Billing = "unknown" })
	test("missing tier", func(p *nodepolicy.Policy, _ *chatAccountSnapshot) { p.Routing.Tiers = nil })
	test("missing API membership", func(p *nodepolicy.Policy, _ *chatAccountSnapshot) { p.Tiers = nil })
	test("remote denied", func(p *nodepolicy.Policy, _ *chatAccountSnapshot) {
		v := p.Routing.Tiers["weak"]
		v.RemoteAllowed = false
		p.Routing.Tiers["weak"] = v
	})
	test("paid denied", func(p *nodepolicy.Policy, _ *chatAccountSnapshot) {
		v := p.Routing.Tiers["weak"]
		v.PaidAllowed = false
		p.Routing.Tiers["weak"] = v
	})
	test("family denied", func(p *nodepolicy.Policy, _ *chatAccountSnapshot) {
		v := p.Routing.Tiers["weak"]
		v.AllowedFamilies = []string{"other"}
		p.Routing.Tiers["weak"] = v
	})
	for _, billing := range []nodepolicy.BillingClass{nodepolicy.BillingFree, nodepolicy.BillingTrial} {
		t.Run(string(billing)+" does not require paid permission", func(t *testing.T) {
			p, _, _, snapshot := egressFixture(t)
			pol := p.admission.currentPolicy()
			pol.Routing.Accounts[0].Billing, snapshot.account.Billing = billing, billing
			pol.Routing.Tiers["weak"] = nodepolicy.TierRouting{RemoteAllowed: true, AllowedFamilies: []string{"alpha"}}
			if err := p.admission.activatePolicy(pol); err != nil {
				t.Fatal(err)
			}
			egressGrant(t, p, 1, 6*time.Second)
			if err := p.publicPermission("weak", snapshot); err != nil {
				t.Fatalf("permitted billing class refused: %v", err)
			}
		})
	}
}

func TestEgressLeaseDisabled(t *testing.T) {
	p, f, _, snapshot := egressFixture(t)
	egressGrant(t, p, 1, 6*time.Second)
	if err := p.publicPermission("weak", snapshot); err != nil {
		t.Fatalf("fixture is not otherwise eligible: %v", err)
	}
	lookups := 0
	poison := func(nodepolicy.CredentialRef) (string, bool) {
		lookups++
		t.Error("POISON RESOLVER: disabled production wrapper disclosed a credential lookup")
		return "synthetic-selected-key", true
	}
	if p.publicAttemptPermitted("weak", snapshot) == nil {
		t.Error("disabled advisory predicate accepted a candidate before body construction")
	}
	_, err := p.runPublicChatAttempt(f.ctx, "weak", snapshot, []byte(f.body), f.profile, f.transport, poison, f.budget.acquire, f.emit)
	var refusal *publicAttemptRefusal
	if !errors.As(err, &refusal) || refusal.Stage != publicBeforeReserve {
		t.Errorf("disabled wrapper refusal = %v, want before-reserve", err)
	}
	if lookups != 0 || len(f.ledger.state.Holds) != 0 || f.calls.Load() != 0 || f.budget.used.Load() != 0 {
		t.Fatalf("disabled effects: resolver=%d holds=%d Do=%d bytes=%d", lookups, len(f.ledger.state.Holds), f.calls.Load(), f.budget.used.Load())
	}
}

func TestEgressLeaseReserveRace(t *testing.T) {
	for _, name := range []string{"revoke", "revision", "expiry"} {
		t.Run(name, func(t *testing.T) {
			p, f, clock, snapshot := egressFixture(t)
			egressGrant(t, p, 1, 6*time.Second)
			entered, resume := make(chan struct{}), make(chan struct{})
			write := f.ledger.write
			f.ledger.write = func(path string, data []byte, perm os.FileMode) error {
				close(entered)
				<-resume
				return write(path, data, perm)
			}
			done := make(chan error, 1)
			go func() {
				_, err := runChatAccountAttempt(f.ctx, snapshot, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire,
					func(context.Context) error { return p.publicPermission("weak", snapshot) }, f.emit)
				done <- err
			}()
			<-entered
			switch name {
			case "revoke":
				egressRevoke(t, p, 2)
			case "revision":
				pol := p.admission.currentPolicy()
				pol.Revision = "8"
				if _, err := p.beginPublicActivation(pol); err != nil {
					t.Fatal(err)
				}
			case "expiry":
				clock.Store(int64(6 * time.Second))
			}
			close(resume)
			if err := <-done; !errors.Is(err, errChatAttempt) {
				t.Fatalf("post-reserve refusal = %v", err)
			}
			egressUnsentHold(t, f)
		})
	}
}

func TestEgressLeaseFinalGuard(t *testing.T) {
	p, f, _, snapshot := egressFixture(t)
	egressGrant(t, p, 1, 6*time.Second)
	checks := 0
	guard := func(context.Context) error {
		checks++
		if checks == 2 {
			egressRevoke(t, p, 2)
		}
		return p.publicPermission("weak", snapshot)
	}
	_, err := runChatAccountAttempt(f.ctx, snapshot, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire, guard, f.emit)
	if !errors.Is(err, errPublicFinalGuard) || checks != 2 {
		t.Fatalf("final refusal = %v after %d checks, want distinct final sentinel and two checks", err, checks)
	}
	egressUnsentHold(t, f)
}

func TestEgressLeaseAcceptedStream(t *testing.T) {
	p, f, _, snapshot := egressFixture(t)
	egressGrant(t, p, 1, 6*time.Second)
	f.body = strings.TrimSuffix(chatSimple, "}") + `,"stream":true}`
	f.response = chatEvent(chatChunk(`{"content":"ok"}`, `"stop"`)) + chatEvent("[DONE]")
	accepted, resume := make(chan struct{}), make(chan struct{})
	transport := f.transport.client.Transport
	f.transport.client.Transport = publicTestRoundTrip(func(req *http.Request) (*http.Response, error) {
		close(accepted)
		<-resume
		return transport.RoundTrip(req)
	})
	done := make(chan error, 1)
	go func() {
		_, err := runChatAccountAttempt(f.ctx, snapshot, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire,
			func(context.Context) error { return p.publicPermission("weak", snapshot) }, f.emit)
		done <- err
	}()
	<-accepted
	egressRevoke(t, p, 2)
	if len(f.ledger.state.Holds) != 1 || f.ledger.state.Holds[0].State != holdReserved || f.ledger.state.Holds[0].Outcome != "" {
		t.Fatal("withdrawal reconciled accepted work before its response")
	}
	close(resume)
	if err := <-done; err != nil {
		t.Fatalf("already accepted response was interrupted: %v", err)
	}
	if f.calls.Load() != 1 || len(f.ledger.state.Holds) != 1 || f.ledger.state.Holds[0].Actual != f.ledger.state.Holds[0].Reserved || !f.ledger.state.Holds[0].evictable() {
		t.Fatal("accepted response did not preserve E's full-bound accounting for missing usage")
	}
}

// In the shipped build this proves before-reserve effects. The same test is
// additionally run in an external disposable source copy with the sole build
// constant changed, so the two retained-hold wrapper stages are exercised too.
func TestEgressLeaseWrapperStages(t *testing.T) {
	if !nodepolicy.PublicDispatchAvailable() {
		p, f, _, snapshot := egressFixture(t)
		egressGrant(t, p, 1, 6*time.Second)
		_, err := p.runPublicChatAttempt(f.ctx, "weak", snapshot, []byte(f.body), f.profile, f.transport, f.resolve, f.budget.acquire, f.emit)
		var refusal *publicAttemptRefusal
		if !errors.As(err, &refusal) || refusal.Stage != publicBeforeReserve || len(f.ledger.state.Holds) != 0 {
			t.Fatalf("before-reserve effects = %v, holds %d", err, len(f.ledger.state.Holds))
		}
		return
	}
	for _, stage := range []publicRefusalStage{publicAfterReserve, publicFinalGuard} {
		t.Run(string(stage), func(t *testing.T) {
			p, f, _, snapshot := egressFixture(t)
			egressGrant(t, p, 1, 6*time.Second)
			write := f.ledger.write
			f.ledger.write = func(path string, data []byte, perm os.FileMode) error {
				err := write(path, data, perm)
				if stage == publicAfterReserve {
					egressRevoke(t, p, 2)
				} else {
					freshChecks := 0
					f.profile.Now = func() time.Time {
						freshChecks++
						if freshChecks == 2 {
							egressRevoke(t, p, 2)
						}
						return f.clock.now()
					}
				}
				return err
			}
			_, err := p.runPublicChatAttempt(f.ctx, "weak", snapshot, []byte(f.body), f.profile, f.transport, f.resolve, f.budget.acquire, f.emit)
			var refusal *publicAttemptRefusal
			if !errors.As(err, &refusal) || refusal.Stage != stage {
				t.Fatalf("wrapper refusal = %v, want stage %s", err, stage)
			}
			egressUnsentHold(t, f)
		})
	}
}

func TestEgressLeasePrivateRPC(t *testing.T) {
	p, w := policyProxy(t)
	rpcOK(t, p, w, nodepolicy.MethodSetPolicy, policyAt("7"))
	ack := rpc(t, p, w, nodepolicy.MethodRenewEgressLease, nodepolicy.RenewEgressLeaseParams{RunID: p.runID, Revision: "7", Sequence: "1", NotAfterNanos: "0"})
	var renewal nodepolicy.RenewEgressLeaseResult
	if ack.Error != nil || json.Unmarshal(ack.Result, &renewal) != nil || renewal.RunID != p.runID || renewal.Revision != "7" || renewal.Sequence != "1" || renewal.NowNanos == "" {
		t.Fatalf("private renewal response = %+v", ack)
	}
	rpcOK(t, p, w, nodepolicy.MethodRevokeEgressLease, nodepolicy.RevokeEgressLeaseParams{RunID: p.runID, Sequence: "2"})
	if p.admission.currentPolicy().Revision != "7" || p.egress.notAfter != 0 {
		t.Fatal("lease RPC changed policy or granted authority")
	}
}

func TestEgressLeaseSchemaControls(t *testing.T) {
	p, _, _, _ := egressFixture(t)
	pol := p.admission.currentPolicy()
	if _, err := nodepolicy.Parse(egressJSON(t, pol)); err == nil {
		t.Fatal("disabled schema accepted enabled public deployment and permissions")
	}
	for i := range pol.Routing.Deployments {
		pol.Routing.Deployments[i].Enabled = false
	}
	pol.Routing.Tiers["weak"] = nodepolicy.TierRouting{BareMode: nodepolicy.BareHybrid}
	if _, err := nodepolicy.Parse(egressJSON(t, pol)); err != nil {
		t.Fatalf("disabled staged schema refused: %v", err)
	}
}

func TestPolicyActivationEgressTransition(t *testing.T) {
	p, _, _, snapshot := egressFixture(t)
	egressGrant(t, p, 1, 6*time.Second)
	policy := p.admission.currentPolicy()
	older := policy
	older.Revision = "6"
	if _, err := p.beginPublicActivation(older); err == nil || p.publicPermission("weak", snapshot) != nil {
		t.Fatal("older rejected policy changed active public authority")
	}
	token, err := p.beginPublicActivation(policy)
	if err != nil {
		t.Fatal(err)
	}
	if p.publicPermission("weak", snapshot) == nil {
		t.Fatal("activation transition accepted permission")
	}
	if _, err := p.renewEgressLease(egressJSON(t, nodepolicy.RenewEgressLeaseParams{RunID: p.runID, Revision: "7", Sequence: "2", NotAfterNanos: "6000000000"})); err == nil {
		t.Fatal("activation transition accepted a renewal")
	}
	p.completePublicActivation(token, nil)
	if p.publicPermission("weak", snapshot) == nil || p.egress.sequence != 1 {
		t.Fatal("same-revision replay renewed authority or reset high-water")
	}
	egressGrant(t, p, 2, 6*time.Second)
	policy.Revision = "8"
	obsolete, err := p.beginPublicActivation(policy)
	if err != nil {
		t.Fatal(err)
	}
	policy.Revision = "9"
	current, err := p.beginPublicActivation(policy)
	if err != nil {
		t.Fatal(err)
	}
	p.completePublicActivation(obsolete, nil)
	if p.egress.ready || p.egress.transition != current {
		t.Fatal("obsolete completion reopened a later transition")
	}
	p.completePublicActivation(current, errLedgerUnavailable)
	if p.egress.ready || p.egress.transition != nil || p.admission.currentPolicy().Revision != "9" {
		t.Fatal("ledger failure reopened public admission or rolled back local policy")
	}
}

func TestPolicyActivationEgressLedgerFailure(t *testing.T) {
	p, w := policyProxy(t)
	pol := policyAt("7")
	pol.Routing = testRouting("1000000", "10000000", "10000000")
	// Disabled staged configuration parses, but there is no ledger directory.
	pol.Routing.Tiers = nodepolicy.Default().Routing.Tiers
	pol.Routing.Limits = nodepolicy.Default().Routing.Limits
	pol.Routing.Listener = nodepolicy.Default().Routing.Listener
	ack := ackOf(t, rpc(t, p, w, nodepolicy.MethodSetPolicy, pol))
	if ack.Revision != "7" || p.admission.currentPolicy().Revision != "7" || p.egress.ready {
		t.Fatal("account activation failure blocked local policy or enabled public lease receipt")
	}
	rpcErr(t, p, w, nodepolicy.MethodRenewEgressLease, nodepolicy.RenewEgressLeaseParams{RunID: p.runID, Revision: "7", Sequence: "1", NotAfterNanos: "0"})
}

func TestPolicyActivationEgressSerialization(t *testing.T) {
	p, f, _, snapshot := egressFixture(t)
	egressGrant(t, p, 1, 6*time.Second)
	// An acceptance already inside the lease critical section completes first.
	// Transition entry then clears authority before changing admission.
	entered, resume := make(chan struct{}), make(chan struct{})
	var first atomic.Bool
	now := p.egress.now
	p.egress.now = func() time.Duration {
		if first.CompareAndSwap(false, true) {
			close(entered)
			<-resume
		}
		return now()
	}
	accepted := make(chan error, 1)
	go func() { accepted <- p.publicPermission("weak", snapshot) }()
	<-entered
	transition := make(chan error, 1)
	go func() {
		pol := policyAt("8")
		_, err := p.beginPublicActivation(pol)
		transition <- err
	}()
	close(resume)
	if err := <-accepted; err != nil {
		t.Fatalf("accepted work lost to later transition: %v", err)
	}
	if err := <-transition; err != nil {
		t.Fatalf("policy transition failed: %v", err)
	}
	if p.publicPermission("weak", snapshot) == nil || len(f.ledger.state.Holds) != 0 {
		t.Fatal("transition failed to withdraw new acceptance")
	}
}

func TestPublicTransportFinalGuard(t *testing.T) {
	for _, name := range []string{"nil guard", "headroom versus revoke", "construction completed before guard"} {
		t.Run(name, func(t *testing.T) {
			p, _, _, snapshot := egressFixture(t)
			egressGrant(t, p, 1, 6*time.Second)
			var calls atomic.Int64
			c, err := newPublicClient("https://example.com/chat", publicTestBounds())
			if err != nil {
				t.Fatal(err)
			}
			c.client.Transport = publicTestRoundTrip(func(*http.Request) (*http.Response, error) {
				calls.Add(1)
				return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("{}"))}, nil
			})
			budget := &publicTestBudget{ceiling: 4096}
			acquire := budget.acquire
			var guard func(context.Context) error
			if name != "nil guard" {
				guard = func(context.Context) error { return p.publicPermission("weak", snapshot) }
			}
			if name == "construction completed before guard" {
				guard = func(context.Context) error { egressRevoke(t, p, 2); return p.publicPermission("weak", snapshot) }
			}
			if name == "headroom versus revoke" {
				entered, resume := make(chan struct{}), make(chan struct{})
				acquire = func(n int64) (func(), error) { close(entered); <-resume; return budget.acquire(n) }
				go func() { <-entered; egressRevoke(t, p, 2); close(resume) }()
			}
			response, err := c.post(context.Background(), strings.NewReader("{}"), "synthetic", false, acquire, guard)
			if !errors.Is(err, errPublicFinalGuard) || response != nil || calls.Load() != 0 || budget.used.Load() != 0 {
				t.Fatalf("final guard effects: response=%v err=%v Do=%d bytes=%d", response, err, calls.Load(), budget.used.Load())
			}
		})
	}
}

func TestEgressLeaseRefusedRevokeKeepsAuthority(t *testing.T) {
	p, _, _, snapshot := egressFixture(t)
	egressGrant(t, p, 2, 6*time.Second)
	for _, in := range []nodepolicy.RevokeEgressLeaseParams{{RunID: "old-run", Sequence: "3"}, {RunID: p.runID, Sequence: "2"}} {
		if _, err := p.revokeEgressLease(egressJSON(t, in)); err == nil || p.publicPermission("weak", snapshot) != nil {
			t.Fatal("invalid withdrawal changed authority")
		}
	}
	if !reflect.DeepEqual(p.admission.currentPolicy().Routing.Accounts[0], snapshot.account) {
		t.Fatal("lease withdrawal changed policy")
	}
}
