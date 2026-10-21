// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"math"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func brokerLeaseReply(method string, raw json.RawMessage) (any, string) {
	switch method {
	case nodepolicy.MethodRenewEgressLease:
		var in nodepolicy.RenewEgressLeaseParams
		if err := json.Unmarshal(raw, &in); err != nil {
			return nil, err.Error()
		}
		return nodepolicy.RenewEgressLeaseResult{RunID: in.RunID, Revision: in.Revision, Sequence: in.Sequence, NowNanos: "1000000000"}, ""
	case nodepolicy.MethodRevokeEgressLease:
		var in nodepolicy.RevokeEgressLeaseParams
		if err := json.Unmarshal(raw, &in); err != nil {
			return nil, err.Error()
		}
		return nodepolicy.RevokeEgressLeaseResult{RunID: in.RunID, Sequence: in.Sequence}, ""
	}
	return nil, "unexpected lease method"
}

func enableBrokerLeaseReplies(t *testing.T, ab *activatedBroker) {
	t.Helper()
	ab.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodRenewEgressLease || method == nodepolicy.MethodRevokeEgressLease {
			return brokerLeaseReply(method, raw)
		}
		return ab.acks.answer(method, raw)
	})
}

func brokerLeaseRenewals(t *testing.T, f *fakeWorker) []nodepolicy.RenewEgressLeaseParams {
	t.Helper()
	var out []nodepolicy.RenewEgressLeaseParams
	for _, call := range f.snapshot() {
		if call.method != nodepolicy.MethodRenewEgressLease {
			continue
		}
		if call.notify {
			t.Fatal("lease renewal was an unbounded notification")
		}
		var in nodepolicy.RenewEgressLeaseParams
		if err := json.Unmarshal(call.params, &in); err != nil {
			t.Fatalf("decode renewal: %v", err)
		}
		out = append(out, in)
	}
	return out
}

func newBrokerLeaseFixture(t *testing.T) (*activatedBroker, *atomic.Int64) {
	t.Helper()
	ab := newActivatedBroker(t)
	enableBrokerLeaseReplies(t, ab)
	clock := new(atomic.Int64)
	base := time.Now()
	r := ab.b.nodePolicyRuntime()
	r.mu.Lock()
	r.egress.now = func() time.Time { return base.Add(time.Duration(clock.Load())) }
	r.mu.Unlock()
	return ab, clock
}

func TestEgressLeaseBrokerStartupHasNoGrant(t *testing.T) {
	b, _ := newPolicyBroker(t)
	_, proxy, _ := attachAckingProxy(t, b)
	b.renewEgressLease(context.Background())
	if calls := proxy.snapshot(); len(calls) != 0 {
		t.Fatalf("unactivated startup wrote methods: %v", proxy.methods(""))
	}
	ab, _ := newBrokerLeaseFixture(t)
	if got := brokerLeaseRenewals(t, ab.proxy); len(got) != 0 {
		t.Fatalf("policy replay granted startup authority: %+v", got)
	}
}

func TestEgressLeaseBrokerEligibilityLossWritesNoRenewal(t *testing.T) {
	tests := []struct {
		name string
		lose func(*policyRuntime)
	}{
		{"persisted revision differs", func(r *policyRuntime) { r.current.Revision = "1" }},
		{"old policy acknowledgement", func(r *policyRuntime) { r.activeRevision = "" }},
		{"policy activation failed", func(r *policyRuntime) { r.activationErr = "policy refused" }},
		{"availability activation failed", func(r *policyRuntime) { r.availabilityErr = "availability refused" }},
		{"policy push pending", func(r *policyRuntime) { r.egress.policyPending = true }},
		{"availability push pending", func(r *policyRuntime) { r.egress.availabilityPending = true }},
		{"recovery required", func(r *policyRuntime) { r.recoveryRequired = true }},
		{"incarnation retired", func(r *policyRuntime) { r.incarnation = nil }},
		{"withdrawn", func(r *policyRuntime) { r.egress.withdrawn = true }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ab, _ := newBrokerLeaseFixture(t)
			ab.b.renewEgressLease(context.Background())
			r := ab.b.nodePolicyRuntime()
			r.mu.Lock()
			test.lose(r)
			r.mu.Unlock()
			ab.b.renewEgressLease(context.Background())
			if got := brokerLeaseRenewals(t, ab.proxy); len(got) != 1 {
				t.Fatalf("eligibility loss wrote renewal: %+v", got)
			}
		})
	}
}

func TestEgressLeaseBrokerPushSlotRechecksEligibility(t *testing.T) {
	ab, _ := newBrokerLeaseFixture(t)
	r := ab.b.nodePolicyRuntime()
	if !r.acquirePush(context.Background()) {
		t.Fatal("acquire push slot")
	}
	done := make(chan struct{})
	go func() { ab.b.renewEgressLease(context.Background()); close(done) }()
	// Loss is published before releasing the slot, so even a tick that has
	// already selected the old revision cannot write it afterward.
	r.mu.Lock()
	r.current.Revision = "1"
	r.egress.policyPending = true
	r.mu.Unlock()
	r.releasePush()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("renewal did not leave push slot")
	}
	if got := brokerLeaseRenewals(t, ab.proxy); len(got) != 0 {
		t.Fatalf("queued tick wrote stale renewal: %+v", got)
	}
}

func TestEgressLeaseBrokerAnchoredBoundSurvivesDelayAndExpiry(t *testing.T) {
	ab, clock := newBrokerLeaseFixture(t)
	// Delay the anchor reply and return much newer samples on later ACKs.
	// Only the anchor's sample may set the run-lifetime clock origin.
	var responses atomic.Int64
	ab.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, errText := brokerLeaseReply(method, raw)
		if method == nodepolicy.MethodRenewEgressLease {
			if responses.Add(1) == 1 {
				clock.Store(int64(2 * time.Second))
			} else {
				var in nodepolicy.RenewEgressLeaseParams
				if err := json.Unmarshal(raw, &in); err != nil {
					return nil, err.Error()
				}
				result = nodepolicy.RenewEgressLeaseResult{RunID: in.RunID, Revision: in.Revision, Sequence: in.Sequence, NowNanos: "90000000000"}
			}
		}
		return result, errText
	})
	ab.b.renewEgressLease(context.Background())
	for _, issuance := range []time.Duration{3 * time.Second, 12 * time.Second, 20 * time.Second} {
		clock.Store(int64(issuance))
		ab.b.renewEgressLease(context.Background())
	}
	got := brokerLeaseRenewals(t, ab.proxy)
	if len(got) != 4 || got[0].NotAfterNanos != "0" {
		t.Fatalf("anchor/grants = %+v", got)
	}
	for i, issuance := range []time.Duration{3 * time.Second, 12 * time.Second, 20 * time.Second} {
		deadline, err := strconv.ParseInt(got[i+1].NotAfterNanos, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		want := int64(time.Second + issuance - 2*time.Second + nodepolicy.MaxEgressLeaseDuration)
		if deadline != want {
			t.Fatalf("grant at %s deadline=%d want conservative anchor bound %d", issuance, deadline, want)
		}
		if got[i+1].Sequence != strconv.Itoa(i+2) {
			t.Fatalf("sequence: %+v", got)
		}
	}
}

func TestEgressLeaseBrokerArithmeticExhaustionWritesNoGrant(t *testing.T) {
	tests := []struct {
		name    string
		exhaust func(*brokerEgressLease, *atomic.Int64)
	}{
		{"sequence exhausted", func(l *brokerEgressLease, _ *atomic.Int64) { l.sequence = math.MaxInt64 }},
		{"anchor plus duration overflows", func(l *brokerEgressLease, _ *atomic.Int64) { l.anchorNanos = math.MaxInt64 }},
		{"elapsed plus duration overflows", func(l *brokerEgressLease, clock *atomic.Int64) { l.anchorNanos = 1; clock.Store(math.MaxInt64) }},
		{"clock moved backwards", func(_ *brokerEgressLease, clock *atomic.Int64) { clock.Store(-1) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ab, clock := newBrokerLeaseFixture(t)
			ab.b.renewEgressLease(context.Background())
			r := ab.b.nodePolicyRuntime()
			r.mu.Lock()
			test.exhaust(&r.egress, clock)
			r.mu.Unlock()
			ab.b.renewEgressLease(context.Background())
			if got := brokerLeaseRenewals(t, ab.proxy); len(got) != 1 {
				t.Fatalf("exhaustion wrote grant: %+v", got)
			}
		})
	}
}

func TestEgressLeaseBrokerTransientTimeoutResumesNextTick(t *testing.T) {
	ab, clock := newBrokerLeaseFixture(t)
	ab.b.renewEgressLease(context.Background())
	beforePolicy := len(ab.proxy.methods(nodepolicy.MethodSetPolicy))
	ab.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, _ := brokerLeaseReply(method, raw)
		return result, holdSuccessReply
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { ab.b.renewEgressLease(ctx); close(done) }()
	ab.proxy.waitFor("blocked renewal", func(calls []policyCall) bool { return len(brokerLeaseRenewals(t, ab.proxy)) == 2 })
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("cancel did not join blocked renewal")
	}
	// Its successful late reply must not reset the anchor or activate policy.
	ab.proxy.releaseHeld()
	enableBrokerLeaseReplies(t, ab)
	clock.Store(int64(7 * time.Second))
	ab.b.renewEgressLease(context.Background())
	got := brokerLeaseRenewals(t, ab.proxy)
	if len(got) != 3 || got[2].NotAfterNanos != "13000000000" {
		t.Fatalf("next tick failed to use original anchor: %+v", got)
	}
	if len(ab.proxy.methods(nodepolicy.MethodSetPolicy)) != beforePolicy {
		t.Fatal("renewal required policy activation after transport cancellation")
	}
	if state := ab.b.localPolicy(); state.Activation != activationActive {
		t.Fatalf("transient failure changed activation: %+v", state)
	}
}

func TestEgressLeaseBrokerRPCRefusalResumesWithoutPolicyPush(t *testing.T) {
	ab, clock := newBrokerLeaseFixture(t)
	ab.b.renewEgressLease(context.Background())
	beforePolicy := len(ab.proxy.methods(nodepolicy.MethodSetPolicy))
	ab.proxy.setAnswer(func(string, json.RawMessage) (any, string) { return nil, "temporary renewal refusal" })
	clock.Store(int64(time.Second))
	ab.b.renewEgressLease(context.Background())
	enableBrokerLeaseReplies(t, ab)
	clock.Store(int64(2 * time.Second))
	ab.b.renewEgressLease(context.Background())
	got := brokerLeaseRenewals(t, ab.proxy)
	if len(got) != 3 || got[1].NotAfterNanos != "7000000000" || got[2].NotAfterNanos != "8000000000" {
		t.Fatalf("RPC refusal lost run anchor or next tick renewal: %+v", got)
	}
	if len(ab.proxy.methods(nodepolicy.MethodSetPolicy)) != beforePolicy {
		t.Fatal("RPC refusal required policy activation")
	}
}

func TestEgressLeaseBrokerRenewalLoopJoinsBlockedHandler(t *testing.T) {
	ab, _ := newBrokerLeaseFixture(t)
	entered, release := make(chan struct{}), make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	ab.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method != nodepolicy.MethodRenewEgressLease {
			return map[string]bool{"ok": true}, ""
		}
		close(entered)
		<-release // Unlike holdSuccessReply, the codec cannot read another request.
		return brokerLeaseReply(method, raw)
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ticks := make(chan time.Time)
	done := make(chan struct{})
	go func() { ab.b.runEgressRenewal(ctx, ticks); close(done) }()
	ticks <- time.Now()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("renewal did not enter synchronous handler")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("renewal loop did not join after cancellation")
	}
	close(release)
	// The next request is read only after the late renewal reply was written.
	checkCtx, checkCancel := context.WithTimeout(context.Background(), time.Second)
	defer checkCancel()
	if _, rpcErr, err := ab.pp.Call(checkCtx, nodepolicy.MethodSetResidency, json.RawMessage(`{}`)); err != nil || rpcErr != nil {
		t.Fatalf("drain late response: %v %v", err, rpcErr)
	}
	r := ab.b.nodePolicyRuntime()
	r.mu.Lock()
	anchored := r.egress.anchored
	r.mu.Unlock()
	if anchored {
		t.Fatal("cancelled handler completion anchored the run")
	}
	if len(brokerLeaseRenewals(t, ab.proxy)) != 1 {
		t.Fatal("cancelled loop issued another renewal")
	}
}

func TestEgressLeaseBrokerDeadlineResumesNextTick(t *testing.T) {
	ab, clock := newBrokerLeaseFixture(t)
	ab.b.renewEgressLease(context.Background())
	beforePolicy := len(ab.proxy.methods(nodepolicy.MethodSetPolicy))
	ab.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, _ := brokerLeaseReply(method, raw)
		return result, holdSuccessReply
	})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	ab.b.renewEgressLease(ctx)
	if ctx.Err() != context.DeadlineExceeded {
		t.Fatalf("renewal did not exhaust deadline: %v", ctx.Err())
	}
	enableBrokerLeaseReplies(t, ab)
	clock.Store(int64(9 * time.Second))
	ab.b.renewEgressLease(context.Background())
	got := brokerLeaseRenewals(t, ab.proxy)
	if len(got) != 3 || got[2].NotAfterNanos != "15000000000" {
		t.Fatalf("deadline lost anchor or next tick: %+v", got)
	}
	if len(ab.proxy.methods(nodepolicy.MethodSetPolicy)) != beforePolicy {
		t.Fatal("deadline required policy:set")
	}
	ab.proxy.releaseHeld()
}

func TestEgressLeaseBrokerLateOldPipeReplyCannotAnchorReplacement(t *testing.T) {
	ab, _ := newBrokerLeaseFixture(t)
	entered, release := make(chan struct{}), make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	ab.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		close(entered)
		<-release
		return brokerLeaseReply(method, raw)
	})
	done := make(chan struct{})
	go func() { ab.b.renewEgressLease(context.Background()); close(done) }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("old pipe did not receive anchor request")
	}
	pp, proxy, acks := attachAckingProxy(t, ab.b)
	close(release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("late old reply did not complete")
	}
	r := ab.b.nodePolicyRuntime()
	r.mu.Lock()
	anchored := r.egress.anchored
	r.mu.Unlock()
	if anchored {
		t.Fatal("old pipe reply anchored after replacement was published")
	}
	if !ab.b.replayPolicyToProxy(context.Background(), pp) {
		t.Fatal("replacement replay")
	}
	enableBrokerLeaseReplies(t, &activatedBroker{b: ab.b, pp: pp, proxy: proxy, acks: acks})
	ab.b.renewEgressLease(context.Background())
	got := brokerLeaseRenewals(t, proxy)
	if len(got) != 1 || got[0].NotAfterNanos != "0" || got[0].RunID != acks.runID {
		t.Fatalf("replacement used old clock anchor: %+v", got)
	}
}

func TestEgressLeaseBrokerFailedPushStopsRenewal(t *testing.T) {
	for _, kind := range []string{pushPolicy, pushAvailability} {
		t.Run(kind, func(t *testing.T) {
			ab, _ := newBrokerLeaseFixture(t)
			ab.b.renewEgressLease(context.Background())
			ab.proxy.setAnswer(func(string, json.RawMessage) (any, string) { return nil, "push refused" })
			r := ab.b.nodePolicyRuntime()
			if !r.acquirePush(context.Background()) {
				t.Fatal("push slot")
			}
			ab.b.sendPolicyStateLocked(context.Background(), ab.pp, kind)
			r.releasePush()
			enableBrokerLeaseReplies(t, ab)
			ab.b.renewEgressLease(context.Background())
			if got := brokerLeaseRenewals(t, ab.proxy); len(got) != 1 {
				t.Fatalf("failed state push renewed: %+v", got)
			}
			if ab.b.policyProxy() != ab.pp {
				t.Fatal("ordinary push failure restarted proxy")
			}
		})
	}
}

func TestEgressLeaseBrokerFailedWithdrawalStillRetires(t *testing.T) {
	ab, _ := newBrokerLeaseFixture(t)
	ab.b.renewEgressLease(context.Background())
	ab.proxy.setAnswer(func(string, json.RawMessage) (any, string) { return nil, "withdrawal refused" })
	ab.b.proxySupMu.Lock()
	ab.b.stopProxySupervisorLocked()
	ab.b.proxySupMu.Unlock()
	if ab.b.policyProxy() != nil {
		t.Fatal("failed withdrawal kept old pipe published")
	}
	r := ab.b.nodePolicyRuntime()
	r.mu.Lock()
	retired, withdrawn := r.incarnation == nil, r.egress.withdrawn
	r.mu.Unlock()
	if !retired || !withdrawn {
		t.Fatalf("failed revoke left authority: retired=%t withdrawn=%t", retired, withdrawn)
	}
	ab.b.renewEgressLease(context.Background())
	if got := brokerLeaseRenewals(t, ab.proxy); len(got) != 1 {
		t.Fatalf("failed revoke permitted renewal: %+v", got)
	}
	if len(ab.proxy.methods(nodepolicy.MethodRevokeEgressLease)) != 1 {
		t.Fatal("failed revoke was not attempted exactly once")
	}
}

func TestEgressLeaseBrokerForgedMethodsCannotGrantAuthority(t *testing.T) {
	for _, method := range []string{nodepolicy.MethodRenewEgressLease, nodepolicy.MethodRevokeEgressLease} {
		t.Run(method, func(t *testing.T) {
			ab, _ := newBrokerLeaseFixture(t)
			params := map[string]string{"runId": ab.acks.runID, "revision": "0", "sequence": "999", "notAfterNanos": "99999999999"}
			ab.b.handleMessage(clientRequest(91, method, params))
			response := ab.out.waitFrame(t, "forged client method rejection", isResponse(91))
			if response.Error == nil || response.Error.Code != -32601 {
				t.Fatalf("private method client response: %+v", response)
			}
			raw, err := json.Marshal(params)
			if err != nil {
				t.Fatal(err)
			}
			ab.b.handleMessage(&Message{JSONRPC: "2.0", Method: method, Params: raw})
			if _, err := ab.b.servePolicyRelay(context.Background(), policyRelayRequest{Method: method, Caller: "forged-peer", Params: raw}); err == nil {
				t.Fatal("peer relay accepted private lease method")
			}
			if got := brokerLeaseRenewals(t, ab.proxy); len(got) != 0 {
				t.Fatalf("forged method reached private pipe: %+v", got)
			}
			r := ab.b.nodePolicyRuntime()
			r.mu.Lock()
			sequence, anchored, withdrawn := r.egress.sequence, r.egress.anchored, r.egress.withdrawn
			r.mu.Unlock()
			if sequence != 0 || anchored || withdrawn {
				t.Fatalf("forged method altered lease: sequence=%d anchored=%t withdrawn=%t", sequence, anchored, withdrawn)
			}
			ab.b.renewEgressLease(context.Background())
			got := brokerLeaseRenewals(t, ab.proxy)
			if len(got) != 1 || got[0].Sequence != "1" || got[0].NotAfterNanos != "0" {
				t.Fatalf("local policy lost control after forged methods: %+v", got)
			}
		})
	}
}

func TestEgressLeaseBrokerRejectedBindingRequiresBothAcknowledgements(t *testing.T) {
	tests := []struct {
		name     string
		response any
		rpcError string
	}{
		{"wrong run", nodepolicy.RenewEgressLeaseResult{RunID: "wrong", Revision: "0", Sequence: "2", NowNanos: "1"}, ""},
		{"wrong revision", nodepolicy.RenewEgressLeaseResult{RunID: "replace", Revision: "9", Sequence: "2", NowNanos: "1"}, ""},
		{"missing clock sample", map[string]string{"runId": "replace", "revision": "0", "sequence": "2"}, ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ab, _ := newBrokerLeaseFixture(t)
			ab.b.renewEgressLease(context.Background())
			ab.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
				if method != nodepolicy.MethodRenewEgressLease {
					return ab.acks.answer(method, raw)
				}
				if ack, ok := test.response.(nodepolicy.RenewEgressLeaseResult); ok {
					if ack.RunID == "replace" {
						ack.RunID = ab.acks.runID
					}
					return ack, test.rpcError
				}
				return test.response, test.rpcError
			})
			ab.b.renewEgressLease(context.Background())
			enableBrokerLeaseReplies(t, ab)
			ab.b.renewEgressLease(context.Background())
			if len(brokerLeaseRenewals(t, ab.proxy)) != 2 {
				t.Fatal("rejected binding renewed without fresh activation")
			}
			r := ab.b.nodePolicyRuntime()
			// Record just the policy ACK: sendPolicyStateLocked also retries
			// failed availability, which would supply both required ACKs.
			ack, err := json.Marshal(nodepolicy.PolicyAck{OriginNodeID: ab.b.nodeID, RunID: ab.acks.runID, Revision: "0"})
			if err != nil {
				t.Fatal(err)
			}
			ab.b.recordActivation(ab.pp, "0", ack, nil, nil)
			ab.b.renewEgressLease(context.Background())
			if len(brokerLeaseRenewals(t, ab.proxy)) != 2 {
				t.Fatal("policy acknowledgement bypassed missing availability acknowledgement")
			}
			if !r.acquirePush(context.Background()) {
				t.Fatal("push slot")
			}
			ab.b.sendPolicyStateLocked(context.Background(), ab.pp, pushAvailability)
			r.releasePush()
			ab.b.renewEgressLease(context.Background())
			got := brokerLeaseRenewals(t, ab.proxy)
			if len(got) != 3 || got[2].NotAfterNanos != "0" || got[2].Sequence != "3" {
				t.Fatalf("fresh activation did not reanchor with increasing sequence: %+v", got)
			}
		})
	}
}

func TestEgressLeaseBrokerReplacementStartsWithNewAnchor(t *testing.T) {
	ab, _ := newBrokerLeaseFixture(t)
	ab.b.renewEgressLease(context.Background())
	ab.b.renewEgressLease(context.Background())
	pp, proxy, acks := attachAckingProxy(t, ab.b)
	ab.b.renewEgressLease(context.Background())
	if len(brokerLeaseRenewals(t, proxy)) != 0 {
		t.Fatal("new pipe received old run renewal before replay")
	}
	if len(brokerLeaseRenewals(t, ab.proxy)) != 2 {
		t.Fatal("old pipe renewed after replacement")
	}
	if !ab.b.replayPolicyToProxy(context.Background(), pp) {
		t.Fatal("replay replacement")
	}
	replacement := &activatedBroker{b: ab.b, pp: pp, proxy: proxy, acks: acks}
	enableBrokerLeaseReplies(t, replacement)
	ab.b.renewEgressLease(context.Background())
	got := brokerLeaseRenewals(t, proxy)
	if len(got) != 1 || got[0].RunID == ab.acks.runID || got[0].NotAfterNanos != "0" || got[0].Sequence != "1" {
		t.Fatalf("replacement inherited lease: %+v", got)
	}
}

func TestEgressLeaseBrokerWithdrawsBeforeRetirement(t *testing.T) {
	ab, _ := newBrokerLeaseFixture(t)
	ab.b.renewEgressLease(context.Background())
	ab.b.renewEgressLease(context.Background())
	observed := make(chan bool, 1)
	ab.proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodRevokeEgressLease {
			r := ab.b.nodePolicyRuntime()
			r.mu.Lock()
			observed <- r.incarnation == ab.pp && r.egress.withdrawn
			r.mu.Unlock()
		}
		return brokerLeaseReply(method, raw)
	})
	ab.b.proxySupMu.Lock()
	ab.b.stopProxySupervisorLocked()
	ab.b.proxySupMu.Unlock()
	select {
	case beforeRetirement := <-observed:
		if !beforeRetirement {
			t.Fatal("retired before disabling renewal and sending revoke")
		}
	default:
		t.Fatal("missing withdrawal")
	}
	var revoke nodepolicy.RevokeEgressLeaseParams
	for _, call := range ab.proxy.snapshot() {
		if call.method == nodepolicy.MethodRevokeEgressLease {
			if err := json.Unmarshal(call.params, &revoke); err != nil {
				t.Fatal(err)
			}
		}
	}
	if revoke.RunID != ab.acks.runID || revoke.Sequence != "3" {
		t.Fatalf("withdrawal binding: %+v", revoke)
	}
	ab.b.renewEgressLease(context.Background())
	if len(brokerLeaseRenewals(t, ab.proxy)) != 2 || ab.b.policyProxy() != nil {
		t.Fatal("retired pipe retained renewal authority")
	}
}
