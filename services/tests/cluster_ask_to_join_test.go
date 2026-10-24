// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

type askProcess struct {
	p    *cmProc
	dir  string
	port int
	id   cmNodeID
}

type askProcessRequest struct {
	RequestID string  `json:"requestId"`
	State     string  `json:"state"`
	Reason    string  `json:"reason"`
	InviteID  *string `json:"inviteId"`
}

func newAskProcess(t *testing.T, env []string) *askProcess {
	t.Helper()
	f := &askProcess{dir: shortClusterTempDir(t), port: temporaryClusterPort(t)}
	f.p = startCMEnv(t, f.dir, f.port, env)
	f.id = decodeResult[cmNodeID](t, f.p.call("cluster:get-node-id", nil))
	t.Cleanup(func() { f.stop(t) })
	return f
}

func (f *askProcess) stop(t *testing.T) {
	t.Helper()
	if f.p == nil {
		return
	}
	p := f.p
	f.p = nil
	if err := p.stdin.Close(); err != nil {
		t.Errorf("close owned manager stdin: %v", err)
	}
	done := make(chan error, 1)
	go func() { done <- p.cmd.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("owned manager exit: %v", err)
		}
	case <-time.After(5 * time.Second):
		if err := p.cmd.Process.Kill(); err != nil {
			t.Errorf("kill owned manager: %v", err)
		}
		<-done
		t.Error("owned manager required forced termination")
	}
}

func (f *askProcess) create(t *testing.T) {
	t.Helper()
	f.p.call("cluster:create", map[string]any{"clusterFriendlyName": "private admission process fixture"})
	f.id = decodeResult[cmNodeID](t, f.p.call("cluster:get-node-id", nil))
	if f.id.ClusterID == "" {
		t.Fatal("create returned empty cluster identity")
	}
}

func askProcessPair(t *testing.T, a, b *askProcess) {
	t.Helper()
	inv := decodeResult[outboundInviteResult](t, a.p.call("cluster:invite-node", map[string]any{"address": "127.0.0.1", "port": b.port, "allowRemoteManagement": false}))
	if inv.State != "pending" || len(inv.Pin) != 6 {
		t.Fatal("normal fixture invite did not provide pending local PIN")
	}
	b.p.waitNotify("cluster:invite-received")
	result := decodeResult[outboundInviteResult](t, b.p.call("cluster:respond-to-invite", map[string]any{"inviteId": inv.InviteID, "accept": true, "pin": inv.Pin, "allowRemoteManagement": false}))
	if result.State != "paired" {
		t.Fatalf("normal fixture pairing state = %q", result.State)
	}
	waitInviteState(t, a.p, inv.InviteID, "paired")
	b.id = decodeResult[cmNodeID](t, b.p.call("cluster:get-node-id", nil))
	waitMembers(t, a.p, 2)
	waitMembers(t, b.p, 2)
}

func askProcessStart(t *testing.T, requester, member *askProcess, replace bool) askProcessRequest {
	t.Helper()
	r := decodeResult[askProcessRequest](t, requester.p.call("cluster:ask-to-join", map[string]any{"address": "127.0.0.1", "port": member.port, "replaceSoloCluster": replace}))
	if r.RequestID == "" || r.State != "awaiting-approval" || r.InviteID != nil {
		t.Fatal("ask did not start without an invite")
	}
	askProcessWait(t, member, r.RequestID, "awaiting-approval")
	return r
}

func askProcessWait(t *testing.T, f *askProcess, id, state string) askProcessRequest {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	last := "missing"
	for time.Now().Before(deadline) {
		snapshot := decodeResult[struct {
			Supported bool                `json:"supported"`
			Requests  []askProcessRequest `json:"requests"`
		}](t, f.p.call("cluster:get-join-requests", map[string]any{}))
		if !snapshot.Supported {
			t.Fatal("join requests unsupported by fixture binary")
		}
		for _, r := range snapshot.Requests {
			if r.RequestID == id {
				last = r.State
				if r.State == state {
					return r
				}
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("request state = %q, want %q", last, state)
	return askProcessRequest{}
}

func askProcessApprove(t *testing.T, f *askProcess, id string) askProcessRequest {
	t.Helper()
	r := decodeResult[askProcessRequest](t, f.p.call("cluster:respond-to-join-request", map[string]any{"requestId": id, "accept": true, "allowRemoteManagement": false}))
	if r.State != "awaiting-requester" || r.InviteID == nil {
		t.Fatal("approval did not allocate correlated invite")
	}
	return r
}

func askProcessFinish(t *testing.T, requester, member *askProcess, r askProcessRequest, wrong bool) {
	t.Helper()
	requester.p.waitNotify("cluster:invite-received")
	// The requester notifies before returning the last Initial Exchange reply.
	// The member derives and publishes its PIN only after receiving that reply.
	var inv outboundInviteResult
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		inv = decodeResult[outboundInviteResult](t, member.p.call("cluster:invite-status", map[string]any{"inviteId": *r.InviteID}))
		if inv.State == "pending" && len(inv.Pin) == 6 {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if inv.State != "pending" || len(inv.Pin) != 6 {
		t.Fatal("approver local API did not display PIN")
	}
	local := decodeResult[outboundInviteResult](t, requester.p.call("cluster:invite-status", map[string]any{"inviteId": *r.InviteID}))
	if local.Pin != "" {
		t.Fatal("requester exposed a display PIN")
	}
	pin, want := inv.Pin, "paired"
	if wrong {
		pin, want = "000000", "failed"
		if pin == inv.Pin {
			pin = "000001"
		}
	}
	result := decodeResult[outboundInviteResult](t, requester.p.call("cluster:respond-to-invite", map[string]any{"inviteId": *r.InviteID, "accept": true, "pin": pin, "allowRemoteManagement": false}))
	if result.State != want {
		t.Fatalf("PIN completion state = %q, want %q", result.State, want)
	}
	if wrong && result.Reason != "incorrect-pin" {
		t.Fatalf("wrong PIN reason = %q", result.Reason)
	}
	waitInviteState(t, member.p, *r.InviteID, want)
	askProcessWait(t, requester, r.RequestID, want)
	askProcessWait(t, member, r.RequestID, want)
}

type askProcessAuthority struct {
	ClusterID       string `json:"clusterId"`
	Epoch           uint64 `json:"epoch"`
	NodeUUID        string `json:"nodeUuid"`
	AdmissionEpoch  uint64 `json:"admissionEpoch"`
	CertFingerprint string `json:"certFingerprint"`
}

func askProcessNoPeer(t *testing.T, f, peer *askProcess, wantCount int) {
	t.Helper()
	if got := memberCount(t, f.p); got != wantCount {
		t.Fatalf("unapproved/failed membership count = %d, want %d", got, wantCount)
	}
	if _, err := os.Stat(filepath.Join(f.dir, "cluster", "trusted", peer.id.NodeUUID+".json")); !os.IsNotExist(err) {
		t.Fatalf("unapproved/failed peer has durable pin: %v", err)
	}
}

func askProcessDurable(t *testing.T, nodes ...*askProcess) {
	t.Helper()
	for _, local := range nodes {
		admission := readOutboundJSON[askProcessAuthority](t, filepath.Join(local.dir, "cluster", "admission.json"))
		if admission.ClusterID != nodes[0].id.ClusterID || admission.Epoch == 0 {
			t.Fatal("durable local admission mismatch")
		}
		roster := readOutboundJSON[[]askProcessAuthority](t, filepath.Join(local.dir, "cluster", "members.json"))
		if len(roster) != len(nodes) {
			t.Fatalf("durable roster count = %d, want %d", len(roster), len(nodes))
		}
		for _, peer := range nodes {
			peerAdmission := readOutboundJSON[askProcessAuthority](t, filepath.Join(peer.dir, "cluster", "admission.json"))
			cert := readLiveDurableFile(t, filepath.Join(peer.dir, "cluster", "node.crt"))
			block, _ := pem.Decode(cert)
			if block == nil {
				t.Fatal("fixture certificate cannot be decoded")
			}
			sum := sha256.Sum256(block.Bytes)
			fingerprint := "sha256:" + hex.EncodeToString(sum[:])
			found := false
			for _, row := range roster {
				if row.NodeUUID == peer.id.NodeUUID && row.ClusterID == admission.ClusterID && row.AdmissionEpoch == peerAdmission.Epoch {
					found = true
				}
			}
			if !found {
				t.Fatal("durable roster missing exact peer admission")
			}
			if peer == local {
				continue
			}
			pin := readOutboundJSON[askProcessAuthority](t, filepath.Join(local.dir, "cluster", "trusted", peer.id.NodeUUID+".json"))
			if pin.NodeUUID != peer.id.NodeUUID || pin.ClusterID != admission.ClusterID || pin.AdmissionEpoch != peerAdmission.Epoch || pin.CertFingerprint != fingerprint {
				t.Fatal("durable exact peer pin mismatch")
			}
		}
	}
}

func TestAskToJoinProcessThreeMembersAndRestart(t *testing.T) {
	a, b, c := newAskProcess(t, nil), newAskProcess(t, nil), newAskProcess(t, nil)
	a.create(t)
	askProcessPair(t, a, b)
	r := askProcessStart(t, c, b, false)
	askProcessNoPeer(t, b, c, 2)
	askProcessNoPeer(t, c, b, 0)
	r = askProcessApprove(t, b, r.RequestID)
	askProcessFinish(t, c, b, r, false)
	c.id = decodeResult[cmNodeID](t, c.p.call("cluster:get-node-id", nil))
	if c.id.ClusterID != a.id.ClusterID {
		t.Fatal("requester did not adopt selected member's cluster")
	}
	for _, f := range []*askProcess{a, b, c} {
		waitMembers(t, f.p, 3)
	}
	askProcessDurable(t, a, b, c)
	for _, f := range []*askProcess{b, c} {
		receipts := readOutboundJSON[map[string]struct {
			State     string `json:"state"`
			Rollback  bool   `json:"rollback"`
			ClusterID string `json:"clusterId"`
		}](t, filepath.Join(f.dir, "cluster", "pairing-receipts.json"))
		receipt, ok := receipts[*r.InviteID]
		if !ok || receipt.State != "paired" || receipt.Rollback || receipt.ClusterID != a.id.ClusterID {
			t.Fatal("correlated invite lacks durable paired receipt")
		}
	}
	c.stop(t)
	c.p = startCMEnv(t, c.dir, c.port, nil)
	c.p.call("cluster:set-identity", map[string]any{"clusterId": c.id.ClusterID, "clusterFriendlyName": "private admission process fixture"})
	if restored := decodeResult[cmNodeID](t, c.p.call("cluster:get-node-id", nil)); restored != c.id {
		t.Fatal("restart failed to restore paired identity")
	}
	waitMembers(t, c.p, 3)
	askProcessDurable(t, a, b, c)
}

func TestAskToJoinProcessSoloReplacement(t *testing.T) {
	a, b := newAskProcess(t, nil), newAskProcess(t, nil)
	a.create(t)
	b.create(t)
	old := b.id.ClusterID
	r := askProcessStart(t, b, a, true)
	if got := decodeResult[cmNodeID](t, b.p.call("cluster:get-node-id", nil)); got.ClusterID != old {
		t.Fatal("solo replaced before approval")
	}
	r = askProcessApprove(t, a, r.RequestID)
	askProcessFinish(t, b, a, r, false)
	b.id = decodeResult[cmNodeID](t, b.p.call("cluster:get-node-id", nil))
	if b.id.ClusterID == old || b.id.ClusterID != a.id.ClusterID {
		t.Fatal("intentional solo replacement failed")
	}
	askProcessDurable(t, a, b)
}

func TestAskToJoinProcessSoloDeclineAndExpiry(t *testing.T) {
	test := func(name, state string, env []string) {
		t.Run(name, func(t *testing.T) {
			a, b := newAskProcess(t, env), newAskProcess(t, env)
			a.create(t)
			b.create(t)
			before := readLiveDurableFile(t, filepath.Join(b.dir, "cluster", "admission.json"))
			r := askProcessStart(t, b, a, true)
			if state == "declined" {
				a.p.call("cluster:respond-to-join-request", map[string]any{"requestId": r.RequestID, "accept": false})
			}
			askProcessWait(t, b, r.RequestID, state)
			askProcessWait(t, a, r.RequestID, state)
			if got := decodeResult[cmNodeID](t, b.p.call("cluster:get-node-id", nil)); got != b.id {
				t.Fatal("pre-approval terminal outcome changed solo identity")
			}
			if !bytes.Equal(before, readLiveDurableFile(t, filepath.Join(b.dir, "cluster", "admission.json"))) {
				t.Fatal("pre-approval terminal outcome changed solo admission")
			}
			askProcessNoPeer(t, a, b, 1)
			askProcessNoPeer(t, b, a, 1)
		})
	}
	test("decline", "declined", nil)
	test("expiry", "expired", []string{"NVPAIR_TEST_INVITE_TTL_MS=700", "NVPAIR_TEST_INVITE_EXPIRY_INTERVAL_MS=25"})
}

func askProcessRosterRecords(t *testing.T, f *askProcess) map[string]map[string]json.RawMessage {
	t.Helper()
	rows := readOutboundJSON[[]map[string]json.RawMessage](t, filepath.Join(f.dir, "cluster", "members.json"))
	records := make(map[string]map[string]json.RawMessage, len(rows))
	for _, row := range rows {
		var uuid string
		if err := json.Unmarshal(row["nodeUuid"], &uuid); err != nil || uuid == "" {
			t.Fatal("durable roster row lacks a valid identity")
		}
		if _, exists := records[uuid]; exists {
			t.Fatalf("durable roster duplicates identity %s", uuid)
		}
		records[uuid] = row
	}
	return records
}

func TestAskToJoinProcessMultiMemberRefusal(t *testing.T) {
	a, b, c := newAskProcess(t, nil), newAskProcess(t, nil), newAskProcess(t, nil)
	a.create(t)
	c.create(t)
	askProcessPair(t, a, b)
	// Establish the paired durable baseline before testing refusal; members.json
	// carries admission tuples, while certificate fingerprints live in pins.
	askProcessDurable(t, a, b)
	before := askProcessRosterRecords(t, a)
	resp := a.p.callExpectError("cluster:ask-to-join", map[string]any{"address": "127.0.0.1", "port": c.port, "replaceSoloCluster": true})
	if resp.Error == nil || resp.Error.Message != "already-clustered" {
		t.Fatal("multi-member requester did not refuse retained relationships")
	}
	after := askProcessRosterRecords(t, a)
	changed := false
	for uuid, record := range before {
		if current, exists := after[uuid]; !exists {
			t.Errorf("refused request removed durable identity %s", uuid)
			changed = true
		} else if !reflect.DeepEqual(record, current) {
			t.Errorf("refused request modified durable identity %s", uuid)
			changed = true
		}
	}
	for uuid := range after {
		if _, exists := before[uuid]; !exists {
			t.Errorf("refused request added durable identity %s", uuid)
			changed = true
		}
	}
	if changed {
		t.FailNow()
	}
	askProcessDurable(t, a, b)
	askProcessNoPeer(t, c, a, 1)
}

func TestAskToJoinProcessWrongPINCompensation(t *testing.T) {
	a, b := newAskProcess(t, nil), newAskProcess(t, nil)
	a.create(t)
	r := askProcessStart(t, b, a, false)
	r = askProcessApprove(t, a, r.RequestID)
	askProcessFinish(t, b, a, r, true)
	askProcessNoPeer(t, a, b, 1)
	askProcessNoPeer(t, b, a, 0)
	if got := decodeResult[cmNodeID](t, b.p.call("cluster:get-node-id", nil)); got.ClusterID != "" {
		t.Fatal("wrong PIN retained adopted identity")
	}
}

func askProcessWire(t *testing.T, port int, operation, id string, requester *askProcess) (int, askProcessRequest) {
	t.Helper()
	wire := map[string]any{"version": 1, "operation": operation, "requestId": id}
	if requester != nil {
		wire["requester"] = map[string]string{"nodeUuid": requester.id.NodeUUID, "name": "unverified fixture"}
		wire["port"] = requester.port
	}
	body, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Post(fmt.Sprintf("http://127.0.0.1:%d/v1/cluster/join-requests", port), "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result askProcessRequest
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&result); err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, result
}

func TestAskToJoinProcessUnattendedReviewAndDuplicateReady(t *testing.T) {
	a, b := newAskProcess(t, nil), newAskProcess(t, nil)
	a.create(t)
	const id = "jr-00000000000000000000000000000004"
	status, r := askProcessWire(t, a.port, "request", id, b)
	if status != http.StatusOK || r.State != "awaiting-approval" {
		t.Fatal("unattended request was not retained")
	}
	// No approver API is polled until after arrival; the getter recovers review.
	askProcessWait(t, a, id, "awaiting-approval")
	askProcessNoPeer(t, a, b, 1)
	first := askProcessApprove(t, a, id)
	second := askProcessApprove(t, a, id)
	if *first.InviteID != *second.InviteID {
		t.Fatal("duplicate approval allocated a second invite")
	}
	status, ready := askProcessWire(t, a.port, "ready", id, nil)
	status2, again := askProcessWire(t, a.port, "ready", id, nil)
	if status != http.StatusOK || status2 != http.StatusOK || ready.State != "pairing" || again.State != "pairing" || ready.InviteID == nil || again.InviteID == nil || *ready.InviteID != *again.InviteID || *ready.InviteID != *first.InviteID {
		t.Fatal("duplicate ready changed correlated invite")
	}
	askProcessNoPeer(t, a, b, 1)
	askProcessNoPeer(t, b, a, 0)
}

func TestAskToJoinProcessCancelFencesLateReady(t *testing.T) {
	a, b := newAskProcess(t, nil), newAskProcess(t, nil)
	a.create(t)
	r := askProcessStart(t, b, a, false)
	b.p.call("cluster:cancel-join-request", map[string]any{"requestId": r.RequestID})
	askProcessWait(t, a, r.RequestID, "canceled")
	status, late := askProcessWire(t, a.port, "ready", r.RequestID, nil)
	if status != http.StatusOK || late.State != "canceled" || late.InviteID != nil {
		t.Fatal("late ready escaped cancellation")
	}
	askProcessWait(t, b, r.RequestID, "canceled")
	askProcessNoPeer(t, a, b, 1)
	askProcessNoPeer(t, b, a, 0)
}

func TestAskToJoinProcessMemberLeaveBeforeReady(t *testing.T) {
	a, b := newAskProcess(t, nil), newAskProcess(t, nil)
	a.create(t)
	const id = "jr-00000000000000000000000000000005"
	status, _ := askProcessWire(t, a.port, "request", id, b)
	if status != http.StatusOK {
		t.Fatal("request failed")
	}
	askProcessApprove(t, a, id)
	a.p.call("cluster:leave", nil)
	status, late := askProcessWire(t, a.port, "ready", id, nil)
	if status != http.StatusOK || late.State != "rejected" || late.Reason != "approver-changed" {
		t.Fatalf("ready after leave: status=%d state=%q reason=%q", status, late.State, late.Reason)
	}
	askProcessNoPeer(t, a, b, 0)
	askProcessNoPeer(t, b, a, 0)
}

func TestAskToJoinProcessTargetLoss(t *testing.T) {
	a, b := newAskProcess(t, nil), newAskProcess(t, nil)
	a.create(t)
	r := askProcessStart(t, b, a, false)
	a.stop(t)
	failed := askProcessWait(t, b, r.RequestID, "failed")
	if failed.Reason == "" {
		t.Fatal("target loss lacks refusal reason")
	}
	askProcessNoPeer(t, b, a, 0)
	if got := decodeResult[cmNodeID](t, b.p.call("cluster:get-node-id", nil)); got.ClusterID != "" {
		t.Fatal("target loss changed requester identity")
	}
}

func TestAskToJoinProcessOldEndpointRefusal(t *testing.T) {
	b := newAskProcess(t, nil)
	old := httptest.NewServer(http.NotFoundHandler())
	defer old.Close()
	var port int
	if _, err := fmt.Sscanf(old.URL, "http://127.0.0.1:%d", &port); err != nil {
		t.Fatal(err)
	}
	r := decodeResult[askProcessRequest](t, b.p.call("cluster:ask-to-join", map[string]any{"address": "127.0.0.1", "port": port, "replaceSoloCluster": false}))
	failed := askProcessWait(t, b, r.RequestID, "failed")
	if failed.Reason != "unsupported-peer" || failed.InviteID != nil {
		t.Fatalf("old endpoint refusal reason = %q", failed.Reason)
	}
	if got := memberCount(t, b.p); got != 0 {
		t.Fatalf("old endpoint granted %d members", got)
	}
}
