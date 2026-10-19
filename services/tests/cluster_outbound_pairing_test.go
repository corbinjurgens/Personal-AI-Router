// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
)

type outboundInviteResult struct {
	InviteID string `json:"inviteId"`
	State    string `json:"state"`
	Pin      string `json:"pin"`
	Reason   string `json:"reason"`
}

type blockedPairingFixture struct {
	a, b         *cmProc
	aDir, bDir   string
	aPort, bPort int
	aID, bID     cmNodeID
	invite       outboundInviteResult
}

func temporaryClusterPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if err := ln.Close(); err != nil {
		t.Fatal(err)
	}
	return port
}

func assertClusterListenerRejects(t *testing.T, port int) {
	t.Helper()
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), time.Second)
	if err != nil {
		t.Fatalf("temporary blocked listener did not bind: %v", err)
	}
	defer conn.Close()
	if err := conn.SetDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	_, _ = conn.Write([]byte("POST /v1/cluster/pairing HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n"))
	var buf [1]byte
	n, err := conn.Read(buf[:])
	if n != 0 || err == nil {
		t.Fatal("temporary inviter listener accepted an inbound HTTP request")
	}
	if timeout, ok := err.(net.Error); ok && timeout.Timeout() {
		t.Fatal("inviter listener stalled rather than rejecting inbound connection")
	}
}

func newBlockedPairingFixture(t *testing.T, joinerEnv []string) blockedPairingFixture {
	return newBlockedPairingFixtureEnv(t, nil, joinerEnv)
}

func newBlockedPairingFixtureEnv(t *testing.T, inviterEnv, joinerEnv []string) blockedPairingFixture {
	t.Helper()
	f := blockedPairingFixture{aDir: shortClusterTempDir(t), bDir: shortClusterTempDir(t), aPort: temporaryClusterPort(t), bPort: temporaryClusterPort(t)}
	f.a = startCMEnv(t, f.aDir, f.aPort, append([]string{"NVPAIR_TEST_PAIR_REJECT_INBOUND=1"}, inviterEnv...))
	f.b = startCMEnv(t, f.bDir, f.bPort, joinerEnv)
	t.Cleanup(f.a.stop)
	t.Cleanup(f.b.stop)
	f.aID = decodeResult[cmNodeID](t, f.a.call("cluster:get-node-id", nil))
	f.bID = decodeResult[cmNodeID](t, f.b.call("cluster:get-node-id", nil))
	time.Sleep(200 * time.Millisecond)
	assertClusterListenerRejects(t, f.aPort)
	f.a.call("cluster:create", map[string]any{"clusterFriendlyName": "isolated outbound regression"})
	f.aID = decodeResult[cmNodeID](t, f.a.call("cluster:get-node-id", nil))
	f.invite = decodeResult[outboundInviteResult](t, f.a.call("cluster:invite-node", map[string]any{"address": "127.0.0.1", "port": f.bPort}))
	if f.invite.State != "pending" || len(f.invite.Pin) != 6 {
		t.Fatalf("initial invite = %+v, want pending with PIN", f.invite)
	}
	f.b.waitNotify("cluster:invite-received")
	return f
}

func readOutboundJSON[T any](t *testing.T, path string) T {
	t.Helper()
	data := readLiveDurableFile(t, path)
	var result T
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	return result
}

func assertOutboundDurablePair(t *testing.T, f blockedPairingFixture) {
	t.Helper()
	type admission struct {
		ClusterID string `json:"clusterId"`
		Epoch     uint64 `json:"epoch"`
	}
	type member struct {
		NodeUUID        string `json:"nodeUuid"`
		ClusterID       string `json:"clusterId"`
		AdmissionEpoch  uint64 `json:"admissionEpoch"`
		CertFingerprint string `json:"certFingerprint"`
	}
	a := readOutboundJSON[admission](t, filepath.Join(f.aDir, "cluster", "admission.json"))
	b := readOutboundJSON[admission](t, filepath.Join(f.bDir, "cluster", "admission.json"))
	if a.ClusterID != f.aID.ClusterID || b.ClusterID != a.ClusterID || a.Epoch == 0 || b.Epoch == 0 {
		t.Fatalf("durable admissions disagree: inviter=%+v joiner=%+v", a, b)
	}
	check := func(dir, peerUUID string, peerEpoch uint64) {
		pin := readOutboundJSON[member](t, filepath.Join(dir, "cluster", "trusted", peerUUID+".json"))
		if pin.NodeUUID != peerUUID || pin.ClusterID != a.ClusterID || pin.AdmissionEpoch != peerEpoch || pin.CertFingerprint == "" {
			t.Fatalf("durable peer pin = %+v, want %s admission %d", pin, peerUUID, peerEpoch)
		}
		members := readOutboundJSON[[]member](t, filepath.Join(dir, "cluster", "members.json"))
		if len(members) != 2 {
			t.Fatalf("durable roster has %d rows, want self and peer", len(members))
		}
		for _, row := range members {
			if row.NodeUUID == peerUUID && row.ClusterID == a.ClusterID && row.AdmissionEpoch == peerEpoch {
				return
			}
		}
		t.Fatalf("durable roster has no matching peer admission: %+v", members)
	}
	check(f.aDir, f.bID.NodeUUID, b.Epoch)
	check(f.bDir, f.aID.NodeUUID, a.Epoch)
}

func TestClusterOutboundPairingBlockedInviter(t *testing.T) {
	f := newBlockedPairingFixture(t, nil)
	response := decodeResult[outboundInviteResult](t, f.b.call("cluster:respond-to-invite", map[string]any{
		"inviteId": f.invite.InviteID, "accept": true, "pin": f.invite.Pin,
	}))
	if response.State != "paired" {
		t.Fatalf("joiner response state = %q, want paired", response.State)
	}
	waitInviteState(t, f.a, f.invite.InviteID, "paired")
	// Paired precedes the joiner's post-commit roster publication. The
	// inviter publishes both confirmed members only after confirming the
	// joiner and completing its own acknowledgement-time roster write.
	f.a.pump(func(msg jsonrpc.Message) bool {
		if msg.Method != "nodes:changed" {
			return false
		}
		var snapshot struct {
			Nodes []struct {
				NodeUUID string `json:"nodeUuid"`
				State    string `json:"state"`
			} `json:"nodes"`
		}
		if err := json.Unmarshal(msg.Params, &snapshot); err != nil {
			t.Fatalf("decode inviter membership notification: %v", err)
		}
		if len(snapshot.Nodes) != 2 {
			return false
		}
		self, peer := false, false
		for _, node := range snapshot.Nodes {
			self = self || node.NodeUUID == f.aID.NodeUUID && node.State == "member"
			peer = peer || node.NodeUUID == f.bID.NodeUUID && node.State == "member"
		}
		return self && peer
	}, 15*time.Second)
	assertClusterListenerRejects(t, f.aPort)
	assertOutboundDurablePair(t, f)
}

func TestClusterOutboundPairingLostFinalReplyAndConfirmation(t *testing.T) {
	f := newBlockedPairingFixture(t, []string{"NVPAIR_TEST_PAIR_DROP_FINAL_REPLY=1", "NVPAIR_TEST_PAIR_DROP_CONFIRMATION=1"})
	response := decodeResult[outboundInviteResult](t, f.b.call("cluster:respond-to-invite", map[string]any{
		"inviteId": f.invite.InviteID, "accept": true, "pin": f.invite.Pin,
	}))
	if response.State != "paired" {
		t.Fatalf("lost final reply changed joiner commit to %q", response.State)
	}
	waitInviteState(t, f.a, f.invite.InviteID, "paired")
	assertOutboundDurablePair(t, f)
	// Restart only this explicitly-created temporary inviter. Its admission,
	// member and pin commits must remain durable despite unknown confirmation.
	f.a.stop()
	f.a = startCMEnv(t, f.aDir, f.aPort, []string{"NVPAIR_TEST_PAIR_REJECT_INBOUND=1"})
	t.Cleanup(f.a.stop)
	f.aID = decodeResult[cmNodeID](t, f.a.call("cluster:get-node-id", nil))
	time.Sleep(300 * time.Millisecond)
	assertClusterListenerRejects(t, f.aPort)
	assertOutboundDurablePair(t, f)
	waitInviteState(t, f.a, f.invite.InviteID, "pending")
	status := decodeResult[outboundInviteResult](t, f.a.call("cluster:invite-status", map[string]any{"inviteId": f.invite.InviteID}))
	if status.Reason != "awaiting-confirmation" {
		t.Fatalf("unknown recovery republished paired instead of awaiting confirmation: %+v", status)
	}
}

func TestClusterOutboundPairingWrongPIN(t *testing.T) {
	f := newBlockedPairingFixture(t, nil)
	wrong := "000000"
	if f.invite.Pin == wrong {
		wrong = "000001"
	}
	response := decodeResult[outboundInviteResult](t, f.b.call("cluster:respond-to-invite", map[string]any{
		"inviteId": f.invite.InviteID, "accept": true, "pin": wrong,
	}))
	if response.State != "failed" || response.Reason != "incorrect-pin" {
		t.Fatalf("wrong PIN outcome = %+v, want failed/incorrect-pin", response)
	}
	waitInviteState(t, f.a, f.invite.InviteID, "failed")
	if got := memberCount(t, f.a); got != 1 {
		t.Fatalf("wrong PIN granted inviter peer membership: %d", got)
	}
	if got := memberCount(t, f.b); got != 0 {
		t.Fatalf("wrong PIN granted joiner membership: %d", got)
	}
	assertReceiverMembershipCleared(t, f.b)
}

func assertReceiverMembershipCleared(t *testing.T, receiver *cmProc) {
	t.Helper()
	msg := receiver.waitNotify("nodes:changed")
	var snapshot struct {
		Nodes []json.RawMessage `json:"nodes"`
	}
	if err := json.Unmarshal(msg.Params, &snapshot); err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Nodes) != 0 {
		t.Fatalf("failed receiver notified stale public membership: %+v", snapshot.Nodes)
	}
}

func TestClusterOutboundPairingReceiptFailureClearsPublicMembership(t *testing.T) {
	f := newBlockedPairingFixture(t, nil)
	if err := os.Mkdir(filepath.Join(f.bDir, "cluster", "pairing-receipts.json"), 0o700); err != nil {
		t.Fatal(err)
	}
	result := decodeResult[outboundInviteResult](t, f.b.call("cluster:respond-to-invite", map[string]any{
		"inviteId": f.invite.InviteID, "accept": true, "pin": f.invite.Pin,
	}))
	if result.State != "failed" {
		t.Fatalf("receipt-writer fault status = %+v", result)
	}
	if count := memberCount(t, f.b); count != 0 {
		t.Fatalf("receipt-writer fault left %d public pending/member rows", count)
	}
	assertReceiverMembershipCleared(t, f.b)
	waitInviteState(t, f.a, f.invite.InviteID, "failed")
}

// renameOutboundFaultFile changes a fault under a live cluster manager. Neither
// paired nor rollback-pending parks its background authority readers or rollback
// retries. Unlike product durable writes, os.Rename does not retry Windows
// ERROR_ACCESS_DENIED (5) / ERROR_SHARING_VIOLATION (32) while those handles close.
func renameOutboundFaultFile(t *testing.T, from, to string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		err := os.Rename(from, to)
		if err == nil {
			return
		}
		if !isWindowsSharingError(err) {
			t.Fatal(err)
		}
		if time.Now().After(deadline) {
			t.Fatalf("rename fault file after transient Windows sharing conflicts: %v", err)
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func TestClusterOutboundPairingRollbackWaitsForDurableCleanup(t *testing.T) {
	f := newBlockedPairingFixtureEnv(t, []string{"NVPAIR_TEST_INVITE_EXPIRY_INTERVAL_MS=25"}, []string{"NVPAIR_TEST_PAIR_CONFIRMATION_DELAY_MS=1000"})
	admissionPath := filepath.Join(f.bDir, "cluster", "admission.json")
	renameOutboundFaultFile(t, admissionPath, admissionPath+".fault-backup")
	if err := os.Mkdir(admissionPath, 0o700); err != nil {
		t.Fatal(err)
	}
	memberPath := filepath.Join(f.aDir, "cluster", "members.json")
	backup := memberPath + ".fault-backup"
	result := decodeResult[outboundInviteResult](t, f.b.call("cluster:respond-to-invite", map[string]any{
		"inviteId": f.invite.InviteID, "accept": true, "pin": f.invite.Pin,
	}))
	if result.State != "failed" {
		t.Fatalf("joiner commit fault status = %+v", result)
	}
	waitInviteState(t, f.a, f.invite.InviteID, "paired")
	renameOutboundFaultFile(t, memberPath, backup)
	if err := os.Mkdir(memberPath, 0o700); err != nil {
		t.Fatal(err)
	}
	waitInviteState(t, f.a, f.invite.InviteID, "pending")
	status := decodeResult[outboundInviteResult](t, f.a.call("cluster:invite-status", map[string]any{"inviteId": f.invite.InviteID}))
	if status.Reason != "rollback-pending" {
		t.Fatalf("incomplete durable compensation status = %+v", status)
	}
	if err := os.Remove(memberPath); err != nil {
		t.Fatal(err)
	}
	// Let the retry recreate members.json from the compensated live roster.
	// Restoring the pre-rollback backup could overwrite a successful retry
	// with the old peer admission (and races its readers on Windows).
	waitInviteState(t, f.a, f.invite.InviteID, "failed")
	if count := memberCount(t, f.a); count != 1 {
		t.Fatalf("completed compensation retained %d inviter rows, want self only", count)
	}
	members := readOutboundJSON[[]struct {
		NodeUUID string `json:"nodeUuid"`
	}](t, memberPath)
	if len(members) != 1 || members[0].NodeUUID != f.aID.NodeUUID {
		t.Fatalf("completed compensation durable inviter roster = %+v, want only self %s", members, f.aID.NodeUUID)
	}
	if _, err := os.Stat(filepath.Join(f.aDir, "cluster", "trusted", f.bID.NodeUUID+".json")); !os.IsNotExist(err) {
		t.Fatalf("completed compensation retained durable peer pin: %v", err)
	}
}

func TestClusterOutboundPairingTerminalOutcomes(t *testing.T) {
	test := func(name string, env []string, action func(*testing.T, blockedPairingFixture), want string) {
		t.Run(name, func(t *testing.T) {
			f := newBlockedPairingFixture(t, env)
			action(t, f)
			waitInviteState(t, f.a, f.invite.InviteID, want)
			waitInviteState(t, f.b, f.invite.InviteID, want)
			if got := memberCount(t, f.a); got != 1 {
				t.Fatalf("terminal %s retained peer on inviter: %d", want, got)
			}
			if got := memberCount(t, f.b); got != 0 {
				t.Fatalf("terminal %s retained peer on joiner: %d", want, got)
			}
			assertClusterListenerRejects(t, f.aPort)
		})
	}
	test("decline without reverse dial", nil, func(t *testing.T, f blockedPairingFixture) {
		f.b.call("cluster:respond-to-invite", map[string]any{"inviteId": f.invite.InviteID, "accept": false})
	}, "declined")
	test("cancel while pending", nil, func(t *testing.T, f blockedPairingFixture) {
		f.a.call("cluster:cancel-invite", map[string]any{"inviteId": f.invite.InviteID})
	}, "canceled")
	test("unanswered expiry without reverse dial", []string{"NVPAIR_TEST_INVITE_TTL_MS=500", "NVPAIR_TEST_INVITE_EXPIRY_INTERVAL_MS=25"}, func(t *testing.T, f blockedPairingFixture) {}, "expired")
}

func TestClusterOutboundPairingAuthenticatedJoinerPinFailure(t *testing.T) {
	f := newBlockedPairingFixture(t, []string{"NVPAIR_TEST_INVITE_EXPIRY_INTERVAL_MS=25"})
	path := filepath.Join(f.bDir, "cluster", "trusted", f.aID.NodeUUID+".json")
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	// Keep PIN submission in flight while querying the authenticated pending
	// outcome. Its bounded submission wait must return pending; polling that
	// same invite after fault repair must follow the durable failed outcome.
	id := f.b.nextID
	f.b.nextID++
	request, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": id,
		"method": "cluster:respond-to-invite", "params": map[string]any{
			"inviteId": f.invite.InviteID, "accept": true, "pin": f.invite.Pin,
		}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.b.stdin.Write(append(request, '\n')); err != nil {
		t.Fatal(err)
	}
	waitInviteState(t, f.a, f.invite.InviteID, "pending")
	deadline := time.Now().Add(5 * time.Second)
	confirmedPending := false
	for time.Now().Before(deadline) {
		status := decodeResult[outboundInviteResult](t, f.a.call("cluster:invite-status", map[string]any{"inviteId": f.invite.InviteID}))
		if status.State == "pending" && status.Reason == "awaiting-confirmation" {
			confirmedPending = true
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if !confirmedPending {
		t.Fatal("incomplete join compensation was not authenticated as pending")
	}
	receipts := readOutboundJSON[map[string]struct {
		Rollback bool   `json:"rollback"`
		State    string `json:"state"`
	}](t, filepath.Join(f.bDir, "cluster", "pairing-receipts.json"))
	if receipt, ok := receipts[f.invite.InviteID]; !ok || !receipt.Rollback || receipt.State == "failed" {
		t.Fatalf("unfinished join compensation lacks durable pending intent: %+v", receipt)
	}
	response := decodeResult[outboundInviteResult](t, f.b.pump(func(m jsonrpc.Message) bool {
		return m.Method == "" && idEquals(m.ID, id)
	}, 40*time.Second))
	if response.State != "pending" || response.Reason != "rollback-pending" {
		t.Fatalf("bounded PIN response for incomplete compensation = %+v", response)
	}
	status := decodeResult[outboundInviteResult](t, f.b.call("cluster:invite-status", map[string]any{"inviteId": f.invite.InviteID}))
	if status.State != "pending" || status.Reason != "rollback-pending" {
		t.Fatalf("matching joiner status before durable cleanup = %+v", status)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	waitInviteState(t, f.b, f.invite.InviteID, "failed")
	waitInviteState(t, f.a, f.invite.InviteID, "failed")
	if _, err := os.Stat(filepath.Join(f.aDir, "cluster", "trusted", f.bID.NodeUUID+".json")); !os.IsNotExist(err) {
		t.Fatalf("authenticated matching failure retained inviter's durable pin: %v", err)
	}
	if got := memberCount(t, f.a); got != 1 {
		t.Fatalf("authenticated failure retained provisional peer: %d members", got)
	}
	if info := decodeResult[cmNodeID](t, f.b.call("cluster:get-node-id", nil)); info.ClusterID != "" {
		t.Fatalf("failed joiner adopted cluster %q", info.ClusterID)
	}
}

// Normal gates exercise the legacy transport at the component boundary. When
// intake supplies an independently built historical binary, this additionally
// drives both mixed-version pairings over the real public process interface.
func TestClusterOutboundPairingLegacyBinary(t *testing.T) {
	legacy := os.Getenv("NVPAIR_TEST_LEGACY_CLUSTER_BINARY")
	if legacy == "" {
		t.Log("historical binary not supplied; mixed-version process proof not measured")
		return
	}
	if info, err := os.Stat(legacy); err != nil || info.IsDir() {
		t.Fatalf("historical binary unavailable: %v", err)
	}
	test := func(name, inviterBin, joinerBin string) {
		t.Run(name, func(t *testing.T) {
			aDir, bDir := shortClusterTempDir(t), shortClusterTempDir(t)
			aPort, bPort := temporaryClusterPort(t), temporaryClusterPort(t)
			a := startCMBinaryEnv(t, inviterBin, aDir, aPort, nil)
			b := startCMBinaryEnv(t, joinerBin, bDir, bPort, nil)
			t.Cleanup(a.stop)
			t.Cleanup(b.stop)
			a.call("cluster:get-node-id", nil)
			bID := decodeResult[cmNodeID](t, b.call("cluster:get-node-id", nil))
			time.Sleep(200 * time.Millisecond)
			a.call("cluster:create", map[string]any{"clusterFriendlyName": "isolated legacy regression"})
			aID := decodeResult[cmNodeID](t, a.call("cluster:get-node-id", nil))
			inv := decodeResult[outboundInviteResult](t, a.call("cluster:invite-node", map[string]any{"address": "127.0.0.1", "port": bPort}))
			if inv.State != "pending" || len(inv.Pin) != 6 {
				t.Fatalf("mixed-version initial = %+v", inv)
			}
			b.waitNotify("cluster:invite-received")
			response := decodeResult[outboundInviteResult](t, b.call("cluster:respond-to-invite", map[string]any{"inviteId": inv.InviteID, "accept": true, "pin": inv.Pin}))
			if response.State != "paired" {
				t.Fatalf("mixed-version completion state = %q", response.State)
			}
			waitInviteState(t, a, inv.InviteID, "paired")
			assertOutboundDurablePair(t, blockedPairingFixture{a: a, b: b, aDir: aDir, bDir: bDir, aID: aID, bID: bID})
		})
	}
	test("new inviter legacy joiner", clusterMgrBin, legacy)
	test("legacy inviter new joiner", legacy, clusterMgrBin)
}
