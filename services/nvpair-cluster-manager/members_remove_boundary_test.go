// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"nvpair-shared/noderec"
)

type signaledPipeBody struct {
	*io.PipeReader
	started chan struct{}
	once    sync.Once
}

func (b *signaledPipeBody) Read(p []byte) (int, error) {
	b.once.Do(func() { close(b.started) })
	return b.PipeReader.Read(p)
}

func removeRequest(t *testing.T, remover *Manager, proof RemovalProof, body io.ReadCloser) *http.Request {
	t.Helper()
	if body == nil {
		body = io.NopCloser(bytes.NewReader(mustJSON(t, membersRemoveRequest{
			NodeUUID: remover.identity.NodeUUID,
			Proof:    proof,
		})))
	}
	req := httptest.NewRequest(http.MethodPost, membersRemovePath, body)
	cert, err := x509.ParseCertificate(remover.identity.Cert.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	req.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{cert}}
	return req
}

func TestMembersRemoveRevalidatesAfterBlockedBody(t *testing.T) {
	victim := newTestManagerPort(t, 15121)
	remover := newTestManagerPort(t, 15122)
	pinTrusted(t, victim, remover.identity.NodeUUID, string(remover.identity.CertPEM), remover.identity.CertFingerprint)
	_, oldEpoch := victim.currentAdmission()
	proof, err := remover.newRemovalProof(victim.identity.NodeUUID, oldEpoch)
	if err != nil {
		t.Fatal(err)
	}
	payload := mustJSON(t, membersRemoveRequest{NodeUUID: remover.identity.NodeUUID, Proof: proof})

	pr, pw := io.Pipe()
	started := make(chan struct{})
	body := &signaledPipeBody{PipeReader: pr, started: started}
	req := removeRequest(t, remover, proof, body)
	rr := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		victim.handleMembersRemove(rr, req)
		close(done)
	}()

	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("remove handler did not begin body read")
	}

	// The old request is authenticated but stalled. Tear down, re-admit into the
	// same cluster with a newer epoch, and restore a valid new composition before
	// releasing the old body.
	victim.teardownClusterLocal()
	newEpoch := activateTestCluster(t, victim, "cluster-1")
	if newEpoch <= oldEpoch {
		t.Fatalf("new admission %d did not advance past %d", newEpoch, oldEpoch)
	}
	pinTrusted(t, victim, remover.identity.NodeUUID, string(remover.identity.CertPEM), remover.identity.CertFingerprint)
	victim.addSelfMember()
	if _, err := pw.Write(payload); err != nil {
		t.Fatal(err)
	}
	_ = pw.Close()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("remove handler did not finish")
	}
	if rr.Code != http.StatusConflict {
		t.Fatalf("status = %d, want %d for stale removal", rr.Code, http.StatusConflict)
	}
	if cid, epoch := victim.currentAdmission(); cid != "cluster-1" || epoch != newEpoch {
		t.Fatalf("stale request cleared new admission: (%q,%d)", cid, epoch)
	}
	if _, ok := victim.trust.Get(remover.identity.NodeUUID); !ok {
		t.Fatal("stale request cleared the newly re-established remover pin")
	}
}

func TestMembersRemoveCurrentAdmissionSucceeds(t *testing.T) {
	victim := newTestManagerPort(t, 15123)
	remover := newTestManagerPort(t, 15124)
	pinTrusted(t, victim, remover.identity.NodeUUID, string(remover.identity.CertPEM), remover.identity.CertFingerprint)
	_, epoch := victim.currentAdmission()
	proof, err := remover.newRemovalProof(victim.identity.NodeUUID, epoch)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	victim.handleMembersRemove(rr, removeRequest(t, remover, proof, nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rr.Code, rr.Body.String())
	}
	assertFullyUnclustered(t, victim)
}

func TestReverseRemovalLostReplyDoesNotReplay(t *testing.T) {
	remover, victim := newClusterPeerFixture(t, false), newClusterPeerFixture(t, true)
	pairClusterFixtures(t, remover, victim)
	startPrivateReverse(t, remover.m)
	connectReverseManagers(t, remover.m, victim.m)
	reverse := countClusterReverseDials(t, remover.m)
	// A roster leaves a reusable stream idle. Removal must open its own fresh
	// stream and must not retry when teardown revokes that stream's reply.
	if outcome, _ := remover.m.reconcileWith([]string{victim.address}, victim.m.identity.NodeUUID); outcome != reconcileAccepted {
		t.Fatalf("prime roster outcome = %v, want accepted", outcome)
	}
	if reverse.Load() != 1 || victim.rosters.Load() != 1 {
		t.Fatalf("prime roster reverse=%d rosters=%d, want 1/1", reverse.Load(), victim.rosters.Load())
	}
	committed := make(chan struct{})
	victim.afterRemove = func() { close(committed) }
	_, epoch := victim.m.currentAdmission()
	proof, err := remover.m.newRemovalProof(victim.m.identity.NodeUUID, epoch)
	if err != nil {
		t.Fatal(err)
	}
	client, err := remover.m.removalPeerClient(victim.m.identity.NodeUUID)
	if err != nil {
		t.Fatal(err)
	}
	body := mustJSON(t, membersRemoveRequest{NodeUUID: remover.m.identity.NodeUUID, Proof: proof})
	response, err := client.Post("https://"+victim.address+membersRemovePath, "application/json", bytes.NewReader(body))
	if response != nil {
		response.Body.Close()
	}
	if err == nil || response != nil {
		t.Fatalf("revoked reverse session exposed a reply: response=%v err=%v", response, err)
	}
	select {
	case <-committed:
	case <-time.After(2 * time.Second):
		t.Fatal("missing reply was not followed by durable remote teardown")
	}
	assertFullyUnclustered(t, victim.m)
	if victim.removes.Load() != 1 || reverse.Load() != 2 {
		t.Fatalf("removal dispatched=%d reverse=%d, want 1 removal and 2 streams including the priming roster", victim.removes.Load(), reverse.Load())
	}
	peer, ok := remover.m.reversePeer(victim.m.identity.NodeUUID)
	if !ok || remover.m.reverse.manager.Has(peer) {
		t.Fatal("remote teardown did not revoke its reverse session while local membership was retained")
	}
	assertRemovalDurableAfterRestart(t, victim.m)
}

func TestNodesRemoveCommitsAfterReverseReplyIsRevoked(t *testing.T) {
	remover, victim := newClusterPeerFixture(t, false), newClusterPeerFixture(t, true)
	pairClusterFixtures(t, remover, victim)
	startPrivateReverse(t, remover.m)
	connectReverseManagers(t, remover.m, victim.m)
	reverse := countClusterReverseDials(t, remover.m)
	committed := make(chan struct{})
	victim.afterRemove = func() { close(committed) }
	var output bytes.Buffer
	remover.m.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &output})
	uuid := victim.m.identity.NodeUUID
	id := json.RawMessage(`1`)
	remover.m.handleNodesRemove(&Message{ID: &id, Params: mustJSON(t, removeParams{NodeUUID: &uuid})})
	select {
	case <-committed:
	case <-time.After(2 * time.Second):
		t.Fatal("remote removal did not commit")
	}
	var result struct {
		Removed bool `json:"removed"`
	}
	decoder := json.NewDecoder(&output)
	responses := 0
	for decoder.More() {
		var message Message
		if err := decoder.Decode(&message); err != nil {
			t.Fatal(err)
		}
		if message.ID == nil || string(*message.ID) != "1" {
			continue
		}
		responses++
		if message.Error != nil {
			t.Fatalf("nodes:remove reported failure after durable teardown: %+v", message.Error)
		}
		if err := json.Unmarshal(message.Result, &result); err != nil {
			t.Fatal(err)
		}
	}
	if responses != 1 || !result.Removed || victim.removes.Load() != 1 || reverse.Load() != 1 {
		t.Fatalf("responses=%d removed=%v notifications=%d reverse=%d, want 1/true/1/1", responses, result.Removed, victim.removes.Load(), reverse.Load())
	}
	if _, ok := remover.m.trust.Get(uuid); ok {
		t.Fatal("local removal retained victim's pin")
	}
	if _, ok := remover.m.memberByNodeID(uuid); ok {
		t.Fatal("local removal retained victim's membership")
	}
	proofs := remover.m.snapshotRemovalProofs()
	if len(proofs) != 1 || proofs[0].Tombstone.NodeUUID != uuid || proofs[0].Tombstone.AdmissionEpoch != 1 {
		t.Fatalf("local durable removal proof = %+v, want victim admission 1", proofs)
	}
	assertFullyUnclustered(t, victim.m)
	assertRemovalDurableAfterRestart(t, victim.m)
}

func assertRemovalDurableAfterRestart(t *testing.T, victim *Manager) {
	t.Helper()
	if cid, epoch := victim.currentAdmission(); cid != "" || epoch != 0 || victim.teardownPending.Load() {
		t.Fatalf("remote durable teardown incomplete: admission=(%q,%d) pending=%v", cid, epoch, victim.teardownPending.Load())
	}
	if pending, err := victim.pendingTeardown(); err != nil || pending {
		t.Fatalf("remote teardown marker: pending=%v err=%v", pending, err)
	}
	restarted, err := NewManager(NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), io.Discard}), victim.peerServiceOptions().BaseConfigDir, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { restarted.reverse.manager.Close() })
	assertFullyUnclustered(t, restarted)
	if _, epoch := restarted.currentAdmission(); epoch != 0 {
		t.Fatalf("restart resurrected admission %d", epoch)
	}
	if _, ok := restarted.resolvePeerServiceRoute(victim.identity.NodeUUID, noderec.ServiceCluster); ok {
		t.Fatal("unclustered restart retained a route")
	}
}
