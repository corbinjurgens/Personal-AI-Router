// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/noderec"
	"nvpair-shared/reach"
)

type removalHintFixture struct {
	a, b                   *Manager
	guard                  *peerDialGuard
	listener               *ownedPeerListener
	lan, tail              string
	descriptor             noderec.PeerServicesDescriptor
	reads, posts, accepted atomic.Int32
	reply                  string
	output                 bytes.Buffer
}

func newRemovalHintFixture(t *testing.T) *removalHintFixture {
	t.Helper()
	f := &removalHintFixture{a: newReverseTestManager(t), b: newReverseTestManager(t)}
	f.descriptor = noderec.PeerServicesDescriptor{HostUUID: f.b.identity.NodeUUID, Name: "removal-peer", Addresses: []string{"192.0.2.55", "100.90.0.8"}, Services: []noderec.PeerService{}}
	f.listener = startWalkServer(t, f.b, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case peerServicesPath:
			f.reads.Add(1)
			if err := json.NewEncoder(w).Encode(f.descriptor); err != nil {
				t.Error(err)
			}
		case membersRemovePath:
			f.posts.Add(1)
			if r.Method != http.MethodPost {
				t.Error("removal was not POST")
			}
			rr := httptest.NewRecorder()
			f.b.handleMembersRemove(rr, r)
			if rr.Code == http.StatusOK {
				f.accepted.Add(1)
			}
			if f.reply == "redirect" {
				w.Header().Set("Location", "https://"+f.tail+"/unexpected-redirect")
				w.WriteHeader(http.StatusTemporaryRedirect)
				return
			}
			if f.reply == "lost" {
				conn, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Error(err)
					return
				}
				if err := conn.Close(); err != nil {
					t.Error(err)
				}
				return
			}
			if f.reply == "partial" {
				w.Header().Set("Content-Length", "100")
				w.WriteHeader(rr.Code)
				if _, err := w.Write([]byte("{")); err != nil {
					t.Error(err)
				}
				w.(http.Flusher).Flush()
				return
			}
			w.WriteHeader(rr.Code)
			if _, err := w.Write(rr.Body.Bytes()); err != nil {
				t.Error(err)
			}
		default:
			t.Errorf("unexpected removal-stage path %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	f.b.port = f.listener.Addr().(*net.TCPAddr).Port
	pairReverseManagers(t, f.a, f.b)
	f.lan = net.JoinHostPort("192.0.2.55", strconv.Itoa(f.b.port))
	f.tail = net.JoinHostPort("100.90.0.8", strconv.Itoa(f.b.port))
	f.a.setMemberAddr(f.b.identity.NodeUUID, "192.0.2.55", f.b.port)
	persistPeerServicesMembers(t, f.a, f.b)
	learn := newPeerDialGuard(t)
	learn.allow(t, f.lan, f.listener)
	replaceClusterPeerServices(t, f.a, f.a.peerServiceOptions(), learn)
	replaceClusterPeerServices(t, f.b, f.b.peerServiceOptions(), newPeerDialGuard(t))
	if err := f.a.refreshPeerServices(context.Background(), f.b.identity.NodeUUID); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(storedHintMember(t, f.a, f.b.identity.NodeUUID).AdvertisedAddresses, f.descriptor.Addresses) {
		t.Fatal("setup did not authenticate and persist hints")
	}
	f.a.browser = newBrowser()
	f.guard = newPeerDialGuard(t)
	f.guard.allow(t, f.tail, f.listener)
	replaceClusterPeerServices(t, f.a, f.a.peerServiceOptions(), f.guard)
	f.a.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &f.output})
	return f
}

func (f *removalHintFixture) remove(t *testing.T) *Message {
	t.Helper()
	uuid := f.b.identity.NodeUUID
	id := json.RawMessage(`1`)
	f.a.handleNodesRemove(&Message{ID: &id, Params: mustJSON(t, removeParams{NodeUUID: &uuid})})
	decoder := json.NewDecoder(&f.output)
	var response *Message
	for decoder.More() {
		var m Message
		if err := decoder.Decode(&m); err != nil {
			t.Fatal(err)
		}
		if m.ID != nil && string(*m.ID) == "1" {
			if response != nil {
				t.Fatal("duplicate removal response")
			}
			response = &m
		}
	}
	if response == nil {
		t.Fatal("missing removal response")
	}
	return response
}

func assertHintRemovalCommitted(t *testing.T, f *removalHintFixture, response *Message) {
	t.Helper()
	var result struct {
		Removed bool `json:"removed"`
	}
	if response.Error != nil {
		t.Fatalf("removal error: %+v", response.Error)
	}
	if err := json.Unmarshal(response.Result, &result); err != nil {
		t.Fatal(err)
	}
	if !result.Removed {
		t.Fatal("removal did not commit")
	}
	if _, ok := f.a.memberByNodeID(f.b.identity.NodeUUID); ok {
		t.Fatal("removed member/hints retained")
	}
	if _, ok := f.a.trust.Get(f.b.identity.NodeUUID); ok {
		t.Fatal("removed pin retained")
	}
	proofs := f.a.snapshotRemovalProofs()
	if len(proofs) != 1 || proofs[0].Tombstone.NodeUUID != f.b.identity.NodeUUID || proofs[0].Tombstone.AdmissionEpoch != 1 {
		t.Fatalf("durable proof=%+v", proofs)
	}
	reopened := testManagerAt(t, filepath.Dir(f.a.clusterDir), 0)
	t.Cleanup(reopened.peerServices.CloseIdle)
	t.Cleanup(reopened.removalPeerServices.CloseIdle)
	t.Cleanup(func() {
		if err := reopened.reverse.manager.Close(); err != nil {
			t.Error(err)
		}
	})
	if _, ok := reopened.memberByNodeID(f.b.identity.NodeUUID); ok {
		t.Fatal("original-base reopen resurrected member/hints")
	}
	if _, ok := reopened.trust.Get(f.b.identity.NodeUUID); ok {
		t.Fatal("original-base reopen resurrected pin")
	}
	if len(reopened.snapshotRemovalProofs()) != 1 {
		t.Fatal("original-base reopen lost convergence proof")
	}
}

func TestNodesRemoveSelectsPersistedHint(t *testing.T) {
	t.Run("current hint authenticates one mutation", func(t *testing.T) {
		f := newRemovalHintFixture(t)
		response := f.remove(t)
		if f.posts.Load() != 1 || f.accepted.Load() != 1 || f.reads.Load() != 1 || f.guard.reverse.Load() != 0 || f.guard.attempted(f.tail) != 1 {
			t.Fatalf("posts=%d accepted=%d descriptorReads=%d reverse=%d tail=%d", f.posts.Load(), f.accepted.Load(), f.reads.Load(), f.guard.reverse.Load(), f.guard.attempted(f.tail))
		}
		f.guard.mu.Lock()
		probes := f.guard.probes[f.tail]
		f.guard.mu.Unlock()
		if probes == 0 || f.guard.physical.Load() != 2 {
			t.Fatalf("removal chooser=%d physical=%d, want positive/2", probes, f.guard.physical.Load())
		}
		assertHintRemovalCommitted(t, f, response)
		assertRemovalDurableAfterRestart(t, f.b)
	})
	t.Run("unavailable selection keeps best effort proof", func(t *testing.T) {
		f := newRemovalHintFixture(t)
		f.guard.targets = make(map[string]string)
		response := f.remove(t)
		if f.posts.Load() != 0 || f.reads.Load() != 1 || f.guard.physical.Load() != 0 {
			t.Fatal("unavailable selection dispatched a remote mutation")
		}
		assertHintRemovalCommitted(t, f, response)
	})
	t.Run("empty candidates keep best effort proof", func(t *testing.T) {
		f := newRemovalHintFixture(t)
		node, _ := f.a.memberByNodeID(f.b.identity.NodeUUID)
		node.IPAddress = ""
		node.AdvertisedAddresses = nil
		f.a.upsertMember(node)
		if err := f.a.persistMembersErr(); err != nil {
			t.Fatal(err)
		}
		response := f.remove(t)
		if f.posts.Load() != 0 || f.guard.physical.Load() != 0 || f.guard.reverse.Load() != 0 {
			t.Fatal("empty selection dispatched work")
		}
		assertHintRemovalCommitted(t, f, response)
	})
	t.Run("pin delete failure preserves local member and hints", func(t *testing.T) {
		f := newRemovalHintFixture(t)
		original := f.a.trust.dir
		blocker := t.TempDir()
		pinDirectory := filepath.Join(blocker, f.b.identity.NodeUUID+".json")
		if err := os.Mkdir(pinDirectory, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(pinDirectory, "held"), []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
		f.a.trust.dir = blocker
		defer func() { f.a.trust.dir = original }()
		response := f.remove(t)
		if response.Error == nil {
			t.Fatal("pin deletion failure was hidden")
		}
		if f.posts.Load() != 1 || f.accepted.Load() != 1 {
			t.Fatal("notification did not precede de-pin")
		}
		if pin, ok := f.a.trust.Get(f.b.identity.NodeUUID); !ok || pin.AdmissionEpoch != 1 {
			t.Fatal("failed de-pin changed pin")
		}
		if !slices.Equal(storedHintMember(t, f.a, f.b.identity.NodeUUID).AdvertisedAddresses, f.descriptor.Addresses) {
			t.Fatal("failed de-pin discarded member hints")
		}
		if len(f.a.snapshotRemovalProofs()) != 1 {
			t.Fatal("failed de-pin lost proof")
		}
	})
}

func TestNodesRemoveSelectionRejectsReadmission(t *testing.T) {
	test := func(name string, mutate func(*removalHintFixture) error) {
		t.Run(name, func(t *testing.T) {
			f := newRemovalHintFixture(t)
			var once sync.Once
			var changed atomic.Bool
			type mutationState struct {
				err    error
				member *ClusterNode
				pin    *TrustedPin
				cid    string
				epoch  uint64
			}
			mutationResult := make(chan mutationState, 1)
			f.a.peerAddrs = reach.NewChooser(reach.WithDial(func(network, address string, timeout time.Duration) (net.Conn, error) {
				once.Do(func() {
					f.a.rosterMu.Lock()
					err := mutate(f)
					member, _ := f.a.memberByNodeID(f.b.identity.NodeUUID)
					pin, _ := f.a.trust.Get(f.b.identity.NodeUUID)
					cid, epoch := f.a.currentAdmission()
					f.a.rosterMu.Unlock()
					mutationResult <- mutationState{err, member, pin, cid, epoch}
					changed.Store(true)
				})
				return f.guard.probe(network, address, timeout)
			}))
			response := f.remove(t)
			if !changed.Load() {
				t.Fatal("removal never performed endpoint selection")
			}
			want := <-mutationResult
			if want.err != nil {
				t.Fatal(want.err)
			}
			if response.Error == nil || f.posts.Load() != 0 || f.reads.Load() != 1 || f.guard.attempted(f.tail) != 0 || f.guard.reverse.Load() != 0 {
				t.Fatalf("stale selection error=%v posts=%d reads=%d tail=%d reverse=%d", response.Error, f.posts.Load(), f.reads.Load(), f.guard.attempted(f.tail), f.guard.reverse.Load())
			}
			current, ok := f.a.memberByNodeID(f.b.identity.NodeUUID)
			if !ok || current.AdmissionEpoch != want.member.AdmissionEpoch || current.IPAddress != want.member.IPAddress || !slices.Equal(current.AdvertisedAddresses, want.member.AdvertisedAddresses) {
				t.Fatal("stale selection removed replacement member")
			}
			pin, pinned := f.a.trust.Get(f.b.identity.NodeUUID)
			if pinned != (want.pin != nil) || (pinned && (pin.CertFingerprint != want.pin.CertFingerprint || pin.AdmissionEpoch != want.pin.AdmissionEpoch)) {
				t.Fatal("stale selection changed replacement pin")
			}
			if cid, epoch := f.a.currentAdmission(); cid != want.cid || epoch != want.epoch {
				t.Fatal("stale selection changed replacement local admission")
			}
			if cid, epoch := f.b.currentAdmission(); cid != "cluster-1" || epoch != 1 {
				t.Fatal("stale selection changed remote authority")
			}
		})
	}
	test("peer readmitted same certificate", func(f *removalHintFixture) error {
		if err := f.a.trust.Remove(f.b.identity.NodeUUID); err != nil {
			return err
		}
		if err := f.a.trust.Pin(&TrustedPin{NodeUUID: f.b.identity.NodeUUID, NodeID: f.b.identity.NodeID, ClusterID: "cluster-1", AdmissionEpoch: 2, CertPem: string(f.b.identity.CertPEM), CertFingerprint: f.b.identity.CertFingerprint}); err != nil {
			return err
		}
		node, _ := f.a.memberByNodeID(f.b.identity.NodeUUID)
		node.AdmissionEpoch = 2
		f.a.upsertMember(node)
		return f.a.persistMembersErr()
	})
	test("local teardown and rejoin", func(f *removalHintFixture) error {
		node, _ := f.a.memberByNodeID(f.b.identity.NodeUUID)
		if err := f.a.teardownClusterLocalLocked(); err != nil {
			return err
		}
		epoch, err := f.a.reserveAdmissionEpoch()
		if err != nil {
			return err
		}
		if err = f.a.activateAdmission("cluster-1", epoch); err != nil {
			return err
		}
		f.a.setClusterIdentity("cluster-1", "rejoined")
		f.a.addSelfMember()
		if err := f.a.trust.Pin(&TrustedPin{NodeUUID: f.b.identity.NodeUUID, NodeID: f.b.identity.NodeID, ClusterID: "cluster-1", AdmissionEpoch: 1, CertPem: string(f.b.identity.CertPEM), CertFingerprint: f.b.identity.CertFingerprint}); err != nil {
			return err
		}
		f.a.upsertMember(node)
		return f.a.persistMembersErr()
	})
	test("pin revoked during selection", func(f *removalHintFixture) error { f.a.trust.Forget(f.b.identity.NodeUUID); return nil })
	test("selected hint withdrawn", func(f *removalHintFixture) error {
		node, _ := f.a.memberByNodeID(f.b.identity.NodeUUID)
		node.IPAddress = "192.0.2.99"
		node.AdvertisedAddresses = nil
		f.a.upsertMember(node)
		return f.a.persistMembersErr()
	})
	beforeWrite := func(name string, mutate func(*testing.T, *removalHintFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newRemovalHintFixture(t)
			proof, err := f.a.newRemovalProof(f.b.identity.NodeUUID, 1)
			if err != nil {
				t.Fatal(err)
			}
			var handoff atomic.Int32
			ctx := httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{GotConn: func(httptrace.GotConnInfo) { handoff.Add(1); mutate(t, f) }})
			f.a.notifyPeerRemoval(ctx, f.tail, f.b.identity.NodeUUID, proof)
			if handoff.Load() != 1 || f.posts.Load() != 0 || f.reads.Load() != 1 || f.guard.reverse.Load() != 0 {
				t.Fatalf("before-write handoff=%d posts=%d reads=%d reverse=%d", handoff.Load(), f.posts.Load(), f.reads.Load(), f.guard.reverse.Load())
			}
			if _, ok := f.a.memberByNodeID(f.b.identity.NodeUUID); !ok {
				t.Fatal("before-write rejection lost replacement member")
			}
		})
	}
	beforeWrite("peer readmission before first application write", func(t *testing.T, f *removalHintFixture) {
		readmitClusterPeer(t, f.a, f.b)
	})
	beforeWrite("pin revoked before first application write", func(t *testing.T, f *removalHintFixture) { f.a.trust.Forget(f.b.identity.NodeUUID) })
	beforeWrite("hint withdrawn before first application write", func(t *testing.T, f *removalHintFixture) {
		node, _ := f.a.memberByNodeID(f.b.identity.NodeUUID)
		node.AdvertisedAddresses = nil
		node.IPAddress = "192.0.2.99"
		f.a.upsertMember(node)
	})
}

func TestNodesRemoveWithdrawnEndpointPreservesAdmissionAfterRestart(t *testing.T) {
	f := newRemovalHintFixture(t)
	uuid := f.b.identity.NodeUUID
	pin, pinned := f.a.trust.Get(uuid)
	if !pinned {
		t.Fatal("fixture lacks the target pin")
	}
	cid, epoch := f.a.currentAdmission()
	var once sync.Once
	changed := make(chan error, 1)
	f.a.peerAddrs = reach.NewChooser(reach.WithDial(func(network, address string, timeout time.Duration) (net.Conn, error) {
		once.Do(func() {
			f.a.rosterMu.Lock()
			member, ok := f.a.memberByNodeID(uuid)
			if !ok {
				f.a.rosterMu.Unlock()
				changed <- errors.New("target disappeared before endpoint withdrawal")
				return
			}
			member.IPAddress = "192.0.2.99"
			member.AdvertisedAddresses = nil
			f.a.upsertMember(member)
			err := f.a.persistMembersErr()
			f.a.rosterMu.Unlock()
			changed <- err
		})
		return f.guard.probe(network, address, timeout)
	}))
	response := f.remove(t)
	select {
	case err := <-changed:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("removal did not reach endpoint selection")
	}
	if response.Error == nil || response.Error.Code != codeInternalError || f.posts.Load() != 0 || f.reads.Load() != 1 || f.guard.reverse.Load() != 0 {
		t.Fatalf("withdrawn endpoint error=%v posts=%d reads=%d reverse=%d", response.Error, f.posts.Load(), f.reads.Load(), f.guard.reverse.Load())
	}
	assertPreserved := func(m *Manager) {
		t.Helper()
		var output bytes.Buffer
		m.codec = NewCodec(struct {
			io.Reader
			io.Writer
		}{bytes.NewReader(nil), &output})
		id := json.RawMessage(`2`)
		m.handleGetInitial(&Message{ID: &id})
		var reply Message
		if err := json.NewDecoder(&output).Decode(&reply); err != nil {
			t.Fatal(err)
		}
		if reply.Error != nil || reply.ID == nil || string(*reply.ID) != "2" {
			t.Fatalf("membership response=%+v", reply)
		}
		var result struct {
			Nodes []ClusterNode `json:"nodes"`
		}
		if err := json.Unmarshal(reply.Result, &result); err != nil {
			t.Fatal(err)
		}
		index := slices.IndexFunc(result.Nodes, func(n ClusterNode) bool { return n.NodeUUID == uuid })
		if index < 0 {
			t.Error("preservation error lost the target from public membership")
		} else if member := result.Nodes[index]; member.ClusterID != cid || member.AdmissionEpoch != pin.AdmissionEpoch || member.IPAddress != "192.0.2.99" || len(member.AdvertisedAddresses) != 0 {
			t.Errorf("preserved public member=%+v", member)
		}
		current, ok := m.trust.Get(uuid)
		if !ok || current.CertFingerprint != pin.CertFingerprint || current.AdmissionEpoch != pin.AdmissionEpoch {
			t.Errorf("preservation error changed the target pin: %+v", current)
		}
		if currentCID, currentEpoch := m.currentAdmission(); currentCID != cid || currentEpoch != epoch {
			t.Errorf("preservation error changed local admission: %s/%d", currentCID, currentEpoch)
		}
		roster := m.buildLocalRoster()
		if len(roster.RemovalProofs) != 0 || len(roster.Tombstones) != 0 {
			t.Errorf("aborted selection published removal authority: proofs=%d tombstones=%d", len(roster.RemovalProofs), len(roster.Tombstones))
		}
	}
	assertPreserved(f.a)
	f.a.peerServices.CloseIdle()
	f.a.removalPeerServices.CloseIdle()
	reopened := testManagerAt(t, filepath.Dir(f.a.clusterDir), 0)
	containHintDials(t, reopened, false)
	t.Cleanup(reopened.peerServices.CloseIdle)
	t.Cleanup(reopened.removalPeerServices.CloseIdle)
	t.Cleanup(func() {
		if err := reopened.reverse.manager.Close(); err != nil {
			t.Error(err)
		}
	})
	if reopened.identity.NodeUUID != f.a.identity.NodeUUID || reopened.identity.CertFingerprint != f.a.identity.CertFingerprint {
		t.Fatal("original-base reopen changed local identity")
	}
	assertPreserved(reopened)
	if remoteCID, remoteEpoch := f.b.currentAdmission(); remoteCID != cid || remoteEpoch != pin.AdmissionEpoch {
		t.Fatalf("zero-dispatch abort changed remote admission: %s/%d", remoteCID, remoteEpoch)
	}
}

func TestNodesRemoveHintLostReplyDoesNotReplay(t *testing.T) {
	for _, reply := range []string{"lost", "partial", "redirect"} {
		t.Run(reply, func(t *testing.T) {
			f := newRemovalHintFixture(t)
			f.reply = reply
			response := f.remove(t)
			if f.posts.Load() != 1 || f.accepted.Load() != 1 || f.reads.Load() != 1 || f.guard.reverse.Load() != 0 || f.guard.attempted(f.tail) != 1 {
				t.Fatalf("lost reply posts=%d accepted=%d reads=%d reverse=%d tail=%d", f.posts.Load(), f.accepted.Load(), f.reads.Load(), f.guard.reverse.Load(), f.guard.attempted(f.tail))
			}
			assertHintRemovalCommitted(t, f, response)
			assertRemovalDurableAfterRestart(t, f.b)
		})
	}
}

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
	containHintDials(t, victim, false)
	containHintDials(t, remover, false)
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
	reverse := countClusterReverseDials(t, remover)
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
	reverse := countClusterReverseDials(t, remover)
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
