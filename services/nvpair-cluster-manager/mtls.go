// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
	"nvpair-shared/splitlisten"
)

// membersRemovePath is the mTLS trusted endpoint for peer-to-peer removal (§7.2).
const membersRemovePath = "/v1/cluster/members/remove"

type membersRemoveRequest struct {
	NodeUUID string       `json:"nodeUuid"`
	Proof    RemovalProof `json:"proof"`
}

// buildServerTLSConfig presents this node's leaf and requires a client cert
// (RequireAnyClientCert); the per-request pin match in the handler turns a
// non-member away with a real HTTP 403 rather than an opaque handshake failure
// (§7.2). The config is built by nvpair-shared/clustertrust so cluster-manager,
// nvpair-errors, and nvpair-workload-manager apply the same pinned-peer mTLS policy.
func (m *Manager) buildServerTLSConfig() *tls.Config {
	return clustertrust.ServerTLSConfig(m.identity.Cert)
}

// peerClient reuses roster connections under current direct/reverse authority.
func (m *Manager) peerClient(peerUUID string) (*http.Client, error) {
	return m.peerServiceClient(peerUUID, m.peerServices)
}

// removalPeerClient keeps removal notifications on fresh connections; a lost
// reply belongs to the removal owner and must never cause transport replay.
func (m *Manager) removalPeerClient(peerUUID string) (*http.Client, error) {
	return m.peerServiceClient(peerUUID, m.removalPeerServices)
}

func (m *Manager) peerServiceClient(peerUUID string, adapter *clustertrust.PeerServiceClient) (*http.Client, error) {
	if m.teardownPending.Load() {
		return nil, fmt.Errorf("cluster teardown is pending")
	}
	m.mesh.Refresh()
	_, ok := m.resolvePeerServiceRoute(peerUUID, noderec.ServiceCluster)
	if !ok {
		return nil, fmt.Errorf("no current cluster route for peer %s", peerUUID)
	}
	return adapter.Client(peerUUID, noderec.ServiceCluster), nil
}

func (m *Manager) buildPairingClientTLSConfig(peerDER []byte) (*tls.Config, error) {
	if m.teardownPending.Load() {
		return nil, fmt.Errorf("cluster teardown is pending")
	}
	if len(peerDER) == 0 {
		return nil, fmt.Errorf("pairing peer certificate is missing")
	}
	return clustertrust.ClientTLSConfig(m.identity.Cert, peerDER), nil
}

// verifyClientPin reads the authenticated client cert, requires that its claimed
// UUID is pinned byte-for-byte, and returns that UUID. Used to gate trusted
// endpoints (§7.2). The cert read + UUID extraction + match flow live in
// nvpair-shared/clustertrust; the pin set and the rejection log stay local.
func (m *Manager) verifyClientPin(r *http.Request) (string, bool) {
	if m.teardownPending.Load() {
		return "", false
	}
	return clustertrust.VerifyClientPin(r, func(uuid string, der []byte) bool {
		if m.trust.MatchDER(uuid, der) {
			return true
		}
		log.Printf("mTLS: rejected client %s (%s): not a pinned peer", uuid, certFingerprintFromDER(der))
		return false
	})
}

// handleMembersRemove is the inbound half of nodes:remove: a pinned peer is
// telling us we have been removed from the cluster. Authenticate the caller,
// then tear down *all* local cluster state (not merely the sender's pin) so
// this node becomes unclustered — matching §4 "being removed leaves the node
// clusterless." The body's nodeUuid must match the client cert's UUID (§7.2).
// Self-initiated leave uses cluster:leave (roster tombstone) instead.
func (m *Manager) handleMembersRemove(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	// Authenticate and snapshot the exact composition before the potentially
	// blocking body read. The same values are revalidated immediately before
	// teardown so this request cannot cross a leave/rejoin boundary.
	m.rosterMu.Lock()
	uuid, ok := m.verifyClientPin(r)
	var clientDER []byte
	if ok && r.TLS != nil && len(r.TLS.PeerCertificates) > 0 {
		clientDER = append([]byte(nil), r.TLS.PeerCertificates[0].Raw...)
	}
	cid0, epoch0 := m.currentAdmission()
	gen0 := m.clusterGen.Load()
	m.rosterMu.Unlock()
	if !ok {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes))
	if err != nil {
		http.Error(w, "read body", http.StatusBadRequest)
		return
	}
	var req membersRemoveRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if req.NodeUUID != "" && req.NodeUUID != uuid {
		http.Error(w, "nodeUuid does not match client identity", http.StatusBadRequest)
		return
	}
	m.rosterMu.Lock()
	cid, epoch := m.currentAdmission()
	stale := cid0 == "" || epoch0 == 0 || cid != cid0 || epoch != epoch0 ||
		m.clusterGen.Load() != gen0 || !m.trust.MatchDER(uuid, clientDER)
	if stale {
		m.rosterMu.Unlock()
		http.Error(w, "stale removal request", http.StatusConflict)
		return
	}
	if req.Proof.Tombstone.NodeUUID != m.identity.NodeUUID ||
		req.Proof.Tombstone.AdmissionEpoch != epoch0 || req.Proof.Tombstone.By != uuid ||
		!m.verifyRemovalProof(req.Proof, cid0) {
		m.rosterMu.Unlock()
		http.Error(w, "invalid or stale removal proof", http.StatusForbidden)
		return
	}
	if err := m.teardownClusterLocalLocked(); err != nil {
		m.rosterMu.Unlock()
		http.Error(w, "durable teardown incomplete", http.StatusInternalServerError)
		log.Printf("mTLS: peer %s removal teardown incomplete: %v", uuid, err)
		return
	}
	m.rosterMu.Unlock()
	m.emitIdentityChanged()
	m.emitNodesChanged()
	log.Printf("mTLS: peer %s removed us; cleared local cluster state", uuid)
	w.WriteHeader(http.StatusOK)
}

// notifyPeerRemoval best-effort tells a peer it has been removed, over mTLS,
// while the peer's pin still exists (call before deleting it locally).
func (m *Manager) notifyPeerRemoval(ctx context.Context, addr, peerUUID string, proof RemovalProof) {
	succeeded := false
	defer func() {
		if !succeeded {
			m.peerAddrs.Forget(peerUUID)
		}
	}()
	if !m.selectPeerServiceAddress(peerUUID, addr) {
		log.Printf("notify removal: no current endpoint for peer %s", peerUUID)
		return
	}
	client, err := m.removalPeerClient(peerUUID)
	if err != nil {
		log.Printf("notify removal: %v", err)
		return
	}
	body, err := json.Marshal(membersRemoveRequest{NodeUUID: m.identity.NodeUUID, Proof: proof})
	if err != nil {
		log.Printf("notify removal: encode request: %v", err)
		return
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://"+addr+membersRemovePath, bytes.NewReader(body))
	if err != nil {
		log.Printf("notify removal: create request: %v", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("notify removal to %s (%s): %v", addr, peerUUID, err)
		return
	}
	_, readErr := io.Copy(io.Discard, io.LimitReader(resp.Body, maxBodyBytes))
	closeErr := resp.Body.Close()
	succeeded = resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices && readErr == nil && closeErr == nil
	if !succeeded {
		log.Printf("notify removal to %s (%s): status=%d read=%v close=%v", addr, peerUUID, resp.StatusCode, readErr, closeErr)
	}
}

// serveSplit serves both transport modes on one listener via the shared
// first-byte dispatcher (nvpair-shared/splitlisten): a TLS record (0x16) is
// terminated by the mTLS server, an ASCII HTTP method byte by the plain pairing
// server. The dispatch/prefix-restore/chan-listener plumbing lives in the shared
// package now so the promoted proxies reuse the exact same split.
func (m *Manager) serveSplit(ctx context.Context, ln net.Listener, plainMux, tlsMux http.Handler) error {
	split := splitlisten.New(reverseForwardListener{Listener: ln, manager: m})

	plainSrv := &http.Server{Handler: plainMux, ConnContext: m.reverseConnContext, ReadHeaderTimeout: 5 * time.Second, IdleTimeout: clustertrust.PeerListenerIdleTimeout}
	tlsSrv := &http.Server{Handler: tlsMux, ConnContext: m.reverseConnContext, ReadHeaderTimeout: 5 * time.Second, IdleTimeout: clustertrust.PeerListenerIdleTimeout}

	go func() { _ = plainSrv.Serve(split.Plain()) }()
	go func() { _ = tlsSrv.Serve(tls.NewListener(split.TLS(), m.buildServerTLSConfig())) }()

	<-ctx.Done()
	sc, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_ = plainSrv.Shutdown(sc)
	_ = tlsSrv.Shutdown(sc)
	_ = split.Close()
	return nil
}
