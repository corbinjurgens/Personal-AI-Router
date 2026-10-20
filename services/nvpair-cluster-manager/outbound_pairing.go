// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	"eapnoob"
)

type rejectingPairingListener struct{ net.Listener }

func (ln rejectingPairingListener) Accept() (net.Conn, error) {
	for {
		conn, err := ln.Listener.Accept()
		if err != nil {
			return nil, err
		}
		_ = conn.Close()
	}
}

// commitJoinerPairing is the one join-transaction owner for both completion
// transports. The caller holds sess.mu; composition eligibility is checked
// again inside the durable transaction, after all network waits have finished.
func (m *Manager) commitJoinerPairing(sess *pairingSession, inv *Invite) (commitErr error) {
	pi := sess.peerPairing
	now := time.Now().UnixMilli()
	m.withClusterComposition(func() {
		cid, _ := m.clusterIdentity()
		admissionCID, epoch := m.currentAdmission()
		current, live := m.getSession(sess.inviteID)
		pending, exists := m.getInvite(sess.inviteID)
		if m.teardownPending.Load() || cid != "" || admissionCID != "" || epoch != 0 ||
			!live || current != sess || !exists || pending.State != inviteStatePending ||
			pending.ClusterID != inv.ClusterID || pi == nil || sess.localAdmissionEpoch == 0 {
			commitErr = errPairingCommitStale
			return
		}
		if m.removalProofBlocksAdmission(pi.NodeUUID, inv.ClusterID, pi.AdmissionEpoch) {
			commitErr = fmt.Errorf("peer admission %d was already removed", pi.AdmissionEpoch)
			return
		}
		if self, ok := m.memberByNodeID(m.identity.NodeUUID); ok && self.AdmissionEpoch != 0 {
			commitErr = fmt.Errorf("another admission transaction is incomplete")
			return
		}
		// Both completion transports persist the same authenticated receipt
		// before the first provisional membership/pin mutation.
		receipt, stageErr := m.stagePairingReceiptLocked(sess, inv)
		if commitErr = stageErr; commitErr != nil {
			return
		}
		admitted := *pi
		admitted.ClusterID = inv.ClusterID
		end := m.endorsePeerForAdmission(admitted.NodeUUID,
			certFingerprintFromDER(sess.peerCert.Raw), inv.ClusterID,
			admitted.AdmissionEpoch, sess.localAdmissionEpoch)
		rollback := func(cause error) error {
			rollbackErr := m.rollbackPairingSessionCompositionLocked(sess, inv)
			if rollbackErr != nil {
				return fmt.Errorf("%w; compensation pending: %v", cause, rollbackErr)
			}
			return cause
		}
		m.recordMember(&admitted, stateMember, &now)
		m.addSelfMemberForAdmission(inv.ClusterID, sess.localAdmissionEpoch)
		if commitErr = m.persistMembersErr(); commitErr != nil {
			commitErr = rollback(commitErr)
			return
		}
		if commitErr = m.pinPeer(&admitted, sess.peerCert, []Endorsement{end}); commitErr != nil {
			commitErr = rollback(commitErr)
			return
		}
		if commitErr = m.activateAdmission(inv.ClusterID, sess.localAdmissionEpoch); commitErr != nil {
			commitErr = rollback(commitErr)
			return
		}
		m.setClusterIdentity(inv.ClusterID, inv.ClusterFriendlyName)
		if commitErr = m.completePairingReceiptLocked(receipt); commitErr != nil {
			commitErr = rollback(commitErr)
			return
		}
		if err := m.clearRemovalProofAfterAdmission(admitted.NodeUUID, admitted.AdmissionEpoch); err != nil {
			log.Printf("pairing %s: clear removal proof: %v", sess.inviteID, err)
		}
	})
	return
}

func respondPairingEnvelope(w http.ResponseWriter, env pairingEnvelope) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(env); err != nil {
		log.Printf("pairing response: %v", err)
	}
}

func (m *Manager) clearPendingPairingMember(peerUUID string) {
	m.withClusterComposition(func() {
		if member, ok := m.memberByNodeID(peerUUID); ok && member.State == statePendingInbound {
			m.removeMemberByUUID(peerUUID)
		}
	})
}

// Ready exposes only pre-commit invite state; it never authorizes rollback of
// a committed peer. No certificate authority follows from knowing an invite ID.
func (m *Manager) handlePairingReady(w http.ResponseWriter, env *pairingEnvelope) {
	inv, exists := m.getInvite(env.InviteID)
	if !exists || inv.FromNodeUUID == m.identity.NodeUUID {
		http.Error(w, "unknown invite", http.StatusConflict)
		return
	}
	ready := false
	if sess, live := m.getSession(env.InviteID); live {
		sess.mu.Lock()
		if sess.role != roleJoiner || sess.completionMode != inviterCompletionV1 {
			sess.mu.Unlock()
			http.Error(w, "completion mode cannot change", http.StatusConflict)
			return
		}
		ready = sess.ready
		sess.mu.Unlock()
	}
	respondPairingEnvelope(w, pairingEnvelope{InviteID: env.InviteID, State: inv.State, Reason: inv.Reason, Ready: ready})
}

func (m *Manager) handleOutboundCompletion(w http.ResponseWriter, env *pairingEnvelope, msg []byte) {
	sess, exists := m.getSession(env.InviteID)
	if !exists || sess.role != roleJoiner {
		http.Error(w, "unknown invite", http.StatusConflict)
		return
	}
	sess.mu.Lock()
	defer sess.mu.Unlock()
	if sess.completionMode != inviterCompletionV1 || !sess.ready || len(msg) == 0 {
		http.Error(w, "completion mode or readiness mismatch", http.StatusConflict)
		return
	}
	if bytes.Equal(msg, sess.cachedRequest) {
		respondPairing(w, sess.cachedResponse)
		return
	}
	var wire struct {
		Type      *int   `json:"Type"`
		EAP       string `json:"eap"`
		ErrorCode *int   `json:"ErrorCode"`
	}
	if err := json.Unmarshal(msg, &wire); err != nil {
		http.Error(w, "invalid EAP message", http.StatusBadRequest)
		return
	}
	// Do not let a new negotiation change the transcript/mode, or let an
	// unauthenticated EAP-Success bypass the Completion MAC verification.
	expectedTypes := []int{1, 5, 6}
	valid := sess.completionStep < len(expectedTypes) && wire.Type != nil &&
		*wire.Type == expectedTypes[sess.completionStep] && wire.EAP == ""
	if sess.completionStep == len(expectedTypes) {
		valid = wire.Type == nil && wire.EAP == "success"
	}
	if wire.Type != nil && *wire.Type == 0 && wire.ErrorCode != nil && sess.completionStep > 0 {
		valid = true
	}
	if !valid {
		http.Error(w, "unexpected completion phase", http.StatusConflict)
		return
	}
	inv, exists := m.getInvite(env.InviteID)
	if !exists || inv.State != inviteStatePending {
		http.Error(w, "invite no longer pending", http.StatusConflict)
		return
	}
	out, err := sess.peer.Receive(msg)
	if err != nil {
		http.Error(w, "invalid completion", http.StatusBadRequest)
		return
	}
	if !out.Done {
		sess.completionStep++
	}
	if out.Done && !out.Success {
		reason := ""
		if out.Err != nil && (out.Err.Code == eapnoob.ErrHMACVerificationFailed || out.Err.Code == eapnoob.ErrUnrecognizedOOBMsgID) {
			reason = reasonIncorrectPIN
		}
		m.finishInviteReason(env.InviteID, inviteStateFailed, reason)
		m.clearPendingPairingMember(inv.FromNodeUUID)
		m.emitNodesChanged()
	}
	if out.Done && out.Success {
		// Persist only after the completed EAP authenticates the transcript. A
		// receipt confers authority solely to confirm this exact invite, even if
		// its membership/pin transaction subsequently fails.
		err = m.commitJoinerPairing(sess, inv)
		state := inviteStatePaired
		if err != nil {
			state = inviteStateFailed
			m.clearPendingPairingMember(inv.FromNodeUUID)
			log.Printf("pairing %s: join commit failed: %v", env.InviteID, err)
		}
		if !sess.rollbackPending {
			m.finishInvite(env.InviteID, state)
		} else {
			state = inviteStatePending
		}
		sess.createdAt = time.Now().UnixMilli()
		m.emitNodesChanged()
		if state == inviteStatePaired {
			m.emitIdentityChanged()
		}
	}
	sess.cachedRequest = append([]byte(nil), msg...)
	sess.cachedResponse = append([]byte(nil), out.Send...)
	if out.Done && out.Success && !sess.droppedFinalReply && os.Getenv("NVPAIR_TEST_PAIR_DROP_FINAL_REPLY") == "1" {
		sess.droppedFinalReply = true
		dropPairingReply(w)
		return
	}
	respondPairing(w, out.Send)
}

func dropPairingReply(w http.ResponseWriter) {
	if hijacker, ok := w.(http.Hijacker); ok {
		conn, _, err := hijacker.Hijack()
		if err == nil {
			_ = conn.Close()
			return
		}
	}
	http.Error(w, "isolated test dropped pairing reply", http.StatusServiceUnavailable)
}

func exchangePairingEnvelope(client *http.Client, scheme, addr string, env pairingEnvelope) (pairingEnvelope, error) {
	var out pairingEnvelope
	body, err := json.Marshal(env)
	if err != nil {
		return out, err
	}
	resp, err := client.Post(scheme+"://"+addr+pairingPath, "application/json", bytes.NewReader(body))
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return out, fmt.Errorf("pairing %s status %d", env.Phase, resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	if err != nil {
		return out, err
	}
	err = decodePairingEnvelope(data, &out)
	return out, err
}

func (m *Manager) driveOutboundCompletion(sess *pairingSession) {
	log.Printf("pairing %s: polling joiner readiness", sess.inviteID)
	client := &http.Client{Timeout: pairingHTTPTimeout}
	defer client.CloseIdleConnections()
	deadline := time.UnixMilli(sess.createdAt).Add(m.effectiveInviteTTL())
	ready := false
	for time.Now().Before(deadline) {
		if !m.invitePending(sess.inviteID) {
			return
		}
		out, err := exchangePairingEnvelope(client, "http", sess.joinerAddr,
			pairingEnvelope{InviteID: sess.inviteID, Phase: "outbound-ready"})
		if err == nil && out.InviteID == sess.inviteID && out.State != inviteStatePending {
			method := ""
			state := out.State
			switch out.State {
			case inviteStateDeclined:
				method = "cluster:invite-declined"
			case inviteStateCanceled:
				// Same-sender replacement cancels the joiner's old prompt but
				// preserves the inviter's established declined outcome.
				state = inviteStateDeclined
				method = "cluster:invite-declined"
			case inviteStateExpired:
				method = "cluster:invite-expired"
			case inviteStateRejected, inviteStateFailed:
				method = "cluster:invite-failed"
			}
			if method != "" {
				reason := ""
				if out.Reason == reasonIncorrectPIN {
					reason = reasonIncorrectPIN
				}
				m.handlePairingTerminalSignal(&pairingEnvelope{InviteID: sess.inviteID}, state, reason, method, "")
				return
			}
		}
		if err == nil && out.InviteID == sess.inviteID && out.State == inviteStatePending && out.Ready {
			log.Printf("pairing %s: joiner ready for inviter-driven completion", sess.inviteID)
			ready = true
			break
		}
		time.Sleep(200 * time.Millisecond)
	}
	if !ready {
		return
	}
	var blob []byte
	var err error
	// EAP operations/commit serialize with cancel. Every HTTP wait below is
	// outside both inviteMu and rosterMu (and outside the per-session lock).
	m.inviteMu.Lock()
	sess.mu.Lock()
	if inv, exists := m.getInvite(sess.inviteID); exists && inv.State == inviteStatePending {
		blob, err = sess.server.Start()
	}
	sess.mu.Unlock()
	m.inviteMu.Unlock()
	if err != nil || len(blob) == 0 {
		return
	}
	for {
		response, postErr := postPairingBlob(client, sess.joinerAddr, sess.inviteID, "outbound-completion", blob)
		if postErr != nil {
			log.Printf("pairing %s: completion transport: %v", sess.inviteID, postErr)
			// Once committed, an HTTP error or silence proves no outcome. Retain
			// that durable admission and try only authenticated confirmation.
			m.confirmOutboundPairing(sess)
			return
		}
		m.inviteMu.Lock()
		sess.mu.Lock()
		if inv, exists := m.getInvite(sess.inviteID); !exists || inv.State != inviteStatePending {
			sess.mu.Unlock()
			m.inviteMu.Unlock()
			m.confirmOutboundPairing(sess)
			return
		}
		out, receiveErr := sess.server.Receive(response)
		if receiveErr != nil || (out.Done && !out.Success) {
			log.Printf("pairing %s: completion rejected by EAP", sess.inviteID)
			reason := ""
			if out.Err != nil && (out.Err.Code == eapnoob.ErrUnrecognizedOOBMsgID || out.Err.Code == eapnoob.ErrHMACVerificationFailed) {
				reason = reasonIncorrectPIN
			}
			sess.mu.Unlock()
			m.inviteMu.Unlock()
			m.handlePairingTerminalSignal(&pairingEnvelope{InviteID: sess.inviteID}, inviteStateFailed,
				reason, "cluster:invite-failed", "")
			if len(out.Send) > 0 {
				_, _ = postPairingBlob(client, sess.joinerAddr, sess.inviteID, "outbound-completion", out.Send)
			}
			return
		}
		blob = out.Send
		if out.Done && out.Success {
			pi, cert, parseErr := parsePairingInfo(sess.server.PeerInfo())
			if parseErr == nil && pi.Completion != sess.completionMode {
				parseErr = fmt.Errorf("completion mode changed in transcript")
			}
			if parseErr == nil {
				sess.peerPairing, sess.peerCert = pi, cert
				sess.completionResponse = append([]byte(nil), blob...)
			}
			if parseErr == nil {
				parseErr = m.onInviterPaired(sess.inviteID, sess)
			}
			if parseErr != nil {
				if !sess.rollbackPending {
					m.finishInvite(sess.inviteID, inviteStateFailed)
					m.deleteSession(sess.inviteID)
				}
				sess.mu.Unlock()
				m.maybeLeaveInviteCreatedClusterLocked()
				m.inviteMu.Unlock()
				return
			}
			sess.completionResponse = append([]byte(nil), blob...)
			sess.mu.Unlock()
			m.inviteMu.Unlock()
			// EAP-Success is not exposed until our pin/member commit succeeded.
			for attempt := 0; attempt < inviterNotifyAttempts; attempt++ {
				_, err = postPairingBlob(client, sess.joinerAddr, sess.inviteID, "outbound-completion", blob)
				if err == nil {
					break
				}
			}
			m.confirmOutboundPairing(sess)
			return
		}
		sess.mu.Unlock()
		m.inviteMu.Unlock()
	}
}

func (m *Manager) confirmOutboundPairing(sess *pairingSession) {
	sess.mu.Lock()
	if !sess.awaitingAck || sess.rollbackPending || sess.peerPairing == nil || sess.peerCert == nil {
		sess.mu.Unlock()
		return
	}
	pi := *sess.peerPairing
	peerDER := append([]byte(nil), sess.peerCert.Raw...)
	clusterID, inviterEpoch := m.currentAdmission()
	deadline := time.UnixMilli(sess.createdAt).Add(m.effectiveInviteTTL())
	sess.mu.Unlock()
	tlsConfig, err := m.buildPairingClientTLSConfig(peerDER)
	if err != nil {
		return
	}
	transport := &http.Transport{TLSClientConfig: tlsConfig}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, Timeout: pairingHTTPTimeout}
	for attempt := 0; attempt < inviterNotifyAttempts && time.Now().Before(deadline); {
		out, err := exchangePairingEnvelope(client, "https", sess.joinerAddr, pairingEnvelope{
			InviteID: sess.inviteID, Phase: "confirm", ClusterID: clusterID,
			InviterEpoch: inviterEpoch, JoinerEpoch: pi.AdmissionEpoch,
		})
		if err != nil {
			attempt++
			continue
		}
		if out.InviteID != sess.inviteID || out.ClusterID != clusterID || out.InviterEpoch != inviterEpoch || out.JoinerEpoch != pi.AdmissionEpoch {
			return
		}
		switch out.State {
		case inviteStatePending:
			if out.Reason != reasonRollbackPending {
				return
			}
			m.inviteMu.Lock()
			sess.mu.Lock()
			current, live := m.getSession(sess.inviteID)
			waiting := live && current == sess && sess.awaitingAck && !sess.rollbackPending
			if waiting {
				m.finishInviteReason(sess.inviteID, inviteStatePending, reasonAwaitingConfirmation)
			}
			sess.mu.Unlock()
			m.inviteMu.Unlock()
			if !waiting {
				return
			}
			time.Sleep(200 * time.Millisecond)
			continue
		case inviteStatePaired:
			m.ackPairing(&pairingEnvelope{InviteID: sess.inviteID}, pi.NodeUUID)
		case inviteStateFailed:
			m.handlePairingTerminalSignal(&pairingEnvelope{InviteID: sess.inviteID}, inviteStateFailed, "", "cluster:invite-failed", pi.NodeUUID)
		}
		return
	}
}

// Confirmation authenticates a completed transcript, not the general pin set.
// Consequently a failed membership/pin write can still report failure, while
// roster/removal/tunnel handlers continue to require an ordinary durable pin.
func (m *Manager) handlePairingConfirmation(w http.ResponseWriter, r *http.Request, env *pairingEnvelope) {
	if r.TLS == nil || len(r.TLS.PeerCertificates) == 0 {
		http.Error(w, "confirmation requires TLS", http.StatusForbidden)
		return
	}
	var pi *PairingInfo
	var cert *x509.Certificate
	state := InviteState("")
	reason := ""
	if sess, live := m.getSession(env.InviteID); live {
		sess.mu.Lock()
		if sess.role == roleJoiner && sess.completionMode == inviterCompletionV1 && sess.peer.State() == eapnoob.StateRegistered {
			pi, cert = sess.peerPairing, sess.peerCert
			if inv, ok := m.getInvite(env.InviteID); ok {
				state = inv.State
				reason = inv.Reason
			}
			if state == inviteStatePaired {
				if receipt, ok := m.getPairingReceipt(env.InviteID); ok {
					state = m.receiptOutcome(receipt, pi, cert)
				} else {
					state = ""
				}
			}
			if env.JoinerEpoch != sess.localAdmissionEpoch {
				pi = nil
			}
		}
		sess.mu.Unlock()
	} else if receipt, ok := m.getPairingReceipt(env.InviteID); ok && receipt.Role == roleJoiner && receipt.LocalEpoch == env.JoinerEpoch {
		var err error
		pi, cert, err = parsePairingInfo(receipt.PeerInfo)
		if err == nil {
			state = m.receiptOutcome(receipt, pi, cert)
			if receipt.Rollback {
				reason = reasonRollbackPending
			}
		}
	}
	if pi == nil || cert == nil || !bytes.Equal(cert.Raw, r.TLS.PeerCertificates[0].Raw) ||
		pi.ClusterID != env.ClusterID || pi.AdmissionEpoch != env.InviterEpoch ||
		(state != inviteStatePaired && state != inviteStateFailed && state != inviteStatePending) {
		http.Error(w, "unknown or mismatched pairing outcome", http.StatusForbidden)
		return
	}
	if os.Getenv("NVPAIR_TEST_PAIR_DROP_CONFIRMATION") == "1" {
		dropPairingReply(w)
		return
	}
	// Isolated fault timeline: let a test block a compensation write after the
	// inviter commit but before its authenticated failure response arrives.
	if ms, err := strconv.Atoi(os.Getenv("NVPAIR_TEST_PAIR_CONFIRMATION_DELAY_MS")); err == nil && ms > 0 && ms < int(pairingHTTPTimeout.Milliseconds()) {
		select {
		case <-time.After(time.Duration(ms) * time.Millisecond):
		case <-r.Context().Done():
			return
		}
	}
	respondPairingEnvelope(w, pairingEnvelope{InviteID: env.InviteID, State: state, Reason: reason, ClusterID: env.ClusterID,
		InviterEpoch: env.InviterEpoch, JoinerEpoch: env.JoinerEpoch})
}
