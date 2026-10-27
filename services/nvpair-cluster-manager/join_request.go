// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"nvpair-shared/durablefile"
)

const (
	joinRequestPollInterval  = 2 * time.Second
	joinRequestRetention     = 5 * time.Minute
	joinRequestIncomingLimit = 8
	joinRequestTerminalLimit = 64
	joinRequestSourceLimit   = 128
	joinRequestLost          = "request-status-lost"
)

// JoinRequest contains unverified display claims, never trust or a PIN.
type JoinRequest struct {
	RequestID    string  `json:"requestId"`
	Direction    string  `json:"direction"`
	PeerNodeUUID string  `json:"peerNodeUuid"`
	PeerName     string  `json:"peerName"`
	PeerAddress  string  `json:"peerAddress"`
	PeerPort     int     `json:"peerPort"`
	State        string  `json:"state"`
	InviteID     *string `json:"inviteId"`
	Reason       string  `json:"reason"`
	CreatedAt    int64   `json:"createdAt"`
	ExpiresAt    int64   `json:"expiresAt"`
}

type joinRequestsSnapshot struct {
	Supported               bool          `json:"supported"`
	CanAskToJoin            bool          `json:"canAskToJoin"`
	RequiresSoloReplacement bool          `json:"requiresSoloReplacement"`
	BlockedReason           string        `json:"blockedReason"`
	Requests                []JoinRequest `json:"requests"`
}

type joinRequestRecord struct {
	JoinRequest
	source                string
	requesterName         string // original bounded claim, for exact duplicate detection
	replaceSolo           bool
	allowRemoteManagement bool
	clusterID             string
	admission             uint64
	generation            uint64
	sessionGeneration     uint64
	ready                 bool
	terminalAt            int64
	stopState             string // a cancellation/expiry deferred behind Completion
	stopReason            string
}

func joinRequestTerminal(state string) bool {
	switch state {
	case "paired", "declined", "canceled", "expired", "failed", "rejected":
		return true
	}
	return false
}

func validJoinRequestID(id string) bool {
	if len(id) != 35 || !strings.HasPrefix(id, "jr-") {
		return false
	}
	b, err := hex.DecodeString(id[3:])
	return err == nil && len(b) == 16 && hex.EncodeToString(b) == id[3:]
}

func newJoinRequestID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return "jr-" + hex.EncodeToString(b[:]), nil
}

func joinRequestName(name string) string {
	name = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) || unicode.Is(unicode.Cf, r) {
			return -1
		}
		return r
	}, name)
	runes := []rune(strings.TrimSpace(name))
	if len(runes) > 80 {
		runes = runes[:80]
	}
	return string(runes)
}

func (m *Manager) joinTime() time.Time {
	if m.joinNow != nil {
		return m.joinNow()
	}
	return time.Now()
}

func (m *Manager) wakeJoinRequests() {
	select {
	case m.joinWake <- struct{}{}:
	default:
	}
}

// joinEligibilityLocked reads the durable authorities, not discovery or the
// startup caches. Caller holds inviteMu and rosterMu. ignoreInvite is the exact
// child whose outcome is already owned by the caller.
func (m *Manager) joinEligibilityLocked(ignoreInvite string) (solo bool, reason string) {
	if m.teardownPending.Load() {
		return false, "state-unavailable"
	}
	if pending, err := m.pendingTeardown(); err != nil || pending {
		return false, "state-unavailable"
	}
	cid, _ := m.clusterIdentity()
	admissionCID, epoch := m.currentAdmission()
	if !m.joinAdmissionAvailableLocked() {
		return false, "state-unavailable"
	}
	if cid != admissionCID {
		return false, "state-unavailable"
	}
	doc, marker, err := m.readRemoteManagementLocked()
	if err != nil || marker.State != "complete" {
		return false, "state-unavailable"
	}
	if len(doc.Grants) > 0 {
		return false, "already-clustered"
	}
	// Even malformed or skipped pins must not permit destructive replacement.
	entries, err := os.ReadDir(m.trust.dir)
	if err != nil {
		return false, "state-unavailable"
	}
	if len(entries) > 0 || len(m.trust.List()) > 0 {
		return false, "already-clustered"
	}
	m.membersStoreMu.Lock()
	data, err := durablefile.ReadFile(m.membersPath())
	m.membersStoreMu.Unlock()
	var members []ClusterNode
	if err == nil {
		if json.Unmarshal(data, &members) != nil || string(data) == "null" {
			return false, "state-unavailable"
		}
	} else if !os.IsNotExist(err) || cid != "" {
		return false, "state-unavailable"
	}
	selfFound := false
	for _, member := range members {
		if member.NodeUUID != m.identity.NodeUUID {
			return false, "already-clustered"
		}
		if member.ClusterID != cid || member.AdmissionEpoch != epoch || member.State != stateMember || selfFound {
			return false, "state-unavailable"
		}
		selfFound = true
	}
	if cid != "" && (!selfFound || epoch == 0) {
		return false, "state-unavailable"
	}
	m.memMu.Lock()
	busy := false
	for _, member := range m.members {
		if member.NodeUUID != m.identity.NodeUUID {
			busy = true
		}
	}
	for id, inv := range m.invites {
		if id != ignoreInvite && (inv.State == inviteStatePending || inv.Reason == reasonRollbackPending || inv.Reason == reasonAwaitingConfirmation) {
			busy = true
		}
	}
	m.memMu.Unlock()
	m.sessMu.Lock()
	for id := range m.sessions {
		if id != ignoreInvite {
			busy = true
		}
	}
	m.sessMu.Unlock()
	if busy {
		return false, "pairing-busy"
	}
	// Re-read receipts: unconfirmed commits and compensation block a new try
	// even after a restart or if a store became unreadable after startup.
	data, err = durablefile.ReadFile(filepath.Join(m.clusterDir, pairingReceiptsFile))
	if err == nil {
		var receipts map[string]pairingReceipt
		if json.Unmarshal(data, &receipts) != nil || receipts == nil {
			return false, "state-unavailable"
		}
		for id, receipt := range receipts {
			if receipt.InviteID != id || receipt.LocalEpoch == 0 || receipt.ClusterID == "" {
				return false, "state-unavailable"
			}
			if id != ignoreInvite && (receipt.Rollback || receipt.State == "" || receipt.State == inviteStatePending) {
				return false, "pairing-busy"
			}
		}
	} else if !os.IsNotExist(err) {
		return false, "state-unavailable"
	}
	for _, request := range m.joinRequests {
		if !joinRequestTerminal(request.State) && (request.Reason == reasonAwaitingConfirmation || request.Reason == reasonRollbackPending) {
			return false, "pairing-busy"
		}
	}
	return cid != "", ""
}

// The on-disk admission is authoritative even when a caller has an older
// in-memory identity. Never approve or replace using an unreadable authority.
func (m *Manager) joinAdmissionAvailableLocked() bool {
	cid, _ := m.clusterIdentity()
	admissionCID, epoch := m.currentAdmission()
	if cid != admissionCID {
		return false
	}
	data, err := durablefile.ReadFile(m.admissionPath())
	if os.IsNotExist(err) {
		return admissionCID == "" && epoch == 0
	}
	if err != nil {
		return false
	}
	var admission admissionState
	return json.Unmarshal(data, &admission) == nil && admission.ClusterID == admissionCID && admission.Epoch == epoch &&
		admission.Epoch <= admission.Counter && admission.Activated <= admission.Counter &&
		(admission.Activated == 0 || admission.Epoch <= admission.Activated) && (admission.ClusterID == "") == (admission.Epoch == 0)
}

func (m *Manager) finishJoinRequestLocked(r *joinRequestRecord, state, reason string) {
	if joinRequestTerminal(r.State) {
		return
	}
	r.State, r.Reason = state, reason
	r.terminalAt = m.joinTime().UnixMilli()
	if r.InviteID != nil {
		if sess, live := m.getSession(*r.InviteID); live && sess.initialCancel != nil {
			sess.initialCancel()
		}
	}
}

// reconcileJoinRequestsLocked requires inviteMu, rosterMu and joinMu, in that
// order. It never waits for a session held across network I/O.
func (m *Manager) reconcileJoinRequestsLocked() {
	now := m.joinTime().UnixMilli()
	for id, r := range m.joinRequests {
		if joinRequestTerminal(r.State) {
			if r.terminalAt+joinRequestRetention.Milliseconds() <= now {
				delete(m.joinRequests, id)
			}
			continue
		}
		if r.InviteID != nil {
			if inv, ok := m.getInvite(*r.InviteID); ok {
				if inv.Reason == reasonRollbackPending || inv.Reason == reasonAwaitingConfirmation {
					r.State, r.Reason = "pairing", inv.Reason
					continue
				}
				if inv.State == inviteStatePaired && r.Direction == "incoming" {
					// Only ackPairing's authenticated exact confirmation settles
					// an admitting member's request. Session loss/expiry never does.
					r.State, r.Reason = "pairing", reasonAwaitingConfirmation
					continue
				}
				if inv.State != inviteStatePending {
					m.finishJoinRequestLocked(r, string(inv.State), inv.Reason)
					continue
				}
			}
		}
		// Losing the ordinary Invite/session cache is not evidence that a
		// committed admission failed. Keep the durable uncertainty visible.
		if r.Reason == reasonAwaitingConfirmation || r.Reason == reasonRollbackPending {
			r.State = "pairing"
			continue
		}
		if r.Direction == "incoming" && (r.sessionGeneration != m.sessGen.Load() || (!r.ready && r.generation != m.clusterGen.Load())) {
			m.finishJoinRequestLocked(r, "rejected", "approver-changed")
			continue
		}
		if r.ExpiresAt <= now && r.stopState == "" {
			r.stopState = "expired"
		}
		if r.stopState != "" {
			if m.stopJoinRequestChildLocked(r) {
				m.finishJoinRequestLocked(r, r.stopState, r.stopReason)
			}
		}
	}
}

// stopJoinRequestChildLocked uses the same pending/session guards and terminal
// helpers as ordinary cancellation and expiry. A completed/uncertain durable
// outcome always wins; cancellation never revokes completed membership.
func (m *Manager) stopJoinRequestChildLocked(r *joinRequestRecord) bool {
	if r.InviteID == nil {
		return true
	}
	id := *r.InviteID
	inv, exists := m.getInvite(id)
	if exists && (inv.Reason == reasonRollbackPending || inv.Reason == reasonAwaitingConfirmation || inv.State == inviteStatePaired) {
		return false
	}
	sess, live := m.getSession(id)
	if !live {
		if exists && inv.State == inviteStatePending {
			m.finishInviteReason(id, InviteState(r.stopState), r.stopReason)
		}
		return true
	}
	if sess.initialCancel != nil {
		sess.initialCancel()
	}
	if !sess.mu.TryLock() {
		return false
	}
	defer sess.mu.Unlock()
	inv, exists = m.getInvite(id)
	if sess.awaitingAck || sess.rollbackPending || (exists && inv.State != inviteStatePending) {
		return false
	}
	m.finishInviteReason(id, InviteState(r.stopState), r.stopReason)
	m.deleteSession(id)
	if sess.role == roleJoiner {
		if sess.peerPairing != nil {
			if m.removeMemberByUUID(sess.peerPairing.NodeUUID) {
				m.emitNodesChanged()
			}
		}
		go m.notifyInviterTerminal(sess.legacyNotifyAddress(), id, "decline", "")
	} else {
		go m.notifyJoinerCanceled(sess.joinerAddr, id)
	}
	m.emitInviteExpired(id)
	return true
}

func (m *Manager) joinSnapshot() joinRequestsSnapshot {
	m.inviteMu.Lock()
	defer m.inviteMu.Unlock()
	m.rosterMu.Lock()
	defer m.rosterMu.Unlock()
	m.joinMu.Lock()
	defer m.joinMu.Unlock()
	m.reconcileJoinRequestsLocked()
	solo, reason := m.joinEligibilityLocked("")
	snapshot := joinRequestsSnapshot{Supported: true, CanAskToJoin: reason == "", RequiresSoloReplacement: solo, BlockedReason: reason, Requests: []JoinRequest{}}
	for _, r := range m.joinRequests {
		if r.Direction == "outgoing" && !joinRequestTerminal(r.State) {
			snapshot.CanAskToJoin, snapshot.BlockedReason = false, "pairing-busy"
		}
		snapshot.Requests = append(snapshot.Requests, cloneJoinRequest(r))
	}
	sort.Slice(snapshot.Requests, func(i, j int) bool { return snapshot.Requests[i].RequestID < snapshot.Requests[j].RequestID })
	return snapshot
}

func cloneJoinRequest(r *joinRequestRecord) JoinRequest {
	cp := r.JoinRequest
	if cp.InviteID != nil {
		id := *cp.InviteID
		cp.InviteID = &id
	}
	return cp
}

func (m *Manager) joinRequestsLoop(ctx context.Context) {
	ticker := time.NewTicker(joinRequestPollInterval)
	defer ticker.Stop()
	var previous string
	for {
		snapshot := m.joinSnapshot()
		data, err := json.Marshal(snapshot)
		if err == nil && string(data) != previous {
			previous = string(data)
			_ = m.codec.Notify("cluster:join-requests-changed", snapshot)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		case <-m.joinWake:
		}
	}
}

func (m *Manager) handleGetJoinRequests(msg *Message) {
	if _, err := decodeRemoteManagementParams(msg.Params, map[string]bool{}); err != nil {
		m.codec.RespondError(msg.ID, codeInvalidParams, "invalid params")
		return
	}
	m.codec.Respond(msg.ID, m.joinSnapshot())
}

type askToJoinParams struct {
	Address            string  `json:"address"`
	Port               *int    `json:"port"`
	NodeID             *string `json:"nodeId"`
	ReplaceSoloCluster *bool   `json:"replaceSoloCluster"`
}

func (m *Manager) handleAskToJoin(msg *Message) {
	var p askToJoinParams
	_, err := decodeRemoteManagementParams(msg.Params, map[string]bool{"address": true, "port": true, "nodeId": true, "replaceSoloCluster": true})
	if err != nil || json.Unmarshal(msg.Params, &p) != nil || p.ReplaceSoloCluster == nil || (p.Port != nil && (*p.Port < 1 || *p.Port > 65535)) {
		m.codec.RespondError(msg.ID, codeInvalidParams, "invalid params: replaceSoloCluster and valid address/port are required")
		return
	}
	endpoints, peerUUID, err := m.joinRequestEndpoints(p.Address, p.Port, p.NodeID)
	if err != nil {
		m.codec.RespondError(msg.ID, codeInvalidParams, err.Error())
		return
	}
	if (p.NodeID != nil && (*p.NodeID == m.identity.NodeUUID || *p.NodeID == m.identity.NodeID)) || m.selfJoinEndpoint(endpoints) {
		m.codec.RespondErrorData(msg.ID, codePrecondition, "self-target", map[string]string{"reason": "self-target"})
		return
	}
	id, err := newJoinRequestID()
	if err != nil {
		m.codec.RespondError(msg.ID, codeInternalError, "create request ID")
		return
	}
	m.inviteMu.Lock()
	m.rosterMu.Lock()
	m.joinMu.Lock()
	m.reconcileJoinRequestsLocked()
	solo, reason := m.joinEligibilityLocked("")
	for _, r := range m.joinRequests {
		if r.Direction == "outgoing" && !joinRequestTerminal(r.State) {
			reason = "pairing-busy"
		}
	}
	if reason == "" && solo && !*p.ReplaceSoloCluster {
		reason = "solo-replacement-required"
	}
	if reason == "" && !m.joinRecordCapacityLocked() {
		reason = "request-capacity"
	}
	if reason != "" {
		m.joinMu.Unlock()
		m.rosterMu.Unlock()
		m.inviteMu.Unlock()
		m.codec.RespondErrorData(msg.ID, codePrecondition, reason, map[string]string{"reason": reason})
		return
	}
	now := m.joinTime()
	host, portText, _ := net.SplitHostPort(endpoints[0])
	port, _ := strconv.Atoi(portText)
	r := &joinRequestRecord{JoinRequest: JoinRequest{RequestID: id, Direction: "outgoing", PeerAddress: host, PeerPort: port, State: "awaiting-approval", CreatedAt: now.UnixMilli(), ExpiresAt: now.Add(m.effectiveInviteTTL()).UnixMilli()}, replaceSolo: *p.ReplaceSoloCluster, generation: m.clusterGen.Load(), sessionGeneration: m.sessGen.Load()}
	r.PeerNodeUUID = peerUUID
	m.joinRequests[id] = r
	result := cloneJoinRequest(r)
	m.joinMu.Unlock()
	m.rosterMu.Unlock()
	m.inviteMu.Unlock()
	m.codec.Respond(msg.ID, result)
	m.wakeJoinRequests()
	go m.driveJoinRequest(id, endpoints)
}

// inviteEndpoints shares ordinary Invite's discovery/explicit-port authority.
func (m *Manager) inviteEndpoints(address string, port *int, nodeID *string) ([]string, error) {
	endpoints, _, err := m.joinRequestEndpoints(address, port, nodeID)
	return endpoints, err
}

// Keep endpoint selection and the expected peer UUID tied to one browser record;
// a hostname selector is never itself a cryptographic peer identity.
func (m *Manager) joinRequestEndpoints(address string, port *int, nodeID *string) ([]string, string, error) {
	p := m.port
	if port != nil {
		p = *port
	}
	endpoints := []string{}
	seen := map[string]bool{}
	add := func(host string, n int) {
		if host == "" || n < 1 || n > 65535 || strings.ContainsAny(host, "/?#@\\") {
			return
		}
		ep := net.JoinHostPort(host, strconv.Itoa(n))
		if !seen[ep] {
			seen[ep] = true
			endpoints = append(endpoints, ep)
		}
	}
	add(address, p)
	peerUUID := ""
	if nodeID != nil && m.browser != nil {
		if hosts, discoveredPort, uuid, ok := m.browser.resolvePeer(*nodeID); ok {
			peerUUID = uuid
			if port == nil && discoveredPort > 0 {
				p = discoveredPort
			}
			for _, host := range hosts {
				add(host, p)
			}
		}
	}
	if len(endpoints) == 0 {
		return nil, "", fmt.Errorf("address is required (or a discoverable nodeId)")
	}
	return endpoints, peerUUID, nil
}

func (m *Manager) selfJoinEndpoint(endpoints []string) bool {
	addresses, err := net.InterfaceAddrs()
	if err != nil {
		return true
	}
	local := map[string]bool{"127.0.0.1": true, "::1": true}
	for _, address := range addresses {
		if ip, _, err := net.ParseCIDR(address.String()); err == nil {
			local[ip.String()] = true
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), pairingHTTPTimeout)
	defer cancel()
	for _, endpoint := range endpoints {
		host, port, err := net.SplitHostPort(endpoint)
		if err != nil || port != strconv.Itoa(m.port) {
			continue
		}
		ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
		if err != nil {
			continue
		}
		for _, ip := range ips {
			if ip.IP.IsLoopback() || local[ip.IP.String()] {
				return true
			}
		}
	}
	return false
}

func (m *Manager) joinRecordCapacityLocked() bool {
	// Reserve a terminal slot for every live record, so a cancellation fence
	// can always be retained without evicting unexpired history.
	return len(m.joinRequests) < joinRequestTerminalLimit
}

func (m *Manager) handleRespondToJoinRequest(msg *Message) {
	var p struct {
		RequestID string `json:"requestId"`
		Accept    *bool  `json:"accept"`
		Allow     *bool  `json:"allowRemoteManagement"`
	}
	fields, err := decodeRemoteManagementParams(msg.Params, map[string]bool{"requestId": true, "accept": true, "allowRemoteManagement": true})
	if err != nil || json.Unmarshal(msg.Params, &p) != nil || !validJoinRequestID(p.RequestID) || p.Accept == nil || (*p.Accept && p.Allow == nil) || (!*p.Accept && len(fields) != 2) {
		m.codec.RespondError(msg.ID, codeInvalidParams, "invalid params")
		return
	}
	m.inviteMu.Lock()
	m.rosterMu.Lock()
	m.joinMu.Lock()
	m.reconcileJoinRequestsLocked()
	r, ok := m.joinRequests[p.RequestID]
	code, reason := 0, ""
	if !ok {
		code, reason = codeUnknownInvite, "unknown requestId"
	} else if r.Direction != "incoming" {
		code, reason = codeInvalidState, "request is not incoming"
	} else if r.State == "awaiting-requester" && *p.Accept && r.allowRemoteManagement == *p.Allow {
		// Idempotent local approval from a second attached client.
	} else if r.State != "awaiting-approval" {
		code, reason = codeInvalidState, "request is not awaiting approval"
	} else if !*p.Accept {
		m.finishJoinRequestLocked(r, "declined", "")
	} else {
		cid, _ := m.clusterIdentity()
		admissionCID, epoch := m.currentAdmission()
		_, marker, permissionErr := m.readRemoteManagementLocked()
		if cid == "" || cid != r.clusterID || admissionCID != cid || epoch != r.admission || m.clusterGen.Load() != r.generation || !m.joinAdmissionAvailableLocked() || permissionErr != nil || marker.State != "complete" {
			m.finishJoinRequestLocked(r, "rejected", "approver-changed")
			code, reason = codePrecondition, "approver-changed"
		} else if inviteID, err := newInviteID(); err != nil {
			code, reason = codeInternalError, "create invite ID"
		} else {
			r.State, r.InviteID, r.allowRemoteManagement = "awaiting-requester", &inviteID, *p.Allow
		}
	}
	var result JoinRequest
	if ok {
		result = cloneJoinRequest(r)
	}
	m.joinMu.Unlock()
	m.rosterMu.Unlock()
	m.inviteMu.Unlock()
	if code != 0 {
		m.codec.RespondError(msg.ID, code, reason)
	} else {
		m.codec.Respond(msg.ID, result)
	}
	m.wakeJoinRequests()
}

func (m *Manager) handleCancelJoinRequest(msg *Message) {
	var p struct {
		RequestID string `json:"requestId"`
	}
	_, err := decodeRemoteManagementParams(msg.Params, map[string]bool{"requestId": true})
	if err != nil || json.Unmarshal(msg.Params, &p) != nil || !validJoinRequestID(p.RequestID) {
		m.codec.RespondError(msg.ID, codeInvalidParams, "invalid params")
		return
	}
	m.inviteMu.Lock()
	m.rosterMu.Lock()
	m.joinMu.Lock()
	m.reconcileJoinRequestsLocked()
	r, ok := m.joinRequests[p.RequestID]
	if !ok || r.Direction != "outgoing" {
		m.joinMu.Unlock()
		m.rosterMu.Unlock()
		m.inviteMu.Unlock()
		code := codeUnknownInvite
		if ok {
			code = codeInvalidState
		}
		m.codec.RespondError(msg.ID, code, "unknown outgoing requestId")
		return
	}
	sendCancel := !joinRequestTerminal(r.State) && r.stopState == ""
	if !joinRequestTerminal(r.State) {
		r.stopState = "canceled"
		m.reconcileJoinRequestsLocked()
	}
	result := cloneJoinRequest(r)
	target := net.JoinHostPort(r.PeerAddress, strconv.Itoa(r.PeerPort))
	m.joinMu.Unlock()
	m.rosterMu.Unlock()
	m.inviteMu.Unlock()
	m.codec.Respond(msg.ID, result)
	m.wakeJoinRequests()
	if sendCancel && result.State != "paired" {
		go m.sendJoinRequestCancel(target, p.RequestID)
	}
}

func (m *Manager) joinRequestForInvite(inviteID string) string {
	m.joinMu.Lock()
	defer m.joinMu.Unlock()
	for id, r := range m.joinRequests {
		if r.InviteID != nil && *r.InviteID == inviteID && !joinRequestTerminal(r.State) {
			return id
		}
	}
	return ""
}

func (m *Manager) isJoinRequestInvite(inviteID string) bool {
	m.joinMu.Lock()
	defer m.joinMu.Unlock()
	for _, r := range m.joinRequests {
		if r.InviteID != nil && *r.InviteID == inviteID {
			return true
		}
	}
	return false
}

// The Initial Exchange shares the request's original deadline, even when an
// endpoint retry starts a new EAP session. Ordinary Invite keeps its own TTL.
func (m *Manager) joinRequestInitialContext(inviteID string) (context.Context, context.CancelFunc, string, error) {
	m.joinMu.Lock()
	defer m.joinMu.Unlock()
	for id, r := range m.joinRequests {
		if r.InviteID == nil || *r.InviteID != inviteID {
			continue
		}
		remaining := time.UnixMilli(r.ExpiresAt).Sub(m.joinTime())
		if joinRequestTerminal(r.State) || r.stopState != "" || remaining <= 0 {
			return nil, nil, id, errPairingAbandoned
		}
		ctx, cancel := context.WithTimeout(context.Background(), remaining)
		return ctx, cancel, id, nil
	}
	return context.Background(), nil, "", nil
}

func (m *Manager) joinRequestInitialLive(requestID, inviteID string, sess *pairingSession) bool {
	m.joinMu.Lock()
	r, ok := m.joinRequests[requestID]
	live := ok && r.InviteID != nil && *r.InviteID == inviteID && !joinRequestTerminal(r.State) &&
		r.stopState == "" && r.ExpiresAt > m.joinTime().UnixMilli()
	m.joinMu.Unlock()
	current, sessionLive := m.getSession(inviteID)
	inv, inviteLive := m.getInvite(inviteID)
	return live && sessionLive && current == sess && inviteLive && inv.State == inviteStatePending
}

// Called with sess.mu held but no composition locks. Membership is still
// unauthenticated here; this only binds the forthcoming PIN-authenticated
// transcript to the exact local request and child invite.
func (m *Manager) matchJoinRequestPeer(inviteID string, pi *PairingInfo) bool {
	m.joinMu.Lock()
	defer m.joinMu.Unlock()
	for id, r := range m.joinRequests {
		if r.InviteID == nil || *r.InviteID != inviteID {
			continue
		}
		return !joinRequestTerminal(r.State) && r.stopState == "" && r.ExpiresAt > m.joinTime().UnixMilli() && pi.JoinRequestID == id &&
			(r.PeerNodeUUID == "" || r.PeerNodeUUID == pi.NodeUUID)
	}
	return pi.JoinRequestID == ""
}

// Confirmation has already verified the exact durable admission and grants.
// Caller holds inviteMu, before publishing the request's final snapshot.
func (m *Manager) confirmJoinRequest(inviteID string) {
	m.joinMu.Lock()
	defer m.joinMu.Unlock()
	for _, r := range m.joinRequests {
		if r.InviteID != nil && *r.InviteID == inviteID && !joinRequestTerminal(r.State) {
			m.finishJoinRequestLocked(r, "paired", "")
		}
	}
	m.wakeJoinRequests()
}
