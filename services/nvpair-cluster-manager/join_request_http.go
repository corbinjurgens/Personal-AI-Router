// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"strconv"
	"time"
)

const joinRequestPath = "/v1/cluster/join-requests"
const joinRequestBodyLimit = 4096

type joinRequestWire struct {
	Version   int            `json:"version"`
	Operation string         `json:"operation"`
	RequestID string         `json:"requestId"`
	Requester *joinRequester `json:"requester,omitempty"`
	Port      *int           `json:"port,omitempty"`
}
type joinRequester struct {
	NodeUUID string `json:"nodeUuid"`
	Name     string `json:"name"`
}
type joinRequestReply struct {
	Version   int     `json:"version"`
	RequestID string  `json:"requestId"`
	State     string  `json:"state"`
	Reason    string  `json:"reason"`
	CreatedAt int64   `json:"createdAt"`
	ExpiresAt int64   `json:"expiresAt"`
	InviteID  *string `json:"inviteId"`
}

// All buckets and source entries are owned by joinMu. A source entry never
// owns a goroutine or unbounded queue.
type joinRequestBucket struct {
	tokens float64
	last   time.Time
}
type joinRequestSource struct {
	create, follow joinRequestBucket
	last           time.Time
}

func (b *joinRequestBucket) take(now time.Time, period time.Duration, burst float64) bool {
	if b.last.IsZero() {
		b.tokens, b.last = burst, now
	}
	if now.After(b.last) {
		b.tokens += float64(now.Sub(b.last)) / float64(period)
		b.last = now
	}
	if b.tokens > burst {
		b.tokens = burst
	}
	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

func joinRequestSourceIP(remote string) (string, bool) {
	host, _, err := net.SplitHostPort(remote)
	if err != nil {
		return "", false
	}
	ip, err := netip.ParseAddr(host)
	if err != nil || ip.Zone() != "" {
		return "", false
	}
	ip = ip.Unmap()
	if ip.IsLoopback() {
		return ip.String(), true
	}
	allowed := ip.IsPrivate() || netip.MustParsePrefix("100.64.0.0/10").Contains(ip)
	return ip.String(), allowed && !ip.IsLinkLocalUnicast() && !ip.IsMulticast() && !ip.IsUnspecified()
}

func decodeJoinRequestWire(data []byte) (joinRequestWire, string) {
	var wire joinRequestWire
	fields, err := decodeRemoteManagementParams(data, map[string]bool{"version": true, "operation": true, "requestId": true, "requester": true, "port": true})
	if err != nil || json.Unmarshal(data, &wire) != nil {
		return wire, "malformed-request"
	}
	if wire.Version != 1 {
		return wire, "unsupported-version"
	}
	if !validJoinRequestID(wire.RequestID) {
		return wire, "malformed-request"
	}
	if wire.Operation == "request" {
		if wire.Requester == nil || wire.Port == nil || *wire.Port < 1 || *wire.Port > 65535 {
			return wire, "malformed-request"
		}
		requesterFields, err := decodeRemoteManagementParams(fields["requester"], map[string]bool{"nodeUuid": true, "name": true})
		if err != nil || len(requesterFields) != 2 || bytes.Equal(requesterFields["name"], []byte("null")) || wire.Requester.NodeUUID == "" || len(wire.Requester.NodeUUID) > 128 || len(wire.Requester.Name) > 320 {
			return wire, "malformed-request"
		}
		if _, err := hexNodeUUID(wire.Requester.NodeUUID); err != nil {
			return wire, "malformed-request"
		}
	} else if wire.Operation == "status" || wire.Operation == "ready" || wire.Operation == "cancel" {
		if len(fields) != 3 || wire.Requester != nil || wire.Port != nil {
			return wire, "malformed-request"
		}
	} else {
		return wire, "malformed-request"
	}
	return wire, ""
}

// UUIDs are bounded claims at this endpoint. Restrict their alphabet so control
// characters cannot enter local cards or correlation records.
func hexNodeUUID(uuid string) (string, error) {
	for _, c := range uuid {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-') {
			return "", fmt.Errorf("invalid node UUID")
		}
	}
	return uuid, nil
}

func writeJoinRequestError(w http.ResponseWriter, code int, reason string) {
	w.Header().Set("Content-Type", "application/json")
	if code == http.StatusTooManyRequests {
		w.Header().Set("Retry-After", "30")
	}
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(struct {
		Version int    `json:"version"`
		Reason  string `json:"reason"`
	}{1, reason})
}

func (m *Manager) handleJoinRequestHTTP(w http.ResponseWriter, r *http.Request) {
	select {
	case m.joinHandlers <- struct{}{}:
		defer func() { <-m.joinHandlers }()
	default:
		writeJoinRequestError(w, http.StatusTooManyRequests, "request-capacity")
		return
	}
	if r.Method != http.MethodPost {
		writeJoinRequestError(w, http.StatusMethodNotAllowed, "method-not-allowed")
		return
	}
	source, allowed := joinRequestSourceIP(r.RemoteAddr)
	if !allowed {
		writeJoinRequestError(w, http.StatusForbidden, "source-not-allowed")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), pairingHTTPTimeout)
	defer cancel()
	controller := http.NewResponseController(w)
	_ = controller.SetReadDeadline(time.Now().Add(pairingHTTPTimeout))
	_ = controller.SetWriteDeadline(time.Now().Add(pairingHTTPTimeout))
	data, err := io.ReadAll(io.LimitReader(r.Body, joinRequestBodyLimit+1))
	if err != nil || len(data) > joinRequestBodyLimit || ctx.Err() != nil {
		writeJoinRequestError(w, http.StatusBadRequest, "malformed-request")
		return
	}
	wire, reason := decodeJoinRequestWire(data)
	if reason != "" {
		writeJoinRequestError(w, http.StatusBadRequest, reason)
		return
	}
	reply, code, reason := m.applyJoinRequestWireContext(ctx, source, wire)
	if code != http.StatusOK {
		writeJoinRequestError(w, code, reason)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(reply)
	m.wakeJoinRequests()
}

func (m *Manager) joinSourceLocked(source string) *joinRequestSource {
	now := m.joinTime()
	for ip, entry := range m.joinSources {
		if now.Sub(entry.last) >= joinRequestRetention {
			delete(m.joinSources, ip)
		}
	}
	entry := m.joinSources[source]
	if entry == nil {
		if len(m.joinSources) >= joinRequestSourceLimit {
			return nil
		}
		entry = &joinRequestSource{}
		m.joinSources[source] = entry
	}
	entry.last = now
	return entry
}

func (m *Manager) applyJoinRequestWire(source string, wire joinRequestWire) (joinRequestReply, int, string) {
	ctx, cancel := context.WithTimeout(context.Background(), pairingHTTPTimeout)
	defer cancel()
	return m.applyJoinRequestWireContext(ctx, source, wire)
}

func (m *Manager) applyJoinRequestWireContext(ctx context.Context, source string, wire joinRequestWire) (joinRequestReply, int, string) {
	unlock, locked := m.lockJoinRequestOperation(ctx)
	if !locked {
		return joinRequestReply{}, http.StatusTooManyRequests, "request-busy"
	}
	defer unlock()
	m.reconcileJoinRequestsLocked()
	fail := func(code int, reason string) (joinRequestReply, int, string) { return joinRequestReply{}, code, reason }
	entry := m.joinSourceLocked(source)
	if entry == nil {
		return fail(http.StatusTooManyRequests, "source-capacity")
	}
	now := m.joinTime()
	r, exists := m.joinRequests[wire.RequestID]
	if wire.Operation != "request" && !entry.follow.take(now, 200*time.Millisecond, 10) {
		return fail(http.StatusTooManyRequests, "rate-limited")
	}
	if exists {
		if wire.Operation == "request" && !entry.follow.take(now, 200*time.Millisecond, 10) {
			return fail(http.StatusTooManyRequests, "rate-limited")
		}
		if r.Direction != "incoming" || r.source != source {
			return fail(http.StatusNotFound, "unknown-request")
		}
		if wire.Operation == "request" && r.PeerNodeUUID != "" && (r.PeerNodeUUID != wire.Requester.NodeUUID || r.requesterName != wire.Requester.Name || r.PeerPort != *wire.Port) {
			return fail(http.StatusConflict, "conflicting-request")
		}
	} else {
		if wire.Operation == "status" || wire.Operation == "ready" {
			return fail(http.StatusNotFound, "unknown-request")
		}
		if !entry.create.take(now, 30*time.Second, 1) || !m.joinGlobal.take(now, time.Second, 4) {
			return fail(http.StatusTooManyRequests, "rate-limited")
		}
		if !m.joinRecordCapacityLocked() {
			return fail(http.StatusTooManyRequests, "request-capacity")
		}
		live := 0
		for _, current := range m.joinRequests {
			if current.Direction == "incoming" && !joinRequestTerminal(current.State) {
				live++
				if current.source == source {
					return fail(http.StatusTooManyRequests, "source-busy")
				}
			}
		}
		if live >= joinRequestIncomingLimit {
			return fail(http.StatusTooManyRequests, "request-capacity")
		}
		cid, _ := m.clusterIdentity()
		admissionCID, epoch := m.currentAdmission()
		if wire.Operation == "request" {
			if cid == "" {
				return fail(http.StatusConflict, "target-not-clustered")
			}
			_, marker, permissionErr := m.readRemoteManagementLocked()
			if admissionCID != cid || epoch == 0 || !m.joinAdmissionAvailableLocked() || m.teardownPending.Load() || permissionErr != nil || marker.State != "complete" {
				return fail(http.StatusConflict, "state-unavailable")
			}
			if wire.Requester.NodeUUID == m.identity.NodeUUID {
				return fail(http.StatusConflict, "self-target")
			}
		}
		r = &joinRequestRecord{JoinRequest: JoinRequest{RequestID: wire.RequestID, Direction: "incoming", PeerAddress: source, PeerPort: m.port, State: "awaiting-approval", CreatedAt: now.UnixMilli(), ExpiresAt: now.Add(m.effectiveInviteTTL()).UnixMilli()}, source: source, clusterID: cid, admission: epoch, generation: m.clusterGen.Load(), sessionGeneration: m.sessGen.Load()}
		if wire.Operation == "request" {
			r.PeerNodeUUID, r.PeerName, r.PeerPort, r.requesterName = wire.Requester.NodeUUID, joinRequestName(wire.Requester.Name), *wire.Port, wire.Requester.Name
		}
		m.joinRequests[wire.RequestID] = r
	}
	if wire.Operation == "cancel" {
		if !joinRequestTerminal(r.State) {
			r.stopState = "canceled"
			m.reconcileJoinRequestsLocked()
		}
	} else if wire.Operation == "ready" {
		if r.State == "awaiting-requester" && !r.ready {
			cid, friendly := m.clusterIdentity()
			admissionCID, epoch := m.currentAdmission()
			_, marker, permissionErr := m.readRemoteManagementLocked()
			if cid != r.clusterID || admissionCID != cid || epoch != r.admission || !m.joinAdmissionAvailableLocked() || m.clusterGen.Load() != r.generation || m.sessGen.Load() != r.sessionGeneration || permissionErr != nil || marker.State != "complete" || m.teardownPending.Load() {
				m.finishJoinRequestLocked(r, "rejected", "approver-changed")
			} else {
				r.ready = true
				r.State = "pairing"
				inviteID := *r.InviteID
				peerID := r.PeerNodeUUID
				m.putInvite(&Invite{InviteID: inviteID, FromNodeID: m.identity.NodeID, FromNodeUUID: m.identity.NodeUUID, FromNodeName: m.identity.Name, ToNodeID: &peerID, ClusterID: cid, ClusterFriendlyName: friendly, State: inviteStatePending, CreatedAt: now.UnixMilli()})
				target := net.JoinHostPort(source, strconv.Itoa(r.PeerPort))
				gen, allow := r.sessionGeneration, r.allowRemoteManagement
				go func() { m.completeInitialInvite(inviteID, []string{target}, cid, gen, allow); m.wakeJoinRequests() }()
			}
		} else if r.State == "awaiting-approval" {
			return fail(http.StatusConflict, "approval-required")
		}
	}
	return joinRequestReply{1, r.RequestID, r.State, r.Reason, r.CreatedAt, r.ExpiresAt, r.InviteID}, http.StatusOK, ""
}

// Bound the wait behind existing composition work without creating a worker or
// holding one outer owner while waiting for another. At most the admitted HTTP
// handlers can wait here, and cancellation/deadline releases their permits.
func (m *Manager) lockJoinRequestOperation(ctx context.Context) (func(), bool) {
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	for {
		if ctx.Err() != nil {
			return nil, false
		}
		if m.inviteMu.TryLock() {
			if m.rosterMu.TryLock() {
				if m.joinMu.TryLock() {
					return func() { m.joinMu.Unlock(); m.rosterMu.Unlock(); m.inviteMu.Unlock() }, true
				}
				m.rosterMu.Unlock()
			}
			m.inviteMu.Unlock()
		}
		select {
		case <-ctx.Done():
			return nil, false
		case <-ticker.C:
		}
	}
}

func callJoinRequest(ctx context.Context, target string, wire joinRequestWire) (joinRequestReply, int, error) {
	var reply joinRequestReply
	data, err := json.Marshal(wire)
	if err != nil {
		return reply, 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+target+joinRequestPath, bytes.NewReader(data))
	if err != nil {
		return reply, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := pairingHTTPClient().Do(req)
	if err != nil {
		return reply, 0, err
	}
	defer resp.Body.Close()
	data, err = io.ReadAll(io.LimitReader(resp.Body, joinRequestBodyLimit+1))
	if err != nil || len(data) > joinRequestBodyLimit {
		return reply, resp.StatusCode, fmt.Errorf("invalid request response")
	}
	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode >= 300 && resp.StatusCode < 400 {
			reply.Reason = "redirect-refused"
			return reply, resp.StatusCode, errPairingRedirect
		}
		var rejection struct {
			Version int    `json:"version"`
			Reason  string `json:"reason"`
		}
		if json.Unmarshal(data, &rejection) == nil && rejection.Version == 1 {
			reply.Reason = rejection.Reason
		}
		return reply, resp.StatusCode, fmt.Errorf("join request refused: status %d", resp.StatusCode)
	}
	fields, err := decodeRemoteManagementParams(data, map[string]bool{"version": true, "requestId": true, "state": true, "reason": true, "createdAt": true, "expiresAt": true, "inviteId": true})
	if err != nil || len(fields) != 7 {
		return reply, resp.StatusCode, fmt.Errorf("invalid request response")
	}
	if bytes.Equal(fields["reason"], []byte("null")) {
		return reply, resp.StatusCode, fmt.Errorf("invalid request response")
	}
	if json.Unmarshal(data, &reply) != nil || reply.Version != 1 || reply.RequestID != wire.RequestID || !validJoinRequestState(reply.State) || reply.CreatedAt <= 0 || reply.ExpiresAt <= reply.CreatedAt {
		return reply, resp.StatusCode, fmt.Errorf("invalid request response")
	}
	return reply, resp.StatusCode, nil
}

func validJoinRequestState(state string) bool {
	return state == "awaiting-approval" || state == "awaiting-requester" || state == "pairing" || joinRequestTerminal(state)
}

func (m *Manager) driveJoinRequest(id string, endpoints []string) {
	ctx := m.joinContext
	if ctx == nil {
		ctx = context.Background()
	}
	operation := "request"
	endpoints = reachableEndpointFirst(endpoints)
	target := endpoints[0]
	endpointIndex := 0
	for {
		m.joinMu.Lock()
		r, ok := m.joinRequests[id]
		if !ok || joinRequestTerminal(r.State) || r.stopState != "" {
			m.joinMu.Unlock()
			return
		}
		deadline := r.ExpiresAt
		m.joinMu.Unlock()
		if m.joinTime().UnixMilli() >= deadline {
			m.joinSnapshot()
			m.wakeJoinRequests()
			return
		}
		wire := joinRequestWire{Version: 1, Operation: operation, RequestID: id}
		if operation == "request" {
			p := m.port
			wire.Port = &p
			wire.Requester = &joinRequester{m.identity.NodeUUID, m.identity.Name}
		}
		remaining := time.Duration(deadline-m.joinTime().UnixMilli()) * time.Millisecond
		if remaining > pairingHTTPTimeout {
			remaining = pairingHTTPTimeout
		}
		callCtx, cancel := context.WithTimeout(ctx, remaining)
		reply, status, err := callJoinRequest(callCtx, target, wire)
		cancel()
		if err != nil {
			if operation == "request" && endpointIndex+1 < len(endpoints) && joinRequestCanTryEndpoint(status, err) {
				endpointIndex++
				target = endpoints[endpointIndex]
				continue
			}
			reason := reply.Reason
			if status == http.StatusNotFound {
				if operation == "request" {
					reason = "unsupported-peer"
				} else {
					reason = joinRequestLost
				}
			}
			if reason == "" {
				reason = inviteFailureReason(err)
				if reason == "" {
					reason = "connection-failed"
				}
			}
			m.failOutgoingJoinRequest(id, reason)
			m.sendJoinRequestCancel(target, id)
			return
		}
		m.inviteMu.Lock()
		m.rosterMu.Lock()
		m.joinMu.Lock()
		m.reconcileJoinRequestsLocked()
		r, ok = m.joinRequests[id]
		if !ok || joinRequestTerminal(r.State) || r.stopState != "" {
			m.joinMu.Unlock()
			m.rosterMu.Unlock()
			m.inviteMu.Unlock()
			m.sendJoinRequestCancel(target, id)
			return
		}
		if operation == "request" {
			host, p, _ := net.SplitHostPort(target)
			r.PeerAddress = host
			r.PeerPort, _ = strconv.Atoi(p)
		}
		if reply.State == "awaiting-requester" && !r.ready {
			solo, reason := m.joinEligibilityLocked("")
			if r.generation != m.clusterGen.Load() || r.sessionGeneration != m.sessGen.Load() {
				reason = "requester-changed"
			}
			if solo && !r.replaceSolo {
				reason = "solo-replacement-required"
			}
			if reply.InviteID == nil || *reply.InviteID == "" {
				reason = "invalid-handoff"
			}
			if reason == "" && solo {
				if err := m.teardownClusterLocalLocked(); err != nil {
					reason = "state-unavailable"
				} else {
					// The request remains memory-only, explicitly re-scoped to the
					// post-teardown generation. No admission is resurrected.
					m.joinRequests[id] = r
					r.generation, r.sessionGeneration = m.clusterGen.Load(), m.sessGen.Load()
					m.emitIdentityChanged()
				}
			}
			if reason != "" {
				m.finishJoinRequestLocked(r, "rejected", reason)
			} else {
				r.InviteID = reply.InviteID
				r.ready = true
				r.State = "awaiting-requester"
				operation = "ready"
			}
		} else {
			operation = "status"
			// Network success is advisory. Only the local exact Invite/receipt
			// can publish paired or an uncertain durable outcome.
			if joinRequestTerminal(reply.State) && reply.State != "paired" {
				r.stopState, r.stopReason = reply.State, reply.Reason
				m.reconcileJoinRequestsLocked()
			}
			if reply.State == "pairing" && r.InviteID != nil {
				r.State = "pairing"
			}
		}
		terminal := joinRequestTerminal(r.State)
		m.joinMu.Unlock()
		m.rosterMu.Unlock()
		m.inviteMu.Unlock()
		m.wakeJoinRequests()
		if terminal {
			m.sendJoinRequestCancel(target, id)
			return
		}
		if operation == "ready" {
			continue
		}
		timer := time.NewTimer(joinRequestPollInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func joinRequestCanTryEndpoint(status int, err error) bool {
	if status == http.StatusNotFound {
		return true
	}
	var network *net.OpError
	// A failed dial cannot have allocated a request at the other endpoint.
	return errors.As(err, &network) && network.Op == "dial"
}

func (m *Manager) failOutgoingJoinRequest(id, reason string) {
	m.inviteMu.Lock()
	m.rosterMu.Lock()
	m.joinMu.Lock()
	m.reconcileJoinRequestsLocked()
	if r := m.joinRequests[id]; r != nil && !joinRequestTerminal(r.State) {
		r.stopState, r.stopReason = "failed", reason
		m.reconcileJoinRequestsLocked()
	}
	m.joinMu.Unlock()
	m.rosterMu.Unlock()
	m.inviteMu.Unlock()
	m.wakeJoinRequests()
}

func (m *Manager) sendJoinRequestCancel(target, id string) {
	ctx, cancel := context.WithTimeout(context.Background(), pairingHTTPTimeout)
	defer cancel()
	_, _, _ = callJoinRequest(ctx, target, joinRequestWire{Version: 1, Operation: "cancel", RequestID: id})
}

func (m *Manager) joinRequestAllowsInvite(inviteID string) bool {
	m.joinMu.Lock()
	defer m.joinMu.Unlock()
	for _, r := range m.joinRequests {
		if r.Direction != "outgoing" || joinRequestTerminal(r.State) {
			continue
		}
		return r.ready && r.stopState == "" && r.ExpiresAt > m.joinTime().UnixMilli() && r.InviteID != nil && *r.InviteID == inviteID
	}
	return true
}
