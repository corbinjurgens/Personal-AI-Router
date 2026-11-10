// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"

	"nvpair-shared/httpcon"
	"nvpair-shared/nodename"
)

// peerNodeNamePath is the mTLS route on which a peer asks this node to rename
// itself. Only a caller this node has granted remote management (state
// allowed) may use it; this node stays the only writer of its own name.
const peerNodeNamePath = "/v1/cluster/node-name"

// Peer error bodies for peerNodeNamePath. The denial code and message match
// engine-manager's remote-management refusal so clients show one text.
const (
	peerNameInvalidCode        = "invalid_name"
	remoteManagementDeniedWire = "remote_management_denied"
	remoteManagementDeniedText = "Remote management is blocked on this device. Change permission locally on that device."
	peerNameUnavailableText    = "the device is offline or does not trust this node"
)

// peerNodeNameRequest is the exact PUT body: {"name": string|null}.
type peerNodeNameRequest struct {
	Name json.RawMessage `json:"name"`
}

// peerNodeNameError is the JSON body of a 400 or 409 answer.
type peerNodeNameError struct {
	Code    string `json:"code"`
	Reason  string `json:"reason,omitempty"`
	Message string `json:"message"`
}

func writePeerJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// decodePeerNodeNameBody parses the exact {"name": string|null} body.
func decodePeerNodeNameBody(body []byte) (*string, bool) {
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	var req peerNodeNameRequest
	if err := dec.Decode(&req); err != nil || req.Name == nil {
		return nil, false
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, false
	}
	var name *string
	if err := json.Unmarshal(req.Name, &name); err != nil {
		return nil, false
	}
	return name, true
}

// handlePeerNodeName applies a pinned peer's request to rename this node, when
// this node has granted that peer remote management. Order: method, pin,
// bounded body decode, grant, rename. It logs one audit line per authenticated
// well-formed request (caller, decision, permission revision), never the body.
func (m *Manager) handlePeerNodeName(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	uuid, ok := m.verifyClientPin(r)
	if !ok {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}
	// Read the body before the grant check, so the grant is evaluated right
	// before the rename and a slow body cannot widen the gap after a revoke.
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes))
	if err != nil {
		http.Error(w, "read body", http.StatusBadRequest)
		return
	}
	custom, ok := decodePeerNodeNameBody(body)
	if !ok {
		writePeerJSON(w, http.StatusBadRequest, peerNodeNameError{Code: peerNameInvalidCode, Message: "body must be exactly {\"name\": string or null}"})
		return
	}
	perm := m.remoteManagementFor(uuid)
	if !perm.Allowed {
		log.Printf("peer rename: caller %s denied (state %s, revision %s)", uuid, perm.State, perm.Revision)
		writePeerJSON(w, http.StatusConflict, peerNodeNameError{Code: remoteManagementDeniedWire, Message: remoteManagementDeniedText})
		return
	}
	log.Printf("peer rename: caller %s allowed (revision %s)", uuid, perm.Revision)
	view, err := m.setSelfName(custom)
	if reason := nodename.Reason(err); reason != "" {
		writePeerJSON(w, http.StatusBadRequest, peerNodeNameError{Code: peerNameInvalidCode, Reason: reason, Message: err.Error()})
		return
	}
	if err != nil {
		log.Printf("peer rename: caller %s: save name: %v", uuid, err)
		http.Error(w, "could not save the name", http.StatusInternalServerError)
		return
	}
	writePeerJSON(w, http.StatusOK, view)
}

// handleSetPeerName asks a member to rename itself. Params are exactly
// {nodeUuid, name: string|null}. The target applies the name only when it has
// granted this node remote management; on its 200 answer this node records the
// returned name for that member at once, through the same received-name path
// as a roster refresh. The target's roster push repairs anything missed here.
func (m *Manager) handleSetPeerName(msg *Message) {
	member, name, errMsg, errData := m.decodePeerTarget(msg.Params, "name")
	if errMsg != "" {
		m.codec.RespondErrorData(msg.ID, codeInvalidParams, errMsg, errData)
		return
	}
	uuid := member.NodeUUID
	body, err := json.Marshal(map[string]*string{"name": name})
	if err != nil {
		m.codec.RespondErrorData(msg.ID, codeInternalError, err.Error(), nil)
		return
	}
	// The removal client keeps fresh connections, so a lost reply is never
	// replayed as a second rename.
	client, err := m.removalPeerClient(uuid)
	if err != nil {
		log.Printf("peer rename: %s: %v", uuid, err)
		m.codec.RespondErrorData(msg.ID, codePeerUnavailable, peerNameUnavailableText, nil)
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), pairingHTTPTimeout)
	defer cancel()
	addrs := m.resolvePeerAddrs(*member)
	for _, addr := range confirmedFirst(addrs, m.peerAddrs.ChooseWithin(ctx, uuid, addrs)) {
		if !m.selectPeerServiceAddress(uuid, addr) {
			continue
		}
		status, answer, answered := sendPeerRename(ctx, client, addr, body)
		if answered {
			m.answerPeerRename(msg, uuid, status, answer)
			return
		}
		m.peerAddrs.Forget(uuid)
		if ctx.Err() != nil {
			break
		}
	}
	m.codec.RespondErrorData(msg.ID, codePeerUnavailable, peerNameUnavailableText, nil)
}

// sendPeerRename sends one rename request to addr. answered is false only
// when the peer never responded there.
func sendPeerRename(ctx context.Context, client *http.Client, addr string, body []byte) (status int, answer []byte, answered bool) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, "https://"+addr+peerNodeNamePath, bytes.NewReader(body))
	if err != nil {
		return 0, nil, false
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("peer rename: %s: %v", addr, err)
		return 0, nil, false
	}
	defer httpcon.DrainAndClose(resp.Body)
	answer, err = io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	if err != nil {
		return 0, nil, false
	}
	return resp.StatusCode, answer, true
}

// answerPeerRename maps the target's answer to the local RPC response.
func (m *Manager) answerPeerRename(msg *Message, uuid string, status int, answer []byte) {
	var peerErr peerNodeNameError
	switch status {
	case http.StatusOK:
		var view nodeNameView
		if err := json.Unmarshal(answer, &view); err != nil || view.NodeUUID != uuid {
			log.Printf("peer rename: %s sent an unusable answer", uuid)
			m.codec.RespondErrorData(msg.ID, codeInternalError, "the device sent an unusable answer", nil)
			return
		}
		if m.refreshMemberIdentity(&RosterEntry{NodeUUID: uuid, Name: view.Name}) {
			m.emitNodesChanged()
		}
		m.codec.Respond(msg.ID, view)
	case http.StatusConflict:
		_ = json.Unmarshal(answer, &peerErr)
		if peerErr.Message == "" {
			peerErr.Message = remoteManagementDeniedText
		}
		m.codec.RespondErrorData(msg.ID, codeRemoteManagementDenied, peerErr.Message, nil)
	case http.StatusBadRequest:
		_ = json.Unmarshal(answer, &peerErr)
		if peerErr.Message == "" {
			peerErr.Message = "the device rejected the name"
		}
		data := map[string]string{"field": "name"}
		if peerErr.Reason != "" {
			data["reason"] = peerErr.Reason
		}
		m.codec.RespondErrorData(msg.ID, codeInvalidParams, peerErr.Message, data)
	case http.StatusForbidden:
		m.codec.RespondErrorData(msg.ID, codePeerUnavailable, peerNameUnavailableText, nil)
	default:
		log.Printf("peer rename: %s answered status %d", uuid, status)
		m.codec.RespondErrorData(msg.ID, codeInternalError, "the device could not apply the name", nil)
	}
}

// peerNicknameView is the cluster:set-peer-nickname result.
type peerNicknameView struct {
	NodeUUID    string  `json:"nodeUuid"`
	Nickname    *string `json:"nickname"`
	DisplayName string  `json:"displayName"`
}

// decodePeerTarget decodes strict {nodeUuid, <valueKey>: string|null} params
// and returns the member nodeUuid names, which must be a current member other
// than this node. On failure it returns the -32602 message and data to send.
func (m *Manager) decodePeerTarget(params json.RawMessage, valueKey string) (member *ClusterNode, value *string, errMsg string, errData map[string]string) {
	var uuid string
	fields, err := decodeRemoteManagementParams(params, map[string]bool{"nodeUuid": true, valueKey: true})
	rawValue, hasValue := fields[valueKey]
	if err != nil || len(fields) != 2 || !hasValue ||
		json.Unmarshal(fields["nodeUuid"], &uuid) != nil || json.Unmarshal(rawValue, &value) != nil {
		return nil, nil, "invalid params", nil
	}
	nodeField := map[string]string{"field": "nodeUuid"}
	if uuid == m.identity.NodeUUID {
		return nil, nil, "nodeUuid is this node", nodeField
	}
	member, ok := m.memberByUUID(uuid)
	if !ok || member.State != stateMember {
		return nil, nil, "nodeUuid is not a cluster member", nodeField
	}
	return member, value, "", nil
}

// handleSetPeerNickname sets (string) or clears (null) the nickname this node
// uses for a peer. Params are exactly {nodeUuid, nickname}. The nickname lives
// on the member record in members.json and never leaves this node.
func (m *Manager) handleSetPeerNickname(msg *Message) {
	target, raw, errMsg, errData := m.decodePeerTarget(msg.Params, "nickname")
	if errMsg != "" {
		m.codec.RespondErrorData(msg.ID, codeInvalidParams, errMsg, errData)
		return
	}
	uuid := target.NodeUUID
	nickname := ""
	if raw != nil {
		normalized, err := nodename.Normalize(*raw)
		if err != nil {
			m.codec.RespondErrorData(msg.ID, codeInvalidParams, err.Error(),
				map[string]string{"field": "nickname", "reason": nodename.Reason(err)})
			return
		}
		nickname = normalized
	}
	member, found, changed := m.setMemberNickname(uuid, nickname)
	if !found {
		m.codec.RespondErrorData(msg.ID, codeInvalidParams, "nodeUuid is not a cluster member",
			map[string]string{"field": "nodeUuid"})
		return
	}
	if changed {
		if err := m.persistMembersErr(); err != nil {
			log.Printf("persist nickname for %s: %v", uuid, err)
			m.codec.RespondErrorData(msg.ID, codeInternalError, "could not save the nickname", nil)
			return
		}
		m.emitNodesChanged()
	}
	m.codec.Respond(msg.ID, peerNicknameView{
		NodeUUID:    uuid,
		Nickname:    optionalString(member.Nickname),
		DisplayName: member.DisplayName,
	})
}
