// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"

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
// this node has granted that peer remote management. It logs one audit line
// per authenticated request (caller, decision, permission revision), never the
// body.
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
	perm := m.remoteManagementFor(uuid)
	if !perm.Allowed {
		log.Printf("peer rename: caller %s denied (state %s, revision %s)", uuid, perm.State, perm.Revision)
		writePeerJSON(w, http.StatusConflict, peerNodeNameError{Code: remoteManagementDeniedWire, Message: remoteManagementDeniedText})
		return
	}
	log.Printf("peer rename: caller %s allowed (revision %s)", uuid, perm.Revision)
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

// peerNicknameView is the cluster:set-peer-nickname result.
type peerNicknameView struct {
	NodeUUID    string  `json:"nodeUuid"`
	Nickname    *string `json:"nickname"`
	DisplayName string  `json:"displayName"`
}

// decodePeerTarget decodes strict {nodeUuid, <valueKey>: string|null} params
// and checks that nodeUuid names a current member other than this node. On
// failure it returns the -32602 message and data to send.
func (m *Manager) decodePeerTarget(params json.RawMessage, valueKey string) (uuid string, value *string, errMsg string, errData map[string]string) {
	fields, err := decodeRemoteManagementParams(params, map[string]bool{"nodeUuid": true, valueKey: true})
	rawValue, hasValue := fields[valueKey]
	if err != nil || len(fields) != 2 || !hasValue ||
		json.Unmarshal(fields["nodeUuid"], &uuid) != nil || json.Unmarshal(rawValue, &value) != nil {
		return "", nil, "invalid params", nil
	}
	nodeField := map[string]string{"field": "nodeUuid"}
	if uuid == m.identity.NodeUUID {
		return "", nil, "nodeUuid is this node", nodeField
	}
	if member, ok := m.memberByUUID(uuid); !ok || member.State != stateMember {
		return "", nil, "nodeUuid is not a cluster member", nodeField
	}
	return uuid, value, "", nil
}

// handleSetPeerNickname sets (string) or clears (null) the nickname this node
// uses for a peer. Params are exactly {nodeUuid, nickname}. The nickname lives
// on the member record in members.json and never leaves this node.
func (m *Manager) handleSetPeerNickname(msg *Message) {
	uuid, raw, errMsg, errData := m.decodePeerTarget(msg.Params, "nickname")
	if errMsg != "" {
		m.codec.RespondErrorData(msg.ID, codeInvalidParams, errMsg, errData)
		return
	}
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
