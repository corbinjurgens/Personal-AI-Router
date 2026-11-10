// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"log"

	"nvpair-shared/nodename"
)

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
