// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"errors"
	"fmt"
	"strings"

	"nvpair-tui/rpc"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"
)

type remoteManagementGrant struct {
	PeerUUID string `json:"peerUuid"`
	Allowed  bool   `json:"allowed"`
	State    string `json:"state"`
}

type remoteManagementSnapshot struct {
	Version  int                     `json:"version"`
	Revision string                  `json:"revision"`
	Grants   []remoteManagementGrant `json:"grants"`
}

type remoteManagementMsg struct {
	epoch    int
	snapshot remoteManagementSnapshot
	err      error
}

type remoteManagementSetMsg struct {
	writeID  int
	epoch    int
	snapshot remoteManagementSnapshot
	err      error
}

var (
	localGrantsKey   = key.NewBinding(key.WithKeys("g"), key.WithHelp("g", "permissions on this device"))
	grantMoveKey     = key.NewBinding(key.WithKeys("up", "down", "k", "j"), key.WithHelp("↑/↓", "peer"))
	grantToggleKey   = key.NewBinding(key.WithKeys(" ", "enter"), key.WithHelp("space/enter", "toggle permission"))
	pairingChoiceKey = key.NewBinding(key.WithKeys("tab"), key.WithHelp("tab", "allow/block management"))
)

func (v *nodesView) pairingChoiceActive() bool {
	return (v.mode == nodesInputPin && v.answering == "") ||
		((v.mode == nodesInputInviteAddress || v.mode == nodesInputInviteChoice) && !v.inviteSending)
}

func (v *nodesView) togglePairingChoice() {
	if !v.pairingChoiceActive() {
		return
	}
	if v.mode == nodesInputPin {
		v.inboundAllow = !v.inboundAllow
	} else {
		v.outboundAllow = !v.outboundAllow
	}
}

func (v *nodesView) pairingChoiceView() string {
	if !v.pairingChoiceActive() {
		return ""
	}
	allow := v.outboundAllow
	if v.mode == nodesInputPin {
		allow = v.inboundAllow
	}
	allowMark, blockMark := "( )", "(*)"
	if allow {
		allowMark, blockMark = "(*)", "( )"
	}
	return ansi.Wrap(joinLines("Remote management of this device (tab to choose)",
		allowMark+" Allow this peer to manage this device (my own devices)",
		blockMark+" Block this peer from changing this device",
		"This choice applies to the device you are using. Reading status and using models this device serves remain available. Engine locks and node pause still apply."), max(1, v.width), "")
}

func (v *nodesView) inviteChoiceName() string {
	for _, row := range v.all {
		if row.key == v.inviteChoiceKey {
			return row.name
		}
	}
	return "selected device"
}

func (v *nodesView) remoteManagementCmd() tea.Cmd {
	v.remoteManagementEpoch++
	epoch := v.remoteManagementEpoch
	return call(v.client, "cluster:get-remote-management", map[string]any{}, func(msg *rpc.Message, err error) tea.Msg {
		out := remoteManagementMsg{epoch: epoch, err: err}
		if err == nil {
			out.err = decodeParams(msg.Result, &out.snapshot)
		}
		return out
	})
}

func (v *nodesView) invalidateRemoteManagement() {
	v.remoteManagementEpoch++
	v.remoteManagementFresh = false
	// A lost reply is never retried. Reattach reads the saved permission.
	v.remoteManagementSetting = false
	v.remoteManagementWriteID++
}

func (v *nodesView) adoptRemoteManagement(snapshot remoteManagementSnapshot) {
	selected := ""
	if v.grantCursor < len(v.remoteManagement.Grants) {
		selected = v.remoteManagement.Grants[v.grantCursor].PeerUUID
	}
	v.remoteManagement = snapshot
	v.remoteManagementFresh = snapshot.Version == 1
	for i, grant := range snapshot.Grants {
		if grant.PeerUUID == selected {
			v.grantCursor = i
			return
		}
	}
	if v.grantCursor >= len(snapshot.Grants) {
		v.grantCursor = max(0, len(snapshot.Grants)-1)
	}
}

func (v *nodesView) handleRemoteManagement(msg remoteManagementMsg) tea.Cmd {
	if msg.epoch != v.remoteManagementEpoch {
		return nil
	}
	v.noteFeed("remote management permissions", msg.err)
	if msg.err != nil {
		v.remoteManagementFresh = false
		return nil
	}
	v.adoptRemoteManagement(msg.snapshot)
	return nil
}

func (v *nodesView) setRemoteManagement(grant remoteManagementGrant) tea.Cmd {
	if v.remoteManagementSetting || !v.remoteManagementFresh || v.remoteManagement.Revision == "" || grant.State == "unavailable" {
		v.status.error("permission unavailable - reload before changing it")
		return nil
	}
	v.remoteManagementSetting = true
	v.remoteManagementWriteID++
	writeID := v.remoteManagementWriteID
	// Do not change the snapshot here: even a failed replacement may have landed.
	// Retire reads started before the write; their result predates its reply.
	v.remoteManagementEpoch++
	epoch := v.remoteManagementEpoch
	params := struct {
		PeerUUID         string `json:"peerUuid"`
		Allowed          bool   `json:"allowed"`
		ExpectedRevision string `json:"expectedRevision"`
	}{grant.PeerUUID, !grant.Allowed, v.remoteManagement.Revision}
	v.status.busy("saving permission for %s on this device...", v.grantName(grant.PeerUUID))
	return call(v.client, "cluster:set-remote-management", params, func(msg *rpc.Message, err error) tea.Msg {
		out := remoteManagementSetMsg{writeID: writeID, epoch: epoch, err: err}
		if err == nil {
			out.err = decodeParams(msg.Result, &out.snapshot)
		}
		return out
	})
}

func (v *nodesView) handleRemoteManagementSet(msg remoteManagementSetMsg) tea.Cmd {
	if msg.writeID != v.remoteManagementWriteID {
		return nil
	}
	// A push or read may overtake this reply. In that case read back instead of
	// replacing a newer snapshot with the command's older result.
	v.remoteManagementSetting = false
	if msg.err != nil {
		v.remoteManagementFresh = false
		var refusal *rpc.RPCError
		if errors.As(msg.err, &refusal) && refusal.Code == -32010 {
			v.status.error("permission revision conflict (-32010) - reloading saved permissions: %s", refusal.Message)
		} else {
			v.status.error("permission change failed - reloading saved permissions: %s", commandFailure(msg.err))
		}
		return v.remoteManagementCmd()
	}
	if msg.epoch != v.remoteManagementEpoch {
		return v.remoteManagementCmd()
	}
	v.adoptRemoteManagement(msg.snapshot)
	v.status.ok("permission saved on this device")
	return nil
}

func permissionState(state string) string {
	switch state {
	case "allowed", "unreviewed", "unavailable", "unsupported":
		return state
	case "denied":
		return "blocked"
	default:
		return "unknown"
	}
}

func (v *nodesView) grantName(uuid string) string {
	for _, row := range v.all {
		if row.key == uuid {
			return row.name
		}
	}
	return uuid
}

func (v *nodesView) grantLine(grant remoteManagementGrant) string {
	state := permissionState(grant.State)
	if !v.remoteManagementFresh {
		state = "unknown (reload required)"
	}
	return fmt.Sprintf("Allow %s to manage this device: %s", truncate(v.grantName(grant.PeerUUID), max(10, v.width-55)), state)
}

func (v *nodesView) selectedGrantLine() string {
	if row := v.selectedRow(); row != nil && !row.self {
		for _, grant := range v.remoteManagement.Grants {
			if grant.PeerUUID == row.key {
				return footerStyle.Render(v.grantLine(grant) + " - g to edit locally")
			}
		}
	}
	return footerStyle.Render("g: remote management permissions on this device")
}

func (v *nodesView) handleGrantKey(msg tea.KeyMsg) tea.Cmd {
	switch msg.String() {
	case "esc", "g":
		v.grantsOpen = false
	case "up", "k":
		v.grantCursor = max(0, v.grantCursor-1)
	case "down", "j":
		v.grantCursor = min(max(0, len(v.remoteManagement.Grants)-1), v.grantCursor+1)
	case "r":
		return v.remoteManagementCmd()
	case " ", "enter":
		if v.grantCursor < len(v.remoteManagement.Grants) {
			return v.setRemoteManagement(v.remoteManagement.Grants[v.grantCursor])
		}
	}
	return nil
}

func (v *nodesView) grantsView() string {
	header := ansi.Wrap(joinLines("Remote management of this device",
		"These permissions let each named peer change THIS device. They do not grant this device permission on a peer.",
		"up/down: peer; space/enter: toggle; r: reload; esc: back"), max(1, v.width), "")
	header = joinLines(v.pairingLine(), header)
	status := v.status.render()
	lines := []string{}
	room := max(1, v.height-countLines(header)-countLines(status)-1)
	start := max(0, v.grantCursor-room+1)
	for i := start; i < len(v.remoteManagement.Grants) && i < start+room; i++ {
		prefix := "  "
		if i == v.grantCursor {
			prefix = "> "
		}
		lines = append(lines, prefix+v.grantLine(v.remoteManagement.Grants[i]))
	}
	if len(lines) == 0 {
		if v.remoteManagementFresh {
			lines = append(lines, "No pinned peers.")
		} else {
			lines = append(lines, "Permissions unavailable - r to reload.")
		}
	}
	return joinLines(header, strings.Join(lines, "\n"), status)
}

type remotePermission struct {
	Version  int    `json:"version"`
	Allowed  bool   `json:"allowed"`
	Revision string `json:"revision"`
	State    string `json:"state"`
}

type detailRemotePermissionMsg struct {
	visit, route int
	generation   int
	permission   remotePermission
	err          error
}

func (d *nodeDetail) remotePermissionCmd() tea.Cmd {
	if !d.remote() || !d.enginesQueryable() || d.policyDisconnected || d.node.presence == presenceOffline {
		return nil
	}
	visit, route := d.telemetryGen, d.policyRoute
	d.remotePermissionGeneration++
	generation := d.remotePermissionGeneration
	return d.own(call(d.client, "engine:remote-management-get", map[string]string{"nodeId": d.node.key}, func(msg *rpc.Message, err error) tea.Msg {
		out := detailRemotePermissionMsg{visit: visit, route: route, generation: generation, err: err}
		if err == nil {
			out.err = decodeParams(msg.Result, &out.permission)
		}
		return out
	}))
}

func (d *nodeDetail) remotePermissionLine() string {
	if !d.remote() {
		return ""
	}
	state := "unknown"
	if d.remotePermissionFresh && d.remotePermission.Version == 1 {
		state = permissionState(d.remotePermission.State)
	}
	return "Remote management: " + state + " (this device on the target; read-only)"
}
