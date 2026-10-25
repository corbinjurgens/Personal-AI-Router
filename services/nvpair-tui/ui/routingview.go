// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"fmt"
	"strings"
	"time"

	"nvpair-shared/nodepolicy"
	"nvpair-tui/rpc"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/table"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"
)

const routingFooter = "Private to this machine · memory only; broker restart clears history · newest 256 requests"

var (
	routingRefreshKey = key.NewBinding(key.WithKeys("r"), key.WithHelp("r", "refresh history"))
	routingExpandKey  = key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "show/hide attempts"))
	routingBackKey    = key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "back"))
)

// routingView retains rendered text only. The history wire remains the sole
// record owner; each record loan ends synchronously while rebuilding the view.
type routingView struct {
	client        *rpc.Client
	wire          *routingHistoryWire
	namer         *nodeNamer
	table         table.Model
	detail        viewport.Model
	selected      nodepolicy.RoutingRequestKey
	expanded      bool
	state         rpc.HistoryState
	width, height int
}

type routingIdentityMsg struct {
	id  clusterIdentity
	err error
}

func newRoutingView(history *RoutingHistoryClient) *routingView {
	v := &routingView{namer: newNodeNamer(), width: defaultTableWidth,
		table: newTable(routingColumns(defaultTableWidth)), detail: viewport.New(defaultTableWidth, 1)}
	if history != nil {
		v.client, v.wire = history.Client, history.wire
		v.state = v.client.HistoryState()
	}
	return v
}

func routingColumns(width int) []table.Column {
	cols := []column{flexCol("MODEL", 8, 2), flexCol("STATE", 9, 1), fixedCol("ATTEMPTS", 8)}
	if width >= 80 {
		cols = append([]column{fixedCol("WHEN", 8)}, cols...)
		cols = append(cols, flexCol("SERVED", 12, 1))
	}
	if width >= 120 {
		cols = append(cols, flexCol("SPEND", 10, 1), flexCol("REASON", 10, 1))
	}
	return layoutColumns(width, cols)
}

func (v *routingView) Title() string { return "Routing" }

func (v *routingView) Init() tea.Cmd {
	if v.client == nil {
		return nil
	}
	v.client.SetHistorySubscription(true)
	v.syncHistory()
	return nodeIdentityCmd(v.client, func(id clusterIdentity, err error) tea.Msg {
		return routingIdentityMsg{id: id, err: err}
	})
}

func (v *routingView) SetSize(width, height int) {
	if v.width == width && v.height == height {
		return
	}
	v.width, v.height = width, height
	// SetColumns renders immediately. Remove cells laid out for the old
	// column set before shrinking it; selected identity restores the cursor.
	v.table.SetRows(nil)
	v.table.SetColumns(routingColumns(width))
	v.table.SetWidth(max(width, 1))
	v.refreshRows(false)
}

func (v *routingView) clear() {
	v.table.SetRows(nil)
	v.selected = nodepolicy.RoutingRequestKey{}
	v.expanded = false
	v.detail.SetContent("")
}

// HistoryReady is a coalesced scalar, never a record event. Drain it on the
// shell's existing one-second UI tick so no wait chain survives a generation
// or introduces another timer. The transport owns subscribe-before-baseline,
// retry, resubscription and every sequence/terminal/eviction merge.
func (v *routingView) syncHistory() {
	if v.client == nil {
		return
	}
	v.state = v.client.HistoryState()
	if !v.state.Connected {
		v.clear()
		return
	}
	select {
	case generation := <-v.client.HistoryReady():
		// A reconnect can publish a newer token after the state read above.
		// Re-read its generation before discarding that token, otherwise an
		// already-complete new baseline could lose its only ready signal.
		if generation != v.state.Generation {
			v.state = v.client.HistoryState()
		}
		if !v.state.Connected {
			v.clear()
			return
		}
		if generation == v.state.Generation {
			v.refreshRows(false)
		}
	default:
	}
}

func (v *routingView) Update(msg tea.Msg) tea.Cmd {
	switch msg := msg.(type) {
	case DisconnectedMsg:
		v.state = rpc.HistoryState{}
		v.namer = newNodeNamer()
		v.clear()
	case sessionReadyMsg:
		return v.Init()
	case TickMsg:
		v.syncHistory()
	case routingIdentityMsg:
		if msg.err == nil {
			v.namer.setSelf(msg.id)
			v.refreshRows(false)
		}
	case NotificationMsg:
		switch msg.Msg.Method {
		case "discovery:nodes-changed":
			var nodes []availableNode
			if decodeOrLog(msg.Msg.Method, msg.Msg.Params, &nodes) {
				v.namer.learnDiscovered(nodes)
			}
		case "nodes:changed":
			var result struct {
				Nodes []clusterNode `json:"nodes"`
			}
			if decodeOrLog(msg.Msg.Method, msg.Msg.Params, &result) {
				v.namer.learnMembers(result.Nodes)
			}
		default:
			return nil
		}
		v.refreshRows(false)
	case tea.KeyMsg:
		v.syncHistory()
		if key.Matches(msg, routingRefreshKey) {
			if v.client != nil {
				v.client.RefreshHistory()
				v.syncHistory()
			}
			return nil
		}
		if key.Matches(msg, routingExpandKey) {
			if len(v.table.Rows()) > 0 {
				v.expanded = !v.expanded
				v.detail.GotoTop()
				v.refreshRows(false)
			}
			return nil
		}
		if v.expanded {
			if key.Matches(msg, routingBackKey) {
				v.expanded = false
				return nil
			}
			var cmd tea.Cmd
			v.detail, cmd = v.detail.Update(msg)
			return cmd
		}
		var cmd tea.Cmd
		v.table, cmd = v.table.Update(msg)
		v.refreshRows(true)
		return cmd
	}
	return nil
}

func routingState(row nodepolicy.RoutingHistoryRecord) string {
	if row.Recoverable {
		return "Streaming paused, waiting to resume"
	}
	switch row.State {
	case "routing":
		return "Routing"
	case "streaming":
		return "Streaming"
	case "completed":
		return "Completed"
	case "failed":
		return "Failed"
	case "cancelled":
		return "Cancelled"
	default:
		return "Unknown"
	}
}

func routingDisplayReason(row nodepolicy.RoutingHistoryRecord, reason string) string {
	if reason == "stream-interrupted" && (row.State != "failed" || row.Recoverable) {
		return ""
	}
	return routingHistoryReason(reason)
}

func (v *routingView) destination(d *nodepolicy.RoutingHistoryDestination) string {
	if d == nil {
		return "—"
	}
	if d.Kind == "local" {
		return fmt.Sprintf("%s / %s", v.namer.name(*d.NodeID), *d.Engine)
	}
	return fmt.Sprintf("deployment %s / account %s", *d.DeploymentID, *d.AccountID)
}

func (v *routingView) summary(row nodepolicy.RoutingHistoryRecord) string {
	if row.Recoverable {
		return "Streaming paused, waiting to resume"
	}
	count := row.DroppedAttempts + int64(len(row.Attempts))
	switch row.State {
	case "completed":
		plural := "s"
		if count == 1 {
			plural = ""
		}
		return fmt.Sprintf("Routing: %d attempt%s, served on %s", count, plural, v.destination(row.Served))
	case "failed":
		return "Routing failed: " + routingHistoryReason(row.Reason)
	case "cancelled":
		return "Routing cancelled: " + routingHistoryReason(row.Reason)
	case "routing":
		return fmt.Sprintf("Routing: attempt %d", count)
	default:
		return routingState(row)
	}
}

func routingAttemptCount(row nodepolicy.RoutingHistoryRecord) string {
	text := fmt.Sprint(len(row.Attempts))
	if row.DroppedAttempts > 0 {
		text += fmt.Sprintf(" (+%d older)", row.DroppedAttempts)
	}
	return text
}

func routingSpend(row nodepolicy.RoutingHistoryRecord) string {
	s := row.Spend
	return fmt.Sprintf("Ceiling: %s · Cumulative granted: %s · Current held: %s · Charged: %s · Accounting: %s",
		routingHistoryMoney(s.CeilingNanoUSD), routingHistoryMoney(s.GrantedNanoUSD),
		routingHistoryMoney(s.HeldNanoUSD), routingHistoryMoney(s.ChargedNanoUSD), s.Accounting)
}

func (v *routingView) detailText(row nodepolicy.RoutingHistoryRecord) string {
	model := row.Requested.Model
	if model == "" {
		model = "model not specified"
	}
	lines := []string{v.summary(row),
		fmt.Sprintf("Request: %s / %s / %s", row.Request.OriginNodeID, row.Request.ProxyRunID, row.Request.RequestID),
		"When: " + time.UnixMilli(row.CreatedAt).Format(time.RFC3339),
		"Model: " + model, "State: " + routingState(row),
		"Attempts: " + routingAttemptCount(row), "Served: " + v.destination(row.Served),
		routingSpend(row), "Reason: " + routingDisplayReason(row, row.Reason)}
	for i, a := range row.Attempts {
		state, held, charged, accounting := "—", "—", "—", "none"
		if a.Account != nil {
			state, held, charged, accounting = a.Account.State, routingHistoryMoney(a.Account.HeldNanoUSD), routingHistoryMoney(a.Account.ChargedNanoUSD), a.Account.Accounting
		}
		status := "—"
		if a.Status != nil {
			status = fmt.Sprint(*a.Status)
		}
		lines = append(lines, "", fmt.Sprintf("Attempt %d: %s", a.Index+1, routingHistoryAttemptLabel(row, i)),
			"Destination: "+v.destination(&a.Destination), "Model: "+routingHistoryModel(a.Destination.Model))
		if a.Destination.Kind == "local" {
			lines = append(lines, "Node ID: "+*a.Destination.NodeID+" · Engine: "+*a.Destination.Engine)
		}
		lines = append(lines, "Reason: "+routingDisplayReason(row, a.Reason),
			fmt.Sprintf("Admission: %s · First byte: %s · HTTP status: %s", routingHistoryWait(&a.Waits.AdmissionMs), routingHistoryWait(a.Waits.FirstByteMs), status),
			"Account state: "+state,
			fmt.Sprintf("Current held: %s · Charged: %s · Accounting: %s", held, charged, accounting))
	}
	return ansi.Hardwrap(strings.Join(lines, "\n"), max(v.width, 1), true)
}

// Hold the same bank fence for the rendered window and selection. No record
// survives its iteration, and a concurrent upsert cannot shift a row mid-read.
func (v *routingView) refreshRows(moved bool) {
	if v.wire == nil || !v.state.Connected {
		return
	}
	v.wire.mu.Lock()
	defer v.wire.mu.Unlock()
	if v.state.Generation != v.wire.generation {
		return
	}
	rows := make([]table.Row, 0, v.wire.current.count)
	cursor := max(v.table.Cursor(), 0)
	found := -1
	for i := 0; i < v.wire.current.count; i++ {
		row, err := v.wire.current.cells[i].record()
		if err != nil {
			continue
		}
		if moved && i == cursor || v.selected.RequestID == "" && i == cursor {
			v.selected = row.Request
		}
		if row.Request == v.selected {
			found = i
			if v.expanded {
				v.detail.SetContent(v.detailText(row))
			}
		}
		model := row.Requested.Model
		if model == "" {
			model = "model not specified"
		}
		cells := table.Row{model, routingState(row), routingAttemptCount(row)}
		if v.width >= 80 {
			cells = append(table.Row{time.UnixMilli(row.CreatedAt).Format("15:04:05")}, cells...)
			cells = append(cells, v.destination(row.Served))
		}
		if v.width >= 120 {
			cells = append(cells, routingSpend(row), routingDisplayReason(row, row.Reason))
		}
		for j := range cells {
			cells[j] = ansi.Truncate(cells[j], v.table.Columns()[j].Width, "…")
		}
		rows = append(rows, cells)
	}
	v.table.SetRows(rows)
	restoreCursor(&v.table, len(rows))
	if found >= 0 {
		v.table.SetCursor(found)
		// SetCursor rebuilds content but does not scroll Bubbles' viewport.
		// A zero-distance move makes the restored selection visible.
		v.table.MoveDown(0)
	} else {
		v.expanded = false
		v.selected = nodepolicy.RoutingRequestKey{}
		v.detail.SetContent("")
	}
}

func (v *routingView) View() string {
	width, height := max(v.width, 1), max(v.height, 0)
	footer := ansi.Hardwrap(routingFooter, width, true)
	status := ""
	switch {
	case !v.state.Connected:
		status = "Disconnected"
	case v.state.Exhausted:
		status = "Could not load routing history."
	case v.state.Loading:
		status = "Loading routing history…"
	case len(v.table.Rows()) == 0:
		status = "No routed requests yet."
	}
	status = ansi.Hardwrap(status, width, true)
	content := ""
	if v.expanded {
		v.detail.Width, v.detail.Height = width, max(0, height-countLines(footer)-countLines(status))
		if v.detail.Height > 0 {
			content = v.detail.View()
		}
	} else if len(v.table.Rows()) > 0 && fitTable(&v.table, height, status, footer) {
		// SetHeight can also leave the selection just below the viewport.
		v.table.MoveDown(0)
		content = v.table.View()
	}
	lines := strings.Split(joinLines(status, content, footer), "\n")
	if len(lines) > height {
		lines = lines[:height]
	}
	for i := range lines {
		lines[i] = ansi.Truncate(lines[i], width, "…")
	}
	return strings.Join(lines, "\n")
}

func (v *routingView) Help() []key.Binding {
	bindings := []key.Binding{routingRefreshKey, routingExpandKey}
	if v.expanded {
		bindings = append(bindings, routingBackKey)
	}
	return bindings
}
