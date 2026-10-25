// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"
)

func routingFixtureView(t *testing.T, rows ...nodepolicy.RoutingHistoryRecord) *routingView {
	t.Helper()
	v := newRoutingView(nil)
	v.wire = newRoutingHistoryWire()
	v.wire.reset(1)
	v.state = rpc.HistoryState{Generation: 1, Connected: true}
	if len(rows) > 0 {
		routingWireApply(t, v.wire, true, rows...)
	}
	v.SetSize(120, 30)
	return v
}

func assertRoutingBounds(t *testing.T, rendered string, width, height int) {
	t.Helper()
	if countLines(rendered) > height {
		t.Fatalf("rendered %d lines, budget %d", countLines(rendered), height)
	}
	for _, line := range strings.Split(rendered, "\n") {
		if ansi.StringWidth(line) > width {
			t.Fatalf("line width %d exceeds %d: %q", ansi.StringWidth(line), width, line)
		}
	}
}

func TestRoutingView(t *testing.T) {
	t.Run("navigation follows Jobs within digit cap", func(t *testing.T) {
		views := defaultViews(nil)
		want := []string{"Nodes", "Jobs", "Routing", "Service", "Errors", "Logs"}
		if len(views) != len(want) || len(views) > maxTabs {
			t.Fatalf("tab count=%d", len(views))
		}
		for i, title := range want {
			if views[i].Title() != title {
				t.Errorf("tab %d=%s want %s", i, views[i].Title(), title)
			}
		}
	})

	t.Run("wait rounding matches desktop toFixed", func(t *testing.T) {
		for _, tc := range []struct {
			ms   int64
			want string
		}{
			{0, "0 ms"}, {350, "350 ms"}, {1000, "1.0 s"}, {1150, "1.1 s"},
			{1234, "1.2 s"}, {1250, "1.3 s"}, {1750, "1.8 s"}, {1999, "2.0 s"},
		} {
			t.Run(strconv.FormatInt(tc.ms, 10), func(t *testing.T) {
				if got := routingHistoryWait(&tc.ms); got != tc.want {
					t.Fatalf("wait=%s want %s", got, tc.want)
				}
			})
		}
	})

	t.Run("maximum retained attempts preserve original numbers", func(t *testing.T) {
		row := routingWireCorpus(t)[0]
		row.Workload = nil
		row.DroppedAttempts = 3
		attempt := row.Attempts[1]
		row.Attempts = make([]nodepolicy.RoutingHistoryAttempt, nodepolicy.RoutingHistoryAttempts)
		for i := range row.Attempts {
			row.Attempts[i] = attempt
			row.Attempts[i].Index = int64(i) + row.DroppedAttempts
			if i < len(row.Attempts)-1 {
				row.Attempts[i].Outcome = "rejected"
			}
		}
		if err := row.Validate(); err != nil {
			t.Fatal(err)
		}
		v := routingFixtureView(t, row)
		text := v.detailText(row)
		for _, want := range []string{"32 (+3 older)", "Attempt 4:", "Attempt 35:", "Routing: 35 attempts"} {
			if !strings.Contains(text, want) {
				t.Errorf("missing %q", want)
			}
		}
		v.Update(tea.KeyMsg{Type: tea.KeyEnter})
		assertRoutingBounds(t, v.View(), 120, 30)
	})
	t.Run("semantic corpus expands selected request", func(t *testing.T) {
		raw, err := os.ReadFile("../../shared/nodepolicy/testdata/routing-request.json")
		if err != nil {
			t.Fatal(err)
		}
		var corpus struct {
			Cases []struct {
				Name   string
				Record nodepolicy.RoutingHistoryRecord
			}
		}
		if err := json.Unmarshal(raw, &corpus); err != nil {
			t.Fatal(err)
		}
		for _, fixture := range corpus.Cases {
			t.Run(fixture.Name, func(t *testing.T) {
				if err := fixture.Record.Validate(); err != nil {
					t.Fatal(err)
				}
				v := routingFixtureView(t, fixture.Record)
				v.Update(tea.KeyMsg{Type: tea.KeyEnter})
				if !v.expanded || v.selected != fixture.Record.Request {
					t.Fatal("enter did not expand selected request")
				}
				text := v.detailText(fixture.Record)
				for _, want := range []string{"Request:", "Model:", "Admission:", "First byte:", "HTTP status:", "Account state:", "Current held:", "Charged:", "Accounting:"} {
					if !strings.Contains(text, want) {
						t.Errorf("detail missing %q", want)
					}
				}
				assertRoutingBounds(t, v.View(), 120, 30)
				v.Update(tea.KeyMsg{Type: tea.KeyEsc})
				if v.expanded {
					t.Fatal("esc did not close attempts")
				}
				v.Update(tea.KeyMsg{Type: tea.KeyEnter})
				v.Update(tea.KeyMsg{Type: tea.KeyEnter})
				if v.expanded {
					t.Fatal("enter did not toggle attempts closed")
				}
			})
		}
	})

	t.Run("newest 256 and eviction floor", func(t *testing.T) {
		base := routingWireCorpus(t)[0]
		base.Workload = nil
		v := routingFixtureView(t)
		for i := 1; i <= 300; i++ {
			row := base
			row.Request.RequestID = strconv.Itoa(i)
			row.CreatedAt += int64(i)
			routingWireApply(t, v.wire, false, row)
		}
		// A delayed baseline must not bring back an evicted older record.
		routingWireApply(t, v.wire, true, base)
		v.refreshRows(false)
		if len(v.table.Rows()) != 256 || v.selected.RequestID != "300" {
			t.Fatalf("rows=%d selected=%+v", len(v.table.Rows()), v.selected)
		}
		v.table.GotoBottom()
		v.refreshRows(true)
		if v.selected.RequestID != "45" {
			t.Fatalf("oldest retained=%s, want 45", v.selected.RequestID)
		}
	})

	t.Run("stale baseline lower seq and terminal cannot reopen", func(t *testing.T) {
		terminal := routingWireCorpus(t)[0]
		v := routingFixtureView(t, terminal)
		stale := terminal
		stale.Seq--
		stale.Requested.Model = "stale-model"
		routingWireApply(t, v.wire, true, stale)
		active := terminal
		active.Seq++
		active.State, active.EndedAt, active.Served = "routing", nil, nil
		routingWireApply(t, v.wire, false, active)
		v.refreshRows(false)
		if got := v.table.Rows()[0]; got[1] != terminal.Requested.Model || got[2] != "Completed" {
			t.Fatalf("terminal changed: %v", got)
		}
	})

	t.Run("full identity preserves expanded selection across insert", func(t *testing.T) {
		row := routingWireCorpus(t)[0]
		v := routingFixtureView(t, row)
		v.Update(tea.KeyMsg{Type: tea.KeyEnter})
		for _, variant := range []string{"origin", "run", "request"} {
			next := row
			next.Workload = nil
			switch variant {
			case "origin":
				next.Request.OriginNodeID += "-other"
			case "run":
				next.Request.ProxyRunID += "-other"
			case "request":
				next.Request.RequestID = "99"
			}
			next.CreatedAt++
			routingWireApply(t, v.wire, false, next)
		}
		v.refreshRows(false)
		if !v.expanded || v.selected != row.Request || v.table.Cursor() != 3 {
			t.Fatalf("selection drifted: %+v cursor=%d", v.selected, v.table.Cursor())
		}
		if len(v.table.Rows()) != 4 {
			t.Fatalf("identities collapsed: %d", len(v.table.Rows()))
		}
	})

	t.Run("recoverability and actual successors own copy", func(t *testing.T) {
		rows := routingWireCorpus(t)
		v := routingFixtureView(t)
		for _, tc := range []struct {
			name         string
			row          nodepolicy.RoutingHistoryRecord
			want, absent string
		}{
			{"recoverable", rows[3], "Streaming paused, waiting to resume", "Stream interrupted"},
			{"linked successor", rows[4], "Interrupted, resumed", "Stream interrupted"},
			{"exhausted interruption", rows[1], "Routing failed: Stream interrupted", "Streaming paused"},
			{"model unknown", rows[5], "model not specified", "Stream interrupted"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				text := v.detailText(tc.row)
				if !strings.Contains(text, tc.want) || strings.Contains(text, tc.absent) {
					t.Fatalf("unexpected copy: %s", text)
				}
			})
		}
		unlinked := rows[3]
		unlinked.Recoverable = false
		if routingState(unlinked) != "Streaming" || routingHistoryAttemptLabel(unlinked, 0) != "Interrupted" {
			t.Fatal("interrupted outcome inferred recovery")
		}
		if routingDisplayReason(unlinked, "synthetic-secret") != "Unknown" {
			t.Fatal("unknown reason echoed")
		}
	})

	t.Run("completed delivery with failed accounting", func(t *testing.T) {
		row := routingWireCorpus(t)[1]
		row.State, row.Reason, row.Served = "completed", "", &row.Attempts[0].Destination
		row.Attempts[0].Outcome, row.Attempts[0].Reason = "committed", ""
		row.Spend.Accounting, row.Attempts[0].Account.Accounting = "failed", "failed"
		if err := row.Validate(); err != nil {
			t.Fatal(err)
		}
		v := routingFixtureView(t, row)
		text := v.detailText(row)
		for _, want := range []string{"State: Completed", "Served", "Accounting: failed", "Ceiling: $1", "Cumulative granted: $0.25", "Current held: $0.25", "Charged: $0.123456789", "deployment synthetic-deployment / account synthetic-account"} {
			if !strings.Contains(text, want) {
				t.Errorf("missing %q in %s", want, text)
			}
		}
	})

	t.Run("long IDs and width fixtures retain scrollable detail", func(t *testing.T) {
		row := routingWireCorpus(t)[1]
		id := strings.Repeat("a", 128)
		model := strings.Repeat("界", 170)
		row.Attempts[0].Destination.AccountID, row.Attempts[0].Destination.DeploymentID = &id, &id
		row.Requested.Model, row.Attempts[0].Destination.Model = model, &model
		if err := row.Validate(); err != nil {
			t.Fatal(err)
		}
		for _, width := range []int{40, 80, 120} {
			t.Run(strconv.Itoa(width), func(t *testing.T) {
				v := routingFixtureView(t, row)
				v.SetSize(width, 18)
				assertRoutingBounds(t, v.View(), width, 18)
				columns := map[string]bool{}
				for _, col := range v.table.Columns() {
					columns[col.Title] = true
				}
				if !columns["MODEL"] || !columns["STATE"] || !columns["ATTEMPTS"] {
					t.Fatalf("priority columns missing: %v", columns)
				}
				v.Update(tea.KeyMsg{Type: tea.KeyEnter})
				assertRoutingBounds(t, v.View(), width, 18)
				v.Update(tea.KeyMsg{Type: tea.KeyPgDown})
				if v.detail.YOffset == 0 {
					t.Fatal("long attempt detail could not scroll")
				}
				assertRoutingBounds(t, v.View(), width, 18)
				// Wrapping must preserve the complete ID rather than ellipsize it.
				if !strings.Contains(strings.ReplaceAll(v.detailText(row), "\n", ""), id) {
					t.Fatal("detail lost full account ID")
				}
			})
		}
	})

	t.Run("empty disconnected loading and generation fence", func(t *testing.T) {
		v := routingFixtureView(t)
		if !strings.Contains(v.View(), "No routed requests yet.") {
			t.Fatal("missing empty copy")
		}
		v.state.Loading = true
		if !strings.Contains(v.View(), "Loading routing history") {
			t.Fatal("missing loading copy")
		}
		row := routingWireCorpus(t)[0]
		routingWireApply(t, v.wire, false, row)
		v.refreshRows(false)
		v.Update(DisconnectedMsg{})
		if len(v.table.Rows()) != 0 || !strings.Contains(v.View(), "Disconnected") {
			t.Fatal("disconnect retained rows")
		}
		v.wire.reset(2)
		raw, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		if err := v.wire.consume(1, false, raw); err == nil {
			t.Fatal("stale generation accepted")
		}
		v.state = rpc.HistoryState{Generation: 2, Connected: true}
		if err := v.wire.consume(2, false, raw); err != nil {
			t.Fatal(err)
		}
		v.refreshRows(false)
		if len(v.table.Rows()) != 1 {
			t.Fatal("new generation did not repopulate")
		}
	})
}

func TestRoutingViewSelectionStaysVisible(t *testing.T) {
	build := func(t *testing.T) *routingView {
		t.Helper()
		base := routingWireCorpus(t)[0]
		base.Workload = nil
		rows := make([]nodepolicy.RoutingHistoryRecord, 60)
		for i := range rows {
			rows[i] = base
			rows[i].Request.RequestID = strconv.Itoa(i + 1)
			rows[i].Requested.Model = "route-" + strconv.Itoa(i)
			rows[i].CreatedAt += int64(i)
		}
		v := routingFixtureView(t, rows...)
		_ = v.View()
		return v
	}
	assertVisible := func(t *testing.T, v *routingView, model string, selected nodepolicy.RoutingRequestKey) {
		t.Helper()
		if v.selected != selected {
			t.Fatalf("selected request changed: %+v", v.selected)
		}
		if !strings.Contains(ansi.Strip(v.View()), model) {
			t.Fatalf("selected model %q is outside visible table:\n%s", model, ansi.Strip(v.View()))
		}
		v.Update(tea.KeyMsg{Type: tea.KeyEnter})
		if !v.expanded || v.selected != selected {
			t.Fatal("enter did not open the visible selection")
		}
	}
	t.Run("resize beyond first page", func(t *testing.T) {
		v := build(t)
		v.table.MoveDown(40)
		v.refreshRows(true)
		model, selected := v.table.SelectedRow()[1], v.selected
		if !strings.Contains(ansi.Strip(v.View()), model) {
			t.Fatal("fixture selection was not initially visible")
		}
		v.SetSize(40, 12)
		assertVisible(t, v, model, selected)
	})
	t.Run("newer insertion above first page bottom", func(t *testing.T) {
		v := build(t)
		v.table.MoveDown(v.table.Height() - 1)
		v.refreshRows(true)
		model, selected := v.table.SelectedRow()[1], v.selected
		if !strings.Contains(ansi.Strip(v.View()), model) {
			t.Fatal("fixture selection was not initially visible")
		}
		newer := routingWireCorpus(t)[0]
		newer.Workload = nil
		newer.Request.RequestID = "61"
		newer.Requested.Model = "newer-model"
		newer.CreatedAt += 60
		routingWireApply(t, v.wire, false, newer)
		v.refreshRows(false)
		assertVisible(t, v, model, selected)
	})
}

func TestRoutingViewClientRefresh(t *testing.T) {
	a, b := net.Pipe()
	t.Cleanup(func() { _ = a.Close(); _ = b.Close() })
	if err := b.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
		t.Fatal(err)
	}
	history, err := NewRoutingHistoryClient(a, func(context.Context) (io.ReadWriteCloser, error) { return nil, io.EOF })
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(history.Close)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	done := make(chan error, 1)
	go func() { done <- history.Run(ctx) }()
	server := rpc.NewCodec(b, b)
	v := newRoutingView(history)
	v.SetSize(120, 30)
	row := routingWireCorpus(t)[0]
	initial, err := json.Marshal(nodepolicy.RoutingHistoryInitial{Requests: []nodepolicy.RoutingHistoryRecord{row}})
	if err != nil {
		t.Fatal(err)
	}
	for _, method := range []string{nodepolicy.MethodRoutingSubscribe, nodepolicy.MethodRoutingGetInitial} {
		request, err := server.Read()
		if err != nil {
			t.Fatal(err)
		}
		if request.Method != method || request.ID == nil {
			t.Fatalf("request=%+v want %s", request, method)
		}
		if err := server.Write(&rpc.Message{JSONRPC: "2.0", ID: request.ID, Result: initial}); err != nil {
			t.Fatal(err)
		}
		if err := server.Write(&rpc.Message{JSONRPC: "2.0", Method: "routing-view-barrier"}); err != nil {
			t.Fatal(err)
		}
		select {
		case event := <-history.Events():
			if event.Message == nil || event.Message.Method != "routing-view-barrier" {
				t.Fatalf("event=%+v", event)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("history consumption stalled")
		}
		v.Update(TickMsg{})
		if len(v.table.Rows()) != 1 || v.state.Loading {
			t.Fatalf("view did not consume ready token: %+v", v.state)
		}
		if method == nodepolicy.MethodRoutingSubscribe {
			v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'r'}})
			if !v.state.Loading || len(v.table.Rows()) != 1 {
				t.Fatal("refresh lost accepted rows or did not start loading")
			}
		}
	}
	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("client did not detach")
	}
}

func TestRoutingViewClientReconnect(t *testing.T) {
	a, b := net.Pipe()
	secondA, secondB := net.Pipe()
	t.Cleanup(func() { _ = a.Close(); _ = b.Close(); _ = secondA.Close(); _ = secondB.Close() })
	for _, peer := range []net.Conn{b, secondB} {
		if err := peer.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
			t.Fatal(err)
		}
	}
	connections := make(chan io.ReadWriteCloser, 1)
	connections <- secondA
	history, err := NewRoutingHistoryClient(a, func(ctx context.Context) (io.ReadWriteCloser, error) {
		select {
		case conn := <-connections:
			return conn, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(history.Close)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	done := make(chan error, 1)
	go func() { done <- history.Run(ctx) }()
	v := newRoutingView(history)
	v.SetSize(120, 30)
	row := routingWireCorpus(t)[0]
	event := func(kind rpc.EventKind, generation uint64) {
		t.Helper()
		select {
		case got := <-history.Events():
			if got.Kind != kind || got.Generation != generation {
				t.Fatalf("event=%+v want kind=%v generation=%d", got, kind, generation)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("service event did not arrive")
		}
	}
	for i, peer := range []net.Conn{b, secondB} {
		generation := uint64(i + 1)
		server := rpc.NewCodec(peer, peer)
		request, err := server.Read()
		if err != nil {
			t.Fatal(err)
		}
		if request.Method != nodepolicy.MethodRoutingSubscribe || request.ID == nil {
			t.Fatalf("reconnect did not subscribe: %+v", request)
		}
		row.Request.ProxyRunID = "synthetic-run-" + strconv.Itoa(i+1)
		row.Workload = nil
		initial, err := json.Marshal(nodepolicy.RoutingHistoryInitial{Requests: []nodepolicy.RoutingHistoryRecord{row}})
		if err != nil {
			t.Fatal(err)
		}
		if err := server.Write(&rpc.Message{JSONRPC: "2.0", ID: request.ID, Result: initial}); err != nil {
			t.Fatal(err)
		}
		if err := server.Write(&rpc.Message{JSONRPC: "2.0", Method: "app:ready"}); err != nil {
			t.Fatal(err)
		}
		event(rpc.EventNotification, generation)
		// Init returns only the ordinary identity command; the history client
		// has already resubscribed and consumed its baseline independently.
		v.Update(sessionReadyMsg{})
		v.Update(TickMsg{})
		if len(v.table.Rows()) != 1 || v.selected != row.Request {
			t.Fatalf("generation %d view=%+v rows=%d", generation, v.selected, len(v.table.Rows()))
		}
		if i == 0 {
			if err := peer.Close(); err != nil {
				t.Fatal(err)
			}
			event(rpc.EventDisconnected, generation)
			v.Update(DisconnectedMsg{})
			if len(v.table.Rows()) != 0 || !strings.Contains(v.View(), "Disconnected") {
				t.Fatal("disconnect did not clear rendered history")
			}
		}
	}
	stale, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	if err := history.wire.consume(1, false, stale); err == nil {
		t.Fatal("old generation reached reconnected bank")
	}
	v.Update(TickMsg{})
	if v.selected.ProxyRunID != "synthetic-run-2" {
		t.Fatal("stale result changed new session")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("reconnecting client did not detach")
	}
}
