// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	svcerrors "nvpair-shared/errors"
	"nvpair-tui/rpc"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// reconnectUIFixture uses only in-memory pipes. The test controls replacement
// and consumes the production event stream through the shell's command seam.
type reconnectUIFixture struct {
	client  *rpc.Client
	peer    net.Conn
	broker  *rpc.Codec
	dials   chan io.ReadWriteCloser
	mu      sync.Mutex
	methods map[string]int
}

func newReconnectUIFixture(t *testing.T) *reconnectUIFixture {
	t.Helper()
	conn, peer := net.Pipe()
	f := &reconnectUIFixture{peer: peer, broker: rpc.NewCodec(peer, peer),
		dials: make(chan io.ReadWriteCloser), methods: map[string]int{}}
	f.client = rpc.NewReconnectingClient(conn, func(ctx context.Context) (io.ReadWriteCloser, error) {
		select {
		case conn := <-f.dials:
			return conn, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	})
	done := make(chan error, 1)
	go func() { done <- f.client.Run(context.Background()) }()
	t.Cleanup(func() {
		f.client.Close()
		_ = f.peer.Close()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Error("UI RPC session did not close")
		}
	})
	return f
}

func (f *reconnectUIFixture) event(t *testing.T) tea.Msg {
	t.Helper()
	result := make(chan tea.Msg, 1)
	go func() { result <- waitForNotification(f.client)() }()
	select {
	case msg := <-result:
		return msg
	case <-time.After(time.Second):
		t.Fatal("missing ordered service event")
		return nil
	}
}

func (f *reconnectUIFixture) replace(t *testing.T, m *Model) tea.Cmd {
	t.Helper()
	if err := f.peer.Close(); err != nil {
		t.Fatal(err)
	}
	updated, _ := m.Update(f.event(t))
	*m = updated.(Model)
	if !m.disconnected || m.ready {
		t.Fatal("transport loss did not mark the existing shell disconnected")
	}
	conn, peer := net.Pipe()
	f.peer, f.broker = peer, rpc.NewCodec(peer, peer)
	select {
	case f.dials <- conn:
	case <-time.After(time.Second):
		t.Fatal("client did not dial replacement")
	}
	if err := f.broker.Write(&rpc.Message{JSONRPC: "2.0", Method: "app:ready", Params: json.RawMessage(`{"version":"replacement"}`)}); err != nil {
		t.Fatal(err)
	}
	updated, cmd := m.Update(f.event(t))
	*m = updated.(Model)
	return cmd
}

func (f *reconnectUIFixture) serve(t *testing.T, results map[string]json.RawMessage) {
	t.Helper()
	broker := f.broker
	go func() {
		for {
			req, err := broker.Read()
			if err != nil {
				return
			}
			f.mu.Lock()
			f.methods[req.Method]++
			f.mu.Unlock()
			result := results[req.Method]
			if result == nil {
				result = json.RawMessage(`{}`)
			}
			if err := broker.Write(&rpc.Message{JSONRPC: "2.0", ID: req.ID, Result: result}); err != nil {
				return
			}
		}
	}()
}

func (f *reconnectUIFixture) count(method string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.methods[method]
}

func runSessionUICommand(t *testing.T, m *Model, cmd tea.Cmd) {
	t.Helper()
	if cmd == nil {
		return
	}
	result := make(chan tea.Msg, 1)
	go func() { result <- cmd() }()
	var msg tea.Msg
	select {
	case msg = <-result:
	case <-time.After(time.Second):
		t.Fatal("session refresh blocked or restarted a UI timer")
	}
	if batch, ok := msg.(tea.BatchMsg); ok {
		for _, child := range batch {
			runSessionUICommand(t, m, child)
		}
		return
	}
	updated, next := m.Update(msg)
	*m = updated.(Model)
	runSessionUICommand(t, m, next)
}

type readyProbeView struct {
	stubView
	refreshes int
}

func (v *readyProbeView) Update(msg tea.Msg) tea.Cmd {
	if _, ok := msg.(sessionReadyMsg); ok {
		v.refreshes++
	}
	return nil
}

func TestSessionReadyRebootstrapsExistingViewsOnce(t *testing.T) {
	f := newReconnectUIFixture(t)
	nodes, jobs, errs, service := newNodesView(f.client), newJobsView(f.client), newErrorsView(f.client), newServiceView(f.client)
	probe := &readyProbeView{}
	m := New(f.client, nil, []View{nodes, jobs, errs, service, probe})
	nodes.filter = "keep-filter"
	nodes.mode = nodesInputManualAddress
	nodes.input.SetValue("unsent-address")
	nodes.feeds.members = []clusterNode{{ID: "removed", NodeUUID: "removed", Name: "keep-filter-old"}}
	nodes.rebuild()
	if len(nodes.rows) != 1 {
		t.Fatal("old Nodes fixture did not contain its removable row")
	}
	jobs.showAll = true
	jobs.upsert(workload{ID: "removed", OriginatedFrom: "old", State: "running"})
	jobs.baselined = true
	errs.pushed = true
	errs.setErrors([]svcerrors.ServiceError{{ID: "removed"}})
	service.errsPushed, service.errsKnown, service.pinged = true, true, true
	service.editing = true
	service.input.SetValue("unsent-name")
	cmd := f.replace(t, &m)
	if m.disconnected || !m.ready || m.brokerVersion != "replacement" || probe.refreshes != 1 {
		t.Fatalf("ready state = disconnected:%v ready:%v version:%q refreshes:%d", m.disconnected, m.ready, m.brokerVersion, probe.refreshes)
	}
	if len(nodes.all) != 0 || len(jobs.byKey) != 0 || jobs.baselined || len(errs.errs) != 0 || errs.pushed || service.errsKnown || service.errsPushed || service.pinged {
		t.Fatal("old authoritative collections or precedence survived replacement")
	}
	for _, row := range service.workers.Rows() {
		if row[1] != "?" {
			t.Fatalf("unread replacement worker status = %q, want unknown", row[1])
		}
	}
	f.serve(t, map[string]json.RawMessage{
		"workloads:get-initial": json.RawMessage(`{"workloads":[{"id":"fresh","originatedFrom":"new","state":"running"}]}`),
		"errors:get-initial":    json.RawMessage(`[{"id":"fresh-error"}]`),
		"cluster:get-node-id":   json.RawMessage(`{"nodeId":"new-id","nodeUuid":"new-uuid"}`),
		"policy:get":            json.RawMessage(`{"availability":"paused"}`),
		"ping":                  json.RawMessage(`{"version":"new-ping","uptime_ms":10}`),
	})
	batch, ok := cmd().(tea.BatchMsg)
	if !ok || len(batch) < 2 {
		t.Fatalf("ready returned %T, want refreshes plus event wait", cmd())
	}
	// The final command is the single event wait. The other commands are the
	// actual shell/view rebootstrap and all of their result continuations.
	for _, child := range batch[:len(batch)-1] {
		runSessionUICommand(t, &m, child)
	}
	if len(jobs.byKey) != 1 || jobs.byKey[jobs.order[0]].ID != "fresh" || !jobs.baselined || !jobs.subscribed {
		t.Fatalf("replacement jobs = %+v, baselined:%v subscribed:%v", jobs.byKey, jobs.baselined, jobs.subscribed)
	}
	if len(errs.errs) != 1 || errs.errs[0].ID != "fresh-error" || !service.errsKnown || service.brokerVersion != "new-ping" || m.avail.value != "paused" || m.avail.selfID != "new-id" || m.avail.selfUUID != "new-uuid" {
		t.Fatal("replacement baselines or shell policy did not apply")
	}
	if nodes.filter != "keep-filter" || nodes.input.Value() != "unsent-address" || !jobs.showAll || !service.editing || service.input.Value() != "unsent-name" {
		t.Fatal("recovery discarded presentation or unsent text")
	}
	for _, method := range []string{"discovery:subscribe", "engine:subscribe", "workloads:subscribe", "workloads:get-initial", "settings/get-cluster-friendly-name"} {
		want := 1
		if method == "settings/get-cluster-friendly-name" {
			want = 2 // Nodes label and Service setting have independent owners.
		}
		if got := f.count(method); got != want {
			t.Errorf("%s calls = %d, want %d", method, got, want)
		}
	}
	updated, _ := m.Update(serviceEventMsg{event: rpc.Event{Kind: rpc.EventNotification, Generation: 2, Message: &rpc.Message{Method: "app:ready"}}})
	m = updated.(Model)
	if probe.refreshes != 1 || len(jobs.byKey) != 1 {
		t.Fatal("repeated ready reset state or scheduled a second refresh")
	}
}

func TestSessionBrokerRespawnRetainsExistingReadyRecovery(t *testing.T) {
	f := newReconnectUIFixture(t)
	nodes := newNodesView(f.client)
	d := newNodeDetail(f.client, nodeRow{key: "self", self: true})
	nodes.detail = d
	d.policyFresh = true
	probe := &readyProbeView{}
	m := New(f.client, nil, []View{nodes, probe})
	m.ready = true
	updated, _ := m.Update(serviceEventMsg{event: rpc.Event{Kind: rpc.EventNotification, Generation: 1, Message: &rpc.Message{Method: "service/broker-restarted"}}})
	m = updated.(Model)
	if m.ready || d.policyFresh {
		t.Fatal("broker respawn retained old readiness or policy")
	}
	updated, _ = m.Update(serviceEventMsg{event: rpc.Event{Kind: rpc.EventNotification, Generation: 1, Message: &rpc.Message{Method: "app:ready"}}})
	m = updated.(Model)
	if !m.ready || !d.policyRefresh || d.performanceLease == nil || probe.refreshes != 0 {
		t.Fatal("same-transport broker ready lost its existing recovery path")
	}
	d.closePerformance()
}

func TestSessionDropsCompletedOldResultAfterFreshBaseline(t *testing.T) {
	f := newReconnectUIFixture(t)
	f.serve(t, map[string]json.RawMessage{"workloads:get-initial": json.RawMessage(`{"workloads":[{"id":"obsolete","originatedFrom":"old"}]}`)})
	jobs := newJobsView(f.client)
	m := New(f.client, nil, []View{jobs})
	noop := func() tea.Msg { return nil }
	old := sessionCommand(f.client, 1, tea.Batch(tea.Batch(jobs.loadCmd(), noop), noop))
	outer := old().(tea.BatchMsg)
	inner := outer[0]().(tea.BatchMsg)
	completed := inner[0]() // RPC completed; application is deliberately delayed.
	f.replace(t, &m)
	jobs.Update(workloadsLoadedMsg{workloads: []workload{{ID: "fresh", OriginatedFrom: "new"}}})
	updated, cmd := m.Update(completed)
	m = updated.(Model)
	if cmd != nil || len(jobs.byKey) != 1 || jobs.byKey[jobs.order[0]].ID != "fresh" {
		t.Fatalf("obsolete completed result replaced fresh baseline: %+v", jobs.byKey)
	}
}

func TestSessionDelayedCommandsCannotMutateReplacement(t *testing.T) {
	f := newReconnectUIFixture(t)
	m := New(f.client, nil, nil)
	oldAvailability := setAvailabilityCmd(f.client, "paused")
	oldReset := resetEngines(f.client)
	oldPaired := pollPairedTelemetryCmd(f.client, "peer", 1, "peer")
	f.replace(t, &m)
	f.serve(t, nil)
	if msg := oldAvailability().(setAvailabilityMsg); msg.err == nil {
		t.Fatal("old availability command succeeded on replacement")
	}
	if _, ok := oldReset().(resetHaltedMsg); !ok {
		t.Fatal("old reset command reached wipe completion")
	}
	if msg := oldPaired().(nodeTelemetryMsg); msg.err == nil {
		t.Fatal("old paired telemetry command reached replacement")
	}
	if got := f.count("node:set-availability") + f.count("engine:uninstall-managed") + f.count("discovery:get-node-info"); got != 0 {
		t.Fatalf("obsolete commands wrote %d replacement requests", got)
	}
}

func TestSessionRecoveryRetainsOneServiceAndManualTimerChain(t *testing.T) {
	f := newReconnectUIFixture(t)
	nodes, service := newNodesView(f.client), newServiceView(f.client)
	m := New(f.client, nil, []View{nodes, service})
	oldServiceTick := sessionCommand(f.client, 1, func() tea.Msg { return serviceTickMsg{} })
	oldManualTick := sessionCommand(f.client, 1, func() tea.Msg { return manualTickMsg{} })
	oldRenderTick := sessionCommand(f.client, 1, func() tea.Msg { return TickMsg{} })
	f.replace(t, &m)
	if _, ok := oldServiceTick().(serviceTickMsg); !ok {
		t.Fatal("transport replacement retired the UI-lifetime service timer")
	}
	if _, ok := oldManualTick().(manualTickMsg); !ok {
		t.Fatal("transport replacement retired the UI-lifetime manual timer")
	}
	testTimer := func(name string, msg tea.Msg, isTick func(tea.Msg) bool) {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				updated, cmd := m.Update(msg)
				m = updated.(Model)
				batch, ok := cmd().(tea.BatchMsg)
				if !ok || len(batch) != 2 {
					t.Fatalf("timer continuation = %T, want one refresh and one successor", cmd())
				}
				if tick := batch[1](); !isTick(tick) {
					t.Fatalf("successor = %T, want this owner's timer", tick)
				}
			})
		})
	}
	testTimer("service timer", serviceTickMsg{}, func(msg tea.Msg) bool { _, ok := msg.(serviceTickMsg); return ok })
	testTimer("manual timer", manualTickMsg{}, func(msg tea.Msg) bool { _, ok := msg.(manualTickMsg); return ok })
	t.Run("render timer", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			updated, render := m.Update(oldRenderTick())
			m = updated.(Model)
			if render == nil {
				t.Fatal("replacement retired the shared render timer")
			}
			if _, ok := render().(TickMsg); !ok {
				t.Fatal("render tick did not rearm exactly one UI-lifetime successor")
			}
		})
	})
}

func TestSessionNotificationArmsOneEventWait(t *testing.T) {
	f := newReconnectUIFixture(t)
	m := New(f.client, nil, nil)
	if err := f.broker.Write(&rpc.Message{JSONRPC: "2.0", Method: "first:push"}); err != nil {
		t.Fatal(err)
	}
	updated, wait := m.Update(f.event(t))
	m = updated.(Model)
	if err := f.broker.Write(&rpc.Message{JSONRPC: "2.0", Method: "second:push"}); err != nil {
		t.Fatal(err)
	}
	msg := wait()
	event, ok := msg.(serviceEventMsg)
	if !ok || event.event.Message == nil || event.event.Message.Method != "second:push" {
		t.Fatalf("push continuation = %T, want one event wait, not a batch of competing consumers", msg)
	}
}

// stubView renders a fixed block of rows, optionally far more (and far wider)
// than the size it was handed, standing in for a view whose own height
// accounting is wrong.
type stubView struct {
	title string
	rows  int
	width int
}

func (s *stubView) Title() string          { return s.title }
func (s *stubView) Init() tea.Cmd          { return nil }
func (s *stubView) SetSize(_, _ int)       {}
func (s *stubView) Update(tea.Msg) tea.Cmd { return nil }
func (s *stubView) Help() []key.Binding    { return nil }
func (s *stubView) View() string {
	line := strings.Repeat("x", s.width)
	lines := make([]string, s.rows)
	for i := range lines {
		lines[i] = line
	}
	return strings.Join(lines, "\n")
}

func newTestModel(views ...View) Model {
	m := New(nil, nil, views)
	m.width, m.height = 80, 24
	return m
}

// TestViewFrameIsExactlyTerminalSized is the regression guard for the duplicated
// bottom rows after a resize. The shell must emit exactly as many rows and
// columns as the terminal has, whatever the active view renders: an over-tall
// frame scrolls the alt screen and leaves the previous frame's tail behind.
func TestViewFrameIsExactlyTerminalSized(t *testing.T) {
	const termWidth, termHeight = 80, 24
	cases := []struct {
		name string
		view *stubView
	}{
		{"view renders far too many rows", &stubView{title: "Over", rows: 200, width: 40}},
		{"view renders too few rows", &stubView{title: "Under", rows: 1, width: 40}},
		{"view renders lines wider than the terminal", &stubView{title: "Wide", rows: 5, width: 500}},
		{"view renders nothing", &stubView{title: "Empty", rows: 0, width: 0}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := New(nil, nil, []View{tc.view})
			m.width, m.height = termWidth, termHeight
			out := m.View()
			if got := lipgloss.Height(out); got != termHeight {
				t.Errorf("frame is %d rows, terminal is %d", got, termHeight)
			}
			if got := lipgloss.Width(out); got > termWidth {
				t.Errorf("frame is %d columns wide, terminal is %d", got, termWidth)
			}
		})
	}
}

// TestFrameNeverOutgrowsTheTerminal pushes the view's budget as low as it goes
// and checks nothing overflows when it gets there.
//
// The budget has a floor of one row, and a floor is only safe if the frame
// never needs to go below it. It did: full help on a node's detail screen lists
// ten keys in one column, so at the twelve-row minimum the footer took ten,
// the budget hit the floor, and the frame came out thirteen rows tall. This
// sweeps every height from one row up, at widths below, at, and above the
// minimum, over every tab, the detail screen, full help, and the banner.
func TestFrameNeverOutgrowsTheTerminal(t *testing.T) {
	withReleaseVersion(t, "0.91.7")
	screens := func() []struct {
		name  string
		build func() Model
	} {
		type screen = struct {
			name  string
			build func() Model
		}
		out := make([]screen, 0, 8)
		for i, v := range defaultViews(nil) {
			out = append(out, screen{v.Title(), func() Model {
				m := New(nil, nil, defaultViews(nil))
				m.selectTab(i)
				return m
			}})
		}
		for _, pane := range []detailPane{detailEngines, detailModels} {
			out = append(out, screen{"node detail", func() Model {
				nodes := newNodesView(nil)
				nodes.feeds.discovered = []availableNode{{HostUUID: "u1", Name: "host", IPAddress: "10.0.0.1", Port: 1}}
				nodes.feeds.selfUUID = "u1"
				nodes.rebuild()
				m := New(nil, nil, []View{nodes})
				nodes.openDetail()
				nodes.detail.pane = pane
				return m
			}})
		}
		return out
	}

	for _, s := range screens() {
		for _, w := range []int{minTerminalWidth - 1, minTerminalWidth, 80, 160} {
			for h := 1; h <= 44; h++ {
				for _, fullHelp := range []bool{false, true} {
					for _, banner := range []bool{false, true} {
						m := s.build()
						m.width, m.height = w, h
						m.help.Width = w
						m.showFullHelp = fullHelp
						if banner {
							m = send(m, updateCheckMsg{latest: "0.92.0"})
						}
						m.resizeViews()
						out := m.View()
						if got := lipgloss.Height(out); got != h {
							t.Fatalf("%s at %dx%d (full help %v, banner %v): frame is %d rows",
								s.name, w, h, fullHelp, banner, got)
						}
						if got := lipgloss.Width(out); got > w {
							t.Fatalf("%s at %dx%d (full help %v, banner %v): frame is %d columns",
								s.name, w, h, fullHelp, banner, got)
						}
					}
				}
			}
		}
	}
}

// TestMinimumWidthKeepsNavigationVisible checks the shell still shows how to
// move between tabs and how to leave at the narrowest supported width.
//
// At forty columns the padded tab bar was forty-six wide and lost its last tab
// off the right, and the footer, truncated from the right, ended at "shift+tab
// prev" — with quit and help both cut off.
func TestMinimumWidthKeepsNavigationVisible(t *testing.T) {
	m := New(nil, nil, defaultViews(nil))
	updated, _ := m.Update(tea.WindowSizeMsg{Width: minTerminalWidth, Height: minTerminalHeight})
	m = updated.(Model)

	lines := strings.Split(m.View(), "\n")
	if !strings.Contains(lines[1], "Logs") {
		t.Errorf("last tab is hidden at minimum width: %q", lines[1])
	}
	footer := lines[len(lines)-1]
	for _, want := range []string{"quit", "help"} {
		if !strings.Contains(footer, want) {
			t.Errorf("footer hides %q at minimum width: %q", want, footer)
		}
	}

	// With an error count the names no longer fit, and every tab must still be
	// reachable by the number the bar shows for it.
	for _, v := range m.views {
		if e, ok := v.(*errorsView); ok {
			e.setErrors([]svcerrors.ServiceError{{ID: "a", Message: "x", Severity: "error"}, {ID: "b", Message: "y", Severity: "error"}})
		}
	}
	bar := strings.Split(m.View(), "\n")[1]
	if got := lipgloss.Width(bar); got > minTerminalWidth {
		t.Fatalf("tab bar is %d wide at %d columns: %q", got, minTerminalWidth, bar)
	}
	for i := range m.views {
		if !strings.Contains(bar, strconv.Itoa(i+1)) {
			t.Errorf("tab %d has no number on the bar at minimum width: %q", i+1, bar)
		}
	}
	if !strings.Contains(bar, m.activeView().Title()) {
		t.Errorf("the active tab is not named at minimum width: %q", bar)
	}
}

// TestFrameStaysExactWithTheUpdateBanner is the arithmetic check for the notice
// row.
//
// The banner adds a row to the frame and takes one from the content budget, and
// those two have to cancel at every height. If they do not, the frame is either
// a row too tall — which scrolls the alt screen and leaves the previous frame's
// tail behind — or a row short of the terminal.
//
// Swept across every supported height rather than sampled, and over the real
// views, because the budget also depends on the active view's footer.
func TestFrameStaysExactWithTheUpdateBanner(t *testing.T) {
	withReleaseVersion(t, "0.91.7")

	for h := minTerminalHeight; h <= 44; h++ {
		for i := range defaultViews(nil) {
			m := newTestModel(defaultViews(nil)...)
			m.width, m.height = 80, h
			m.selectTab(i)

			plain := m.contentHeight()
			m = send(m, updateCheckMsg{latest: "0.92.0"})

			if got := lipgloss.Height(m.View()); got != h {
				t.Fatalf("height %d, tab %d: frame is %d rows with the banner up", h, i+1, got)
			}
			// One row taken, unless the budget had already bottomed out at its
			// floor of one — below that there is nothing left to give.
			if withBanner := m.contentHeight(); plain > 1 && withBanner != plain-1 {
				t.Fatalf("height %d, tab %d: budget %d -> %d, want one row taken",
					h, i+1, plain, withBanner)
			}
		}
	}
}

// TestLeavingATabReturnsItToItsOwnFirstScreen checks a drill-down does not
// outlive the visit that opened it.
//
// The Nodes tab replaces itself with one machine's detail screen. Left open,
// switching to another tab and back put the operator inside that machine
// again rather than on the list they asked for — the list one more keypress
// away, with nothing on screen explaining why.
func TestLeavingATabReturnsItToItsOwnFirstScreen(t *testing.T) {
	nodes := newNodesView(nil)
	nodes.feeds.discovered = []availableNode{
		{HostUUID: "u1", Name: "this-host", IPAddress: "10.0.0.1", Port: 1},
	}
	nodes.rebuild()

	m := newTestModel(nodes, &stubView{title: "Jobs", rows: 1})
	nodes.openDetail()
	if nodes.detail == nil {
		t.Fatal("the detail screen did not open")
	}

	m.selectTab(1)
	if nodes.detail != nil {
		t.Error("leaving the tab left the drill-down open")
	}

	// And the tab still works normally on return: opening one again, then
	// coming back to it directly, also lands on the list.
	m.selectTab(0)
	nodes.openDetail()
	m.selectTab(0)
	if nodes.detail != nil {
		t.Error("re-selecting the tab did not return to the list")
	}
}

// TestViewFrameHeightAcrossTerminalSizes checks the budget holds at the small
// sizes where the header, tab bar, and footer alone can exceed the terminal.
func TestViewFrameHeightAcrossTerminalSizes(t *testing.T) {
	for _, h := range []int{4, 5, 10, 24, 60} {
		m := New(nil, nil, []View{&stubView{title: "T", rows: 100, width: 10}})
		m.width, m.height = 80, h
		if got := lipgloss.Height(m.View()); got != h {
			t.Errorf("height %d: frame is %d rows", h, got)
		}
	}
}

// TestViewBeforeFirstResize checks the shell renders a placeholder rather than a
// zero-sized frame before the terminal size arrives.
func TestViewBeforeFirstResize(t *testing.T) {
	m := New(nil, nil, []View{&stubView{title: "T", rows: 3, width: 10}})
	if out := m.View(); out != "starting..." {
		t.Errorf("pre-resize view = %q, want the placeholder", out)
	}
}

// TestJumpDigitsMatchTheTabsExactly checks the digit binding is neither short
// nor long.
//
// Short means a tab reachable only by tabbing to it, with nothing on screen to
// explain why its number did nothing. Long is what shipped: the footer read
// "1-9 go to tab" against five tabs, advertising four keys that do nothing —
// which is the more common failure, because the range is a string a reader has
// to remember to update.
func TestJumpDigitsMatchTheTabsExactly(t *testing.T) {
	views := defaultViews(nil)
	keys := newGlobalKeyMap(len(views)).JumpTab

	if got, want := len(keys.Keys()), len(views); got != want {
		t.Errorf("%d jump digits (%v) for %d tabs", got, keys.Keys(), want)
	}
	if got, want := keys.Help().Key, "1-"+strconv.Itoa(len(views)); got != want {
		t.Errorf("footer advertises %q, want %q", got, want)
	}
}

// TestJumpDigitsLabelDegenerateCounts covers the label at the edges, since it is
// assembled rather than written out.
func TestJumpDigitsLabelDegenerateCounts(t *testing.T) {
	cases := map[int]string{1: "1", 2: "1-2", 5: "1-5", 9: "1-9"}
	for tabs, want := range cases {
		if got := tabDigitsHelp(tabs); got != want {
			t.Errorf("%d tabs labelled %q, want %q", tabs, got, want)
		}
	}
}

// closingView records being released.
type closingView struct {
	stubView
	closed bool
}

func (c *closingView) close() { c.closed = true }

// TestViewsAreReleasedWithoutTheFinalModel checks cleanup reaches the views
// themselves. After a panic Bubble Tea's Run returns no model to release them
// through, which left a running demo's dispatcher processes behind.
func TestViewsAreReleasedWithoutTheFinalModel(t *testing.T) {
	holding := &closingView{stubView: stubView{title: "Jobs"}}
	closeViews([]View{&stubView{title: "Nodes"}, holding})
	if !holding.closed {
		t.Error("a view holding children was not released")
	}
}

// TestMoreTabsThanDigitsIsRefused checks a tenth tab fails loudly instead of
// quietly having no key. The tab set is fixed at build time, so this is a
// mistake in how the shell was put together, found on the first run.
func TestMoreTabsThanDigitsIsRefused(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("a tenth tab was accepted without a key to select it")
		}
	}()
	newGlobalKeyMap(maxTabs + 1)
}

// TestTheShellFitsItsDigitKeys keeps the real tab set inside that limit, so the
// panic above is never what finds it.
func TestTheShellFitsItsDigitKeys(t *testing.T) {
	if n := len(defaultViews(nil)); n > maxTabs {
		t.Errorf("%d tabs, but only %d have a digit key", n, maxTabs)
	}
}

// TestDigitKeysSelectTabs pins the shortcut the numbered tab bar advertises.
func TestDigitKeysSelectTabs(t *testing.T) {
	m := newTestModel(
		&stubView{title: "One", rows: 1},
		&stubView{title: "Two", rows: 1},
		&stubView{title: "Three", rows: 1},
	)

	press := func(k string) {
		updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(k)})
		m = updated.(Model)
	}

	press("3")
	if m.active != 2 {
		t.Errorf("after '3', active = %d, want 2", m.active)
	}
	press("1")
	if m.active != 0 {
		t.Errorf("after '1', active = %d, want 0", m.active)
	}
	// Out of range for three tabs: the selection must not move.
	press("9")
	if m.active != 0 {
		t.Errorf("after out-of-range '9', active = %d, want 0", m.active)
	}
}

// TestTabWrapsBothDirections checks prev from the first tab lands on the last
// rather than going negative.
func TestTabWrapsBothDirections(t *testing.T) {
	m := newTestModel(
		&stubView{title: "One", rows: 1},
		&stubView{title: "Two", rows: 1},
	)
	m.selectTab(-1)
	if m.active != 1 {
		t.Errorf("selectTab(-1) = %d, want 1", m.active)
	}
	m.selectTab(2)
	if m.active != 0 {
		t.Errorf("selectTab(2) = %d, want 0", m.active)
	}
}

// TestErrorTabLabelCarriesTheCount checks the tab bar is the error indicator, so
// nothing extra is needed to notice a problem from another tab.
func TestErrorTabLabelCarriesTheCount(t *testing.T) {
	v := newErrorsView(nil)

	if got := v.Title(); got != "Errors" {
		t.Errorf("clean label = %q, want a bare title", got)
	}

	v.setErrors([]svcerrors.ServiceError{{ID: "a", Message: "boom", Severity: "error"}})
	if got := v.Title(); got != "Errors (1)" {
		t.Errorf("label = %q, want a count", got)
	}

	v.setErrors([]svcerrors.ServiceError{
		{ID: "a", Message: "boom", Severity: "error"},
		{ID: "b", Message: "meh", Severity: "warning"},
	})
	if got := v.Title(); got != "Errors (2)" {
		t.Errorf("label = %q, want the updated count", got)
	}

	// Clearing the last error takes the count away again.
	v.setErrors(nil)
	if got := v.Title(); got != "Errors" {
		t.Errorf("label = %q after clearing, want a bare title", got)
	}
}

// TestErrorsTabIsReachableLikeAnyOther checks it behaves as a plain tab: the
// digit selects it and nothing intercepts the keyboard.
func TestErrorsTabIsReachableLikeAnyOther(t *testing.T) {
	errors := newErrorsView(nil)
	m := newTestModel(
		&stubView{title: "One", rows: 1},
		&stubView{title: "Two", rows: 1},
		errors,
	)

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("3")})
	m = updated.(Model)
	if m.active != 2 {
		t.Fatalf("digit 3 selected tab %d, want the errors tab", m.active)
	}
	if m.activeView() != View(errors) {
		t.Error("active view is not the errors tab")
	}

	// And tabbing away works, unlike an overlay that had to be dismissed.
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("1")})
	m = updated.(Model)
	if m.active != 0 {
		t.Error("could not leave the errors tab with a digit")
	}
}

// TestErrorsTabFrameStaysBounded checks a long error list obeys the same frame
// budget as any other tab.
func TestErrorsTabFrameStaysBounded(t *testing.T) {
	errors := newErrorsView(nil)
	m := newTestModel(errors)
	m.resizeViews()

	errs := make([]svcerrors.ServiceError, 200)
	for i := range errs {
		errs[i] = svcerrors.ServiceError{ID: strconv.Itoa(i), Message: strings.Repeat("y", 300)}
	}
	errors.setErrors(errs)

	out := m.View()
	if got := lipgloss.Height(out); got != m.height {
		t.Errorf("frame is %d rows, terminal is %d", got, m.height)
	}
	if got := lipgloss.Width(out); got > m.width {
		t.Errorf("frame is %d columns, terminal is %d", got, m.width)
	}
}

func TestFitLines(t *testing.T) {
	if got := fitLines("a\nb\nc", 2); got != "a\nb" {
		t.Errorf("truncate: got %q", got)
	}
	if got := fitLines("a", 3); got != "a\n\n" {
		t.Errorf("pad: got %q", got)
	}
	if got := fitLines("a\nb", 2); got != "a\nb" {
		t.Errorf("exact: got %q", got)
	}
}
