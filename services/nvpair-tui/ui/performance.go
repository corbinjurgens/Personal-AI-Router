// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"

	"nvpair-shared/performancewire"
	"nvpair-tui/rpc"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"
)

const performancePeerLimit = 128
const performanceProjectionLimit = (performancewire.MaxPayloadBytes + 2048) * (performancePeerLimit + 1)

var detailPerformanceKey = key.NewBinding(key.WithKeys("v"), key.WithHelp("v", "performance / engines"))
var performanceScrollKey = key.NewBinding(key.WithKeys("up", "down", "pgup", "pgdown"), key.WithHelp("up/down pgup/pgdn", "scroll"))

// These are the broker's client projection, not peer publication frames. Local
// history can have revision zero and no admission; projected ages can exceed
// the raw summary window after time spent in a receiver's cache.
type performanceNode struct {
	MeasuringHost  string                   `json:"measuringHost"`
	Status         string                   `json:"status"`
	LastSeenAgeMs  *int64                   `json:"lastSeenAgeMs"`
	Summary        *performancewire.Summary `json:"summary,omitempty"`
	ExpiredEntries *int                     `json:"expiredEntries"`
}

type performanceStats struct {
	Keys, Samples                                           int
	Dirty                                                   bool
	PublishedRevision                                       int64
	HistoryReset                                            bool
	EvictedKeys, EvictedSamples, ExpiredSamples, Duplicates uint64
}

type performanceProjection struct {
	Local       *performanceNode  `json:"local"`
	Peers       []performanceNode `json:"peers"`
	Persistence *struct {
		Degraded           bool              `json:"degraded"`
		PublicationBlocked bool              `json:"publicationBlocked"`
		Error              string            `json:"error,omitempty"`
		Stats              *performanceStats `json:"stats"`
	} `json:"persistence"`
	ExpiredPeers *uint64 `json:"expiredPeers"`
}

func decodePerformanceProjection(raw []byte) (*performanceProjection, error) {
	invalid := errors.New("invalid performance projection")
	if len(raw) == 0 || len(raw) > performanceProjectionLimit {
		return nil, invalid
	}
	if !performanceFieldsPresent(raw) {
		return nil, invalid
	}
	var p performanceProjection
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if dec.Decode(&p) != nil || dec.Decode(&struct{}{}) != io.EOF || p.Local == nil || p.Peers == nil ||
		len(p.Peers) > performancePeerLimit || p.Persistence == nil || p.Persistence.Stats == nil || p.ExpiredPeers == nil {
		return nil, invalid
	}
	stats := p.Persistence.Stats
	if stats.Keys < 0 || stats.Keys > performancewire.MaxEntries || stats.Samples < 0 ||
		stats.Samples > performancewire.MaxEntries*performancewire.MaxSamplesPerKey || stats.PublishedRevision < 0 || len(p.Persistence.Error) > 2048 {
		return nil, invalid
	}
	seen := map[string]bool{}
	nodes := append([]performanceNode{*p.Local}, p.Peers...)
	for i, node := range nodes {
		// Validate the host with the shared identifier contract, without exposing
		// decoder errors (which can contain an unknown private field's name).
		if (performancewire.ProfileKey{Host: node.MeasuringHost, Engine: "display", Model: "display", Fingerprint: "display"}).Validate() != nil ||
			seen[node.MeasuringHost] || node.LastSeenAgeMs == nil || *node.LastSeenAgeMs < 0 || node.ExpiredEntries == nil ||
			*node.ExpiredEntries < 0 || *node.ExpiredEntries > performancewire.MaxEntries {
			return nil, invalid
		}
		seen[node.MeasuringHost] = true
		switch node.Status {
		case "fresh", "stale", "unknown", "missing":
		default:
			return nil, invalid
		}
		if node.Summary == nil {
			if node.Status == "fresh" || node.Status == "stale" {
				return nil, invalid
			}
			continue
		}
		if node.Status == "unknown" || node.Summary.MeasuringHost != node.MeasuringHost || node.Summary.Entries == nil {
			return nil, invalid
		}
		if node.Status == "missing" && len(node.Summary.Entries) != 0 {
			return nil, invalid
		}
		summary := *node.Summary
		if i == 0 {
			if summary.Revision < 0 {
				return nil, invalid
			}
			if summary.Revision == 0 {
				summary.Revision = 1
			}
			if summary.AdmissionIncarnation == "" {
				summary.AdmissionIncarnation = "display"
			}
		}
		summary.Entries = append([]performancewire.Entry(nil), summary.Entries...)
		for j := range summary.Entries {
			e := &summary.Entries[j]
			if e.NewestAgeMs < 0 || e.NewestAgeMs > summary.Window.MaxAgeMs || e.OldestAgeMs < e.NewestAgeMs {
				return nil, invalid
			}
			// The broker expires a profile when its newest sample expires. Its
			// oldest source age can already be beyond the original window.
			e.OldestAgeMs = min(e.OldestAgeMs, summary.Window.MaxAgeMs)
		}
		if summary.Validate() != nil {
			return nil, invalid
		}
		encoded, err := json.Marshal(node.Summary)
		if err != nil || len(encoded) > performancewire.MaxPayloadBytes {
			return nil, invalid
		}
	}
	return &p, nil
}

// The shared Go numeric types intentionally model valid wire values. Check
// presence separately so a malformed omitted/null percentile cannot decode
// into a measured zero before the shared range checks run.
func performanceObject(raw []byte, required ...string) (map[string]json.RawMessage, bool) {
	var fields map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil || fields == nil {
		return nil, false
	}
	for _, name := range required {
		v, ok := fields[name]
		if !ok || bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
			return nil, false
		}
	}
	return fields, true
}

func performanceFieldsPresent(raw []byte) bool {
	root, ok := performanceObject(raw, "local", "peers", "persistence", "expiredPeers")
	if !ok {
		return false
	}
	persistence, ok := performanceObject(root["persistence"], "degraded", "publicationBlocked", "stats")
	if !ok {
		return false
	}
	if _, ok := performanceObject(persistence["stats"], "Keys", "Samples", "Dirty", "PublishedRevision", "HistoryReset", "EvictedKeys", "EvictedSamples", "ExpiredSamples", "Duplicates"); !ok {
		return false
	}
	var nodes []json.RawMessage
	if json.Unmarshal(root["peers"], &nodes) != nil || len(nodes) > performancePeerLimit {
		return false
	}
	nodes = append(nodes, root["local"])
	for _, rawNode := range nodes {
		node, ok := performanceObject(rawNode, "measuringHost", "status", "lastSeenAgeMs", "expiredEntries")
		if !ok {
			return false
		}
		rawSummary := node["summary"]
		if len(rawSummary) == 0 || bytes.Equal(bytes.TrimSpace(rawSummary), []byte("null")) {
			continue
		}
		summary, ok := performanceObject(rawSummary, "version", "measuringHost", "admissionIncarnation", "revision", "window", "entries")
		if !ok {
			return false
		}
		var entries []json.RawMessage
		if json.Unmarshal(summary["entries"], &entries) != nil || len(entries) > performancewire.MaxEntries {
			return false
		}
		for _, rawEntry := range entries {
			entry, ok := performanceObject(rawEntry, "key", "samples", "completed", "incomplete", "failed", "cancelled", "newestAgeMs", "oldestAgeMs", "newestObservedAtMs")
			if !ok {
				return false
			}
			for _, name := range []string{"admissionWaitMs", "headerMs", "firstBodyMs", "firstTokenMs", "totalMs", "loadNs", "prefill", "decode"} {
				metric := entry[name]
				if len(metric) == 0 || bytes.Equal(bytes.TrimSpace(metric), []byte("null")) {
					continue
				}
				required := []string{"samples", "median", "p95"}
				if name == "prefill" || name == "decode" {
					required = []string{"samples", "tokens", "durationNs", "tokensPerSecond"}
				}
				if _, ok := performanceObject(metric, required...); !ok {
					return false
				}
			}
		}
	}
	return true
}

// A broker subscription is connection-wide. Serialize lease changes so an old
// screen's delayed unsubscribe cannot turn off a newly opened screen's stream.
var performanceSubscriptions = struct {
	sync.Mutex
	current map[*rpc.Client]*performanceLease
}{current: make(map[*rpc.Client]*performanceLease)}

type performanceLease struct {
	client  *rpc.Client
	serial  *sync.Mutex
	retired atomic.Bool
	ctx     context.Context
	cancel  context.CancelFunc
}

func newPerformanceLease(client *rpc.Client) *performanceLease {
	l := &performanceLease{client: client, serial: &sync.Mutex{}}
	l.ctx, l.cancel = context.WithCancel(context.Background())
	performanceSubscriptions.Lock()
	if previous := performanceSubscriptions.current[client]; previous != nil {
		previous.retired.Store(true)
		previous.cancel()
		l.serial = previous.serial
	}
	performanceSubscriptions.current[client] = l
	performanceSubscriptions.Unlock()
	return l
}

func (l *performanceLease) release() {
	l.retired.Store(true)
	l.cancel()
	l.serial.Lock()
	defer l.serial.Unlock()
	performanceSubscriptions.Lock()
	current := performanceSubscriptions.current[l.client] == l
	performanceSubscriptions.Unlock()
	if !current {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), callTimeout)
	defer cancel()
	_, _ = l.client.Call(ctx, "performance:unsubscribe", nil)
	performanceSubscriptions.Lock()
	if performanceSubscriptions.current[l.client] == l {
		delete(performanceSubscriptions.current, l.client)
	}
	performanceSubscriptions.Unlock()
}

type detailPerformanceMsg struct {
	lease         *performanceLease
	host          string
	route, pushes int
	projection    *performanceProjection
	err           error
}

func (d *nodeDetail) startPerformance() tea.Cmd {
	if d.client == nil {
		return nil
	}
	d.performanceLease = newPerformanceLease(d.client)
	d.performance = nil
	d.performanceUnavailable = false
	return d.performanceCmd(true)
}

func (d *nodeDetail) performanceCmd(subscribe bool) tea.Cmd {
	l := d.performanceLease
	if l == nil || l.retired.Load() {
		return nil
	}
	reply := detailPerformanceMsg{lease: l, host: d.hostUUID, route: d.performanceRoute, pushes: d.performancePushes}
	return func() tea.Msg {
		l.serial.Lock()
		defer l.serial.Unlock()
		if l.retired.Load() {
			return nil
		}
		ctx, cancel := context.WithTimeout(l.ctx, callTimeout)
		defer cancel()
		if subscribe {
			msg, err := l.client.Call(ctx, "performance:subscribe", nil)
			var ack struct {
				Subscribed bool `json:"subscribed"`
			}
			if err != nil || msg == nil || json.Unmarshal(msg.Result, &ack) != nil || !ack.Subscribed {
				reply.err = errors.New("performance subscription unavailable")
				return reply
			}
		}
		msg, err := l.client.Call(ctx, "performance:get", nil)
		if err != nil || msg == nil {
			reply.err = errors.New("performance unavailable")
		} else {
			reply.projection, reply.err = decodePerformanceProjection(msg.Result)
		}
		return reply
	}
}

func (d *nodeDetail) closePerformance() {
	if l := d.performanceLease; l != nil {
		l.retired.Store(true)
		l.cancel()
		d.performanceLease = nil
		go l.release()
	}
	d.performance = nil
	d.performanceUnavailable = true
}

func (d *nodeDetail) performanceText() string {
	if d.performance == nil {
		if d.performanceUnavailable {
			return "Performance: unknown - cached measurements unavailable."
		}
		return "Performance: missing - waiting for cached measurements."
	}
	p := d.performance
	var node *performanceNode
	if d.telemetryFound {
		if d.telemetrySelf && (d.hostUUID == "" || d.hostUUID == p.Local.MeasuringHost) {
			node = p.Local
		}
		for i := range p.Peers {
			if !d.telemetrySelf && p.Peers[i].MeasuringHost == d.hostUUID {
				node = &p.Peers[i]
				break
			}
		}
	}
	if node == nil {
		return "Performance: missing - no cached measurements for this host."
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Performance: %s | measuring host %s | last seen age %d ms\n", node.Status, node.MeasuringHost, *node.LastSeenAgeMs)
	fmt.Fprintf(&b, "Expired entries %d | expired peers %d\n", *node.ExpiredEntries, *p.ExpiredPeers)
	if d.telemetrySelf {
		s := p.Persistence.Stats
		fmt.Fprintf(&b, "Local store: evicted keys %d | evicted samples %d | expired samples %d\n", s.EvictedKeys, s.EvictedSamples, s.ExpiredSamples)
		if p.Persistence.Degraded || p.Persistence.PublicationBlocked {
			b.WriteString("History persistence degraded or publication blocked\n")
		}
		if s.HistoryReset {
			b.WriteString("Unreadable history was reset\n")
		}
	}
	if node.Summary == nil {
		b.WriteString("No cached profile measurements.")
		return b.String()
	}
	summary := node.Summary
	fmt.Fprintf(&b, "Revision %d | omitted entries %d\n", summary.Revision, summary.OmittedEntries)
	if len(summary.Entries) == 0 {
		b.WriteString("No retained profile measurements.")
	}
	for _, e := range summary.Entries {
		quant, context, family := e.Profile.Quantization, "--", e.Profile.EngineFamily
		if family == "" {
			family = "--"
		}
		if quant == "" {
			quant = "--"
		}
		if e.Profile.ContextTokens != nil {
			context = fmt.Sprint(*e.Profile.ContextTokens)
		}
		fmt.Fprintf(&b, "\nHost %s | engine %s | model %s\nProfile %s | quantization %s | context %s tokens\n", e.Key.Host, e.Key.Engine, e.Key.Model, e.Key.Fingerprint, quant, context)
		fmt.Fprintf(&b, "Engine family %s\n", family)
		fmt.Fprintf(&b, "Samples %d | completed %d | failed %d | cancelled %d | protocol incomplete %d\n", e.Samples, e.Completed, e.Failed, e.Cancelled, e.Incomplete)
		fmt.Fprintf(&b, "Source age newest %d ms / oldest %d ms", e.NewestAgeMs, e.OldestAgeMs)
		if e.ClockUncertain {
			b.WriteString(" (clock uncertain)")
		}
		b.WriteByte('\n')
		for _, metric := range []struct {
			name  string
			value *performancewire.Distribution
		}{
			{"TTFT", e.FirstTokenMs}, {"First body", e.FirstBodyMs}, {"Header", e.HeaderMs}, {"Total", e.TotalMs},
			{"Admission wait", e.AdmissionWaitMs},
		} {
			if metric.value == nil {
				fmt.Fprintf(&b, "%s: median -- / p95 -- ms | coverage 0/%d\n", metric.name, e.Samples)
			} else {
				fmt.Fprintf(&b, "%s: median %d / p95 %d ms | coverage %d/%d\n", metric.name, metric.value.Median, metric.value.P95, metric.value.Samples, e.Samples)
			}
		}
		if e.LoadNs == nil {
			fmt.Fprintf(&b, "Engine load: median -- / p95 -- ms | coverage 0/%d\n", e.Samples)
		} else {
			fmt.Fprintf(&b, "Engine load: median %g / p95 %g ms | coverage %d/%d\n", float64(e.LoadNs.Median)/1e6, float64(e.LoadNs.P95)/1e6, e.LoadNs.Samples, e.Samples)
		}
		for _, phase := range []struct {
			name  string
			value *performancewire.Rate
		}{{"Prefill", e.Prefill}, {"Decode", e.Decode}} {
			if phase.value == nil {
				fmt.Fprintf(&b, "%s: -- tokens/s | qualified 0/%d\n", phase.name, e.Samples)
			} else {
				fmt.Fprintf(&b, "%s: %g tokens/s | qualified %d/%d\n", phase.name, phase.value.TokensPerSecond, phase.value.Samples, e.Samples)
			}
		}
	}
	return strings.TrimSuffix(b.String(), "\n")
}

func (d *nodeDetail) performanceView(identity string) string {
	hardware := d.hardwareBlock()
	d.performanceViewport.Width = max(1, d.width)
	d.performanceViewport.Height = max(1, d.height-countLines(identity)-countLines(hardware)-1)
	// Wrap before the viewport so long model names and profile fingerprints
	// remain available by vertical scrolling rather than being silently clipped.
	d.performanceViewport.SetContent(ansi.Hardwrap(d.performanceText(), max(1, d.width), true))
	return joinLines(identity, hardware, "Cached performance (v returns; up/down scroll)", d.performanceViewport.View())
}
