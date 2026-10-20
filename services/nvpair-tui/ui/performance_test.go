// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"nvpair-shared/performancewire"
	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
)

const performanceFixtureJSON = `{
	"local":{"measuringHost":"local-host","status":"fresh","lastSeenAgeMs":0,"expiredEntries":2,
		"summary":{"version":2,"measuringHost":"local-host","admissionIncarnation":"","revision":0,
			"window":{"maxAgeMs":604800000,"maxSamplesPerKey":64},"omittedEntries":3,"entries":[
				{"key":{"host":"local-host","engine":"llamacpp-custom","model":"actual-model","fingerprint":"profile-a"},
				"profile":{"quantization":"Q4_K_M","contextTokens":8192},
				"samples":5,"completed":3,"failed":1,"cancelled":1,"incomplete":1,
				"newestAgeMs":1000,"oldestAgeMs":604801000,"newestObservedAtMs":1700000000000,
				"firstTokenMs":{"samples":2,"median":0,"p95":14},"firstBodyMs":null,
				"headerMs":{"samples":3,"median":2,"p95":8},"totalMs":{"samples":2,"median":100,"p95":200},
				"prefill":{"samples":2,"tokens":20,"durationNs":1000000000,"tokensPerSecond":20},
				"decode":{"samples":1,"tokens":0,"durationNs":1000000000,"tokensPerSecond":0}}
			]}},
	"peers":[],"expiredPeers":4,
	"persistence":{"degraded":false,"publicationBlocked":false,"stats":{
		"Keys":1,"Samples":5,"Dirty":true,"PublishedRevision":0,"HistoryReset":false,
		"EvictedKeys":6,"EvictedSamples":7,"ExpiredSamples":8,"Duplicates":9}}
}`

func performanceFixture(t *testing.T) *performanceProjection {
	t.Helper()
	p, err := decodePerformanceProjection([]byte(performanceFixtureJSON))
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return p
}

func performanceJSON(t *testing.T, p *performanceProjection) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("encode projection: %v", err)
	}
	return raw
}

func assertPerformanceText(t *testing.T, text string, wants ...string) {
	t.Helper()
	for _, want := range wants {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q in performance text:\n%s", want, text)
		}
	}
}

func TestPerformanceNullableMetricsAndCoverage(t *testing.T) {
	d := localDetail()
	d.performance = performanceFixture(t)
	text := d.performanceText()
	assertPerformanceText(t, text,
		"Performance: fresh | measuring host local-host | last seen age 0 ms",
		"engine llamacpp-custom | model actual-model", "Profile profile-a | quantization Q4_K_M | context 8192 tokens",
		"Samples 5 | completed 3 | failed 1 | cancelled 1 | protocol incomplete 1",
		"TTFT: median 0 / p95 14 ms | coverage 2/5", "First body: median -- / p95 -- ms | coverage 0/5",
		"Header: median 2 / p95 8 ms | coverage 3/5", "Total: median 100 / p95 200 ms | coverage 2/5",
		"Prefill: 20 tokens/s | qualified 2/5", "Decode: 0 tokens/s | qualified 1/5",
		"Source age newest 1000 ms / oldest 604801000 ms", "Revision 0 | omitted entries 3",
		"Expired entries 2 | expired peers 4", "evicted keys 6 | evicted samples 7 | expired samples 8")
}

func TestPerformanceAdditionalProfileMetricsMatchDesktop(t *testing.T) {
	d := localDetail()
	d.performance = performanceFixture(t)
	e := &d.performance.Local.Summary.Entries[0]
	e.Profile.EngineFamily = "llamacpp"
	e.AdmissionWaitMs = &performancewire.Distribution{Samples: 2, Median: 0, P95: 5}
	e.LoadNs = &performancewire.Distribution{Samples: 1, Median: 1_500_000, P95: 2_500_000}
	assertPerformanceText(t, d.performanceText(), "Engine family llamacpp",
		"Admission wait: median 0 / p95 5 ms | coverage 2/5",
		"Engine load: median 1.5 / p95 2.5 ms | coverage 1/5")
	e.AdmissionWaitMs, e.LoadNs = nil, nil
	assertPerformanceText(t, d.performanceText(), "Admission wait: median -- / p95 -- ms | coverage 0/5",
		"Engine load: median -- / p95 -- ms | coverage 0/5")
}

func TestPerformanceEngineReportedDecodePresentation(t *testing.T) {
	test := func(name string, rate float64, want string) {
		t.Run(name, func(t *testing.T) {
			d := localDetail()
			d.performance = performanceFixture(t)
			d.performance.Local.Summary.Entries[0].EngineDecode = &performancewire.ReportedRate{
				Samples: 2, DurationNs: 1_000_000_000, TokensPerSecond: rate,
			}
			decoded, err := decodePerformanceProjection(performanceJSON(t, d.performance))
			if err != nil {
				t.Fatalf("decode reported rate: %v", err)
			}
			d.performance = decoded
			text := d.performanceText()
			assertPerformanceText(t, text, want,
				"Duration-weighted mean of reported rates, which may be rounded by the engine; no reconstructed token total.",
				"Host local-host | engine llamacpp-custom | model actual-model", "Profile profile-a")
			if strings.Contains(text, "Decode: 0 tokens/s") {
				t.Fatal("reported and legacy decode populations were mixed")
			}
		})
	}
	test("reported rate and coverage", 49.125, "Engine-reported decode: 49.125 tokens/s | qualified 2/5")
	test("measured zero", 0, "Engine-reported decode: 0 tokens/s | qualified 2/5")
}

func TestPerformanceUnknownDecodeWithoutEitherAggregate(t *testing.T) {
	d := localDetail()
	d.performance = performanceFixture(t)
	d.performance.Local.Summary.Entries[0].Decode = nil
	assertPerformanceText(t, d.performanceText(), "Decode: -- tokens/s | qualified 0/5")
}

func TestPerformanceProjectionRejectsInvalidReportedDecode(t *testing.T) {
	test := func(name, metric string) {
		t.Run(name, func(t *testing.T) {
			raw := strings.Replace(performanceFixtureJSON, `"decode":`, `"engineDecode":`+metric+`,"decode":`, 1)
			if p, err := decodePerformanceProjection([]byte(raw)); err == nil || p != nil {
				t.Fatal("invalid reported rate was accepted")
			}
		})
	}
	test("missing samples", `{"durationNs":100,"tokensPerSecond":0}`)
	test("null samples", `{"samples":null,"durationNs":100,"tokensPerSecond":0}`)
	test("missing duration", `{"samples":1,"tokensPerSecond":0}`)
	test("null duration", `{"samples":1,"durationNs":null,"tokensPerSecond":0}`)
	test("zero duration", `{"samples":1,"durationNs":0,"tokensPerSecond":0}`)
	test("excessive duration", `{"samples":1,"durationNs":86400000000001,"tokensPerSecond":0}`)
	test("missing rate", `{"samples":1,"durationNs":100}`)
	test("null rate", `{"samples":1,"durationNs":100,"tokensPerSecond":null}`)
	test("negative rate", `{"samples":1,"durationNs":100,"tokensPerSecond":-1}`)
	test("excessive rate", `{"samples":1,"durationNs":100,"tokensPerSecond":10000001}`)
	test("incomplete coverage", `{"samples":3,"durationNs":100,"tokensPerSecond":1}`)
	test("invented tokens", `{"samples":1,"durationNs":100,"tokensPerSecond":1,"tokens":1}`)
}

func TestPerformanceProjectionAllowsDisplayOnlyLocalAndAgedValues(t *testing.T) {
	p := performanceFixture(t)
	if p.Local.Summary.Revision != 0 || p.Local.Summary.AdmissionIncarnation != "" || p.Local.Summary.Entries[0].OldestAgeMs != 604801000 {
		t.Fatalf("projected values were rewritten: %+v", p.Local.Summary)
	}
	if err := p.Local.Summary.Validate(); err == nil {
		t.Fatal("fixture unexpectedly satisfies the stricter raw publication contract")
	}
}

func TestPerformanceProjectionRejectsMalformedAndPrivateFrames(t *testing.T) {
	test := func(name string, raw []byte) {
		t.Run(name, func(t *testing.T) {
			p, err := decodePerformanceProjection(raw)
			if err == nil || p != nil {
				t.Fatal("invalid projection was accepted")
			}
			if err.Error() != "invalid performance projection" {
				t.Fatalf("decoder exposed frame details: %q", err)
			}
		})
	}
	test("null", []byte("null"))
	test("v1 summary", []byte(strings.Replace(performanceFixtureJSON, `"version":2`, `"version":1`, 1)))
	test("missing local", []byte(`{"peers":[]}`))
	test("private field", []byte(strings.Replace(performanceFixtureJSON, `"model":"actual-model"`, `"model":"actual-model","prompt":"private"`, 1)))
	test("invalid counts", []byte(strings.Replace(performanceFixtureJSON, `"samples":5`, `"samples":6`, 1)))
	test("negative metric", []byte(strings.Replace(performanceFixtureJSON, `"median":0`, `"median":-1`, 1)))
	test("missing percentile", []byte(strings.Replace(performanceFixtureJSON, `"median":0,`, ``, 1)))
	test("null percentile", []byte(strings.Replace(performanceFixtureJSON, `"median":0`, `"median":null`, 1)))
	test("null eviction counter", []byte(strings.Replace(performanceFixtureJSON, `"EvictedKeys":6`, `"EvictedKeys":null`, 1)))
	test("invalid model control", []byte(strings.Replace(performanceFixtureJSON, `actual-model`, `actual\u001bmodel`, 1)))
	test("unknown status", []byte(strings.Replace(performanceFixtureJSON, `"status":"fresh"`, `"status":"ready"`, 1)))
	test("trailing data", append([]byte(performanceFixtureJSON), []byte(" {}")...))
	test("oversized", []byte(strings.Repeat(" ", performanceProjectionLimit+1)))
	p := performanceFixture(t)
	p.Peers = append(p.Peers, *p.Local)
	test("duplicate host", performanceJSON(t, p))
}

func TestPerformancePeerProfilesKeepActualHostAndFingerprint(t *testing.T) {
	p := performanceFixture(t)
	peer := *p.Local
	summary := *peer.Summary
	peer.MeasuringHost, peer.Status, peer.Summary = "peer-host", "stale", &summary
	summary.MeasuringHost, summary.AdmissionIncarnation, summary.Revision = "peer-host", "7", 3
	summary.Entries = append([]performancewire.Entry(nil), summary.Entries...)
	summary.Entries[0].Key.Host = "peer-host"
	second := summary.Entries[0]
	second.Key.Fingerprint = "profile-b"
	second.Profile = performancewire.Profile{}
	second.FirstTokenMs, second.Prefill, second.Decode = nil, nil, nil
	summary.Entries = append(summary.Entries, second)
	p.Peers = append(p.Peers, peer)
	decoded, err := decodePerformanceProjection(performanceJSON(t, p))
	if err != nil {
		t.Fatalf("decode peer projection: %v", err)
	}
	d := newNodeDetail(nil, nodeRow{key: "peer-host", name: "peer"})
	d.performance = decoded
	text := d.performanceText()
	assertPerformanceText(t, text, "Performance: stale | measuring host peer-host", "Host peer-host | engine llamacpp-custom", "Profile profile-a", "Profile profile-b | quantization -- | context -- tokens", "TTFT: median --", "Prefill: -- tokens/s")
	if strings.Contains(text, "local-host") || strings.Contains(text, "Local store:") {
		t.Fatalf("peer text mixed local data: %s", text)
	}
}

func TestPerformanceExplicitEmptyAndUnknownStates(t *testing.T) {
	test := func(name, status, want string, withSummary bool) {
		t.Run(name, func(t *testing.T) {
			p := performanceFixture(t)
			p.Local.Status = status
			p.Local.Summary.Entries = []performancewire.Entry{}
			if !withSummary {
				p.Local.Summary = nil
			}
			d := localDetail()
			decoded, err := decodePerformanceProjection(performanceJSON(t, p))
			if err != nil {
				t.Fatalf("decode empty projection: %v", err)
			}
			d.performance = decoded
			assertPerformanceText(t, d.performanceText(), "Performance: "+status, want)
		})
	}
	test("empty local history", "missing", "No retained profile measurements.", true)
	test("unknown authority", "unknown", "No cached profile measurements.", false)
	test("missing peer history", "missing", "No cached profile measurements.", false)
}

func performanceReplyFromRPC(t *testing.T, broker *rpc.Codec, cmd tea.Cmd) tea.Msg {
	t.Helper()
	if cmd == nil {
		t.Fatal("no performance command")
	}
	done := make(chan tea.Msg, 1)
	go func() { done <- cmd() }()
	for _, method := range []string{"performance:subscribe", "performance:get"} {
		req, err := broker.Read()
		if err != nil {
			t.Fatalf("read %s: %v", method, err)
		}
		if req.Method != method || req.ID == nil || (len(req.Params) > 0 && string(req.Params) != "null") {
			t.Fatalf("request = %+v, want %s with empty params", req, method)
		}
		result := json.RawMessage(`{"subscribed":true}`)
		if method == "performance:get" {
			result = json.RawMessage(performanceFixtureJSON)
		}
		if err := broker.Write(&rpc.Message{JSONRPC: "2.0", ID: req.ID, Result: result}); err != nil {
			t.Fatalf("respond %s: %v", method, err)
		}
	}
	select {
	case msg := <-done:
		return msg
	case <-time.After(time.Second):
		t.Fatal("performance command did not return")
		return nil
	}
}

func assertPerformanceUnsubscribe(t *testing.T, broker *rpc.Codec) {
	t.Helper()
	req, err := broker.Read()
	if err != nil {
		t.Fatalf("read unsubscribe: %v", err)
	}
	if req.Method != "performance:unsubscribe" {
		t.Fatalf("request = %q, want unsubscribe", req.Method)
	}
	if err := broker.Write(&rpc.Message{JSONRPC: "2.0", ID: req.ID, Result: json.RawMessage(`{"subscribed":false}`)}); err != nil {
		t.Fatalf("respond unsubscribe: %v", err)
	}
}

func TestPerformanceNodeDetailRPCAndRender(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "self", self: true, name: "this-host"})
	d.SetSize(160, 80)
	// Run the real detail Init, including its existing nested proxy batch.
	methods := make(chan string, 32)
	serverDone := make(chan error, 1)
	go func() {
		for {
			req, err := broker.Read()
			if err != nil {
				serverDone <- err
				return
			}
			methods <- req.Method
			result := json.RawMessage(`{}`)
			switch req.Method {
			case "performance:subscribe":
				result = json.RawMessage(`{"subscribed":true}`)
			case "performance:get":
				result = json.RawMessage(performanceFixtureJSON)
			case "performance:unsubscribe":
				result = json.RawMessage(`{"subscribed":false}`)
			}
			if err := broker.Write(&rpc.Message{JSONRPC: "2.0", ID: req.ID, Result: result}); err != nil {
				serverDone <- err
				return
			}
			if req.Method == "performance:unsubscribe" {
				serverDone <- nil
				return
			}
		}
	}()
	var run func(tea.Cmd)
	run = func(cmd tea.Cmd) {
		if cmd == nil {
			return
		}
		msg := cmd()
		if batch, ok := msg.(tea.BatchMsg); ok {
			for _, child := range batch {
				run(child)
			}
			return
		}
		d.update(msg)
	}
	run(d.Init())
	if d.performance == nil {
		t.Fatal("detail Init did not fetch performance")
	}
	d.update(runeKey("v"))
	// The self row reads this host's own hardware, so expect whatever its block renders.
	hardware := strings.SplitN(d.hardwareBlock(), "\n", 2)[0]
	assertPerformanceText(t, d.View(), "Cached performance", "TTFT: median 0 / p95 14", "Decode: 0 tokens/s", hardware)
	if cmd, _ := d.update(runeKey("s")); cmd != nil {
		t.Fatal("performance view dispatched an engine start")
	}
	d.update(runeKey("v"))
	if !strings.Contains(d.View(), "Engines") {
		t.Fatal("v did not return to engine controls")
	}
	d.update(tea.KeyMsg{Type: tea.KeyEsc})
	select {
	case err := <-serverDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("unsubscribe did not finish")
	}
	close(methods)
	counts := map[string]int{}
	for method := range methods {
		counts[method]++
	}
	for _, method := range []string{"performance:subscribe", "performance:get", "performance:unsubscribe"} {
		if counts[method] != 1 {
			t.Errorf("%s count = %d, want 1", method, counts[method])
		}
	}
}

func TestPerformanceLateResponsesCannotCrossScreenOrIdentity(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "peer-host"})
	msg := performanceReplyFromRPC(t, broker, d.startPerformance())
	d.followNode(nodeRow{key: "other-host"}, true)
	d.update(msg)
	if d.performance != nil {
		t.Fatal("old identity's response became current")
	}
	d.followNode(nodeRow{key: "peer-host"}, true)
	d.update(msg)
	if d.performance != nil {
		t.Fatal("ABA identity change revived an old response")
	}
	reopened := newNodeDetail(client, nodeRow{key: "peer-host"})
	start := reopened.startPerformance()
	reopened.update(msg)
	if reopened.performance != nil {
		t.Fatal("previous visit's response became current")
	}
	// A queued older cleanup must not unsubscribe the new screen.
	d.closePerformance()
	newMsg := performanceReplyFromRPC(t, broker, start)
	reopened.update(newMsg)
	if reopened.performance == nil {
		t.Fatal("new visit's current reply was ignored")
	}
	reopened.closePerformance()
	assertPerformanceUnsubscribe(t, broker)
}

func TestPerformanceNotificationSupersedesInFlightGet(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "self", self: true})
	old := performanceReplyFromRPC(t, broker, d.startPerformance())
	p := performanceFixture(t)
	p.Local.Summary.Revision = 12
	d.update(NotificationMsg{Msg: &rpc.Message{Method: "performance:changed", Params: performanceJSON(t, p)}})
	d.update(old)
	if d.performance == nil || d.performance.Local.Summary.Revision != 12 {
		t.Fatal("get overwrote the newer push")
	}
	d.update(NotificationMsg{Msg: &rpc.Message{Method: "performance:changed", Params: json.RawMessage(`{"prompt":"private"}`)}})
	if d.performance != nil || !d.performanceUnavailable || strings.Contains(d.performanceText(), "private") {
		t.Fatal("invalid frame retained or exposed data")
	}
	d.closePerformance()
	assertPerformanceUnsubscribe(t, broker)
}

func TestPerformanceResetAndCloseReleaseSubscriptions(t *testing.T) {
	test := func(name string, leave func(*nodesView)) {
		t.Run(name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			v := newNodesView(client)
			d := newNodeDetail(client, nodeRow{key: "self", self: true})
			d.update(performanceReplyFromRPC(t, broker, d.startPerformance()))
			v.detail = d
			leave(v)
			if d.performanceLease != nil || d.performance != nil {
				t.Fatal("leaving retained performance state")
			}
			assertPerformanceUnsubscribe(t, broker)
		})
	}
	test("tab reset", func(v *nodesView) {
		v.reset()
		if v.detail != nil {
			t.Fatal("reset retained detail")
		}
	})
	test("program close", func(v *nodesView) { closeViews([]View{v}) })
}

func TestPerformanceShellDisconnectClearsDetail(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	v := newNodesView(client)
	d := newNodeDetail(client, nodeRow{key: "self", self: true})
	old := performanceReplyFromRPC(t, broker, d.startPerformance())
	d.update(old)
	v.detail = d
	m := newTestModel(v)
	m.Update(DisconnectedMsg{})
	d.update(old)
	if d.performance != nil || !d.performanceUnavailable || d.performanceLease != nil {
		t.Fatal("shell disconnect retained fresh performance authority")
	}
	assertPerformanceUnsubscribe(t, broker)
}

func TestPerformanceBrokerRestartRequiresNewSnapshot(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "self", self: true})
	old := performanceReplyFromRPC(t, broker, d.startPerformance())
	d.update(old)
	d.update(NotificationMsg{Msg: &rpc.Message{Method: "service/broker-restarted"}})
	if d.performance != nil || !d.performanceUnavailable {
		t.Fatal("restart kept old measurements fresh")
	}
	assertPerformanceUnsubscribe(t, broker)
	start, _ := d.update(NotificationMsg{Msg: &rpc.Message{Method: "app:ready"}})
	d.update(old)
	if d.performance != nil {
		t.Fatal("late pre-restart snapshot became current")
	}
	d.update(performanceReplyFromRPC(t, broker, start))
	if d.performance == nil {
		t.Fatal("new broker snapshot was not accepted")
	}
	d.closePerformance()
	assertPerformanceUnsubscribe(t, broker)
}

func TestPerformanceTransportGenerationRejectsOldResultAndCleanup(t *testing.T) {
	f := newReconnectUIFixture(t)
	d := newNodeDetail(f.client, nodeRow{key: "self", self: true})
	old := performanceReplyFromRPC(t, f.broker, d.startPerformance())
	lease := d.performanceLease
	m := New(f.client, nil, nil)
	f.replace(t, &m)
	// Deliberately keep the same screen, visit, route and lease pointer. Those
	// guards alone cannot reject a response completed on the old transport.
	d.update(old)
	if d.performance != nil {
		t.Fatal("old completed performance response crossed transport generation")
	}
	f.serve(t, map[string]json.RawMessage{
		"performance:subscribe": json.RawMessage(`{"subscribed":true}`),
		"performance:get":       json.RawMessage(performanceFixtureJSON),
	})
	// A replacement subscriber can exist while the old screen is still the
	// registry owner. The old lease may not unsubscribe this transport.
	subscribe := call(f.client, "performance:subscribe", nil, func(_ *rpc.Message, err error) tea.Msg { return err })
	if result := subscribe(); result != nil {
		t.Fatalf("replacement subscribe failed: %v", result)
	}
	lease.release()
	if got := f.count("performance:unsubscribe"); got != 0 {
		t.Fatalf("old lease unsubscribed replacement %d time(s)", got)
	}
	current := d.startPerformance()
	d.update(current())
	if d.performance == nil {
		t.Fatal("new performance baseline did not apply")
	}
	lease.release()
	if got := f.count("performance:unsubscribe"); got != 0 {
		t.Fatalf("delayed cleanup unsubscribed new lease %d time(s)", got)
	}
	d.closePerformance()
}

func TestPerformanceLeaveCancelsAnInFlightGet(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "self", self: true})
	done := make(chan tea.Msg, 1)
	cmd := d.startPerformance()
	go func() { done <- cmd() }()
	subscribe, err := broker.Read()
	if err != nil {
		t.Fatalf("read subscribe: %v", err)
	}
	if subscribe.Method != "performance:subscribe" {
		t.Fatalf("method = %q", subscribe.Method)
	}
	if err := broker.Write(&rpc.Message{JSONRPC: "2.0", ID: subscribe.ID, Result: json.RawMessage(`{"subscribed":true}`)}); err != nil {
		t.Fatalf("reply subscribe: %v", err)
	}
	get, err := broker.Read()
	if err != nil {
		t.Fatalf("read get: %v", err)
	}
	if get.Method != "performance:get" {
		t.Fatalf("method = %q", get.Method)
	}
	d.update(tea.KeyMsg{Type: tea.KeyEsc})
	select {
	case late := <-done:
		d.update(late)
	case <-time.After(time.Second):
		t.Fatal("leaving did not cancel the in-flight get")
	}
	assertPerformanceUnsubscribe(t, broker)
	if d.performance != nil {
		t.Fatal("cancelled response revived closed screen")
	}
}

func TestPerformanceViewScrollsWithoutChangingHardware(t *testing.T) {
	d := localDetail()
	d.performance = performanceFixture(t)
	d.SetSize(70, 12)
	hardware := d.hardwareBlock()
	d.update(runeKey("v"))
	first := d.View()
	assertPerformanceText(t, first, "Cached performance", "Performance: fresh", hardware)
	d.update(tea.KeyMsg{Type: tea.KeyPgDown})
	second := d.View()
	if first == second {
		t.Fatal("paging did not reveal more measurements")
	}
	if !strings.Contains(second, hardware) || d.hardwareBlock() != hardware {
		t.Fatal("performance paging changed hardware output")
	}
	d.performanceViewport.GotoBottom()
	assertPerformanceText(t, d.View(), "Decode: 0 tokens/s")
}

func TestPerformanceRPCFailuresShowUnknownWithoutPrivateError(t *testing.T) {
	for _, failingMethod := range []string{"performance:subscribe", "performance:get"} {
		t.Run(failingMethod, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			d := newNodeDetail(client, nodeRow{key: "self", self: true})
			cmd := d.startPerformance()
			done := make(chan tea.Msg, 1)
			go func() { done <- cmd() }()
			for _, method := range []string{"performance:subscribe", "performance:get"} {
				req, err := broker.Read()
				if err != nil {
					t.Fatalf("read %s: %v", method, err)
				}
				if req.Method != method {
					t.Fatalf("method = %q, want %s", req.Method, method)
				}
				reply := &rpc.Message{JSONRPC: "2.0", ID: req.ID, Result: json.RawMessage(`{"subscribed":true}`)}
				if method == failingMethod {
					reply.Result = nil
					reply.Error = &rpc.RPCError{Code: -32000, Message: "private error body"}
				}
				if err := broker.Write(reply); err != nil {
					t.Fatalf("reply %s: %v", method, err)
				}
				if method == failingMethod {
					break
				}
			}
			select {
			case msg := <-done:
				d.update(msg)
			case <-time.After(time.Second):
				t.Fatal("failed call did not return")
			}
			if d.performance != nil || !d.performanceUnavailable {
				t.Fatal("failed RPC retained performance")
			}
			text := d.performanceText()
			assertPerformanceText(t, text, "Performance: unknown")
			if strings.Contains(text, "private") {
				t.Fatal("raw RPC error reached the display")
			}
			d.closePerformance()
			assertPerformanceUnsubscribe(t, broker)
		})
	}
}
