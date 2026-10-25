// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engineintent"
	"nvpair-shared/engines"
	"nvpair-shared/enginesettings"
	"nvpair-shared/modeldownload"
	"nvpair-shared/noderec"
	"nvpair-tui/rpc"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
)

func tuiDownloadInventory(snapshot *modeldownload.Snapshot) modelsResult {
	return modelsResult{ModelsByEngine: map[string][]string{"ollama": {"exact/model"}},
		LoadedByEngine: map[string][]string{"ollama": {}}, DownloadState: snapshot, current: true}
}

func tuiDownloadProgress(t *testing.T, method, node string, snapshot *modeldownload.Snapshot, entry modeldownload.Entry) NotificationMsg {
	t.Helper()
	return NotificationMsg{Msg: &rpc.Message{Method: method, Params: tuiDownloadJSON(t, map[string]any{
		"node": node, "op": "pull", "id": entry.ID, "engine": entry.Engine, "model": entry.Model, "downloadState": snapshot,
	})}}
}

func TestModelDownloadRowsUniteLifecycleInventoryAndResidency(t *testing.T) {
	d := localDetail()
	snapshot := tuiDownloadSnapshot()
	percent := 21
	snapshot.Entries[0].Percent = &percent
	queued := snapshot.Entries[0]
	queued.ID, queued.Model, queued.Phase, queued.Percent = strings.Repeat("c", 32), "second/model", modeldownload.PhaseQueued, nil
	failed := queued
	failed.ID, failed.Model, failed.Phase = strings.Repeat("d", 32), "failed/model", modeldownload.PhaseFailed
	unknown := queued
	unknown.ID, unknown.Model, unknown.Phase = strings.Repeat("e", 32), "unknown/model", modeldownload.PhaseUnknown
	snapshot.Entries = append(snapshot.Entries, queued, failed, unknown)
	inventory := tuiDownloadInventory(snapshot)
	inventory.LoadedByEngine["ollama"] = []string{"exact/model", "resident-only"}
	d.applyModels(inventory, true)
	rows := map[string]detailModelRow{}
	labels := map[string]string{}
	for i, row := range d.modelRows {
		rows[row.model], labels[row.model] = row, d.modelTable.Rows()[i][3]
	}
	if len(rows) != 5 || len(d.modelRows) != len(rows) {
		t.Fatalf("union rows = %+v", d.modelRows)
	}
	if row := rows["exact/model"]; !row.loaded || !row.loadKnown || row.ready || row.download.ID != snapshot.Entries[0].ID {
		t.Fatalf("loaded unsafe model = %+v", row)
	}
	if row := rows["resident-only"]; !row.loaded || row.ready {
		t.Fatalf("residency certified content: %+v", row)
	}
	want := map[string]string{"exact/model": "Downloading (21%)", "second/model": "Download queued", "failed/model": "Download failed", "unknown/model": "Download status unknown"}
	for model, label := range want {
		if labels[model] != label {
			t.Errorf("%s label = %q, want %q", model, labels[model], label)
		}
	}
}

func TestModelDownloadDispatcherRequiresCurrentCanonicalReadyContent(t *testing.T) {
	test := func(name string, mutate func(*modelsResult), ready bool) {
		t.Run(name, func(t *testing.T) {
			d := localDetail()
			snapshot := tuiDownloadSnapshot()
			snapshot.Entries[0].Phase, snapshot.Entries[0].BlocksReadiness = modeldownload.PhaseComplete, false
			inventory := tuiDownloadInventory(snapshot)
			mutate(&inventory)
			d.applyModels(inventory, true)
			for _, op := range []string{"load", "copy"} {
				act := modelActions["load"]
				act.op = op
				if got := d.modelCmd("ollama", act, "exact/model") != nil; got != ready {
					t.Fatalf("%s dispatcher returned command = %v, want %v", op, got, ready)
				}
			}
			if d.modelCmd("ollama", modelActions["unload"], "exact/model") == nil {
				t.Fatal("readiness changed eject authority")
			}
		})
	}
	test("ready completed content", func(r *modelsResult) {}, true)
	test("complete before inventory", func(r *modelsResult) { r.ModelsByEngine = nil }, false)
	test("unsupported lifecycle", func(r *modelsResult) { r.DownloadState = nil }, false)
	test("stale canonical inventory", func(r *modelsResult) { r.current = false }, false)
	test("recovery required", func(r *modelsResult) { r.DownloadState.RecoveryRequired = true }, false)
	test("listed partial", func(r *modelsResult) {
		r.DownloadState.Entries[0].Phase, r.DownloadState.Entries[0].BlocksReadiness = modeldownload.PhaseInProgress, true
	}, false)
	test("loaded unsafe", func(r *modelsResult) {
		r.LoadedByEngine["ollama"] = []string{"exact/model"}
		r.DownloadState.Entries[0].Phase, r.DownloadState.Entries[0].BlocksReadiness = modeldownload.PhaseFailed, true
	}, false)
	test("pre-effect failure keeps ready content", func(r *modelsResult) { r.DownloadState.Entries[0].Phase = modeldownload.PhaseFailed }, true)
}

func TestModelDownloadActivePullSuppressesDeleteKeyConfirmationAndDispatch(t *testing.T) {
	test := func(name, phase, requested, reported string) {
		t.Run(name, func(t *testing.T) {
			f := newReconnectUIFixture(t)
			f.serve(t, nil)
			d := newNodeDetail(f.client, nodeRow{key: "self", self: true, presence: presenceOnline})
			d.SetSize(120, 30)
			d.pane = detailModels
			d.descriptors = engineDescriptors{"ollama": {
				Version: engines.DescriptorVersion, Engine: "ollama", Family: "ollama", Label: "Ollama",
				ModelNaming: engines.ModelNamingImpliedLatestTag,
				Operations:  map[string]engines.OperationCapability{engines.OperationDelete: {}, engines.OperationUnload: {}},
			}}
			snapshot := tuiDownloadSnapshot()
			snapshot.Entries = []modeldownload.Entry{}
			if !d.applyModels(modelsResult{ModelsByEngine: map[string][]string{"ollama": {requested}}, DownloadState: snapshot, current: true}, true) {
				t.Fatal("initial canonical inventory rejected")
			}
			press := func(k string) {
				t.Helper()
				if cmd := d.handleKeyForTest(t, k); cmd != nil {
					d.update(cmd())
				}
			}
			press("d")
			if d.pending == nil || f.count("engine:delete-model") != 0 {
				t.Fatal("initial Delete did not wait for confirmation")
			}
			active := tuiDownloadSnapshot()
			active.Revision, active.Entries[0].Model, active.Entries[0].Phase = "8", reported, phase
			d.update(tuiDownloadProgress(t, "engine:pull-progress", "", active, active.Entries[0]))
			// Confirm a Delete armed before the pull started. Any mistakenly
			// returned command runs against the pipe fixture and is counted.
			press("y")
			if d.pending != nil || f.count("engine:delete-model") != 0 || !strings.Contains(d.status.text, "Download in progress") {
				t.Fatal("stale confirmation dispatched Delete during an active pull")
			}
			press("d")
			if d.pending != nil || f.count("engine:delete-model") != 0 || !strings.Contains(d.status.text, "Download in progress") {
				t.Fatal("Delete key armed or dispatched during an active pull")
			}
			if cmd := d.modelCmd("ollama", modelActions["delete"], requested); cmd != nil {
				d.update(cmd())
			}
			if f.count("engine:delete-model") != 0 {
				t.Fatal("direct dispatcher bypassed the active-pull guard")
			}
			complete := active.Clone()
			complete.Revision, complete.Entries[0].Phase, complete.Entries[0].BlocksReadiness = "9", modeldownload.PhaseComplete, false
			d.update(tuiDownloadProgress(t, "engine:pull-progress", "", &complete, complete.Entries[0]))
			press("d")
			if d.pending == nil {
				t.Fatal("Delete did not arm after the active pull finished")
			}
			press("y")
			if d.pending != nil || f.count("engine:delete-model") != 1 {
				t.Fatal("confirmed Delete did not dispatch once after the active pull finished")
			}
		})
	}
	test("queued exact model", modeldownload.PhaseQueued, "exact/model", "exact/model")
	test("in progress implied latest", modeldownload.PhaseInProgress, "demo", "demo:latest")
	test("queued explicit latest", modeldownload.PhaseQueued, "demo:latest", "demo")
}

func TestModelDownloadDeleteOutsideMatchingActivePullKeepsAuthority(t *testing.T) {
	test := func(name string, mutate func(*modelsResult)) {
		t.Run(name, func(t *testing.T) {
			d := localDetail()
			snapshot := tuiDownloadSnapshot()
			inventory := tuiDownloadInventory(snapshot)
			mutate(&inventory)
			if !d.applyModels(inventory, true) {
				t.Fatal("canonical inventory rejected")
			}
			if d.modelCmd("ollama", modelActions["delete"], "exact/model") == nil {
				t.Fatal("non-active download state changed Delete authority")
			}
		})
	}
	test("failed unresolved content", func(r *modelsResult) { r.DownloadState.Entries[0].Phase = modeldownload.PhaseFailed })
	test("unknown unresolved content", func(r *modelsResult) { r.DownloadState.Entries[0].Phase = modeldownload.PhaseUnknown })
	test("different model is active", func(r *modelsResult) { r.DownloadState.Entries[0].Model = "other/model" })
	test("different engine is active", func(r *modelsResult) { r.DownloadState.Entries[0].Engine = "llamacpp" })
	test("unsupported lifecycle", func(r *modelsResult) { r.DownloadState = nil })
	test("last known active snapshot is stale", func(r *modelsResult) { r.current = false })
}

func TestModelDownloadProgressHydratesWithoutInitiatorAndFencesReorderedIdentity(t *testing.T) {
	d := localDetail()
	snapshot := tuiDownloadSnapshot()
	cmd, _ := d.update(tuiDownloadProgress(t, "engine:pull-progress", "", snapshot, snapshot.Entries[0]))
	if cmd == nil || d.models.DownloadState != nil || d.modelsGeneration == 0 {
		t.Fatal("unseeded notification selected a source instead of requesting hydration")
	}
	d.update(detailModelsMsg{models: tuiDownloadInventory(snapshot), generation: d.modelsGeneration})
	if len(d.modelRows) != 1 || d.modelRows[0].download.ID != snapshot.Entries[0].ID || d.modelRows[0].ready {
		t.Fatalf("unseeded progress = %+v", d.modelRows)
	}
	newer := snapshot.Clone()
	newer.Revision, newer.Entries[0].ID = "8", strings.Repeat("c", 32)
	d.update(tuiDownloadProgress(t, "engine:pull-progress", "", &newer, newer.Entries[0]))
	d.update(tuiDownloadProgress(t, "engine:pull-progress", "", snapshot, snapshot.Entries[0]))
	if got := d.modelRows[0].download.ID; got != newer.Entries[0].ID {
		t.Fatalf("late old operation replaced %s", got)
	}
	// A queued second model retains its own identity and indeterminate percent.
	queued := newer.Entries[0]
	queued.ID, queued.Model, queued.Phase = strings.Repeat("d", 32), "second/model", modeldownload.PhaseQueued
	newer.Revision, newer.Entries = "9", append(newer.Entries, queued)
	d.update(tuiDownloadProgress(t, "engine:pull-progress", "", &newer, newer.Entries[0]))
	if len(d.modelRows) != 2 || d.modelRows[1].download.ID != queued.ID || d.modelRows[1].download.Percent != nil {
		t.Fatalf("queued model inherited another download: %+v", d.modelRows)
	}
	// A frame's envelope must identify an entry in its full snapshot.
	wrong := queued
	wrong.Model = "wrong/model"
	d.update(tuiDownloadProgress(t, "engine:pull-progress", "", &newer, wrong))
	if d.models.current || d.downloadsCurrent {
		t.Fatal("wrong identity established current state")
	}
}

func TestModelDownloadSourceReplacementRequiresCurrentCanonicalObservation(t *testing.T) {
	d := localDetail()
	original := tuiDownloadSnapshot()
	d.applyModels(tuiDownloadInventory(original), true)
	replacement := original.Clone()
	replacement.SourceID, replacement.Revision = strings.Repeat("f", 32), "0"
	stale := tuiDownloadInventory(&replacement)
	stale.current = false
	if d.applyModels(stale, true) || d.models.DownloadState.SourceID != original.SourceID || d.models.current {
		t.Fatal("stale source replaced baseline or retained ready state")
	}
	if !d.applyModels(tuiDownloadInventory(&replacement), true) {
		t.Fatal("current replacement rejected")
	}
	if d.applyModels(tuiDownloadInventory(original), true) || d.models.DownloadState.SourceID != replacement.SourceID {
		t.Fatal("retired in-flight source replaced current manager")
	}
}

func TestModelDownloadDiscoveryStalenessAndSameRevisionRecovery(t *testing.T) {
	d := remoteDetail()
	snapshot := tuiDownloadSnapshot()
	snapshot.Entries = []modeldownload.Entry{}
	current := true
	peer := availableNode{HostUUID: "peer", DownloadState: snapshot, DownloadStateCurrent: &current,
		ModelsByEngine: map[string][]string{"ollama": {"exact/model"}}}
	d.update(discoveryPush(peer))
	if !d.modelReady("ollama", "exact/model") {
		t.Fatal("current peer inventory did not become ready")
	}
	current = false
	d.update(discoveryPush(peer))
	if d.modelCmd("ollama", modelActions["load"], "exact/model") != nil {
		t.Fatal("stale peer dispatched Load")
	}
	current = true
	d.update(discoveryPush(peer))
	if !d.modelReady("ollama", "exact/model") {
		t.Fatal("same-revision current read did not restore readiness")
	}
	d.update(DisconnectedMsg{})
	if d.modelReady("ollama", "exact/model") {
		t.Fatal("disconnect retained readiness")
	}
	if len(d.modelRows) == 0 || d.modelTable.Rows()[0][3] != "Download status unknown" {
		t.Fatal("disconnect hid last-known model or claimed readiness")
	}
}

func TestModelDownloadHydrationSubscribesBeforeCanonicalFetch(t *testing.T) {
	test := func(name string, remote bool, subscribe, fetch string) {
		t.Run(name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			d := newNodeDetail(client, nodeRow{key: "target", self: !remote, presence: presenceOnline})
			for _, reconnect := range []bool{false, true} {
				if reconnect {
					d.update(DisconnectedMsg{})
					d.policyDisconnected = false
				}
				done := make(chan tea.Msg, 1)
				go func() { done <- d.modelsCmd()() }()
				for _, method := range []string{subscribe, fetch} {
					request, err := broker.Read()
					if err != nil {
						t.Fatalf("read hydration request: %v", err)
					}
					if request.Method != method {
						t.Fatalf("hydration requested %q, want %q", request.Method, method)
					}
					result := tuiDownloadJSON(t, map[string]bool{"subscribed": true})
					if method == fetch {
						snapshot := tuiDownloadSnapshot()
						snapshot.Entries = []modeldownload.Entry{}
						result = tuiDownloadJSON(t, tuiDownloadInventory(snapshot))
						if remote {
							current := true
							result = tuiDownloadJSON(t, map[string][]availableNode{"nodes": {{HostUUID: "target", ModelsByEngine: map[string][]string{"ollama": {"exact/model"}}, DownloadState: snapshot, DownloadStateCurrent: &current}}})
						}
					}
					if err := broker.Write(&rpc.Message{JSONRPC: "2.0", ID: request.ID, Result: result}); err != nil {
						t.Fatalf("reply hydration: %v", err)
					}
				}
				select {
				case msg := <-done:
					d.update(msg)
				case <-time.After(time.Second):
					t.Fatal("hydration did not finish")
				}
				if !d.modelReady("ollama", "exact/model") {
					t.Fatal("canonical hydration did not establish readiness")
				}
			}
		})
	}
	test("local", false, "engine:subscribe", "engine:models")
	test("discovery peer", true, "discovery:subscribe", "discovery:get-nodes")
}

func TestModelDownloadModelsChangedUsesCanonicalNestedInventory(t *testing.T) {
	d := localDetail()
	snapshot := tuiDownloadSnapshot()
	d.applyModels(tuiDownloadInventory(snapshot), true)
	complete := snapshot.Clone()
	complete.Revision, complete.Entries[0].Phase, complete.Entries[0].BlocksReadiness = "8", modeldownload.PhaseComplete, false
	d.update(tuiDownloadProgress(t, "engine:pull-progress", "", &complete, complete.Entries[0]))
	if d.modelCmd("ollama", modelActions["load"], "exact/model") != nil {
		t.Fatal("completion without inventory dispatched Load")
	}
	params := tuiDownloadJSON(t, map[string]any{"engine": "ollama", "models": tuiDownloadInventory(&complete)})
	cmd, _ := d.update(NotificationMsg{Msg: &rpc.Message{Method: "engine:models-changed", Params: params}})
	if cmd != nil || !d.modelReady("ollama", "exact/model") || d.modelRows[0].loaded {
		t.Fatal("nested full inventory was not used independently of residency")
	}
	// A malformed extension must revoke readiness while retaining the last
	// authenticated observation until canonical hydration succeeds.
	bad := `{"models":{"modelsByEngine":{"ollama":["exact/model"]},"loadedByEngine":{"ollama":["exact/model"]},"downloadState":{"version":2}}}`
	d.update(NotificationMsg{Msg: &rpc.Message{Method: "engine:models-changed", Params: []byte(bad)}})
	if d.modelReady("ollama", "exact/model") || d.modelRows[0].loaded || d.modelTable.Rows()[0][3] != "Download status unknown" {
		t.Fatal("malformed support promoted content or replaced the authenticated observation")
	}
	canonical := tuiDownloadInventory(&complete)
	canonical.LoadedByEngine["ollama"] = []string{"exact/model"}
	d.update(detailModelsMsg{models: canonical, generation: d.modelsGeneration})
	if !d.modelRows[0].loaded || !d.modelReady("ollama", "exact/model") {
		t.Fatal("canonical read did not restore residency and readiness independently")
	}
}

func tuiSourceNotification(t *testing.T, method string, snapshot *modeldownload.Snapshot) NotificationMsg {
	t.Helper()
	if method == "engine:pull-progress" {
		return tuiDownloadProgress(t, method, "", snapshot, snapshot.Entries[0])
	}
	return NotificationMsg{Msg: &rpc.Message{Method: method, Params: tuiDownloadJSON(t,
		map[string]any{"models": tuiDownloadInventory(snapshot)})}}
}

func TestModelDownloadModelsChangedCannotSelectInitialSource(t *testing.T) {
	d := localDetail()
	snapshot := tuiDownloadSnapshot()
	oldGeneration := d.modelsGeneration
	cmd, _ := d.update(tuiSourceNotification(t, "engine:models-changed", snapshot))
	if cmd == nil || d.models.DownloadState != nil || d.modelsGeneration != oldGeneration+1 {
		t.Fatal("initial models notification selected an incarnation instead of requesting a canonical read")
	}
	d.update(detailModelsMsg{models: tuiDownloadInventory(snapshot), generation: oldGeneration})
	if d.models.DownloadState != nil {
		t.Fatal("pre-notification read selected the initial incarnation")
	}
	d.update(detailModelsMsg{models: tuiDownloadInventory(snapshot), generation: d.modelsGeneration})
	if d.models.DownloadState == nil || !d.models.DownloadState.Equal(*snapshot) || !d.downloadsCurrent {
		t.Fatal("current canonical read did not establish the initial incarnation")
	}
}

func TestModelDownloadUnseenRetiredNotificationCannotReplaceCanonicalSource(t *testing.T) {
	for _, method := range []string{"engine:pull-progress", "engine:models-changed"} {
		t.Run(method, func(t *testing.T) {
			d := localDetail()
			current := tuiDownloadSnapshot()
			current.Entries[0].Phase, current.Entries[0].BlocksReadiness = modeldownload.PhaseComplete, false
			if !d.applyModels(tuiDownloadInventory(current), true) || !d.modelReady("ollama", "exact/model") {
				t.Fatal("current canonical baseline was not ready")
			}
			// This source belonged to a worker retired before this client attached.
			// It is deliberately absent from the client's retired-source set.
			unseen := tuiDownloadSnapshot()
			unseen.SourceID, unseen.Revision = strings.Repeat("f", 32), "999"
			oldGeneration := d.modelsGeneration
			cmd, _ := d.update(tuiSourceNotification(t, method, unseen))
			if cmd == nil || d.modelsGeneration != oldGeneration+1 || d.models.current || d.downloadsCurrent {
				t.Fatal("unseen-source notification did not revoke currentness and request fresh hydration")
			}
			if !d.models.DownloadState.Equal(*current) || d.retiredDownloadSources[current.SourceID] {
				t.Fatal("unseen retired notification replaced or retired the authenticated source")
			}
			d.update(detailModelsMsg{models: tuiDownloadInventory(unseen), generation: oldGeneration})
			if !d.models.DownloadState.Equal(*current) || d.models.current {
				t.Fatal("in-flight pre-notification read selected a different worker")
			}
			d.update(detailModelsMsg{models: tuiDownloadInventory(current), generation: d.modelsGeneration})
			if !d.modelReady("ollama", "exact/model") || !d.models.DownloadState.Equal(*current) {
				t.Fatal("real current RPC source was rejected after an unseen retired notification")
			}
		})
	}
}

func TestModelDownloadWorkerReplacementRequiresNewCanonicalRead(t *testing.T) {
	for _, method := range []string{"engine:pull-progress", "engine:models-changed"} {
		t.Run(method, func(t *testing.T) {
			d := localDetail()
			original := tuiDownloadSnapshot()
			if !d.applyModels(tuiDownloadInventory(original), true) {
				t.Fatal("original canonical baseline rejected")
			}
			replacement := tuiDownloadSnapshot()
			replacement.SourceID, replacement.Revision = strings.Repeat("e", 32), "0"
			replacement.Entries[0].Phase, replacement.Entries[0].BlocksReadiness = modeldownload.PhaseComplete, false
			cmd, _ := d.update(tuiSourceNotification(t, method, replacement))
			if cmd == nil || !d.models.DownloadState.Equal(*original) || d.retiredDownloadSources[original.SourceID] {
				t.Fatal("replacement notification selected a worker before canonical hydration")
			}
			// Even a matching old-source notification cannot restore currentness
			// while the worker selection requires canonical hydration.
			d.update(tuiSourceNotification(t, method, original))
			if d.models.current || d.downloadsCurrent {
				t.Fatal("old-source notification restored currentness before hydration")
			}
			d.update(detailModelsMsg{models: tuiDownloadInventory(replacement), generation: d.modelsGeneration})
			if !d.models.DownloadState.Equal(*replacement) || !d.retiredDownloadSources[original.SourceID] || !d.modelReady("ollama", "exact/model") {
				t.Fatal("new current canonical read did not select the replacement worker")
			}
			if d.applyModels(tuiDownloadInventory(original), true) || !d.models.DownloadState.Equal(*replacement) {
				t.Fatal("retired worker read replaced the selected incarnation")
			}
		})
	}
}

func TestModelDownloadReadyLoadDispatchesExactTypedRequest(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "self", self: true, presence: presenceOnline})
	snapshot := tuiDownloadSnapshot()
	snapshot.Entries = []modeldownload.Entry{}
	d.applyModels(tuiDownloadInventory(snapshot), true)
	request, _ := terminalControlsReply(t, broker, d.modelOp("load"), &rpc.Message{Result: []byte(`{}`)})
	var params struct {
		Engine string `json:"engine"`
		Model  string `json:"model"`
	}
	if err := json.Unmarshal(request.Params, &params); err != nil {
		t.Fatalf("decode Load params: %v", err)
	}
	if request.Method != "engine:load-model" || params.Engine != "ollama" || params.Model != "exact/model" {
		t.Fatalf("Load request = %s %+v", request.Method, params)
	}
}

func TestNodeDetailSwitchExitCancelsPendingRPC(t *testing.T) {
	for _, action := range []string{"reset", "close", "escape"} {
		t.Run(action, func(t *testing.T) {
			d, f := switchDetail(t, false)
			v := newNodesView(f.client)
			v.detail = d
			done := holdReplacementCommand(t, f, d.submitSwitch(), enginesettings.SwitchMethod)
			ctx := d.engineSwitch.ctx
			switch action {
			case "reset":
				v.reset()
			case "close":
				v.close()
			case "escape":
				d.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
			}
			select {
			case <-ctx.Done():
			default:
				t.Fatal("view exit left switch context alive")
			}
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("view exit left pending switch RPC")
			}
		})
	}
}

func TestReplacementInterruptsEngineOperationStatus(t *testing.T) {
	f := newReconnectUIFixture(t)
	v := newNodesView(f.client)
	d := newNodeDetail(f.client, nodeRow{key: "self", name: "this-host", self: true, presence: presenceOnline})
	d.SetSize(100, 30)
	v.detail = d
	m := New(f.client, nil, []View{v})
	t.Cleanup(d.closePerformance)
	d.engines = []engineStatus{{Engine: "ollama", DisplayName: "Ollama", Installed: true}}
	d.refreshEngines()
	generation := f.client.Generation()
	done := holdReplacementCommand(t, f, d.lifecycle(d.selectedEngine(), "start"), "engine:start")
	if !d.status.sticky || !strings.Contains(d.status.render(), "start") {
		t.Fatal("submitted engine operation did not display persistent progress")
	}
	oldProgress := d.status.text
	d.mode = detailInputLaunchArgs
	d.input.SetValue("--unsent-draft")
	f.replace(t, &m)
	if err := f.peer.SetDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	// The replacement reports that the submitted operation took effect. Loss
	// of its acknowledgement must not be treated as a server-side rollback.
	request, fresh := terminalControlsReply(t, f.broker,
		sessionCommand(f.client, f.client.Generation(), d.enginesCmd()),
		&rpc.Message{Result: []byte(`{"engines":[{"engine":"ollama","display_name":"Ollama","installed":true,"running":true,"healthy":true,"port":11434}]}`)})
	if request.Method != "engine:get-installed" {
		t.Fatalf("fresh baseline method = %q", request.Method)
	}
	updated, _ := m.Update(fresh)
	m = updated.(Model)
	if d.status.sticky || d.status.text == oldProgress {
		t.Error("replacement retained the interrupted engine operation's progress")
	}
	if len(d.engines) != 1 || !d.engines[0].Running || d.enginesStale {
		t.Fatal("replacement engine baseline was not applied")
	}
	if d.mode != detailInputLaunchArgs || d.input.Value() != "--unsent-draft" {
		t.Error("recovery discarded the unsent engine-settings draft")
	}
	status := d.status
	late := interruptedReplacementResult(t, done, generation)
	reply, ok := late.msg.(detailReply)
	if !ok {
		t.Fatal("interrupted engine operation lost its detail owner")
	}
	if op, ok := reply.msg.(engineOpMsg); !ok || op.err == nil {
		t.Fatal("held engine operation did not return its interrupted failure")
	}
	updated, cmd := m.Update(late)
	m = updated.(Model)
	if cmd != nil || d.status != status || len(d.engines) != 1 || !d.engines[0].Running || d.enginesStale {
		t.Error("old engine failure changed recovered status, baseline, or scheduled follow-up work")
	}
	if d.mode != detailInputLaunchArgs || d.input.Value() != "--unsent-draft" {
		t.Error("old engine failure changed the unsent draft")
	}
}

func TestReplacementRetiresDetailConfirmation(t *testing.T) {
	f := newReconnectUIFixture(t)
	v := newNodesView(f.client)
	d := newNodeDetail(f.client, nodeRow{key: "self", name: "this-host", self: true, presence: presenceOnline})
	v.detail = d
	m := New(f.client, nil, []View{v})
	t.Cleanup(d.closePerformance)
	d.uninstallEngine(&engineStatus{Engine: "ollama", DisplayName: "Ollama", Installed: true})
	if d.pending == nil || !d.status.sticky || !strings.Contains(d.status.render(), "press y to confirm") {
		t.Fatal("uninstall confirmation was not visible before disconnect")
	}
	f.replace(t, &m)
	if d.pending != nil || d.status.sticky || strings.Contains(d.status.render(), "press y to confirm") {
		t.Error("replacement retained the retired engine confirmation")
	}
	if cmd := d.handleKeyForTest(t, "y"); cmd != nil {
		t.Error("retired confirmation still submitted an engine operation")
	}
}

func TestDirectTelemetryCannotCrossTransportGeneration(t *testing.T) {
	f := newReconnectUIFixture(t)
	var polls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		polls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{"hostUuid":"peer-host","telemetryValid":true,"msSince":0}`)); err != nil {
			t.Errorf("write telemetry: %v", err)
		}
	}))
	t.Cleanup(server.Close)
	host, portText, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatal(err)
	}
	d := newNodeDetail(f.client, nodeRow{key: "peer-host", address: host, port: port, presence: presenceOnline})
	completed := d.telemetryCmd()()
	delayed := d.telemetryCmd()
	if polls.Load() != 1 {
		t.Fatal("initial direct telemetry did not reach loopback HTTP fixture")
	}
	m := New(f.client, nil, nil)
	f.replace(t, &m)
	if cmd, _ := d.update(completed); cmd != nil || d.telemetryOK {
		t.Fatal("old completed direct telemetry applied or restarted polling")
	}
	d.update(delayed())
	if polls.Load() != 1 || d.telemetryOK {
		t.Fatal("old scheduled direct telemetry polled or applied after replacement")
	}
}

func TestSessionCatalogRefreshPreservesSearchDraft(t *testing.T) {
	f := newReconnectUIFixture(t)
	v := newNodesView(f.client)
	d := newNodeDetail(f.client, nodeRow{key: "self", self: true})
	v.detail = d
	d.catalog = newCatalogBrowser(f.client, "ollama", "Ollama", "this host", false)
	old := d.catalog
	old.all = []catalogModel{{ID: "removed", Name: "removed"}}
	old.browse = old.all
	old.loading = false
	old.gen = 7
	old.filter, old.query, old.sortBy = "fresh", "previous query", catalogSortSize
	old.input.SetValue("unsent search")
	old.searching = true
	old.refresh()
	m := New(f.client, nil, []View{v})
	cmd := f.replace(t, &m)
	if d.catalog == old || len(d.catalog.all) != 0 || len(d.catalog.browse) != 0 || len(d.catalog.table.Rows()) != 0 {
		t.Fatal("replacement catalog retained old authoritative rows")
	}
	if !d.catalog.searching || d.catalog.input.Value() != "unsent search" || d.catalog.filter != "fresh" || d.catalog.query != "previous query" || d.catalog.sortBy != catalogSortSize {
		t.Fatal("replacement catalog discarded unsent search or presentation")
	}
	f.serve(t, map[string]json.RawMessage{
		"performance:subscribe": json.RawMessage(`{"subscribed":true}`),
		"performance:get":       json.RawMessage(performanceFixtureJSON),
		"engine:catalog":        json.RawMessage(`{"models":[{"id":"fresh","name":"fresh model"}],"searchable":true}`),
	})
	batch := cmd().(tea.BatchMsg)
	for _, child := range batch[:len(batch)-1] {
		runSessionUICommand(t, &m, child)
	}
	if len(d.catalog.all) != 1 || d.catalog.all[0].ID != "fresh" || f.count("engine:catalog") != 1 {
		t.Fatal("catalog did not load one fresh replacement baseline")
	}
	d.update(catalogLoadedMsg{engine: "ollama", gen: 7, models: []catalogModel{{ID: "obsolete"}}})
	if len(d.catalog.all) != 1 || d.catalog.all[0].ID != "fresh" {
		t.Fatal("old catalog reply replaced fresh rows")
	}
	d.closePerformance()
}

func TestSessionDetailWaitsForFreshTargetBeforeTelemetry(t *testing.T) {
	f := newReconnectUIFixture(t)
	v := newNodesView(f.client)
	row := nodeRow{key: "peer-host", membership: membershipMember, presence: presenceOnline, address: "192.0.2.1"}
	d := newNodeDetail(f.client, row)
	v.detail = d
	m := New(f.client, nil, []View{v})
	f.replace(t, &m)
	if d.telemetryFound || d.telemetryRunning || d.telemetryCmd() != nil {
		t.Fatal("recovery polled a retained address before fresh feeds validated target")
	}
	d.followNode(row, true)
	f.serve(t, map[string]json.RawMessage{
		"discovery:get-node-info": json.RawMessage(`{"hostUuid":"peer-host","telemetryValid":true,"msSince":0}`),
	})
	cmd := d.telemetryCmd()
	if cmd == nil {
		t.Fatal("fresh target did not restore telemetry polling")
	}
	msg := cmd().(nodeTelemetryMsg)
	if msg.err != nil || f.count("discovery:get-node-info") != 1 {
		t.Fatalf("validated paired target did not use discovery getter: %v", msg.err)
	}
	d.update(msg)
	if !d.telemetryOK {
		t.Fatal("fresh paired telemetry was rejected")
	}
	d.closePerformance()
}

func TestSessionDetailKeepsOnlyUnknownEngineIdentityUntilReady(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", DisplayName: "Ollama", Installed: true, Running: true, Healthy: true, Port: 12345}}
	d.intent = engineintent.Snapshot{EnabledByEngine: map[string]bool{"ollama": false}, OffByEngine: map[string]engineintent.Origin{}}
	d.refreshEngines()
	d.mode = detailInputLaunchArgs
	d.input.SetValue("unsent arguments")
	d.update(DisconnectedMsg{})
	if len(d.engines) != 1 || d.engines[0] != (engineStatus{Engine: "ollama", DisplayName: "Ollama"}) || d.intent.EnabledByEngine != nil || d.intent.OffByEngine != nil {
		t.Fatal("disconnect retained runtime or intent facts beyond engine identity")
	}
	rows := d.engineTable.Rows()
	if len(rows) != 1 || rows[0][0] != "Ollama" {
		t.Fatalf("disconnected engine identity rows = %v", rows)
	}
	for _, cell := range rows[0][1:] {
		if cell != "?" {
			t.Fatalf("disconnected runtime cell = %q, want unknown", cell)
		}
	}
	if cmd := d.own(func() tea.Msg { t.Fatal("disconnected detail started work"); return nil }); cmd != nil {
		t.Fatal("disconnected detail scheduled an RPC command")
	}
	if cmd := d.submitInput(); cmd != nil || d.mode != detailInputLaunchArgs || d.input.Value() != "unsent arguments" {
		t.Fatal("disconnected submit replayed or discarded the draft")
	}
	d.update(sessionReadyMsg{})
	if len(d.engines) != 0 || len(d.engineTable.Rows()) != 0 {
		t.Fatal("replacement ready retained phantom engine identity rows")
	}
}

func localDetail() *nodeDetail {
	d := newNodeDetail(nil, nodeRow{key: "self", name: "this-host", self: true, presence: presenceOnline})
	d.SetSize(100, 30)
	return d
}

func TestSettingsServingSurvivesUnrelatedEditAndFailedApply(t *testing.T) {
	d := localDetail()
	snapshot := ollamaSettings()
	ram, vram := int64(4096), int64(0)
	snapshot.Settings.Serving = &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{{ModelID: "exact/model", MaxOutputTokens: 128, Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram}, Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: strings.Repeat("a", 64), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}}}}}
	request, ok := d.settingsRequest(snapshot, detailInputProxyPort, "12500")
	if !ok || request.Settings.ProxyPort != 12500 || !request.Settings.Serving.Equal(snapshot.Settings.Serving) {
		t.Fatal("unrelated terminal edit lost serving")
	}
	_, apply, _ := judgeSettingsPreview(enginePreviewMsg{request: request, preview: enginesettings.Preview{Settings: request.Settings}})
	data, err := json.Marshal(apply)
	if err != nil {
		t.Fatal(err)
	}
	var wire enginesettings.Request
	if err := json.Unmarshal(data, &wire); err != nil {
		t.Fatal(err)
	}
	if !wire.Settings.Serving.Equal(snapshot.Settings.Serving) {
		t.Fatal("terminal apply wire lost serving")
	}
	snapshot.Settings = wire.Settings
	snapshot.Phase = settingsPhaseFailed
	snapshot.RequestID = apply.RequestID
	snapshot.Error = "synthetic apply failure"
	d.settingsAwaited = &awaitedSettings{engine: "ollama", requestID: apply.RequestID}
	d.reportSettingsOutcome(snapshot)
	retry, ok := d.settingsRequest(snapshot, detailInputLaunchArgs, "--unrelated")
	if !ok || !retry.Settings.Serving.Equal(wire.Settings.Serving) {
		t.Fatal("failed apply retry lost serving")
	}
}

func remoteDetail() *nodeDetail {
	d := newNodeDetail(nil, nodeRow{key: "peer", name: "peer-host", presence: presenceOnline})
	d.SetSize(100, 30)
	return d
}

// discoveryPush builds a discovery:nodes-changed notification.
func discoveryPush(nodes ...availableNode) NotificationMsg {
	params, _ := json.Marshal(nodes)
	return NotificationMsg{Msg: &rpc.Message{Method: "discovery:nodes-changed", Params: params}}
}

// TestRemoteDetailFollowsDiscoveryModels is the regression guard for a peer's
// detail screen freezing at the moment it was opened.
//
// A peer's model inventory arrives only in the enriched discovery snapshot —
// there is no per-node models RPC — and the screen seeded itself from the
// snapshot it was constructed with and then ignored every later one. Pulling a
// model onto that peer, including from this very screen, changed nothing on
// screen until the operator backed out and re-entered.
func TestRemoteDetailFollowsDiscoveryModels(t *testing.T) {
	d := remoteDetail()
	d.models = modelsResult{Models: []string{"old-model"}}
	d.refreshModels()

	d.update(discoveryPush(availableNode{
		HostUUID:       "peer",
		Name:           "peer-host",
		Models:         []string{"old-model", "new-model"},
		ModelsByEngine: map[string][]string{"ollama": {"old-model", "new-model"}},
	}))

	if len(d.models.Models) != 2 {
		t.Fatalf("models = %v, want the refreshed pair from discovery", d.models.Models)
	}
	if !contains(d.View(), "new-model") {
		t.Error("a model that appeared on the peer is not on screen")
	}
}

// TestRemoteDetailIgnoresOtherNodesDiscovery checks the screen only takes the
// entry for its own node, so a busy cluster cannot overwrite it.
func TestRemoteDetailIgnoresOtherNodesDiscovery(t *testing.T) {
	d := remoteDetail()
	d.models = modelsResult{Models: []string{"mine"}}
	d.refreshModels()

	d.update(discoveryPush(availableNode{
		HostUUID: "somebody-else",
		Name:     "other-host",
		Models:   []string{"theirs"},
	}))

	if len(d.models.Models) != 1 || d.models.Models[0] != "mine" {
		t.Errorf("another node's discovery entry overwrote this one: %v", d.models.Models)
	}
}

// TestUnreportedLoadStateIsUnknown is the regression guard for "not loaded"
// shown for a model whose engine cannot say. An engine with no loaded endpoint
// is left out of loadedByEngine, and reading that as an empty list marked every
// one of its models as not in memory.
func TestUnreportedLoadStateIsUnknown(t *testing.T) {
	d := localDetail()
	d.models = modelsResult{
		ModelsByEngine: map[string][]string{
			"ollama":   {"idle", "resident"},
			"llamacpp": {"unreported"},
		},
		LoadedByEngine: map[string][]string{"ollama": {"resident"}},
	}
	d.refreshModels()

	got := map[string]string{}
	for _, row := range d.modelTable.Rows() {
		got[row[0]] = row[2]
	}
	want := map[string]string{"idle": "no", "resident": "yes", "unreported": "?"}
	for model, loaded := range want {
		if got[model] != loaded {
			t.Errorf("%s shows loaded %q, want %q", model, got[model], loaded)
		}
	}
}

// TestLocalDetailIgnoresDiscoveryModels checks this machine keeps using the
// authoritative engine:models RPC rather than the discovery summary.
func TestLocalDetailIgnoresDiscoveryModels(t *testing.T) {
	d := localDetail()
	d.models = modelsResult{Models: []string{"authoritative"}}
	d.refreshModels()

	d.update(discoveryPush(availableNode{
		HostUUID: "self",
		Name:     "this-host",
		Models:   []string{"stale-summary"},
	}))

	if len(d.models.Models) != 1 || d.models.Models[0] != "authoritative" {
		t.Errorf("local detail took models from discovery: %v", d.models.Models)
	}
}

// TestRemoteDetailPollsEngines checks a peer's engine state is re-read on a
// tick. engine:state-changed carries a local snapshot with no node on it, so it
// cannot be attributed to a peer — polling is the only way an open remote
// screen notices an engine starting or stopping over there.
func TestRemoteDetailPollsEngines(t *testing.T) {
	d := remoteDetail()
	if cmd, _ := d.update(detailEnginesTickMsg{gen: d.telemetryGen}); cmd == nil {
		t.Error("remote detail did not re-read engines on its tick")
	}

	// A superseded screen's tick is dropped, matching the telemetry chain.
	if cmd, _ := d.update(detailEnginesTickMsg{gen: d.telemetryGen + 1}); cmd != nil {
		t.Error("a stale chain's tick was extended")
	}

	// This machine has real pushes, so it must not poll.
	local := localDetail()
	if cmd, _ := local.update(detailEnginesTickMsg{gen: local.telemetryGen}); cmd != nil {
		t.Error("local detail polls engines despite receiving engine:state-changed")
	}
}

// TestDetailSectionsAreSeparated is the guard for the two tables reading as one
// with a stray header in the middle: there must be a blank line between them.
func TestDetailSectionsAreSeparated(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true, Running: true, Port: 11434}}
	d.refreshEngines()

	lines := strings.Split(d.View(), "\n")
	modelsAt := -1
	for i, l := range lines {
		if strings.Contains(l, "Models") {
			modelsAt = i
			break
		}
	}
	if modelsAt <= 0 {
		t.Fatalf("no Models heading found in:\n%s", d.View())
	}
	if strings.TrimSpace(lines[modelsAt-1]) != "" {
		t.Errorf("no blank line before the Models heading; previous line was %q", lines[modelsAt-1])
	}
}

// TestLocalDetailShowsBothPorts is the guard for the two ports being managed in
// different places: on this machine they sit side by side on the engine's row.
func TestLocalDetailShowsBothPorts(t *testing.T) {
	d := localDetail()
	if d.proxy == nil {
		t.Fatal("no proxy tracker on the local node")
	}
	d.proxy.apply(proxyStatusMsg{idx: 0, ready: true, port: 11435})
	d.engines = []engineStatus{{Engine: "ollama", Installed: true, Running: true, Port: 11434}}
	d.refreshEngines()

	cols := detailEngineColumns(100, false)
	titles := make([]string, 0, len(cols))
	for _, c := range cols {
		titles = append(titles, c.Title)
	}
	joined := strings.Join(titles, " ")
	if !strings.Contains(joined, "ENGINE PORT") || !strings.Contains(joined, "PROXY PORT") {
		t.Fatalf("local engine columns = %q, want both ports", joined)
	}

	row := d.engineTable.Rows()[0]
	if row[4] != "11434" {
		t.Errorf("engine port cell = %q, want 11434", row[4])
	}
	if row[5] != "11435" {
		t.Errorf("proxy port cell = %q, want 11435", row[5])
	}
}

// TestRemoteDetailHidesProxyPort checks a peer's endpoints are not presented as
// ours to configure.
func TestRemoteDetailHidesProxyPort(t *testing.T) {
	d := remoteDetail()
	if d.proxy != nil {
		t.Error("a remote node should carry no proxy tracker")
	}
	for _, c := range detailEngineColumns(100, true) {
		if c.Title == "PROXY PORT" {
			t.Error("remote engine table offers a proxy port column")
		}
	}

	d.engines = []engineStatus{{Engine: "ollama", Port: 11434}}
	d.refreshEngines()
	if got := len(d.engineTable.Rows()[0]); got != 5 {
		t.Errorf("remote row has %d cells, want 5", got)
	}
}

func TestDetailProxyPortUnknownAfterFailedStatusRead(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true, Running: true, Port: 11434}}
	idx := d.proxy.indexForEngine("ollama")
	if idx < 0 {
		t.Fatal("local detail has no Ollama proxy tracker")
	}
	cell := func() string {
		t.Helper()
		rows := d.engineTable.Rows()
		if len(rows) != 1 || len(rows[0]) != 6 {
			t.Fatal("local detail did not render its engine and proxy port columns")
		}
		return rows[0][5]
	}
	assertDown := func(stage string) {
		t.Helper()
		if got := cell(); !strings.Contains(got, "down") {
			t.Errorf("%s: proxy cell = %q, want confirmed down", stage, got)
		}
		if port, usable := d.proxy.portForEngine("ollama"); usable || port != 11435 {
			t.Errorf("%s: endpoint port=%d usable=%v, want 11435/false", stage, port, usable)
		}
	}
	assertUsable := func(stage string) {
		t.Helper()
		if got := cell(); got != "11435" {
			t.Errorf("%s: proxy cell = %q, want usable port 11435", stage, got)
		}
		if port, usable := d.proxy.portForEngine("ollama"); !usable || port != 11435 {
			t.Errorf("%s: endpoint port=%d usable=%v, want 11435/true", stage, port, usable)
		}
	}
	d.update(proxyStatusMsg{idx: idx, ready: true, port: 11435})
	assertUsable("successful status read")
	d.update(proxyStatusMsg{idx: idx, err: errors.New("status read unavailable")})
	if got := cell(); strings.Contains(got, "down") || (!strings.Contains(got, "unknown") && !strings.Contains(got, "?")) {
		t.Errorf("failed status read: proxy cell = %q, want unknown", got)
	}
	if port, usable := d.proxy.portForEngine("ollama"); usable || port != 11435 {
		t.Errorf("failed status read: endpoint port=%d usable=%v, want retained 11435/false", port, usable)
	}
	d.update(proxyStatusMsg{idx: idx, ready: false, port: 0})
	assertDown("successful ready:false read")
	d.update(proxyStatusMsg{idx: idx, err: errors.New("status read unavailable")})
	d.update(NotificationMsg{Msg: &rpc.Message{Method: "ollama-proxy:error", Params: []byte(`{}`)}})
	assertDown("explicit error push")
	d.update(NotificationMsg{Msg: &rpc.Message{Method: "ollama-proxy:ready", Params: []byte(`{"port":11435}`)}})
	assertUsable("ready push")
}

// TestProxyPortCellStates checks the cell distinguishes a live endpoint from a
// port whose proxy is down, and from an engine no proxy fronts.
func TestProxyPortCellStates(t *testing.T) {
	d := localDetail()

	d.proxy.apply(proxyStatusMsg{idx: 0, ready: true, port: 11435})
	if got := d.proxyPortCell("ollama"); got != "11435" {
		t.Errorf("live endpoint = %q", got)
	}

	d.proxy.engines[0].ready = false
	if got := d.proxyPortCell("ollama"); !strings.Contains(got, "down") {
		t.Errorf("endpoint with the proxy down = %q, want it marked down", got)
	}

	if got := d.proxyPortCell("not-an-engine"); got != "-" {
		t.Errorf("engine with no proxy = %q, want %q", got, "-")
	}
}

// TestProxyIndexForEngine pins the engine-to-proxy pairing the proxy port
// column depends on.
func TestProxyIndexForEngine(t *testing.T) {
	p := newProxyTracker()
	if got := p.indexForEngine("ollama"); got != 0 {
		t.Errorf("ollama -> %d, want 0", got)
	}
	if got := p.indexForEngine("lmstudio"); got != 1 {
		t.Errorf("lmstudio -> %d, want 1", got)
	}
	// Case and padding must not decide whether a port renders.
	if got := p.indexForEngine("  OLLAMA "); got != 0 {
		t.Errorf("normalisation failed: %d", got)
	}
	if got := p.indexForEngine("vllm"); got != -1 {
		t.Errorf("unknown engine -> %d, want -1", got)
	}
}

// seedSettings puts a settings snapshot in the cache, as a fetch or a push
// would, so a test can press an edit key without a broker behind it.
func seedSettings(d *nodeDetail, snap enginesettings.Snapshot) {
	if d.settings == nil {
		d.settings = map[string]enginesettings.Snapshot{}
	}
	d.settings[snap.Engine] = snap
}

// ollamaSettings is an editable snapshot for the engine the detail tests use.
func ollamaSettings() enginesettings.Snapshot {
	return enginesettings.Snapshot{
		Engine:   "ollama",
		Revision: 7,
		Editable: true,
		Settings: enginesettings.Config{
			ServerPort: 11434,
			ProxyPort:  11435,
			LaunchText: "OLLAMA_KEEP_ALIVE=5m",
		},
	}
}

// TestSettingsKeysAskTheBackendBeforeOpening checks an edit key fetches the
// snapshot rather than opening a field against values this screen guessed.
//
// The revision is the point. A field opened without one has nothing to write
// against, and the backend's whole defence against two clients overwriting each
// other is that every write carries the revision it was based on.
func TestSettingsKeysAskTheBackendBeforeOpening(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true, Port: 11434}}
	d.refreshEngines()

	cmd := d.handleEngineKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("e")})
	if d.mode != detailInputNone {
		t.Errorf("opened a field before the snapshot arrived (mode %v)", d.mode)
	}
	if cmd == nil {
		t.Fatal("no settings request was issued")
	}
	if d.settingsWanted == nil || d.settingsWanted.mode != detailInputEnginePort {
		t.Fatalf("the requested edit was not remembered: %+v", d.settingsWanted)
	}
}

// TestSettingsEditorsAreDistinct checks the three fields are told apart, so a
// typed value cannot be applied to the wrong one.
func TestSettingsEditorsAreDistinct(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true, Port: 11434}}
	d.refreshEngines()
	seedSettings(d, ollamaSettings())

	cases := []struct {
		key   string
		mode  detailInputMode
		value string
		label string
	}{
		{"e", detailInputEnginePort, "11434", "engine"},
		{"p", detailInputProxyPort, "11435", "proxy"},
		{"a", detailInputLaunchArgs, "OLLAMA_KEEP_ALIVE=5m", "arguments"},
	}
	for _, tc := range cases {
		d.mode = detailInputNone
		d.handleEngineKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(tc.key)})
		if d.mode != tc.mode {
			t.Fatalf("%q opened mode %v, want %v", tc.key, d.mode, tc.mode)
		}
		if got := d.input.Value(); got != tc.value {
			t.Errorf("%q seeded with %q, want %q", tc.key, got, tc.value)
		}
		if !strings.Contains(d.inputLabel(), tc.label) {
			t.Errorf("label %q does not say which field", d.inputLabel())
		}
	}
}

// TestSettingsRefusalComesFromTheBackend checks an engine the backend will not
// let us configure is refused in the backend's own words.
//
// This screen used to decide for itself, refusing every port change on a peer
// because the old RPC had no remote form. The settings path does, so the
// judgement belongs to the side that knows why — an adopted engine, an
// unreachable settings worker — rather than to a rule here that would drift.
func TestSettingsRefusalComesFromTheBackend(t *testing.T) {
	d := remoteDetail()
	d.engines = []engineStatus{{Engine: "ollama", Port: 11434}}
	d.refreshEngines()

	snap := ollamaSettings()
	snap.Editable = false
	snap.Adopted = true
	snap.Reason = "this engine was started outside PAIR"
	seedSettings(d, snap)

	d.handleEngineKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("a")})
	if d.mode != detailInputNone {
		t.Error("opened an editor for an engine the backend said is not editable")
	}
	if got := d.status.render(); !strings.Contains(got, "started outside PAIR") {
		t.Errorf("status %q does not carry the backend's reason", got)
	}
}

// TestSettingsWriteCarriesTheRevisionAndResolution checks the two things a
// settings write cannot be wrong about.
//
// The revision is what makes a concurrent edit fail instead of silently
// winning. The resolution decides which side gives way when the numeric port
// field and the port inside the command text disagree: editing the field
// should rewrite the command, and editing the command should move the field.
// Send the wrong one and the backend quietly rewrites what the operator typed.
func TestSettingsWriteCarriesTheRevisionAndResolution(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true, Port: 11434}}
	d.refreshEngines()
	snap := ollamaSettings()
	seedSettings(d, snap)

	cases := []struct {
		name       string
		mode       detailInputMode
		value      string
		resolution string
		check      func(enginesettings.Config) error
	}{
		{
			name:       "the port field wins over the command",
			mode:       detailInputEnginePort,
			value:      "11500",
			resolution: resolutionServer,
		},
		{
			name:       "the command wins over the port field",
			mode:       detailInputLaunchArgs,
			value:      "OLLAMA_HOST=127.0.0.1:11500",
			resolution: resolutionLaunch,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, ok := d.settingsRequest(snap, tc.mode, tc.value)
			if !ok {
				t.Fatal("the request was rejected")
			}
			if req.ExpectedRevision != snap.Revision {
				t.Errorf("revision %d, want the snapshot's %d",
					req.ExpectedRevision, snap.Revision)
			}
			if req.Resolution != tc.resolution {
				t.Errorf("resolution %q, want %q", req.Resolution, tc.resolution)
			}
		})
	}
}

// TestLaunchTextIsSentAsTyped checks the arguments field is not tidied here.
//
// Whitespace is significant to a tokenizer, and the backend normalizes to rules
// this screen does not carry. Trimming on the way out would disagree with it
// and, worse, would disagree invisibly.
func TestLaunchTextIsSentAsTyped(t *testing.T) {
	d := localDetail()
	snap := ollamaSettings()
	const typed = `  OLLAMA_ORIGINS="https://example.com"  `

	req, ok := d.settingsRequest(snap, detailInputLaunchArgs, typed)
	if !ok {
		t.Fatal("the request was rejected")
	}
	if req.Settings.LaunchText != typed {
		t.Errorf("sent %q, want the text exactly as typed", req.Settings.LaunchText)
	}
}

// TestSettingsApplyUsesTheNormalizedDraft checks the commit sends what the
// backend validated, not what was typed.
//
// The preview returns normalized settings — quoting settled, ports reconciled
// — and applying the raw draft instead would save text that was never checked,
// with the preview's approval standing behind it.
func TestSettingsApplyUsesTheNormalizedDraft(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true}}
	d.refreshEngines()

	typed := enginesettings.Config{ServerPort: 11434, ProxyPort: 11435, LaunchText: "--flag   x"}
	normalized := enginesettings.Config{ServerPort: 11434, ProxyPort: 11435, LaunchText: "--flag x"}

	verdict, sent, problem := judgeSettingsPreview(enginePreviewMsg{
		request: enginesettings.Request{
			Engine:           "ollama",
			ExpectedRevision: 7,
			Settings:         typed,
			Resolution:       resolutionLaunch,
		},
		preview: enginesettings.Preview{Settings: normalized},
	})

	if verdict != settingsWrite {
		t.Fatalf("a clean preview did not write (verdict %v, problem %q)", verdict, problem)
	}
	if sent.Settings.LaunchText != normalized.LaunchText {
		t.Errorf("applied %q, want the normalized %q",
			sent.Settings.LaunchText, normalized.LaunchText)
	}
	if sent.Resolution != "" {
		t.Errorf("resolution %q survived into the commit; the draft is already settled",
			sent.Resolution)
	}
	if sent.ExpectedRevision != 7 {
		t.Errorf("revision %d, want the one the draft was based on", sent.ExpectedRevision)
	}
	_ = d
}

// settingsPush builds an engine:settings-changed notification.
func settingsPush(snap enginesettings.Snapshot) NotificationMsg {
	params, _ := json.Marshal(snap)
	return NotificationMsg{Msg: &rpc.Message{Method: "engine:settings-changed", Params: params}}
}

// TestLocalSettingsPushIsNotDiscarded is the regression guard for a second
// save that could never succeed.
//
// The broker stamps every snapshot with this node's UUID. Matching that
// against the node argument the local RPCs take — which is empty, precisely
// because they are local — discarded every push for this machine. The cached
// revision then stayed at whatever the first read returned, so the save after
// a successful one was rejected as stale, and stayed rejected until the screen
// was closed and reopened.
func TestLocalSettingsPushIsNotDiscarded(t *testing.T) {
	const uuid = "33983c39-c0a6-41d2-9488-455b5e61e25f"
	d := newNodeDetail(nil, nodeRow{key: uuid, name: "this-host", self: true, presence: presenceOnline})
	d.SetSize(100, 30)
	seedSettings(d, ollamaSettings()) // revision 7

	moved := ollamaSettings()
	moved.NodeID = uuid
	moved.Revision = 8
	d.handleNotification(settingsPush(moved).Msg)

	if got := d.settings["ollama"].Revision; got != 8 {
		t.Fatalf("cached revision is %d after a push for this machine, want 8", got)
	}

	// And a push for a different machine is still ignored.
	other := ollamaSettings()
	other.NodeID = "some-other-node"
	other.Revision = 99
	d.handleNotification(settingsPush(other).Msg)
	if got := d.settings["ollama"].Revision; got != 8 {
		t.Errorf("a peer's snapshot overwrote this machine's: revision %d", got)
	}
}

// TestFailedSaveReloadsTheSnapshot checks a rejected write leaves the screen
// able to try again.
//
// A revision the backend will not accept is not recoverable by repeating the
// same write: without dropping it, every later attempt fails identically and
// the only way out is to leave the screen.
func TestFailedSaveReloadsTheSnapshot(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true}}
	d.refreshEngines()
	seedSettings(d, ollamaSettings())
	d.settingsAwaited = &awaitedSettings{engine: "ollama", requestID: "req-1"}

	cmd, _ := d.update(engineSettingsAppliedMsg{
		engine: "ollama",
		err:    errors.New("settings changed on this device; reload before applying"),
	})

	if _, still := d.settings["ollama"]; still {
		t.Error("the rejected snapshot is still cached, so the next attempt repeats the failure")
	}
	if cmd == nil {
		t.Error("nothing re-read the settings, so the next edit has nothing to write against")
	}
	if got := d.status.render(); !strings.Contains(got, "try again") {
		t.Errorf("status %q does not tell the operator what to do: %q", got, "try again")
	}
}

// TestSettingsCommitCarriesARequestIdentifier is the regression guard for a
// save that failed after the check had passed.
//
// The identifier is the backend's idempotency key: it records a receipt against
// it, returns the original outcome when one is replayed, and refuses a replay
// carrying different settings. A commit without one is rejected outright — and
// because only the commit needs it, the preview succeeded first, so the
// interface reported the settings as valid and then refused to save them.
func TestSettingsCommitCarriesARequestIdentifier(t *testing.T) {
	previewOf := func(port int) enginePreviewMsg {
		return enginePreviewMsg{
			request: enginesettings.Request{Engine: "ollama", ExpectedRevision: 7},
			preview: enginesettings.Preview{
				Settings: enginesettings.Config{ServerPort: port, ProxyPort: 11434},
			},
		}
	}

	_, first, _ := judgeSettingsPreview(previewOf(11500))
	if first.RequestID == "" {
		t.Fatal("the commit carries no request identifier; the backend will refuse it")
	}
	if len(first.RequestID) > 128 {
		t.Errorf("identifier is %d characters; the broker allows 128", len(first.RequestID))
	}

	// Asserted on the wire form, not the Go field. `requestId` is omitempty, so
	// an unset one does not travel as an empty string — it disappears from the
	// object altogether, which is precisely how this shipped: the struct had
	// the field, the JSON did not, and the broker refused the call.
	wire, err := json.Marshal(first)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(wire), `"requestId"`) {
		t.Errorf("the request sent to the broker has no requestId: %s", wire)
	}

	// A different change must not reuse it: the backend refuses an identifier
	// replayed with settings that do not match its receipt.
	_, second, _ := judgeSettingsPreview(previewOf(11501))
	if second.RequestID == first.RequestID {
		t.Error("two different changes share one identifier; the second would be refused")
	}
}

// TestSettingsRestartIsConfirmed checks a change that restarts the engine asks
// first, and that the confirmation applies the same request it armed.
func TestSettingsRestartIsConfirmed(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true}}
	d.refreshEngines()

	normalized := enginesettings.Config{ServerPort: 11500, ProxyPort: 11435}
	restarting := enginePreviewMsg{
		request: enginesettings.Request{Engine: "ollama", ExpectedRevision: 7},
		preview: enginesettings.Preview{Settings: normalized, Restart: true},
	}

	if verdict, _, _ := judgeSettingsPreview(restarting); verdict != settingsConfirmFirst {
		t.Fatalf("a restarting change was not held for confirmation (verdict %v)", verdict)
	}

	if cmd := d.applySettingsPreview(restarting); cmd != nil {
		t.Fatal("a restarting change was sent without asking")
	}
	if d.settingsConfirm == nil {
		t.Fatal("no confirmation was armed")
	}
	if got := d.status.render(); !strings.Contains(got, "restart") {
		t.Errorf("prompt %q does not say the engine will restart", got)
	}

	// Anything other than y walks away, and the armed request goes with it.
	if cmd := d.resolveSettingsConfirm(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("n")}); cmd != nil {
		t.Error("a non-confirming key still applied the change")
	}
	if d.settingsConfirm != nil {
		t.Error("the armed request outlived the cancellation")
	}

	d.applySettingsPreview(restarting)
	armed := d.settingsConfirm
	if armed == nil || armed.Settings.ServerPort != normalized.ServerPort {
		t.Fatalf("armed the wrong request: %+v", armed)
	}
	// The identifier is minted when the change is judged, not when it is sent,
	// so confirming is a replay of the arming rather than a second write.
	if armed.RequestID == "" {
		t.Error("the armed request has no identifier, so confirming it would be refused")
	}
	if cmd := d.resolveSettingsConfirm(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("y")}); cmd == nil {
		t.Error("y did not apply the armed change")
	}
}

// TestSettingsPreviewFailuresAreExplained checks a rejected draft says why and
// saves nothing.
func TestSettingsPreviewFailuresAreExplained(t *testing.T) {
	cases := []struct {
		name    string
		preview enginesettings.Preview
		err     error
		want    string
	}{
		{
			name:    "a field the backend rejected",
			preview: enginesettings.Preview{Errors: map[string]string{"launchText": "unbalanced quote"}},
			want:    "unbalanced quote",
		},
		{
			name:    "the two ports disagree",
			preview: enginesettings.Preview{Conflict: &enginesettings.Conflict{ServerPort: 11434, LaunchPort: 11500}},
			want:    "11500",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := localDetail()
			d.engines = []engineStatus{{Engine: "ollama", Installed: true}}
			d.refreshEngines()

			cmd := d.applySettingsPreview(enginePreviewMsg{
				request: enginesettings.Request{Engine: "ollama"},
				preview: tc.preview,
				err:     tc.err,
			})
			if cmd != nil {
				t.Error("a rejected draft was sent anyway")
			}
			if d.settingsConfirm != nil {
				t.Error("a rejected draft was armed for confirmation")
			}
			if got := d.status.render(); !strings.Contains(got, tc.want) {
				t.Errorf("status %q does not mention %q", got, tc.want)
			}
		})
	}
}

// TestSettingsErrorsReadTheSameEveryTime checks the per-field errors are
// ordered, since a map would reshuffle the same failure between attempts.
func TestSettingsErrorsReadTheSameEveryTime(t *testing.T) {
	errs := map[string]string{
		"serverPort": "port in use",
		"launchText": "unbalanced quote",
		"proxyPort":  "port in use",
	}
	first := joinSettingsErrors(errs)
	for i := 0; i < 20; i++ {
		if got := joinSettingsErrors(errs); got != first {
			t.Fatalf("attempt %d rendered %q, want the stable %q", i, got, first)
		}
	}
	if !strings.Contains(first, "unbalanced quote") {
		t.Errorf("rendered %q, want every field's message", first)
	}
}

// TestNoBindingRequiresShift is the guard for the mixed-case keyboard: needing
// shift for some keys and not others makes every press a guess. Named keys like
// shift+tab are exempt; this is about letters.
func TestNoBindingRequiresShift(t *testing.T) {
	bindings := map[string][]key.Binding{
		"global":         globalBindings(),
		"nodes":          newNodesView(nil).Help(),
		"jobs":           newJobsView(nil).Help(),
		"service":        newServiceView(nil).Help(),
		"logs":           newLogsView(nil).Help(),
		"errors overlay": newErrorsView(nil).Help(),
		"catalog":        newCatalogBrowser(nil, "ollama", "Ollama", "this-host", false).Help(),
		"detail engines": localDetail().Help(),
	}

	models := localDetail()
	models.pane = detailModels
	bindings["detail models"] = models.Help()

	for where, set := range bindings {
		for _, b := range set {
			for _, k := range b.Keys() {
				// A single upper-case letter is the shift-dependent case; the
				// named keys (esc, enter, shift+tab) are longer than one rune.
				if len(k) == 1 && k >= "A" && k <= "Z" {
					t.Errorf("%s: binding %q uses shift-dependent key %q", where, b.Help().Desc, k)
				}
			}
		}
	}
}

// TestSettingsOutcomeReportsTheBoundPort is the regression guard for a change
// that was refused and reported as done.
//
// A running engine outranks the proxy for a port, so the backend binds
// elsewhere and reports the difference as the effective port. Treating the
// absence of an error as success meant asking for a port an engine held
// produced "updated" while the table went on showing the old one.
func TestSettingsOutcomeReportsTheBoundPort(t *testing.T) {
	cases := []struct {
		name      string
		snapshot  enginesettings.Snapshot
		wantKind  toastKind
		wantHas   []string
		wantNotIn []string
	}{
		{
			// The proxy's port is read live from the proxy process, not
			// observed in passing, so a difference here is real.
			name: "the endpoint could not take the port",
			snapshot: enginesettings.Snapshot{
				Engine:              "lmstudio",
				Phase:               settingsPhaseSucceeded,
				Settings:            enginesettings.Config{ProxyPort: 1235, ServerPort: 1236},
				EffectiveProxyPort:  1234,
				EffectiveServerPort: 1236,
			},
			wantKind: toastError,
			// Both numbers: which port it is on, and which one it could not have.
			wantHas:   []string{"1234", "1235"},
			wantNotIn: []string{"saved"},
		},
		{
			// The backend refused it, and says why. Its words, not a guess
			// assembled from the ports.
			name: "the backend refused the change",
			snapshot: enginesettings.Snapshot{
				Engine:   "ollama",
				Phase:    settingsPhaseFailed,
				Error:    "port 11500 is reserved by another service",
				Settings: enginesettings.Config{ProxyPort: 11434, ServerPort: 11500},
			},
			wantKind:  toastError,
			wantHas:   []string{"reserved by another service"},
			wantNotIn: []string{"saved"},
		},
		{
			// The engine's effective port is observed when the apply replies,
			// and an engine that restarts onto the new port finishes after
			// that -- LM Studio's server re-launches detached. Reading failure
			// into the lag reported a move that had happened as one that had
			// not, naming a port nothing was listening on.
			name: "the engine port reading lags a successful move",
			snapshot: enginesettings.Snapshot{
				Engine:              "ollama",
				Phase:               settingsPhaseSucceeded,
				Settings:            enginesettings.Config{ProxyPort: 11434, ServerPort: 11500},
				EffectiveProxyPort:  11434,
				EffectiveServerPort: 11435,
			},
			wantKind:  toastOK,
			wantHas:   []string{"saved"},
			wantNotIn: []string{"11435"},
		},
		{
			name: "honoured",
			snapshot: enginesettings.Snapshot{
				Engine:              "ollama",
				Phase:               settingsPhaseSucceeded,
				Settings:            enginesettings.Config{ProxyPort: 11434, ServerPort: 11500},
				EffectiveProxyPort:  11434,
				EffectiveServerPort: 11500,
			},
			wantKind: toastOK,
			wantHas:  []string{"saved"},
		},
		{
			name: "nothing bound yet",
			snapshot: enginesettings.Snapshot{
				Engine:   "ollama",
				Phase:    settingsPhaseSucceeded,
				Settings: enginesettings.Config{ProxyPort: 11434, ServerPort: 11500},
			},
			// A stopped engine has no port in force. Saying it "stayed on :0"
			// would be worse than not saying where it landed.
			wantKind:  toastOK,
			wantNotIn: []string{":0"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := localDetail()
			snap := tc.snapshot
			snap.RequestID = "req-1"
			d.settingsAwaited = &awaitedSettings{engine: snap.Engine, requestID: "req-1"}
			d.reportSettingsOutcome(snap)

			if d.status.kind != tc.wantKind {
				t.Errorf("toast kind = %v, want %v", d.status.kind, tc.wantKind)
			}
			got := d.status.render()
			for _, want := range tc.wantHas {
				if !strings.Contains(got, want) {
					t.Errorf("message %q does not mention %q", got, want)
				}
			}
			for _, unwanted := range tc.wantNotIn {
				if strings.Contains(got, unwanted) {
					t.Errorf("message %q should not contain %q", got, unwanted)
				}
			}
		})
	}
}

// TestSettingsOutcomeIsSilentForSomeoneElsesChange checks the report is scoped
// to a change this screen made.
//
// These snapshots also arrive whenever the desktop app or another operator
// saves something, and a note on each would be noise about work the person at
// this terminal did not do.
func TestSettingsOutcomeIsSilentForSomeoneElsesChange(t *testing.T) {
	d := localDetail()
	snap := enginesettings.Snapshot{
		Engine:             "ollama",
		RequestID:          "ours",
		Phase:              settingsPhaseSucceeded,
		Settings:           enginesettings.Config{ProxyPort: 11434},
		EffectiveProxyPort: 11999,
	}

	d.reportSettingsOutcome(snap)
	if got := d.status.render(); got != "" {
		t.Errorf("reported %q for a change this screen did not make", got)
	}

	// The same engine, changed by someone else while this screen waits: the
	// desktop app, say. Matching on the engine alone consumed the await and
	// reported their result as ours.
	d.settingsAwaited = &awaitedSettings{engine: "ollama", requestID: "ours"}
	theirs := snap
	theirs.RequestID = "theirs"
	d.reportSettingsOutcome(theirs)
	if got := d.status.render(); got != "" {
		t.Errorf("reported another client's change as this screen's: %q", got)
	}
	if d.settingsAwaited == nil {
		t.Fatal("another client's snapshot spent this screen's await")
	}

	// And it speaks exactly once for a change it did make.
	d.reportSettingsOutcome(snap)
	if d.status.render() == "" {
		t.Fatal("said nothing about this screen's own change")
	}
	d.status = toast{}
	d.reportSettingsOutcome(snap)
	if got := d.status.render(); got != "" {
		t.Errorf("repeated the outcome as %q on a later snapshot", got)
	}
}

// TestAcceptedIsNotApplied is the regression guard for a verdict reported
// before the backend had one.
//
// The broker publishes a snapshot the moment it accepts a write — "applying",
// new settings saved, old port still in force — and only then stops the
// engine and does the work. Reading that as the outcome reported "saved"
// immediately; on a proxy change it reported a healthy move as a failure,
// since accepted and effective ports differ by definition at that point; and
// it spent the await, so the real verdict was never reported.
func TestAcceptedIsNotApplied(t *testing.T) {
	d := localDetail()
	d.settingsAwaited = &awaitedSettings{engine: "ollama", requestID: "req-1"}

	applying := enginesettings.Snapshot{
		Engine:    "ollama",
		RequestID: "req-1",
		Phase:     settingsPhaseApplying,
		Settings:  enginesettings.Config{ProxyPort: 11500, ServerPort: 11435},
		// Still where it was: the change has been accepted, not carried out.
		EffectiveProxyPort: 11434,
	}
	d.reportSettingsOutcome(applying)
	if got := d.status.render(); got != "" {
		t.Errorf("reported %q while the change was still applying", got)
	}
	if d.settingsAwaited == nil {
		t.Fatal("the applying snapshot spent the await, so the real verdict will be ignored")
	}

	// A never-configured engine reports no phase at all; that is not a verdict
	// either.
	idle := applying
	idle.Phase = ""
	d.reportSettingsOutcome(idle)
	if d.settingsAwaited == nil {
		t.Fatal("an empty phase was read as a verdict")
	}

	// The terminal snapshot for the same request is the one that speaks.
	done := applying
	done.Phase = settingsPhaseSucceeded
	done.EffectiveProxyPort = 11500
	d.reportSettingsOutcome(done)
	if got := d.status.render(); !strings.Contains(got, "saved") {
		t.Errorf("the terminal snapshot reported %q, want a save", got)
	}

	// And a genuine failure after "applying" is reported, not lost.
	d = localDetail()
	d.settingsAwaited = &awaitedSettings{engine: "ollama", requestID: "req-2"}
	applying.RequestID = "req-2"
	d.reportSettingsOutcome(applying)
	failed := applying
	failed.Phase = settingsPhaseFailed
	failed.Error = "engine did not become ready"
	d.reportSettingsOutcome(failed)
	if got := d.status.render(); !strings.Contains(got, "did not become ready") {
		t.Errorf("the failure after applying reported %q", got)
	}
}

// TestSlowApplyIsStillApplying checks a deadline on the apply reply is not
// reported as a failure.
//
// Accepted work is target-owned: the broker finishes a port swap whether or
// not anyone is still listening. Treating the deadline as a rejection reported
// failure on the path where the change was working, then dropped the await and
// the cached snapshot, so the real verdict that followed went unreported.
func TestSlowApplyIsStillApplying(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true}}
	d.refreshEngines()
	seedSettings(d, ollamaSettings())
	d.settingsAwaited = &awaitedSettings{engine: "ollama", requestID: "req-1"}

	cmd, _ := d.update(engineSettingsAppliedMsg{engine: "ollama", err: context.DeadlineExceeded})
	if cmd != nil {
		t.Error("a deadline triggered a re-read, as though the write had been rejected")
	}
	if d.settingsAwaited == nil {
		t.Fatal("a deadline cleared the await, so the verdict that follows will be ignored")
	}
	if _, kept := d.settings["ollama"]; !kept {
		t.Error("a deadline discarded the snapshot the in-flight write was based on")
	}
	if got := d.status.render(); strings.Contains(got, "fail") || !strings.Contains(got, "applying") {
		t.Errorf("status %q does not say the change is still applying", got)
	}

	d.reportSettingsOutcome(enginesettings.Snapshot{
		Engine: "ollama", RequestID: "req-1", Phase: settingsPhaseSucceeded,
	})
	if got := d.status.render(); !strings.Contains(got, "saved") {
		t.Errorf("the verdict after a deadline reported %q", got)
	}
}

// TestSettingsApplyBudgetIsTheBackends checks the apply call is given the time
// the backend works to, not the one sized for control calls.
func TestSettingsApplyBudgetIsTheBackends(t *testing.T) {
	if got := settingsApplyBudget(enginesettings.Request{Engine: "ollama"}); got != enginesettings.CallBudget {
		t.Errorf("a local apply is allowed %v, want the shared call budget %v",
			got, enginesettings.CallBudget)
	}
	if got := settingsApplyBudget(enginesettings.Request{NodeID: "peer", Engine: "ollama"}); got != enginesettings.RelayBudget {
		t.Errorf("a relayed apply is allowed %v, want the shared relay budget %v",
			got, enginesettings.RelayBudget)
	}
	if settingsApplyBudget(enginesettings.Request{}) <= callTimeout {
		t.Error("an apply is no longer allowed more than a control call")
	}
}

// TestSettingsPromptsHoldTheKeyboard checks the two settings waiting states
// keep the shell's global keys away.
//
// With a restart confirmation armed, q quit instead of answering it and tab
// left the restart armed behind a prompt no longer on screen. With a settings
// read in flight, the shell was free to act on keys that belong to the field
// about to open.
func TestSettingsPromptsHoldTheKeyboard(t *testing.T) {
	d := localDetail()
	d.settingsConfirm = &enginesettings.Request{Engine: "ollama"}
	if !d.CapturingInput() {
		t.Error("an armed restart confirmation does not hold the keyboard")
	}

	d = localDetail()
	d.settingsWanted = &pendingSettingsEdit{engine: "ollama", mode: detailInputEnginePort}
	if !d.CapturingInput() {
		t.Error("a settings read in flight does not hold the keyboard")
	}
}

// TestSettingsWriteFollowsTheFieldNotTheCursor is the regression guard for a
// value typed for one engine being written to another.
//
// A field can open after a settings read, and the cursor is free to move while
// the read is in flight. Building the request from the selection wrote the
// value typed into Ollama's field into LM Studio's settings.
func TestSettingsWriteFollowsTheFieldNotTheCursor(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{
		{Engine: "ollama", DisplayName: "Ollama", Installed: true, Port: 11435},
		{Engine: "lmstudio", DisplayName: "LM Studio", Installed: true, Port: 1235},
	}
	d.refreshEngines()
	seedSettings(d, ollamaSettings())
	lms := ollamaSettings()
	lms.Engine = "lmstudio"
	lms.Revision = 3
	seedSettings(d, lms)

	// Open Ollama's engine port field, then move the cursor to LM Studio.
	d.openSettingsField(d.settings["ollama"], detailInputEnginePort)
	d.engineTable.SetCursor(1)
	if sel := d.selectedEngine(); sel == nil || sel.Engine != "lmstudio" {
		t.Fatalf("setup: cursor is not on LM Studio (%+v)", sel)
	}

	if d.submitSettings(detailInputEnginePort, "11600", "11600") == nil {
		t.Fatal("the submit issued no preview")
	}
	// The submit names the engine it built the request for. Resolved from the
	// cursor, this read "checking LM Studio settings".
	got := d.status.render()
	if !strings.Contains(got, "Ollama") || strings.Contains(got, "LM Studio") {
		t.Errorf("status %q: the write was aimed at the cursor, not the field", got)
	}
	if d.settingsEngine != "" {
		t.Error("the field's engine outlived the submit")
	}
}

// TestLateSettingsReplyDoesNotOpenAnotherNodesEditor is the regression guard
// for one machine's settings being written to another.
//
// A settings read is not cancelled when its screen closes. Its reply landed on
// whichever detail screen was open by then, opened that node's field prefilled
// with the first node's value, and cached the first node's revision against it.
func TestLateSettingsReplyDoesNotOpenAnotherNodesEditor(t *testing.T) {
	v := newNodesView(nil)
	v.SetSize(100, 30)
	open := func(key string) *nodeDetail {
		d := newNodeDetail(nil, nodeRow{key: key, name: key, presence: presenceOnline})
		d.SetSize(100, 30)
		d.engines = []engineStatus{{Engine: "ollama", Installed: true}}
		d.refreshEngines()
		v.detail = d
		if d.editSetting(&d.engines[0], detailInputEnginePort) == nil {
			t.Fatalf("%s did not start a settings read", key)
		}
		return d
	}

	open("node-a")
	v.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if v.detail != nil {
		t.Fatal("node A's detail did not close while its settings read was pending")
	}
	second := open("node-b")

	stale := ollamaSettings()
	stale.NodeID = "node-a"
	stale.Settings.ServerPort = 11500
	v.Update(detailReply{node: "node-a", msg: engineSettingsMsg{snapshot: stale}})
	v.Update(detailReply{node: "node-a", msg: engineSettingsMsg{err: errors.New("node-a went away")}})
	v.Update(detailReply{node: "node-a", msg: enginePreviewMsg{request: enginesettings.Request{NodeID: "node-a", Engine: "ollama"}}})

	if second.mode != detailInputNone {
		t.Errorf("node A's reply opened node B's editor with port %q", second.input.Value())
	}
	if _, cached := second.settings["ollama"]; cached {
		t.Error("node A's settings reply was cached on node B")
	}
	if second.settingsWanted == nil {
		t.Error("node A's reply consumed node B's pending settings edit")
	}

	// And node B's own reply still does what it is for; without this the
	// assertions above would pass if every reply were dropped.
	own := ollamaSettings()
	own.NodeID = "node-b"
	v.Update(detailReply{node: "node-b", msg: engineSettingsMsg{snapshot: own}})
	if second.mode != detailInputEnginePort || second.input.Value() != "11434" {
		t.Errorf("node B's own reply did not open its field (mode %v, value %q)",
			second.mode, second.input.Value())
	}
}

// TestLateOperationReplyStaysWithItsNode is the regression guard for one
// machine's outcome being reported on another's screen.
//
// A slow operation's reply outlives the screen that sent it. Start something
// on node A, open node B, and A's "start failed" landed on B.
func TestLateOperationReplyStaysWithItsNode(t *testing.T) {
	b := newNodeDetail(nil, nodeRow{key: "node-b", name: "B", presence: presenceOnline})
	b.SetSize(100, 30)
	b.engines = []engineStatus{{Engine: "ollama", DisplayName: "Ollama", Installed: true}}
	b.refreshEngines()

	failed := classifyOpResult("start", "ollama", "start", errors.New("engine did not come up"))
	b.update(detailReply{node: "node-a", msg: failed})
	if got := b.status.render(); strings.Contains(got, "failed") {
		t.Errorf("node A's outcome was reported on node B: %q", got)
	}

	b.update(detailReply{node: "node-b", msg: failed})
	if got := b.status.render(); !strings.Contains(got, "start (Ollama) failed") {
		t.Errorf("node B's own outcome was not reported: %q", got)
	}
}

// TestDetailRepliesCarryTheirNode checks the other half of the envelope: a
// reply leaves addressed to the screen that asked, whatever screen is open by
// the time it lands.
func TestDetailRepliesCarryTheirNode(t *testing.T) {
	a := newNodeDetail(nil, nodeRow{key: "node-a", name: "A", presence: presenceOnline})
	reply := engineOpMsg{what: "start", engine: "ollama"}
	got := a.own(func() tea.Msg { return reply })()
	tagged, ok := got.(detailReply)
	if !ok || tagged.node != "node-a" || tagged.msg != reply {
		t.Errorf("reply left as %#v, want it addressed to node-a", got)
	}
	if a.own(nil) != nil {
		t.Error("no command was turned into one")
	}
	if msg := a.own(func() tea.Msg { return nil })(); msg != nil {
		t.Errorf("a command with nothing to say was given an envelope: %#v", msg)
	}
}

// TestSecondSettingsChangeWaitsForTheFirst is the regression guard for a
// second save replacing the first before its verdict came back. The screen
// follows one outstanding write, so the first outcome was never reported.
func TestSecondSettingsChangeWaitsForTheFirst(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{
		{Engine: "ollama", DisplayName: "Ollama", Installed: true},
		{Engine: "lmstudio", DisplayName: "LM Studio", Installed: true},
	}
	d.refreshEngines()
	d.settingsAwaited = &awaitedSettings{engine: "ollama", requestID: "first"}

	if cmd := d.editSetting(&d.engines[1], detailInputEnginePort); cmd != nil {
		t.Error("a second settings change started while the first was still applying")
	}
	if d.mode != detailInputNone || d.settingsWanted != nil {
		t.Error("a second settings field was opened while the first change was applying")
	}
	if got := d.status.render(); !strings.Contains(got, "wait for the Ollama settings change") {
		t.Errorf("the refusal did not say what to wait for: %q", got)
	}
}

// progressPush builds a progress notification from its fields as sent.
func progressPush(method string, fields map[string]any) NotificationMsg {
	params, _ := json.Marshal(fields)
	return NotificationMsg{Msg: &rpc.Message{Method: method, Params: params}}
}

// TestInstallProgressStaysOnThisMachine checks that a local install's progress
// is not reported on a peer's screen. The payload carries no node, so shown
// there it read as the peer installing something.
func TestInstallProgressStaysOnThisMachine(t *testing.T) {
	frame := progressPush("engine:install-progress",
		map[string]any{"engine": "ollama", "stage": "downloading", "percent": 40})

	peer := remoteDetail()
	peer.update(frame)
	if got := peer.status.render(); strings.Contains(got, "install") {
		t.Errorf("this machine's install was reported on a peer's screen: %q", got)
	}

	local := localDetail()
	local.update(frame)
	if got := local.status.render(); !strings.Contains(got, "downloading (40%)") {
		t.Errorf("this machine's install progress was not shown: %q", got)
	}
}

// TestProgressWithoutAPercentShowsNone is the regression guard for progress
// frames read as stuck downloads. The engine manager leaves the percent out
// while progress is indeterminate and sends -1 on failure; printed, those read
// "(0%)" and "(-1%)".
func TestProgressWithoutAPercentShowsNone(t *testing.T) {
	peer := remoteDetail()
	snapshot := tuiDownloadSnapshot()
	peer.applyModels(tuiDownloadInventory(snapshot), true)
	peer.update(tuiDownloadProgress(t, "engine:remote-progress", "peer", snapshot, snapshot.Entries[0]))
	if got := peer.status.render(); !strings.Contains(got, "Downloading") || strings.Contains(got, "%") {
		t.Errorf("indeterminate remote progress rendered as %q", got)
	}
	percent := 25
	snapshot.Revision, snapshot.Entries[0].Percent = "8", &percent
	peer.update(tuiDownloadProgress(t, "engine:remote-progress", "peer", snapshot, snapshot.Entries[0]))
	if got := peer.status.render(); !strings.Contains(got, "Downloading (25%)") {
		t.Errorf("a reported percent was dropped: %q", got)
	}

	local := localDetail()
	snapshot.Entries[0].Percent = nil
	local.applyModels(tuiDownloadInventory(snapshot), true)
	local.update(tuiDownloadProgress(t, "engine:pull-progress", "", snapshot, snapshot.Entries[0]))
	if got := local.status.render(); strings.Contains(got, "%") {
		t.Errorf("indeterminate pull progress rendered as %q", got)
	}

	local.update(progressPush("engine:install-progress",
		map[string]any{"engine": "ollama", "stage": "failed", "percent": -1, "error": "disk full"}))
	got := local.status.render()
	if strings.Contains(got, "%") || !strings.Contains(got, "install") || !strings.Contains(got, "failed: disk full") {
		t.Errorf("a failed install rendered as %q", got)
	}
}

// TestHiddenListsRefuseTheirKeys is the regression guard for keys acting on a
// list the frame had no room to show: on a short terminal a delete could be
// armed against a model the operator could not see.
func TestHiddenListsRefuseTheirKeys(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", DisplayName: "Ollama", Installed: true, Running: true}}
	models := []string{"model-a", "model-b", "model-c"}
	d.models = modelsResult{Models: models, ModelsByEngine: map[string][]string{"ollama": models}}
	d.refreshEngines()
	d.refreshModels()

	modelsOnly, both := 0, 0
	for h := 30; h > 0; h-- {
		d.SetSize(100, h)
		d.View()
		if d.modelsHidden && !d.enginesHidden && modelsOnly == 0 {
			modelsOnly = h
		}
		if d.enginesHidden && both == 0 {
			both = h
		}
	}
	if modelsOnly == 0 || both == 0 {
		t.Fatalf("no height hid the lists (models only at %d, both at %d)", modelsOnly, both)
	}

	d.SetSize(100, 30)
	d.View()
	if d.enginesHidden || d.modelsHidden {
		t.Fatal("a tall screen still counted a list as hidden")
	}

	d.SetSize(100, modelsOnly)
	d.View()
	d.pane = detailModels
	d.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	if d.pending != nil {
		t.Error("a delete was armed against a model list that was not on screen")
	}
	if got := d.status.render(); !strings.Contains(got, "too little room to show the models") {
		t.Errorf("the refusal did not say why: %q", got)
	}

	d.SetSize(100, both)
	d.View()
	d.pane = detailEngines
	if cmd, _ := d.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("s")}); cmd != nil {
		t.Error("an engine was started from an engine list that was not on screen")
	}
}

// TestOpenDetailFollowsPresenceAndMembership is the regression guard for a
// detail screen that kept describing a peer as it was when the screen opened.
//
// Presence and membership are merged from feeds the detail does not read — the
// cluster roster among them — so a peer that left the cluster or dropped off
// the network went on reading "Member" and "Online" until the screen was
// reopened, with its engine controls still offered.
func TestOpenDetailFollowsPresenceAndMembership(t *testing.T) {
	v := newNodesView(nil)
	v.SetSize(100, 30)
	peer := availableNode{HostUUID: "peer", Name: "peer-host", IPAddress: "10.0.0.2", Port: 9000, Trusted: true}
	v.feeds.discovered = []availableNode{peer}
	v.rebuild()
	v.selectedKey = "peer"
	v.restoreSelection()
	v.openDetail()
	d := v.detail
	d.engines = []engineStatus{{Engine: "ollama", Installed: true, Running: true}}
	d.refreshEngines()
	if d.node.membership != membershipMember || d.node.presence != presenceOnline {
		t.Fatalf("setup: detail opened as %v/%v", d.node.presence, d.node.membership)
	}

	// The peer leaves the cluster but is still on the network.
	peer.Trusted = false
	v.Update(discoveryPush(peer))
	if d.node.membership == membershipMember {
		t.Error("the detail still reads Member after the peer left the cluster")
	}
	if len(d.engines) != 0 {
		t.Error("engine controls are still offered for a node no longer in the cluster")
	}
	if !contains(d.View(), "not in this cluster") {
		t.Errorf("the engines pane does not say why the list went: %q", d.View())
	}

	// And then drops off the network entirely.
	v.Update(discoveryPush())
	if d.node.presence == presenceOnline {
		t.Error("the detail still reads Online after the node dropped out of every feed")
	}
}

// TestEngineNameDoesNotDependOnTheEngineFetch is the regression guard for the
// same engine reading "Ollama" on one machine and "ollama" on another.
//
// A remote node's models come from discovery while its engine list comes from a
// separate call, so the models table routinely has rows while the engine list is
// still empty — the manager not running, the read failing, or a reply that
// genuinely lists nothing. Resolving the name only against that list meant the
// spelling depended on whether an unrelated fetch had landed.
func TestEngineNameDoesNotDependOnTheEngineFetch(t *testing.T) {
	withList := localDetail()
	withList.engines = []engineStatus{
		{Engine: "ollama", DisplayName: "Ollama"},
		{Engine: "lmstudio", DisplayName: "LM Studio"},
	}
	empty := localDetail() // no engine list at all

	for _, engine := range []string{"ollama", "lmstudio"} {
		want := withList.engineLabel(engine)
		if got := empty.engineLabel(engine); got != want {
			t.Errorf("engine %q reads %q with no engine list but %q with one",
				engine, got, want)
		}
		if want == engine {
			t.Errorf("engine %q resolved to its own wire id, not a display name", engine)
		}
	}

	// A genuinely unknown engine still has to render as something.
	if got := empty.engineLabel("some-new-engine"); got != "some-new-engine" {
		t.Errorf("unknown engine rendered %q, want the id passed through", got)
	}
}

// TestEmptyEngineListExplainsItself checks the empty states read differently.
// The manager answering with nothing, the manager not answering, and the node
// being one we cannot ask at all have different causes, and the operator's
// next move differs for each.
func TestEmptyEngineListExplainsItself(t *testing.T) {
	local := localDetail()
	if got := local.emptyEnginesHint(); !strings.Contains(got, "engine manager") {
		t.Errorf("local hint does not point at the engine manager: %q", got)
	}

	member := remoteDetail()
	member.node.membership = membershipMember
	if got := member.emptyEnginesHint(); !strings.Contains(got, "this node") {
		t.Errorf("remote member hint does not attribute the gap to the peer: %q", got)
	}

	if local.emptyEnginesHint() == member.emptyEnginesHint() {
		t.Error("local and remote read identically; the causes are different")
	}
}

// TestUnpairedNodeIsNotCalledSilent is the regression guard for a machine that
// was answering perfectly well being reported as unresponsive.
//
// A peer's engines are fetched over pin-based mTLS, so a node outside the
// cluster cannot be asked at all. That call was made anyway, and its failure
// rendered as "not answering" — beside a hardware readout, polled over an
// endpoint that needs no pairing, visibly updating for the same machine.
func TestUnpairedNodeIsNotCalledSilent(t *testing.T) {
	cases := map[nodeMembership]string{
		membershipNone:    "not in this cluster",
		membershipForeign: "another cluster",
		membershipPending: "still pairing",
	}
	for membership, want := range cases {
		d := remoteDetail()
		d.node.membership = membership

		if d.enginesCmd() != nil {
			t.Errorf("%v: asked for engines over a link that cannot carry the question", membership)
		}
		got := d.emptyEnginesHint()
		if !strings.Contains(got, want) {
			t.Errorf("%v: hint %q does not say %q", membership, got, want)
		}
		if strings.Contains(got, "not answering") {
			t.Errorf("%v: hint %q calls a reachable node silent", membership, got)
		}
		// Models come from what the node advertises over discovery, which
		// needs no pairing, so the models pane must not blame a stopped engine
		// for what it cannot see either way.
		d.node.presence = presenceOnline
		if m := d.emptyModelsHint(); strings.Contains(m, "start an engine") {
			t.Errorf("%v: models hint %q claims an engine needs starting", membership, m)
		}
	}

	// A member is still asked, and is still allowed to be silent.
	member := remoteDetail()
	member.node.membership = membershipMember
	if member.enginesCmd() == nil {
		t.Error("a cluster member was not asked for its engines")
	}
}

// TestUnpairedHintAgreesWithTheModelList checks the engines pane does not deny
// what the pane below it is already showing.
//
// Discovery carries which engine serves each model and needs no pairing to do
// it, so an unpaired node's model list arrives with its ENGINE column filled
// in. "Its engines are not visible from here", printed directly above a list
// naming Ollama, claimed less than the screen displayed. What pairing actually
// buys is their state and their controls.
func TestUnpairedHintAgreesWithTheModelList(t *testing.T) {
	d := remoteDetail()
	d.node.membership = membershipNone
	d.models = modelsResult{ModelsByEngine: map[string][]string{
		"ollama": {"gemma3:4b", "gemma4:12b"},
	}}
	d.refreshModels()

	hint := d.emptyEnginesHint()
	if !strings.Contains(hint, "Ollama") {
		t.Errorf("hint %q does not name the engine the model list attributes rows to", hint)
	}
	if !strings.Contains(hint, "manage") {
		t.Errorf("hint %q does not say what pairing would actually add", hint)
	}

	// With nothing advertised there is nothing to name, and the sentence must
	// not trail off into an empty list.
	bare := remoteDetail()
	bare.node.membership = membershipNone
	if got := bare.emptyEnginesHint(); strings.Contains(got, "advertises") {
		t.Errorf("hint %q claims advertised engines for a node reporting none", got)
	}
}

// TestNodeKeysDoNotCollide checks no two verbs on the Nodes tab claim the same
// key, and that none of them shadows a shell binding.
//
// A collision is silent: whichever case the switch reaches first wins and the
// other verb simply stops working, with the footer still advertising it. The
// risk is concentrated here because this tab has eleven verbs competing for one
// letter each, and they have been renamed more than once — pair moved from i to
// p and accept from p to a, which is exactly the edit that lands two verbs on
// one key if the whole set is not considered at once.
func TestNodeKeysDoNotCollide(t *testing.T) {
	nodeKeys := map[string]key.Binding{
		"details":         nodeDetailKey,
		"pair":            nodeInviteKey,
		"pair by address": nodeInviteAddrKey,
		"find by address": nodeAddKey,
		"remove":          nodeRemoveKey,
		"accept pairing":  nodePairKey,
		"decline":         nodeDeclineKey,
		"leave cluster":   nodeLeaveKey,
		"cancel invite":   nodeCancelKey,
		"filter":          nodeFilterKey,
		"confirm":         nodeConfirmKey,
	}

	// esc is deliberately excluded: nodeClearKey shares it with the universal
	// "get out of here" gesture, and both mean the same thing.
	seen := map[string]string{}
	for verb, binding := range nodeKeys {
		for _, k := range binding.Keys() {
			if other, dup := seen[k]; dup {
				t.Errorf("key %q is bound to both %q and %q", k, other, verb)
			}
			seen[k] = verb
		}
	}

	shell := newGlobalKeyMap(len(defaultViews(nil)))
	for _, g := range []key.Binding{shell.NextTab, shell.PrevTab, shell.JumpTab, shell.Help, shell.Quit} {
		for _, k := range g.Keys() {
			if verb, clash := seen[k]; clash {
				t.Errorf("node verb %q claims %q, which the shell uses for %q",
					verb, k, g.Help().Desc)
			}
		}
	}
}

// globalBindings is the shell's own key set, for the shift audit above, built
// for the real tab count so the digit binding matches what ships.
func globalBindings() []key.Binding {
	k := newGlobalKeyMap(len(defaultViews(nil)))
	return []key.Binding{k.NextTab, k.PrevTab, k.JumpTab, k.Help, k.Quit}
}

// TestDetailResizesWhenHardwareArrives is the regression guard for the status
// line falling off the frame. The model table is sized against the hardware
// block's height, and that height changes when a telemetry reading lands.
func TestDetailResizesWhenHardwareArrives(t *testing.T) {
	const budget = 20
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true, Running: true}}
	models := make([]string, 30)
	for i := range models {
		models[i] = "model-" + string(rune('a'+i%26))
	}
	d.models = modelsResult{Models: models, ModelsByEngine: map[string][]string{"ollama": models}}
	d.SetSize(100, budget)
	d.refreshEngines()
	d.refreshModels()
	d.status.error("something to push off the bottom")

	if got := renderedRows(d.View()); got > budget {
		t.Fatalf("setup already overflows: %d rows into %d", got, budget)
	}

	// A dual-GPU reading is four hardware lines instead of the unavailable one.
	d.update(nodeTelemetryMsg{nodeKey: d.node.key, self: true, gen: d.telemetryGen, telemetry: nodeTelemetry{
		TelemetryValid: true,
		GPUs: []noderec.GPUInfo{
			{Name: "GPU 0", VramBytes: 1 << 30},
			{Name: "GPU 1", VramBytes: 1 << 30},
		},
		CPU:    &noderec.CPUInfo{Name: "CPU", Cores: 8},
		Memory: &noderec.MemoryInfo{TotalBytes: 1 << 34, UsedBytes: 1 << 33},
	}})

	if d.hardwareHeight() < 4 {
		t.Errorf("hardwareHeight = %d, want one row per reading", d.hardwareHeight())
	}
	// The frame is what matters, not any particular table's height. Asserting on
	// the height instead measured a value SetSize wrote and the renderer then
	// overwrote — so it passed whether or not the frame actually fit.
	if got := renderedRows(d.View()); got > budget {
		t.Errorf("rendered %d rows into %d after the hardware block grew; "+
			"the shell will delete the status line", got, budget)
	}
	if !contains(d.View(), "something to push off the bottom") {
		t.Error("the status line was squeezed out by the hardware block")
	}
}

// TestDetailResizesCatalogBrowser checks the browser follows a terminal resize
// rather than keeping the size it was opened at.
func TestDetailResizesCatalogBrowser(t *testing.T) {
	d := localDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true, Running: true}}
	d.refreshEngines()
	d.openCatalog()
	if d.catalog == nil {
		t.Fatal("catalog did not open")
	}

	d.SetSize(140, 40)
	if d.catalog.width != 140 || d.catalog.height != 40 {
		t.Errorf("catalog is %dx%d after resize, want 140x40", d.catalog.width, d.catalog.height)
	}
}

// TestModelSelectionSurvivesARebuild is the regression guard for a delete
// landing on the wrong model.
//
// The list is sorted and rebuilt wholesale whenever a download finishes or a
// peer republishes its inventory, so a row inserted above the cursor shifts
// everything below it. An earlier attempt at this captured the selection after
// the rebuild, which reads back whatever now sits at the old index — the very
// row the cursor slid onto — so it restored nothing.
func TestModelSelectionSurvivesARebuild(t *testing.T) {
	d := localDetail()
	d.models = modelsResult{
		ModelsByEngine: map[string][]string{"ollama": {"bravo", "charlie", "delta"}},
	}
	d.refreshModels()
	d.modelTable.SetCursor(1)
	if got := d.selectedModel(); got == nil || got.model != "charlie" {
		t.Fatalf("setup: selected %v", got)
	}

	// A background pull lands "alpha", which sorts first.
	d.models = modelsResult{
		ModelsByEngine: map[string][]string{"ollama": {"alpha", "bravo", "charlie", "delta"}},
	}
	d.refreshModels()

	got := d.selectedModel()
	if got == nil {
		t.Fatal("nothing selected after the rebuild")
	}
	if got.model != "charlie" {
		t.Errorf("selection slid to %q; a delete would now destroy the wrong model", got.model)
	}
}

// TestActionsWorkAfterAnEmptyRefresh is the regression guard for keys that went
// dead on the normal startup path.
//
// bubbles clamps an out-of-range cursor to len-1, so a table handed zero rows
// lands on -1 and stays there — refilling it never moves a cursor already below
// the range. A local node's inventory is empty until engine:models replies, so
// every model action reported "no model selected" for the life of the screen.
func TestActionsWorkAfterAnEmptyRefresh(t *testing.T) {
	d := localDetail() // constructed with no inventory, as at startup

	d.models = modelsResult{ModelsByEngine: map[string][]string{"ollama": {"a", "b"}}}
	d.refreshModels()
	if d.selectedModel() == nil {
		t.Errorf("no model selected after the inventory arrived (cursor %d); "+
			"load, eject, and delete are all dead", d.modelTable.Cursor())
	}

	// Same for the engines pane, which gates install/start/stop.
	d.engines = []engineStatus{{Engine: "ollama", Installed: true}}
	d.refreshEngines()
	if d.selectedEngine() == nil {
		t.Errorf("no engine selected after engines arrived (cursor %d)",
			d.engineTable.Cursor())
	}
}

// TestArmedActionOwnsTheKeyboard checks a pending confirmation cannot be
// escaped by a global key, which would leave it armed behind an off-screen
// prompt for whatever the operator pressed on returning.
func TestArmedActionOwnsTheKeyboard(t *testing.T) {
	d := localDetail()
	d.models = modelsResult{ModelsByEngine: map[string][]string{"ollama": {"victim"}}}
	d.refreshModels()
	d.pane = detailModels

	d.deleteSelectedModel()
	if d.pending == nil {
		t.Fatal("delete did not arm")
	}
	if !d.CapturingInput() {
		t.Error("an armed action does not capture input, so tab or a digit escapes it")
	}
	if !contains(d.status.render(), "press y to confirm") {
		t.Error("the confirmation prompt is not on screen")
	}
	// The prompt must not expire out from under the armed state.
	if d.status.expired() {
		t.Error("the confirmation prompt expires while the action stays armed")
	}
}

// TestArmedActionRunsOnConfirmAndNotOtherwise covers the half of the gate that
// was never tested: that `y` actually runs the captured action and that any
// other key abandons it. Arming was covered; confirming was not, on the one
// screen whose arming implementation the others were copied from.
func TestArmedActionRunsOnConfirmAndNotOtherwise(t *testing.T) {
	build := func() *nodeDetail {
		d := localDetail()
		d.models = modelsResult{ModelsByEngine: map[string][]string{"ollama": {"victim"}}}
		d.refreshModels()
		d.pane = detailModels
		d.deleteSelectedModel()
		return d
	}

	// Any other key cancels, and says so.
	d := build()
	if cmd := d.handleKeyForTest(t, "n"); cmd != nil {
		t.Error("a non-confirming key ran the destructive action")
	}
	if d.pending != nil {
		t.Error("the action stayed armed after being cancelled")
	}
	if !contains(d.status.render(), "cancelled") {
		t.Errorf("cancelling said nothing: %q", d.status.render())
	}

	// y runs it and disarms.
	d = build()
	if cmd := d.handleKeyForTest(t, "y"); cmd == nil {
		t.Error("confirming produced no command; the delete never ran")
	}
	if d.pending != nil {
		t.Error("the action stayed armed after being confirmed")
	}
}

// TestUninstallRefusedOnAPeerBeforeArming checks the refusal comes before the
// confirmation, not after it. Staging a gigabyte-destroying prompt and then
// answering that the operation is unavailable is worse than refusing outright.
func TestUninstallRefusedOnAPeerBeforeArming(t *testing.T) {
	d := remoteDetail()
	d.engines = []engineStatus{{Engine: "ollama", Installed: true}}
	d.refreshEngines()
	d.pane = detailEngines

	d.handleKeyForTest(t, "u")

	if d.pending != nil {
		t.Error("armed a confirmation for an uninstall that cannot run on a peer")
	}
	if !contains(d.status.render(), "machine running it") {
		t.Errorf("no explanation given: %q", d.status.render())
	}
}

// handleKeyForTest presses one key through the detail screen's key handling.
func (d *nodeDetail) handleKeyForTest(t *testing.T, k string) tea.Cmd {
	t.Helper()
	cmd, _ := d.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(k)})
	return cmd
}

func TestParsePort(t *testing.T) {
	valid := map[string]int{"1": 1, "11434": 11434, "65535": 65535}
	for in, want := range valid {
		if got, ok := parsePort(in); !ok || got != want {
			t.Errorf("parsePort(%q) = %d, %v", in, got, ok)
		}
	}
	for _, in := range []string{"", "0", "-1", "65536", "abc", "80x"} {
		if _, ok := parsePort(in); ok {
			t.Errorf("parsePort(%q) accepted an invalid port", in)
		}
	}
}

// TestServiceTabNoLongerOffersPorts checks the ports are configured in exactly
// one place, not two.
func TestServiceTabNoLongerOffersPorts(t *testing.T) {
	v := newServiceView(nil)
	for _, it := range v.items {
		if strings.Contains(strings.ToLower(it.label), "port") {
			t.Errorf("Service tab still offers %q; ports belong on the node", it.label)
		}
	}
}

// TestUndeclaredModelOperationIsRefusedBeforeArming checks a model operation
// this machine's descriptor says the engine lacks is refused at the keypress,
// naming the engine by its descriptor label, rather than armed for a
// confirmation and then failed by the engine manager.
func TestUndeclaredModelOperationIsRefusedBeforeArming(t *testing.T) {
	d := localDetail()
	d.descriptors = engineDescriptors{
		"strata": {
			Version: engines.DescriptorVersion, Engine: "strata", Family: "strata", Label: "Strata",
			ModelNaming: engines.ModelNamingExact,
			Operations:  map[string]engines.OperationCapability{engines.OperationLoad: {}},
		},
	}
	snapshot := tuiDownloadSnapshot()
	snapshot.Entries = []modeldownload.Entry{}
	if !d.applyModels(modelsResult{ModelsByEngine: map[string][]string{"strata": {"victim"}}, DownloadState: snapshot, current: true}, true) {
		t.Fatal("ready canonical inventory rejected")
	}
	d.pane = detailModels

	if cmd := d.deleteSelectedModel(); cmd != nil || d.pending != nil {
		t.Fatal("an undeclared delete was armed")
	}
	if !contains(d.status.render(), "Strata does not support delete") {
		t.Errorf("refusal not explained: %q", d.status.render())
	}
	if cmd := d.modelOp("load"); cmd == nil {
		t.Error("a declared load was refused")
	}
	if got := d.engineLabel("strata"); got != "Strata" {
		t.Errorf("engine label = %q, want the descriptor's", got)
	}
}
