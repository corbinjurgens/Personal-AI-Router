// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

const typedSettingsLoadRefusal = "settings.modelLoads:"

// typedClientPreview builds the request both clients build: the saved snapshot
// with one field edited, the saved revision and no request identifier
// (desktop EngineSettingsSection check, TUI settingsRequest).
func typedClientPreview(t *testing.T, f *modelBufferProcessFixture, resolution string, edit func(*settings.Config)) (settings.Request, settings.Preview) {
	t.Helper()
	saved := f.snapshot(t)
	request := settings.Request{Engine: "llamacpp", ExpectedRevision: saved.Revision, Settings: saved.Settings.Clone(), Resolution: resolution}
	edit(&request.Settings)
	var preview settings.Preview
	modelSettingsDecode(t, f.rpcOK(t, "engine:preview-settings", request), &preview)
	if len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("client preview: errors=%v conflict=%v", preview.Errors, preview.Conflict)
	}
	return request, preview
}

// typedClientCommit is the commit both clients send after a clean preview: the
// preview's settings whole, no resolution, and a new request identifier
// (desktop confirmedRequest, TUI judgeSettingsPreview).
func typedClientCommit(request settings.Request, preview settings.Preview, id string) settings.Request {
	request.Settings = preview.Settings.Clone()
	request.Resolution = ""
	request.RequestID = id
	return request
}

// typedClientEdit runs the whole client flow and returns the settled snapshot.
func typedClientEdit(t *testing.T, f *modelBufferProcessFixture, id, resolution string, edit func(*settings.Config)) (settings.Preview, settings.Snapshot) {
	t.Helper()
	request, preview := typedClientPreview(t, f, resolution, edit)
	commit := typedClientCommit(request, preview, id)
	if response := f.rpc(t, "engine:apply-settings", commit); response.Error != nil {
		t.Fatalf("whole preview apply refused: code=%d message=%s", response.Error.Code, response.Error.Message)
	}
	state := f.settled(t, id, "succeeded")
	if !state.Settings.Equal(preview.Settings) {
		t.Fatal("saved settings differ from the preview the client applied")
	}
	return preview, state
}

func typedSourceDigest(t *testing.T, config settings.Config) string {
	t.Helper()
	if len(config.ModelLoads) == 0 || config.ModelLoads[0].Resolved == nil || len(config.ModelLoads[0].Resolved.SourceInputDigest) != 64 {
		t.Fatalf("configuration has no typed source digest: %+v", config.ModelLoads)
	}
	return config.ModelLoads[0].Resolved.SourceInputDigest
}

func typedDeclaration(t *testing.T, f *modelBufferProcessFixture) *engines.ServingConfig {
	t.Helper()
	declaration := &engines.ServingConfig{Version: 1}
	for _, name := range []string{"medium", "weak"} {
		model, err := os.ReadFile(filepath.Join(f.root, name+".gguf"))
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(model)
		ram, vram := int64(1048576), int64(0)
		declaration.Models = append(declaration.Models, engines.ServingModel{
			ModelID: name + "-model", MaxOutputTokens: 512,
			Memory:    engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram},
			Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: hex.EncodeToString(digest[:]), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}},
		})
	}
	return declaration
}

// typedAssertRefused sends one apply and requires a refusal that changed
// nothing: not the revision, the settings, the receipt or the journal bytes.
func typedAssertRefused(t *testing.T, f *modelBufferProcessFixture, request settings.Request, want string) {
	t.Helper()
	before := f.snapshot(t)
	journal := readLiveDurableFile(t, filepath.Join(f.base, "engine-settings-operations.json"))
	response := f.rpc(t, "engine:apply-settings", request)
	if response.Error == nil {
		t.Fatalf("apply %s was accepted", request.RequestID)
	}
	if !strings.Contains(response.Error.Message, want) {
		t.Fatalf("apply %s refused as %d %q, want %q", request.RequestID, response.Error.Code, response.Error.Message, want)
	}
	after := f.snapshot(t)
	if after.Revision != before.Revision || !after.Settings.Equal(before.Settings) || after.RequestID != before.RequestID || after.Phase != before.Phase {
		t.Fatalf("refused apply %s changed saved state: revision %d/%d", request.RequestID, before.Revision, after.Revision)
	}
	if !bytes.Equal(journal, readLiveDurableFile(t, filepath.Join(f.base, "engine-settings-operations.json"))) {
		t.Fatalf("refused apply %s wrote the settings journal", request.RequestID)
	}
}

// typedAssertForgedLoadsRefused sends otherwise reviewed commits that carry
// caller-chosen load values. None of them is the target's saved load set or the
// set its own compile produces.
func typedAssertForgedLoadsRefused(t *testing.T, f *modelBufferProcessFixture, label string, serverPort int) {
	t.Helper()
	request, preview := typedClientPreview(t, f, "server", func(c *settings.Config) { c.ServerPort = serverPort })
	for _, forged := range []struct {
		name   string
		change func(*[]modelconfig.ModelLoad)
	}{
		{"replaced allocation", func(loads *[]modelconfig.ModelLoad) {
			load := &(*loads)[0]
			r := load.Resolved
			r.PerSlotContext, r.TotalContext = r.PerSlotContext*2, r.TotalContext*2
			if load.Requested.Context != nil {
				load.Requested.Context.Tokens = r.PerSlotContext
			}
			if r.Options.Context != nil {
				r.Options.Context.Tokens = r.PerSlotContext
			}
		}},
		{"replaced artifact path", func(loads *[]modelconfig.ModelLoad) {
			(*loads)[0].Resolved.Artifacts[0].Path += ".other"
			(*loads)[0].Resolved.Artifacts[0].ResolvedTarget += ".other"
		}},
		{"replaced build", func(loads *[]modelconfig.ModelLoad) {
			(*loads)[0].Resolved.NativeBuildID = strings.Repeat("9", 64)
		}},
		{"forged source digest", func(loads *[]modelconfig.ModelLoad) {
			(*loads)[0].Resolved.SourceInputDigest = strings.Repeat("9", 64)
		}},
		{"forged preset digest", func(loads *[]modelconfig.ModelLoad) {
			(*loads)[0].Resolved.PresetDigest = strings.Repeat("9", 64)
		}},
		{"removed resolution", func(loads *[]modelconfig.ModelLoad) {
			(*loads)[0].Resolved = nil
		}},
		{"removed load", func(loads *[]modelconfig.ModelLoad) {
			*loads = (*loads)[1:]
		}},
		{"introduced load", func(loads *[]modelconfig.ModelLoad) {
			extra := (*loads)[0]
			extra.ProfileKey = "introduced-model"
			*loads = append(*loads, extra)
		}},
	} {
		commit := typedClientCommit(request, preview, label+"-"+strings.ReplaceAll(forged.name, " ", "-"))
		forged.change(&commit.Settings.ModelLoads)
		typedAssertRefused(t, f, commit, typedSettingsLoadRefusal)
	}
}

// S1: a user of either client edits the server port or launch text on a typed
// configuration, with no declaration, with a first declaration, and with an
// existing one. Every edit goes through preview and back whole.
func TestModelSettingsTypedClientApply(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	if saved := f.snapshot(t); len(saved.Settings.ModelLoads) == 0 || saved.Settings.Serving != nil {
		t.Fatalf("fixture is not typed without a declaration: loads=%d serving=%v", len(saved.Settings.ModelLoads), saved.Settings.Serving)
	}

	t.Run("no declaration server port", func(t *testing.T) {
		before := typedSourceDigest(t, f.snapshot(t).Settings)
		port := freePort(t)
		preview, _ := typedClientEdit(t, f, "typed-none-server", "server", func(c *settings.Config) { c.ServerPort = port })
		if preview.Settings.ServerPort != port || typedSourceDigest(t, preview.Settings) == before {
			t.Fatalf("server port %d, digest moved=%t", preview.Settings.ServerPort, typedSourceDigest(t, preview.Settings) != before)
		}
	})
	t.Run("no declaration launch text", func(t *testing.T) {
		before := typedSourceDigest(t, f.snapshot(t).Settings)
		preview, state := typedClientEdit(t, f, "typed-none-launch", "launch", func(c *settings.Config) { c.LaunchText = strings.TrimSpace(c.LaunchText + " --metrics") })
		if !strings.Contains(state.Settings.LaunchText, "--metrics") || typedSourceDigest(t, preview.Settings) == before {
			t.Fatalf("launch %q, digest moved=%t", state.Settings.LaunchText, typedSourceDigest(t, preview.Settings) != before)
		}
	})
	t.Run("no declaration proxy port", func(t *testing.T) {
		before := typedSourceDigest(t, f.snapshot(t).Settings)
		port := freePort(t)
		preview, state := typedClientEdit(t, f, "typed-none-proxy", "", func(c *settings.Config) { c.ProxyPort = port })
		if state.Settings.ProxyPort != port || typedSourceDigest(t, preview.Settings) != before {
			t.Fatalf("proxy port %d, digest stayed=%t", state.Settings.ProxyPort, typedSourceDigest(t, preview.Settings) == before)
		}
	})
	t.Run("no declaration forged loads", func(t *testing.T) {
		typedAssertForgedLoadsRefused(t, f, "typed-none-forged", freePort(t))
	})
	// V42: a configuration with no declaration still has no reviewed digest to
	// require, so a baseline-loads apply needs no preview. V43 follows below.
	t.Run("no declaration baseline apply without preview", func(t *testing.T) {
		saved := f.snapshot(t)
		request := settings.Request{Engine: "llamacpp", ExpectedRevision: saved.Revision, Settings: saved.Settings.Clone(), Resolution: "server", RequestID: "typed-none-unpreviewed"}
		request.Settings.ServerPort = freePort(t)
		if response := f.rpc(t, "engine:apply-settings", request); response.Error != nil {
			t.Fatalf("baseline apply: %d %s", response.Error.Code, response.Error.Message)
		}
		f.settled(t, request.RequestID, "succeeded")
		port := freePort(t)
		f.rpcOK(t, "engine:set-port", map[string]any{"engine": "llamacpp", "port": port})
		if got := f.snapshot(t).Settings.ServerPort; got != port {
			t.Fatalf("set-port saved %d, want %d", got, port)
		}
	})

	t.Run("first declaration", func(t *testing.T) {
		declaration := typedDeclaration(t, f)
		preview, state := typedClientEdit(t, f, "typed-first-declaration", "", func(c *settings.Config) { c.Serving = declaration })
		if state.Settings.Serving == nil || len(state.Settings.Serving.Models) != 2 || state.Settings.Serving.Models[0].ExpectedNativeSourceDigest != typedSourceDigest(t, preview.Settings) {
			t.Fatalf("first declaration saved as %+v", state.Settings.Serving)
		}
	})
	// The remaining cases need a saved declaration even when the case above fails.
	if f.snapshot(t).Settings.Serving == nil {
		typedDeclareWithBaselineLoads(t, f, "typed-setup-declaration")
	}
	t.Run("declaration server port", func(t *testing.T) {
		before := f.snapshot(t).Settings.Serving.Models[0].ExpectedNativeSourceDigest
		port := freePort(t)
		_, state := typedClientEdit(t, f, "typed-declared-server", "server", func(c *settings.Config) { c.ServerPort = port })
		after := state.Settings.Serving.Models[0].ExpectedNativeSourceDigest
		if state.Settings.ServerPort != port || after == before || after != typedSourceDigest(t, state.Settings) {
			t.Fatalf("server port %d, digest %s to %s", state.Settings.ServerPort, before, after)
		}
	})
	t.Run("declaration launch text", func(t *testing.T) {
		before := f.snapshot(t).Settings.Serving.Models[0].ExpectedNativeSourceDigest
		_, state := typedClientEdit(t, f, "typed-declared-launch", "launch", func(c *settings.Config) { c.LaunchText = strings.TrimSpace(c.LaunchText + " --no-webui") })
		after := state.Settings.Serving.Models[0].ExpectedNativeSourceDigest
		if !strings.Contains(state.Settings.LaunchText, "--no-webui") || after == before || after != typedSourceDigest(t, state.Settings) {
			t.Fatalf("launch %q, digest %s to %s", state.Settings.LaunchText, before, after)
		}
	})
	t.Run("declaration proxy port", func(t *testing.T) {
		port := freePort(t)
		_, state := typedClientEdit(t, f, "typed-declared-proxy", "", func(c *settings.Config) { c.ProxyPort = port })
		if state.Settings.ProxyPort != port {
			t.Fatalf("proxy port %d, want %d", state.Settings.ProxyPort, port)
		}
	})
	t.Run("declaration forged loads", func(t *testing.T) {
		typedAssertForgedLoadsRefused(t, f, "typed-declared-forged", freePort(t))
	})
	t.Run("declaration forged and empty digest", func(t *testing.T) {
		const review = "native source review is required"
		port := freePort(t)
		request, preview := typedClientPreview(t, f, "server", func(c *settings.Config) { c.ServerPort = port })
		forged := typedClientCommit(request, preview, "typed-declared-forged-digest")
		forged.Settings.Serving.Models[0].ExpectedNativeSourceDigest = strings.Repeat("9", 64)
		typedAssertRefused(t, f, forged, review)
		empty := typedClientCommit(request, preview, "typed-declared-empty-digest")
		empty.Settings.Serving.Models[0].ExpectedNativeSourceDigest = ""
		typedAssertRefused(t, f, empty, review)
		// The saved digest is valid for the saved port, not for the new one.
		saved := f.snapshot(t)
		old := typedClientCommit(request, preview, "typed-declared-old-digest")
		old.Settings.Serving = saved.Settings.Serving.Clone()
		typedAssertRefused(t, f, old, review)
	})
	// V43: the port-only RPC carries the saved digest, so it stays refused.
	t.Run("declaration set-port stays refused", func(t *testing.T) {
		saved := f.snapshot(t)
		response := f.rpc(t, "engine:set-port", map[string]any{"engine": "llamacpp", "port": freePort(t)})
		if response.Error == nil || !strings.Contains(response.Error.Message, "native source review is required") {
			t.Fatalf("set-port on a declaration: %+v", response.Error)
		}
		if after := f.snapshot(t); after.Revision != saved.Revision || !after.Settings.Equal(saved.Settings) {
			t.Fatal("refused set-port changed saved settings")
		}
	})
	// A commit whose preview no longer matches what the target would compile is
	// neither the saved loads nor the fresh ones.
	t.Run("stale preview", func(t *testing.T) {
		port := freePort(t)
		request, preview := typedClientPreview(t, f, "server", func(c *settings.Config) { c.ServerPort = port })
		stale := typedClientCommit(request, preview, "typed-stale-preview")
		stale.Settings.LaunchText = strings.TrimSpace(stale.Settings.LaunchText + " --log-disable")
		current := stale.Settings.Clone()
		current.ModelLoads = request.Settings.ModelLoads
		var fresh settings.Preview
		modelSettingsDecode(t, f.rpcOK(t, "engine:preview-settings", settings.Request{Engine: "llamacpp", ExpectedRevision: stale.ExpectedRevision, Settings: current}), &fresh)
		for i := range stale.Settings.Serving.Models {
			stale.Settings.Serving.Models[i].ExpectedNativeSourceDigest = fresh.Settings.Serving.Models[i].ExpectedNativeSourceDigest
		}
		typedAssertRefused(t, f, stale, typedSettingsLoadRefusal)
	})
	// A model-change apply keeps its own preview digest and baseline rule.
	t.Run("model change requires baseline", func(t *testing.T) {
		port := freePort(t)
		request, preview := typedClientPreview(t, f, "server", func(c *settings.Config) { c.ServerPort = port })
		change := f.request(t, "typed-model-change-whole", nil)
		reviewed, _ := f.review(t, change)
		reviewed.Settings = typedClientCommit(request, preview, "").Settings
		reviewed.Settings.ServerPort = f.snapshot(t).Settings.ServerPort
		response := f.rpc(t, "engine:apply-settings", reviewed)
		if response.Error == nil {
			t.Fatal("model-change apply with previewed loads was accepted")
		}
		if !strings.Contains(response.Error.Message, "settings") {
			t.Fatalf("model-change refusal = %q", response.Error.Message)
		}
		applied, _ := f.review(t, f.request(t, "typed-model-change-baseline", nil))
		if response := f.rpc(t, "engine:apply-settings", applied); response.Error != nil {
			t.Fatalf("model-change apply on the baseline: %d %s", response.Error.Code, response.Error.Message)
		}
		f.settled(t, applied.RequestID, "succeeded")
	})
}

// typedDeclareWithBaselineLoads saves a declaration by sending the preview's
// reviewed digest with the target's own loads. This apply shape predates the
// client path under test, so a failure there cannot hide what follows.
func typedDeclareWithBaselineLoads(t *testing.T, f *modelBufferProcessFixture, id string) settings.Snapshot {
	t.Helper()
	declaration := typedDeclaration(t, f)
	request, preview := typedClientPreview(t, f, "", func(c *settings.Config) { c.Serving = declaration })
	commit := typedClientCommit(request, preview, id)
	commit.Settings.ModelLoads = request.Settings.ModelLoads
	if response := f.rpc(t, "engine:apply-settings", commit); response.Error != nil {
		t.Fatalf("declaration with baseline loads: code=%d message=%s", response.Error.Code, response.Error.Message)
	}
	state := f.settled(t, id, "succeeded")
	if state.Settings.Serving == nil {
		t.Fatal("declaration was not saved")
	}
	return state
}

// typedRecoverStaleDigest drives V38's recovery from the saved snapshot and
// checks everything that must stay refused on the way.
func typedRecoverStaleDigest(t *testing.T, f *modelBufferProcessFixture, staleDigest string) (settings.Preview, settings.Snapshot) {
	t.Helper()
	saved := f.snapshot(t)
	if got := saved.Settings.Serving.Models[0].ExpectedNativeSourceDigest; got != staleDigest {
		t.Fatalf("saved digest %s changed before review, want %s", got, staleDigest)
	}
	if saved.Serving == nil || saved.Serving.Status != "mismatch" {
		t.Fatalf("readback with a stale typed digest = %+v, want a mismatch so a client offers Apply", saved.Serving)
	}
	request := settings.Request{Engine: "llamacpp", ExpectedRevision: saved.Revision, Settings: saved.Settings.Clone()}
	var preview settings.Preview
	modelSettingsDecode(t, f.rpcOK(t, "engine:preview-settings", request), &preview)
	if len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("preview of the stale saved settings: errors=%v conflict=%v", preview.Errors, preview.Conflict)
	}
	fresh := preview.Settings.Serving.Models[0].ExpectedNativeSourceDigest
	if fresh == "" || fresh == staleDigest || fresh != typedSourceDigest(t, preview.Settings) {
		t.Fatalf("preview digest %q after stale %q", fresh, staleDigest)
	}

	const review = "native source review is required"
	echoed := request
	echoed.RequestID = "typed-stale-echo"
	typedAssertRefused(t, f, echoed, review)
	empty := request
	empty.Settings = request.Settings.Clone()
	empty.RequestID = "typed-stale-empty"
	empty.Settings.Serving.Models[0].ExpectedNativeSourceDigest = ""
	typedAssertRefused(t, f, empty, review)
	forged := typedClientCommit(request, preview, "typed-stale-forged")
	forged.Settings.Serving.Models[0].ExpectedNativeSourceDigest = strings.Repeat("9", 64)
	typedAssertRefused(t, f, forged, review)

	commit := typedClientCommit(request, preview, "typed-stale-reviewed")
	if response := f.rpc(t, "engine:apply-settings", commit); response.Error != nil {
		t.Fatalf("V38 recovery apply refused: code=%d message=%s", response.Error.Code, response.Error.Message)
	}
	state := f.settled(t, commit.RequestID, "succeeded")
	if !state.Settings.Equal(preview.Settings) || (state.Serving != nil && state.Serving.Status == "mismatch") {
		t.Fatalf("recovered settings: equal=%t readback=%+v", state.Settings.Equal(preview.Settings), state.Serving)
	}
	return preview, state
}

// typedLaunchIdentities lists the launch digests the manager has logged for the
// declared configuration, oldest first.
func typedLaunchIdentities(f *modelBufferProcessFixture) []string {
	identity := regexp.MustCompile(`engine identity engine=llamacpp run_id=run-[0-9a-f]+ launch_digest=([0-9a-f]{64}) models_with_digest=2`)
	var digests []string
	for _, match := range identity.FindAllStringSubmatch(f.p.output.String(), -1) {
		digests = append(digests, match[1])
	}
	return digests
}

// typedAssertLaunches requires one more launch of the declared configuration
// than the caller had seen, so an earlier launch cannot satisfy it.
func typedAssertLaunches(t *testing.T, f *modelBufferProcessFixture, seen int) {
	t.Helper()
	task1Eventually(t, "typed launch identity", 30*time.Second, func() bool {
		return len(typedLaunchIdentities(f)) > seen
	})
}

// typedCycleAvailability sets the node's availability and, on a pause, waits for
// the engine to stop.
func typedCycleAvailability(t *testing.T, f *modelBufferProcessFixture, state string) {
	t.Helper()
	f.rpcOK(t, "node:set-availability", map[string]string{"state": state})
	if state == "paused" {
		task1Eventually(t, "engine stopped by pause", 30*time.Second, func() bool {
			return !f.snapshot(t).Running
		})
	}
}

// S2: the native build changes after approval, as a rebuilt llama-server does.
// The stub library the fixture owns is part of the build fingerprint, so the
// saved digest and NativeBuildID both go stale without any settings change.
func TestModelSettingsTypedStaleDigestAfterBuildChange(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	original, err := os.ReadFile(f.library)
	if err != nil {
		t.Fatal(err)
	}
	state := typedDeclareWithBaselineLoads(t, f, "typed-stale-declare")
	staleDigest := state.Settings.Serving.Models[0].ExpectedNativeSourceDigest
	staleBuild := state.Settings.ModelLoads[0].Resolved.NativeBuildID

	t.Cleanup(func() {
		if err := os.WriteFile(f.library, original, 0o600); err != nil {
			t.Errorf("restore fixture stub: %v", err)
		}
	})
	if err := os.WriteFile(f.library, append(bytes.Clone(original), []byte("Synthetic rebuild.\n")...), 0o600); err != nil {
		t.Fatal(err)
	}
	var settled settings.Snapshot
	task1Eventually(t, "stale typed digest readback", 30*time.Second, func() bool {
		settled = f.snapshot(t)
		return settled.Serving != nil && settled.Serving.Status == "mismatch"
	})

	preview, recovered := typedRecoverStaleDigest(t, f, staleDigest)
	if got := preview.Settings.ModelLoads[0].Resolved.NativeBuildID; got == staleBuild {
		t.Fatalf("native build %s did not move with the library", got)
	}
	if recovered.Settings.ModelLoads[0].Resolved.NativeBuildID == staleBuild {
		t.Fatal("recovered settings kept the old native build")
	}
	// The node stayed paused throughout, so only resuming starts the engine.
	typedAssertStaysStopped(t, f, "re-approved on a paused node")
	f.resume(t)
	typedAssertLaunches(t, f, 0)
}

// S2, digest only, in the order an upgrade happens: an approved engine ran, was
// stopped, and a replacement manager finds a changed digest input. Only the
// digests move in the loads, as with a build's compiled defaults. Restore must
// refuse the old approval, and the reviewed one must launch.
func TestModelSettingsTypedStaleDigestAfterSourceChange(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	state := typedDeclareWithBaselineLoads(t, f, "typed-source-declare")
	staleDigest := state.Settings.Serving.Models[0].ExpectedNativeSourceDigest
	f.resume(t)
	typedAssertLaunches(t, f, 0)
	typedCycleAvailability(t, f, "paused")

	path := filepath.Join(jointAppDir(f.base), "engines", "llamacpp.json")
	var manifest map[string]any
	modelSettingsDecode(t, readLiveDurableFile(t, path), &manifest)
	platforms, ok := manifest["platforms"].(map[string]any)
	if !ok {
		t.Fatal("missing fixture platforms")
	}
	platform, ok := platforms[runtime.GOOS+"/"+runtime.GOARCH].(map[string]any)
	if !ok {
		t.Fatal("missing fixture platform")
	}
	for _, owner := range []map[string]any{manifest, platform} {
		rt, ok := owner["runtime"].(map[string]any)
		if !ok {
			t.Fatal("missing fixture runtime")
		}
		env, ok := rt["env"].(map[string]any)
		if !ok {
			t.Fatal("missing fixture environment")
		}
		env["NVPAIR_TEST_TYPED_SOURCE_CHANGE"] = "1"
	}
	jointWriteJSON(t, path, manifest)
	f.replaceManager(t)
	task1Eventually(t, "replacement manager", 60*time.Second, func() bool {
		return f.rpc(t, "engine:get-settings", map[string]string{"engine": "llamacpp"}).Error == nil
	})
	launches := len(typedLaunchIdentities(f))
	typedCycleAvailability(t, f, "available")
	const refused = "Declaration requires a fresh typed preview"
	task1Eventually(t, "restore refusal", 30*time.Second, func() bool {
		return strings.Contains(f.p.output.String(), refused)
	})
	if saved := f.snapshot(t); saved.Running {
		t.Fatal("the replacement manager started the engine on the old approval")
	}
	if got := len(typedLaunchIdentities(f)); got != launches {
		t.Fatalf("the replacement manager launched a stale approval: %d launches, had %d", got, launches)
	}

	preview, _ := typedRecoverStaleDigest(t, f, staleDigest)
	before, after := state.Settings.ModelLoads[0].Resolved, preview.Settings.ModelLoads[0].Resolved
	if before.NativeBuildID != after.NativeBuildID || before.PerSlotContext != after.PerSlotContext {
		t.Fatalf("source-only change moved more than the digests: build %s to %s", before.NativeBuildID, after.NativeBuildID)
	}
	// V44: the restore refused this approval, so replacing it starts the engine.
	typedAssertLaunches(t, f, launches)
	identities := typedLaunchIdentities(f)
	if identities[0] == identities[len(identities)-1] {
		t.Fatal("the reviewed launch has the digest of the stale one")
	}
}
