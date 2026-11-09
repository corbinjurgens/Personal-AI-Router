// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// staysStoppedWindow is well past a fixture engine's start time, which the
// positive cases below show is a few seconds.
const staysStoppedWindow = 8 * time.Second

const (
	restoreRefusedLine  = "could not be restored"
	restoreOccupiedLine = "the port is occupied by a service"
)

// typedRebuild changes the native build under the running fixture, as an
// upgrade does: the stub library is part of the build fingerprint.
func typedRebuild(t *testing.T, f *modelBufferProcessFixture) {
	t.Helper()
	original, err := os.ReadFile(f.library)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		stub := []byte("Synthetic metadata binding only, never a loadable library.\n")
		if err := os.WriteFile(f.library, stub, 0o600); err != nil {
			t.Errorf("restore fixture stub: %v", err)
		}
	})
	if err := os.WriteFile(f.library, append(bytes.Clone(original), []byte("Synthetic rebuild.\n")...), 0o600); err != nil {
		t.Fatal(err)
	}
}

// foreignListener holds a loopback port in a process of another image, so restore
// refuses the occupied port. A listener in this test process would be reclaimed:
// the fixture's engine runs this same executable. The returned func releases it.
func foreignListener(t *testing.T, port int) func() {
	t.Helper()
	shell, err := exec.LookPath("powershell")
	if err != nil {
		t.Skip("no foreign listener process available: ", err)
	}
	script := fmt.Sprintf("$l=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,%d);$l.Start();[Console]::Out.WriteLine('ready');[Console]::In.ReadLine()", port)
	command := exec.Command(shell, "-NoProfile", "-NonInteractive", "-Command", script)
	stdin, err := command.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := command.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	release := func() {
		_ = stdin.Close()
		_ = command.Wait()
	}
	t.Cleanup(release)
	line, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil || !strings.Contains(line, "ready") {
		t.Fatalf("foreign listener on port %d: %q %v", port, line, err)
	}
	return release
}

func typedPause(t *testing.T, f *modelBufferProcessFixture) {
	t.Helper()
	// A pause directly after a start can be refused while its identity settles.
	task1Eventually(t, "pause accepted", 30*time.Second, func() bool {
		return f.rpc(t, "node:set-availability", map[string]string{"state": "paused"}).Error == nil
	})
	task1Eventually(t, "engine stopped by pause", 30*time.Second, func() bool {
		return !f.snapshot(t).Running
	})
}

// typedRestoreRefused takes a running approved engine through a pause, a rebuild
// and a resume, and returns once the restore has refused the old approval.
func typedRestoreRefused(t *testing.T, f *modelBufferProcessFixture) settings.Snapshot {
	t.Helper()
	return typedRestoreRefusedAfter(t, f, func() { typedRebuild(t, f) })
}

// typedRestoreRefusedAfter is typedRestoreRefused with the paused change given.
func typedRestoreRefusedAfter(t *testing.T, f *modelBufferProcessFixture, change func()) settings.Snapshot {
	t.Helper()
	if !f.snapshot(t).Running {
		f.resume(t)
	}
	typedPause(t, f)
	change()
	refusals := strings.Count(f.p.output.String(), restoreRefusedLine)
	f.modelSettingsProcessFixture.rpcOK(t, "node:set-availability", map[string]string{"state": "available"})
	task1Eventually(t, "restore refused the stale approval", 30*time.Second, func() bool {
		return strings.Count(f.p.output.String(), restoreRefusedLine) > refusals
	})
	var stale settings.Snapshot
	task1Eventually(t, "stale approval readback", 30*time.Second, func() bool {
		stale = f.snapshot(t)
		return stale.Serving != nil && stale.Serving.Status == "mismatch"
	})
	if stale.Running {
		t.Fatal("the engine started on a stale approval")
	}
	return stale
}

func typedAssertStaysStopped(t *testing.T, f *modelBufferProcessFixture, label string) {
	t.Helper()
	for deadline := time.Now().Add(staysStoppedWindow); time.Now().Before(deadline); time.Sleep(500 * time.Millisecond) {
		if f.snapshot(t).Running {
			t.Fatalf("%s: the engine started", label)
		}
	}
}

func typedAssertRunningSoon(t *testing.T, f *modelBufferProcessFixture, label string) {
	t.Helper()
	task1Eventually(t, label, 30*time.Second, func() bool { return f.snapshot(t).Running })
}

func typedAssertNoMismatch(t *testing.T, f *modelBufferProcessFixture, label string) {
	t.Helper()
	// Status refresh and the snapshot are separate observations.
	time.Sleep(2 * time.Second)
	if state := f.snapshot(t); state.Serving != nil && state.Serving.Status == "mismatch" {
		t.Errorf("%s: false mismatch: %s", label, state.Serving.Reason)
	}
}

func typedNoEdit(*settings.Config) {}

// D1: typed loads carry their own digests, so a configuration with no declaration
// loses its approval on a rebuild too. Its readback must say so even though the
// engine is stopped, and a no-edit apply recovers it.
func TestModelSettingsUndeclaredStaleApprovalRecovery(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	if saved := f.snapshot(t); len(saved.Settings.ModelLoads) == 0 || saved.Settings.Serving != nil {
		t.Fatalf("fixture is not typed without a declaration: loads=%d serving=%v", len(saved.Settings.ModelLoads), saved.Settings.Serving)
	}
	stale := typedRestoreRefused(t, f)
	if stale.Phase != "succeeded" || !strings.Contains(stale.Serving.Reason, "fresh preview") {
		t.Fatalf("stale readback: phase=%s reason=%q", stale.Phase, stale.Serving.Reason)
	}
	if !strings.Contains(f.p.output.String(), "Typed inputs changed") {
		t.Fatal("the refusal was not the saved loads' comparison")
	}

	preview, recovered := typedClientEdit(t, f, "undeclared-reapprove", "", typedNoEdit)
	if typedSourceDigest(t, preview.Settings) == typedSourceDigest(t, stale.Settings) {
		t.Fatal("the no-edit preview kept the stale digest")
	}
	if recovered.Serving != nil && recovered.Serving.Status == "mismatch" {
		t.Fatalf("readback after the reviewed apply: %+v", recovered.Serving)
	}
	if recovered.Settings.Serving != nil {
		t.Fatal("a no-edit apply created a declaration")
	}
	typedAssertRunningSoon(t, f, "engine started by the reviewed apply")
	typedAssertNoMismatch(t, f, "running after the reviewed apply")
}

// The readback now compares saved loads, so nothing ordinary may move them. This
// is the state a configuration with no declaration passes through in normal use.
func TestModelSettingsUndeclaredReadbackStaysClean(t *testing.T) {
	f := newModelBufferProcessFixture(t, func(f *modelSettingsProcessFixture) {
		f.enableMemory(t, modelMemoryPool("ram", "system", "17179869184", "8589934592"))
	})
	if saved := f.snapshot(t); len(saved.Settings.ModelLoads) == 0 || saved.Settings.Serving != nil {
		t.Fatalf("fixture is not typed without a declaration: loads=%d serving=%v", len(saved.Settings.ModelLoads), saved.Settings.Serving)
	}
	typedAssertNoMismatch(t, f, "stopped")
	f.waitMemory(t, "17179869184")
	f.mu.Lock()
	f.memory.observation = &noderec.MemoryObservation{RunID: "private-memory-run-2", Domains: []noderec.MemoryDomain{modelMemoryPool("ram", "system", "34359738368", "33554432")}}
	f.mu.Unlock()
	f.waitMemory(t, "34359738368")
	typedAssertNoMismatch(t, f, "memory moved")

	f.resume(t)
	typedAssertNoMismatch(t, f, "running")
	digest := typedSourceDigest(t, f.snapshot(t).Settings)
	for _, op := range []string{"initial", "sleep", "wakeup", "exit"} {
		f.control(t, "medium-model", op)
		typedAssertNoMismatch(t, f, "child model "+op)
	}
	if err := os.WriteFile(filepath.Join(f.root, "unrelated.gguf"), []byte("unrelated synthetic"), 0o600); err != nil {
		t.Fatal(err)
	}
	typedAssertNoMismatch(t, f, "unrelated file in the store")
	policy := f.policy(t)
	policy.Idle.UnloadAfterMinutes = 9
	f.setPolicy(t, policy)
	typedAssertNoMismatch(t, f, "policy edit")
	typedClientEdit(t, f, "undeclared-clean-proxy", "", func(c *settings.Config) { c.ProxyPort = freePort(t) })
	typedAssertNoMismatch(t, f, "proxy port edit")
	if got := typedSourceDigest(t, f.snapshot(t).Settings); got != digest {
		t.Fatalf("ordinary changes moved the digest: %s to %s", digest, got)
	}
	limit := 4096
	typedClientEdit(t, f, "undeclared-clean-limit", "", func(c *settings.Config) { c.WorkingSetLimitMiB = &limit })
	typedAssertRunningSoon(t, f, "running with a working-set limit")
	typedAssertNoMismatch(t, f, "working-set limit set")
	typedClientEdit(t, f, "undeclared-clean-limit-clear", "", func(c *settings.Config) { c.WorkingSetLimitMiB = nil })
	typedAssertRunningSoon(t, f, "running with no working-set limit")
	typedAssertNoMismatch(t, f, "working-set limit cleared")

	before := f.snapshot(t)
	typedClientEdit(t, f, "undeclared-clean-noop", "", typedNoEdit)
	if after := f.snapshot(t); after.Revision != before.Revision {
		t.Errorf("no-op apply moved the revision: %d to %d", before.Revision, after.Revision)
	}
	typedAssertNoMismatch(t, f, "no-op apply")

	typedPause(t, f)
	typedAssertNoMismatch(t, f, "paused")
	base := f.snapshot(t).Settings.LaunchText
	for i, shape := range []struct {
		name string
		text string
	}{
		{"added flag", base + " --metrics"},
		{"environment assignment", "NVPAIR_REVIEW_ENV=one " + base},
		{"repeated flag", base + " --metrics --metrics"},
		{"empty text", ""},
		{"whitespace", "   "},
	} {
		t.Run("launch text "+shape.name, func(t *testing.T) {
			_, state := typedClientEdit(t, f, fmt.Sprintf("undeclared-clean-shape-%d", i), "launch", func(c *settings.Config) { c.LaunchText = strings.TrimSpace(shape.text) })
			typedAssertNoMismatch(t, f, "saved")
			var again settings.Preview
			modelSettingsDecode(t, f.rpcOK(t, "engine:preview-settings", settings.Request{Engine: "llamacpp", ExpectedRevision: state.Revision, Settings: state.Settings.Clone()}), &again)
			if !again.Settings.Equal(state.Settings) {
				t.Error("a second compile moved the saved settings")
			}
			f.resume(t)
			typedAssertNoMismatch(t, f, "launched")
			typedPause(t, f)
		})
	}
}

// V44, typed: a re-approval starts an engine whose restore was refused for its
// stale approval, with no pause and resume. An existing test covers a declaration.
func TestModelSettingsReapprovalHeldByPause(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	typedRestoreRefused(t, f)
	typedPause(t, f)
	typedClientEdit(t, f, "reapprove-paused", "", typedNoEdit)
	typedAssertStaysStopped(t, f, "re-approved on a paused node")
	f.resume(t)
}

// C2: the record outlives the staleness. Inputs back at the approved state leave
// nothing for an apply to replace, so an unrelated edit does not start the engine;
// a later staleness on the same stopped engine is replaced and does.
func TestModelSettingsRecoveredInputsThenUnrelatedEdit(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	original, err := os.ReadFile(f.library)
	if err != nil {
		t.Fatal(err)
	}
	typedRestoreRefused(t, f)
	if err := os.WriteFile(f.library, original, 0o600); err != nil {
		t.Fatal(err)
	}
	var recovered settings.Snapshot
	task1Eventually(t, "staleness gone with the inputs", 30*time.Second, func() bool {
		recovered = f.snapshot(t)
		return recovered.Serving != nil && recovered.Serving.Status != "mismatch"
	})
	_, edited := typedClientEdit(t, f, "recovered-unrelated", "", func(c *settings.Config) { c.ProxyPort = freePort(t) })
	if typedSourceDigest(t, edited.Settings) != typedSourceDigest(t, recovered.Settings) {
		t.Fatal("the unrelated edit moved the digest")
	}
	typedAssertStaysStopped(t, f, "unrelated edit after the inputs recovered")

	typedRebuild(t, f)
	task1Eventually(t, "stale again", 30*time.Second, func() bool {
		state := f.snapshot(t)
		return state.Serving != nil && state.Serving.Status == "mismatch"
	})
	typedClientEdit(t, f, "recovered-reapprove", "", typedNoEdit)
	typedAssertRunningSoon(t, f, "engine started by replacing the later stale approval")
}

// typedKillEngine kills the fixture engine as a crash would. Windows only, as is
// the fixture that reaches it.
func typedKillEngine(t *testing.T, f *modelBufferProcessFixture) {
	t.Helper()
	read := func(script string) string {
		out, err := exec.Command("powershell", "-NoProfile", "-NonInteractive", "-Command", script).Output()
		if err != nil {
			t.Fatal(err)
		}
		return strings.TrimSpace(string(out))
	}
	pid, err := strconv.Atoi(read(fmt.Sprintf("(Get-NetTCPConnection -LocalPort %d -State Listen | Select-Object -First 1).OwningProcess", f.targetPort)))
	if err != nil {
		t.Fatal(err)
	}
	if image := read(fmt.Sprintf("(Get-Process -Id %d).Path", pid)); pid == os.Getpid() || !strings.EqualFold(image, f.image) {
		t.Fatalf("listener %d is not the fixture engine: %s", pid, image)
	}
	process, err := os.FindProcess(pid)
	if err != nil {
		t.Fatal(err)
	}
	if err := process.Kill(); err != nil {
		t.Fatal(err)
	}
}

// V44 counts a crash relaunch: it goes through the same restore as a startup.
func TestModelSettingsCrashRelaunchRefusalCounts(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	f.resume(t)
	typedRebuild(t, f)
	task1Eventually(t, "stale readback while running", 30*time.Second, func() bool {
		state := f.snapshot(t)
		return state.Serving != nil && state.Serving.Status == "mismatch"
	})
	typedKillEngine(t, f)
	task1Eventually(t, "relaunch refused the stale approval", 30*time.Second, func() bool {
		return strings.Contains(f.p.output.String(), "unexpected-exit restore failed")
	})
	// The relaunch tries again 30 s after this refusal and would start a re-approved
	// engine itself, so only a start before then shows that the apply did it.
	refused := time.Now()
	typedClientEdit(t, f, "crash-reapprove", "", typedNoEdit)
	task1Eventually(t, "engine started by the reviewed apply, before the next relaunch attempt", time.Until(refused.Add(25*time.Second)), func() bool {
		return f.snapshot(t).Running
	})
}

// V44, typed, negatives: an engine stopped for any reason other than a refused
// restore keeps today's behaviour, and a refused wake does not count.
func TestModelSettingsApplyDoesNotStartOtherStoppedEngine(t *testing.T) {
	// A pinned client's terminal request is the one that asks for a wake.
	f := newModelBufferProcessFixture(t, func(f *modelSettingsProcessFixture) { f.pinned(t) })
	f.serveModelSettingsInference(t)
	f.resume(t)
	typedPause(t, f)

	// The restore fails for a reason an apply does not clear.
	release := foreignListener(t, f.targetPort)
	f.rpcOK(t, "node:set-availability", map[string]string{"state": "available"})
	task1Eventually(t, "restore failed on the occupied port", 30*time.Second, func() bool {
		return strings.Contains(f.p.output.String(), restoreOccupiedLine)
	})
	release()
	if f.snapshot(t).Running {
		t.Fatal("the engine started on an occupied port")
	}
	typedClientEdit(t, f, "other-stop-launch", "launch", func(c *settings.Config) { c.LaunchText = strings.TrimSpace(c.LaunchText + " --metrics") })
	typedAssertStaysStopped(t, f, "ordinary apply after a failed restore")

	// The approval goes stale while the engine is stopped. A request's wake is
	// refused, which is not a refused restore, and the next request retries it.
	typedRebuild(t, f)
	task1Eventually(t, "stale readback on the stopped engine", 30*time.Second, func() bool {
		state := f.snapshot(t)
		return state.Serving != nil && state.Serving.Status == "mismatch"
	})
	request := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("https://127.0.0.1:%d/v1/chat/completions", f.targetFacadePort), strings.NewReader(`{"model":"medium-model","messages":[{"role":"user","content":"synthetic"}],"max_tokens":1}`))
		if err != nil {
			return
		}
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set(nodepolicy.AdmissionWaitHeader, "6")
		if response, err := f.client.Do(request); err == nil {
			_ = response.Body.Close()
		}
	}
	go request()
	const wakeRefused = "on-demand engine start failed"
	task1Eventually(t, "wake refused the stale approval", 60*time.Second, func() bool {
		return strings.Contains(f.p.output.String(), wakeRefused)
	})
	typedClientEdit(t, f, "other-stop-reapprove", "", typedNoEdit)
	typedAssertStaysStopped(t, f, "re-approval after a refused wake")
	go request()
	typedAssertRunningSoon(t, f, "the next request wakes the re-approved engine")
}

// neApprovedFixture approves a finite configuration and returns the approved
// source text. The node is available and the engine stopped: this fixture's first
// restore has nothing to serve, and an apply does not repair that.
func neApprovedFixture(t *testing.T) (*neServingFixture, string) {
	t.Helper()
	if runtime.GOOS != "windows" {
		t.Skip("NE serving envelope process acceptance is reserved for Windows PC3")
	}
	f := newNEServingFixture(t)
	modelBytes := []byte("NE synthetic model; no model runtime or hardware qualification")
	modelPath := filepath.Join(f.root, "model.gguf")
	if err := os.WriteFile(modelPath, modelBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	source := "[*]\nfit = off\nload-on-startup = false\n\n[" + neModel + "]\nmodel = " + filepath.ToSlash(modelPath) + "\nctx-size = 2048\nparallel = 1\nkv-unified = false\nsleep-idle-seconds = -1\n"
	if err := os.WriteFile(f.source, []byte(source), 0o600); err != nil {
		t.Fatal(err)
	}
	f.start(t)
	before := f.snapshot(t)
	ram, vram := int64(neMemory), int64(0)
	sum := sha256.Sum256(modelBytes)
	request := settings.Request{Engine: "llamacpp", ExpectedRevision: before.Revision, RequestID: "ne-approve", Settings: settings.Config{
		ServerPort: f.enginePort, ProxyPort: f.facadePort,
		LaunchText: neRuntimeArg + " --ne-serving-config " + strconv.Quote(filepath.Join(f.root, "runtime.json")) + " --models-preset " + strconv.Quote(f.source) + " --models-max 1",
		Serving:    &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{{ModelID: neModel, MaxOutputTokens: neOutput, Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram}, Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: hex.EncodeToString(sum[:]), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}}}}},
	}}
	neReviewedApply(t, f, request)
	return f, source
}

func neReviewedApply(t *testing.T, f *neServingFixture, request settings.Request) {
	t.Helper()
	var preview settings.Preview
	f.call(t, "engine:preview-settings", request, &preview)
	if len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("preview rejected: errors=%v conflict=%v", preview.Errors, preview.Conflict)
	}
	request.Settings = preview.Settings.Clone()
	f.call(t, "engine:apply-settings", request, nil)
	task1Eventually(t, "reviewed apply settled", 30*time.Second, func() bool {
		state := f.snapshot(t)
		return state.RequestID == request.RequestID && state.Phase == "succeeded"
	})
}

func neWaitRunning(t *testing.T, f *neServingFixture, want bool, label string) {
	t.Helper()
	task1Eventually(t, label, 30*time.Second, func() bool { return f.snapshot(t).Running == want })
}

func neAvailability(t *testing.T, f *neServingFixture, state string) {
	t.Helper()
	// The fixture's custom sibling can report an unverified stop; llamacpp's own
	// state is what the callers below assert.
	if response := callRPC(t, f.p.in, f.p.msgs, "node:set-availability", map[string]string{"state": state}, neCallBudget); response.Error != nil {
		t.Logf("availability %s: %d %s", state, response.Error.Code, response.Error.Message)
	}
}

// neRestoreRefused stops the approved engine, makes its approval stale while the
// node is paused and returns once the restore has refused it.
func neRestoreRefused(t *testing.T, f *neServingFixture, source string) {
	t.Helper()
	neAvailability(t, f, "paused")
	neAvailability(t, f, "available")
	neWaitRunning(t, f, true, "approved engine running")
	neAvailability(t, f, "paused")
	neWaitRunning(t, f, false, "stopped by pause")
	if err := os.WriteFile(f.source, []byte(source+"# changed after approval\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	refusals := strings.Count(f.p.output.String(), restoreRefusedLine)
	neAvailability(t, f, "available")
	task1Eventually(t, "restore refused the stale approval", 30*time.Second, func() bool {
		return strings.Count(f.p.output.String(), restoreRefusedLine) > refusals
	})
	task1Eventually(t, "stale approval readback", 30*time.Second, func() bool {
		state := f.snapshot(t)
		return state.Serving != nil && state.Serving.Status == "mismatch"
	})
	if f.snapshot(t).Running {
		t.Fatal("the engine started on a stale approval")
	}
}

func neAssertStaysStopped(t *testing.T, f *neServingFixture, label string) {
	t.Helper()
	for deadline := time.Now().Add(staysStoppedWindow); time.Now().Before(deadline); time.Sleep(500 * time.Millisecond) {
		if f.snapshot(t).Running {
			t.Fatalf("%s: the engine started", label)
		}
	}
}

func neNoEditRequest(t *testing.T, f *neServingFixture, id string) settings.Request {
	t.Helper()
	saved := f.snapshot(t)
	return settings.Request{Engine: "llamacpp", ExpectedRevision: saved.Revision, RequestID: id, Settings: saved.Settings.Clone()}
}

// V44, finite.
func TestNEServingReapprovalStartsRefusedRestore(t *testing.T) {
	f, source := neApprovedFixture(t)
	neRestoreRefused(t, f, source)
	neReviewedApply(t, f, neNoEditRequest(t, f, "ne-reapprove"))
	neWaitRunning(t, f, true, "engine started by the reviewed apply")
	if state := f.snapshot(t); state.Serving != nil && state.Serving.Status == "mismatch" {
		t.Fatalf("readback after the reviewed apply: %+v", state.Serving)
	}
}

func TestNEServingReapprovalHeldByPause(t *testing.T) {
	f, source := neApprovedFixture(t)
	neRestoreRefused(t, f, source)
	neAvailability(t, f, "paused")
	neReviewedApply(t, f, neNoEditRequest(t, f, "ne-reapprove-paused"))
	neAssertStaysStopped(t, f, "re-approved on a paused node")
	neAvailability(t, f, "available")
	neWaitRunning(t, f, true, "resume starts the re-approved engine")
}

func TestNEServingApplyDoesNotStartOtherStoppedEngine(t *testing.T) {
	f, _ := neApprovedFixture(t)
	neAvailability(t, f, "paused")
	release := foreignListener(t, f.enginePort)
	neAvailability(t, f, "available")
	task1Eventually(t, "restore failed on the occupied port", 30*time.Second, func() bool {
		return strings.Contains(f.p.output.String(), restoreOccupiedLine)
	})
	release()
	if f.snapshot(t).Running {
		t.Fatal("the engine started on an occupied port")
	}
	request := neNoEditRequest(t, f, "ne-other-stop")
	request.Settings.ProxyPort = freePort(t)
	neReviewedApply(t, f, request)
	neAssertStaysStopped(t, f, "ordinary apply after a failed restore")
}
