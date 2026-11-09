// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

const movedFileReason = "has a new modified time"

// typedFirstArtifact is the file a saved load binds. Every load of the fixture
// that shares it is bound to the same evidence.
func typedFirstArtifact(t *testing.T, config settings.Config) modelconfig.ArtifactBinding {
	t.Helper()
	if len(config.ModelLoads) == 0 || config.ModelLoads[0].Resolved == nil || len(config.ModelLoads[0].Resolved.Artifacts) == 0 {
		t.Fatalf("configuration has no artifact binding: %+v", config.ModelLoads)
	}
	return config.ModelLoads[0].Resolved.Artifacts[0]
}

func typedArtifactBytes(t *testing.T, artifact modelconfig.ArtifactBinding) []byte {
	t.Helper()
	data, err := os.ReadFile(artifact.Path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func typedSetArtifactTime(t *testing.T, artifact modelconfig.ArtifactBinding, when time.Time) {
	t.Helper()
	if err := os.Chtimes(artifact.Path, when, when); err != nil {
		t.Fatal(err)
	}
}

func typedArtifactOriginalTime(t *testing.T, artifact modelconfig.ArtifactBinding) time.Time {
	t.Helper()
	ns, err := strconv.ParseInt(artifact.MtimeNS, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	return time.Unix(0, ns)
}

// typedAssertPreviewNames expects a client preview refused with a reason naming
// the file, with no change saved.
func typedAssertPreviewNames(t *testing.T, f *modelBufferProcessFixture, name, want string, edit func(*settings.Request)) {
	t.Helper()
	saved := f.snapshot(t)
	request := settings.Request{Engine: "llamacpp", ExpectedRevision: saved.Revision, Settings: saved.Settings.Clone()}
	edit(&request)
	var preview settings.Preview
	modelSettingsDecode(t, f.rpcOK(t, "engine:preview-settings", request), &preview)
	joined := ""
	for _, reason := range preview.Errors {
		joined += reason
	}
	if !strings.Contains(joined, name) || !strings.Contains(joined, want) {
		t.Fatalf("preview errors %v do not name %q with %q", preview.Errors, name, want)
	}
	if after := f.snapshot(t); after.Revision != saved.Revision {
		t.Fatalf("a refused preview moved the revision: %d to %d", saved.Revision, after.Revision)
	}
}

// Same content, new modified time: a restore or a copy. The time is the only
// signal for a change between the samples, so the saved approval stays refused,
// the readback names the file, and only an explicit no-edit Apply accepts it.
func TestModelSettingsMovedFileTimeRecovery(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	saved := f.snapshot(t)
	if len(saved.Settings.ModelLoads) == 0 || saved.Settings.Serving != nil {
		t.Fatalf("fixture is not typed without a declaration: %+v", saved.Settings)
	}
	artifact := typedFirstArtifact(t, saved.Settings)
	name := filepath.Base(artifact.Path)
	original := typedArtifactBytes(t, artifact)
	digest := typedSourceDigest(t, saved.Settings)

	stale := typedRestoreRefusedAfter(t, f, func() {
		typedSetArtifactTime(t, artifact, typedArtifactOriginalTime(t, artifact).Add(time.Hour))
	})
	if stale.Phase != "succeeded" || !strings.Contains(stale.Serving.Reason, name) || !strings.Contains(stale.Serving.Reason, movedFileReason) {
		t.Fatalf("stale readback: phase=%s reason=%q", stale.Phase, stale.Serving.Reason)
	}
	if output := f.p.output.String(); !strings.Contains(output, name+" "+movedFileReason) {
		t.Fatal("the restore refusal does not name the file")
	}
	typedAssertStaysStopped(t, f, "moved file with the saved approval")

	t.Run("an edit does not carry the acceptance along", func(t *testing.T) {
		typedAssertPreviewNames(t, f, name, movedFileReason, func(r *settings.Request) { r.Settings.LaunchText += " --metrics" })
		typedAssertPreviewNames(t, f, name, movedFileReason, func(r *settings.Request) { r.Settings.ServerPort, r.Resolution = freePort(t), "server" })
		typedAssertPreviewNames(t, f, name, movedFileReason, func(r *settings.Request) { r.Settings.ProxyPort = freePort(t) })
	})
	t.Run("a finite model change names the file", func(t *testing.T) {
		request := f.request(t, "", nil)
		request.ExpectedRevision = f.snapshot(t).Revision
		var preview settings.Preview
		modelSettingsDecode(t, f.rpcOK(t, "engine:preview-settings", request), &preview)
		joined := ""
		for _, reason := range preview.Errors {
			joined += reason
		}
		if !strings.Contains(joined, name) || !strings.Contains(joined, movedFileReason) {
			t.Fatalf("model change errors %v", preview.Errors)
		}
	})
	t.Run("an apply nobody previewed is refused", func(t *testing.T) {
		current := f.snapshot(t)
		response := f.rpc(t, "engine:apply-settings", settings.Request{Engine: "llamacpp", ExpectedRevision: current.Revision, Settings: current.Settings.Clone(), RequestID: "moved-no-preview"})
		if response.Error == nil || !strings.Contains(response.Error.Message, "review is required") {
			t.Fatalf("no-preview apply: %+v", response.Error)
		}
		if after := f.snapshot(t); after.Revision != current.Revision {
			t.Fatalf("a refused apply moved the revision: %d to %d", current.Revision, after.Revision)
		}
	})

	preview, recovered := typedClientEdit(t, f, "moved-accept", "", typedNoEdit)
	moved := typedFirstArtifact(t, preview.Settings)
	if moved.MtimeNS == artifact.MtimeNS || moved.Path != artifact.Path || moved.ResolvedTarget != artifact.ResolvedTarget || moved.Size != artifact.Size || moved.SampleSHA256 != artifact.SampleSHA256 {
		t.Fatalf("the accepted binding changed more than the time: %+v from %+v", moved, artifact)
	}
	if typedSourceDigest(t, preview.Settings) == digest {
		t.Fatal("the accepted preview kept the old digest")
	}
	if recovered.Serving != nil && recovered.Serving.Status == "mismatch" {
		t.Fatalf("readback after the reviewed apply: %+v", recovered.Serving)
	}
	typedAssertRunningSoon(t, f, "engine started by the reviewed apply")
	typedAssertNoMismatch(t, f, "running after the reviewed apply")
	if !bytes.Equal(typedArtifactBytes(t, artifact), original) {
		t.Fatal("recovery touched the model file")
	}

	review := f.request(t, "moved-finite-change", nil)
	review.ExpectedRevision = f.snapshot(t).Revision
	f.review(t, review)
}

// Different bytes with a new time are not the saved file: nothing an apply can
// accept. Restoring the original bytes and time recovers.
func TestModelSettingsChangedFileStaysRefused(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	artifact := typedFirstArtifact(t, f.snapshot(t).Settings)
	name := filepath.Base(artifact.Path)
	original := typedArtifactBytes(t, artifact)
	originalTime := typedArtifactOriginalTime(t, artifact)

	stale := typedRestoreRefusedAfter(t, f, func() {
		changed := bytes.Clone(original)
		changed[len(changed)-1] ^= 0xff
		if err := os.WriteFile(artifact.Path, changed, 0o600); err != nil {
			t.Fatal(err)
		}
		typedSetArtifactTime(t, artifact, originalTime.Add(time.Hour))
	})
	if !strings.Contains(stale.Serving.Reason, name) || strings.Contains(stale.Serving.Reason, movedFileReason) {
		t.Fatalf("changed file readback: %q", stale.Serving.Reason)
	}
	typedAssertPreviewNames(t, f, name, "Restore the original file", func(*settings.Request) {})
	current := f.snapshot(t)
	response := f.rpc(t, "engine:apply-settings", settings.Request{Engine: "llamacpp", ExpectedRevision: current.Revision, Settings: current.Settings.Clone(), RequestID: "changed-refused"})
	if response.Error == nil {
		t.Fatal("an apply accepted changed content")
	}
	if after := f.snapshot(t); after.Revision != current.Revision {
		t.Fatalf("a refused apply moved the revision: %d to %d", current.Revision, after.Revision)
	}
	typedAssertStaysStopped(t, f, "changed file")

	if err := os.WriteFile(artifact.Path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	typedSetArtifactTime(t, artifact, originalTime)
	task1Eventually(t, "readback clears once the file is restored", 30*time.Second, func() bool {
		state := f.snapshot(t)
		return state.Serving == nil || state.Serving.Status != "mismatch"
	})
}

// A declaration carries its own copy of the binding and a review digest, so an
// acceptance has to move both, and a commit without the digest stays refused.
func TestModelSettingsDeclaredMovedFileTimeRecovery(t *testing.T) {
	f := newModelBufferProcessFixture(t)
	state := typedDeclareWithBaselineLoads(t, f, "moved-declare")
	artifact := typedFirstArtifact(t, state.Settings)
	staleDigest := state.Settings.Serving.Models[0].ExpectedNativeSourceDigest

	stale := typedRestoreRefusedAfter(t, f, func() {
		typedSetArtifactTime(t, artifact, typedArtifactOriginalTime(t, artifact).Add(time.Hour))
	})
	if !strings.Contains(stale.Serving.Reason, filepath.Base(artifact.Path)) {
		t.Fatalf("declared readback: %q", stale.Serving.Reason)
	}
	request, preview := typedClientPreview(t, f, "", typedNoEdit)
	forged := typedClientCommit(request, preview, "moved-declared-stale")
	forged.Settings.Serving = stale.Settings.Serving.Clone()
	if response := f.rpc(t, "engine:apply-settings", forged); response.Error == nil || !strings.Contains(response.Error.Message, "review is required") {
		t.Fatalf("stale digest with the accepted loads: %+v", response.Error)
	}
	if got := preview.Settings.Serving.Models[0].ExpectedNativeSourceDigest; got == staleDigest {
		t.Fatal("the preview kept the stale digest")
	}
	commit := typedClientCommit(request, preview, "moved-declared-accept")
	if response := f.rpc(t, "engine:apply-settings", commit); response.Error != nil {
		t.Fatalf("declared acceptance: code=%d message=%s", response.Error.Code, response.Error.Message)
	}
	recovered := f.settled(t, "moved-declared-accept", "succeeded")
	if recovered.Serving != nil && recovered.Serving.Status == "mismatch" {
		t.Fatalf("readback after the reviewed apply: %+v", recovered.Serving)
	}
	typedAssertRunningSoon(t, f, "declared engine started by the reviewed apply")
	typedAssertNoMismatch(t, f, "declared engine running after the reviewed apply")
}
