// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

const movedFileReason = "has a new modified time"

// movedFileFixture saves finite typed loads over a model file larger than both
// samples together, so a byte can change between them.
func movedFileFixture(t *testing.T) (modelLoadFixture, string) {
	t.Helper()
	f := newModelLoadFixture(t)
	path := filepath.Join(f.root, "medium.gguf")
	file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write(make([]byte, 4*(64<<10))); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	rt := convertModelLoadFixture(t, f)
	f.st.plat.Runtime = rt
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	preview, err := f.ex.PreviewLaunch(settings.Request{Engine: "fake", Settings: settings.ConfigFromLaunch(state, 54301)})
	if err != nil || len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("saving typed loads: errors=%v conflict=%v err=%v", preview.Errors, preview.Conflict, err)
	}
	f.st.plat.Runtime.ModelLoads = preview.Settings.ModelLoads
	return f, path
}

func movedFileLater(t *testing.T, path string) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	later := info.ModTime().Add(time.Hour)
	if err := os.Chtimes(path, later, later); err != nil {
		t.Fatal(err)
	}
}

func movedFileApproved(t *testing.T, f modelLoadFixture) error {
	t.Helper()
	_, err := compileTypedLlamaInputs(context.Background(), f.st, typedFixtureCommand(t, f, f.st.plat.Runtime), f.st.plat.Runtime, true)
	return err
}

func movedFileRequest(t *testing.T, f modelLoadFixture) settings.Request {
	t.Helper()
	state, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	return settings.Request{Engine: "fake", Settings: settings.ConfigFromLaunch(state, 54301)}
}

// movedFilesUnnamed returns each file whose modified time preview moves from
// saved without a diff entry naming it, the files an apply would accept unseen.
func movedFilesUnnamed(saved []modelconfig.ModelLoad, preview settings.Preview) []string {
	var unnamed []string
	for _, load := range preview.Settings.ModelLoads {
		if load.Resolved == nil {
			continue
		}
		for _, before := range saved {
			if before.ProfileKey != load.ProfileKey || before.Resolved == nil {
				continue
			}
			for _, artifact := range load.Resolved.Artifacts {
				for _, old := range before.Resolved.Artifacts {
					if old.Path != artifact.Path || old.MtimeNS == artifact.MtimeNS {
						continue
					}
					named := false
					for _, diff := range preview.Diff {
						named = named || (strings.Contains(string(diff.After), filepath.Base(artifact.Path)) && string(diff.Before) != string(diff.After))
					}
					if !named {
						unnamed = append(unnamed, filepath.Base(artifact.Path))
					}
				}
			}
		}
	}
	return unnamed
}

// movedFileRecalculate previews Recalculate of the frozen Auto weak-model.
func movedFileRecalculate(t *testing.T, f modelLoadFixture, owner *autoTestMemoryOwner) (settings.Preview, error) {
	t.Helper()
	recalculate := movedFileRequest(t, f)
	set := autoTestRequest()
	set.Context = &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 8192}
	recalculate.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: set}}
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
	return f.ex.PreviewLaunch(recalculate, ctx)
}

func movedFileErrors(preview settings.Preview) string {
	joined := ""
	for _, reason := range preview.Errors {
		joined += reason
	}
	return joined
}

// An untouched file compiles to the saved loads and reads as healthy.
func TestMovedFileNoFalseMismatch(t *testing.T) {
	f, _ := movedFileFixture(t)
	saved := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
	if err := movedFileApproved(t, f); err != nil {
		t.Fatal(err)
	}
	if state, err := f.ex.LaunchSettings("fake"); err != nil || state.ServingReadback == nil || state.ServingReadback.Status == "mismatch" {
		t.Fatalf("healthy readback: %+v err=%v", state.ServingReadback, err)
	}
	preview, err := f.ex.PreviewLaunch(movedFileRequest(t, f))
	if err != nil || len(preview.Errors) != 0 || !modelconfig.EqualLoads(saved, preview.Settings.ModelLoads) {
		t.Fatalf("healthy preview moved the loads: errors=%v err=%v", preview.Errors, err)
	}
}

func TestMovedFileNeedsReview(t *testing.T) {
	assertApprovalRefused := func(t *testing.T, f modelLoadFixture) {
		t.Helper()
		err := movedFileApproved(t, f)
		if err == nil || !isStaleApproval(err) || !strings.Contains(err.Error(), "medium.gguf "+movedFileReason) {
			t.Fatalf("approved compile: %v", err)
		}
		state, stateErr := f.ex.LaunchSettings("fake")
		if stateErr != nil || state.ServingReadback == nil || state.ServingReadback.Status != "mismatch" || !strings.Contains(state.ServingReadback.Reason, "medium.gguf") {
			t.Fatalf("readback: %+v err=%v", state.ServingReadback, stateErr)
		}
	}
	assertAccepted := func(t *testing.T, f modelLoadFixture) {
		t.Helper()
		before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
		preview, err := f.ex.PreviewLaunch(movedFileRequest(t, f))
		if err != nil || len(preview.Errors) != 0 {
			t.Fatalf("no-edit preview: errors=%v err=%v", preview.Errors, err)
		}
		if got := modelconfig.MovedArtifact(before, preview.Settings.ModelLoads); filepath.Base(got) != "medium.gguf" {
			t.Fatalf("moved artifact = %q", got)
		}
		if unnamed := movedFilesUnnamed(before, preview); len(unnamed) != 0 {
			t.Fatalf("the preview accepts %v without naming it", unnamed)
		}
		if !modelconfig.EqualLoads(before, f.st.plat.Runtime.ModelLoads) {
			t.Fatal("a preview changed the saved loads")
		}
		f.st.plat.Runtime.ModelLoads = preview.Settings.ModelLoads
		if err := movedFileApproved(t, f); err != nil {
			t.Fatalf("approved compile after acceptance: %v", err)
		}
	}

	t.Run("same content with a new time", func(t *testing.T) {
		f, path := movedFileFixture(t)
		movedFileLater(t, path)
		assertApprovalRefused(t, f)
		assertAccepted(t, f)
	})
	// A restore or copy leaves another file object, which a saved binding does
	// not record: the same refusal and acceptance apply.
	t.Run("same content in a replacement file", func(t *testing.T) {
		f, path := movedFileFixture(t)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		replacement := path + ".restored"
		if err := os.WriteFile(replacement, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(replacement, path); err != nil {
			t.Fatal(err)
		}
		movedFileLater(t, path)
		assertApprovalRefused(t, f)
		assertAccepted(t, f)
	})
	// The samples cannot tell this from a touch, which is why the time is kept:
	// these bytes are indistinguishable, and must not start with no one asked.
	t.Run("a change between the samples with a new time", func(t *testing.T) {
		f, path := movedFileFixture(t)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		data[len(data)/2] ^= 0xff
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		movedFileLater(t, path)
		assertApprovalRefused(t, f)
		assertAccepted(t, f)
	})
	t.Run("an edit does not carry the acceptance along", func(t *testing.T) {
		f, path := movedFileFixture(t)
		movedFileLater(t, path)
		for name, edit := range map[string]func(*settings.Request){
			"launch text": func(r *settings.Request) { r.Settings.LaunchText += " --metrics" },
			"server port": func(r *settings.Request) { r.Settings.ServerPort, r.Resolution = 54399, "server" },
			"limit":       func(r *settings.Request) { limit := 4096; r.Settings.WorkingSetLimitMiB = &limit },
		} {
			request := movedFileRequest(t, f)
			edit(&request)
			preview, err := f.ex.PreviewLaunch(request)
			if err != nil || len(preview.Errors) == 0 {
				t.Fatalf("%s: errors=%v err=%v", name, preview.Errors, err)
			}
		}
		saved := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
		if _, err := f.ex.SetPort(context.Background(), "fake", 54398); err == nil {
			t.Fatal("a port change accepted the moved file")
		}
		// A request to keep the port carries no edit, and still is no review.
		if _, err := f.ex.SetPort(context.Background(), "fake", f.st.plat.Runtime.Port); err == nil {
			t.Fatal("setting the same port accepted the moved file")
		}
		if !modelconfig.EqualLoads(saved, f.st.plat.Runtime.ModelLoads) {
			t.Fatal("a port request saved the moved file's time")
		}
	})
	t.Run("a finite model change names the file", func(t *testing.T) {
		f, path := movedFileFixture(t)
		movedFileLater(t, path)
		request := movedFileRequest(t, f)
		request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: new(int32)}}}
		*request.ModelChange.Load.Set.Slots = 2
		preview, err := f.ex.PreviewLaunch(request)
		if err != nil || !strings.Contains(movedFileErrors(preview), "medium.gguf "+movedFileReason) {
			t.Fatalf("model change: errors=%v err=%v", preview.Errors, err)
		}
	})
}

func TestMovedFileChangedContentIsRefused(t *testing.T) {
	for name, change := range map[string]func(data []byte) []byte{
		"head": func(data []byte) []byte { data[0] ^= 0xff; return data },
		"tail": func(data []byte) []byte { data[len(data)-1] ^= 0xff; return data },
		"size": func(data []byte) []byte { return append(data, 0) },
	} {
		t.Run(name, func(t *testing.T) {
			f, path := movedFileFixture(t)
			saved := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, change(data), 0o600); err != nil {
				t.Fatal(err)
			}
			movedFileLater(t, path)
			err = movedFileApproved(t, f)
			if err == nil || isStaleApproval(err) || !strings.Contains(err.Error(), "medium.gguf") {
				t.Fatalf("approved compile: %v", err)
			}
			preview, previewErr := f.ex.PreviewLaunch(movedFileRequest(t, f))
			if previewErr != nil || !strings.Contains(movedFileErrors(preview), "Restore the original file") {
				t.Fatalf("preview accepted changed content: errors=%v err=%v", preview.Errors, previewErr)
			}
			if !modelconfig.EqualLoads(saved, f.st.plat.Runtime.ModelLoads) {
				t.Fatal("a refused preview changed the saved loads")
			}
		})
	}
}

// Auto froze a fit against this binding, so plain Apply stays refused and
// Recalculate, which reviews a new result, accepts the time.
func TestMovedFileAutoRecalculates(t *testing.T) {
	f, request, owner := autoTestFrozenLaunch(t)
	f.st.plat.Runtime.ModelLoads = modelconfig.CloneLoads(request.Settings.ModelLoads)
	path := filepath.Join(f.root, "weak.gguf")
	before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
	movedFileLater(t, path)

	err := movedFileApproved(t, f)
	if err == nil || !isStaleApproval(err) || !strings.Contains(err.Error(), "weak.gguf "+movedFileReason) || !strings.Contains(err.Error(), "Recalculate") {
		t.Fatalf("approved compile: %v", err)
	}
	preview, previewErr := f.ex.PreviewLaunch(movedFileRequest(t, f))
	if previewErr != nil || !strings.Contains(movedFileErrors(preview), "Recalculate") {
		t.Fatalf("plain preview: errors=%v err=%v", preview.Errors, previewErr)
	}

	preview, previewErr = movedFileRecalculate(t, f, owner)
	if previewErr != nil || len(preview.Errors) != 0 {
		t.Fatalf("Recalculate: errors=%v err=%v", preview.Errors, previewErr)
	}
	if got := modelconfig.MovedArtifact(before, preview.Settings.ModelLoads); filepath.Base(got) != "weak.gguf" {
		t.Fatalf("Recalculate moved artifact = %q", got)
	}
	if unnamed := movedFilesUnnamed(before, preview); len(unnamed) != 0 {
		t.Fatalf("Recalculate accepts %v without naming it", unnamed)
	}
	f.st.plat.Runtime.ModelLoads = preview.Settings.ModelLoads
	if err := movedFileApproved(t, f); err != nil {
		t.Fatalf("approved compile after Recalculate: %v", err)
	}
}

// Recalculate also accepts a finite sibling's moved file, since plain Apply
// cannot pass the Auto load, but only with the file named in its preview: the
// apply then saves exactly the times that preview showed.
func TestMovedFileRecalculateNamesSiblingFile(t *testing.T) {
	f, request, owner := autoTestFrozenLaunch(t)
	f.st.plat.Runtime.ModelLoads = modelconfig.CloneLoads(request.Settings.ModelLoads)
	saved := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
	movedFileLater(t, filepath.Join(f.root, "medium.gguf"))

	preview, err := movedFileRecalculate(t, f, owner)
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("Recalculate: errors=%v err=%v", preview.Errors, err)
	}
	if got := modelconfig.MovedArtifact(saved, preview.Settings.ModelLoads); filepath.Base(got) != "medium.gguf" {
		t.Fatalf("Recalculate moved artifact = %q", got)
	}
	if unnamed := movedFilesUnnamed(saved, preview); len(unnamed) != 0 {
		t.Fatalf("Recalculate accepts %v without naming it", unnamed)
	}
	if !modelconfig.EqualLoads(saved, f.st.plat.Runtime.ModelLoads) {
		t.Fatal("a preview changed the saved loads")
	}
}
