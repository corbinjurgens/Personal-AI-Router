// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func modelBufferDiagnosticLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return &logs
}

func assertModelBufferDiagnostic(t *testing.T, logs *bytes.Buffer, stage string) map[string]any {
	t.Helper()
	if bytes.Count(logs.Bytes(), []byte("\n")) != 1 {
		t.Fatalf("expected one warning, got %s", logs.String())
	}
	var record map[string]any
	if err := json.Unmarshal(logs.Bytes(), &record); err != nil {
		t.Fatal(err)
	}
	if record["stage"] != stage || record["level"] != "WARN" || record["msg"] != "model buffer capture unavailable" {
		t.Fatalf("unexpected warning: %+v", record)
	}
	return record
}

func TestModelBufferCaptureCreationDiagnostics(t *testing.T) {
	test := func(name, stage, reason string, change func(*testing.T, *modelBufferTestFixture, **llamaServingLaunch, *launchCommand)) {
		t.Run(name, func(t *testing.T) {
			f := newModelBufferTestFixture(t)
			logs := modelBufferDiagnosticLogs(t)
			prepared := f.capture.prepared
			command := f.load.st.plat.Runtime.servingCommand
			change(t, f, &prepared, &command)
			c := f.load.ex.newModelBufferCapture(f.load.st, "fake", prepared, nil, command)
			if c != nil {
				defer c.retire()
			}
			// Projection must not repeat the creation warning on every Inspect.
			f.load.st.bufferCapture = c
			for range 2 {
				got := f.load.ex.projectModelBufferObservation(f.load.st, "fake", "medium-model", nil)
				if got.Status != "unknown" || got.Reason != "unsupported-build" {
					t.Fatalf("observation changed: %+v", got)
				}
			}
			record := assertModelBufferDiagnostic(t, logs, stage)
			if reason != "" && record["reason"] != reason {
				t.Fatalf("reason = %v, want %q", record["reason"], reason)
			}
			if strings.Contains(logs.String(), f.load.ex.baseDir) {
				t.Fatal("warning leaked a filesystem path")
			}
		})
	}
	test("unprepared", "capture-create", "unprepared", func(_ *testing.T, _ *modelBufferTestFixture, p **llamaServingLaunch, _ *launchCommand) { *p = nil })
	test("no loads", "capture-create", "no-loads", func(_ *testing.T, _ *modelBufferTestFixture, p **llamaServingLaunch, _ *launchCommand) {
		*p = &llamaServingLaunch{}
	})
	test("budget", "capture-create", "budget", func(_ *testing.T, f *modelBufferTestFixture, _ **llamaServingLaunch, _ *launchCommand) {
		f.load.ex.modelBuffers.bytes = modelBufferCaptureBytes
	})
	test("constrain command", "constrain-command", "Reviewed Auto GPU requires an exact physical binding.", func(_ *testing.T, _ *modelBufferTestFixture, p **llamaServingLaunch, _ *launchCommand) {
		(*p).ModelLoads[0].Resolved.ResolverRevision = "auto/test"
		(*p).ModelLoads[0].Resolved.GPU.Mode = "single"
		(*p).ModelLoads[0].Resolved.GPU.Layers = nil
		(*p).ModelLoads[0].Resolved.DeviceBinding = nil
	})
	test("runtime snapshot", "runtime-snapshot", "", func(t *testing.T, _ *modelBufferTestFixture, _ **llamaServingLaunch, c *launchCommand) {
		if err := os.Mkdir(filepath.Join(filepath.Dir(c.Bin), "ggml-invalid.dll"), 0o700); err != nil {
			t.Fatal(err)
		}
	})
	missingExecutable := filepath.Join(t.TempDir(), "missing")
	_, bindingErr := servingFileBinding(missingExecutable)
	if bindingErr == nil {
		t.Fatal("missing executable unexpectedly has a serving binding")
	}
	test("executable binding", "executable-binding", modelBufferDiagnosticError(bindingErr), func(_ *testing.T, _ *modelBufferTestFixture, _ **llamaServingLaunch, c *launchCommand) {
		c.Bin = missingExecutable
	})
	test("native environment", "native-environment", "Reviewed Auto native library search escapes its selected build.", func(_ *testing.T, _ *modelBufferTestFixture, _ **llamaServingLaunch, c *launchCommand) {
		c.Env = map[string]string{"LD_LIBRARY_PATH": "private-missing-directory"}
	})
	test("device environment", "device-environment", "unsupported native injection/MIG/MPS environment", func(_ *testing.T, _ *modelBufferTestFixture, _ **llamaServingLaunch, c *launchCommand) {
		c.Env = map[string]string{"LD_PRELOAD": "private-value"}
	})
}

// Creation can be the manager's first full hash of the native build. A CUDA
// build directory of about 880 MB took seconds, so a fixed 200 ms budget left every run
// started by a fresh manager without a source identity, and so without samples.
func TestModelBufferCaptureCreationSourceBudget(t *testing.T) {
	t.Run("slow native hash still binds the source", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		logs := modelBufferDiagnosticLogs(t)
		nativeBuildHashes.gate <- struct{}{} // another hash in progress
		released := make(chan struct{})
		go func() {
			time.Sleep(500 * time.Millisecond)
			<-nativeBuildHashes.gate
			close(released)
		}()
		t.Cleanup(func() { <-released })
		command := f.load.st.plat.Runtime.servingCommand
		c := f.load.ex.newModelBufferCapture(f.load.st, "fake", f.capture.prepared, nil, command, context.Background())
		if c == nil {
			t.Fatal("capture unavailable")
		}
		defer c.retire()
		if c.sourceID == "" || c.sourceID != f.capture.sourceID || c.buildID != f.capture.buildID {
			t.Fatalf("source %q build %q, want %q and %q", c.sourceID, c.buildID, f.capture.sourceID, f.capture.buildID)
		}
		if logs.Len() != 0 {
			t.Fatalf("unexpected warning: %s", logs.String())
		}
	})
	t.Run("cancelled start still ends the check", func(t *testing.T) {
		f := newModelBufferTestFixture(t)
		logs := modelBufferDiagnosticLogs(t)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		command := f.load.st.plat.Runtime.servingCommand
		c := f.load.ex.newModelBufferCapture(f.load.st, "fake", f.capture.prepared, nil, command, ctx)
		if c != nil {
			defer c.retire()
			if c.sourceID != "" {
				t.Fatal("cancelled start bound a source identity")
			}
		}
		assertModelBufferDiagnostic(t, logs, "runtime-snapshot")
	})
}

func TestModelBufferCaptureWithoutInstallDeclaration(t *testing.T) {
	f := newModelBufferTestFixture(t)
	f.load.st.plat.Install = nil
	command := f.load.st.plat.Runtime.servingCommand
	capture := f.load.ex.newModelBufferCapture(f.load.st, "fake", f.capture.prepared, nil, command)
	if capture == nil {
		t.Fatal("content-identified runtime without install metadata lost buffer capture")
	}
	defer capture.retire()
}

func TestModelBufferNativeSourceDiagnosticStages(t *testing.T) {
	test := func(name, stage string, setup func(string) (context.Context, string)) {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			logs := modelBufferDiagnosticLogs(t)
			ctx, root := setup(root)
			var diagnostic modelBufferSourceDiagnostic
			_, err := modelReviewedNativeRuntimeSource(ctx, modelDeviceRuntime{Root: root}, launchCommand{}, func(stage string, err error) {
				diagnostic = modelBufferSourceDiagnostic{stage: stage, err: err}
			})
			if err == nil || diagnostic.stage != stage || diagnostic.err == nil {
				t.Fatalf("diagnostic = %+v, error = %v", diagnostic, err)
			}
			diagnostic.warn()
			assertModelBufferDiagnostic(t, logs, stage)
		})
	}
	test("root read", "root-read", func(root string) (context.Context, string) {
		return context.Background(), filepath.Join(root, "missing")
	})
	test("driver library", "driver-library", func(root string) (context.Context, string) {
		if err := os.WriteFile(filepath.Join(root, "nvcuda.dll"), []byte("synthetic"), 0o600); err != nil {
			t.Fatal(err)
		}
		return context.Background(), root
	})
}

func TestModelBufferComparisonDiagnosticsOncePerCapture(t *testing.T) {
	test := func(name, stage string, change func(*modelBufferProps)) {
		t.Run(name, func(t *testing.T) {
			f := newModelBufferTestFixture(t)
			logs := modelBufferDiagnosticLogs(t)
			f.initial("medium-model", 31001)
			frozen := f.capture.frozen["medium-model"]
			p := modelBufferProps{Build: "b12020-0ec4027a2", RouterBuild: "b12020-0ec4027a2", Context: int64(frozen.context), Slots: int64(frozen.slots)}
			change(&p)
			for sweep := uint64(1); sweep <= 2; sweep++ {
				f.load.ex.observeModelBuffers(context.Background(), f.load.st, f.capture, sweep, []string{"medium-model"}, true, json.RawMessage(`{"data":[{"id":"medium-model","status":{"value":"loaded"}}]}`), map[string]modelBufferProps{"medium-model": p})
			}
			record := assertModelBufferDiagnostic(t, logs, stage)
			switch stage {
			case "router-build":
				if record["build"] != p.Build {
					t.Fatalf("build = %v", record["build"])
				}
				if stage == "router-build" && record["router_build"] != p.RouterBuild {
					t.Fatalf("router build = %v", record["router_build"])
				}
			case "context":
				if record["context"] != float64(p.Context) || record["frozen_context"] != float64(frozen.context) {
					t.Fatalf("context values = %+v", record)
				}
			case "slots":
				if record["slots"] != float64(p.Slots) || record["frozen_slots"] != float64(frozen.slots) {
					t.Fatalf("slot values = %+v", record)
				}
			}
			if !f.capture.children[31001].poisoned {
				t.Fatal("comparison no longer poisons child")
			}
		})
	}
	test("missing build", "router-build", func(p *modelBufferProps) { p.Build = "" })
	test("router build mismatch", "router-build", func(p *modelBufferProps) { p.RouterBuild = "other" })
	test("context mismatch", "context", func(p *modelBufferProps) { p.Context++ })
	test("slots mismatch", "slots", func(p *modelBufferProps) { p.Slots++ })
}

func TestModelBufferSourceDigestDiagnostic(t *testing.T) {
	f := newModelBufferTestFixture(t)
	logs := modelBufferDiagnosticLogs(t)
	runtime := modelDeviceRuntime{
		Root:      t.TempDir(),
		Libraries: []modelDeviceLibrary{{Name: strings.Repeat("a", modelDevicesBytes)}},
	}
	var diagnostic modelBufferSourceDiagnostic
	_, err := modelReviewedNativeRuntimeSource(context.Background(), runtime, f.load.st.plat.Runtime.servingCommand, func(stage string, err error) {
		diagnostic = modelBufferSourceDiagnostic{stage: stage, err: err}
	})
	if err == nil || diagnostic.err == nil {
		t.Fatalf("expected digest bound failure, got %v", err)
	}
	diagnostic.warn()
	record := assertModelBufferDiagnostic(t, logs, "source-digest")
	if record["reason"] != "device identity exceeds bound" {
		t.Fatalf("reason = %v", record["reason"])
	}
}

func TestModelBufferDiagnosticRedaction(t *testing.T) {
	for _, err := range []error{&os.PathError{Op: "open", Path: "private-file", Err: os.ErrNotExist}, &os.LinkError{Op: "rename", Old: "secret", New: "other", Err: os.ErrPermission}, errors.New("open /private/file"), errors.New(`open C:\private\file`), errors.New("bad\nline")} {
		if got, want := modelBufferDiagnosticError(err), fmt.Sprintf("%T", err); got != want {
			t.Fatalf("redacted error = %q, want %q", got, want)
		}
	}
	if got := modelBufferDiagnosticError(errors.New("context deadline exceeded")); got != "context deadline exceeded" {
		t.Fatalf("safe error = %q", got)
	}
	if got := modelBufferDiagnosticBuild(`C:\private\build`); got != "redacted" {
		t.Fatalf("unsafe build = %q", got)
	}
}
