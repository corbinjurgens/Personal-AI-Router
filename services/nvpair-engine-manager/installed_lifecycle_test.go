// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"nvpair-shared/modelconfig"
)

func TestDroppedEngineIntentPreservesCleanupEvidence(t *testing.T) {
	state, err := decodeDesiredState([]byte(`{"engines":{"fake":false},"offBy":{"fake":"peer"},"allocatedPorts":{"fake":23456}}`))
	if err != nil {
		t.Fatal(err)
	}
	if state.AllocatedPorts["fake"] != 23456 {
		t.Fatalf("cleanup evidence=%+v", state)
	}
	raw, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"engines", "offBy"} {
		if _, ok := fields[name]; ok {
			t.Fatalf("retained %s", name)
		}
	}
}

func TestInstalledEngineRestoresAfterResumeWithoutIntent(t *testing.T) {
	e := desiredTestExecutor(t, testEngineManifest(fakeEngineBin), t.TempDir())
	t.Cleanup(e.StopAll)
	e.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	if err := e.RestoreInstalled(context.Background()); err != nil && rpcCode(err) != codeLocalVeto {
		t.Fatalf("restore while paused: %v", err)
	}
	status, err := e.Status("fake")
	if err != nil || status.Running {
		t.Fatalf("paused status=%+v err=%v", status, err)
	}
	e.nodeProjection.set(nodeProjection{})
	if err := e.RestoreInstalled(context.Background()); err != nil {
		t.Fatal(err)
	}
	status, err = e.Status("fake")
	if err != nil || !status.Running {
		t.Fatalf("resumed status=%+v err=%v", status, err)
	}
}

func TestTypedRouterCapacityIsManagerOwnedUnlimited(t *testing.T) {
	command := llamaSnapshotCommand(launchCommand{Args: []string{"--models-max", "2"}}, "preset.ini", false, []modelconfig.ModelLoad{{ProfileKey: "profile"}})
	i := slices.Index(command.Args, "--models-max")
	if i < 0 || i+1 >= len(command.Args) || command.Args[i+1] != "0" {
		t.Fatalf("args=%v", command.Args)
	}
}

// The router's --log-verbosity overrides every child preset section.
func TestNativeVerbosityOptionIsPairOwned(t *testing.T) {
	for _, key := range []string{"log-verbosity", "verbosity", "lv", "verbose", "log-verbose", "v"} {
		err := validateModelExtraOptions("profile", map[string]string{key: "2"})
		if err == nil || !strings.Contains(err.Error(), "PAIR owns "+key+".") {
			t.Errorf("%s: error=%v", key, err)
		}
	}
}

func TestNativeIdleOptionIsPairOwned(t *testing.T) {
	err := validateModelExtraOptions("profile", map[string]string{"sleep-idle-seconds": "300"})
	if err == nil || !strings.Contains(err.Error(), "PAIR owns sleep-idle-seconds.") {
		t.Fatalf("error=%v", err)
	}
	options, err := llamaModelSettingsOptions()
	if err != nil {
		t.Fatal(err)
	}
	for _, option := range options {
		if option.Descriptor.Field == "sleepIdleSeconds" {
			t.Fatal("native idle descriptor retained")
		}
	}
}

func TestTypedPresetDisablesNativeIdleForEveryModel(t *testing.T) {
	f := newModelLoadFixture(t)
	rt := convertModelLoadFixture(t, f)
	command := typedFixtureCommand(t, f, rt)
	prepared, err := compilePreparedServing(f.st, command, rt, false)
	if err != nil {
		t.Fatal(err)
	}
	sections, err := parseLlamaServingPresetAllowing(prepared.Snapshot, modelLoadAllowedPresetKeys(prepared.ModelLoads))
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared.ModelLoads) == 0 {
		t.Fatal("fixture has no typed loads")
	}
	for _, load := range prepared.ModelLoads {
		if sections[load.ProfileKey]["sleep-idle-seconds"] != "-1" {
			t.Fatalf("model %s options=%v", load.ProfileKey, sections[load.ProfileKey])
		}
	}
}
