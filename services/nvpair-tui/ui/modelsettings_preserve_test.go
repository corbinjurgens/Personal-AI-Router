// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"strings"
	"testing"

	"nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

func TestModelSettingsPreservedByOrdinaryEdits(t *testing.T) {
	for _, tc := range []struct {
		name  string
		mode  detailInputMode
		value string
	}{
		{"server port", detailInputEnginePort, "12500"},
		{"proxy port", detailInputProxyPort, "12501"},
		{"launch arguments", detailInputLaunchArgs, "--verbose"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := localDetail()
			snapshot := llamaCPPSettings()
			slots := int32(2)
			snapshot.Settings.ModelLoads = []modelconfig.ModelLoad{{ModelID: "exact/model", Requested: modelconfig.RequestedLoad{Slots: &slots}}}
			request, ok := d.settingsRequest(snapshot, tc.mode, tc.value)
			if !ok || !modelconfig.EqualLoads(request.Settings.ModelLoads, snapshot.Settings.ModelLoads) {
				t.Fatal("ordinary edit did not preserve model loads")
			}
			*request.Settings.ModelLoads[0].Requested.Slots = 3
			if *snapshot.Settings.ModelLoads[0].Requested.Slots != 2 {
				t.Fatal("request aliases the snapshot")
			}
		})
	}
}

// The preview request carries the saved loads, and the commit carries the loads
// the preview returned, recompiled by the target. The TUI adds no rule of its own.
func TestSettingsCommitSendsThePreviewLoadsWhole(t *testing.T) {
	saved := []modelconfig.ModelLoad{{ProfileKey: "exact/model", ModelID: "exact/model", Resolved: &modelconfig.ResolvedLoad{
		NativeBuildID: "build", SourceInputDigest: strings.Repeat("a", 64), PresetDigest: strings.Repeat("b", 64),
	}}}
	compiled := modelconfig.CloneLoads(saved)
	compiled[0].Resolved.SourceInputDigest = strings.Repeat("c", 64)
	compiled[0].Resolved.PresetDigest = strings.Repeat("d", 64)
	for _, tc := range []struct {
		name       string
		mode       detailInputMode
		value      string
		resolution string
	}{
		{"server port", detailInputEnginePort, "12500", resolutionServer},
		{"launch arguments", detailInputLaunchArgs, "--verbose", resolutionLaunch},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := localDetail()
			snapshot := llamaCPPSettings()
			snapshot.Settings.ModelLoads = modelconfig.CloneLoads(saved)
			request, ok := d.settingsRequest(snapshot, tc.mode, tc.value)
			if !ok || request.Resolution != tc.resolution || !modelconfig.EqualLoads(request.Settings.ModelLoads, saved) {
				t.Fatalf("preview request = %+v", request)
			}
			verdict, commit, problem := judgeSettingsPreview(enginePreviewMsg{request: request, preview: enginesettings.Preview{Settings: enginesettings.Config{ModelLoads: compiled}}})
			if verdict != settingsWrite || problem != "" {
				t.Fatalf("verdict %v, problem %q", verdict, problem)
			}
			if !modelconfig.EqualLoads(commit.Settings.ModelLoads, compiled) || commit.Resolution != "" || commit.RequestID == "" {
				t.Fatalf("commit = %+v", commit)
			}
		})
	}
}
