// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"testing"

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
			snapshot := ollamaSettings()
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
