// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package enginesettings

import (
	"encoding/json"
	"testing"

	"nvpair-shared/engines"
)

func TestConfigCloneAndLaunchReconstruction(t *testing.T) {
	ram, vram := int64(1024), int64(0)
	launch := LaunchState{ServerPort: 8080, LaunchText: "--opaque", Serving: &engines.ServingConfig{Version: 1, Models: []engines.ServingModel{{ModelID: "exact", Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram}}}}}
	c := ConfigFromLaunch(launch, 8081)
	copy := c.Clone()
	if !c.Equal(copy) || c.ProxyPort != 8081 {
		t.Fatal("launch reconstruction differs")
	}
	*copy.Serving.Models[0].Memory.HostRAMBytes = 2
	if c.Equal(copy) || *launch.Serving.Models[0].Memory.HostRAMBytes != 1024 {
		t.Fatal("config clone aliases launch")
	}
	snapshot := Snapshot{Settings: c, Serving: &engines.ServingReadback{Observed: []engines.ServingObservation{{Slots: 1}}}}
	cloned := snapshot.Clone()
	cloned.Settings.Serving.Models[0].ModelID = "changed"
	cloned.Serving.Observed[0].Slots = 2
	if snapshot.Settings.Serving.Models[0].ModelID != "exact" || snapshot.Serving.Observed[0].Slots != 1 {
		t.Fatal("snapshot clone aliases original")
	}
}

func TestConfigLegacyJSONByteStability(t *testing.T) {
	data, err := json.Marshal(Config{ServerPort: 8080, ProxyPort: 8081, LaunchText: "--opaque"})
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"serverPort":8080,"proxyPort":8081,"launchText":"--opaque"}` {
		t.Fatal("legacy config encoding changed")
	}
}
