// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engineintent

import (
	"encoding/json"
	"testing"
)

func TestSnapshotValidate(t *testing.T) {
	valid := func() Snapshot {
		return Snapshot{EnabledByEngine: map[string]bool{"on": true, "local": false, "peer": false, "unknown": false}, OffByEngine: map[string]Origin{"local": "local", "peer": "peer"}}
	}
	t.Run("accepts saved origins and unknown intent", func(t *testing.T) {
		if err := valid().Validate(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("accepts present empty maps", func(t *testing.T) {
		if err := (Snapshot{EnabledByEngine: map[string]bool{}, OffByEngine: map[string]Origin{}}).Validate(); err != nil {
			t.Fatal(err)
		}
	})
	reject := func(name string, mutate func(*Snapshot)) {
		t.Run(name, func(t *testing.T) {
			s := valid()
			mutate(&s)
			if err := s.Validate(); err == nil {
				t.Fatalf("accepted invalid snapshot: %+v", s)
			}
		})
	}
	reject("missing enabled map", func(s *Snapshot) { s.EnabledByEngine = nil })
	reject("missing origin map", func(s *Snapshot) { s.OffByEngine = nil })
	reject("invalid origin", func(s *Snapshot) { s.OffByEngine["local"] = "remote" })
	reject("empty origin", func(s *Snapshot) { s.OffByEngine["local"] = "" })
	reject("origin for enabled engine", func(s *Snapshot) { s.OffByEngine["on"] = "local" })
	reject("origin for unreported engine", func(s *Snapshot) { s.OffByEngine["absent"] = "peer" })
}

func TestSnapshotJSONRequiredMaps(t *testing.T) {
	s := Snapshot{EnabledByEngine: map[string]bool{}, OffByEngine: map[string]Origin{}}
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"enabledByEngine", "offByEngine"} {
		if string(fields[name]) != "{}" {
			t.Fatalf("%s = %s, want present empty object", name, fields[name])
		}
	}
}

func TestSnapshotJSONRejectsNonBooleanEnabledValue(t *testing.T) {
	for _, value := range []string{"null", `"false"`, "0", "[]", "{}"} {
		t.Run(value, func(t *testing.T) {
			var snapshot Snapshot
			data := `{"enabledByEngine":{"ollama":` + value + `},"offByEngine":{}}`
			if err := json.Unmarshal([]byte(data), &snapshot); err == nil {
				t.Fatalf("accepted non-boolean enabled value: %s", value)
			}
		})
	}
}

func TestSnapshotJSONPreservesIntent(t *testing.T) {
	var snapshot Snapshot
	data := `{"enabledByEngine":{"on":true,"off":false},"offByEngine":{"off":"peer"}}`
	if err := json.Unmarshal([]byte(data), &snapshot); err != nil {
		t.Fatal(err)
	}
	if err := snapshot.Validate(); err != nil {
		t.Fatal(err)
	}
	offEnabled, offReported := snapshot.EnabledByEngine["off"]
	if len(snapshot.EnabledByEngine) != 2 || !snapshot.EnabledByEngine["on"] || !offReported || offEnabled {
		t.Fatalf("enabledByEngine = %v, want on=true and off=false", snapshot.EnabledByEngine)
	}
	if len(snapshot.OffByEngine) != 1 || snapshot.OffByEngine["off"] != "peer" {
		t.Fatalf("offByEngine = %v, want off=peer", snapshot.OffByEngine)
	}
}
