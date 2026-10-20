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
