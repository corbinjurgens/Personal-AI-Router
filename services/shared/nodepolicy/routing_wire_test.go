// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
)

type strictFixtureCase struct {
	Name   string                `json:"name"`
	Kind   string                `json:"kind"`
	Path   []string              `json:"path"`
	Value  any                   `json:"value"`
	Remove bool                  `json:"remove"`
	Valid  bool                  `json:"valid"`
	State  StrictCapabilityState `json:"state"`
}

type strictFixtures struct {
	Identity map[string]any      `json:"identity"`
	Request  map[string]any      `json:"request"`
	Model    map[string]any      `json:"model"`
	Cases    []strictFixtureCase `json:"cases"`
}

func loadStrictFixtures(t *testing.T) strictFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/strict-admission.json")
	if err != nil {
		t.Fatal(err)
	}
	var f strictFixtures
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// The desktop suite consumes this exact corpus, including nested mutations.
func TestStrictAdmissionParityCorpus(t *testing.T) {
	for _, tc := range loadStrictFixtures(t).Cases {
		t.Run(tc.Name, func(t *testing.T) {
			f := loadStrictFixtures(t)
			f.Request["identity"] = f.Identity
			f.Model["identity"] = f.Identity
			capability := map[string]any{"version": 1, "models": []any{f.Model}}
			target := map[string]map[string]any{"request": f.Request, "identity": f.Identity, "capability": capability, "model": f.Model}[tc.Kind]
			if target == nil {
				t.Fatalf("unknown fixture kind %q", tc.Kind)
			}
			if len(tc.Path) > 0 {
				for _, key := range tc.Path[:len(tc.Path)-1] {
					child, ok := target[key].(map[string]any)
					if !ok {
						t.Fatalf("invalid fixture path %v", tc.Path)
					}
					target = child
				}
				key := tc.Path[len(tc.Path)-1]
				if tc.Remove {
					delete(target, key)
				} else {
					target[key] = tc.Value
				}
			}
			var input any = f.Request
			if tc.Kind == "capability" || tc.Kind == "model" {
				input = capability
			}
			data, err := json.Marshal(input)
			if err != nil {
				t.Fatal(err)
			}
			if tc.Kind == "capability" || tc.Kind == "model" {
				var got StrictAdmissionCapability
				err = json.Unmarshal(data, &got)
				if (err == nil) != tc.Valid {
					t.Fatalf("decode error = %v, want valid %v", err, tc.Valid)
				}
				if err != nil {
					return
				}
				want := tc.State
				if want == "" {
					want = StrictCapabilityValid
				}
				if got.State() != want {
					t.Fatalf("state = %q, want %q", got.State(), want)
				}
				encoded, err := json.Marshal(got)
				if err != nil {
					t.Fatal(err)
				}
				var again StrictAdmissionCapability
				if err := json.Unmarshal(encoded, &again); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, again) {
					t.Fatalf("unstable round trip: %+v / %+v", got, again)
				}
			} else {
				var got StrictAdmissionRequest
				err = json.Unmarshal(data, &got)
				if (err == nil) != tc.Valid {
					t.Fatalf("decode error = %v, want valid %v", err, tc.Valid)
				}
				if err != nil {
					return
				}
				encoded, err := json.Marshal(got)
				if err != nil {
					t.Fatal(err)
				}
				var again StrictAdmissionRequest
				if err := json.Unmarshal(encoded, &again); err != nil {
					t.Fatal(err)
				}
				if got != again {
					t.Fatalf("unstable round trip: %+v / %+v", got, again)
				}
			}
		})
	}
}

func TestStrictAdmissionNumericSpellings(t *testing.T) {
	f := loadStrictFixtures(t)
	f.Request["identity"] = f.Identity
	data, err := json.Marshal(f.Request)
	if err != nil {
		t.Fatal(err)
	}
	for _, spelling := range []string{"1.0", "1e0"} {
		t.Run(spelling, func(t *testing.T) {
			var got StrictAdmissionRequest
			if err := json.Unmarshal([]byte(strings.Replace(string(data), `"version":1`, `"version":`+spelling, 1)), &got); err != nil {
				t.Fatal(err)
			}
			if got.Version != 1 {
				t.Fatalf("version = %d", got.Version)
			}
		})
	}
}

func TestStrictAdmissionUnsupportedRequestError(t *testing.T) {
	err := (StrictAdmissionRequest{Version: 2}).Validate()
	if !errors.Is(err, ErrStrictAdmissionVersion) {
		t.Fatalf("error = %v", err)
	}
}

// decodeStrictContract must reject a duplicate key at its object level before
// the map collapse silently keeps the last occurrence. The duplicate value is
// listed first so a collapsing decode would accept the object.
func TestStrictContractDuplicateObjectKeys(t *testing.T) {
	identity := `{"engineRunId":"engine-1","settingsFingerprint":"` + strings.Repeat("a", 64) + `","modelDigest":"` + strings.Repeat("b", 64) + `"}`
	duplicateIdentity := `{"engineRunId":"rogue-run","engineRunId":"engine-1","settingsFingerprint":"` + strings.Repeat("a", 64) + `","modelDigest":"` + strings.Repeat("b", 64) + `"}`
	prepareResident := func(inner string) string {
		return `{"policyRevision":"1","proxyRunId":"proxy-1","engine":"ollama","model":"m:latest","identity":` + inner + `,"operationId":"resident:1","timeoutMs":60000}`
	}
	reject := func(t *testing.T, err error) {
		t.Helper()
		if err == nil {
			t.Fatal("accepted a contract with a duplicated key")
		}
		if !strings.Contains(err.Error(), "duplicate field") {
			t.Fatalf("error = %v, want a duplicate field error", err)
		}
	}
	t.Run("prepare resident rejects nested duplicate identity field", func(t *testing.T) {
		var got PrepareResidentParams
		err := json.Unmarshal([]byte(prepareResident(duplicateIdentity)), &got)
		reject(t, err)
	})
	t.Run("identity rejects duplicate field", func(t *testing.T) {
		var got StrictModelIdentity
		err := json.Unmarshal([]byte(duplicateIdentity), &got)
		reject(t, err)
	})
	t.Run("control: duplicate-free objects decode", func(t *testing.T) {
		var params PrepareResidentParams
		if err := json.Unmarshal([]byte(prepareResident(identity)), &params); err != nil {
			t.Fatal(err)
		}
		if params.Identity.EngineRunID != "engine-1" {
			t.Fatalf("identity.engineRunId = %q, want engine-1", params.Identity.EngineRunID)
		}
		var direct StrictModelIdentity
		if err := json.Unmarshal([]byte(identity), &direct); err != nil {
			t.Fatal(err)
		}
		if direct.EngineRunID != "engine-1" {
			t.Fatalf("engineRunId = %q, want engine-1", direct.EngineRunID)
		}
	})
}

func TestStrictCapabilityCardinalityAndDuplicates(t *testing.T) {
	f := loadStrictFixtures(t)
	f.Model["identity"] = f.Identity
	data, err := json.Marshal(f.Model)
	if err != nil {
		t.Fatal(err)
	}
	var model StrictAdmissionModel
	if err := json.Unmarshal(data, &model); err != nil {
		t.Fatal(err)
	}
	t.Run("duplicate model", func(t *testing.T) {
		c := StrictAdmissionCapability{Version: 1, Models: []StrictAdmissionModel{model, model}}
		if err := c.Validate(); err == nil {
			t.Fatal("accepted duplicate engine/model")
		}
	})
	t.Run("too many models", func(t *testing.T) {
		c := StrictAdmissionCapability{Version: 1, Models: make([]StrictAdmissionModel, 1025)}
		if err := c.Validate(); err == nil {
			t.Fatal("accepted excess models")
		}
	})
}
