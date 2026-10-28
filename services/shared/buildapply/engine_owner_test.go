// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func engineOwnerWireFixture(t *testing.T) EngineOwnerEvidence {
	t.Helper()
	revision := "sha256:" + strings.Repeat("a", 64)
	e := EngineOwnerEvidence{
		ManagerIncarnation: "native-manager",
		Intent:             Known(EngineOwnerIntent{Revision: revision, EnabledByEngine: map[string]bool{"fake": false}, OffByEngine: map[string]string{"fake": "local"}}),
		Locks:              Known(EngineOwnerLocks{Revision: revision, LockedByEngine: map[string]bool{"fake": false}}),
		Universe:           Known([]EngineOwnerUniverseEntry{{Engine: "fake", Registered: true, Supported: Known(true), Installed: Known(false), Running: Known(false), TerminationVerified: Known(true), ModelRoots: Known([]string{}), DataRoots: Known([]string{}), RootsComplete: true}}),
		StopAuthority:      Known([]EngineOwnerStopAuthority{{Engine: "fake", Scoped: Known(true), Adopted: Known(false), ReclaimPending: Known(false), Reason: "no_live_process"}}),
		Reservations:       Known(EngineOwnerReservations{Revision: revision, BlocksApply: Known(false), UnsafeCopiesComplete: true, Operations: []EngineOwnerOperation{}}),
		Reasons:            []EngineOwnerReason{},
	}
	if err := e.Finalize(); err != nil {
		t.Fatalf("finalize fixture: %v", err)
	}
	return e
}

func TestEngineOwnerEvidenceWireRoundTrip(t *testing.T) {
	e := engineOwnerWireFixture(t)
	data, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var got EngineOwnerEvidence
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(e, got) || !got.Complete {
		t.Fatalf("roundtrip = %+v", got)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	want := []string{"managerIncarnation", "captureToken", "intent", "locks", "universe", "stopAuthority", "reservations", "complete", "reasons"}
	if len(fields) != len(want) {
		t.Fatalf("unexpected response fields: %v", fields)
	}
	for _, name := range want {
		if _, exists := fields[name]; !exists {
			t.Fatalf("missing %s", name)
		}
	}
}

func TestEngineOwnerEvidenceWireRejectsInvalidFacts(t *testing.T) {
	test := func(name string, mutate func(*EngineOwnerEvidence)) {
		t.Run(name, func(t *testing.T) {
			e := engineOwnerWireFixture(t)
			mutate(&e)
			if err := e.Finalize(); err == nil {
				t.Fatal("accepted invalid owner facts")
			}
		})
	}
	test("empty incarnation", func(e *EngineOwnerEvidence) { e.ManagerIncarnation = "" })
	test("missing intent revision", func(e *EngineOwnerEvidence) { e.Intent.Value.Revision = "" })
	test("missing locks revision", func(e *EngineOwnerEvidence) { e.Locks.Value.Revision = "" })
	test("missing reservation revision", func(e *EngineOwnerEvidence) { e.Reservations.Value.Revision = "" })
	test("duplicate engines", func(e *EngineOwnerEvidence) {
		entries := append(*e.Universe.Value, (*e.Universe.Value)[0])
		e.Universe = Known(entries)
	})
	test("duplicate authority", func(e *EngineOwnerEvidence) {
		entries := append(*e.StopAuthority.Value, (*e.StopAuthority.Value)[0])
		e.StopAuthority = Known(entries)
	})
	test("missing authority engine", func(e *EngineOwnerEvidence) { e.StopAuthority = Known([]EngineOwnerStopAuthority{}) })
	test("known null universe", func(e *EngineOwnerEvidence) { e.Universe = Known[[]EngineOwnerUniverseEntry](nil) })
	test("known null roots", func(e *EngineOwnerEvidence) { (*e.Universe.Value)[0].ModelRoots = Known[[]string](nil) })
	test("complete roots with unknown data", func(e *EngineOwnerEvidence) { (*e.Universe.Value)[0].DataRoots = Observation[[]string]{} })
	test("missing explicit Off origin", func(e *EngineOwnerEvidence) { delete(e.Intent.Value.OffByEngine, "fake") })
	test("orphan durable key", func(e *EngineOwnerEvidence) { e.Locks.Value.LockedByEngine["retained"] = false })
	test("locked peer Off", func(e *EngineOwnerEvidence) {
		e.Locks.Value.LockedByEngine["fake"] = true
		e.Intent.Value.OffByEngine["fake"] = "peer"
	})
	test("unbounded reason text", func(e *EngineOwnerEvidence) { (*e.StopAuthority.Value)[0].Reason = "raw native command error" })
	test("scoped process without lifetime", func(e *EngineOwnerEvidence) { (*e.StopAuthority.Value)[0].Reason = "owned_process_lifetime" })
	test("known zero PID", func(e *EngineOwnerEvidence) {
		(*e.StopAuthority.Value)[0].ProcessLifetime = Known(EngineOwnerProcessLifetime{EngineRunID: "run", PID: Known(0)})
	})
	test("known null operations", func(e *EngineOwnerEvidence) { e.Reservations.Value.Operations = nil })
	test("duplicate operation identities", func(e *EngineOwnerEvidence) {
		op := EngineOwnerOperation{ID: "op", Engine: "fake", Kind: "copy", Phase: "active", SourceIncarnation: Known(e.ManagerIncarnation)}
		e.Reservations.Value.BlocksApply = Known(true)
		e.Reservations.Value.Operations = []EngineOwnerOperation{op, op}
	})
	test("unexplained unknown section", func(e *EngineOwnerEvidence) { e.Intent = Observation[EngineOwnerIntent]{} })
}

func TestEngineOwnerEvidenceWireRejectsAmbiguousShape(t *testing.T) {
	fixture := engineOwnerWireFixture(t)
	data, err := json.Marshal(fixture)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	for field := range fields {
		t.Run("missing "+field, func(t *testing.T) {
			copy := make(map[string]json.RawMessage, len(fields))
			for key, value := range fields {
				if key != field {
					copy[key] = value
				}
			}
			input, err := json.Marshal(copy)
			if err != nil {
				t.Fatal(err)
			}
			var got EngineOwnerEvidence
			if err := json.Unmarshal(input, &got); err == nil {
				t.Fatal("accepted missing field")
			}
		})
	}
	test := func(name, input string) {
		t.Run(name, func(t *testing.T) {
			var got EngineOwnerEvidence
			if err := json.Unmarshal([]byte(input), &got); err == nil {
				t.Fatal("accepted ambiguous response")
			}
		})
	}
	test("duplicate field", strings.Replace(string(data), `"complete":true`, `"complete":true,"complete":true`, 1))
	test("case variant field", strings.Replace(string(data), `"complete":true`, `"Complete":true`, 1))
	test("case variant nested field", strings.Replace(string(data), `"rootsComplete":true`, `"RootsComplete":true`, 1))
	test("null saved boolean", strings.Replace(string(data), `"enabledByEngine":{"fake":false}`, `"enabledByEngine":{"fake":null}`, 1))
	test("null lock boolean", strings.Replace(string(data), `"lockedByEngine":{"fake":false}`, `"lockedByEngine":{"fake":null}`, 1))
	test("false completeness", strings.Replace(string(data), `"complete":true`, `"complete":false`, 1))
	test("changed capture content", strings.Replace(string(data), `"native-manager"`, `"replacement-manager"`, 1))
	// Keep padding inside the object so encoding/json passes it to the decoder.
	test("oversized response", strings.Replace(string(data), "{", "{"+strings.Repeat(" ", MaxStatusBytes), 1))
}

func TestEngineOwnerEvidenceCaptureBindsUnknownAndAuthority(t *testing.T) {
	test := func(name string, mutate func(*EngineOwnerEvidence)) {
		t.Run(name, func(t *testing.T) {
			e := engineOwnerWireFixture(t)
			before := e.CaptureToken
			mutate(&e)
			if err := e.Finalize(); err != nil {
				t.Fatal(err)
			}
			if e.CaptureToken == before {
				t.Fatal("capture did not bind changed facts")
			}
		})
	}
	test("native incarnation", func(e *EngineOwnerEvidence) { e.ManagerIncarnation = "next-manager" })
	test("intent key presence", func(e *EngineOwnerEvidence) {
		delete(e.Intent.Value.EnabledByEngine, "fake")
		delete(e.Intent.Value.OffByEngine, "fake")
	})
	test("false lock key presence", func(e *EngineOwnerEvidence) { delete(e.Locks.Value.LockedByEngine, "fake") })
	test("unknown roots", func(e *EngineOwnerEvidence) {
		(*e.Universe.Value)[0].RootsComplete = false
		(*e.Universe.Value)[0].ModelRoots = Observation[[]string]{}
	})
	test("content authority revision", func(e *EngineOwnerEvidence) { e.Reservations.Value.Revision = "sha256:" + strings.Repeat("b", 64) })
	test("retained unknown origin", func(e *EngineOwnerEvidence) {
		e.Reservations.Value.Operations = []EngineOwnerOperation{{ID: "prior", Engine: "fake", Kind: "download", Phase: "unsafe", Retained: true, Unsafe: true}}
		e.Reservations.Value.BlocksApply = Known(true)
		e.Reservations.Value.UnsafeCopiesComplete = false
	})
}

func TestEngineOwnerEvidenceWireRejectsEncodingOverflow(t *testing.T) {
	e := engineOwnerWireFixture(t)
	for i := 0; i < MaxStatusBytes/64; i++ {
		id := fmt.Sprintf("retained-%06d", i)
		e.Intent.Value.EnabledByEngine[id] = true
		e.Locks.Value.LockedByEngine[id] = false
		entries := append(*e.Universe.Value, EngineOwnerUniverseEntry{Engine: id})
		e.Universe = Known(entries)
		authority := append(*e.StopAuthority.Value, EngineOwnerStopAuthority{Engine: id, Reason: "unobserved"})
		e.StopAuthority = Known(authority)
	}
	if err := e.Finalize(); err != nil {
		t.Fatal(err)
	}
	if _, err := json.Marshal(e); err == nil {
		t.Fatal("encoded oversized owner evidence")
	}
}

func TestEngineOwnerEvidenceKnownEmptyUniverse(t *testing.T) {
	e := engineOwnerWireFixture(t)
	e.Intent.Value.EnabledByEngine, e.Intent.Value.OffByEngine = map[string]bool{}, map[string]string{}
	e.Locks.Value.LockedByEngine = map[string]bool{}
	e.Universe, e.StopAuthority = Known([]EngineOwnerUniverseEntry{}), Known([]EngineOwnerStopAuthority{})
	if err := e.Finalize(); err != nil {
		t.Fatal(err)
	}
	if !e.Complete {
		t.Fatal("known empty universe became incomplete")
	}
}

func TestEngineOwnerImpactEnvelopeWire(t *testing.T) {
	t.Run("known private manager evidence", func(t *testing.T) {
		value := ImpactEnvelope{ManagerEvidence: Known(engineOwnerWireFixture(t))}
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		var got ImpactEnvelope
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(value, got) {
			t.Fatalf("envelope roundtrip: %+v", got)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			t.Fatal(err)
		}
		if len(fields) != 2 || fields["impact"] == nil || fields["managerEvidence"] == nil {
			t.Fatalf("unexpected envelope fields: %v", fields)
		}
	})
	t.Run("unknown manager is explicit", func(t *testing.T) {
		data, err := json.Marshal(ImpactEnvelope{})
		if err != nil {
			t.Fatal(err)
		}
		var got ImpactEnvelope
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatal(err)
		}
		if got.ManagerEvidence.Known || got.ManagerEvidence.Value != nil {
			t.Fatal("unknown manager became known")
		}
	})
	test := func(name, input string) {
		t.Run(name, func(t *testing.T) {
			var got ImpactEnvelope
			if err := json.Unmarshal([]byte(input), &got); err == nil {
				t.Fatal("accepted invalid private envelope")
			}
		})
	}
	test("missing evidence", `{"impact":{}}`)
	test("null evidence", `{"impact":{},"managerEvidence":null}`)
	test("extra roots", `{"impact":{},"managerEvidence":{"known":false,"value":null},"roots":[]}`)
	test("case variant", `{"impact":{},"ManagerEvidence":{"known":false,"value":null}}`)
	test("duplicate evidence", `{"impact":{},"managerEvidence":{"known":false,"value":null},"managerEvidence":{"known":false,"value":null}}`)
	test("overflow", `{}`+strings.Repeat(" ", MaxStatusBytes))
}
