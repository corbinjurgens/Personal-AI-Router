// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modeldownload

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func modelDownloadSnapshot() Snapshot {
	percent := 21
	return Snapshot{
		Version:  Version,
		SourceID: "0123456789abcdef0123456789abcdef",
		Revision: "7",
		Entries: []Entry{{
			ID:              "abcdef0123456789abcdef0123456789",
			Engine:          "llamacpp-custom",
			Model:           "org/model:Q4_K_M",
			Phase:           PhaseInProgress,
			BlocksReadiness: true,
			Percent:         &percent,
		}},
	}
}

func modelDownloadJSON(t *testing.T, snapshot Snapshot) []byte {
	t.Helper()
	data, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatalf("marshal snapshot: %v", err)
	}
	return data
}

func TestModelDownloadRoundTrip(t *testing.T) {
	test := func(name string, mutate func(*Snapshot)) {
		t.Run(name, func(t *testing.T) {
			want := modelDownloadSnapshot()
			mutate(&want)
			if err := want.Validate(); err != nil {
				t.Fatalf("valid snapshot rejected: %v", err)
			}
			got, err := Parse(modelDownloadJSON(t, want))
			if err != nil {
				t.Fatalf("parse snapshot: %v", err)
			}
			if !got.Equal(want) {
				t.Fatalf("round trip = %+v, want %+v", got, want)
			}
		})
	}
	test("measured progress", func(s *Snapshot) {})
	test("measured full percent remains in progress", func(s *Snapshot) { *s.Entries[0].Percent = 100 })
	test("queued indeterminate", func(s *Snapshot) {
		s.Entries[0].Phase, s.Entries[0].Percent = PhaseQueued, nil
	})
	test("complete without readiness block", func(s *Snapshot) {
		s.Entries[0].Phase, s.Entries[0].BlocksReadiness = PhaseComplete, false
	})
	test("pre-effect failure preserves ready content", func(s *Snapshot) {
		s.Entries[0].Phase, s.Entries[0].BlocksReadiness = PhaseFailed, false
		s.Entries[0].Reason = ReasonBusy
	})
	test("failed unsafe content", func(s *Snapshot) {
		s.Entries[0].Phase, s.Entries[0].Reason = PhaseFailed, ReasonDownloadFailed
	})
	test("recovered unknown content", func(s *Snapshot) {
		s.Entries[0].Phase, s.Entries[0].Reason = PhaseUnknown, ReasonInterrupted
	})
	test("explicit known empty", func(s *Snapshot) { s.Entries = []Entry{} })
	test("recovery required", func(s *Snapshot) {
		s.Entries, s.RecoveryRequired = []Entry{}, true
	})
	test("initial revision", func(s *Snapshot) { s.Revision = "0" })
	test("maximum revision", func(s *Snapshot) { s.Revision = "18446744073709551615" })
	test("exact model spelling", func(s *Snapshot) { s.Entries[0].Model = " 日本語/模型:latest " })
	test("model byte limit", func(s *Snapshot) {
		s.Entries[0].Model = strings.Repeat("é", nodepolicy.MaxModelIdentifierBytes/2)
	})
	test("engine byte limit", func(s *Snapshot) {
		s.Entries[0].Engine = strings.Repeat("e", engines.MaxDynamicIDLen)
	})
}

func TestModelDownloadValidateRejectsMalformedFields(t *testing.T) {
	test := func(name string, mutate func(*Snapshot)) {
		t.Run(name, func(t *testing.T) {
			s := modelDownloadSnapshot()
			mutate(&s)
			if err := s.Validate(); err == nil {
				t.Fatal("invalid snapshot accepted")
			}
			if _, err := Parse(modelDownloadJSON(t, s)); err == nil {
				t.Fatal("invalid wire snapshot accepted")
			}
		})
	}
	test("unsupported version", func(s *Snapshot) { s.Version++ })
	test("missing source", func(s *Snapshot) { s.SourceID = "" })
	test("uppercase source", func(s *Snapshot) { s.SourceID = strings.Repeat("A", 32) })
	test("nonhex source", func(s *Snapshot) { s.SourceID = strings.Repeat("g", 32) })
	test("short source", func(s *Snapshot) { s.SourceID = strings.Repeat("a", 31) })
	test("long source", func(s *Snapshot) { s.SourceID = strings.Repeat("a", 33) })
	test("missing revision", func(s *Snapshot) { s.Revision = "" })
	test("revision overflow", func(s *Snapshot) { s.Revision = "18446744073709551616" })
	test("revision leading zero", func(s *Snapshot) { s.Revision = "07" })
	test("revision plus sign", func(s *Snapshot) { s.Revision = "+7" })
	test("negative revision", func(s *Snapshot) { s.Revision = "-7" })
	test("revision whitespace", func(s *Snapshot) { s.Revision = " 7" })
	test("fractional revision", func(s *Snapshot) { s.Revision = "7.0" })
	test("nil entries", func(s *Snapshot) { s.Entries = nil })
	test("missing ID", func(s *Snapshot) { s.Entries[0].ID = "" })
	test("uppercase ID", func(s *Snapshot) { s.Entries[0].ID = strings.Repeat("F", 32) })
	test("nonhex ID", func(s *Snapshot) { s.Entries[0].ID = strings.Repeat("z", 32) })
	test("missing engine", func(s *Snapshot) { s.Entries[0].Engine = "" })
	test("engine path", func(s *Snapshot) { s.Entries[0].Engine = "../engine" })
	test("engine dot", func(s *Snapshot) { s.Entries[0].Engine = "." })
	test("engine double dot", func(s *Snapshot) { s.Entries[0].Engine = ".." })
	test("engine over limit", func(s *Snapshot) {
		s.Entries[0].Engine = strings.Repeat("e", engines.MaxDynamicIDLen+1)
	})
	test("missing model", func(s *Snapshot) { s.Entries[0].Model = "" })
	test("model over limit", func(s *Snapshot) {
		s.Entries[0].Model = strings.Repeat("m", nodepolicy.MaxModelIdentifierBytes+1)
	})
	test("model C0 control", func(s *Snapshot) { s.Entries[0].Model = "model\n" })
	test("model C1 control", func(s *Snapshot) { s.Entries[0].Model = "model\u0085" })
	test("missing phase", func(s *Snapshot) { s.Entries[0].Phase = "" })
	test("unsupported phase", func(s *Snapshot) { s.Entries[0].Phase = "downloaded" })
	test("queued must block", func(s *Snapshot) {
		s.Entries[0].Phase, s.Entries[0].BlocksReadiness = PhaseQueued, false
	})
	test("in progress must block", func(s *Snapshot) { s.Entries[0].BlocksReadiness = false })
	test("complete cannot block", func(s *Snapshot) { s.Entries[0].Phase = PhaseComplete })
	test("negative percent", func(s *Snapshot) { *s.Entries[0].Percent = -1 })
	test("percent over limit", func(s *Snapshot) { *s.Entries[0].Percent = 101 })
	test("raw reason", func(s *Snapshot) { s.Entries[0].Reason = "/private/vendor/error" })
	test("progress reason", func(s *Snapshot) { s.Entries[0].Reason = ReasonDownloadFailed })
	test("complete reason", func(s *Snapshot) {
		s.Entries[0].Phase, s.Entries[0].BlocksReadiness = PhaseComplete, false
		s.Entries[0].Reason = ReasonDownloadFailed
	})
	test("unsupported persistence", func(s *Snapshot) { s.Entries[0].Persistence = "durable" })
	test("progress persistence", func(s *Snapshot) { s.Entries[0].Persistence = PersistenceUncertain })
	test("queued persistence", func(s *Snapshot) {
		s.Entries[0].Phase, s.Entries[0].Persistence = PhaseQueued, PersistenceUncertain
	})
}

func TestModelDownloadRejectsInvalidTypedUTF8(t *testing.T) {
	s := modelDownloadSnapshot()
	s.Entries[0].Model = string([]byte{0xff})
	if err := s.Validate(); err == nil {
		t.Fatal("invalid typed model UTF-8 accepted")
	}
}

func TestModelDownloadFixedReasons(t *testing.T) {
	for _, reason := range []string{
		"busy", "capacity", "download_failed", "interrupted", "progress_timeout",
		"acceptance_unknown", "cleanup_unknown", "persistence_error", "unsupported",
	} {
		t.Run(reason, func(t *testing.T) {
			s := modelDownloadSnapshot()
			s.Entries[0].Phase, s.Entries[0].Reason = PhaseFailed, reason
			if _, err := Parse(modelDownloadJSON(t, s)); err != nil {
				t.Fatalf("fixed reason rejected: %v", err)
			}
		})
	}
}

func TestModelDownloadPersistenceQualifiesOnlyOutcomes(t *testing.T) {
	for _, phase := range []string{PhaseFailed, PhaseComplete, PhaseUnknown} {
		t.Run(phase, func(t *testing.T) {
			s := modelDownloadSnapshot()
			s.Entries[0].Phase, s.Entries[0].BlocksReadiness = phase, false
			s.Entries[0].Persistence = PersistenceUncertain
			got, err := Parse(modelDownloadJSON(t, s))
			if err != nil {
				t.Fatalf("qualified outcome rejected: %v", err)
			}
			if got.Entries[0].Persistence != PersistenceUncertain {
				t.Fatal("persistence qualification lost")
			}
		})
	}
}

func TestModelDownloadSnapshotCapacity(t *testing.T) {
	makeSnapshot := func(unsafe, recent int) Snapshot {
		s := modelDownloadSnapshot()
		s.Entries = []Entry{}
		for i := range unsafe + recent {
			e := modelDownloadSnapshot().Entries[0]
			e.ID, e.Model = fmt.Sprintf("%032x", i), fmt.Sprintf("model-%d", i)
			if i >= unsafe {
				e.Phase, e.BlocksReadiness = PhaseComplete, false
			}
			s.Entries = append(s.Entries, e)
		}
		return s
	}
	t.Run("full bounded snapshot", func(t *testing.T) {
		s := makeSnapshot(MaxUnsafeEntries, MaxRecentEntries)
		if len(s.Entries) != MaxEntries {
			t.Fatal("capacity fixture does not reach full snapshot bound")
		}
		if _, err := Parse(modelDownloadJSON(t, s)); err != nil {
			t.Fatalf("full snapshot rejected: %v", err)
		}
	})
	test := func(name string, unsafe, recent int) {
		t.Run(name, func(t *testing.T) {
			if _, err := Parse(modelDownloadJSON(t, makeSnapshot(unsafe, recent))); err == nil {
				t.Fatal("snapshot over capacity accepted")
			}
		})
	}
	test("unsafe slots exceeded", MaxUnsafeEntries+1, 0)
	test("recent outcomes exceeded", 0, MaxRecentEntries+1)
	test("total entries exceeded", MaxUnsafeEntries, MaxRecentEntries+1)
}

func TestModelDownloadDuplicateIdentityAndKey(t *testing.T) {
	test := func(name string, mutate func(*Entry), wantValid bool) {
		t.Run(name, func(t *testing.T) {
			s := modelDownloadSnapshot()
			s.Entries[0].Engine, s.Entries[0].Model = "ollama", "demo"
			second := s.Entries[0]
			mutate(&second)
			s.Entries = append(s.Entries, second)
			_, err := Parse(modelDownloadJSON(t, s))
			if (err == nil) != wantValid {
				t.Fatalf("Parse error = %v, want valid %v", err, wantValid)
			}
		})
	}
	test("duplicate ID across keys", func(e *Entry) { e.Model = "other" }, false)
	test("duplicate exact key across IDs", func(e *Entry) { e.ID = strings.Repeat("1", 32) }, false)
	test("same model different engine", func(e *Entry) {
		e.ID, e.Engine = strings.Repeat("1", 32), "llamacpp"
	}, true)
	test("naming belongs to owner", func(e *Entry) {
		e.ID, e.Model = strings.Repeat("1", 32), "demo:latest"
	}, true)
}

func TestModelDownloadDecodeRequiresFieldsAndTypes(t *testing.T) {
	s := modelDownloadSnapshot()
	base := string(modelDownloadJSON(t, s))
	entries, err := json.Marshal(s.Entries)
	if err != nil {
		t.Fatalf("marshal entry fixture: %v", err)
	}
	test := func(name, from, to string) {
		t.Run(name, func(t *testing.T) {
			if !strings.Contains(base, from) {
				t.Fatal("replacement fixture does not match snapshot")
			}
			if _, err := Parse([]byte(strings.Replace(base, from, to, 1))); err == nil {
				t.Fatal("malformed wire value accepted")
			}
		})
	}
	test("missing readiness boolean", `,"blocksReadiness":true`, "")
	test("null readiness boolean", `"blocksReadiness":true`, `"blocksReadiness":null`)
	test("string readiness boolean", `"blocksReadiness":true`, `"blocksReadiness":"true"`)
	test("missing recovery boolean", `"recoveryRequired":false,`, "")
	test("null recovery boolean", `"recoveryRequired":false`, `"recoveryRequired":null`)
	test("numeric revision", `"revision":"7"`, `"revision":7`)
	test("missing entries", `,"entries":`+string(entries), "")
	test("null entries", `"entries":`+string(entries), `"entries":null`)
	test("fractional percent", `"percent":21`, `"percent":21.5`)
	test("null percent", `"percent":21`, `"percent":null`)
	test("null reason", `"percent":21`, `"reason":null`)
	test("null persistence", `"percent":21`, `"persistence":null`)
	test("unknown field", `"percent":21`, `"vendorError":"private sentinel"`)
	test("duplicate field", `"blocksReadiness":true`, `"blocksReadiness":false,"blocksReadiness":true`)
	test("noncanonical key", `"blocksReadiness":true`, `"BlocksReadiness":true`)
	test("raw invalid UTF-8", `org/model:Q4_K_M`, string([]byte{0xff}))
}

func TestModelDownloadFalseReadinessAndZeroPercentSurviveJSON(t *testing.T) {
	s := modelDownloadSnapshot()
	e := &s.Entries[0]
	e.Phase, e.BlocksReadiness, *e.Percent = PhaseComplete, false, 0
	data := modelDownloadJSON(t, s)
	var fields struct {
		Entries []struct {
			BlocksReadiness *bool `json:"blocksReadiness"`
			Percent         *int  `json:"percent"`
		} `json:"entries"`
	}
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatalf("decode encoded fields: %v", err)
	}
	if len(fields.Entries) != 1 || fields.Entries[0].BlocksReadiness == nil || *fields.Entries[0].BlocksReadiness ||
		fields.Entries[0].Percent == nil || *fields.Entries[0].Percent != 0 {
		t.Fatalf("explicit false or measured zero omitted: %+v", fields)
	}
	got, err := Parse(data)
	if err != nil || !got.Equal(s) {
		t.Fatalf("round trip lost explicit values: %+v, %v", got, err)
	}
}

func TestModelDownloadParseBoundsAndMalformedInput(t *testing.T) {
	data := modelDownloadJSON(t, modelDownloadSnapshot())
	atLimit := append(bytes.Clone(data), bytes.Repeat([]byte(" "), MaxSnapshotBytes-len(data))...)
	if _, err := Parse(atLimit); err != nil {
		t.Fatalf("snapshot at byte limit rejected: %v", err)
	}
	for name, raw := range map[string][]byte{
		"over byte limit": append(bytes.Clone(atLimit), ' '),
		"trailing value":  append(bytes.Clone(data), []byte(` {}`)...),
		"null snapshot":   []byte(`null`),
		"array snapshot":  []byte(`[]`),
		"truncated":       data[:len(data)-1],
		"empty":           nil,
	} {
		t.Run(name, func(t *testing.T) {
			got, err := Parse(raw)
			if err == nil || !got.Equal(Snapshot{}) {
				t.Fatalf("malformed input returned usable snapshot: %+v, %v", got, err)
			}
		})
	}
}

func TestModelDownloadNestedDecodeRejectsMissingFalse(t *testing.T) {
	s := modelDownloadSnapshot()
	s.Entries[0].Phase, s.Entries[0].BlocksReadiness = PhaseComplete, false
	data := modelDownloadJSON(t, s)
	data = bytes.Replace(data, []byte(`,"blocksReadiness":false`), nil, 1)
	var envelope struct {
		DownloadState *Snapshot `json:"downloadState"`
	}
	raw := append([]byte(`{"downloadState":`), data...)
	raw = append(raw, '}')
	if err := json.Unmarshal(raw, &envelope); err == nil {
		t.Fatal("nested decode accepted missing false readiness boolean")
	}
}

func TestModelDownloadDiagnosticsDoNotEchoInput(t *testing.T) {
	sentinel := "private-vendor-output-sentinel"
	s := modelDownloadSnapshot()
	s.Entries[0].Reason = sentinel
	if err := s.Validate(); err == nil || strings.Contains(err.Error(), sentinel) {
		t.Fatalf("unsafe validation diagnostic: %v", err)
	}
	raw := strings.Replace(string(modelDownloadJSON(t, modelDownloadSnapshot())), `"percent":21`, `"`+sentinel+`":"`+sentinel+`"`, 1)
	var snapshot Snapshot
	if err := json.Unmarshal([]byte(raw), &snapshot); err == nil || strings.Contains(err.Error(), sentinel) {
		t.Fatalf("unsafe nested decoder diagnostic: %v", err)
	}
	if _, err := Parse([]byte(raw)); err == nil || strings.Contains(err.Error(), sentinel) {
		t.Fatalf("unsafe parse diagnostic: %v", err)
	}
}

func TestModelDownloadCloneOwnsMutableState(t *testing.T) {
	original := modelDownloadSnapshot()
	clone := original.Clone()
	if !clone.Equal(original) {
		t.Fatal("clone changed snapshot")
	}
	clone.Entries[0].Model = "other"
	*clone.Entries[0].Percent = 80
	if original.Entries[0].Model != "org/model:Q4_K_M" || *original.Entries[0].Percent != 21 {
		t.Fatal("clone aliases original entries or percent")
	}
	for _, entries := range [][]Entry{nil, {}} {
		s := modelDownloadSnapshot()
		s.Entries = entries
		if !s.Clone().Equal(s) {
			t.Fatal("clone changed absent versus empty entries")
		}
	}
}

func TestModelDownloadEqualObservesEveryField(t *testing.T) {
	test := func(name string, mutate func(*Snapshot)) {
		t.Run(name, func(t *testing.T) {
			s := modelDownloadSnapshot()
			other := s.Clone()
			mutate(&other)
			if s.Equal(other) || other.Equal(s) {
				t.Fatal("changed observation compares equal")
			}
		})
	}
	test("version", func(s *Snapshot) { s.Version++ })
	test("source", func(s *Snapshot) { s.SourceID = strings.Repeat("2", 32) })
	test("revision", func(s *Snapshot) { s.Revision = "8" })
	test("recovery", func(s *Snapshot) { s.RecoveryRequired = true })
	test("entries", func(s *Snapshot) { s.Entries = []Entry{} })
	test("ID", func(s *Snapshot) { s.Entries[0].ID = strings.Repeat("2", 32) })
	test("engine", func(s *Snapshot) { s.Entries[0].Engine = "ollama" })
	test("model", func(s *Snapshot) { s.Entries[0].Model = "other" })
	test("phase", func(s *Snapshot) { s.Entries[0].Phase = PhaseUnknown })
	test("readiness", func(s *Snapshot) { s.Entries[0].BlocksReadiness = false })
	test("percent", func(s *Snapshot) { *s.Entries[0].Percent = 22 })
	test("unknown percent", func(s *Snapshot) { s.Entries[0].Percent = nil })
	test("reason", func(s *Snapshot) { s.Entries[0].Reason = ReasonInterrupted })
	test("persistence", func(s *Snapshot) { s.Entries[0].Persistence = PersistenceUncertain })
	empty := modelDownloadSnapshot()
	empty.Entries = []Entry{}
	absent := empty.Clone()
	absent.Entries = nil
	if empty.Equal(absent) {
		t.Fatal("absent and known-empty entries compare equal")
	}
	zero := modelDownloadSnapshot()
	*zero.Entries[0].Percent = 0
	unknown := zero.Clone()
	unknown.Entries[0].Percent = nil
	if zero.Equal(unknown) {
		t.Fatal("unknown and measured zero percent compare equal")
	}
}
