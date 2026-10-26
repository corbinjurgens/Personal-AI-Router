// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	settings "nvpair-shared/enginesettings"
)

func switchStoreReceipt(id string) settings.SwitchReceipt {
	now := time.Date(2026, 10, 19, 0, 0, 0, 0, time.UTC)
	revision := "e3v1:" + strings.Repeat("a", 64)
	req := settings.SwitchRequest{NodeID: "node", RequestID: id, SourceEngine: "llamacpp", TargetEngine: "vllm", ExpectedRevision: revision}
	return settings.SwitchReceipt{Version: settings.SwitchVersion, NodeID: "node", RequestID: id, Caller: "local",
		PayloadHash: settings.SwitchPayloadHash("local", "node", req), Request: req,
		ExpectedRevision: revision, AcceptedRevision: revision, Sequence: 1, Phase: settings.SwitchAccepted,
		AcceptedAt: now, UpdatedAt: now, Deadline: now.Add(settings.SwitchOperationBudget),
		Source: settings.SwitchEngineState{Intent: settings.SwitchIntentUnknown}, Target: settings.SwitchEngineState{Intent: settings.SwitchIntentUnknown}}
}

func TestEngineSwitchStoreAutomaticRingPreservesManualAndUnsettled(t *testing.T) {
	s := switchStoreOpen(t)
	manual := switchStoreAccept(t, s, switchStoreReceipt("manual"))
	manual = switchStoreNext(manual, settings.SwitchValidating)
	if err := s.Advance(manual.RequestID, manual); err != nil {
		t.Fatal(err)
	}
	if err := s.Advance(manual.RequestID, switchStoreNext(manual, settings.SwitchFailedValidation)); err != nil {
		t.Fatal(err)
	}
	var unsettled string
	for i := 0; i < settings.MaxAutomaticSwitchReceipts+5; i++ {
		r := switchStoreReceipt(fmt.Sprintf("binding:run:%d", i))
		r.Automatic, r.Deadline = true, r.AcceptedAt.Add(settings.AutomaticSwitchBudget)
		switchStoreAccept(t, s, r)
		r = switchStoreNext(r, settings.SwitchValidating)
		if err := s.Advance(r.RequestID, r); err != nil {
			t.Fatal(err)
		}
		terminal := switchStoreNext(r, settings.SwitchFailedValidation)
		if i == 0 {
			terminal.RecoveryRequired = true
			unsettled = r.RequestID
		}
		if err := s.Advance(r.RequestID, terminal); err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			if err := s.SaveLocalObligation(r.RequestID, settings.SwitchLocalObligation{TargetOff: true}); err != nil {
				t.Fatal(err)
			}
			if err := s.SettleAutomatic(r.RequestID); err == nil {
				t.Fatal("undischarged automatic row settled")
			}
		} else if err := s.SettleAutomatic(r.RequestID); err != nil {
			t.Fatal(err)
		}
	}
	j := switchStoreDisk(t, s)
	if len(j.AutomaticSettled) != settings.MaxAutomaticSwitchReceipts || len(j.Receipts) != settings.MaxAutomaticSwitchReceipts+2 {
		t.Fatalf("retained receipts=%d settled=%d", len(j.Receipts), len(j.AutomaticSettled))
	}
	if _, ok := j.Receipts["manual"]; !ok {
		t.Fatal("manual receipt evicted")
	}
	if r, ok := j.Receipts[unsettled]; !ok || !r.RecoveryRequired {
		t.Fatal("unsettled recovery row evicted")
	}
	if _, ok := j.Receipts["binding:run:1"]; ok {
		t.Fatal("old settled automatic receipt retained")
	}
	if err := s.SaveLocalObligation(unsettled, settings.SwitchLocalObligation{TargetOff: true, Discharged: true}); err != nil {
		t.Fatal(err)
	}
	if err := s.SettleAutomatic(unsettled); err != nil {
		t.Fatal(err)
	}
	j = switchStoreDisk(t, s)
	if j.Receipts[unsettled].RecoveryRequired || j.AutomaticSettled[unsettled] == 0 {
		t.Fatal("cleanup settlement not durable")
	}
	if j.Receipts[unsettled].Sequence <= 3 {
		t.Fatal("recovery settlement did not advance authoritative readback sequence")
	}
}

func TestEngineSwitchStoreAutomaticPruneUncertainWrite(t *testing.T) {
	s := switchStoreOpen(t)
	for i := 0; i <= settings.MaxAutomaticSwitchReceipts; i++ {
		r := switchStoreReceipt(fmt.Sprintf("binding:run:%d", i))
		r.Automatic, r.Deadline = true, r.AcceptedAt.Add(settings.AutomaticSwitchBudget)
		switchStoreAccept(t, s, r)
		r = switchStoreNext(r, settings.SwitchValidating)
		if err := s.Advance(r.RequestID, r); err != nil {
			t.Fatal(err)
		}
		if err := s.Advance(r.RequestID, switchStoreNext(r, settings.SwitchFailedValidation)); err != nil {
			t.Fatal(err)
		}
		if i < settings.MaxAutomaticSwitchReceipts {
			if err := s.SettleAutomatic(r.RequestID); err != nil {
				t.Fatal(err)
			}
			continue
		}
		before := s.Generation()
		s.write = func(path string, data []byte, mode os.FileMode) error {
			if err := durablefile.WriteFile(path, data, mode); err != nil {
				return err
			}
			return durablefile.ErrReplaced
		}
		if err := s.SettleAutomatic(r.RequestID); !errors.Is(err, durablefile.ErrReplaced) {
			t.Fatalf("settlement error=%v", err)
		}
		j := switchStoreDisk(t, s)
		if s.Generation() != before+1 || j.Generation != before+1 || len(j.AutomaticSettled) != settings.MaxAutomaticSwitchReceipts {
			t.Fatalf("uncertain replacement generation=%d rows=%d", j.Generation, len(j.AutomaticSettled))
		}
	}
}

func switchStoreNext(r settings.SwitchReceipt, phase string) settings.SwitchReceipt {
	r = cloneSwitchReceipt(r)
	r.Sequence++
	r.Phase, r.Terminal = phase, settings.IsTerminal(phase)
	r.UpdatedAt = r.UpdatedAt.Add(time.Second)
	if r.Terminal {
		finished := r.UpdatedAt
		r.FinishedAt = &finished
		if phase != settings.SwitchCompleted {
			r.Error = &settings.SwitchError{Code: settings.SwitchOperationErrorCode, Kind: settings.SwitchErrorInterrupted, Message: "test failure"}
		}
	}
	return r
}

func switchStoreOpen(t *testing.T) *engineSwitchStore {
	t.Helper()
	s := newEngineSwitchStore(filepath.Join(t.TempDir(), "private", "engine-switch-operations.json"))
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	return s
}

func switchStoreAccept(t *testing.T, s *engineSwitchStore, r settings.SwitchReceipt) settings.SwitchReceipt {
	t.Helper()
	got, err := s.Accept(r)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func switchStoreKind(t *testing.T, err error, kind string) {
	t.Helper()
	var coded *engineSwitchStoreError
	if !errors.As(err, &coded) || coded.Kind != kind {
		t.Fatalf("error = %v, want kind %s", err, kind)
	}
}

func switchStoreDisk(t *testing.T, s *engineSwitchStore) settings.SwitchJournal {
	t.Helper()
	data, err := os.ReadFile(s.path)
	if err != nil {
		t.Fatal(err)
	}
	j, err := settings.DecodeSwitchJournal(data)
	if err != nil {
		t.Fatal(err)
	}
	return j
}

func TestEngineSwitchStoreDedupe(t *testing.T) {
	s := switchStoreOpen(t)
	r := switchStoreAccept(t, s, switchStoreReceipt("one"))
	advanced := switchStoreNext(r, settings.SwitchValidating)
	if err := s.Advance(r.RequestID, advanced); err != nil {
		t.Fatal(err)
	}
	got, err := s.Accept(r)
	if err != nil || !reflect.DeepEqual(got, advanced) || s.Generation() != 2 {
		t.Fatalf("retry = %+v, %v; generation %d", got, err, s.Generation())
	}
	t.Run("different payload", func(t *testing.T) {
		changed := r
		changed.Request.TargetEngine = "sglang"
		changed.PayloadHash = settings.SwitchPayloadHash(changed.Caller, changed.NodeID, changed.Request)
		_, err := s.Accept(changed)
		switchStoreKind(t, err, settings.SwitchErrorReusedID)
	})
	t.Run("different caller", func(t *testing.T) {
		changed := r
		changed.Caller = "peer"
		changed.PayloadHash = settings.SwitchPayloadHash(changed.Caller, changed.NodeID, changed.Request)
		_, err := s.Accept(changed)
		switchStoreKind(t, err, settings.SwitchErrorReusedID)
	})
	if s.Generation() != 2 {
		t.Fatal("retries changed generation")
	}
}

func TestEngineSwitchStoreBusyAndReject(t *testing.T) {
	s := switchStoreOpen(t)
	r := switchStoreAccept(t, s, switchStoreReceipt("active"))
	_, err := s.Accept(switchStoreReceipt("other"))
	switchStoreKind(t, err, settings.SwitchErrorBusy)
	rejected := switchStoreNext(switchStoreReceipt("refused"), settings.SwitchRejected)
	if _, err := s.Reject(rejected); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Reject(rejected); err != nil {
		t.Fatal(err)
	}
	j := switchStoreDisk(t, s)
	if j.ActiveRequestID != r.RequestID || j.Generation != 2 || len(j.Receipts) != 2 {
		t.Fatalf("journal = %+v", j)
	}
}

func TestEngineSwitchStoreTransitions(t *testing.T) {
	cases := []struct {
		name   string
		change func(*settings.SwitchReceipt)
	}{
		{"skips phase", func(r *settings.SwitchReceipt) { r.Phase = settings.SwitchDraining }},
		{"same phase", func(r *settings.SwitchReceipt) { r.Phase = settings.SwitchAccepted }},
		{"same sequence", func(r *settings.SwitchReceipt) { r.Sequence = 1 }},
		{"backwards time", func(r *settings.SwitchReceipt) { r.UpdatedAt = r.AcceptedAt.Add(-time.Second) }},
		{"changed identity", func(r *settings.SwitchReceipt) { r.AcceptedRevision = "e3v1:" + strings.Repeat("b", 64) }},
		{"invalid observation", func(r *settings.SwitchReceipt) { r.Source.TerminationVerified = true }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := switchStoreOpen(t)
			r := switchStoreAccept(t, s, switchStoreReceipt("active"))
			next := switchStoreNext(r, settings.SwitchValidating)
			tc.change(&next)
			if err := s.Advance(r.RequestID, next); err == nil {
				t.Fatal("invalid transition succeeded")
			}
			if got, _ := s.Lookup(r.RequestID); !reflect.DeepEqual(got, r) || s.Generation() != 1 {
				t.Fatal("invalid transition changed state")
			}
		})
	}
}

func TestEngineSwitchStoreTerminalImmutable(t *testing.T) {
	s := switchStoreOpen(t)
	r := switchStoreAccept(t, s, switchStoreReceipt("active"))
	terminal := switchStoreNext(r, settings.SwitchFailedInterrupted)
	if err := s.Advance(r.RequestID, terminal); err != nil {
		t.Fatal(err)
	}
	if err := s.Advance(r.RequestID, switchStoreNext(terminal, settings.SwitchFailedInterrupted)); err == nil {
		t.Fatal("terminal changed")
	}
	got, err := s.Accept(r)
	if err != nil || !reflect.DeepEqual(got, terminal) {
		t.Fatalf("terminal retry = %+v, %v", got, err)
	}
	if switchStoreDisk(t, s).ActiveRequestID != "" {
		t.Fatal("terminal did not clear active request")
	}
	got.Error.Message = "mutated"
	*got.FinishedAt = time.Time{}
	again, _ := s.Lookup(r.RequestID)
	if !reflect.DeepEqual(again, terminal) {
		t.Fatal("lookup leaked mutable pointers")
	}
	switchStoreAccept(t, s, switchStoreReceipt("new"))
	if s.Generation() != 3 {
		t.Fatalf("generation = %d", s.Generation())
	}
}

func TestEngineSwitchStoreFullCountPreservesProgress(t *testing.T) {
	s := switchStoreOpen(t)
	j := settings.SwitchJournal{Version: settings.SwitchVersion, Generation: 1, Receipts: make(map[string]settings.SwitchReceipt)}
	for i := 0; i < settings.MaxSwitchReceipts-1; i++ {
		r := switchStoreNext(switchStoreReceipt(fmt.Sprintf("old-%d", i)), settings.SwitchRejected)
		j.Receipts[r.RequestID] = r
	}
	data, err := json.Marshal(j)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(s.path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	s = newEngineSwitchStore(s.path)
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	r := switchStoreAccept(t, s, switchStoreReceipt("last"))
	_, err = s.Reject(switchStoreNext(switchStoreReceipt("overflow"), settings.SwitchRejected))
	switchStoreKind(t, err, settings.SwitchErrorFull)
	if _, err := s.Accept(r); err != nil {
		t.Fatal(err)
	}
	r = switchStoreNext(r, settings.SwitchValidating)
	if err := s.Advance(r.RequestID, r); err != nil {
		t.Fatal(err)
	}
	if err := s.InterruptNonterminal(r.UpdatedAt.Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	_, err = s.Accept(switchStoreReceipt("fresh"))
	switchStoreKind(t, err, settings.SwitchErrorFull)
	j = switchStoreDisk(t, s)
	if len(j.Receipts) != settings.MaxSwitchReceipts || j.Receipts["last"].Phase != settings.SwitchFailedInterrupted {
		t.Fatal("full journal lost receipts or progress")
	}
}

func TestEngineSwitchStoreByteReservation(t *testing.T) {
	s := switchStoreOpen(t)
	r := switchStoreReceipt("large")
	r.Caller = strings.Repeat("x", settings.MaxSwitchJournalBytes-10000)
	r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	_, err := s.Accept(r)
	switchStoreKind(t, err, settings.SwitchErrorFull)
	if s.Generation() != 0 {
		t.Fatal("full request changed generation")
	}
	// Leave room for acceptance's reservation, then consume the available space
	// with a rejected attempt. Its admission must still protect the active entry.
	r.Caller = strings.Repeat("x", settings.MaxSwitchJournalBytes-32000)
	r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
	r = switchStoreAccept(t, s, r)
	rejected := switchStoreNext(switchStoreReceipt("refusal"), settings.SwitchRejected)
	rejected.Caller = strings.Repeat("y", 12000)
	rejected.PayloadHash = settings.SwitchPayloadHash(rejected.Caller, rejected.NodeID, rejected.Request)
	if _, err := s.Reject(rejected); err != nil {
		t.Fatal(err)
	}
	tooLarge := rejected
	tooLarge.RequestID, tooLarge.Request.RequestID = "overflow", "overflow"
	tooLarge.PayloadHash = settings.SwitchPayloadHash(tooLarge.Caller, tooLarge.NodeID, tooLarge.Request)
	_, err = s.Reject(tooLarge)
	switchStoreKind(t, err, settings.SwitchErrorFull)
	if _, err := s.Accept(r); err != nil {
		t.Fatal(err)
	}
	r = switchStoreNext(r, settings.SwitchValidating)
	if err := s.Advance(r.RequestID, r); err != nil {
		t.Fatal(err)
	}
	r = switchStoreNext(r, settings.SwitchFailedInterrupted)
	r.Error.Message = strings.Repeat("\x00", settings.MaxSwitchErrorMessageBytes)
	if err := s.Advance(r.RequestID, r); err != nil {
		t.Fatal(err)
	}
	if got := switchStoreDisk(t, s).Receipts[r.RequestID]; !reflect.DeepEqual(got, r) {
		t.Fatal("reserved terminal state was not persisted")
	}
}

func TestEngineSwitchStoreRestartInterrupt(t *testing.T) {
	for _, phase := range []string{settings.SwitchAccepted, settings.SwitchValidating, settings.SwitchDraining, settings.SwitchSourceStopping, settings.SwitchSourceOff, settings.SwitchTargetStarting, settings.SwitchTargetReady} {
		t.Run(phase, func(t *testing.T) {
			s := switchStoreOpen(t)
			r := switchStoreReceipt("restart")
			r.Phase = phase
			j := settings.SwitchJournal{Version: settings.SwitchVersion, Generation: 9, ActiveRequestID: r.RequestID, Receipts: map[string]settings.SwitchReceipt{r.RequestID: r}}
			data, err := json.Marshal(j)
			if err != nil {
				t.Fatal(err)
			}
			s = newEngineSwitchStore(s.path)
			s.read = func(string) ([]byte, error) { return data, nil }
			if err := s.Load(); err != nil {
				t.Fatal(err)
			}
			if err := s.InterruptNonterminal(r.UpdatedAt.Add(-time.Hour)); err != nil {
				t.Fatal(err)
			}
			got, ok := s.Lookup(r.RequestID)
			if !ok || got.Phase != settings.SwitchFailedInterrupted || !got.Terminal || got.Sequence != 2 || !got.RecoveryRequired || got.Error.Kind != settings.SwitchErrorInterrupted || !reflect.DeepEqual(got.Source, r.Source) {
				t.Fatalf("interrupted = %+v", got)
			}
			if err := s.InterruptNonterminal(time.Now()); err != nil {
				t.Fatal(err)
			}
			if s.Generation() != 10 || switchStoreDisk(t, s).ActiveRequestID != "" {
				t.Fatal("repeat recovery mutated journal")
			}
		})
	}
}

func TestEngineSwitchStoreUntrustedPreservesBytes(t *testing.T) {
	for _, tc := range []struct {
		name    string
		data    []byte
		readErr error
	}{
		{"corrupt", []byte("{broken"), nil},
		{"oversized", bytes.Repeat([]byte(" "), settings.MaxSwitchJournalBytes+1), nil},
		{"invalid", []byte(`{"version":999,"generation":0,"activeRequestId":"","receipts":{}}`), nil},
		{"unreadable", []byte("original"), os.ErrPermission},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newEngineSwitchStore(filepath.Join(t.TempDir(), "journal.json"))
			if err := os.WriteFile(s.path, tc.data, 0o600); err != nil {
				t.Fatal(err)
			}
			if tc.readErr != nil {
				s.read = func(string) ([]byte, error) { return nil, tc.readErr }
			}
			writes := 0
			s.write = func(string, []byte, os.FileMode) error { writes++; return nil }
			var recovery *engineSwitchRecoveryError
			if err := s.Load(); !errors.As(err, &recovery) {
				t.Fatalf("load = %v", err)
			}
			if err := s.Load(); !errors.As(err, &recovery) {
				t.Fatalf("second load = %v", err)
			}
			r := switchStoreReceipt("blocked")
			if _, err := s.Accept(r); !errors.As(err, &recovery) {
				t.Fatalf("accept = %v", err)
			}
			if _, err := s.Reject(switchStoreNext(r, settings.SwitchRejected)); !errors.As(err, &recovery) {
				t.Fatalf("reject = %v", err)
			}
			if err := s.Advance(r.RequestID, r); !errors.As(err, &recovery) {
				t.Fatalf("advance = %v", err)
			}
			if err := s.InterruptNonterminal(time.Now()); !errors.As(err, &recovery) {
				t.Fatalf("interrupt = %v", err)
			}
			data, err := os.ReadFile(s.path)
			if err != nil {
				t.Fatal(err)
			}
			if writes != 0 || !bytes.Equal(data, tc.data) {
				t.Fatal("untrusted journal was overwritten")
			}
		})
	}
}

func TestEngineSwitchStoreWriteFaults(t *testing.T) {
	for _, landed := range []bool{false, true} {
		name := "before replacement"
		if landed {
			name = "after replacement"
		}
		t.Run(name, func(t *testing.T) {
			s := switchStoreOpen(t)
			r := switchStoreAccept(t, s, switchStoreReceipt("fault"))
			before, err := os.ReadFile(s.path)
			if err != nil {
				t.Fatal(err)
			}
			fault := errors.New("injected storage failure")
			s.write = func(path string, data []byte, mode os.FileMode) error {
				if landed {
					if err := durablefile.WriteFile(path, data, mode); err != nil {
						return err
					}
					return errors.Join(durablefile.ErrReplaced, fault)
				}
				return fault
			}
			next := switchStoreNext(r, settings.SwitchValidating)
			err = s.Advance(r.RequestID, next)
			if !errors.Is(err, fault) || errors.Is(err, durablefile.ErrReplaced) != landed {
				t.Fatalf("write error = %v", err)
			}
			got, _ := s.Lookup(r.RequestID)
			want, generation := r, uint64(1)
			if landed {
				want, generation = next, 2
			}
			if !reflect.DeepEqual(got, want) || s.Generation() != generation {
				t.Fatalf("memory = %+v; generation %d", got, s.Generation())
			}
			after, err := os.ReadFile(s.path)
			if err != nil {
				t.Fatal(err)
			}
			if !landed && !bytes.Equal(before, after) {
				t.Fatal("pre-replace failure changed bytes")
			}
			if disk := switchStoreDisk(t, s); !reflect.DeepEqual(disk, s.journal) {
				t.Fatal("memory differs from landed journal")
			}
		})
	}
}

func TestEngineSwitchStorePOSIXModes(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX permission bits are not enforced on Windows")
	}
	s := switchStoreOpen(t)
	switchStoreAccept(t, s, switchStoreReceipt("permissions"))
	for path, want := range map[string]os.FileMode{s.path: 0o600, filepath.Dir(s.path): 0o700} {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if got := info.Mode().Perm(); got != want {
			t.Fatalf("%s mode = %o, want %o", path, got, want)
		}
	}
}

func TestEngineSwitchStoreCounterExhaustion(t *testing.T) {
	s := switchStoreOpen(t)
	s.journal.Generation = math.MaxUint64
	if _, err := s.Accept(switchStoreReceipt("overflow")); err == nil {
		t.Fatal("generation wrapped")
	}
	if s.Generation() != math.MaxUint64 || len(s.journal.Receipts) != 0 {
		t.Fatal("failed write changed memory")
	}
}

func TestEngineSwitchStoreMissingFileAndPath(t *testing.T) {
	s := switchStoreOpen(t)
	if s.Generation() != 0 || len(s.journal.Receipts) != 0 || s.journal.ActiveRequestID != "" {
		t.Fatalf("missing file journal = %+v", s.journal)
	}
	if _, err := os.Stat(s.path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("Load created a file: %v", err)
	}
	b := &Broker{}
	settingsPath, err := b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	switchPath, err := b.engineSwitchPath()
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Dir(switchPath) != filepath.Dir(settingsPath) || filepath.Base(switchPath) != "engine-switch-operations.json" {
		t.Fatalf("switch path = %s; settings path = %s", switchPath, settingsPath)
	}
}

func TestEngineSwitchStoreAcceptanceWriteFaults(t *testing.T) {
	for _, tc := range []struct {
		name   string
		landed bool
	}{
		{"pre-replace failure", false},
		{"durability uncertainty", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := switchStoreOpen(t)
			fault := errors.New("acceptance write failure")
			s.write = func(path string, data []byte, mode os.FileMode) error {
				if !tc.landed {
					return fault
				}
				if err := durablefile.WriteFile(path, data, mode); err != nil {
					return err
				}
				return errors.Join(durablefile.ErrReplaced, fault)
			}
			r := switchStoreReceipt("acceptance")
			got, err := s.Accept(r)
			if !errors.Is(err, fault) || errors.Is(err, durablefile.ErrReplaced) != tc.landed {
				t.Fatalf("acceptance error = %v", err)
			}
			stored, exists := s.Lookup(r.RequestID)
			if exists != tc.landed {
				t.Fatalf("receipt present = %v, want %v", exists, tc.landed)
			}
			if !tc.landed {
				if s.Generation() != 0 || got.RequestID != "" {
					t.Fatal("failed acceptance was reported or adopted")
				}
				if _, err := os.Stat(s.path); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("failed acceptance changed disk: %v", err)
				}
				return
			}
			if s.Generation() != 1 || !reflect.DeepEqual(got, r) || !reflect.DeepEqual(stored, r) {
				t.Fatal("landed acceptance was not adopted")
			}
			if disk := switchStoreDisk(t, s); !reflect.DeepEqual(disk, s.journal) {
				t.Fatal("landed acceptance differs from memory")
			}
			writes := 0
			s.write = func(string, []byte, os.FileMode) error { writes++; return fault }
			if _, err := s.Accept(r); err != nil || writes != 0 {
				t.Fatalf("retry wrote again: %v; writes %d", err, writes)
			}
		})
	}
}

func TestEngineSwitchStoreReceiptCopies(t *testing.T) {
	s := switchStoreOpen(t)
	r := switchStoreReceipt("copies")
	running, healthy, observed := true, true, r.AcceptedAt
	r.Source.Running, r.Source.Healthy, r.Source.ObservedAt = &running, &healthy, &observed
	switchStoreAccept(t, s, r)
	running, healthy, observed = false, false, time.Time{}
	got, _ := s.Lookup(r.RequestID)
	if !*got.Source.Running || !*got.Source.Healthy || got.Source.ObservedAt.IsZero() {
		t.Fatal("acceptance retained caller pointers")
	}
	*got.Source.Running, *got.Source.Healthy, *got.Source.ObservedAt = false, false, time.Time{}
	again, _ := s.Lookup(r.RequestID)
	if !*again.Source.Running || !*again.Source.Healthy || again.Source.ObservedAt.IsZero() {
		t.Fatal("lookup leaked store pointers")
	}
}

func TestEngineSwitchStoreCompletedRoundTrip(t *testing.T) {
	s := switchStoreOpen(t)
	r := switchStoreAccept(t, s, switchStoreReceipt("completed"))
	for _, phase := range []string{settings.SwitchValidating, settings.SwitchDraining, settings.SwitchSourceStopping,
		settings.SwitchSourceOff, settings.SwitchTargetStarting, settings.SwitchTargetReady, settings.SwitchCompleted} {
		r = switchStoreNext(r, phase)
		if err := s.Advance(r.RequestID, r); err != nil {
			t.Fatalf("advance to %s: %v", phase, err)
		}
	}
	reopened := newEngineSwitchStore(s.path)
	if err := reopened.Load(); err != nil {
		t.Fatal(err)
	}
	got, ok := reopened.Lookup(r.RequestID)
	if !ok || !reflect.DeepEqual(got, r) || reopened.Generation() != r.Sequence || reopened.journal.ActiveRequestID != "" {
		t.Fatalf("reopened receipt = %+v; generation %d", got, reopened.Generation())
	}
	if err := reopened.InterruptNonterminal(time.Now()); err != nil {
		t.Fatal(err)
	}
	again, _ := reopened.Lookup(r.RequestID)
	if !reflect.DeepEqual(again, r) || reopened.Generation() != r.Sequence {
		t.Fatal("restart rewrote a terminal receipt")
	}
}

func TestEngineSwitchStoreFenceOwnershipAndEviction(t *testing.T) {
	s := switchStoreOpen(t)
	fence := func(caller, id string) settings.SwitchFenceResult {
		t.Helper()
		result, err := s.Fence("node", caller, id)
		if err != nil {
			t.Fatal(err)
		}
		if err := result.Validate(); err != nil {
			t.Fatal(err)
		}
		return result
	}
	if !fence("local", "lost").Fenced || s.Generation() != 1 {
		t.Fatal("fence was not saved")
	}
	if !fence("local", "lost").Fenced || s.Generation() != 1 {
		t.Fatal("duplicate fence wrote")
	}
	_, err := s.Accept(switchStoreReceipt("lost"))
	switchStoreKind(t, err, settings.SwitchErrorFenced)
	changed := switchStoreReceipt("lost")
	changed.Request.TargetEngine = "sglang"
	changed.PayloadHash = settings.SwitchPayloadHash(changed.Caller, changed.NodeID, changed.Request)
	_, err = s.Accept(changed)
	switchStoreKind(t, err, settings.SwitchErrorFenced)
	if !fence("peer", "lost").Fenced || s.Generation() != 2 {
		t.Fatal("pair-scoped fence was not independent")
	}
	foreign := switchStoreReceipt("lost")
	foreign.Caller = "other"
	foreign.PayloadHash = settings.SwitchPayloadHash(foreign.Caller, foreign.NodeID, foreign.Request)
	accepted := switchStoreAccept(t, s, foreign)
	result := fence("other", "lost")
	if result.Fenced || result.Receipt == nil || !reflect.DeepEqual(*result.Receipt, accepted) || s.Generation() != 3 {
		t.Fatal("fence changed accepted receipt")
	}
	result, err = s.Fence("node", "local", "lost")
	if err != nil || result.Receipt != nil || !result.Fenced || s.Generation() != 3 {
		t.Fatalf("retained own proof lost behind foreign receipt: %+v %v", result, err)
	}
	result, err = s.Fence("node", "no-prior-proof", "lost")
	switchStoreKind(t, err, settings.SwitchErrorReusedID)
	if result.Receipt != nil || result.Fenced || s.Generation() != 3 {
		t.Fatal("foreign receipt disclosed or fenced without prior own proof")
	}
	for i := 0; i < settings.MaxSwitchTombstones; i++ {
		fence("local", fmt.Sprintf("new-%d", i))
	}
	if len(s.journal.Tombstones) != settings.MaxSwitchTombstones || s.journal.Tombstones[0].RequestID != "new-0" {
		t.Fatal("oldest insertion not evicted")
	}
	reopened := newEngineSwitchStore(s.path)
	if err := reopened.Load(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(reopened.journal, s.journal) {
		t.Fatal("restart changed journal")
	}
	if _, err := reopened.Fence("node", "local", "new-0"); err != nil {
		t.Fatal(err)
	}
	if reopened.Generation() != s.Generation() {
		t.Fatal("restart duplicate changed generation")
	}
	if _, err := reopened.Fence("node", "local", "new-32"); err != nil {
		t.Fatal(err)
	}
	if reopened.IsFenced("local", "new-0") || reopened.journal.Tombstones[0].RequestID != "new-1" {
		t.Fatal("duplicate retry refreshed insertion order")
	}
}

func TestEngineSwitchStoreFenceForeignReceiptDoesNotBypassStorageLatch(t *testing.T) {
	s := switchStoreOpen(t)
	if _, err := s.Fence("node", "peer", "lost"); err != nil {
		t.Fatal(err)
	}
	accepted := switchStoreAccept(t, s, switchStoreReceipt("lost"))
	s.write = func(path string, data []byte, mode os.FileMode) error {
		if err := durablefile.WriteFile(path, data, mode); err != nil {
			return err
		}
		return durablefile.ErrReplaced
	}
	if _, err := s.Fence("node", "peer", "uncertain"); !errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("uncertain write = %v", err)
	}
	before := s.copyJournal()
	result, err := s.Fence("node", "peer", "lost")
	if !errors.Is(err, durablefile.ErrReplaced) || result.Fenced || result.Receipt != nil {
		t.Fatalf("latched retry = %+v %v", result, err)
	}
	if !reflect.DeepEqual(s.journal, before) {
		t.Fatal("latched retry changed journal")
	}
	result, err = s.Fence("node", "local", "lost")
	if err != nil || result.Fenced || result.Receipt == nil || !reflect.DeepEqual(*result.Receipt, accepted) {
		t.Fatalf("owned receipt behind latch = %+v %v", result, err)
	}
}

func TestEngineSwitchStoreFenceIdentityBounds(t *testing.T) {
	for _, pair := range []struct{ name, caller, id string }{
		{"missing caller", "", "lost"},
		{"caller contains spaces", "bad caller", "lost"},
		{"caller exceeds bound", strings.Repeat("x", 129), "lost"},
		{"missing request ID", "local", ""},
		{"request ID exceeds bound", "local", strings.Repeat("x", 129)},
	} {
		t.Run(pair.name, func(t *testing.T) {
			s := switchStoreOpen(t)
			result, err := s.Fence("node", pair.caller, pair.id)
			switchStoreKind(t, err, settings.SwitchErrorInvalidRequest)
			if result.Fenced || s.Generation() != 0 {
				t.Fatal("invalid identity changed journal")
			}
		})
	}
}

func TestEngineSwitchStoreFenceWriteFaultRetry(t *testing.T) {
	for _, replaced := range []bool{false, true} {
		name := "before replacement"
		if replaced {
			name = "uncertain replacement"
		}
		t.Run(name, func(t *testing.T) {
			s := switchStoreOpen(t)
			fault := errors.New("fence write failed")
			writes := 0
			s.write = func(path string, data []byte, mode os.FileMode) error {
				writes++
				if replaced {
					if err := durablefile.WriteFile(path, data, mode); err != nil {
						return err
					}
					return errors.Join(durablefile.ErrReplaced, fault)
				}
				return fault
			}
			for attempt := 0; attempt < 2; attempt++ {
				result, err := s.Fence("node", "local", "lost")
				if !errors.Is(err, fault) || result.Fenced || result.Receipt != nil {
					t.Fatalf("false acknowledgement: %+v, %v", result, err)
				}
			}
			if writes != 1 {
				t.Fatalf("latched failure retried storage: %d", writes)
			}
			if replaced {
				reopened := newEngineSwitchStore(s.path)
				if err := reopened.Load(); err != nil {
					t.Fatal(err)
				}
				result, err := reopened.Fence("node", "local", "lost")
				if err != nil || !result.Fenced || reopened.Generation() != 1 {
					t.Fatalf("trusted restart: %+v %v", result, err)
				}
			}
		})
	}
}

func TestEngineSwitchStoreFenceCapacityReservesActiveTerminal(t *testing.T) {
	s := switchStoreOpen(t)
	r := switchStoreReceipt("active")
	r.Caller = strings.Repeat("x", settings.MaxSwitchJournalBytes-32000)
	r.PayloadHash = settings.SwitchPayloadHash(r.Caller, r.NodeID, r.Request)
	r = switchStoreAccept(t, s, r)
	// Fill only the unreserved bytes; retained receipt identity is deliberately
	// large, as in the existing byte-reservation regression.
	refused := switchStoreNext(switchStoreReceipt("large-refusal"), settings.SwitchRejected)
	refused.Caller = strings.Repeat("y", 12000)
	refused.PayloadHash = settings.SwitchPayloadHash(refused.Caller, refused.NodeID, refused.Request)
	if _, err := s.Reject(refused); err != nil {
		t.Fatal(err)
	}
	for i := 0; ; i++ {
		before := s.Generation()
		result, err := s.Fence("node", strings.Repeat("z", 128), fmt.Sprintf("%0128d", i))
		if err != nil {
			switchStoreKind(t, err, settings.SwitchErrorFull)
			if result.Fenced || s.Generation() != before {
				t.Fatal("full fence acknowledged or changed journal")
			}
			break
		}
		if i >= settings.MaxSwitchTombstones {
			t.Fatal("fixture did not fill byte capacity")
		}
	}
	r = switchStoreNext(r, settings.SwitchFailedInterrupted)
	r.Error.Message = strings.Repeat("\x00", settings.MaxSwitchErrorMessageBytes)
	if err := s.Advance(r.RequestID, r); err != nil {
		t.Fatal(err)
	}
	if got := switchStoreDisk(t, s).Receipts[r.RequestID]; !reflect.DeepEqual(got, r) {
		t.Fatal("terminal reservation consumed by fences")
	}
}
