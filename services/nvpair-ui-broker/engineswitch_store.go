// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"time"

	"nvpair-shared/durablefile"
	settings "nvpair-shared/enginesettings"
)

// engineSwitchRecoveryError means the journal cannot be trusted. The coordinator
// must fence recovery and mutations; Load never repairs or overwrites its bytes.
type engineSwitchRecoveryError struct{ Err error }

func (e *engineSwitchRecoveryError) Error() string {
	return "engine switch journal requires recovery: " + e.Err.Error()
}
func (e *engineSwitchRecoveryError) Unwrap() error { return e.Err }

type engineSwitchStoreError struct {
	Kind string
	Err  error
}

func (e *engineSwitchStoreError) Error() string {
	return "engine switch store " + e.Kind + ": " + e.Err.Error()
}
func (e *engineSwitchStoreError) Unwrap() error { return e.Err }

func switchStoreError(kind, message string) error {
	return &engineSwitchStoreError{Kind: kind, Err: errors.New(message)}
}

// engineSwitchStore has no internal mutex. The coordinator serializes ALL calls
// under engineConfigMu. Returned receipts are detached copies. It must stop side
// effects on any persistence error, including errors.Is(err, durablefile.ErrReplaced):
// that flag means the new state landed, but its durability is uncertain.
type engineSwitchStore struct {
	path      string
	read      func(string) ([]byte, error)
	write     func(string, []byte, os.FileMode) error
	journal   settings.SwitchJournal
	loaded    bool
	untrusted *engineSwitchRecoveryError
	fenceErr  error // latched fence-write failure; adopted uncertainty is never an ACK
}

// Reuse the settings journal's directory resolution without changing its owner.
func (b *Broker) engineSwitchPath() (string, error) {
	path, err := b.engineSettingsPath()
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Dir(path), "engine-switch-operations.json"), nil
}

func newEngineSwitchStore(path string) *engineSwitchStore {
	return &engineSwitchStore{path: path, read: readEngineSwitchJournal, write: durablefile.WriteFile}
}

// Unlike os.ReadFile, this caps allocation even if the file grows during the read.
// Injected readers are also checked before decode by Load.
func readEngineSwitchJournal(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	data, readErr := io.ReadAll(io.LimitReader(f, settings.MaxSwitchJournalBytes+1))
	closeErr := f.Close()
	if readErr != nil {
		return nil, readErr
	}
	if closeErr != nil {
		return nil, closeErr
	}
	return data, nil
}

func (s *engineSwitchStore) Load() error {
	if s.loaded {
		return s.trustError()
	}
	s.loaded = true
	data, err := s.read(s.path)
	if errors.Is(err, os.ErrNotExist) {
		s.journal = settings.SwitchJournal{Version: settings.SwitchVersion, Receipts: make(map[string]settings.SwitchReceipt)}
		return nil
	}
	if err == nil {
		s.journal, err = settings.DecodeSwitchJournal(data)
	}
	if err != nil {
		s.untrusted = &engineSwitchRecoveryError{Err: err}
	}
	return s.trustError()
}

func (s *engineSwitchStore) trustError() error {
	if s.untrusted != nil {
		return s.untrusted
	}
	if !s.loaded {
		return switchStoreError(settings.SwitchErrorStorage, "journal has not been loaded")
	}
	return nil
}

func (s *engineSwitchStore) Generation() uint64 { return s.journal.Generation }

func (s *engineSwitchStore) IsFenced(caller, id string) bool {
	for _, tombstone := range s.journal.Tombstones {
		if tombstone.Caller == caller && tombstone.RequestID == id {
			return true
		}
	}
	return false
}

// Fence is serialized with both admission lookups by engineConfigMu. Receipts
// win over absence fencing and retain their global ID ownership rule. The
// bounded tombstones reserve only (caller, ID), never another caller's namespace.
func (s *engineSwitchStore) Fence(nodeID, caller, id string) (settings.SwitchFenceResult, error) {
	if err := s.trustError(); err != nil {
		return settings.SwitchFenceResult{}, err
	}
	tombstone := settings.SwitchTombstone{Caller: caller, RequestID: id}
	if err := tombstone.Validate(); err != nil {
		return settings.SwitchFenceResult{}, switchStoreError(settings.SwitchErrorInvalidRequest, err.Error())
	}
	result := settings.SwitchFenceResult{NodeID: nodeID, RequestID: id}
	if nodeID == "" {
		return settings.SwitchFenceResult{}, switchStoreError(settings.SwitchErrorInvalidRequest, "node identity is required")
	}
	if receipt, ok := s.Lookup(id); ok {
		if receipt.Caller != caller {
			return settings.SwitchFenceResult{}, switchStoreError(settings.SwitchErrorReusedID, "request ID belongs to another caller")
		}
		result.Receipt = &receipt
		if err := result.Validate(); err != nil {
			return settings.SwitchFenceResult{}, err
		}
		return result, nil
	}
	if s.fenceErr != nil {
		return settings.SwitchFenceResult{}, s.fenceErr
	}
	if !s.IsFenced(caller, id) {
		next := s.copyJournal()
		if len(next.Tombstones) == settings.MaxSwitchTombstones {
			next.Tombstones = next.Tombstones[1:]
		}
		next.Tombstones = append(next.Tombstones, tombstone)
		if err := switchStoreCapacity(next); err != nil {
			return settings.SwitchFenceResult{}, err
		}
		if err := s.persist(next); err != nil {
			s.fenceErr = err
			return settings.SwitchFenceResult{}, err
		}
	}
	result.Fenced = true
	return result, nil
}

func (s *engineSwitchStore) Lookup(id string) (settings.SwitchReceipt, bool) {
	if !s.loaded || s.untrusted != nil {
		return settings.SwitchReceipt{}, false
	}
	r, ok := s.journal.Receipts[id]
	return cloneSwitchReceipt(r), ok
}

func (s *engineSwitchStore) duplicate(r settings.SwitchReceipt) (settings.SwitchReceipt, bool, error) {
	old, ok := s.Lookup(r.RequestID)
	if !ok {
		return old, false, nil
	}
	if old.Caller != r.Caller || old.PayloadHash != r.PayloadHash {
		return settings.SwitchReceipt{}, true, switchStoreError(settings.SwitchErrorReusedID, "request ID belongs to another caller or payload")
	}
	return old, true, nil
}

func (s *engineSwitchStore) Accept(r settings.SwitchReceipt) (settings.SwitchReceipt, error) {
	return s.insert(r, false)
}

// Reject stores an authorized refused attempt without replacing an active switch.
func (s *engineSwitchStore) Reject(r settings.SwitchReceipt) (settings.SwitchReceipt, error) {
	return s.insert(r, true)
}

func (s *engineSwitchStore) insert(r settings.SwitchReceipt, rejected bool) (settings.SwitchReceipt, error) {
	if err := s.trustError(); err != nil {
		return settings.SwitchReceipt{}, err
	}
	if old, exists, err := s.duplicate(r); exists {
		return old, err
	}
	if s.fenceErr != nil {
		return settings.SwitchReceipt{}, s.fenceErr
	}
	if s.IsFenced(r.Caller, r.RequestID) {
		return settings.SwitchReceipt{}, switchStoreError(settings.SwitchErrorFenced, "request ID was fenced by this caller")
	}
	phase := settings.SwitchAccepted
	if rejected {
		phase = settings.SwitchRejected
	}
	if r.Phase != phase {
		return settings.SwitchReceipt{}, switchStoreError(settings.SwitchErrorInvalidRequest, "invalid initial phase")
	}
	if err := r.Validate(); err != nil {
		return settings.SwitchReceipt{}, err
	}
	if !rejected && s.journal.ActiveRequestID != "" {
		return settings.SwitchReceipt{}, switchStoreError(settings.SwitchErrorBusy, "another switch is active")
	}
	if len(s.journal.Receipts) >= settings.MaxSwitchReceipts {
		return settings.SwitchReceipt{}, switchStoreError(settings.SwitchErrorFull, "receipt capacity exhausted")
	}
	next := s.copyJournal()
	next.Receipts[r.RequestID] = cloneSwitchReceipt(r)
	if !rejected {
		next.ActiveRequestID = r.RequestID
	}
	if err := switchStoreCapacity(next); err != nil {
		return settings.SwitchReceipt{}, err
	}
	err := s.persist(next)
	landed, _ := s.Lookup(r.RequestID)
	return landed, err
}

func (s *engineSwitchStore) Advance(id string, r settings.SwitchReceipt) error {
	if err := s.trustError(); err != nil {
		return err
	}
	old, ok := s.Lookup(id)
	if !ok {
		return switchStoreError(settings.SwitchErrorUnknownReceipt, "unknown request ID")
	}
	if !settings.ValidTransition(old.Phase, r.Phase) || r.Sequence <= old.Sequence || r.UpdatedAt.Before(old.UpdatedAt) {
		return switchStoreError(settings.SwitchErrorInvalidRequest, "invalid transition or nonmonotonic sequence/time")
	}
	// Only progress, observations and the terminal outcome may change.
	identity := r
	identity.Sequence, identity.Phase, identity.Terminal = old.Sequence, old.Phase, old.Terminal
	identity.UpdatedAt, identity.FinishedAt = old.UpdatedAt, old.FinishedAt
	identity.Source, identity.Target, identity.Error = old.Source, old.Target, old.Error
	identity.RecoveryRequired = old.RecoveryRequired
	if !reflect.DeepEqual(identity, old) {
		return switchStoreError(settings.SwitchErrorInvalidRequest, "receipt identity changed")
	}
	if err := r.Validate(); err != nil {
		return err
	}
	next := s.copyJournal()
	next.Receipts[id] = cloneSwitchReceipt(r)
	if r.Terminal {
		next.ActiveRequestID = ""
	}
	return s.persist(next)
}

// InterruptNonterminal does not guess runtime facts or resume switching. Even an
// accepted receipt can have a provisional manager reservation, so all interrupted
// operations require reconciliation by the coordinator before admitting work.
func (s *engineSwitchStore) InterruptNonterminal(now time.Time) error {
	if err := s.trustError(); err != nil {
		return err
	}
	for id, old := range s.journal.Receipts {
		if old.Terminal {
			continue
		}
		if old.Sequence == math.MaxUint64 {
			return switchStoreError(settings.SwitchErrorStorage, "receipt sequence exhausted")
		}
		r := cloneSwitchReceipt(old)
		r.Sequence++
		r.Phase, r.Terminal = settings.SwitchFailedInterrupted, true
		if now.Before(r.UpdatedAt) {
			now = r.UpdatedAt
		}
		r.UpdatedAt, r.FinishedAt = now, &now
		r.RecoveryRequired = true
		r.Error = &settings.SwitchError{Code: settings.SwitchOperationErrorCode, Kind: settings.SwitchErrorInterrupted, Message: "Engine switch interrupted by broker restart."}
		return s.Advance(id, r)
	}
	return nil
}

// The immutable receipt portion has no independent string-length ceiling. Keep
// its actual encoded size, reserving 16 KiB for ALL mutable fields (including a
// 1024-byte error expanded sixfold by JSON escaping, observations/timestamps and
// uint64 sequence). This exceeds their worst-case encoding. The active entry's
// reservation is included when Reject adds another receipt, too. No eviction.
func switchStoreCapacity(j settings.SwitchJournal) error {
	data, err := json.Marshal(j)
	if err != nil {
		return err
	}
	reserved := len(data) + 20 // generation may grow to a uint64
	for _, r := range j.Receipts {
		if r.Terminal {
			continue
		}
		encoded, err := json.Marshal(r)
		if err != nil {
			return err
		}
		r.Sequence, r.Phase, r.Terminal = 0, "", false
		r.UpdatedAt, r.FinishedAt = time.Time{}, nil
		r.Source, r.Target, r.Error = settings.SwitchEngineState{}, settings.SwitchEngineState{}, nil
		r.RecoveryRequired = false
		base, err := json.Marshal(r)
		if err != nil {
			return err
		}
		reserved += len(base) + 16*1024 - len(encoded)
	}
	if reserved > settings.MaxSwitchJournalBytes {
		return switchStoreError(settings.SwitchErrorFull, "journal capacity including terminal reservation exhausted")
	}
	return nil
}

func (s *engineSwitchStore) copyJournal() settings.SwitchJournal {
	j := s.journal
	j.Tombstones = append([]settings.SwitchTombstone(nil), s.journal.Tombstones...)
	if s.journal.LocalObligations != nil {
		j.LocalObligations = make(map[string]settings.SwitchLocalObligation, len(s.journal.LocalObligations))
		for id, obligation := range s.journal.LocalObligations {
			j.LocalObligations[id] = obligation
		}
	}
	j.Receipts = make(map[string]settings.SwitchReceipt, len(s.journal.Receipts))
	for id, r := range s.journal.Receipts {
		j.Receipts[id] = cloneSwitchReceipt(r)
	}
	return j
}

func (s *engineSwitchStore) SaveLocalObligation(id string, obligation settings.SwitchLocalObligation) error {
	if err := s.trustError(); err != nil {
		return err
	}
	next := s.copyJournal()
	if next.LocalObligations == nil {
		next.LocalObligations = make(map[string]settings.SwitchLocalObligation)
	}
	next.LocalObligations[id] = obligation
	return s.persist(next)
}

func (s *engineSwitchStore) persist(next settings.SwitchJournal) error {
	if s.journal.Generation == math.MaxUint64 {
		return switchStoreError(settings.SwitchErrorStorage, "journal generation exhausted")
	}
	next.Generation++
	if err := next.Validate(); err != nil {
		return err
	}
	data, err := json.Marshal(next)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	err = s.write(s.path, data, 0o600)
	if err == nil || errors.Is(err, durablefile.ErrReplaced) {
		s.journal = next
	}
	if err != nil {
		return &engineSwitchStoreError{Kind: settings.SwitchErrorStorage, Err: fmt.Errorf("persist journal: %w", err)}
	}
	return nil
}

func cloneSwitchReceipt(r settings.SwitchReceipt) settings.SwitchReceipt {
	cloneState := func(s settings.SwitchEngineState) settings.SwitchEngineState {
		if s.Running != nil {
			v := *s.Running
			s.Running = &v
		}
		if s.Healthy != nil {
			v := *s.Healthy
			s.Healthy = &v
		}
		if s.ObservedAt != nil {
			v := *s.ObservedAt
			s.ObservedAt = &v
		}
		return s
	}
	r.Source, r.Target = cloneState(r.Source), cloneState(r.Target)
	if r.Error != nil {
		v := *r.Error
		r.Error = &v
	}
	if r.FinishedAt != nil {
		v := *r.FinishedAt
		r.FinishedAt = &v
	}
	return r
}
