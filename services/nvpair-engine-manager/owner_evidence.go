// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"time"

	"nvpair-shared/buildapply"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modeldownload"
)

var errOwnerAdmissionCapacity = errors.New("engine owner admission generation exhausted")
var errOwnerEvidenceBusy = errors.New("engine owner evidence busy")

// These detached captures compose the existing owners. Equality rejects an
// observed mixed read; it cannot pin mutations after the response is produced.
type ownerSections struct {
	Saved        desiredOwnerEvidence
	SavedKnown   bool
	Universe     ownerUniverseEvidence
	Reservations ownerReservationEvidence
}

func (m *Manager) captureOwnerSections() ownerSections {
	var c ownerSections
	if m.exec.desired != nil {
		var err error
		c.Saved, err = m.exec.desired.tryOwnerEvidence()
		c.SavedKnown = err == nil
		c.Universe = m.ownerUniverse()
	}
	c.Reservations = m.ownerReservations()
	return c
}

func (m *Manager) engineOwnerEvidence() (buildapply.EngineOwnerEvidence, error) {
	result := buildapply.EngineOwnerEvidence{Reasons: []buildapply.EngineOwnerReason{}}
	if m.exec == nil || m.exec.switchIncarnation == "" {
		return result, errors.New("engine owner evidence unavailable")
	}
	result.ManagerIncarnation = m.exec.switchIncarnation
	a, b := m.captureOwnerSections(), m.captureOwnerSections()
	reason := func(section, engine, code string) {
		result.Reasons = append(result.Reasons, buildapply.EngineOwnerReason{Section: section, Engine: engine, Code: code})
	}
	if !reflect.DeepEqual(a, b) {
		for _, section := range []string{"intent", "locks", "universe", "stopAuthority", "reservations"} {
			reason(section, "", "changed")
		}
		err := result.Finalize()
		return result, err
	}
	if a.SavedKnown {
		enabled, off, locks := map[string]bool{}, map[string]string{}, map[string]bool{}
		for name, value := range a.Saved.State.Engines {
			enabled[name] = value
		}
		for name, value := range a.Saved.State.OffBy {
			off[name] = string(value)
		}
		for name, value := range a.Saved.State.Locked {
			locks[name] = value
		}
		result.Intent = buildapply.Known(buildapply.EngineOwnerIntent{Revision: a.Saved.IntentRevision, EnabledByEngine: enabled, OffByEngine: off})
		result.Locks = buildapply.Known(buildapply.EngineOwnerLocks{Revision: a.Saved.LockRevision, LockedByEngine: locks})
	} else {
		reason("intent", "", "saved_state_unknown")
		reason("locks", "", "saved_state_unknown")
	}
	if a.Universe.Complete {
		universe := make([]buildapply.EngineOwnerUniverseEntry, 0, len(a.Universe.Entries))
		authority := make([]buildapply.EngineOwnerStopAuthority, 0, len(a.Universe.Entries))
		for _, entry := range a.Universe.Entries {
			u := buildapply.EngineOwnerUniverseEntry{Engine: entry.Name, Registered: entry.Registered, Installed: entry.Installed,
				Running: entry.Running, TerminationVerified: entry.PositiveStop, RootsComplete: entry.RootsComplete}
			if entry.Registered {
				u.Supported = buildapply.Known(entry.Supported)
			}
			if entry.RunID != "" {
				u.EngineRunID = buildapply.Known(entry.RunID)
			}
			if entry.ModelRoot != "" {
				u.ModelRoots = buildapply.Known([]string{entry.ModelRoot})
			}
			// DataRoots contains represented owner paths, including the managed
			// install root. RootsComplete separately qualifies exhaustive coverage.
			if entry.InstallRoot != "" {
				u.DataRoots = buildapply.Known([]string{entry.InstallRoot})
			}
			if entry.RootsComplete {
				if !u.ModelRoots.Known {
					u.ModelRoots = buildapply.Known([]string{})
				}
				if !u.DataRoots.Known {
					u.DataRoots = buildapply.Known([]string{})
				}
			} else {
				reason("universe", entry.Name, "unmodeled_data_roots")
			}
			if !u.Installed.Known || !u.Running.Known || !u.TerminationVerified.Known {
				reason("universe", entry.Name, "unobserved")
			}
			stop := buildapply.EngineOwnerStopAuthority{Engine: entry.Name, Scoped: entry.StopAuthority,
				ReclaimPending: entry.ReclaimPending, ProcessLifetime: entry.ProcessLifetime, Reason: entry.AuthorityReason}
			if entry.Running.Known || entry.StopAuthority.Known {
				stop.Adopted = buildapply.Known(entry.Adopted)
			}
			if !stop.Scoped.Known || stop.Scoped.Value != nil && !*stop.Scoped.Value {
				reason("stopAuthority", entry.Name, entry.AuthorityReason)
			}
			universe = append(universe, u)
			authority = append(authority, stop)
		}
		result.Universe, result.StopAuthority = buildapply.Known(universe), buildapply.Known(authority)
	} else {
		code := a.Universe.Reason
		if code == "" {
			code = "unavailable"
		}
		reason("universe", "", code)
		reason("stopAuthority", "", code)
	}
	r := a.Reservations
	if r.Reason == "busy" {
		reason("reservations", "", "busy")
	} else if r.Token.DownloadSource != result.ManagerIncarnation {
		reason("reservations", "", "source_mismatch")
	} else {
		reservations, err := projectOwnerReservations(r, a.Universe.Generation)
		if err != nil {
			return result, err
		}
		result.Reservations = buildapply.Known(reservations)
		if !r.Complete {
			reason("reservations", "", "content_authority_unknown")
		}
	}
	err := result.Finalize()
	return result, err
}

func projectOwnerReservations(r ownerReservationEvidence, admission uint64) (buildapply.EngineOwnerReservations, error) {
	result := buildapply.EngineOwnerReservations{BlocksApply: r.Busy, UnsafeCopiesComplete: r.Complete, Operations: []buildapply.EngineOwnerOperation{}}
	// Admission and all content/copy authority generations participate even if
	// the visible operations returned to their prior values between captures.
	revision, err := switchDigest(struct {
		Token     ownerReservationToken
		Admission uint64
	}{r.Token, admission})
	if err != nil {
		return result, err
	}
	result.Revision = revision
	ops := map[string]buildapply.EngineOwnerOperation{}
	add := func(id, engine, kind, phase, origin string, retained, unsafe bool) error {
		op := buildapply.EngineOwnerOperation{ID: id, Engine: engine, Kind: kind, Phase: phase, Retained: retained, Unsafe: unsafe}
		if origin != "" {
			op.SourceIncarnation = buildapply.Known(origin)
		}
		if prior, exists := ops[id]; exists {
			if prior.Engine != engine || prior.Kind != kind || prior.SourceIncarnation.Known && op.SourceIncarnation.Known && *prior.SourceIncarnation.Value != *op.SourceIncarnation.Value {
				return errors.New("ambiguous engine owner operation")
			}
			// A marker without provenance must remain unknown even if an active
			// entry happens to carry the same ID in this manager.
			if !prior.SourceIncarnation.Known || !op.SourceIncarnation.Known {
				op.SourceIncarnation = buildapply.Observation[string]{}
			}
			op.Retained, op.Unsafe = retained || prior.Retained, unsafe || prior.Unsafe
		}
		ops[id] = op
		return nil
	}
	for _, v := range r.ActiveDownloads {
		if err := add(v.Entry.ID, v.Entry.Engine, "download", v.Entry.Phase, v.Incarnation, false, false); err != nil {
			return result, err
		}
	}
	for _, v := range r.UnsafeDownloads {
		if err := add(v.Marker.ID, v.Marker.Engine, "download", "unsafe", v.Incarnation, true, true); err != nil {
			return result, err
		}
	}
	for _, v := range r.Copies {
		if err := add(v.OpID, v.Engine, "copy", "active", v.Incarnation, false, false); err != nil {
			return result, err
		}
	}
	for _, v := range r.RetainedCopies {
		if err := add(v.OpID, v.Engine, "copy", v.Phase, v.ManagerIncarnation, true, true); err != nil {
			return result, err
		}
	}
	content := func(engine, kind string) error {
		id, err := switchDigest(struct{ Engine, Kind string }{engine, kind})
		if err != nil {
			return err
		}
		return add(strings.TrimPrefix(id, "sha256:"), engine, kind, "active", r.Token.Incarnation, false, false)
	}
	for engine := range r.Readers {
		if err := content(engine, "content_reader"); err != nil {
			return result, err
		}
	}
	for _, engine := range r.Writers {
		if err := content(engine, "content_writer"); err != nil {
			return result, err
		}
	}
	for _, op := range ops {
		result.Operations = append(result.Operations, op)
	}
	sort.Slice(result.Operations, func(i, j int) bool { return result.Operations[i].ID < result.Operations[j].ID })
	return result, nil
}

// A comparison token for one read, never an admission pin. Percent-only
// notifications change DownloadRevision without changing ContentGeneration.
type ownerReservationToken struct {
	Incarnation, DownloadSource                       string
	CopyRevision, DownloadRevision, ContentGeneration uint64
}

type ownerDownloadReservation struct {
	Incarnation string // Empty for retained markers whose format has no origin.
	Entry       modeldownload.Entry
}

type ownerUnsafeDownload struct {
	Incarnation string
	Marker      modelDownloadMarker
}

type ownerCopyReservation struct {
	copyRequest
	Incarnation        string
	Role               string
	PendingPublication bool
	Progress           *copyProgress
}

type ownerReservationEvidence struct {
	Token                                  ownerReservationToken
	Complete                               bool
	Busy                                   buildapply.Observation[bool]
	Reason                                 string
	Shutdown                               bool
	RecoveryRequired, CopyRecoveryRequired bool
	Uncertain                              bool
	ActiveDownloads                        []ownerDownloadReservation
	UnsafeDownloads                        []ownerUnsafeDownload
	Copies                                 []ownerCopyReservation
	RetainedCopies                         []modelCopyMarker
	Readers                                map[string]int
	Writers                                []string
}

// This is the only joint reservation capture. The lock order is copyMu then
// downloads.mu. Mutation paths take these separately and release them before
// journal I/O, transport, notifications, cancellation or completion callbacks.
// Both gate and list are evaluated here, from precisely the same owner state.
func (m *Manager) ownerReservations() ownerReservationEvidence {
	r := ownerReservationEvidence{Reason: "busy"}
	if m.exec != nil {
		r.Token.Incarnation = m.exec.switchIncarnation
	}
	if !m.copyMu.TryLock() {
		return r
	}
	defer m.copyMu.Unlock()
	r.Token.CopyRevision, r.Shutdown = m.copyRevision, m.copiesClosed
	for _, entry := range m.copies {
		copy := ownerCopyReservation{copyRequest: entry.copyRequest, Incarnation: r.Token.Incarnation, Role: entry.role, PendingPublication: entry.pendingPublication}
		copy.Progress = cloneCopyProgress(entry.Progress)
		if copy.Progress != nil {
			// Free-form diagnostics and filenames are not reservation authority.
			copy.Progress.Message, copy.Progress.File = "", ""
		}
		r.Copies = append(r.Copies, copy)
	}
	busy := r.Shutdown || len(r.Copies) != 0
	d := m.copyJournal
	if d == nil || !d.mu.TryLock() {
		if busy {
			r.Busy = buildapply.Known(true)
		}
		return r
	}
	defer d.mu.Unlock()
	r.Token.DownloadSource, r.Token.DownloadRevision, r.Token.ContentGeneration = d.source, d.revision, d.authorityGeneration
	r.RecoveryRequired, r.CopyRecoveryRequired, r.Uncertain = d.recoveryRequired, d.copyRecoveryRequired, d.authorityUncertain
	r.Complete = d.copiesComplete && !r.RecoveryRequired && !r.CopyRecoveryRequired && !r.Uncertain && d.source != "" && r.Token.Incarnation == d.source
	r.Reason = ""
	for _, copy := range r.Copies {
		if copy.Role != "controller" && copy.Role != "destination" {
			r.Complete = false
		}
	}
	for _, attempt := range d.active {
		entry := attempt.entry
		entry.Percent = cloneDownloadPercent(entry.Percent)
		r.ActiveDownloads = append(r.ActiveDownloads, ownerDownloadReservation{Incarnation: d.source, Entry: entry})
	}
	for key, marker := range d.unsafe {
		origin := d.unsafeOrigins[key]
		r.UnsafeDownloads = append(r.UnsafeDownloads, ownerUnsafeDownload{Incarnation: origin, Marker: marker})
		active := d.active[key]
		if origin == "" || active == nil || active.entry.ID != marker.ID {
			r.Complete = false
		}
	}
	for _, marker := range d.copies {
		r.RetainedCopies = append(r.RetainedCopies, marker)
		live := m.copies[marker.OpID]
		identity := copyRequest{OpID: marker.OpID, Engine: marker.Engine, Model: marker.Model, SourceNodeID: marker.SourceNodeID, DestinationNodeID: marker.DestinationNodeID, ExpectedExportID: marker.ExpectedExportID}
		if marker.ManagerIncarnation != d.source || marker.Phase == "uncertain" || live == nil || live.copyRequest != identity || live.role != marker.Role {
			r.Complete = false
		}
	}
	r.Readers = make(map[string]int)
	for engine, count := range d.readers {
		if count != 0 {
			r.Readers[engine] = count
		}
	}
	for engine, writing := range d.writers {
		if writing {
			r.Writers = append(r.Writers, engine)
		}
	}
	busy = busy || len(r.ActiveDownloads) != 0 || len(r.UnsafeDownloads) != 0 || len(r.RetainedCopies) != 0 || len(r.Readers) != 0 || len(r.Writers) != 0
	if busy || r.Complete {
		r.Busy = buildapply.Known(busy)
	}
	if !r.Complete {
		r.Reason = "content_authority_unknown"
	}
	sort.Slice(r.Copies, func(i, j int) bool { return r.Copies[i].OpID < r.Copies[j].OpID })
	sort.Slice(r.ActiveDownloads, func(i, j int) bool { return r.ActiveDownloads[i].Entry.ID < r.ActiveDownloads[j].Entry.ID })
	sort.Slice(r.UnsafeDownloads, func(i, j int) bool { return r.UnsafeDownloads[i].Marker.ID < r.UnsafeDownloads[j].Marker.ID })
	sort.Slice(r.RetainedCopies, func(i, j int) bool { return r.RetainedCopies[i].OpID < r.RetainedCopies[j].OpID })
	sort.Strings(r.Writers)
	return r
}

func (e *Executor) ownerAdmissionCapacityLocked() bool {
	return math.MaxUint64-e.ownerAdmissionGeneration >= e.ownerAdmissionHeld &&
		math.MaxUint64-e.ownerAdmissionGeneration-e.ownerAdmissionHeld >= 2
}

// Caller holds switchAdmission. The returned retirement also requires it.
func (e *Executor) reserveOwnerAdmissionLocked() (func(), error) {
	if !e.ownerAdmissionCapacityLocked() {
		return nil, errOwnerAdmissionCapacity
	}
	e.ownerAdmissionGeneration++
	e.ownerAdmissionHeld++
	retired := false
	return func() {
		if !retired {
			retired = true
			e.ownerAdmissionHeld--
			e.ownerAdmissionGeneration++
		}
	}, nil
}

// Switch checkpoint work has its own existing exclusion policy. This metadata
// fences its observations without changing that policy or adding a reservation.
func (e *Executor) admitOwnerCheckpoint() (func(), error) {
	e.switchAdmission.Lock()
	retire, err := e.reserveOwnerAdmissionLocked()
	e.switchAdmission.Unlock()
	if err != nil {
		return nil, err
	}
	return func() { e.switchAdmission.Lock(); retire(); e.switchAdmission.Unlock() }, nil
}

// All fields are guarded by st.mu. A publication is bound to the actual state
// configuration and retained process object, never to a PID or an OS lookup.
type ownerStateKnowledge struct {
	InstalledKnown             bool
	InstalledConfig            string
	InstalledValue             bool
	InstalledRunID             string
	PresenceKnown              bool
	PositiveStop               bool
	PresenceConfig             string
	Process                    *managedProc
	RunID                      string
	RunningValue, AdoptedValue bool
}

func (st *engineState) ownerConfigLocked() string {
	if st.plat == nil {
		return ""
	}
	digest, err := switchDigest(struct {
		Platform        *Platform
		Install, Models string
	}{st.plat, st.installDir, st.modelsDir})
	if err != nil {
		return ""
	}
	return digest
}

func (st *engineState) withdrawOwnerKnowledgeLocked() { st.ownerKnowledge = ownerStateKnowledge{} }

func (st *engineState) observeOwnerInstalledLocked() {
	st.ownerKnowledge.InstalledValue = st.installed
	st.ownerKnowledge.InstalledRunID = st.runID
	st.ownerKnowledge.InstalledConfig = st.ownerConfigLocked()
	st.ownerKnowledge.InstalledKnown = st.ownerKnowledge.InstalledConfig != ""
}

func (st *engineState) observeOwnerPresenceLocked(positiveStop bool) {
	k := &st.ownerKnowledge
	k.PresenceConfig = st.ownerConfigLocked()
	k.PresenceKnown = k.PresenceConfig != ""
	k.PositiveStop = positiveStop && !st.running && st.proc == nil && !st.reclaimFailed && st.reclaimPID == 0
	k.Process, k.RunID = st.proc, st.runID
	k.RunningValue, k.AdoptedValue = st.running, st.adopted
}

func (e *Executor) withdrawOwnerKnowledge(engine string) {
	e.mu.Lock()
	states := make([]*engineState, 0)
	for name, st := range e.engines {
		if engine == "" || name == engine {
			states = append(states, st)
		}
	}
	for name, st := range e.stopOnly {
		if engine == "" || name == engine {
			states = append(states, st)
		}
	}
	e.mu.Unlock()
	for _, st := range states {
		st.mu.Lock()
		st.withdrawOwnerKnowledgeLocked()
		st.mu.Unlock()
	}
}

type ownerEngineEvidence struct {
	Name                                            string
	Registered, Supported, Cleanup                  bool
	Installed, Running, PositiveStop, StopAuthority buildapply.Observation[bool]
	RunID, RuntimeKind, AuthorityReason             string
	InstallRoot, ModelRoot                          string
	RootsComplete                                   bool
	RootsReason                                     string
	Adopted, ReclaimFailed                          bool
	ReclaimPID                                      bool // retained locator only; never signal authority
	ProcessLifetime                                 buildapply.Observation[buildapply.EngineOwnerProcessLifetime]
	ReclaimPending                                  buildapply.Observation[bool]
}

type ownerUniverseEvidence struct {
	Incarnation string
	Generation  uint64
	Complete    bool
	Reason      string
	Entries     []ownerEngineEvidence
}

type ownerAdmissionCapture struct {
	Generation, Held uint64
	Mutations        map[string]int
	Bound            *boundEngineActivation
	Shutdown         bool
}

func (e *Executor) captureOwnerAdmission() (ownerAdmissionCapture, bool) {
	if !e.switchAdmission.TryLock() {
		return ownerAdmissionCapture{}, false
	}
	defer e.switchAdmission.Unlock()
	a := ownerAdmissionCapture{Generation: e.ownerAdmissionGeneration, Held: e.ownerAdmissionHeld, Mutations: make(map[string]int), Shutdown: e.shuttingDown.Load()}
	for name, count := range e.switchMutations {
		if count != 0 {
			a.Mutations[name] = count
		}
	}
	if e.boundActivation != nil {
		copy := *e.boundActivation
		a.Bound = &copy
	}
	return a, true
}

type ownerStateCapture struct {
	Relaunch                                                   *engineRelaunch
	Current, StopOnly                                          *engineState
	Config                                                     string
	Knowledge                                                  ownerStateKnowledge
	Installed, Running, Adopted, Stopping, ReclaimFailed, Done bool
	ReclaimPID, Port                                           int
	Process                                                    *managedProc
	RunID, Kind, Install, Models                               string
}

// Registry construction is startup-owned. Membership is copied, then released
// before independently taking each existing state lock in sorted name order.
func (e *Executor) captureOwnerStates(saved desiredOwnerEvidence, reservations []string) ([]ownerUniverseEntry, map[string]ownerStateCapture, bool) {
	if !e.mu.TryLock() {
		return nil, nil, false
	}
	members := make(map[string]ownerStateCapture)
	for name, st := range e.engines {
		c := members[name]
		c.Current = st
		members[name] = c
	}
	for name, st := range e.stopOnly {
		c := members[name]
		c.StopOnly = st
		members[name] = c
	}
	e.mu.Unlock()
	entries, ok := e.ownerUniverseMembership(saved)
	if !ok {
		return nil, nil, false
	}
	byName := make(map[string]ownerUniverseEntry)
	for _, entry := range entries {
		byName[entry.Name] = entry
	}
	for _, name := range reservations {
		entry := byName[name]
		entry.Name = name
		byName[name] = entry
	}
	entries = entries[:0]
	for _, entry := range byName {
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
	for _, entry := range entries {
		c := members[entry.Name]
		st := c.Current
		if st == nil {
			st = c.StopOnly
		}
		if st != nil {
			if !st.mu.TryLock() {
				return nil, nil, false
			}
			c.Config, c.Knowledge = st.ownerConfigLocked(), st.ownerKnowledge
			c.Installed, c.Running, c.Adopted, c.Stopping = st.installed, st.running, st.adopted, st.stopping
			c.ReclaimFailed, c.ReclaimPID, c.Port = st.reclaimFailed, st.reclaimPID, st.port
			c.Process, c.RunID, c.Install, c.Models = st.proc, st.runID, st.installDir, st.modelsDir
			c.Relaunch = st.relaunch
			if st.plat != nil {
				c.Kind = st.plat.Runtime.modeOrDefault()
			}
			if c.Process != nil {
				if c.Process.done == nil {
					c.Done = true
				} else {
					select {
					case <-c.Process.done:
						c.Done = true
					default:
					}
				}
			}
			st.mu.Unlock()
		}
		members[entry.Name] = c
	}
	return entries, members, true
}

// Only existing reservation maps participate. No model inventory is consulted.
// Manager supplies its existing copy owner through the same capture boundary.
func (e *Executor) ownerReservationNames() ([]string, bool) {
	if e.downloads == nil {
		return nil, true
	}
	d := e.downloads
	if !d.mu.TryLock() {
		return nil, false
	}
	defer d.mu.Unlock()
	names := map[string]bool{}
	for k := range d.active {
		names[k.engine] = true
	}
	for k := range d.unsafe {
		names[k.engine] = true
	}
	for _, marker := range d.copies {
		names[marker.Engine] = true
	}
	for k := range d.outcomes {
		names[k.engine] = true
	}
	for name := range d.writers {
		names[name] = true
	}
	for name := range d.readers {
		names[name] = true
	}
	result := make([]string, 0, len(names))
	for name := range names {
		result = append(result, name)
	}
	sort.Strings(result)
	return result, true
}

func (m *Manager) ownerUniverse() ownerUniverseEvidence {
	return m.exec.captureOwnerUniverse(m.exec.desired.tryOwnerEvidence, m.ownerCopyNames)
}

func (m *Manager) ownerCopyNames() ([]string, bool) {
	if !m.copyMu.TryLock() {
		return nil, false
	}
	defer m.copyMu.Unlock()
	names := make([]string, 0, len(m.copies))
	for _, copy := range m.copies {
		names = append(names, copy.Engine)
	}
	sort.Strings(names)
	return names, true
}

// Two detached captures reject changed owners, including admission enter/exit
// ABA. This is evidence, not an atomic acceptance pin or physical liveness proof.
func (e *Executor) captureOwnerUniverse(savedRead func() (desiredOwnerEvidence, error), copyNames func() ([]string, bool)) ownerUniverseEvidence {
	result := ownerUniverseEvidence{Incarnation: e.switchIncarnation, Reason: "busy"}
	a, ok := e.captureOwnerAdmission()
	if !ok {
		return result
	}
	saved, err := savedRead()
	if err != nil {
		if !errors.Is(err, errOwnerEvidenceBusy) {
			result.Reason = "saved_state_unknown"
		}
		return result
	}
	reservations, ok := e.ownerReservationNames()
	if !ok {
		return result
	}
	copies, ok := copyNames()
	if !ok {
		return result
	}
	names, states, ok := e.captureOwnerStates(saved, append(reservations, copies...))
	if !ok {
		return result
	}
	// Recheck state/config/run before the final admission capture. Sampling done
	// again also rejects a closed lifetime without consulting the operating system.
	savedAgain, err := savedRead()
	if err != nil || !reflect.DeepEqual(savedAgain, saved) {
		return result
	}
	reservationsAgain, ok := e.ownerReservationNames()
	if !ok || !reflect.DeepEqual(reservations, reservationsAgain) {
		return result
	}
	copiesAgain, ok := copyNames()
	if !ok || !reflect.DeepEqual(copies, copiesAgain) {
		return result
	}
	namesAgain, statesAgain, ok := e.captureOwnerStates(savedAgain, append(reservationsAgain, copiesAgain...))
	if !ok || !reflect.DeepEqual(names, namesAgain) || !reflect.DeepEqual(states, statesAgain) {
		return result
	}
	b, ok := e.captureOwnerAdmission()
	if !ok || !reflect.DeepEqual(a, b) {
		return result
	}
	result.Generation, result.Complete, result.Reason = a.Generation, true, ""
	for _, name := range names {
		c := states[name.Name]
		entry := ownerEngineEvidence{Name: name.Name, Registered: name.Registered, Supported: name.Supported, Cleanup: name.Cleanup,
			RunID: c.RunID, RuntimeKind: c.Kind, Adopted: c.Adopted, ReclaimFailed: c.ReclaimFailed, ReclaimPID: c.ReclaimPID != 0,
			InstallRoot: c.Install, ModelRoot: c.Models, RootsReason: "unmodeled_data_roots", AuthorityReason: "unobserved"}
		// Only represented, resolved paths are usable protection roots. The
		// manifest has no exhaustive data-root contract, so coverage stays unknown.
		if c.Current == nil && c.StopOnly == nil && name.Registered && e.reg != nil {
			if manifest := e.reg.engines[name.Name]; manifest != nil {
				if platform, supported := manifest.Platforms[runtime.GOOS+"/"+runtime.GOARCH]; supported {
					entry.RuntimeKind = platform.Runtime.modeOrDefault()
					if filepath.IsAbs(e.baseDir) {
						entry.InstallRoot = filepath.Join(e.baseDir, name.Name)
					}
					entry.ModelRoot = platform.ModelsDir
				}
			}
		}
		if !ownerResolvedRoot(entry.InstallRoot) {
			entry.InstallRoot = ""
		}
		if !ownerResolvedRoot(entry.ModelRoot) {
			entry.ModelRoot = ""
		}
		busy := a.Held != 0 || a.Shutdown || a.Mutations[""] != 0 || a.Mutations[name.Name] != 0 || a.Bound != nil || c.Relaunch != nil
		if cp := saved.State.PendingSwitch; cp != nil && cp.Phase != settings.SwitchCheckpointReleased && (cp.Automatic || cp.SourceEngine == name.Name || cp.TargetEngine == name.Name) {
			busy = true
		}
		k := c.Knowledge
		if busy {
			entry.AuthorityReason = "busy"
		} else {
			entry.ReclaimPending = buildapply.Known(c.ReclaimFailed || c.ReclaimPID != 0 && !c.Adopted)
			if k.InstalledKnown && k.InstalledValue == c.Installed && k.InstalledRunID == c.RunID && c.Config != "" && k.InstalledConfig == c.Config {
				entry.Installed = buildapply.Known(c.Installed)
			}
			valid := k.PresenceKnown && c.Config != "" && k.PresenceConfig == c.Config && k.Process == c.Process && k.RunID == c.RunID && k.RunningValue == c.Running && k.AdoptedValue == c.Adopted && !c.Done
			if valid && !c.ReclaimFailed {
				entry.Running = buildapply.Known(c.Running)
				entry.PositiveStop = buildapply.Known(k.PositiveStop)
			}
			switch {
			case c.ReclaimFailed || c.ReclaimPID != 0 && !c.Adopted:
				entry.AuthorityReason = "reclaim_unknown"
			case !valid:
				if c.Done {
					entry.AuthorityReason = "process_lifetime_closed"
				}
			case c.Adopted:
				entry.StopAuthority, entry.AuthorityReason = buildapply.Known(false), "external_management"
			case k.PositiveStop && !c.Running && c.Process == nil:
				entry.StopAuthority, entry.AuthorityReason = buildapply.Known(true), "no_live_process"
			case c.Running && !c.Stopping && c.Process != nil && c.RunID != "":
				entry.StopAuthority, entry.AuthorityReason = buildapply.Known(true), "owned_process_lifetime"
				lifetime := buildapply.EngineOwnerProcessLifetime{EngineRunID: c.RunID}
				if c.Process.cmd != nil && c.Process.cmd.Process != nil && c.Process.cmd.Process.Pid > 0 {
					lifetime.PID = buildapply.Known(c.Process.cmd.Process.Pid)
				}
				entry.ProcessLifetime = buildapply.Known(lifetime)
			default:
				entry.AuthorityReason = "scoped_lifetime_unknown"
			}
		}
		result.Entries = append(result.Entries, entry)
	}
	return result
}

func ownerResolvedRoot(path string) bool {
	return filepath.IsAbs(path) && !strings.ContainsAny(path, "{}$%~")
}

// ownerUniverseEntry is a detached, deliberately incomplete starting point for
// the owner projection. Cached booleans do not establish lifecycle knowledge.
type ownerUniverseEntry struct {
	Name       string
	Registered bool
	Supported  bool
	Cleanup    bool
}

// ownerUniverseMembership captures only owners that can be read without
// creating state. The caller must still fence admission, reservations and
// lifecycle projections before treating this membership as complete.
func (e *Executor) ownerUniverseMembership(saved desiredOwnerEvidence) ([]ownerUniverseEntry, bool) {
	if !e.mu.TryLock() {
		return nil, false
	}
	names := make(map[string]ownerUniverseEntry)
	if e.reg != nil {
		for name, manifest := range e.reg.engines {
			_, supported := manifest.Platforms[runtime.GOOS+"/"+runtime.GOARCH]
			names[name] = ownerUniverseEntry{Name: name, Registered: true, Supported: supported}
		}
	}
	for name := range e.engines {
		entry := names[name]
		entry.Name = name
		names[name] = entry
	}
	for name := range e.stopOnly {
		entry := names[name]
		entry.Name, entry.Cleanup = name, true
		names[name] = entry
	}
	e.mu.Unlock()
	add := func(name string, cleanup bool) {
		entry := names[name]
		entry.Name = name
		entry.Cleanup = entry.Cleanup || cleanup
		names[name] = entry
	}
	for name := range saved.State.Engines {
		add(name, false)
	}
	for name := range saved.State.Locked {
		add(name, false)
	}
	for name := range saved.State.OffBy {
		add(name, false)
	}
	for name := range saved.State.ProcessStops {
		add(name, true)
	}
	for name := range saved.State.AllocatedPorts {
		add(name, true)
	}
	if cp := saved.State.PendingSwitch; cp != nil {
		add(cp.SourceEngine, false)
		add(cp.TargetEngine, false)
	}
	result := make([]ownerUniverseEntry, 0, len(names))
	for _, entry := range names {
		result = append(result, entry)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result, true
}

// desiredOwnerEvidence is a detached view of one saved document. Revisions
// describe readable semantics, not write durability, mutation history or an
// absence of intervening ABA edits. They also serve as protected semantic
// digests across restarts; a consumer must bind its fence to the manager's
// separate incarnation. No runtime stop or completion fact follows from State.
type desiredOwnerEvidence struct {
	State          desiredStateFile
	IntentRevision string
	LockRevision   string
}

func (s *desiredStateStore) ownerEvidence() (desiredOwnerEvidence, error) {
	return s.captureOwnerEvidence(s.readDocument)
}

func (s *desiredStateStore) captureOwnerEvidence(read func() ([]byte, error)) (desiredOwnerEvidence, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.captureOwnerEvidenceLocked(read)
}

func (s *desiredStateStore) tryOwnerEvidence() (desiredOwnerEvidence, error) {
	if !s.mu.TryLock() {
		return desiredOwnerEvidence{}, errOwnerEvidenceBusy
	}
	defer s.mu.Unlock()
	return s.captureOwnerEvidenceLocked(s.readDocument)
}

func (s *desiredStateStore) captureOwnerEvidenceLocked(read func() ([]byte, error)) (desiredOwnerEvidence, error) {
	if s.path == "" {
		return desiredOwnerEvidence{}, errors.New("no user data directory is available")
	}
	data, err := read()
	if err != nil {
		return desiredOwnerEvidence{}, err
	}
	// Ordinary intent accessors retain their existing decoding behavior. Evidence
	// must not turn malformed null booleans into explicit Off or unlocked values.
	if err := validateDesiredOwnerDocument(data); err != nil {
		return desiredOwnerEvidence{}, err
	}
	state, err := decodeDesiredState(data)
	if err != nil {
		return desiredOwnerEvidence{}, err
	}
	for name, origin := range state.OffBy {
		enabled, present := state.Engines[name]
		if !present || enabled || (origin != intentLocal && origin != intentPeer) {
			return desiredOwnerEvidence{}, fmt.Errorf("invalid off provenance for %s", name)
		}
	}
	for name, enabled := range state.Engines {
		if !enabled {
			if _, present := state.OffBy[name]; !present {
				state.OffBy[name] = intentLocal
			}
		}
	}
	for name, locked := range state.Locked {
		enabled, present := state.Engines[name]
		if locked && (!present || enabled || state.OffBy[name] != intentLocal) {
			return desiredOwnerEvidence{}, fmt.Errorf("locked engine %s lacks local Off intent", name)
		}
	}
	if cp := state.PendingSwitch; cp != nil {
		for _, observed := range []settings.SwitchEngineState{cp.PreSource, cp.PreTarget, cp.Source, cp.Target} {
			if err := observed.Validate(); err != nil {
				return desiredOwnerEvidence{}, fmt.Errorf("invalid pending switch state: %w", err)
			}
		}
		for _, prior := range []struct {
			intent string
			origin string
		}{{cp.PreSource.Intent, cp.SourceProvenance}, {cp.PreTarget.Intent, cp.TargetProvenance}} {
			if prior.origin != "" && (prior.intent != settings.SwitchIntentOff || (prior.origin != string(intentLocal) && prior.origin != string(intentPeer))) {
				return desiredOwnerEvidence{}, errors.New("invalid pending switch Off provenance")
			}
		}
		if cp.Error != nil {
			if err := cp.Error.Validate(); err != nil {
				return desiredOwnerEvidence{}, fmt.Errorf("invalid pending switch error: %w", err)
			}
		}
	} else if state.PendingSwitchPorts != nil {
		return desiredOwnerEvidence{}, errors.New("pending switch endpoints lack a checkpoint")
	}
	// encoding/json sorts map keys. Explicit false lock entries and intent key
	// presence remain part of their respective baselines. Each decode owns all
	// maps and nested checkpoint pointers, so no store or other read aliases them.
	intent, err := switchDigest(struct {
		Engines map[string]bool         `json:"engines"`
		OffBy   map[string]intentOrigin `json:"offBy"`
	}{state.Engines, state.OffBy})
	if err != nil {
		return desiredOwnerEvidence{}, err
	}
	locks, err := switchDigest(state.Locked)
	if err != nil {
		return desiredOwnerEvidence{}, err
	}
	return desiredOwnerEvidence{State: state, IntentRevision: intent, LockRevision: locks}, nil
}

func validateDesiredOwnerDocument(data []byte) error {
	if err := validateOwnerUniqueMembers(json.NewDecoder(bytes.NewReader(data)), 0); err != nil {
		return err
	}
	if err := validateOwnerScalarPresence(data, reflect.TypeOf(desiredStateFile{})); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return fmt.Errorf("decode desired engine evidence: %w", err)
	}
	if fields == nil {
		return errors.New("desired engine evidence must be an object")
	}
	for name, raw := range fields {
		switch name {
		case "engines", "offBy", "locked", "processStops", "allocatedPorts":
			var entries map[string]json.RawMessage
			if err := json.Unmarshal(raw, &entries); err != nil || entries == nil {
				return fmt.Errorf("desired engine evidence %s must be an object", name)
			}
			if name == "engines" || name == "locked" {
				for engine, value := range entries {
					value = bytes.TrimSpace(value)
					if !bytes.Equal(value, []byte("true")) && !bytes.Equal(value, []byte("false")) {
						return fmt.Errorf("desired engine evidence %s.%s must be boolean", name, engine)
					}
				}
			}
		case "pendingSwitch", "pendingSwitchPorts":
			// The document decoder validates these optional checkpoint records.
		default:
			return fmt.Errorf("unrecognized desired engine evidence field %q", name)
		}
	}
	return nil
}

// JSON null must not silently become a known zero-valued scalar. Pointer fields
// retain their existing unknown/absent semantics; ordinary accessors are unchanged.
func validateOwnerScalarPresence(raw json.RawMessage, typ reflect.Type) error {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		if typ.Kind() == reflect.Pointer {
			return nil
		}
		return errors.New("null nonnullable field in desired engine evidence")
	}
	if typ.Kind() == reflect.Pointer {
		return validateOwnerScalarPresence(raw, typ.Elem())
	}
	if typ == reflect.TypeOf(time.Time{}) {
		return nil // The document decoder validates the timestamp string.
	}
	switch typ.Kind() {
	case reflect.Struct:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			return err
		}
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			if field.Anonymous {
				if err := validateOwnerScalarPresence(raw, field.Type); err != nil {
					return err
				}
				continue
			}
			name := strings.Split(field.Tag.Get("json"), ",")[0]
			if name == "" {
				name = field.Name
			}
			for key := range fields {
				if key != name && strings.EqualFold(key, name) {
					return errors.New("noncanonical field name in desired engine evidence")
				}
			}
			if value, ok := fields[name]; ok {
				if err := validateOwnerScalarPresence(value, field.Type); err != nil {
					return err
				}
			}
		}
	case reflect.Map:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			return err
		}
		for _, value := range fields {
			if err := validateOwnerScalarPresence(value, typ.Elem()); err != nil {
				return err
			}
		}
	}
	return nil
}

// A map decode loses duplicate members, while decoding into the saved-state
// structs may merge them. Reject that ambiguity before either projection.
func validateOwnerUniqueMembers(dec *json.Decoder, depth int) error {
	if depth > 64 {
		return errors.New("desired engine evidence nesting exceeds its document shape")
	}
	token, err := dec.Token()
	if err != nil {
		return err
	}
	delim, composite := token.(json.Delim)
	if !composite {
		return nil
	}
	seen := map[string]bool{}
	for dec.More() {
		if delim == '{' {
			key, err := dec.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[name] {
				return errors.New("duplicate member in desired engine evidence")
			}
			seen[name] = true
		}
		if err := validateOwnerUniqueMembers(dec, depth+1); err != nil {
			return err
		}
	}
	_, err = dec.Token()
	return err
}
