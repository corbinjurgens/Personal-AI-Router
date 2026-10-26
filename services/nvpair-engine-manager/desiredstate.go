// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"nvpair-shared/engineintent"
	settings "nvpair-shared/enginesettings"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	desiredStateFileName  = "engine-state.json"
	restoreEnabledMethod  = "engine:restore-enabled"
	prepareShutdownMethod = "engine:prepare-shutdown"
)

type desiredStateFile struct {
	PendingSwitch      *settings.SwitchCheckpoint `json:"pendingSwitch,omitempty"`
	PendingSwitchPorts *switchCheckpointPorts     `json:"pendingSwitchPorts,omitempty"`
	Engines            map[string]bool            `json:"engines"`
	// AllocatedPorts locates process-mode port-zero launches after manager death.
	// A port is evidence only; reclaim still checks the current managed image.
	AllocatedPorts map[string]int `json:"allocatedPorts,omitempty"`
	// ProcessStops survives removal of a manifest's host block. No arguments,
	// environment, commands or credentials belong in this cleanup-only record.
	ProcessStops map[string]processStopEvidence `json:"processStops,omitempty"`
	// OffBy records who saved each engine Off: intentLocal or intentPeer. An
	// engine saved Off with no entry predates provenance and counts as local.
	OffBy  map[string]intentOrigin `json:"offBy,omitempty"`
	Locked map[string]bool         `json:"locked,omitempty"`
}

type processStopEvidence struct {
	Bin     string `json:"bin"`
	Port    int    `json:"port"`
	Address string `json:"address"`
	// Zero means a crash could have occurred between spawn and saving its PID.
	PID int `json:"pid"`
}

func (evidence processStopEvidence) valid() bool {
	// Match probeListenerAddress: an empty TCP host denotes the local system.
	_, _, addressErr := net.SplitHostPort(evidence.Address)
	return filepath.IsAbs(evidence.Bin) && evidence.Port > 0 && evidence.Port <= 65535 && evidence.PID >= 0 && addressErr == nil
}

func (s *desiredStateStore) processStop(engine string) (processStopEvidence, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	evidence, known := state.ProcessStops[engine]
	return evidence, known, err
}

func (s *desiredStateStore) setProcessStop(engine string, evidence processStopEvidence) error {
	if !evidence.valid() {
		return fmt.Errorf("invalid process stop evidence for %s", engine)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	if err != nil {
		return err
	}
	if s.path == "" {
		return errors.New("no user data directory is available")
	}
	if state.ProcessStops == nil {
		state.ProcessStops = make(map[string]processStopEvidence)
	}
	state.ProcessStops[engine] = evidence
	return s.save(state)
}

func (state *desiredStateFile) UnmarshalJSON(data []byte) error {
	type document desiredStateFile
	var decoded document
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for name, raw := range fields {
		if name == "pendingSwitch" && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			var checkpoint map[string]json.RawMessage
			if err := json.Unmarshal(raw, &checkpoint); err != nil {
				return err
			}
			for key, cause := range checkpoint {
				if strings.EqualFold(key, "automatic") && (key != "automatic" || (!bytes.Equal(bytes.TrimSpace(cause), []byte("true")) && !bytes.Equal(bytes.TrimSpace(cause), []byte("false")))) {
					return errors.New("pending switch automatic must be boolean")
				}
			}
		}
		if !strings.EqualFold(name, "locked") {
			continue
		}
		var values map[string]*bool
		if err := json.Unmarshal(raw, &values); err != nil || values == nil {
			return errors.New("locked must be an object of boolean values")
		}
		for engine, value := range values {
			if value == nil {
				return fmt.Errorf("locked.%s must be boolean", engine)
			}
		}
	}
	*state = desiredStateFile(decoded)
	return nil
}

// Allocated ports are execution evidence, excluded from configuration tokens.
// Keep them beside the shared checkpoint so crash cleanup can find a port-zero
// engine's actual listener after the in-memory lifecycle state is lost.
type switchCheckpointPorts struct {
	Source int `json:"source"`
	Target int `json:"target"`
	// Automatic source execution evidence, never permission to signal a PID.
	SourcePID   int    `json:"sourcePid,omitempty"`
	SourceRunID string `json:"sourceRunId,omitempty"`
}

// intentOrigin says who saved an engine's intent: this node's own clients
// (stdio engine:start / engine:stop) or a paired node over the ec surface.
type intentOrigin string

const (
	intentLocal intentOrigin = "local"
	intentPeer  intentOrigin = "peer"
)

// desiredStateStore records explicit user intent. A missing engine key is a
// legacy/unknown state and deliberately does not auto-start.
type desiredStateStore struct {
	mu   sync.Mutex
	path string
}

func newDesiredStateStore(baseDir string) *desiredStateStore {
	path := ""
	if baseDir != "" {
		path = filepath.Join(baseDir, desiredStateFileName)
	}
	return &desiredStateStore{path: path}
}

func (s *desiredStateStore) get(engine string) (enabled, known bool, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	if err != nil {
		return false, false, err
	}
	enabled, known = state.Engines[engine]
	return enabled, known, nil
}

func (s *desiredStateStore) allocatedPort(engine string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	return state.AllocatedPorts[engine], err
}

// setAllocatedPort records the endpoint before spawn. Zero clears execution
// evidence after a clean stop without changing the user's saved intent.
func (s *desiredStateStore) setAllocatedPort(engine string, port int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	if err != nil {
		return err
	}
	if state.AllocatedPorts[engine] == port {
		return nil
	}
	if s.path == "" {
		return errors.New("no user data directory is available")
	}
	if port == 0 {
		delete(state.AllocatedPorts, engine)
	} else {
		if port < 1 || port > 65535 {
			return errors.New("invalid allocated engine port")
		}
		if state.AllocatedPorts == nil {
			state.AllocatedPorts = make(map[string]int)
		}
		state.AllocatedPorts[engine] = port
	}
	return s.save(state)
}

// Callers hold opMu and have verified that this launch is stopped (or never
// spawned). A failed durable removal remains work for the next stop owner.
func (e *Executor) clearAllocatedPort(st *engineState, engine string) error {
	err := e.desired.setAllocatedPort(engine, 0)
	st.mu.Lock()
	st.portCleanupPending = err != nil
	st.mu.Unlock()
	return err
}

// snapshot projects registered engines from one read under the store lock.
// It never writes the file, including when legacy Off needs a local origin.
func (s *desiredStateStore) snapshot(names []string) (engineintent.Snapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	if err != nil {
		return engineintent.Snapshot{}, err
	}
	out := engineintent.Snapshot{
		EnabledByEngine: make(map[string]bool, len(names)),
		OffByEngine:     make(map[string]engineintent.Origin),
	}
	for _, name := range names {
		enabled, known := state.Engines[name]
		out.EnabledByEngine[name] = known && enabled
		if known && !enabled {
			origin, present := state.OffBy[name]
			if !present {
				origin = intentLocal
			}
			out.OffByEngine[name] = engineintent.Origin(origin)
		}
	}
	if err := out.Validate(); err != nil {
		return engineintent.Snapshot{}, err
	}
	return out, nil
}

func (s *desiredStateStore) locked(engine string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	if err != nil {
		return false, err
	}
	return state.Locked[engine], nil
}

func (s *desiredStateStore) setLocked(engine string, locked bool) (intentChanged bool, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.path == "" {
		return false, errors.New("no user data directory is available")
	}
	state, err := s.load()
	if err != nil {
		return false, err
	}
	wasEnabled, known := state.Engines[engine]
	if locked {
		intentChanged = !known || wasEnabled || state.OffBy[engine] != intentLocal
		state.Locked[engine] = true
		state.setIntent(engine, false, intentLocal)
	} else {
		delete(state.Locked, engine)
	}
	return intentChanged, s.save(state)
}

// set saves engine's intent. Saving On clears the Off provenance; saving Off
// records origin, except that a paired node's Off never replaces the
// provenance of an engine already saved Off, so it cannot downgrade a local Off.
func (s *desiredStateStore) set(engine string, enabled bool, origin intentOrigin) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.path == "" {
		return errors.New("no user data directory is available")
	}
	state, err := s.load()
	if err != nil {
		return err
	}
	if enabled && state.Locked[engine] {
		return &localVetoError{code: codeEngineLocked, reason: engineLockedVeto(engine)}
	}
	state.setIntent(engine, enabled, origin)
	return s.save(state)
}

func (state *desiredStateFile) setIntent(engine string, enabled bool, origin intentOrigin) {
	// All persistence callers report the refusal before saving. Keep this
	// primitive fail-closed too, including checkpoint assembly.
	if enabled && state.Locked[engine] {
		return
	}
	was, known := state.Engines[engine]
	state.Engines[engine] = enabled
	switch {
	case enabled:
		delete(state.OffBy, engine)
	case origin == intentPeer && known && !was:
		// A paired node's Off over an Off keeps the provenance it had.
	default:
		state.OffBy[engine] = origin
	}
}

func (s *desiredStateStore) load() (desiredStateFile, error) {
	state := desiredStateFile{Engines: make(map[string]bool), OffBy: make(map[string]intentOrigin), Locked: make(map[string]bool)}
	if s.path == "" {
		return state, nil
	}
	data, err := os.ReadFile(s.path)
	if os.IsNotExist(err) {
		return state, nil
	}
	if err != nil {
		return state, fmt.Errorf("read desired engine state: %w", err)
	}
	if err := json.Unmarshal(data, &state); err != nil {
		return desiredStateFile{}, fmt.Errorf("decode desired engine state: %w", err)
	}
	if state.Engines == nil {
		state.Engines = make(map[string]bool)
	}
	if state.OffBy == nil {
		state.OffBy = make(map[string]intentOrigin)
	}
	if state.Locked == nil {
		state.Locked = make(map[string]bool)
	}
	for engine, port := range state.AllocatedPorts {
		if port < 1 || port > 65535 {
			return desiredStateFile{}, fmt.Errorf("invalid allocated port for %s", engine)
		}
	}
	for engine, evidence := range state.ProcessStops {
		if !evidence.valid() {
			return desiredStateFile{}, fmt.Errorf("invalid process stop evidence for %s", engine)
		}
	}
	for engine, locked := range state.Locked {
		if locked && state.Engines[engine] {
			return desiredStateFile{}, fmt.Errorf("invalid desired engine state: %s is both locked and On", engine)
		}
	}
	if cp := state.PendingSwitch; cp != nil {
		if ports := state.PendingSwitchPorts; ports != nil && (ports.Source < 0 || ports.Source > 65535 || ports.Target < 0 || ports.Target > 65535 || ports.SourcePID < 0 || len(ports.SourceRunID) > settings.MaxSwitchRequestIDBytes) {
			return desiredStateFile{}, errors.New("invalid pending switch endpoint")
		}
		if err := validSwitchEnvelope(cp.SwitchPrepareRequest); err != nil {
			return desiredStateFile{}, fmt.Errorf("invalid pending switch: %w", err)
		}
		switch cp.Phase {
		case settings.SwitchCheckpointReserved, settings.SwitchCheckpointStopPending, settings.SwitchCheckpointSourceOff, settings.SwitchCheckpointStartPending, settings.SwitchCheckpointTargetCommitted, settings.SwitchCheckpointReleased:
		default:
			return desiredStateFile{}, errors.New("invalid pending switch phase")
		}
	}
	return state, nil
}

func (s *desiredStateStore) save(state desiredStateFile) error {
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create desired engine state directory: %w", err)
	}
	if err := writeJSONAtomic(s.path, state); err != nil {
		return fmt.Errorf("save desired engine state: %w", err)
	}
	return nil
}

// setDesiredEnabled saves engine's intent; origin is who asked, which an Off
// records (desiredStateStore.set).
func (e *Executor) setDesiredEnabled(engine string, enabled bool, origin intentOrigin) error {
	if err := e.desired.set(engine, enabled, origin); err != nil {
		return fmt.Errorf("persist %s desired state: %w", engine, err)
	}
	// The broker mirrors saved intent into the proxy so admission can refuse a
	// request for an engine saved Off instead of asking to wake it.
	e.emitIntent()
	return nil
}

// RestoreEnabled starts only engines with an explicit saved ON state. Each
// engine rechecks its intent while holding the lifecycle lock so a concurrent
// explicit OFF always wins.
func (e *Executor) RestoreEnabled(ctx context.Context) error {
	var wg sync.WaitGroup
	errs := make(chan error, len(e.reg.Names()))
	for _, engine := range e.reg.Names() {
		enabled, known, err := e.desired.get(engine)
		if err != nil {
			errs <- fmt.Errorf("restore %s: %w", engine, err)
			continue
		}
		if !known || !enabled {
			continue
		}
		wg.Add(1)
		go func(engine string) {
			defer wg.Done()
			if err := e.restoreEnabled(ctx, engine); err != nil {
				errs <- fmt.Errorf("restore %s: %w", engine, err)
			}
		}(engine)
	}
	wg.Wait()
	close(errs)
	var joined []error
	for err := range errs {
		joined = append(joined, err)
	}
	return errors.Join(joined...)
}

func (e *Executor) restoreEnabled(ctx context.Context, engine string) error {
	done, err := e.admitMutation(engine)
	if err != nil {
		// A pending switch must be reconciled before automatic restore.
		var busy *switchBusyError
		if errors.As(err, &busy) {
			return nil
		}
		return err
	}
	defer done()
	st, err := e.state(engine)
	if err != nil {
		return err
	}
	// RestoreEnabled passes the manager's lifetime context through to hashing.
	if err := prepareStrataState(ctx, st); err != nil {
		return err
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	enabled, known, err := e.desired.get(engine)
	if err != nil || !known || !enabled {
		return err
	}
	return e.doStart(ctx, st, engine, startOpts{ReclaimOwned: true})
}

// reclaimOrphan is shared by restore and switch recovery/start. It runs before
// readiness/adoption, since an interrupted managed child may be unhealthy.
// Callers hold opMu and register startCancel before entering this owner.
func (e *Executor) reclaimOrphan(ctx context.Context, st *engineState, engine string, port int) (bool, error) {
	return e.reclaimOrphanForPID(ctx, st, engine, port, 0)
}

// requiredPID narrows automatic source cleanup to its original child. The
// managed-install/image, previous-reclaim and exit predicates remain mandatory.
func (e *Executor) reclaimOrphanForPID(ctx context.Context, st *engineState, engine string, port, requiredPID int) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	st.mu.Lock()
	binPath, installDir, rt, owned := st.binPath, st.installDir, st.plat.Runtime, st.proc != nil
	st.mu.Unlock()
	if owned || port <= 0 || rt.modeOrDefault() != "process" || !isManagedInstallPath(binPath, installDir) {
		return false, nil
	}
	// A child may still be initializing when its manager dies. Listener-only
	// inspection would miss it and let a cold child race its eventual bind.
	// Saved identity locates a candidate, but only the current image predicate
	// authorizes a signal, just as it does for an already listening orphan.
	evidence, known, err := e.desired.processStop(engine)
	if err != nil {
		return false, err
	}
	var pid int
	var image string
	var found bool
	if known && evidence.PID > 0 && evidence.Port == port && isOurEngineImage(evidence.Bin, binPath) {
		image = imagePathForPID(evidence.PID)
		if isOurEngineImage(image, binPath) {
			pid, found = evidence.PID, true
		}
	}
	if !found {
		pid, image, found = pidOnPort(port)
	}
	if !found || !isOurEngineImage(image, binPath) {
		return false, nil
	}
	if requiredPID > 0 && pid != requiredPID {
		return false, fmt.Errorf("cannot automatically reclaim %s: source process changed", engine)
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	st.mu.Lock()
	previousPID := st.reclaimPID
	st.mu.Unlock()
	if previousPID != 0 && previousPID != pid && !orphanPIDExited(previousPID) {
		return true, fmt.Errorf("reclaim owned %s orphan: previous pid %d termination is unverified", engine, previousPID)
	}
	st.mu.Lock()
	st.reclaimPID = pid
	st.mu.Unlock()
	if err := terminatePIDContext(ctx, pid, stopGrace(rt)); err != nil {
		err = fmt.Errorf("reclaim owned %s orphan: %w", engine, err)
		e.failOrphanReclaim(st, engine, err)
		return true, err
	}
	if err := e.confirmOrphanStoppedContext(ctx, engine, pid, port, rt); err != nil {
		return true, err
	}
	e.markStopped(st, engine)
	if rt.Port == 0 {
		return true, e.clearAllocatedPort(st, engine)
	}
	return true, nil
}

func (e *Executor) failOrphanReclaim(st *engineState, engine string, err error) {
	st.mu.Lock()
	st.reclaimFailed = true
	st.running, st.healthy, st.adopted = false, false, false
	st.runID = ""
	if st.healthStop != nil {
		st.healthStop()
		st.healthStop = nil
	}
	st.mu.Unlock()
	e.reportStartFailed(engine, err)
	e.emitState(engine)
}

// confirmOrphanStopped checks termination separately from sending signals so
// failure handling can be exercised without an unkillable process fixture.
func (e *Executor) confirmOrphanStopped(engine string, pid, port int, rt Runtime) error {
	return e.confirmOrphanStoppedContext(context.Background(), engine, pid, port, rt)
}

func (e *Executor) confirmOrphanStoppedContext(ctx context.Context, engine string, pid, port int, rt Runtime) error {
	var err error
	if rt.Ready != nil {
		// The signal owner has already consumed any graceful interval.
		if !e.waitUnavailableContext(ctx, rt.Ready, port, 2*time.Second) {
			err = fmt.Errorf("reclaim owned %s orphan: listener on port %d did not stop", engine, port)
		}
	}
	if err == nil && !orphanPIDExited(pid) {
		err = fmt.Errorf("reclaim owned %s orphan: pid %d did not stop", engine, pid)
	}
	if ctx.Err() != nil {
		err = fmt.Errorf("reclaim owned %s orphan: %w", engine, ctx.Err())
	}
	if err != nil {
		st, stateErr := e.state(engine)
		if stateErr != nil {
			e.mu.Lock()
			st = e.stopOnly[engine]
			e.mu.Unlock()
		}
		if st != nil {
			st.mu.Lock()
			if st.reclaimPID == 0 {
				st.reclaimPID = pid
			}
			st.mu.Unlock()
			e.failOrphanReclaim(st, engine, err)
		}
	}
	return err
}
