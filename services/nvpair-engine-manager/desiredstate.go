// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"nvpair-shared/durablefile"
	"nvpair-shared/modelconfig"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	desiredStateFileName   = "engine-state.json"
	restoreInstalledMethod = "engine:restore-installed"
	prepareShutdownMethod  = "engine:prepare-shutdown"
)

type desiredStateFile struct {
	// AllocatedPorts locates process-mode port-zero launches after manager death.
	// A port is evidence only; reclaim still checks the current managed image.
	AllocatedPorts map[string]int `json:"allocatedPorts,omitempty"`
	// ProcessStops survives removal of a manifest's host block. No arguments,
	// environment, commands or credentials belong in this cleanup-only record.
	ProcessStops      map[string]processStopEvidence `json:"processStops,omitempty"`
	PendingDiagnostic *diagnosticCheckpoint          `json:"pendingDiagnostic,omitempty"`
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

// desiredStateStore retains process cleanup evidence and allocated endpoints.
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

func (s *desiredStateStore) allocatedPort(engine string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	return state.AllocatedPorts[engine], err
}

// setAllocatedPort records the endpoint before spawn. Zero clears execution
// evidence after a clean stop.
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

func (s *desiredStateStore) load() (desiredStateFile, error) {
	data, err := s.readDocument()
	if err != nil {
		return desiredStateFile{}, err
	}
	return decodeDesiredState(data)
}

// Callers hold mu. An absent document has empty saved semantics, but reading it
// never creates the owner directory or the file.
func (s *desiredStateStore) readDocument() ([]byte, error) {
	if s.path == "" {
		return []byte(`{}`), nil
	}
	f, err := durablefile.Open(s.path)
	if os.IsNotExist(err) {
		return []byte(`{}`), nil
	}
	if err != nil {
		return nil, fmt.Errorf("read desired engine state: %w", err)
	}
	data, readErr := io.ReadAll(io.LimitReader(f, desiredStateMaxBytes+1))
	closeErr := f.Close()
	if readErr != nil || closeErr != nil {
		return nil, fmt.Errorf("read desired engine state: %w", errors.Join(readErr, closeErr))
	}
	if len(data) > desiredStateMaxBytes {
		return nil, errors.New("desired engine state exceeds size limit")
	}
	return data, nil
}

func decodeDesiredState(data []byte) (desiredStateFile, error) {
	if len(data) > desiredStateMaxBytes {
		return desiredStateFile{}, errors.New("desired engine state exceeds size limit")
	}
	// Retired intent/transaction fields remain ignored. The surviving diagnostic
	// obligation is strict, so unknown fields cannot silently lose cleanup work.
	var fields map[string]json.RawMessage
	if err := modelconfig.DecodeStrict(data, &fields); err != nil || fields == nil {
		return desiredStateFile{}, errors.New("decode desired engine state: expected an object")
	}
	for field := range fields {
		if strings.EqualFold(field, "pendingDiagnostic") && field != "pendingDiagnostic" {
			return desiredStateFile{}, errors.New("decode desired engine state: invalid diagnostic field spelling")
		}
	}
	var state desiredStateFile
	if err := json.Unmarshal(data, &state); err != nil {
		return desiredStateFile{}, fmt.Errorf("decode desired engine state: %w", err)
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
	if raw, present := fields["pendingDiagnostic"]; present {
		var checkpoint diagnosticCheckpoint
		if err := modelconfig.DecodeStrict(raw, &checkpoint); err != nil {
			return desiredStateFile{}, fmt.Errorf("decode pending diagnostic: %w", err)
		}
		if err := checkpoint.validate(); err != nil {
			return desiredStateFile{}, fmt.Errorf("invalid pending diagnostic: %w", err)
		}
		state.PendingDiagnostic = &checkpoint
	}
	return state, nil
}

func (s *desiredStateStore) save(state desiredStateFile) error {
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	if _, err := decodeDesiredState(data); err != nil {
		return err
	}
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create desired engine state directory: %w", err)
	}
	// A checkpoint publication with uncertain durability must never authorize
	// the next diagnostic effect. Keep the existing semantics for other writes.
	if state.PendingDiagnostic != nil {
		if err := engineFileReplacer.WriteFile(s.path, data, 0o600); err != nil {
			return fmt.Errorf("save desired engine state: %w", err)
		}
		return nil
	}
	if err := writeJSONAtomic(s.path, state); err != nil {
		return fmt.Errorf("save desired engine state: %w", err)
	}
	return nil
}

// RestoreInstalled starts installed engines subject to activation admission.
func (e *Executor) RestoreInstalled(ctx context.Context) error {
	var wg sync.WaitGroup
	errs := make(chan error, len(e.reg.Names()))
	for _, engine := range e.reg.Names() {
		wg.Add(1)
		go func(engine string) {
			defer wg.Done()
			if err := e.restoreInstalled(ctx, engine); err != nil {
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

func (e *Executor) restoreInstalled(ctx context.Context, engine string) error {
	done, err := e.admitMutation(engine)
	if err != nil {
		// Global mutation admission also fences automatic restore.
		var busy *mutationBusyError
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
	// RestoreInstalled passes the manager's lifetime context through to hashing.
	if err := prepareStrataState(ctx, st); err != nil {
		return err
	}
	// A PAIR upgrade may have moved this engine's pin. Download and stage the
	// new build before taking opMu, so settings reads that need opMu are not
	// held up by the download. A failed refresh keeps the installed build,
	// which starts as before.
	staged := false
	if installed, _ := e.Detect(engine); installed && e.managedInstallStale(st) {
		staged = e.stageManagedRefresh(ctx, st, engine)
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	e.recoverInterruptedRefresh(engine, st)
	installed, err := e.Detect(engine)
	if err != nil || !installed {
		return err
	}
	var finish refreshFinish
	if staged && e.managedInstallStale(st) {
		finish = e.swapManagedRefresh(ctx, st, engine)
	} else if staged {
		staging, _ := e.refreshDirs(engine)
		_ = os.RemoveAll(staging)
	}
	start := func() error { return e.doStart(ctx, st, engine, startOpts{ReclaimOwned: true}) }
	err = start()
	if finish != nil {
		err = finish(err, start)
	}
	st.mu.Lock()
	st.restoreApprovalRefused = isStaleApproval(err)
	st.mu.Unlock()
	return err
}

// reclaimOrphan is shared by restore and start. It runs before
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
	if owned || port <= 0 || !isManagedInstallPath(binPath, installDir) {
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
	st.withdrawOwnerKnowledgeLocked()
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
	st.withdrawOwnerKnowledgeLocked()
	st.reclaimFailed = true
	st.running, st.healthy, st.adopted = false, false, false
	st.runID = ""
	st.buildInfo, st.buildID = "", ""
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
