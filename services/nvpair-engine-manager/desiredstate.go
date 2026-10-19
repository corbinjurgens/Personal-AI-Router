// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	settings "nvpair-shared/enginesettings"
	"os"
	"path/filepath"
	"sync"
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
	// OffBy records who saved each engine Off: intentLocal or intentPeer. An
	// engine saved Off with no entry predates provenance and counts as local.
	OffBy map[string]intentOrigin `json:"offBy,omitempty"`
}

// Allocated ports are execution evidence, excluded from configuration tokens.
// Keep them beside the shared checkpoint so crash cleanup can find a port-zero
// engine's actual listener after the in-memory lifecycle state is lost.
type switchCheckpointPorts struct {
	Source int `json:"source"`
	Target int `json:"target"`
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

// offLocally reports whether engine is saved Off by this node itself: an Off
// saved by a local client, or one saved before provenance was recorded.
func (s *desiredStateStore) offLocally(engine string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	state, err := s.load()
	if err != nil {
		return false, err
	}
	enabled, known := state.Engines[engine]
	return known && !enabled && state.OffBy[engine] != intentPeer, nil
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
	state.setIntent(engine, enabled, origin)
	return s.save(state)
}

func (state *desiredStateFile) setIntent(engine string, enabled bool, origin intentOrigin) {
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
	state := desiredStateFile{Engines: make(map[string]bool), OffBy: make(map[string]intentOrigin)}
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
	if cp := state.PendingSwitch; cp != nil {
		if ports := state.PendingSwitchPorts; ports != nil && (ports.Source < 0 || ports.Source > 65535 || ports.Target < 0 || ports.Target > 65535) {
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
	st.opMu.Lock()
	defer st.opMu.Unlock()
	enabled, known, err := e.desired.get(engine)
	if err != nil || !known || !enabled {
		return err
	}
	return e.doStart(ctx, st, engine, startOpts{})
}
