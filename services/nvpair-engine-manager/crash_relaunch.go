// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"log/slog"
	"time"
)

type engineRelaunch struct {
	cancel context.CancelFunc
	done   chan struct{}
}

type crashRelaunchContextKey struct{}

func automaticCrashRelaunch(ctx context.Context) bool {
	return ctx.Value(crashRelaunchContextKey{}) != nil
}

// An episode ends at readiness; a later unexpected exit starts a fresh episode.
// Failed starts consume the bounded sequence, leaving exited-unexpectedly set.
func (e *Executor) scheduleCrashRelaunch(st *engineState, engine string) {
	st.mu.Lock()
	if st.stopping || st.running || st.relaunch != nil || e.shuttingDown.Load() {
		st.mu.Unlock()
		return
	}
	ctx, cancel := context.WithCancel(context.WithValue(context.Background(), crashRelaunchContextKey{}, true))
	recovery := &engineRelaunch{cancel: cancel, done: make(chan struct{})}
	st.relaunch = recovery
	st.mu.Unlock()
	go e.runCrashRelaunch(ctx, st, engine, recovery)
}

func (e *Executor) runCrashRelaunch(ctx context.Context, st *engineState, engine string, recovery *engineRelaunch) {
	defer func() {
		recovery.cancel()
		st.mu.Lock()
		if st.relaunch == recovery {
			st.relaunch = nil
		}
		st.mu.Unlock()
		close(recovery.done)
	}()
	backoff := [...]time.Duration{5 * time.Second, 30 * time.Second, 2 * time.Minute}
	for attempt := 0; attempt < len(backoff); attempt++ {
		timer := time.NewTimer(backoff[attempt])
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
		if ctx.Err() != nil || e.shuttingDown.Load() {
			return
		}
		if paused, _ := e.nodeProjection.pausedLocally(); paused {
			return
		}
		enabled, known, err := e.desired.get(engine)
		if err != nil || !known || !enabled {
			return
		}
		st.mu.Lock()
		if st.running || st.relaunch != recovery {
			st.mu.Unlock()
			return
		}
		st.mu.Unlock()
		// This is the existing restore owner, including current On intent under
		// opMu, pending-switch deference and precise reclaim/adoption fencing.
		err = e.restoreEnabled(ctx, engine)
		st.mu.Lock()
		ready := st.running
		st.mu.Unlock()
		if ready || ctx.Err() != nil {
			return
		}
		if err == nil {
			return // Reserved switch or saved Off: do not bypass its owner.
		}
		slog.Warn("unexpected-exit restore failed", "engine", engine, "attempt", attempt+1, "err", err)
	}
}

// Returns the cancelled episode's completion so shutdown can join it outside
// opMu. Keep the cancelled handle until completion so shutdown can still join
// an episode already cancelled by a pause projection.
func cancelCrashRelaunch(st *engineState) <-chan struct{} {
	st.mu.Lock()
	recovery := st.relaunch
	// A crash watcher may not have scheduled yet when an intentional stop
	// arrives. Fence that late scheduling even when there is no live process.
	st.stopping = true
	st.mu.Unlock()
	if recovery == nil {
		return nil
	}
	recovery.cancel()
	return recovery.done
}

// A pause projection may arrive after the retry's first check, while restore
// awaits opMu or prepares inputs. Cancel that operation through its parent
// context; explicit client starts and other engines' work are not affected.
func (e *Executor) cancelPausedCrashRelaunches() {
	e.mu.Lock()
	states := make([]*engineState, 0, len(e.engines))
	for _, st := range e.engines {
		states = append(states, st)
	}
	e.mu.Unlock()
	for _, st := range states {
		st.mu.Lock()
		recovery := st.relaunch
		if recovery != nil {
			st.stopping = true
		}
		st.mu.Unlock()
		if recovery != nil {
			recovery.cancel()
		}
	}
}
