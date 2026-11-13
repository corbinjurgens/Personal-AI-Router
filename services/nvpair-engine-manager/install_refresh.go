// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"
)

// A managed install lives in one fixed directory per engine, so a PAIR build
// that moves an engine's pin must replace the files there. The marker records
// which pinned install block produced them; when the manifest's block differs,
// the new build is staged beside the managed directory and verified by
// detection, then swapped in after the engine has stopped. The old directory is
// kept until the new build has started, and restored if the new build is not
// detected in place or fails to start.
//
// Only PAIR's own pinned installs are refreshed: an engine whose executable is
// outside the managed directory (a user-selected build or an override path),
// one without PAIR's marker, a script install, and a layout whose model store
// lives inside the managed directory are all left alone.

func refreshFailedID(engine string) string { return "engine-manager:install-refresh-failed:" + engine }

// installRecipeID identifies a pinned install block: its checksum-pinned
// downloads and the command that unpacks them. Unpinned blocks have none.
func installRecipeID(inst *Install) string {
	if inst == nil || len(inst.Script) > 0 || inst.Fetch == nil && len(inst.Artifacts) == 0 {
		return ""
	}
	if inst.Fetch != nil && inst.Fetch.SHA256 == "" {
		return ""
	}
	for _, artifact := range inst.Artifacts {
		if artifact.SHA256 == "" {
			return ""
		}
	}
	data, err := json.Marshal(struct {
		Fetch     *Fetch            `json:"fetch,omitempty"`
		Artifacts []InstallArtifact `json:"artifacts,omitempty"`
		Run       []string          `json:"run,omitempty"`
	}{inst.Fetch, inst.Artifacts, inst.Run})
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// installMarkerFor describes the install now in st's install directory.
func (e *Executor) installMarkerFor(ctx context.Context, st *engineState, engine string) installMarker {
	marker := installMarker{Engine: engine, Recipe: installRecipeID(st.plat.Install)}
	if st.manifest.FamilyOrDefault() == "llamacpp" {
		if build, err := nativeBuildID(ctx, st); err == nil {
			marker.Build = build
		} else {
			slog.Warn("installed build identity unavailable", "engine", engine, "err", err)
		}
	}
	return marker
}

func (e *Executor) refreshDirs(engine string) (staging, previous string) {
	return filepath.Join(e.baseDir, "."+engine+".staging"), filepath.Join(e.baseDir, "."+engine+".previous")
}

// managedInstallStale reports whether PAIR's own managed install was produced
// by a different pinned install block than the manifest now declares. The
// caller has just run Detect, which resolved st.binPath.
func (e *Executor) managedInstallStale(st *engineState) bool {
	if e.baseDir == "" {
		return false
	}
	recipe := installRecipeID(st.plat.Install)
	if recipe == "" {
		return false
	}
	st.mu.Lock()
	binPath, installDir, modelsDir := st.binPath, st.installDir, st.modelsDir
	st.mu.Unlock()
	if !isManagedInstallPath(binPath, installDir) || modelsDir != "" && isManagedInstallPath(modelsDir, installDir) {
		return false
	}
	marker, ok := readInstallMarker(installDir)
	return ok && marker.Recipe != recipe
}

// prepareManagedRefresh repairs what an interrupted refresh left behind. The
// caller holds st.refreshMu, so no refresh is in flight. When the managed
// directory is missing but the previous sibling exists, a refresh stopped
// between its two renames, and the previous build is put back. Any remaining
// staging or previous sibling beside an existing managed directory is a
// leftover and is removed. Uninstall removes both siblings too, so a leftover
// cannot resurrect an uninstalled engine.
func (e *Executor) prepareManagedRefresh(engine string, st *engineState) {
	if e.baseDir == "" {
		return
	}
	_, previous := e.refreshDirs(engine)
	if _, err := os.Stat(st.installDir); errors.Is(err, os.ErrNotExist) {
		if _, err := os.Stat(previous); err != nil {
			return
		}
		if err := renameRetrying(previous, st.installDir); err != nil {
			slog.Error("could not restore the managed install after an interrupted refresh", "engine", engine, "err", err)
			return
		}
		slog.Warn("restored the managed install after an interrupted refresh", "engine", engine)
	} else if err != nil {
		return
	}
	e.clearRefreshDirs(engine, st)
}

// clearRefreshDirs removes refresh leftovers after an uninstall.
func (e *Executor) clearRefreshDirs(engine string, st *engineState) {
	if e.baseDir == "" {
		return
	}
	staging, previous := e.refreshDirs(engine)
	for _, dir := range []string{staging, previous} {
		if err := removeTreePreserving(dir, st.modelsDir); err != nil {
			slog.Warn("could not remove a managed install refresh leftover", "engine", engine, "path", dir, "err", err)
		}
	}
}

// renameRetrying renames, retrying briefly: on Windows a scanner or indexer
// can hold a freshly written file open for a moment.
func renameRetrying(from, to string) error {
	var err error
	for attempt := 0; attempt < 10; attempt++ {
		if err = os.Rename(from, to); err == nil {
			return nil
		}
		time.Sleep(200 * time.Millisecond)
	}
	return err
}

func (e *Executor) reportRefreshFailed(st *engineState, engine string, err error) {
	slog.Error("managed install refresh failed; keeping the installed build", "engine", engine, "err", err)
	e.emitInstallFrame(engine, "failed", -1, err.Error())
	e.reporter.report(serviceError{
		ID: refreshFailedID(engine), Message: fmt.Sprintf("Could not update %s to this PAIR build's pinned version; the installed build is kept: %v", st.manifest.DisplayName, err),
		Severity: "warning", Action: "retry", EngineType: engine, Operation: "install",
	})
}

// stageManagedRefresh downloads and unpacks the manifest's pinned build into
// the staging sibling and checks that it is detected there. It does not touch
// the installed build, so startup restoration runs it without st.opMu. False
// means the failure was reported and the installed build is kept.
func (e *Executor) stageManagedRefresh(ctx context.Context, st *engineState, engine string) bool {
	staging, _ := e.refreshDirs(engine)
	err := func() error {
		if err := os.RemoveAll(staging); err != nil {
			return fmt.Errorf("clear staging directory: %w", err)
		}
		if err := os.MkdirAll(staging, 0o755); err != nil {
			return fmt.Errorf("create staging directory: %w", err)
		}
		if err := e.runPinnedInstall(ctx, engine, st.plat.Install, staging); err != nil {
			return err
		}
		staged := &engineState{manifest: st.manifest, plat: st.plat, installDir: staging, modelsDir: st.modelsDir}
		if staged.detectExecutable() == "" || !staged.detectRequiredFiles() {
			return fmt.Errorf("the new build was not detected after staging")
		}
		if err := writeInstallMarker(staging, e.installMarkerFor(ctx, staged, engine)); err != nil {
			return fmt.Errorf("record the new build: %w", err)
		}
		return nil
	}()
	if err != nil {
		_ = os.RemoveAll(staging)
		e.reportRefreshFailed(st, engine, err)
		return false
	}
	return true
}

// refreshFinish completes a swapped refresh after the caller's start attempt.
// start restarts the engine on the old build after a rollback.
type refreshFinish func(startErr error, start func() error) error

// swapManagedRefresh stops the installed build and moves the staged one into
// the managed directory, keeping the old build as the previous sibling. The
// caller holds mutation admission and st.opMu, starts the engine and passes the
// outcome to the returned finish: a started build deletes the previous one; a
// build that fails to start is replaced by the previous one, which is started.
// A nil finish means the swap failed, was reported, and the old build is in
// place.
func (e *Executor) swapManagedRefresh(ctx context.Context, st *engineState, engine string) refreshFinish {
	staging, previous := e.refreshDirs(engine)
	err := e.swapStagedBuild(ctx, st, engine)
	_, _ = e.Detect(engine)
	if err != nil {
		// Staging may hold the only copy when a rollback could not put the
		// old build back; it is removed only beside an existing install.
		if _, statErr := os.Stat(st.installDir); statErr == nil {
			_ = os.RemoveAll(staging)
		}
		e.reportRefreshFailed(st, engine, err)
		return nil
	}
	return func(startErr error, start func() error) error {
		var veto *localVetoError
		if startErr == nil || errors.As(startErr, &veto) || isStaleApproval(startErr) || ctx.Err() != nil || start == nil {
			// A vetoed, cancelled or skipped start says nothing about the new
			// build; detection is then its only check. A stale-approval refusal
			// is the expected result of a pin move for an approved
			// configuration: the new build stays, and the existing reviewed
			// apply re-approves it (V44).
			if err := removeTreePreserving(previous, st.modelsDir); err != nil {
				slog.Warn("previous managed build could not be removed; the next refresh retries", "engine", engine, "err", err)
			}
			e.reporter.clear(refreshFailedID(engine))
			e.emitInstallProgress(engine, "done", 100)
			slog.Info("managed install refreshed to the pinned build", "engine", engine)
			return startErr
		}
		if err := e.restorePreviousBuild(st, engine); err != nil {
			e.reportRefreshFailed(st, engine, fmt.Errorf("the new build did not start (%v), and %w", startErr, err))
			return startErr
		}
		e.reportRefreshFailed(st, engine, fmt.Errorf("the new build did not start: %w", startErr))
		return start()
	}
}

func (e *Executor) swapStagedBuild(ctx context.Context, st *engineState, engine string) error {
	staging, previous := e.refreshDirs(engine)
	// The swap needs the old build stopped: a running executable holds its
	// directory on Windows. Stop exactly as Uninstall does.
	st.mu.Lock()
	port := st.port
	st.mu.Unlock()
	presence := e.reconcilePresence(ctx, engine, st, true, port, false)
	st.mu.Lock()
	owned := st.proc != nil
	st.mu.Unlock()
	if presence.Occupied && !presence.Identified && !owned {
		return fmt.Errorf("port %d is occupied by an unidentified service", port)
	}
	if err := e.doStop(st, engine); err != nil {
		return fmt.Errorf("stop the installed build: %w", err)
	}
	if err := removeTreePreserving(previous, st.modelsDir); err != nil {
		return fmt.Errorf("clear the previous build directory: %w", err)
	}
	st.mu.Lock()
	st.withdrawOwnerKnowledgeLocked()
	st.mu.Unlock()
	if err := renameRetrying(st.installDir, previous); err != nil {
		return fmt.Errorf("move the installed build aside: %w", err)
	}
	if err := renameRetrying(staging, st.installDir); err != nil {
		if restoreErr := renameRetrying(previous, st.installDir); restoreErr != nil {
			return fmt.Errorf("move the new build into place: %w; restoring the old build also failed: %v", err, restoreErr)
		}
		return fmt.Errorf("move the new build into place: %w", err)
	}
	if ok, _ := e.Detect(engine); !ok {
		if err := e.restorePreviousBuild(st, engine); err != nil {
			return fmt.Errorf("the new build was not detected in place, and %w", err)
		}
		return fmt.Errorf("the new build was not detected in place; the old build was restored")
	}
	return nil
}

// restorePreviousBuild puts the previous build back after the new one failed
// in place. The new build returns to staging and is removed.
// It never leaves the managed directory empty: without a previous build the
// new one stays, and a failed restore moves the new build back into place.
func (e *Executor) restorePreviousBuild(st *engineState, engine string) error {
	staging, previous := e.refreshDirs(engine)
	if info, err := os.Stat(previous); err != nil || !info.IsDir() {
		return fmt.Errorf("the old build is missing (%v); the new build stays in place", err)
	}
	if err := e.doStop(st, engine); err != nil {
		return fmt.Errorf("stopping the new build failed: %w", err)
	}
	st.mu.Lock()
	st.withdrawOwnerKnowledgeLocked()
	st.mu.Unlock()
	if err := renameRetrying(st.installDir, staging); err != nil {
		return fmt.Errorf("moving the new build out failed: %w", err)
	}
	if err := renameRetrying(previous, st.installDir); err != nil {
		if backErr := renameRetrying(staging, st.installDir); backErr != nil {
			return fmt.Errorf("restoring the old build failed: %w; moving the new build back also failed and it remains in %s: %v", err, staging, backErr)
		}
		_, _ = e.Detect(engine)
		return fmt.Errorf("restoring the old build failed: %w; the new build stays in place", err)
	}
	_ = os.RemoveAll(staging)
	_, _ = e.Detect(engine)
	return nil
}
