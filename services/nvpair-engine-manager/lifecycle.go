// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/httpcon"
)

const (
	unavailableConfirmations  = 3
	engineIdentityProbeHeader = "X-NVPAIR-Engine-Identity-Probe"
	maxProbeJSONBytes         = 1 << 20
)

type listenerProbeResult uint8

const (
	listenerProbeIndeterminate listenerProbeResult = iota
	listenerProbeReachable
	listenerProbeRefused
)

// waitDetect polls Detect until it reports `want` or the timeout
// elapses. Installers/uninstallers (notably Windows Inno setups) often
// finish their file work asynchronously, so a single immediate check
// races them.
func (e *Executor) waitDetect(engine string, want bool, timeout time.Duration) bool {
	return waitForDetection(func() bool {
		ok, _ := e.Detect(engine)
		return ok == want
	}, timeout)
}

func waitForDetection(detected func() bool, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for {
		if detected() {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(500 * time.Millisecond)
	}
}

// startOpts carries per-call start overrides. Zero values mean "use the
// manifest": Port 0 => the manifest port (which may itself be 0 =
// auto-assign a free loopback port); Bind "" => the manifest's
// runtime.bind (which itself defaults to loopback).
type startOpts struct {
	// Private cold activation repeats authority after lifecycle admission.
	boundGuard       func() error
	boundOperationID string
	RequireOwned     bool
	ReclaimOwned     bool
	// Automatic switches retain their deadline through failed-start cleanup.
	BoundedCleanup bool
	Port           int
	Bind           string
	// Origin is who asked for the start. A paired node's start (intentPeer)
	// is checked against the local veto again under the lifecycle lock.
	Origin intentOrigin
}

// effectiveBind picks the listen address substituted as {host}: a per-call
// override wins, else the manifest's runtime.bind, else loopback (the safe
// default for an engine that does not opt into LAN exposure).
func effectiveBind(manifestBind, override string) string {
	switch {
	case override != "":
		return override
	case manifestBind != "":
		return manifestBind
	default:
		return "127.0.0.1"
	}
}

// Start launches the engine and waits for its readiness probe, then
// begins the health loop. No-op if already running. It branches on the
// runtime mode: "process" (spawn + own the process) or "command" (run
// bring-up commands; liveness comes from the probe).
func (e *Executor) Start(ctx context.Context, engine string) error {
	return e.StartWith(ctx, engine, startOpts{})
}

// StartWith is Start with per-call overrides. The overrides are not
// persisted — a service restart reverts to the manifest. A paired node's start
// rechecks target pause and lock after taking the lifecycle lock. Saved Off
// alone does not prohibit a fresh explicit start.
func (e *Executor) StartWith(ctx context.Context, engine string, opts startOpts) error {
	var done func()
	var err error
	if opts.boundGuard != nil {
		done, err = e.admitBoundEngineMutation(engine, opts.boundOperationID)
	} else {
		done, err = e.admitMutation(engine)
	}
	if err != nil {
		return err
	}
	defer done()
	if opts.boundGuard != nil {
		if err := opts.boundGuard(); err != nil {
			return err
		}
	}
	st, err := e.state(engine)
	if err != nil {
		return err
	}
	if err := prepareStrataState(ctx, st); err != nil {
		return err
	}
	if opts.boundGuard != nil {
		if err := lockBoundEngineLifecycle(ctx, st); err != nil {
			return err
		}
	} else {
		st.opMu.Lock()
	}
	defer st.opMu.Unlock()
	if opts.boundGuard != nil {
		if err := opts.boundGuard(); err != nil {
			return err
		}
	}
	if opts.Origin == intentPeer {
		if err := e.remoteVeto(engine); err != nil {
			return err
		}
	}
	if err := e.doStart(ctx, st, engine, opts); err != nil {
		return err
	}
	if err := e.setDesiredEnabled(engine, true, opts.Origin); err != nil {
		if opts.BoundedCleanup {
			return errors.Join(err, e.doStopContext(ctx, st, engine))
		}
		return errors.Join(err, e.doStop(st, engine))
	}
	return nil
}

// doStart performs the start. Callers must hold st.opMu.
func (e *Executor) doStart(ctx context.Context, st *engineState, engine string, opts startOpts) (err error) {
	ctx, finish := e.startContext(ctx, st)
	cleanupCtx := context.Background()
	if opts.BoundedCleanup {
		cleanupCtx = ctx
	}
	defer func() {
		err = vetoCause(ctx, err)
		finish()
	}()
	if err := e.activationVeto(engine); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	pathInstalled, _ := e.Detect(engine)
	if !pathInstalled && len(st.plat.DetectRequired) > 0 {
		return fmt.Errorf("engine %q is not installed: mandatory detection files are incomplete", engine)
	}
	st.mu.Lock()
	running := st.running
	adopted := st.adopted
	if running && !adopted {
		st.mu.Unlock()
		return nil
	}
	rt := st.plat.Runtime
	port := st.port
	st.mu.Unlock()
	if port == 0 && rt.modeOrDefault() == "process" && opts.ReclaimOwned {
		port, err = e.desired.allocatedPort(engine)
		if err != nil {
			return err
		}
		st.mu.Lock()
		st.port = port
		st.mu.Unlock()
	}
	reclaimPort := port
	if opts.Port > 0 {
		port = opts.Port
	}
	if port == 0 {
		p, err := freePort()
		if err != nil {
			return fmt.Errorf("allocate port: %w", err)
		}
		port = p
	}
	if err := e.reservedPortError(port); err != nil {
		return err
	}
	if opts.ReclaimOwned && reclaimPort > 0 {
		reclaimed, err := e.reclaimOrphan(ctx, st, engine, port)
		if err != nil {
			return err
		}
		if reclaimed {
			opts.RequireOwned = true
		}
	}
	if err := ctx.Err(); err != nil {
		// Successful reclaim already verified termination and cleared its fence.
		// Cancellation here prevents launch without invalidating that evidence.
		return err
	}
	st.mu.Lock()
	fenced, reclaimPID := st.reclaimFailed, st.reclaimPID
	st.mu.Unlock()
	presence := e.reconcilePresence(activationAdoptionContext(ctx), engine, st, pathInstalled, port, true)
	if err := ctx.Err(); err != nil {
		return err
	}
	// A pause that landed while the identity probe ran refused its adoption.
	if err := e.activationVeto(engine); err != nil {
		return err
	}
	st.mu.Lock()
	installed := st.installed
	st.mu.Unlock()
	if !installed && len(st.plat.DetectRequired) > 0 {
		return fmt.Errorf("engine %q is not installed: mandatory detection files are incomplete", engine)
	}
	if fenced && (presence.Occupied || !orphanPIDExited(reclaimPID)) {
		return fmt.Errorf("cannot start engine %q: an owned orphan reclaim failed; stop it before retrying", engine)
	}
	if presence.Identified {
		if opts.RequireOwned {
			return fmt.Errorf("cannot start owned engine: another process already owns the server port")
		}
		slog.Info("adopting already-running external engine", "engine", engine, "port", port)
		st.mu.Lock()
		st.gen++
		gen := st.gen
		recovery := st.relaunch
		st.relaunch = nil
		st.mu.Unlock()
		if recovery != nil {
			recovery.cancel()
		}
		e.reporter.clear(startFailedID(engine))
		e.reporter.clear(exitedID(engine))
		e.reporter.clear(unhealthyID(engine))
		if rt.Health != nil {
			hctx, cancel := context.WithCancel(context.Background())
			st.mu.Lock()
			if st.healthStop != nil {
				st.healthStop()
			}
			st.healthStop = cancel
			st.mu.Unlock()
			go e.runHealth(hctx, st, engine, port, gen)
		}
		return nil
	}
	if presence.Occupied && (rt.modeOrDefault() == "process" || len(st.plat.DetectRequired) > 0) {
		return fmt.Errorf("cannot start engine %q on port %d: the port is occupied by a service that did not identify as %s", engine, port, st.manifest.DisplayName)
	}
	if !pathInstalled {
		return fmt.Errorf("engine %q is not installed", engine)
	}
	vars := st.pathVars()
	vars["host"] = effectiveBind(rt.Bind, opts.Bind)
	vars["port"] = strconv.Itoa(port)
	if rt.CLI != "" {
		vars["cli"] = expandPath(rt.CLI)
	}

	st.mu.Lock()
	st.port = port
	st.stopping = false
	st.mu.Unlock()

	if err := e.bringUpWithCleanup(ctx, cleanupCtx, st, engine, rt, port, vars); err != nil {
		return err
	}
	if len(st.plat.DetectRequired) > 0 {
		if installed, _ := e.Detect(engine); !installed {
			st.mu.Lock()
			st.running = true
			st.mu.Unlock()
			return errors.Join(fmt.Errorf("engine %q is not installed: mandatory detection files are incomplete", engine), e.doStopContext(cleanupCtx, st, engine))
		}
	}
	if err := e.activationVeto(engine); err != nil {
		st.mu.Lock()
		st.running = true
		st.mu.Unlock()
		return errors.Join(err, e.doStopContext(cleanupCtx, st, engine))
	}

	// Linearize automatic readiness with the pause projection. A pause that
	// wins this gate must not be followed by a running publication, even if
	// the last readiness response was already successful. No I/O under gates.
	e.nodeProjection.mu.Lock()
	st.mu.Lock()
	readyErr := ctx.Err()
	if readyErr == nil && e.nodeProjection.projection != nil && e.nodeProjection.projection.ManagementBlocked {
		readyErr = &localVetoError{code: codeLocalVeto, reason: nodePausedVeto}
	}
	if readyErr != nil {
		// bringUp succeeded: this start owns the launched process/control
		// command. Let the existing stop owner reap it before returning.
		// This internal flag is never emitted as a running notification.
		st.running = true
		st.mu.Unlock()
		e.nodeProjection.mu.Unlock()
		return errors.Join(readyErr, e.doStopContext(cleanupCtx, st, engine))
	}
	st.running = true
	st.runID = newEngineRunID()
	st.healthy = true
	st.adopted = false
	st.reclaimFailed = false
	st.reclaimPID = 0
	st.gen++
	gen := st.gen
	recovery := st.relaunch
	st.relaunch = nil
	st.mu.Unlock()
	e.nodeProjection.mu.Unlock()
	if recovery != nil {
		recovery.cancel()
	}
	e.reporter.clear(startFailedID(engine))
	e.reporter.clear(exitedID(engine))
	e.reporter.clear(unhealthyID(engine))
	e.emitState(engine)

	st.mu.Lock()
	proc := st.proc
	st.mu.Unlock()
	if proc != nil {
		e.watch(st, engine, proc)
	}

	if rt.Health != nil {
		hctx, cancel := context.WithCancel(context.Background())
		st.mu.Lock()
		st.healthStop = cancel
		st.mu.Unlock()
		go e.runHealth(hctx, st, engine, port, gen)
	}
	return nil
}

// Register before reclaim as well as launch so StopAll can cancel either stage.
// Callers hold opMu for the entire lifetime of this registration.
func (e *Executor) startContext(ctx context.Context, st *engineState) (context.Context, func()) {
	ctx, cancel := context.WithCancelCause(ctx)
	st.mu.Lock()
	st.startCancel = cancel
	st.mu.Unlock()
	if e.shuttingDown.Load() {
		cancel(nil)
	}
	return ctx, func() {
		cancel(nil)
		st.mu.Lock()
		st.startCancel = nil
		st.mu.Unlock()
	}
}

func (e *Executor) bringUp(ctx context.Context, st *engineState, engine string, rt Runtime, port int, vars map[string]string) error {
	return e.bringUpWithCleanup(ctx, context.Background(), st, engine, rt, port, vars)
}

func (e *Executor) bringUpWithCleanup(ctx, cleanupCtx context.Context, st *engineState, engine string, rt Runtime, port int, vars map[string]string) error {
	if len(rt.ModelLoads) != 0 && (st.manifest.FamilyOrDefault() != "llamacpp" || rt.modeOrDefault() != "process") {
		return modelLoadError("", "modelLoads", "Typed loads require an owned llama.cpp process.")
	}
	if strataServingRuntime(st, rt.Serving) && rt.modeOrDefault() != "process" {
		return servingError("Prepared Strata requires an owned process launch.")
	}
	if rt.modeOrDefault() == "command" {
		return e.bringUpCommand(ctx, cleanupCtx, st, engine, rt, port, vars)
	}
	return e.bringUpProcessWithCleanup(ctx, cleanupCtx, st, engine, rt, port, vars, prepareServingSnapshot)
}

// bringUpProcess spawns the engine as a foreground process this service
// owns, then waits for readiness.
func (e *Executor) bringUpProcess(ctx context.Context, st *engineState, engine string, rt Runtime, port int, vars map[string]string) error {
	return e.bringUpProcessWithSnapshot(ctx, st, engine, rt, port, vars, prepareServingSnapshot)
}

// Snapshot preparation is passed explicitly so fake-process tests can exercise
// an input change between preparation and the final check without timing races.
func (e *Executor) bringUpProcessWithSnapshot(ctx context.Context, st *engineState, engine string, rt Runtime, port int, vars map[string]string, prepareSnapshot func(string, []byte) (string, func(), error)) error {
	return e.bringUpProcessWithCleanup(ctx, context.Background(), st, engine, rt, port, vars, prepareSnapshot)
}

func (e *Executor) bringUpProcessWithCleanup(ctx, cleanupCtx context.Context, st *engineState, engine string, rt Runtime, port int, vars map[string]string, prepareSnapshot func(string, []byte) (string, func(), error)) error {
	rt.modelLoadOwner = e
	recheckContent := func() error { return nil }
	if len(rt.ModelLoads) != 0 {
		if e.downloads == nil {
			return modelLoadError("", "artifacts", "Model content admission is unavailable.")
		}
		release, recheck, err := e.modelLoadContentRead(engine)
		if err != nil {
			return modelLoadError("", "artifacts", "Model content is busy or unsafe.")
		}
		defer release()
		recheckContent = recheck
	}
	serving := preparedServingRuntime(st, rt)
	if llamaServingRuntime(st, rt) {
		rt = llamaServingBase(rt)
	}
	st.mu.Lock()
	binPath := st.binPath
	st.mu.Unlock()
	launch, err := resolveProcessLaunch(rt, binPath, vars)
	if err != nil {
		if serving {
			return servingError("NE launch resolution failed; check the launch options.")
		}
		return err
	}
	if err := validateEffectiveLaunch(rt, launch, vars["host"], vars["port"]); err != nil {
		if serving {
			return servingError("NE launch validation failed; check the launch options.")
		}
		return err
	}
	var prepared *llamaServingLaunch
	var releaseSnapshot func()
	originalLaunch := launch
	digest := launchDigest(launch)
	baseDigest := digest
	if serving {
		prepared, err = compilePreparedServing(st, launch, rt, true)
		if err != nil {
			return fmt.Errorf("NE preset validation: %w", err)
		}
		var snapshot string
		if pending := rt.servingSnapshot; pending != nil && pending.Prepared.Digest == prepared.Digest {
			snapshot, releaseSnapshot = pending.Path, pending.Release
			st.mu.Lock()
			st.plat.Runtime.servingSnapshot = nil
			st.mu.Unlock()
		} else {
			st.mu.Lock()
			pending := st.plat.Runtime.servingSnapshot
			st.plat.Runtime.servingSnapshot = nil
			st.mu.Unlock()
			if pending != nil {
				pending.Release()
			}
			snapshot, releaseSnapshot, err = prepareSnapshot(filepath.Join(e.baseDir, "serving", engine), prepared.Snapshot)
			if err != nil {
				return fmt.Errorf("NE snapshot preparation: %w", err)
			}
		}
		defer func() {
			if releaseSnapshot != nil {
				releaseSnapshot()
			}
		}()
		snapshotBytes, _, err := servingSource(snapshot)
		if err != nil || string(snapshotBytes) != string(prepared.Snapshot) {
			return servingError("NE snapshot validation: prepared snapshot changed before spawn.")
		}
		if err := recheckPreparedServing(st, launch, rt, prepared); err != nil {
			return fmt.Errorf("NE source recheck: %w", err)
		}
		digest = servingLaunchDigest(originalLaunch, prepared.Digest)
		launch = preparedSnapshotCommand(st, launch, snapshot)
		if prepared.Credential != "" {
			// Added after every source check, which rejects any STRATA_* name
			// a manifest or the inherited environment supplies.
			env := maps.Clone(launch.Env)
			if env == nil {
				env = map[string]string{}
			}
			env["STRATA_API_KEY"] = prepared.Credential
			launch.Env = env
		}
	}

	diagnostics := newStartupOutput(launchDiagnosticArgs(rt), launch.Env)
	defer diagnostics.close()
	if err := ctx.Err(); err != nil {
		return err
	}
	// Persist before spawn so even a hard kill during readiness leaves enough
	// endpoint evidence for the existing managed-path/image reclaim predicate.
	allocated := rt.Port == 0
	if allocated {
		if err := e.desired.setAllocatedPort(engine, port); err != nil {
			return fmt.Errorf("persist %s allocated port: %w", engine, err)
		}
		st.mu.Lock()
		st.portCleanupPending = false
		st.mu.Unlock()
	}
	processBin, err := exec.LookPath(launch.Bin)
	if err != nil {
		return err
	}
	processBin, err = filepath.Abs(processBin)
	if err != nil {
		return err
	}
	address, addressErr := probeListenerAddress(rt.Ready, port)
	if addressErr != nil {
		address = net.JoinHostPort(vars["host"], strconv.Itoa(port))
	}
	evidence := processStopEvidence{Bin: processBin, Port: port, Address: address}
	if err := e.desired.setProcessStop(engine, evidence); err != nil {
		return fmt.Errorf("persist %s process stop evidence: %w", engine, err)
	}
	if err := recheckContent(); err != nil {
		return err
	}
	if serving {
		if err := recheckPreparedServing(st, originalLaunch, rt, prepared); err != nil {
			return fmt.Errorf("NE pre-spawn source recheck: %w", err)
		}
	}
	proc, err := startManagedProc(launch.Bin, launch.Args, launch.Env, func(stream, line string) {
		if serving {
			if prepared.Credential != "" {
				line = strings.ReplaceAll(line, prepared.Credential, "[redacted]")
			}
			line = redactServingLine(line, st.modelsDir)
			st.logs.append(stream, line)
			_, _ = diagnostics.Write([]byte(line + "\n"))
			return
		}
		_, _ = diagnostics.Write([]byte(line + "\n"))
		// Vendor diagnostics may echo arbitrary custom arguments. Keep those
		// out of exported engine logs; report exit/readiness failures separately.
		if !rt.hasCustomLaunch() {
			st.logs.append(stream, line)
		}
	})
	if err != nil {
		if allocated {
			err = errors.Join(err, e.clearAllocatedPort(st, engine))
		}
		werr := fmt.Errorf("start %s: %w", engine, err)
		if serving {
			werr = servingError("NE spawn: process could not be started.")
		}
		e.reportStartFailedUnlessShuttingDown(ctx, engine, werr)
		return werr
	}
	if releaseSnapshot != nil {
		release := releaseSnapshot
		releaseSnapshot = nil
		go func() {
			<-proc.done
			release()
		}()
	}
	if !serving {
		digest = launchDigest(launch)
	}
	st.mu.Lock()
	st.proc = proc
	st.launchDigest = digest
	st.plat.Runtime.servingLaunch = prepared
	if len(rt.ModelLoads) != 0 {
		st.plat.Runtime.modelLoadOwner = e
	}
	st.plat.Runtime.servingCommand = originalLaunch
	st.plat.Runtime.servingBaseDigest = baseDigest
	st.plat.Runtime.servingDrift = ""
	st.mu.Unlock()
	evidence.PID = proc.cmd.Process.Pid
	if err := e.desired.setProcessStop(engine, evidence); err != nil {
		if stopErr := proc.stopContext(cleanupCtx, stopGrace(rt)); stopErr != nil {
			st.mu.Lock()
			st.running, st.healthy, st.stopping = true, false, true
			st.mu.Unlock()
			return errors.Join(fmt.Errorf("persist %s process identity: %w", engine, err), stopErr)
		}
		e.markStopped(st, engine)
		return fmt.Errorf("persist %s process identity: %w", engine, err)
	}

	readyCtx, readyCancel := context.WithCancel(ctx)
	defer readyCancel()
	go func() {
		select {
		case <-proc.done:
			readyCancel()
		case <-readyCtx.Done():
		}
	}()
	readyErr := e.waitReady(readyCtx, rt.Ready, port, rt.AuthStatuses)
	select {
	case <-proc.done:
		readyErr = fmt.Errorf("process exited before readiness; check the launch options")
		if serving {
			readyErr = fmt.Errorf("NE readiness: process exited before readiness (exit code %d)", proc.cmd.ProcessState.ExitCode())
		}
	default:
		if serving && readyErr != nil {
			readyErr = fmt.Errorf("NE readiness: readiness wait failed: %w", readyErr)
		}
	}
	if err := readyErr; err != nil {
		// Clean up the failed start before returning its single readiness error.
		st.mu.Lock()
		st.stopping = true
		st.mu.Unlock()
		if stopErr := proc.stopContext(cleanupCtx, stopGrace(rt)); stopErr != nil {
			st.mu.Lock()
			st.running, st.healthy = true, false
			st.mu.Unlock()
			return errors.Join(err, stopErr) // preserve the owned handle and endpoint
		}
		st.mu.Lock()
		st.proc = nil
		st.mu.Unlock()
		if allocated {
			err = errors.Join(err, e.clearAllocatedPort(st, engine))
		}
		werr := fmt.Errorf("engine %q did not become ready: %w", engine, diagnostics.failure(err))
		e.reportStartFailedUnlessShuttingDown(ctx, engine, werr)
		e.emitState(engine)
		return werr
	}
	return nil
}

// bringUpCommand runs the manifest's ordered start commands (for a
// daemon-style engine such as LM Studio), then waits for readiness.
// There is no owned process — liveness comes from the probe.
func (e *Executor) bringUpCommand(ctx, cleanupCtx context.Context, st *engineState, engine string, rt Runtime, port int, vars map[string]string) error {
	st.mu.Lock()
	st.proc = nil
	st.mu.Unlock()
	startAttempted := false
	cleanup := func(startErr error) error {
		if !startAttempted {
			return startErr
		}
		if stopErr := e.runCommandStopContext(cleanupCtx, st, engine, rt, port); stopErr != nil {
			// A non-zero stop CLI can still have taken the daemon down. Confirm
			// that before surfacing a cleanup failure.
			if rt.Ready != nil && e.waitUnavailableContext(cleanupCtx, rt.Ready, port, time.Second) {
				return startErr
			}
			return errors.Join(startErr, fmt.Errorf("clean up failed %s start: %w", engine, stopErr))
		}
		return startErr
	}
	for index := range rt.Start {
		launch, err := resolveRuntimeCommand(rt, index, vars)
		if err != nil {
			e.reportStartFailedUnlessShuttingDown(ctx, engine, err)
			return cleanup(err)
		}
		if launch.Bin == "" {
			continue
		}
		if rt.EditableLaunch != nil && index == rt.EditableLaunch.StartIndex {
			if err := validateEffectiveLaunch(rt, launch, vars["host"], vars["port"]); err != nil {
				return cleanup(err)
			}
		}
		argv := append([]string{launch.Bin}, launch.Args...)
		run := e.runCommand
		if rt.LaunchArgs != nil || rt.LaunchEnv != nil || len(launch.Env) > 0 {
			run = func(ctx context.Context, argv []string) error {
				return runPrivateLaunchCommand(ctx, argv, launch.Env, launchDiagnosticArgs(rt))
			}
		}
		// A control CLI may detach its daemon before returning, then fail or
		// be cancelled. Attempt official cleanup once any start command runs,
		// even when that command never reports success.
		startAttempted = true
		if err := run(ctx, argv); err != nil {
			werr := fmt.Errorf("start command failed: %w", err)
			werr = e.diagnoseStartFailure(ctx, st, port, err, werr)
			e.reportStartFailedUnlessShuttingDown(ctx, engine, werr)
			return cleanup(werr)
		}
	}
	if err := e.waitReady(ctx, rt.Ready, port, rt.AuthStatuses); err != nil {
		werr := fmt.Errorf("engine %q did not become ready: %w", engine, err)
		werr = e.diagnoseStartFailure(ctx, st, port, err, werr)
		e.reportStartFailedUnlessShuttingDown(ctx, engine, werr)
		e.emitState(engine)
		return cleanup(werr)
	}
	return nil
}

// Capture vendor startup diagnostics without interpreting engine options.
func runPrivateLaunchCommand(ctx context.Context, argv []string, environment map[string]string, privateArgs []string) error {
	if len(argv) == 0 {
		return fmt.Errorf("launch executable is missing")
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	diagnostics := newStartupOutput(privateArgs, environment)
	defer diagnostics.close()
	cmd.Stdout, cmd.Stderr = diagnostics, diagnostics
	cmd.Env = childEnv(environment)
	configureSysProcAttr(cmd)
	if err := cmd.Run(); err != nil {
		return diagnostics.failure(fmt.Errorf("launch command failed: %w", err))
	}
	return nil
}

// watch reports an unexpected engine exit (one not initiated by Stop).
func (e *Executor) watch(st *engineState, engine string, proc *managedProc) {
	go func() {
		<-proc.done
		st.mu.Lock()
		stopping := st.stopping
		current := st.proc == proc
		if current {
			st.proc = nil
			st.running = false
			st.runID = ""
			st.healthy = false
			if st.healthStop != nil {
				st.healthStop()
				st.healthStop = nil
			}
		}
		st.mu.Unlock()
		if current && !stopping {
			e.reporter.report(serviceError{
				ID: exitedID(engine), Message: engine + " exited unexpectedly",
				Severity: "error", Action: "none", EngineType: engine,
			})
			e.emitState(engine)
			e.scheduleCrashRelaunch(st, engine)
		}
	}()
}

// Stop terminates a running engine. In process mode it signals the
// owned process (graceful then forced); in command mode it runs the
// manifest's stop command. No-op if not running. It is a local client's stop.
func (e *Executor) Stop(engine string) error {
	return e.StopFrom(engine, intentLocal)
}

// StopFrom is Stop on behalf of origin, which the saved Off records. A local
// stop is a local veto: it first cancels the paired nodes' accepted starts and
// loads of the engine, so one in progress does not hold the lifecycle lock
// until it finishes.
func (e *Executor) StopFrom(engine string, origin intentOrigin) error {
	return e.stopFrom(engine, origin, "")
}

// StopForSwitch applies a broker-admitted local Off to its remote reservation,
// after release or a rejected prepare it is a complete no-op. The broker owns
// durable discharge before relinquishing the original operation lock.
func (e *Executor) StopForSwitch(engine, operationID string) error {
	return e.stopFrom(engine, intentLocal, operationID)
}

func (e *Executor) stopFrom(engine string, origin intentOrigin, operationID string) error {
	return e.stopFromApplied(engine, origin, operationID, nil)
}

func (e *Executor) stopFromApplied(engine string, origin intentOrigin, operationID string, applied *bool) error {
	st, err := e.state(engine)
	if err != nil {
		return err
	}
	done, err := e.admitStopScoped(engine, origin, operationID, applied != nil)
	if err != nil {
		return err
	}
	if done == nil {
		return nil
	}
	if applied != nil {
		*applied = true
	}
	defer done()
	if origin == intentLocal {
		e.remoteWork.cancelEngine(engine, &localVetoError{code: codeLocalVeto, reason: engineOffVeto(engine)})
	}
	cancelCrashRelaunch(st)
	finishLoads := stopServingLoads(st)
	defer finishLoads()
	st.opMu.Lock()
	defer st.opMu.Unlock()
	stopErr := e.doStop(st, engine)
	// A user-initiated Stop is explicit OFF intent. Persist it even when
	// doStop declines a genuinely-foreign listener we can't terminate, so the
	// health loop and RestoreEnabled honor the OFF instead of flipping the
	// engine back on. Restart and StopAll call doStop directly and never
	// rewrite intent — shutdown must not erase the user's saved choice.
	setErr := e.setDesiredEnabled(engine, false, origin)
	return errors.Join(stopErr, setErr)
}

func (e *Executor) verifyStopped(engine string) (settings.SwitchEngineState, error) {
	st, err := e.state(engine)
	if err != nil {
		return settings.SwitchEngineState{}, err
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	e.desired.mu.Lock()
	state, err := e.desired.load()
	e.desired.mu.Unlock()
	if err != nil {
		return settings.SwitchEngineState{}, err
	}
	observed := e.switchObserved(st, switchIntent(state, engine), e.switchTerminated(st))
	if observed.Intent != settings.SwitchIntentOff || !observed.TerminationVerified {
		return observed, errors.New("local Off termination is unverified")
	}
	e.desired.mu.Lock()
	defer e.desired.mu.Unlock()
	state, err = e.desired.load()
	if err != nil {
		return observed, err
	}
	if cp := state.PendingSwitch; cp != nil && cp.Phase != settings.SwitchCheckpointReleased {
		if engine == cp.SourceEngine && cp.SourceLocalOff {
			cp.Source = observed
		}
		if engine == cp.TargetEngine && cp.TargetLocalOff {
			cp.Target = observed
		}
		if err := e.desired.save(state); err != nil {
			return observed, err
		}
	}
	return observed, nil
}

// doStop performs the stop. Callers must hold st.opMu.
func (e *Executor) doStop(st *engineState, engine string) error {
	return e.doStopContext(context.Background(), st, engine)
}

// Switch recovery carries its cancellation through fallback orphan cleanup.
func (e *Executor) doStopContext(ctx context.Context, st *engineState, engine string) (err error) {
	return e.doStopOwnedContext(ctx, st, engine, nil)
}

// stopOwnership pins the observed source incarnation through the stop effect.
// A switch must never turn a lost owned handle into permission to reclaim an
// adopted replacement. Nil keeps ordinary Stop and manual E4 semantics.
type stopOwnership struct {
	process *managedProc
	runID   string
}

type stopOwnershipChangedError struct{ engine string }

func (err *stopOwnershipChangedError) Error() string {
	return fmt.Sprintf("cannot automatically stop engine %q: source ownership changed", err.engine)
}

func (e *Executor) doStopOwnedContext(ctx context.Context, st *engineState, engine string, ownership *stopOwnership) (err error) {
	if err := ctx.Err(); err != nil {
		return err
	}
	st.mu.Lock()
	rt := st.plat.Runtime
	if ownership != nil && (st.proc != ownership.process || st.runID != ownership.runID || !st.running || st.adopted || st.reclaimFailed || st.reclaimPID != 0) {
		st.mu.Unlock()
		return &stopOwnershipChangedError{engine}
	}
	fenced, port, reclaimPID := st.reclaimFailed, st.port, st.reclaimPID
	needsPortCleanup := st.running || fenced || st.portCleanupPending
	st.mu.Unlock()
	defer func() {
		if err == nil && needsPortCleanup && rt.Port == 0 && rt.modeOrDefault() == "process" {
			err = e.clearAllocatedPort(st, engine)
		}
	}()
	if fenced {
		if !e.shuttingDown.Load() {
			reclaimed, err := e.reclaimOrphan(ctx, st, engine, port)
			if err != nil {
				return err
			}
			if reclaimed {
				return nil
			}
		}
		if !orphanPIDExited(reclaimPID) || rt.Ready == nil || !e.waitUnavailableContext(ctx, rt.Ready, port, time.Second) {
			return fmt.Errorf("cannot stop engine %q: orphan termination could not be verified", engine)
		}
		e.markStopped(st, engine)
		return nil
	}
	mode := rt.modeOrDefault()
	st.mu.Lock()
	if ownership != nil && (st.proc != ownership.process || st.runID != ownership.runID || !st.running || st.adopted || st.reclaimFailed || st.reclaimPID != 0) {
		st.mu.Unlock()
		return &stopOwnershipChangedError{engine}
	}
	if !st.running {
		st.mu.Unlock()
		return nil
	}
	proc := st.proc
	port = st.port
	adopted := st.adopted
	binPath := st.binPath
	if mode != "command" && proc == nil {
		st.mu.Unlock()
		if adopted {
			if rt.Ready != nil && e.waitUnavailableContext(ctx, rt.Ready, port, time.Second) {
				e.markStopped(st, engine)
				return nil
			}
			// The listener is still up. If it's a PAIR-managed orphan on our
			// own port — our engine binary left running by a prior run that
			// lost its handle — reclaim it by terminating the PID bound to the
			// port, so a user OFF actually takes effect instead of being
			// refused forever. This is precise to the port, so a genuine
			// desktop app on a different port (e.g. Ollama's own :11434 while
			// we manage :11435) is never touched; we decline only a process
			// whose image we can't confirm is ours.
			pid, image, ok := pidOnPort(port)
			if ok && isManagedInstallPath(binPath, st.installDir) && isOurEngineImage(image, binPath) {
				if err := ctx.Err(); err != nil {
					return err
				}
				st.mu.Lock()
				st.stopping = true
				st.reclaimPID = pid
				if st.healthStop != nil {
					st.healthStop()
					st.healthStop = nil
				}
				st.mu.Unlock()
				grace := stopGrace(rt)
				if err := terminatePIDContext(ctx, pid, grace); err != nil && ctx.Err() != nil {
					e.failOrphanReclaim(st, engine, err)
					return err
				}
				if !pidAlive(pid) {
					e.markStopped(st, engine)
					return nil
				}
				if rt.Ready == nil || e.waitUnavailableContext(ctx, rt.Ready, port, grace+2*time.Second) {
					e.markStopped(st, engine)
					return nil
				}
				if err := ctx.Err(); err != nil {
					e.failOrphanReclaim(st, engine, err)
					return err
				}
			}
			e.emitState(engine)
			if ok {
				return fmt.Errorf("cannot stop engine %q: it is running under external management (pid %d, %s); stop it in its own application, then retry", engine, pid, image)
			}
			return fmt.Errorf("cannot stop engine %q: it is running under external management; stop it in its own application, then retry", engine)
		}
		e.emitState(engine)
		return fmt.Errorf("cannot stop engine %q: no NVPAIR-owned process is available to stop", engine)
	}
	st.stopping = true
	if st.healthStop != nil {
		st.healthStop()
		st.healthStop = nil
	}
	st.mu.Unlock()

	if mode == "command" {
		sp := rt.Stop
		if sp == nil || len(sp.Cmd) == 0 {
			return e.reconcileFailedCommandStop(st, engine, true, fmt.Errorf("cannot stop engine %q: no stop command is configured", engine))
		}
		stopErr := e.runCommandStopContext(ctx, st, engine, rt, port)
		if stopErr != nil {
			if ownership != nil || ctx.Err() != nil {
				return stopErr // leave uncertain termination reserved
			}
			return e.reconcileFailedCommandStop(st, engine, rt.Ready == nil || !e.waitUnavailable(rt.Ready, port, time.Second), stopErr)
		}
	} else if proc != nil {
		if err := proc.stopContext(ctx, stopGrace(rt)); err != nil {
			return err
		}
	}

	e.markStopped(st, engine)
	return nil
}

// runCommandStop executes a command-mode engine's bounded official stop CLI
// and confirms that its listener closes. It is shared by normal Stop and by
// failed/cancelled startup cleanup after a start CLI has detached a daemon.
func (e *Executor) runCommandStop(st *engineState, engine string, rt Runtime, port int) error {
	return e.runCommandStopContext(context.Background(), st, engine, rt, port)
}

func (e *Executor) runCommandStopContext(ctx context.Context, st *engineState, engine string, rt Runtime, port int) error {
	sp := rt.Stop
	if sp == nil || len(sp.Cmd) == 0 {
		return fmt.Errorf("cannot stop engine %q: no stop command is configured", engine)
	}
	vars := st.pathVars()
	vars["port"] = strconv.Itoa(port)
	if rt.CLI != "" {
		vars["cli"] = expandPath(rt.CLI)
	}
	argv, err := resolveArgs(sp.Cmd, vars)
	if err != nil {
		return fmt.Errorf("resolve %s stop command: %w", engine, err)
	}
	for i := range argv {
		argv[i] = expandPath(argv[i])
	}
	// Bound the stop CLI (e.g. `lms server stop`): the broker no longer
	// force-kills engine-manager on a timeout, so an unbounded stop command
	// would wedge StopAll and, in turn, the whole app shutdown.
	stopCtx, cancelStop := context.WithTimeout(ctx, commandStopTimeout(rt))
	runErr := e.runCommand(stopCtx, argv)
	cancelStop()
	if runErr != nil {
		return fmt.Errorf("stop %s: %w", engine, runErr)
	}
	if rt.Ready != nil {
		grace := 5 * time.Second
		if sp.GraceS > 0 {
			grace = time.Duration(sp.GraceS) * time.Second
		}
		if !e.waitUnavailableContext(ctx, rt.Ready, port, grace) {
			return fmt.Errorf("stop %s: engine is still serving on port %d after %s", engine, port, grace)
		}
	}
	return nil
}

// stopGrace is how long to wait after a graceful stop before escalating to a
// forced kill, taken from the manifest's stop spec: default 5s, the spec's
// grace_s when set, or 0 when the spec asks for an immediate kill.
func stopGrace(rt Runtime) time.Duration {
	grace := 5 * time.Second
	if sp := rt.Stop; sp != nil {
		if sp.GraceS > 0 {
			grace = time.Duration(sp.GraceS) * time.Second
		}
		if strings.EqualFold(sp.Signal, "kill") {
			grace = 0
		}
	}
	return grace
}

// commandStopTimeout bounds how long a command-mode stop CLI is allowed to run
// before it is cancelled. Derived from the manifest grace with a floor, so a
// hung stop command can never wedge StopAll now that the broker no longer
// force-kills engine-manager on a timeout.
func commandStopTimeout(rt Runtime) time.Duration {
	t := stopGrace(rt)
	if t < 5*time.Second {
		t = 5 * time.Second
	}
	return t
}

func (e *Executor) markStopped(st *engineState, engine string) {
	st.mu.Lock()
	if st.healthStop != nil {
		st.healthStop()
		st.healthStop = nil
	}
	st.running = false
	st.runID = ""
	st.healthy = false
	st.stopping = true
	st.adopted = false
	st.proc = nil
	st.reclaimFailed = false
	st.reclaimPID = 0
	st.mu.Unlock()
	// A deliberately-stopped engine has no live unhealthy/crash condition.
	e.reporter.clear(unhealthyID(engine))
	e.reporter.clear(exitedID(engine))
	e.emitState(engine)
}

// waitUnavailable confirms that an engine's listener is gone before reporting
// it stopped. Readiness and reachability are deliberately separate here: a
// live listener returning 503 is unhealthy, not stopped. A successful control
// command is likewise not proof that its daemon actually exited.
func (e *Executor) waitUnavailable(p *Probe, port int, timeout time.Duration) bool {
	return e.waitUnavailableContext(context.Background(), p, port, timeout)
}

func (e *Executor) waitUnavailableContext(ctx context.Context, p *Probe, port int, timeout time.Duration) bool {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	misses := 0
	for {
		result := probeListener(ctx, p, port)
		if result == listenerProbeRefused {
			misses++
			if misses == unavailableConfirmations {
				return true
			}
		} else {
			// A reachable listener, timeout, or malformed/indeterminate target
			// cannot prove shutdown. Require a fresh consecutive sequence.
			misses = 0
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(100 * time.Millisecond):
		}
	}
}

// probeListener reports whether the transport endpoint described by a
// readiness probe accepts connections, explicitly refuses them, or cannot be
// determined. Callers must fail closed on indeterminate results.
func probeListener(ctx context.Context, p *Probe, port int) listenerProbeResult {
	addr, err := probeListenerAddress(p, port)
	if err != nil {
		return listenerProbeIndeterminate
	}
	conn, err := (&net.Dialer{}).DialContext(ctx, "tcp", addr)
	if err == nil {
		_ = conn.Close()
		return listenerProbeReachable
	}
	if ctx.Err() != nil {
		return listenerProbeIndeterminate
	}
	if !isConnectionRefused(err) {
		return listenerProbeIndeterminate
	}
	// Only an explicit connection refusal proves there is no listener. DNS,
	// routing, firewall, timeout, and resource errors remain indeterminate.
	return listenerProbeRefused
}

func isConnectionRefused(err error) bool {
	var errno syscall.Errno
	if !errors.As(err, &errno) {
		return false
	}
	if runtime.GOOS == "windows" {
		const wsaeconnrefused syscall.Errno = 10061
		return errno == wsaeconnrefused
	}
	return errno == syscall.ECONNREFUSED
}

func probeListenerAddress(p *Probe, port int) (string, error) {
	if p == nil {
		return "", errors.New("no probe configured")
	}
	vars := map[string]string{"port": strconv.Itoa(port)}
	if p.HTTP != "" {
		raw, err := resolvePlaceholders(p.HTTP, vars)
		if err != nil {
			return "", err
		}
		u, err := url.Parse(raw)
		if err != nil || u.Hostname() == "" {
			return "", fmt.Errorf("invalid HTTP probe URL %q", raw)
		}
		servicePort := u.Port()
		if servicePort == "" {
			switch strings.ToLower(u.Scheme) {
			case "http":
				servicePort = "80"
			case "https":
				servicePort = "443"
			default:
				return "", fmt.Errorf("HTTP probe URL %q has no port", raw)
			}
		}
		return net.JoinHostPort(u.Hostname(), servicePort), nil
	}
	if p.TCP != "" {
		addr, err := resolvePlaceholders(p.TCP, vars)
		if err != nil {
			return "", err
		}
		if _, _, err := net.SplitHostPort(addr); err != nil {
			return "", fmt.Errorf("invalid TCP probe address %q: %w", addr, err)
		}
		return addr, nil
	}
	return "", errors.New("probe has no HTTP or TCP endpoint")
}

// reconcileFailedCommandStop keeps the in-memory state aligned with the
// readiness endpoint after a stop command fails or fails to take effect.
func (e *Executor) reconcileFailedCommandStop(st *engineState, engine string, running bool, stopErr error) error {
	if !running {
		e.markStopped(st, engine)
		slog.Warn("engine stop command returned an error, but the endpoint is confirmed down", "engine", engine, "err", stopErr)
		return nil
	}
	st.mu.Lock()
	rt := st.plat.Runtime
	st.mu.Unlock()
	var port int
	var healthCtx context.Context
	var gen int64

	st.mu.Lock()
	port = st.port
	st.running = true
	// The stop did not take effect: the same process keeps its run id.
	if st.runID == "" {
		st.runID = newEngineRunID()
	}
	st.healthy = true
	st.stopping = false
	if rt.Health != nil {
		st.gen++
		gen = st.gen
		var cancel context.CancelFunc
		healthCtx, cancel = context.WithCancel(context.Background())
		st.healthStop = cancel
	}
	st.mu.Unlock()

	if healthCtx != nil {
		go e.runHealth(healthCtx, st, engine, port, gen)
	}
	e.emitState(engine)
	return stopErr
}

// Restart stops then starts the engine, holding the op lock across both
// so nothing can interleave between the stop and the start.
func (e *Executor) Restart(ctx context.Context, engine string) error {
	done, err := e.admitMutation(engine)
	if err != nil {
		return err
	}
	defer done()
	st, err := e.state(engine)
	if err != nil {
		return err
	}
	if err := prepareStrataState(ctx, st); err != nil {
		return err
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	if err := e.doStop(st, engine); err != nil {
		return err
	}
	if err := e.doStart(ctx, st, engine, startOpts{}); err != nil {
		return err
	}
	return e.setDesiredEnabled(engine, true, intentLocal)
}

// StopAll rejects new starts and terminates every running engine during
// shutdown without changing the user's saved ON/OFF intent.
func (e *Executor) StopAll() {
	e.shuttingDown.Store(true)
	e.mu.Lock()
	names := make([]string, 0, len(e.engines))
	for n := range e.engines {
		names = append(names, n)
	}
	e.mu.Unlock()
	// Stop engines concurrently. Each doStop can wait up to the manifest grace
	// (default 5s) for its engine to exit, so a sequential sweep would take
	// sum(grace) across engines and let a slow first engine delay — or, if the
	// parent severs us mid-sweep, entirely prevent — every later engine from
	// even being signaled. Running them in parallel bounds the whole sweep to
	// ~one grace and guarantees every engine gets its stop signal immediately.
	// Each engine still serializes on its own opMu, so this is safe.
	var wg sync.WaitGroup
	for _, n := range names {
		st, err := e.state(n)
		if err != nil {
			continue
		}
		wg.Add(1)
		go func(st *engineState, n string) {
			defer wg.Done()
			if recoveryDone := cancelCrashRelaunch(st); recoveryDone != nil {
				defer func() { <-recoveryDone }()
			}
			finishLoads := stopServingLoads(st)
			defer finishLoads()
			st.mu.Lock()
			cancel := st.startCancel
			st.mu.Unlock()
			if cancel != nil {
				cancel(nil)
			}
			st.opMu.Lock()
			defer st.opMu.Unlock()
			if err := e.doStop(st, n); err != nil {
				slog.Warn("engine stop during shutdown failed", "engine", n, "err", err)
			}
			st.mu.Lock()
			pending := st.plat.Runtime.servingSnapshot
			st.plat.Runtime.servingSnapshot = nil
			st.mu.Unlock()
			if pending != nil {
				pending.Release()
			}
		}(st, n)
	}
	wg.Wait()
}

func (e *Executor) runHealth(ctx context.Context, st *engineState, engine string, port int, gen int64) {
	h := st.plat.Runtime.Health
	interval := 5 * time.Second
	if h.IntervalS > 0 {
		interval = time.Duration(h.IntervalS) * time.Second
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	reported := false
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			ok := e.probe(ctx, h, port)
			st.mu.Lock()
			// Bail if stopped or if a newer start superseded this loop —
			// a stale loop must never write health for a later run.
			if ctx.Err() != nil || !st.running || st.gen != gen {
				st.mu.Unlock()
				return
			}
			was := st.healthy
			st.healthy = ok
			st.mu.Unlock()
			message := engine + " failed its health probe"
			if !ok && !reported {
				if d := e.diagnoseConnection(ctx, st, port); d != diagnosisNone {
					message = diagnosisMessage(st.manifest.DisplayName, d)
				}
			}
			st.mu.Lock()
			// Diagnosis can outlive a stop or restart. Publish only for the run
			// whose probe failed, ordered with lifecycle changes under mu.
			if ctx.Err() != nil || !st.running || st.gen != gen {
				st.mu.Unlock()
				return
			}
			switch {
			case !ok && !reported:
				e.reporter.report(serviceError{
					ID: unhealthyID(engine), Message: message,
					Severity: "warning", Action: "none", EngineType: engine,
				})
				reported = true
			case ok && reported:
				e.reporter.clear(unhealthyID(engine))
				reported = false
			}
			st.mu.Unlock()
			if was != ok {
				e.emitState(engine)
			}
		}
	}
}

func (e *Executor) waitReady(ctx context.Context, p *Probe, port int, authStatuses []int) error {
	if p == nil {
		return nil
	}
	timeout := 20 * time.Second
	if p.TimeoutS > 0 {
		timeout = time.Duration(p.TimeoutS) * time.Second
	}
	deadline := time.Now().Add(timeout)
	authCount := 0
	for {
		ok, status := e.probeStatus(ctx, p, port)
		if ok {
			return nil
		}
		if slices.Contains(authStatuses, status) {
			authCount++
			if authCount == 3 && ctx.Err() == nil {
				return errAuthRequired
			}
		} else {
			authCount = 0
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("not ready after %s", timeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(300 * time.Millisecond):
		}
	}
}

func (e *Executor) probe(ctx context.Context, p *Probe, port int) bool {
	ok, _ := e.probeStatus(ctx, p, port)
	return ok
}

func (e *Executor) probeStatus(ctx context.Context, p *Probe, port int) (bool, int) {
	if p == nil {
		return true, 0
	}
	vars := map[string]string{"port": strconv.Itoa(port)}
	if p.HTTP != "" {
		u, err := resolvePlaceholders(p.HTTP, vars)
		if err != nil {
			return false, 0
		}
		pctx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(pctx, http.MethodGet, u, nil)
		if err != nil {
			return false, 0
		}
		req.Header.Set(engineIdentityProbeHeader, "1")
		resp, err := e.client.Do(req)
		if err != nil {
			return false, 0
		}

		want := p.Status
		if want == 0 {
			want = 200
		}
		statusMatches := resp.StatusCode == want
		if p.JSONMatch == nil {
			httpcon.DrainAndClose(resp.Body)
			return statusMatches, resp.StatusCode
		}
		body, err := io.ReadAll(io.LimitReader(resp.Body, maxProbeJSONBytes+1))
		httpcon.DrainAndClose(resp.Body)
		if !statusMatches || err != nil || len(body) > maxProbeJSONBytes {
			return false, resp.StatusCode
		}
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(body, &obj); err != nil {
			return false, resp.StatusCode
		}
		value, ok := resolveObjectPath(obj, p.JSONMatch.Field)
		if !ok {
			return false, resp.StatusCode
		}
		var actual string
		return json.Unmarshal(value, &actual) == nil && actual == p.JSONMatch.Value, resp.StatusCode
	}
	if p.TCP != "" {
		addr, err := resolvePlaceholders(p.TCP, vars)
		if err != nil {
			return false, 0
		}
		pctx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		conn, err := (&net.Dialer{}).DialContext(pctx, "tcp", addr)
		if err != nil {
			return false, 0
		}
		conn.Close()
		return true, 0
	}
	return true, 0
}

func (e *Executor) reportStartFailed(engine string, err error) {
	e.reporter.report(serviceError{
		ID: startFailedID(engine), Message: err.Error(),
		Severity: "error", Action: "retry", EngineType: engine, Operation: "start",
	})
}

func (e *Executor) reportStartFailedUnlessShuttingDown(ctx context.Context, engine string, err error) {
	if ctx.Value(settingsApplyContextKey{}) != nil {
		return
	}
	// CommandContext can surface an exit status instead of context.Canceled on
	// Windows, so the canceled operation context is the authoritative signal.
	if e.shuttingDown.Load() && ctx.Err() != nil {
		return
	}
	e.reportStartFailed(engine, err)
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func fileExists(p string) bool {
	if p == "" {
		return false
	}
	_, err := os.Stat(p)
	return err == nil
}
