// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"runtime"
	"strings"
	"time"
)

const presenceRefusalWindow = 500 * time.Millisecond

// presenceResult distinguishes a positively identified engine from an
// occupied/indeterminate target. Callers that might spawn or install must fail
// closed when Occupied is true: a listener that did not pass the identity probe
// is not permission to overwrite it.
type presenceResult struct {
	Identified bool
	Occupied   bool
}

// Detect updates installation completeness while retaining the executable path
// independently, so missing configuration cannot defeat owned-process cleanup.
func (e *Executor) Detect(engine string) (bool, error) {
	st, err := e.state(engine)
	if err != nil {
		return false, err
	}
	binPath := st.detectExecutable()
	installed := binPath != "" && st.detectRequiredFiles()
	st.mu.Lock()
	st.installed = installed
	if binPath != "" {
		st.binPath = binPath
	}
	st.mu.Unlock()
	return installed, nil
}

// detectExecutable preserves detect's ordered alternatives. Required files
// never select the program used by launch or orphan reclamation.
func (st *engineState) detectExecutable() string {
	for _, p := range st.plat.Detect {
		if len(st.plat.DetectRequired) > 0 {
			if ep := detectionFile(p, st.pathVars()); ep != "" {
				return ep
			}
			continue
		}
		r, err := resolvePlaceholders(p, st.pathVars())
		if err != nil {
			continue // detect path references something other than a known directory
		}
		if ep := expandPath(r); fileExists(ep) {
			return ep
		}
	}
	return ""
}

func (st *engineState) detectRequiredFiles() bool {
	for _, template := range st.plat.DetectRequired {
		if detectionFile(template, st.pathVars()) == "" {
			return false
		}
	}
	return true
}

// detectionFile resolves mandatory evidence without letting an unresolved or
// empty environment reference collapse to an existing path suffix.
func detectionFile(template string, vars map[string]string) string {
	envPattern := unixEnvRe
	if runtime.GOOS == "windows" {
		envPattern = winEnvRe
	}
	for _, token := range envPattern.FindAllString(template, -1) {
		if strings.TrimSpace(os.Getenv(strings.Trim(token, "$%{}"))) == "" {
			return ""
		}
	}
	// Expand first so ${VAR} is not mistaken for a runner placeholder.
	p, err := resolvePlaceholders(expandPath(template), vars)
	if err != nil || strings.TrimSpace(p) == "" || strings.HasPrefix(p, "~") {
		return ""
	}
	info, err := os.Stat(p)
	if err != nil || !info.Mode().IsRegular() {
		return ""
	}
	return p
}

// Status returns a fresh snapshot (re-detecting installed-ness).
func (e *Executor) Status(engine string) (EngineStatus, error) {
	return e.StatusAtPort(engine, 0)
}

// StatusAtPort is Status with an optional one-shot probe port. A positive port
// lets upgrade orchestration identify a legacy listener before changing the
// persisted backend port; successful identification updates the live state.
func (e *Executor) StatusAtPort(engine string, probePort int) (EngineStatus, error) {
	st, err := e.state(engine)
	if err != nil {
		// Known engine with no host-platform block: return a shell status
		// (matching get-installed) rather than erroring, so status is
		// consistent for the same engine across methods.
		if m, ok := e.reg.Get(engine); ok {
			return e.unsupportedStatus(engine, m), nil
		}
		return EngineStatus{}, err
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	pathInstalled, _ := e.Detect(engine)
	st.mu.Lock()
	port := st.port
	st.mu.Unlock()
	if probePort > 0 {
		port = probePort
	}
	e.reconcilePresence(context.Background(), engine, st, pathInstalled, port, false)
	return e.snapshot(engine, st), nil
}

// GetInstalled lists every known engine with its current status.
func (e *Executor) GetInstalled() []EngineStatus {
	var out []EngineStatus
	for _, name := range e.reg.Names() {
		st, err := e.state(name)
		if err != nil {
			// Known engine with no block for this host: surface a shell
			// status so the UI can still list it as unavailable here.
			if m, ok := e.reg.Get(name); ok {
				out = append(out, e.unsupportedStatus(name, m))
			}
			continue
		}
		st.opMu.Lock()
		pathInstalled, _ := e.Detect(name)
		st.mu.Lock()
		port := st.port
		st.mu.Unlock()
		e.reconcilePresence(context.Background(), name, st, pathInstalled, port, false)
		out = append(out, e.snapshot(name, st))
		st.opMu.Unlock()
	}
	return out
}

// Logs returns the engine's captured-output ring.
func (e *Executor) Logs(engine string) ([]LogLine, error) {
	st, err := e.state(engine)
	if err != nil {
		return nil, err
	}
	return st.logs.snapshot(), nil
}

// Errors returns the recent structured-error ring.
func (e *Executor) Errors() []serviceError {
	return e.reporter.snapshot()
}

func (e *Executor) snapshot(engine string, st *engineState) EngineStatus {
	locked := e.lockedStatus(engine)
	st.mu.Lock()
	defer st.mu.Unlock()
	return EngineStatus{
		Engine:      engine,
		DisplayName: st.manifest.DisplayName,
		Installed:   st.installed,
		Running:     st.running,
		Healthy:     st.healthy,
		Port:        st.port,
		Locked:      locked,
	}
}

func (e *Executor) lockedStatus(engine string) *bool {
	locked, err := e.desired.locked(engine)
	if err != nil {
		return nil
	}
	return &locked
}

// reconcilePresence reconciles filesystem detection with a fixed-port engine
// service. A successful engine-specific readiness probe is authoritative even
// when the binary lives outside NVPAIR's managed paths. Conversely, readiness
// failure is not proof of shutdown: an HTTP 503 still has a live listener.
// An adopted service is only marked stopped after repeated explicit connection
// refusals, preventing a transient failed readiness response from clearing it.
//
// allowWhileStopping is true only for an explicit Start. Passive status checks
// must not undo a deliberate OFF by re-adopting a service behind st.stopping.
func (e *Executor) reconcilePresence(ctx context.Context, engine string, st *engineState, pathInstalled bool, port int, allowWhileStopping bool) presenceResult {
	executablePresent := st.detectExecutable() != ""
	st.mu.Lock()
	running := st.running
	adopted := st.adopted
	proc := st.proc
	ready := st.plat.Runtime.Ready
	fenced := st.reclaimFailed
	st.mu.Unlock()
	if fenced {
		// A surviving failed reclaim must never become a healthy adoption,
		// including through explicit Start's allowWhileStopping path.
		if ready != nil && e.waitUnavailableContext(ctx, ready, port, presenceRefusalWindow) {
			return presenceResult{}
		}
		return presenceResult{Occupied: true}
	}

	// A command-mode engine needs its control CLI. A compatible HTTP endpoint
	// alone (for example another OpenAI server on LM Studio's port) is not an
	// installation and must not suppress the installer.
	if !pathInstalled && len(st.plat.DetectRequired) == 0 && st.plat.Runtime.modeOrDefault() != "process" {
		return presenceResult{}
	}

	if port == 0 || ready == nil || proc != nil || (running && !adopted) {
		return presenceResult{Identified: running, Occupied: running}
	}
	// Readiness can establish a running service, but cannot supply declared
	// mandatory files. Keep liveness truthful without promoting installation.
	serviceInstalled := pathInstalled || len(st.plat.DetectRequired) == 0

	probeCtx, cancel := context.WithTimeout(ctx, presenceRefusalWindow)
	listener := probeListener(probeCtx, ready, port)
	cancel()
	if listener == listenerProbeRefused {
		if !running || !adopted || !e.waitUnavailable(ready, port, presenceRefusalWindow) {
			return presenceResult{}
		}
		st.mu.Lock()
		changed := st.running || st.healthy || st.adopted || st.installed != pathInstalled
		if st.healthStop != nil {
			st.healthStop()
			st.healthStop = nil
		}
		st.installed = pathInstalled
		st.running = false
		st.runID = ""
		st.healthy = false
		st.adopted = false
		st.proc = nil
		if !executablePresent {
			// A service-only adoption has no managed image. Never leave a stale
			// path that orphan reclamation could terminate.
			st.binPath = ""
		}
		st.mu.Unlock()
		if changed {
			e.emitState(engine)
		}
		return presenceResult{}
	}
	if listener == listenerProbeIndeterminate {
		// DNS/routing/resource failures cannot prove the port is free. Preserve
		// existing state and prevent install/start from writing over it.
		if running && adopted {
			st.mu.Lock()
			st.installed = serviceInstalled
			if !executablePresent {
				st.binPath = ""
			}
			st.mu.Unlock()
		}
		return presenceResult{Occupied: true}
	}

	identified := e.probe(ctx, ready, port)
	if len(st.plat.DetectRequired) > 0 {
		// The probe may outlive a file change. Refresh mandatory evidence after
		// network I/O before publishing either healthy or transient adoption.
		serviceInstalled, _ = e.Detect(engine)
	}
	if !identified {
		if running && adopted {
			st.mu.Lock()
			changed := st.healthy
			st.installed = serviceInstalled
			st.healthy = false
			if !executablePresent {
				st.binPath = ""
			}
			st.mu.Unlock()
			if changed {
				e.emitState(engine)
			}
			return presenceResult{Occupied: true}
		}
		// A listener is present but did not identify as this engine. Do not
		// adopt it, spawn over it, or download a second copy for this port.
		return presenceResult{Occupied: true}
	}

	// A successful probe is not permission to publish an adoption after its
	// caller was cancelled. Order activation-owned adoption (automatic recovery
	// and explicit start) against pause just like owned readiness; the external
	// listener is never stopped on refusal. Passive status checks and cleanup
	// reconciliation keep adopting so reporting and stop stay truthful.
	e.nodeProjection.mu.Lock()
	st.mu.Lock()
	if ctx.Err() != nil || ((automaticCrashRelaunch(ctx) || activationAdoption(ctx)) && e.nodeProjection.projection != nil && e.nodeProjection.projection.ManagementBlocked) {
		st.mu.Unlock()
		e.nodeProjection.mu.Unlock()
		return presenceResult{Occupied: true}
	}
	if st.stopping && !allowWhileStopping {
		st.mu.Unlock()
		e.nodeProjection.mu.Unlock()
		return presenceResult{Identified: true, Occupied: true}
	}
	// Detect resets installed=false on every path-miss. Do not treat that
	// internal refresh as a state transition for an already-adopted service,
	// or every status poll would emit a duplicate state-changed event.
	changed := !st.running || !st.healthy || !st.adopted || st.port != port
	// A service that was not running, healthy and adopted on this port may be
	// another process now, so its adoption is a new run.
	if changed || st.runID == "" {
		st.runID = newEngineRunID()
	}
	st.installed = serviceInstalled
	st.running = true
	st.healthy = true
	st.adopted = true
	st.port = port
	if allowWhileStopping {
		st.stopping = false
	}
	if !executablePresent {
		// Detection missed, so this is explicitly an external service. Keep
		// binPath empty so stop/uninstall can never treat it as our orphan.
		st.binPath = ""
	}
	st.mu.Unlock()
	e.nodeProjection.mu.Unlock()
	if changed {
		e.emitState(engine)
	}
	return presenceResult{Identified: true, Occupied: true}
}

func (e *Executor) emitState(engine string) {
	st, err := e.state(engine)
	if err != nil {
		return
	}
	e.publishMu.Lock()
	e.notify("engine:state-changed", e.snapshot(engine, st))
	e.publishMu.Unlock()
	// A start, stop or adoption may have changed the engine's run identity;
	// the watcher's next sweep reports it (model_identity.go).
	e.pokeLoaded()
}
