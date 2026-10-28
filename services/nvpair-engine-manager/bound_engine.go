// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/nodepolicy"
)

func lockBoundEngineLifecycle(ctx context.Context, st *engineState) error {
	for !st.opMu.TryLock() {
		timer := time.NewTimer(10 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return context.Cause(ctx)
		case <-timer.C:
		}
	}
	return nil
}

// The empty mutation key reserves cold activation against other engine
// mutations. It uses the existing admission owner and is never durable state.
type boundEngineActivation struct{ operationID, engine string }

func (e *Executor) admitBoundEngineMutation(engine, operationID string) (func(), error) {
	e.switchAdmission.Lock()
	defer e.switchAdmission.Unlock()
	for _, n := range e.switchMutations {
		if n > 0 {
			return nil, &switchBusyError{engine}
		}
	}
	// This entry reserves the same evidence generation's retirement before
	// publishing the global key or cold activation. Keep it held until both
	// are retired below, including the interval before done reacquires the gate.
	done, err := e.admitMutationLocked(engine)
	if err != nil {
		return nil, err
	}
	e.switchMutations[""]++
	reservation := &boundEngineActivation{operationID, engine}
	e.boundActivation = reservation
	return func() {
		e.switchAdmission.Lock()
		e.switchMutations[""]--
		if e.boundActivation == reservation {
			e.boundActivation = nil
		}
		e.switchAdmission.Unlock()
		done()
	}, nil
}

func (m *Manager) boundEngineAuthority(p nodepolicy.ActivateBoundEngineRequest) error {
	if m.exec.switchIncarnation != p.ManagerIncarnation {
		return &switchBusyError{p.Engine}
	}
	adv := m.exec.nodeProjection.advertisement()
	if adv == nil || adv.Revision != p.PolicyRevision {
		return &localVetoError{code: -32010, reason: "Bound engine policy revision changed."}
	}
	if p.PeerID != "" {
		if m.mesh == nil {
			m.logBoundAuthorityRefusal(p, "engine_manager.peer_incarnation", "not_checked", clustertrust.PeerServiceDiagnostic{Predicate: "mesh_unavailable"})
			return &localVetoError{code: codeRemoteManagementDenied, reason: remoteManagementDeniedMessage}
		}
		m.mesh.Refresh()
		incarnation, ok, diagnostic := m.mesh.PeerServiceIncarnationWithDiagnostic(p.PeerID)
		if !ok || incarnation != p.PeerIncarnation {
			if ok {
				diagnostic.Predicate = "incarnation_mismatch"
			}
			diagnostic.ObservedIncarnation = incarnation
			m.logBoundAuthorityRefusal(p, "engine_manager.peer_incarnation", "not_checked", diagnostic)
			return &localVetoError{code: codeRemoteManagementDenied, reason: remoteManagementDeniedMessage}
		}
		permission := m.mesh.RemoteManagement(p.PeerID)
		if !permission.Allowed {
			m.logBoundAuthorityRefusal(p, "engine_manager.remote_management", permission.State, permission.Diagnostic)
			return &localVetoError{code: codeRemoteManagementDenied, reason: remoteManagementDeniedMessage}
		}
	}
	return m.exec.remoteVeto(p.Engine)
}

// This is also used by the accepted activation's authority watcher. Log the
// original predicate, without re-reading authority to explain the failure.
func (m *Manager) logBoundAuthorityRefusal(p nodepolicy.ActivateBoundEngineRequest, site, permission string, diagnostic clustertrust.PeerServiceDiagnostic) {
	if diagnostic.ExpectedIncarnation == "" {
		diagnostic.ExpectedIncarnation = p.PeerIncarnation
	}
	slog.Warn("bound engine management refused",
		"site", site, "reason", diagnostic.Predicate,
		"node_id", diagnostic.NodeID, "peer_id", p.PeerID, "cluster_id", diagnostic.ClusterID,
		"epoch", diagnostic.Epoch, "counter", diagnostic.Counter, "io_error_code", diagnostic.ErrorCode,
		"expected_incarnation", diagnostic.ExpectedIncarnation, "observed_incarnation", diagnostic.ObservedIncarnation,
		"changed_component_authority_compared", diagnostic.ChangedComponentCompared,
		"changed_component_authority_same", diagnostic.ChangedComponentSame,
		"manager_run_id", m.exec.switchIncarnation,
		"permission", permission)
}

func (m *Manager) activateBoundEngine(ctx context.Context, p nodepolicy.ActivateBoundEngineRequest) error {
	if err := p.Validate(); err != nil {
		return err
	}
	if !p.Deadline.After(time.Now()) || p.Deadline.After(time.Now().Add(time.Duration(nodepolicy.MaxBoundEngineBudgetMs)*time.Millisecond)) {
		return context.DeadlineExceeded
	}
	ctx, cancelDeadline := context.WithDeadline(ctx, p.Deadline)
	defer cancelDeadline()
	if err := m.boundEngineAuthority(p); err != nil {
		return err
	}
	// Local automatic demand also yields to a later local Off through the
	// existing remote-work cancellation owner. Intent retains its real origin.
	ctx, done, err := m.exec.beginRemote(ctx, p.Engine)
	if err != nil {
		return err
	}
	defer done()
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	finished := make(chan struct{})
	defer close(finished)
	go func() {
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-finished:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				if err := m.boundEngineAuthority(p); err != nil {
					cancel(err)
					return
				}
			}
		}
	}()
	guard := func() error {
		if err := ctx.Err(); err != nil {
			return context.Cause(ctx)
		}
		if err := m.boundEngineAuthority(p); err != nil {
			return err
		}
		// Admission excludes competing starts before these observations. No
		// inference-driven operation can choose or stop a source here.
		for _, name := range m.exec.reg.Names() {
			st, err := m.exec.state(name)
			if err != nil {
				if name == p.Engine {
					return err
				}
				continue // Unsupported host entries cannot run here.
			}
			st.mu.Lock()
			running := st.running || st.stopping || (name != p.Engine && (st.reclaimFailed || st.reclaimPID != 0))
			st.mu.Unlock()
			if running {
				return &switchBusyError{name}
			}
		}
		return nil
	}
	origin := intentLocal
	if p.PeerID != "" {
		origin = intentPeer
	}
	err = m.exec.StartWith(ctx, p.Engine, startOpts{Origin: origin, ReclaimOwned: true, RequireOwned: true, BoundedCleanup: true, boundGuard: guard, boundOperationID: p.OperationID})
	if cause := context.Cause(ctx); cause != nil {
		return errors.Join(cause, err)
	}
	return err
}

func (m *Manager) runActivateBoundEngine(ctx context.Context, msg *Message) {
	var p nodepolicy.ActivateBoundEngineRequest
	if err := json.Unmarshal(msg.Params, &p); err != nil {
		_ = m.codec.RespondError(msg.ID, -32602, "Invalid bound engine activation.")
		return
	}
	if err := m.activateBoundEngine(ctx, p); err != nil {
		code := -32000
		var veto *localVetoError
		if errors.As(err, &veto) {
			code = veto.code
		}
		_ = m.codec.RespondError(msg.ID, code, "Bound engine activation was refused.")
		return
	}
	_ = m.codec.Respond(msg.ID, map[string]bool{"success": true})
}
