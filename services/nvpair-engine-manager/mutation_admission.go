// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"sync"
)

type diagnosticBusyError struct{ engine string }

func (err *diagnosticBusyError) Error() string {
	return fmt.Sprintf("engine %q is reserved by a memory measurement (busy)", err.engine)
}

type mutationBusyError struct{ engine string }

func (err *mutationBusyError) Error() string {
	return fmt.Sprintf("engine %q is busy with another mutation", err.engine)
}

// admitMutation covers waiting for opMu as well as the operation itself.
func (e *Executor) admitMutation(engine string) (func(), error) {
	return e.admitMutationContext(context.Background(), engine)
}

// Only the explicitly scoped diagnostic context can use its reservation. All
// other admission checks, including readable desired state, still apply.
func (e *Executor) admitMutationContext(ctx context.Context, engine string) (func(), error) {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	if err := e.mutationAllowedContextLocked(ctx, engine); err != nil {
		return nil, err
	}
	done, err := e.trackMutationLocked(engine)
	if err == nil && e.diagnosticAuthorityLocked(ctx, engine) {
		e.diagnostic.workUnconfirmed = true
	}
	return done, err
}

func (e *Executor) admitMutationLocked(engine string) (func(), error) {
	if err := e.mutationAllowedLocked(engine); err != nil {
		return nil, err
	}
	return e.trackMutationLocked(engine)
}

func (e *Executor) mutationAllowedLocked(engine string) error {
	return e.mutationAllowedContextLocked(context.Background(), engine)
}

func (e *Executor) mutationAllowedContextLocked(ctx context.Context, engine string) error {
	if _, scoped := ctx.Value(diagnosticAuthorityKey{}).(diagnosticAuthority); scoped && !e.diagnosticAuthorityLocked(ctx, engine) {
		return diagnosticConflict("expired or incorrectly scoped reservation authority")
	}
	if e.diagnostic != nil && !e.diagnosticAuthorityLocked(ctx, engine) {
		return &diagnosticBusyError{engine}
	}
	if e.activeMutations[""] > 0 {
		// Authority removes only the diagnostic's own global ticket. It must
		// never bypass a different global admission owner.
		if e.activeMutations[""] != 1 || !e.diagnosticAuthorityLocked(ctx, engine) {
			return &mutationBusyError{engine}
		}
	}
	e.desired.mu.Lock()
	state, err := e.desired.load()
	e.desired.mu.Unlock()
	if err != nil {
		return err
	}
	if state.PendingDiagnostic != nil && e.diagnostic == nil {
		return &diagnosticBusyError{engine}
	}
	return nil
}

func (e *Executor) trackMutationLocked(engine string) (func(), error) {
	retire, err := e.reserveOwnerAdmissionLocked()
	if err != nil {
		return nil, err
	}
	return e.trackOwnerMutationLocked(engine, retire), nil
}

// The stop path reserves this same ticket before loading desired state, then
// transfers its retirement to the physical operation after desired.mu is
// released. A failed load still advances the entry/retirement ABA fence.
func (e *Executor) trackOwnerMutationLocked(engine string, retire func()) func() {
	if e.activeMutations == nil {
		e.activeMutations = make(map[string]int)
	}
	e.activeMutations[engine]++
	e.withdrawOwnerKnowledge(engine)
	return sync.OnceFunc(func() {
		e.mutationAdmission.Lock()
		e.activeMutations[engine]--
		retire()
		e.mutationAdmission.Unlock()
	})
}

func (e *Executor) admitStop(engine string) (func(), error) {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	retire, err := e.reserveOwnerAdmissionLocked()
	if err != nil {
		return nil, err
	}
	transferred := false
	defer func() {
		if !transferred {
			retire()
		}
	}()
	e.withdrawOwnerKnowledge(engine)
	track := func() func() { transferred = true; return e.trackOwnerMutationLocked(engine, retire) }
	e.desired.mu.Lock()
	_, err = e.desired.load()
	e.desired.mu.Unlock()
	if err != nil {
		return nil, err
	}
	if err := e.mutationAllowedLocked(engine); err != nil {
		return nil, err
	}
	return track(), nil
}
