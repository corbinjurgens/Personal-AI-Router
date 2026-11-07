// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "fmt"

type mutationBusyError struct{ engine string }

func (err *mutationBusyError) Error() string {
	return fmt.Sprintf("engine %q is busy with another mutation", err.engine)
}

// admitMutation covers waiting for opMu as well as the operation itself.
func (e *Executor) admitMutation(engine string) (func(), error) {
	e.mutationAdmission.Lock()
	defer e.mutationAdmission.Unlock()
	return e.admitMutationLocked(engine)
}

func (e *Executor) admitMutationLocked(engine string) (func(), error) {
	if err := e.mutationAllowedLocked(engine); err != nil {
		return nil, err
	}
	return e.trackMutationLocked(engine)
}

func (e *Executor) mutationAllowedLocked(engine string) error {
	if e.activeMutations[""] > 0 {
		return &mutationBusyError{engine}
	}
	e.desired.mu.Lock()
	_, err := e.desired.load()
	e.desired.mu.Unlock()
	if err != nil {
		return err
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
	return func() {
		e.mutationAdmission.Lock()
		e.activeMutations[engine]--
		retire()
		e.mutationAdmission.Unlock()
	}
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
