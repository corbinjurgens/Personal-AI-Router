// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"nvpair-shared/nodepolicy"
)

// Task5's node-scope lease, retained from 65954dfc's engine-switch drain
// owner after removal of switch operations. Uses the existing drain registry,
// settings serialization, push slot and proxy admission snapshot. There is no
// diagnostic-specific admission or reservation mechanism here.
type nodeDrainLease struct {
	broker       *Broker
	added        []string
	settingsHeld bool
}

func (b *Broker) drainNode(parent context.Context) (lease *nodeDrainLease, err error) {
	ctx, cancel := context.WithTimeout(parent, time.Minute)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !b.settingsApplyMu.TryLock() {
		return nil, fmt.Errorf("an engine configuration operation is active")
	}
	settingsHeld := true
	defer func() {
		if settingsHeld {
			b.settingsApplyMu.Unlock()
		}
	}()
	r := b.nodePolicyRuntime()
	if !r.acquirePush(ctx) {
		return nil, ctx.Err()
	}
	defer r.releasePush()
	pp := b.policyProxy()
	r.mu.Lock()
	runID := r.runID
	r.mu.Unlock()
	if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
		return nil, err
	}
	profiles := b.frontedEngineProfiles()
	if len(profiles) == 0 {
		return nil, fmt.Errorf("node engine catalog unavailable")
	}
	owned := &nodeDrainLease{broker: b, settingsHeld: true}
	settingsHeld = false
	r.mu.Lock()
	for _, profile := range profiles {
		if !r.drains[profile.Name] {
			r.drains[profile.Name] = true
			owned.added = append(owned.added, profile.Name)
		}
	}
	r.mu.Unlock()
	defer func() {
		if err == nil {
			return
		}
		cleanup, cancel := context.WithTimeout(context.Background(), proxyCallTimeout)
		defer cancel()
		if cleanupErr := owned.releaseUnderPush(cleanup, pp); cleanupErr != nil {
			lease = owned
			err = errors.Join(err, cleanupErr)
		}
	}()
	for _, profile := range profiles {
		if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
			return nil, err
		}
		if err := engineDrainAck(ctx, pp, profile.Name, true); err != nil {
			return nil, err
		}
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
			return nil, err
		}
		raw, rpcErr, err := pp.Call(ctx, nodepolicy.MethodAdmissionSnapshot, settingsJSON(nodepolicy.AdmissionSnapshotParams{}))
		if err != nil {
			return nil, err
		}
		if rpcErr != nil {
			return nil, fmt.Errorf("admission snapshot refused")
		}
		if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
			return nil, err
		}
		idle, err := nodeDrainSnapshotIdle(raw, runID)
		if err != nil {
			return nil, err
		}
		if idle {
			return owned, nil
		}
		timer := time.NewTimer(100 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}

func (l *nodeDrainLease) release(ctx context.Context) error {
	if err := l.clear(ctx); err != nil {
		return err
	}
	l.releaseSettings()
	return nil
}

// Clear proxy/registry drains while retaining settings serialization. The
// diagnostic ledger uses this until its following manager release is settled,
// so a failed release can reassert drains without a settings writer racing it.
func (l *nodeDrainLease) clear(ctx context.Context) error {
	r := l.broker.nodePolicyRuntime()
	if !r.acquirePush(ctx) {
		return ctx.Err()
	}
	defer r.releasePush()
	return l.clearUnderPush(ctx, nil)
}

func (l *nodeDrainLease) releaseUnderPush(ctx context.Context, acquiringProxy *proxyProcess) error {
	if err := l.clearUnderPush(ctx, acquiringProxy); err != nil {
		return err
	}
	l.releaseSettings()
	return nil
}

func (l *nodeDrainLease) clearUnderPush(ctx context.Context, acquiringProxy *proxyProcess) error {
	b := l.broker
	r := b.nodePolicyRuntime()
	for len(l.added) != 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		pp := b.policyProxy()
		r.mu.Lock()
		incarnation, runID := r.incarnation, r.runID
		r.mu.Unlock()
		if pp != nil && pp != incarnation && (acquiringProxy == nil || pp == acquiringProxy) {
			return fmt.Errorf("node drain cleanup proxy run is unverified")
		}
		if pp != nil && pp == incarnation {
			if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
				return err
			}
			if err := engineDrainAck(ctx, pp, l.added[0], false); err != nil {
				return err
			}
			if err := b.nodeDrainProxyCurrent(pp, runID); err != nil {
				return err
			}
		} else if b.policyProxy() != pp {
			return fmt.Errorf("proxy changed during node drain cleanup")
		}
		r.mu.Lock()
		delete(r.drains, l.added[0])
		r.mu.Unlock()
		l.added = l.added[1:]
	}
	return nil
}

func (l *nodeDrainLease) releaseSettings() {
	if l.settingsHeld {
		l.settingsHeld = false
		l.broker.settingsApplyMu.Unlock()
	}
}

func (b *Broker) nodeDrainProxyCurrent(pp *proxyProcess, runID string) error {
	if pp == nil || b.policyProxy() != pp {
		return fmt.Errorf("proxy changed during node drain")
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.incarnation != pp || runID == "" || r.runID != runID {
		return fmt.Errorf("node drain proxy run is unverified")
	}
	return nil
}

func nodeDrainSnapshotIdle(raw json.RawMessage, runID string) (bool, error) {
	var snapshot struct {
		RunID          *string         `json:"runId"`
		Active         *int            `json:"active"`
		Queued         *int            `json:"queued"`
		ActiveByEngine map[string]*int `json:"activeByEngine"`
	}
	if err := json.Unmarshal(raw, &snapshot); err != nil {
		return false, err
	}
	if snapshot.RunID == nil || *snapshot.RunID != runID || snapshot.Active == nil || snapshot.Queued == nil || snapshot.ActiveByEngine == nil {
		return false, fmt.Errorf("missing or wrong-incarnation node admission evidence")
	}
	if *snapshot.Active < 0 || *snapshot.Queued < 0 {
		return false, fmt.Errorf("invalid node admission counts")
	}
	idle := *snapshot.Active == 0 && *snapshot.Queued == 0
	for _, count := range snapshot.ActiveByEngine {
		if count == nil || *count < 0 {
			return false, fmt.Errorf("invalid engine admission count")
		}
		idle = idle && *count == 0
	}
	return idle, nil
}
