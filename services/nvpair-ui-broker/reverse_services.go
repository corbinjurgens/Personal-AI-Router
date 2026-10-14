// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"log/slog"
	"time"

	"nvpair-shared/noderec"
)

const setLocalServicesMethod = "cluster:set-local-services"
const localServicesWriteTimeout = 5 * time.Second

// This is private parent control metadata. Listener protocol policy belongs to
// cluster-manager, not the discovery transport labels in these registrations.
type localServicesSnapshot struct {
	Generation uint64                   `json:"generation"`
	Services   []noderec.RegisterParams `json:"services"`
}

// pushLocalServicesLocked captures an authoritative snapshot and queues it on
// the current child. Caller holds localServicesMu, including on child install,
// so a respawn always gets the newest generation even if no service changed.
func (b *Broker) pushLocalServicesLocked() {
	if b.regCache == nil {
		return
	}
	b.workersMu.Lock()
	cm, ctx := b.clusterMgr, b.serveCtx
	b.workersMu.Unlock()
	if cm == nil {
		return
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if cm.localServicesUpdates == nil {
		cm.localServicesUpdates = make(chan localServicesSnapshot, 1)
		cm.localServicesStop = make(chan struct{})
		go b.sendLocalServices(ctx, cm, cm.localServicesUpdates, cm.localServicesStop)
	}
	snapshot := localServicesSnapshot{
		Generation: b.localServicesGeneration,
		Services:   b.regCache.Snapshot(),
	}
	// Only the newest pending full snapshot matters. The producer is serialized
	// by localServicesMu; the sender consumes in order and never reads the cache.
	select {
	case <-cm.localServicesUpdates:
	default:
	}
	cm.localServicesUpdates <- snapshot
}

func (b *Broker) sendLocalServices(ctx context.Context, c *clusterManagerProcess, updates <-chan localServicesSnapshot, stop <-chan struct{}) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-stop:
			return
		case <-c.peer.Done():
			return
		case snapshot := <-updates:
			if b.getClusterMgr() != c {
				return
			}
			if !c.writeLocalServices(ctx, stop, snapshot) {
				return
			}
		}
	}
}

// Peer.Notify can block writing a full subprocess pipe. Bound that write and
// release it by closing this child's stdin on cancellation or timeout. Never
// hold broker locks, nor wait for a dead child to finish writing on shutdown.
func (c *clusterManagerProcess) writeLocalServices(ctx context.Context, stop <-chan struct{}, snapshot localServicesSnapshot) bool {
	written := make(chan error, 1)
	go func() { written <- c.peer.Notify(setLocalServicesMethod, snapshot) }()
	timer := time.NewTimer(localServicesWriteTimeout)
	defer timer.Stop()
	select {
	case err := <-written:
		if err != nil {
			slog.Debug("failed to push local services to cluster-manager", "err", err)
		}
		return err == nil
	case <-ctx.Done():
	case <-stop:
	case <-c.peer.Done():
	case <-timer.C:
		slog.Warn("cluster-manager local service write timed out")
	}
	if c.stdin != nil {
		_ = c.stdin.Close()
	}
	c.peer.Close()
	return false
}
