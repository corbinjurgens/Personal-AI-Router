// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"log/slog"
	"time"

	"nvpair-ui-broker/workloadstore"
)

const departedRunWorkloadError = "proxy exited before the request finished"

// departedWorkload names exactly the record selected at departure, including
// its generation. A replacement may publish before the captured work is emitted.
type departedWorkload struct {
	key       workloadstore.Key
	createdAt int64
}

func departedWorkloadIdentity(rec workloadstore.Record) departedWorkload {
	in, _ := workloadstore.ParseIncoming(rec.Info)
	return departedWorkload{
		key:       workloadstore.Key{Origin: rec.Origin, Engine: in.Engine, RunID: rec.RunID, ID: rec.ID},
		createdAt: rec.CreatedAt,
	}
}

// captureDepartedRunWorkloads runs after handle withdrawal, before the old
// supervisor can be replaced. Source validation and apply share workersMu, so
// an admitted old callback either applied before this snapshot or is rejected.
// It never waits for workloadEmitMu or output: Stop may be joining this monitor.
// The spawnProxy bring-up failure stops are intentionally not hooked: policy
// replay fails before any facade is enabled, and a failed facade bring-up stops
// the process only when no facade was enabled and routingListenerEnabled is false.
func (b *Broker) captureDepartedRunWorkloads(cause string) {
	if b.workloads == nil {
		return
	}
	records := b.workloads.ActiveLocalOutsideRun(b.nodeID, "")
	if len(records) == 0 {
		return
	}
	b.proxySupWorkMu.Lock()
	defer b.proxySupWorkMu.Unlock()
	if b.proxyRetirePending == nil {
		b.proxyRetirePending = make(map[departedWorkload]string)
	}
	for _, rec := range records {
		b.proxyRetirePending[departedWorkloadIdentity(rec)] = cause
	}
	b.resumeProxyRetirementsLocked()
}

// resumeProxyRetirementsLocked is the output stage of the existing coalesced
// supervision owner. There is at most one emitter and one pending identity set.
// While it is blocked, workloadEmitMu prevents new workload acceptance; repeated
// crashes only coalesce the same open-or-inferred identities, not more frames or
// goroutines. Supervision can continue independently of the stalled output.
// Caller holds proxySupWorkMu.
func (b *Broker) resumeProxyRetirementsLocked() {
	if len(b.proxyRetirePending) == 0 || b.proxyRetireDone != nil {
		return
	}
	done := make(chan struct{})
	b.proxyRetireDone = done
	go func() {
		for {
			// A capture can come from Stop while it owns proxySupMu. Cross that
			// boundary before taking workloadEmitMu -> schedulerFeedMu or doing
			// client/worker I/O. No workload owner takes proxySupMu in reverse.
			b.proxySupMu.Lock()
			b.proxySupMu.Unlock()
			b.proxySupWorkMu.Lock()
			pending := b.proxyRetirePending
			b.proxyRetirePending = nil
			if len(pending) == 0 {
				b.proxyRetireDone = nil
				close(done)
				b.proxySupWorkMu.Unlock()
				return
			}
			b.proxySupWorkMu.Unlock()
			b.workloadEmitMu.Lock()
			frames := b.applyDepartedRunWorkloadsLocked(pending)
			for _, frame := range frames {
				b.publishProxyWorkloadLocked(frame.method, frame.params)
			}
			b.workloadEmitMu.Unlock()
		}
	}()
}

func (b *Broker) waitProxyRetirements() {
	b.proxySupWorkMu.Lock()
	done := b.proxyRetireDone
	b.proxySupWorkMu.Unlock()
	if done == nil {
		return
	}
	// Leave shared shutdown time for later cleanup. An unarmed clock still
	// grants a bounded opportunity for already-accepted output to drain.
	allowance := workerSignalGrace
	if remaining, active := teardownRemaining(); active && remaining < allowance {
		allowance = remaining
	}
	if !waitClosed(done, allowance) {
		slog.Warn("proxy workload retirement exhausted its shutdown allowance")
	}
}

func (b *Broker) departedRunCandidates(liveRunID, cause string) map[departedWorkload]string {
	pending := make(map[departedWorkload]string)
	for _, rec := range b.workloads.ActiveLocalOutsideRun(b.nodeID, liveRunID) {
		pending[departedWorkloadIdentity(rec)] = cause
	}
	return pending
}

// applyDepartedRunWorkloadsLocked reselects open-or-inferred records at the event
// owner. Captures cannot overwrite authoritative completed/cancelled outcomes
// or a later generation, nor restore an entry removed in the meantime. Use the
// current Info/Seq so a placement update accepted since capture is preserved.
// Caller holds workloadEmitMu; there is no output here, including on the
// replacement fence.
func (b *Broker) applyDepartedRunWorkloadsLocked(pending map[departedWorkload]string) []proxyWorkloadEvent {
	var frames []proxyWorkloadEvent
	if len(pending) == 0 {
		return frames
	}
	now := time.Now()
	counts := make(map[string]int)
	for _, rec := range b.workloads.ActiveLocalOutsideRun(b.nodeID, "") {
		cause, selected := pending[departedWorkloadIdentity(rec)]
		if !selected {
			continue
		}
		params := markWorkloadTerminal(rec.Info, "failed", now, departedRunWorkloadError, rec.Seq+1)
		projected, _, _ := projectWorkloadFrame("workload:errored", params)
		if projected != nil && b.applyProxyWorkloadLocked("workload:errored", projected) {
			frames = append(frames, proxyWorkloadEvent{method: "workload:errored", params: projected})
			counts[cause]++
		}
	}
	for cause, count := range counts {
		slog.Info("retired workloads of departed proxy run", "cause", cause, "count", count)
	}
	return frames
}
