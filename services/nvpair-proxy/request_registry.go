// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"nvpair-shared/nodepolicy"
)

// identityMu guards the registry and its secondary index. Each entry reserves
// linkage, key headers, lease and closure storage plus its owned key bytes.
const routingRegistryEntryBytes int64 = 512

// Backing is funded separately from rows, including deleted map slots. A fresh
// map has explicit spare capacity; insertions consume that capacity even when
// rows were deleted, bounding tombstone/overflow churn before a charged rebuild.
type routingMapStorage struct {
	lease   *bufferLease
	slots   int
	inserts int
	source  *routingSourceDirectory // Independent native lifetime; not funded by registry grants.
}

const routingMapBaseBytes int64 = 256
const routingMapSlotBytes int64 = 256

// reserveInsertion returns nil when already-funded backing can be reused.
// Otherwise the replacement grant overlaps the entire old backing charge;
// it never waits for capacity and does not mutate either map or storage owner.
func (s *routingMapStorage) reserveInsertion(root *bufferBudget, live int) (*bufferLease, int, error) {
	if s.lease != nil && s.inserts < s.slots {
		return nil, 0, nil
	}
	maxInt := int(^uint(0) >> 1)
	if live == maxInt {
		return nil, 0, errBufferInvalid
	}
	needed, slots := live+1, 8
	for slots/2 < needed {
		if slots > maxInt/2 || int64(slots) > (math.MaxInt64-routingMapBaseBytes)/(2*routingMapSlotBytes) {
			return nil, 0, errBufferInvalid
		}
		slots *= 2
	}
	root.mu.Lock()
	lease, err := root.acquireLocked(context.Background(), routingMapBaseBytes+int64(slots)*routingMapSlotBytes, nil, false)
	return lease, slots, err
}

// install follows the typed map swap. The caller closes old only after the
// old map is discarded and outside the request mutex. identityMu stays held.
func (s *routingMapStorage) install(lease *bufferLease, slots, live int) *bufferLease {
	old := s.lease
	*s = routingMapStorage{lease: lease, slots: slots, inserts: live, source: s.source}
	return old
}

func (s *routingMapStorage) discard() {
	old := s.lease
	*s = routingMapStorage{source: s.source}
	if old != nil {
		old.close()
	}
}

// The existing parent/owned codecs accept 1 MiB frames. This is a transport
// ceiling, not a second storage pool; snapshots also need a root grant.
const routingSnapshotFrameBytes int64 = 1 << 20

type routingRequest struct {
	key   nodepolicy.RoutingRequestKey
	owner *routingSpend
	lease *bufferLease
}

func (p *Proxy) registerRoutingRequest(owner *routingSpend) (*routingRequest, error) {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	if err := p.routingIdentityLocked(); err != nil {
		return nil, err
	}
	if owner == nil {
		return nil, fmt.Errorf("routing request owner is unavailable")
	}
	if p.requestSequence == math.MaxInt64 {
		return nil, fmt.Errorf("routing request identity exhausted")
	}
	next := p.requestSequence + 1
	digits := int64(1)
	for n := next; n >= 10; n /= 10 {
		digits++
	}
	charge := routingRegistryEntryBytes + int64(len(p.originNodeID)) + int64(len(p.runID)) + digits
	// Nonwaiting root admission cannot enqueue or call back into identityMu.
	// No owned key allocation, publication or counter advance precedes it.
	p.buffers.mu.Lock()
	lease, err := p.buffers.acquireLocked(context.Background(), charge, nil, false)
	if err != nil {
		return nil, err
	}
	backing, slots, err := p.requestStorage.reserveInsertion(p.buffers, len(p.requests))
	if err != nil {
		lease.close()
		return nil, err
	}
	if backing != nil {
		nextMap := make(map[string]*routingRequest, slots)
		for id, request := range p.requests {
			nextMap[id] = request
		}
		p.requests = nextMap
		if old := p.requestStorage.install(backing, slots, len(p.requests)); old != nil {
			old.close()
		}
	}
	key := nodepolicy.RoutingRequestKey{OriginNodeID: strings.Clone(p.originNodeID), ProxyRunID: strings.Clone(p.runID), RequestID: strconv.FormatInt(next, 10)}
	owner.key = key // Immutable before publication in the registry.
	request := &routingRequest{key: key, owner: owner, lease: lease}
	owner.detachWorkload = func() { p.unregisterWorkload(request) }
	p.requests[key.RequestID] = request
	p.requestStorage.inserts++
	p.requestSequence = next
	return request, nil
}

func (p *Proxy) unregisterRoutingRequest(request *routingRequest) {
	if request == nil {
		return
	}
	request.owner.finish()
	p.identityMu.Lock()
	if p.requests[request.key.RequestID] != request {
		p.identityMu.Unlock()
		return
	}
	p.unregisterWorkloadLocked(request)
	request.owner.freezeFacts()
	delete(p.requests, request.key.RequestID)
	if len(p.requests) == 0 {
		p.requests = nil
		p.requestStorage.discard()
	}
	p.identityMu.Unlock()
	request.lease.close()
}

func (p *Proxy) registerWorkload(request *routingRequest, key workloadKey) error {
	// Prepare inline metadata before either registry or request lock.
	var descriptor *routingWorkloadDescriptor
	if factIdentifier(key.engine) && factIdentifier(key.id) {
		descriptor = &routingWorkloadDescriptor{}
		descriptor.engineLen = uint16(copy(descriptor.engine[:], key.engine))
		descriptor.idLen = uint16(copy(descriptor.id[:], key.id))
	}
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	if request == nil || p.requests[request.key.RequestID] != request {
		return fmt.Errorf("workload requires a live unlinked routing request")
	}
	var old *bufferLease
	defer func() {
		if old != nil {
			old.close()
		}
	}()
	request.owner.mu.Lock()
	defer request.owner.mu.Unlock()
	if request.owner.done || request.owner.workload != nil {
		return fmt.Errorf("workload requires a live unlinked routing request")
	}
	if _, exists := p.workloadRequests[key]; exists {
		return fmt.Errorf("workload is already linked")
	}
	backing, slots, err := p.workloadStorage.reserveInsertion(p.buffers, len(p.workloadRequests))
	if err != nil {
		return err
	}
	if backing != nil {
		nextMap := make(map[workloadKey]string, slots)
		for key, id := range p.workloadRequests {
			nextMap[key] = id
		}
		p.workloadRequests = nextMap
		old = p.workloadStorage.install(backing, slots, len(p.workloadRequests))
	}
	p.workloadRequests[key] = request.key.RequestID
	p.workloadStorage.inserts++
	request.owner.workload = &key
	if f := request.owner.facts; descriptor != nil && f.begin() {
		if old := f.workload.Load(); old != nil {
			descriptor.generation = old.generation + 1
		} else {
			descriptor.generation = 1
		}
		f.workload.Store(descriptor)
		f.binding.Store(factAttached) // Pending generation, not yet representable.
		// A registration after attempt creation can bind that actual local
		// attempt. Ordinal and engine are checked, never an assumed index zero.
		if a := request.owner.active; a != nil {
			c := &f.attempts[(a.ordinal-1)%32]
			if d := c.destination.Load(); c.ordinal.Load() == a.ordinal && d != nil && d.kind == 1 && d.secondLen == descriptor.engineLen && d.second == descriptor.engine {
				f.binding.Store(a.ordinal | factAttached)
			}
		}
		f.end()
	}
	return nil
}

func (p *Proxy) unregisterWorkload(request *routingRequest) {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	if request != nil && p.requests[request.key.RequestID] == request {
		p.unregisterWorkloadLocked(request)
	}
}

func (p *Proxy) unregisterWorkloadLocked(request *routingRequest) {
	defer func() {
		if len(p.workloadRequests) == 0 {
			p.workloadRequests = nil
			p.workloadStorage.discard()
		}
	}()
	request.owner.mu.Lock()
	defer request.owner.mu.Unlock()
	if request.owner.workload != nil {
		delete(p.workloadRequests, *request.owner.workload)
		request.owner.workload = nil
		if f := request.owner.facts; f != nil {
			if w := f.workload.Load(); w != nil {
				request.owner.nativeWorkloadDetachedLocked(w.generation)
			}
		}
	}
}

func (p *Proxy) workloadControls(engine, id string) []*routingSpend {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	var matches []*routingSpend
	for workload, requestID := range p.workloadRequests {
		if workload.id == id && (engine == "" || workload.engine == engine) {
			if request := p.requests[requestID]; request != nil {
				matches = append(matches, request.owner)
			}
		}
	}
	return matches
}

func (p *Proxy) workloadCancelControl(target nodepolicy.WorkloadCancelParams) *routingSpend {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	if target.Validate() != nil || p.routingIdentityLocked() != nil || target.OriginatedFrom != p.originNodeID || target.RunID != p.runID {
		return nil
	}
	requestID := p.workloadRequests[workloadKey{engine: target.Engine, id: target.WorkloadID}]
	if request := p.requests[requestID]; request != nil {
		return request.owner
	}
	return nil
}

func (p *Proxy) routingRequestControl(key nodepolicy.RoutingRequestKey) *routingSpend {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	if key.Validate() != nil || key.OriginNodeID != p.originNodeID || key.ProxyRunID != p.runID {
		return nil
	}
	if request := p.requests[key.RequestID]; request != nil {
		return request.owner
	}
	return nil
}

// snapshotRoutingRequests admits the entire owned copy and encoding overlap
// before allocating. Original registry strings may disappear after unlock, so
// the snapshot owns independently charged clones through Respond's final write.
func (p *Proxy) snapshotRoutingRequests(id *json.RawMessage) ([]nodepolicy.RoutingRequestKey, *bufferLease, error) {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	resultBytes := int64(len(`{"requests":[]}`))
	keyBytes := int64(0)
	for _, request := range p.requests {
		key := request.key
		n := int64(len(key.OriginNodeID) + len(key.ProxyRunID) + len(key.RequestID))
		// An extra comma per entry is conservative, including the first.
		encoded := int64(len(`{"originNodeId":"","proxyRunId":"","requestId":""}`)) + n + 1
		if encoded > routingSnapshotFrameBytes-resultBytes {
			return nil, nil, errBufferBusy
		}
		resultBytes += encoded
		keyBytes += n
	}
	idBytes := int64(0)
	if id != nil {
		// Raw IDs may require HTML escaping in the outer JSON marshal.
		// Check before multiplying, including on 32-bit hosts.
		if int64(len(*id)) > routingSnapshotFrameBytes/6 {
			return nil, nil, errBufferBusy
		}
		idBytes = 6 * int64(len(*id))
	}
	frameBytes := int64(len(`{"jsonrpc":"2.0","id":,"result":}`)) + idBytes + resultBytes + 1
	if frameBytes >= routingSnapshotFrameBytes {
		return nil, nil, errBufferBusy
	}
	// 128 per key covers slice/key headers and allocation rounding. Four
	// times each encoding bounds result RawMessage, outer frame plus newline,
	// and their marshal scratch/rounding overlap while both coexist.
	// All terms are bounded by the frame check before arithmetic/allocation.
	charge := int64(512) + int64(len(p.requests))*128 + keyBytes + 4*resultBytes + 4*frameBytes
	p.buffers.mu.Lock()
	lease, err := p.buffers.acquireLocked(context.Background(), charge, nil, false)
	if err != nil {
		return nil, nil, err
	}
	keys := make([]nodepolicy.RoutingRequestKey, 0, len(p.requests))
	for _, request := range p.requests {
		key := request.key
		keys = append(keys, nodepolicy.RoutingRequestKey{OriginNodeID: strings.Clone(key.OriginNodeID), ProxyRunID: strings.Clone(key.ProxyRunID), RequestID: strings.Clone(key.RequestID)})
	}
	return keys, lease, nil
}

func sortRoutingRequests(keys []nodepolicy.RoutingRequestKey) {
	slices.SortFunc(keys, func(a, b nodepolicy.RoutingRequestKey) int {
		if len(a.RequestID) < len(b.RequestID) {
			return -1
		}
		if len(a.RequestID) > len(b.RequestID) {
			return 1
		}
		return strings.Compare(a.RequestID, b.RequestID)
	})
}
