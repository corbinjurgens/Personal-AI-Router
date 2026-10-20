// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"nvpair-shared/nodepolicy"
)

// Request metadata has a separate bound so inference byte pressure cannot hide
// cancellation targets. identityMu guards this registry and its secondary index.
const maxLiveRoutingRequests = 1024

type routingRequest struct {
	key      nodepolicy.RoutingRequestKey
	control  *workloadCtl
	workload *workloadKey
}

func (p *Proxy) registerRoutingRequest(control *workloadCtl) (*routingRequest, error) {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	if err := p.routingIdentityLocked(); err != nil {
		return nil, err
	}
	if control == nil || len(p.requests) >= maxLiveRoutingRequests {
		return nil, fmt.Errorf("routing request registry is full or unavailable")
	}
	if p.requestSequence == math.MaxInt64 {
		return nil, fmt.Errorf("routing request identity exhausted")
	}
	p.requestSequence++
	key := nodepolicy.RoutingRequestKey{OriginNodeID: p.originNodeID, ProxyRunID: p.runID, RequestID: strconv.FormatInt(p.requestSequence, 10)}
	request := &routingRequest{key: key, control: control}
	if p.requests == nil {
		p.requests = map[string]*routingRequest{}
	}
	p.requests[key.RequestID] = request
	return request, nil
}

func (p *Proxy) unregisterRoutingRequest(request *routingRequest) {
	if request == nil {
		return
	}
	request.control.finish()
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	if p.requests[request.key.RequestID] != request {
		return
	}
	p.unregisterWorkloadLocked(request)
	delete(p.requests, request.key.RequestID)
}

func (p *Proxy) registerWorkload(request *routingRequest, key workloadKey) error {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	if request == nil || p.requests[request.key.RequestID] != request || request.workload != nil {
		return fmt.Errorf("workload requires a live unlinked routing request")
	}
	if _, exists := p.workloadRequests[key]; exists {
		return fmt.Errorf("workload is already linked")
	}
	if p.workloadRequests == nil {
		p.workloadRequests = map[workloadKey]string{}
	}
	p.workloadRequests[key] = request.key.RequestID
	request.workload = &key
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
	if request.workload != nil {
		delete(p.workloadRequests, *request.workload)
		request.workload = nil
	}
}

func (p *Proxy) workloadControls(engine, id string) []*workloadCtl {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	var matches []*workloadCtl
	for workload, requestID := range p.workloadRequests {
		if workload.id == id && (engine == "" || workload.engine == engine) {
			if request := p.requests[requestID]; request != nil {
				matches = append(matches, request.control)
			}
		}
	}
	return matches
}

func (p *Proxy) routingRequestControl(key nodepolicy.RoutingRequestKey) *workloadCtl {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	if key.Validate() != nil || key.OriginNodeID != p.originNodeID || key.ProxyRunID != p.runID {
		return nil
	}
	if request := p.requests[key.RequestID]; request != nil {
		return request.control
	}
	return nil
}

func (p *Proxy) activeRoutingRequests() []nodepolicy.RoutingRequestKey {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	keys := make([]nodepolicy.RoutingRequestKey, 0, len(p.requests))
	for _, request := range p.requests {
		keys = append(keys, request.key)
	}
	slices.SortFunc(keys, func(a, b nodepolicy.RoutingRequestKey) int { return strings.Compare(a.RequestID, b.RequestID) })
	return keys
}
