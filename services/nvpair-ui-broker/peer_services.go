// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"time"

	"nvpair-shared/noderec"
)

const peerServicesEvent = "cluster:peer-services"
const peerServicesIngest = "discovery:peer-services"

// Child replacement and private delivery share this boundary. An old child's
// delayed event cannot pass a new child's snapshot, and no remote metadata is
// ever installed in the broker's local registration/forwarding cache.
func (b *Broker) relayPeerServices(child *clusterManagerProcess, params json.RawMessage) {
	b.peerServicesMu.Lock()
	defer b.peerServicesMu.Unlock()
	if child == nil || b.getClusterMgr() != child || len(params) > 2*noderec.PeerServicesMaxBytes {
		return
	}
	var claim noderec.PeerServicesClaim
	if err := json.Unmarshal(params, &claim); err != nil {
		return
	}
	if err := noderec.ValidatePeerServicesDescriptor(claim.Descriptor, claim.Descriptor.HostUUID); err != nil {
		return
	}
	scanner := b.getScanner()
	if scanner == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	written := make(chan error, 1)
	go func() { written <- scanner.peer.Notify(peerServicesIngest, claim) }()
	select {
	case <-written:
	case <-ctx.Done():
		scanner.interruptWrite()
	case <-scanner.done:
	}
}
