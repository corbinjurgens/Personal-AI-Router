// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/rand"
	"encoding/hex"

	"nvpair-shared/nodepolicy"
)

func newRunID() string {
	var nonce [16]byte
	// Go 1.25's crypto/rand.Read fills the buffer or terminates the process.
	rand.Read(nonce[:])
	return hex.EncodeToString(nonce[:])
}

func (p *Proxy) routingIdentityLocked() error {
	return nodepolicy.ValidateRoutingIdentity(p.originNodeID, p.runID)
}

func (p *Proxy) routingIdentityReady() bool {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	return p.routingIdentityLocked() == nil
}
