// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "nvpair-shared/nodepolicy"

// workloadKey is the registry's secondary index, never a second owner.
type workloadKey struct {
	engine string
	id     string
}

func (p *Proxy) handleWorkloadCancel(msg *Message) {
	var in nodepolicy.WorkloadCancelParams
	if err := decodeStrict(msg.Params, &in); err != nil {
		_ = p.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	owner := p.workloadCancelControl(in)
	found := owner != nil && owner.cancelQualified(in.Regenerate, &workloadKey{engine: in.Engine, id: in.WorkloadID})
	_ = p.codec.Respond(msg.ID, nodepolicy.WorkloadCancelResult{Found: found})
}
