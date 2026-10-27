// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "nvpair-shared/buildapply"

// Admission is local, before namespace forwarding. This is not the settings
// maintenance exclusion: Task 3 must wire it at the authoritative settings owner.
func (b *Broker) handleBuildApply(msg *Message) {
	var request any
	switch msg.Method {
	case "build:apply-impact":
		request = &buildapply.ImpactRequest{}
	case "build:apply-prepare":
		request = &buildapply.PrepareRequest{}
	case "build:apply-finish":
		request = &buildapply.FinishRequest{}
	}
	if err := buildapply.DecodeRequest(msg.Params, request); err != nil {
		_ = b.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	if msg.Method == "build:apply-impact" {
		_ = b.codec.Respond(msg.ID, buildapply.Unavailable[buildapply.Impact]())
	} else {
		_ = b.codec.Respond(msg.ID, buildapply.Unavailable[buildapply.BuildApplyOperation]())
	}
}

// The broker admits these locally and does not forward them to a worker.
func isBuildApplyMethod(method string) bool {
	switch method {
	case "build:apply-impact", "build:apply-prepare", "build:apply-finish":
		return true
	default:
		return false
	}
}
