// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"nvpair-shared/buildapply"
	"nvpair-shared/jsonrpc"
)

func (s *Service) handleBuildApply(c *client, msg *jsonrpc.Message) {
	var request any
	switch msg.Method {
	case "service/build-apply-preview":
		request = &buildapply.PreviewRequest{}
	case "service/build-apply-start":
		request = &buildapply.StartRequest{}
	case "service/build-apply-status":
		request = &buildapply.StatusRequest{}
	case "service/build-apply-gui-ready":
		request = &buildapply.GUIReadyRequest{}
	}
	if err := buildapply.DecodeRequest(msg.Params, request); err != nil {
		c.respondError(msg.ID, -32602, err.Error())
		return
	}
	if msg.Method == "service/build-apply-preview" {
		previewRequest, ok := request.(*buildapply.PreviewRequest)
		if !ok {
			c.respondError(msg.ID, -32602, "invalid preview request")
			return
		}
		s.respondBuildApplyPreview(c, msg.ID, *previewRequest)
	} else {
		c.respond(msg.ID, buildapply.Unavailable[buildapply.BuildApplyOperation]())
	}
}

// publishBuildApplyChanged reuses the service's existing local-client transport.
func (s *Service) publishBuildApplyChanged(operation buildapply.BuildApplyOperation) {
	s.broadcast("service/build-apply-changed", operation)
}

// Transaction methods remain unavailable until their owner is implemented.
// Neither preview nor unavailable start launches a process.
func isBuildApplyMethod(method string) bool {
	switch method {
	case "service/build-apply-preview", "service/build-apply-start", "service/build-apply-status", "service/build-apply-gui-ready":
		return true
	default:
		return false
	}
}
