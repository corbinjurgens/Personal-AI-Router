// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"path"
	"strings"
)

type memoryPlanAdapter struct {
	Adapter string `json:"adapter"`
}

type memoryPlanCommand struct {
	Command          []string `json:"command"`
	WorkingDirectory string   `json:"working_directory"`
}

func (c *memoryPlanCommand) validate(platform string) error {
	absolute := func(s string) bool {
		if strings.ContainsAny(s, "\x00\r\n{}") || len(s) > 4096 {
			return false
		}
		if strings.HasPrefix(platform, "windows/") {
			return len(s) >= 3 && ((s[0] >= 'a' && s[0] <= 'z') || (s[0] >= 'A' && s[0] <= 'Z')) && s[1] == ':' && (s[2] == '\\' || s[2] == '/')
		}
		return path.IsAbs(s)
	}
	if len(c.Command) != 8 || !absolute(c.Command[0]) || !absolute(c.WorkingDirectory) {
		return errors.New("invalid memory plan command or working directory")
	}
	want := []string{"-B", "-s", "-m", "serve.server", "--estimate-memory", "--memory-plan-request", "{memory_plan_request}"}
	for i, arg := range want {
		if c.Command[i+1] != arg {
			return errors.New("invalid memory plan arguments")
		}
	}
	return nil
}
