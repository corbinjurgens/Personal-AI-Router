// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package buildapply defines the local build-apply wire contract.
package buildapply

// Layout identifies packaging paths relative to a complete extracted GUI build.
// These names describe packaging, not validation or authority to replace files.
type Layout struct {
	RootName          string `json:"rootName"`
	Executable        string `json:"executable"`
	ServiceDir        string `json:"serviceDir"`
	ServiceExecutable string `json:"serviceExecutable"`
}

func LayoutFor(platform string) (Layout, bool) {
	switch platform {
	case "windows":
		return Layout{Executable: "PAIR.exe", ServiceDir: "resources/cli-bin", ServiceExecutable: "nvpair-service.exe"}, true
	case "darwin":
		return Layout{RootName: "NVIDIA PAIR.app", Executable: "Contents/MacOS/NVIDIA PAIR", ServiceDir: "Contents/Resources/cli-bin", ServiceExecutable: "nvpair-service"}, true
	case "linux":
		return Layout{Executable: "nvpair", ServiceDir: "resources/cli-bin", ServiceExecutable: "nvpair-service"}, true
	default:
		return Layout{}, false
	}
}
