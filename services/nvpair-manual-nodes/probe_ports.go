// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "fmt"

// probePorts is startup configuration, shared by initial, restored and periodic
// probes. Zero disables that leg; an entry's explicit TLS hint still takes
// precedence over the plain node-info target.
type probePorts struct {
	nodeInfo int
}

func defaultProbePorts() probePorts {
	return probePorts{nodeInfo: 14318}
}

func (p probePorts) validate() error {
	for _, target := range []struct {
		name string
		port int
	}{
		{"node-info-port", p.nodeInfo},
	} {
		if target.port < 0 || target.port > 65535 {
			return fmt.Errorf("--%s must be between 0 and 65535 (got %d)", target.name, target.port)
		}
	}
	return nil
}
