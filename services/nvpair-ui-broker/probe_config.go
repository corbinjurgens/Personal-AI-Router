// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"strconv"
)

// manualProbePorts configures the supervised manual-node worker, independently
// of this node's own listener plan. Nil in workerPaths keeps worker defaults;
// an explicit zero disables a probe rather than selecting a default target.
type manualProbePorts struct {
	ollama, lmstudio, nodeInfo int
}

func (p *manualProbePorts) validate() error {
	if p == nil {
		return nil
	}
	for _, target := range []struct {
		name string
		port int
	}{
		{"manual-ollama-port", p.ollama},
		{"manual-lmstudio-port", p.lmstudio},
		{"manual-node-info-port", p.nodeInfo},
	} {
		if target.port < 0 || target.port > 65535 {
			return fmt.Errorf("--%s must be between 0 and 65535 (got %d)", target.name, target.port)
		}
	}
	return nil
}

func (b *Broker) manualNodesArgs() []string {
	args := append(b.logLevelArgs(), b.clusterDirArgs()...)
	if p := b.manualProbePorts; p != nil {
		args = append(args, "--ollama-port", strconv.Itoa(p.ollama), "--lmstudio-port", strconv.Itoa(p.lmstudio), "--node-info-port", strconv.Itoa(p.nodeInfo))
	}
	return args
}

func (b *Broker) scannerArgs() []string {
	args := b.clusterDirArgs()
	if b.disableMDNS {
		args = append(args, "--mdns=false")
	}
	return args
}
