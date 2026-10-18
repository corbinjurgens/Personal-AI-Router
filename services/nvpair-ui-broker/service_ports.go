// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"strconv"
)

// Default ports of the broker-owned local services. Each equals the matching
// worker's own --port default, so a default plan spawns the worker without the
// flag. The engine-manager defaults live beside engineManagerPorts.
const (
	nodeInfoHTTPPort       = 14318
	errorsHTTPPort         = 14319
	workloadHTTPPort       = 14320
	clusterManagerHTTPPort = 14321
)

// servicePortPlan is the broker's immutable startup configuration for the local
// service listeners it spawns, registers with discovery and reserves against
// engine ports and the OLLAMA_HOST alias. One value per service: the port
// passed to the worker is the port registered, replayed and reserved.
type servicePortPlan struct {
	nodeInfo          int
	errors            int
	workload          int
	clusterManager    int
	engineManagerHTTP int
	engineControl     int
}

// servicePort names one planned listener for diagnostics: the flag that
// configures it and the service a colliding user is told about.
type servicePort struct {
	flag  string
	owner string
	port  int
}

func (p servicePortPlan) entries() []servicePort {
	return []servicePort{
		{flag: "--node-info-port", owner: "the node-info service", port: p.nodeInfo},
		{flag: "--errors-port", owner: "the errors service", port: p.errors},
		{flag: "--workload-manager-port", owner: "the workload service", port: p.workload},
		{flag: "--cluster-manager-port", owner: "the cluster manager", port: p.clusterManager},
		{flag: "--engine-manager-http-port", owner: "the engine model service", port: p.engineManagerHTTP},
		{flag: "--engine-manager-control-port", owner: "the engine control service", port: p.engineControl},
	}
}

// owner names the planned service listening on port, or "".
func (p servicePortPlan) owner(port int) string {
	for _, entry := range p.entries() {
		if entry.port == port {
			return entry.owner
		}
	}
	return ""
}

// servicePorts resolves the startup plan. Bare Broker fixtures and zero-valued
// workerPaths retain the product defaults.
func (b *Broker) servicePorts() servicePortPlan {
	httpPort, controlPort := b.engineManagerPorts()
	return servicePortPlan{
		nodeInfo:          portOrDefault(b.nodeInfoPort, nodeInfoHTTPPort),
		errors:            portOrDefault(b.errorsPort, errorsHTTPPort),
		workload:          portOrDefault(b.workloadMgrPort, workloadHTTPPort),
		clusterManager:    portOrDefault(b.clusterMgrPort, clusterManagerHTTPPort),
		engineManagerHTTP: httpPort,
		engineControl:     controlPort,
	}
}

func portOrDefault(port, defaultPort int) int {
	if port == 0 {
		return defaultPort
	}
	return port
}

// validateServicePorts checks the whole local service set before any child is
// spawned: every port in range, no two services sharing a port, and none on a
// port an engine's proxy listener claims. Those listener reservations hold for
// every engine, including one deselected via --proxy-engines, matching
// siblingEngineProxyPorts: the engine's configuration outlives this process.
func validateServicePorts(p servicePortPlan) error {
	entries := p.entries()
	for _, entry := range entries {
		if entry.port < 1 || entry.port > 65535 {
			return fmt.Errorf("%s must be between 1 and 65535 (got %d)", entry.flag, entry.port)
		}
	}
	for i, a := range entries {
		for _, b := range entries[i+1:] {
			if a.port == b.port {
				return fmt.Errorf("%s and %s must differ (both are %d)", a.flag, b.flag, a.port)
			}
		}
	}
	for _, entry := range entries {
		for _, profile := range engineProxyProfiles {
			switch entry.port {
			case profile.FacadePort:
				return fmt.Errorf("%s %d is reserved: the %s compatibility proxy uses port %d",
					entry.flag, entry.port, profile.DisplayName, entry.port)
			case profile.EnginePortBase:
				return fmt.Errorf("%s %d is reserved: the managed %s backend uses port %d",
					entry.flag, entry.port, profile.DisplayName, entry.port)
			}
		}
	}
	return nil
}

// servicePortArgs is a worker's --port override. A default port is left to the
// worker's identical flag default, so a default plan keeps the historical argv.
func servicePortArgs(port, defaultPort int) []string {
	if port == defaultPort {
		return nil
	}
	return []string{"--port", strconv.Itoa(port)}
}
