// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package servicekey holds the compact TXT keys of the consolidated
// _nvpair-node record and the static transport policy derived from them.
// noderec re-exports every name here; consumers use the noderec names.
//
// It is a separate package with no imports so that nvpair-shared/engines can
// name an engine's key without importing noderec: noderec carries the node
// policy advertisement (nodepolicy), and nodepolicy validates engine names
// against engines, so engines importing noderec would be an import cycle.
package servicekey

// Key is the compact TXT key for one advertised service's port.
type Key string

const (
	NodeInfo Key = "ni"
	Ollama   Key = "ol"
	LMStudio Key = "lm"
	LlamaCPP Key = "lc"
	Errors   Key = "er"
	Workload Key = "wl"
	Cluster  Key = "cl"
	// EngineManager is nvpair-engine-manager's LAN HTTP endpoint (the model
	// list, and future remote engine operations). Its /v1/models is how a peer's
	// discovery daemon learns a node's models now that the list is off the
	// size-limited mDNS TXT record. Plain HTTP (model names aren't secret).
	EngineManager Key = "em"
	// EngineControl is nvpair-engine-manager's cluster-scoped remote-control
	// endpoint (remote install/pull/start/stop + engine status). Unlike em, it
	// is pin-based mTLS (cluster peers only) because it performs privileged
	// operations, and it binds only when the node is clustered.
	EngineControl Key = "ec"
)

// Transport is the connection policy for a service, derived (not advertised).
type Transport int

const (
	// TransportPlain: always plain HTTP, even on a clustered node (node-info and
	// the local inference engines).
	TransportPlain Transport = iota
	// TransportMTLSWhenClustered: cluster-scoped pin-based mTLS when the target
	// node is clustered, else plain (nvpair-errors, nvpair-workload-manager).
	TransportMTLSWhenClustered
	// TransportSplit: nvpair-cluster-manager's :14321 — a plain PIN-authenticated
	// pairing channel plus mTLS trusted endpoints on one port, demuxed by the
	// first byte. cluster-manager owns this; generic consumers don't dial it.
	TransportSplit
)

// Transport returns the static transport policy for the service.
func (s Key) Transport() Transport {
	switch s {
	case Errors, Workload, EngineControl:
		return TransportMTLSWhenClustered
	case Cluster:
		return TransportSplit
	default: // node-info, engines (em model list is plain)
		return TransportPlain
	}
}

// UsesMTLS reports whether a consumer should dial this service over cluster
// mTLS, given whether the target node is clustered. node-info is deliberately
// plain even on a clustered node; errors/workload gate on the target's cluster
// state; the cluster-manager split channel's trusted plane is mTLS.
func (s Key) UsesMTLS(targetClustered bool) bool {
	switch s.Transport() {
	case TransportPlain:
		return false
	case TransportMTLSWhenClustered, TransportSplit:
		return targetClustered
	default:
		return false
	}
}
