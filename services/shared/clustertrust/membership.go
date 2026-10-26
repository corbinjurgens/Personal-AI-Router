// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// admissionFileName is nvpair-cluster-manager's durable admission record inside
// the cluster dir. Kept in sync with nvpair-cluster-manager/admission_store.go.
const admissionFileName = "admission.json"

// membersFileName is the cluster-manager's atomically persisted member roster.
const membersFileName = "members.json"

// peerServiceMember requires a unique confirmed durable peer admission. Pins
// alone cannot establish a new admission when the roster still records the old
// epoch. Only the cluster-manager writes or upgrades these records.
func peerServiceMember(data []byte, peerUUID, clusterID string, pinEpoch uint64) (uint64, bool) {
	var members []struct {
		NodeUUID       string `json:"nodeUuid"`
		State          string `json:"state"`
		ClusterID      string `json:"clusterId"`
		AdmissionEpoch uint64 `json:"admissionEpoch"`
	}
	if err := json.Unmarshal(data, &members); err != nil {
		return 0, false
	}
	var epoch uint64
	found := false
	for _, member := range members {
		if member.NodeUUID != peerUUID {
			continue
		}
		if found || member.State != "member" || member.ClusterID != clusterID ||
			member.AdmissionEpoch == 0 || member.AdmissionEpoch != pinEpoch {
			return 0, false
		}
		found, epoch = true, member.AdmissionEpoch
	}
	return epoch, found
}

// admissionRecord mirrors the subset of admission.json that answers "does this
// node currently belong to a cluster". nvpair-cluster-manager owns the full
// schema (a monotonic counter, retirement marker, etc.); the read side only
// needs the active pair. A live admission carries both a clusterId and a nonzero
// epoch; teardown (leave/removal) clears both while deliberately keeping the
// counter, so absence of an active pair is the authoritative "no longer a
// member" signal — even though the node.key/node.crt keypair persists.
type admissionRecord struct {
	ClusterID string `json:"clusterId"`
	Epoch     uint64 `json:"epoch"`
}

// peerServiceAdmission validates the owner's high-water marks as well as the
// active pair. Unlike the membership compatibility gate below, service-route
// authority requires an explicit, complete admission record.
func peerServiceAdmission(data []byte) (peerServiceAdmissionProjection, bool) {
	var rec peerServiceAdmissionProjection
	if err := json.Unmarshal(data, &rec); err != nil {
		return peerServiceAdmissionProjection{}, false
	}
	// The owner loads records predating activated with epoch as the high-water
	// mark; use the same interpretation without upgrading anything on disk.
	rec.effectiveActivated = rec.Activated
	if rec.effectiveActivated == 0 {
		rec.effectiveActivated = rec.Epoch
	}
	if rec.ClusterID == "" || rec.Epoch == 0 || rec.Retired ||
		rec.Epoch > rec.effectiveActivated || rec.effectiveActivated > rec.Counter {
		return peerServiceAdmissionProjection{}, false
	}
	return rec, true
}

// Keep raw and effective activated values so compatibility interpretation does
// not hide a high-water representation change between bracket observations.
type peerServiceAdmissionProjection struct {
	admissionRecord
	Counter            uint64 `json:"counter"`
	Activated          uint64 `json:"activated"`
	Retired            bool   `json:"retired"`
	effectiveActivated uint64
}

// hasActiveAdmission reports whether clusterDir/admission.json records a live
// admission. A missing, unreadable, malformed, or cleared file reads as false
// (not a member). This is the durable, restart-surviving membership fact, in
// contrast to keypair presence which outlives a membership by design.
//
// It is re-read on every Mesh.Refresh rather than cached: membership is exactly
// the fact that changes underneath a running service when the user creates,
// joins, or leaves a cluster.
func hasActiveAdmission(clusterDir string) bool {
	if clusterDir == "" {
		return false
	}
	data, err := os.ReadFile(filepath.Join(clusterDir, admissionFileName))
	if err != nil {
		return false
	}
	var rec admissionRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return false
	}
	return rec.ClusterID != "" && rec.Epoch != 0
}
