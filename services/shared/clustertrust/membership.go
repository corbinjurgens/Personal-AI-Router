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
func peerServiceAdmission(data []byte) (admissionRecord, bool) {
	var rec struct {
		admissionRecord
		Counter   uint64 `json:"counter"`
		Activated uint64 `json:"activated"`
		Retired   bool   `json:"retired"`
	}
	if err := json.Unmarshal(data, &rec); err != nil {
		return admissionRecord{}, false
	}
	// The owner loads records predating activated with epoch as the high-water
	// mark; use the same interpretation without upgrading anything on disk.
	if rec.Activated == 0 {
		rec.Activated = rec.Epoch
	}
	if rec.ClusterID == "" || rec.Epoch == 0 || rec.Retired ||
		rec.Epoch > rec.Activated || rec.Activated > rec.Counter {
		return admissionRecord{}, false
	}
	return rec.admissionRecord, true
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
