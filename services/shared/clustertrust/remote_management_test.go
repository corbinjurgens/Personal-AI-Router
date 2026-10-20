// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func managementEncode(t *testing.T, value interface{}) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode permission fixture: %v", err)
	}
	return data
}

func managementMarker(t *testing.T) RemoteManagementMigration {
	t.Helper()
	snapshot := RemoteManagementMigrationSnapshot{
		Local: RemoteManagementMigrationAuthority{PeerUUID: "self", CertFingerprint: "sha256:" + strings.Repeat("a", 64)},
		Peers: []RemoteManagementMigrationAuthority{},
	}
	id, err := RemoteManagementMigrationID(snapshot)
	if err != nil {
		t.Fatalf("snapshot digest: %v", err)
	}
	return RemoteManagementMigration{Version: 1, ID: id, State: "complete", Snapshot: snapshot}
}

func managementFixture(t *testing.T) (incarnationFixture, RemoteManagementDocument, RemoteManagementMigration) {
	t.Helper()
	f := newIncarnationFixture(t)
	marker := managementMarker(t)
	doc := RemoteManagementDocument{Version: 1, Revision: "1", MigrationID: marker.ID,
		Grants: []RemoteManagementGrant{{PeerUUID: "peer", Incarnation: requireIncarnation(t, f.mesh), Allowed: true, DecisionRevision: "1"}}}
	writeIncarnationFile(t, filepath.Join(f.dir, RemoteManagementMigrationFile), managementEncode(t, marker))
	writeIncarnationFile(t, filepath.Join(f.dir, RemoteManagementFile), managementEncode(t, doc))
	return f, doc, marker
}

func TestRemoteManagementDocumentStrictFields(t *testing.T) {
	valid := `{"version":1,"revision":"1","migrationId":"` + strings.Repeat("a", 64) + `","grants":[{"peerUuid":"peer","incarnation":"` + strings.Repeat("b", 64) + `","allowed":false,"decisionRevision":"1"}]}`
	doc, err := DecodeRemoteManagementDocument([]byte(valid))
	if err != nil || len(doc.Grants) != 1 || doc.Grants[0].Allowed {
		t.Fatalf("explicit false grant: %+v, %v", doc, err)
	}
	olderDecision := strings.Replace(valid, `"revision":"1"`, `"revision":"2"`, 1)
	olderDoc, err := DecodeRemoteManagementDocument([]byte(olderDecision))
	if err != nil || olderDoc.Revision != "2" || len(olderDoc.Grants) != 1 || olderDoc.Grants[0].DecisionRevision != "1" {
		t.Fatalf("older per-peer decision in a newer document: %+v, %v", olderDoc, err)
	}
	reject := func(name, input string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeRemoteManagementDocument([]byte(input)); err == nil {
				t.Fatal("accepted invalid document")
			}
		})
	}
	reject("missing boolean", strings.Replace(valid, `,"allowed":false`, "", 1))
	reject("null boolean", strings.Replace(valid, `false`, `null`, 1))
	reject("string boolean", strings.Replace(valid, `false`, `"false"`, 1))
	reject("missing decision revision", strings.Replace(valid, `,"decisionRevision":"1"`, "", 1))
	reject("null decision revision", strings.Replace(valid, `"decisionRevision":"1"`, `"decisionRevision":null`, 1))
	reject("numeric decision revision", strings.Replace(valid, `"decisionRevision":"1"`, `"decisionRevision":1`, 1))
	reject("zero decision revision", strings.Replace(valid, `"decisionRevision":"1"`, `"decisionRevision":"0"`, 1))
	reject("noncanonical decision revision", strings.Replace(valid, `"decisionRevision":"1"`, `"decisionRevision":"01"`, 1))
	reject("future decision revision", strings.Replace(valid, `"decisionRevision":"1"`, `"decisionRevision":"2"`, 1))
	reject("overflow decision revision", strings.Replace(valid, `"decisionRevision":"1"`, `"decisionRevision":"18446744073709551616"`, 1))
	reject("duplicate nested field", strings.Replace(valid, `"allowed":false`, `"allowed":true,"allowed":false`, 1))
	reject("duplicate root field", strings.Replace(valid, `"version":1`, `"version":2,"version":1`, 1))
	reject("unknown field", strings.Replace(valid, `"allowed":false`, `"allowed":false,"other":0`, 1))
	reject("leading zero revision", strings.Replace(valid, `"revision":"1"`, `"revision":"01"`, 1))
	reject("zero revision", strings.Replace(valid, `"revision":"1"`, `"revision":"0"`, 1))
	reject("overflow revision", strings.Replace(valid, `"revision":"1"`, `"revision":"18446744073709551616"`, 1))
	reject("unsupported version", strings.Replace(valid, `"version":1`, `"version":2`, 1))
	reject("invalid principal", strings.Replace(valid, `"peer"`, `"peer/name"`, 1))
	reject("whitespace principal", strings.Replace(valid, `"peer"`, `"peer name"`, 1))
	reject("uppercase incarnation", strings.Replace(valid, strings.Repeat("b", 64), strings.Repeat("B", 64), 1))
	reject("trailing data", valid+` {}`)
	reject("oversized file", strings.Repeat(" ", RemoteManagementMaxBytes+1))
	reject("null grants", `{"version":1,"revision":"1","migrationId":"`+strings.Repeat("a", 64)+`","grants":null}`)
	doc.Grants = append(doc.Grants, doc.Grants[0])
	reject("duplicate principal", string(managementEncode(t, doc)))
	doc.Grants = make([]RemoteManagementGrant, remoteManagementMaxPeers+1)
	reject("oversized array", string(managementEncode(t, doc)))
}

func TestRemoteManagementMigrationBindsCapturedSnapshot(t *testing.T) {
	marker := managementMarker(t)
	data := managementEncode(t, marker)
	got, err := DecodeRemoteManagementMigration(data)
	if err != nil || got.Snapshot.Local.AdmissionEpoch != 0 || got.Snapshot.Local.ClusterID != "" {
		t.Fatalf("legacy capture: %+v, %v", got, err)
	}
	marker.Snapshot.Local.AdmissionEpoch = 1
	if _, err := DecodeRemoteManagementMigration(managementEncode(t, marker)); err == nil {
		t.Fatal("accepted changed snapshot with original digest")
	}
	reject := func(name string, mutate func(*RemoteManagementMigration)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			candidate := managementMarker(t)
			mutate(&candidate)
			if _, err := DecodeRemoteManagementMigration(managementEncode(t, candidate)); err == nil {
				t.Fatal("accepted invalid migration")
			}
		})
	}
	reject("null peers", func(m *RemoteManagementMigration) { m.Snapshot.Peers = nil })
	reject("self entry", func(m *RemoteManagementMigration) { m.Snapshot.Peers = append(m.Snapshot.Peers, m.Snapshot.Local) })
	reject("invalid fingerprint", func(m *RemoteManagementMigration) { m.Snapshot.Local.CertFingerprint = "sha256:bad" })
	reject("unknown state", func(m *RemoteManagementMigration) { m.State = "unknown" })
	reject("unknown version", func(m *RemoteManagementMigration) { m.Version = 2 })
}

func TestMeshRemoteManagementFreshDecisions(t *testing.T) {
	f, doc, _ := managementFixture(t)
	if got := f.mesh.RemoteManagement("peer"); !got.Allowed || got.State != "allowed" || got.Revision != "1" || got.Incarnation != doc.Grants[0].Incarnation {
		t.Fatalf("allow = %+v", got)
	}
	f.mesh.Refresh()
	if got := f.mesh.RemoteManagement("peer"); !got.Allowed {
		t.Fatalf("same admission lost grant: %+v", got)
	}
	doc.Revision, doc.Grants[0].Allowed = "2", false
	doc.Grants[0].DecisionRevision = "2"
	writeIncarnationFile(t, filepath.Join(f.dir, RemoteManagementFile), managementEncode(t, doc))
	if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "denied" || got.Revision != "2" {
		t.Fatalf("revoke = %+v", got)
	}
	doc.Grants = []RemoteManagementGrant{}
	writeIncarnationFile(t, filepath.Join(f.dir, RemoteManagementFile), managementEncode(t, doc))
	if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "unreviewed" {
		t.Fatalf("unreviewed = %+v", got)
	}
}

func TestMeshRemoteManagementAdmissionMismatch(t *testing.T) {
	f, _, _ := managementFixture(t)
	writeAdmission(t, f.dir, "cluster", 4)
	if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "unreviewed" {
		t.Fatalf("readmission reused grant: %+v", got)
	}
	f.pin.AdmissionEpoch++
	f.writePin(t)
	if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "unavailable" {
		t.Fatalf("incoherent admission = %+v", got)
	}
}

func TestMeshRemoteManagementAuthorityChangesAcrossPermissionRead(t *testing.T) {
	f, _, _ := managementFixture(t)
	changed := false
	got := f.mesh.remoteManagement("peer", func(path string) ([]byte, error) {
		data, err := os.ReadFile(path)
		if filepath.Base(path) == RemoteManagementFile && !changed {
			changed = true
			writeAdmission(t, f.dir, "cluster", 4)
		}
		return data, err
	})
	if !changed || got.Allowed || got.State != "unavailable" {
		t.Fatalf("authority changed across permission read: %+v, changed=%v", got, changed)
	}
}

func TestMeshRemoteManagementLeafMismatch(t *testing.T) {
	f, _, _ := managementFixture(t)
	peerPEM, _, _ := genLeaf(t, "peer")
	f.pin.CertPem = string(peerPEM)
	f.writePin(t)
	if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "unavailable" {
		t.Fatalf("changed leaf matched cached pin: %+v", got)
	}
}

func TestMeshRemoteManagementUnavailableStorage(t *testing.T) {
	for _, name := range []string{"missing document", "missing marker", "corrupt document", "corrupt marker", "oversize", "pending", "marker mismatch"} {
		t.Run(name, func(t *testing.T) {
			f, doc, marker := managementFixture(t)
			path := filepath.Join(f.dir, RemoteManagementFile)
			switch name {
			case "missing document", "missing marker":
				if name == "missing marker" {
					path = filepath.Join(f.dir, RemoteManagementMigrationFile)
				}
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
			case "corrupt document":
				writeIncarnationFile(t, path, []byte("{"))
			case "corrupt marker":
				writeIncarnationFile(t, filepath.Join(f.dir, RemoteManagementMigrationFile), []byte("{}"))
			case "oversize":
				writeIncarnationFile(t, path, []byte(strings.Repeat(" ", RemoteManagementMaxBytes+1)))
			case "pending":
				marker.State = "pending"
				writeIncarnationFile(t, filepath.Join(f.dir, RemoteManagementMigrationFile), managementEncode(t, marker))
			case "marker mismatch":
				doc.MigrationID = strings.Repeat("b", 64)
				writeIncarnationFile(t, path, managementEncode(t, doc))
			}
			if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "unavailable" {
				t.Fatalf("permission = %+v", got)
			}
		})
	}
}

func TestMeshRemoteManagementChangingOrUnreadableSnapshot(t *testing.T) {
	for _, file := range []string{RemoteManagementFile, RemoteManagementMigrationFile, admissionFileName, membersFileName, "peer.json"} {
		for _, failure := range []string{"changed", "unreadable"} {
			t.Run(file+" "+failure, func(t *testing.T) {
				f, _, _ := managementFixture(t)
				reads := 0
				got := f.mesh.remoteManagement("peer", func(path string) ([]byte, error) {
					data, err := os.ReadFile(path)
					if filepath.Base(path) == file {
						reads++
						if reads == 2 {
							if failure == "unreadable" {
								return nil, fmt.Errorf("private filesystem detail")
							}
							return append(data, ' '), err
						}
					}
					return data, err
				})
				if got.Allowed || got.State != "unavailable" || strings.Contains(got.Reason, "private") {
					t.Fatalf("changing snapshot = %+v", got)
				}
			})
		}
	}
}

func TestMeshRemoteManagementInitializingGuardPresent(t *testing.T) {
	for _, content := range []string{"", "not JSON"} {
		t.Run(fmt.Sprintf("content %q", content), func(t *testing.T) {
			f, _, _ := managementFixture(t)
			writeIncarnationFile(t, filepath.Join(f.dir, RemoteManagementInitializingFile), []byte(content))
			if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "unavailable" {
				t.Fatalf("initializing guard permission = %+v", got)
			}
		})
	}
}

func TestMeshRemoteManagementInitializingGuardReadError(t *testing.T) {
	for _, failRead := range []int{1, 2} {
		t.Run(fmt.Sprintf("guard read %d", failRead), func(t *testing.T) {
			f, _, _ := managementFixture(t)
			guardReads := 0
			got := f.mesh.remoteManagement("peer", func(path string) ([]byte, error) {
				if filepath.Base(path) == RemoteManagementInitializingFile {
					guardReads++
					if guardReads == failRead {
						return nil, os.ErrPermission
					}
				}
				return os.ReadFile(path)
			})
			if guardReads != failRead || got.Allowed || got.State != "unavailable" {
				t.Fatalf("guard read error permission = %+v; reads = %d, want %d", got, guardReads, failRead)
			}
		})
	}
}

func TestMeshRemoteManagementInitializingGuardAppearsDuringRead(t *testing.T) {
	f, _, _ := managementFixture(t)
	created := false
	got := f.mesh.remoteManagement("peer", func(path string) ([]byte, error) {
		data, err := os.ReadFile(path)
		if filepath.Base(path) == RemoteManagementFile && !created {
			created = true
			writeIncarnationFile(t, filepath.Join(f.dir, RemoteManagementInitializingFile), []byte("pending"))
		}
		return data, err
	})
	if !created || got.Allowed || got.State != "unavailable" {
		t.Fatalf("guard appeared during read: %+v; created = %v", got, created)
	}
}
