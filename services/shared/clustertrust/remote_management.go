// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"unicode"
)

const (
	RemoteManagementFile             = "remote-management.json"
	RemoteManagementMigrationFile    = "remote-management-migration.json"
	RemoteManagementInitializingFile = "remote-management-initializing.json"
	RemoteManagementMaxBytes         = 1 << 20
	remoteManagementMaxPeers         = 1024
)

type RemoteManagementGrant struct {
	PeerUUID    string `json:"peerUuid"`
	Incarnation string `json:"incarnation"`
	Allowed     bool   `json:"allowed"`
	// The document revision of this peer's last local decision, including a
	// same-value choice. Unrelated peer decisions preserve it.
	DecisionRevision string `json:"decisionRevision"`
}

type RemoteManagementDocument struct {
	Version     int                     `json:"version"`
	Revision    string                  `json:"revision"`
	MigrationID string                  `json:"migrationId"`
	Grants      []RemoteManagementGrant `json:"grants"`
}

type RemoteManagementMigrationAuthority struct {
	PeerUUID        string `json:"peerUuid"`
	CertFingerprint string `json:"certFingerprint"`
	ClusterID       string `json:"clusterId"`
	AdmissionEpoch  uint64 `json:"admissionEpoch"`
}

type RemoteManagementMigrationSnapshot struct {
	Local RemoteManagementMigrationAuthority   `json:"local"`
	Peers []RemoteManagementMigrationAuthority `json:"peers"`
}

type RemoteManagementMigration struct {
	Version  int                               `json:"version"`
	ID       string                            `json:"id"`
	State    string                            `json:"state"`
	Snapshot RemoteManagementMigrationSnapshot `json:"snapshot"`
}

func remoteManagementPrincipal(s string) bool {
	return s != "" && len(s) <= 128 && s != "." && s != ".." && !strings.ContainsAny(s, "/\\:\x00") &&
		strings.IndexFunc(s, func(r rune) bool { return unicode.IsSpace(r) || unicode.IsControl(r) }) == -1
}

func remoteManagementDigest(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// remoteManagementJSON rejects duplicate keys before decoding, including nested objects.
func remoteManagementJSON(d *json.Decoder, depth int) error {
	if depth > 16 {
		return fmt.Errorf("excessive nesting")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[name] {
				return fmt.Errorf("duplicate or invalid key")
			}
			seen[name] = true
			if err := remoteManagementJSON(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := remoteManagementJSON(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("invalid delimiter")
	}
	_, err = d.Token()
	return err
}

// Every DTO field is required and null is never a value in this storage schema.
func remoteManagementShape(data []byte, typ reflect.Type) error {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return fmt.Errorf("null value")
	}
	switch typ.Kind() {
	case reflect.Struct:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			return err
		}
		if len(fields) != typ.NumField() {
			return fmt.Errorf("missing or unknown fields")
		}
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			raw, ok := fields[field.Tag.Get("json")]
			if !ok {
				return fmt.Errorf("missing field")
			}
			if err := remoteManagementShape(raw, field.Type); err != nil {
				return err
			}
		}
	case reflect.Slice:
		var entries []json.RawMessage
		if err := json.Unmarshal(data, &entries); err != nil {
			return err
		}
		if len(entries) > remoteManagementMaxPeers {
			return fmt.Errorf("too many peers")
		}
		for _, entry := range entries {
			if err := remoteManagementShape(entry, typ.Elem()); err != nil {
				return err
			}
		}
	}
	return nil
}

func decodeRemoteManagement(data []byte, dst interface{}) error {
	if len(data) > RemoteManagementMaxBytes {
		return fmt.Errorf("permission storage too large")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	if err := remoteManagementJSON(d, 0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("trailing data")
	}
	if err := remoteManagementShape(data, reflect.TypeOf(dst).Elem()); err != nil {
		return err
	}
	return json.Unmarshal(data, dst)
}

func DecodeRemoteManagementDocument(data []byte) (RemoteManagementDocument, error) {
	var doc RemoteManagementDocument
	if err := decodeRemoteManagement(data, &doc); err != nil {
		return doc, err
	}
	revision, err := strconv.ParseUint(doc.Revision, 10, 64)
	if doc.Version != 1 || err != nil || revision == 0 || strconv.FormatUint(revision, 10) != doc.Revision || !remoteManagementDigest(doc.MigrationID) {
		return RemoteManagementDocument{}, fmt.Errorf("invalid permission document")
	}
	seen := make(map[string]bool)
	for _, grant := range doc.Grants {
		decisionRevision, err := strconv.ParseUint(grant.DecisionRevision, 10, 64)
		if !remoteManagementPrincipal(grant.PeerUUID) || !remoteManagementDigest(grant.Incarnation) || seen[grant.PeerUUID] ||
			err != nil || decisionRevision == 0 || decisionRevision > revision || strconv.FormatUint(decisionRevision, 10) != grant.DecisionRevision {
			return RemoteManagementDocument{}, fmt.Errorf("invalid grant")
		}
		seen[grant.PeerUUID] = true
	}
	return doc, nil
}

func RemoteManagementMigrationID(snapshot RemoteManagementMigrationSnapshot) (string, error) {
	valid := func(a RemoteManagementMigrationAuthority) bool {
		return remoteManagementPrincipal(a.PeerUUID) && strings.HasPrefix(a.CertFingerprint, "sha256:") && remoteManagementDigest(strings.TrimPrefix(a.CertFingerprint, "sha256:"))
	}
	if !valid(snapshot.Local) || snapshot.Peers == nil || len(snapshot.Peers) > remoteManagementMaxPeers {
		return "", fmt.Errorf("invalid migration snapshot")
	}
	seen := map[string]bool{snapshot.Local.PeerUUID: true}
	for _, peer := range snapshot.Peers {
		if !valid(peer) || seen[peer.PeerUUID] {
			return "", fmt.Errorf("invalid migration peer")
		}
		seen[peer.PeerUUID] = true
	}
	data, err := json.Marshal(snapshot)
	if err != nil {
		return "", err
	}
	if len(data) > RemoteManagementMaxBytes {
		return "", fmt.Errorf("migration snapshot too large")
	}
	return fmt.Sprintf("%x", sha256.Sum256(data)), nil
}

func DecodeRemoteManagementMigration(data []byte) (RemoteManagementMigration, error) {
	var marker RemoteManagementMigration
	if err := decodeRemoteManagement(data, &marker); err != nil {
		return marker, err
	}
	id, err := RemoteManagementMigrationID(marker.Snapshot)
	if err != nil || marker.Version != 1 || marker.ID != id || (marker.State != "pending" && marker.State != "complete") {
		return RemoteManagementMigration{}, fmt.Errorf("invalid migration marker")
	}
	return marker, nil
}

type RemoteManagementPermission struct {
	Allowed     bool
	State       string
	Reason      string
	Revision    string
	Incarnation string
	Diagnostic  PeerServiceDiagnostic
}

func (p RemoteManagementPermission) refused(predicate string, err error) RemoteManagementPermission {
	p.Diagnostic.refuse(predicate, err)
	return p
}

func (m *Mesh) RemoteManagement(peerUUID string) RemoteManagementPermission {
	return m.remoteManagement(peerUUID, func(path string) ([]byte, error) {
		f, err := openAuthorityFile(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return io.ReadAll(io.LimitReader(f, RemoteManagementMaxBytes+1))
	})
}

func (m *Mesh) remoteManagement(peerUUID string, readFile func(string) ([]byte, error)) RemoteManagementPermission {
	unavailable := RemoteManagementPermission{State: "unavailable", Reason: "Remote management permission is unavailable."}
	if m == nil || m.dir == "" || !remoteManagementPrincipal(peerUUID) {
		return unavailable.refused("permission_peer_invalid", nil)
	}
	guardPath := filepath.Join(m.dir, RemoteManagementInitializingFile)
	if _, err := readFile(guardPath); !os.IsNotExist(err) {
		return unavailable.refused("permission_initializing_guard", err)
	}
	before, ok := m.peerServiceIncarnationDiagnostic(peerUUID, readFile, &unavailable.Diagnostic)
	if !ok {
		return unavailable
	}
	unavailable.Diagnostic.ExpectedIncarnation = before
	markerPath := filepath.Join(m.dir, RemoteManagementMigrationFile)
	docPath := filepath.Join(m.dir, RemoteManagementFile)
	markerData, err := readFile(markerPath)
	if err != nil {
		return unavailable.refused("permission_marker_read", err)
	}
	marker, err := DecodeRemoteManagementMigration(markerData)
	if err != nil {
		return unavailable.refused("permission_marker_invalid", nil)
	}
	if marker.State != "complete" {
		return unavailable.refused("permission_migration_incomplete", nil)
	}
	docData, err := readFile(docPath)
	if err != nil {
		return unavailable.refused("permission_document_read", err)
	}
	doc, err := DecodeRemoteManagementDocument(docData)
	if err != nil {
		return unavailable.refused("permission_document_invalid", nil)
	}
	if doc.MigrationID != marker.ID {
		return unavailable.refused("permission_migration_mismatch", nil)
	}
	currentDoc, err := readFile(docPath)
	if err != nil {
		return unavailable.refused("permission_document_bracket_read", err)
	}
	if !bytes.Equal(docData, currentDoc) {
		return unavailable.refused("permission_document_changed", nil)
	}
	currentMarker, err := readFile(markerPath)
	if err != nil {
		return unavailable.refused("permission_marker_bracket_read", err)
	}
	if !bytes.Equal(markerData, currentMarker) {
		return unavailable.refused("permission_marker_changed", nil)
	}
	after, ok := m.peerServiceIncarnationDiagnostic(peerUUID, readFile, &unavailable.Diagnostic)
	unavailable.Diagnostic.ObservedIncarnation = after
	if !ok {
		return unavailable
	}
	if before != after {
		return unavailable.refused("permission_incarnation_changed", nil)
	}
	if _, err := readFile(guardPath); !os.IsNotExist(err) {
		return unavailable.refused("permission_initializing_guard_changed", err)
	}
	result := RemoteManagementPermission{State: "unreviewed", Reason: "Remote management has not been granted.", Revision: doc.Revision, Incarnation: before, Diagnostic: unavailable.Diagnostic}
	result.Diagnostic.Predicate = "permission_unreviewed"
	for _, grant := range doc.Grants {
		if grant.PeerUUID != peerUUID || grant.Incarnation != before {
			continue
		}
		result.Allowed = grant.Allowed
		result.State, result.Reason = "denied", "Remote management is blocked on this device."
		result.Diagnostic.Predicate = "permission_denied"
		if grant.Allowed {
			result.State, result.Reason = "allowed", "Remote management is allowed."
			result.Diagnostic.Predicate = ""
		}
		break
	}
	return result
}
