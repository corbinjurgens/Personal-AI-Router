// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func TestPeerServiceAuthorityDiagnosticPreservesDecision(t *testing.T) {
	f := newIncarnationFixture(t)
	want := requireIncarnation(t, f.mesh)
	got, ok, diagnostic := f.mesh.PeerServiceIncarnationWithDiagnostic("peer")
	if !ok || got != want || diagnostic.Predicate != "" || diagnostic.ErrorCode != 0 ||
		diagnostic.NodeID != "self" || diagnostic.ClusterID != "cluster" || diagnostic.Epoch != 3 || diagnostic.Counter != 3 {
		t.Fatalf("successful authority diagnostic = %q, %v, %+v", got, ok, diagnostic)
	}
}

func TestPeerServiceAuthorityDiagnosticCapturesDecidingRead(t *testing.T) {
	t.Run("numeric read failure", func(t *testing.T) {
		f := newIncarnationFixture(t)
		var diagnostic PeerServiceDiagnostic
		reads := 0
		incarnation, ok := f.mesh.peerServiceIncarnationDiagnostic("peer", func(path string) ([]byte, error) {
			if path == filepath.Join(f.dir, membersFileName) {
				reads++
				return nil, &os.PathError{Op: "open", Path: "unlogged-path-sentinel", Err: syscall.Errno(5)}
			}
			return os.ReadFile(path)
		}, &diagnostic)
		if ok || incarnation != "" || reads != 1 || diagnostic.Predicate != "members_read" || diagnostic.ErrorCode != 5 ||
			diagnostic.NodeID != "self" || diagnostic.ClusterID != "cluster" || diagnostic.Epoch != 3 || diagnostic.Counter != 3 {
			t.Fatalf("read refusal = %q, %v, reads=%d, %+v", incarnation, ok, reads, diagnostic)
		}
	})
	t.Run("changed bracket", func(t *testing.T) {
		f := newIncarnationFixture(t)
		var diagnostic PeerServiceDiagnostic
		reads := 0
		_, ok := f.mesh.peerServiceIncarnationDiagnostic("peer", func(path string) ([]byte, error) {
			data, err := os.ReadFile(path)
			if path == filepath.Join(f.dir, membersFileName) {
				reads++
				if reads == 2 && err == nil {
					data = append(data, '\n')
				}
			}
			return data, err
		}, &diagnostic)
		if ok || reads != 2 || diagnostic.Predicate != "members_changed" || diagnostic.ErrorCode != 0 ||
			!diagnostic.ChangedComponentCompared || !diagnostic.ChangedComponentSame {
			t.Fatalf("changed snapshot = %v, reads=%d, %+v", ok, reads, diagnostic)
		}
	})
}

func TestPeerServiceAuthorityChangedComponentDiagnostic(t *testing.T) {
	test := func(name, file, predicate string, compared, same bool, change func([]byte) []byte) {
		t.Run(name, func(t *testing.T) {
			f := newIncarnationFixture(t)
			var diagnostic PeerServiceDiagnostic
			reads := 0
			_, ok := f.mesh.peerServiceIncarnationDiagnostic("peer", func(path string) ([]byte, error) {
				data, err := os.ReadFile(path)
				if path == filepath.Join(f.dir, file) {
					reads++
					if reads == 2 && err == nil {
						changed := change(data)
						if bytes.Equal(changed, data) {
							t.Fatal("fixture did not change bracket bytes")
						}
						data = changed
					}
				}
				return data, err
			}, &diagnostic)
			if ok || reads != 2 || diagnostic.Predicate != predicate ||
				diagnostic.ChangedComponentCompared != compared || diagnostic.ChangedComponentSame != same {
				t.Fatalf("changed component = %v, reads=%d, %+v", ok, reads, diagnostic)
			}
		})
	}
	metadata := func(data []byte) []byte { return append(data, '\n') }
	test("pin metadata only", "trusted/peer.json", "pin_changed", true, true, metadata)
	test("admission metadata only", "admission.json", "admission_changed", true, true, metadata)
	test("member authority changed", membersFileName, "members_changed", true, false, func(data []byte) []byte {
		return bytes.Replace(data, []byte(`"state":"member"`), []byte(`"state":"departed"`), 1)
	})
	test("incomplete member rewrite", membersFileName, "members_changed", false, false, func([]byte) []byte { return []byte(`{`) })
}

func TestRemoteManagementDiagnosticCapturesOriginalPredicate(t *testing.T) {
	t.Run("permission storage", func(t *testing.T) {
		f, _, _ := managementFixture(t)
		reads := 0
		permission := f.mesh.remoteManagement("peer", func(path string) ([]byte, error) {
			if path == filepath.Join(f.dir, RemoteManagementFile) {
				reads++
				return nil, &os.PathError{Op: "read", Path: "unlogged-secret-path", Err: syscall.Errno(5)}
			}
			return os.ReadFile(path)
		})
		if permission.Allowed || permission.State != "unavailable" || reads != 1 ||
			permission.Diagnostic.Predicate != "permission_document_read" || permission.Diagnostic.ErrorCode != 5 ||
			permission.Diagnostic.NodeID != "self" || permission.Diagnostic.Epoch != 3 {
			t.Fatalf("permission refusal = %+v, reads=%d", permission, reads)
		}
	})
	t.Run("explicit denial and success", func(t *testing.T) {
		f, doc, _ := managementFixture(t)
		doc.Grants[0].Allowed = false
		path := filepath.Join(f.dir, RemoteManagementFile)
		writeIncarnationFile(t, path, managementEncode(t, doc))
		permission := f.mesh.RemoteManagement("peer")
		if permission.Allowed || permission.State != "denied" || permission.Diagnostic.Predicate != "permission_denied" {
			t.Fatalf("explicit refusal = %+v", permission)
		}
		doc.Grants[0].Allowed = true
		writeIncarnationFile(t, path, managementEncode(t, doc))
		if permission := f.mesh.RemoteManagement("peer"); !permission.Allowed || permission.Diagnostic.Predicate != "" {
			t.Fatalf("successful permission = %+v", permission)
		}
	})
	t.Run("authority error before permission", func(t *testing.T) {
		f, _, _ := managementFixture(t)
		permission := f.mesh.remoteManagement("peer", func(path string) ([]byte, error) {
			if path == filepath.Join(f.dir, membersFileName) {
				return nil, errors.New("unlogged-error-sentinel")
			}
			return os.ReadFile(path)
		})
		if permission.Allowed || permission.Diagnostic.Predicate != "members_read" || permission.Diagnostic.ErrorCode != 0 {
			t.Fatalf("authority refusal = %+v", permission)
		}
	})
}
