// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package clustertrust

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/sys/windows"

	"nvpair-shared/durablefile"
)

// Hold the access used by a replacing/deleting writer, without changing the
// bytes or relying on a timed race. The reader's share mask must permit this
// already-open DELETE access even though this handle permits every share mode.
func holdAuthorityDeleteAccess(t *testing.T, path string) {
	t.Helper()
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		t.Fatalf("encode authority path: %v", err)
	}
	handle, err := windows.CreateFile(name, windows.DELETE,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatalf("hold DELETE access: %v", err)
	}
	t.Cleanup(func() {
		if err := windows.CloseHandle(handle); err != nil {
			t.Errorf("close DELETE handle: %v", err)
		}
	})
}

func TestAuthorityReadSharesDeleteAccess(t *testing.T) {
	test := func(name, relativePath string) {
		t.Run(name, func(t *testing.T) {
			f, doc, _ := managementFixture(t)
			path := filepath.Join(f.dir, relativePath)
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read fixture before holding DELETE access: %v", err)
			}
			holdAuthorityDeleteAccess(t, path)
			if _, err := os.ReadFile(path); !errors.Is(err, windows.ERROR_SHARING_VIOLATION) {
				t.Fatalf("old os.ReadFile error = %v, want ERROR_SHARING_VIOLATION", err)
			}
			got, err := readAuthorityFile(path)
			if err != nil || !bytes.Equal(got, want) {
				t.Fatalf("replacement-compatible read: matching bytes=%v, error=%v", bytes.Equal(got, want), err)
			}
			// Open a fresh Mesh so identity and loaded pins cannot be supplied by
			// a cache populated before the conflicting handle was acquired.
			mesh := Open(f.dir)
			mesh.Refresh()
			if got := requireIncarnation(t, mesh); got != doc.Grants[0].Incarnation {
				t.Fatal("DELETE handle changed the service incarnation")
			}
			if got, ok, diagnostic := mesh.PeerServiceIncarnationWithDiagnostic("peer"); !ok || got != doc.Grants[0].Incarnation || diagnostic.Predicate != "" {
				t.Fatalf("diagnostic incarnation: ok=%v, predicate=%q, error code=%d", ok, diagnostic.Predicate, diagnostic.ErrorCode)
			}
			if epoch, ok := mesh.LocalAdmissionEpoch(); !ok || epoch != 3 {
				t.Fatalf("local admission = %d, %v; want 3, true", epoch, ok)
			}
			if epoch, ok := mesh.PeerAdmissionEpoch("peer"); !ok || epoch != 7 {
				t.Fatalf("peer admission = %d, %v; want 7, true", epoch, ok)
			}
			if got := mesh.RemoteManagement("peer"); !got.Allowed || got.State != "allowed" || got.Incarnation != doc.Grants[0].Incarnation {
				t.Fatalf("permission: allowed=%v, state=%q, predicate=%q, error code=%d", got.Allowed, got.State, got.Diagnostic.Predicate, got.Diagnostic.ErrorCode)
			}
		})
	}
	test("pin", filepath.Join("trusted", "peer.json"))
	test("members", membersFileName)
	test("admission", admissionFileName)
	test("permission document", RemoteManagementFile)
	test("permission marker", RemoteManagementMigrationFile)
	test("identity certificate", "node.crt")
	test("identity key", "node.key")
}

// Use the production replacement primitive, but fail on its first conflict.
// A retry must never hide a missing FILE_SHARE_DELETE on the authority reader.
func replaceAuthorityFile(t *testing.T, path string, data []byte) {
	t.Helper()
	replacer := durablefile.Replacer{
		IsTransient: func(error) bool { return false },
		Sleep: func(time.Duration) {
			t.Fatal("authority replacement unexpectedly retried")
		},
	}
	if err := replacer.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("replace authority: %v", err)
	}
}

func TestAuthorityReadClosesBeforeReplacement(t *testing.T) {
	f, doc, _ := managementFixture(t)
	path := filepath.Join(f.dir, "trusted", "peer.json")
	oldData := managementEncode(t, f.pin)
	writeIncarnationFile(t, path, oldData)
	retained, err := readAuthorityFile(path)
	if err != nil || !bytes.Equal(retained, oldData) {
		t.Fatalf("read before replacement: original bytes=%v, error=%v", bytes.Equal(retained, oldData), err)
	}
	// A compatible leaked handle no longer blocks replacement. An exclusive
	// open proves ReadFile actually released its handle before returning.
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		t.Fatalf("encode pin path: %v", err)
	}
	exclusive, err := windows.CreateFile(name, windows.GENERIC_READ, 0,
		nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatalf("exclusive open after completed read: %v", err)
	}
	if err := windows.CloseHandle(exclusive); err != nil {
		t.Fatalf("close exclusive reader: %v", err)
	}
	// A fresh open after the read observes the new complete file.
	newData := append(bytes.Clone(oldData), '\n')
	replaceAuthorityFile(t, path, newData)
	current, err := readAuthorityFile(path)
	if err != nil || !bytes.Equal(current, newData) {
		t.Fatalf("fresh path after replacement: new bytes=%v, error=%v", bytes.Equal(current, newData), err)
	}
	f.mesh.Refresh()
	if got := requireIncarnation(t, f.mesh); got != doc.Grants[0].Incarnation {
		t.Fatal("metadata replacement changed authority")
	}
	if got := f.mesh.RemoteManagement("peer"); !got.Allowed {
		t.Fatalf("metadata replacement lost grant: state=%q, predicate=%q", got.State, got.Diagnostic.Predicate)
	}
}

func TestAuthorityReaderCoexistsWithReplacement(t *testing.T) {
	f, doc, _ := managementFixture(t)
	path := filepath.Join(f.dir, "trusted", "peer.json")
	oldData, err := readAuthorityFile(path)
	if err != nil {
		t.Fatalf("read original pin: %v", err)
	}
	reader, err := openAuthorityFile(path)
	if err != nil {
		t.Fatalf("hold authority reader: %v", err)
	}
	t.Cleanup(func() {
		if err := reader.Close(); err != nil {
			t.Errorf("close held authority reader: %v", err)
		}
	})
	newData := append(bytes.Clone(oldData), '\n')
	replaceAuthorityFile(t, path, newData)
	retained, err := io.ReadAll(reader)
	if err != nil || !bytes.Equal(retained, oldData) {
		t.Fatalf("held reader: original bytes=%v, error=%v", bytes.Equal(retained, oldData), err)
	}
	current, err := readAuthorityFile(path)
	if err != nil || !bytes.Equal(current, newData) {
		t.Fatalf("fresh reader: replacement bytes=%v, error=%v", bytes.Equal(current, newData), err)
	}
	f.mesh.Refresh()
	if got := requireIncarnation(t, f.mesh); got != doc.Grants[0].Incarnation {
		t.Fatal("metadata replacement changed authority")
	}
	if got := f.mesh.RemoteManagement("peer"); !got.Allowed {
		t.Fatalf("metadata replacement lost grant: state=%q, predicate=%q", got.State, got.Diagnostic.Predicate)
	}
}

func TestAuthorityReplacementDuringReadRefuses(t *testing.T) {
	f := newIncarnationFixture(t)
	pinPath := filepath.Join(f.dir, "trusted", "peer.json")
	replaced := false
	var diagnostic PeerServiceDiagnostic
	got, ok := f.mesh.peerServiceIncarnationDiagnostic("peer", func(path string) ([]byte, error) {
		data, err := readAuthorityFile(path)
		if err == nil && path == pinPath && !replaced {
			held, err := openAuthorityFile(path)
			if err != nil {
				t.Fatalf("hold pin reader during authority replacement: %v", err)
			}
			defer func() {
				if err := held.Close(); err != nil {
					t.Errorf("close held pin reader: %v", err)
				}
			}()
			replaced = true
			f.pin.AdmissionEpoch++
			replaceAuthorityFile(t, path, managementEncode(t, f.pin))
		}
		return data, err
	}, &diagnostic)
	if !replaced || ok || got != "" || diagnostic.Predicate != "pin_changed" {
		t.Fatalf("changed authority: replaced=%v, ok=%v, predicate=%q, error code=%d", replaced, ok, diagnostic.Predicate, diagnostic.ErrorCode)
	}
	f.mesh.Refresh()
	requireNoIncarnation(t, f.mesh, "peer")
}

func TestAuthorityHeldReaderDoesNotPreserveRevokedGrant(t *testing.T) {
	f, doc, _ := managementFixture(t)
	path := filepath.Join(f.dir, RemoteManagementFile)
	oldData, err := readAuthorityFile(path)
	if err != nil {
		t.Fatalf("read original permission: %v", err)
	}
	held, err := openAuthorityFile(path)
	if err != nil {
		t.Fatalf("hold permission reader: %v", err)
	}
	t.Cleanup(func() {
		if err := held.Close(); err != nil {
			t.Errorf("close held permission reader: %v", err)
		}
	})
	doc.Revision, doc.Grants[0].DecisionRevision = "2", "2"
	doc.Grants[0].Allowed = false
	replaceAuthorityFile(t, path, managementEncode(t, doc))
	retained, err := io.ReadAll(held)
	if err != nil || !bytes.Equal(retained, oldData) {
		t.Fatalf("held permission reader: original bytes=%v, error=%v", bytes.Equal(retained, oldData), err)
	}
	if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "denied" || got.Revision != "2" {
		t.Fatalf("revoked grant: allowed=%v, state=%q, revision=%q", got.Allowed, got.State, got.Revision)
	}
}

func denyAuthorityRead(t *testing.T, path string) {
	t.Helper()
	original, err := windows.GetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION)
	if err != nil {
		t.Fatalf("read fixture DACL: %v", err)
	}
	originalDACL, _, err := original.DACL()
	if err != nil {
		t.Fatalf("extract fixture DACL: %v", err)
	}
	// Deny read to Everyone, including the current owner. Ownership still
	// permits restoring the DACL; this touches only a disposable fixture file.
	denied, err := windows.SecurityDescriptorFromString("D:(D;;FR;;;WD)(A;;FA;;;WD)")
	if err != nil {
		t.Fatalf("create denied-read DACL: %v", err)
	}
	deniedDACL, _, err := denied.DACL()
	if err != nil {
		t.Fatalf("extract denied-read DACL: %v", err)
	}
	if err := windows.SetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION, nil, nil, deniedDACL, nil); err != nil {
		t.Fatalf("deny fixture read: %v", err)
	}
	t.Cleanup(func() {
		if err := windows.SetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION, nil, nil, originalDACL, nil); err != nil {
			t.Errorf("restore fixture DACL: %v", err)
		}
	})
}

func TestAuthorityReadFailuresRemainClosed(t *testing.T) {
	t.Run("missing pin", func(t *testing.T) {
		f, _, _ := managementFixture(t)
		path := filepath.Join(f.dir, "trusted", "peer.json")
		if err := os.Remove(path); err != nil {
			t.Fatalf("remove pin: %v", err)
		}
		if _, err := readAuthorityFile(path); !os.IsNotExist(err) {
			t.Fatalf("missing authority error = %v, want not exist", err)
		}
		requireNoIncarnation(t, f.mesh, "peer")
		if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "unavailable" {
			t.Fatalf("missing authority: allowed=%v, state=%q", got.Allowed, got.State)
		}
		f.mesh.Refresh()
		if f.mesh.HasPin("peer") {
			t.Fatal("refresh retained removed pin")
		}
	})
	t.Run("denied pin read", func(t *testing.T) {
		f, _, _ := managementFixture(t)
		path := filepath.Join(f.dir, "trusted", "peer.json")
		denyAuthorityRead(t, path)
		if _, err := readAuthorityFile(path); !errors.Is(err, windows.ERROR_ACCESS_DENIED) {
			t.Fatalf("denied authority error = %v, want ERROR_ACCESS_DENIED", err)
		}
		got, ok, diagnostic := f.mesh.PeerServiceIncarnationWithDiagnostic("peer")
		if ok || got != "" || diagnostic.Predicate != "pin_read" || diagnostic.ErrorCode != uint64(windows.ERROR_ACCESS_DENIED) {
			t.Fatalf("denied incarnation: ok=%v, predicate=%q, error code=%d", ok, diagnostic.Predicate, diagnostic.ErrorCode)
		}
		f.mesh.Refresh()
		requireNoIncarnation(t, f.mesh, "peer")
		if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "unavailable" {
			t.Fatalf("denied authority: allowed=%v, state=%q", got.Allowed, got.State)
		}
	})
	t.Run("denied permission read", func(t *testing.T) {
		f, _, _ := managementFixture(t)
		denyAuthorityRead(t, filepath.Join(f.dir, RemoteManagementFile))
		if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "unavailable" || got.Diagnostic.Predicate != "permission_document_read" || got.Diagnostic.ErrorCode != uint64(windows.ERROR_ACCESS_DENIED) {
			t.Fatalf("denied permission: allowed=%v, state=%q, predicate=%q, error code=%d", got.Allowed, got.State, got.Diagnostic.Predicate, got.Diagnostic.ErrorCode)
		}
	})
	t.Run("revoked grant with DELETE access held", func(t *testing.T) {
		f, doc, _ := managementFixture(t)
		path := filepath.Join(f.dir, RemoteManagementFile)
		doc.Revision, doc.Grants[0].DecisionRevision = "2", "2"
		doc.Grants[0].Allowed = false
		replaceAuthorityFile(t, path, managementEncode(t, doc))
		holdAuthorityDeleteAccess(t, path)
		if got := f.mesh.RemoteManagement("peer"); got.Allowed || got.State != "denied" || got.Revision != "2" || got.Diagnostic.Predicate != "permission_denied" || got.Diagnostic.ErrorCode != 0 {
			t.Fatalf("revoked permission: allowed=%v, state=%q, revision=%q, predicate=%q, error code=%d", got.Allowed, got.State, got.Revision, got.Diagnostic.Predicate, got.Diagnostic.ErrorCode)
		}
	})
}
