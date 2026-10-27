// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package nodeid

import (
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/windows"
)

func TestResolveSharesDeleteAccessForClusterIdentity(t *testing.T) {
	base := t.TempDir()
	cluster := filepath.Join(base, "cluster")
	if err := os.Mkdir(cluster, 0o700); err != nil {
		t.Fatal(err)
	}
	const want = "11111111-2222-4333-8444-555555555555"
	path := filepath.Join(cluster, "identity.json")
	if err := os.WriteFile(path, []byte(`{"node_uuid":"`+want+`","created_at":1}`), 0o600); err != nil {
		t.Fatal(err)
	}
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		t.Fatal(err)
	}
	handle, err := windows.CreateFile(name, windows.DELETE,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := windows.CloseHandle(handle); err != nil {
			t.Errorf("close identity handle: %v", err)
		}
	})
	if got := Resolve(base); got != want {
		t.Fatalf("Resolve with DELETE handle held = %q, want cluster identity %q", got, want)
	}
	if _, err := os.Stat(filepath.Join(base, "node-id.json")); !os.IsNotExist(err) {
		t.Fatalf("fallback identity unexpectedly created: %v", err)
	}
}
