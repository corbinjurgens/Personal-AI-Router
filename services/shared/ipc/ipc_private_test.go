// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package ipc

import (
	"os"
	"path/filepath"
	"testing"
)

func TestListenPrivateCreatesOwnerOnlySocket(t *testing.T) {
	// Not t.TempDir: its path embeds the test name and can overrun the Unix
	// socket path limit (104 bytes on macOS).
	dir, err := os.MkdirTemp("", "ipc")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	path := filepath.Join(dir, "p.sock")

	l, err := ListenPrivate(path)
	if err != nil {
		t.Fatalf("ListenPrivate: %v", err)
	}
	defer l.Close()

	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if mode := st.Mode().Perm(); mode != 0o600 {
		t.Fatalf("socket mode = %o, want 600", mode)
	}

	go func() {
		c, err := l.Accept()
		if err == nil {
			_, _ = c.Write([]byte("ok"))
			_ = c.Close()
		}
	}()
	c, err := Dial(path)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()
	buf := make([]byte, 2)
	if _, err := c.Read(buf); err != nil || string(buf) != "ok" {
		t.Fatalf("read = %q, %v", buf, err)
	}
}
