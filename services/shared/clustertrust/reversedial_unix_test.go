// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package clustertrust

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestReverseDialPrivateEndpointForceKilledStartupCheckpoints(t *testing.T) {
	if path := os.Getenv("PAIR_REVERSE_PRIVATE_CHECKPOINT_PATH"); path != "" {
		wanted := os.Getenv("PAIR_REVERSE_PRIVATE_CHECKPOINT")
		listener, err := listenReversePrivate(path, func(stage string) {
			if stage != wanted {
				return
			}
			if stage == "recording" {
				if err := os.WriteFile(path+".lock", []byte(`{"version":`), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := fmt.Fprintln(os.Stdout, "ready "+stage); err != nil {
				t.Fatal(err)
			}
			<-time.After(time.Hour)
		})
		if err == nil {
			err = listener.Close()
		}
		t.Fatalf("startup did not stop at checkpoint %q: %v", wanted, err)
	}
	for _, stage := range []string{"before-bind", "bound", "recording", "recorded", "published"} {
		t.Run(stage, func(t *testing.T) {
			path := ReverseDialPath(reverseDialConfigFixture(t))
			reversePrivateKillAtCheckpoint(t, path, stage)
			info, err := os.Lstat(path)
			if stage == "published" {
				if err != nil || info.Mode()&os.ModeSocket == 0 {
					t.Fatalf("published checkpoint lacks canonical socket: info=%v, err=%v", info, err)
				}
			} else if !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("canonical socket appeared before durable publication: info=%v, err=%v", info, err)
			}
			entries, err := os.ReadDir(filepath.Dir(path))
			if err != nil {
				t.Fatal(err)
			}
			stagingSockets := 0
			for _, entry := range entries {
				entryInfo, err := entry.Info()
				if err != nil {
					t.Fatal(err)
				}
				if entry.Name() != filepath.Base(path) && entryInfo.Mode()&os.ModeSocket != 0 {
					stagingSockets++
				}
			}
			wantStaging := 1
			if stage == "before-bind" {
				wantStaging = 0
			}
			if stagingSockets != wantStaging {
				t.Fatalf("checkpoint %s left %d staging sockets, want %d", stage, stagingSockets, wantStaging)
			}
			if stage == "bound" {
				if record, err := os.ReadFile(path + ".lock"); err != nil || len(record) != 0 {
					t.Fatalf("bound checkpoint is after ownership recording: %q, %v", record, err)
				}
			}
			var recordedStage string
			if stage == "recorded" || stage == "published" {
				record, err := os.ReadFile(path + ".lock")
				if err != nil {
					t.Fatal(err)
				}
				var evidence struct {
					Version int    `json:"version"`
					Device  uint64 `json:"device"`
					Inode   uint64 `json:"inode"`
					Stage   string `json:"stage"`
				}
				if err := json.Unmarshal(record, &evidence); err != nil {
					t.Fatal(err)
				}
				recordedStage = filepath.Join(filepath.Dir(path), evidence.Stage)
				stageInfo, err := os.Lstat(recordedStage)
				if err != nil {
					t.Fatal(err)
				}
				stat := stageInfo.Sys().(*syscall.Stat_t)
				if evidence.Version != 1 || evidence.Stage == "" || evidence.Inode != uint64(stat.Ino) || evidence.Device != uint64(stat.Dev) {
					t.Fatalf("record does not identify staging socket: %+v", evidence)
				}
				if stage == "published" && !os.SameFile(info, stageInfo) {
					t.Fatal("published endpoint differs from recorded staging inode")
				}
			}
			listener, err := ListenReversePrivate(path)
			if err != nil {
				t.Fatalf("restart after checkpoint %s: %v", stage, err)
			}
			t.Cleanup(func() {
				if err := listener.Close(); err != nil {
					t.Errorf("close checkpoint replacement: %v", err)
				}
			})
			if recordedStage != "" {
				if _, err := os.Lstat(recordedStage); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("recorded staging socket not reclaimed: %v", err)
				}
			}
			reverseDialAssertListenerReachable(t, path, listener)
		})
	}
}

func reversePrivateKillAtCheckpoint(t *testing.T, path, stage string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestReverseDialPrivateEndpointForceKilledStartupCheckpoints$")
	cmd.Env = append(os.Environ(), "PAIR_REVERSE_PRIVATE_CHECKPOINT_PATH="+path, "PAIR_REVERSE_PRIVATE_CHECKPOINT="+stage)
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waited := false
	t.Cleanup(func() {
		if !waited {
			if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
				t.Errorf("kill checkpoint fixture: %v", err)
			}
			_ = cmd.Wait()
		}
	})
	if ready, err := bufio.NewReader(stdout).ReadString('\n'); err != nil || ready != "ready "+stage+"\n" {
		t.Fatalf("startup checkpoint not reached: %q, %v", ready, err)
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	err = cmd.Wait()
	waited = true
	var killed *exec.ExitError
	if !errors.As(err, &killed) {
		t.Fatalf("checkpoint fixture did not exit by force: %v", err)
	}
}

func TestReverseDialPrivateEndpointCloseRetainsEvidenceWhenUnlinkFails(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root bypasses directory write permissions")
	}
	base, first := reverseDialFixture(t)
	path := ReverseDialPath(base)
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	record, err := os.ReadFile(path + ".lock")
	if err != nil {
		t.Fatal(err)
	}
	parent := filepath.Dir(path)
	if err := os.Chmod(parent, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chmod(parent, 0o700); err != nil {
			t.Errorf("restore endpoint directory permissions: %v", err)
		}
	})
	if err := first.Close(); !errors.Is(err, os.ErrPermission) {
		t.Fatalf("unwritable parent did not prevent socket removal: %v", err)
	}
	after, err := os.Stat(path)
	if err != nil || !os.SameFile(before, after) {
		t.Fatalf("owned socket unexpectedly removed: info=%v, err=%v", after, err)
	}
	retained, err := os.ReadFile(path + ".lock")
	if err != nil || string(retained) != string(record) {
		t.Fatalf("failed unlink lost ownership evidence: %q, %v", retained, err)
	}
	if err := os.Chmod(parent, 0o700); err != nil {
		t.Fatal(err)
	}
	replacement, err := ListenReversePrivate(path)
	if err != nil {
		t.Fatalf("restart after restoring directory permissions: %v", err)
	}
	t.Cleanup(func() {
		if err := replacement.Close(); err != nil {
			t.Errorf("close permission-restored replacement: %v", err)
		}
	})
	if err := first.Close(); !errors.Is(err, os.ErrPermission) {
		t.Fatalf("repeated Close changed the first result: %v", err)
	}
	reverseDialAssertListenerReachable(t, path, replacement)
}

func TestReverseDialPrivateEndpointClosePreservesDifferentInode(t *testing.T) {
	base, first := reverseDialFixture(t)
	path := ReverseDialPath(base)
	if err := os.Rename(path, path+".owned"); err != nil {
		t.Fatal(err)
	}
	other, err := net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := other.Close(); err != nil {
			t.Errorf("close different inode: %v", err)
		}
	})
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := os.Stat(path)
	if err != nil || !os.SameFile(before, after) {
		t.Fatalf("Close removed a different socket inode: info=%v, err=%v", after, err)
	}
	reverseDialAssertListenerReachable(t, path, other)
}

func TestReverseDialPrivateEndpointPublicationPreservesConcurrentOwner(t *testing.T) {
	path := ReverseDialPath(reverseDialConfigFixture(t))
	var other *net.UnixListener
	var before os.FileInfo
	listener, err := listenReversePrivate(path, func(stage string) {
		if stage == "recorded" {
			var err error
			other, err = net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
			if err != nil {
				t.Fatal(err)
			}
			before, err = os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
		}
	})
	if err == nil {
		closeErr := listener.Close()
		t.Fatalf("publication replaced a concurrent owner; close: %v", closeErr)
	}
	if other == nil {
		t.Fatalf("publication checkpoint not reached: %v", err)
	}
	after, statErr := os.Stat(path)
	if statErr != nil || !os.SameFile(before, after) {
		t.Fatalf("publication changed concurrent socket inode: info=%v, err=%v", after, statErr)
	}
	t.Cleanup(func() {
		if other != nil {
			if err := other.Close(); err != nil {
				t.Errorf("close concurrent owner: %v", err)
			}
		}
	})
	reverseDialAssertListenerReachable(t, path, other)
	if err := other.Close(); err != nil {
		t.Fatal(err)
	}
	other = nil
	replacement, err := ListenReversePrivate(path)
	if err != nil {
		t.Fatalf("publication failure retained lifetime lock: %v", err)
	}
	if err := replacement.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestReverseDialPrivateEndpointStagingBindFailureReleasesOwnership(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root bypasses directory write permissions")
	}
	base, first := reverseDialFixture(t)
	path := ReverseDialPath(base)
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	parent := filepath.Dir(path)
	if err := os.Chmod(parent, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chmod(parent, 0o700); err != nil {
			t.Errorf("restore bind directory permissions: %v", err)
		}
	})
	listener, err := ListenReversePrivate(path)
	if err == nil {
		closeErr := listener.Close()
		t.Fatalf("staging bind succeeded in unwritable directory; close: %v", closeErr)
	}
	if !errors.Is(err, os.ErrPermission) {
		t.Fatalf("fixture did not fail at staging bind: %v", err)
	}
	if err := os.Chmod(parent, 0o700); err != nil {
		t.Fatal(err)
	}
	replacement, err := ListenReversePrivate(path)
	if err != nil {
		t.Fatalf("failed staging bind retained ownership: %v", err)
	}
	if err := replacement.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestReverseDialPrivateEndpointSupportsMaximumCanonicalPath(t *testing.T) {
	base := reverseDialConfigFixture(t)
	limit := len(syscall.RawSockaddrUnix{}.Path) - 1
	parent := filepath.Join(base, strings.Repeat("d", limit-len(base)-3))
	path := filepath.Join(parent, "s")
	if len(path) != limit {
		t.Fatalf("boundary fixture has %d bytes, want %d", len(path), limit)
	}
	listener, err := ListenReversePrivate(path)
	if err != nil {
		t.Fatalf("valid maximum-length canonical path refused: %v", err)
	}
	t.Cleanup(func() {
		if err := listener.Close(); err != nil {
			t.Errorf("close boundary listener: %v", err)
		}
	})
	if listener.Addr().String() != path {
		t.Fatalf("listener address %q, want canonical path %q", listener.Addr(), path)
	}
	reverseDialAssertListenerReachable(t, path, listener)
}
