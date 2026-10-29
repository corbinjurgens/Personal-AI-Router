// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

func TestCopyBaselineAmendWindowsPIDAndPipeErrors(t *testing.T) {
	entry := windows.ProcessEntry32{ProcessID: 42}
	copy(entry.ExeFile[:], windows.StringToUTF16("nvpair-engine-manager.exe"))
	r := windowsProcessRecord(entry)
	if r.PID != 42 || r.Image != "nvpair-engine-manager.exe" {
		t.Fatalf("record %+v", r)
	}
	for _, failure := range []error{windows.ERROR_PIPE_BUSY, windows.ERROR_ACCESS_DENIED, windows.ERROR_SEM_TIMEOUT, context.Canceled, context.DeadlineExceeded} {
		e := checkCopyBaselineQuiescence(context.Background(), "", 42, func(context.Context) ([]processRecord, error) { return []processRecord{r}, nil }, func() (string, error) { return "private", nil }, func(context.Context, string) (io.ReadWriteCloser, error) { return nil, failure })
		if e == nil {
			t.Fatalf("pipe error accepted %v", failure)
		}
	}
}

func TestCopyBaselineAmendWindowsLocalBaseCLI(t *testing.T) {
	for _, shape := range []string{"missing", "empty", "relative", "current ineligible", "previous marker"} {
		t.Run(shape, func(t *testing.T) {
			cfg, home := freshTestBase(t), freshTestBase(t)
			path := filepath.Join(cfg, "Nvidia Corporation", "Personal AI Router", "engine-bin", modelDownloadFile)
			if e := os.MkdirAll(filepath.Dir(path), 0700); e != nil {
				t.Fatal(e)
			}
			data := amendEmpty
			if shape == "current ineligible" {
				data = `{}`
			}
			if e := os.WriteFile(path, []byte(data), 0600); e != nil {
				t.Fatal(e)
			}
			if shape == "previous marker" {
				pp := filepath.Join(cfg, "NVIDIA Corporation", "PAIR", "engine-bin", modelDownloadFile)
				if e := os.MkdirAll(filepath.Dir(pp), 0700); e != nil {
					t.Fatal(e)
				}
				if e := os.WriteFile(pp, []byte(`{}`), 0600); e != nil {
					t.Fatal(e)
				}
			}
			before := freshTree(t, cfg)
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, managerBin, "--amend-copy-baseline", "--expected-journal-sha256="+journalDigest([]byte(data)), "--copy-baseline-disposition=complete-empty-history", "--attest-profile-quiescent", "--previous-profile-reviewed")
			cmd.Env = isolatedManagerEnv(cfg, home)
			if shape == "missing" || shape == "empty" || shape == "relative" {
				var env []string
				for _, item := range cmd.Env {
					if !strings.HasPrefix(strings.ToUpper(item), "LOCALAPPDATA=") {
						env = append(env, item)
					}
				}
				if shape == "empty" {
					env = append(env, "LOCALAPPDATA=")
				}
				if shape == "relative" {
					env = append(env, "LOCALAPPDATA=relative")
				}
				cmd.Env = env
			}
			if out, e := cmd.CombinedOutput(); e == nil {
				t.Fatalf("invalid mode succeeded %s", out)
			}
			if !reflect.DeepEqual(before, freshTree(t, cfg)) {
				t.Fatal("CLI guard changed Local base")
			}
		})
	}
}

func TestCopyBaselineAmendWindowsLocalBase(t *testing.T) {
	for _, shape := range []string{"missing", "empty", "relative", "current ineligible", "previous markers"} {
		t.Run(shape, func(t *testing.T) {
			c, p, path, o := amendFixture(t, amendEmpty)
			t.Setenv("LOCALAPPDATA", filepath.Dir(c))
			t.Setenv("APPDATA", t.TempDir())
			switch shape {
			case "missing":
				if e := os.Unsetenv("LOCALAPPDATA"); e != nil {
					t.Fatal(e)
				}
			case "empty":
				t.Setenv("LOCALAPPDATA", "")
			case "relative":
				t.Setenv("LOCALAPPDATA", "relative")
			case "current ineligible":
				if e := os.WriteFile(path, []byte(`{}`), 0600); e != nil {
					t.Fatal(e)
				}
			case "previous markers":
				o.PreviousReviewed = true
				if e := os.MkdirAll(filepath.Join(p, "engine-bin"), 0700); e != nil {
					t.Fatal(e)
				}
				if e := os.WriteFile(filepath.Join(p, "engine-bin", modelDownloadFile), []byte(`{"entries":[{}]}`), 0600); e != nil {
					t.Fatal(e)
				}
			}
			before := amendRead(t, path)
			if _, e := amendRun(c, p, o); e == nil {
				t.Fatal("invalid Local/profile accepted")
			}
			amendPreserved(t, path, before)
			amendNoBackup(t, path)
		})
	}
}
