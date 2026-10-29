// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
	"nvpair-shared/servicectl"
)

const amendEmpty = ` { "copies": [], "version":1, "copyRecoveryRequired":false, "copiesComplete":false, "entries":[] } `
const amendOld = ` { "copiesComplete":true, "entries":[], "version":1, "copies":[] } `

func amendFixture(t *testing.T, data string) (string, string, string, copyBaselineOptions) {
	t.Helper()
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	current, previous := filepath.Join(base, "app"), filepath.Join(base, "previous")
	path := filepath.Join(current, "engine-bin", modelDownloadFile)
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	return current, previous, path, copyBaselineOptions{Mode: true, Expected: journalDigest([]byte(data)), Disposition: "complete-empty-history", Attest: true}
}

func amendRead(t *testing.T, path string) []byte {
	t.Helper()
	b, e := os.ReadFile(path)
	if e != nil {
		t.Fatal(e)
	}
	return b
}
func amendRun(current, previous string, o copyBaselineOptions) (copyBaselineResult, error) {
	return amendCopyBaseline(current, previous, o, func() error { return nil }, durablefile.Replacer{}, durablefile.Replacer{})
}
func amendPreserved(t *testing.T, path string, before []byte) {
	t.Helper()
	if !bytes.Equal(amendRead(t, path), before) {
		t.Fatal("journal changed on refusal")
	}
}
func amendNoBackup(t *testing.T, path string) {
	t.Helper()
	matches, e := filepath.Glob(path + ".before-copy-baseline-*")
	if e != nil || len(matches) != 0 {
		t.Fatalf("unexpected backup: %v %v", matches, e)
	}
}

func TestCopyBaselineAmendCurrentAndOld(t *testing.T) {
	for _, input := range []struct {
		name, data string
		old        bool
	}{{"current", amendEmpty, false}, {"old", amendOld, true}} {
		t.Run(input.name, func(t *testing.T) {
			current, previous, path, o := amendFixture(t, input.data)
			o.AllowOld = input.old
			if input.old {
				if _, _, _, _, e := newModelDownloadStore(filepath.Dir(path)).loadJournal(); e == nil {
					t.Fatal("ordinary loader accepted old")
				}
			}
			r, e := amendRun(current, previous, o)
			if e != nil {
				t.Fatal(e)
			}
			if r.Publication != "confirmed" || r.Before != journalDigest([]byte(input.data)) || r.After != journalDigest(amendRead(t, path)) || r.Load != "strict-current" || !r.Complete || r.Recovery || r.Copies != 0 || r.Downloads != 0 {
				t.Fatalf("summary %+v", r)
			}
			if !bytes.Equal(amendRead(t, filepath.Join(filepath.Dir(path), r.Backup)), []byte(input.data)) {
				t.Fatal("backup lost raw bytes")
			}
			source := newOpID()
			d := newModelDownloads(source, filepath.Dir(path), map[string]string{"llamacpp": engines.ModelNamingExact})
			reg := &Registry{engines: map[string]*Manifest{"llamacpp": {Engine: "llamacpp", ModelNaming: engines.ModelNamingExact}}}
			m := &Manager{exec: &Executor{switchIncarnation: source}, copyJournal: d, copyRegistry: reg}
			evidence := m.ownerReservations()
			if !evidence.Complete || evidence.Token.DownloadSource != source || evidence.Token.Incarnation != source {
				t.Fatalf("reload evidence %+v", evidence)
			}
			requireOwnerBool(t, evidence.Busy, false)
		})
	}
}

func TestCopyBaselineAmendStrictDecoderEquivalence(t *testing.T) {
	cases := map[string]string{"current": amendEmpty, "old": amendOld, "trailing": amendEmpty + ` {}`, "utf8": amendEmpty + string([]byte{255}), "bounds": strings.Repeat(" ", modeldownload.MaxJournalBytes+1)}
	var object map[string]json.RawMessage
	if e := json.Unmarshal([]byte(amendEmpty), &object); e != nil {
		t.Fatal(e)
	}
	for key := range object {
		for _, kind := range []string{"missing", "null", "duplicate", "case", "type", "unknown"} {
			var b []byte
			clone := make(map[string]json.RawMessage)
			for k, v := range object {
				clone[k] = v
			}
			switch kind {
			case "missing":
				delete(clone, key)
			case "null":
				clone[key] = json.RawMessage("null")
			case "type":
				clone[key] = json.RawMessage(`"wrong"`)
			case "case":
				clone[strings.ToUpper(key)] = clone[key]
				delete(clone, key)
			case "unknown":
				clone["extra"] = json.RawMessage("false")
			}
			var e error
			b, e = json.Marshal(clone)
			if e != nil {
				t.Fatal(e)
			}
			if kind == "duplicate" {
				b = append([]byte(fmt.Sprintf(`{"%s":%s,`, key, object[key])), b[1:]...)
			}
			cases[key+"/"+kind] = string(b)
		}
	}
	cases["version"] = strings.Replace(amendEmpty, `"version":1`, `"version":2`, 1)
	for name, input := range cases {
		t.Run(name, func(t *testing.T) {
			_, _, path, _ := amendFixture(t, input)
			a, b, c, d, e := decodeModelDownloadJournal([]byte(input))
			aa, bb, cc, dd, ee := newModelDownloadStore(filepath.Dir(path)).loadJournal()
			if (e == nil) != (ee == nil) || !reflect.DeepEqual(a, aa) || !reflect.DeepEqual(b, bb) || c != cc || d != dd {
				t.Fatal("decoder/load differ")
			}
			if name != "current" && e == nil {
				t.Fatal("malformed accepted")
			}
			if name != "current" {
				current := filepath.Dir(filepath.Dir(path))
				o := copyBaselineOptions{Mode: true, Expected: journalDigest([]byte(input)), Disposition: "complete-empty-history", Attest: true}
				_, err := amendRun(current, filepath.Join(filepath.Dir(current), "previous"), o)
				if err == nil {
					t.Fatal("amend accepted")
				}
				amendPreserved(t, path, []byte(input))
				amendNoBackup(t, path)
			}
		})
	}
}

func TestCopyBaselineAmendOptionsRefuse(t *testing.T) {
	cases := map[string]func(*copyBaselineOptions){"mode": func(o *copyBaselineOptions) { o.Mode = false }, "attestation": func(o *copyBaselineOptions) { o.Attest = false }, "hash missing": func(o *copyBaselineOptions) { o.Expected = "" }, "hash uppercase": func(o *copyBaselineOptions) { o.Expected = strings.Repeat("A", 64) }, "hash wrong": func(o *copyBaselineOptions) { o.Expected = strings.Repeat("0", 64) }, "disposition missing": func(o *copyBaselineOptions) { o.Disposition = "" }, "disposition unsupported": func(o *copyBaselineOptions) { o.Disposition = "clear" }, "old flag with current": func(o *copyBaselineOptions) { o.AllowOld = true }}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			c, p, path, o := amendFixture(t, amendEmpty)
			change(&o)
			if _, e := amendRun(c, p, o); e == nil {
				t.Fatal("accepted")
			}
			amendPreserved(t, path, []byte(amendEmpty))
			amendNoBackup(t, path)
		})
	}
}

func TestCopyBaselineAmendRetainedRefusals(t *testing.T) {
	download := modelDownloadMarker{ID: strings.Repeat("a", 32), Engine: "llamacpp", Model: "model", ContentKey: "model", Naming: engines.ModelNamingExact}
	copy := modelCopyMarker{OpID: strings.Repeat("b", 32), Engine: "llamacpp", Model: "model", ContentKey: "model", Naming: engines.ModelNamingExact, Role: "controller", OwnerNodeID: "source", SourceNodeID: "source", DestinationNodeID: "destination", ManagerIncarnation: strings.Repeat("c", 32), Phase: "unsafe"}
	for _, name := range []string{"download", "naming mismatch", "unsafe copy", "uncertain predecessor", "recovery empty", "already complete", "old without flag"} {
		t.Run(name, func(t *testing.T) {
			j := modelDownloadJournal{Version: 1, Entries: []modelDownloadMarker{}, Copies: []modelCopyMarker{}}
			switch name {
			case "download":
				j.Entries = append(j.Entries, download)
			case "naming mismatch":
				download.ContentKey = "mismatch"
				j.Entries = append(j.Entries, download)
			case "unsafe copy":
				j.Copies = append(j.Copies, copy)
			case "uncertain predecessor":
				copy.Phase = "uncertain"
				j.Copies = append(j.Copies, copy)
			case "recovery empty":
				j.CopyRecoveryRequired = true
			case "already complete":
				j.CopiesComplete = true
			}
			data, e := json.Marshal(j)
			if e != nil {
				t.Fatal(e)
			}
			if name == "old without flag" {
				data = []byte(amendOld)
			}
			c, p, path, o := amendFixture(t, string(data))
			if _, e := amendRun(c, p, o); e == nil {
				t.Fatal("retained accepted")
			}
			amendPreserved(t, path, data)
			amendNoBackup(t, path)
			if name == "unsafe copy" || name == "uncertain predecessor" {
				d := newModelDownloads(newOpID(), filepath.Dir(path), map[string]string{"llamacpp": engines.ModelNamingExact})
				if len(d.copies) != 1 {
					t.Fatal("retained reload lost marker")
				}
				reg := &Registry{engines: map[string]*Manifest{"llamacpp": {Engine: "llamacpp", ModelNaming: engines.ModelNamingExact}}}
				m := &Manager{exec: &Executor{switchIncarnation: d.source}, copyJournal: d, copyRegistry: reg}
				if m.ownerReservations().Complete {
					t.Fatal("retained reload promoted owner evidence")
				}
			}
		})
	}
}

func TestCopyBaselineAmendPreviousPolicy(t *testing.T) {
	for _, name := range []string{"absent", "no journal reviewed", "no journal unreviewed", "empty current", "empty old", "retained", "invalid", "old retained", "partial migration"} {
		t.Run(name, func(t *testing.T) {
			c, p, path, o := amendFixture(t, amendEmpty)
			o.PreviousReviewed = true
			if name != "absent" {
				if e := os.MkdirAll(filepath.Join(p, "engine-bin"), 0700); e != nil {
					t.Fatal(e)
				}
			}
			var before []byte
			pp := filepath.Join(p, "engine-bin", modelDownloadFile)
			switch name {
			case "empty current":
				before = []byte(strings.Replace(amendEmpty, `"copyRecoveryRequired":false`, `"copyRecoveryRequired":true`, 1))
			case "empty old":
				before = []byte(amendOld)
			case "retained", "partial migration":
				before = []byte(strings.Replace(amendEmpty, `"entries":[]`, `"entries":[{}]`, 1))
			case "old retained":
				before = []byte(strings.Replace(amendOld, `"entries":[]`, `"entries":[{}]`, 1))
			case "invalid":
				before = []byte(`{`)
			case "no journal unreviewed":
				o.PreviousReviewed = false
			}
			if before != nil {
				if e := os.WriteFile(pp, before, 0600); e != nil {
					t.Fatal(e)
				}
			}
			_, e := amendRun(c, p, o)
			want := name == "absent" || name == "no journal reviewed" || name == "empty current" || name == "empty old"
			if (e == nil) != want {
				t.Fatalf("err %v want success %t", e, want)
			}
			if !want {
				amendPreserved(t, path, []byte(amendEmpty))
				amendNoBackup(t, path)
			}
			if before != nil {
				amendPreserved(t, pp, before)
			}
		})
	}
}

func TestCopyBaselineAmendQuiescence(t *testing.T) {
	for _, name := range []string{"absent", "override", "missing self", "enumeration", "cancelled", "timeout", "permission", "busy pipe", "endpoint", "connected", "close error", "partial", "duplicate"} {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			if name == "cancelled" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			override := ""
			if name == "override" {
				override = "elsewhere"
			}
			enum := func(context.Context) ([]processRecord, error) {
				if name == "enumeration" {
					return nil, io.ErrUnexpectedEOF
				}
				if name == "missing self" {
					return []processRecord{{2, "unrelated"}}, nil
				}
				if name == "partial" {
					return []processRecord{{1, ""}}, nil
				}
				if name == "duplicate" {
					return []processRecord{{1, "self"}, {1, "self"}}, nil
				}
				return []processRecord{{1, "nvpair-engine-manager"}}, nil
			}
			endpoint := func() (string, error) {
				if name == "endpoint" {
					return "", os.ErrPermission
				}
				return "private", nil
			}
			dial := func(context.Context, string) (io.ReadWriteCloser, error) {
				switch name {
				case "timeout":
					return nil, context.DeadlineExceeded
				case "permission":
					return nil, os.ErrPermission
				case "busy pipe":
					return nil, errors.New("pipe busy")
				case "connected":
					return &amendConn{}, nil
				case "close error":
					return &amendConn{closeErr: os.ErrPermission}, nil
				}
				return nil, servicectl.ErrNotRunning
			}
			e := checkCopyBaselineQuiescence(ctx, override, 1, enum, endpoint, dial)
			if (e == nil) != (name == "absent") {
				t.Fatalf("quiescence %v", e)
			}
		})
	}
}

type amendConn struct{ closeErr error }

func (*amendConn) Read([]byte) (int, error)    { return 0, io.EOF }
func (*amendConn) Write(p []byte) (int, error) { return len(p), nil }
func (c *amendConn) Close() error              { return c.closeErr }

func TestCopyBaselineAmendOwners(t *testing.T) {
	for _, image := range []string{"nvpair-service", "nvpair-ui-broker", "nvpair-engine-manager", "nvpair-future-worker.exe", "PAIR", "NVIDIA PAIR", "nvpair", "PAIR Helper (Renderer)", "NVIDIA PAIR Helper", "nvpair Helper", "/another profile/nvpair-engine-manager"} {
		t.Run(image, func(t *testing.T) {
			c, p, path, o := amendFixture(t, amendEmpty)
			probe := func() error {
				return checkCopyBaselineQuiescence(context.Background(), "", 1, func(context.Context) ([]processRecord, error) {
					return []processRecord{{1, "nvpair-engine-manager"}, {2, image}}, nil
				}, func() (string, error) { t.Fatal("dial after owner"); return "", nil }, nil)
			}
			if _, e := amendCopyBaseline(c, p, o, probe, durablefile.Replacer{}, durablefile.Replacer{}); e == nil {
				t.Fatal("owner accepted")
			}
			amendNoBackup(t, path)
			amendPreserved(t, path, []byte(amendEmpty))
		})
	}
}

func TestCopyBaselineAmendChangedBetweenChecks(t *testing.T) {
	for _, boundary := range []int{2, 3} {
		t.Run(fmt.Sprint(boundary), func(t *testing.T) {
			c, p, path, o := amendFixture(t, amendEmpty)
			calls := 0
			changed := []byte(amendEmpty + "\n")
			probe := func() error {
				calls++
				if calls == boundary {
					if e := os.WriteFile(path, changed, 0600); e != nil {
						t.Fatal(e)
					}
				}
				return nil
			}
			if _, e := amendCopyBaseline(c, p, o, probe, durablefile.Replacer{}, durablefile.Replacer{}); e == nil {
				t.Fatal("change overwritten")
			}
			amendPreserved(t, path, changed)
			if boundary == 2 {
				amendNoBackup(t, path)
			}
		})
	}
}

func TestCopyBaselineAmendPreviousChanges(t *testing.T) {
	for _, boundary := range []int{2, 3} {
		for _, kind := range []string{"new app", "markers", "inspection"} {
			t.Run(fmt.Sprintf("%d/%s", boundary, kind), func(t *testing.T) {
				c, p, path, o := amendFixture(t, amendEmpty)
				fs := defaultCopyBaselineFS()
				if kind == "markers" {
					o.PreviousReviewed = true
					if e := os.MkdirAll(filepath.Join(p, "engine-bin"), 0700); e != nil {
						t.Fatal(e)
					}
					if e := os.WriteFile(filepath.Join(p, "engine-bin", modelDownloadFile), []byte(amendEmpty), 0600); e != nil {
						t.Fatal(e)
					}
				}
				calls := 0
				fs.lstat = func(s string) (os.FileInfo, error) {
					if s == path {
						calls++
					}
					if calls >= boundary && kind == "inspection" && s == p {
						return nil, os.ErrPermission
					}
					if calls == boundary && s == path && kind != "inspection" {
						if e := os.MkdirAll(filepath.Join(p, "engine-bin"), 0700); e != nil {
							t.Fatal(e)
						}
						if kind == "markers" {
							if e := os.WriteFile(filepath.Join(p, "engine-bin", modelDownloadFile), []byte(`{"version":1,"entries":[{"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","engine":"llamacpp","model":"model","contentKey":"model","naming":"exact"}],"copies":[],"copiesComplete":false,"copyRecoveryRequired":false}`), 0600); e != nil {
								t.Fatal(e)
							}
						}
					}
					return os.Lstat(s)
				}
				if _, e := amendCopyBaselineWithFS(c, p, o, func() error { return nil }, durablefile.Replacer{}, durablefile.Replacer{}, fs); e == nil {
					t.Fatal("previous change accepted")
				}
				amendPreserved(t, path, []byte(amendEmpty))
			})
		}
	}
}

func TestCopyBaselineAmendContention(t *testing.T) {
	c, p, path, o := amendFixture(t, amendEmpty)
	var wg sync.WaitGroup
	results := make(chan error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, e := amendRun(c, p, o); results <- e }()
	}
	wg.Wait()
	close(results)
	success := 0
	for e := range results {
		if e == nil {
			success++
		}
	}
	if success != 1 {
		t.Fatalf("successes %d", success)
	}
	backup := path + ".before-copy-baseline-" + o.Expected + ".bak"
	amendPreserved(t, backup, []byte(amendEmpty))
}

func TestCopyBaselineAmendCLI(t *testing.T) {
	for _, name := range []string{"success", "old", "old missing flag", "missing hash", "missing attestation", "missing disposition", "false mode", "false attestation", "false previous", "false old", "version", "uninstall", "runtime default", "fresh", "dangling", "positional before", "positional after", "unsupported disposition"} {
		t.Run(name, func(t *testing.T) {
			cfg, e := filepath.EvalSymlinks(t.TempDir())
			if e != nil {
				t.Fatal(e)
			}
			home, e := filepath.EvalSymlinks(t.TempDir())
			if e != nil {
				t.Fatal(e)
			}
			data := amendEmpty
			if name == "old" || name == "old missing flag" {
				data = amendOld
			}
			// Match the platform base selected by isolatedManagerEnv.
			base := cfg
			if runtime.GOOS == "darwin" {
				base = filepath.Join(home, "Library", "Application Support")
			}
			path := filepath.Join(base, "Nvidia Corporation", "Personal AI Router", "engine-bin", modelDownloadFile)
			if e := os.MkdirAll(filepath.Dir(path), 0700); e != nil {
				t.Fatal(e)
			}
			if e := os.WriteFile(path, []byte(data), 0600); e != nil {
				t.Fatal(e)
			}
			args := []string{"--amend-copy-baseline", "--expected-journal-sha256=" + journalDigest([]byte(data)), "--copy-baseline-disposition=complete-empty-history", "--attest-profile-quiescent"}
			switch name {
			case "old":
				args = append(args, "--allow-empty-old-journal")
			case "missing hash":
				args = append(args[:1], args[2:]...)
			case "missing attestation":
				args = args[:3]
			case "missing disposition":
				args = append(args[:2], args[3:]...)
			case "false mode":
				args[0] = "--amend-copy-baseline=false"
			case "false attestation":
				args[3] = "--attest-profile-quiescent=false"
			case "false previous":
				args = append(args, "--previous-profile-reviewed=false")
			case "false old":
				args = append(args, "--allow-empty-old-journal=false")
			case "version":
				args = append(args, "--version")
			case "uninstall":
				args = append(args, "--uninstall-managed")
			case "runtime default":
				args = append(args, "--http-port=0")
			case "fresh":
				args = append(args, "--initialize-fresh-profile")
			case "dangling":
				args = args[1:]
			case "positional before":
				args = append([]string{"unexpected"}, args...)
			case "positional after":
				args = append(args, "unexpected")
			case "unsupported disposition":
				args[2] = "--copy-baseline-disposition=clear"
			}
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, managerBin, args...)
			cmd.Env = isolatedManagerEnv(cfg, home)
			out, e := cmd.CombinedOutput()
			success := name == "success" || name == "old"
			if (e == nil) != success {
				t.Fatalf("exit %v: %s", e, out)
			}
			for _, bad := range []string{"jsonrpc", "ready", "listening", "using stdio", "loaded "} {
				if strings.Contains(string(out), bad) {
					t.Fatalf("runtime output %s", out)
				}
			}
			if success {
				if !strings.Contains(string(out), "publication=confirmed") || !strings.Contains(string(out), "after-sha256="+journalDigest(amendRead(t, path))) {
					t.Fatalf("summary %s", out)
				}
			} else {
				amendPreserved(t, path, []byte(data))
				amendNoBackup(t, path)
			}
			for _, root := range []string{cfg, home} {
				for file, content := range freshTree(t, root) {
					if content != "directory" && filepath.Base(file) != modelDownloadFile && !strings.Contains(filepath.Base(file), ".before-copy-baseline-") {
						t.Fatalf("runtime artifact %s", file)
					}
				}
			}
		})
	}
}

type amendFaultFile struct {
	*os.File
	step string
}

func (f *amendFaultFile) Write(b []byte) (int, error) {
	if f.step == "write" {
		return 0, io.ErrShortWrite
	}
	return f.File.Write(b)
}
func (f *amendFaultFile) Sync() error {
	if f.step == "sync" {
		return os.ErrPermission
	}
	return f.File.Sync()
}
func (f *amendFaultFile) Close() error {
	e := f.File.Close()
	if f.step == "close" {
		return os.ErrPermission
	}
	return e
}

type amendCloseFailure struct{ io.Closer }

func (f amendCloseFailure) Close() error {
	if e := f.Closer.Close(); e != nil {
		return e
	}
	return os.ErrPermission
}

func TestCopyBaselineAmendFaultTimeline(t *testing.T) {
	for _, stage := range []string{"reservation", "reservation close", "backup create", "backup write", "backup sync", "backup close", "backup replace", "backup durability", "backup readback", "backup mismatch", "journal create", "journal write", "journal sync", "journal close", "journal replace", "journal landed uncertain", "journal readback", "journal invalid readback", "journal mismatched readback"} {
		t.Run(stage, func(t *testing.T) {
			c, p, path, o := amendFixture(t, amendEmpty)
			fs := defaultCopyBaselineFS()
			backup, journal := durablefile.Replacer{}, durablefile.Replacer{}
			fail := errors.New("injected")
			if stage == "reservation" {
				fs.reserve = func(string) (io.Closer, error) { return nil, fail }
			}
			if stage == "reservation close" {
				reserve := fs.reserve
				fs.reserve = func(s string) (io.Closer, error) {
					f, e := reserve(s)
					if e != nil {
						return nil, e
					}
					return amendCloseFailure{f}, nil
				}
			}
			if strings.HasPrefix(stage, "backup ") {
				step := strings.TrimPrefix(stage, "backup ")
				switch step {
				case "create":
					backup.CreateTemp = func(string, string, os.FileMode) (durablefile.File, error) { return nil, fail }
				case "write", "sync", "close":
					backup.CreateTemp = func(dir, base string, mode os.FileMode) (durablefile.File, error) {
						f, e := os.CreateTemp(dir, base)
						if e != nil {
							return nil, e
						}
						return &amendFaultFile{f, step}, nil
					}
				case "replace":
					backup.Replace = func(string, string) error { return fail }
				case "durability":
					backup.Replace = func(tmp, target string) error {
						if e := os.Rename(tmp, target); e != nil {
							return e
						}
						return errors.Join(durablefile.ErrReplaced, fail)
					}
				case "readback", "mismatch":
					read := fs.read
					fs.read = func(s string) ([]byte, error) {
						if strings.HasSuffix(s, ".bak") {
							if step == "mismatch" {
								return []byte(amendOld), nil
							}
							return nil, fail
						}
						return read(s)
					}
				}
			}
			attempted := false
			journal.Replace = func(tmp, target string) error {
				attempted = true
				if stage == "journal replace" {
					return fail
				}
				if e := os.Rename(tmp, target); e != nil {
					return e
				}
				if stage == "journal landed uncertain" {
					return errors.Join(durablefile.ErrReplaced, fail)
				}
				return nil
			}
			if stage == "journal create" {
				journal.CreateTemp = func(string, string, os.FileMode) (durablefile.File, error) { attempted = true; return nil, fail }
			}
			for _, step := range []string{"write", "sync", "close"} {
				if stage == "journal "+step {
					journal.CreateTemp = func(dir, base string, mode os.FileMode) (durablefile.File, error) {
						attempted = true
						f, e := os.CreateTemp(dir, base)
						if e != nil {
							return nil, e
						}
						return &amendFaultFile{f, step}, nil
					}
				}
			}
			if stage == "journal readback" || stage == "journal invalid readback" || stage == "journal mismatched readback" {
				read := fs.read
				fs.read = func(s string) ([]byte, error) {
					if attempted && s == path {
						if stage == "journal mismatched readback" {
							return []byte(amendEmpty), nil
						}
						if stage == "journal invalid readback" {
							return []byte(`{}`), nil
						}
						return nil, fail
					}
					return read(s)
				}
			}
			r, e := amendCopyBaselineWithFS(c, p, o, func() error { return nil }, backup, journal, fs)
			if e == nil || r.Publication == "confirmed" {
				t.Fatalf("fault succeeded %+v %v", r, e)
			}
			journalStage := strings.HasPrefix(stage, "journal ")
			if !journalStage && attempted {
				t.Fatal("journal attempted before confirmed backup")
			}
			landed := stage == "journal landed uncertain" || stage == "journal readback" || stage == "journal invalid readback" || stage == "journal mismatched readback"
			if !landed {
				amendPreserved(t, path, []byte(amendEmpty))
			} else {
				_, _, complete, recovery, e := decodeModelDownloadJournal(amendRead(t, path))
				if e != nil || !complete || recovery {
					t.Fatal("landed bytes missing")
				}
			}
			if journalStage {
				backupPath := path + ".before-copy-baseline-" + o.Expected + ".bak"
				amendPreserved(t, backupPath, []byte(amendEmpty))
			}
			if stage == "journal landed uncertain" && (r.After != journalDigest(amendRead(t, path)) || !r.Complete || !strings.Contains(r.Publication, "uncertain-true")) {
				t.Fatalf("uncertainty summary %+v", r)
			}
		})
	}
}

func TestCopyBaselineAmendUnavailableBeforeReadback(t *testing.T) {
	current, previous, path, options := amendFixture(t, amendEmpty)
	t.Setenv("LOCALAPPDATA", filepath.Dir(current))
	fs := defaultCopyBaselineFS()
	read := fs.read
	reads := 0
	fs.read = func(name string) ([]byte, error) {
		if name == path {
			reads++
			if reads == 1 {
				if err := os.WriteFile(path, []byte(amendOld), 0600); err != nil {
					t.Fatal(err)
				}
				return nil, os.ErrPermission
			}
		}
		return read(name)
	}
	result, err := amendCopyBaselineWithFS(current, previous, options, func() error { return nil }, durablefile.Replacer{}, durablefile.Replacer{}, fs)
	if !errors.Is(err, os.ErrPermission) {
		t.Fatalf("initial read failure not retained: %v", err)
	}
	if result.Before != "unavailable" || result.After != journalDigest([]byte(amendOld)) || result.Load != "invalid" || result.Publication != "not-attempted" {
		t.Fatalf("readback fabricated an earlier observation: %+v", result)
	}
	amendNoBackup(t, path)
	amendPreserved(t, path, []byte(amendOld))
}

func TestCopyBaselineAmendPathRefusals(t *testing.T) {
	for _, kind := range []string{"missing journal", "journal directory", "journal symlink", "install symlink", "profile symlink", "ancestor symlink", "inspection", "previous symlink", "previous install symlink", "previous read", "backup exists", "backup dangling"} {
		t.Run(kind, func(t *testing.T) {
			c, p, path, o := amendFixture(t, amendEmpty)
			fs := defaultCopyBaselineFS()
			o.PreviousReviewed = true
			backup := path + ".before-copy-baseline-" + o.Expected + ".bak"
			switch kind {
			case "missing journal":
				if e := os.Remove(path); e != nil {
					t.Fatal(e)
				}
			case "journal directory":
				if e := os.Remove(path); e != nil {
					t.Fatal(e)
				}
				if e := os.Mkdir(path, 0700); e != nil {
					t.Fatal(e)
				}
			case "inspection":
				fs.lstat = func(s string) (os.FileInfo, error) {
					if s == c {
						return nil, os.ErrPermission
					}
					return os.Lstat(s)
				}
			case "previous read":
				if e := os.MkdirAll(filepath.Join(p, "engine-bin"), 0700); e != nil {
					t.Fatal(e)
				}
				pp := filepath.Join(p, "engine-bin", modelDownloadFile)
				if e := os.WriteFile(pp, []byte(amendEmpty), 0600); e != nil {
					t.Fatal(e)
				}
				read := fs.read
				fs.read = func(s string) ([]byte, error) {
					if s == pp {
						return nil, os.ErrPermission
					}
					return read(s)
				}
			case "backup exists":
				if e := os.WriteFile(backup, []byte("existing backup"), 0600); e != nil {
					t.Fatal(e)
				}
			default:
				link := path
				target := path + ".real"
				switch kind {
				case "install symlink":
					link = filepath.Dir(path)
					target = link + ".real"
				case "profile symlink":
					link = c
					target = c + ".real"
				case "ancestor symlink":
					link = filepath.Dir(c)
					target = link + ".real"
				case "previous symlink":
					link = p
					target = c
				case "previous install symlink":
					if e := os.Mkdir(p, 0700); e != nil {
						t.Fatal(e)
					}
					link = filepath.Join(p, "engine-bin")
					target = filepath.Dir(path)
				case "backup dangling":
					link = backup
					target = backup + ".missing"
				}
				if kind == "ancestor symlink" { // Use injected metadata to avoid moving the test's owned root.
					info, e := os.Lstat(path)
					if e != nil {
						t.Fatal(e)
					}
					fs.lstat = func(s string) (os.FileInfo, error) {
						if s == link {
							return amendSymlinkInfo{info}, nil
						}
						return os.Lstat(s)
					}
				} else {
					if kind != "previous symlink" && kind != "previous install symlink" && kind != "backup dangling" {
						if e := os.Rename(link, target); e != nil {
							t.Fatal(e)
						}
					}
					if e := os.Symlink(target, link); e != nil {
						t.Fatal(e)
					}
				}
			}
			before := freshTree(t, filepath.Dir(c))
			if _, e := amendCopyBaselineWithFS(c, p, o, func() error { return nil }, durablefile.Replacer{}, durablefile.Replacer{}, fs); e == nil {
				t.Fatal("unsafe path accepted")
			}
			if !reflect.DeepEqual(before, freshTree(t, filepath.Dir(c))) {
				t.Fatal("refusal mutated namespace")
			}
		})
	}
}

type amendSymlinkInfo struct{ os.FileInfo }

func (amendSymlinkInfo) Mode() os.FileMode { return os.ModeSymlink | 0777 }

func TestCopyBaselineAmendOldQualificationCountercases(t *testing.T) {
	var object map[string]json.RawMessage
	if e := json.Unmarshal([]byte(amendOld), &object); e != nil {
		t.Fatal(e)
	}
	cases := map[string][]byte{"retained entries": []byte(strings.Replace(amendOld, `"entries":[]`, `"entries":[{}]`, 1)), "retained copies": []byte(strings.Replace(amendOld, `"copies":[]`, `"copies":[{}]`, 1)), "utf8": append([]byte(amendOld), 255), "bounds": []byte(strings.Repeat(" ", modeldownload.MaxJournalBytes+1)), "version": []byte(strings.Replace(amendOld, `"version":1`, `"version":2`, 1)), "trailing": []byte(amendOld + ` {}`)}
	for key := range object {
		for _, kind := range []string{"missing", "null", "case", "duplicate", "type", "unknown"} {
			clone := make(map[string]json.RawMessage)
			for k, v := range object {
				clone[k] = v
			}
			switch kind {
			case "missing":
				delete(clone, key)
			case "null":
				clone[key] = json.RawMessage("null")
			case "case":
				delete(clone, key)
				clone[strings.ToUpper(key)] = object[key]
			case "type":
				clone[key] = json.RawMessage(`"wrong"`)
			case "unknown":
				clone["extra"] = json.RawMessage("true")
			}
			b, e := json.Marshal(clone)
			if e != nil {
				t.Fatal(e)
			}
			if kind == "duplicate" {
				b = append([]byte(fmt.Sprintf(`{"%s":%s,`, key, object[key])), b[1:]...)
			}
			cases[key+"/"+kind] = b
		}
	}
	for name, b := range cases {
		t.Run(name, func(t *testing.T) {
			c, p, path, o := amendFixture(t, string(b))
			o.AllowOld = true
			if _, e := amendRun(c, p, o); e == nil {
				t.Fatal("unsupported old accepted")
			}
			amendPreserved(t, path, b)
			amendNoBackup(t, path)
		})
	}
}

func TestCopyBaselineAmendPreviousRetainedMarkers(t *testing.T) {
	for _, kind := range []string{"download", "copy", "old download", "old copy"} {
		t.Run(kind, func(t *testing.T) {
			c, p, path, o := amendFixture(t, amendEmpty)
			o.PreviousReviewed = true
			o.AllowOld = true
			install := filepath.Join(p, "engine-bin")
			if e := os.MkdirAll(install, 0700); e != nil {
				t.Fatal(e)
			}
			store := newModelDownloadStore(install)
			d := map[downloadContentKey]modelDownloadMarker{}
			copies := map[string]modelCopyMarker{}
			if strings.Contains(kind, "download") {
				d[downloadContentKey{"llamacpp", "model"}] = modelDownloadMarker{ID: strings.Repeat("a", 32), Engine: "llamacpp", Model: "model", ContentKey: "model", Naming: engines.ModelNamingExact}
			} else {
				m := modelCopyMarker{OpID: strings.Repeat("b", 32), Engine: "llamacpp", Model: "model", ContentKey: "model", Naming: engines.ModelNamingExact, Role: "controller", OwnerNodeID: "source", SourceNodeID: "source", DestinationNodeID: "destination", ManagerIncarnation: strings.Repeat("c", 32), Phase: "uncertain"}
				copies[m.OpID] = m
			}
			if _, _, e := store.save(d, copies, false, false); e != nil {
				t.Fatal(e)
			}
			before := amendRead(t, store.path)
			if strings.HasPrefix(kind, "old ") {
				var obj map[string]json.RawMessage
				if e := json.Unmarshal(before, &obj); e != nil {
					t.Fatal(e)
				}
				delete(obj, "copyRecoveryRequired")
				var e error
				before, e = json.Marshal(obj)
				if e != nil {
					t.Fatal(e)
				}
				if e := os.WriteFile(store.path, before, 0600); e != nil {
					t.Fatal(e)
				}
			}
			// Old opt-in does not waive previous-marker refusal. Use a qualified
			// current old input so current eligibility cannot mask that policy.
			if e := os.WriteFile(path, []byte(amendOld), 0600); e != nil {
				t.Fatal(e)
			}
			o.Expected = journalDigest([]byte(amendOld))
			r, e := amendRun(c, p, o)
			if e == nil || r.Previous != "retained-or-invalid" {
				t.Fatalf("previous accepted %+v %v", r, e)
			}
			amendPreserved(t, path, []byte(amendOld))
			amendPreserved(t, store.path, before)
			amendNoBackup(t, path)
		})
	}
}

func TestCopyBaselineAmendRepeatedOwnerRefusal(t *testing.T) {
	for _, boundary := range []int{1, 2, 3} {
		t.Run(fmt.Sprint(boundary), func(t *testing.T) {
			c, p, path, o := amendFixture(t, amendEmpty)
			calls := 0
			probe := func() error {
				calls++
				if calls == boundary {
					return os.ErrPermission
				}
				return nil
			}
			if _, e := amendCopyBaseline(c, p, o, probe, durablefile.Replacer{}, durablefile.Replacer{}); e == nil {
				t.Fatal("owner uncertainty accepted")
			}
			amendPreserved(t, path, []byte(amendEmpty))
			if boundary < 3 {
				amendNoBackup(t, path)
			}
		})
	}
}

func TestCopyBaselineAmendSharedOrgAlias(t *testing.T) {
	for _, reviewed := range []bool{false, true} {
		t.Run(fmt.Sprint(reviewed), func(t *testing.T) {
			base := freshTestBase(t)
			org := filepath.Join(base, "Nvidia Corporation")
			alias := filepath.Join(base, "NVIDIA Corporation")
			current := filepath.Join(org, "Personal AI Router")
			previous := filepath.Join(alias, "PAIR")
			path := filepath.Join(current, "engine-bin", modelDownloadFile)
			if e := os.MkdirAll(filepath.Dir(path), 0700); e != nil {
				t.Fatal(e)
			}
			if e := os.Mkdir(filepath.Join(org, "PAIR"), 0700); e != nil {
				t.Fatal(e)
			}
			if e := os.WriteFile(path, []byte(amendEmpty), 0600); e != nil {
				t.Fatal(e)
			}
			fs := defaultCopyBaselineFS()
			a, ea := os.Stat(org)
			b, eb := os.Stat(alias)
			if ea == nil && eb == nil && os.SameFile(a, b) {
				t.Log("native case-insensitive org lookup exercised")
			} else {
				t.Log("native org aliasing UNVERIFIED; injected shared-parent lookup exercised")
				remap := func(s string) string {
					if s == alias || strings.HasPrefix(s, alias+string(filepath.Separator)) {
						return org + strings.TrimPrefix(s, alias)
					}
					return s
				}
				fs.lstat = func(s string) (os.FileInfo, error) { return os.Lstat(remap(s)) }
				fs.read = func(s string) ([]byte, error) { return readAmendJournal(remap(s)) }
			}
			o := copyBaselineOptions{Mode: true, Expected: journalDigest([]byte(amendEmpty)), Disposition: "complete-empty-history", Attest: true, PreviousReviewed: reviewed}
			r, e := amendCopyBaselineWithFS(current, previous, o, func() error { return nil }, durablefile.Replacer{}, durablefile.Replacer{}, fs)
			if (e == nil) != reviewed {
				t.Fatalf("shared parent policy %+v %v", r, e)
			}
			if !reviewed {
				amendPreserved(t, path, []byte(amendEmpty))
				amendNoBackup(t, path)
			} else if r.Previous != "absent-journal" {
				t.Fatalf("classification %+v", r)
			}
		})
	}
}
