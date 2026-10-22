// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/secretenv/secretenvtest"
)

// fakeBrokerEnv makes the test binary act as a broker process, so the real
// exec-based spawner can be exercised end to end.
const fakeBrokerEnv = "NVPAIR_SERVICE_TEST_BROKER"

const (
	serviceCommandEnv    = "NVPAIR_SERVICE_TEST_COMMAND"
	serviceProcessDirEnv = "NVPAIR_SERVICE_TEST_PROCESS_DIR"
	processBrokerArg     = "--nvpair-service-test-broker"
)

type serviceProcessRecord struct {
	PID     int      `json:"pid"`
	Cwd     string   `json:"cwd"`
	Args    []string `json:"args"`
	Image   string   `json:"image,omitempty"`
	Created uint64   `json:"created,omitempty"`
}

func TestMain(m *testing.M) {
	secretenvtest.RunChild()
	command := os.Getenv(serviceCommandEnv) == "1"
	broker := len(os.Args) > 1 && os.Args[1] == processBrokerArg
	if os.Getenv(fakeBrokerEnv) == "1" && (!command || broker) {
		if err := recordServiceTestProcess("broker"); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		runProcessBroker()
		return
	}
	if command {
		// The reexecuted daemon keeps the command role. The broker selects its
		// existing fake role with an explicit first argument on the same image.
		args := os.Args[1:]
		if len(args) == 0 || (args[0] != "autostart" && args[0] != "status" && args[0] != "call" && args[0] != "stop") {
			opts, err := parseRunArgs(args, io.Discard)
			if err == nil && !opts.showVersion {
				if err := recordServiceTestProcess("daemon"); err != nil {
					fmt.Fprintln(os.Stderr, err)
					os.Exit(2)
				}
			}
		}
		os.Exit(run(args, os.Stdout, os.Stderr))
	}
	os.Exit(m.Run())
}

// Records exist before listening or app:ready, so failed startup can still be
// cleaned up by verified process identity. Only process tests supply this private directory.
func recordServiceTestProcess(role string) error {
	dir := os.Getenv(serviceProcessDirEnv)
	if dir == "" {
		return nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	image, created, err := serviceTestProcessIdentity()
	if err != nil {
		return err
	}
	data, err := json.Marshal(serviceProcessRecord{PID: os.Getpid(), Cwd: cwd, Args: os.Args[1:], Image: image, Created: created})
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, fmt.Sprintf("%s-%d.json", role, os.Getpid())), data, 0o600)
}

// runProcessBroker announces its working directory and arguments in app:ready,
// writes a stderr line at start and on "log", and exits on shutdown or stdin
// EOF.
func runProcessBroker() {
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	params, err := json.Marshal(map[string]any{"version": "proc", "cwd": cwd, "args": os.Args[1:], "pid": os.Getpid()})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	fmt.Printf(`{"jsonrpc":"2.0","method":"app:ready","params":%s}`+"\n", params)
	fmt.Fprintln(os.Stderr, "[nvpair-ui-broker] INFO started")
	sc := bufio.NewScanner(os.Stdin)
	for sc.Scan() {
		var m struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if json.Unmarshal(sc.Bytes(), &m) != nil {
			continue
		}
		switch m.Method {
		case "echo":
			fmt.Printf(`{"jsonrpc":"2.0","id":%s,"result":%s}`+"\n", m.ID, orNull(m.Params))
		case "log":
			fmt.Fprintln(os.Stderr, "[nvpair-ui-broker] INFO asked to log")
			fmt.Printf(`{"jsonrpc":"2.0","id":%s,"result":null}`+"\n", m.ID)
		case "shutdown":
			fmt.Printf(`{"jsonrpc":"2.0","id":%s,"result":null}`+"\n", m.ID)
			os.Exit(0)
		}
	}
}

func TestExecSpawnerRunsBrokerFromItsDirectoryWithArgs(t *testing.T) {
	t.Setenv(fakeBrokerEnv, "1")
	exe, err := filepath.Abs(os.Args[0])
	if err != nil {
		t.Fatal(err)
	}
	h := &harness{t: t, dir: shortTempDir(t), brokers: make(chan *fakeBroker, 1), runErr: make(chan error, 1)}
	h.endpoint = testEndpoint(h.dir)
	h.svc = newService(config{
		endpoint:               h.endpoint,
		version:                "test",
		logPath:                filepath.Join(h.dir, "broker.log"),
		spawn:                  execSpawner(exe, []string{"--log-level", "debug"}),
		backoff:                func(int) time.Duration { return 10 * time.Millisecond },
		stopGrace:              5 * time.Second,
		shutdownRequestTimeout: 2 * time.Second,
	})
	go func() { h.runErr <- h.svc.Run(t.Context()) }()
	waitUntil(t, "listening", func() bool { return alreadyAnswering(t.Context(), h.endpoint) })

	c := h.dial()
	ready := c.notification("app:ready")
	var p struct {
		Cwd  string   `json:"cwd"`
		Args []string `json:"args"`
	}
	if err := json.Unmarshal(ready.Params, &p); err != nil {
		t.Fatal(err)
	}
	wantDir, _ := filepath.EvalSymlinks(filepath.Dir(exe))
	gotDir, _ := filepath.EvalSymlinks(p.Cwd)
	if gotDir != wantDir {
		t.Fatalf("broker cwd = %q, want %q", gotDir, wantDir)
	}
	if strings.Join(p.Args, " ") != "--log-level debug" {
		t.Fatalf("broker args = %q", p.Args)
	}
	// Lines written before this client attached are in broker.log only, so
	// ask for a fresh one.
	c.call("0", "log", "")
	c.next("service/log for the requested line", func(m *jsonrpc.Message) bool {
		return m.Method == "service/log" && strings.Contains(string(m.Params), "asked to log")
	})
	waitUntil(t, "startup line in broker.log", func() bool {
		data, err := os.ReadFile(filepath.Join(h.dir, "broker.log"))
		return err == nil && strings.Contains(string(data), "INFO started\n")
	})

	r := c.call("1", "service/stop", "")
	if r.Error != nil {
		t.Fatalf("stop: %v", r.Error)
	}
	select {
	case err := <-h.runErr:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(waitTimeout):
		t.Fatal("service did not exit")
	}
	if st := h.svc.status(); st.BrokerRestarts != 0 {
		t.Fatalf("broker restarted during a clean stop: %+v", st)
	}
}

func TestResolveBrokerPathOverride(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "fake-broker")
	if err := os.WriteFile(bin, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := resolveBrokerPath(bin)
	if err != nil || got != bin {
		t.Fatalf("override = %q, %v", got, err)
	}
	if _, err := resolveBrokerPath(filepath.Join(dir, "missing")); err == nil {
		t.Fatal("missing override accepted")
	}
}

// The service-to-broker launch is a trusted carrier: the broker hands the
// configured credential references to the proxy, so it keeps the whole
// provider-secret namespace. The sentinel child exits nonzero after reporting,
// so the failure path's captured output is checked too.
func TestExecSpawnerKeepsProviderSecretsForTheBroker(t *testing.T) {
	report := secretenvtest.Prepare(t)
	t.Setenv(secretenvtest.ExitEnv, "3")
	exe, err := filepath.Abs(os.Args[0])
	if err != nil {
		t.Fatal(err)
	}
	proc, err := execSpawner(exe, []string{"--log-level", "debug"})()
	if err != nil {
		t.Fatalf("spawn: %v", err)
	}
	defer proc.stdin.Close()
	defer proc.closeOutput()
	stdout, stderr := make(chan []byte, 1), make(chan []byte, 1)
	go func() { data, _ := io.ReadAll(proc.stdout); stdout <- data }()
	go func() { data, _ := io.ReadAll(proc.stderr); stderr <- data }()
	if code := proc.wait(); code != 3 {
		t.Fatalf("sentinel broker exit code = %d, want 3", code)
	}
	secretenvtest.AssertNoValue(t, "broker stdout", string(<-stdout))
	secretenvtest.AssertNoValue(t, "broker stderr", string(<-stderr))
	secretenvtest.AssertRetained(t, secretenvtest.Wait(t, report))
}

func TestExecSpawnerFailureDoesNotExposeProviderSecrets(t *testing.T) {
	secretenvtest.Prepare(t)
	proc, err := execSpawner(filepath.Join(t.TempDir(), "missing-broker"), []string{"--log-level", "debug"})()
	if err == nil {
		_ = proc.kill()
		t.Fatal("spawned a missing broker")
	}
	secretenvtest.AssertNoValue(t, "spawn error", err.Error())
}
