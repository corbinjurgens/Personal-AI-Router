// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
)

// fakeBrokerEnv makes the test binary act as a broker process, so the real
// exec-based spawner can be exercised end to end.
const fakeBrokerEnv = "NVPAIR_SERVICE_TEST_BROKER"

func TestMain(m *testing.M) {
	if os.Getenv(fakeBrokerEnv) == "1" {
		runProcessBroker()
		return
	}
	os.Exit(m.Run())
}

// runProcessBroker announces its working directory and arguments in app:ready,
// writes a stderr line at start and on "log", and exits on shutdown or stdin
// EOF.
func runProcessBroker() {
	cwd, _ := os.Getwd()
	params, _ := json.Marshal(map[string]any{"version": "proc", "cwd": cwd, "args": os.Args[1:]})
	fmt.Printf(`{"jsonrpc":"2.0","method":"app:ready","params":%s}`+"\n", params)
	fmt.Fprintln(os.Stderr, "[nvpair-ui-broker] INFO started")
	sc := bufio.NewScanner(os.Stdin)
	for sc.Scan() {
		var m struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		if json.Unmarshal(sc.Bytes(), &m) != nil {
			continue
		}
		switch m.Method {
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
