// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
)

func init() {
	if worker := os.Getenv("NVPAIR_TEST_STALLED_PORT_WORKER"); worker != "" && worker == os.Args[0] {
		if err := runStalledPortWorker(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}

// Only this copied fixture ignores EOF and graceful termination. It forces the
// real broker through its worker-stop escalation without stalling a real app.
func runStalledPortWorker() error {
	signal.Ignore(os.Interrupt, syscall.SIGTERM)
	ln, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", os.Getenv("NVPAIR_TEST_PORT")))
	if err != nil {
		return err
	}
	go func() {
		if err := http.Serve(ln, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		})); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}()
	codec := jsonrpc.NewCodec(struct {
		io.Reader
		io.Writer
	}{os.Stdin, os.Stdout})
	if err := codec.Notify("ready", map[string]string{"version": "stalled-port-fixture"}); err != nil {
		return err
	}
	if _, err := io.Copy(io.Discard, os.Stdin); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "fixture_stalled_worker_ignoring_eof pid=%d\n", os.Getpid())
	select {}
}

func TestPortInteropBridgeBrokerTeardownKillsStalledWorker(t *testing.T) {
	base := t.TempDir()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	worker := filepath.Join(base, "stalled-worker"+filepath.Ext(errorsBin))
	if err := os.WriteFile(worker, data, 0o700); err != nil {
		t.Fatal(err)
	}
	port := freePort(t)
	bridge, env := newPortInteropBridge(t, base, worker, port)
	env = append(env, "NVPAIR_TEST_STALLED_PORT_WORKER="+worker)
	p := startInteropProcess(t, base, brokerBin, port, env,
		"--cluster-dir", filepath.Join(base, "cluster"), "--scanner-path", scannerBin, "--errors-path", bridge)
	for _, path := range []string{scannerBin, bridge, worker} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		p.paths = append(p.paths, canonical)
	}
	p.inventory = &serviceProcessInventory{paths: p.paths, known: make(map[int32]serviceProcessIdentity)}
	// Preserve cleanup even when a regression leaves an orphan. The assertion
	// in p.cleanup runs first; this guard only removes the proven owned survivor.
	t.Cleanup(func() {
		p.cleanup()
		owned, err := p.inventory.scan()
		if err != nil {
			t.Errorf("inspect fault fixture during emergency cleanup: %v", err)
		}
		for _, child := range owned {
			proc, err := os.FindProcess(int(child.PID))
			if err != nil {
				t.Errorf("find owned fault fixture pid=%d: %v", child.PID, err)
				continue
			}
			if err := proc.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
				t.Errorf("kill owned fault fixture pid=%d: %v", child.PID, err)
			}
			if err := proc.Release(); err != nil {
				t.Errorf("release owned fault fixture pid=%d: %v", child.PID, err)
			}
		}
	})
	waitForMethod(t, p.msgs, "app:ready", 10*time.Second)
	waitTCP(t, net.JoinHostPort("127.0.0.1", fmt.Sprint(port)), 5*time.Second)
	owned, err := p.inventory.scan()
	if err != nil {
		t.Fatal(err)
	}
	canonicalWorker, err := canonicalServicePath(worker)
	if err != nil {
		t.Fatal(err)
	}
	var workerPID int32
	for _, child := range owned {
		if child.Path == canonicalWorker {
			workerPID = child.PID
		}
	}
	if workerPID == 0 {
		t.Fatalf("listening fault worker missing from owned process inventory: %+v", owned)
	}
	t.Logf("stalled_fixture_owned pid=%d port=%d processes=%+v", workerPID, port, owned)
	// This closes broker stdin and waits for its real supervisor to escalate.
	// The shared cleanup independently requires no owned processes or listener.
	p.cleanup()
	if !strings.Contains(p.output.String(), fmt.Sprintf("fixture_stalled_worker_ignoring_eof pid=%d", workerPID)) {
		t.Fatalf("fixture did not reach its stalled EOF path; stderr: %s", p.output.String())
	}
}
