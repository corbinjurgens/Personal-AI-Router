// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Cross-process tests for client-originated errors relayed through the broker.
package tests

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/testbin"
)

// This adapter only changes a real worker's listen port; its stdio protocol
// passes through unchanged, following the workload interop adapter precedent.
// Broker fixtures configure their service ports with the broker's own port
// flags instead (see fixture_ports_test.go); the adapter remains for fixtures
// that start a worker outside that plan.
func init() {
	if bridge := os.Getenv("NVPAIR_TEST_PORT_BRIDGE"); bridge != "" && bridge == os.Args[0] {
		if err := runPortInteropBridge(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
}

func runPortInteropBridge() error {
	args := append(os.Args[1:], "--port", os.Getenv("NVPAIR_TEST_PORT"))
	return runOwnedPortWorker(os.Getenv("NVPAIR_TEST_PORT_BINARY"), args)
}

func newPortInteropBridge(t *testing.T, binary string, port int) (string, []string) {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	bridge := filepath.Join(testbin.Dir(t, "tests", "port-bridge", "port-bridge"), testbin.Exe("port-bridge"))
	if err := copyPortBridge(func() error { return os.WriteFile(bridge, data, 0o700) }, time.Sleep); err != nil {
		t.Fatalf("copy port bridge to %q (a previous process may still be running): %v", bridge, err)
	}
	return bridge, []string{
		"NVPAIR_TEST_PORT_BRIDGE=" + bridge,
		"NVPAIR_TEST_PORT_BINARY=" + binary,
		"NVPAIR_TEST_PORT=" + strconv.Itoa(port),
	}
}

const (
	portBridgeCopyAttempts = 41
	portBridgeCopyDelay    = 50 * time.Millisecond
)

// The adapter is a broker child, so this fixture has no exec.Cmd to wait on.
// Windows can retain its executable lock briefly after process cleanup. Allow
// that lock two seconds to clear, while returning every other copy error at once.
func copyPortBridge(write func() error, sleep func(time.Duration)) error {
	for attempt := 1; ; attempt++ {
		err := write()
		if err == nil || !isPortBridgeSharingViolation(err) || attempt == portBridgeCopyAttempts {
			return err
		}
		sleep(portBridgeCopyDelay)
	}
}

func startClientErrorsBroker(t *testing.T) *interopProcess {
	t.Helper()
	base := t.TempDir()
	ports := reserveFixtureServicePorts(t)
	args := append([]string{"--cluster-dir", filepath.Join(base, "cluster"), "--scanner-path", scannerBin, "--errors-path", errorsBin}, ports.args()...)
	p := startInteropProcess(t, base, brokerBin, ports.errors, nil, args...)
	for _, path := range []string{scannerBin, errorsBin} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		p.paths = append(p.paths, canonical)
	}
	waitForMethod(t, p.msgs, "app:ready", 10*time.Second)
	return p
}

// TestBrokerAcceptsClientErrorsReport is the leg-B repro: a client-originated
// errors:report (notification form) must be forwarded into nvpair-errors and
// reflected in the next errors:update — the same as a supervised worker
// emitting one on its stdout.
func TestBrokerAcceptsClientErrorsReport(t *testing.T) {
	p := startClientErrorsBroker(t)
	stdin, msgs := p.in, p.msgs

	const id = "client:leg-b-notif"
	report := fmt.Sprintf(`{"jsonrpc":"2.0","method":"errors:report","params":{"id":%q,"message":"client-synthesized","severity":"error"}}`, id) + "\n"
	if _, err := stdin.Write([]byte(report)); err != nil {
		t.Fatalf("write errors:report notification: %v", err)
	}

	// The broker forwards it into nvpair-errors, which stores it and pushes an
	// errors:update the broker relays back unconditionally.
	waitForErrorsUpdateContaining(t, msgs, id, 10*time.Second)
	t.Logf("client errors:report %q reflected in errors:update", id)
}

// TestBrokerAcceptsClientErrorsReportRequest covers the request form of
// leg B: an id-bearing errors:report is acked with a null result and also
// reflected in errors:update.
func TestBrokerAcceptsClientErrorsReportRequest(t *testing.T) {
	p := startClientErrorsBroker(t)
	stdin, msgs := p.in, p.msgs

	const (
		reqID = 950
		id    = "client:leg-b-req"
	)
	report := fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":"errors:report","params":{"id":%q,"message":"client-synthesized","severity":"error"}}`, reqID, id) + "\n"
	if _, err := stdin.Write([]byte(report)); err != nil {
		t.Fatalf("write errors:report request: %v", err)
	}

	// Expect both the ack (result for reqID, no error) and the errors:update.
	gotAck := false
	gotUpdate := false
	deadline := time.After(10 * time.Second)
	for !(gotAck && gotUpdate) {
		select {
		case msg, ok := <-msgs:
			if !ok {
				t.Fatal("broker stream closed before the report was acked and reflected")
			}
			switch {
			case msg.Method == "" && msg.ID != nil:
				if string(*msg.ID) == fmt.Sprintf("%d", reqID) {
					if msg.Error != nil {
						t.Fatalf("errors:report request returned an error: %+v", msg.Error)
					}
					gotAck = true
				}
			case msg.Method == methodErrorsUpdate:
				if errorsListHasID(t, msg.Params, id) {
					gotUpdate = true
				}
			}
		case <-deadline:
			t.Fatalf("timed out (ack=%v update=%v) for errors:report request %q", gotAck, gotUpdate, id)
		}
	}
}
