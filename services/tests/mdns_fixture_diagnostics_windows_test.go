// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"testing"
)

var fixtureDiagnosticBytes atomic.Int64

func mdnsFixtureAdvertiserOutput(t *testing.T, output string) {
	t.Helper()
	if output != "" {
		t.Logf("mDNS advertiser diagnostics:\n%s", output)
	}
}

// Preserve selected operational evidence before a test's stderr consumer
// discards it. Bound the whole executable's diagnostic tee to 16 MiB.
func mdnsFixtureLogLine(line string) {
	if !strings.EqualFold(os.Getenv("NVPAIR_LOG_LEVEL"), "debug") || !isMDNSFixtureDiagnostic(line) {
		return
	}
	if fixtureDiagnosticBytes.Add(int64(len(line)+1)) <= 16*1024*1024 {
		fmt.Fprintln(os.Stderr, line)
	}
}

func isMDNSFixtureDiagnostic(line string) bool {
	_, message, scanner := strings.Cut(line, " [nvpair-node-scanner] ")
	if !scanner {
		var manager bool
		_, message, manager = strings.Cut(line, " [nvpair-workload-manager] ")
		if !manager {
			return false
		}
	}
	_, message, ok := strings.Cut(message, " ") // level, then the known message
	if !ok {
		return false
	}
	// These messages carry interface/address, identity, count, state, or
	// transport-error metadata. Raw JSON-RPC and malformed payload logs are
	// deliberately absent from the allowlist.
	for _, prefix := range []string{
		"mdns send:", "mdns:", "mDNS ", "failed to create mDNS resolver",
		"_nvpair-node responder exited", "node trust changed", "node-info is ",
		"advertised addresses re-ranked", "peer-observed local addresses updated",
		"peer set changed", "discovery refresh failed", "failed to subscribe to discovery relay",
		"inter-node server listening", "inter-node interface switched personality",
		"pushing active workload snapshot to peers", "re-syncing workloads to peers",
		"broadcasting local lifecycle", "broadcast skipped:", "broadcast delivered",
		"broadcast attempt failed", "broadcast dropped after max attempts",
		"using stdio transport", "shutdown complete",
	} {
		if strings.HasPrefix(message, prefix) {
			return true
		}
	}
	return false
}

func TestMDNSFixtureDiagnosticsSelectOnlyOperationalWorkers(t *testing.T) {
	test := func(name, line string, want bool) {
		t.Run(name, func(t *testing.T) {
			if got := isMDNSFixtureDiagnostic(line); got != want {
				t.Errorf("diagnostic selection = %v, want %v", got, want)
			}
		})
	}
	test("scanner query source", "12:00:00.000 [nvpair-node-scanner] DEBUG mdns send: query sent iface=Wi-Fi ip=192.168.11.45", true)
	test("peer count", "12:00:00.000 [nvpair-workload-manager] INFO peer set changed total=1 source=relay", true)
	test("snapshot count", "12:00:00.000 [nvpair-workload-manager] DEBUG pushing active workload snapshot to peers count=1 peers=1", true)
	test("other worker", "12:00:00.000 [nvpair-proxy] DEBUG mdns send: query sent", false)
	test("malformed payload", "12:00:00.000 [nvpair-workload-manager] WARN dropping malformed local lifecycle err=payload", false)
	test("raw frame", `{"jsonrpc":"2.0","params":{"message":"private"}}`, false)
}
