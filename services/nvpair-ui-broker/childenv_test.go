// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/secretenv"
	"nvpair-shared/secretenv/secretenvtest"
)

// unconfiguredRef is a canonical reference no test policy configures.
const unconfiguredRef = secretenv.Prefix + "UNCONFIGURED"

// fakeProxyEnvironmentMethod asks a fake proxy which reserved names it
// received. It answers with a secretenvtest.Report, names only.
const fakeProxyEnvironmentMethod = "test/environment"

// fakeProxyEnvironment is the fake proxy's answer to fakeProxyEnvironmentMethod.
func fakeProxyEnvironment() secretenvtest.Report {
	report := secretenvtest.Report{Reserved: []string{}}
	for _, kv := range os.Environ() {
		name, _, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(strings.ToUpper(name), secretenv.Prefix) {
			report.Reserved = append(report.Reserved, name)
		}
	}
	slices.Sort(report.Reserved)
	report.ArgsCarryValue = slices.ContainsFunc(os.Args, func(arg string) bool {
		return strings.Contains(arg, secretenvtest.Value)
	})
	return report
}

// prepareProviderSecrets puts the sentinel secrets in this test's environment,
// with a reserved name no policy configures and, where names are case-sensitive,
// a lowercase variant of the configured reference. It returns the report path.
func prepareProviderSecrets(t *testing.T) string {
	t.Helper()
	report := secretenvtest.Prepare(t)
	t.Setenv(unconfiguredRef, secretenvtest.Value)
	if runtime.GOOS != "windows" {
		t.Setenv(strings.ToLower(secretenvtest.Name), secretenvtest.Value)
	}
	return report
}

func testExecutable(t *testing.T) string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return exe
}

// captureLogs collects the broker's log output for the rest of the test.
func captureLogs(t *testing.T) *syncBuffer {
	t.Helper()
	logs := &syncBuffer{}
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return logs
}

// Every broker child but the proxy starts without the reserved namespace.
func TestWorkerLaunchesRemoveProviderSecrets(t *testing.T) {
	test := func(name string, start func(exe string) (supervisedHandle, error)) {
		t.Run(name, func(t *testing.T) {
			report := prepareProviderSecrets(t)

			h, err := start(testExecutable(t))
			if err != nil {
				secretenvtest.AssertNoValue(t, "start error", err.Error())
				t.Fatalf("start: %v", err)
			}
			t.Cleanup(h.Stop)

			secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
		})
	}

	test("scanner", func(exe string) (supervisedHandle, error) {
		return startScanner(exe, "info", newDiscoveryStore(), nil, nil, nil, nil, nil)
	})
	test("node-info", func(exe string) (supervisedHandle, error) {
		return startNodeInfo(exe, "info", nil)
	})
	test("workload-manager", func(exe string) (supervisedHandle, error) {
		return startWorkloadManager(exe, "info", nil, nil)
	})
	test("errors", func(exe string) (supervisedHandle, error) {
		return startErrors(exe, "info", nil, nil)
	})
	test("rpc worker", func(exe string) (supervisedHandle, error) {
		return startRPCWorker("engine-manager", exe, []string{"--log-level", "info"}, nil)
	})
	test("cluster-manager", func(exe string) (supervisedHandle, error) {
		return startClusterManager(exe, "info", t.TempDir(), nil, nil, nil)
	})
}

// The proxy keeps exactly the configured references from the namespace.
func TestStartProxyRetainsOnlyConfiguredReferences(t *testing.T) {
	test := func(name string, refs []nodepolicy.CredentialRef, want []string) {
		t.Run(name, func(t *testing.T) {
			report := prepareProviderSecrets(t)

			pp, err := startProxy("proxy", testExecutable(t), "info", refs, nil, nil)
			if err != nil {
				t.Fatalf("startProxy: %v", err)
			}
			t.Cleanup(pp.Stop)

			got := secretenvtest.Wait(t, report)
			if !slices.Equal(got.Reserved, want) {
				t.Fatalf("proxy received %q, want %q", got.Reserved, want)
			}
			if !slices.Equal(pp.credentialRefs, refs) {
				t.Fatalf("proxy snapshot = %q, want %q", pp.credentialRefs, refs)
			}
		})
	}

	test("configured reference", []nodepolicy.CredentialRef{secretenvtest.Name}, []string{secretenvtest.Name})
	test("no configured reference", nil, []string{})
}

// A reference whose value the proxy could not tell apart fails the launch, and
// the error names the reference without a value.
func TestProxyEnvRejectsAmbiguousDuplicate(t *testing.T) {
	env := []string{
		"PATH=/bin",
		secretenvtest.Name + "=" + secretenvtest.Value,
		secretenvtest.Name + "=" + secretenvtest.Value + "-second",
	}

	got, err := proxyEnv(env, []nodepolicy.CredentialRef{secretenvtest.Name})

	if err == nil {
		t.Fatalf("proxyEnv = %q, want an ambiguity error", got)
	}
	if !strings.Contains(err.Error(), secretenvtest.Name) {
		t.Errorf("error %q does not name the reference", err)
	}
	secretenvtest.AssertNoValue(t, "proxyEnv error", err.Error())
}

// An environment the proxy cannot be given stops the launch before a process
// starts.
func TestStartProxyFailsClosedOnEnvironmentError(t *testing.T) {
	prepareProviderSecrets(t)

	pp, err := startProxy("proxy", testExecutable(t), "info", []nodepolicy.CredentialRef{"nvpair_provider_secret_sentinel"}, nil, nil)

	if err == nil {
		pp.Stop()
		t.Fatal("startProxy started a proxy with a reference it could not retain")
	}
	if !strings.Contains(err.Error(), "proxy environment") {
		t.Errorf("error %q does not say the proxy environment failed", err)
	}
	secretenvtest.AssertNoValue(t, "startProxy error", err.Error())
}

// fakeProxyReserved asks a running fake proxy which reserved names it received.
func fakeProxyReserved(t *testing.T, pp *proxyProcess) []string {
	t.Helper()
	result, rpcErr, err := pp.Call(context.Background(), fakeProxyEnvironmentMethod, json.RawMessage(`{}`))
	if err != nil || rpcErr != nil {
		t.Fatalf("%s: %v %+v", fakeProxyEnvironmentMethod, err, rpcErr)
	}
	var report secretenvtest.Report
	if err := json.Unmarshal(result, &report); err != nil {
		t.Fatal(err)
	}
	if report.ArgsCarryValue {
		t.Error("the sentinel secret reached the proxy's arguments")
	}
	return report.Reserved
}

// setAccounts is a policy:set that configures one free account per reference
// and changes nothing else but ceiling, which a test uses to make an unrelated
// change.
func (sb *supervisedBroker) setAccounts(t *testing.T, ceiling string, refs ...nodepolicy.CredentialRef) {
	t.Helper()
	doc := sb.b.localPolicy().Policy
	doc.Routing.Accounts = nil
	for i, ref := range refs {
		doc.Routing.Accounts = append(doc.Routing.Accounts, nodepolicy.Account{
			ID:                "acct-" + string(rune('a'+i)),
			Credential:        ref,
			Billing:           nodepolicy.BillingFree,
			PerRequestCeiling: nodepolicy.NanoUSD(ceiling),
			DailyCeiling:      "0",
			MonthlyCeiling:    "0",
		})
	}
	got := sb.call(t, methodPolicySet, map[string]any{"policy": doc, "expectedPersistedRevision": doc.Revision}, 2*policyActivationTimeout)
	if got.Error != nil {
		t.Fatalf("policy:set accounts %q: %+v", refs, got.Error)
	}
}

// A policy change that alters the reference set replaces the running proxy
// exactly once with one launched with the new set; a change that keeps the set
// leaves it alone. Neither the replies, the client stream nor the logs carry a
// secret value.
func TestProxyRestartsOnceForAChangedReferenceSet(t *testing.T) {
	test := func(name string, listener bool, facades ...string) {
		t.Run(name, func(t *testing.T) {
			prepareProviderSecrets(t)
			logs := captureLogs(t)
			sb := newSupervisedBroker(t, listener, facades...)
			if len(facades) > 0 {
				t.Setenv(fakeProxyEnv, fakeProxyAcceptsFacades)
			}
			b := sb.b
			b.startProxySupervision(len(facades) > 0)
			pp := b.proxyProcessHandle()
			requireProxyRunning(t, pp, "the proxy starts")
			if got := fakeProxyReserved(t, pp); len(got) != 0 {
				t.Fatalf("a proxy with no configured reference received %q", got)
			}
			spawns := b.currentOllamaProxyGeneration()

			step := func(what, ceiling string, restart bool, refs []nodepolicy.CredentialRef, want []string) {
				sb.setAccounts(t, ceiling, refs...)
				waitReconciled(t, b)
				if restart {
					requireProxyExited(t, pp, what)
					spawns++
				}
				if got := b.currentOllamaProxyGeneration(); got != spawns {
					t.Fatalf("%s: spawn generation = %d, want %d", what, got, spawns)
				}
				next := b.proxyProcessHandle()
				requireProxyRunning(t, next, what)
				if (next != pp) != restart {
					t.Fatalf("%s: replaced the proxy = %t, want %t", what, next != pp, restart)
				}
				if got := fakeProxyReserved(t, next); !slices.Equal(got, want) {
					t.Fatalf("%s: proxy received %q, want %q", what, got, want)
				}
				pp = next
			}

			step("adding a reference", "0", true, []nodepolicy.CredentialRef{secretenvtest.Name}, []string{secretenvtest.Name})
			step("an unrelated change", "5", false, []nodepolicy.CredentialRef{secretenvtest.Name}, []string{secretenvtest.Name})
			step("replacing the reference", "5", true, []nodepolicy.CredentialRef{unconfiguredRef}, []string{unconfiguredRef})
			step("removing every reference", "5", true, nil, []string{})

			if len(facades) > 0 && b.engineProxyHandle(ollamaProxyProfile) != pp {
				t.Error("the restarted proxy does not serve the facade")
			}
			secretenvtest.AssertNoValue(t, "client stream", bufferText(sb.out))
			secretenvtest.AssertNoValue(t, "broker log", bufferText(logs))
		})
	}

	test("listener only", true)
	test("serving a facade", false, ollamaProxyProfile.Name)
}

func bufferText(s *syncBuffer) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.String()
}
