// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/secretenv/secretenvtest"
)

// Every process engine-manager starts loses the reserved provider-secret
// namespace, including names a caller passes as environment overrides. Each
// test starts this test binary as a sentinel child through one launch site;
// the child reports only which reserved names it saw.

// sentinelOverrides tries to reintroduce the namespace through the overrides a
// launch site merges over the inherited environment.
func sentinelOverrides() map[string]string {
	return map[string]string{
		secretenvtest.Name:                    secretenvtest.Value,
		"nvpair_provider_secret_override_low": secretenvtest.Value,
		"PAIR_TEST_UNRELATED":                 "kept",
	}
}

func TestChildEnvRemovesProviderSecretsAfterMergingOverrides(t *testing.T) {
	secretenvtest.Prepare(t)
	env := childEnv(sentinelOverrides())
	unrelated := false
	for _, kv := range env {
		name, _, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(strings.ToUpper(name), "NVPAIR_PROVIDER_SECRET_") {
			t.Errorf("child environment keeps %s", name)
		}
		if kv == "PAIR_TEST_UNRELATED=kept" {
			unrelated = true
		}
	}
	if !unrelated {
		t.Error("an unrelated override was dropped")
	}
}

func TestStartManagedProcRemovesProviderSecrets(t *testing.T) {
	for _, tc := range []struct {
		name      string
		overrides map[string]string
		exit      string
	}{
		{"inherited environment", nil, "0"},
		{"caller overrides", sentinelOverrides(), "0"},
		{"child exits with failure", sentinelOverrides(), "2"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report := secretenvtest.Prepare(t)
			t.Setenv(secretenvtest.ExitEnv, tc.exit)
			var mu sync.Mutex
			var lines []string
			proc, err := startManagedProc(os.Args[0], []string{"serve"}, tc.overrides, func(stream, line string) {
				mu.Lock()
				defer mu.Unlock()
				lines = append(lines, line)
			})
			if err != nil {
				t.Fatalf("startManagedProc: %v", err)
			}
			select {
			case <-proc.done:
			case <-time.After(10 * time.Second):
				proc.stop(0)
				t.Fatal("sentinel engine did not exit")
			}
			secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
			mu.Lock()
			defer mu.Unlock()
			secretenvtest.AssertNoValue(t, "engine output", strings.Join(lines, "\n"))
		})
	}
}

func TestStartManagedProcFailureDoesNotExposeProviderSecrets(t *testing.T) {
	secretenvtest.Prepare(t)
	_, err := startManagedProc(filepath.Join(t.TempDir(), "missing-engine"), nil, sentinelOverrides(), nil)
	if err == nil {
		t.Fatal("started a missing engine")
	}
	secretenvtest.AssertNoValue(t, "start error", err.Error())
}

func TestRunPrivateLaunchCommandRemovesProviderSecrets(t *testing.T) {
	for _, tc := range []struct {
		name      string
		overrides map[string]string
		exit      string
	}{
		{"inherited environment", nil, "0"},
		{"caller overrides", sentinelOverrides(), "0"},
		{"command fails", sentinelOverrides(), "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report := secretenvtest.Prepare(t)
			t.Setenv(secretenvtest.ExitEnv, tc.exit)
			err := runPrivateLaunchCommand(context.Background(), []string{os.Args[0], "server", "start"}, tc.overrides, nil)
			if (err != nil) != (tc.exit != "0") {
				t.Fatalf("runPrivateLaunchCommand error = %v, want failure %v", err, tc.exit != "0")
			}
			if err != nil {
				secretenvtest.AssertNoValue(t, "launch error", err.Error())
			}
			secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
		})
	}
}

func TestRunManifestCommandRemovesProviderSecrets(t *testing.T) {
	for _, tc := range []struct {
		name      string
		overrides map[string]string
		exit      string
	}{
		{"inherited environment", nil, "0"},
		{"manifest environment", sentinelOverrides(), "0"},
		{"command fails", sentinelOverrides(), "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report := secretenvtest.Prepare(t)
			t.Setenv(secretenvtest.ExitEnv, tc.exit)
			err := runManifestCommand(context.Background(), []string{os.Args[0], "install"}, tc.overrides)
			if (err != nil) != (tc.exit != "0") {
				t.Fatalf("runManifestCommand error = %v, want failure %v", err, tc.exit != "0")
			}
			if err != nil {
				secretenvtest.AssertNoValue(t, "install error", err.Error())
			}
			secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
		})
	}
}

func TestRunCommandOutputRemovesProviderSecrets(t *testing.T) {
	for _, exit := range []string{"0", "1"} {
		t.Run("exit "+exit, func(t *testing.T) {
			report := secretenvtest.Prepare(t)
			t.Setenv(secretenvtest.ExitEnv, exit)
			out, err := new(Executor).runCommandOutput(context.Background(), []string{os.Args[0], "pull", "model"})
			if (err != nil) != (exit != "0") {
				t.Fatalf("runCommandOutput error = %v, want failure %v", err, exit != "0")
			}
			secretenvtest.AssertNoValue(t, "action output", out)
			if err != nil {
				secretenvtest.AssertNoValue(t, "action error", err.Error())
			}
			secretenvtest.AssertRemoved(t, secretenvtest.Wait(t, report))
		})
	}
}

// Launch settings may not assign a provider secret: engines never receive one,
// and the error names the variable but never its value.
func TestSettingsRejectProviderSecretEnvironment(t *testing.T) {
	for _, name := range []string{"NVPAIR_PROVIDER_SECRET_OPENAI", "nvpair_provider_secret_openai", "Nvpair_Provider_Secret_X"} {
		t.Run(name, func(t *testing.T) {
			e := settingsExecutor(t, false)
			p := settingsRequest(t, e)
			p.Settings.LaunchText = name + `="` + secretenvtest.Value + `" ` + p.Settings.LaunchText
			result, err := e.PreviewLaunch(p)
			if err != nil {
				t.Fatalf("PreviewLaunch: %v", err)
			}
			message := result.Errors["launchText"]
			if !strings.Contains(message, name) || !strings.Contains(message, "reserved for provider credentials") {
				t.Fatalf("preview error = %q, want it to name %s as reserved", message, name)
			}
			secretenvtest.AssertNoValue(t, "preview error", message)
			for _, kv := range result.Env {
				secretenvtest.AssertNoValue(t, "preview environment", kv)
			}
			_, err = e.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: p.Settings}, func() error { return nil })
			if err == nil {
				t.Fatal("ConfigureLaunch accepted a provider secret")
			}
			secretenvtest.AssertNoValue(t, "configure error", err.Error())
			assertNoSettingsOverride(t, e)
		})
	}
}

// A saved override is parsed by literalEnvironment at load and at launch, so a
// provider secret written into one by hand is refused there too.
func TestLiteralEnvironmentRejectsProviderSecrets(t *testing.T) {
	for _, name := range []string{"NVPAIR_PROVIDER_SECRET_OPENAI", "nvpair_provider_secret_openai"} {
		_, err := literalEnvironment([]string{"PAIR_TEST=1", name + "=" + secretenvtest.Value})
		if err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("literalEnvironment error = %v, want it to name %s", err, name)
		}
		secretenvtest.AssertNoValue(t, "literal environment error", err.Error())
	}
	rt := Runtime{EditableLaunch: &EditableLaunch{FixedArgs: []string{"serve"}}}
	env := []string{secretenvtest.Name + "=" + secretenvtest.Value}
	rt.LaunchEnv = &env
	if _, err := resolveProcessLaunch(rt, "engine", map[string]string{"host": "127.0.0.1", "port": "12345"}); err == nil {
		t.Fatal("a saved provider secret reached the launch command")
	}
}
