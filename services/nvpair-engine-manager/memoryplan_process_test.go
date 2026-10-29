// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The private test executable accepts exactly the production Python argv.
func init() {
	if len(os.Args) != 8 || os.Args[1] != "-B" || os.Args[2] != "-s" || os.Args[3] != "-m" || os.Args[4] != "serve.server" || os.Args[5] != "--estimate-memory" || os.Args[6] != "--memory-plan-request" {
		return
	}
	raw, err := os.ReadFile(os.Args[7])
	if err != nil {
		os.Exit(2)
	}
	var request struct {
		Scenario string `json:"scenario"`
	}
	if json.Unmarshal(raw, &request) != nil {
		os.Exit(2)
	}
	switch request.Scenario {
	case "environment":
		cwd, err := os.Getwd()
		if err != nil {
			os.Exit(2)
		}
		_ = json.NewEncoder(os.Stdout).Encode(map[string]string{"cwd": cwd, "secret": os.Getenv("PAIR_TEST_SECRET"), "python": os.Getenv("PYTHONPATH"), "home": os.Getenv("HOME")})
	case "multiple":
		_, _ = os.Stdout.WriteString("{} {}")
	case "malformed":
		_, _ = os.Stdout.WriteString("{")
	case "sleep":
		time.Sleep(10 * time.Second)
	case "flood":
		_, _ = os.Stdout.Write(make([]byte, memoryPlanOutputLimit+1))
	default:
		_, _ = os.Stdout.WriteString(`{"ok":true}`)
	}
	os.Exit(0)
}

func memoryPlanTestCommand(t *testing.T) memoryPlanCommand {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return memoryPlanCommand{Command: []string{exe, "-B", "-s", "-m", "serve.server", "--estimate-memory", "--memory-plan-request", "{memory_plan_request}"}, WorkingDirectory: t.TempDir()}
}

func TestMemoryPlanProcessEnvironment(t *testing.T) {
	t.Setenv("PAIR_TEST_SECRET", "secret")
	t.Setenv("PYTHONPATH", "secret")
	cmd := memoryPlanTestCommand(t)
	result, err := new(Executor).runMemoryPlanCommand(context.Background(), cmd, []byte(`{"scenario":"environment"}`))
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]string
	if err := json.Unmarshal(result, &got); err != nil {
		t.Fatal(err)
	}
	cwd, err := os.Stat(got["cwd"])
	if err != nil {
		t.Fatal(err)
	}
	wantCwd, err := os.Stat(cmd.WorkingDirectory)
	if err != nil {
		t.Fatal(err)
	}
	if got["secret"] != "" || got["python"] != "" || !os.SameFile(cwd, wantCwd) || got["home"] == "" || !filepath.IsAbs(got["home"]) {
		t.Fatalf("unexpected isolated environment: %+v", got)
	}
}

func TestMemoryPlanProcessRefusals(t *testing.T) {
	for _, scenario := range []string{"multiple", "malformed", "flood", "sleep"} {
		t.Run(scenario, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 800*time.Millisecond)
			defer cancel()
			_, err := new(Executor).runMemoryPlanCommand(ctx, memoryPlanTestCommand(t), []byte(`{"scenario":"`+scenario+`"}`))
			if err == nil {
				t.Fatal("expected refusal")
			}
		})
	}
}

func TestMemoryPlanProcessBusyAndExpired(t *testing.T) {
	e := new(Executor)
	e.modelInspection.active.Store(true)
	if _, err := e.runMemoryPlanCommand(context.Background(), memoryPlanTestCommand(t), []byte(`{}`)); err == nil {
		t.Fatal("busy query accepted")
	}
	e.modelInspection.active.Store(false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := e.runMemoryPlanCommand(ctx, memoryPlanTestCommand(t), []byte(`{}`)); err == nil {
		t.Fatal("expired query accepted")
	}
}
