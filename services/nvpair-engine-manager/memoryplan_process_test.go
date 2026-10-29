// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
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
		PIDFile  string `json:"pidFile"`
	}
	if json.Unmarshal(raw, &request) != nil {
		os.Exit(2)
	}
	switch request.Scenario {
	case "grandchild":
		// Publish readiness only after the complete PID has been closed. File
		// creation alone races cancellation against the PID write.
		if err := os.WriteFile(request.PIDFile+".pending", []byte(strconv.Itoa(os.Getpid())), 0600); err != nil {
			os.Exit(2)
		}
		if err := os.Rename(request.PIDFile+".pending", request.PIDFile); err != nil {
			os.Exit(2)
		}
		time.Sleep(30 * time.Second)
	case "tree-exit", "tree-pipes", "tree-timeout", "tree-cancel", "tree-flood":
		raw, err := json.Marshal(map[string]string{"scenario": "grandchild", "pidFile": request.PIDFile})
		if err != nil {
			os.Exit(2)
		}
		childRequest := filepath.Join(filepath.Dir(os.Args[7]), "child.json")
		if os.WriteFile(childRequest, raw, 0600) != nil {
			os.Exit(2)
		}
		args := append([]string(nil), os.Args[1:]...)
		args[len(args)-1] = childRequest
		child := exec.Command(os.Args[0], args...)
		if request.Scenario == "tree-pipes" {
			child.Stdout = os.Stdout
			child.Stderr = os.Stderr
		}
		if child.Start() != nil {
			os.Exit(2)
		}
		deadline := time.Now().Add(time.Second)
		for {
			if _, err := os.Stat(request.PIDFile); err == nil {
				break
			}
			if time.Now().After(deadline) {
				os.Exit(2)
			}
			time.Sleep(time.Millisecond)
		}
		if request.Scenario == "tree-timeout" || request.Scenario == "tree-cancel" {
			time.Sleep(30 * time.Second)
		}
		if request.Scenario == "tree-flood" {
			_, _ = os.Stderr.Write(make([]byte, memoryPlanOutputLimit+1))
		}
		_, _ = os.Stdout.WriteString(`{"ok":true}`)
	case "environment":
		cwd, err := os.Getwd()
		if err != nil {
			os.Exit(2)
		}
		_ = json.NewEncoder(os.Stdout).Encode(map[string]string{"cwd": cwd, "secret": os.Getenv("PAIR_TEST_SECRET"), "python": os.Getenv("PYTHONPATH"), "home": os.Getenv("HOME"), "temp": os.Getenv("TEMP"), "tmp": os.Getenv("TMP"), "tmpdir": os.Getenv("TMPDIR"), "env": strings.Join(os.Environ(), "\n"), "request": os.Args[7]})
	case "multiple":
		_, _ = os.Stdout.WriteString("{} {}")
	case "malformed":
		_, _ = os.Stdout.WriteString("{")
	case "sleep":
		time.Sleep(10 * time.Second)
	case "flood":
		_, _ = os.Stdout.Write(make([]byte, memoryPlanOutputLimit+1))
	case "stderr-flood":
		_, _ = os.Stderr.Write(make([]byte, memoryPlanOutputLimit+1))
	case "shared-flood":
		_, _ = os.Stdout.Write(make([]byte, memoryPlanOutputLimit/2))
		_, _ = os.Stderr.Write(make([]byte, memoryPlanOutputLimit/2+1))
	case "failure":
		_, _ = os.Stderr.WriteString("secret private path")
		os.Exit(2)
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
	for _, key := range []string{"HTTP_PROXY", "STRATA_TOKEN", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"} {
		t.Setenv(key, "secret")
	}
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
	for _, key := range []string{"temp", "tmp", "tmpdir"} {
		if got[key] != got["home"] {
			t.Fatalf("%s not private", key)
		}
	}
	if strings.Contains(got["env"], "secret") {
		t.Fatal("ambient environment leaked")
	}
	if filepath.Dir(got["request"]) != got["home"] {
		t.Fatal("request outside private root")
	}
	if _, err := os.Stat(got["home"]); !os.IsNotExist(err) {
		t.Fatalf("private root not removed: %v", err)
	}
}

func TestMemoryPlanProcessRefusals(t *testing.T) {
	for _, scenario := range []string{"multiple", "malformed", "flood", "stderr-flood", "shared-flood", "failure", "sleep"} {
		t.Run(scenario, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 800*time.Millisecond)
			defer cancel()
			_, err := new(Executor).runMemoryPlanCommand(ctx, memoryPlanTestCommand(t), []byte(`{"scenario":"`+scenario+`"}`))
			if err == nil {
				t.Fatal("expected refusal")
			}
			if strings.Contains(err.Error(), "secret") {
				t.Fatal("stderr leaked")
			}
		})
	}
}

func testMemoryPlanProcessTrees(t *testing.T) {
	t.Helper()
	for _, scenario := range []string{"tree-exit", "tree-pipes", "tree-timeout", "tree-cancel", "tree-flood"} {
		t.Run(scenario, func(t *testing.T) {
			pidFile := filepath.Join(t.TempDir(), "pid")
			request, err := json.Marshal(map[string]string{"scenario": scenario, "pidFile": pidFile})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2500*time.Millisecond)
			defer cancel()
			cancelledWhenReady := make(chan bool, 1)
			if scenario == "tree-cancel" {
				go func() {
					for {
						if _, err := os.Stat(pidFile); err == nil {
							cancel()
							cancelledWhenReady <- true
							return
						}
						select {
						case <-ctx.Done():
							cancelledWhenReady <- false
							return
						case <-time.After(time.Millisecond):
						}
					}
				}()
			}
			started := time.Now()
			_, runErr := new(Executor).runMemoryPlanCommand(ctx, memoryPlanTestCommand(t), request)
			if scenario == "tree-cancel" && !<-cancelledWhenReady {
				t.Fatalf("cancellation never observed descendant readiness (run: %v)", runErr)
			}
			raw, err := os.ReadFile(pidFile)
			if err != nil {
				t.Fatalf("child never started: %v (run: %v)", err, runErr)
			}
			pid, err := strconv.Atoi(string(raw))
			if err != nil {
				t.Fatal(err)
			}
			assertMemoryPlanProcessStopped(t, pid)
			if time.Since(started) > 3*time.Second {
				t.Fatal("query exceeded wall bound")
			}
			if scenario == "tree-exit" && runErr != nil {
				t.Fatalf("normal exit: %v", runErr)
			}
			if scenario != "tree-exit" && runErr == nil {
				t.Fatal("expected bounded refusal")
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

func TestMemoryPlanProcessAdmissionReleasedOnRefusal(t *testing.T) {
	for _, scenario := range []string{"invalid request", "oversized request", "invalid command", "start failure", "malformed response"} {
		t.Run(scenario, func(t *testing.T) {
			e := new(Executor)
			cmd := memoryPlanTestCommand(t)
			request := []byte(`{}`)
			switch scenario {
			case "invalid request":
				request = []byte(`{`)
			case "oversized request":
				request = []byte(strings.Repeat(" ", memoryPlanRequestLimit+1))
			case "invalid command":
				cmd.Command[1] = "--config"
			case "start failure":
				cmd.Command[0] = filepath.Join(t.TempDir(), "missing.exe")
			case "malformed response":
				request = []byte(`{"scenario":"malformed"}`)
			}
			if _, err := e.runMemoryPlanCommand(context.Background(), cmd, request); err == nil {
				t.Fatal("expected refusal")
			}
			if e.modelInspection.active.Load() {
				t.Fatal("admission retained after confirmed cleanup")
			}
		})
	}
}

func TestMemoryPlanProcessStartupReapDeadline(t *testing.T) {
	blocked := make(chan struct{})
	defer close(blocked)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	start := time.Now()
	err := reapMemoryPlanStartFailure(ctx, func() error { <-blocked; return nil })
	if !errors.Is(err, errMemoryPlanCleanup) {
		t.Fatalf("unconfirmed reap: %v", err)
	}
	if time.Since(start) > time.Second {
		t.Fatal("startup reap exceeded deadline")
	}
	if err := reapMemoryPlanStartFailure(context.Background(), func() error { return errors.New("expected exit status") }); err != nil {
		t.Fatal(err)
	}
}
