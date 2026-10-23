// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"nvpair-shared/servicectl"
)

// These tests run the diagnose command as its own process, the way an
// operator does, against a private fake-service endpoint. The process is this
// test binary in command mode (see TestMain), so it is the image the fixed-path
// test wrapper already covers. The command only dials: it never listens, starts
// a service, or reaches the user's service endpoint or profile.

// nativeRunCap is the test's own limit on one command process. The command
// bounds itself with diagnoseTimeout, so reaching this cap is a hang.
const nativeRunCap = 30 * time.Second

// nativeSandbox is what one command process may touch: a private endpoint and
// an empty profile directory, where a mistakenly launched serve would write
// its logs.
type nativeSandbox struct {
	dir      string
	endpoint string
	profile  string
}

func newNativeSandbox(t *testing.T, endpoint string) nativeSandbox {
	t.Helper()
	dir := shortTempDir(t)
	profile := filepath.Join(dir, "profile")
	if err := os.Mkdir(profile, 0o700); err != nil {
		t.Fatalf("create the empty profile: %v", err)
	}
	return nativeSandbox{dir: dir, endpoint: endpoint, profile: profile}
}

// unusedEndpoint returns a private endpoint nothing listens on, after checking
// that nothing does.
func unusedEndpoint(t *testing.T) string {
	t.Helper()
	endpoint := testEndpoint(shortTempDir(t))
	if alreadyAnswering(context.Background(), endpoint) {
		t.Fatalf("private endpoint %q is already answering", endpoint)
	}
	if runtime.GOOS != "windows" {
		if _, err := os.Lstat(endpoint); !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("private endpoint %q already exists: %v", endpoint, err)
		}
	}
	return endpoint
}

// fixtureEndpoint is the endpoint newDiagnoseFixture listens on.
func fixtureEndpoint() string { return os.Getenv(servicectl.EndpointEnv) }

// env is the parent's environment with the command role selected and every
// path the command could resolve pointed into the sandbox.
func (s nativeSandbox) env() []string {
	override := map[string]string{
		serviceCommandEnv:      "1",
		fakeBrokerEnv:          "",
		serviceProcessDirEnv:   "",
		servicectl.EndpointEnv: s.endpoint,
		"LOCALAPPDATA":         s.profile,
		"HOME":                 s.profile,
		"XDG_CONFIG_HOME":      s.profile,
	}
	var env []string
	for _, kv := range os.Environ() {
		key, _, _ := strings.Cut(kv, "=")
		if _, replaced := override[strings.ToUpper(key)]; replaced {
			continue
		}
		env = append(env, kv)
	}
	for key, value := range override {
		env = append(env, key+"="+value)
	}
	return env
}

func (s nativeSandbox) command(t *testing.T, ctx context.Context, args ...string) *exec.Cmd {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("locate the command image: %v", err)
	}
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Env, cmd.Dir, cmd.WaitDelay = s.env(), s.dir, 2*time.Second
	return cmd
}

// exitCode is the process's exit code, or -1 when a signal ended it. A run
// stopped by the test's cap is a failure, not an exit.
func exitCode(t *testing.T, ctx context.Context, err error) int {
	t.Helper()
	if ctx.Err() != nil {
		t.Fatalf("command was still running at the %s cap", nativeRunCap)
	}
	if err == nil {
		return 0
	}
	var exit *exec.ExitError
	if !errors.As(err, &exit) {
		t.Fatalf("run the command: %v", err)
	}
	return exit.ExitCode()
}

type nativeResult struct {
	code           int
	stdout, stderr string
	elapsed        time.Duration
}

// run executes the command with captured output.
func (s nativeSandbox) run(t *testing.T, args ...string) nativeResult {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), nativeRunCap)
	defer cancel()
	cmd := s.command(t, ctx, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	start := time.Now()
	err := cmd.Run()
	return nativeResult{code: exitCode(t, ctx, err), stdout: stdout.String(), stderr: stderr.String(), elapsed: time.Since(start)}
}

// assertServeNotLaunched checks that the command left the profile empty, which
// a launched serve would not.
func (s nativeSandbox) assertServeNotLaunched(t *testing.T) {
	t.Helper()
	entries, err := os.ReadDir(s.profile)
	if err != nil {
		t.Fatalf("read the profile: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("the command wrote %d entries to the profile; serve may have started", len(entries))
	}
}

func decodeNativeReport(t *testing.T, stdout string) diagnoseReport {
	t.Helper()
	var report diagnoseReport
	if err := json.Unmarshal([]byte(stdout), &report); err != nil {
		t.Fatalf("decode the JSON report: %v\n%s", err, stdout)
	}
	return report
}

// nativeReplies answers each request from a scripted service, passing the
// method's occurrence so a later read can differ from the first. Notifications
// and a reply to another request id come before every answer.
func nativeReplies(t *testing.T, service *fakeDiagnoseService) func(method string) diagnoseReply {
	t.Helper()
	var mu sync.Mutex
	seen := map[string]int{}
	return func(method string) diagnoseReply {
		answer, ok := service.answers[method]
		if !ok {
			return diagnoseReply{frames: []string{`{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"` + sentinelError + `"}}`}}
		}
		mu.Lock()
		occurrence := seen[method]
		seen[method]++
		mu.Unlock()
		result, err := answer(context.Background(), occurrence)
		if err != nil {
			t.Errorf("scripted answer for %s: %v", method, err)
			return diagnoseReply{drop: true}
		}
		return diagnoseReply{frames: []string{
			`{"jsonrpc":"2.0","method":"service/log","params":{"text":"` + sentinelNotice + `"}}`,
			`{"jsonrpc":"2.0","id":99,"result":{"text":"` + sentinelNotice + `"}}`,
			`{"jsonrpc":"2.0","id":%s,"result":` + string(result) + `}`,
		}}
	}
}

func TestDiagnoseNativeNotRunningAtUnusedEndpoint(t *testing.T) {
	sandbox := newNativeSandbox(t, unusedEndpoint(t))

	t.Run("json", func(t *testing.T) {
		got := sandbox.run(t, "diagnose", "--json")
		if got.code != diagnoseExitNotRunning || got.stderr != "" {
			t.Fatalf("exit code = %d, stderr = %q; want %d and no stderr", got.code, got.stderr, diagnoseExitNotRunning)
		}
		report := decodeNativeReport(t, got.stdout)
		if report.Complete || report.Service.Status != diagnoseNotRunning || report.Policy.Status != diagnoseNotRunning ||
			report.ResidentConfiguration.Status != diagnoseNotRunning {
			t.Fatalf("report = %+v, want a not-running report", report)
		}
		for _, facade := range report.BuiltinFacades {
			if facade.Status != diagnoseNotRunning {
				t.Fatalf("facade %s is %s, want %s", facade.Engine, facade.Status, diagnoseNotRunning)
			}
		}
		assertProbeStatements(t, report, "Inference and peer connectivity were not tested.")
	})
	t.Run("text", func(t *testing.T) {
		got := sandbox.run(t, "diagnose")
		if got.code != diagnoseExitNotRunning || got.stderr != "" {
			t.Fatalf("exit code = %d, stderr = %q; want %d and no stderr", got.code, got.stderr, diagnoseExitNotRunning)
		}
		for _, want := range []string{"Service: not-running", "Capture: incomplete.", "Inference and peer connectivity were not tested."} {
			if !strings.Contains(got.stdout, want) {
				t.Fatalf("text output lacks %q:\n%s", want, got.stdout)
			}
		}
	})

	sandbox.assertServeNotLaunched(t)
	if alreadyAnswering(context.Background(), sandbox.endpoint) {
		t.Fatal("a listener appeared at the unused endpoint")
	}
}

func TestDiagnoseNativeReportsBrokerRestartsAndFenceChanges(t *testing.T) {
	moved := healthyPolicy()
	moved.revision = "1792285135482000001"
	nextGeneration := healthyPolicy()
	nextGeneration.generation++

	test := func(name string, status, policy diagnoseAnswer, wantCode int, edit func(*diagnoseReport)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			service := newHealthyDiagnoseService(t)
			service.answers[methodStatus] = status
			service.answers["policy:get"] = policy
			fixture := newDiagnoseFixture(t, nativeReplies(t, service))
			sandbox := newNativeSandbox(t, fixtureEndpoint())

			got := sandbox.run(t, "diagnose", "--json")
			if got.code != wantCode || got.stderr != "" {
				t.Fatalf("exit code = %d, stderr = %q; want %d and no stderr\n%s", got.code, got.stderr, wantCode, got.stdout)
			}
			assertNoSentinels(t, got.stdout)
			want := healthyReport()
			edit(&want)
			if report := decodeNativeReport(t, got.stdout); !reflect.DeepEqual(report, want) {
				t.Fatalf("report mismatch\ngot:  %+v\nwant: %+v", report, want)
			}
			assertCalls(t, fixture.recorded(), wantDiagnoseSequence())
			waitClosedByClient(t, fixture, len(wantDiagnoseSequence()))
			sandbox.assertServeNotLaunched(t)
		})
	}
	policy := fixed(policyGetResult(t, healthyPolicy()))
	changedFacades := func(r *diagnoseReport) { r.BuiltinFacades = facades(diagnoseChanged) }

	test("restarted broker with steady fences", fixed(serviceStatusResult(t, 2)), policy, diagnoseExitComplete,
		func(r *diagnoseReport) { r.Service = knownService(2) })
	test("broker restarts between the fences", changing(serviceStatusResult(t, 0), serviceStatusResult(t, 1)), policy, diagnoseExitIncomplete,
		func(r *diagnoseReport) {
			r.Complete = false
			r.Service = diagnoseService{Status: diagnoseChanged}
			r.Policy = diagnosePolicy{Status: diagnoseChanged}
			r.ResidentConfiguration = diagnoseResidentConfig{Status: diagnoseChanged}
			changedFacades(r)
		})
	test("policy revision moves", fixed(serviceStatusResult(t, 0)),
		changing(policyGetResult(t, healthyPolicy()), policyGetResult(t, moved)), diagnoseExitIncomplete,
		func(r *diagnoseReport) {
			r.Complete = false
			r.Policy = diagnosePolicy{Status: diagnoseChanged}
			r.ResidentConfiguration = diagnoseResidentConfig{Status: diagnoseChanged}
		})
	test("resident observation generation moves", fixed(serviceStatusResult(t, 0)),
		changing(policyGetResult(t, healthyPolicy()), policyGetResult(t, nextGeneration)), diagnoseExitIncomplete,
		func(r *diagnoseReport) {
			r.Complete = false
			r.ResidentConfiguration = diagnoseResidentConfig{Status: diagnoseChanged}
		})
}

func TestDiagnoseNativeTextShowsBrokerRestarts(t *testing.T) {
	service := newHealthyDiagnoseService(t)
	service.answers[methodStatus] = fixed(serviceStatusResult(t, 2))
	newDiagnoseFixture(t, nativeReplies(t, service))
	sandbox := newNativeSandbox(t, fixtureEndpoint())

	got := sandbox.run(t, "diagnose")
	if got.code != diagnoseExitComplete || got.stderr != "" {
		t.Fatalf("exit code = %d, stderr = %q; want %d and no stderr\n%s", got.code, got.stderr, diagnoseExitComplete, got.stdout)
	}
	assertNoSentinels(t, got.stdout)
	for _, want := range []string{"pid 4242, broker pid 4243, broker restarts 2", "Capture: complete.", "Inference and peer connectivity were not tested."} {
		if !strings.Contains(got.stdout, want) {
			t.Fatalf("text output lacks %q:\n%s", want, got.stdout)
		}
	}
}

func TestDiagnoseNativeDeadlineEndsASlowReply(t *testing.T) {
	healthy := nativeReplies(t, newHealthyDiagnoseService(t))
	// The policy read is never answered; only the command's own deadline ends it.
	fixture := newDiagnoseFixture(t, func(method string) diagnoseReply {
		if method == "policy:get" {
			return diagnoseReply{}
		}
		return healthy(method)
	})
	sandbox := newNativeSandbox(t, fixtureEndpoint())

	got := sandbox.run(t, "diagnose", "--json")
	t.Logf("the slow capture ended after %s", got.elapsed)
	if got.code != diagnoseExitIncomplete || got.stderr != "" {
		t.Fatalf("exit code = %d, stderr = %q; want %d and no stderr\n%s", got.code, got.stderr, diagnoseExitIncomplete, got.stdout)
	}
	assertNoSentinels(t, got.stdout)
	want := healthyReport()
	want.Complete = false
	want.Policy = diagnosePolicy{Status: diagnoseUnavailable}
	want.ResidentConfiguration = diagnoseResidentConfig{Status: diagnoseUnavailable}
	want.BuiltinFacades = facades(diagnoseUnavailable)
	if report := decodeNativeReport(t, got.stdout); !reflect.DeepEqual(report, want) {
		t.Fatalf("report mismatch\ngot:  %+v\nwant: %+v", report, want)
	}
	assertCalls(t, fixture.recorded(), wantDiagnoseSequence()[:2])
	// The command closed both connections, the answered one and the stalled one.
	waitClosedByClient(t, fixture, 2)
	sandbox.assertServeNotLaunched(t)
}

func TestDiagnoseNativeOutputSurvivesRedirectedAndClosedStreams(t *testing.T) {
	// start runs a capture against a fresh fixture with the command's standard
	// streams set by attach, and returns the finished command and its exit code.
	start := func(t *testing.T, attach func(cmd *exec.Cmd)) (*exec.Cmd, int) {
		t.Helper()
		fixture := newDiagnoseFixture(t, nativeReplies(t, newHealthyDiagnoseService(t)))
		sandbox := newNativeSandbox(t, fixtureEndpoint())
		ctx, cancel := context.WithTimeout(context.Background(), nativeRunCap)
		defer cancel()
		cmd := sandbox.command(t, ctx, "diagnose", "--json")
		attach(cmd)
		code := exitCode(t, ctx, cmd.Run())
		assertCalls(t, fixture.recorded(), wantDiagnoseSequence())
		waitClosedByClient(t, fixture, len(wantDiagnoseSequence()))
		sandbox.assertServeNotLaunched(t)
		return cmd, code
	}

	t.Run("output redirected to files", func(t *testing.T) {
		var stdoutPath, stderrPath string
		_, code := start(t, func(cmd *exec.Cmd) {
			out, err := os.CreateTemp(t.TempDir(), "stdout")
			if err != nil {
				t.Fatalf("create the stdout file: %v", err)
			}
			errFile, err := os.CreateTemp(t.TempDir(), "stderr")
			if err != nil {
				t.Fatalf("create the stderr file: %v", err)
			}
			t.Cleanup(func() { _ = out.Close(); _ = errFile.Close() })
			stdoutPath, stderrPath = out.Name(), errFile.Name()
			cmd.Stdout, cmd.Stderr = out, errFile
		})
		if code != diagnoseExitComplete {
			t.Fatalf("exit code = %d, want %d", code, diagnoseExitComplete)
		}
		stdout, err := os.ReadFile(stdoutPath)
		if err != nil {
			t.Fatalf("read the redirected stdout: %v", err)
		}
		stderr, err := os.ReadFile(stderrPath)
		if err != nil {
			t.Fatalf("read the redirected stderr: %v", err)
		}
		if len(stderr) != 0 {
			t.Fatalf("redirected stderr = %q, want empty", stderr)
		}
		assertNoSentinels(t, string(stdout))
		if report := decodeNativeReport(t, string(stdout)); !reflect.DeepEqual(report, healthyReport()) {
			t.Fatalf("report = %+v, want %+v", report, healthyReport())
		}
	})
	t.Run("output discarded to the null device", func(t *testing.T) {
		// A nil stream is the null device, which is also what stdin reads as
		// end of file.
		_, code := start(t, func(cmd *exec.Cmd) { cmd.Stdin, cmd.Stdout, cmd.Stderr = nil, nil, nil })
		if code != diagnoseExitComplete {
			t.Fatalf("exit code = %d, want %d", code, diagnoseExitComplete)
		}
	})
	t.Run("stdout reader already closed", func(t *testing.T) {
		cmd, code := start(t, func(cmd *exec.Cmd) {
			r, w, err := os.Pipe()
			if err != nil {
				t.Fatalf("create the pipe: %v", err)
			}
			if err := r.Close(); err != nil {
				t.Fatalf("close the pipe reader: %v", err)
			}
			t.Cleanup(func() { _ = w.Close() })
			cmd.Stdout = w
		})
		// Windows reports the failed write and the command keeps its exit
		// code. Elsewhere a write to a broken pipe on standard output ends a Go
		// program with SIGPIPE, as it ends any pipeline writer.
		if code == diagnoseExitComplete {
			return
		}
		status, ok := cmd.ProcessState.Sys().(syscall.WaitStatus)
		if runtime.GOOS == "windows" || !ok || !status.Signaled() || status.Signal() != syscall.SIGPIPE {
			t.Fatalf("exit code = %d (%v), want %d or death by SIGPIPE", code, cmd.ProcessState, diagnoseExitComplete)
		}
	})
}

func TestDiagnoseNativeHelpListsDiagnose(t *testing.T) {
	sandbox := newNativeSandbox(t, unusedEndpoint(t))
	test := func(name, flag string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			got := sandbox.run(t, flag)
			if got.code != 0 || got.stdout != "" {
				t.Fatalf("exit code = %d, stdout = %q; want 0 and no stdout", got.code, got.stdout)
			}
			for _, want := range []string{"usage: nvpair-service", diagnoseUsage} {
				if !strings.Contains(got.stderr, want) {
					t.Fatalf("help lacks %q:\n%s", want, got.stderr)
				}
			}
		})
	}
	test("short help flag", "-h")
	test("long help flag", "--help")
	sandbox.assertServeNotLaunched(t)
	if alreadyAnswering(context.Background(), sandbox.endpoint) {
		t.Fatal("a listener appeared at the unused endpoint")
	}
}
