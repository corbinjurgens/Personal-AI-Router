// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"nvpair-shared/buildapply"
	"nvpair-shared/servicectl"
	"nvpair-shared/testbin"
)

// The native child below reuses the registered service test image. It opens no
// listener and executes only the explicitly selected synthetic test mode.
type buildApplyExitFixture struct {
	events  *[]string
	name    string
	waitErr error
}

func (p *buildApplyExitFixture) Wait(ctx context.Context) error {
	*p.events = append(*p.events, "wait:"+p.name)
	return p.waitErr
}
func (p *buildApplyExitFixture) Close() error {
	*p.events = append(*p.events, "close:"+p.name)
	return nil
}

func TestBuildApplyProcessOpensIndependentProofBeforeStop(t *testing.T) {
	var events []string
	old := []buildapply.RunnerReservation{{PID: 1, Identity: buildapply.ProcessIdentity{Executable: "service", Created: "123"}}, {PID: 2, Identity: buildapply.ProcessIdentity{Executable: "broker", Created: "456"}}}
	err := stopBuildApplyOwned(context.Background(), old, func(ctx context.Context, r buildapply.RunnerReservation) (buildApplyExitProof, error) {
		events = append(events, "open:"+r.Identity.Executable)
		return &buildApplyExitFixture{events: &events, name: r.Identity.Executable}, nil
	}, func(context.Context) error { events = append(events, "ordered-service-stop"); return nil })
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"open:service", "open:broker", "ordered-service-stop", "wait:service", "wait:broker", "close:service", "close:broker"}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("process ordering: %v", events)
	}
}

func TestBuildApplyProcessUncertainIdentityPreventsStop(t *testing.T) {
	stops := 0
	err := stopBuildApplyOwned(context.Background(), []buildapply.RunnerReservation{{PID: 1}}, func(context.Context, buildapply.RunnerReservation) (buildApplyExitProof, error) {
		return nil, errors.New("identity unavailable")
	}, func(context.Context) error { stops++; return nil })
	if err == nil || stops != 0 {
		t.Fatalf("uncertain identity: stops=%d, error=%v", stops, err)
	}
}

func TestBuildApplyProcessReplyLossStillWaitsAndRemainsUncertain(t *testing.T) {
	var events []string
	lost := errors.New("reply lost")
	err := stopBuildApplyOwned(context.Background(), []buildapply.RunnerReservation{{PID: 1}}, func(context.Context, buildapply.RunnerReservation) (buildApplyExitProof, error) {
		return &buildApplyExitFixture{events: &events, name: "service"}, nil
	}, func(context.Context) error { return lost })
	if !errors.Is(err, lost) || !reflect.DeepEqual(events, []string{"wait:service", "close:service"}) {
		t.Fatalf("reply loss: events=%v, error=%v", events, err)
	}
}

func TestBuildApplyProcessExitUnknownCannotAuthorizeSwap(t *testing.T) {
	var events []string
	unknown := errors.New("independent exit unknown")
	err := stopBuildApplyOwned(context.Background(), []buildapply.RunnerReservation{{PID: 1}}, func(context.Context, buildapply.RunnerReservation) (buildApplyExitProof, error) {
		return &buildApplyExitFixture{events: &events, name: "service", waitErr: unknown}, nil
	}, func(context.Context) error { return nil })
	if !errors.Is(err, unknown) {
		t.Fatalf("unknown exit accepted: %v", err)
	}
}

const buildApplyProcessChildEnv = "NVPAIR_BUILD_APPLY_TEST_CHILD"

func TestBuildApplyProcessChild(t *testing.T) {
	if os.Getenv(buildApplyProcessChildEnv) != "1" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	self, err := inspectBuildApplyProcess(ctx, os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	if err := json.NewEncoder(os.Stdout).Encode(self); err != nil {
		t.Fatal(err)
	}
	released := make(chan error, 1)
	go func() { var b [1]byte; _, err := io.ReadFull(os.Stdin, b[:]); released <- err }()
	select {
	case err := <-released:
		if err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal("bounded child lifetime expired")
	}
}

func TestBuildApplyProcessWindowsHandleFencesRealChildExit(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows process-handle acceptance is executed on PC3")
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	root := os.Getenv(testbin.EnvRoot)
	if !filepath.IsAbs(root) {
		t.Fatal("native process proof requires the existing fixed-path test wrapper")
	}
	expected := filepath.Join(root, "go-test", "nvpair-service", "nvpair-service.test.exe")
	if !strings.EqualFold(filepath.Clean(exe), filepath.Clean(expected)) {
		t.Fatalf("service test image is not its registered fixed destination: %s", exe)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-test.run=^TestBuildApplyProcessChild$", "-test.count=1")
	var env []string
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if key != fakeBrokerEnv && key != serviceCommandEnv && key != buildApplyProcessChildEnv {
			env = append(env, entry)
		}
	}
	cmd.Env = append(env, buildApplyProcessChildEnv+"=1")
	input, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	output, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waited := make(chan error, 1)
	go func() { waited <- cmd.Wait() }()
	waitConsumed := false
	t.Cleanup(func() {
		_ = input.Close()
		cancel()
		if !waitConsumed {
			select {
			case <-waited:
			case <-time.After(5 * time.Second):
				t.Error("owned fixed-path child did not exit")
			}
		}
	})
	var child buildapply.RunnerReservation
	if err := json.NewDecoder(output).Decode(&child); err != nil {
		t.Fatal(err)
	}
	actual, err := inspectBuildApplyProcess(ctx, cmd.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	if child != actual {
		t.Fatalf("child self-report differs from independent inspection: %+v / %+v", child, actual)
	}
	wrong := actual
	wrong.Identity.Created += "0"
	if proof, err := openBuildApplyExit(ctx, wrong); err == nil {
		_ = proof.Close()
		t.Fatal("held handle accepted a different creation lifetime")
	}
	proof, err := openBuildApplyExit(ctx, actual)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := proof.Close(); err != nil {
			t.Error(err)
		}
	}()
	live, cancelLive := context.WithTimeout(ctx, 50*time.Millisecond)
	err = proof.Wait(live)
	cancelLive()
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("live child authorized swap: %v", err)
	}
	err = stopBuildApplyOwned(ctx, []buildapply.RunnerReservation{actual}, func(ctx context.Context, want buildapply.RunnerReservation) (buildApplyExitProof, error) {
		return openBuildApplyExit(ctx, want)
	}, func(context.Context) error {
		if _, err := input.Write([]byte{1}); err != nil {
			return err
		}
		return input.Close()
	})
	if err != nil {
		t.Fatalf("native stop-before-swap boundary: %v", err)
	}
	if err := proof.Wait(ctx); err != nil {
		t.Fatalf("real child exit proof: %v", err)
	}
	select {
	case err := <-waited:
		waitConsumed = true
		if err != nil {
			t.Fatalf("synthetic child exit: %v", err)
		}
	case <-ctx.Done():
		t.Fatal("owned child did not reap after swap was authorized")
	}
}

func TestBuildApplyAdmissionWindowsStartupGateUsesPrivateJournal(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows startup gate fixture is executed on PC3")
	}
	id, err := buildapply.NewOperationID()
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(t.TempDir(), "private-"+id)
	// Install the protected ACL at creation. No inherited TEMP ACL is trusted.
	if err := createBuildApplyDirectory(base); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LOCALAPPDATA", base)
	t.Setenv(buildApplyOperationEnv, "")
	t.Setenv(buildApplyRoleEnv, "")
	if err := servicectl.CheckBuildApplyStartup(); err != nil {
		t.Fatalf("absent journal: %v", err)
	}
	j, err := buildapply.OpenJournal(true)
	if err != nil {
		t.Fatal(err)
	}
	defer j.Close()
	lock, err := j.TryLock()
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	f := newBuildApplyRunFixture(t)
	record, err := json.Marshal(struct {
		Path       string
		Size       int64
		Mode       uint32
		Hash, Link string
	}{"nvpair", 5, 0o755, strings.Repeat("a", 64), ""})
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(append([]byte(fmt.Sprintf("%d:", len(record))), record...))
	digest := hex.EncodeToString(hash[:])
	f.plan.receipt.CandidateSHA256 = digest
	inventory := buildapply.Inventory{OperationID: f.plan.receipt.OperationID, Platform: runtime.GOOS, Architecture: runtime.GOARCH, PackageSHA256: digest, Entries: []buildapply.InventoryEntry{{Path: "nvpair", Size: 5, Mode: 0o755, SHA256: strings.Repeat("a", 64)}}}
	if _, err := j.Begin(lock, f.plan.receipt, inventory); err != nil {
		t.Fatal(err)
	}
	if err := servicectl.CheckBuildApplyStartup(); !errors.Is(err, buildapply.ErrRecoveryNeeded) {
		t.Fatalf("pending startup gate: %v", err)
	}
	if err := checkBuildApplyProcessEntry(context.Background()); !errors.Is(err, buildapply.ErrRecoveryNeeded) {
		t.Fatalf("ordinary Windows service contender: %v", err)
	}
	if err := os.WriteFile(filepath.Join(j.Path(), "operation.json"), []byte("invalid"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := servicectl.CheckBuildApplyStartup(); !errors.Is(err, buildapply.ErrRecoveryNeeded) {
		t.Fatalf("unreadable startup gate: %v", err)
	}
}

func TestBuildApplyTransactionWindowsMetadataPreservesNamedStream(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows stream metadata is executed on PC3")
	}
	base := filepath.Join(t.TempDir(), "private")
	if err := createBuildApplyDirectory(base); err != nil {
		t.Fatal(err)
	}
	source, destination := filepath.Join(base, "source.dat"), filepath.Join(base, "destination.dat")
	for _, path := range []string{source, destination} {
		if err := os.WriteFile(path, []byte("synthetic bundle data"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	want := []byte("[ZoneTransfer]\r\nZoneId=3\r\n")
	if err := os.WriteFile(source+":Zone.Identifier", want, 0o600); err != nil {
		t.Fatal(err)
	}
	metadata, err := readBuildApplyNativeMetadata(source)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(metadata.Streams["Zone.Identifier"], want) {
		t.Fatalf("source named stream inventory: %+v", metadata.Streams)
	}
	if err := writeBuildApplyNativeMetadata(destination, metadata); err != nil {
		t.Fatal(err)
	}
	got, err := readBuildApplyNativeMetadata(destination)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, metadata) {
		t.Fatalf("copied native stream metadata: %+v / %+v", got, metadata)
	}
}
