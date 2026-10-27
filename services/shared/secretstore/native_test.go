// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package secretstore

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"flag"
	"io"
	"os"
	"os/exec"
	"runtime"
	"testing"
	"time"
)

var nativeEnabled = flag.Bool("secretstore-native", false, "opt in to exact-entry native persistence validation")
var nativeStep = flag.String("secretstore-native-step", "", "native persistence child operation")

type nativeRequest struct {
	ID    string
	Value []byte
}

// No native adapter is initialized unless explicitly opted in. Each child gets
// its fake test credential only over anonymous stdin, never arguments or env.
func TestNativeSecretStore(t *testing.T) {
	if !*nativeEnabled {
		t.Skip("native store validation requires -secretstore-native; UNTESTED")
	}
	if *nativeStep != "" {
		nativeChild(t)
		return
	}
	random := make([]byte, 32)
	if _, err := rand.Read(random); err != nil {
		t.Fatal("random test data unavailable")
	}
	request := nativeRequest{ID: "native-" + hex.EncodeToString(random[:12]), Value: []byte(hex.EncodeToString(random[12:]))}
	defer nativeProcess(t, "delete", request)
	nativeProcess(t, "write", request)
	nativeProcess(t, "read", request)
	if _, err := rand.Read(random); err != nil {
		t.Fatal("replacement test data unavailable")
	}
	request.Value = []byte(hex.EncodeToString(random))
	nativeProcess(t, "write", request)
	nativeProcess(t, "read", request)
	nativeProcess(t, "delete", request)
	nativeProcess(t, "missing", request)
}

func nativeProcess(t *testing.T, step string, request nativeRequest) {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal("test executable unavailable")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, executable, "-test.run=^TestNativeSecretStore$", "-secretstore-native", "-secretstore-native-step="+step)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal("child input pipe unavailable")
	}
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		t.Fatal("native child start failed")
	}
	encodeErr := json.NewEncoder(stdin).Encode(request)
	closeErr := stdin.Close()
	waitErr := cmd.Wait()
	if encodeErr != nil || closeErr != nil || waitErr != nil {
		t.Fatalf("native persistence child %s failed (diagnostics omitted)", step)
	}
}

func nativeChild(t *testing.T) {
	var request nativeRequest
	if err := json.NewDecoder(io.LimitReader(os.Stdin, 4096)).Decode(&request); err != nil {
		t.Fatal("invalid child input")
	}
	if !validName(request.ID) || len(request.ID) < 7 || request.ID[:7] != "native-" {
		t.Fatal("invalid native entry id")
	}
	s := New()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	switch *nativeStep {
	case "write":
		if s.Set(ctx, "native-test", request.ID, request.Value) != nil {
			t.Fatal("native write failed")
		}
	case "read":
		value, err := s.Get(ctx, "native-test", request.ID)
		wantValue(t, value, request.Value, err)
	case "delete":
		if s.Delete(ctx, "native-test", request.ID) != nil {
			t.Fatal("native exact-entry cleanup failed")
		}
	case "missing":
		_, err := s.Get(ctx, "native-test", request.ID)
		wantError(t, err, ErrNotFound)
	default:
		t.Fatal("invalid child step")
	}
}

func TestUnsupportedNativeOS(t *testing.T) {
	if runtime.GOOS == "darwin" || runtime.GOOS == "windows" || runtime.GOOS == "linux" {
		t.Skip("unsupported-OS adapter is cross-platform only")
	}
	s := New()
	_, err := s.Get(t.Context(), "purpose", "id")
	wantError(t, err, ErrUnavailable)
}
