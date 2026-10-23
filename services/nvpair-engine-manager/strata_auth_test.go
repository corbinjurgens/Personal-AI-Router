// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
)

const strataAuthKey = "FAKE-AUTH-KEY-0123456789abcdef"

// startKeyedStrata approves, applies and starts the fake prepared Strata with
// a native API key, recording the key the server process receives.
func startKeyedStrata(t *testing.T) (strataServingFixture, string) {
	t.Helper()
	f, keyLog := prepareKeyedStrata(t, "")
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.ex.Stop("fake") })
	return f, keyLog
}

// prepareKeyedStrata approves and applies the keyed launch without starting
// it. A non-empty echo makes the fake server print the key it was given.
func prepareKeyedStrata(t *testing.T, echo string) (strataServingFixture, string) {
	t.Helper()
	f := newStrataServingFixture(t)
	keyLog := filepath.Join(f.root, "key.log")
	settingsState(t, f.ex).plat.Runtime.Env["FAKE_API_KEY_LOG"] = keyLog
	if echo != "" {
		settingsState(t, f.ex).plat.Runtime.Env["FAKE_ECHO_API_KEY"] = echo
	}
	key := strataAuthKey
	f.config.APIKey = &key
	f.saveConfig(t)
	p := approveLlamaServing(t, f.llamaServingFixture)
	if data, err := json.Marshal(p); err != nil || bytes.Contains(data, []byte(strataAuthKey)) {
		t.Fatalf("preview carries the key (marshal error %v)", err)
	}
	applyLlamaServing(t, f.llamaServingFixture, p)
	return f, keyLog
}

// A server that prints its key leaves only [redacted] in PAIR's engine logs.
func TestStrataAuthKeyNeverReachesEngineLogs(t *testing.T) {
	f, _ := prepareKeyedStrata(t, "print")
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.ex.Stop("fake") })
	deadline := time.Now().Add(5 * time.Second)
	for {
		logs, err := f.ex.Logs("fake")
		if err != nil {
			t.Fatal(err)
		}
		encoded, _ := json.Marshal(logs)
		if bytes.Contains(encoded, []byte(strataAuthKey)) {
			t.Fatal("engine logs carry the key")
		}
		if bytes.Count(encoded, []byte("server key [redacted]")) >= 2 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("the printed key lines never reached the logs: %s", encoded)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// A server that prints its key and exits fails the start without the key in
// the error or the logs.
func TestStrataAuthKeyNeverReachesStartupFailure(t *testing.T) {
	f, _ := prepareKeyedStrata(t, "fail")
	err := f.ex.Start(context.Background(), "fake")
	t.Cleanup(func() { _ = f.ex.Stop("fake") })
	if err == nil {
		t.Fatal("a server that exits at once started")
	}
	if strings.Contains(err.Error(), strataAuthKey) {
		t.Fatal("start error carries the key")
	}
	logs, _ := f.ex.Logs("fake")
	if encoded, _ := json.Marshal(logs); bytes.Contains(encoded, []byte(strataAuthKey)) {
		t.Fatal("engine logs carry the key after a failed start")
	}
}

func TestStrataAuthKeyReachesOnlyTheServerEnvironment(t *testing.T) {
	f, keyLog := startKeyedStrata(t)
	if got, err := os.ReadFile(keyLog); err != nil || string(got) != strataAuthKey {
		t.Fatalf("server environment key = %q, %v", got, err)
	}
	data, err := os.ReadFile(f.startLog)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(data, []byte(strataAuthKey)) {
		t.Fatal("key appears in the server command line")
	}
	var args []string
	if err := json.Unmarshal(bytes.TrimSpace(data), &args); err != nil {
		t.Fatal(err)
	}
	i := slices.Index(args, "--config")
	snapshot, err := os.ReadFile(args[i+1])
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(snapshot, []byte(strataAuthKey)) || bytes.Contains(snapshot, []byte("api_key")) {
		t.Fatal("snapshot carries the key")
	}
	readback, err := f.ex.LaunchSettings("fake")
	if err != nil {
		t.Fatal(err)
	}
	if encoded, _ := json.Marshal(readback); bytes.Contains(encoded, []byte(strataAuthKey)) {
		t.Fatal("launch readback carries the key")
	}
	if logs, _ := f.ex.Logs("fake"); logs != nil {
		if encoded, _ := json.Marshal(logs); bytes.Contains(encoded, []byte(strataAuthKey)) {
			t.Fatal("engine logs carry the key")
		}
	}
	st := settingsState(t, f.ex)
	st.mu.Lock()
	port := st.port
	st.mu.Unlock()
	credential, err := f.ex.LocalCredential("fake")
	if err != nil || credential.Credential != strataAuthKey || credential.Port != port || port == 0 {
		t.Fatalf("local credential = %+v, %v (port %d)", credential, err, port)
	}
	if err := f.ex.Stop("fake"); err != nil {
		t.Fatal(err)
	}
	if credential, err := f.ex.LocalCredential("fake"); err != nil || credential.Credential != "" {
		t.Fatalf("stopped engine still offers a key: %+v, %v", credential.Port, err)
	}
}

func TestStrataAuthUnkeyedLaunchOffersNoCredential(t *testing.T) {
	f := newStrataServingFixture(t)
	p := approveLlamaServing(t, f.llamaServingFixture)
	applyLlamaServing(t, f.llamaServingFixture, p)
	if err := f.ex.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.ex.Stop("fake") })
	if credential, err := f.ex.LocalCredential("fake"); err != nil || credential != (engines.LocalCredential{}) {
		t.Fatalf("unkeyed launch credential = %+v, %v", credential, err)
	}
}

// PAIR's own requests to the keyed engine carry the key, never follow a
// redirect with it and never repeat it in an error.
func TestStrataAuthEngineRequestsCarryTheKey(t *testing.T) {
	f, _ := startKeyedStrata(t)
	var landed atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorized := r.Header.Get("Authorization") == "Bearer "+strataAuthKey
		switch r.URL.Path {
		case "/v1/models":
			if !authorized {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			_, _ = w.Write([]byte(`{"object":"list","data":[]}`))
		case "/redirect":
			http.Redirect(w, r, "/landing", http.StatusFound)
		case "/landing":
			landed.Add(1)
		case "/echo":
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"wrong key ` + strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ") + `"}`))
		}
	}))
	t.Cleanup(server.Close)
	st := settingsState(t, f.ex)
	st.mu.Lock()
	enginePort := st.port
	st.port = server.Listener.Addr().(*net.TCPAddr).Port
	st.mu.Unlock()
	t.Cleanup(func() {
		st.mu.Lock()
		st.port = enginePort
		st.mu.Unlock()
	})
	call := func(path string) (json.RawMessage, error) {
		return f.ex.dispatchAction(context.Background(), st, "fake", "probe", Action{HTTP: &ActionHTTP{Method: http.MethodGet, Path: path}}, nil)
	}
	if _, err := call("/v1/models"); err != nil {
		t.Fatalf("keyed request rejected: %v", err)
	}
	var httpErr *ActionHTTPError
	if _, err := call("/redirect"); !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusFound || landed.Load() != 0 {
		t.Fatalf("redirect: error %v, landing hits %d", err, landed.Load())
	}
	_, err := call("/echo")
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusUnauthorized || strings.Contains(httpErr.Body, strataAuthKey) || !strings.Contains(httpErr.Body, "[redacted]") {
		t.Fatalf("echoed key not redacted: %v", err)
	}
	if strings.Contains(err.Error(), strataAuthKey) {
		t.Fatal("error text carries the key")
	}
}

func TestStrataAuthKeyShape(t *testing.T) {
	for _, tc := range []struct {
		name, key, expect string
	}{
		{"generated form", "Ab-_cd12EF34gh56IJ78kl90", ""},
		{"too short", "short-key", "is not a reviewed value"},
		{"empty", "", "is not a reviewed value"},
		{"list of keys", "FAKE-AUTH-KEY-one,FAKE-AUTH-KEY-two", "is not a reviewed value"},
		{"space", "FAKE AUTH KEY 0123456789", "is not a reviewed value"},
		{"control character", "FAKE-AUTH-KEY-0123456789\t", "is not a reviewed value"},
		{"non-ASCII", "FAKE-AUTH-KEY-0123456789é", "is not a reviewed value"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			base := forkBaselineRaw(t, f)
			base["api_key"] = tc.key
			err := f.deriveRaw(t, base)
			requireForkOutcome(t, err, tc.expect)
			if err != nil && tc.key != "" && strings.Contains(err.Error(), tc.key) {
				t.Fatal("rejection repeats the key")
			}
		})
	}
	t.Run("wrong type", func(t *testing.T) {
		if _, err := parseStrataServingConfig([]byte(`{"api_key":["FAKE-AUTH-KEY-0123456789"]}`)); err == nil {
			t.Fatal("key list accepted")
		}
	})
}
