// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
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
	keyFile := writeStrataKeyFile(t, f, strataAuthKey+"\n")
	f.config.APIKeyFile = &keyFile
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

// writeStrataKeyFile writes the key file beside the fixture's config.
func writeStrataKeyFile(t *testing.T, f strataServingFixture, content string) string {
	t.Helper()
	path := filepath.Join(f.root, "strata.key")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// keyedRaw is the accepted baseline config naming the key file.
func keyedRaw(t *testing.T, f strataServingFixture, keyFile string) map[string]any {
	t.Helper()
	raw := forkBaselineRaw(t, f)
	raw["api_key_file"] = keyFile
	return raw
}

func TestStrataAuthInlineKeyIsRejected(t *testing.T) {
	f := newStrataServingFixture(t)
	raw := forkBaselineRaw(t, f)
	raw["api_key"] = strataAuthKey
	err := f.deriveRaw(t, raw)
	requireForkOutcome(t, err, "Unsupported or malformed prepared Strata config.")
	if strings.Contains(err.Error(), strataAuthKey) {
		t.Fatal("rejection repeats the key")
	}
}

func TestStrataAuthKeyFileRejections(t *testing.T) {
	for _, tc := range []struct {
		name   string
		setup  func(t *testing.T, f strataServingFixture) string
		expect string
	}{
		{"missing file", func(t *testing.T, f strataServingFixture) string { return filepath.Join(f.root, "absent.key") }, "api_key_file is missing"},
		{"empty file", func(t *testing.T, f strataServingFixture) string { return writeStrataKeyFile(t, f, "") }, "api_key_file is missing"},
		{"directory", func(t *testing.T, f strataServingFixture) string { return filepath.Join(f.root, "lib") }, "api_key_file is missing"},
		{"oversized file", func(t *testing.T, f strataServingFixture) string {
			return writeStrataKeyFile(t, f, strings.Repeat("k", strataKeyFileLimit+1))
		}, "api_key_file is missing"},
		{"relative path", func(t *testing.T, f strataServingFixture) string { return "strata.key" }, "absolute clean path"},
		{"declared artifact", func(t *testing.T, f strataServingFixture) string { return filepath.Join(f.root, "server.py") }, "declared artifact"},
		{"too short", func(t *testing.T, f strataServingFixture) string { return writeStrataKeyFile(t, f, "short-key") }, "one reviewed key"},
		{"list of keys", func(t *testing.T, f strataServingFixture) string {
			return writeStrataKeyFile(t, f, "FAKE-AUTH-KEY-one-0001,FAKE-AUTH-KEY-two-0002")
		}, "one reviewed key"},
		{"space", func(t *testing.T, f strataServingFixture) string {
			return writeStrataKeyFile(t, f, "FAKE AUTH KEY 0123456789")
		}, "one reviewed key"},
		{"two lines", func(t *testing.T, f strataServingFixture) string {
			return writeStrataKeyFile(t, f, strataAuthKey+"\n"+strataAuthKey+"\n")
		}, "one reviewed key"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			err := f.deriveRaw(t, keyedRaw(t, f, tc.setup(t, f)))
			requireForkOutcome(t, err, tc.expect)
			for _, secret := range []string{strataAuthKey, "FAKE-AUTH-KEY-one", "FAKE AUTH KEY", "short-key"} {
				if strings.Contains(err.Error(), secret) {
					t.Fatal("rejection repeats the file content")
				}
			}
		})
	}
	t.Run("accepted with or without a line end", func(t *testing.T) {
		for _, content := range []string{strataAuthKey, strataAuthKey + "\n", strataAuthKey + "\r\n"} {
			f := newStrataServingFixture(t)
			launch, err := f.deriveLaunch(t, keyedRaw(t, f, writeStrataKeyFile(t, f, content)))
			if err != nil || launch.Credential != strataAuthKey {
				t.Fatalf("content %q: credential ok %v, error %v", content, launch != nil && launch.Credential == strataAuthKey, err)
			}
			if bytes.Contains(launch.Snapshot, []byte("api_key")) {
				t.Fatal("snapshot names the key file")
			}
		}
	})
}

// Nothing advertised derives from the key's content: rewriting the file in
// place with another key of the same length and modification time leaves
// every advertised field identical. That is the documented residual: such a
// rewrite is also not detected.
func TestStrataAuthAdvertisedFieldsIgnoreKeyContent(t *testing.T) {
	f := newStrataServingFixture(t)
	keyFile := writeStrataKeyFile(t, f, "aaaaaaaaaaaaaaaa")
	info, err := os.Stat(keyFile)
	if err != nil {
		t.Fatal(err)
	}
	// Write the config once: rewriting it would change its own binding.
	if _, err := f.deriveLaunch(t, keyedRaw(t, f, keyFile)); err != nil {
		t.Fatal(err)
	}
	advertised := func(t *testing.T) []byte {
		t.Helper()
		st := settingsState(t, f.ex)
		command, err := launchForState(st, f.request.Settings.ServerPort)
		if err != nil {
			t.Fatal(err)
		}
		rt := st.plat.Runtime
		rt.Serving = f.request.Settings.Serving
		launch, err := deriveStrataServing(command, rt)
		if err != nil {
			t.Fatal(err)
		}
		data, err := json.Marshal(struct {
			Config    *engines.ServingConfig
			Envelopes []engines.ServingEnvelope
			Digest    string
			Snapshot  []byte
		}{launch.Config, launch.Envelopes, launch.Digest, launch.Snapshot})
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	weak := advertised(t)
	if err := os.WriteFile(keyFile, []byte("bbbbbbbbbbbbbbbb"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(keyFile, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if other := advertised(t); !bytes.Equal(weak, other) {
		t.Fatal("an advertised field depends on the key content")
	}
	for _, guess := range []string{"aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"} {
		for _, form := range []string{guess, guess + "\n"} {
			sum := sha256.Sum256([]byte(form))
			if bytes.Contains(weak, []byte(form)) || bytes.Contains(weak, []byte(hex.EncodeToString(sum[:]))) {
				t.Fatal("an advertised field carries the key or its hash")
			}
		}
	}
}

// Replacing the key file in the ordinary way changes the approval binding:
// the approved digest no longer compiles, so the key change takes a fresh
// approval and a restart, while the old key keeps serving until then.
func TestStrataAuthKeyChangeRequiresApprovalAndRestart(t *testing.T) {
	f, _ := startKeyedStrata(t)
	st := settingsState(t, f.ex)
	command, err := launchForState(st, f.request.Settings.ServerPort)
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	approved := st.plat.Runtime.Serving.Clone()
	running := st.plat.Runtime.servingLaunch
	st.mu.Unlock()
	rt := st.plat.Runtime
	rt.Serving = approved
	if _, err := compileStrataServing(command, rt, true); err != nil {
		t.Fatalf("unchanged key no longer compiles: %v", err)
	}
	keyFile := filepath.Join(f.root, "strata.key")
	time.Sleep(20 * time.Millisecond)
	if err := os.Remove(keyFile); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keyFile, []byte("FAKE-AUTH-KEY-replacement-0001\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := compileStrataServing(command, rt, true); err == nil || !strings.Contains(err.Error(), "fresh preview") {
		t.Fatalf("a changed key compiled against the old approval: %v", err)
	}
	if err := recheckPreparedServing(st, command, rt, running); err == nil {
		t.Fatal("a changed key passed the pre-spawn recheck")
	}
	if credential, _ := f.ex.LocalCredential("fake"); credential.Credential != strataAuthKey {
		t.Fatal("the running launch lost the key it was started with")
	}
}
