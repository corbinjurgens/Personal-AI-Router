// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
)

const (
	strataContractTimeout       = 10 * time.Second
	strataContractStartup       = 15 * time.Second
	strataContractLimit         = 64 << 10
	strataContractRuntimeArg    = "--strata-contract-runtime"
	strataContractRuntimeEnv    = "NVPAIR_STRATA_CONTRACT_EXECUTABLE"
	strataContractControlPath   = "/__strata_contract/control"
	strataContractLoadedModel   = "Org/Strata-Loaded:Q4_K_M"
	strataContractUnloadedModel = "Org/Strata-Unloaded:Q5_K_M"
	strataContractLoadingModel  = "Org/Strata-Loading:Q6_K"
	strataContractMissingModel  = "Org/Strata-NoStatus:Q8_0"
)

// This dispatch precedes TestMain: the manager reuses the wrapper-owned test
// executable, without rebuilding products, parsing test flags, or selecting tests.
// An environment variable alone can never select the fake runtime.
func init() {
	if len(os.Args) < 2 || os.Args[1] != strataContractRuntimeArg {
		return
	}
	executable, err := os.Executable()
	if err != nil || len(os.Args) != 3 || os.Getenv(strataContractRuntimeEnv) != executable || runtime.GOOS == "darwin" {
		os.Exit(2)
	}
	if err := strataContractRunRuntime(os.Args[2]); err != nil {
		fmt.Fprintf(os.Stderr, "strata contract runtime: %v\n", err)
		os.Exit(2)
	}
	os.Exit(0)
}

type strataContractRuntimeConfig struct {
	Host     string `json:"host"`
	Port     int    `json:"port"`
	Identity string `json:"identity"`
}

type strataContractRequest struct {
	Method string `json:"method"`
	Path   string `json:"path"`
	Model  string `json:"model"`
	Count  int    `json:"count"`
}

type strataContractControl struct {
	Identity       string                  `json:"identity"`
	Requests       []strataContractRequest `json:"requests"`
	Models         map[string]string       `json:"models"`
	NextLoadStatus int                     `json:"nextLoadStatus"`
}

type strataContractRuntime struct {
	mu             sync.Mutex
	config         strataContractRuntimeConfig
	models         map[string]string
	requests       []strataContractRequest
	nextLoadStatus int
}

func strataContractSafePort(port int) bool {
	return port > 0 && port <= 65535 && port != 8080 && port != 11434 && port != 1234 && (port < 14318 || port > 14323)
}

func strataContractRunRuntime(path string) error {
	if !filepath.IsAbs(path) {
		return errors.New("runtime configuration must be absolute")
	}
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("read private runtime configuration: %w", err)
	}
	defer file.Close()
	var config strataContractRuntimeConfig
	if err := strataContractReadJSON(file, &config); err != nil {
		return fmt.Errorf("decode runtime configuration: %w", err)
	}
	if config.Host != "127.0.0.1" || !strataContractSafePort(config.Port) || len(config.Identity) != 32 {
		return errors.New("invalid private runtime configuration")
	}
	if _, err := hex.DecodeString(config.Identity); err != nil {
		return errors.New("invalid fixture identity")
	}
	listener, err := net.Listen("tcp4", net.JoinHostPort(config.Host, strconv.Itoa(config.Port)))
	if err != nil {
		failure, encodeErr := json.Marshal(struct {
			BindCollision bool `json:"bindCollision"`
		}{strataContractAddressInUse(err)})
		if encodeErr != nil {
			return encodeErr
		}
		if writeErr := os.WriteFile(filepath.Join(filepath.Dir(path), "start-error.json"), failure, 0o600); writeErr != nil {
			return writeErr
		}
		return fmt.Errorf("bind private runtime port: %w", err)
	}
	defer listener.Close()
	pid, err := json.Marshal(struct {
		PID      int    `json:"pid"`
		Identity string `json:"identity"`
	}{os.Getpid(), config.Identity})
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(filepath.Dir(path), "runtime-pid.json"), pid, 0o600); err != nil {
		return err
	}
	fake := &strataContractRuntime{config: config, models: map[string]string{
		strataContractLoadedModel: "loaded", strataContractUnloadedModel: "unloaded",
		strataContractLoadingModel: "loading", strataContractMissingModel: "",
	}}
	server := &http.Server{Handler: fake, ReadHeaderTimeout: strataContractTimeout, ReadTimeout: strataContractTimeout, WriteTimeout: strataContractTimeout, MaxHeaderBytes: strataContractLimit}
	return server.Serve(listener)
}

func (f *strataContractRuntime) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	if r.URL.Path == strataContractControlPath {
		if r.Method == http.MethodPost {
			var control struct {
				NextLoadStatus int `json:"nextLoadStatus"`
			}
			if err := strataContractReadJSON(r.Body, &control); err != nil {
				http.Error(w, "invalid control body", http.StatusBadRequest)
				return
			}
			if control.NextLoadStatus < 400 || control.NextLoadStatus > 599 {
				http.Error(w, "expected an HTTP error status", http.StatusBadRequest)
				return
			}
			f.nextLoadStatus = control.NextLoadStatus
		} else if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		strataContractRuntimeJSON(w, strataContractControl{f.config.Identity, f.requests, f.models, f.nextLoadStatus})
		return
	}
	var request struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if r.Method != http.MethodGet {
		if err := strataContractReadJSON(r.Body, &request); err != nil {
			http.Error(w, "invalid request body", http.StatusBadRequest)
			return
		}
	}
	index := slices.IndexFunc(f.requests, func(seen strataContractRequest) bool {
		return seen.Method == r.Method && seen.Path == r.URL.Path && seen.Model == request.Model
	})
	if index >= 0 {
		f.requests[index].Count++
	} else {
		if len(f.requests) >= 128 {
			http.Error(w, "fixture counter limit", http.StatusTooManyRequests)
			return
		}
		f.requests = append(f.requests, strataContractRequest{r.Method, r.URL.Path, request.Model, 1})
	}
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/health":
		strataContractRuntimeJSON(w, map[string]any{"service": "strata", "fixture": f.config.Identity, "pid": os.Getpid()})
	case r.Method == http.MethodGet && r.URL.Path == "/v1/models":
		rows := make([]map[string]any, 0, len(f.models))
		for _, model := range []string{strataContractLoadedModel, strataContractUnloadedModel, strataContractLoadingModel, strataContractMissingModel} {
			row := map[string]any{"id": model, "object": "model", "digest": strings.Repeat("a", 64)}
			if status := f.models[model]; status != "" {
				row["status"] = map[string]string{"value": status}
			}
			rows = append(rows, row)
		}
		strataContractRuntimeJSON(w, map[string]any{"object": "list", "data": rows})
	case r.Method == http.MethodPost && (r.URL.Path == "/v1/load" || r.URL.Path == "/v1/unload"):
		if _, ok := f.models[request.Model]; !ok {
			w.WriteHeader(http.StatusNotFound)
			strataContractRuntimeJSON(w, map[string]string{"error": "unknown exact model"})
			return
		}
		if r.URL.Path == "/v1/load" && f.nextLoadStatus != 0 {
			status := f.nextLoadStatus
			f.nextLoadStatus = 0
			w.WriteHeader(status)
			strataContractRuntimeJSON(w, map[string]string{"error": "fixture load refused"})
			return
		}
		status := "unloaded"
		if r.URL.Path == "/v1/load" {
			status = "loaded"
		}
		f.models[request.Model] = status
		strataContractRuntimeJSON(w, map[string]any{"model": request.Model, "status": status, "ok": true})
	case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
		if _, ok := f.models[request.Model]; !ok {
			w.WriteHeader(http.StatusNotFound)
			strataContractRuntimeJSON(w, map[string]string{"error": "unknown exact model"})
			return
		}
		if request.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, chunk := range []string{
				`data: {"id":"strata-contract","object":"chat.completion.chunk","model":` + strconv.Quote(request.Model) + `,"choices":[{"index":0,"delta":{"role":"assistant","content":"fixture reply"},"finish_reason":null}]}` + "\n\n",
				`data: {"id":"strata-contract","object":"chat.completion.chunk","model":` + strconv.Quote(request.Model) + `,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
				"data: [DONE]\n\n",
			} {
				if _, err := io.WriteString(w, chunk); err != nil {
					return
				}
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
			}
			return
		}
		strataContractRuntimeJSON(w, map[string]any{"id": "strata-contract", "object": "chat.completion", "model": request.Model, "choices": []any{map[string]any{"index": 0, "message": map[string]string{"role": "assistant", "content": "fixture reply"}, "finish_reason": "stop"}}})
	default:
		w.WriteHeader(http.StatusNotFound)
		strataContractRuntimeJSON(w, map[string]string{"error": "fixture unsupported path", "path": r.URL.Path})
	}
}

func strataContractRuntimeJSON(w http.ResponseWriter, value any) {
	if err := json.NewEncoder(w).Encode(value); err != nil {
		fmt.Fprintln(os.Stderr, "strata contract runtime: response write failed")
	}
}

func strataContractReadJSON(reader io.Reader, value any) error {
	data, err := io.ReadAll(io.LimitReader(reader, strataContractLimit+1))
	if err != nil {
		return err
	}
	if len(data) > strataContractLimit {
		return errors.New("fixture JSON exceeds limit")
	}
	return json.Unmarshal(data, value)
}

func strataContractPlatform(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "darwin" {
		t.Skip("Strata cross-process acceptance is reserved for Windows PC3; no Mac processes")
	}
	if runtime.GOOS == "windows" {
		if !strings.Contains(os.Getenv("GOFLAGS"), "nvpair-test-exec") || !filepath.IsAbs(os.Getenv("NVPAIR_TEST_BIN_DIR")) {
			t.Fatal("Windows Strata fixture requires the configured fixed-path wrapper and NVPAIR_TEST_BIN_DIR")
		}
	}
}

func strataContractListen(t *testing.T) net.Listener {
	t.Helper()
	for attempt := 0; attempt < 8; attempt++ {
		listener, err := net.Listen("tcp4", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("allocate private loopback listener: %v", err)
		}
		_, text, err := net.SplitHostPort(listener.Addr().String())
		if err != nil {
			listener.Close()
			t.Fatal(err)
		}
		port, err := strconv.Atoi(text)
		if err != nil {
			listener.Close()
			t.Fatal(err)
		}
		if strataContractSafePort(port) {
			return listener
		}
		if err := listener.Close(); err != nil {
			t.Fatal(err)
		}
	}
	t.Fatal("no nonprotected loopback port allocated")
	return nil
}

func strataContractPort(t *testing.T) int {
	t.Helper()
	listener := strataContractListen(t)
	_, text, err := net.SplitHostPort(listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(text)
	if err != nil {
		t.Fatal(err)
	}
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return port
}

func strataContractEnvironment(root string) (string, []string) {
	base := root
	if runtime.GOOS == "darwin" {
		base = filepath.Join(root, "Library", "Application Support")
	}
	app := filepath.Join(base, "Nvidia Corporation", "Personal AI Router")
	overrides := map[string]string{"HOME": root, "XDG_CONFIG_HOME": root, "APPDATA": root, "LOCALAPPDATA": root, "USERPROFILE": root}
	env := make([]string, 0, len(os.Environ())+len(overrides))
	for _, item := range os.Environ() {
		key, _, _ := strings.Cut(item, "=")
		if _, replaced := overrides[strings.ToUpper(key)]; replaced || strings.EqualFold(key, strataContractRuntimeEnv) {
			continue
		}
		env = append(env, item)
	}
	for key, value := range overrides {
		env = append(env, key+"="+value)
	}
	return app, env
}

func strataContractWriteJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write private fixture JSON %s: %v", path, err)
	}
}

func strataContractDecode(t *testing.T, data json.RawMessage, value any) {
	t.Helper()
	if err := json.Unmarshal(data, value); err != nil {
		t.Fatalf("decode fixture RPC value: %v", err)
	}
}

type strataContractOutput struct {
	mu   sync.Mutex
	data []byte
}

func (o *strataContractOutput) Write(data []byte) (int, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	n := len(data)
	if remaining := strataContractLimit - len(o.data); remaining > 0 {
		o.data = append(o.data, data[:min(remaining, len(data))]...)
	}
	return n, nil
}
func (o *strataContractOutput) strataContractString() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return string(o.data)
}

type strataContractFrame struct {
	message jsonrpc.Message
	err     error
}
type strataContractRPC struct {
	cmd        *exec.Cmd
	stdin      io.WriteCloser
	frames     chan strataContractFrame
	readerDone chan struct{}
	readerExit chan struct{}
	done       chan struct{}
	exitErr    error
	stderr     strataContractOutput
	pending    []jsonrpc.Message
	nextID     int
	closed     bool
}

func strataContractStartChild(t *testing.T, executable string, env []string, args ...string) *strataContractRPC {
	t.Helper()
	c := &strataContractRPC{cmd: exec.Command(executable, args...), frames: make(chan strataContractFrame, 128), readerDone: make(chan struct{}), readerExit: make(chan struct{}), done: make(chan struct{})}
	c.cmd.Env = env
	c.cmd.Stderr = &c.stderr
	var err error
	c.stdin, err = c.cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := c.cmd.StdoutPipe()
	if err != nil {
		c.stdin.Close()
		t.Fatal(err)
	}
	if err := c.cmd.Start(); err != nil {
		c.stdin.Close()
		stdout.Close()
		t.Fatalf("start fixture child: %v", err)
	}
	t.Cleanup(func() { c.strataContractClose(t) })
	go func() {
		defer close(c.readerExit)
		defer close(c.frames)
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 4096), strataContractLimit)
		for scanner.Scan() {
			var message jsonrpc.Message
			err := json.Unmarshal(scanner.Bytes(), &message)
			if err == nil && (message.JSONRPC != "2.0" || (!message.IsNotification() && !message.IsResponse() && !message.IsRequest())) {
				err = errors.New("invalid JSON-RPC envelope")
			}
			select {
			case c.frames <- strataContractFrame{message, err}:
			case <-c.readerDone:
				return
			}
			if err != nil {
				return
			}
		}
		err := scanner.Err()
		if err == nil {
			err = io.EOF
		}
		select {
		case c.frames <- strataContractFrame{err: err}:
		case <-c.readerDone:
		}
	}()
	go func() { <-c.readerExit; c.exitErr = c.cmd.Wait(); close(c.done) }()
	return c
}

func (c *strataContractRPC) strataContractSend(ctx context.Context, method string, id int, params any) error {
	frame := map[string]any{"jsonrpc": "2.0", "method": method, "params": params}
	if id != 0 {
		frame["id"] = id
	}
	data, err := json.Marshal(frame)
	if err != nil {
		return err
	}
	if len(data) > strataContractLimit {
		return errors.New("outbound fixture RPC exceeds limit")
	}
	data = append(data, '\n')
	written := make(chan error, 1)
	go func() {
		n, err := c.stdin.Write(data)
		if err == nil && n != len(data) {
			err = io.ErrShortWrite
		}
		written <- err
	}()
	select {
	case err := <-written:
		return err
	case <-ctx.Done():
		// Closing this child's pipe also unblocks a stalled writer.
		c.stdin.Close()
		return ctx.Err()
	}
}

func (c *strataContractRPC) strataContractWait(ctx context.Context, match func(jsonrpc.Message) bool) (jsonrpc.Message, error) {
	for i, message := range c.pending {
		if match(message) {
			c.pending = slices.Delete(c.pending, i, i+1)
			return message, nil
		}
	}
	for {
		select {
		case frame, ok := <-c.frames:
			if !ok {
				return jsonrpc.Message{}, fmt.Errorf("child RPC stream closed; stderr: %s", c.stderr.strataContractString())
			}
			if frame.err != nil {
				return jsonrpc.Message{}, fmt.Errorf("child RPC reader: %w; stderr: %s", frame.err, c.stderr.strataContractString())
			}
			if match(frame.message) {
				return frame.message, nil
			}
			if len(c.pending) >= 512 {
				return jsonrpc.Message{}, errors.New("fixture RPC pending-frame limit reached")
			}
			c.pending = append(c.pending, frame.message)
		case <-ctx.Done():
			return jsonrpc.Message{}, fmt.Errorf("fixture RPC wait: %w; stderr: %s", ctx.Err(), c.stderr.strataContractString())
		}
	}
}

func (c *strataContractRPC) strataContractCallWithin(method string, params any, timeout time.Duration) (jsonrpc.Message, error) {
	c.nextID++
	id := c.nextID
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	if err := c.strataContractSend(ctx, method, id, params); err != nil {
		return jsonrpc.Message{}, fmt.Errorf("child RPC send: %w; stderr: %s", err, c.stderr.strataContractString())
	}
	return c.strataContractWait(ctx, func(message jsonrpc.Message) bool {
		return message.IsResponse() && string(*message.ID) == strconv.Itoa(id)
	})
}

func (c *strataContractRPC) strataContractCall(t *testing.T, method string, params any) jsonrpc.Message {
	t.Helper()
	message, err := c.strataContractCallWithin(method, params, strataContractTimeout)
	if err != nil {
		t.Fatalf("%s: %v", method, err)
	}
	return message
}

func (c *strataContractRPC) strataContractOK(t *testing.T, method string, params any, result any) jsonrpc.Message {
	t.Helper()
	message := c.strataContractCall(t, method, params)
	if message.Error != nil {
		t.Fatalf("%s: RPC %d: %s", method, message.Error.Code, message.Error.Message)
	}
	if result != nil {
		strataContractDecode(t, message.Result, result)
	}
	return message
}

func (c *strataContractRPC) strataContractNotify(t *testing.T, method string, params any) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), strataContractTimeout)
	defer cancel()
	if err := c.strataContractSend(ctx, method, 0, params); err != nil {
		t.Fatalf("notify %s: %v", method, err)
	}
}

func (c *strataContractRPC) strataContractNotification(t *testing.T, method string, match func(json.RawMessage) bool) jsonrpc.Message {
	t.Helper()
	return c.strataContractNotificationWithin(t, method, match, strataContractTimeout)
}

func (c *strataContractRPC) strataContractNotificationWithin(t *testing.T, method string, match func(json.RawMessage) bool, timeout time.Duration) jsonrpc.Message {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	message, err := c.strataContractWait(ctx, func(message jsonrpc.Message) bool {
		return message.IsNotification() && message.Method == method && (match == nil || match(message.Params))
	})
	if err != nil {
		t.Fatalf("notification %s: %v", method, err)
	}
	return message
}

// Call first, then select the terminal notification by payload (op/node/engine).
// The same reader retains notifications arriving before the matching response.
func (c *strataContractRPC) strataContractCallAndNotification(t *testing.T, method string, params any, notification string, match func(json.RawMessage) bool) (jsonrpc.Message, jsonrpc.Message) {
	t.Helper()
	response := c.strataContractCall(t, method, params)
	return response, c.strataContractNotification(t, notification, match)
}

func (c *strataContractRPC) strataContractClose(t *testing.T) {
	t.Helper()
	if c.closed {
		return
	}
	c.closed = true
	close(c.readerDone)
	if err := c.stdin.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
		t.Errorf("close fixture stdin: %v", err)
	}
	select {
	case <-c.done:
	case <-time.After(strataContractTimeout):
		if err := c.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			t.Errorf("kill owned fixture child: %v", err)
		}
		select {
		case <-c.done:
		case <-time.After(strataContractTimeout):
			t.Error("owned fixture child did not reap")
		}
		t.Errorf("fixture child exceeded shutdown deadline; stderr: %s", c.stderr.strataContractString())
	}
	select {
	case <-c.done:
		if c.exitErr != nil {
			t.Errorf("fixture child exit: %v; stderr: %s", c.exitErr, c.stderr.strataContractString())
		}
	default:
	}
}

type strataContractFixture struct {
	Root, AppDir, ClusterDir string
	Env                      []string
	Port, ControlPort        int
	Identity                 string
	Manager                  *strataContractRPC
	runtimeProcess           *os.Process
	runtimeStarted           bool
}

func strataContractNew(t *testing.T, installed bool) *strataContractFixture {
	t.Helper()
	strataContractPlatform(t)
	f := &strataContractFixture{Root: t.TempDir(), Port: strataContractPort(t)}
	f.AppDir, f.Env = strataContractEnvironment(f.Root)
	f.ClusterDir = filepath.Join(f.AppDir, "cluster")
	for _, directory := range []string{f.ClusterDir, filepath.Join(f.AppDir, "engines")} {
		if err := os.MkdirAll(directory, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	identity := make([]byte, 16)
	if _, err := rand.Read(identity); err != nil {
		t.Fatal(err)
	}
	f.Identity = hex.EncodeToString(identity)
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join("..", "nvpair-engine-manager", "testdata", "strata", "strata.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest map[string]json.RawMessage
	strataContractDecode(t, data, &manifest)
	var rt map[string]json.RawMessage
	strataContractDecode(t, manifest["runtime"], &rt)
	// The fake executable reads host/port from runtime.json and accepts only
	// the dispatch marker and config path. It has no editable Python CLI.
	for key, value := range map[string]any{
		"bin": executable, "args": []string{strataContractRuntimeArg, filepath.Join(f.Root, "runtime.json")},
		"editable_launch": nil,
		"env":             map[string]string{strataContractRuntimeEnv: executable}, "port": f.Port,
		"ready": map[string]any{"http": "http://127.0.0.1:{port}/health", "status": 200, "timeout_s": 8, "json_match": map[string]string{"field": "service", "value": "strata"}},
		"stop":  map[string]any{"signal": "term", "grace_s": 1},
	} {
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		rt[key] = encoded
	}
	manifest["runtime"], err = json.Marshal(rt)
	if err != nil {
		t.Fatal(err)
	}
	var platforms map[string]json.RawMessage
	strataContractDecode(t, manifest["platforms"], &platforms)
	detect := executable
	if !installed {
		detect = filepath.Join(f.Root, "absent-runtime")
	}
	for platform := range platforms {
		platforms[platform], err = json.Marshal(map[string]any{"detect": []string{detect}})
		if err != nil {
			t.Fatal(err)
		}
	}
	manifest["platforms"], err = json.Marshal(platforms)
	if err != nil {
		t.Fatal(err)
	}
	strataContractWriteJSON(t, filepath.Join(f.AppDir, "engines", "strata.json"), manifest)
	f.strataContractWriteRuntime(t)
	for _, engine := range engines.All() {
		override := map[string]any{"engine": engine.Name, "platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{
			"detect":  []string{filepath.Join(f.Root, "absent-"+engine.Name)},
			"runtime": map[string]any{"port": 0, "ready": nil, "health": nil},
		}}}
		if engine.Name == "lmstudio" {
			// Its dormant delete action requires readiness when restart_after is
			// true. Disable that flag so the null-probe override remains valid.
			override["actions"] = map[string]any{"delete_model": map[string]bool{"restart_after": false}}
		}
		strataContractWriteJSON(t, filepath.Join(f.AppDir, "engines", engine.Name+".json"), override)
	}
	return f
}

func (f *strataContractFixture) strataContractWriteRuntime(t *testing.T) {
	t.Helper()
	if !strataContractSafePort(f.Port) {
		t.Fatalf("unsafe runtime port %d", f.Port)
	}
	strataContractWriteJSON(t, filepath.Join(f.Root, "runtime.json"), strataContractRuntimeConfig{"127.0.0.1", f.Port, f.Identity})
}

func (f *strataContractFixture) strataContractStartManager(t *testing.T) {
	t.Helper()
	if f.Manager != nil {
		t.Fatal("fixture manager already started")
	}
	if f.ControlPort != 0 && !strataContractSafePort(f.ControlPort) {
		t.Fatal("unsafe control port")
	}
	f.Manager = strataContractStartChild(t, engineMgrBin, f.Env, "--cluster-dir", f.ClusterDir, "--control-port", strconv.Itoa(f.ControlPort), "--http-port", "0", "--loaded-poll-interval", "1", "--log-level", "warn")
	t.Cleanup(func() { f.strataContractCleanupRuntime(t) })
	f.Manager.strataContractNotificationWithin(t, "engine:ready", nil, strataContractStartup)
	completeEngineManagerRecovery(t, func(method string, params any) jsonrpc.Message {
		response, err := f.Manager.strataContractCallWithin(method, params, strataContractStartup)
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		return response
	})
	// Check the merged host block before any explicit inventory/status request.
	for _, engine := range engines.All() {
		var manifest struct {
			Platforms map[string]struct {
				Detect  []string `json:"detect"`
				Runtime struct {
					Port          int `json:"port"`
					Ready, Health json.RawMessage
				} `json:"runtime"`
			} `json:"platforms"`
		}
		f.Manager.strataContractOK(t, "engine:describe", map[string]string{"engine": engine.Name}, &manifest)
		platform, ok := manifest.Platforms[runtime.GOOS+"/"+runtime.GOARCH]
		if !ok || !reflect.DeepEqual(platform.Detect, []string{filepath.Join(f.Root, "absent-"+engine.Name)}) || platform.Runtime.Port != 0 || (len(platform.Runtime.Ready) != 0 && string(platform.Runtime.Ready) != "null") || (len(platform.Runtime.Health) != 0 && string(platform.Runtime.Health) != "null") {
			t.Fatalf("built-in %s host override did not disable detection and probes", engine.Name)
		}
	}
}

func (f *strataContractFixture) strataContractDescriptor(t *testing.T) engines.Descriptor {
	t.Helper()
	var result struct {
		Engines []engines.Descriptor `json:"engines"`
	}
	f.Manager.strataContractOK(t, "engine:descriptors", nil, &result)
	for _, descriptor := range result.Engines {
		if descriptor.Engine == "strata" {
			return descriptor
		}
	}
	t.Fatal("manager did not publish the private Strata descriptor")
	return engines.Descriptor{}
}

type strataContractStatus struct {
	Engine    string `json:"engine"`
	Installed bool   `json:"installed"`
	Running   bool   `json:"running"`
	Healthy   bool   `json:"healthy"`
	Port      int    `json:"port"`
}

type strataContractModels struct {
	Models         []string            `json:"models"`
	ModelsByEngine map[string][]string `json:"modelsByEngine"`
	LoadedByEngine map[string][]string `json:"loadedByEngine"`
}

type strataContractIdentity struct {
	Engines map[string]struct {
		EngineRunID  string            `json:"engineRunId"`
		LaunchDigest string            `json:"launchDigest"`
		ModelDigests map[string]string `json:"modelDigests"`
	} `json:"engines"`
}

func (f *strataContractFixture) strataContractStartRuntime(t *testing.T) {
	t.Helper()
	if f.runtimeStarted || f.runtimeProcess != nil {
		t.Fatal("fixture runtime already started")
	}
	for attempt := 0; attempt < 8; attempt++ {
		f.strataContractWriteRuntime(t)
		response, err := f.Manager.strataContractCallWithin("engine:start", map[string]any{"engine": "strata", "port": f.Port, "bind": "127.0.0.1"}, strataContractStartup)
		if err != nil {
			t.Fatal(err)
		}
		if response.Error == nil {
			f.strataContractRetainRuntime(t)
			f.strataContractWaitStatus(t, true)
			return
		}
		var failure struct {
			BindCollision bool `json:"bindCollision"`
		}
		data, readErr := os.ReadFile(filepath.Join(f.Root, "start-error.json"))
		if readErr != nil || json.Unmarshal(data, &failure) != nil || !failure.BindCollision {
			t.Fatalf("engine:start: RPC %d: %s", response.Error.Code, response.Error.Message)
		}
		if err := os.Remove(filepath.Join(f.Root, "start-error.json")); err != nil {
			t.Fatal(err)
		}
		f.Port = strataContractPort(t)
	}
	t.Fatal("runtime bind collisions exhausted bounded retry")
}

func (f *strataContractFixture) strataContractRetainRuntime(t *testing.T) {
	t.Helper()
	// A marker can outlive the process that wrote it. Failed starts remain the
	// manager's cleanup responsibility; only a live handshake grants ownership.
	data, err := os.ReadFile(filepath.Join(f.Root, "runtime-pid.json"))
	if err != nil {
		t.Fatalf("read private runtime ownership marker: %v", err)
	}
	var hint struct {
		PID      int    `json:"pid"`
		Identity string `json:"identity"`
	}
	strataContractDecode(t, data, &hint)
	if hint.Identity != f.Identity || hint.PID <= 0 {
		t.Fatal("private runtime ownership marker mismatch")
	}
	candidate, err := os.FindProcess(hint.PID)
	if err != nil {
		t.Fatalf("acquire candidate runtime handle: %v", err)
	}
	defer func() {
		if f.runtimeProcess != candidate {
			if err := candidate.Release(); err != nil {
				t.Errorf("release unverified runtime handle: %v", err)
			}
		}
	}()
	if err := f.strataContractRuntimeIdentity(candidate.Pid); err != nil {
		t.Fatalf("verify candidate runtime after handle acquisition: %v", err)
	}
	f.runtimeProcess = candidate
	f.runtimeStarted = true
}

func (f *strataContractFixture) strataContractRuntimeIdentity(pid int) error {
	status, _, body, err := strataContractRequestHTTP(http.MethodGet, f.Port, "/health", nil)
	if err != nil {
		return err
	}
	var health struct {
		Service, Fixture string
		PID              int
	}
	if err := json.Unmarshal(body, &health); err != nil {
		return errors.New("runtime health identity is not valid JSON")
	}
	if status != http.StatusOK || health.Service != "strata" || health.Fixture != f.Identity || health.PID != pid {
		return errors.New("runtime health did not identify this fixture and PID")
	}
	return nil
}

func (f *strataContractFixture) strataContractWaitStatus(t *testing.T, running bool) strataContractStatus {
	t.Helper()
	deadline := time.NewTimer(strataContractTimeout)
	defer deadline.Stop()
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
	for {
		var status strataContractStatus
		f.Manager.strataContractOK(t, "engine:status", map[string]string{"engine": "strata"}, &status)
		if status.Engine == "strata" && status.Installed && status.Running == running && status.Healthy == running && status.Port == f.Port {
			return status
		}
		select {
		case <-tick.C:
		case <-deadline.C:
			t.Fatalf("Strata status did not reach running=%v: %+v", running, status)
		}
	}
}

func (f *strataContractFixture) strataContractStopRuntime(t *testing.T) {
	t.Helper()
	f.Manager.strataContractOK(t, "engine:stop", map[string]string{"engine": "strata"}, nil)
	f.strataContractWaitStatus(t, false)
	strataContractWaitClosed(t, f.Port)
	f.runtimeStarted = false
	if err := os.Remove(filepath.Join(f.Root, "runtime-pid.json")); err != nil {
		t.Errorf("remove stopped runtime ownership marker: %v", err)
	}
	if f.runtimeProcess != nil {
		if err := f.runtimeProcess.Release(); err != nil {
			t.Errorf("release runtime process handle: %v", err)
		}
		f.runtimeProcess = nil
	}
}

func (f *strataContractFixture) strataContractCleanupRuntime(t *testing.T) {
	t.Helper()
	// Stop only this manager's Strata; never invoke orphan reclamation on built-ins.
	if f.runtimeStarted {
		response, err := f.Manager.strataContractCallWithin("engine:stop", map[string]string{"engine": "strata"}, strataContractTimeout)
		if err != nil || response.Error != nil {
			t.Errorf("cleanup owned Strata: transport=%v RPC=%v", err, response.Error)
		}
	}
	f.Manager.strataContractClose(t)
	if f.runtimeProcess != nil {
		// An unrelated listener must never authorize termination, even with a
		// previously verified handle. Recheck the nonce and PID before killing.
		if err := f.strataContractRuntimeIdentity(f.runtimeProcess.Pid); err == nil {
			if killErr := f.runtimeProcess.Kill(); killErr != nil && !errors.Is(killErr, os.ErrProcessDone) {
				t.Errorf("kill retained owned runtime: %v", killErr)
			}
			t.Error("owned runtime listener survived manager shutdown")
		} else if !strataContractConnectionRefused(err) {
			t.Errorf("cannot authenticate runtime during cleanup: %v", err)
		}
		if err := f.runtimeProcess.Release(); err != nil {
			t.Errorf("release owned runtime handle: %v", err)
		}
		f.runtimeProcess = nil
	}
	// Finalize the runtime handle before any fatal listener assertion.
	if f.ControlPort != 0 {
		strataContractWaitClosed(t, f.ControlPort)
	}
	if f.runtimeStarted {
		strataContractWaitClosed(t, f.Port)
		f.runtimeStarted = false
	}
}

func strataContractConnectionRefused(err error) bool {
	var errno syscall.Errno
	if !errors.As(err, &errno) {
		return false
	}
	if runtime.GOOS == "windows" {
		const wsaeconnrefused syscall.Errno = 10061
		return errno == wsaeconnrefused
	}
	return errno == syscall.ECONNREFUSED
}

func strataContractAddressInUse(err error) bool {
	var errno syscall.Errno
	if !errors.As(err, &errno) {
		return false
	}
	if runtime.GOOS == "windows" {
		const wsaeaddrinuse syscall.Errno = 10048
		return errno == wsaeaddrinuse
	}
	return errno == syscall.EADDRINUSE
}

func strataContractWaitClosed(t *testing.T, port int) {
	t.Helper()
	if !strataContractSafePort(port) {
		t.Fatal("unsafe listener-close check")
	}
	deadline := time.NewTimer(strataContractTimeout)
	defer deadline.Stop()
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
	for {
		// Windows retries a refused loopback SYN for about two seconds before
		// reporting WSAECONNREFUSED, so a shorter dial timeout never observes it.
		connection, err := net.DialTimeout("tcp4", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), 3*time.Second)
		if strataContractConnectionRefused(err) {
			return
		}
		if connection != nil {
			if err := connection.Close(); err != nil {
				t.Fatal(err)
			}
		}
		select {
		case <-tick.C:
		case <-deadline.C:
			t.Fatalf("owned listener on port %d did not close (last dial error: %v)", port, err)
		}
	}
}

func strataContractHTTP(t *testing.T, method string, port int, path string, value any) (int, http.Header, []byte) {
	t.Helper()
	status, header, body, err := strataContractRequestHTTP(method, port, path, value)
	if err != nil {
		t.Fatalf("fixture HTTP %s %s: %v", method, path, err)
	}
	return status, header, body
}

// Cleanup must record failed identity probes without aborting handle release.
func strataContractRequestHTTP(method string, port int, path string, value any) (int, http.Header, []byte, error) {
	if !strataContractSafePort(port) || !strings.HasPrefix(path, "/") {
		return 0, nil, nil, errors.New("unsafe fixture HTTP target")
	}
	var body io.Reader
	if value != nil {
		data, err := json.Marshal(value)
		if err != nil {
			return 0, nil, nil, err
		}
		body = bytes.NewReader(data)
	}
	ctx, cancel := context.WithTimeout(context.Background(), strataContractTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, method, "http://127.0.0.1:"+strconv.Itoa(port)+path, body)
	if err != nil {
		return 0, nil, nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	transport := &http.Transport{Proxy: nil}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, Timeout: strataContractTimeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	response, err := client.Do(request)
	if err != nil {
		return 0, nil, nil, err
	}
	data, readErr := io.ReadAll(io.LimitReader(response.Body, strataContractLimit+1))
	closeErr := response.Body.Close()
	if readErr != nil || closeErr != nil {
		return 0, nil, nil, fmt.Errorf("fixture HTTP read=%v close=%v", readErr, closeErr)
	}
	if len(data) > strataContractLimit {
		return 0, nil, nil, errors.New("fixture HTTP response exceeds limit")
	}
	return response.StatusCode, response.Header, data, nil
}

// Exclusions omit exact entries, never their children. Callers name every
// permitted mutation; all other files are hashed and directories recorded.
func strataContractArtifacts(t *testing.T, root string, excluded ...string) map[string]string {
	t.Helper()
	artifacts := make(map[string]string)
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if slices.Contains(excluded, path) {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if entry.IsDir() {
			artifacts[relative] = "directory"
			return nil
		}
		if !entry.Type().IsRegular() {
			return fmt.Errorf("unexpected private artifact type at %s: %s", relative, entry.Type())
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		artifacts[relative] = fmt.Sprintf("%x", sha256.Sum256(data))
		return nil
	})
	if err != nil {
		t.Fatalf("snapshot private fixture artifacts: %v", err)
	}
	return artifacts
}

func (f *strataContractFixture) strataContractControl(t *testing.T, nextLoadStatus int) strataContractControl {
	t.Helper()
	method := http.MethodGet
	var value any
	if nextLoadStatus != 0 {
		method = http.MethodPost
		value = map[string]int{"nextLoadStatus": nextLoadStatus}
	}
	status, _, body := strataContractHTTP(t, method, f.Port, strataContractControlPath, value)
	if status != http.StatusOK {
		t.Fatalf("fixture control HTTP status=%d", status)
	}
	var control strataContractControl
	strataContractDecode(t, body, &control)
	if control.Identity != f.Identity {
		t.Fatal("fixture control identity mismatch")
	}
	return control
}

func strataContractCount(control strataContractControl, method, path, model string) int {
	count := 0
	for _, request := range control.Requests {
		if request.Method == method && request.Path == path && request.Model == model {
			count += request.Count
		}
	}
	return count
}

type strataContractProxy struct {
	Root, AppDir, ClusterDir string
	Env                      []string
	RPC                      *strataContractRPC
	Port                     int
}

func strataContractStartProxy(t *testing.T, prepare func(*strataContractProxy)) *strataContractProxy {
	t.Helper()
	strataContractPlatform(t)
	p := &strataContractProxy{Root: t.TempDir()}
	p.AppDir, p.Env = strataContractEnvironment(p.Root)
	p.ClusterDir = filepath.Join(p.AppDir, "cluster")
	if err := os.MkdirAll(p.ClusterDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if prepare != nil {
		prepare(p)
	}
	p.RPC = strataContractStartChild(t, proxyBin, p.Env, "--cluster-dir", p.ClusterDir, "--log-level", "warn")
	t.Cleanup(func() {
		p.RPC.strataContractClose(t)
		if p.Port != 0 {
			strataContractWaitClosed(t, p.Port)
		}
	})
	return p
}

func (p *strataContractProxy) strataContractEnable(t *testing.T, descriptor engines.Descriptor) int {
	t.Helper()
	for attempt := 0; attempt < 8; attempt++ {
		port := strataContractPort(t)
		response := p.RPC.strataContractCall(t, "facade/enable", map[string]any{"engine": "strata", "descriptor": descriptor, "port": port, "ignorePersistedPort": true})
		if response.Error == nil {
			p.Port = port
			p.RPC.strataContractNotificationWithin(t, "e-strata:ready", nil, strataContractStartup)
			return port
		}
		if response.Error.Code != -32010 {
			t.Fatalf("enable Strata facade: %v", response.Error)
		}
	}
	t.Fatal("facade bind collisions exhausted bounded retry")
	return 0
}

func TestStrataContractDescriptorAndLifecycle(t *testing.T) {
	strataContractPlatform(t)
	t.Run("descriptor and owned lifecycle", func(t *testing.T) {
		fixture := strataContractNew(t, true)
		fixture.strataContractStartManager(t)
		descriptor := fixture.strataContractDescriptor(t)
		want := engines.Descriptor{Version: 1, Engine: "strata", Family: "strata", Label: "Strata", Available: true,
			Platforms: []string{"darwin/arm64", "linux/amd64", "windows/amd64"}, Protocols: []string{"openai-chat"}, ModelNaming: "exact",
			Operations: map[string]engines.OperationCapability{"load": {ResponseTimeoutS: 600}, "unload": {ResponseTimeoutS: 30}}}
		if !reflect.DeepEqual(descriptor, want) {
			t.Fatalf("Strata descriptor=%+v, want %+v", descriptor, want)
		}
		if !engines.ValidDynamicID(descriptor.Engine) || !engines.ValidRoutingID(descriptor.Engine) {
			t.Fatal("actual Strata descriptor ID rejected by dynamic routing syntax")
		}
		if err := engines.ProxyCanFront(descriptor); err != nil {
			t.Fatalf("actual Strata descriptor cannot be fronted: %v", err)
		}
		// Consume the initial stopped snapshot so it cannot satisfy withdrawal.
		fixture.Manager.strataContractNotification(t, "engine:model-identity", func(raw json.RawMessage) bool {
			var value strataContractIdentity
			strataContractDecode(t, raw, &value)
			_, present := value.Engines["strata"]
			return !present
		})
		fixture.strataContractStartRuntime(t)
		identity := fixture.Manager.strataContractNotification(t, "engine:model-identity", func(raw json.RawMessage) bool {
			var value strataContractIdentity
			strataContractDecode(t, raw, &value)
			_, ok := value.Engines["strata"]
			return ok
		})
		var observed strataContractIdentity
		strataContractDecode(t, identity.Params, &observed)
		owned := observed.Engines["strata"]
		digest, err := hex.DecodeString(owned.LaunchDigest)
		if err != nil || len(digest) != 32 || strings.ToLower(owned.LaunchDigest) != owned.LaunchDigest || owned.EngineRunID == "" || len(owned.ModelDigests) != 0 {
			t.Fatalf("owned runtime identity invalid or fabricated model digests: %+v", owned)
		}
		fixture.strataContractStopRuntime(t)
		fixture.Manager.strataContractNotification(t, "engine:model-identity", func(raw json.RawMessage) bool {
			var value strataContractIdentity
			strataContractDecode(t, raw, &value)
			_, present := value.Engines["strata"]
			return !present
		})
	})
	t.Run("wrong service is never adopted or stopped", func(t *testing.T) {
		listener := strataContractListen(t)
		_, text, err := net.SplitHostPort(listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		port, err := strconv.Atoi(text)
		if err != nil {
			t.Fatal(err)
		}
		server := &http.Server{ReadHeaderTimeout: strataContractTimeout, ReadTimeout: strataContractTimeout, WriteTimeout: strataContractTimeout, Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			strataContractRuntimeJSON(w, map[string]string{"service": "foreign-test-service"})
		})}
		serverDone := make(chan error, 1)
		go func() { serverDone <- server.Serve(listener) }()
		t.Cleanup(func() {
			if err := server.Close(); err != nil {
				t.Error(err)
			}
			select {
			case err := <-serverDone:
				if err != nil && !errors.Is(err, http.ErrServerClosed) {
					t.Error(err)
				}
			case <-time.After(strataContractTimeout):
				t.Error("foreign fixture listener did not reap")
			}
			strataContractWaitClosed(t, port)
		})
		fixture := strataContractNew(t, true)
		fixture.Port = port
		fixture.strataContractWriteRuntime(t)
		// The manager must probe the occupied target even in its initial manifest.
		var manifest map[string]json.RawMessage
		path := filepath.Join(fixture.AppDir, "engines", "strata.json")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		strataContractDecode(t, data, &manifest)
		var rt map[string]json.RawMessage
		strataContractDecode(t, manifest["runtime"], &rt)
		rt["port"] = json.RawMessage(strconv.Itoa(port))
		manifest["runtime"], err = json.Marshal(rt)
		if err != nil {
			t.Fatal(err)
		}
		strataContractWriteJSON(t, path, manifest)
		fixture.strataContractStartManager(t)
		response := fixture.Manager.strataContractCall(t, "engine:start", map[string]any{"engine": "strata", "port": port})
		if response.Error == nil || response.Error.Code != -32000 || !strings.Contains(response.Error.Message, "occupied by a service that did not identify as Strata") {
			t.Fatalf("wrong-service start error=%v", response.Error)
		}
		var status strataContractStatus
		fixture.Manager.strataContractOK(t, "engine:status", map[string]string{"engine": "strata"}, &status)
		if status.Running || status.Healthy {
			t.Fatalf("foreign service was adopted: %+v", status)
		}
		if _, err := os.Stat(filepath.Join(fixture.Root, "runtime-pid.json")); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("fake runtime launch marker exists or stat failed: %v", err)
		}
		fixture.strataContractCleanupRuntime(t)
		httpStatus, _, body := strataContractHTTP(t, http.MethodGet, port, "/health", nil)
		var health struct {
			Service string `json:"service"`
		}
		strataContractDecode(t, body, &health)
		if httpStatus != http.StatusOK || health.Service != "foreign-test-service" {
			t.Fatal("foreign owned listener did not survive manager cleanup")
		}
	})
}

// Future owner-approved RTX runtime procedure
//
// No code in this file installs or downloads software or launches real Strata
// by default. After explicit owner approval and an idle MSI interval, an
// operator uses already prepared Strata software/models, private configuration,
// and approved nonprotected ports. Record Strata, model, and driver versions
// and effective configuration; verify service identity, exact inventory and
// residency, load/unload, JSON/SSE chat, and capability rejection against that
// actual release. Stop only owned processes and record limitations/failures.
// Successful chat does not authorize strict capability. Genuine qualification
// requires B's complete content/launch/settings identity and tokenizer, template,
// context, output, and configuration evidence in local-qualification.json, with
// freshness checks. This procedure is outside this task and its acceptance run.
