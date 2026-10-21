// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

const srRuntimeArg = "--sr-process-runtime"
const srRuntimeEnv = "NVPAIR_SR_PROCESS_EXECUTABLE"
const srNeighbor = "sr-neighbor"
const srOllamaRuntimeEnv = "NVPAIR_SR_OLLAMA_EXECUTABLE"
const srOllamaModel = "sr-ollama:latest"

// This dispatch reuses the wrapper's fixed test image, rather than introducing
// another executable path or a native engine. All qualification below is fixture-only.
func init() {
	// Broker worker arguments distinguish the relay from the synthetic native
	// runtime, which uses this same fixed image with --host instead.
	if os.Getenv("NVPAIR_SR_RELAY_IMAGE") != "" && len(os.Args) > 2 && os.Args[1] == "--log-level" {
		image, err := os.Executable()
		if err != nil {
			os.Exit(2)
		}
		if os.Getenv("NVPAIR_SR_RELAY_IMAGE") == image {
			if runtime.GOOS != "windows" || !neFixedTestImage(image) {
				os.Exit(2)
			}
			if err := runSRWorkerRelay(); err != nil {
				// Frame contents, policy, native arguments and paths stay private.
				fmt.Fprintln(os.Stderr, "SR worker relay failed:", err)
				os.Exit(2)
			}
			os.Exit(0)
		}
	}
	if selected := os.Getenv(srOllamaRuntimeEnv); selected != "" {
		image, err := os.Executable()
		if err != nil || runtime.GOOS != "windows" || selected != image || !neFixedTestImage(image) || len(os.Args) != 2 || os.Args[1] != "serve" {
			os.Exit(2)
		}
		if err := runSROllamaRuntime(); err != nil {
			fmt.Fprintln(os.Stderr, "SR Ollama fixture launch failed:", err)
			os.Exit(2)
		}
		os.Exit(0)
	}
	if len(os.Args) < 6 || os.Args[5] != srRuntimeArg {
		return
	}
	image, err := os.Executable()
	if err != nil || runtime.GOOS != "windows" || os.Getenv(srRuntimeEnv) != image || !neFixedTestImage(image) {
		os.Exit(2)
	}
	if err := runSRRuntime(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "SR fixture launch failed:", err)
		os.Exit(2)
	}
	os.Exit(0)
}

type srEffects struct {
	Loaded  bool `json:"loaded"`
	Loads   int  `json:"loads"`
	Unloads int  `json:"unloads"`
	Chats   int  `json:"chats"`
}

type srRuntimeState struct {
	Base                 neRuntimeState       `json:"base"`
	Models               map[string]srEffects `json:"models"`
	Content              int                  `json:"content"`
	HeldChats            int                  `json:"heldChats"`
	InventoryUnavailable bool                 `json:"inventoryUnavailable"`
	DropNextLoadResponse bool                 `json:"dropNextLoadResponse"`
	HideInventoryOnDrop  bool                 `json:"hideInventoryOnDrop"`
}

// This owned process supplies synthetic B identity and observable cold inference,
// without claiming an Ollama native serving envelope or resident-load support.
func runSROllamaRuntime() error {
	path := os.Getenv("NVPAIR_SR_OLLAMA_CONFIG")
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var config neRuntimeConfig
	if err := neDecode(bytes.NewReader(data), &config); err != nil {
		return err
	}
	if !neSafePort(config.Port) || !filepath.IsAbs(config.Root) || path != filepath.Join(config.Root, "switch-runtime.json") || config.Nonce == "" {
		return errors.New("invalid private Ollama fixture configuration")
	}
	listener, err := net.Listen("tcp4", net.JoinHostPort("127.0.0.1", strconv.Itoa(config.Port)))
	if err != nil {
		return err
	}
	defer listener.Close()
	client := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{Proxy: nil}}
	defer client.CloseIdleConnections()
	response, err := client.Get(os.Getenv("NVPAIR_TEST_SWITCH_CONTROL") + "/started?engine=ollama&pid=" + strconv.Itoa(os.Getpid()))
	if err != nil {
		return err
	}
	if err := response.Body.Close(); err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("private Ollama start callback: HTTP %d", response.StatusCode)
	}
	digest := sha256.Sum256([]byte(srOllamaModel))
	state := srRuntimeState{Base: neRuntimeState{Nonce: config.Nonce, PID: os.Getpid()}, Models: map[string]srEffects{srOllamaModel: {}}}
	var mu sync.Mutex
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		var result any
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/__sr/control" && r.Header.Get("X-NE-Fixture-Nonce") == config.Nonce:
			result = state
		case r.Method == http.MethodGet && r.URL.Path == "/":
			// The broker probes Ollama's root independently of manifest readiness.
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			if _, err := io.WriteString(w, "Ollama is running"); err != nil {
				state.Base.Unexpected++
			}
			return
		case r.Method == http.MethodGet && r.URL.Path == "/api/tags":
			result = map[string]any{"models": []any{map[string]string{"name": srOllamaModel, "digest": hex.EncodeToString(digest[:])}}}
		case r.Method == http.MethodGet && r.URL.Path == "/api/ps":
			models := []any{}
			if state.Models[srOllamaModel].Loaded {
				models = append(models, map[string]string{"name": srOllamaModel})
			}
			result = map[string]any{"models": models}
		case r.Method == http.MethodGet && r.URL.Path == "/api/version":
			result = map[string]string{"version": "fixture-only"}
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models":
			result = map[string]any{"object": "list", "data": []any{map[string]string{"id": srOllamaModel}}}
		case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
			state.Base.Chats++
			if r.Header.Get(nodepolicy.StrictAdmissionHeader) != "" {
				state.Base.LeakedHeaders++
			}
			var request struct {
				Model    string `json:"model"`
				Messages []struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"messages"`
				MaxTokens int  `json:"max_tokens"`
				Stream    bool `json:"stream"`
			}
			if err := neDecode(r.Body, &request); err != nil || request.Model != srOllamaModel || request.Stream || len(request.Messages) != 1 || request.MaxTokens != neOutput {
				state.Base.Unexpected++
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			effect := state.Models[srOllamaModel]
			if !effect.Loaded {
				effect.Loaded = true
				effect.Loads++
				state.Base.Loads++
				state.Base.Loaded = true
			}
			effect.Chats++
			state.Models[srOllamaModel] = effect
			result = map[string]any{"id": "sr-ollama-chat", "object": "chat.completion", "model": srOllamaModel, "choices": []any{map[string]any{"index": 0, "message": map[string]string{"role": "assistant", "content": "fixture-only"}, "finish_reason": "stop"}}}
		default:
			state.Content++
			state.Base.Unexpected++
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if err := json.NewEncoder(w).Encode(result); err != nil {
			state.Base.Unexpected++
		}
	})
	return (&http.Server{Handler: handler, ReadHeaderTimeout: 2 * time.Second}).Serve(listener)
}

func runSRRuntime(args []string) error {
	if len(args) < 9 || args[0] != "--host" || args[1] != "127.0.0.1" || args[2] != "--port" || args[4] != srRuntimeArg {
		return errors.New("invalid managed fixture prefix")
	}
	options := make(map[string]string)
	for i := 0; i < len(args); i += 2 {
		if i == 4 {
			// The marker has no value; index 5 is the next option key.
			i++
		}
		if i+1 >= len(args) {
			return errors.New("missing fixture argument")
		}
		switch args[i] {
		case "--host", "--port", "--cors-origins", "--sr-config", "--models-preset", "--models-max":
		default:
			return errors.New("unexpected fixture argument")
		}
		if _, exists := options[args[i]]; exists {
			return errors.New("duplicate fixture argument")
		}
		options[args[i]] = args[i+1]
	}
	data, err := os.ReadFile(options["--sr-config"])
	if err != nil {
		return err
	}
	var config neRuntimeConfig
	if err := neDecode(bytes.NewReader(data), &config); err != nil {
		return err
	}
	port, err := strconv.Atoi(options["--port"])
	if err != nil || !neSafePort(port) || port != config.Port || options["--models-max"] != "2" || options["--sr-config"] != filepath.Join(config.Root, "runtime.json") || !neWithin(filepath.Join(jointAppDir(config.Root), "engine-bin", "serving", "llamacpp"), options["--models-preset"]) {
		return errors.New("fixture endpoint or generated preset mismatch")
	}
	listener, err := net.Listen("tcp4", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
	if err != nil {
		return fmt.Errorf("reserved fake bind: %w", err)
	}
	defer listener.Close()
	log, err := os.OpenFile(filepath.Join(config.Root, "starts.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	err = json.NewEncoder(log).Encode(neRuntimeStart{Nonce: config.Nonce, PID: os.Getpid(), Preset: options["--models-preset"], Args: args})
	if err := errors.Join(err, log.Close()); err != nil {
		return err
	}
	state := srRuntimeState{Base: neRuntimeState{Nonce: config.Nonce, PID: os.Getpid()}, Models: map[string]srEffects{neModel: {}, srNeighbor: {}}}
	var mu sync.Mutex
	gate := make(chan struct{})
	close(gate)
	held := false
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		var response any
		switch {
		case r.Method == http.MethodGet && (r.URL.Path == "/health" || r.URL.Path == "/__ne/control"):
			response = state.Base
		case r.URL.Path == "/__sr/control" && r.Header.Get("X-NE-Fixture-Nonce") == config.Nonce:
			switch r.URL.Query().Get("action") {
			case "hold":
				if !held {
					gate = make(chan struct{})
					held = true
				}
			case "release":
				if held {
					close(gate)
					held = false
				}
			case "inventory-unavailable":
				state.InventoryUnavailable = true
			case "inventory-available":
				state.InventoryUnavailable = false
			case "drop-next-load-response":
				state.DropNextLoadResponse = true
			case "drop-load-and-hide-inventory":
				state.DropNextLoadResponse = true
				state.HideInventoryOnDrop = true
			}
			response = state
		case r.Method == http.MethodGet && r.URL.Path == "/props" && r.URL.RawQuery == "":
			response = map[string]string{"role": "router"}
		case r.Method == http.MethodGet && r.URL.Path == "/props":
			model := r.URL.Query().Get("model")
			effect, ok := state.Models[model]
			if !ok || !effect.Loaded || r.URL.Query().Get("autoload") != "false" {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			context := neContext
			if model == srNeighbor {
				context = 1024
			}
			state.Base.ModelProps++
			response = map[string]any{"default_generation_settings": map[string]int{"n_ctx": context}, "total_slots": 1}
		case r.Method == http.MethodGet && r.URL.Path == "/models":
			if state.InventoryUnavailable {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			var models []any
			for _, model := range []string{neModel, srNeighbor} {
				status := "unloaded"
				if state.Models[model].Loaded {
					status = "loaded"
				}
				models = append(models, map[string]any{"id": model, "status": map[string]string{"value": status}})
			}
			response = map[string]any{"data": models}
		case r.Method == http.MethodPost && (r.URL.Path == "/models/load" || r.URL.Path == "/models/unload"):
			var request struct {
				Model string `json:"model"`
			}
			if err := neDecode(r.Body, &request); err != nil {
				state.Base.Unexpected++
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			effect, ok := state.Models[request.Model]
			if !ok {
				state.Base.Unexpected++
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			effect.Loaded = r.URL.Path == "/models/load"
			if effect.Loaded {
				effect.Loads++
				state.Base.Loads++
			} else {
				effect.Unloads++
				state.Base.Unloads++
			}
			state.Models[request.Model] = effect
			state.Base.Loaded = state.Models[neModel].Loaded
			if effect.Loaded && state.DropNextLoadResponse {
				state.DropNextLoadResponse = false
				if state.HideInventoryOnDrop {
					state.InventoryUnavailable = true
					state.HideInventoryOnDrop = false
				}
				conn, _, err := http.NewResponseController(w).Hijack()
				if err != nil {
					state.Base.Unexpected++
					return
				}
				_ = conn.Close()
				return
			}
			response = map[string]any{"ok": true, "model": request.Model}
		case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
			var request struct {
				Model    string `json:"model"`
				Messages []struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"messages"`
				MaxTokens int  `json:"max_tokens"`
				Stream    bool `json:"stream"`
			}
			state.Base.Chats++
			if r.Header.Get(nodepolicy.StrictAdmissionHeader) != "" {
				state.Base.LeakedHeaders++
			}
			if err := neDecode(r.Body, &request); err != nil {
				state.Base.Unexpected++
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			effect, ok := state.Models[request.Model]
			if !ok || !effect.Loaded || request.Stream {
				state.Base.Unexpected++
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			effect.Chats++
			state.Models[request.Model] = effect
			wait := gate
			if held {
				state.HeldChats++
				mu.Unlock()
				select {
				case <-wait:
				case <-r.Context().Done():
				}
				mu.Lock()
				state.HeldChats--
			}
			response = map[string]any{"id": "sr-chat", "object": "chat.completion", "model": request.Model, "choices": []any{map[string]any{"index": 0, "message": map[string]string{"role": "assistant", "content": "fixture-only"}, "finish_reason": "stop"}}}
		default:
			state.Content++
			state.Base.Unexpected++
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			state.Base.Unexpected++
		}
	})
	return (&http.Server{Handler: handler, ReadHeaderTimeout: 2 * time.Second}).Serve(listener)
}

type srProcessFixture struct {
	*neServingFixture
	relay            *srWorkerRelay
	peerClient       *http.Client
	switchTarget     bool
	switchEnginePort int
	switchFacadePort int
	switchControl    *httptest.Server
	switchMu         sync.Mutex
	switchStarts     []int32
}

// NE owns fixed-image dispatch, service-port reservations, private manifests,
// pre-start facade settings and process cleanup. SR adds two synthetic models.
func newSRProcessFixture(t *testing.T) *srProcessFixture {
	return newSRProcessFixtureWithRetention(t, false, "-1")
}

func newSRSwitchProcessFixture(t *testing.T) *srProcessFixture {
	return newSRProcessFixtureWithRetention(t, true, "-1")
}

func newSRProcessFixtureWithRetention(t *testing.T, switchTarget bool, retention string) *srProcessFixture {
	t.Helper()
	if runtime.GOOS != "windows" {
		t.Skip("SR process evidence requires the Windows fixed-path lane")
	}
	f := &srProcessFixture{neServingFixture: newNEServingFixture(t)}
	if switchTarget {
		f.enableSwitchTarget(t)
	}
	peerDir := filepath.Join(t.TempDir(), "cluster")
	targetDir := filepath.Join(f.root, "cluster")
	if err := os.MkdirAll(peerDir, 0o700); err != nil {
		t.Fatal(err)
	}
	targetCert := mintClusterIdentity(t, targetDir, "sr-target")
	peerCert := mintClusterIdentity(t, peerDir, "sr-peer")
	for _, member := range []struct{ dir, id, peer, cert string }{{targetDir, "sr-target", "sr-peer", peerCert}, {peerDir, "sr-peer", "sr-target", targetCert}} {
		writeActiveAdmission(t, member.dir)
		writePin(t, member.dir, member.peer, member.cert)
		writeMembers(t, member.dir, "sr-target", "sr-peer")
		jointWriteJSON(t, filepath.Join(member.dir, "identity.json"), map[string]string{"node_uuid": member.id})
	}
	mesh := clustertrust.Open(peerDir)
	tlsConfig, ok := mesh.ClientTLSConfig("sr-target")
	if !ok {
		t.Fatal("SR paired fixture TLS unavailable")
	}
	f.peerClient = &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{TLSClientConfig: tlsConfig, Proxy: nil}}
	t.Cleanup(f.peerClient.CloseIdleConnections)
	for _, port := range []int{f.enginePort, f.facadePort, f.scannerPort, f.inertPort} {
		if !neSafePort(port) {
			t.Fatalf("protected fixture port %d", port)
		}
	}
	manifestPath := filepath.Join(f.app, "engines", "llamacpp.json")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var manifest map[string]any
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	// Decode the known fixture structure explicitly; never carry the NE launch
	// selector into SR's child because its dispatch deliberately rejects it.
	var override struct {
		Runtime   map[string]any `json:"runtime"`
		Platforms map[string]struct {
			Runtime map[string]any `json:"runtime"`
		} `json:"platforms"`
	}
	if err := json.Unmarshal(data, &override); err != nil {
		t.Fatal(err)
	}
	override.Runtime["env"] = map[string]string{srRuntimeEnv: f.image}
	manifest["runtime"] = override.Runtime
	manifest["platforms"] = map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{f.image}, "runtime": override.Runtime, "models_dir": filepath.Join(f.root, "models")}}
	jointWriteJSON(t, manifestPath, manifest)
	content := []byte("SR synthetic fixture-only model bytes; no native engine qualification")
	modelPath := filepath.Join(f.root, "model.gguf")
	if err := os.WriteFile(modelPath, content, 0o600); err != nil {
		t.Fatal(err)
	}
	source := "[*]\nfit = off\nload-on-startup = false\n"
	for _, model := range []string{neModel, srNeighbor} {
		context := neContext
		if model == srNeighbor {
			context = 1024
		}
		source += fmt.Sprintf("\n[%s]\nmodel = %s\nctx-size = %d\nparallel = 1\nkv-unified = false\nsleep-idle-seconds = %s\n", model, filepath.ToSlash(modelPath), context, retention)
	}
	if err := os.WriteFile(f.source, []byte(source), 0o600); err != nil {
		t.Fatal(err)
	}
	policy := nodepolicy.Default()
	policy.Revision = "1"
	policy.Admission.MaxResidentModels = 2
	policy.Admission.MaxConcurrentPerModel = 1
	policy.Admission.MemoryBudgetBytes = 2 * neMemory
	policy.Admission.QueueTimeoutSeconds = 1
	policy.Idle.StartOnDemand = false
	for _, model := range []string{neModel, srNeighbor} {
		context, output := neContext, neOutput
		if model == srNeighbor {
			context, output = 1024, 64
		}
		policy.Profiles = append(policy.Profiles, nodepolicy.Profile{Name: model, Engine: "llamacpp", Model: model, ContextTokens: context, MaxOutputTokens: output, MemoryBytes: neMemory, MaxConcurrent: 1, Qualification: &nodepolicy.QualificationRef{ID: "sr-" + model, Revision: "1"}})
	}
	if err := policy.Validate(); err != nil {
		t.Fatal(err)
	}
	jointWriteJSON(t, filepath.Join(f.app, nodepolicy.FileName), policy)
	f.start(t)
	before := f.snapshot(t)
	ram, vram := int64(neMemory), int64(0)
	hash := sha256.Sum256(content)
	declaration := &engines.ServingConfig{Version: 1}
	for _, model := range []string{neModel, srNeighbor} {
		output := int64(neOutput)
		if model == srNeighbor {
			output = 64
		}
		declaration.Models = append(declaration.Models, engines.ServingModel{ModelID: model, MaxOutputTokens: output, Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram}, Artifacts: []engines.ServingArtifact{{Role: "model", SHA256: hex.EncodeToString(hash[:]), Provenance: engines.ServingProvenance{Kind: "supplied-sha256"}}}})
	}
	request := settings.Request{Engine: "llamacpp", ExpectedRevision: before.Revision, RequestID: "sr-apply", Settings: settings.Config{ServerPort: f.enginePort, ProxyPort: f.facadePort, LaunchText: srRuntimeArg + " --sr-config " + strconv.Quote(filepath.Join(f.root, "runtime.json")) + " --models-preset " + strconv.Quote(f.source) + " --models-max 2", Serving: declaration}}
	var preview settings.Preview
	f.call(t, "engine:preview-settings", request, &preview)
	if len(preview.Errors) != 0 || preview.Conflict != nil {
		t.Fatalf("fixture-only finite preview: errors=%v conflict=%v", preview.Errors, preview.Conflict)
	}
	request.Settings = preview.Settings.Clone()
	request.ExpectedRevision = preview.Revision
	f.call(t, "engine:apply-settings", request, nil)
	f.call(t, "engine:start", map[string]string{"engine": "llamacpp"}, nil)
	f.retainRuntime(t)
	snapshot := f.snapshot(t)
	if snapshot.Serving == nil || len(snapshot.Serving.LaunchValidated) != 2 {
		t.Fatal("two-model launch validation absent")
	}
	identity := regexp.MustCompile(`engine identity engine=llamacpp run_id=(run-[0-9a-f]+) launch_digest=([0-9a-f]{64}) models_with_digest=2`)
	var launch string
	neEventually(t, "two-model manager identity", neCallBudget, func() bool {
		matches := identity.FindAllStringSubmatch(f.p.output.String(), -1)
		if len(matches) == 0 {
			return false
		}
		launch = matches[len(matches)-1][2]
		return true
	})
	var qualifications []any
	for _, envelope := range snapshot.Serving.LaunchValidated {
		qualifications = append(qualifications, map[string]any{"id": "sr-" + envelope.ModelID, "revision": "1", "engine": "llamacpp", "model": envelope.ModelID, "modelDigests": []string{envelope.ContentDigest}, "launchDigests": []string{launch}, "tokenizer": "synthetic-sr-tokenizer", "chatTemplate": "synthetic-sr-template", "contextTokens": envelope.PerSlotContextTokens, "maxOutputTokens": envelope.MaxOutputTokens, "routes": []string{"/v1/chat/completions"}, "shapes": map[string]any{"tools": false, "structuredOutput": []string{}, "reasoningReplay": false}, "accounting": map[string]any{"method": "utf8-bytes", "requestOverheadTokens": 8, "messageOverheadTokens": 4, "toolOverheadTokens": 0, "schemaOverheadTokens": 0}, "evidence": "Fixture-only SR/NE/B wiring evidence. No native engine, model, hardware or performance qualification."})
	}
	f.stop(t)
	jointWriteJSON(t, filepath.Join(f.app, "local-qualification.json"), map[string]any{"version": 2, "qualifications": qualifications})
	f.start(t)
	f.call(t, "engine:start", map[string]string{"engine": "llamacpp"}, nil)
	f.retainRuntime(t)
	neEventually(t, "two fixture-only strict capabilities", neCallBudget, func() bool {
		models := f.models(t)
		return models.Policy != nil && models.Policy.StrictAdmission != nil && len(models.Policy.StrictAdmission.Models) == 2
	})
	f.call(t, "llamacpp-proxy:node/add-manual", map[string]any{"id": "sr-self", "host": "127.0.0.1", "port": f.facadePort, "addresses": []string{"127.0.0.1"}, "models": []string{neModel, srNeighbor}}, nil)
	f.call(t, "llamacpp-proxy:node/select", map[string]string{"id": "sr-self"}, nil)
	neEventually(t, "healthy local backend advertised to facade", neCallBudget, func() bool {
		status, _, err := f.http(http.MethodGet, f.facadePort, "/health", nil, nil)
		return err == nil && status == http.StatusOK
	})
	t.Cleanup(func() {
		if _, err := f.neServingFixture.state(); f.p != nil && err == nil {
			f.control(t, "release")
		}
	})
	t.Log("SR evidence is fixture-only: real broker/manager/proxy and synthetic NE/B data, with no native model qualification")
	return f
}

// Switch and cross-engine contention own a second fake engine. Its settings and
// executable are private before the first broker start; other SR cases keep
// the NE fixture's single-facade startup and inert Ollama manifest.
func (f *srProcessFixture) enableSwitchTarget(t *testing.T) {
	t.Helper()
	f.switchTarget = true
	reservations := []net.Listener{listenFixtureLoopback(t), listenFixtureLoopback(t)}
	f.switchEnginePort = reservations[0].Addr().(*net.TCPAddr).Port
	f.switchFacadePort = reservations[1].Addr().(*net.TCPAddr).Port
	servicePorts := []int{f.ports.nodeInfo, f.ports.errors, f.ports.workload, f.ports.clusterManager, f.ports.engineManagerHTTP, f.ports.engineControl}
	for _, port := range []int{f.switchEnginePort, f.switchFacadePort} {
		if !neSafePort(port) || port == f.enginePort || port == f.facadePort || port == f.scannerPort || port == f.inertPort {
			t.Fatalf("switch fixture selected occupied or protected port %d", port)
		}
		for _, servicePort := range servicePorts {
			if port == servicePort {
				t.Fatalf("switch fixture selected service port %d", port)
			}
		}
	}
	for _, listener := range reservations {
		if err := listener.Close(); err != nil {
			t.Fatalf("release switch port reservation: %v", err)
		}
	}
	f.switchControl = newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/started" || r.URL.Query().Get("engine") != "ollama" {
			http.Error(w, "unexpected fake engine callback", http.StatusBadRequest)
			return
		}
		pid, err := strconv.ParseInt(r.URL.Query().Get("pid"), 10, 32)
		if err != nil || pid <= 0 {
			http.Error(w, "invalid fake engine PID", http.StatusBadRequest)
			return
		}
		f.switchMu.Lock()
		f.switchStarts = append(f.switchStarts, int32(pid))
		f.switchMu.Unlock()
	}))
	t.Cleanup(f.switchControl.Close)
	probe := map[string]any{"http": "http://127.0.0.1:{port}/api/tags", "status": http.StatusOK, "timeout_s": 10, "interval_s": 60, "json_match": nil}
	configPath := filepath.Join(f.root, "switch-runtime.json")
	jointWriteJSON(t, configPath, neRuntimeConfig{Root: f.root, Nonce: f.nonce, Port: f.switchEnginePort})
	runtimeConfig := map[string]any{
		"mode": "process", "bin": f.image, "cli": filepath.Join(f.root, "absent-ollama-cli"), "port": f.switchEnginePort, "bind": "127.0.0.1", "ready": probe, "health": probe,
		"env":  map[string]string{srOllamaRuntimeEnv: f.image, "NVPAIR_SR_OLLAMA_CONFIG": configPath, "NVPAIR_TEST_SWITCH_CONTROL": f.switchControl.URL},
		"stop": map[string]any{"signal": "term", "grace_s": 1},
	}
	modelsDir := filepath.Join(f.root, "switch-models")
	jointWriteJSON(t, filepath.Join(f.app, "engines", "ollama.json"), map[string]any{
		"engine": "ollama", "runtime": runtimeConfig, "models_dir": modelsDir,
		"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{"detect": []string{f.image}, "runtime": runtimeConfig, "models_dir": modelsDir}},
	})
	journalPath := filepath.Join(f.root, "engine-settings-operations.json")
	data, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	var journal map[string]any
	if err := json.Unmarshal(data, &journal); err != nil {
		t.Fatal(err)
	}
	journal["ollama"] = map[string]any{"explicit": true, "snapshot": settings.Snapshot{
		Engine: "ollama", Revision: 1, AppliedRevision: 1, Phase: "idle", Format: "pair-arguments-v1", Settings: settings.Config{ServerPort: f.switchEnginePort, ProxyPort: f.switchFacadePort},
	}}
	jointWriteJSON(t, journalPath, journal)
}

func (f *srProcessFixture) start(t *testing.T) {
	t.Helper()
	if f.relay == nil && !f.switchTarget {
		f.neServingFixture.start(t)
		return
	}
	for _, worker := range []string{"nvpair-node-info", "nvpair-workload-manager", "nvpair-errors", "nvpair-manual-nodes", "nvpair-node-settings", "nvpair-cluster-manager", "nvpair-job-scheduler"} {
		if _, err := os.Stat(filepath.Join(f.root, worker+".exe")); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("optional worker %s must be absent from private cwd: %v", worker, err)
		}
	}
	if f.relay != nil {
		f.startRelayed(t)
		return
	}
	if f.p != nil {
		t.Fatal("fixture broker already running")
	}
	// The default NE isolation checks expect Ollama to remain inert. Check the
	// switch target separately, and still check every other private engine.
	for _, engine := range engines.Names() {
		data, err := os.ReadFile(filepath.Join(f.app, "engines", engine+".json"))
		if err != nil || len(data) > neLimit {
			t.Fatalf("%s cannot read private manifest: %v", engine, err)
		}
		var manifest struct {
			Runtime   neFixtureRuntime             `json:"runtime"`
			ModelsDir string                       `json:"models_dir"`
			Platforms map[string]neFixturePlatform `json:"platforms"`
		}
		if err := json.Unmarshal(data, &manifest); err != nil {
			t.Fatal(err)
		}
		if engine == "ollama" {
			platform, ok := manifest.Platforms[runtime.GOOS+"/"+runtime.GOARCH]
			if !ok || !reflect.DeepEqual(platform.Detect, []string{f.image}) || manifest.Runtime.Bin != f.image || manifest.Runtime.CLI != filepath.Join(f.root, "absent-ollama-cli") || manifest.Runtime.Port != f.switchEnginePort || platform.Runtime.Port != f.switchEnginePort || platform.Runtime.Ready.HTTP != "http://127.0.0.1:{port}/api/tags" || platform.ModelsDir != filepath.Join(f.root, "switch-models") {
				t.Fatal("switch target escaped private image, port or model directory")
			}
			continue
		}
		f.assertRuntimeIsolation(t, engine, "raw shared", manifest.Runtime, manifest.ModelsDir)
		f.assertPlatformIsolation(t, engine, "raw host", manifest.Platforms)
	}
	args := append([]string{"--scanner-path", f.scanner, "--engine-manager-path", engineMgrBin, "--proxy-path", proxyBin, "--proxy-engines=llamacpp,ollama", "--cluster-dir", filepath.Join(f.root, "cluster"), "--log-level", "info"}, f.ports.args()...)
	f.p = startInteropProcess(t, f.root, brokerBin, f.ports.engineManagerHTTP, []string{"OLLAMA_HOST=", "NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + f.scanner, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(f.scannerPort)}, args...)
	f.p.ports = append(f.p.ports, f.ports.engineControl, f.scannerPort, f.facadePort, f.switchEnginePort, f.switchFacadePort)
	for _, path := range []string{engineMgrBin, proxyBin, f.scanner} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		f.p.paths = append(f.p.paths, canonical)
	}
	waitForMethod(t, f.p.msgs, "app:ready", neCallBudget)
	for _, facade := range []struct {
		name string
		port int
	}{{"llamacpp-proxy", f.facadePort}, {"ollama-proxy", f.switchFacadePort}} {
		if port := waitEngineProxyReady(t, facade.name, f.p.in, f.p.msgs, neCallBudget); port != facade.port {
			t.Fatalf("%s bound port %d, want reserved %d", facade.name, port, facade.port)
		}
	}
	for _, engine := range engines.Names() {
		var manifest struct {
			Platforms map[string]neFixturePlatform `json:"platforms"`
		}
		f.call(t, "engine:describe", map[string]string{"engine": engine}, &manifest)
		if engine == "ollama" {
			platform, ok := manifest.Platforms[runtime.GOOS+"/"+runtime.GOARCH]
			if !ok || !reflect.DeepEqual(platform.Detect, []string{f.image}) || platform.Runtime.Bin != f.image || platform.Runtime.CLI != filepath.Join(f.root, "absent-ollama-cli") || platform.Runtime.Port != f.switchEnginePort || platform.Runtime.Ready.HTTP != "http://127.0.0.1:{port}/api/tags" || platform.ModelsDir != filepath.Join(f.root, "switch-models") {
				t.Fatal("effective switch target escaped private image, port, probe or model directory")
			}
			continue
		}
		f.assertPlatformIsolation(t, engine, "effective host", manifest.Platforms)
	}
}

// The relay has one bounded correlation table and never decodes/re-encodes a
// forwarded frame. Only resident RPCs reach its private loopback controller.
// Both roles run the real product binary, and the relay owns its Cmd and Wait.
type srRelayEvent struct {
	Role     string            `json:"role"`
	PID      int               `json:"pid"`
	RelayPID int               `json:"relayPid"`
	Phase    string            `json:"phase"`
	Method   string            `json:"method,omitempty"`
	Params   json.RawMessage   `json:"params,omitempty"`
	Result   json.RawMessage   `json:"result,omitempty"`
	Error    *jsonrpc.RPCError `json:"error,omitempty"`
}

type srRelayAction struct {
	Action string                             `json:"action,omitempty"`
	Probe  *nodepolicy.DispatchResidentParams `json:"probe,omitempty"`
}

func runSRWorkerRelay() error {
	role, binary := "proxy", os.Getenv("NVPAIR_SR_RELAY_PROXY")
	for _, arg := range os.Args[1:] {
		if arg == "--http-port" {
			role, binary = "manager", os.Getenv("NVPAIR_SR_RELAY_MANAGER")
		}
	}
	if !neWithin(os.Getenv("NVPAIR_TEST_BIN_DIR"), binary) {
		return errors.New("worker outside fixed executable root")
	}
	client := &http.Client{Timeout: neCallBudget + 5*time.Second, Transport: &http.Transport{Proxy: nil}}
	defer client.CloseIdleConnections()
	cmd := exec.Command(binary, os.Args[1:]...)
	for _, assignment := range os.Environ() {
		if !strings.HasPrefix(assignment, "NVPAIR_SR_RELAY_") {
			cmd.Env = append(cmd.Env, assignment)
		}
	}
	cmd.Stderr = os.Stderr
	input, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	defer input.Close()
	// Own the pipe explicitly so Wait cannot close stdout before its final
	// response/notification has drained through the transparent pump.
	output, writer, err := os.Pipe()
	if err != nil {
		return err
	}
	defer output.Close()
	cmd.Stdout = writer
	if err := cmd.Start(); err != nil {
		_ = writer.Close()
		return err
	}
	if err := writer.Close(); err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return err
	}
	wait := make(chan error, 1)
	go func() { wait <- cmd.Wait() }()
	// Normal exit paths kill/reap the actual worker if EOF shutdown stalls.
	// TerminateProcess bypasses defers; the controller retains both process
	// handles for explicit survivor cleanup after broker escalation.
	reaped := false
	defer func() {
		if !reaped {
			_ = cmd.Process.Kill()
			<-wait
		}
	}()
	report := func(event srRelayEvent) (srRelayAction, error) {
		event.Role, event.PID, event.RelayPID = role, cmd.Process.Pid, os.Getpid()
		data, err := json.Marshal(event)
		if err != nil {
			return srRelayAction{}, err
		}
		response, err := client.Post(os.Getenv("NVPAIR_SR_RELAY_CONTROL"), "application/json", bytes.NewReader(data))
		if err != nil {
			return srRelayAction{}, err
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			return srRelayAction{}, fmt.Errorf("private controller refused role=%s phase=%s pid=%d relay_pid=%d: HTTP %d", role, event.Phase, event.PID, event.RelayPID, response.StatusCode)
		}
		var action srRelayAction
		err = neDecode(response.Body, &action)
		return action, err
	}
	if _, err := report(srRelayEvent{Phase: "started"}); err != nil {
		return err
	}
	type call struct {
		method string
		params json.RawMessage
	}
	var mu, inputMu sync.Mutex
	pending := make(map[string]call)
	probeID := fmt.Sprintf(`"sr-worker-loss-probe-%d"`, cmd.Process.Pid)
	killBoundary := errors.New("requested actual worker loss")
	forward := func(reader io.Reader, destination io.Writer, fromBroker bool) error {
		scanner := bufio.NewScanner(reader)
		scanner.Buffer(make([]byte, 4096), jsonrpc.WorkerFrameBytes)
		// Retain the delimiter and all bytes, including extension fields and
		// CRLF. The bound applies equally to inspected and unrelated frames.
		scanner.Split(func(data []byte, eof bool) (int, []byte, error) {
			if i := bytes.IndexByte(data, '\n'); i >= 0 {
				return i + 1, data[:i+1], nil
			}
			if eof && len(data) > 0 {
				return len(data), data, nil
			}
			return 0, nil, nil
		})
		for scanner.Scan() {
			frame := scanner.Bytes()
			var message jsonrpc.Message
			var event srRelayEvent
			if json.Unmarshal(frame, &message) == nil && message.ID != nil {
				id := string(*message.ID)
				mu.Lock()
				if fromBroker && message.IsRequest() {
					switch message.Method {
					case nodepolicy.MethodPrepareResident, nodepolicy.MethodDispatchResident, nodepolicy.MethodFinishResident, nodepolicy.MethodObserveResident, nodepolicy.MethodReconcileResident:
						if len(pending) >= 32 || id == probeID {
							mu.Unlock()
							return errors.New("resident relay correlation bound")
						}
						pending[id] = call{message.Method, bytes.Clone(message.Params)}
						event = srRelayEvent{Phase: "request", Method: message.Method, Params: message.Params}
					}
				} else if !fromBroker && message.IsResponse() {
					if request, ok := pending[id]; ok {
						delete(pending, id)
						event = srRelayEvent{Phase: "response", Method: request.method, Params: request.params, Result: message.Result, Error: message.Error}
					}
					if id == probeID {
						event = srRelayEvent{Phase: "probe", Method: nodepolicy.MethodDispatchResident, Result: message.Result, Error: message.Error}
					}
				}
				mu.Unlock()
			}
			if event.Phase != "" {
				action, err := report(event)
				if err != nil {
					return err
				}
				if action.Action == "kill" {
					return killBoundary
				}
				if action.Probe != nil {
					probe, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": fmt.Sprintf("sr-worker-loss-probe-%d", cmd.Process.Pid), "method": nodepolicy.MethodDispatchResident, "params": action.Probe})
					if err != nil {
						return err
					}
					inputMu.Lock()
					_, err = input.Write(append(probe, '\n'))
					inputMu.Unlock()
					if err != nil {
						return err
					}
				}
				if action.Action == "drop" || event.Phase == "probe" {
					continue
				}
			}
			if fromBroker {
				inputMu.Lock()
			}
			_, err := destination.Write(frame)
			if fromBroker {
				inputMu.Unlock()
			}
			if err != nil {
				return err
			}
		}
		return scanner.Err()
	}
	pumps := make(chan error, 2)
	outputDone := make(chan error, 1)
	go func() { pumps <- forward(os.Stdin, input, true) }()
	go func() {
		err := forward(output, os.Stdout, false)
		outputDone <- err
		pumps <- err
	}()
	var pumpErr error
	select {
	case pumpErr = <-pumps:
	case <-time.After(5 * time.Minute):
		return errors.New("relay lifetime bound")
	}
	forced := errors.Is(pumpErr, killBoundary)
	if forced {
		if err := cmd.Process.Kill(); err != nil {
			return err
		}
	}
	_ = input.Close()
	select {
	case err := <-wait:
		reaped = true
		if !forced && (err != nil || pumpErr != nil) {
			return fmt.Errorf("worker pid=%d or relay pump failed: %w", cmd.Process.Pid, errors.Join(err, pumpErr))
		}
	case <-time.After(8 * time.Second):
		return errors.New("worker did not stop on relay EOF")
	}
	select {
	case err := <-outputDone:
		if err != nil && !forced {
			return fmt.Errorf("worker pid=%d output relay: %w", cmd.Process.Pid, err)
		}
	case <-time.After(2 * time.Second):
		return errors.New("worker output did not drain")
	}
	if _, err := report(srRelayEvent{Phase: "reaped"}); err != nil {
		return err
	}
	return nil
}

type srWorkerRelay struct {
	server            *httptest.Server
	mu                sync.Mutex
	events            []srRelayEvent
	fired             bool
	oldPID            int
	oldGrant          *nodepolicy.DispatchResidentParams
	probed            bool
	release           chan struct{}
	releaseOnce       sync.Once
	completionRelease chan struct{}
	completionOnce    sync.Once
	processes         map[int]*os.Process
	failure           error
	stopping          bool
}

func newSRWorkerRelay(t *testing.T, mode string) *srWorkerRelay {
	t.Helper()
	r := &srWorkerRelay{release: make(chan struct{}), completionRelease: make(chan struct{}), processes: make(map[int]*os.Process)}
	r.server = newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		r.mu.Lock()
		defer r.mu.Unlock()
		fail := func(err error) {
			if r.failure == nil {
				r.failure = err
			}
			w.WriteHeader(http.StatusBadRequest)
		}
		var event srRelayEvent
		if err := neDecode(request.Body, &event); err != nil {
			fail(fmt.Errorf("decode relay event: %w", err))
			return
		}
		if request.Method != http.MethodPost || event.PID <= 0 || event.RelayPID <= 0 || event.PID == event.RelayPID || event.PID == os.Getpid() || event.RelayPID == os.Getpid() || (event.Role != "proxy" && event.Role != "manager") {
			fail(fmt.Errorf("invalid relay event: method=%s event=%+v", request.Method, event))
			return
		}
		if len(r.events) >= 256 {
			fail(errors.New("relay event bound exceeded"))
			return
		}
		r.events = append(r.events, event)
		if event.Phase == "started" {
			for _, pid := range []int{event.RelayPID, event.PID} {
				if _, exists := r.processes[pid]; exists {
					fail(fmt.Errorf("duplicate relay startup pid=%d event=%+v", pid, event))
					return
				}
				process, err := os.FindProcess(pid)
				if err != nil {
					fail(fmt.Errorf("retain relay cleanup pid=%d: %w", pid, err))
					return
				}
				r.processes[pid] = process
			}
		}
		if mode == "manager-completion" && event.Phase == "reaped" && !r.stopping {
			fail(fmt.Errorf("worker reaped before deliberate full-stack shutdown: %+v", event))
			return
		}
		action := srRelayAction{}
		if !r.fired && mode == "proxy-before-grant" && event.Role == "proxy" && event.Phase == "request" && event.Method == nodepolicy.MethodDispatchResident {
			var grant nodepolicy.DispatchResidentParams
			if err := json.Unmarshal(event.Params, &grant); err != nil {
				fail(fmt.Errorf("decode intercepted grant pid=%d: %w", event.PID, err))
				return
			}
			r.fired, r.oldPID, r.oldGrant = true, event.PID, &grant
			action.Action = "kill"
		}
		if !r.fired && mode == "manager-completion" && event.Role == "manager" && event.Phase == "response" && event.Method == nodepolicy.MethodReconcileResident && event.Error == nil {
			var result nodepolicy.ReconcileResidentResult
			if err := json.Unmarshal(event.Result, &result); err != nil {
				fail(fmt.Errorf("decode manager completion pid=%d: %w", event.PID, err))
				return
			}
			if result.Outcome == nodepolicy.ResidentLoaded {
				r.fired, r.oldPID = true, event.PID
				action.Action = "drop"
			}
		}
		if r.oldGrant != nil && !r.probed && event.Role == "proxy" && event.PID != r.oldPID && event.Phase == "response" && event.Method == nodepolicy.MethodPrepareResident && event.Error == nil {
			var prepared nodepolicy.PrepareResidentResult
			if err := json.Unmarshal(event.Result, &prepared); err != nil {
				fail(fmt.Errorf("decode replacement prepare pid=%d: %w", event.PID, err))
				return
			}
			if prepared.ReservationID != "" {
				r.probed = true
				action.Probe = r.oldGrant
			}
		}
		gate := mode == "proxy-before-grant" && r.fired && event.Role == "proxy" && event.PID == r.oldPID && event.Phase == "reaped"
		completionGate := false
		if r.oldGrant != nil && event.Role == "manager" && event.Phase == "response" && event.Method == nodepolicy.MethodReconcileResident {
			var params nodepolicy.ReconcileResidentParams
			if err := json.Unmarshal(event.Params, &params); err != nil {
				fail(fmt.Errorf("decode retired manager operation pid=%d: %w", event.PID, err))
				return
			}
			completionGate = params.OperationID == r.oldGrant.OperationID && params.Reservation.ProxyRunID == r.oldGrant.ProxyRunID
		}
		r.mu.Unlock()
		defer r.mu.Lock()
		var release <-chan struct{}
		if gate {
			release = r.release
		} else if completionGate {
			release = r.completionRelease
		}
		if release != nil {
			select {
			case <-release:
			case <-request.Context().Done():
				return
			case <-time.After(neCallBudget):
				w.WriteHeader(http.StatusGatewayTimeout)
				return
			}
		}
		if err := json.NewEncoder(w).Encode(action); err != nil {
			return
		}
	}))
	t.Cleanup(r.server.Close)
	t.Cleanup(r.resumeCompletion)
	t.Cleanup(r.resume)
	return r
}

func (r *srWorkerRelay) resume() {
	r.releaseOnce.Do(func() { close(r.release) })
}

func (r *srWorkerRelay) resumeCompletion() {
	r.completionOnce.Do(func() { close(r.completionRelease) })
}

// Windows FindProcess retains a process handle, so forced cleanup cannot kill
// a reused PID. The shared executable path also belongs to this test runner and
// the synthetic native runtime: only reported relay/worker handles belong here.
func (r *srWorkerRelay) reap(t *testing.T) {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	for pid, process := range r.processes {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		stopped, err := serviceProcessStopped(ctx, int32(pid))
		cancel()
		if err != nil || !stopped {
			t.Errorf("relay cleanup survivor pid=%d stopped=%t err=%v", pid, stopped, err)
			if err := process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
				t.Errorf("kill retained relay/worker pid=%d: %v", pid, err)
			}
		}
		// These relays run only on Windows, where Wait supports retained handles
		// even when the broker or relay was the process's original parent.
		wait := make(chan error, 1)
		go func() { _, err := process.Wait(); wait <- err }()
		select {
		case err := <-wait:
			if err != nil {
				t.Errorf("reap retained relay/worker pid=%d: %v", pid, err)
			}
		case <-time.After(5 * time.Second):
			t.Errorf("retained relay/worker pid=%d did not stop after kill", pid)
		}
		delete(r.processes, pid)
	}
	if r.failure != nil {
		t.Errorf("relay controller: %v", r.failure)
	}
}

func (r *srWorkerRelay) wait(t *testing.T, description string, match func(srRelayEvent) bool) srRelayEvent {
	t.Helper()
	var found srRelayEvent
	defer func() {
		if found.Phase == "" {
			r.mu.Lock()
			defer r.mu.Unlock()
			t.Logf("unmatched relay wait %q: failure=%v events=%+v", description, r.failure, r.events)
		}
	}()
	neEventually(t, description, 2*neCallBudget, func() bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.failure != nil {
			t.Fatalf("%s: relay controller: %v; events=%+v", description, r.failure, r.events)
		}
		for _, event := range r.events {
			if match(event) {
				found = event
				return true
			}
		}
		return false
	})
	return found
}

func (f *srProcessFixture) startRelayed(t *testing.T) {
	t.Helper()
	if f.p != nil || f.switchTarget {
		t.Fatal("worker-loss relay requires a stopped single-facade fixture")
	}
	f.assertIsolationFiles(t)
	args := append([]string{"--scanner-path", f.scanner, "--engine-manager-path", f.image, "--proxy-path", f.image, "--proxy-engines=llamacpp", "--cluster-dir", filepath.Join(f.root, "cluster"), "--log-level", "info"}, f.ports.args()...)
	env := []string{"OLLAMA_HOST=", "NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + f.scanner, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(f.scannerPort), "NVPAIR_SR_RELAY_IMAGE=" + f.image, "NVPAIR_SR_RELAY_MANAGER=" + engineMgrBin, "NVPAIR_SR_RELAY_PROXY=" + proxyBin, "NVPAIR_SR_RELAY_CONTROL=" + f.relay.server.URL}
	f.relay.mu.Lock()
	f.relay.stopping = false
	f.relay.mu.Unlock()
	f.p = startInteropProcess(t, f.root, brokerBin, f.ports.engineManagerHTTP, env, args...)
	cleanup := f.p.cleanup
	f.p.cleanup = func() {
		f.relay.mu.Lock()
		f.relay.stopping = true
		f.relay.mu.Unlock()
		f.relay.resume()
		f.relay.resumeCompletion()
		cleanup()
		f.relay.reap(t)
	}
	// startInteropProcess also registered its original cleanup. Its stopOnce
	// makes that later call harmless; fatal startup paths still reap our PIDs.
	t.Cleanup(f.p.cleanup)
	f.p.ports = append(f.p.ports, f.ports.engineControl, f.scannerPort, f.facadePort)
	for _, path := range []string{engineMgrBin, proxyBin, f.scanner} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		f.p.paths = append(f.p.paths, canonical)
	}
	waitForMethod(t, f.p.msgs, "app:ready", neCallBudget)
	if port := waitEngineProxyReady(t, "llamacpp-proxy", f.p.in, f.p.msgs, neCallBudget); port != f.facadePort {
		t.Fatalf("relayed proxy port=%d, want reserved %d", port, f.facadePort)
	}
	f.assertIsolation(t)
	t.Cleanup(f.relay.resume)
}

func newSRWorkerLossFixture(t *testing.T, mode string) *srProcessFixture {
	t.Helper()
	f := newSRProcessFixture(t)
	f.stop(t)
	f.relay = newSRWorkerRelay(t, mode)
	// Close the controller only after the broker and both actual children have
	// been reaped, including a fatal assertion at the gated loss boundary.
	t.Cleanup(func() { f.relay.resume(); f.relay.resumeCompletion(); f.stop(t) })
	f.start(t)
	f.call(t, "engine:start", map[string]string{"engine": "llamacpp"}, nil)
	f.retainRuntime(t)
	neEventually(t, "relayed qualified source", neCallBudget, func() bool { return neCapability(f.models(t)) != nil })
	return f
}

func (f *srProcessFixture) control(t *testing.T, action string) srRuntimeState {
	t.Helper()
	status, data, err := f.http(http.MethodGet, f.enginePort, "/__sr/control?action="+action, nil, nil)
	if err != nil || status != http.StatusOK {
		t.Fatalf("SR private control status=%d err=%v", status, err)
	}
	var state srRuntimeState
	if err := neDecode(bytes.NewReader(data), &state); err != nil {
		t.Fatal(err)
	}
	if state.Base.Nonce != f.nonce || state.Base.PID <= 0 {
		t.Fatal("SR runtime authentication failed")
	}
	return state
}

type srPolicyView struct {
	Policy             nodepolicy.Policy              `json:"policy"`
	ActiveRevision     string                         `json:"activeRevision"`
	Activation         string                         `json:"activation"`
	ModelConfiguration *nodepolicy.ModelConfiguration `json:"modelConfiguration"`
}

func (f *srProcessFixture) policy(t *testing.T) srPolicyView {
	t.Helper()
	var view srPolicyView
	f.call(t, "policy:get", nil, &view)
	return view
}

func (f *srProcessFixture) edit(t *testing.T, change func(*nodepolicy.Policy)) srPolicyView {
	t.Helper()
	view := f.policy(t)
	next := view.Policy
	change(&next)
	var accepted srPolicyView
	f.call(t, "policy:set", map[string]any{"policy": next, "expectedPersistedRevision": view.Policy.Revision}, &accepted)
	neEventually(t, "SR policy activation", neCallBudget, func() bool {
		current := f.policy(t)
		return current.ActiveRevision == accepted.Policy.Revision && current.Activation == "active"
	})
	return accepted
}

func (f *srProcessFixture) entry(t *testing.T, model string, state nodepolicy.ModelConfigurationState, reason nodepolicy.ModelConfigurationReason) {
	t.Helper()
	deadline := time.Now().Add(neCallBudget)
	var last nodepolicy.ModelConfigurationEntry
	for {
		view := f.policy(t)
		if view.ModelConfiguration == nil {
			if time.Now().After(deadline) {
				t.Fatalf("SR model configuration %s missing after %s", model, neCallBudget)
			}
			time.Sleep(500 * time.Millisecond)
			continue
		}
		if err := view.ModelConfiguration.Validate(); err != nil {
			t.Fatal(err)
		}
		if view.ModelConfiguration.PolicyRevision != view.Policy.Revision {
			if time.Now().After(deadline) {
				t.Fatalf("SR model configuration %s still belongs to revision %s, want %s", model, view.ModelConfiguration.PolicyRevision, view.Policy.Revision)
			}
			time.Sleep(500 * time.Millisecond)
			continue
		}
		for _, entry := range view.ModelConfiguration.Models {
			if entry.Model == model {
				last = entry
				if entry.State == state && entry.Reason == reason {
					return
				}
			}
		}
		if time.Now().After(deadline) {
			loaded := "unknown"
			if last.ObservedLoaded != nil {
				loaded = strconv.FormatBool(*last.ObservedLoaded)
			}
			t.Fatalf("SR model configuration %s want %s/%s, got %s/%s loaded=%s managerLoaded=%v after %s", model, state, reason, last.State, last.Reason, loaded, f.models(t).Loaded, neCallBudget)
		}
		time.Sleep(500 * time.Millisecond)
	}
}

func (f *srProcessFixture) resident(t *testing.T, models ...string) {
	t.Helper()
	f.edit(t, func(p *nodepolicy.Policy) {
		p.ResidentModels = nil
		for _, model := range models {
			p.ResidentModels = append(p.ResidentModels, nodepolicy.Role{Engine: "llamacpp", Model: model})
		}
	})
}

func (f *srProcessFixture) load(t *testing.T, model string) {
	t.Helper()
	f.call(t, "engine:load-model", map[string]string{"engine": "llamacpp", "model": model}, nil)
	neEventually(t, "SR observed load "+model, neCallBudget, func() bool {
		for _, loaded := range f.models(t).Loaded["llamacpp"] {
			if loaded == model {
				return true
			}
		}
		return false
	})
}

func (f *srProcessFixture) chat(t *testing.T, model string, output int, strict *nodepolicy.StrictAdmissionRequest) (int, []byte) {
	t.Helper()
	body, err := json.Marshal(map[string]any{"model": model, "messages": []any{map[string]string{"role": "user", "content": "fixture"}}, "max_tokens": output, "stream": false})
	if err != nil {
		t.Fatal(err)
	}
	status, data, err := f.http(http.MethodPost, f.facadePort, "/v1/chat/completions", body, strict)
	if err != nil {
		t.Fatalf("SR chat transport: %v", err)
	}
	return status, data
}

func (f *srProcessFixture) strictRequest(t *testing.T, model, requestID string) nodepolicy.StrictAdmissionRequest {
	t.Helper()
	models := f.models(t)
	if models.Policy == nil || models.Policy.StrictAdmission == nil {
		t.Fatal("strict capability absent before permission change")
	}
	for _, capability := range models.Policy.StrictAdmission.Models {
		if capability.Engine == "llamacpp" && capability.Model == model {
			request := nodepolicy.StrictAdmissionRequest{Version: 1, QueueBudgetMs: 0, ColdStartBudgetMs: 0, PreOutputBudgetMs: 2000, Request: nodepolicy.RoutingRequestKey{OriginNodeID: f.snapshot(t).NodeID, ProxyRunID: "sr-fixture-origin", RequestID: requestID}, Identity: capability.Identity}
			if err := request.Validate(); err != nil {
				t.Fatal(err)
			}
			return request
		}
	}
	t.Fatalf("strict capability for %s absent before permission change", model)
	return nodepolicy.StrictAdmissionRequest{}
}

func srAssertNoEffects(t *testing.T, before, after srRuntimeState) {
	t.Helper()
	if before.Base.PID != after.Base.PID || before.Base.Loads != after.Base.Loads || before.Base.Unloads != after.Base.Unloads || before.Base.Chats != after.Base.Chats || before.Content != after.Content || before.Base.Unexpected != after.Base.Unexpected {
		t.Fatalf("refusal caused effects: before=%+v after=%+v", before, after)
	}
}

// Task 602 proves configuration equivalence on the production paired routes.
// Profiles carry fixture-only qualification references without registry evidence:
// desired residency readback is not native loading or SR/NE/B qualification proof.
func TestServedResidentPolicyPairedRoutes(t *testing.T) {
	test := func(name string, reverse bool) {
		t.Run(name, func(t *testing.T) {
			p := newPinnedSwitchFixture(t, reverse)
			mesh, f := p.mesh, p.f
			path := filepath.Join(jointAppDir(f.base), nodepolicy.FileName)
			if !reverse {
				// The fixture blocks the outer upgrade and advertises the ec
				// observer directly. Remove the private gateway owner as well:
				// every successful read/write below must use direct pinned TLS.
				mesh.a.cm.stop()
			} else if mesh.directory(mesh.b).Services[noderec.ServiceEngineControl].Port != mesh.blockedPort || mesh.a.observers[noderec.ServiceCluster].established(mesh.b.uuid) == 0 {
				t.Fatal("reverse case lacks blocked direct ec and an observed pinned outer session")
			}

			local := func(t *testing.T) srPolicyView {
				t.Helper()
				var view srPolicyView
				task2Decode(t, f.rpcOK(t, "policy:get", nil), &view)
				return view
			}
			remote := func(t *testing.T, method string, params any) jsonrpc.Message {
				t.Helper()
				route := "/v1/node-policy/get"
				if method == "engine:remote-policy-set" {
					route = "/v1/node-policy/set"
				}
				before := p.ec.count(mesh.a.uuid, http.MethodPost, route)
				response := task2Response(t, p.origin, p.origin.send(t, method, params, true))
				if actual := p.ec.count(mesh.a.uuid, http.MethodPost, route); actual != before+1 {
					t.Fatalf("authenticated %s dispatches=%d, want exactly one", route, actual-before)
				}
				return response
			}
			assertDesired := func(t *testing.T, configuration *nodepolicy.ModelConfiguration, want nodepolicy.Policy) {
				t.Helper()
				if configuration == nil {
					t.Fatal("model configuration readback absent")
				}
				if err := configuration.Validate(); err != nil {
					t.Fatalf("model configuration readback: %v", err)
				}
				if configuration.PolicyRevision != want.Revision || len(configuration.Models) != len(want.ResidentModels) {
					t.Fatalf("desired residency readback=%+v, want revision=%s residents=%v", configuration, want.Revision, want.ResidentModels)
				}
				for _, entry := range configuration.Models {
					found := false
					for _, ref := range want.ResidentModels {
						if ref.Engine == entry.Engine && ref.Model == entry.Model {
							found = true
						}
					}
					if !found || !entry.DesiredResident {
						t.Fatalf("unexpected desired residency entry: %+v", entry)
					}
				}
			}
			assertPeer := func(t *testing.T, response jsonrpc.Message, want nodepolicy.Policy) {
				t.Helper()
				if response.Error != nil {
					t.Fatalf("paired policy response: %+v", response.Error)
				}
				var view struct {
					Policy             nodepolicy.PeerPolicy          `json:"policy"`
					ModelConfiguration *nodepolicy.ModelConfiguration `json:"modelConfiguration"`
				}
				task2Decode(t, response.Result, &view)
				if !reflect.DeepEqual(view.Policy, want.PeerProjection()) {
					t.Fatalf("paired policy=%+v, want=%+v", view.Policy, want.PeerProjection())
				}
				assertDesired(t, view.ModelConfiguration, want)
			}
			saved := false
			readStored := func(t *testing.T) ([]byte, bool) {
				t.Helper()
				data, err := os.ReadFile(path)
				if errors.Is(err, os.ErrNotExist) && !saved {
					return nil, false
				}
				if err != nil {
					t.Fatalf("read stored policy: %v", err)
				}
				return data, true
			}
			canonicalPolicy := func(t *testing.T, policy nodepolicy.Policy) nodepolicy.Policy {
				t.Helper()
				data, err := json.Marshal(policy)
				if err != nil {
					t.Fatalf("encode configuration control: %v", err)
				}
				parsed, err := nodepolicy.Parse(data)
				if err != nil {
					t.Fatalf("invalid configuration control: %v", err)
				}
				return parsed
			}
			readback := func(t *testing.T, want nodepolicy.Policy) {
				t.Helper()
				view := local(t)
				if !reflect.DeepEqual(view.Policy, want) {
					t.Fatalf("local policy=%+v, want=%+v", view.Policy, want)
				}
				if _, exists := readStored(t); exists {
					assertStoredPolicy(t, path, want)
				} else if want.Revision != "0" {
					t.Fatal("unsaved policy must retain revision zero")
				}
				assertDesired(t, view.ModelConfiguration, want)
				assertPeer(t, remote(t, "engine:remote-policy-get", map[string]string{"nodeId": mesh.b.uuid}), want)
			}
			writeLocal := func(t *testing.T, want nodepolicy.Policy) nodepolicy.Policy {
				t.Helper()
				want = canonicalPolicy(t, want)
				before := local(t).Policy
				var accepted srPolicyView
				task2Decode(t, f.rpcOK(t, "policy:set", map[string]any{"policy": want, "expectedPersistedRevision": before.Revision}), &accepted)
				if accepted.Policy.Revision == "" || accepted.Policy.Revision == before.Revision {
					t.Fatal("local save did not advance revision")
				}
				want.Revision = accepted.Policy.Revision
				if !reflect.DeepEqual(accepted.Policy, want) {
					t.Fatalf("local accepted policy=%+v, want=%+v", accepted.Policy, want)
				}
				saved = true
				assertDesired(t, accepted.ModelConfiguration, accepted.Policy)
				readback(t, accepted.Policy)
				return accepted.Policy
			}

			f.mu.Lock()
			starts := make(map[string][]int32)
			for engine, pids := range f.starts {
				starts[engine] = append([]int32(nil), pids...)
			}
			f.mu.Unlock()
			assertNoEffects := func(t *testing.T) {
				t.Helper()
				f.mu.Lock()
				unchanged := reflect.DeepEqual(f.starts, starts)
				f.mu.Unlock()
				if !unchanged {
					t.Fatal("configuration edit started or restarted a synthetic engine")
				}
				select {
				case <-f.inflight:
					t.Fatal("unqualified configuration edit dispatched synthetic generate/load")
				default:
				}
			}
			model := nodepolicy.Role{Engine: "ollama", Model: "switch-model:latest"}
			neighbor := nodepolicy.Role{Engine: "ollama", Model: srNeighbor + ":latest"}
			var profiles []nodepolicy.Profile
			for _, ref := range []nodepolicy.Role{model, neighbor} {
				profiles = append(profiles, nodepolicy.Profile{
					Name: ref.Model, Engine: ref.Engine, Model: ref.Model,
					ContextTokens: neContext, MaxOutputTokens: neOutput,
					MemoryBytes: neMemory, MaxConcurrent: 1,
					Qualification: &nodepolicy.QualificationRef{ID: "sr-paired-" + ref.Model, Revision: "1"},
				})
			}
			configure := func(name string, served nodepolicy.ServedModels, resident nodepolicy.ResidentModels) {
				t.Run(name, func(t *testing.T) {
					before := local(t).Policy
					want := before
					// The first accepted edit seeds finite profiles for subsequent
					// resident preferences; references alone cannot authorize loads.
					want.Profiles = profiles
					want.ServedModels, want.ResidentModels = served, resident
					want = canonicalPolicy(t, want)
					managementSet(t, mesh.b.cm, mesh.a.uuid, false)
					stored, existed := readStored(t)
					request := map[string]any{"nodeId": mesh.b.uuid, "policy": want.PeerProjection(), "expectedPersistedRevision": before.Revision}
					denied := remote(t, "engine:remote-policy-set", request)
					if denied.Error == nil || denied.Error.Code != -32012 || len(denied.Result) != 0 {
						t.Fatalf("PG denial=%+v, want -32012 without a result", denied)
					}
					if after, exists := readStored(t); exists != existed || !bytes.Equal(after, stored) {
						t.Fatal("PG-denied configuration changed policy file presence or bytes")
					}
					readback(t, before) // Denied peers retain authenticated read access.
					assertNoEffects(t)
					localWant := writeLocal(t, want) // PG does not gate the local owner.
					assertNoEffects(t)
					restored := writeLocal(t, before)
					managementSet(t, mesh.b.cm, mesh.a.uuid, true)
					// Restore the same starting configuration, then perform a real
					// remote change; sending the already-applied local state would
					// only demonstrate preservation, not equivalent mutation.
					request["expectedPersistedRevision"] = restored.Revision
					accepted := remote(t, "engine:remote-policy-set", request)
					if accepted.Error != nil {
						t.Fatalf("PG-allowed configuration: %+v", accepted.Error)
					}
					var result struct {
						Policy nodepolicy.PeerPolicy `json:"policy"`
					}
					task2Decode(t, accepted.Result, &result)
					if result.Policy.Revision == "" || result.Policy.Revision == restored.Revision {
						t.Fatal("paired save did not advance revision")
					}
					localWant.Revision = result.Policy.Revision
					assertPeer(t, accepted, localWant)
					readback(t, localWant)
					assertNoEffects(t)
				})
			}
			configure("explicit empty served permission", nodepolicy.ServedModels{}, nil)
			configure("served subset with desired resident", nodepolicy.ServedModels{model, neighbor}, nodepolicy.ResidentModels{model})
			configure("replace desired resident within served subset", nodepolicy.ServedModels{model, neighbor}, nodepolicy.ResidentModels{neighbor})
			configure("restore unrestricted serving and clear residency", nil, nil)

			if reverse {
				// Positive inner mTLS observations plus this gateway-loss control
				// distinguish reverse dispatch from mere successful policy calls.
				before := local(t).Policy
				count := p.ec.count(mesh.a.uuid, http.MethodPost, "/v1/node-policy/get")
				mesh.a.cm.stop()
				response := task2Response(t, p.origin, p.origin.send(t, "engine:remote-policy-get", map[string]string{"nodeId": mesh.b.uuid}, true))
				if response.Error == nil || len(response.Result) != 0 || p.ec.count(mesh.a.uuid, http.MethodPost, "/v1/node-policy/get") != count {
					t.Fatal("blocked-direct policy read did not depend on the private reverse gateway")
				}
				if !reflect.DeepEqual(local(t).Policy, before) {
					t.Fatal("gateway loss changed local policy")
				}
				assertStoredPolicy(t, path, before)
			}
			assertNoEffects(t)
		})
	}
	test("authenticated direct without reverse gateway", false)
	test("authenticated reverse with blocked direct ec", true)
}

func TestServedResidentProcessPermission(t *testing.T) {
	f := newSRProcessFixture(t)
	neighborRequest := f.strictRequest(t, srNeighbor, "2")
	f.load(t, neModel)
	t.Run("absent served permission accepts self", func(t *testing.T) {
		if f.policy(t).Policy.ServedModels != nil {
			t.Fatal("fixture must begin with absent served permission")
		}
		status, _ := f.chat(t, neModel, neOutput, nil)
		if status != http.StatusOK {
			t.Fatalf("unrestricted self status=%d", status)
		}
	})
	t.Run("explicit empty refuses before effects", func(t *testing.T) {
		request := f.strictRequest(t, neModel, "1")
		f.edit(t, func(p *nodepolicy.Policy) { p.ServedModels = nodepolicy.ServedModels{} })
		before := f.control(t, "")
		status, data := f.chat(t, neModel, neOutput, &request)
		var refusal struct {
			Code string `json:"code"`
		}
		if err := json.Unmarshal(data, &refusal); err != nil {
			t.Fatal(err)
		}
		if status != http.StatusServiceUnavailable || refusal.Code != "strict-admission" {
			t.Fatalf("empty permission status=%d refusal=%+v", status, refusal)
		}
		srAssertNoEffects(t, before, f.control(t, ""))
	})
	t.Run("paired destination enforces current served permission", func(t *testing.T) {
		before := f.control(t, "")
		status, _, data := managementHTTPResponse(t, f.peerClient, f.facadePort, http.MethodPost, "/v1/chat/completions", map[string]any{"model": neModel, "messages": []any{map[string]string{"role": "user", "content": "fixture"}}, "max_tokens": neOutput, "stream": false})
		var refusal struct {
			Reason string `json:"reason"`
		}
		if err := json.Unmarshal(data, &refusal); err != nil {
			t.Fatal(err)
		}
		if status != http.StatusServiceUnavailable || refusal.Reason != "not-served" {
			t.Fatalf("paired destination status=%d reason=%s", status, refusal.Reason)
		}
		srAssertNoEffects(t, before, f.control(t, ""))
	})
	t.Run("subset allows selected model and rejects neighbor", func(t *testing.T) {
		f.edit(t, func(p *nodepolicy.Policy) {
			p.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: neModel}}
		})
		status, _ := f.chat(t, neModel, neOutput, nil)
		if status != http.StatusOK {
			t.Fatalf("subset allowed model status=%d", status)
		}
		before := f.control(t, "")
		status, _ = f.chat(t, srNeighbor, 64, &neighborRequest)
		if status != http.StatusServiceUnavailable {
			t.Fatalf("subset excluded model status=%d", status)
		}
		srAssertNoEffects(t, before, f.control(t, ""))
	})
}

func TestServedResidentProcessResidency(t *testing.T) {
	f := newSRProcessFixture(t)
	t.Run("desired resident loads without inference or content", func(t *testing.T) {
		before := f.control(t, "")
		f.resident(t, neModel)
		f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
		after := f.control(t, "")
		if after.Models[neModel].Loads != before.Models[neModel].Loads+1 || !after.Models[neModel].Loaded || after.Base.Chats != 0 || after.Content != 0 || after.Base.Unexpected != 0 {
			t.Fatalf("resident effects=%+v", after)
		}
	})
	t.Run("declaration removal retains allocation", func(t *testing.T) {
		before := f.control(t, "")
		f.resident(t)
		neEventually(t, "resident declaration removed from current projection", neCallBudget, func() bool {
			view := f.policy(t)
			return view.ModelConfiguration != nil && view.ModelConfiguration.PolicyRevision == view.Policy.Revision && len(view.ModelConfiguration.Models) == 0
		})
		after := f.control(t, "")
		if !after.Models[neModel].Loaded {
			t.Fatal("removing desire unloaded model")
		}
		srAssertNoEffects(t, before, after)
	})
	t.Run("per model finite output refuses before effects", func(t *testing.T) {
		f.load(t, srNeighbor)
		models := f.models(t)
		for i, model := range models.Policy.StrictAdmission.Models {
			limit := neOutput
			if model.Model == srNeighbor {
				limit = 64
			}
			request := nodepolicy.StrictAdmissionRequest{Version: 1, QueueBudgetMs: 0, ColdStartBudgetMs: 0, PreOutputBudgetMs: 2000, Request: nodepolicy.RoutingRequestKey{OriginNodeID: f.snapshot(t).NodeID, ProxyRunID: "sr-bound-origin", RequestID: strconv.Itoa(i + 1)}, Identity: model.Identity}
			before := f.control(t, "")
			status, data := f.chat(t, model.Model, limit+1, &request)
			var refusal struct {
				Code string `json:"code"`
			}
			if err := json.Unmarshal(data, &refusal); err != nil {
				t.Fatal(err)
			}
			if status != http.StatusBadRequest || refusal.Code != "strict-qualification" {
				t.Fatalf("oversized output %s status=%d code=%s", model.Model, status, refusal.Code)
			}
			srAssertNoEffects(t, before, f.control(t, ""))
		}
	})
	t.Run("hard resident is not evicted for neighbor", func(t *testing.T) {
		f.resident(t, neModel)
		f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
		f.call(t, "engine:unload-model", map[string]string{"engine": "llamacpp", "model": srNeighbor}, nil)
		neEventually(t, "neighbor observed unloaded", neCallBudget, func() bool { return !f.control(t, "").Models[srNeighbor].Loaded })
		f.edit(t, func(p *nodepolicy.Policy) {
			p.Admission.MaxResidentModels = 1
			p.Admission.MemoryBudgetBytes = neMemory
			p.Admission.SwitchModels = true
		})
		request := f.strictRequest(t, srNeighbor, "3")
		request.QueueBudgetMs = 100
		request.ColdStartBudgetMs = 1000
		before := f.control(t, "")
		status, _ := f.chat(t, srNeighbor, 64, &request)
		if status != http.StatusTooManyRequests {
			t.Fatalf("protected resident contention status=%d, want occupied refusal", status)
		}
		srAssertNoEffects(t, before, f.control(t, ""))
	})
	t.Run("last model slot admits one request", func(t *testing.T) {
		f.control(t, "hold")
		body := []byte(`{"model":"ne-fixture","messages":[{"role":"user","content":"fixture"}],"max_tokens":128,"stream":false}`)
		type result struct {
			status int
			err    error
		}
		finished := make(chan result, 1)
		go func() {
			status, _, err := f.http(http.MethodPost, f.facadePort, "/v1/chat/completions", body, nil)
			finished <- result{status: status, err: err}
		}()
		defer f.control(t, "release")
		neEventually(t, "first request owns final slot", 3*time.Second, func() bool { return f.control(t, "").HeldChats == 1 })
		before := f.control(t, "")
		request := f.strictRequest(t, neModel, "4")
		request.QueueBudgetMs = 100
		request.PreOutputBudgetMs = 1000
		status, _ := f.chat(t, neModel, neOutput, &request)
		if status != http.StatusTooManyRequests {
			t.Fatalf("contending request status=%d, want occupied refusal", status)
		}
		srAssertNoEffects(t, before, f.control(t, ""))
		f.control(t, "release")
		select {
		case response := <-finished:
			if response.err != nil || response.status != http.StatusOK {
				t.Fatalf("first request status=%d err=%v", response.status, response.err)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("first request did not finish after release")
		}
	})
}

func TestServedResidentProcessCrossEngineLastSlot(t *testing.T) {
	f := newSRSwitchProcessFixture(t)
	// Observe the owned fake's actual launch digest before writing its synthetic
	// qualification. The proxy reads that registry only at process startup.
	f.call(t, "engine:start", map[string]string{"engine": "ollama"}, nil)
	identity := regexp.MustCompile(`engine identity engine=ollama run_id=(run-[0-9a-f]+) launch_digest=([0-9a-f]{64}) models_with_digest=1`)
	var launch string
	neEventually(t, "owned Ollama fixture content identity", neCallBudget, func() bool {
		matches := identity.FindAllStringSubmatch(f.p.output.String(), -1)
		if len(matches) == 0 {
			return false
		}
		launch = matches[len(matches)-1][2]
		return true
	})
	f.edit(t, func(p *nodepolicy.Policy) {
		p.Admission.MaxResidentModels = 1
		// Leave memory for both models so only the cross-engine model cap can
		// explain refusal; a per-engine cap would incorrectly admit the target.
		p.Admission.MemoryBudgetBytes = 2 * neMemory
		p.Admission.SwitchModels = true
		p.Idle.StartOnDemand = true
		p.Profiles = append(p.Profiles, nodepolicy.Profile{Name: "sr-ollama", Engine: "ollama", Model: srOllamaModel, ContextTokens: neContext, MaxOutputTokens: neOutput, MemoryBytes: neMemory, MaxConcurrent: 1, Qualification: &nodepolicy.QualificationRef{ID: "sr-ollama", Revision: "1"}})
	})
	f.stop(t)
	path := filepath.Join(f.app, "local-qualification.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var registry struct {
		Version        int               `json:"version"`
		Qualifications []json.RawMessage `json:"qualifications"`
	}
	if err := neDecode(bytes.NewReader(data), &registry); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256([]byte(srOllamaModel))
	qualification, err := json.Marshal(map[string]any{"id": "sr-ollama", "revision": "1", "engine": "ollama", "model": srOllamaModel, "modelDigests": []string{hex.EncodeToString(digest[:])}, "launchDigests": []string{launch}, "tokenizer": "synthetic-sr-tokenizer", "chatTemplate": "synthetic-sr-template", "contextTokens": neContext, "maxOutputTokens": neOutput, "routes": []string{"/v1/chat/completions"}, "shapes": map[string]any{"tools": false, "structuredOutput": []string{}, "reasoningReplay": false}, "accounting": map[string]any{"method": "utf8-bytes", "requestOverheadTokens": 8, "messageOverheadTokens": 4, "toolOverheadTokens": 0, "schemaOverheadTokens": 0}, "evidence": "Fixture-only cross-engine admission proof. No native Ollama, model, hardware or performance qualification."})
	if err != nil {
		t.Fatal(err)
	}
	registry.Qualifications = append(registry.Qualifications, qualification)
	jointWriteJSON(t, path, registry)
	f.start(t)
	f.call(t, "engine:start", map[string]string{"engine": "llamacpp"}, nil)
	f.retainRuntime(t)
	f.call(t, "engine:start", map[string]string{"engine": "ollama"}, nil)
	var capability nodepolicy.StrictAdmissionModel
	neEventually(t, "qualified Ollama destination after registry restart", neCallBudget, func() bool {
		models := f.models(t)
		if models.Policy != nil && models.Policy.StrictAdmission != nil {
			for _, model := range models.Policy.StrictAdmission.Models {
				if model.Engine == "ollama" && model.Model == srOllamaModel {
					capability = model
					return true
				}
			}
		}
		return false
	})
	f.call(t, "ollama-proxy:node/add-manual", map[string]any{"id": "sr-ollama-self", "host": "127.0.0.1", "port": f.switchFacadePort, "addresses": []string{"127.0.0.1"}, "models": []string{srOllamaModel}}, nil)
	f.call(t, "ollama-proxy:node/select", map[string]string{"id": "sr-ollama-self"}, nil)
	neEventually(t, "healthy Ollama facade", neCallBudget, func() bool {
		status, data, err := f.http(http.MethodGet, f.switchFacadePort, "/", nil, nil)
		return err == nil && status == http.StatusOK && string(data) == "Ollama is running"
	})
	f.resident(t, neModel)
	f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
	neEventually(t, "source resident occupies the global model slot", neCallBudget, func() bool {
		models := f.models(t)
		return reflect.DeepEqual(models.Loaded["llamacpp"], []string{neModel}) && len(models.Loaded["ollama"]) == 0
	})
	targetState := func() srRuntimeState {
		t.Helper()
		status, data, err := f.http(http.MethodGet, f.switchEnginePort, "/__sr/control", nil, nil)
		var state srRuntimeState
		if err != nil || status != http.StatusOK {
			t.Fatalf("private Ollama control status=%d err=%v", status, err)
		}
		if err := neDecode(bytes.NewReader(data), &state); err != nil {
			t.Fatal(err)
		}
		if state.Base.Nonce != f.nonce || state.Base.PID <= 0 {
			t.Fatal("private Ollama runtime authentication failed")
		}
		return state
	}
	targetStarts := func() int {
		f.switchMu.Lock()
		defer f.switchMu.Unlock()
		return len(f.switchStarts)
	}
	source, target := f.control(t, ""), targetState()
	sourceStarts, starts := len(f.starts(t)), targetStarts()
	if source.Models[neModel] != (srEffects{Loaded: true, Loads: 1}) || target.Models[srOllamaModel] != (srEffects{}) || source.Base.Chats != 0 || target.Base.Chats != 0 || source.Content != 0 || target.Content != 0 || source.Base.Unexpected != 0 || target.Base.Unexpected != 0 || starts == 0 {
		t.Fatalf("last-slot fixture must start with only an idle hard resident: source=%+v target=%+v starts=%d", source, target, starts)
	}
	request := nodepolicy.StrictAdmissionRequest{Version: 1, QueueBudgetMs: 100, ColdStartBudgetMs: 1000, PreOutputBudgetMs: 2000, Request: nodepolicy.RoutingRequestKey{OriginNodeID: f.snapshot(t).NodeID, ProxyRunID: "sr-cross-engine-origin", RequestID: "1"}, Identity: capability.Identity}
	if err := request.Validate(); err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"model":"sr-ollama:latest","messages":[{"role":"user","content":"fixture"}],"max_tokens":128,"stream":false}`)
	status, data, err := f.http(http.MethodPost, f.switchFacadePort, "/v1/chat/completions", body, &request)
	if err != nil {
		t.Fatal(err)
	}
	var refusal struct {
		Code  string `json:"code"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(data, &refusal); err != nil {
		t.Fatal(err)
	}
	// AdmitStrict's fitWait path expires the queue as occupied, before cold
	// dispatch; neither an unqualified nor a cold-budget refusal proves this.
	if status != http.StatusTooManyRequests || refusal.Code != "strict-admission" || refusal.Error != "strict admission: occupied" {
		t.Fatalf("cross-engine last slot status=%d refusal=%+v", status, refusal)
	}
	afterSource, afterTarget := f.control(t, ""), targetState()
	srAssertNoEffects(t, source, afterSource)
	srAssertNoEffects(t, target, afterTarget)
	if !reflect.DeepEqual(source.Models, afterSource.Models) || !reflect.DeepEqual(target.Models, afterTarget.Models) || len(f.starts(t)) != sourceStarts || targetStarts() != starts || afterTarget.Base.LeakedHeaders != 0 {
		t.Fatalf("contention changed loaded models, starts or internal headers: source=%+v target=%+v", afterSource, afterTarget)
	}
	f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
	// Removing desire alone retains allocation. Explicitly unload it, and wait
	// for the same inventory owner that strict admission uses to release space.
	f.resident(t)
	f.call(t, "engine:unload-model", map[string]string{"engine": "llamacpp", "model": neModel}, nil)
	neEventually(t, "global model slot released by source readback", neCallBudget, func() bool {
		models := f.models(t)
		return len(models.Loaded["llamacpp"]) == 0 && len(models.Loaded["ollama"]) == 0 && !f.control(t, "").Models[neModel].Loaded
	})
	released := f.control(t, "")
	if released.Models[neModel] != (srEffects{Loads: 1, Unloads: 1}) {
		t.Fatalf("explicit source release effects=%+v", released)
	}
	request.Request.RequestID = "2"
	status, data, err = f.http(http.MethodPost, f.switchFacadePort, "/v1/chat/completions", body, &request)
	if err != nil {
		t.Fatal(err)
	}
	var completion struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &completion); err != nil {
		t.Fatal(err)
	}
	if status != http.StatusOK || completion.Model != srOllamaModel || len(completion.Choices) != 1 || completion.Choices[0].Message.Content != "fixture-only" {
		t.Fatalf("released cross-engine slot status=%d completion=%+v", status, completion)
	}
	afterTarget = targetState()
	if afterTarget.Base.PID != target.Base.PID || afterTarget.Models[srOllamaModel] != (srEffects{Loaded: true, Loads: 1, Chats: 1}) || afterTarget.Base.Loads != 1 || afterTarget.Base.Unloads != 0 || afterTarget.Base.Chats != 1 || afterTarget.Content != 0 || afterTarget.Base.Unexpected != 0 || afterTarget.Base.LeakedHeaders != 0 || targetStarts() != starts || len(f.starts(t)) != sourceStarts {
		t.Fatalf("admissible target effects=%+v", afterTarget)
	}
	finalSource := f.control(t, "")
	// Target admission may repeat the already-completed source unload while
	// reconciling its allocation. It must not reload or infer on that source.
	if finalSource.Base.PID != released.Base.PID || finalSource.Models[neModel].Loaded || finalSource.Base.Loads != released.Base.Loads || finalSource.Base.Chats != released.Base.Chats || finalSource.Content != released.Content || finalSource.Base.Unexpected != released.Base.Unexpected || finalSource.Base.Unloads < released.Base.Unloads || finalSource.Base.Unloads > released.Base.Unloads+1 {
		t.Fatalf("released source changed outside bounded admission unload: before=%+v after=%+v", released, finalSource)
	}
	neEventually(t, "only the cross-engine demand owns the released slot", neCallBudget, func() bool {
		models := f.models(t)
		return len(models.Loaded["llamacpp"]) == 0 && reflect.DeepEqual(models.Loaded["ollama"], []string{srOllamaModel})
	})
}

func TestServedResidentProcessRetention(t *testing.T) {
	f := newSRProcessFixtureWithRetention(t, false, "60")
	before := f.control(t, "")
	f.resident(t, neModel)
	f.entry(t, neModel, nodepolicy.ModelStateBlocked, nodepolicy.ModelReasonUnsupportedLoad)
	srAssertNoEffects(t, before, f.control(t, ""))
}

func TestServedResidentProcessVeto(t *testing.T) {
	f := newSRProcessFixture(t)
	t.Run("local pause blocks resident loading", func(t *testing.T) {
		f.call(t, "node:set-availability", map[string]string{"state": "paused"}, nil)
		before := f.control(t, "")
		f.resident(t, neModel)
		f.entry(t, neModel, nodepolicy.ModelStateBlocked, nodepolicy.ModelReasonPausedLocal)
		srAssertNoEffects(t, before, f.control(t, ""))
	})
	t.Run("resume reconciles without inference", func(t *testing.T) {
		f.call(t, "node:set-availability", map[string]string{"state": "available"}, nil)
		f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
		state := f.control(t, "")
		if state.Base.Loads != 1 || state.Base.Chats != 0 || state.Content != 0 {
			t.Fatalf("resume effects=%+v", state)
		}
	})
	t.Run("paired management denial cannot change desired or loaded models", func(t *testing.T) {
		beforePolicy := f.policy(t).Policy
		before := f.control(t, "")
		peer := beforePolicy.PeerProjection()
		peer.ResidentModels = nil
		peer.ServedModels = nodepolicy.ServedModels{}
		status, _, data := managementHTTPResponse(t, f.peerClient, f.ports.engineControl, http.MethodPost, "/v1/node-policy/set", map[string]any{"policy": peer, "expectedPersistedRevision": beforePolicy.Revision})
		var refusal struct {
			Code string `json:"code"`
		}
		if err := json.Unmarshal(data, &refusal); err != nil {
			t.Fatal(err)
		}
		if status != http.StatusConflict || refusal.Code != "remote_management_denied" {
			t.Fatalf("PG denial status=%d code=%s", status, refusal.Code)
		}
		if !reflect.DeepEqual(f.policy(t).Policy, beforePolicy) {
			t.Fatal("denied peer changed persisted policy")
		}
		srAssertNoEffects(t, before, f.control(t, ""))
	})
	t.Run("local Off blocks desire without restarting engine", func(t *testing.T) {
		f.call(t, "engine:stop", map[string]string{"engine": "llamacpp"}, nil)
		starts := len(f.starts(t))
		f.entry(t, neModel, nodepolicy.ModelStateBlocked, nodepolicy.ModelReasonEngineOffLocal)
		if len(f.starts(t)) != starts {
			t.Fatal("resident desire restarted local Off engine")
		}
	})
}

func TestServedResidentProcessRecovery(t *testing.T) {
	t.Run("full stack restart", func(t *testing.T) {
		f := newSRProcessFixture(t)
		f.resident(t, neModel)
		f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
		before := f.models(t)
		old := neCapability(before)
		if old == nil {
			t.Fatal("qualified pre-restart identity absent")
		}
		f.stop(t)
		f.start(t)
		f.call(t, "engine:start", map[string]string{"engine": "llamacpp"}, nil)
		f.retainRuntime(t)
		f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
		neEventually(t, "new source identity", neCallBudget, func() bool {
			capability := neCapability(f.models(t))
			return capability != nil && capability.Identity.EngineRunID != old.Identity.EngineRunID
		})
		state := f.control(t, "")
		if state.Base.Loads != 1 || state.Base.Chats != 0 || state.Content != 0 || state.Base.Unexpected != 0 {
			t.Fatalf("recovered source effects=%+v", state)
		}
		stale := nodepolicy.StrictAdmissionRequest{Version: 1, QueueBudgetMs: 0, ColdStartBudgetMs: 0, PreOutputBudgetMs: 2000, Request: nodepolicy.RoutingRequestKey{OriginNodeID: f.snapshot(t).NodeID, ProxyRunID: "sr-stale-router", RequestID: "1"}, Identity: old.Identity}
		status, data := f.chat(t, neModel, neOutput, &stale)
		var refusal struct {
			Code  string `json:"code"`
			Error string `json:"error"`
		}
		if err := json.Unmarshal(data, &refusal); err != nil {
			t.Fatal(err)
		}
		if status != http.StatusServiceUnavailable || refusal.Code != "strict-admission" || refusal.Error != "strict admission: unavailable" {
			t.Fatalf("stale router identity status=%d refusal=%+v", status, refusal)
		}
		srAssertNoEffects(t, state, f.control(t, ""))
	})
	t.Run("lost load response is completed by source readback", func(t *testing.T) {
		f := newSRProcessFixture(t)
		f.control(t, "drop-next-load-response")
		f.resident(t, neModel)
		f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
		state := f.control(t, "")
		if state.DropNextLoadResponse || state.InventoryUnavailable || state.Models[neModel].Loads != 1 || !state.Models[neModel].Loaded || state.Base.Chats != 0 || state.Content != 0 || state.Base.Unexpected != 0 {
			t.Fatalf("readback-completed load effects=%+v", state)
		}
	})
	t.Run("lost load response without inventory remains unknown", func(t *testing.T) {
		f := newSRProcessFixture(t)
		f.control(t, "drop-load-and-hide-inventory")
		f.resident(t, neModel)
		neEventually(t, "one dispatched load with unreadable inventory", neCallBudget, func() bool {
			state := f.control(t, "")
			return state.Models[neModel].Loads == 1 && state.InventoryUnavailable && !state.DropNextLoadResponse
		})
		neEventually(t, "unknown completion without readback", neCallBudget, func() bool {
			view := f.policy(t)
			if view.ModelConfiguration == nil || view.ModelConfiguration.PolicyRevision != view.Policy.Revision {
				return false
			}
			for _, entry := range view.ModelConfiguration.Models {
				if entry.Model == neModel {
					if entry.State == nodepolicy.ModelStateSatisfied || entry.ObservedLoaded != nil && *entry.ObservedLoaded {
						t.Fatalf("unreadable inventory falsely satisfied resident: %+v", entry)
					}
					return entry.State == nodepolicy.ModelStateUnknown && entry.Reason == nodepolicy.ModelReasonOperationTimeout
				}
			}
			return false
		})
		state := f.control(t, "")
		if state.Models[neModel].Loads != 1 || state.Base.Chats != 0 || state.Content != 0 || state.Base.Unexpected != 0 {
			t.Fatalf("uncertain load caused extra effects=%+v", state)
		}
		f.control(t, "inventory-available")
		f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
		state = f.control(t, "")
		if state.Models[neModel].Loads != 1 || state.Base.Chats != 0 || state.Content != 0 || state.Base.Unexpected != 0 {
			t.Fatalf("source-bound settlement caused extra effects=%+v", state)
		}
	})
}

func srRelayDecode(t *testing.T, data json.RawMessage, value any) {
	t.Helper()
	if err := json.Unmarshal(data, value); err != nil {
		t.Fatalf("decode resident relay metadata into %T: %v", value, err)
	}
}

func TestServedResidentProcessProxyLossBeforeGrant(t *testing.T) {
	f := newSRWorkerLossFixture(t, "proxy-before-grant")
	before := f.control(t, "")
	f.edit(t, func(p *nodepolicy.Policy) {
		p.Admission.SwitchTimeoutSeconds = 10
		p.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: neModel}}
	})
	lost := f.relay.wait(t, "actual proxy killed and reaped before grant", func(event srRelayEvent) bool {
		return event.Role == "proxy" && event.Phase == "reaped"
	})
	assertSwitchProcessStopped(t, int32(lost.PID))
	f.relay.mu.Lock()
	old := f.relay.oldGrant
	f.relay.mu.Unlock()
	if old == nil {
		t.Fatal("proxy loss did not intercept the final grant request")
	}
	prepared := f.relay.wait(t, "successful correlated prepare on lost proxy", func(event srRelayEvent) bool {
		if event.Role != "proxy" || event.PID != lost.PID || event.Phase != "response" || event.Method != nodepolicy.MethodPrepareResident {
			return false
		}
		var params nodepolicy.PrepareResidentParams
		srRelayDecode(t, event.Params, &params)
		return params.OperationID == old.OperationID && params.ProxyRunID == old.ProxyRunID
	})
	var reservation nodepolicy.PrepareResidentResult
	srRelayDecode(t, prepared.Result, &reservation)
	if prepared.Error != nil || reservation.ReservationID != old.ReservationID || len(reservation.Refusals) != 0 {
		t.Fatalf("loss boundary: error=%+v prepared=%+v intercepted grant=%+v", prepared.Error, reservation, old)
	}
	// This gate delays relay exit after the real proxy has been reaped. The
	// broker may still time out the grant while it is held; it is not evidence
	// that the old manager operation has finished without dispatching.
	srAssertNoEffects(t, before, f.control(t, ""))
	f.relay.resume()
	retired := f.relay.wait(t, "retired operation explicitly completes without dispatch", func(event srRelayEvent) bool {
		if event.Role != "manager" || event.Phase != "response" || event.Method != nodepolicy.MethodReconcileResident {
			return false
		}
		var params nodepolicy.ReconcileResidentParams
		srRelayDecode(t, event.Params, &params)
		return params.OperationID == old.OperationID && params.Reservation.ProxyRunID == old.ProxyRunID
	})
	var notDispatched nodepolicy.ReconcileResidentResult
	srRelayDecode(t, retired.Result, &notDispatched)
	if retired.Error != nil || notDispatched.Outcome != nodepolicy.ResidentNotDispatched || notDispatched.Observation != nil {
		t.Fatalf("retired operation=%s proxy_run=%s error=%+v result=%+v", old.OperationID, old.ProxyRunID, retired.Error, notDispatched)
	}
	// Hold this terminal response before forwarding it to the broker, so a
	// fresh resident operation cannot race the old operation's no-effects check.
	srAssertNoEffects(t, before, f.control(t, ""))
	f.relay.resumeCompletion()
	replacement := f.relay.wait(t, "replacement proxy prepared with a fresh run", func(event srRelayEvent) bool {
		if event.Role != "proxy" || event.PID == lost.PID || event.Phase != "response" || event.Method != nodepolicy.MethodPrepareResident || event.Error != nil {
			return false
		}
		var params nodepolicy.PrepareResidentParams
		var result nodepolicy.PrepareResidentResult
		srRelayDecode(t, event.Params, &params)
		srRelayDecode(t, event.Result, &result)
		return params.ProxyRunID != old.ProxyRunID && result.ReservationID != ""
	})
	// This probes only process scope: an old run/reservation tuple is unknown
	// to a new proxy. It does not isolate same-run revocation or durable holds.
	probe := f.relay.wait(t, "old reservation refused by replacement proxy", func(event srRelayEvent) bool {
		return event.PID == replacement.PID && event.Phase == "probe"
	})
	var refusal nodepolicy.DispatchResidentResult
	srRelayDecode(t, probe.Result, &refusal)
	if probe.Error != nil || refusal.Dispatched || refusal.Reason != nodepolicy.DispatchReasonReservationUnknown {
		t.Fatalf("retired reservation probe pid=%d error=%+v result=%+v old=%+v", probe.PID, probe.Error, refusal, old)
	}
	completed := f.relay.wait(t, "fresh source observation after replacement grant", func(event srRelayEvent) bool {
		if event.Role != "manager" || event.Phase != "response" || event.Method != nodepolicy.MethodReconcileResident || event.Error != nil {
			return false
		}
		var params nodepolicy.ReconcileResidentParams
		var result nodepolicy.ReconcileResidentResult
		srRelayDecode(t, event.Params, &params)
		srRelayDecode(t, event.Result, &result)
		return params.Reservation.ProxyRunID != old.ProxyRunID && result.Outcome == nodepolicy.ResidentLoaded
	})
	var request nodepolicy.ReconcileResidentParams
	var result nodepolicy.ReconcileResidentResult
	srRelayDecode(t, completed.Params, &request)
	srRelayDecode(t, completed.Result, &result)
	if result.Observation == nil || !result.Observation.Loaded || result.Observation.Generation == 0 || result.Observation.ManagerIncarnation != request.Reservation.ManagerIncarnation || result.Observation.Identity != request.Identity {
		t.Fatalf("replacement observation: request=%+v result=%+v observation=%+v", request, result, result.Observation)
	}
	f.relay.mu.Lock()
	for _, event := range f.relay.events {
		if event.Role == "proxy" && event.PID == lost.PID && event.Phase == "response" && event.Method == nodepolicy.MethodDispatchResident {
			f.relay.mu.Unlock()
			t.Fatalf("killed proxy emitted a dispatch response: %+v", event)
		}
	}
	f.relay.mu.Unlock()
	f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
	after := f.control(t, "")
	if after.Base.PID != before.Base.PID || after.Models[neModel].Loads != before.Models[neModel].Loads+1 || after.Base.Chats != 0 || after.Base.Unloads != 0 || after.Content != 0 || after.Base.Unexpected != 0 {
		t.Fatalf("proxy replacement effects: before=%+v after=%+v", before, after)
	}
	// This proves process-scoped refusal and recovery, not durable holds.
}

func TestServedResidentProcessLostManagerCompletion(t *testing.T) {
	f := newSRWorkerLossFixture(t, "manager-completion")
	f.edit(t, func(p *nodepolicy.Policy) {
		p.Admission.SwitchTimeoutSeconds = 10
		p.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: neModel}}
	})
	dropped := f.relay.wait(t, "correlated loaded manager completion suppressed", func(event srRelayEvent) bool {
		if event.Role != "manager" || event.Phase != "response" || event.Method != nodepolicy.MethodReconcileResident || event.Error != nil {
			return false
		}
		var result nodepolicy.ReconcileResidentResult
		srRelayDecode(t, event.Result, &result)
		return result.Outcome == nodepolicy.ResidentLoaded
	})
	var operation nodepolicy.ReconcileResidentParams
	var completion nodepolicy.ReconcileResidentResult
	srRelayDecode(t, dropped.Params, &operation)
	srRelayDecode(t, dropped.Result, &completion)
	if completion.Observation == nil || !completion.Observation.Loaded || completion.Observation.Generation == 0 {
		t.Fatalf("suppressed completion=%+v observation=%+v", completion, completion.Observation)
	}
	grant := f.relay.wait(t, "proxy granted the suppressed manager operation", func(event srRelayEvent) bool {
		if event.Role != "proxy" || event.Phase != "response" || event.Method != nodepolicy.MethodDispatchResident {
			return false
		}
		var params nodepolicy.DispatchResidentParams
		srRelayDecode(t, event.Params, &params)
		return params.OperationID == operation.OperationID && params.ReservationID == operation.Reservation.ReservationID
	})
	var granted nodepolicy.DispatchResidentResult
	srRelayDecode(t, grant.Result, &granted)
	if grant.Error != nil || !granted.Dispatched {
		t.Fatalf("load grant: error=%+v result=%+v operation=%+v", grant.Error, granted, operation)
	}
	uncertain := f.relay.wait(t, "original proxy retains the uncertain dispatched hold", func(event srRelayEvent) bool {
		if event.Role != "proxy" || event.PID != grant.PID || event.Phase != "response" || event.Method != nodepolicy.MethodFinishResident || event.Error != nil {
			return false
		}
		var params nodepolicy.FinishResidentParams
		var result nodepolicy.FinishResidentResult
		srRelayDecode(t, event.Params, &params)
		srRelayDecode(t, event.Result, &result)
		return params.OperationID == operation.OperationID && params.Outcome == nodepolicy.ResidentUnknown && params.Observation == nil && result.RecoveryID != ""
	})
	var held nodepolicy.FinishResidentResult
	srRelayDecode(t, uncertain.Result, &held)
	if held.ReservationID != operation.Reservation.ReservationID || held.Outcome != nodepolicy.ResidentUnknown {
		t.Fatalf("uncertain hold=%+v reservation=%+v", held, operation.Reservation)
	}
	f.entry(t, neModel, nodepolicy.ModelStateUnknown, nodepolicy.ModelReasonOperationTimeout)
	view := f.policy(t)
	if view.ModelConfiguration == nil {
		t.Fatal("uncertain model configuration missing")
	}
	generation := view.ModelConfiguration.ObservationGeneration
	// Require another actual reconciliation, rather than a sleep or a cached
	// status read. Loaded watcher inventory cannot become a completion barrier.
	neEventually(t, "uncertainty survives another reconciliation in the same proxy run", 2*neCallBudget, func() bool {
		current := f.policy(t)
		if current.ModelConfiguration == nil {
			return false
		}
		for _, entry := range current.ModelConfiguration.Models {
			if entry.Model != neModel {
				continue
			}
			if entry.State == nodepolicy.ModelStateSatisfied {
				t.Fatalf("lost terminal completion falsely satisfied resident desire: generation=%d initial=%d entry=%+v", current.ModelConfiguration.ObservationGeneration, generation, entry)
			}
			return current.ModelConfiguration.ObservationGeneration > generation && entry.State == nodepolicy.ModelStateUnknown && entry.Reason == nodepolicy.ModelReasonOperationTimeout && entry.ObservedLoaded != nil && *entry.ObservedLoaded
		}
		return false
	})
	f.entry(t, neModel, nodepolicy.ModelStateUnknown, nodepolicy.ModelReasonOperationTimeout)
	f.relay.mu.Lock()
	retries := 0
	for _, event := range f.relay.events {
		var result nodepolicy.FinishResidentResult
		if (event.Role == "proxy" || event.Role == "manager") && event.Phase == "reaped" {
			f.relay.mu.Unlock()
			t.Fatalf("uncertainty interval crossed worker loss: %+v", event)
		}
		if event.Role == "proxy" && event.PID == grant.PID && event.Phase == "response" && event.Method == nodepolicy.MethodFinishResident && event.Error == nil {
			if err := json.Unmarshal(event.Result, &result); err != nil {
				f.relay.mu.Unlock()
				t.Fatalf("decode finish retry pid=%d: %v", event.PID, err)
			}
			if result == held {
				retries++
			}
		}
	}
	f.relay.mu.Unlock()
	if retries < 2 {
		t.Fatalf("uncertain hold identical retry receipts=%d, want >=2; receipt=%+v", retries, held)
	}
	before := f.control(t, "")
	if before.Models[neModel].Loads != 1 || !before.Models[neModel].Loaded || before.Base.Chats != 0 || before.Base.Unloads != 0 || before.Content != 0 || before.Base.Unexpected != 0 {
		t.Fatalf("uncertain completion effects=%+v", before)
	}
	// Full-stack restart: broker, proxy, manager and synthetic native runtime.
	// This makes no manager-only recovery or durable-hold claim and never
	// clears uncertainty in the old run.
	f.stop(t)
	f.start(t)
	f.call(t, "engine:start", map[string]string{"engine": "llamacpp"}, nil)
	f.retainRuntime(t)
	f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
	recovered := f.relay.wait(t, "replacement manager supplies a fresh loaded observation", func(event srRelayEvent) bool {
		if event.Role != "manager" || event.PID == dropped.PID || event.Phase != "response" || event.Method != nodepolicy.MethodReconcileResident || event.Error != nil {
			return false
		}
		var result nodepolicy.ReconcileResidentResult
		srRelayDecode(t, event.Result, &result)
		return result.Outcome == nodepolicy.ResidentLoaded
	})
	var replacement nodepolicy.ReconcileResidentParams
	var observed nodepolicy.ReconcileResidentResult
	srRelayDecode(t, recovered.Params, &replacement)
	srRelayDecode(t, recovered.Result, &observed)
	if replacement.Reservation.ProxyRunID == operation.Reservation.ProxyRunID || replacement.Reservation.ManagerIncarnation == operation.Reservation.ManagerIncarnation || replacement.Identity.EngineRunID == operation.Identity.EngineRunID || observed.Observation == nil || !observed.Observation.Loaded || observed.Observation.Generation == 0 || observed.Observation.Identity != replacement.Identity || observed.Observation.ManagerIncarnation != replacement.Reservation.ManagerIncarnation {
		t.Fatalf("replacement evidence: old=%+v replacement=%+v result=%+v observation=%+v", operation, replacement, observed, observed.Observation)
	}
	if current := f.policy(t); current.Policy.Revision != view.Policy.Revision || !reflect.DeepEqual(current.Policy.ResidentModels, view.Policy.ResidentModels) {
		t.Fatalf("replacement changed resident desire: before=%+v after=%+v", view.Policy, current.Policy)
	}
	after := f.control(t, "")
	if after.Base.PID == before.Base.PID || after.Models[neModel].Loads != 1 || after.Base.Chats != 0 || after.Base.Unloads != 0 || after.Content != 0 || after.Base.Unexpected != 0 {
		t.Fatalf("full-stack restart native effects: before=%+v after=%+v", before, after)
	}
}

func TestServedResidentProcessSwitch(t *testing.T) {
	f := newSRSwitchProcessFixture(t)
	f.resident(t, neModel)
	f.entry(t, neModel, nodepolicy.ModelStateSatisfied, "")
	// Capture every engine settings baseline before obtaining the CAS revision.
	// A periodic broker refresh must not add an unrelated baseline mid-switch.
	for _, engine := range engines.Names() {
		f.call(t, "engine:get-settings", map[string]string{"engine": engine}, nil)
	}
	source := f.control(t, "hold")
	sourceStarts := len(f.starts(t))
	if source.Base.PID <= 0 || source.Models[neModel].Loads != 1 {
		t.Fatalf("switch source was not resident before drain: %+v", source)
	}
	body := []byte(`{"model":"ne-fixture","messages":[{"role":"user","content":"fixture"}],"max_tokens":128,"stream":false}`)
	completed := make(chan error, 1)
	go func() {
		client := &http.Client{Timeout: 80 * time.Second, Transport: &http.Transport{Proxy: nil}}
		response, err := client.Post(fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", f.facadePort), "application/json", bytes.NewReader(body))
		if err == nil {
			if response.StatusCode != http.StatusOK {
				err = fmt.Errorf("held inference status=%d", response.StatusCode)
			}
			if closeErr := response.Body.Close(); err == nil {
				err = closeErr
			}
		}
		completed <- err
	}()
	neEventually(t, "source inference reached fake runtime", 10*time.Second, func() bool { return f.control(t, "").HeldChats == 1 })
	var status settings.SwitchStatus
	f.call(t, settings.SwitchStatusMethod, settings.SwitchStatusRequest{}, &status)
	request := settings.SwitchRequest{RequestID: "sr-resident-drain", SourceEngine: "llamacpp", TargetEngine: "ollama", ExpectedRevision: status.ConfigurationRevision}
	f.call(t, settings.SwitchMethod, request, nil)
	neEventually(t, "switch entered drain", 40*time.Second, func() bool {
		f.call(t, settings.SwitchStatusMethod, settings.SwitchStatusRequest{RequestID: request.RequestID}, &status)
		if status.Receipt != nil && status.Receipt.Terminal {
			t.Fatalf("switch became terminal before held inference released: %+v", status.Receipt)
		}
		return status.Receipt != nil && status.Receipt.Phase == settings.SwitchDraining
	})
	beforeDrain := f.control(t, "")
	f.entry(t, neModel, nodepolicy.ModelStateBlocked, nodepolicy.ModelReasonSwitchBusy)
	srAssertNoEffects(t, beforeDrain, f.control(t, ""))
	stopped, err := serviceProcessStopped(context.Background(), int32(source.Base.PID))
	if err != nil || stopped || f.control(t, "").HeldChats != 1 {
		t.Fatalf("switch stopped a busy resident source: stopped=%t err=%v", stopped, err)
	}
	f.control(t, "release")
	select {
	case err := <-completed:
		if err != nil {
			t.Fatalf("held inference did not finish: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("held inference did not finish after release")
	}
	neEventually(t, "switch completed after resident request", 45*time.Second, func() bool {
		f.call(t, settings.SwitchStatusMethod, settings.SwitchStatusRequest{RequestID: request.RequestID}, &status)
		return status.Receipt != nil && status.Receipt.Terminal
	})
	receipt := status.Receipt
	if receipt.Phase != settings.SwitchCompleted || receipt.Error != nil || receipt.RecoveryRequired || receipt.Source.Intent != settings.SwitchIntentOff || receipt.Target.Intent != settings.SwitchIntentOn || !receipt.Source.TerminationVerified {
		t.Fatalf("resident switch receipt=%+v", receipt)
	}
	assertSwitchProcessStopped(t, int32(source.Base.PID))
	assertInteropListenerClosed(t, f.enginePort)
	f.switchMu.Lock()
	starts := append([]int32(nil), f.switchStarts...)
	f.switchMu.Unlock()
	if len(starts) != 1 {
		t.Fatalf("target starts=%v, want one", starts)
	}
	stopped, err = serviceProcessStopped(context.Background(), starts[0])
	if err != nil || stopped {
		t.Fatalf("switch target not running: stopped=%t err=%v", stopped, err)
	}
	f.entry(t, neModel, nodepolicy.ModelStateBlocked, nodepolicy.ModelReasonEngineOffLocal)
	if !reflect.DeepEqual(f.policy(t).Policy.ResidentModels, nodepolicy.ResidentModels{{Engine: "llamacpp", Model: neModel}}) {
		t.Fatal("switch changed resident declaration")
	}
	if len(f.starts(t)) != sourceStarts {
		t.Fatal("saved source Off automatically restarted resident engine")
	}
}
