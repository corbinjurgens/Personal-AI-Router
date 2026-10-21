// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

const srRuntimeArg = "--sr-process-runtime"
const srRuntimeEnv = "NVPAIR_SR_PROCESS_EXECUTABLE"
const srNeighbor = "sr-neighbor"

// This dispatch reuses the wrapper's fixed test image, rather than introducing
// another executable path or a native engine. All qualification below is fixture-only.
func init() {
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
	return newSRProcessFixtureWithSwitch(t, false)
}

func newSRSwitchProcessFixture(t *testing.T) *srProcessFixture {
	return newSRProcessFixtureWithSwitch(t, true)
}

func newSRProcessFixtureWithSwitch(t *testing.T, switchTarget bool) *srProcessFixture {
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
		source += fmt.Sprintf("\n[%s]\nmodel = %s\nctx-size = %d\nparallel = 1\nkv-unified = false\nsleep-idle-seconds = -1\n", model, filepath.ToSlash(modelPath), context)
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

// Only the switch case owns a second fake engine. Its facade settings and
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
	runtimeConfig := map[string]any{
		"mode": "process", "bin": f.image, "cli": filepath.Join(f.root, "absent-ollama-cli"), "port": f.switchEnginePort, "bind": "127.0.0.1", "ready": probe, "health": probe,
		"env":  map[string]string{"NVPAIR_TEST_SWITCH_ENGINE": f.image, "NVPAIR_TEST_SWITCH_NAME": "ollama", "NVPAIR_TEST_SWITCH_PORT": "{port}", "NVPAIR_TEST_SWITCH_CONTROL": f.switchControl.URL},
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
	if !f.switchTarget {
		f.neServingFixture.start(t)
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
