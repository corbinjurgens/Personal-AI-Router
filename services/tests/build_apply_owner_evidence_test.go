// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"context"
	"debug/buildinfo"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/buildapply"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/testbin"
)

// TestMain redirects profile variables after its product builds. Retain the
// original toolchain/cache environment for this fixture's later overlay build.
// Product and synthetic children receive only the private launch environment.
var ownerEvidenceBuildEnv = os.Environ()

// Dispatch copied package images before TestMain can build products or replace
// the child's private profile. These selectors exist only in this test package.
func init() {
	for _, role := range []string{"BROKER", "MANAGER", "ENGINE"} {
		if image := os.Getenv("NVPAIR_OWNER_" + role); image != "" && image == os.Args[0] {
			var err error
			if role == "ENGINE" {
				err = runOwnerEvidenceEngine()
			} else {
				err = runOwnerEvidenceBridge(role)
			}
			if err != nil {
				fmt.Fprintln(os.Stderr, "owner-evidence fixture:", err)
				os.Exit(1)
			}
			os.Exit(0)
		}
	}
}

type ownerBridgeStats struct {
	Requests map[string]int `json:"requests"`
	PID      int            `json:"pid"`
}

type ownerHeldReply struct {
	entered   chan buildapply.EngineOwnerEvidence
	release   chan struct{}
	departed  chan struct{}
	delivered chan struct{}
	unblock   func()
}

func newOwnerHeldReply() *ownerHeldReply {
	g := &ownerHeldReply{entered: make(chan buildapply.EngineOwnerEvidence, 1), release: make(chan struct{}), departed: make(chan struct{}), delivered: make(chan struct{})}
	g.unblock = sync.OnceFunc(func() { close(g.release) })
	return g
}

type ownerReplyEvent struct {
	PID   int             `json:"pid"`
	Frame jsonrpc.Message `json:"frame"`
}

type ownerManagerLaunchGate struct {
	entered chan struct{}
	release chan struct{}
	unblock func()
}

// The bridge forwards native frames without replacing results. Its private
// manager /call route compares raw producer data. Service-chain assertions use
// the service test child's real production owner and private broker transport.
func runOwnerEvidenceBridge(role string) (resultErr error) {
	port, err := strconv.Atoi(os.Getenv("NVPAIR_OWNER_" + role + "_PORT"))
	if err != nil || validateFixturePort(port) != nil {
		return errors.New("bridge requires an allocated private port")
	}
	image := os.Getenv("NVPAIR_OWNER_" + role + "_REAL")
	if !filepath.IsAbs(image) || image == os.Args[0] {
		return errors.New("bridge requires its distinct fixed real image")
	}
	args := slices.Clone(os.Args[1:])
	if role == "MANAGER" {
		// The unused LAN model-list surface is wildcard-only. The existing
		// flags disable that listener and automatic engine watcher sweeps.
		args = append(args, "--http-port=0", "--loaded-poll-interval=0")
	}
	cmd := exec.Command(image, args...)
	if role == "BROKER" {
		// Production resolves optional default workers only in cwd. Keep it
		// beside the bridge's deliberately sparse, enumerated service bundle,
		// even though the native image lives among TestMain's full products.
		cmd.Dir = filepath.Dir(os.Args[0])
	}
	cmd.Env, cmd.Stderr = os.Environ(), os.Stderr
	in, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return err
	}
	defer listener.Close()
	if role == "MANAGER" {
		client := &http.Client{Timeout: 2 * time.Minute, Transport: &http.Transport{Proxy: nil}}
		defer client.CloseIdleConnections()
		response, err := client.Post(os.Getenv("NVPAIR_OWNER_CONTROL")+"/manager-start", "application/json", bytes.NewReader([]byte(`{}`)))
		if err != nil {
			return err
		}
		status := response.StatusCode
		if err := response.Body.Close(); err != nil {
			return err
		}
		if status != http.StatusOK {
			return errors.New("private manager launch gate refused")
		}
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	reaped := false
	defer func() {
		if !reaped {
			closeErr := in.Close()
			killErr := cmd.Process.Kill()
			if errors.Is(killErr, os.ErrProcessDone) {
				killErr = nil
			}
			resultErr = errors.Join(resultErr, closeErr, killErr, cmd.Wait())
		}
	}()
	var mu, writeMu, forwardMu sync.Mutex
	stats := ownerBridgeStats{Requests: map[string]int{}, PID: cmd.Process.Pid}
	pending := map[string]chan *jsonrpc.Message{}
	methods := map[string]string{}
	var sequence int
	encoder := json.NewEncoder(in)
	write := func(frame *jsonrpc.Message) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		mu.Lock()
		stats.Requests[frame.Method]++
		if frame.ID != nil && frame.Method != "" {
			methods[string(*frame.ID)] = frame.Method
		}
		mu.Unlock()
		return encoder.Encode(frame)
	}
	stopped := make(chan struct{})
	defer close(stopped)
	// Hold ordinary broker installation refreshes until the never-observed
	// capture finishes. Forward the original requests after release, without
	// inventing responses or blocking unrelated native owner reads.
	installedRelease := make(chan struct{})
	releaseInstalled := sync.OnceFunc(func() { close(installedRelease) })
	mux := http.NewServeMux()
	mux.HandleFunc("/release-installed", func(w http.ResponseWriter, r *http.Request) {
		if role != "MANAGER" || r.Method != http.MethodPost {
			http.Error(w, "manager release only", http.StatusBadRequest)
			return
		}
		releaseInstalled()
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("/stats", func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if err := json.NewEncoder(w).Encode(stats); err != nil {
			return
		}
	})
	mux.HandleFunc("/call", func(w http.ResponseWriter, r *http.Request) {
		var frame jsonrpc.Message
		if r.Method != http.MethodPost || json.NewDecoder(io.LimitReader(r.Body, jsonrpc.WorkerFrameBytes)).Decode(&frame) != nil ||
			role != "MANAGER" || frame.Method != buildapply.EngineOwnerEvidenceMethod {
			http.Error(w, "private read only", http.StatusBadRequest)
			return
		}
		mu.Lock()
		sequence++
		// Public clients use numeric service IDs. This string namespace never
		// escapes into the service's pending map or consumes its replies.
		id := json.RawMessage(strconv.Quote(fmt.Sprintf("owner-observer-%d", sequence)))
		ch := make(chan *jsonrpc.Message, 1)
		pending[string(id)] = ch
		mu.Unlock()
		defer func() { mu.Lock(); delete(pending, string(id)); mu.Unlock() }()
		frame.JSONRPC, frame.ID = "2.0", &id
		if err := write(&frame); err != nil {
			http.Error(w, "native transport unavailable", http.StatusServiceUnavailable)
			return
		}
		select {
		case reply := <-ch:
			if err := json.NewEncoder(w).Encode(reply); err != nil {
				return
			}
		case <-stopped:
			http.Error(w, "native child departed", http.StatusServiceUnavailable)
		case <-r.Context().Done():
		}
	})
	server := &http.Server{Handler: mux, ReadHeaderTimeout: serviceCallTimeout}
	defer server.Close()
	done := make(chan error, 3)
	go func() { done <- server.Serve(listener) }()
	go func() {
		codec := jsonrpc.NewCodecMaxFrame(struct {
			io.Reader
			io.Writer
		}{os.Stdin, io.Discard}, jsonrpc.WorkerFrameBytes)
		for {
			frame, err := codec.Read()
			if err != nil {
				done <- err
				return
			}
			if role == "MANAGER" && frame.Method == "engine:get-installed" {
				go func(frame *jsonrpc.Message) {
					select {
					case <-installedRelease:
						if err := write(frame); err != nil {
							done <- err
						}
					case <-stopped:
					}
				}(frame)
				continue
			}
			if err := write(frame); err != nil {
				done <- err
				return
			}
		}
	}()
	go func() {
		codec := jsonrpc.NewCodecMaxFrame(struct {
			io.Reader
			io.Writer
		}{out, io.Discard}, jsonrpc.WorkerFrameBytes)
		forward := json.NewEncoder(os.Stdout)
		forwardFrame := func(frame *jsonrpc.Message) error {
			forwardMu.Lock()
			defer forwardMu.Unlock()
			return forward.Encode(frame)
		}
		for {
			frame, err := codec.Read()
			if err != nil {
				done <- err
				return
			}
			var ch chan *jsonrpc.Message
			var method string
			if frame.IsResponse() {
				mu.Lock()
				ch = pending[string(*frame.ID)]
				method = methods[string(*frame.ID)]
				delete(methods, string(*frame.ID))
				mu.Unlock()
				// A cancelled private read remains private even after retirement.
				if strings.HasPrefix(string(*frame.ID), `"owner-observer-`) {
					if ch != nil {
						ch <- frame
					}
					continue
				}
			}
			if role == "MANAGER" && method == buildapply.EngineOwnerEvidenceMethod {
				// Holding a native response must not hide EOF. Native death still
				// exits this wrapper and retires the broker's actual worker pipe.
				go func(frame *jsonrpc.Message) {
					data, err := json.Marshal(ownerReplyEvent{PID: cmd.Process.Pid, Frame: *frame})
					if err != nil {
						done <- err
						return
					}
					client := &http.Client{Timeout: 2 * time.Minute, Transport: &http.Transport{Proxy: nil}}
					defer client.CloseIdleConnections()
					response, err := client.Post(os.Getenv("NVPAIR_OWNER_CONTROL")+"/manager-reply", "application/json", bytes.NewReader(data))
					if err != nil {
						done <- err
						return
					}
					status := response.StatusCode
					if err := response.Body.Close(); err != nil {
						done <- err
						return
					}
					if status != http.StatusOK {
						done <- errors.New("native reply observation refused")
						return
					}
					if err := forwardFrame(frame); err != nil {
						done <- err
						return
					}
					delivered, err := client.Post(os.Getenv("NVPAIR_OWNER_CONTROL")+"/manager-delivered", "application/json", bytes.NewReader(data))
					if err != nil {
						done <- err
						return
					}
					if err := delivered.Body.Close(); err != nil {
						done <- err
						return
					}
					if delivered.StatusCode != http.StatusOK {
						done <- errors.New("native delivery acknowledgement refused")
					}
				}(frame)
				continue
			}
			if err := forwardFrame(frame); err != nil {
				done <- err
				return
			}
		}
	}()
	if err := <-done; err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	if err := in.Close(); err != nil {
		return err
	}
	err = cmd.Wait()
	reaped = true
	return err
}

// A pure Go fake records every launch and every HTTP contact at the parent.
// No model bytes, credential store, inference or child processes exist.
func runOwnerEvidenceEngine() error {
	control := os.Getenv("NVPAIR_OWNER_CONTROL")
	client := &http.Client{Timeout: 2 * time.Minute, Transport: &http.Transport{Proxy: nil}}
	defer client.CloseIdleConnections()
	record := func(path string) error {
		response, err := client.Post(control+path, "application/json", bytes.NewReader([]byte(`{}`)))
		if err != nil {
			return err
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			return errors.New("fake observation rejected")
		}
		return nil
	}
	if len(os.Args) == 2 && os.Args[1] == "pull" {
		if err := record("/pull"); err != nil {
			return err
		}
		return errors.New("controlled synthetic download failure")
	}
	port, err := strconv.Atoi(os.Getenv("NVPAIR_OWNER_ENGINE_PORT"))
	if err != nil || validateFixturePort(port) != nil {
		return errors.New("fake engine requires an allocated private port")
	}
	if err := record("/started"); err != nil {
		return err
	}
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return err
	}
	defer listener.Close()
	return http.Serve(listener, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := record("/contact"); err != nil {
			http.Error(w, "observation unavailable", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"status":"ok","data":[]}`)
	}))
}

type ownerEvidenceProcessFixture struct {
	*serviceFixture
	brokerURL, managerURL    string
	serviceURL               string
	managerImage, fake       string
	enginePorts              map[string]int
	client                   *http.Client
	mu                       sync.Mutex
	starts, contacts         int
	pullEntered, pullRelease chan struct{}
	releasePull              func()
	heldReply                *ownerHeldReply
	managerLaunch            *ownerManagerLaunchGate
	deliveries               map[string]*ownerHeldReply
	allocatedPorts           map[int]bool
}

type ownerServiceEvidence struct {
	ManagerIncarnation  string            `json:"managerIncarnation"`
	ManagerCaptureToken string            `json:"managerCaptureToken"`
	Impact              buildapply.Impact `json:"impact"`
	ProtectedRoots      []string          `json:"protectedRoots"`
	Facts               struct {
		ManagerIdleRunID         bool `json:"managerIdleRunId"`
		ModelDataRoots           bool `json:"modelDataRoots"`
		DownloadCopyReservations bool `json:"downloadCopyReservations"`
	} `json:"facts"`
	Saved        bool                           `json:"saved"`
	Enforced     bool                           `json:"enforced"`
	PhysicalStop bool                           `json:"physicalStop"`
	OwnedWorkers []buildapply.RunnerReservation `json:"ownedWorkers"`
}

type ownerServiceObservation struct {
	Profile           string                     `json:"profile"`
	ProductionOwner   bool                       `json:"productionOwner"`
	Evidence          *ownerServiceEvidence      `json:"evidence"`
	Envelope          *buildapply.ImpactEnvelope `json:"envelope"`
	ReadRefused       bool                       `json:"readRefused"`
	ReadError         string                     `json:"readError"`
	PreflightRefused  bool                       `json:"preflightRefused"`
	PreflightError    string                     `json:"preflightError"`
	PendingOwnerReads int                        `json:"pendingOwnerReads"`
	PreviewRetained   bool                       `json:"previewRetained"`
	Effects           map[string]int             `json:"effects"`
}

// The executing test image is authoritative: GOFLAGS alone misses a command-line
// -race and can also contain an overridden value. Explicitly apply this image's
// setting to every native owner build, then inspect the output without running it.
func ownerEvidenceBuildRace(t *testing.T) bool {
	t.Helper()
	info, ok := debug.ReadBuildInfo()
	if !ok {
		t.Fatal("owner fixture test image has no build settings")
	}
	for _, setting := range info.Settings {
		if setting.Key == "-race" {
			race, err := strconv.ParseBool(setting.Value)
			if err != nil {
				t.Fatalf("test image race build setting: %v", err)
			}
			return race
		}
	}
	return false
}

func (f *ownerEvidenceProcessFixture) buildOwnerImage(t *testing.T, module, output string, race bool, args ...string) {
	t.Helper()
	if err := f.stageExecutable(func() error {
		buildArgs := append(slices.Clone(args), "-race="+strconv.FormatBool(race), "-o", output, ".")
		cmd := exec.Command("go", buildArgs...)
		cmd.Dir = filepath.Join("..", module)
		cmd.Env = ownerEvidenceBuildEnv
		cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
		return cmd.Run()
	}); err != nil {
		t.Fatalf("compile fixed %s image: %v", module, err)
	}
	info, err := buildinfo.ReadFile(output)
	if err != nil {
		t.Fatalf("inspect fixed %s build settings: %v", module, err)
	}
	gotRace := false
	for _, setting := range info.Settings {
		if setting.Key == "-race" {
			gotRace, err = strconv.ParseBool(setting.Value)
			if err != nil {
				t.Fatalf("%s race build setting: %v", module, err)
			}
		}
	}
	if gotRace != race {
		t.Fatalf("%s race instrumentation=%t, test image=%t", module, gotRace, race)
	}
	t.Logf("owner_native_build module=%s image=%q race=%t", module, output, gotRace)
}

func ownerEvidenceServiceOverlay(t *testing.T, dir string) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	replacements := map[string]string{}
	for _, sourceFile := range []struct {
		name  string
		sites []struct{ signature, effect string }
	}{
		{"service.go", []struct{ signature, effect string }{{"func (s *Service) stop() {", "shutdown"}}},
		{"build_apply_run.go", []struct{ signature, effect string }{
			{"func (owner serviceBuildApplyEvidence) Read(ctx context.Context) (buildApplyEvidence, error) {", "reads"},
			{"func (files *nativeBuildApplyFiles) Rename(ctx context.Context, from, to string) error {", "swap"},
			{"func launchBuildApplyProcess(ctx context.Context, launch buildApplyLaunch) (buildapply.RunnerReservation, error) {", "runner"},
		}},
	} {
		source, err := filepath.Abs(filepath.Join("..", "nvpair-service", sourceFile.name))
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(source)
		if err != nil {
			t.Fatal(err)
		}
		text := strings.ReplaceAll(string(data), "\r\n", "\n")
		for _, site := range sourceFile.sites {
			if strings.Count(text, site.signature) != 1 {
				t.Fatalf("service effect observation site changed: %s", site.signature)
			}
			text = strings.Replace(text, site.signature, site.signature+"\n\townerEvidenceObserveEffect("+strconv.Quote(site.effect)+")", 1)
		}
		if sourceFile.name == "service.go" {
			text += "\nvar ownerEvidenceObserveEffect = func(string) {}\n"
		}
		target := filepath.Join(dir, sourceFile.name)
		if err := os.WriteFile(target, []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
		replacements[source] = target
	}
	// go test can also compile the ordinary package while constructing its test
	// variant. Keep overlay production files self-contained; bind the observer
	// from this package's internal test file only in the selected child.
	testSource, err := filepath.Abs(filepath.Join("..", "nvpair-service", "build_apply_preview_test.go"))
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(testSource)
	if err != nil {
		t.Fatal(err)
	}
	text := strings.ReplaceAll(string(data), "\r\n", "\n")
	const binding = "func bindBuildApplyOwnerProcessObserver() {}"
	if strings.Count(text, binding) != 1 {
		t.Fatal("service test observer binding site changed")
	}
	text = strings.Replace(text, binding, "func bindBuildApplyOwnerProcessObserver() { ownerEvidenceObserveEffect = buildApplyOwnerProcessEffect }", 1)
	testTarget := filepath.Join(dir, "build_apply_preview_test.go")
	if err := os.WriteFile(testTarget, []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
	replacements[testSource] = testTarget
	path := filepath.Join(dir, "overlay.json")
	jointWriteJSON(t, path, map[string]map[string]string{"Replace": replacements})
	return path
}

// Generated overlay sources are disposable test inputs outside the checkout.
// Like managerFixtureOverlay in engine-manager/main_test.go, this confines
// registry and credential I/O to private fixtures. Initialization is observed;
// the descendantless fake's signal boundary uses its retained process handle.
// Evidence production, manager identity and lifecycle authority stay real.
func ownerEvidenceManagerOverlay(t *testing.T, dir, fake string) string {
	t.Helper()
	replacements := map[string]string{}
	replace := func(name, site, replacement string) {
		t.Helper()
		source, err := filepath.Abs(filepath.Join("..", "nvpair-engine-manager", name))
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(source)
		if err != nil {
			t.Fatal(err)
		}
		text := strings.ReplaceAll(string(data), "\r\n", "\n")
		if strings.Count(text, site) != 1 {
			t.Fatalf("owner fixture overlay site changed in %s", name)
		}
		target := filepath.Join(dir, name)
		if err := os.WriteFile(target, []byte(strings.Replace(text, site, replacement, 1)), 0o600); err != nil {
			t.Fatal(err)
		}
		replacements[source] = target
	}
	replace("main.go", "reg := buildRegistry(manifestDir)", `reg := NewRegistry()
	if err := reg.LoadOverrideDir(manifestDir); err != nil { log.Fatalf("private registry: %v", err) }`)
	replace("credentialstore.go", "var newCredentialBackend = func() credentialBackend { return secretstore.New() }", `var newCredentialBackend = func() credentialBackend { return &ownerFixtureCredentials{} }
type ownerFixtureCredentials struct{}
func (*ownerFixtureCredentials) Set(context.Context, string, string, []byte) error { return secretstore.ErrNotFound }
func (*ownerFixtureCredentials) Get(context.Context, string, string) ([]byte, error) { return nil, secretstore.ErrNotFound }
func (*ownerFixtureCredentials) Delete(context.Context, string, string) error { return secretstore.ErrNotFound }`)
	// This observation is after actual lazy creation, not an inferred absence
	// from PID snapshots or a substituted owner-evidence result.
	replace("executor.go", "e.engines[engine] = st", fmt.Sprintf(`e.engines[engine] = st
	if err := os.WriteFile(filepath.Join(%q, "initialized-"+engine), []byte("created"), 0600); err != nil { panic(err) }`, dir))
	// The pure Go fake never has descendants. Terminate that exact retained
	// process handle and join the real managedProc.done channel, avoiding an
	// unenumerated taskkill child. No owner result or stop proof is substituted.
	// The ordinary lifecycle owner still saves Off, observes exit and emits it.
	replace("proc.go", `force := grace <= 0
	_ = signalPIDContext(ctx, mp.cmd.Process.Pid, force)
	if err := ctx.Err(); err != nil {
		return err
	}
	if !force {
		timer := time.NewTimer(grace)
		defer timer.Stop()
		select {
		case <-mp.done:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
		}
		_ = signalPIDContext(ctx, mp.cmd.Process.Pid, true)
	}
	select {
	case <-mp.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}`, fmt.Sprintf(`if !strings.EqualFold(filepath.Clean(mp.cmd.Path), filepath.Clean(%q)) { return os.ErrPermission }
	if err := mp.cmd.Process.Kill(); err != nil { return err }
	select { case <-mp.done: return nil; case <-ctx.Done(): return ctx.Err() }`, fake))
	path := filepath.Join(dir, "overlay.json")
	jointWriteJSON(t, path, map[string]map[string]string{"Replace": replacements})
	return path
}

func newOwnerEvidenceProcessFixture(t *testing.T) *ownerEvidenceProcessFixture {
	t.Helper()
	if runtime.GOOS != "windows" || !filepath.IsAbs(os.Getenv(testbin.EnvRoot)) {
		t.Fatal("owner process proof requires Windows and the fixed testbin root; compile only elsewhere")
	}
	f := &ownerEvidenceProcessFixture{serviceFixture: newServiceFixture(t), enginePorts: map[string]int{},
		pullEntered: make(chan struct{}, 1), pullRelease: make(chan struct{}), deliveries: map[string]*ownerHeldReply{}, allocatedPorts: map[int]bool{}}
	args := f.ports.args()
	for i := 1; i < len(args); i += 2 {
		port, err := strconv.Atoi(args[i])
		if err != nil {
			t.Fatal(err)
		}
		f.allocatedPorts[port] = true
	}
	f.defaultClusterProfile = true
	f.releasePull = sync.OnceFunc(func() { close(f.pullRelease) })
	f.client = &http.Client{Timeout: 20 * time.Second, Transport: &http.Transport{Proxy: nil}}
	t.Cleanup(f.client.CloseIdleConnections)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	f.copyExecutable(self, f.scannerPath)
	f.env = ownerEvidencePrivateEnv(f.env)
	f.env = append(f.env, "NVPAIR_TEST_SUBSCRIPTION_SCANNER="+f.scannerPath,
		"NVPAIR_TEST_SUBSCRIPTION_PORT="+strconv.Itoa(f.allocatePort(t)))
	control := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/manager-delivered" {
			var event ownerReplyEvent
			if err := json.NewDecoder(io.LimitReader(r.Body, jsonrpc.WorkerFrameBytes)).Decode(&event); err != nil || event.Frame.ID == nil {
				http.Error(w, "invalid delivery", http.StatusBadRequest)
				return
			}
			key := fmt.Sprintf("%d/%s", event.PID, string(*event.Frame.ID))
			f.mu.Lock()
			gate := f.deliveries[key]
			delete(f.deliveries, key)
			f.mu.Unlock()
			if gate != nil {
				close(gate.delivered)
			}
			return
		}
		if r.URL.Path == "/manager-start" {
			f.mu.Lock()
			gate := f.managerLaunch
			f.managerLaunch = nil
			f.mu.Unlock()
			if gate != nil {
				close(gate.entered)
				select {
				case <-gate.release:
				case <-r.Context().Done():
				}
			}
			return
		}
		if r.URL.Path == "/manager-reply" {
			var event ownerReplyEvent
			if err := json.NewDecoder(io.LimitReader(r.Body, jsonrpc.WorkerFrameBytes)).Decode(&event); err != nil {
				http.Error(w, "invalid native reply", http.StatusBadRequest)
				return
			}
			var evidence buildapply.EngineOwnerEvidence
			if event.Frame.Error != nil {
				return
			}
			if err := json.Unmarshal(event.Frame.Result, &evidence); err != nil {
				http.Error(w, "invalid native document", http.StatusBadRequest)
				return
			}
			f.mu.Lock()
			gate := f.heldReply
			f.heldReply = nil
			f.mu.Unlock()
			if gate != nil {
				if event.Frame.ID == nil {
					http.Error(w, "native response has no request identity", http.StatusBadRequest)
					return
				}
				key := fmt.Sprintf("%d/%s", event.PID, string(*event.Frame.ID))
				f.mu.Lock()
				f.deliveries[key] = gate
				f.mu.Unlock()
				gate.entered <- evidence
				select {
				case <-gate.release:
				case <-r.Context().Done():
					f.mu.Lock()
					delete(f.deliveries, key)
					f.mu.Unlock()
					close(gate.departed)
				}
			}
			return
		}
		if r.URL.Path == "/pull" {
			select {
			case f.pullEntered <- struct{}{}:
			default:
			}
			select {
			case <-f.pullRelease:
			case <-r.Context().Done():
			}
			return
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		switch r.URL.Path {
		case "/started":
			f.starts++
		case "/contact":
			f.contacts++
		default:
			http.Error(w, "unknown observation", http.StatusNotFound)
		}
	}))
	t.Cleanup(control.Close)
	controlPort := portOfURL(t, control.URL)
	if f.allocatedPorts[controlPort] {
		t.Fatal("private controller collided with a reserved fixture port")
	}
	f.allocatedPorts[controlPort] = true
	t.Cleanup(f.releasePull)
	// All children reuse previously enumerated destinations. execSpawner starts
	// the broker bridge in binDir; it preserves that cwd for the real broker.
	// Missing default siblings disable optional workers. Explicit missing paths
	// would instead be fatal in broker main and must never be passed.
	bridge := filepath.Join(testbin.Dir(t, "tests", "model-settings", "model-settings-manager"), testbin.Exe("model-settings-manager"))
	f.fake = filepath.Join(testbin.Dir(t, "tests", "policy-joint", "joint-origin-engine"), testbin.Exe("joint-origin-engine"))
	f.managerImage = filepath.Join(f.binDir, testbin.Exe("nvpair-engine-manager"))
	for _, path := range []string{bridge, f.fake, brokerBin} {
		canonical, err := serviceExecutablePath(path)
		if err != nil {
			t.Fatal(err)
		}
		f.inventory.bundlePaths = append(f.inventory.bundlePaths, canonical)
	}
	bridge = f.copyExecutable(self, bridge)
	f.fake = f.copyExecutable(self, f.fake)
	f.copyExecutable(self, f.brokerPath)
	race := ownerEvidenceBuildRace(t)
	// TestMain's goBuild does not propagate a command-line -race. Rebuild this
	// fixture's real broker at its existing registered destination before launch.
	f.buildOwnerImage(t, "nvpair-ui-broker", brokerBin, race, "build")
	overlay := ownerEvidenceManagerOverlay(t, f.dir, f.fake)
	f.buildOwnerImage(t, "nvpair-engine-manager", f.managerImage, race, "build", "-overlay", overlay)
	serviceOverlay := ownerEvidenceServiceOverlay(t, filepath.Join(f.dir, "service-overlay"))
	f.buildOwnerImage(t, "nvpair-service", f.servicePath, race, "test", "-c", "-overlay", serviceOverlay)
	f.inventory.paths = append(f.inventory.paths, f.managerImage)
	brokerPort, managerPort, servicePort := f.allocatePort(t), f.allocatePort(t), f.allocatePort(t)
	f.brokerURL = fmt.Sprintf("http://127.0.0.1:%d", brokerPort)
	f.managerURL = fmt.Sprintf("http://127.0.0.1:%d", managerPort)
	f.serviceURL = fmt.Sprintf("http://127.0.0.1:%d", servicePort)
	f.env = append(f.env, "NVPAIR_OWNER_BROKER="+f.brokerPath, "NVPAIR_OWNER_BROKER_REAL="+brokerBin,
		"NVPAIR_OWNER_BROKER_PORT="+strconv.Itoa(brokerPort), "NVPAIR_OWNER_MANAGER="+bridge,
		"NVPAIR_OWNER_MANAGER_REAL="+f.managerImage, "NVPAIR_OWNER_MANAGER_PORT="+strconv.Itoa(managerPort),
		"NVPAIR_OWNER_ENGINE="+f.fake, "NVPAIR_OWNER_CONTROL="+control.URL,
		"NVPAIR_OWNER_SERVICE="+f.servicePath, "NVPAIR_OWNER_SERVICE_PORT="+strconv.Itoa(servicePort))
	f.additionalBrokerArgs = []string{"--engine-manager-path", bridge, "--engine-manager-control-host", "127.0.0.1"}
	for _, program := range []string{"nvpair-node-info", "nvpair-proxy", "nvpair-workload-manager", "nvpair-errors", "nvpair-manual-nodes", "nvpair-node-settings", "nvpair-cluster-manager", "nvpair-job-scheduler"} {
		path := filepath.Join(f.binDir, testbin.Exe(program))
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("optional worker must be an absent default sibling, %q: %v", path, err)
		}
	}
	return f
}

// Publish only while the previous native manager is stopped, before its next
// process incarnation reads this private profile. The service itself stays up.
func (f *ownerEvidenceProcessFixture) installManifests(t *testing.T) {
	t.Helper()
	manifestDir := filepath.Join(jointAppDir(f.configDir), "engines")
	for _, engine := range []string{"owner-never", "owner-owned", "owner-external"} {
		port := f.allocatePort(t)
		f.enginePorts[engine] = port
		probe := map[string]any{"http": "http://127.0.0.1:{port}/health", "status": http.StatusOK, "timeout_s": 5}
		rt := map[string]any{"mode": "process", "bin": f.fake, "bind": "127.0.0.1", "port": port,
			// Readiness proves the contact counter is live. Omit the optional
			// periodic health probe so only the evidence read is under test.
			"env": map[string]string{"NVPAIR_OWNER_ENGINE_PORT": "{port}"}, "ready": probe,
			"stop": map[string]any{"signal": "term", "grace_s": 1}}
		jointWriteJSON(t, filepath.Join(manifestDir, engine+".json"), map[string]any{"engine": engine, "display_name": engine, "manifest_version": 1, "model_naming": "exact",
			"operations": map[string]any{"pull": map[string]string{"action": "pull_model", "model_param": "name"}},
			"actions":    map[string]any{"pull_model": map[string]any{"cmd": []string{f.fake, "pull"}}},
			"platforms": map[string]any{runtime.GOOS + "/" + runtime.GOARCH: map[string]any{
				"detect": []string{f.fake}, "models_dir": filepath.Join(f.configDir, "models", engine), "runtime": rt}}})
	}
	// Private preexisting uncertainty is input to the real durable owner. It is
	// not a forged evidence result or a claim of historical completion.
	jointWriteJSON(t, filepath.Join(jointAppDir(f.configDir), "engine-bin", "model-download-state.json"), map[string]any{
		"version": 1, "entries": []map[string]string{{"id": strings.Repeat("d", 32), "engine": "owner-owned",
			"model": "predecessor-download", "contentKey": "predecessor-download", "naming": "exact"}},
		"copies": []any{}, "copiesComplete": false, "copyRecoveryRequired": false,
	})
}

func (f *ownerEvidenceProcessFixture) allocatePort(t *testing.T) int {
	t.Helper()
	for attempt := 0; attempt < 32; attempt++ {
		port := freePort(t)
		if !f.allocatedPorts[port] {
			f.allocatedPorts[port] = true
			return port
		}
	}
	t.Fatal("fixture could not allocate a distinct private port")
	return 0
}

// Overlay builds use ownerEvidenceBuildEnv. Products receive no ambient proxy,
// engine, credential, preset or test-child selectors, and no installed CLI PATH.
func ownerEvidencePrivateEnv(env []string) []string {
	var clean []string
	for _, entry := range env {
		key, _, ok := strings.Cut(entry, "=")
		upper := strings.ToUpper(key)
		if !ok || upper == "PATH" || strings.Contains(upper, "PROXY") || strings.Contains(upper, "TOKEN") || strings.Contains(upper, "SECRET") ||
			strings.Contains(upper, "PASSWORD") || strings.Contains(upper, "KEY") || strings.Contains(upper, "CREDENTIAL") || strings.Contains(upper, "AUTH") ||
			(strings.HasPrefix(upper, "NVPAIR_") && strings.Contains(upper, "TEST")) || strings.HasPrefix(upper, "NVPAIR_OWNER_") || strings.HasPrefix(upper, "FAKE_") ||
			strings.HasPrefix(upper, "LLAMA") || strings.HasPrefix(upper, "OLLAMA") || strings.HasPrefix(upper, "STRATA") ||
			strings.HasPrefix(upper, "LMSTUDIO") || strings.HasPrefix(upper, "PAIR_BUILD_APPLY") || strings.HasPrefix(upper, "NVPAIR_BUILD_APPLY_") ||
			strings.HasPrefix(upper, "AWS_") || strings.HasPrefix(upper, "AZURE_") || strings.HasPrefix(upper, "GOOGLE_") || strings.HasPrefix(upper, "HF_") {
			continue
		}
		clean = append(clean, entry)
	}
	return append(clean, "PATH=")
}

func (f *ownerEvidenceProcessFixture) readNativeOwner(t *testing.T) (*jsonrpc.Message, error) {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, f.managerURL+"/call", bytes.NewReader(mustMarshal(t, jsonrpc.Message{
		JSONRPC: "2.0", Method: buildapply.EngineOwnerEvidenceMethod, Params: json.RawMessage(`{}`),
	})))
	if err != nil {
		return nil, err
	}
	response, err := f.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bridge read HTTP %d", response.StatusCode)
	}
	var frame jsonrpc.Message
	err = json.NewDecoder(io.LimitReader(response.Body, jsonrpc.WorkerFrameBytes)).Decode(&frame)
	return &frame, err
}

func (f *ownerEvidenceProcessFixture) impact(t *testing.T) buildapply.ImpactEnvelope {
	t.Helper()
	deadline := time.NewTimer(30 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
	var lastErr error
	for {
		observation, err := f.requestObservation(t.Context(), "/impact")
		if err == nil && observation.ProductionOwner && !observation.ReadRefused {
			if result := observation.Envelope; result != nil && result.ManagerEvidence.Known && result.ManagerEvidence.Value != nil {
				owner := result.ManagerEvidence.Value
				if owner.Universe.Known && owner.StopAuthority.Known && owner.Reservations.Known {
					return *result
				}
			}
		}
		lastErr = err
		select {
		case <-deadline.C:
			t.Fatalf("native broker/manager owner capture unavailable: %v", lastErr)
		case <-tick.C:
		}
	}
}

func (f *ownerEvidenceProcessFixture) requestObservation(ctx context.Context, path string) (ownerServiceObservation, error) {
	var observation ownerServiceObservation
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, f.serviceURL+path, nil)
	if err != nil {
		return observation, err
	}
	response, err := f.client.Do(request)
	if err != nil {
		return observation, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return observation, fmt.Errorf("service observer HTTP %d", response.StatusCode)
	}
	err = json.NewDecoder(io.LimitReader(response.Body, buildapply.MaxStatusBytes)).Decode(&observation)
	return observation, err
}

func (f *ownerEvidenceProcessFixture) observe(t *testing.T, path string) ownerServiceObservation {
	t.Helper()
	observation, err := f.requestObservation(t.Context(), path)
	if err != nil {
		t.Fatalf("real service %s: %v", path, err)
	}
	if !observation.ProductionOwner {
		t.Fatal("service did not retain its production owner")
	}
	return observation
}

// Read both real service boundaries while the owned/adopted fake can receive
// HTTP. Pre-start zero counters cannot prove that a read avoids native probes.
func (f *ownerEvidenceProcessFixture) readWhileEngineListening(t *testing.T, engine string) buildapply.ImpactEnvelope {
	t.Helper()
	address := fmt.Sprintf("127.0.0.1:%d", f.enginePorts[engine])
	waitTCP(t, address, serviceCallTimeout)
	f.mu.Lock()
	starts, contacts := f.starts, f.contacts
	f.mu.Unlock()
	if starts == 0 || contacts == 0 {
		t.Fatal("live fake engine has no recorded launch/readiness contact")
	}
	impact := f.impact(t)
	read := f.observe(t, "/read")
	if !read.ReadRefused || read.Evidence != nil || read.ReadError != "manager root coverage is unavailable" {
		t.Fatalf("production owner did not preserve unknown root coverage: %+v", read)
	}
	f.mu.Lock()
	afterStarts, afterContacts := f.starts, f.contacts
	f.mu.Unlock()
	if afterStarts != starts || afterContacts != contacts {
		t.Fatalf("evidence read launched/contacted live %s: starts %d -> %d, contacts %d -> %d", engine, starts, afterStarts, contacts, afterContacts)
	}
	waitTCP(t, address, serviceCallTimeout)
	t.Logf("owner_evidence_live_reads engine=%s starts_before=%d starts_after=%d contacts_before=%d contacts_after=%d", engine, starts, afterStarts, contacts, afterContacts)
	f.assertNoApplyEffects(t)
	return impact
}

func (f *ownerEvidenceProcessFixture) assertNoApplyEffects(t *testing.T) {
	t.Helper()
	observation := f.observe(t, "/stats")
	if observation.PendingOwnerReads != 0 || observation.PreviewRetained {
		t.Fatal("service retained a completed owner request or preview")
	}
	for _, kind := range []string{"shutdown", "swap", "runner"} {
		if observation.Effects[kind] != 0 {
			t.Fatalf("read/refused operation invoked real service %s %d times", kind, observation.Effects[kind])
		}
	}
}

func (f *ownerEvidenceProcessFixture) replaceManager(t *testing.T, configure func()) buildapply.ImpactEnvelope {
	t.Helper()
	previous := *f.impact(t).ManagerEvidence.Value
	stats := f.stats(t, f.managerURL)
	gate := &ownerManagerLaunchGate{entered: make(chan struct{}), release: make(chan struct{})}
	gate.unblock = sync.OnceFunc(func() { close(gate.release) })
	t.Cleanup(gate.unblock)
	f.mu.Lock()
	f.managerLaunch = gate
	f.mu.Unlock()
	var native serviceProcessIdentity
	for _, identity := range f.scan() {
		if int(identity.PID) == stats.PID && identity.Path == f.managerImage {
			native = identity
		}
	}
	if native.PID == 0 {
		t.Fatal("native predecessor manager lacks its fixed-image identity")
	}
	if err := f.inventory.kill(native); err != nil {
		t.Fatal(err)
	}
	select {
	case <-gate.entered:
	case <-time.After(30 * time.Second):
		t.Fatal("replacement did not reach its before-launch gate")
	}
	if configure != nil {
		configure()
	}
	gate.unblock()
	next := f.impact(t)
	if next.ManagerEvidence.Value.ManagerIncarnation == previous.ManagerIncarnation {
		t.Fatal("native manager incarnation survived process replacement")
	}
	return next
}

func (f *ownerEvidenceProcessFixture) stats(t *testing.T, base string) ownerBridgeStats {
	t.Helper()
	response, err := f.client.Get(base + "/stats")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var stats ownerBridgeStats
	if response.StatusCode != http.StatusOK {
		t.Fatalf("bridge stats HTTP %d", response.StatusCode)
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 64*1024)).Decode(&stats); err != nil {
		t.Fatal(err)
	}
	return stats
}

func ownerEvidenceEntry(t *testing.T, e buildapply.EngineOwnerEvidence, engine string) (buildapply.EngineOwnerUniverseEntry, buildapply.EngineOwnerStopAuthority) {
	t.Helper()
	if !e.Universe.Known || e.Universe.Value == nil || !e.StopAuthority.Known || e.StopAuthority.Value == nil {
		t.Fatal("native universe or authority unavailable")
	}
	if len(*e.Universe.Value) != len(*e.StopAuthority.Value) {
		t.Fatal("native authority and universe lengths differ")
	}
	for i, entry := range *e.Universe.Value {
		if entry.Engine == engine {
			authority := (*e.StopAuthority.Value)[i]
			if authority.Engine != engine {
				t.Fatal("native authority not paired with universe")
			}
			return entry, authority
		}
	}
	t.Fatalf("native universe omitted %s", engine)
	return buildapply.EngineOwnerUniverseEntry{}, buildapply.EngineOwnerStopAuthority{}
}

// One serial service lifetime. All replacement, reservation and refusal reads
// below use the real service's private owner/transport, never a public relay of
// engine:owner-evidence or an acceptance receipt assembled by this fixture.
func TestBuildApplyOwnerEvidenceProcess(t *testing.T) {
	if runtime.GOOS != "windows" || !filepath.IsAbs(os.Getenv(testbin.EnvRoot)) || !strings.Contains(os.Getenv("GOFLAGS"), "nvpair-test-exec") {
		t.Fatal("owner process proof requires PC3's Windows fixed-path wrapper setup; compile only elsewhere")
	}
	firewall := readOwnerFirewallSnapshot(t)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	actual, err := serviceExecutablePath(self)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := serviceExecutablePath(filepath.Join(os.Getenv(testbin.EnvRoot), "go-test", "tests", "tests.test.exe"))
	if err != nil || actual != expected {
		t.Fatalf("test runner escaped its enumerated fixed destination: %q %v", self, err)
	}
	requireOwnerFirewallRules(t, firewall, []string{self})
	before, err := snapshotInstalledServiceProcesses()
	if err != nil {
		t.Fatal(err)
	}
	f := newOwnerEvidenceProcessFixture(t)
	requireOwnerFirewallRules(t, firewall, append(f.inventory.ownedPaths(), self))
	t.Cleanup(func() {
		if err := f.cleanupChecked(); err != nil {
			t.Error(err)
			return
		}
		after, err := snapshotInstalledServiceProcesses()
		if err != nil {
			t.Error(err)
			return
		}
		if !reflect.DeepEqual(before, after) {
			t.Errorf("installed identity/listener snapshot changed: before=%+v after=%+v", before, after)
		}
	})
	f.start()
	c := f.readyClient()
	serviceLifetime := serviceStatus(t, c)
	brokerLifetime := f.stats(t, f.brokerURL).PID
	profile := f.observe(t, "/stats").Profile
	if !filepath.IsAbs(profile) {
		t.Fatal("service child did not report its canonical private profile")
	}
	f.configDir = profile
	f.clusterDir = filepath.Join(jointAppDir(profile), "cluster")
	for _, key := range []string{"HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME"} {
		f.env = append(f.env, key+"="+profile)
	}
	t.Run("production service idle owner reads native empty manager without initialization", func(t *testing.T) {
		raw := f.impact(t)
		owner := raw.ManagerEvidence.Value
		if owner.Universe.Value == nil || len(*owner.Universe.Value) != 0 {
			t.Fatal("initial fixture registry is not actually empty")
		}
		observation := f.observe(t, "/read")
		e := observation.Evidence
		if observation.ReadRefused || e == nil || e.ManagerIncarnation != owner.ManagerIncarnation || e.ManagerCaptureToken != owner.CaptureToken || !e.Facts.ManagerIdleRunID || !e.Facts.ModelDataRoots || !e.Facts.DownloadCopyReservations {
			t.Fatal("production idle read did not preserve the native manager identity and fresh baseline")
		}
		if e.Saved || e.Enforced || e.PhysicalStop || e.OwnedWorkers != nil || e.Impact.Complete {
			t.Fatal("production idle projection invented unrelated owner authority")
		}
		for _, root := range e.ProtectedRoots {
			rel, err := filepath.Rel(profile, root)
			if err != nil || !filepath.IsAbs(root) || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
				t.Fatalf("service protected root escaped canonical fixture profile: %q %v", root, err)
			}
		}
		initialized, err := filepath.Glob(filepath.Join(f.dir, "initialized-*"))
		if err != nil || len(initialized) != 0 {
			t.Fatalf("idle evidence initialized an engine: %v %v", initialized, err)
		}
		f.mu.Lock()
		starts, contacts := f.starts, f.contacts
		f.mu.Unlock()
		if starts != 0 || contacts != 0 {
			t.Fatalf("idle evidence effects: starts=%d contacts=%d", starts, contacts)
		}
		f.assertNoApplyEffects(t)
	})
	t.Run("real late manager response cannot resurrect a retired service owner read", func(t *testing.T) {
		previous := f.observe(t, "/read")
		if previous.ReadRefused || previous.Evidence == nil {
			t.Fatal("idle owner unavailable before late reply")
		}
		gate := newOwnerHeldReply()
		t.Cleanup(gate.unblock)
		f.mu.Lock()
		f.heldReply = gate
		f.mu.Unlock()
		type result struct {
			observation ownerServiceObservation
			err         error
		}
		finished := make(chan result, 1)
		ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
		defer cancel()
		go func() { observation, err := f.requestObservation(ctx, "/read"); finished <- result{observation, err} }()
		select {
		case native := <-gate.entered:
			if native.ManagerIncarnation != previous.Evidence.ManagerIncarnation {
				t.Fatal("late response lacks the actual source incarnation")
			}
		case <-ctx.Done():
			t.Fatal("native reply did not enter delivery gate")
		}
		// Completion comes from the real bounded owner read while its genuine
		// producer response is still held, not an arbitrary sleep or RPC error
		// manufactured by the observer.
		select {
		case old := <-finished:
			if old.err != nil || !old.observation.ReadRefused || old.observation.Evidence != nil {
				t.Fatalf("held read did not retire: %v %+v", old.err, old.observation)
			}
		case <-ctx.Done():
			t.Fatal("bounded service owner read did not finish")
		}
		if f.observe(t, "/stats").PendingOwnerReads != 0 {
			t.Fatal("timed-out service owner read retained a pending entry")
		}
		gate.unblock()
		select {
		case <-gate.delivered:
		case <-ctx.Done():
			t.Fatal("genuine late response was not written to the old request's live worker pipe")
		}
		before := f.stats(t, f.managerURL).Requests[buildapply.EngineOwnerEvidenceMethod]
		current := f.observe(t, "/read")
		after := f.stats(t, f.managerURL).Requests[buildapply.EngineOwnerEvidenceMethod]
		if current.ReadRefused || current.Evidence == nil || current.Evidence.ManagerIncarnation != previous.Evidence.ManagerIncarnation || after < before+2 {
			t.Fatal("late response replaced a fresh production read or poisoned its native source")
		}
		f.assertNoApplyEffects(t)
	})
	t.Run("held native predecessor reply cannot satisfy service after manager replacement", func(t *testing.T) {
		previous := f.observe(t, "/read")
		if previous.ReadRefused || previous.Evidence == nil {
			t.Fatal("idle predecessor read unavailable")
		}
		gate := newOwnerHeldReply()
		t.Cleanup(gate.unblock)
		f.mu.Lock()
		f.heldReply = gate
		f.mu.Unlock()
		type result struct {
			observation ownerServiceObservation
			err         error
		}
		finished := make(chan result, 1)
		ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
		defer cancel()
		go func() { observation, err := f.requestObservation(ctx, "/read"); finished <- result{observation, err} }()
		select {
		case native := <-gate.entered:
			if native.ManagerIncarnation != previous.Evidence.ManagerIncarnation {
				t.Fatal("held response did not originate at the actual predecessor")
			}
		case <-time.After(serviceCallTimeout):
			t.Fatal("native manager reply did not enter the delivery gate")
		}
		if pending := f.observe(t, "/stats").PendingOwnerReads; pending != 1 {
			t.Fatalf("held real owner read pending=%d", pending)
		}
		next := f.replaceManager(t, nil)
		select {
		case old := <-finished:
			if old.err != nil || !old.observation.ReadRefused || old.observation.Evidence != nil {
				t.Fatalf("retired native reply was accepted or retained: error=%v observation=%+v", old.err, old.observation)
			}
		case <-ctx.Done():
			t.Fatal("service did not retire predecessor read")
		}
		// The predecessor transport really departed with a genuine response
		// held at delivery. No frame is reassigned to the successor's pipe.
		select {
		case <-gate.departed:
		case <-ctx.Done():
			t.Fatal("old manager reply transport survived its producer")
		}
		gate.unblock()
		current := f.observe(t, "/read")
		if current.ReadRefused || current.Evidence == nil || current.Evidence.ManagerIncarnation != next.ManagerEvidence.Value.ManagerIncarnation || current.Evidence.ManagerIncarnation == previous.Evidence.ManagerIncarnation {
			t.Fatal("late predecessor contaminated the current production owner read")
		}
		f.assertNoApplyEffects(t)
	})
	f.replaceManager(t, func() { f.installManifests(t) })
	var first buildapply.ImpactEnvelope
	t.Run("idle native broker manager read preserves never observed unknown", func(t *testing.T) {
		initializedBefore, err := filepath.Glob(filepath.Join(f.dir, "initialized-*"))
		if err != nil {
			t.Fatal(err)
		}
		first = f.impact(t)
		e := *first.ManagerEvidence.Value
		second := f.impact(t)
		if second.ManagerEvidence.Value.ManagerIncarnation != e.ManagerIncarnation || second.ManagerEvidence.Value.CaptureToken != e.CaptureToken {
			t.Fatal("stable idle captures changed native manager identity or evidence token")
		}
		frame, err := f.readNativeOwner(t)
		if err != nil || frame.Error != nil {
			t.Fatalf("direct native owner read: frame=%+v error=%v", frame, err)
		}
		var native buildapply.EngineOwnerEvidence
		if err := json.Unmarshal(frame.Result, &native); err != nil {
			t.Fatal(err)
		}
		if native.ManagerIncarnation != e.ManagerIncarnation || native.CaptureToken != e.CaptureToken {
			t.Fatal("broker source fence differs from the actual native owner document")
		}
		if len(e.ManagerIncarnation) != 32 || e.CaptureToken == "" || e.Complete || first.Impact.Complete {
			t.Fatal("idle capture lacks native identity or claims completeness")
		}
		entry, authority := ownerEvidenceEntry(t, e, "owner-never")
		if !entry.Registered || !entry.Supported.Known || entry.Supported.Value == nil || !*entry.Supported.Value ||
			entry.Installed.Known || entry.Running.Known || entry.TerminationVerified.Known || authority.Scoped.Known || entry.RootsComplete {
			t.Fatal("never-observed supported entry fabricated lifecycle, stop authority or root coverage")
		}
		var names []string
		for _, member := range *e.Universe.Value {
			names = append(names, member.Engine)
			for _, roots := range []buildapply.Observation[[]string]{member.ModelRoots, member.DataRoots} {
				if !roots.Known || roots.Value == nil {
					t.Fatal("explicit private fixture roots disappeared")
				}
				for _, root := range *roots.Value {
					rel, err := filepath.Rel(f.configDir, root)
					if err != nil || !filepath.IsAbs(root) || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
						t.Fatalf("native root escaped private profile: %q error=%v", root, err)
					}
				}
			}
		}
		slices.Sort(names)
		if !slices.Equal(names, []string{"owner-external", "owner-never", "owner-owned"}) {
			t.Fatalf("compiled registry was not restricted to private manifests: %v", names)
		}
		if !e.Reservations.Known || e.Reservations.Value == nil || e.Reservations.Value.UnsafeCopiesComplete {
			t.Fatal("missing historical copy baseline became safe or disappeared")
		}
		initializedAfter, err := filepath.Glob(filepath.Join(f.dir, "initialized-*"))
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(initializedBefore, initializedAfter) {
			t.Fatal("idle evidence lazily initialized an engine")
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		if f.starts != 0 || f.contacts != 0 {
			t.Fatalf("idle read launched/contacted fake engines: starts=%d contacts=%d", f.starts, f.contacts)
		}
	})
	if first.ManagerEvidence.Value == nil {
		t.Fatal("idle capture prerequisite failed")
	}
	t.Run("unknown roots refuse real service preflight without effects", func(t *testing.T) {
		read := f.observe(t, "/read")
		if !read.ReadRefused || read.Evidence != nil || read.ReadError != "manager root coverage is unavailable" {
			t.Fatalf("production owner did not refuse unknown roots: %+v", read)
		}
		preflight := f.observe(t, "/preflight")
		if !preflight.PreflightRefused || preflight.PreflightError != read.ReadError || preflight.Effects["reads"] <= read.Effects["reads"] {
			t.Fatalf("preflight did not refuse at its production root boundary: %+v", preflight)
		}
		f.assertNoApplyEffects(t)
		// O2 leaves every nonempty private registry's RootsComplete false.
		// serviceBuildApplyEvidence.Read calls buildApplyManagerRoots first,
		// before projecting reservation facts or checking completeness in
		// captureBuildApplyPlan. These cases therefore prove genuine reservation
		// provenance and operation_busy through the real service impact read,
		// separately from this zero-effect unknown-roots preflight refusal.
		t.Log("UNMET/deferred: isolated service preflight reservation refusal; buildApplyManagerRoots refuses unknown root coverage first under O2, before reservation admission can be isolated")
	})
	response, err := f.client.Post(f.managerURL+"/release-installed", "application/json", bytes.NewReader([]byte(`{}`)))
	if err != nil {
		t.Fatalf("release native installation refresh: %v", err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusNoContent {
		t.Fatalf("installation refresh release status = %d", response.StatusCode)
	}
	t.Run("public relay does not disclose private owner documents", func(t *testing.T) {
		for i, method := range []string{buildapply.EngineOwnerEvidenceMethod, "build:apply-impact"} {
			frame, err := c.call(strconv.Itoa(800+i), method, json.RawMessage(`{}`), serviceCallTimeout)
			if err != nil || frame.Error == nil || frame.Error.Code != -32601 || len(frame.Result) != 0 {
				t.Fatalf("private relay method %s: frame=%+v error=%v", method, frame, err)
			}
		}
	})
	t.Run("owned fake has a retained process lifetime", func(t *testing.T) {
		c.mustCall(t, "810", "engine:start", mustMarshal(t, map[string]string{"engine": "owner-owned"}))
		e := *f.readWhileEngineListening(t, "owner-owned").ManagerEvidence.Value
		entry, authority := ownerEvidenceEntry(t, e, "owner-owned")
		if !entry.Running.Known || entry.Running.Value == nil || !*entry.Running.Value ||
			!authority.Scoped.Known || authority.Scoped.Value == nil || !*authority.Scoped.Value ||
			!authority.Adopted.Known || authority.Adopted.Value == nil || *authority.Adopted.Value ||
			!authority.ProcessLifetime.Known || authority.ProcessLifetime.Value == nil || authority.ProcessLifetime.Value.EngineRunID == "" {
			t.Fatal("owned fake lacks native scoped process authority")
		}
		lifetime := authority.ProcessLifetime.Value
		if !lifetime.PID.Known || lifetime.PID.Value == nil || !entry.EngineRunID.Known || entry.EngineRunID.Value == nil || lifetime.EngineRunID != *entry.EngineRunID.Value {
			t.Fatal("native owned lifetime has no paired PID and engine run")
		}
		found := slices.ContainsFunc(f.scan(), func(identity serviceProcessIdentity) bool {
			return int(identity.PID) == *lifetime.PID.Value && identity.Path == f.fake
		})
		if !found {
			t.Fatal("owner's retained process is not the fixed synthetic image")
		}
		c.mustCall(t, "811", "engine:stop", mustMarshal(t, map[string]string{"engine": "owner-owned"}))
	})
	t.Run("external fake adoption does not grant process stop authority", func(t *testing.T) {
		cmd := exec.Command(f.fake)
		cmd.Env = append(slices.Clone(f.env), "NVPAIR_OWNER_ENGINE_PORT="+strconv.Itoa(f.enginePorts["owner-external"]))
		child := f.launchChild(cmd)
		// A real start performs the deliberate adoption. Evidence reads never
		// call Status, Detect or a native HTTP endpoint to fill missing facts.
		waitTCP(t, fmt.Sprintf("127.0.0.1:%d", f.enginePorts["owner-external"]), serviceCallTimeout)
		c.mustCall(t, "812", "engine:start", mustMarshal(t, map[string]string{"engine": "owner-external"}))
		entry, authority := ownerEvidenceEntry(t, *f.readWhileEngineListening(t, "owner-external").ManagerEvidence.Value, "owner-external")
		if !entry.Running.Known || entry.Running.Value == nil || !*entry.Running.Value ||
			!authority.Adopted.Known || authority.Adopted.Value == nil || !*authority.Adopted.Value ||
			!authority.Scoped.Known || authority.Scoped.Value == nil || *authority.Scoped.Value || authority.ProcessLifetime.Known {
			t.Fatal("adopted external fake received owned-process authority")
		}
		var external serviceProcessIdentity
		for _, identity := range f.scan() {
			if int(identity.PID) == child.cmd.Process.Pid && identity.Path == f.fake {
				external = identity
			}
		}
		if external.PID == 0 {
			t.Fatal("external fake lacks its recorded fixed-image identity")
		}
		refused, err := c.call("813", "engine:stop", mustMarshal(t, map[string]string{"engine": "owner-external"}), serviceCallTimeout)
		if err != nil || refused.Error == nil {
			t.Fatalf("external stop did not refuse: reply=%+v error=%v", refused, err)
		}
		current, err := serviceProcessIdentityAt(t.Context(), external.PID)
		if err != nil || current != external {
			t.Fatalf("manager stopped or replaced the external fake: identity=%+v error=%v", current, err)
		}
		if err := f.inventory.kill(external); err != nil {
			t.Fatal(err)
		}
		f.waitChild(child, serviceCallTimeout)
	})
	t.Run("retained unsafe download provenance and operation busy reach real service impact read", func(t *testing.T) {
		impact := f.impact(t)
		r := impact.ManagerEvidence.Value.Reservations
		if !r.Known || r.Value == nil || !r.Value.BlocksApply.Known || r.Value.BlocksApply.Value == nil || !*r.Value.BlocksApply.Value {
			t.Fatal("retained unsafe download did not block apply")
		}
		found := false
		for _, op := range r.Value.Operations {
			if op.ID == strings.Repeat("d", 32) {
				found = op.Engine == "owner-owned" && op.Kind == "download" && op.Phase == "unsafe" && op.Retained && op.Unsafe && !op.SourceIncarnation.Known
			}
		}
		busy := slices.ContainsFunc(impact.Impact.Refusals, func(r buildapply.Refusal) bool { return r.Code == "operation_busy" })
		if !found || !busy || impact.Impact.Complete {
			t.Fatal("unsafe provenance or broker refusal was lost")
		}
		f.assertNoApplyEffects(t)
	})
	t.Run("active reservation provenance and operation busy reach real service impact read before fake effect completes", func(t *testing.T) {
		defer f.releasePull()
		if err := c.request("814", "engine:action", mustMarshal(t, map[string]any{"engine": "owner-owned", "action": "pull_model", "params": map[string]string{"name": "active-download"}})); err != nil {
			t.Fatal(err)
		}
		select {
		case <-f.pullEntered:
		case <-time.After(30 * time.Second):
			t.Fatal("real download owner did not reach the controlled fixed fake")
		}
		impact := f.impact(t)
		r := impact.ManagerEvidence.Value.Reservations
		if !r.Known || r.Value == nil || !r.Value.BlocksApply.Known || r.Value.BlocksApply.Value == nil || !*r.Value.BlocksApply.Value {
			t.Fatal("active reservation did not block apply")
		}
		active := slices.ContainsFunc(r.Value.Operations, func(op buildapply.EngineOwnerOperation) bool {
			return op.Engine == "owner-owned" && op.Kind == "content_writer" && op.Phase == "active" && !op.Retained && !op.Unsafe && op.SourceIncarnation.Known && op.SourceIncarnation.Value != nil && *op.SourceIncarnation.Value == impact.ManagerEvidence.Value.ManagerIncarnation
		})
		if !active || !slices.ContainsFunc(impact.Impact.Refusals, func(r buildapply.Refusal) bool { return r.Code == "operation_busy" }) || impact.Impact.Complete {
			t.Fatal("active operation lost native source or broker refusal")
		}
		f.assertNoApplyEffects(t)
		f.releasePull()
		frame, err := c.response("814", serviceCallTimeout)
		if err != nil || frame.Error == nil {
			t.Fatalf("controlled failed effect: reply=%+v error=%v", frame, err)
		}
	})
	t.Run("refused preview and start do not send lifecycle commands", func(t *testing.T) {
		beforeBroker, beforeManager := f.stats(t, f.brokerURL), f.stats(t, f.managerURL)
		preview := c.mustCall(t, "820", "service/build-apply-preview", mustMarshal(t, buildapply.PreviewRequest{
			CandidatePath: filepath.Join(f.dir, "absent-candidate"),
			GUI:           buildapply.GUIFacts{PID: os.Getpid(), Executable: os.Args[0]},
		}))
		var result buildapply.Result[buildapply.BuildApplyPreview]
		if err := json.Unmarshal(preview.Result, &result); err != nil {
			t.Fatal(err)
		}
		if result.Available || result.Value != nil || result.Refusal == nil || result.Refusal.Code != "unsafe_package" {
			t.Fatal("nonpackaged fixture unexpectedly produced an eligible preview")
		}
		start := c.mustCall(t, "821", "service/build-apply-start", mustMarshal(t, buildapply.StartRequest{
			PreviewID: strings.Repeat("a", 32), Consent: buildapply.Consent{AllowEngineStop: true, AllowInterruption: true},
			GUI: buildapply.GUIFacts{PID: os.Getpid(), Executable: os.Args[0]},
		}))
		var operation buildapply.Result[buildapply.BuildApplyOperation]
		if err := json.Unmarshal(start.Result, &operation); err != nil {
			t.Fatal(err)
		}
		if operation.Available || operation.Value != nil || operation.Refusal == nil || operation.Refusal.Code != "stale_preview" {
			t.Fatal("missing preview unexpectedly admitted a build transaction")
		}
		afterBroker, afterManager := f.stats(t, f.brokerURL), f.stats(t, f.managerURL)
		for _, method := range []string{"shutdown", "engine:prepare-shutdown", "engine:stop", "build:apply-prepare", "build:apply-finish"} {
			if beforeBroker.Requests[method] != afterBroker.Requests[method] || beforeManager.Requests[method] != afterManager.Requests[method] {
				t.Fatalf("refused operation sent %s", method)
			}
		}
		f.assertNoApplyEffects(t)
	})
	t.Run("manager restart replaces native incarnation and withdraws prior running facts", func(t *testing.T) {
		previous := *f.impact(t).ManagerEvidence.Value
		next := *f.replaceManager(t, nil).ManagerEvidence.Value
		if next.ManagerIncarnation == previous.ManagerIncarnation || next.CaptureToken == previous.CaptureToken {
			t.Fatal("replacement manager reused predecessor authority")
		}
		entry, authority := ownerEvidenceEntry(t, next, "owner-never")
		if entry.Running.Known || entry.TerminationVerified.Known || authority.Scoped.Known {
			t.Fatal("replacement fabricated predecessor stop knowledge")
		}
		owned, stop := ownerEvidenceEntry(t, next, "owner-owned")
		if owned.Running.Known || owned.TerminationVerified.Known || stop.Scoped.Known || stop.ProcessLifetime.Known {
			t.Fatal("replacement retained predecessor owned-process or stopped authority")
		}
		f.assertNoApplyEffects(t)
	})
	if f.observe(t, "/stats").Effects["reads"] == 0 {
		t.Fatal("service source observation overlay was not exercised")
	}
	afterLifetime := serviceStatus(t, c)
	if afterLifetime.PID != serviceLifetime.PID || afterLifetime.BrokerPID != serviceLifetime.BrokerPID || afterLifetime.BrokerRestarts != serviceLifetime.BrokerRestarts || f.stats(t, f.brokerURL).PID != brokerLifetime {
		t.Fatal("manager scenarios replaced the service or broker instead of retaining their fixture lifetimes")
	}
	if err := f.cleanupChecked(); err != nil {
		t.Fatal(err)
	}
	f.assertStopped()
}
