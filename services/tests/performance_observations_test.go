// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/performancewire"
	"nvpair-shared/testbin"
)

const observationModel = "observation-model"
const observationLlamaFinal = `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"timings":{"predicted_n":11,"predicted_ms":250,"predicted_per_second":40}}`
const observationStrataFinal = `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":20},"timings":{"predicted_n":20,"predicted_ms":400,"predicted_per_second":55}}`
const observationLlamaPartial = `{"choices":[{"index":0,"delta":{},"finish_reason":null}],"timings":{"predicted_n":11,"predicted_ms":250,"predicted_per_second":40}}`

// These are process fixtures, never native-engine evidence. Every upstream is
// a loopback HTTP handler and every identity/digest is explicitly synthetic.
type observationProxy struct {
	rpc          *task1Process
	port         int
	engine, host string
}

func observationPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if err := ln.Close(); err != nil {
		t.Fatal(err)
	}
	if _, installed := installedServicePorts[port]; installed {
		t.Fatal("fixture selected installed port")
	}
	return port
}

func observationIdentity(engine string) nodepolicy.SetModelIdentityParams {
	return nodepolicy.SetModelIdentityParams{Engines: map[string]nodepolicy.EngineIdentity{engine: {
		EngineRunID: "synthetic-observation-run", SettingsFingerprint: strings.Repeat("a", 64),
		LaunchDigest: strings.Repeat("b", 64), ModelDigests: map[string]string{observationModel: strings.Repeat("c", 64)},
	}}}
}

func observationDescriptor(engine string) engines.Descriptor {
	return engines.Descriptor{Version: engines.DescriptorVersion, Engine: engine, Family: engine, Label: "Synthetic observation engine", Platforms: []string{runtime.GOOS + "/" + runtime.GOARCH}, Available: true, Protocols: []string{engines.ProtocolOpenAIChat}, ModelNaming: "exact", Operations: map[string]engines.OperationCapability{}}
}

func observationStartProxy(t *testing.T, base, cluster, host, engine, upstream string) *observationProxy {
	t.Helper()
	p := &observationProxy{rpc: startTask1Process(t, base, proxyBin, "--cluster-dir", cluster), port: observationPort(t), engine: engine, host: host}
	bindProxyFixtureIdentity(t, host, func(method string, params any) jsonrpc.Message {
		return p.rpc.call(t, method, params)
	})
	params := map[string]any{"engine": engine, "port": p.port, "ignorePersistedPort": true}
	if engine == "strata" {
		params["descriptor"] = observationDescriptor(engine)
	}
	p.rpc.call(t, "facade/enable", params)
	p.identify(t, true)
	p.backend(t, upstream)
	return p
}

func (p *observationProxy) identify(t *testing.T, present bool) {
	t.Helper()
	identity := observationIdentity(p.engine)
	if !present {
		identity.Engines = map[string]nodepolicy.EngineIdentity{}
	}
	p.rpc.call(t, nodepolicy.MethodSetModelIdentity, identity)
	p.rpc.send(t, engines.AddressMethod(p.engine, noderec.NotifyNodes), noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{p.directory()}}, false)
	p.rpc.call(t, engines.AddressMethod(p.engine, "nodes/list"), nil)
}

func (p *observationProxy) directory() noderec.DirectoryNode {
	service := noderec.ServiceLlamaCPP
	if p.engine == "strata" {
		service = noderec.ServiceKey("e-strata")
	}
	return noderec.DirectoryNode{HostUUID: p.host, ClusterUUID: p.host, IP: "127.0.0.1", Trusted: true, Services: map[noderec.ServiceKey]noderec.ServiceStatus{service: {Port: p.port}}, ModelsByEngine: map[string][]string{p.engine: {observationModel}}}
}

func (p *observationProxy) backend(t *testing.T, upstream string) {
	t.Helper()
	port := portOfURL(t, upstream)
	p.rpc.call(t, engines.AddressMethod(p.engine, "node/set-local-backend"), map[string]any{"engine": p.engine, "host": "127.0.0.1", "port": port, "healthy": true})
}

func observationFrames(t *testing.T, p *task1Process, method string) []jsonrpc.Message {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	var frames []jsonrpc.Message
	for _, frame := range p.frames {
		if frame.Method == method {
			frames = append(frames, frame)
		}
	}
	return frames
}

func observationSamples(t *testing.T, p *task1Process) []performancewire.Sample {
	t.Helper()
	var samples []performancewire.Sample
	for _, frame := range observationFrames(t, p, "performance:sample") {
		var sample performancewire.Sample
		if err := json.Unmarshal(frame.Params, &sample); err != nil {
			t.Fatal(err)
		}
		if err := sample.Validate(); err != nil {
			t.Fatal(err)
		}
		samples = append(samples, sample)
	}
	return samples
}

func observationWaitSamples(t *testing.T, p *task1Process, n int) []performancewire.Sample {
	t.Helper()
	var samples []performancewire.Sample
	task1Eventually(t, "destination observation", 10*time.Second, func() bool { samples = observationSamples(t, p); return len(samples) >= n })
	if len(samples) != n {
		t.Fatalf("observations=%d, want %d", len(samples), n)
	}
	return samples
}

func observationUpstream(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewUnstartedServer(handler)
	if err := srv.Listener.Close(); err != nil {
		t.Fatal(err)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv.Listener = ln
	srv.Start()
	t.Cleanup(srv.Close)
	return srv
}

func observationRequest(t *testing.T, p *observationProxy, ctx context.Context, stream, usage bool) *http.Response {
	t.Helper()
	return observationRequestWithTimeout(t, p, ctx, stream, usage, 10*time.Second)
}

func observationRequestWithTimeout(t *testing.T, p *observationProxy, ctx context.Context, stream, usage bool, timeout time.Duration) *http.Response {
	t.Helper()
	body := map[string]any{"model": observationModel, "messages": []any{}, "stream": stream}
	if usage {
		body["stream_options"] = map[string]bool{"include_usage": true}
	}
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", p.port), bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: timeout}
	t.Cleanup(client.CloseIdleConnections)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func observationRead(t *testing.T, resp *http.Response) []byte {
	t.Helper()
	body, err := io.ReadAll(resp.Body)
	if closeErr := resp.Body.Close(); closeErr != nil {
		t.Error(closeErr)
	}
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func observationAssertRate(t *testing.T, s performancewire.Sample, host, engine string, rate float64, duration int64) {
	t.Helper()
	if s.Key.Host != host || s.Key.Engine != engine || s.Key.Model != observationModel || len(s.Key.Fingerprint) != 64 {
		t.Fatalf("destination profile=%+v", s.Key)
	}
	if s.Outcome != performancewire.OutcomeCompleted || s.ProtocolComplete == nil || !*s.ProtocolComplete || s.EngineDecodeTokensPerSecond == nil || *s.EngineDecodeTokensPerSecond != rate || s.EngineDecodeNs == nil || *s.EngineDecodeNs != duration || s.DecodeNs != nil {
		t.Fatalf("native measurement=%+v", s.Measurement)
	}
	if s.FirstTokenMs != nil || s.OutputTokens == nil || float64(*s.OutputTokens)*1e9/float64(duration) == rate {
		t.Fatalf("visible count was substituted for native speed: %+v", s.Measurement)
	}
}

func TestPerformanceObservationNativeRates(t *testing.T) {
	test := func(name, engine, payload string, usage bool, rate float64, duration int64) {
		t.Run(name, func(t *testing.T) {
			upstream := observationUpstream(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				if _, err := io.WriteString(w, payload); err != nil {
					t.Error(err)
				}
			})
			base := t.TempDir()
			p := observationStartProxy(t, base, filepath.Join(base, "cluster"), "destination-b", engine, upstream.URL)
			resp := observationRequest(t, p, context.Background(), true, usage)
			body := observationRead(t, resp)
			if resp.StatusCode != http.StatusOK || !bytes.Equal(body, []byte(payload)) {
				t.Fatalf("inference status=%d body=%q", resp.StatusCode, body)
			}
			s := observationWaitSamples(t, p.rpc, 1)[0]
			observationAssertRate(t, s, "destination-b", engine, rate, duration)
		})
	}
	test("llama finish contains timings without usage option", "llamacpp", "data: "+observationLlamaFinal+"\n\ndata: [DONE]\n\n", false, 40, 250000000)
	test("llama final empty choices usage contains timings", "llamacpp", "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":11},\"timings\":{\"predicted_n\":11,\"predicted_ms\":250,\"predicted_per_second\":40}}\n\ndata: [DONE]\n\n", true, 40, 250000000)
	test("Strata preserves native work rate with usage option", "strata", "data: "+observationStrataFinal+"\n\ndata: [DONE]\n\n", true, 55, 400000000)
	test("Strata preserves native work rate without usage option", "strata", "data: "+observationStrataFinal+"\n\ndata: [DONE]\n\n", false, 55, 400000000)
}

func TestPerformanceObservationInvalidTimingRecovery(t *testing.T) {
	test := func(name, invalidTimings string) {
		t.Run(name, func(t *testing.T) {
			var request int
			var requestMu sync.Mutex
			upstream := observationUpstream(t, func(w http.ResponseWriter, r *http.Request) {
				requestMu.Lock()
				request++
				invalid := request == 2
				requestMu.Unlock()
				w.Header().Set("Content-Type", "text/event-stream")
				payload := "data: " + observationLlamaFinal + "\n\ndata: [DONE]\n\n"
				if invalid {
					// Valid evidence on both sides of the invalid object must not
					// revive this request's rate. The next request may qualify.
					payload = "data: " + observationLlamaPartial + "\n\ndata: {\"choices\":[],\"timings\":" + invalidTimings + "}\n\n" + payload
				}
				if _, err := io.WriteString(w, payload); err != nil {
					t.Error(err)
				}
			})
			b, p := observationBrokerFixture(t, upstream.URL)
			observationRead(t, observationRequest(t, p, context.Background(), true, false))
			first := observationWaitBrokerSamples(t, b, 1)[0]
			observationAssertRate(t, first, p.host, p.engine, 40, 250000000)
			observationWaitBrokerCoverage(t, b, first.Key, 1, 1, 0, 0, 1)

			observationRead(t, observationRequest(t, p, context.Background(), true, false))
			invalid := observationWaitBrokerSamples(t, b, 2)[1]
			if invalid.Key != first.Key || invalid.Seq != 2 || invalid.Outcome != performancewire.OutcomeCompleted || invalid.ProtocolComplete == nil || !*invalid.ProtocolComplete || invalid.EngineDecodeTokensPerSecond != nil || invalid.DecodeNs != nil {
				t.Fatalf("invalid timing retained qualified rate: %+v", invalid)
			}
			observationWaitBrokerCoverage(t, b, first.Key, 2, 2, 0, 0, 1)

			observationRead(t, observationRequest(t, p, context.Background(), true, false))
			recovered := observationWaitBrokerSamples(t, b, 3)[2]
			if recovered.Key != first.Key || recovered.Seq != 3 {
				t.Fatalf("recovery profile/sequence=%+v", recovered)
			}
			observationAssertRate(t, recovered, p.host, p.engine, 40, 250000000)
			observationWaitBrokerCoverage(t, b, first.Key, 3, 3, 0, 0, 2)
		})
	}
	test("negative primary duration", `{"predicted_ms":-1}`)
	test("non-numeric primary rate", `{"predicted_per_second":"invalid"}`)
	test("invalid auxiliary with valid primary", `{"predicted_n":11,"predicted_ms":250,"predicted_per_second":40,"predicted_per_token_ms":"invalid"}`)
}

func TestPerformanceObservationNonstreamPassThrough(t *testing.T) {
	test := func(name string, status int, body string, want performancewire.Outcome) {
		t.Run(name, func(t *testing.T) {
			upstream := observationUpstream(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				if _, err := io.WriteString(w, body); err != nil {
					t.Error(err)
				}
			})
			base := t.TempDir()
			p := observationStartProxy(t, base, filepath.Join(base, "cluster"), "destination-b", "llamacpp", upstream.URL)
			resp := observationRequest(t, p, context.Background(), false, false)
			if got := observationRead(t, resp); resp.StatusCode != status || !bytes.Equal(got, []byte(body)) {
				t.Fatalf("nonstream status=%d bytes=%q", resp.StatusCode, got)
			}
			s := observationWaitSamples(t, p.rpc, 1)[0]
			if s.Outcome != want || s.ProtocolComplete != nil || s.EngineDecodeNs != nil || s.EngineDecodeTokensPerSecond != nil || s.OutputTokens != nil {
				t.Fatalf("unobserved nonstream=%+v", s.Measurement)
			}
		})
	}
	test("accepted EOF is completed but protocol remains unknown", http.StatusOK, "  "+observationLlamaFinal+"\n", performancewire.OutcomeCompleted)
	test("malformed accepted JSON remains byte exact and unobserved", http.StatusOK, "{\"choices\": [", performancewire.OutcomeCompleted)
	test("error status is failed without protocol inference", http.StatusBadRequest, " {\"error\":{\"message\":\"synthetic rejection\"}}\n", performancewire.OutcomeFailed)
}

func TestPerformanceObservationUnqualifiedStreams(t *testing.T) {
	test := func(name, payload string, want performancewire.Outcome, complete bool) {
		t.Run(name, func(t *testing.T) {
			upstream := observationUpstream(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				if _, err := io.WriteString(w, payload); err != nil {
					t.Error(err)
				}
			})
			base := t.TempDir()
			p := observationStartProxy(t, base, filepath.Join(base, "cluster"), "destination-b", "llamacpp", upstream.URL)
			observationRead(t, observationRequest(t, p, context.Background(), true, false))
			s := observationWaitSamples(t, p.rpc, 1)[0]
			if s.Outcome != want || s.ProtocolComplete == nil || *s.ProtocolComplete != complete || s.EngineDecodeTokensPerSecond != nil || s.DecodeNs != nil {
				t.Fatalf("unqualified measurement=%+v", s.Measurement)
			}
		})
	}
	test("usage only has output count but unknown speed", "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":11}}\n\ndata: [DONE]\n\n", performancewire.OutcomeCompleted, true)
	test("upstream protocol error is failed", "data: {\"error\":{\"message\":\"synthetic error\"}}\n\n", performancewire.OutcomeFailed, false)
}

func TestPerformanceObservationInterruption(t *testing.T) {
	upstream := observationUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, "data: "+observationLlamaPartial+"\n\n"); err != nil {
			t.Error(err)
		}
	})
	b, p := observationBrokerFixture(t, upstream.URL)
	observationRead(t, observationRequest(t, p, context.Background(), true, false))
	s := observationWaitBrokerSamples(t, b, 1)[0]
	if s.Outcome != performancewire.OutcomeFailed || s.ProtocolComplete == nil || *s.ProtocolComplete {
		t.Fatalf("EOF without terminal=%+v", s.Measurement)
	}
	if s.EngineDecodeTokensPerSecond == nil || *s.EngineDecodeTokensPerSecond != 40 || s.EngineDecodeNs == nil || *s.EngineDecodeNs != 250000000 || s.OutputTokens == nil || *s.OutputTokens != 11 {
		t.Fatalf("interrupted stream lost valid numeric evidence: %+v", s.Measurement)
	}
	observationWaitBrokerCoverage(t, b, s.Key, 1, 0, 1, 0, 0)
}

func TestPerformanceObservationMissingIdentity(t *testing.T) {
	upstream := observationUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, "data: "+observationLlamaFinal+"\n\ndata: [DONE]\n\n"); err != nil {
			t.Error(err)
		}
	})
	base := t.TempDir()
	p := observationStartProxy(t, base, filepath.Join(base, "cluster"), "destination-b", "llamacpp", upstream.URL)
	p.identify(t, false)
	resp := observationRequest(t, p, context.Background(), true, false)
	observationRead(t, resp)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("missing identity changed inference status=%d", resp.StatusCode)
	}
	// Request completion follows performance finalization. Keep identity absent
	// until then so restoration cannot suppress a mistakenly queued first sample
	// through the stale-identity guard.
	task1Eventually(t, "identity-less dispatch finalized", 10*time.Second, func() bool {
		frames := observationFrames(t, p.rpc, engines.AddressMethod(p.engine, "proxy/request"))
		return len(frames) == 1
	})
	// Restore identity and complete a second request. Its positive observation
	// supplies the asynchronous sender barrier; the first dispatch must not count.
	p.identify(t, true)
	observationRead(t, observationRequest(t, p, context.Background(), true, false))
	// Bind the barrier to the restored EngineIdentity, not just the first frame.
	id := observationIdentity(p.engine).Engines[p.engine]
	encoded, err := json.Marshal(struct {
		Run, Settings, Launch, Model, API, Schema string
		Options                                   map[string]any
	}{id.EngineRunID, id.SettingsFingerprint, id.LaunchDigest, id.ModelDigests[observationModel], "/v1/chat/completions", "llamacpp:llamacpp-b12020-0ec4027a", map[string]any{"stream": true}})
	if err != nil {
		t.Fatal(err)
	}
	fingerprint := fmt.Sprintf("%x", sha256.Sum256(encoded))
	expected := performancewire.ProfileKey{Host: p.host, Engine: p.engine, Model: observationModel, Fingerprint: fingerprint}
	task1Eventually(t, "sample from restored identity", 10*time.Second, func() bool {
		samples := observationSamples(t, p.rpc)
		for i, sample := range samples {
			if sample.Key != expected {
				t.Fatalf("sample from identity-less request: %+v", sample)
			}
			if i != 0 || sample.Seq != 1 || len(samples) != 1 {
				t.Fatalf("sample preceded restored identity or consumed sequence: %+v", samples)
			}
			observationAssertRate(t, sample, p.host, p.engine, 40, 250000000)
			return true
		}
		return false
	})
}

func TestPerformanceObservationCancellation(t *testing.T) {
	canceled := make(chan struct{})
	upstream := observationUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, "data: "+observationLlamaPartial+"\n\n"); err != nil {
			t.Error(err)
			return
		}
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(canceled)
	})
	b, p := observationBrokerFixture(t, upstream.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resp := observationRequest(t, p, ctx, true, false)
	reader := bufio.NewReader(resp.Body)
	if _, err := reader.ReadString('\n'); err != nil {
		t.Fatal(err)
	}
	cancel()
	if err := resp.Body.Close(); err != nil {
		t.Error(err)
	}
	select {
	case <-canceled:
	case <-time.After(10 * time.Second):
		t.Fatal("upstream did not receive cancellation")
	}
	s := observationWaitBrokerSamples(t, b, 1)[0]
	if s.Outcome != performancewire.OutcomeCancelled || s.ProtocolComplete == nil || *s.ProtocolComplete {
		t.Fatalf("cancellation=%+v", s.Measurement)
	}
	if s.EngineDecodeTokensPerSecond == nil || *s.EngineDecodeTokensPerSecond != 40 || s.EngineDecodeNs == nil || *s.EngineDecodeNs != 250000000 || s.OutputTokens == nil || *s.OutputTokens != 11 {
		t.Fatalf("cancelled stream lost valid numeric evidence: %+v", s.Measurement)
	}
	observationWaitBrokerCoverage(t, b, s.Key, 1, 0, 0, 1, 0)
}

func TestPerformanceObservationFailedDispatchThenSuccess(t *testing.T) {
	upstream := observationUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, "data: "+observationLlamaFinal+"\n\ndata: [DONE]\n\n"); err != nil {
			t.Error(err)
		}
	})
	base := t.TempDir()
	p := observationStartProxy(t, base, filepath.Join(base, "cluster"), "destination-b", "llamacpp", upstream.URL)
	closedPort := observationPort(t)
	p.backend(t, fmt.Sprintf("http://127.0.0.1:%d", closedPort))
	// nvpair-proxy/http_handler.go: maxDispatchAttempts permits five dials,
	// defaultRetryBackoff totals 15s and backoffFor adds at most 25% jitter.
	// proxy.go: proxyDialTimeout bounds each refused dial at 10s. This is
	// 68.75s, below jobDeadline (10m); allow another 10s for the response.
	const failedDispatches = 5
	const failureTimeout = failedDispatches*10*time.Second + 15*time.Second*5/4 + 10*time.Second
	resp := observationRequestWithTimeout(t, p, context.Background(), true, false, failureTimeout)
	observationRead(t, resp)
	if resp.StatusCode == http.StatusOK {
		t.Fatal("failed local dial returned success")
	}
	failed := observationWaitSamples(t, p.rpc, failedDispatches)
	p.backend(t, upstream.URL)
	observationRead(t, observationRequest(t, p, context.Background(), true, false))
	samples := observationWaitSamples(t, p.rpc, failedDispatches+1)
	success := samples[failedDispatches]
	for i, sample := range failed {
		if sample.Outcome != performancewire.OutcomeFailed || sample.Key != success.Key || sample.Producer != success.Producer || sample.Seq != uint64(i+1) {
			t.Fatalf("failed dispatch %d=%+v, success=%+v", i+1, sample, success)
		}
	}
	if success.Seq != failedDispatches+1 {
		t.Fatalf("success sequence=%d, want %d", success.Seq, failedDispatches+1)
	}
	observationAssertRate(t, success, "destination-b", "llamacpp", 40, 250000000)
}

// The broker has no public sample-injection RPC. This worker fixture preserves
// its real proxy child's stdio and offers only the test process a loopback
// replay boundary. The manager personality observes ordinary T3b publications;
// actual authenticated T3a managers are exercised separately below.
func init() {
	self, err := os.Executable()
	if err != nil {
		return
	}
	role := ""
	if self == os.Getenv("NVPAIR_OBSERVATION_PROXY_FIXTURE") {
		role = "proxy"
	}
	if self == os.Getenv("NVPAIR_OBSERVATION_MANAGER_FIXTURE") {
		role = "manager"
	}
	if role == "" {
		return
	}
	if err := observationRunWorker(role); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	os.Exit(0)
}

func observationRunWorker(role string) error {
	portEnv := "NVPAIR_OBSERVATION_PROXY_CONTROL"
	if role == "manager" {
		portEnv = "NVPAIR_OBSERVATION_MANAGER_CONTROL"
	}
	ln, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", os.Getenv(portEnv)))
	if err != nil {
		return err
	}
	defer ln.Close()
	var mu, outputMu, inputMu sync.Mutex
	var frames []jsonrpc.Message
	pending := map[string]chan jsonrpc.Message{}
	controlID := 1000000
	output := json.NewEncoder(os.Stdout)
	emit := func(frame jsonrpc.Message) error {
		outputMu.Lock()
		defer outputMu.Unlock()
		return output.Encode(frame)
	}
	var child *exec.Cmd
	var input io.WriteCloser
	if role == "proxy" {
		child = exec.Command(os.Getenv("NVPAIR_OBSERVATION_REAL_PROXY"), os.Args[1:]...)
		input, err = child.StdinPipe()
		if err != nil {
			return err
		}
		out, pipeErr := child.StdoutPipe()
		if pipeErr != nil {
			return pipeErr
		}
		child.Stderr = os.Stderr
		if err := child.Start(); err != nil {
			return err
		}
		go func() {
			codec := jsonrpc.NewCodec(struct {
				io.Reader
				io.Writer
			}{out, io.Discard})
			for {
				frame, readErr := codec.Read()
				if readErr != nil {
					return
				}
				mu.Lock()
				frames = append(frames, *frame)
				id := ""
				if frame.ID != nil {
					id = string(*frame.ID)
				}
				reply := pending[id]
				if reply != nil {
					delete(pending, id)
				}
				mu.Unlock()
				if reply != nil {
					reply <- *frame
					continue
				}
				if emit(*frame) != nil {
					return
				}
			}
		}()
	}
	srv := &http.Server{ReadHeaderTimeout: 5 * time.Second, Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/frames":
			mu.Lock()
			snapshot := append([]jsonrpc.Message(nil), frames...)
			mu.Unlock()
			if err := json.NewEncoder(w).Encode(snapshot); err != nil {
				fmt.Fprintln(os.Stderr, err)
			}
		case r.Method == http.MethodPost && r.URL.Path == "/emit":
			var frame jsonrpc.Message
			if err := json.NewDecoder(io.LimitReader(r.Body, performancewire.MaxPayloadBytes)).Decode(&frame); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if err := emit(frame); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		case role == "proxy" && r.Method == http.MethodPost && r.URL.Path == "/control":
			var frame jsonrpc.Message
			if err := json.NewDecoder(io.LimitReader(r.Body, performancewire.MaxPayloadBytes)).Decode(&frame); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			_, method := engines.SplitAddressedMethod(frame.Method)
			if method == noderec.NotifyNodes {
				inputMu.Lock()
				err := json.NewEncoder(input).Encode(frame)
				inputMu.Unlock()
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}
				if err := json.NewEncoder(w).Encode(jsonrpc.Message{JSONRPC: "2.0"}); err != nil {
					fmt.Fprintln(os.Stderr, err)
				}
				return
			}
			mu.Lock()
			controlID++
			id := strconv.Itoa(controlID)
			rawID := json.RawMessage(id)
			frame.ID = &rawID
			reply := make(chan jsonrpc.Message, 1)
			pending[id] = reply
			mu.Unlock()
			inputMu.Lock()
			err := json.NewEncoder(input).Encode(frame)
			inputMu.Unlock()
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			select {
			case response := <-reply:
				if err := json.NewEncoder(w).Encode(response); err != nil {
					fmt.Fprintln(os.Stderr, err)
				}
			case <-r.Context().Done():
			case <-time.After(10 * time.Second):
				http.Error(w, "proxy control timeout", http.StatusGatewayTimeout)
			}
		default:
			http.NotFound(w, r)
		}
	})}
	go func() {
		if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
			fmt.Fprintln(os.Stderr, err)
		}
	}()
	codec := jsonrpc.NewCodec(struct {
		io.Reader
		io.Writer
	}{os.Stdin, io.Discard})
	for {
		frame, readErr := codec.Read()
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
		if role == "manager" {
			mu.Lock()
			frames = append(frames, *frame)
			mu.Unlock()
			continue
		}
		inputMu.Lock()
		err := json.NewEncoder(input).Encode(frame)
		inputMu.Unlock()
		if err != nil {
			return err
		}
	}
	if err := srv.Close(); err != nil {
		return err
	}
	if child != nil {
		if err := input.Close(); err != nil {
			return err
		}
		return child.Wait()
	}
	return nil
}

type observationBroker struct {
	in             io.WriteCloser
	msgs           <-chan jsonrpc.Message
	stop           func()
	proxy, manager string
	client         *http.Client
}

func observationWorkerHTTP(t *testing.T, client *http.Client, endpoint, path string, value, out any) {
	t.Helper()
	method := http.MethodGet
	var reader io.Reader
	if value != nil {
		method = http.MethodPost
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		reader = bytes.NewReader(data)
	}
	req, err := http.NewRequest(method, endpoint+path, reader)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			t.Error(err)
		}
	}()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("fixture %s status=%d", path, resp.StatusCode)
	}
	if out != nil {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			t.Fatal(err)
		}
	} else {
		if _, err := io.Copy(io.Discard, resp.Body); err != nil {
			t.Fatal(err)
		}
	}
}

func (b *observationBroker) control(t *testing.T, method string, params any) {
	t.Helper()
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var reply jsonrpc.Message
	observationWorkerHTTP(t, b.client, b.proxy, "/control", jsonrpc.Message{JSONRPC: "2.0", Method: method, Params: data}, &reply)
	if reply.Error != nil {
		t.Fatalf("fixture proxy %s: %+v", method, reply.Error)
	}
}

func observationStartBroker(t *testing.T, base, cluster, host, upstream string) (*observationBroker, *observationProxy) {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binDir := testbin.Dir(t, "tests", "performance-observations", "observation-proxy", "observation-manager", "observation-scanner")
	proxyFixture := filepath.Join(binDir, testbin.Exe("observation-proxy"))
	managerFixture := filepath.Join(binDir, testbin.Exe("observation-manager"))
	scannerFixture := filepath.Join(binDir, testbin.Exe("observation-scanner"))
	for _, path := range []string{proxyFixture, managerFixture, scannerFixture} {
		jointCopyExecutable(t, self, path)
	}
	proxyControl, managerControl, scannerControl := observationPort(t), observationPort(t), observationPort(t)
	facadePort := observationPort(t)
	if err := os.MkdirAll(jointAppDir(base), 0o700); err != nil {
		t.Fatal(err)
	}
	task1WriteJSON(t, filepath.Join(jointAppDir(base), "llamacpp-proxy-port.json"), map[string]int{"port": facadePort})
	in, msgs, stderr, stop := startBrokerWithDirsAndEnv(t, base, cluster, []string{
		"NVPAIR_OBSERVATION_PROXY_FIXTURE=" + proxyFixture, "NVPAIR_OBSERVATION_MANAGER_FIXTURE=" + managerFixture,
		"NVPAIR_OBSERVATION_PROXY_CONTROL=" + strconv.Itoa(proxyControl), "NVPAIR_OBSERVATION_MANAGER_CONTROL=" + strconv.Itoa(managerControl),
		"NVPAIR_OBSERVATION_REAL_PROXY=" + proxyBin, "NVPAIR_TEST_SUBSCRIPTION_SCANNER=" + scannerFixture, "NVPAIR_TEST_SUBSCRIPTION_PORT=" + strconv.Itoa(scannerControl),
	}, "--scanner-path", scannerFixture, "--proxy-path", proxyFixture, "--proxy-engines", "llamacpp", "--workload-manager-path", managerFixture)
	var stopOnce sync.Once
	stopBroker := func() { stopOnce.Do(stop) }
	t.Cleanup(stopBroker)
	go func() {
		for range stderr {
		}
	}()
	waitForMethod(t, msgs, "app:ready", 20*time.Second)
	for _, port := range []int{proxyControl, managerControl} {
		waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), 10*time.Second)
	}
	b := &observationBroker{in: in, msgs: msgs, stop: stopBroker, proxy: fmt.Sprintf("http://127.0.0.1:%d", proxyControl), manager: fmt.Sprintf("http://127.0.0.1:%d", managerControl), client: &http.Client{Timeout: 15 * time.Second}}
	t.Cleanup(b.client.CloseIdleConnections)
	// This path has a real broker: observe its own activation, never inject a
	// fixture policy that could conceal a broken broker-to-proxy handshake.
	var activeRevision string
	task1Eventually(t, "broker bound proxy routing identity", 10*time.Second, func() bool {
		response := callRPC(t, b.in, b.msgs, "policy:get", nil, 10*time.Second)
		if response.Error != nil {
			t.Fatalf("broker policy:get: %+v", response.Error)
		}
		var state struct {
			Policy         nodepolicy.Policy `json:"policy"`
			Activation     string            `json:"activation"`
			ActiveRevision string            `json:"activeRevision"`
		}
		if err := json.Unmarshal(response.Result, &state); err != nil {
			t.Fatal(err)
		}
		activeRevision = state.ActiveRevision
		return state.Activation == "active" && activeRevision == state.Policy.Revision
	})
	var frames []jsonrpc.Message
	observationWorkerHTTP(t, b.client, b.proxy, "/frames", nil, &frames)
	acknowledged := false
	for _, frame := range frames {
		var ack nodepolicy.PolicyAck
		if frame.ID != nil && frame.Error == nil && json.Unmarshal(frame.Result, &ack) == nil && ack.OriginNodeID == host && ack.Revision == activeRevision {
			acknowledged = true
			break
		}
	}
	if !acknowledged {
		t.Fatal("real broker activation lacks the proxy's complete origin/revision/run acknowledgement")
	}
	// Enabling with an explicit port also provides a readiness barrier even if
	// the broker's initial asynchronous facade enable has not finished yet.
	b.control(t, "facade/enable", map[string]any{"engine": "llamacpp", "port": facadePort, "ignorePersistedPort": true})
	p := &observationProxy{port: facadePort, engine: "llamacpp", host: host}
	b.control(t, nodepolicy.MethodSetModelIdentity, observationIdentity("llamacpp"))
	b.control(t, engines.AddressMethod("llamacpp", noderec.NotifyNodes), noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{p.directory()}})
	b.control(t, engines.AddressMethod("llamacpp", "node/set-local-backend"), map[string]any{"engine": "llamacpp", "host": "127.0.0.1", "port": portOfURL(t, upstream), "healthy": true})
	return b, p
}

type observationProjection struct {
	Local struct {
		MeasuringHost string                   `json:"measuringHost"`
		Summary       *performancewire.Summary `json:"summary"`
	} `json:"local"`
	Peers []struct {
		MeasuringHost string                   `json:"measuringHost"`
		Summary       *performancewire.Summary `json:"summary"`
	} `json:"peers"`
}

func observationBrokerFixture(t *testing.T, upstream string) (*observationBroker, *observationProxy) {
	t.Helper()
	base := t.TempDir()
	cluster := filepath.Join(base, "cluster")
	if err := os.MkdirAll(cluster, 0o700); err != nil {
		t.Fatal(err)
	}
	const host = "observation-broker-b"
	task1WriteJSON(t, filepath.Join(cluster, "identity.json"), map[string]any{"node_uuid": host})
	task1WriteJSON(t, filepath.Join(cluster, "admission.json"), map[string]any{"counter": 1, "activated": 1, "clusterId": task1ClusterID, "epoch": 1})
	return observationStartBroker(t, base, cluster, host, upstream)
}

func observationWaitBrokerSamples(t *testing.T, b *observationBroker, n int) []performancewire.Sample {
	t.Helper()
	var samples []performancewire.Sample
	task1Eventually(t, "real proxy samples at broker boundary", 10*time.Second, func() bool {
		var frames []jsonrpc.Message
		observationWorkerHTTP(t, b.client, b.proxy, "/frames", nil, &frames)
		samples = nil
		for _, frame := range frames {
			if frame.Method != "performance:sample" {
				continue
			}
			var sample performancewire.Sample
			if err := json.Unmarshal(frame.Params, &sample); err != nil {
				t.Fatal(err)
			}
			if err := sample.Validate(); err != nil {
				t.Fatal(err)
			}
			samples = append(samples, sample)
		}
		return len(samples) >= n
	})
	if len(samples) != n {
		t.Fatalf("observations=%d, want %d", len(samples), n)
	}
	return samples
}

func observationWaitBrokerCoverage(t *testing.T, b *observationBroker, key performancewire.ProfileKey, samples, completed, failed, cancelled, rates int) {
	t.Helper()
	var summary *performancewire.Summary
	task1Eventually(t, "broker execution count", 10*time.Second, func() bool {
		summary = b.projection(t).Local.Summary
		return summary != nil && len(summary.Entries) == 1 && summary.Entries[0].Samples >= samples
	})
	observationAssertDisplaySummary(t, *summary, key)
	e := summary.Entries[0]
	if summary.MeasuringHost != key.Host || e.Key != key || e.Samples != samples || e.Completed != completed || e.Failed != failed || e.Cancelled != cancelled || e.Decode != nil || e.Prefill != nil {
		t.Fatalf("broker execution/outcome counts=%+v", summary)
	}
	if rates == 0 {
		if e.EngineDecode != nil {
			t.Fatalf("unqualified sample contributed rate: %+v", e.EngineDecode)
		}
	} else if e.EngineDecode == nil || e.EngineDecode.Samples != rates || e.EngineDecode.DurationNs != int64(rates)*250000000 || e.EngineDecode.TokensPerSecond != 40 {
		t.Fatalf("broker qualified-rate coverage=%+v, want %d", e.EngineDecode, rates)
	}
}

func (b *observationBroker) projection(t *testing.T) observationProjection {
	t.Helper()
	response := callRPC(t, b.in, b.msgs, "performance:get", nil, 10*time.Second)
	if response.Error != nil {
		t.Fatalf("performance:get: %+v", response.Error)
	}
	var projection observationProjection
	if err := json.Unmarshal(response.Result, &projection); err != nil {
		t.Fatal(err)
	}
	return projection
}

// Local display summaries from performance:get and performance:changed have no
// publication revision or admission incarnation (performancestore.Store.Local).
// Entry.validate is private, so check the display fields and public validators.
func observationAssertDisplaySummary(t *testing.T, summary performancewire.Summary, key performancewire.ProfileKey) {
	t.Helper()
	if summary.Version != performancewire.SchemaVersion || summary.MeasuringHost != key.Host || summary.Window.MaxAgeMs != performancewire.RetentionMs || summary.Window.MaxSamplesPerKey != performancewire.MaxSamplesPerKey || summary.OmittedEntries != 0 {
		t.Fatalf("display summary metadata=%+v", summary)
	}
	if len(summary.Entries) != 1 {
		t.Fatalf("summary entries=%+v", summary.Entries)
	}
	e := summary.Entries[0]
	if err := e.Key.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := e.Profile.Validate(); err != nil {
		t.Fatal(err)
	}
	if e.Key != key || e.Samples < 1 || e.Samples > summary.Window.MaxSamplesPerKey || e.Completed < 0 || e.Failed < 0 || e.Cancelled < 0 || e.Completed+e.Failed+e.Cancelled != e.Samples || e.Incomplete < 0 || e.Incomplete > e.Completed {
		t.Fatalf("display entry identity/counts=%+v", e)
	}
	if e.NewestAgeMs < 0 || e.NewestAgeMs > e.OldestAgeMs || e.OldestAgeMs > summary.Window.MaxAgeMs || e.NewestObservedAtMs <= 0 {
		t.Fatalf("display entry ages=%+v", e)
	}
}

func observationAssertSummary(t *testing.T, summary performancewire.Summary, key performancewire.ProfileKey, samples int) {
	t.Helper()
	observationAssertDisplaySummary(t, summary, key)
	entry := summary.Entries[0]
	if summary.MeasuringHost != key.Host || entry.Key != key || entry.Samples != samples || entry.Completed != samples || entry.Decode != nil || entry.EngineDecode == nil || entry.EngineDecode.Samples != samples || entry.EngineDecode.DurationNs != int64(samples)*250000000 || entry.EngineDecode.TokensPerSecond != 40 {
		t.Fatalf("summary destination/native rate=%+v", summary)
	}
}

func TestPerformanceObservationBrokerPersistenceAndPublication(t *testing.T) {
	base := t.TempDir()
	cluster := filepath.Join(base, "cluster")
	if err := os.MkdirAll(cluster, 0o700); err != nil {
		t.Fatal(err)
	}
	const host = "observation-broker-b"
	task1WriteJSON(t, filepath.Join(cluster, "identity.json"), map[string]any{"node_uuid": host})
	// Publication requires the admitted certificate principal, not just node metadata.
	clustertrusttest.WriteKeypair(t, cluster, host)
	task1WriteJSON(t, filepath.Join(cluster, "admission.json"), map[string]any{"counter": 1, "activated": 1, "clusterId": task1ClusterID, "epoch": 1})
	upstream := observationUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, "data: "+observationLlamaFinal+"\n\ndata: [DONE]\n\n"); err != nil {
			t.Error(err)
		}
	})
	b, p := observationStartBroker(t, base, cluster, host, upstream.URL)
	response := callRPC(t, b.in, b.msgs, "performance:subscribe", nil, 10*time.Second)
	if response.Error != nil {
		t.Fatalf("performance:subscribe: %+v", response.Error)
	}
	observationRead(t, observationRequest(t, p, context.Background(), true, false))
	var sample performancewire.Sample
	task1Eventually(t, "real proxy sample at broker boundary", 10*time.Second, func() bool {
		var frames []jsonrpc.Message
		observationWorkerHTTP(t, b.client, b.proxy, "/frames", nil, &frames)
		for _, frame := range frames {
			if frame.Method == "performance:sample" {
				if err := json.Unmarshal(frame.Params, &sample); err != nil {
					t.Fatal(err)
				}
				return true
			}
		}
		return false
	})
	observationAssertRate(t, sample, host, "llamacpp", 40, 250000000)
	// Subscription pushes and get expose the same destination profile.
	var changed observationProjection
	task1Eventually(t, "subscribed destination sample", 10*time.Second, func() bool {
		select {
		case frame := <-b.msgs:
			if frame.Method == "performance:changed" {
				if err := json.Unmarshal(frame.Params, &changed); err != nil {
					t.Fatal(err)
				}
				return changed.Local.Summary != nil && len(changed.Local.Summary.Entries) == 1
			}
		default:
		}
		return false
	})
	observationAssertSummary(t, *changed.Local.Summary, sample.Key, 1)
	projection := b.projection(t)
	observationAssertSummary(t, *projection.Local.Summary, sample.Key, 1)
	frame := jsonrpc.Message{JSONRPC: "2.0", Method: "performance:sample", Params: mustMarshal(t, sample)}
	observationWorkerHTTP(t, b.client, b.proxy, "/emit", frame, nil)
	// A new producer can restart its sequence at one. Sending it after the
	// replay makes its observed count a barrier for the duplicate's ingestion.
	sample.Producer = "synthetic-new-proxy-incarnation"
	sample.Seq = 1
	frame.Params = mustMarshal(t, sample)
	observationWorkerHTTP(t, b.client, b.proxy, "/emit", frame, nil)
	task1Eventually(t, "new producer accepted without duplicate count", 10*time.Second, func() bool {
		projection = b.projection(t)
		return projection.Local.Summary != nil && len(projection.Local.Summary.Entries) == 1 && projection.Local.Summary.Entries[0].Samples == 2
	})
	observationAssertSummary(t, *projection.Local.Summary, sample.Key, 2)
	var published performancewire.Summary
	task1Eventually(t, "normal durable T3b publication", 20*time.Second, func() bool {
		var frames []jsonrpc.Message
		observationWorkerHTTP(t, b.client, b.manager, "/frames", nil, &frames)
		for _, frame := range frames {
			if frame.Method == "performance:publish" {
				var summary performancewire.Summary
				if err := json.Unmarshal(frame.Params, &summary); err != nil {
					t.Fatal(err)
				}
				if len(summary.Entries) == 1 && summary.Entries[0].Samples == 2 {
					published = summary
					return true
				}
			}
		}
		return false
	})
	if err := published.Validate(); err != nil {
		t.Fatal(err)
	}
	observationAssertSummary(t, published, sample.Key, 2)
	// Observe the existing save boundary, then restart only the isolated broker
	// with the same config directory. No installed history is read or touched.
	stop := b.stop
	b.stop = func() {}
	stop()
	b2, _ := observationStartBroker(t, base, cluster, host, upstream.URL)
	projection = b2.projection(t)
	if projection.Local.Summary == nil {
		t.Fatal("saved local summary missing after restart")
	}
	observationAssertSummary(t, *projection.Local.Summary, sample.Key, 2)
}

func observationSummary(t *testing.T, sample performancewire.Sample, revision int64) performancewire.Summary {
	t.Helper()
	observationAssertRate(t, sample, sample.Key.Host, "llamacpp", 40, 250000000)
	summary := performancewire.Summary{Version: performancewire.SchemaVersion, MeasuringHost: sample.Key.Host, AdmissionIncarnation: "1", Revision: revision, Window: performancewire.Window{MaxAgeMs: performancewire.RetentionMs, MaxSamplesPerKey: performancewire.MaxSamplesPerKey}, Entries: []performancewire.Entry{{Key: sample.Key, Profile: sample.Profile, Samples: 1, Completed: 1, NewestObservedAtMs: time.Now().UnixMilli(), EngineDecode: &performancewire.ReportedRate{Samples: 1, DurationNs: *sample.EngineDecodeNs, TokensPerSecond: *sample.EngineDecodeTokensPerSecond}}}}
	if err := summary.Validate(); err != nil {
		t.Fatal(err)
	}
	return summary
}

func observationReceived(t *testing.T, p *task1Process, host string) []performancewire.Summary {
	t.Helper()
	var summaries []performancewire.Summary
	for _, frame := range observationFrames(t, p, "performance:received") {
		var received struct {
			Summary           performancewire.Summary `json:"summary"`
			AdmissionIdentity string                  `json:"admissionIdentity"`
		}
		if err := json.Unmarshal(frame.Params, &received); err != nil {
			t.Fatal(err)
		}
		if received.Summary.MeasuringHost == host {
			if received.AdmissionIdentity == "" {
				t.Fatal("unauthenticated replacement")
			}
			summaries = append(summaries, received.Summary)
		}
	}
	return summaries
}

func TestPerformanceObservationDestinationAndAuthenticatedReplacement(t *testing.T) {
	nodes, blockedPort := newTask1Nodes(t)
	a, b := nodes[0], nodes[1]
	proxies := map[string]*observationProxy{}
	var countMu sync.Mutex
	calls := map[string]int{}
	for _, node := range []*task1Node{a, b} {
		n := node
		upstream := observationUpstream(t, func(w http.ResponseWriter, r *http.Request) {
			countMu.Lock()
			calls[n.uuid]++
			countMu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			if _, err := io.WriteString(w, "data: "+observationLlamaFinal+"\n\ndata: [DONE]\n\n"); err != nil {
				t.Error(err)
			}
		})
		p := observationStartProxy(t, n.base, filepath.Join(n.base, "cluster"), n.uuid, "llamacpp", upstream.URL)
		proxies[n.uuid] = p
		n.ports[noderec.ServiceLlamaCPP] = p.port
		n.observers[noderec.ServiceLlamaCPP] = newTask1Observer(t, n, noderec.ServiceLlamaCPP, nodes)
		var services []noderec.RegisterParams
		for _, key := range []noderec.ServiceKey{noderec.ServiceCluster, noderec.ServiceWorkload, noderec.ServiceErrors, noderec.ServiceLlamaCPP} {
			services = append(services, noderec.RegisterParams{Service: key, Port: n.observers[key].port})
		}
		n.cm.call(t, "cluster:set-local-services", map[string]any{"generation": 2, "services": services})
	}
	task1Eventually(t, "authenticated reverse session", 15*time.Second, func() bool { return a.observers[noderec.ServiceCluster].established(b.uuid) > 0 })
	for _, node := range []*task1Node{a, b} {
		peer := a
		if node == a {
			peer = b
		}
		p := proxies[node.uuid]
		directory := proxies[peer.uuid].directory()
		port := peer.observers[noderec.ServiceLlamaCPP].port
		if peer == b {
			port = blockedPort
		}
		directory.Services[noderec.ServiceLlamaCPP] = noderec.ServiceStatus{Port: port}
		p.rpc.send(t, engines.AddressMethod("llamacpp", noderec.NotifyNodes), noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{p.directory(), directory}}, false)
		p.rpc.call(t, engines.AddressMethod("llamacpp", "node/select"), map[string]string{"id": peer.uuid})
		node.workloads.send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{task1Directory(peer, blockedPort)}}, false)
	}
	// A cluster upgrade does not fence the workload discovery tick or prove
	// workload delivery. Backfilled lifecycle events exercise both service routes
	// without consuming performance samples or negotiating performance support.
	for _, node := range []*task1Node{a, b} {
		node.workloads.send(t, "workload:started", task1Workload(node.uuid, "observation-readiness", "running", 1), false)
	}
	task1Eventually(t, "authenticated workload routes in both directions", 30*time.Second, func() bool {
		a.mesh.Refresh()
		b.mesh.Refresh()
		_, aAdmitsB := a.mesh.PeerServiceIncarnation(b.uuid)
		_, bAdmitsA := b.mesh.PeerServiceIncarnation(a.uuid)
		return aAdmitsB && bAdmitsA &&
			task1Upserts(t, a.workloads, b.uuid, "observation-readiness", "running") == 1 &&
			task1Upserts(t, b.workloads, a.uuid, "observation-readiness", "running") == 1
	})
	test := func(name string, origin, destination *task1Node) {
		t.Run(name, func(t *testing.T) {
			observationRead(t, observationRequest(t, proxies[origin.uuid], context.Background(), true, false))
			sample := observationWaitSamples(t, proxies[destination.uuid].rpc, 1)[0]
			observationAssertRate(t, sample, destination.uuid, "llamacpp", 40, 250000000)
			countMu.Lock()
			dispatches := calls[destination.uuid]
			countMu.Unlock()
			if dispatches != 1 {
				t.Fatalf("destination dispatches=%d, want one", dispatches)
			}
			summary := observationSummary(t, sample, 1)
			destination.workloads.call(t, "performance:publish", summary)
			var received []performancewire.Summary
			// A transient authority/read or HTTP failure defers the retained
			// replacement to the manager's 30-second heartbeat. Allow that retry.
			task1Eventually(t, "authenticated replacement", 45*time.Second, func() bool {
				received = observationReceived(t, origin.workloads, destination.uuid)
				return len(received) > 0
			})
			if len(received) != 1 || !reflect.DeepEqual(received[0], summary) {
				t.Fatalf("replacement changed destination summary: %+v", received)
			}
			before := len(observationFrames(t, origin.workloads, "performance:status"))
			destination.workloads.call(t, "performance:publish", summary)
			task1Eventually(t, "replacement replay status acknowledgment", 45*time.Second, func() bool { return len(observationFrames(t, origin.workloads, "performance:status")) > before })
			if received := observationReceived(t, origin.workloads, destination.uuid); len(received) != 1 {
				t.Fatalf("replayed replacement produced %d data mutations", len(received))
			}
		})
	}
	// B is the blocked direct endpoint, so B-to-A publication is direct and
	// A-to-B publication uses the real cluster-manager reverse transport.
	test("reverse inference B observation reaches origin by direct replacement", a, b)
	test("direct inference A observation reaches origin by reverse replacement", b, a)
	for _, node := range []*task1Node{a, b} {
		samples := observationSamples(t, proxies[node.uuid].rpc)
		if len(samples) != 1 || samples[0].Key.Host != node.uuid {
			t.Fatalf("origin counted forwarding execution on %s: %+v", node.uuid, samples)
		}
	}
}

func TestPerformanceObservationOldCapabilityDoesNotAffectInference(t *testing.T) {
	nodes, blockedPort := newTask1Nodes(t)
	a, c := nodes[0], nodes[2]
	const readinessID = "old-capability-readiness"
	var mu sync.Mutex
	probes, summaries := 0, 0
	lifecycleReady := false
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{ReadHeaderTimeout: 5 * time.Second, Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		principal, pinned := c.mesh.VerifyClientPin(r)
		if !pinned || principal != a.uuid {
			http.Error(w, "fixture requires authenticated peer", http.StatusForbidden)
			return
		}
		var frame jsonrpc.Message
		if err := json.NewDecoder(r.Body).Decode(&frame); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		if frame.Method == "workload:started" {
			var params struct {
				WorkloadInfo struct {
					wlInfo
					RunID     string `json:"runId"`
					Seq       int    `json:"seq"`
					CreatedAt int64  `json:"createdAt"`
				} `json:"workloadInfo"`
			}
			if err := json.Unmarshal(frame.Params, &params); err != nil {
				t.Errorf("decode old telemetry readiness workload: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			info := params.WorkloadInfo
			if info.ID != readinessID || info.OriginatedFrom != a.uuid || info.Engine != "ollama" || info.Model != "task1-model" || info.RunID != "task1-run" || info.State != "running" || info.Seq != 1 || info.CreatedAt != 1000 {
				t.Errorf("old telemetry readiness workload=%+v", info)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			lifecycleReady = true
		}
		if frame.Method == "performance:capabilities" {
			probes++
			w.Header().Set("Content-Type", "application/json")
			if _, err := io.WriteString(w, `{"version":1}`); err != nil {
				t.Error(err)
			}
			return
		}
		if frame.Method == "performance:summary" {
			summaries++
		}
		w.WriteHeader(http.StatusNoContent)
	})}
	done := make(chan error, 1)
	go func() { done <- srv.Serve(tls.NewListener(ln, c.mesh.ServerTLSConfig())) }()
	t.Cleanup(func() {
		if err := srv.Close(); err != nil {
			t.Error(err)
		}
		if err := <-done; err != http.ErrServerClosed {
			t.Error(err)
		}
	})
	directory := task1Directory(c, blockedPort)
	directory.Services[noderec.ServiceWorkload] = noderec.ServiceStatus{Port: ln.Addr().(*net.TCPAddr).Port}
	a.workloads.send(t, noderec.NotifyNodes, noderec.GetNodesResult{Nodes: []noderec.DirectoryNode{directory}}, false)
	// Discovery and admission metadata do not fence the asynchronous target
	// refresh or prove that its authenticated HTTP service route is usable.
	a.workloads.send(t, "workload:started", task1Workload(a.uuid, readinessID, "running", 1), false)
	task1Eventually(t, "old telemetry authenticated workload route", 15*time.Second, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return lifecycleReady
	})
	upstream := observationUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, "data: "+observationLlamaFinal+"\n\ndata: [DONE]\n\n"); err != nil {
			t.Error(err)
		}
	})
	p := observationStartProxy(t, a.base, filepath.Join(a.base, "cluster"), a.uuid, "llamacpp", upstream.URL)
	observationRead(t, observationRequest(t, p, context.Background(), true, false))
	sample := observationWaitSamples(t, p.rpc, 1)[0]
	a.workloads.call(t, "performance:publish", observationSummary(t, sample, 1))
	task1Eventually(t, "old telemetry capability response", 15*time.Second, func() bool { mu.Lock(); defer mu.Unlock(); return probes > 0 })
	a.workloads.call(t, "performance:publish", observationSummary(t, sample, 2))
	task1Eventually(t, "second old telemetry capability response", 15*time.Second, func() bool { mu.Lock(); defer mu.Unlock(); return probes > 1 })
	resp := observationRequest(t, p, context.Background(), true, false)
	observationRead(t, resp)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("capability rejection changed inference status=%d", resp.StatusCode)
	}
	observationWaitSamples(t, p.rpc, 2)
	mu.Lock()
	defer mu.Unlock()
	if summaries != 0 {
		t.Fatalf("old capability accepted %d summaries", summaries)
	}
}
