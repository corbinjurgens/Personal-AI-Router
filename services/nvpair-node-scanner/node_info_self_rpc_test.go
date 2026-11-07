// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/noderec"
)

func newSelfNodeInfoRPCDaemon(t *testing.T, handler http.HandlerFunc) (*daemon, int) {
	t.Helper()
	server := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/v1/node-info", handler))
	t.Cleanup(server.Close)
	host, portString, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	if host != loopbackHost {
		t.Fatalf("fixture host = %q, want %q", host, loopbackHost)
	}
	port, err := strconv.Atoi(portString)
	if err != nil {
		t.Fatal(err)
	}
	d := newSelfTestDaemon("self-host", "192.0.2.10")
	d.reg.register(noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: port})
	transport := nodeInfoTransport(nil)
	transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		if network != "tcp" || address != server.Listener.Addr().String() {
			t.Errorf("self read attempted non-fixture destination %s %s", network, address)
			return nil, fmt.Errorf("fixture refuses unknown destination")
		}
		return nodeInfoDialer().DialContext(ctx, network, address)
	}
	d.http = &http.Client{Timeout: nodeInfoFetchTimeout, Transport: transport}
	t.Cleanup(d.http.CloseIdleConnections)
	return d, port
}

func selfNodeInfoRPCSample(principal string) NodeInfoResponse {
	return NodeInfoResponse{
		HostUUID: "self-host", ClusterUUID: &principal,
		ServiceBuildID: scannerCleanBuild,
		GPUs:           []GPUInfo{{Name: "live GPU", PhysicalID: "GPU-first", VramBytes: 4096}},
		CPU:            &CPUInfo{Name: "self CPU", Cores: 8}, Memory: &MemoryInfo{TotalBytes: 8192, UsedBytes: 2048},
		TelemetryValid: true, MSSince: 100,
	}
}

func TestNodeInfoRPCSelfUnclusteredFullSample(t *testing.T) {
	want := selfNodeInfoRPCSample("")
	var calls atomic.Int32
	d, port := newSelfNodeInfoRPCDaemon(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writeScannerFixtureResponse(t, w, want, 0)
	})
	// A stale directory/cache cannot choose self's endpoint or supply its sample.
	d.dir.upsert(noderec.DirectoryNode{HostUUID: "self-host", IP: "192.0.2.99", ClusterUUID: "impostor",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: port + 1}}})
	d.lastInfo["self-host"] = NodeInfoResponse{HostUUID: "self-host", GPUs: []GPUInfo{{Name: "stale GPU"}}}
	d.tlsHTTP = &http.Client{Transport: serviceBuildTransport(func(*http.Request) (*http.Response, error) {
		t.Error("self RPC used operator TLS")
		return nil, fmt.Errorf("unexpected TLS read")
	})}
	client := nodeInfoRPCTestClient(t, d)
	result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"self-host"}`)
	if rpcErr != nil {
		t.Fatalf("unclustered self refused: %+v", rpcErr)
	}
	var got NodeInfoResponse
	if err := json.Unmarshal(result, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) || calls.Load() != 1 {
		t.Fatalf("sample=%+v want=%+v calls=%d", got, want, calls.Load())
	}
	if d.lastInfo["self-host"].GPUs[0].Name != "stale GPU" {
		t.Fatal("fresh getter changed enrichment cache")
	}
}

func TestNodeInfoRPCSelfClusteredUsesRegisteredLoopback(t *testing.T) {
	want := selfNodeInfoRPCSample("self-principal")
	d, _ := newSelfNodeInfoRPCDaemon(t, func(w http.ResponseWriter, _ *http.Request) {
		writeScannerFixtureResponse(t, w, want, 0)
	})
	clusterDir := t.TempDir()
	clustertrusttest.Join(t, clusterDir, "fixture-cluster", "self-principal")
	d.mesh = clustertrust.Open(clusterDir)
	if !d.mesh.Clustered() || d.mesh.NodeUUID() != "self-principal" {
		t.Fatal("fixture is not clustered as self-principal")
	}
	d.reg.setIdentity("self-host", "self-principal", []string{"192.0.2.10"})
	// No remote peer admission or directory route is needed for verified self.
	client := nodeInfoRPCTestClient(t, d)
	result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"self-host"}`)
	if rpcErr != nil {
		t.Fatalf("clustered self refused: %+v", rpcErr)
	}
	var got NodeInfoResponse
	if err := json.Unmarshal(result, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("clustered self sample=%+v want=%+v", got, want)
	}
}

func TestNodeInfoRPCSelfRequiresLocalRegistration(t *testing.T) {
	test := func(name, requested string, change func(*daemon)) {
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			d, port := newSelfNodeInfoRPCDaemon(t, func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				writeScannerFixtureResponse(t, w, selfNodeInfoRPCSample(""), 0)
			})
			d.dir.upsert(noderec.DirectoryNode{HostUUID: requested, IP: loopbackHost,
				Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: port}}})
			change(d)
			client := nodeInfoRPCTestClient(t, d)
			params, err := json.Marshal(nodeInfoRPCParams{HostUUID: requested})
			if err != nil {
				t.Fatal(err)
			}
			result, rpcErr := callNodeInfoRPC(t, client, string(params))
			if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 || calls.Load() != 0 {
				t.Fatalf("unowned target: result=%s error=%+v calls=%d", result, rpcErr, calls.Load())
			}
		})
	}
	test("directory self without ni registration", "self-host", func(d *daemon) { d.reg.unregister(noderec.ServiceNodeInfo) })
	test("mDNS-only loopback peer", "claimed-self", func(*daemon) {})
	test("caller uses loopback as identity", loopbackHost, func(*daemon) {})
	test("caller uses LAN address as identity", "192.0.2.10", func(*daemon) {})
	test("retired self host", "self-host", func(d *daemon) { d.reg.setIdentity("new-self", "", nil) })
	test("missing registry", "self-host", func(d *daemon) { d.reg = nil })
	test("invalid registered port", "self-host", func(d *daemon) {
		d.reg.register(noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 65536})
	})
}

func TestNodeInfoRPCSelfRejectsWrongIdentityAndBody(t *testing.T) {
	test := func(name, principal string, change func(*NodeInfoResponse), ending int32) {
		t.Run(name, func(t *testing.T) {
			sample := selfNodeInfoRPCSample(principal)
			change(&sample)
			d, _ := newSelfNodeInfoRPCDaemon(t, func(w http.ResponseWriter, _ *http.Request) {
				writeScannerFixtureResponse(t, w, sample, ending)
			})
			d.reg.setIdentity("self-host", principal, nil)
			client := nodeInfoRPCTestClient(t, d)
			result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"self-host"}`)
			if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 {
				t.Fatalf("invalid self response accepted: result=%s error=%+v", result, rpcErr)
			}
		})
	}
	test("wrong host", "", func(s *NodeInfoResponse) { s.HostUUID = "impostor" }, 0)
	test("missing host", "", func(s *NodeInfoResponse) { s.HostUUID = "" }, 0)
	test("unclustered unknown principal", "", func(s *NodeInfoResponse) { s.ClusterUUID = nil }, 0)
	test("unclustered conflicting principal", "", func(s *NodeInfoResponse) { principal := "impostor"; s.ClusterUUID = &principal }, 0)
	test("clustered unknown principal", "self-principal", func(s *NodeInfoResponse) { s.ClusterUUID = nil }, 0)
	test("clustered empty principal", "self-principal", func(s *NodeInfoResponse) { empty := ""; s.ClusterUUID = &empty }, 0)
	test("clustered conflicting principal", "self-principal", func(s *NodeInfoResponse) { principal := "other"; s.ClusterUUID = &principal }, 0)
	test("truncated HTTP body", "", func(*NodeInfoResponse) {}, scannerIncompleteFraming)
	test("additional JSON", "", func(*NodeInfoResponse) {}, scannerAdditionalJSON)
}

func TestNodeInfoRPCSelfRefusesRedirect(t *testing.T) {
	var redirected atomic.Int32
	target := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/v1/node-info", func(w http.ResponseWriter, _ *http.Request) {
		redirected.Add(1)
		writeScannerFixtureResponse(t, w, selfNodeInfoRPCSample(""), 0)
	}))
	t.Cleanup(target.Close)
	d, _ := newSelfNodeInfoRPCDaemon(t, func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+"/v1/node-info", http.StatusTemporaryRedirect)
	})
	client := nodeInfoRPCTestClient(t, d)
	result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"self-host"}`)
	if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 || redirected.Load() != 0 {
		t.Fatalf("redirect result=%s error=%+v target calls=%d", result, rpcErr, redirected.Load())
	}
}

func TestNodeInfoRPCSelfCancelsRetiredRegistration(t *testing.T) {
	test := func(name string, change func(*daemon, int)) {
		t.Run(name, func(t *testing.T) {
			entered, cancelled, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
			d, port := newSelfNodeInfoRPCDaemon(t, func(_ http.ResponseWriter, r *http.Request) {
				close(entered)
				select {
				case <-r.Context().Done():
					close(cancelled)
				case <-release:
				}
			})
			t.Cleanup(func() { close(release) })
			done := make(chan bool, 1)
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			go func() { _, ok := d.getPeerNodeInfo(ctx, "self-host"); done <- ok }()
			select {
			case <-entered:
			case <-time.After(time.Second):
				t.Fatal("self read did not reach fixture")
			}
			change(d, port)
			select {
			case <-cancelled:
			case <-time.After(time.Second):
				t.Fatal("registration retirement did not cancel in-flight HTTP")
			}
			select {
			case ok := <-done:
				if ok {
					t.Fatal("retired registration returned success")
				}
			case <-time.After(time.Second):
				t.Fatal("retired read did not return")
			}
		})
	}
	test("unregister", func(d *daemon, _ int) { d.reg.unregister(noderec.ServiceNodeInfo) })
	test("port replacement", func(d *daemon, port int) {
		d.reg.register(noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: port + 1})
	})
	test("same-port replacement", func(d *daemon, port int) {
		d.reg.register(noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: port})
	})
	test("unregister and re-register same port", func(d *daemon, port int) {
		d.reg.unregister(noderec.ServiceNodeInfo)
		d.reg.register(noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: port})
	})
	test("host rekey and restoration", func(d *daemon, _ int) {
		d.reg.setIdentity("new-host", "", nil)
		d.reg.setIdentity("self-host", "", nil)
	})
	test("principal replacement and restoration", func(d *daemon, _ int) {
		d.reg.setIdentity("self-host", "new-principal", nil)
		d.reg.setIdentity("self-host", "", nil)
	})
}

func TestNodeInfoRPCSelfDiscardsResponseAfterRetirement(t *testing.T) {
	test := func(name string, wrap func(io.Reader, func()) io.ReadCloser) {
		t.Run(name, func(t *testing.T) {
			d := newSelfTestDaemon("self-host", loopbackHost)
			d.reg.register(noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 1})
			body, err := json.Marshal(selfNodeInfoRPCSample(""))
			if err != nil {
				t.Fatal(err)
			}
			d.http.Transport = serviceBuildTransport(func(r *http.Request) (*http.Response, error) {
				retire := func() {
					d.reg.unregister(noderec.ServiceNodeInfo)
					d.reg.register(noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 1})
				}
				return &http.Response{StatusCode: http.StatusOK, Body: wrap(strings.NewReader(string(body)), retire), Header: make(http.Header), Request: r}, nil
			})
			client := nodeInfoRPCTestClient(t, d)
			result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"self-host"}`)
			if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 {
				t.Fatalf("retired buffered response accepted: result=%s error=%+v", result, rpcErr)
			}
		})
	}
	test("retired before decoding buffered response", func(body io.Reader, retire func()) io.ReadCloser {
		retire()
		return io.NopCloser(body)
	})
	test("retired at body close after fetch validation", func(body io.Reader, retire func()) io.ReadCloser {
		return &serviceBuildCloseBody{Reader: body, onClose: func() error { retire(); return nil }}
	})
}

func TestNodeInfoRPCSelfDeadlineIncludesOriginQueue(t *testing.T) {
	var calls atomic.Int32
	d, port := newSelfNodeInfoRPCDaemon(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writeScannerFixtureResponse(t, w, selfNodeInfoRPCSample(""), 0)
	})
	origin := "http://" + net.JoinHostPort(loopbackHost, strconv.Itoa(port))
	release, acquired := d.nodeInfoOrigins.acquire(t.Context(), origin)
	if !acquired {
		t.Fatal("could not hold loopback origin gate")
	}
	defer release()
	client := nodeInfoRPCTestClient(t, d)
	started := time.Now()
	result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"self-host"}`)
	elapsed := time.Since(started)
	if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 || elapsed < time.Second || elapsed > 2500*time.Millisecond || calls.Load() != 0 {
		t.Fatalf("queued self read: result=%s error=%+v elapsed=%s calls=%d", result, rpcErr, elapsed, calls.Load())
	}
}

func TestNodeInfoRPCSelfAddressAndOtherServicesDoNotRetireRead(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	want := selfNodeInfoRPCSample("")
	d, _ := newSelfNodeInfoRPCDaemon(t, func(w http.ResponseWriter, _ *http.Request) {
		once.Do(func() { close(entered) })
		<-release
		writeScannerFixtureResponse(t, w, want, 0)
	})
	var releaseOnce sync.Once
	releaseReply := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseReply)
	done := make(chan bool, 1)
	go func() { _, ok := d.getPeerNodeInfo(t.Context(), "self-host"); done <- ok }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("self read did not reach fixture")
	}
	d.reg.setAddresses([]string{"192.0.2.20"})
	d.reg.setIdentity("self-host", "", []string{"192.0.2.30"})
	d.reg.register(noderec.RegisterParams{Service: noderec.ServiceEngineManager, Port: 1})
	d.reg.unregister(noderec.ServiceEngineManager)
	releaseReply()
	select {
	case ok := <-done:
		if !ok {
			t.Fatal("unrelated registry change retired loopback authority")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("self read did not return")
	}
}

func TestNodeInfoRPCSelfPhysicalAliasesRetained(t *testing.T) {
	test := func(name, kind string, aliases []string) {
		t.Run(name, func(t *testing.T) {
			want := selfNodeInfoRPCSample("")
			want.GPUs[0].PhysicalID = aliases[0]
			unknown := noderec.MemoryFact{Reason: "fixture unavailable"}
			want.MemoryObservation = &noderec.MemoryObservation{RunID: "node-info-run", Domains: []noderec.MemoryDomain{{
				ID: "physical-pool", Kind: kind, DeviceIDs: aliases, Total: unknown, Free: unknown,
			}}}
			d, _ := newSelfNodeInfoRPCDaemon(t, func(w http.ResponseWriter, _ *http.Request) {
				writeScannerFixtureResponse(t, w, want, 0)
			})
			client := nodeInfoRPCTestClient(t, d)
			result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"self-host"}`)
			if rpcErr != nil {
				t.Fatalf("physical sample refused: %+v", rpcErr)
			}
			var got NodeInfoResponse
			if err := json.Unmarshal(result, &got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("physical sample=%+v want=%+v", got, want)
			}
		})
	}
	test("Linux UUID and PCI alias", "dedicated", []string{"GPU-first", "0000:01:00.0"})
	test("Windows LUID", "dedicated", []string{"luid_0x00000000_0x00000001_phys_0"})
	test("Darwin shared RAM", "shared", []string{"ioreg:42"})
}
