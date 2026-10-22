// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
	"time"

	"nvpair-shared/noderec"
)

func TestDescriptorAddresses(t *testing.T) {
	test := func(name string, scanner, overlay, want []string) {
		t.Run(name, func(t *testing.T) {
			originalScanner, originalOverlay := slices.Clone(scanner), slices.Clone(overlay)
			got := descriptorAddresses(scanner, overlay)
			if !slices.Equal(got, want) {
				t.Fatalf("descriptor addresses = %v, want %v", got, want)
			}
			if again := descriptorAddresses(scanner, overlay); !slices.Equal(again, got) {
				t.Fatalf("repeated assembly = %v, want stable order %v", again, got)
			}
			if !slices.Equal(scanner, originalScanner) || !slices.Equal(overlay, originalOverlay) {
				t.Fatalf("assembly mutated inputs: scanner=%v overlay=%v", scanner, overlay)
			}
			descriptor := noderec.PeerServicesDescriptor{
				HostUUID: "descriptor-test", Name: "descriptor", Addresses: got,
				Services: []noderec.PeerService{},
			}
			err := noderec.ValidatePeerServicesDescriptor(descriptor, descriptor.HostUUID)
			if len(want) == 0 {
				if err == nil {
					t.Fatal("overlay fabricated a valid descriptor without scanner addresses")
				}
			} else if err != nil {
				t.Fatalf("assembled descriptor rejected: %v", err)
			}
			if len(got) > 0 {
				got[0] = "changed output"
				if !slices.Equal(scanner, originalScanner) || !slices.Equal(overlay, originalOverlay) {
					t.Fatal("assembled output aliases an input")
				}
			}
		})
	}
	test("LAN plus overlay", []string{"192.168.42.8"}, []string{"100.91.248.18"}, []string{"192.168.42.8", "100.91.248.18"})
	test("duplicate overlap", []string{"192.168.42.8", "100.91.248.18", "192.168.42.8"}, []string{"100.91.248.18", "100.91.248.18"}, []string{"192.168.42.8", "100.91.248.18"})
	test("full cap reserves overlay", []string{"192.168.42.8", "192.168.42.9", "192.168.42.10", "192.168.42.11"}, []string{"100.91.248.18"}, []string{"192.168.42.8", "192.168.42.9", "192.168.42.10", "100.91.248.18"})
	test("full cap keeps only first missing overlay", []string{"192.168.42.8", "192.168.42.9", "192.168.42.10", "192.168.42.11"}, []string{"100.91.248.19", "100.91.248.18"}, []string{"192.168.42.8", "192.168.42.9", "192.168.42.10", "100.91.248.18"})
	test("multiple overlays limited room", []string{"192.168.42.8", "192.168.42.9"}, []string{"100.91.248.20", "100.91.248.18", "100.91.248.19"}, []string{"192.168.42.8", "192.168.42.9", "100.91.248.18", "100.91.248.19"})
	test("no overlay", []string{"192.168.42.8", "192.168.42.9", "192.168.42.8"}, nil, []string{"192.168.42.8", "192.168.42.9"})
	test("empty scanner", nil, []string{"100.91.248.18"}, nil)
	test("empty inputs", nil, nil, nil)
}

func setDescriptorSelf(t *testing.T, m *Manager) {
	t.Helper()
	m.setSelfServicesIdentity([]noderec.DirectoryNode{{HostUUID: m.identity.NodeUUID, Name: "descriptor-host", IP: "192.168.42.8", IPs: []string{"192.168.42.8"}}})
}

func persistPeerServicesMembers(t *testing.T, managers ...*Manager) {
	t.Helper()
	for _, manager := range managers {
		if err := manager.persistMembersErr(); err != nil {
			t.Fatalf("persist descriptor fixture membership: %v", err)
		}
	}
}

func TestPeerServicesSelfUsesCurrentLocalRegistrations(t *testing.T) {
	m := newReverseTestManager(t)
	setDescriptorSelf(t, m)
	setReverseServices(t, m, 30, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 22441}, noderec.RegisterParams{Service: noderec.ServiceEngineControl, Port: 22442})
	d, err := m.selfServicesDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	if d.HostUUID != m.identity.NodeUUID || d.Addresses[0] != "192.168.42.8" || len(d.Services) != 2 {
		t.Fatalf("self snapshot=%+v", d)
	}
	setReverseServices(t, m, 31, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 22443})
	d, err = m.selfServicesDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Services) != 1 || d.Services[0].Service != noderec.ServiceNodeInfo || d.Services[0].Port != 22443 {
		t.Fatalf("withdrawal/rebind=%+v", d.Services)
	}
	setReverseServices(t, m, 32)
	d, err = m.selfServicesDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	if d.Services == nil || len(d.Services) != 0 {
		t.Fatalf("complete withdrawal=%+v", d.Services)
	}
}

func TestPeerServicesRequiresSelfIdentityAndInitializedRegistry(t *testing.T) {
	m := newReverseTestManager(t)
	setDescriptorSelf(t, m)
	if _, err := m.selfServicesDescriptor(); err == nil {
		t.Fatal("uninitialized registrations accepted")
	}
	setReverseServices(t, m, 0)
	m.setSelfServicesIdentity([]noderec.DirectoryNode{{HostUUID: "other", Name: "other", IP: "192.168.42.9"}})
	if _, err := m.selfServicesDescriptor(); err == nil {
		t.Fatal("third party identity accepted")
	}
	m.setSelfServicesIdentity([]noderec.DirectoryNode{{HostUUID: m.identity.NodeUUID, Name: "self", IP: "127.0.0.1"}})
	if _, err := m.selfServicesDescriptor(); err == nil {
		t.Fatal("loopback self descriptor accepted")
	}
	m.setSelfServicesIdentity([]noderec.DirectoryNode{{HostUUID: m.identity.NodeUUID, Name: "self"}})
	if _, err := m.selfServicesDescriptor(); err == nil {
		t.Fatal("missing scanner addresses accepted")
	}
	m.setSelfServicesIdentity([]noderec.DirectoryNode{{HostUUID: m.identity.NodeUUID, Name: "self", IPs: []string{"192.168.42.8", "192.168.42.9", "192.168.42.10", "127.0.0.1"}}})
	if _, err := m.selfServicesDescriptor(); err == nil {
		t.Fatal("invalid scanner address hidden by overlay reservation")
	}
}

func TestPeerServicesEndpointRequiresCurrentAdmission(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	pairReverseManagers(t, a, b)
	persistPeerServicesMembers(t, a, b)
	setDescriptorSelf(t, b)
	setReverseServices(t, b, 1, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 23441})
	cert, err := x509.ParseCertificate(a.identity.Cert.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	request := func() *http.Request {
		r := httptest.NewRequest(http.MethodGet, peerServicesPath, nil)
		r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{cert}}
		return r
	}
	w := httptest.NewRecorder()
	b.handlePeerServices(w, request())
	if w.Code != http.StatusOK {
		t.Fatalf("current peer status=%d body=%s", w.Code, w.Body)
	}
	d, err := noderec.DecodePeerServices(w.Body, b.identity.NodeUUID)
	if err != nil {
		t.Fatal(err)
	}
	if d.Services[0].Port != 23441 {
		t.Fatalf("services=%+v", d.Services)
	}
	b.removeMemberByUUID(a.identity.NodeUUID)
	persistPeerServicesMembers(t, b)
	w = httptest.NewRecorder()
	b.handlePeerServices(w, request())
	if w.Code != http.StatusForbidden {
		t.Fatalf("removed admitted member status=%d", w.Code)
	}
}

func TestPeerServicesEndpointRejectsBodyAndQuery(t *testing.T) {
	m := newReverseTestManager(t)
	test := func(name, method, url string, body io.Reader, want int) {
		t.Run(name, func(t *testing.T) {
			r := httptest.NewRequest(method, url, body)
			w := httptest.NewRecorder()
			m.handlePeerServices(w, r)
			if w.Code != want {
				t.Fatalf("status=%d want=%d", w.Code, want)
			}
		})
	}
	test("post", http.MethodPost, peerServicesPath, nil, http.StatusMethodNotAllowed)
	test("body", http.MethodGet, peerServicesPath, bytes.NewBufferString("{}"), http.StatusBadRequest)
	test("query", http.MethodGet, peerServicesPath+"?peer=other", nil, http.StatusBadRequest)
	test("unpaired", http.MethodGet, peerServicesPath, nil, http.StatusForbidden)
}

func TestPeerServicesFetchBindsOriginalAdmission(t *testing.T) {
	test := func(name string, retire bool) {
		t.Run(name, func(t *testing.T) {
			a, b := newReverseTestManager(t), newReverseTestManager(t)
			setDescriptorSelf(t, b)
			setReverseServices(t, b, 1, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 24441})
			held := make(chan struct{})
			release, releaseHeld := context.WithCancel(context.Background())
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if retire {
					close(held)
					select {
					case <-release.Done():
					case <-r.Context().Done():
						return
					}
				}
				b.handlePeerServices(w, r)
			}))
			server.TLS = b.buildServerTLSConfig()
			server.StartTLS()
			defer server.Close()
			defer releaseHeld()
			b.port = server.Listener.Addr().(*net.TCPAddr).Port
			pairReverseManagers(t, a, b)
			persistPeerServicesMembers(t, a, b)
			a.mesh.Refresh()
			originalIncarnation, admitted := a.mesh.PeerServiceIncarnation(b.identity.NodeUUID)
			if !admitted {
				t.Fatal("paired fixture has no durable admission")
			}
			var published bytes.Buffer
			a.codec = NewCodec(struct {
				io.Reader
				io.Writer
			}{bytes.NewReader(nil), &published})
			a.serviceRefresh.source = "test-process"
			result := make(chan error, 1)
			go func() { result <- a.refreshPeerServices(context.Background(), b.identity.NodeUUID) }()
			if retire {
				select {
				case <-held:
				case err := <-result:
					t.Fatalf("descriptor fetch completed before held response: %v", err)
				case <-time.After(5 * time.Second):
					t.Fatal("descriptor fetch did not reach held response")
				}
				// Retire and re-admit the same certificate. A pin-only comparison
				// would accept the held response as fresh work for the new epoch.
				a.removeMemberByUUID(b.identity.NodeUUID)
				persistPeerServicesMembers(t, a)
				epoch, err := b.reserveAdmissionEpoch()
				if err != nil {
					t.Fatalf("reserve same-certificate readmission: %v", err)
				}
				if epoch != 2 {
					t.Fatalf("readmission epoch = %d, want 2", epoch)
				}
				if err := b.activateAdmission("cluster-1", epoch); err != nil {
					t.Fatalf("activate same-certificate readmission: %v", err)
				}
				b.addSelfMember()
				persistPeerServicesMembers(t, b)
				node := &ClusterNode{NodeUUID: b.identity.NodeUUID, ID: b.identity.NodeID, IPAddress: "127.0.0.1", Port: b.port, ClusterID: "cluster-1", AdmissionEpoch: 2, State: stateMember}
				a.upsertMember(node)
				pin, ok := a.trust.Get(b.identity.NodeUUID)
				if !ok {
					t.Fatal("peer pin disappeared")
				}
				pin.AdmissionEpoch = 2
				if err := a.trust.Pin(pin); err != nil {
					t.Fatal(err)
				}
				persistPeerServicesMembers(t, a)
				a.mesh.Refresh()
				currentIncarnation, admitted := a.mesh.PeerServiceIncarnation(b.identity.NodeUUID)
				if !admitted || currentIncarnation == originalIncarnation {
					t.Fatal("readmission did not establish a distinct current durable incarnation")
				}
				releaseHeld()
			}
			var err error
			select {
			case err = <-result:
			case <-time.After(10 * time.Second):
				t.Fatal("descriptor fetch did not finish")
			}
			if retire {
				if err == nil {
					t.Fatal("held response accepted after same-certificate readmission")
				}
				codec := NewCodec(struct {
					io.Reader
					io.Writer
				}{bytes.NewReader(published.Bytes()), io.Discard})
				for {
					msg, readErr := codec.Read()
					if readErr == io.EOF {
						break
					}
					if readErr != nil {
						t.Fatal(readErr)
					}
					if msg.Method == peerServicesEvent {
						t.Fatal("held response published an obsolete services claim")
					}
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			msg, err := NewCodec(struct {
				io.Reader
				io.Writer
			}{bytes.NewReader(published.Bytes()), io.Discard}).Read()
			if err != nil {
				t.Fatal(err)
			}
			var claim noderec.PeerServicesClaim
			if err := json.Unmarshal(msg.Params, &claim); err != nil {
				t.Fatal(err)
			}
			if msg.Method != peerServicesEvent || claim.Descriptor.HostUUID != b.identity.NodeUUID || claim.Descriptor.Services[0].Port != 24441 || claim.Incarnation == "" || claim.Sequence != 1 || claim.Source != "test-process" {
				t.Fatalf("claim=%+v method=%s", claim, msg.Method)
			}
		})
	}
	test("current pinned direct peer", false)
	test("same certificate readmission rejects held response", true)
}

func TestPeerServicesBlockedDirectUsesRealReverse(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, true)
	pairClusterFixtures(t, a, b)
	persistPeerServicesMembers(t, a.m, b.m)
	setDescriptorSelf(t, b.m)
	inner := httptest.NewUnstartedServer(http.HandlerFunc(b.m.handlePeerServices))
	inner.Listener = reverseForwardListener{Listener: inner.Listener, manager: b.m}
	inner.Config.ConnContext = b.m.reverseConnContext
	inner.TLS = b.m.buildServerTLSConfig()
	inner.StartTLS()
	defer inner.Close()
	setReverseServices(t, b.m, 2, noderec.RegisterParams{Service: noderec.ServiceCluster, Port: inner.Listener.Addr().(*net.TCPAddr).Port}, noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 26441})
	startPrivateReverse(t, a.m)
	connectReverseManagers(t, a.m, b.m)
	reverse := countClusterReverseDials(t, a.m)
	var published bytes.Buffer
	a.m.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &published})
	a.m.serviceRefresh.source = "reverse-process"
	if err := a.m.refreshPeerServices(context.Background(), b.m.identity.NodeUUID); err != nil {
		t.Fatal(err)
	}
	msg, err := NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(published.Bytes()), io.Discard}).Read()
	if err != nil {
		t.Fatal(err)
	}
	var claim noderec.PeerServicesClaim
	if err := json.Unmarshal(msg.Params, &claim); err != nil {
		t.Fatal(err)
	}
	if reverse.Load() != 1 || claim.Descriptor.Addresses[0] != "192.168.42.8" || len(claim.Descriptor.Services) != 2 {
		t.Fatalf("reverse calls=%d descriptor=%+v", reverse.Load(), claim.Descriptor)
	}
	member, ok := b.m.memberByNodeID(a.m.identity.NodeUUID)
	if !ok || member.IPAddress != "localhost" || member.Port != a.m.port {
		t.Fatalf("reverse gateway poisoned member locator: %+v", member)
	}
}

func TestPeerServicesUnavailableRoutePreservesMembership(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, true)
	pairClusterFixtures(t, a, b)
	persistPeerServicesMembers(t, a.m, b.m)
	startPrivateReverse(t, a.m)
	countClusterReverseDials(t, a.m)
	before, ok := a.m.mesh.PeerServiceIncarnation(b.m.identity.NodeUUID)
	if !ok {
		t.Fatal("missing initial authority")
	}
	if err := a.m.refreshPeerServices(context.Background(), b.m.identity.NodeUUID); err == nil {
		t.Fatal("unavailable direct/reverse route succeeded")
	}
	a.m.mesh.Refresh()
	after, ok := a.m.mesh.PeerServiceIncarnation(b.m.identity.NodeUUID)
	if !ok || after != before {
		t.Fatal("route failure altered durable admission/pin")
	}
	member, ok := a.m.memberByNodeID(b.m.identity.NodeUUID)
	if !ok || member.State != stateMember {
		t.Fatalf("route failure altered membership: %+v", member)
	}
}
