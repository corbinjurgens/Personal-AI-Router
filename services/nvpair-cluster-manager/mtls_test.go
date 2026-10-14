// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

func TestReverseUpgradeRejectsUnadmittedCertificates(t *testing.T) {
	test := func(name string, change func(*Manager, *Manager, *http.Request)) {
		t.Run(name, func(t *testing.T) {
			a, b := newReverseTestManager(t), newReverseTestManager(t)
			pairReverseManagers(t, a, b)
			cert, err := x509.ParseCertificate(b.identity.Cert.Certificate[0])
			if err != nil {
				t.Fatal(err)
			}
			request := httptest.NewRequest(http.MethodGet, "https://peer"+clustertrust.ReverseUpgradePath, nil)
			request.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{cert}}
			request.Header.Set("Connection", "Upgrade")
			request.Header.Set("Upgrade", clustertrust.ReverseUpgradeProtocol)
			request.Header.Set(clustertrust.ReverseSessionHeader, "0123456789abcdef0123456789abcdef")
			change(a, b, request)
			response := httptest.NewRecorder()
			a.handleReverseUpgrade(response, request)
			if response.Code != http.StatusForbidden {
				t.Fatalf("status %d, want forbidden", response.Code)
			}
		})
	}
	test("plain caller", func(a, b *Manager, r *http.Request) { r.TLS = nil })
	test("missing certificate", func(a, b *Manager, r *http.Request) { r.TLS.PeerCertificates = nil })
	test("unpinned certificate", func(a, b *Manager, r *http.Request) { a.trust.Forget(b.identity.NodeUUID) })
	test("pending member with exact pin", func(a, b *Manager, r *http.Request) {
		node, _ := a.memberByNodeID(b.identity.NodeUUID)
		node.State = statePendingOutbound
		a.upsertMember(node)
	})
	test("stale pin admission", func(a, b *Manager, r *http.Request) {
		node, _ := a.memberByNodeID(b.identity.NodeUUID)
		node.AdmissionEpoch++
		a.upsertMember(node)
	})
	test("pending teardown", func(a, b *Manager, r *http.Request) { a.teardownPending.Store(true) })
}

type reverseUpgradeHijackRecorder struct {
	*httptest.ResponseRecorder
	hijacked bool
}

func (w *reverseUpgradeHijackRecorder) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	w.hijacked = true
	return nil, nil, errors.New("unexpected obsolete-certificate hijack")
}

func TestReverseUpgradeBindsCandidateToAuthenticatedCertificate(t *testing.T) {
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	pairReverseManagers(t, a, b)
	cert, err := x509.ParseCertificate(b.identity.Cert.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodGet, "https://peer"+clustertrust.ReverseUpgradePath, nil)
	request.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{cert}}
	request.Header.Set("Connection", "Upgrade")
	request.Header.Set("Upgrade", clustertrust.ReverseUpgradeProtocol)
	request.Header.Set(clustertrust.ReverseSessionHeader, "0123456789abcdef0123456789abcdef")
	response := &reverseUpgradeHijackRecorder{ResponseRecorder: httptest.NewRecorder()}
	// Holding membership blocks candidate construction after the request's
	// initial exact-pin check. Stack observation establishes that boundary
	// without a production hook or an assumed scheduling delay.
	a.memMu.Lock()
	locked := true
	defer func() {
		if locked {
			a.memMu.Unlock()
		}
	}()
	done := make(chan struct{})
	go func() { a.handleReverseUpgrade(response, request); close(done) }()
	deadline := time.Now().Add(time.Second)
	for {
		stack := make([]byte, 128*1024)
		n := runtime.Stack(stack, true)
		blocked := false
		for _, trace := range strings.Split(string(stack[:n]), "\n\n") {
			if strings.Contains(trace, "memberByNodeID") && strings.Contains(trace, "handleReverseUpgrade") {
				blocked = true
				break
			}
		}
		if blocked {
			break
		}
		if time.Now().After(deadline) {
			a.memMu.Unlock()
			locked = false
			<-done
			t.Fatal("handler never reached candidate boundary after pin verification")
		}
		time.Sleep(time.Millisecond)
	}
	newCert, _, err := generateLeaf(b.identity.NodeUUID, b.identity.NodeID)
	if err != nil {
		t.Fatal(err)
	}
	fingerprint, err := certFingerprintFromPEM(newCert)
	if err != nil {
		t.Fatal(err)
	}
	if err = a.trust.Remove(b.identity.NodeUUID); err != nil {
		t.Fatal(err)
	}
	if err = a.trust.Pin(&TrustedPin{NodeUUID: b.identity.NodeUUID, NodeID: b.identity.NodeID, ClusterID: "cluster-1", AdmissionEpoch: 2, CertPem: string(newCert), CertFingerprint: fingerprint}); err != nil {
		t.Fatal(err)
	}
	node := cloneClusterNode(a.members[b.identity.NodeUUID])
	node.AdmissionEpoch = 2
	a.members[b.identity.NodeUUID] = node
	a.memMu.Unlock()
	locked = false
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("handler did not complete after replacement admission")
	}
	if response.hijacked || response.Code != http.StatusForbidden {
		t.Fatalf("old certificate crossed replacement admission: hijacked=%v status=%d", response.hijacked, response.Code)
	}
	if _, ok := a.reversePeer(b.identity.NodeUUID); !ok {
		t.Fatal("replacement admission fixture is not current")
	}
}
