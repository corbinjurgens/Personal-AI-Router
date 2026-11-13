// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

// genLeaf mints an Ed25519 self-signed leaf carrying uuid in CN + urn:nvpair:node
// URI SAN, matching nvpair-cluster-manager's generateLeaf.
func genLeaf(t *testing.T, uuid string) (certPEM, keyPEM []byte) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("genkey: %v", err)
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		t.Fatalf("serial: %v", err)
	}
	uri, err := url.Parse("urn:nvpair:node:" + uuid)
	if err != nil {
		t.Fatalf("certificate URI: %v", err)
	}
	tmpl := &x509.Certificate{
		SerialNumber:          serial,
		Subject:               pkix.Name{CommonName: uuid},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().AddDate(1, 0, 0),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		URIs:                  []*url.URL{uri},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, pub, priv)
	if err != nil {
		t.Fatalf("create cert: %v", err)
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		t.Fatalf("marshal key: %v", err)
	}
	certPEM = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM = pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER})
	return certPEM, keyPEM
}

// setupNode writes a cluster dir holding this node's identity (node.crt/key) and
// a pin (trusted/<uuid>.json) for each peer uuid -> cert PEM in pins.
func setupNode(t *testing.T, certPEM, keyPEM []byte, pins map[string][]byte) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "cluster")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "admission.json"), []byte(`{"clusterId":"test-cluster","epoch":1,"counter":1,"activated":1}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "node.crt"), certPEM, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "node.key"), keyPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	td := filepath.Join(dir, "trusted")
	if err := os.MkdirAll(td, 0o700); err != nil {
		t.Fatal(err)
	}
	var members []map[string]any
	for uuid, pcert := range pins {
		body := mustJSON(t, struct {
			NodeUUID       string `json:"nodeUuid"`
			CertPEM        string `json:"certPem"`
			ClusterID      string `json:"clusterId"`
			AdmissionEpoch uint64 `json:"admissionEpoch"`
		}{uuid, string(pcert), "test-cluster", 1})
		if err := os.WriteFile(filepath.Join(td, uuid+".json"), body, 0o600); err != nil {
			t.Fatal(err)
		}
		members = append(members, map[string]any{"nodeUuid": uuid, "state": "member", "clusterId": "test-cluster", "admissionEpoch": 1})
	}
	if err := os.WriteFile(filepath.Join(dir, "members.json"), mustJSON(t, members), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

// servePinnedErrorsMux starts mgr's errors surface behind cluster mTLS, wired to
// a publisher as startPeerSync wires it, and returns a client that dials it as a
// pinned peer. The surface is mTLS unconditionally, so there is no plaintext way
// to reach the handlers — every handler test goes through here. The client has
// no total timeout so it can hold a stream open.
func servePinnedErrorsMux(t *testing.T, mgr *Manager) (*httptest.Server, *http.Client) {
	t.Helper()
	selfCert, selfKey := genLeaf(t, "uuid-self")
	peerCert, peerKey := genLeaf(t, "uuid-peer")
	selfMesh := clustertrust.Open(setupNode(t, selfCert, selfKey, map[string][]byte{"uuid-peer": peerCert}))
	peerMesh := clustertrust.Open(setupNode(t, peerCert, peerKey, map[string][]byte{"uuid-self": selfCert}))

	pub := newErrorsPublisher()
	pub.publish(mgr)
	mgr.SetOnLocalChange(func() { pub.publish(mgr) })
	srv := httptest.NewUnstartedServer(newErrorsMux(mgr, selfMesh, pub))
	srv.TLS = selfMesh.ServerTLSConfig()
	srv.StartTLS()
	t.Cleanup(srv.Close)
	return srv, meshClient(t, peerMesh, "uuid-self")
}

// meshClient dials peerUUID with mesh's pinned client configuration.
func meshClient(t *testing.T, mesh *clustertrust.Mesh, peerUUID string) *http.Client {
	t.Helper()
	cfg, ok := mesh.ClientTLSConfig(peerUUID)
	if !ok {
		t.Fatalf("no pin for %s", peerUUID)
	}
	transport := &http.Transport{TLSClientConfig: cfg, ResponseHeaderTimeout: 5 * time.Second}
	t.Cleanup(transport.CloseIdleConnections)
	return &http.Client{Transport: transport}
}

// TestErrorsPeerSync_MTLSGate is the end-to-end evidence: the error stream over
// mTLS authenticates and gates to pinned cluster members. A pinned member is
// accepted; a node that can complete the TLS handshake but is not pinned by the
// receiver is rejected at the pin gate; and subscribing to a peer we hold no pin
// for can't even build a client.
func TestErrorsPeerSync_MTLSGate(t *testing.T) {
	aCert, aKey := genLeaf(t, "uuid-a")
	bCert, bKey := genLeaf(t, "uuid-b")
	cCert, cKey := genLeaf(t, "uuid-c")

	// A trusts only B. B trusts A. C trusts A (so C can dial A) but A does NOT
	// trust C — the asymmetry that proves the receiver-side gate.
	dirA := setupNode(t, aCert, aKey, map[string][]byte{"uuid-b": bCert})
	dirB := setupNode(t, bCert, bKey, map[string][]byte{"uuid-a": aCert})
	dirC := setupNode(t, cCert, cKey, map[string][]byte{"uuid-a": aCert})

	mtlsA, mtlsB, mtlsC := clustertrust.Open(dirA), clustertrust.Open(dirB), clustertrust.Open(dirC)
	if !mtlsA.Clustered() || !mtlsB.Clustered() || !mtlsC.Clustered() {
		t.Fatal("a populated cluster dir must read as clustered")
	}

	// A serves its errors stream over mTLS.
	mgrA := managerForNode("uuid-a")
	pub := newErrorsPublisher()
	pub.publish(mgrA)
	srv := httptest.NewUnstartedServer(newErrorsMux(mgrA, mtlsA, pub))
	srv.TLS = mtlsA.ServerTLSConfig()
	srv.StartTLS()
	defer srv.Close()

	subscribe := func(m *clustertrust.Mesh) int {
		t.Helper()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/v1/errors/events", nil)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := meshClient(t, m, "uuid-a").Do(req)
		if err != nil {
			t.Fatalf("subscribe: %v", err)
		}
		defer resp.Body.Close()
		return resp.StatusCode
	}

	// Pinned member B -> A: accepted.
	if code := subscribe(mtlsB); code != http.StatusOK {
		t.Fatalf("pinned member stream: code=%d, want 200", code)
	}

	// C completes the handshake (it pins A) but A doesn't pin C -> 403 at the gate.
	if code := subscribe(mtlsC); code != http.StatusForbidden {
		t.Fatalf("non-member stream: code=%d, want 403", code)
	}

	// Client-side gate: B holds no pin for an unknown peer, so the DER lookup
	// fails and it would never build a client to contact a non-member.
	if mtlsB.HasPin("uuid-unknown") {
		t.Fatal("an unpinned peer must not resolve a pin (cluster gate)")
	}
}
