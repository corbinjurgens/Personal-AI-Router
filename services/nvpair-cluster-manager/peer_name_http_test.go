// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// setTestGrant records target's local remote-management decision for peer
// through the production CAS writer.
func setTestGrant(t *testing.T, target *Manager, peer string, allowed bool) {
	t.Helper()
	target.rosterMu.Lock()
	defer target.rosterMu.Unlock()
	doc, _, err := target.readRemoteManagementLocked()
	if err != nil {
		t.Fatalf("read grants: %v", err)
	}
	if err := target.setRemoteManagementLocked(peer, allowed, doc.Revision); err != nil {
		t.Fatalf("set grant: %v", err)
	}
}

// peerNameRequest builds a PUT to peerNodeNamePath authenticated as caller's
// leaf, as the mTLS server presents it to the handler.
func peerNameRequest(t *testing.T, method string, caller *Manager, body string) *http.Request {
	t.Helper()
	r := httptest.NewRequest(method, "https://peer"+peerNodeNamePath, strings.NewReader(body))
	if caller != nil {
		cert, err := x509.ParseCertificate(caller.identity.Cert.Certificate[0])
		if err != nil {
			t.Fatal(err)
		}
		r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{cert}}
	}
	return r
}

func servePeerName(target *Manager, r *http.Request) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	target.handlePeerNodeName(w, r)
	return w
}

func decodePeerNameError(t *testing.T, w *httptest.ResponseRecorder) peerNodeNameError {
	t.Helper()
	var body peerNodeNameError
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode error body %q: %v", w.Body.String(), err)
	}
	return body
}

// peerNameFixture pairs target and caller; target grants caller remote
// management when allowed is true.
func peerNameFixture(t *testing.T, allowed bool) (target, caller *Manager, out *syncBuffer) {
	t.Helper()
	target, caller = newReverseTestManager(t), newReverseTestManager(t)
	pairReverseManagers(t, target, caller)
	// Mesh reads the durable roster when it evaluates a grant.
	for _, m := range []*Manager{target, caller} {
		if err := m.persistMembersErr(); err != nil {
			t.Fatal(err)
		}
	}
	if allowed {
		setTestGrant(t, target, caller.identity.NodeUUID, true)
	}
	return target, caller, captureCodec(target)
}

func TestPeerNodeNameAllowedCallerRenames(t *testing.T) {
	target, caller, out := peerNameFixture(t, true)
	w := servePeerName(target, peerNameRequest(t, http.MethodPut, caller, `{"name":" Desk PC "}`))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d body %q, want 200", w.Code, w.Body.String())
	}
	var view nodeNameView
	if err := json.Unmarshal(w.Body.Bytes(), &view); err != nil {
		t.Fatal(err)
	}
	if view.NodeUUID != target.identity.NodeUUID || view.Name != "Desk PC" || view.CustomName == nil || *view.CustomName != "Desk PC" {
		t.Fatalf("answer = %+v, want target renamed to Desk PC", view)
	}
	if got := readIdentityFile(t, target).DisplayName; got != "Desk PC" {
		t.Fatalf("identity.json display_name = %q, want Desk PC", got)
	}
	notes := nameNotifications(t, out)
	if len(notes) != 1 || notes[0].Name != "Desk PC" {
		t.Fatalf("cluster:node-name-changed = %+v, want one with Desk PC", notes)
	}
}

func TestPeerNodeNameAllowedCallerClearsWithNull(t *testing.T) {
	target, caller, _ := peerNameFixture(t, true)
	if w := servePeerName(target, peerNameRequest(t, http.MethodPut, caller, `{"name":"Temp"}`)); w.Code != http.StatusOK {
		t.Fatalf("set status %d", w.Code)
	}
	w := servePeerName(target, peerNameRequest(t, http.MethodPut, caller, `{"name":null}`))
	if w.Code != http.StatusOK {
		t.Fatalf("clear status %d body %q", w.Code, w.Body.String())
	}
	var view nodeNameView
	if err := json.Unmarshal(w.Body.Bytes(), &view); err != nil {
		t.Fatal(err)
	}
	if view.CustomName != nil || view.Name != defaultNodeName(target.identity.NodeUUID) {
		t.Fatalf("answer = %+v, want the generated default", view)
	}
}

func TestPeerNodeNameRefusesWithoutGrant(t *testing.T) {
	for _, tt := range []struct {
		name      string
		state     string
		configure func(t *testing.T, target, caller *Manager)
	}{
		{"unreviewed", "unreviewed", func(*testing.T, *Manager, *Manager) {}},
		{"denied", "denied", func(t *testing.T, target, caller *Manager) {
			setTestGrant(t, target, caller.identity.NodeUUID, false)
		}},
		{"unavailable", "unavailable", func(t *testing.T, target, caller *Manager) {
			setTestGrant(t, target, caller.identity.NodeUUID, true)
			_, marker := target.remoteManagementPaths()
			if err := os.Remove(marker); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			target, caller, out := peerNameFixture(t, false)
			tt.configure(t, target, caller)
			if got := target.remoteManagementFor(caller.identity.NodeUUID); got.Allowed || got.State != tt.state {
				t.Fatalf("fixture permission = %+v, want state %s", got, tt.state)
			}
			before := target.selfNameView()
			w := servePeerName(target, peerNameRequest(t, http.MethodPut, caller, `{"name":"Hijacked"}`))
			if w.Code != http.StatusConflict {
				t.Fatalf("status %d, want 409", w.Code)
			}
			body := decodePeerNameError(t, w)
			if body.Code != "remote_management_denied" ||
				body.Error != "Remote management is blocked on this device. Change permission locally on that device." {
				t.Fatalf("denial body = %+v", body)
			}
			if after := target.selfNameView(); after.Name != before.Name || after.CustomName != nil {
				t.Fatalf("name changed to %+v without a grant", after)
			}
			if len(nameNotifications(t, out)) != 0 {
				t.Fatal("refused rename emitted cluster:node-name-changed")
			}
		})
	}
}

func TestPeerNodeNameRefusesUnauthenticatedAndWrongMethod(t *testing.T) {
	target, caller, _ := peerNameFixture(t, true)
	stranger := newReverseTestManager(t)
	for _, tt := range []struct {
		name string
		r    *http.Request
		want int
	}{
		{"unpinned caller", peerNameRequest(t, http.MethodPut, stranger, `{"name":"x"}`), http.StatusForbidden},
		{"plaintext caller", peerNameRequest(t, http.MethodPut, nil, `{"name":"x"}`), http.StatusForbidden},
		{"GET", peerNameRequest(t, http.MethodGet, caller, ""), http.StatusMethodNotAllowed},
		{"POST", peerNameRequest(t, http.MethodPost, caller, `{"name":"x"}`), http.StatusMethodNotAllowed},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if w := servePeerName(target, tt.r); w.Code != tt.want {
				t.Fatalf("status %d, want %d", w.Code, tt.want)
			}
			if target.customSelfName() != nil {
				t.Fatal("refused request changed the name")
			}
		})
	}
}

func TestPeerNodeNameRejectsMalformedBodies(t *testing.T) {
	target, caller, _ := peerNameFixture(t, true)
	for _, tt := range []struct {
		name, body, wantReason string
	}{
		{"unknown key", `{"name":"x","extra":1}`, ""},
		{"missing name", `{}`, ""},
		{"number name", `{"name":5}`, ""},
		{"trailing data", `{"name":"x"} {}`, ""},
		{"not JSON", `name=x`, ""},
		{"case-variant key", `{"NAME":"x"}`, ""},
		{"duplicate key", `{"name":"a","name":"b"}`, ""},
		{"invalid name", `{"name":"bad\u0007name"}`, "invalid-character"},
		{"empty name", `{"name":"  "}`, "empty"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			w := servePeerName(target, peerNameRequest(t, http.MethodPut, caller, tt.body))
			if w.Code != http.StatusBadRequest {
				t.Fatalf("status %d, want 400", w.Code)
			}
			body := decodePeerNameError(t, w)
			if body.Code != "invalid_name" || body.Reason != tt.wantReason || body.Error == "" {
				t.Fatalf("body = %+v, want invalid_name reason %q", body, tt.wantReason)
			}
			if target.customSelfName() != nil {
				t.Fatal("rejected body changed the name")
			}
		})
	}
}

// The body is decoded before the grant check, so the grant is evaluated right
// before the rename: a caller without the grant gets 400 for a malformed body
// and 409 only for a well-formed one.
func TestPeerNodeNameDecodesBodyBeforeGrant(t *testing.T) {
	target, caller, _ := peerNameFixture(t, false)
	if w := servePeerName(target, peerNameRequest(t, http.MethodPut, caller, `{"bogus":1}`)); w.Code != http.StatusBadRequest {
		t.Fatalf("malformed body without grant: status %d, want 400", w.Code)
	}
	if w := servePeerName(target, peerNameRequest(t, http.MethodPut, caller, `{"name":"x"}`)); w.Code != http.StatusConflict {
		t.Fatalf("well-formed body without grant: status %d, want 409", w.Code)
	}
	if target.customSelfName() != nil {
		t.Fatal("refused request changed the name")
	}
}
