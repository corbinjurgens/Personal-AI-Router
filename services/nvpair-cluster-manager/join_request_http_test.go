// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func joinHTTPManager(t *testing.T, clustered bool) *Manager {
	t.Helper()
	codec := NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), io.Discard})
	m, err := NewManager(codec, t.TempDir(), 14998)
	if err != nil {
		t.Fatalf("new manager: %v", err)
	}
	t.Cleanup(m.peerServices.CloseIdle)
	t.Cleanup(m.removalPeerServices.CloseIdle)
	t.Cleanup(m.clients.CloseIdle)
	m.joinNow = func() time.Time { return time.Unix(1800000000, 0) }
	if clustered {
		if _, _, err := m.foundCluster("HTTP fixture"); err != nil {
			t.Fatalf("found isolated cluster: %v", err)
		}
	}
	return m
}

func joinHTTPID(n int) string {
	return fmt.Sprintf("jr-%032x", n)
}

func joinHTTPCreate(id string) string {
	return fmt.Sprintf(`{"version":1,"operation":"request","requestId":%q,"requester":{"nodeUuid":"requester-fixture","name":"Requesting device"},"port":14321}`, id)
}

func joinHTTPFollow(operation, id string) string {
	return fmt.Sprintf(`{"version":1,"operation":%q,"requestId":%q}`, operation, id)
}

func joinHTTPServe(m *Manager, source, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(http.MethodPost, joinRequestPath, strings.NewReader(body))
	r.RemoteAddr = source
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	m.handleJoinRequestHTTP(w, r)
	return w
}

func joinHTTPFields(t *testing.T, data []byte, want ...string) map[string]json.RawMessage {
	t.Helper()
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatalf("decode JSON object: %v", err)
	}
	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	sort.Strings(want)
	if !reflect.DeepEqual(keys, want) {
		t.Fatalf("JSON fields = %v, want %v", keys, want)
	}
	return fields
}

func joinHTTPReply(t *testing.T, w *httptest.ResponseRecorder) joinRequestReply {
	t.Helper()
	if w.Code != http.StatusOK {
		t.Fatalf("HTTP status = %d, want %d", w.Code, http.StatusOK)
	}
	if got := w.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	joinHTTPFields(t, w.Body.Bytes(), "version", "requestId", "state", "reason", "createdAt", "expiresAt", "inviteId")
	var reply joinRequestReply
	if err := json.Unmarshal(w.Body.Bytes(), &reply); err != nil {
		t.Fatalf("decode join-request reply: %v", err)
	}
	if reply.Version != 1 {
		t.Fatalf("reply version = %d, want 1", reply.Version)
	}
	return reply
}

func joinHTTPError(t *testing.T, w *httptest.ResponseRecorder, status int, reason string) {
	t.Helper()
	if w.Code != status {
		t.Fatalf("HTTP status = %d, want %d (%s)", w.Code, status, reason)
	}
	if got := w.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	joinHTTPFields(t, w.Body.Bytes(), "version", "reason")
	var reply struct {
		Version int    `json:"version"`
		Reason  string `json:"reason"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &reply); err != nil {
		t.Fatalf("decode error reply: %v", err)
	}
	if reply.Version != 1 || reply.Reason != reason {
		t.Fatalf("error reply = %+v, want version 1 and reason %q", reply, reason)
	}
	if status == http.StatusTooManyRequests {
		seconds, err := strconv.Atoi(w.Header().Get("Retry-After"))
		if err != nil || seconds <= 0 {
			t.Fatalf("Retry-After must be positive seconds, got %q", w.Header().Get("Retry-After"))
		}
	}
}

func TestJoinRequestHTTPCreateReplyShape(t *testing.T) {
	m := joinHTTPManager(t, true)
	id := joinHTTPID(1)
	reply := joinHTTPReply(t, joinHTTPServe(m, "192.168.1.7:43210", joinHTTPCreate(id)))
	if reply.RequestID != id || reply.State != "awaiting-approval" || reply.Reason != "" || reply.InviteID != nil {
		t.Fatal("create must return the exact request ID, pending state, empty reason, and null inviteId")
	}
	now := m.joinTime()
	if reply.CreatedAt != now.UnixMilli() || reply.ExpiresAt != now.Add(m.effectiveInviteTTL()).UnixMilli() {
		t.Fatal("create deadlines must use the manager's clock and effective invite TTL")
	}
	r := m.joinRequests[id]
	if r == nil || r.Direction != "incoming" || r.PeerNodeUUID != "requester-fixture" || r.PeerName != "Requesting device" || r.PeerAddress != "192.168.1.7" || r.PeerPort != 14321 {
		t.Fatal("incoming request must retain claimed display identity, observed source IP, and validated pairing port")
	}
	if len(m.invites) != 0 || len(m.sessions) != 0 {
		t.Fatal("request creation must not start a pairing exchange")
	}
}

func TestJoinRequestHTTPStrictCreateValidation(t *testing.T) {
	valid := joinHTTPCreate(joinHTTPID(1))
	test := func(name, body, reason string) {
		t.Run(name, func(t *testing.T) {
			m := joinHTTPManager(t, true)
			joinHTTPError(t, joinHTTPServe(m, "10.2.3.4:40000", body), http.StatusBadRequest, reason)
			if len(m.joinRequests) != 0 || len(m.joinSources) != 0 {
				t.Fatal("malformed request allocated request or source state")
			}
		})
	}
	malformed := func(name, body string) { test(name, body, "malformed-request") }
	malformed("empty body", "")
	malformed("null body", "null")
	malformed("array body", "[]")
	malformed("incomplete JSON", valid[:len(valid)-1])
	malformed("trailing object", valid+" {}")
	malformed("trailing null", valid+" null")
	malformed("unknown top-level field", strings.Replace(valid, `"version":1`, `"version":1,"pin":"untrusted"`, 1))
	malformed("callback host forbidden", strings.Replace(valid, `"port":14321`, `"port":14321,"host":"10.9.9.9"`, 1))
	malformed("callback URL forbidden", strings.Replace(valid, `"port":14321`, `"port":14321,"url":"http://10.9.9.9/"`, 1))
	malformed("duplicate version", strings.Replace(valid, `"version":1`, `"version":1,"version":1`, 1))
	malformed("escaped duplicate version", strings.Replace(valid, `"version":1`, `"version":1,"\u0076ersion":1`, 1))
	malformed("duplicate operation", strings.Replace(valid, `"operation":"request"`, `"operation":"request","operation":"request"`, 1))
	malformed("duplicate request ID", strings.Replace(valid, `"requestId":`, `"requestId":"`+joinHTTPID(2)+`","requestId":`, 1))
	malformed("duplicate requester", strings.Replace(valid, `"requester":`, `"requester":null,"requester":`, 1))
	malformed("duplicate port", strings.Replace(valid, `"port":14321`, `"port":14321,"port":14321`, 1))
	malformed("unknown nested field", strings.Replace(valid, `"name":"Requesting device"`, `"name":"Requesting device","address":"10.9.9.9"`, 1))
	malformed("duplicate nested name", strings.Replace(valid, `"name":"Requesting device"`, `"name":"Requesting device","name":"Requesting device"`, 1))
	malformed("duplicate nested UUID", strings.Replace(valid, `"nodeUuid":"requester-fixture"`, `"nodeUuid":"requester-fixture","nodeUuid":"requester-fixture"`, 1))
	malformed("case-sensitive field names", strings.Replace(valid, `"port":`, `"Port":`, 1))
	test("unsupported version", strings.Replace(valid, `"version":1`, `"version":2`, 1), "unsupported-version")
	test("missing version", strings.Replace(valid, `"version":1,`, "", 1), "unsupported-version")
	test("null version", strings.Replace(valid, `"version":1`, `"version":null`, 1), "unsupported-version")
	malformed("string version", strings.Replace(valid, `"version":1`, `"version":"1"`, 1))
	malformed("fractional version", strings.Replace(valid, `"version":1`, `"version":1.5`, 1))
	malformed("unknown operation", strings.Replace(valid, `"operation":"request"`, `"operation":"approve"`, 1))
	malformed("null operation", strings.Replace(valid, `"operation":"request"`, `"operation":null`, 1))
	malformed("missing request ID", strings.Replace(valid, `"requestId":"`+joinHTTPID(1)+`",`, "", 1))
	malformed("null request ID", strings.Replace(valid, `"requestId":"`+joinHTTPID(1)+`"`, `"requestId":null`, 1))
	malformed("short request ID", strings.Replace(valid, joinHTTPID(1), "jr-1234", 1))
	malformed("uppercase request ID", strings.Replace(valid, joinHTTPID(1), "jr-"+strings.Repeat("A", 32), 1))
	malformed("nonhex request ID", strings.Replace(valid, joinHTTPID(1), "jr-"+strings.Repeat("z", 32), 1))
	malformed("missing requester", strings.Replace(valid, `"requester":{"nodeUuid":"requester-fixture","name":"Requesting device"},`, "", 1))
	malformed("null requester", strings.Replace(valid, `{"nodeUuid":"requester-fixture","name":"Requesting device"}`, "null", 1))
	malformed("missing name", strings.Replace(valid, `,"name":"Requesting device"`, "", 1))
	malformed("null name", strings.Replace(valid, `"name":"Requesting device"`, `"name":null`, 1))
	malformed("null UUID", strings.Replace(valid, `"nodeUuid":"requester-fixture"`, `"nodeUuid":null`, 1))
	malformed("control character in UUID", strings.Replace(valid, `"nodeUuid":"requester-fixture"`, `"nodeUuid":"requester\nfixture"`, 1))
	malformed("oversize UUID", strings.Replace(valid, "requester-fixture", strings.Repeat("a", 129), 1))
	malformed("oversize name", strings.Replace(valid, "Requesting device", strings.Repeat("a", 321), 1))
	malformed("missing port", strings.Replace(valid, `,"port":14321`, "", 1))
	malformed("null port", strings.Replace(valid, `"port":14321`, `"port":null`, 1))
	malformed("zero port", strings.Replace(valid, `"port":14321`, `"port":0`, 1))
	malformed("negative port", strings.Replace(valid, `"port":14321`, `"port":-1`, 1))
	malformed("port above range", strings.Replace(valid, `"port":14321`, `"port":65536`, 1))
	malformed("string port", strings.Replace(valid, `"port":14321`, `"port":"14321"`, 1))
	malformed("fractional port", strings.Replace(valid, `"port":14321`, `"port":14321.5`, 1))
}

func TestJoinRequestHTTPStrictFollowValidation(t *testing.T) {
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) {
			m := joinHTTPManager(t, true)
			joinHTTPError(t, joinHTTPServe(m, "10.2.3.4:40000", body), http.StatusBadRequest, "malformed-request")
			if len(m.joinRequests) != 0 || len(m.joinSources) != 0 {
				t.Fatal("invalid follow-up allocated request or source state")
			}
		})
	}
	status := joinHTTPFollow("status", joinHTTPID(1))
	ready := joinHTTPFollow("ready", joinHTTPID(1))
	cancel := joinHTTPFollow("cancel", joinHTTPID(1))
	test("status cannot carry requester", strings.Replace(status, `"version":1`, `"version":1,"requester":null`, 1))
	test("ready cannot carry port", strings.Replace(ready, `"version":1`, `"version":1,"port":null`, 1))
	test("cancel cannot carry consent", strings.Replace(cancel, `"version":1`, `"version":1,"allowRemoteManagement":true`, 1))
	test("status rejects duplicate operation", strings.Replace(status, `"version":1`, `"version":1,"operation":"status"`, 1))
	test("ready rejects trailing JSON", ready+" {}")
	test("cancel rejects null ID", strings.Replace(cancel, `"requestId":"`+joinHTTPID(1)+`"`, `"requestId":null`, 1))
}

func TestJoinRequestHTTPBodyLimit(t *testing.T) {
	test := func(name string, size, status int) {
		t.Run(name, func(t *testing.T) {
			m := joinHTTPManager(t, true)
			body := joinHTTPCreate(joinHTTPID(1))
			body += strings.Repeat(" ", size-len(body))
			w := joinHTTPServe(m, "10.2.3.4:40000", body)
			if status == http.StatusOK {
				joinHTTPReply(t, w)
			} else {
				joinHTTPError(t, w, status, "malformed-request")
				if len(m.joinRequests) != 0 || len(m.joinSources) != 0 {
					t.Fatal("oversize body allocated state")
				}
			}
		})
	}
	test("exactly 4 KiB accepted", 4096, http.StatusOK)
	test("one byte over 4 KiB refused", 4097, http.StatusBadRequest)
}

func TestJoinRequestHTTPPortBoundaries(t *testing.T) {
	test := func(name string, port int) {
		t.Run(name, func(t *testing.T) {
			m := joinHTTPManager(t, true)
			id := joinHTTPID(1)
			body := strings.Replace(joinHTTPCreate(id), `"port":14321`, fmt.Sprintf(`"port":%d`, port), 1)
			joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40000", body))
			if m.joinRequests[id].PeerPort != port {
				t.Fatalf("pairing port = %d, want %d", m.joinRequests[id].PeerPort, port)
			}
		})
	}
	test("lowest valid port", 1)
	test("highest valid port", 65535)
}

func TestJoinRequestHTTPAllowedSources(t *testing.T) {
	test := func(name, source, canonical string) {
		t.Run(name, func(t *testing.T) {
			m := joinHTTPManager(t, true)
			id := joinHTTPID(1)
			joinHTTPReply(t, joinHTTPServe(m, source, joinHTTPCreate(id)))
			if m.joinRequests[id].PeerAddress != canonical || m.joinRequests[id].source != canonical {
				t.Fatalf("callback/source must be canonical observed IP %q", canonical)
			}
		})
	}
	test("private 10", "10.0.0.1:40000", "10.0.0.1")
	test("private 172 lower boundary", "172.16.0.1:40000", "172.16.0.1")
	test("private 172 upper boundary", "172.31.255.254:40000", "172.31.255.254")
	test("private 192", "192.168.1.1:40000", "192.168.1.1")
	test("Tailscale lower boundary", "100.64.0.1:40000", "100.64.0.1")
	test("Tailscale upper boundary", "100.127.255.254:40000", "100.127.255.254")
	test("IPv6 ULA fc", "[fc00::123]:40000", "fc00::123")
	test("IPv6 ULA fd", "[fdff::123]:40000", "fdff::123")
	test("IPv4 loopback", "127.0.0.1:40000", "127.0.0.1")
	test("IPv6 loopback", "[::1]:40000", "::1")
	test("mapped private IPv4", "[::ffff:192.168.1.1]:40000", "192.168.1.1")
	test("mapped Tailscale IPv4", "[::ffff:100.64.0.1]:40000", "100.64.0.1")
	test("mapped loopback IPv4", "[::ffff:127.0.0.1]:40000", "127.0.0.1")
}

func TestJoinRequestHTTPRefusedSources(t *testing.T) {
	test := func(name, source string) {
		t.Run(name, func(t *testing.T) {
			m := joinHTTPManager(t, true)
			joinHTTPError(t, joinHTTPServe(m, source, joinHTTPCreate(joinHTTPID(1))), http.StatusForbidden, "source-not-allowed")
			if len(m.joinSources) != 0 || len(m.joinRequests) != 0 {
				t.Fatal("refused source allocated state")
			}
		})
	}
	test("public IPv4", "8.8.8.8:40000")
	test("below private 172 range", "172.15.255.254:40000")
	test("above private 172 range", "172.32.0.1:40000")
	test("below Tailscale range", "100.63.255.254:40000")
	test("above Tailscale range", "100.128.0.1:40000")
	test("IPv4 link local", "169.254.1.1:40000")
	test("IPv4 multicast", "224.0.0.1:40000")
	test("IPv4 unspecified", "0.0.0.0:40000")
	test("IPv4 broadcast", "255.255.255.255:40000")
	test("IPv6 public", "[2001:4860:4860::8888]:40000")
	test("IPv6 link local", "[fe80::1]:40000")
	test("IPv6 multicast", "[ff02::1]:40000")
	test("IPv6 unspecified", "[::]:40000")
	test("IPv6 scoped address", "[fd00::1%en0]:40000")
	test("mapped public IPv4", "[::ffff:8.8.8.8]:40000")
	test("mapped link local IPv4", "[::ffff:169.254.1.1]:40000")
	test("hostname", "requester.example:40000")
	test("missing source port", "192.168.1.1")
}

func TestJoinRequestHTTPForwardedHeadersIgnored(t *testing.T) {
	t.Run("private header cannot authorize public transport", func(t *testing.T) {
		m := joinHTTPManager(t, true)
		r := httptest.NewRequest(http.MethodPost, joinRequestPath, strings.NewReader(joinHTTPCreate(joinHTTPID(1))))
		r.RemoteAddr = "8.8.8.8:40000"
		r.Header.Set("X-Forwarded-For", "192.168.1.7")
		r.Header.Set("Forwarded", "for=192.168.1.7")
		r.Header.Set("X-Real-IP", "192.168.1.7")
		w := httptest.NewRecorder()
		m.handleJoinRequestHTTP(w, r)
		joinHTTPError(t, w, http.StatusForbidden, "source-not-allowed")
		if len(m.joinRequests) != 0 || len(m.joinSources) != 0 {
			t.Fatal("spoofed headers allocated state")
		}
	})
	t.Run("header cannot replace callback address", func(t *testing.T) {
		m := joinHTTPManager(t, true)
		id := joinHTTPID(1)
		r := httptest.NewRequest(http.MethodPost, joinRequestPath, strings.NewReader(joinHTTPCreate(id)))
		r.RemoteAddr = "192.168.1.7:40000"
		r.Header.Set("X-Forwarded-For", "10.9.9.9")
		r.Header.Set("Forwarded", "for=10.9.9.9")
		w := httptest.NewRecorder()
		m.handleJoinRequestHTTP(w, r)
		joinHTTPReply(t, w)
		if m.joinRequests[id].PeerAddress != "192.168.1.7" || m.joinSources["10.9.9.9"] != nil {
			t.Fatal("forwarded header replaced the transport source")
		}
	})
}

func TestJoinRequestHTTPFollowSourceBinding(t *testing.T) {
	test := func(operation string) {
		t.Run(operation, func(t *testing.T) {
			m := joinHTTPManager(t, true)
			id := joinHTTPID(1)
			created := joinHTTPReply(t, joinHTTPServe(m, "192.168.1.7:40000", joinHTTPCreate(id)))
			r := httptest.NewRequest(http.MethodPost, joinRequestPath, strings.NewReader(joinHTTPFollow(operation, id)))
			r.RemoteAddr = "192.168.1.8:40001"
			r.Header.Set("X-Forwarded-For", "192.168.1.7")
			w := httptest.NewRecorder()
			m.handleJoinRequestHTTP(w, r)
			joinHTTPError(t, w, http.StatusNotFound, "unknown-request")
			current := joinHTTPReply(t, joinHTTPServe(m, "[::ffff:192.168.1.7]:49999", joinHTTPFollow("status", id)))
			if !reflect.DeepEqual(current, created) {
				t.Fatal("wrong-source follow-up mutated the request or mapped source lost ownership")
			}
		})
	}
	test("status")
	test("ready")
	test("cancel")
}

func TestJoinRequestHTTPFollowUnknownID(t *testing.T) {
	test := func(operation string) {
		t.Run(operation, func(t *testing.T) {
			m := joinHTTPManager(t, true)
			joinHTTPError(t, joinHTTPServe(m, "10.2.3.4:40000", joinHTTPFollow(operation, joinHTTPID(1))), http.StatusNotFound, "unknown-request")
			if len(m.joinRequests) != 0 {
				t.Fatal("unknown status/ready created a request")
			}
		})
	}
	test("status")
	test("ready")
}

func TestJoinRequestHTTPApprovedFollowSourceBinding(t *testing.T) {
	test := func(operation string) {
		t.Run(operation, func(t *testing.T) {
			m := joinHTTPManager(t, true)
			id := joinHTTPID(1)
			joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40000", joinHTTPCreate(id)))
			rpcID := json.RawMessage(`1`)
			params := json.RawMessage(fmt.Sprintf(`{"requestId":%q,"accept":true,"allowRemoteManagement":false}`, id))
			m.handleRespondToJoinRequest(&Message{JSONRPC: "2.0", ID: &rpcID, Params: params})
			approved := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40001", joinHTTPFollow("status", id)))
			if approved.State != "awaiting-requester" || approved.InviteID == nil || *approved.InviteID == "" {
				t.Fatal("local approval must reserve an exact child invite before the source-binding check")
			}
			joinHTTPError(t, joinHTTPServe(m, "10.2.3.5:40002", joinHTTPFollow(operation, id)), http.StatusNotFound, "unknown-request")
			current := joinHTTPReply(t, joinHTTPServe(m, "[::ffff:10.2.3.4]:40003", joinHTTPFollow("status", id)))
			if !reflect.DeepEqual(current, approved) || m.joinRequests[id].ready || len(m.invites) != 0 || len(m.sessions) != 0 {
				t.Fatal("wrong-source follow-up changed approval or started the child invite")
			}
		})
	}
	test("status")
	test("ready")
	test("cancel")
}

func TestJoinRequestHTTPReadyRequiresApproval(t *testing.T) {
	m := joinHTTPManager(t, true)
	id := joinHTTPID(1)
	joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40000", joinHTTPCreate(id)))
	joinHTTPError(t, joinHTTPServe(m, "10.2.3.4:40001", joinHTTPFollow("ready", id)), http.StatusConflict, "approval-required")
	if m.joinRequests[id].State != "awaiting-approval" || m.joinRequests[id].ready || len(m.invites) != 0 || len(m.sessions) != 0 {
		t.Fatal("unapproved ready started pairing or changed the request")
	}
}

func TestJoinRequestHTTPDuplicateDoesNotRefresh(t *testing.T) {
	m := joinHTTPManager(t, true)
	now := m.joinTime()
	m.joinNow = func() time.Time { return now }
	id := joinHTTPID(1)
	created := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40000", joinHTTPCreate(id)))
	now = now.Add(2 * time.Minute)
	duplicate := joinHTTPReply(t, joinHTTPServe(m, "[::ffff:10.2.3.4]:40001", joinHTTPCreate(id)))
	if !reflect.DeepEqual(duplicate, created) || len(m.joinRequests) != 1 {
		t.Fatal("duplicate create changed deadlines/state or allocated a second review card")
	}
	now = time.UnixMilli(created.ExpiresAt)
	expired := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40002", joinHTTPCreate(id)))
	if expired.State != "expired" || expired.CreatedAt != created.CreatedAt || expired.ExpiresAt != created.ExpiresAt {
		t.Fatal("duplicate create refreshed or resurrected an expired request")
	}
}

func TestJoinRequestHTTPConflictingDuplicate(t *testing.T) {
	test := func(name, original, changed string) {
		t.Run(name, func(t *testing.T) {
			m := joinHTTPManager(t, true)
			created := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40000", original))
			joinHTTPError(t, joinHTTPServe(m, "10.2.3.4:40001", changed), http.StatusConflict, "conflicting-request")
			current := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40002", joinHTTPFollow("status", created.RequestID)))
			if !reflect.DeepEqual(current, created) {
				t.Fatal("conflicting reuse changed original request")
			}
		})
	}
	valid := joinHTTPCreate(joinHTTPID(1))
	test("different UUID", valid, strings.Replace(valid, "requester-fixture", "other-requester", 1))
	test("different name", valid, strings.Replace(valid, "Requesting device", "Other device", 1))
	test("different port", valid, strings.Replace(valid, "14321", "14322", 1))
	test("original empty name still binds claims", strings.Replace(valid, "Requesting device", "", 1), valid)
	test("sanitized-equal names still conflict", strings.Replace(valid, "Requesting device", `Requesting\ndevice`, 1), strings.Replace(valid, "Requesting device", "Requestingdevice", 1))
}

func TestJoinRequestHTTPCancelBeforeCreate(t *testing.T) {
	m := joinHTTPManager(t, true)
	id := joinHTTPID(1)
	canceled := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40000", joinHTTPFollow("cancel", id)))
	if canceled.State != "canceled" || canceled.InviteID != nil {
		t.Fatal("early cancel must create a canceled tombstone without an invite")
	}
	late := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40001", joinHTTPCreate(id)))
	if !reflect.DeepEqual(late, canceled) || len(m.joinRequests) != 1 {
		t.Fatal("reordered create resurrected or refreshed an early cancellation")
	}
	joinHTTPError(t, joinHTTPServe(m, "10.2.3.5:40000", joinHTTPCreate(id)), http.StatusNotFound, "unknown-request")
	ready := joinHTTPReply(t, joinHTTPServe(m, "[::ffff:10.2.3.4]:40002", joinHTTPFollow("ready", id)))
	if !reflect.DeepEqual(ready, canceled) || len(m.invites) != 0 || len(m.sessions) != 0 {
		t.Fatal("late ready or source conflict resurrected a cancellation")
	}
}

func TestJoinRequestHTTPCancelIsSourceBoundAndIdempotent(t *testing.T) {
	m := joinHTTPManager(t, true)
	id := joinHTTPID(1)
	created := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40000", joinHTTPCreate(id)))
	canceled := joinHTTPReply(t, joinHTTPServe(m, "[::ffff:10.2.3.4]:40001", joinHTTPFollow("cancel", id)))
	if canceled.State != "canceled" || canceled.CreatedAt != created.CreatedAt || canceled.ExpiresAt != created.ExpiresAt {
		t.Fatal("owner cancel must terminate the exact request without changing deadlines")
	}
	again := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40002", joinHTTPFollow("cancel", id)))
	if !reflect.DeepEqual(again, canceled) {
		t.Fatal("repeated owner cancel changed terminal outcome")
	}
}

func TestJoinRequestHTTPIncomingQueueBound(t *testing.T) {
	m := joinHTTPManager(t, true)
	now := m.joinTime()
	m.joinNow = func() time.Time { return now }
	for i := 1; i <= 8; i++ {
		joinHTTPReply(t, joinHTTPServe(m, fmt.Sprintf("10.0.0.%d:40000", i), joinHTTPCreate(joinHTTPID(i))))
		now = now.Add(time.Second)
	}
	joinHTTPError(t, joinHTTPServe(m, "10.0.0.9:40000", joinHTTPCreate(joinHTTPID(9))), http.StatusTooManyRequests, "request-capacity")
	if len(m.joinRequests) != 8 {
		t.Fatal("full incoming queue allocated excess request")
	}
	joinHTTPReply(t, joinHTTPServe(m, "10.0.0.1:40001", joinHTTPFollow("cancel", joinHTTPID(1))))
	now = now.Add(time.Second)
	joinHTTPReply(t, joinHTTPServe(m, "10.0.0.10:40000", joinHTTPCreate(joinHTTPID(10))))
	if len(m.joinRequests) != 9 || m.joinRequests[joinHTTPID(1)].State != "canceled" {
		t.Fatal("freeing a live slot must preserve its cancellation fence and admit another source")
	}
}

func TestJoinRequestHTTPSourceQueueBound(t *testing.T) {
	m := joinHTTPManager(t, true)
	now := m.joinTime()
	m.joinNow = func() time.Time { return now }
	joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40000", joinHTTPCreate(joinHTTPID(1))))
	now = now.Add(30 * time.Second)
	joinHTTPError(t, joinHTTPServe(m, "[::ffff:10.2.3.4]:40001", joinHTTPCreate(joinHTTPID(2))), http.StatusTooManyRequests, "source-busy")
	if len(m.joinRequests) != 1 || len(m.joinSources) != 1 {
		t.Fatal("mapped address bypassed one-live-request-per-source bound")
	}
}

func TestJoinRequestHTTPCreateSourceRate(t *testing.T) {
	m := joinHTTPManager(t, true)
	now := m.joinTime()
	m.joinNow = func() time.Time { return now }
	joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40000", joinHTTPCreate(joinHTTPID(1))))
	joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40001", joinHTTPFollow("cancel", joinHTTPID(1))))
	joinHTTPError(t, joinHTTPServe(m, "[::ffff:10.2.3.4]:40002", joinHTTPCreate(joinHTTPID(2))), http.StatusTooManyRequests, "rate-limited")
	now = now.Add(30*time.Second - time.Millisecond)
	joinHTTPError(t, joinHTTPServe(m, "10.2.3.4:40003", joinHTTPCreate(joinHTTPID(2))), http.StatusTooManyRequests, "rate-limited")
	now = now.Add(time.Millisecond)
	joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40004", joinHTTPCreate(joinHTTPID(2))))
}

func TestJoinRequestHTTPCreateGlobalRate(t *testing.T) {
	m := joinHTTPManager(t, true)
	now := m.joinTime()
	m.joinNow = func() time.Time { return now }
	for i := 1; i <= 4; i++ {
		joinHTTPReply(t, joinHTTPServe(m, fmt.Sprintf("10.0.0.%d:40000", i), joinHTTPCreate(joinHTTPID(i))))
	}
	joinHTTPError(t, joinHTTPServe(m, "10.0.0.5:40000", joinHTTPCreate(joinHTTPID(5))), http.StatusTooManyRequests, "rate-limited")
	now = now.Add(time.Second - time.Millisecond)
	joinHTTPError(t, joinHTTPServe(m, "10.0.0.6:40000", joinHTTPCreate(joinHTTPID(6))), http.StatusTooManyRequests, "rate-limited")
	now = now.Add(time.Millisecond)
	joinHTTPReply(t, joinHTTPServe(m, "10.0.0.7:40000", joinHTTPCreate(joinHTTPID(7))))
}

func TestJoinRequestHTTPFollowRateSharedAcrossOperations(t *testing.T) {
	m := joinHTTPManager(t, true)
	now := m.joinTime()
	m.joinNow = func() time.Time { return now }
	id := joinHTTPID(1)
	joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40000", joinHTTPCreate(id)))
	for i := 0; i < 8; i++ {
		joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40001", joinHTTPFollow("status", id)))
	}
	joinHTTPError(t, joinHTTPServe(m, "10.2.3.4:40002", joinHTTPFollow("ready", id)), http.StatusConflict, "approval-required")
	joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40003", joinHTTPFollow("cancel", id)))
	joinHTTPError(t, joinHTTPServe(m, "[::ffff:10.2.3.4]:40004", joinHTTPFollow("status", id)), http.StatusTooManyRequests, "rate-limited")
	now = now.Add(200*time.Millisecond - time.Nanosecond)
	joinHTTPError(t, joinHTTPServe(m, "10.2.3.4:40005", joinHTTPFollow("status", id)), http.StatusTooManyRequests, "rate-limited")
	now = now.Add(time.Nanosecond)
	joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40006", joinHTTPFollow("status", id)))
	joinHTTPError(t, joinHTTPServe(m, "10.2.3.4:40007", joinHTTPFollow("cancel", id)), http.StatusTooManyRequests, "rate-limited")
}

func TestJoinRequestHTTPSourceTableBoundAndIdleExpiry(t *testing.T) {
	m := joinHTTPManager(t, true)
	now := m.joinTime()
	m.joinNow = func() time.Time { return now }
	for i := 1; i <= 128; i++ {
		joinHTTPError(t, joinHTTPServe(m, fmt.Sprintf("10.0.0.%d:40000", i), joinHTTPFollow("status", joinHTTPID(i))), http.StatusNotFound, "unknown-request")
	}
	joinHTTPError(t, joinHTTPServe(m, "10.0.0.129:40000", joinHTTPFollow("status", joinHTTPID(129))), http.StatusTooManyRequests, "source-capacity")
	if len(m.joinSources) != 128 || len(m.joinRequests) != 0 {
		t.Fatal("source table exceeded its bound or unknown status allocated request records")
	}
	now = now.Add(4 * time.Minute)
	joinHTTPError(t, joinHTTPServe(m, "10.0.0.1:40001", joinHTTPFollow("status", joinHTTPID(1))), http.StatusNotFound, "unknown-request")
	now = now.Add(time.Minute - time.Nanosecond)
	joinHTTPError(t, joinHTTPServe(m, "10.0.0.129:40001", joinHTTPFollow("status", joinHTTPID(129))), http.StatusTooManyRequests, "source-capacity")
	now = now.Add(time.Nanosecond)
	joinHTTPError(t, joinHTTPServe(m, "10.0.0.129:40002", joinHTTPFollow("status", joinHTTPID(129))), http.StatusNotFound, "unknown-request")
	if len(m.joinSources) != 2 || m.joinSources["10.0.0.1"] == nil || m.joinSources["10.0.0.129"] == nil {
		t.Fatal("idle expiry must retain the touched source and admit the new source at five minutes")
	}
}

func TestJoinRequestHTTPTerminalCapacityPreservesTombstones(t *testing.T) {
	m := joinHTTPManager(t, true)
	start := m.joinTime()
	now := start
	m.joinNow = func() time.Time { return now }
	for i := 1; i <= 64; i++ {
		canceled := joinHTTPReply(t, joinHTTPServe(m, fmt.Sprintf("10.0.0.%d:40000", i), joinHTTPFollow("cancel", joinHTTPID(i))))
		if canceled.State != "canceled" {
			t.Fatal("cancel-before-create failed to retain tombstone")
		}
		now = now.Add(time.Second)
	}
	joinHTTPError(t, joinHTTPServe(m, "10.0.0.65:40000", joinHTTPFollow("cancel", joinHTTPID(65))), http.StatusTooManyRequests, "request-capacity")
	joinHTTPError(t, joinHTTPServe(m, "10.0.0.66:40000", joinHTTPCreate(joinHTTPID(66))), http.StatusTooManyRequests, "request-capacity")
	if len(m.joinRequests) != 64 {
		t.Fatal("terminal record cap evicted an unexpired cancellation fence or allocated excess work")
	}
	first := joinHTTPReply(t, joinHTTPServe(m, "10.0.0.1:40001", joinHTTPCreate(joinHTTPID(1))))
	if first.State != "canceled" {
		t.Fatal("full table resurrected the oldest unexpired cancellation")
	}
	now = start.Add(5*time.Minute - time.Millisecond)
	joinHTTPError(t, joinHTTPServe(m, "10.0.0.67:40000", joinHTTPFollow("cancel", joinHTTPID(67))), http.StatusTooManyRequests, "request-capacity")
	now = start.Add(5 * time.Minute)
	joinHTTPReply(t, joinHTTPServe(m, "10.0.0.68:40000", joinHTTPFollow("cancel", joinHTTPID(68))))
	if len(m.joinRequests) != 64 || m.joinRequests[joinHTTPID(1)] != nil || m.joinRequests[joinHTTPID(2)] == nil {
		t.Fatal("terminal retention must expire only elapsed fences at the five-minute boundary")
	}
}

func TestJoinRequestHTTPExpiryFencesReadyAndCancel(t *testing.T) {
	m := joinHTTPManager(t, true)
	now := m.joinTime()
	m.joinNow = func() time.Time { return now }
	id := joinHTTPID(1)
	created := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40000", joinHTTPCreate(id)))
	now = time.UnixMilli(created.ExpiresAt).Add(-time.Millisecond)
	pending := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40001", joinHTTPFollow("status", id)))
	if pending.State != "awaiting-approval" {
		t.Fatal("request expired before its local deadline")
	}
	now = now.Add(time.Millisecond)
	expired := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40002", joinHTTPFollow("ready", id)))
	if expired.State != "expired" || expired.InviteID != nil || expired.ExpiresAt != created.ExpiresAt {
		t.Fatal("ready at the deadline must return expired without handoff")
	}
	lateCancel := joinHTTPReply(t, joinHTTPServe(m, "10.2.3.4:40003", joinHTTPFollow("cancel", id)))
	if !reflect.DeepEqual(lateCancel, expired) || len(m.invites) != 0 || len(m.sessions) != 0 {
		t.Fatal("late cancel changed expired outcome or late ready created pairing state")
	}
	now = now.Add(5 * time.Minute)
	joinHTTPError(t, joinHTTPServe(m, "10.2.3.4:40004", joinHTTPFollow("status", id)), http.StatusNotFound, "unknown-request")
}

type joinHTTPBlockingBody struct {
	entered chan<- struct{}
	release <-chan struct{}
	once    sync.Once
}

func (b *joinHTTPBlockingBody) Read([]byte) (int, error) {
	b.once.Do(func() { b.entered <- struct{}{} })
	<-b.release
	return 0, io.EOF
}

func (*joinHTTPBlockingBody) Close() error { return nil }

type joinHTTPCountingBody struct{ reads atomic.Int32 }

func (b *joinHTTPCountingBody) Read([]byte) (int, error) {
	b.reads.Add(1)
	return 0, io.EOF
}

func (*joinHTTPCountingBody) Close() error { return nil }

func TestJoinRequestHTTPHandlerSemaphorePrecedesBodyRead(t *testing.T) {
	m := joinHTTPManager(t, true)
	entered := make(chan struct{}, 8)
	release := make(chan struct{})
	var unblock sync.Once
	var handlers sync.WaitGroup
	results := make(chan *httptest.ResponseRecorder, 8)
	t.Cleanup(func() {
		unblock.Do(func() { close(release) })
		handlers.Wait()
	})
	for i := 0; i < 8; i++ {
		handlers.Add(1)
		go func() {
			defer handlers.Done()
			r := httptest.NewRequest(http.MethodPost, joinRequestPath, nil)
			r.RemoteAddr = "10.2.3.4:40000"
			r.Body = &joinHTTPBlockingBody{entered: entered, release: release}
			w := httptest.NewRecorder()
			m.handleJoinRequestHTTP(w, r)
			results <- w
		}()
	}
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	for i := 0; i < 8; i++ {
		select {
		case <-entered:
		case <-deadline.C:
			t.Fatal("eight admitted handlers did not enter their body reads")
		}
	}
	body := &joinHTTPCountingBody{}
	r := httptest.NewRequest(http.MethodPost, joinRequestPath, nil)
	r.RemoteAddr = "10.2.3.5:40001"
	r.Body = body
	w := httptest.NewRecorder()
	done := make(chan struct{})
	handlers.Add(1)
	go func() {
		defer handlers.Done()
		m.handleJoinRequestHTTP(w, r)
		close(done)
	}()
	select {
	case <-done:
	case <-deadline.C:
		t.Fatal("ninth handler queued instead of returning immediately")
	}
	joinHTTPError(t, w, http.StatusTooManyRequests, "request-capacity")
	if body.reads.Load() != 0 || len(m.joinSources) != 0 || len(m.joinRequests) != 0 {
		t.Fatal("excess handler read its body or allocated source/request state")
	}
	unblock.Do(func() { close(release) })
	handlers.Wait()
	for i := 0; i < 8; i++ {
		joinHTTPError(t, <-results, http.StatusBadRequest, "malformed-request")
	}
	if len(m.joinHandlers) != 0 {
		t.Fatal("finished handlers leaked semaphore permits")
	}
	joinHTTPReply(t, joinHTTPServe(m, "10.2.3.6:40002", joinHTTPCreate(joinHTTPID(1))))
}

func TestJoinRequestHTTPUnclusteredTargetDoesNotAutoFound(t *testing.T) {
	m := joinHTTPManager(t, false)
	beforeCID, beforeEpoch := m.currentAdmission()
	joinHTTPError(t, joinHTTPServe(m, "10.2.3.4:40000", joinHTTPCreate(joinHTTPID(1))), http.StatusConflict, "target-not-clustered")
	if cid, _ := m.clusterIdentity(); cid != "" {
		t.Fatal("unsolicited request auto-founded a cluster")
	}
	if cid, epoch := m.currentAdmission(); cid != beforeCID || epoch != beforeEpoch {
		t.Fatal("unsolicited request changed durable admission")
	}
	if len(m.joinRequests) != 0 || len(m.members) != 0 || len(m.invites) != 0 || len(m.sessions) != 0 || len(m.trust.List()) != 0 {
		t.Fatal("unclustered refusal created request, membership, pairing, or trust state")
	}
}

func TestJoinRequestHTTPRejectsNonPostBeforeBodyRead(t *testing.T) {
	m := joinHTTPManager(t, true)
	body := &joinHTTPCountingBody{}
	r := httptest.NewRequest(http.MethodGet, joinRequestPath, nil)
	r.RemoteAddr = "10.2.3.4:40000"
	r.Body = body
	w := httptest.NewRecorder()
	m.handleJoinRequestHTTP(w, r)
	joinHTTPError(t, w, http.StatusMethodNotAllowed, "method-not-allowed")
	if body.reads.Load() != 0 || len(m.joinSources) != 0 || len(m.joinRequests) != 0 {
		t.Fatal("invalid HTTP method read a body or allocated state")
	}
}

func TestJoinRequestHTTPClientRequestShape(t *testing.T) {
	test := func(name string, wire joinRequestWire, fields ...string) {
		t.Run(name, func(t *testing.T) {
			type observedRequest struct {
				method, path, contentType string
				body                      []byte
				err                       error
			}
			observed := make(chan observedRequest, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				observed <- observedRequest{r.Method, r.URL.RequestURI(), r.Header.Get("Content-Type"), body, err}
				w.Header().Set("Content-Type", "application/json")
				if err := json.NewEncoder(w).Encode(joinRequestReply{Version: 1, RequestID: wire.RequestID, State: "awaiting-approval", CreatedAt: 1800000000000, ExpiresAt: 1800000300000}); err != nil {
					t.Errorf("encode isolated HTTP reply: %v", err)
				}
			}))
			defer server.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			reply, status, err := callJoinRequest(ctx, strings.TrimPrefix(server.URL, "http://"), wire)
			if err != nil || status != http.StatusOK {
				t.Fatalf("call join-request endpoint: status %d, error %v", status, err)
			}
			if reply.RequestID != wire.RequestID || reply.State != "awaiting-approval" || reply.InviteID != nil {
				t.Fatal("HTTP client changed the successful response")
			}
			got := <-observed
			if got.err != nil {
				t.Fatalf("read observed request: %v", got.err)
			}
			if got.method != http.MethodPost || got.path != "/v1/cluster/join-requests" || got.contentType != "application/json" {
				t.Fatalf("request route = %s %s (%s)", got.method, got.path, got.contentType)
			}
			bodyFields := joinHTTPFields(t, got.body, fields...)
			var body joinRequestWire
			if err := json.Unmarshal(got.body, &body); err != nil {
				t.Fatalf("decode observed request: %v", err)
			}
			if !reflect.DeepEqual(body, wire) {
				t.Fatal("HTTP client changed meaningful request fields")
			}
			if wire.Operation == "request" {
				joinHTTPFields(t, bodyFields["requester"], "nodeUuid", "name")
			}
		})
	}
	port := 14321
	id := joinHTTPID(1)
	test("request carries claims and pairing port", joinRequestWire{Version: 1, Operation: "request", RequestID: id, Requester: &joinRequester{NodeUUID: "requester-fixture", Name: "Requesting device"}, Port: &port}, "version", "operation", "requestId", "requester", "port")
	test("status carries only correlation", joinRequestWire{Version: 1, Operation: "status", RequestID: id}, "version", "operation", "requestId")
	test("ready carries only correlation", joinRequestWire{Version: 1, Operation: "ready", RequestID: id}, "version", "operation", "requestId")
	test("cancel carries only correlation", joinRequestWire{Version: 1, Operation: "cancel", RequestID: id}, "version", "operation", "requestId")
}

func TestJoinRequestHTTPClientStrictReplyValidation(t *testing.T) {
	id := joinHTTPID(1)
	valid := fmt.Sprintf(`{"version":1,"requestId":%q,"state":"awaiting-approval","reason":"","createdAt":1800000000000,"expiresAt":1800000300000,"inviteId":null}`, id)
	test := func(name, response string) {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if _, err := io.WriteString(w, response); err != nil {
					t.Errorf("write isolated HTTP reply: %v", err)
				}
			}))
			defer server.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, status, err := callJoinRequest(ctx, strings.TrimPrefix(server.URL, "http://"), joinRequestWire{Version: 1, Operation: "status", RequestID: id})
			if err == nil || status != http.StatusOK {
				t.Fatalf("malformed success reply: HTTP status %d, validation error %v", status, err)
			}
		})
	}
	test("unknown field", strings.Replace(valid, `"version":1`, `"version":1,"pin":"untrusted"`, 1))
	test("duplicate field", strings.Replace(valid, `"version":1`, `"version":1,"version":1`, 1))
	test("trailing JSON", valid+" {}")
	test("null reply", "null")
	test("unsupported version", strings.Replace(valid, `"version":1`, `"version":2`, 1))
	test("null version", strings.Replace(valid, `"version":1`, `"version":null`, 1))
	test("mismatched request ID", strings.Replace(valid, id, joinHTTPID(2), 1))
	test("unknown state", strings.Replace(valid, "awaiting-approval", "approved", 1))
	test("null state", strings.Replace(valid, `"state":"awaiting-approval"`, `"state":null`, 1))
	test("missing reason", strings.Replace(valid, `"reason":"",`, "", 1))
	test("null reason", strings.Replace(valid, `"reason":""`, `"reason":null`, 1))
	test("missing invite ID", strings.Replace(valid, `,"inviteId":null`, "", 1))
	test("numeric invite ID", strings.Replace(valid, `"inviteId":null`, `"inviteId":42`, 1))
	test("null creation time", strings.Replace(valid, `"createdAt":1800000000000`, `"createdAt":null`, 1))
	test("equal expiry and creation", strings.Replace(valid, "1800000300000", "1800000000000", 1))
	test("response exceeds 4 KiB", valid+strings.Repeat(" ", 4097-len(valid)))
}

func TestJoinRequestHTTPClientDoesNotFollowRedirect(t *testing.T) {
	var redirected atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		redirected.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer target.Close()
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+joinRequestPath, http.StatusTemporaryRedirect)
	}))
	defer origin.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, status, err := callJoinRequest(ctx, strings.TrimPrefix(origin.URL, "http://"), joinRequestWire{Version: 1, Operation: "status", RequestID: joinHTTPID(1)})
	if err == nil || status != http.StatusTemporaryRedirect || redirected.Load() != 0 {
		t.Fatalf("redirect handling: HTTP status %d, error %v, redirected requests %d", status, err, redirected.Load())
	}
}
