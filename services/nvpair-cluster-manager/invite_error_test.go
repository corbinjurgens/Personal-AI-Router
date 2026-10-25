// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"eapnoob"
)

func TestInviteFailureResultRetainsConnectionReason(t *testing.T) {
	// A private loopback listener drops the Initial reply. No real PAIR listener
	// or user data is involved in exercising the public handler and status record.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { dropPairingReply(w) }))
	t.Cleanup(server.Close)
	address := strings.TrimPrefix(server.URL, "http://")
	host, portText, err := net.SplitHostPort(address)
	if err != nil {
		t.Fatal(err)
	}
	var port int
	if _, err := fmt.Sscan(portText, &port); err != nil {
		t.Fatal(err)
	}
	m := newTestManager(t)
	var output bytes.Buffer
	m.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), &output})
	allowRemoteManagement := false
	params, err := json.Marshal(inviteNodeParams{Address: host, Port: &port, AllowRemoteManagement: &allowRemoteManagement})
	if err != nil {
		t.Fatal(err)
	}
	id := json.RawMessage("1")
	m.handleInviteNode(&Message{ID: &id, Params: params})
	decoder := json.NewDecoder(&output)
	var result struct {
		InviteID string `json:"inviteId"`
		State    string `json:"state"`
		Reason   string `json:"reason"`
		Error    string `json:"error"`
	}
	for decoder.More() {
		var message Message
		if err := decoder.Decode(&message); err != nil {
			t.Fatal(err)
		}
		if message.ID != nil && string(*message.ID) == "1" {
			if err := json.Unmarshal(message.Result, &result); err != nil {
				t.Fatal(err)
			}
			break
		}
	}
	if result.State != "failed" || result.Reason != "connection-reset" || result.Error == "" || result.InviteID == "" {
		t.Fatalf("failed invite result = %+v", result)
	}
	stored, ok := m.getInvite(result.InviteID)
	if !ok || stored.State != "failed" || stored.Reason != "connection-reset" {
		t.Fatalf("stored status = %+v, found=%v", stored, ok)
	}
}

func TestInviteFailureReason(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want string
	}{
		{"reset", syscall.ECONNRESET, "connection-reset"},
		{"dropped reply", io.EOF, "connection-reset"},
		{"truncated reply", io.ErrUnexpectedEOF, "connection-reset"},
		{"refused", syscall.ECONNREFUSED, "connection-refused"},
		{"host unreachable", syscall.EHOSTUNREACH, "connection-unreachable"},
		{"network unreachable", syscall.ENETUNREACH, "connection-unreachable"},
		{"timeout", context.DeadlineExceeded, "connection-timeout"},
		{"protocol", errors.New("peer says connection refused"), ""},
		{"wrong PIN", errors.New("incorrect PIN"), ""},
		{"rejected", &pairingRejectedError{reason: "already-clustered"}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := &url.Error{Op: "Post", URL: "http://127.0.0.1/pair", Err: &net.OpError{Op: "dial", Net: "tcp", Err: fmt.Errorf("wrapped: %w", tc.err)}}
			if got := inviteFailureReason(err); got != tc.want {
				t.Errorf("inviteFailureReason = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestPairingRedirectResponsesFailClosed(t *testing.T) {
	for _, tc := range []struct {
		name     string
		status   int
		location bool
	}{
		{"multiple choices", http.StatusMultipleChoices, true},
		{"moved permanently", http.StatusMovedPermanently, true},
		{"found", http.StatusFound, true},
		{"see other", http.StatusSeeOther, true},
		{"not modified", http.StatusNotModified, true},
		{"temporary redirect", http.StatusTemporaryRedirect, true},
		{"permanent redirect", http.StatusPermanentRedirect, true},
		{"redirect without location", http.StatusTemporaryRedirect, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var contacted atomic.Int32
			target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				contacted.Add(1)
				respondPairing(w, nil)
			}))
			t.Cleanup(target.Close)
			origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.location {
					w.Header().Set("Location", target.URL+pairingPath)
				}
				w.WriteHeader(tc.status)
			}))
			t.Cleanup(origin.Close)
			addr := strings.TrimPrefix(origin.URL, "http://")
			client := pairingHTTPClient()
			t.Cleanup(client.CloseIdleConnections)
			test := func(name string, post func() error) {
				t.Run(name, func(t *testing.T) {
					if err := post(); !errors.Is(err, errPairingRedirect) {
						t.Fatalf("pairing error = %v, want redirect refusal", err)
					}
					if got := contacted.Load(); got != 0 {
						t.Fatalf("redirect target contacts = %d, want 0", got)
					}
				})
			}
			test("initial", func() error {
				_, err := postPairingBlob(client, addr, "ordinary-invite", "initial", nil)
				return err
			})
			test("cancel", func() error {
				_, err := postPairingBlob(client, addr, "ordinary-invite", "cancel", nil)
				return err
			})
			test("legacy completion", func() error {
				_, err := postCompletionBlob(client, addr, "ordinary-invite", nil)
				return err
			})
			test("inviter completion", func() error {
				_, err := exchangePairingEnvelope(client, "http", addr, pairingEnvelope{InviteID: "ordinary-invite", Phase: "outbound-completion"})
				return err
			})
			test("terminal signal", func() error {
				return postPairingSignal(client, addr, "ordinary-invite", "fail", "")
			})
		})
	}
}

func TestOrdinaryInviteInitialNeverFollowsRedirect(t *testing.T) {
	test := func(name string, redirectOnFirst bool) {
		t.Run(name, func(t *testing.T) {
			m := joinTestManager(t)
			activateTestCluster(t, m, "redirect-test-cluster")
			joiner := joinTestManager(t)
			var contacted atomic.Int32
			target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				contacted.Add(1)
				respondPairing(w, nil)
			}))
			t.Cleanup(target.Close)
			var initialPosts atomic.Int32
			origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// A successful Initial Exchange has Discovery, Negotiation,
				// KeyExchange and a final EAP-Failure delivery.
				if !redirectOnFirst && initialPosts.Add(1) < 4 {
					joiner.handlePairing(w, r)
					return
				}
				http.Redirect(w, r, target.URL+pairingPath, http.StatusTemporaryRedirect)
			}))
			t.Cleanup(origin.Close)
			const id = "ordinary-initial-redirect"
			m.putInvite(&Invite{InviteID: id, ClusterID: "redirect-test-cluster", State: inviteStatePending, CreatedAt: time.Now().UnixMilli()})
			result := m.completeInitialInvite(id, []string{strings.TrimPrefix(origin.URL, "http://")}, "redirect-test-cluster", m.sessGen.Load(), false)
			if result["state"] != inviteStateFailed || result["reason"] != "redirect-refused" || result["pin"] != nil {
				t.Fatalf("Initial result = %+v, want failed/redirect-refused without a PIN", result)
			}
			inv, ok := m.getInvite(id)
			if !ok || inv.State != inviteStateFailed || inv.Reason != "redirect-refused" {
				t.Fatalf("stored Initial result = %+v, found=%v", inv, ok)
			}
			if _, ok := m.getSession(id); ok {
				t.Fatal("redirecting Initial retained a session")
			}
			if contacted.Load() != 0 || len(m.trust.List()) != 0 {
				t.Fatalf("redirect contacts = %d, trusted peers = %d, want 0", contacted.Load(), len(m.trust.List()))
			}
		})
	}
	test("first Initial message", true)
	test("final Initial message", false)
}

func TestOrdinaryInviteCancellationNeverFollowsRedirect(t *testing.T) {
	m := joinTestManager(t)
	const id = "ordinary-cancel-redirect"
	m.putInvite(&Invite{InviteID: id, State: inviteStateCanceled})
	var contacted, cancellations atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		contacted.Add(1)
		respondPairing(w, nil)
	}))
	t.Cleanup(target.Close)
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var env pairingEnvelope
		if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
			t.Error(err)
			return
		}
		if env.InviteID != id || env.Phase != "cancel" {
			t.Errorf("cancel envelope = %+v", env)
		}
		cancellations.Add(1)
		http.Redirect(w, r, target.URL+pairingPath, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(origin.Close)
	m.notifyJoinerCanceled(strings.TrimPrefix(origin.URL, "http://"), id)
	inv, ok := m.getInvite(id)
	if !ok || inv.State != inviteStateCanceled {
		t.Fatalf("best-effort cancellation changed local outcome: %+v, found=%v", inv, ok)
	}
	if cancellations.Load() != 1 || contacted.Load() != 0 {
		t.Fatalf("cancel attempts = %d, redirect contacts = %d, want 1/0", cancellations.Load(), contacted.Load())
	}
}

func TestOrdinaryInviteCompletionNeverFollowsRedirect(t *testing.T) {
	t.Run("legacy joiner completion", func(t *testing.T) {
		f := newOutboundPairingFixture(t)
		var contacted atomic.Int32
		target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			contacted.Add(1)
			respondPairing(w, nil)
		}))
		t.Cleanup(target.Close)
		origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, target.URL+pairingPath, http.StatusTemporaryRedirect)
		}))
		t.Cleanup(origin.Close)
		f.joinSession.addr = strings.TrimPrefix(origin.URL, "http://")
		f.joinSession.mu.Lock()
		err := f.joiner.runCompletionExchangeLocked(f.joinSession, f.pin)
		f.joinSession.mu.Unlock()
		var authErr *completionAuthError
		if !errors.Is(err, errPairingRedirect) || errors.As(err, &authErr) {
			t.Fatalf("Completion error = %v, want redirect refusal rather than incorrect PIN", err)
		}
		if f.joinSession.peer.State() == eapnoob.StateRegistered || len(f.joiner.trust.List()) != 0 || contacted.Load() != 0 {
			t.Fatalf("Completion state = %s, trusted peers = %d, redirect contacts = %d", f.joinSession.peer.State(), len(f.joiner.trust.List()), contacted.Load())
		}
	})
	t.Run("inviter completion", func(t *testing.T) {
		f := newOutboundPairingFixture(t)
		f.ready(t)
		var contacted atomic.Int32
		target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			contacted.Add(1)
			respondPairing(w, nil)
		}))
		t.Cleanup(target.Close)
		origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var env pairingEnvelope
			if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
				t.Error(err)
				return
			}
			if env.Phase == "outbound-ready" {
				f.joiner.handlePairingReady(w, &env)
				return
			}
			http.Redirect(w, r, target.URL+pairingPath, http.StatusTemporaryRedirect)
		}))
		t.Cleanup(origin.Close)
		f.session.joinerAddr = strings.TrimPrefix(origin.URL, "http://")
		f.inviter.driveOutboundCompletion(f.session)
		inv, ok := f.inviter.getInvite(f.session.inviteID)
		if !ok || inv.State != inviteStateFailed || inv.Reason != "redirect-refused" {
			t.Fatalf("Completion outcome = %+v, found=%v", inv, ok)
		}
		if contacted.Load() != 0 || len(f.inviter.trust.List()) != 0 || len(f.joiner.trust.List()) != 0 {
			t.Fatalf("redirect contacts = %d, inviter/joiner pins = %d/%d", contacted.Load(), len(f.inviter.trust.List()), len(f.joiner.trust.List()))
		}
	})
}
