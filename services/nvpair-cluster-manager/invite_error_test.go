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
	"syscall"
	"testing"
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
