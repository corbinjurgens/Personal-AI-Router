// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/errors"
)

// shortenErrorsStreamKeepalive sets the keepalive for one test.
func shortenErrorsStreamKeepalive(t *testing.T, d time.Duration) {
	t.Helper()
	previous := errorsStreamKeepalive
	errorsStreamKeepalive = d
	t.Cleanup(func() { errorsStreamKeepalive = previous })
}

// openErrorsStream subscribes with client and returns the response and its
// lines. The channel closes when the stream ends.
func openErrorsStream(t *testing.T, client *http.Client, url string) (*http.Response, <-chan []byte) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url+"/v1/errors/events", nil)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	resp, err := client.Do(req)
	if err != nil {
		cancel()
		t.Fatalf("subscribe: %v", err)
	}
	t.Cleanup(func() {
		cancel()
		resp.Body.Close()
	})
	lines := make(chan []byte, 16)
	go func() {
		defer close(lines)
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			lines <- slices.Clone(scanner.Bytes())
		}
	}()
	return resp, lines
}

// nextErrorsFrame returns the next non-keepalive line as an envelope.
func nextErrorsFrame(t *testing.T, lines <-chan []byte) errors.SyncEnvelope {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		select {
		case line, open := <-lines:
			if !open {
				t.Fatal("stream ended before a frame")
			}
			if string(line) == "null" {
				continue
			}
			var env errors.SyncEnvelope
			if err := json.Unmarshal(line, &env); err != nil {
				t.Fatalf("decode frame %q: %v", line, err)
			}
			return env
		case <-deadline:
			t.Fatal("timed out waiting for a frame")
		}
	}
}

func TestErrorsStreamSendsSnapshotThenChanges(t *testing.T) {
	shortenErrorsStreamKeepalive(t, 50*time.Millisecond)
	m := managerForNode("node-a")
	m.upsert(localErr("a:one", "node-a", 1000))
	m.reconcilePeer("node-b", []ServiceError{localErr("b:one", "node-b", 1000)})
	srv, client := servePinnedErrorsMux(t, m)

	resp, lines := openErrorsStream(t, client, srv.URL)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := resp.Header.Get("Content-Type"); got != "application/x-ndjson" {
		t.Errorf("Content-Type = %q, want application/x-ndjson", got)
	}
	if got := resp.Header.Get("Cache-Control"); got != "no-store" {
		t.Errorf("Cache-Control = %q, want no-store", got)
	}

	t.Run("first frame is the current local-origin snapshot", func(t *testing.T) {
		env := nextErrorsFrame(t, lines)
		if env.NodeID != "node-a" || !slices.Equal(env.Errors, []ServiceError{localErr("a:one", "node-a", 1000)}) {
			t.Fatalf("first frame = %+v, want node-a with only a:one", env)
		}
	})
	t.Run("keepalives arrive while nothing changes", func(t *testing.T) {
		keepalives := 0
		deadline := time.After(2 * time.Second)
		for keepalives < 3 {
			select {
			case line, open := <-lines:
				if !open {
					t.Fatal("stream ended")
				}
				if string(line) != "null" {
					t.Fatalf("unchanged snapshot sent a frame: %s", line)
				}
				keepalives++
			case <-deadline:
				t.Fatalf("got %d keepalives in 2 s at a 50 ms interval", keepalives)
			}
		}
	})
	t.Run("a local report sends a new frame", func(t *testing.T) {
		m.handleReport(nil, mustJSON(t, localErr("a:two", "node-a", 2000)))
		env := nextErrorsFrame(t, lines)
		want := []ServiceError{localErr("a:one", "node-a", 1000), localErr("a:two", "node-a", 2000)}
		if env.NodeID != "node-a" || !slices.Equal(env.Errors, want) {
			t.Fatalf("frame after report = %+v, want %+v", env, want)
		}
	})
}

func TestErrorsStreamRequiresPinnedCallerIdentity(t *testing.T) {
	selfCert, selfKey := genLeaf(t, "uuid-self")
	peerCert, peerKey := genLeaf(t, "uuid-peer")
	impostorCert, impostorKey := genLeaf(t, "uuid-peer")
	otherCert, otherKey := genLeaf(t, "uuid-other")
	selfDir := setupNode(t, selfCert, selfKey, map[string][]byte{"uuid-peer": peerCert})
	selfMesh := clustertrust.Open(selfDir)
	m := managerForNode("uuid-self")
	m.upsert(localErr("a:one", "uuid-self", 1000))
	pub := newErrorsPublisher()
	pub.publish(m)
	srv := httptest.NewUnstartedServer(newErrorsMux(m, selfMesh, pub))
	srv.TLS = selfMesh.ServerTLSConfig()
	srv.StartTLS()
	t.Cleanup(srv.Close)

	// Every caller pins self, so each completes the handshake; only the pin
	// gate on self can tell them apart.
	callerMesh := func(cert, key []byte) *clustertrust.Mesh {
		return clustertrust.Open(setupNode(t, cert, key, map[string][]byte{"uuid-self": selfCert}))
	}
	for _, tc := range []struct {
		name string
		mesh *clustertrust.Mesh
		want int
	}{
		{"unpinned caller", callerMesh(otherCert, otherKey), http.StatusForbidden},
		{"caller claiming a pinned uuid with another certificate", callerMesh(impostorCert, impostorKey), http.StatusForbidden},
		{"pinned caller", callerMesh(peerCert, peerKey), http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp, lines := openErrorsStream(t, meshClient(t, tc.mesh, "uuid-self"), srv.URL)
			if resp.StatusCode != tc.want {
				t.Fatalf("status = %d, want %d", resp.StatusCode, tc.want)
			}
			if tc.want == http.StatusOK {
				if env := nextErrorsFrame(t, lines); env.NodeID != "uuid-self" {
					t.Fatalf("frame nodeId = %q, want uuid-self", env.NodeID)
				}
			}
		})
	}

	t.Run("removing the caller's pin ends an open stream", func(t *testing.T) {
		shortenErrorsStreamKeepalive(t, 50*time.Millisecond)
		resp, lines := openErrorsStream(t, meshClient(t, callerMesh(peerCert, peerKey), "uuid-self"), srv.URL)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
		nextErrorsFrame(t, lines)
		if err := os.Remove(filepath.Join(selfDir, "trusted", "uuid-peer.json")); err != nil {
			t.Fatal(err)
		}
		deadline := time.After(2 * time.Second)
		for {
			select {
			case _, open := <-lines:
				if !open {
					return
				}
			case <-deadline:
				t.Fatal("stream outlived the removed pin")
			}
		}
	})
}

func TestErrorsStreamRejectsPost(t *testing.T) {
	srv, client := servePinnedErrorsMux(t, managerForNode("node-a"))
	resp, err := client.Post(srv.URL+"/v1/errors/events", "application/json", nil)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed || resp.Header.Get("Allow") != "GET" {
		t.Fatalf("status = %d Allow = %q, want 405 GET", resp.StatusCode, resp.Header.Get("Allow"))
	}
}

func TestErrorsPublisherClonesSnapshots(t *testing.T) {
	m := managerForNode("node-a")
	m.upsert(localErr("a:one", "node-a", 1000))
	pub := newErrorsPublisher()
	pub.publish(m)
	first, ok := pub.hub.Latest()
	if !ok {
		t.Fatal("no published snapshot")
	}
	first.Errors[0].Message = "mutated by a reader"
	second, _ := pub.hub.Latest()
	if second.Errors[0] != localErr("a:one", "node-a", 1000) {
		t.Fatalf("a reader's copy changed the published snapshot: %+v", second.Errors[0])
	}
}
