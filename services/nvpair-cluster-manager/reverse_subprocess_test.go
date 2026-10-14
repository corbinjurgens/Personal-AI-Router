// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

func TestReverseSubprocessPrivateDialMetadataReplayAndRebind(t *testing.T) {
	buildCtx, buildCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer buildCancel()
	binary := filepath.Join(t.TempDir(), "cm")
	build := exec.CommandContext(buildCtx, "go", "build", "-o", binary, ".")
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build isolated CM: %v\n%s", err, output)
	}
	a, b := newReverseTestManager(t), newReverseTestManager(t)
	startReverseHTTP(t, a, false)
	free, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	b.port = free.Addr().(*net.TCPAddr).Port
	if err = free.Close(); err != nil {
		t.Fatal(err)
	}
	pairReverseManagers(t, a, b)
	if err = b.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	makeService := func(body string) (int, func()) {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		server := http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/cluster/reverse/dial" {
				http.NotFound(w, r)
				return
			}
			io.WriteString(w, body)
		})}
		go server.Serve(ln)
		close := func() { server.Close() }
		t.Cleanup(close)
		return ln.Addr().(*net.TCPAddr).Port, close
	}
	aPort, _ := makeService("a-service")
	setReverseServices(t, a, 0, noderec.RegisterParams{Service: "ni", Port: aPort})
	bPort, _ := makeService("b-generation-1")
	bNewPort, _ := makeService("b-generation-2")
	base := filepath.Dir(b.clusterDir)
	type child struct {
		cmd   *exec.Cmd
		stdin io.WriteCloser
		codec *Codec
		stop  func()
	}
	start := func() child {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		cmd := exec.CommandContext(ctx, binary, "--config-dir", base, "--port", fmt.Sprint(b.port), "--log-level", "error")
		cmd.Env = append(os.Environ(), "NVPAIR_TEST_PAIR_REJECT_INBOUND=1")
		stdin, err := cmd.StdinPipe()
		if err != nil {
			cancel()
			t.Fatal(err)
		}
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			cancel()
			t.Fatal(err)
		}
		cmd.Stderr = io.Discard
		if err = cmd.Start(); err != nil {
			cancel()
			t.Fatal(err)
		}
		codec := NewCodec(struct {
			io.Reader
			io.Writer
		}{stdout, stdin})
		var once sync.Once
		stop := func() {
			once.Do(func() {
				stdin.Close()
				err := cmd.Wait()
				cancel()
				if err != nil {
					t.Errorf("isolated child exit: %v", err)
				}
			})
		}
		t.Cleanup(stop)
		return child{cmd, stdin, codec, stop}
	}
	request := func(c child, id string, method string, params any) {
		wire, err := json.Marshal(params)
		if err != nil {
			t.Fatal(err)
		}
		raw := json.RawMessage(id)
		if err = json.NewEncoder(c.stdin).Encode(&Message{JSONRPC: "2.0", ID: &raw, Method: method, Params: wire}); err != nil {
			t.Fatal(err)
		}
		for {
			reply, err := c.codec.Read()
			if err != nil {
				t.Fatalf("child response: %v", err)
			}
			if reply.ID == nil {
				continue
			}
			if string(*reply.ID) != id {
				t.Fatalf("unexpected response id %s", *reply.ID)
			}
			if reply.Error != nil {
				t.Fatalf("child RPC failed: %+v", reply.Error)
			}
			return
		}
	}
	waitSession := func() {
		peer, ok := a.reversePeer(b.identity.NodeUUID)
		if !ok {
			t.Fatal("seeded child not admitted")
		}
		deadline := time.Now().Add(5 * time.Second)
		for !a.reverse.manager.Has(peer) {
			if time.Now().After(deadline) {
				t.Fatal("child did not initiate blocked-direction reverse session")
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	readService := func(conn net.Conn, want string) {
		defer conn.Close()
		if _, err := io.WriteString(conn, "GET /data HTTP/1.1\r\nHost: peer\r\nConnection: close\r\n\r\n"); err != nil {
			t.Fatal(err)
		}
		response, err := http.ReadResponse(bufio.NewReader(conn), &http.Request{Method: http.MethodGet})
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		body, err := io.ReadAll(response.Body)
		if err != nil {
			t.Fatal(err)
		}
		if response.StatusCode != http.StatusOK || string(body) != want {
			t.Fatalf("service response status %d body %q, want %q", response.StatusCode, body, want)
		}
	}
	openB := func() net.Conn {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		t.Cleanup(cancel)
		peer, _ := a.reversePeer(b.identity.NodeUUID)
		conn, err := a.reverse.manager.Open(ctx, peer, "ni")
		if err != nil {
			t.Fatal(err)
		}
		return conn
	}
	first := start()
	request(first, "1", "cluster:get-node-id", struct{}{})
	waitSession()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	private, err := clustertrust.DialReverse(ctx, base, a.identity.NodeUUID, "ni")
	if err != nil {
		t.Fatal(err)
	}
	readService(private, "a-service")
	request(first, "2", "cluster:set-local-services", localServicesSnapshot{1, []noderec.RegisterParams{{Service: "ni", Port: bPort}}})
	readService(openB(), "b-generation-1")
	request(first, "3", "cluster:set-local-services", localServicesSnapshot{2, []noderec.RegisterParams{{Service: "ni", Port: bNewPort}}})
	readService(openB(), "b-generation-2")
	first.stop()
	peer, _ := a.reversePeer(b.identity.NodeUUID)
	deadline := time.Now().Add(2 * time.Second)
	for a.reverse.manager.Has(peer) {
		if time.Now().After(deadline) {
			t.Fatal("child EOF retained session")
		}
		time.Sleep(time.Millisecond)
	}
	second := start()
	request(second, "1", "cluster:get-node-id", struct{}{})
	waitSession()
	ctxEmpty, cancelEmpty := context.WithTimeout(context.Background(), time.Second)
	defer cancelEmpty()
	if conn, err := a.reverse.manager.Open(ctxEmpty, peer, "ni"); err == nil {
		conn.Close()
		t.Fatal("restart retained service metadata before replay")
	}
	request(second, "2", "cluster:set-local-services", localServicesSnapshot{2, []noderec.RegisterParams{{Service: "ni", Port: bNewPort}}})
	readService(openB(), "b-generation-2")
	second.stop()
}
