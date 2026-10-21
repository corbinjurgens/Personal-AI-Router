// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"

	"nvpair-shared/appdir"
	"nvpair-shared/applog"
	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

// initializeProxy validates the effective CLI body limit before startup gives
// the proxy account paths, cluster trust, listeners, or dispatch work.
func initializeProxy(codec *Codec, maxRequestBytes int64) (*Proxy, error) {
	proxy := NewProxy(codec)
	proxy.maxRequestBytes = maxRequestBytes
	if _, _, err := proxy.bufferSettings(nodepolicy.Default()); err != nil {
		return nil, fmt.Errorf("invalid --max-request-bytes: %w", err)
	}
	return proxy, nil
}

func main() {
	ipcPath := flag.String("ipc", "", "IPC endpoint: Unix domain socket path or Windows named pipe (default: stdin/stdout)")
	clusterDir := flag.String("cluster-dir", "", "cluster trust directory (node.crt/key + trusted pins); enables the LAN mTLS inference ingress when this node is clustered")
	showVersion := flag.Bool("version", false, "print version and exit")
	maxRequestBytes := flag.Int64("max-request-bytes", defaultMaxRequestBytes, "largest request body the proxy buffers for failover; larger requests get 413")
	resolveLevel := applog.RegisterFlag(nil, slog.LevelInfo)
	flag.Parse()

	if *showVersion {
		fmt.Println(Version)
		os.Exit(0)
	}

	// The process starts with no engine facade and no listener. Which engines
	// it fronts, on which ports, and with which alias addresses all arrive over
	// facade/enable, because the broker computes a different port plan per
	// engine and a flag cannot express more than one of them.
	//
	// The log component is therefore the process, not an engine: there is no
	// profile to name here. Facade-scoped records carry an engine field so a
	// line is still attributable once several facades share this sink.
	applog.Init(engines.ProxyComponent, resolveLevel())

	var transport io.ReadWriteCloser
	if *ipcPath != "" {
		conn, err := dialIPC(*ipcPath)
		if err != nil {
			log.Fatalf("failed to connect to IPC endpoint %q: %v", *ipcPath, err)
		}
		transport = conn
		log.Printf("using IPC transport: %s", *ipcPath)
	} else {
		conn, err := newStdioTransport()
		if err != nil {
			log.Fatalf("failed to prepare stdio transport: %v", err)
		}
		transport = conn
		log.Print("using stdio transport")
	}
	defer transport.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	failSession, joinSession := newTerminalSession(ctx, cancel, transport)
	defer joinSession()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		select {
		case sig := <-sigCh:
			log.Printf("received %s, shutting down", sig)
			cancel()
		case <-ctx.Done():
		}
	}()

	codec := newOwnedCodec(ctx, transport, failSession, notificationWriteBudget)
	proxy, err := initializeProxy(codec, *maxRequestBytes)
	if err != nil {
		log.Fatal(err)
	}
	// The account store opens on first use, when an activated policy names a
	// provider account; a node without one creates no account files.
	proxy.accounts.setDir(appdir.Dir())
	// Open a live view of this node's cluster mTLS trust fabric. While unclustered
	// the proxy serves only the loopback plaintext personality; once this node is
	// a member the same listener also serves the pin-gated LAN mTLS ingress, and
	// peers become routable candidates. The proxy needs no restart to notice
	// either transition — it re-derives membership per request and on a watch.
	//
	// Membership is gated on an active admission or a pin, never on keypair
	// presence: a left/removed node keeps its keypair by design, and would
	// otherwise keep logging cluster_ingress with no cluster peers to serve.
	proxy.mesh = clustertrust.Open(*clusterDir)
	if *clusterDir != "" {
		proxy.baseConfigDir = filepath.Dir(*clusterDir)
	}

	go proxy.mesh.Watch(ctx, func(clustered bool) {
		slog.Info("cluster inference ingress switched personality", "cluster_ingress", clustered)
		proxy.dropUnpinnedPeerTransports()
	})

	if err := proxy.Run(ctx); err != nil && ctx.Err() == nil {
		log.Fatalf("proxy error: %v", err)
	}
	log.Print("shutdown complete")
}

// The session owns the parent transport independently of Proxy's locks.
// Closing both directions wakes readLoop; Run remains the shutdown owner.
func newTerminalSession(ctx context.Context, cancel context.CancelFunc, transport io.ReadWriteCloser) (func(string, error), func()) {
	var closeOnce, failOnce sync.Once
	closeTransport := func() {
		closeOnce.Do(func() {
			if err := transport.Close(); err != nil {
				slog.Warn("parent transport close failed", "err", err)
			}
		})
	}
	fail := func(method string, err error) {
		failOnce.Do(func() {
			slog.Error("terminal parent transport failure", "method", method, "err", err)
			cancel()
			closeTransport()
		})
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		<-ctx.Done()
		closeTransport()
	}()
	return fail, func() { cancel(); <-done }
}
