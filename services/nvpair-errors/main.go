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
	"syscall"

	"nvpair-shared/applog"
	"nvpair-shared/clustertrust"
	"nvpair-shared/nodeid"
)

func main() {
	ipcPath := flag.String("ipc", "", "IPC endpoint: Unix domain socket path or Windows named pipe (default: stdin/stdout)")
	peerSync := flag.Bool("peer-sync", false, "serve the cross-node errors endpoint and push errors to peer nodes learned from the broker's discovery relay")
	port := flag.Int("port", 14319, "HTTP port for the cross-node errors endpoint (only used with --peer-sync)")
	clusterDir := flag.String("cluster-dir", "", "cluster config dir (node.crt/node.key + trusted/); cross-node error sync is pin-based cluster mTLS, so without a usable identity and live membership here this node serves and pushes nothing")
	nodeID := flag.String("node-id", "", "this node's stable origin id (the broker passes its resolved per-host UUID); when unset, peer-sync resolves the shared stable UUID and local-only mode defaults to the hostname")
	showVersion := flag.Bool("version", false, "print version and exit")
	resolveLevel := applog.RegisterFlag(nil, slog.LevelInfo)
	flag.Parse()

	if *showVersion {
		fmt.Println(Version)
		os.Exit(0)
	}

	applog.Init("nvpair-errors", resolveLevel())

	var transport io.ReadWriteCloser
	if *ipcPath != "" {
		conn, err := dialIPC(*ipcPath)
		if err != nil {
			log.Fatalf("failed to connect to IPC endpoint %q: %v", *ipcPath, err)
		}
		transport = conn
		log.Printf("using IPC transport: %s", *ipcPath)
	} else {
		transport = newStdioTransport()
		log.Print("using stdio transport")
	}
	defer transport.Close()

	codec := NewCodec(transport)
	mgr := newStartupManager(codec, *peerSync, *nodeID, *clusterDir)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

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

	// Networking is opt-in. Without --peer-sync, nvpair-errors is a
	// stdio-only local datastore with no network surface (and the
	// integration tests that spawn the bare binary keep working). With
	// it, the same Manager additionally learns peer nvpair-errors
	// instances from the broker's discovery relay, serves an HTTPS
	// ingest endpoint, and pushes its local-origin errors to peers.
	if *peerSync {
		// Gate cross-node mTLS on live cluster membership, not keypair presence: a
		// left/removed node keeps its keypair by design. The mesh is a live view of
		// the cluster dir, so this node follows a create/join/leave that happens
		// while it runs instead of latching whatever it read at startup.
		startPeerSync(ctx, mgr, *port, clustertrust.Open(*clusterDir), *clusterDir)
	}

	if err := mgr.Run(ctx); err != nil && ctx.Err() == nil {
		log.Fatalf("manager error: %v", err)
	}
	log.Print("shutdown complete")
}

func newStartupManager(codec *Codec, peerSync bool, nodeID, clusterDir string) *Manager {
	mgr := NewManager(codec)
	if peerSync && nodeID == "" {
		// Match the broker's config root: clusterDir is <base>/cluster;
		// empty selects nodeid's shared default. Resolve before NewPeerSync
		// captures the origin, even if cluster creation will happen later.
		base := ""
		if clusterDir != "" {
			base = filepath.Dir(clusterDir)
		}
		nodeID = nodeid.Resolve(base)
		if nodeID == "" {
			log.Fatal("could not resolve a stable node UUID (system CSPRNG unavailable); refusing to start without a node identity")
		}
	}
	mgr.SetLocalNodeID(nodeID)
	return mgr
}

// startPeerSync wires the cross-node networking layer onto mgr and
// launches its goroutines under ctx. They unwind when ctx is cancelled
// (stdin EOF, signal, or the shutdown RPC) — the same lifecycle as the
// stdio manager, so there's no separate teardown to coordinate.
//
// A failure to bind the HTTP port is fatal: without an inbound endpoint
// peers can't reach this node, so silently degrading to "advertises but
// never receives" would be worse than a clear crash the supervisor logs.
func startPeerSync(ctx context.Context, mgr *Manager, port int, mesh *clustertrust.Mesh, clusterDir string) {
	peers := NewPeerSync(mgr, mesh, clusterDir)
	mgr.SetOnLocalChange(peers.TriggerPush)

	// Peers come solely from the broker's discovery relay. The Manager reconciles
	// the relay's discovery:nodes snapshot (the full filtered er set) into peer
	// events on this channel, and subscribes upward for er nodes. This service
	// runs no mDNS of its own: the node-scanner daemon advertises this node on its
	// single _nvpair-node record (er=<port>), which the broker registers for us.
	events := make(chan DiscoveryEvent, 32)
	mgr.SetRelayEvents(events)

	go func() {
		if err := runHTTPServer(ctx, port, mgr, mesh); err != nil {
			log.Fatalf("errors HTTP server failed: %v", err)
		}
	}()

	go peers.Run(ctx, events)
	// Follow this node into and out of a cluster. Both the ingress gate and the
	// push transport read live membership, so the watch only has to notice the
	// change and re-push: the peer set was filtered under the old personality, so
	// a fresh full snapshot is what repairs a divergence the outage created.
	go mesh.Watch(ctx, func(clustered bool) {
		slog.Info("error sync switched personality", "clustered", clustered, "clusterUuid", mesh.NodeUUID())
		peers.TriggerPush()
	})

	if mesh.Clustered() {
		log.Printf("peer-sync enabled (cluster mTLS): serving %q (uuid %s) on port %d", mgr.LocalNodeID(), mesh.NodeUUID(), port)
	} else {
		log.Printf("peer-sync enabled: serving %q on port %d", mgr.LocalNodeID(), port)
	}
}
