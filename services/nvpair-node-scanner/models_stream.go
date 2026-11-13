// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

// Pinned peers serve their model inventory as a snapshot stream
// (engine-manager's GET /v1/models/events): the full /v1/models document on
// connect and whenever it changes, and a null keepalive every 10 s. The scanner
// holds one stream per stream peer that advertises engine-manager instead of
// fetching /v1/models every 15 s; self and other nodes keep the fetch.

// modelsStreamIdle drops a models stream after 25 s without a line (two missed
// keepalives and some slack). A var so tests can shorten it.
var modelsStreamIdle = 25 * time.Second

const modelsStreamPath = "/v1/models/events"

// errInvalidModelsFrame ends a models stream whose frame fails to decode.
var errInvalidModelsFrame = errors.New("models stream frame is not a valid inventory document")

// runModelsStream keeps one models stream to hostUUID open until ctx ends.
// Each ended attempt takes the failed-fetch path once, so a peer whose stream
// is down loses download freshness as a failed fetch did. A supervisor cancel
// skips it: the node is no longer a stream peer and the 15 s refresh owns it.
func (d *daemon) runModelsStream(ctx context.Context, hostUUID string, _ uint64) {
	runPeerStream(ctx, func(ctx context.Context) bool {
		return d.streamPeerOnce(hostUUID, noderec.ServiceEngineManager, func(client *http.Client, current func() bool, node noderec.DirectoryNode, host string, port int) (bool, bool) {
			return d.readModelsStream(ctx, client, current, hostUUID, node.IP, host, port)
		})
	}, func() {
		if ctx.Err() != nil {
			return
		}
		generation := d.dir.routeGeneration(hostUUID)
		node, found := d.dir.get(hostUUID)
		if em, ok := node.Services[noderec.ServiceEngineManager]; found && ok {
			d.markModelsUnavailable(hostUUID, node.IP, em.Port, generation)
		}
	})
}

// readModelsStream connects to one address and reads its models stream. Each
// frame takes the successful-fetch path. guardIP is the node's canonical
// address, the stale-result guard.
func (d *daemon) readModelsStream(ctx context.Context, client *http.Client, current func() bool, hostUUID, guardIP, host string, port int) (healthy, tryNext bool) {
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	url := "https://" + net.JoinHostPort(host, strconv.Itoa(port)) + modelsStreamPath
	resp, tryNext := connectPeerStream(ctx, streamCtx, cancel, client, url, "models", hostUUID, host)
	if resp == nil {
		return false, tryNext
	}
	defer resp.Body.Close()
	opened := time.Now()
	gotFrame := false
	err := clustertrust.ReadSnapshotStream(streamCtx, cancel, resp.Body, clustertrust.SnapshotStreamClient[modelInventory]{
		Idle:    modelsStreamIdle,
		MaxLine: modelsResponseBytes,
		Current: func() bool {
			_, stillPeer := d.streamPeerNode(hostUUID)
			return stillPeer && current()
		},
		Decode: func(line []byte) (modelInventory, error) {
			inventory, ok := decodeModelsInventory(line, host, d.downloadSourceBaseline(hostUUID))
			if !ok {
				return modelInventory{}, errInvalidModelsFrame
			}
			return inventory, nil
		},
		OnFrame: func(inventory modelInventory) error {
			d.applyModelsInventory(hostUUID, guardIP, port, d.dir.routeGeneration(hostUUID), inventory)
			gotFrame = true
			return nil
		},
	})
	slog.Debug("models stream ended", "host_uuid", hostUUID, "ip", host, "err", err)
	return gotFrame && time.Since(opened) >= peerStreamHealthy, false
}
