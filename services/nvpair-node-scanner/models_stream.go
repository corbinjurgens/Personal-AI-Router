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
// Each ended attempt takes the failed-fetch path once, guarded like a fetch by
// the node's address, em port and route generation at the attempt's start: a
// peer whose stream is down loses download freshness as a failed fetch did,
// while a stream ended by a route change leaves that to the next attempt. A
// supervisor cancel skips it: the node is no longer a stream peer and the 15 s
// refresh owns it.
func (d *daemon) runModelsStream(ctx context.Context, hostUUID string, id uint64) {
	var guardIP string
	var emPort int
	var generation uint64
	var guarded bool
	runPeerStream(ctx, func(ctx context.Context) bool {
		generation = d.dir.routeGeneration(hostUUID)
		node, found := d.dir.get(hostUUID)
		em, advertised := node.Services[noderec.ServiceEngineManager]
		guardIP, emPort, guarded = node.IP, em.Port, found && advertised
		return d.streamPeerOnce(hostUUID, noderec.ServiceEngineManager, func(client *http.Client, current func() bool, node noderec.DirectoryNode, host string, port int) (bool, bool) {
			return d.readModelsStream(ctx, client, current, hostUUID, node.IP, host, port, id)
		})
	}, func() {
		d.dropModelsLive(hostUUID, id)
		if guarded && ctx.Err() == nil {
			d.markModelsUnavailable(hostUUID, guardIP, emPort, generation)
		}
	})
}

// modelsStreamLive reports whether hostUUID's models stream has applied a
// frame and is still open.
func (d *daemon) modelsStreamLive(hostUUID string) bool {
	s := &d.nodeInfoStreams
	s.mu.Lock()
	defer s.mu.Unlock()
	_, live := s.modelsLive[hostUUID]
	return live
}

// activeModelsStream reports whether id is hostUUID's supervised models stream.
func (d *daemon) activeModelsStream(hostUUID string, id uint64) bool {
	s := &d.nodeInfoStreams
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.models[hostUUID].id == id
}

// markModelsLive records that stream id applied a frame, while it is still
// hostUUID's supervised models stream.
func (d *daemon) markModelsLive(hostUUID string, id uint64) {
	s := &d.nodeInfoStreams
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.models[hostUUID].id != id {
		return
	}
	if s.modelsLive == nil {
		s.modelsLive = make(map[string]uint64)
	}
	s.modelsLive[hostUUID] = id
}

func (d *daemon) dropModelsLive(hostUUID string, id uint64) {
	s := &d.nodeInfoStreams
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.modelsLive[hostUUID] == id {
		delete(s.modelsLive, hostUUID)
	}
}

// readModelsStream connects to one address and reads its models stream. Each
// frame takes the successful-fetch path. guardIP is the node's canonical
// address, the stale-result guard.
func (d *daemon) readModelsStream(ctx context.Context, client *http.Client, current func() bool, hostUUID, guardIP, host string, port int, id uint64) (healthy, tryNext bool) {
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
			// A superseded stream's last frame must not overwrite its
			// replacement's.
			if !d.activeModelsStream(hostUUID, id) {
				return context.Canceled
			}
			d.applyModelsInventory(hostUUID, guardIP, port, d.dir.routeGeneration(hostUUID), inventory)
			// Live only once the frame is cached, so browse enrichment that
			// defers to the stream finds its inventory.
			d.markModelsLive(hostUUID, id)
			gotFrame = true
			return nil
		},
	})
	slog.Debug("models stream ended", "host_uuid", hostUUID, "ip", host, "err", err)
	return gotFrame && time.Since(opened) >= peerStreamHealthy, false
}
