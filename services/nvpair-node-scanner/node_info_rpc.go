// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"nvpair-shared/noderec"
)

const nodeInfoRPCBudget = 1500 * time.Millisecond

type nodeInfoRPCParams struct {
	HostUUID string `json:"hostUuid"`
}

func (d *daemon) handleNodeInfoRPC(msg *Message) {
	var p nodeInfoRPCParams
	decoder := json.NewDecoder(bytes.NewReader(msg.Params))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&p); err != nil || strings.TrimSpace(p.HostUUID) == "" {
		_ = d.codec.RespondError(msg.ID, -32602, "expected {hostUuid: string}")
		return
	}
	if err := decoder.Decode(new(json.RawMessage)); err != io.EOF {
		_ = d.codec.RespondError(msg.ID, -32602, "expected one params object")
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), nodeInfoRPCBudget)
	defer cancel()
	info, ok := d.getPeerNodeInfo(ctx, p.HostUUID)
	if !ok {
		_ = d.codec.RespondError(msg.ID, -32000, "node information unavailable")
		return
	}
	_ = d.codec.Respond(msg.ID, info)
}

// getPeerNodeInfo reads a fresh full sample from this scanner's registered self
// over loopback, or from directory and admitted peer authority. It never
// consults the enrichment cache or grants local authority to a directory claim.
func (d *daemon) getPeerNodeInfo(ctx context.Context, hostUUID string) (NodeInfoResponse, bool) {
	if d.reg != nil && hostUUID != "" && hostUUID == d.reg.record().HostUUID {
		return d.getSelfNodeInfo(ctx, hostUUID)
	}
	if d.dir == nil || d.mesh == nil {
		return NodeInfoResponse{}, false
	}
	node, found := d.dir.get(hostUUID)
	endpoint, registered := node.Services[noderec.ServiceNodeInfo]
	if !found || node.HostUUID != hostUUID || node.ClusterUUID == "" || !registered || endpoint.Port <= 0 || endpoint.Port > 65535 {
		return NodeInfoResponse{}, false
	}
	for _, host := range node.CandidateIPs() {
		client, current, admitted := d.peerServiceClient(hostUUID, node.ClusterUUID, host, endpoint.Port, noderec.ServiceNodeInfo, false)
		if !admitted {
			return NodeInfoResponse{}, false
		}
		info, ok := d.readNodeInfoWithin(ctx, client, "http", host, endpoint.Port, current)
		if ok && info.HostUUID == hostUUID && (info.ClusterUUID == nil || *info.ClusterUUID == node.ClusterUUID) {
			return info, true
		}
		if ctx.Err() != nil || !current() {
			return NodeInfoResponse{}, false
		}
	}
	return NodeInfoResponse{}, false
}

func (d *daemon) getSelfNodeInfo(ctx context.Context, hostUUID string) (NodeInfoResponse, bool) {
	authority, registered := d.reg.selfNodeInfo(hostUUID)
	if !registered || d.http == nil {
		return NodeInfoResponse{}, false
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stop := context.AfterFunc(authority.incarnation, cancel)
	defer stop()
	current := func() bool { return ctx.Err() == nil && authority.incarnation.Err() == nil }
	// Use the same loopback HTTP transport as self enrichment. Neither a LAN
	// candidate, operator TLS, nor a redirect may supply this scanner's sample.
	client := *d.http
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	info, ok := d.readNodeInfoWithin(ctx, &client, "http", loopbackHost, authority.port, current)
	if !ok || info.HostUUID != authority.hostUUID || info.ClusterUUID == nil || *info.ClusterUUID != authority.clusterUUID || !current() {
		return NodeInfoResponse{}, false
	}
	return info, true
}
