// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

type nodeNameIdentity struct {
	NodeUUID   string  `json:"nodeUuid"`
	Name       string  `json:"name"`
	CustomName *string `json:"customName"`
}

type nodeNameMember struct {
	NodeUUID    string `json:"nodeUuid"`
	Name        string `json:"name"`
	DisplayName string `json:"displayName"`
}

func nodeNameMembers(t *testing.T, p *cmProc) map[string]nodeNameMember {
	t.Helper()
	res := decodeResult[struct {
		Nodes []nodeNameMember `json:"nodes"`
	}](t, p.call("nodes:get-initial", nil))
	out := make(map[string]nodeNameMember, len(res.Nodes))
	for _, n := range res.Nodes {
		out[n.NodeUUID] = n
	}
	return out
}

// nodeNameSetGrant records target's local remote-management decision for peer
// through cluster:set-remote-management.
func nodeNameSetGrant(t *testing.T, target *cmProc, peer string, allowed bool) {
	t.Helper()
	before := decodeResult[managementSnapshot](t, target.call("cluster:get-remote-management", map[string]any{}))
	after := decodeResult[managementSnapshot](t, target.call("cluster:set-remote-management", map[string]any{
		"peerUuid": peer, "allowed": allowed, "expectedRevision": before.Revision,
	}))
	for _, grant := range after.Grants {
		if grant.PeerUUID == peer && grant.Allowed == allowed {
			return
		}
	}
	t.Fatalf("grant for %s did not become allowed=%t", peer, allowed)
}

func nodeNameAssertDenied(t *testing.T, viewer *cmProc, target, name string) {
	t.Helper()
	resp := viewer.callExpectError("cluster:set-peer-name", map[string]any{"nodeUuid": target, "name": name})
	if resp.Error == nil || resp.Error.Code != -32012 ||
		resp.Error.Message != "Remote management is blocked on this device. Change permission locally on that device." {
		t.Fatalf("set-peer-name response = %+v, want -32012 with the target's denial", resp.Error)
	}
}

// TestNodeNameRemoteRenameBoundary drives the peer rename and nickname across
// real cluster-manager processes. A pairs B and C; B learns C through
// endorsement, so B's grant for C starts unreviewed. C is the viewer, B the
// target.
func TestNodeNameRemoteRenameBoundary(t *testing.T) {
	aDir, bDir, cDir := shortClusterTempDir(t), shortClusterTempDir(t), shortClusterTempDir(t)
	aPort, bPort, cPort := temporaryClusterPort(t), temporaryClusterPort(t), temporaryClusterPort(t)
	a, b, c := startCM(t, aDir, aPort), startCM(t, bDir, bPort), startCM(t, cDir, cPort)
	t.Cleanup(a.stop)
	t.Cleanup(b.stop)
	t.Cleanup(c.stop)
	a.call("cluster:get-node-id", nil)
	bID := decodeResult[nodeNameIdentity](t, b.call("cluster:get-node-id", nil))
	cID := decodeResult[nodeNameIdentity](t, c.call("cluster:get-node-id", nil))
	for _, port := range []int{aPort, bPort, cPort} {
		waitTCP(t, "127.0.0.1:"+strconv.Itoa(port), 10*time.Second)
	}
	a.call("cluster:create", map[string]any{"clusterFriendlyName": "node name fixture"})
	pairNodes(t, a, b, bPort)
	pairNodes(t, a, c, cPort)
	waitMembers(t, b, 3)
	waitMembers(t, c, 3)
	mesh := clustertrust.Open(filepath.Join(bDir, "cluster"))
	task1Eventually(t, "B holds C's endorsed admission, unreviewed", 10*time.Second, func() bool {
		mesh.Refresh()
		view := mesh.RemoteManagement(cID.NodeUUID)
		return mesh.HasPin(cID.NodeUUID) && view.State == "unreviewed"
	})

	t.Run("unreviewed grant refuses", func(t *testing.T) {
		nodeNameAssertDenied(t, c, bID.NodeUUID, "Renamed by C")
		if got := decodeResult[nodeNameIdentity](t, b.call("cluster:get-node-id", nil)); got.Name != bID.Name || got.CustomName != nil {
			t.Fatalf("target identity = %+v, want unchanged %q", got, bID.Name)
		}
	})

	t.Run("allowed grant renames the target", func(t *testing.T) {
		nodeNameSetGrant(t, b, cID.NodeUUID, true)
		result := decodeResult[nodeNameIdentity](t, c.call("cluster:set-peer-name", map[string]any{
			"nodeUuid": bID.NodeUUID, "name": "Renamed by C",
		}))
		if result.NodeUUID != bID.NodeUUID || result.Name != "Renamed by C" {
			t.Fatalf("set-peer-name result = %+v", result)
		}
		got := decodeResult[nodeNameIdentity](t, b.call("cluster:get-node-id", nil))
		if got.Name != "Renamed by C" || got.CustomName == nil || *got.CustomName != "Renamed by C" {
			t.Fatalf("target identity = %+v, want name and customName Renamed by C", got)
		}
		if member := nodeNameMembers(t, c)[bID.NodeUUID]; member.Name != "Renamed by C" {
			t.Fatalf("viewer member = %+v, want name Renamed by C", member)
		}
	})

	t.Run("revoked grant refuses", func(t *testing.T) {
		nodeNameSetGrant(t, b, cID.NodeUUID, false)
		nodeNameAssertDenied(t, c, bID.NodeUUID, "Second rename")
		if got := decodeResult[nodeNameIdentity](t, b.call("cluster:get-node-id", nil)); got.Name != "Renamed by C" {
			t.Fatalf("target name = %q after a refused rename", got.Name)
		}
	})

	t.Run("nickname stays on the viewer", func(t *testing.T) {
		result := decodeResult[struct {
			NodeUUID    string  `json:"nodeUuid"`
			Nickname    *string `json:"nickname"`
			DisplayName string  `json:"displayName"`
		}](t, c.call("cluster:set-peer-nickname", map[string]any{"nodeUuid": bID.NodeUUID, "nickname": "C's name for B"}))
		if result.Nickname == nil || *result.Nickname != "C's name for B" || result.DisplayName != "C's name for B" {
			t.Fatalf("set-peer-nickname result = %+v", result)
		}
		if member := nodeNameMembers(t, c)[bID.NodeUUID]; member.DisplayName != "C's name for B" || member.Name != "Renamed by C" {
			t.Fatalf("viewer member = %+v, want displayName from the nickname and own name kept", member)
		}
		if got := decodeResult[nodeNameIdentity](t, b.call("cluster:get-node-id", nil)); got.Name != "Renamed by C" {
			t.Fatalf("target name = %q; a nickname must not reach the target", got.Name)
		}
		if member := nodeNameMembers(t, a)[bID.NodeUUID]; member.DisplayName == "C's name for B" {
			t.Fatal("C's nickname reached another node")
		}
	})
}
