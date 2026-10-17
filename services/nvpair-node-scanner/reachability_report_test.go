// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	sharederrors "nvpair-shared/errors"
	"nvpair-shared/noderec"
)

// These cover how a paired peer that is down reaches the errors view when the
// peer is known through a reverse session rather than through its mDNS record:
// its claims stop arriving, and the claim's age is what says so.

const peerUnreachableID = "node-scanner:node-info-unreachable:peer"

// peerServicesWithCapturedCodec is newPeerServicesScanner with the codec output
// captured. Only the errors frames are returned, so the directory events the
// same codec carries do not have to be accounted for.
func peerServicesWithCapturedCodec(t *testing.T) (*daemon, noderec.PeerServicesClaim, func() []Message) {
	t.Helper()
	d, claim := newPeerServicesScanner(t)
	var out bytes.Buffer
	d.codec = NewCodec(&out)
	sent := func() []Message {
		var msgs []Message
		decoder := json.NewDecoder(bytes.NewReader(out.Bytes()))
		for decoder.More() {
			var msg Message
			if err := decoder.Decode(&msg); err != nil {
				t.Fatalf("decode captured frame: %v", err)
			}
			if strings.HasPrefix(msg.Method, "errors:") {
				msgs = append(msgs, msg)
			}
		}
		return msgs
	}
	return d, claim, sent
}

// expireClaim ages the peer's stored claim past peerServicesMaxAge without
// waiting for it.
func expireClaim(d *daemon, hostUUID string) {
	d.dir.mu.Lock()
	defer d.dir.mu.Unlock()
	claim := d.dir.peerServices[hostUUID]
	claim.ObservedAt = time.Now().Add(-2 * peerServicesMaxAge).UnixMilli()
	d.dir.peerServices[hostUUID] = claim
}

func TestLapsedClaimIsReportedOnceAndClearedByTheNextClaim(t *testing.T) {
	d, claim, sent := peerServicesWithCapturedCodec(t)
	if !d.acceptPeerServices(claim) {
		t.Fatal("claim rejected")
	}

	d.reconcilePeerServices()
	requireErrorsFrames(t, sent())

	expireClaim(d, "peer")
	d.reconcilePeerServices()
	d.reconcilePeerServices()
	msgs := sent()
	requireErrorsFrames(t, msgs, "errors:report")
	var report sharederrors.ServiceError
	if err := json.Unmarshal(msgs[0].Params, &report); err != nil {
		t.Fatal(err)
	}
	if report.ID != peerUnreachableID || report.Severity != "warning" || report.Action != "none" || !strings.Contains(report.Message, "peer-name") {
		t.Fatalf("report = %+v", report)
	}

	next := claim
	next.Sequence = 2
	next.ObservedAt = time.Now().UnixMilli()
	if !d.acceptPeerServices(next) {
		t.Fatal("renewed claim rejected")
	}
	msgs = sent()
	requireErrorsFrames(t, msgs, "errors:report", "errors:clear")
	var cleared sharederrors.ClearParams
	if err := json.Unmarshal(msgs[1].Params, &cleared); err != nil {
		t.Fatal(err)
	}
	if cleared.ID != peerUnreachableID {
		t.Fatalf("clear names %q, want %q", cleared.ID, peerUnreachableID)
	}
}

// A peer whose mDNS record survives the lapse reverts to it rather than being
// forgotten, and the revert must not take the report out with it.
func TestLapsedClaimReportSurvivesRevertToBrowse(t *testing.T) {
	d, claim, sent := peerServicesWithCapturedCodec(t)
	d.onBrowse(DiscoveryEvent{Type: "discovered", Node: RawNode{ID: "browse-name", Addresses: []string{"192.0.2.90"}, TXT: []string{"uuid=peer", "cluster-uuid=peer", "cl=14444"}}})
	if !d.acceptPeerServices(claim) {
		t.Fatal("claim rejected")
	}

	expireClaim(d, "peer")
	d.reconcilePeerServices()
	requireErrorsFrames(t, sent(), "errors:report")
	if node := peerServicesNode(t, d); node.Name != "browse-name" {
		t.Fatalf("lapse did not revert to browse: %+v", node)
	}
}

// A peer that stops being paired is no longer something to report, and nothing
// else would clear it: it stops being enriched and its claims stop mattering.
func TestLapsedClaimReportIsClearedWhenThePeerIsUnpaired(t *testing.T) {
	d, claim, sent := peerServicesWithCapturedCodec(t)
	if !d.acceptPeerServices(claim) {
		t.Fatal("claim rejected")
	}
	expireClaim(d, "peer")
	d.reconcilePeerServices()
	requireErrorsFrames(t, sent(), "errors:report")

	if err := os.Remove(filepath.Join(d.clusterDir, "trusted", "peer.json")); err != nil {
		t.Fatal(err)
	}
	d.reloadTrust()
	requireErrorsFrames(t, sent(), "errors:report", "errors:clear")
}

// A claim that lapses after the peer was unpaired is not a paired peer going
// quiet, so it reports nothing.
func TestLapsedClaimOfAnUnadmittedPeerIsNotReported(t *testing.T) {
	d, claim, sent := peerServicesWithCapturedCodec(t)
	if !d.acceptPeerServices(claim) {
		t.Fatal("claim rejected")
	}
	if err := os.Remove(filepath.Join(d.clusterDir, "trusted", "peer.json")); err != nil {
		t.Fatal(err)
	}
	d.mesh.Refresh()
	expireClaim(d, "peer")
	d.reconcilePeerServices()
	requireErrorsFrames(t, sent())
}

// Unpairing a peer reported by the browse path clears it through reconcileTrust.
func TestNodeInfoOutageIsClearedWhenThePeerIsUnpaired(t *testing.T) {
	d, _, sent := peerServicesWithCapturedCodec(t)
	d.onBrowse(DiscoveryEvent{Type: "discovered", Node: RawNode{ID: "browse-name", Addresses: []string{"192.0.2.90"}, TXT: []string{"uuid=peer", "cluster-uuid=peer", "cl=14444"}}})
	node := peerServicesNode(t, d)
	if !node.Trusted {
		t.Fatalf("fixture peer is not trusted: %+v", node)
	}

	d.noteNodeInfo(node, "192.0.2.90:14300", false)
	requireErrorsFrames(t, sent(), "errors:report")

	if err := os.Remove(filepath.Join(d.clusterDir, "trusted", "peer.json")); err != nil {
		t.Fatal(err)
	}
	d.reloadTrust()
	requireErrorsFrames(t, sent(), "errors:report", "errors:clear")
}

// A scanner that starts while an admitted, reverse-only peer is down is never
// handed a claim, so it never sees one lapse. After the same grace a claim has,
// the peer is reported once, and the next claim clears it.
func TestSilentPeerIsReportedOnceAfterTheStartupGraceAndClearedByAClaim(t *testing.T) {
	d, claim, sent := peerServicesWithCapturedCodec(t)
	d.mesh.Refresh()
	started := time.Now()
	d.startedAt = started

	d.reconcilePeerServices()
	requireErrorsFrames(t, sent())

	d.startedAt = started.Add(-peerServicesMaxAge + time.Second)
	d.reconcilePeerServices()
	requireErrorsFrames(t, sent())

	d.startedAt = started.Add(-peerServicesMaxAge)
	for range 3 {
		d.reconcilePeerServices()
	}
	msgs := sent()
	requireErrorsFrames(t, msgs, "errors:report")
	var report sharederrors.ServiceError
	if err := json.Unmarshal(msgs[0].Params, &report); err != nil {
		t.Fatal(err)
	}
	if report.ID != peerUnreachableID || report.Severity != "warning" || report.Action != "none" {
		t.Fatalf("report = %+v", report)
	}

	claim.ObservedAt = time.Now().UnixMilli()
	if !d.acceptPeerServices(claim) {
		t.Fatal("claim rejected")
	}
	requireErrorsFrames(t, sent(), "errors:report", "errors:clear")
}

// A peer with a directory entry has an mDNS record, so enrichment owns it; the
// startup look must not report it a second time.
func TestSilentPeerWithADirectoryEntryIsLeftToEnrichment(t *testing.T) {
	d, _, sent := peerServicesWithCapturedCodec(t)
	d.mesh.Refresh()
	d.onBrowse(DiscoveryEvent{Type: "discovered", Node: RawNode{ID: "browse-name", Addresses: []string{"192.0.2.90"}, TXT: []string{"uuid=peer", "cluster-uuid=peer", "cl=14444"}}})
	d.startedAt = time.Now().Add(-peerServicesMaxAge)

	d.reconcilePeerServices()
	requireErrorsFrames(t, sent())
}

// The startup look is a single look: a peer evicted later is cleared by the
// eviction and not reported straight back.
func TestStartupLookDoesNotReportAnEvictedPeerAgain(t *testing.T) {
	d, _, sent := peerServicesWithCapturedCodec(t)
	d.mesh.Refresh()
	d.startedAt = time.Now().Add(-peerServicesMaxAge)
	d.reconcilePeerServices()
	requireErrorsFrames(t, sent(), "errors:report")

	d.forget("peer")
	d.reconcilePeerServices()
	requireErrorsFrames(t, sent(), "errors:report", "errors:clear")
}
