// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"nvpair-shared/clustertrust"
	"nvpair-shared/durablefile"
)

func TestRemoteManagementStrictParams(t *testing.T) {
	for _, input := range []string{`null`, `[]`, `{"allowed":true,"allowed":false}`, `{"caller":"local"}`, `{} {}`} {
		t.Run(input, func(t *testing.T) {
			if _, err := decodeRemoteManagementParams([]byte(input), map[string]bool{"allowed": true}); err == nil {
				t.Fatalf("accepted malformed params %s", input)
			}
		})
	}
}

func TestRemoteManagementUnconfirmedReplacementPreservesLandedDenial(t *testing.T) {
	path := filepath.Join(t.TempDir(), "grant.json")
	if err := os.WriteFile(path, []byte("allow"), 0o600); err != nil {
		t.Fatal(err)
	}
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	clusterFileReplacer = durablefile.Replacer{SyncDir: func(string) error { return errors.New("sync fault") }}
	if err := remoteManagementWrite(path, []byte("deny")); err == nil {
		t.Fatal("reported success with both durability confirmations failed")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, []byte("deny")) {
		t.Fatalf("landed denial rolled back: %q", data)
	}
}

func TestRemoteManagementConfirmsLandedReplacement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "grant.json")
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	calls := 0
	clusterFileReplacer = durablefile.Replacer{SyncDir: func(string) error {
		calls++
		if calls == 1 {
			return errors.New("first sync fault")
		}
		return nil
	}}
	if err := remoteManagementWrite(path, []byte("deny")); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("durability attempts = %d, want 2", calls)
	}
}

func TestRemoteManagementPreReplacePreservesPreviousBytes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "grant.json")
	if err := os.WriteFile(path, []byte("deny"), 0o600); err != nil {
		t.Fatal(err)
	}
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	clusterFileReplacer = durablefile.Replacer{Replace: func(string, string) error { return errors.New("replace fault") }}
	if err := remoteManagementWrite(path, []byte("allow")); err == nil {
		t.Fatal("reported success")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "deny" {
		t.Fatalf("previous denial lost: %q", data)
	}
}

func remoteManagementRPCFixture(t *testing.T) (*Manager, string, *bytes.Buffer) {
	t.Helper()
	m := remoteManagementUpgradeFixture(t)
	peer := remoteManagementAddPeer(t, m)
	if err := m.captureRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if err := m.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	output := new(bytes.Buffer)
	m.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), output})
	return m, peer.identity.NodeUUID, output
}

func remoteManagementRPCRequest(m *Manager, id, method string, params []byte) {
	rawID := json.RawMessage(id)
	m.handleMessage(&Message{ID: &rawID, Method: method, Params: params})
}

func remoteManagementRPCSetParams(t *testing.T, peer string, allowed bool, revision string) []byte {
	t.Helper()
	params, err := json.Marshal(struct {
		Peer     string `json:"peerUuid"`
		Allowed  bool   `json:"allowed"`
		Revision string `json:"expectedRevision"`
	}{peer, allowed, revision})
	if err != nil {
		t.Fatal(err)
	}
	return params
}

func remoteManagementRPCFrames(t *testing.T, output *bytes.Buffer) []Message {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(output.Bytes()))
	var frames []Message
	for {
		var frame Message
		if err := decoder.Decode(&frame); err == io.EOF {
			break
		} else if err != nil {
			t.Fatalf("decode RPC output: %v", err)
		}
		frames = append(frames, frame)
	}
	return frames
}

func remoteManagementRPCResponse(t *testing.T, frames []Message, id string) Message {
	t.Helper()
	var response Message
	count := 0
	for _, frame := range frames {
		if frame.ID != nil && string(*frame.ID) == id {
			response = frame
			count++
		}
	}
	if count != 1 {
		t.Fatalf("responses for id %s = %d, want 1", id, count)
	}
	return response
}

func remoteManagementAssertRPCSnapshot(t *testing.T, raw []byte, peer, revision, state string, allowed bool) {
	t.Helper()
	var snapshot remoteManagementSnapshot
	if err := json.Unmarshal(raw, &snapshot); err != nil {
		t.Fatalf("decode snapshot: %v", err)
	}
	if snapshot.Version != 1 || snapshot.Revision != revision || len(snapshot.Grants) != 1 {
		t.Fatalf("snapshot = %+v, want version 1 revision %s and one peer", snapshot, revision)
	}
	grant := snapshot.Grants[0]
	if grant.PeerUUID != peer || grant.State != state || grant.Allowed != allowed {
		t.Fatalf("grant = %+v, want peer %s state %s allowed %t", grant, peer, state, allowed)
	}
}

func TestRemoteManagementRPCPersistsDecisionsAndNotifiesAfterReopen(t *testing.T) {
	m, peer, output := remoteManagementRPCFixture(t)
	for _, decision := range []struct {
		allowed               bool
		expected, next, state string
	}{
		{false, "1", "2", "denied"}, {true, "2", "3", "allowed"},
	} {
		output.Reset()
		remoteManagementRPCRequest(m, "1", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, decision.allowed, decision.expected))
		frames := remoteManagementRPCFrames(t, output)
		response := remoteManagementRPCResponse(t, frames, "1")
		if response.Error != nil {
			t.Fatalf("set decision: %+v", response.Error)
		}
		remoteManagementAssertRPCSnapshot(t, response.Result, peer, decision.next, decision.state, decision.allowed)
		changed := 0
		for _, frame := range frames {
			if frame.Method == "cluster:remote-management-changed" {
				changed++
				remoteManagementAssertRPCSnapshot(t, frame.Params, peer, decision.next, decision.state, decision.allowed)
			}
		}
		if changed != 1 {
			t.Fatalf("changed notifications = %d, want 1", changed)
		}
		var err error
		m, err = NewManager(m.codec, filepath.Dir(m.clusterDir), m.port)
		if err != nil {
			t.Fatal(err)
		}
		output.Reset()
		remoteManagementRPCRequest(m, "2", "cluster:get-remote-management", []byte(`{}`))
		response = remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "2")
		if response.Error != nil {
			t.Fatalf("get after reopen: %+v", response.Error)
		}
		remoteManagementAssertRPCSnapshot(t, response.Result, peer, decision.next, decision.state, decision.allowed)
		permission := clustertrust.Open(m.clusterDir).RemoteManagement(peer)
		if permission.Allowed != decision.allowed || permission.State != decision.state || permission.Revision != decision.next {
			t.Fatalf("fresh permission after reopen = %+v", permission)
		}
		doc, _, err := m.readRemoteManagementLocked()
		if err != nil {
			t.Fatal(err)
		}
		if len(doc.Grants) != 1 || doc.Grants[0].DecisionRevision != decision.next || doc.Grants[0].Allowed != decision.allowed {
			t.Fatalf("persisted decision = %+v", doc)
		}
	}
}

func TestRemoteManagementRPCConcurrentCASCommitsExactlyOneDecision(t *testing.T) {
	m, peer, output := remoteManagementRPCFixture(t)
	ready := make(chan struct{}, 2)
	release := make(chan struct{})
	var workers sync.WaitGroup
	for _, request := range []struct {
		id      string
		allowed bool
	}{{"1", false}, {"2", true}} {
		params := remoteManagementRPCSetParams(t, peer, request.allowed, "1")
		workers.Add(1)
		go func(id string, params []byte) {
			defer workers.Done()
			ready <- struct{}{}
			<-release
			remoteManagementRPCRequest(m, id, "cluster:set-remote-management", params)
		}(request.id, params)
	}
	<-ready
	<-ready
	close(release)
	workers.Wait()
	frames := remoteManagementRPCFrames(t, output)
	successes, conflicts, changed := 0, 0, 0
	winnerAllowed := false
	for _, id := range []string{"1", "2"} {
		response := remoteManagementRPCResponse(t, frames, id)
		if response.Error == nil {
			successes++
			winnerAllowed = id == "2"
			state := "denied"
			if winnerAllowed {
				state = "allowed"
			}
			remoteManagementAssertRPCSnapshot(t, response.Result, peer, "2", state, winnerAllowed)
		} else if response.Error.Code == -32010 {
			conflicts++
		} else {
			t.Fatalf("unexpected setter error: %+v", response.Error)
		}
	}
	for _, frame := range frames {
		if frame.Method == "cluster:remote-management-changed" {
			changed++
			state := "denied"
			if winnerAllowed {
				state = "allowed"
			}
			remoteManagementAssertRPCSnapshot(t, frame.Params, peer, "2", state, winnerAllowed)
		}
	}
	if successes != 1 || conflicts != 1 || changed != 1 {
		t.Fatalf("concurrent outcomes success=%d conflict=%d changed=%d, want 1 each", successes, conflicts, changed)
	}
	doc, _, err := m.readRemoteManagementLocked()
	if err != nil {
		t.Fatal(err)
	}
	if doc.Revision != "2" || len(doc.Grants) != 1 || doc.Grants[0].Allowed != winnerAllowed || doc.Grants[0].DecisionRevision != "2" {
		t.Fatalf("CAS persisted wrong winner: %+v", doc)
	}
}

func TestRemoteManagementRPCMalformedParamsCannotChangeStorage(t *testing.T) {
	m, peer, output := remoteManagementRPCFixture(t)
	docPath, markerPath := m.remoteManagementPaths()
	beforeDoc := mustRemoteManagementRead(t, docPath)
	beforeMarker := mustRemoteManagementRead(t, markerPath)
	for _, test := range []struct{ name, fields string }{
		{"missing boolean", `"expectedRevision":"1"`},
		{"missing revision", `"allowed":false`},
		{"null boolean", `"allowed":null,"expectedRevision":"1"`},
		{"string boolean", `"allowed":"false","expectedRevision":"1"`},
		{"null revision", `"allowed":false,"expectedRevision":null`},
		{"numeric revision", `"allowed":false,"expectedRevision":1`},
		{"noncanonical revision", `"allowed":false,"expectedRevision":"01"`},
		{"duplicate field", `"allowed":false,"allowed":true,"expectedRevision":"1"`},
		{"unknown field", `"allowed":false,"expectedRevision":"1","caller":"local"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			output.Reset()
			params := []byte(fmt.Sprintf(`{"peerUuid":%q,%s}`, peer, test.fields))
			remoteManagementRPCRequest(m, "1", "cluster:set-remote-management", params)
			frames := remoteManagementRPCFrames(t, output)
			response := remoteManagementRPCResponse(t, frames, "1")
			if response.Error == nil || response.Error.Code != -32602 || len(response.Result) != 0 {
				t.Fatalf("malformed request response = %+v", response)
			}
			if len(frames) != 1 {
				t.Fatalf("malformed request emitted extra frames: %+v", frames)
			}
			if !bytes.Equal(beforeDoc, mustRemoteManagementRead(t, docPath)) || !bytes.Equal(beforeMarker, mustRemoteManagementRead(t, markerPath)) {
				t.Fatal("malformed request changed authority bytes")
			}
			doc, _, err := m.readRemoteManagementLocked()
			if err != nil {
				t.Fatal(err)
			}
			if doc.Revision != "1" {
				t.Fatalf("malformed request changed revision: %s", doc.Revision)
			}
		})
	}
}

func TestRemoteManagementRPCPreReplaceFailureRetainsAllow(t *testing.T) {
	m, peer, output := remoteManagementRPCFixture(t)
	docPath, _ := m.remoteManagementPaths()
	before := mustRemoteManagementRead(t, docPath)
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	clusterFileReplacer = durablefile.Replacer{Replace: func(string, string) error { return errors.New("RPC replace fault") }}
	remoteManagementRPCRequest(m, "1", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, false, "1"))
	frames := remoteManagementRPCFrames(t, output)
	response := remoteManagementRPCResponse(t, frames, "1")
	if response.Error == nil || response.Error.Code != codeInternalError || len(response.Result) != 0 {
		t.Fatalf("failed replacement reported success: %+v", response)
	}
	if len(frames) != 1 {
		t.Fatalf("pre-replace failure published change: %+v", frames)
	}
	if !bytes.Equal(before, mustRemoteManagementRead(t, docPath)) {
		t.Fatal("pre-replace failure changed durable allow")
	}
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(peer); !permission.Allowed || permission.Revision != "1" {
		t.Fatalf("pre-replace failure lost old allow: %+v", permission)
	}
}

func TestRemoteManagementRPCUncertainDenyReportsFailureAndPublishesTruth(t *testing.T) {
	m, peer, output := remoteManagementRPCFixture(t)
	docPath, _ := m.remoteManagementPaths()
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	attempts := 0
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if target != docPath {
			t.Fatalf("unexpected setter replacement: %s", target)
		}
		attempts++
		if err := os.Rename(tmp, target); err != nil {
			return err
		}
		return durablefile.ErrReplaced
	}}
	remoteManagementRPCRequest(m, "1", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, false, "1"))
	frames := remoteManagementRPCFrames(t, output)
	response := remoteManagementRPCResponse(t, frames, "1")
	if response.Error == nil || response.Error.Code != codeInternalError || len(response.Result) != 0 {
		t.Fatalf("uncertainty reported success: %+v", response)
	}
	if attempts != 2 {
		t.Fatalf("uncertain denial replacement attempts = %d, want 2", attempts)
	}
	changed := 0
	for _, frame := range frames {
		if frame.Method == "cluster:remote-management-changed" {
			changed++
			remoteManagementAssertRPCSnapshot(t, frame.Params, peer, "2", "denied", false)
		}
	}
	if changed != 1 {
		t.Fatalf("truthful landed change notifications = %d, want 1", changed)
	}
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(peer); permission.Allowed || permission.State != "denied" || permission.Revision != "2" {
		t.Fatalf("landed denial not effective: %+v", permission)
	}
	output.Reset()
	remoteManagementRPCRequest(m, "2", "cluster:get-remote-management", []byte(`{}`))
	response = remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "2")
	if response.Error != nil {
		t.Fatalf("readback after uncertainty: %+v", response.Error)
	}
	remoteManagementAssertRPCSnapshot(t, response.Result, peer, "2", "denied", false)
}

func TestRemoteManagementRPCUncertainAllowDoesNotAuthorizeIndependentMesh(t *testing.T) {
	m, peer, output := remoteManagementRPCFixture(t)
	remoteManagementRPCRequest(m, "1", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, false, "1"))
	denied := remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "1")
	if denied.Error != nil {
		t.Fatalf("save initial local denial: %+v", denied.Error)
	}
	remoteManagementAssertRPCSnapshot(t, denied.Result, peer, "2", "denied", false)
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(peer); permission.Allowed || permission.State != "denied" || permission.Revision != "2" {
		t.Fatalf("initial durable denial unavailable: %+v", permission)
	}
	output.Reset()
	docPath, _ := m.remoteManagementPaths()
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	grantReplacements := 0
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if err := os.Rename(tmp, target); err != nil {
			return err
		}
		if target == docPath {
			grantReplacements++
			return durablefile.ErrReplaced
		}
		return nil
	}}
	remoteManagementRPCRequest(m, "2", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, true, "2"))
	frames := remoteManagementRPCFrames(t, output)
	response := remoteManagementRPCResponse(t, frames, "2")
	if response.Error == nil || response.Error.Code != codeInternalError || len(response.Result) != 0 {
		t.Fatalf("uncertain allow reported successful grant: %+v", response)
	}
	if grantReplacements != 2 {
		t.Fatalf("uncertain allow replacements = %d, want 2", grantReplacements)
	}
	if permission := clustertrust.Open(m.clusterDir).RemoteManagement(peer); permission.Allowed {
		t.Fatalf("independent reader authorized unconfirmed local allow after durable denial: %+v", permission)
	}
	for _, frame := range frames {
		if frame.Method == "cluster:remote-management-changed" {
			var snapshot remoteManagementSnapshot
			if err := json.Unmarshal(frame.Params, &snapshot); err != nil {
				t.Fatal(err)
			}
			for _, grant := range snapshot.Grants {
				if grant.Allowed {
					t.Fatalf("uncertain allow published allowed change: %+v", snapshot)
				}
			}
		}
	}
}

func remoteManagementRPCDeniedFixture(t *testing.T) (*Manager, string, *bytes.Buffer) {
	t.Helper()
	m, peer, output := remoteManagementRPCFixture(t)
	remoteManagementRPCRequest(m, "1", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, false, "1"))
	response := remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "1")
	if response.Error != nil {
		t.Fatalf("save durable denial: %+v", response.Error)
	}
	remoteManagementAssertRPCSnapshot(t, response.Result, peer, "2", "denied", false)
	output.Reset()
	return m, peer, output
}

func TestRemoteManagementRPCAllowGuardAndDocumentFailuresRemainDenied(t *testing.T) {
	for _, fault := range []string{"guard before replacement", "guard repeated uncertainty", "document before replacement"} {
		t.Run(fault, func(t *testing.T) {
			m, peer, output := remoteManagementRPCDeniedFixture(t)
			incarnation, ok := m.mesh.PeerServiceIncarnation(peer)
			if !ok {
				t.Fatal("fixture incarnation unavailable")
			}
			docPath, _ := m.remoteManagementPaths()
			beforeDoc := mustRemoteManagementRead(t, docPath)
			var audit bytes.Buffer
			oldWriter, oldFlags, oldPrefix := log.Writer(), log.Flags(), log.Prefix()
			log.SetOutput(&audit)
			log.SetFlags(0)
			log.SetPrefix("")
			t.Cleanup(func() { log.SetOutput(oldWriter); log.SetFlags(oldFlags); log.SetPrefix(oldPrefix) })
			original := clusterFileReplacer
			t.Cleanup(func() { clusterFileReplacer = original })
			guardWrites, docWrites := 0, 0
			const sentinel = "PRIVATE_GUARD_FILESYSTEM_ERROR_DO_NOT_LOG"
			clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
				if target == m.remoteManagementInitializingPath() {
					guardWrites++
					if fault == "guard before replacement" {
						return errors.New(sentinel)
					}
					if fault == "guard repeated uncertainty" {
						if err := os.Rename(tmp, target); err != nil {
							return err
						}
						return fmt.Errorf("%w: %s", durablefile.ErrReplaced, sentinel)
					}
				}
				if target == docPath {
					docWrites++
					if fault == "document before replacement" {
						return errors.New(sentinel)
					}
				}
				return os.Rename(tmp, target)
			}}
			remoteManagementRPCRequest(m, "2", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, true, "2"))
			frames := remoteManagementRPCFrames(t, output)
			response := remoteManagementRPCResponse(t, frames, "2")
			if response.Error == nil || response.Error.Code != codeInternalError || len(response.Result) != 0 {
				t.Fatalf("failed allow reported success: %+v", response)
			}
			wantGuards, wantDocs, outcome := 1, 0, "failed"
			if fault == "guard repeated uncertainty" {
				wantGuards, outcome = 2, "uncertain"
			}
			if fault == "document before replacement" {
				wantDocs = 1
			}
			if guardWrites != wantGuards || docWrites != wantDocs {
				t.Fatalf("writes guard=%d document=%d, want %d/%d", guardWrites, docWrites, wantGuards, wantDocs)
			}
			if !bytes.Equal(beforeDoc, mustRemoteManagementRead(t, docPath)) {
				t.Fatal("failed allow changed prior denial")
			}
			if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer); got.Allowed {
				t.Fatalf("failed allow became effective: %+v", got)
			}
			if fault == "document before replacement" {
				if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer); got.State != "denied" || got.Revision != "2" {
					t.Fatalf("failed allow suppressed prior denial: %+v", got)
				}
				if _, err := os.Lstat(m.remoteManagementInitializingPath()); !os.IsNotExist(err) {
					t.Fatalf("pre-replace failure retained operation guard: %v", err)
				}
			}
			for _, frame := range frames {
				if frame.Method == "cluster:remote-management-changed" {
					var snapshot remoteManagementSnapshot
					if err := json.Unmarshal(frame.Params, &snapshot); err != nil {
						t.Fatal(err)
					}
					for _, grant := range snapshot.Grants {
						if grant.Allowed {
							t.Fatal("failed allow pushed allowed=true")
						}
					}
				}
			}
			event := audit.String()
			if strings.Count(event, "\n") != 1 || len(event) > 1024 || strings.Contains(event, sentinel) || strings.Contains(event, m.clusterDir) {
				t.Fatalf("invalid bounded private audit: %q", event)
			}
			for _, field := range []string{"actor=local", "peer=" + peer, "attemptRevision=3", "previousRevision=2", "incarnation=" + incarnation, "old=false", "new=true", "outcome=" + outcome} {
				if !strings.Contains(event, field) {
					t.Fatalf("audit missing %s: %q", field, event)
				}
			}
			clusterFileReplacer = original
			// Recovery confirms the surviving document, even when the allow never
			// reached its replacement. It must preserve the original deny.
			reopened, err := NewManager(m.codec, filepath.Dir(m.clusterDir), m.port)
			if err != nil {
				t.Fatal(err)
			}
			if got := clustertrust.Open(reopened.clusterDir).RemoteManagement(peer); got.Allowed || got.State != "denied" || got.Revision != "2" {
				t.Fatalf("failed allow recovery lost denial: %+v", got)
			}
		})
	}
}

func TestRemoteManagementRPCFailedAllowBeforeReplacePreservesConfirmedAllow(t *testing.T) {
	m, peer, output := remoteManagementRPCFixture(t)
	docPath, _ := m.remoteManagementPaths()
	before := mustRemoteManagementRead(t, docPath)
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	docWrites := 0
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if target == docPath {
			docWrites++
			return errors.New("allow document replace fault")
		}
		return os.Rename(tmp, target)
	}}
	remoteManagementRPCRequest(m, "1", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, true, "1"))
	frames := remoteManagementRPCFrames(t, output)
	response := remoteManagementRPCResponse(t, frames, "1")
	if response.Error == nil || response.Error.Code != codeInternalError || len(response.Result) != 0 {
		t.Fatalf("failed redundant allow response = %+v", response)
	}
	if len(frames) != 1 || docWrites != 1 {
		t.Fatalf("failed allow frames=%d document attempts=%d, want 1 each", len(frames), docWrites)
	}
	if !bytes.Equal(before, mustRemoteManagementRead(t, docPath)) {
		t.Fatal("pre-replace failure changed prior allow bytes")
	}
	if _, err := os.Lstat(m.remoteManagementInitializingPath()); !os.IsNotExist(err) {
		t.Fatalf("operation guard suppressed confirmed prior allow: %v", err)
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer); !got.Allowed || got.State != "allowed" || got.Revision != "1" {
		t.Fatalf("failed redundant allow suppressed confirmed authority: %+v", got)
	}
	output.Reset()
	remoteManagementRPCRequest(m, "2", "cluster:get-remote-management", []byte(`{}`))
	response = remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "2")
	if response.Error != nil {
		t.Fatalf("prior allow readback failed: %+v", response.Error)
	}
	remoteManagementAssertRPCSnapshot(t, response.Result, peer, "1", "allowed", true)
}

func TestRemoteManagementRPCFailedAllowGuardCleanupFailureStaysFenced(t *testing.T) {
	m, peer, output := remoteManagementRPCFixture(t)
	docPath, _ := m.remoteManagementPaths()
	before := mustRemoteManagementRead(t, docPath)
	originalReplacer, originalRemove := clusterFileReplacer, removeFile
	t.Cleanup(func() { clusterFileReplacer = originalReplacer; removeFile = originalRemove })
	docWrites, cleanups := 0, 0
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if target == docPath {
			docWrites++
			return errors.New("pre-replace document fault")
		}
		return os.Rename(tmp, target)
	}}
	removeFile = func(path string) error {
		if path == m.remoteManagementInitializingPath() {
			cleanups++
			return errors.New("operation guard cleanup fault")
		}
		return originalRemove(path)
	}
	remoteManagementRPCRequest(m, "1", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, true, "1"))
	response := remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "1")
	if response.Error == nil || response.Error.Code != codeInternalError || len(response.Result) != 0 {
		t.Fatalf("cleanup failure reported success: %+v", response)
	}
	if docWrites != 1 || cleanups != 1 {
		t.Fatalf("document attempts=%d cleanups=%d, want 1 each", docWrites, cleanups)
	}
	if !bytes.Equal(before, mustRemoteManagementRead(t, docPath)) {
		t.Fatal("cleanup failure changed prior authority")
	}
	if _, err := os.Lstat(m.remoteManagementInitializingPath()); err != nil {
		t.Fatalf("cleanup failure lost guard: %v", err)
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer); got.Allowed || got.State != "unavailable" {
		t.Fatalf("cleanup failure did not retain fence: %+v", got)
	}
	output.Reset()
	remoteManagementRPCRequest(m, "2", "cluster:get-remote-management", []byte(`{}`))
	response = remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "2")
	if response.Error != nil {
		t.Fatalf("unavailable readback failed: %+v", response.Error)
	}
	remoteManagementAssertRPCSnapshot(t, response.Result, peer, "", "unavailable", false)
	if cleanups != 1 || docWrites != 1 {
		t.Fatal("readback attempted live repair")
	}
	clusterFileReplacer, removeFile = originalReplacer, originalRemove
	reopened, err := NewManager(m.codec, filepath.Dir(m.clusterDir), m.port)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, mustRemoteManagementRead(t, docPath)) {
		t.Fatal("startup reconstructed old authority")
	}
	if got := clustertrust.Open(reopened.clusterDir).RemoteManagement(peer); !got.Allowed || got.State != "allowed" || got.Revision != "1" {
		t.Fatalf("startup did not restore confirmed old allow: %+v", got)
	}
}

func TestRemoteManagementRPCUncertainAllowReopenConfirmsSameBytes(t *testing.T) {
	m, peer, output := remoteManagementRPCDeniedFixture(t)
	docPath, markerPath := m.remoteManagementPaths()
	beforeMarker := mustRemoteManagementRead(t, markerPath)
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if err := os.Rename(tmp, target); err != nil {
			return err
		}
		if target == docPath {
			return durablefile.ErrReplaced
		}
		return nil
	}}
	remoteManagementRPCRequest(m, "2", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, true, "2"))
	response := remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "2")
	if response.Error == nil {
		t.Fatal("uncertain allow reported success")
	}
	beforeDoc := mustRemoteManagementRead(t, docPath)
	beforeGuard := mustRemoteManagementRead(t, m.remoteManagementInitializingPath())
	output.Reset()
	remoteManagementRPCRequest(m, "3", "cluster:get-remote-management", []byte(`{}`))
	response = remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "3")
	if response.Error != nil {
		t.Fatalf("read-only unavailable snapshot failed: %+v", response.Error)
	}
	remoteManagementAssertRPCSnapshot(t, response.Result, peer, "", "unavailable", false)
	if !bytes.Equal(beforeDoc, mustRemoteManagementRead(t, docPath)) || !bytes.Equal(beforeGuard, mustRemoteManagementRead(t, m.remoteManagementInitializingPath())) {
		t.Fatal("getter repaired uncertain allow")
	}
	remoteManagementRPCRequest(m, "4", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, false, "3"))
	response = remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "4")
	if response.Error == nil {
		t.Fatal("later setter bypassed unresolved publication guard")
	}
	if !bytes.Equal(beforeDoc, mustRemoteManagementRead(t, docPath)) {
		t.Fatal("blocked later setter changed authority")
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer); got.Allowed {
		t.Fatalf("uncertain allow leaked before recovery: %+v", got)
	}
	clusterFileReplacer = original
	reopened, err := NewManager(m.codec, filepath.Dir(m.clusterDir), m.port)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(beforeDoc, mustRemoteManagementRead(t, docPath)) || !bytes.Equal(beforeMarker, mustRemoteManagementRead(t, markerPath)) {
		t.Fatal("recovery reconstructed decision or migration")
	}
	if _, err := os.Lstat(m.remoteManagementInitializingPath()); !os.IsNotExist(err) {
		t.Fatalf("confirmed guard remains: %v", err)
	}
	if got := clustertrust.Open(reopened.clusterDir).RemoteManagement(peer); !got.Allowed || got.State != "allowed" || got.Revision != "3" {
		t.Fatalf("confirmed same-byte allow unavailable: %+v", got)
	}
	output.Reset()
	remoteManagementRPCRequest(reopened, "5", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, false, "3"))
	response = remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "5")
	if response.Error != nil {
		t.Fatalf("later local deny failed: %+v", response.Error)
	}
	remoteManagementAssertRPCSnapshot(t, response.Result, peer, "4", "denied", false)
	if err := reopened.finalizeRemoteManagementMigrationLocked(); err != nil {
		t.Fatal(err)
	}
	if got := clustertrust.Open(reopened.clusterDir).RemoteManagement(peer); got.Allowed || got.State != "denied" || got.Revision != "4" {
		t.Fatalf("reconfirmation replay lost later local deny: %+v", got)
	}
}

func TestRemoteManagementRPCConfirmedAllowPublishesAfterGuardedRewrite(t *testing.T) {
	m, peer, output := remoteManagementRPCDeniedFixture(t)
	docPath, _ := m.remoteManagementPaths()
	original := clusterFileReplacer
	t.Cleanup(func() { clusterFileReplacer = original })
	docWrites := 0
	clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
		if err := os.Rename(tmp, target); err != nil {
			return err
		}
		if target == docPath {
			docWrites++
			if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer); got.Allowed {
				t.Fatalf("allow published inside unconfirmed rewrite: %+v", got)
			}
			if docWrites == 1 {
				return durablefile.ErrReplaced
			}
		}
		return nil
	}}
	remoteManagementRPCRequest(m, "2", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, true, "2"))
	frames := remoteManagementRPCFrames(t, output)
	response := remoteManagementRPCResponse(t, frames, "2")
	if response.Error != nil {
		t.Fatalf("confirmed allow failed: %+v", response.Error)
	}
	if docWrites < 2 {
		t.Fatalf("no confirming document replacement: %d", docWrites)
	}
	remoteManagementAssertRPCSnapshot(t, response.Result, peer, "3", "allowed", true)
	changed := 0
	for _, frame := range frames {
		if frame.Method == "cluster:remote-management-changed" {
			changed++
			remoteManagementAssertRPCSnapshot(t, frame.Params, peer, "3", "allowed", true)
		}
	}
	if changed != 1 {
		t.Fatalf("confirmed allow change pushes = %d, want 1", changed)
	}
	if got := clustertrust.Open(m.clusterDir).RemoteManagement(peer); !got.Allowed || got.Revision != "3" {
		t.Fatalf("confirmed allow not published: %+v", got)
	}
}

func TestRemoteManagementRPCAuditRecordsEachWriteOutcomeWithoutPrivateData(t *testing.T) {
	for _, outcome := range []string{"committed", "failed", "uncertain"} {
		t.Run(outcome, func(t *testing.T) {
			m, peer, output := remoteManagementRPCFixture(t)
			incarnation, ok := m.mesh.PeerServiceIncarnation(peer)
			if !ok {
				t.Fatal("fixture incarnation unavailable")
			}
			var audit bytes.Buffer
			oldWriter, oldFlags, oldPrefix := log.Writer(), log.Flags(), log.Prefix()
			log.SetOutput(&audit)
			log.SetFlags(0)
			log.SetPrefix("")
			t.Cleanup(func() { log.SetOutput(oldWriter); log.SetFlags(oldFlags); log.SetPrefix(oldPrefix) })
			const sentinel = "PRIVATE_PIN_CREDENTIAL_POLICY_BODY_DO_NOT_LOG"
			privateParams := []byte(fmt.Sprintf(`{"peerUuid":%q,"allowed":false,"expectedRevision":"1","private":%q}`, peer, sentinel))
			remoteManagementRPCRequest(m, "1", "cluster:set-remote-management", privateParams)
			invalid := remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "1")
			if invalid.Error == nil || invalid.Error.Code != -32602 {
				t.Fatalf("private extra field accepted: %+v", invalid)
			}
			if audit.Len() != 0 {
				t.Fatalf("rejected private params produced decision output: %q", audit.String())
			}
			output.Reset()
			original := clusterFileReplacer
			t.Cleanup(func() { clusterFileReplacer = original })
			attempts := 0
			if outcome != "committed" {
				clusterFileReplacer = durablefile.Replacer{Replace: func(tmp, target string) error {
					attempts++
					if outcome == "failed" {
						return errors.New(sentinel)
					}
					if err := os.Rename(tmp, target); err != nil {
						return err
					}
					return fmt.Errorf("%w: %s", durablefile.ErrReplaced, sentinel)
				}}
			}
			remoteManagementRPCRequest(m, "2", "cluster:set-remote-management", remoteManagementRPCSetParams(t, peer, false, "1"))
			response := remoteManagementRPCResponse(t, remoteManagementRPCFrames(t, output), "2")
			if outcome == "committed" {
				if response.Error != nil {
					t.Fatalf("successful audited write failed: %+v", response.Error)
				}
				remoteManagementAssertRPCSnapshot(t, response.Result, peer, "2", "denied", false)
			} else {
				if response.Error == nil || response.Error.Code != codeInternalError || len(response.Result) != 0 {
					t.Fatalf("failed audited write reported success: %+v", response)
				}
				wantAttempts := 1
				if outcome == "uncertain" {
					wantAttempts = 2
				}
				if attempts != wantAttempts {
					t.Fatalf("write attempts = %d, want %d", attempts, wantAttempts)
				}
			}
			text := audit.String()
			if strings.Contains(text, sentinel) || strings.Contains(text, m.clusterDir) || strings.Contains(text, "CERTIFICATE") || len(text) > 1024 {
				t.Fatalf("audit contains private or unbounded data: %q", text)
			}
			lines := strings.Split(strings.TrimSpace(text), "\n")
			if len(lines) != 1 {
				t.Fatalf("audit events = %d, want 1: %q", len(lines), text)
			}
			fields := strings.Fields(lines[0])
			if len(fields) != 10 || fields[0] != "remote" || fields[1] != "management" {
				t.Fatalf("unexpected audit shape: %q", text)
			}
			metadata := make(map[string]string)
			for _, field := range fields[2:] {
				key, value, found := strings.Cut(field, "=")
				if !found {
					t.Fatalf("invalid audit field %q", field)
				}
				if _, exists := metadata[key]; exists {
					t.Fatalf("duplicate audit key %q", key)
				}
				metadata[key] = value
			}
			for key, want := range map[string]string{"actor": "local", "peer": peer, "attemptRevision": "2", "previousRevision": "1", "incarnation": incarnation, "old": "true", "new": "false", "outcome": outcome} {
				if metadata[key] != want {
					t.Fatalf("audit %s = %q, want %q", key, metadata[key], want)
				}
			}
			permission := clustertrust.Open(m.clusterDir).RemoteManagement(peer)
			if outcome == "failed" {
				if !permission.Allowed || permission.Revision != "1" {
					t.Fatalf("pre-replace audit disagrees with authority: %+v", permission)
				}
			} else if permission.Allowed || permission.State != "denied" || permission.Revision != "2" {
				t.Fatalf("landed audit disagrees with authority: %+v", permission)
			}
		})
	}
}
