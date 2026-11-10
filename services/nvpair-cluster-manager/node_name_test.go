// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// syncBuffer is a goroutine-safe codec sink.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

// lines returns every JSON-RPC frame written so far.
func (b *syncBuffer) lines(t *testing.T) []map[string]json.RawMessage {
	t.Helper()
	b.mu.Lock()
	data := append([]byte(nil), b.buf.Bytes()...)
	b.mu.Unlock()
	var out []map[string]json.RawMessage
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		var frame map[string]json.RawMessage
		if err := json.Unmarshal(scanner.Bytes(), &frame); err != nil {
			t.Fatalf("decode frame %q: %v", scanner.Text(), err)
		}
		out = append(out, frame)
	}
	return out
}

func captureCodec(m *Manager) *syncBuffer {
	out := &syncBuffer{}
	m.codec = NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), out})
	return out
}

// nameNotifications returns the params of every cluster:node-name-changed
// notification in out.
func nameNotifications(t *testing.T, out *syncBuffer) []nodeNameView {
	t.Helper()
	var views []nodeNameView
	for _, frame := range out.lines(t) {
		if string(frame["method"]) != `"cluster:node-name-changed"` {
			continue
		}
		var view nodeNameView
		if err := json.Unmarshal(frame["params"], &view); err != nil {
			t.Fatalf("decode notification: %v", err)
		}
		views = append(views, view)
	}
	return views
}

func readIdentityFile(t *testing.T, m *Manager) identityFile {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(m.clusterDir, "identity.json"))
	if err != nil {
		t.Fatal(err)
	}
	var idf identityFile
	if err := json.Unmarshal(data, &idf); err != nil {
		t.Fatal(err)
	}
	return idf
}

func callSetNodeName(t *testing.T, m *Manager, params string) map[string]json.RawMessage {
	t.Helper()
	out := captureCodec(m)
	id := json.RawMessage(`7`)
	m.handleSetNodeName(&Message{ID: &id, Params: json.RawMessage(params)})
	for _, frame := range out.lines(t) {
		if string(frame["id"]) == "7" {
			return frame
		}
	}
	t.Fatal("no response")
	return nil
}

func reopenManager(t *testing.T, m *Manager) *Manager {
	t.Helper()
	reopened, err := NewManager(NewCodec(struct {
		io.Reader
		io.Writer
	}{strings.NewReader(""), io.Discard}), filepath.Dir(m.clusterDir), m.port)
	if err != nil {
		t.Fatalf("reopen manager: %v", err)
	}
	t.Cleanup(reopened.peerServices.CloseIdle)
	t.Cleanup(reopened.removalPeerServices.CloseIdle)
	return reopened
}

func TestNodeNameDefaultShownWhenUnset(t *testing.T) {
	m := newTestManager(t)
	want := defaultNodeName(m.identity.NodeUUID)
	view := m.selfNameView()
	if view.Name != want || view.CustomName != nil || view.NodeUUID != m.identity.NodeUUID {
		t.Fatalf("view = %+v, want default name %q and nil custom name", view, want)
	}
}

func TestNodeNameSetPersistsAcrossRestart(t *testing.T) {
	m := newTestManager(t)
	createdAt := readIdentityFile(t, m).CreatedAt
	name := "  Studio Mac  "
	view, err := m.setSelfName(&name)
	if err != nil {
		t.Fatalf("setSelfName: %v", err)
	}
	if view.Name != "Studio Mac" || view.CustomName == nil || *view.CustomName != "Studio Mac" {
		t.Fatalf("view = %+v, want normalized custom name", view)
	}
	idf := readIdentityFile(t, m)
	if idf.DisplayName != "Studio Mac" || idf.CreatedAt != createdAt || idf.NodeUUID != m.identity.NodeUUID {
		t.Fatalf("identity.json = %+v, want display_name and unchanged created_at %d", idf, createdAt)
	}

	reopened := reopenManager(t, m)
	if got := reopened.selfName(); got != "Studio Mac" {
		t.Fatalf("name after restart = %q, want %q", got, "Studio Mac")
	}
	if reopened.identity.NodeUUID != m.identity.NodeUUID {
		t.Fatal("restart changed the node UUID")
	}
}

func TestNodeNameClearReturnsToDefault(t *testing.T) {
	m := newTestManager(t)
	name := "Custom"
	if _, err := m.setSelfName(&name); err != nil {
		t.Fatal(err)
	}
	frame := callSetNodeName(t, m, `{"name":null}`)
	var view nodeNameView
	if err := json.Unmarshal(frame["result"], &view); err != nil {
		t.Fatalf("decode result %s: %v", frame["result"], err)
	}
	if view.Name != m.generatedName || view.CustomName != nil {
		t.Fatalf("view = %+v, want generated name and nil custom name", view)
	}
	if idf := readIdentityFile(t, m); idf.DisplayName != "" {
		t.Fatalf("display_name = %q after clear, want empty", idf.DisplayName)
	}
}

func TestNodeNameRejectsInvalidParams(t *testing.T) {
	tests := []struct {
		name       string
		params     string
		wantReason string
		wantMsg    string
	}{
		{name: "missing key", params: `{}`},
		{name: "unknown key", params: `{"name":"a","other":1}`},
		{name: "duplicate key", params: `{"name":"a","name":"b"}`},
		{name: "number value", params: `{"name":5}`},
		{name: "not an object", params: `"a"`},
		{name: "empty name", params: `{"name":"   "}`, wantReason: "empty", wantMsg: "node name is empty"},
		{name: "control character", params: `{"name":"a\u0007b"}`, wantReason: "invalid-character"},
		{name: "too long", params: `{"name":"` + strings.Repeat("x", 129) + `"}`, wantReason: "too-long", wantMsg: "node name is longer than 128 bytes"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := newTestManager(t)
			before := readIdentityFile(t, m)
			frame := callSetNodeName(t, m, tt.params)
			var rpcErr struct {
				Code    int               `json:"code"`
				Message string            `json:"message"`
				Data    map[string]string `json:"data"`
			}
			if err := json.Unmarshal(frame["error"], &rpcErr); err != nil {
				t.Fatalf("decode error %s: %v", frame["error"], err)
			}
			if rpcErr.Code != codeInvalidParams || rpcErr.Data["field"] != "name" || rpcErr.Data["reason"] != tt.wantReason {
				t.Fatalf("error = %+v, want code %d field name reason %q", rpcErr, codeInvalidParams, tt.wantReason)
			}
			if tt.wantMsg != "" && rpcErr.Message != tt.wantMsg {
				t.Fatalf("message = %q, want %q", rpcErr.Message, tt.wantMsg)
			}
			if after := readIdentityFile(t, m); after != before {
				t.Fatalf("identity.json changed on a rejected name: %+v -> %+v", before, after)
			}
		})
	}
}

func TestNodeNameSameAsDefaultIsPersisted(t *testing.T) {
	m := newTestManager(t)
	out := captureCodec(m)
	name := m.generatedName
	view, err := m.setSelfName(&name)
	if err != nil {
		t.Fatal(err)
	}
	if view.CustomName == nil || *view.CustomName != name {
		t.Fatalf("view = %+v, want custom name %q", view, name)
	}
	if idf := readIdentityFile(t, m); idf.DisplayName != name {
		t.Fatalf("display_name = %q, want %q", idf.DisplayName, name)
	}
	notes := nameNotifications(t, out)
	if len(notes) != 1 || notes[0].CustomName == nil || *notes[0].CustomName != name {
		t.Fatalf("notifications = %+v, want one carrying the custom name", notes)
	}
}

func TestNodeNameChangeEmitsNotification(t *testing.T) {
	m := newTestManager(t)
	out := captureCodec(m)
	name := "Desk PC"
	if _, err := m.setSelfName(&name); err != nil {
		t.Fatal(err)
	}
	if _, err := m.setSelfName(&name); err != nil {
		t.Fatal(err)
	}
	notes := nameNotifications(t, out)
	if len(notes) != 1 {
		t.Fatalf("got %d notifications, want 1 (an unchanged set emits none)", len(notes))
	}
	if notes[0].NodeUUID != m.identity.NodeUUID || notes[0].Name != "Desk PC" || notes[0].CustomName == nil || *notes[0].CustomName != "Desk PC" {
		t.Fatalf("notification = %+v", notes[0])
	}
}

func TestNodeNameRenamesSelfMemberAndReconciles(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, false)
	pairClusterFixtures(t, a, b)
	name := "Renamed A"
	if _, err := a.m.setSelfName(&name); err != nil {
		t.Fatal(err)
	}
	if self, ok := a.m.memberByUUID(a.m.identity.NodeUUID); !ok || self.Name != name {
		t.Fatalf("self member = %+v, want name %q", self, name)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		peer, ok := b.m.memberByUUID(a.m.identity.NodeUUID)
		if b.rosters.Load() > 0 && ok && peer.Name == name {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("peer did not learn the rename: rosters=%d member=%+v", b.rosters.Load(), peer)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestSnapshotDisplayNamePrecedence(t *testing.T) {
	m := newTestManager(t)
	nick, named, bare := "11111111-0000-4000-8000-000000000000", "22222222-0000-4000-8000-000000000000", "33333333-0000-4000-8000-000000000000"
	m.upsertMember(&ClusterNode{NodeUUID: nick, Name: "own", Nickname: "nick", State: stateMember})
	m.upsertMember(&ClusterNode{NodeUUID: named, Name: "own", State: stateMember})
	m.upsertMember(&ClusterNode{NodeUUID: bare, State: stateMember})
	want := map[string]string{nick: "nick", named: "own", bare: "PAIR 33333333"}
	for _, n := range m.snapshotNodes() {
		if n.DisplayName != want[n.NodeUUID] {
			t.Fatalf("displayName for %s = %q, want %q", n.NodeUUID, n.DisplayName, want[n.NodeUUID])
		}
	}
	stored, _ := m.memberByUUID(named)
	if stored.DisplayName != "" {
		t.Fatalf("stored member carries displayName %q; it is snapshot-only", stored.DisplayName)
	}
}

func TestUpsertMemberKeepsNickname(t *testing.T) {
	m := newTestManager(t)
	uuid := "44444444-0000-4000-8000-000000000000"
	m.upsertMember(&ClusterNode{NodeUUID: uuid, Name: "old", Nickname: "mine", State: stateMember})
	m.upsertMember(&ClusterNode{NodeUUID: uuid, Name: "new", State: stateMember})
	n, ok := m.memberByUUID(uuid)
	if !ok || n.Name != "new" || n.Nickname != "mine" {
		t.Fatalf("member = %+v, want name new and nickname kept", n)
	}
}
