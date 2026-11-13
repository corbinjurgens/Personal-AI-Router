// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
)

func TestInvalidSettingsRecordOffersLocalReset(t *testing.T) {
	refusal := &rpc.RPCError{Code: settingsRecordInvalidCode, Message: `saved settings for engine "llamacpp" are invalid (entry is empty)`}

	t.Run("peer refusal is reported without a reset", func(t *testing.T) {
		d := newNodeDetail(nil, nodeRow{key: "peer", membership: membershipMember})
		if cmd, _ := d.update(engineSettingsMsg{engine: "llamacpp", err: refusal}); cmd != nil || d.pending != nil {
			t.Fatal("a peer's invalid record armed a local reset")
		}
		if !strings.Contains(d.status.text, "invalid") {
			t.Fatalf("status = %q", d.status.text)
		}
	})

	t.Run("local refusal resets after confirmation", func(t *testing.T) {
		client, broker := nodeInfoRPCFixture(t)
		d := newNodeDetail(client, nodeRow{key: "self", self: true, presence: presenceOnline})
		if cmd, _ := d.update(engineSettingsMsg{engine: "llamacpp", err: refusal}); cmd != nil || d.pending == nil {
			t.Fatal("invalid local record did not arm a reset")
		}
		cmd := d.resolvePending(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("y")})
		request, reply := terminalControlsReply(t, broker, cmd, &rpc.Message{Result: json.RawMessage(`{"engine":"llamacpp","backup":"journal.llamacpp.invalid.json"}`)})
		var params map[string]string
		if err := json.Unmarshal(request.Params, &params); err != nil {
			t.Fatal(err)
		}
		if request.Method != "engine:reset-settings-record" || !reflect.DeepEqual(params, map[string]string{"engine": "llamacpp"}) {
			t.Fatalf("reset frame = %s %v", request.Method, params)
		}
		d.update(reply)
		if !strings.Contains(d.status.text, "reset") || !strings.Contains(d.status.text, "journal.llamacpp.invalid.json") {
			t.Fatalf("status = %q", d.status.text)
		}
	})
}
