// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"testing"

	"nvpair-shared/schedulerwire"
)

// node/set-priority is the broker's own call to a proxy. A client that
// addresses it to an engine component must be refused before the generic proxy
// relay, which would strip the component address and hand the proxy a ranking
// it applies, clearing its reservations.
func TestSetPriorityIsRefusedToClients(t *testing.T) {
	for _, profile := range engineProxyProfiles {
		t.Run(profile.Name, func(t *testing.T) {
			b, out := newPolicyBroker(t)
			peer, proxy := newFakeWorkerPipe(t)
			b.setEngineProxyHandle(profile, &proxyProcess{peer: peer})

			b.handleMessage(clientRequest(1, profile.ComponentName()+":node/set-priority",
				schedulerwire.Priority{Generation: 1, Nodes: []string{"peer"}}))
			response := out.waitFrame(t, "private method refusal", isResponse(1))
			if response.Error == nil || response.Error.Code != -32601 || len(response.Result) != 0 {
				t.Fatalf("client node/set-priority was not refused: %+v", response)
			}

			// A relayed request reaches the proxy, so the wiring above can see
			// a leak; the broker's own delivery must keep working.
			b.handleMessage(clientRequest(2, profile.ComponentName()+":nodes/list", nil))
			if reply := out.waitFrame(t, "ordinary relay", isResponse(2)); reply.Error != nil {
				t.Fatalf("ordinary addressed request was refused: %+v", reply.Error)
			}
			b.sendPriority(context.Background(), schedulerwire.Priority{Generation: 2, Nodes: []string{"peer"}})

			got := proxy.methods("")
			want := []string{profile.addressed("nodes/list"), "node/set-priority"}
			if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
				t.Fatalf("proxy received %v, want %v", got, want)
			}
		})
	}
}
