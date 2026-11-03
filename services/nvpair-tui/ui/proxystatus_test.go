// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"testing"

	"nvpair-tui/rpc"
)

// TestProxyErrorTakesProxyDown is the regression guard for a strip that could
// only go green. Readiness was set on :ready and never cleared, so a proxy that
// failed kept advertising its port — pointing local clients at something that
// had stopped listening, which is the one thing this strip exists to tell them.
func TestProxyErrorTakesProxyDown(t *testing.T) {
	p := newProxyTracker()
	p.handleNotification(&rpc.Message{
		Method: "llamacpp-proxy:ready",
		Params: []byte(`{"port":8080}`),
	})
	if port, ready := p.portForEngine("llamacpp"); !ready || port != 8080 {
		t.Fatalf("after ready: port=%d ready=%v", port, ready)
	}

	p.handleNotification(&rpc.Message{Method: "llamacpp-proxy:error", Params: []byte(`{}`)})
	port, ready := p.portForEngine("llamacpp")
	if ready {
		t.Error("proxy still reads ready after an error frame")
	}
	if port != 8080 {
		t.Errorf("port = %d; the configured port should survive so the strip can name "+
			"which endpoint is down", port)
	}
	if !contains(p.strip(), "down") {
		t.Errorf("strip does not report the proxy down: %s", p.strip())
	}

	// Unknown facades are not tracked.
	if _, ready := p.portForEngine("unknown"); ready {
		t.Error("an unknown facade became ready")
	}
}

// TestProxyNotificationsAreScopedByPrefix checks unknown facades are ignored.
func TestProxyNotificationsAreScopedByPrefix(t *testing.T) {
	p := newProxyTracker()
	p.handleNotification(&rpc.Message{
		Method: "unknown-proxy:ready",
		Params: []byte(`{"port":1234}`),
	})

	if port, ready := p.portForEngine("unknown"); ready || port != 0 {
		t.Errorf("unknown proxy: port=%d ready=%v, want 0/false", port, ready)
	}
	if _, ready := p.portForEngine("llamacpp"); ready {
		t.Error("an unknown-proxy frame marked the llamacpp proxy ready")
	}
}

// TestProxyPushesUseTheFacadePrefix is the regression guard for the Ollama
// proxy's pushes being dropped.
//
// The broker forwards each facade's pushes as <engine>-proxy:<method>. Routing
// still looked for the bare "proxy:" of the single-engine proxy this replaced,
// so Ollama readiness and failure only ever reached the screen through the
// periodic status poll — and a proxy that died read as up until the next one.
func TestProxyPushesUseTheFacadePrefix(t *testing.T) {
	p := newProxyTracker()
	for _, e := range p.engines {
		p.handleNotification(&rpc.Message{Method: e.prefix + ":ready", Params: []byte(`{"port":4000}`)})
		if port, ready := p.portForEngine(e.engine); !ready || port != 4000 {
			t.Errorf("%s: a ready push under %q was not applied (port=%d ready=%v)",
				e.engine, e.prefix, port, ready)
		}
	}

	p = newProxyTracker()
	p.handleNotification(&rpc.Message{Method: "proxy:ready", Params: []byte(`{"port":4000}`)})
	for _, e := range p.engines {
		if _, ready := p.portForEngine(e.engine); ready {
			t.Errorf("%s: an unaddressed push was attributed to it", e.engine)
		}
	}
}

// TestFailedStatusReadTakesTheProxyDown checks a status read that failed is not
// ignored. Ignoring it left the strip showing the last good answer — a green
// port nothing may be listening on — for as long as the reads kept failing.
func TestFailedStatusReadTakesTheProxyDown(t *testing.T) {
	p := newProxyTracker()
	p.apply(proxyStatusMsg{idx: 0, ready: true, port: 8080})
	p.apply(proxyStatusMsg{idx: 0, err: errFake{}})
	port, ready := p.portForEngine("llamacpp")
	if ready {
		t.Error("the proxy still reads ready after its status read failed")
	}
	if port != 8080 {
		t.Errorf("port = %d; the last known port should stay, shown as down", port)
	}
}

// TestNotRunningKeepsTheLastKnownPort checks the broker's {ready:false, port:0}
// for a facade it is not running does not blank the port — the same rule the
// error push already followed, so the two paths no longer disagree.
func TestNotRunningKeepsTheLastKnownPort(t *testing.T) {
	p := newProxyTracker()
	p.apply(proxyStatusMsg{idx: 0, ready: true, port: 8080})
	p.apply(proxyStatusMsg{idx: 0, ready: false, port: 0})
	if port, ready := p.portForEngine("llamacpp"); ready || port != 8080 {
		t.Errorf("port=%d ready=%v, want 8080 shown as down", port, ready)
	}
}

// TestPortForEngineDistinguishesDownFromUnknown checks a port is not treated as
// usable just because it is known: an engine with no proxy and a proxy that is
// down both have to read as unusable.
func TestPortForEngineDistinguishesDownFromUnknown(t *testing.T) {
	p := newProxyTracker()
	if _, ready := p.portForEngine("llamacpp"); ready {
		t.Error("a proxy that has never reported reads as ready")
	}
	if port, ready := p.portForEngine("vllm"); ready || port != 0 {
		t.Errorf("unknown engine: port=%d ready=%v, want 0/false", port, ready)
	}
}
