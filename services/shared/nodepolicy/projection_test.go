// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func inferenceLockPolicy() Policy {
	p := Default()
	p.Profiles = []Profile{
		{Name: "locked", Engine: "llamacpp", Model: "stored-model", LocalOnly: true, RequestOptions: json.RawMessage(`{"temperature":0.5,"nested":{"seed":9007199254740993},"stop":["a","b"]}`)},
		{Name: "editable", Engine: "lmstudio", Model: "editable-model", RequestOptions: json.RawMessage(`{"temperature":1}`)},
	}
	return p
}

func TestPeerInferenceSettingsLockRefusesWholeWrite(t *testing.T) {
	test := func(name string, change func(*Policy), wantLocked bool) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			stored := inferenceLockPolicy()
			incoming := stored
			incoming.Profiles = copyProfiles(stored.Profiles)
			incoming.Idle.UnloadAfterMinutes = 17
			change(&incoming)
			raw, err := json.Marshal(incoming.PeerProjection())
			if err != nil {
				t.Fatal(err)
			}
			merged, err := MergePeerSet(stored, raw)
			var locked *RemoteSettingLockedError
			if errors.As(err, &locked) != wantLocked {
				t.Fatalf("error=%v, want locked=%v", err, wantLocked)
			}
			if wantLocked {
				if !reflect.DeepEqual(merged, Policy{}) {
					t.Fatal("refusal returned a partially merged policy")
				}
				if strings.Contains(err.Error(), "incoming-secret") || strings.Contains(err.Error(), "temperature") || strings.Contains(err.Error(), "9007199254740993") || strings.Contains(err.Error(), "locked\"") {
					t.Fatal("lock diagnostic leaked nonidentity fields")
				}
			} else if err != nil || merged.Idle.UnloadAfterMinutes != 17 {
				t.Fatalf("allowed edit refused: %v", err)
			}
			if stored.Profiles[0].Name != "locked" || string(stored.Profiles[0].RequestOptions) != `{"temperature":0.5,"nested":{"seed":9007199254740993},"stop":["a","b"]}` {
				t.Fatal("stored policy changed")
			}
		})
	}
	test("unlocked options", func(p *Policy) { p.Profiles[1].RequestOptions = json.RawMessage(`{"top_k":7}`) }, false)
	test("display metadata", func(p *Policy) { p.Profiles[0].Name = "renamed"; p.Profiles[0].MemoryBytes = 100 }, false)
	test("reordered profiles", func(p *Policy) { p.Profiles[0], p.Profiles[1] = p.Profiles[1], p.Profiles[0] }, false)
	test("explicit default", func(p *Policy) { p.Profiles[0].RequestOptionsMode = RequestOptionsDefaults }, false)
	test("clear lock", func(p *Policy) { p.Profiles[0].LocalOnly = false }, true)
	test("set existing unlocked lock", func(p *Policy) { p.Profiles[1].LocalOnly = true }, true)
	test("delete protected identity", func(p *Policy) { p.Profiles = p.Profiles[1:] }, true)
	test("change protected model", func(p *Policy) { p.Profiles[0].Model = "incoming-secret" }, true)
	test("change protected engine", func(p *Policy) { p.Profiles[0].Engine = "lmstudio" }, true)
	test("change mode", func(p *Policy) { p.Profiles[0].RequestOptionsMode = RequestOptionsEnforced }, true)
	test("remove options", func(p *Policy) { p.Profiles[0].RequestOptions = nil }, true)
	test("add option", func(p *Policy) {
		p.Profiles[0].RequestOptions = json.RawMessage(`{"temperature":0.5,"nested":{"seed":9007199254740993},"stop":["a","b"],"top_k":2}`)
	}, true)
	test("delete option", func(p *Policy) { p.Profiles[0].RequestOptions = json.RawMessage(`{"temperature":0.5}`) }, true)
	test("precise numeric change", func(p *Policy) {
		p.Profiles[0].RequestOptions = json.RawMessage(`{"temperature":0.5,"nested":{"seed":9007199254740992},"stop":["a","b"]}`)
	}, true)
	test("array reorder", func(p *Policy) {
		p.Profiles[0].RequestOptions = json.RawMessage(`{"temperature":0.5,"nested":{"seed":9007199254740993},"stop":["b","a"]}`)
	}, true)
	test("object order whitespace equivalent numeric spelling", func(p *Policy) {
		p.Profiles[0].RequestOptions = json.RawMessage(`{ "stop":["a","b"], "nested":{"seed":9007199254740993.0}, "temperature":5e-1 }`)
	}, false)
}

func TestPeerInferenceSettingsNewLockAndInvalidIdentity(t *testing.T) {
	stored := inferenceLockPolicy()
	t.Run("new lock uses fixed message", func(t *testing.T) {
		peer := stored.PeerProjection()
		peer.Profiles = append(peer.Profiles, Profile{Name: "private-display", Engine: "llamacpp", Model: "incoming-secret", LocalOnly: true})
		raw, err := json.Marshal(peer)
		if err != nil {
			t.Fatal(err)
		}
		merged, err := MergePeerSet(stored, raw)
		var locked *RemoteSettingLockedError
		if !errors.As(err, &locked) || locked.Error() != "inference settings can only be locked on the target device" || !reflect.DeepEqual(merged, Policy{}) {
			t.Fatalf("unexpected refusal: %v", err)
		}
	})
	t.Run("duplicate identity rejected before lock comparison", func(t *testing.T) {
		peer := stored.PeerProjection()
		peer.Profiles[0].LocalOnly = false
		duplicate := peer.Profiles[0]
		duplicate.Name = "duplicate"
		peer.Profiles = append(peer.Profiles, duplicate)
		raw, err := json.Marshal(peer)
		if err != nil {
			t.Fatal(err)
		}
		_, err = MergePeerSet(stored, raw)
		var locked *RemoteSettingLockedError
		if err == nil || errors.As(err, &locked) {
			t.Fatalf("duplicate identity classified as lock: %v", err)
		}
	})
	t.Run("bounded stored identity only", func(t *testing.T) {
		err := &RemoteSettingLockedError{engine: strings.Repeat("e", 10000), model: strings.Repeat("m", 10000), stored: true}
		if len(err.Error()) > 512 {
			t.Fatalf("unbounded diagnostic bytes=%d", len(err.Error()))
		}
	})
}

func TestPeerInferenceSettingsProjectionOwnsRawOptions(t *testing.T) {
	stored := inferenceLockPolicy()
	stored.Profiles[0].RequestOptionsMode = RequestOptionsEnforced
	peer := stored.PeerProjection()
	if !peer.Profiles[0].LocalOnly || peer.Profiles[0].RequestOptionsMode != RequestOptionsEnforced {
		t.Fatal("projection lost fields")
	}
	raw, err := json.Marshal(peer)
	if err != nil {
		t.Fatal(err)
	}
	var decoded PeerPolicy
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(peer, decoded) {
		t.Fatal("peer settings roundtrip changed")
	}
	peer.Profiles[0].RequestOptions[2] = 'X'
	if stored.Profiles[0].RequestOptions[2] == 'X' {
		t.Fatal("projection aliases stored options")
	}
	merged, err := MergePeerSet(stored, raw)
	if err != nil {
		t.Fatal(err)
	}
	merged.Profiles[0].RequestOptions[2] = 'Y'
	if stored.Profiles[0].RequestOptions[2] == 'Y' {
		t.Fatal("merged options alias stored state")
	}
}

func TestPeerBufferSettingsStayOriginLocal(t *testing.T) {
	stored := Default()
	stored.Routing.Buffers = BufferLimits{CapacityMiB: 384, WaitSeconds: 0}
	raw := peerDocument(t, stored, nil)
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if _, exists := doc["routing"]; exists {
		t.Fatal("peer projection exposed origin routing")
	}
	merged, err := MergePeerSet(stored, raw)
	if err != nil || merged.Routing.Buffers != (BufferLimits{384, 0}) {
		t.Fatalf("peer round trip changed local buffers: %+v, %v", merged.Routing.Buffers, err)
	}
	for _, key := range []string{"routing", "Routing"} {
		attempt := peerDocument(t, stored, func(doc map[string]any) {
			doc[key] = map[string]any{"buffers": map[string]any{"capacityMiB": 1024, "waitSeconds": 600}}
		})
		if _, err := MergePeerSet(stored, attempt); err == nil {
			t.Fatalf("peer changed protected buffers through %s", key)
		}
	}
}

func TestServedResidentPeerRoundTrip(t *testing.T) {
	for _, name := range []string{"absent", "empty", "subset"} {
		t.Run(name, func(t *testing.T) {
			p := stagedPolicy(t)
			if name == "empty" {
				p.ServedModels = ServedModels{}
			}
			if name == "subset" {
				desired := servedResidentPolicy()
				p.ServedModels, p.ResidentModels, p.Profiles = desired.ServedModels, desired.ResidentModels, desired.Profiles
			}
			projection := p.PeerProjection()
			for range 3 {
				raw, err := json.Marshal(projection)
				if err != nil {
					t.Fatal(err)
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(raw, &fields); err != nil {
					t.Fatal(err)
				}
				if fields["routing"] != nil {
					t.Fatal("private routing exposed")
				}
				if (fields["servedModels"] != nil) != (p.ServedModels != nil) {
					t.Fatal("permission presence lost")
				}
				var next PeerPolicy
				if err := json.Unmarshal(raw, &next); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(projection, next) {
					t.Fatal("unstable peer projection")
				}
				merged, err := MergePeerSet(p, raw)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(merged.ServedModels, p.ServedModels) || !reflect.DeepEqual(merged.ResidentModels, p.ResidentModels) || !reflect.DeepEqual(merged.Routing, p.Routing) {
					t.Fatal("merge lost desired fields or private routing")
				}
				projection = next
			}
		})
	}
}

func TestServedResidentPeerOwnedSlices(t *testing.T) {
	p := servedResidentPolicy()
	peer := p.PeerProjection()
	p.ServedModels[0].Model = "caller change"
	p.ResidentModels[0].Model = "caller change"
	if peer.ServedModels[0].Model != "m" || peer.ResidentModels[0].Model != "m" {
		t.Fatal("projection aliases caller slices")
	}
	peer.ServedModels[0].Model = "projection change"
	peer.ResidentModels[0].Model = "projection change"
	if p.ServedModels[0].Model != "caller change" || p.ResidentModels[0].Model != "caller change" {
		t.Fatal("caller slices alias projection")
	}
}

func TestServedResidentPeerRawKeys(t *testing.T) {
	for _, raw := range []string{
		`{"servedModels":[],"servedModels":[]}`, `{"servedModels":[],"ServedModels":[]}`,
		`{"residentModels":null}`, `{"servedModels":null}`,
		`{"residentModels":[{"engine":"ollama","engine":"ollama","model":"m"}]}`,
	} {
		var peer PeerPolicy
		if err := json.Unmarshal([]byte(raw), &peer); err == nil {
			t.Fatalf("peer accepted %s", raw)
		}
		if _, err := MergePeerSet(Default(), []byte(raw)); err == nil {
			t.Fatalf("merge accepted %s", raw)
		}
	}
}

func TestServedResidentRoleAuthority(t *testing.T) {
	p := servedResidentPolicy()
	p.ServedModels = ServedModels{}
	p.ResidentModels = nil
	p.Roles = map[string]Role{TierStrong: {Engine: "ollama", Model: "outside"}}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(p.PeerProjection())
	if err != nil {
		t.Fatal(err)
	}
	merged, err := MergePeerSet(p, raw)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(merged.Roles, p.Roles) || !reflect.DeepEqual(p.PeerProjection().Roles, p.Roles) {
		t.Fatal("roles rewritten outside served permission")
	}
	adv := p.Advertisement()
	if len(adv.Roles) != 1 || adv.Roles[TierStrong].Model != "outside" {
		t.Fatal("advertisement lost declaring role")
	}
	if _, ok := adv.Roles[TierWeak]; ok {
		t.Fatal("omitted role invented")
	}
}

// peerDocument is the JSON a peer sends back: the stored policy's projection
// with edit applied to its top-level keys.
func peerDocument(t *testing.T, stored Policy, edit func(doc map[string]any)) []byte {
	t.Helper()
	data, err := json.Marshal(stored.PeerProjection())
	if err != nil {
		t.Fatalf("marshal projection: %v", err)
	}
	var doc map[string]any
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("decode projection: %v", err)
	}
	if edit != nil {
		edit(doc)
	}
	out, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal peer document: %v", err)
	}
	return out
}

func TestPeerProjectionOmitsRoutingAndAPIEntries(t *testing.T) {
	stored := stagedPolicy(t)
	data, err := json.Marshal(stored.PeerProjection())
	if err != nil {
		t.Fatalf("marshal projection: %v", err)
	}

	var doc map[string]json.RawMessage
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("decode projection: %v", err)
	}
	if _, ok := doc["routing"]; ok {
		t.Errorf("projection has a routing key: %s", data)
	}
	for _, leak := range []string{
		"routing", "NVPAIR_PROVIDER_SECRET", "free-acct", "paid-acct", "free-small", "paid-big",
		"deployment", `"api"`, "9223372036854775807", "9007199254740993", "allowedOrigins",
	} {
		if strings.Contains(string(data), leak) {
			t.Errorf("projection contains %q: %s", leak, data)
		}
	}

	var got PeerPolicy
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("decode projection: %v", err)
	}
	wantTiers := map[string][]TierEntry{"weak": {{Kind: EntryLocal, Engine: "ollama", Model: "qwen3:8b"}}}
	if !reflect.DeepEqual(got.Tiers, wantTiers) {
		t.Errorf("projected tiers = %+v, want %+v (local entries only, no API-only tier)", got.Tiers, wantTiers)
	}
	if got.Revision != "12" {
		t.Errorf("projected revision = %q, want the source document's %q", got.Revision, "12")
	}
	if got.TierPolicy != stored.TierPolicy || got.Idle != stored.Idle || got.Admission != stored.Admission {
		t.Errorf("projection changed device fields or tierPolicy: %+v", got)
	}
}

func TestMergePeerSetAcceptsUnchangedProjection(t *testing.T) {
	test := func(name string, edit func(doc map[string]any), want func(*Policy)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			stored := stagedPolicy(t)
			merged, err := MergePeerSet(stored, peerDocument(t, stored, edit))
			if err != nil {
				t.Fatalf("MergePeerSet: %v", err)
			}
			wantPolicy := stagedPolicy(t)
			want(&wantPolicy)
			if !reflect.DeepEqual(merged, wantPolicy) {
				t.Errorf("merged policy differs from the stored policy plus the device edit:\n got %+v\nwant %+v", merged, wantPolicy)
			}
			if !reflect.DeepEqual(stored, stagedPolicy(t)) {
				t.Errorf("MergePeerSet modified the stored policy")
			}
		})
	}

	test("no edit", nil, func(*Policy) {})
	test("device edit keeps the protected sections", func(doc map[string]any) {
		doc["idle"].(map[string]any)["unloadAfterMinutes"] = 7
		doc["pause"].(map[string]any)["drainTimeoutSeconds"] = 9
	}, func(p *Policy) {
		p.Idle.UnloadAfterMinutes = 7
		p.Pause.DrainTimeoutSeconds = 9
	})
	test("peer availability is ignored", func(doc map[string]any) {
		doc["availability"] = "paused"
	}, func(*Policy) {})
	test("peer revision is ignored", func(doc map[string]any) {
		doc["revision"] = "99"
	}, func(*Policy) {})
	test("explicit empty tier equals an absent tier", func(doc map[string]any) {
		doc["tiers"].(map[string]any)["medium"] = []any{}
	}, func(*Policy) {})
	test("declared roles", func(doc map[string]any) {
		doc["roles"] = map[string]any{
			"weak":   map[string]any{"engine": "ollama", "model": "qwen3:8b"},
			"strong": map[string]any{"engine": "lmstudio", "model": "big"},
		}
	}, func(p *Policy) {
		p.Roles = map[string]Role{TierWeak: {Engine: "ollama", Model: "qwen3:8b"}, TierStrong: {Engine: "lmstudio", Model: "big"}}
	})
	test("peer pausedBy is ignored", func(doc map[string]any) {
		doc["pausedBy"] = "peer"
	}, func(*Policy) {})
	test("peer pause with provenance is ignored", func(doc map[string]any) {
		doc["availability"] = "paused"
		doc["PausedBy"] = "local"
	}, func(*Policy) {})
}

func TestPeerProjectionCarriesRolesAndPausedBy(t *testing.T) {
	stored := stagedPolicy(t)
	stored.Roles = map[string]Role{TierMedium: {Engine: "ollama", Model: "qwen3:8b"}}
	stored.Availability = Paused
	stored.PausedBy = PausedLocally

	got := stored.PeerProjection()
	if !reflect.DeepEqual(got.Roles, stored.Roles) || got.PausedBy != PausedLocally {
		t.Errorf("projection roles/pausedBy = %+v/%q, want %+v/%q", got.Roles, got.PausedBy, stored.Roles, PausedLocally)
	}
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal projection: %v", err)
	}
	if !strings.Contains(string(data), `"roles":{"medium":{"engine":"ollama","model":"qwen3:8b"}}`) || !strings.Contains(string(data), `"pausedBy":"local"`) {
		t.Errorf("projection JSON lacks roles or pausedBy: %s", data)
	}
}

// A peer cannot change who paused the node: the stored availability and its
// provenance survive whatever the document says about either.
func TestMergePeerSetKeepsStoredPausedBy(t *testing.T) {
	test := func(name string, edit func(doc map[string]any)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			stored := stagedPolicy(t)
			stored.Availability = Paused
			stored.PausedBy = PausedLocally
			merged, err := MergePeerSet(stored, peerDocument(t, stored, edit))
			if err != nil {
				t.Fatalf("MergePeerSet: %v", err)
			}
			if merged.Availability != Paused || merged.PausedBy != PausedLocally {
				t.Errorf("merged availability/pausedBy = %q/%q, want the stored paused/local", merged.Availability, merged.PausedBy)
			}
		})
	}

	test("unchanged projection", nil)
	test("resumed and provenance cleared", func(doc map[string]any) {
		doc["availability"] = "available"
		delete(doc, "pausedBy")
	})
	test("provenance changed to peer", func(doc map[string]any) { doc["pausedBy"] = "peer" })
	test("provenance changed under a capitalized key", func(doc map[string]any) {
		delete(doc, "pausedBy")
		doc["PAUSEDBY"] = "peer"
	})
	test("unknown provenance", func(doc map[string]any) { doc["pausedBy"] = "remote" })
}

func TestMergePeerSetAcceptsLegacyDocumentOnDefaultTiers(t *testing.T) {
	stored := Default()
	merged, err := MergePeerSet(stored, []byte(`{"version":1,"idle":{"unloadAfterMinutes":3}}`))
	if err != nil {
		t.Fatalf("MergePeerSet: %v", err)
	}
	want := Default()
	want.Idle.UnloadAfterMinutes = 3
	want.Idle.StartOnDemand = true
	want.Idle.WakeTimeoutSeconds = 90
	if !reflect.DeepEqual(merged, want) {
		t.Errorf("merged = %+v, want %+v", merged, want)
	}
}

func TestMergePeerSetRejectsProtectedChanges(t *testing.T) {
	test := func(name string, edit func(doc map[string]any), wantErr string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			stored := stagedPolicy(t)
			_, err := MergePeerSet(stored, peerDocument(t, stored, edit))
			if err == nil {
				t.Fatalf("MergePeerSet = nil, want an error containing %q", wantErr)
			}
			if !strings.Contains(err.Error(), wantErr) {
				t.Fatalf("MergePeerSet = %q, want it to contain %q", err, wantErr)
			}
			if !reflect.DeepEqual(stored, stagedPolicy(t)) {
				t.Errorf("a rejected merge modified the stored policy")
			}
		})
	}
	localEntry := func(model string) map[string]any {
		return map[string]any{"kind": "local", "engine": "ollama", "model": model}
	}
	weak := func(doc map[string]any, entries ...any) {
		doc["tiers"].(map[string]any)["weak"] = entries
	}

	test("changed tier entry", func(doc map[string]any) { weak(doc, localEntry("llama3:8b")) }, "cannot change the tiers")
	test("added local tier entry", func(doc map[string]any) {
		weak(doc, localEntry("qwen3:8b"), localEntry("llama3:8b"))
	}, "cannot change the tiers")
	test("removed local tier entry", func(doc map[string]any) { weak(doc) }, "cannot change the tiers")
	test("added tier", func(doc map[string]any) {
		doc["tiers"].(map[string]any)["medium"] = []any{localEntry("llama3:8b")}
	}, "cannot change the tiers")
	// The incoming document is validated without the stored routing, so a peer
	// cannot tell whether a deployment id it names exists on this node.
	test("added api tier entry", func(doc map[string]any) {
		weak(doc, localEntry("qwen3:8b"), map[string]any{"kind": "api", "deployment": "free-small"})
	}, "is not defined")
	test("api entry replacing the local entry", func(doc map[string]any) {
		weak(doc, map[string]any{"kind": "api", "deployment": "paid-big"})
	}, "is not defined")
	test("changed entry capabilities", func(doc map[string]any) {
		entry := localEntry("qwen3:8b")
		entry["capabilities"] = []any{"tools"}
		weak(doc, entry)
	}, "cannot change the tiers")
	test("changed tiers under a capitalized key", func(doc map[string]any) {
		delete(doc, "tiers")
		doc["Tiers"] = map[string]any{"weak": []any{localEntry("llama3:8b")}}
	}, "cannot change the tiers")
	test("changed tier policy under an uppercase key", func(doc map[string]any) {
		delete(doc, "tierPolicy")
		doc["TIERPOLICY"] = map[string]any{"allowStronger": true, "allowWeaker": true, "preferLoaded": "tier"}
	}, "cannot change the tier policy")
	test("changed tier policy", func(doc map[string]any) {
		doc["tierPolicy"].(map[string]any)["allowWeaker"] = true
	}, "cannot change the tier policy")
	test("changed preferLoaded", func(doc map[string]any) {
		doc["tierPolicy"].(map[string]any)["preferLoaded"] = "any"
	}, "cannot change the tier policy")
	test("routing section", func(doc map[string]any) {
		doc["routing"] = map[string]any{"limits": map[string]any{"maxAttempts": 10}}
	}, "cannot change the routing configuration")
	test("empty routing section", func(doc map[string]any) { doc["routing"] = map[string]any{} }, "cannot change the routing configuration")
	test("null routing section", func(doc map[string]any) { doc["routing"] = nil }, "cannot change the routing configuration")
	test("capitalized routing key", func(doc map[string]any) {
		doc["Routing"] = map[string]any{"limits": map[string]any{"maxAttempts": 10}}
	}, "cannot change the routing configuration")
	test("uppercase routing key", func(doc map[string]any) {
		doc["ROUTING"] = map[string]any{"limits": map[string]any{"maxAttempts": 10}}
	}, "cannot change the routing configuration")
	test("uppercase null routing key", func(doc map[string]any) { doc["ROUTING"] = nil }, "cannot change the routing configuration")
	test("lowercase and mixed-case routing keys together", func(doc map[string]any) {
		doc["routing"] = map[string]any{}
		doc["rOuTiNg"] = map[string]any{}
	}, "duplicate field")
	test("new account in the routing section", func(doc map[string]any) {
		doc["routing"] = map[string]any{"accounts": []any{map[string]any{
			"id": "stolen", "credential": "NVPAIR_PROVIDER_SECRET_STOLEN", "billing": "free",
			"perRequestCeiling": "0", "dailyCeiling": "0", "monthlyCeiling": "0",
		}}}
	}, "cannot change the routing configuration")
	test("invalid device field", func(doc map[string]any) {
		doc["idle"].(map[string]any)["unloadAfterMinutes"] = -1
	}, "idle.unloadAfterMinutes")
	test("role for an unknown tier", func(doc map[string]any) {
		doc["roles"] = map[string]any{"huge": map[string]any{"engine": "ollama", "model": "m"}}
	}, `unknown tier "huge"`)
	test("role naming a reserved alias", func(doc map[string]any) {
		doc["roles"] = map[string]any{"weak": map[string]any{"engine": "ollama", "model": "strong-local"}}
	}, "reserved alias")
	test("role for an unknown engine", func(doc map[string]any) {
		doc["roles"] = map[string]any{"weak": map[string]any{"engine": "no/pe", "model": "m"}}
	}, `unknown engine "no/pe"`)
}

func TestMergePeerSetRejectsMalformedDocuments(t *testing.T) {
	test := func(name string, raw []byte, wantErr string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			if _, err := MergePeerSet(stagedPolicy(t), raw); err == nil || !strings.Contains(err.Error(), wantErr) {
				t.Fatalf("MergePeerSet = %v, want an error containing %q", err, wantErr)
			}
		})
	}
	test("empty", nil, "policy is required")
	test("not an object", []byte(`[1]`), "policy must be a JSON object")
	test("null", []byte(`null`), "policy must be a JSON object")
}

func TestMergePeerSetRejectsDuplicateFields(t *testing.T) {
	stored := stagedPolicy(t)
	unchangedTiers := `{"weak":[{"kind":"local","engine":"ollama","model":"qwen3:8b"}]}`
	changedTiers := `{"weak":[{"kind":"local","engine":"ollama","model":"llama3:8b"}]}`
	unchangedPolicy := `{"allowStronger":true,"allowWeaker":false,"preferLoaded":"tier"}`
	changedPolicy := `{"allowStronger":true,"allowWeaker":true,"preferLoaded":"tier"}`
	test := func(name, body string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			_, err := MergePeerSet(stored, []byte(`{"version":2,`+body+`}`))
			if err == nil || !strings.Contains(err.Error(), "duplicate field") {
				t.Fatalf("MergePeerSet = %v, want a duplicate field error", err)
			}
		})
	}
	test("tierPolicy unchanged then TIERPOLICY changed", `"tierPolicy":`+unchangedPolicy+`,"TIERPOLICY":`+changedPolicy)
	test("TIERPOLICY changed then tierPolicy unchanged", `"TIERPOLICY":`+changedPolicy+`,"tierPolicy":`+unchangedPolicy)
	test("tiers unchanged then Tiers changed", `"tiers":`+unchangedTiers+`,"Tiers":`+changedTiers)
	test("Tiers changed then tiers unchanged", `"Tiers":`+changedTiers+`,"tiers":`+unchangedTiers)
	test("tiers repeated exactly", `"tiers":`+changedTiers+`,"tiers":`+unchangedTiers)
	test("device field in two spellings", `"idle":{"unloadAfterMinutes":1},"IDLE":{"unloadAfterMinutes":2}`)
}

func TestMergePeerSetErrorsDoNotEchoRoutingContent(t *testing.T) {
	stored := stagedPolicy(t)
	doc := peerDocument(t, stored, func(doc map[string]any) {
		doc["routing"] = map[string]any{"accounts": []any{map[string]any{"credential": "NVPAIR_PROVIDER_SECRET_LEAKED"}}}
	})
	_, err := MergePeerSet(stored, doc)
	if err == nil {
		t.Fatal("MergePeerSet = nil, want a rejection")
	}
	if strings.Contains(err.Error(), "NVPAIR_PROVIDER_SECRET") {
		t.Errorf("error echoes a credential reference: %q", err)
	}
}

// The deployment endpoint schema lives inside routing, so a peer never sees an
// endpoint, protocol, transport bound or provider qualification.
func TestPeerProjectionOmitsDeploymentEndpointSchema(t *testing.T) {
	stored := stagedPolicy(t)
	completeDeployment(&stored.Routing.Deployments[0])
	if err := stored.Validate(); err != nil {
		t.Fatalf("Validate(complete deployment) = %v", err)
	}
	data, err := json.Marshal(stored.PeerProjection())
	if err != nil {
		t.Fatalf("marshal projection: %v", err)
	}
	for _, leak := range []string{
		"api.example.com", "https://", "openai-chat-completions", "transport", "maxResponseBytes",
		"provider-q", "qualification", "endpoint", "protocol",
	} {
		if strings.Contains(string(data), leak) {
			t.Errorf("projection contains %q: %s", leak, data)
		}
	}
	doc := peerDocument(t, stored, func(doc map[string]any) {
		doc["routing"] = map[string]any{"deployments": []any{map[string]any{"endpoint": "https://evil.example.com/v1"}}}
	})
	if _, err := MergePeerSet(stored, doc); err == nil || !strings.Contains(err.Error(), "routing") {
		t.Fatalf("MergePeerSet(routing with an endpoint) err = %v, want a routing rejection", err)
	}
}
