// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

func TestModelControlsServiceStoredPolicy(t *testing.T) {
	f := newServiceFixture(t)
	f.withInertEngineManager()
	policy := nodepolicy.Default()
	policy.Revision = "304"
	policy.Admission.MaxConcurrentPerModel = 7
	policy.Profiles = []nodepolicy.Profile{
		{Name: "first", Engine: "llamacpp", Model: "first"},
		{Name: "invalid", Engine: "llamacpp", Model: "j304-invalid-placeholder"},
		{Name: "last", Engine: "llamacpp", Model: "last"},
	}
	policy.Roles["weak"] = nodepolicy.Role{Engine: "llamacpp", Model: "first"}
	policy.Tiers["weak"] = []nodepolicy.TierEntry{{Kind: nodepolicy.EntryLocal, Engine: "llamacpp", Model: "first"}}
	policy.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "first"}}
	if err := policy.Validate(); err != nil {
		t.Fatalf("validate stored positive control: %v", err)
	}
	raw := mustMarshal(t, policy)
	if bytes.Count(raw, []byte(`"j304-invalid-placeholder"`)) != 1 {
		t.Fatal("stored fixture does not contain exactly its intended raw model token")
	}
	raw = bytes.Replace(raw, []byte(`"j304-invalid-placeholder"`), []byte("\"raw-"+string([]byte{0xff})+"\""), 1)
	path := f.policyPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	f.start()
	c := f.readyClient()
	want := policy
	want.Profiles = []nodepolicy.Profile{policy.Profiles[0], policy.Profiles[2]}
	assertPolicySnapshot(t, fetchServicePolicy(t, c), want, nodepolicy.Available)
	if !bytes.Equal(readPolicyBytes(t, path), raw) {
		t.Fatal("load-only sanitation rewrote stored bytes")
	}
	if _, err := os.Stat(path + ".bad"); !os.IsNotExist(err) {
		t.Fatalf("mixed stored model inventory entered malformed-document recovery: %v", err)
	}
	// The service's own capped output is not broker stderr. Read the private
	// pumpStderr-owned file through the broker's post-sanitation log marker;
	// an ordinary policy RPC does not drain this independent stream.
	logPath := filepath.Join(f.dir, "logs", "broker.log")
	spillPath := filepath.Join(filepath.Dir(path), "logs", "nvpair-broker-unsent.log")
	readSanitationLog := func() []byte {
		t.Helper()
		const maxCapture = 4 << 20
		const maxBrokerLine = 1 << 20 // pumpStderr silently truncates at this bound.
		deadline := time.Now().Add(serviceCallTimeout)
		var first os.FileInfo
		var largest int64
		for {
			for _, forbidden := range []string{logPath + ".1", spillPath} {
				if _, err := os.Stat(forbidden); err == nil {
					t.Fatalf("privacy capture cannot accept rotation or overflow: %s", forbidden)
				} else if !os.IsNotExist(err) {
					t.Fatalf("inspect privacy capture generation: %v", err)
				}
			}
			file, err := os.Open(logPath)
			if os.IsNotExist(err) {
				if time.Now().After(deadline) {
					t.Fatal("broker stderr file never appeared")
				}
				time.Sleep(20 * time.Millisecond)
				continue
			}
			if err != nil {
				t.Fatalf("open private broker stderr: %v", err)
			}
			info, statErr := file.Stat()
			data, readErr := io.ReadAll(io.LimitReader(file, maxCapture+1))
			closeErr := file.Close()
			if statErr != nil || readErr != nil || closeErr != nil {
				t.Fatalf("read private broker stderr: stat=%v read=%v close=%v", statErr, readErr, closeErr)
			}
			current, err := os.Stat(logPath)
			if err != nil || !os.SameFile(info, current) || (first != nil && !os.SameFile(first, info)) || info.Size() < largest {
				t.Fatal("broker stderr disappeared, rotated or was truncated during capture")
			}
			first, largest = info, info.Size()
			if !info.Mode().IsRegular() || info.Size() > maxCapture || len(data) > maxCapture || int64(len(data)) < info.Size() {
				t.Fatal("broker stderr capture is incomplete or exceeds its explicit bound")
			}
			for _, line := range bytes.Split(data, []byte{'\n'}) {
				if len(line) >= maxBrokerLine || bytes.Contains(line, []byte("log reader fell behind")) {
					t.Fatal("broker stderr reported overflow or reached the pump's truncation bound")
				}
			}
			complete := data[:bytes.LastIndexByte(data, '\n')+1]
			marker := []byte("[nvpair-ui-broker] INFO node policy loaded ")
			for _, line := range bytes.Split(complete, []byte{'\n'}) {
				if bytes.Contains(line, marker) && bytes.Contains(line, []byte(" revision=304 ")) && bytes.Contains(line, []byte("recoveryRequired=false")) {
					// FIFO stderr writes put the sanitation records before this
					// same-owner marker. Fail closed if that positive is absent.
					if !bytes.Contains(complete, []byte("[nvpair-ui-broker] WARN invalid stored model entries removed category=profiles count=1")) {
						t.Fatal("broker sanitation marker missing from captured stderr")
					}
					for _, forbidden := range []string{logPath + ".1", spillPath} {
						if _, err := os.Stat(forbidden); !os.IsNotExist(err) {
							t.Fatalf("capture generation/overflow check failed: %s: %v", forbidden, err)
						}
					}
					return complete
				}
			}
			if time.Now().After(deadline) {
				t.Fatal("broker post-sanitation stderr barrier never arrived")
			}
			time.Sleep(20 * time.Millisecond)
		}
	}
	if bytes.Contains(readSanitationLog(), []byte("raw-")) {
		t.Fatal("stored model token appeared in broker diagnostics")
	}
}

func TestModelControlsServicePolicyWrites(t *testing.T) {
	f := newServiceFixture(t)
	f.withInertEngineManager()
	f.start()
	c := f.readyClient()
	observer := f.readyClient()
	policy := servicePolicy(t, c)
	policy.Profiles = []nodepolicy.Profile{{Name: "valid", Engine: "llamacpp", Model: strings.Repeat("a", 512)}}
	policy = setServicePolicy(t, c, policy)
	before := fetchServicePolicy(t, c)
	stored := assertStoredPolicy(t, f.policyPath(), policy)
	// A second client's get is an ordered barrier after the valid publication.
	assertPolicySnapshot(t, fetchServicePolicy(t, observer), policy, nodepolicy.Available)
	observer.mu.Lock()
	observer.frames = nil
	observer.bytes = 0
	observer.mu.Unlock()
	cases := []struct{ name, token string }{
		{"empty", `""`},
		{"raw invalid UTF8", "\"raw-" + string([]byte{0xff}) + "\""},
		{"escaped C0", `"bad\u001f"`},
		{"DEL", `"bad\u007f"`},
		{"C1 start", `"bad\u0080"`},
		{"C1 end", `"bad\u009f"`},
		{"ASCII 513 bytes", `"` + strings.Repeat("b", 513) + `"`},
		{"multibyte 513 bytes", `"` + strings.Repeat("界", 171) + `"`},
	}
	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			draft := policy
			draft.Profiles = append(append([]nodepolicy.Profile{}, policy.Profiles...), nodepolicy.Profile{Name: "rejected", Engine: "llamacpp", Model: "j304-invalid-placeholder"})
			params := mustMarshal(t, map[string]any{"policy": draft, "expectedPersistedRevision": policy.Revision})
			if bytes.Count(params, []byte(`"j304-invalid-placeholder"`)) != 1 {
				t.Fatal("write fixture does not contain exactly its intended raw model token")
			}
			params = bytes.Replace(params, []byte(`"j304-invalid-placeholder"`), []byte(tc.token), 1)
			m, err := c.call(strconv.Itoa(30400+i), "policy:set", params, serviceCallTimeout)
			if err != nil {
				t.Fatalf("invalid write transport: %v", err)
			}
			if m.Error == nil || m.Error.Code != -32602 || len(m.Result) != 0 {
				t.Fatal("invalid new model write was accepted")
			}
			if strings.Contains(m.Error.Message, "raw-") || strings.Contains(m.Error.Message, "bad") || strings.Contains(m.Error.Message, strings.Repeat("b", 513)) || strings.Contains(m.Error.Message, strings.Repeat("界", 171)) {
				t.Fatal("invalid write echoed model text")
			}
			if !reflect.DeepEqual(fetchServicePolicy(t, c), before) || !bytes.Equal(readPolicyBytes(t, f.policyPath()), stored) {
				t.Fatal("invalid write changed policy, revision, availability or persisted bytes")
			}
			assertPolicySnapshot(t, fetchServicePolicy(t, observer), policy, nodepolicy.Available)
			observer.mu.Lock()
			frames := append([]*jsonrpc.Message{}, observer.frames...)
			observer.mu.Unlock()
			for _, frame := range frames {
				if frame.Method == "policy:changed" {
					var event struct {
						Policy nodepolicy.Policy `json:"policy"`
					}
					if err := json.Unmarshal(frame.Params, &event); err != nil {
						t.Fatalf("decode policy publication: %v", err)
					}
					if !reflect.DeepEqual(event.Policy, policy) {
						t.Fatal("invalid draft reached policy publication")
					}
				}
			}
		})
	}
}

func TestServicePolicy(t *testing.T) {
	t.Run("valid update notifies and survives service restart", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		command := f.readyClient()
		observer := f.readyClient()
		nodeID := fixturePolicyNodeID(t, f)
		want := servicePolicy(t, command)
		if want.Admission.MaxConcurrentPerModel == 7 {
			t.Fatal("fresh policy already has the update's concurrency value")
		}
		want.Admission.MaxConcurrentPerModel = 7
		want = setServicePolicy(t, command, want)
		assertPolicyChanged(t, observer, nodeID, want)
		assertPolicySnapshot(t, fetchServicePolicy(t, command), want, nodepolicy.Available)
		path := f.policyPath()
		assertStoredPolicy(t, path, want)
		restartPolicyFixture(t, f)
		fresh := f.readyClient()
		assertPolicySnapshot(t, fetchServicePolicy(t, fresh), want, nodepolicy.Available)
		assertStoredPolicy(t, path, want)
	})

	t.Run("invalid complete policy preserves snapshot and stored bytes", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		c := f.readyClient()
		want := servicePolicy(t, c)
		want.Admission.MaxConcurrentPerModel = 5
		want = setServicePolicy(t, c, want)
		before := fetchServicePolicy(t, c)
		assertPolicySnapshot(t, before, want, nodepolicy.Available)
		path := f.policyPath()
		stored := assertStoredPolicy(t, path, want)
		invalid := before.Policy
		invalid.TierPolicy.PreferLoaded = "invalid"
		m := c.mustPolicyCall(t, "202", invalid)
		if m.Error == nil || m.Error.Code != -32602 || m.Error.Message != `tierPolicy.preferLoaded must be "tier" or "any"` || len(m.Result) != 0 {
			t.Fatalf("invalid preferLoaded response: error=%+v result_bytes=%d", m.Error, len(m.Result))
		}
		after := fetchServicePolicy(t, c)
		if !reflect.DeepEqual(after, before) {
			t.Fatal("rejected policy changed the complete policy/live snapshot")
		}
		current := readPolicyBytes(t, path)
		if !bytes.Equal(current, stored) {
			t.Fatal("rejected policy changed persisted bytes")
		}
	})

	t.Run("stale revision conflicts and leaves the stored policy as it was", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		a := f.readyClient()
		b := f.readyClient()
		draft := servicePolicy(t, b)
		next := servicePolicy(t, a)
		next.Admission.MaxConcurrentPerModel = 6
		next = setServicePolicy(t, a, next)
		path := f.policyPath()
		stored := assertStoredPolicy(t, path, next)
		before := fetchServicePolicy(t, a)

		draft.Admission.MaxConcurrentPerModel = 8
		m := b.mustPolicyCall(t, "204", draft)
		wantMessage := "policy conflict: the persisted revision is " + next.Revision + ", not the expected " + draft.Revision + "; reload the policy and review it before saving"
		if m.Error == nil || m.Error.Code != -32010 || m.Error.Message != wantMessage || len(m.Result) != 0 {
			t.Fatalf("stale policy:set response: error=%+v result_bytes=%d, want code -32010 and %q", m.Error, len(m.Result), wantMessage)
		}
		if !reflect.DeepEqual(fetchServicePolicy(t, b), before) {
			t.Fatal("conflicting policy:set changed the policy/live snapshot")
		}
		if !bytes.Equal(readPolicyBytes(t, path), stored) {
			t.Fatal("conflicting policy:set changed persisted bytes")
		}
		// The reloaded revision is accepted: a conflict is resolved by review, not retried as is.
		reloaded := servicePolicy(t, b)
		reloaded.Admission.MaxConcurrentPerModel = 8
		setServicePolicy(t, b, reloaded)
	})

	t.Run("missing precondition is invalid params and leaves the stored policy as it was", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		c := f.readyClient()
		// A fresh node has no stored file yet; the first set writes it.
		first := servicePolicy(t, c)
		first.Admission.MaxConcurrentPerModel = 3
		setServicePolicy(t, c, first)
		before := fetchServicePolicy(t, c)
		path := f.policyPath()
		stored := readPolicyBytes(t, path)
		params, err := json.Marshal(struct {
			Policy nodepolicy.Policy `json:"policy"`
		}{Policy: before.Policy})
		if err != nil {
			t.Fatal(err)
		}
		m, err := c.call("205", "policy:set", params, serviceCallTimeout)
		if err != nil {
			t.Fatalf("policy:set transport: %v", err)
		}
		if m.Error == nil || m.Error.Code != -32602 || !strings.Contains(m.Error.Message, "expectedPersistedRevision") || len(m.Result) != 0 {
			t.Fatalf("policy:set without a precondition: error=%+v result_bytes=%d", m.Error, len(m.Result))
		}
		if !reflect.DeepEqual(fetchServicePolicy(t, c), before) || !bytes.Equal(readPolicyBytes(t, path), stored) {
			t.Fatal("refused policy:set changed the snapshot or persisted bytes")
		}
	})

	t.Run("a real proxy activates the persisted revision after set, pause and resume", func(t *testing.T) {
		f := newProxyServiceFixture(t)
		f.withInertEngineManager()
		f.start()
		c := f.readyClient()
		assertActive := func(step string) servicePolicySnapshot {
			t.Helper()
			deadline := time.Now().Add(serviceCallTimeout)
			for {
				got := fetchServicePolicy(t, c)
				if got.Activation == "active" && got.ActiveRevision == got.Policy.Revision {
					return got
				}
				if time.Now().After(deadline) {
					t.Fatalf("after %s the proxy never activated the persisted revision: revision=%q activation=%q activeRevision=%q", step, got.Policy.Revision, got.Activation, got.ActiveRevision)
				}
				time.Sleep(50 * time.Millisecond)
			}
		}
		assertActive("startup")

		want := servicePolicy(t, c)
		want.Admission.MaxConcurrentPerModel = 4
		started := time.Now().UnixNano()
		m := c.mustPolicyCall(t, "206", want)
		finished := time.Now().UnixNano()
		if m.Error != nil {
			t.Fatalf("policy:set RPC error: %+v", m.Error)
		}
		var set struct {
			Policy         nodepolicy.Policy `json:"policy"`
			Activation     string            `json:"activation"`
			ActiveRevision string            `json:"activeRevision"`
		}
		if err := json.Unmarshal(m.Result, &set); err != nil {
			t.Fatalf("decode policy:set: %v", err)
		}
		assertNextRevision(t, want.Revision, set.Policy.Revision, started, finished)
		if set.Activation != "active" || set.ActiveRevision != set.Policy.Revision {
			t.Fatalf("policy:set result activation=%q activeRevision=%q, want active at %q", set.Activation, set.ActiveRevision, set.Policy.Revision)
		}
		afterSet := assertActive("policy:set")
		if afterSet.Policy.Revision != set.Policy.Revision {
			t.Fatalf("policy:get revision %q, want the set's %q", afterSet.Policy.Revision, set.Policy.Revision)
		}

		setServiceAvailability(t, c, nodepolicy.Paused)
		paused := assertActive("pause")
		if paused.Availability != nodepolicy.Paused || paused.Policy.Revision == afterSet.Policy.Revision {
			t.Fatalf("pause: availability=%q revision=%q, want paused with a new revision after %q", paused.Availability, paused.Policy.Revision, afterSet.Policy.Revision)
		}
		setServiceAvailability(t, c, nodepolicy.Available)
		resumed := assertActive("resume")
		if resumed.Availability != nodepolicy.Available || resumed.Policy.Revision == paused.Policy.Revision {
			t.Fatalf("resume: availability=%q revision=%q, want available with a new revision after %q", resumed.Availability, resumed.Policy.Revision, paused.Policy.Revision)
		}
	})

	t.Run("pause notifies observer and persists before restart and resume", func(t *testing.T) {
		f := newServiceFixture(t)
		f.withInertEngineManager()
		f.start()
		command := f.readyClient()
		observer := f.readyClient()
		nodeID := fixturePolicyNodeID(t, f)
		want := servicePolicy(t, command)
		assertPolicySnapshot(t, fetchServicePolicy(t, command), want, nodepolicy.Available)
		started := time.Now().UnixNano()
		setServiceAvailability(t, command, nodepolicy.Paused)
		finished := time.Now().UnixNano()
		paused := fetchServicePolicy(t, command)
		assertNextRevision(t, want.Revision, paused.Policy.Revision, started, finished)
		want.Revision = paused.Policy.Revision
		want.Availability = nodepolicy.Paused
		want.PausedBy = nodepolicy.PausedLocally
		assertPolicyChanged(t, observer, nodeID, want)
		assertAvailabilityChanged(t, observer, nodeID, nodepolicy.Paused)
		assertPolicySnapshot(t, fetchServicePolicy(t, command), want, nodepolicy.Paused)
		path := f.policyPath()
		assertStoredPolicy(t, path, want)
		restartPolicyFixture(t, f)
		command = f.readyClient()
		observer = f.readyClient()
		if got := fixturePolicyNodeID(t, f); got != nodeID {
			t.Fatalf("restart changed fixture node identity: got %q, want %q", got, nodeID)
		}
		assertPolicySnapshot(t, fetchServicePolicy(t, command), want, nodepolicy.Paused)
		started = time.Now().UnixNano()
		setServiceAvailability(t, command, nodepolicy.Available)
		finished = time.Now().UnixNano()
		resumed := fetchServicePolicy(t, command)
		assertNextRevision(t, want.Revision, resumed.Policy.Revision, started, finished)
		want.Revision = resumed.Policy.Revision
		want.Availability = nodepolicy.Available
		want.PausedBy = ""
		assertPolicyChanged(t, observer, nodeID, want)
		assertAvailabilityChanged(t, observer, nodeID, nodepolicy.Available)
		assertPolicySnapshot(t, fetchServicePolicy(t, command), want, nodepolicy.Available)
		assertStoredPolicy(t, path, want)
	})

	t.Run("CLI call transports complete policy from private endpoint", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		c := f.readyClient()
		want := servicePolicy(t, c)
		want.Admission.MaxConcurrentPerModel = 9
		want = setServicePolicy(t, c, want)
		output, code := f.cli(serviceCallTimeout, "call", "policy:get", "{}")
		if code != 0 {
			t.Fatalf("CLI call policy:get exit=%d output=%s", code, output)
		}
		snapshot := decodeServicePolicySnapshot(t, &jsonrpc.Message{Result: json.RawMessage(output)})
		assertPolicySnapshot(t, snapshot, want, nodepolicy.Available)
		assertPolicySnapshot(t, fetchServicePolicy(t, c), want, nodepolicy.Available)
	})

	t.Run("client detach leaves updated policy on surviving connection", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		command := f.readyClient()
		observer := f.readyClient()
		before := serviceStatus(t, observer)
		want := servicePolicy(t, command)
		want.Admission.MaxConcurrentPerModel = 11
		want = setServicePolicy(t, command, want)
		assertPolicyChanged(t, observer, fixturePolicyNodeID(t, f), want)
		assertPolicySnapshot(t, fetchServicePolicy(t, observer), want, nodepolicy.Available)
		command.mustClose(t)
		after := waitServiceClientCount(t, observer, 1)
		assertServiceLifetime(t, before, after)
		assertPolicySnapshot(t, fetchServicePolicy(t, observer), want, nodepolicy.Available)
		assertStoredPolicy(t, f.policyPath(), want)
	})
}

func TestServicePolicyModelConfiguration(t *testing.T) {
	f := newServiceFixture(t)
	// No manager is intentional: configuration must retain unknown loaded
	// inventory. Start paused without requesting physical shutdown proof.
	initial := nodepolicy.Default()
	initial.Availability = nodepolicy.Paused
	initial.PausedBy = nodepolicy.PausedLocally
	if err := initial.Validate(); err != nil {
		t.Fatalf("invalid paused startup policy: %v", err)
	}
	jointWriteJSON(t, f.policyPath(), initial)
	f.start()
	command := f.readyClient()
	assertPolicySnapshot(t, fetchServicePolicy(t, command), initial, nodepolicy.Paused)
	observer := f.readyClient()
	want := servicePolicy(t, command)
	want.Profiles = append(want.Profiles, fixtureResidentProfile())
	want.ServedModels = nodepolicy.ServedModels{{Engine: "llamacpp", Model: "sr-status-fixture"}}
	want.ResidentModels = nodepolicy.ResidentModels{{Engine: "llamacpp", Model: "sr-status-fixture"}}
	result := command.mustPolicyCall(t, "241", want)
	var accepted struct {
		Policy nodepolicy.Policy `json:"policy"`
	}
	if result.Error != nil {
		t.Fatalf("save paused resident declaration: %+v", result.Error)
	}
	if err := json.Unmarshal(result.Result, &accepted); err != nil {
		t.Fatalf("decode paused resident declaration: %v", err)
	}
	want.Revision = accepted.Policy.Revision
	if !reflect.DeepEqual(accepted.Policy, want) {
		t.Fatal("policy:set changed the paused resident declaration")
	}
	assertFixtureResidentConfiguration(t, result.Result, want.Revision)
	assertFixtureUnknownLoaded(t, result.Result, want.Revision)
	readback := command.mustCall(t, "242", "policy:get", nil)
	assertFixtureResidentConfiguration(t, readback.Result, want.Revision)
	assertFixtureUnknownLoaded(t, readback.Result, want.Revision)
	event := fixturePolicyConfigurationChange(t, observer, want.Revision)
	assertFixtureResidentConfiguration(t, event, want.Revision)
	assertFixtureUnknownLoaded(t, event, want.Revision)
	assertStoredPolicy(t, f.policyPath(), want)

	// Removing the preference does not turn unknown inventory into loaded evidence.
	want.ServedModels = nodepolicy.ServedModels{}
	want.ResidentModels = nil
	result = command.mustPolicyCall(t, "243", want)
	if result.Error != nil {
		t.Fatalf("remove resident declaration: %+v", result.Error)
	}
	// Policy.UnmarshalJSON keeps omitted fields, so decode into a fresh value.
	accepted.Policy = nodepolicy.Policy{}
	if err := json.Unmarshal(result.Result, &accepted); err != nil {
		t.Fatalf("decode removed resident declaration: %v", err)
	}
	want.Revision = accepted.Policy.Revision
	if !reflect.DeepEqual(accepted.Policy, want) {
		t.Fatal("policy:set lost the explicit empty served declaration")
	}
	configuration := decodeFixtureModelConfiguration(t, result.Result, want.Revision)
	if len(configuration.Models) != 0 {
		t.Fatalf("explicit empty served set retained %d configuration entries", len(configuration.Models))
	}
	readback = command.mustCall(t, "244", "policy:get", nil)
	configuration = decodeFixtureModelConfiguration(t, readback.Result, want.Revision)
	if len(configuration.Models) != 0 {
		t.Fatal("policy:get retained removed model configuration")
	}
	event = fixturePolicyConfigurationChange(t, observer, want.Revision)
	configuration = decodeFixtureModelConfiguration(t, event, want.Revision)
	if len(configuration.Models) != 0 {
		t.Fatal("policy:changed retained removed model configuration")
	}
	assertStoredPolicy(t, f.policyPath(), want)
}

// This profile is a declaration only; it supplies no live qualification or
// loaded/effective-envelope evidence and must never produce satisfied status.
func fixtureResidentProfile() nodepolicy.Profile {
	return nodepolicy.Profile{
		Name: "sr-status", Engine: "llamacpp", Model: "sr-status-fixture",
		ContextTokens: 8192, MaxOutputTokens: 1024, MaxConcurrent: 1, MemoryBytes: 1,
		Qualification: &nodepolicy.QualificationRef{ID: "fixture-only-sr-status", Revision: "1"},
	}
}

func decodeFixtureModelConfiguration(t *testing.T, raw []byte, revision string) nodepolicy.ModelConfiguration {
	t.Helper()
	var envelope struct {
		ModelConfiguration *nodepolicy.ModelConfiguration `json:"modelConfiguration"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		t.Fatalf("decode model configuration: %v", err)
	}
	if envelope.ModelConfiguration == nil {
		t.Fatal("current broker omitted modelConfiguration")
	}
	configuration := *envelope.ModelConfiguration
	if err := configuration.Validate(); err != nil {
		t.Fatalf("invalid model configuration: %v", err)
	}
	if configuration.PolicyRevision != revision {
		t.Fatalf("model configuration revision=%q, want %q", configuration.PolicyRevision, revision)
	}
	return configuration
}

func assertFixtureResidentConfiguration(t *testing.T, raw []byte, revision string) {
	t.Helper()
	configuration := decodeFixtureModelConfiguration(t, raw, revision)
	if len(configuration.Models) != 1 {
		t.Fatalf("model configuration has %d entries, want one desired reference", len(configuration.Models))
	}
	entry := configuration.Models[0]
	if entry.Engine != "llamacpp" || entry.Model != "sr-status-fixture" || !entry.DesiredResident {
		t.Fatalf("configuration lost desired resident reference: %+v", entry)
	}
	if entry.ObservedLoaded != nil && *entry.ObservedLoaded || entry.State == nodepolicy.ModelStateSatisfied {
		t.Fatalf("absent engine supplied loaded evidence or satisfaction: %+v", entry)
	}
}

func assertFixtureUnknownLoaded(t *testing.T, raw []byte, revision string) {
	t.Helper()
	configuration := decodeFixtureModelConfiguration(t, raw, revision)
	for _, entry := range configuration.Models {
		if entry.ObservedLoaded != nil {
			t.Fatal("fixture without an engine-manager published known loaded inventory")
		}
	}
}

func fixturePolicyConfigurationChange(t *testing.T, c *serviceRPCClient, revision string) []byte {
	t.Helper()
	var decodeErr error
	m, err := c.next(serviceCallTimeout, func(m *jsonrpc.Message) bool {
		if !m.IsNotification() || m.Method != "policy:changed" {
			return false
		}
		var event struct {
			Policy nodepolicy.Policy `json:"policy"`
		}
		decodeErr = json.Unmarshal(m.Params, &event)
		return decodeErr != nil || event.Policy.Revision == revision
	})
	if err != nil {
		t.Fatalf("wait for policy configuration revision %q: %v", revision, err)
	}
	if decodeErr != nil {
		t.Fatalf("decode policy configuration notification: %v", decodeErr)
	}
	return m.Params
}

func fetchServicePolicy(t *testing.T, c *serviceRPCClient) servicePolicySnapshot {
	t.Helper()
	return decodeServicePolicySnapshot(t, c.mustCall(t, "201", "policy:get", nil))
}

// mustPolicyCall sends policy based on its own revision, which callers take
// from the preceding policy:get.
func (c *serviceRPCClient) mustPolicyCall(t *testing.T, id string, policy nodepolicy.Policy) *jsonrpc.Message {
	t.Helper()
	return c.mustPolicyCallBased(t, id, policy, policy.Revision)
}

func (c *serviceRPCClient) mustPolicyCallBased(t *testing.T, id string, policy nodepolicy.Policy, expected string) *jsonrpc.Message {
	t.Helper()
	params, err := json.Marshal(struct {
		Policy                    nodepolicy.Policy `json:"policy"`
		ExpectedPersistedRevision string            `json:"expectedPersistedRevision"`
	}{Policy: policy, ExpectedPersistedRevision: expected})
	if err != nil {
		t.Fatalf("encode complete policy: %v", err)
	}
	m, err := c.call(id, "policy:set", params, serviceCallTimeout)
	if err != nil {
		t.Fatalf("policy:set transport: %v", err)
	}
	return m
}

// setServicePolicy sends want and returns it stamped with the revision the
// broker assigned. The broker ignores the revision in want, so the result must
// be the next one after it.
func setServicePolicy(t *testing.T, c *serviceRPCClient, want nodepolicy.Policy) nodepolicy.Policy {
	t.Helper()
	if err := want.Validate(); err != nil {
		t.Fatalf("invalid update fixture: %v", err)
	}
	started := time.Now().UnixNano()
	m := c.mustPolicyCall(t, "202", want)
	finished := time.Now().UnixNano()
	if m.Error != nil {
		t.Fatalf("valid policy:set RPC error: %+v", m.Error)
	}
	var result struct {
		Policy nodepolicy.Policy `json:"policy"`
	}
	if err := json.Unmarshal(m.Result, &result); err != nil {
		t.Fatalf("decode policy:set: %v", err)
	}
	assertNextRevision(t, want.Revision, result.Policy.Revision, started, finished)
	stamped := want
	stamped.Revision = result.Policy.Revision
	if !reflect.DeepEqual(result.Policy, stamped) {
		t.Fatal("policy:set returned a different complete policy")
	}
	return stamped
}

// assertNextRevision requires got to be the revision the broker assigns after
// previous: the larger of previous+1 and the broker's clock in Unix
// nanoseconds, which was read between started and finished.
func assertNextRevision(t *testing.T, previous, got string, started, finished int64) {
	t.Helper()
	prev, err := strconv.ParseInt(previous, 10, 64)
	if err != nil {
		t.Fatalf("previous revision %q: %v", previous, err)
	}
	rev, err := strconv.ParseInt(got, 10, 64)
	if err != nil || strconv.FormatInt(rev, 10) != got {
		t.Fatalf("revision %q is not a canonical decimal: %v", got, err)
	}
	if lowest, highest := max(prev+1, started), max(prev+1, finished); rev < lowest || rev > highest {
		t.Fatalf("revision %d after %d is outside the next-revision range [%d, %d]", rev, prev, lowest, highest)
	}
}

func assertPolicySnapshot(t *testing.T, got servicePolicySnapshot, want nodepolicy.Policy, live nodepolicy.Availability) {
	t.Helper()
	if !reflect.DeepEqual(got.Policy, want) || got.Availability != live {
		t.Fatalf("policy snapshot differs: policy_equal=%t live=%q want_live=%q", reflect.DeepEqual(got.Policy, want), got.Availability, live)
	}
}

func assertPolicyChanged(t *testing.T, c *serviceRPCClient, nodeID string, want nodepolicy.Policy) {
	t.Helper()
	m := c.notification(t, "policy:changed")
	var event struct {
		NodeID string            `json:"nodeId"`
		Policy nodepolicy.Policy `json:"policy"`
	}
	if err := json.Unmarshal(m.Params, &event); err != nil {
		t.Fatalf("decode policy:changed: %v", err)
	}
	if event.NodeID != nodeID || !reflect.DeepEqual(event.Policy, want) {
		t.Fatalf("policy:changed differs: node_id=%q want_node_id=%q policy_equal=%t", event.NodeID, nodeID, reflect.DeepEqual(event.Policy, want))
	}
}

func setServiceAvailability(t *testing.T, c *serviceRPCClient, want nodepolicy.Availability) {
	t.Helper()
	params, err := json.Marshal(struct {
		State nodepolicy.Availability `json:"state"`
	}{State: want})
	if err != nil {
		t.Fatalf("encode availability: %v", err)
	}
	m := c.mustCall(t, "203", "node:set-availability", params)
	var result struct {
		Availability nodepolicy.Availability `json:"availability"`
	}
	if err := json.Unmarshal(m.Result, &result); err != nil {
		t.Fatalf("decode node:set-availability: %v", err)
	}
	if result.Availability != want {
		t.Fatalf("node:set-availability=%q, want %q", result.Availability, want)
	}
}

func assertAvailabilityChanged(t *testing.T, c *serviceRPCClient, nodeID string, want nodepolicy.Availability) {
	t.Helper()
	type availabilityEvent struct {
		NodeID       string                  `json:"nodeId"`
		Availability nodepolicy.Availability `json:"availability"`
		Active       *int                    `json:"active"`
	}
	var event availabilityEvent
	var decodeErr error
	_, err := c.next(serviceCallTimeout, func(m *jsonrpc.Message) bool {
		if !m.IsNotification() || m.Method != "node:availability-changed" {
			return false
		}
		var candidate availabilityEvent
		if decodeErr = json.Unmarshal(m.Params, &candidate); decodeErr != nil {
			return true
		}
		if candidate.Availability != want {
			return false
		}
		event = candidate
		return true
	})
	if err != nil {
		t.Fatalf("wait for availability %q: %v", want, err)
	}
	if decodeErr != nil {
		t.Fatalf("decode node:availability-changed: %v", decodeErr)
	}
	if event.NodeID != nodeID || event.Active == nil || *event.Active != 0 {
		t.Fatalf("availability event differs: node_id=%q want_node_id=%q active=%v", event.NodeID, nodeID, event.Active)
	}
}

func fixturePolicyNodeID(t *testing.T, f *serviceFixture) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(f.dir, "node-id.json"))
	if err != nil {
		t.Fatalf("read fixture node identity: %v", err)
	}
	var identity struct {
		NodeUUID string `json:"node_uuid"`
	}
	if err := json.Unmarshal(data, &identity); err != nil {
		t.Fatalf("decode fixture node identity: %v", err)
	}
	if identity.NodeUUID == "" {
		t.Fatal("fixture node identity is empty")
	}
	return identity.NodeUUID
}

func restartPolicyFixture(t *testing.T, f *serviceFixture) {
	t.Helper()
	if err := f.cleanupChecked(); err != nil {
		t.Fatalf("stop fixture before restart: %v", err)
	}
	f.assertStopped()
	f.start()
}

func readPolicyBytes(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read stored policy: %v", err)
	}
	return data
}

func assertStoredPolicy(t *testing.T, path string, want nodepolicy.Policy) []byte {
	t.Helper()
	data := readPolicyBytes(t, path)
	var stored nodepolicy.Policy
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatalf("decode stored complete policy: %v", err)
	}
	if err := stored.Validate(); err != nil {
		t.Fatalf("stored policy invalid: %v", err)
	}
	if !reflect.DeepEqual(stored, want) {
		t.Fatal("persisted complete policy differs from the successful RPC result")
	}
	return data
}
