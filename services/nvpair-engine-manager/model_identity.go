// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// model_identity.go is engine-manager's part of the strict admission identity
// (docs/local-qualification.mdx): which process incarnation of each engine is
// running, what it was launched with, and the content identity of each model
// it lists. The broker adds its settings fingerprint and gives the whole
// identity to the proxy, which binds strict capacity and qualification to it.
// Only runs this manager launched as an owned process report one: an adopted
// service or a command-mode daemon was not started from a launch it can name.

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"os"
	"reflect"
	"regexp"
	"sort"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

// modelIdentityMethod is the notification carrying modelIdentityParams. The
// loaded-model watcher emits it after its first sweep and after every sweep
// that changed it; every state change pokes the watcher.
const modelIdentityMethod = "engine:model-identity"

// modelIdentityParams is a whole snapshot: an engine missing from it has no
// identity now.
type modelIdentityParams struct {
	Engines map[string]engineModelIdentity `json:"engines"`
	// Local publication generation; never part of the wire contract.
	epochs map[string]uint64
}

type engineModelIdentity struct {
	EngineRunID  string                             `json:"engineRunId"`
	LaunchDigest string                             `json:"launchDigest"`
	ModelDigests map[string]string                  `json:"modelDigests"`
	Serving      map[string]engines.ServingEnvelope `json:"serving,omitempty"`
}

// observedDigests are the content digests one inventory request returned,
// with the run it was made against.
type observedDigests struct {
	runID  string
	models map[string]string
	// epoch and suppressed belong to the inventory owner. Observations carry
	// the epoch captured before I/O until storeInventory accepts them.
	epoch      uint64
	suppressed bool
}

var contentDigest = regexp.MustCompile(`^[0-9a-f]{64}$`)

func newEngineRunID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "run-" + hex.EncodeToString(b[:])
}

// launchDigest identifies a process launch: the executable by path, size and
// modification time (so an engine upgrade changes it), the arguments and the
// explicit environment overrides. Inherited environment is not included. An
// executable that cannot be read yields no digest.
func launchDigest(l launchCommand) string {
	info, err := os.Stat(l.Bin)
	if err != nil || !info.Mode().IsRegular() {
		return ""
	}
	env := make([]string, 0, len(l.Env))
	for k, v := range l.Env {
		env = append(env, k+"="+v)
	}
	sort.Strings(env)
	doc, err := json.Marshal(struct {
		Version    int      `json:"version"`
		Executable string   `json:"executable"`
		Size       int64    `json:"size"`
		ModTimeNs  int64    `json:"modTimeNs"`
		Args       []string `json:"args"`
		Env        []string `json:"env"`
	}{1, l.Bin, info.Size(), info.ModTime().UnixNano(), l.Args, env})
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(doc)
	return hex.EncodeToString(sum[:])
}

// engineRunID reads the engine's current run id, empty when it is not
// running or unknown here.
func (e *Executor) engineRunID(engine string) string {
	st, err := e.state(engine)
	if err != nil {
		return ""
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.runID
}

// extractDigests reads spec.Digest beside spec.Field from each element of a
// list_models response. Rows whose digest is not lowercase SHA-256 hex are
// left out, as is a name listed twice with different digests.
func extractDigests(raw json.RawMessage, spec *ActionResult) map[string]string {
	var obj map[string]json.RawMessage
	if spec == nil || spec.Digest == "" || json.Unmarshal(raw, &obj) != nil {
		return nil
	}
	var rows []map[string]json.RawMessage
	if json.Unmarshal(obj[spec.Array], &rows) != nil {
		return nil
	}
	out := make(map[string]string, len(rows))
	conflicting := map[string]bool{}
	for _, row := range rows {
		if spec.Match != nil && !matchRow(row, spec.Match) {
			continue
		}
		var name, digest string
		if nodepolicy.ValidateModelIdentifierJSON(row[spec.Field]) != nil || json.Unmarshal(row[spec.Field], &name) != nil ||
			json.Unmarshal(row[spec.Digest], &digest) != nil || !contentDigest.MatchString(digest) {
			continue
		}
		if prev, seen := out[name]; seen && prev != digest {
			conflicting[name] = true
		}
		out[name] = digest
	}
	for name := range conflicting {
		delete(out, name)
	}
	return out
}

// modelIdentity snapshots every engine this manager launched and still owns:
// its run id, launch digest and the digests its inventory listed under that
// same run (none when the last listing belonged to another run).
func (e *Executor) modelIdentity() modelIdentityParams {
	engineNames, naming := e.inventoryModelNaming()
	e.inventoryMu.Lock()
	defer e.inventoryMu.Unlock()
	digests := e.inventoryDigests
	out := modelIdentityParams{Engines: map[string]engineModelIdentity{}}
	for name, observed := range digests {
		if observed.epoch != 0 {
			if out.epochs == nil {
				out.epochs = make(map[string]uint64)
			}
			out.epochs[name] = observed.epoch
		}
	}
	for _, name := range engineNames {
		if digests[name].suppressed {
			continue
		}
		st, err := e.state(name)
		if err != nil {
			continue
		}
		st.mu.Lock()
		owned := st.running && !st.adopted && st.proc != nil && st.runID != "" && st.launchDigest != ""
		id := engineModelIdentity{EngineRunID: st.runID, LaunchDigest: st.launchDigest, ModelDigests: map[string]string{}}
		serving := st.plat.Runtime.Serving != nil
		if owned && serving && st.plat.Runtime.servingDrift == "" {
			id = servingIdentityLocked(st)
		}
		st.mu.Unlock()
		if !owned {
			continue
		}
		if observed, ok := digests[name]; !serving && ok && observed.runID == id.EngineRunID {
			for model, digest := range observed.models {
				if nodepolicy.ValidateModelIdentifier(model) == nil {
					id.ModelDigests[model] = digest
				}
			}
		}
		out.Engines[name] = id
	}
	if e.downloads != nil {
		e.downloads.mu.Lock()
		defer e.downloads.mu.Unlock()
	}
	for name, id := range out.Engines {
		for model := range id.ModelDigests {
			if !e.modelReadyLocked(name, model, naming[name]) {
				delete(id.ModelDigests, model)
				delete(id.Serving, model)
			}
		}
	}
	return out
}

// publishModelIdentity emits the identity snapshot when it differs from last,
// or when nothing was sent yet, and returns what was last sent. The snapshot
// is read and sent under publishMu, like engine:state-changed, so the broker
// never receives a running engine's identity after its stop.
func (e *Executor) publishModelIdentity(last *modelIdentityParams) *modelIdentityParams {
	e.publishMu.Lock()
	defer e.publishMu.Unlock()
	cur := e.modelIdentity()
	if last != nil && reflect.DeepEqual(*last, cur) {
		return last
	}
	// The launch digest is what a qualification artifact names; logging it
	// lets the qualifying operator read it. Both values are public hashes.
	for name, id := range cur.Engines {
		slog.Info("engine identity", "engine", name, "run_id", id.EngineRunID, "launch_digest", id.LaunchDigest, "models_with_digest", len(id.ModelDigests))
	}
	e.notify(modelIdentityMethod, cur)
	return &cur
}
