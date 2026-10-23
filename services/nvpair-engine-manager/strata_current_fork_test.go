// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"nvpair-shared/engines"
)

// These tests characterize what the prepared Strata parser and compiler accept
// today against the shape the owner's current Strata fork generates
// (testdata/strata/current-fork-contract.json). They call the existing
// helpers only: no server, engine or installed port is contacted. A rejection
// here is a recorded fact, not a defect; the "class" of each rejected item
// says what the follow-up task must do with it:
//
//	incompatibility          the fork generates it and the prepared subset has no place for it
//	approval-binding         accepting it needs the effective value bound into the approval
//	fail-closed-until-task3  must stay rejected until local authentication exists
//	supported                accepted today
//
// A fixture mistake is a rejection caused by the specimen instead of the fork;
// TestStrataCurrentForkProfilePeel proves there is none left by removing every
// recorded blocker in order and requiring acceptance at the end.

const forkPlaceholder = "/REPLACE_WITH_ABSOLUTE_PATH"

// forkBooleanFlags are the native flags that take no value.
var forkBooleanFlags = map[string]bool{"--serve": true, "--no-prefill-borrow": true}

type forkField struct {
	Key    string `json:"key"`
	Class  string `json:"class"`
	Strata string `json:"strata"`
	Why    string `json:"why"`
}

type forkStage struct {
	ID         string            `json:"id"`
	Fields     bool              `json:"fields"`
	SetArgs    map[string]string `json:"set_args"`
	RemoveArgs []string          `json:"remove_args"`
	AddArgs    []string          `json:"add_args"`
	Class      string            `json:"class"`
	Expect     string            `json:"expect"`
	Strata     string            `json:"strata"`
	Reason     string            `json:"reason"`
}

type forkInventory struct {
	Name        string          `json:"name"`
	Body        json.RawMessage `json:"body"`
	List        []string        `json:"list"`
	ListKnown   bool            `json:"list_known"`
	Loaded      []string        `json:"loaded"`
	LoadedKnown bool            `json:"loaded_known"`
}

type forkAuthPath struct {
	ID     string `json:"id"`
	File   string `json:"file"`
	Symbol string `json:"symbol"`
	Today  string `json:"today"`
}

type forkContract struct {
	Contract string `json:"contract"`
	Version  int    `json:"version"`
	Sources  struct {
		Strata struct {
			Repository string `json:"repository"`
			Commit     string `json:"commit"`
			Generator  string `json:"generator"`
		} `json:"strata"`
		Pair struct {
			Base string `json:"base"`
		} `json:"pair"`
		Note string `json:"note"`
	} `json:"sources"`
	ModelID         string          `json:"model_id"`
	GeneratedConfig map[string]any  `json:"generated_config"`
	ExtraFields     []forkField     `json:"extra_fields"`
	Stages          []forkStage     `json:"stages"`
	Variants        []forkStage     `json:"variants"`
	Inventory       []forkInventory `json:"inventory"`
	AuthPaths       []forkAuthPath  `json:"auth_paths"`
}

const forkContractPath = "testdata/strata/current-fork-contract.json"

func loadForkContract(t *testing.T) forkContract {
	t.Helper()
	data, err := os.ReadFile(forkContractPath)
	if err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var contract forkContract
	if err := decoder.Decode(&contract); err != nil {
		t.Fatalf("decode %s: %v", forkContractPath, err)
	}
	return contract
}

// forkResolve replaces the path placeholder with the fixture root in native
// separators, as newStrataServingFixture does for its template.
func forkResolve(value any, root string) any {
	switch v := value.(type) {
	case string:
		if suffix, ok := strings.CutPrefix(v, forkPlaceholder); ok {
			return filepath.Join(root, filepath.FromSlash(suffix))
		}
		return v
	case []any:
		out := make([]any, len(v))
		for i := range v {
			out[i] = forkResolve(v[i], root)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(v))
		for key, item := range v {
			out[key] = forkResolve(item, root)
		}
		return out
	}
	return value
}

func forkClone(t *testing.T, raw map[string]any) map[string]any {
	t.Helper()
	data, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// forkArgs reads the native argument list, which is a []string when a test set
// it and a []any after a JSON round trip.
func forkArgs(t *testing.T, raw map[string]any) []string {
	t.Helper()
	switch v := raw["args"].(type) {
	case []string:
		return slices.Clone(v)
	case []any:
		out := make([]string, len(v))
		for i, item := range v {
			s, ok := item.(string)
			if !ok {
				t.Fatalf("args[%d] = %T, want string", i, item)
			}
			out[i] = s
		}
		return out
	}
	t.Fatalf("args = %T, want a list", raw["args"])
	return nil
}

// forkArgSpan returns a flag with its value, or the flag alone when it takes none.
func forkArgSpan(args []string, flag string) []string {
	i := slices.Index(args, flag)
	if i < 0 {
		return nil
	}
	if forkBooleanFlags[flag] {
		return args[i : i+1]
	}
	return args[i : i+2]
}

func forkRemoveArg(args []string, flag string) []string {
	span := forkArgSpan(args, flag)
	if span == nil {
		return args
	}
	i := slices.Index(args, flag)
	return slices.Delete(slices.Clone(args), i, i+len(span))
}

func forkSetArg(args []string, flag, value string) []string {
	out := slices.Clone(args)
	if i := slices.Index(out, flag); i >= 0 {
		out[i+1] = value
		return out
	}
	return append(out, flag, value)
}

// applyStage returns the config after the stage's remediation.
func (c forkContract) applyStage(t *testing.T, raw map[string]any, stage forkStage) map[string]any {
	t.Helper()
	next := forkClone(t, raw)
	if stage.Fields {
		for _, field := range c.ExtraFields {
			delete(next, field.Key)
		}
	}
	args := forkArgs(t, next)
	for flag, value := range stage.SetArgs {
		args = forkSetArg(args, flag, value)
	}
	for _, flag := range stage.RemoveArgs {
		args = forkRemoveArg(args, flag)
	}
	next["args"] = append(slices.Clone(stage.AddArgs), args...)
	return next
}

// newForkProfileFixture is the prepared fixture carrying the generated model
// identity. The generated args select no draft, so the baseline's draft files
// are no longer declared: leaving them is a fixture mistake, pinned below.
func newForkProfileFixture(t *testing.T, c forkContract) strataServingFixture {
	t.Helper()
	f := newStrataServingFixture(t)
	model := &f.request.Settings.Serving.Models[0]
	model.ModelID = c.ModelID
	model.Artifacts = slices.DeleteFunc(model.Artifacts, func(a engines.ServingArtifact) bool { return a.Role == "draft" })
	return f
}

func forkBaselineRaw(t *testing.T, f strataServingFixture) map[string]any {
	t.Helper()
	data, err := json.Marshal(f.config)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	return raw
}

func forkGeneratedRaw(t *testing.T, c forkContract, root string) map[string]any {
	t.Helper()
	raw, ok := forkResolve(c.GeneratedConfig, root).(map[string]any)
	if !ok {
		t.Fatal("generated config is not an object")
	}
	return raw
}

// deriveRaw writes the config bytes, rebinds the config artifact and runs the
// compiler's derivation. Nothing is started or persisted.
func (f *strataServingFixture) deriveRaw(t *testing.T, raw map[string]any) error {
	t.Helper()
	data, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	writeServingFixture(t, f.source, data)
	f.bindConfig(t)
	st := settingsState(t, f.ex)
	command, err := launchForState(st, f.request.Settings.ServerPort)
	if err != nil {
		t.Fatal(err)
	}
	rt := st.plat.Runtime
	rt.Serving = f.request.Settings.Serving
	_, err = deriveStrataServing(command, rt)
	return err
}

func requireForkOutcome(t *testing.T, err error, expect string) {
	t.Helper()
	switch {
	case expect == "" && err != nil:
		t.Fatalf("rejected, want accepted: %v", err)
	case expect != "" && err == nil:
		t.Fatalf("accepted, want a rejection containing %q", expect)
	case expect != "" && !strings.Contains(err.Error(), expect):
		t.Fatalf("error = %q, want it to contain %q", err, expect)
	}
}

// The generated profile is rejected, and removing the recorded blockers one at
// a time in derivation order reaches acceptance. The final acceptance shows no
// unrecorded blocker remains; each stage's class is the verdict for Task 2.
func TestStrataCurrentForkProfilePeel(t *testing.T) {
	c := loadForkContract(t)
	t.Run("prepared baseline is accepted", func(t *testing.T) {
		f := newStrataServingFixture(t)
		if err := f.deriveRaw(t, forkBaselineRaw(t, f)); err != nil {
			t.Fatalf("baseline rejected: %v", err)
		}
	})
	f := newForkProfileFixture(t, c)
	raw := forkGeneratedRaw(t, c, f.root)
	for _, stage := range c.Stages {
		err := f.deriveRaw(t, raw)
		if err == nil || !strings.Contains(err.Error(), stage.Expect) {
			t.Fatalf("stage %s: error = %v, want it to contain %q", stage.ID, err, stage.Expect)
		}
		t.Logf("%s [%s] %s: %v", stage.ID, stage.Class, stage.Strata, err)
		raw = c.applyStage(t, raw, stage)
	}
	if err := f.deriveRaw(t, raw); err != nil {
		t.Fatalf("profile after every recorded remediation is still rejected: %v", err)
	}
}

// Each blocker added alone to the accepted baseline is rejected for its own
// reason, so none of the rejections comes from another specimen detail.
func TestStrataCurrentForkIsolatedIncompatibilities(t *testing.T) {
	c := loadForkContract(t)
	type probe struct {
		name, class, expect string
		mutate              func(t *testing.T, base, generated map[string]any)
	}
	var probes []probe
	for _, stage := range c.Stages {
		if stage.Fields {
			for _, field := range c.ExtraFields {
				probes = append(probes, probe{"field " + field.Key, field.Class, stage.Expect, func(t *testing.T, base, generated map[string]any) {
					base[field.Key] = generated[field.Key]
				}})
			}
		}
		for flag := range stage.SetArgs {
			probes = append(probes, probe{"generated " + flag, stage.Class, stage.Expect, func(t *testing.T, base, generated map[string]any) {
				span := forkArgSpan(forkArgs(t, generated), flag)
				base["args"] = forkSetArg(forkArgs(t, base), flag, span[1])
			}})
		}
		for _, flag := range stage.RemoveArgs {
			probes = append(probes, probe{"generated " + flag, stage.Class, stage.Expect, func(t *testing.T, base, generated map[string]any) {
				span := forkArgSpan(forkArgs(t, generated), flag)
				base["args"] = append(slices.Clone(span), forkArgs(t, base)...)
			}})
		}
		for _, flag := range stage.AddArgs {
			probes = append(probes, probe{"without " + flag, stage.Class, stage.Expect, func(t *testing.T, base, generated map[string]any) {
				base["args"] = forkRemoveArg(forkArgs(t, base), flag)
			}})
		}
	}
	for _, variant := range c.Variants {
		probes = append(probes, probe{"variant " + variant.ID, variant.Class, variant.Expect, func(t *testing.T, base, generated map[string]any) {
			args := forkArgs(t, base)
			for flag, value := range variant.SetArgs {
				args = forkSetArg(args, flag, value)
			}
			base["args"] = append(slices.Clone(variant.AddArgs), args...)
		}})
	}
	for _, tc := range probes {
		t.Run(tc.name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			base := forkBaselineRaw(t, f)
			tc.mutate(t, base, forkGeneratedRaw(t, c, f.root))
			requireForkOutcome(t, f.deriveRaw(t, base), tc.expect)
			t.Logf("class %s", tc.class)
		})
	}
}

// The baseline's draft files are declared artifacts. Keeping them while the
// generated args select no draft is rejected for that reason alone: a mistake of
// the specimen, not of the fork.
func TestStrataCurrentForkFixtureMistakeControl(t *testing.T) {
	c := loadForkContract(t)
	f := newStrataServingFixture(t)
	f.request.Settings.Serving.Models[0].ModelID = c.ModelID
	raw := forkGeneratedRaw(t, c, f.root)
	for _, stage := range c.Stages {
		raw = c.applyStage(t, raw, stage)
	}
	requireForkOutcome(t, f.deriveRaw(t, raw), "Unselected Strata model artifact")
}

// Packs written by the fork's packer normally name expert shards through
// native_experts.txt instead of carrying an experts.bin.
func TestStrataCurrentForkPackShape(t *testing.T) {
	t.Run("indirection table beside experts.bin", func(t *testing.T) {
		f := newStrataServingFixture(t)
		writeServingFixture(t, filepath.Join(f.root, "pack", "native_experts.txt"), []byte("synthetic table"))
		requireForkOutcome(t, f.deriveRaw(t, forkBaselineRaw(t, f)), "native_experts indirection is unsupported")
	})
	t.Run("indirection table without experts.bin", func(t *testing.T) {
		f := newStrataServingFixture(t)
		writeServingFixture(t, filepath.Join(f.root, "pack", "native_experts.txt"), []byte("synthetic table"))
		if err := os.Remove(filepath.Join(f.root, "pack", "experts.bin")); err != nil {
			t.Fatal(err)
		}
		err := f.deriveRaw(t, forkBaselineRaw(t, f))
		if err == nil {
			t.Fatal("pack without experts.bin accepted")
		}
		t.Logf("real pack shape rejected: %v", err)
	})
}

func TestStrataCurrentForkConfigGrammar(t *testing.T) {
	t.Run("byte order mark is rejected", func(t *testing.T) {
		if _, err := parseStrataServingConfig([]byte("\xef\xbb\xbf{}")); err == nil {
			t.Fatal("byte order mark accepted")
		}
	})
	t.Run("wrong types are rejected", func(t *testing.T) {
		for _, data := range []string{`{"parallel":"2"}`, `{"args":"--serve"}`, `{"lib_dirs":"lib"}`} {
			if _, err := parseStrataServingConfig([]byte(data)); err == nil {
				t.Fatalf("%s accepted", data)
			}
		}
	})
	// Known gap for Task 2: the duplicate-key scan is case sensitive while the
	// typed decoder folds case, so a case alias silently replaces the first value.
	t.Run("case alias of a key is accepted today", func(t *testing.T) {
		config, err := parseStrataServingConfig([]byte(`{"parallel":1,"Parallel":2}`))
		if err != nil || config.Parallel != 2 {
			t.Fatalf("recorded behaviour changed: parallel = %d, error = %v", config.Parallel, err)
		}
	})
}

// The generated config's env block names these variables, and the server turns
// them into child environment. The prepared launch rejects the names themselves.
func TestStrataCurrentForkLaunchEnvironment(t *testing.T) {
	c := loadForkContract(t)
	f := newStrataServingFixture(t)
	command, err := launchForState(settingsState(t, f.ex), f.request.Settings.ServerPort)
	if err != nil {
		t.Fatal(err)
	}
	generated, ok := c.GeneratedConfig["env"].(map[string]any)
	if !ok || len(generated) == 0 {
		t.Fatal("fixture lost the generated environment")
	}
	names := []string{"STRATA_API_KEY", "PYTHONUSERBASE"}
	for key := range generated {
		names = append(names, key, strings.ToLower(key))
	}
	slices.Sort(names)
	withEnv := func(name, value string) launchCommand {
		candidate := command
		candidate.Env = maps.Clone(command.Env)
		if candidate.Env == nil {
			candidate.Env = map[string]string{}
		}
		candidate.Env[name] = value
		return candidate
	}
	for _, name := range slices.Compact(names) {
		for _, value := range []string{"1", ""} {
			t.Run(fmt.Sprintf("%s=%q", name, value), func(t *testing.T) {
				_, err := strataLaunchSource(withEnv(name, value))
				requireForkOutcome(t, err, "environment overrides are unsupported")
			})
		}
	}
	t.Run("unrelated variable is accepted", func(t *testing.T) {
		if _, err := strataLaunchSource(withEnv("PAIR_FIXTURE_UNRELATED", "1")); err != nil {
			t.Fatal(err)
		}
	})
}

// Inventory is parsed by the bundled manifest's own extractors. Only a row whose
// status is exactly "loaded" is resident; an unclassifiable row never counts.
func TestStrataCurrentForkInventory(t *testing.T) {
	c := loadForkContract(t)
	manifest := strataManifest(t)
	list, loaded := manifest.Actions["list_models"].Result, manifest.Actions["loaded_models"].Result
	if list == nil || loaded == nil {
		t.Fatal("bundled Strata manifest lost its inventory extractors")
	}
	for _, tc := range c.Inventory {
		t.Run(tc.Name, func(t *testing.T) {
			gotList, listKnown := extractStringsResult(tc.Body, list)
			gotLoaded, loadedKnown := extractStringsResult(tc.Body, loaded)
			if listKnown != tc.ListKnown || !slices.Equal(gotList, tc.List) {
				t.Fatalf("list = %v (known %v), want %v (known %v)", gotList, listKnown, tc.List, tc.ListKnown)
			}
			if loadedKnown != tc.LoadedKnown || !slices.Equal(gotLoaded, tc.Loaded) {
				t.Fatalf("loaded = %v (known %v), want %v (known %v)", gotLoaded, loadedKnown, tc.Loaded, tc.LoadedKnown)
			}
		})
	}
}

// The recorded owners are where the credential path will have to change. Pin
// that each still exists so a move is noticed, and that the fixture stays
// sanitized.
func TestStrataCurrentForkAuthPathsAndSanitizedFixture(t *testing.T) {
	c := loadForkContract(t)
	for _, owner := range c.AuthPaths {
		t.Run(owner.ID, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("..", filepath.FromSlash(owner.File)))
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(data), owner.Symbol) {
				t.Fatalf("%s no longer contains %q", owner.File, owner.Symbol)
			}
		})
	}
	data, err := os.ReadFile(forkContractPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{`C:\`, `C:/`, "/Users/", "/home/", "BEGIN ", "sk-"} {
		if bytes.Contains(data, []byte(forbidden)) {
			t.Errorf("fixture contains %q", forbidden)
		}
	}
	if key, _ := c.GeneratedConfig["api_key"].(string); !strings.HasPrefix(key, "FAKE-") {
		t.Errorf("api_key = %q, want an obviously fake value", key)
	}
	revision := regexp.MustCompile(`^[0-9a-f]{40}$`)
	for name, value := range map[string]string{"strata": c.Sources.Strata.Commit, "pair": c.Sources.Pair.Base} {
		if !revision.MatchString(value) {
			t.Errorf("%s revision = %q, want a full commit hash", name, value)
		}
	}
}

// LoadOverrideDir reads every JSON file in testdata/strata. The sidecar must not
// register as an engine or the manifest fixtures would change under it.
func TestStrataCurrentForkSidecarIsNotAManifest(t *testing.T) {
	with := NewRegistry()
	if err := with.LoadOverrideDir("testdata/strata"); err != nil {
		t.Fatal(err)
	}
	manifest, err := os.ReadFile("testdata/strata/strata.json")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	writeServingFixture(t, filepath.Join(dir, "strata.json"), manifest)
	without := NewRegistry()
	if err := without.LoadOverrideDir(dir); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(with.Names(), without.Names()) || !slices.Equal(with.Names(), []string{"strata"}) {
		t.Fatalf("engines with sidecar = %v, without = %v", with.Names(), without.Names())
	}
}
