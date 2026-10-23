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

// These tests pin what the prepared Strata parser and compiler accept and reject
// for the shape the owner's current Strata fork generates
// (testdata/strata/current-fork-contract.json). They call the existing helpers
// only: no server, engine or installed port is contacted. The "class" of an item
// says why it was a blocker, and its decision and outcome in the fixture say what
// the adapter now does with it:
//
//	incompatibility          the fork generates it; bound, agreed or ignored now
//	approval-binding         needs the effective value bound into approval; still rejected
//	fail-closed-until-task3  must stay rejected until local authentication exists
//	supported                accepted
//
// A fixture mistake is a rejection caused by the specimen instead of the fork;
// TestStrataCurrentForkProfilePeel proves there is none left by removing the
// still-rejected blockers in order and requiring acceptance at the end.

const forkPlaceholder = "/REPLACE_WITH_ABSOLUTE_PATH"

// forkBooleanFlags are the native flags that take no value.
var forkBooleanFlags = map[string]bool{"--serve": true, "--no-prefill-borrow": true}

// forkField is one top-level field the fork generates. Decision is what the
// prepared adapter does with it (bind: forwarded to the server, ignore: validated
// and not forwarded, agree: must match the PAIR launch, keep rejecting) and
// Outcome is whether the field is accepted or rejected today.
type forkField struct {
	Key      string `json:"key"`
	Class    string `json:"class"`
	Decision string `json:"decision"`
	Outcome  string `json:"outcome"`
	Strata   string `json:"strata"`
	Why      string `json:"why"`
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
			Inputs     string `json:"inputs"`
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
			if field.Outcome == "rejected" {
				delete(next, field.Key)
			}
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

// deriveLaunch writes the config bytes, rebinds the config artifact and runs the
// compiler's derivation. Nothing is started or persisted.
func (f *strataServingFixture) deriveLaunch(t *testing.T, raw map[string]any) (*llamaServingLaunch, error) {
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
	return deriveStrataServing(command, rt)
}

func (f *strataServingFixture) deriveRaw(t *testing.T, raw map[string]any) error {
	t.Helper()
	_, err := f.deriveLaunch(t, raw)
	return err
}

// forkPeeled is the generated profile with only the still-rejected blockers
// remediated: the shape every other decision must accept.
func forkPeeled(t *testing.T, c forkContract, root string) map[string]any {
	t.Helper()
	raw := forkGeneratedRaw(t, c, root)
	for _, stage := range c.Stages {
		raw = c.applyStage(t, raw, stage)
	}
	return raw
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
				expect := stage.Expect
				if field.Outcome == "accepted" {
					expect = ""
				}
				probes = append(probes, probe{"field " + field.Key, field.Class + "/" + field.Decision, expect, func(t *testing.T, base, generated map[string]any) {
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
			for _, flag := range variant.RemoveArgs {
				args = forkRemoveArg(args, flag)
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

// useForkExpertTable turns the fixture's pack into the fork packer's shape: no
// experts.bin, a native_experts.txt that is declared like any pack file.
func useForkExpertTable(t *testing.T, f *strataServingFixture, table string) {
	t.Helper()
	if err := os.Remove(filepath.Join(f.root, "pack", "experts.bin")); err != nil {
		t.Fatal(err)
	}
	model := &f.request.Settings.Serving.Models[0]
	model.Artifacts = slices.DeleteFunc(model.Artifacts, func(a engines.ServingArtifact) bool { return a.Role == "model:pack-experts" })
	path := filepath.Join(f.root, "pack", "native_experts.txt")
	writeServingFixture(t, path, []byte(table))
	model.Artifacts = append(model.Artifacts, strataArtifact(t, path, "model:pack-native-experts"))
}

const (
	forkTableV3   = "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]\n"
	forkTableV4   = "# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard | gate,up,down]\n"
	forkShardOne  = "model-00001-of-00002.gguf"
	forkShardTwo  = "model-00002-of-00002.gguf"
	forkLayerZero = "0 12 14 0 4096 100 200 300"
	forkLayerOne  = "1 12 14 4096 4096 400 500 600"
)

// The fork's packer writes native_experts.txt, a table naming the shards that
// hold each layer's experts, and normally no experts.bin. The table and every
// shard it names are bound; an unreadable or ambiguous table is rejected.
func TestStrataCurrentForkNativeExpertsTable(t *testing.T) {
	for _, tc := range []struct {
		name, table, expect string
	}{
		{"v3 without a shard column", forkTableV3 + forkLayerZero + "\n" + forkLayerOne + "\n", ""},
		{"v3 with one shard column", forkTableV3 + forkLayerZero + " " + forkShardTwo + "\n", ""},
		{"v4 per-role shards", forkTableV4 + forkLayerZero + " " + forkShardOne + "," + forkShardTwo + "," + forkShardOne + "\r\n", ""},
		{"comments and blank lines", forkTableV3 + "\n# note\n" + forkLayerZero + "\n", ""},
		{"v4 empty components mean the first shard", forkTableV4 + forkLayerZero + " ," + forkShardTwo + ",\n", ""},
		{"v4 list that names no shard", forkTableV4 + forkLayerZero + " ,,\n", "native_experts table"},
		{"empty component in a v3 column", forkTableV3 + forkLayerZero + " ," + forkShardTwo + ",\n", "native_experts table"},
		{"later header that agrees", forkTableV4 + forkLayerZero + "\n" + forkTableV4, ""},
		{"later header above the supported versions", forkTableV3 + forkLayerZero + "\n# strata native experts v5: layer\n", "native_experts table"},
		{"later header that disagrees", forkTableV3 + forkLayerZero + "\n" + forkTableV4, "native_experts table"},
		{"later tag without a version", forkTableV3 + forkLayerZero + "\n# strata native experts vx\n", "native_experts table"},
		{"unsupported version", "# strata native experts v5: layer\n" + forkLayerZero + "\n", "native_experts table"},
		{"missing header", forkLayerZero + "\n", "native_experts table"},
		{"no layers", forkTableV3, "native_experts table"},
		{"layer does not start at zero", forkTableV3 + forkLayerOne + "\n", "native_experts table"},
		{"duplicate layer", forkTableV3 + forkLayerZero + "\n" + forkLayerZero + "\n", "native_experts table"},
		{"short line", forkTableV3 + "0 12 14 0 4096 100 200\n", "native_experts table"},
		{"non-numeric field", forkTableV3 + "0 12 14 0 4096 100 200 x\n", "native_experts table"},
		{"negative field", forkTableV3 + "0 12 14 0 4096 100 200 -3\n", "native_experts table"},
		{"overflowing field", forkTableV3 + "0 12 14 0 4096 100 200 99999999999999999999\n", "native_experts table"},
		{"v3 with a role list", forkTableV3 + forkLayerZero + " " + forkShardOne + "," + forkShardTwo + "," + forkShardOne + "\n", "native_experts table"},
		{"two shard names", forkTableV4 + forkLayerZero + " " + forkShardOne + "," + forkShardTwo + "\n", "native_experts table"},
		{"parent traversal", forkTableV3 + forkLayerZero + " ../" + forkShardTwo + "\n", "native_experts table"},
		{"absolute name", forkTableV3 + forkLayerZero + " /" + forkShardTwo + "\n", "native_experts table"},
		{"backslash name", forkTableV3 + forkLayerZero + ` sub\` + forkShardTwo + "\n", "native_experts table"},
		{"stream name", forkTableV3 + forkLayerZero + " " + forkShardTwo + ":stream\n", "native_experts table"},
		{"not a gguf name", forkTableV3 + forkLayerZero + " experts.bin\n", "native_experts table"},
		{"undeclared shard", forkTableV3 + forkLayerZero + " other-00001-of-00001.gguf\n", "no declared artifact"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			writeServingFixture(t, filepath.Join(f.root, "other-00001-of-00001.gguf"), []byte("undeclared shard"))
			useForkExpertTable(t, &f, tc.table)
			requireForkOutcome(t, f.deriveRaw(t, forkBaselineRaw(t, f)), tc.expect)
		})
	}
	t.Run("table beside experts.bin is ambiguous", func(t *testing.T) {
		f := newStrataServingFixture(t)
		writeServingFixture(t, filepath.Join(f.root, "pack", "native_experts.txt"), []byte(forkTableV3+forkLayerZero+"\n"))
		requireForkOutcome(t, f.deriveRaw(t, forkBaselineRaw(t, f)), "expert source is ambiguous")
	})
	t.Run("a table in the draft pack is still unsupported", func(t *testing.T) {
		f := newStrataServingFixture(t)
		writeServingFixture(t, filepath.Join(f.root, "draft", "native_experts.txt"), []byte(forkTableV3+forkLayerZero+"\n"))
		requireForkOutcome(t, f.deriveRaw(t, forkBaselineRaw(t, f)), "native_experts indirection is unsupported")
	})
	t.Run("changed table needs a fresh binding", func(t *testing.T) {
		f := newStrataServingFixture(t)
		useForkExpertTable(t, &f, forkTableV3+forkLayerZero+"\n")
		if err := f.deriveRaw(t, forkBaselineRaw(t, f)); err != nil {
			t.Fatal(err)
		}
		writeServingFixture(t, filepath.Join(f.root, "pack", "native_experts.txt"), []byte(forkTableV3+forkLayerZero+"\n"+forkLayerOne+"\n"))
		requireForkOutcome(t, f.deriveRaw(t, forkBaselineRaw(t, f)), "binding changed")
	})
}

func TestStrataCurrentForkConfigGrammar(t *testing.T) {
	t.Run("byte order mark is rejected", func(t *testing.T) {
		if _, err := parseStrataServingConfig([]byte("\xef\xbb\xbf{}")); err == nil {
			t.Fatal("byte order mark accepted")
		}
	})
	t.Run("wrong types are rejected", func(t *testing.T) {
		for _, data := range []string{`{"parallel":"2"}`, `{"args":"--serve"}`, `{"lib_dirs":"lib"}`, `{"gpu":"0"}`, `{"gpu":0.5}`, `{"env":{"STRATA_FILE_CACHE_MIB":5}}`, `{"fit_max_tokens":"yes"}`, `{"log":null}`} {
			if _, err := parseStrataServingConfig([]byte(data)); err == nil {
				t.Fatalf("%s accepted", data)
			}
		}
	})
	t.Run("keys that differ only by case are ambiguous", func(t *testing.T) {
		for _, data := range []string{`{"parallel":1,"Parallel":2}`, `{"PARALLEL":2,"parallel":1}`, `{"env":{"STRATA_FILE_CACHE_MIB":"1","strata_file_cache_mib":"2"}}`, `{"env":{"STRATA_OWNED_PRICE":"exact","STRATA_OWNED_PRICE":"exact"}}`} {
			if _, err := parseStrataServingConfig([]byte(data)); err == nil || !strings.Contains(err.Error(), "Duplicate") {
				t.Fatalf("%s: error = %v, want a duplicate rejection", data, err)
			}
		}
	})
}

// The config env block is bound through a closed list of the memory controls
// the fork's planner generates, each with its value syntax.
func TestStrataCurrentForkEnvironmentAllowList(t *testing.T) {
	c := loadForkContract(t)
	for _, tc := range []struct {
		name   string
		env    map[string]any
		expect string
	}{
		{"generated block", c.GeneratedConfig["env"].(map[string]any), ""},
		{"empty block", map[string]any{}, ""},
		{"zero recent cache", map[string]any{"STRATA_FILE_CACHE_MIB": "0"}, ""},
		{"fractional headroom", map[string]any{"STRATA_RESIDENT_HEADROOM_GIB": "1.5"}, ""},
		{"unreviewed name", map[string]any{"STRATA_API_KEY": "x"}, "is not a reviewed value"},
		{"lower case name", map[string]any{"strata_file_cache_mib": "1"}, "is not a reviewed value"},
		{"python name", map[string]any{"PYTHONPATH": "x"}, "is not a reviewed value"},
		{"negative cache", map[string]any{"STRATA_FILE_CACHE_MIB": "-1"}, "is not a reviewed value"},
		{"cache above the planner limit", map[string]any{"STRATA_FILE_CACHE_MIB": "17592186044416"}, "is not a reviewed value"},
		{"signed cache", map[string]any{"STRATA_FILE_CACHE_MIB": "+5"}, "is not a reviewed value"},
		{"compact flag value", map[string]any{"STRATA_FILE_CACHE_COMPACT": "2"}, "is not a reviewed value"},
		{"price mode", map[string]any{"STRATA_OWNED_PRICE": "cheap"}, "is not a reviewed value"},
		{"empty headroom", map[string]any{"STRATA_RESIDENT_HEADROOM_GIB": ""}, "is not a reviewed value"},
		{"non-numeric headroom", map[string]any{"STRATA_RESIDENT_HEADROOM_GIB": "abc"}, "is not a reviewed value"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			base := forkBaselineRaw(t, f)
			base["env"] = tc.env
			requireForkOutcome(t, f.deriveRaw(t, base), tc.expect)
		})
	}
}

// The values the server does not read, or reads from the command line, are
// validated and then not forwarded; the others reach the snapshot, and every
// reviewed control appears in the approved placement, in a fixed order.
func TestStrataCurrentForkSnapshotAndPlacement(t *testing.T) {
	c := loadForkContract(t)
	f := newForkProfileFixture(t, c)
	launch, err := f.deriveLaunch(t, forkPeeled(t, c, f.root))
	if err != nil {
		t.Fatalf("profile rejected: %v", err)
	}
	var snapshot map[string]json.RawMessage
	if err := json.Unmarshal(launch.Snapshot, &snapshot); err != nil {
		t.Fatal(err)
	}
	for _, kept := range []string{"log", "env", "gpu", "fit_max_tokens", "exe", "args", "tokenizer", "model_name", "lib_dirs"} {
		if _, ok := snapshot[kept]; !ok {
			t.Errorf("snapshot lost %s", kept)
		}
	}
	for _, dropped := range []string{"port", "host", "build_metadata", "gpus_asked", "open_browser", "api_key", "memory_policy"} {
		if _, ok := snapshot[dropped]; ok {
			t.Errorf("snapshot forwards %s", dropped)
		}
	}
	want := []string{
		"--expert-cache=64", "--kv=q4_0", "--spec=2", "--suffix-draft=0", "--lookup-chain=0",
		"--resident-budget-gib=3", "--prefill=64", "--no-prefill-borrow", "--kv-resident=20480",
		"env:STRATA_FILE_CACHE_COMPACT=1", "env:STRATA_FILE_CACHE_MIB=2136", "env:STRATA_OWNED_PRICE=exact", "env:STRATA_RESIDENT_HEADROOM_GIB=4",
	}
	if len(launch.Envelopes) != 1 || !slices.Equal(launch.Envelopes[0].Placement, want) {
		t.Fatalf("placement = %v, want %v", launch.Envelopes, want)
	}
}

// memory_policy and an inline api_key each keep the otherwise accepted profile
// rejected; the key belongs in the file api_key_file names.
func TestStrataCurrentForkFailClosedFields(t *testing.T) {
	c := loadForkContract(t)
	for _, key := range []string{"api_key", "memory_policy"} {
		t.Run(key, func(t *testing.T) {
			f := newForkProfileFixture(t, c)
			raw := forkPeeled(t, c, f.root)
			if err := f.deriveRaw(t, raw); err != nil {
				t.Fatalf("control rejected: %v", err)
			}
			raw[key] = forkResolve(c.GeneratedConfig[key], f.root)
			requireForkOutcome(t, f.deriveRaw(t, raw), "Unsupported or malformed prepared Strata config.")
		})
	}
}

// The server opens the log for append, so it must not be another name for a
// declared input: judged by file identity, not by spelling.
func TestStrataCurrentForkLogIdentity(t *testing.T) {
	logOver := func(t *testing.T, f *strataServingFixture, path string) error {
		t.Helper()
		base := forkBaselineRaw(t, *f)
		base["log"] = path
		return f.deriveRaw(t, base)
	}
	t.Run("hard link to a declared file", func(t *testing.T) {
		f := newStrataServingFixture(t)
		alias := filepath.Join(f.root, "alias.log")
		if err := os.Link(filepath.Join(f.root, "server.py"), alias); err != nil {
			t.Fatal(err)
		}
		requireForkOutcome(t, logOver(t, &f, alias), "Strata log")
	})
	t.Run("directory", func(t *testing.T) {
		f := newStrataServingFixture(t)
		dir := filepath.Join(f.root, "logdir")
		if err := os.Mkdir(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		requireForkOutcome(t, logOver(t, &f, dir), "Strata log")
	})
	t.Run("symbolic link", func(t *testing.T) {
		f := newStrataServingFixture(t)
		alias := filepath.Join(f.root, "link.log")
		if err := os.Symlink(filepath.Join(f.root, "server.py"), alias); err != nil {
			t.Logf("symbolic links are not permitted here, so this case is not exercised: %v", err)
			return
		}
		requireForkOutcome(t, logOver(t, &f, alias), "Strata log")
	})
	t.Run("existing regular log", func(t *testing.T) {
		f := newStrataServingFixture(t)
		old := filepath.Join(f.root, "old.log")
		writeServingFixture(t, old, []byte("earlier output"))
		requireForkOutcome(t, logOver(t, &f, old), "")
	})
}

// Validation of the fields that are accepted without being forwarded, and of the
// log location that is.
func TestStrataCurrentForkFieldValidation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		key    string
		value  func(root string) any
		expect string
	}{
		{"port zero", "port", func(string) any { return 0 }, "is not a reviewed value"},
		{"port too large", "port", func(string) any { return 70000 }, "is not a reviewed value"},
		{"port text", "port", func(string) any { return "8080" }, "Unsupported or malformed"},
		{"host not loopback", "host", func(string) any { return "0.0.0.0" }, "is not a reviewed value"},
		{"browser opening", "open_browser", func(string) any { return true }, "is not a reviewed value"},
		{"empty receipt path", "build_metadata", func(string) any { return "" }, "is not a reviewed value"},
		{"negative gpu", "gpu", func(string) any { return -1 }, "is not a reviewed value"},
		{"large gpu", "gpu", func(string) any { return 1000 }, "is not a reviewed value"},
		{"relative log", "log", func(string) any { return "strata.log" }, "Strata log"},
		{"empty log", "log", func(string) any { return "" }, "Strata log"},
		{"unclean log", "log", func(root string) any {
			return root + string(filepath.Separator) + "." + string(filepath.Separator) + "strata.log"
		}, "Strata log"},
		{"log in a missing directory", "log", func(root string) any { return filepath.Join(root, "absent", "strata.log") }, "Strata log"},
		{"log over a declared file", "log", func(root string) any { return filepath.Join(root, "server.py") }, "Strata log"},
		{"log in an existing directory", "log", func(root string) any { return filepath.Join(root, "strata.log") }, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newStrataServingFixture(t)
			base := forkBaselineRaw(t, f)
			base[tc.key] = tc.value(f.root)
			requireForkOutcome(t, f.deriveRaw(t, base), tc.expect)
		})
	}
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
