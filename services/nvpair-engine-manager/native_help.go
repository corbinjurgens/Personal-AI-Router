// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"nvpair-shared/modelconfig"
)

const nativeHelpBytes = 256 << 10
const nativeHelpBudget = 5 * time.Second
const nativeHelpCacheLimit = 128

type nativeHelpEntry struct {
	done    chan struct{}
	options []modelconfig.NativeOption
}

type nativeHelpCache struct {
	mu      sync.Mutex
	entries map[string]*nativeHelpEntry
}

var nativeHelpResults nativeHelpCache

type nativeHelpInspectionKey struct{}
type nativeHelpDiscoveryKey struct{}

// lookup never starts or waits for native work. Compilation may consume only
// completed evidence for the exact selected-file content identity.
func (c *nativeHelpCache) lookup(build string) []modelconfig.NativeOption {
	c.mu.Lock()
	entry := c.entries[build]
	c.mu.Unlock()
	if entry == nil {
		return nil
	}
	select {
	case <-entry.done:
		return cloneNativeOptions(entry.options)
	default:
		return nil
	}
}

func cloneNativeOptions(options []modelconfig.NativeOption) []modelconfig.NativeOption {
	if options == nil {
		return nil
	}
	out := make([]modelconfig.NativeOption, len(options))
	for i, option := range options {
		out[i] = option
		out[i].Choices = append([]string(nil), option.Choices...)
	}
	return out
}

// A full cache refuses new discoveries rather than evicting failures and
// repeatedly initializing the same native build. A manager restart clears it.
func (c *nativeHelpCache) get(ctx context.Context, build string, read func() []modelconfig.NativeOption) []modelconfig.NativeOption {
	c.mu.Lock()
	if c.entries == nil {
		c.entries = make(map[string]*nativeHelpEntry)
	}
	entry, found := c.entries[build]
	if !found {
		if len(c.entries) >= nativeHelpCacheLimit {
			c.mu.Unlock()
			return nil
		}
		entry = &nativeHelpEntry{done: make(chan struct{})}
		c.entries[build] = entry
	}
	c.mu.Unlock()
	if !found {
		entry.options = read()
		close(entry.done)
	}
	select {
	case <-ctx.Done():
		return nil
	case <-entry.done:
		if entry.options == nil {
			return nil
		}
		return cloneNativeOptions(entry.options)
	}
}

// nativeOptions is an inspection/preview seam, never a status-polling hook.
// nil/false means unknown, including pause and failed cached discovery.
func (e *Executor) nativeOptions(ctx context.Context, st *engineState) ([]modelconfig.NativeOption, bool) {
	if st == nil {
		return nil, false
	}
	st.mu.Lock()
	engine := ""
	if st.manifest != nil {
		engine = st.manifest.Engine
	}
	st.mu.Unlock()
	if blocked, known := e.nodeProjection.managementBlocked(); !known || blocked {
		return nil, false
	}
	ctx, release := registerServingLoad(ctx, st)
	defer release()
	ctx = e.activationDispatchContext(ctx, engine)
	bin, err := nativeBuildExecutable(st)
	if err != nil {
		return nil, false
	}
	executable, err := captureSource(bin, false)
	if err != nil {
		return nil, false
	}
	selected, libraries, err := nativeBuildSnapshotExpected(ctx, executable, servingFileAccess{}, nil)
	if err != nil {
		return nil, false
	}
	if !st.opMu.TryLock() {
		return nil, false
	}
	snapshot := snapshotLlamaState(st)
	command, err := launchForState(snapshot, snapshot.plat.Runtime.Port)
	st.opMu.Unlock()
	if err != nil {
		return nil, false
	}
	env, err := modelNativeEnvironment(filepath.Dir(executable.target), command.Env)
	if err != nil {
		return nil, false
	}
	env, err = modelDeviceEnvironment(env, "")
	if err != nil {
		return nil, false
	}
	current := func() bool {
		if ctx.Err() != nil || e.activationVeto(engine) != nil || validateSource(executable) != nil {
			return false
		}
		if !st.opMu.TryLock() {
			return false
		}
		afterState := snapshotLlamaState(st)
		afterCommand, launchErr := launchForState(afterState, afterState.plat.Runtime.Port)
		st.opMu.Unlock()
		if launchErr != nil || afterState.gen != snapshot.gen || !reflect.DeepEqual(command, afterCommand) {
			return false
		}
		for _, file := range libraries {
			if validateSource(file) != nil {
				return false
			}
		}
		after, _, err := nativeBuildSnapshotExpected(ctx, executable, servingFileAccess{}, libraries)
		return err == nil && after.Build == selected.Build
	}
	options := nativeHelpResults.get(ctx, selected.Build, func() []modelconfig.NativeOption {
		if !current() {
			return nil
		}
		data, err := runNativeHelpCommand(ctx, executable.target, []string{"--help"}, env)
		if err != nil || !current() {
			return nil
		}
		return parseNativeHelp(data)
	})
	if !current() {
		return nil, false
	}
	return options, options != nil
}

func validateNativeExtraOptions(model string, extra map[string]string, options []modelconfig.NativeOption) error {
	if err := validateModelExtraOptions(model, extra); err != nil {
		return err
	}
	if options == nil {
		return nil
	}
	reported := make(map[string]bool, len(options))
	for _, option := range options {
		reported[option.Name] = true
	}
	for _, key := range sortedKeys(extra) {
		if !reported[key] {
			return modelLoadError(model, "extraOptions", "This llama.cpp build does not report --"+key+".")
		}
	}
	return nil
}

// Compare the actual launch declarations and every generated section, including
// defaults and siblings. Help spelling supplies warnings, never allocation data.
func unreportedNativeLaunchOptions(options []modelconfig.NativeOption, command launchCommand, preset []byte, loads []modelconfig.ModelLoad) ([]string, error) {
	if options == nil {
		return nil, nil
	}
	reported, missing := map[string]bool{}, map[string]bool{}
	for _, option := range options {
		reported[option.Name] = true
	}
	check := func(key string) {
		if !reported[key] {
			missing[key] = true
		}
	}
	if len(preset) != 0 {
		command = llamaSnapshotCommand(command, "", loads)
	}
	for _, arg := range command.Args {
		if strings.HasPrefix(arg, "--") {
			key, _, _ := strings.Cut(strings.TrimPrefix(arg, "--"), "=")
			if nativeHelpFlag.FindString("--"+key) == "--"+key {
				check(key)
			}
		}
	}
	if len(preset) != 0 {
		sections, err := parseLlamaServingPresetAllowing(preset, modelLoadAllowedPresetKeys(loads))
		if err != nil {
			return nil, err
		}
		for _, section := range sections {
			for key := range section {
				check(key)
			}
		}
	}
	return sortedKeys(missing), nil
}

func runNativeHelpCommand(ctx context.Context, executable string, args, env []string) (string, error) {
	deadline := time.Now().Add(nativeHelpBudget)
	if parent, ok := ctx.Deadline(); ok && parent.Before(deadline) {
		deadline = parent
	}
	ctx, cancel := context.WithDeadline(ctx, deadline.Add(-100*time.Millisecond))
	defer cancel()
	cmd := exec.CommandContext(ctx, executable, args...)
	cmd.Dir = filepath.Dir(executable)
	cmd.Env = env
	cmd.WaitDelay = 100 * time.Millisecond
	configureSysProcAttr(cmd)
	budget := &modelDeviceOutputBudget{cancel: cancel, limit: nativeHelpBytes}
	output := &modelDeviceBoundedWriter{budget: budget}
	cmd.Stdout, cmd.Stderr = output, output
	finish, err := beginActivationEffect(ctx)
	if err != nil {
		return "", err
	}
	err = cmd.Start()
	finish()
	if err == nil {
		err = cmd.Wait()
	}
	if err != nil || ctx.Err() != nil || budget.exceeded {
		return "", errors.New("native help failed/bounded")
	}
	return output.data.String(), nil
}

var nativeHelpFlag = regexp.MustCompile(`--([a-z0-9][a-z0-9-]*)`)
var nativeHelpAliases = regexp.MustCompile(`^((?:--?[a-zA-Z0-9][a-zA-Z0-9-]*[ \t]*[,/][ \t]*)*--?[a-zA-Z0-9][a-zA-Z0-9-]*)(.*)$`)
var nativeHelpChoices = regexp.MustCompile(`^\{([^{}]+)\}|^([a-zA-Z0-9_.-]+(?:\|[a-zA-Z0-9_.-]+)+)`)

func parseNativeHelp(help string) []modelconfig.NativeOption {
	options := make(map[string]modelconfig.NativeOption)
	for _, line := range strings.Split(help, "\n") {
		// Only declaration columns contribute flags, not examples in prose.
		// llama.cpp aligns wrapped help prose at column 40, including lines
		// whose first word is another option name.
		if len(line)-len(strings.TrimLeft(line, " \t")) >= 40 {
			continue
		}
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "-") {
			continue
		}
		// llama.cpp pads the first short alias before its long name. Consume
		// the alias list before looking for the declaration/prose separator.
		aliases := nativeHelpAliases.FindStringSubmatch(line)
		if aliases == nil {
			continue
		}
		if rest := aliases[2]; rest != "" && rest[0] != ' ' && rest[0] != '\t' && rest[0] != '=' {
			continue
		}
		// The formatter appends one value hint after the complete alias list.
		hint := strings.TrimSpace(strings.Trim(strings.SplitN(aliases[2], "  ", 2)[0], ", /="))
		for _, match := range nativeHelpFlag.FindAllStringSubmatchIndex(aliases[1], -1) {
			name := aliases[1][match[2]:match[3]]
			option := modelconfig.NativeOption{Name: name, Kind: "text"}
			if hint == "" {
				option.Kind = "switch"
			}
			if fields := strings.Fields(hint); len(fields) > 0 && fields[0] == "N" {
				option.Kind = "number"
			}
			if choice := nativeHelpChoices.FindStringSubmatch(hint); choice != nil {
				value := choice[1]
				if value == "" {
					value = choice[2]
				}
				option.Kind = "choice"
				for _, item := range strings.FieldsFunc(value, func(r rune) bool { return r == ',' || r == '|' }) {
					if item = strings.TrimSpace(item); item != "" {
						option.Choices = append(option.Choices, item)
					}
				}
			}
			options[name] = option
		}
	}
	if len(options) == 0 {
		return nil
	}
	for name, option := range options {
		other := "no-" + name
		if strings.HasPrefix(name, "no-") {
			other = strings.TrimPrefix(name, "no-")
		}
		if _, paired := options[other]; paired {
			option.Kind, option.Choices = "switch", nil
			options[name] = option
		}
	}
	out := make([]modelconfig.NativeOption, 0, len(options))
	for _, option := range options {
		out = append(out, option)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	// Keep bounded native output inside the strict public hint contract. An
	// unrecognized or unrepresentable grammar remains unknown/manual input.
	projection := modelconfig.InspectResult{Engine: "llamacpp", ModelID: "native-help", Status: "unknown", Estimate: modelconfig.Estimate{Status: "unknown"}, NativeOptions: out}
	if projection.Validate() != nil {
		return nil
	}
	return out
}
