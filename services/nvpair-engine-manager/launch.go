// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"runtime"
	"sort"
	"strings"

	"nvpair-shared/secretenv"
)

// launchCommand contains exactly what is handed to exec.Command. Keeping
// resolution here gives the settings preview and actual launch one builder.
// Env is the explicitly configured environment, never the inherited one.
type launchCommand struct {
	Bin  string
	Args []string
	Env  map[string]string
}

func environmentKey(key string) string {
	if runtime.GOOS == "windows" {
		return strings.ToUpper(key)
	}
	return key
}

func validEnvironmentKey(key string) bool {
	if key == "" {
		return false
	}
	for i, c := range key {
		if c != '_' && !(c >= 'A' && c <= 'Z') && !(c >= 'a' && c <= 'z') && !(i > 0 && c >= '0' && c <= '9') {
			return false
		}
	}
	return true
}

// reservedEnvironmentKey reports whether key is in the provider-secret
// namespace, whatever its case. Engines never receive provider secrets, so
// launch settings may not assign one; childEnv would drop it anyway.
func reservedEnvironmentKey(key string) bool {
	return strings.HasPrefix(strings.ToUpper(key), secretenv.Prefix)
}

func literalEnvironment(assignments []string) (map[string]string, error) {
	if _, err := formatLaunchText(append([]string{"environment"}, assignments...)); err != nil {
		return nil, err
	}
	env := make(map[string]string, len(assignments))
	seen := make(map[string]bool, len(assignments))
	for _, assignment := range assignments {
		key, value, ok := strings.Cut(assignment, "=")
		if !ok || !validEnvironmentKey(key) {
			return nil, fmt.Errorf("environment variable names must start with a letter or underscore and contain only letters, digits, or underscores")
		}
		if reservedEnvironmentKey(key) {
			return nil, fmt.Errorf("environment variable %s is reserved for provider credentials and cannot be set for an engine", key)
		}
		if seen[environmentKey(key)] {
			return nil, fmt.Errorf("an environment variable is assigned more than once")
		}
		seen[environmentKey(key)] = true
		env[key] = value
	}
	return env, nil
}

// Share native argument boundaries between default merging and typed conversion.
// A flag's value is never interpreted again as an independent native option.
func llamaLaunchArgument(token string) (string, int) {
	name, _, attached := strings.Cut(token, "=")
	name = llamaLongOption(name)
	for _, option := range llamaServingOptions {
		for _, flag := range option.flags {
			if name == flag {
				if option.bool {
					return option.key, 0
				}
				values := 1
				if option.key == "control-vector-layer-range" {
					values = 2
				}
				if attached {
					values--
				}
				return option.key, values
			}
		}
	}
	// Other pinned native server options take one value, except these
	// switches (7fe450e1 common/arg.cpp). Consume values even when they
	// look like flags; e.g. --log-file --models-preset is a filename.
	switch name {
	case "-h", "--help", "--usage", "--version", "-cl", "--cache-list", "--completion-bash",
		"--perf", "--no-perf", "-e", "--escape", "--no-escape", "-sp", "--special", "--spm-infill",
		"--ignore-eos", "--list-devices", "--check-tensors", "--reuse-port",
		"--cors-credentials", "--no-cors-credentials", "--ui-mcp-proxy", "--webui-mcp-proxy", "--no-ui-mcp-proxy", "--no-webui-mcp-proxy",
		"-ag", "--agent", "-no-ag", "--no-agent", "--ui", "--webui", "--no-ui", "--no-webui",
		"--embedding", "--embeddings", "--rerank", "--reranking", "--metrics", "--props", "--slots", "--no-slots",
		"--jinja", "--no-jinja", "--reasoning-preserve", "--no-reasoning-preserve", "--skip-chat-parsing", "--no-skip-chat-parsing",
		"--prefill-assistant", "--no-prefill-assistant", "--lora-init-without-apply", "--log-disable", "--log-jsonl", "--no-log-jsonl",
		"-v", "--verbose", "--log-verbose", "--offline", "--log-prefix", "--no-log-prefix", "--log-timestamps", "--no-log-timestamps":
		return name, 0
	}
	if attached || !strings.HasPrefix(name, "-") {
		return name, 0
	}
	return name, 1
}

// fixedLaunchArgs preserves startup defaults unless a custom native llama
// router launch lets its preset (or an explicit user option) own the setting.
func fixedLaunchArgs(rt Runtime) []string {
	if rt.EditableLaunch == nil {
		return nil
	}
	fixed := rt.EditableLaunch.FixedArgs
	if rt.LaunchArgs == nil {
		return fixed
	}
	flags := func(args []string) map[string]bool {
		found := map[string]bool{}
		for i := 0; i < len(args); i++ {
			key, values := llamaLaunchArgument(args[i])
			if strings.HasPrefix(args[i], "-") {
				found[key] = true
			}
			i += values
		}
		return found
	}
	supplied := flags(*rt.LaunchArgs)
	if !supplied["models-preset"] && !flags(fixed)["models-preset"] {
		return fixed
	}
	var result []string
	for i := 0; i < len(fixed); i++ {
		key, values := llamaLaunchArgument(fixed[i])
		end := min(i+1+values, len(fixed))
		// All four bundled defaults are model options, not router options.
		perModel := key == "ctx-size" || key == "parallel" || key == "sleep-idle-seconds" || key == "cache-ram"
		if !perModel && !supplied[key] {
			result = append(result, fixed[i:end]...)
		}
		i = end - 1
	}
	return result
}

// argumentText omits the manifest-owned executable and effective startup prefix.
func (command launchCommand) argumentText(rt Runtime) (string, error) {
	fixed := fixedLaunchArgs(rt)
	if rt.EditableLaunch == nil || len(command.Args) < len(fixed) {
		return "", fmt.Errorf("engine has no editable argument definition")
	}
	return formatLaunchParts(command.Env, command.Args[len(fixed):])
}

func formatLaunchParts(environment map[string]string, args []string) (string, error) {
	keys := make([]string, 0, len(environment))
	for key := range environment {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	tokens := make([]string, 0, len(keys)+1)
	for _, key := range keys {
		// Quote assignment values consistently, including simple values, so
		// normalization does not appear to remove a user's environment quotes.
		value := strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(environment[key])
		tokens = append(tokens, key+`="`+value+`"`)
	}
	argv, err := formatLaunchText(args)
	if err != nil {
		return "", err
	}
	if argv != "" {
		tokens = append(tokens, argv)
	}
	text := strings.Join(tokens, " ")
	if _, err := parseLaunchText(text); err != nil {
		return "", err
	}
	return text, nil
}

func resolveProcessLaunch(rt Runtime, binPath string, vars map[string]string) (launchCommand, error) {
	if binPath == "" {
		bin, err := resolvePlaceholders(rt.Bin, vars)
		if err != nil {
			return launchCommand{}, err
		}
		binPath = expandPath(bin)
	}
	// Resolution must not change the caller's context (in particular {bin}).
	resolvedVars := make(map[string]string, len(vars)+1)
	for key, value := range vars {
		resolvedVars[key] = value
	}
	resolvedVars["bin"] = binPath
	args, err := resolveArgs(rt.Args, resolvedVars)
	if err != nil {
		return launchCommand{}, err
	}
	env := make(map[string]string, len(rt.Env))
	for key, value := range rt.Env {
		resolved, err := resolvePlaceholders(value, resolvedVars)
		if err != nil {
			return launchCommand{}, err
		}
		// Not path-expanded, here or in applyLiteralLaunch, so a default launch
		// and a customised one produce the same environment. A manifest env
		// value that names a directory therefore has to arrive already
		// absolute: {models_dir} does, because the executor expands it once when
		// it resolves the engine's state.
		env[key] = resolved
	}
	return applyLiteralLaunch(rt, launchCommand{Bin: binPath, Args: args, Env: env}, resolvedVars)
}

// resolveCommandLaunch preserves manifest expansion for trusted start steps.
// Future user arguments must be added after this step so their values remain
// literal. They must never be placed into the manifest template array.
func resolveCommandLaunch(template []string, vars map[string]string) (launchCommand, error) {
	argv, err := resolveArgs(template, vars)
	if err != nil {
		return launchCommand{}, err
	}
	if len(argv) == 0 {
		return launchCommand{}, nil
	}
	for i := range argv {
		argv[i] = expandPath(argv[i])
	}
	if argv[0] == "" {
		return launchCommand{}, fmt.Errorf("start command executable is empty")
	}
	return launchCommand{Bin: argv[0], Args: argv[1:]}, nil
}
