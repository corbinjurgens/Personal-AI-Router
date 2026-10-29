// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

func listProcessRecords(ctx context.Context) ([]processRecord, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "ps", "-axo", "pid=,comm=")
	cmd.Env = childEnv(nil)
	out, err := cmd.Output()
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if err != nil {
		return nil, err
	}
	return parsePSRecords(out)
}

func parsePSRecords(out []byte) ([]processRecord, error) {
	if !utf8.Valid(out) {
		return nil, fmt.Errorf("invalid process encoding")
	}
	var records []processRecord
	seen := make(map[int]bool)
	for _, line := range strings.Split(strings.TrimSuffix(string(out), "\n"), "\n") {
		line = strings.TrimSpace(line)
		i := strings.IndexAny(line, " \t")
		if i < 1 {
			return nil, fmt.Errorf("partial process record")
		}
		pid, err := strconv.Atoi(line[:i])
		for _, digit := range line[:i] {
			if digit < '0' || digit > '9' {
				return nil, fmt.Errorf("invalid process PID")
			}
		}
		image := strings.TrimSpace(line[i:])
		if err != nil || pid <= 0 || image == "" || seen[pid] {
			return nil, fmt.Errorf("invalid process record")
		}
		seen[pid] = true
		records = append(records, processRecord{pid, image})
	}
	return records, nil
}

func parsePSComm(out []byte) []string {
	var names []string
	for _, line := range strings.Split(string(out), "\n") {
		if name := strings.TrimSpace(line); name != "" {
			names = append(names, name)
		}
	}
	return names
}
