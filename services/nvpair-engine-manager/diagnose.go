// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
)

var errAuthRequired = errors.New("engine requires an API key")

type connectionDiagnosis uint8

const (
	diagnosisNone connectionDiagnosis = iota
	diagnosisAuthRequired
	diagnosisAppRunning
	diagnosisNotRunning
)

func (e *Executor) diagnoseConnection(ctx context.Context, st *engineState, port int) connectionDiagnosis {
	st.mu.Lock()
	rt := st.plat.Runtime
	st.mu.Unlock()
	if len(rt.AuthStatuses) == 0 && len(rt.AppProcesses) == 0 {
		return diagnosisNone
	}
	_, status := e.probeStatus(ctx, rt.Ready, port)
	if slices.Contains(rt.AuthStatuses, status) {
		return diagnosisAuthRequired
	}
	images, err := e.processes(ctx)
	if err != nil {
		return diagnosisNone
	}
	if anyProcessMatches(images, rt.AppProcesses) {
		return diagnosisAppRunning
	}
	if len(rt.AppProcesses) > 0 {
		return diagnosisNotRunning
	}
	return diagnosisNone
}

func diagnosisMessage(displayName string, d connectionDiagnosis) string {
	switch d {
	case diagnosisAuthRequired:
		return fmt.Sprintf("%s is running but requires an API key, and PAIR does not have one. Turn off API authentication in %s's server settings, then try again.", displayName, displayName)
	case diagnosisAppRunning:
		return fmt.Sprintf("Can't connect to %s. Does it have an API key set, or is its server off?", displayName)
	case diagnosisNotRunning:
		return fmt.Sprintf("Can't connect to %s, and no %s process is running. Open %s, then try again.", displayName, displayName, displayName)
	default:
		return ""
	}
}

func (e *Executor) diagnoseStartFailure(ctx context.Context, st *engineState, port int, cause, fallback error) error {
	if ctx.Err() != nil {
		return fallback
	}
	d := diagnosisAuthRequired
	if !errors.Is(cause, errAuthRequired) {
		d = e.diagnoseConnection(ctx, st, port)
	}
	// A pause or shutdown may have cancelled a slow diagnosis. Preserve the
	// original failure and let the lifecycle's cancellation owner handle it.
	if ctx.Err() != nil || d == diagnosisNone {
		return fallback
	}
	return errors.New(diagnosisMessage(st.manifest.DisplayName, d))
}
