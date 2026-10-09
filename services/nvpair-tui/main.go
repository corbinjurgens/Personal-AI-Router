// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Command nvpair-tui is a terminal UI for nvpair-service, the per-user process
// that owns nvpair-ui-broker (and, through it, the whole NVPAIR subprocess
// fleet). It attaches to the running service, starting it detached when it is
// not running, and speaks the broker's JSON-RPC through it. Quitting detaches
// and leaves the service and its inference running. It is designed to run
// comfortably over SSH on a headless server where the bundled graphical UI
// cannot run.
//
// This file is the process entrypoint: it parses flags, initialises
// logging, attaches to the service, and runs the UI. Logging goes to
// stderr so it never collides with the full-screen TUI on stdout.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"nvpair-shared/appdir"
	"nvpair-shared/applog"
	"nvpair-tui/ui"
)

// Version is stamped at build time via -ldflags "-X main.Version=...".
// It mirrors the convention every other component in this repo uses so
// `nvpair-tui --version` reports the value from versions.json.
var Version = "dev"

func main() {
	servicePath := flag.String("service-path", "", "path to nvpair-service binary, started when no service is running (default: ./nvpair-service alongside this executable)")
	stopService := flag.Bool("stop-service", false, "stop the running nvpair-service (and with it the broker and inference) and exit")
	showVersion := flag.Bool("version", false, "print version and exit")
	appearance := flag.String("appearance", "auto", "terminal background: auto, light, or dark")
	resolveLevel := applog.RegisterFlag(nil, slog.LevelInfo)
	flag.Parse()

	if *showVersion {
		fmt.Println(Version)
		os.Exit(0)
	}

	if *stopService {
		applog.Init("nvpair-tui", resolveLevel())
		running, err := stopRunningService()
		switch {
		case err != nil:
			fmt.Fprintln(os.Stderr, "nvpair-tui: could not stop nvpair-service:", err)
			os.Exit(1)
		case running:
			fmt.Println("nvpair-service stopped")
		default:
			fmt.Println("nvpair-service is not running")
		}
		os.Exit(0)
	}

	// Started before the program, not merely before the first draw. On auto
	// this asks the terminal for its background and reads the answer, which
	// only works while stdin is still ours — once Bubble Tea is running, its
	// reader takes the reply and the query learns nothing. Joined below, after
	// the service is attached, so a terminal that never answers costs its timeout
	// alongside startup rather than in front of it.
	chosen, ok := ui.ParseAppearance(*appearance)
	if !ok {
		fmt.Fprintf(os.Stderr, "unknown --appearance %q: use auto, light, or dark\n", *appearance)
		os.Exit(2)
	}
	appearanceSettled := ui.StartAppearance(chosen)

	// Before Init, which builds its handler from the output it finds. While the
	// full-screen program runs this sends the log to the Logs tab, where it can
	// be read; stderr is hidden behind the program then.
	applog.SetOutput(ui.LogOutput())
	applog.Init("nvpair-tui", resolveLevel())

	resolvedService, err := resolveServicePath(*servicePath)
	if err != nil {
		slog.Error("cannot locate nvpair-service", "err", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		select {
		case <-sigCh:
			cancel()
		case <-ctx.Done():
		}
	}()

	link, err := connectService(ctx, resolvedService)
	if err != nil {
		slog.Error("failed to attach to nvpair-service", "err", err)
		os.Exit(1)
	}

	// The last moment the answer can be had: Bubble Tea takes stdin next, and
	// the first frame is already choosing colours with it.
	appearanceSettled()
	// Recorded because a wrong answer is visible but unexplained: the operator
	// sees colours that do not suit their terminal and has nothing telling
	// them what PAIR concluded, or that --appearance would override it.
	slog.Debug("terminal appearance", "requested", string(chosen),
		"using", string(ui.DetectedAppearance()))

	// The broker's stderr (its logs plus every worker's, prefixed), relayed by
	// the service as service/log, is fed into the UI's Logs view rather than
	// the terminal, so it never collides with the full-screen TUI on stdout.
	outcome, err := ui.Run(link.Client, link.Logs)
	if err != nil {
		slog.Error("ui error", "err", err)
	}

	// Quitting detaches: the service, the broker, and inference keep running.
	// Stopping them is the explicit choice the operator made with Q, or part
	// of a data reset.
	if outcome.StopService || outcome.WipeData {
		if err := link.stopService(); err != nil {
			slog.Error("could not stop nvpair-service", "err", err)
			if outcome.WipeData {
				// The workers still hold the files; wiping now would race them.
				slog.Error("data directory not reset because the service is still running")
				outcome.WipeData = false
			}
		}
	}
	link.detach()

	// Only now, with the service stopped and every worker joined, is the data
	// directory unowned. Wiping it while the broker ran would race a
	// shutting-down worker into recreating the files we deleted.
	if outcome.WipeData {
		wipeAppData()
	}

	slog.Info("shutdown complete")
}

// wipeAppData deletes the per-user data directory: node settings, cluster
// identity, trusted peers, and persisted ports. appdir.Dir is the single
// location every component agrees on, so there is one path to remove and no
// guessing at layout.
func wipeAppData() {
	dir, err := appdir.Dir()
	if err != nil {
		slog.Error("cannot resolve the data directory to reset", "err", err)
		return
	}
	// appdir always appends two product segments, so this cannot be a bare home
	// or root directory today. Asserted anyway: this is the one irreversible
	// path in the program, and a relative path would be resolved against
	// whatever directory the process happens to be running in.
	if !filepath.IsAbs(dir) {
		slog.Error("refusing to reset a non-absolute data directory", "dir", dir)
		return
	}
	if err := os.RemoveAll(dir); err != nil {
		slog.Error("failed to reset data directory", "dir", dir, "err", err)
		return
	}
	slog.Info("data directory reset", "dir", dir)
}
