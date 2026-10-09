// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Command nvpair-service is the per-user, long-running owner of
// nvpair-ui-broker. It spawns the broker over stdio exactly as the desktop app
// and the terminal UI used to, restarts it if it dies, and lets any number of
// clients attach to and detach from it over a local socket or named pipe.
// Quitting a client leaves the service, and the inference it serves, running.
//
// Usage:
//
//	nvpair-service [flags] [-- broker args]
//	nvpair-service status
//	nvpair-service stop
//	nvpair-service autostart enable [flags] [-- broker args]
//	nvpair-service autostart disable|status
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"nvpair-shared/appdir"
	"nvpair-shared/applog"
	"nvpair-shared/servicectl"
)

// Version is stamped at build time via -ldflags "-X main.Version=...".
var Version = "dev"

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		switch args[0] {
		case "autostart":
			if err := runAutostart(args[1:], stdout); err != nil {
				fmt.Fprintln(stderr, "nvpair-service:", err)
				return 1
			}
			return 0
		case "status":
			return runStatus(stdout, stderr)
		case "stop":
			return runStop(stdout, stderr)
		}
	}

	opts, err := parseRunArgs(args, stderr)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		fmt.Fprintln(stderr, "nvpair-service:", err)
		return 2
	}
	if opts.showVersion {
		fmt.Fprintln(stdout, Version)
		return 0
	}
	return serve(opts, stdout)
}

// runOptions are the flags of the default (serve) command.
type runOptions struct {
	showVersion bool
	brokerPath  string
	endpoint    string
	logDir      string
	logLevel    slog.Level
	brokerArgs  []string
}

func newRunFlags(stderr io.Writer) (*flag.FlagSet, *runOptions, func() slog.Level, func() bool) {
	fs := flag.NewFlagSet("nvpair-service", flag.ContinueOnError)
	fs.SetOutput(stderr)
	opts := &runOptions{}
	fs.BoolVar(&opts.showVersion, "version", false, "print version and exit")
	fs.StringVar(&opts.brokerPath, "broker-path", "", "path to nvpair-ui-broker (default: beside this executable)")
	fs.StringVar(&opts.endpoint, "endpoint", "", "socket or named pipe to listen on (default: <appdir>/service.sock, or \\\\.\\pipe\\nvpair-service-<user> on Windows; $"+servicectl.EndpointEnv+" overrides)")
	fs.StringVar(&opts.logDir, "log-dir", "", "directory for broker.log and service.log (default: <appdir>/logs)")
	resolveLevel := applog.RegisterFlag(fs, slog.LevelInfo)
	levelSet := func() bool {
		set := false
		fs.Visit(func(f *flag.Flag) {
			if f.Name == "log-level" {
				set = true
			}
		})
		return set
	}
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: nvpair-service [flags] [-- broker args]")
		fmt.Fprintln(stderr, "       nvpair-service status | stop")
		fmt.Fprintln(stderr, "       nvpair-service autostart enable [flags] [-- broker args] | disable | status")
		fs.PrintDefaults()
	}
	return fs, opts, resolveLevel, levelSet
}

// parseRunArgs parses service flags up to "--" and keeps everything after it as
// broker arguments. An explicit --log-level is also passed to the broker unless
// the broker arguments carry their own, mirroring how the broker hands its
// level to every worker.
func parseRunArgs(args []string, stderr io.Writer) (*runOptions, error) {
	serviceArgs, brokerArgs := splitArgs(args)
	fs, opts, resolveLevel, levelSet := newRunFlags(stderr)
	if err := fs.Parse(serviceArgs); err != nil {
		return nil, err
	}
	if fs.NArg() > 0 {
		return nil, fmt.Errorf("unexpected argument %q (broker arguments go after --)", fs.Arg(0))
	}
	opts.logLevel = resolveLevel()
	opts.brokerArgs = brokerArgs
	if levelSet() && !hasFlag(brokerArgs, "log-level") {
		opts.brokerArgs = append(append([]string(nil), brokerArgs...), "--log-level", applog.LevelName(opts.logLevel))
	}
	return opts, nil
}

// validateRunArgs checks a command line recorded for autostart.
func validateRunArgs(args []string) error {
	_, err := parseRunArgs(args, io.Discard)
	return err
}

func splitArgs(args []string) (service, broker []string) {
	for i, a := range args {
		if a == "--" {
			return args[:i], append([]string(nil), args[i+1:]...)
		}
	}
	return args, nil
}

// hasFlag reports whether args set the named flag in any of Go's spellings.
func hasFlag(args []string, name string) bool {
	for _, a := range args {
		trimmed := strings.TrimLeft(a, "-")
		if trimmed == a {
			continue
		}
		if trimmed == name || strings.HasPrefix(trimmed, name+"=") {
			return true
		}
	}
	return false
}

func serve(opts *runOptions, stdout io.Writer) int {
	logDir := opts.logDir
	if logDir == "" {
		dir, err := appdir.Path("logs")
		if err != nil {
			fmt.Fprintln(os.Stderr, "nvpair-service: cannot resolve the log directory:", err)
			return 1
		}
		logDir = dir
	}

	// The service's own log goes to a file as well as stderr: started detached,
	// its stderr is the null device. The file comes first so a closed terminal
	// on stderr cannot stop the file write.
	if own, err := openRotatingLog(filepath.Join(logDir, "service.log"), defaultLogMaxSize); err == nil {
		defer own.Close()
		applog.SetOutput(io.MultiWriter(own, os.Stderr))
	}
	applog.Init("nvpair-service", opts.logLevel)

	endpoint := opts.endpoint
	if endpoint == "" {
		e, err := servicectl.Endpoint()
		if err != nil {
			slog.Error("cannot resolve the service endpoint", "err", err)
			return 1
		}
		endpoint = e
	}

	brokerPath, err := resolveBrokerPath(opts.brokerPath)
	if err != nil {
		slog.Error("cannot locate the broker", "err", err)
		return 1
	}

	svc := newService(config{
		endpoint:               endpoint,
		version:                Version,
		logPath:                filepath.Join(logDir, "broker.log"),
		logMaxSize:             defaultLogMaxSize,
		spawn:                  execSpawner(brokerPath, opts.brokerArgs),
		backoff:                defaultBackoff,
		stopGrace:              defaultStopGrace,
		shutdownRequestTimeout: defaultShutdownRequestTimeout,
	})

	// A service started from a terminal outlives that terminal, as a detached
	// one does.
	signal.Ignore(syscall.SIGHUP)
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if err := svc.Run(ctx); err != nil {
		if errors.Is(err, errAlreadyRunning) {
			fmt.Fprintln(stdout, "nvpair-service: already running")
			return 0
		}
		slog.Error("service failed", "err", err)
		return 1
	}
	return 0
}
