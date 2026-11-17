# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# POSIX shortcuts for the developer commands in scripts/dev.mjs, which is the
# implementation on every platform. On Windows, run
# `node scripts/dev.mjs <command>` directly.

.DEFAULT_GOAL := help

DEV := node scripts/dev.mjs

.PHONY: help tools dev build run package check test build-services headers \
	headers-fix lint typecheck contracts clean

help: ## List available targets
	@printf 'Personal AI Router — make targets (each runs node scripts/dev.mjs)\n\n'
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN { FS = ":.*## " } { printf "  \033[1m%-16s\033[0m%s\n", $$1, $$2 }'

tools: ## Check the Go, Node.js, npm and Git versions
	$(DEV) tools

dev: ## Install the npm packages and fill the Go module cache
	$(DEV) setup

build: ## Build the service binaries and the desktop bundles
	$(DEV) build

run: ## Start the desktop app in development mode
	$(DEV) start

package: ## Build the unpacked desktop app for this platform and verify its stamp
	$(DEV) package

check: ## Run the gates: headers, build scripts, lint, typecheck, contracts, dead code, unit tests
	$(DEV) check

test: ## Run the desktop unit tests and go test in every services module
	$(DEV) test

build-services: ## Stage the standalone services bundle in services/build/bin
	$(DEV) services

headers: ## Check that every file carries the SPDX copyright and license header
	node scripts/spdx-headers.mjs

headers-fix: ## Insert the SPDX header into any file that is missing one
	$(DEV) headers-fix

lint: ## Lint the desktop sources
	cd desktop && npm run lint

typecheck: ## Typecheck the desktop node, web, and test projects
	cd desktop && npm run typecheck

contracts: ## Check the desktop and services JSON-RPC contract surface
	cd desktop && npm run service-contracts:check

clean: ## Remove built binaries, bundles, and packages
	$(DEV) clean
