// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SupportedPlatform } from '@/shared/types/platform'

export type ModularProcessName =
    | 'nvpair-proxy'
    | 'broker'
    | 'node-info'
    | 'scanner'
    | 'manual-nodes'
    | 'node-settings'
    | 'engine-manager'
    | 'workload-manager'
    | 'cluster-manager'
    | 'errors'
    | 'job-scheduler'

export type ModularPackageArch = 'x64' | 'arm64'

/**
 * Who is responsible for spawning a modular binary at runtime:
 *
 * - `'broker'` — the binary is a worker the `nvpair-ui-broker` supervises. Electron
 *   passes its path to the broker (e.g. `--proxy-path`) but never spawns it
 *   directly. The broker owns its lifecycle, readiness, and log fan-out.
 * - `'service'` — `nvpair-service` spawns and supervises the binary, and Electron
 *   attaches to the service over its local socket. Only the broker is launched
 *   this way; it is then the parent of every broker-owned worker. Electron
 *   itself launches only `nvpair-service`, and only when nothing answers.
 *
 * As the broker absorbs more workers, flip the owner from `'service'` to
 * `'broker'` — never run a worker from both owners at once.
 */
type ModularLaunchOwner = 'broker' | 'service'

interface ModularRuntimeBinary {
    processName: ModularProcessName
    baseName: string
    args: string[]
    launchOwner: ModularLaunchOwner
    needsFirewallAccess: boolean
    /**
     * When true, the supervisor skips spawning this binary if its file is
     * missing instead of throwing. Binaries are built from the sibling
     * `services/` tree (`build:modular-binaries`), so all sources are always
     * present; this flag only guards a missing/failed build of a non-critical
     * worker.
     */
    optional?: boolean
}

export const MODULAR_RUNTIME_BINARIES: ModularRuntimeBinary[] = [
    // Broker-owned workers. The broker spawns and supervises every one of these;
    // Electron only passes their resolved paths to the broker (see
    // `brokerStartupArgs` in modular-supervisor.ts) and never spawns them.
    {
        // One reverse-proxy process fronts every engine: the broker starts it
        // once and enables a facade per engine over `facade/enable`, so this
        // single entry is the whole proxy tier. Each facade binds its HTTP
        // listener on all interfaces, so it needs firewall access to be
        // reachable.
        //
        // Named for the binary, which is also the process: there is one, and
        // naming it after an engine would claim a per-engine process that does
        // not exist.
        // The per-engine identities are the relay sources (`ollama-proxy` /
        // `lmstudio-proxy`), which live in proxy-engines.ts.
        processName: 'nvpair-proxy',
        baseName: 'nvpair-proxy',
        args: [],
        launchOwner: 'broker',
        needsFirewallAccess: true
    },
    {
        processName: 'scanner',
        baseName: 'nvpair-node-scanner',
        args: [],
        launchOwner: 'broker',
        needsFirewallAccess: true
    },
    {
        processName: 'node-info',
        baseName: 'nvpair-node-info',
        args: [],
        launchOwner: 'broker',
        needsFirewallAccess: true
    },
    {
        processName: 'workload-manager',
        baseName: 'nvpair-workload-manager',
        args: [],
        launchOwner: 'broker',
        needsFirewallAccess: true,
        optional: true
    },
    {
        processName: 'cluster-manager',
        baseName: 'nvpair-cluster-manager',
        args: [],
        launchOwner: 'broker',
        needsFirewallAccess: true,
        optional: true
    },
    {
        processName: 'node-settings',
        baseName: 'nvpair-node-settings',
        args: [],
        launchOwner: 'broker',
        needsFirewallAccess: false
    },
    {
        // Client only: dials remote nodes' endpoints to probe manually added
        // hosts. It opens no network listener, so it needs no firewall access.
        processName: 'manual-nodes',
        baseName: 'nvpair-manual-nodes',
        args: [],
        launchOwner: 'broker',
        needsFirewallAccess: false,
        optional: true
    },
    {
        // Serves its engine model list at `/v1/models` on all interfaces (the
        // broker starts it with `--http-port` so peers can enrich discovery from
        // it), so it needs firewall access.
        processName: 'engine-manager',
        baseName: 'nvpair-engine-manager',
        args: [],
        launchOwner: 'broker',
        needsFirewallAccess: true,
        optional: true
    },
    {
        // The broker spawns nvpair-errors with `--peer-sync` itself, so PAIR UI
        // passes only the path (`--errors-path`).
        processName: 'errors',
        baseName: 'nvpair-errors',
        args: [],
        launchOwner: 'broker',
        needsFirewallAccess: true,
        optional: true
    },
    {
        // Read-only routing policy: the broker fans it the discovery + workload
        // streams, and it emits `schedule:priority` which the broker fans out to
        // the proxies via `node/set-priority` (broker-internal — never surfaces
        // to the PAIR UI bridge). No network listener; stdio only.
        processName: 'job-scheduler',
        baseName: 'nvpair-job-scheduler',
        args: [],
        launchOwner: 'broker',
        needsFirewallAccess: false,
        optional: true
    },
    // Service-owned: only the broker. nvpair-service spawns it over stdio
    // JSON-RPC and Electron reaches it through the service; it is the parent of
    // every worker above.
    {
        processName: 'broker',
        baseName: 'nvpair-ui-broker',
        args: [],
        launchOwner: 'service',
        needsFirewallAccess: false
    }
]

/** The per-user process that owns `nvpair-ui-broker` for its clients and outlives them. */
export const SERVICE_BINARY_BASE_NAME = 'nvpair-service'

/**
 * Binaries bundled in the installer but not supervised by the broker.
 * `nvpair-service` is the one Electron launches (detached, only when nothing
 * answers on its endpoint) and attaches to; `nvpair-tui` is a headless terminal
 * client that attaches to it the same way — see `services/nvpair-service/README.md`
 * and `services/nvpair-tui/README.md`.
 */
export const MODULAR_BUNDLED_BINARIES: { baseName: string }[] = [
    { baseName: 'nvpair-tui' },
    { baseName: SERVICE_BINARY_BASE_NAME }
]

/** Every backend binary shipped in the installer (runtime workers + bundled tools). */
export function modularShippedBinaryBaseNames(): string[] {
    return [
        ...MODULAR_RUNTIME_BINARIES.map(binary => binary.baseName),
        ...MODULAR_BUNDLED_BINARIES.map(binary => binary.baseName)
    ]
}

export function modularFirewallBinaryBaseNames(): string[] {
    return MODULAR_RUNTIME_BINARIES.filter(binary => binary.needsFirewallAccess).map(
        binary => binary.baseName
    )
}

export function modularBinaryFileName(baseName: string, platform: SupportedPlatform): string {
    return platform === 'win32' ? `${baseName}.exe` : baseName
}
