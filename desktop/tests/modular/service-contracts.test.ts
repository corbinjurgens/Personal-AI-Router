// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync, type writeFileSync } from 'node:fs'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { buildReports, extractSurface } from '../../scripts/verify-service-contracts'
import { trackTmpDir, untrackTmpDir } from '../fixtures/isolation'

const importEffects = vi.hoisted(() => {
    const argv = [...process.argv]
    process.argv.push('--write')
    return {
        argv,
        log: vi.spyOn(console, 'log').mockImplementation(() => {}),
        exit: vi.spyOn(process, 'exit').mockImplementation(() => {
            throw new Error('Importing the contract verifier must not exit the process')
        }),
        write: vi.fn<typeof writeFileSync>()
    }
})

vi.mock('node:fs', async importOriginal => {
    const original = await importOriginal<typeof import('node:fs')>()
    return { ...original, writeFileSync: importEffects.write }
})

const importCalls = {
    log: importEffects.log.mock.calls.length,
    exit: importEffects.exit.mock.calls.length,
    write: importEffects.write.mock.calls.length
}
process.argv = importEffects.argv
importEffects.log.mockRestore()
importEffects.exit.mockRestore()

const fixtureRoots = new Set<string>()
const fixturePrefix = path.join(tmpdir(), 'pair-service-contracts-')
afterEach(async () => {
    for (const root of fixtureRoots) {
        if (path.dirname(root) !== path.resolve(tmpdir()) || !root.startsWith(fixturePrefix)) {
            throw new Error(`Unexpected contract fixture cleanup path: ${root}`)
        }
        await rm(root, { recursive: true, force: true, maxRetries: 3 })
        untrackTmpDir(root)
        fixtureRoots.delete(root)
    }
})

async function fixture(files: Record<string, string>): Promise<string> {
    const root = await mkdtemp(fixturePrefix)
    fixtureRoots.add(root)
    trackTmpDir(root)
    for (const [relative, source] of Object.entries(files)) {
        const file = path.join(root, relative)
        await mkdir(path.dirname(file), { recursive: true })
        await writeFile(file, source)
    }
    return path.join(root, 'component')
}

describe('service contract extraction', () => {
    it('imports without running the CLI even when --write is present', () => {
        expect(importCalls).toEqual({ log: 0, exit: 0, write: 0 })
    })

    it('preserves literal and package-local requests, notifications and dynamic sites', async () => {
        const dir = await fixture({
            'component/go.mod': 'module example.test/component\n',
            'component/methods.go': `package main
const LocalRequest = "local:request"
const LocalNotification = "local:changed"
`,
            'component/main.go': `package main
func handle(method string) {
    switch method {
    case "literal:request", LocalRequest:
    }
    notify("literal:changed", nil)
    codec.Notify(LocalNotification, nil)
    codec.NotifyContext(ctx, LocalNotification, nil)
    notify(method, nil)
    notify("engine:" + suffix, nil)
    signal.Notify(ch, interrupt)
    hub.Broadcast(ctx, frame)
}
`,
            'component/ignored_test.go': 'package main\nconst LocalRequest = "test:request"\n',
            'component/testdata/ignored.go':
                'package main\nconst LocalRequest = "fixture:request"\n'
        })

        const surface = extractSurface(dir)
        expect([...surface.handles].sort()).toEqual(['literal:request', 'local:request'])
        expect([...surface.emits].sort()).toEqual(['literal:changed', 'local:changed'])
        expect(surface.dynamic).toEqual([
            { label: 'method (var)', file: 'main.go', line: 9 },
            { label: 'engine:*', file: 'main.go', line: 10 }
        ])
    })

    it('resolves imported constants through local replacements and file-specific aliases', async () => {
        const dir = await fixture({
            'component/go.mod': `module example.test/component
replace (
    example.test/shared => ../shared
)
`,
            'component/main.go': `package main
import (
    lease "example.test/shared/nodepolicy"
    other "example.test/shared/other"
)
func handle(method string) {
    switch method {
    case lease.MethodRenewEgressLease, other.MethodRenewEgressLease:
    }
    notify(lease.Changed, nil)
}
`,
            'component/other.go': `package main
import lease "example.test/shared/other"
func handleOther(method string) {
    switch method {
    case lease.MethodRevokeEgressLease:
    }
}
`,
            'shared/nodepolicy/wire.go': `package nodepolicy
const MethodRenewEgressLease = "node/renew-egress-lease"
const MethodRevokeEgressLease = "node/revoke-egress-lease"
const Changed = "node/lease-changed"
`,
            'shared/other/wire.go': `package other
const MethodRenewEgressLease = "other:renew"
const MethodRevokeEgressLease = "other:revoke"
`,
            'shared/unrelated/wire.go': `package unrelated
const MethodRenewEgressLease = "unrelated:renew"
`
        })

        const surface = extractSurface(dir)
        expect([...surface.handles].sort()).toEqual([
            'node/renew-egress-lease',
            'other:renew',
            'other:revoke'
        ])
        expect([...surface.emits]).toEqual(['node/lease-changed'])
        expect(surface.dynamic).toEqual([])
    })

    it('uses the declared package name for single imports within the current module', async () => {
        const dir = await fixture({
            'component/go.mod': 'module example.test/component\n',
            'component/main.go': `package main
import "example.test/component/wirev2"
func handle(method string) {
    switch method {
    case nodepolicy.MethodRenewEgressLease:
    }
}
`,
            'component/wirev2/wire.go': `package nodepolicy
const MethodRenewEgressLease = "node/renew-egress-lease"
`
        })

        expect([...extractSurface(dir).handles]).toEqual(['node/renew-egress-lease'])
    })

    it('does not resolve selectors through unrelated local or imported constant names', async () => {
        const dir = await fixture({
            'component/go.mod': `module example.test/component
replace example.test/shared => ../shared
`,
            'component/main.go': `package main
import lease "example.test/shared/nodepolicy"
const MethodRenewEgressLease = "local:collision"
func handle(method string) {
    value := struct { MethodRenewEgressLease string }{"opaque"}
    switch method {
    case value.MethodRenewEgressLease, lease.MethodRenewEgressLease:
    }
}
`,
            'component/other.go': `package main
import lease "example.test/shared/other"
func handleOther(method string) {
    switch method {
    case lease.MethodRevokeEgressLease:
    }
}
`,
            'component/unrelated/wire.go': `package unrelated
const MethodRenewEgressLease = "subpackage:collision"
`,
            'shared/nodepolicy/wire.go': `package nodepolicy
const MethodRenewEgressLease = "node/renew-egress-lease"
const MethodRevokeEgressLease = "node/revoke-egress-lease"
`,
            'shared/other/wire.go': `package other
var MethodRevokeEgressLease string
`
        })

        expect([...extractSurface(dir).handles]).toEqual(['node/renew-egress-lease'])
    })

    it.each([
        { name: 'line comment', comment: '// A closing ) belongs to the comment.' },
        { name: 'block comment', comment: '/* A closing ) belongs to the comment. */' },
        { name: 'multiline block comment', comment: '/*\n)\n*/' }
    ])('resolves imports after a closing parenthesis in a $name', async ({ comment }) => {
        const dir = await fixture({
            'component/go.mod': `module example.test/component
replace example.test/shared => ../shared
`,
            'component/main.go': `package main
import (
    ${comment}
    lease "example.test/shared/nodepolicy"
)
func handle(method string) {
    switch method {
    case lease.MethodRenewEgressLease:
    }
    notify(lease.Changed, nil)
}
`,
            'shared/nodepolicy/wire.go': `package nodepolicy
const MethodRenewEgressLease = "node/renew-egress-lease"
const Changed = "node/lease-changed"
`
        })

        const surface = extractSurface(dir)
        expect([...surface.handles]).toEqual(['node/renew-egress-lease'])
        expect([...surface.emits]).toEqual(['node/lease-changed'])
        expect(surface.dynamic).toEqual([])
    })

    it('keeps three-part selectors unresolved without matching their constant-name tails', async () => {
        const dir = await fixture({
            'component/go.mod': `module example.test/component
replace example.test/shared => ../shared
`,
            'component/main.go': `package main
import lease "example.test/shared/nodepolicy"
const MethodRenewEgressLease = "local:request-collision"
const Changed = "local:changed-collision"
func handle(method string) {
    switch method {
    case holder.lease.MethodRenewEgressLease, lease.MethodRenewEgressLease:
    }
    notify(holder.lease.Changed, nil)
    codec.NotifyContext(ctx, holder.lease.Changed, nil)
}
`,
            'shared/nodepolicy/wire.go': `package nodepolicy
const MethodRenewEgressLease = "node/renew-egress-lease"
const Changed = "node/lease-changed"
`
        })

        const surface = extractSurface(dir)
        expect([...surface.handles]).toEqual(['node/renew-egress-lease'])
        expect([...surface.emits]).toEqual([])
        expect(surface.dynamic).toEqual([
            { label: 'holder.lease.Changed (var)', file: 'main.go', line: 9 },
            { label: 'holder.lease.Changed (var)', file: 'main.go', line: 10 }
        ])
    })

    it('leaves external and standard-library imports unresolved despite local name collisions', async () => {
        const dir = await fixture({
            'component/go.mod': 'module example.test/component\n',
            'component/main.go': `package main
import (
    "net/http"
    remote "example.net/contracts"
)
const Changed = "local:changed-collision"
const MethodGet = "local:request-collision"
func handle(method string) {
    switch method {
    case "literal:request", http.MethodGet, remote.MethodGet:
    }
    notify("literal:changed", nil)
    notify(http.MethodGet, nil)
    notify(remote.Changed, nil)
}
`
        })

        const surface = extractSurface(dir)
        expect([...surface.handles]).toEqual(['literal:request'])
        expect([...surface.emits]).toEqual(['literal:changed'])
        expect(surface.dynamic).toEqual([
            { label: 'http.MethodGet (var)', file: 'main.go', line: 13 },
            { label: 'remote.Changed (var)', file: 'main.go', line: 14 }
        ])
    })

    it('does not bind dot or underscore imports to local or package-qualified names', async () => {
        const dir = await fixture({
            'component/go.mod': `module example.test/component
replace example.test/shared => ../shared
`,
            'component/main.go': `package main
import (
    . "example.test/shared/nodepolicy"
    _ "example.test/shared/other"
)
func handle(method string) {
    switch method {
    case MethodRenewEgressLease, nodepolicy.MethodRenewEgressLease, other.MethodRevokeEgressLease:
    }
    notify(Changed, nil)
    notify(nodepolicy.Changed, nil)
    notify(other.Changed, nil)
}
`,
            'shared/nodepolicy/wire.go': `package nodepolicy
const MethodRenewEgressLease = "node/renew-egress-lease"
const Changed = "node/lease-changed"
`,
            'shared/other/wire.go': `package other
const MethodRevokeEgressLease = "other:revoke"
const Changed = "other:changed"
`
        })

        const surface = extractSurface(dir)
        expect([...surface.handles]).toEqual([])
        expect([...surface.emits]).toEqual([])
        expect(surface.dynamic).toEqual([
            { label: 'Changed (var)', file: 'main.go', line: 10 },
            { label: 'nodepolicy.Changed (var)', file: 'main.go', line: 11 },
            { label: 'other.Changed (var)', file: 'main.go', line: 12 }
        ])
    })

    it('requires an exact module name or slash boundary before resolving local replacements', async () => {
        const dir = await fixture({
            'component/go.mod': `module example.test/component
replace example.test/shared => ../shared
`,
            'component/main.go': `package main
import (
    root "example.test/shared"
    lease "example.test/shared/nodepolicy"
    lookalike "example.test/shared2/nodepolicy"
)
func handle(method string) {
    switch method {
    case root.Request, lease.MethodRenewEgressLease, lookalike.MethodRenewEgressLease:
    }
    notify(lookalike.Changed, nil)
}
`,
            'shared/wire.go': `package shared
const Request = "shared:root"
`,
            'shared/nodepolicy/wire.go': `package nodepolicy
const MethodRenewEgressLease = "node/renew-egress-lease"
`,
            'shared/2/nodepolicy/wire.go': `package nodepolicy
const MethodRenewEgressLease = "wrong-prefix:request"
const Changed = "wrong-prefix:changed"
`,
            'shared2/nodepolicy/wire.go': `package nodepolicy
const MethodRenewEgressLease = "unreplaced:request"
const Changed = "unreplaced:changed"
`
        })

        const surface = extractSurface(dir)
        expect([...surface.handles].sort()).toEqual(['node/renew-egress-lease', 'shared:root'])
        expect([...surface.emits]).toEqual([])
        expect(surface.dynamic).toEqual([
            { label: 'lookalike.Changed (var)', file: 'main.go', line: 11 }
        ])
    })

    it('labels unresolved imported and unknown notification selectors with their full names', async () => {
        const dir = await fixture({
            'component/go.mod': `module example.test/component
replace example.test/shared => ../shared
`,
            'component/main.go': `package main
import lease "example.test/shared/nodepolicy"
const Changed = "local:changed-collision"
func handle() {
    notify(lease.Changed, nil)
    notify(missing.Changed, nil)
    codec.NotifyContext(ctx, lease.Unknown, nil)
}
`,
            'shared/nodepolicy/wire.go': `package nodepolicy
const Changed = "node/lease-changed"
`
        })

        const surface = extractSurface(dir)
        expect([...surface.handles]).toEqual([])
        expect([...surface.emits]).toEqual(['node/lease-changed'])
        expect(surface.dynamic).toEqual([
            { label: 'missing.Changed (var)', file: 'main.go', line: 6 },
            { label: 'lease.Unknown (var)', file: 'main.go', line: 7 }
        ])
    })

    it('extracts both real proxy lease methods as ignored private requests', () => {
        const desktop = fileURLToPath(new URL('../../', import.meta.url))
        const services = path.resolve(desktop, '..', 'services')
        const exceptions: { ignoredMethods: Record<string, string> } = JSON.parse(
            readFileSync(path.join(desktop, 'docs/service-contract-exceptions.json'), 'utf8')
        )
        const methods = ['node/renew-egress-lease', 'node/revoke-egress-lease']
        const surface = extractSurface(path.join(services, 'nvpair-proxy'))
        expect(methods.map(method => surface.handles.has(method))).toEqual([true, true])

        const { reports, drift } = buildReports(services, exceptions, { 'nvpair-proxy': '' })
        expect(reports).toHaveLength(1)
        expect(reports[0].rows.filter(row => methods.includes(row.method))).toEqual([
            {
                method: 'node/renew-egress-lease',
                direction: 'request',
                referenced: false,
                ignored: true
            },
            {
                method: 'node/revoke-egress-lease',
                direction: 'request',
                referenced: false,
                ignored: true
            }
        ])
        expect(
            drift.unusedRequests.filter(method => methods.some(lease => method.endsWith(lease)))
        ).toEqual([])
    })

    it('extracts internally consumed notifications as ignored rows at their emitting components', () => {
        const desktop = fileURLToPath(new URL('../../', import.meta.url))
        const services = path.resolve(desktop, '..', 'services')
        const exceptions: { ignoredMethods: Record<string, string> } = JSON.parse(
            readFileSync(path.join(desktop, 'docs/service-contract-exceptions.json'), 'utf8')
        )
        const expected = [
            { dir: 'nvpair-node-info', method: 'nodeinfo:observed-addresses' },
            { dir: 'nvpair-node-scanner', method: 'discovery:node-removed' },
            { dir: 'nvpair-node-scanner', method: 'discovery:node-telemetry' },
            { dir: 'nvpair-node-scanner', method: 'discovery:node-updated' },
            { dir: 'nvpair-proxy', method: 'admission/state' },
            { dir: 'nvpair-proxy', method: 'admission/strict-readiness' },
            { dir: 'nvpair-ui-broker', method: 'discovery:node-activity' },
            { dir: 'nvpair-ui-broker', method: 'scheduler:telemetry' }
        ]
        const versions: Record<string, string> = {}
        for (const { dir } of expected) versions[dir] = ''

        const { reports } = buildReports(services, exceptions, versions)
        const rows = reports.flatMap(report =>
            report.rows
                .filter(row =>
                    expected.some(item => item.dir === report.dir && item.method === row.method)
                )
                .map(row => ({ dir: report.dir, ...row }))
        )
        expect(rows).toEqual(
            expected.map(item => ({
                ...item,
                direction: 'notification',
                referenced: false,
                ignored: true
            }))
        )
    })
})
