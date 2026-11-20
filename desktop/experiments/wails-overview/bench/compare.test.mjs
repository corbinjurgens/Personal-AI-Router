// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { after, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { compare } from './compare.mjs'

const path = file => fileURLToPath(new URL(file, import.meta.url))
const expected = JSON.parse(readFileSync(path('fixtures/expected-summary.json'), 'utf8'))
const example = JSON.parse(readFileSync(path('fixtures/manifest.json'), 'utf8'))
const root = mkdtempSync(join(tmpdir(), 'pair-overview-bench-'))
after(() => rmSync(root, { recursive: true, force: true }))

function fixture() {
    const { runs: examples, ...manifest } = structuredClone(example)
    const runs = []
    for (const os of ['windows', 'macos']) {
        const template = examples.find(run => run.os === os)
        let order = 0
        const add = (scene, kind, pair, arm, counterFile) => {
            const run = {
                ...structuredClone(template),
                scene,
                kind,
                pair,
                arm,
                order: ++order,
                inputMode: 'replay',
                intervalMs: 1000,
                activeUse: scene === 'active'
            }
            if (pair === 'restart' && arm === 'wails') {
                run.fence.before.brokerRestartCount = 1
                run.fence.after.brokerRestartCount = 1
            }
            const inferenceRequests = []
            const inferenceTimings = [20, 21, 22].map((tokensPerSecond, index) => {
                const requestId = `fixture-${order}-${index}`
                inferenceRequests.push({
                    requestId,
                    attempt: 1,
                    success: true,
                    singleAttempt: true,
                    localModel: { pid: 31, createdUtc: '2026-11-20T00:00:00.000Z' }
                })
                return { requestId, attempt: 1, tokensPerSecond, firstTokenMs: 100 + 10 * index }
            })
            run.measurements = {
                warmReopenMs: [500, 520, 580],
                coldOpenMs: [1200],
                missedFramesPercent: [2, 4, 6],
                completionPercent: [100, 100, 100],
                inferenceTimings,
                inferenceRequests
            }
            const text = readFileSync(path(`fixtures/${os}-${counterFile}.txt`), 'utf8')
            const samples = text
                .trim()
                .split('\n')
                .map(line => {
                    const sample = JSON.parse(line)
                    sample.timestamp = new Date(
                        Date.parse(sample.timestamp) + 10_000 * order
                    ).toISOString()
                    return sample
                })
            runs.push({ run, samples })
        }
        for (const scene of ['visible', 'tray', 'active']) {
            add(scene, 'comparison', 'a', 'matched-electron', 'baseline')
            add(scene, 'comparison', 'a', 'wails', 'candidate')
            add(scene, 'control', 'noise-baseline', 'matched-electron', 'baseline')
            add(scene, 'control', 'noise-baseline', 'matched-electron', 'control')
            add(scene, 'control', 'noise-candidate', 'wails', 'candidate')
            add(scene, 'control', 'noise-candidate', 'wails', 'candidate')
            add(scene, 'comparison', 'b', 'wails', os === 'windows' ? 'candidate-two' : 'candidate')
            add(scene, 'comparison', 'b', 'matched-electron', 'baseline')
            add(scene, 'comparison', 'c', 'matched-electron', 'baseline')
            add(
                scene,
                'comparison',
                'c',
                'wails',
                os === 'windows' ? 'candidate-three' : 'candidate'
            )
        }
        if (os === 'windows') {
            for (const pair of ['missing', 'invalid', 'restart']) {
                add('visible', 'comparison', pair, 'matched-electron', 'baseline')
                add('visible', 'comparison', pair, 'wails', pair)
            }
        }
    }
    return { manifest, runs }
}

function writeFixture(edit = () => {}) {
    const directory = mkdtempSync(join(root, 'probe-'))
    const { manifest, runs } = fixture()
    edit(runs)
    manifest.runs = runs.map(({ run, samples }) => {
        run.file = `${run.os}-${run.order}.jsonl`
        writeFileSync(
            join(directory, run.file),
            samples.map(sample => JSON.stringify(sample)).join('\n') + '\n'
        )
        return run
    })
    const file = join(directory, 'manifest.json')
    writeFileSync(file, JSON.stringify(manifest))
    return file
}

const baseline = writeFixture()
const result = spawnSync(process.execPath, [path('compare.mjs'), baseline], {
    encoding: 'utf8',
    maxBuffer: 4 * 1024 ** 2
})
assert.equal(result.status, 0, result.stderr)
const summary = JSON.parse(result.stdout)

for (const os of ['windows', 'macos']) {
    test(`${os} verdict uses steady medians, signed pair deltas and same-host noise`, () => {
        const platform = summary.platforms[os]
        const { verdict, threshold, median, noise, range, deltas } =
            platform.metrics.visibleMemoryMiB
        assert.deepEqual(
            { verdict, threshold, median, noise, range, deltas: deltas.map(row => row.value) },
            expected[os].visibleMemory
        )
        assert.equal(platform.verdict, expected[os].verdict)
        assert.deepEqual(platform.winCriteria, expected[os].winCriteria)
        assert.match(
            result.stderr,
            new RegExp(`\\| ${os} \\| Overall \\| \\| \\| ${platform.verdict} \\|`)
        )
    })
}

test('PID reuse is reported and does not carry CPU history into the new process', () => {
    const run = summary.runs.find(
        row =>
            row.os === 'windows' &&
            row.scene === 'visible' &&
            row.pair === 'a' &&
            row.arm === 'wails'
    )
    assert.deepEqual(run.pidReuse, [expected.pidReuse])
    assert.equal(run.issues.length, 0)
    assert.deepEqual(run.process.gui.renderer.cpuMsPerSec, { median: 21, observed: 1 })
    assert.deepEqual(run.process.gui.total.memoryBytes, { median: 167772160, observed: 3 })
    assert.equal(run.process.model.total.memoryBytes.median, 536870912)
    assert.equal(run.process.service.total.memoryBytes.median, 50331648)
    assert.equal(run.system.pageInPerSec, 4)
})

for (const id of ['missing', 'invalid', 'restart']) {
    test(`${id} pair is excluded and retains its specific rejection reason`, () => {
        const pair = summary.pairs.find(row => row.id === id && row.scene === 'visible')
        assert.ok(
            pair.reasons.some(reason => reason.endsWith(expected.rejected[id])),
            pair.reasons.join('\n')
        )
        assert.deepEqual(pair.deltas, {})
        assert.equal(pair.counterDifferences, null)
        assert.ok(
            !summary.platforms.windows.metrics.visibleMemoryMiB.deltas.some(row => row.pair === id)
        )
    })
}

test('raw schema validation fails the CLI on the invalid sample', () => {
    const invalid = spawnSync(
        process.execPath,
        [path('compare.mjs'), '--validate-samples', path('fixtures/windows-invalid.txt')],
        { encoding: 'utf8' }
    )
    assert.equal(invalid.status, 1)
    assert.equal(invalid.stderr.trim(), expected.rejected.invalid)
})

const visible = ({ run }) => run.os === 'windows' && run.scene === 'visible'
const comparison = record => visible(record) && record.run.kind === 'comparison'
const firstCandidate = runs =>
    runs.find(record => comparison(record) && record.run.pair === 'a' && record.run.arm === 'wails')

test('two adjacent seconds cannot stand in for a five-minute steady window', () => {
    const result = compare(
        writeFixture(runs => {
            const record = firstCandidate(runs)
            record.run.windowMs = [30_000, 330_000]
            record.samples = record.samples.slice(0, 2)
            record.samples.forEach((sample, index) => {
                sample.elapsedMs = 30_000 + 1000 * index
            })
        })
    )
    assert.equal(result.platforms.windows.verdict, 'inconclusive')
    assert.ok(
        result.runs
            .find(run => run.os === 'windows' && run.order === 2)
            .issues.includes('steady window sample coverage does not match declared interval')
    )
})

test('coverage checks interior timer slots even with the expected count and both endpoints', () => {
    const file = writeFixture(runs => {
        firstCandidate(runs).samples[1].elapsedMs = 400
    })
    const pair = compare(file).pairs.find(
        pair => pair.os === 'windows' && pair.scene === 'visible' && pair.id === 'a'
    )
    assert.deepEqual(pair.deltas, {})
    assert.ok(
        pair.reasons.some(reason =>
            reason.endsWith('steady window sample coverage does not match declared interval')
        )
    )
})

test('coverage accepts timer jitter around the scheduled endpoints', () => {
    const file = writeFixture(runs => {
        firstCandidate(runs).samples.forEach(sample => {
            sample.elapsedMs += 50
        })
    })
    const result = compare(file)
    assert.equal(result.platforms.windows.verdict, 'material-win')
    assert.equal(result.runs.find(run => run.os === 'windows' && run.order === 2).samples, 3)
})

test('replay and live arms cannot form a pair', () => {
    const file = writeFixture(runs => {
        firstCandidate(runs).run.inputMode = 'live'
    })
    const result = compare(file)
    assert.equal(result.platforms.windows.verdict, 'inconclusive')
    assert.ok(
        result.pairs
            .find(pair => pair.os === 'windows' && pair.scene === 'visible' && pair.id === 'a')
            .reasons.includes('input modes differ across pair')
    )
})

test('the input mode and sample interval must be declared', () => {
    for (const field of ['inputMode', 'intervalMs']) {
        const file = writeFixture(runs => {
            delete firstCandidate(runs).run[field]
        })
        assert.throws(() => compare(file), /invalid run declaration/)
    }
})

test('live controls cannot establish noise for replay runs of the same arms', () => {
    const file = writeFixture(runs => {
        for (const { run } of runs.filter(visible)) {
            if (run.kind === 'control') run.inputMode = 'live'
        }
    })
    const platform = compare(file).platforms.windows
    assert.equal(platform.metrics.visibleMemoryMiB.noise, null)
    assert.equal(platform.verdict, 'inconclusive')
})

test('a raw sample with ambiguous attribution invalidates the run despite manifest attestation', () => {
    const file = writeFixture(runs => {
        const record = runs.find(
            ({ run }) => run.os === 'macos' && run.scene === 'visible' && run.order === 2
        )
        record.samples[1].attributionClear = false
    })
    const result = compare(file)
    assert.equal(result.platforms.macos.verdict, 'inconclusive')
    assert.ok(
        result.runs
            .find(run => run.os === 'macos' && run.order === 2)
            .issues.includes('sampled process attribution ambiguous')
    )
})

test('distinct paths with identical counter contents invalidate both pairs', () => {
    const file = writeFixture(runs => {
        const reused = runs.find(
            record => comparison(record) && record.run.pair === 'b' && record.run.arm === 'wails'
        )
        reused.samples = firstCandidate(runs).samples
    })
    const result = compare(file)
    assert.equal(result.platforms.windows.verdict, 'inconclusive')
    const pairs = result.pairs.filter(
        pair => pair.os === 'windows' && pair.scene === 'visible' && ['a', 'b'].includes(pair.id)
    )
    assert.equal(pairs.length, 2)
    for (const pair of pairs) {
        assert.deepEqual(pair.deltas, {})
        assert.ok(pair.reasons.some(reason => reason.endsWith('raw capture reused')))
    }
})

test('a UTF-8 BOM and CRLF do not make a reused capture independent', () => {
    const file = writeFixture()
    const { runs } = JSON.parse(readFileSync(file, 'utf8'))
    const candidates = runs.filter(
        run => run.os === 'windows' && run.scene === 'visible' && run.arm === 'wails'
    )
    const source = candidates.find(run => run.pair === 'a')
    const copy = candidates.find(run => run.pair === 'b')
    const original = readFileSync(join(dirname(file), source.file), 'utf8')
    const converted = '\uFEFF' + original.replaceAll('\n', '\r\n')
    writeFileSync(join(dirname(file), copy.file), converted)

    const result = compare(file)
    assert.equal(result.platforms.windows.verdict, 'inconclusive')
    for (const [declaration, raw] of [
        [source, original],
        [copy, converted]
    ]) {
        const run = result.runs.find(run => run.file === declaration.file)
        assert.deepEqual(run.issues, ['raw capture reused'])
        assert.equal(run.captureSha256, createHash('sha256').update(raw).digest('hex'))
        const pair = result.pairs.find(
            pair => pair.os === 'windows' && pair.scene === 'visible' && pair.id === run.pair
        )
        assert.deepEqual(pair.deltas, {})
    }
})

for (const replaced of [['matched-electron', 'wails'], ['wails']]) {
    test(`stock noise cannot substitute for ${replaced.join(' or ')} controls`, () => {
        const file = writeFixture(runs => {
            for (const { run } of runs.filter(visible)) {
                if (run.kind === 'control' && replaced.includes(run.arm)) run.arm = 'stock-electron'
            }
        })
        const { platforms, pairs } = compare(file)
        assert.equal(platforms.windows.verdict, 'inconclusive')
        assert.equal(platforms.windows.metrics.visibleMemoryMiB.noise, null)
        assert.deepEqual(
            pairs.find(
                pair =>
                    pair.os === 'windows' &&
                    pair.scene === 'visible' &&
                    pair.id === 'noise-candidate'
            ).arms,
            ['stock-electron', 'stock-electron']
        )
    })
}

for (const [role, group, reason] of [
    ['main', 'model', 'missing major helper: gui/main'],
    ['engine', 'gui', 'missing model engine'],
    ['renderer', 'model', 'missing major helper: gui/renderer']
]) {
    test(`a busy pair requires ${role} in the same steady samples as the other groups`, () => {
        const file = writeFixture(runs => {
            const { run, samples } = firstCandidate(runs)
            run.requiredGroup = group
            samples[1].processes = samples[1].processes.filter(row => row.role !== role)
        })
        const result = compare(file)
        assert.equal(result.platforms.windows.verdict, 'inconclusive')
        assert.ok(
            result.runs.find(run => run.os === 'windows' && run.order === 2).issues.includes(reason)
        )
    })
}

test('one unkeyed completion cannot authorize three unkeyed inference observations', () => {
    const file = writeFixture(runs => {
        const { measurements } = firstCandidate(runs).run
        delete measurements.inferenceTimings
        delete measurements.inferenceRequests
        measurements.tokensPerSecond = [200, 210, 220]
        measurements.completionPercent = [100]
    })
    const result = compare(file)
    assert.equal(result.platforms.windows.verdict, 'inconclusive')
    const run = result.runs.find(run => run.os === 'windows' && run.order === 2)
    assert.equal(run.outcome.tokensPerSecond, null)
    assert.equal(run.inference.excluded, 3)
})

test('inference credits only a unique successful local request and its matching attempt', () => {
    const file = writeFixture(runs => {
        const { inferenceTimings, inferenceRequests } = firstCandidate(runs).run.measurements
        inferenceTimings[0].attempt = 2
        inferenceRequests[2].localModel.createdUtc = '2026-11-19T00:00:00.000Z'
        inferenceTimings.push({
            requestId: 'failed',
            attempt: 1,
            tokensPerSecond: 5000,
            firstTokenMs: 1
        })
        inferenceRequests.push({ ...inferenceRequests[1], requestId: 'failed', success: false })
        inferenceTimings.push({
            requestId: 'retried',
            attempt: 1,
            tokensPerSecond: 9000,
            firstTokenMs: 1
        })
        inferenceRequests.push({
            ...inferenceRequests[1],
            requestId: 'retried',
            singleAttempt: false
        })
    })
    const run = compare(file).runs.find(run => run.os === 'windows' && run.order === 2)
    assert.equal(run.outcome.tokensPerSecond, 21)
    assert.equal(run.outcome.firstTokenMs, 110)
    assert.ok(run.issues.includes('inference completion failed or unverified'))
    assert.deepEqual(run.inference, {
        credited: 1,
        excluded: 4,
        exclusionReasons: {
            'request or attempt unmatched': 1,
            'local model not sampled': 1,
            'request unsuccessful': 1,
            'single attempt unverified': 1
        }
    })
})

test('duplicate timings or request records cannot provide independent inference evidence', () => {
    const file = writeFixture(runs => {
        const { inferenceTimings, inferenceRequests } = firstCandidate(runs).run.measurements
        inferenceTimings.push({ ...inferenceTimings[0] })
        inferenceRequests.push({ ...inferenceRequests[1], attempt: 2 })
    })
    const run = compare(file).runs.find(run => run.os === 'windows' && run.order === 2)
    assert.equal(run.outcome.tokensPerSecond, 22)
    assert.equal(run.outcome.firstTokenMs, 120)
    assert.deepEqual(run.inference, {
        credited: 1,
        excluded: 3,
        exclusionReasons: {
            'duplicate timing for request': 2,
            'multiple attempts or records for request': 1
        }
    })
})

test('inference joins equivalent process timestamps and distinguishes 100 ns', () => {
    const file = writeFixture(runs => {
        const { run, samples } = firstCandidate(runs)
        const { inferenceRequests, inferenceTimings } = run.measurements
        for (const [index, sample] of samples.entries()) {
            const model = sample.processes.find(row => row.group === 'model')
            model.createdUtc =
                index === 0 ? '2026-11-20T00:00:00.1234000Z' : '2026-11-20T00:00:00.123400Z'
        }
        for (const request of inferenceRequests) {
            request.localModel.createdUtc = '2026-11-20T00:00:00.1234Z'
        }
        run.fence.after.broker.createdUtc = '2026-11-20T00:00:00Z'
        inferenceRequests.push({
            ...inferenceRequests[0],
            requestId: 'different-incarnation',
            localModel: { pid: 31, createdUtc: '2026-11-20T00:00:00.1234001Z' }
        })
        inferenceTimings.push({
            requestId: 'different-incarnation',
            attempt: 1,
            firstTokenMs: 1,
            tokensPerSecond: 1000
        })
    })

    const result = compare(file)
    const run = result.runs.find(run => run.os === 'windows' && run.order === 2)
    assert.deepEqual(run.inference, {
        credited: 3,
        excluded: 1,
        exclusionReasons: { 'local model not sampled': 1 }
    })
    assert.equal(run.outcome.tokensPerSecond, 21)
    assert.equal(run.outcome.firstTokenMs, 110)
    assert.deepEqual(run.pidReuse, [expected.pidReuse])
    assert.equal(result.platforms.windows.verdict, 'material-win')
})

function resourceWin(runs, { resource, outcome, noise = 0, disagree = false }) {
    const windows = runs.filter(visible)
    for (const { run, samples } of windows) {
        const candidate = run.arm === 'wails'
        const { measurements } = run
        if (outcome === 'tokens') {
            const tokensPerSecond = candidate && run.kind === 'comparison' ? 21 : 20
            measurements.inferenceTimings.forEach(timing => {
                timing.tokensPerSecond = tokensPerSecond
            })
        }
        if (candidate && run.kind === 'comparison') {
            if (outcome === 'first-token') {
                measurements.inferenceTimings.forEach(timing => {
                    timing.firstTokenMs *= 0.9
                })
            } else if (outcome === 'frames') measurements.missedFramesPercent = [0, 1, 2]
        } else if (outcome === 'frames') measurements.missedFramesPercent = [5, 6, 7]
        for (const sample of samples) {
            const gui = sample.processes.filter(row => row.group === 'gui')
            for (const row of gui) {
                row.memoryBytes = (candidate ? 180 : 200) * 1024 ** 2
                row[resource] = 0
            }
            const main = gui.find(row => row.role === 'main')
            if (!main) continue
            main[resource] = candidate ? 61 : 100
            if (run.kind === 'control' && run.order === 6) main[resource] += noise
            if (disagree && candidate && run.pair === 'b') main[resource] = 101
        }
    }
}

for (const [resource, outcome, metric, gain] of [
    ['cpuMsPerSec', 'tokens', 'tokensPerSecondPercent', 5],
    ['gpuEnginePercent', 'first-token', 'firstTokenPercent', 10],
    ['cpuMsPerSec', 'frames', 'missedFramesPp', 5]
]) {
    test(`${resource} with a repeatable ${outcome} improvement can win without 100 MiB`, () => {
        const file = writeFixture(runs => resourceWin(runs, { resource, outcome }))
        const platform = compare(file).platforms.windows
        const demand = resource === 'cpuMsPerSec' ? 'guiCpuMsPerSec' : 'guiGpuEnginePercent'
        assert.equal(platform.metrics.visibleMemoryMiB.median, 40)
        assert.equal(platform.metrics.visibleMemoryMiB.verdict, 'no-material-win')
        assert.equal(platform.metrics[demand].median, 39)
        assert.equal(platform.metrics[metric].median, gain)
        assert.equal(platform.verdict, 'material-win')
        assert.deepEqual(platform.winCriteria, [{ metric: demand, outcome: metric }])
    })
}

for (const [scenario, memoryLossMiB, verdict, winCriteria] of [
    [
        'one pair loses 4 MiB',
        4,
        'material-win',
        [{ metric: 'guiCpuMsPerSec', outcome: 'tokensPerSecondPercent' }]
    ],
    ['one pair loses 101 MiB', 101, 'inconclusive', []]
]) {
    test(`CPU and throughput verdict when ${scenario}`, () => {
        const file = writeFixture(runs => {
            resourceWin(runs, { resource: 'cpuMsPerSec', outcome: 'tokens' })
            for (const { run, samples } of runs.filter(comparison)) {
                const memoryMiB =
                    run.arm === 'matched-electron'
                        ? 400
                        : run.pair === 'b'
                          ? 400 + memoryLossMiB
                          : 240
                for (const sample of samples) {
                    for (const row of sample.processes.filter(row => row.group === 'gui')) {
                        row.memoryBytes = row.role === 'main' ? memoryMiB * 1024 ** 2 : 0
                    }
                }
            }
        })
        const platform = compare(file).platforms.windows
        const memory = platform.metrics.visibleMemoryMiB
        assert.equal(memory.verdict, 'inconclusive')
        assert.deepEqual(
            memory.deltas.map(delta => delta.value),
            [160, -memoryLossMiB, 160, 160]
        )
        assert.equal(platform.metrics.guiCpuMsPerSec.median, 39)
        assert.equal(platform.metrics.tokensPerSecondPercent.median, 5)
        assert.equal(platform.verdict, verdict)
        assert.deepEqual(platform.winCriteria, winCriteria)
    })
}

for (const [scenario, change] of [
    ['demand savings equal noise', { noise: 39 }],
    ['one valid pair spends more GUI CPU', { disagree: true }]
]) {
    test(`the resource alternative cannot win when ${scenario}`, () => {
        const file = writeFixture(runs =>
            resourceWin(runs, { resource: 'cpuMsPerSec', outcome: 'tokens', ...change })
        )
        const platform = compare(file).platforms.windows
        assert.equal(platform.verdict, 'no-material-win')
        assert.deepEqual(platform.winCriteria, [])
    })
}

test('a single usable GUI CPU delta per run does not establish a demand win', () => {
    const file = writeFixture(runs => {
        resourceWin(runs, { resource: 'cpuMsPerSec', outcome: 'tokens' })
        for (const { samples } of runs.filter(visible)) {
            for (const [index, sample] of samples.entries()) {
                for (const row of sample.processes.filter(row => row.group === 'gui')) {
                    if (index < 2) row.cpuMsPerSec = null
                    row.gpuEnginePercent = null
                }
            }
        }
    })
    const platform = compare(file).platforms.windows
    assert.equal(platform.metrics.tokensPerSecondPercent.verdict, 'material-win')
    assert.equal(platform.metrics.guiCpuMsPerSec.median, null)
    assert.equal(platform.verdict, 'inconclusive')
    assert.deepEqual(platform.winCriteria, [])
})
