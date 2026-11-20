#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const arms = ['stock-electron', 'matched-electron', 'wails']
const systems = ['windows', 'macos']
const roles = {
    gui: ['main', 'renderer', 'gpu', 'other'],
    service: ['service', 'broker', 'worker'],
    model: ['engine']
}
const counters = [
    'memoryBytes',
    'commitBytes',
    'cpuMsPerSec',
    'cpuPercentOneCore',
    'gpuEnginePercent',
    'gpuDedicatedBytes',
    'gpuSharedBytes'
]
const systemCounters = [
    'availableBytes',
    'pageInPerSec',
    'pageOutPerSec',
    'pageSizeBytes',
    'pageInCount',
    'pageOutCount',
    'swapInCount',
    'swapOutCount',
    'pressureFreePercent'
]
const outcomes = ['coldOpenMs', 'warmReopenMs', 'missedFramesPercent', 'completionPercent']
const inferenceMetrics = ['firstTokenMs', 'tokensPerSecond']
const MiB = 1024 ** 2
const tuple = (...parts) => JSON.stringify(parts)
const quantity = value => Number.isFinite(value) && value >= 0
const pid = value => Number.isSafeInteger(value) && value > 0
const counter = value => value === null || quantity(value)
const label = value => typeof value === 'string' && value.length > 0
const isUnclassifiedGui = row => row.group === 'gui' && row.role === 'other'

function canonicalUtc(value) {
    if (typeof value !== 'string') return null
    const match = /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.(\d{1,7}))?Z$/.exec(value)
    if (!match) return null
    const date = new Date(value)
    if (!Number.isFinite(date.getTime())) return null
    // Date resolves the whole second. Keep the original fractional precision.
    const second = date.toISOString().slice(0, -5)
    const [, fraction = ''] = match
    return `${second}.${fraction.padEnd(7, '0')}Z`
}

function identity(row) {
    if (!pid(row?.pid)) return null
    const createdUtc = canonicalUtc(row.createdUtc)
    return createdUtc && tuple(row.pid, createdUtc)
}

function requiredProcessSets(processes) {
    const sets = new Map()
    for (const row of processes) {
        if (isUnclassifiedGui(row)) continue
        const key = `${row.group}/${row.role}`
        if (!sets.has(key)) sets.set(key, new Set())
        sets.get(key).add(identity(row))
    }
    return sets
}

export function validateSample(sample) {
    const timestamp = canonicalUtc(sample?.timestamp)
    if (
        sample?.schema !== 1 ||
        !systems.includes(sample.os) ||
        !quantity(sample.elapsedMs) ||
        !timestamp ||
        !Array.isArray(sample.processes) ||
        !sample.system ||
        (sample.attributionClear !== undefined && typeof sample.attributionClear !== 'boolean') ||
        !systemCounters.every(key => counter(sample.system[key]))
    )
        return 'invalid sample envelope'
    const seen = new Set()
    for (const row of sample.processes) {
        const createdUtc = canonicalUtc(row?.createdUtc)
        if (
            !row ||
            !pid(row.pid) ||
            !createdUtc ||
            seen.has(row.pid) ||
            createdUtc > timestamp ||
            !Object.hasOwn(roles, row.group) ||
            !roles[row.group].includes(row.role) ||
            !counters.every(key => counter(row[key]))
        )
            return 'invalid process record'
        // memoryBytes is the primary memory metric. Only unclassified GUI helpers may omit it.
        if (row.memoryBytes === null && !isUnclassifiedGui(row))
            return `missing memoryBytes for ${row.group}/${row.role}`
        seen.add(row.pid)
    }
    return null
}

function median(values) {
    if (!values.length) return null
    const sorted = [...values].sort((a, b) => a - b)
    const middle = Math.floor(sorted.length / 2)
    return sorted.length % 2 ? sorted[middle] : sorted[middle - 1] / 2 + sorted[middle] / 2
}

function series(text, visit, invalid) {
    text.replace(/^\uFEFF/, '')
        .split(/\r?\n/)
        .forEach((line, index) => {
            if (!line.trim()) return
            let sample
            try {
                sample = JSON.parse(line)
            } catch {
                invalid(index + 1, 'malformed JSON')
                return
            }
            const problem = validateSample(sample)
            if (problem) invalid(index + 1, problem)
            else visit(sample, index + 1)
        })
}

function sortObjectKeys(_key, value) {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return value
    return Object.fromEntries(
        Object.keys(value)
            .sort()
            .map(key => [key, value[key]])
    )
}

function canonicalObservation(sample) {
    return JSON.stringify(
        {
            ...sample,
            timestamp: canonicalUtc(sample.timestamp),
            processes: sample.processes
                .map(row => ({
                    ...row,
                    createdUtc: canonicalUtc(row.createdUtc)
                }))
                .sort((a, b) => a.pid - b.pid)
        },
        sortObjectKeys
    )
}

function verifyManifest(manifest) {
    const { primaryScene, trayScene, activeScene, runs } = manifest
    if (
        ![primaryScene, trayScene, activeScene].every(label) ||
        primaryScene === trayScene ||
        activeScene === trayScene ||
        !Array.isArray(runs)
    )
        throw new Error('invalid scene configuration')
    const orders = new Set()
    for (const run of runs) {
        if (
            !systems.includes(run.os) ||
            !arms.includes(run.arm) ||
            ![run.host, run.scene, run.pair, run.file].every(label) ||
            !['comparison', 'control', 'stock-reference'].includes(run.kind) ||
            !pid(run.order) ||
            !['replay', 'live'].includes(run.inputMode) ||
            !pid(run.intervalMs) ||
            !Array.isArray(run.windowMs) ||
            run.windowMs.length !== 2 ||
            !run.windowMs.every(quantity) ||
            run.windowMs[0] >= run.windowMs[1] ||
            !Array.isArray(run.requiredGuiRoles) ||
            !run.requiredGuiRoles.every(role => roles.gui.includes(role)) ||
            !['gui', 'service', 'model'].includes(run.requiredGroup ?? 'gui')
        ) {
            throw new Error('invalid run declaration')
        }
        const order = tuple(run.os, run.host, run.order)
        if (orders.has(order)) throw new Error('duplicate run order on one host')
        orders.add(order)
    }
}

function fenceKey(fence) {
    const service = identity(fence?.service)
    const broker = identity(fence?.broker)
    if (
        !service ||
        !broker ||
        ![fence.serviceBuild, fence.brokerBuild, fence.modelSettings].every(label) ||
        !Number.isSafeInteger(fence.brokerRestartCount) ||
        fence.brokerRestartCount < 0
    )
        return null
    return tuple(
        service,
        broker,
        fence.serviceBuild,
        fence.brokerBuild,
        fence.brokerRestartCount,
        fence.modelSettings
    )
}

function summarizeInference(measurements, samples, issues) {
    const requests = Array.isArray(measurements?.inferenceRequests)
        ? measurements.inferenceRequests
        : []
    const timings = Array.isArray(measurements?.inferenceTimings)
        ? measurements.inferenceTimings
        : []
    const requestsById = Map.groupBy(requests, row => row?.requestId)
    const timingsById = Map.groupBy(timings, row => row?.requestId)
    const localModels = new Set(
        samples.flatMap(sample =>
            sample.processes.filter(row => row.group === 'model').map(identity)
        )
    )
    const outcomeValues = Object.fromEntries(inferenceMetrics.map(metric => [metric, []]))
    const exclusionReasons = {}
    let credited = 0
    function exclude(reason, count = 1) {
        exclusionReasons[reason] = (exclusionReasons[reason] ?? 0) + count
    }
    for (const timing of timings) {
        const fields = inferenceMetrics.filter(metric => timing?.[metric] !== undefined)
        if (
            !label(timing?.requestId) ||
            !pid(timing.attempt) ||
            !fields.length ||
            !fields.every(
                metric =>
                    quantity(timing[metric]) && (metric !== 'tokensPerSecond' || timing[metric] > 0)
            )
        ) {
            exclude('invalid timing')
            continue
        }
        const matching = requestsById.get(timing.requestId) ?? []
        const [request] = matching
        if (timingsById.get(timing.requestId).length !== 1) exclude('duplicate timing for request')
        else if (matching.length > 1) exclude('multiple attempts or records for request')
        else if (request?.attempt !== timing.attempt) exclude('request or attempt unmatched')
        else if (request.success !== true) exclude('request unsuccessful')
        else if (request.singleAttempt !== true) exclude('single attempt unverified')
        else if (!localModels.has(identity(request.localModel))) exclude('local model not sampled')
        else {
            for (const metric of fields) outcomeValues[metric].push(timing[metric])
            credited++
        }
    }
    for (const metric of inferenceMetrics) {
        const unkeyed = measurements?.[metric]
        if (Array.isArray(unkeyed) && unkeyed.length) exclude('unkeyed timing', unkeyed.length)
    }
    if (
        requests.some(row => row?.success === false) ||
        (Array.isArray(measurements?.completionPercent) &&
            measurements.completionPercent.some(value => value !== 100))
    ) {
        issues.add('inference completion failed or unverified')
    }
    const outcome = Object.fromEntries(
        inferenceMetrics.map(metric => [metric, median(outcomeValues[metric])])
    )
    const excluded = Object.values(exclusionReasons).reduce((sum, count) => sum + count, 0)
    return { outcome, inference: { credited, excluded, exclusionReasons } }
}

function summarizeRun(run, base, { primaryScene, trayScene, activeScene }) {
    const issues = new Set()
    const samples = []
    const slots = new Set()
    const firstSlot = Math.ceil(run.windowMs[0] / run.intervalMs)
    const lastSlot = Math.floor(run.windowMs[1] / run.intervalMs)
    const coverageIssue = 'steady window sample coverage does not match declared interval'
    const busy = run.scene === primaryScene || run.scene === activeScene
    const requiredGuiRoles = new Set(run.requiredGuiRoles)
    if (busy || (run.requiredGroup ?? 'gui') === 'gui') requiredGuiRoles.add('main')
    const lastProcess = new Map()
    const pidReuse = []
    let previous
    let steadyProcessSets
    const expectedFence = fenceKey(run.fence?.before)
    if (!expectedFence || !fenceKey(run.fence?.after))
        issues.add('missing process/build/settings fence')
    else if (expectedFence !== fenceKey(run.fence.after))
        issues.add('service or broker fence changed within run')
    if (run.parity !== true) issues.add('workload or visual parity unverified')
    if (run.attributionClear !== true) issues.add('process attribution unverified')
    if (run.scene !== trayScene && run.usableCurrentData !== true)
        issues.add('usable current data unverified')

    const capture = readFileSync(resolve(base, run.file))
    const captureSha256 = createHash('sha256').update(capture).digest('hex')
    const observationHash = createHash('sha256')
    let hasObservations = false
    series(
        capture.toString('utf8'),
        (sample, line) => {
            observationHash.update(canonicalObservation(sample) + '\n')
            hasObservations = true
            if (
                sample.os !== run.os ||
                (previous &&
                    (sample.elapsedMs <= previous.elapsedMs ||
                        canonicalUtc(sample.timestamp) < canonicalUtc(previous.timestamp)))
            ) {
                issues.add(`line ${line}: platform or sample order mismatch`)
                return
            }
            previous = sample
            for (const row of sample.processes) {
                const old = lastProcess.get(row.pid)
                if (old && identity(old) !== identity(row))
                    pidReuse.push({
                        pid: row.pid,
                        previousCreatedUtc: old.createdUtc,
                        createdUtc: row.createdUtc,
                        line
                    })
                lastProcess.set(row.pid, row)
            }
            for (const role of ['service', 'broker']) {
                const matching = sample.processes.filter(
                    row => row.group === 'service' && row.role === role
                )
                if (
                    expectedFence &&
                    (matching.length !== 1 ||
                        identity(matching[0]) !== identity(run.fence.before[role]))
                )
                    issues.add(`sampled ${role} identity changed or missing`)
            }
            // Both samplers schedule from elapsed zero. Keep boundary slots despite small timer jitter.
            const slot = Math.round(sample.elapsedMs / run.intervalMs)
            if (slot < firstSlot || slot > lastSlot) return
            if (
                slots.has(slot) ||
                Math.abs(sample.elapsedMs - slot * run.intervalMs) > run.intervalMs / 10
            ) {
                issues.add(coverageIssue)
            }
            slots.add(slot)
            if (sample.attributionClear === false)
                issues.add('sampled process attribution ambiguous')
            const currentProcessSets = requiredProcessSets(sample.processes)
            steadyProcessSets ??= currentProcessSets
            const processSetKeys = new Set([
                ...steadyProcessSets.keys(),
                ...currentProcessSets.keys()
            ])
            for (const key of processSetKeys) {
                const expected = steadyProcessSets.get(key)
                const current = currentProcessSets.get(key)
                if (
                    !expected ||
                    !current ||
                    expected.size !== current.size ||
                    !expected.isSubsetOf(current)
                ) {
                    issues.add(`required process set changed during steady window: ${key}`)
                }
            }
            const gui = sample.processes.filter(row => row.group === 'gui')
            for (const role of requiredGuiRoles) {
                if (!gui.some(row => row.role === role))
                    issues.add(`missing major helper: gui/${role}`)
            }
            if (
                (busy || run.requiredGroup === 'model') &&
                !sample.processes.some(row => row.group === 'model')
            ) {
                issues.add('missing model engine')
            }
            samples.push(sample)
        },
        (line, reason) => issues.add(`line ${line}: ${reason}`)
    )
    if (samples.length < 2) issues.add('fewer than two steady samples')
    if (slots.size !== lastSlot - firstSlot + 1) issues.add(coverageIssue)

    const process = {}
    let guiCpuCovered = false
    for (const [group, groupRoles] of Object.entries(roles)) {
        process[group] = {}
        for (const role of ['total', ...groupRoles]) {
            process[group][role] = Object.fromEntries(
                counters.map(key => {
                    const values = samples.map(({ processes }) => {
                        const rows = processes.filter(
                            row => row.group === group && (role === 'total' || row.role === role)
                        )
                        return rows.length && rows.every(row => row[key] !== null)
                            ? rows.reduce((sum, row) => sum + row[key], 0)
                            : null
                    })
                    const observed = values.filter(value => value !== null)
                    if (group === 'gui' && role === 'total' && key === 'cpuMsPerSec') {
                        guiCpuCovered =
                            samples.length >= 2 && values.slice(1).every(value => value !== null)
                    }
                    // CPU deltas need a prior sample of the same process incarnation.
                    const usable =
                        key === 'cpuMsPerSec'
                            ? observed.length > 0
                            : observed.length === samples.length
                    return [
                        key,
                        { median: usable ? median(observed) : null, observed: observed.length }
                    ]
                })
            )
        }
    }
    const system = Object.fromEntries(
        systemCounters.map(key => {
            const values = samples.map(sample => sample.system[key])
            return [key, values.every(quantity) ? median(values) : null]
        })
    )
    for (const [rate, count] of [
        ['pageInPerSec', 'pageInCount'],
        ['pageOutPerSec', 'pageOutCount'],
        ['swapInPerSec', 'swapInCount'],
        ['swapOutPerSec', 'swapOutCount']
    ]) {
        if (system[rate] !== undefined && system[rate] !== null) continue
        const values = samples.slice(1).map((sample, index) => {
            const previous = samples[index]
            const currentCount = sample.system[count]
            const priorCount = previous.system[count]
            return quantity(currentCount) && quantity(priorCount) && currentCount >= priorCount
                ? (1000 * (currentCount - priorCount)) / (sample.elapsedMs - previous.elapsedMs)
                : null
        })
        system[rate] = values.every(quantity) ? median(values) : null
    }
    const outcome = Object.fromEntries(
        outcomes.map(key => {
            const values = run.measurements?.[key]
            const valid =
                Array.isArray(values) &&
                values.every(
                    value =>
                        quantity(value) &&
                        (!key.endsWith('Percent') || value <= 100) &&
                        (key !== 'tokensPerSecond' || value > 0)
                )
            return [key, valid ? median(values) : null]
        })
    )
    const timings = summarizeInference(run.measurements, samples, issues)
    return {
        ...run,
        captureSha256,
        captureIdentity: hasObservations ? observationHash.digest('hex') : null,
        fenceIdentity: expectedFence,
        issues: [...issues],
        pidReuse,
        samples: samples.length,
        process,
        system,
        guiCpuCovered,
        outcome: { ...outcome, ...timings.outcome },
        inference: timings.inference
    }
}

const guiMemory = run => run.process.gui.total.memoryBytes.median
const definitions = {
    visibleMemoryMiB: [run => guiMemory(run) / MiB, 100, 'lower'],
    trayMemoryMiB: [run => guiMemory(run) / MiB, 50, 'lower'],
    missedFramesPp: [run => run.outcome.missedFramesPercent, 5, 'lower'],
    tokensPerSecondPercent: [run => run.outcome.tokensPerSecond, 5, 'higher-relative'],
    firstTokenPercent: [run => run.outcome.firstTokenMs, 10, 'lower-relative'],
    warmReopenMs: [run => run.outcome.warmReopenMs, 250, 'lower'],
    guiCpuMsPerSec: [
        run => (run.guiCpuCovered ? run.process.gui.total.cpuMsPerSec.median : null),
        0,
        'lower'
    ],
    guiGpuEnginePercent: [run => run.process.gui.total.gpuEnginePercent.median, 0, 'lower']
}
const demandMetrics = ['guiCpuMsPerSec', 'guiGpuEnginePercent']
const improvementMetrics = ['missedFramesPp', 'tokensPerSecondPercent', 'firstTokenPercent']

function difference(metric, first, second) {
    if (metric.endsWith('MemoryMiB') && [first, second].some(run => guiMemory(run) === null))
        return null
    const [value, , direction] = definitions[metric]
    const a = value(first)
    const b = value(second)
    if (a === null || b === null || (direction.endsWith('relative') && a === 0)) return null
    const benefit = direction.startsWith('higher') ? b - a : a - b
    return direction.endsWith('relative') ? (100 * benefit) / a : benefit
}

function counterDifferences(baseline, candidate) {
    const subtract = (a, b) => (a === null || b === null ? null : b - a)
    const process = {}
    for (const [group, groupRoles] of Object.entries(baseline.process)) {
        process[group] = {}
        for (const [role, metrics] of Object.entries(groupRoles)) {
            process[group][role] = Object.fromEntries(
                Object.entries(metrics).map(([key, value]) => [
                    key,
                    subtract(value.median, candidate.process[group][role][key].median)
                ])
            )
        }
    }
    const system = Object.fromEntries(
        Object.entries(baseline.system).map(([key, value]) => [
            key,
            subtract(value, candidate.system[key])
        ])
    )
    const outcome = Object.fromEntries(
        Object.entries(baseline.outcome).map(([key, value]) => [
            key,
            subtract(value, candidate.outcome[key])
        ])
    )
    return { process, system, outcome }
}

function makePairs(runs) {
    const groups = Map.groupBy(runs, run => tuple(run.os, run.host, run.scene, run.kind, run.pair))
    return [...groups.values()].map(group => {
        const [first] = group
        const sorted = [...group].sort((a, b) => a.order - b.order)
        const [baseline, candidate] =
            first.kind === 'control'
                ? sorted
                : [
                      group.find(
                          run =>
                              run.arm ===
                              (first.kind === 'stock-reference'
                                  ? 'stock-electron'
                                  : 'matched-electron')
                      ),
                      group.find(
                          run =>
                              run.arm ===
                              (first.kind === 'stock-reference' ? 'matched-electron' : 'wails')
                      )
                  ]
        const reasons = group.flatMap(run =>
            run.issues.map(reason => `${run.arm} #${run.order}: ${reason}`)
        )
        if (
            group.length !== 2 ||
            !baseline ||
            !candidate ||
            (first.kind === 'control' && baseline.arm !== candidate.arm)
        )
            reasons.push('invalid pair arms')
        else {
            if (baseline.fenceIdentity !== candidate.fenceIdentity)
                reasons.push('service or broker fence changed across pair')
            if (baseline.inputMode !== candidate.inputMode)
                reasons.push('input modes differ across pair')
            if (baseline.intervalMs !== candidate.intervalMs)
                reasons.push('sampling intervals differ across pair')
        }
        return {
            os: first.os,
            host: first.host,
            scene: first.scene,
            kind: first.kind,
            id: first.pair,
            arms: [baseline?.arm, candidate?.arm],
            inputMode: reasons.length ? null : baseline.inputMode,
            intervalMs: reasons.length ? null : baseline.intervalMs,
            orders: sorted.map(run => run.order),
            reasons,
            fenceIdentity: reasons.length ? null : baseline.fenceIdentity,
            activeUse: group.every(run => run.activeUse === true),
            counterDifferences: reasons.length ? null : counterDifferences(baseline, candidate),
            deltas: reasons.length
                ? {}
                : Object.fromEntries(
                      Object.keys(definitions).map(metric => [
                          metric,
                          difference(metric, baseline, candidate)
                      ])
                  )
        }
    })
}

function evaluate(metric, comparisons, controls) {
    const threshold = definitions[metric][1]
    const valid = comparisons.filter(pair => !pair.reasons.length)
    const deltas = valid.map(pair => ({
        host: pair.host,
        pair: pair.id,
        value: pair.deltas[metric]
    }))
    const values = deltas.map(delta => delta.value).filter(value => value !== null)
    const context = pair => tuple(pair.host, pair.fenceIdentity, pair.inputMode, pair.intervalMs)
    const contexts = Map.groupBy(valid, context)
    const noisePairs = controls.filter(control => {
        if (control.reasons.length || control.deltas[metric] === null) return false
        const comparable = contexts.get(context(control))
        if (!comparable?.[0].arms.includes(control.arms[0])) return false
        const orders = comparable.flatMap(pair => pair.orders)
        return (
            control.orders[0] > Math.min(...orders) && control.orders.at(-1) < Math.max(...orders)
        )
    })
    const noise =
        valid.every(pair =>
            pair.arms.every(arm =>
                noisePairs.some(
                    control => context(control) === context(pair) && control.arms[0] === arm
                )
            )
        ) && noisePairs.length
            ? Math.max(...noisePairs.map(control => Math.abs(control.deltas[metric])))
            : null
    const middle = median(values)
    const result = {
        verdict: 'inconclusive',
        reason: 'missing valid pairs, decisive measurements or interleaved controls',
        threshold,
        noise,
        withinRegressionTolerance: null,
        median: middle,
        range: values.length ? [Math.min(...values), Math.max(...values)] : null,
        deltas
    }
    if (middle === null || noise === null || values.length !== deltas.length) return result
    // The pre-registered tolerance is a fixed ceiling; noise only decides
    // whether a breach is a confirmed regression or an inconclusive result.
    const limit = Math.max(threshold, noise)
    result.withinRegressionTolerance = values.every(value => value >= -threshold)
    const thresholdMet = improvementMetrics.includes(metric)
        ? middle >= threshold - 1e-12
        : middle > threshold
    if (thresholdMet && middle > noise && values.every(value => value > 0)) {
        return {
            ...result,
            verdict: 'material-win',
            reason: 'all valid pairs favor candidate and median clears threshold and noise'
        }
    }
    if (middle < -limit && values.every(value => value < 0)) {
        return {
            ...result,
            verdict: 'regression',
            reason: 'all valid pairs favor baseline and median exceeds tolerance and noise'
        }
    }
    if (!result.withinRegressionTolerance || Math.abs(middle) > limit) {
        return {
            ...result,
            reason: 'pairs disagree on direction or a pair exceeds regression tolerance'
        }
    }
    return {
        ...result,
        verdict: 'no-material-win',
        reason: 'median does not clear threshold and noise'
    }
}

export function compare(manifestPath) {
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8').replace(/^\uFEFF/, ''))
    verifyManifest(manifest)
    const runs = manifest.runs.map(run => summarizeRun(run, dirname(manifestPath), manifest))
    const observedRuns = runs.filter(run => run.captureIdentity !== null)
    for (const captures of Map.groupBy(observedRuns, run => run.captureIdentity).values()) {
        if (captures.length > 1) for (const run of captures) run.issues.push('raw capture reused')
    }
    const pairs = makePairs(runs)
    const platforms = Object.fromEntries(
        systems.map(os => {
            const select = (scene, kind) =>
                pairs.filter(
                    pair =>
                        pair.os === os &&
                        pair.scene === scene &&
                        pair.kind === kind &&
                        (scene !== manifest.activeScene || pair.activeUse)
                )
            const metrics = {}
            for (const metric of Object.keys(definitions)) {
                const scene =
                    metric === 'trayMemoryMiB' ? manifest.trayScene : manifest.primaryScene
                metrics[metric] = evaluate(
                    metric,
                    select(scene, 'comparison'),
                    select(scene, 'control')
                )
            }
            const activeUse = Object.fromEntries(
                improvementMetrics.map(metric => [
                    metric,
                    evaluate(
                        metric,
                        select(manifest.activeScene, 'comparison'),
                        select(manifest.activeScene, 'control')
                    )
                ])
            )
            const stockReference = evaluate(
                'visibleMemoryMiB',
                select(manifest.primaryScene, 'stock-reference'),
                select(manifest.primaryScene, 'control')
            )
            const checks = [
                ...Object.entries(metrics)
                    .filter(([metric]) => !demandMetrics.includes(metric))
                    .map(([, result]) => result),
                ...Object.values(activeUse)
            ]
            const decisionModes = new Set(
                [manifest.primaryScene, manifest.trayScene, manifest.activeScene]
                    .flatMap(scene => select(scene, 'comparison'))
                    .filter(pair => !pair.reasons.length)
                    .map(pair => pair.inputMode)
            )
            const winCriteria = []
            let verdict = 'no-material-win'
            let reason =
                'neither the visible memory nor resource and outcome criterion cleared threshold and noise'
            if (checks.some(check => check.verdict === 'regression')) {
                verdict = 'regression'
                reason = 'at least one decisive metric regressed'
            } else if (checks.some(check => check.withinRegressionTolerance !== true)) {
                verdict = 'inconclusive'
                reason =
                    'at least one metric exceeds regression tolerance or lacks decisive evidence'
            } else if (
                [metrics.visibleMemoryMiB, metrics.trayMemoryMiB].some(
                    check => check.deltas.length < 3
                )
            ) {
                verdict = 'inconclusive'
                reason = 'fewer than three valid visible or tray comparisons'
            } else if (decisionModes.size > 1) {
                verdict = 'inconclusive'
                reason = 'decisive comparisons span replay and live input modes'
            } else {
                if (metrics.visibleMemoryMiB.verdict === 'material-win')
                    winCriteria.push({ metric: 'visibleMemoryMiB' })
                for (const metric of demandMetrics) {
                    if (metrics[metric].verdict !== 'material-win') continue
                    for (const outcome of improvementMetrics) {
                        if (metrics[outcome].verdict === 'material-win')
                            winCriteria.push({ metric, outcome })
                    }
                }
                if (winCriteria.length) {
                    verdict = 'material-win'
                    reason = 'a material win criterion passed with decisive guardrails'
                } else if (
                    improvementMetrics.some(metric => metrics[metric].verdict === 'material-win') &&
                    demandMetrics.every(
                        metric =>
                            metrics[metric].noise === null ||
                            metrics[metric].deltas.some(delta => delta.value === null)
                    )
                ) {
                    verdict = 'inconclusive'
                    reason =
                        'outcome improved but comparable GUI CPU or GPU demand evidence is missing'
                }
            }
            return [os, { verdict, reason, winCriteria, metrics, activeUse, stockReference }]
        })
    )
    return { platforms, pairs, runs }
}

function markdown({ platforms }) {
    const lines = [
        '| OS | Metric | Median benefit | Same-host noise | Verdict |',
        '| --- | --- | ---: | ---: | --- |'
    ]
    const display = value => (value === null ? 'unknown' : Number(value.toFixed(2)))
    for (const [os, result] of Object.entries(platforms)) {
        const rows = [
            ...Object.entries(result.metrics),
            ...Object.entries(result.activeUse).map(([metric, row]) => [
                `Active use / ${metric}`,
                row
            ]),
            ['Stock reference / visibleMemoryMiB', result.stockReference]
        ]
        for (const [metric, row] of rows) {
            lines.push(
                `| ${os} | ${metric} | ${display(row.median)} | ${display(row.noise)} | ${row.verdict} |`
            )
        }
        lines.push(`| ${os} | Overall | | | ${result.verdict} |`)
        if (result.winCriteria.length) {
            const criteria = result.winCriteria.map(({ metric, outcome }) =>
                outcome ? `${metric} + ${outcome}` : metric
            )
            lines.push(`| ${os} | Win criteria | | | ${criteria.join(', ')} |`)
        }
    }
    return lines.join('\n')
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
    try {
        const [input, samples] = process.argv.slice(2)
        if (input === '--validate-samples' && samples) {
            let count = 0
            series(
                readFileSync(samples === '-' ? 0 : samples, 'utf8'),
                () => count++,
                (line, reason) => {
                    throw new Error(`line ${line}: ${reason}`)
                }
            )
            if (!count) throw new Error('no samples')
            console.log(`Validated ${count} samples`)
        } else {
            if (!input || samples)
                throw new Error(
                    'usage: compare.mjs MANIFEST.json | --validate-samples FILE_OR_DASH'
                )
            const summary = compare(resolve(input))
            console.log(JSON.stringify(summary, null, 2))
            console.error(markdown(summary))
        }
    } catch (error) {
        console.error(error.message)
        process.exitCode = 1
    }
}
