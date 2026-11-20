#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const arms = ['stock-electron', 'matched-electron', 'wails']
const systems = ['windows', 'macos']
const roles = {
  gui: ['main', 'renderer', 'gpu', 'other'],
  service: ['service', 'broker', 'worker'],
  model: ['engine'],
}
const counters = ['memoryBytes', 'commitBytes', 'cpuMsPerSec', 'cpuPercentOneCore',
  'gpuEnginePercent', 'gpuDedicatedBytes', 'gpuSharedBytes']
const systemCounters = ['availableBytes', 'pageInPerSec', 'pageOutPerSec', 'pageSizeBytes',
  'pageInCount', 'pageOutCount', 'swapInCount', 'swapOutCount', 'pressureFreePercent']
const outcomes = ['coldOpenMs', 'warmReopenMs', 'missedFramesPercent', 'firstTokenMs',
  'tokensPerSecond', 'completionPercent']
const MiB = 1024 ** 2
const tuple = (...parts) => JSON.stringify(parts)
const identity = ({ pid, createdUtc }) => tuple(pid, createdUtc)
const quantity = value => Number.isFinite(value) && value >= 0
const pid = value => Number.isSafeInteger(value) && value > 0
const counter = value => value === null || quantity(value)
const utc = value => typeof value === 'string' &&
  /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,7})?Z$/.test(value) && Number.isFinite(Date.parse(value))
const label = value => typeof value === 'string' && value.length > 0

export function validateSample(sample) {
  if (sample?.schema !== 1 || !systems.includes(sample.os) || !quantity(sample.elapsedMs) ||
      !utc(sample.timestamp) || !Array.isArray(sample.processes) || !sample.system ||
      !systemCounters.every(key => counter(sample.system[key]))) return 'invalid sample envelope'
  const seen = new Set()
  for (const row of sample.processes) {
    if (!row || !pid(row.pid) || !utc(row.createdUtc) || seen.has(row.pid) ||
        Date.parse(row.createdUtc) > Date.parse(sample.timestamp) ||
        !Object.hasOwn(roles, row.group) || !roles[row.group].includes(row.role) ||
        !counters.every(key => counter(row[key]))) return 'invalid process record'
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
  text.replace(/^\uFEFF/, '').split(/\r?\n/).forEach((line, index) => {
    if (!line.trim()) return
    let sample
    try { sample = JSON.parse(line) } catch {
      invalid(index + 1, 'malformed JSON')
      return
    }
    const problem = validateSample(sample)
    if (problem) invalid(index + 1, problem)
    else visit(sample, index + 1)
  })
}

function verifyManifest(manifest) {
  const { primaryScene, trayScene, activeScene, runs } = manifest
  if (![primaryScene, trayScene, activeScene].every(label) || primaryScene === trayScene ||
      activeScene === trayScene || !Array.isArray(runs)) throw new Error('invalid scene configuration')
  const orders = new Set()
  for (const run of runs) {
    if (!systems.includes(run.os) || !arms.includes(run.arm) ||
        ![run.host, run.scene, run.pair, run.file].every(label) ||
        !['comparison', 'control', 'stock-reference'].includes(run.kind) || !pid(run.order) ||
        !Array.isArray(run.windowMs) || run.windowMs.length !== 2 || !run.windowMs.every(quantity) ||
        run.windowMs[0] >= run.windowMs[1] || !Array.isArray(run.requiredGuiRoles) ||
        !run.requiredGuiRoles.every(role => roles.gui.includes(role)) ||
        !['gui', 'service', 'model'].includes(run.requiredGroup ?? 'gui')) {
      throw new Error('invalid run declaration')
    }
    const order = tuple(run.os, run.host, run.order)
    if (orders.has(order)) throw new Error('duplicate run order on one host')
    orders.add(order)
  }
}

function fenceKey(fence) {
  if (!fence || ![fence.serviceBuild, fence.brokerBuild, fence.modelSettings].every(label) ||
      !Number.isSafeInteger(fence.brokerRestartCount) || fence.brokerRestartCount < 0 ||
      ![fence.service, fence.broker].every(row => row && pid(row.pid) && utc(row.createdUtc))) return null
  return tuple(identity(fence.service), identity(fence.broker), fence.serviceBuild,
    fence.brokerBuild, fence.brokerRestartCount, fence.modelSettings)
}

function summarizeRun(run, base) {
  const issues = new Set()
  const samples = []
  const lastCreation = new Map()
  const pidReuse = []
  let previous
  const expectedFence = fenceKey(run.fence?.before)
  if (!expectedFence || !fenceKey(run.fence?.after)) issues.add('missing process/build/settings fence')
  else if (expectedFence !== fenceKey(run.fence.after)) issues.add('service or broker fence changed within run')
  if (run.parity !== true) issues.add('workload or visual parity unverified')
  if (run.attributionClear !== true) issues.add('process attribution unverified')
  if (run.scene !== run.trayScene && run.usableCurrentData !== true) issues.add('usable current data unverified')

  series(readFileSync(resolve(base, run.file), 'utf8'), (sample, line) => {
    if (sample.os !== run.os || (previous && (sample.elapsedMs <= previous.elapsedMs ||
        Date.parse(sample.timestamp) < Date.parse(previous.timestamp)))) {
      issues.add(`line ${line}: platform or sample order mismatch`)
      return
    }
    previous = sample
    for (const row of sample.processes) {
      const old = lastCreation.get(row.pid)
      if (old && old !== row.createdUtc) pidReuse.push({
        pid: row.pid, previousCreatedUtc: old, createdUtc: row.createdUtc, line,
      })
      lastCreation.set(row.pid, row.createdUtc)
    }
    for (const role of ['service', 'broker']) {
      const matching = sample.processes.filter(row => row.group === 'service' && row.role === role)
      if (expectedFence && (matching.length !== 1 ||
          identity(matching[0]) !== identity(run.fence.before[role]))) issues.add(`sampled ${role} identity changed or missing`)
    }
    if (sample.elapsedMs < run.windowMs[0] || sample.elapsedMs > run.windowMs[1]) return
    const gui = sample.processes.filter(row => row.group === 'gui')
    const required = run.requiredGroup === 'gui' || run.requiredGroup === undefined
      ? new Set(['main', ...run.requiredGuiRoles]) : new Set(run.requiredGuiRoles)
    for (const role of required) {
      if (!gui.some(row => row.role === role)) issues.add(`missing major helper: gui/${role}`)
    }
    if (run.requiredGroup === 'model' && !sample.processes.some(row => row.group === 'model')) {
      issues.add('missing model engine')
    }
    samples.push(sample)
  }, (line, reason) => issues.add(`line ${line}: ${reason}`))
  if (samples.length < 2) issues.add('fewer than two steady samples')

  const process = {}
  for (const [group, groupRoles] of Object.entries(roles)) {
    process[group] = {}
    for (const role of ['total', ...groupRoles]) {
      process[group][role] = Object.fromEntries(counters.map(key => {
        const values = samples.map(({ processes }) => {
          const rows = processes.filter(row => row.group === group && (role === 'total' || row.role === role))
          return rows.length && rows.every(row => row[key] !== null)
            ? rows.reduce((sum, row) => sum + row[key], 0) : null
        })
        const observed = values.filter(value => value !== null)
        // CPU deltas need a prior sample of the same process incarnation.
        const usable = key === 'cpuMsPerSec' ? observed.length > 0 : observed.length === samples.length
        return [key, { median: usable ? median(observed) : null, observed: observed.length }]
      }))
    }
  }
  const system = Object.fromEntries(systemCounters.map(key => {
    const values = samples.map(sample => sample.system[key])
    return [key, values.every(quantity) ? median(values) : null]
  }))
  for (const [rate, count] of [['pageInPerSec', 'pageInCount'], ['pageOutPerSec', 'pageOutCount'],
    ['swapInPerSec', 'swapInCount'], ['swapOutPerSec', 'swapOutCount']]) {
    if (system[rate] !== undefined && system[rate] !== null) continue
    const values = samples.slice(1).map((sample, index) => {
      const previous = samples[index]
      const currentCount = sample.system[count]
      const priorCount = previous.system[count]
      return quantity(currentCount) && quantity(priorCount) && currentCount >= priorCount
        ? 1000 * (currentCount - priorCount) / (sample.elapsedMs - previous.elapsedMs) : null
    })
    system[rate] = values.every(quantity) ? median(values) : null
  }
  const outcome = Object.fromEntries(outcomes.map(key => {
    const values = run.measurements?.[key]
    const valid = Array.isArray(values) && values.every(value => quantity(value) &&
      (!key.endsWith('Percent') || value <= 100) && (key !== 'tokensPerSecond' || value > 0))
    return [key, valid ? median(values) : null]
  }))
  const completion = run.measurements?.completionPercent
  if ((run.measurements?.firstTokenMs || run.measurements?.tokensPerSecond) &&
      (!Array.isArray(completion) || !completion.length || !completion.every(value => value === 100))) {
    issues.add('timed inference completion unverified or failed')
  }
  return { ...run, fenceIdentity: expectedFence, issues: [...issues], pidReuse,
    samples: samples.length, process, system, outcome }
}

const guiMemory = run => run.process.gui.total.memoryBytes.median
const definitions = {
  visibleMemoryMiB: [run => guiMemory(run) / MiB, 100, 'lower'],
  trayMemoryMiB: [run => guiMemory(run) / MiB, 50, 'lower'],
  missedFramesPp: [run => run.outcome.missedFramesPercent, 5, 'lower'],
  tokensPerSecondPercent: [run => run.outcome.tokensPerSecond, 5, 'higher-relative'],
  firstTokenPercent: [run => run.outcome.firstTokenMs, 10, 'lower-relative'],
  warmReopenMs: [run => run.outcome.warmReopenMs, 250, 'lower'],
}

function difference(metric, first, second) {
  if (metric.endsWith('MemoryMiB') && [first, second].some(run => guiMemory(run) === null)) return null
  const [value, , direction] = definitions[metric]
  const a = value(first)
  const b = value(second)
  if (a === null || b === null || (direction.endsWith('relative') && a === 0)) return null
  const benefit = direction.startsWith('higher') ? b - a : a - b
  return direction.endsWith('relative') ? 100 * benefit / a : benefit
}

function counterDifferences(baseline, candidate) {
  const subtract = (a, b) => a === null || b === null ? null : b - a
  const process = {}
  for (const [group, groupRoles] of Object.entries(baseline.process)) {
    process[group] = {}
    for (const [role, metrics] of Object.entries(groupRoles)) {
      process[group][role] = Object.fromEntries(Object.entries(metrics).map(([key, value]) =>
        [key, subtract(value.median, candidate.process[group][role][key].median)]))
    }
  }
  const system = Object.fromEntries(Object.entries(baseline.system).map(([key, value]) =>
    [key, subtract(value, candidate.system[key])]))
  const outcome = Object.fromEntries(Object.entries(baseline.outcome).map(([key, value]) =>
    [key, subtract(value, candidate.outcome[key])]))
  return { process, system, outcome }
}

function makePairs(runs) {
  const groups = Map.groupBy(runs, run => tuple(run.os, run.host, run.scene, run.kind, run.pair))
  return [...groups.values()].map(group => {
    const [first] = group
    const sorted = [...group].sort((a, b) => a.order - b.order)
    const [baseline, candidate] = first.kind === 'control' ? sorted : [
      group.find(run => run.arm === (first.kind === 'stock-reference' ? 'stock-electron' : 'matched-electron')),
      group.find(run => run.arm === (first.kind === 'stock-reference' ? 'matched-electron' : 'wails')),
    ]
    const reasons = group.flatMap(run => run.issues.map(reason => `${run.arm} #${run.order}: ${reason}`))
    if (group.length !== 2 || !baseline || !candidate ||
        (first.kind === 'control' && baseline.arm !== candidate.arm)) reasons.push('invalid pair arms')
    else if (baseline.fenceIdentity !== candidate.fenceIdentity) reasons.push('service or broker fence changed across pair')
    return { os: first.os, host: first.host, scene: first.scene, kind: first.kind, id: first.pair,
      orders: sorted.map(run => run.order), reasons,
      fenceIdentity: reasons.length ? null : baseline.fenceIdentity,
      activeUse: group.every(run => run.activeUse === true),
      counterDifferences: reasons.length ? null : counterDifferences(baseline, candidate),
      deltas: reasons.length ? {} : Object.fromEntries(Object.keys(definitions)
        .map(metric => [metric, difference(metric, baseline, candidate)])),
    }
  })
}

function evaluate(metric, comparisons, controls) {
  const threshold = definitions[metric][1]
  const valid = comparisons.filter(pair => !pair.reasons.length)
  const deltas = valid.map(pair => ({
    host: pair.host, pair: pair.id, value: pair.deltas[metric],
  }))
  const values = deltas.map(delta => delta.value).filter(value => value !== null)
  const noisePairs = controls.filter(control => {
    if (control.reasons.length || control.deltas[metric] === null) return false
    const orders = valid.filter(pair => pair.host === control.host &&
      pair.fenceIdentity === control.fenceIdentity).flatMap(pair => pair.orders)
    return orders.length && control.orders[0] > Math.min(...orders) && control.orders.at(-1) < Math.max(...orders)
  })
  const noise = valid.every(pair => noisePairs.some(control =>
    control.host === pair.host && control.fenceIdentity === pair.fenceIdentity)) && noisePairs.length
    ? Math.max(...noisePairs.map(control => Math.abs(control.deltas[metric]))) : null
  const middle = median(values)
  const result = { verdict: 'inconclusive', reason: 'missing valid pairs, decisive measurements or interleaved controls',
    threshold, noise, median: middle, range: values.length ? [Math.min(...values), Math.max(...values)] : null, deltas }
  if (middle === null || noise === null || values.length !== deltas.length) return result
  const limit = Math.max(threshold, noise)
  if (middle > limit && values.every(value => value > 0)) {
    return { ...result, verdict: 'material-win', reason: 'all valid pairs favor candidate and median clears threshold and noise' }
  }
  if (middle < -limit && values.every(value => value < 0)) {
    return { ...result, verdict: 'regression', reason: 'all valid pairs favor baseline and median exceeds tolerance and noise' }
  }
  if (values.some(value => value < -limit) || Math.abs(middle) > limit) {
    return { ...result, reason: 'pairs disagree on direction or a pair exceeds regression tolerance' }
  }
  return { ...result, verdict: 'no-material-win', reason: 'median does not clear threshold and noise' }
}

export function compare(manifestPath) {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8').replace(/^\uFEFF/, ''))
  verifyManifest(manifest)
  const runs = manifest.runs.map(run => summarizeRun({ ...run, trayScene: manifest.trayScene }, dirname(manifestPath)))
  const pairs = makePairs(runs)
  const platforms = Object.fromEntries(systems.map(os => {
    const select = (scene, kind) => pairs.filter(pair => pair.os === os && pair.scene === scene && pair.kind === kind &&
      (scene !== manifest.activeScene || pair.activeUse))
    const metrics = {}
    for (const metric of Object.keys(definitions)) {
      const scene = metric === 'trayMemoryMiB' ? manifest.trayScene : manifest.primaryScene
      metrics[metric] = evaluate(metric, select(scene, 'comparison'), select(scene, 'control'))
    }
    const activeUse = Object.fromEntries(['missedFramesPp', 'tokensPerSecondPercent', 'firstTokenPercent']
      .map(metric => [metric, evaluate(metric, select(manifest.activeScene, 'comparison'), select(manifest.activeScene, 'control'))]))
    const stockReference = evaluate('visibleMemoryMiB', select(manifest.primaryScene, 'stock-reference'),
      select(manifest.primaryScene, 'control'))
    const checks = [...Object.values(metrics), ...Object.values(activeUse)]
    let verdict = metrics.visibleMemoryMiB.verdict
    let reason = metrics.visibleMemoryMiB.reason
    if (checks.some(check => check.verdict === 'regression')) {
      verdict = 'regression'
      reason = 'at least one decisive metric regressed'
    } else if (checks.some(check => check.verdict === 'inconclusive')) {
      verdict = 'inconclusive'
      reason = 'at least one decisive metric or active-use comparison is inconclusive'
    } else if ([metrics.visibleMemoryMiB, metrics.trayMemoryMiB].some(check => check.deltas.length < 3)) {
      verdict = 'inconclusive'
      reason = 'fewer than three valid visible or tray comparisons'
    }
    return [os, { verdict, reason, metrics, activeUse, stockReference }]
  }))
  return { platforms, pairs, runs }
}

function markdown({ platforms }) {
  const lines = ['| OS | Metric | Median benefit | Same-host noise | Verdict |',
    '| --- | --- | ---: | ---: | --- |']
  const display = value => value === null ? 'unknown' : Number(value.toFixed(2))
  for (const [os, result] of Object.entries(platforms)) {
    const rows = [...Object.entries(result.metrics),
      ...Object.entries(result.activeUse).map(([metric, row]) => [`Active use / ${metric}`, row]),
      ['Stock reference / visibleMemoryMiB', result.stockReference]]
    for (const [metric, row] of rows) {
      lines.push(`| ${os} | ${metric} | ${display(row.median)} | ${display(row.noise)} | ${row.verdict} |`)
    }
    lines.push(`| ${os} | Overall | | | ${result.verdict} |`)
  }
  return lines.join('\n')
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const [input, samples] = process.argv.slice(2)
    if (input === '--validate-samples' && samples) {
      let count = 0
      series(readFileSync(samples === '-' ? 0 : samples, 'utf8'), () => count++,
        (line, reason) => { throw new Error(`line ${line}: ${reason}`) })
      if (!count) throw new Error('no samples')
      console.log(`Validated ${count} samples`)
    } else {
      if (!input || samples) throw new Error('usage: compare.mjs MANIFEST.json | --validate-samples FILE_OR_DASH')
      const summary = compare(resolve(input))
      console.log(JSON.stringify(summary, null, 2))
      console.error(markdown(summary))
    }
  } catch (error) {
    console.error(error.message)
    process.exitCode = 1
  }
}
