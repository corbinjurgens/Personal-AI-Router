// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from 'node:child_process'
import { basename } from 'node:path'
import { setTimeout } from 'node:timers/promises'
import { parseArgs } from 'node:util'

const serviceNames = /^nvpair-(service|ui-broker|cluster-manager|engine-manager|errors|job-scheduler|manual-nodes|node-info|node-scanner|node-settings|proxy|workload-manager)$/

function command(file, ...args) {
  const result = spawnSync(file, args, { encoding: 'utf8', maxBuffer: 4 * 1024 ** 2, timeout: 8000 })
  return result.status === 0 ? result.stdout : null
}

const key = row => `${row.pid}/${row.createdUtc}`

function snapshot() {
  const output = command('/bin/ps', '-axww', '-o', 'pid=,ppid=,lstart=,time=,comm=')
  if (!output) throw new Error('cannot read macOS process snapshot')
  const rows = new Map()
  for (const line of output.split('\n')) {
    const match = line.match(/^\s*(\d+)\s+(\d+)\s+(\w+\s+\w+\s+\d+\s+[\d:]+\s+\d+)\s+([\d:.]+)\s+(.*)$/)
    if (!match) continue
    const [, pid, parent, started, time, image] = match
    const startMs = Date.parse(`${started} UTC`)
    if (!Number.isFinite(startMs)) continue
    const cpuSeconds = time.split(':').reduce((sum, interval) => sum * 60 + Number(interval), 0)
    rows.set(Number(pid), { pid: Number(pid), parent: Number(parent), startMs,
      createdUtc: new Date(startMs).toISOString(), cpuSeconds, image, name: basename(image) })
  }
  if (!rows.has(process.pid)) throw new Error('cannot parse macOS process snapshot')
  return rows
}

function descendants(rows, selected, classify) {
  let changed
  do {
    changed = false
    for (const row of rows.values()) {
      if (selected.has(row.pid)) continue
      const parent = rows.get(row.parent)
      if (!parent || !selected.has(parent.pid) || parent.startMs > row.startMs) continue
      selected.set(row.pid, classify(row, selected.get(parent.pid)))
      changed = true
    }
  } while (changed)
}

function footprint(pids) {
  if (!pids.length) return new Map()
  const output = command('/usr/bin/top', '-l', '1', '-stats', 'pid,mem', '-n', String(pids.length),
    ...pids.flatMap(pid => ['-pid', String(pid)]))
  const units = { B: 1, K: 1024, M: 1024 ** 2, G: 1024 ** 3, T: 1024 ** 4 }
  const result = new Map()
  for (const line of (output ?? '').split('\n')) {
    const match = line.match(/^\s*(\d+)\s+([\d.]+)([BKMGT]?)[+-]?\s*$/)
    if (match) result.set(Number(match[1]), Math.round(Number(match[2]) * units[match[3] || 'B']))
  }
  return result
}

function systemCounters() {
  const vm = command('/usr/bin/vm_stat') ?? ''
  const field = name => {
    const match = vm.match(new RegExp(`^${name}:\\s+(\\d+)\\.`, 'm'))
    return match ? Number(match[1]) : null
  }
  const pageSize = vm.match(/page size of (\d+) bytes/)
  const pressure = command('/usr/bin/memory_pressure', '-Q')?.match(/free percentage: (\d+)%/)
  return {
    availableBytes: null, pageInPerSec: null, pageOutPerSec: null,
    pageSizeBytes: pageSize ? Number(pageSize[1]) : null,
    pageInCount: field('Pageins'), pageOutCount: field('Pageouts'),
    swapInCount: field('Swapins'), swapOutCount: field('Swapouts'),
    pressureFreePercent: pressure ? Number(pressure[1]) : null,
  }
}

async function sample() {
  const options = Object.fromEntries(['gui-path', 'gui-name', 'gui-pid'].map(name =>
    [name, { type: 'string', multiple: true, default: [] }]))
  for (const name of ['helper', 'renderer', 'gpu', 'service', 'model', 'interval', 'duration']) {
    options[name] = { type: 'string' }
  }
  const { values } = parseArgs({ options })
  const paths = new Set(values['gui-path'])
  const names = new Set(values['gui-name'])
  const pids = new Set(values['gui-pid'].map(Number))
  const interval = Number(values.interval ?? 1) * 1000
  const duration = Number(values.duration ?? 300) * 1000
  if (![interval, duration].every(value => Number.isFinite(value) && value >= 1) ||
      [...pids].some(pid => !Number.isSafeInteger(pid) || pid <= 0)) throw new Error('invalid sampler arguments')
  const pattern = name => values[name] ? new RegExp(values[name], 'i') : /(?!)/
  const helper = pattern('helper')
  const renderer = pattern('renderer')
  const gpu = pattern('gpu')
  const service = values.service ? pattern('service') : serviceNames
  const model = pattern('model')
  const guiRole = row => gpu.test(row.image) ? 'gpu' : renderer.test(row.image) ? 'renderer' : 'other'
  const seeds = new Set()
  let previous = new Map()
  let known = new Map()
  let previousMs = 0
  const started = performance.now()
  for (let next = 0; next < duration;) {
    const wait = started + next - performance.now()
    if (wait > 0) await setTimeout(wait)
    const elapsedMs = performance.now() - started
    if (next > 0 && elapsedMs >= duration) break
    const rows = snapshot()
    const timestamp = new Date().toISOString()
    const excluded = new Map([[process.pid, true]])
    descendants(rows, excluded, () => true)
    for (const pid of excluded.keys()) rows.delete(pid)
    const selected = new Map()
    for (const row of rows.values()) {
      const id = key(row)
      if (next === 0 && pids.has(row.pid)) seeds.add(id)
      if (model.test(row.image)) selected.set(row.pid, { group: 'model', role: 'engine' })
      else if (service.test(row.name)) {
        const role = row.name === 'nvpair-service' ? 'service' : row.name === 'nvpair-ui-broker' ? 'broker' : 'worker'
        selected.set(row.pid, { group: 'service', role })
      } else if (paths.has(row.image) || names.has(row.name) || seeds.has(id)) {
        selected.set(row.pid, { group: 'gui', role: 'main' })
      } else if (helper.test(row.image)) selected.set(row.pid, { group: 'gui', role: guiRole(row) })
      else if (known.has(id)) selected.set(row.pid, known.get(id))
    }
    descendants(rows, selected, (row, parent) => parent.group === 'gui' ? { group: 'gui', role: guiRole(row) }
      : { group: parent.group, role: parent.group === 'service' ? 'worker' : 'engine' })
    const memory = footprint([...selected.keys()])
    const after = snapshot()
    const processes = []
    const current = new Map()
    known = new Map()
    for (const [pid, attribution] of selected) {
      const row = rows.get(pid)
      const id = key(row)
      const last = previous.get(id)
      const cpuMsPerSec = last !== undefined && row.cpuSeconds >= last
        ? (row.cpuSeconds - last) * 1e6 / (elapsedMs - previousMs) : null
      const stillPresent = after.has(pid) && key(after.get(pid)) === id
      processes.push({ pid, createdUtc: row.createdUtc, ...attribution,
        memoryBytes: stillPresent ? memory.get(pid) ?? null : null,
        cpuMsPerSec, cpuPercentOneCore: null, commitBytes: null,
        gpuEnginePercent: null, gpuDedicatedBytes: null, gpuSharedBytes: null })
      current.set(id, row.cpuSeconds)
      known.set(id, attribution)
    }
    console.log(JSON.stringify({ schema: 1, os: 'macos', elapsedMs, timestamp, processes, system: systemCounters() }))
    previousMs = elapsedMs
    previous = current
    next += interval
    while (started + next < performance.now()) next += interval
  }
}

try {
  await sample()
} catch (error) {
  console.error(error.message)
  process.exitCode = 1
}
