// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict'
import { test } from 'node:test'
import { cpuDeltas, parseSelectors, selectProcesses } from './sample-macos.mjs'

const started = '2026-11-20T00:00:00.000Z'
const replacement = '2026-11-20T00:00:02.000Z'
const webContent = '/WebKit/com.apple.WebKit.WebContent'
const processRow = (pid, parent, image, createdUtc = started) => ({
  pid, parent, image, name: image.split('/').at(-1), createdUtc, startMs: Date.parse(createdUtc),
})
const snapshot = (...rows) => new Map(rows.map(row => [row.pid, row]))
const main = processRow(100, 1, '/Trial/Wails')
const owned = processRow(101, 1, webContent)
const unrelated = processRow(102, 1, webContent)
const renderer = ['--renderer', 'WebContent$']
const verified = ['--helper', `101:${started}:renderer`]
const gui = ['--gui-path', '/Trial/Wails', ...renderer]
const projected = selection => [...selection.selected].map(([pid, attribution]) => [pid, attribution.role])

test('an exact helper incarnation selects one detached WebKit renderer with its declared role', () => {
  const rows = snapshot(main, owned, unrelated)
  const ancestry = selectProcesses(rows, parseSelectors(gui))
  assert.deepEqual(projected(ancestry), [[100, 'main']])

  const explicit = selectProcesses(rows, parseSelectors([...gui, ...verified, '--gui-pid', '101']))
  assert.deepEqual(projected(explicit), [[100, 'main'], [101, 'renderer']])
  assert.equal(explicit.attributionClear, true)

  const reused = selectProcesses(snapshot(main, processRow(101, 1, webContent, replacement), unrelated),
    parseSelectors([...gui, ...verified, '--gui-pid', '101']), explicit.known)
  assert.deepEqual(projected(reused), [[100, 'main']])
})

test('a broad regex leaves detached WebKit attribution ambiguous even with a verified helper', () => {
  const rows = snapshot(main, owned, unrelated)
  const broad = selectProcesses(rows, parseSelectors([...gui, ...verified, '--helper-pattern', 'WebContent$']))
  assert.deepEqual(projected(broad), [[100, 'main'], [101, 'renderer'], [102, 'renderer']])
  assert.equal(broad.attributionClear, false)
  const retained = selectProcesses(rows, parseSelectors([...gui, ...verified]), broad.known)
  assert.equal(retained.attributionClear, false)
})

test('explicit helpers reject a main role or conflicting roles for one incarnation', () => {
  assert.throws(() => parseSelectors(['--helper', `101:${started}:main`]), /invalid helper selector/)
  assert.throws(() => parseSelectors([...verified, '--helper', `101:${started}:gpu`]), /conflicting helper roles/)
})

test('CPU deltas use only the previous counter for the same process incarnation', () => {
  const prior = [{ pid: 5, createdUtc: started, cpuSeconds: 1 }, { pid: 7, createdUtc: started, cpuSeconds: 18 }]
  const current = [{ pid: 5, createdUtc: replacement, cpuSeconds: 10 },
    { pid: 7, createdUtc: started, cpuSeconds: 18.125 }]
  assert.deepEqual(cpuDeltas(prior, current, 500), [null, 250])
  assert.deepEqual(cpuDeltas(current, [{ pid: 5, createdUtc: replacement, cpuSeconds: 10.25 }], 500), [500])
})
