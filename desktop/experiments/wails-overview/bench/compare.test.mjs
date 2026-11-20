// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'

const path = file => fileURLToPath(new URL(file, import.meta.url))
const expected = JSON.parse(readFileSync(path('fixtures/expected-summary.json'), 'utf8'))
const result = spawnSync(process.execPath, [path('compare.mjs'), path('fixtures/manifest.json')],
  { encoding: 'utf8', maxBuffer: 4 * 1024 ** 2 })
assert.equal(result.status, 0, result.stderr)
const summary = JSON.parse(result.stdout)

for (const os of ['windows', 'macos']) {
  test(`${os} verdict uses steady medians, signed pair deltas and same-host noise`, () => {
    const platform = summary.platforms[os]
    const { verdict, threshold, median, noise, range, deltas } = platform.metrics.visibleMemoryMiB
    assert.deepEqual({ verdict, threshold, median, noise, range, deltas: deltas.map(row => row.value) },
      expected[os].visibleMemory)
    assert.equal(platform.verdict, expected[os].verdict)
    assert.match(result.stderr, new RegExp(`\\| ${os} \\| Overall \\| \\| \\| ${platform.verdict} \\|`))
  })
}

test('PID reuse is reported and does not carry CPU history into the new process', () => {
  const run = summary.runs.find(row => row.os === 'windows' && row.scene === 'visible' &&
    row.pair === 'a' && row.arm === 'wails')
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
    assert.ok(pair.reasons.some(reason => reason.endsWith(expected.rejected[id])), pair.reasons.join('\n'))
    assert.deepEqual(pair.deltas, {})
    assert.equal(pair.counterDifferences, null)
    assert.ok(!summary.platforms.windows.metrics.visibleMemoryMiB.deltas.some(row => row.pair === id))
  })
}

test('raw schema validation fails the CLI on the invalid sample', () => {
  const invalid = spawnSync(process.execPath, [path('compare.mjs'), '--validate-samples',
    path('fixtures/windows-invalid.txt')], { encoding: 'utf8' })
  assert.equal(invalid.status, 1)
  assert.equal(invalid.stderr.trim(), expected.rejected.invalid)
})
