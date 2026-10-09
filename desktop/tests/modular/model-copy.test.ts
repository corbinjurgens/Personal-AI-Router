// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
    isModelCopyTerminal,
    modelCopyProgress,
    parseModelCopyFrame
} from '@/electron/service-bridge/model-copy'
import { isEngineCopyInProgress } from '@/shared/utils/engine-progress'
import { formatCopyProgressLabel } from '@/ui/utils/formatters'

describe('parseModelCopyFrame', () => {
    it('reads the source node, engine and optional progress fields', () => {
        const frame = parseModelCopyFrame({
            node: 'peer-1',
            engine: 'ollama',
            op: 'copy',
            stage: 'downloading',
            percent: 42.5,
            file: 'blob-1',
            bytesDone: 10,
            bytesTotal: 100
        })
        expect(frame).toEqual({
            nodeId: 'peer-1',
            engineType: 'ollama',
            stage: 'downloading',
            percent: 42.5,
            file: 'blob-1',
            bytesDone: 10,
            bytesTotal: 100,
            message: undefined
        })
    })

    it('defaults the stage, caps the percent and drops bad numbers', () => {
        const frame = parseModelCopyFrame({
            node: 'peer-1',
            engine: 'ollama',
            percent: 140,
            bytesDone: -1
        })
        expect(frame?.stage).toBe('working')
        expect(frame?.percent).toBe(100)
        expect(frame?.bytesDone).toBeUndefined()
    })

    it('rejects frames without a node or a known engine', () => {
        expect(parseModelCopyFrame({ engine: 'ollama' })).toBeNull()
        expect(parseModelCopyFrame({ node: 'peer-1', engine: 'nope' })).toBeNull()
        expect(parseModelCopyFrame(undefined)).toBeNull()
        expect(parseModelCopyFrame([])).toBeNull()
    })
})

describe('copy progress', () => {
    it('ends on done or error only', () => {
        expect(isModelCopyTerminal('done')).toBe(true)
        expect(isModelCopyTerminal('error')).toBe(true)
        expect(isModelCopyTerminal('downloading')).toBe(false)
    })

    it('builds an entry keyed on the source node and model', () => {
        const frame = parseModelCopyFrame({
            node: 'peer-1',
            engine: 'ollama',
            stage: 'verifying',
            bytesDone: 5,
            bytesTotal: 9
        })
        expect(frame).not.toBeNull()
        if (!frame) return
        const progress = modelCopyProgress(frame, 'qwen3:4b', 'Peer')
        expect(progress).toMatchObject({
            nodeId: 'peer-1',
            nodeName: 'Peer',
            operation: 'copy',
            model: 'qwen3:4b',
            status: 'verifying',
            completed: 5,
            total: 9
        })
        expect(isEngineCopyInProgress(progress)).toBe(true)
        expect(isEngineCopyInProgress({ operation: 'copy', status: 'idle' })).toBe(false)
        expect(isEngineCopyInProgress({ operation: 'pull', status: 'downloading' })).toBe(false)
    })

    it('labels the stage with the percent when known', () => {
        expect(formatCopyProgressLabel({ status: 'starting' })).toBe(
            'Copying to this PC · starting'
        )
        expect(formatCopyProgressLabel({ status: 'downloading', percent: 41.6 })).toBe(
            'Copying to this PC · downloading · 42%'
        )
    })
})
