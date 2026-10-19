// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
    EnabledEngineTypes,
    EngineDefaultLinks,
    EngineDisplayNames,
    EngineManagerNames,
    EngineTypes
} from '@/shared/constants/engines'
import {
    engineManagerName,
    engineTypeFromManagerName,
    isEngineType,
    isValidEngineId,
    isValidRoutingEngineId
} from '@/shared/utils/engines'
import { EngineCapabilities } from '@/ui/constants/engine-capabilities'
import { getWelcomeEngineCandidates, WELCOME_ENGINE_DEFAULT_SELECTED } from '@/ui/constants/welcome'
import { gatewayEndpointDisplayUrl } from '@/ui/utils/gateway-inference-paths'

describe('engine identity', () => {
    it('enables llama.cpp workflows on every platform', () => {
        expect(EngineTypes).toEqual(['ollama', 'lm-studio', 'llama-cpp'])
        expect(EnabledEngineTypes).toEqual(['ollama', 'lm-studio', 'llama-cpp'])
        expect(getWelcomeEngineCandidates('Windows')).toContain('llama-cpp')
        expect(getWelcomeEngineCandidates('MacOS')).toContain('llama-cpp')
        expect(getWelcomeEngineCandidates('Linux')).toContain('llama-cpp')
    })

    it('preselects every engine during onboarding', () => {
        expect(WELCOME_ENGINE_DEFAULT_SELECTED).toEqual({
            ollama: true,
            'lm-studio': true,
            'llama-cpp': true
        })
    })

    it('maps the desktop id to the sole engine-manager wire id', () => {
        expect(EngineManagerNames['llama-cpp']).toBe('llamacpp')
        expect(engineManagerName('llama-cpp')).toBe('llamacpp')
        expect(engineTypeFromManagerName('llamacpp')).toBe('llama-cpp')
        expect(isEngineType('llama-cpp')).toBe(true)
    })

    it('provides complete display metadata and capabilities', () => {
        for (const engineType of EngineTypes) {
            expect(EngineDisplayNames[engineType]).toBeTruthy()
            expect(EngineDefaultLinks[engineType].docsUrl).toMatch(/^https:\/\//)
            expect(EngineDefaultLinks[engineType].installUrl).toMatch(/^https:\/\//)
            expect(EngineCapabilities[engineType]).toBeDefined()
        }

        expect(EngineDisplayNames['llama-cpp']).toBe('llama.cpp')
        expect(EngineCapabilities['llama-cpp']).toMatchObject({
            hasExpiry: false,
            hasEject: true,
            hasInstall: ['win32', 'darwin', 'linux'],
            hasEnginePort: true,
            hasInstallPath: false,
            hasProxyWebUI: false,
            hasPreferredNode: false,
            hasCrashAlert: false,
            hasModelSearchOnlyWhenRunning: true,
            modelOpsWhenStopped: false,
            hasDeleteModel: true,
            engineHub: { label: 'llama.cpp' }
        })
        expect(gatewayEndpointDisplayUrl(8080, 'llama-cpp')).toBe('http://127.0.0.1:8080')
    })
})

describe('routing engine IDs match Go engines.ValidRoutingID', () => {
    it.each([
        'ollama',
        'lmstudio',
        'llamacpp',
        'llamacpp-custom',
        'strata',
        'errors',
        'e-x',
        'lm-studio',
        'llama-cpp',
        'A_1.-',
        '-engine',
        '_engine',
        '.engine',
        '...',
        'a'.repeat(40)
    ])('accepts %j without requiring a local descriptor', id => {
        expect(isValidRoutingEngineId(id)).toBe(true)
    })

    it.each([
        { name: 'empty', id: '' },
        { name: 'dot', id: '.' },
        { name: 'dotdot', id: '..' },
        { name: 'slash', id: 'a/b' },
        { name: 'backslash', id: 'a\\b' },
        { name: 'colon', id: 'a:b' },
        { name: 'space', id: 'a b' },
        { name: 'overlong', id: 'a'.repeat(41) },
        { name: 'final newline', id: 'a\n' },
        { name: 'embedded newline', id: 'a\nb' },
        { name: 'carriage return', id: 'a\r' },
        { name: 'CRLF', id: 'a\r\n' },
        { name: 'line separator', id: 'a\u2028' },
        { name: 'paragraph separator', id: 'a\u2029' },
        { name: 'unicode', id: 'engin\u00e9' },
        { name: 'unpaired surrogate', id: 'a\ud800' },
        { name: 'tab', id: 'a\t' },
        { name: 'NUL', id: 'a\0' }
    ])('rejects $name', ({ id }) => {
        expect(isValidRoutingEngineId(id)).toBe(false)
    })
})

describe('descriptor engine and family IDs match Go engines.ValidID', () => {
    it('keeps generic IDs unbounded by the routing limit', () => {
        const id = 'a'.repeat(41)
        expect(isValidEngineId(id)).toBe(true)
        expect(engineTypeFromManagerName(id)).toBe(id)
        expect(isValidRoutingEngineId(id)).toBe(false)
    })

    it.each(['a\n', 'a\r', 'a\r\n', 'a\u2028', 'a\u2029'])(
        'rejects final line terminators in %j',
        id => {
            expect(isValidEngineId(id)).toBe(false)
            expect(engineTypeFromManagerName(id)).toBeNull()
        }
    )
})
