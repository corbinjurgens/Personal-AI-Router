// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() },
    BrowserWindow: { getAllWindows: () => [] }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))

import { EngineDisplayNames, EngineTypes } from '@/shared/constants/engines'
import type { EngineDescriptor, EngineStatusData, EngineType } from '@/shared/types/engines'
import type { JsonValue } from '@/electron/service-bridge/json-rpc-client'
import {
    engineAdapterType,
    engineLabel,
    engineManagerName,
    engineTypeFromManagerName,
    isEngineType
} from '@/shared/utils/engines'
import { parseEngineDescriptors } from '@/electron/service-bridge/engine-descriptors'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { getModularSupervisor, pullModelParams } from '@/electron/service-bridge/modular-supervisor'
import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import {
    EngineCapabilities,
    SAFE_DEFAULT_ENGINE_CAPS,
    engineCapabilities
} from '@/ui/constants/engine-capabilities'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { engineTypesForNode } from '@/ui/utils/get-engines-for-node'
import EngineIcon from '@/ui/components/EngineIcon'
import ModelRow from '@/ui/components/ModelManager/ModelRow'
import type { ModelItem } from '@/shared/types/engines'
import type { EngineCaps } from '@/ui/types/engine-manifest'
import type { EngineCommandType } from '@/shared/types/engine-api'

/** One `engine:descriptors` entry as engine-manager publishes it. */
function wireDescriptor(
    engine: string,
    family: string,
    label: string,
    operations: Record<string, Record<string, number>>,
    adapters: { modelStore?: string; catalog?: string } = {}
): JsonValue {
    return {
        version: 1,
        engine,
        family,
        label,
        platforms: ['darwin/arm64', 'linux/amd64', 'windows/amd64'],
        available: true,
        protocols: ['openai-v1'],
        modelNaming: 'exact',
        ...adapters,
        operations
    }
}

/** A llama.cpp fork with load and unload but no pull or delete. */
const FORK_WIRE = wireDescriptor('llamacpp-custom', 'llamacpp', 'llama.cpp (custom)', {
    load: {},
    unload: {}
})
/** An engine of a family this build has no adapters for. */
const STRATA_WIRE = wireDescriptor('strata', 'strata', 'Strata', { load: {}, unload: {} })
const OLLAMA_WIRE = wireDescriptor(
    'ollama',
    'ollama',
    'Ollama',
    { load: { responseTimeoutS: 600 }, unload: {}, pull: {}, delete: {} },
    { modelStore: 'ollama', catalog: 'ollama' }
)

function engineId(name: string): EngineType {
    const engineType = engineTypeFromManagerName(name)
    if (!engineType) throw new Error(`${name} is not a valid engine id`)
    return engineType
}

function descriptorsFrom(...entries: JsonValue[]): Map<EngineType, EngineDescriptor> {
    return new Map(parseEngineDescriptors({ engines: entries }).map(d => [d.engineType, d]))
}

function downloadedModel(name: string): ModelItem {
    return {
        name,
        size: 1,
        downloaded: true,
        status: 'idle',
        parameterSize: '',
        quantization: '',
        family: '',
        digest: '',
        sizeVram: null,
        expiresAt: null,
        expiry: '-1',
        capabilities: []
    }
}

let container: HTMLDivElement
let root: Root

function render(element: ReturnType<typeof createElement>): void {
    act(() => root.render(element))
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
    useEngineStatusStore.setState({ statusByNode: new Map(), descriptors: new Map() })
})

afterEach(() => {
    act(() => root.unmount())
    container.remove()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
})

describe('engine descriptor parsing', () => {
    it('keeps valid descriptors and drops ones this build cannot honor', () => {
        const parsed = parseEngineDescriptors({
            engines: [
                FORK_WIRE,
                { ...wireDescriptorObject(STRATA_WIRE), version: 2 },
                { ...wireDescriptorObject(STRATA_WIRE), engine: 'other', operations: { warm: {} } },
                { ...wireDescriptorObject(STRATA_WIRE), engine: 'proto', protocols: ['grpc'] },
                { ...wireDescriptorObject(STRATA_WIRE), engine: '../escape' }
            ]
        })

        expect(parsed).toEqual([
            {
                engineType: 'llamacpp-custom',
                family: 'llamacpp',
                label: 'llama.cpp (custom)',
                available: true,
                protocols: ['openai-v1'],
                modelNaming: 'exact',
                modelStore: '',
                catalog: '',
                operations: { load: { responseTimeoutS: 0 }, unload: { responseTimeoutS: 0 } }
            }
        ])
    })

    it('maps built-in engines to their desktop spelling and refuses a second identity for them', () => {
        expect(parseEngineDescriptors({ engines: [OLLAMA_WIRE] })[0].engineType).toBe('ollama')
        expect(engineTypeFromManagerName('lmstudio')).toBe('lm-studio')
        expect(engineTypeFromManagerName('llamacpp-custom')).toBe('llamacpp-custom')
        expect(engineManagerName(engineId('llamacpp-custom'))).toBe('llamacpp-custom')
        // The desktop spelling of a built-in engine is not a variant of it.
        expect(engineTypeFromManagerName('lm-studio')).toBeNull()
        expect(engineTypeFromManagerName('..')).toBeNull()
        expect(isEngineType('a/b')).toBe(false)
    })
})

describe('built-in engines keep their presentation', () => {
    it('uses the curated label, capabilities and icon whatever the descriptor says', () => {
        const restrictive = descriptorsFrom(
            wireDescriptor('ollama', 'ollama', 'Something else', {})
        ).get('ollama')
        for (const engineType of EngineTypes) {
            expect(engineCapabilities(engineType, restrictive)).toBe(EngineCapabilities[engineType])
            expect(engineLabel(engineType, restrictive)).toBe(EngineDisplayNames[engineType])
            expect(engineAdapterType(engineType, undefined)).toBe(engineType)
        }

        render(createElement(EngineIcon, { type: 'ollama' }))
        expect(container.querySelector('img')?.getAttribute('alt')).toBe('Ollama')
        render(createElement(EngineIcon, { type: 'lm-studio' }))
        expect(container.querySelector('img')?.getAttribute('alt')).toBe('LM Studio')
        render(createElement(EngineIcon, { type: 'llama-cpp' }))
        expect(container.querySelector('img')?.getAttribute('alt')).toBe('llama.cpp')
    })

    it('lists the built-in engines first and in order, then reported variants by label', () => {
        const fork = engineId('llamacpp-custom')
        const strata = engineId('strata')
        const shell = engineId('windows-only')
        const status = (engineType: EngineType, installed: boolean): EngineStatusData => ({
            engineType,
            nodeId: 'node',
            processStatus: installed ? 'running' : 'not-installed',
            enginePort: null,
            proxyPort: null
        })
        const statusByNode = new Map([
            [
                'node',
                new Map([
                    [strata, status(strata, true)],
                    [fork, status(fork, true)],
                    // Registered but not installed here: PAIR cannot install it,
                    // so it is not offered.
                    [shell, status(shell, false)]
                ])
            ]
        ])

        expect(
            engineTypesForNode('node', statusByNode, descriptorsFrom(FORK_WIRE, STRATA_WIRE))
        ).toEqual(['ollama', 'lm-studio', 'llama-cpp', 'llamacpp-custom', 'strata'])
        expect(engineTypesForNode('other', statusByNode, new Map())).toEqual([
            'ollama',
            'lm-studio',
            'llama-cpp'
        ])
    })
})

describe('unknown engines fall back safely', () => {
    it('gives a variant of a known family that family icon', () => {
        const fork = engineId('llamacpp-custom')
        useEngineStatusStore.setState({ descriptors: descriptorsFrom(FORK_WIRE) })

        render(createElement(EngineIcon, { type: fork }))

        expect(container.querySelector('img')?.getAttribute('alt')).toBe('llama.cpp')
    })

    it('gives an unknown family and an engine with no descriptor the generic icon and its id', () => {
        const strata = engineId('strata')
        const peerOnly = engineId('peer-variant')
        const descriptors = descriptorsFrom(STRATA_WIRE)
        useEngineStatusStore.setState({ descriptors })

        render(createElement(EngineIcon, { type: strata }))
        expect(container.querySelector('img')).toBeNull()
        expect(container.querySelector('[role="img"]')?.getAttribute('aria-label')).toBe('Strata')
        expect(engineAdapterType(strata, descriptors.get(strata))).toBeNull()

        render(createElement(EngineIcon, { type: peerOnly }))
        expect(container.querySelector('[role="img"]')?.getAttribute('aria-label')).toBe(
            'peer-variant'
        )
        expect(engineLabel(peerOnly, undefined)).toBe('peer-variant')
        expect(engineCapabilities(peerOnly, undefined)).toBe(SAFE_DEFAULT_ENGINE_CAPS)
    })
})

describe('unsupported operations are not offered', () => {
    it('derives model actions, hub and copy from the descriptor', () => {
        const fork = engineId('llamacpp-custom')
        const caps = engineCapabilities(fork, descriptorsFrom(FORK_WIRE).get(fork))

        expect(caps).toMatchObject({
            hasLoad: true,
            hasEject: true,
            hasDeleteModel: false,
            hasPull: false,
            hasModelCopy: false,
            hasInstall: [],
            hasProxyWebUI: false
        })
        expect(caps.engineHub).toBeUndefined()

        // A variant whose family has a catalogue and a store gets both.
        const full = wireDescriptor(
            'llamacpp-main',
            'llamacpp',
            'llama.cpp (main)',
            { load: {}, unload: {}, pull: {}, delete: {} },
            { modelStore: 'llamacpp', catalog: 'llamacpp' }
        )
        const main = engineId('llamacpp-main')
        expect(engineCapabilities(main, descriptorsFrom(full).get(main))).toMatchObject({
            hasPull: true,
            hasDeleteModel: true,
            hasModelCopy: true,
            engineHub: EngineCapabilities['llama-cpp'].engineHub
        })
    })

    it('renders no action menu for an engine that supports no model operation', () => {
        const row = (capabilities: EngineCaps) =>
            createElement(ModelRow, {
                model: downloadedModel('demo'),
                isRunning: true,
                capabilities,
                canCopy: false,
                displayName: (name: string) => name,
                onAction: () => {},
                onExpiryChange: () => {}
            })
        const actionMenu = () => container.querySelector('[aria-label="Open demo actions"]')

        render(row(EngineCapabilities['llama-cpp']))
        expect(actionMenu()).not.toBeNull()

        render(row(SAFE_DEFAULT_ENGINE_CAPS))
        expect(actionMenu()).toBeNull()
    })
})

describe('local engine variants reach the UI', () => {
    it('publishes a configured variant with its descriptor, and a peer-only one with none', () => {
        const state = getModularBridgeState()
        const selfId = 'descriptor-local'
        const peerId = 'descriptor-peer'
        state.setSelfId(selfId)
        state.setEngineDescriptors(parseEngineDescriptors({ engines: [FORK_WIRE, OLLAMA_WIRE] }))
        state.applyEngineManagerStatus({
            engine: 'llamacpp-custom',
            installed: true,
            running: true,
            healthy: true,
            port: 8090
        })
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: {
                nodes: [
                    {
                        hostUuid: peerId,
                        name: 'descriptor-peer-host',
                        ipAddress: '10.0.0.9',
                        port: 14318,
                        models: ['strata-model'],
                        modelsByEngine: { strata: ['strata-model'] }
                    }
                ]
            }
        })
        state.applyRemoteEngineFacts(peerId, {
            engines: [{ engine: 'strata', installed: true, running: true, port: 9000 }]
        })

        const initial = state.getEngineInitialState()
        expect(initial.descriptors.map(d => d.engineType).sort()).toEqual([
            'llamacpp-custom',
            'ollama'
        ])
        expect(
            initial.statuses.find(s => s.nodeId === selfId && s.engineType === 'llamacpp-custom')
        ).toMatchObject({ processStatus: 'running', enginePort: 8090, proxyPort: null })
        expect(
            initial.statuses.find(s => s.nodeId === peerId && s.engineType === 'strata')
        ).toMatchObject({ processStatus: 'running', enginePort: 9000, proxyPort: null })
        expect(
            initial.models.find(m => m.nodeId === peerId && m.engineType === 'strata')?.models
        ).toEqual([expect.objectContaining({ name: 'strata-model' })])
        expect(state.getEngineDescriptor(engineId('strata'))).toBeUndefined()
    })
})

describe('model operations use the typed engine-manager methods', () => {
    it('loads, unloads and deletes a variant without spelling its action body', async () => {
        const state = getModularBridgeState()
        const supervisor = getModularSupervisor()
        state.setSelfId('typed-ops-local')
        const send = vi.spyOn(supervisor, 'sendProcess').mockImplementation(() => {})
        vi.spyOn(supervisor, 'hasProcess').mockReturnValue(true)
        const call = vi.spyOn(supervisor, 'callProcess').mockImplementation(
            async (_name, method): Promise<JsonValue> =>
                method === 'engine:models'
                    ? {
                          models: [],
                          modelsByEngine: { 'llamacpp-custom': [] },
                          loadedByEngine: {},
                          downloadState: {
                              version: 1,
                              sourceId: 'd'.repeat(32),
                              revision: '1',
                              recoveryRequired: false,
                              entries: []
                          }
                      }
                    : {}
        )
        const fork = engineId('llamacpp-custom')

        const commands: EngineCommandType[] = ['loadModel', 'unloadModel']
        for (const command of commands) {
            await handleServiceBridgeInvoke('engine:command', {
                command,
                engineType: fork,
                nodeId: 'typed-ops-local',
                model: 'org/model:Q4'
            })
        }
        expect(send.mock.calls.map(([, method, params]) => [method, params])).toEqual([
            ['engine:load-model', { engine: 'llamacpp-custom', model: 'org/model:Q4' }],
            ['engine:unload-model', { engine: 'llamacpp-custom', model: 'org/model:Q4' }]
        ])

        // deleteModel refuses before the broker is ready; mark it ready.
        Object.assign(supervisor, { brokerReady: true })
        try {
            await supervisor.deleteModel('llamacpp-custom', fork, 'org/model:Q4')
        } finally {
            Object.assign(supervisor, { brokerReady: false })
        }
        expect(call.mock.calls[0].slice(0, 3)).toEqual([
            'broker',
            'engine:delete-model',
            { engine: 'llamacpp-custom', model: 'org/model:Q4' }
        ])
        expect(call).toHaveBeenNthCalledWith(2, 'broker', 'engine:models')
    })

    it('spells a pull by the engine family, which has no typed method yet', () => {
        const state = getModularBridgeState()
        state.setEngineDescriptors(
            parseEngineDescriptors({
                engines: [
                    FORK_WIRE,
                    wireDescriptor('ollama-nightly', 'ollama', 'Ollama nightly', { pull: {} })
                ]
            })
        )

        expect(pullModelParams(state.engineFamily('ollama-nightly'), 'm')).toEqual({ name: 'm' })
        expect(pullModelParams(state.engineFamily('llamacpp-custom'), 'm')).toEqual({ model: 'm' })
        expect(pullModelParams(state.engineFamily('unknown-engine'), 'm')).toEqual({ model: 'm' })
    })
})

/** The object form of a wire descriptor fixture, for spreading variations. */
function wireDescriptorObject(value: JsonValue): Record<string, JsonValue> {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        throw new Error('fixture is not an object')
    }
    return value
}
