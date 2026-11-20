// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ServiceInvokeArgs, ServiceTransport } from '@/shared/types/service-bridge'
import type { WsInvokeChannel } from '@/shared/types/ws-channels'
import { useMetricsStore } from '@/ui/stores/metrics.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useWorkloadsStore } from '@/ui/stores/workloads.store'
import { FoundationsThemeProvider } from '@/ui/theme/FoundationsThemeProvider'
import { replayOverviewInput } from '../../../experiments/wails-overview/ui/fixture'
import {
    connectOverview,
    type OverviewStatus
} from '../../../experiments/wails-overview/ui/overview-session'
import { OverviewScene } from '../../../experiments/wails-overview/ui/scene'

const historyCard = 'replay-self/history-1'
const sharedNodeCards = ['replay-peer/job-01', 'replay-self/job-01']
const twelveCards = [
    ...sharedNodeCards,
    'replay-self/job-03',
    'replay-self/job-04',
    'replay-self/job-05',
    'replay-self/job-06',
    'replay-self/job-07',
    'replay-self/job-08',
    'replay-self/job-09',
    'replay-self/job-10',
    'replay-self/job-11',
    'replay-self/job-12'
]

describe('Overview replay', () => {
    let container: HTMLDivElement
    let root: Root
    let disconnect: (() => void) | undefined

    beforeEach(() => {
        vi.useFakeTimers()
        vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
        vi.stubGlobal(
            'ResizeObserver',
            class {
                observe = vi.fn<ResizeObserver['observe']>()
                unobserve = vi.fn<ResizeObserver['unobserve']>()
                disconnect = vi.fn<ResizeObserver['disconnect']>()
            }
        )
        vi.stubGlobal(
            'matchMedia',
            (media: string): MediaQueryList => ({
                media,
                matches: false,
                onchange: null,
                addListener: vi.fn(),
                removeListener: vi.fn(),
                addEventListener: vi.fn(),
                removeEventListener: vi.fn(),
                dispatchEvent: vi.fn(() => true)
            })
        )
        container = document.createElement('div')
        document.body.append(container)
        root = createRoot(container)
    })
    afterEach(async () => {
        await act(async () => root.unmount())
        container.remove()
        disconnect?.()
        disconnect = undefined
        await vi.advanceTimersByTimeAsync(0)
        vi.useRealTimers()
        vi.unstubAllGlobals()
    })

    it.each([
        { scene: 'idle', nodesFirst: false, active: [], count: 0, gpu: 4 },
        { scene: 'shared-node', nodesFirst: true, active: sharedNodeCards, count: 2, gpu: 73 },
        { scene: 'running-12', nodesFirst: false, active: twelveCards, count: 12, gpu: 73 },
        {
            scene: 'running-13',
            nodesFirst: true,
            active: [...twelveCards, 'replay-self/job-13'],
            count: 13,
            gpu: 73
        }
    ])('renders the $scene roster, card identities and telemetry', async data => {
        const input = replayOverviewInput(data.scene)
        if (!input) throw new Error('Fixture scene is missing')
        const onStatus = vi.fn<(status: OverviewStatus) => void>()
        disconnect = connectOverview(input, onStatus)
        await vi.advanceTimersByTimeAsync(0)
        expect(onStatus.mock.calls).toEqual([['connecting'], ['ready']])

        vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (
            this: Element
        ) {
            if (this.hasAttribute('data-workload-id')) {
                return new DOMRect(data.nodesFirst ? 560 : 0, 100, 240, 80)
            }
            if (this.hasAttribute('data-node-id')) {
                return new DOMRect(data.nodesFirst ? 0 : 320, 70, 460, 80)
            }
            if (this.hasAttribute('data-node-gpu-chart-anchor')) {
                return new DOMRect(data.nodesFirst ? 20 : 340, 85, 60, 60)
            }
            return new DOMRect(data.nodesFirst ? 460 : 240, 0, 80, 700)
        })
        await act(async () =>
            root.render(
                <FoundationsThemeProvider>
                    <OverviewScene status="ready" nodesFirst={data.nodesFirst} generation={1} />
                </FoundationsThemeProvider>
            )
        )
        expect(useNodesStore.getState().nodes.size).toBe(0)
        expect(container.querySelectorAll('.node-card')).toHaveLength(2)
        expect(container.querySelector('[data-node-id="replay-self"]')?.textContent).toContain(
            'Replay workstation'
        )
        expect(container.textContent).toContain('Replay model host')
        expect(container.textContent).toContain('(Offline)')
        expect(container.querySelector('[data-node-id="replay-peer"]')).toBeNull()
        expect(container.querySelectorAll('[data-workload-id]')).toHaveLength(0)

        await act(async () => vi.advanceTimersByTimeAsync(1100))
        await act(async () => vi.advanceTimersByTimeAsync(100))
        expect(
            Array.from(container.querySelectorAll('.node-card'), card =>
                card.getAttribute('data-node-id')
            )
        ).toEqual(['replay-self', 'replay-peer'])
        expect(useNodesStore.getState().nodes.has('replay-pending')).toBe(true)
        expect(container.textContent).not.toContain('Replay pending peer')
        expect(container.textContent).not.toContain('(Offline)')
        const peer = container.querySelector('[data-node-id="replay-peer"]')
        expect(peer?.textContent).toContain('Replay model host')
        expect(peer?.textContent).toContain('Replay GPU')
        expect(peer?.textContent).toContain('llama.cpp: Running')
        expect(peer?.querySelectorAll('[data-node-gpu-chart-anchor]')).toHaveLength(1)
        expect(
            useMetricsStore.getState().nodeMetrics.get('replay-peer')?.gpuUtilization[0]?.data[0]
                ?.value
        ).toBe(data.gpu)

        const cards = container.querySelectorAll('[data-workload-id]')
        expect(cards).toHaveLength(data.count + 1)
        expect(
            Array.from(cards, card =>
                [
                    card.getAttribute('data-workload-origin'),
                    card.getAttribute('data-workload-id')
                ].join('/')
            ).sort()
        ).toEqual([...data.active, historyCard].sort())
        expect(
            Array.from(useWorkloadsStore.getState().workloads.values()).filter(
                job => job.state === 'running' && job.scheduledOn === 'replay-peer'
            )
        ).toHaveLength(data.count)
        const jobsColumn = container.querySelector('.workloads-list-container')
        expect(jobsColumn?.parentElement?.style.flexDirection).toBe(
            data.nodesFirst ? 'row-reverse' : ''
        )
        const paths = container.querySelectorAll('g[clip-path="url(#connClip)"] path')
        expect(paths).toHaveLength(data.count)
        for (const path of paths) {
            expect(path.getAttribute('d')).toBe(
                data.nodesFirst
                    ? 'M 0,115 C 50,115 50,110 100,110'
                    : 'M 0,110 C 40,110 40,115 80,115'
            )
            expect(path.getAttribute('filter')).toBe(data.count === 13 ? null : 'url(#lineGlow)')
            const beads = path.parentElement?.querySelectorAll('animateMotion')
            expect(beads).toHaveLength(data.count === 13 ? 1 : 3)
            for (const bead of beads ?? []) {
                expect(bead.getAttribute('keyPoints')).toBe(data.nodesFirst ? '1;0' : '0;1')
            }
        }
    })

    it('clears old jobs and withholds readiness after a required read fails', async () => {
        const replay = replayOverviewInput('shared-node')
        if (!replay) throw new Error('Fixture scene is missing')
        let notify: ((transport: ServiceTransport | null) => void) | undefined
        let original: ServiceTransport | undefined
        const onStatus = vi.fn<(status: OverviewStatus) => void>()
        disconnect = connectOverview(
            {
                platform: replay.platform,
                subscribe(listener) {
                    notify = listener
                    return replay.subscribe(transport => {
                        original = transport ?? undefined
                        listener(transport)
                    })
                }
            },
            onStatus
        )
        await vi.advanceTimersByTimeAsync(1100)
        expect(useWorkloadsStore.getState().workloads.size).toBe(3)

        replay.subscribe(next => {
            if (!next || !notify) throw new Error('Fixture transport is missing')
            notify({
                ...next,
                async invoke<C extends WsInvokeChannel>(...args: ServiceInvokeArgs<C>) {
                    if (args[0] === 'workloads:get-initial') throw new Error('Fixture read failed')
                    return next.invoke<C>(...args)
                }
            })
        })
        await vi.advanceTimersByTimeAsync(1100)
        expect(onStatus.mock.calls).toEqual([
            ['connecting'],
            ['ready'],
            ['connecting'],
            ['unavailable']
        ])
        expect(original?.connected).toBe(false)
        expect(useWorkloadsStore.getState().workloads.size).toBe(0)
        expect(useNodesStore.getState().nodes.size).toBe(0)
    })

    it('retires an in-flight baseline and refuses mutation before the replacement host sees it', async () => {
        const replay = replayOverviewInput('shared-node')
        if (!replay) throw new Error('Fixture scene is missing')
        const readStarted = Promise.withResolvers<void>()
        const releaseRead = Promise.withResolvers<void>()
        let notify: ((transport: ServiceTransport | null) => void) | undefined
        const onStatus = vi.fn<(status: OverviewStatus) => void>()
        disconnect = connectOverview(
            {
                platform: replay.platform,
                subscribe(listener) {
                    notify = listener
                    return replay.subscribe(source => {
                        if (!source) throw new Error('Fixture transport is missing')
                        listener({
                            ...source,
                            async invoke<C extends WsInvokeChannel>(...args: ServiceInvokeArgs<C>) {
                                const response = await source.invoke<C>(...args)
                                if (args[0] === 'nodes:get-initial') {
                                    readStarted.resolve()
                                    await releaseRead.promise
                                }
                                return response
                            }
                        })
                    })
                }
            },
            onStatus
        )
        await readStarted.promise
        expect(onStatus.mock.calls).toEqual([['connecting']])

        const replacementInvokes: WsInvokeChannel[] = []
        replay.subscribe(source => {
            if (!notify || !source) throw new Error('Fixture transport is missing')
            notify({
                ...source,
                invoke<C extends WsInvokeChannel>(...args: ServiceInvokeArgs<C>) {
                    replacementInvokes.push(args[0])
                    return source.invoke<C>(...args)
                }
            })
        })
        try {
            await vi.advanceTimersByTimeAsync(1100)
            expect(onStatus.mock.calls).toEqual([['connecting'], ['connecting'], ['ready']])
            await expect(
                window.pairApi.workloads.cancel({
                    originatedFrom: 'replay-self',
                    workloadId: 'job-01',
                    engine: 'llama-cpp',
                    runId: 'replay-run',
                    regenerate: false
                })
            ).rejects.toThrow('Overview is read-only: workloads:cancel')
            expect(replacementInvokes).toContain('workloads:get-initial')
            expect(replacementInvokes).not.toContain('workloads:cancel')
        } finally {
            releaseRead.resolve()
        }
        await vi.advanceTimersByTimeAsync(0)
        expect(onStatus.mock.calls).toEqual([['connecting'], ['connecting'], ['ready']])
        expect(useNodesStore.getState().nodes.has('replay-peer')).toBe(true)
        expect(useWorkloadsStore.getState().workloads.size).toBe(3)
    })
})
