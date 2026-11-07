// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { expect, it, vi } from 'vitest'
import ContentBar from '@/ui/components/MainApp/ContentBar'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'

it('orders the cluster tabs and navigates to Models and Profiles', async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal(
        'ResizeObserver',
        class {
            observe = vi.fn<ResizeObserver['observe']>()
            unobserve = vi.fn<ResizeObserver['unobserve']>()
            disconnect = vi.fn<ResizeObserver['disconnect']>()
        }
    )
    useConnectionStore.setState({ selfId: null })
    useOverviewUiStore.setState({ activeTab: 'overview' })
    const container = document.createElement('div')
    document.body.append(container)
    const root = createRoot(container)
    try {
        await act(async () => root.render(<ContentBar />))
        const tabs = Array.from(container.querySelectorAll('[role="tab"]'))
        const labels = ['Overview', 'Models', 'Profiles', 'Settings']
        expect(tabs).toHaveLength(labels.length)
        labels.forEach((label, index) => expect(tabs[index].textContent).toContain(label))
        for (const index of [1, 2]) {
            const label = labels[index]
            const tab = tabs[index]
            if (!(tab instanceof HTMLElement)) throw new Error(`Missing ${label} tab`)
            await act(async () => tab.click())
            expect(useOverviewUiStore.getState().activeTab).toBe(label.toLowerCase())
        }
    } finally {
        await act(async () => root.unmount())
        container.remove()
        vi.unstubAllGlobals()
    }
})
