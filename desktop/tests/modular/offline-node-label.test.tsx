// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import OfflineNode from '@/ui/components/NodeList/OfflineNode'
import { useConnectionStore } from '@/ui/stores/connection.store'

let container: HTMLDivElement
let root: Root

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    useConnectionStore.setState({ selfId: 'self-uuid' })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.unstubAllGlobals()
})

describe('offline node card', () => {
    it('shows the distinguishing label beside the name', async () => {
        await act(async () =>
            root.render(
                <OfflineNode nodeId="peer-uuid" name="Studio" distinguishingLabel="RTX 4090" />
            )
        )
        expect(container.textContent).toContain('Studio')
        expect(container.textContent).toContain('(RTX 4090)')
        expect(container.textContent).toContain('(Offline)')
    })

    it('shows no label when none is given', async () => {
        await act(async () => root.render(<OfflineNode nodeId="peer-uuid" name="Studio" />))
        expect(container.textContent).not.toContain('(RTX')
    })
})
