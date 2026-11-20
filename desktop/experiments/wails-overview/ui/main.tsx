// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { FoundationsThemeProvider } from '@/ui/theme/FoundationsThemeProvider'
import '@/ui/styles/index.css'
import { replayOverviewInput } from './fixture'
import { liveOverviewInput } from './host-input'
import { connectOverview, type OverviewStatus } from './overview-session'
import { OverviewScene } from './scene'

const container = document.getElementById('root')
if (!container) throw new Error('Overview root is missing')

const root = createRoot(container)
const params = new URLSearchParams(window.location.search)
const order = params.get('order')
const mode = params.get('mode') ?? 'replay'
const host = window.__PAIR_OVERVIEW_HOST__
const input =
    mode === 'live' && host
        ? liveOverviewInput(host)
        : mode === 'replay'
          ? replayOverviewInput(params.get('scene'))
          : null

let generation = 0
function render(status: OverviewStatus) {
    if (status === 'connecting') generation++
    root.render(
        <StrictMode>
            <FoundationsThemeProvider>
                <OverviewScene
                    status={status}
                    nodesFirst={order === 'nodes-first'}
                    generation={generation}
                />
            </FoundationsThemeProvider>
        </StrictMode>
    )
}

let disconnect: (() => void) | undefined
try {
    render(input ? 'connecting' : 'unavailable')
    if (input) disconnect = connectOverview(input, render)
} catch {
    render('unavailable')
}
window.addEventListener(
    'pagehide',
    () => {
        root.unmount()
        disconnect?.()
    },
    { once: true }
)
