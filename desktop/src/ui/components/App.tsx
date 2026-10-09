// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { lazy, Suspense, type ReactNode } from 'react'
import MainApp from '@/ui/components/MainApp/MainApp'
import { isTrayWindow } from '@/ui/utils/window-kind'

const TrayApp = lazy(() => import('@/ui/components/TrayApp/TrayApp'))

function App() {
    let content: ReactNode
    if (isTrayWindow) {
        content = (
            <Suspense fallback={null}>
                <TrayApp />
            </Suspense>
        )
    } else {
        content = <MainApp />
    }

    return content
}

export default App
