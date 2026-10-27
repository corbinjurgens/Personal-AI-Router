// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { app } from 'electron'
import { initPlatform } from '@/electron/globals'
import { ElectronPathProvider, setPaths } from '@/electron/path'
import { initFileLogger } from '@/shared/utils/log'
import { APP_DISPLAY_NAME } from '@/shared/constants/app'
import { handleBuildApplyEntry } from '@/electron/build-apply'
import { handleShortcutEntry } from '@/electron/windows-shortcut'

const init = async (): Promise<void> => {
    try {
        // These finite command modes must exit before ordinary main startup,
        // including single-instance focus, connector auto-start and updater work.
        const shortcutExit = await handleShortcutEntry(process.argv)
        if (shortcutExit !== null) {
            app.exit(shortcutExit)
            return
        }
        const applyExit = await handleBuildApplyEntry(process.argv)
        if (applyExit !== null) {
            app.exit(applyExit)
            return
        }
        app.setName(APP_DISPLAY_NAME)
        await setPaths()

        const paths = new ElectronPathProvider(app)
        initPlatform(paths)
        initFileLogger(paths)

        await import('./main')
    } catch (error) {
        console.error('Error initializing app:', error)
        process.exit(1)
    }
}

init()
