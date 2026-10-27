// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { registerSettingsIpc } from '@/electron/ipc/settings.ipc'
import { registerWindowIpc } from '@/electron/ipc/window.ipc'
import { registerServiceIpc } from '@/electron/ipc/service.ipc'
import { registerAppDataIpc } from '@/electron/ipc/app-data.ipc'
import { registerUpdateIpc } from '@/electron/ipc/update.ipc'
import { registerBootstrapIpc } from '@/electron/ipc/bootstrap.ipc'
import { registerServiceBridgeIpc } from '@/electron/ipc/service-bridge.ipc'
import { registerInferenceDemoIpc } from '@/electron/ipc/inference-demo.ipc'
import { registerBuildApplyIpc } from '@/electron/ipc/build-apply.ipc'

export function registerAllIpc(): void {
    registerBuildApplyIpc()
    registerSettingsIpc()
    registerWindowIpc()
    registerServiceIpc()
    registerAppDataIpc()
    registerUpdateIpc()
    registerBootstrapIpc()
    registerServiceBridgeIpc()
    registerInferenceDemoIpc()
}
