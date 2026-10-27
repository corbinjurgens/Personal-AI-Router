// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { safeHandle } from './safe-handle'
import {
    chooseBuildApplyPreview,
    startBuildApply,
    getBuildApplyStatus
} from '@/electron/build-apply'

export function registerBuildApplyIpc(): void {
    safeHandle('build-apply:preview', () => chooseBuildApplyPreview())
    safeHandle('build-apply:start', (_event, request) => startBuildApply(request))
    safeHandle('build-apply:status', (_event, request) => getBuildApplyStatus(request))
}
