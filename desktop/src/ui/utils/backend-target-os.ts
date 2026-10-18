// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PlatformDisplayName } from '@/shared/types/platform'
import { displayNameToPlatform } from '@/shared/utils/platform'
import type { EngineType } from '@/shared/types/engines'
import { isBuiltinEngineType } from '@/shared/utils/engines'
import { EngineCapabilities } from '@/ui/constants/engine-capabilities'

/**
 * Whether PAIR can auto-install this backend on the given cluster OS. Only a
 * built-in engine has an install recipe PAIR knows.
 */
export function canAutoInstallBackendForOs(
    backendType: EngineType,
    targetOs: PlatformDisplayName
): boolean {
    if (!isBuiltinEngineType(backendType)) return false
    const targetPlatform = displayNameToPlatform(targetOs)
    return EngineCapabilities[backendType].hasInstall.includes(targetPlatform)
}
