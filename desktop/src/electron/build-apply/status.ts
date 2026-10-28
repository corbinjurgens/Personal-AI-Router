// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFile } from 'node:child_process'
import path from 'node:path'
import { BUILD_APPLY_MAX_STATUS_BYTES } from '@/shared/types/build-apply'
import type { BuildApplyResult, BuildApplyStatus } from '@/shared/types/build-apply'
import type { JsonValue } from '@/shared/types/json'
import { parseBuildApplyStatusResult } from '@/electron/service-bridge/build-apply'

export async function readBuildApplyStatus(
    binaryPath: string
): Promise<BuildApplyResult<BuildApplyStatus>> {
    return new Promise(resolve => {
        execFile(
            binaryPath,
            ['apply-build', 'status', '--json'],
            {
                cwd: path.dirname(binaryPath),
                timeout: 10_000,
                maxBuffer: BUILD_APPLY_MAX_STATUS_BYTES,
                windowsHide: true
            },
            (error, stdout) => {
                try {
                    if (error || Buffer.byteLength(stdout, 'utf8') > BUILD_APPLY_MAX_STATUS_BYTES)
                        throw new Error('Status unavailable')
                    const value: JsonValue = JSON.parse(stdout)
                    resolve(parseBuildApplyStatusResult(value))
                } catch {
                    resolve({
                        available: false,
                        value: null,
                        refusal: {
                            code: 'unavailable',
                            detail: 'Build apply status is unavailable. Read status before starting or stopping the service.'
                        }
                    })
                }
            }
        )
    })
}

export async function requireBuildApplyStartup(binaryPath: string): Promise<void> {
    const status = await readBuildApplyStatus(binaryPath)
    if (!status.available || !status.value.startupAllowed) {
        throw new Error(
            'Build apply status prevents service lifecycle changes. Read status and recover the operation.'
        )
    }
}
