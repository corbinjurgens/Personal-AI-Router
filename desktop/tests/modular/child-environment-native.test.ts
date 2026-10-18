// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from 'node:child_process'
import { expect, it } from 'vitest'
import { removeProviderSecrets } from '@/electron/child-environment'

it('removes both provider-secret case variants at the native Node child boundary', () => {
    const env = removeProviderSecrets({
        ...process.env,
        NVPAIR_PROVIDER_SECRET_TEST: 'test-sentinel',
        nvpair_provider_secret_test: 'lowercase-sentinel',
        C2_UNRELATED: 'kept'
    })
    const stdout = execFileSync(
        process.execPath,
        [
            '-e',
            `
        const keys = Object.keys(process.env);
        console.log(keys.some(key => key.toUpperCase() === 'NVPAIR_PROVIDER_SECRET_TEST'));
        console.log(process.env.C2_UNRELATED === 'kept');
    `
        ],
        { env, encoding: 'utf8', windowsHide: true, timeout: 10000 }
    )

    expect(stdout.trim().split(/\r?\n/)).toEqual(['false', 'true'])
})
