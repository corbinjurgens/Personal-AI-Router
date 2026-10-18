// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { removeProviderSecrets } from '@/electron/child-environment'

const SENTINEL = 'sentinel-secret-value-7f3a'

describe('removeProviderSecrets', () => {
    it('drops every reserved name and keeps the rest', () => {
        const env = {
            PATH: '/bin',
            NVPAIR_PROVIDER_SECRET_A: SENTINEL,
            HOME: '/h',
            NVPAIR_PROVIDER_SECRET_B: SENTINEL
        }

        expect(removeProviderSecrets(env)).toEqual({ PATH: '/bin', HOME: '/h' })
    })

    it('drops case variants of the namespace', () => {
        const env = {
            nvpair_provider_secret_a: SENTINEL,
            Nvpair_Provider_Secret_B: SENTINEL,
            KEEP: '1'
        }

        expect(removeProviderSecrets(env)).toEqual({ KEEP: '1' })
    })

    it('keeps names that only resemble the namespace', () => {
        const env = {
            NVPAIR_PROVIDER_SECRET: '1',
            XNVPAIR_PROVIDER_SECRET_A: '2',
            NVPAIR_PROVIDER_SECRETS: '3',
            NOTE: 'NVPAIR_PROVIDER_SECRET_A=x'
        }

        expect(removeProviderSecrets(env)).toEqual(env)
    })

    it('keeps an undefined value for an unrelated name', () => {
        expect(removeProviderSecrets({ UNSET: undefined })).toEqual({ UNSET: undefined })
    })

    it('does not modify its input', () => {
        const env = { NVPAIR_PROVIDER_SECRET_A: SENTINEL, KEEP: '1' }

        removeProviderSecrets(env)

        expect(env).toEqual({ NVPAIR_PROVIDER_SECRET_A: SENTINEL, KEEP: '1' })
    })
})
