// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { parseAdvertisementEngineBindingOffers } from '../../src/electron/service-bridge/node-policy'
import type { JsonValue } from '../../src/electron/service-bridge/json-rpc-client'

describe('engine binding offers', () => {
    const models = [{ engine: 'fixture-engine', model: 'org/model' }]
    it('keeps preparation separate from actual inventory', () => {
        const advertisement = { engineBindingOffers: { version: 1, models } }
        expect(parseAdvertisementEngineBindingOffers(advertisement)).toEqual({
            state: 'valid',
            version: 1,
            models
        })
        expect(advertisement).toEqual({ engineBindingOffers: { version: 1, models } })
    })
    it('distinguishes absent and unsupported support', () => {
        expect(parseAdvertisementEngineBindingOffers({})).toEqual({ state: 'absent' })
        expect(
            parseAdvertisementEngineBindingOffers({ engineBindingOffers: { version: 2 } })
        ).toEqual({ state: 'unsupported', version: 2 })
    })
    const invalidOffers: JsonValue[] = [
        null,
        { version: 1, models: [] },
        { version: 1, models, ready: true },
        { version: 1, models: [...models, ...models] },
        { version: 1, models: [{ engine: 'fixture-engine', model: 'weak' }] },
        { version: 1, models: [{ engine: 'fixture-engine', model: '<'.repeat(100) }] }
    ]
    it.each(invalidOffers)('disables malformed optional offers (%j)', engineBindingOffers => {
        expect(parseAdvertisementEngineBindingOffers({ engineBindingOffers })).toEqual({
            state: 'invalid'
        })
    })
    it('refuses case variant extension names', () => {
        expect(
            parseAdvertisementEngineBindingOffers({ EngineBindingOffers: { version: 1, models } })
        ).toEqual({ state: 'invalid' })
    })
})
