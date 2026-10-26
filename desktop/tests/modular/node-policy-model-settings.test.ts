// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, createElement, useState } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { JsonValue } from '@/shared/types/json'
import {
    applyModelSettings,
    createModelSettingsProfile,
    parseModelSettings,
    removeModelSettingsProfile
} from '@/shared/utils/node-policy-model-settings'
import ModelSettingsEditor from '@/ui/components/RoutingSettings/ModelSettingsEditor'

const identity = { engine: 'llamacpp', model: 'native/model' }
const other = { engine: 'lmstudio', model: identity.model }
function policy(extra: Record<string, JsonValue> = {}): string {
    return JSON.stringify({
        version: 2,
        revision: '9007199254740993',
        engineBindings: [identity],
        routing: { accounts: [{ id: 'private' }] },
        profiles: [
            {
                ...identity,
                name: 'display name',
                qualification: { id: 'q', revision: '1' },
                ...extra
            },
            other
        ]
    })
}
function applied(text: string, edit: Parameters<typeof applyModelSettings>[2]): string {
    const result = applyModelSettings(text, identity, edit)
    if (!result.ok) throw new Error(result.error.message)
    return result.value
}
function selected(text: string) {
    const result = parseModelSettings(text)
    if (!result.ok) throw new Error(result.error.message)
    const found = result.value.find(
        profile => profile.engine === identity.engine && profile.model === identity.model
    )
    if (!found) throw new Error('Missing selected profile')
    return found
}

describe('model settings policy helpers', () => {
    it('changes only the selected value span across whitespace and escaped delimiters', () => {
        const text = String.raw` {
            "profiles": [{"engine":"llamacpp","model":"native/model","requestOptions":{"note":"escaped quote \" and brackets } ],","nested":[{"x":[1,2]}],"temperature":1}}],
            "engineBindings": []
        } `
        expect(applied(text, { kind: 'option', key: 'temperature', value: '0.25' })).toBe(
            text.replace('"temperature":1', '"temperature":0.25')
        )
    })
    it.each([
        { kind: 'option', key: 'temperature', value: '0.5' },
        { kind: 'mode', value: 'enforced' },
        { kind: 'localOnly', value: true }
    ] satisfies Parameters<typeof applyModelSettings>[2][])(
        'preserves opaque number tokens when changing $kind',
        edit => {
            const text =
                '{"version":2,"opaque":9007199254740993,"profiles":[{"engine":"llamacpp","model":"native/model","requestOptions":{"temperature":1,"nested":{"precise":1.0000000000000000001,"huge":1e400},"array":[9007199254740995]}},{"engine":"lmstudio","model":"other","requestOptions":{"seed":9007199254740997}}],"engineBindings":[]}'
            const result = applied(text, edit)
            // Inspect original JSON tokens, not numbers rounded by JSON.parse.
            for (const token of [
                '"opaque":9007199254740993',
                '"precise":1.0000000000000000001',
                '"huge":1e400',
                '[9007199254740995]',
                '"seed":9007199254740997'
            ])
                expect(result).toContain(token)
        }
    )
    it.each(['{}', '{"profiles":[]}', '{"profiles":null}'])(
        'accepts an empty policy profile set %s',
        text => {
            expect(parseModelSettings(text)).toEqual({ ok: true, value: [] })
        }
    )
    it.each(['{', 'null', '[]', '{"profiles":{}}', '{"profiles":[null]}'])(
        'keeps malformed policy %s as a structured failure',
        text => {
            expect(parseModelSettings(text)).toMatchObject({
                ok: false,
                error: { code: expect.any(String) }
            })
            expect(
                applyModelSettings(text, identity, {
                    kind: 'option',
                    key: 'temperature',
                    value: '1'
                }).ok
            ).toBe(false)
        }
    )

    const malformedSettings: Record<string, JsonValue>[] = [
        { requestOptionsMode: 'force' },
        { requestOptionsMode: 1 },
        { requestOptionsMode: null },
        { localOnly: 'false' },
        { localOnly: 0 },
        { localOnly: null },
        { requestOptions: [] },
        { requestOptionsMode: 'enforced' },
        { requestOptionsMode: 'enforced', requestOptions: {} }
    ]
    it.each(malformedSettings)('rejects malformed profile settings %j', extra => {
        expect(parseModelSettings(policy(extra)).ok).toBe(false)
    })

    it('identifies profiles by engine and model regardless of display name or order', () => {
        const text = JSON.stringify({
            profiles: [
                { ...other, name: 'same' },
                { ...identity, name: 'same' }
            ]
        })
        const result = parseModelSettings(text)
        if (!result.ok) throw new Error(result.error.message)
        expect(result.value.map(profile => profile.key)).toEqual([
            JSON.stringify([other.engine, other.model]),
            JSON.stringify([identity.engine, identity.model])
        ])
        const edited = applied(text, { kind: 'option', key: 'temperature', value: '0.3' })
        expect(JSON.parse(edited)).toEqual({
            profiles: [
                { ...other, name: 'same' },
                { ...identity, name: 'same', requestOptions: { temperature: 0.3 } }
            ]
        })
    })

    it('rejects ambiguous duplicate identities', () => {
        expect(
            parseModelSettings(
                JSON.stringify({ profiles: [identity, { ...identity, name: 'other' }] })
            ).ok
        ).toBe(false)
    })

    it.each([
        ['temperature', '0'],
        ['temperature', '20'],
        ['top_p', '0'],
        ['top_p', '1'],
        ['top_k', '0'],
        ['top_k', '9007199254740991'],
        ['repeat_penalty', '0.01'],
        ['frequency_penalty', '-2'],
        ['frequency_penalty', '2'],
        ['presence_penalty', '-2'],
        ['presence_penalty', '2'],
        ['seed', '-9007199254740991'],
        ['seed', '9007199254740991'],
        ['min_p', '0'],
        ['min_p', '1']
    ])('accepts %s boundary %s', (key, value) => {
        expect(selected(applied(policy(), { kind: 'option', key, value })).options[key]).toBe(
            Number(value)
        )
    })

    it.each([
        ['temperature', '-0.1'],
        ['temperature', 'Infinity'],
        ['temperature', 'NaN'],
        ['temperature', '1e309'],
        ['temperature', '0x10'],
        ['temperature', 'partial'],
        ['top_p', '-0.1'],
        ['top_p', '1.01'],
        ['top_k', '-1'],
        ['top_k', '0.5'],
        ['top_k', '9007199254740992'],
        ['repeat_penalty', '0'],
        ['repeat_penalty', '-1'],
        ['frequency_penalty', '-2.01'],
        ['frequency_penalty', '2.01'],
        ['presence_penalty', '-2.01'],
        ['presence_penalty', '2.01'],
        ['seed', '1.1'],
        ['seed', '9007199254740992'],
        ['seed', '-9007199254740992'],
        ['min_p', '-0.1'],
        ['min_p', '1.01']
    ])('rejects %s value %s without a replacement draft', (key, value) => {
        expect(
            applyModelSettings(policy(), identity, { kind: 'option', key, value })
        ).toMatchObject({ ok: false, error: { code: 'option' } })
    })

    it('clears one option while preserving nested options, arrays, bindings and private fields', () => {
        const options = { temperature: 1, stop: ['end'], uncommon: { leaf: 2 } }
        const text = applied(
            policy({ requestOptions: options, requestOptionsMode: 'enforced', localOnly: true }),
            { kind: 'option', key: 'temperature', value: '' }
        )
        expect(JSON.parse(text)).toEqual(
            JSON.parse(
                policy({
                    requestOptions: { stop: ['end'], uncommon: { leaf: 2 } },
                    requestOptionsMode: 'enforced',
                    localOnly: true
                })
            )
        )
    })

    it('removes the last option and enforced mode while retaining localOnly', () => {
        const text = applied(
            policy({
                requestOptions: { temperature: 1 },
                requestOptionsMode: 'enforced',
                localOnly: true
            }),
            { kind: 'option', key: 'temperature', value: ' ' }
        )
        expect(JSON.parse(text)).toEqual(JSON.parse(policy({ localOnly: true })))
    })

    it('normalizes defaults to omission and requires nonempty enforced options', () => {
        expect(selected(policy()).mode).toBe('defaults')
        expect(applyModelSettings(policy(), identity, { kind: 'mode', value: 'enforced' }).ok).toBe(
            false
        )
        expect(applyModelSettings(policy(), identity, { kind: 'mode', value: 'invalid' }).ok).toBe(
            false
        )
        const forced = applied(policy({ requestOptions: { stop: ['end'] } }), {
            kind: 'mode',
            value: 'enforced'
        })
        expect(selected(forced).mode).toBe('enforced')
        expect(JSON.parse(applied(forced, { kind: 'mode', value: 'defaults' }))).toEqual(
            JSON.parse(policy({ requestOptions: { stop: ['end'] } }))
        )
        expect(
            JSON.parse(
                applied(policy({ requestOptionsMode: 'defaults' }), {
                    kind: 'option',
                    key: 'seed',
                    value: '1'
                })
            )
        ).toEqual(JSON.parse(policy({ requestOptions: { seed: 1 } })))
    })

    it('locks and unlocks locally without changing request options or metadata', () => {
        const initial = policy({ requestOptions: { temperature: 1 } })
        const locked = applied(initial, { kind: 'localOnly', value: true })
        expect(selected(locked).localOnly).toBe(true)
        expect(JSON.parse(applied(locked, { kind: 'localOnly', value: false }))).toEqual(
            JSON.parse(initial)
        )
    })

    it('preserves unsupported engines and LM Studio min_p through supported edits', () => {
        const ollama = {
            engine: 'ollama',
            model: 'native',
            requestOptions: { options: { temperature: 2 } }
        }
        const lm = { ...other, requestOptions: { min_p: 0.2, uncommon: [1, 2] } }
        const text = JSON.stringify({ profiles: [ollama, lm] })
        const result = applyModelSettings(text, other, {
            kind: 'option',
            key: 'temperature',
            value: '0.7'
        })
        if (!result.ok) throw new Error(result.error.message)
        expect(JSON.parse(result.value)).toEqual({
            profiles: [
                ollama,
                { ...lm, requestOptions: { ...lm.requestOptions, temperature: 0.7 } }
            ]
        })
        expect(
            applyModelSettings(text, other, { kind: 'option', key: 'min_p', value: '0.4' }).ok
        ).toBe(false)
        expect(
            applyModelSettings(text, ollama, { kind: 'option', key: 'temperature', value: '0.4' })
                .ok
        ).toBe(false)
    })
})

describe('typed profile lifecycle', () => {
    const first = { name: 'first', ...identity }
    const second = { name: 'second', ...other }
    function success(result: ReturnType<typeof createModelSettingsProfile>): string {
        if (!result.ok) throw new Error(result.error.message)
        return result.value
    }

    it.each(['{}', '{"profiles":null}', '{"profiles":[]}', ' \n{ "profiles": [ ] }\n'])(
        'creates the first minimal profile in %s',
        text => {
            const result = success(createModelSettingsProfile(text, identity, 'first'))
            expect(JSON.parse(result)).toEqual({ profiles: [first] })
            expect(parseModelSettings(result)).toMatchObject({
                ok: true,
                value: [{ ...identity, name: 'first' }]
            })
        }
    )

    it('appends without rewriting existing profiles, numeric tokens or unrelated references', () => {
        const text = String.raw` {"opaque":9007199254740993,"profiles":[{"name":"first","engine":"llamacpp","model":"native/model","requestOptions":{"precise":1.00000000000000001,"huge":1e400,"nested":[null,{},[]]}}],"residentModels":null,"servedModels":[],"roles":{"nine":{"engine":"llamacpp","model":"native/model"}},"engineBindings":null} `
        const escaped = { engine: 'lmstudio', model: 'owner/"model\\file/模型' }
        const raw = JSON.stringify({ name: 'new/model', ...escaped })
        const edited = success(createModelSettingsProfile(text, escaped, 'new/model'))
        expect(edited).toBe(text.replace('}}],"residentModels"', `}},${raw}],"residentModels"`))
        expect(JSON.parse(edited).profiles[1]).toEqual({ name: 'new/model', ...escaped })
    })

    it.each([
        { label: 'first', profiles: [first, second], selected: identity, remaining: [second] },
        { label: 'last', profiles: [second, first], selected: identity, remaining: [second] },
        {
            label: 'middle',
            profiles: [second, first, { name: 'third', engine: 'ollama', model: 'm' }],
            selected: identity,
            remaining: [second, { name: 'third', engine: 'ollama', model: 'm' }]
        },
        { label: 'only', profiles: [first], selected: identity, remaining: [] }
    ])('removes only the $label profile', ({ profiles, selected, remaining }) => {
        const text = JSON.stringify({
            profiles,
            residentModels: [],
            servedModels: [identity],
            engineBindings: [identity],
            roles: { nine: identity },
            files: ['unchanged']
        })
        const edited = success(removeModelSettingsProfile(text, selected))
        expect(JSON.parse(edited)).toEqual({
            profiles: remaining,
            residentModels: [],
            servedModels: [identity],
            engineBindings: [identity],
            roles: { nine: identity },
            files: ['unchanged']
        })
    })

    it('removes one raw span while retaining opaque numbers and null/empty values', () => {
        const rawFirst = JSON.stringify(first)
        const rawSecond =
            '{"name":"second","engine":"lmstudio","model":"native/model","requestOptions":{"nested":{"n":9007199254740993,"x":1e400}},"opaque":null}'
        const text = ` {"profiles":[ ${rawFirst}, \n${rawSecond} ],"servedModels":null,"residentModels":[],"opaque":[] } `
        expect(success(removeModelSettingsProfile(text, identity))).toBe(
            text.replace(`${rawFirst}, \n`, '')
        )
    })

    it('requires removing the exact resident reference first', () => {
        const text = JSON.stringify({ profiles: [first, second], residentModels: [identity] })
        expect(removeModelSettingsProfile(text, identity)).toEqual({
            ok: false,
            error: {
                code: 'profile',
                message: 'Remove this model from residentModels before removing its profile.'
            }
        })
        expect(JSON.parse(success(removeModelSettingsProfile(text, other))).profiles).toEqual([
            first
        ])
    })

    it.each(['', 'has space', '-leading', 'a'.repeat(129), 'line\nend', 'tail\n'])(
        'rejects invalid new name %j',
        name => {
            expect(createModelSettingsProfile('{}', identity, name)).toMatchObject({
                ok: false,
                error: { code: 'profile' }
            })
        }
    )

    it.each([
        { engine: '', model: 'm' },
        { engine: 'llamacpp', model: '' },
        { engine: 'llamacpp', model: 'line\nbreak' },
        { engine: 'llamacpp', model: '\ud800' },
        { engine: 'llamacpp', model: 'é'.repeat(257) },
        { engine: 'llamacpp', model: 'weak-local' },
        { engine: 'bad/engine', model: 'm' },
        { engine: 'x'.repeat(41), model: 'm' }
    ])('rejects incomplete exact identity %j', ref => {
        expect(createModelSettingsProfile('{}', ref, 'first').ok).toBe(false)
    })

    it('rejects duplicate names or exact identities and missing removal targets', () => {
        const text = JSON.stringify({ profiles: [first] })
        expect(createModelSettingsProfile(text, other, first.name).ok).toBe(false)
        expect(createModelSettingsProfile(text, identity, 'different').ok).toBe(false)
        expect(removeModelSettingsProfile(text, other).ok).toBe(false)
    })

    it.each([
        '{',
        'null',
        '{"profiles":{}}',
        '{"profiles":[null]}',
        '{"profiles":[],"profi\\u006ces":[]}',
        '{"profiles":[{"name":"a","engine":"ollama","engine":"llamacpp","model":"m"}]}',
        '{"profiles":[{"name":"a","name":"b","engine":"llamacpp","model":"m"}]}',
        '{"profiles":[{"name":"a","engine":"llamacpp","model":"m","requestOptions":{"temperature":1,"temperature":2}}]}',
        JSON.stringify({ profiles: [identity] }),
        JSON.stringify({ profiles: [first, { ...second, name: first.name }] }),
        JSON.stringify({ profiles: [first, { ...first, name: 'other' }] }),
        JSON.stringify({ profiles: [{ ...first, name: 'invalid name' }] }),
        JSON.stringify({ profiles: [{ ...first, requestOptionsMode: 'enforced' }] })
    ])('returns only an error for malformed or ambiguous draft %s', text => {
        expect(
            createModelSettingsProfile(text, { engine: 'llamacpp', model: 'new' }, 'new')
        ).toMatchObject({ ok: false, error: { code: expect.any(String) } })
        expect(removeModelSettingsProfile(text, identity)).toMatchObject({
            ok: false,
            error: { code: expect.any(String) }
        })
    })

    it.each([{}, [null], [{ engine: 'llamacpp' }]])(
        'refuses malformed residency %j',
        residentModels => {
            expect(
                removeModelSettingsProfile(
                    JSON.stringify({ profiles: [first], residentModels }),
                    identity
                ).ok
            ).toBe(false)
        }
    )

    it.each([
        '{"Profiles":[{"name":"a","engine":"llamacpp","model":"m"}]}',
        '{"profiles":[],"Profi\\u006ces":[]}',
        '{"profi\\u006ceS":[]}',
        '{"profiles":[],"ResidentModels":[]}',
        '{"profiles":[{"name":"a","engine":"llamacpp","model":"m","Model":"shadow"}]}',
        '{"profiles":[{"name":"a","engine":"llamacpp","model":"m","Mo\\u0064el":"shadow"}]}',
        '{"profiles":[{"Name":"a","engine":"llamacpp","model":"m"}]}',
        '{"profiles":[{"name":"a","engine":"llamacpp","model":"m","RequestOptions":{}}]}',
        '{"profiles":[{"name":"a","engine":"llamacpp","model":"m","localOnly":true,"LocalOnly":false}]}',
        '{"profiles":[{"name":"a","engine":"llamacpp","model":"m","requestOptions":{"stop":1,"ſtop":2}}]}'
    ])('refuses service-equivalent key ambiguity without returning a draft: %s', text => {
        for (const result of [
            createModelSettingsProfile(text, { engine: 'llamacpp', model: 'new' }, 'new'),
            removeModelSettingsProfile(text, { engine: 'llamacpp', model: 'm' })
        ]) {
            expect(result).toMatchObject({ ok: false, error: { code: 'parse' } })
            expect(result).not.toHaveProperty('value')
        }
    })
})

describe('model settings form controls', () => {
    let container: HTMLDivElement
    let root: Root
    const validity = vi.fn<(valid: boolean) => void>()
    const changed = vi.fn<(text: string) => void>()
    beforeEach(() => {
        vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
        validity.mockReset()
        changed.mockReset()
        container = document.createElement('div')
        document.body.append(container)
        root = createRoot(container)
    })
    afterEach(async () => {
        await act(async () => root.unmount())
        container.remove()
        vi.unstubAllGlobals()
    })
    async function render(
        text: string,
        savedText = text,
        remote = false,
        disabled = false,
        target = identity
    ) {
        await act(async () =>
            root.render(
                createElement(ModelSettingsEditor, {
                    identity: target,
                    text,
                    savedText,
                    remote,
                    disabled,
                    nodeName: remote ? 'Peer' : 'This PC',
                    onTextChange: changed,
                    onValidityChange: validity
                })
            )
        )
    }
    function input(label: string): HTMLInputElement {
        const found = container.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)
        if (!found) throw new Error(`Missing ${label} input`)
        return found
    }
    async function type(label: string, value: string) {
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
        if (!setter) throw new Error('Missing input value setter')
        await act(async () => {
            setter.call(input(label), value)
            input(label).dispatchEvent(new Event('input', { bubbles: true }))
        })
    }

    function selectTrigger(label: string): HTMLButtonElement {
        const field = Array.from(container.querySelectorAll('label')).find(
            item => item.textContent?.trim() === label
        )
        const trigger = field && document.getElementById(field.htmlFor)
        if (!(trigger instanceof HTMLButtonElement)) throw new Error(`Missing ${label} selector`)
        return trigger
    }

    async function choose(label: string, value: string) {
        const trigger = selectTrigger(label)
        await act(async () => trigger.click())
        const option = Array.from(
            document.querySelectorAll<HTMLButtonElement>('[role="option"]')
        ).find(item => item.textContent?.trim() === value)
        if (!option) throw new Error(`Missing ${value} option`)
        await act(async () => option.click())
    }

    it('edits the bound engine/model profile and hides min_p for LM Studio', async () => {
        await render(policy(), policy(), false, false, other)
        expect(container.querySelector('input[aria-label="Min P"]')).toBeNull()
        await type('Temperature', '0.4')
        const text = changed.mock.lastCall?.[0]
        if (!text) throw new Error('Missing selected-profile edit')
        const parsed = parseModelSettings(text)
        if (!parsed.ok) throw new Error(parsed.error.message)
        expect(parsed.value.map(profile => profile.options)).toEqual([{}, { temperature: 0.4 }])
    })

    it('saves the selected override mode through the shared JSON draft', async () => {
        const text = policy({ requestOptions: { temperature: 1 } })
        await render(text)
        await choose('Request overrides', 'Force PAIR values')
        expect(changed).toHaveBeenCalledExactlyOnceWith(
            applied(text, { kind: 'mode', value: 'enforced' })
        )
    })

    it('cancels a rejected enforced selection and restores draft validity', async () => {
        const text = policy()
        await render(text)
        await choose('Request overrides', 'Force PAIR values')
        expect(changed).not.toHaveBeenCalled()
        expect(validity).toHaveBeenLastCalledWith(false)
        expect(container.textContent).toContain('requires at least one request option')
        await choose('Request overrides', 'Allow request overrides')
        expect(validity).toHaveBeenLastCalledWith(true)
        expect(container.textContent).not.toContain('requires at least one request option')
    })

    it('rejects typed invalid values before mutating the policy draft', async () => {
        await render(policy({ requestOptions: { temperature: 1 } }))
        await type('Temperature', '-1')
        expect(changed).not.toHaveBeenCalled()
        expect(input('Temperature').value).toBe('-1')
        expect(container.textContent).toContain('At least 0')
        expect(validity).toHaveBeenLastCalledWith(false)
        await type('Temperature', '0.5')
        expect(changed).toHaveBeenCalledExactlyOnceWith(
            applied(policy({ requestOptions: { temperature: 1 } }), {
                kind: 'option',
                key: 'temperature',
                value: '0.5'
            })
        )
        expect(validity).toHaveBeenLastCalledWith(true)
    })

    it('keeps decimal input spelling while updating the shared JSON draft', async () => {
        function Editor() {
            const [text, setText] = useState(policy())
            return createElement(ModelSettingsEditor, {
                identity,
                text,
                savedText: policy(),
                remote: false,
                disabled: false,
                nodeName: 'This PC',
                onTextChange: value => {
                    changed(value)
                    setText(value)
                },
                onValidityChange: validity
            })
        }
        await act(async () => root.render(createElement(Editor)))
        await type('Temperature', '0.')
        expect(input('Temperature').value).toBe('0.')
        await type('Temperature', '0.4')
        expect(input('Temperature').value).toBe('0.4')
        const draft = changed.mock.lastCall?.[0]
        if (!draft) throw new Error('Missing decimal draft')
        expect(selected(draft).options.temperature).toBe(0.4)
        expect(validity).toHaveBeenLastCalledWith(true)
    })

    it('keeps local options and the lock checkbox editable while locally locked', async () => {
        await render(policy({ localOnly: true, requestOptions: { temperature: 1 } }))
        const checkbox = container.querySelector<HTMLInputElement>('input[type="checkbox"]')
        expect(checkbox?.checked).toBe(true)
        expect(checkbox?.disabled).toBe(false)
        expect(input('Temperature').disabled).toBe(false)
        if (!checkbox) throw new Error('Missing local lock checkbox')
        await act(async () => checkbox.click())
        const result = changed.mock.lastCall?.[0]
        if (!result) throw new Error('Missing local unlock draft')
        expect(selected(result).localOnly).toBe(false)
    })

    it('keeps every remote option and mode locked after raw JSON clears the draft flag', async () => {
        await render(
            policy({ localOnly: false, requestOptions: { temperature: 2 } }),
            policy({ localOnly: true, requestOptions: { temperature: 1 } }),
            true
        )
        expect(container.textContent).toContain(
            "Locked on Peer. Change this profile's inference settings on that device."
        )
        expect(
            Array.from(container.querySelectorAll<HTMLInputElement>('input')).every(
                item => item.disabled
            )
        ).toBe(true)
        expect(selectTrigger('Request overrides').disabled).toBe(true)
        expect(container.querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(
            true
        )
        expect(changed).not.toHaveBeenCalled()
    })

    it('uses the saved identity when the draft reorders or renames profiles', async () => {
        const savedText = policy({ localOnly: true })
        const text = JSON.stringify({ profiles: [{ ...identity, name: 'renamed' }, other] })
        await render(text, savedText, true)
        expect(input('Temperature').disabled).toBe(true)
    })

    it('keeps the remote lock checkbox disabled even for an unlocked saved profile', async () => {
        await render(policy(), policy(), true)
        expect(input('Temperature').disabled).toBe(false)
        expect(container.querySelector<HTMLInputElement>('input[type="checkbox"]')?.disabled).toBe(
            true
        )
    })

    it('fails closed when the target snapshot has a malformed lock flag', async () => {
        await render(policy(), policy({ localOnly: 'false' }), true)
        expect(input('Temperature').disabled).toBe(true)
        expect(container.textContent).toContain("Reload the target's saved policy")
    })

    it('renders unsupported engines as an empty state without modifying JSON', async () => {
        await render(JSON.stringify({ profiles: [{ engine: 'ollama', model: 'm' }] }))
        expect(container.textContent).toContain('No supported profiles')
        expect(container.querySelector('input')).toBeNull()
        expect(changed).not.toHaveBeenCalled()
    })

    it('resets field validation when external JSON changes', async () => {
        await render(policy())
        await type('Temperature', '-1')
        await render(policy({ requestOptions: { temperature: 2 } }))
        expect(input('Temperature').value).toBe('2')
        expect(validity).toHaveBeenLastCalledWith(true)
    })

    it('keeps typed values readable while the parent disables edits', async () => {
        const text = policy({ requestOptions: { temperature: 0.7 } })
        await render(text, text, true, true)
        expect(input('Temperature').value).toBe('0.7')
        expect(input('Temperature').disabled).toBe(true)
        expect(changed).not.toHaveBeenCalled()
    })
})
