// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { JsonValue } from '@/shared/types/json'
import type { NodeModelReference } from '@/shared/types/node-policy'
import type { ModelSettingsIdentity } from '@/shared/types/model-config'
import { isValidRoutingEngineId } from '@/shared/utils/engines'
import { isReservedProfileKey } from '@/shared/utils/node-roles'

type JsonMap = { [key: string]: JsonValue }
type SettingsError = { code: 'parse' | 'profile' | 'option' | 'mode' | 'flag'; message: string }
type Result<T> = { ok: true; value: T } | { ok: false; error: SettingsError }

interface ModelSettingsProfile extends NodeModelReference {
    key: string
    displayName: string
    icon: string
    options: JsonMap
    mode: 'defaults' | 'enforced'
    localOnly: boolean
}

/** Mirror nodepolicy.ValidateProfileKey, including its broad colon reservation. */
export function isValidProfileKey(key: string): boolean {
    return validText(key, 512) && !isReservedProfileKey(key)
}

/** Only the suggested slug has an ASCII length limit; saved keys retain UTF-8. */
export function defaultProfileKey(model: string): string {
    return model
        .toLowerCase()
        .replace(/[^a-z0-9._-]+/g, '-')
        .slice(0, 64)
}

function validText(text: string, limit: number): boolean {
    return (
        text.length > 0 &&
        text.isWellFormed() &&
        new TextEncoder().encode(text).length <= limit &&
        !Array.from(text).some(character => {
            const code = character.codePointAt(0) ?? 0
            return code < 0x20 || (code >= 0x7f && code <= 0x9f)
        })
    )
}

function validIcon(icon: string): boolean {
    return ['', 'chat-single', 'code', 'cube', 'document', 'image'].includes(icon)
}

export const MODEL_OPTION_FIELDS = [
    { key: 'temperature', label: 'Temperature', help: 'At least 0.' },
    { key: 'top_p', label: 'Top P', help: 'Between 0 and 1.' },
    { key: 'top_k', label: 'Top K', help: 'A nonnegative safe integer.' },
    { key: 'repeat_penalty', label: 'Repeat penalty', help: 'Greater than 0.' },
    { key: 'frequency_penalty', label: 'Frequency penalty', help: 'Between -2 and 2.' },
    { key: 'presence_penalty', label: 'Presence penalty', help: 'Between -2 and 2.' },
    { key: 'seed', label: 'Seed', help: 'A safe integer.' },
    { key: 'min_p', label: 'Min P', help: 'Between 0 and 1. llama.cpp only.' }
]

function object(value: JsonValue | undefined): value is JsonMap {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function optionValid(key: string, value: number): boolean {
    if (!Number.isFinite(value)) return false
    switch (key) {
        case 'temperature':
            return value >= 0
        case 'top_p':
        case 'min_p':
            return value >= 0 && value <= 1
        case 'top_k':
            return Number.isSafeInteger(value) && value >= 0
        case 'repeat_penalty':
            return value > 0
        case 'frequency_penalty':
        case 'presence_penalty':
            return value >= -2 && value <= 2
        case 'seed':
            return Number.isSafeInteger(value)
        default:
            return false
    }
}

function policyDraft(text: string): Result<{ policy: JsonMap; profiles: JsonMap[] }> {
    let policy: JsonValue
    try {
        policy = JSON.parse(text)
    } catch {
        return {
            ok: false,
            error: {
                code: 'parse',
                message:
                    'Fix the Advanced policy JSON to edit inference settings. Your policy text is kept.'
            }
        }
    }
    if (
        !object(policy) ||
        (policy.profiles !== undefined &&
            policy.profiles !== null &&
            !Array.isArray(policy.profiles))
    )
        return {
            ok: false,
            error: { code: 'parse', message: 'The policy must be an object with a profiles list.' }
        }
    const profiles: JsonMap[] = []
    const rootFields = objectFields(text, whitespaceEnd(text, 0))
    const profileField = rootFields.find(field => field.key === 'profiles')
    const spans =
        profileField && Array.isArray(policy.profiles) ? arrayValues(text, profileField.start) : []
    if (
        ambiguousFields(rootFields, ['profiles', 'residentModels']) ||
        spans.some(span => {
            if (text[span.start] !== '{') return false
            const fields = objectFields(text, span.start)
            const options = fields.find(field => field.key === 'requestOptions')
            return (
                ambiguousFields(fields, [
                    'key',
                    'displayName',
                    'icon',
                    'engine',
                    'model',
                    'requestOptions',
                    'requestOptionsMode',
                    'localOnly'
                ]) ||
                (options !== undefined &&
                    text[options.start] === '{' &&
                    ambiguousFields(objectFields(text, options.start)))
            )
        })
    )
        return {
            ok: false,
            error: {
                code: 'parse',
                message: 'Remove ambiguous duplicate policy or profile keys before editing.'
            }
        }
    const identities = new Set<string>()
    for (const profile of Array.isArray(policy.profiles) ? policy.profiles : []) {
        if (
            !object(profile) ||
            typeof profile.engine !== 'string' ||
            !profile.engine ||
            typeof profile.model !== 'string' ||
            !profile.model ||
            !profileIdentityValid({ engine: profile.engine, model: profile.model }) ||
            typeof profile.key !== 'string' ||
            !isValidProfileKey(profile.key) ||
            typeof profile.displayName !== 'string' ||
            !validText(profile.displayName, 128) ||
            (profile.icon !== undefined &&
                (typeof profile.icon !== 'string' || !validIcon(profile.icon)))
        )
            return {
                ok: false,
                error: {
                    code: 'profile',
                    message:
                        'Each profile needs a valid key, display name, icon, engine and served model.'
                }
            }
        const key = profile.key
        if (identities.has(key))
            return {
                ok: false,
                error: {
                    code: 'profile',
                    message: 'Profiles must have distinct keys.'
                }
            }
        identities.add(key)
        if (
            profile.requestOptionsMode !== undefined &&
            profile.requestOptionsMode !== 'defaults' &&
            profile.requestOptionsMode !== 'enforced'
        )
            return {
                ok: false,
                error: { code: 'mode', message: 'requestOptionsMode must be defaults or enforced.' }
            }
        if (profile.localOnly !== undefined && typeof profile.localOnly !== 'boolean')
            return { ok: false, error: { code: 'flag', message: 'localOnly must be a boolean.' } }
        if (
            profile.requestOptions !== undefined &&
            profile.requestOptions !== null &&
            !object(profile.requestOptions)
        )
            return {
                ok: false,
                error: { code: 'option', message: 'requestOptions must be a JSON object.' }
            }
        if (
            profile.requestOptionsMode === 'enforced' &&
            (!object(profile.requestOptions) || Object.keys(profile.requestOptions).length === 0)
        )
            return {
                ok: false,
                error: {
                    code: 'mode',
                    message: 'Force PAIR values requires at least one request option.'
                }
            }
        profiles.push(profile)
    }
    return { ok: true, value: { policy, profiles } }
}

/** New drafts cannot claim an existing key, including profiles of another engine. */
export function newProfileKeyStatus(
    text: string,
    key?: string
): 'invalid' | 'duplicate' | 'available' {
    const draft = policyDraft(text)
    if (key === undefined || !isValidProfileKey(key) || !draft.ok) return 'invalid'
    return draft.value.profiles.some(profile => profile.key === key) ? 'duplicate' : 'available'
}

/** Supported controls are deliberately independent of inventory and profile display names. */
export function parseModelSettings(text: string): Result<ModelSettingsProfile[]> {
    const draft = policyDraft(text)
    if (!draft.ok) return draft
    const profiles: ModelSettingsProfile[] = []
    for (const profile of draft.value.profiles) {
        if (profile.engine !== 'llamacpp') continue
        if (
            typeof profile.model !== 'string' ||
            typeof profile.key !== 'string' ||
            typeof profile.displayName !== 'string'
        )
            continue
        profiles.push({
            displayName: profile.displayName,
            icon: typeof profile.icon === 'string' ? profile.icon : '',
            engine: profile.engine,
            model: profile.model,
            key: profile.key,
            options: object(profile.requestOptions) ? profile.requestOptions : {},
            mode: profile.requestOptionsMode === 'enforced' ? 'enforced' : 'defaults',
            localOnly: profile.localOnly === true
        })
    }
    return { ok: true, value: profiles }
}

type ProfileEdit =
    | { kind: 'option'; key: string; value: string }
    | { kind: 'mode'; value: string }
    | { kind: 'localOnly'; value: boolean }

type ValueSpan = { start: number; end: number }

function sameFieldKey(left: string, right: string): boolean {
    const literal = left.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    return new RegExp(`^(?:${literal})$`, 'iu').exec(right)?.[0] === right
}

function ambiguousFields(fields: { key: string }[], canonical: string[] = []): boolean {
    return fields.some(
        (field, index) =>
            fields.slice(0, index).some(previous => sameFieldKey(previous.key, field.key)) ||
            canonical.some(key => key !== field.key && sameFieldKey(key, field.key))
    )
}

function profileIdentityValid(identity: NodeModelReference): boolean {
    return (
        isValidRoutingEngineId(identity.engine) &&
        identity.engine.trim() === identity.engine &&
        identity.model.length > 0 &&
        identity.model.isWellFormed() &&
        new TextEncoder().encode(identity.model).length <= 512 &&
        !Array.from(identity.model).some(char => {
            const code = char.charCodeAt(0)
            return code < 0x20 || (code >= 0x7f && code <= 0x9f)
        })
    )
}

function whitespaceEnd(text: string, start: number): number {
    while (/\s/.test(text[start] ?? '') && start < text.length) start++
    return start
}

// These offsets are read only after JSON.parse has validated the whole draft.
// Keep opaque values as source text instead of round-tripping them through Number.
function valueEnd(text: string, start: number): number {
    if (text[start] === '"') {
        for (let index = start + 1; index < text.length; index++) {
            if (text[index] === '\\') index++
            else if (text[index] === '"') return index + 1
        }
    }
    if (text[start] === '{' || text[start] === '[') {
        let depth = 1
        for (let index = start + 1; index < text.length; index++) {
            const char = text[index]
            if (char === '"') index = valueEnd(text, index) - 1
            else if (char === '{' || char === '[') depth++
            else if ((char === '}' || char === ']') && --depth === 0) return index + 1
        }
    }
    let end = start
    while (end < text.length && !/[\s,}\]]/.test(text[end])) end++
    return end
}

function objectFields(text: string, start = 0) {
    const fields: (ValueSpan & { key: string; keyStart: number })[] = []
    let index = whitespaceEnd(text, start + 1)
    while (text[index] !== '}' && index < text.length) {
        const keyStart = index
        const keyEnd = valueEnd(text, index)
        const key: string = JSON.parse(text.slice(index, keyEnd))
        index = whitespaceEnd(text, whitespaceEnd(text, keyEnd) + 1)
        const end = valueEnd(text, index)
        fields.push({ key, keyStart, start: index, end })
        index = whitespaceEnd(text, end)
        if (text[index] === ',') index = whitespaceEnd(text, index + 1)
    }
    return fields
}

function arrayValues(text: string, start: number): ValueSpan[] {
    const values: ValueSpan[] = []
    let index = whitespaceEnd(text, start + 1)
    while (text[index] !== ']' && index < text.length) {
        const end = valueEnd(text, index)
        values.push({ start: index, end })
        index = whitespaceEnd(text, end)
        if (text[index] === ',') index = whitespaceEnd(text, index + 1)
    }
    return values
}

function replaceField(text: string, key: string, raw: string | undefined): string {
    const fields = objectFields(text)
    const index = fields.findLastIndex(field => field.key === key)
    const field = fields[index]
    if (field) {
        if (raw !== undefined) return text.slice(0, field.start) + raw + text.slice(field.end)
        const previous = fields[index - 1]
        const next = fields[index + 1]
        const start = previous ? previous.end : field.keyStart
        const end = !previous && next ? next.keyStart : field.end
        return text.slice(0, start) + text.slice(end)
    }
    if (raw === undefined) return text
    const last = fields.at(-1)
    const at = last ? last.end : 1
    return text.slice(0, at) + (last ? ',' : '') + JSON.stringify(key) + ':' + raw + text.slice(at)
}

/** Replace only the selected profile's settings; the service owns save authorization. */
export function applyModelSettings(
    text: string,
    identity: ModelSettingsIdentity,
    edit: ProfileEdit
): Result<string> {
    const draft = policyDraft(text)
    if (!draft.ok) return draft
    const profile = draft.value.profiles.find(
        item => item.engine === identity.engine && item.key === identity.profileKey
    )
    if (!profile || identity.engine !== 'llamacpp')
        return {
            ok: false,
            error: { code: 'profile', message: 'Select a supported profile in this policy.' }
        }
    const updated = { ...profile }
    if (edit.kind === 'option') {
        const field = MODEL_OPTION_FIELDS.find(
            item =>
                item.key === edit.key && (item.key !== 'min_p' || identity.engine === 'llamacpp')
        )
        if (!field)
            return {
                ok: false,
                error: { code: 'option', message: 'Edit this option in Advanced policy JSON.' }
            }
        const value = edit.value.trim()
        const options = { ...(object(profile.requestOptions) ? profile.requestOptions : {}) }
        if (value === '') delete options[edit.key]
        else {
            const number = Number(value)
            if (
                !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value) ||
                !optionValid(edit.key, number)
            )
                return {
                    ok: false,
                    error: {
                        code: 'option',
                        message: `${field.label}: enter a finite number. ${field.help}`
                    }
                }
            options[edit.key] = number
        }
        if (Object.keys(options).length === 0) {
            delete updated.requestOptions
            delete updated.requestOptionsMode
        } else updated.requestOptions = options
    } else if (edit.kind === 'mode') {
        if (edit.value !== 'defaults' && edit.value !== 'enforced')
            return {
                ok: false,
                error: {
                    code: 'mode',
                    message: 'Select Allow request overrides or Force PAIR values.'
                }
            }
        if (edit.value === 'enforced') {
            if (!object(updated.requestOptions) || Object.keys(updated.requestOptions).length === 0)
                return {
                    ok: false,
                    error: {
                        code: 'mode',
                        message: 'Force PAIR values requires at least one request option.'
                    }
                }
            updated.requestOptionsMode = 'enforced'
        } else delete updated.requestOptionsMode
    } else {
        if (edit.value) updated.localOnly = true
        else delete updated.localOnly
    }
    if (updated.requestOptionsMode === 'defaults') delete updated.requestOptionsMode
    const profilesField = objectFields(text, whitespaceEnd(text, 0)).findLast(
        field => field.key === 'profiles'
    )
    // Identity was validated above; its array position locates only the source span.
    const span =
        profilesField &&
        arrayValues(text, profilesField.start)[draft.value.profiles.indexOf(profile)]
    if (!span)
        return {
            ok: false,
            error: { code: 'parse', message: 'Could not locate this profile in the policy draft.' }
        }
    let raw = text.slice(span.start, span.end)
    if (edit.kind === 'option') {
        const optionsField = objectFields(raw).findLast(field => field.key === 'requestOptions')
        const options =
            optionsField && object(profile.requestOptions)
                ? raw.slice(optionsField.start, optionsField.end)
                : '{}'
        raw = replaceField(
            raw,
            'requestOptions',
            updated.requestOptions === undefined
                ? undefined
                : replaceField(
                      options,
                      edit.key,
                      edit.value.trim() === '' ? undefined : JSON.stringify(Number(edit.value))
                  )
        )
    }
    if (updated.requestOptionsMode !== profile.requestOptionsMode)
        raw = replaceField(raw, 'requestOptionsMode', JSON.stringify(updated.requestOptionsMode))
    if (edit.kind === 'localOnly')
        raw = replaceField(raw, 'localOnly', JSON.stringify(updated.localOnly))
    return { ok: true, value: text.slice(0, span.start) + raw + text.slice(span.end) }
}
