// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelConfiguration, NodeModelReference } from '@/shared/types/node-policy'
import type { JsonValue } from '@/shared/types/json'
import type { InspectModelResult, ModelChange, RequestedLoad } from '@/shared/types/model-config'
import {
    applyModelSettings,
    MODEL_OPTION_FIELDS,
    parseModelSettings
} from '@/shared/utils/node-policy-model-settings'

type CopyError = { code: string; message: string }
type Result<T> = { ok: true; value: T } | { ok: false; error: CopyError }
type Profile = Extract<ReturnType<typeof parseModelSettings>, { ok: true }>['value'][number]

/** The common copy default is tuning only. Routing always requires an explicit selection. */
export function portableLoadFields(inspection: InspectModelResult): Array<keyof RequestedLoad> {
    return loadFields.filter(key =>
        inspection.options?.some(
            option => option.field === key && option.portable && option.supported
        )
    )
}

export function defaultPortableLoadFields(
    inspection: InspectModelResult,
    requested: RequestedLoad
): Array<keyof RequestedLoad> {
    return portableLoadFields(inspection).filter(field => requested[field] !== undefined)
}

export const loadFields: Array<keyof RequestedLoad> = [
    'context',
    'slots',
    'gpu',
    'cacheTypeK',
    'cacheTypeV',
    'flashAttention',
    'batchSize',
    'ubatchSize',
    'threads',
    'threadsBatch',
    'kvOffload',
    'opOffload',
    'sleepIdleSeconds',
    'cacheRAM'
]

/** Copy requested leaves only. No resolved allocation or local artifact authority crosses nodes. */
export function portableLoadPatch(
    requested: RequestedLoad,
    fields: Array<keyof RequestedLoad>
): NonNullable<ModelChange['load']> {
    const set: RequestedLoad = {}
    const clear: Array<keyof RequestedLoad> = []
    for (const key of fields) {
        // Object.assign retains the closed RequestedLoad shape without a dynamic index assertion.
        if (requested[key] === undefined) clear.push(key)
        else Object.assign(set, { [key]: structuredClone(requested[key]) })
    }
    return { set, clear }
}

/** Turn only selected-profile generation edits into the closed service patch. */
export function modelPolicyPatch(
    baseline: string,
    draft: string,
    identity: NodeModelReference
): Result<NonNullable<ModelChange['policy']>> {
    if (baseline === draft) return { ok: true, value: {} }
    const from = profile(baseline, identity)
    const to = profile(draft, identity)
    if (!from.ok) return from
    if (!to.ok) return to
    const generation: NonNullable<NonNullable<ModelChange['policy']>['generation']> = {}
    const keys: Array<keyof typeof generation> = [
        'temperature',
        'top_p',
        'top_k',
        'min_p',
        'repeat_penalty',
        'frequency_penalty',
        'presence_penalty',
        'seed'
    ]
    for (const key of keys) {
        if (from.value.options[key] === to.value.options[key]) continue
        const value = to.value.options[key]
        if (value !== undefined && typeof value !== 'number')
            return { ok: false, error: { code: 'option', message: 'Generation must be numeric.' } }
        generation[key] = value ?? null
        const changed = applyModelSettings(draft, identity, {
            kind: 'option',
            key,
            value: value === undefined ? '' : String(value)
        })
        if (!changed.ok) return changed
    }
    if (from.value.mode !== to.value.mode) {
        const changed = applyModelSettings(draft, identity, { kind: 'mode', value: to.value.mode })
        if (!changed.ok) return changed
    }
    if (
        policyRemainder(baseline, identity, Object.keys(generation)) !==
        policyRemainder(draft, identity, Object.keys(generation))
    )
        return {
            ok: false,
            error: {
                code: 'selection',
                message:
                    'Save or discard profile, lock and unrelated Advanced policy edits before a combined model change.'
            }
        }
    return {
        ok: true,
        value: {
            ...(Object.keys(generation).length ? { generation } : {}),
            ...(from.value.mode !== to.value.mode ? { requestOptionsMode: to.value.mode } : {})
        }
    }
}

function jsonObject(value: JsonValue | undefined): value is { [key: string]: JsonValue } {
    return (
        value !== undefined && value !== null && typeof value === 'object' && !Array.isArray(value)
    )
}

/** Compare untouched content losslessly, allowing object-key ordering and selected leaf additions. */
function policyRemainder(text: string, identity: NodeModelReference, selected: string[]): string {
    const document: JsonValue = JSON.parse(text)
    const profiles =
        jsonObject(document) && Array.isArray(document.profiles) ? document.profiles : []
    const profileIndex = profiles.findIndex(
        item => jsonObject(item) && item.engine === identity.engine && item.model === identity.model
    )
    // Keep primitive tokens verbatim. A tagged syntax tree avoids numeric rounding
    // and cannot confuse an original object/string with a protected number.
    const tokens =
        text.match(
            /"(?:[^"\\]|\\.)*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null|[{}\u005b\u005d,:]/g
        ) ?? []
    let index = 0
    const value = (path: string[]): JsonValue => {
        const token = tokens[index++]
        if (token === '{') {
            const entries: Array<[string, JsonValue]> = []
            while (tokens[index] !== '}') {
                const key: string = JSON.parse(tokens[index++])
                index++ // colon, already validated by parseModelSettings
                const child = value([...path, key])
                const selectedProfile =
                    path.length === 2 && path[0] === 'profiles' && path[1] === String(profileIndex)
                const selectedOptions =
                    path.length === 3 &&
                    path[0] === 'profiles' &&
                    path[1] === String(profileIndex) &&
                    path[2] === 'requestOptions'
                if (
                    !(selectedOptions && selected.includes(key)) &&
                    !(
                        selectedProfile &&
                        (key === 'requestOptionsMode' ||
                            (key === 'requestOptions' && JSON.stringify(child) === '["object",[]]'))
                    )
                )
                    entries.push([key, child])
                if (tokens[index] === ',') index++
            }
            index++
            return ['object', entries.sort(([a], [b]) => a.localeCompare(b))]
        }
        if (token === '[') {
            const children: JsonValue[] = []
            while (tokens[index] !== ']') {
                children.push(value([...path, String(children.length)]))
                if (tokens[index] === ',') index++
            }
            index++
            return ['array', children]
        }
        return ['primitive', token]
    }
    return JSON.stringify(value([]))
}

export function modelRoutingState(
    text: string,
    identity: NodeModelReference,
    evidence?: ModelConfiguration
) {
    let document: JsonValue
    try {
        document = JSON.parse(text)
    } catch {
        document = null
    }
    const root =
        document !== null && typeof document === 'object' && !Array.isArray(document)
            ? document
            : {}
    const matches = (item: JsonValue) =>
        item !== null &&
        typeof item === 'object' &&
        !Array.isArray(item) &&
        item.engine === identity.engine &&
        item.model === identity.model
    const roles =
        root.roles !== null && typeof root.roles === 'object' && !Array.isArray(root.roles)
            ? root.roles
            : {}
    const member = (key: string) => Array.isArray(root[key]) && root[key].some(matches)
    const parsed = profile(text, identity)
    const profiles = Array.isArray(root.profiles) ? root.profiles : []
    const qualified = profiles.some(item => {
        if (!matches(item) || item === null || typeof item !== 'object' || Array.isArray(item))
            return false
        return (
            typeof item.contextTokens === 'number' &&
            item.contextTokens > 0 &&
            typeof item.maxOutputTokens === 'number' &&
            item.maxOutputTokens > 0 &&
            typeof item.maxConcurrent === 'number' &&
            Number.isSafeInteger(item.maxConcurrent) &&
            item.maxConcurrent > 0 &&
            typeof item.memoryBytes === 'number' &&
            Number.isSafeInteger(item.memoryBytes) &&
            item.memoryBytes > 0 &&
            jsonObject(item.qualification) &&
            typeof item.qualification.id === 'string' &&
            !!item.qualification.id &&
            typeof item.qualification.revision === 'string' &&
            /^[1-9]\d*$/.test(item.qualification.revision) &&
            evidence?.models.some(
                model =>
                    model.engine === identity.engine &&
                    model.model === identity.model &&
                    model.state === 'satisfied'
            ) === true &&
            evidence.policyRevision === root.revision
        )
    })
    return {
        roles,
        served:
            root.servedModels == null
                ? 'unrestricted'
                : member('servedModels')
                  ? 'included'
                  : 'excluded',
        resident: member('residentModels'),
        binding: member('engineBindings'),
        locked: parsed.ok && parsed.value.localOnly,
        qualified
    }
}

function profile(text: string, identity: NodeModelReference): Result<Profile> {
    const parsed = parseModelSettings(text)
    if (!parsed.ok) return parsed
    const found = parsed.value.find(
        item => item.engine === identity.engine && item.model === identity.model
    )
    return found
        ? { ok: true, value: found }
        : {
              ok: false,
              error: { code: 'profile', message: 'The selected model needs an existing profile.' }
          }
}

function supported(key: string, identity: NodeModelReference): boolean {
    return (
        MODEL_OPTION_FIELDS.some(field => field.key === key) &&
        (key !== 'min_p' || identity.engine === 'llamacpp')
    )
}

/** Initial selection contains only numeric options supported by both engines. */
export function defaultModelCopyFields(
    sourceText: string,
    source: NodeModelReference,
    target: NodeModelReference
): Result<string[]> {
    const parsed = profile(sourceText, source)
    if (!parsed.ok) return parsed
    return {
        ok: true,
        value: MODEL_OPTION_FIELDS.filter(field => {
            const value = parsed.value.options[field.key]
            return (
                supported(field.key, target) &&
                supported(field.key, source) &&
                typeof value === 'number' &&
                Number.isFinite(value)
            )
        }).map(field => field.key)
    }
}

/** Apply selected generation leaves to a fresh target draft. Lock authority comes from saved target state. */
export function copyModelSettings(
    sourceText: string,
    source: NodeModelReference,
    targetText: string,
    target: NodeModelReference,
    selection: { fields: string[]; mode: boolean },
    savedTargetLocked: boolean
): Result<string> {
    const from = profile(sourceText, source)
    if (!from.ok) return from
    const to = profile(targetText, target)
    if (!to.ok) return to
    if (savedTargetLocked && (selection.fields.length > 0 || selection.mode))
        return { ok: false, error: { code: 'locked', message: 'The target profile is locked.' } }

    const fields = [...new Set(selection.fields)]
    const edits: { key: string; value: string }[] = []
    const resultingOptions = { ...to.value.options }
    for (const key of fields) {
        if (!supported(key, source) || !supported(key, target))
            return {
                ok: false,
                error: {
                    code: 'option',
                    message: 'The selected option is not supported by both engines.'
                }
            }
        const value = from.value.options[key]
        if (value === undefined) {
            delete resultingOptions[key]
            edits.push({ key, value: '' })
        } else if (typeof value === 'number' && Number.isFinite(value)) {
            resultingOptions[key] = value
            edits.push({ key, value: String(value) })
        } else {
            return {
                ok: false,
                error: {
                    code: 'option',
                    message: 'The selected source option is not a finite number.'
                }
            }
        }
    }
    const mode = selection.mode ? from.value.mode : to.value.mode
    if (mode === 'enforced' && Object.keys(resultingOptions).length === 0)
        return {
            ok: false,
            error: {
                code: 'mode',
                message: 'Force PAIR values requires at least one request option.'
            }
        }

    let text = targetText
    // Setting before clearing keeps a valid enforced profile during the aggregate patch.
    for (const edit of edits.sort((a, b) => Number(a.value === '') - Number(b.value === ''))) {
        const result = applyModelSettings(text, target, { kind: 'option', ...edit })
        if (!result.ok) return result
        text = result.value
    }
    if (selection.mode || (mode === 'enforced' && edits.length > 0)) {
        const result = applyModelSettings(text, target, { kind: 'mode', value: mode })
        if (!result.ok) return result
        text = result.value
    }
    return { ok: true, value: text }
}
