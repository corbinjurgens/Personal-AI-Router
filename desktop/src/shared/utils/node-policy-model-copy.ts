// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelConfiguration } from '@/shared/types/node-policy'
import type { ModelIdentity } from '@/shared/types/engines'
import type { JsonValue } from '@/shared/types/json'
import { parseNodeRoles } from '@/shared/utils/node-roles'
import type {
    InspectModelResult,
    ModelChange,
    ModelSettingsIdentity,
    RequestedLoad
} from '@/shared/types/model-config'
import {
    applyModelSettings,
    MODEL_OPTION_FIELDS,
    parseModelSettings
} from '@/shared/utils/node-policy-model-settings'

type CopyError = { code: string; message: string }
type Result<T> = { ok: true; value: T } | { ok: false; error: CopyError }
type Profile = Extract<ReturnType<typeof parseModelSettings>, { ok: true }>['value'][number]

/** Only recorded full hashes establish artifact correspondence. */
export function profileDeployMatch(
    source: ModelIdentity | undefined,
    target: ModelIdentity | undefined
): {
    strength: 'verified' | 'companions-unverified' | 'possible' | 'different'
    missingCompanions: string[]
} {
    if (!source || !target) return { strength: 'different', missingCompanions: [] }
    if (
        source.contentId &&
        source.contentId === target.contentId &&
        source.strength === 'verified' &&
        target.strength === 'verified'
    ) {
        const missingCompanions = (source.files ?? [])
            .filter(file => file.role !== 'model')
            .filter(
                file =>
                    !file.sha256 ||
                    !/^[0-9a-f]{64}$/.test(file.sha256) ||
                    !target.files?.some(
                        other => other.role === file.role && other.sha256 === file.sha256
                    )
            )
            .map(file => file.name || file.role)
        return {
            strength: missingCompanions.length ? 'companions-unverified' : 'verified',
            missingCompanions
        }
    }
    const possible =
        (!source.contentId || !target.contentId) &&
        !!source.repository &&
        source.repository === target.repository &&
        !!source.quantization &&
        source.quantization === target.quantization &&
        source.totalBytes !== undefined &&
        source.totalBytes === target.totalBytes
    return { strength: possible ? 'possible' : 'different', missingCompanions: [] }
}

/** Build a closed patch only; the coupled ModelChange owner creates and persists profiles. */
export function deployModelPolicyPatch(
    sourceText: string,
    source: ModelSettingsIdentity,
    targetText: string,
    target: ModelSettingsIdentity,
    selection: { fields: string[]; mode: boolean },
    savedTargetLocked: boolean,
    metadata: { displayName?: string; icon?: string },
    roles: string[]
): Result<NonNullable<ModelChange['policy']>> {
    const from = profile(sourceText, source)
    if (!from.ok) return from
    const parsed = parseModelSettings(targetText)
    if (!parsed.ok) return parsed
    const existing = parsed.value.find(item => item.key === target.profileKey)
    if (existing && existing.engine !== target.engine)
        return { ok: false, error: { code: 'profile', message: 'The target key is in use.' } }
    const document: JsonValue = JSON.parse(targetText)
    const declarations = jsonObject(document) ? document.roles : undefined
    const declared = parseNodeRoles(declarations)
    if (
        (declarations !== undefined &&
            (!Array.isArray(declarations) ||
                declarations.length !== declared.length ||
                declarations.some(
                    (role, index) =>
                        !jsonObject(role) ||
                        !Array.isArray(role.profiles) ||
                        role.profiles.length !== declared[index].profiles.length
                ))) ||
        new Set(declared.map(role => role.name)).size !== declared.length ||
        new Set(roles).size !== roles.length ||
        roles.some(name => !declared.some(role => role.name === name))
    )
        return { ok: false, error: { code: 'roles', message: 'Select declared target roles.' } }
    const displayName = metadata.displayName ?? existing?.displayName ?? from.value.displayName
    const icon = metadata.icon ?? existing?.icon ?? from.value.icon
    // A separate minimal baseline avoids reserializing target-owned numeric tokens.
    const synthetic = JSON.stringify({
        profiles: [
            {
                key: target.profileKey,
                engine: target.engine,
                model: target.profileKey,
                displayName,
                icon
            }
        ]
    })
    const validMetadata = parseModelSettings(synthetic)
    if (!validMetadata.ok) return validMetadata
    const baseline = existing ? targetText : synthetic
    const copied = copyModelSettings(
        sourceText,
        source,
        baseline,
        target,
        selection,
        savedTargetLocked
    )
    if (!copied.ok) return copied
    const patch = modelPolicyPatch(baseline, copied.value, target)
    if (!patch.ok) return patch
    const memberships: NonNullable<NonNullable<ModelChange['policy']>['roles']> = {}
    for (const role of declared) {
        if (roles.includes(role.name)) memberships[role.name] = true
    }
    return {
        ok: true,
        value: {
            ...patch.value,
            ...(!existing || metadata.displayName !== undefined ? { displayName } : {}),
            ...(!existing || metadata.icon !== undefined ? { icon } : {}),
            ...(roles.length ? { roles: memberships } : {})
        }
    }
}

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
    'cpuExperts',
    'fit',
    'fitTarget',
    'fitCtx',
    'extraOptions',
    'cacheTypeK',
    'cacheTypeV',
    'flashAttention',
    'batchSize',
    'ubatchSize',
    'threads',
    'threadsBatch',
    'kvOffload',
    'opOffload',
    'cacheRAM'
]

/** Copy requested leaves only. No resolved allocation or local artifact authority crosses nodes. */
export function portableLoadPatch(
    requested: RequestedLoad,
    fields: Array<keyof RequestedLoad>,
    inspection?: InspectModelResult
): NonNullable<ModelChange['load']> {
    const set: RequestedLoad = {}
    const clear: Array<keyof RequestedLoad> = []
    for (const key of fields) {
        if (key === 'fitTarget') continue
        if (key === 'extraOptions') {
            const descriptor = inspection?.options?.find(option => option.field === key)
            if (descriptor?.kind !== 'options' || !descriptor.supported || !descriptor.portable)
                throw new Error('Extra options require a supported portable source descriptor.')
            const options = Object.fromEntries(
                Object.entries(requested.extraOptions ?? {}).filter(
                    ([name]) => !descriptor.nonPortableKeys?.includes(name)
                )
            )
            if (Object.keys(options).length) set.extraOptions = options
            else clear.push(key)
            continue
        }
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
    identity: ModelSettingsIdentity
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
            ...(from.value.displayName !== to.value.displayName
                ? { displayName: to.value.displayName }
                : {}),
            ...(from.value.icon !== to.value.icon ? { icon: to.value.icon } : {}),
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
function policyRemainder(
    text: string,
    identity: ModelSettingsIdentity,
    selected: string[]
): string {
    const document: JsonValue = JSON.parse(text)
    const profiles =
        jsonObject(document) && Array.isArray(document.profiles) ? document.profiles : []
    const profileIndex = profiles.findIndex(
        item =>
            jsonObject(item) && item.engine === identity.engine && item.key === identity.profileKey
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
                        (key === 'displayName' ||
                            key === 'icon' ||
                            key === 'requestOptionsMode' ||
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
    identity: ModelSettingsIdentity,
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
    const parsed = profile(text, identity)
    const servedModel = parsed.ok ? parsed.value.model : undefined
    const matches = (item: JsonValue) =>
        item !== null &&
        typeof item === 'object' &&
        !Array.isArray(item) &&
        item.engine === identity.engine &&
        servedModel !== undefined &&
        item.model === servedModel
    const roles = parseNodeRoles(root.roles)
    const member = (key: string) => Array.isArray(root[key]) && root[key].some(matches)
    const profiles = Array.isArray(root.profiles) ? root.profiles : []
    const selected = profiles.find(
        item =>
            jsonObject(item) && item.key === identity.profileKey && item.engine === identity.engine
    )
    const profileKey =
        jsonObject(selected) && typeof selected.key === 'string' ? selected.key : undefined
    const qualified = profiles.some(item => {
        if (item !== selected || !parsed.ok || !jsonObject(item)) return false
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
                    model.model === servedModel &&
                    model.state === 'satisfied'
            ) === true &&
            evidence.policyRevision === root.revision
        )
    })
    return {
        roles,
        profileKey,
        served:
            root.servedModels == null
                ? 'unrestricted'
                : member('servedModels')
                  ? 'included'
                  : 'excluded',
        resident: member('residentModels'),
        locked: parsed.ok && parsed.value.localOnly,
        qualified
    }
}

function profile(text: string, identity: ModelSettingsIdentity): Result<Profile> {
    const parsed = parseModelSettings(text)
    if (!parsed.ok) return parsed
    const found = parsed.value.find(
        item => item.engine === identity.engine && item.key === identity.profileKey
    )
    return found
        ? { ok: true, value: found }
        : {
              ok: false,
              error: { code: 'profile', message: 'The selected model needs an existing profile.' }
          }
}

function supported(key: string, identity: ModelSettingsIdentity): boolean {
    return (
        MODEL_OPTION_FIELDS.some(field => field.key === key) &&
        (key !== 'min_p' || identity.engine === 'llamacpp')
    )
}

/** Initial selection contains only numeric options supported by both engines. */
export function defaultModelCopyFields(
    sourceText: string,
    source: ModelSettingsIdentity,
    target: ModelSettingsIdentity
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
    source: ModelSettingsIdentity,
    targetText: string,
    target: ModelSettingsIdentity,
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
