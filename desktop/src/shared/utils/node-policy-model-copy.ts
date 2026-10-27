// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NodeModelReference } from '@/shared/types/node-policy'
import {
    applyModelSettings,
    MODEL_OPTION_FIELDS,
    parseModelSettings
} from '@/shared/utils/node-policy-model-settings'

type CopyError = { code: string; message: string }
type Result<T> = { ok: true; value: T } | { ok: false; error: CopyError }
type Profile = Extract<ReturnType<typeof parseModelSettings>, { ok: true }>['value'][number]

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
