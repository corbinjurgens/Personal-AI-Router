// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Button, FormField, Select, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { EngineSettingsTarget } from '@/shared/types/engine-settings'
import type { ModelChange, RequestedLoad } from '@/shared/types/model-config'
import {
    loadFields,
    modelPolicyPatch,
    modelRoutingState
} from '@/shared/utils/node-policy-model-copy'
import {
    engineSettingsKey,
    modelSettingsKey,
    useEngineSettingsStore,
    watchEngineSettings
} from '@/ui/stores/engine-settings.store'
import ModelSettingsPreview from './ModelSettingsPreview'
import ModelExtraOptions from './ModelExtraOptions'
import { useConnectionStore } from '@/ui/stores/connection.store'
import type { ModelConfiguration } from '@/shared/types/node-policy'
import { isValidProfileKey, newProfileKeyStatus } from '@/shared/utils/node-policy-model-settings'

function fieldPatch(
    field: keyof RequestedLoad,
    value: string,
    current: RequestedLoad
): RequestedLoad | null {
    const number = Number(value)
    switch (field) {
        case 'fit':
            return value === 'on' || value === 'off' ? { fit: value } : null
        case 'fitTarget':
            return /^[0-9]{1,7}(,[0-9]{1,7}){0,15}$/.test(value) ? { fitTarget: value } : null
        case 'fitCtx':
            return Number.isSafeInteger(number) && number >= 0 && number <= 16777216
                ? { fitCtx: number }
                : null
        case 'context':
            if (value === 'auto') return { context: { mode: 'auto', floor: 4096, cap: 8192 } }
            return Number.isSafeInteger(number)
                ? { context: { mode: 'fixed', tokens: number } }
                : null
        case 'gpu':
            if (value === 'auto') return { gpu: { mode: 'auto' } }
            if (value === 'cpu' || value === 'native-default') return { gpu: { mode: value } }
            if (value === 'layers')
                return { gpu: { mode: 'layers', layers: current.gpu?.layers ?? 0 } }
            return Number.isSafeInteger(number) ? { gpu: { mode: 'layers', layers: number } } : null
        case 'cacheTypeK':
        case 'cacheTypeV':
            return value === 'f16' || value === 'bf16' || value === 'q8_0' || value === 'q4_0'
                ? { [field]: value }
                : null
        case 'flashAttention':
            return value === 'inherit' || value === 'on' || value === 'off'
                ? { flashAttention: value }
                : null
        case 'kvOffload':
        case 'opOffload':
            return value === 'true' || value === 'false' ? { [field]: value === 'true' } : null
        default:
            return Number.isSafeInteger(number) ? { [field]: number } : null
    }
}

export default function ModelLoadSettings({
    target,
    modelId,
    profileKey,
    newProfile = false,
    policy,
    savedPolicy,
    policyRevision,
    authority,
    disabled,
    evidence
}: {
    target: EngineSettingsTarget
    modelId: string
    profileKey?: string
    newProfile?: boolean
    policy: string
    savedPolicy: string
    policyRevision: string
    authority: string
    disabled: boolean
    evidence?: ModelConfiguration
}) {
    const newKeyUnavailable = newProfile && newProfileKeyStatus(policy, profileKey) !== 'available'
    const key = modelSettingsKey(target, newKeyUnavailable ? undefined : profileKey)
    const entry = useEngineSettingsStore(state => state.entries[engineSettingsKey(target)])
    const draft = useEngineSettingsStore(state => state.modelDrafts[key])
    const unkeyedInspection = useEngineSettingsStore(
        state => state.modelDrafts[modelSettingsKey(target)]?.inspection
    )
    const [values, setValues] = useState<Record<string, string>>({})
    const [validation, setValidation] = useState<Record<string, string>>({})
    const validateField = (field: string, error?: string) =>
        setValidation(previous => {
            const next = { ...previous }
            delete next[field]
            if (error) next[field] = error
            return next
        })
    const snapshot = entry?.snapshot
    const savedLoad = newProfile
        ? undefined
        : snapshot?.settings.modelLoads?.find(item => item.profileKey === profileKey)
    const inspection =
        newProfile || !savedLoad
            ? unkeyedInspection?.modelId === modelId
                ? unkeyedInspection
                : undefined
            : draft?.inspection
    const identity = { engine: target.engine, model: modelId, profileKey }
    const generation = modelPolicyPatch(savedPolicy, policy, identity)
    const routing = modelRoutingState(savedPolicy, identity, evidence)
    const load = draft?.change?.load
    const removing = draft?.change?.remove === true
    const dirty = !!load || !!draft?.change?.policy || removing
    useEffect(() => {
        if (!draft?.change) {
            setValues({})
            setValidation({})
        }
    }, [draft?.change, key])
    const { nodeId, engine } = target
    useEffect(() => {
        const stop = watchEngineSettings()
        const store = useEngineSettingsStore.getState()
        const currentTarget = { nodeId, engine }
        let active = true
        void store.reload(currentTarget).then(() => {
            if (!active) return
            const currentSnapshot =
                useEngineSettingsStore.getState().entries[engineSettingsKey(currentTarget)]
                    ?.snapshot
            const typed =
                !newProfile &&
                currentSnapshot?.settings.modelLoads?.some(load => load.profileKey === profileKey)
            void store.inspectModel(currentTarget, modelId, typed ? profileKey : undefined)
        })
        return () => {
            active = false
            stop()
        }
    }, [nodeId, engine, modelId, profileKey, newProfile])
    useEffect(() => {
        useEngineSettingsStore.getState().invalidateModel(key)
        return () => useEngineSettingsStore.getState().invalidateModel(key)
    }, [key, authority, policy, policyRevision, disabled])
    const edit = (patch: Partial<ModelChange>) => {
        if (unavailable || removing) return
        useEngineSettingsStore.getState().editModel(key, {
            ...draft?.change,
            modelId,
            profileKey,
            expectedPolicyRevision: policyRevision,
            ...patch
        })
    }
    const unavailable =
        disabled ||
        newKeyUnavailable ||
        (profileKey !== undefined && !isValidProfileKey(profileKey)) ||
        !!entry?.unavailable ||
        !snapshot?.editable ||
        (!!draft?.busy && !draft.previewOperation) ||
        !!draft?.uncertain
    const blocked = unavailable || removing
    const requested = savedLoad?.requested ?? {}
    const resolved = savedLoad?.resolved
    const effectiveContext = load?.clear?.includes('context')
        ? undefined
        : (load?.set.context ?? requested.context)
    const effectiveGpu = load?.clear?.includes('gpu') ? undefined : (load?.set.gpu ?? requested.gpu)
    const effectiveFit = load?.clear?.includes('fit') ? undefined : (load?.set.fit ?? requested.fit)
    const effectiveCpuExperts = load?.clear?.includes('cpuExperts')
        ? undefined
        : (load?.set.cpuExperts ?? requested.cpuExperts)
    const cpuDescriptor = inspection?.options?.find(option => option.field === 'cpuExperts')
    const cpuLayersDescriptor = inspection?.options?.find(
        option => option.field === 'cpuExperts.layers'
    )
    const editCpuExperts = (value: RequestedLoad['cpuExperts']) => {
        const set = { ...load?.set }
        delete set.cpuExperts
        const clear: Array<keyof RequestedLoad> = (load?.clear ?? []).filter(
            field => field !== 'cpuExperts'
        )
        if (value) set.cpuExperts = value
        else clear.push('cpuExperts')
        validateField('cpuExperts.layers')
        edit({ load: { set, clear } })
    }
    const autoSelected = effectiveContext?.mode === 'auto' || effectiveGpu?.mode === 'auto'
    const draftIdentity = JSON.stringify(draft?.change)
    const invalid = Object.keys(validation).length > 0
    useEffect(() => {
        if (!draftIdentity || disabled || invalid || !policyRevision) return
        const timer = window.setTimeout(() => {
            const currentDraft = useEngineSettingsStore.getState().modelDrafts[key]
            const current = currentDraft?.change
            if (!currentDraft?.review && current?.load && JSON.stringify(current) === draftIdentity)
                void useEngineSettingsStore
                    .getState()
                    .previewModel({ nodeId, engine }, current, authority, true)
        }, 250)
        return () => window.clearTimeout(timer)
    }, [draftIdentity, disabled, invalid, policyRevision, key, nodeId, engine, authority])
    const controls = (basic: boolean) =>
        loadFields
            .filter(
                field =>
                    field !== 'cpuExperts' &&
                    ['context', 'slots', 'gpu', 'cacheTypeK', 'cacheTypeV'].includes(field) ===
                        basic
            )
            .map(field => {
                const descriptor = inspection?.options?.find(option => option.field === field)
                if (!descriptor)
                    return (
                        <Text key={field} kind="body/regular/sm">
                            {field}: support Unknown
                        </Text>
                    )
                const current = load?.clear?.includes(field)
                    ? undefined
                    : (load?.set[field] ?? requested[field])
                if (descriptor.kind === 'options')
                    return (
                        <ModelExtraOptions
                            key={`${key}:${field}`}
                            descriptor={descriptor}
                            value={
                                load?.clear?.includes('extraOptions')
                                    ? undefined
                                    : (load?.set.extraOptions ?? requested.extraOptions)
                            }
                            nativeOptions={inspection?.nativeOptions}
                            disabled={
                                blocked || !descriptor.supported || resolved?.kvUnified === true
                            }
                            reset={!draft?.change}
                            onChange={(options, error) => {
                                validateField('extraOptions', error)
                                if (error) {
                                    useEngineSettingsStore.getState().invalidateModel(key)
                                    return
                                }
                                const set = { ...load?.set }
                                delete set.extraOptions
                                const clear: Array<keyof RequestedLoad> = (
                                    load?.clear ?? []
                                ).filter(item => item !== 'extraOptions')
                                if (options) set.extraOptions = options
                                else clear.push('extraOptions')
                                edit({ load: { set, clear } })
                            }}
                        />
                    )
                if (
                    field === 'context' &&
                    typeof current === 'object' &&
                    'mode' in current &&
                    current.mode === 'auto'
                )
                    return null
                const display =
                    values[field] ??
                    (load?.clear?.includes(field)
                        ? ''
                        : typeof current === 'object'
                          ? field === 'context'
                              ? current.mode === 'auto'
                                  ? 'auto'
                                  : 'tokens' in current
                                    ? String(current.tokens ?? '')
                                    : ''
                              : 'mode' in current
                                ? current.mode
                                : ''
                          : current === undefined
                            ? ''
                            : String(current))
                const change = (value: string) => {
                    setValues(previous => ({ ...previous, [field]: value }))
                    const set = { ...load?.set }
                    delete set[field]
                    const clear = (load?.clear ?? []).filter(key => key !== field)
                    if (!value) clear.push(field)
                    else {
                        const patch = fieldPatch(field, value, { ...requested, ...set })
                        if (!patch) {
                            validateField(field, `${descriptor.label}: invalid value`)
                            edit({ load: { set, clear } })
                            return
                        }
                        Object.assign(set, patch)
                    }
                    validateField(field)
                    if (field === 'gpu') validateField('gpu.layers')
                    edit({ load: { set, clear } })
                }
                return (
                    <FormField
                        key={field}
                        slotLabel={descriptor.label}
                        slotHelp={`${descriptor.reason ?? ''} Inherited: ${JSON.stringify(descriptor.default) ?? 'Unknown'} (${descriptor.defaultSource ?? 'Unknown source'}). Range: ${descriptor.minimum ?? 'Unknown'} to ${descriptor.maximum ?? 'Unknown'}.`}
                    >
                        {descriptor.kind === 'enum' || descriptor.kind === 'boolean' ? (
                            <Select
                                aria-label={descriptor.label}
                                value={display}
                                disabled={
                                    blocked || !descriptor.supported || resolved?.kvUnified === true
                                }
                                items={[
                                    { value: '', children: 'Inherited' },
                                    ...(
                                        descriptor.values ??
                                        (descriptor.kind === 'boolean' ? ['true', 'false'] : [])
                                    ).map(value => ({ value, children: value })),
                                    ...(field === 'gpu' && !descriptor.values?.includes('auto')
                                        ? [{ value: 'auto', children: 'auto' }]
                                        : [])
                                ]}
                                onValueChange={change}
                            />
                        ) : (
                            <TextInput
                                aria-label={descriptor.label}
                                type={
                                    descriptor.kind === 'integer' || descriptor.kind === 'number'
                                        ? 'number'
                                        : 'text'
                                }
                                min={descriptor.minimum}
                                max={descriptor.maximum}
                                value={display}
                                disabled={
                                    blocked || !descriptor.supported || resolved?.kvUnified === true
                                }
                                onValueChange={change}
                            />
                        )}
                        <Button
                            size="small"
                            kind="tertiary"
                            disabled={
                                blocked || !descriptor.supported || resolved?.kvUnified === true
                            }
                            onClick={() => change('')}
                        >
                            Reset {descriptor.label} to inherited
                        </Button>
                        {field === 'gpu' && (load?.set.gpu ?? requested.gpu)?.mode === 'layers' && (
                            <TextInput
                                aria-label="Finite GPU layers"
                                type="number"
                                min={
                                    inspection?.options?.find(
                                        option => option.field === 'gpu.layers'
                                    )?.minimum
                                }
                                max={
                                    inspection?.options?.find(
                                        option => option.field === 'gpu.layers'
                                    )?.maximum ?? inspection?.facts?.layerCount
                                }
                                disabled={
                                    blocked ||
                                    resolved?.kvUnified === true ||
                                    !inspection?.options?.some(
                                        option => option.field === 'gpu.layers' && option.supported
                                    )
                                }
                                value={String((load?.set.gpu ?? requested.gpu)?.layers ?? '')}
                                onValueChange={value => {
                                    const layers = Number(value)
                                    if (!value || !Number.isSafeInteger(layers)) {
                                        validateField('gpu.layers', 'Enter finite GPU layers.')
                                        useEngineSettingsStore.getState().invalidateModel(key)
                                        return
                                    }
                                    validateField('gpu.layers')
                                    edit({
                                        load: {
                                            set: { ...load?.set, gpu: { mode: 'layers', layers } },
                                            clear: load?.clear?.filter(field => field !== 'gpu')
                                        }
                                    })
                                }}
                            />
                        )}
                        {field === 'gpu' &&
                            Number(inspection?.facts?.dimensions?.expert_count) > 0 && (
                                <FormField
                                    slotLabel="CPU expert placement"
                                    slotHelp="None keeps normal placement. All puts routed experts on CPU. First N puts routed experts from the first N model layers on CPU. Shared experts keep normal placement."
                                >
                                    <Select
                                        aria-label="CPU expert placement"
                                        value={effectiveCpuExperts?.mode ?? ''}
                                        disabled={
                                            blocked ||
                                            !cpuDescriptor?.supported ||
                                            resolved?.kvUnified === true
                                        }
                                        items={[
                                            { value: '', children: 'Inherited' },
                                            ...(cpuDescriptor?.values ?? []).map(value => ({
                                                value,
                                                children:
                                                    value === 'none'
                                                        ? 'None'
                                                        : value === 'all'
                                                          ? 'All'
                                                          : 'First N layers'
                                            }))
                                        ]}
                                        onValueChange={value => {
                                            setValues(previous => {
                                                const next = { ...previous }
                                                delete next['cpuExperts.layers']
                                                return next
                                            })
                                            if (!value) editCpuExperts(undefined)
                                            else if (value === 'none' || value === 'all')
                                                editCpuExperts({ mode: value })
                                            else if (value === 'layers')
                                                editCpuExperts({
                                                    mode: 'layers',
                                                    layers: effectiveCpuExperts?.layers ?? 0
                                                })
                                        }}
                                    />
                                    {effectiveCpuExperts?.mode === 'layers' && (
                                        <TextInput
                                            aria-label="CPU expert layers"
                                            type="number"
                                            min={cpuLayersDescriptor?.minimum ?? 0}
                                            max={
                                                cpuLayersDescriptor?.maximum ??
                                                inspection?.facts?.layerCount
                                            }
                                            value={
                                                values['cpuExperts.layers'] ??
                                                String(effectiveCpuExperts.layers ?? '')
                                            }
                                            disabled={
                                                blocked ||
                                                !cpuLayersDescriptor?.supported ||
                                                resolved?.kvUnified === true
                                            }
                                            onValueChange={value => {
                                                setValues(previous => ({
                                                    ...previous,
                                                    ['cpuExperts.layers']: value
                                                }))
                                                const layers = Number(value)
                                                const maximum =
                                                    cpuLayersDescriptor?.maximum ??
                                                    inspection?.facts?.layerCount ??
                                                    2147483647
                                                if (
                                                    !value ||
                                                    !Number.isInteger(layers) ||
                                                    layers < 0 ||
                                                    layers > Math.min(maximum, 2147483647)
                                                ) {
                                                    validateField(
                                                        'cpuExperts.layers',
                                                        'Enter a nonnegative CPU expert layer count within the model range.'
                                                    )
                                                    useEngineSettingsStore
                                                        .getState()
                                                        .invalidateModel(key)
                                                    return
                                                }
                                                editCpuExperts({ mode: 'layers', layers })
                                            }}
                                        />
                                    )}
                                    <Button
                                        size="small"
                                        kind="tertiary"
                                        disabled={
                                            blocked ||
                                            !cpuDescriptor?.supported ||
                                            resolved?.kvUnified === true
                                        }
                                        onClick={() => {
                                            setValues(previous => {
                                                const next = { ...previous }
                                                delete next['cpuExperts.layers']
                                                return next
                                            })
                                            editCpuExperts(undefined)
                                        }}
                                    >
                                        Reset CPU expert placement to inherited
                                    </Button>
                                </FormField>
                            )}
                    </FormField>
                )
            })
    return (
        <Stack gap="3">
            <Text kind="body/semibold/sm">Basic load</Text>
            <Text kind="body/regular/sm">
                Off remains Off. Saving settings never loads a model. The saved generation lock does
                not lock load settings.
            </Text>
            {controls(true)}
            {effectiveFit === 'on' && (
                <Text kind="body/regular/sm">
                    Native fit: PAIR does not estimate memory for this configuration.
                </Text>
            )}
            {inspection?.configurationKeyReason === 'gpu-selection' && (
                <Text kind="body/regular/sm">
                    GPU selection: PAIR does not estimate memory for this configuration.
                </Text>
            )}
            {!!inspection?.unreportedLaunchOptions?.length && (
                <Text role="alert" kind="body/regular/sm">
                    This llama.cpp build does not report:{' '}
                    {inspection.unreportedLaunchOptions.map(flag => `--${flag}`).join(', ')}. The
                    engine may fail to start.
                </Text>
            )}
            <FormField
                slotLabel="Context mode"
                slotHelp="Auto uses a target/cap of 8192 tokens, a floor of 4096 and one slot unless explicitly changed."
            >
                <Select
                    aria-label="Context mode"
                    value={effectiveContext?.mode ?? 'inherited'}
                    disabled={blocked || resolved?.kvUnified === true}
                    items={[
                        { value: 'inherited', children: 'Inherited' },
                        { value: 'fixed', children: 'Fixed' },
                        { value: 'auto', children: 'Auto' }
                    ]}
                    onValueChange={value => {
                        setValidation(previous => {
                            const next = { ...previous }
                            delete next['auto.floor']
                            delete next['auto.cap']
                            delete next.context
                            return next
                        })
                        const set = { ...load?.set }
                        const clear: Array<keyof RequestedLoad> = (load?.clear ?? []).filter(
                            field => field !== 'context'
                        )
                        if (value === 'inherited') {
                            delete set.context
                            clear.push('context')
                        } else if (value === 'auto')
                            set.context = { mode: 'auto', floor: 4096, cap: 8192 }
                        else set.context = { mode: 'fixed', tokens: 8192 }
                        setValues(previous => {
                            const next = { ...previous }
                            delete next.context
                            delete next['auto.floor']
                            delete next['auto.cap']
                            return next
                        })
                        edit({ load: { set, clear } })
                    }}
                />
            </FormField>
            {effectiveContext?.mode === 'auto' &&
                (['floor', 'cap'] satisfies Array<'floor' | 'cap'>).map(field => (
                    <FormField key={field} slotLabel={`Auto context ${field}`}>
                        <TextInput
                            aria-label={`Auto context ${field}`}
                            type="number"
                            value={
                                values[`auto.${field}`] ??
                                String(
                                    effectiveContext?.[field] ?? (field === 'floor' ? 4096 : 8192)
                                )
                            }
                            disabled={blocked}
                            onValueChange={value => {
                                setValues(previous => ({ ...previous, [`auto.${field}`]: value }))
                                const number = Number(value)
                                if (!value || !Number.isSafeInteger(number) || number <= 0) {
                                    validateField(
                                        `auto.${field}`,
                                        `Auto ${field}: enter a positive integer.`
                                    )
                                    useEngineSettingsStore.getState().invalidateModel(key)
                                    return
                                }
                                validateField(`auto.${field}`)
                                const context = load?.set.context ?? requested.context
                                edit({
                                    load: {
                                        set: {
                                            ...load?.set,
                                            context: {
                                                mode: 'auto',
                                                floor: context?.floor ?? 4096,
                                                cap: context?.cap ?? 8192,
                                                [field]: number,
                                                ...(field === 'cap' &&
                                                number < 4096 &&
                                                (context?.floor ?? 4096) === 4096
                                                    ? { floor: Math.floor(number / 256) * 256 }
                                                    : {})
                                            }
                                        },
                                        clear: load?.clear?.filter(item => item !== 'context')
                                    }
                                })
                            }}
                        />
                    </FormField>
                ))}
            <Text kind="body/regular/sm">
                Saved requested overrides: {JSON.stringify(requested)}. Unset leaves remain
                inherited.
            </Text>
            {resolved?.kvUnified === true && (
                <Text kind="body/regular/sm">
                    Existing unified KV allocation is preserved. Typed load changes are unavailable
                    for this allocation.
                </Text>
            )}
            <Text kind="body/regular/sm">
                Saved native total allocation: {resolved?.totalContext ?? 'Unknown'} tokens · Per
                request: {resolved?.perSlotContext ?? 'Unknown'} · Slots:{' '}
                {resolved?.slots ?? 'Unknown'}
            </Text>
            <Text kind="body/semibold/sm">Routing/residency</Text>
            <Text kind="body/regular/sm">
                Served permission: {routing.served} (
                {routing.served === 'unrestricted'
                    ? 'omitted/null allows all'
                    : 'explicit list, empty allows none'}
                ). Changes require explicit target review. No qualifications are created.
            </Text>
            {['weak', 'medium', 'strong'].map(tier => (
                <Text key={tier} kind="body/regular/sm">
                    {tier} role currently:{' '}
                    {routing.profileKey &&
                    routing.roles.some(
                        role =>
                            role.name === tier && role.profiles.includes(routing.profileKey ?? '')
                    )
                        ? 'Assigned'
                        : 'Unassigned'}
                    . Assigning adds this profile to this tier.
                </Text>
            ))}
            {(
                ['weak', 'medium', 'strong'] satisfies Array<
                    keyof NonNullable<NonNullable<ModelChange['policy']>['roles']>
                >
            ).map(tier => (
                <FormField key={tier} slotLabel={`${tier} role`}>
                    <Select
                        aria-label={`${tier} role`}
                        value={
                            draft?.change?.policy?.roles?.[tier] === undefined
                                ? ''
                                : String(draft.change.policy.roles[tier])
                        }
                        disabled={blocked}
                        items={[
                            { value: '', children: 'Unchanged' },
                            { value: 'true', children: 'Assign, replacing target tier' },
                            { value: 'false', children: 'Remove this model from tier' }
                        ]}
                        onValueChange={value => {
                            const roles = { ...draft?.change?.policy?.roles }
                            if (!value) delete roles[tier]
                            else roles[tier] = value === 'true'
                            edit({ policy: { ...draft?.change?.policy, roles } })
                        }}
                    />
                </FormField>
            ))}
            {(
                ['served', 'resident', 'binding'] satisfies Array<
                    keyof NonNullable<ModelChange['policy']>
                >
            ).map(field => (
                <FormField
                    key={field}
                    slotLabel={
                        field === 'resident'
                            ? 'Keep loaded'
                            : field === 'served'
                              ? 'Served membership'
                              : 'Engine binding'
                    }
                    slotHelp={
                        field === 'resident'
                            ? 'Requires the target’s existing finite qualified profile and current readiness; no evidence is authored here.'
                            : 'Explicit selected membership only; target preview owns validation.'
                    }
                >
                    <Select
                        aria-label={field}
                        value={
                            draft?.change?.policy?.[field] === undefined
                                ? ''
                                : String(draft.change.policy[field])
                        }
                        disabled={blocked}
                        items={[
                            { value: '', children: 'Unchanged' },
                            ...(!routing.qualified && field === 'resident'
                                ? []
                                : [{ value: 'true', children: 'Include' }]),
                            { value: 'false', children: 'Remove' }
                        ]}
                        onValueChange={value => {
                            const policy = { ...draft?.change?.policy }
                            if (!value) delete policy[field]
                            else policy[field] = value === 'true'
                            edit({ policy })
                        }}
                    />
                </FormField>
            ))}
            <Text kind="body/semibold/sm">Advanced supported</Text>
            {controls(false)}
            {inspection?.options
                ?.filter(
                    option =>
                        !loadFields.some(field => field === option.field) &&
                        !option.field.startsWith('generation.') &&
                        option.field !== 'gpu.layers' &&
                        option.field !== 'cpuExperts.layers'
                )
                .map(option => (
                    <Text key={option.field} kind="body/regular/sm">
                        {option.label}: {option.reason ?? 'Unavailable in this editor'} · Inherited
                        source: {option.defaultSource ?? 'Unknown'}
                    </Text>
                ))}
            {!generation.ok && <Text kind="body/regular/sm">{generation.error.message}</Text>}
            {Object.entries((draft?.review?.preview ?? draft?.advisory)?.errors ?? {}).map(
                ([field, error]) => (
                    <Text key={field} role="alert" kind="body/regular/sm">
                        {field}: {error}
                    </Text>
                )
            )}
            {Object.keys(validation).length > 0 && (
                <Text role="alert" kind="body/regular/sm">
                    {Object.values(validation).join('; ')}
                </Text>
            )}
            {draft?.message && (
                <Text role="status" kind="body/regular/sm">
                    {draft.message}
                </Text>
            )}
            <ModelSettingsPreview
                profileKey={newProfile ? undefined : profileKey}
                inspection={inspection}
                preview={draft?.review?.preview ?? draft?.advisory}
                reviewed={!!draft?.review}
                stale={!!draft?.previewOperation || (!!load && !draft?.review && !draft?.advisory)}
                snapshot={snapshot}
            />
            <Button
                kind="secondary"
                disabled={
                    unavailable ||
                    (!dirty && !autoSelected) ||
                    !generation.ok ||
                    Object.keys(validation).length > 0 ||
                    (!!load && resolved?.kvUnified === true)
                }
                onClick={() => {
                    if (unavailable || !generation.ok) return
                    const selectedPolicy = { ...draft?.change?.policy }
                    delete selectedPolicy.generation
                    delete selectedPolicy.requestOptionsMode
                    const savedAuto =
                        !load && autoSelected
                            ? {
                                  set: {
                                      ...(requested.context?.mode === 'auto'
                                          ? { context: requested.context }
                                          : {}),
                                      ...(requested.gpu?.mode === 'auto'
                                          ? { gpu: requested.gpu }
                                          : {})
                                  }
                              }
                            : undefined
                    void useEngineSettingsStore.getState().previewModel(
                        target,
                        removing
                            ? {
                                  modelId,
                                  profileKey,
                                  remove: true,
                                  expectedPolicyRevision: policyRevision
                              }
                            : {
                                  ...draft?.change,
                                  ...(savedAuto ? { load: savedAuto } : {}),
                                  modelId,
                                  profileKey,
                                  expectedPolicyRevision: policyRevision,
                                  policy: { ...selectedPolicy, ...generation.value }
                              },
                        authority
                    )
                }}
            >
                {autoSelected ? 'Recalculate' : 'Preview model changes'}
            </Button>
            <Button
                disabled={
                    unavailable ||
                    Object.keys(validation).length > 0 ||
                    !draft?.review ||
                    draft.review.authority !== authority
                }
                onClick={() => {
                    if (!unavailable)
                        void useEngineSettingsStore.getState().applyModel(key, authority)
                }}
            >
                Apply reviewed model changes
            </Button>
            {draft?.attempt && (
                <Button
                    kind="secondary"
                    disabled={draft.busy}
                    onClick={() => void useEngineSettingsStore.getState().readModelOutcome(key)}
                >
                    Read operation status
                </Button>
            )}
            {snapshot?.phase === 'failed' &&
                snapshot.requestId &&
                target.nodeId === useConnectionStore.getState().selfId && (
                    <Button
                        kind="secondary"
                        disabled={blocked}
                        onClick={() => edit({ supersedesRequestId: snapshot.requestId })}
                    >
                        Review local supersession of failed operation
                    </Button>
                )}
        </Stack>
    )
}
