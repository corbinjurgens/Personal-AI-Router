// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Checkbox, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { EngineManagerName, EngineType, ModelItem } from '@/shared/types/engines'
import type {
    InspectModelResult,
    ModelSettingsIdentity,
    RequestedLoad
} from '@/shared/types/model-config'
import type { ModelCopyOutcome } from '@/shared/types/model-copy'
import type { NodePolicyDocument } from '@/shared/types/node-policy'
import {
    defaultModelCopyFields,
    defaultPortableLoadFields,
    deployModelPolicyPatch,
    portableLoadFields,
    portableLoadPatch,
    profileDeployMatch
} from '@/shared/utils/node-policy-model-copy'
import {
    MODEL_OPTION_FIELDS,
    newProfileKeyStatus,
    parseModelSettings
} from '@/shared/utils/node-policy-model-settings'
import { EngineManagerNames } from '@/shared/constants/engines'
import { POLICY_CONFLICT_CODE } from '@/shared/constants/service-errors'
import { engineTypeFromManagerName } from '@/shared/utils/engines'
import getErrorString from '@/shared/utils/get-error-string'
import { serviceErrorCode } from '@/ui/api/service-transport'
import { ModelCopyModal } from '@/ui/components/ModelManager/ModelCopyModal'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import ModelSettingsPreview from './ModelSettingsPreview'
import {
    engineSettingsKey,
    modelSettingsKey,
    useEngineSettingsStore,
    watchEngineSettings
} from '@/ui/stores/engine-settings.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import {
    nodePaused,
    policyRoleProfiles,
    policyRoles,
    useNodeConfigurationStore
} from '@/ui/stores/node-configuration.store'

interface Target {
    id: string
    nodeId?: string
    name: string
}
interface Selection {
    fields: string[]
    mode: boolean
    loads: Array<keyof RequestedLoad>
}
type CopyProps = {
    identity: ModelSettingsIdentity
    policy: string
    targetId: string
    disabled?: boolean
    onConfigureTarget: (target: {
        targetId: string
        nodeId?: string
        nodeName: string
        identity: ModelSettingsIdentity
    }) => void
}

function profile(text: string, identity: ModelSettingsIdentity) {
    const parsed = parseModelSettings(text)
    return parsed.ok
        ? parsed.value.find(
              item => item.engine === identity.engine && item.key === identity.profileKey
          )
        : undefined
}

export default function ModelSettingsCopy(props: CopyProps) {
    const { identity, policy, targetId, disabled = false } = props
    const selfId = useConnectionStore(state => state.selfId)
    const nodes = useNodesStore(state => state.nodes)
    const inventory = useEngineModelsStore(state => state.models)
    const settings = useEngineSettingsStore(state => state.entries)
    const drafts = useEngineSettingsStore(state => state.modelDrafts)
    const engine = Object.values(EngineManagerNames).find(value => value === identity.engine)
    const engineType = engineTypeFromManagerName(identity.engine)
    const sourceTarget = useMemo(
        () => (engine ? { nodeId: targetId, engine } : undefined),
        [targetId, engine]
    )
    const defaults = useMemo(
        () => defaultModelCopyFields(policy, identity, identity),
        [policy, identity]
    )
    const [fields, setFields] = useState<string[]>(() => (defaults.ok ? defaults.value : []))
    const [mode, setMode] = useState(false)
    const [loads, setLoads] = useState<Array<keyof RequestedLoad>>([])
    const initializedFields = useRef(defaults.ok)
    const initializedLoads = useRef(false)
    const queue = useRef<Promise<void>>(Promise.resolve())
    const serializePreview = useCallback((run: () => Promise<void>) => {
        const next = queue.current.then(run, run)
        queue.current = next.catch(() => {})
        return next
    }, [])
    const sourceLoad = sourceTarget
        ? settings[engineSettingsKey(sourceTarget)]?.snapshot?.settings.modelLoads?.find(
              load => load.profileKey === identity.profileKey
          )
        : undefined
    const sourceInspection = sourceTarget
        ? drafts[modelSettingsKey(sourceTarget, sourceLoad?.profileKey)]?.inspection
        : undefined
    const inspection = sourceInspection?.modelId === identity.model ? sourceInspection : undefined
    const requested = sourceLoad?.requested ?? {}
    useEffect(() => {
        if (!initializedFields.current && defaults.ok) {
            initializedFields.current = true
            setFields(defaults.value)
        }
    }, [defaults])
    useEffect(() => {
        if (!sourceTarget || disabled) return
        const stop = watchEngineSettings()
        const store = useEngineSettingsStore.getState()
        let active = true
        void store.reload(sourceTarget).then(() => {
            if (!active) return
            const saved = useEngineSettingsStore
                .getState()
                .entries[
                    engineSettingsKey(sourceTarget)
                ]?.snapshot?.settings.modelLoads?.find(load => load.profileKey === identity.profileKey)
            void store.inspectModel(sourceTarget, identity.model, saved?.profileKey)
        })
        return () => {
            active = false
            stop()
        }
    }, [sourceTarget, identity.model, identity.profileKey, disabled])
    useEffect(() => {
        if (!initializedLoads.current && inspection && Object.keys(requested).length) {
            initializedLoads.current = true
            setLoads(defaultPortableLoadFields(inspection, requested))
        }
    }, [inspection, requested])
    const source = profile(policy, identity)
    const sourceIdentity =
        inventory
            .get(`${targetId}:${engineType}`)
            ?.models.find(item => item.name === identity.model)?.identity ??
        policyRoleProfiles(policy).find(
            item => item.key === identity.profileKey && item.engine === identity.engine
        )?.modelIdentity
    const selection = { fields, mode, loads }
    const sourceVersion = JSON.stringify([
        identity,
        policy,
        requested,
        inspection?.status,
        inspection?.artifacts,
        inspection?.options,
        sourceIdentity,
        selection
    ])
    const targets: Target[] = [
        ...(selfId && selfId !== targetId ? [{ id: selfId, name: 'This PC' }] : []),
        ...Array.from(nodes.values())
            .filter(node => node.id !== targetId && node.id !== selfId)
            .map(node => ({ id: node.id, nodeId: node.id, name: node.name }))
    ]
    return (
        <Stack gap="3">
            <Text kind="body/semibold/sm">Deploy to node</Text>
            <Text kind="body/regular/sm">
                Deploy this profile with an independent file choice, review and confirmation on each
                target. Copied files stay on the target if settings are refused or cancelled.
            </Text>
            {!defaults.ok && <InlineErrorBanner message={defaults.error.message} />}
            {MODEL_OPTION_FIELDS.filter(
                field => field.key !== 'min_p' || identity.engine === 'llamacpp'
            ).map(field => (
                <Checkbox
                    key={field.key}
                    slotLabel={`${field.label}: ${source?.options[field.key] === undefined ? 'Clear on target' : `Set ${JSON.stringify(source.options[field.key])}`}`}
                    checked={fields.includes(field.key)}
                    disabled={disabled}
                    onCheckedChange={value => {
                        initializedFields.current = true
                        setFields(current =>
                            value === true
                                ? [...current, field.key]
                                : current.filter(key => key !== field.key)
                        )
                    }}
                />
            ))}
            <Checkbox
                slotLabel={`Copy request precedence (${source?.mode ?? 'defaults'})`}
                checked={mode}
                disabled={disabled}
                onCheckedChange={value => setMode(value === true)}
            />
            <Text kind="body/regular/sm">
                Portable requested load settings only. Target allocation, memory, bindings and
                approval digests stay target owned.
            </Text>
            {inspection &&
                portableLoadFields(inspection)
                    .filter(field => field !== 'fitTarget')
                    .map(field => (
                        <Checkbox
                            key={field}
                            slotLabel={`Copy load ${field}: ${JSON.stringify(requested[field]) ?? 'Clear on target'}`}
                            checked={loads.includes(field)}
                            disabled={disabled}
                            onCheckedChange={value => {
                                initializedLoads.current = true
                                setLoads(current =>
                                    value === true
                                        ? [...current, field]
                                        : current.filter(item => item !== field)
                                )
                            }}
                        />
                    ))}
            {engine &&
                engineType &&
                targets.map(target => (
                    <DeployTarget
                        key={target.id}
                        {...props}
                        target={target}
                        engine={engine}
                        engineType={engineType}
                        selection={selection}
                        requested={requested}
                        inspection={inspection}
                        sourceVersion={sourceVersion}
                        serializePreview={serializePreview}
                    />
                ))}
        </Stack>
    )
}

function DeployTarget({
    identity,
    policy,
    targetId,
    disabled = false,
    onConfigureTarget,
    target,
    engine,
    engineType,
    selection,
    requested,
    inspection,
    sourceVersion,
    serializePreview
}: CopyProps & {
    target: Target
    engine: EngineManagerName
    engineType: EngineType
    selection: Selection
    requested: RequestedLoad
    inspection?: InspectModelResult
    sourceVersion: string
    serializePreview: (run: () => Promise<void>) => Promise<void>
}) {
    const [selected, setSelected] = useState(false)
    const [key, setKey] = useState(identity.profileKey ?? '')
    const [displayName, setDisplayName] = useState<string>()
    const [icon, setIcon] = useState<string>()
    const [roles, setRoles] = useState<string[]>([])
    const [baseline, setBaseline] = useState<NodePolicyDocument>()
    const [candidate, setCandidate] = useState<ModelItem>()
    const [match, setMatch] = useState<ReturnType<typeof profileDeployMatch>>()
    const [copyOpen, setCopyOpen] = useState(false)
    const [copyContext, setCopyContext] = useState<{
        id: string
        ticket: number
        model: string
        source: string
        sourceIdentity: string
    }>()
    const [working, setWorking] = useState(false)
    const [review, setReview] = useState<{
        modelKey: string
        authority: string
        revision: string
        modelId: string
    }>()
    const [attemptId, setAttemptId] = useState('')
    const attempt = useRef('')
    const revision = useRef(0)
    const inFlight = useRef(false)
    const acceptedApply = useRef<{ id: string; previousRequestId?: string } | undefined>(undefined)
    const copied = useRef<
        | { modelId: string; sourceModel: string; sourceNodeId: string; sourceIdentity: string }
        | undefined
    >(undefined)
    const sourceArtifact = useRef('')
    const reviewedKey = useRef<string | undefined>(undefined)
    const connected = useConnectionStore(state => state.connected)
    const permissions = useClusterInvitationsStore(state => state.remoteManagementContext)
    const management = useNodeConfigurationStore(state => state.management)
    const deployments = usePendingActionsStore(state => state.deployments)
    const drafts = useEngineSettingsStore(state => state.modelDrafts)
    const settings = useEngineSettingsStore(state => state.entries)
    const row = deployments.get(attemptId)
    const modelKey = row?.modelKey ?? review?.modelKey
    const draft = modelKey ? drafts[modelKey] : undefined
    const snapshot = settings[engineSettingsKey({ nodeId: target.id, engine })]?.snapshot
    const invalidate = useCallback(() => {
        revision.current++
        if (inFlight.current) return
        if (reviewedKey.current)
            useEngineSettingsStore.getState().invalidateModel(reviewedKey.current)
        reviewedKey.current = undefined
        setReview(undefined)
    }, [])
    useEffect(() => {
        invalidate()
    }, [sourceVersion, connected, permissions, disabled, invalidate])
    const update = (
        id: string,
        patch: Parameters<ReturnType<typeof usePendingActionsStore.getState>['updateDeploy']>[1]
    ) => {
        usePendingActionsStore.getState().updateDeploy(id, patch)
    }
    useEffect(() => {
        if (
            !attemptId ||
            acceptedApply.current?.id !== attemptId ||
            !draft?.attempt ||
            !modelKey ||
            draft.attempt.requestId === acceptedApply.current.previousRequestId
        )
            return
        const correlated = snapshot?.requestId === draft.attempt.requestId
        const disposition =
            !draft.uncertain && draft.message?.startsWith('Refused.')
                ? 'refused'
                : correlated && snapshot?.phase === 'succeeded' && !draft.uncertain
                  ? 'applied'
                  : correlated && snapshot?.phase === 'failed' && !draft.uncertain
                    ? 'refused'
                    : 'unknown'
        usePendingActionsStore.getState().updateDeploy(attemptId, {
            stage: 'apply',
            disposition,
            message: draft.message ?? 'Outcome unknown; read operation status.'
        })
    }, [attemptId, draft, modelKey, snapshot])
    const authority = (document: NodePolicyDocument, modelId: string) =>
        JSON.stringify([
            target.id,
            engine,
            key,
            modelId,
            document.revision,
            sourceVersion,
            useConnectionStore.getState().connected,
            useNodeConfigurationStore.getState().management.get(target.id),
            useClusterInvitationsStore.getState().remoteManagementContext
        ])
    const refresh = async (readManagement = true) => {
        if (!useConnectionStore.getState().connected) throw new Error('Disconnected')
        if (readManagement)
            await useNodeConfigurationStore
                .getState()
                .readManagement(target.id, target.nodeId, true)
        const permission = useNodeConfigurationStore.getState().management.get(target.id)?.state
        if (permission !== 'local' && permission !== 'allowed')
            throw new Error('Denied: current target management permission required.')
        const document = await window.pairApi.nodePolicy.get(target.nodeId)
        if (target.nodeId && nodePaused(document.policy, document.availability))
            throw new Error('Denied: target is paused; Resume on that device.')
        await useEngineSettingsStore.getState().reload({ nodeId: target.id, engine })
        const entry =
            useEngineSettingsStore.getState().entries[
                engineSettingsKey({ nodeId: target.id, engine })
            ]
        if (!entry?.snapshot || entry.unavailable || !entry.snapshot.editable)
            throw new Error(
                entry?.unavailable ?? entry?.snapshot?.reason ?? 'Target settings unavailable.'
            )
        const inventory = await window.pairApi.engines.getInitialState()
        const models = inventory.models.find(
            item => item.nodeId === target.id && item.engineType === engineType
        )
        if (!models)
            throw new Error('Target inventory unavailable; configure this target in node settings.')
        return { document, models: models.models, snapshot: entry.snapshot, inventory }
    }
    const fail = (id: string, message: string) => update(id, { disposition: 'refused', message })
    const prepare = async (id: string, modelId: string, ticket: number, verifiedMatch = false) =>
        serializePreview(async () => {
            if (attempt.current !== id || revision.current !== ticket) return
            const current = await refresh()
            if (attempt.current !== id || revision.current !== ticket) return
            if (verifiedMatch) {
                const sourceIdentity =
                    current.inventory.models
                        .find(item => item.nodeId === targetId && item.engineType === engineType)
                        ?.models.find(item => item.name === identity.model)?.identity ??
                    policyRoleProfiles(policy).find(
                        item => item.key === identity.profileKey && item.engine === engine
                    )?.modelIdentity
                const targetIdentity = current.models.find(
                    item => item.name === modelId && item.downloaded
                )?.identity
                if (profileDeployMatch(sourceIdentity, targetIdentity).strength !== 'verified')
                    throw new Error('Artifact correspondence changed. Review again.')
            }
            setBaseline(current.document)
            const targetIdentity = { engine, model: modelId, profileKey: key }
            const parsed = parseModelSettings(current.document.policy)
            if (!parsed.ok) throw new Error(parsed.error.message)
            const existing = parsed.value.find(item => item.key === key)
            const targetLoad = current.snapshot.settings.modelLoads?.find(
                load => load.profileKey === key
            )
            const artifact = targetLoad?.modelId ?? existing?.model
            const keyStatus = newProfileKeyStatus(current.document.policy, key)
            if (keyStatus === 'invalid') throw new Error('Enter a valid profile key.')
            if (
                keyStatus === 'duplicate' &&
                (!existing || existing.engine !== engine || artifact !== modelId)
            )
                throw new Error(
                    'Key clash: this key belongs to a different model. Enter a new profile key.'
                )
            const selectedRoles = policyRoles(current.document.policy)
                .filter(role => roles.includes(role.name))
                .map(role => role.name)
            const patch = deployModelPolicyPatch(
                policy,
                identity,
                current.document.policy,
                targetIdentity,
                { fields: selection.fields, mode: selection.mode },
                target.nodeId !== undefined && existing?.localOnly === true,
                {
                    ...(displayName !== undefined ? { displayName } : {}),
                    ...(icon !== undefined ? { icon } : {})
                },
                selectedRoles
            )
            if (!patch.ok) throw new Error(patch.error.message)
            const context = authority(current.document, modelId)
            const settingsTarget = { nodeId: target.id, engine }
            const nextKey = modelSettingsKey(settingsTarget, key)
            const load = selection.loads.length
                ? portableLoadPatch(requested, selection.loads, inspection)
                : !existing && !targetLoad
                  ? portableLoadPatch({}, ['context'])
                  : undefined
            update(id, {
                stage: 'review',
                disposition: undefined,
                modelKey: nextKey,
                message: existing ? 'Review profile update.' : 'Review new profile.'
            })
            await useEngineSettingsStore.getState().previewModel(
                settingsTarget,
                {
                    modelId,
                    profileKey: key,
                    expectedPolicyRevision: current.document.revision,
                    ...(load ? { load } : {}),
                    policy: patch.value
                },
                context
            )
            if (attempt.current !== id || revision.current !== ticket) {
                useEngineSettingsStore.getState().invalidateModel(nextKey)
                return
            }
            const result = useEngineSettingsStore.getState().modelDrafts[nextKey]
            if (!result?.review) throw new Error(result?.message ?? 'Target review unavailable.')
            reviewedKey.current = nextKey
            setReview({
                modelKey: nextKey,
                authority: context,
                revision: current.document.revision,
                modelId
            })
        })
    const start = async () => {
        if (inFlight.current || working || disabled || draft?.uncertain) return
        const id = crypto.randomUUID()
        acceptedApply.current = undefined
        attempt.current = id
        setAttemptId(id)
        const ticket = ++revision.current
        setReview(undefined)
        setWorking(true)
        usePendingActionsStore
            .getState()
            .beginDeploy(id, { nodeId: target.id, engineType, profileKey: key })
        try {
            const current = await refresh()
            if (attempt.current !== id || revision.current !== ticket) return
            setBaseline(current.document)
            const sourceIdentity =
                current.inventory.models
                    .find(item => item.nodeId === targetId && item.engineType === engineType)
                    ?.models.find(item => item.name === identity.model)?.identity ??
                policyRoleProfiles(policy).find(
                    item => item.key === identity.profileKey && item.engine === engine
                )?.modelIdentity
            sourceArtifact.current = JSON.stringify(sourceIdentity) ?? ''
            if (
                copied.current &&
                (copied.current.sourceModel !== identity.model ||
                    copied.current.sourceNodeId !== targetId ||
                    copied.current.sourceIdentity !== sourceArtifact.current)
            )
                copied.current = undefined
            const copiedModel = copied.current?.modelId
            const candidates = current.models.filter(item => item.downloaded)
            const found = copiedModel
                ? candidates.find(item => item.name === copiedModel)
                : (candidates.find(
                      item =>
                          profileDeployMatch(sourceIdentity, item.identity).strength === 'verified'
                  ) ??
                  candidates.find(item =>
                      ['companions-unverified', 'possible'].includes(
                          profileDeployMatch(sourceIdentity, item.identity).strength
                      )
                  ))
            const correspondence = profileDeployMatch(sourceIdentity, found?.identity)
            setCandidate(found)
            setMatch(correspondence)
            if (copiedModel || correspondence.strength === 'verified') {
                const modelId = copiedModel ?? found?.name
                if (!modelId) throw new Error('Verified target model unavailable.')
                await prepare(id, modelId, ticket, !copiedModel)
            } else if (correspondence.strength === 'different') {
                update(id, { stage: 'copy', message: 'Copy model files before target review.' })
                setCopyContext({
                    id,
                    ticket,
                    model: identity.model,
                    source: targetId,
                    sourceIdentity: sourceArtifact.current
                })
                setCopyOpen(true)
            } else
                update(id, {
                    stage: 'review',
                    message: 'Choose whether to use target files or copy. Nothing is selected.'
                })
        } catch (error) {
            fail(id, getErrorString(error))
        } finally {
            setWorking(false)
        }
    }
    const chooseExisting = async () => {
        if (!candidate) return
        setWorking(true)
        try {
            await prepare(attempt.current, candidate.name, revision.current)
        } catch (error) {
            fail(attempt.current, getErrorString(error))
        } finally {
            setWorking(false)
        }
    }
    const copyOutcome = async (outcome: ModelCopyOutcome) => {
        if (!copyContext || attempt.current !== copyContext.id) return
        const id = copyContext.id
        if (
            outcome.sourceNodeId !== copyContext.source ||
            outcome.destinationNodeId !== target.id ||
            outcome.engine !== engine ||
            outcome.model !== copyContext.model
        )
            return
        update(id, {
            stage: 'copy',
            copyOpId: outcome.opId,
            ...(outcome.disposition !== 'copied' ? { disposition: outcome.disposition } : {}),
            message: outcome.reason ?? outcome.disposition
        })
        if (outcome.disposition !== 'copied') return
        if (!outcome.result) {
            update(id, {
                disposition: 'unknown',
                message: 'Copy completion has no verified result. Read copy status.'
            })
            return
        }
        copied.current = {
            modelId: outcome.result.model,
            sourceModel: copyContext.model,
            sourceNodeId: copyContext.source,
            sourceIdentity: copyContext.sourceIdentity
        }
        setCopyOpen(false)
        if (revision.current !== copyContext.ticket) {
            update(id, {
                stage: 'review',
                disposition: undefined,
                message:
                    'Files copied. Source or selection changed; Review again before deploying settings.'
            })
            return
        }
        setWorking(true)
        try {
            await prepare(id, outcome.result.model, copyContext.ticket)
        } catch (error) {
            fail(id, getErrorString(error))
        } finally {
            setWorking(false)
        }
    }
    const confirm = async () => {
        if (!review || inFlight.current || disabled) return
        const accepted = review
        const id = attempt.current
        const ticket = revision.current
        setWorking(true)
        try {
            // applyModel rechecks the target's grant immediately before dispatch.
            // Publishing a transient management loading state here would invalidate
            // the very review being confirmed through the shared settings watcher.
            const current = await refresh(false)
            if (
                ticket !== revision.current ||
                authority(current.document, accepted.modelId) !== accepted.authority ||
                current.document.revision !== accepted.revision
            ) {
                setReview(undefined)
                fail(id, 'Refused: policy or authority changed. Review again.')
                return
            }
            inFlight.current = true
            acceptedApply.current = {
                id,
                previousRequestId:
                    useEngineSettingsStore.getState().modelDrafts[accepted.modelKey]?.attempt
                        ?.requestId
            }
            update(id, { stage: 'apply', message: 'Applying reviewed profile.' })
            await useEngineSettingsStore
                .getState()
                .applyModel(accepted.modelKey, accepted.authority)
            const result = useEngineSettingsStore.getState().modelDrafts[accepted.modelKey]
            if (
                !result?.attempt ||
                result.attempt.requestId === acceptedApply.current?.previousRequestId ||
                (!result.uncertain && result.message?.startsWith('Refused.'))
            )
                fail(id, result?.message ?? 'Apply refused; Review again.')
            setReview(undefined)
        } catch (error) {
            fail(
                id,
                `${serviceErrorCode(error) === POLICY_CONFLICT_CODE ? 'Refused: policy revision conflict. Review again. ' : ''}${getErrorString(error)}`
            )
        } finally {
            inFlight.current = false
            setWorking(false)
        }
    }
    const controlsLocked =
        disabled ||
        working ||
        !!draft?.uncertain ||
        inFlight.current ||
        (row?.stage === 'copy' && (!row.disposition || row.disposition === 'unknown'))
    const invalidKey = newProfileKeyStatus(baseline?.policy ?? policy, key) === 'invalid'
    return (
        <Stack gap="2" role="group" aria-label={`Deploy ${target.name}`}>
            <Checkbox
                slotLabel={`Deploy to ${target.name}`}
                checked={selected}
                disabled={disabled}
                onCheckedChange={value => {
                    setSelected(value === true)
                    invalidate()
                }}
            />
            {selected && (
                <>
                    <FormField slotLabel={`Profile key on ${target.name}`}>
                        <TextInput
                            aria-label={`Profile key ${target.id}`}
                            value={key}
                            disabled={controlsLocked}
                            onValueChange={value => {
                                invalidate()
                                setKey(value)
                            }}
                        />
                    </FormField>
                    {invalidKey && <Text kind="body/regular/sm">Enter a valid profile key.</Text>}
                    <FormField
                        slotLabel={`Display name on ${target.name}`}
                        slotHelp="Leave unchanged to keep target metadata on updates; new profiles use source defaults."
                    >
                        <TextInput
                            aria-label={`Display name ${target.id}`}
                            value={displayName ?? ''}
                            disabled={controlsLocked}
                            onValueChange={value => {
                                invalidate()
                                setDisplayName(value)
                            }}
                        />
                    </FormField>
                    <FormField slotLabel={`Icon on ${target.name}`}>
                        <TextInput
                            aria-label={`Icon ${target.id}`}
                            value={icon ?? ''}
                            disabled={controlsLocked}
                            onValueChange={value => {
                                invalidate()
                                setIcon(value)
                            }}
                        />
                    </FormField>
                    {baseline &&
                        (policyRoles(baseline.policy).length ? (
                            policyRoles(baseline.policy).map(role => (
                                <Checkbox
                                    key={role.name}
                                    slotLabel={`${role.name} on ${target.name}`}
                                    aria-label={`${role.name} on ${target.name}`}
                                    title={role.name}
                                    className="break-all"
                                    checked={roles.includes(role.name)}
                                    disabled={controlsLocked}
                                    onCheckedChange={value => {
                                        invalidate()
                                        setRoles(current =>
                                            value === true
                                                ? [...current, role.name]
                                                : current.filter(item => item !== role.name)
                                        )
                                    }}
                                />
                            ))
                        ) : (
                            <Text kind="body/regular/sm">
                                No roles on {target.name}; create one in node settings.
                            </Text>
                        ))}
                    {match?.strength === 'companions-unverified' && (
                        <Text kind="body/regular/sm">
                            Model file matches; {match.missingCompanions.join(', ')} not verified
                        </Text>
                    )}
                    {match?.strength === 'possible' && (
                        <Text kind="body/regular/sm">Possibly the same file (not verified)</Text>
                    )}
                    {match &&
                        ['possible', 'companions-unverified'].includes(match.strength) &&
                        !review &&
                        !row?.disposition && (
                            <>
                                <Button
                                    disabled={controlsLocked}
                                    onClick={() => void chooseExisting()}
                                >
                                    Use the {match.strength === 'possible' ? 'file' : 'files'} on{' '}
                                    {target.name}
                                </Button>
                                <Button
                                    disabled={controlsLocked}
                                    onClick={() => {
                                        update(attempt.current, {
                                            stage: 'copy',
                                            message: 'Copying model files.'
                                        })
                                        setCopyContext({
                                            id: attempt.current,
                                            ticket: revision.current,
                                            model: identity.model,
                                            source: targetId,
                                            sourceIdentity: sourceArtifact.current
                                        })
                                        setCopyOpen(true)
                                    }}
                                >
                                    Copy
                                </Button>
                            </>
                        )}
                    {row && (
                        <Text role="status" kind="body/regular/sm">
                            {target.name}: {row.stage === 'review' ? 'target review' : row.stage}
                            {row.disposition ? `: ${row.disposition}` : ''}. {row.message}
                        </Text>
                    )}
                    {(review?.modelId ?? candidate?.name) && (
                        <Text kind="body/regular/sm">
                            Target model: {review?.modelId ?? candidate?.name}
                        </Text>
                    )}
                    {modelKey && draft && (
                        <ModelSettingsPreview
                            profileKey={key}
                            inspection={draft.inspection}
                            preview={draft.review?.preview ?? draft.advisory}
                            reviewed={!!draft.review}
                            stale={!draft.review}
                            snapshot={snapshot}
                        />
                    )}
                    {review && (
                        <Button
                            disabled={
                                disabled ||
                                working ||
                                !connected ||
                                !draft?.review ||
                                management.get(target.id)?.state === 'denied'
                            }
                            onClick={() => void confirm()}
                        >
                            Confirm deploy to {target.name}
                        </Button>
                    )}
                    <Button
                        disabled={controlsLocked || !connected || invalidKey}
                        onClick={() => void start()}
                    >
                        {row ? 'Review again' : `Review deploy to ${target.name}`}
                    </Button>
                    {copyContext && row?.stage === 'copy' && (
                        <Button kind="tertiary" onClick={() => setCopyOpen(true)}>
                            Open copy status on {target.name}
                        </Button>
                    )}
                    {draft?.attempt && (
                        <Button
                            kind="tertiary"
                            onClick={() =>
                                modelKey &&
                                void useEngineSettingsStore.getState().readModelOutcome(modelKey)
                            }
                        >
                            Read deploy operation status on {target.name}
                        </Button>
                    )}
                    <Button
                        kind="tertiary"
                        onClick={() =>
                            onConfigureTarget({
                                targetId: target.id,
                                nodeId: target.nodeId,
                                nodeName: target.name,
                                identity: {
                                    engine,
                                    model: review?.modelId ?? identity.model,
                                    profileKey: key
                                }
                            })
                        }
                    >
                        Configure target {target.name}
                    </Button>
                </>
            )}
            {copyContext && (
                <ModelCopyModal
                    key={copyContext.id}
                    open={copyOpen}
                    onOpenChange={setCopyOpen}
                    engineType={engineType}
                    initialSource={copyContext.source}
                    initialModel={copyContext.model}
                    fixedDestination={target.id}
                    onOutcome={outcome => void copyOutcome(outcome)}
                />
            )}
        </Stack>
    )
}
