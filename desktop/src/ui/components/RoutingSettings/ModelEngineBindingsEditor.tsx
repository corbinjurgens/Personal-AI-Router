// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useId, useMemo } from 'react'
import {
    Button,
    Flex,
    FormField,
    Select,
    Stack,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import { useShallow } from 'zustand/react/shallow'
import type { JsonValue } from '@/shared/types/json'
import type { NodeModelReference } from '@/shared/types/node-policy'
import { engineLabel, engineManagerName } from '@/shared/utils/engines'
import { replaceJsonFieldSource } from '@/shared/utils/json-source'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { engineTypesForNode } from '@/ui/utils/get-engines-for-node'

function bindingDraft(text: string) {
    let policy: JsonValue
    try {
        policy = JSON.parse(text)
    } catch {
        return null
    }
    if (policy === null || typeof policy !== 'object' || Array.isArray(policy)) return null
    const value = policy.engineBindings
    const bindings: NodeModelReference[] = []
    if (value !== undefined) {
        if (!Array.isArray(value)) return null
        for (const item of value) {
            if (
                item === null ||
                typeof item !== 'object' ||
                Array.isArray(item) ||
                typeof item.engine !== 'string' ||
                typeof item.model !== 'string'
            )
                return null
            bindings.push({ ...item, engine: item.engine, model: item.model })
        }
    }
    return { policy, bindings }
}

/** Edits only engineBindings in the routing form's existing policy draft. */
export default function ModelEngineBindingsEditor({
    targetId,
    text,
    onTextChange,
    disabled
}: {
    targetId: string
    text: string
    onTextChange: (text: string) => void
    disabled: boolean
}) {
    const listId = useId()
    const draft = useMemo(() => bindingDraft(text), [text])
    const descriptors = useEngineStatusStore(state => state.descriptors)
    const statusByNode = useEngineStatusStore(state => state.statusByNode)
    const inventory = useEngineModelsStore(
        useShallow(state =>
            Array.from(state.models.values()).filter(row => row.nodeId === targetId)
        )
    )
    const engines = useMemo(
        () =>
            engineTypesForNode(targetId, statusByNode, descriptors).map(engine => ({
                value: engineManagerName(engine),
                children: engineLabel(engine, descriptors.get(engine))
            })),
        [targetId, statusByNode, descriptors]
    )
    const readOnly = disabled || draft === null
    const change = (bindings: NodeModelReference[]) => {
        if (readOnly || !draft) return
        onTextChange(replaceJsonFieldSource(text, 'engineBindings', JSON.stringify(bindings)))
    }
    const edit = (index: number, field: 'engine' | 'model', value: string) => {
        if (!draft) return
        change(
            draft.bindings.map((binding, row) =>
                row === index ? { ...binding, [field]: value } : binding
            )
        )
    }

    return (
        <Stack gap="2">
            <Text kind="body/semibold/sm">Model engine bindings</Text>
            <Text kind="body/regular/sm">
                Requests for this model can wait while this node activates its assigned engine.
            </Text>
            <Text kind="body/regular/sm" className="text-subtle-color">
                Use the engine&apos;s native model name, not a tier or alias. Type names for stopped
                engines; observed model suggestions can be incomplete. A saved binding does not mean
                the model is ready. Saving starts nothing. Changes share the policy draft and Save
                button below.
            </Text>
            {draft === null && (
                <Text kind="body/regular/sm" role="status">
                    Fix the policy JSON object and engineBindings list below to edit bindings here.
                    Your policy text is kept.
                </Text>
            )}
            {draft?.bindings.map((binding, index) => {
                const items =
                    !binding.engine || engines.some(engine => engine.value === binding.engine)
                        ? engines
                        : [
                              ...engines,
                              {
                                  value: binding.engine,
                                  children: `${binding.engine} (not reported)`
                              }
                          ]
                const observations = inventory
                    .filter(row => engineManagerName(row.engineType) === binding.engine)
                    .flatMap(row => row.models.map(model => model.name))
                const suggestionsId = `${listId}-${index}`
                return (
                    <Flex key={index} align="end" gap="3" wrap="wrap">
                        <div className="min-w-0 flex-1">
                            <FormField slotLabel={`Engine ${index + 1}`}>
                                <Select
                                    size="small"
                                    attributes={{
                                        SelectTrigger: {
                                            'aria-label': `Binding engine ${index + 1}`
                                        }
                                    }}
                                    items={items}
                                    value={binding.engine}
                                    onValueChange={(value: string) => edit(index, 'engine', value)}
                                    placeholder="Select engine"
                                    disabled={readOnly}
                                />
                            </FormField>
                        </div>
                        <div className="min-w-0 flex-1">
                            <FormField
                                slotLabel={`Native model name ${index + 1}`}
                                slotHelp="Suggestions are observations from this node, not a complete inventory."
                            >
                                <TextInput
                                    size="small"
                                    aria-label={`Binding model ${index + 1}`}
                                    value={binding.model}
                                    onValueChange={(value: string) => edit(index, 'model', value)}
                                    list={suggestionsId}
                                    disabled={readOnly}
                                />
                                <datalist id={suggestionsId}>
                                    {[...new Set(observations)].map(model => (
                                        <option
                                            key={model}
                                            value={model}
                                            label="Observed on this node"
                                        />
                                    ))}
                                </datalist>
                            </FormField>
                        </div>
                        <Button
                            kind="secondary"
                            size="small"
                            aria-label={`Remove binding ${index + 1}`}
                            disabled={readOnly}
                            onClick={() => change(draft.bindings.filter((_, row) => row !== index))}
                        >
                            Remove
                        </Button>
                    </Flex>
                )
            })}
            {draft?.bindings.length === 0 && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    No model engine bindings in this draft.
                </Text>
            )}
            <Flex gap="2" wrap="wrap">
                <Button
                    kind="secondary"
                    size="small"
                    disabled={readOnly}
                    onClick={() => change([...(draft?.bindings ?? []), { engine: '', model: '' }])}
                >
                    Add binding
                </Button>
                <Button
                    kind="tertiary"
                    size="small"
                    disabled={readOnly || draft?.bindings.length === 0}
                    onClick={() => change([])}
                >
                    Clear bindings
                </Button>
            </Flex>
            <Text kind="body/regular/sm" className="text-subtle-color">
                The node can refuse edits to a binding reserved by an automatic switch as busy.
                Existing engine switch controls show activation phases and errors. Locks and pause
                still apply; unlock or Resume on the target node.
            </Text>
        </Stack>
    )
}
