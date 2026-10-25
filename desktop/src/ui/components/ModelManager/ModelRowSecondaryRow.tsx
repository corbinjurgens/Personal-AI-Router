// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react'
import { Dropdown, Flex, Stack, Text, type DropdownEntry } from '@nvidia/foundations-react-core'
import type { ModelItem } from '@/shared/types/engines'
import type { ModelExpiry } from '@/shared/types/engines'
import { isModelExpiry } from '@/shared/utils/engines'
import { formatBytes } from '@/ui/utils/formatters'
import { IncomingSyncPullRow } from './IncomingSyncPullRow'
import { EXPIRY_OPTIONS, EXPIRY_SHORT_LABELS } from '@/ui/constants/model-row'
import type { ModelRowDropdownControl } from '@/ui/types/model-row'

export function ModelRowSecondaryRow({
    model,
    busyLabel,
    hasExpiry,
    expiryDd,
    onExpiryChange
}: {
    model: ModelItem
    /** Optimistic "working" label (load/eject/delete in flight) shown with a spinner. */
    busyLabel?: string | null
    hasExpiry: boolean
    expiryDd: ModelRowDropdownControl
    onExpiryChange: (modelName: string, value: ModelExpiry) => void
}) {
    const expiryItems: DropdownEntry[] = useMemo(
        () =>
            EXPIRY_OPTIONS.map(item => ({
                children: item.children,
                onSelect: () => {
                    if (isModelExpiry(item.id)) onExpiryChange(model.name, item.id)
                }
            })),
        [model.name, onExpiryChange]
    )
    const download = model.downloadState

    return (
        <Flex align="center" justify="between" gap="2" className="min-w-0 ml-4">
            {download || model.downloadStateCurrent !== true || !model.downloaded ? (
                <Stack gap="1" className="min-w-0 grow">
                    {(download || model.downloadStateCurrent !== true) && (
                        <IncomingSyncPullRow
                            row={{
                                rawModel: model.name,
                                label: model.name,
                                status: download?.phase ?? 'unknown',
                                current: model.downloadStateCurrent,
                                percent: download?.percent,
                                persistence: download?.persistence
                            }}
                        />
                    )}
                    {!model.downloaded &&
                        model.downloadStateCurrent === true &&
                        (!download || download.phase === 'complete') && (
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                Content readiness unknown
                            </Text>
                        )}
                    {busyLabel && (
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            {busyLabel}
                        </Text>
                    )}
                </Stack>
            ) : busyLabel ? (
                <Flex align="center" className="min-w-0 shrink">
                    <Flex align="center" gap="2" className="shrink-0 whitespace-nowrap">
                        <span
                            className="spinner-element"
                            role="status"
                            aria-label=""
                            style={{ margin: '0' }}
                        />
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            {busyLabel}
                        </Text>
                    </Flex>
                </Flex>
            ) : model.size > 0 ? (
                <Text
                    kind="body/regular/sm"
                    className="shrink-0 text-subtle-color whitespace-nowrap"
                >
                    {formatBytes(model.size)}
                </Text>
            ) : (
                <span />
            )}
            <Flex align="center" gap="1" className="shrink-0">
                {hasExpiry && (
                    <Dropdown
                        items={expiryItems}
                        size="small"
                        className="compact-dropdown"
                        onOpenChange={expiryDd.onOpenChange}
                        attributes={{
                            DropdownContent: {
                                className: expiryDd.dropDownContentClassName
                            }
                        }}
                    >
                        <Text
                            kind="body/regular/sm"
                            className="text-subtle-color truncate max-w-[7rem]"
                        >
                            {`Expires ${EXPIRY_SHORT_LABELS[model.expiry ?? '10m']}`}
                        </Text>
                    </Dropdown>
                )}
            </Flex>
        </Flex>
    )
}
