// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react'
import { Flex, Stack, Text } from '@nvidia/foundations-react-core'
import type { NodeItem } from '@/shared/types/nodes'
import { LocalBadge } from '@/ui/components/LocalBadge'

/** Title, one-line description and page actions at the top of a content page. */
export function PageHeading({
    title,
    description,
    actions
}: {
    title: string
    description?: ReactNode
    actions?: ReactNode
}) {
    return (
        <Flex align="end" justify="between" gap="4" wrap="wrap">
            <Stack gap="1" className="min-w-0">
                <Text kind="title/md" role="heading" aria-level={1}>
                    {title}
                </Text>
                {description !== undefined && (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {description}
                    </Text>
                )}
            </Stack>
            {actions !== undefined && (
                <Flex align="center" gap="2" wrap="wrap">
                    {actions}
                </Flex>
            )}
        </Flex>
    )
}

/** One summary figure with a small caption above it. */
export function StatTile({ label, value }: { label: string; value: ReactNode }) {
    return (
        <div className="pair-tile px-4 py-3 min-w-0" role="group" aria-label={label}>
            <span className="pair-eyebrow">{label}</span>
            <Text kind="title/sm" className="mt-1 block truncate">
                {value}
            </Text>
        </div>
    )
}

const MONOGRAM_COLORS = ['#76b900', '#4c9bff', '#b083ff', '#ff9f43', '#2ec4b6', '#ff6b8b']

/** A coloured initial that gives each model or profile a recognisable mark. */
export function Monogram({ name, size = 36 }: { name: string; size?: number }) {
    const letter = (name.match(/[A-Za-z0-9]/)?.[0] ?? '?').toUpperCase()
    let hash = 0
    for (const ch of name) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0
    const color = MONOGRAM_COLORS[hash % MONOGRAM_COLORS.length]
    return (
        <span
            aria-hidden="true"
            className="shrink-0 inline-flex items-center justify-center rounded-lg font-semibold select-none"
            style={{
                width: size,
                height: size,
                fontSize: Math.round(size * 0.44),
                color,
                backgroundColor: `color-mix(in srgb, ${color} 15%, transparent)`,
                boxShadow: `inset 0 0 0 1px color-mix(in srgb, ${color} 32%, transparent)`
            }}
        >
            {letter}
        </span>
    )
}

/** A node name in a small pill with a status dot that dims while the node is offline. */
export function NodeChip({
    node,
    local = false,
    title
}: {
    node: NodeItem
    local?: boolean
    title?: string
}) {
    return (
        <span className="pair-node-chip" data-offline={node.status === 'offline'} title={title}>
            {node.name}
            {local && <LocalBadge />}
        </span>
    )
}
