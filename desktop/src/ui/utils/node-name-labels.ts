// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NodeItem } from '@/shared/types/nodes'

function shortId(id: string): string {
    return id.slice(0, 8)
}

/** The one comparison for "same shown name": ignores case and surrounding spaces. */
function comparable(name: string): string {
    return name.trim().toLocaleLowerCase()
}

function hardwareLabel(node: NodeItem): string {
    const gpu = node.topology.gpus.find(entry => entry.name)?.name
    if (gpu) return gpu
    if (node.topology.cpu.model) return node.topology.cpu.model
    return `ID ${shortId(node.id)}`
}

/**
 * Labels that tell apart nodes sharing a shown name (compared as
 * `duplicateShownName` does), keyed by node UUID. A node whose name is unique
 * gets no entry. The label is the first GPU name,
 * else the CPU model, else `ID` and the first UUID characters; labels still
 * equal within a group gain the first UUID characters in parentheses.
 */
export function distinguishingLabels(nodes: readonly NodeItem[]): ReadonlyMap<string, string> {
    const groups = new Map<string, NodeItem[]>()
    for (const node of nodes) {
        const key = comparable(node.name)
        if (key) groups.set(key, [...(groups.get(key) ?? []), node])
    }
    const labels = new Map<string, string>()
    for (const group of groups.values()) {
        if (group.length < 2) continue
        const base = new Map(group.map(node => [node.id, hardwareLabel(node)]))
        for (const node of group) {
            const label = base.get(node.id) ?? ''
            const shared = group.some(other => other.id !== node.id && base.get(other.id) === label)
            labels.set(node.id, shared ? `${label} (${shortId(node.id)})` : label)
        }
    }
    return labels
}

/**
 * The other node's shown name that `candidate` matches, ignoring case and
 * surrounding spaces, or null. `others` must exclude the node being renamed.
 */
export function duplicateShownName(candidate: string, others: readonly string[]): string | null {
    const wanted = comparable(candidate)
    if (!wanted) return null
    return others.find(name => comparable(name) === wanted) ?? null
}
