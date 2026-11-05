// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const HF_PREFIX = /^(https?:\/\/)?(hf\.co|huggingface\.co)\//
const HUB_REGISTRY_ARROW = ' \u2192 '

/**
 * Universal display formatter for model names. Display-only -- never
 * send the return value to the server; always use the raw model id for
 * wire operations.
 *
 * Applies generic HF/hub heuristics only. Does not truncate; callers
 * handle layout constraints themselves.
 */
/** Upper bound on the length of any untrusted name fed to a formatter. Model /
 *  workload names arrive over the wire from cluster peers and engine model
 *  lists; capping length keeps every formatter linear-time on hostile input. */
const MAX_FORMAT_INPUT_CHARS = 256

export function formatModelDisplayName(name: string): string {
    if (!name) {
        return name
    }

    if (name.length > MAX_FORMAT_INPUT_CHARS) {
        name = name.slice(0, MAX_FORMAT_INPUT_CHARS)
    }

    if (name.includes(HUB_REGISTRY_ARROW)) {
        return formatHubRegistryDisplayLabel(name)
    }

    let formatted = formatHuggingFaceModelUrl(name)
    const isHfModel = HF_PREFIX.test(formatted)

    if (isHfModel) {
        formatted = formatHuggingFaceModelName(formatted)
    }

    if (formatted.includes('/')) {
        return formatHuggingFaceModelName(formatted)
    }
    return formatted
}

/**
 * Format a HuggingFace model ID into a readable name.
 * "hf.co/unsloth/Qwen3.5-35B-A3B-GGUF" -> "Qwen3.5 35B A3B"
 * "bartowski/Llama-3.2-1B-Instruct-GGUF" -> "Llama 3.2 1B Instruct"
 */
function formatHuggingFaceModelName(modelId: string): string {
    let name = modelId.replace(HF_PREFIX, '')

    const segments = name.split('/')
    name = segments.length > 1 ? segments.slice(1).join('/') : segments[0]

    const tagAt = name.indexOf(':')
    const tag = tagAt >= 0 ? name.slice(tagAt + 1) : ''
    if (tagAt >= 0) name = name.slice(0, tagAt)

    name = name
        .replace(/-GGUF$/i, '')
        .replace(/[-_]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()

    return name ? `${name}${tag ? ` (${tag})` : ''}` : modelId
}

function formatHuggingFaceModelUrl(modelUrl: string): string {
    return modelUrl.replace(/^(https:\/\/hf\.co|https:\/\/huggingface\.co)\//, '')
}

/**
 * `org/repo -> weight` from on-demand Hub search -- show repo + org only.
 */
function formatHubRegistryDisplayLabel(label: string): string {
    const arrowAt = label.indexOf(HUB_REGISTRY_ARROW)
    if (arrowAt === -1) {
        return label.includes('/') ? formatHuggingFaceModelName(label) : label
    }
    const repoPart = label.slice(0, arrowAt).trim()
    if (!repoPart) {
        return label.includes('/') ? formatHuggingFaceModelName(label) : label
    }
    const segments = repoPart.split('/')
    const author = segments[0] ?? ''
    const repoTail = segments.slice(1).join('/')
    const humanRepo = repoTail
        .replace(/-GGUF$/i, '')
        .replace(/[-_]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
    if (author && humanRepo) return `${humanRepo} (${author})`
    return humanRepo || author || repoPart
}
