// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatModelDisplayName } from './format-model-display-name'

/**
 * Display facts read from a catalog row's name and Hugging Face tags. These
 * are labels for choosing a model, never inputs to a download: the pull name
 * stays the row's own `name`.
 */
interface HubModelFacts {
    /** `owner/repo`, the part of the name before the quantization. */
    repo: string
    /** The repository's display name, from `formatModelDisplayName`. */
    title: string
    /** The quantization the catalog offers, when the name carries one. */
    quantization?: string
    license?: string
    baseModel?: string
    /** Total parameters such as "35B": the catalog's size, else the repository name's. */
    parameters?: string
    /** Active parameters of a mixture-of-experts model such as "3B", from the repository name. */
    activeParameters?: string
    languages: string[]
    /** Readable capability labels such as "Vision" or "Reasoning". */
    capabilities: string[]
}

const LICENSE_NAMES: Record<string, string> = {
    'apache-2.0': 'Apache 2.0',
    mit: 'MIT',
    'cc-by-4.0': 'CC BY 4.0',
    'cc-by-nc-4.0': 'CC BY-NC 4.0',
    'cc-by-sa-4.0': 'CC BY-SA 4.0',
    gemma: 'Gemma',
    llama2: 'Llama 2',
    llama3: 'Llama 3',
    'llama3.1': 'Llama 3.1',
    'llama3.2': 'Llama 3.2',
    'llama3.3': 'Llama 3.3',
    llama4: 'Llama 4',
    openrail: 'OpenRAIL',
    other: 'Custom licence',
    unknown: 'Not stated'
}

const CAPABILITY_TAGS: [string, string][] = [
    ['image-text-to-text', 'Vision'],
    ['vision', 'Vision'],
    ['multimodal', 'Multimodal'],
    ['reasoning', 'Reasoning'],
    ['code', 'Code'],
    ['tool-use', 'Tools'],
    ['function-calling', 'Tools'],
    ['conversational', 'Chat'],
    ['imatrix', 'imatrix']
]

const PARAMETERS = /(?:^|[-_ .])(\d+(?:\.\d+)?)B(?:-A(\d+(?:\.\d+)?)B)?(?=$|[-_ .])/i

function tagValue(tags: readonly string[], prefix: string): string[] {
    return tags
        .filter(tag => tag.toLowerCase().startsWith(prefix))
        .map(tag => tag.slice(prefix.length).trim())
        .filter(value => value !== '')
}

/** `parameterSize` is the catalog's own label; the name is read only where it is absent. */
export function hubModelFacts(
    name: string,
    tags: readonly string[] = [],
    parameterSize?: string
): HubModelFacts {
    const colon = name.lastIndexOf(':')
    const repo = colon > 0 ? name.slice(0, colon) : name
    const quantization = colon > 0 ? name.slice(colon + 1) : undefined
    const slash = repo.indexOf('/')
    const title = formatModelDisplayName(repo)
    const licenseId = tagValue(tags, 'license:')[0]?.toLowerCase()
    // `base_model:quantized:owner/name` and plain `base_model:owner/name` both
    // name the original; the relation prefix is dropped.
    const baseModel = tagValue(tags, 'base_model:')
        .map(value => value.slice(value.lastIndexOf(':') + 1))
        .find(value => value.includes('/'))
    const parameters = (slash >= 0 ? repo.slice(slash + 1) : repo).match(PARAMETERS)
    const lower = new Set(tags.map(tag => tag.toLowerCase()))
    const capabilities = [
        ...new Set(CAPABILITY_TAGS.filter(([tag]) => lower.has(tag)).map(([, label]) => label))
    ]
    if (parameters?.[2] && !capabilities.includes('Mixture of experts'))
        capabilities.unshift('Mixture of experts')
    return {
        repo,
        title,
        quantization,
        license: licenseId ? (LICENSE_NAMES[licenseId] ?? licenseId) : undefined,
        baseModel,
        parameters: parameterSize || (parameters ? `${parameters[1]}B` : undefined),
        activeParameters: parameters?.[2] ? `${parameters[2]}B` : undefined,
        languages: tags.filter(tag => /^[a-z]{2}$/.test(tag)),
        capabilities
    }
}
