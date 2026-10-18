// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BuiltinEngineType, ModelExpiry } from '@/shared/types/engines'

// The engines `nvpair-engine-manager` ships a manifest for, and therefore the
// only ones PAIR can install and route to through a proxy facade. Any other
// engine a node's engine-manager registers (a locally configured fork, for
// example) is known only through its descriptor; see `EngineDescriptor`.
export const EngineTypes = ['ollama', 'lm-studio', 'llama-cpp'] as const

// Kept as a distinct export so a future engine can ship behind it rather than
// appearing the moment its type exists.
export const EnabledEngineTypes: BuiltinEngineType[] = ['ollama', 'lm-studio', 'llama-cpp']

/**
 * How `nvpair-engine-manager` spells each engine on the wire. This is the
 * single translation table — use `engineManagerName()` and
 * `engineTypeFromManagerName()` rather than re-deriving it from a literal.
 */
export const EngineManagerNames = {
    ollama: 'ollama',
    'lm-studio': 'lmstudio',
    'llama-cpp': 'llamacpp'
} as const satisfies Record<BuiltinEngineType, string>

export const EngineSources = ['bundled', 'detected', 'installed'] as const

export const EngineDisplayNames: Record<BuiltinEngineType, string> = {
    ollama: 'Ollama',
    'lm-studio': 'LM Studio',
    'llama-cpp': 'llama.cpp'
} as const

/** Default docs/install URLs for built-in backends. Single source of truth for UI and adapter buildInfo(). */
export const EngineDefaultLinks: Record<
    BuiltinEngineType,
    { docsUrl: string; installUrl: string }
> = {
    ollama: { docsUrl: 'https://docs.ollama.com/', installUrl: 'https://ollama.com/download' },
    'lm-studio': { docsUrl: 'https://lmstudio.ai/docs', installUrl: 'https://lmstudio.ai/' },
    'llama-cpp': {
        docsUrl: 'https://github.com/ggml-org/llama.cpp/tree/master/docs',
        installUrl: 'https://github.com/ggml-org/llama.cpp/releases'
    }
} as const

export const ModelItemStatuses = ['idle', 'loading', 'loaded', 'ejecting', 'pulling'] as const

export const ModelExpiries = ['0', '1s', '10s', '1m', '10m', '-1'] as const

export const ModelExpiryLabels: Record<ModelExpiry, string> = {
    '0': 'Immediately',
    '1s': '1 second',
    '10s': '10 seconds',
    '1m': '1 minute',
    '10m': '10 minutes',
    '-1': 'Never'
} as const

export const EngineProcessStatuses = [
    'running',
    'stopped',
    'not-installed',
    'installing',
    'uninstalling',
    'starting',
    'stopping',
    'initializing'
] as const

/**
 * Engine progress domain -- delivered from the backend as progress push events.
 * Unified progress channel for all operations: install, pull, load, etc.
 * Keyed by ${nodeId}:${engineType}:${operation}:${model?} for concurrent ops.
 */

export const EngineOperationTypes = [
    'install',
    'uninstall',
    'pull',
    'load',
    'unload',
    'delete',
    'copy'
] as const

/**
 * The semantic model operations an engine descriptor can declare
 * (`nvpair-shared/engines.Operations`). An operation absent from a descriptor
 * is unsupported by that engine.
 */
export const EngineDescriptorOperations = ['load', 'unload', 'pull', 'delete'] as const

/** The descriptor shape this build understands (`engines.DescriptorVersion`). */
export const ENGINE_DESCRIPTOR_VERSION = 1

/** Protocol profiles a descriptor may declare (`engines.Protocols`). */
export const EngineDescriptorProtocols = [
    'ollama-api',
    'openai-v1',
    'anthropic-messages',
    'llamacpp-router'
] as const

/** Model naming rules a descriptor may declare (`engines.ModelNamings`). */
export const EngineDescriptorModelNamings = ['exact', 'implied-latest-tag'] as const

/** Upper bound on a declared operation response budget (`engines.MaxOperationResponseTimeoutS`). */
export const ENGINE_OPERATION_MAX_RESPONSE_TIMEOUT_S = 30 * 60
