// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { BuiltinEngineType, EngineDescriptor, EngineType } from '@/shared/types/engines'
import { engineTypeFromManagerName, isBuiltinEngineType } from '@/shared/utils/engines'
import type { EngineCaps } from '@/ui/types/engine-manifest'

/**
 * Curated capabilities of the built-in engines. Other engines derive theirs
 * from their descriptor through {@link engineCapabilities}.
 */
export const EngineCapabilities: Record<BuiltinEngineType, EngineCaps> = {
    ollama: {
        // Ollama has no model keep-alive/expiry UI; unload uses unload_model
        // (POST /api/generate with keep_alive: 0). See docs/services-parity.md#models.
        hasExpiry: false,
        hasLoad: true,
        // Ollama unloads via POST /api/generate with keep_alive: 0 (unload_model).
        hasEject: true,
        hasInstall: ['win32', 'darwin', 'linux'],
        hasEnginePort: true,
        hasInstallPath: false,
        hasProxyWebUI: false,
        hasPreferredNode: false,
        hasCrashAlert: false,
        hasModelSearchOnlyWhenRunning: true,
        modelOpsWhenStopped: false,
        hasDeleteModel: true,
        hasPull: true,
        hasModelCopy: true,
        engineHub: { label: 'Ollama', url: 'https://ollama.com/library' }
    },
    'lm-studio': {
        hasExpiry: false,
        hasLoad: true,
        // LM Studio's `unload_model` action (`lms unload`) is a real eject
        // path. The backend reports which models are loaded in memory, so
        // ModelRow.tsx offers Eject only for a model whose `status` is
        // `'loaded'`.
        hasEject: true,
        hasInstall: ['win32', 'darwin', 'linux'],
        hasEnginePort: true,
        hasInstallPath: false,
        hasProxyWebUI: false,
        hasPreferredNode: false,
        hasCrashAlert: false,
        hasModelSearchOnlyWhenRunning: true,
        modelOpsWhenStopped: false,
        hasDeleteModel: true,
        // LM Studio answers /v1/models from an index it builds at startup and
        // exposes no rescan, so nvpair-engine-manager's delete_model restarts the
        // server. Deleting therefore interrupts inference and needs a warning.
        restartsOnModelDelete: true,
        hasPull: true,
        hasModelCopy: true,
        engineHub: { label: 'LM Studio', url: 'https://lmstudio.ai/models' }
    },
    'llama-cpp': {
        hasExpiry: false,
        hasLoad: true,
        hasEject: true,
        hasInstall: ['win32', 'darwin', 'linux'],
        hasEnginePort: true,
        hasInstallPath: false,
        hasProxyWebUI: false,
        hasPreferredNode: false,
        hasCrashAlert: false,
        hasModelSearchOnlyWhenRunning: true,
        modelOpsWhenStopped: false,
        // The router removes cached models in-process via DELETE /models.
        hasDeleteModel: true,
        hasPull: true,
        hasModelCopy: true,
        engineHub: {
            label: 'llama.cpp',
            url: 'https://huggingface.co/models?library=gguf'
        }
    }
}

/**
 * Capabilities of an engine this node holds no descriptor for, such as a
 * paired node's engine variant that is not configured here. It offers status
 * and the generic start/stop only: no install, model operation, copy or hub,
 * because nothing says the engine supports them.
 */
export const SAFE_DEFAULT_ENGINE_CAPS: EngineCaps = {
    hasExpiry: false,
    hasLoad: false,
    hasEject: false,
    hasInstall: [],
    hasEnginePort: true,
    hasInstallPath: false,
    hasProxyWebUI: false,
    hasPreferredNode: false,
    hasCrashAlert: false,
    hasModelSearchOnlyWhenRunning: true,
    modelOpsWhenStopped: false,
    hasDeleteModel: false,
    hasPull: false,
    hasModelCopy: false
}

/**
 * An engine's capabilities. Built-in engines keep their curated table. Any
 * other engine offers exactly the model operations its descriptor declares,
 * the catalogue its family selects and a model copy only with a store adapter;
 * PAIR has no install recipe or proxy facade for it, so those stay off. With
 * no descriptor the engine gets {@link SAFE_DEFAULT_ENGINE_CAPS}.
 */
export function engineCapabilities(
    engine: EngineType,
    descriptor: EngineDescriptor | undefined
): EngineCaps {
    if (isBuiltinEngineType(engine)) return EngineCapabilities[engine]
    if (!descriptor) return SAFE_DEFAULT_ENGINE_CAPS
    const catalog = descriptor.catalog ? engineTypeFromManagerName(descriptor.catalog) : null
    const engineHub =
        catalog && isBuiltinEngineType(catalog) ? EngineCapabilities[catalog].engineHub : undefined
    const operations = descriptor.operations
    return {
        ...SAFE_DEFAULT_ENGINE_CAPS,
        hasLoad: operations.load !== undefined,
        hasEject: operations.unload !== undefined,
        hasDeleteModel: operations.delete !== undefined,
        hasPull: operations.pull !== undefined && engineHub !== undefined,
        hasModelCopy: descriptor.modelStore !== '',
        ...(engineHub ? { engineHub } : {})
    }
}
