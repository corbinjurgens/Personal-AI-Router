// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    EngineDescriptorOperations,
    EngineManagerNames,
    EngineOperationTypes,
    EngineProcessStatuses,
    EngineSources,
    EngineTypes,
    ModelExpiries,
    ModelItemStatuses
} from '@/shared/constants/engines'
import type { ModelCopyResult } from '@/shared/types/model-copy'
import type { ModelDownloadEntry, ModelDownloadSnapshot } from '@/shared/types/model-download'

/**
 * An engine PAIR ships: a closed literal union of the `EngineTypes` tuple.
 * Built-in engines have a proxy facade, curated presentation and an install
 * recipe. Prefer it wherever behavior is specific to those engines, and narrow
 * with `isBuiltinEngineType()`.
 */
export type BuiltinEngineType = (typeof EngineTypes)[number]

declare const descriptorEngineBrand: unique symbol

/**
 * Any other engine a node's `nvpair-engine-manager` registers, such as a
 * locally configured llama.cpp fork. It is spelled exactly as engine-manager
 * spells it, and what PAIR may know about it comes from its descriptor
 * (`EngineDescriptor`). The brand means a value can only come from
 * `isDescriptorEngineType()` / `engineTypeFromManagerName()`, which check the
 * engine-manager ID syntax and refuse the built-in spellings, so an arbitrary
 * string never becomes an engine identity by accident.
 */
export type DescriptorEngineType = string & { readonly [descriptorEngineBrand]: true }

/**
 * Canonical engine identifier inside PAIR: a built-in engine or a
 * descriptor-known one. Strings arriving from service bridge payloads or
 * persisted JSON must be narrowed via `isEngineType()` /
 * `engineTypeFromManagerName()` at the boundary.
 */
export type EngineType = BuiltinEngineType | DescriptorEngineType

/**
 * A built-in engine identifier in `nvpair-engine-manager`'s spelling. Convert
 * at the boundary with `engineManagerName()` / `engineTypeFromManagerName()`;
 * keep `EngineType` everywhere inside PAIR. Descriptor engines are spelled the
 * same way on both sides.
 */
export type EngineManagerName = (typeof EngineManagerNames)[BuiltinEngineType]

/** A semantic model operation an engine descriptor can declare. */
export type EngineDescriptorOperation = (typeof EngineDescriptorOperations)[number]

/**
 * The desktop projection of one `engine:descriptors` entry
 * (`nvpair-shared/engines.Descriptor`), validated at the bridge. It carries
 * identity, presentation and capabilities only; nothing in it can name a
 * binary, command, path or credential.
 */
export interface EngineDescriptor {
    engineType: EngineType
    /** Engine-manager family name (a llama.cpp fork has family `llamacpp`). */
    family: string
    label: string
    /** Whether the publishing host is one of the manifest's platforms. */
    available: boolean
    protocols: string[]
    modelNaming: string
    /** Coded model-store adapter, in engine-manager spelling; empty when none. */
    modelStore: string
    /** Coded download catalogue, in engine-manager spelling; empty when none. */
    catalog: string
    /** Supported model operations. An absent key is an unsupported operation. */
    operations: Partial<Record<EngineDescriptorOperation, { responseTimeoutS: number }>>
}

export type EngineSource = (typeof EngineSources)[number]

export interface EngineStatusData {
    engineType: EngineType
    nodeId: string
    processStatus: EngineProcessStatus
    /**
     * Configured bind port for the engine HTTP server when installed (same value when stopped vs running).
     * Null when not installed, or for an engine with no server port.
     */
    enginePort: number | null
    /** PAIR's proxy port for this engine (e.g. Ollama 54301). Always set when the engine is installed. */
    proxyPort: number | null
    /**
     * Installed engine binary version reported by the owning node, when the
     * current status source exposes it. Undefined for nodes that have not
     * reported version data or engines that are not installed.
     */
    installedVersion?: string
    /**
     * The engine's durable lock on its owning node. Present true or false is
     * authoritative; undefined means lock state is unavailable, as from an
     * older target, and is never read as unlocked.
     */
    locked?: boolean
}

export type ModelItemStatus = (typeof ModelItemStatuses)[number]

export type ModelExpiry = (typeof ModelExpiries)[number]

/** nodeId → engineType → status (sparse: only types reported by the service). */
export type EngineStatusByNode = Map<string, Map<EngineType, EngineStatusData>>

/**
 * Engine status domain -- broadcast per engine per node when process state changes.
 * Replaces the process/port/autoStart portion of the monolithic BackendInfo.
 */

export type EngineProcessStatus = (typeof EngineProcessStatuses)[number]

export type EngineOperationType = (typeof EngineOperationTypes)[number]

export interface EngineProgress {
    engineType: EngineType
    nodeId: string
    nodeName: string
    operation: EngineOperationType
    /** Set for model operations (pull, load, unload, delete). */
    model?: string
    /** 'started' | 'downloading' | 'complete' | 'error' | engine-specific status string. */
    status: string
    percent?: number
    completed?: number
    total?: number
    error?: string
    /** Copies are keyed by the complete tuple and displayed at the destination. */
    opId?: string
    sourceNodeId?: string
    destinationNodeId?: string
    expectedExportId?: string
    exportId?: string
    file?: string
    message?: string
    /** Retains setup/durability qualifications without inventing a model row. */
    copyResult?: ModelCopyResult
    downloadState?: ModelDownloadEntry
    downloadStateCurrent?: boolean
    downloadSourceId?: string
    downloadRevision?: string
}

/**
 * Available engine-update signal for one engine on one node. The backend
 * reports it per node; PAIR receives it through `engines:state-changed` and
 * the renderer's `engine-update-available.store`, so the Edit Node UI can
 * surface an Update affordance for any node — local or remote — the user can
 * reach in the cluster.
 */
export interface EngineUpdateAvailable {
    nodeId: string
    engineType: EngineType
    currentVersion: string
    latestVersion: string
    releaseUrl: string
    installType: 'managed' | 'external'
}

export interface EngineModels {
    engineType: EngineType
    nodeId: string
    models: ModelItem[]
    downloadState?: ModelDownloadSnapshot
    downloadStateCurrent?: boolean
}

export interface ModelItem {
    name: string
    size: number
    downloaded: boolean
    downloadState?: ModelDownloadEntry
    downloadStateCurrent?: boolean
    status: ModelItemStatus
    parameterSize: string
    quantization: string
    family: string
    digest: string
    /** VRAM consumed by the model when loaded (bytes). */
    sizeVram: number | null
    /** Timestamp when the engine will auto-unload this model. */
    expiresAt: string | null
    /** User-configured keep_alive duration sent when loading the model. */
    expiry: ModelExpiry
    /**
     * Original key used to pull/download this model (e.g. HuggingFace `author/repo/file.gguf`).
     * Stored alongside the model so sync can reproduce the pull on other nodes.
     * Absent for models whose `name` is already a valid pull key (e.g. Ollama tags).
     */
    pullKey?: string
    /**
     * Engine-reported model capabilities (lowercase). Known values include
     * `'vision'`, `'tools'`, `'embedding'`, `'completion'`, `'thinking'`.
     * Empty array when the engine does not expose capability metadata for
     * this model. Consumers that don't care can ignore; the chat UI uses
     * `'vision'` to gate image attachment affordances.
     */
    capabilities: string[]
}
