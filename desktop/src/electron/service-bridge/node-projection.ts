// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineStatusData, EngineType, ModelItem } from '@/shared/types/engines'
import type { NodeItemMetrics } from '@/shared/types/metrics'
import type { NodeItem } from '@/shared/types/nodes'
import type { NodePolicyAdvertisement } from '@/shared/types/node-roles'
import type { NodePerformance } from '@/shared/types/performance'
import { MODULAR_NODE_INFO_SELF_HOST } from '@/shared/constants/modular-runtime'
import { engineManagerName } from '@/shared/utils/engines'
import { parsePolicyAdvertisement, samePolicyAdvertisement } from '@/shared/utils/node-roles'
import { currentPlatform, platformDisplayName } from '@/shared/utils/platform'
import type { JsonObject, JsonValue } from './json-rpc-client'
import {
    PROXY_ENGINES,
    proxySourceForEngine,
    type ProxyEngine,
    type ProxyNodeSource
} from './proxy-engines'

// Live node sources are the reverse proxies, relayed through the broker, and
// the broker's consolidated discovery snapshot. Electron does not consume
// worker discovery protocols directly.
type BrokerNodeSource = ProxyNodeSource | 'broker'

/** Per-engine presence on a node — each proxy reports its own engine. */
interface EnginePresence {
    up: boolean
    /**
     * The node's promoted inference **proxy** port for this engine, as
     * advertised in discovery. Under secure inference the broker registers the
     * `ol`/`lm` service at the proxy's port — never the engine's own port, which
     * is loopback-private and reachable by peers only through that proxy's
     * cluster-mTLS ingress. The engine's real server port is not in discovery;
     * it comes from `engine:remote-get-installed` facts (a peer) or
     * engine-manager status (self).
     */
    port: number
    version: string | null
}

interface ModularNode {
    performance?: NodePerformance
    // Canonical cross-domain node key = the backend's stable per-host UUID
    // (`shared/nodeid`). The broker carries it on `AvailableNode.hostUuid`, the
    // proxies key `Node.ID` by it, and `/v1/node-info` reports it, so every
    // domain now agrees on this id: workload `originatedFrom`/`scheduledOn`,
    // error `nodeId`, and proxy routing all resolve against it. Never key,
    // dedupe, or attribute by hostname — that is display only (`name`).
    id: string
    sources: BrokerNodeSource[]
    // Display hostname (broker `AvailableNode.name` / proxy `Node.Host`). Shown
    // to the user; never a correlation key.
    name: string
    host: string
    // Whether the node is a locally-trusted (pinned) peer and whether it already
    // belongs to a cluster, from broker discovery (`AvailableNode.trusted` /
    // `.clustered`). Absent on the proxy `node/*` feed (defaults false), so a
    // proxy-sourced update keeps the last broker-reported values on merge.
    // `clustered` gates invite eligibility (a node already in a cluster cannot be
    // invited into another).
    trusted: boolean
    clustered: boolean
    nodeInfoPort: number
    nodeInfoUp: boolean
    // The node's own ranked candidate list from broker discovery
    // (`AvailableNode.ipAddresses`, or its single `ipAddress`). Authoritative and
    // ordered: a broker snapshot replaces this wholesale, so an address the node
    // has re-ranked away stops being polled and stops being projected. It is a
    // separate field from {@link proxyAddresses} for exactly that reason — while
    // the two feeds shared one list, a union was the only merge that did not
    // discard the proxy's addresses, and a union can never drop anything.
    brokerAddresses: string[]
    // Addresses known only to the proxy `node/*` feed (`Node.Addresses`), which
    // discovers independently of the broker. Projected behind the broker's order
    // by {@link nodeAddresses}, and the only address list a proxy-only node has.
    proxyAddresses: string[]
    // The node's canonical LAN address. Both discovery sources now stamp it with
    // the backend's shared `netpick.Primary` result: the broker's
    // `AvailableNode.ipAddress` and the proxy `node/*` `ip` field. Used directly
    // for the display IP and the node-info poll target — PAIR UI no longer ranks
    // addresses itself. The broker value is preferred on merge; a proxy-only node
    // carries the proxy's value. Empty only until a node's first event arrives.
    reachableAddress: string
    gpus: ModularGpu[]
    cpu: ModularCpu | null
    memory: ModularMemory | null
    // Validity and observation time apply only to GPU telemetry, not CPU/RAM.
    telemetryValid?: boolean
    observedAt?: number
    // Node-level list of inference-ready hardware ids from /v1/node-info, mapped
    // straight to SystemTopology.inferenceHardwareIds. undefined = the backend
    // does not report readiness yet (UI shows all GPUs). See the routing
    // limitations in docs/services-parity.md.
    inferenceHardwareIds?: string[]
    // The node's unified model list, enriched by the broker daemon from the
    // peer's engine-manager `/v1/models` (models-http) and carried on
    // `AvailableNode.models`. It replaces the retired per-engine proxy
    // `models=` TXT as the remote-node model source; the local node still prefers
    // engine-manager `list_models`. Kept as the flat node-level union for the
    // fallback path (a peer that sends no per-engine attribution).
    models: string[]
    // Per-engine attribution of the enriched model list, keyed by engine-manager
    // engine name ("ollama", "lmstudio", "llamacpp"), carried on
    // `AvailableNode.modelsByEngine`.
    // When present it is authoritative — {@link toEngineModels} attributes each
    // model to the engine that actually serves it, so a dual-engine node no
    // longer blanks its cards. A present engine key with [] is known empty; an
    // empty object means no successful attribution (or a peer that predates
    // per-engine attribution) and falls back to {@link models}. Mirrors
    // noderec.DirectoryNode.EngineModels.
    modelsByEngine: Record<string, string[]>
    // Per-engine set of models currently loaded in memory, keyed by
    // engine-manager engine name ("ollama", "lmstudio", "llamacpp"), carried on
    // `AvailableNode.loadedByEngine`. Normally a subset of
    // {@link modelsByEngine}. An engine key with an empty list means
    // "running, nothing loaded"; a missing key means loaded state wasn't
    // reported. Drives {@link ModelItem.status} `'loaded'`. Empty object when no
    // engine reports loaded state. Mirrors noderec.DirectoryNode.LoadedByEngine.
    loadedByEngine: Record<string, string[]>
    // The node's policy advertisement (declared roles, pause origin), carried
    // on `AvailableNode.policy` from the same inventory fetch as the models.
    // Absent when the node published none, which declares nothing. Only the
    // broker reports it; a proxy-sourced update keeps the broker's value.
    policy?: NodePolicyAdvertisement
    engines: Record<ProxyEngine, EnginePresence>
    lastSeen: number
}

function emptyPresence(): EnginePresence {
    return { up: false, port: 0, version: null }
}

function emptyEngines(): Record<ProxyEngine, EnginePresence> {
    return {
        ollama: emptyPresence(),
        'lm-studio': emptyPresence(),
        'llama-cpp': emptyPresence()
    }
}

/** Immutably set one engine's presence, preserving the other. */
function setEngine(
    engines: Record<ProxyEngine, EnginePresence>,
    engine: ProxyEngine,
    presence: EnginePresence
): Record<ProxyEngine, EnginePresence> {
    return { ...engines, [engine]: presence }
}

/**
 * A node's primary reachable service port: a live engine's promoted proxy port
 * when one is up (the endpoint peers route inference to), else the node-info
 * HTTP port. Display-only — never dialed directly for inference.
 */
function primaryServicePort(node: ModularNode): number {
    for (const engine of PROXY_ENGINES) {
        const presence = node.engines[engine]
        if (presence.up && presence.port > 0) return presence.port
    }
    return node.nodeInfoPort
}

/** Whether any proxy engine reports the node as up. */
function anyEngineUp(node: ModularNode): boolean {
    return PROXY_ENGINES.some(engine => node.engines[engine].up)
}

interface ModularGpu {
    name: string
    vramBytes: number
    vramUsedBytes?: number
    utilizationPercent?: number
}

interface ModularCpu {
    name: string
    cores: number
    utilizationPercent?: number
}

interface ModularMemory {
    totalBytes: number
    usedBytes?: number
}

function objectValue(value: JsonValue | undefined): JsonObject | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

function stringValue(value: JsonValue | undefined): string {
    return typeof value === 'string' ? value : ''
}

function numberValue(value: JsonValue | undefined): number {
    return typeof value === 'number' ? value : 0
}

function observationValue(value: JsonValue | undefined): number | undefined {
    return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}
function booleanValue(value: JsonValue | undefined): boolean {
    return typeof value === 'boolean' ? value : false
}

function stringArrayValue(value: JsonValue | undefined): string[] {
    if (!Array.isArray(value)) return []
    return value.filter((entry): entry is string => typeof entry === 'string')
}

/**
 * Parse the broker `AvailableNode.modelsByEngine` map (engine name -> model
 * names). A missing/malformed field yields `{}`, which {@link modelsForEngine}
 * treats as "no attribution" and falls back to the flat union. Keyed by
 * engine-manager engine name ("ollama", "lmstudio").
 */
function parseModelsByEngine(value: JsonValue | undefined): Record<string, string[]> {
    const obj = objectValue(value)
    if (!obj) return {}
    const out: Record<string, string[]> = {}
    for (const [engine, list] of Object.entries(obj)) {
        out[engine] = stringArrayValue(list)
    }
    return out
}

/**
 * Read an optional string array, preserving the absent vs empty distinction the
 * `inference_hardware_ids` contract relies on: a missing field returns
 * `undefined` (backend reports no readiness — show all GPUs), an empty array
 * returns `[]` (no inference-ready hardware). See the routing limitations in
 * docs/services-parity.md.
 */
function optionalStringArrayValue(value: JsonValue | undefined): string[] | undefined {
    if (!Array.isArray(value)) return undefined
    return value.filter((entry): entry is string => typeof entry === 'string')
}

/**
 * The set of models a node reports loaded in memory for an engine, read from
 * `ModularNode.loadedByEngine` (enriched from the backend's per-engine
 * `loadedByEngine`), whose keys use the engine-manager's spelling.
 */
function loadedNamesForEngine(node: ModularNode | undefined, engineType: EngineType): Set<string> {
    return new Set(node?.loadedByEngine[engineManagerName(engineType)] ?? [])
}

function sameStringList(left: string[], right: string[]): boolean {
    if (left.length !== right.length) return false
    for (let index = 0; index < left.length; index += 1) {
        if (left[index] !== right[index]) return false
    }
    return true
}

/**
 * Order-insensitive comparison for de-duped string lists. mDNS browses report a
 * node's addresses (and TXT) in whatever order the responder returned them, so
 * an order-only change must NOT count as a node change.
 */
function sameStringSet(left: string[], right: string[]): boolean {
    if (left.length !== right.length) return false
    const set = new Set(left)
    for (const value of right) {
        if (!set.has(value)) return false
    }
    return true
}

/** Union two address lists, de-duped, keeping the first list's order. */
function mergeAddresses(existing: string[], next: string[]): string[] {
    const seen = new Set<string>()
    const out: string[] = []
    for (const addr of [...existing, ...next]) {
        if (!addr || seen.has(addr)) continue
        seen.add(addr)
        out.push(addr)
    }
    return out
}

/**
 * Every address a node is currently known to answer at: the broker's ranked
 * order first, then whatever only the proxy feed reported. The single effective
 * list — poll order, `NodeItem.allIpAddresses`, and `AvailableNode.ipAddresses`
 * all derive from it, so none of them can outlive a broker re-rank.
 */
function nodeAddresses(node: ModularNode): string[] {
    return mergeAddresses(node.brokerAddresses, node.proxyAddresses)
}

/**
 * The node's primary IP for display and node-info polling. Both discovery
 * sources now stamp `ModularNode.reachableAddress` with the backend's canonical
 * `netpick.Primary` LAN address (the broker's `AvailableNode.ipAddress` and the
 * proxy `node/*` `ip` field), so there is no local heuristic to fall
 * back to; `host` only covers a node that somehow carries no canonical address.
 */
function primaryNodeAddress(node: ModularNode): string {
    return node.reachableAddress || node.host
}

function nodeInfoPollHosts(node: ModularNode, selfId: string | null): string[] {
    if (node.id === selfId) return [MODULAR_NODE_INFO_SELF_HOST]
    if (node.trusted) return nodeAddresses(node)
    return mergeAddresses([primaryNodeAddress(node)], nodeAddresses(node))
}

function mergeSources(sources: BrokerNodeSource[], source: BrokerNodeSource): BrokerNodeSource[] {
    return sources.includes(source) ? sources : [...sources, source]
}

function removeSource(sources: BrokerNodeSource[], source: BrokerNodeSource): BrokerNodeSource[] {
    return sources.filter(entry => entry !== source)
}

function nvidiaGpuRank(gpu: ModularGpu): number {
    return gpu.name.toLowerCase().includes('nvidia') ? 0 : 1
}

function gpuArrayValue(value: JsonValue | undefined): ModularGpu[] {
    if (!Array.isArray(value)) return []
    const gpus: ModularGpu[] = []
    for (const entry of value) {
        const obj = objectValue(entry)
        if (!obj) continue
        gpus.push({
            name: stringValue(obj.name),
            vramBytes: numberValue(obj.vram_bytes),
            vramUsedBytes: observationValue(obj.vram_used_bytes),
            utilizationPercent: observationValue(obj.utilization_percent)
        })
    }
    return gpus.sort((left, right) => nvidiaGpuRank(left) - nvidiaGpuRank(right))
}

function cpuValue(value: JsonValue | undefined): ModularCpu | null {
    const obj = objectValue(value)
    if (!obj) return null
    return {
        name: stringValue(obj.name),
        cores: numberValue(obj.cores),
        utilizationPercent: observationValue(obj.utilization_percent)
    }
}

function memoryValue(value: JsonValue | undefined): ModularMemory | null {
    const obj = objectValue(value)
    if (!obj) return null
    return {
        totalBytes: numberValue(obj.total_bytes),
        usedBytes: observationValue(obj.used_bytes)
    }
}

function normalizeLastSeen(value: number): number {
    if (value <= 0) return Date.now()
    return value < 10_000_000_000 ? value * 1000 : value
}

function sameGpu(left: ModularGpu, right: ModularGpu): boolean {
    return (
        left.name === right.name &&
        left.vramBytes === right.vramBytes &&
        left.vramUsedBytes === right.vramUsedBytes &&
        left.utilizationPercent === right.utilizationPercent
    )
}

function sameGpuList(left: ModularGpu[], right: ModularGpu[]): boolean {
    if (left.length !== right.length) return false
    for (let index = 0; index < left.length; index += 1) {
        if (!sameGpu(left[index], right[index])) return false
    }
    return true
}

function sameCpu(left: ModularCpu | null, right: ModularCpu | null): boolean {
    if (!left || !right) return left === right
    return (
        left.name === right.name &&
        left.cores === right.cores &&
        left.utilizationPercent === right.utilizationPercent
    )
}

function sameMemory(left: ModularMemory | null, right: ModularMemory | null): boolean {
    if (!left || !right) return left === right
    return left.totalBytes === right.totalBytes && left.usedBytes === right.usedBytes
}

function sameInferenceHardwareIds(left?: string[], right?: string[]): boolean {
    // Preserve absent (undefined) vs empty ([]) — they mean different things.
    if (!left || !right) return left === right
    return sameStringList(left, right)
}

/**
 * Whether only live readings differ: GPU and CPU utilization, VRAM and memory
 * in use. Everything else node-info reports is hardware that a node card, the
 * discovery list, and engine status are built from; these readings appear
 * only in `metrics:update`.
 */
function sameHardware(
    node: ModularNode,
    gpus: ModularGpu[],
    cpu: ModularCpu | null,
    memory: ModularMemory | null,
    inferenceHardwareIds: string[] | undefined
): boolean {
    if (node.gpus.length !== gpus.length) return false
    for (let index = 0; index < gpus.length; index += 1) {
        const left = node.gpus[index]
        const right = gpus[index]
        if (left.name !== right.name || left.vramBytes !== right.vramBytes) return false
    }
    if (!node.cpu || !cpu) {
        if (node.cpu !== cpu) return false
    } else if (node.cpu.name !== cpu.name || node.cpu.cores !== cpu.cores) {
        return false
    }
    if (!node.memory || !memory) {
        if (node.memory !== memory) return false
    } else if (node.memory.totalBytes !== memory.totalBytes) {
        return false
    }
    return sameInferenceHardwareIds(node.inferenceHardwareIds, inferenceHardwareIds)
}

function sameTelemetry(
    node: ModularNode,
    gpus: ModularGpu[],
    cpu: ModularCpu | null,
    memory: ModularMemory | null,
    inferenceHardwareIds: string[] | undefined
): boolean {
    return (
        sameGpuList(node.gpus, gpus) &&
        sameCpu(node.cpu, cpu) &&
        sameMemory(node.memory, memory) &&
        sameInferenceHardwareIds(node.inferenceHardwareIds, inferenceHardwareIds)
    )
}

function modelItem(name: string, loaded = false): ModelItem {
    return {
        name,
        size: 0,
        downloaded: true,
        status: loaded ? 'loaded' : 'idle',
        parameterSize: '',
        quantization: '',
        family: '',
        digest: '',
        sizeVram: null,
        expiresAt: null,
        expiry: '10m',
        capabilities: []
    }
}

function toNodeItem(node: ModularNode, selfId: string | null): NodeItem {
    const ipAddress = primaryNodeAddress(node)
    const addresses = nodeAddresses(node)
    // `NodeItem.port` is the node's primary reachable service port: a live
    // engine's promoted proxy port when one is up, otherwise the node-info HTTP
    // port. This keeps the value meaningful regardless of which discovery source
    // produced the node (a proxy reports the promoted proxy port, the broker
    // reports the node-info port).
    const port = primaryServicePort(node)
    return {
        id: node.id,
        ...(node.performance ? { performance: node.performance } : {}),
        // Display label: the hostname, else the reachable address; never the raw
        // UUID key (falls back to it only if a node somehow has neither).
        name: node.name || ipAddress || node.id,
        // The backend evicts mDNS nodes on its own (~60s of unbroken silence with
        // no inference traffic from the node either), but manual nodes persist and
        // report reachability via flags. Reflect that so an unreachable/sleeping
        // node drops into the UI's offline group instead of lingering as an active
        // card.
        status: anyEngineUp(node) || node.nodeInfoUp ? 'active' : 'offline',
        ipAddress,
        port,
        allIpAddresses: addresses.length > 0 ? addresses : [ipAddress],
        topology: {
            cpu: {
                model: node.cpu?.name ?? '',
                cores: node.cpu?.cores ?? 0,
                threads: node.cpu?.cores ?? 0
            },
            gpus: node.gpus.map((gpu, index) => ({
                id: `${node.id}:gpu:${index}`,
                name: gpu.name,
                vramTotal: gpu.vramBytes
            })),
            ram: node.memory?.totalBytes ?? 0,
            storage: [],
            inferenceHardwareIds: node.inferenceHardwareIds
        },
        // The local node's OS is known from the running process; the backend's
        // discovery/node-info plane reports no OS for remote nodes, so they fall
        // back to a placeholder. Remote OS is not reported by the current
        // discovery/node-info contract.
        os: node.id === selfId ? platformDisplayName(currentPlatform()) : 'Windows'
    }
}

function toMetrics(node: ModularNode): NodeItemMetrics {
    const gpus = node.telemetryValid && node.observedAt !== undefined ? node.gpus : []
    const memoryUsage =
        node.memory && node.memory.totalBytes > 0 && node.memory.usedBytes !== undefined
            ? (node.memory.usedBytes / node.memory.totalBytes) * 100
            : undefined
    const current = {
        timestamp: Date.now(),
        cpuUtilization: node.cpu?.utilizationPercent,
        memoryUsage,
        gpuUtilization: gpus.map((gpu, index) => ({
            id: `${node.id}:gpu:${index}`,
            timestamp: node.observedAt,
            value: gpu.utilizationPercent
        })),
        gpuVramUsage: gpus.map((gpu, index) => ({
            id: `${node.id}:gpu:${index}`,
            timestamp: node.observedAt,
            value:
                gpu.vramBytes > 0 && gpu.vramUsedBytes !== undefined
                    ? (gpu.vramUsedBytes / gpu.vramBytes) * 100
                    : undefined
        }))
    }
    return { id: node.id, current, historical: [current] }
}

function toEngineStatus(
    node: ModularNode,
    engine: ProxyEngine,
    proxyPort: number | null
): EngineStatusData {
    const presence = node.engines[engine]
    const status: EngineStatusData = {
        engineType: engine,
        nodeId: node.id,
        processStatus: presence.up ? 'running' : 'stopped',
        // Discovery advertises the promoted proxy port, never the engine's own
        // (now loopback-private) port, so the engine port is unknown from a
        // proxy `node/*` event; it is filled only from engine-manager facts.
        enginePort: null,
        // Prefer an authoritative proxy port from the caller (the local broker's
        // bound port). Otherwise the port the peer advertised in discovery IS
        // its promoted proxy port for this engine — never a guess.
        proxyPort:
            proxyPort && proxyPort > 0
                ? proxyPort
                : presence.up && presence.port > 0
                  ? presence.port
                  : null
    }
    if (presence.version) {
        status.installedVersion = presence.version
    }
    return status
}

/**
 * Parse a proxy `node/*` payload for a given engine. The `ollama-proxy` and
 * `lmstudio-proxy` both emit the same `Node` shape (id/host/port/addresses/txt);
 * the engine is determined by which relay namespace it arrived on. The presence
 * is stamped onto that engine only; the other engine stays empty.
 */
function parseProxyNode(params: JsonValue | undefined, engine: ProxyEngine): ModularNode | null {
    const obj = objectValue(params)
    if (!obj) return null
    // The proxy keys `Node.ID` by the stable per-host UUID (and rejects empty),
    // so this is already the canonical node key — no TXT parsing needed.
    const id = stringValue(obj.id)
    if (!id) return null
    const engines = emptyEngines()
    engines[engine] = {
        up: true,
        // Under secure inference this is the peer's promoted inference proxy
        // port, not the engine's own (loopback-private) port.
        port: numberValue(obj.port),
        // The proxy `Node` carries no version field; engine version comes from
        // nvpair-engine-manager, not discovery.
        version: null
    }
    return {
        id,
        sources: [proxySourceForEngine(engine)],
        // `Node.Host` is the hostname; empty for the self-bridge manual node,
        // in which case the broker discovery entry supplies the display name on
        // merge (see mergeNode). Never fall back to the UUID id here.
        name: stringValue(obj.name) || stringValue(obj.host),
        host: stringValue(obj.host),
        // The proxy `node/*` feed carries no trust/cluster flags; keep whatever
        // the broker last reported (see mergeNode).
        trusted: false,
        clustered: false,
        // The proxy `node/*` payload no longer carries a model list (models-http
        // moved it off the mDNS TXT to the broker's `AvailableNode.models`); a
        // proxy-sourced node contributes engine presence only, never models.
        models: [],
        modelsByEngine: {},
        // A proxy `node/*` event carries no loaded-model state either; the
        // broker-enriched value is preserved across the proxy merge (see mergeNode).
        loadedByEngine: {},
        nodeInfoPort: 0,
        nodeInfoUp: false,
        // A proxy `node/*` event carries no broker ranking; its `Addresses` are
        // this feed's own contribution and are kept behind the broker's list.
        brokerAddresses: [],
        proxyAddresses: stringArrayValue(obj.addresses),
        // The proxy stamps the node's canonical LAN address (netpick.Primary over
        // its TXT + addresses) on the `ip` field of every node/* payload,
        // so a proxy-only node carries a real reachable address instead
        // of forcing PAIR UI to re-rank the raw address list itself.
        reachableAddress: stringValue(obj.ip),
        gpus: [],
        cpu: null,
        memory: null,
        engines,
        lastSeen: Date.now()
    }
}

function parseBrokerNode(params: JsonValue | undefined): ModularNode | null {
    const obj = objectValue(params)
    if (!obj) return null
    // `AvailableNode.hostUuid` is the canonical node key. The broker's discovery
    // store rejects an empty UUID, so every node it reports carries one; a
    // payload without it is malformed and dropped rather than keyed by hostname.
    const hostUuid = stringValue(obj.hostUuid)
    if (!hostUuid) return null
    // `AvailableNode.id`/`.name` are the mDNS instance hostname (display only).
    const hostname = stringValue(obj.name) || stringValue(obj.id)
    const ipAddress = stringValue(obj.ipAddress) || stringValue(obj.ip_address)
    const port = numberValue(obj.port)
    // `AvailableNode.ipAddresses` is the node's own ranked candidate list, omitted
    // when it published a single address (where it would only repeat ipAddress).
    const ipAddresses = stringArrayValue(obj.ipAddresses)
    const brokerAddresses = ipAddresses.length > 0 ? ipAddresses : ipAddress ? [ipAddress] : []
    return {
        id: hostUuid,
        sources: ['broker'],
        name: hostname,
        host: ipAddress,
        // Broker discovery is authoritative for trust/cluster membership state.
        trusted: booleanValue(obj.trusted),
        clustered: booleanValue(obj.clustered),
        nodeInfoPort: port,
        nodeInfoUp: Boolean(ipAddress) && port > 0,
        brokerAddresses,
        proxyAddresses: [],
        // Broker `AvailableNode.ipAddress` is the head of the node's own ranked
        // candidate list — one canonical address shared across broker /
        // cluster-manager / errors / workload-manager, so the display IP and the
        // node-info poll target agree. The backend owns the ranking and the
        // failover across the rest of the list; PAIR never re-ranks.
        reachableAddress: ipAddress,
        gpus: [],
        cpu: null,
        memory: null,
        // Broker `AvailableNode.models`: the node's model list, enriched by its
        // daemon from the peer's engine-manager `/v1/models` (models-http). This
        // is the remote-node model source now that the proxy `models=` TXT is
        // retired. `omitempty` on the wire — absent when the node runs no engines.
        models: stringArrayValue(obj.models),
        // Broker `AvailableNode.modelsByEngine`: the same list attributed per
        // engine. `omitempty` on the wire — an empty object (no engines, or a
        // peer that predates per-engine attribution) drives the
        // {@link modelsForEngine} fallback to the flat union.
        modelsByEngine: parseModelsByEngine(obj.modelsByEngine),
        // Broker `AvailableNode.loadedByEngine`: the per-engine set of models
        // loaded in memory, enriched from the peer's engine-manager
        // `/v1/models`. `omitempty` on the wire — absent for a peer that reports
        // no loaded state, which stamps every model `idle`.
        loadedByEngine: parseModelsByEngine(obj.loadedByEngine),
        // Broker `AvailableNode.policy`: `omitempty` on the wire, absent until
        // the node's broker has published one.
        policy: parsePolicyAdvertisement(obj.policy),
        engines: emptyEngines(),
        lastSeen: normalizeLastSeen(numberValue(obj.lastSeen) || numberValue(obj.last_seen))
    }
}

function brokerNodesValue(value: JsonValue | undefined): ModularNode[] {
    const obj = objectValue(value)
    const entries = Array.isArray(value) ? value : Array.isArray(obj?.nodes) ? obj.nodes : []
    const nodes = new Map<string, ModularNode>()
    for (const entry of entries) {
        const node = parseBrokerNode(entry)
        if (node) nodes.set(node.id, node)
    }
    return Array.from(nodes.values())
}

function samePresence(left: EnginePresence, right: EnginePresence): boolean {
    return left.up === right.up && left.port === right.port && left.version === right.version
}

function sameEngines(
    left: Record<ProxyEngine, EnginePresence>,
    right: Record<ProxyEngine, EnginePresence>
): boolean {
    for (const engine of PROXY_ENGINES) {
        if (!samePresence(left[engine], right[engine])) return false
    }
    return true
}

function sameNode(left: ModularNode, right: ModularNode): boolean {
    return (
        JSON.stringify(left.performance) === JSON.stringify(right.performance) &&
        left.id === right.id &&
        sameStringList(left.sources, right.sources) &&
        left.name === right.name &&
        left.host === right.host &&
        left.trusted === right.trusted &&
        left.clustered === right.clustered &&
        left.reachableAddress === right.reachableAddress &&
        left.nodeInfoPort === right.nodeInfoPort &&
        left.nodeInfoUp === right.nodeInfoUp &&
        // Order matters for the broker's list and only for it: that order is the
        // node's own ranking of where to reach it, and it drives the poll order.
        sameStringList(left.brokerAddresses, right.brokerAddresses) &&
        sameStringSet(left.proxyAddresses, right.proxyAddresses) &&
        sameTelemetry(left, right.gpus, right.cpu, right.memory, right.inferenceHardwareIds) &&
        sameStringList(left.models, right.models) &&
        sameModelsByEngine(left.modelsByEngine, right.modelsByEngine) &&
        // A residency-only change (a model loaded/evicted with the installed set
        // unchanged) must still count as a node change so the loaded dot updates.
        sameModelsByEngine(left.loadedByEngine, right.loadedByEngine) &&
        // A role edit changes nothing else on the node, so it must count alone.
        samePolicyAdvertisement(left.policy, right.policy) &&
        sameEngines(left.engines, right.engines) &&
        left.lastSeen === right.lastSeen
    )
}

/**
 * Compare two per-engine model maps. The attribution can change while the flat
 * union stays identical (e.g. a model that multiple engines now serve moved between
 * the per-engine buckets), so this must be checked independently of
 * {@link sameStringList} on `models` or a card would miss the re-attribution.
 */
function sameModelsByEngine(
    left: Record<string, string[]>,
    right: Record<string, string[]>
): boolean {
    const leftKeys = Object.keys(left)
    if (leftKeys.length !== Object.keys(right).length) return false
    for (const key of leftKeys) {
        if (!(key in right) || !sameStringList(left[key], right[key])) return false
    }
    return true
}

export type { BrokerNodeSource, ModularNode }
export {
    emptyPresence,
    setEngine,
    primaryServicePort,
    objectValue,
    stringValue,
    numberValue,
    booleanValue,
    optionalStringArrayValue,
    loadedNamesForEngine,
    sameStringList,
    nodeAddresses,
    primaryNodeAddress,
    nodeInfoPollHosts,
    mergeSources,
    removeSource,
    gpuArrayValue,
    cpuValue,
    memoryValue,
    sameHardware,
    modelItem,
    toNodeItem,
    toMetrics,
    toEngineStatus,
    parseProxyNode,
    brokerNodesValue,
    sameNode,
    sameModelsByEngine
}
