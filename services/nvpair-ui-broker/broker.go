// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/applog"
	"nvpair-shared/buildinfo"
	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	"nvpair-shared/errors"
	"nvpair-shared/nodeid"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/schedulerwire"

	"nvpair-ui-broker/relay"
	"nvpair-ui-broker/workloadstore"
)

// Version is stamped at build time via -ldflags "-X main.Version=...".
// See versions.json at the repo root for the source of truth.
var Version = "dev"

const restoreEnabledEnginesMethod = "engine:restore-enabled"
const prepareEngineShutdownMethod = "engine:prepare-shutdown"

// ReadyParams is the payload of the "app:ready" notification the broker
// emits on every new connection (or once at startup in stdio mode).
// The method name uses the external schema's app: namespace; the
// payload itself remains a single version field.
type ReadyParams struct {
	Version string `json:"version"`
}

// PingResult is the response to the "ping" request. uptime_ms is included
// so callers can sanity-check they're talking to a freshly started broker
// (vs. one that's been running for hours and may be holding stale state
// once we add real state).
type PingResult struct {
	Pong     bool   `json:"pong"`
	Version  string `json:"version"`
	UptimeMS int64  `json:"uptime_ms"`
}

// VersionResult is the response to the "version" request.
type VersionResult struct {
	Version string `json:"version"`
}

// AvailableNode is the per-node wire format used both as the array
// element of the discovery:get-nodes response and as each element of
// the discovery:nodes-changed notification payload. It is intentionally
// narrower than the rich EnrichedNode the broker holds internally —
// only the fields the external schema requires. JSON keys are
// camelCase here because the external consumer's schema demands it;
// the rest of the NVPAIR codebase uses snake_case, so this is the
// deliberate camelCase island at the broker's outer boundary.
type AvailableNode struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	// HostUUID is the node's stable per-host identity (the daemon's uuid= TXT).
	// It is invariant across a PC rename and unique per physical machine, so it
	// is the key clients should dedup/track nodes by; id/name stay the hostname
	// for display. Two machines sharing a hostname are distinct by HostUUID. It
	// is the same value the cluster surface exposes as nodeUuid.
	// Omitted for manual nodes, which carry no UUID and are keyed by id.
	HostUUID  string `json:"hostUuid,omitempty"`
	IPAddress string `json:"ipAddress"`
	// IPAddresses is every address this node published, in its own ranked order
	// with IPAddress first. A multi-homed node has no single address every peer
	// can reach — a direct-connect link is reachable only from the machine on its
	// far end — so a client that must connect walks this list rather than
	// treating IPAddress as the only answer. Omitted when the node published one
	// address, where it would only repeat IPAddress.
	IPAddresses []string `json:"ipAddresses,omitempty"`
	Port        int      `json:"port"`
	LastSeen    int64    `json:"lastSeen"`
	// Trusted is the additive annotation: whether this node is a paired
	// cluster peer (we hold a pin for its cluster-uuid). Populated from the
	// promoted daemon's directory; false for legacy/non-cluster entries.
	Trusted bool `json:"trusted"`
	// Clustered reports whether this node belongs to some cluster (it advertises
	// a cluster-uuid), independent of whether we are paired with it. A client
	// uses it to suppress an invite that an already-clustered peer would reject.
	Clustered bool `json:"clustered,omitempty"`
	// Models is the node's advertised model list, enriched by the daemon from the
	// peer's engine-manager /v1/models endpoint (models-http). Omitted when the
	// node advertises no engines or none are running.
	Models []string `json:"models,omitempty"`
	// ModelsByEngine attributes Models to the engine that serves each model,
	// keyed by engine-manager engine name (e.g. "ollama", "lmstudio"). Additive:
	// Models stays the flat de-duplicated union for consumers that only need the
	// node-level set; this is the per-engine breakdown a consumer needs to show a
	// remote node's models under the correct engine (the UI's engine card).
	// Omitted when no engine reports models.
	ModelsByEngine map[string][]string `json:"modelsByEngine,omitempty"`
	// LoadedByEngine names the models currently resident in memory per engine
	// (normally a subset of ModelsByEngine), enriched by the daemon from the
	// peer's engine-manager /v1/models loadedByEngine field. Lets a per-engine
	// consumer show which of a remote node's models are loaded. Omitted
	// when no engine reports loaded state.
	LoadedByEngine map[string][]string `json:"loadedByEngine,omitempty"`
	// Policy is the node's own policy advertisement (declared roles, policy
	// revision, pause origin) from the same /v1/models fetch. Omitted when the
	// node publishes none, which means it declares nothing.
	Policy *nodepolicy.Advertisement `json:"policy,omitempty"`

	ServiceBuildID buildinfo.ID `json:"serviceBuildId,omitempty"`
}

// GetNodesResult is the response to "discovery:get-nodes". Wrapped in an
// object (rather than a bare array) so we can add summary fields later
// without breaking clients. The notification payload uses a bare array
// instead — see emitNodesChanged.
type GetNodesResult struct {
	Nodes []AvailableNode `json:"nodes"`
}

// SubscriptionResult is the ack returned by both "discovery:subscribe"
// and "discovery:unsubscribe". It echoes the resulting subscription
// state so a caller can confirm the broker's view without tracking it
// locally. Both methods are idempotent: re-subscribing or
// re-unsubscribing returns the same state and never errors.
type SubscriptionResult struct {
	Subscribed bool `json:"subscribed"`
}

// ProxyStatusResult is the response to "<engine>-proxy:get-status". Ready is
// false (and Port 0) until that facade has emitted its "ready" notification —
// or always, if no proxy is being supervised. Clients poll this to learn where
// the local facade is listening, since the proxy is optional and comes up
// asynchronously after app:ready.
type ProxyStatusResult struct {
	Ready bool `json:"ready"`
	Port  int  `json:"port"`
}

// Broker holds per-session state. A new Broker is constructed for every
// client connection in listen mode so future per-session caches (auth
// tokens, watched-resource cursors, etc.) don't bleed across clients.
type Broker struct {
	// routingForward bounds routing control forwarding to one operation.
	routingForward atomic.Bool
	// settingsApplyMu serializes engine settings applies end to end, including
	// the engine stop and restart. engineConfigMu guards only the journal and
	// is released while an engine restarts, so this is what keeps two applies —
	// which may compete for the same port — from interleaving. Always take it
	// before engineConfigMu, never the reverse.
	settingsApplyMu sync.Mutex
	engineConfigMu  sync.Mutex
	// Only live install continuations are retained, under engineConfigMu.
	pendingInstallStarts map[*pendingInstallStart]struct{}
	switchInit           sync.Once
	engineSwitch         *engineSwitchCoordinator
	engineSettingsLoaded bool
	engineSettingsError  error
	engineSettingsEpoch  string
	engineSettings       map[string]*engineSettingsRecord
	settingsRelayMu      sync.Mutex
	settingsCancels      map[string]context.CancelFunc
	activeSettings       map[string]activeSettingsOperation
	codec                *Codec
	cancel               context.CancelFunc
	startedAt            time.Time
	nodeID               string
	scannerPath          string
	nodeInfoPath         string
	proxyPath            string
	proxyEngines         []string
	workloadMgrPath      string
	errorsPath           string
	engineMgrPath        string
	engineMgrHTTPPort    int
	engineMgrControlPort int
	nodeInfoPort         int
	errorsPort           int
	workloadMgrPort      int
	clusterMgrPort       int
	manualNodesPath      string
	settingsPath         string
	clusterMgrPath       string
	schedulerPath        string
	clusterDir           string
	// Managed-port state is prepared before proxy startup and read by the proxy
	// supervisor/reader goroutines. Ollama commits its pending backend move after
	// its proxy reserves :11434; LM Studio moves through engine-manager first,
	// then releases its gate after the proxy owns :1234 and has its backend.
	//
	// managedFacade, backendPort and startupPort now live per engine on
	// engineProxyRuntime. What remains here is Ollama's deferred-move state,
	// which only an adopted engine has.
	managedOllamaBackend atomic.Int32
	ollamaMoveInFlight   atomic.Bool
	// ollamaHostAlias is one synchronized snapshot because engine-manager and
	// proxy supervisors can respawn while startup is still committing (or
	// rolling back) the inherited, local-http OLLAMA_HOST alias.
	ollamaHostAliasMu                sync.RWMutex
	ollamaHostAlias                  ollamaHostAlias
	ollamaProxyGeneration            uint64 // guarded by ollamaHostAliasMu
	ollamaHostAliasSyncMu            sync.Mutex
	stickyErrorsMu                   sync.Mutex
	stickyErrors                     map[string]errors.ServiceError // see stickyErrorIDs
	ollamaPortReady                  chan struct{}
	ollamaPortReadyOnce              sync.Once
	lmstudioProxyGeneration          atomic.Uint64
	lmstudioProxyPublishedGeneration atomic.Uint64
	lmstudioPortReady                chan struct{}
	lmstudioPortReadyOnce            sync.Once
	lmstudioReadyMu                  sync.Mutex
	store                            *discoveryStore
	telemetry                        *telemetryCache
	// relayDir is the discovery directory, fed by the promoted daemon's
	// discovery:node-* stream. It runs alongside the legacy store during the
	// migration; the client-facing get-nodes/subscribe path moves onto it at the
	// flag-day cutover.
	relayDir *relay.Directory
	// regCache holds this node's service registrations. The broker owns
	// registration for the workers it spawns (it knows their ports; transport is
	// policy-derived), relaying the set to the daemon and replaying it whenever
	// the daemon (re)starts.
	regCache *relay.RegistrationCache
	// localServicesMu serializes cache mutations, generations and delivery to
	// the current cluster-manager. No child I/O runs while this lock is held.
	localServicesMu         sync.Mutex
	localServicesGeneration uint64

	// workersMu guards the supervised worker handles below. They are no
	// longer assigned once in Serve and then treated as read-only: each
	// worker's supervisor swaps its handle on a restart (on the monitor
	// goroutine) while request handlers and the log-level fan-out read
	// them (on the read-loop / reader goroutines). Every access therefore
	// goes through the get*/set* helpers under this lock.
	workersMu sync.Mutex
	// serveCtx is the broker's own lifetime, recorded so work started on a
	// spawn goroutine can be abandoned when the broker is shutting down.
	// Guarded by workersMu because a respawn reads it off the monitor
	// goroutine. See proxyBringUpContext.
	serveCtx            context.Context
	scanner             *scannerProcess
	peerServicesMu      sync.Mutex
	nodeInfo            *nodeInfoProcess
	workloadMgr         *workloadManagerProcess
	errorsProc          *errorsProcess
	engineMgr           *rpcWorker
	engineMgrRecovering bool // workersMu; replacement is unavailable until switch recovery settles
	// engineMgrSource is the source token spawnEngineMgr bound engineMgr to,
	// and engineMgrBindCh wakes anyone waiting for that binding. The pair is
	// the only place a model-mutation confirmation looks; getEngineMgr hides
	// recovery and cannot name the worker that sent a request. See
	// model_mutation.go.
	engineMgrSource *engineManagerSource
	engineMgrBindCh chan struct{}
	manualNodes     *rpcWorker
	settings        *rpcWorker
	clusterMgr      *clusterManagerProcess
	scheduler       *rpcWorker
	// proxyProc is the one proxy process whether or not any facade came up in
	// it. The per-engine handles say which facades serve; this one is what the
	// node policy and the routing listener, which belong to the process, talk
	// to. Guarded by workersMu; read through proxyProcessHandle.
	proxyProc *proxyProcess

	// proxySupMu serializes starting and stopping proxySup. It is held for as
	// long as a start or stop takes, so nothing on a request path takes it; see
	// reconcileProxySupervision. proxySupWorkMu guards both coalesced stages.
	proxySupMu      sync.Mutex
	proxySupPhase   proxySupPhase
	proxySupWorkMu  sync.Mutex
	proxySupDirty   bool
	proxySupWorking bool
	// Retirement is a separate coalesced stage of this owner: blocked workload
	// output must not occupy the supervision worker or the monitor it joins.
	// Pending entries contain identities only, never a queue of wire frames.
	proxyRetirePending map[departedWorkload]string
	proxyRetireDone    chan struct{}
	// Set after the supervisor's terminal callback completes. An absent process
	// otherwise belongs to its existing spawn/backoff and must not be replaced.
	proxySupExhausted atomic.Bool
	// Published before shutdown waits for proxySupMu, so queued starts see it.
	proxySupClosing atomic.Bool
	// Optional publication barriers, set before supervision starts by tests.
	beforeProxySpawnPublished      func()
	beforeProxyExhaustionPublished func()
	beforeProxyShutdownLocked      func()

	// Per-worker supervisors own the (re)spawn + crash-detect + restart
	// lifecycle of each worker. nil when the worker's binary wasn't resolved
	// (or its first spawn failed).
	//
	// proxySup covers every engine: one nvpair-proxy process hosts a facade per
	// enabled engine, so there is no per-engine proxy supervisor.
	scannerSup     *supervisor
	nodeInfoSup    *supervisor
	proxySup       *supervisor
	workloadMgrSup *supervisor
	errorsSup      *supervisor
	engineMgrSup   *supervisor
	manualNodesSup *supervisor
	settingsSup    *supervisor
	clusterMgrSup  *supervisor
	schedulerSup   *supervisor

	// subMu guards subscribed. The discovery:nodes-changed stream is
	// opt-in: emitNodesChanged (called on the scanner-event goroutine)
	// reads this flag while the discovery:subscribe / discovery:unsubscribe
	// handlers (called on the read-loop goroutine) flip it, so the two
	// goroutines need a lock between them.
	subMu      sync.Mutex
	subscribed bool

	// proxyMu guards every engineProxyRuntime.subscribed. The
	// <namespace>:<event> streams are opt-in like discovery's: the forward
	// hooks (on each proxy's reader goroutine) read the flag while the
	// subscribe / unsubscribe handlers (on the read-loop goroutine) flip it.
	proxyMu sync.Mutex

	// engineProxies holds per-engine proxy state, one entry per engine in the
	// descriptor table. Built on first use; see engineProxy.
	engineProxiesOnce sync.Once
	engineProxies     map[string]*engineProxyRuntime

	// dynamicEngines is the descriptor engines fronted beyond the built-in
	// table; see enginecatalog.go.
	dynamicEngines dynamicEngineCatalog

	// workloadsMu guards workloadsSubscribed. The workloads:* stream is
	// opt-in too: emitWorkloadEvent (called on the proxy reader goroutine
	// for local echoes and on the workload-manager reader goroutine for
	// peer-origin relays) reads the flag while the workloads:subscribe /
	// workloads:unsubscribe handlers (on the read-loop goroutine) flip it.
	workloadsMu         sync.Mutex
	workloadsSubscribed bool

	// workloadEmitMu serializes the apply→fan→notify sequence in
	// emitWorkloadEvent so the client-visible order matches the order events
	// are applied to the store, even when the node-loss sweep (a separate
	// goroutine) races a relayed event for the same workload.
	workloadEmitMu sync.Mutex

	// workloads is the broker's authoritative index of cluster workloads
	// (current + historic), keyed by (originatedFrom, id). It applies an
	// order-independent, monotonic merge so a stale/out-of-order event can't
	// resurrect a finished workload, and it backs the node-loss sweep
	// (failWorkloadsForNode). Written from the proxy / workload-manager reader
	// goroutines (via emitWorkloadEvent) and the scanner-event goroutine (via
	// failWorkloadsForNode); it is internally synchronized.
	workloads   *workloadstore.Store
	performance *performanceRuntime
	// lastLocalRunID is guarded by workloadEmitMu. Only actual proxy input
	// advances the fence; broker-synthesized retirement never does.
	lastLocalRunID string
	// Optional barrier before the serialized source check, set by tests before
	// proxy notifications start.
	beforeProxyWorkloadApply func()

	// engineMu guards engineSubscribed. The engine:<event> stream is
	// opt-in like proxy's: forwardEngineNotificationFrom (engine-manager reader
	// goroutine) reads the flag while the engine:subscribe /
	// engine:unsubscribe handlers (read-loop goroutine) flip it.
	engineMu         sync.Mutex
	engineSubscribed bool

	// engineRelayMu guards engineRelaySubID, the relay-directory subscription
	// id for engine-manager's ec peer set (its upward discovery:subscribe). A
	// (re)subscribe replaces the prior registration; worker exit clears it.
	engineRelayMu    sync.Mutex
	engineRelaySubID int

	// manualMu guards manualNodeKeys, mapping each manual node's id to the
	// discovery-store key it currently occupies — its real hostUuid once
	// node-info reports one, else its manual id until then. Tracked
	// per manual id so the broker can (a) re-key the store entry when a manual
	// node's node-info first reveals its stable UUID, and (b) evict the right
	// key on removal or a manual-nodes crash — since the store key is no longer
	// the manual id once the UUID is known. Several manual aliases (distinct
	// names/addresses for one machine) can map to the same key; the store/proxy
	// manual claim is dropped only once the last alias for a key is gone, and
	// when an alias that a key was currently projecting from leaves, a surviving
	// alias is reprojected from manualNodeStatuses (its last-seen payload) so the
	// removed alias's address/models don't linger (the survivor's own periodic
	// probe emits no update while its status is unchanged).
	manualMu           sync.Mutex
	manualNodeKeys     map[string]string
	manualNodeStatuses map[string]manualNodeStatusEntry

	// scannerErrorIDsMu guards scannerErrorIDs: the error ids the scanner
	// currently has standing in nvpair-errors, so a scanner that dies can have
	// them cleared. Its own record of what it reported dies with it, and nothing
	// else would ever clear them.
	scannerErrorIDsMu sync.Mutex
	scannerErrorIDs   map[string]struct{}

	// schedMu guards the cached ranking and its generation.
	//
	// One cache, not one per engine: the scheduler computes a single node-wide
	// ranking and emits it once per engine, so per-engine copies would hold
	// identical content and prevent the duplicate from being recognized as one.
	// havePriority distinguishes "no ranking yet" from a cached empty one, which
	// is a legitimate state meaning the scheduler has no influence.
	//
	// priorityApplyMu serializes asynchronous node/set-priority deliveries; a
	// stale generation is skipped before it can overwrite a newer proxy order.
	schedMu            sync.Mutex
	lastPriority       schedulerwire.Priority
	havePriority       bool
	priorityGeneration uint64
	priorityApplyMu    sync.Mutex

	// schedulerFeedMu serializes scheduler initialization with live discovery
	// and workload fanout. A restarted scheduler receives its active workload
	// baseline before its discovery baseline, then only newer live events.
	schedulerFeedMu sync.Mutex

	// nodePolicy is the fork's node policy: the policy file, availability,
	// and the state mirrored into the proxy. See policy.go.
	nodePolicy policyRuntime
}

// workerPaths bundles the resolved binary paths for every worker the
// broker can supervise. Carrying them as a struct keeps NewBroker's
// signature stable as more workers are adopted. The scanner path is
// required; every other field is optional (an empty string means the
// broker won't spawn that worker because its binary couldn't be resolved).
type workerPaths struct {
	scanner  string
	nodeInfo string
	// proxy is the single nvpair-proxy binary, started once per entry in
	// proxyEngines and then asked for that engine's facade via facade/enable.
	proxy        string
	proxyEngines []string
	workloadMgr  string
	errors       string
	engineMgr    string
	// Zero ports retain the product defaults for programmatic callers.
	engineMgrHTTPPort    int
	engineMgrControlPort int
	nodeInfoPort         int
	errorsPort           int
	workloadMgrPort      int
	clusterMgrPort       int
	manualNodes          string
	settings             string
	clusterMgr           string
	scheduler            string
	// clusterDir is the cluster-manager config dir (node.crt/node.key +
	// trusted/). Threaded to every worker that does cluster-scoped inter-node
	// mTLS so they serve/dial pinned peers once this node joins a cluster.
	clusterDir string
}

// NewBroker constructs a per-session broker. paths.scanner is required —
// the scanner is the broker's core worker. Every other path is optional:
// an empty string means the broker won't spawn that worker (e.g. the
// binary couldn't be resolved). Without node-info the broker runs without
// local node advertisement; without the proxy, no local Ollama reverse
// proxy; without the workload-manager, no cluster workload relay; without
// nvpair-errors, the service-error pipeline is disabled (producers' errors
// are dropped, as before); without the cluster-manager, no cluster pairing
// and membership (cluster:* / nodes:*). Discovery still works in every case.
func NewBroker(codec *Codec, paths workerPaths) *Broker {
	// Resolve the local node id once, as the stable per-host UUID (the same
	// value the node-scanner advertises as hostUuid and the cluster-manager
	// uses as nodeUuid). It's stamped onto every local-origin workload
	// (originatedFrom) and every local error (nodeId) so the whole cluster keys
	// this node by an identity that survives a PC rename — hostname is display
	// only. The same value is passed to nvpair-errors via --node-id so
	// its localNodeID stays in lockstep with what the broker stamps.
	nodeID := resolveLocalNodeID(paths.clusterDir)
	return &Broker{
		codec:                codec,
		startedAt:            time.Now(),
		nodeID:               nodeID,
		scannerPath:          paths.scanner,
		nodeInfoPath:         paths.nodeInfo,
		proxyPath:            paths.proxy,
		proxyEngines:         paths.proxyEngines,
		workloadMgrPath:      paths.workloadMgr,
		errorsPath:           paths.errors,
		engineMgrPath:        paths.engineMgr,
		engineMgrHTTPPort:    paths.engineMgrHTTPPort,
		engineMgrControlPort: paths.engineMgrControlPort,
		nodeInfoPort:         paths.nodeInfoPort,
		errorsPort:           paths.errorsPort,
		workloadMgrPort:      paths.workloadMgrPort,
		clusterMgrPort:       paths.clusterMgrPort,
		manualNodesPath:      paths.manualNodes,
		settingsPath:         paths.settings,
		clusterMgrPath:       paths.clusterMgr,
		schedulerPath:        paths.scheduler,
		clusterDir:           paths.clusterDir,
		store:                newDiscoveryStore(),
		telemetry:            newTelemetryCache(),
		relayDir:             relay.NewDirectory(),
		regCache:             relay.NewRegistrationCache(),
		manualNodeKeys:       make(map[string]string),
		manualNodeStatuses:   make(map[string]manualNodeStatusEntry),
		workloads:            workloadstore.New(),
		ollamaPortReady:      make(chan struct{}),
		lmstudioPortReady:    make(chan struct{}),
	}
}

// registerService records a local service in the discovery registration cache
// and relays it to the promoted daemon so it's advertised on this node's
// _nvpair-node record. Idempotent — only a real change is pushed. Broker-local:
// the broker registers on behalf of the workers it spawns.
func (b *Broker) registerService(p noderec.RegisterParams) {
	b.localServicesMu.Lock()
	defer b.localServicesMu.Unlock()
	if !b.regCache.Register(p) {
		return
	}
	b.localServicesGeneration++
	b.pushLocalServicesLocked()
	if sc := b.getScanner(); sc != nil {
		go sc.pushRegister(p)
	}
}

// unregisterService removes a local service from the cache and the daemon.
func (b *Broker) unregisterService(svc noderec.ServiceKey) {
	b.localServicesMu.Lock()
	defer b.localServicesMu.Unlock()
	if !b.regCache.Unregister(svc) {
		return
	}
	b.localServicesGeneration++
	b.pushLocalServicesLocked()
	if sc := b.getScanner(); sc != nil {
		go sc.pushUnregister(svc)
	}
}

// get*/set* are the workersMu-guarded accessors for the supervised worker
// handles. A supervisor swaps a handle on the monitor goroutine when it
// restarts a worker, so request handlers and the log-level fan-out must
// read through these rather than touching the fields directly.
func (b *Broker) setScanner(s *scannerProcess) {
	b.workersMu.Lock()
	b.scanner = s
	b.workersMu.Unlock()
}

func (b *Broker) getScanner() *scannerProcess {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.scanner
}

func (b *Broker) setNodeInfo(n *nodeInfoProcess) {
	b.workersMu.Lock()
	b.nodeInfo = n
	b.workersMu.Unlock()
}

func (b *Broker) getNodeInfo() *nodeInfoProcess {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.nodeInfo
}

func (b *Broker) setProxy(p *proxyProcess) {
	b.setEngineProxyHandle(ollamaProxyProfile, p)
}

func (b *Broker) getProxy() *proxyProcess {
	return b.engineProxyHandle(ollamaProxyProfile)
}

func (b *Broker) setWorkloadMgr(m *workloadManagerProcess) {
	b.workersMu.Lock()
	b.workloadMgr = m
	b.workersMu.Unlock()
}

func (b *Broker) getWorkloadMgr() *workloadManagerProcess {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.workloadMgr
}

func (b *Broker) setErrors(e *errorsProcess) {
	b.workersMu.Lock()
	b.errorsProc = e
	b.workersMu.Unlock()
}

func (b *Broker) getErrors() *errorsProcess {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.errorsProc
}

func (b *Broker) setEngineMgr(w *rpcWorker) {
	c := b.switchCoordinator()
	b.workersMu.Lock()
	if b.engineMgr != w {
		// A different worker was installed, so any source the previous pair
		// named is no longer that pair's. Only spawnEngineMgr binds again.
		b.engineMgrSource = nil
		b.wakeEngineMgrBindingLocked()
		if c.activeCancel != nil {
			b.engineMgrRecovering = true
			c.interrupted = true
			c.activeCancel()
		}
	}
	b.engineMgr = w
	b.workersMu.Unlock()
}

// bindEngineMgr records the worker spawnEngineMgr started for source. The pair
// is what a model-mutation confirmation looks up: a harness or a replacement
// installed by setEngineMgr alone has no bound source and cannot confirm one.
func (b *Broker) bindEngineMgr(worker *rpcWorker, source *engineManagerSource) {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	if worker == nil || source == nil || b.engineMgr != worker {
		return
	}
	b.engineMgrSource = source
	b.wakeEngineMgrBindingLocked()
}

// clearEngineMgrBinding drops the pair when it still holds source and wakes
// every waiter. Called after retirement, so a request read before the bind (or
// before the worker exits) fails promptly instead of waiting out its budget.
func (b *Broker) clearEngineMgrBinding(source *engineManagerSource) {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	if b.engineMgrSource == source {
		b.engineMgrSource = nil
	}
	b.wakeEngineMgrBindingLocked()
}

// engineMgrBindingSignalLocked returns the channel closed by the next bind or
// clear. It is created on first use because a test may construct a Broker
// without NewBroker.
func (b *Broker) engineMgrBindingSignalLocked() <-chan struct{} {
	if b.engineMgrBindCh == nil {
		b.engineMgrBindCh = make(chan struct{})
	}
	return b.engineMgrBindCh
}

// wakeEngineMgrBindingLocked closes the current signal and replaces it, so
// every waiter re-reads the pair. Caller holds workersMu.
func (b *Broker) wakeEngineMgrBindingLocked() {
	if b.engineMgrBindCh != nil {
		close(b.engineMgrBindCh)
	}
	b.engineMgrBindCh = make(chan struct{})
}

func (b *Broker) engineManagerExited(w *rpcWorker) {
	c := b.switchCoordinator()
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	if b.engineMgr == w {
		b.engineMgrRecovering = true
	}
	if c.activeManager == w && c.activeCancel != nil {
		c.interrupted = true
		c.activeCancel()
	}
}

func (b *Broker) getEngineMgr() *rpcWorker {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	if b.engineMgrRecovering {
		return nil
	}
	return b.engineMgr
}

// The replacement stays hidden from ordinary callers until conservative switch
// recovery has run. Cancellation above lets the active owner release its lock.
func (b *Broker) recoverReplacementEngineMgr(w *rpcWorker) bool {
	const attempts = 3
	for attempt := 0; attempt < attempts; attempt++ {
		b.workersMu.Lock()
		current := b.engineMgr == w && w != nil
		b.workersMu.Unlock()
		if !current {
			return false
		}
		select {
		case <-w.Done():
			return false
		default:
		}
		if b.recoverEngineSwitchOn(w) {
			b.workersMu.Lock()
			current = b.engineMgr == w
			select {
			case <-w.Done():
				current = false
			default:
			}
			if current {
				b.engineMgrRecovering = false
			}
			b.workersMu.Unlock()
			if current {
				b.forwardErrorsClear(engineSwitchRecoveryBlockedID)
			}
			return current
		}
		b.engineConfigMu.Lock()
		permanent := b.switchCoordinator().err != nil
		b.engineConfigMu.Unlock()
		if permanent || attempt == attempts-1 {
			b.reportEngineSwitchRecoveryBlocked()
			return false
		}
		// Recovery retries only conservative inspection/reconcile/release on the
		// pinned replacement. The ordinary manager surface remains fenced.
		timer := time.NewTimer(time.Second << attempt)
		select {
		case <-w.Done():
			timer.Stop()
			return false
		case <-timer.C:
		}
	}
	return false
}

func (b *Broker) restoreEnabledEngines(w *rpcWorker) {
	if w == nil {
		return
	}
	// An unreadable journal must not discard the component stores' enabled
	// intent. Recovery logs the failure; restore still uses the saved runtime.
	if !b.recoverEngineSwitch() {
		return
	}
	b.recoverEngineSettings()
	if b.skipRestoreWhilePaused() {
		slog.Info("node is paused; engines are restored when it resumes")
		return
	}
	if err := w.Notify(restoreEnabledEnginesMethod, nil); err != nil {
		slog.Warn("failed to request enabled-engine restoration", "err", err)
	}
}

func (b *Broker) restoreEnabledEnginesAfterPortGate(ctx context.Context) bool {
	if !b.waitForManagedPortOwnership(ctx) {
		return false
	}
	b.restoreEnabledEngines(b.getEngineMgr())
	return true
}

func (b *Broker) runEngineAvailabilityAfterPortGates(
	ctx context.Context,
	runners ...func(context.Context),
) bool {
	if !b.restoreEnabledEnginesAfterPortGate(ctx) {
		return false
	}
	for index, run := range runners {
		if index == len(runners)-1 {
			run(ctx)
			break
		}
		go run(ctx)
	}
	return true
}

func (b *Broker) setManualNodes(w *rpcWorker) {
	b.workersMu.Lock()
	b.manualNodes = w
	b.workersMu.Unlock()
}

func (b *Broker) getManualNodes() *rpcWorker {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.manualNodes
}

func (b *Broker) setSettings(w *rpcWorker) {
	b.workersMu.Lock()
	b.settings = w
	b.workersMu.Unlock()
}

func (b *Broker) getSettings() *rpcWorker {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.settings
}

func (b *Broker) setScheduler(w *rpcWorker) {
	b.workersMu.Lock()
	b.scheduler = w
	b.workersMu.Unlock()
}

func (b *Broker) getScheduler() *rpcWorker {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.scheduler
}

func (b *Broker) setClusterMgr(c *clusterManagerProcess) {
	b.peerServicesMu.Lock()
	defer b.peerServicesMu.Unlock()
	b.localServicesMu.Lock()
	defer b.localServicesMu.Unlock()
	b.workersMu.Lock()
	old := b.clusterMgr
	b.clusterMgr = c
	b.workersMu.Unlock()
	if old != nil && old != c && old.localServicesStop != nil {
		close(old.localServicesStop)
		old.localServicesStop = nil
	}
	b.pushLocalServicesLocked()
}

func (b *Broker) getClusterMgr() *clusterManagerProcess {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.clusterMgr
}

// forwardProxyProcessNotification routes one proxy process's notifications to
// the engine each belongs to.
//
// A facade-scoped notification is addressed, so it goes to that engine's
// handler with that engine's generation. A process-scoped one is unaddressed
// and must be handled exactly once, not once per engine — routing a workload
// record through both handlers would record every job twice.
func (b *Broker) forwardProxyProcessNotification(
	ollamaGeneration, lmstudioGeneration uint64, method string, params json.RawMessage,
) {
	engine, bare := engines.SplitAddressedMethod(method)
	switch engine {
	case ollamaProxyProfile.Name:
		b.forwardProxyNotificationForGeneration(ollamaGeneration, method, params)
	case lmstudioProxyProfile.Name:
		b.forwardLMStudioProxyNotificationForGeneration(lmstudioGeneration, method, params)
	case "":
		if !b.routeProcessScopedProxyNotification(bare, params) {
			slog.Debug("ignoring unaddressed proxy notification", "method", bare)
		}
	default:
		profile, known := engineProxyProfileFor(engine)
		if !known {
			profile, known = b.dynamicEngineProfile(engine)
		}
		if known {
			b.forwardDefaultEngineProxyNotification(profile, method, params)
			return
		}
		slog.Warn("proxy addressed a notification without a handler",
			"engine", engine, "method", bare, "known", known)
	}
}

// convergeScannerIdentity waits for cluster-manager to report its node id (which
// confirms identity.json is written), then tells the scanner to re-resolve its
// identity so its advertised uuid= matches the cluster principal. This closes
// the fresh-host startup-order window (scanner minted node-id before
// cluster-manager wrote identity.json); a membership change reconverges via
// applyClusterIdentityChange and the scanner's own membership watch.
// Best-effort and idempotent.
func (b *Broker) convergeScannerIdentity() {
	cm := b.getClusterMgr()
	if cm == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, _, err := cm.Call(ctx, "cluster:get-node-id", nil); err != nil {
		slog.Warn("scanner identity convergence skipped: cluster-manager node id unavailable", "err", err)
		return
	}
	if sc := b.getScanner(); sc != nil {
		sc.reloadIdentity()
	}
}

// restoreClusterIdentity re-injects the persisted cluster identity from
// nvpair-node-settings into a freshly (re)spawned cluster-manager. The
// cluster-manager persists its member roster (members.json) but NOT the
// clusterId — that lives in nvpair-node-settings and must be reflected back via
// cluster:set-identity. Without this the manager comes up unclustered after a
// restart while still reporting its saved members, so cluster:get-node-id says
// "not clustered" while nodes:get-initial lists members (and roster reconcile,
// which bails on an empty clusterId, silently stops). Runs synchronously during
// the first spawn (before app:ready / readLoop, so a client's cluster:get-node-id
// sees the restored id with no race) and on every crash respawn. No-op when
// unclustered or when settings is unavailable.
func (b *Broker) restoreClusterIdentity(cm *clusterManagerProcess) {
	settings := b.getSettings()
	if settings == nil || cm == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	id := getSettingValue(ctx, settings, "settings/get-cluster-id")
	if id == "" {
		return // unclustered: nothing to restore
	}
	name := getSettingValue(ctx, settings, "settings/get-cluster-friendly-name")
	params, _ := json.Marshal(map[string]string{"clusterId": id, "clusterFriendlyName": name})
	if _, rpcErr, err := cm.Call(ctx, "cluster:set-identity", params); err != nil || rpcErr != nil {
		slog.Warn("restore cluster identity failed", "err", err, "rpcErr", rpcErr)
		return
	}
	slog.Info("restored cluster identity into cluster-manager", "clusterId", id)
}

// getSettingValue calls a nvpair-node-settings getter that returns {"value": "..."}
// and returns the string, or "" on any error. Mirrors the settings reads in
// prepareManagedOllamaFacade.
func getSettingValue(ctx context.Context, settings *rpcWorker, method string) string {
	result, rpcErr, err := settings.Call(ctx, method, nil)
	if err != nil || rpcErr != nil {
		slog.Warn("failed to read setting", "method", method, "err", err, "rpcErr", rpcErr)
		return ""
	}
	var r struct {
		Value string `json:"value"`
	}
	if err := json.Unmarshal(result, &r); err != nil {
		slog.Warn("failed to decode setting", "method", method, "err", err)
		return ""
	}
	return r.Value
}

// persistClusterIdentity mirrors a cluster:identity-changed report from the
// cluster-manager (create / adopt-on-join / leave) into nvpair-node-settings, which
// is the durable home of the clusterId that restoreClusterIdentity re-injects on
// the next startup. Closing this loop is what makes a cluster join survive a
// restart and, crucially, what makes leaving a cluster stick — otherwise the
// stale clusterId in settings would be restored and the node would rejoin the
// cluster it just left. No-op when settings is unavailable; runs in a goroutine
// so it never blocks the cluster-manager's reader.
func (b *Broker) persistClusterIdentity(clusterID, friendlyName string) {
	settings := b.getSettings()
	if settings == nil {
		return
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		setSettingValue(ctx, settings, "settings/set-cluster-id", clusterID)
		setSettingValue(ctx, settings, "settings/set-cluster-friendly-name", friendlyName)
	}()
}

// setSettingValue calls a nvpair-node-settings setter that takes {"value": "..."},
// logging (but not surfacing) any failure — the authoritative state already
// lives in the cluster-manager; settings is the durable mirror.
func setSettingValue(ctx context.Context, settings *rpcWorker, method, value string) {
	params, _ := json.Marshal(map[string]string{"value": value})
	if _, rpcErr, err := settings.Call(ctx, method, params); err != nil || rpcErr != nil {
		slog.Warn("failed to persist setting", "method", method, "err", err, "rpcErr", rpcErr)
	}
}

// logLevelArgs is the standard --log-level argument vector at the broker's
// currently-resolved level, recomputed on each (re)spawn so a runtime
// log/set-level is reflected when a worker restarts.
func (b *Broker) logLevelArgs() []string {
	return []string{"--log-level", applog.LevelString()}
}

// clusterManagerConfigDir is the base directory the cluster-manager should own,
// derived from the broker's cluster dir (<base>/cluster). Empty when the broker
// resolved no cluster dir, which leaves the manager on its own default — the one
// case where the two can still diverge, and the reason an unresolved cluster dir
// is reported loudly at startup.
func (b *Broker) clusterManagerConfigDir() string {
	if b.clusterDir == "" {
		return ""
	}
	return filepath.Dir(b.clusterDir)
}

// clusterDirArgs returns ["--cluster-dir", dir] when a cluster dir is
// configured, else nil. Threaded into every worker that does cluster-scoped
// inter-node mTLS (nvpair-errors, nvpair-workload-manager, nvpair-node-scanner,
// nvpair-manual-nodes, nvpair-engine-manager) so each serves/dials pinned cluster
// peers once this node is clustered; absent it, those workers stay on plain HTTP
// (or, for engine-manager's ec surface, don't bind it at all). nvpair-node-info is
// deliberately excluded — it stays plain HTTP even on a clustered node (see the
// spawnNodeInfo call site), so it is never passed --cluster-dir.
func (b *Broker) clusterDirArgs() []string {
	if b.clusterDir == "" {
		return nil
	}
	return []string{"--cluster-dir", b.clusterDir}
}

// resolveLocalNodeID resolves this node's stable per-host UUID from the same
// on-disk identity the node-scanner and cluster-manager use (nodeid.Resolve
// prefers cluster/identity.json, then node-id.json, else mints and persists a
// fresh one), so the broker's originatedFrom/errors stamp matches this node's
// advertised hostUuid. The base is the parent of the cluster dir (which is
// <base>/cluster); an empty cluster dir selects nodeid's shared default. There
// is no hostname fallback: nodeid.Resolve returns empty only if the OS CSPRNG
// is unavailable, and a node with no stable identity must not run (it would
// stamp inconsistently across the cluster), so we fail loudly instead.
func resolveLocalNodeID(clusterDir string) string {
	base := ""
	if clusterDir != "" {
		base = filepath.Dir(clusterDir)
	}
	id := nodeid.Resolve(base)
	if id == "" {
		log.Fatal("could not resolve a stable node UUID (system CSPRNG unavailable); refusing to start without a node identity")
	}
	return id
}

// applyClusterIdentityChange converges this node's advertisement after the
// cluster-manager reports cluster:identity-changed (this node created or joined a
// cluster, or left one).
//
// It deliberately does NOT restart the cluster-scoped workers. Each of them holds
// a live view of the cluster dir (nvpair-shared/clustertrust) and re-derives its
// own membership, so it changes personality in place: the proxies' LAN mTLS
// ingress, nvpair-errors' and nvpair-workload-manager's inter-node interfaces,
// engine-manager's remote-control surface, and manual-node probes all follow a
// join or leave without a new process. Restarting them instead was both
// disruptive (it recycled inference ingress and discovery on every membership
// change) and unreliable: it was a single unverified attempt whose outcome
// depended on the workers reading a cluster dir the manager was writing at that
// instant, and a worker that read it a moment too early stayed unclustered for
// the rest of its life — in the roster and ranked by the scheduler, yet
// exchanging no cluster traffic.
//
// Two workers are told directly, because they publish this node's membership
// rather than deriving it. The scanner owns the mDNS record: cluster-uuid= is how
// a peer finds the pin for this node, so it should flip now rather than at the
// scanner's next membership poll. node-info carries the same fact over plain
// HTTP, which is a peer's only way to learn it when this node's mDNS record does
// not reach it — and it holds no cluster dir, so the value has to be pushed. Both
// are idempotent and run on their own goroutine, because they are
// bounded-but-blocking writes and the caller is the cluster-manager's reader.
func (b *Broker) applyClusterIdentityChange() {
	if b.clusterDir == "" {
		return
	}
	slog.Info("cluster identity changed; converging this node's advertised cluster identity")
	if sc := b.getScanner(); sc != nil {
		go sc.reloadIdentity()
	}
	go b.pushClusterIdentityToNodeInfo()
}

// clusterPrincipal is the cluster principal this node currently holds, or empty
// when it belongs to no cluster. It is read from the trust store on each call
// rather than cached: membership changes under the broker while it runs, and this
// is the value peers key their pins on.
func (b *Broker) clusterPrincipal() string {
	if b.clusterDir == "" {
		return ""
	}
	mesh := clustertrust.Open(b.clusterDir)
	mesh.Refresh()
	if !mesh.Clustered() {
		return ""
	}
	return mesh.NodeUUID()
}

// pushClusterIdentityToNodeInfo sends node-info the current cluster principal.
// Best-effort on a node-info that isn't running (never spawned, or mid-restart):
// the next spawn pushes again. A failed write is warned about rather than traced,
// because until the next push lands peers cannot learn this node's membership
// over HTTP — which is the whole point of reporting it.
func (b *Broker) pushClusterIdentityToNodeInfo() {
	np := b.getNodeInfo()
	if np == nil {
		return
	}
	if err := np.SetClusterIdentity(b.clusterPrincipal()); err != nil {
		slog.Warn("failed to push cluster identity to node-info", "err", err)
	}
}

// enginePeerRelayNotification reports whether method is a relay notification
// engine-manager forwards for a paired node. Only the worker currently bound as
// this broker's engine manager may serve or answer one: a retired worker's
// request is stale and must not reach a broker-owned writer.
func enginePeerRelayNotification(method string) bool {
	switch method {
	case policyRelayRequestMethod, policyRelayCancelMethod,
		"engine:settings-request", "engine:settings-cancel",
		"engine:switch-request", "engine:switch-cancel":
		return true
	}
	return false
}

// forwardEngineNotificationFrom handles the bound engine-manager's notifications.
// Identity and running-state observations are fenced by its source; control-plane
// requests and subscribed client notifications follow their respective relays.
func (b *Broker) forwardEngineNotificationFrom(source *engineManagerSource, method string, params json.RawMessage) {
	if method == residentDispatchRequestMethod {
		b.dispatchResidentFrom(source, params)
		return
	}
	if method == engineModelIdentityMethod {
		defer b.signalModelConfig()
	}
	if enginePeerRelayNotification(method) && !b.engineManagerSourceCurrent(source) {
		slog.Debug("dropped a peer relay notification from a retired engine manager", "method", method)
		return
	}
	if method == nodepolicy.MethodModelMutationRequest {
		// Internal to engine-manager and the broker: answered below, never
		// forwarded to engine:subscribe'd clients.
		b.observeModelMutationRequest(source, params)
		return
	}
	if method == engineModelIdentityMethod {
		b.observeModelIdentity(source, params)
		return
	}
	if method == engineStateChangedMethod {
		var st struct {
			Engine  string `json:"engine"`
			Running bool   `json:"running"`
		}
		if json.Unmarshal(params, &st) == nil && st.Engine != "" {
			if !b.observeEngineRunningFrom(source, st.Engine, st.Running) {
				return
			}
		}
	}
	if b.handleSwitchEngineNotification(source, method, params) {
		return
	}
	if b.handlePolicyEngineNotification(source, method, params) {
		return
	}
	if method == "engine:settings-request" {
		b.handleSettingsRelay(source, params)
		return
	}
	if method == "engine:settings-cancel" {
		var p struct {
			ID string `json:"id"`
		}
		if json.Unmarshal(params, &p) == nil {
			b.settingsRelayMu.Lock()
			cancel := b.settingsCancels[p.ID]
			b.settingsRelayMu.Unlock()
			if cancel != nil {
				cancel()
			}
		}
		return
	}
	if b.dispatchErrorsNotif("engine-manager", method, params) {
		return
	}
	if method == "engine:ready" {
		b.reconcileLMStudioProxyAfterEngineManagerReady()
		// A (re)started engine-manager holds no node projection until it is
		// sent one. Off the read pump, like the settings projection below.
		go b.publishNodeProjection()
		go b.replayLoadedEngineSettings()
	}
	if method == noderec.MethodSubscribe {
		// engine-manager subscribes upward for its ec peer set (nodes exposing
		// the remote-control surface): wire it to the relay directory and push
		// discovery:nodes snapshots down its stdin, exactly like nvpair-errors.
		b.handleEngineSubscribe(params)
		return
	}
	// engine:ready / engine:state-changed / engine:install-progress /
	// engine:pull-progress are re-emitted verbatim to engine:subscribe'd clients
	// (their method names are already engine:-prefixed, so no translation). Off
	// by default.
	b.engineMu.Lock()
	subscribed := b.engineSubscribed
	b.engineMu.Unlock()
	if !subscribed {
		return
	}
	if err := b.codec.Notify(method, params); err != nil {
		slog.Warn("forward engine notification failed", "method", method, "err", err)
	}
}

// handleEngineSubscribe wires engine-manager's discovery:subscribe into the
// relay directory: it registers a subscriber whose Send pushes a
// discovery:nodes snapshot down engine-manager's stdin, then delivers the
// initial snapshot. A re-subscribe drops the prior registration first so it
// isn't double-fed; worker exit clears it (see the engine-manager onExit hook).
func (b *Broker) handleEngineSubscribe(params json.RawMessage) {
	if b.relayDir == nil {
		return
	}
	// The manager's own directory subscription must work during recovery;
	// only forward operations are hidden by getEngineMgr.
	b.workersMu.Lock()
	em := b.engineMgr
	b.workersMu.Unlock()
	if em == nil {
		return
	}
	send := func(nodes []noderec.DirectoryNode) {
		if err := em.Notify(noderec.NotifyNodes, noderec.GetNodesResult{Nodes: nodes}); err != nil {
			slog.Debug("failed to push node snapshot to engine-manager", "err", err)
		}
	}
	b.engineRelayMu.Lock()
	if b.engineRelaySubID != 0 {
		b.relayDir.Unsubscribe(b.engineRelaySubID)
	}
	id, sub, err := subscribeRelay(b.relayDir, params, send)
	b.engineRelaySubID = id
	b.engineRelayMu.Unlock()
	if err != nil {
		slog.Warn("engine-manager sent invalid discovery:subscribe", "err", err)
		return
	}
	b.relayDir.Deliver(sub)
}

// clearEngineRelaySub drops engine-manager's relay subscription. Called from its
// supervisor onExit so a crashed/stopped worker doesn't leave a dangling
// subscriber pushing to a closed stdin.
func (b *Broker) clearEngineRelaySub() {
	b.engineRelayMu.Lock()
	if b.engineRelaySubID != 0 && b.relayDir != nil {
		b.relayDir.Unsubscribe(b.engineRelaySubID)
		b.engineRelaySubID = 0
	}
	b.engineRelayMu.Unlock()
}

func (b *Broker) forwardManualNodesNotification(method string, params json.RawMessage) {
	if b.dispatchErrorsNotif("manual-nodes", method, params) {
		return
	}
	// Merge manual nodes into the same discovery store the scanner feeds,
	// so manual + mDNS nodes surface on one discovery:get-nodes /
	// discovery:nodes-changed snapshot.
	switch method {
	case "node/discovered", "node/updated":
		var s manualNodeStatus
		if err := json.Unmarshal(params, &s); err != nil {
			slog.Warn("manual-nodes emitted invalid node payload", "method", method, "err", err)
			return
		}
		if s.ID == "" {
			slog.Warn("manual-nodes node payload missing id", "method", method)
			return
		}
		b.upsertManualNode(s)
		slog.Debug("manual node event", "event", method, "id", s.ID)
	case "node/removed":
		var s manualNodeStatus
		if err := json.Unmarshal(params, &s); err != nil {
			slog.Warn("manual-nodes emitted invalid removed payload", "err", err)
			return
		}
		b.removeManualNode(s.ID)
		slog.Debug("manual node event", "event", method, "id", s.ID)
	case "ready":
		slog.Info("manual-nodes reported ready", "params", string(params))
	default:
		slog.Debug("ignoring manual-nodes notification", "method", method)
	}
}

// trackManualNode records the discovery-store key a manual node currently
// occupies and reports the previous key when it changed (a manual node's
// node-info first reported its real hostUuid, moving it off the manual id). On a
// rekey it drops the manual claim under the old key (the scanner's claim, if
// any, survives); the caller rebridges the proxy overlay off the old key.
// upsertManualNode ingests a manual node status: it records the alias's payload,
// keys the discovery store by the node's operational identity (its hostUuid once
// node-info reports one, else the manual id), and bridges the proxy candidate
// under that same key so scheduler priority and scheduledOn resolve to it. If
// the alias's key changed (node-info revealed its real UUID) the
// old key is reprojected from a surviving alias or released.
func (b *Broker) upsertManualNode(s manualNodeStatus) {
	en := manualToEnriched(s)
	key := en.storeKey()
	receivedAt := time.Now()

	b.manualMu.Lock()
	oldKey, existed := b.manualNodeKeys[s.ID]
	b.manualNodeKeys[s.ID] = key
	b.manualNodeStatuses[s.ID] = manualNodeStatusEntry{status: s, receivedAt: receivedAt}
	b.manualMu.Unlock()

	b.store.Upsert(en, sourceManual)
	b.ingestTelemetryAt(sourceManual, manualNodeTelemetry(s, key), receivedAt)
	// Bridge a reachable manual node into each engine's proxy (ollama-proxy /
	// lmstudio-proxy) so inference can route to it; an unreachable engine is
	// pulled back out. No-op for a proxy the broker doesn't supervise.
	b.bridgeManualNode(s, key)

	if existed && oldKey != key {
		// This alias moved off oldKey: refresh oldKey from a surviving alias, or
		// release its store/proxy claim if this was its last owner.
		b.reprojectOrRelease(oldKey)
	}
}

// removeManualNode forgets a manual alias, then reprojects or releases the key
// it occupied. When another alias for the same machine still owns the key, the
// node stays and is reprojected from that survivor's last-seen payload — so the
// removed alias's address/models don't linger (the survivor's steady-state probe
// emits no update to restore itself); only the final alias's removal drops the
// store/proxy claim.
func (b *Broker) removeManualNode(id string) {
	b.manualMu.Lock()
	key, ok := b.manualNodeKeys[id]
	delete(b.manualNodeKeys, id)
	delete(b.manualNodeStatuses, id)
	b.manualMu.Unlock()
	if !ok {
		return
	}
	b.reprojectOrRelease(key)
}

// reprojectOrRelease refreshes or drops a key's manual claim after an alias
// stopped pointing at it. If another manual alias still resolves to key, that
// survivor's payload is re-projected into the store's single manual slot and its
// proxy candidate rebridged; otherwise the store and proxy manual claim is
// released.
func (b *Broker) reprojectOrRelease(key string) {
	if key == "" {
		return
	}
	b.manualMu.Lock()
	survivor, ok := b.survivingAliasLocked(key)
	b.manualMu.Unlock()
	if ok {
		b.store.Upsert(manualToEnriched(survivor.status), sourceManual)
		b.bridgeManualNode(survivor.status, key)
		b.ingestTelemetryAt(sourceManual, manualNodeTelemetry(survivor.status, key), survivor.receivedAt)
		return
	}
	b.store.Remove(key, sourceManual)
	b.removeManualNodeFromProxies(key)
	b.removeTelemetry(sourceManual, key)
}

// survivingAliasLocked returns the last-seen status of a manual alias still
// resolving to key, picking the lexicographically smallest alias id for a stable
// choice. Callers must hold manualMu.
func (b *Broker) survivingAliasLocked(key string) (manualNodeStatusEntry, bool) {
	best := ""
	for id, k := range b.manualNodeKeys {
		if k == key && (best == "" || id < best) {
			best = id
		}
	}
	if best == "" {
		return manualNodeStatusEntry{}, false
	}
	return b.manualNodeStatuses[best], true
}

// clearManualNodesState is the manual-nodes supervisor's clearHandle: on a
// crash it drops the worker handle and evicts every manual-origin node from
// the discovery store. The restarted process comes up with no entries (it
// keeps no persistent state and the broker doesn't re-feed them), so
// leaving the old manual nodes in the snapshot would strand stale entries
// that never age out. Clients re-add manual nodes after a restart.
func (b *Broker) clearManualNodesState() {
	b.setManualNodes(nil)
	b.manualMu.Lock()
	keys := b.manualNodeKeys
	b.manualNodeKeys = make(map[string]string)
	b.manualNodeStatuses = make(map[string]manualNodeStatusEntry)
	b.manualMu.Unlock()
	seen := make(map[string]bool, len(keys))
	for _, key := range keys {
		// Aliases can share a key; drop each unique claim once.
		if seen[key] {
			continue
		}
		seen[key] = true
		// Drop only the manual claim; a co-located scanner node keeps its record.
		b.store.Remove(key, sourceManual)
		b.removeTelemetry(sourceManual, key)
		// Pull the now-orphaned node out of every proxy too, so inference
		// doesn't keep a stale manual target the crashed prober can no
		// longer vouch for. Clients re-add manual nodes after the restart.
		b.removeManualNodeFromProxies(key)
	}
}

func (b *Broker) forwardSettingsNotification(method string, params json.RawMessage) {
	if b.dispatchErrorsNotif("settings", method, params) {
		return
	}
	// connection/cluster-identity and connection/cluster-auto-sync are the
	// change-only cluster signals nvpair-node-settings pushes. Re-emit them
	// verbatim and unconditionally — the same way the UI receives them from
	// the settings subprocess today (no opt-in subscribe for these).
	switch method {
	case "connection/cluster-identity", "connection/cluster-auto-sync":
		if err := b.codec.Notify(method, params); err != nil {
			slog.Warn("forward settings notification failed", "method", method, "err", err)
		}
	default:
		slog.Debug("ignoring settings notification", "method", method)
	}
}

// forwardSchedulerNotification is the hook startRPCWorker invokes on the
// scheduler's reader goroutine for every notification it emits. Its errors:*
// reports go into the pipeline like every other producer's; schedule:priority
// is cached and fanned out to the matching proxy via node/set-priority.
func (b *Broker) forwardSchedulerNotification(method string, params json.RawMessage) {
	if b.dispatchErrorsNotif("scheduler", method, params) {
		return
	}
	if method == "schedule:priority" {
		var p schedulerwire.EnginePriority
		if err := json.Unmarshal(params, &p); err != nil {
			slog.Warn("bad schedule:priority payload", "err", err)
			return
		}
		generation, fresh := b.cachePrioritySnapshot(p.Snapshot())
		if !fresh {
			// The scheduler's duplicate emission of one recompute. Dropped here,
			// before it mints a generation, so nothing downstream sees it at all.
			return
		}
		// Dispatch off the reader goroutine: the node/set-priority round-trip
		// blocks on the proxy's response, so calling it inline would stall the
		// scheduler's notification stream.
		go b.applyPriority(generation)
		return
	}
	slog.Debug("ignoring scheduler notification", "method", method)
}

// forwardNodeInfoNotification is the hook startNodeInfo invokes on node-info's
// reader goroutine for every notification it emits. Its observed local addresses
// go to the scanner, which owns what this node advertises: node-info learns which
// addresses peers actually reach us on, and that is the only direct evidence
// address selection can have about reachability from somewhere else.
func (b *Broker) forwardNodeInfoNotification(method string, params json.RawMessage) {
	if method != noderec.NotifyObservedAddresses {
		slog.Debug("ignoring node-info notification", "method", method)
		return
	}
	var p noderec.ObservedAddressesParams
	if err := json.Unmarshal(params, &p); err != nil {
		slog.Warn("bad observed-addresses payload", "err", err)
		return
	}
	sp := b.getScanner()
	if sp == nil {
		return
	}
	// Off the reader goroutine: the relay waits on the scanner's response, and
	// blocking here would stall node-info's stdout.
	go sp.pushObservedAddresses(p.Addresses)
}

// cachePrioritySnapshot records a new ranking and returns its generation, or
// reports fresh=false when the ranking is byte-identical to the cached one.
//
// The cache is process-wide, not per engine, and that is what makes the dedupe
// work: one scheduler recompute emits the same ranking once per engine
// (schedule.go computes one order and loops the engine list), so the second
// frame is a duplicate that differs only in a routing key the proxy never sees.
// Comparing per engine would never match those two against each other.
//
// Dropping the duplicate here, before it mints a generation, is deliberate.
// The alternative — comparing content at delivery — silently breaks
// repushPriority, whose whole job is to re-send an unchanged ranking to a proxy
// that lost its state, and which is therefore always byte-identical to the last
// delivery.
func (b *Broker) cachePrioritySnapshot(priority schedulerwire.Priority) (generation uint64, fresh bool) {
	b.schedMu.Lock()
	defer b.schedMu.Unlock()

	if b.havePriority && b.lastPriority.SameRanking(priority) {
		return b.priorityGeneration, false
	}
	b.lastPriority = priority.Clone()
	b.havePriority = true
	b.priorityGeneration++
	b.lastPriority.Generation = b.priorityGeneration
	return b.priorityGeneration, true
}

// applyPriority delivers one cached generation to every live proxy. A stale
// generation is silently skipped; an engine with no proxy is a logged no-op and
// picks the snapshot up from the replay on its next spawn.
//
// It fans out rather than targeting one engine because the ranking is node-wide
// and the cache is now process-wide: with the duplicate emission dropped at
// ingestion, the surviving frame is whichever engine the scheduler emitted
// first, so delivering only to that engine's proxy would starve the other.
func (b *Broker) applyPriority(generation uint64) {
	b.deliverPrioritySnapshot(generation, func(priority schedulerwire.Priority) {
		b.sendPriority(context.Background(), priority)
	})
}

// sendPriority delivers one ranking to every live proxy. Shared by the
// scheduler's gated fan-out and the ungated spawn replay, so both agree on how
// a delivery is made and only differ on when one is allowed.
func (b *Broker) sendPriority(ctx context.Context, priority schedulerwire.Priority) {
	params, err := json.Marshal(priority)
	if err != nil {
		slog.Warn("marshal node/set-priority failed", "err", err)
		return
	}
	for _, target := range b.livePriorityTargets() {
		if _, rpcErr, err := target.proxy.Call(ctx, "node/set-priority", params); err != nil {
			slog.Warn("node/set-priority call failed", "engine", target.engine, "err", err)
		} else if rpcErr != nil {
			slog.Warn("node/set-priority rejected", "engine", target.engine,
				"code", rpcErr.Code, "msg", rpcErr.Message)
		}
	}
}

// priorityTarget is one proxy to deliver a ranking to, with the engine name
// kept for log context.
type priorityTarget struct {
	engine string
	proxy  *proxyProcess
}

// livePriorityTargets returns the distinct live proxies to deliver a ranking to.
//
// Distinct by handle, not by engine: once one process hosts every facade, two
// engine entries resolve to the same *proxyProcess, and calling it once per
// engine would deliver the same generation twice. The child treats the repeat as
// an acknowledged no-op, but sending it is still pointless work on the request
// path's critical peer.
func (b *Broker) livePriorityTargets() []priorityTarget {
	var out []priorityTarget
	seen := make(map[*proxyProcess]bool)
	for _, profile := range engineProxyProfiles {
		p := b.engineProxyHandle(profile)
		if p == nil {
			slog.Debug("no proxy for engine; ranking arrives on its next spawn", "engine", profile.Name)
			continue
		}
		if seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, priorityTarget{engine: profile.Name, proxy: p})
	}
	// A process with no facade still takes the ranking and the log level.
	if p := b.proxyProcessHandle(); p != nil && !seen[p] {
		out = append(out, priorityTarget{engine: engines.ProxyComponent, proxy: p})
	}
	return out
}

// deliverPrioritySnapshot serializes deliveries and invokes deliver only if
// generation is still current. It is split from the proxy call so ordering can
// be verified without a subprocess.
func (b *Broker) deliverPrioritySnapshot(generation uint64, deliver func(schedulerwire.Priority)) {
	b.priorityApplyMu.Lock()
	defer b.priorityApplyMu.Unlock()

	b.schedMu.Lock()
	if b.priorityGeneration != generation {
		b.schedMu.Unlock()
		return
	}
	priority := b.lastPriority.Clone()
	b.schedMu.Unlock()

	deliver(priority)
}

// replayProxyStateAfterSpawn restores the broker-pushed state a fresh proxy
// process does not have. It is wired to every proxy supervisor's onSpawned,
// which is the only event that loses that state — a set-port rebind keeps the
// child's in-process state, so it needs no replay.
//
// The spawn hook rather than the proxy's own "ready" notification, for a
// specific reason: spawn publishes the handle before returning, so a replay
// from here is guaranteed to find it. A replay triggered by "ready" races that
// publication, because the child can emit ready before the broker has stored
// the handle — and a replay that misses the handle is dropped by the generation
// check and never retried, leaving the new child with no ranking at all.
func (b *Broker) replayProxyStateAfterSpawn() {
	// Abandoned once teardown starts, which the onSpawned field comment
	// requires of any callback that talks to the worker: this one calls the
	// proxy, and a teardown is about to close that worker's stdin. It runs on
	// its own goroutine so it cannot stall Stop, but there is no reason to
	// spend a call timeout re-seeding a child that is going away.
	ctx, cancel := b.proxyBringUpContext()
	defer cancel()
	b.repushPriority(ctx)
}

// repushPriority re-sends the cached ranking after a proxy (re)spawns, because
// a fresh child starts with no scheduler state. A no-op when nothing has been
// cached yet.
//
// It reuses the current generation rather than minting one, so the child's
// idempotency check makes a repush to a proxy that never lost its state
// harmless — which is what lets several spawn paths call this without
// coordinating.
func (b *Broker) repushPriority(ctx context.Context) {
	// Serialized against the scheduler's own deliveries, but deliberately not
	// gated on a generation.
	//
	// The gate stops an older generation from overwriting a newer delivery,
	// which is right for the scheduler's fan-out and wrong here. A replay races
	// a snapshot that arrived while no handle was published: that snapshot found
	// no targets, and gating the replay on the generation it read then discards
	// it as stale, so neither is delivered. The proxy is left with an empty
	// ranking, reserveCandidate returns early, and no reservations are taken at
	// all — the behaviour the shared map exists to provide — until the next
	// genuine ranking change.
	//
	// Unconditional is cheap: the child applies at most one snapshot per
	// generation, so a repush to a proxy that never lost its state costs one
	// ignored call.
	b.priorityApplyMu.Lock()
	defer b.priorityApplyMu.Unlock()

	b.schedMu.Lock()
	have := b.havePriority
	priority := b.lastPriority.Clone()
	b.schedMu.Unlock()
	if !have {
		return
	}
	b.sendPriority(ctx, priority)
}

// onErrorsUpdate relays an errors:update snapshot from nvpair-errors straight
// to the connected client. nvpair-errors pushes the full sorted ServiceError
// list on every change; following the established error-reporting protocol
// the broker forwards it verbatim and unconditionally (no opt-in
// subscribe), allowing the bundled UI or another client to fan errors:update
// into its presentation layer. Invoked on the nvpair-errors reader
// goroutine.
func (b *Broker) onErrorsUpdate(params json.RawMessage) {
	if err := b.codec.Notify(methodErrorsUpdate, params); err != nil {
		slog.Warn("relay errors:update failed", "err", err)
	}
}

// Serve runs the JSON-RPC loop until the peer disconnects, the context is
// cancelled, or a "shutdown" request is received. The returned error is
// nil on graceful close; a non-nil error indicates an unrecoverable
// transport problem or a failure to spawn the scanner.
//
// Ordering matters: the scanner is spawned BEFORE "app:ready" is
// emitted so the client can assume, once it sees app:ready, that
// discovery events are already flowing into the broker's store (even
// if no nodes have been found yet — mDNS browsing takes a moment to
// populate).
func (b *Broker) Serve(ctx context.Context) error {
	if err := validateServicePorts(b.servicePorts()); err != nil {
		return fmt.Errorf("service ports: %w", err)
	}
	ctx, cancel := context.WithCancel(ctx)
	b.cancel = cancel
	b.setServeCtx(ctx)
	defer cancel()
	stopPerformance := b.startPerformance(ctx)
	defer stopPerformance()

	// Enable durable workload history: load the prior session's terminal
	// records so a late-subscribing client / workloads:get-initial has a
	// baseline, then start the coalescing flusher (which also flushes on
	// ctx cancellation). Best-effort — an unresolved data dir or a bad file
	// leaves the store purely in-memory; it never blocks startup. The returned
	// stop func is deferred so the flusher's shutdown flush is JOINED before
	// Serve returns — otherwise the process could exit after cancel() but before
	// the goroutine's final flush, losing a workload that finished since the
	// last periodic flush. Registered after the top-level defer cancel(), so it
	// runs first (LIFO): late terminal events from workers shutting down are
	// still captured, then the flusher is cancelled and joined.
	if p, err := appdir.Path("workloads-history.json"); err != nil {
		slog.Warn("could not resolve workload history path; running in-memory only", "err", err)
	} else if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		slog.Warn("could not create data dir for workload history; running in-memory only", "err", err)
	} else {
		b.workloads.WithPersistence(p)
		if err := b.workloads.Load(); err != nil {
			slog.Warn("failed to load persisted workload history", "path", p, "err", err)
		}
		stopFlusher := b.runWorkloadHistoryFlusher(ctx)
		defer stopFlusher()
	}

	// Bound how long a lost terminal event can keep a remote workload displaying
	// as in-flight. Independent of persistence above: it guards the live set.
	go b.runStaleWorkloadSweep(ctx)
	go b.refreshEngineSettings(ctx)

	// The node policy is loaded before engine-manager and the proxy start: the
	// proxy is handed it before it can admit anything, and a paused node must
	// not have its engines restored.
	if path, err := defaultNodePolicyPath(); err != nil {
		slog.Warn("could not resolve the node policy path; using the default policy in memory", "err", err)
	} else {
		b.loadNodePolicy(path)
	}
	go b.runIdlePolicy(ctx)
	stopEgressRenewal := b.startEgressRenewal(ctx)
	defer stopEgressRenewal()

	// The scanner is the broker's core worker. Its supervisor's first
	// spawn is synchronous so a hard startup failure stays fatal — the
	// broker has no job without discovery. Once up, the supervisor watches
	// it and (when auto-restart is enabled) respawns it on an unexpected
	// exit instead of leaving the discovery snapshot frozen.
	b.scannerSup = newSupervisor("scanner", defaultRestartPolicy(), b.spawnScanner)
	b.scannerSup.onCrash, b.scannerSup.onRecovered = b.supervisedWorkerCallbacks("scanner", func() {
		b.setScanner(nil)
		b.clearScannerErrors()
	})
	if err := b.scannerSup.Start(); err != nil {
		return fmt.Errorf("start scanner: %w", err)
	}
	defer b.scannerSup.Stop()

	// nvpair-errors is the service-error datastore and the backbone of the
	// error-surfacing pipeline: producers' errors:report / errors:clear
	// are forwarded into it and its errors:update snapshots are relayed to
	// the client. Spawn it early (right after the scanner, before the
	// other producers) so a sink exists by the time they start emitting.
	// It runs with --peer-sync for cross-node error propagation. Like the
	// other auxiliaries a failed first
	// spawn (or never-resolved binary) is non-fatal — the broker logs and
	// runs with the pipeline disabled (producer errors are dropped).
	if b.errorsPath != "" {
		b.errorsSup = newSupervisor("errors", defaultRestartPolicy(), b.spawnErrors)
		b.errorsSup.onCrash = b.errorsCrashFallback
		b.errorsSup.onRecovered = func() { slog.Info("nvpair-errors recovered") }
		if err := b.errorsSup.Start(); err != nil {
			slog.Warn("nvpair-errors failed to start; continuing with the error pipeline disabled", "path", b.errorsPath, "err", err)
			b.errorsSup = nil
		} else {
			defer b.errorsSup.Stop()
		}
	} else {
		slog.Info("nvpair-errors path not resolved; running with the error pipeline disabled")
	}

	// Settings and engine-manager must be available before either proxy starts.
	// Managed ownership prepares Ollama's deferred move and lets engine-manager
	// safely move an identified LM Studio command runtime. Unknown owners are
	// never terminated; preparation instead selects an explicit fallback.
	b.settingsSup = b.startOptionalWorker("settings", b.settingsPath, b.spawnSettings, func() { b.setSettings(nil) })
	if b.settingsSup != nil {
		defer b.settingsSup.Stop()
	}
	b.prepareOllamaHostAliasCandidate()
	b.engineMgrSup = b.startOptionalWorker("engine-manager", b.engineMgrPath, b.spawnEngineMgr, func() { b.setEngineMgr(nil); b.clearEngineRelaySub() })
	if b.engineMgrSup != nil {
		defer b.engineMgrSup.Stop()
	}
	b.migrateLegacyEngineSettings()
	b.loadDynamicEngines()
	b.prepareEnabledFacades()

	// node-info is an auxiliary worker: spawning it lets the broker's own
	// host advertise its hardware inventory on the network, but it is NOT
	// essential to the broker's core job (serving the
	// discovery snapshot). A spawn failure — or a never-resolved binary
	// (empty path) — is therefore non-fatal: we log and carry on rather
	// than tearing down discovery.
	if b.nodeInfoPath != "" {
		b.nodeInfoSup = newSupervisor("node-info", defaultRestartPolicy(), b.spawnNodeInfo)
		b.nodeInfoSup.onCrash, b.nodeInfoSup.onRecovered = b.supervisedWorkerCallbacks("node-info", func() { b.setNodeInfo(nil) })
		if err := b.nodeInfoSup.Start(); err != nil {
			slog.Warn("node-info failed to start; continuing without local node advertisement", "path", b.nodeInfoPath, "err", err)
			b.nodeInfoSup = nil
		} else {
			defer b.nodeInfoSup.Stop()
		}
	} else {
		slog.Info("node-info path not resolved; running without local node advertisement")
	}

	// nvpair-proxy is another auxiliary worker: one process that runs a local
	// reverse proxy per enabled engine and exposes a control plane the broker
	// relays to clients. Like node-info, a spawn failure or never-resolved
	// binary is non-fatal. It does NOT gate app:ready: each facade announces
	// its listen port asynchronously via a "ready" notification (after its HTTP
	// bind), so clients learn the port by polling <engine>-proxy:get-status
	// rather than by the presence of app:ready.
	//
	// One supervisor for every engine. That is what puts the estimated-work
	// reservations of all facades in one place, which is the point of the
	// unified proxy; the cost is shared fate, so a crash takes every engine's
	// facade down and the supervisor brings them all back together.
	//
	// It also runs for the routing listener alone, which the persisted policy
	// enables independently of any engine: see startProxySupervision.
	anyProxyEnabled := false
	for _, profile := range engineProxyProfiles {
		if b.proxyEnabled(profile) {
			anyProxyEnabled = true
			continue
		}
		reason, report := proxyDisabledReason(b.proxyPath)
		slog.Info("engine proxy not enabled; running without one",
			"engine", profile.Name, "reason", reason)
		// Preparation was skipped for a disabled engine, so managedFacade is
		// necessarily false and there is no facade to block. Surface the
		// unavailability directly instead, when it is worth surfacing.
		if report {
			b.reportProxyUnavailable(profile, reason)
		}
		b.finishEngineProxyStartup(profile)
	}
	b.startProxySupervision(anyProxyEnabled || len(b.dynamicEngineProfiles()) > 0)
	defer func() {
		stopEgressRenewal()
		b.shutdownProxySupervision()
	}()

	// Restore engines and begin advertising only after both managed proxy
	// startup attempts have established either readiness or a terminal outcome.
	// This prevents a restored engine from taking a persisted proxy port before
	// the broker can resolve ownership.
	availabilityRunners := []func(context.Context){b.runAutoAdvertise, b.runAutoAdvertiseLMStudio}
	for _, profile := range engineProxyProfiles {
		if !b.proxyEnabled(profile) {
			continue
		}
		if profile.Name == ollamaProxyProfile.Name || profile.Name == lmstudioProxyProfile.Name {
			continue
		}
		profile := profile
		availabilityRunners = append(availabilityRunners, func(ctx context.Context) {
			b.runAutoAdvertiseEngine(ctx, profile)
		})
	}
	availabilityRunners = append(availabilityRunners, b.dynamicEngineAvailabilityRunners()...)
	go b.runEngineAvailabilityAfterPortGates(ctx, availabilityRunners...)

	// nvpair-workload-manager is another auxiliary worker: it relays local
	// workload lifecycle events to peer nodes and surfaces peer events
	// back. Like node-info and the proxy, a spawn failure or never-resolved
	// binary is non-fatal — the broker simply runs without cluster workload
	// relay. It does NOT gate app:ready: it has no readiness handshake the
	// broker waits on, and workload traffic only flows once the proxy emits
	// events.
	if b.workloadMgrPath != "" {
		b.workloadMgrSup = newSupervisor("workload-manager", defaultRestartPolicy(), b.spawnWorkloadManager)
		b.workloadMgrSup.onCrash, b.workloadMgrSup.onRecovered = b.supervisedWorkerCallbacks("workload-manager", func() { b.setWorkloadMgr(nil) })
		if err := b.workloadMgrSup.Start(); err != nil {
			slog.Warn("workload-manager failed to start; continuing without cluster workload relay", "path", b.workloadMgrPath, "err", err)
			b.workloadMgrSup = nil
		} else {
			defer b.workloadMgrSup.Stop()
		}
	} else {
		slog.Info("workload-manager path not resolved; running without cluster workload relay")
	}

	// nvpair-manual-nodes and nvpair-cluster-manager are the remaining auxiliary
	// workers. Each is a
	// bidirectional JSON-RPC peer run under a supervisor (auto-restart +
	// crash surfacing); the broker demuxes their errors:* notifications
	// into the pipeline and relays their control-plane requests.
	b.manualNodesSup = b.startOptionalWorker("manual-nodes", b.manualNodesPath, b.spawnManualNodes, b.clearManualNodesState)
	if b.manualNodesSup != nil {
		defer b.manualNodesSup.Stop()
	}
	// nvpair-cluster-manager owns the node's cryptographic identity, the
	// trusted-node store, and the PIN-based pairing flow, and answers
	// cluster:* / nodes:* requests the broker relays.
	b.clusterMgrSup = b.startOptionalWorker("cluster-manager", b.clusterMgrPath, b.spawnClusterManager, func() { b.setClusterMgr(nil) })
	if b.clusterMgrSup != nil {
		defer b.clusterMgrSup.Stop()
	}
	// nvpair-job-scheduler ranks the discovered nodes least-loaded-first and emits
	// schedule:priority; the broker fans it out to the proxies. Non-fatal like
	// the other auxiliaries — absent it, the proxies keep their default order.
	b.schedulerSup = b.startOptionalWorker("scheduler", b.schedulerPath, b.spawnJobScheduler, func() { b.setScheduler(nil) })
	if b.schedulerSup != nil {
		defer b.schedulerSup.Stop()
	}

	if err := b.codec.Notify("app:ready", ReadyParams{Version: Version}); err != nil {
		return fmt.Errorf("send app:ready: %w", err)
	}

	// Attach the change hook only AFTER app:ready is on the wire so a
	// scanner event arriving during startup can't race ahead of
	// app:ready and produce a discovery:nodes-changed before the
	// client has seen the broker come up. Detaching on the way out
	// keeps a last-millisecond scanner event from trying to push a
	// notification through a codec we're about to tear down. The hook
	// is always attached; whether it actually emits is gated by the
	// per-session subscription flag inside emitNodesChanged.
	b.store.SetOnChange(b.emitNodesChanged)
	defer b.store.SetOnChange(nil)

	err := b.readLoop(ctx)
	cancel()
	b.shutdownInferenceStack()
	return err
}

// emitNodesChanged is the hook the discoveryStore invokes after every
// successful mutation. It is a no-op unless the peer has opted into the
// stream via discovery:subscribe — the discovery:nodes-changed stream
// is opt-in, so an unsubscribed (or never-subscribed) peer sees
// nothing. Once subscribed it defers to pushNodesChanged.
func (b *Broker) emitNodesChanged() {
	// The scheduler is fanned the node universe unconditionally (it's an
	// internal consumer, not a subscribing client), independent of whether the
	// external peer opted into discovery:nodes-changed.
	b.fanDiscoveryToScheduler()

	b.subMu.Lock()
	subscribed := b.subscribed
	b.subMu.Unlock()
	if !subscribed {
		return
	}
	b.pushNodesChanged()
}

// fanDiscoveryToScheduler pushes the current discovery snapshot to the scheduler
// child as a discovery:nodes-changed notification (same bare-array shape the
// client stream uses). A no-op when no scheduler is running.
func (b *Broker) fanDiscoveryToScheduler() {
	b.schedulerFeedMu.Lock()
	defer b.schedulerFeedMu.Unlock()

	sc := b.getScheduler()
	if sc == nil {
		return
	}
	if err := sc.Notify("discovery:nodes-changed", b.store.Snapshot()); err != nil {
		slog.Warn("fan discovery:nodes-changed to scheduler failed", "err", err)
	}
}

// pushNodesChanged takes a fresh snapshot and writes it to the peer as a
// bare-array discovery:nodes-changed notification — the payload shape is
// intentionally just []AvailableNode, not wrapped, per the external
// schema. Notify errors are logged but not fatal: the next mutation
// gets its own chance, and a client that's gone away will be torn down
// by the regular EOF path on the read side. It does NOT consult the
// subscription flag, so callers must gate on it themselves (the hook
// path via emitNodesChanged; the subscribe path because it has just set
// the flag true and wants an immediate baseline regardless of races).
func (b *Broker) pushNodesChanged() {
	if err := b.codec.Notify("discovery:nodes-changed", b.store.Snapshot()); err != nil {
		slog.Warn("emit discovery:nodes-changed failed", "err", err)
	}
}

// forwardProxyNotification is the hook startProxy invokes on the proxy
// reader goroutine for every notification the proxy emits. It re-emits the
// notification to the client as proxy:<method> (the proxy's native method
// name, prefixed) with params passed through unchanged — but only when the
// peer has opted in via proxy:subscribe. The stream is off by default, so
// an unsubscribed peer sees nothing. errors:report / errors:clear are
// routed into the nvpair-errors pipeline (the broker now supervises it)
// rather than dropped, so the proxy's upstream-unreachable reports reach
// the registry.
func (b *Broker) forwardProxyNotification(method string, params json.RawMessage) {
	b.forwardProxyNotificationForGeneration(b.currentOllamaProxyGeneration(), method, params)
}

func (b *Broker) forwardProxyNotificationForGeneration(generation uint64, method string, params json.RawMessage) {
	if generation != b.currentOllamaProxyGeneration() {
		return
	}
	// Strip the facade address before anything dispatches on the method: every
	// comparison below, and the errors relay in particular, matches bare names.
	method, addressed := facadeMethodFor(ollamaProxyProfile, method)
	if !addressed {
		return
	}
	// The alias warning is also the proxy's authoritative "bind did not win"
	// receipt. Release the candidate reservation before forwarding the sticky
	// warning so an existing Ollama owner can be adopted normally.
	if method == methodErrorsReport {
		var report errors.ServiceError
		if json.Unmarshal(params, &report) == nil && report.ID == ollamaHostAliasBlockedID {
			if !b.releaseOllamaHostAliasReservationForGeneration(generation) {
				return
			}
		}
	}
	if b.dispatchErrorsNotif(ollamaProxyProfile.ComponentName(), method, params) {
		return
	}
	// A process can win the port after startup preflight but before the facade
	// binds. Choose a safe fallback here, without calling back into this reader
	// goroutine; the backend move is still pending, so no engine state needs
	// rollback.
	//
	// This notification precedes the failing enable's response on the same
	// stream, so the port chosen here is what ollamaFallbackPort finds when the
	// enable retries. Deciding it here rather than there is deliberate: only
	// this path knows whether the managed facade has to be given up, and it
	// owns telling the user.
	if method == "error" {
		var ep struct {
			Code string `json:"code"`
			Port int    `json:"port"`
		}
		if json.Unmarshal(params, &ep) == nil && ep.Code == "bind-failed" && !b.ollamaState().explicitSettings.Load() {
			switch {
			case b.ollamaState().managedFacade.Load() && ep.Port == managedOllamaFacadePort:
				b.blockManagedOllamaFacade("another process acquired the compatibility port during startup")
			default:
				// Without this the retry asks for the same port and the enable
				// fails identically, leaving the facade down for the life of
				// the process.
				fallback := b.setOllamaProxyFallback(ep.Port)
				slog.Warn("Ollama proxy bind failed; retrying on fallback", "port", ep.Port, "fallback", fallback)
			}
		}
	}
	// When the proxy announces a (re)bound port — notably its restored port
	// on startup — steer it clear of any running engine's port. Dispatched
	// on a separate goroutine: the corrective set-port round-trips through
	// this same proxy-reader goroutine, so calling it inline would deadlock.
	if method == "ready" {
		var rp proxyReadyParams
		if err := json.Unmarshal(params, &rp); err == nil && rp.Port > 0 {
			go b.reconcileProxyPortOnReady(rp.Port)
		}
	}
	b.forwardEngineProxyNotification(ollamaProxyProfile, method, params)
}

// routeNodeActivity relays a proxy's liveness report about a peer down to the
// node-scanner, which uses it to keep a node that is busy serving inference
// rather than evicting it for failing a probe it had no CPU to answer.
//
// It is consumed here rather than forwarded to clients: this is discovery input,
// not a proxy control-plane event, and no client has any use for it. The handoff
// is a non-blocking queue push (reportNodeActivity), so this stays on the proxy
// reader goroutine without letting a wedged scanner stall it.
func (b *Broker) routeNodeActivity(params json.RawMessage) {
	var p noderec.NodeActivityParams
	if err := json.Unmarshal(params, &p); err != nil || p.HostUUID == "" {
		return
	}
	sp := b.getScanner()
	if sp == nil {
		return
	}
	sp.reportNodeActivity(p.HostUUID, time.Now().Add(-clampActivityAge(p.MSSince)))
}

// clampActivityAge turns a reported age in milliseconds into a duration, bounded
// at both ends.
//
// The value crosses a process boundary, and the arithmetic is the fragile part: a
// large enough figure overflows the multiplication into a negative duration,
// which would place the observation in the FUTURE and vouch for the node forever.
// A negative input does the same directly. Neither is reachable from today's
// producers, which derive the value from time.Since — which is exactly why it is
// worth pinning here rather than relying on that staying true.
func clampActivityAge(msSince int64) time.Duration {
	if msSince <= 0 {
		return 0
	}
	// Compared before the multiplication, which is the operation that overflows.
	if msSince > int64(activityAgeCeiling/time.Millisecond) {
		return activityAgeCeiling
	}
	return time.Duration(msSince) * time.Millisecond
}

// activityAgeCeiling is far past any age the scanner still treats as fresh, so
// clamping to it cannot turn a stale report into a live one.
const activityAgeCeiling = time.Hour

func (b *Broker) readLoop(ctx context.Context) error {
	// codec.Read() blocks on stdin, so we run it on its own goroutine and
	// select against ctx.Done(). Otherwise a SIGINT/SIGTERM (which cancels
	// ctx) would leave us parked in Read until a line or EOF arrived on
	// stdin — Serve would never return, its deferred worker teardown would
	// never run, and the broker would appear to hang on Ctrl+C.
	type readResult struct {
		msg *Message
		err error
	}
	reads := make(chan readResult)
	go func() {
		for {
			msg, err := b.codec.Read()
			select {
			case reads <- readResult{msg: msg, err: err}:
			case <-ctx.Done():
				return
			}
			// EOF is terminal (stream closed); other errors are per-line
			// (e.g. a bad JSON frame) and the next Read advances past them.
			if err == io.EOF {
				return
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return nil
		case r := <-reads:
			if r.err != nil {
				if r.err == io.EOF || ctx.Err() != nil {
					return nil
				}
				slog.Warn("JSON-RPC read error", "err", r.err)
				continue
			}
			if !b.handlePerformanceRequest(r.msg) {
				b.handleMessage(r.msg)
			}
			if ctx.Err() != nil {
				return nil
			}
		}
	}
}

// forwardClusterManagerNotification is the hook startClusterManager invokes on
// the manager's reader goroutine for every notification it emits. The
// cluster-manager's events (cluster:invite-received, cluster:identity-changed,
// nodes:changed, ...) are low-volume, important push events with no opt-in
// stream, so they are relayed to the client verbatim. errors:report /
// errors:clear are routed into the nvpair-errors pipeline (the broker now
// supervises it) rather than dropped.
func (b *Broker) forwardClusterManagerNotification(method string, params json.RawMessage) {
	// Private authority envelopes never reach ordinary clients, even if an
	// alternate internal caller bypasses the child-bound notification hook.
	if method == peerServicesEvent {
		return
	}
	if b.dispatchErrorsNotif("cluster-manager", method, params) {
		return
	}
	if b.performance != nil && (method == "cluster:identity-changed" || method == "cluster:trust-changed" || method == "nodes:changed") {
		performanceSignal(b.performance.changeWake)
	}
	if err := b.codec.Notify(method, params); err != nil {
		slog.Warn("forward cluster-manager notification failed", "method", method, "err", err)
	}
	// When this node's cluster identity is (re)established — cluster:create,
	// adopt-on-join, or cleared by cluster:leave — persist it to nvpair-node-settings
	// (the durable home restoreClusterIdentity reads on next startup) and converge
	// this node's advertised cluster identity. The cluster-scoped workers pick the
	// membership change up themselves; see applyClusterIdentityChange.
	if method == "cluster:identity-changed" {
		var id struct {
			ClusterID           string `json:"clusterId"`
			ClusterFriendlyName string `json:"clusterFriendlyName"`
		}
		if err := json.Unmarshal(params, &id); err != nil {
			slog.Warn("cluster:identity-changed had undecodable params; not persisting", "err", err)
		} else {
			b.persistClusterIdentity(id.ClusterID, id.ClusterFriendlyName)
		}
		b.applyClusterIdentityChange()
	}
	// A pairing or removal changed the pin set. The cluster-scoped workers read
	// their gates live and need nothing, but the node-scanner caches a per-peer
	// trusted annotation it can only otherwise refresh when that peer's mDNS
	// record moves — which a peer already advertising when we joined never does.
	if method == "cluster:trust-changed" {
		b.applyClusterTrustChange()
	}
}

// applyClusterTrustChange re-derives the scanner's per-peer trust annotation
// after nvpair-cluster-manager reports a pin-set change. Fire-and-forget on its
// own goroutine, for the same reason as applyClusterIdentityChange: it is a
// bounded-but-blocking RPC and the caller is the cluster-manager's reader.
//
// node-info is pushed here too, not only from applyClusterIdentityChange: a pin
// on its own can decide membership (a cluster dir predating admission.json is
// clustered on pins alone — see clustertrust.Mesh), so this event can change the
// principal we publish without any identity change being reported. Peers now
// read that value over HTTP in preference to our mDNS record, so it must not have
// fewer refresh triggers than the record does — otherwise a peer converges on a
// principal we have already stopped advertising.
func (b *Broker) applyClusterTrustChange() {
	slog.Info("cluster pin set changed; re-deriving peer trust")
	if sc := b.getScanner(); sc != nil {
		go sc.reloadTrust()
	}
	go b.pushClusterIdentityToNodeInfo()
}
