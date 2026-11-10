// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/applog"
	"nvpair-shared/buildinfo"
	"nvpair-shared/clustertrust"
	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
	"nvpair-shared/errors"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/modeldownload"
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

const restoreInstalledEnginesMethod = "engine:restore-installed"
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
	// keyed by engine-manager engine name (e.g. "llamacpp"). Additive:
	// Models stays the flat de-duplicated union for consumers that only need the
	// node-level set; this is the per-engine breakdown a consumer needs to show a
	// remote node's models under the correct engine (the UI's engine card).
	// Omitted only when attribution is nil; a nonnil empty map authoritatively
	// reports no per-engine models and must remain present on the client wire.
	ModelsByEngine map[string][]string `json:"modelsByEngine,omitzero"`
	// ModelIdentities carries validated display metadata for listed models.
	ModelIdentities map[string]map[string]engines.ModelIdentity `json:"modelIdentities,omitempty"`
	// LoadedByEngine names the models currently resident in memory per engine
	// (normally a subset of ModelsByEngine), enriched by the daemon from the
	// peer's engine-manager /v1/models loadedByEngine field. Lets a per-engine
	// consumer show which of a remote node's models are loaded. Omitted
	// when no engine reports loaded state.
	LoadedByEngine map[string][]string `json:"loadedByEngine,omitempty"`
	// Download state is target-owned; freshness is the scanner's observation.
	DownloadState        *modeldownload.Snapshot `json:"downloadState,omitempty"`
	DownloadStateCurrent *bool                   `json:"downloadStateCurrent,omitempty"`
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
	// portAvailableCheck is fixed before use; nil keeps the production bind check.
	portAvailableCheck func(int) bool
	// routingForward bounds routing control forwarding to one operation.
	routingForward atomic.Bool
	// Constructed by main before the local codec. History data and its
	// writer are independent of discovery/workload subscriptions.
	routingHistory *brokerRoutingHistory
	// All source identity/fence fields below are guarded by workersMu. No
	// history mutex, copy or output is allowed while workersMu is held.
	historyIncarnation    uint64
	historyRetiredThrough uint64
	historyOrigin         string
	historyRun            string
	// settingsApplyMu serializes engine settings applies end to end, including
	// the engine stop and restart. engineConfigMu guards only the journal and
	// is released while an engine restarts, so this is what keeps two applies —
	// which may compete for the same port — from interleaving. Always take it
	// before engineConfigMu, never the reverse.
	settingsApplyMu sync.Mutex
	engineConfigMu  sync.Mutex
	modelReviews    []modelSettingsRetainedReview // guarded by engineConfigMu
	// Build maintenance uses the settings journal lock for admission only;
	// its reservation survives unlocked drain/stop and client disconnect.
	buildApplyMaintenance buildApplyMaintenance
	// Nil until complete, side-effect-free owner evidence can be supplied.
	buildApplyOwners buildApplyOwners
	// Only live install continuations are retained, under engineConfigMu.
	engineSettingsLoaded   bool
	engineSettingsError    error
	engineSettingsEpoch    string
	engineSettings         map[string]*engineSettingsRecord
	engineSettingsReplacer durablefile.Replacer
	// Assigned under engineConfigMu; awaited and closed without that mutex.
	settingsPublicationTail <-chan struct{}
	settingsRelayMu         sync.Mutex
	settingsCancels         map[string]context.CancelFunc
	activeSettings          map[string]activeSettingsOperation
	diagnostics             modelDiagnosticRuntime
	codec                   *Codec
	// Published before workers start; the service-facing session owns the
	// relay and its physical transport interruption, including blocked writes.
	engineNotifications  *engineNotificationRelay
	disconnectClient     func()
	cancel               context.CancelFunc
	startedAt            time.Time
	nodeID               string
	scannerPath          string
	disableMDNS          bool
	scannerNamespace     *noderec.MDNSNamespace
	manualProbePorts     *manualProbePorts
	nodeInfoPath         string
	proxyPath            string
	proxyEngines         []string
	workloadMgrPath      string
	errorsPath           string
	engineMgrPath        string
	engineMgrHTTPPort    int
	engineMgrControlPort int
	engineMgrControlHost string
	nodeInfoPort         int
	errorsPort           int
	workloadMgrPort      int
	clusterMgrPort       int
	manualNodesPath      string
	settingsPath         string
	clusterMgrPath       string
	schedulerPath        string
	clusterDir           string
	// Fixture wait overrides are copied at construction and never changed in use.
	// Zero retains the production interval, including for bare Broker fixtures.
	residentRetryDelay time.Duration
	advertiseCadence   time.Duration
	scannerWaits       scannerWaitOverrides
	// facadesReady closes after the initial facade startup attempt, before
	// enabled engines are restored. Nil means no startup wait is configured.
	facadesReady   chan struct{}
	stickyErrorsMu sync.Mutex
	stickyErrors   map[string]errors.ServiceError // see stickyErrorIDs
	store          *discoveryStore
	telemetry      *telemetryCache
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
	// selfNameMu guards this node's UUID and name as the cluster manager last
	// reported them (cluster:get-node-id, cluster:node-name-changed). The
	// broker relays the name to the scanner, which advertises it as name=.
	selfNameMu   sync.Mutex
	selfNodeUUID string
	selfNodeName string
	// selfNamePushMu serializes name pushes to the scanner; each push reads
	// the cached name under it, so the last push always carries the latest.
	selfNamePushMu sync.Mutex

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
	scannerSource       *scannerProcess
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
	peerSizing  *peerSizingRuntime
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
	// Serialize source admission and synchronous projection with retirement.
	// This lock never nests inside workersMu.
	manualMu           sync.Mutex
	manualNodesSource  *rpcWorker
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
	disableMDNS      bool
	scannerNamespace *noderec.MDNSNamespace
	manualProbePorts *manualProbePorts
	scanner          string
	nodeInfo         string
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
	// Nil retains the production default; an explicitly empty host is invalid.
	engineMgrControlHost *string
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
	// Fixture-only startup overrides; zero retains the production intervals.
	residentRetryDelay time.Duration
	advertiseCadence   time.Duration
	scannerWaits       scannerWaitOverrides
}

// NewBroker constructs a per-session broker. paths.scanner is required —
// the scanner is the broker's core worker. Every other path is optional:
// an empty string means the broker won't spawn that worker (e.g. the
// binary couldn't be resolved). Without node-info the broker runs without
// local node advertisement; without the proxy, no local engine reverse
// proxy; without the workload-manager, no cluster workload relay; without
// nvpair-errors, the service-error pipeline is disabled (producers' errors
// are dropped, as before); without the cluster-manager, no cluster pairing
// and membership (cluster:* / nodes:*). Discovery still works in every case.
func NewBroker(codec *Codec, paths workerPaths) *Broker {
	controlHost := defaultEngineManagerControlHost
	if paths.engineMgrControlHost != nil {
		controlHost = *paths.engineMgrControlHost
	}
	validNamespace := true
	if paths.scannerNamespace != nil {
		_, err := paths.scannerNamespace.Normalize()
		validNamespace = err == nil
	}
	// Resolve the local node id once, as the stable per-host UUID (the same
	// value the node-scanner advertises as hostUuid and the cluster-manager
	// uses as nodeUuid). It's stamped onto every local-origin workload
	// (originatedFrom) and every local error (nodeId) so the whole cluster keys
	// this node by an identity that survives a PC rename — hostname is display
	// only. The same value is passed to nvpair-errors via --node-id so
	// its localNodeID stays in lockstep with what the broker stamps.
	nodeID := ""
	if validNamespace && validateEngineManagerControlHost(controlHost) == nil && paths.residentRetryDelay >= 0 && paths.advertiseCadence >= 0 && paths.scannerWaits.validate() == nil {
		nodeID = resolveLocalNodeID(paths.clusterDir)
	}
	return &Broker{
		codec:                codec,
		startedAt:            time.Now(),
		nodeID:               nodeID,
		scannerPath:          paths.scanner,
		disableMDNS:          paths.disableMDNS,
		scannerNamespace:     paths.scannerNamespace,
		manualProbePorts:     paths.manualProbePorts,
		nodeInfoPath:         paths.nodeInfo,
		proxyPath:            paths.proxy,
		proxyEngines:         paths.proxyEngines,
		workloadMgrPath:      paths.workloadMgr,
		errorsPath:           paths.errors,
		engineMgrPath:        paths.engineMgr,
		engineMgrHTTPPort:    paths.engineMgrHTTPPort,
		engineMgrControlPort: paths.engineMgrControlPort,
		engineMgrControlHost: controlHost,
		nodeInfoPort:         paths.nodeInfoPort,
		errorsPort:           paths.errorsPort,
		workloadMgrPort:      paths.workloadMgrPort,
		clusterMgrPort:       paths.clusterMgrPort,
		manualNodesPath:      paths.manualNodes,
		settingsPath:         paths.settings,
		clusterMgrPath:       paths.clusterMgr,
		schedulerPath:        paths.scheduler,
		clusterDir:           paths.clusterDir,
		residentRetryDelay:   paths.residentRetryDelay,
		advertiseCadence:     paths.advertiseCadence,
		scannerWaits:         paths.scannerWaits,
		store:                newDiscoveryStore(),
		telemetry:            newTelemetryCache(),
		relayDir:             relay.NewDirectory(),
		regCache:             relay.NewRegistrationCache(),
		manualNodeKeys:       make(map[string]string),
		manualNodeStatuses:   make(map[string]manualNodeStatusEntry),
		workloads:            workloadstore.New(),
		facadesReady:         make(chan struct{}),
	}
}

const defaultEngineManagerControlHost = "0.0.0.0"

func validateEngineManagerControlHost(host string) error {
	if net.ParseIP(host) == nil {
		return fmt.Errorf("--engine-manager-control-host must be a literal IP address (got %q)", host)
	}
	return nil
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
	if s == nil {
		b.scannerSource = nil
		if b.telemetry != nil {
			b.telemetry.retireScannerMemory()
		}
	}
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
	b.workersMu.Lock()
	if b.engineMgr != w {
		// A different worker was installed, so any source the previous pair
		// named is no longer that pair's. Only spawnEngineMgr binds again.
		b.engineMgrSource = nil
		b.wakeEngineMgrBindingLocked()
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
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	if b.engineMgr == w {
		b.engineMgrRecovering = true
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

// The replacement stays hidden until its pause projection and shutdown proof
// have been restored. Settings recovery still precedes enabled restoration.
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
		// A retained diagnostic owns settingsApplyMu. Reconstruct its drains
		// first; ordinary pause recovery cannot wait on that same lease. The
		// manager reservation fences activation until cleanup and release.
		if b.reconcileDiagnostics(context.Background(), w) == nil && (b.diagnosticHoldActive() || b.recoverEngineManagerPause(w)) {
			b.publishNodeProjection()
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
			return current
		}
		if attempt == attempts-1 {
			slog.Error("engine manager remains unavailable because pause recovery could not be verified")
			return false
		}
		// Retry the pinned replacement while its ordinary surface stays fenced.
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

func (b *Broker) recoverEngineManagerPause(w *rpcWorker) bool {
	r := b.nodePolicyRuntime()
	r.transitionMu.Lock()
	defer r.transitionMu.Unlock()
	b.settingsApplyMu.Lock()
	defer b.settingsApplyMu.Unlock()
	worker, source := b.pauseManager()
	if worker != w || w == nil {
		return false
	}
	b.publishNodeProjection()
	r.mu.Lock()
	paused := r.current.Availability == nodepolicy.Paused
	if paused {
		r.live = nodepolicy.Draining
	}
	r.mu.Unlock()
	if !paused {
		return true
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	err := b.pauseEnginesOn(ctx, w)
	if err == nil {
		err = b.publishPhysicalPause(ctx, w, source)
	}
	b.announceActivationChange(b.pushPolicyNow(context.Background(), pushAvailability))
	b.notifyAvailability()
	if err != nil {
		slog.Error("engine manager pause recovery failed", "err", err)
	}
	return err == nil
}

func (b *Broker) restoreInstalledEngines(w *rpcWorker) {
	if w == nil {
		return
	}
	// Reconstruct settings withdrawal before restoring installed engines.
	b.publishNodeProjection()
	if !b.recoverEngineSettings() {
		return
	}
	if b.skipRestoreWhilePaused() {
		slog.Info("node is paused; engines are restored when it resumes")
		return
	}
	if err := w.Notify(restoreInstalledEnginesMethod, nil); err != nil {
		slog.Warn("failed to request installed-engine restoration", "err", err)
	}
}

func (b *Broker) facadeStartupReady() bool {
	if b.facadesReady == nil {
		return true
	}
	select {
	case <-b.facadesReady:
		return true
	default:
		return false
	}
}

func (b *Broker) runEngineAvailability(
	ctx context.Context,
	runners ...func(context.Context),
) bool {
	if b.facadesReady != nil {
		select {
		case <-ctx.Done():
			return false
		case <-b.facadesReady:
		}
	}
	if ctx.Err() != nil {
		return false
	}
	b.restoreInstalledEngines(b.getEngineMgr())
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
// handler. The source process is fenced by forwardProxyNotificationFrom.
// A process-scoped notification is unaddressed
// and must be handled exactly once, not once per engine — routing a workload
// record through both handlers would record every job twice.
func (b *Broker) forwardProxyProcessNotification(method string, params json.RawMessage) {
	engine, bare := engines.SplitAddressedMethod(method)
	switch engine {
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
	result, rpcErr, err := cm.Call(ctx, "cluster:get-node-id", nil)
	if err != nil {
		slog.Warn("scanner identity convergence skipped: cluster-manager node id unavailable", "err", err)
		return
	}
	if rpcErr == nil {
		var id nodeNameIdentity
		if err := json.Unmarshal(result, &id); err != nil {
			slog.Warn("cluster:get-node-id result undecodable; node name not relayed", "err", err)
		} else {
			b.recordSelfName(id)
			b.pushSelfName()
		}
	}
	if sc := b.getScanner(); sc != nil {
		sc.reloadIdentity()
	}
}

// nodeNameIdentity is the part of cluster:get-node-id and
// cluster:node-name-changed the broker relays: this node's UUID and its
// effective name.
type nodeNameIdentity struct {
	NodeUUID string `json:"nodeUuid"`
	Name     string `json:"name"`
}

// recordSelfName caches this node's UUID and name as the cluster manager
// reported them, and hands them to the discovery store, which shows this node
// with that name.
func (b *Broker) recordSelfName(id nodeNameIdentity) {
	b.selfNameMu.Lock()
	b.selfNodeUUID = id.NodeUUID
	b.selfNodeName = id.Name
	b.selfNameMu.Unlock()
	if b.store != nil {
		b.store.SetSelf(id.NodeUUID, id.Name)
	}
}

// clusterNodesPayload is the part of the nodes:get-initial result and the
// nodes:changed params the discovery store needs: each member's UUID and the
// display name the cluster manager decided for it.
type clusterNodesPayload struct {
	Nodes []struct {
		NodeUUID    string `json:"nodeUuid"`
		DisplayName string `json:"displayName"`
		State       string `json:"state"`
	} `json:"nodes"`
}

// applyMemberNames feeds the members' display names from a nodes:get-initial
// result or nodes:changed params into the discovery store. Pending invitees
// are not members and keep their discovery name.
func (b *Broker) applyMemberNames(raw json.RawMessage) {
	if b.store == nil {
		return
	}
	var payload clusterNodesPayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		slog.Warn("cluster nodes payload undecodable; member names not updated", "err", err)
		return
	}
	names := make(map[string]string, len(payload.Nodes))
	for _, node := range payload.Nodes {
		if node.State == "member" && node.NodeUUID != "" && node.DisplayName != "" {
			names[node.NodeUUID] = node.DisplayName
		}
	}
	b.store.SetMemberNames(names)
}

// refreshMemberNames reads nodes:get-initial from a freshly spawned cluster
// manager, so member names are current before its first nodes:changed.
func (b *Broker) refreshMemberNames() {
	cm := b.getClusterMgr()
	if cm == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, rpcErr, err := cm.Call(ctx, "nodes:get-initial", nil)
	if err != nil || rpcErr != nil {
		slog.Warn("member names unavailable from cluster-manager", "err", err, "rpcError", rpcErr != nil)
		return
	}
	b.applyMemberNames(result)
}

// pushSelfName sends the cached name to the current scanner with
// discovery:set-self-name. A name not yet reported is not sent. The scanner
// acknowledges at once and republishes in the background, so this blocks at
// most for one round trip; callers on a reader goroutine still run it on its
// own goroutine.
func (b *Broker) pushSelfName() {
	b.selfNamePushMu.Lock()
	defer b.selfNamePushMu.Unlock()
	b.selfNameMu.Lock()
	name := b.selfNodeName
	b.selfNameMu.Unlock()
	if name == "" {
		return
	}
	if sc := b.getScanner(); sc != nil {
		sc.setSelfName(name)
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
// and returns the string, or "" on any error.
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
		"engine:diagnostic-request", "engine:diagnostic-request-cancel":
		return true
	}
	return false
}

// forwardEngineNotificationFrom handles the bound engine-manager's notifications.
// Identity and running-state observations are fenced by its source; control-plane
// requests and subscribed client notifications follow their respective relays.
func (b *Broker) forwardEngineNotificationFrom(source *engineManagerSource, method string, params json.RawMessage) {
	if routingHistoryMethod(method) {
		return
	}
	if method == "engine:diagnostic-changed" {
		if !b.engineManagerSourceCurrent(source) {
			return
		}
		b.diagnosticNotification(source, params)
		return
	}
	if method == residentDispatchRequestMethod {
		b.dispatchResidentFrom(source, params)
		return
	}
	if method == admittedLoadRequestMethod {
		b.admittedLoadFrom(source, params)
		return
	}
	if method == admittedLoadReplyMethod {
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
	if b.handlePolicyEngineNotification(source, method, params) {
		return
	}
	if method == "engine:diagnostic-request" {
		b.handleDiagnosticRelay(source, params)
		return
	}
	if method == "engine:diagnostic-request-cancel" {
		b.cancelDiagnosticRelay(source, params)
		return
	}
	if method == "engine:diagnostic-reply" {
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
		// A (re)started engine-manager holds no node projection until it is
		// sent one. Off the read pump, like the settings projection below.
		go func() {
			b.workersMu.Lock()
			worker := b.engineMgr
			b.workersMu.Unlock()
			if b.engineManagerSourceCurrent(source) {
				b.publishNodeProjection()
				if err := b.reconcileDiagnostics(context.Background(), worker); err != nil {
					slog.Warn("diagnostic checkpoint reconciliation requires retry")
					return
				}
				b.replayLoadedEngineSettings()
			}
		}()
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
	b.queueEngineNotification(source, method, params)
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
	b.peerSizing.signal()
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
	b.peerSizing.signal()
	b.engineRelayMu.Unlock()
}

func (b *Broker) forwardManualNodesNotification(method string, params json.RawMessage) {
	b.manualMu.Lock()
	defer b.manualMu.Unlock()
	b.forwardManualNodesNotificationLocked(method, params)
}

func (b *Broker) forwardManualNodesNotificationLocked(method string, params json.RawMessage) {
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
		b.upsertManualNodeLocked(s)
		slog.Debug("manual node event", "event", method, "id", s.ID)
	case "node/removed":
		var s manualNodeStatus
		if err := json.Unmarshal(params, &s); err != nil {
			slog.Warn("manual-nodes emitted invalid removed payload", "err", err)
			return
		}
		b.removeManualNodeLocked(s.ID)
		slog.Debug("manual node event", "event", method, "id", s.ID)
	case "ready":
		slog.Info("manual-nodes reported ready", "params", string(params))
	default:
		slog.Debug("ignoring manual-nodes notification", "method", method)
	}
}

// upsertManualNode ingests a manual node status: it records the alias's payload,
// keys the discovery store by the node's operational identity (its hostUuid once
// node-info reports one, else the manual id), and updates scheduler telemetry.
// If the alias's key changed (node-info revealed its real UUID) the
// old key is reprojected from a surviving alias or released.
func (b *Broker) upsertManualNode(s manualNodeStatus) {
	b.manualMu.Lock()
	defer b.manualMu.Unlock()
	b.upsertManualNodeLocked(s)
}

func (b *Broker) upsertManualNodeLocked(s manualNodeStatus) {
	s.MemoryObservation = s.MemoryObservation.Clone()
	if s.MemoryHop != nil {
		hop := *s.MemoryHop
		s.MemoryHop = &hop
	}
	en := manualToEnriched(s)
	key := en.storeKey()
	receivedAt := time.Now()

	oldKey, existed := b.manualNodeKeys[s.ID]
	b.manualNodeKeys[s.ID] = key
	b.manualNodeStatuses[s.ID] = manualNodeStatusEntry{status: s, receivedAt: receivedAt}

	b.store.Upsert(en, sourceManual)
	b.ingestTelemetryAt(sourceManual, manualNodeTelemetry(s, key), receivedAt)

	if existed && oldKey != key {
		// This alias moved off oldKey: refresh oldKey from a surviving alias, or
		// release its discovery claim if this was its last owner.
		b.reprojectOrReleaseLocked(oldKey)
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
	defer b.manualMu.Unlock()
	b.removeManualNodeLocked(id)
}

func (b *Broker) removeManualNodeLocked(id string) {
	key, ok := b.manualNodeKeys[id]
	delete(b.manualNodeKeys, id)
	delete(b.manualNodeStatuses, id)
	if !ok {
		return
	}
	b.reprojectOrReleaseLocked(key)
}

// reprojectOrRelease refreshes or drops a key's manual claim after an alias
// stopped pointing at it. If another manual alias still resolves to key, that
// survivor's payload is re-projected into the store's single manual slot;
// otherwise the discovery and telemetry claims are released.
func (b *Broker) reprojectOrRelease(key string) {
	b.manualMu.Lock()
	defer b.manualMu.Unlock()
	b.reprojectOrReleaseLocked(key)
}

func (b *Broker) reprojectOrReleaseLocked(key string) {
	if key == "" {
		return
	}
	survivor, ok := b.survivingAliasLocked(key)
	if ok {
		b.store.Upsert(manualToEnriched(survivor.status), sourceManual)
		b.ingestTelemetryAt(sourceManual, manualNodeTelemetry(survivor.status, key), survivor.receivedAt)
		return
	}
	b.store.Remove(key, sourceManual)
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
	b.manualMu.Lock()
	defer b.manualMu.Unlock()
	b.clearManualNodesStateLocked()
}

func (b *Broker) clearManualNodesStateLocked() {
	b.manualNodesSource = nil
	b.setManualNodes(nil)
	keys := b.manualNodeKeys
	b.manualNodeKeys = make(map[string]string)
	b.manualNodeStatuses = make(map[string]manualNodeStatusEntry)
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
	b.sendRoutingCosts(ctx, true)
	b.sendPeerSizing(ctx, true)
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
	return b.serve(ctx, nil)
}

// Private main construction owns both standard cancellation branches. The
// public Serve path continues deriving from its caller's original context.
func (b *Broker) serveMainLifetime(lifetime *brokerMainLifetime) error {
	if lifetime == nil {
		return jsonrpc.ErrHistoryInvalid
	}
	return b.serve(nil, lifetime)
}

func (b *Broker) newServeContext(parent context.Context, lifetime *brokerMainLifetime) (context.Context, context.CancelFunc, error) {
	if lifetime == nil {
		ctx, cancel := context.WithCancel(parent)
		return ctx, cancel, nil
	}
	if b.routingHistory != nil && b.routingHistory.lifetime != lifetime {
		return nil, nil, jsonrpc.ErrHistoryInvalid
	}
	return lifetime.startServe()
}

func (b *Broker) serve(ctx context.Context, lifetime *brokerMainLifetime) error {
	if err := b.scannerWaits.validate(); err != nil {
		return err
	}
	if err := validateTestWaitIntervals(b.modelConfigRetryInterval(), b.advertisementInterval()); err != nil {
		return err
	}
	if err := validateServicePorts(b.servicePorts()); err != nil {
		return fmt.Errorf("service ports: %w", err)
	}
	if err := b.manualProbePorts.validate(); err != nil {
		return err
	}
	if _, err := b.scannerArgs(); err != nil {
		return err
	}
	if err := validateEngineManagerControlHost(b.engineMgrControlHost); err != nil {
		return err
	}
	ctx, cancel, err := b.newServeContext(ctx, lifetime)
	if err != nil {
		return err
	}
	b.cancel = cancel
	b.setServeCtx(ctx)
	defer cancel()
	stopEngineNotifications := b.startEngineNotificationRelay(ctx)
	defer stopEngineNotifications()
	if h := b.routingHistory; h != nil {
		// Required local replies use background write contexts and can hold the
		// dispatcher inside RespondHistoryError. Interrupt output independently
		// of Serve returning. Main's startServe cancel wrapper cancels history
		// before Serve, so both root cancellation and b.cancel reach this hook
		// without registering J work on the ordinary Serve/RPC child graph.
		interrupt := h.interruptLocalWrite
		interrupted := make(chan struct{})
		stopInterrupt := context.AfterFunc(h.parent, func() {
			if interrupt != nil {
				interrupt()
			}
			close(interrupted) // final application access of this callback
		})
		done := make(chan struct{})
		go func() { defer close(done); h.run() }()
		defer func() {
			h.lifetime.historyCancel()
			if !stopInterrupt() {
				<-interrupted
			} else if interrupt != nil {
				interrupt()
			}
			<-h.readers.retire(historyLocalControl, b.codec)
			<-done
			h.initialWriters.Wait()
		}()
	}
	stopPerformance := b.startPerformance(ctx)
	defer stopPerformance()
	stopPeerSizing := b.startPeerSizing(ctx)
	defer stopPeerSizing()

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

	// Settings and engine-manager must be available before facade preparation.
	b.settingsSup = b.startOptionalWorker("settings", b.settingsPath, b.spawnSettings, func() { b.setSettings(nil) })
	if b.settingsSup != nil {
		defer b.settingsSup.Stop()
	}
	b.engineMgrSup = b.startOptionalWorker("engine-manager", b.engineMgrPath, b.spawnEngineMgr, func() { b.setEngineMgr(nil); b.clearEngineRelaySub() })
	if b.engineMgrSup != nil {
		defer b.engineMgrSup.Stop()
	}
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
		// Preparation was skipped for a disabled engine. Surface its
		// unavailability directly when it is worth surfacing.
		if report {
			b.reportProxyUnavailable(profile, reason)
		}
	}
	b.startProxySupervision(anyProxyEnabled || len(b.dynamicEngineProfiles()) > 0)
	if b.facadesReady != nil {
		close(b.facadesReady)
	}
	defer func() {
		stopEgressRenewal()
		b.shutdownProxySupervision()
	}()

	// Restore engines and begin advertising after the initial facade attempts.
	// This prevents a restored engine from taking a persisted proxy port before
	// the broker can resolve ownership.
	var availabilityRunners []func(context.Context)
	for _, profile := range engineProxyProfiles {
		if !b.proxyEnabled(profile) {
			continue
		}
		profile := profile
		availabilityRunners = append(availabilityRunners, func(ctx context.Context) {
			b.runAutoAdvertiseEngine(ctx, profile)
		})
	}
	availabilityRunners = append(availabilityRunners, b.dynamicEngineAvailabilityRunners()...)
	go b.runEngineAvailability(ctx, availabilityRunners...)

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

	err = b.readLoop(ctx)
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
	if err := sc.Notify("discovery:nodes-changed", discoveryNotificationNodes(b.store.Snapshot())); err != nil {
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
	if err := b.codec.Notify("discovery:nodes-changed", discoveryNotificationNodes(b.store.Snapshot())); err != nil {
		slog.Warn("emit discovery:nodes-changed failed", "err", err)
	}
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
		msg      *Message
		err      error
		release  func()
		consumed chan struct{}
	}
	codec := b.codec
	retirementParent := ctx
	var beforeRetirementDone func()
	if h := b.routingHistory; h != nil {
		var err error
		retirementParent, err = h.lifetime.localRetirementParent(ctx)
		if err != nil {
			return err
		}
		beforeRetirementDone = h.beforeLocalRetirementDone
	}
	retire := func() {
		if b.routingHistory != nil {
			b.routingHistory.readers.retire(historyLocalControl, codec)
		} else {
			codec.RetireRead()
		}
	}
	if b.routingHistory == nil {
		stopRetirement := context.AfterFunc(ctx, retire)
		defer func() { stopRetirement(); retire() }()
	} else {
		retirementDone := make(chan struct{})
		stopRetirement := context.AfterFunc(retirementParent, func() {
			retire()
			if beforeRetirementDone != nil {
				beforeRetirementDone()
			}
			close(retirementDone) // final application access of the actual callback
		})
		defer func() {
			if !stopRetirement() {
				<-retirementDone
			}
			retire()
		}()
	}
	reads := make(chan readResult)
	go func() {
		for {
			msg, release, err := codec.ReadWithRelease()
			var consumed chan struct{}
			if release != nil {
				consumed = make(chan struct{})
			}
			select {
			case reads <- readResult{msg: msg, err: err, release: release, consumed: consumed}:
			case <-ctx.Done():
				if release != nil {
					release()
				}
				return
			}
			// History Params/ID borrow reader backing. The dispatcher finishes
			// synchronous use (or copies a bounded ID) before the next Read.
			// Once handed off, only the dispatcher can release. Cancellation
			// retires the reader but cannot stand in for actual consumption.
			if consumed != nil {
				<-consumed
			}
			if ctx.Err() != nil {
				return
			}
			var decode *jsonrpc.DecodeError
			var history *jsonrpc.HistoryFailure
			if err != nil && !stderrors.As(err, &decode) && !stderrors.As(err, &history) {
				return
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return nil
		case r := <-reads:
			err := func() error {
				if r.consumed != nil {
					defer close(r.consumed)
				}
				if r.release != nil {
					defer r.release()
				}
				if ctx.Err() != nil {
					return nil
				}
				if r.err == nil {
					if !b.handlePerformanceRequest(r.msg) {
						b.handleMessage(r.msg)
					}
					return nil
				}
				var history *jsonrpc.HistoryFailure
				if stderrors.As(r.err, &history) {
					// Refusal IDs borrow even when msg is nil. The synchronous
					// response finishes before release/next Read; no raw-ID tail
					// or cancellation-based early reuse is admitted.
					if refusal := history.Refusal; refusal != nil {
						respondLocalHistoryError(codec, &refusal.ReplyID, refusal.Code)
					}
					slog.Debug("routing history control rejected", "category", "history-invalid", "count", history.Count)
					return nil
				}
				var decode *jsonrpc.DecodeError
				if stderrors.As(r.err, &decode) {
					slog.Warn("JSON-RPC read error", "category", "invalid-shape")
					return nil
				}
				if r.err == io.EOF {
					return io.EOF
				}
				return jsonrpc.ErrHistoryFraming
			}()
			if err == io.EOF {
				return nil
			}
			if err != nil {
				return err
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
	if routingHistoryMethod(method) || method == notifyEngineNotificationsReset {
		return
	}
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
	if method == "cluster:identity-changed" || method == "cluster:trust-changed" || method == "nodes:changed" {
		b.peerSizing.signal()
	}
	if method == "nodes:changed" {
		b.applyMemberNames(params)
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
	// This node's name changed: relay it to the scanner, which advertises it.
	// The client already has the notification from the verbatim forward above.
	if method == "cluster:node-name-changed" {
		var id nodeNameIdentity
		if err := json.Unmarshal(params, &id); err != nil {
			slog.Warn("cluster:node-name-changed had undecodable params; not relaying", "err", err)
		} else {
			b.recordSelfName(id)
			go b.pushSelfName()
		}
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
