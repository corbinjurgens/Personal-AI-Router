# services JSON-RPC surface (GENERATED)

> **Do not edit by hand.** Regenerate with `npm run service-contracts:write`
> (or `tsx scripts/verify-service-contracts.ts --write`). Extracts the
> current sibling `services/` tree and cross-checks it against the
> Electron bridge. Integration notes live in `docs/services-backend.md`;
> capability status lives in `docs/services-parity.md`.

- **Source tree**: `services/` in this monorepo
- **Versions**: see `services/versions.json` (services suite and per-component)
- **Legend**: ✅ referenced by the bridge · ❌ MISSING (no consumer/caller) · ➖ ignored (see `docs/service-contract-exceptions.json`)

## Drift summary

### Notifications emitted but NOT consumed by the bridge
- none ✅

### Requests the backend handles but the bridge never calls (unused capability)
- ⚠️ nvpair-engine-manager → engine:configure-launch
- ⚠️ nvpair-engine-manager → engine:configured-ports
- ⚠️ nvpair-engine-manager → engine:describe
- ⚠️ nvpair-engine-manager → engine:errors
- ⚠️ nvpair-engine-manager → engine:get-launch
- ⚠️ nvpair-engine-manager → engine:logs
- ⚠️ nvpair-engine-manager → engine:preview-launch
- ⚠️ nvpair-engine-manager → engine:remote-apply-settings
- ⚠️ nvpair-engine-manager → engine:remote-get-settings
- ⚠️ nvpair-engine-manager → engine:remote-preview-settings
- ⚠️ nvpair-engine-manager → engine:remote-switch
- ⚠️ nvpair-engine-manager → engine:remote-switch-status
- ⚠️ nvpair-engine-manager → engine:restart
- ⚠️ nvpair-engine-manager → engine:set-port
- ⚠️ nvpair-engine-manager → engine:sleep
- ⚠️ nvpair-engine-manager → engine:switch-prepare
- ⚠️ nvpair-engine-manager → engine:switch-reconcile
- ⚠️ nvpair-engine-manager → engine:switch-recovery-ready
- ⚠️ nvpair-engine-manager → engine:switch-step
- ⚠️ nvpair-engine-manager → engine:wake
- ⚠️ nvpair-engine-manager → internal:set-reserved-port
- ⚠️ nvpair-job-scheduler → scheduler:get-interval
- ⚠️ nvpair-job-scheduler → scheduler:get-status
- ⚠️ nvpair-job-scheduler → scheduler:set-interval
- ⚠️ nvpair-job-scheduler → scheduler:tick
- ⚠️ nvpair-node-info → n/a
- ⚠️ nvpair-node-scanner → discovery:register
- ⚠️ nvpair-node-scanner → discovery:reload-identity
- ⚠️ nvpair-node-scanner → discovery:reload-trust
- ⚠️ nvpair-node-scanner → discovery:set-observed-addresses
- ⚠️ nvpair-node-scanner → discovery:unregister
- ⚠️ nvpair-node-scanner → discovery:update-txt
- ⚠️ nvpair-node-settings → settings/get-cluster-auto-sync
- ⚠️ nvpair-node-settings → settings/get-force-ports
- ⚠️ nvpair-node-settings → settings/set-cluster-auto-sync
- ⚠️ nvpair-node-settings → settings/set-force-ports
- ⚠️ nvpair-proxy → facade/enable
- ⚠️ nvpair-proxy → node/admission-snapshot
- ⚠️ nvpair-proxy → node/invalidate-engine
- ⚠️ nvpair-proxy → node/selected
- ⚠️ nvpair-proxy → node/set-availability
- ⚠️ nvpair-proxy → node/set-engine-drain
- ⚠️ nvpair-proxy → node/set-engine-intent
- ⚠️ nvpair-proxy → node/set-local-backend
- ⚠️ nvpair-proxy → node/set-model-identity
- ⚠️ nvpair-proxy → node/set-policy
- ⚠️ nvpair-proxy → node/set-residency
- ⚠️ nvpair-ui-broker → admission/unload
- ⚠️ nvpair-ui-broker → admission/wake
- ⚠️ nvpair-ui-broker → discovery:node-discovered
- ⚠️ nvpair-ui-broker → discovery:unsubscribe
- ⚠️ nvpair-ui-broker → engine:restart
- ⚠️ nvpair-ui-broker → engine:set-port
- ⚠️ nvpair-ui-broker → engine:unsubscribe
- ⚠️ nvpair-ui-broker → engine:wake
- ⚠️ nvpair-ui-broker → workloads:unsubscribe
- ⚠️ nvpair-workload-manager → performance:capabilities
- ⚠️ nvpair-workload-manager → performance:publish
- ⚠️ nvpair-workload-manager → performance:summary

### Backend binaries not listed in `modular-binaries.ts`
- none ✅

## nvpair-cluster-manager

| Method | Direction | In bridge? |
|---|---|---|
| `cluster:identity-changed` | notification (we consume) | ✅ yes |
| `cluster:invite-canceled` | notification (we consume) | ✅ yes |
| `cluster:invite-expired` | notification (we consume) | ✅ yes |
| `cluster:invite-failed` | notification (we consume) | ✅ yes |
| `cluster:invite-received` | notification (we consume) | ✅ yes |
| `cluster:peer-services` | notification (we consume) | ➖ ignored |
| `cluster:remote-management-changed` | notification (we consume) | ✅ yes |
| `cluster:trust-changed` | notification (we consume) | ➖ ignored |
| `discovery:subscribe` | notification (we consume) | ✅ yes |
| `nodes:changed` | notification (we consume) | ✅ yes |
| `cluster:cancel-invite` | request (we call) | ✅ yes |
| `cluster:create` | request (we call) | ✅ yes |
| `cluster:get-node-id` | request (we call) | ✅ yes |
| `cluster:get-remote-management` | request (we call) | ✅ yes |
| `cluster:invite-node` | request (we call) | ✅ yes |
| `cluster:invite-status` | request (we call) | ✅ yes |
| `cluster:leave` | request (we call) | ✅ yes |
| `cluster:respond-to-invite` | request (we call) | ✅ yes |
| `cluster:set-identity` | request (we call) | ✅ yes |
| `cluster:set-local-services` | request (we call) | ➖ ignored |
| `cluster:set-remote-management` | request (we call) | ✅ yes |
| `discovery:nodes` | request (we call) | ✅ yes |
| `nodes:get-initial` | request (we call) | ✅ yes |
| `nodes:remove` | request (we call) | ✅ yes |

**Dynamic / unresolved notify sites (verify by hand — `npm run service-contracts` prints the line numbers):**
- `notifyMethod (var)  (httpserver.go)`

## nvpair-engine-manager

| Method | Direction | In bridge? |
|---|---|---|
| `discovery:subscribe` | notification (we consume) | ✅ yes |
| `engine:install-progress` | notification (we consume) | ✅ yes |
| `engine:intent-changed` | notification (we consume) | ✅ yes |
| `engine:model-identity` | notification (we consume) | ➖ ignored |
| `engine:models-changed` | notification (we consume) | ✅ yes |
| `engine:pull-progress` | notification (we consume) | ✅ yes |
| `engine:ready` | notification (we consume) | ✅ yes |
| `engine:remote-progress` | notification (we consume) | ✅ yes |
| `engine:settings-changed` | notification (we consume) | ✅ yes |
| `engine:settings-disconnected` | notification (we consume) | ✅ yes |
| `engine:state-changed` | notification (we consume) | ✅ yes |
| `engine:switch-changed` | notification (we consume) | ✅ yes |
| `errors:clear` | notification (we consume) | ✅ yes |
| `errors:report` | notification (we consume) | ✅ yes |
| `engine:action` | request (we call) | ✅ yes |
| `engine:catalog` | request (we call) | ✅ yes |
| `engine:configure-launch` | request (we call) | ⚠️ not called |
| `engine:configured-ports` | request (we call) | ⚠️ not called |
| `engine:delete-model` | request (we call) | ✅ yes |
| `engine:describe` | request (we call) | ⚠️ not called |
| `engine:descriptors` | request (we call) | ✅ yes |
| `engine:errors` | request (we call) | ⚠️ not called |
| `engine:get-installed` | request (we call) | ✅ yes |
| `engine:get-launch` | request (we call) | ⚠️ not called |
| `engine:install` | request (we call) | ✅ yes |
| `engine:intent` | request (we call) | ✅ yes |
| `engine:load-model` | request (we call) | ✅ yes |
| `engine:logs` | request (we call) | ⚠️ not called |
| `engine:models` | request (we call) | ✅ yes |
| `engine:prepare-shutdown` | request (we call) | ➖ ignored |
| `engine:preview-launch` | request (we call) | ⚠️ not called |
| `engine:reconcile-resident` | request (we call) | ➖ ignored |
| `engine:remote-apply-settings` | request (we call) | ⚠️ not called |
| `engine:remote-copy-model` | request (we call) | ✅ yes |
| `engine:remote-delete-model` | request (we call) | ✅ yes |
| `engine:remote-get-installed` | request (we call) | ✅ yes |
| `engine:remote-get-settings` | request (we call) | ⚠️ not called |
| `engine:remote-install` | request (we call) | ✅ yes |
| `engine:remote-load-model` | request (we call) | ✅ yes |
| `engine:remote-management-get` | request (we call) | ✅ yes |
| `engine:remote-preview-settings` | request (we call) | ⚠️ not called |
| `engine:remote-pull-model` | request (we call) | ✅ yes |
| `engine:remote-start` | request (we call) | ✅ yes |
| `engine:remote-stop` | request (we call) | ✅ yes |
| `engine:remote-switch` | request (we call) | ⚠️ not called |
| `engine:remote-switch-status` | request (we call) | ⚠️ not called |
| `engine:remote-unload-model` | request (we call) | ✅ yes |
| `engine:restart` | request (we call) | ⚠️ not called |
| `engine:set-port` | request (we call) | ⚠️ not called |
| `engine:sleep` | request (we call) | ⚠️ not called |
| `engine:start` | request (we call) | ✅ yes |
| `engine:status` | request (we call) | ✅ yes |
| `engine:stop` | request (we call) | ✅ yes |
| `engine:switch-prepare` | request (we call) | ⚠️ not called |
| `engine:switch-reconcile` | request (we call) | ⚠️ not called |
| `engine:switch-recovery-ready` | request (we call) | ⚠️ not called |
| `engine:switch-step` | request (we call) | ⚠️ not called |
| `engine:uninstall` | request (we call) | ✅ yes |
| `engine:uninstall-managed` | request (we call) | ✅ yes |
| `engine:unload-model` | request (we call) | ✅ yes |
| `engine:wake` | request (we call) | ⚠️ not called |
| `error` | request (we call) | ✅ yes |
| `internal:set-reserved-port` | request (we call) | ⚠️ not called |

**Dynamic / unresolved notify sites (verify by hand — `npm run service-contracts` prints the line numbers):**
- `method (var)  (codec.go, 2 sites)`
- `method (var)  (executor.go)`

## nvpair-errors

| Method | Direction | In bridge? |
|---|---|---|
| `discovery:subscribe` | notification (we consume) | ✅ yes |
| `errors:update` | notification (we consume) | ✅ yes |
| `ready` | notification (we consume) | ✅ yes |
| `discovery:nodes` | request (we call) | ✅ yes |
| `errors:clear` | request (we call) | ✅ yes |
| `errors:get-initial` | request (we call) | ✅ yes |
| `errors:report` | request (we call) | ✅ yes |

## nvpair-job-scheduler

| Method | Direction | In bridge? |
|---|---|---|
| `ready` | notification (we consume) | ✅ yes |
| `schedule:priority` | notification (we consume) | ✅ yes |
| `discovery:nodes-changed` | request (we call) | ✅ yes |
| `scheduler:get-interval` | request (we call) | ⚠️ not called |
| `scheduler:get-status` | request (we call) | ⚠️ not called |
| `scheduler:set-interval` | request (we call) | ⚠️ not called |
| `scheduler:telemetry` | request (we call) | ➖ ignored |
| `scheduler:tick` | request (we call) | ⚠️ not called |
| `workloads:remove` | request (we call) | ✅ yes |
| `workloads:upsert` | request (we call) | ✅ yes |

## nvpair-manual-nodes

| Method | Direction | In bridge? |
|---|---|---|
| `errors:clear` | notification (we consume) | ✅ yes |
| `errors:report` | notification (we consume) | ✅ yes |
| `node/discovered` | notification (we consume) | ✅ yes |
| `node/removed` | notification (we consume) | ✅ yes |
| `node/updated` | notification (we consume) | ✅ yes |
| `ready` | notification (we consume) | ✅ yes |
| `node/add` | request (we call) | ✅ yes |
| `node/remove` | request (we call) | ✅ yes |
| `nodes/list` | request (we call) | ✅ yes |

## nvpair-node-info

| Method | Direction | In bridge? |
|---|---|---|
| `nodeinfo:observed-addresses` | notification (we consume) | ➖ ignored |
| `n/a` | request (we call) | ⚠️ not called |

## nvpair-node-scanner

| Method | Direction | In bridge? |
|---|---|---|
| `discovery:node-removed` | notification (we consume) | ➖ ignored |
| `discovery:node-telemetry` | notification (we consume) | ➖ ignored |
| `discovery:node-updated` | notification (we consume) | ➖ ignored |
| `errors:clear` | notification (we consume) | ✅ yes |
| `errors:report` | notification (we consume) | ✅ yes |
| `ready` | notification (we consume) | ✅ yes |
| `discovery:get-node-info` | request (we call) | ✅ yes |
| `discovery:get-nodes` | request (we call) | ✅ yes |
| `discovery:node-activity` | request (we call) | ➖ ignored |
| `discovery:peer-services` | request (we call) | ➖ ignored |
| `discovery:register` | request (we call) | ⚠️ not called |
| `discovery:reload-identity` | request (we call) | ⚠️ not called |
| `discovery:reload-trust` | request (we call) | ⚠️ not called |
| `discovery:set-observed-addresses` | request (we call) | ⚠️ not called |
| `discovery:unregister` | request (we call) | ⚠️ not called |
| `discovery:update-txt` | request (we call) | ⚠️ not called |

**Dynamic / unresolved notify sites (verify by hand — `npm run service-contracts` prints the line numbers):**
- `method (var)  (daemon.go, 3 sites)`
- `method (var)  (peer_services.go)`

## nvpair-node-settings

| Method | Direction | In bridge? |
|---|---|---|
| `connection/cluster-auto-sync` | notification (we consume) | ➖ ignored |
| `connection/cluster-identity` | notification (we consume) | ✅ yes |
| `ready` | notification (we consume) | ✅ yes |
| `settings/get-cluster-auto-sync` | request (we call) | ⚠️ not called |
| `settings/get-cluster-friendly-name` | request (we call) | ✅ yes |
| `settings/get-cluster-id` | request (we call) | ✅ yes |
| `settings/get-force-ports` | request (we call) | ⚠️ not called |
| `settings/set-cluster-auto-sync` | request (we call) | ⚠️ not called |
| `settings/set-cluster-friendly-name` | request (we call) | ✅ yes |
| `settings/set-cluster-id` | request (we call) | ✅ yes |
| `settings/set-force-ports` | request (we call) | ⚠️ not called |

## nvpair-proxy

| Method | Direction | In bridge? |
|---|---|---|
| `admission/state` | notification (we consume) | ➖ ignored |
| `admission/strict-readiness` | notification (we consume) | ➖ ignored |
| `discovery:subscribe` | notification (we consume) | ✅ yes |
| `error` | notification (we consume) | ✅ yes |
| `errors:clear` | notification (we consume) | ✅ yes |
| `errors:report` | notification (we consume) | ✅ yes |
| `node/activity` | notification (we consume) | ➖ ignored |
| `node/discovered` | notification (we consume) | ✅ yes |
| `node/removed` | notification (we consume) | ✅ yes |
| `node/selection-changed` | notification (we consume) | ➖ ignored |
| `node/updated` | notification (we consume) | ✅ yes |
| `performance:sample` | notification (we consume) | ➖ ignored |
| `proxy/request` | notification (we consume) | ✅ yes |
| `proxy/request-started` | notification (we consume) | ➖ ignored |
| `ready` | notification (we consume) | ✅ yes |
| `discovery:nodes` | request (we call) | ✅ yes |
| `facade/enable` | request (we call) | ⚠️ not called |
| `node/add-manual` | request (we call) | ✅ yes |
| `node/admission-snapshot` | request (we call) | ⚠️ not called |
| `node/dispatch-resident` | request (we call) | ➖ ignored |
| `node/finish-resident` | request (we call) | ➖ ignored |
| `node/invalidate-engine` | request (we call) | ⚠️ not called |
| `node/observe-resident` | request (we call) | ➖ ignored |
| `node/prepare-resident` | request (we call) | ➖ ignored |
| `node/remove-manual` | request (we call) | ✅ yes |
| `node/renew-egress-lease` | request (we call) | ➖ ignored |
| `node/revoke-egress-lease` | request (we call) | ➖ ignored |
| `node/select` | request (we call) | ✅ yes |
| `node/selected` | request (we call) | ⚠️ not called |
| `node/set-availability` | request (we call) | ⚠️ not called |
| `node/set-engine-drain` | request (we call) | ⚠️ not called |
| `node/set-engine-intent` | request (we call) | ⚠️ not called |
| `node/set-local-backend` | request (we call) | ⚠️ not called |
| `node/set-model-identity` | request (we call) | ⚠️ not called |
| `node/set-policy` | request (we call) | ⚠️ not called |
| `node/set-priority` | request (we call) | ✅ yes |
| `node/set-residency` | request (we call) | ⚠️ not called |
| `nodes/list` | request (we call) | ✅ yes |

**Dynamic / unresolved notify sites (verify by hand — `npm run service-contracts` prints the line numbers):**
- `n.method (var)  (admission.go)`
- `method (var)  (codec.go, 2 sites)`
- `method (var)  (proxy.go, 2 sites)`

## nvpair-tui

| Method | Direction | In bridge? |
|---|---|---|
| `app:ready` | request (we call) | ✅ yes |
| `cluster:identity-changed` | request (we call) | ✅ yes |
| `cluster:invite-canceled` | request (we call) | ✅ yes |
| `cluster:invite-declined` | request (we call) | ✅ yes |
| `cluster:invite-expired` | request (we call) | ✅ yes |
| `cluster:invite-failed` | request (we call) | ✅ yes |
| `cluster:invite-received` | request (we call) | ✅ yes |
| `cluster:remote-management-changed` | request (we call) | ✅ yes |
| `discovery:nodes-changed` | request (we call) | ✅ yes |
| `engine:install-progress` | request (we call) | ✅ yes |
| `engine:intent-changed` | request (we call) | ✅ yes |
| `engine:models-changed` | request (we call) | ✅ yes |
| `engine:pull-progress` | request (we call) | ✅ yes |
| `engine:ready` | request (we call) | ✅ yes |
| `engine:remote-progress` | request (we call) | ✅ yes |
| `engine:settings-changed` | request (we call) | ✅ yes |
| `engine:settings-disconnected` | request (we call) | ✅ yes |
| `engine:state-changed` | request (we call) | ✅ yes |
| `engine:switch-changed` | request (we call) | ✅ yes |
| `error` | request (we call) | ✅ yes |
| `errors:update` | request (we call) | ✅ yes |
| `node:availability-changed` | request (we call) | ✅ yes |
| `nodes:changed` | request (we call) | ✅ yes |
| `performance:changed` | request (we call) | ✅ yes |
| `policy:changed` | request (we call) | ✅ yes |
| `service/broker-restarted` | request (we call) | ✅ yes |
| `workloads:remove` | request (we call) | ✅ yes |
| `workloads:upsert` | request (we call) | ✅ yes |

**Dynamic / unresolved notify sites (verify by hand — `npm run service-contracts` prints the line numbers):**
- `ctx (var)  (client.go, 2 sites)`

## nvpair-ui-broker

| Method | Direction | In bridge? |
|---|---|---|
| `app:ready` | notification (we consume) | ✅ yes |
| `cluster:set-local-services` | notification (we consume) | ➖ ignored |
| `discovery:node-activity` | notification (we consume) | ➖ ignored |
| `discovery:nodes` | notification (we consume) | ✅ yes |
| `discovery:nodes-changed` | notification (we consume) | ✅ yes |
| `discovery:peer-services` | notification (we consume) | ➖ ignored |
| `engine:model-mutation-reply` | notification (we consume) | ➖ ignored |
| `engine:node-projection` | notification (we consume) | ➖ ignored |
| `engine:restore-enabled` | notification (we consume) | ➖ ignored |
| `engine:settings-changed` | notification (we consume) | ✅ yes |
| `engine:settings-projection` | notification (we consume) | ➖ ignored |
| `errors:clear` | notification (we consume) | ✅ yes |
| `errors:report` | notification (we consume) | ✅ yes |
| `errors:update` | notification (we consume) | ✅ yes |
| `log/set-level` | notification (we consume) | ✅ yes |
| `performance:changed` | notification (we consume) | ✅ yes |
| `scheduler:telemetry` | notification (we consume) | ➖ ignored |
| `workloads:remove` | notification (we consume) | ✅ yes |
| `workloads:upsert` | notification (we consume) | ✅ yes |
| `admission/state` | request (we call) | ➖ ignored |
| `admission/strict-readiness` | request (we call) | ➖ ignored |
| `admission/unload` | request (we call) | ⚠️ not called |
| `admission/wake` | request (we call) | ⚠️ not called |
| `connection/cluster-auto-sync` | request (we call) | ➖ ignored |
| `connection/cluster-identity` | request (we call) | ✅ yes |
| `discovery:get-node-info` | request (we call) | ✅ yes |
| `discovery:get-nodes` | request (we call) | ✅ yes |
| `discovery:node-discovered` | request (we call) | ⚠️ not called |
| `discovery:node-removed` | request (we call) | ➖ ignored |
| `discovery:node-telemetry` | request (we call) | ➖ ignored |
| `discovery:node-updated` | request (we call) | ➖ ignored |
| `discovery:subscribe` | request (we call) | ✅ yes |
| `discovery:unsubscribe` | request (we call) | ⚠️ not called |
| `engine:apply-settings` | request (we call) | ✅ yes |
| `engine:get-settings` | request (we call) | ✅ yes |
| `engine:install` | request (we call) | ✅ yes |
| `engine:intent-changed` | request (we call) | ✅ yes |
| `engine:models-changed` | request (we call) | ✅ yes |
| `engine:preview-settings` | request (we call) | ✅ yes |
| `engine:restart` | request (we call) | ⚠️ not called |
| `engine:set-port` | request (we call) | ⚠️ not called |
| `engine:start` | request (we call) | ✅ yes |
| `engine:status` | request (we call) | ✅ yes |
| `engine:stop` | request (we call) | ✅ yes |
| `engine:subscribe` | request (we call) | ✅ yes |
| `engine:switch` | request (we call) | ✅ yes |
| `engine:switch-cancel` | request (we call) | ➖ ignored |
| `engine:switch-changed` | request (we call) | ✅ yes |
| `engine:switch-request` | request (we call) | ➖ ignored |
| `engine:switch-status` | request (we call) | ✅ yes |
| `engine:uninstall` | request (we call) | ✅ yes |
| `engine:unsubscribe` | request (we call) | ⚠️ not called |
| `engine:wake` | request (we call) | ⚠️ not called |
| `errors:get-initial` | request (we call) | ✅ yes |
| `node/activity` | request (we call) | ➖ ignored |
| `node/add` | request (we call) | ✅ yes |
| `node/discovered` | request (we call) | ✅ yes |
| `node/remove` | request (we call) | ✅ yes |
| `node/removed` | request (we call) | ✅ yes |
| `node/updated` | request (we call) | ✅ yes |
| `node:set-availability` | request (we call) | ✅ yes |
| `nodes/list` | request (we call) | ✅ yes |
| `performance:received` | request (we call) | ➖ ignored |
| `performance:sample` | request (we call) | ➖ ignored |
| `performance:status` | request (we call) | ➖ ignored |
| `policy:cancel` | request (we call) | ➖ ignored |
| `policy:get` | request (we call) | ✅ yes |
| `policy:request` | request (we call) | ➖ ignored |
| `policy:set` | request (we call) | ✅ yes |
| `ready` | request (we call) | ✅ yes |
| `workloads:cancel` | request (we call) | ✅ yes |
| `workloads:get-initial` | request (we call) | ✅ yes |
| `workloads:subscribe` | request (we call) | ✅ yes |
| `workloads:unsubscribe` | request (we call) | ⚠️ not called |

**Dynamic / unresolved notify sites (verify by hand — `npm run service-contracts` prints the line numbers):**
- `method (var)  (broker.go, 3 sites)`
- `method (var)  (clustermanager.go)`
- `method (var)  (errors.go)`
- `msg.Method (var)  (nodeinfo.go)`
- `method (var)  (policy.go)`
- `method (var)  (proxy.go, 2 sites)`
- `method (var)  (remote_management.go)`
- `method (var)  (rpcworker.go, 3 sites)`
- `method (var)  (workload_projection.go, 2 sites)`
- `msg.Method (var)  (workloadmanager.go)`

## nvpair-workload-manager

| Method | Direction | In bridge? |
|---|---|---|
| `discovery:subscribe` | notification (we consume) | ✅ yes |
| `performance:received` | notification (we consume) | ➖ ignored |
| `performance:status` | notification (we consume) | ➖ ignored |
| `ready` | notification (we consume) | ✅ yes |
| `workloads:cancel` | notification (we consume) | ✅ yes |
| `workloads:remove` | notification (we consume) | ✅ yes |
| `workloads:upsert` | notification (we consume) | ✅ yes |
| `discovery:nodes` | request (we call) | ✅ yes |
| `performance:capabilities` | request (we call) | ⚠️ not called |
| `performance:publish` | request (we call) | ⚠️ not called |
| `performance:summary` | request (we call) | ⚠️ not called |

**Dynamic / unresolved notify sites (verify by hand — `npm run service-contracts` prints the line numbers):**
- `method (var)  (performance.go)`

