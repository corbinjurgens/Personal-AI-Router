<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Host-local secret store

`secretstore.New()` is lazy and performs no native calls. The owning Go service
uses `Set`, `Get` and `Delete` directly, with a context, fixed purpose (for example
`engine-api-key`) and validated engine id. Purpose and id each match
`^[a-z][a-z0-9-]{0,63}$`. The service namespace is
`com.nvidia.pair.secretstore.` plus purpose; the account is the id. Windows uses
the target `service/account`. Names are non-secret opaque identifiers, never
paths, URLs, labels or arbitrary user text. There is no enumeration or namespace
selection API.

Values are nonempty valid UTF-8, at most 512 bytes. Clear calls `Delete`.
Input and output buffers are copied. Callers must drop references after use;
Go memory is not guaranteed to be securely erased. Never log values or native
diagnostics, or expose retrieval through renderer, peer or credential RPC APIs.
There is no plaintext file, environment fallback, cache or daemon in this package.
This unused foundation does not change any shipped binary or existing Strata
key-file contract. Consumers remain separate work.

Each Store serializes operations through native completion. Initialization
failures are retried on the next operation. Operations are never automatically
retried after an ambiguous write. Missing Get alone returns `ErrNotFound`;
missing Delete succeeds. Invalid input returns `ErrInvalidName` or
`ErrInvalidSecret`; all other native errors and malformed replies return
`ErrUnavailable`, without native text. Use `errors.Is` with these sentinels.
Cross-process writes are not coordinated: one owning service must write each
purpose/id. A failed write may have completed durably if transport or cleanup
failed; callers must re-read rather than assume rollback.

## Platform prerequisites

- macOS uses Security and CoreFoundation through purego, with the current user's
  default file-based Keychain. Queries select exact generic passwords, exclude
  synchronizable items and disallow authentication UI. Since the file-based
  Keychain does not honor that query flag by itself, a package-wide mutex guards
  reading, disabling and restoring the process-wide user-interaction setting
  through the complete operation, across Store instances. Reading or disabling
  failure prevents item access. Restoration failure returns unavailable even
  after a completed write. It changes no OS-wide preference. Other code in the
  same process using this setting must coordinate with this owner. A locked or
  denied keychain fails without an interaction fallback.
- Windows uses wincred generic credentials with explicit `PersistLocalMachine`,
  current-user credentials on that machine, rather than Enterprise persistence.
  Windows and macOS calls are synchronous. Context is checked before calling;
  cancellation during a native call does not replace its completed outcome.
- Linux needs an existing unlocked default Secret Service collection and a local
  authenticated Unix session bus for the service's user. The bus address comes
  from `DBUS_SESSION_BUS_ADDRESS`, or the standard `/run/user/<uid>/bus` socket
  when unset. TCP and autolaunch addresses are refused. Each operation opens a
  private connection, authenticates with EXTERNAL as the current uid and closes
  its connection and session. Dialing, authentication, calls and cleanup share
  a three-second maximum; a shorter caller deadline wins. Caller cancellation
  and deadline return context sentinels; the internal timeout is unavailable.
  The package checks collection and item locks, refuses duplicates and prompts,
  and never unlocks, invokes Prompt or creates a collection. The Secret Service
  `plain` algorithm describes transport over the local authenticated bus, not
  at-rest storage: the native collection owns at-rest protection.
- Other operating systems return unavailable.

Native access depends on the actual service launch identity and session. A
desktop terminal's access does not prove background-service access. These stores
do not protect against every other process with the same user's authority or an
administrator. No per-read user presence or access-group policy is installed.

## Validation

Default tests use only injected fake adapters and fake native calls. They never
initialize a native store. Run `go test -race ./secretstore` and `go test ./...`
from `services/shared`; platform fake tests compile only on their target OS.
Cross-compilation and vet establish build validity, not native operation.

Native validation is **UNTESTED** on macOS, Windows and Linux. Do not run the
following opt-in command without an authorized, attended validation interval:

```sh
go test ./secretstore -run '^TestNativeSecretStore$' -args -secretstore-native
```

The opt-in test generates a random validated `native-` id and fake values. It
writes, reads, replaces and deletes the exact entry through successive child
test processes, so persistence cannot be supplied by process memory. Inputs
travel over anonymous stdin pipes, never argv, environment or stdout; failure
messages omit values. Cleanup touches only that random entry. A killed test may
leave that exact test entry requiring authorized cleanup. On Windows use the
project's fixed-path test wrapper and firewall setup before native execution.
Native acceptance must record the OS, session and process identity, persistence,
replacement, deletion, cleanup and prompt observation. Locked-keychain macOS
validation must establish no prompt by observation, rather than inspecting flags.
No native readiness is established until those platform checks pass.

The shared requirements pin purego v0.10.2, wincred v1.2.3 and godbus v5.1.0.
The shipped-binary dependency collector uses `go list -deps`; this unlinked
foundation adds no third-party notices until a consumer imports it.

Protocol references: [Apple SecItem declarations](https://github.com/apple-oss-distributions/Security/blob/main/keychain/headers/SecItem.h)
and the [Secret Service API](https://specifications.freedesktop.org/secret-service/latest/).
