<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Estimate calibration fixtures

These fixtures exercise the advisory estimator and calibration harness. They
are invented data, not native measurements, accuracy evidence or production
calibration. The production `modelEstimateCalibrations` table remains empty.

| Fixture | Role | Provenance |
| --- | --- | --- |
| `synthetic-dense-cpu.json` | synthetic | Invented llama CPU weights and diagnostic cache/workspace/host peak, 2026-10-30, synthetic machine, b11146 |
| `synthetic-moe-cuda.json` | synthetic | Invented qwen2moe allocation and diagnostic intervals/device peak, 2026-10-30, synthetic machine, b11146 |

The fixture contract is version 1, checked by
`modelsettings_estimate_calibration_test.go`. JSON is bounded to 8 MiB and
unknown fields and trailing documents are rejected. Fields are:

- `role`: `fit`, `check` or `synthetic`. Synthetic rows never count as evidence.
- `class`: optional, checked against the estimator's classification of the
  decoded facts when present. The exporter omits it so the harness uses the
  existing estimator owner rather than a second classifier.
- `backend`, `nativePin`: finite backend family and full pinned native source
  revision. Provenance records the closed machine alias (`synthetic`, `pc3`,
  `msi`), ISO date and normalized recognized native build. The exporter uses
  the stored recognized upstream pin as the full b11146 build alias, not an
  executable hash or newly probed native build. These are source provenance,
  not a claim of native execution or complete accuracy.
- `facts`: bounded `gguffacts.Facts` fields. The synthetic header test builds
  disposable GGUF bytes and calls the existing `gguffacts.Decode`, checking
  the decoded shards against the fixtures. Decoder usage counters are omitted.
  No second metadata reader exists here. The initial privacy allowlist covers
  the llama/qwen dense and MoE dimensions and canonical tensor names exercised
  here; other architectures or fields require an explicit reviewed extension.
  Templates, arbitrary tensor names and unrelated numeric metadata are rejected.
- `artifacts`: ordered opaque identity, role, size and sampled-local-binding
  assurance summaries. `sample`
  retains the stored configuration key and summaries; they must match the
  fixture join exactly. Synthetic digests are placeholders. Equality in this
  harness does not prove file identity. The exporter establishes its join
  through the authoritative binding owner.
- `allocation`: finite context, slots, CPU or layer allocation, typed effective
  options and the SWA choice. It excludes target-owned paths, raw arguments,
  presets, environment and launch authority. It cannot be applied or launched.
- `observed`: per-component lower/upper byte intervals, the stored comparison
  status and a `supported` or `diagnostic` correspondence. Byte strings preserve
  integer precision. The CPU weight interval illustrates outward rounding of
  the native two-decimal MiB report; it intentionally exceeds a derived
  zero-width row. Uncalibrated rows are reported without coverage assertions.
- `poolPeaks`: optional finite pool and peak bytes tagged `diagnostic`. A pool
  peak never sets or checks a component band.

Only Task 14's compatible CPU weights correspondence may be `supported`:
CPU allocation, model-role artifacts and compatible weight comparison.
The upstream comparison owner additionally checks topology, complete capture,
CPU-labelled model rows and pool identity (`modelBufferComparisons` in
`model_buffer_capture.go`). An exporter must preserve that stored decision,
not manufacture compatibility from fixture data. KV, compute and GPU weight
intervals remain diagnostic. Backend provenance alone cannot supply an actual
device map; the CUDA fixture intentionally retains unmapped placement.

For every injected or later reviewed table entry the harness requires distinct
existing eligible `fit` fixture names, at least three, and eligible held-out
`check` coverage for the same component, class and backend. It checks fit and
check intervals against emitted calibrated ranges. Failure names the component
and class. Reported signed errors are `observed - expected` at both interval
ends. Only supported evidence can validate a band. A diagnostic row is reported
even when outside a calibrated range. Unit-constructed fit/check cases test
these rules and restore the global table; they are not checked-in evidence.

Privacy follows model-setup Task 14: no paths, model content, prompts, messages,
responses, templates, tokens, search strings, raw logs, native arguments,
environment, credentials, URLs, device descriptions, arbitrary diagnostic text
or key material. Source facts are projected through the closed schema rather
than copying arbitrary metadata. Binding identities are opaque local digests,
not full content hashes or secrets. Exporting never authorizes additional loads
or collection on installed models. Real fit/check collection belongs to the
separately authorized estimation-direction Task 14 interval.

The standalone exporter uses the module-private `internal/artifactbinding`
owner also used by the manager. It captures original regular-file evidence,
samples the existing first/last 64 KiB change detector, opens checked handles
for `gguffacts.Decode`, and revalidates the original evidence and sample after
decoding. It preserves configured symlink spelling and resolved target in the
private join, hashing the exact ordered `ArtifactBinding` JSON representation.
Neither those paths nor the sample bytes appear in fixtures. Same-stat middle
modifications outside the samples remain undetected, as with the manager; this
is neither a content-integrity nor publisher/native proof.

Run it from `tools/calibration-export` with an explicit sample-store file, new
output directory and ordered artifact selectors (repeat `-artifact` for shards,
then draft and companion roles):

```sh
go run . -store /temporary/model-buffer-samples.json -out /temporary/fixtures \
  -machine synthetic -role synthetic -artifact model=/temporary/synthetic.gguf
```

Selectors are not evidence. The complete ordered role/size/identity set must
match every store sample's existing configuration key. Missing keys, missing
finite captured options, incomplete or mismatched sets and changed files refuse
the export before publishing any fixture. There is no key reconstruction,
old-path lookup or defaulting of missing effective options. SWAFull is omitted
when unavailable, not inferred from an opaque key. The initial closed privacy
and class allowlist can refuse otherwise valid GGUFs. CPU weights reuse the
stored compatible comparison only with consistent CPU model rows and topology;
other bounded rows remain diagnostic. No pool peaks are invented from native
component rows. Output uses new `sample-<index>.json` files; reviewed evidence
should use Task 14's class/backend naming convention. Machine and role flags
are provenance, not permission to acquire samples or load models. Neither
service build script references `calibration-export`.

Run the manager selector, plain and with `-race`:

```sh
go test -count=1 -run '^TestModelSettingsEstimateCalibration' .
go test -race -count=1 -run '^TestModelSettingsEstimateCalibration' .
```

From the exporter directory run `go test -count=1 .`. No real accuracy or
calibrated confidence claim follows from synthetic success.
