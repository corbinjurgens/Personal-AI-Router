<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dependency license sources

Run `npm run licenses` from `desktop/` to regenerate the repository's
`THIRD_PARTY_NOTICES.md`. The packaged application ships that file. Generation
reads installed npm packages and cached Go modules; Go dependency and toolchain
downloads are disabled. Install dependencies and populate the Go module cache
before generation. No license URL is fetched during generation or packaging.

Go module grants must match the exact module version and complete text digest
in `go-license-file-hashes.ts`. Missing, truncated, changed or unreviewed module
licenses stop generation instead of producing a placeholder. Review the entire
version-associated module grant before updating that inventory. LICENSE,
COPYING and NOTICE texts are retained together in a deterministic order, with
identical texts deduplicated. A module without a LICENSE can use a version-bound
supplement; go-localereader declares MIT in its pinned version's README.

For npm dependencies, the generator checks the installed package name, version
and declared license against the scanner identity. It reads the complete
package LICENSE or COPYING file directly, rejects README fallbacks, and checks
required grant, condition and disclaimer clauses for the supported license
families. The package file must also match the exact version's reviewed SHA-256
in `npm-license-file-hashes.ts`, so clause keywords alone cannot admit an edited
or incomplete grant. An unfamiliar license family requires review before generation can
proceed. For an SPDX OR expression, one complete alternative is sufficient.

When a published package lacks a LICENSE, `license-supplements.ts` provides
an exact `name@version` supplement, its declared license, an immutable source
URL, the version association and the SHA-256 of the original text. The generator
checks the declaration and hash before using it. A dependency upgrade cannot
reuse the prior version's supplement automatically.

The upstream supplements were checked against the npm registry version's
`gitHead` on 2026-10-26, including the original release supplements for
overlayscrollbars, react-flip-toolkit and saxes. A monorepo's root LICENSE applies
to its packages. Where that source has no LICENSE, the owner's distribution
policy uses the declared license's standard SPDX text, explicitly marked
`standard SPDX text for declared license, no upstream LICENSE file at <version>`
in the metadata and generated notice. Those texts come from the pinned SPDX
license-list-data revision in the source URL; copyright template placeholders
remain part of the standard text and are not assertions about package authors.
Existing package README copyright notices are retained separately, with their
version-associated source URL. For react-remove-scroll-bar, the npm `gitHead`
is unavailable upstream (HTTP 404); its published package also has no LICENSE,
so its standard-text source records that limitation explicitly.

To update a supplement, verify the package's exact version and declared license,
locate the version-associated upstream grant (or document its absence and use
the standard SPDX text), and retain the original bytes and their hash. Never
copy a mutable default-branch license or substitute a README usage section.
For packages shipping their own LICENSE, review the entire file before adding
its version and digest to `npm-license-file-hashes.ts`; do not automatically
approve a new digest just because the filename or clause markers match.
Run the notice regression tests, regenerate with `npm run licenses`, and review
the generated diff. Add full-clause validation for any new license family.
