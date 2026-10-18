// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package secretenv

import (
	"runtime"
	"slices"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
)

const sentinel = "sentinel-secret-value-7f3a"

func TestRemove(t *testing.T) {
	test := func(name string, env, want []string) {
		t.Run(name, func(t *testing.T) {
			if got := Remove(env); !slices.Equal(got, want) {
				t.Fatalf("Remove(%q) = %q, want %q", env, got, want)
			}
		})
	}

	test("drops every reserved name and keeps the rest in order",
		[]string{"PATH=/bin", "NVPAIR_PROVIDER_SECRET_A=" + sentinel, "HOME=/h", "NVPAIR_PROVIDER_SECRET_B=" + sentinel},
		[]string{"PATH=/bin", "HOME=/h"})
	test("drops case variants of the namespace",
		[]string{"nvpair_provider_secret_a=" + sentinel, "Nvpair_Provider_Secret_B=" + sentinel, "KEEP=1"},
		[]string{"KEEP=1"})
	test("drops a reserved name without a value separator",
		[]string{"NVPAIR_PROVIDER_SECRET_A", "KEEP=1"},
		[]string{"KEEP=1"})
	test("drops duplicate reserved entries without error",
		[]string{"NVPAIR_PROVIDER_SECRET_A=1", "NVPAIR_PROVIDER_SECRET_A=2"},
		[]string{})
	test("keeps names that only resemble the namespace",
		[]string{"NVPAIR_PROVIDER_SECRET=1", "XNVPAIR_PROVIDER_SECRET_A=2", "NVPAIR_PROVIDER_SECRETS=3"},
		[]string{"NVPAIR_PROVIDER_SECRET=1", "XNVPAIR_PROVIDER_SECRET_A=2", "NVPAIR_PROVIDER_SECRETS=3"})
	test("keeps a value that mentions the namespace",
		[]string{"NOTE=NVPAIR_PROVIDER_SECRET_A=x"},
		[]string{"NOTE=NVPAIR_PROVIDER_SECRET_A=x"})
	test("keeps Windows per-drive entries",
		[]string{`=C:=C:\work`, "NVPAIR_PROVIDER_SECRET_A=" + sentinel},
		[]string{`=C:=C:\work`})
	test("keeps unrelated duplicates for the launcher to resolve",
		[]string{"PATH=/a", "PATH=/b"},
		[]string{"PATH=/a", "PATH=/b"})
}

func TestRemoveDoesNotModifyInput(t *testing.T) {
	env := []string{"NVPAIR_PROVIDER_SECRET_A=" + sentinel, "KEEP=1"}
	original := slices.Clone(env)

	Remove(env)

	if !slices.Equal(env, original) {
		t.Fatalf("input changed to %q, want %q", env, original)
	}
}

func TestFilterRetainsConfiguredReferences(t *testing.T) {
	refs := []nodepolicy.CredentialRef{"NVPAIR_PROVIDER_SECRET_A"}
	test := func(name string, foldCase bool, env, want []string) {
		t.Run(name, func(t *testing.T) {
			got, err := filter(env, refs, foldCase)
			if err != nil {
				t.Fatalf("filter: %v", err)
			}
			if !slices.Equal(got, want) {
				t.Fatalf("filter(%q) = %q, want %q", env, got, want)
			}
		})
	}

	test("keeps the configured reference and drops others",
		false,
		[]string{"PATH=/bin", "NVPAIR_PROVIDER_SECRET_A=" + sentinel, "NVPAIR_PROVIDER_SECRET_B=" + sentinel},
		[]string{"PATH=/bin", "NVPAIR_PROVIDER_SECRET_A=" + sentinel})
	test("case-sensitive platform drops a case variant of the reference",
		false,
		[]string{"NVPAIR_PROVIDER_SECRET_A=" + sentinel, "nvpair_provider_secret_a=other"},
		[]string{"NVPAIR_PROVIDER_SECRET_A=" + sentinel})
	test("case-sensitive platform keeps nothing for a case variant alone",
		false,
		[]string{"nvpair_provider_secret_a=" + sentinel},
		[]string{})
	test("case-insensitive platform keeps a case variant of the reference",
		true,
		[]string{"Nvpair_Provider_Secret_A=" + sentinel, "NVPAIR_PROVIDER_SECRET_B=" + sentinel},
		[]string{"Nvpair_Provider_Secret_A=" + sentinel})
	test("a missing reference is not an error",
		true,
		[]string{"PATH=/bin"},
		[]string{"PATH=/bin"})
}

func TestFilterRejectsAmbiguousReference(t *testing.T) {
	refs := []nodepolicy.CredentialRef{"NVPAIR_PROVIDER_SECRET_A"}
	test := func(name string, foldCase bool, env []string) {
		t.Run(name, func(t *testing.T) {
			got, err := filter(env, refs, foldCase)
			if err == nil {
				t.Fatalf("filter(%q) = %q, want an ambiguity error", env, got)
			}
			if !strings.Contains(err.Error(), "NVPAIR_PROVIDER_SECRET_A") {
				t.Fatalf("error %q does not name the reference", err)
			}
			if strings.Contains(err.Error(), sentinel) {
				t.Fatalf("error %q contains an environment value", err)
			}
		})
	}

	test("exact duplicate on a case-sensitive platform",
		false,
		[]string{"NVPAIR_PROVIDER_SECRET_A=" + sentinel, "NVPAIR_PROVIDER_SECRET_A=" + sentinel + "2"})
	test("case-equivalent duplicate on a case-insensitive platform",
		true,
		[]string{"NVPAIR_PROVIDER_SECRET_A=" + sentinel, "nvpair_provider_secret_a=" + sentinel + "2"})
}

func TestFilterRejectsNoncanonicalReference(t *testing.T) {
	test := func(name string, ref nodepolicy.CredentialRef) {
		t.Run(name, func(t *testing.T) {
			got, err := filter([]string{"PATH=/bin"}, []nodepolicy.CredentialRef{ref}, false)
			if err == nil {
				t.Fatalf("filter = %q, want a reference error", got)
			}
		})
	}

	test("lowercase reference", "nvpair_provider_secret_a")
	test("reference outside the namespace", "PATH")
	test("bare prefix", Prefix)
}

func TestRetainUsesPlatformCaseRules(t *testing.T) {
	env := []string{"nvpair_provider_secret_a=" + sentinel}

	got, err := Retain(env, []nodepolicy.CredentialRef{"NVPAIR_PROVIDER_SECRET_A"})
	if err != nil {
		t.Fatalf("Retain: %v", err)
	}

	want := []string{}
	if runtime.GOOS == "windows" {
		want = env
	}
	if !slices.Equal(got, want) {
		t.Fatalf("Retain on %s = %q, want %q", runtime.GOOS, got, want)
	}
}

func TestRetainWithoutReferencesMatchesRemove(t *testing.T) {
	env := []string{"PATH=/bin", "NVPAIR_PROVIDER_SECRET_A=" + sentinel, "nvpair_provider_secret_b=" + sentinel}

	got, err := Retain(env, nil)
	if err != nil {
		t.Fatalf("Retain: %v", err)
	}

	if want := Remove(env); !slices.Equal(got, want) {
		t.Fatalf("Retain(nil) = %q, Remove = %q", got, want)
	}
}
