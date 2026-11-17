// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"path"
	"regexp"
	"sort"
	"strings"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

// engine:catalog-files lists the quantizations one Hugging Face GGUF repository
// offers, so an operator who pastes a repository name or link can pick any of
// them rather than only the catalogue's fixed Q4_K_M. Each entry's Name is
// pull-ready for llama.cpp, like CatalogModel.Name.

// CatalogFilesResult is the engine:catalog-files reply.
type CatalogFilesResult struct {
	// Repo is the normalised owner/name, spelled as the provider spells it.
	Repo string `json:"repo"`
	// Revision is the 40-hex commit the listing was read at.
	Revision  string             `json:"revision"`
	Files     []CatalogQuantFile `json:"files"`
	FetchedAt string             `json:"fetchedAt"`
}

// CatalogQuantFile is one quantization of a repository, all its shards counted.
type CatalogQuantFile struct {
	Quantization string `json:"quantization"`
	// Name is pull-ready: "<repo>:<quantization>".
	Name      string `json:"name"`
	SizeBytes int64  `json:"sizeBytes"`
	Shards    int    `json:"shards"`
}

// catalogFilesParams is the engine:catalog-files request.
type catalogFilesParams struct {
	Engine string `json:"engine"`
	Repo   string `json:"repo"`
}

// maxCatalogRepoInput bounds the pasted repository text before it is parsed.
const maxCatalogRepoInput = 300

var (
	// ggufQuantization is a quantization label as GGUF file names carry it.
	ggufQuantization = regexp.MustCompile(`(?i)^(UD-)?(I?Q[0-9]+_[A-Z0-9_]+|Q[0-9]+_[0-9]|BF16|F16|F32|MXFP4[A-Z0-9_]*)$`)
	hexRevision      = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// errCatalogRepoInput marks input that is not a repository, which the method
// reports as invalid parameters rather than a provider failure.
var errCatalogRepoInput = errors.New("Not a Hugging Face model repository")

// normalizeCatalogRepo turns what an operator pastes into owner/name: the bare
// id, the id with a :TAG, or a huggingface.co or hf.co link to the repository
// or anything inside it.
func normalizeCatalogRepo(input string) (string, error) {
	text := strings.TrimSpace(input)
	refuse := func() (string, error) {
		shown := text
		if len(shown) > 80 {
			shown = shown[:80]
		}
		return "", fmt.Errorf("%w: %s", errCatalogRepoInput, shown)
	}
	if len(input) > maxCatalogRepoInput || text == "" {
		return refuse()
	}
	lower := strings.ToLower(text)
	repo := text
	for _, host := range []string{"huggingface.co/", "hf.co/"} {
		for _, scheme := range []string{"https://", "http://", ""} {
			prefix := scheme + host
			if strings.HasPrefix(lower, prefix) {
				rest := text[len(prefix):]
				if cut := strings.IndexAny(rest, "?#"); cut >= 0 {
					rest = rest[:cut]
				}
				segments := strings.Split(rest, "/")
				if len(segments) < 2 {
					return refuse()
				}
				repo = segments[0] + "/" + segments[1]
				break
			}
		}
		if repo != text {
			break
		}
	}
	if repo == text {
		if cut := strings.IndexByte(repo, ':'); cut >= 0 {
			repo = repo[:cut]
		}
	}
	if !llamaCPPRepoPattern.MatchString(repo) {
		return refuse()
	}
	return repo, nil
}

// CatalogFiles lists the quantizations of one repository, cached per repository
// like a catalogue search.
func (c *catalogService) CatalogFiles(ctx context.Context, engine, repo string) (CatalogFilesResult, error) {
	if c.adapterFor(engine) != engines.NameLlamaCPP {
		return CatalogFilesResult{}, fmt.Errorf("no model catalog for engine %q", engine)
	}
	repo, err := normalizeCatalogRepo(repo)
	if err != nil {
		return CatalogFilesResult{}, err
	}
	c.searchMu.Lock()
	cache := c.repoFiles.entry(strings.ToLower(repo), func() *listingCache[CatalogFilesResult] {
		return &listingCache[CatalogFilesResult]{
			label: "repository file listing",
			fetch: func(ctx context.Context) (CatalogFilesResult, error) { return c.fetchRepoFiles(ctx, repo) },
		}
	})
	c.searchMu.Unlock()
	res, at, err := cache.get(ctx)
	if err != nil {
		return CatalogFilesResult{}, err
	}
	res.FetchedAt = at.UTC().Format(time.RFC3339)
	return res, nil
}

// fetchRepoFiles reads the repository's current file list. Only an explicitly
// open repository is listed: a gated or private one fails to download without
// credentials, which PAIR does not hold.
func (c *catalogService) fetchRepoFiles(ctx context.Context, repo string) (CatalogFilesResult, error) {
	body, err := c.fetchHFBody(ctx, c.endpoint()+"/"+repo+"/revision/main?blobs=true")
	if err != nil {
		var status hfStatusError
		// The provider answers 401 rather than 404 for a repository that does
		// not exist, and the same for a private one.
		if errors.As(err, &status) && (status.code == http.StatusNotFound || status.code == http.StatusUnauthorized) {
			return CatalogFilesResult{}, fmt.Errorf("Repository not found, or it is private: %s", repo)
		}
		return CatalogFilesResult{}, err
	}
	var row struct {
		ID       string          `json:"id"`
		SHA      string          `json:"sha"`
		Private  bool            `json:"private"`
		Gated    json.RawMessage `json:"gated"`
		Siblings []struct {
			RFilename string `json:"rfilename"`
			Size      int64  `json:"size"`
			LFS       *struct {
				Size int64 `json:"size"`
			} `json:"lfs"`
		} `json:"siblings"`
	}
	if err := json.Unmarshal(body, &row); err != nil {
		return CatalogFilesResult{}, fmt.Errorf("decode repository file listing: %w", err)
	}
	if row.Private || string(bytes.TrimSpace(row.Gated)) != "false" {
		return CatalogFilesResult{}, errors.New("This repository is private or gated; PAIR downloads only open repositories.")
	}
	if !hexRevision.MatchString(row.SHA) {
		return CatalogFilesResult{}, errors.New("repository file listing has no revision")
	}
	if len(row.Siblings) > maxCopyArtifacts {
		return CatalogFilesResult{}, fmt.Errorf("repository lists more than %d files", maxCopyArtifacts)
	}
	// Keep the provider's spelling when it names the same repository, so the
	// pull name matches the repository's own case.
	if strings.EqualFold(row.ID, repo) && llamaCPPRepoPattern.MatchString(row.ID) {
		repo = row.ID
	}

	type group struct {
		file  CatalogQuantFile
		order int
	}
	byStem := map[string]*group{}
	taken := map[string]bool{}
	for i, sibling := range row.Siblings {
		quant, stem, ok := quantFileStem(sibling.RFilename)
		if !ok {
			continue
		}
		size := sibling.Size
		if size == 0 && sibling.LFS != nil {
			size = sibling.LFS.Size
		}
		g := byStem[stem]
		if g == nil {
			// Two differently named files at one quantization would both answer
			// the same pull name; the first listed is the one offered.
			if taken[quant] {
				continue
			}
			taken[quant] = true
			name := repo + ":" + quant
			if nodepolicy.ValidateModelIdentifier(name) != nil {
				continue
			}
			g = &group{file: CatalogQuantFile{Quantization: quant, Name: name}, order: i}
			byStem[stem] = g
		}
		g.file.SizeBytes += max(size, 0)
		g.file.Shards++
	}
	groups := make([]*group, 0, len(byStem))
	for _, g := range byStem {
		groups = append(groups, g)
	}
	sort.Slice(groups, func(i, j int) bool {
		if groups[i].file.SizeBytes != groups[j].file.SizeBytes {
			return groups[i].file.SizeBytes < groups[j].file.SizeBytes
		}
		return groups[i].order < groups[j].order
	})
	files := make([]CatalogQuantFile, len(groups))
	for i, g := range groups {
		files[i] = g.file
	}
	return CatalogFilesResult{Repo: repo, Revision: row.SHA, Files: files}, nil
}

// quantFileStem reports a model GGUF's quantization and the path that groups
// its shards, or false for a file that is not a model at a known quantization.
func quantFileStem(file string) (string, string, bool) {
	lower := strings.ToLower(file)
	if !strings.HasSuffix(lower, ".gguf") {
		return "", "", false
	}
	base := path.Base(lower)
	for _, auxiliary := range llamaCPPAuxiliaryGGUF {
		if strings.Contains(base, auxiliary) {
			return "", "", false
		}
	}
	stem := file
	if loc := splitGGUFPattern.FindStringIndex(stem); loc != nil {
		stem = stem[:loc[0]]
	} else {
		stem = stem[:len(stem)-len(".gguf")]
	}
	quant, ok := ggufFileQuantization(path.Base(stem))
	return quant, strings.ToLower(stem), ok
}

// ggufFileQuantization finds the quantization a file name ends with: the
// longest tail, starting after a "-", "." or "_", that is a quantization label.
func ggufFileQuantization(name string) (string, bool) {
	if ggufQuantization.MatchString(name) {
		return strings.ToUpper(name), true
	}
	for i := 0; i < len(name); i++ {
		if strings.IndexByte("-._", name[i]) < 0 {
			continue
		}
		if tail := name[i+1:]; ggufQuantization.MatchString(tail) {
			return strings.ToUpper(tail), true
		}
	}
	return "", false
}
