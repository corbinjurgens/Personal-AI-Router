// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// ModelIdentity describes model files for display and grouping only. It is
// never strict-admission authority, even when free blob names are available.
type ModelIdentity struct {
	Repository              string              `json:"repository,omitempty"`
	Revision                string              `json:"revision,omitempty"`
	Quantization            string              `json:"quantization,omitempty"`
	QuantizationSource      string              `json:"quantizationSource,omitempty"`
	GeneralName             string              `json:"generalName,omitempty"`
	SizeLabel               string              `json:"sizeLabel,omitempty"`
	Architecture            string              `json:"architecture,omitempty"`
	TrainedContextTokens    int64               `json:"trainedContextTokens,omitempty"`
	ConfiguredContextTokens int64               `json:"configuredContextTokens,omitempty"`
	Files                   []ModelIdentityFile `json:"files,omitempty"`
	TotalBytes              int64               `json:"totalBytes,omitempty"`
	ContentID               string              `json:"contentId,omitempty"`
	Strength                string              `json:"strength"`
}

type ModelIdentityFile struct {
	Role      string `json:"role"`
	Name      string `json:"name"`
	SizeBytes int64  `json:"sizeBytes"`
	SHA256    string `json:"sha256,omitempty"`
}

var identityRepository = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$`)
var identityRevision = regexp.MustCompile(`^[0-9a-f]{40}$`)
var identitySHA256 = regexp.MustCompile(`^[0-9a-f]{64}$`)

func (m ModelIdentity) Validate() error {
	text := func(value string) bool {
		return len(value) <= 256 && utf8.ValidString(value) && !strings.ContainsFunc(value, unicode.IsControl)
	}
	for _, value := range []string{m.Repository, m.Revision, m.Quantization, m.QuantizationSource, m.GeneralName, m.SizeLabel, m.Architecture, m.ContentID, m.Strength} {
		if !text(value) {
			return fmt.Errorf("invalid model identity text")
		}
	}
	if m.Repository != "" && !identityRepository.MatchString(m.Repository) {
		return fmt.Errorf("invalid identity repository")
	}
	if m.Revision != "" && !identityRevision.MatchString(m.Revision) {
		return fmt.Errorf("invalid identity revision")
	}
	if m.ContentID != "" && !identitySHA256.MatchString(m.ContentID) {
		return fmt.Errorf("invalid identity content digest")
	}
	switch m.QuantizationSource {
	case "":
	case "hf-tag", "file-name", "gguf-file-type":
		if m.Quantization == "" {
			return fmt.Errorf("quantization source requires a label")
		}
	default:
		return fmt.Errorf("invalid quantization source")
	}
	wantStrength := "declared"
	if m.ContentID != "" {
		wantStrength = "verified"
	}
	if m.Strength != wantStrength {
		return fmt.Errorf("invalid identity strength")
	}
	if m.TotalBytes < 0 || m.TrainedContextTokens < 0 || m.ConfiguredContextTokens < 0 || len(m.Files) > 64 {
		return fmt.Errorf("identity exceeds numeric or file bounds")
	}
	for _, file := range m.Files {
		if !text(file.Role) || !text(file.Name) || !text(file.SHA256) || strings.ContainsAny(file.Name, `/\`) || file.SizeBytes < 0 {
			return fmt.Errorf("invalid identity file")
		}
		switch file.Role {
		case "model", "draft", "companion":
		default:
			return fmt.Errorf("invalid identity file role")
		}
		if file.SHA256 != "" && !identitySHA256.MatchString(file.SHA256) {
			return fmt.Errorf("invalid identity file digest")
		}
	}
	return nil
}
