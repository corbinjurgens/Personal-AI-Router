// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"encoding/json"
	"net/http"
)

// The remote-management denial: a pinned peer's management operation that this
// node's grant (Mesh.RemoteManagement) does not allow. Every service that
// refuses such an operation, on a peer route or in a JSON-RPC reply, uses these
// values so clients see one code and one text.
const (
	// RemoteManagementDeniedRPCCode is the JSON-RPC error code of the denial.
	// It is distinct from an unpinned 403 and from a local pause veto.
	RemoteManagementDeniedRPCCode = -32012
	// RemoteManagementDeniedCode is the denial's code in a CodedRefusal body.
	RemoteManagementDeniedCode = "remote_management_denied"
	// RemoteManagementDeniedMessage is the bounded, user-facing denial text. It
	// carries no device, policy, certificate or filesystem detail.
	RemoteManagementDeniedMessage = "Remote management is blocked on this device. Change permission locally on that device."
)

// CodedRefusal is the JSON body of a peer route's typed refusal: a
// machine-readable code and bounded human text.
type CodedRefusal struct {
	Code  string `json:"code"`
	Error string `json:"error"`
}

// WriteCodedRefusal answers a peer request with status and a CodedRefusal body.
func WriteCodedRefusal(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(CodedRefusal{Code: code, Error: message})
}

// WriteRemoteManagementDenied answers a peer request with the denial: HTTP 409
// and {"code":"remote_management_denied","error":<RemoteManagementDeniedMessage>}.
func WriteRemoteManagementDenied(w http.ResponseWriter) {
	WriteCodedRefusal(w, http.StatusConflict, RemoteManagementDeniedCode, RemoteManagementDeniedMessage)
}
