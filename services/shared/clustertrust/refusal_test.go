// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWriteRemoteManagementDenied(t *testing.T) {
	w := httptest.NewRecorder()
	WriteRemoteManagementDenied(w)
	if w.Code != http.StatusConflict {
		t.Fatalf("status = %d, want 409", w.Code)
	}
	if got := w.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	want := `{"code":"remote_management_denied","error":"Remote management is blocked on this device. Change permission locally on that device."}`
	if got := strings.TrimSpace(w.Body.String()); got != want {
		t.Fatalf("body = %s, want %s", got, want)
	}
}
