// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"reflect"
	"strings"
	"testing"
)

func TestValidateProfileKey(t *testing.T) {
	accept := func(name, key string) {
		t.Run(name, func(t *testing.T) {
			if err := ValidateProfileKey(key); err != nil {
				t.Fatalf("ValidateProfileKey(%q): %v", key, err)
			}
		})
	}
	accept("context profile", "lfm-64k")
	accept("user slug", "joes-tiel35b-udq4kxl")
	accept("existing canonical model", "unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL")
	accept("UTF-8 model", "models/模型")
	accept("future API role alias", "coder-api")
	accept("future local role alias", "coder-local")
	accept("suffix with invalid role base", "org/model-api")
	accept("maximum length", strings.Repeat("x", 512))
	for name, key := range map[string]string{
		"empty": "", "bare alias": "weak", "local alias": "medium-local",
		"API alias": "strong-api", "oversized": strings.Repeat("x", 513),
		"control": "model\nkey", "invalid UTF-8": string([]byte{0xff}),
	} {
		t.Run(name, func(t *testing.T) {
			if err := ValidateProfileKey(key); err == nil {
				t.Fatalf("accepted profile key %q", key)
			}
		})
	}
}

func TestProfileIcons(t *testing.T) {
	want := []string{"chat-single", "code", "cube", "document", "image"}
	if !reflect.DeepEqual(ProfileIcons(), want) {
		t.Fatalf("icons = %v, want %v", ProfileIcons(), want)
	}
	for _, icon := range append(want, "") {
		t.Run("accept "+icon, func(t *testing.T) {
			if err := ValidateProfileIcon(icon); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, icon := range []string{"ChatSingle", "chat_single", "unknown", "code ", "code\n"} {
		t.Run("reject "+icon, func(t *testing.T) {
			if err := ValidateProfileIcon(icon); err == nil {
				t.Fatal("accepted unsupported icon")
			}
		})
	}
	icons := ProfileIcons()
	icons[0] = "changed"
	if ProfileIcons()[0] != "chat-single" {
		t.Fatal("consumer changed the validation list")
	}
}

func TestValidateProfileDisplayName(t *testing.T) {
	for _, name := range []string{"My model", "模型", strings.Repeat("x", 128)} {
		if err := ValidateProfileDisplayName(name); err != nil {
			t.Fatalf("name %q: %v", name, err)
		}
	}
	for _, name := range []string{"", strings.Repeat("x", 129), strings.Repeat("模", 43), "bad\nname", "bad\u0085name", string([]byte{0xff})} {
		if err := ValidateProfileDisplayName(name); err == nil {
			t.Fatalf("accepted display name %q", name)
		}
	}
}
