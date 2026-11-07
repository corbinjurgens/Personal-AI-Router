// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestCORSCatalogPreservesEnginePermissions(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		for _, scenario := range []struct {
			name   string
			status int
			origin string
			body   string
			want   int
		}{
			{name: "approved", status: http.StatusOK, origin: "http://app.test", body: corsModels(tc), want: http.StatusOK},
			{name: "denied", status: http.StatusForbidden, want: http.StatusForbidden},
			{name: "different origin", status: http.StatusOK, origin: "http://other.test", want: http.StatusForbidden},
			{name: "missing permission", status: http.StatusOK, want: http.StatusForbidden},
			{name: "unavailable", status: http.StatusServiceUnavailable, want: http.StatusBadGateway},
			{name: "invalid inventory keeps approved permissions", status: http.StatusOK, origin: "http://app.test", body: "{malformed", want: http.StatusBadGateway},
		} {
			t.Run(scenario.name, func(t *testing.T) {
				var hits atomic.Int32
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					hits.Add(1)
					w.Header().Set("Access-Control-Allow-Origin", scenario.origin)
					w.Header().Set("Access-Control-Allow-Credentials", "true")
					w.WriteHeader(scenario.status)
					if _, err := io.WriteString(w, scenario.body); err != nil {
						t.Error(err)
					}
				}))
				defer upstream.Close()
				n := nodeFor(t, "peer", upstream.URL)
				n.Profiles = []nodepolicy.AdvertisedProfile{{Key: "public-profile", Engine: tc.profile.Name, Model: "native-model"}}
				discovery := NewDiscovery()
				discovery.AddManual(n)
				p := testProxy(tc.profile, discovery, tc.profile.FacadePort).soleFacade()
				rec := httptest.NewRecorder()
				p.handlePlain(rec, corsRequest(http.MethodGet, tc.modelListPath, "http://app.test"))
				if rec.Code != scenario.want {
					t.Fatalf("catalog status=%d want %d", rec.Code, scenario.want)
				}
				if scenario.want == http.StatusOK {
					assertProfileCatalog(t, rec, "public-profile")
					if rec.Header().Get("Access-Control-Allow-Origin") != "http://app.test" || rec.Header().Get("Access-Control-Allow-Credentials") != "true" {
						t.Fatalf("approved browser permissions lost: %v", rec.Header())
					}
				} else if scenario.name == "invalid inventory keeps approved permissions" {
					if rec.Header().Get("Access-Control-Allow-Origin") != "http://app.test" {
						t.Fatal("invalid inventory lost approved origin")
					}
				} else if rec.Header().Get("Access-Control-Allow-Origin") != "" {
					t.Fatal("refused catalog granted browser permission")
				}
				if hits.Load() != 1 {
					t.Fatalf("catalog permission queries=%d, want 1", hits.Load())
				}
			})
		}
	})
}

func TestModelListDoesNotForwardCredentials(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		var hits atomic.Int32
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			hits.Add(1)
			if r.Header.Get("Authorization") != "" || r.Header.Get("Cookie") != "" {
				t.Error("caller credentials reached catalog permission check")
			}
			http.Redirect(w, r, "/credential-sink", http.StatusTemporaryRedirect)
		}))
		defer upstream.Close()
		p := proxyForCORSTargets(t, tc, upstream, upstream)
		for _, origin := range []string{"", "http://app.test"} {
			t.Run("origin="+origin, func(t *testing.T) {
				request := corsRequest(http.MethodGet, tc.modelListPath, origin)
				request.Header.Set("Authorization", "Bearer private")
				request.Header.Set("Cookie", "session=private")
				rec := httptest.NewRecorder()
				p.handlePlain(rec, request)
				wantHits := int32(0)
				if origin == "" {
					assertProfileCatalog(t, rec)
				} else {
					wantHits = 1
					if rec.Code != http.StatusForbidden {
						t.Fatalf("redirect status=%d", rec.Code)
					}
				}
				if hits.Load() != wantHits {
					t.Fatalf("catalog forwarded caller request %d times", hits.Load())
				}
			})
		}
	})
}

func TestCORSCatalogRequiresAllRespondingEngines(t *testing.T) {
	tc := llamacppCase(t)
	for _, scenario := range []struct {
		name, origin string
		status, want int
	}{
		{"agreement", "http://app.test", http.StatusOK, http.StatusOK},
		{"explicit denial", "", http.StatusForbidden, http.StatusForbidden},
		{"missing origin permission", "", http.StatusOK, http.StatusForbidden},
		{"unavailable engine has no opinion", "", http.StatusServiceUnavailable, http.StatusOK},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			a := corsEngine(t, tc, "http://app.test", http.StatusOK)
			defer a.Close()
			b := corsEngine(t, tc, scenario.origin, scenario.status)
			defer b.Close()
			f := proxyForCORSTargets(t, tc, a, b)
			w := httptest.NewRecorder()
			f.handlePlain(w, corsRequest(http.MethodGet, tc.modelListPath, "http://app.test"))
			if w.Code != scenario.want {
				t.Fatalf("catalog status=%d, want %d", w.Code, scenario.want)
			}
			if scenario.want == http.StatusOK {
				assertProfileCatalog(t, w)
				if w.Header().Get("Access-Control-Allow-Origin") != "http://app.test" {
					t.Fatal("approved origin lost")
				}
			} else if w.Header().Get("Access-Control-Allow-Origin") != "" {
				t.Fatal("denied origin permitted")
			}
		})
	}
}
