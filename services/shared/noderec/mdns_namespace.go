// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"fmt"
	"strings"
)

// MDNSNamespace selects the DNS-SD names used by both scanner discovery owners.
// It changes naming only, never service keys, identity or peer authorization.
type MDNSNamespace struct {
	Service string
	Domain  string
}

// Normalize accepts case variants and one optional trailing root dot. Empty or
// malformed explicit names are errors, including when multicast is disabled.
func (n MDNSNamespace) Normalize() (MDNSNamespace, error) {
	for _, c := range n.Service + n.Domain {
		if c > 127 {
			return MDNSNamespace{}, fmt.Errorf("mDNS names must use ASCII DNS labels")
		}
	}
	n.Service = strings.ToLower(strings.TrimSuffix(n.Service, "."))
	n.Domain = strings.ToLower(strings.TrimSuffix(n.Domain, "."))
	service, transport, ok := strings.Cut(n.Service, ".")
	if !ok || transport != "_tcp" || !strings.HasPrefix(service, "_") || len(service) > 63 || !dnsLabel(strings.TrimPrefix(service, "_")) {
		return MDNSNamespace{}, fmt.Errorf("invalid mDNS service %q: expected _name._tcp", n.Service)
	}
	// Leave room for a maximal instance label and the DNS root in the full
	// instance.service.domain name (255 wire bytes).
	if len(n.Service)+len(n.Domain)+67 > 255 {
		return MDNSNamespace{}, fmt.Errorf("mDNS namespace is too long")
	}
	for _, label := range strings.Split(n.Domain, ".") {
		if !dnsLabel(label) {
			return MDNSNamespace{}, fmt.Errorf("invalid mDNS domain %q", n.Domain)
		}
	}
	return n, nil
}

func dnsLabel(s string) bool {
	if len(s) == 0 || len(s) > 63 || s[0] == '-' || s[len(s)-1] == '-' {
		return false
	}
	for _, c := range s {
		if c != '-' && (c < 'a' || c > 'z') && (c < '0' || c > '9') {
			return false
		}
	}
	return true
}
