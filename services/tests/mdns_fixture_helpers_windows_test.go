// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"errors"
	"net"
	"slices"
	"strings"
	"testing"

	"github.com/miekg/dns"
)

func TestMDNSFixtureSelectsEligibleLANWithoutRouteProbe(t *testing.T) {
	lan := net.Interface{Name: "Wi-Fi", Index: 19, Flags: net.FlagUp | net.FlagMulticast}
	ifaces := []net.Interface{
		{Name: "Tailscale", Index: 8, Flags: net.FlagUp},
		{Name: "down", Index: 9, Flags: net.FlagMulticast},
		{Name: "loopback", Index: 1, Flags: net.FlagUp | net.FlagLoopback | net.FlagMulticast},
		{Name: "IPv6 only", Index: 10, Flags: net.FlagUp | net.FlagMulticast},
		lan,
	}
	var inspected []int
	iface, source, err := pickMDNSFixtureIPv4(ifaces, func(iface net.Interface) ([]net.Addr, error) {
		inspected = append(inspected, iface.Index)
		if iface.Index == 10 {
			return []net.Addr{&net.IPNet{IP: net.ParseIP("fd00::1"), Mask: net.CIDRMask(64, 128)}}, nil
		}
		return []net.Addr{&net.IPNet{IP: net.IPv4(192, 168, 11, 45), Mask: net.CIDRMask(24, 32)}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if iface.Index != lan.Index || !source.Equal(net.IPv4(192, 168, 11, 45)) {
		t.Fatalf("selection = %v source %s, want LAN %v source 192.168.11.45", iface, source, lan)
	}
	if !slices.Equal(inspected, []int{10, 19}) {
		t.Errorf("address enumerations = %v, want only eligible interfaces [10 19]", inspected)
	}
}

func TestMDNSFixtureRejectsUnusableAddresses(t *testing.T) {
	test := func(name, address string, bits int) {
		t.Run(name, func(t *testing.T) {
			_, _, err := pickMDNSFixtureIPv4([]net.Interface{{Name: "fixture", Flags: net.FlagUp | net.FlagMulticast}}, func(net.Interface) ([]net.Addr, error) {
				return []net.Addr{&net.IPNet{IP: net.ParseIP(address), Mask: net.CIDRMask(bits, 32)}}, nil
			})
			if err == nil {
				t.Fatalf("accepted unusable source %s", address)
			}
		})
	}
	test("loopback address", "127.0.0.1", 8)
	test("unspecified address", "0.0.0.0", 0)
	test("multicast address", "224.0.0.251", 24)
	test("link local address", "169.254.1.2", 16)
}

func TestMDNSFixtureAddressFailureIncludesInterface(t *testing.T) {
	wantErr := errors.New("enumeration failed")
	_, _, err := pickMDNSFixtureIPv4([]net.Interface{{Name: "Wi-Fi", Flags: net.FlagUp | net.FlagMulticast}}, func(net.Interface) ([]net.Addr, error) {
		return nil, wantErr
	})
	if !errors.Is(err, wantErr) || !strings.Contains(err.Error(), "Wi-Fi") {
		t.Fatalf("selection error = %v, want interface and wrapped enumeration failure", err)
	}
}

func TestMDNSFixtureRecordsRetainIdentityAndLoopback(t *testing.T) {
	fixture := mdnsAdvertiserFixture{Instance: "fixture-node", Service: "_test-disc._tcp", Port: 55555, TXT: []string{"env=test", "uuid=fixture"}}
	packet, err := fixtureMDNSRecords(fixture, false).Pack()
	if err != nil {
		t.Fatal(err)
	}
	var msg dns.Msg
	if err := msg.Unpack(packet); err != nil {
		t.Fatal(err)
	}
	if !msg.Response || !msg.Authoritative || len(msg.Answer) != 4 {
		t.Fatalf("response header = %v, records = %d", msg.MsgHdr, len(msg.Answer))
	}
	ptr, ptrOK := msg.Answer[0].(*dns.PTR)
	srv, srvOK := msg.Answer[1].(*dns.SRV)
	txt, txtOK := msg.Answer[2].(*dns.TXT)
	addr, addrOK := msg.Answer[3].(*dns.A)
	if !ptrOK || !srvOK || !txtOK || !addrOK {
		t.Fatalf("record types = %T %T %T %T", msg.Answer[0], msg.Answer[1], msg.Answer[2], msg.Answer[3])
	}
	if ptr.Hdr.Name != "_test-disc._tcp.local." || ptr.Ptr != "fixture-node._test-disc._tcp.local." || ptr.Hdr.Ttl != 3200 {
		t.Errorf("PTR = %+v", ptr)
	}
	if srv.Hdr.Name != ptr.Ptr || srv.Port != 55555 || srv.Target != "fixture-node.local." || srv.Hdr.Ttl != 3200 {
		t.Errorf("SRV = %+v", srv)
	}
	if txt.Hdr.Name != ptr.Ptr || !slices.Equal(txt.Txt, fixture.TXT) || txt.Hdr.Ttl != 3200 {
		t.Errorf("TXT = %+v", txt)
	}
	if addr.Hdr.Name != srv.Target || !addr.A.Equal(net.IPv4(127, 0, 0, 1)) || addr.Hdr.Ttl != 120 {
		t.Errorf("A = %+v", addr)
	}
}

func TestMDNSFixtureGoodbyeWithdrawsEveryRecord(t *testing.T) {
	fixture := mdnsAdvertiserFixture{Instance: "removable-node", Service: "_test-rmv._tcp", Port: 55556}
	announcement, goodbye := fixtureMDNSRecords(fixture, false), fixtureMDNSRecords(fixture, true)
	for i, rr := range goodbye.Answer {
		if rr.Header().Ttl != 0 || rr.Header().Name != announcement.Answer[i].Header().Name || rr.Header().Rrtype != announcement.Answer[i].Header().Rrtype {
			t.Errorf("goodbye record = %v, want TTL 0 for %v", rr, announcement.Answer[i])
		}
	}
}

func TestMDNSFixtureReplyScope(t *testing.T) {
	fixture := mdnsAdvertiserFixture{Instance: "fixture", Service: "_test._tcp", Port: 55555}
	from := &net.UDPAddr{IP: net.IPv4(192, 168, 11, 45), Port: 5353}
	test := func(name, qname string, qtype, qclass uint16, response bool, want *net.UDPAddr) {
		t.Run(name, func(t *testing.T) {
			query := &dns.Msg{MsgHdr: dns.MsgHdr{Response: response}, Question: []dns.Question{{Name: qname, Qtype: qtype, Qclass: qclass}}}
			if got := fixtureMDNSReplyTarget(query, fixture, from); got != want {
				t.Errorf("reply target = %v, want %v", got, want)
			}
		})
	}
	test("browse multicast", "_test._tcp.local.", dns.TypePTR, dns.ClassINET, false, fixtureMDNSTarget)
	test("browse unicast bit", "_test._tcp.local.", dns.TypePTR, dns.ClassINET|1<<15, false, from)
	test("instance lookup", "fixture._test._tcp.local.", dns.TypeSRV, dns.ClassINET, false, fixtureMDNSTarget)
	test("host address", "fixture.local.", dns.TypeA, dns.ClassINET, false, fixtureMDNSTarget)
	test("unrelated service", "_other._tcp.local.", dns.TypePTR, dns.ClassINET, false, nil)
	test("unrelated type", "_test._tcp.local.", dns.TypeAAAA, dns.ClassINET, false, nil)
	test("response is not a query", "_test._tcp.local.", dns.TypePTR, dns.ClassINET, true, nil)
	test("unrelated class", "_test._tcp.local.", dns.TypePTR, dns.ClassCHAOS, false, nil)
}

func TestMDNSFixtureTransportUsesSelectedSourceForAllPackets(t *testing.T) {
	fixture := mdnsAdvertiserFixture{Instance: "fixture", Service: "_test._tcp", Port: 55555}
	source := net.IPv4(192, 168, 11, 45)
	iface := net.Interface{Name: "Wi-Fi", Index: 19}
	test := func(name string, msg *dns.Msg, wantResponse bool) {
		t.Run(name, func(t *testing.T) {
			writes := 0
			sender := fixtureMDNSSender{iface: iface, source: source, write: func(packet []byte, gotIface *net.Interface, gotSource net.IP, target *net.UDPAddr) error {
				writes++
				if gotIface.Index != iface.Index || !gotSource.Equal(source) || !target.IP.Equal(net.IPv4(224, 0, 0, 251)) || target.Port != 5353 {
					t.Errorf("write interface=%v source=%s target=%s", gotIface, gotSource, target)
				}
				var decoded dns.Msg
				if err := decoded.Unpack(packet); err != nil {
					return err
				}
				if decoded.Response != wantResponse {
					t.Errorf("response = %v, want %v", decoded.Response, wantResponse)
				}
				if !wantResponse && (len(decoded.Question) != 1 || decoded.Question[0].Name != "_test._tcp.local." || decoded.Question[0].Qtype != dns.TypePTR || decoded.RecursionDesired) {
					t.Errorf("browse query = %+v", decoded)
				}
				return nil
			}}
			if err := sender.send(msg, fixtureMDNSTarget); err != nil {
				t.Fatal(err)
			}
			if writes != 1 {
				t.Errorf("writes = %d, want 1", writes)
			}
		})
	}
	test("query", fixtureMDNSQuery(fixture.Service), false)
	test("response and announcement", fixtureMDNSRecords(fixture, false), true)
	test("goodbye", fixtureMDNSRecords(fixture, true), true)
}

func TestMDNSFixtureTransportReportsWriteFailure(t *testing.T) {
	wantErr := errors.New("write failed")
	sender := fixtureMDNSSender{iface: net.Interface{Name: "Wi-Fi", Index: 19}, source: net.IPv4(192, 168, 11, 45), write: func([]byte, *net.Interface, net.IP, *net.UDPAddr) error { return wantErr }}
	err := sender.send(fixtureMDNSQuery("_test._tcp"), fixtureMDNSTarget)
	if !errors.Is(err, wantErr) || !strings.Contains(err.Error(), "source=192.168.11.45:5353") {
		t.Fatalf("send error = %v, want wrapped write failure and source", err)
	}
}

func TestMDNSFixtureTransportRejectsMalformedDNSBeforeWrite(t *testing.T) {
	writes := 0
	sender := fixtureMDNSSender{write: func([]byte, *net.Interface, net.IP, *net.UDPAddr) error {
		writes++
		return nil
	}}
	err := sender.send(fixtureMDNSQuery(strings.Repeat("x", 64)+"._tcp"), fixtureMDNSTarget)
	if err == nil || !strings.Contains(err.Error(), "pack fixture mDNS") {
		t.Fatalf("send error = %v, want DNS packing failure", err)
	}
	if writes != 0 {
		t.Errorf("writes = %d, want no write for malformed DNS", writes)
	}
}
