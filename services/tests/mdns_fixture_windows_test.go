// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"strings"
	"testing"
	"time"

	"nvpair-shared/mdns"

	"github.com/grandcat/zeroconf"
	"github.com/miekg/dns"
	"golang.org/x/net/ipv4"
)

var fixtureMDNSTarget = &net.UDPAddr{IP: net.IPv4(224, 0, 0, 251), Port: 5353}

// Enumerating interfaces never consults the global multicast route. In
// particular, an up tunnel without multicast capability cannot win selection.
func pickMDNSFixtureIPv4(ifaces []net.Interface, addresses func(net.Interface) ([]net.Addr, error)) (net.Interface, net.IP, error) {
	var failures []error
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagMulticast == 0 || iface.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, err := addresses(iface)
		if err != nil {
			failures = append(failures, fmt.Errorf("interface %s addresses: %w", iface.Name, err))
			continue
		}
		for _, addr := range addrs {
			ip, _, err := net.ParseCIDR(addr.String())
			if err != nil {
				failures = append(failures, fmt.Errorf("interface %s address %q: %w", iface.Name, addr, err))
				continue
			}
			if source := ip.To4(); source != nil && source.IsGlobalUnicast() {
				return iface, source, nil
			}
		}
	}
	return net.Interface{}, nil, fmt.Errorf("no active multicast IPv4 LAN interface: %w", errors.Join(append([]error{errors.New("no eligible address")}, failures...)...))
}

func fixtureInterfaceAddrs(iface net.Interface) ([]net.Addr, error) { return iface.Addrs() }

func mdnsFixtureInterface(t *testing.T) net.Interface {
	t.Helper()
	ifaces, err := net.Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	iface, source, err := pickMDNSFixtureIPv4(ifaces, fixtureInterfaceAddrs)
	if err != nil {
		t.Fatalf("select mDNS fixture: %v; interfaces: %v", err, ifaces)
	}
	t.Logf("mDNS fixture interface: %s index=%d flags=%s source=%s:5353", iface.Name, iface.Index, iface.Flags, source)
	return iface
}

func mdnsResolverOptions(ifaces []net.Interface) []zeroconf.ClientOption {
	options := []zeroconf.ClientOption{zeroconf.SelectIPTraffic(zeroconf.IPv4)}
	if ifaces != nil {
		options = append(options, zeroconf.SelectIfaces(ifaces))
	}
	return options
}

type fixtureMDNSSender struct {
	iface  net.Interface
	source net.IP
	write  func([]byte, *net.Interface, net.IP, *net.UDPAddr) error
}

func newFixtureMDNSSender(ifaces []net.Interface) (fixtureMDNSSender, error) {
	iface, source, err := pickMDNSFixtureIPv4(ifaces, fixtureInterfaceAddrs)
	if err != nil {
		return fixtureMDNSSender{}, err
	}
	return fixtureMDNSSender{iface: iface, source: source, write: mdns.SendFromInterface}, nil
}

func (s fixtureMDNSSender) send(msg *dns.Msg, target *net.UDPAddr) error {
	packet, err := msg.Pack()
	if err != nil {
		return fmt.Errorf("pack fixture mDNS: %w", err)
	}
	if err := s.write(packet, &s.iface, s.source, target); err != nil {
		return fmt.Errorf("fixture mDNS interface=%s index=%d source=%s:5353 target=%s: %w", s.iface.Name, s.iface.Index, s.source, target, err)
	}
	return nil
}

func fixtureMDNSQuery(service, domain string) *dns.Msg {
	return &dns.Msg{Question: []dns.Question{{Name: strings.Trim(service, ".") + "." + strings.TrimSuffix(domain, ".") + ".", Qtype: dns.TypePTR, Qclass: dns.ClassINET}}}
}

// One complete DNS-SD packet lets zeroconf resolve the fixture without a second
// lookup. Its address remains loopback, independently of the transport source.
func fixtureMDNSRecords(f mdnsAdvertiserFixture, goodbye bool) *dns.Msg {
	domain := strings.TrimSuffix(f.Domain, ".") + "."
	service := strings.Trim(f.Service, ".") + "." + domain
	instance := f.Instance + "." + service
	host := f.Instance + "." + domain
	txt := append([]string(nil), f.TXT...)
	if len(txt) == 0 {
		txt = []string{""}
	}
	msg := &dns.Msg{MsgHdr: dns.MsgHdr{Response: true, Authoritative: true}, Compress: true}
	msg.Answer = []dns.RR{
		&dns.PTR{Hdr: dns.RR_Header{Name: service, Rrtype: dns.TypePTR, Class: dns.ClassINET, Ttl: 3200}, Ptr: instance},
		&dns.SRV{Hdr: dns.RR_Header{Name: instance, Rrtype: dns.TypeSRV, Class: dns.ClassINET | 1<<15, Ttl: 3200}, Port: uint16(f.Port), Target: host},
		&dns.TXT{Hdr: dns.RR_Header{Name: instance, Rrtype: dns.TypeTXT, Class: dns.ClassINET | 1<<15, Ttl: 3200}, Txt: txt},
		&dns.A{Hdr: dns.RR_Header{Name: host, Rrtype: dns.TypeA, Class: dns.ClassINET | 1<<15, Ttl: 120}, A: net.IPv4(127, 0, 0, 1)},
	}
	if goodbye {
		for _, rr := range msg.Answer {
			rr.Header().Ttl = 0
		}
	}
	return msg
}

func fixtureMDNSReplyTarget(query *dns.Msg, fixture mdnsAdvertiserFixture, from *net.UDPAddr) *net.UDPAddr {
	if query.Response {
		return nil
	}
	domain := strings.TrimSuffix(fixture.Domain, ".") + "."
	service := strings.Trim(fixture.Service, ".") + "." + domain
	instance, host := fixture.Instance+"."+service, fixture.Instance+"."+domain
	matched, unicast := false, false
	for _, q := range query.Question {
		if q.Qclass&0x7fff != dns.ClassINET {
			continue
		}
		if (strings.EqualFold(q.Name, service) && (q.Qtype == dns.TypePTR || q.Qtype == dns.TypeANY)) ||
			(strings.EqualFold(q.Name, instance) && (q.Qtype == dns.TypeSRV || q.Qtype == dns.TypeTXT || q.Qtype == dns.TypeANY)) ||
			(strings.EqualFold(q.Name, host) && (q.Qtype == dns.TypeA || q.Qtype == dns.TypeANY)) {
			matched = true
			unicast = unicast || q.Qclass&(1<<15) != 0
		}
	}
	if !matched {
		return nil
	}
	if unicast {
		return from
	}
	return fixtureMDNSTarget
}

// The child owns both the receive socket and all advertisement writes. Its
// parent retains the existing readiness/stdin shutdown/forced-exit bounds.
func registerMDNSFixture(fixture mdnsAdvertiserFixture, iface net.Interface) (func() error, error) {
	sender, err := newFixtureMDNSSender([]net.Interface{iface})
	if err != nil {
		return nil, err
	}
	conn, err := net.ListenMulticastUDP("udp4", &iface, fixtureMDNSTarget)
	if err != nil {
		return nil, fmt.Errorf("join fixture mDNS on %s: %w", iface.Name, err)
	}
	// Go disables multicast loopback on ListenMulticastUDP. Keep same-host
	// product scanners and zeroconf receivers eligible to receive our packets.
	if err := ipv4.NewPacketConn(conn).SetMulticastLoopback(true); err != nil {
		return nil, errors.Join(fmt.Errorf("enable fixture multicast loopback: %w", err), conn.Close())
	}
	if err := sender.send(fixtureMDNSRecords(fixture, false), fixtureMDNSTarget); err != nil {
		return nil, errors.Join(err, conn.Close())
	}
	fmt.Fprintf(os.Stderr, "mDNS advertiser interface=%s index=%d source=%s:5353 ready\n", iface.Name, iface.Index, sender.source)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- serveMDNSFixture(ctx, conn, sender, fixture)
	}()
	return func() error {
		cancel()
		closeErr := conn.Close()
		serveErr := <-done
		return errors.Join(closeErr, serveErr, sender.send(fixtureMDNSRecords(fixture, true), fixtureMDNSTarget))
	}, nil
}

func serveMDNSFixture(ctx context.Context, conn *net.UDPConn, sender fixtureMDNSSender, fixture mdnsAdvertiserFixture) error {
	packet := make([]byte, 65536)
	nextAnnouncement := time.Now().Add(time.Second)
	for ctx.Err() == nil {
		if err := conn.SetReadDeadline(time.Now().Add(250 * time.Millisecond)); err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return fmt.Errorf("fixture mDNS read deadline: %w", err)
		}
		n, from, err := conn.ReadFromUDP(packet)
		if err != nil && ctx.Err() == nil {
			if timeout, ok := err.(net.Error); !ok || !timeout.Timeout() {
				return fmt.Errorf("fixture mDNS receive: %w", err)
			}
		}
		if ctx.Err() != nil {
			return nil
		}
		if err == nil {
			var query dns.Msg
			if err := query.Unpack(packet[:n]); err == nil {
				if target := fixtureMDNSReplyTarget(&query, fixture, from); target != nil {
					fmt.Fprintf(os.Stderr, "mDNS fixture query received interface=%s source=%s\n", sender.iface.Name, from)
					if err := sender.send(fixtureMDNSRecords(fixture, false), target); err != nil {
						return err
					}
				}
			}
		}
		if time.Now().After(nextAnnouncement) {
			if err := sender.send(fixtureMDNSRecords(fixture, false), fixtureMDNSTarget); err != nil {
				return err
			}
			nextAnnouncement = time.Now().Add(60 * time.Second)
		}
	}
	return nil
}

// zeroconf still receives/decodes entries. Its Windows IfIndex writes do not
// select egress, so supplement them after Browse has started its receiver.
func mdnsBrowse(t *testing.T, resolver *zeroconf.Resolver, ctx context.Context, service, domain string, entries chan<- *zeroconf.ServiceEntry, ifaces []net.Interface) error {
	t.Helper()
	if ifaces == nil {
		ifaces = []net.Interface{mdnsFixtureInterface(t)}
	}
	sender, err := newFixtureMDNSSender(ifaces)
	if err != nil {
		return err
	}
	queryCtx, cancel := context.WithCancel(ctx)
	if err := resolver.Browse(queryCtx, service, domain, entries); err != nil {
		cancel()
		return err
	}
	query := fixtureMDNSQuery(service, domain)
	if err := sender.send(query, fixtureMDNSTarget); err != nil {
		cancel()
		return err
	}
	t.Logf("mDNS fixture query sent interface=%s index=%d source=%s:5353", sender.iface.Name, sender.iface.Index, sender.source)
	done := make(chan error, 1)
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-queryCtx.Done():
				done <- nil
				return
			case <-ticker.C:
				if err := sender.send(query, fixtureMDNSTarget); err != nil {
					done <- err
					return
				}
			}
		}
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-done:
			if err != nil {
				t.Error(err)
			}
		case <-time.After(2 * time.Second):
			t.Error("fixture mDNS query sender did not stop")
		}
	})
	return nil
}
