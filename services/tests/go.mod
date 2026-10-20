module tests

go 1.25.0

require (
	github.com/grandcat/zeroconf v1.0.0
	github.com/shirou/gopsutil/v4 v4.26.7
	nvpair-shared v0.0.0-00010101000000-000000000000
	nvpair-tui v0.0.0-00010101000000-000000000000
)

replace nvpair-shared => ../shared

replace nvpair-tui => ../nvpair-tui

require (
	github.com/Microsoft/go-winio v0.6.2 // indirect
	github.com/cenkalti/backoff v2.2.1+incompatible // indirect
	github.com/ebitengine/purego v0.10.2 // indirect
	github.com/go-ole/go-ole v1.2.6 // indirect
	github.com/hashicorp/yamux v0.1.2 // indirect
	github.com/lufia/plan9stats v0.0.0-20211012122336-39d0f177ccd0 // indirect
	github.com/miekg/dns v1.1.55 // indirect
	github.com/power-devops/perfstat v0.0.0-20240221224432-82ca36839d55 // indirect
	github.com/tklauser/go-sysconf v0.3.16 // indirect
	github.com/tklauser/numcpus v0.11.0 // indirect
	github.com/yusufpapurcu/wmi v1.2.4 // indirect
	golang.org/x/mod v0.12.0 // indirect
	golang.org/x/net v0.58.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/tools v0.11.0 // indirect
)
