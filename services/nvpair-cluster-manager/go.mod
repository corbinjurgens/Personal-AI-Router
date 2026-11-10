module nvpair-cluster-manager

go 1.25.0

require (
	eapnoob v0.0.0-00010101000000-000000000000
	golang.org/x/sys v0.47.0
	nvpair-shared v0.0.0-00010101000000-000000000000
)

require (
	github.com/Microsoft/go-winio v0.6.2 // indirect
	github.com/cenkalti/backoff v2.2.1+incompatible // indirect
	github.com/grandcat/zeroconf v1.0.0 // indirect
	github.com/hashicorp/yamux v0.1.2 // indirect
	github.com/miekg/dns v1.1.72 // indirect
	golang.org/x/mod v0.38.0 // indirect
	golang.org/x/net v0.58.0 // indirect
	golang.org/x/sync v0.22.0 // indirect
	golang.org/x/tools v0.48.0 // indirect
)

replace nvpair-shared => ../shared

replace eapnoob => ../eap-noob
