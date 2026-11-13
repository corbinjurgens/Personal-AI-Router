module nvpair-ui-broker

go 1.26.0

toolchain go1.27.4

require (
	golang.org/x/sys v0.47.0
	nvpair-shared v0.0.0-00010101000000-000000000000
)

replace nvpair-shared => ../shared

require (
	github.com/Microsoft/go-winio v0.6.2 // indirect
	github.com/hashicorp/yamux v0.1.2 // indirect
	golang.org/x/text v0.41.0 // indirect
)
