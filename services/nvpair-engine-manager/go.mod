module nvpair-engine-manager

go 1.25.0

require nvpair-shared v0.0.0-00010101000000-000000000000

require (
	github.com/danieljoos/wincred v1.2.3 // indirect
	github.com/godbus/dbus/v5 v5.1.0 // indirect
	github.com/hashicorp/yamux v0.1.2 // indirect
)

replace nvpair-shared => ../shared

require (
	github.com/Microsoft/go-winio v0.6.2 // indirect
	github.com/ebitengine/purego v0.10.2
	golang.org/x/sys v0.47.0
)
