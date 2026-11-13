module nvpair-engine-manager

go 1.26.0

toolchain go1.27.4

require nvpair-shared v0.0.0-00010101000000-000000000000

require (
	github.com/gpustack/gguf-parser-go v0.27.0
	github.com/hashicorp/yamux v0.1.2 // indirect
	github.com/henvic/httpretty v0.1.4 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/smallnest/ringbuffer v0.0.0-20241116012123-461381446e3d // indirect
	golang.org/x/crypto v0.59.0 // indirect
	golang.org/x/exp v0.0.0-20241108190413-2d47ceb2692f // indirect
	golang.org/x/mod v0.41.0 // indirect
	golang.org/x/sync v0.24.0 // indirect
	golang.org/x/tools v0.51.0 // indirect
	gonum.org/v1/gonum v0.15.1 // indirect
)

replace nvpair-shared => ../shared

require (
	github.com/Microsoft/go-winio v0.6.2 // indirect
	github.com/ebitengine/purego v0.10.2
	golang.org/x/sys v0.50.0
)
