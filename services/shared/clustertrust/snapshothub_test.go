// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"slices"
	"sync"
	"testing"
)

func pending[T any](t *testing.T, ch <-chan T) (T, bool) {
	t.Helper()
	select {
	case value := <-ch:
		return value, true
	default:
		var zero T
		return zero, false
	}
}

func TestSnapshotHub(t *testing.T) {
	t.Run("subscribe before publish receives nothing until the first publish", func(t *testing.T) {
		var hub SnapshotHub[int]
		ch, cancel, ok := hub.Subscribe()
		if !ok {
			t.Fatal("Subscribe refused below the limit")
		}
		defer cancel()
		if value, got := pending(t, ch); got {
			t.Fatalf("received %d before any Publish", value)
		}
		hub.Publish(7)
		if value, got := pending(t, ch); !got || value != 7 {
			t.Fatalf("after Publish got (%d, %v), want (7, true)", value, got)
		}
	})

	t.Run("subscribe after publish receives the latest value at once", func(t *testing.T) {
		var hub SnapshotHub[string]
		hub.Publish("first")
		hub.Publish("second")
		ch, cancel, _ := hub.Subscribe()
		defer cancel()
		if value, got := pending(t, ch); !got || value != "second" {
			t.Fatalf("seed = (%q, %v), want (second, true)", value, got)
		}
	})

	t.Run("two publishes without a read leave only the second pending", func(t *testing.T) {
		var hub SnapshotHub[int]
		ch, cancel, _ := hub.Subscribe()
		defer cancel()
		hub.Publish(1)
		hub.Publish(2)
		if value, got := pending(t, ch); !got || value != 2 {
			t.Fatalf("pending = (%d, %v), want (2, true)", value, got)
		}
		if value, got := pending(t, ch); got {
			t.Fatalf("second pending value %d, want none", value)
		}
	})

	t.Run("reader limit refuses further subscriptions", func(t *testing.T) {
		hub := SnapshotHub[int]{Limit: 2}
		for i := range 2 {
			if _, cancel, ok := hub.Subscribe(); !ok {
				t.Fatalf("Subscribe %d refused below the limit", i)
			} else {
				defer cancel()
			}
		}
		ch, cancel, ok := hub.Subscribe()
		if ok || ch != nil {
			t.Fatalf("Subscribe over the limit = (%v, %v), want (nil, false)", ch, ok)
		}
		cancel() // the refused cancel is a no-op
	})

	t.Run("default limit is 64", func(t *testing.T) {
		var hub SnapshotHub[int]
		for i := range 64 {
			if _, cancel, ok := hub.Subscribe(); !ok {
				t.Fatalf("Subscribe %d refused below the default limit", i)
			} else {
				defer cancel()
			}
		}
		if _, _, ok := hub.Subscribe(); ok {
			t.Fatal("65th Subscribe accepted")
		}
	})

	t.Run("cancel stops delivery and frees the slot", func(t *testing.T) {
		hub := SnapshotHub[int]{Limit: 1}
		ch, cancel, _ := hub.Subscribe()
		cancel()
		hub.Publish(3)
		if value, got := pending(t, ch); got {
			t.Fatalf("cancelled reader received %d", value)
		}
		if _, again, ok := hub.Subscribe(); !ok {
			t.Fatal("slot not freed by cancel")
		} else {
			again()
		}
	})

	t.Run("clone is applied per reader", func(t *testing.T) {
		hub := SnapshotHub[[]int]{Clone: slices.Clone[[]int]}
		first, cancelFirst, _ := hub.Subscribe()
		defer cancelFirst()
		second, cancelSecond, _ := hub.Subscribe()
		defer cancelSecond()
		source := []int{1, 2, 3}
		hub.Publish(source)
		a, _ := pending(t, first)
		b, _ := pending(t, second)
		a[0] = 100
		source[1] = 200
		if !slices.Equal(b, []int{1, 2, 3}) {
			t.Fatalf("second reader saw %v after other mutations, want [1 2 3]", b)
		}
		if latest, _ := hub.Latest(); !slices.Equal(latest, []int{1, 2, 3}) {
			t.Fatalf("Latest = %v after reader and publisher mutations, want [1 2 3]", latest)
		}
	})

	t.Run("latest reports false before and the value after publish", func(t *testing.T) {
		var hub SnapshotHub[int]
		if value, ok := hub.Latest(); ok {
			t.Fatalf("Latest before Publish = (%d, true), want false", value)
		}
		hub.Publish(0)
		if value, ok := hub.Latest(); !ok || value != 0 {
			t.Fatalf("Latest after Publish(0) = (%d, %v), want (0, true)", value, ok)
		}
	})

	t.Run("concurrent publish and subscribe never deliver an older value", func(t *testing.T) {
		var hub SnapshotHub[int]
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 1; i <= 2000; i++ {
				hub.Publish(i)
			}
		}()
		for range 8 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for range 50 {
					ch, cancel, ok := hub.Subscribe()
					if !ok {
						t.Error("Subscribe refused below the limit")
						return
					}
					last := 0
					for range 5 {
						if value, got := pending(t, ch); got {
							if value < last {
								t.Errorf("received %d after %d", value, last)
							}
							last = value
						}
					}
					cancel()
				}
			}()
		}
		wg.Wait()
	})
}
