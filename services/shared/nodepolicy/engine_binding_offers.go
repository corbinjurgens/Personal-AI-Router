// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"errors"
)

// EngineBindingOffers is advisory preparation support, never installed inventory,
// loaded state or strict capacity. Only version 1 has meaning to this router.
type EngineBindingOffers struct {
	Version int    `json:"version"`
	Models  []Role `json:"models"`
}

func (o EngineBindingOffers) Validate() error {
	if o.Version != 1 || len(o.Models) == 0 {
		return errors.New("unsupported or empty engine binding offers")
	}
	return EngineBindings(o.Models).validate()
}

func (o *EngineBindingOffers) UnmarshalJSON(data []byte) error {
	fields, err := modelObject(data)
	if err != nil {
		return err
	}
	if len(fields) != 2 || fields["version"] == nil || fields["models"] == nil {
		return errors.New("invalid engine binding offers shape")
	}
	var version int
	if err := json.Unmarshal(fields["version"], &version); err != nil {
		return err
	}
	models, err := decodeModelList("engineBindingOffers.models", fields["models"])
	if err != nil {
		return err
	}
	next := EngineBindingOffers{Version: version, Models: models}
	if err := next.Validate(); err != nil {
		return err
	}
	*o = next
	return nil
}

// Clone detaches optional peer metadata without turning it into inventory.
func (o *EngineBindingOffers) Clone() *EngineBindingOffers {
	if o == nil || o.Validate() != nil {
		return nil
	}
	return &EngineBindingOffers{Version: o.Version, Models: append([]Role(nil), o.Models...)}
}
