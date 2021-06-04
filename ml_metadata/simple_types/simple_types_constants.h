/* Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef THIRD_PARTY_ML_METADATA_SIMPLE_TYPES_SIMPLE_TYPES_CONSTANTS_H_
#define THIRD_PARTY_ML_METADATA_SIMPLE_TYPES_SIMPLE_TYPES_CONSTANTS_H_

#include <array>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"

namespace ml_metadata {

static const int kNumSimpleTypes = 9;

// The in-memory encoding of SimpleTypes proto.
ABSL_CONST_INIT extern const absl::string_view kSimpleTypes;

// A list of simple type names.
ABSL_CONST_INIT extern const std::array<absl::string_view, kNumSimpleTypes>
    kSimpleTypeNames;

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_SIMPLE_TYPES_SIMPLE_TYPES_CONSTANTS_H_
