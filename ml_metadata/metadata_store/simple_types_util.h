/* Copyright 2021 Google LLC

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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_SIMPLE_TYPES_UTIL_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_SIMPLE_TYPES_UTIL_H_

#include "absl/status/status.h"
#include "ml_metadata/simple_types/proto/simple_types.pb.h"

namespace ml_metadata {

// A util to create a SimpleTypes proto from a constant string at runtime.
absl::Status LoadSimpleTypes(SimpleTypes& simple_types);

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_SIMPLE_TYPES_UTIL_H_
