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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_UTIL_H
#define ML_METADATA_TOOLS_MLMD_BENCH_UTIL_H

#include <vector>

#include "absl/types/variant.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// Defines a Type can be ArtifactType / ExecutionType / ContextType.
using Type = ::absl::variant<ArtifactType, ExecutionType, ContextType>;

// Gets all the existing types (the specific types that indicated by
// `fill_types_config`) inside db and store them into `existing_types`.
// Returns detailed error if query executions failed.
tensorflow::Status GetExistingTypes(const FillTypesConfig& fill_types_config,
                                    MetadataStore* store,
                                    std::vector<Type>& existing_types);

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_UTIL_H
