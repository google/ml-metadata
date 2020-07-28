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
using Type = absl::variant<ArtifactType, ExecutionType, ContextType>;

// Defines a NodeType can be Artifact / Execution / Context.
using NodeType = absl::variant<Artifact, Execution, Context>;

// Gets all the existing types (the specific types will be indicated by
// `specification`) inside db and store them into `existing_types`.
// Returns detailed error if query executions failed.
tensorflow::Status GetExistingTypes(int specification, MetadataStore* store,
                                    std::vector<Type>& existing_types);

// Gets all the existing nodes (the specific node types will be indicated by
// `specification`) inside db and store them into `existing_nodes`.
// Returns detailed error if query executions failed.
tensorflow::Status GetExistingNodes(int specification, MetadataStore* store,
                                    std::vector<NodeType>& existing_nodes);

// Inserts some types into db to set it up in different start status in
// testing.
tensorflow::Status InsertTypesInDb(int64 num_artifact_types,
                                   int64 num_execution_types,
                                   int64 num_context_types,
                                   MetadataStore* store);

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_UTIL_H
