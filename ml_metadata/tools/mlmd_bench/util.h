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

// Defines a Type that can be ArtifactType / ExecutionType / ContextType.
using Type = ::absl::variant<ArtifactType, ExecutionType, ContextType>;

// Defines a Node that can be Artifact / Execution / Context.
using Node = ::absl::variant<Artifact, Execution, Context>;

// Gets all the existing types inside db and store them into
// `existing_types` given `fill_types_config`. Returns detailed error if query
// executions failed.
tensorflow::Status GetExistingTypes(const FillTypesConfig& fill_types_config,
                                    MetadataStore& store,
                                    std::vector<Type>& existing_types);

// Gets all the existing types inside db and store them into
// `existing_types` given `fill_nodes_config`. Returns detailed error if query
// executions failed.
tensorflow::Status GetExistingTypes(const FillNodesConfig& fill_nodes_config,
                                    MetadataStore& store,
                                    std::vector<Type>& existing_types);

// Gets all the existing types inside db and store them into `existing_types`
// given `read_types_config`. Returns detailed error if query executions failed.
tensorflow::Status GetExistingTypes(const ReadTypesConfig& read_types_config,
                                    MetadataStore& store,
                                    std::vector<Type>& existing_types);

// Gets all the existing nodes inside db and store them into `existing_nodes`
// given `fill_nodes_config`.
// Returns detailed error if query executions failed.
tensorflow::Status GetExistingNodes(const FillNodesConfig& fill_nodes_config,
                                    MetadataStore& store,
                                    std::vector<Node>& existing_nodes);

// Gets all the existing non-context and context nodes inside db and store
// them into `existing_non_context_nodes` and `existing_context_nodes` given
// `fill_context_edges_config`.
// Returns detailed error if query executions failed.
tensorflow::Status GetExistingNodes(
    const FillContextEdgesConfig& fill_context_edges_config,
    MetadataStore& store, std::vector<Node>& existing_non_context_nodes,
    std::vector<Node>& existing_context_nodes);

// Gets existing artifacts and executions inside db and store them into
// `existing_artifact_nodes` and `existing_execution_nodes` given
// `fill_events_config`. Returns detailed error if query executions
// failed.
tensorflow::Status GetExistingNodes(
    const FillEventsConfig& fill_events_config, MetadataStore& store,
    std::vector<Node>& existing_artifact_nodes,
    std::vector<Node>& existing_execution_nodes);

// Gets all the existing nodes inside db and store them into `existing_nodes`
// given `read_nodes_by_properties_config`. Returns detailed error if query
// executions failed.
tensorflow::Status GetExistingNodes(
    const ReadNodesByPropertiesConfig& read_nodes_by_properties_config,
    MetadataStore& store, std::vector<Node>& existing_nodes);

// Gets all the existing nodes inside db and store them into `existing_nodes`
// given `read_nodes_via_context_edges_config`. Returns detailed error if query
// executions failed.
tensorflow::Status GetExistingNodes(
    const ReadNodesViaContextEdgesConfig& read_nodes_via_context_edges_config,
    MetadataStore& store, std::vector<Node>& existing_nodes);

// Gets all the existing nodes inside db and store them into `existing_nodes`
// given `read_events_config`. Returns detailed error if query executions
// failed.
tensorflow::Status GetExistingNodes(const ReadEventsConfig& read_events_config,
                                    MetadataStore& store,
                                    std::vector<Node>& existing_nodes);

// Inserts some types into db for setting up in testing. Returns detailed error
// if query executions failed.
tensorflow::Status InsertTypesInDb(int64 num_artifact_types,
                                   int64 num_execution_types,
                                   int64 num_context_types,
                                   MetadataStore& store);

// Inserts some nodes into db for setting up in testing. Returns detailed error
// if query executions failed.
tensorflow::Status InsertNodesInDb(int64 num_artifact_nodes,
                                   int64 num_execution_nodes,
                                   int64 num_context_nodes,
                                   MetadataStore& store);

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_UTIL_H
