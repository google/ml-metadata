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
#include "ml_metadata/tools/mlmd_bench/fill_nodes_workload.h"

#include <random>
#include <type_traits>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {
namespace {

// Parameters for nodes to be inserted per put request.
struct NodesParam {
  std::string nodes_name;
  int64 num_properties;
  int64 string_value_bytes;
};

// Template function that initializes `put_request` to contain `num_nodes` nodes
// inside.
template <typename T>
void InitializePutRequest(const int64 num_nodes,
                          FillNodesWorkItemType& put_request) {
  put_request.emplace<T>();
  for (int64 i = 0; i < num_nodes; ++i) {
    if (std::is_same<T, PutArtifactsRequest>::value) {
      absl::get<PutArtifactsRequest>(put_request).add_artifacts();
    } else if (std::is_same<T, PutExecutionsRequest>::value) {
      absl::get<PutExecutionsRequest>(put_request).add_executions();
    } else if (std::is_same<T, PutContextsRequest>::value) {
      absl::get<PutContextsRequest>(put_request).add_contexts();
    } else {
      LOG(FATAL) << "Unexpected Node Types for initializing current node batch";
    }
  }
}

// Gets all types inside db. Returns FAILED_PRECONDITION if there is no types
// inside db for any nodes to insert.
tensorflow::Status GetAndValidateExistingTypes(
    const FillNodesConfig& fill_nodes_config, MetadataStore& store,
    std::vector<Type>& existing_types) {
  TF_RETURN_IF_ERROR(
      GetExistingTypes(fill_nodes_config, store, existing_types));
  if (existing_types.empty()) {
    return tensorflow::errors::FailedPrecondition(
        "There are no types inside db for inserting nodes!");
  }
  return tensorflow::Status::OK();
}

// Generates random integer within the range of specified `dist`.
int64 GenerateRandomNumberFromUD(const UniformDistribution& dist,
                                 std::minstd_rand0& gen) {
  std::uniform_int_distribution<int64> uniform_dist{dist.minimum(),
                                                    dist.maximum()};
  return uniform_dist(gen);
}

// Initializes the parameters of current node batch inside current put request.
void InitializeCurrentNodeBatchParameters(
    const int64 i, const FillNodesConfig& fill_nodes_config,
    std::uniform_int_distribution<int64>& uniform_dist_type_index,
    std::minstd_rand0& gen, int64& num_nodes, int64& type_index,
    NodesParam& nodes_param) {
  nodes_param.nodes_name =
      absl::StrCat("nodes", absl::FormatTime(absl::Now()), "_", i);
  nodes_param.num_properties =
      GenerateRandomNumberFromUD(fill_nodes_config.num_properties(), gen);
  nodes_param.string_value_bytes =
      GenerateRandomNumberFromUD(fill_nodes_config.string_value_bytes(), gen);
  num_nodes = GenerateRandomNumberFromUD(fill_nodes_config.num_nodes(), gen);
  type_index = uniform_dist_type_index(gen);
}

// Calculates the transferred bytes for each node that will be inserted later.
template <typename NT>
int64 GetTransferredBytes(const NT& node) {
  int64 bytes = 0;
  // Increases `bytes` with the size of current node's `name` and
  // `type_id`.
  bytes += node.name().size() + 8;
  // Increases `bytes` with the size of current node's `properties`.
  for (const auto& property : node.properties()) {
    // Includes the bytes for properties' name size.
    bytes += property.first.size();
    // Includes the bytes for properties' value size.
    bytes += property.second.string_value().size();
  }
  // Similarly, increases bytes for `custom_properties`.
  for (const auto& property : node.custom_properties()) {
    bytes += property.first.size();
    bytes += property.second.string_value().size();
  }
  return bytes;
}

// Generates insert node.
// For insert cases, the node's type will be `type` and its properties will be
// generated w.r.t. `node_name`, `num_properties` and `string_value_bytes`.
// Leaves it as return `void` instead of return `int64` for now because
// FillNodes update mode will return `tensorflow::Status` in the future.
template <typename T, typename N>
void GenerateNodes(const NodesParam& nodes_param, const T& type,
                   google::protobuf::RepeatedPtrField<N>& nodes,
                   int64& curr_bytes) {
  CHECK((std::is_same<T, ArtifactType>::value ||
         std::is_same<T, ExecutionType>::value ||
         std::is_same<T, ContextType>::value))
      << "Unexpected Types";
  CHECK((std::is_same<N, Artifact>::value ||
         std::is_same<N, Execution>::value || std::is_same<N, Context>::value))
      << "Unexpected Node Types";
  // Insert nodes cases.
  // Loops over all the node inside `nodes` and sets up one by one.
  for (int64 i = 0; i < nodes.size(); ++i) {
    nodes[i].set_name(absl::StrCat(nodes_param.nodes_name, "_node_", i));
    nodes[i].set_type_id(type.id());
    // Uses "********" as the fake property value for current node.
    std::string property_value(nodes_param.string_value_bytes, '*');
    // Loops over the types properties while generating the node's properties
    // accordingly.
    // TODO(briansong) Adds more property types support.
    int64 populated_num_properties = 0;
    for (const auto& p : type.properties()) {
      (*nodes[i].mutable_properties())[p.first].set_string_value(
          property_value);
      if (++populated_num_properties > nodes_param.num_properties) {
        break;
      }
    }
    // If the node's number of properties is greater than the type, uses custom
    // properties instead.
    while (populated_num_properties++ < nodes_param.num_properties) {
      (*nodes[i].mutable_custom_properties())[absl::StrCat(
                                                  "custom_p-",
                                                  populated_num_properties)]
          .set_string_value(property_value);
    }
    curr_bytes += GetTransferredBytes<N>(nodes[i]);
  }
}

// Sets additional fields(uri, state) for artifacts and returns transferred
// bytes for these additional fields.
int64 SetArtifactsAdditionalFields(
    const std::string& nodes_name,
    google::protobuf::RepeatedPtrField<Artifact>& nodes) {
  int64 bytes = 0;
  for (int64 i = 0; i < nodes.size(); ++i) {
    nodes[i].set_uri(absl::StrCat(nodes_name, "_node_", i, "_uri"));
    nodes[i].set_state(Artifact::UNKNOWN);
    bytes += nodes[i].uri().size() + 1;
  }
  return bytes;
}

// Sets additional field(state) for executions and returns transferred
// bytes for these additional fields.
int64 SetExecutionsAdditionalFields(
    google::protobuf::RepeatedPtrField<Execution>& nodes) {
  int64 bytes = 0;
  for (int64 i = 0; i < nodes.size(); ++i) {
    nodes[i].set_last_known_state(Execution::UNKNOWN);
    bytes += 1;
  }
  return bytes;
}

}  // namespace

FillNodes::FillNodes(const FillNodesConfig& fill_nodes_config,
                     const int64 num_operations)
    : fill_nodes_config_(fill_nodes_config),
      num_operations_(num_operations),
      name_(([fill_nodes_config]() {
        std::string name =
            absl::StrCat("FILL_", fill_nodes_config.Specification_Name(
                                      fill_nodes_config.specification()));
        if (fill_nodes_config.update()) {
          name += "(UPDATE)";
        }
        return name;
      }())) {}

tensorflow::Status FillNodes::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";
  int64 curr_bytes = 0;

  // Gets all the specific types in db to choose from when generating nodes.
  // If there's no types in the store, return error.
  std::vector<Type> existing_types;
  TF_RETURN_IF_ERROR(
      GetAndValidateExistingTypes(fill_nodes_config_, *store, existing_types));
  std::uniform_int_distribution<int64> uniform_dist_type_index{
      0, (int64)(existing_types.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  // TODO(briansong) Adds update support.
  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    FillNodesWorkItemType put_request;
    int64 num_nodes, type_index;
    NodesParam nodes_param;
    InitializeCurrentNodeBatchParameters(i, fill_nodes_config_,
                                         uniform_dist_type_index, gen,
                                         num_nodes, type_index, nodes_param);
    switch (fill_nodes_config_.specification()) {
      case FillNodesConfig::ARTIFACT: {
        InitializePutRequest<PutArtifactsRequest>(num_nodes, put_request);
        auto nodes =
            absl::get<PutArtifactsRequest>(put_request).mutable_artifacts();
        GenerateNodes<ArtifactType, Artifact>(
            nodes_param, absl::get<ArtifactType>(existing_types[type_index]),
            *nodes, curr_bytes);
        curr_bytes +=
            SetArtifactsAdditionalFields(nodes_param.nodes_name, *nodes);
        break;
      }
      case FillNodesConfig::EXECUTION: {
        InitializePutRequest<PutExecutionsRequest>(num_nodes, put_request);
        auto nodes =
            absl::get<PutExecutionsRequest>(put_request).mutable_executions();
        GenerateNodes<ExecutionType, Execution>(
            nodes_param, absl::get<ExecutionType>(existing_types[type_index]),
            *nodes, curr_bytes);
        curr_bytes += SetExecutionsAdditionalFields(*nodes);
        break;
      }
      case FillNodesConfig::CONTEXT: {
        InitializePutRequest<PutContextsRequest>(num_nodes, put_request);
        GenerateNodes<ContextType, Context>(
            nodes_param, absl::get<ContextType>(existing_types[type_index]),
            *absl::get<PutContextsRequest>(put_request).mutable_contexts(),
            curr_bytes);
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for FillNodes!";
    }
    work_items_.emplace_back(put_request, curr_bytes);
  }
  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status FillNodes::RunOpImpl(const int64 work_items_index,
                                        MetadataStore* store) {
  switch (fill_nodes_config_.specification()) {
    case FillNodesConfig::ARTIFACT: {
      PutArtifactsRequest put_request =
          absl::get<PutArtifactsRequest>(work_items_[work_items_index].first);
      PutArtifactsResponse put_response;
      return store->PutArtifacts(put_request, &put_response);
    }
    case FillNodesConfig::EXECUTION: {
      PutExecutionsRequest put_request =
          absl::get<PutExecutionsRequest>(work_items_[work_items_index].first);
      PutExecutionsResponse put_response;
      return store->PutExecutions(put_request, &put_response);
    }
    case FillNodesConfig::CONTEXT: {
      PutContextsRequest put_request =
          absl::get<PutContextsRequest>(work_items_[work_items_index].first);
      PutContextsResponse put_response;
      return store->PutContexts(put_request, &put_response);
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
}

tensorflow::Status FillNodes::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string FillNodes::GetName() { return name_; }

}  // namespace ml_metadata
