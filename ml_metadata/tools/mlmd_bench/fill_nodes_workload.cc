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

// Template function that initializes the properties of the `put_request`.
template <typename T>
void InitializePutRequest(FillNodesWorkItemType& put_request) {
  put_request.emplace<T>();
}

// Gets all types inside db. Returns FAILED_PRECONDITION if there is no types
// inside db for any nodes to insert.
tensorflow::Status GetAndValidateExistingTypes(
    const int specification, MetadataStore* store,
    std::vector<Type>& existing_types) {
  TF_RETURN_IF_ERROR(GetExistingTypes(specification, store, existing_types));
  if (existing_types.size() == 0) {
    return tensorflow::errors::FailedPrecondition(
        "There are no types inside db for inserting nodes!");
  }
  return tensorflow::Status::OK();
}

// Generate random integer within the range of specified `dist`.
int64 GenerateRandomNumberFromUD(const UniformDistribution& dist,
                                 std::minstd_rand0& gen) {
  std::uniform_int_distribution<int64> uniform_dist{dist.minimum(),
                                                    dist.maximum()};
  return uniform_dist(gen);
}

// Gets the transferred bytes for certain `properties` of the current node and
// increases `curr_bytes` accordingly.
void GetTransferredBytesForNodeProperties(
    const google::protobuf::Map<std::string, Value>& properties,
    int64& curr_bytes) {
  for (const auto& property : properties) {
    // Includes the bytes for properties' name size.
    curr_bytes += property.first.size();
    // Includes the bytes for properties' value size.
    curr_bytes += property.second.string_value().size();
  }
}

// Calculates the transferred bytes for each node that will be inserted later.
template <typename T, typename NT>
void GetTransferredBytes(const T& type, const NT& node, int64& curr_bytes) {
  // Increases `curr_bytes` with the size of current node's `name` and
  // `type_id`.
  curr_bytes += node.name().size() + 8;
  // Increases `curr_bytes` with the size of current node's `properties` and
  // `custom_properties`.
  GetTransferredBytesForNodeProperties(node.properties(), curr_bytes);
  GetTransferredBytesForNodeProperties(node.custom_properties(), curr_bytes);
}

// Generates insert node.
// For insert cases, it takes `node_name`, `number_properties`,
// `string_value_bytes` and `type` to set the insert node. The node's type
// will be `type` and its properties will be generated w.r.t. `num_properties`
// and `string_value_bytes`.
template <typename T, typename NT>
void GenerateNode(const std::string& node_name, const int64 num_properties,
                  const int64 string_value_bytes, const T& type, NT& node,
                  int64& curr_bytes) {
  // Insert nodes cases.
  node.set_name(node_name);
  node.set_type_id(type.id());
  std::string property_value(string_value_bytes, '*');
  int64 curr_num_properties = 0;
  // Loops over the types properties while generating the node's properties
  // accordingly.
  auto it = type.properties().begin();
  while (curr_num_properties < num_properties &&
         it != type.properties().end()) {
    (*node.mutable_properties())[it->first].set_string_value(property_value);
    curr_num_properties++;
    it++;
  }
  // If the node's number of properties is greater than the type(the iterator
  // of the type properties has reached the end, but `curr_num_properties`
  // is still less than `num_properties`), uses custom properties instead.
  while (curr_num_properties < num_properties) {
    (*node.mutable_custom_properties())[absl::StrCat("custom_p-",
                                                     curr_num_properties)]
        .set_string_value(property_value);
    curr_num_properties++;
  }
  GetTransferredBytes<T, NT>(type, node, curr_bytes);
}

// Sets additional fields(url, state) for artifacts.
void SetArtifactAdditionalFields(const string& node_name, Artifact& node,
                                 int64& curr_bytes) {
  node.set_uri(absl::StrCat(node_name, "_uri"));
  node.set_state(Artifact::UNKNOWN);
  curr_bytes += node.uri().size() + 1;
}

// Sets additional field(state) for executions.
void SetExecutionAdditionalFields(Execution& node, int64& curr_bytes) {
  node.set_last_known_state(Execution::UNKNOWN);
  curr_bytes += 1;
}

}  // namespace

FillNodes::FillNodes(const FillNodesConfig& fill_nodes_config,
                     const int64 num_operations)
    : fill_nodes_config_(fill_nodes_config), num_operations_(num_operations) {
  switch (fill_nodes_config_.specification()) {
    case FillNodesConfig::ARTIFACT: {
      name_ = "fill_artifact";
      break;
    }
    case FillNodesConfig::EXECUTION: {
      name_ = "fill_execution";
      break;
    }
    case FillNodesConfig::CONTEXT: {
      name_ = "fill_context";
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillNodes!";
  }
  if (fill_nodes_config_.update()) {
    name_ += "(update)";
  }
}

tensorflow::Status FillNodes::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";
  int64 curr_bytes = 0;

  // All the specific types in db to choose from when generating nodes.
  std::vector<Type> existing_types;
  TF_RETURN_IF_ERROR(GetAndValidateExistingTypes(
      fill_nodes_config_.specification(), store, existing_types));
  std::uniform_int_distribution<int64> uniform_dist_type_index{
      0, (int64)(existing_types.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  // TODO(briansong) Adds update support.
  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    FillNodesWorkItemType put_request;
    const std::string node_name =
        absl::StrCat("node_", absl::FormatTime(absl::Now()), "_", i);
    const int64 num_properties =
        GenerateRandomNumberFromUD(fill_nodes_config_.num_properties(), gen);
    const int64 string_value_bytes = GenerateRandomNumberFromUD(
        fill_nodes_config_.string_value_bytes(), gen);
    const int64 type_index = uniform_dist_type_index(gen);
    switch (fill_nodes_config_.specification()) {
      case FillNodesConfig::ARTIFACT: {
        InitializePutRequest<PutArtifactsRequest>(put_request);
        Artifact* curr_node =
            absl::get<PutArtifactsRequest>(put_request).add_artifacts();
        GenerateNode<ArtifactType, Artifact>(
            node_name, num_properties, string_value_bytes,
            absl::get<ArtifactType>(existing_types[type_index]), *curr_node,
            curr_bytes);
        SetArtifactAdditionalFields(node_name, *curr_node, curr_bytes);
        break;
      }
      case FillNodesConfig::EXECUTION: {
        InitializePutRequest<PutExecutionsRequest>(put_request);
        Execution* curr_node =
            absl::get<PutExecutionsRequest>(put_request).add_executions();
        GenerateNode<ExecutionType, Execution>(
            node_name, num_properties, string_value_bytes,
            absl::get<ExecutionType>(existing_types[type_index]), *curr_node,
            curr_bytes);
        SetExecutionAdditionalFields(*curr_node, curr_bytes);
        break;
      }
      case FillNodesConfig::CONTEXT: {
        InitializePutRequest<PutContextsRequest>(put_request);
        Context* curr_node =
            absl::get<PutContextsRequest>(put_request).add_contexts();
        GenerateNode<ContextType, Context>(
            node_name, num_properties, string_value_bytes,
            absl::get<ContextType>(existing_types[type_index]), *curr_node,
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
      TF_RETURN_IF_ERROR(store->PutArtifacts(put_request, &put_response));
      return tensorflow::Status::OK();
    }
    case FillNodesConfig::EXECUTION: {
      PutExecutionsRequest put_request =
          absl::get<PutExecutionsRequest>(work_items_[work_items_index].first);
      PutExecutionsResponse put_response;
      TF_RETURN_IF_ERROR(store->PutExecutions(put_request, &put_response));
      return tensorflow::Status::OK();
    }
    case FillNodesConfig::CONTEXT: {
      PutContextsRequest put_request =
          absl::get<PutContextsRequest>(work_items_[work_items_index].first);
      PutContextsResponse put_response;
      TF_RETURN_IF_ERROR(store->PutContexts(put_request, &put_response));
      return tensorflow::Status::OK();
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
