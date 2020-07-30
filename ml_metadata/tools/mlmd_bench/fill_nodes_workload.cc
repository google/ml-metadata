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
    const int specification, MetadataStore* store,
    std::vector<Type>& existing_types) {
  TF_RETURN_IF_ERROR(GetExistingTypes(specification, store, existing_types));
  if (existing_types.size() == 0) {
    return tensorflow::errors::FailedPrecondition(
        "There are no types inside db for inserting nodes!");
  }
  return tensorflow::Status::OK();
}

// Initializes the parameters of current node batch inside current put request.
void InitializeCurrentNodeBatchParameters(
    const int64 i, const FillNodesConfig& fill_nodes_config,
    std::uniform_int_distribution<int64>& uniform_dist_type_index,
    std::minstd_rand0& gen, std::string& node_batch_name, int64& num_properties,
    int64& string_value_bytes, int64& num_nodes, int64& type_index) {
  node_batch_name =
      absl::StrCat("node_batch", absl::FormatTime(absl::Now()), "_", i);
  num_properties =
      GenerateRandomNumberFromUD(fill_nodes_config.num_properties(), gen);
  string_value_bytes =
      GenerateRandomNumberFromUD(fill_nodes_config.string_value_bytes(), gen);
  num_nodes = GenerateRandomNumberFromUD(fill_nodes_config.num_nodes(), gen);
  type_index = uniform_dist_type_index(gen);
}

// Gets the transferred bytes for certain `properties` of the current node.
int64 GetTransferredBytesForNodeProperties(
    const google::protobuf::Map<std::string, Value>& properties) {
  int64 bytes = 0;
  for (const auto& property : properties) {
    // Includes the bytes for properties' name size.
    bytes += property.first.size();
    // Includes the bytes for properties' value size.
    bytes += property.second.string_value().size();
  }
  return bytes;
}

// Calculates the transferred bytes for each node that will be inserted later.
template <typename NT>
int64 GetTransferredBytes(const NT& node) {
  int64 bytes = 0;
  // Increases `bytes` with the size of current node's `name` and
  // `type_id`.
  bytes += node.name().size() + 8;
  // Increases `bytes` with the size of current node's `properties` and
  // `custom_properties`.
  bytes += GetTransferredBytesForNodeProperties(node.properties()) +
           GetTransferredBytesForNodeProperties(node.custom_properties());
  return bytes;
}

// Generates insert node.
// For insert cases, the node's type will be `type` and its properties will be
// generated w.r.t. `node_name`, `num_properties` and `string_value_bytes`.
// Leaves it as return `void` instead of return `int64` for now because
// FillNodes update mode will return `tensorflow::Status` in the future.
template <typename T, typename NT>
void GenerateNodes(const std::string& node_batch_name,
                   const int64 num_properties, const int64 string_value_bytes,
                   const T& type,
                   google::protobuf::RepeatedPtrField<NT>& node_batch,
                   int64& curr_bytes) {
  CHECK((std::is_same<T, ArtifactType>::value ||
         std::is_same<T, ExecutionType>::value ||
         std::is_same<T, ContextType>::value))
      << "Unexpected Types";
  CHECK((std::is_same<NT, Artifact>::value ||
         std::is_same<NT, Execution>::value ||
         std::is_same<NT, Context>::value))
      << "Unexpected Node Types";
  // Insert nodes cases.
  // Loops over all the node inside `node_batch` and sets up one by one.
  for (int64 i = 0; i < node_batch.size(); ++i) {
    node_batch[i].set_name(absl::StrCat(node_batch_name, "_node_", i));
    node_batch[i].set_type_id(type.id());
    // Uses "********" as the fake property value for current node.
    std::string property_value(string_value_bytes, '*');
    int64 curr_num_properties = 0;
    // Loops over the types properties while generating the node's properties
    // accordingly.
    auto it = type.properties().begin();
    while (curr_num_properties < num_properties &&
           it != type.properties().end()) {
      (*node_batch[i].mutable_properties())[it->first].set_string_value(
          property_value);
      curr_num_properties++;
      it++;
    }
    // If the node's number of properties is greater than the type(the iterator
    // of the type properties has reached the end, but `curr_num_properties`
    // is still less than `num_properties`), uses custom properties instead.
    while (curr_num_properties < num_properties) {
      (*node_batch[i]
            .mutable_custom_properties())[absl::StrCat("custom_p-",
                                                       curr_num_properties)]
          .set_string_value(property_value);
      curr_num_properties++;
    }
    curr_bytes += GetTransferredBytes<NT>(node_batch[i]);
  }
}

// Sets additional fields(uri, state) for artifacts and returns transferred
// bytes for these additional fields.
int64 SetArtifactsAdditionalFields(
    const std::string& node_batch_name,
    google::protobuf::RepeatedPtrField<Artifact>& node_batch) {
  int64 bytes = 0;
  for (int64 i = 0; i < node_batch.size(); ++i) {
    node_batch[i].set_uri(absl::StrCat(node_batch_name, "_node_", i, "_uri"));
    node_batch[i].set_state(Artifact::UNKNOWN);
    bytes += node_batch[i].uri().size() + 1;
  }
  return bytes;
}

// Sets additional field(state) for executions and returns transferred
// bytes for these additional fields.
int64 SetExecutionsAdditionalFields(
    google::protobuf::RepeatedPtrField<Execution>& node_batch) {
  int64 bytes = 0;
  for (int64 i = 0; i < node_batch.size(); ++i) {
    node_batch[i].set_last_known_state(Execution::UNKNOWN);
    bytes += 1;
  }
  return bytes;
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

  // Gets all the specific types in db to choose from when generating nodes.
  std::vector<Type> existing_types;
  TF_RETURN_IF_ERROR(GetAndValidateExistingTypes(
      fill_nodes_config_.specification(), store, existing_types));
  std::uniform_int_distribution<int64> uniform_dist_type_index{
      0, (int64)(existing_types.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  std::string node_batch_name;
  int64 curr_bytes, num_properties, string_value_bytes, num_nodes, type_index;

  // TODO(briansong) Adds update support.
  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    FillNodesWorkItemType put_request;
    InitializeCurrentNodeBatchParameters(
        i, fill_nodes_config_, uniform_dist_type_index, gen, node_batch_name,
        num_properties, string_value_bytes, num_nodes, type_index);
    switch (fill_nodes_config_.specification()) {
      case FillNodesConfig::ARTIFACT: {
        InitializePutRequest<PutArtifactsRequest>(num_nodes, put_request);
        google::protobuf::RepeatedPtrField<Artifact>& node_batch =
            *absl::get<PutArtifactsRequest>(put_request).mutable_artifacts();
        GenerateNodes<ArtifactType, Artifact>(
            node_batch_name, num_properties, string_value_bytes,
            absl::get<ArtifactType>(existing_types[type_index]), node_batch,
            curr_bytes);
        curr_bytes += SetArtifactsAdditionalFields(node_batch_name, node_batch);
        break;
      }
      case FillNodesConfig::EXECUTION: {
        InitializePutRequest<PutExecutionsRequest>(num_nodes, put_request);
        google::protobuf::RepeatedPtrField<Execution>& node_batch =
            *absl::get<PutExecutionsRequest>(put_request).mutable_executions();
        GenerateNodes<ExecutionType, Execution>(
            node_batch_name, num_properties, string_value_bytes,
            absl::get<ExecutionType>(existing_types[type_index]), node_batch,
            curr_bytes);
        curr_bytes += SetExecutionsAdditionalFields(node_batch);
        break;
      }
      case FillNodesConfig::CONTEXT: {
        InitializePutRequest<PutContextsRequest>(num_nodes, put_request);
        google::protobuf::RepeatedPtrField<Context>& node_batch =
            *absl::get<PutContextsRequest>(put_request).mutable_contexts();
        GenerateNodes<ContextType, Context>(
            node_batch_name, num_properties, string_value_bytes,
            absl::get<ContextType>(existing_types[type_index]), node_batch,
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
