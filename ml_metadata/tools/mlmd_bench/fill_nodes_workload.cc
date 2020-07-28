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
#include "tensorflow/core/platform/logging.h"

namespace ml_metadata {
namespace {

// Template function that initializes the properties of the `put_request`.
template <typename T>
void InitializePutRequest(FillNodesWorkItemType& put_request) {
  put_request.emplace<T>();
}

// Gets the transferred bytes for certain `properties` and increases
// `curr_bytes` accordingly.
tensorflow::Status GetTransferredBytesForNodeProperties(
    const google::protobuf::Map<std::string, Value>& properties,
    int64& curr_bytes) {
  for (auto& pair : properties) {
    // Includes the bytes for properties' name size.
    curr_bytes += pair.first.size();
    // Includes the bytes for properties' value size.
    switch (pair.second.value_case()) {
      case 1: {
        // int64 takes 8 bytes.
        curr_bytes += 8;
        break;
      }
      case 2: {
        // double takes 8 bytes.
        curr_bytes += 8;
        break;
      }
      case 3: {
        // string cases.
        curr_bytes += pair.second.string_value().size();
        break;
      }
      default:
        return tensorflow::errors::InvalidArgument("Invalid ValueType!");
    }
  }
  return tensorflow::Status::OK();
}

// Calculates the transferred bytes for each node that will be inserted later.
template <typename T, typename NT>
tensorflow::Status GetTransferredBytes(const T& type, const NT& node,
                                       int64& curr_bytes) {
  curr_bytes += node.name().size();
  curr_bytes += 8;
  curr_bytes += type.name().size();
  TF_RETURN_IF_ERROR(
      GetTransferredBytesForNodeProperties(node.properties(), curr_bytes));
  TF_RETURN_IF_ERROR(GetTransferredBytesForNodeProperties(
      node.custom_properties(), curr_bytes));
  return tensorflow::Status::OK();
}

// Generates insert node.
// For insert cases, it takes `node_name`, `type`, `number_properties` and
// `string_value_bytes` to set the insert node. The node's type will be `type`
// and its properties will be generated w.r.t. `num_properties` and
// `string_value_bytes`. Returns detailed error if query executions failed.
template <typename T, typename NT>
tensorflow::Status GenerateNode(const std::string& node_name,
                                const int64 num_properties,
                                const int64 string_value_bytes, const T& type,
                                NT& node, int64& curr_bytes) {
  CHECK((std::is_same<T, ArtifactType>::value ||
         std::is_same<T, ExecutionType>::value ||
         std::is_same<T, ContextType>::value))
      << "Unexpected Types";
  CHECK((std::is_same<NT, Artifact>::value ||
         std::is_same<NT, Execution>::value ||
         std::is_same<NT, Context>::value))
      << "Unexpected Node Types";
  // Insert nodes cases.
  node.set_name(node_name);
  node.set_type_id(type.id());
  int64 curr_num_properties = 0;
  // Loops over the types properties and use it to generate the node's
  // properties accordingly.
  auto it = type.properties().begin();
  while (curr_num_properties < num_properties &&
         it != type.properties().end()) {
    std::string value(string_value_bytes, '*');
    (*node.mutable_properties())[it->first].set_string_value(value);
    curr_num_properties++;
    it++;
  }
  // If the node's number of properties is greater than the type, use custom
  // properties instead.
  while (curr_num_properties < num_properties) {
    std::string value(string_value_bytes, '*');
    (*node.mutable_custom_properties())[absl::StrCat("custom_p-",
                                                     curr_num_properties)]
        .set_string_value(value);
    curr_num_properties++;
  }
  return GetTransferredBytes<T, NT>(type, node, curr_bytes);
}

}  // namespace

FillNodes::FillNodes(const FillNodesConfig& fill_nodes_config,
                     int64 num_operations)
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

  // Sets the uniform distributions for generating the `num_properties` and
  // `string_value_bytes`.
  UniformDistribution num_properties_dist = fill_nodes_config_.num_properties();
  std::uniform_int_distribution<int64> uniform_dist_properties{
      num_properties_dist.minimum(), num_properties_dist.maximum()};

  UniformDistribution string_value_bytes_dist =
      fill_nodes_config_.string_value_bytes();
  std::uniform_int_distribution<int64> uniform_dist_string{
      string_value_bytes_dist.minimum(), string_value_bytes_dist.maximum()};

  // Gets all the specific types in db to choose from when generating nodes.
  std::vector<Type> existing_types;
  TF_RETURN_IF_ERROR(GetExistingTypes(fill_nodes_config_.specification(), store,
                                      existing_types));

  if (existing_types.size() == 0) {
    LOG(FATAL) << "There are no types inside db for inserting node!";
  }

  // Sets the uniform distributions for selecting a registered type randomly.
  std::uniform_int_distribution<int64> uniform_dist_type_index{
      0, (int64)(existing_types.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  // TODO(briansong) Adds update support.
  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    FillNodesWorkItemType put_request;
    const std::string node_name =
        absl::StrCat("node_", absl::FormatTime(absl::Now()), "_", i);
    const int64 num_properties = uniform_dist_properties(gen);
    const int64 string_value_bytes = uniform_dist_string(gen);
    const int64 type_index = uniform_dist_type_index(gen);
    switch (fill_nodes_config_.specification()) {
      case FillNodesConfig::ARTIFACT: {
        InitializePutRequest<PutArtifactsRequest>(put_request);
        TF_RETURN_IF_ERROR(GenerateNode<ArtifactType, Artifact>(
            node_name, num_properties, string_value_bytes,
            absl::get<ArtifactType>(existing_types[type_index]),
            *(absl::get<PutArtifactsRequest>(put_request).add_artifacts()),
            curr_bytes));
        break;
      }
      case FillNodesConfig::EXECUTION: {
        InitializePutRequest<PutExecutionsRequest>(put_request);
        TF_RETURN_IF_ERROR(GenerateNode<ExecutionType, Execution>(
            node_name, num_properties, string_value_bytes,
            absl::get<ExecutionType>(existing_types[type_index]),
            *(absl::get<PutExecutionsRequest>(put_request).add_executions()),
            curr_bytes));
        break;
      }
      case FillNodesConfig::CONTEXT: {
        InitializePutRequest<PutContextsRequest>(put_request);
        TF_RETURN_IF_ERROR(GenerateNode<ContextType, Context>(
            node_name, num_properties, string_value_bytes,
            absl::get<ContextType>(existing_types[type_index]),
            *(absl::get<PutContextsRequest>(put_request).add_contexts()),
            curr_bytes));
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
