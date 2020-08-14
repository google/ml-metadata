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
#include "ml_metadata/tools/mlmd_bench/read_nodes_by_properties_workload.h"

#include <random>
#include <vector>

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

template <typename T>
void InitializeReadRequest(ReadNodesByPropertiesWorkItemType& read_request) {
  read_request.emplace<T>();
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
template <typename NT>
tensorflow::Status GetTransferredBytes(const NT& node, int64& curr_bytes) {
  curr_bytes += 8 * 2;
  curr_bytes += node.name().size();
  curr_bytes += node.type().size();
  TF_RETURN_IF_ERROR(
      GetTransferredBytesForNodeProperties(node.properties(), curr_bytes));
  TF_RETURN_IF_ERROR(GetTransferredBytesForNodeProperties(
      node.custom_properties(), curr_bytes));
  return tensorflow::Status::OK();
}

template <typename NT>
tensorflow::Status GetTransferredBytesForAllNodes(
    const std::string type_name, const std::vector<NodeType>& existing_nodes,
    int64& curr_bytes) {
  for (auto& node : existing_nodes) {
    if (absl::get<NT>(node).type() == type_name) {
      TF_RETURN_IF_ERROR(
          GetTransferredBytes<NT>(absl::get<NT>(node), curr_bytes));
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace

ReadNodesByProperties::ReadNodesByProperties(
    const ReadNodesByPropertiesConfig& read_nodes_by_properties_config,
    const int64 num_operations)
    : read_nodes_by_properties_config_(read_nodes_by_properties_config),
      num_operations_(num_operations) {
  switch (read_nodes_by_properties_config_.specification()) {
    case ReadNodesByPropertiesConfig::ARTIFACTS_BY_ID: {
      name_ = "read_artifacts_by_id";
      break;
    }
    case ReadNodesByPropertiesConfig::EXECUTIONS_BY_ID: {
      name_ = "read_executions_by_id";
      break;
    }
    case ReadNodesByPropertiesConfig::CONTEXTS_BY_ID: {
      name_ = "read_contexts_by_id";
      break;
    }
    case ReadNodesByPropertiesConfig::ARTIFACTS_BY_TYPE: {
      name_ = "read_artifacts_by_type";
      break;
    }
    case ReadNodesByPropertiesConfig::EXECUTIONS_BY_TYPE: {
      name_ = "read_executions_by_type";
      break;
    }
    case ReadNodesByPropertiesConfig::CONTEXTS_BY_TYPE: {
      name_ = "read_contexts_by_type";
      break;
    }
    case ReadNodesByPropertiesConfig::ARTIFACT_BY_TYPE_AND_NAME: {
      name_ = "read_artifact_by_type_and_name";
      break;
    }
    case ReadNodesByPropertiesConfig::EXECUTION_BY_TYPE_AND_NAME: {
      name_ = "read_execution_by_type_and_name";
      break;
    }
    case ReadNodesByPropertiesConfig::CONTEXT_BY_TYPE_AND_NAME: {
      name_ = "read_context_by_type_and_name";
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for ReadNodesByProperties!";
  }
}

tensorflow::Status ReadNodesByProperties::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  int64 curr_bytes = 0;

  std::vector<NodeType> existing_nodes;
  TF_RETURN_IF_ERROR(
      GetExistingNodes(read_nodes_by_properties_config_.specification() % 3,
                       store, existing_nodes));

  std::vector<Type> existing_types;
  TF_RETURN_IF_ERROR(
      GetExistingTypes(read_nodes_by_properties_config_.specification() % 3,
                       store, existing_types));

  std::uniform_int_distribution<int64> uniform_dist_node_index{
      0, (int64)(existing_nodes.size() - 1)};

  std::uniform_int_distribution<int64> uniform_dist_type_index{
      0, (int64)(existing_types.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    ReadNodesByPropertiesWorkItemType read_request;
    const int64 node_index = uniform_dist_node_index(gen);
    const int64 type_index = uniform_dist_type_index(gen);
    switch (read_nodes_by_properties_config_.specification()) {
      case ReadNodesByPropertiesConfig::ARTIFACTS_BY_ID: {
        InitializeReadRequest<GetArtifactsByIDRequest>(read_request);
        Artifact picked_node = absl::get<Artifact>(existing_nodes[node_index]);
        absl::get<GetArtifactsByIDRequest>(read_request)
            .add_artifact_ids(picked_node.id());
        TF_RETURN_IF_ERROR(
            GetTransferredBytes<Artifact>(picked_node, curr_bytes));
        break;
      }
      case ReadNodesByPropertiesConfig::EXECUTIONS_BY_ID: {
        InitializeReadRequest<GetExecutionsByIDRequest>(read_request);
        Execution picked_node =
            absl::get<Execution>(existing_nodes[node_index]);
        absl::get<GetExecutionsByIDRequest>(read_request)
            .add_execution_ids(picked_node.id());
        TF_RETURN_IF_ERROR(
            GetTransferredBytes<Execution>(picked_node, curr_bytes));
        break;
      }
      case ReadNodesByPropertiesConfig::CONTEXTS_BY_ID: {
        InitializeReadRequest<GetContextsByIDRequest>(read_request);
        Context picked_node = absl::get<Context>(existing_nodes[node_index]);
        absl::get<GetContextsByIDRequest>(read_request)
            .add_context_ids(picked_node.id());
        TF_RETURN_IF_ERROR(
            GetTransferredBytes<Context>(picked_node, curr_bytes));
        break;
      }
      case ReadNodesByPropertiesConfig::ARTIFACTS_BY_TYPE: {
        InitializeReadRequest<GetArtifactsByTypeRequest>(read_request);
        ArtifactType picked_type =
            absl::get<ArtifactType>(existing_types[type_index]);
        absl::get<GetArtifactsByTypeRequest>(read_request)
            .set_type_name(picked_type.name());
        TF_RETURN_IF_ERROR(GetTransferredBytesForAllNodes<Artifact>(
            picked_type.name(), existing_nodes, curr_bytes));
        break;
      }
      case ReadNodesByPropertiesConfig::EXECUTIONS_BY_TYPE: {
        InitializeReadRequest<GetExecutionsByTypeRequest>(read_request);
        ExecutionType picked_type =
            absl::get<ExecutionType>(existing_types[type_index]);
        absl::get<GetExecutionsByTypeRequest>(read_request)
            .set_type_name(picked_type.name());
        TF_RETURN_IF_ERROR(GetTransferredBytesForAllNodes<Execution>(
            picked_type.name(), existing_nodes, curr_bytes));
        break;
      }
      case ReadNodesByPropertiesConfig::CONTEXTS_BY_TYPE: {
        InitializeReadRequest<GetContextsByTypeRequest>(read_request);
        ContextType picked_type =
            absl::get<ContextType>(existing_types[type_index]);
        absl::get<GetContextsByTypeRequest>(read_request)
            .set_type_name(picked_type.name());
        TF_RETURN_IF_ERROR(GetTransferredBytesForAllNodes<Context>(
            picked_type.name(), existing_nodes, curr_bytes));
        break;
      }
      case ReadNodesByPropertiesConfig::ARTIFACT_BY_TYPE_AND_NAME: {
        InitializeReadRequest<GetArtifactByTypeAndNameRequest>(read_request);
        Artifact picked_node = absl::get<Artifact>(existing_nodes[node_index]);
        absl::get<GetArtifactByTypeAndNameRequest>(read_request)
            .set_type_name(picked_node.type());
        absl::get<GetArtifactByTypeAndNameRequest>(read_request)
            .set_artifact_name(picked_node.name());
        TF_RETURN_IF_ERROR(
            GetTransferredBytes<Artifact>(picked_node, curr_bytes));
        break;
      }
      case ReadNodesByPropertiesConfig::EXECUTION_BY_TYPE_AND_NAME: {
        InitializeReadRequest<GetExecutionByTypeAndNameRequest>(read_request);
        Execution picked_node =
            absl::get<Execution>(existing_nodes[node_index]);
        absl::get<GetExecutionByTypeAndNameRequest>(read_request)
            .set_type_name(picked_node.type());
        absl::get<GetExecutionByTypeAndNameRequest>(read_request)
            .set_execution_name(picked_node.name());
        TF_RETURN_IF_ERROR(
            GetTransferredBytes<Execution>(picked_node, curr_bytes));
        break;
      }
      case ReadNodesByPropertiesConfig::CONTEXT_BY_TYPE_AND_NAME: {
        InitializeReadRequest<GetContextByTypeAndNameRequest>(read_request);
        Context picked_node = absl::get<Context>(existing_nodes[node_index]);
        absl::get<GetContextByTypeAndNameRequest>(read_request)
            .set_type_name(picked_node.type());
        absl::get<GetContextByTypeAndNameRequest>(read_request)
            .set_context_name(picked_node.name());
        TF_RETURN_IF_ERROR(
            GetTransferredBytes<Context>(picked_node, curr_bytes));
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for ReadNodesByProperties!";
    }
    work_items_.emplace_back(read_request, curr_bytes);
  }

  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status ReadNodesByProperties::RunOpImpl(
    const int64 work_items_index, MetadataStore* store) {
  switch (read_nodes_by_properties_config_.specification()) {
    case ReadNodesByPropertiesConfig::ARTIFACTS_BY_ID: {
      GetArtifactsByIDRequest request = absl::get<GetArtifactsByIDRequest>(
          work_items_[work_items_index].first);
      GetArtifactsByIDResponse response;
      return store->GetArtifactsByID(request, &response);
      break;
    }
    case ReadNodesByPropertiesConfig::EXECUTIONS_BY_ID: {
      GetExecutionsByIDRequest request = absl::get<GetExecutionsByIDRequest>(
          work_items_[work_items_index].first);
      GetExecutionsByIDResponse response;
      return store->GetExecutionsByID(request, &response);
      break;
    }
    case ReadNodesByPropertiesConfig::CONTEXTS_BY_ID: {
      GetContextsByIDRequest request = absl::get<GetContextsByIDRequest>(
          work_items_[work_items_index].first);
      GetContextsByIDResponse response;
      return store->GetContextsByID(request, &response);
      break;
    }
    case ReadNodesByPropertiesConfig::ARTIFACTS_BY_TYPE: {
      GetArtifactsByTypeRequest request = absl::get<GetArtifactsByTypeRequest>(
          work_items_[work_items_index].first);
      GetArtifactsByTypeResponse response;
      return store->GetArtifactsByType(request, &response);
      break;
    }
    case ReadNodesByPropertiesConfig::EXECUTIONS_BY_TYPE: {
      GetExecutionsByTypeRequest request =
          absl::get<GetExecutionsByTypeRequest>(
              work_items_[work_items_index].first);
      GetExecutionsByTypeResponse response;
      return store->GetExecutionsByType(request, &response);
      break;
    }
    case ReadNodesByPropertiesConfig::CONTEXTS_BY_TYPE: {
      GetContextsByTypeRequest request = absl::get<GetContextsByTypeRequest>(
          work_items_[work_items_index].first);
      GetContextsByTypeResponse response;
      return store->GetContextsByType(request, &response);
      break;
    }
    case ReadNodesByPropertiesConfig::ARTIFACT_BY_TYPE_AND_NAME: {
      GetArtifactByTypeAndNameRequest request =
          absl::get<GetArtifactByTypeAndNameRequest>(
              work_items_[work_items_index].first);
      GetArtifactByTypeAndNameResponse response;
      return store->GetArtifactByTypeAndName(request, &response);
      break;
    }
    case ReadNodesByPropertiesConfig::EXECUTION_BY_TYPE_AND_NAME: {
      GetExecutionByTypeAndNameRequest request =
          absl::get<GetExecutionByTypeAndNameRequest>(
              work_items_[work_items_index].first);
      GetExecutionByTypeAndNameResponse response;
      return store->GetExecutionByTypeAndName(request, &response);
      break;
    }
    case ReadNodesByPropertiesConfig::CONTEXT_BY_TYPE_AND_NAME: {
      GetContextByTypeAndNameRequest request =
          absl::get<GetContextByTypeAndNameRequest>(
              work_items_[work_items_index].first);
      GetContextByTypeAndNameResponse response;
      return store->GetContextByTypeAndName(request, &response);
      break;
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
}

tensorflow::Status ReadNodesByProperties::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string ReadNodesByProperties::GetName() { return name_; }

}  // namespace ml_metadata
