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
    curr_bytes += pair.second.string_value().size();
  }
  return tensorflow::Status::OK();
}

tensorflow::Status GetTransferredBytes(const Artifact& node,
                                       int64& curr_bytes) {
  curr_bytes += 8 * 4;
  curr_bytes += node.name().size();
  curr_bytes += node.type().size();
  curr_bytes += node.uri().size();
  curr_bytes += 1;
  TF_RETURN_IF_ERROR(
      GetTransferredBytesForNodeProperties(node.properties(), curr_bytes));
  TF_RETURN_IF_ERROR(GetTransferredBytesForNodeProperties(
      node.custom_properties(), curr_bytes));
  return tensorflow::Status::OK();
}

tensorflow::Status GetTransferredBytes(const Execution& node,
                                       int64& curr_bytes) {
  curr_bytes += 8 * 4;
  curr_bytes += node.name().size();
  curr_bytes += node.type().size();
  curr_bytes += 1;
  TF_RETURN_IF_ERROR(
      GetTransferredBytesForNodeProperties(node.properties(), curr_bytes));
  TF_RETURN_IF_ERROR(GetTransferredBytesForNodeProperties(
      node.custom_properties(), curr_bytes));
  return tensorflow::Status::OK();
}

tensorflow::Status GetTransferredBytes(const Context& node, int64& curr_bytes) {
  curr_bytes += 8 * 4;
  curr_bytes += node.name().size();
  curr_bytes += node.type().size();
  TF_RETURN_IF_ERROR(
      GetTransferredBytesForNodeProperties(node.properties(), curr_bytes));
  TF_RETURN_IF_ERROR(GetTransferredBytesForNodeProperties(
      node.custom_properties(), curr_bytes));
  return tensorflow::Status::OK();
}

template <typename NT>
tensorflow::Status GetTransferredBytesForAllNodesUnderAType(
    const std::string type_name, const std::vector<Node>& existing_nodes,
    int64& curr_bytes) {
  for (auto& node : existing_nodes) {
    if (absl::get<NT>(node).type() == type_name) {
      TF_RETURN_IF_ERROR(GetTransferredBytes(absl::get<NT>(node), curr_bytes));
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status SetUpImplForReadNodesByIds(
    const ReadNodesByPropertiesConfig& read_nodes_by_properties_config,
    const std::vector<Node>& existing_nodes,
    std::uniform_int_distribution<int64>& node_index_dist,
    std::minstd_rand0& gen, ReadNodesByPropertiesWorkItemType& request,
    int64& curr_bytes) {
  UniformDistribution num_ids_proto_dist =
      read_nodes_by_properties_config.maybe_num_queries();
  std::uniform_int_distribution<int64> num_ids_dist{
      num_ids_proto_dist.minimum(), num_ids_proto_dist.maximum()};
  const int64 num_ids = num_ids_dist(gen);
  for (int64 i = 0; i < num_ids; ++i) {
    const int64 node_index = node_index_dist(gen);
    switch (read_nodes_by_properties_config.specification()) {
      case ReadNodesByPropertiesConfig::ARTIFACTS_BY_IDs: {
        // TODO: Remove InitializeReadRequest.
        InitializeReadRequest<GetArtifactsByIDRequest>(request);
        absl::get<GetArtifactsByIDRequest>(request).add_artifact_ids(
            absl::get<Artifact>(existing_nodes[node_index]).id());
        // TODO: Remove <Artifact>.
        TF_RETURN_IF_ERROR(GetTransferredBytes(
            absl::get<Artifact>(existing_nodes[node_index]), curr_bytes));
        break;
      }
      case ReadNodesByPropertiesConfig::EXECUTIONS_BY_IDs: {
        InitializeReadRequest<GetExecutionsByIDRequest>(request);
        absl::get<GetExecutionsByIDRequest>(request).add_execution_ids(
            absl::get<Execution>(existing_nodes[node_index]).id());
        TF_RETURN_IF_ERROR(GetTransferredBytes(
            absl::get<Execution>(existing_nodes[node_index]), curr_bytes));
        break;
      }
      case ReadNodesByPropertiesConfig::CONTEXTS_BY_IDs: {
        InitializeReadRequest<GetContextsByIDRequest>(request);
        absl::get<GetContextsByIDRequest>(request).add_context_ids(
            absl::get<Context>(existing_nodes[node_index]).id());
        TF_RETURN_IF_ERROR(GetTransferredBytes(
            absl::get<Context>(existing_nodes[node_index]), curr_bytes));
        break;
      }
      default:
        return tensorflow::errors::Unimplemented(
            "Wrong ReadNodesByProperties specification for read nodes by ids "
            "in db.");
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status SetUpImplForReadArtifactsByURIs(
    const ReadNodesByPropertiesConfig& read_nodes_by_properties_config,
    const std::vector<Node>& existing_nodes,
    std::uniform_int_distribution<int64>& node_index_dist,
    std::minstd_rand0& gen, ReadNodesByPropertiesWorkItemType& request,
    int64& curr_bytes) {
  if (read_nodes_by_properties_config.specification() !=
      ReadNodesByPropertiesConfig::ARTIFACTS_BY_URIs) {
    return tensorflow::errors::Unimplemented(
        "Wrong ReadNodesByProperties specification for read artifacts by uris "
        "in db.");
  }
  UniformDistribution num_uris_proto_dist =
      read_nodes_by_properties_config.maybe_num_queries();
  std::uniform_int_distribution<int64> num_uris_dist{
      num_uris_proto_dist.minimum(), num_uris_proto_dist.maximum()};
  const int64 num_uris = num_uris_dist(gen);
  for (int64 i = 0; i < num_uris; ++i) {
    const int64 node_index = node_index_dist(gen);
    InitializeReadRequest<GetArtifactsByURIRequest>(request);
    absl::get<GetArtifactsByURIRequest>(request).add_uris(
        absl::get<Artifact>(existing_nodes[node_index]).uri());
    TF_RETURN_IF_ERROR(GetTransferredBytes(
        absl::get<Artifact>(existing_nodes[node_index]), curr_bytes));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status SetUpImplForReadNodesByType(
    const ReadNodesByPropertiesConfig& read_nodes_by_properties_config,
    const std::vector<Node>& existing_nodes,
    std::uniform_int_distribution<int64>& node_index_dist,
    std::minstd_rand0& gen, ReadNodesByPropertiesWorkItemType& request,
    int64& curr_bytes) {
  const int64 node_index = node_index_dist(gen);
  switch (read_nodes_by_properties_config.specification()) {
    case ReadNodesByPropertiesConfig::ARTIFACTS_BY_TYPE: {
      InitializeReadRequest<GetArtifactsByTypeRequest>(request);
      absl::get<GetArtifactsByTypeRequest>(request).set_type_name(
          absl::get<Artifact>(existing_nodes[node_index]).type());
      return GetTransferredBytesForAllNodesUnderAType<Artifact>(
          absl::get<Artifact>(existing_nodes[node_index]).type(),
          existing_nodes, curr_bytes);
    }
    case ReadNodesByPropertiesConfig::EXECUTIONS_BY_TYPE: {
      InitializeReadRequest<GetExecutionsByTypeRequest>(request);
      absl::get<GetExecutionsByTypeRequest>(request).set_type_name(
          absl::get<Execution>(existing_nodes[node_index]).type());
      return GetTransferredBytesForAllNodesUnderAType<Execution>(
          absl::get<Execution>(existing_nodes[node_index]).type(),
          existing_nodes, curr_bytes);
    }
    case ReadNodesByPropertiesConfig::CONTEXTS_BY_TYPE: {
      InitializeReadRequest<GetContextsByTypeRequest>(request);
      absl::get<GetContextsByTypeRequest>(request).set_type_name(
          absl::get<Context>(existing_nodes[node_index]).type());
      return GetTransferredBytesForAllNodesUnderAType<Context>(
          absl::get<Context>(existing_nodes[node_index]).type(), existing_nodes,
          curr_bytes);
    }
    default:
      return tensorflow::errors::Unimplemented(
          "Wrong ReadNodesByProperties specification for read nodes by type in "
          "db.");
  }
}

tensorflow::Status SetUpImplForReadNodeByTypeAndName(
    const ReadNodesByPropertiesConfig& read_nodes_by_properties_config,
    const std::vector<Node>& existing_nodes,
    std::uniform_int_distribution<int64>& node_index_dist,
    std::minstd_rand0& gen, ReadNodesByPropertiesWorkItemType& request,
    int64& curr_bytes) {
  const int64 node_index = node_index_dist(gen);
  switch (read_nodes_by_properties_config.specification()) {
    case ReadNodesByPropertiesConfig::ARTIFACT_BY_TYPE_AND_NAME: {
      InitializeReadRequest<GetArtifactByTypeAndNameRequest>(request);
      Artifact picked_node = absl::get<Artifact>(existing_nodes[node_index]);
      absl::get<GetArtifactByTypeAndNameRequest>(request).set_type_name(
          picked_node.type());
      absl::get<GetArtifactByTypeAndNameRequest>(request).set_artifact_name(
          picked_node.name());
      return GetTransferredBytes(picked_node, curr_bytes);
    }
    case ReadNodesByPropertiesConfig::EXECUTION_BY_TYPE_AND_NAME: {
      InitializeReadRequest<GetExecutionByTypeAndNameRequest>(request);
      Execution picked_node = absl::get<Execution>(existing_nodes[node_index]);
      absl::get<GetExecutionByTypeAndNameRequest>(request).set_type_name(
          picked_node.type());
      absl::get<GetExecutionByTypeAndNameRequest>(request).set_execution_name(
          picked_node.name());
      return GetTransferredBytes(picked_node, curr_bytes);
    }
    case ReadNodesByPropertiesConfig::CONTEXT_BY_TYPE_AND_NAME: {
      InitializeReadRequest<GetContextByTypeAndNameRequest>(request);
      Context picked_node = absl::get<Context>(existing_nodes[node_index]);
      absl::get<GetContextByTypeAndNameRequest>(request).set_type_name(
          picked_node.type());
      absl::get<GetContextByTypeAndNameRequest>(request).set_context_name(
          picked_node.name());
      return GetTransferredBytes(picked_node, curr_bytes);
    }
    default:
      return tensorflow::errors::Unimplemented(
          "Wrong ReadNodesByProperties specification for read node by type and "
          "name in db.");
  }
}

}  // namespace

ReadNodesByProperties::ReadNodesByProperties(
    const ReadNodesByPropertiesConfig& read_nodes_by_properties_config,
    const int64 num_operations)
    : read_nodes_by_properties_config_(read_nodes_by_properties_config),
      num_operations_(num_operations),
      name_(absl::StrCat(
          "READ_", read_nodes_by_properties_config_.Specification_Name(
                       read_nodes_by_properties_config_.specification()))) {}

tensorflow::Status ReadNodesByProperties::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  int64 curr_bytes = 0;

  std::vector<Node> existing_nodes;
  TF_RETURN_IF_ERROR(GetExistingNodes(read_nodes_by_properties_config_, *store,
                                      existing_nodes));
  std::uniform_int_distribution<int64> node_index_dist{
      0, (int64)(existing_nodes.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    ReadNodesByPropertiesWorkItemType read_request;
    switch (read_nodes_by_properties_config_.specification()) {
      case ReadNodesByPropertiesConfig::ARTIFACTS_BY_IDs:
      case ReadNodesByPropertiesConfig::EXECUTIONS_BY_IDs:
      case ReadNodesByPropertiesConfig::CONTEXTS_BY_IDs:
        TF_RETURN_IF_ERROR(SetUpImplForReadNodesByIds(
            read_nodes_by_properties_config_, existing_nodes, node_index_dist,
            gen, read_request, curr_bytes));
        break;
      case ReadNodesByPropertiesConfig::ARTIFACTS_BY_URIs:
        TF_RETURN_IF_ERROR(SetUpImplForReadArtifactsByURIs(
            read_nodes_by_properties_config_, existing_nodes, node_index_dist,
            gen, read_request, curr_bytes));
        break;
      case ReadNodesByPropertiesConfig::ARTIFACTS_BY_TYPE:
      case ReadNodesByPropertiesConfig::EXECUTIONS_BY_TYPE:
      case ReadNodesByPropertiesConfig::CONTEXTS_BY_TYPE:
        TF_RETURN_IF_ERROR(SetUpImplForReadNodesByType(
            read_nodes_by_properties_config_, existing_nodes, node_index_dist,
            gen, read_request, curr_bytes));
        break;
      case ReadNodesByPropertiesConfig::ARTIFACT_BY_TYPE_AND_NAME:
      case ReadNodesByPropertiesConfig::EXECUTION_BY_TYPE_AND_NAME:
      case ReadNodesByPropertiesConfig::CONTEXT_BY_TYPE_AND_NAME:
        TF_RETURN_IF_ERROR(SetUpImplForReadNodeByTypeAndName(
            read_nodes_by_properties_config_, existing_nodes, node_index_dist,
            gen, read_request, curr_bytes));
        break;
      default:
        LOG(FATAL) << "Wrong specification for ReadNodesByProperties!";
    }
    std::cout << curr_bytes << std::endl;
    work_items_.emplace_back(read_request, curr_bytes);
  }
  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status ReadNodesByProperties::RunOpImpl(
    const int64 work_items_index, MetadataStore* store) {
  switch (read_nodes_by_properties_config_.specification()) {
    case ReadNodesByPropertiesConfig::ARTIFACTS_BY_IDs: {
      GetArtifactsByIDRequest request = absl::get<GetArtifactsByIDRequest>(
          work_items_[work_items_index].first);
      GetArtifactsByIDResponse response;
      return store->GetArtifactsByID(request, &response);
    }
    case ReadNodesByPropertiesConfig::EXECUTIONS_BY_IDs: {
      GetExecutionsByIDRequest request = absl::get<GetExecutionsByIDRequest>(
          work_items_[work_items_index].first);
      GetExecutionsByIDResponse response;
      return store->GetExecutionsByID(request, &response);
    }
    case ReadNodesByPropertiesConfig::CONTEXTS_BY_IDs: {
      GetContextsByIDRequest request = absl::get<GetContextsByIDRequest>(
          work_items_[work_items_index].first);
      GetContextsByIDResponse response;
      return store->GetContextsByID(request, &response);
    }
    case ReadNodesByPropertiesConfig::ARTIFACTS_BY_TYPE: {
      GetArtifactsByTypeRequest request = absl::get<GetArtifactsByTypeRequest>(
          work_items_[work_items_index].first);
      GetArtifactsByTypeResponse response;
      return store->GetArtifactsByType(request, &response);
    }
    case ReadNodesByPropertiesConfig::EXECUTIONS_BY_TYPE: {
      GetExecutionsByTypeRequest request =
          absl::get<GetExecutionsByTypeRequest>(
              work_items_[work_items_index].first);
      GetExecutionsByTypeResponse response;
      return store->GetExecutionsByType(request, &response);
    }
    case ReadNodesByPropertiesConfig::CONTEXTS_BY_TYPE: {
      GetContextsByTypeRequest request = absl::get<GetContextsByTypeRequest>(
          work_items_[work_items_index].first);
      GetContextsByTypeResponse response;
      return store->GetContextsByType(request, &response);
    }
    case ReadNodesByPropertiesConfig::ARTIFACT_BY_TYPE_AND_NAME: {
      GetArtifactByTypeAndNameRequest request =
          absl::get<GetArtifactByTypeAndNameRequest>(
              work_items_[work_items_index].first);
      GetArtifactByTypeAndNameResponse response;
      return store->GetArtifactByTypeAndName(request, &response);
    }
    case ReadNodesByPropertiesConfig::EXECUTION_BY_TYPE_AND_NAME: {
      GetExecutionByTypeAndNameRequest request =
          absl::get<GetExecutionByTypeAndNameRequest>(
              work_items_[work_items_index].first);
      GetExecutionByTypeAndNameResponse response;
      return store->GetExecutionByTypeAndName(request, &response);
    }
    case ReadNodesByPropertiesConfig::CONTEXT_BY_TYPE_AND_NAME: {
      GetContextByTypeAndNameRequest request =
          absl::get<GetContextByTypeAndNameRequest>(
              work_items_[work_items_index].first);
      GetContextByTypeAndNameResponse response;
      return store->GetContextByTypeAndName(request, &response);
    }
    case ReadNodesByPropertiesConfig::ARTIFACTS_BY_URIs: {
      GetArtifactsByURIRequest request = absl::get<GetArtifactsByURIRequest>(
          work_items_[work_items_index].first);
      GetArtifactsByURIResponse response;
      return store->GetArtifactsByURI(request, &response);
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
