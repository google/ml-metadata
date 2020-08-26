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

#include <algorithm>
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

constexpr int64 kInt64TypeIdSize = 8;

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

// Gets the transferred bytes for certain property of a node and returns the
// bytes.
int64 GetTransferredBytesForNodeProperty(
    const std::string& property_name,
    const std::string& property_string_value) {
  int64 bytes = property_name.size();
  bytes += property_string_value.size();
  return bytes;
}

// Calculates and returns the transferred bytes for each node that will be
// inserted or updated later.
template <typename NT>
int64 GetTransferredBytesForNodeAttributes(const bool& update, const NT& node) {
  // For update, since node's name and type id will not be updated, returns 0.
  if (update) {
    return 0;
  }
  return node.name().size() + kInt64TypeIdSize;
}

// Sets ArtifactType `update_type` for modifying the selected update node's
// properties in later SetNodePropertiesGivenType(). Returns detailed error if
// query executions failed.
tensorflow::Status SetTypeForUpdateNode(const Artifact& selected_node,
                                        MetadataStore& store,
                                        ArtifactType& update_type) {
  GetArtifactTypesByIDRequest request;
  request.add_type_ids(selected_node.type_id());
  GetArtifactTypesByIDResponse response;
  TF_RETURN_IF_ERROR(store.GetArtifactTypesByID(request, &response));
  update_type = response.artifact_types(0);
  return tensorflow::Status::OK();
}

// Sets ExecutionType `update_type` for modifying the selected update node's
// properties in later SetNodePropertiesGivenType(). Returns detailed error if
// query executions failed.
tensorflow::Status SetTypeForUpdateNode(const Execution& selected_node,
                                        MetadataStore& store,
                                        ExecutionType& update_type) {
  GetExecutionTypesByIDRequest request;
  request.add_type_ids(selected_node.type_id());
  GetExecutionTypesByIDResponse response;
  TF_RETURN_IF_ERROR(store.GetExecutionTypesByID(request, &response));
  update_type = response.execution_types(0);
  return tensorflow::Status::OK();
}

// Sets ContextType `update_type` for modifying the selected update node's
// properties in later SetNodePropertiesGivenType(). Returns detailed error if
// query executions failed.
tensorflow::Status SetTypeForUpdateNode(const Context& selected_node,
                                        MetadataStore& store,
                                        ContextType& update_type) {
  GetContextTypesByIDRequest request;
  request.add_type_ids(selected_node.type_id());
  GetContextTypesByIDResponse response;
  TF_RETURN_IF_ERROR(store.GetContextTypesByID(request, &response));
  update_type = response.context_types(0);
  return tensorflow::Status::OK();
}

// Inserts the makeup artifact into db for later update. Returns
// detailed error if query executions failed.
tensorflow::Status PrepareMakeUpNodeInDb(MetadataStore& store, Artifact& node) {
  PutArtifactsRequest request;
  request.add_artifacts()->CopyFrom(node);
  PutArtifactsResponse response;
  TF_RETURN_IF_ERROR(store.PutArtifacts(request, &response));
  // Sets the node id for indicating this is an update when calling the APIs.
  node.set_id(response.artifact_ids(0));
  return tensorflow::Status::OK();
}

// Inserts the makeup execution into db for later update. Returns
// detailed error if query executions failed.
tensorflow::Status PrepareMakeUpNodeInDb(MetadataStore& store,
                                         Execution& node) {
  PutExecutionsRequest request;
  request.add_executions()->CopyFrom(node);
  PutExecutionsResponse response;
  TF_RETURN_IF_ERROR(store.PutExecutions(request, &response));
  // Sets the node id for indicating this is an update when   calling the APIs.
  node.set_id(response.execution_ids(0));
  return tensorflow::Status::OK();
}

// Inserts the makeup context into db for later update. Returns
// detailed error if query executions failed.
tensorflow::Status PrepareMakeUpNodeInDb(MetadataStore& store, Context& node) {
  PutContextsRequest request;
  request.add_contexts()->CopyFrom(node);
  PutContextsResponse response;
  TF_RETURN_IF_ERROR(store.PutContexts(request, &response));
  // Sets the node id for indicating this is an update when calling the APIs.
  node.set_id(response.context_ids(0));
  return tensorflow::Status::OK();
}

// Prepares update node for later update. If there is no left existing nodes in
// `existing_nodes` to select from, generates a random makeup node under
// `insert_type` and inserts it into db. Returns detailed error if query
// executions failed.
template <typename T, typename N>
tensorflow::Status PrepareNodeForUpdate(const T& insert_type, const int64 i,
                                        MetadataStore& store,
                                        std::vector<Node>& existing_nodes,
                                        N& node, T& update_type) {
  if (!existing_nodes.empty()) {
    Node existing_node = existing_nodes.back();
    node = absl::get<N>(existing_node);
    TF_RETURN_IF_ERROR(SetTypeForUpdateNode(node, store, update_type));
    // Removes this existing update node from `existing_nodes` for avoiding
    // abort errors under multi-thread environment.
    existing_nodes.pop_back();
  } else {
    // Sets `update_type` to `insert_type` for makeup node.
    update_type = insert_type;
    node.set_type_id(insert_type.id());
    node.set_name(
        absl::StrCat("makeup_node", absl::FormatTime(absl::Now()), "_", i));
    // Inserts the makeup node into db.
    TF_RETURN_IF_ERROR(PrepareMakeUpNodeInDb(store, node));
  }
  return tensorflow::Status::OK();
}

// Clears some custom node properties for update node to ensure that for update
// cases, the number of properties / custom properties being added, deleted or
// updated is equal to `nodes_param.num_properties`. Returns the cleared
// properties' bytes.
template <typename N>
int64 ClearSomeProperitesForUpdateNode(const NodesParam& nodes_param,
                                       int64& num_custom_properties_to_clear,
                                       N& node) {
  int64 bytes = 0;
  num_custom_properties_to_clear = std::min(
      (int64)node.custom_properties_size(), nodes_param.num_properties);
  auto it = node.custom_properties().begin();
  for (int64 i = 0; i < num_custom_properties_to_clear; ++i) {
    node.mutable_custom_properties()->erase(it->first);
    bytes += GetTransferredBytesForNodeProperty(it->first,
                                                it->second.string_value());
    it++;
  }
  return bytes;
}

// Adds or updates `node`'s properties / custom properties. Returns the added or
// updated properties' bytes.
template <typename T, typename N>
int64 AddOrUpdateNodeProperties(const NodesParam& nodes_param, const T& type,
                                int64 remain_num_properties_to_change,
                                N& node) {
  int bytes = 0;
  // Uses "********" as the fake property value for current node.
  std::string property_value(nodes_param.string_value_bytes, '*');
  // Loops over the types properties while generating the node's properties
  // accordingly.
  // TODO(briansong) Adds more property types support.
  for (const auto& p : type.properties()) {
    (*node.mutable_properties())[p.first].set_string_value(property_value);
    bytes += GetTransferredBytesForNodeProperty(p.first, property_value);
    if (--remain_num_properties_to_change == 0) {
      break;
    }
  }
  // If the node's number of properties is greater than the type, uses custom
  // properties instead.
  while (remain_num_properties_to_change-- > 0) {
    (*node.mutable_custom_properties())[absl::StrCat(
                                            "custom_p-",
                                            remain_num_properties_to_change)]
        .set_string_value(property_value);
    bytes += GetTransferredBytesForNodeProperty(
        absl::StrCat("custom_p-", remain_num_properties_to_change),
        property_value);
  }
  return bytes;
}

// Sets node's properties and custom properties given `type`. Returns the
// transferred bytes for inserting or updating current `node`.
template <typename T, typename N>
int64 SetNodePropertiesGivenType(const FillNodesConfig& fill_nodes_config,
                                 const NodesParam& nodes_param, const T& type,
                                 N& node) {
  int64 num_custom_properties_to_clear = 0;
  int64 transferred_bytes_for_cleared_node_properties = 0;
  if (fill_nodes_config.update()) {
    transferred_bytes_for_cleared_node_properties +=
        ClearSomeProperitesForUpdateNode(nodes_param,
                                         num_custom_properties_to_clear, node);
  }

  int64 remain_num_properties_to_change =
      nodes_param.num_properties - num_custom_properties_to_clear;
  // If there are no properties that needed to be added or updated further,
  // return the current total transferred bytes directly.
  if (remain_num_properties_to_change == 0) {
    return GetTransferredBytesForNodeAttributes<N>(fill_nodes_config.update(),
                                                   node) +
           transferred_bytes_for_cleared_node_properties;
  }

  int64 transferred_bytes_for_added_or_updated_node_properties =
      AddOrUpdateNodeProperties(nodes_param, type,
                                remain_num_properties_to_change, node);

  return GetTransferredBytesForNodeAttributes<N>(fill_nodes_config.update(),
                                                 node) +
         transferred_bytes_for_cleared_node_properties +
         transferred_bytes_for_added_or_updated_node_properties;
}

// Generates insert / update node.  Returns detailed error if query executions
// failed.
template <typename T, typename N>
tensorflow::Status GenerateNodes(const FillNodesConfig& fill_nodes_config,
                                 const NodesParam& nodes_param,
                                 const T& insert_type, MetadataStore& store,
                                 std::vector<Node>& existing_nodes,
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
    if (fill_nodes_config.update()) {
      // Update mode.
      T update_type;
      TF_RETURN_IF_ERROR(PrepareNodeForUpdate<T, N>(
          insert_type, i, store, existing_nodes, nodes[i], update_type));
      // Uses `update_type` when calling SetNodePropertiesGivenType() for update
      // mode.
      curr_bytes += SetNodePropertiesGivenType(fill_nodes_config, nodes_param,
                                               update_type, nodes[i]);
    } else {
      // Insert mode.
      nodes[i].set_name(absl::StrCat(nodes_param.nodes_name, "_node_", i));
      nodes[i].set_type_id(insert_type.id());
      // Uses `insert_type` when calling SetNodePropertiesGivenType() for insert
      // mode.
      curr_bytes += SetNodePropertiesGivenType(fill_nodes_config, nodes_param,
                                               insert_type, nodes[i]);
    }
  }
  return tensorflow::Status::OK();
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

  // Gets all the specific types in db to choose from when inserting nodes.
  // If there's no types in the store, return error.
  std::vector<Type> existing_types;
  TF_RETURN_IF_ERROR(
      GetAndValidateExistingTypes(fill_nodes_config_, *store, existing_types));
  std::uniform_int_distribution<int64> uniform_dist_type_index{
      0, (int64)(existing_types.size() - 1)};
  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));
  // Gets all existing nodes in db to choose from when updating nodes.
  std::vector<Node> existing_nodes;
  TF_RETURN_IF_ERROR(
      GetExistingNodes(fill_nodes_config_, *store, existing_nodes));
  std::shuffle(std::begin(existing_nodes), std::end(existing_nodes), gen);

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
        TF_RETURN_IF_ERROR(GenerateNodes<ArtifactType, Artifact>(
            fill_nodes_config_, nodes_param,
            absl::get<ArtifactType>(existing_types[type_index]), *store,
            existing_nodes, *nodes, curr_bytes));
        curr_bytes +=
            SetArtifactsAdditionalFields(nodes_param.nodes_name, *nodes);
        break;
      }
      case FillNodesConfig::EXECUTION: {
        InitializePutRequest<PutExecutionsRequest>(num_nodes, put_request);
        auto nodes =
            absl::get<PutExecutionsRequest>(put_request).mutable_executions();
        TF_RETURN_IF_ERROR(GenerateNodes<ExecutionType, Execution>(
            fill_nodes_config_, nodes_param,
            absl::get<ExecutionType>(existing_types[type_index]), *store,
            existing_nodes, *nodes, curr_bytes));
        curr_bytes += SetExecutionsAdditionalFields(*nodes);
        break;
      }
      case FillNodesConfig::CONTEXT: {
        InitializePutRequest<PutContextsRequest>(num_nodes, put_request);
        TF_RETURN_IF_ERROR(GenerateNodes<ContextType, Context>(
            fill_nodes_config_, nodes_param,
            absl::get<ContextType>(existing_types[type_index]), *store,
            existing_nodes,
            *absl::get<PutContextsRequest>(put_request).mutable_contexts(),
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
