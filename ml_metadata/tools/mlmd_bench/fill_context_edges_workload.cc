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
#include "ml_metadata/tools/mlmd_bench/fill_context_edges_workload.h"

#include <random>
#include <vector>

#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace ml_metadata {
namespace {

template <typename T>
void InitializePutRequest(const int64 num_edges,
                          PutAttributionsAndAssociationsRequest& put_request) {
  for (int64 i = 0; i < num_edges; ++i) {
    if (std::is_same<T, Attribution>::value) {
      put_request.add_attributions();
    } else if (std::is_same<T, Association>::value) {
      put_request.add_associations();
    } else {
      LOG(FATAL) << "Unexpected Context Edges Type for initializing current "
                    "edge batch";
    }
  }
}
template <typename T>
tensorflow::Status GenerateContextEdge(
    const int64 non_context_node_id, const int64 context_node_id,
    PutAttributionsAndAssociationsRequest& put_request,
    std::unordered_map<int64, std::unordered_set<int64>>& unique_checker,
    int64& curr_bytes) {
  CHECK((std::is_same<T, Attribution>::value ||
         std::is_same<T, Association>::value))
      << "Unexpected Types";
  if (unique_checker.find(context_node_id) != unique_checker.end() &&
      unique_checker.at(context_node_id).find(non_context_node_id) !=
          unique_checker.at(context_node_id).end()) {
    return tensorflow::errors::AlreadyExists(("Existing context edge!"));
  }
  if (std::is_same<T, Attribution>::value) {
    Attribution* attribution = put_request.add_attributions();
    attribution->set_artifact_id(non_context_node_id);
    attribution->set_context_id(context_node_id);
  } else if (std::is_same<T, Association>::value) {
    Association* association = put_request.add_associations();
    association->set_execution_id(non_context_node_id);
    association->set_context_id(context_node_id);
  }
  if (unique_checker.find(context_node_id) == unique_checker.end()) {
    unique_checker.insert({context_node_id, std::unordered_set<int64>{}});
  }
  unique_checker.at(context_node_id).insert(non_context_node_id);
  curr_bytes += 8;
  return tensorflow::Status::OK();
}

}  // namespace

FillContextEdges::FillContextEdges(
    const FillContextEdgesConfig& fill_context_edges_config,
    int64 num_operations)
    : fill_context_edges_config_(fill_context_edges_config),
      num_operations_(num_operations) {
  switch (fill_context_edges_config_.specification()) {
    case FillContextEdgesConfig::ATTRIBUTION: {
      name_ = "fill_attribution";
      break;
    }
    case FillContextEdgesConfig::ASSOCIATION: {
      name_ = "fill_association";
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillContextEdges!";
  }
}

tensorflow::Status FillContextEdges::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  int64 curr_bytes = 0;

  std::vector<NodeType> existing_non_context_nodes;
  TF_RETURN_IF_ERROR(
      GetExistingNodes(fill_context_edges_config_.specification(), store,
                       existing_non_context_nodes));
  std::uniform_int_distribution<int64> uniform_dist_non_context_index{
      0, (int64)(existing_non_context_nodes.size() - 1)};
  std::vector<NodeType> existing_context_nodes;
  TF_RETURN_IF_ERROR(
      GetExistingNodes(/*specification=*/2, store, existing_context_nodes));
  std::uniform_int_distribution<int64> uniform_dist_context_index{
      0, (int64)(existing_context_nodes.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  for (int i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    PutAttributionsAndAssociationsRequest put_request;
    const int64 num_edges =
        GenerateRandomNumberFromUD(fill_context_edges_config_.num_edges(), gen);
    switch (fill_context_edges_config_.specification()) {
      case FillContextEdgesConfig::ATTRIBUTION: {
        InitializePutRequest<Attribution>(num_edges, put_request);
        google::protobuf::RepeatedPtrField<Attribution>& edge_batch =
            *put_request.mutable_attributions();
        // TF_RETURN_IF_ERROR(GenerateContextEdge<Attribution>(
        //     existing_non_context_nodes, existing_context_nodes,
        //     uniform_dist_non_context_index, uniform_dist_context_index, gen,
        //     put_request, unique_checker_, curr_bytes));
        break;
      }
      case FillContextEdgesConfig::ASSOCIATION: {
        InitializePutRequest<Association>(num_edges, put_request);
        google::protobuf::RepeatedPtrField<Association>& edge_batch =
            *put_request.mutable_associations();
        // TF_RETURN_IF_ERROR(GenerateContextEdge<Association>(
        //     absl::get<Execution>(
        //         existing_non_context_nodes[non_context_node_index])
        //         .id(),
        //     absl::get<Context>(existing_context_nodes[context_node_index]).id(),
        //     put_request, unique_checker_, curr_bytes));
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for FillContextEdges!";
    }
    work_items_.emplace_back(put_request, curr_bytes);
  }

  return tensorflow::Status::OK();
}  // namespace ml_metadata

// Executions of work items.
tensorflow::Status FillContextEdges::RunOpImpl(const int64 work_items_index,
                                               MetadataStore* store) {
  PutAttributionsAndAssociationsRequest put_request =
      work_items_[work_items_index].first;
  PutAttributionsAndAssociationsResponse put_response;
  return store->PutAttributionsAndAssociations(put_request, &put_response);
}

tensorflow::Status FillContextEdges::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string FillContextEdges::GetName() { return name_; }

}  // namespace ml_metadata
