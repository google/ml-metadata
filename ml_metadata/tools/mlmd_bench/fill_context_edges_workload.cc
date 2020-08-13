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
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
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

constexpr int64 kInt64IdSize = 8;
constexpr int64 kNumNodeIdsPerEdge = 2;

// Generates and returns a categorical distribution with Dirichlet Prior
// specified by `dist`.
std::discrete_distribution<int64>
GenerateCategoricalDistributionWithDirichletPrior(
    const int64 sample_size, const CategoricalDistribution& dist,
    std::minstd_rand0& gen) {
  // With a source of Gamma-distributed random variates, draws `sample_size`
  // independent random samples and store in `weights`.
  std::gamma_distribution<double> gamma_dist(dist.dirichlet_alpha(), 1.0);
  std::vector<double> weights(sample_size);
  for (int64 i = 0; i < sample_size; ++i) {
    weights[i] = gamma_dist(gen);
  }
  // Uses these random number generated w.r.t. a Dirichlet distribution with
  // `concentration_param` to represent the possibility of being chosen for each
  // integer within [0, sample_size) in a discrete distribution.
  return std::discrete_distribution<int64>{weights.begin(), weights.end()};
}

// Chooses a non context node from `existing_non_context_nodes` with
// `non_context_node_index` and returns its node id.
template <typename T>
int64 GenerateNonContextNodeId(const std::vector<Node>& non_context_nodes,
                               const int64 non_context_node_index) {
  if (std::is_same<T, Attribution>::value) {
    return absl::get<Artifact>(non_context_nodes[non_context_node_index]).id();
  }
  return absl::get<Execution>(non_context_nodes[non_context_node_index]).id();
}

// Checks if current `non_context_node_id` and `context_node_id` pair has been
// seen in current setup (all the previous pairs have been stored inside
// `edges_seen`).
// Returns true if the current pair is a duplicate one.
bool CheckDuplicateContextEdge(
    const int64 non_context_node_id, const int64 context_node_id,
    absl::flat_hash_map<int64, absl::flat_hash_set<int64>>& edges_seen) {
  if (edges_seen.contains(context_node_id) &&
      edges_seen[context_node_id].contains(non_context_node_id)) {
    return true;
  }
  // Stores the current pair into the `edges_seen` for future duplicate
  // checking.
  if (!edges_seen.contains(context_node_id)) {
    edges_seen.insert({context_node_id, {}});
  }
  edges_seen[context_node_id].insert(non_context_node_id);
  return false;
}

// Adds a new context edge to `request`. Its properties will be set according to
// `non_context_node_id` and `context_node_id`.
template <typename T>
void SetCurrentContextEdge(const int64 non_context_node_id,
                           const int64 context_node_id,
                           PutAttributionsAndAssociationsRequest& request) {
  if (std::is_same<T, Attribution>::value) {
    Attribution* context_edge = request.add_attributions();
    context_edge->set_artifact_id(non_context_node_id);
    context_edge->set_context_id(context_node_id);
  } else {
    Association* context_edge = request.add_associations();
    context_edge->set_execution_id(non_context_node_id);
    context_edge->set_context_id(context_node_id);
  }
}

// Generates `num_edges` context edges within `request`.
// Selects `num_edges` non-context and context node pairs from
// `existing_non_context_nodes` and `existing_context_nodes` w.r.t. to the
// categorical distribution with a Dirichlet prior. Also, performs rejection
// sampling to ensure current generated pair is unique. Returns detailed error
// if query executions failed.
template <typename T>
tensorflow::Status GenerateContextEdges(
    const std::vector<Node>& existing_non_context_nodes,
    const std::vector<Node>& existing_context_nodes, const int64 num_edges,
    std::discrete_distribution<int64>& non_context_node_index_dist,
    std::discrete_distribution<int64>& context_node_index_dist,
    std::minstd_rand0& gen, MetadataStore& store,
    absl::flat_hash_map<int64, absl::flat_hash_set<int64>>& edges_seen,
    PutAttributionsAndAssociationsRequest& request, int64& curr_bytes) {
  CHECK((std::is_same<T, Attribution>::value ||
         std::is_same<T, Association>::value))
      << "Unexpected Types";
  int64 i = 0;
  // Uses a while loop to perform rejection sampling. If the current context
  // edge has been seen before in the current setup, uses `continue`
  // to jump into the next iteration for generating a new context edge.
  while (i < num_edges) {
    const int64 non_context_node_id = GenerateNonContextNodeId<T>(
        existing_non_context_nodes, non_context_node_index_dist(gen));
    const int64 context_node_id =
        absl::get<Context>(existing_context_nodes[context_node_index_dist(gen)])
            .id();
    // Rejection sampling.
    if (CheckDuplicateContextEdge(non_context_node_id, context_node_id,
                                  edges_seen)) {
      continue;
    }
    SetCurrentContextEdge<T>(non_context_node_id, context_node_id, request);
    // Increases `curr_bytes` with the size of `non_context_node_id` and
    // `context_node_id`(int64 takes 8 bytes).
    curr_bytes += kInt64IdSize * kNumNodeIdsPerEdge;
    i++;
  }
  return tensorflow::Status::OK();
}

}  // namespace

FillContextEdges::FillContextEdges(
    const FillContextEdgesConfig& fill_context_edges_config,
    int64 num_operations)
    : fill_context_edges_config_(fill_context_edges_config),
      num_operations_(num_operations),
      name_(absl::StrCat("FILL_",
                         fill_context_edges_config_.Specification_Name(
                             fill_context_edges_config_.specification()))) {}

tensorflow::Status FillContextEdges::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  int64 curr_bytes = 0;
  std::uniform_int_distribution<int64> num_edges_dist{
      fill_context_edges_config_.num_edges().minimum(),
      fill_context_edges_config_.num_edges().maximum()};

  std::vector<Node> existing_non_context_nodes;
  std::vector<Node> existing_context_nodes;
  TF_RETURN_IF_ERROR(GetExistingNodes(fill_context_edges_config_, *store,
                                      existing_non_context_nodes,
                                      existing_context_nodes));

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));
  // Generates categorical distribution with a Dirichlet distribution with
  // a concentrate parameter.
  std::discrete_distribution<int64> non_context_node_index_dist =
      GenerateCategoricalDistributionWithDirichletPrior(
          existing_non_context_nodes.size(),
          fill_context_edges_config_.non_context_node_popularity(), gen);
  std::discrete_distribution<int64> context_node_index_dist =
      GenerateCategoricalDistributionWithDirichletPrior(
          existing_context_nodes.size(),
          fill_context_edges_config_.context_node_popularity(), gen);

  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    PutAttributionsAndAssociationsRequest put_request;
    const int64 num_edges = num_edges_dist(gen);
    switch (fill_context_edges_config_.specification()) {
      case FillContextEdgesConfig::ATTRIBUTION: {
        TF_RETURN_IF_ERROR(GenerateContextEdges<Attribution>(
            existing_non_context_nodes, existing_context_nodes, num_edges,
            non_context_node_index_dist, context_node_index_dist, gen, *store,
            edges_seen_, put_request, curr_bytes));
        break;
      }
      case FillContextEdgesConfig::ASSOCIATION: {
        TF_RETURN_IF_ERROR(GenerateContextEdges<Association>(
            existing_non_context_nodes, existing_context_nodes, num_edges,
            non_context_node_index_dist, context_node_index_dist, gen, *store,
            edges_seen_, put_request, curr_bytes));
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for FillContextEdges!";
    }
    work_items_.emplace_back(put_request, curr_bytes);
  }

  return tensorflow::Status::OK();
}

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
