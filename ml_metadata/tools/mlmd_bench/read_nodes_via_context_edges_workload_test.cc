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
#include "ml_metadata/tools/mlmd_bench/read_nodes_via_context_edges_workload.h"

#include <random>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

using ::testing::ValuesIn;

constexpr int kNumberOfOperations = 100;
constexpr int kNumberOfExistedTypesInDb = 100;
constexpr int kNumberOfExistedNodesInDb = 100;
constexpr int kNumberOfExistedContextEdgesInDb = 1000;

// Enumerates the workload configurations as the test parameters that ensure
// test coverage.
std::vector<WorkloadConfig> EnumerateConfigs() {
  std::vector<WorkloadConfig> configs;
  std::vector<ReadNodesViaContextEdgesConfig::Specification> specifications = {
      ReadNodesViaContextEdgesConfig::ARTIFACTS_BY_CONTEXT,
      ReadNodesViaContextEdgesConfig::EXECUTIONS_BY_CONTEXT,
      ReadNodesViaContextEdgesConfig::CONTEXTS_BY_ARTIFACT,
      ReadNodesViaContextEdgesConfig::CONTEXTS_BY_EXECUTION};

  for (const ReadNodesViaContextEdgesConfig::Specification& specification :
       specifications) {
    WorkloadConfig config;
    config.set_num_operations(kNumberOfOperations);
    config.mutable_read_nodes_via_context_edges_config()->set_specification(
        specification);
    configs.push_back(config);
  }

  return configs;
}

// Inserts some context edges(Attributions / Associations) into db so that we
// can read nodes via these context edges. Returns detailed error if query
// executions failed.
tensorflow::Status InsertContextEdgesInDb(const int64 num_attributions,
                                          const int64 num_associations,
                                          MetadataStore& store) {
  PutAttributionsAndAssociationsRequest request;
  std::minstd_rand0 gen(0);
  // Inserts Attributions.
  {
    FillContextEdgesConfig fill_context_edges_config;
    fill_context_edges_config.set_specification(
        FillContextEdgesConfig::ATTRIBUTION);
    std::vector<Node> existing_non_context_nodes;
    std::vector<Node> existing_context_nodes;
    TF_RETURN_IF_ERROR(GetExistingNodes(fill_context_edges_config, store,
                                        existing_non_context_nodes,
                                        existing_context_nodes));
    std::uniform_int_distribution<int64> non_context_node_index_dist{
        0, (int64)(existing_non_context_nodes.size() - 1)};
    std::uniform_int_distribution<int64> context_node_index_dist{
        0, (int64)(existing_context_nodes.size() - 1)};
    for (int i = 0; i < kNumberOfExistedContextEdgesInDb; ++i) {
      Attribution* context_edge = request.add_attributions();
      context_edge->set_artifact_id(
          absl::get<Artifact>(
              existing_non_context_nodes[non_context_node_index_dist(gen)])
              .id());
      context_edge->set_context_id(
          absl::get<Context>(
              existing_context_nodes[context_node_index_dist(gen)])
              .id());
    }
  }

  // Inserts Associations.
  {
    FillContextEdgesConfig fill_context_edges_config;
    fill_context_edges_config.set_specification(
        FillContextEdgesConfig::ASSOCIATION);
    std::vector<Node> existing_non_context_nodes;
    std::vector<Node> existing_context_nodes;
    TF_RETURN_IF_ERROR(GetExistingNodes(fill_context_edges_config, store,
                                        existing_non_context_nodes,
                                        existing_context_nodes));
    std::uniform_int_distribution<int64> non_context_node_index_dist{
        0, (int64)(existing_non_context_nodes.size() - 1)};
    std::uniform_int_distribution<int64> context_node_index_dist{
        0, (int64)(existing_context_nodes.size() - 1)};
    for (int i = 0; i < kNumberOfExistedContextEdgesInDb; ++i) {
      Association* context_edge = request.add_associations();
      context_edge->set_execution_id(
          absl::get<Execution>(
              existing_non_context_nodes[non_context_node_index_dist(gen)])
              .id());
      context_edge->set_context_id(
          absl::get<Context>(
              existing_context_nodes[context_node_index_dist(gen)])
              .id());
    }
  }

  PutAttributionsAndAssociationsResponse response;
  return store.PutAttributionsAndAssociations(request, &response);
}

// Test fixture that uses the same data configuration for multiple following
// parameterized ReadNodesViaContextEdges tests.
// The parameter here is the specific Workload configuration that contains
// the ReadNodesViaContextEdges configuration and the number of operations.
class ReadNodesViaContextEdgesParameterizedTestFixture
    : public ::testing::TestWithParam<WorkloadConfig> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    read_nodes_via_context_edges_ =
        absl::make_unique<ReadNodesViaContextEdges>(ReadNodesViaContextEdges(
            GetParam().read_nodes_via_context_edges_config(),
            GetParam().num_operations()));
    TF_ASSERT_OK(InsertTypesInDb(
        /*num_artifact_types=*/kNumberOfExistedTypesInDb,
        /*num_execution_types=*/kNumberOfExistedTypesInDb,
        /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
    TF_ASSERT_OK(InsertNodesInDb(
        /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
        /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
        /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));
    TF_ASSERT_OK(InsertContextEdgesInDb(
        /*num_attributions=*/kNumberOfExistedContextEdgesInDb,
        /*num_associations=*/kNumberOfExistedContextEdgesInDb, *store_));
  }

  std::unique_ptr<ReadNodesViaContextEdges> read_nodes_via_context_edges_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for ReadNodesViaContextEdges. Checks the SetUpImpl()
// indeed prepares a list of work items whose length is the same as the
// specified number of operations.
TEST_P(ReadNodesViaContextEdgesParameterizedTestFixture, SetUpImplTest) {
  TF_ASSERT_OK(read_nodes_via_context_edges_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(),
            read_nodes_via_context_edges_->num_operations());
}

// Tests the RunOpImpl() for ReadNodesViaContextEdges. Checks indeed all the
// work items have been executed and some bytes are transferred during the
// reading process.
TEST_P(ReadNodesViaContextEdgesParameterizedTestFixture, RunOpImplTest) {
  TF_ASSERT_OK(read_nodes_via_context_edges_->SetUp(store_.get()));

  int64 total_done = 0;
  ThreadStats stats;
  stats.Start();
  for (int64 i = 0; i < read_nodes_via_context_edges_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(
        read_nodes_via_context_edges_->RunOp(i, store_.get(), op_stats));
    stats.Update(op_stats, total_done);
  }
  stats.Stop();
  EXPECT_EQ(stats.done(), GetParam().num_operations());
  // Checks that the transferred bytes is greater that 0(the reading process
  // indeed occurred).
  EXPECT_GT(stats.bytes(), 0);
}

INSTANTIATE_TEST_CASE_P(ReadNodesViaContextEdgesTest,
                        ReadNodesViaContextEdgesParameterizedTestFixture,
                        ValuesIn(EnumerateConfigs()));

}  // namespace
}  // namespace ml_metadata
