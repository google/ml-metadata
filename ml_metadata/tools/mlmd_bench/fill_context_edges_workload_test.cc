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

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

constexpr int kNumberOfOperations = 100;
constexpr int kNumberOfExistedTypesInDb = 132;
constexpr int kNumberOfExistedNodesInDb = 145;
constexpr int kNumberOfExistedContextEdgesInDb = 105;
constexpr int kNumberOfEdgesPerRequest = 5;

constexpr auto config_str = R"(
        fill_context_edges_config: {
          non_context_node_popularity: {dirichlet_alpha : 1}
          context_node_popularity: {dirichlet_alpha : 1}
        }
      )";

// Enumerates the workload configurations as the test parameters that ensure
// test coverage.
std::vector<WorkloadConfig> EnumerateConfigs() {
  std::vector<WorkloadConfig> configs;

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(config_str);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_fill_context_edges_config()
        ->mutable_num_edges()
        ->set_minimum(kNumberOfEdgesPerRequest);
    config.mutable_fill_context_edges_config()
        ->mutable_num_edges()
        ->set_maximum(kNumberOfEdgesPerRequest);
    config.mutable_fill_context_edges_config()->set_specification(
        FillContextEdgesConfig::ATTRIBUTION);
    configs.push_back(config);
  }

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(config_str);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_fill_context_edges_config()
        ->mutable_num_edges()
        ->set_minimum(kNumberOfEdgesPerRequest);
    config.mutable_fill_context_edges_config()
        ->mutable_num_edges()
        ->set_maximum(kNumberOfEdgesPerRequest);
    config.mutable_fill_context_edges_config()->set_specification(
        FillContextEdgesConfig::ASSOCIATION);
    configs.push_back(config);
  }

  return configs;
}

// Gets the number of existed context edges in db. Returns detailed error
// if query executions failed.
tensorflow::Status GetNumberOfContextEdgesInDb(
    const FillContextEdgesConfig& fill_context_edges_config,
    MetadataStore& store, int64& num_context_edges) {
  GetContextsResponse get_response;
  TF_RETURN_IF_ERROR(store.GetContexts(
      /*request=*/{}, &get_response));
  for (const auto& context : get_response.contexts()) {
    switch (fill_context_edges_config.specification()) {
      case FillContextEdgesConfig::ATTRIBUTION: {
        GetArtifactsByContextRequest request;
        request.set_context_id(context.id());
        GetArtifactsByContextResponse response;
        TF_RETURN_IF_ERROR(store.GetArtifactsByContext(request, &response));
        num_context_edges += response.artifacts_size();
        break;
      }
      case FillContextEdgesConfig::ASSOCIATION: {
        GetExecutionsByContextRequest request;
        request.set_context_id(context.id());
        GetExecutionsByContextResponse response;
        TF_RETURN_IF_ERROR(store.GetExecutionsByContext(request, &response));
        num_context_edges += response.executions_size();
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for FillContextEdges!";
    }
  }
  return tensorflow::Status::OK();
}

// Inserts `num_attributions` attributions and `num_associations` associations
// inside db. Returns detailed error if query executions failed.
tensorflow::Status InsertContextEdgesInDb(const int64 num_attributions,
                                          const int64 num_associations,
                                          MetadataStore& store) {
  GetArtifactsResponse get_artifacts_response;
  TF_RETURN_IF_ERROR(store.GetArtifacts(
      /*request=*/{}, &get_artifacts_response));
  GetExecutionsResponse get_executions_response;
  TF_RETURN_IF_ERROR(store.GetExecutions(
      /*request=*/{}, &get_executions_response));
  GetContextsResponse get_contexts_response;
  TF_RETURN_IF_ERROR(store.GetContexts(
      /*request=*/{}, &get_contexts_response));

  PutAttributionsAndAssociationsRequest put_request;

  int64 populated_attributions = 0;
  int64 populated_associations = 0;

  for (int64 i = 0; i < get_artifacts_response.artifacts_size() &&
                    populated_attributions < num_attributions;
       ++i) {
    for (int64 j = 0; j < get_contexts_response.contexts_size() &&
                      populated_attributions < num_attributions;
         ++j) {
      Attribution* context_edge = put_request.add_attributions();
      context_edge->set_artifact_id(get_artifacts_response.artifacts(i).id());
      context_edge->set_context_id(get_contexts_response.contexts(j).id());
      populated_attributions++;
    }
  }

  for (int64 i = 0; i < get_executions_response.executions_size() &&
                    populated_associations < num_associations;
       ++i) {
    for (int64 j = 0; j < get_contexts_response.contexts_size() &&
                      populated_associations < num_associations;
         ++j) {
      Association* context_edge = put_request.add_associations();
      context_edge->set_execution_id(
          get_executions_response.executions(i).id());
      context_edge->set_context_id(get_contexts_response.contexts(j).id());
      populated_associations++;
    }
  }

  PutAttributionsAndAssociationsResponse response;
  return store.PutAttributionsAndAssociations(put_request, &response);
}

// Test fixture that uses the same data configuration for multiple following
// parameterized FillContextEdges tests.
// The parameter here is the specific Workload configuration that contains
// the FillContextEdges configuration and the number of operations.
class FillContextEdgesParameterizedTestFixture
    : public ::testing::TestWithParam<WorkloadConfig> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    fill_context_edges_ = absl::make_unique<FillContextEdges>(FillContextEdges(
        GetParam().fill_context_edges_config(), GetParam().num_operations()));
  }

  std::unique_ptr<FillContextEdges> fill_context_edges_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for FillContextEdges when db contains no context edges
// in the beginning. Checks the SetUpImpl() indeed prepares a list of work items
// whose length is the same as the specified number of operations.
TEST_P(FillContextEdgesParameterizedTestFixture,
       SetUpImplWhenNoContextEdgesExistTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
      /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
      /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));

  TF_ASSERT_OK(fill_context_edges_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_context_edges_->num_operations());
}

// Tests the RunOpImpl() for FillContextEdges when db contains no context edges
// in the beginning. Checks indeed all the work items have been executed and the
// number of the context edges inside db is the same as the number of operations
// specified in the workload.
TEST_P(FillContextEdgesParameterizedTestFixture,
       InsertWhenNoContextEdgesExistTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
      /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
      /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));

  TF_ASSERT_OK(fill_context_edges_->SetUp(store_.get()));
  for (int64 i = 0; i < fill_context_edges_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_context_edges_->RunOp(i, store_.get(), op_stats));
  }
  int64 num_context_edges = 0;
  TF_ASSERT_OK(GetNumberOfContextEdgesInDb(
      GetParam().fill_context_edges_config(), *store_, num_context_edges));
  EXPECT_EQ(GetParam().num_operations() * kNumberOfEdgesPerRequest,
            num_context_edges);
}

// Tests the SetUpImpl() for FillContextEdges when db contains some context
// edges in the beginning. Checks the SetUpImpl() indeed prepares a list of work
// items whose length is the same as the specified number of operations.
TEST_P(FillContextEdgesParameterizedTestFixture,
       SetUpImplWhenSomeContextEdgesExistTest) {
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

  TF_ASSERT_OK(fill_context_edges_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_context_edges_->num_operations());
}

// Tests the RunOpImpl() for FillContextEdges when db contains some context
// edges in the beginning. Checks indeed all the work items have been executed
// and the number of new added context edges inside db is the same as the
// number of operations specified in the workload.
TEST_P(FillContextEdgesParameterizedTestFixture,
       InsertWhenSomeContextEdgesExistTest) {
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

  int64 num_context_edges_before = 0;
  TF_ASSERT_OK(
      GetNumberOfContextEdgesInDb(GetParam().fill_context_edges_config(),
                                  *store_, num_context_edges_before));
  TF_ASSERT_OK(fill_context_edges_->SetUp(store_.get()));

  for (int64 i = 0; i < fill_context_edges_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_context_edges_->RunOp(i, store_.get(), op_stats));
  }

  int64 num_context_edges_after = 0;
  TF_ASSERT_OK(
      GetNumberOfContextEdgesInDb(GetParam().fill_context_edges_config(),
                                  *store_, num_context_edges_after));

  EXPECT_EQ(GetParam().num_operations() * kNumberOfEdgesPerRequest,
            num_context_edges_after - num_context_edges_before);
}

INSTANTIATE_TEST_CASE_P(FillContextEdgesTest,
                        FillContextEdgesParameterizedTestFixture,
                        ::testing::ValuesIn(EnumerateConfigs()));

}  // namespace
}  // namespace ml_metadata
