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

#include <vector>

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

constexpr int kNumberOfOperations = 100;
constexpr int kNumberOfExistedTypesInDb = 100;
constexpr int kNumberOfExistedNodesInDb = 100;
constexpr int kNumberOfNodesPerRequest = 10;

constexpr char kConfig[] = R"(
        fill_nodes_config: {
          num_properties: { minimum: 1 maximum: 10 }
          string_value_bytes: { minimum: 1 maximum: 10 }
        }
      )";

// Enumerates the workload configurations as the test parameters that ensure
// test coverage.
std::vector<WorkloadConfig> EnumerateConfigs(const bool is_update) {
  std::vector<WorkloadConfig> configs;

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(kConfig);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_minimum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_maximum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->set_update(is_update);
    config.mutable_fill_nodes_config()->set_specification(
        FillNodesConfig::ARTIFACT);
    configs.push_back(config);
  }

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(kConfig);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_minimum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_maximum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->set_update(is_update);
    config.mutable_fill_nodes_config()->set_specification(
        FillNodesConfig::EXECUTION);
    configs.push_back(config);
  }

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(kConfig);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_minimum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_maximum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->set_update(is_update);
    config.mutable_fill_nodes_config()->set_specification(
        FillNodesConfig::CONTEXT);
    configs.push_back(config);
  }

  return configs;
}

// Test fixture that uses the same data configuration for multiple following
// parameterized FillNodes insert tests.
// The parameter here is the specific Workload configuration that contains
// the FillNodes insert configuration and the number of operations.
class FillNodesInsertParameterizedTestFixture
    : public ::testing::TestWithParam<WorkloadConfig> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    fill_nodes_ = absl::make_unique<FillNodes>(
        FillNodes(GetParam().fill_nodes_config(), GetParam().num_operations()));
  }

  std::unique_ptr<FillNodes> fill_nodes_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the fail cases when there are no types inside db for inserting nodes.
TEST_P(FillNodesInsertParameterizedTestFixture, NonTypesExistTest) {
  EXPECT_EQ(fill_nodes_->SetUp(store_.get()).code(),
            tensorflow::error::FAILED_PRECONDITION);
}

// Tests the SetUpImpl() for FillNodes insert cases when db contains no nodes in
// the beginning. Checks the SetUpImpl() indeed prepares a list of work items
// whose length is the same as the specified number of operations.
TEST_P(FillNodesInsertParameterizedTestFixture, SetUpImplWhenNoNodesExistTest) {
  // Inserts some types into db so that nodes can be inserted later.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  TF_ASSERT_OK(fill_nodes_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_nodes_->num_operations());
}

// Tests the RunOpImpl() for FillNodes insert cases when db contains no nodes in
// the beginning. Checks indeed all the work items have been executed and the
// number of the nodes inside db is the same as the number of operations
// specified in the workload.
TEST_P(FillNodesInsertParameterizedTestFixture, InsertWhenNoNodesExistTest) {
  // Inserts some types into db so that nodes can be inserted later.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  TF_ASSERT_OK(fill_nodes_->SetUp(store_.get()));
  for (int64 i = 0; i < fill_nodes_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_nodes_->RunOp(i, store_.get(), op_stats));
  }
  // Gets all the existing current nodes inside db after insertion.
  std::vector<Node> existing_nodes;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes));
  EXPECT_EQ(GetParam().num_operations() * kNumberOfNodesPerRequest,
            existing_nodes.size());
}

// Tests the SetUpImpl() for FillNodes insert cases when db contains some nodes
// in the beginning. Checks the SetUpImpl() indeed prepares a list of work items
// whose length is the same as the specified number of operations.
TEST_P(FillNodesInsertParameterizedTestFixture,
       SetUpImplWhenSomeNodesExistTest) {
  // Inserts some types into db so that nodes can be inserted later.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  // Inserts some nodes into db in the first beginning.
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
      /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
      /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));
  TF_ASSERT_OK(fill_nodes_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_nodes_->num_operations());
}

// Tests the RunOpImpl() for FillNodes insert cases when db contains some nodes
// in the beginning. Checks indeed all the work items have been executed and the
// number of the new added nodes inside db is the same as the number of
// operations specified in the workload.
TEST_P(FillNodesInsertParameterizedTestFixture, InsertWhenSomeNodesExistTest) {
  // Inserts some types into db so that nodes can be inserted later.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  // Inserts some nodes into db in the first beginning.
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
      /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
      /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));

  // Gets all the pre-inserted nodes inside db before insertion.
  std::vector<Node> existing_nodes_before_insert;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_before_insert));
  TF_ASSERT_OK(fill_nodes_->SetUp(store_.get()));
  for (int64 i = 0; i < fill_nodes_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_nodes_->RunOp(i, store_.get(), op_stats));
  }
  // Gets all the existing current nodes inside db after insertion.
  std::vector<Node> existing_nodes_after_insert;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_after_insert));

  EXPECT_EQ(
      GetParam().num_operations() * kNumberOfNodesPerRequest,
      existing_nodes_after_insert.size() - existing_nodes_before_insert.size());
}

INSTANTIATE_TEST_CASE_P(
    FillNodesInsertTest, FillNodesInsertParameterizedTestFixture,
    ::testing::ValuesIn(EnumerateConfigs(/*is_update=*/false)));

}  // namespace
}  // namespace ml_metadata
