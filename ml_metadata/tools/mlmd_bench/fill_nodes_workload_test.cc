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

// Enumerates the workload configurations as the test parameters that ensure
// test coverage.
std::vector<WorkloadConfig> EnumerateConfigs(const bool is_update) {
  std::vector<WorkloadConfig> config_vector;
  WorkloadConfig template_config = testing::ParseTextProtoOrDie<WorkloadConfig>(
      R"(
        fill_nodes_config: {
          num_properties: { minimum: 1 maximum: 10 }
          string_value_bytes: { minimum: 1 maximum: 10 }
        }
      )");

  template_config.set_num_operations(kNumberOfOperations);
  template_config.mutable_fill_nodes_config()->set_update(is_update);
  template_config.mutable_fill_nodes_config()->set_specification(
      FillNodesConfig::ARTIFACT);
  config_vector.push_back(template_config);
  template_config.mutable_fill_nodes_config()->set_specification(
      FillNodesConfig::EXECUTION);
  config_vector.push_back(template_config);
  template_config.mutable_fill_nodes_config()->set_specification(
      FillNodesConfig::CONTEXT);
  config_vector.push_back(template_config);

  return config_vector;
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

// Tests the SetUpImpl() for FillNodes insert cases.
// Checks the SetUpImpl() indeed prepares a list of work items whose length is
// the same as the specified number of operations.
TEST_P(FillNodesInsertParameterizedTestFixture, SetUpImplTest) {
  // Inserts some types into db so that nodes can be inserted later.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, store_.get()));
  TF_ASSERT_OK(fill_nodes_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_nodes_->num_operations());
}

// Tests the RunOpImpl() for FillNodes insert cases.
// Checks indeed all the work items have been executed and the number of the
// nodes inside db is the same as the number of operations specified in the
// workload.
TEST_P(FillNodesInsertParameterizedTestFixture, InsertTest) {
  // Inserts some types into db so that nodes can be inserted later.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, store_.get()));
  TF_ASSERT_OK(fill_nodes_->SetUp(store_.get()));
  for (int64 i = 0; i < fill_nodes_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_nodes_->RunOp(i, store_.get(), op_stats));
  }
  // Gets all the existing current nodes inside db after the insert.
  std::vector<NodeType> existing_nodes;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config().specification(),
                                store_.get(), existing_nodes));
  EXPECT_EQ(GetParam().num_operations(), existing_nodes.size());
}

INSTANTIATE_TEST_CASE_P(
    FillNodesInsertTest, FillNodesInsertParameterizedTestFixture,
    ::testing::ValuesIn(EnumerateConfigs(/*is_update=*/false)));

}  // namespace
}  // namespace ml_metadata
