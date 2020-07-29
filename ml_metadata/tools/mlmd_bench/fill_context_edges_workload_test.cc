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

// Enumerates the workload configurations as the test parameters that ensure
// test coverage.
std::vector<WorkloadConfig> EnumerateConfigs() {
  std::vector<WorkloadConfig> config_vector;
  WorkloadConfig template_config;

  template_config.set_num_operations(kNumberOfOperations);

  template_config.mutable_fill_context_edges_config()->set_specification(
      FillContextEdgesConfig::ATTRIBUTION);
  config_vector.push_back(template_config);
  template_config.mutable_fill_context_edges_config()->set_specification(
      FillContextEdgesConfig::ASSOCIATION);
  config_vector.push_back(template_config);

  return config_vector;
}

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

TEST_P(FillContextEdgesParameterizedTestFixture, SetUpImplTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, store_.get()));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
      /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
      /*num_context_nodes=*/kNumberOfExistedNodesInDb, store_.get()));

  TF_ASSERT_OK(fill_context_edges_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_context_edges_->num_operations());
}

TEST_P(FillContextEdgesParameterizedTestFixture, InsertTest) {}

INSTANTIATE_TEST_CASE_P(FillContextEdgesTest,
                        FillContextEdgesParameterizedTestFixture,
                        ::testing::ValuesIn(EnumerateConfigs()));

}  // namespace
}  // namespace ml_metadata
