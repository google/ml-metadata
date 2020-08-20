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
#include "ml_metadata/tools/mlmd_bench/read_types_workload.h"

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
constexpr int kNumberOfExistedTypesInDb = 300;

constexpr char kConfig[] =
    "read_types_config: { maybe_num_ids { minimum: 1 maximum: 10 } }";

// Enumerates the workload configurations as the test parameters that ensure
// test coverage.
std::vector<WorkloadConfig> EnumerateConfigs() {
  std::vector<WorkloadConfig> configs;
  std::vector<ReadTypesConfig::Specification> specifications = {
      ReadTypesConfig::ALL_ARTIFACT_TYPES,
      ReadTypesConfig::ALL_EXECUTION_TYPES,
      ReadTypesConfig::ALL_CONTEXT_TYPES,
      ReadTypesConfig::ARTIFACT_TYPES_BY_ID,
      ReadTypesConfig::EXECUTION_TYPES_BY_ID,
      ReadTypesConfig::CONTEXT_TYPES_BY_ID,
      ReadTypesConfig::ARTIFACT_TYPE_BY_NAME,
      ReadTypesConfig::EXECUTION_TYPE_BY_NAME,
      ReadTypesConfig::CONTEXT_TYPE_BY_NAME};

  for (const ReadTypesConfig::Specification& specification : specifications) {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(kConfig);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_read_types_config()->set_specification(specification);
    configs.push_back(config);
  }

  return configs;
}

// Test fixture that uses the same data configuration for multiple following
// parameterized ReadTypes tests.
// The parameter here is the specific Workload configuration that contains
// the ReadTypes configuration and the number of operations.
class ReadTypesParameterizedTestFixture
    : public ::testing::TestWithParam<WorkloadConfig> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    read_types_ = absl::make_unique<ReadTypes>(
        ReadTypes(GetParam().read_types_config(), GetParam().num_operations()));
    TF_ASSERT_OK(InsertTypesInDb(
        /*num_artifact_types=*/kNumberOfExistedTypesInDb,
        /*num_execution_types=*/kNumberOfExistedTypesInDb,
        /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  }

  std::unique_ptr<ReadTypes> read_types_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for ReadTypes. Checks the SetUpImpl() indeed prepares
// a list of work items whose length is the same as the specified number of
// operations.
TEST_P(ReadTypesParameterizedTestFixture, SetUpImplTest) {
  TF_ASSERT_OK(read_types_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), read_types_->num_operations());
}

// Tests the RunOpImpl() for ReadTypes. Checks indeed all the work items have
// been executed and some bytes are transferred during the reading process.
TEST_P(ReadTypesParameterizedTestFixture, RunOpImplTest) {
  TF_ASSERT_OK(read_types_->SetUp(store_.get()));

  int64 total_done = 0;
  ThreadStats stats;
  stats.Start();
  for (int64 i = 0; i < read_types_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(read_types_->RunOp(i, store_.get(), op_stats));
    stats.Update(op_stats, total_done);
  }
  stats.Stop();
  EXPECT_EQ(GetParam().num_operations(), stats.done());
  // Checks that the transferred bytes is greater that 0(the reading process
  // indeed occurred).
  EXPECT_LT(0, stats.bytes());
}

INSTANTIATE_TEST_CASE_P(ReadTypesTest, ReadTypesParameterizedTestFixture,
                        ::testing::ValuesIn(EnumerateConfigs()));

}  // namespace
}  // namespace ml_metadata
