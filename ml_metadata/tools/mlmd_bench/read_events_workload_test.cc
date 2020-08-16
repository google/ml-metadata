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
#include "ml_metadata/tools/mlmd_bench/read_events_workload.h"

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/fill_events_workload.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

constexpr int kNumberOfOperations = 100;
constexpr int kNumberOfExistedTypesInDb = 100;
constexpr int kNumberOfExistedNodesInDb = 200;
constexpr int kNumberOfExistedEventsInDb = 100;

constexpr char kConfig[] = R"(
    read_events_config: { num_ids: { minimum: 1 maximum: 10 } })";

constexpr char kInputEventConfig[] = R"(
    specification: INPUT
    execution_node_popularity: { dirichlet_alpha : 1000 } 
    artifact_node_popularity_zipf: { skew : 0 }
    num_events: { minimum: 1 maximum: 1 })";

constexpr char kOutputEventConfig[] = R"(
    specification: OUTPUT
    execution_node_popularity: { dirichlet_alpha : 1000 }
    artifact_node_popularity_categorical: { dirichlet_alpha : 1000 }
    num_events: { minimum: 1 maximum: 1 })";

std::vector<WorkloadConfig> EnumerateConfigs() {
  std::vector<WorkloadConfig> configs;

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(kConfig);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_read_events_config()->set_specification(
        ReadEventsConfig::EVENTS_BY_ARTIFACT_IDS);
    configs.push_back(config);
  }

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(kConfig);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_read_events_config()->set_specification(
        ReadEventsConfig::EVENTS_BY_ARTIFACT_IDS);
    configs.push_back(config);
  }

  return configs;
}

tensorflow::Status InsertEventsInDb(const int64 num_input_events,
                                    const int64 num_output_events,
                                    MetadataStore& store) {
  {
    FillEventsConfig fill_events_config =
        testing::ParseTextProtoOrDie<FillEventsConfig>(kInputEventConfig);
    std::unique_ptr<FillEvents> prepared_db_workload =
        absl::make_unique<FillEvents>(
            FillEvents(fill_events_config, num_input_events));
    TF_RETURN_IF_ERROR(prepared_db_workload->SetUp(&store));
    for (int64 i = 0; i < prepared_db_workload->num_operations(); ++i) {
      OpStats op_stats;
      TF_RETURN_IF_ERROR(prepared_db_workload->RunOp(i, &store, op_stats));
    }
  }

  {
    FillEventsConfig fill_events_config =
        testing::ParseTextProtoOrDie<FillEventsConfig>(kOutputEventConfig);
    std::unique_ptr<FillEvents> prepared_db_workload =
        absl::make_unique<FillEvents>(
            FillEvents(fill_events_config, num_output_events));
    TF_RETURN_IF_ERROR(prepared_db_workload->SetUp(&store));
    for (int64 i = 0; i < prepared_db_workload->num_operations(); ++i) {
      OpStats op_stats;
      TF_RETURN_IF_ERROR(prepared_db_workload->RunOp(i, &store, op_stats));
    }
  }
  return tensorflow::Status::OK();
}

class ReadEventsParameterizedTestFixture
    : public ::testing::TestWithParam<WorkloadConfig> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    read_events_ = absl::make_unique<ReadEvents>(ReadEvents(
        GetParam().read_events_config(), GetParam().num_operations()));
    TF_ASSERT_OK(InsertTypesInDb(
        /*num_artifact_types=*/kNumberOfExistedTypesInDb,
        /*num_execution_types=*/kNumberOfExistedTypesInDb,
        /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
    TF_ASSERT_OK(InsertNodesInDb(
        /*num_artifact_types=*/kNumberOfExistedNodesInDb,
        /*num_execution_types=*/kNumberOfExistedNodesInDb,
        /*num_context_types=*/kNumberOfExistedNodesInDb, *store_));
    TF_ASSERT_OK(InsertEventsInDb(
        /*num_input_events=*/kNumberOfExistedEventsInDb,
        /*num_output_events=*/kNumberOfExistedEventsInDb, *store_));
  }

  std::unique_ptr<ReadEvents> read_events_;
  std::unique_ptr<MetadataStore> store_;
};

TEST_P(ReadEventsParameterizedTestFixture, SetUpImplTest) {
  TF_ASSERT_OK(read_events_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), read_events_->num_operations());
}

TEST_P(ReadEventsParameterizedTestFixture, RunOpImplTest) {
  TF_ASSERT_OK(read_events_->SetUp(store_.get()));

  int64 total_done = 0;
  ThreadStats stats;
  stats.Start();
  for (int64 i = 0; i < read_events_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(read_events_->RunOp(i, store_.get(), op_stats));
    stats.Update(op_stats, total_done);
  }
  stats.Stop();
  EXPECT_EQ(stats.done(), GetParam().num_operations());
  // Checks that the transferred bytes is greater that 0(the reading process
  // indeed occurred).
  EXPECT_GT(stats.bytes(), 0);
}

INSTANTIATE_TEST_CASE_P(ReadEventsTest, ReadEventsParameterizedTestFixture,
                        ::testing::ValuesIn(EnumerateConfigs()));

}  // namespace
}  // namespace ml_metadata
