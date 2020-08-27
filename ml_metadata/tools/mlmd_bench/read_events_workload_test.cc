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
constexpr int kNumberOfExistedEventsInDb = 1000;
constexpr char kConfig[] = R"(
    read_events_config: { num_ids: { minimum: 1 maximum: 10 } })";

// Enumerates the workload configurations as the test parameters that ensure
// test coverage.
std::vector<WorkloadConfig> EnumerateConfigs() {
  std::vector<WorkloadConfig> configs;

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(kConfig);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_read_events_config()->set_specification(
        ReadEventsConfig::EVENTS_BY_ARTIFACT_ID);
    configs.push_back(config);
  }

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(kConfig);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_read_events_config()->set_specification(
        ReadEventsConfig::EVENTS_BY_EXECUTION_ID);
    configs.push_back(config);
  }

  return configs;
}

// Inserts some Input / Output Events into db so that we can read events from db
// later. Returns detailed error if query executions failed.
tensorflow::Status InsertEventsInDb(const int64 num_input_events,
                                    const int64 num_output_events,
                                    MetadataStore& store) {
  PutEventsRequest request;
  std::minstd_rand0 gen(0);
  FillEventsConfig fill_events_config;
  std::vector<Node> existing_artifact_nodes;
  std::vector<Node> existing_execution_nodes;
  TF_RETURN_IF_ERROR(GetExistingNodes(fill_events_config, store,
                                      existing_artifact_nodes,
                                      existing_execution_nodes));
  std::uniform_int_distribution<int64> artifact_node_index_dist{
      0, (int64)(existing_artifact_nodes.size() - 1)};
  std::uniform_int_distribution<int64> execution_node_index_dist{
      0, (int64)(existing_execution_nodes.size() - 1)};

  for (int i = 0; i < kNumberOfExistedEventsInDb; ++i) {
    // Inserts Input Events.
    Event* input_event = request.add_events();
    input_event->set_type(Event::INPUT);
    input_event->set_artifact_id(
        absl::get<Artifact>(
            existing_artifact_nodes[artifact_node_index_dist(gen)])
            .id());
    input_event->set_execution_id(
        absl::get<Execution>(
            existing_execution_nodes[execution_node_index_dist(gen)])
            .id());
    // Inserts Output Events.
    Event* output_event = request.add_events();
    output_event->set_type(Event::OUTPUT);
    output_event->set_artifact_id(
        absl::get<Artifact>(
            existing_artifact_nodes[artifact_node_index_dist(gen)])
            .id());
    output_event->set_execution_id(
        absl::get<Execution>(
            existing_execution_nodes[execution_node_index_dist(gen)])
            .id());
  }

  PutEventsResponse response;
  return store.PutEvents(request, &response);
}

// Test fixture that uses the same data configuration for multiple following
// parameterized ReadEvents tests.
// The parameter here is the specific Workload configuration that contains
// the ReadEvents configuration and the number of operations.
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
        /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
        /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
        /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));
    TF_ASSERT_OK(InsertEventsInDb(
        /*num_input_events=*/kNumberOfExistedEventsInDb,
        /*num_output_events=*/kNumberOfExistedEventsInDb, *store_));
  }

  std::unique_ptr<ReadEvents> read_events_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for ReadEvents. Checks the SetUpImpl() indeed prepares
// a list of work items whose length is the same as the specified number of
// operations.
TEST_P(ReadEventsParameterizedTestFixture, SetUpImplTest) {
  TF_ASSERT_OK(read_events_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), read_events_->num_operations());
}

// Tests the RunOpImpl() for ReadEvents. Checks indeed all the work items have
// been executed and some bytes are transferred during the reading process.
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
                        ValuesIn(EnumerateConfigs()));

}  // namespace
}  // namespace ml_metadata
