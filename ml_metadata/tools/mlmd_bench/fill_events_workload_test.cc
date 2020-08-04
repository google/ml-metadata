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
#include "ml_metadata/tools/mlmd_bench/fill_events_workload.h"

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

constexpr int kNumberOfOperations = 50;
constexpr int kNumberOfExistedTypesInDb = 100;
constexpr int kNumberOfExistedNodesInDb = 300;
constexpr int kNumberOfExistedInputEventsInDb = 100;
constexpr int kNumberOfExistedOutputEventsInDb = 100;
constexpr int kNumberOfEventsPerRequest = 5;

// Enumerates the workload configurations as the test parameters that ensure
// test coverage.
std::vector<WorkloadConfig> EnumerateConfigs() {
  std::vector<WorkloadConfig> configs;
  WorkloadConfig config;
  config.set_num_operations(kNumberOfOperations);
  config.mutable_fill_events_config()->mutable_num_events()->set_minimum(
      kNumberOfEventsPerRequest);
  config.mutable_fill_events_config()->mutable_num_events()->set_maximum(
      kNumberOfEventsPerRequest);
  config.mutable_fill_events_config()->set_specification(
      FillEventsConfig::INPUT);
  configs.push_back(config);
  config.mutable_fill_events_config()->set_specification(
      FillEventsConfig::OUTPUT);
  configs.push_back(config);

  return configs;
}

tensorflow::Status GetNumOfEventsByExecution(MetadataStore* store,
                                             int64& num_events) {
  // Gets all the existing executions inside db.
  std::vector<Node> existing_executions;
  FillNodesConfig config;
  config.set_specification(FillNodesConfig::EXECUTION);
  TF_RETURN_IF_ERROR(GetExistingNodes(config, *store, existing_executions));
  GetEventsByExecutionIDsRequest request;
  for (auto& execution : existing_executions) {
    request.add_execution_ids(absl::get<Execution>(execution).id());
  }
  GetEventsByExecutionIDsResponse response;
  TF_RETURN_IF_ERROR(store->GetEventsByExecutionIDs(request, &response));
  num_events = response.events_size();
  return tensorflow::Status::OK();
}

class FillEventsParameterizedTestFixture
    : public ::testing::TestWithParam<WorkloadConfig> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    fill_events_ = absl::make_unique<FillEvents>(FillEvents(
        GetParam().fill_events_config(), GetParam().num_operations()));
  }

  std::unique_ptr<FillEvents> fill_events_;
  std::unique_ptr<MetadataStore> store_;
};

TEST_P(FillEventsParameterizedTestFixture, SetUpImplTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
      /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
      /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));

  TF_ASSERT_OK(fill_events_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_events_->num_operations());
}

TEST_P(FillEventsParameterizedTestFixture, InsertTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
      /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
      /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));

  TF_ASSERT_OK(fill_events_->SetUp(store_.get()));
  for (int64 i = 0; i < fill_events_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_events_->RunOp(i, store_.get(), op_stats));
  }

  int64 num_events;
  TF_ASSERT_OK(GetNumOfEventsByExecution(store_.get(), num_events));
  EXPECT_EQ(GetParam().num_operations() * kNumberOfEventsPerRequest,
            num_events);
}

INSTANTIATE_TEST_CASE_P(FillEventsTest, FillEventsParameterizedTestFixture,
                        ::testing::ValuesIn(EnumerateConfigs()));

}  // namespace
}  // namespace ml_metadata
