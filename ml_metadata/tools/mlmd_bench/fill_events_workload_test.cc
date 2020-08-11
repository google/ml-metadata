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
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

constexpr int kNumberOfOperations = 50;
constexpr int kNumberOfExistedTypesInDb = 100;
constexpr int kNumberOfExistedNodesInDb = 500;
constexpr int kNumberOfExistedEventsInDb = 50;
constexpr int kNumberOfEventsPerRequest = 3;
constexpr double kConfigurableSkew = 1.0;
constexpr double kDirichletAlpha = 1.0;

// Enumerates the workload configurations as the test parameters that ensure
// test coverage.
std::vector<WorkloadConfig> EnumerateConfigs() {
  std::vector<WorkloadConfig> configs;

  {
    WorkloadConfig config;
    config.set_num_operations(kNumberOfOperations);
    config.mutable_fill_events_config()->mutable_num_events()->set_minimum(
        kNumberOfEventsPerRequest);
    config.mutable_fill_events_config()->mutable_num_events()->set_maximum(
        kNumberOfEventsPerRequest);
    config.mutable_fill_events_config()->set_specification(
        FillEventsConfig::INPUT);
    config.mutable_fill_events_config()
        ->mutable_artifact_node_popularity_zipf()
        ->set_skew(kConfigurableSkew);
    configs.push_back(config);
  }

  {
    WorkloadConfig config;
    config.set_num_operations(kNumberOfOperations);
    config.mutable_fill_events_config()->mutable_num_events()->set_minimum(
        kNumberOfEventsPerRequest);
    config.mutable_fill_events_config()->mutable_num_events()->set_maximum(
        kNumberOfEventsPerRequest);
    config.mutable_fill_events_config()->set_specification(
        FillEventsConfig::OUTPUT);
    config.mutable_fill_events_config()
        ->mutable_artifact_node_popularity_categorical()
        ->set_dirichlet_alpha(kDirichletAlpha);
    configs.push_back(config);
  }

  return configs;
}

// Gets the number of existed events in db. Returns detailed error
// if query executions failed.
tensorflow::Status GetNumOfEventsInDb(MetadataStore& store, int64& num_events) {
  std::vector<Node> existing_executions;
  FillNodesConfig config;
  config.set_specification(FillNodesConfig::EXECUTION);
  TF_RETURN_IF_ERROR(GetExistingNodes(config, store, existing_executions));
  GetEventsByExecutionIDsRequest request;
  for (auto& execution : existing_executions) {
    request.add_execution_ids(absl::get<Execution>(execution).id());
  }
  GetEventsByExecutionIDsResponse response;
  TF_RETURN_IF_ERROR(store.GetEventsByExecutionIDs(request, &response));
  num_events = response.events_size();
  return tensorflow::Status::OK();
}

// Inserts some events into db for setting up. Returns detailed error
// if query executions failed.
tensorflow::Status InsertEventsInDb(const FillEventsConfig& fill_events_config,
                                    MetadataStore& store) {
  // Inserts some events beforehand so that the db contains some events in the
  // beginning.
  std::unique_ptr<FillEvents> prepared_db_workload =
      absl::make_unique<FillEvents>(
          FillEvents(fill_events_config, kNumberOfExistedEventsInDb));
  TF_RETURN_IF_ERROR(prepared_db_workload->SetUp(&store));
  for (int64 i = 0; i < prepared_db_workload->num_operations(); ++i) {
    OpStats op_stats;
    TF_RETURN_IF_ERROR(prepared_db_workload->RunOp(i, &store, op_stats));
  }
  return tensorflow::Status::OK();
}

// Test fixture that uses the same data configuration for multiple following
// parameterized FillEvents tests.
// The parameter here is the specific Workload configuration that contains
// the FillEvents configuration and the number of operations.
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
    TF_ASSERT_OK(InsertTypesInDb(
        /*num_artifact_types=*/kNumberOfExistedTypesInDb,
        /*num_execution_types=*/kNumberOfExistedTypesInDb,
        /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
    TF_ASSERT_OK(InsertNodesInDb(
        /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
        /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
        /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));
  }

  std::unique_ptr<FillEvents> fill_events_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for FillEvents when db contains no events in the
// beginning. Checks the SetUpImpl() indeed prepares a list of work items whose
// length is the same as the specified number of operations.
TEST_P(FillEventsParameterizedTestFixture, SetUpImplWhenNoEventsExistTest) {
  TF_ASSERT_OK(fill_events_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_events_->num_operations());
}

// Tests the RunOpImpl() for FillEvents when db contains no events in the
// beginning. Checks indeed all the work items have been executed and the number
// of the events inside db is the same as the number of operations
// specified in the workload.
TEST_P(FillEventsParameterizedTestFixture, InsertWhenNoEventsExistTest) {
  TF_ASSERT_OK(fill_events_->SetUp(store_.get()));
  for (int64 i = 0; i < fill_events_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_events_->RunOp(i, store_.get(), op_stats));
  }

  int64 num_events;
  TF_ASSERT_OK(GetNumOfEventsInDb(*store_, num_events));
  EXPECT_EQ(GetParam().num_operations() * kNumberOfEventsPerRequest,
            num_events);
}

// Tests the SetUpImpl() for FillEvents when db contains some events in the
// beginning. Checks the SetUpImpl() indeed prepares a list of work items whose
// length is the same as the specified number of operations.
TEST_P(FillEventsParameterizedTestFixture, SetUpImplWhenSomeEventsExistTest) {
  TF_ASSERT_OK(InsertEventsInDb(GetParam().fill_events_config(), *store_));

  TF_ASSERT_OK(fill_events_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_events_->num_operations());
}

// Tests the RunOpImpl() for FillEvents when db contains some events in the
// beginning. Checks indeed all the work items have been executed and the number
// of new added events inside db is the same as the number of operations
// specified in the workload.
TEST_P(FillEventsParameterizedTestFixture, InsertWhenSomeEventsExistTest) {
  TF_ASSERT_OK(InsertEventsInDb(GetParam().fill_events_config(), *store_));

  int64 num_events_before;
  TF_ASSERT_OK(GetNumOfEventsInDb(*store_, num_events_before));

  TF_ASSERT_OK(fill_events_->SetUp(store_.get()));
  for (int64 i = 0; i < fill_events_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_events_->RunOp(i, store_.get(), op_stats));
  }

  int64 num_events_after;
  TF_ASSERT_OK(GetNumOfEventsInDb(*store_, num_events_after));

  EXPECT_EQ(GetParam().num_operations() * kNumberOfEventsPerRequest,
            num_events_after - num_events_before);
}

INSTANTIATE_TEST_CASE_P(FillEventsTest, FillEventsParameterizedTestFixture,
                        ::testing::ValuesIn(EnumerateConfigs()));

}  // namespace
}  // namespace ml_metadata
