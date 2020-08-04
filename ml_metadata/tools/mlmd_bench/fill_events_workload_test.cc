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
constexpr int kNumberOfExistedOutputEventsInDb = 30;
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

tensorflow::Status GetNumOfEventsByExecution(MetadataStore& store,
                                             int64& num_events) {
  // Gets all the existing executions inside db.
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

tensorflow::Status InsertEventsInDb(const int64 num_input_events,
                                    const int64 num_output_events,
                                    MetadataStore& store) {
  if (num_output_events > kNumberOfExistedNodesInDb) {
    return tensorflow::errors::FailedPrecondition(
        "Cannot insert so many output events due to limited number of existing "
        "artifacts!");
  }

  GetArtifactsResponse get_artifacts_response;
  TF_RETURN_IF_ERROR(store.GetArtifacts(
      /*request=*/{}, &get_artifacts_response));
  GetExecutionsResponse get_executions_response;
  TF_RETURN_IF_ERROR(store.GetExecutions(
      /*request=*/{}, &get_executions_response));

  PutEventsRequest put_request;
  int64 populated_input_events = 0;
  int64 populated_output_events = 0;

  for (int64 i = 0; i < get_artifacts_response.artifacts_size() &&
                    populated_input_events < num_input_events;
       ++i) {
    for (int64 j = 0; j < get_executions_response.executions_size() &&
                      populated_input_events < num_input_events;
         ++j) {
      Event* event = put_request.add_events();
      event->set_type(Event::INPUT);
      event->set_artifact_id(get_artifacts_response.artifacts(i).id());
      event->set_execution_id(get_executions_response.executions(j).id());
      populated_input_events++;
    }
  }

  for (int64 i = 0; i < num_output_events; ++i) {
    Event* event = put_request.add_events();
    event->set_type(Event::OUTPUT);
    event->set_artifact_id(get_artifacts_response.artifacts(i).id());
    event->set_execution_id(
        get_executions_response
            .executions(i % get_executions_response.executions_size())
            .id());
    populated_output_events++;
  }

  PutEventsResponse response;
  return store.PutEvents(put_request, &response);
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

TEST_P(FillEventsParameterizedTestFixture, SetUpImplWhenNoEventsExistTest) {
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

TEST_P(FillEventsParameterizedTestFixture, InsertWhenNoEventsExistTest) {
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
  TF_ASSERT_OK(GetNumOfEventsByExecution(*store_, num_events));
  EXPECT_EQ(GetParam().num_operations() * kNumberOfEventsPerRequest,
            num_events);
}

TEST_P(FillEventsParameterizedTestFixture, SetUpImplWhenSomeEventsExistTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
      /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
      /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));
  TF_ASSERT_OK(InsertEventsInDb(
      /*num_input_events=*/kNumberOfExistedInputEventsInDb,
      /*num_output_events=*/kNumberOfExistedOutputEventsInDb, *store_));

  TF_ASSERT_OK(fill_events_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_events_->num_operations());
}

TEST_P(FillEventsParameterizedTestFixture, InsertWhenSomeEventsExistTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
      /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
      /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));
  TF_ASSERT_OK(InsertEventsInDb(
      /*num_input_events=*/kNumberOfExistedInputEventsInDb,
      /*num_output_events=*/kNumberOfExistedOutputEventsInDb, *store_));

  int64 num_events_before;
  TF_ASSERT_OK(GetNumOfEventsByExecution(*store_, num_events_before));

  TF_ASSERT_OK(fill_events_->SetUp(store_.get()));
  for (int64 i = 0; i < fill_events_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_events_->RunOp(i, store_.get(), op_stats));
  }

  int64 num_events_after;
  TF_ASSERT_OK(GetNumOfEventsByExecution(*store_, num_events_after));

  EXPECT_EQ(GetParam().num_operations() * kNumberOfEventsPerRequest,
            num_events_after - num_events_before);
}

INSTANTIATE_TEST_CASE_P(FillEventsTest, FillEventsParameterizedTestFixture,
                        ::testing::ValuesIn(EnumerateConfigs()));

}  // namespace
}  // namespace ml_metadata
