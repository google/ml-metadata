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
#include "ml_metadata/tools/mlmd_bench/fill_types_workload.h"

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
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

constexpr int64 kNumOperations = 100;
constexpr int64 kNumOperationsForMakeUpCases = 200;

// Generates FillTypes Workload where `update` indicates insert or update cases.
void GenerateFillTypesWorkload(
    const bool update, const int64 num_operations,
    std::vector<std::unique_ptr<FillTypes>>& fill_types) {
  FillTypesConfig fill_artifact_types_config =
      testing::ParseTextProtoOrDie<FillTypesConfig>(
          R"(
            specification: ARTIFACT_TYPE
            num_properties { minimum: 1 maximum: 10 }
          )");
  fill_artifact_types_config.set_update(update);
  std::unique_ptr<FillTypes> fill_artifact_types = absl::make_unique<FillTypes>(
      FillTypes(fill_artifact_types_config, num_operations));

  FillTypesConfig fill_execution_types_config =
      testing::ParseTextProtoOrDie<FillTypesConfig>(
          R"(
            specification: EXECUTION_TYPE
            num_properties { minimum: 1 maximum: 10 }
          )");
  fill_execution_types_config.set_update(update);
  std::unique_ptr<FillTypes> fill_execution_types =
      absl::make_unique<FillTypes>(
          FillTypes(fill_execution_types_config, num_operations));

  FillTypesConfig fill_context_types_config =
      testing::ParseTextProtoOrDie<FillTypesConfig>(
          R"(
            specification: CONTEXT_TYPE
            num_properties { minimum: 1 maximum: 10 }
          )");
  fill_context_types_config.set_update(update);
  std::unique_ptr<FillTypes> fill_context_types = absl::make_unique<FillTypes>(
      FillTypes(fill_context_types_config, num_operations));

  fill_types.push_back(std::move(fill_artifact_types));
  fill_types.push_back(std::move(fill_execution_types));
  fill_types.push_back(std::move(fill_context_types));
}

// Checks types update status. If the updates are working properly, the type id
// should remain the same even after the updates. On the other hand, the
// properties size for each type should be greater than before since some new
// fields have been added in the update process.
template <typename Type>
bool CheckTypesUpdateStatus(const Type& type_before, const Type& type_after) {
  if (type_before.id() != type_after.id()) {
    return false;
  }
  if (type_before.properties().size() >= type_after.properties().size()) {
    return false;
  }
  return true;
}

// Updates the get types response with the current store instance.
tensorflow::Status UpdateGetResponse(
    MetadataStore* store, GetArtifactTypesResponse& get_artifact_types_response,
    GetExecutionTypesResponse& get_execution_types_response,
    GetContextTypesResponse& get_context_types_response) {
  TF_RETURN_IF_ERROR(
      store->GetArtifactTypes(/*request=*/{}, &get_artifact_types_response));
  TF_RETURN_IF_ERROR(
      store->GetExecutionTypes(/*request=*/{}, &get_execution_types_response));
  TF_RETURN_IF_ERROR(
      store->GetContextTypes(/*request=*/{}, &get_context_types_response));
  return tensorflow::Status::OK();
}

// Executes the given FillTypes workloads.
tensorflow::Status SetUpAndExecuteWorkload(
    MetadataStore* store,
    std::vector<std::unique_ptr<FillTypes>>& fill_types_workloads) {
  for (auto& workload : fill_types_workloads) {
    TF_RETURN_IF_ERROR(workload->SetUp(store));
    for (int64 i = 0; i < workload->num_operations(); ++i) {
      OpStats op_stats;
      TF_RETURN_IF_ERROR(workload->RunOp(i, store, op_stats));
    }
  }
  return tensorflow::Status::OK();
}

// Test fixture that uses the same data configuration for multiple following
// FillTypes insert tests.
class FillTypesInsertTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    // `false` indicates the insert cases for FillTypes.
    GenerateFillTypesWorkload(false, kNumOperations, fill_types_insert_);
  }

  std::vector<std::unique_ptr<FillTypes>> fill_types_insert_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for FillTypes insert cases.
// Checks the SetUpImpl() indeed prepares a list of work items whose length is
// the same as the specified number of operations.
TEST_F(FillTypesInsertTest, SetUpImplTest) {
  for (auto& workload : fill_types_insert_) {
    TF_ASSERT_OK(workload->SetUp(store_.get()));
    ASSERT_EQ(kNumOperations, workload->num_operations());
  }
}

// Tests the RunOpImpl() for insert types.
TEST_F(FillTypesInsertTest, InsertTest) {
  TF_ASSERT_OK(SetUpAndExecuteWorkload(store_.get(), fill_types_insert_));

  GetArtifactTypesResponse get_artifact_types_response;
  GetExecutionTypesResponse get_execution_types_response;
  GetContextTypesResponse get_context_types_response;
  TF_ASSERT_OK(UpdateGetResponse(store_.get(), get_artifact_types_response,
                                 get_execution_types_response,
                                 get_context_types_response));

  // Checks indeed all the work items have been executed and all the types have
  // been inserted into the db. The number of specific types inside db is the
  // same as the number of insert operations.
  EXPECT_EQ(get_artifact_types_response.artifact_types_size(), kNumOperations);
  EXPECT_EQ(get_execution_types_response.execution_types_size(),
            kNumOperations);
  EXPECT_EQ(get_context_types_response.context_types_size(), kNumOperations);
}

// Test fixture that uses the same data configuration for multiple following
// FillTypes update tests.
class FillTypesUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    // Inserts some types into the db for later update since the test db is an
    // empty one in the beginning.
    GenerateFillTypesWorkload(false, kNumOperations, fill_types_insert_);
    TF_ASSERT_OK(SetUpAndExecuteWorkload(store_.get(), fill_types_insert_));
  }

  std::vector<std::unique_ptr<FillTypes>> fill_types_insert_;
  std::vector<std::unique_ptr<FillTypes>> fill_types_update_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for FillTypes update cases.
// Checks the SetUpImpl() indeed prepares a list of work items whose length is
// the same as the specified number of operations.
TEST_F(FillTypesUpdateTest, SetUpImplTest) {
  // `true` indicates the update cases for FillTypes.
  GenerateFillTypesWorkload(true, kNumOperations, fill_types_update_);
  for (auto& workload : fill_types_update_) {
    TF_ASSERT_OK(workload->SetUp(store_.get()));
    ASSERT_EQ(kNumOperations, workload->num_operations());
  }
}

// Tests the normal update cases(the number of types inside db are enough for
// updating).
TEST_F(FillTypesUpdateTest, NormalUpdateTest) {
  // Gets the get_response_before_update for later comparison.
  GetArtifactTypesResponse get_artifact_types_response_before_update;
  GetExecutionTypesResponse get_execution_types_response_before_update;
  GetContextTypesResponse get_context_types_response_before_update;
  TF_ASSERT_OK(UpdateGetResponse(store_.get(),
                                 get_artifact_types_response_before_update,
                                 get_execution_types_response_before_update,
                                 get_context_types_response_before_update));

  // The update number_opeartions is the same the insert which means the number
  // of types inside db are enough for update.
  GenerateFillTypesWorkload(true, kNumOperations, fill_types_update_);

  // Prepares and executes the update workloads.
  TF_ASSERT_OK(SetUpAndExecuteWorkload(store_.get(), fill_types_update_));

  // Gets the get_response_after_update for later comparison.
  GetArtifactTypesResponse get_artifact_types_response_after_update;
  GetExecutionTypesResponse get_execution_types_response_after_update;
  GetContextTypesResponse get_context_types_response_after_update;
  TF_ASSERT_OK(UpdateGetResponse(store_.get(),
                                 get_artifact_types_response_after_update,
                                 get_execution_types_response_after_update,
                                 get_context_types_response_after_update));

  // Uses CheckTypesUpdateStatus() to check update status.
  for (int64 i = 0; i < kNumOperations; ++i) {
    EXPECT_TRUE(CheckTypesUpdateStatus<ArtifactType>(
        get_artifact_types_response_before_update.artifact_types()[i],
        get_artifact_types_response_after_update.artifact_types()[i]));
    EXPECT_TRUE(CheckTypesUpdateStatus<ExecutionType>(
        get_execution_types_response_before_update.execution_types()[i],
        get_execution_types_response_after_update.execution_types()[i]));
    EXPECT_TRUE(CheckTypesUpdateStatus<ContextType>(
        get_context_types_response_before_update.context_types()[i],
        get_context_types_response_after_update.context_types()[i]));
  }

  // There should no more types inside db after update since the updates only
  // update the existed types inserted before by fill_types_insert_.
  EXPECT_EQ(get_artifact_types_response_after_update.artifact_types_size(),
            kNumOperations);
  EXPECT_EQ(get_execution_types_response_after_update.execution_types_size(),
            kNumOperations);
  EXPECT_EQ(get_context_types_response_after_update.context_types_size(),
            kNumOperations);
}

// Tests the update cases where the number of types inside db are not enough for
// updating so making up and inserting new types in db for update is needed.
TEST_F(FillTypesUpdateTest, MakeUpUpdateTest) {
  // Gets the `get_response_before_update` for later comparison.
  GetArtifactTypesResponse get_artifact_types_response_before_update;
  GetExecutionTypesResponse get_execution_types_response_before_update;
  GetContextTypesResponse get_context_types_response_before_update;
  TF_ASSERT_OK(UpdateGetResponse(store_.get(),
                                 get_artifact_types_response_before_update,
                                 get_execution_types_response_before_update,
                                 get_context_types_response_before_update));

  // The num_operations passed into the `fill_types_update_` is bigger than the
  // existed number of types inside db(the number of inserted types by
  // `fill_types_insert_`). So, the `fill_types_update_` will make up and insert
  // some new types in db and update them together later.
  GenerateFillTypesWorkload(true, kNumOperationsForMakeUpCases,
                            fill_types_update_);

  TF_ASSERT_OK(SetUpAndExecuteWorkload(store_.get(), fill_types_update_));

  // Gets the get_response_after_update for later comparison.
  GetArtifactTypesResponse get_artifact_types_response_after_update;
  GetExecutionTypesResponse get_execution_types_response_after_update;
  GetContextTypesResponse get_context_types_response_after_update;
  TF_ASSERT_OK(UpdateGetResponse(store_.get(),
                                 get_artifact_types_response_after_update,
                                 get_execution_types_response_after_update,
                                 get_context_types_response_after_update));

  // For the types inserted before in `fill_types_insert_` workload, uses
  // CheckTypesUpdateStatus() to check update status.
  for (int64 i = 0; i < kNumOperations; ++i) {
    EXPECT_TRUE(CheckTypesUpdateStatus<ArtifactType>(
        get_artifact_types_response_before_update.artifact_types()[i],
        get_artifact_types_response_after_update.artifact_types()[i]));
    EXPECT_TRUE(CheckTypesUpdateStatus<ExecutionType>(
        get_execution_types_response_before_update.execution_types()[i],
        get_execution_types_response_after_update.execution_types()[i]));
    EXPECT_TRUE(CheckTypesUpdateStatus<ContextType>(
        get_context_types_response_before_update.context_types()[i],
        get_context_types_response_after_update.context_types()[i]));
  }

  // Since the update process makes up new types in db, the current number
  // of types in db should be the same as the num_operations of the
  // fill_type_update(`kNumOperationsForMakeUpCases`).
  EXPECT_EQ(get_artifact_types_response_after_update.artifact_types_size(),
            kNumOperationsForMakeUpCases);
  EXPECT_EQ(get_execution_types_response_after_update.execution_types_size(),
            kNumOperationsForMakeUpCases);
  EXPECT_EQ(get_context_types_response_after_update.context_types_size(),
            kNumOperationsForMakeUpCases);
}

}  // namespace
}  // namespace ml_metadata
