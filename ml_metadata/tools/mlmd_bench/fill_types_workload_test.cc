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
#include "ml_metadata/tools/mlmd_bench/util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

constexpr int kNumberOfOperations = 100;
constexpr int kNumberOfExisitedTypeInInsert = 80;
constexpr int kNumberOfExistedTypesButNotEnoughInUpdate = 50;
constexpr int kNumberOfExistedTypesEnoughInUpdate = 200;

// Enumerates the workload configurations as the test parameters that ensure
// test coverage.
std::vector<WorkloadConfig> EnumerateConfigs(const bool is_update) {
  std::vector<WorkloadConfig> config_vector;
  WorkloadConfig template_config = testing::ParseTextProtoOrDie<WorkloadConfig>(
      R"(
        fill_types_config: { num_properties { minimum: 1 maximum: 10 } }
      )");

  template_config.set_num_operations(kNumberOfOperations);
  template_config.mutable_fill_types_config()->set_update(is_update);
  template_config.mutable_fill_types_config()->set_specification(
      FillTypesConfig::ARTIFACT_TYPE);
  config_vector.push_back(template_config);
  template_config.mutable_fill_types_config()->set_specification(
      FillTypesConfig::EXECUTION_TYPE);
  config_vector.push_back(template_config);
  template_config.mutable_fill_types_config()->set_specification(
      FillTypesConfig::CONTEXT_TYPE);
  config_vector.push_back(template_config);

  return config_vector;
}

// Inserts some types into db to set it up in different start status in
// testing.
tensorflow::Status InsertTypesInDb(const int64 num_artifact_types,
                                   const int64 num_execution_types,
                                   const int64 num_context_types,
                                   MetadataStore* store) {
  PutTypesRequest put_request;
  PutTypesResponse put_response;

  for (int64 i = 0; i < num_artifact_types; i++) {
    ArtifactType* curr_type = put_request.add_artifact_types();
    curr_type->set_name(absl::StrCat("pre_insert_artifact_type-", i));
    (*curr_type->mutable_properties())["property"] = STRING;
  }

  for (int64 i = 0; i < num_execution_types; i++) {
    ExecutionType* curr_type = put_request.add_execution_types();
    curr_type->set_name(absl::StrCat("pre_insert_execution_type-", i));
    (*curr_type->mutable_properties())["property"] = STRING;
  }

  for (int64 i = 0; i < num_context_types; i++) {
    ContextType* curr_type = put_request.add_context_types();
    curr_type->set_name(absl::StrCat("pre_insert_context_type-", i));
    (*curr_type->mutable_properties())["property"] = STRING;
  }

  TF_RETURN_IF_ERROR(store->PutTypes(put_request, &put_response));
  return tensorflow::Status::OK();
}

// Executes the given FillTypes workloads.
tensorflow::Status ExecuteWorkload(MetadataStore* store,
                                   std::unique_ptr<FillTypes>& workload) {
  for (int64 i = 0; i < workload->num_operations(); ++i) {
    OpStats op_stats;
    TF_RETURN_IF_ERROR(workload->RunOp(i, store, op_stats));
  }
  return tensorflow::Status::OK();
}

// Checks types update status. If the updates are working properly, the type id
// should remain the same after the updates. On the other hand, the properties
// size for each type should be greater than before since some new fields have
// been added in the update process.
template <typename T>
bool CheckTypeUpdateStatus(const Type& type_before, const Type& type_after) {
  if (absl::get<T>(type_before).id() != absl::get<T>(type_after).id()) {
    return false;
  }
  if (absl::get<T>(type_before).properties().size() >=
      absl::get<T>(type_after).properties().size()) {
    return false;
  }
  return true;
}

// Checks the first `num_operations` types in `types_before` and `types_after`
// for their update status.
void CheckUpdates(const FillTypesConfig& fill_types_config,
                  const std::vector<Type>& types_before,
                  const std::vector<Type>& types_after,
                  const int64 num_operations) {
  for (int64 i = 0; i < num_operations; ++i) {
    switch (fill_types_config.specification()) {
      case FillTypesConfig::ARTIFACT_TYPE: {
        EXPECT_TRUE(CheckTypeUpdateStatus<ArtifactType>(types_before[i],
                                                        types_after[i]));
        break;
      }
      case FillTypesConfig::EXECUTION_TYPE: {
        EXPECT_TRUE(CheckTypeUpdateStatus<ExecutionType>(types_before[i],
                                                         types_after[i]));
        break;
      }
      case FillTypesConfig::CONTEXT_TYPE: {
        EXPECT_TRUE(CheckTypeUpdateStatus<ContextType>(types_before[i],
                                                       types_after[i]));
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for FillTypes configuration input "
                      "in testing !";
    }
  }
}

// Test fixture that uses the same data configuration for multiple following
// parameterized FillTypes insert tests.
// The parameter here is the specific Workload configuration that contains
// the FillTypes insert configuration and the number of operations.
class FillTypesInsertParameterizedTestFixture
    : public ::testing::TestWithParam<WorkloadConfig> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    fill_types_ = absl::make_unique<FillTypes>(
        FillTypes(GetParam().fill_types_config(), GetParam().num_operations()));
  }

  std::unique_ptr<FillTypes> fill_types_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for FillTypes insert cases when the db is empty.
// Checks the SetUpImpl() indeed prepares a list of work items whose length is
// the same as the specified number of operations.
TEST_P(FillTypesInsertParameterizedTestFixture, SetUpImplWhenDbIsEmptyTest) {
  TF_ASSERT_OK(fill_types_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_types_->num_operations());
}

// Tests the RunOpImpl() for FillTypes insert cases when the db is empty.
// Checks indeed all the work items have been executed and the number of the
// types inside db is the same as the number of operations specified in the
// workload.
TEST_P(FillTypesInsertParameterizedTestFixture, InsertWhenDbIsEmptyTest) {
  TF_ASSERT_OK(fill_types_->SetUp(store_.get()));
  TF_ASSERT_OK(ExecuteWorkload(store_.get(), fill_types_));
  // Gets all the existing current types inside db after the insert.
  std::vector<Type> existing_types;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types));
  EXPECT_EQ(GetParam().num_operations(), existing_types.size());
}

// Tests the SetUpImpl() for FillTypes insert cases when the db contains some
// types. Checks the SetUpImpl() indeed prepares a list of work items whose
// length is the same as the specified number of operations.
TEST_P(FillTypesInsertParameterizedTestFixture,
       SetUpImplWhenDbContainsTypesTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExisitedTypeInInsert,
      /*num_execution_types=*/kNumberOfExisitedTypeInInsert,
      /*num_context_types=*/kNumberOfExisitedTypeInInsert, store_.get()));
  TF_ASSERT_OK(fill_types_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_types_->num_operations());
}

// Tests the RunOpImpl() for FillTypes insert cases when the db contains some
// types. Checks indeed all the work items have been executed and the number of
// the new added types inside db is the same as the number of operations
// specified in the workload.
TEST_P(FillTypesInsertParameterizedTestFixture, InsertWhenDbContainsTypesTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExisitedTypeInInsert,
      /*num_execution_types=*/kNumberOfExisitedTypeInInsert,
      /*num_context_types=*/kNumberOfExisitedTypeInInsert, store_.get()));
  // Gets the number of types before insert for later comparison.
  std::vector<Type> existing_types_before_insert;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types_before_insert));

  TF_ASSERT_OK(fill_types_->SetUp(store_.get()));
  TF_ASSERT_OK(ExecuteWorkload(store_.get(), fill_types_));
  // Gets the number of types after insert for later comparison.
  std::vector<Type> existing_types_after_insert;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types_after_insert));

  EXPECT_EQ(
      GetParam().num_operations(),
      existing_types_after_insert.size() - existing_types_before_insert.size());
}

// Test fixture that uses the same data configuration for multiple following
// parameterized FillTypes update tests.
// The parameter here is the specific Workload configuration that contains
// the FillTypes update configuration and the number of operations.
class FillTypesUpdateParameterizedTestFixture
    : public ::testing::TestWithParam<WorkloadConfig> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    fill_types_update_ = absl::make_unique<FillTypes>(
        FillTypes(GetParam().fill_types_config(), GetParam().num_operations()));
  }

  std::unique_ptr<FillTypes> fill_types_update_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for FillTypes update cases when db is empty in the
// first place. Then, the SetUpImpl() will make up and insert all the new types
// into db for later update. The number of types in db after SetUpImpl() should
// be the same as the specified number of update operations. Also, checks the
// SetUpImpl() indeed prepares a list of work items whose length is the same as
// the specified number of operations.
TEST_P(FillTypesUpdateParameterizedTestFixture, SetUpImplWhenDbIsEmptyTest) {
  TF_ASSERT_OK(fill_types_update_->SetUp(store_.get()));
  // Gets the number of types inside db after SetUpImpl().
  std::vector<Type> existing_types;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types));
  EXPECT_EQ(existing_types.size(), GetParam().num_operations());
  EXPECT_EQ(GetParam().num_operations(), fill_types_update_->num_operations());
}

// Tests the RunOpImpl() for FillTypes update cases when db is empty in the
// first place.
// Checks indeed all the work items have been executed and there are certain
// number of existed types inside db have been updated.
TEST_P(FillTypesUpdateParameterizedTestFixture, UpdateWhenDbIsEmptyTest) {
  TF_ASSERT_OK(fill_types_update_->SetUp(store_.get()));
  // Gets the existing types in db after set up but before update for later
  // comparison. The `existing_types_before_update` will contains the types that
  // were made up and inserted in the SetUpImpl().
  std::vector<Type> existing_types_before_update;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types_before_update));

  // Executes the update workloads.
  TF_ASSERT_OK(ExecuteWorkload(store_.get(), fill_types_update_));

  // Gets the existing types in db after update for later comparison.
  std::vector<Type> existing_types_after_update;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types_after_update));

  // The update should update all the made up and inserted types in the
  // SetUpImpl() and not introduce any new types in the db.
  ASSERT_EQ(existing_types_before_update.size(),
            existing_types_after_update.size());

  // For all the types inside db, they should be updated. Checks their update
  // status.
  CheckUpdates(GetParam().fill_types_config(), existing_types_before_update,
               existing_types_after_update,
               (int64)existing_types_after_update.size());
}

// Tests the SetUpImpl() for FillTypes update cases when db contains not enough
// types for update. Then, the SetUpImpl() will make up and insert some new
// types into db for later update. The number of types in db after SetUpImpl()
// should be the same as the specified number of update operations. Also, checks
// the SetUpImpl() indeed prepares a list of work items whose length is the same
// as the specified number of operations.
TEST_P(FillTypesUpdateParameterizedTestFixture,
       SetUpImplWhenDbContainsNotEnoughTypesTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesButNotEnoughInUpdate,
      /*num_execution_types=*/kNumberOfExistedTypesButNotEnoughInUpdate,
      /*num_context_types=*/kNumberOfExistedTypesButNotEnoughInUpdate,
      store_.get()));
  TF_ASSERT_OK(fill_types_update_->SetUp(store_.get()));
  // Gets the number of types inside db after SetUpImpl().
  std::vector<Type> existing_types;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types));
  EXPECT_EQ(existing_types.size(), fill_types_update_->num_operations());
  EXPECT_EQ(GetParam().num_operations(), fill_types_update_->num_operations());
}

// Tests the RunOpImpl() for FillTypes update cases when db contains not enough
// types for update.
// Checks indeed all the work items have been executed and there are certain
// number of existed types inside db have been updated.
TEST_P(FillTypesUpdateParameterizedTestFixture,
       UpdateWhenDbContainsNotEnoughTypesTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesButNotEnoughInUpdate,
      /*num_execution_types=*/kNumberOfExistedTypesButNotEnoughInUpdate,
      /*num_context_types=*/kNumberOfExistedTypesButNotEnoughInUpdate,
      store_.get()));
  TF_ASSERT_OK(fill_types_update_->SetUp(store_.get()));
  // Gets the existing types in db after set up but before update for later
  // comparison. The `existing_types_before_update` will contains the existed
  // types and some new types that were made up and inserted in the SetUpImpl().
  std::vector<Type> existing_types_before_update;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types_before_update));

  // Executes the update workloads.
  TF_ASSERT_OK(ExecuteWorkload(store_.get(), fill_types_update_));

  // Gets the existing types in db after update for later comparison.
  std::vector<Type> existing_types_after_update;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types_after_update));

  // The update should update all the types that includes some old existed
  // types and some new made up and inserted types and not introduce any new
  // types in the db.
  ASSERT_EQ(existing_types_before_update.size(),
            existing_types_after_update.size());

  // For all the types inside db, they should be updated. Checks their update
  // status.
  CheckUpdates(GetParam().fill_types_config(), existing_types_before_update,
               existing_types_after_update,
               (int64)existing_types_after_update.size());
}

// Tests the SetUpImpl() for FillTypes update cases when db contains enough
// types for update. Then, the SetUpImpl() will not  make up and insert some new
// types into db for later update. The number of types in db should remain
// the same before and after the SetUpImpl(). Also, checks the SetUpImpl()
// indeed prepares a list of work items whose length is the same as the
// specified number of operations.
TEST_P(FillTypesUpdateParameterizedTestFixture,
       SetUpImplWhenDbContainsEnoughTypesTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesEnoughInUpdate,
      /*num_execution_types=*/kNumberOfExistedTypesEnoughInUpdate,
      /*num_context_types=*/kNumberOfExistedTypesEnoughInUpdate, store_.get()));
  // Gets the number of types inside db before SetUpImpl().
  std::vector<Type> existing_types_before_setup;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types_before_setup));
  TF_ASSERT_OK(fill_types_update_->SetUp(store_.get()));
  // Gets the number of types inside db after SetUpImpl().
  std::vector<Type> existing_types_after_setup;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types_after_setup));
  EXPECT_EQ(existing_types_before_setup.size(),
            existing_types_after_setup.size());
  EXPECT_EQ(GetParam().num_operations(), fill_types_update_->num_operations());
}

// Tests the RunOpImpl() for FillTypes update cases when db contains enough
// types for update. Checks indeed all the work items have been executed and
// there are certain number of existed types inside db have been updated.
TEST_P(FillTypesUpdateParameterizedTestFixture,
       UpdateWhenDbContainsEnoughTypesTest) {
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesEnoughInUpdate,
      /*num_execution_types=*/kNumberOfExistedTypesEnoughInUpdate,
      /*num_context_types=*/kNumberOfExistedTypesEnoughInUpdate, store_.get()));
  TF_ASSERT_OK(fill_types_update_->SetUp(store_.get()));
  // Gets the existing types in db after set up but before update for later
  // comparison. The `existing_types_before_update` will contains the existed
  // types in db.
  std::vector<Type> existing_types_before_update;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types_before_update));

  // Executes the update workloads.
  TF_ASSERT_OK(ExecuteWorkload(store_.get(), fill_types_update_));

  // Gets the existing types in db after update for later comparison.
  std::vector<Type> existing_types_after_update;
  TF_ASSERT_OK(GetExistingTypes(GetParam().fill_types_config(), store_.get(),
                                existing_types_after_update));

  // The update should update the first number of update operations existed
  // types inside db and not introduce any new types in the db.
  ASSERT_EQ(existing_types_before_update.size(),
            existing_types_after_update.size());

  // For the first number of update operations existed types inside db, they
  // should be updated. Checks their update status.
  CheckUpdates(GetParam().fill_types_config(), existing_types_before_update,
               existing_types_after_update,
               fill_types_update_->num_operations());
}

INSTANTIATE_TEST_CASE_P(
    FillTypesInsertTest, FillTypesInsertParameterizedTestFixture,
    ::testing::ValuesIn(EnumerateConfigs(/*is_update=*/false)));

INSTANTIATE_TEST_CASE_P(
    FillTypesUpdateTest, FillTypesUpdateParameterizedTestFixture,
    ::testing::ValuesIn(EnumerateConfigs(/*is_update=*/true)));

}  // namespace
}  // namespace ml_metadata
