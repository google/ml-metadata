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
constexpr int kNumberOfExistedNodesInDb = 100;
constexpr int kNumberOfExistedNodesButNotEnoughForUpdate = 50;
constexpr int kNumberOfExistedNodesEnoughForUpdate = 100;
constexpr int kNumberOfNodesPerRequest = 1;

constexpr char kConfig[] = R"(
        fill_nodes_config: {
          num_properties: { minimum: 10 maximum: 10 }
          string_value_bytes: { minimum: 1 maximum: 10 }
        }
      )";

// Enumerates the workload configurations as the test parameters that ensure
// test coverage.
std::vector<WorkloadConfig> EnumerateConfigs(const bool is_update) {
  std::vector<WorkloadConfig> configs;

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(kConfig);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_minimum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_maximum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->set_update(is_update);
    config.mutable_fill_nodes_config()->set_specification(
        FillNodesConfig::ARTIFACT);
    configs.push_back(config);
  }

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(kConfig);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_minimum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_maximum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->set_update(is_update);
    config.mutable_fill_nodes_config()->set_specification(
        FillNodesConfig::EXECUTION);
    configs.push_back(config);
  }

  {
    WorkloadConfig config =
        testing::ParseTextProtoOrDie<WorkloadConfig>(kConfig);
    config.set_num_operations(kNumberOfOperations);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_minimum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->mutable_num_nodes()->set_maximum(
        kNumberOfNodesPerRequest);
    config.mutable_fill_nodes_config()->set_update(is_update);
    config.mutable_fill_nodes_config()->set_specification(
        FillNodesConfig::CONTEXT);
    configs.push_back(config);
  }

  return configs;
}

// Checks nodes update status. If the updates are working properly, the number
// of node custom properties should increase after the updates. This is due to
// when we insert existing node when setting up db, the number of node custom
// properties is less than the number of custom properties when we update the
// nodes.
void CheckUpdates(const FillNodesConfig& fill_nodes_config,
                  const std::vector<Node>& nodes_before,
                  const std::vector<Node>& nodes_after,
                  const int64 num_operations) {
  for (int64 i = 0; i < num_operations; ++i) {
    switch (fill_nodes_config.specification()) {
      case FillNodesConfig::ARTIFACT: {
        EXPECT_LT(
            absl::get<Artifact>(nodes_before[i]).custom_properties().size(),
            absl::get<Artifact>(nodes_after[i]).custom_properties().size());
        break;
      }
      case FillNodesConfig::EXECUTION: {
        EXPECT_LT(
            absl::get<Execution>(nodes_before[i]).custom_properties().size(),
            absl::get<Execution>(nodes_after[i]).custom_properties().size());
        break;
      }
      case FillNodesConfig::CONTEXT: {
        EXPECT_LT(
            absl::get<Context>(nodes_before[i]).custom_properties().size(),
            absl::get<Context>(nodes_after[i]).custom_properties().size());
        break;
      }
      default:
        LOG(FATAL) << "Wrong specification for FillTypes configuration input "
                      "in testing !";
    }
  }
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

// Tests the fail cases when there are no types inside db for inserting nodes.
TEST_P(FillNodesInsertParameterizedTestFixture, NonTypesExistTest) {
  EXPECT_EQ(fill_nodes_->SetUp(store_.get()).code(),
            tensorflow::error::FAILED_PRECONDITION);
}

// Tests the SetUpImpl() for FillNodes insert cases when db contains no nodes in
// the beginning. Checks the SetUpImpl() indeed prepares a list of work items
// whose length is the same as the specified number of operations.
TEST_P(FillNodesInsertParameterizedTestFixture, SetUpImplWhenNoNodesExistTest) {
  // Inserts some types into db so that nodes can be inserted later.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  TF_ASSERT_OK(fill_nodes_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_nodes_->num_operations());
}

// Tests the RunOpImpl() for FillNodes insert cases when db contains no nodes in
// the beginning. Checks indeed all the work items have been executed and the
// number of the nodes inside db is the same as the number of operations
// specified in the workload.
TEST_P(FillNodesInsertParameterizedTestFixture, InsertWhenNoNodesExistTest) {
  // Inserts some types into db so that nodes can be inserted later.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  TF_ASSERT_OK(fill_nodes_->SetUp(store_.get()));
  for (int64 i = 0; i < fill_nodes_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_nodes_->RunOp(i, store_.get(), op_stats));
  }
  // Gets all the existing current nodes inside db after insertion.
  std::vector<Node> existing_nodes;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes));
  EXPECT_EQ(GetParam().num_operations() * kNumberOfNodesPerRequest,
            existing_nodes.size());
}

// Tests the SetUpImpl() for FillNodes insert cases when db contains some nodes
// in the beginning. Checks the SetUpImpl() indeed prepares a list of work items
// whose length is the same as the specified number of operations.
TEST_P(FillNodesInsertParameterizedTestFixture,
       SetUpImplWhenSomeNodesExistTest) {
  // Inserts some types into db so that nodes can be inserted later.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  // Inserts some nodes into db in the first beginning.
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
      /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
      /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));
  TF_ASSERT_OK(fill_nodes_->SetUp(store_.get()));
  EXPECT_EQ(GetParam().num_operations(), fill_nodes_->num_operations());
}

// Tests the RunOpImpl() for FillNodes insert cases when db contains some nodes
// in the beginning. Checks indeed all the work items have been executed and the
// number of the new added nodes inside db is the same as the number of
// operations specified in the workload.
TEST_P(FillNodesInsertParameterizedTestFixture, InsertWhenSomeNodesExistTest) {
  // Inserts some types into db so that nodes can be inserted later.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfExistedTypesInDb,
      /*num_execution_types=*/kNumberOfExistedTypesInDb,
      /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  // Inserts some nodes into db in the first beginning.
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesInDb,
      /*num_execution_nodes=*/kNumberOfExistedNodesInDb,
      /*num_context_nodes=*/kNumberOfExistedNodesInDb, *store_));

  // Gets all the pre-inserted nodes inside db before insertion.
  std::vector<Node> existing_nodes_before_insert;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_before_insert));
  TF_ASSERT_OK(fill_nodes_->SetUp(store_.get()));
  for (int64 i = 0; i < fill_nodes_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_nodes_->RunOp(i, store_.get(), op_stats));
  }
  // Gets all the existing current nodes inside db after insertion.
  std::vector<Node> existing_nodes_after_insert;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_after_insert));

  EXPECT_EQ(
      GetParam().num_operations() * kNumberOfNodesPerRequest,
      existing_nodes_after_insert.size() - existing_nodes_before_insert.size());
}

// Test fixture that uses the same data configuration for multiple following
// parameterized FillNodes update tests.
// The parameter here is the specific Workload configuration that contains
// the FillNodes update configuration and the number of operations.
class FillNodesUpdateParameterizedTestFixture
    : public ::testing::TestWithParam<WorkloadConfig> {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    fill_nodes_update_ = absl::make_unique<FillNodes>(
        FillNodes(GetParam().fill_nodes_config(), GetParam().num_operations()));
    TF_ASSERT_OK(InsertTypesInDb(
        /*num_artifact_types=*/kNumberOfExistedTypesInDb,
        /*num_execution_types=*/kNumberOfExistedTypesInDb,
        /*num_context_types=*/kNumberOfExistedTypesInDb, *store_));
  }

  std::unique_ptr<FillNodes> fill_nodes_update_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for FillNodes update cases when db is empty in the
// first place. Then, the SetUpImpl() will make up and insert all the new nodes
// into db for later update. The number of nodes in db after SetUpImpl() should
// be the same as the specified number of update operations. Also, checks the
// SetUpImpl() indeed prepares a list of work items whose length is the same as
// the specified number of operations.
TEST_P(FillNodesUpdateParameterizedTestFixture,
       SetUpImplWhenDbContainsNoNodesTest) {
  TF_ASSERT_OK(fill_nodes_update_->SetUp(store_.get()));
  // Gets the number of nodes inside db after SetUpImpl().
  std::vector<Node> existing_nodes;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes));
  EXPECT_EQ(existing_nodes.size(),
            GetParam().num_operations() * kNumberOfNodesPerRequest);
  EXPECT_EQ(GetParam().num_operations(), fill_nodes_update_->num_operations());
}

// Tests the RunOpImpl() for FillNodes update cases when db is empty in the
// first place.
// Checks indeed all the work items have been executed and there are certain
// number of existed nodes inside db have been updated.
TEST_P(FillNodesUpdateParameterizedTestFixture,
       UpdateWhenDbContainsNoNodesTest) {
  TF_ASSERT_OK(fill_nodes_update_->SetUp(store_.get()));
  // Gets the existing nodes in db after set up but before update for later
  // comparison. The `existing_nodes_before_update` will contains the nodes that
  // were made up and inserted in the SetUpImpl().
  std::vector<Node> existing_nodes_before_update;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_before_update));

  for (int64 i = 0; i < fill_nodes_update_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_nodes_update_->RunOp(i, store_.get(), op_stats));
  }

  // Gets the existing nodes in db after update for later comparison.
  std::vector<Node> existing_nodes_after_update;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_after_update));

  // The update should update all the made up and inserted nodes in the
  // SetUpImpl() and not introduce any new nodes in the db.
  ASSERT_EQ(existing_nodes_before_update.size(),
            existing_nodes_after_update.size());

  // For all the nodes inside db, they should be updated. Checks their update
  // status.
  CheckUpdates(GetParam().fill_nodes_config(), existing_nodes_before_update,
               existing_nodes_after_update,
               (int64)existing_nodes_after_update.size());
}

// Tests the SetUpImpl() for FillNodes update cases when db contains not enough
// nodes for update. Then, the SetUpImpl() will make up and insert some new
// nodes into db for later update. The number of nodes in db after SetUpImpl()
// should be the same as the specified number of update operations. Also, checks
// the SetUpImpl() indeed prepares a list of work items whose length is the same
// as the specified number of operations.
TEST_P(FillNodesUpdateParameterizedTestFixture,
       SetUpImplWhenDbContainsNotEnoughNodesTest) {
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesButNotEnoughForUpdate,
      /*num_execution_nodes=*/kNumberOfExistedNodesButNotEnoughForUpdate,
      /*num_context_nodes=*/kNumberOfExistedNodesButNotEnoughForUpdate,
      *store_));
  TF_ASSERT_OK(fill_nodes_update_->SetUp(store_.get()));
  // Gets the number of nodes inside db after SetUpImpl().
  std::vector<Node> existing_nodes;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes));
  EXPECT_EQ(existing_nodes.size(),
            fill_nodes_update_->num_operations() * kNumberOfNodesPerRequest);
  EXPECT_EQ(GetParam().num_operations(), fill_nodes_update_->num_operations());
}

// Tests the RunOpImpl() for FillNodes update cases when db contains not enough
// nodes for update.
// Checks indeed all the work items have been executed and there are certain
// number of existed nodes inside db have been updated.
TEST_P(FillNodesUpdateParameterizedTestFixture,
       UpdateWhenDbContainsNotEnoughNodesTest) {
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesButNotEnoughForUpdate,
      /*num_execution_nodes=*/kNumberOfExistedNodesButNotEnoughForUpdate,
      /*num_context_nodes=*/kNumberOfExistedNodesButNotEnoughForUpdate,
      *store_));
  TF_ASSERT_OK(fill_nodes_update_->SetUp(store_.get()));
  // Gets the existing nodes in db after set up but before update for later
  // comparison. The `existing_nodes_before_update` will contains the existed
  // nodes and some new nodes that were made up and inserted in the SetUpImpl().
  std::vector<Node> existing_nodes_before_update;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_before_update));

  for (int64 i = 0; i < fill_nodes_update_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_nodes_update_->RunOp(i, store_.get(), op_stats));
  }

  // Gets the existing nodes in db after update for later comparison.
  std::vector<Node> existing_nodes_after_update;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_after_update));

  // The update should update all the nodes that includes some old existed
  // nodes and some new made up and inserted nodes and not introduce any new
  // nodes in the db.
  ASSERT_EQ(existing_nodes_before_update.size(),
            existing_nodes_after_update.size());

  // For all the nodes inside db, they should be updated. Checks their update
  // status.
  CheckUpdates(GetParam().fill_nodes_config(), existing_nodes_before_update,
               existing_nodes_after_update,
               (int64)existing_nodes_after_update.size());
}

// Tests the SetUpImpl() for FillNodes update cases when db contains enough
// nodes for update. Then, the SetUpImpl() will not make up and insert any new
// nodes into db for later update. The number of nodes in db should remain
// the same before and after the SetUpImpl(). Also, checks the SetUpImpl()
// indeed prepares a list of work items whose length is the same as the
// specified number of operations.
TEST_P(FillNodesUpdateParameterizedTestFixture,
       SetUpImplWhenDbContainsEnoughNodesTest) {
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesEnoughForUpdate,
      /*num_execution_nodes=*/kNumberOfExistedNodesEnoughForUpdate,
      /*num_context_nodes=*/kNumberOfExistedNodesEnoughForUpdate, *store_));
  // Gets the number of nodes inside db before SetUpImpl().
  std::vector<Node> existing_nodes_before_setup;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_before_setup));
  TF_ASSERT_OK(fill_nodes_update_->SetUp(store_.get()));
  // Gets the number of nodes inside db after SetUpImpl().
  std::vector<Node> existing_nodes_after_setup;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_after_setup));
  EXPECT_EQ(existing_nodes_before_setup.size(),
            existing_nodes_after_setup.size());
  EXPECT_EQ(GetParam().num_operations(), fill_nodes_update_->num_operations());
}

// Tests the RunOpImpl() for FillNodes update cases when db contains enough
// nodes for update. Checks indeed all the work items have been executed and
// there are certain number of existed nodes inside db have been updated.
TEST_P(FillNodesUpdateParameterizedTestFixture,
       UpdateWhenDbContainsEnoughNodesTest) {
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfExistedNodesEnoughForUpdate,
      /*num_execution_nodes=*/kNumberOfExistedNodesEnoughForUpdate,
      /*num_context_nodes=*/kNumberOfExistedNodesEnoughForUpdate, *store_));
  TF_ASSERT_OK(fill_nodes_update_->SetUp(store_.get()));
  // Gets the existing nodes in db after set up but before update for later
  // comparison. The `existing_nodes_before_update` will contains the existed
  // nodes in db.
  std::vector<Node> existing_nodes_before_update;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_before_update));

  for (int64 i = 0; i < fill_nodes_update_->num_operations(); ++i) {
    OpStats op_stats;
    TF_ASSERT_OK(fill_nodes_update_->RunOp(i, store_.get(), op_stats));
  }

  // Gets the existing nodes in db after update for later comparison.
  std::vector<Node> existing_nodes_after_update;
  TF_ASSERT_OK(GetExistingNodes(GetParam().fill_nodes_config(), *store_,
                                existing_nodes_after_update));

  // The update should update the number of update operations existed
  // nodes inside db and not introduce any new nodes in the db.
  ASSERT_EQ(existing_nodes_before_update.size(),
            existing_nodes_after_update.size());

  // For the number of update operations existed nodes inside db, they
  // should be updated. Checks their update status.
  CheckUpdates(GetParam().fill_nodes_config(), existing_nodes_before_update,
               existing_nodes_after_update,
               fill_nodes_update_->num_operations());
}

INSTANTIATE_TEST_CASE_P(
    FillNodesInsertTest, FillNodesInsertParameterizedTestFixture,
    ::testing::ValuesIn(EnumerateConfigs(/*is_update=*/false)));

INSTANTIATE_TEST_CASE_P(
    FillNodesUpdateTest, FillNodesUpdateParameterizedTestFixture,
    ::testing::ValuesIn(EnumerateConfigs(/*is_update=*/true)));

}  // namespace
}  // namespace ml_metadata
