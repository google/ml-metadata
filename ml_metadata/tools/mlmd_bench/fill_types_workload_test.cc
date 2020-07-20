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

// Test fixture that uses the same data configuration for multiple following
// tests.
class FillTypesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
    FillTypesConfig fill_types_config;
    fill_types_config = testing::ParseTextProtoOrDie<FillTypesConfig>(
        R"(
          update: false
          specification: CONTEXT_TYPE
          num_properties { minimum: 1 maximum: 10 }
        )");
    fill_types_ = absl::make_unique<FillTypes>(
        FillTypes(fill_types_config, kNumOperations));
  }

  std::unique_ptr<FillTypes> fill_types_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the SetUpImpl() for FillTypes.
// Checks the SetUpImpl() indeed prepares a list of work items whose length is
// the same as the specified number of operations.
TEST_F(FillTypesTest, SetUpImplTest) {
  TF_ASSERT_OK(fill_types_->SetUp(store_.get()));
  EXPECT_EQ(kNumOperations, fill_types_->num_operations());
}

// Tests the RunOpImpl() for insert types.
// Checks indeed all the work items have been executed and all the types have
// been inserted into the db.
TEST_F(FillTypesTest, InsertTest) {
  TF_ASSERT_OK(fill_types_->SetUp(store_.get()));
  for (int64 i = 0; i < fill_types_->num_operations(); ++i) {
    OpStats op_stats;
    TF_EXPECT_OK(fill_types_->RunOp(i, store_.get(), op_stats));
  }

  GetContextTypesResponse get_response;
  TF_ASSERT_OK(store_->GetContextTypes(/*request=*/{}, &get_response));
  EXPECT_EQ(get_response.context_types_size(), fill_types_->num_operations());
}

}  // namespace
}  // namespace ml_metadata
