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
#include "ml_metadata/tools/mlmd_bench/workload.h"

#include <gtest/gtest.h>
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

// A fake workload class that implements the Workload interface.
// Its SetUpImpl(), RunOpImpl() and TearDownImpl() are designed
// specifically for testing.
class FakeWorkload : public Workload<std::string> {
  tensorflow::Status SetUpImpl(MetadataStore* store) {
    // The work items for this fake workload will be ten pairs of string and
    // integer.
    for (int i = 0; i < 100; ++i) {
      work_items_.emplace_back("abcd", 3456);
    }
    return tensorflow::Status::OK();
  }

  tensorflow::Status RunOpImpl(const int64 i, MetadataStore* store) {
    return tensorflow::Status::OK();
  }

  tensorflow::Status TearDownImpl() {
    work_items_.clear();
    return tensorflow::Status::OK();
  }

  std::string GetName() { return "fake_workload"; }
};

// Test fixture that uses the same data configuration for multiple following
// tests.
class WorkloadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ConnectionConfig mlmd_config;
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store_));
  }

  FakeWorkload workload_;
  std::unique_ptr<MetadataStore> store_;
};

// Tests the cases when executing in the right sequence.
TEST_F(WorkloadTest, RunInRightSequenceTest) {
  OpStats op_stats;
  TF_ASSERT_OK(workload_.SetUp(store_.get()));
  TF_EXPECT_OK(workload_.RunOp(0, store_.get(), op_stats));
  TF_EXPECT_OK(workload_.TearDown());
}

// Tests the cases when executing RunOp() / TearDown() before calling SetUp().
// The Failed Precondition error should be returned.
TEST_F(WorkloadTest, FailedPreconditionTest) {
  OpStats op_stats;
  EXPECT_EQ(workload_.RunOp(0, store_.get(), op_stats).code(),
            tensorflow::error::FAILED_PRECONDITION);
  EXPECT_EQ(workload_.TearDown().code(),
            tensorflow::error::FAILED_PRECONDITION);
}

// Tests the cases when inputting invalid work item index to RunOp().
// The Invalid Argument error should be returned.
TEST_F(WorkloadTest, InvalidWorkItemIndexTest) {
  OpStats op_stats;
  TF_ASSERT_OK(workload_.SetUp(store_.get()));
  // -1 and 100 are not within the range [0, 100), the RunOp() should return
  // Invalid Argument error.
  EXPECT_EQ(workload_.RunOp(-1, store_.get(), op_stats).code(),
            tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(workload_.RunOp(100, store_.get(), op_stats).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace ml_metadata
