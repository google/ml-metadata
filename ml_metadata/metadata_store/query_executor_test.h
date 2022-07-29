/* Copyright 2022 Google LLC

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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_QUERY_EXECUTOR_TEST_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_QUERY_EXECUTOR_TEST_H_

#include <memory>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {
namespace testing {

class QueryExecutorContainer {
 public:
  virtual ~QueryExecutorContainer() = default;

  virtual MetadataSource* GetMetadataSource() = 0;

  virtual QueryExecutor* GetQueryExecutor() = 0;

  virtual absl::Status Init() {
    MLMD_RETURN_IF_ERROR(GetQueryExecutor()->InitMetadataSource());
    return absl::OkStatus();
  }

  // Adds a commit point in the tests.
  // Default to be a no-op for SQLite, MySQL.
  virtual absl::Status AddCommitPoint() { return absl::OkStatus(); }
};

class QueryConfigExecutorContainer : public QueryExecutorContainer {
 public:
  QueryConfigExecutorContainer(const MetadataSourceQueryConfig& config)
      : config_(config) {}

  virtual ~QueryConfigExecutorContainer() = default;

  virtual QueryExecutor* GetQueryExecutor() = 0;

 private:
  MetadataSourceQueryConfig config_;
};

// Represents the type of the Gunit Test param for the parameterized
// QueryExecutorTest.
using QueryExecutorContainerFactory =
    std::function<std::unique_ptr<QueryExecutorContainer>()>;

// A parameterized abstract test fixture to run tests for
// QueryExecutor created with different MetadataSource types.
// See query_executor_test.cc for list of test cases using this
// fixture.
class QueryExecutorTest
    : public ::testing::TestWithParam<QueryExecutorContainerFactory> {
 protected:
  void SetUp() override {
    query_executor_container_ = GetParam()();
    metadata_source_ = query_executor_container_->GetMetadataSource();
    query_executor_ = query_executor_container_->GetQueryExecutor();
    CHECK_EQ(absl::OkStatus(), metadata_source_->Begin());
  }

  void TearDown() override {
    CHECK_EQ(absl::OkStatus(), metadata_source_->Commit());
    metadata_source_ = nullptr;
    query_executor_ = nullptr;
    query_executor_container_ = nullptr;
  }

  absl::Status Init() { return query_executor_container_->Init(); }

  // Uses to a add commit point if needed in the tests.
  // Default to be a no-op for SQLite, MySQL.
  absl::Status AddCommitPointIfNeeded() {
    return query_executor_container_->AddCommitPoint();
  }

  std::unique_ptr<QueryExecutorContainer> query_executor_container_;

  // metadata_source_ and query_executor_ are unowned.
  MetadataSource* metadata_source_;
  QueryExecutor* query_executor_;
};

}  // namespace testing
}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_QUERY_EXECUTOR_TEST_H_
