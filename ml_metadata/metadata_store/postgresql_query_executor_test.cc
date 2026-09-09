/* Copyright 2023 Google LLC

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
// Test suite for a PostgreSQL query config-based QueryExecutor.
#include "ml_metadata/metadata_store/postgresql_query_executor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "absl/strings/string_view.h"
#include "gflags/gflags.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/metadata_store/query_executor_test.h"
#include "ml_metadata/metadata_store/test_postgresql_metadata_source_initializer.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"

namespace ml_metadata {
namespace testing {

namespace {
// PostgreSQLQueryExecutorContainer implements
// QueryConfigExecutorContainer to generate and retrieve a
// QueryExecutor based on a PostgreSQLMetadataSource.
class PostgreSQLQueryExecutorContainer : public QueryConfigExecutorContainer {
 public:
  PostgreSQLQueryExecutorContainer()
      : QueryConfigExecutorContainer(
            util::GetPostgreSQLMetadataSourceQueryConfig()) {
    metadata_source_initializer_ = GetTestPostgreSQLMetadataSourceInitializer();
    metadata_source_ = metadata_source_initializer_->Init();
    if (!metadata_source_->is_connected())
      CHECK_EQ(metadata_source_->Connect(), absl::OkStatus());
    query_executor_ = absl::WrapUnique(new PostgreSQLQueryExecutor(
        util::GetPostgreSQLMetadataSourceQueryConfig(), metadata_source_));
  }

  ~PostgreSQLQueryExecutorContainer() override {
    metadata_source_initializer_->Cleanup();
  }

  MetadataSource* GetMetadataSource() override { return metadata_source_; }
  QueryExecutor* GetQueryExecutor() override { return query_executor_.get(); }

 private:
  // An unowned TestPostgreSQLMetadataSourceInitializer from a call to
  // GetTestPostgreSQLMetadataSourceInitializer().
  std::unique_ptr<TestPostgreSQLMetadataSourceInitializer>
      metadata_source_initializer_;
  // An unowned PostgreSQLMetadataSource from a call to
  // metadata_source_initializer->Init().
  PostgreSQLMetadataSource* metadata_source_;
  std::unique_ptr<QueryExecutor> query_executor_;
};

// Two independent connections are required: a process-local mutex would not
// prevent different metadata server requests from creating duplicate types.
class PostgreSQLTypeUpsertLockTest
    : public ::testing::TestWithParam<std::tuple<TypeKind, bool>> {
 protected:
  void SetUp() override {
    first_initializer_ = GetTestPostgreSQLMetadataSourceInitializer();
    second_initializer_ = GetTestPostgreSQLMetadataSourceInitializer();
    first_source_ = first_initializer_->Init();
    if (!first_source_->is_connected()) {
      ASSERT_EQ(first_source_->Connect(), absl::OkStatus());
    }
    first_executor_ = std::make_unique<PostgreSQLQueryExecutor>(
        util::GetPostgreSQLMetadataSourceQueryConfig(), first_source_);
    ASSERT_EQ(first_source_->Begin(), absl::OkStatus());
    ASSERT_EQ(first_executor_->InitMetadataSource(), absl::OkStatus());
    ASSERT_EQ(first_source_->Commit(), absl::OkStatus());
    second_source_ = second_initializer_->Init();
    if (!second_source_->is_connected()) {
      ASSERT_EQ(second_source_->Connect(), absl::OkStatus());
    }
    second_executor_ = std::make_unique<PostgreSQLQueryExecutor>(
        util::GetPostgreSQLMetadataSourceQueryConfig(), second_source_);
  }

  void TearDown() override {
    // Close both connections before dropping their shared database, including
    // when a fatal assertion leaves either transaction open.
    if (second_source_ != nullptr && second_source_->is_connected()) {
      EXPECT_EQ(second_source_->Close(), absl::OkStatus());
    }
    if (first_source_ != nullptr && first_source_->is_connected()) {
      EXPECT_EQ(first_source_->Close(), absl::OkStatus());
    }
    first_initializer_->Cleanup();
  }

  absl::Status FindTypes(QueryExecutor* executor, RecordSet* result) {
    std::vector<std::pair<std::string, std::string>> names = {
        {"concurrent_type", std::get<1>(GetParam()) ? "v1" : ""}};
    return executor->SelectTypesByNamesAndVersions(
        absl::MakeSpan(names), std::get<0>(GetParam()), result);
  }

  absl::Status InsertType(int64_t* id) {
    std::optional<absl::string_view> version = std::nullopt;
    if (std::get<1>(GetParam())) version = "v1";
    switch (std::get<0>(GetParam())) {
      case TypeKind::ARTIFACT_TYPE:
        return first_executor_->InsertArtifactType(
            "concurrent_type", version, std::nullopt, std::nullopt, id);
      case TypeKind::EXECUTION_TYPE:
        return first_executor_->InsertExecutionType("concurrent_type", version,
                                                    std::nullopt, nullptr,
                                                    nullptr, std::nullopt, id);
      case TypeKind::CONTEXT_TYPE:
        return first_executor_->InsertContextType(
            "concurrent_type", version, std::nullopt, std::nullopt, id);
      default:
        return absl::InvalidArgumentError("Unexpected type kind");
    }
  }

  std::unique_ptr<TestPostgreSQLMetadataSourceInitializer> first_initializer_;
  std::unique_ptr<TestPostgreSQLMetadataSourceInitializer> second_initializer_;
  PostgreSQLMetadataSource* first_source_ = nullptr;
  PostgreSQLMetadataSource* second_source_ = nullptr;
  std::unique_ptr<PostgreSQLQueryExecutor> first_executor_;
  std::unique_ptr<PostgreSQLQueryExecutor> second_executor_;
};

TEST_P(PostgreSQLTypeUpsertLockTest, MissingTypeIsLockedUntilCommit) {
  ASSERT_EQ(first_source_->Begin(), absl::OkStatus());
  RecordSet first_result;
  ASSERT_EQ(FindTypes(first_executor_.get(), &first_result), absl::OkStatus());
  ASSERT_EQ(first_result.records_size(), 0);

  ASSERT_EQ(second_source_->Begin(), absl::OkStatus());
  ASSERT_EQ(
      second_source_->ExecuteQuery("SET LOCAL lock_timeout = '100ms'", nullptr),
      absl::OkStatus());
  // Ordinary type reads do not participate in the upsert lock.
  RecordSet ordinary_result;
  ASSERT_EQ(second_executor_->SelectAllTypes(std::get<0>(GetParam()),
                                             &ordinary_result),
            absl::OkStatus());
  RecordSet second_result;
  const absl::Status blocked =
      FindTypes(second_executor_.get(), &second_result);
  // A lock timeout makes the overlap deterministic without timing threads.
  // Without the fix this read succeeds and observes the same missing type.
  EXPECT_FALSE(blocked.ok());
  EXPECT_NE(blocked.message().find("lock timeout"), std::string::npos);
  ASSERT_EQ(second_source_->Rollback(), absl::OkStatus());

  int64_t id;
  ASSERT_EQ(InsertType(&id), absl::OkStatus());
  ASSERT_EQ(first_source_->Commit(), absl::OkStatus());

  ASSERT_EQ(second_source_->Begin(), absl::OkStatus());
  ASSERT_EQ(
      second_source_->ExecuteQuery("SET LOCAL lock_timeout = '1s'", nullptr),
      absl::OkStatus());
  second_result.Clear();
  ASSERT_EQ(FindTypes(second_executor_.get(), &second_result),
            absl::OkStatus());
  ASSERT_EQ(second_result.records_size(), 1);
  EXPECT_EQ(second_result.records(0).values(0), std::to_string(id));
  ASSERT_EQ(second_source_->Commit(), absl::OkStatus());
}

TEST_P(PostgreSQLTypeUpsertLockTest, RollbackReleasesMissingTypeLock) {
  ASSERT_EQ(first_source_->Begin(), absl::OkStatus());
  RecordSet result;
  ASSERT_EQ(FindTypes(first_executor_.get(), &result), absl::OkStatus());
  ASSERT_EQ(result.records_size(), 0);
  ASSERT_EQ(first_source_->Rollback(), absl::OkStatus());

  ASSERT_EQ(second_source_->Begin(), absl::OkStatus());
  ASSERT_EQ(
      second_source_->ExecuteQuery("SET LOCAL lock_timeout = '1s'", nullptr),
      absl::OkStatus());
  ASSERT_EQ(FindTypes(second_executor_.get(), &result), absl::OkStatus());
  EXPECT_EQ(result.records_size(), 0);
  ASSERT_EQ(second_source_->Commit(), absl::OkStatus());
}

INSTANTIATE_TEST_SUITE_P(
    PostgreSQL, PostgreSQLTypeUpsertLockTest,
    ::testing::Combine(::testing::Values(TypeKind::ARTIFACT_TYPE,
                                         TypeKind::EXECUTION_TYPE,
                                         TypeKind::CONTEXT_TYPE),
                       ::testing::Bool()));

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    PostgreSQLQueryExecutorContainer, QueryExecutorTest,
    ::testing::Values([]() {
      return std::make_unique<PostgreSQLQueryExecutorContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
