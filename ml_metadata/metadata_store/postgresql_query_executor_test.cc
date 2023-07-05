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

#include <memory>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/metadata_store/query_executor_test.h"
#include "ml_metadata/metadata_store/test_postgresql_metadata_source_initializer.h"
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

  ~PostgreSQLQueryExecutorContainer() override = default;

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

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    PostgreSQLQueryExecutorContainer, QueryExecutorTest,
    ::testing::Values([]() {
      return std::make_unique<PostgreSQLQueryExecutorContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata
