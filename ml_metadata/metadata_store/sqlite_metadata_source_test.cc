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
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_source_test_suite.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "tensorflow/core/platform/env.h"

namespace ml_metadata {
namespace testing {

namespace {
using ml_metadata::testing::EqualsProto;

class SqliteMetadataSourceContainer : public MetadataSourceContainer {
 public:
  SqliteMetadataSourceContainer() : MetadataSourceContainer() {
    SqliteMetadataSourceConfig config;
    metadata_source_ = absl::make_unique<SqliteMetadataSource>(config);
  }

  explicit SqliteMetadataSourceContainer(SqliteMetadataSourceConfig config)
      : MetadataSourceContainer() {
    metadata_source_ = absl::make_unique<SqliteMetadataSource>(config);
  }

  ~SqliteMetadataSourceContainer() override = default;

  MetadataSource* GetMetadataSource() override {
    return metadata_source_.get();
  }

  // InitTestSchema creates a new table t1(c1 INT, c2 VARCHAR(255)).
  void InitTestSchema() override {
    TF_CHECK_OK(metadata_source_->Connect());
    TF_CHECK_OK(metadata_source_->Begin());
    TF_CHECK_OK(metadata_source_->ExecuteQuery(
        "CREATE TABLE t1 (c1 INT, c2 VARCHAR(255));", nullptr));
    TF_CHECK_OK(metadata_source_->Commit());
  }

  // InitSchemaAndPopulateRows creates table t1(c1 INT, c2 VARCHAR(255)) and
  // adds 3 rows to this table: (1,'v1'), (2,'v2'), (3, 'v3').
  void InitSchemaAndPopulateRows() override {
    InitTestSchema();
    TF_CHECK_OK(metadata_source_->Begin());
    TF_CHECK_OK(metadata_source_->ExecuteQuery(
        "INSERT INTO t1 VALUES (1, 'v1')", nullptr));
    TF_CHECK_OK(metadata_source_->ExecuteQuery(
        "INSERT INTO t1 VALUES (2, 'v2')", nullptr));
    TF_CHECK_OK(metadata_source_->ExecuteQuery(
        "INSERT INTO t1 VALUES (3, 'v3')", nullptr));
    TF_CHECK_OK(metadata_source_->Commit());
  }

 private:
  // by default, use a in-memory Sqlite3 metadata source
  std::unique_ptr<SqliteMetadataSource> metadata_source_;
};

// Note that if this method fails, it does not clean up the file it created,
// causing issues.
TEST(SqliteMetadataSourceExtendedTest, TestPhysicalFile) {
  RecordSet expected_results;
  const std::string filename_uri =
      absl::StrCat(::testing::TempDir(), "test_physical_file_test.db");
  SqliteMetadataSourceConfig config;
  config.set_filename_uri(filename_uri);

  {
    SqliteMetadataSourceContainer container(config);
    MetadataSource* metadata_source = container.GetMetadataSource();

    container.InitSchemaAndPopulateRows();

    TF_ASSERT_OK(metadata_source->Begin());
    TF_ASSERT_OK(
        metadata_source->ExecuteQuery("SELECT * FROM t1", &expected_results));
    TF_ASSERT_OK(metadata_source->Commit());
    TF_ASSERT_OK(metadata_source->Close());
  }

  RecordSet query_results;
  // Connects to the same database without initialize the schema and rows
  SqliteMetadataSourceContainer container(config);
  MetadataSource* metadata_source = container.GetMetadataSource();
  TF_ASSERT_OK(metadata_source->Connect());
  TF_ASSERT_OK(metadata_source->Begin());
  TF_ASSERT_OK(
      metadata_source->ExecuteQuery("SELECT * FROM t1", &query_results));
  TF_ASSERT_OK(metadata_source->Commit());
  EXPECT_THAT(query_results, EqualsProto(expected_results));
  if (!filename_uri.empty()) {
    TF_CHECK_OK(tensorflow::Env::Default()->DeleteFile(filename_uri));
  }
}

// Test EscapeString utility method.
TEST(SqliteMetadataSourceExtendedTest, TestEscapeString) {
  SqliteMetadataSourceContainer container;
  MetadataSource* metadata_source = container.GetMetadataSource();
  EXPECT_EQ(metadata_source->EscapeString("''"), "''''");
  EXPECT_EQ(metadata_source->EscapeString("'\'"), "''''");
  EXPECT_EQ(metadata_source->EscapeString("'\"text\"'"), "''\"text\"''");
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    SqliteMetadataSourceCommonTest, MetadataSourceTestSuite,
    ::testing::Values([]() {
      return absl::make_unique<SqliteMetadataSourceContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata
