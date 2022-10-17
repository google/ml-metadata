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

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_source_test_suite.h"
#include "ml_metadata/metadata_store/test_util.h"

namespace ml_metadata {
namespace testing {

namespace {
using ml_metadata::testing::EqualsProto;

class SqliteMetadataSourceContainer : public MetadataSourceContainer {
 public:
  SqliteMetadataSourceContainer() : MetadataSourceContainer() {
    SqliteMetadataSourceConfig config;
    metadata_source_ = std::make_unique<SqliteMetadataSource>(config);
  }

  explicit SqliteMetadataSourceContainer(SqliteMetadataSourceConfig config)
      : MetadataSourceContainer() {
    metadata_source_ = std::make_unique<SqliteMetadataSource>(config);
  }

  ~SqliteMetadataSourceContainer() override = default;

  MetadataSource* GetMetadataSource() override {
    return metadata_source_.get();
  }

  // InitTestSchema creates a new table t1(c1 INT, c2 VARCHAR(255)).
  void InitTestSchema() override {
    CHECK_EQ(absl::OkStatus(), metadata_source_->Connect());
    CHECK_EQ(absl::OkStatus(), metadata_source_->Begin());
    CHECK_EQ(absl::OkStatus(),
             metadata_source_->ExecuteQuery(
                 "CREATE TABLE t1 (c1 INT, c2 VARCHAR(255));", nullptr));
    CHECK_EQ(absl::OkStatus(), metadata_source_->Commit());
  }

  // InitSchemaAndPopulateRows creates table t1(c1 INT, c2 VARCHAR(255)) and
  // adds 3 rows to this table: (1,'v1'), (2,'v2'), (3, 'v3').
  void InitSchemaAndPopulateRows() override {
    InitTestSchema();
    CHECK_EQ(absl::OkStatus(), metadata_source_->Begin());
    CHECK_EQ(absl::OkStatus(), metadata_source_->ExecuteQuery(
                                   "INSERT INTO t1 VALUES (1, 'v1')", nullptr));
    CHECK_EQ(absl::OkStatus(), metadata_source_->ExecuteQuery(
                                   "INSERT INTO t1 VALUES (2, 'v2')", nullptr));
    CHECK_EQ(absl::OkStatus(), metadata_source_->ExecuteQuery(
                                   "INSERT INTO t1 VALUES (3, 'v3')", nullptr));
    CHECK_EQ(absl::OkStatus(), metadata_source_->Commit());
  }

 private:
  // by default, use a in-memory Sqlite3 metadata source
  std::unique_ptr<SqliteMetadataSource> metadata_source_;
};


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
      return std::make_unique<SqliteMetadataSourceContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata
