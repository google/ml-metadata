/* Copyright 2019 Google LLC

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
#include "absl/strings/str_cat.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {
using testing::EqualsProto;
using testing::ParseTextProtoOrDie;

class SqliteMetadataSourceTest : public ::testing::Test {
 protected:
  SqliteMetadataSourceTest() {
    SqliteMetadataSourceConfig config;
    metadata_source_ = absl::make_unique<SqliteMetadataSource>(config);
  }

  ~SqliteMetadataSourceTest() override = default;

  void InitTestSchema() {
    TF_CHECK_OK(metadata_source_->Connect());
    TF_CHECK_OK(metadata_source_->Begin());
    TF_CHECK_OK(metadata_source_->ExecuteQuery(
        "CREATE TABLE t1 (c1 INT, c2 VARCHAR);", nullptr));
    TF_CHECK_OK(metadata_source_->Commit());
  }

  static void InitSchemaAndPopulateRows(MetadataSource* metadata_source) {
    TF_CHECK_OK(metadata_source->Connect());
    TF_CHECK_OK(metadata_source->Begin());
    TF_CHECK_OK(metadata_source->ExecuteQuery(
        "CREATE TABLE t1 (c1 INT, c2 VARCHAR);", nullptr));
    TF_CHECK_OK(metadata_source->ExecuteQuery("INSERT INTO t1 VALUES (1, 'v1')",
                                              nullptr));
    TF_CHECK_OK(metadata_source->ExecuteQuery("INSERT INTO t1 VALUES (2, 'v2')",
                                              nullptr));
    TF_CHECK_OK(metadata_source->ExecuteQuery("INSERT INTO t1 VALUES (3, 'v3')",
                                              nullptr));
    TF_CHECK_OK(metadata_source->Commit());
  }

  void TearDown() override { metadata_source_.reset(); }

  // by default, use a in-memory Sqlite3 metadata source for each tes
  std::unique_ptr<MetadataSource> metadata_source_;
};

TEST_F(SqliteMetadataSourceTest, TestQueryWithoutConnect) {
  tensorflow::Status s =
      metadata_source_->ExecuteQuery("CREATE TABLE foo(bar INT)", nullptr);
  EXPECT_EQ(s.code(), tensorflow::error::FAILED_PRECONDITION);
}

TEST_F(SqliteMetadataSourceTest, TestInsert) {
  InitTestSchema();
  TF_EXPECT_OK(metadata_source_->Begin());
  TF_EXPECT_OK(metadata_source_->ExecuteQuery("INSERT INTO t1 VALUES (1, 'v1')",
                                              nullptr));
  RecordSet expected_results = ParseTextProtoOrDie<RecordSet>(
      R"(column_names: "c1"
         column_names: "c2"
         records: { values: "1" values: "v1" })");

  RecordSet query_results;
  TF_EXPECT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  TF_EXPECT_OK(metadata_source_->Commit());
  EXPECT_EQ(1, query_results.records().size());
  EXPECT_THAT(query_results, EqualsProto(expected_results));
}

TEST_F(SqliteMetadataSourceTest, TestEscapeString) {
  TF_CHECK_OK(metadata_source_->Connect());
  EXPECT_EQ(metadata_source_->EscapeString("''"), "''''");
  EXPECT_EQ(metadata_source_->EscapeString("'\'"), "''''");
  EXPECT_EQ(metadata_source_->EscapeString("'\"text\"'"), "''\"text\"''");
}

TEST_F(SqliteMetadataSourceTest, TestInsertWithEscapedStringValue) {
  InitTestSchema();
  TF_EXPECT_OK(metadata_source_->Begin());
  TF_EXPECT_OK(metadata_source_->ExecuteQuery(
      absl::StrCat("INSERT INTO t1 VALUES (1, '",
                   metadata_source_->EscapeString("''"), "')"),
      nullptr));
  RecordSet expected_results = ParseTextProtoOrDie<RecordSet>(
      R"(column_names: "c1"
         column_names: "c2"
         records: { values: "1" values: "''" })");

  RecordSet query_results;
  TF_EXPECT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  TF_EXPECT_OK(metadata_source_->Commit());
  EXPECT_EQ(1, query_results.records().size());
  EXPECT_THAT(query_results, EqualsProto(expected_results));
}

TEST_F(SqliteMetadataSourceTest, TestDelete) {
  InitSchemaAndPopulateRows(metadata_source_.get());
  RecordSet query_results;
  TF_EXPECT_OK(metadata_source_->Begin());
  TF_EXPECT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  EXPECT_LT(0, query_results.records().size());

  TF_EXPECT_OK(metadata_source_->ExecuteQuery("DELETE FROM t1", nullptr));

  query_results.Clear();
  TF_EXPECT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  TF_EXPECT_OK(metadata_source_->Commit());
  EXPECT_EQ(0, query_results.records().size());
}

TEST_F(SqliteMetadataSourceTest, TestUpdate) {
  InitSchemaAndPopulateRows(metadata_source_.get());
  TF_EXPECT_OK(metadata_source_->Begin());
  TF_EXPECT_OK(metadata_source_->ExecuteQuery(
      "UPDATE t1 SET c2 = 'v100' WHERE c1 == 1", nullptr));

  RecordSet expected_results = ParseTextProtoOrDie<RecordSet>(
      R"(column_names: "c1"
         column_names: "c2"
         records: { values: "1" values: "v100" })");

  RecordSet query_results;
  TF_EXPECT_OK(metadata_source_->ExecuteQuery("SELECT * FROM t1 WHERE c1 == 1",
                                              &query_results));
  TF_EXPECT_OK(metadata_source_->Commit());
  EXPECT_THAT(query_results, EqualsProto(expected_results));
}

// multiple queries, and rollback, nothing should happen
TEST_F(SqliteMetadataSourceTest, TestMultiQueryTransaction) {
  InitSchemaAndPopulateRows(metadata_source_.get());
  RecordSet expected_results;
  TF_ASSERT_OK(metadata_source_->Begin());
  TF_ASSERT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &expected_results));
  TF_ASSERT_OK(metadata_source_->Commit());

  TF_ASSERT_OK(metadata_source_->Begin());
  TF_ASSERT_OK(metadata_source_->ExecuteQuery("DELETE FROM t1", nullptr));
  TF_ASSERT_OK(metadata_source_->Rollback());

  // rollback the delete query, there should be no change in the database
  RecordSet query_results;
  TF_ASSERT_OK(metadata_source_->Begin());
  TF_ASSERT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  EXPECT_THAT(query_results, EqualsProto(expected_results));
  TF_ASSERT_OK(metadata_source_->Commit());

  // now it is in a new transaction, insert two records, then commit
  TF_ASSERT_OK(metadata_source_->Begin());
  TF_ASSERT_OK(metadata_source_->ExecuteQuery("INSERT INTO t1 VALUES (1, 'v1')",
                                              nullptr));
  TF_ASSERT_OK(metadata_source_->ExecuteQuery("INSERT INTO t1 VALUES (2, 'v2')",
                                              nullptr));
  TF_ASSERT_OK(metadata_source_->Commit());

  TF_ASSERT_OK(metadata_source_->Begin());
  query_results.Clear();
  TF_EXPECT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  TF_ASSERT_OK(metadata_source_->Commit());
  int expected_num_rows = expected_results.records().size() + 2;
  EXPECT_EQ(expected_num_rows, query_results.records().size());
}


}  // namespace
}  // namespace ml_metadata
