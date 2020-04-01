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
#include "ml_metadata/metadata_store/mysql_metadata_source.h"

#include <memory>

#include "gflags/gflags.h"
#include "google/protobuf/text_format.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/test_mysql_metadata_source_initializer.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"

DEFINE_bool(enable_sockets, true, "Whether to run socket tests.");

namespace ml_metadata {
namespace testing {
namespace {

using ::tensorflow::Status;

class MySqlMetadataSourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    metadata_source_initializer_ = GetTestMySqlMetadataSourceInitializer();
    // Use TCP by default. Tests that need other connection types should not use
    // this test fixture.
    metadata_source_ = metadata_source_initializer_->Init(
        TestMySqlMetadataSourceInitializer::ConnectionType::kTcp);
  }

  void TearDown() override { metadata_source_initializer_->Cleanup(); }

  void InitTestSchema() {
    TF_CHECK_OK(metadata_source_->Connect());
    TF_CHECK_OK(metadata_source_->Begin());
    TF_CHECK_OK(metadata_source_->ExecuteQuery(
        "CREATE TABLE t1 (c1 INT, c2 VARCHAR(255));", nullptr));
    TF_CHECK_OK(metadata_source_->Commit());
  }

  void InitSchemaAndPopulateRows() {
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

  // An unowned TestMySqlMetadataSourceInitializer from a call to
  // GetTestMySqlMetadataSourceInitializer().
  std::unique_ptr<TestMySqlMetadataSourceInitializer>
      metadata_source_initializer_;
  // An unowned MySqlMetadataSource from a call to
  // metadata_source_initializer->Init().
  MySqlMetadataSource* metadata_source_;
};

// This test is to verify we can connect to the DB using sockets.
// We use a fixtureless test to avoid conflicting with the default
// metadata_source_initializer defined in MySqlMetadataSourceTest.
TEST(MySqlMetadataSourceSocketTest, TestConnectBySocket) {
  // TODO(b/140584643) Fix MacOS Kokoro test to enable connecting via sockets.
  if (!FLAGS_enable_sockets) {
    GTEST_SKIP() << "Socket tests disabled.";
  }

  auto metadata_source_initializer = GetTestMySqlMetadataSourceInitializer();
  auto metadata_source = metadata_source_initializer->Init(
      TestMySqlMetadataSourceInitializer::ConnectionType::kSocket);
  TF_ASSERT_OK(metadata_source->Connect());
  TF_ASSERT_OK(metadata_source->Begin());
  TF_ASSERT_OK(metadata_source->ExecuteQuery(
      "CREATE TABLE t1 (c1 INT, c2 VARCHAR(255));", nullptr));
  TF_ASSERT_OK(metadata_source->Commit());
  metadata_source_initializer->Cleanup();
}

TEST_F(MySqlMetadataSourceTest, TestQueryWithoutConnect) {
  Status s =
      metadata_source_->ExecuteQuery("CREATE TABLE foo(bar INT)", nullptr);
  EXPECT_EQ(s.code(), tensorflow::error::FAILED_PRECONDITION);
}

TEST_F(MySqlMetadataSourceTest, TestInsert) {
  InitTestSchema();
  TF_ASSERT_OK(metadata_source_->Begin());
  TF_ASSERT_OK(metadata_source_->ExecuteQuery("INSERT INTO t1 VALUES (1, 'v1')",
                                              nullptr));
  RecordSet expected_results;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"(column_names: "c1"
         column_names: "c2"
         records: {
            values: "1"
            values: "v1"
         })",
      &expected_results));

  RecordSet query_results;
  TF_ASSERT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  TF_ASSERT_OK(metadata_source_->Commit());

  EXPECT_EQ(1, query_results.records().size());
  EXPECT_THAT(query_results, EqualsProto(expected_results));
}

// TODO(b/149022372): add this test to the shared metadata_source test suite.
TEST_F(MySqlMetadataSourceTest, TestNull) {
  InitTestSchema();
  TF_EXPECT_OK(metadata_source_->Begin());
  TF_EXPECT_OK(metadata_source_->ExecuteQuery("INSERT INTO t1 VALUES (1, NULL)",
                                              nullptr));
  RecordSet expected_results = ParseTextProtoOrDie<RecordSet>(
      absl::Substitute(
        R"(column_names: "c1"
           column_names: "c2"
           records: { values: "1" values: "$0" })", kMetadataSourceNull));

  RecordSet query_results;
  TF_EXPECT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  TF_EXPECT_OK(metadata_source_->Commit());
  EXPECT_EQ(1, query_results.records().size());
  EXPECT_THAT(query_results, EqualsProto(expected_results));
}


TEST_F(MySqlMetadataSourceTest, TestEscapeString) {
  TF_CHECK_OK(metadata_source_->Connect());
  EXPECT_EQ(metadata_source_->EscapeString("''"), "\\'\\'");
  EXPECT_EQ(metadata_source_->EscapeString("'\'"), "\\'\\\'");
  EXPECT_EQ(metadata_source_->EscapeString("'\"text\"'"), "\\'\\\"text\\\"\\'");
}

TEST_F(MySqlMetadataSourceTest, TestInsertWithEscapedStringValue) {
  InitTestSchema();
  TF_ASSERT_OK(metadata_source_->Begin());
  TF_ASSERT_OK(metadata_source_->ExecuteQuery(
      absl::StrCat("INSERT INTO t1 VALUES (1, '",
                   metadata_source_->EscapeString("''"), "')"),
      nullptr));
  RecordSet expected_results;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"(column_names: "c1"
         column_names: "c2"
         records: {
            values: "1"
            values: "''"
         })",
      &expected_results));

  RecordSet query_results;
  TF_ASSERT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  TF_EXPECT_OK(metadata_source_->Commit());
  EXPECT_EQ(1, query_results.records().size());
  EXPECT_THAT(query_results, EqualsProto(expected_results));
}

TEST_F(MySqlMetadataSourceTest, TestDelete) {
  InitSchemaAndPopulateRows();
  RecordSet query_results;
  TF_ASSERT_OK(metadata_source_->Begin());
  TF_ASSERT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  ASSERT_LT(0, query_results.records().size());

  TF_ASSERT_OK(metadata_source_->ExecuteQuery("DELETE FROM t1", nullptr));

  query_results.Clear();
  TF_ASSERT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  TF_ASSERT_OK(metadata_source_->Commit());
  EXPECT_EQ(0, query_results.records().size());
}

TEST_F(MySqlMetadataSourceTest, TestUpdate) {
  InitSchemaAndPopulateRows();
  TF_ASSERT_OK(metadata_source_->Begin());
  TF_ASSERT_OK(metadata_source_->ExecuteQuery(
      "UPDATE t1 SET c2 = 'v100' WHERE c1 = 1", nullptr));

  RecordSet expected_results;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"(column_names: "c1"
         column_names: "c2"
         records: {
            values: "1"
            values: "v100"
         })",
      &expected_results));

  RecordSet query_results;
  TF_EXPECT_OK(metadata_source_->ExecuteQuery("SELECT * FROM t1 WHERE c1 = 1",
                                              &query_results));
  EXPECT_THAT(query_results, EqualsProto(expected_results));
  TF_ASSERT_OK(metadata_source_->Commit());
}

// multiple queries, and rollback, nothing should happen
TEST_F(MySqlMetadataSourceTest, TestMultiQueryTransaction) {
  InitSchemaAndPopulateRows();
  RecordSet expected_results;
  TF_EXPECT_OK(metadata_source_->Begin());
  TF_EXPECT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &expected_results));
  TF_EXPECT_OK(metadata_source_->Commit());

  // disable auto commit
  TF_EXPECT_OK(metadata_source_->Begin());
  TF_EXPECT_OK(metadata_source_->ExecuteQuery("DELETE FROM t1", nullptr));
  TF_EXPECT_OK(metadata_source_->Rollback());

  // rollback the delete query, there should be no change in the database
  RecordSet query_results;
  TF_EXPECT_OK(metadata_source_->Begin());
  TF_EXPECT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  EXPECT_THAT(query_results, EqualsProto(expected_results));

  // now it is in a new transaction, insert two records, then commit
  TF_EXPECT_OK(metadata_source_->ExecuteQuery("INSERT INTO t1 VALUES (1, 'v1')",
                                              nullptr));
  TF_EXPECT_OK(metadata_source_->ExecuteQuery("INSERT INTO t1 VALUES (2, 'v2')",
                                              nullptr));
  TF_EXPECT_OK(metadata_source_->Commit());
  query_results.Clear();
  TF_EXPECT_OK(metadata_source_->Begin());
  TF_EXPECT_OK(
      metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  int expected_num_rows = expected_results.records().size() + 2;
  TF_EXPECT_OK(metadata_source_->Commit());
  EXPECT_EQ(expected_num_rows, query_results.records().size());
}

}  // namespace
}  // namespace testing
}  // namespace ml_metadata

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
