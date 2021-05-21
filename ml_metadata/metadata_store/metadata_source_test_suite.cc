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
#include "ml_metadata/metadata_store/metadata_source_test_suite.h"

#include <memory>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/test_util.h"

namespace ml_metadata {
namespace testing {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

// Test Insert execution.
// Initialization: creates an empty table t1 (c1 INT, c2 VARCHAR(255)) with test
// schema.
// Execution: Insert a new row (1,'v1') into t1.
// Expectation: all the retrieved rows in t1 are (1, 'v1').
TEST_P(MetadataSourceTestSuite, TestInsert) {
  metadata_source_container_->InitTestSchema();
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->ExecuteQuery(
                                  "INSERT INTO t1 VALUES (1, 'v1')", nullptr));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Commit());

  EXPECT_EQ(1, query_results.records().size());
  EXPECT_THAT(query_results, EqualsProto(expected_results));
}

// Test Insert execution with escaped string value.
// Initialization: creates an empty table t1 (c1 INT, c2 VARCHAR(255)) with test
// schema.
// Execution: Insert a new row (1,'') into the table.
// Expectation: all the retrieved rows in t1 are (1, '').
TEST_P(MetadataSourceTestSuite, TestInsertWithEscapedStringValue) {
  metadata_source_container_->InitTestSchema();
  EXPECT_EQ(absl::OkStatus(), metadata_source_->Begin());
  EXPECT_EQ(absl::OkStatus(),
            metadata_source_->ExecuteQuery(
                absl::StrCat("INSERT INTO t1 VALUES (1, '",
                             metadata_source_->EscapeString("''"), "')"),
                nullptr));
  RecordSet expected_results = ParseTextProtoOrDie<RecordSet>(
      R"(column_names: "c1"
         column_names: "c2"
         records: { values: "1" values: "''" })");

  RecordSet query_results;
  EXPECT_EQ(absl::OkStatus(),
            metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  EXPECT_EQ(absl::OkStatus(), metadata_source_->Commit());
  EXPECT_EQ(1, query_results.records().size());
  EXPECT_THAT(query_results, EqualsProto(expected_results));
}

// Test Delete execution.
// Initialization: creates an empty table t1 (c1 INT, c2 VARCHAR(255)) with test
// schema and adds 3 rows to this table: (1,'v1'), (2,'v2'), (3, 'v3').
// Execution: Delete all the rows in the table.
// Expectation: query results are empty.
TEST_P(MetadataSourceTestSuite, TestDelete) {
  metadata_source_container_->InitSchemaAndPopulateRows();
  RecordSet query_results;
  EXPECT_EQ(absl::OkStatus(), metadata_source_->Begin());
  EXPECT_EQ(absl::OkStatus(),
            metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  EXPECT_THAT(query_results.records(), Not(IsEmpty()));

  EXPECT_EQ(absl::OkStatus(),
            metadata_source_->ExecuteQuery("DELETE FROM t1", nullptr));

  query_results.Clear();
  EXPECT_EQ(absl::OkStatus(),
            metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  EXPECT_EQ(absl::OkStatus(), metadata_source_->Commit());
  EXPECT_THAT(query_results.records(), IsEmpty());
}

// Test Update execution.
// Initialization: creates an empty table t1 (c1 INT, c2 VARCHAR(255)) with test
// schema and adds 3 rows to t1: (1,'v1'), (2,'v2'), (3, 'v3').
// Execution: Update c2 to 'v100' in rows where c1 = 1 in table t1.
// Expectation: c2 was updated to 'v100' in rows where c1 = 1 in query results.
TEST_P(MetadataSourceTestSuite, TestUpdate) {
  metadata_source_container_->InitSchemaAndPopulateRows();
  EXPECT_EQ(absl::OkStatus(), metadata_source_->Begin());
  EXPECT_EQ(absl::OkStatus(),
            metadata_source_->ExecuteQuery(
                "UPDATE t1 SET c2 = 'v100' WHERE c1 = 1", nullptr));

  RecordSet expected_results = ParseTextProtoOrDie<RecordSet>(
      R"(column_names: "c1"
         column_names: "c2"
         records: { values: "1" values: "v100" })");

  RecordSet query_results;
  EXPECT_EQ(absl::OkStatus(),
            metadata_source_->ExecuteQuery("SELECT * FROM t1 WHERE c1 = 1",
                                           &query_results));
  EXPECT_EQ(absl::OkStatus(), metadata_source_->Commit());
  EXPECT_THAT(query_results, EqualsProto(expected_results));
}

// Test query execution without open connection to metadata source.
// Expectation: returns FAILED_PRECONDITION status.
TEST_P(MetadataSourceTestSuite, TestQueryWithoutConnect) {
  absl::Status s =
      metadata_source_->ExecuteQuery("CREATE TABLE foo(bar INT)", nullptr);
  EXPECT_TRUE(absl::IsFailedPrecondition(s));
  EXPECT_THAT(std::string(s.message()), HasSubstr("No opened connection"));
}

// Test multiple queries, and rollback. Nothing should happen.
// Initialization: creates an empty table t1 (c1 INT, c2 VARCHAR(255)) with test
// schema and adds 3 rows to this table: (1,'v1'), (2,'v2'), (3, 'v3').
// Execution 1: Delete all the rows in t1 and then Rollback.
// Expectation 1: the retrieved rows remain the same.
// Execution 2: Insert 2 new rows into t1: (1, 'v1'), (2, 'v2').
// Expectation 2: 2 more rows were added in the original query results.
TEST_P(MetadataSourceTestSuite, TestMultiQueryTransaction) {
  metadata_source_container_->InitSchemaAndPopulateRows();
  RecordSet expected_results;
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->ExecuteQuery(
                                  "SELECT * FROM t1", &expected_results));
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Commit());

  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
  ASSERT_EQ(absl::OkStatus(),
            metadata_source_->ExecuteQuery("DELETE FROM t1", nullptr));
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Rollback());

  // rollback the Delete query, there should be no change in the database.
  {
    RecordSet query_results;
    ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
    ASSERT_EQ(absl::OkStatus(), metadata_source_->ExecuteQuery(
                                    "SELECT * FROM t1", &query_results));
    ASSERT_EQ(absl::OkStatus(), metadata_source_->Commit());
    EXPECT_THAT(query_results, EqualsProto(expected_results));
  }

  // now it is in a new transaction, insert two records, then commit
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->ExecuteQuery(
                                  "INSERT INTO t1 VALUES (1, 'v1')", nullptr));
  ASSERT_EQ(absl::OkStatus(), metadata_source_->ExecuteQuery(
                                  "INSERT INTO t1 VALUES (2, 'v2')", nullptr));
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Commit());

  {
    RecordSet query_results;
    ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
    EXPECT_EQ(absl::OkStatus(), metadata_source_->ExecuteQuery(
                                    "SELECT * FROM t1", &query_results));
    ASSERT_EQ(absl::OkStatus(), metadata_source_->Commit());
    int expected_num_rows = expected_results.records().size() + 2;
    EXPECT_EQ(expected_num_rows, query_results.records().size());
  }
}

// Test Insert execution with NULL values.
// Initialization: creates an empty table t1 (c1 INT, c2 VARCHAR(255)) with test
// schema.
// Execution: Insert a new row (1, NULL) into t1.
// Expectation: all the retrieved rows in t1 are (1, kMetadataSourceNull).
TEST_P(MetadataSourceTestSuite, TestNull) {
  metadata_source_container_->InitTestSchema();
  EXPECT_EQ(absl::OkStatus(), metadata_source_->Begin());
  EXPECT_EQ(absl::OkStatus(), metadata_source_->ExecuteQuery(
                                  "INSERT INTO t1 VALUES (1, NULL)", nullptr));
  RecordSet expected_results = ParseTextProtoOrDie<RecordSet>(absl::Substitute(
      R"(column_names: "c1"
           column_names: "c2"
           records: { values: "1" values: "$0" })",
      kMetadataSourceNull));

  RecordSet query_results;
  EXPECT_EQ(absl::OkStatus(),
            metadata_source_->ExecuteQuery("SELECT * FROM t1", &query_results));
  EXPECT_EQ(absl::OkStatus(), metadata_source_->Commit());
  EXPECT_EQ(1, query_results.records().size());
  EXPECT_THAT(query_results, EqualsProto(expected_results));
}

}  // namespace
}  // namespace testing
}  // namespace ml_metadata
