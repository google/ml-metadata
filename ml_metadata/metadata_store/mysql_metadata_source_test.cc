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
#include "ml_metadata/metadata_store/mysql_metadata_source.h"

#include <memory>

#include "gflags/gflags.h"
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/metadata_source_test_suite.h"
#include "ml_metadata/metadata_store/test_mysql_metadata_source_initializer.h"

DEFINE_bool(enable_sockets, true, "Whether to run socket tests.");

namespace ml_metadata {
namespace testing {
namespace {


class MysqlMetadataSourceContainer : public MetadataSourceContainer {
 public:
  MysqlMetadataSourceContainer() : MetadataSourceContainer() {
    metadata_source_initializer_ = GetTestMySqlMetadataSourceInitializer();
    // Use TCP connection type for common Mysql metadata source tests.
    metadata_source_ = metadata_source_initializer_->Init(
        TestMySqlMetadataSourceInitializer::ConnectionType::kTcp);
  }

  ~MysqlMetadataSourceContainer() override = default;

  MetadataSource* GetMetadataSource() override { return metadata_source_; }

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
  // A TestMySqlMetadataSourceInitializer from a call to
  // GetTestMySqlMetadataSourceInitializer().
  std::unique_ptr<TestMySqlMetadataSourceInitializer>
      metadata_source_initializer_;
  // An unowned MySqlMetadataSource from a call to
  // metadata_source_initializer->Init().
  MySqlMetadataSource* metadata_source_;
};

// This test is to verify we can connect to the DB using sockets.
// We use a fixtureless test because it does not share the same connection type
// with TestEscapeString below.
TEST(MySqlMetadataSourceExtendedTest, TestConnectBySocket) {
  // TODO(b/140584643) Fix MacOS Kokoro test to enable connecting via sockets.
  if (!(FLAGS_enable_sockets)) {
    GTEST_SKIP() << "Socket tests disabled.";
  }

  auto metadata_source_initializer = GetTestMySqlMetadataSourceInitializer();
  auto metadata_source = metadata_source_initializer->Init(
      TestMySqlMetadataSourceInitializer::ConnectionType::kSocket);
  ASSERT_EQ(absl::OkStatus(), metadata_source->Connect());
  ASSERT_EQ(absl::OkStatus(), metadata_source->Begin());
  ASSERT_EQ(absl::OkStatus(),
            metadata_source->ExecuteQuery(
                "CREATE TABLE t1 (c1 INT, c2 VARCHAR(255));", nullptr));
  ASSERT_EQ(absl::OkStatus(), metadata_source->Commit());
  metadata_source_initializer->Cleanup();
}



// Test EscapeString utility method.
// Same here, we adopt a fixtureless test here because it is using TCP
// connection type, different from TestConnectBySocket.
TEST(MySqlMetadataSourceExtendedTest, TestEscapeString) {
  auto metadata_source_initializer = GetTestMySqlMetadataSourceInitializer();
  auto metadata_source = metadata_source_initializer->Init(
      TestMySqlMetadataSourceInitializer::ConnectionType::kTcp);
  CHECK_EQ(absl::OkStatus(), metadata_source->Connect());
  EXPECT_EQ(metadata_source->EscapeString("''"), "\\'\\'");
  EXPECT_EQ(metadata_source->EscapeString("'\'"), "\\'\\\'");
  EXPECT_EQ(metadata_source->EscapeString("'\"text\"'"), "\\'\\\"text\\\"\\'");
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    MysqlMetadataSourceCommonTest, MetadataSourceTestSuite,
    ::testing::Values([]() {
      return std::make_unique<MysqlMetadataSourceContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
