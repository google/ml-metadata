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

#include "ml_metadata/metadata_store/postgresql_metadata_source.h"

#include "gflags/gflags.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ml_metadata/metadata_store/metadata_source_test_suite.h"
#include "ml_metadata/metadata_store/test_postgresql_metadata_source_initializer.h"

namespace ml_metadata {
namespace testing {
namespace {

// This container calls initializer to set up PostgreSQL database. After DB
// becomes available, this container will get metadata source from initializer.
// Then schema for testing can be initialized by this container.
class PostgreSQLMetadataSourceContainer : public MetadataSourceContainer {
 public:
  PostgreSQLMetadataSourceContainer() : MetadataSourceContainer() {
    metadata_source_initializer_ = GetTestPostgreSQLMetadataSourceInitializer();
    metadata_source_ = metadata_source_initializer_->Init();
  }

  ~PostgreSQLMetadataSourceContainer() override {
    metadata_source_initializer_->Cleanup();
  }

  MetadataSource* GetMetadataSource() override { return metadata_source_; }

  // Creates a new table t1(c1 INT, c2 VARCHAR(255)).
  void InitTestSchema() override {
    CHECK_EQ(metadata_source_->Connect(), absl::OkStatus());
    CHECK_EQ(metadata_source_->Begin(), absl::OkStatus());
    CHECK_EQ(metadata_source_->ExecuteQuery(
                 "CREATE TABLE t1 (c1 INT, c2 VARCHAR(255));", nullptr),
             absl::OkStatus());
    CHECK_EQ(metadata_source_->Commit(), absl::OkStatus());
  }

  // Creates table t1(c1 INT, c2 VARCHAR(255)) and
  // adds 3 rows to this table: (1,'v1'), (2,'v2'), (3, 'v3').
  void InitSchemaAndPopulateRows() override {
    InitTestSchema();
    CHECK_EQ(metadata_source_->Begin(), absl::OkStatus());
    CHECK_EQ(metadata_source_->ExecuteQuery("INSERT INTO t1 VALUES (1, 'v1')",
                                            nullptr),
             absl::OkStatus());
    CHECK_EQ(metadata_source_->ExecuteQuery("INSERT INTO t1 VALUES (2, 'v2')",
                                            nullptr),
             absl::OkStatus());
    CHECK_EQ(metadata_source_->ExecuteQuery("INSERT INTO t1 VALUES (3, 'v3')",
                                            nullptr),
             absl::OkStatus());
    CHECK_EQ(metadata_source_->Commit(), absl::OkStatus());
  }

 private:
  // A TestPostgreSQLMetadataSourceInitializer from a call to
  // GetTestPostgreSQLMetadataSourceInitializer().
  std::unique_ptr<TestPostgreSQLMetadataSourceInitializer>
      metadata_source_initializer_;

  PostgreSQLMetadataSource* metadata_source_;
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    PostgreSQLMetadataSourceTest, MetadataSourceTestSuite,
    ::testing::Values([]() {
      return std::make_unique<PostgreSQLMetadataSourceContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
