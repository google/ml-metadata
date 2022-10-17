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
// Test suite for a SqliteMetadataSource based RDBMSMetadataAccessObject.
#include <memory>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/query_config_executor.h"
#include "ml_metadata/metadata_store/rdbms_metadata_access_object.h"
#include "ml_metadata/metadata_store/rdbms_metadata_access_object_test.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"

namespace ml_metadata {
namespace testing {

namespace {

absl::Status CreateRDBMSMetadataAccessObject(
    const MetadataSourceQueryConfig& query_config,
    MetadataSource* const metadata_source,
    std::unique_ptr<RDBMSMetadataAccessObject>* result) {
  if (!metadata_source->is_connected())
    MLMD_RETURN_IF_ERROR(metadata_source->Connect());
  std::unique_ptr<QueryExecutor> executor =
   absl::WrapUnique(new QueryConfigExecutor(query_config, metadata_source));
  *result =
      absl::WrapUnique(new RDBMSMetadataAccessObject(std::move(executor)));
  return absl::OkStatus();
}

// SqliteRDBMSMetadataAccessObjectContainer implements
// RDBMSMetadataAccessObjectContainer to generate and retrieve a
// RDBMSMetadataAccessObject based on a SqliteMetadataSource.
class SqliteRDBMSMetadataAccessObjectContainer
    : public QueryConfigRDBMSMetadataAccessObjectContainer {
 public:
  SqliteRDBMSMetadataAccessObjectContainer()
      : QueryConfigRDBMSMetadataAccessObjectContainer(
            util::GetSqliteMetadataSourceQueryConfig()) {
    SqliteMetadataSourceConfig config;
    metadata_source_ = std::make_unique<SqliteMetadataSource>(config);
    CHECK_EQ(
        absl::OkStatus(),
        CreateRDBMSMetadataAccessObject(
            util::GetSqliteMetadataSourceQueryConfig(), metadata_source_.get(),
            &rmdbs_metadata_access_object_));
  }

  ~SqliteRDBMSMetadataAccessObjectContainer() override = default;

  MetadataSource* GetMetadataSource() override {
    return metadata_source_.get();
  }
  RDBMSMetadataAccessObject* GetRDBMSMetadataAccessObject() override {
    return rmdbs_metadata_access_object_.get();
  }

 private:
  std::unique_ptr<SqliteMetadataSource> metadata_source_;
  std::unique_ptr<RDBMSMetadataAccessObject> rmdbs_metadata_access_object_;
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    SqliteRDBMSMetadataAccessObjectTest, RDBMSMetadataAccessObjectTest,
    ::testing::Values([]() {
      return std::make_unique<SqliteRDBMSMetadataAccessObjectContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata
