/* Copyright 2021 Google LLC

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
// Test suite for a SqliteMetadataSource based MetadataAccessObject.

#include <cstdint>
#include <memory>
#include <optional>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "ml_metadata/metadata_store/metadata_access_object_factory.h"
#include "ml_metadata/metadata_store/metadata_access_object_test.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"

namespace ml_metadata {
namespace testing {

namespace {

// SqliteMetadataAccessObjectContainer implements MetadataAccessObjectContainer
// to generate and retrieve a MetadataAccessObject based on a
// SqliteMetadataSource.
class SqliteMetadataAccessObjectContainer
    : public QueryConfigMetadataAccessObjectContainer {
 public:
  SqliteMetadataAccessObjectContainer(
      std::optional<int64_t> earlier_schema_version = absl::nullopt)
      : QueryConfigMetadataAccessObjectContainer(
            util::GetSqliteMetadataSourceQueryConfig(),
            earlier_schema_version) {
    SqliteMetadataSourceConfig config;
    metadata_source_ = std::make_unique<SqliteMetadataSource>(config);
    CHECK_EQ(
        absl::OkStatus(),
        CreateMetadataAccessObject(
            util::GetSqliteMetadataSourceQueryConfig(), metadata_source_.get(),
            earlier_schema_version, &metadata_access_object_));
  }

  ~SqliteMetadataAccessObjectContainer() override = default;

  MetadataSource* GetMetadataSource() override {
    return metadata_source_.get();
  }
  MetadataAccessObject* GetMetadataAccessObject() override {
    return metadata_access_object_.get();
  }

 private:
  std::unique_ptr<SqliteMetadataSource> metadata_source_;
  std::unique_ptr<MetadataAccessObject> metadata_access_object_;
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    SqliteMetadataAccessObjectTest, MetadataAccessObjectTest,
    ::testing::Values(
        []() {
          return std::make_unique<SqliteMetadataAccessObjectContainer>();
        },
        // TODO(b/257334039) Cleanup after V10+ migration
        []() {
          return std::make_unique<SqliteMetadataAccessObjectContainer>(
              /*earlier_schema_version=*/9);
        },
        []() {
          return std::make_unique<SqliteMetadataAccessObjectContainer>(
              /*earlier_schema_version=*/8);
        },
        []() {
          return std::make_unique<SqliteMetadataAccessObjectContainer>(
              /*earlier_schema_version=*/7);
        }));

}  // namespace testing
}  // namespace ml_metadata
