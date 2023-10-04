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
// Test suite for a MySqlMetadataSource based MetadataAccessObject.

#include <cstdint>
#include <memory>
#include <optional>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "ml_metadata/metadata_store/metadata_access_object_factory.h"
#include "ml_metadata/metadata_store/metadata_access_object_test.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/mysql_metadata_source.h"
#include "ml_metadata/metadata_store/test_mysql_metadata_source_initializer.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"

namespace ml_metadata {
namespace testing {

namespace {

// MySqlMetadataAccessObjectContainer implements MetadataAccessObjectContainer
// to generate and retrieve a MetadataAccessObject based off a
// MySqlMetadataSource.
class MySqlMetadataAccessObjectContainer
    : public QueryConfigMetadataAccessObjectContainer {
 public:
  MySqlMetadataAccessObjectContainer(
      std::optional<int64_t> earlier_schema_version = absl::nullopt)
      : QueryConfigMetadataAccessObjectContainer(
            util::GetMySqlMetadataSourceQueryConfig(), earlier_schema_version) {
    metadata_source_initializer_ = GetTestMySqlMetadataSourceInitializer();
    metadata_source_ = metadata_source_initializer_->Init(
        TestMySqlMetadataSourceInitializer::ConnectionType::kTcp);
    CHECK_EQ(absl::OkStatus(),
             CreateMetadataAccessObject(
                 util::GetMySqlMetadataSourceQueryConfig(), metadata_source_,
                 earlier_schema_version, &metadata_access_object_));
  }

  ~MySqlMetadataAccessObjectContainer() override {
    metadata_source_initializer_->Cleanup();
  };

  MetadataSource* GetMetadataSource() override { return metadata_source_; }
  MetadataAccessObject* GetMetadataAccessObject() override {
    return metadata_access_object_.get();
  }

  std::string GetTableNumQuery() final {
    return "select count(*) from `information_schema`.`tables` where "
           "`table_schema`=(SELECT DATABASE())";
  }

  std::string GetIndexNumQuery() final {
    return "select count(*) from `information_schema`.`statistics` where "
           "`table_schema`=(SELECT DATABASE())";
  }

 private:
  // An unowned TestMySqlMetadataSourceInitializer from a call to
  // GetTestMySqlMetadataSourceInitializer().
  std::unique_ptr<TestMySqlMetadataSourceInitializer>
      metadata_source_initializer_;
  // An unowned MySqlMetadataSource from a call to
  // metadata_source_initializer->Init().
  MySqlMetadataSource* metadata_source_;
  std::unique_ptr<MetadataAccessObject> metadata_access_object_;
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    MySqlMetadataAccessObjectTest, MetadataAccessObjectTest,
    ::testing::Values(
        []() { return std::make_unique<MySqlMetadataAccessObjectContainer>(); },
        // TODO(b/257334039) Cleanup after V10+ migration
        []() {
          return std::make_unique<MySqlMetadataAccessObjectContainer>(
              /*earlier_schema_version=*/9);
        },
        []() {
          return std::make_unique<MySqlMetadataAccessObjectContainer>(
              /*earlier_schema_version=*/8);
        },
        []() {
          return std::make_unique<MySqlMetadataAccessObjectContainer>(
              /*earlier_schema_version=*/7);
        }));

}  // namespace testing
}  // namespace ml_metadata
