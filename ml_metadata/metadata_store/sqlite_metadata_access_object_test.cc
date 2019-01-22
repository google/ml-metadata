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
// Test suite for a SqliteMetadataSource based MetadataAccessObject.

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_access_object_test.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace testing {

namespace {

// SqliteMetadataAccessObjectContainer implements MetadataAccessObjectContainer
// to generate and retrieve a MetadataAccessObject based on a
// SqliteMetadataSource.
class SqliteMetadataAccessObjectContainer
    : public MetadataAccessObjectContainer {
 public:
  SqliteMetadataAccessObjectContainer() : MetadataAccessObjectContainer() {
    SqliteMetadataSourceConfig config;
    metadata_source_ = absl::make_unique<SqliteMetadataSource>(config);
    TF_CHECK_OK(MetadataAccessObject::Create(
        util::GetSqliteMetadataSourceQueryConfig(), metadata_source_.get(),
        &metadata_access_object_));
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
    ::testing::Values([]() {
      return absl::make_unique<SqliteMetadataAccessObjectContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata
