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
#include "ml_metadata/metadata_store/metadata_access_object_factory.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"
#include "ml_metadata/util/status_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

// Explicitly checks CreateMetadataAccessObject. Tests it with
// SQLite and replicates InitMetadataSourceCheckSchemaVersion from the
// MetadataAccessObjectTest.
TEST(MetadataAccessObjectFactory, CreateMetadataAccessObject) {
  SqliteMetadataSourceConfig config;
  std::unique_ptr<MetadataSource> metadata_source =
      absl::make_unique<SqliteMetadataSource>(config);
  std::unique_ptr<MetadataAccessObject> metadata_access_object;
  TF_ASSERT_OK(CreateMetadataAccessObject(
      util::GetSqliteMetadataSourceQueryConfig(), metadata_source.get(),
      &metadata_access_object));
  TF_ASSERT_OK(FromABSLStatus(metadata_source->Begin()));
  TF_ASSERT_OK(metadata_access_object->InitMetadataSource());

  int64 schema_version;
  TF_ASSERT_OK(metadata_access_object->GetSchemaVersion(&schema_version));
  TF_ASSERT_OK(FromABSLStatus(metadata_source->Commit()));

  int64 library_version = metadata_access_object->GetLibraryVersion();
  EXPECT_EQ(schema_version, library_version);
}

}  // namespace
}  // namespace ml_metadata
