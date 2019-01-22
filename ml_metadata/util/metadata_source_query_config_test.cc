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
#include "ml_metadata/util/metadata_source_query_config.h"

#include <gtest/gtest.h>

namespace ml_metadata {
namespace util {
namespace {

TEST(MetadataSourceQueryConfig, GetMySqlMetadataSourceQueryConfig) {
  const MetadataSourceQueryConfig config = GetMySqlMetadataSourceQueryConfig();
  EXPECT_EQ(config.metadata_source_type(), MYSQL_METADATA_SOURCE);
}

TEST(MetadataSourceQueryConfig, GetSqliteMetadataSourceQueryConfig) {
  const MetadataSourceQueryConfig config = GetSqliteMetadataSourceQueryConfig();
  EXPECT_EQ(config.metadata_source_type(), SQLITE_METADATA_SOURCE);
}

TEST(MetadataSourceQueryConfig, GetFakeMetadataSourceQueryConfig) {
  const MetadataSourceQueryConfig config = GetFakeMetadataSourceQueryConfig();
  EXPECT_EQ(config.metadata_source_type(), FAKE_METADATA_SOURCE);
}

}  // namespace
}  // namespace util
}  // namespace ml_metadata
