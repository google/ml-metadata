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
#ifndef ML_METADATA_UTIL_METADATA_SOURCE_QUERY_CONFIG_H_
#define ML_METADATA_UTIL_METADATA_SOURCE_QUERY_CONFIG_H_

#include "ml_metadata/proto/metadata_source.pb.h"

namespace ml_metadata {
namespace util {

// Gets the MetadataSourceQueryConfig for MySQLMetadataSource.
MetadataSourceQueryConfig GetMySqlMetadataSourceQueryConfig();

// Gets the MetadataSourceQueryConfig for SQLiteMetadataSource.
MetadataSourceQueryConfig GetSqliteMetadataSourceQueryConfig();

// Gets the MetadataSourceQueryConfig for FakeMetadataSource.
MetadataSourceQueryConfig GetFakeMetadataSourceQueryConfig();

}  // namespace util
}  // namespace ml_metadata
#endif  // ML_METADATA_UTIL_METADATA_SOURCE_QUERY_CONFIG_H_
