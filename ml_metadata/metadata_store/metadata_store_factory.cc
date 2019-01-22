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
#include "ml_metadata/metadata_store/metadata_store_factory.h"

#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/mysql_metadata_source.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"
#include "ml_metadata/util/metadata_source_query_config.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ml_metadata {

namespace {

tensorflow::Status CreateMySQLMetadataStore(
    const MySQLDatabaseConfig& config, std::unique_ptr<MetadataStore>* result) {
  TF_RETURN_IF_ERROR(MetadataStore::Create(
      util::GetMySqlMetadataSourceQueryConfig(),
      absl::make_unique<MySqlMetadataSource>(config), result));
  return (*result)->InitMetadataStoreIfNotExists();
}

tensorflow::Status CreateSqliteMetadataStore(
    const SqliteMetadataSourceConfig& config,
    std::unique_ptr<MetadataStore>* result) {
  TF_RETURN_IF_ERROR(MetadataStore::Create(
      util::GetSqliteMetadataSourceQueryConfig(),
      absl::make_unique<SqliteMetadataSource>(config), result));
  return (*result)->InitMetadataStoreIfNotExists();
}

}  // namespace

tensorflow::Status CreateMetadataStore(const ConnectionConfig& config,
                                       std::unique_ptr<MetadataStore>* result) {
  switch (config.config_case()) {
    case ConnectionConfig::CONFIG_NOT_SET:

      return tensorflow::errors::InvalidArgument("Set exactly one config.");
    case ConnectionConfig::kFakeDatabase:
      // Creates an in-memory SQLite database for testing.
      return CreateSqliteMetadataStore(SqliteMetadataSourceConfig(), result);
    case ConnectionConfig::kMysql:
      return CreateMySQLMetadataStore(config.mysql(), result);
    case ConnectionConfig::kSqlite:
      return CreateSqliteMetadataStore(config.sqlite(), result);
    default:
      return tensorflow::errors::Unimplemented("Unknown database type.");
  }
}

}  // namespace ml_metadata
