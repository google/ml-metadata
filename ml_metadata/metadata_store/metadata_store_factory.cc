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
#include "ml_metadata/metadata_store/transaction_executor.h"
#ifndef _WIN32
#include "ml_metadata/metadata_store/mysql_metadata_source.h"
#endif
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"
#include "ml_metadata/util/metadata_source_query_config.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ml_metadata {

namespace {

#ifndef _WIN32
tensorflow::Status CreateMySQLMetadataStore(
    const MySQLDatabaseConfig& config,
    const MigrationOptions& migration_options,
    std::unique_ptr<MetadataStore>* result) {
  auto metadata_source = absl::make_unique<MySqlMetadataSource>(config);
  auto transaction_executor =
      absl::make_unique<RdbmsTransactionExecutor>(metadata_source.get());
  TF_RETURN_IF_ERROR(MetadataStore::Create(
      util::GetMySqlMetadataSourceQueryConfig(), migration_options,
      std::move(metadata_source), std::move(transaction_executor), result));
  return (*result)->InitMetadataStoreIfNotExists(
      migration_options.enable_upgrade_migration());
}
#else
tensorflow::Status CreateMySQLMetadataStore(
    const MySQLDatabaseConfig& config,
    const MigrationOptions& migration_options,
    std::unique_ptr<MetadataStore>* result) {
  return tensorflow::errors::Unimplemented(
             "MySQL is not supported in Windows yet");
}
#endif

tensorflow::Status CreateSqliteMetadataStore(
    const SqliteMetadataSourceConfig& config,
    const MigrationOptions& migration_options,
    std::unique_ptr<MetadataStore>* result) {
  auto metadata_source = absl::make_unique<SqliteMetadataSource>(config);
  auto transaction_executor =
      absl::make_unique<RdbmsTransactionExecutor>(metadata_source.get());
  TF_RETURN_IF_ERROR(MetadataStore::Create(
      util::GetSqliteMetadataSourceQueryConfig(), migration_options,
      std::move(metadata_source), std::move(transaction_executor), result));
  return (*result)->InitMetadataStoreIfNotExists(
      migration_options.enable_upgrade_migration());
}


}  // namespace

tensorflow::Status CreateMetadataStore(const ConnectionConfig& config,
                                       const MigrationOptions& options,
                                       std::unique_ptr<MetadataStore>* result) {
  switch (config.config_case()) {
    case ConnectionConfig::CONFIG_NOT_SET:
      // TODO(b/123345695): make this longer when that bug is resolved.
      // Must specify a metadata store type.
      return tensorflow::errors::InvalidArgument("Unset");
    case ConnectionConfig::kFakeDatabase:
      // Creates an in-memory SQLite database for testing.
      return CreateSqliteMetadataStore(SqliteMetadataSourceConfig(), options,
                                       result);
    case ConnectionConfig::kMysql:
      return CreateMySQLMetadataStore(config.mysql(), options, result);
    case ConnectionConfig::kSqlite:
      return CreateSqliteMetadataStore(config.sqlite(), options, result);
    default:
      return tensorflow::errors::Unimplemented("Unknown database type.");
  }
}

tensorflow::Status CreateMetadataStore(const ConnectionConfig& config,
                                       std::unique_ptr<MetadataStore>* result) {
  return CreateMetadataStore(config, {}, result);
}

}  // namespace ml_metadata
