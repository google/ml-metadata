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
#ifndef ML_METADATA_METADATA_STORE_POSTGRESQL_METADATA_SOURCE_H_
#define ML_METADATA_METADATA_STORE_POSTGRESQL_METADATA_SOURCE_H_

#include <string>

#include "absl/status/status.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include <libpq-fe.h>

namespace ml_metadata {

// A MetadataSource based on a PostgreSQL Database backend.
// This class is thread-unsafe.
class PostgreSQLMetadataSource : public MetadataSource {
 public:
  // Initializes the PostgreSQLMetadataSource with given config.
  // Check-fails if config is invalid.
  explicit PostgreSQLMetadataSource(const PostgreSQLDatabaseConfig& config);

  // Disallow copy and assign.
  PostgreSQLMetadataSource(const PostgreSQLMetadataSource&) = delete;
  PostgreSQLMetadataSource& operator=(const PostgreSQLMetadataSource&) = delete;

  ~PostgreSQLMetadataSource() override;

  // Escape strings with backslashes for special characters for PostgreSQL. The
  // implementation uses PQescapeLiteral in PostgreSQL C API. It aborts if
  // the metadata source is not connected.
  std::string EscapeString(absl::string_view value) const final;

  // Proto string is provided, performs base64 encoding for PostgreSQL.
  std::string EncodeBytes(absl::string_view value) const final;

  // DecodeBytes can return an absl::Status if decoding fails
  absl::StatusOr<std::string> DecodeBytes(absl::string_view value) const final;

  std::string GetDbName() const;

 private:
  // Converts the PGresult in `pg_result_` to `record_set_out`.
  absl::Status ConvertResultToRecordSet(PGresult* res,
                                        RecordSet* record_set_ptr);

  // Runs the given query and stores the PGresult in pg_result_.
  // Any existing PGresult in `pg_result_` is cleaned up prior to issuing
  // the given query.
  // Returns an INTERNAL error upon any errors from the PostgreSQL backend.
  absl::Status RunPostgresqlStatement(const std::string& query);

  // Executes a SQL statement and returns the rows if any.
  // Returns an INTERNAL error upon any errors from the PostgreSQL backend.
  absl::Status ExecuteQueryImpl(const std::string& query,
                                RecordSet* results) final;

  // Create PostgreSQL connection based on database config and whether to use
  // default db. PostgreSQL connection requires providing a dbname, however,
  // the MLMD db might not exist when connection happens. So we need to connect
  // to the default db first using this function.
  void extracted(const PostgreSQLDatabaseConfig& config, bool& use_default_db,
                 std::string& connection_config);
  PGconn* ConnectToPostgreSQLDb(const PostgreSQLDatabaseConfig& config,
                                bool use_default_db);

  // Connects to the PostgreSQL backend specified in options_.
  // PostgreSQL connection requires providing a dbname, however, the MLMD db
  // might not exist when connection happens. In that case, this function will
  // connect to default db first to create the desired db, then connect to the
  // desired db.
  // Returns an INTERNAL error upon any errors from the PostgreSQL backend.
  absl::Status ConnectImpl() final;

  // Closes the existing open connection to the PostgreSQL backend.
  // Any existing PGresult in `pg_result_` is also cleaned up.
  absl::Status CloseImpl() final;

  // Opens a transaction.
  absl::Status BeginImpl() final;

  // Commits the currently open transaction.
  absl::Status CommitImpl() final;

  // Rollbacks the currently open transaction.
  absl::Status RollbackImpl() final;


  // Discards any existing PGresult in `pg_result_`.
  void DiscardResultSet();

  // The PGresult from the previously executed query in RunQuery.
  PGresult* pg_result_ = nullptr;

  // Config to connect to the PostgreSQL backend.
  const PostgreSQLDatabaseConfig config_;

  // database_name_ stores the last database that the MetadataSource has been
  // connected to.
  std::string database_name_;

  PGconn* conn_ = nullptr;
};

std::string buildConnectionConfig(const PostgreSQLDatabaseConfig& config,
                                  bool& use_default_db);

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_POSTGRESQL_METADATA_SOURCE_H_
