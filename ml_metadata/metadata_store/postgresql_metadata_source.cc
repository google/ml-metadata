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
#include "ml_metadata/metadata_store/postgresql_metadata_source.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source_util.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/util/return_utils.h"
#include <libpq-fe.h>

namespace ml_metadata {

namespace {

constexpr absl::string_view kBeginTransaction = "BEGIN";
constexpr absl::string_view kCommitTransaction = "COMMIT";
constexpr absl::string_view kRollbackTransaction = "ROLLBACK";

// Checks if config is valid.
absl::Status CheckConfig(const PostgreSQLDatabaseConfig& config) {
  std::vector<std::string> config_errors;
  if (config.host().empty() == config.hostaddr().empty()) {
    config_errors.push_back(
        "exactly one of host or hostaddr must be specified");
  }
  if (config.port().empty()) {
    config_errors.push_back("port must not be empty");
  }
  if (config.dbname().empty()) {
    config_errors.push_back("dbname must not be empty");
  }
  if (config.user().empty()) {
    config_errors.push_back("user parameter must not be empty");
  }

  if (!config_errors.empty()) {
    return absl::InvalidArgumentError(absl::StrJoin(config_errors, ";"));
  }
  return absl::OkStatus();
}

// Builds absl::Status for the `status_code` and attaches `PQresultErrorMessage`
// to the payload of the status object.
absl::Status BuildErrorStatus(const absl::StatusCode status_code,
                              absl::string_view error_message) {
  return absl::Status(
      status_code,
      absl::StrCat("PostgreSQL metadata source error: ", error_message));
}
}  // namespace

PostgreSQLMetadataSource::PostgreSQLMetadataSource(
    const PostgreSQLDatabaseConfig& config)
    : MetadataSource(), config_(config) {
  CHECK_EQ(CheckConfig(config), absl::OkStatus());
}

PostgreSQLMetadataSource::~PostgreSQLMetadataSource() {
  CHECK_EQ(CloseImpl(), absl::OkStatus());
}

std::string PostgreSQLMetadataSource::GetDbName() const {
  return database_name_;
}

absl::Status PostgreSQLMetadataSource::ConvertResultToRecordSet(
    PGresult* res, RecordSet* record_set_ptr = nullptr) {
  if (record_set_ptr == nullptr || res == nullptr) {
    return absl::OkStatus();
  }

  int num_rows = PQntuples(res);
  int num_cols = PQnfields(res);
  bool is_column_name_initted =
      (record_set_ptr->column_names_size() == num_cols);
  if (!is_column_name_initted) {
    record_set_ptr->clear_column_names();
  }

  for (int i = 0; i < num_rows; i++) {
    RecordSet::Record new_record;
    for (int j = 0; j < num_cols; j++) {
      if (i == 0 && !is_column_name_initted) {
        record_set_ptr->add_column_names(PQfname(res, j));
      }
      new_record.add_values(PQgetisnull(res, i, j) ? kMetadataSourceNull.data()
                                                   : PQgetvalue(res, i, j));
    }
    record_set_ptr->mutable_records()->Add(std::move(new_record));
  }
  return absl::OkStatus();
}

absl::Status PostgreSQLMetadataSource::RunPostgresqlStatement(
    const std::string& query) {
  DiscardResultSet();
  PGresult* res = PQexec(conn_, query.c_str());
  if (PQresultStatus(res) != PGRES_COMMAND_OK &&
      PQresultStatus(res) != PGRES_TUPLES_OK) {
    const std::string error_str = std::string(PQresultErrorMessage(res));
    LOG(ERROR) << "Execution failed: " << error_str;
    PQclear(res);
    MLMD_RETURN_IF_ERROR(
        BuildErrorStatus(absl::StatusCode::kInternal, error_str));
  }

  pg_result_ = res;
  return absl::OkStatus();
}

absl::Status PostgreSQLMetadataSource::ExecuteQueryImpl(
    const std::string& query, RecordSet* results) {
  // Run the query.
  MLMD_RETURN_IF_ERROR(RunPostgresqlStatement(query));

  // If the query is successful, convert the results.
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertResultToRecordSet(pg_result_, results),
      "ConvertResultToRecordSet for query", query);

  return absl::OkStatus();
}

std::string buildConnectionConfig(const PostgreSQLDatabaseConfig& config,
                                  bool& use_default_db) {
  std::string connection_config;
  std::string dbname = "postgres";  // Default DB Name for PostgreSQL.
  if (!use_default_db) {
    dbname = config.dbname().data();
  }

  if (!dbname.empty()) {
    absl::StrAppend(&connection_config, " dbname=", dbname);
  }

  if (config.has_host()) {
    absl::StrAppend(&connection_config, " host=", config.host().data());
  }
  if (config.has_hostaddr()) {
    absl::StrAppend(&connection_config, " hostaddr=", config.hostaddr().data());
  }
  if (config.has_port()) {
    absl::StrAppend(&connection_config, " port=", config.port().data());
  }
  if (config.has_user()) {
    absl::StrAppend(&connection_config, " user=", config.user().data());
  }
  if (config.has_password()) {
    absl::StrAppend(&connection_config, " password=", config.password().data());
  }
  if (config.has_passfile()) {
    absl::StrAppend(&connection_config, " passfile=", config.passfile().data());
  }
  if (config.ssloption().has_sslmode()) {
    absl::StrAppend(&connection_config,
                    " sslmode=", config.ssloption().sslmode().data());
  }
  if (config.ssloption().has_sslcert()) {
    absl::StrAppend(&connection_config,
                    " sslcert=", config.ssloption().sslcert().data());
  }
  if (config.ssloption().has_sslkey()) {
    absl::StrAppend(&connection_config,
                    " sslkey=", config.ssloption().sslkey().data());
  }
  if (config.ssloption().has_sslpassword()) {
    absl::StrAppend(&connection_config,
                    " sslpassword=", config.ssloption().sslpassword().data());
  }
  if (config.ssloption().has_sslrootcert()) {
    absl::StrAppend(&connection_config,
                    " sslrootcert=", config.ssloption().sslrootcert().data());
  }

  return connection_config;
}
PGconn* PostgreSQLMetadataSource::ConnectToPostgreSQLDb(
    const PostgreSQLDatabaseConfig& config, bool use_default_db = false) {
  std::string connection_config = buildConnectionConfig(config, use_default_db);

  LOG(INFO) << "Connecting to database. ";
  PGconn* conn = PQconnectdb(connection_config.c_str());
  if (PQstatus(conn) != CONNECTION_OK) {
    LOG(ERROR) << "PostgreSQL error: " << PQerrorMessage(conn);
    PQfinish(conn);
    return nullptr;
  } else {
    LOG(INFO) << "Connection to database succeed.";
  }
  return conn;
}

absl::Status PostgreSQLMetadataSource::ConnectImpl() {
  if (!config_.skip_db_creation()) {
    PGconn* connDefault =
        ConnectToPostgreSQLDb(config_, /*use_default_db=*/true);
    if (connDefault == nullptr) {
      MLMD_RETURN_IF_ERROR(BuildErrorStatus(absl::StatusCode::kInternal,
                                            PQerrorMessage(connDefault)));
    }

    // Check whether the target DB already exists.
    const std::string check_database_command = absl::Substitute(
        "SELECT datname FROM pg_catalog.pg_database WHERE lower(datname) = "
        "lower('$0');",
        config_.dbname().data());
    PGresult* res_check = PQexec(connDefault, check_database_command.c_str());
    if (PQresultStatus(res_check) != PGRES_COMMAND_OK &&
        PQresultStatus(res_check) != PGRES_TUPLES_OK) {
      const std::string error_str =
          std::string(PQresultErrorMessage(res_check));
      LOG(ERROR) << "Checking database existence for " << config_.dbname()
                 << " failure: " << error_str;
      PQclear(res_check);
      PQfinish(connDefault);
      MLMD_RETURN_IF_ERROR(
          BuildErrorStatus(absl::StatusCode::kInternal, error_str));
    }

    // Create Database if not exists.
    RecordSet record_set;
    absl::Status databaseExistenceStatus =
        ConvertResultToRecordSet(res_check, &record_set);
    PQclear(res_check);
    MLMD_RETURN_IF_ERROR(databaseExistenceStatus);
    if (record_set.records_size() == 0) {
      const std::string create_database_cmd =
          absl::Substitute("CREATE DATABASE $0;", config_.dbname().data());
      PGresult* res = PQexec(connDefault, create_database_cmd.c_str());
      if (PQresultStatus(res) != PGRES_COMMAND_OK &&
          PQresultStatus(res) != PGRES_TUPLES_OK) {
        const std::string error_create = std::string(PQresultErrorMessage(res));
        LOG(ERROR) << "Creating database " << config_.dbname()
                   << " failure: " << error_create;
        PQclear(res);
        PQfinish(connDefault);
        MLMD_RETURN_IF_ERROR(
            BuildErrorStatus(absl::StatusCode::kInternal, error_create));
      }
      PQclear(res);
    }

    PQfinish(connDefault);
  }

  PGconn* conn = ConnectToPostgreSQLDb(config_, /*use_default_db=*/false);
  conn_ = conn;
  database_name_ = config_.dbname();

  return absl::OkStatus();
}

absl::Status PostgreSQLMetadataSource::CloseImpl() {
  if (conn_ != nullptr) {
    DiscardResultSet();
    PQfinish(conn_);
    conn_ = nullptr;
  }
  return absl::OkStatus();
}

absl::Status PostgreSQLMetadataSource::BeginImpl() {
  return RunPostgresqlStatement(kBeginTransaction.data());
}

absl::Status PostgreSQLMetadataSource::CommitImpl() {
  return RunPostgresqlStatement(kCommitTransaction.data());
}

absl::Status PostgreSQLMetadataSource::RollbackImpl() {
  return RunPostgresqlStatement(kRollbackTransaction.data());
}

void PostgreSQLMetadataSource::DiscardResultSet() {
  if (pg_result_ != nullptr) {
    PQclear(pg_result_);
    pg_result_ = nullptr;
  }
}

std::string PostgreSQLMetadataSource::EscapeString(
    absl::string_view value) const {
  if (conn_ == nullptr || value.data() == nullptr) {
    LOG(ERROR) << "Connection `conn_` is null or input parameter `value` is "
                  "null when calling EscapeString().";
    return "";
  }

  char* escaped_str = PQescapeLiteral(conn_, value.data(), value.size());
  std::string result{escaped_str};
  // PQescapeLiteral will wrap the escaped string in '', which is redundant to
  // the existing MLMD syntax. Therefore stripping the outer '' from the escaped
  // string.
  std::string substring = result.substr(1, std::strlen(result.data()) - 2);
  PQfreemem(escaped_str);
  return substring;
}

std::string PostgreSQLMetadataSource::EncodeBytes(
    absl::string_view value) const {
  // Reuse SQLite utility to encode base64.
  return SqliteEncodeBytes(value);
}
absl::StatusOr<std::string> PostgreSQLMetadataSource::DecodeBytes(
    absl::string_view value) const {
  // Reuse SQLite utility to decode base64.
  return SqliteDecodeBytes(value);
}


}  // namespace ml_metadata
