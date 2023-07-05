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
#include "ml_metadata/metadata_store/mysql_metadata_source.h"

#include <string>
#include <utility>

#include <glog/logging.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source_util.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/return_utils.h"
#include "mysql.h"

namespace ml_metadata {

namespace {

using absl::Status;

constexpr absl::string_view kSetIsolationLevel =
    "SET TRANSACTION ISOLATION LEVEL ";
constexpr absl::string_view kBeginTransaction = "START TRANSACTION";
constexpr absl::string_view kCommitTransaction = "COMMIT";
constexpr absl::string_view kRollbackTransaction = "ROLLBACK";

// url key used for storing ustom error information in the absl::Status payload.
constexpr absl::string_view kStatusErrorInfoUrl = "mysql-error-info";

// A class that invokes mysql_thread_init() when constructed, and
// mysql_thread_end() when destructed.  It can be used as a
// thread_local to ensure that this happens exactly once per thread
// (see ThreadInitAccess).
class ThreadInitializer {
 public:
  ThreadInitializer() : initialized_(!mysql_thread_init()) {
    // Note: mysql_thread_init returns 0 on success
  }

  // Disallow copy-and-assign.
  ThreadInitializer(const ThreadInitializer&) = delete;
  ThreadInitializer& operator=(const ThreadInitializer&) = delete;

  ~ThreadInitializer() {
    if (initialized_) mysql_thread_end();
  }

  bool initialized() const { return initialized_; }

 private:
  // Indicates whether the call to mysql_thread_init was successfull.
  const bool initialized_;
};

// Initializes a ThreadInitializer for the current thread (if not already done).
// Returns an error if the MYSQL thread initialization fails.
//
// Note: All top-level methods that are invoked by the MetadataSource base class
// need to invoke ThreadInitAccess.
Status ThreadInitAccess() {
  // Cause mysql_thread_init() to be invoked exactly once per thread
  // the first time ThreadInitAccess() is called, and
  // mysql_thread_end() exactly once when the thread dies.
  thread_local ThreadInitializer initializer;
  if (!initializer.initialized()) {
    return absl::InternalError("mysql thread initialization not done");
  }
  return absl::OkStatus();
}

// Checks if config is valid.
Status CheckConfig(const MySQLDatabaseConfig& config) {
  std::vector<std::string> config_errors;
  if (config.host().empty() == config.socket().empty()) {
    config_errors.push_back("exactly one of host or socket must be specified");
  }
  if (config.database().empty()) {
    config_errors.push_back("database must not be empty");
  }

  if (!config_errors.empty()) {
    return absl::InvalidArgumentError(absl::StrJoin(config_errors, ";"));
  }
  return absl::OkStatus();
}

// Builds absl::Status for the `status_code` and attaches `mysql_error_code`
// to the payload of the status object.
absl::Status BuildErrorStatus(const absl::StatusCode status_code,
                              absl::string_view error_message,
                              const int64_t mysql_error_code,
                              absl::string_view mysql_error_message) {
  auto error_status = absl::Status(
      status_code, absl::StrCat(error_message, ": errno: ", mysql_error_message,
                                ", error: ", mysql_error_message));
  MySQLSourceErrorInfo error_info;
  error_info.set_mysql_error_code(mysql_error_code);
  error_status.SetPayload(kStatusErrorInfoUrl,
                          absl::Cord(error_info.SerializeAsString()));
  return error_status;
}
}  // namespace

MySqlMetadataSource::MySqlMetadataSource(const MySQLDatabaseConfig& config)
    : MetadataSource(), config_(config) {
  CHECK_EQ(absl::OkStatus(), CheckConfig(config));
}

MySqlMetadataSource::~MySqlMetadataSource() {
  CHECK_EQ(absl::OkStatus(), CloseImpl());
}

Status MySqlMetadataSource::ConnectImpl() {

  // Initialize the MYSQL object.
  db_ = mysql_init(nullptr);
  if (!db_) {
    LOG(ERROR) << "MySQL error: " << mysql_errno(db_) << ": "
               << mysql_error(db_);
    return BuildErrorStatus(absl::StatusCode::kInternal, "mysql_init failed",
                            mysql_errno(db_), mysql_error(db_));
  }

  // Explicitly setup the thread-local initializer.
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(ThreadInitAccess(),
                                    "MySql thread init failed at ConnectImpl");

  // Set connection options
  if (config_.has_ssl_options()) {
    const MySQLDatabaseConfig::SSLOptions& ssl = config_.ssl_options();
    // The method set mysql_options, and always return 0. The connection options
    // are used in the `mysql_real_connect`.
    mysql_ssl_set(db_, ssl.key().empty() ? nullptr : ssl.key().c_str(),
                  ssl.cert().empty() ? nullptr : ssl.cert().c_str(),
                  ssl.ca().empty() ? nullptr : ssl.ca().c_str(),
                  ssl.capath().empty() ? nullptr : ssl.capath().c_str(),
                  ssl.cipher().empty() ? nullptr : ssl.cipher().c_str());
    my_bool verify_server_cert = ssl.verify_server_cert() ? 1 : 0;
    mysql_options(db_, MYSQL_OPT_SSL_VERIFY_SERVER_CERT, &verify_server_cert);
  }

  mysql_options(db_, MYSQL_DEFAULT_AUTH, "mysql_native_password");
  // Connect to the MYSQL server.
  db_ = mysql_real_connect(
          db_, config_.host().empty() ? nullptr : config_.host().c_str(),
          config_.user().empty() ? nullptr : config_.user().c_str(),
          config_.password().empty() ? nullptr : config_.password().c_str(),
          /*db=*/nullptr, config_.port(),
          config_.socket().empty() ? nullptr : config_.socket().c_str(),
          /*clientflag=*/0UL);

  if (!db_) {
    LOG(ERROR)
        << "MySQL database was not initialized. Please ensure your "
           "MySQL server is running. Also, this error might be caused by "
           "starting from MySQL 8.0, mysql_native_password used by MLMD is not "
           "supported as a default for authentication plugin. Please follow "
           "<https://dev.mysql.com/blog-archive/"
           "upgrading-to-mysql-8-0-default-authentication-plugin-"
           "considerations/>"
           "to fix this issue.";
    return BuildErrorStatus(absl::StatusCode::kInternal,
                            "mysql_real_connect failed", mysql_errno(db_),
                            mysql_error(db_));
  }

  // Return an error if the default storage engine doesn't support transactions.
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
      CheckTransactionSupport(),
      "checking transaction support of default storage engine");


  // Create the database if not already present and skip_db_creation is false.
  if (!config_.skip_db_creation()) {
    const std::string create_database_cmd =
        absl::StrCat("CREATE DATABASE IF NOT EXISTS ", config_.database());
    MLMD_RETURN_WITH_CONTEXT_IF_ERROR(RunQuery(create_database_cmd),
                                      "Creating database ", config_.database(),
                                      " in ConnectImpl");
  }

  // Switch to the database.
  if (database_name_.empty()) {
    database_name_ = config_.database();
  }
  const std::string use_database_cmd = absl::StrCat("USE ", database_name_);
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(RunQuery(use_database_cmd),
                                    "Changing to database ", database_name_,
                                    " in ConnectImpl");

  return absl::OkStatus();
}

Status MySqlMetadataSource::CloseImpl() {
  if (db_ != nullptr) {
    MLMD_RETURN_IF_ERROR(ThreadInitAccess());
    DiscardResultSet();
    mysql_close(db_);
    db_ = nullptr;
  }
  return absl::OkStatus();
}

Status MySqlMetadataSource::ExecuteQueryImpl(const std::string& query,
                                             RecordSet* results) {
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
      ThreadInitAccess(), "MySql thread init failed at ExecuteQueryImpl");

  // Run the query.
  MLMD_RETURN_IF_ERROR(RunQuery(query));

  // If query is successfull, convert the results.
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(ConvertMySqlRowSetToRecordSet(results),
                                    "ConvertMySqlRowSetToRecordSet for query ",
                                    query);
  return absl::OkStatus();
}

Status MySqlMetadataSource::CommitImpl() {
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(ThreadInitAccess(),
                                    "MySql thread init failed at CommitImpl");
  return RunQuery(kCommitTransaction.data());
}

Status MySqlMetadataSource::RollbackImpl() {
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(ThreadInitAccess(),
                                    "MySql thread init failed at RollbackImpl");

  return RunQuery(kRollbackTransaction.data());
}

Status MySqlMetadataSource::BeginImpl() {
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(ThreadInitAccess(),
                                    "MySql thread init failed at BeginImpl");

  return RunQuery(kBeginTransaction.data());
}


Status MySqlMetadataSource::CheckTransactionSupport() {
  constexpr absl::string_view kCheckTransactionSupport =
      "SELECT ENGINE, TRANSACTIONS FROM INFORMATION_SCHEMA.ENGINES WHERE "
      "ENGINE=(SELECT @@default_storage_engine)";
  MLMD_RETURN_IF_ERROR(RunQuery(kCheckTransactionSupport.data()));

  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(ConvertMySqlRowSetToRecordSet(&record_set));
  if (record_set.records_size() != 1 ||
      record_set.records(0).values_size() != 2) {
    return absl::InternalError(
        absl::StrCat("Expected query ", kCheckTransactionSupport,
                     " to generate exactly single row with 2 columns, but got ",
                     record_set.DebugString()));
  }
  const RecordSet::Record& record = record_set.records(0);
  if (record.values(1) != "YES") {
    return absl::InternalError(
        absl::StrCat("no transaction support for default_storage_engine ",
                     record.values(0)));
  }

  return absl::OkStatus();
}

Status MySqlMetadataSource::RunQuery(const std::string& query) {
  DiscardResultSet();

  int query_status = mysql_query(db_, query.c_str());
  if (query_status) {
    int64_t error_number = mysql_errno(db_);
    // 2006: sever closes the connection due to inactive client;
    // client reports server has gone away, we reconnect the server for the
    // client if the query is begin transaction.
    if (error_number == 2006 && query == kBeginTransaction) {
      MLMD_RETURN_IF_ERROR(CloseImpl());
      MLMD_RETURN_IF_ERROR(ConnectImpl());

      return RunQuery(query);
    }
    // When running concurrent transactions for error codes:
    // 1213: Deadlock detection on wait lock.
    // 1205: Lock wait timeout.
    // returns Aborted for client side to retry.
    if (error_number == 1213 || error_number == 1205) {
      return BuildErrorStatus(absl::StatusCode::kAborted, "mysql_query aborted",
                              error_number, mysql_error(db_));
    }

    return BuildErrorStatus(absl::StatusCode::kInternal, "mysql_query failed",
                            error_number, mysql_error(db_));
  }
  // Updated database_name_ if the incoming query was "USE <database>" query and
  // run successfully.
  MaybeUpdateDatabaseNameFromQuery(query);

  result_set_ = mysql_store_result(db_);
  if (!result_set_ && mysql_field_count(db_) != 0) {
    return BuildErrorStatus(
        absl::StatusCode::kInternal,
        absl::StrCat("mysql_query ", query,
                     "returned an unexpected NULL result_set"),
        mysql_errno(db_), mysql_error(db_));
  }

  return absl::OkStatus();
}

void MySqlMetadataSource::MaybeUpdateDatabaseNameFromQuery(
    const std::string& query) {
  std::vector<absl::string_view> tokens =
      absl::StrSplit(query, absl::ByChar(' '), absl::SkipEmpty());
  if (tokens.size() == 2 &&
      absl::EqualsIgnoreCase(absl::StripAsciiWhitespace(tokens[0]), "USE")) {
    database_name_ = std::string(tokens[1]);
  }
}

void MySqlMetadataSource::DiscardResultSet() {
  if (result_set_ != nullptr) {
    // Fetch any leftover rows (MySQL requires this).
    // Note: Since we always store the query result locally in RunQuery(),
    // we might not need to explicitly fetch the row here. But fetching rows
    // here won't hurt and will be a bit more robust in case we decide to lazily
    // fetch the rows from the server.
    while (mysql_fetch_row(result_set_)) {
    }
    mysql_free_result(result_set_);
    result_set_ = nullptr;
  }
}

Status MySqlMetadataSource::ConvertMySqlRowSetToRecordSet(
    RecordSet* record_set_out) {
  RecordSet record_set;

  if (result_set_ == nullptr) {
    return absl::OkStatus();
  }

  MYSQL_ROW row;
  while ((row = mysql_fetch_row(result_set_)) != nullptr) {
    RecordSet::Record record;
    std::vector<std::string> col_names;

    int64_t num_cols = mysql_num_fields(result_set_);
    for (int64_t col = 0; col < num_cols; ++col) {
      MYSQL_FIELD* field = mysql_fetch_field_direct(result_set_, col);
      if (field == nullptr) {
        return absl::InternalError(absl::StrCat(
            "Error in retrieving column description for index ", col));
      }
      const std::string col_name(field->name);
      if (record_set.column_names().empty()) {
        col_names.push_back(col_name);
      }

      if (row[col] == nullptr && !(field->flags & NOT_NULL_FLAG)) {
        record.add_values(kMetadataSourceNull.data());
      } else {
        record.add_values(absl::StrCat(row[col]));
      }
    }
    *record_set.mutable_records()->Add() = record;

    if (record_set.column_names().empty()) {
      *record_set.mutable_column_names() = {col_names.begin(), col_names.end()};
    }
  }

  if (record_set_out != nullptr) {
    *record_set_out = std::move(record_set);
  }
  return absl::OkStatus();
}

std::string MySqlMetadataSource::EscapeString(absl::string_view value) const {
  CHECK(db_ != nullptr);
  // in the worst case, each character needs to be escaped by backslash, and the
  // string is appended an additional terminating null character.
  char* buffer = new char[value.length() * 2 + 1];
  CHECK(mysql_real_escape_string(db_, buffer, value.data(), value.length()) !=
        -1UL)
      << "NO_BACKSLASH_ESCAPES SQL mode should not be enabled.";
  std::string result(buffer);
  delete[] buffer;
  return result;
}

std::string MySqlMetadataSource::EncodeBytes(absl::string_view value) const {
  return SqliteEncodeBytes(value);
}
absl::StatusOr<std::string> MySqlMetadataSource::DecodeBytes(
    absl::string_view value) const {
  return SqliteDecodeBytes(value);
}

}  // namespace ml_metadata
