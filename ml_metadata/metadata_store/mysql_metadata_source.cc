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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "mysql.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

namespace {

namespace errors = tensorflow::errors;

using ::tensorflow::Status;

constexpr char kBeginTransaction[] = "START TRANSACTION";
constexpr char kCommitTransaction[] = "COMMIT";
constexpr char kRollbackTransaction[] = "ROLLBACK";

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
    return errors::Internal("mysql thread initialization not done");
  }
  return Status::OK();
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
    return errors::InvalidArgument(absl::StrJoin(config_errors, ";"));
  }
  return Status::OK();
}

}  // namespace

MySqlMetadataSource::MySqlMetadataSource(const MySQLDatabaseConfig& config)
    : MetadataSource(), config_(config) {
  TF_CHECK_OK(CheckConfig(config));
}

MySqlMetadataSource::~MySqlMetadataSource() { TF_CHECK_OK(CloseImpl()); }

Status MySqlMetadataSource::ConnectImpl() {
  // Initialize the MYSQL object.
  db_ = mysql_init(nullptr);
  if (!db_) {
    LOG(ERROR) << "MySQL error: " << mysql_errno(db_) << ": "
               << mysql_error(db_);
    return errors::Internal("mysql_init failed: errno: ",
                            mysql_errno(db_), ", error: ", mysql_error(db_));
  }

  // Explicitly setup the thread-local initializer.
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ThreadInitAccess(),
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

  // Connect to the MYSQL server.
  db_ = mysql_real_connect(
          db_, config_.host().empty() ? nullptr : config_.host().c_str(),
          config_.user().empty() ? nullptr : config_.user().c_str(),
          config_.password().empty() ? nullptr : config_.password().c_str(),
          /*db=*/nullptr, config_.port(),
          config_.socket().empty() ? nullptr : config_.socket().c_str(),
          /*clientflag=*/0UL);

  if (!db_) {
    return errors::Internal("mysql_real_connect failed: errno: ",
                            mysql_errno(db_), ", error: ", mysql_error(db_));
  }

  // Return an error if the default storage engine doesn't support transactions.
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      CheckTransactionSupport(),
      "checking transaction support of default storage engine");

  // Create the database if not already present and switch to it.
  const std::string create_database_cmd =
      absl::StrCat("CREATE DATABASE IF NOT EXISTS ", config_.database());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(RunQuery(create_database_cmd),
                                  "Creating database ", config_.database(),
                                  " in ConnectImpl");
  const std::string use_database_cmd = absl::StrCat("USE ", config_.database());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(RunQuery(use_database_cmd),
                                  "Changing to database ", config_.database(),
                                  " in ConnectImpl");

  return Status::OK();
}

Status MySqlMetadataSource::CloseImpl() {
  if (db_ != nullptr) {
    TF_RETURN_IF_ERROR(ThreadInitAccess());
    DiscardResultSet();
    mysql_close(db_);
    db_ = nullptr;
  }
  return Status::OK();
}

Status MySqlMetadataSource::ExecuteQueryImpl(const std::string& query,
                                             RecordSet* results) {
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ThreadInitAccess(), "MySql thread init failed at ExecuteQueryImpl");

  // Run the query.
  Status status = RunQuery(query);

  // Return on failure.
  TF_RETURN_IF_ERROR(status);

  // If query is successfull, convert the results.
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ConvertMySqlRowSetToRecordSet(results),
                                  "ConvertMySqlRowSetToRecordSet for query ",
                                  query);
  return Status::OK();
}

Status MySqlMetadataSource::CommitImpl() {
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ThreadInitAccess(),
                                  "MySql thread init failed at CommitImpl");
  return RunQuery(kCommitTransaction);
}

Status MySqlMetadataSource::RollbackImpl() {
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ThreadInitAccess(),
                                  "MySql thread init failed at RollbackImpl");
  return RunQuery(kRollbackTransaction);
}

Status MySqlMetadataSource::BeginImpl() {
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ThreadInitAccess(),
                                  "MySql thread init failed at BeginImpl");
  return RunQuery(kBeginTransaction);
}

Status MySqlMetadataSource::CheckTransactionSupport() {
  constexpr char kCheckTransactionSupport[] =
      "SELECT ENGINE, TRANSACTIONS FROM INFORMATION_SCHEMA.ENGINES WHERE "
      "ENGINE=(SELECT @@default_storage_engine)";

  TF_RETURN_IF_ERROR(RunQuery(kCheckTransactionSupport));

  RecordSet record_set;
  TF_RETURN_IF_ERROR(ConvertMySqlRowSetToRecordSet(&record_set));
  if (record_set.records_size() != 1 ||
      record_set.records(0).values_size() != 2) {
    return errors::Internal(
        "Expected query ", kCheckTransactionSupport,
        " to generate exactly single row with 2 columns, but got ",
        record_set.DebugString());
  }
  const RecordSet::Record& record = record_set.records(0);
  if (record.values(1) != "YES") {
    return errors::Internal(
        "no transaction support for default_storage_engine ", record.values(0));
  }

  return Status::OK();
}

Status MySqlMetadataSource::RunQuery(const std::string& query) {
  DiscardResultSet();

  int query_status = mysql_query(db_, query.c_str());
  if (query_status) {
    int64 error_number = mysql_errno(db_);
    // 2006: sever closes the connection due to inactive client;
    // client reports server has gone away, we reconnect the server for the
    // client if the query is begin transaction.
    if (error_number == 2006 && query == kBeginTransaction) {
      TF_RETURN_IF_ERROR(CloseImpl());
      TF_RETURN_IF_ERROR(ConnectImpl());
      return RunQuery(query);
    }
    // 1213: inno db aborts deadlock when running concurrent transactions.
    // returns Aborted for client side to retry.
    if (error_number == 1213 || error_number == 1205) {
      return errors::Aborted("mysql_query aborted: errno: ", error_number,
                             ", error: ", mysql_error(db_));
    }
    return errors::Internal("mysql_query failed: errno: ", error_number,
                            ", error: ", mysql_error(db_));
  }

  result_set_ = mysql_store_result(db_);
  if (!result_set_ && mysql_field_count(db_) != 0) {
    return errors::Internal("mysql_query ", query,
                            " returned an unexpected NULL result_set: Errno: ",
                            mysql_errno(db_), ", Error: ", mysql_error(db_));
  }

  return Status::OK();
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
    return Status::OK();
  }

  MYSQL_ROW row;
  while ((row = mysql_fetch_row(result_set_)) != nullptr) {
    RecordSet::Record record;
    std::vector<std::string> col_names;

    uint32 num_cols = mysql_num_fields(result_set_);
    for (uint32 col = 0; col < num_cols; ++col) {
      MYSQL_FIELD* field = mysql_fetch_field_direct(result_set_, col);
      if (field == nullptr) {
        return errors::Internal(
            "Error in retrieving column description for index ", col);
      }
      const std::string col_name(field->org_name);
      if (record_set.column_names().empty()) {
        col_names.push_back(col_name);
      }

      if (row[col] == nullptr && !(field->flags & NOT_NULL_FLAG)) {
        record.add_values(kMetadataSourceNull);
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
  return Status::OK();
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

}  // namespace ml_metadata
