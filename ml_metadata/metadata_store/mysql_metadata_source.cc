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
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "mysql/mysql.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

namespace {

namespace errors = tensorflow::errors;

using ::tensorflow::Status;

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
  std::vector<string> config_errors;
  if (config.host().empty()) {
    config_errors.push_back("host must not be empty");
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
  if (db_ == nullptr) {
    return errors::Internal("mysql_init failed");
  }

  // Explicitly setup the thread-local initializer.
  TF_RETURN_IF_ERROR(ThreadInitAccess());

  // Connect via TCP.
  int protocol = MYSQL_PROTOCOL_TCP;
  if (mysql_options(db_, MYSQL_OPT_PROTOCOL, &protocol)) {
    return errors::Internal(
        "Failed to set connection protocol to MYSQL_PROTOCOL_TCP");
  }

  // Connect to the MYSQL server.
  if (!mysql_real_connect(
          db_, config_.host().c_str(),
          config_.user().empty() ? nullptr : config_.user().c_str(),
          config_.password().empty() ? nullptr : config_.password().c_str(),
          /*db=*/nullptr, config_.port(),
          /*unix_socket=*/nullptr, /*clientflag=*/0UL)) {
    return errors::Internal("mysql_real_connect failed: errno: ",
                            mysql_errno(db_), ", error: ", mysql_error(db_));
  }

  // Return an error if the default storage engine doesn't support transactions.
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      CheckTransactionSupport(),
      "checking transaction support of default storage engine");

  // Create the database if not already present and switch to it.
  const string create_database_cmd =
      absl::StrCat("CREATE DATABASE IF NOT EXISTS ", config_.database());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(RunQuery(create_database_cmd),
                                  "Creating database ", config_.database(),
                                  " in ConnectImpl");
  const string use_database_cmd = absl::StrCat("USE ", config_.database());
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

Status MySqlMetadataSource::ExecuteQueryImpl(const string& query,
                                             RecordSet* results) {
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ThreadInitAccess(), "ExecuteQueryImpl");

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
  constexpr char kCommitTransaction[] = "COMMIT";

  TF_RETURN_WITH_CONTEXT_IF_ERROR(ThreadInitAccess(), "CommitImpl");
  return RunQuery(kCommitTransaction);
}

Status MySqlMetadataSource::RollbackImpl() {
  constexpr char kRollbackTransaction[] = "ROLLBACK";

  TF_RETURN_WITH_CONTEXT_IF_ERROR(ThreadInitAccess(), "RollbackImpl");
  return RunQuery(kRollbackTransaction);
}

Status MySqlMetadataSource::BeginImpl() {
  constexpr char kBeginTransaction[] = "START TRANSACTION";
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

Status MySqlMetadataSource::RunQuery(const string& query) {
  DiscardResultSet();

  if (mysql_query(db_, query.c_str())) {
    return errors::Internal("mysql_query failed: errno: ", mysql_errno(db_),
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
    std::vector<string> col_names;

    uint32 num_cols = mysql_num_fields(result_set_);
    for (uint32 col = 0; col < num_cols; ++col) {
      MYSQL_FIELD* field = mysql_fetch_field_direct(result_set_, col);
      if (field == nullptr) {
        return errors::Internal(
            "Error in retrieving column description for index ", col);
      }
      const string col_name(field->org_name);
      if (record_set.column_names().empty()) {
        col_names.push_back(col_name);
      }

      if (row[col] == nullptr && !(field->flags & NOT_NULL_FLAG)) {
        record.add_values("");
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

string MySqlMetadataSource::EscapeString(absl::string_view value) const {
  CHECK(db_ != nullptr);
  // in the worst case, each character needs to be escaped by backslash, and the
  // string is appended an additional terminating null character.
  char* buffer = new char[value.length() * 2 + 1];
  CHECK(mysql_real_escape_string(db_, buffer, value.data(), value.length()) !=
        -1UL)
      << "NO_BACKSLASH_ESCAPES SQL mode should not be enabled.";
  string result(buffer);
  delete[] buffer;
  return result;
}

}  // namespace ml_metadata
