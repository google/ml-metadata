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
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "sqlite3.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

namespace {

constexpr char kInMemoryConnection[] = ":memory:";
constexpr char kBeginTransaction[] = "BEGIN;";
constexpr char kCommitTransaction[] = "COMMIT;";
constexpr char kRollbackTransaction[] = "ROLLBACK;";

// Returns a Sqlite3 connection flags based on the SqliteMetadataSourceConfig.
// (see https://www.sqlite.org/c3ref/open.html for details)
int GetConnectionFlag(const SqliteMetadataSourceConfig& config) {
  int result = SQLITE_OPEN_URI;
  switch (config.connection_mode()) {
    case SqliteMetadataSourceConfig::READONLY: {
      result |= SQLITE_OPEN_READONLY;
      break;
    }
    case SqliteMetadataSourceConfig::READWRITE: {
      result |= SQLITE_OPEN_READWRITE;
      break;
    }
    case SqliteMetadataSourceConfig::READWRITE_OPENCREATE: {
      result |= (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
      break;
    }
    default:
      LOG(FATAL) << "Unknown connection mode.";
  }
  return result;
}

// A callback of sqlite3_exec. It converts the query results (`column_vals`) if
// any to a RecordSet (`results`). The `results` should be owned by the caller
// of sqlite3_exec. If the given RecordSet (`results`) is nullptr, the query
// result is ignored.
// (see https://www.sqlite.org/c3ref/exec.html for details)
int ConvertSqliteResultsToRecordSet(void* results, int column_num,
                                    char** column_vals, char** column_names) {
  // return if queries return no result, e.g., create, insert, update, etc.
  if (column_num == 0) return SQLITE_OK;
  RecordSet* record_set = static_cast<RecordSet*>(results);
  // ignore the results of the query, if the user passes a nullptr for results.
  if (record_set == nullptr) return SQLITE_OK;
  // parse the query results to the record_sets (column_names and records)
  // as sqlite calls the callback repetitively for each row, it sets the column
  // names at the first time.
  bool is_column_name_initted = (record_set->column_names_size() == column_num);
  if (!is_column_name_initted) record_set->clear_column_names();
  RecordSet::Record* record = record_set->add_records();
  for (int i = 0; i < column_num; i++) {
    if (!is_column_name_initted) record_set->add_column_names(column_names[i]);
    record->add_values(column_vals[i] ? column_vals[i] : "");
  }
  return SQLITE_OK;
}

// A set of options when waiting for table locks in a sqlite3_busy_handler.
// see WaitThenRetry for details.
struct WaitThenRetryOptions {
  // the wait time in milliseconds
  absl::Duration sleep_time;
  // the max time to wait for a single lock
  absl::Duration max_retried_time;
};

// A callback of sqlite3_busy_handler. Concurrent access to a table may prevent
// query to proceed, the callback returns zero to continue waiting, and non-zero
// to abort the query and returns a SQLITE_BUSY error. The function takes a
// WaitThenRetryOptions (`options`), waits for a lock and then indicates sqlite3
// to retry. The default values for sleep is 100 millisecond (`sleep_time`)
// for each wait and 2 times at maximum (`max_retried_time`/`sleep_time`).
// (see https://www.sqlite.org/c3ref/busy_handler.html for details)
int WaitThenRetry(void* options, int retried_times) {
  static constexpr WaitThenRetryOptions kDefaultWaitThenRetryOptions = {
      absl::Milliseconds(100), absl::Milliseconds(200)};
  const WaitThenRetryOptions* opts =
      (options != nullptr) ? static_cast<WaitThenRetryOptions*>(options)
                           : &kDefaultWaitThenRetryOptions;
  // aborts the query with SQLITE_BUSY
  if (retried_times >= opts->max_retried_time / opts->sleep_time) return 0;
  absl::SleepFor(opts->sleep_time);
  // allow further retry
  return 1;
}

}  // namespace

SqliteMetadataSource::SqliteMetadataSource(
    const SqliteMetadataSourceConfig& config)
    : config_(config) {
  if (config_.filename_uri().empty())
    config_.set_filename_uri(kInMemoryConnection);
  if (!config_.connection_mode())
    config_.set_connection_mode(
        SqliteMetadataSourceConfig::READWRITE_OPENCREATE);
}

SqliteMetadataSource::~SqliteMetadataSource() { TF_CHECK_OK(CloseImpl()); }

tensorflow::Status SqliteMetadataSource::ConnectImpl() {
  if (sqlite3_open_v2(config_.filename_uri().c_str(), &db_,
                      GetConnectionFlag(config_), nullptr) != SQLITE_OK) {
    string error_message = sqlite3_errmsg(db_);
    sqlite3_close(db_);
    db_ = nullptr;
    return tensorflow::errors::Internal("Cannot connect sqlite3 database: ",
                                        error_message);
  }
  // required to handle cases when tables are locked when executing queries
  sqlite3_busy_handler(db_, &WaitThenRetry, nullptr);
  return tensorflow::Status::OK();
}

tensorflow::Status SqliteMetadataSource::CloseImpl() {
  if (db_ != nullptr) {
    int error_code = sqlite3_close(db_);
    if (error_code != SQLITE_OK) {
      return tensorflow::errors::Internal(
          absl::StrCat("Cannot close sqlite3 database: ", error_code));
    }
    db_ = nullptr;
  }
  return tensorflow::Status::OK();
}

tensorflow::Status SqliteMetadataSource::RunStatement(
    const string& query, RecordSet* results = nullptr) {
  char* error_message;
  if (sqlite3_exec(db_, query.c_str(), &ConvertSqliteResultsToRecordSet,
                   results, &error_message) != SQLITE_OK) {
    string error_details = error_message;
    sqlite3_free(error_message);
    return tensorflow::errors::Internal(
        "Error when executing query: ", error_details, "query: ", query);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status SqliteMetadataSource::ExecuteQueryImpl(const string& query,
                                                          RecordSet* results) {
  return RunStatement(query, results);
}

tensorflow::Status SqliteMetadataSource::BeginImpl() {
  return RunStatement(kBeginTransaction);
}

tensorflow::Status SqliteMetadataSource::CommitImpl() {
  return RunStatement(kCommitTransaction);
}

tensorflow::Status SqliteMetadataSource::RollbackImpl() {
  return RunStatement(kRollbackTransaction);
}

string SqliteMetadataSource::EscapeString(absl::string_view value) const {
  char* buffer = sqlite3_mprintf("%q", value.data());  // NOLINT
  string result(buffer);
  sqlite3_free(buffer);
  return result;
}

}  // namespace ml_metadata
