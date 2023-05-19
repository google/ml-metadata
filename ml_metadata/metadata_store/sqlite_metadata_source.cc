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

#include <random>

#include <glog/logging.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "sqlite3.h"

namespace ml_metadata {

namespace {

constexpr absl::string_view kInMemoryConnection = ":memory:";
constexpr absl::string_view kBeginTransaction = "BEGIN;";
constexpr absl::string_view kCommitTransaction = "COMMIT;";
constexpr absl::string_view kRollbackTransaction = "ROLLBACK;";

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

// A set of options when waiting for table locks in a sqlite3_busy_handler.
// see WaitThenRetry for details.
struct WaitThenRetryOptions {
  // the min wait time in milliseconds.
  absl::Duration min_sleep_time;
  // the max wait time in milliseconds.
  absl::Duration max_sleep_time;
  // the max number of retries.
  int max_num_retries;
};

// A callback of sqlite3_busy_handler. Concurrent access to a table may prevent
// query to proceed, the callback returns zero to continue waiting, and non-zero
// to abort the query and returns a SQLITE_BUSY error. The function takes a
// WaitThenRetryOptions (`options`), waits for a lock and then indicates sqlite3
// to retry. The default sleep time mean is 200 millisecond and sleep 10 times
// at maximum (see https://www.sqlite.org/c3ref/busy_handler.html for details).
int WaitThenRetry(void* options, int retried_times) {
  static constexpr WaitThenRetryOptions kDefaultWaitThenRetryOptions = {
      absl::Milliseconds(100), absl::Milliseconds(300), 10};
  const WaitThenRetryOptions* opts =
      (options != nullptr) ? static_cast<WaitThenRetryOptions*>(options)
                           : &kDefaultWaitThenRetryOptions;
  // aborts the query with SQLITE_BUSY
  if (retried_times >= opts->max_num_retries) {
    return 0;
  }
  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));
  std::uniform_int_distribution<int64_t> uniform_dist(
      opts->min_sleep_time / absl::Milliseconds(1),
      opts->max_sleep_time / absl::Milliseconds(1));
  const absl::Duration sleep_time = absl::Milliseconds(uniform_dist(gen));
  absl::SleepFor(sleep_time);
  // allow further retry
  return 1;
}

}  // namespace

SqliteMetadataSource::SqliteMetadataSource(
    const SqliteMetadataSourceConfig& config)
    : config_(config) {
  if (config_.filename_uri().empty())
    config_.set_filename_uri(kInMemoryConnection.data());
  if (!config_.connection_mode())
    config_.set_connection_mode(
        SqliteMetadataSourceConfig::READWRITE_OPENCREATE);
}

SqliteMetadataSource::~SqliteMetadataSource() {
  CHECK_EQ(absl::OkStatus(), CloseImpl());
}

absl::Status SqliteMetadataSource::ConnectImpl() {
  if (sqlite3_open_v2(config_.filename_uri().c_str(), &db_,
                      GetConnectionFlag(config_), nullptr) != SQLITE_OK) {
    std::string error_message = sqlite3_errmsg(db_);
    sqlite3_close(db_);
    db_ = nullptr;
    return absl::InternalError(
        absl::StrCat("Cannot connect sqlite3 database: ", error_message));
  }
  // required to handle cases when tables are locked when executing queries
  sqlite3_busy_handler(db_, &WaitThenRetry, nullptr);
  return absl::OkStatus();
}

absl::Status SqliteMetadataSource::CloseImpl() {
  if (db_ != nullptr) {
    int error_code = sqlite3_close(db_);
    if (error_code != SQLITE_OK) {
      return absl::InternalError(
          absl::StrCat("Cannot close sqlite3 database: ", error_code));
    }
    db_ = nullptr;
  }
  return absl::OkStatus();
}

absl::Status SqliteMetadataSource::RunStatement(const std::string& query,
                                                RecordSet* results = nullptr) {
  char* error_message;
  if (sqlite3_exec(db_, query.c_str(), &ConvertSqliteResultsToRecordSet,
                   results, &error_message) != SQLITE_OK) {
    std::string error_details = error_message;
    sqlite3_free(error_message);
    if (absl::StrContains(error_details, "database is locked")) {
      return absl::AbortedError(
          "Concurrent writes aborted after max number of retries.");
    }
    return absl::InternalError(absl::StrCat(
        "Error when executing query: ", error_details, " query: ", query));
  }
  return absl::OkStatus();
}

absl::Status SqliteMetadataSource::ExecuteQueryImpl(const std::string& query,
                                                    RecordSet* results) {
  return RunStatement(query, results);
}

absl::Status SqliteMetadataSource::BeginImpl() {
  return RunStatement(kBeginTransaction.data());
}


absl::Status SqliteMetadataSource::CommitImpl() {
  return RunStatement(kCommitTransaction.data());
}

absl::Status SqliteMetadataSource::RollbackImpl() {
  return RunStatement(kRollbackTransaction.data());
}

std::string SqliteMetadataSource::EscapeString(absl::string_view value) const {
  return SqliteEscapeString(value);
}

std::string SqliteMetadataSource::EncodeBytes(absl::string_view value) const {
  return SqliteEncodeBytes(value);
}
absl::StatusOr<std::string> SqliteMetadataSource::DecodeBytes(
    absl::string_view value) const {
  return SqliteDecodeBytes(value);
}

}  // namespace ml_metadata
