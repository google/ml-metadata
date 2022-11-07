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
#ifndef ML_METADATA_METADATA_STORE_SQLITE_METADATA_SOURCE_UTIL_H_
#define ML_METADATA_METADATA_STORE_SQLITE_METADATA_SOURCE_UTIL_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ml_metadata {

// Escapes strings having single quotes using built-in printf in Sqlite3 C API.
std::string SqliteEscapeString(absl::string_view value);

// Encode bytes values before binding to queries.
std::string SqliteEncodeBytes(absl::string_view value);

// Decode bytes values that were encoded with SqliteEncodeBytes
absl::StatusOr<std::string> SqliteDecodeBytes(absl::string_view value);

// Converts the query results (`column_vals`) if any to a RecordSet (`results`).
// It is used as a callback of sqlite3_exec. The `results` should be owned by
// the caller of sqlite3_exec. If the given RecordSet (`results`) is nullptr,
// the query result is ignored.
// (see https://www.sqlite.org/c3ref/exec.html for details)
int ConvertSqliteResultsToRecordSet(void* results, int column_num,
                                    char** column_vals, char** column_names);

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_SQLITE_METADATA_SOURCE_UTIL_H_
