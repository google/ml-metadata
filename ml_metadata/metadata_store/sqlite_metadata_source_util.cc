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
#include "ml_metadata/metadata_store/sqlite_metadata_source_util.h"
#include <string>

#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/metadata_store/constants.h"
#include "sqlite3.h"
#include "absl/strings/escaping.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ml_metadata {

std::string SqliteEscapeString(absl::string_view value) {
  char* buffer = sqlite3_mprintf("%q", value.data());  // NOLINT
  std::string result(buffer);
  sqlite3_free(buffer);

  return result;
}

// Encode bytes to workaround escaping difficulties.
std::string SqliteEncodeBytes(absl::string_view value) {
  return absl::Base64Escape(value);
}

absl::StatusOr<std::string> SqliteDecodeBytes(absl::string_view value) {
  absl::StatusOr<std::string> decoded_or = std::string();
  if (!absl::Base64Unescape(value, &decoded_or.value())) {
    decoded_or = absl::InternalError(absl::StrCat(
      "Failed to Base64Unescape value '", value, "'"));
  }
  return decoded_or;
}

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
    record->add_values(column_vals[i] ? column_vals[i]
                                      : kMetadataSourceNull.data());
  }
  return SQLITE_OK;
}

}  // namespace ml_metadata
