/* Copyright 2020 Google LLC

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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_LIST_OPERATION_QUERY_HELPER_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_LIST_OPERATION_QUERY_HELPER_H_

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {

// Utility methods to generate SQL queries for List Operations.

// Generates the WHERE clause for ListOperation.
// On success `sql_query_clause` is appended with the constructed parameterized
// WHERE clause based on |options| and qualifying column names with
// |table_alias|.
// For the first page of results based on the ordering field this method
// generates a ordering clause based only on that field, e.g.,
// `create_time_since_epoch`  <= INT_MAX (for desc) and
// `create_time_since_epoch` >= 0 (for asc).
// For following pages method first decodes the next_page_token set in `options`
// and validates the token.
// Based on the ordering field the generated clause is as follows:
//  1. CREATE_TIME
//   `create_time_since_epoch` <= |options.field_offset| AND
//       `id` < |options.id_offset|`
//  2. LAST_UPDATE_TIME
//   `last_update_time_since_epoch` <= |options.field_offset| AND
//       `id` NOT IN(options.list_ids)`
//    list_ids - Is the list of trailing node ids in the previous page that have
//    the same last_update_time_since_epoch value.
//  3. ID
//    `id < |options.field_offset|.
//
// Returns INVALID_ARGUMENT error if the `options` or `next_page_token`
// specified is invalid.
absl::Status AppendOrderingThresholdClause(
    const ListOperationOptions& options,
    std::optional<absl::string_view> table_alias,
    std::string& sql_query_clause);

// Generates the ORDER BY clause for ListOperation.
// On success `sql_query_clause` is appended with the constructed ORDER BY
// clause based on |options| and qualifying column names with |table_alias|.
// Returns INVALID_ARGUMENT error if the |options|
// specified is invalid.
// For example, given a ListOperationOptions message:
// {
//    max_result_size: 1,
//    order_by_field: {
//      field: CREATE_TIME,
//      is_asc: false
//    }
// }
// Appends "ORDER BY `create_time_since_epoch` DESC, `id` DESC" at the end of
// `sql_query_clause`.
absl::Status AppendOrderByClause(const ListOperationOptions& options,
                                 std::optional<absl::string_view> table_alias,
                                 std::string& sql_query_clause);

// Generates the LIMIT clause for ListOperation.
// On success `sql_query_clause` is appended with the constructed LIMIT clause
// based on |options|.
// If |options| does not specify max_result_size a default value of 20 is used
// and if the max_result_size is greater than 100, the LIMIT clause coerces the
// value to 100.
// For example, given a ListOperationOptions message:
// {
//    max_result_size: 1,
//    order_by_field: {
//      field: CREATE_TIME,
//      is_asc: false
//    }
// }
// Appends "LIMIT 1" at the end of `sql_query_clause`.
absl::Status AppendLimitClause(const ListOperationOptions& options,
                               std::string& sql_query_clause);

// Gets the maximum number of returned resources for List operation.
inline constexpr int GetDefaultMaxListOperationResultSize() {
  return kDefaultMaxListOperationResultSize;
}

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_LIST_OPERATION_QUERY_HELPER_H_
