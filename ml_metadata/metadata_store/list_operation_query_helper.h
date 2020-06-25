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

#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ml_metadata {

// Utility methods to generate SQL queries for List Operations.

// Generates the WHERE clause for ListOperation.
// On success `sql_query_clause` is appended with the constructed WHERE clause
// based on |options|.
// |id_offset| is used along with field offset as Id is the unique field used to
// break ties for fields that might have duplicate entries. For e.g. there could
// be two resources with same last_update_time. In such cases to  break the tie
// in ordering, id offset is used. Also if during the calls, the node is updated
// and its last_update_time has been changed, running the pagination query
// again, the updated node will moved to earlier pages.
// NOTE: The word 'WHERE' is not part of the returned clause the caller is
// expected to add the word and construct the complete query.
// Returns INVALID_ARGUMENT error if the |options| specified is invalid.
// For example, given a ListOperationOptions message:
// {
//    max_result_size: 1,
//    order_by_field: {
//      field: CREATE_TIME,
//      is_asc: false
//    }
// }
// Appends "`create_time_since_epoch` < |field_offset| AND `id` < |id_offset|`"
// at the end of `sql_query_clause`.
tensorflow::Status AppendOrderingThresholdClause(
    const ListOperationOptions& options, int64 id_offset, int64 field_offset,
    std::string& sql_query_clause);

// Generates the ORDER BY clause for ListOperation.
// On success `sql_query_clause` is appended with the constructed ORDER BY
// clause based on |options|.
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
tensorflow::Status AppendOrderByClause(const ListOperationOptions& options,
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
tensorflow::Status AppendLimitClause(const ListOperationOptions& options,
                                     std::string& sql_query_clause);

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_LIST_OPERATION_QUERY_HELPER_H_
