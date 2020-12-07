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
#include "ml_metadata/metadata_store/list_operation_query_helper.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "ml_metadata/metadata_store/list_operation_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ml_metadata {

namespace {

// Default maximum number of returned resources for List operation.
constexpr int kDefaultMaxListOperationResultSize = 100;

// Helper method to map Proto ListOperationOptions::OrderByField::Field to
// Database column name.
tensorflow::Status GetDbColumnNameForProtoField(
    const ListOperationOptions::OrderByField::Field field,
    std::string& column_name) {
  switch (field) {
    case ListOperationOptions::OrderByField::CREATE_TIME:
      column_name = "create_time_since_epoch";
      break;
    case ListOperationOptions::OrderByField::LAST_UPDATE_TIME:
      column_name = "last_update_time_since_epoch";
      break;
    case ListOperationOptions::OrderByField::ID:
      column_name = "id";
      break;
    default:
      return tensorflow::errors::InvalidArgument(
          absl::StrCat("Unsupported field: ",
                       ListOperationOptions::OrderByField::Field_Name(field),
                       " specified in ListOperationOptions"));
  }

  return tensorflow::Status::OK();
}
}  // namespace

tensorflow::Status AppendOrderingThresholdClause(
    const ListOperationOptions& options, const int64 id_offset,
    const int64 field_offset, std::string& sql_query_clause) {
  if (options.order_by_field().field() !=
      ListOperationOptions::OrderByField::ID) {
    std::string column_name;
    TF_RETURN_IF_ERROR(GetDbColumnNameForProtoField(
        options.order_by_field().field(), column_name));

    absl::SubstituteAndAppend(
        &sql_query_clause, " `$0` $1 $2 AND `id` $3 $4 ", column_name,
        options.order_by_field().is_asc() ? ">=" : "<=", field_offset,
        options.order_by_field().is_asc() ? ">" : "<", id_offset);

  } else {
    absl::SubstituteAndAppend(&sql_query_clause, " `id` $0 $1 ",
                              options.order_by_field().is_asc() ? ">" : "<",
                              id_offset);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status AppendOrderByClause(const ListOperationOptions& options,
                                       std::string& sql_query_clause) {
  const std::string ordering_direction =
      options.order_by_field().is_asc() ? "ASC" : "DESC";

  std::string column_name;
  TF_RETURN_IF_ERROR(GetDbColumnNameForProtoField(
      options.order_by_field().field(), column_name));

  absl::SubstituteAndAppend(&sql_query_clause, " ORDER BY `$0` $1", column_name,
                            ordering_direction);
  if (options.order_by_field().field() !=
      ListOperationOptions::OrderByField::ID) {
    absl::SubstituteAndAppend(&sql_query_clause, ", `id` $0",
                              ordering_direction);
  }

  absl::StrAppend(&sql_query_clause, " ");
  return tensorflow::Status::OK();
}

tensorflow::Status AppendLimitClause(const ListOperationOptions& options,
                                     std::string& sql_query_clause) {
  if (options.max_result_size() <= 0) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("max_result_size field value is required to be greater "
                     "than 0. Set value: ",
                     options.max_result_size()));
  }

  const int max_result_size =
      std::min(options.max_result_size(), kDefaultMaxListOperationResultSize);
  absl::SubstituteAndAppend(&sql_query_clause, " LIMIT $0 ", max_result_size);
  return tensorflow::Status::OK();
}
}  // namespace ml_metadata
