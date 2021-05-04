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
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/list_operation_util.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

namespace {

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

tensorflow::Status ValidateAndDecodeNextPageToken(
    const ListOperationOptions& options,
    ListOperationNextPageToken& next_page_token) {
  TF_RETURN_IF_ERROR(DecodeListOperationNextPageToken(options.next_page_token(),
                                                      next_page_token));
  TF_RETURN_IF_ERROR(ValidateListOperationOptionsAreIdentical(
      next_page_token.set_options(), options));

  return tensorflow::Status::OK();
}

// Constructures the WHERE condition for the field on which ordering is
// specified.
tensorflow::Status ConstructOrderingFieldClause(
    const ListOperationOptions& options, const int64 field_offset,
    std::string& ordering_clause) {
  std::string column_name;
  TF_RETURN_IF_ERROR(GetDbColumnNameForProtoField(
      options.order_by_field().field(), column_name));

  std::string ordering_operator = options.order_by_field().is_asc() ? ">" : "<";
  if (options.order_by_field().field() !=
      ListOperationOptions::OrderByField::ID) {
    absl::StrAppend(&ordering_operator, "=");
  }

  ordering_clause = absl::Substitute(" `$0` $1 $2 ", column_name,
                                     ordering_operator, field_offset);
  return tensorflow::Status::OK();
}

// Constructs the WHERE clause on the id field for CREATE_TIME ordering.
std::string ConstructIdOrderCaluse(const ListOperationOptions& options,
                                   const int64 id_offset) {
  return absl::Substitute(
      "`id` $0 $1 ", options.order_by_field().is_asc() ? ">" : "<", id_offset);
}

// Constructs the WHERE clause on the id field for LAST_UPDATE_TIME ordering.
tensorflow::Status ConstructIdNotInCaluse(absl::Span<const int64> listed_ids,
                                          std::string& not_in_clause) {
  if (listed_ids.empty()) {
    return tensorflow::errors::Internal(
        "Invalid NextPageToken in List Operation. listed_ids field should not "
        "be empty.");
  }
  not_in_clause =
      absl::Substitute("`id` NOT IN ($0) ", absl::StrJoin(listed_ids, ","));
  return tensorflow::Status::OK();
}

// Constructs the WHERE clause on the id field. This clause is constructed only
// if the ordering field is CREATE_TIME or LAST_UPDATE_TIME.
tensorflow::Status ConstructIdClause(
    const ListOperationOptions& options,
    const ListOperationNextPageToken& next_page_token, std::string& id_clause) {
  switch (options.order_by_field().field()) {
    case ListOperationOptions::OrderByField::CREATE_TIME:
      id_clause = ConstructIdOrderCaluse(options, next_page_token.id_offset());
      break;
    case ListOperationOptions::OrderByField::LAST_UPDATE_TIME: {
      std::vector<int64> listed_ids;
      for (auto it = next_page_token.listed_ids().begin();
           it != next_page_token.listed_ids().end(); it++) {
        listed_ids.push_back(*it);
      }
      TF_RETURN_IF_ERROR(ConstructIdNotInCaluse(listed_ids, id_clause));
      break;
    }
    case ListOperationOptions::OrderByField::ID:
      id_clause = "";
      break;
    default:
      return tensorflow::errors::InvalidArgument(
          absl::StrCat("Unsupported field: ",
                       ListOperationOptions::OrderByField::Field_Name(
                           options.order_by_field().field()),
                       " specified in ListOperationOptions"));
  }
  return tensorflow::Status::OK();
}

}  // namespace

tensorflow::Status AppendOrderingThresholdClause(
    const ListOperationOptions& options, std::string& sql_query_clause) {
  std::string field_clause, id_clause;

  if (!options.next_page_token().empty()) {
    ListOperationNextPageToken next_page_token;
    // TODO(b/187104516): Refactor the code to move the validation to earlier in
    //  the API call.
    TF_RETURN_IF_ERROR(
        ValidateAndDecodeNextPageToken(options, next_page_token));
    TF_RETURN_IF_ERROR(ConstructOrderingFieldClause(
        options, next_page_token.field_offset(), field_clause));
    TF_RETURN_IF_ERROR(ConstructIdClause(options, next_page_token, id_clause));
  } else {
    TF_RETURN_IF_ERROR(ConstructOrderingFieldClause(
        options, options.order_by_field().is_asc() ? 0 : LLONG_MAX,
        field_clause));
  }

  absl::StrAppend(&sql_query_clause, field_clause);
  if (!id_clause.empty()) {
    absl::SubstituteAndAppend(&sql_query_clause, "AND $0", id_clause);
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
      std::min(options.max_result_size(),
               GetDefaultMaxListOperationResultSize() + 1);
  absl::SubstituteAndAppend(&sql_query_clause, " LIMIT $0 ", max_result_size);
  return tensorflow::Status::OK();
}
}  // namespace ml_metadata
