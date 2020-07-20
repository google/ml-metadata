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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_LIST_OPERATION_UTIL_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_LIST_OPERATION_UTIL_H_

#include "absl/strings/escaping.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ml_metadata {

// Sets initial field and id offset values based on List operation options.
void SetListOperationInitialValues(const ListOperationOptions& options,
                                   int64& field_offset, int64& id_offset);

// Decodes ListOperationNextPageToken encoded in `next_page_token`.
tensorflow::Status DecodeListOperationNextPageToken(
    const absl::string_view next_page_token,
    ListOperationNextPageToken& list_operation_next_page_token);

// Generates encoded list operation next page token string.
template <typename Node>
tensorflow::Status BuildListOperationNextPageToken(
    const Node& node, const ListOperationOptions& options,
    std::string* next_page_token) {
  int64 field_offset;
  switch (options.order_by_field().field()) {
    case ListOperationOptions::OrderByField::CREATE_TIME:
      field_offset = node.create_time_since_epoch();
      break;
    case ListOperationOptions::OrderByField::LAST_UPDATE_TIME:
      field_offset = node.last_update_time_since_epoch();
      break;
    case ListOperationOptions::OrderByField::ID:
      field_offset = node.id();
      break;
    default:
      return tensorflow::errors::InvalidArgument(
          absl::StrCat("Unsupported field: ",
                       ListOperationOptions::OrderByField::Field_Name(
                           options.order_by_field().field()),
                       " specified in ListOperationOptions"));
  }

  ListOperationNextPageToken list_operation_next_page_token;
  list_operation_next_page_token.set_field_offset(field_offset);
  list_operation_next_page_token.set_id_offset(node.id());
  *list_operation_next_page_token.mutable_set_options() = options;

  *next_page_token = absl::WebSafeBase64Escape(
      list_operation_next_page_token.SerializeAsString());
  return tensorflow::Status::OK();
}

// Ensures that ListOperationOptions have not changed between
// calls. |previous_options| represents options used in the previous call and
// |current_options| represents options used in the current call.
// Validation only validates order_by_fields in ListOperationOptions.
tensorflow::Status ValidateListOperationOptionsAreIdentical(
    const ListOperationOptions& previous_options,
    const ListOperationOptions& current_options);
}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_LIST_OPERATION_UTIL_H_
