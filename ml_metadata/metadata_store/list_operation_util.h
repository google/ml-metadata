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

#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {

// Sets initial field and id offset values based on List operation options.
// TODO(b/187082552): Remove this method.
void SetListOperationInitialValues(const ListOperationOptions& options,
                                   int64_t& field_offset, int64_t& id_offset);

// Decodes ListOperationNextPageToken encoded in `next_page_token`.
absl::Status DecodeListOperationNextPageToken(
    absl::string_view next_page_token,
    ListOperationNextPageToken& list_operation_next_page_token);

// Generates encoded list operation next page token string.
template <typename Node>
absl::Status BuildListOperationNextPageToken(
    absl::Span<const Node> nodes, const ListOperationOptions& options,
    std::string* next_page_token) {
  const Node& last_node = nodes.back();
  ListOperationNextPageToken list_operation_next_page_token;
  switch (options.order_by_field().field()) {
    case ListOperationOptions::OrderByField::CREATE_TIME: {
      list_operation_next_page_token.set_field_offset(
          last_node.create_time_since_epoch());
      list_operation_next_page_token.set_id_offset(last_node.id());
      break;
    }
    case ListOperationOptions::OrderByField::LAST_UPDATE_TIME: {
      list_operation_next_page_token.add_listed_ids(last_node.id());
      list_operation_next_page_token.set_field_offset(
          last_node.last_update_time_since_epoch());
      for (int i = nodes.size() - 2; i >= 0; i--) {
        if (nodes[i].last_update_time_since_epoch() !=
            last_node.last_update_time_since_epoch()) {
          break;
        }
        list_operation_next_page_token.add_listed_ids(nodes[i].id());
      }
      break;
    }
    case ListOperationOptions::OrderByField::ID: {
      list_operation_next_page_token.set_field_offset(last_node.id());
      list_operation_next_page_token.set_id_offset(last_node.id());
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported field: ",
                       ListOperationOptions::OrderByField::Field_Name(
                           options.order_by_field().field()),
                       " specified in ListOperationOptions"));
  }
  *list_operation_next_page_token.mutable_set_options() = options;
  // Clear previous `next_page_token` as it is changing for each request and it
  // is a set-only field. If not clean this field, the `next_page_token` in
  // the follow-up request encodes itself and the size grows exponentially.
  list_operation_next_page_token.mutable_set_options()->clear_next_page_token();
  *next_page_token = absl::WebSafeBase64Escape(
      list_operation_next_page_token.SerializeAsString());
  return absl::OkStatus();
}

// Ensures that ListOperationOptions have not changed between
// calls. |previous_options| represents options used in the previous call and
// |current_options| represents options used in the current call.
// Validation validates order_by_fields and filter_query in
// ListOperationOptions.
absl::Status ValidateListOperationOptionsAreIdentical(
    const ListOperationOptions& previous_options,
    const ListOperationOptions& current_options);
}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_LIST_OPERATION_UTIL_H_
