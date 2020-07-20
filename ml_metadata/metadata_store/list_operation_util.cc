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
#include "ml_metadata/metadata_store/list_operation_util.h"

#include "absl/strings/escaping.h"

namespace ml_metadata {

// Helper method to set initial field and id offset values based on List
// operation options.
void SetListOperationInitialValues(const ListOperationOptions& options,
                                   int64& field_offset, int64& id_offset) {
  const bool is_asc = options.order_by_field().is_asc();
  field_offset = is_asc ? 0 : LLONG_MAX;
  id_offset = field_offset;
}

tensorflow::Status DecodeListOperationNextPageToken(
    const absl::string_view next_page_token,
    ListOperationNextPageToken& list_operation_next_page_token) {
  std::string token_str;
  if (!absl::WebSafeBase64Unescape(next_page_token, &token_str)) {
    return tensorflow::errors::InvalidArgument(
        "Failed to decode next page token string");
  }

  if (!list_operation_next_page_token.ParseFromString(token_str)) {
    return tensorflow::errors::InvalidArgument(
        "Failed to parse decoded next page token into "
        "ListOperationNextPageToken proto message ");
  }

  return tensorflow::Status::OK();
}

tensorflow::Status ValidateListOperationOptionsAreIdentical(
    const ListOperationOptions& previous_options,
    const ListOperationOptions& current_options) {
  if (previous_options.order_by_field().is_asc() ==
          current_options.order_by_field().is_asc() &&
      previous_options.order_by_field().field() ==
          current_options.order_by_field().field()) {
    return tensorflow::Status::OK();
  }

  return tensorflow::errors::InvalidArgument(absl::StrCat(
      "ListOperationOptions mismatch between calls Initial Options: ",
      previous_options.DebugString(),
      " Current Options: ", current_options.DebugString()));
}
}  // namespace ml_metadata
