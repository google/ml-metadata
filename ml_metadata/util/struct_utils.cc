/* Copyright 2021 Google LLC

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
#include "ml_metadata/util/struct_utils.h"

#include "google/protobuf/struct.pb.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

namespace ml_metadata {
namespace {

// TODO(b/178235219): Remove this once we migrate to schema v7 and add a column
// for storing serialized `Struct` values.
constexpr absl::string_view kSerializedStructPrefix = "mlmd-struct::";
}

bool IsStructSerializedString(absl::string_view serialized_value) {
  return absl::StartsWith(serialized_value, kSerializedStructPrefix);
}

std::string StructToString(const google::protobuf::Struct& struct_value) {
  return absl::StrCat(kSerializedStructPrefix,
                      absl::Base64Escape(struct_value.SerializeAsString()));
}

absl::Status StringToStruct(absl::string_view serialized_value,
                            google::protobuf::Struct& struct_value) {
  if (!absl::StartsWith(serialized_value, kSerializedStructPrefix)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Not a valid serialized `Struct`: ", serialized_value));
  }
  std::string unescaped;

  if (!absl::Base64Unescape(
          absl::StripPrefix(serialized_value, kSerializedStructPrefix),
          &unescaped)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unable to parse serialized `google.protobuf.Struct` value: ",
        unescaped));
  }

  if (!struct_value.ParseFromString(unescaped)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unable to parse serialized `google.protobuf.Struct` value: ",
        unescaped));
  }

  return absl::OkStatus();
}

}  // namespace ml_metadata
