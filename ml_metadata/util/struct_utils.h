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
#ifndef THIRD_PARTY_ML_METADATA_UTIL_STRUCT_UTILS_H_
#define THIRD_PARTY_ML_METADATA_UTIL_STRUCT_UTILS_H_

#include <string>

#include "google/protobuf/struct.pb.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace ml_metadata {

// Returns true if |serialized_value| is a `google.protobuf.Struct` serialized
// for storage.
bool IsStructSerializedString(absl::string_view serialized_value);

// Returns a serialized string representing |struct_value| suitable for storage
// in MLMD's Property table.
std::string StructToString(const google::protobuf::Struct& struct_value);

// Returns the corresponding `google.protobuf.Struct` deserialized from
// |serialized_value| in |struct_value|.
// Returns INVALID_ARGUMENT if the value in |serialized_value| does not look
// like a valid string serialized using |StructToString| above.
absl::Status StringToStruct(absl::string_view serialized_value,
                            google::protobuf::Struct& struct_value);

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_UTIL_STRUCT_UTILS_H_
