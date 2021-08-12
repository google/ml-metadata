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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_SIMPLE_TYPES_UTIL_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_SIMPLE_TYPES_UTIL_H_

#include "google/protobuf/descriptor.pb.h"
#include "absl/status/status.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/simple_types/proto/simple_types.pb.h"

namespace ml_metadata {

// A util to create a SimpleTypes proto from a constant string at runtime.
absl::Status LoadSimpleTypes(SimpleTypes& simple_types);

// Returns the SystemTypeExtension of the enum field base_type in input 'type'.
template <typename T>
absl::Status GetSystemTypeExtension(const T& type,
                                    SystemTypeExtension& output_extension) {
  if (!type.has_base_type()) {
    return absl::NotFoundError("base_type field is null");
  }
  const std::string enum_field_name = "base_type";
  auto* message_descriptor = type.GetDescriptor();
  auto* field_descriptor = message_descriptor->FindFieldByName(enum_field_name);
  if (!field_descriptor) {
    return absl::InvalidArgumentError(
        absl::StrCat("Field ", enum_field_name, " is missing in ",
                     message_descriptor->full_name()));
  }
  const int enum_value =
      type.GetReflection()->GetEnumValue(type, field_descriptor);

  auto* enum_descriptor =
      field_descriptor->enum_type()->FindValueByNumber(enum_value);
  if (!enum_descriptor) {
    return absl::InvalidArgumentError(
        absl::StrCat("Enum value of ", enum_field_name, " is: ", enum_value,
                     ". Failed to get its enum descriptor"));
  }

  if (!enum_descriptor->options().HasExtension(
          ml_metadata::system_type_extension)) {
    return absl::InvalidArgumentError(absl::StrCat(
        enum_field_name, " does not have extension to enum value options"));
  }
  output_extension = enum_descriptor->options().GetExtension(
      ml_metadata::system_type_extension);
  return absl::OkStatus();
}

// Asserts whether the symtem type is UNSET based on 'extension'.
bool IsUnsetBaseType(const SystemTypeExtension& extension);

// Gets SystemDefinedArtifactType enum based on artifact type 'extension'.
absl::Status GetSystemTypeEnum(const SystemTypeExtension& extension,
                               ArtifactType::SystemDefinedBaseType& type_enum);

// Gets SystemDefinedExecutionType enum based on execution type 'extension'.
absl::Status GetSystemTypeEnum(const SystemTypeExtension& extension,
                               ExecutionType::SystemDefinedBaseType& type_enum);

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_SIMPLE_TYPES_UTIL_H_
