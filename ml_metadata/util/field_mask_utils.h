/* Copyright 2023 Google LLC

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
#ifndef THIRD_PARTY_ML_METADATA_UTIL_FIELD_MASK_UTILS_H_
#define THIRD_PARTY_ML_METADATA_UTIL_FIELD_MASK_UTILS_H_

#include "google/protobuf/field_mask.pb.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "google/protobuf/descriptor.h"

namespace ml_metadata {

// Gets names of `properties` or `custom_properties` from `mask`.
// For example, if `mask` contains {"properties.x", "properties.y",
// "external_id"} and `is_custom_properties` is false, then the splited results
// will be: {"x", "y"}.
// Returns INTERNAL error if `mask` contains the field path for
// "properties" without specifying the names in it, e.g. {"properties"}, and
// `is_custom_properties` is false.
// Returns INTERNAL error if `mask` contains the field path for
// "custom_properties" without specifying the names in it ,
// e.g. {"custom_properties"}, and `is_custom_properties` is true.
absl::StatusOr<absl::flat_hash_set<absl::string_view>> GetPropertyNamesFromMask(
    const google::protobuf::FieldMask& mask, bool is_custom_properties);

// Gets names of `properties` or `custom_properties` from `mask` or the union of
// `curr_properties` and `prev_properties` if `mask` is empty.
// For example:
// If `mask` contains {"properties.x", "properties.y", "external_id"} and
// `is_custom_properties` is false, then the result will contain: {"x", "y"}
// If `mask` contains {"external_id"} (does not contain property names),
// then the result will be empty.
// If `mask` is empty, the the result will contain the union of property names
// from both `curr_properties` and `prev_properties`.
// Returns INTERNAL error if any error occurs.
absl::StatusOr<absl::flat_hash_set<absl::string_view>>
GetPropertyNamesFromMaskOrUnionOfProperties(
    const google::protobuf::FieldMask& mask, const bool is_custom_properties,
    const google::protobuf::Map<std::string, Value>& curr_properties,
    const google::protobuf::Map<std::string, Value>& prev_properties);

// Gets a valid submask only containing paths of fields from `mask`.
// For example, if `mask` contains {"properties.x", "properties.y",
// "external_id", "invalid_filed_path"} , then the submask will be:
// {"external_id"}.
// Returns INVALID_ARGUMENT error if `descriptor` is a null-pointer.
absl::StatusOr<google::protobuf::FieldMask> GetFieldsSubMaskFromMask(
    const google::protobuf::FieldMask& mask,
    const google::protobuf::Descriptor* descriptor);

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_UTIL_FIELD_MASK_UTILS_H_
