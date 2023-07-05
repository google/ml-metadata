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
#include "ml_metadata/util/field_mask_utils.h"

#include <algorithm>

#include "google/protobuf/field_mask.pb.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "google/protobuf/descriptor.h"

namespace ml_metadata {

namespace {
constexpr absl::string_view kPropertiesFieldPathPrefix = "properties";
constexpr absl::string_view kCustomPropertiesFieldPathPrefix =
    "custom_properties";
}  // namespace

absl::StatusOr<absl::flat_hash_set<absl::string_view>> GetPropertyNamesFromMask(
    const google::protobuf::FieldMask& mask, const bool is_custom_properties) {
  absl::flat_hash_set<absl::string_view> property_names;
  absl::string_view prefix = is_custom_properties
                                 ? kCustomPropertiesFieldPathPrefix
                                 : kPropertiesFieldPathPrefix;
  for (absl::string_view path : mask.paths()) {
    if (path == prefix) {
      return absl::InternalError(
          absl::StrCat("Cannot split property names from ", prefix,
                       " if no property name is specified for ", prefix,
                       " field. Please update ", prefix, " field as a whole."));
    }
    if (absl::StartsWith(path, prefix)) {
      property_names.insert(absl::StripPrefix(path, absl::StrCat(prefix, ".")));
    }
  }
  return property_names;
}

absl::StatusOr<absl::flat_hash_set<absl::string_view>>
GetPropertyNamesFromMaskOrUnionOfProperties(
    const google::protobuf::FieldMask& mask, const bool is_custom_properties,
    const google::protobuf::Map<std::string, Value>& curr_properties,
    const google::protobuf::Map<std::string, Value>& prev_properties) {
  absl::flat_hash_set<absl::string_view> property_names_in_properties_union;
  for (const auto& property : curr_properties) {
    property_names_in_properties_union.insert(property.first);
  }
  for (const auto& property : prev_properties) {
    property_names_in_properties_union.insert(property.first);
  }
  if (mask.paths().empty()) {
    return property_names_in_properties_union;
  }

  absl::StatusOr<absl::flat_hash_set<absl::string_view>>
      property_names_in_mask_or_internal_error =
          GetPropertyNamesFromMask(mask, is_custom_properties);
  if (property_names_in_mask_or_internal_error.status().code() ==
      absl::StatusCode::kInternal) {
    return property_names_in_properties_union;
  }

  return property_names_in_mask_or_internal_error;
}

absl::StatusOr<google::protobuf::FieldMask> GetFieldsSubMaskFromMask(
    const google::protobuf::FieldMask& mask,
    const google::protobuf::Descriptor* descriptor) {
  if (descriptor == nullptr) {
    return absl::InvalidArgumentError("`descriptor` cannot be null.");
  }
  google::protobuf::FieldMask submask;

  std::copy_if(mask.paths().begin(), mask.paths().end(),
               google::protobuf::RepeatedFieldBackInserter(submask.mutable_paths()),
               [descriptor](absl::string_view p) {
                 return !absl::StartsWith(p, kPropertiesFieldPathPrefix) &&
                        !absl::StartsWith(p,
                                          kCustomPropertiesFieldPathPrefix) &&
                        descriptor->FindFieldByName(p.data()) != nullptr;
               });
  return submask;
}
}  // namespace ml_metadata
