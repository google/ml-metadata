/* Copyright 2019 Google LLC

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
#include "ml_metadata/metadata_store/simple_types_util.h"

#include "google/protobuf/text_format.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/simple_types/proto/simple_types.pb.h"
#include "ml_metadata/simple_types/simple_types_constants.h"

namespace ml_metadata {

absl::Status LoadSimpleTypes(SimpleTypes& simple_types) {
  if (!google::protobuf::TextFormat::ParseFromString(std::string(kSimpleTypes),
                                           &simple_types)) {
    return absl::InvalidArgumentError(
        "Failed to parse simple types from string");
  }
  return absl::OkStatus();
}

bool IsUnsetBaseType(const SystemTypeExtension& extension) {
  return absl::StartsWith(extension.type_name(), "unset");
}

absl::Status GetSystemTypeEnum(const SystemTypeExtension& extension,
                               ArtifactType::SystemDefinedBaseType& type_enum) {
  static const auto& type_name_to_enum =
      *new absl::flat_hash_map<std::string,
                               ArtifactType::SystemDefinedBaseType>(
          {{"unset_artifact_type", ArtifactType::UNSET},
           {"mlmd.Dataset", ArtifactType::DATASET},
           {"mlmd.Model", ArtifactType::MODEL},
           {"mlmd.Metrics", ArtifactType::METRICS},
           {"mlmd.Statistics", ArtifactType::STATISTICS}});

  if (type_name_to_enum.contains(extension.type_name())) {
    type_enum = type_name_to_enum.at(extension.type_name());
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("invalid system type name: ", extension.type_name()));
}

absl::Status GetSystemTypeEnum(
    const SystemTypeExtension& extension,
    ExecutionType::SystemDefinedBaseType& type_enum) {
  static const auto& type_name_to_enum =
      *new absl::flat_hash_map<std::string,
                               ExecutionType::SystemDefinedBaseType>(
          {{"unset_execution_type", ExecutionType::UNSET},
           {"mlmd.Train", ExecutionType::TRAIN},
           {"mlmd.Transform", ExecutionType::TRANSFORM},
           {"mlmd.Process", ExecutionType::PROCESS},
           {"mlmd.Evaluate", ExecutionType::EVALUATE},
           {"mlmd.Deploy", ExecutionType::DEPLOY}});

  if (type_name_to_enum.contains(extension.type_name())) {
    type_enum = type_name_to_enum.at(extension.type_name());
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("invalid system type name: ", extension.type_name()));
}

}  // namespace ml_metadata
