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
#include "absl/status/status.h"
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

}  // namespace ml_metadata
