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
#ifndef THIRD_PARTY_ML_METADATA_UTIL_STATUS_UTILS_H_
#define THIRD_PARTY_ML_METADATA_UTIL_STATUS_UTILS_H_

#include "absl/status/status.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// Transforms an absl status to its tensorflow status form and returns it.
tensorflow::Status FromABSLStatus(const absl::Status& s);

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_UTIL_STATUS_UTILS_H_
