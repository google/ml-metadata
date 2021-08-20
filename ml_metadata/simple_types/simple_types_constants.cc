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
#include "ml_metadata/simple_types/simple_types_constants.h"

#include <array>

#include "absl/strings/string_view.h"

namespace ml_metadata {

// LINT.IfChange
const absl::string_view kSimpleTypes = R"pb(
  artifact_types { name: "mlmd.Dataset" }
  artifact_types { name: "mlmd.Model" }
  artifact_types { name: "mlmd.Metrics" }
  artifact_types { name: "mlmd.Statistics" }
  execution_types { name: "mlmd.Train" }
  execution_types { name: "mlmd.Transform" }
  execution_types { name: "mlmd.Process" }
  execution_types { name: "mlmd.Evaluate" }
  execution_types { name: "mlmd.Deploy" }
)pb";
// LINT.ThenChange(../proto/metadata_store.proto,
//                 ../metadata_store/simple_types_util.cc)

const std::array<absl::string_view, kNumSimpleTypes> kSimpleTypeNames = {
    "mlmd.Dataset",    "mlmd.Model",    "mlmd.Metrics",
    "mlmd.Statistics", "mlmd.Train",    "mlmd.Transform",
    "mlmd.Process",    "mlmd.Evaluate", "mlmd.Deploy"};

}  // namespace ml_metadata
