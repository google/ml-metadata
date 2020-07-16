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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_UTIL_H
#define ML_METADATA_TOOLS_MLMD_BENCH_UTIL_H

#include <vector>

#include "absl/types/variant.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// Defines a Type can be ArtifactType / ExecutionType / ContextType.
using Type = absl::variant<ArtifactType, ExecutionType, ContextType>;

// Gets all the existing types (the specific types that indicated by
// `fill_types_config`) inside db and store them into `existing_types`.
// Returns detailed error if query executions failed.
tensorflow::Status GetExistingTypes(const FillTypesConfig& fill_types_config,
                                    MetadataStore* store,
                                    std::vector<Type>& existing_types) {
  switch (fill_types_config.specification()) {
    case FillTypesConfig::ARTIFACT_TYPE: {
      GetArtifactTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetArtifactTypes(
          /*request=*/{}, &get_response));
      for (auto& artifact_type : get_response.artifact_types()) {
        existing_types.push_back(artifact_type);
      }
      break;
    }
    case FillTypesConfig::EXECUTION_TYPE: {
      GetExecutionTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetExecutionTypes(
          /*request=*/{}, &get_response));
      for (auto& execution_type : get_response.execution_types()) {
        existing_types.push_back(execution_type);
      }
      break;
    }
    case FillTypesConfig::CONTEXT_TYPE: {
      GetContextTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetContextTypes(
          /*request=*/{}, &get_response));
      for (auto& context_type : get_response.context_types()) {
        existing_types.push_back(context_type);
      }
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillTypes!";
  }
  return tensorflow::Status::OK();
}

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_UTIL_H
