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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_READ_TYPES_WORKLOAD_H
#define ML_METADATA_TOOLS_MLMD_BENCH_READ_TYPES_WORKLOAD_H

#include "absl/types/variant.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// Defines a ReadTypesWorkItemType that can be different requests to look for
// MLMD types.
using ReadTypesWorkItemType =
    ::absl::variant<GetArtifactTypesRequest, GetExecutionTypesRequest,
                    GetContextTypesRequest, GetArtifactTypesByIDRequest,
                    GetExecutionTypesByIDRequest, GetContextTypesByIDRequest,
                    GetArtifactTypeRequest, GetExecutionTypeRequest,
                    GetContextTypeRequest>;

// A specific workload for getting types: ArtifactTypes / ExecutionTypes /
// ContextTypes.
class ReadTypes : public Workload<ReadTypesWorkItemType> {
 public:
  ReadTypes(const ReadTypesConfig& read_types_config, int64 num_operations);
  ~ReadTypes() override = default;

 protected:
  // Specific implementation of SetUpImpl() for ReadTypes workload according to
  // its semantic. A list of work items(ReadTypesWorkItemType) will be
  // generated. When querying types by parameters, select from existing types
  // uniformly to finalize the query string. If the specification of current
  // workload is ARTIFACT_TYPES_BY_IDs / EXECUTION_TYPES_BY_IDs /
  // CONTEXT_TYPES_BY_IDs, the number of ids per request will be generated
  // w.r.t. the uniform distribution `maybe_num_ids`. Returns detailed error if
  // query executions failed.
  tensorflow::Status SetUpImpl(MetadataStore* store) final;

  // Specific implementation of RunOpImpl() for ReadTypes workload according to
  // its semantic. Runs the work items(ReadTypesWorkItemType) on the store.
  // Returns detailed error if query executions failed.
  tensorflow::Status RunOpImpl(int64 work_items_index,
                               MetadataStore* store) final;

  // Specific implementation of TearDownImpl() for ReadTypes workload according
  // to its semantic. Cleans the work items.
  tensorflow::Status TearDownImpl() final;

  // Gets the current workload's name, which is used in stats report for this
  // workload.
  std::string GetName() final;

 private:
  // Workload configurations specified by the users.
  const ReadTypesConfig read_types_config_;
  // Number of operations for the current workload.
  const int64 num_operations_;
  // String for indicating the name of current workload instance.
  const std::string name_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_READ_TYPES_WORKLOAD_H
