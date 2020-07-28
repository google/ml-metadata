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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_FILL_NODES_WORKLOAD_H
#define ML_METADATA_TOOLS_MLMD_BENCH_FILL_NODES_WORKLOAD_H

#include "absl/types/variant.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// Defines a FillNodesWorkItemType that can be PutArtifactsRequest /
// PutExecutionsRequest / PutContextsRequest.
using FillNodesWorkItemType =
    absl::variant<PutArtifactsRequest, PutExecutionsRequest,
                  PutContextsRequest>;

// A specific workload for creating and updating nodes: Artifacts / Executions /
// Contexts.
class FillNodes : public Workload<FillNodesWorkItemType> {
 public:
  FillNodes(const FillNodesConfig& fill_nodes_config, int64 num_operations);
  ~FillNodes() override = default;

 protected:
  // Specific implementation of SetUpImpl() for FillNodes workload according to
  // its semantic.
  // For inserting, it will generate the list of work items (FillNodesRequests)
  // by selecting a registered type randomly and generate the number of
  // properties for the node w.r.t. the uniform distribution. If the number of
  // properties is greater than the type, use custom properties instead. When
  // generating string-valued properties, use `string_value_bytes` to determine
  // the length of the value. Returns detailed error if query executions failed.
  tensorflow::Status SetUpImpl(MetadataStore* store) final;

  // Specific implementation of RunOpImpl() for FillNodes workload according to
  // its semantic. Runs the work items(FillNodesRequests) on the store. Returns
  // detailed error if query executions failed.
  tensorflow::Status RunOpImpl(int64 work_items_index,
                               MetadataStore* store) final;

  // Specific implementation of TearDownImpl() for FillNodes workload according
  // to its semantic. Cleans the work items.
  tensorflow::Status TearDownImpl() final;

  // Gets the current workload's name, which is used in stats report for this
  // workload.
  std::string GetName() final;

 private:
  // Workload configurations specified by the users.
  const FillNodesConfig fill_nodes_config_;
  // Number of operations for the current workload.
  const int64 num_operations_;
  // String for indicating the name of current workload instance.
  std::string name_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_FILL_NODES_WORKLOAD_H
