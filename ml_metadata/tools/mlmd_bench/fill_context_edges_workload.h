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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_FILL_CONTEXT_EDGES_WORKLOAD_H
#define ML_METADATA_TOOLS_MLMD_BENCH_FILL_CONTEXT_EDGES_WORKLOAD_H

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// A specific workload for insert context edges: Attributions / Associations.
class FillContextEdges
    : public Workload<PutAttributionsAndAssociationsRequest> {
 public:
  FillContextEdges(const FillContextEdgesConfig& fill_context_edges_config,
                   int64 num_operations);
  ~FillContextEdges() override = default;

 protected:
  // Specific implementation of SetUpImpl() for FillContextEdges workload
  // according to its semantic.
  // A list of work items(PutAttributionsAndAssociationsRequest) will be
  // generated.
  // A preferential attachment will be performed to generate the edges for the
  // bipartite graph: Context X Artifact or Context X Execution. The context
  // nodes and non-context nodes will be selected according to specified node
  // popularity. If the current context / non-context pair is seen before in
  // current SetUpImpl(), a rejection sampling will be performed. Here, we do
  // not check duplicate context edge inside db. If the context edge
  // exists inside db, the query will do nothing and this is also an expected
  // behavior that also should be included in the performance
  // measurement.
  // Returns detailed error if query executions failed.
  tensorflow::Status SetUpImpl(MetadataStore* store) final;

  // Specific implementation of RunOpImpl() for FillContextEdges workload
  // according to its semantic.
  // Runs the work items(PutAttributionsAndAssociationsRequest) on the store.
  // Returns detailed error if query executions failed.
  tensorflow::Status RunOpImpl(int64 work_items_index,
                               MetadataStore* store) final;

  // Specific implementation of TearDownImpl() for FillContextEdges workload
  // according to its semantic. Cleans the work items.
  tensorflow::Status TearDownImpl() final;

  // Gets the current workload's name, which is used in stats report for this
  // workload.
  std::string GetName() final;

 private:
  // Workload configurations specified by the users.
  const FillContextEdgesConfig fill_context_edges_config_;
  // Number of operations for the current workload.
  const int64 num_operations_;
  // String for indicating the name of current workload instance.
  const std::string name_;
  // Records all the context and non context id pairs seen in current setup.
  absl::flat_hash_map<int64, absl::flat_hash_set<int64>> edges_seen_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_FILL_CONTEXT_EDGES_WORKLOAD_H
