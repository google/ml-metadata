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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_FILL_EVENTS_WORKLOAD_H
#define ML_METADATA_TOOLS_MLMD_BENCH_FILL_EVENTS_WORKLOAD_H

#include "absl/container/flat_hash_set.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// A specific workload for insert input / output events.
class FillEvents : public Workload<PutEventsRequest> {
 public:
  FillEvents(const FillEventsConfig& fill_events_config, int64 num_operations);
  ~FillEvents() override = default;

 protected:
  // Specific implementation of SetUpImpl() for FillEvents workload according to
  // its semantic. A list of work items(PutEventsRequest) will be generated.
  // A preferential attachment will be performed to generate the edges for the
  // bipartite graph: Artifact X Execution. Execution will be selected first, as
  // MLMD execution is a record of a component run at a particular time, its
  // popularity is modeled as a categorical distribution with a Dirichlet
  // prior. When picking the artifact node, it is picked w.r.t. the
  // `artifact_node_popularity`. It can be a categorical distribution with a
  // Dirichlet prior when the event is an output event. On the other hand, if
  // current event's type is input, `artifact_node_popularity` will be a zipf
  // distribution with a configurable skew will be used to model the popularity
  // of heavily used input artifacts. Additionally, for the output event, since
  // each artifact can only be picked once, a rejection sampling will be
  // performed in the process.
  // Returns detailed error if query executions failed.
  tensorflow::Status SetUpImpl(MetadataStore* store) final;

  // Specific implementation of RunOpImpl() for FillEvents workload according to
  // its semantic. Runs the work items(PutEventsRequest) on the store.
  // Returns detailed error if query executions failed.
  tensorflow::Status RunOpImpl(int64 work_items_index,
                               MetadataStore* store) final;

  // Specific implementation of TearDownImpl() for FillEvents workload according
  // to its semantic. Cleans the work items.
  tensorflow::Status TearDownImpl() final;

  // Gets the current workload's name, which is used in stats report for this
  // workload.
  std::string GetName() final;

 private:
  // Workload configurations specified by the users.
  const FillEventsConfig fill_events_config_;
  // Number of operations for the current workload.
  const int64 num_operations_;
  // String for indicating the name of current workload instance.
  const std::string name_;
  // Records all the outputted artifacts' ids seen in current setup.
  absl::flat_hash_set<int64> output_artifact_ids_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_FILL_EVENTS_WORKLOAD_H
