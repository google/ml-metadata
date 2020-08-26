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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_READ_EVENTS_WORKLOAD_H
#define ML_METADATA_TOOLS_MLMD_BENCH_READ_EVENTS_WORKLOAD_H

#include "absl/types/variant.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// Defines a ReadEventsWorkItemType that can be different requests to look for
// MLMD Events by Artifact / Execution IDs.
using ReadEventsWorkItemType = absl::variant<GetEventsByArtifactIDsRequest,
                                             GetEventsByExecutionIDsRequest>;

// A specific workload for getting Input / Output Events by Artifact / Execution
// IDs.
class ReadEvents : public Workload<ReadEventsWorkItemType> {
 public:
  ReadEvents(const ReadEventsConfig& read_events_config, int64 num_operations);
  ~ReadEvents() override = default;

 protected:
  // Specific implementation of SetUpImpl() for ReadEvents workload according to
  // its semantic. A list of work items(ReadEventsWorkItemType) will be
  // generated. The APIs traverse from nodes {Artifact, Execution} to Events. To
  // finalize the query string, we choose existing nodes uniformly. Returns
  // detailed error if query executions failed.
  tensorflow::Status SetUpImpl(MetadataStore* store) final;

  // Specific implementation of RunOpImpl() for ReadEvents workload according to
  // its semantic. Runs the work items(ReadEventsWorkItemType) on the store.
  // Returns detailed error if query executions failed.
  tensorflow::Status RunOpImpl(int64 work_items_index,
                               MetadataStore* store) final;

  // Specific implementation of TearDownImpl() for ReadEvents workload according
  // to its semantic. Cleans the work items.
  tensorflow::Status TearDownImpl() final;

  // Gets the current workload's name, which is used in stats report for this
  // workload.
  std::string GetName() final;

 private:
  // Workload configurations specified by the users.
  const ReadEventsConfig read_events_config_;
  // Number of operations for the current workload.
  const int64 num_operations_;
  // String for indicating the name of current workload instance.
  const std::string name_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_READ_EVENTS_WORKLOAD_H
