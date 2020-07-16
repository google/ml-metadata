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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_WORKLOAD_H
#define ML_METADATA_TOOLS_MLMD_BENCH_WORKLOAD_H

#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/tools/mlmd_bench/stats.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// A base class for all Workloads. Each workload subclass takes a config.,
// creates the datasets, then for each work item of the dataset, it runs an
// operation against MLMD, and measures its performance.
class WorkloadBase {
 public:
  WorkloadBase() = default;
  virtual ~WorkloadBase() = default;

  // Prepares a list of work items in memory. It may reads db to prepare the
  // work items.
  virtual tensorflow::Status SetUp(MetadataStore* store) = 0;

  // Runs the operation related to the workload and measures performance for the
  // workload operation on individual work item on MLMD.
  virtual tensorflow::Status RunOp(int64 i, MetadataStore* store,
                                   OpStats& op_stats) = 0;

  // Cleans the list of work items and related resources.
  virtual tensorflow::Status TearDown() = 0;

  // Gets the number of operations for current workload.
  virtual int64 num_operations() = 0;

  // Gets the current workload's name, which is used in stats report for this
  // workload.
  virtual std::string GetName() = 0;
};

// A base class for all specific workloads (FillTypes, FillNodes, ...).
// It is a template class where WorkItemType defines the type for the list of
// work items prepared in SetUp().
template <typename WorkItemType>
class Workload : public WorkloadBase {
 public:
  Workload() : is_setup_(false) {}
  virtual ~Workload() = default;

  // Prepares datasets related to a workload. It may read the information in the
  // store. The method must run before RunOp() / TearDown() to isolate the data
  // preparation operations with the operations to be measured. The subclass
  // should implement SetUpImpl(). The given store should be not null and
  // connected. Returns detailed error if query executions failed.
  tensorflow::Status SetUp(MetadataStore* store) final {
    TF_RETURN_IF_ERROR(SetUpImpl(store));
    // Set the is_setup_ to true for ensuring correct execution sequence.
    is_setup_ = true;
    return tensorflow::Status::OK();
  }

  // Runs the operation of the workload on a work item at `work_items_index` on
  // the store. The operation is measured and kept in `op_stats`. The subclass
  // should implement RunOpImpl(), and does not perform irrelevant operations to
  // avoid being counted in `op_stats`.
  // Returns FailedPrecondition error, if SetUp() is not finished before
  // running the operation.
  // Returns InvalidArgument error, if the `work_items_index` is invalid.
  // Returns detailed error if query executions failed.
  tensorflow::Status RunOp(const int64 work_items_index, MetadataStore* store,
                           OpStats& op_stats) final {
    // Checks is_setup to ensure execution sequence.
    if (!is_setup_) {
      return tensorflow::errors::FailedPrecondition("Set up is not finished!");
    }
    // Check if the work item index i is valid.
    if (work_items_index < 0 || work_items_index >= (int64)work_items_.size()) {
      return tensorflow::errors::InvalidArgument("Work item index invalid!");
    }
    absl::Time start_time = absl::Now();
    TF_RETURN_IF_ERROR(RunOpImpl(work_items_index, store));
    // Each operation will have an op_stats to record the statistic of the
    // current single operation.
    op_stats.elapsed_time = absl::Now() - start_time;
    op_stats.transferred_bytes = work_items_[work_items_index].second;
    return tensorflow::Status::OK();
  }

  // Cleans the list of work items and related resources.
  // The cleaning operation will not be included for performance measurement.
  // The subclass should implement TearDownImpl(). Returns Failed Precondition
  // error, if SetUp() is not finished before running the operation. Returns
  // detailed error if query executions failed.
  tensorflow::Status TearDown() final {
    // Checks is_setup to ensure execution sequence.
    if (!is_setup_) {
      return tensorflow::errors::FailedPrecondition("Set up is not finished!");
    }
    TF_RETURN_IF_ERROR(TearDownImpl());
    return tensorflow::Status::OK();
  }

  int64 num_operations() final { return work_items_.size(); }

 protected:
  // The implementation of the SetUp(). It is called in SetUp() and responsible
  // for preparing the work_item_ for RunOpImpl()'s execution. The detail
  // implementation will depend on each specific workload's semantic. Returns
  // detailed error if query executions failed.
  virtual tensorflow::Status SetUpImpl(MetadataStore* store) = 0;

  // The implementation of the RunOp(). It is called in RunOp() and responsible
  // for executing the work_item_ prepared in SetUpImpl(). The detail
  // implementation will depend on each specific workload's semantic. Returns
  // detailed error if query executions failed.
  virtual tensorflow::Status RunOpImpl(int64 work_items_index,
                                       MetadataStore* store) = 0;

  // The implementation of the TearDown(). It is called in TearDown() and
  // responsible for cleaning the work_item_ and related resources. The detail
  // implementation will depend on each specific workload's semantic. Returns
  // detailed error if query executions failed.
  virtual tensorflow::Status TearDownImpl() = 0;

  // Gets the current workload's name, which is used in stats report for this
  // workload.
  virtual std::string GetName() = 0;

  // Boolean for indicating whether the work items have been prepared or not. It
  // will be used to ensure the right execution sequence. SetUp() will set it to
  // true and RunOp() and TearDown() will check its state. If it is false, then
  // Failed Precondition error will be returned for RunOp() and TearDown().
  bool is_setup_;

  // The work items for a workload. It is a vector of pairs where each pair
  // consists of each individual work item and the transferred bytes to the
  // database. It is created in SetUpImpl(), and each RunOpImpl()
  // processes one at a time.
  std::vector<std::pair<WorkItemType, int64>> work_items_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_WORKLOAD_H
