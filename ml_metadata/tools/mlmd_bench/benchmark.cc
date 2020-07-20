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
#include "ml_metadata/tools/mlmd_bench/benchmark.h"

#include <vector>

#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/tools/mlmd_bench/fill_types_workload.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"
#include "tensorflow/core/platform/logging.h"

namespace ml_metadata {
namespace {

void CreateWorkload(const WorkloadConfig& workload_config,
                    std::vector<std::unique_ptr<WorkloadBase>>& workloads,
                    std::vector<int64>& workloads_num_operations) {
  if (workload_config.has_fill_types_config()) {
    std::unique_ptr<FillTypes> fill_types(new FillTypes(
        workload_config.fill_types_config(), workload_config.num_operations()));
    workloads.push_back(std::move(fill_types));
    workloads_num_operations.push_back(workload_config.num_operations());
  } else {
    LOG(FATAL) << "Cannot find corresponding workload!";
  }
}

}  // namespace

Benchmark::Benchmark(const MLMDBenchConfig& mlmd_bench_config) {
  // For each `workload_config`, calls CreateWorkload() to create corresponding
  // workload.
  for (const WorkloadConfig& workload_config :
       mlmd_bench_config.workload_configs()) {
    CreateWorkload(workload_config, workloads_, workloads_num_operations_);
  }
}

WorkloadBase* Benchmark::workload(const int64 workload_index) {
  return workloads_[workload_index].get();
}

int64 Benchmark::workload_num_operations(const int64 workload_index) {
  return workloads_num_operations_[workload_index];
}

int64 Benchmark::workloads_size() { return workloads_.size(); }

}  // namespace ml_metadata
