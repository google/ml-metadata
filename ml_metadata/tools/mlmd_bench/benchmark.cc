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

#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/tools/mlmd_bench/fill_types_workload.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"
#include "tensorflow/core/platform/logging.h"

namespace ml_metadata {
namespace {

// Creates the executable workload given `workload_config`.
void CreateWorkload(const WorkloadConfig& workload_config,
                    std::unique_ptr<WorkloadBase>& workload) {
  if (!workload_config.has_fill_types_config()) {
    LOG(FATAL) << "Cannot find corresponding workload!";
  }
  workload = absl::make_unique<FillTypes>(FillTypes(
      workload_config.fill_types_config(), workload_config.num_operations()));
}

}  // namespace

Benchmark::Benchmark(const MLMDBenchConfig& mlmd_bench_config) {
  workloads_.resize(mlmd_bench_config.workload_configs_size());

  // For each `workload_config`, calls CreateWorkload() to create corresponding
  // workload.
  for (int i = 0; i < mlmd_bench_config.workload_configs_size(); ++i) {
    CreateWorkload(mlmd_bench_config.workload_configs(i), workloads_[i]);
  }
}

WorkloadBase* Benchmark::workload(const int64 workload_index) {
  return workloads_[workload_index].get();
}

}  // namespace ml_metadata
