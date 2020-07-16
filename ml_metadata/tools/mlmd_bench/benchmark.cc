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

Benchmark::Benchmark(const MLMDBenchConfig& mlmd_bench_config) {
  // For each `workload_config`, calls CreateWorkload() to create corresponding
  // workload.
  for (const WorkloadConfig& workload_config :
       mlmd_bench_config.workload_configs()) {
    CreateWorkload(workload_config);
  }
}

Benchmark::~Benchmark() {
  for (auto& workload : workloads_) {
    workload.first.reset();
  }
}

void Benchmark::CreateWorkload(const WorkloadConfig& workload_config) {
  if (workload_config.has_fill_types_config()) {
    std::unique_ptr<FillTypes> fill_types(new FillTypes(
        workload_config.fill_types_config(), workload_config.num_operations()));
    workloads_.push_back(std::make_pair(std::move(fill_types),
                                        workload_config.num_operations()));
  } else {
    LOG(FATAL) << "Cannot find corresponding workload!";
  }
}

std::vector<std::pair<std::unique_ptr<WorkloadBase>, int64>>&
Benchmark::workloads() {
  return workloads_;
}

}  // namespace ml_metadata
