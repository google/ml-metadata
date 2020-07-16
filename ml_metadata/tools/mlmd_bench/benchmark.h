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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_BENCHMARK_H
#define ML_METADATA_TOOLS_MLMD_BENCH_BENCHMARK_H

#include <vector>

#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"

namespace ml_metadata {

// Contains a list of workloads to be executed by ThreadRunner.
// The executable workloads are generated according to `mlmd_bench_config`.
class Benchmark {
 public:
  Benchmark(const MLMDBenchConfig& mlmd_bench_config);
  ~Benchmark();

  // Takes `workload_config` and generates corresponding executable workloads.
  void CreateWorkload(const WorkloadConfig& workload_config);

  std::vector<std::pair<std::unique_ptr<WorkloadBase>, int64>>& workloads();

 private:
  // A list of executable workloads and their according number of operations.
  std::vector<std::pair<std::unique_ptr<WorkloadBase>, int64>> workloads_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_BENCHMARK_H
