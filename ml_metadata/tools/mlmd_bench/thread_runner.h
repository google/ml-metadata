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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_THREAD_RUNNER_H
#define ML_METADATA_TOOLS_MLMD_BENCH_THREAD_RUNNER_H

#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/tools/mlmd_bench/benchmark.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// The ThreadRunner class is the execution component of the `mlmd_bench`
// It takes the benchmark and runs the workloads.
class ThreadRunner {
 public:
  ThreadRunner(const ConnectionConfig& mlmd_config, int64 num_threads);
  ~ThreadRunner() = default;

  // Execution unit of `mlmd_bench`. Returns detailed error if query executions
  // failed.
  tensorflow::Status Run(Benchmark& benchmark);

 private:
  // Connection configuration that will be used to create the MetadataStore.
  const ConnectionConfig mlmd_config_;
  // Number of threads for the thread runner.
  const int64 num_threads_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_THREAD_RUNNER_H
