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
#include <fcntl.h>

#include <fstream>
#include <iostream>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "ml_metadata/tools/mlmd_bench/benchmark.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/thread_runner.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace ml_metadata {
namespace {

// Initializes the `mlmd_bench_config` with the specified configuration .pbtxt
// file.
tensorflow::Status InitMLMDBenchConfigFromPbtxtFile(
    char** argv, MLMDBenchConfig& mlmd_bench_config) {
  const std::string config_pbtxt_path = argv[1];
  if (tensorflow::Env::Default()->FileExists(config_pbtxt_path).ok()) {
    return ReadTextProto(tensorflow::Env::Default(), config_pbtxt_path,
                         &mlmd_bench_config);
  }
  return tensorflow::errors::NotFound(
      "Could not find mlmd_bench configuration .pb or "
      ".pbtxt at supplied directory path: ",
      config_pbtxt_path);
}

// Initializes the `mlmd_bench_report` with `mlmd_bench_config`.
void InitMLMDBenchReport(MLMDBenchConfig& mlmd_bench_config,
                         MLMDBenchReport& mlmd_bench_report) {
  for (auto& workload_config : mlmd_bench_config.workload_configs()) {
    mlmd_bench_report.add_summaries()->mutable_workload_config()->CopyFrom(
        workload_config);
  }
}

// Writes the performance report into a specified file.
tensorflow::Status WriteProtoResultToDisk(char** argv,
                                          MLMDBenchReport& mlmd_bench_report) {
  const std::string output_path = argv[2];
  if (tensorflow::Env::Default()->IsDirectory(output_path).ok()) {
    return tensorflow::WriteTextProto(
        tensorflow::Env::Default(),
        absl::StrCat(output_path, "mlmd_bench_report.txt"), mlmd_bench_report);
  }
  return tensorflow::errors::NotFound(
      "Could not find valid output directory path as: ", output_path);
}

}  // namespace
}  // namespace ml_metadata

int main(int argc, char** argv) {
  // Configurations for mlmd_bench.
  ml_metadata::MLMDBenchConfig mlmd_bench_config;
  tensorflow::Status init_config_status =
      ml_metadata::InitMLMDBenchConfigFromPbtxtFile(argv, mlmd_bench_config);
  if (!init_config_status.ok()) {
    LOG(WARNING) << init_config_status;
  }

  // Performance report for workloads specified inside `mlmd_bench_config`.
  ml_metadata::MLMDBenchReport mlmd_bench_report;
  ml_metadata::InitMLMDBenchReport(mlmd_bench_config, mlmd_bench_report);

  // Feeds the config. into the benchmark for generating executable workloads.
  ml_metadata::Benchmark benchmark(mlmd_bench_config);
  // Executes the workloads inside the benchmark with the thread runner.
  ml_metadata::ThreadRunner runner(
      mlmd_bench_config.mlmd_config(),
      mlmd_bench_config.thread_env_config().num_threads());
  tensorflow::Status execuate_status = runner.Run(benchmark, mlmd_bench_report);
  if (!execuate_status.ok()) {
    LOG(WARNING) << execuate_status;
  }

  tensorflow::Status write_status =
      ml_metadata::WriteProtoResultToDisk(argv, mlmd_bench_report);
  if (!write_status.ok()) {
    LOG(WARNING) << write_status;
  }

  return 0;
}
