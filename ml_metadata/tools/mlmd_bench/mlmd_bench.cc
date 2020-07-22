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
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace ml_metadata {
namespace {

// Initializes the mlmd_bench_config with the specified configuration pbtxt
// file.
void InitMLMDBenchConfigFromPbtxtFile(char** argv,
                                      MLMDBenchConfig& mlmd_bench_config,
                                      MLMDBenchReport& mlmd_bench_report) {
  int file_descriptor = open(argv[1], O_RDONLY);
  if (file_descriptor < 0) {
    LOG(ERROR) << "Cannot open the config. file: " << argv[1] << " !";
  }
  google::protobuf::io::FileInputStream file_input(file_descriptor);
  if (!google::protobuf::TextFormat::Parse(&file_input, &mlmd_bench_config)) {
    LOG(ERROR) << "Fail to parse the config. file: " << argv[1] << " !";
  }
  // Initializes `mlmd_bench_report` with the updated `mlmd_bench_config`.
  for (auto& workload_config : mlmd_bench_config.workload_configs()) {
    mlmd_bench_report.add_summaries()->mutable_workload_config()->CopyFrom(
        workload_config);
  }
}

void WriteProtoResultToDisk(char** argv, MLMDBenchReport& mlmd_bench_report) {
  std::fstream output(argv[2], std::ios::binary | std::ios::out);

  if (!mlmd_bench_report.SerializeToOstream(&output)) {
    LOG(ERROR) << "Failed to write the performance result!";
  }
  output.close();
}

}  // namespace
}  // namespace ml_metadata

int main(int argc, char** argv) {
  // Configurations for mlmd_bench.
  ml_metadata::MLMDBenchConfig mlmd_bench_config;
  // Performance report for workloads specified inside `mlmd_bench_config`.
  ml_metadata::MLMDBenchReport mlmd_bench_report;
  ml_metadata::InitMLMDBenchConfigFromPbtxtFile(argv, mlmd_bench_config,
                                                mlmd_bench_report);

  // Feeds the config. into the benchmark for generating executable workloads.
  ml_metadata::Benchmark benchmark(mlmd_bench_config);
  // Executes the workloads inside the benchmark with the thread runner.
  ml_metadata::ThreadRunner runner(
      mlmd_bench_config.mlmd_config(),
      mlmd_bench_config.thread_env_config().num_threads());
  tensorflow::Status status = runner.Run(benchmark, mlmd_bench_report);
  if (!status.ok()) {
    LOG(WARNING) << status;
  }

  ml_metadata::WriteProtoResultToDisk(argv, mlmd_bench_report);

  return 0;
}
