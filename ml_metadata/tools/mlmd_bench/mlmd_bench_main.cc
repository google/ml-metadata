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
#include <fstream>
#include <iostream>

#include "gflags/gflags.h"
#include "ml_metadata/tools/mlmd_bench/benchmark.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/thread_runner.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

namespace ml_metadata {
namespace {

// Initializes the `mlmd_bench_config` with the specified configuration .pb or
// .pbtxt file. Returns detailed error if the execution failed.
tensorflow::Status InitAndValidateMLMDBenchConfig(
    const std::string& config_file_path, MLMDBenchConfig& mlmd_bench_config) {
  if (!tensorflow::Env::Default()->FileExists(config_file_path).ok()) {
    return tensorflow::errors::NotFound(
        "Could not find mlmd_bench configuration .pb or "
        ".pbtxt file at supplied configuration file path: ",
        config_file_path);
  }
  return ReadTextProto(tensorflow::Env::Default(), config_file_path,
                       &mlmd_bench_config);
}

// Writes the performance report into a specified file in the output path.
// Returns detailed error if the execution failed.
tensorflow::Status WriteProtoResultToDisk(
    const std::string& output_report_path,
    const MLMDBenchReport& mlmd_bench_report) {
  return tensorflow::WriteTextProto(tensorflow::Env::Default(),
                                    output_report_path, mlmd_bench_report);
}

}  // namespace
}  // namespace ml_metadata

// mlmd_bench input and output file path command line options.
DEFINE_string(config_file_path, "",
              "Input mlmd_bench configuration .pb or .pbtxt file path.");
DEFINE_string(output_report_path, "./mlmd_bench_report.pb.txt",
              "Output mlmd_bench performance report file path.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // Configurations for `mlmd_bench`.
  ml_metadata::MLMDBenchConfig mlmd_bench_config;
  TF_CHECK_OK(ml_metadata::InitAndValidateMLMDBenchConfig(
      FLAGS_config_file_path, mlmd_bench_config));
  // Feeds the `mlmd_bench_config` into the benchmark for generating executable
  // workloads.
  ml_metadata::Benchmark benchmark(mlmd_bench_config);
  // Executes the workloads inside the benchmark with the thread runner.
  ml_metadata::ThreadRunner runner(
      mlmd_bench_config.mlmd_config(),
      mlmd_bench_config.thread_env_config().num_threads());
  TF_CHECK_OK(runner.Run(benchmark));

  TF_CHECK_OK(ml_metadata::WriteProtoResultToDisk(
      FLAGS_output_report_path, benchmark.mlmd_bench_report()));
  return 0;
}
