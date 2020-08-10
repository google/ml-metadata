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

#include <gtest/gtest.h>
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"

namespace ml_metadata {
namespace {

// Tests the CreateWorkload() of Benchmark class.
TEST(BenchmarkTest, CreatWorkloadTest) {
  MLMDBenchConfig mlmd_bench_config =
      testing::ParseTextProtoOrDie<MLMDBenchConfig>(
          R"(
            workload_configs: {
              fill_types_config: {
                update: false
                specification: ARTIFACT_TYPE
                num_properties: { minimum: 1 maximum: 10 }
              }
              num_operations: 100
            }
            workload_configs: {
              fill_types_config: {
                update: true
                specification: EXECUTION_TYPE
                num_properties: { minimum: 1 maximum: 10 }
              }
              num_operations: 500
            }
            workload_configs: {
              fill_types_config: {
                update: false
                specification: CONTEXT_TYPE
                num_properties: { minimum: 1 maximum: 10 }
              }
              num_operations: 300
            }
          )");
  Benchmark benchmark(mlmd_bench_config);
  // Checks that all workload configurations have transformed into executable
  // workloads inside benchmark.
  EXPECT_EQ(benchmark.num_workloads(),
            mlmd_bench_config.workload_configs_size());
}

}  // namespace
}  // namespace ml_metadata
