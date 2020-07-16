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
#include "ml_metadata/tools/mlmd_bench/thread_runner.h"

#include <gtest/gtest.h>
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/benchmark.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

// Tests the Run() of ThreadRunner class in single-thread mode.
TEST(ThreadRunnerTest, RunInSingleThreadTest) {
  MLMDBenchConfig mlmd_bench_config =
      testing::ParseTextProtoOrDie<MLMDBenchConfig>(
          R"(
            mlmd_config: {
              sqlite: {
                filename_uri: "mlmd-bench-test1.db"
                connection_mode: READWRITE_OPENCREATE
              }
            }
            workload_configs: {
              fill_types_config: {
                update: false
                specification: ARTIFACT_TYPE
                num_properties: { minimum: 1 maximum: 10 }
              }
              num_operations: 100
            }
            thread_env_config: { num_threads: 1 }
          )");

  Benchmark benchmark(mlmd_bench_config);
  ThreadRunner runner(mlmd_bench_config);
  TF_ASSERT_OK(runner.Run(benchmark));

  std::unique_ptr<MetadataStore> store;
  TF_ASSERT_OK(CreateMetadataStore(mlmd_bench_config.mlmd_config(), &store));

  GetArtifactTypesResponse get_response;
  TF_ASSERT_OK(store->GetArtifactTypes(/*request=*/{}, &get_response));
  // Checks that the workload indeed be executed by the thread_runner.
  EXPECT_EQ(get_response.artifact_types_size(),
            mlmd_bench_config.workload_configs()[0].num_operations());
}

// Tests the Run() of ThreadRunner class in multi-thread mode.
TEST(ThreadRunnerTest, RunInMultiThreadTest) {
  MLMDBenchConfig mlmd_bench_config =
      testing::ParseTextProtoOrDie<MLMDBenchConfig>(
          R"(
            mlmd_config: {
              sqlite: {
                filename_uri: "mlmd-bench-test2.db"
                connection_mode: READWRITE_OPENCREATE
              }
            }
            workload_configs: {
              fill_types_config: {
                update: false
                specification: ARTIFACT_TYPE
                num_properties: { minimum: 1 maximum: 10 }
              }
              num_operations: 100
            }
            thread_env_config: { num_threads: 10 }
          )");

  Benchmark benchmark(mlmd_bench_config);
  ThreadRunner runner(mlmd_bench_config);
  TF_ASSERT_OK(runner.Run(benchmark));

  std::unique_ptr<MetadataStore> store;
  TF_ASSERT_OK(CreateMetadataStore(mlmd_bench_config.mlmd_config(), &store));

  GetArtifactTypesResponse get_response;
  TF_ASSERT_OK(store->GetArtifactTypes(/*request=*/{}, &get_response));
  // Checks that the workload indeed be executed by the thread_runner.
  EXPECT_EQ(get_response.artifact_types_size(),
            mlmd_bench_config.workload_configs()[0].num_operations());
}

}  // namespace
}  // namespace ml_metadata
