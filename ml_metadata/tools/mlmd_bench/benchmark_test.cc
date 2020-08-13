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

// Tests the CreateWorkload() for FillTypes workload, checks that all FillTypes
// workload configurations have transformed into executable workloads inside
// benchmark.
TEST(BenchmarkTest, CreatFillTypesWorkloadTest) {
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
  EXPECT_EQ(benchmark.num_workloads(),
            mlmd_bench_config.workload_configs_size());
  EXPECT_STREQ("FILL_ARTIFACT_TYPE", benchmark.workload(0)->GetName().c_str());
  EXPECT_STREQ("FILL_EXECUTION_TYPE(UPDATE)",
               benchmark.workload(1)->GetName().c_str());
  EXPECT_STREQ("FILL_CONTEXT_TYPE", benchmark.workload(2)->GetName().c_str());
}

// Tests the CreateWorkload() for FillNodes workload, checks that all FillNodes
// workload configurations have transformed into executable workloads inside
// benchmark.
TEST(BenchmarkTest, CreatFillNodesWorkloadTest) {
  MLMDBenchConfig mlmd_bench_config =
      testing::ParseTextProtoOrDie<MLMDBenchConfig>(
          R"(
            workload_configs: {
              fill_nodes_config: {
                update: true
                specification: ARTIFACT
                num_properties: { minimum: 1 maximum: 10 }
                string_value_bytes: { minimum: 1 maximum: 10 }
                num_nodes: { minimum: 1 maximum: 10 }
              }
              num_operations: 200
            }
            workload_configs: {
              fill_nodes_config: {
                update: false
                specification: EXECUTION
                num_properties: { minimum: 1 maximum: 10 }
                string_value_bytes: { minimum: 1 maximum: 10 }
                num_nodes: { minimum: 1 maximum: 10 }
              }
              num_operations: 250
            }
            workload_configs: {
              fill_nodes_config: {
                update: true
                specification: CONTEXT
                num_properties: { minimum: 1 maximum: 10 }
                string_value_bytes: { minimum: 1 maximum: 10 }
                num_nodes: { minimum: 1 maximum: 10 }
              }
              num_operations: 150
            }
          )");
  Benchmark benchmark(mlmd_bench_config);
  EXPECT_EQ(benchmark.num_workloads(),
            mlmd_bench_config.workload_configs_size());
  EXPECT_STREQ("FILL_ARTIFACT(UPDATE)",
               benchmark.workload(0)->GetName().c_str());
  EXPECT_STREQ("FILL_EXECUTION", benchmark.workload(1)->GetName().c_str());
  EXPECT_STREQ("FILL_CONTEXT(UPDATE)",
               benchmark.workload(2)->GetName().c_str());
}

// Tests the CreateWorkload() for FillContextEdges workload, checks that all
// FillContextEdges workload configurations have transformed into executable
// workloads inside benchmark.
TEST(BenchmarkTest, CreatFillContextEdgesWorkloadTest) {
  MLMDBenchConfig mlmd_bench_config =
      testing::ParseTextProtoOrDie<MLMDBenchConfig>(
          R"(
            workload_configs: {
              fill_context_edges_config: {
                specification: ATTRIBUTION
                non_context_node_popularity: { dirichlet_alpha: 1 }
                context_node_popularity: { dirichlet_alpha: 1 }
                num_edges: { minimum: 1 maximum: 10 }
              }
              num_operations: 250
            }
            workload_configs: {
              fill_context_edges_config: {
                specification: ASSOCIATION
                non_context_node_popularity: { dirichlet_alpha: 1 }
                context_node_popularity: { dirichlet_alpha: 1 }
                num_edges: { minimum: 1 maximum: 10 }
              }
              num_operations: 150
            }
          )");

  Benchmark benchmark(mlmd_bench_config);
  // Checks that all workload configurations have transformed into executable
  // workloads inside benchmark.
  EXPECT_EQ(benchmark.num_workloads(),
            mlmd_bench_config.workload_configs_size());
  EXPECT_STREQ("FILL_ATTRIBUTION", benchmark.workload(0)->GetName().c_str());
  EXPECT_STREQ("FILL_ASSOCIATION", benchmark.workload(1)->GetName().c_str());
}

// Tests the CreateWorkload() for ReadTypes workload, checks that all
// ReadTypes workload configurations have transformed into executable
// workloads inside benchmark.
TEST(BenchmarkTest, CreatReadTypesWorkloadTest) {
  MLMDBenchConfig mlmd_bench_config =
      testing::ParseTextProtoOrDie<MLMDBenchConfig>(
          R"(
            workload_configs: {
              read_types_config: { specification: ALL_ARTIFACT_TYPES }
              num_operations: 120
            }
            workload_configs: {
              read_types_config: { specification: ALL_EXECUTION_TYPES }
              num_operations: 100
            }
            workload_configs: {
              read_types_config: { specification: ALL_CONTEXT_TYPES }
              num_operations: 150
            }
            workload_configs: {
              read_types_config: {
                specification: ARTIFACT_TYPES_BY_IDs
                maybe_num_ids: { minimum: 1 maximum: 10 }
              }
              num_operations: 120
            }
            workload_configs: {
              read_types_config: {
                specification: EXECUTION_TYPES_BY_IDs
                maybe_num_ids: { minimum: 1 maximum: 10 }
              }
              num_operations: 100
            }
            workload_configs: {
              read_types_config: {
                specification: CONTEXT_TYPES_BY_IDs
                maybe_num_ids: { minimum: 1 maximum: 10 }
              }
              num_operations: 150
            }
            workload_configs: {
              read_types_config: { specification: ARTIFACT_TYPE_BY_NAME }
              num_operations: 120
            }
            workload_configs: {
              read_types_config: { specification: EXECUTION_TYPE_BY_NAME }
              num_operations: 100
            }
            workload_configs: {
              read_types_config: { specification: CONTEXT_TYPE_BY_NAME }
              num_operations: 150
            }
          )");

  Benchmark benchmark(mlmd_bench_config);
  // Checks that all workload configurations have transformed into executable
  // workloads inside benchmark.
  EXPECT_EQ(benchmark.num_workloads(),
            mlmd_bench_config.workload_configs_size());
  EXPECT_STREQ("READ_ALL_ARTIFACT_TYPES",
               benchmark.workload(0)->GetName().c_str());
  EXPECT_STREQ("READ_ALL_EXECUTION_TYPES",
               benchmark.workload(1)->GetName().c_str());
  EXPECT_STREQ("READ_ALL_CONTEXT_TYPES",
               benchmark.workload(2)->GetName().c_str());
  EXPECT_STREQ("READ_ARTIFACT_TYPES_BY_IDs",
               benchmark.workload(3)->GetName().c_str());
  EXPECT_STREQ("READ_EXECUTION_TYPES_BY_IDs",
               benchmark.workload(4)->GetName().c_str());
  EXPECT_STREQ("READ_CONTEXT_TYPES_BY_IDs",
               benchmark.workload(5)->GetName().c_str());
  EXPECT_STREQ("READ_ARTIFACT_TYPE_BY_NAME",
               benchmark.workload(6)->GetName().c_str());
  EXPECT_STREQ("READ_EXECUTION_TYPE_BY_NAME",
               benchmark.workload(7)->GetName().c_str());
  EXPECT_STREQ("READ_CONTEXT_TYPE_BY_NAME",
               benchmark.workload(8)->GetName().c_str());
}

}  // namespace
}  // namespace ml_metadata
