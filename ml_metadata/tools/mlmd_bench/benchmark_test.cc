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
  EXPECT_STREQ(benchmark.workload(0)->GetName().c_str(), "FILL_ARTIFACT_TYPE");
  EXPECT_STREQ(benchmark.workload(1)->GetName().c_str(),
               "FILL_EXECUTION_TYPE(UPDATE)");
  EXPECT_STREQ(benchmark.workload(2)->GetName().c_str(), "FILL_CONTEXT_TYPE");
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
  EXPECT_STREQ(benchmark.workload(0)->GetName().c_str(),
               "FILL_ARTIFACT(UPDATE)");
  EXPECT_STREQ(benchmark.workload(1)->GetName().c_str(), "FILL_EXECUTION");
  EXPECT_STREQ(benchmark.workload(2)->GetName().c_str(),
               "FILL_CONTEXT(UPDATE)");
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
  EXPECT_STREQ(benchmark.workload(0)->GetName().c_str(), "FILL_ATTRIBUTION");
  EXPECT_STREQ(benchmark.workload(1)->GetName().c_str(), "FILL_ASSOCIATION");
}

// Tests the CreateWorkload() for FillEvents workload, checks that all
// FillEvents workload configurations have transformed into executable
// workloads inside benchmark.
TEST(BenchmarkTest, CreatFillEventsWorkloadTest) {
  MLMDBenchConfig mlmd_bench_config =
      testing::ParseTextProtoOrDie<MLMDBenchConfig>(
          R"(
            workload_configs: {
              fill_events_config: {
                specification: INPUT
                execution_node_popularity: { dirichlet_alpha: 1 }
                artifact_node_popularity_zipf: { skew: 0.5 }
                num_events: { minimum: 1 maximum: 10 }
              }
              num_operations: 250
            }
            workload_configs: {
              fill_events_config: {
                specification: OUTPUT
                execution_node_popularity: { dirichlet_alpha: 1 }
                artifact_node_popularity_categorical: { dirichlet_alpha: 1 }
                num_events: { minimum: 1 maximum: 10 }
              }
              num_operations: 150
            }
          )");

  Benchmark benchmark(mlmd_bench_config);
  // Checks that all workload configurations have transformed into executable
  // workloads inside benchmark.
  EXPECT_EQ(benchmark.num_workloads(),
            mlmd_bench_config.workload_configs_size());
  EXPECT_STREQ(benchmark.workload(0)->GetName().c_str(), "FILL_INPUT_EVENT");
  EXPECT_STREQ(benchmark.workload(1)->GetName().c_str(), "FILL_OUTPUT_EVENT");
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
  std::vector<std::string> workload_names{
      "READ_ALL_ARTIFACT_TYPES",     "READ_ALL_EXECUTION_TYPES",
      "READ_ALL_CONTEXT_TYPES",      "READ_ARTIFACT_TYPES_BY_IDs",
      "READ_EXECUTION_TYPES_BY_IDs", "READ_CONTEXT_TYPES_BY_IDs",
      "READ_ARTIFACT_TYPE_BY_NAME",  "READ_EXECUTION_TYPE_BY_NAME",
      "READ_CONTEXT_TYPE_BY_NAME"};

  Benchmark benchmark(mlmd_bench_config);
  // Checks that all workload configurations have transformed into executable
  // workloads inside benchmark.
  EXPECT_EQ(benchmark.num_workloads(),
            mlmd_bench_config.workload_configs_size());
  for (int i = 0; i < mlmd_bench_config.workload_configs_size(); ++i) {
    EXPECT_STREQ(benchmark.workload(i)->GetName().c_str(),
                 workload_names[i].c_str());
  }
}

// Tests the CreateWorkload() for ReadNodesByProperties workload, checks that
// all ReadNodesByProperties workload configurations have transformed into
// executable workloads inside benchmark.
TEST(BenchmarkTest, CreatReadNodesByPropertiesWorkloadTest) {
  MLMDBenchConfig mlmd_bench_config = testing::ParseTextProtoOrDie<
      MLMDBenchConfig>(
      R"(
        workload_configs: {
          read_nodes_by_properties_config: {
            specification: ARTIFACTS_BY_IDs
            maybe_num_queries: { minimum: 1 maximum: 10 }
          }
          num_operations: 120
        }
        workload_configs: {
          read_nodes_by_properties_config: {
            specification: EXECUTIONS_BY_IDs
            maybe_num_queries: { minimum: 1 maximum: 10 }
          }
          num_operations: 100
        }
        workload_configs: {
          read_nodes_by_properties_config: {
            specification: CONTEXTS_BY_IDs
            maybe_num_queries: { minimum: 1 maximum: 10 }
          }
          num_operations: 150
        }
        workload_configs: {
          read_nodes_by_properties_config: { specification: ARTIFACTS_BY_TYPE }
          num_operations: 120
        }
        workload_configs: {
          read_nodes_by_properties_config: { specification: EXECUTIONS_BY_TYPE }
          num_operations: 100
        }
        workload_configs: {
          read_nodes_by_properties_config: { specification: CONTEXTS_BY_TYPE }
          num_operations: 150
        }
        workload_configs: {
          read_nodes_by_properties_config: {
            specification: ARTIFACT_BY_TYPE_AND_NAME
          }
          num_operations: 120
        }
        workload_configs: {
          read_nodes_by_properties_config: {
            specification: EXECUTION_BY_TYPE_AND_NAME
          }
          num_operations: 100
        }
        workload_configs: {
          read_nodes_by_properties_config: {
            specification: CONTEXT_BY_TYPE_AND_NAME
          }
          num_operations: 150
        }
        workload_configs: {
          read_nodes_by_properties_config: {
            specification: ARTIFACTS_BY_URIs
            maybe_num_queries: { minimum: 1 maximum: 10 }
          }
          num_operations: 150
        }
      )");
  std::vector<std::string> workload_names{
      "READ_ARTIFACTS_BY_IDs",          "READ_EXECUTIONS_BY_IDs",
      "READ_CONTEXTS_BY_IDs",           "READ_ARTIFACTS_BY_TYPE",
      "READ_EXECUTIONS_BY_TYPE",        "READ_CONTEXTS_BY_TYPE",
      "READ_ARTIFACT_BY_TYPE_AND_NAME", "READ_EXECUTION_BY_TYPE_AND_NAME",
      "READ_CONTEXT_BY_TYPE_AND_NAME",  "READ_ARTIFACTS_BY_URIs"};

  Benchmark benchmark(mlmd_bench_config);
  // Checks that all workload configurations have transformed into executable
  // workloads inside benchmark.
  EXPECT_EQ(benchmark.num_workloads(),
            mlmd_bench_config.workload_configs_size());
  for (int i = 0; i < mlmd_bench_config.workload_configs_size(); ++i) {
    EXPECT_STREQ(benchmark.workload(i)->GetName().c_str(),
                 workload_names[i].c_str());
  }
}

// Tests the CreateWorkload() for ReadNodesViaContextEdges workload, checks that
// all ReadNodesViaContextEdges workload configurations have transformed into
// executable workloads inside benchmark.
TEST(BenchmarkTest, CreateReadNodesViaContextEdgesWorkloadTest) {
  MLMDBenchConfig mlmd_bench_config =
      testing::ParseTextProtoOrDie<MLMDBenchConfig>(
          R"(
            workload_configs: {
              read_nodes_via_context_edges_config: {
                specification: CONTEXTS_BY_ARTIFACT
              }
              num_operations: 120
            }
            workload_configs: {
              read_nodes_via_context_edges_config: {
                specification: CONTEXTS_BY_EXECUTION
              }
              num_operations: 100
            }
            workload_configs: {
              read_nodes_via_context_edges_config: {
                specification: ARTIFACTS_BY_CONTEXT
              }
              num_operations: 150
            }
            workload_configs: {
              read_nodes_via_context_edges_config: {
                specification: EXECUTIONS_BY_CONTEXT
              }
              num_operations: 120
            }
          )");
  std::vector<std::string> workload_names{
      "READ_CONTEXTS_BY_ARTIFACT", "READ_CONTEXTS_BY_EXECUTION",
      "READ_ARTIFACTS_BY_CONTEXT", "READ_EXECUTIONS_BY_CONTEXT"};

  Benchmark benchmark(mlmd_bench_config);
  // Checks that all workload configurations have transformed into executable
  // workloads inside benchmark.
  EXPECT_EQ(benchmark.num_workloads(),
            mlmd_bench_config.workload_configs_size());
  for (int i = 0; i < mlmd_bench_config.workload_configs_size(); ++i) {
    EXPECT_STREQ(benchmark.workload(i)->GetName().c_str(),
                 workload_names[i].c_str());
  }
}

// Tests the CreateWorkload() for ReadEvents workload, checks that all
// ReadEvents workload configurations have transformed into executable
// workloads inside benchmark.
TEST(BenchmarkTest, CreatReadEventsWorkloadTest) {
  MLMDBenchConfig mlmd_bench_config =
      testing::ParseTextProtoOrDie<MLMDBenchConfig>(
          R"(
            workload_configs: {
              read_events_config: {
                specification: EVENTS_BY_ARTIFACT_IDS
                num_ids: { minimum: 1 maximum: 10 }
              }
              num_operations: 250
            }
            workload_configs: {
              read_events_config: {
                specification: EVENTS_BY_EXECUTION_IDS
                num_ids: { minimum: 1 maximum: 10 }
              }
              num_operations: 150
            }
          )");

  Benchmark benchmark(mlmd_bench_config);
  // Checks that all workload configurations have transformed into executable
  // workloads inside benchmark.
  EXPECT_EQ(benchmark.num_workloads(),
            mlmd_bench_config.workload_configs_size());
  EXPECT_STREQ(benchmark.workload(0)->GetName().c_str(),
               "READ_EVENTS_BY_ARTIFACT_IDS");
  EXPECT_STREQ(benchmark.workload(1)->GetName().c_str(),
               "READ_EVENTS_BY_EXECUTION_IDS");
}

}  // namespace
}  // namespace ml_metadata
