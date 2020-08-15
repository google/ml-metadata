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

#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/tools/mlmd_bench/fill_context_edges_workload.h"
#include "ml_metadata/tools/mlmd_bench/fill_events_workload.h"
#include "ml_metadata/tools/mlmd_bench/fill_nodes_workload.h"
#include "ml_metadata/tools/mlmd_bench/fill_types_workload.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/read_nodes_by_properties_workload.h"
#include "ml_metadata/tools/mlmd_bench/read_nodes_via_context_edges_workload.h"
#include "ml_metadata/tools/mlmd_bench/read_types_workload.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"

namespace ml_metadata {
namespace {

// Creates the executable workload given `workload_config`.
std::unique_ptr<WorkloadBase> CreateWorkload(
    const WorkloadConfig& workload_config) {
  switch (workload_config.workload_config_case()) {
    case WorkloadConfig::kFillTypesConfig: {
      return absl::make_unique<FillTypes>(
          FillTypes(workload_config.fill_types_config(),
                    workload_config.num_operations()));
    }
    case WorkloadConfig::kFillNodesConfig: {
      return absl::make_unique<FillNodes>(
          FillNodes(workload_config.fill_nodes_config(),
                    workload_config.num_operations()));
    }
    case WorkloadConfig::kFillContextEdgesConfig: {
      return absl::make_unique<FillContextEdges>(
          FillContextEdges(workload_config.fill_context_edges_config(),
                           workload_config.num_operations()));
    }
    case WorkloadConfig::kFillEventsConfig: {
      return absl::make_unique<FillEvents>(
          FillEvents(workload_config.fill_events_config(),
                     workload_config.num_operations()));
    }
    case WorkloadConfig::kReadTypesConfig: {
      return absl::make_unique<ReadTypes>(
          ReadTypes(workload_config.read_types_config(),
                    workload_config.num_operations()));
    }
    case WorkloadConfig::kReadNodesByPropertiesConfig: {
      return absl::make_unique<ReadNodesByProperties>(ReadNodesByProperties(
          workload_config.read_nodes_by_properties_config(),
          workload_config.num_operations()));
    }
    case WorkloadConfig::kReadNodesViaContextEdgesConfig: {
      return absl::make_unique<ReadNodesViaContextEdges>(
          ReadNodesViaContextEdges(
              workload_config.read_nodes_via_context_edges_config(),
              workload_config.num_operations()));
    }
    default:
      LOG(FATAL) << "Cannot find corresponding workload!";
  }
}

// Initializes `mlmd_bench_report` with `mlmd_bench_config`.
void InitMLMDBenchReport(const MLMDBenchConfig& mlmd_bench_config,
                         MLMDBenchReport& mlmd_bench_report) {
  for (auto& workload_config : mlmd_bench_config.workload_configs()) {
    mlmd_bench_report.add_summaries()->mutable_workload_config()->CopyFrom(
        workload_config);
  }
}

}  // namespace

Benchmark::Benchmark(const MLMDBenchConfig& mlmd_bench_config) {
  workloads_.resize(mlmd_bench_config.workload_configs_size());
  // For each `workload_config`, create corresponding workload.
  for (int i = 0; i < mlmd_bench_config.workload_configs_size(); ++i) {
    workloads_[i] = CreateWorkload(mlmd_bench_config.workload_configs(i));
  }
  // Initializes the performance report with given `mlmd_bench_config`.
  InitMLMDBenchReport(mlmd_bench_config, mlmd_bench_report_);
}

WorkloadBase* Benchmark::workload(const int64 workload_index) {
  return workloads_[workload_index].get();
}

}  // namespace ml_metadata
