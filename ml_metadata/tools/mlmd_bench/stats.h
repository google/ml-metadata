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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_STATS_H
#define ML_METADATA_TOOLS_MLMD_BENCH_STATS_H

#include "absl/time/time.h"
#include "ml_metadata/metadata_store/types.h"

namespace ml_metadata {

// OpStats records the statics(elapsed time, transferred bytes) of each
// operation. It will be used to update the thread stats.
struct OpStats {
  absl::Duration elapsed_time;
  int64 transferred_bytes;
};

class Stats {
 public:
  Stats();
  ~Stats() = default;

  // Starts the current thread stats and initializes the member variables.
  void Start();

  // Updates the current thread stats with op_stats.
  void Update(const OpStats& op_stats, int64& total_done);

  // Records the end time for each thread after the current thread has finished
  // all the operations.
  void Stop();

  // Merges the thread stats instances into a workload stats that will be used
  // for report purpose.
  void Merge(const Stats& other);

  // Reports the metrics of interests: microsecond per operation and total bytes
  // per seconds for the current workload.
  void Report(const std::string& specification);

  absl::Time start();

  absl::Time finish();

  double micro_seconds();

  int64 done();

  int64 bytes();

 private:
  absl::Time start_;
  absl::Time finish_;
  double micro_seconds_;
  int64 done_;
  int64 next_report_;
  int64 bytes_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_STATS_H
