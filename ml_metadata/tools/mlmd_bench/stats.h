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

// ThreadStats records the statics(start time, end time, elapsed time, total
// operations done, transferred bytes) of each thread. It will be updated by
// Opstats. Every ThreadStats of a particular workload will be merged together
// after each thread has finished execution to generate a workload stats for
// reporting the performance of current workload.
class ThreadStats {
 public:
  ThreadStats();
  ~ThreadStats() = default;

  // Starts the current thread stats and initializes the member variables.
  void Start();

  // Updates the current thread stats with op_stats.
  void Update(const OpStats& op_stats, int64 approx_total_done);

  // Records the end time for each thread after the current thread has finished
  // all the operations.
  void Stop();

  // Merges the thread stats instances into a workload stats that will be used
  // for report purpose.
  void Merge(const ThreadStats& other);

  // Reports the metrics of interests: microsecond per operation and total bytes
  // per seconds for the current workload.
  void Report(const std::string& specification);

  // Gets the start time of current thread stats.
  absl::Time start() const { return start_; }

  // Gets the finish time of current thread stats.
  absl::Time finish() const { return finish_; }

  // Gets the accumulated elapsed time of current thread stats.
  absl::Duration accumulated_elapsed_time() const {
    return accumulated_elapsed_time_;
  }

  // Gets the number of total finished operations of current thread stats.
  int64 done() const { return done_; }

  // Gets the total transferred bytes of current thread stats.
  int64 bytes() const { return bytes_; }

 private:
  // Records the start time of current thread stats.
  absl::Time start_;
  // Records the finish time of current thread stats.
  absl::Time finish_;
  // Records the accumulated elapsed time of current thread stats.
  absl::Duration accumulated_elapsed_time_;
  // Records the number of total finished operations of current thread stats.
  int64 done_;
  // Records the total transferred bytes of current thread stats.
  int64 bytes_;
  // Uses in Report() for console reporting.
  int64 next_report_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_STATS_H
