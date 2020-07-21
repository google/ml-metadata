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
#include "ml_metadata/tools/mlmd_bench/stats.h"

#include <numeric>

#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace ml_metadata {
namespace {

// Tests the Update() of Stats class.
TEST(ThreadStatsTest, UpdateTest) {
  srand(time(NULL));

  ThreadStats stats;
  stats.Start();

  // Prepares the list of `op_stats` to update `stats`.
  std::vector<absl::Duration> op_stats_time;
  std::vector<int> op_stats_bytes;
  for (int i = 0; i < 10000; ++i) {
    op_stats_time.push_back(absl::Microseconds(rand() % 99999));
    op_stats_bytes.push_back(rand() % 99999);
  }

  // Updates `stats` with the list of `op_stats`.
  for (int64 i = 0; i < 10000; ++i) {
    OpStats curr_op_stats{op_stats_time[i], op_stats_bytes[i]};
    stats.Update(curr_op_stats, i);
  }

  // Since the `done_`, `bytes_` and `accumulated_elapsed_time_` are accumulated
  // by each update, the finial `done_`, `bytes_` and
  // `accumulated_elapsed_time_` of `stats` should be the sum of the list of
  // `op_stats`.
  EXPECT_EQ(stats.done(), 10000);
  EXPECT_EQ(stats.accumulated_elapsed_time(),
            std::accumulate(op_stats_time.begin(), op_stats_time.end(),
                            absl::Microseconds(0)));
  EXPECT_EQ(stats.bytes(),
            std::accumulate(op_stats_bytes.begin(), op_stats_bytes.end(), 0));
}

// Tests the Merge() of Stats class.
TEST(ThreadStatsTest, MergeTest) {
  srand(time(NULL));

  ThreadStats stats1;
  stats1.Start();
  ThreadStats stats2;
  stats2.Start();
  absl::Time ealiest_start_time = std::min(stats1.start(), stats2.start());

  std::vector<absl::Duration> op_stats_time;
  std::vector<int> op_stats_bytes;
  for (int i = 0; i < 10000; ++i) {
    op_stats_time.push_back(absl::Microseconds(rand() % 99999));
    op_stats_bytes.push_back(rand() % 99999);
  }

  // Updates the stats with the prepared list of `op_stats`.
  for (int64 i = 0; i < 10000; ++i) {
    OpStats curr_op_stats{op_stats_time[i], op_stats_bytes[i]};
    if (i <= 4999) {
      stats1.Update(curr_op_stats, i);
    } else {
      stats2.Update(curr_op_stats, i);
    }
  }

  stats1.Stop();
  stats2.Stop();
  absl::Time latest_end_time = std::max(stats1.finish(), stats2.finish());

  stats1.Merge(stats2);

  // Since the Merge() accumulates the `done_`, `bytes_` and
  // `accumulated_elapsed_time_` of each merged stats, the final stats's
  // `done_`, `bytes_` and `accumulated_elapsed_time_` should be the sum of the
  // stats.
  EXPECT_EQ(stats1.done(), 10000);
  EXPECT_EQ(stats1.accumulated_elapsed_time(),
            std::accumulate(op_stats_time.begin(), op_stats_time.end(),
                            absl::Microseconds(0)));
  EXPECT_EQ(stats1.bytes(),
            std::accumulate(op_stats_bytes.begin(), op_stats_bytes.end(), 0));

  // In Merge(), we takes the earliest start time and latest end time as
  // the start and end time of the merged stats.
  EXPECT_EQ(stats1.start(), ealiest_start_time);
  EXPECT_EQ(stats1.finish(), latest_end_time);
}

}  // namespace
}  // namespace ml_metadata
