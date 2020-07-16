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
TEST(StatsTest, UpdateTest) {
  srand(time(NULL));

  Stats stats;
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

  // Since the `done_`, `bytes_` and `micro_seconds_` are accumulated by each
  // update, the finial `done_`, `bytes_` and `micro_seconds_` of `stats` should
  // be the sum of the list of `op_stats`.
  EXPECT_EQ(stats.done(), 10000);
  EXPECT_EQ(stats.micro_seconds(),
            std::accumulate(op_stats_time.begin(), op_stats_time.end(),
                            absl::Microseconds(0)) /
                absl::Microseconds(1));
  EXPECT_EQ(stats.bytes(),
            std::accumulate(op_stats_bytes.begin(), op_stats_bytes.end(), 0));
}

// Tests the Merge() of Stats class.
TEST(StatsTest, MergeTest) {
  srand(time(NULL));
  const absl::Duration sleep_time = absl::Milliseconds(10);

  Stats stats1;
  stats1.Start();
  absl::SleepFor(sleep_time);
  Stats stats2;
  stats2.Start();
  absl::SleepFor(sleep_time);
  Stats stats3;
  stats3.Start();
  // Since `stats1` starts the earliest among the three.
  absl::Time ealiest_start_time = stats1.start();

  std::vector<absl::Duration> op_stats_time;
  std::vector<int> op_stats_bytes;
  for (int i = 0; i < 10000; ++i) {
    op_stats_time.push_back(absl::Microseconds(rand() % 99999));
    op_stats_bytes.push_back(rand() % 99999);
  }

  // Updates the three stats with the prepared list of `op_stats`.
  for (int64 i = 0; i < 10000; ++i) {
    OpStats curr_op_stats{op_stats_time[i], op_stats_bytes[i]};
    if (i <= 3333) {
      stats1.Update(curr_op_stats, i);
    } else if (i <= 6666) {
      stats2.Update(curr_op_stats, i);
    } else {
      stats3.Update(curr_op_stats, i);
    }
  }

  stats1.Stop();
  absl::SleepFor(sleep_time);
  stats2.Stop();
  absl::SleepFor(sleep_time);
  stats3.Stop();
  absl::SleepFor(sleep_time);
  // Since the `stats3` stops the latest among the three.
  absl::Time latest_end_time = stats3.finish();

  stats1.Merge(stats2);
  stats1.Merge(stats3);

  // Since the Merge() accumulates the `done_`, `bytes_` and `micro_seconds_` of
  // each merged stats, the final stats's `done_`, `bytes_` and `micro_seconds_`
  // should be the sum of the three stats.
  EXPECT_EQ(stats1.done(), 10000);
  EXPECT_EQ(stats1.micro_seconds(),
            std::accumulate(op_stats_time.begin(), op_stats_time.end(),
                            absl::Microseconds(0)) /
                absl::Microseconds(1));
  EXPECT_EQ(stats1.bytes(),
            std::accumulate(op_stats_bytes.begin(), op_stats_bytes.end(), 0));

  // In Merge(), we takes the earliest start time and latest end time as
  // the start and end time of the merged stats.
  EXPECT_EQ(stats1.start(), ealiest_start_time);
  EXPECT_EQ(stats1.finish(), latest_end_time);
}

}  // namespace
}  // namespace ml_metadata
