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

#include <vector>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"

namespace ml_metadata {
namespace {
// The cutoff threshold numbers when printing progress to consoles.
constexpr int kReportThresholds[]{1000,   5000,   10000,  50000,
                                  100000, 500000, 1000000};

// The threshold of starting console printing the thread status.
constexpr int kStartConsolePrintThreshold = 100;
}  // namespace

ThreadStats::ThreadStats()
    : accumulated_elapsed_time_(absl::ZeroDuration()),
      done_(0),
      bytes_(0),
      next_report_(kStartConsolePrintThreshold) {}

void ThreadStats::Start() { start_ = absl::Now(); }

void ThreadStats::Update(const OpStats& op_stats,
                         const int64 approx_total_done) {
  bytes_ += op_stats.transferred_bytes;
  accumulated_elapsed_time_ += op_stats.elapsed_time;
  done_++;
  // Reports the current progress with `approx_total_done`.
  if (approx_total_done < next_report_) {
    return;
  }
  absl::FPrintF(stderr, "... finished %d ops%30s\r", approx_total_done, "");
  std::fflush(stderr);
  // Update next_report_ to the next threshold.
  int threshold_index = 0;
  while (next_report_ >= kReportThresholds[threshold_index]) {
    threshold_index++;
  }
  next_report_ += kReportThresholds[threshold_index] / 10;
}

void ThreadStats::Stop() { finish_ = absl::Now(); }

void ThreadStats::Merge(const ThreadStats& other) {
  // Accumulates done_, bytes_ and accumulated_elapsed_time_ of each thread
  // stats.
  done_ += other.done();
  bytes_ += other.bytes();
  accumulated_elapsed_time_ += other.accumulated_elapsed_time();
  // Chooses the earliest start time and latest end time of each merged
  // thread stats.
  start_ = std::min(start_, other.start());
  finish_ = std::max(finish_, other.finish());
}

void ThreadStats::Report(const std::string& specification,
                         WorkloadConfigResult& workload_summary) {
  if (done_ == 0) {
    return;
  }

  const double microseconds_per_operation =
      (accumulated_elapsed_time_ / absl::Microseconds(1)) / done_;
  workload_summary.set_microseconds_per_operation(microseconds_per_operation);

  std::string maybe_byte_rate;
  // Not all workloads will transfer bytes in the process.
  if (bytes_ > 0) {
    // Rate is computed on actual elapsed time (latest end time minus
    // earliest start time of each thread) instead of the sum of per-thread
    // elapsed times.
    const double bytes_per_second =
        bytes_ / absl::ToDoubleSeconds(finish_ - start_);
    workload_summary.set_bytes_per_second(bytes_per_second);
    maybe_byte_rate = absl::StrFormat("%6.1f KB/s", bytes_per_second / 1024.0);
  }

  absl::FPrintF(stdout, "%-12s : %11.3f micros/op; %s\n", specification.c_str(),
                microseconds_per_operation, maybe_byte_rate.c_str());
  std::fflush(stdout);
}

}  // namespace ml_metadata
