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

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_metadata/metadata_store/types.h"

namespace ml_metadata {

Stats::Stats() : micro_seconds_(0), done_(0), next_report_(100), bytes_(0) {}

void Stats::Start() { start_ = absl::Now(); }

void Stats::Update(const OpStats& op_stats, int64& total_done) {
  bytes_ += op_stats.transferred_bytes;
  micro_seconds_ += op_stats.elapsed_time / absl::Microseconds(1);
  done_++;
  // Reports the current progress.
  if (total_done >= next_report_) {
    if (next_report_ < 1000) {
      next_report_ += 100;
    } else if (next_report_ < 5000) {
      next_report_ += 500;
    } else if (next_report_ < 10000) {
      next_report_ += 1000;
    } else if (next_report_ < 50000) {
      next_report_ += 5000;
    } else if (next_report_ < 100000) {
      next_report_ += 10000;
    } else if (next_report_ < 500000) {
      next_report_ += 50000;
    } else {
      next_report_ += 100000;
    }
    std::fprintf(stderr, "... finished %lld ops%30s\r", total_done, "");
    std::fflush(stderr);
  }
}

void Stats::Stop() { finish_ = absl::Now(); }

void Stats::Merge(const Stats& other) {
  // Accumulates done_, bytes_ and micro_seconds_ of each thread stats.
  done_ += other.done_;
  bytes_ += other.bytes_;
  micro_seconds_ += other.micro_seconds_;
  // Chooses the earliest start time and latest end time of each merged
  // thread stats.
  if (other.start_ < start_) start_ = other.start_;
  if (other.finish_ > finish_) finish_ = other.finish_;
}

void Stats::Report(const std::string& specification) {
  std::string extra;
  if (bytes_ > 0) {
    // Rate is computed on actual elapsed time (latest end time minus
    // earliest start time of each thread) instead of the sum of per-thread
    // elapsed times.
    double elapsed_seconds = micro_seconds_ * 1e-6;
    char rate[100];
    std::snprintf(rate, sizeof(rate), "%6.1f KB/s",
                  (bytes_ / 1024.0) / elapsed_seconds);
    extra = rate;
  }

  std::fprintf(stdout, "%-12s : %11.3f micros/op;%s%s\n", specification.c_str(),
               micro_seconds_ / done_, (extra.empty() ? "" : " "),
               extra.c_str());
  std::fflush(stdout);
}

absl::Time Stats::start() { return start_; }

absl::Time Stats::finish() { return finish_; }

double Stats::micro_seconds() { return micro_seconds_; }

int64 Stats::done() { return done_; }

int64 Stats::bytes() { return bytes_; }

}  // namespace ml_metadata
