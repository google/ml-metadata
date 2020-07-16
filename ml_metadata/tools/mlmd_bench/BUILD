# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//ml_metadata:ml_metadata.bzl",
    "ml_metadata_cc_test",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])

cc_library(
    name = "fill_types_workload",
    srcs = ["fill_types_workload.cc"],
    hdrs = ["fill_types_workload.h"],
    deps = [
        ":workload",
        ":util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:variant",
        "//ml_metadata/metadata_store",
        "//ml_metadata/metadata_store:types",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "//ml_metadata/tools/mlmd_bench/proto:mlmd_bench_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

ml_metadata_cc_test(
    name = "fill_types_workload_test",
    size = "small",
    srcs = ["fill_types_workload_test.cc"],
    deps = [
        ":fill_types_workload",
        ":util",
        "@com_google_absl//absl/memory",
        #"@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/metadata_store",
        "//ml_metadata/metadata_store:metadata_store_factory",
        "//ml_metadata/metadata_store:test_util",
        "//ml_metadata/metadata_store:types",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "//ml_metadata/tools/mlmd_bench/proto:mlmd_bench_proto",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "stats",
    srcs = ["stats.cc"],
    hdrs = ["stats.h"],
    deps = [
        "@com_google_absl//absl/time",
        "//ml_metadata/metadata_store:types",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

ml_metadata_cc_test(
    name = "stats_test",
    size = "small",
    srcs = ["stats_test.cc"],
    deps = [
        ":stats",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    hdrs = ["benchmark.h"],
    deps = [
        ":fill_types_workload",
        ":workload",
        "//ml_metadata/metadata_store:types",
        "//ml_metadata/tools/mlmd_bench/proto:mlmd_bench_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

ml_metadata_cc_test(
    name = "benchmark_test",
    size = "small",
    srcs = ["benchmark_test.cc"],
    deps = [
        "benchmark",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/metadata_store:test_util",
        "//ml_metadata/tools/mlmd_bench/proto:mlmd_bench_proto",
    ],
)

cc_library(
    name = "thread_runner",
    srcs = ["thread_runner.cc"],
    hdrs = ["thread_runner.h"],
    deps = [
        ":benchmark",
        ":stats",
        ":workload",
        "@com_google_absl//absl/synchronization",
        "//ml_metadata/metadata_store:metadata_store",
        "//ml_metadata/metadata_store:metadata_store_factory",
        "//ml_metadata/metadata_store:types",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/tools/mlmd_bench/proto:mlmd_bench_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

ml_metadata_cc_test(
    name = "thread_runner_test",
    size = "small",
    srcs = ["thread_runner_test.cc"],
    deps = [
        ":benchmark",
        ":stats",
        ":thread_runner",
        ":workload",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/metadata_store:metadata_store",
        "//ml_metadata/metadata_store:metadata_store_factory",
        "//ml_metadata/metadata_store:test_util",
        "//ml_metadata/proto:metadata_store_service_proto",
        "//ml_metadata/tools/mlmd_bench/proto:mlmd_bench_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "@com_google_absl//absl/types:variant",
        "//ml_metadata/metadata_store",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "//ml_metadata/tools/mlmd_bench/proto:mlmd_bench_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "workload",
    hdrs = ["workload.h"],
    deps = [
        ":stats",
        "@com_google_absl//absl/time",
        "//ml_metadata/metadata_store",
        "//ml_metadata/metadata_store:types",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

ml_metadata_cc_test(
    name = "workload_test",
    size = "small",
    srcs = ["workload_test.cc"],
    deps = [
        ":workload",
        #"@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/metadata_store",
        "//ml_metadata/metadata_store:metadata_store_factory",
        "//ml_metadata/metadata_store:types",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
