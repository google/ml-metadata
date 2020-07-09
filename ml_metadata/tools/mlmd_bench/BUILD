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
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "//ml_metadata/metadata_store:metadata_store",
        "//ml_metadata/metadata_store:types",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "//ml_metadata/tools/mlmd_bench/proto:mlmd_bench_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ]
)

ml_metadata_cc_test(
    name = "fill_types_workload_test",
    size = "small",
    srcs = ["fill_types_workload_test.cc"],
    deps = [
        ":fill_types_workload",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/metadata_store:metadata_store",
        "//ml_metadata/metadata_store:metadata_store_factory",
        "//ml_metadata/metadata_store:test_util",
        "//ml_metadata/metadata_store:types",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "//ml_metadata/tools/mlmd_bench/proto:mlmd_bench_proto",
        "@org_tensorflow//tensorflow/core:test",
    ]
)

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
        "@com_google_absl//absl/time",
        "//ml_metadata/metadata_store:types",
    ]
)

cc_library(
    name = "workload",
    hdrs = ["workload.h"],
    deps = [
        ":stats",
        "@com_google_absl//absl/time",
        "//ml_metadata/metadata_store:metadata_store",
        "//ml_metadata/metadata_store:types",
        "@org_tensorflow//tensorflow/core:lib",
    ]
)

ml_metadata_cc_test(
    name = "workload_test",
    size = "small",
    srcs = ["workload_test.cc"],
    deps = [
        ":workload",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/metadata_store:metadata_store",
        "//ml_metadata/metadata_store:metadata_store_factory",
        "//ml_metadata/metadata_store:types",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ]
)
