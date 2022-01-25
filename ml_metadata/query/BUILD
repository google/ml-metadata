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

licenses(["notice"])

cc_library(
    name = "filter_query_ast_resolver",
    srcs = ["filter_query_ast_resolver.cc"],
    hdrs = ["filter_query_ast_resolver.h"],
    visibility = ["//ml_metadata:__subpackages__"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:return_utils",
        "@com_googlesource_code_re2//:re2",
        "@com_google_zetasql//zetasql/public:analyzer",
        "@com_google_zetasql//zetasql/public:simple_catalog",
    ],
)

ml_metadata_cc_test(
    name = "filter_query_ast_resolver_test",
    size = "small",
    srcs = ["filter_query_ast_resolver_test.cc"],
    deps = [
        ":filter_query_ast_resolver",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//ml_metadata/metadata_store:test_util",
        "//ml_metadata/proto:metadata_store_proto",
    ],
)

cc_library(
    name = "filter_query_builder",
    srcs = ["filter_query_builder.cc"],
    hdrs = ["filter_query_builder.h"],
    visibility = ["//ml_metadata:__subpackages__"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//ml_metadata/metadata_store:constants",
        "//ml_metadata/proto:metadata_store_proto",
        "@com_google_glog//:glog",
        "@com_google_zetasql//zetasql/public:strings",
        "@com_google_zetasql//zetasql/resolved_ast:sql_builder",
    ],
)

ml_metadata_cc_test(
    name = "filter_query_builder_test",
    size = "small",
    srcs = ["filter_query_builder_test.cc"],
    deps = [
        ":filter_query_ast_resolver",
        ":filter_query_builder",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//ml_metadata/metadata_store:test_util",
        "//ml_metadata/proto:metadata_store_proto",
    ],
)
