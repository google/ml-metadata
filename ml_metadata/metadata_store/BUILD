# Copyright 2019 Google LLC
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

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "metadata_access_object",
    srcs = ["metadata_access_object.cc"],
    hdrs = ["metadata_access_object.h"],
    deps = [
        ":metadata_source",
        "@protobuf_archive//:protobuf",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "metadata_source",
    srcs = ["metadata_source.cc"],
    hdrs = ["metadata_source.h"],
    deps = [
        ":types",
        "//ml_metadata/proto:metadata_source_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "metadata_source_test",
    size = "small",
    srcs = ["metadata_source_test.cc"],
    deps = [
        ":metadata_source",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/proto:metadata_source_proto",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "metadata_store",
    srcs = ["metadata_store.cc"],
    hdrs = ["metadata_store.h"],
    deps = [
        ":metadata_access_object",
        ":metadata_source",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

# Added cc to name to not collide with python.
cc_test(
    name = "metadata_store_cc_test",
    srcs = ["metadata_store_test.cc"],
    deps = [
        ":metadata_access_object",
        ":metadata_store",
        ":sqlite_metadata_source",
        ":test_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "metadata_store_factory",
    srcs = ["metadata_store_factory.cc"],
    hdrs = ["metadata_store_factory.h"],
    deps = [
        ":metadata_store",
        ":mysql_metadata_source",
        ":sqlite_metadata_source",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "metadata_store_factory_test",
    srcs = ["metadata_store_factory_test.cc"],
    deps = [
        ":metadata_access_object",
        ":metadata_store",
        ":metadata_store_factory",
        ":test_util",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "sqlite_metadata_source",
    srcs = ["sqlite_metadata_source.cc"],
    hdrs = ["sqlite_metadata_source.h"],
    deps = [
        ":metadata_source",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_sqlite",
    ],
)

cc_test(
    name = "sqlite_metadata_source_test",
    size = "small",
    srcs = ["sqlite_metadata_source_test.cc"],
    deps = [
        ":sqlite_metadata_source",
        ":test_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        ":types",
        "@com_google_protobuf//:protobuf",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
    ],
)

# Note that the name of the target should follow specific naming
# pattern specified in tensorflow/tf_exported_symbols.lds in order
# for the init function in the generated .so file to be exported.
tf_py_wrap_cc(
    name = "pywrap_metadata_store_serialized",
    srcs = ["metadata_store_serialized.i"],
    deps = [
        ":metadata_store_factory",
        "@local_config_python//:python_headers",
        "@org_tensorflow//tensorflow/c:tf_status_helper",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "mysql_metadata_source",
    srcs = ["mysql_metadata_source.cc"],
    hdrs = ["mysql_metadata_source.h"],
    linkopts = ["-lmysqlclient"],
    deps = [
        ":metadata_source",
        ":types",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "metadata_store_service_impl",
    srcs = ["metadata_store_service_impl.cc"],
    hdrs = ["metadata_store_service_impl.h"],
    deps = [
        ":metadata_store",
        "@com_google_absl//absl/synchronization",
        "//ml_metadata/proto:metadata_store_service_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@grpc//:grpc++",
    ],
)

cc_binary(
    name = "metadata_store_server",
    srcs = ["metadata_store_server_main.cc"],
    deps = [
        ":metadata_store",
        ":metadata_store_factory",
        ":metadata_store_service_impl",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@com_github_gflags_gflags//:gflags_nothreads",
        "@grpc//:grpc++",
    ],
)

# An abstract type for testing MetadataAccessObject
# implementations.
cc_library(
    name = "metadata_access_object_test",
    testonly = 1,
    srcs = ["metadata_access_object_test.cc"],
    hdrs = ["metadata_access_object_test.h"],
    deps = [
        ":metadata_access_object",
        ":test_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/time",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
