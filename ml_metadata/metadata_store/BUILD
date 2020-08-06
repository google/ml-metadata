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

load(
    "//ml_metadata:ml_metadata.bzl",
    "ml_metadata_cc_test",
    "ml_metadata_go_library",
    "ml_metadata_go_test",
    "ml_metadata_go_wrap_cc",
    "ml_metadata_py_wrap_cc",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "metadata_access_object_base",
    hdrs = [
        "metadata_access_object.h",
    ],
    deps = [
        ":metadata_source",
        "@com_google_protobuf//:protobuf",
        
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "constants",
    hdrs = [
        "constants.h",
    ],
)

cc_library(
    name = "rdbms_metadata_access_object",
    srcs = [
        "rdbms_metadata_access_object.cc",
    ],
    hdrs = [
        "rdbms_metadata_access_object.h",
    ],
    deps = [
        ":constants",
        ":list_operation_util",
        ":metadata_access_object_base",
        ":metadata_source",
        ":query_executor",
        "@com_google_protobuf//:protobuf",
        
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "query_executor",
    hdrs = [
        "query_executor.h",
    ],
    deps = [
        ":constants",
        ":metadata_access_object_base",
        ":metadata_source",
        "@com_google_protobuf//:protobuf",
        
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "query_config_executor",
    srcs = [
        "query_config_executor.cc",
    ],
    hdrs = [
        "query_config_executor.h",
    ],
    deps = [
        ":list_operation_query_helper",
        ":list_operation_util",
        ":metadata_access_object_base",
        ":metadata_source",
        ":query_executor",
        "@com_google_protobuf//:protobuf",
        
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "list_operation_query_helper",
    srcs = ["list_operation_query_helper.cc"],
    hdrs = ["list_operation_query_helper.h"],
    deps = [
        ":list_operation_util",
        ":types",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "list_operation_util",
    srcs = ["list_operation_util.cc"],
    hdrs = ["list_operation_util.h"],
    deps = [
        ":types",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

ml_metadata_cc_test(
    name = "list_operation_query_helper_test",
    size = "small",
    srcs = ["list_operation_query_helper_test.cc"],
    deps = [
        ":list_operation_query_helper",
        ":test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "metadata_access_object_factory",
    srcs = [
        "metadata_access_object_factory.cc",
    ],
    hdrs = [
        "metadata_access_object.h",
        "metadata_access_object_factory.h",
    ],
    deps = [
        ":metadata_access_object_base",
        ":metadata_source",
        ":query_config_executor",
        ":rdbms_metadata_access_object",
        "@com_google_absl//absl/memory",
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

ml_metadata_cc_test(
    name = "metadata_source_test",
    size = "small",
    srcs = ["metadata_source_test.cc"],
    deps = [
        ":metadata_source",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/proto:metadata_source_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "transaction_executor",
    srcs = ["transaction_executor.cc"],
    hdrs = ["transaction_executor.h"],
    deps = [
        ":metadata_source",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

ml_metadata_cc_test(
    name = "transaction_executor_test",
    srcs = ["transaction_executor_test.cc"],
    deps = [
        ":metadata_source",
        ":transaction_executor",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "metadata_store_service_interface",
    hdrs = ["metadata_store_service_interface.h"],
    deps = [
        "//ml_metadata/proto:metadata_store_service_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "metadata_store",
    srcs = ["metadata_store.cc"],
    hdrs = ["metadata_store.h"],
    deps = [
        ":metadata_access_object_factory",
        ":metadata_source",
        ":metadata_store_service_interface",
        ":transaction_executor",
        "@com_google_protobuf//:protobuf",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "metadata_store_test_suite",
    testonly = 1,
    srcs = ["metadata_store_test_suite.cc"],
    hdrs = ["metadata_store_test_suite.h"],
    deps = [
        ":constants",
        ":metadata_store",
        ":test_util",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

ml_metadata_cc_test(
    name = "metadata_store_cc_test",
    srcs = ["metadata_store_test.cc"],
    deps = [
        ":metadata_store",
        ":metadata_store_test_suite",
        ":sqlite_metadata_source",
        ":test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "metadata_store_headers",
    hdrs = [
        "metadata_access_object.h",
        "metadata_access_object_factory.h",
        "metadata_source.h",
        "metadata_store.h",
        "metadata_store_factory.h",
        "metadata_store_service_interface.h",
        "transaction_executor.h",
    ],
    deps = [
        ":types",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "@org_tensorflow//tensorflow/core:lib",
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
        ":transaction_executor",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

ml_metadata_cc_test(
    name = "metadata_store_factory_test",
    srcs = ["metadata_store_factory_test.cc"],
    deps = [
        ":metadata_access_object_factory",
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
    name = "sqlite_metadata_source_util",
    srcs = ["sqlite_metadata_source_util.cc"],
    hdrs = ["sqlite_metadata_source_util.h"],
    deps = [
        ":constants",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "@org_sqlite",
    ],
)

cc_library(
    name = "sqlite_metadata_source",
    srcs = ["sqlite_metadata_source.cc"],
    hdrs = ["sqlite_metadata_source.h"],
    deps = [
        ":metadata_source",
        ":sqlite_metadata_source_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_sqlite",
    ],
)

ml_metadata_cc_test(
    name = "sqlite_metadata_source_test",
    size = "small",
    srcs = ["sqlite_metadata_source_test.cc"],
    deps = [
        ":metadata_source_test_suite",
        ":sqlite_metadata_source",
        ":test_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    hdrs = ["test_util.h"],
    deps = [
        "@com_google_protobuf//:protobuf",
        "@com_google_googletest//:gtest",
    ],
)

ml_metadata_cc_test(
    name = "test_util_test",
    srcs = ["test_util_test.cc"],
    deps = [
        ":test_util",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/proto:metadata_store_proto",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
)

# Note that the name of the target should follow specific naming
# pattern specified in tensorflow/tf_exported_symbols.lds in order
# for the init function in the generated .so file to be exported.
ml_metadata_py_wrap_cc(
    name = "pywrap_tf_metadata_store_serialized",
    srcs = ["tf_metadata_store_serialized.i"],
    deps = [
        ":metadata_store_factory",
        "@local_config_python//:python_headers",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "mysql_metadata_source",
    srcs = ["mysql_metadata_source.cc"],
    hdrs = ["mysql_metadata_source.h"],
    deps = [
        ":constants",
        ":metadata_source",
        ":types",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@libmysqlclient",
    ],
)

cc_library(
    name = "metadata_store_service_impl",
    srcs = ["metadata_store_service_impl.cc"],
    hdrs = ["metadata_store_service_impl.h"],
    deps = [
        ":metadata_store",
        ":metadata_store_factory",
        "//ml_metadata/proto:metadata_store_proto",
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

# An abstract type for testing MetadataAccessObject implementations.
cc_library(
    name = "metadata_access_object_test",
    testonly = 1,
    srcs = ["metadata_access_object_test.cc"],
    hdrs = ["metadata_access_object_test.h"],
    deps = [
        ":metadata_access_object_factory",
        ":test_util",
        "@com_google_protobuf//:protobuf",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/time",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@com_github_gflags_gflags//:gflags_nothreads",
    ],
)

# An abstract type for testing MetadataSource implementations.
cc_library(
    name = "metadata_source_test_suite",
    testonly = 1,
    srcs = ["metadata_source_test_suite.cc"],
    hdrs = ["metadata_source_test_suite.h"],
    deps = [
        ":constants",
        ":metadata_source",
        ":test_util",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "test_mysql_metadata_source_initializer",
    testonly = 1,
    hdrs = [
        "test_mysql_metadata_source_initializer.h",
    ],
    deps = [":mysql_metadata_source"],
)

cc_library(
    name = "test_standalone_mysql_metadata_source_initializer",
    testonly = 1,
    srcs = ["test_standalone_mysql_metadata_source_initializer.cc"],
    deps = [
        ":mysql_metadata_source",
        ":test_mysql_metadata_source_initializer",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_store_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@com_github_gflags_gflags//:gflags_nothreads",
    ],
)

cc_library(
    name = "mysql_metadata_source_test",
    testonly = 1,
    srcs = [
        "mysql_metadata_source_test.cc",
    ],
    deps = [
        ":constants",
        ":metadata_source",
        ":metadata_source_test_suite",
        ":mysql_metadata_source",
        ":test_mysql_metadata_source_initializer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_github_gflags_gflags//:gflags_nothreads",
    ],
)

# This test does not run on a Bazel sandbox because it requires a MYSQL server
# that is separately spawned. Details of the MYSQL server are specified by
# flags passed to the test. For e.g.,
#
# bazel run :standalone_mysql_metadata_source_test -- \
#     --db_name="foo" \
#     --user_name="me" \
#     --host_name="localhost"
#
# See test_standalone_mysql_metadata_source_initializer.cc for the full flag
# list.
cc_test(
    name = "standalone_mysql_metadata_source_test",
    tags = [
        "local",
        "manual",
    ],
    deps = [
        ":mysql_metadata_source_test",
        ":test_standalone_mysql_metadata_source_initializer",
    ],
)

ml_metadata_cc_test(
    name = "sqlite_metadata_access_object_test",
    size = "small",
    srcs = ["sqlite_metadata_access_object_test.cc"],
    deps = [
        ":metadata_access_object_factory",
        ":metadata_access_object_test",
        ":metadata_source",
        ":sqlite_metadata_source",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "mysql_metadata_access_object_test",
    testonly = 1,
    srcs = ["mysql_metadata_access_object_test.cc"],
    deps = [
        ":metadata_access_object_factory",
        ":metadata_access_object_test",
        ":metadata_source",
        ":mysql_metadata_source",
        ":test_mysql_metadata_source_initializer",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

# This test does not run on a Bazel sandbox because it requires a MYSQL server
# that is separately spawned. Details of the MYSQL server are specified by
# flags passed to the test. For e.g.,
#
# bazel run :standalone_mysql_metadata_access_object_test -- \
#     --db_name="foo" \
#     --user_name="me" \
#     --host_name="localhost"
#
# See test_standalone_mysql_metadata_source_initializer.cc for the full flag
# list.
cc_test(
    name = "standalone_mysql_metadata_access_object_test",
    tags = [
        "local",
        "manual",
    ],
    deps = [
        ":mysql_metadata_access_object_test",
        ":test_standalone_mysql_metadata_source_initializer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "metadata_access_object_factory_test",
    srcs = ["metadata_access_object_factory_test.cc"],
    deps = [
        ":metadata_access_object_factory",
        ":metadata_source",
        ":sqlite_metadata_source",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

ml_metadata_go_wrap_cc(
    name = "metadata_store_go_wrap",
    importpath = "ml_metadata/metadata_store/metadata_store_go_wrap",
    libname = "metadata_store_cgo_wrap",
    swigfile = "metadata_store_go",
    deps = [
        ":metadata_store_factory",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

ml_metadata_go_library(
    name = "metadata_store_go",
    srcs = ["metadata_store.go"],
    cdeps = ["metadata_store_cgo_wrap"],
    cgo = True,
    importpath = "ml_metadata/metadata_store/mlmetadata",
    deps = [
        ":metadata_store_go_wrap",
        "//ml_metadata/proto:metadata_store_go_proto",
        "//ml_metadata/proto:metadata_store_service_go_proto",
        "@com_github_golang_protobuf//proto:go_default_library",
    ],
)

ml_metadata_go_test(
    name = "metadata_store_go_test",
    size = "small",
    srcs = ["metadata_store_test.go"],
    library = ":metadata_store_go",
    deps = [
        "//ml_metadata/proto:metadata_store_go_proto",
        "@com_github_golang_protobuf//proto:go_default_library",
        "@com_github_google_go_cmp//cmp:go_default_library",
    ],
)
