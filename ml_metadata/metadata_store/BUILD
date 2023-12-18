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
)

licenses(["notice"])

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
        ":constants",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_store_proto",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "constants",
    hdrs = [
        "constants.h",
    ],
    deps = ["@com_google_absl//absl/strings"],
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
        ":query_executor",
        "@com_google_protobuf//:protobuf",
        
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/simple_types:simple_types_constants",
        "//ml_metadata/util:field_mask_utils",
        "//ml_metadata/util:record_parsing_utils",
        "//ml_metadata/util:return_utils",
        "//ml_metadata/util:struct_utils",
        "@com_google_glog//:glog",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "rdbms_metadata_access_object_test",
    testonly = 1,
    srcs = [
        "rdbms_metadata_access_object_test.cc",
    ],
    hdrs = [
        "rdbms_metadata_access_object_test.h",
    ],
    deps = [
        ":rdbms_metadata_access_object",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:return_utils",
        "@com_google_glog//:glog",
    ],
)

ml_metadata_cc_test(
    name = "sqlite_rdbms_metadata_access_object_test",
    size = "small",
    srcs = ["sqlite_rdbms_metadata_access_object_test.cc"],
    deps = [
        ":metadata_source",
        ":query_config_executor",
        ":rdbms_metadata_access_object",
        ":rdbms_metadata_access_object_test",
        ":sqlite_metadata_source",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "query_executor",
    srcs = [
        "query_executor.cc",
    ],
    hdrs = [
        "query_executor.h",
    ],
    deps = [
        ":constants",
        ":metadata_source",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:return_utils",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "query_executor_test",
    testonly = 1,
    srcs = [
        "query_executor_test.cc",
    ],
    hdrs = [
        "query_executor_test.h",
    ],
    deps = [
        ":constants",
        ":metadata_source",
        ":query_executor",
        ":test_util",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:return_utils",
        "@com_google_glog//:glog",
    ],
)

ml_metadata_cc_test(
    name = "sqlite_query_config_executor_test",
    srcs = [
        "sqlite_query_config_executor_test.cc",
    ],
    deps = [
        ":metadata_source",
        ":query_config_executor",
        ":query_executor",
        ":query_executor_test",
        ":sqlite_metadata_source",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "mysql_query_config_executor_test",
    testonly = 1,
    srcs = [
        "mysql_query_config_executor_test.cc",
    ],
    deps = [
        ":metadata_source",
        ":mysql_metadata_source",
        ":query_config_executor",
        ":query_executor",
        ":query_executor_test",
        ":test_mysql_metadata_source_initializer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@com_google_glog//:glog",
    ],
)

# This test does not run on a Bazel sandbox because it requires a MYSQL server
# that is separately spawned. Details of the MYSQL server are specified by
# flags passed to the test. For e.g.,
#
# bazel run :standalone_mysql_query_config_executor_test -- \
#     --db_name="foo" \
#     --user_name="me" \
#     --host_name="localhost"
#
# See test_standalone_mysql_metadata_source_initializer.cc for the full flag
# list.
cc_test(
    name = "standalone_mysql_query_config_executor_test",
    tags = [
        "local",
        "manual",
    ],
    deps = [
        ":mysql_query_config_executor_test",
        ":test_standalone_mysql_metadata_source_initializer",
        "@com_google_googletest//:gtest_main",
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
        ":metadata_source",
        ":query_executor",
        "@com_google_protobuf//:protobuf",
        
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/query:filter_query_ast_resolver",
        "//ml_metadata/query:filter_query_builder",
        "//ml_metadata/util:return_utils",
        "//ml_metadata/util:struct_utils",
        "@com_google_glog//:glog",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "list_operation_query_helper",
    srcs = ["list_operation_query_helper.cc"],
    hdrs = ["list_operation_query_helper.h"],
    deps = [
        ":constants",
        ":list_operation_util",
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:return_utils",
    ],
)

cc_library(
    name = "list_operation_util",
    srcs = ["list_operation_util.cc"],
    hdrs = ["list_operation_util.h"],
    deps = [
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_store_proto",
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
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_store_proto",
        "@com_google_glog//:glog",
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
        ":constants",
        ":metadata_access_object_base",
        ":metadata_source",
        ":postgresql_query_executor",
        ":query_config_executor",
        ":rdbms_metadata_access_object",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:return_utils",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "metadata_source",
    srcs = ["metadata_source.cc"],
    hdrs = ["metadata_source.h"],
    deps = [
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/util:return_utils",
    ],
)

ml_metadata_cc_test(
    name = "metadata_source_test",
    size = "small",
    srcs = ["metadata_source_test.cc"],
    deps = [
        ":metadata_source",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//ml_metadata/proto:metadata_source_proto",
    ],
)

cc_library(
    name = "transaction_executor",
    srcs = ["transaction_executor.cc"],
    hdrs = ["transaction_executor.h"],
    deps = [
        ":metadata_source",
        "@com_google_absl//absl/status",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:return_utils",
    ],
)

ml_metadata_cc_test(
    name = "transaction_executor_test",
    srcs = ["transaction_executor_test.cc"],
    deps = [
        ":metadata_source",
        ":transaction_executor",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "metadata_store_service_interface",
    hdrs = ["metadata_store_service_interface.h"],
    deps = [
        "@com_google_absl//absl/status",
        "//ml_metadata/proto:metadata_store_service_proto",
    ],
)

cc_library(
    name = "metadata_store",
    srcs = ["metadata_store.cc"],
    hdrs = ["metadata_store.h"],
    deps = [
        ":constants",
        ":metadata_access_object_factory",
        ":metadata_source",
        ":metadata_store_service_interface",
        ":rdbms_metadata_access_object",
        ":simple_types_util",
        ":transaction_executor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "//ml_metadata/simple_types:simple_types_constants",
        "//ml_metadata/simple_types/proto:simple_types_proto",
        "//ml_metadata/util:return_utils",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "metadata_store_test_suite",
    testonly = 1,
    srcs = ["metadata_store_test_suite.cc"],
    hdrs = ["metadata_store_test_suite.h"],
    deps = [
        ":metadata_store",
        ":simple_types_util",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "//ml_metadata/simple_types/proto:simple_types_proto",
    ],
)

ml_metadata_cc_test(
    name = "metadata_store_cc_test",
    srcs = ["metadata_store_test.cc"],
    env = {
        "ASAN_OPTIONS": "detect_odr_violation=0",
    },
    deps = [
        ":metadata_store",
        ":metadata_store_test_suite",
        ":sqlite_metadata_source",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "//ml_metadata/util:return_utils",
        "@com_google_glog//:glog",
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
        ":constants",
        ":types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto:metadata_store_service_proto",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "metadata_store_factory",
    srcs = ["metadata_store_factory.cc"],
    hdrs = ["metadata_store_factory.h"],
    deps = [
        ":metadata_store",
        ":mysql_metadata_source",
        ":postgresql_metadata_source",
        ":sqlite_metadata_source",
        ":transaction_executor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "//ml_metadata/util:return_utils",
    ],
)

ml_metadata_cc_test(
    name = "metadata_store_factory_test",
    srcs = ["metadata_store_factory_test.cc"],
    env = {
        "ASAN_OPTIONS": "detect_odr_violation=0",
    },
    deps = [
        ":metadata_access_object_factory",
        ":metadata_store",
        ":metadata_store_factory",
        ":test_util",
        "@com_google_googletest//:gtest_main",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "sqlite_metadata_source_util",
    srcs = ["sqlite_metadata_source_util.cc"],
    hdrs = ["sqlite_metadata_source_util.h"],
    deps = [
        ":constants",
        "@com_google_absl//absl/status:statusor",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//ml_metadata/proto:metadata_store_proto",
        "@com_google_glog//:glog",
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
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    hdrs = ["test_util.h"],
    deps = [
        "@com_google_protobuf//:protobuf",
        "@com_google_googletest//:gtest",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "simple_types_util",
    srcs = ["simple_types_util.cc"],
    hdrs = ["simple_types_util.h"],
    deps = [
        "@com_google_protobuf//:protobuf",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/simple_types:simple_types_constants",
        "//ml_metadata/simple_types/proto:simple_types_proto",
    ],
)

ml_metadata_cc_test(
    name = "simple_types_util_test",
    srcs = ["simple_types_util_test.cc"],
    deps = [
        ":simple_types_util",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//ml_metadata/proto:metadata_store_proto",
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

cc_library(
    name = "mysql_metadata_source",
    srcs = ["mysql_metadata_source.cc"],
    hdrs = ["mysql_metadata_source.h"],
    deps = [
        ":constants",
        ":metadata_source",
        ":sqlite_metadata_source_util",
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:return_utils",
        "@com_google_glog//:glog",
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
        "@com_google_absl//absl/status",
        "//ml_metadata/proto:metadata_store_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_glog//:glog",
    ],
)

cc_binary(
    name = "metadata_store_server",
    srcs = ["metadata_store_server_main.cc"],
    deps = [
        ":metadata_store",
        ":metadata_store_factory",
        ":metadata_store_service_impl",
        "@com_google_protobuf//:protobuf",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_store_proto",
        "@com_github_gflags_gflags//:gflags_nothreads",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_glog//:glog",
    ],
)

# An abstract type for testing MetadataAccessObject implementations.
cc_library(
    name = "metadata_access_object_test",
    testonly = 1,
    srcs = ["metadata_access_object_test.cc"],
    hdrs = ["metadata_access_object_test.h"],
    deps = [
        ":constants",
        ":metadata_access_object_base",
        ":metadata_source",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/proto/testing:mock_proto",
        "//ml_metadata/util:return_utils",
        "@com_github_gflags_gflags//:gflags_nothreads",
        "@com_google_glog//:glog",
        "@com_google_protobuf//:cc_wkt_protos",
        "@com_google_protobuf//:protobuf",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
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
        "@com_github_gflags_gflags//:gflags_nothreads",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "mysql_metadata_source_test",
    testonly = 1,
    srcs = [
        "mysql_metadata_source_test.cc",
    ],
    deps = [
        ":metadata_source",
        ":metadata_source_test_suite",
        ":mysql_metadata_source",
        ":test_mysql_metadata_source_initializer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_github_gflags_gflags//:gflags_nothreads",
        "@com_google_glog//:glog",
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
    srcs = ["sqlite_metadata_access_object_test.cc"],
    deps = [
        ":metadata_access_object_factory",
        ":metadata_access_object_test",
        ":metadata_source",
        ":sqlite_metadata_source",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@com_google_glog//:glog",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/util:metadata_source_query_config",
        "@com_google_glog//:glog",
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
        ":test_standalone_mysql_metadata_source_initializer",
        "@com_google_googletest//:gtest_main",
    ],
)

# This test requires a PostgreSQL server that is separately spawned. Details of the PostgreSQL
# server are specified by flags passed to the test. For e.g.,
#
# bazel run :standalone_postgresql_metadata_access_object_test -- \
#     --db_name="foo" \
#     --host_name="localhost"\
#     --user_name="me" \
#     --password="thisispassword"\
#     --port="3456"
#
# See standalone_postgresql_metadata_access_object_test.cc for the full flag
# list.
cc_test(
    name = "standalone_postgresql_metadata_access_object_test",
    env = {
        "ASAN_OPTIONS": "detect_odr_violation=0",
    },
    tags = [
        "local",
        "manual",
    ],
    deps = [
        ":test_postgresql_standalone_metadata_source_initializer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "standalone_postgresql_metadata_source_test",
    tags = [
        "local",
        "manual",
    ],
    deps = [
        ":postgresql_metadata_source_test",
        ":test_postgresql_standalone_metadata_source_initializer",
    ],
)

cc_library(
    name = "postgresql_metadata_source_test",
    testonly = 1,
    srcs = [
        "postgresql_metadata_source_test.cc",
    ],
    deps = [
        ":metadata_source",
        ":metadata_source_test_suite",
        ":postgresql_metadata_source",
        ":test_postgresql_metadata_source_initializer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_github_gflags_gflags//:gflags_nothreads",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "test_postgresql_standalone_metadata_source_initializer",
    testonly = 1,
    srcs = ["test_postgresql_standalone_metadata_source_initializer.cc"],
    deps = [
        ":postgresql_metadata_source",
        ":test_postgresql_metadata_source_initializer",
        "@com_google_absl//absl/memory",
        "//ml_metadata/proto:metadata_store_proto",
        "@com_github_gflags_gflags//:gflags_nothreads",
        "@com_google_glog//:glog",
        "@postgresql",
    ],
)

# PostgreSQL build and test targets.
cc_library(
    name = "postgresql_metadata_source",
    srcs = ["postgresql_metadata_source.cc"],
    hdrs = ["postgresql_metadata_source.h"],
    deps = [
        ":constants",
        ":metadata_source",
        ":sqlite_metadata_source_util",
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/util:return_utils",
        "@com_google_glog//:glog",
        "@postgresql",
    ],
)

cc_library(
    name = "postgresql_query_executor",
    srcs = ["postgresql_query_executor.cc"],
    hdrs = ["postgresql_query_executor.h"],
    deps = [
        ":list_operation_query_helper",
        ":metadata_source",
        ":query_config_executor",
        ":query_executor",
        "@com_google_protobuf//:protobuf",
        
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/query:filter_query_ast_resolver",
        "//ml_metadata/query:filter_query_builder",
        "//ml_metadata/util:return_utils",
        "//ml_metadata/util:struct_utils",
        "@com_google_glog//:glog",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "postgresql_query_executor_test",
    testonly = 1,
    srcs = [
        "postgresql_query_executor_test.cc",
    ],
    deps = [
        ":postgresql_query_executor",
        ":query_executor",
        ":query_executor_test",
        ":test_postgresql_metadata_source_initializer",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "//ml_metadata/util:metadata_source_query_config",
    ],
)

cc_library(
    name = "test_postgresql_metadata_source_initializer",
    testonly = 1,
    hdrs = [
        "test_postgresql_metadata_source_initializer.h",
    ],
    deps = [":postgresql_metadata_source"],
)

ml_metadata_cc_test(
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
        "@com_google_glog//:glog",
    ],
)
