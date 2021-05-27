# Utilities for ML Metadata

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "metadata_source_query_config",
    srcs = ["metadata_source_query_config.cc"],
    hdrs = ["metadata_source_query_config.h"],
    deps = [
        "@com_google_protobuf//:protobuf",
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "status_utils",
    srcs = ["status_utils.cc"],
    hdrs = ["status_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "return_utils",
    hdrs = ["return_utils.h"],
)

cc_library(
    name = "struct_utils",
    srcs = ["struct_utils.cc"],
    hdrs = ["struct_utils.h"],
    deps = [
        "@com_google_protobuf//:protobuf",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_test(
    name = "struct_utils_test",
    srcs = ["struct_utils_test.cc"],
    deps = [
        ":struct_utils",
        "@com_google_protobuf//:protobuf",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//ml_metadata/metadata_store:test_util",
        "@com_google_glog//:glog",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_test(
    name = "metadata_source_query_config_test",
    srcs = ["metadata_source_query_config_test.cc"],
    deps = [
        ":metadata_source_query_config",
        "@com_google_googletest//:gtest_main",
    ],
)
