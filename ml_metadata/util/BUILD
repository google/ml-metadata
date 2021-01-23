# Utilities for ML Metadata

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "metadata_source_query_config",
    srcs = ["metadata_source_query_config.cc"],
    hdrs = ["metadata_source_query_config.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "struct_utils",
    srcs = ["struct_utils.cc"],
    hdrs = ["struct_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:lib",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_test(
    name = "struct_utils_test",
    srcs = ["struct_utils_test.cc"],
    deps = [
        ":struct_utils",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "//ml_metadata/metadata_store:test_util",
        "@org_tensorflow//tensorflow/core:test",
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
