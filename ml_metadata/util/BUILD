# Utilities for ML Metadata

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "metadata_source_query_config",
    srcs = ["metadata_source_query_config.cc"],
    hdrs = ["metadata_source_query_config.h"],
    deps = [
        "//ml_metadata/proto:metadata_source_proto",
        "@org_tensorflow//tensorflow/core:lib",
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
