# Utilities for ML Metadata

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "metadata_source_query_config",
    srcs = ["metadata_source_query_config.cc"],
    hdrs = ["metadata_source_query_config.h"],
    deps = [
        "@protobuf//:protobuf",
        "@abseil-cpp//absl/strings",
        "//ml_metadata/proto:metadata_source_proto",
        "@glog//:glog",
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
        "@protobuf//:protobuf",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@protobuf//:cc_wkt_protos",
    ],
)

cc_test(
    name = "struct_utils_test",
    srcs = ["struct_utils_test.cc"],
    deps = [
        ":struct_utils",
        "@protobuf//:protobuf",
        "@googletest//:gtest_main",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "//ml_metadata/metadata_store:test_util",
        "@glog//:glog",
        "@protobuf//:cc_wkt_protos",
    ],
)

cc_test(
    name = "metadata_source_query_config_test",
    srcs = ["metadata_source_query_config_test.cc"],
    deps = [
        ":metadata_source_query_config",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "field_mask_utils",
    srcs = ["field_mask_utils.cc"],
    hdrs = ["field_mask_utils.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "//ml_metadata/proto:metadata_store_proto",
        "@protobuf//:protobuf",
    ],
)

cc_test(
    name = "field_mask_utils_test",
    srcs = ["field_mask_utils_test.cc"],
    deps = [
        ":field_mask_utils",
        "@googletest//:gtest_main",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "//ml_metadata/proto:metadata_store_proto",
        "@protobuf//:protobuf",
    ],
)

cc_library(
    name = "record_parsing_utils",
    srcs = ["record_parsing_utils.cc"],
    hdrs = ["record_parsing_utils.h"],
    deps = [
        ":return_utils",
        
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "//ml_metadata/metadata_store:constants",
        "//ml_metadata/proto:metadata_source_proto",
        "//ml_metadata/proto:metadata_store_proto",
        "@glog//:glog",
    ],
)

cc_test(
    name = "record_parsing_utils_test",
    srcs = ["record_parsing_utils_test.cc"],
    deps = [
        ":record_parsing_utils",
        "@googletest//:gtest_main",
        "@abseil-cpp//absl/status",
        "//ml_metadata/metadata_store:test_util",
        "//ml_metadata/proto:metadata_store_proto",
    ],
)
