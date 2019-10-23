licenses(["notice"])  # Apache 2.0

config_setting(
    name = "windows",
    constraint_values = [
        "@bazel_tools//platforms:windows",
    ],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "//ml_metadata/metadata_store:pywrap_tf_metadata_store_serialized.py",
        "//ml_metadata/proto:metadata_store_pb2.py",
        "//ml_metadata/proto:metadata_store_service_pb2.py",
        "//ml_metadata/proto:metadata_store_service_pb2_grpc.py",
    ] + select({
        ":windows": [
            "//ml_metadata/metadata_store:_pywrap_tf_metadata_store_serialized.pyd",
        ],
        "//conditions:default": [
            "//ml_metadata/metadata_store:_pywrap_tf_metadata_store_serialized.so",
        ],
    }),
)
