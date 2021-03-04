licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = [
        "@bazel_tools//platforms:windows",
    ],
)

config_setting(
    name = "macos",
    constraint_values = [
        "@bazel_tools//platforms:osx",
    ],
)

_public_protos = [
    "//ml_metadata/proto:metadata_store_pb2.py",
    "//ml_metadata/proto:metadata_store_service_pb2.py",
    "//ml_metadata/proto:metadata_store_service_pb2_grpc.py",
]

_py_extension = select({
    ":windows": [
        "//ml_metadata/metadata_store/pywrap:metadata_store_extension.pyd",
    ],
    "//conditions:default": [
        "//ml_metadata/metadata_store/pywrap:metadata_store_extension.so",
    ],
})


sh_binary(
    name = "move_generated_files",
    srcs = ["move_generated_files.sh"],
    data = _py_extension + _public_protos,
)
