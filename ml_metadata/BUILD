licenses(["notice"])  # Apache 2.0

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "//ml_metadata/metadata_store:_pywrap_metadata_store_serialized.so",
        "//ml_metadata/metadata_store:pywrap_metadata_store_serialized.py",
        "//ml_metadata/proto:metadata_store_pb2.py",
        "//ml_metadata/proto:metadata_store_service_pb2.py",
    ],
)
