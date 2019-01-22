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

# GOOGLE BEGIN-INTERNAL

load("//ml_metadata:ml_metadata.bzl", "ml_metadata_proto_library", "ml_metadata_proto_library_py")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

ml_metadata_proto_library(
    name = "metadata_store_proto",
    srcs = ["metadata_store.proto"],
    cc_api_version = 2,
    deps = [
    ],
)

ml_metadata_proto_library(
    name = "metadata_store_service_proto",
    srcs = ["metadata_store_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    deps = [
        ":metadata_store_proto",
    ],
)

ml_metadata_proto_library_py(
    name = "metadata_store_py_pb2",
    srcs = ["metadata_store.proto"],
    api_version = 2,
    proto_library = "metadata_store_proto",
)

ml_metadata_proto_library_py(
    name = "metadata_store_service_py_pb2",
    srcs = ["metadata_store_service.proto"],
    api_version = 2,
    oss_deps = [":metadata_store_py_pb2"],
    proto_library = "metadata_store_service_proto",
)


ml_metadata_proto_library(
    name = "metadata_source_proto",
    srcs = ["metadata_source.proto"],
    cc_api_version = 2,
)
