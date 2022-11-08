# Copyright 2022 Google LLC
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

# This mock protobuffer is used to unit test PROTO value storage.
# The mock_proto cc_library provides a utility function for building the proto

load("//ml_metadata:ml_metadata.bzl", "ml_metadata_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

ml_metadata_proto_library(
    name = "mock_proto",
    srcs = ["mock.proto"],
    cc_api_version = 2,
)
