# Copyright 2021 Google LLC
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

# Placeholder: load py_library
load(
    "//ml_metadata:ml_metadata.bzl",
    "ml_metadata_pybind_extension",
)

package(default_visibility = [
    "//ml_metadata:__subpackages__",
])

licenses(["notice"])

ml_metadata_pybind_extension(
    name = "metadata_store_extension",
    srcs = [
        "metadata_store_extension.cc",
    ],
    module_name = "metadata_store_extension",
    deps = [
        "@com_google_absl//absl/status",
        "//ml_metadata/metadata_store:metadata_store_factory",
        "//ml_metadata/metadata_store:simple_types_util",
        "//ml_metadata/proto:metadata_store_proto",
        "//ml_metadata/simple_types/proto:simple_types_proto",
        "@pybind11",
        "@local_config_python//:python_headers",
    ],
)

py_library(
    name = "metadata_store_extension_lib",
    srcs = ["__init__.py"],
    deps = [
        ":metadata_store_extension",
    ],
)
