# Copyright 2018 Google LLC
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

"""ML METADATA Data Validation external dependencies that can be loaded in WORKSPACE files.
"""

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

# Sanitize a dependency so that it works correctly from code that includes
# ML Metadata as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def ml_metadata_workspace():
    """All ML Metadata external dependencies."""
    tf_workspace(
        path_prefix = "",
        tf_repo_name = "org_tensorflow",
    )

    # for grpc
    native.bind(
        name = "libssl",
        actual = "@boringssl//:ssl",
    )

    native.bind(
        name = "zlib",
        actual = "@zlib_archive//:zlib",
    )

    native.bind(
        name = "cares",
        actual = "@grpc//third_party/nanopb:nanopb",
    )
