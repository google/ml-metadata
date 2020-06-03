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
"""
This module contains build rules for ml_metadata in OSS.
"""

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")
load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")
load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")

def ml_metadata_cc_test(
        name,
        srcs = [],
        deps = [],
        tags = [],
        args = [],
        size = None,
        data = None):
    _ignore = [data]
    native.cc_test(
        name = name,
        srcs = srcs,
        deps = deps,
        tags = tags,
        args = args,
        size = size,
        # cc_tests with ".so"s in srcs incorrectly link on Darwin unless
        # linkstatic=1 (https://github.com/bazelbuild/bazel/issues/3450).
        linkstatic = select({
            "//ml_metadata/metadata_store:darwin": 1,
            "//conditions:default": 0,
        }),
    )

def ml_metadata_proto_library(
        name,
        srcs = [],
        has_services = False,
        deps = [],
        visibility = None,
        testonly = 0,
        cc_grpc_version = None,
        cc_api_version = 2):
    """Opensource cc_proto_library."""
    _ignore = [has_services, cc_api_version]
    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs,
        testonly = testonly,
    )

    use_grpc_plugin = None
    if cc_grpc_version:
        use_grpc_plugin = True
    cc_proto_library(
        name = name,
        srcs = srcs,
        deps = deps,
        cc_libs = ["@com_google_protobuf//:protobuf"],
        protoc = "@com_google_protobuf//:protoc",
        default_runtime = "@com_google_protobuf//:protobuf",
        use_grpc_plugin = use_grpc_plugin,
        testonly = testonly,
        visibility = visibility,
    )

def ml_metadata_proto_library_py(
        name,
        proto_library = None,
        api_version = None,
        srcs = [],
        deps = [],
        visibility = None,
        testonly = 0,
        oss_deps = [],
        use_grpc_plugin = False):
    """Opensource py_proto_library."""
    _ignore = [proto_library, api_version, oss_deps]
    py_proto_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY2AND3",
        deps = ["@com_google_protobuf//:protobuf_python"] + deps + oss_deps,
        default_runtime = "@com_google_protobuf//:protobuf_python",
        protoc = "@com_google_protobuf//:protoc",
        visibility = visibility,
        testonly = testonly,
        use_grpc_plugin = use_grpc_plugin,
    )

def ml_metadata_proto_library_go(
        name,
        deps = [],
        srcs = [],
        importpath = None,
        cc_proto_deps = [],
        go_proto_deps = [],
        gen_oss_grpc = False):
    """Opensource go_proto_library."""
    proto_library_name = deps[0][1:] + "_copy"

    # add a proto_library rule for bazel go rules
    proto_library_deps = []
    for dep in cc_proto_deps:
        proto_library_deps.append(dep + "_copy")
    native.proto_library(
        name = proto_library_name,
        srcs = srcs,
        deps = proto_library_deps,
    )

    go_proto_library(
        name = name,
        importpath = importpath,
        proto = ":" + proto_library_name,
        deps = go_proto_deps,
        compilers = ["@io_bazel_rules_go//proto:go_grpc"] if gen_oss_grpc else None,
    )

def ml_metadata_go_library(
        name,
        srcs = [],
        deps = [],
        importpath = None,
        cgo = None,
        cdeps = None):
    """Opensource go_library"""
    go_library(
        name = name,
        srcs = srcs,
        importpath = importpath,
        deps = deps,
        cgo = cgo,
        cdeps = cdeps,
    )

def ml_metadata_go_test(
        name,
        srcs = [],
        size = None,
        library = None,
        deps = []):
    """Opensource go_test"""
    go_test(
        name = name,
        size = size,
        srcs = srcs,
        embed = [library],
        deps = deps,
    )

# The rule builds a static cc library with the `libname` as target name,
# and `swigfile`_swig.cc as its srcs. In addition the rule builds a
# go_library in -cgo mode with `name` as the target name, `name`.go as its srcs
# and links to the `libname` with cgo dependency in `cdeps`.
# Note: the `swigfile`_swig.cc and `name`.go is auto-generated, and should be
#       provided when using the rule.
def ml_metadata_go_wrap_cc(
        name,
        swigfile = None,
        deps = [],
        libname = None,
        importpath = None):
    native.cc_library(
        name = libname,
        srcs = [swigfile + "_swig.cc"],
        linkstatic = 1,
        deps = deps,
    )

    ml_metadata_go_library(
        name = name,
        srcs = [name + ".go"],
        importpath = importpath,
        cgo = True,
        cdeps = [libname],
    )

# The rule builds a python extension module of the given `name` from the swig
# files listed in `srcs` and list of cpp dependencies in `deps`.
def ml_metadata_py_wrap_cc(
        name,
        srcs = [],
        swig_includes = [],
        deps = [],
        copts = [],
        version_script = None,
        **kwargs):
    # TODO(b/143236826) drop the tf OSS dependency. currently we still use tf
    # error code and status. When building wheels, we ping a specific tf
    # version to build the py extension via swigging c++ modules.
    tf_py_wrap_cc(
        name = name,
        srcs = srcs,
        swig_includes = swig_includes,
        deps = deps,
        copts = copts,
        version_script = version_script,
        **kwargs
    )
