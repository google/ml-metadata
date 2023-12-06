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

load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")
load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")

def ml_metadata_cc_test(
        name,
        srcs = [],
        deps = [],
        env = {},
        tags = [],
        args = [],
        size = None,
        data = None):
    _ignore = [data]
    native.cc_test(
        name = name,
        srcs = srcs,
        deps = deps,
        env = env,
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
        deps = ["@com_google_protobuf//:well_known_types_py_pb2"] + deps + oss_deps,
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

# The rule builds a pybind11 extension.
def ml_metadata_pybind_extension(
        name,
        srcs,
        module_name,
        deps = [],
        visibility = None):
    """Builds a pybind1 py_extension module.

    Args:
      name: Name of the target.
      srcs: C++ source files.
      module_name: Ignored.
      deps: Dependencies.
      visibility: Visibility.
    """
    _ignore = [module_name]
    p = name.rfind("/")
    if p == -1:
        sname = name
        prefix = ""
    else:
        sname = name[p + 1:]
        prefix = name[:p + 1]
    so_file = "%s%s.so" % (prefix, sname)
    pyd_file = "%s%s.pyd" % (prefix, sname)
    exported_symbols = [
        "init%s" % sname,
        "init_%s" % sname,
        "PyInit_%s" % sname,
    ]

    exported_symbols_file = "%s-exported-symbols.lds" % name
    version_script_file = "%s-version-script.lds" % name

    exported_symbols_output = "\n".join(["_%s" % symbol for symbol in exported_symbols])
    version_script_output = "\n".join([" %s;" % symbol for symbol in exported_symbols])

    native.genrule(
        name = name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '%s' >$@" % exported_symbols_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
    )

    native.genrule(
        name = name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n%s\n local: *;};' >$@" % version_script_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
    )

    native.cc_binary(
        name = so_file,
        srcs = srcs,
        copts = [
            "-fno-strict-aliasing",
            "-fexceptions",
        ] + select({
            "//conditions:default": [
                "-fvisibility=hidden",
            ],
        }),
        linkopts = select({
            "//ml_metadata:macos": [
                # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                # not being exported.  There should be a better way to deal with this.
                "-Wl,-w",
                "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
            ],
            "//conditions:default": [
                "-Wl,--version-script",
                "$(location %s)" % version_script_file,
            ],
        }),
        deps = deps + [
            exported_symbols_file,
            version_script_file,
        ],
        features = ["-use_header_modules"],
        linkshared = 1,
        visibility = visibility,
    )
    native.genrule(
        name = name + "_pyd_copy",
        srcs = [so_file],
        outs = [pyd_file],
        cmd = "cp $< $@",
        output_to_bindir = True,
        visibility = visibility,
    )
    native.py_library(
        name = name,
        data = select({
            "//conditions:default": [so_file],
        }),
        srcs_version = "PY3",
        visibility = visibility,
    )
