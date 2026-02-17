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

# Custom provider for descriptor proto files
ProtoDescriptorInfo = provider(
    fields = {
        "direct_sources": "Direct proto source files",
        "transitive_sources": "Transitive proto source files",
    }
)

# Helper rule to make proto files available without compilation
def _proto_descriptor_impl(ctx):
    proto_sources = depset(direct = ctx.files.srcs)
    return [
        DefaultInfo(files = proto_sources),
        ProtoDescriptorInfo(
            direct_sources = ctx.files.srcs,
            transitive_sources = proto_sources,
        )
    ]

proto_descriptor = rule(
    implementation = _proto_descriptor_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".proto"]),
    },
)

def _py_proto_library_impl(ctx):
    proto_deps = ctx.attr.deps
    use_grpc = ctx.attr.use_grpc_plugin

    all_sources = []
    py_infos = []
    all_proto_infos = []

    for dep in proto_deps:
        if ProtoInfo in dep:
            all_sources.extend(dep[ProtoInfo].direct_sources)
            all_proto_infos.append(dep[ProtoInfo])
        elif ProtoDescriptorInfo in dep:
            # Handle proto_descriptor custom provider
            all_sources.extend(dep[ProtoDescriptorInfo].direct_sources)
        elif PyInfo in dep:
            py_infos.append(dep[PyInfo])

    workspace_sources = []
    for src in all_sources:
        if not src.short_path.startswith("external/") and not src.short_path.startswith("../"):
            workspace_sources.append(src)

    py_outputs = []
    for proto_src in workspace_sources:
        basename = proto_src.basename[:-6]
        py_outputs.append(ctx.actions.declare_file(basename + "_pb2.py"))
        # Add grpc output file if grpc plugin is enabled
        if use_grpc:
            py_outputs.append(ctx.actions.declare_file(basename + "_pb2_grpc.py"))

    if py_outputs:
        proto_path_args = ["--proto_path=."]
        proto_paths = {".": True}

        for ws in workspace_sources:
            ws_dir = "/".join(ws.short_path.split("/")[:-1])
            if ws_dir and ws_dir not in proto_paths:
                proto_paths[ws_dir] = True
                proto_path_args.append("--proto_path=" + ws_dir)

        # Extract proto paths from dependencies
        for proto_info in all_proto_infos:
            # Extract _virtual_imports paths from direct and transitive sources
            for src in proto_info.direct_sources:
                src_path = src.path
                if "_virtual_imports" in src_path:
                    # Extract path up to and including _virtual_imports/XXX
                    parts = src_path.split("/_virtual_imports/")
                    if len(parts) == 2:
                        virtual_import_path = parts[0] + "/_virtual_imports/" + parts[1].split("/")[0]
                        if virtual_import_path not in proto_paths:
                            proto_paths[virtual_import_path] = True
                            proto_path_args.append("--proto_path=" + virtual_import_path)

            # Also process transitive sources
            for src in proto_info.transitive_sources.to_list():
                src_path = src.path
                if "_virtual_imports" in src_path:
                    # Extract path up to and including _virtual_imports/XXX
                    parts = src_path.split("/_virtual_imports/")
                    if len(parts) == 2:
                        virtual_import_path = parts[0] + "/_virtual_imports/" + parts[1].split("/")[0]
                        if virtual_import_path not in proto_paths:
                            proto_paths[virtual_import_path] = True
                            proto_path_args.append("--proto_path=" + virtual_import_path)

        proto_file_args = [src.short_path for src in workspace_sources]

        # Build protoc arguments
        protoc_args = ["--python_out=" + ctx.bin_dir.path]

        # Add grpc plugin if enabled
        tools = []
        if use_grpc and ctx.executable._grpc_plugin:
            protoc_args.append("--grpc_python_out=" + ctx.bin_dir.path)
            protoc_args.append("--plugin=protoc-gen-grpc_python=" + ctx.executable._grpc_plugin.path)
            tools.append(ctx.executable._grpc_plugin)

        ctx.actions.run(
            inputs = depset(
                direct = workspace_sources,
                transitive = [
                    dep[ProtoInfo].transitive_sources
                    for dep in proto_deps
                    if ProtoInfo in dep
                ]
            ),
            outputs = py_outputs,
            executable = ctx.executable._protoc,
            arguments = protoc_args + proto_path_args + proto_file_args,
            tools = tools,
            mnemonic = "ProtocPython",
        )

    all_transitive_sources = [depset(py_outputs)]
    all_imports = [depset([ctx.bin_dir.path])] if py_outputs else []

    for py_info in py_infos:
        all_transitive_sources.append(py_info.transitive_sources)
        if hasattr(py_info, "imports"):
            all_imports.append(py_info.imports)

    return [
        DefaultInfo(files = depset(py_outputs)),
        PyInfo(
            transitive_sources = depset(transitive = all_transitive_sources),
            imports = depset(transitive = all_imports),
            has_py2_only_sources = False,
            has_py3_only_sources = True,
        ),
    ]

_py_proto_library_rule = rule(
    implementation = _py_proto_library_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [[ProtoInfo], [PyInfo]],
        ),
        "use_grpc_plugin": attr.bool(
            default = False,
            doc = "Whether to use the gRPC plugin to generate service stubs",
        ),
        "_protoc": attr.label(
            default = "@com_google_protobuf//:protoc",
            executable = True,
            cfg = "exec",
        ),
        "_grpc_plugin": attr.label(
            default = "@com_github_grpc_grpc//src/compiler:grpc_python_plugin",
            executable = True,
            cfg = "exec",
        ),
    },
    provides = [PyInfo],
)

# Wrapper for cc_proto_library to maintain compatibility with Protobuf 4.x.
def cc_proto_library(
        name,
        srcs = [],
        deps = [],
        cc_libs = [],
        protoc = None,
        default_runtime = None,
        use_grpc_plugin = None,
        testonly = 0,
        visibility = None,
        **kwargs):
    _ignore = [cc_libs, protoc, default_runtime, use_grpc_plugin, kwargs]

    native.proto_library(
        name = name + "_proto",
        srcs = srcs,
        deps = [d + "_proto" if not d.startswith("@") else d for d in deps],
        testonly = testonly,
        visibility = visibility,
    )

    native.cc_proto_library(
        name = name,
        deps = [":" + name + "_proto"],
        testonly = testonly,
        visibility = visibility,
    )

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
        cc_grpc_version = None):
    """Opensource cc_proto_library."""
    _ignore = [has_services]
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
    _ignore = [api_version, srcs]
    if not proto_library:
        fail("proto_library parameter is required for ml_metadata_proto_library_py")

    actual_proto_library = ":" + proto_library + "_proto"

    _py_proto_library_rule(
        name = name,
        deps = [actual_proto_library] + deps + oss_deps,
        use_grpc_plugin = use_grpc_plugin,
        visibility = visibility,
        testonly = testonly,
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

    # For macOS, only export PyInit_* (Python 3)
    # macOS linker requires all exported symbols to exist
    exported_symbols_macos = [
        "PyInit_%s" % sname,
    ]

    # For Linux, include Python 2 symbols for compatibility
    # (version script allows undefined symbols)
    exported_symbols_linux = [
        "init%s" % sname,
        "init_%s" % sname,
        "PyInit_%s" % sname,
    ]

    exported_symbols_file = "%s-exported-symbols.lds" % name
    version_script_file = "%s-version-script.lds" % name

    exported_symbols_output = "\n".join(["_%s" % symbol for symbol in exported_symbols_macos])
    version_script_output = "\n".join([" %s;" % symbol for symbol in exported_symbols_linux])

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
