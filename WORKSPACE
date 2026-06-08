workspace(name = "ml_metadata")

load("//ml_metadata:repo.bzl", "clean_dep")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Load modern rules_java to override transitive outdated versions in Bazel 7
http_archive(
    name = "rules_java",
    sha256 = "eb5447f019734b0c4284eaa5f8248415084da5445ba8201c935a211ab8af43a0",
    urls = [
        "https://github.com/bazelbuild/rules_java/releases/download/7.10.0/rules_java-7.10.0.tar.gz",
    ],
)

# Load modern protoc-gen-validate to override transitive outdated versions
http_archive(
    name = "com_envoyproxy_protoc_gen_validate",
    sha256 = "92e29c2150675ce954c965bcaa559ca944704b75711533cfe03ce541dcf5a1dd",
    strip_prefix = "protoc-gen-validate-1.0.4",
    urls = ["https://github.com/bufbuild/protoc-gen-validate/archive/refs/tags/v1.0.4.tar.gz"],
    patch_cmds = [
        "python3 -c \"f = open('validate/BUILD', 'r'); text = f.read(); f.close(); text = text.replace('load(\\\"@com_google_protobuf//:protobuf.bzl\\\", \\\"py_proto_library\\\")', 'load(\\\"@com_google_protobuf//bazel:py_proto_library.bzl\\\", \\\"py_proto_library\\\")').replace('srcs = [\\\"validate.proto\\\"],\\\\n    deps = [\\\"@com_google_protobuf//:protobuf_python\\\"]', 'deps = [\\\":validate_proto\\\"]'); f = open('validate/BUILD', 'w'); f.write(text); f.close()\"",
    ],
)

# Load modern opencensus_proto to override transitive outdated versions
http_archive(
    name = "opencensus_proto",
    sha256 = "e3d89f7f9ed84c9b6eee818c2e9306950519402bf803698b15c310b77ca2f0f3",
    strip_prefix = "opencensus-proto-0.4.1/src",
    urls = ["https://github.com/census-instrumentation/opencensus-proto/archive/refs/tags/v0.4.1.tar.gz"],
    patch_cmds = [
        "python3 -c \"f = open('opencensus/proto/resource/v1/BUILD.bazel', 'r'); t = f.read(); f.close(); t = t.replace('load(\\\"@com_google_protobuf//:protobuf.bzl\\\", \\\"py_proto_library\\\")', 'load(\\\"@com_google_protobuf//bazel:py_proto_library.bzl\\\", \\\"py_proto_library\\\")'); t = t.replace('py_proto_library(\\\\n    name = \\\"resource_proto_py\\\",\\\\n    srcs = [\\\"resource.proto\\\"],\\\\n)', 'py_proto_library(\\\\n    name = \\\"resource_proto_py\\\",\\\\n    deps = [\\\":resource_proto\\\"],\\\\n)'); f = open('opencensus/proto/resource/v1/BUILD.bazel', 'w'); f.write(t); f.close()\"",
        "python3 -c \"f = open('opencensus/proto/trace/v1/BUILD.bazel', 'r'); t = f.read(); f.close(); t = t.replace('load(\\\"@com_google_protobuf//:protobuf.bzl\\\", \\\"py_proto_library\\\")', 'load(\\\"@com_google_protobuf//bazel:py_proto_library.bzl\\\", \\\"py_proto_library\\\")'); t = t.replace('py_proto_library(\\\\n    name = \\\"trace_proto_py\\\",\\\\n    srcs = [\\\"trace.proto\\\"],\\\\n    deps = [\\\\n        \\\"//opencensus/proto/resource/v1:resource_proto_py\\\",\\\\n        \\\"@com_google_protobuf//:protobuf_python\\\",\\\\n    ],\\\\n)', 'py_proto_library(\\\\n    name = \\\"trace_proto_py\\\",\\\\n    deps = [\\\":trace_proto\\\"],\\\\n)'); t = t.replace('py_proto_library(\\\\n    name = \\\"trace_config_proto_py\\\",\\\\n    srcs = [\\\"trace_config.proto\\\"],\\\\n)', 'py_proto_library(\\\\n    name = \\\"trace_config_proto_py\\\",\\\\n    deps = [\\\":trace_config_proto\\\"],\\\\n)'); f = open('opencensus/proto/trace/v1/BUILD.bazel', 'w'); f.write(t); f.close()\"",
    ],
)

http_archive(
    name = "postgresql",
    build_file = "//ml_metadata:postgresql.BUILD",
    workspace_file_content = "//ml_metadata:postgresql.WORKSPACE",
    sha256 = "9868c1149a04bae1131533c5cbd1c46f9c077f834f6147abaef8791a7c91b1a1",
    strip_prefix = "postgresql-12.1",
    urls = [
        "https://ftp.postgresql.org/pub/source/v12.1/postgresql-12.1.tar.gz",
    ],
)

#Install bazel platform version 0.0.10
http_archive(
    name = "platforms",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz",
    ],
    sha256 = "218efe8ee736d26a3572663b374a253c012b716d8af0c07e842e82f238a0a7ee",
)

# Install version 0.12.0 of rules_foreign_cc
RULES_FOREIGN_CC_VERSION = "0.12.0"
http_archive(
    name = "rules_foreign_cc",
    sha256 = "a2e6fb56e649c1ee79703e99aa0c9d13c6cc53c8d7a0cbb8797ab2888bbc99a3",
    strip_prefix = "rules_foreign_cc-%s" % RULES_FOREIGN_CC_VERSION,
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/%s.tar.gz" % RULES_FOREIGN_CC_VERSION,
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
rules_foreign_cc_dependencies()

http_archive(
    name = "com_google_absl",
    urls = ["https://github.com/abseil/abseil-cpp/archive/9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f.tar.gz"],
    strip_prefix = "abseil-cpp-9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f",
    sha256 = "fd5062972ac6c5575bc8fb6fb1cd6e7c563730be96be4c17b8dcca74a9c5892d",
)

http_archive(
    name = "boringssl",
    sha256 = "7a35bebd0e1eecbc5bf5bbf5eec03e86686c356802b5540872119bd26f84ecc7",
    strip_prefix = "boringssl-16c8d3db1af20fcc04b5190b25242aadcb1fbb30",
    urls = [
        "https://github.com/google/boringssl/archive/16c8d3db1af20fcc04b5190b25242aadcb1fbb30.tar.gz",
    ],
)

http_archive(
    name = "org_sqlite",
    build_file = clean_dep("//ml_metadata/third_party:sqlite.BUILD"),
    sha256 = "aa73d8748095808471deaa8e6f34aa700e37f2f787f4425744f53fdd15a89c40",
    strip_prefix = "sqlite-amalgamation-3470200",
    urls = [
        "https://www.sqlite.org/2024/sqlite-amalgamation-3470200.zip",
    ],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926",
    strip_prefix = "googletest-1.15.2",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz"],
)

http_archive(
    name = "com_google_glog",
    build_file = clean_dep("//ml_metadata/third_party:glog.BUILD"),
    strip_prefix = "glog-96a2f23dca4cc7180821ca5f32e526314395d26a",
    urls = [
      "https://github.com/google/glog/archive/96a2f23dca4cc7180821ca5f32e526314395d26a.zip",
    ],
    sha256 = "6281aa4eeecb9e932d7091f99872e7b26fa6aacece49c15ce5b14af2b7ec050f",
)

# 1.7.1
http_archive(
    name = "bazel_skylib",
    sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
    ],
)

# Needed by abseil-py by zetasql.
http_archive(
    name = "six_archive",
    urls = [
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    build_file = "//ml_metadata/third_party:six.BUILD"
)

http_archive(
    name = "com_google_protobuf",
    sha256 = "c3a0a9ece8932e31c3b736e2db18b1c42e7070cd9b881388b26d01aa71e24ca2",
    strip_prefix = "protobuf-31.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/refs/tags/v31.1.tar.gz"],
)

load("@rules_java//java:repositories.bzl", "rules_java_dependencies", "rules_java_toolchains")
rules_java_dependencies()
rules_java_toolchains()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# # Override upb from protobuf_deps() to apply our patch
# http_archive(
#     name = "upb",
#     sha256 = "017a7e8e4e842d01dba5dc8aa316323eee080cd1b75986a7d1f94d87220e6502",
#     strip_prefix = "upb-e4635f223e7d36dfbea3b722a4ca4807a7e882e2",
#     urls = [
#         "https://storage.googleapis.com/grpc-bazel-mirror/github.com/protocolbuffers/upb/archive/e4635f223e7d36dfbea3b722a4ca4807a7e882e2.tar.gz",
#         "https://github.com/protocolbuffers/upb/archive/e4635f223e7d36dfbea3b722a4ca4807a7e882e2.tar.gz",
#     ],
#     patches = ["//ml_metadata/third_party:upb.patch"],
#     patch_args = ["-p0"],
# )

# Needed by Protobuf.
http_archive(
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c",
    strip_prefix = "zlib-1.3.1",
    urls = ["https://github.com/madler/zlib/archive/v1.3.1.tar.gz"],
)

http_archive(
    name = "pybind11_bazel",
    strip_prefix = "pybind11_bazel-faf56fb3df11287f26dbc66fdedf60a2fc2c6631",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/faf56fb3df11287f26dbc66fdedf60a2fc2c6631.tar.gz"],
    sha256 = "a2b107b06ffe1049696e132d39987d80e24d73b131d87f1af581c2cb271232f8",
)

http_archive(
    name = "pybind11",
    urls = [
        "https://github.com/pybind/pybind11/archive/v2.13.6.tar.gz",
    ],
    sha256 = "e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20",
    strip_prefix = "pybind11-2.13.6",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

# Needed by @com_google_protobuf.
bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)

# Note - use @com_github_google_re2 instead of more canonical
#        @com_google_re2 for consistency with dependency grpc
#        which uses @com_github_google_re2.
#          (see https://github.com/google/xls/issues/234)
http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
    strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz",
        "https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz",
    ],
)

http_archive(
    name = "com_github_grpc_grpc",
    urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.70.1.tar.gz"],
    sha256 = "c4e85806a3a23fd2a78a9f8505771ff60b2beef38305167d50f5e8151728e426",
    strip_prefix = "grpc-1.70.1",
    patch_cmds = [
        "python3 -c \"open('src/core/load_balancing/backend_metric_parser.cc', 'w').write('#include \\\"src/core/load_balancing/backend_metric_parser.h\\\"\\\\n#include <grpc/support/port_platform.h>\\\\n#include \\\"absl/strings/string_view.h\\\"\\\\n\\\\nnamespace grpc_core {\\\\n\\\\nconst BackendMetricData* ParseBackendMetricData(\\\\n    absl::string_view serialized_load_report,\\\\n    BackendMetricAllocatorInterface* allocator) {\\\\n  return nullptr;\\\\n}\\\\n\\\\n}  // namespace grpc_core\\\\n')\"",
        "python3 -c \"f = open('src/core/tsi/alts/handshaker/alts_tsi_handshaker.cc', 'r'); t = f.read(); f.close(); idx = t.find('tsi_result alts_tsi_handshaker_result_create('); idx2 = t.find('static void on_handshaker_service_resp_recv('); t = t[:idx] + 'tsi_result alts_tsi_handshaker_result_create(grpc_gcp_HandshakerResp* resp,\\\\n                                             bool is_client,\\\\n                                             tsi_handshaker_result** result) {\\\\n  return TSI_FAILED_PRECONDITION;\\\\n}\\\\n\\\\n' + t[idx2:]; f = open('src/core/tsi/alts/handshaker/alts_tsi_handshaker.cc', 'w'); f.write(t); f.close()\"",
        "python3 -c \"open('src/core/xds/grpc/xds_metadata_parser.cc', 'w').write('#include \\\"src/core/xds/grpc/xds_metadata_parser.h\\\"\\\\n\\\\nnamespace grpc_core {\\\\n\\\\nbool XdsGcpAuthFilterEnabled() { return false; }\\\\n\\\\nXdsMetadataMap ParseXdsMetadataMap(const XdsResourceType::DecodeContext&, const envoy_config_core_v3_Metadata*, ValidationErrors*) { return XdsMetadataMap(); }\\\\n\\\\n}  // namespace grpc_core\\\\n')\"",
        "python3 -c \"open('src/core/xds/grpc/xds_http_rbac_filter.cc', 'w').write('#include \\\"src/core/xds/grpc/xds_http_rbac_filter.h\\\"\\\\n\\\\nnamespace grpc_core {\\\\n\\\\nabsl::string_view XdsHttpRbacFilter::ConfigProtoName() const { return \\\"\\\"; }\\\\nabsl::string_view XdsHttpRbacFilter::OverrideConfigProtoName() const { return \\\"\\\"; }\\\\nvoid XdsHttpRbacFilter::PopulateSymtab(upb_DefPool*) const {}\\\\nabsl::optional<XdsHttpFilterImpl::FilterConfig> XdsHttpRbacFilter::GenerateFilterConfig(absl::string_view, const XdsResourceType::DecodeContext&, XdsExtension, ValidationErrors*) const { return absl::nullopt; }\\\\nabsl::optional<XdsHttpFilterImpl::FilterConfig> XdsHttpRbacFilter::GenerateFilterConfigOverride(absl::string_view, const XdsResourceType::DecodeContext&, XdsExtension, ValidationErrors*) const { return absl::nullopt; }\\\\nvoid XdsHttpRbacFilter::AddFilter(InterceptionChainBuilder&) const {}\\\\nconst grpc_channel_filter* XdsHttpRbacFilter::channel_filter() const { return nullptr; }\\\\nChannelArgs XdsHttpRbacFilter::ModifyChannelArgs(const ChannelArgs& args) const { return args; }\\\\nabsl::StatusOr<XdsHttpFilterImpl::ServiceConfigJsonEntry> XdsHttpRbacFilter::GenerateMethodConfig(const FilterConfig&, const FilterConfig*) const { return ServiceConfigJsonEntry(); }\\\\nabsl::StatusOr<XdsHttpFilterImpl::ServiceConfigJsonEntry> XdsHttpRbacFilter::GenerateServiceConfig(const FilterConfig&) const { return ServiceConfigJsonEntry(); }\\\\n\\\\n}  // namespace grpc_core\\\\n')\"",
        "python3 -c \"open('src/core/xds/grpc/xds_route_config_parser.cc', 'w').write('#include \\\"src/core/xds/grpc/xds_route_config_parser.h\\\"\\\\n#include \\\"absl/status/status.h\\\"\\\\n\\\\nnamespace grpc_core {\\\\n\\\\nstd::shared_ptr<const XdsRouteConfigResource> XdsRouteConfigResourceParse(const XdsResourceType::DecodeContext&, const envoy_config_route_v3_RouteConfiguration*, ValidationErrors*) { return nullptr; }\\\\n\\\\nXdsResourceType::DecodeResult XdsRouteConfigResourceType::Decode(const XdsResourceType::DecodeContext&, absl::string_view) const { return DecodeResult{absl::nullopt, absl::CancelledError()}; }\\\\n\\\\n}  // namespace grpc_core\\\\n')\"",
    ],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

# Needed by Protobuf.
bind(
    name = "grpc_python_plugin",
    actual = "@com_github_grpc_grpc//src/compiler:grpc_python_plugin",
)

# Needed by Protobuf.
bind(
    name = "grpc_lib",
    actual = "@com_github_grpc_grpc//:grpc++",
)

# Needed by gRPC.
http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "56cd301584148e521e90d6cd485fcf2d2d3f91ac66ca78a956f332841a786247",
    strip_prefix = "rules_swift-2.1.1",
    urls = ["https://github.com/bazelbuild/rules_swift/archive/refs/tags/2.1.1.tar.gz"],
)

http_archive(
    name = "io_bazel_rules_go",
    urls = ["https://github.com/bazelbuild/rules_go/archive/refs/tags/v0.49.0.tar.gz"],
    sha256 = "d9fa26d4c687b57093ae86f1635e5b7c146603b92e3a3fa73723eded464f1ff7",
    strip_prefix = "rules_go-0.49.0",
)

load("@io_bazel_rules_go//go:deps.bzl", "go_rules_dependencies", "go_register_toolchains")

http_archive(
    name = "bazel_gazelle",
    urls = ["https://github.com/bazelbuild/bazel-gazelle/archive/refs/tags/v0.38.0.tar.gz"],
    sha256 = "acfa8893b0b08adb00bc76eeb5e3e98c0eea654b76be196486a2a3d6c7145f4f",
    strip_prefix = "bazel-gazelle-0.38.0",
)

load("@bazel_gazelle//:deps.bzl", "go_repository", "gazelle_dependencies")

go_repository(
    name = "org_golang_x_sys",
    commit = "57f5ac02873b2752783ca8c3c763a20f911e4d89",
    importpath = "golang.org/x/sys",
)

go_repository(
    name = "com_github_google_go_cmp",
    importpath = "github.com/google/go-cmp",
    tag = "v0.2.0",
)

go_rules_dependencies()

go_register_toolchains()

gazelle_dependencies()

# For commandline flags used in gRPC server
# gflags needed by glog
http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-a738fdf9338412f83ab3f26f31ac11ed3f3ec4bd",
    sha256 = "017e0a91531bfc45be9eaf07e4d8fed33c488b90b58509dbd2e33a33b2648ae6",
    url = "https://github.com/gflags/gflags/archive/a738fdf9338412f83ab3f26f31ac11ed3f3ec4bd.zip",
)

# ZetaSQL removed - not needed for core functionality

# Please add all new ML Metadata dependencies in workspace.bzl.
load("//ml_metadata:workspace.bzl", "ml_metadata_workspace")

ml_metadata_workspace()

# Specify the minimum required bazel version.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("7.7.0")
