workspace(name = "ml_metadata")

load("//ml_metadata:repo.bzl", "clean_dep")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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


#Install bazel platform version 0.0.6
http_archive(
    name = "platforms",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.6/platforms-0.0.6.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/0.0.6/platforms-0.0.6.tar.gz",
    ],
    sha256 = "5308fc1d8865406a49427ba24a9ab53087f17f5266a7aabbfc28823f3916e1ca",
)

# Install version 0.9.0 of rules_foreign_cc, as default version causes an
# invalid escape sequence error to be raised, which can't be avoided with
# the --incompatible_restrict_string_escapes=false flag (flag was removed in
# Bazel 5.0).
RULES_FOREIGN_CC_VERSION = "0.9.0"
http_archive(
    name = "rules_foreign_cc",
    sha256 = "2a4d07cd64b0719b39a7c12218a3e507672b82a97b98c6a89d38565894cf7c51",
    strip_prefix = "rules_foreign_cc-%s" % RULES_FOREIGN_CC_VERSION,
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/%s.tar.gz" % RULES_FOREIGN_CC_VERSION,
    patch_tool = "patch",
    patches = ["//ml_metadata/third_party:rules_foreign_cc.patch",],
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
rules_foreign_cc_dependencies()

# lts_20230125.3
ABSL_COMMIT = "c2435f8342c2d0ed8101cb43adfd605fdc52dca2"
http_archive(
    name = "com_google_absl",
    sha256 = "9892836ab0d3f099b8c15076c6f4168144f452d097bd49da215fe0df36a2d48c",
    strip_prefix = "abseil-cpp-%s" % ABSL_COMMIT,
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/%s.tar.gz" % ABSL_COMMIT,
    ],
)

BORINGSSL_COMMIT = "adde128bd706c6caf26dd595e631871b09f40bf6"
http_archive(
    name = "boringssl",
    sha256 = "7fc7632ce455575025be2cac610a23377753b01a4932d94a452cf831a5b32e9e",
    strip_prefix = "boringssl-%s" % BORINGSSL_COMMIT,
    urls = [
        "https://github.com/google/boringssl/archive/%s.tar.gz" % BORINGSSL_COMMIT,
    ],
)

http_archive(
    name = "org_sqlite",
    build_file = clean_dep("//ml_metadata/third_party:sqlite.BUILD"),
    sha256 = "87775784f8b22d0d0f1d7811870d39feaa7896319c7c20b849a4181c5a50609b",
    strip_prefix = "sqlite-amalgamation-3390200",
    urls = [
        "https://www.sqlite.org/2022/sqlite-amalgamation-3390200.zip",
    ],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "b976cf4fd57b318afdb1bdb27fc708904b3e4bed482859eb94ba2b4bdd077fe2",
    strip_prefix = "googletest-f8d7d77c06936315286eb55f8de22cd23c188571",
    urls = [
        "https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip",
    ],
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

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)

# Needed by abseil-py by zetasql.
http_archive(
    name = "six_archive",
    urls = [
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.16.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.16.0.tar.gz",
    ],
    sha256 = "1e61c37477a1626458e36f7b1d82aa5c9b094fa4802892072e49de9c60c4c926",
    strip_prefix = "six-1.16.0",
    build_file = "//ml_metadata/third_party:six.BUILD"
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
        "https://github.com/pybind/pybind11/archive/v2.10.1.tar.gz",
    ],
    strip_prefix = "pybind11-2.10.1",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

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

# gRPC. Official release 1.50.0. Name is required by Google APIs.
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "76900ab068da86378395a8e125b5cc43dfae671e09ff6462ddfef18676e2165a",
    strip_prefix = "grpc-1.50.0",
    urls = ["https://github.com/grpc/grpc/archive/v1.50.0.tar.gz"],
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

# pulled in transitively by gRPC.
load("@io_bazel_rules_go//go:deps.bzl", "go_rules_dependencies", "go_register_toolchains")
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

ZETASQL_COMMIT = "ac37cf5c0d80b5605176fc0f29e87b12f00be693" # 08/10/2022
http_archive(
    name = "com_google_zetasql",
    urls = ["https://github.com/google/zetasql/archive/%s.zip" % ZETASQL_COMMIT],
    strip_prefix = "zetasql-%s" % ZETASQL_COMMIT,
    #patches = ["//ml_metadata/third_party:zetasql.patch"],
    sha256 = '651a768cd51627f58aa6de7039aba9ddab22f4b0450521169800555269447840'
)

load("@com_google_zetasql//bazel:zetasql_deps_step_1.bzl", "zetasql_deps_step_1")
zetasql_deps_step_1()
load("@com_google_zetasql//bazel:zetasql_deps_step_2.bzl", "zetasql_deps_step_2")
zetasql_deps_step_2(
    analyzer_deps = True,
    evaluator_deps = True,
    tools_deps = False,
    java_deps = False,
    testing_deps = False)

# This is part of what zetasql_deps_step_3() does.
load("@com_google_googleapis//:repository_rules.bzl", "switched_rules_by_language")
switched_rules_by_language(
    name = "com_google_googleapis_imports",
    cc = True,
)



# Please add all new ML Metadata dependencies in workspace.bzl.
load("//ml_metadata:workspace.bzl", "ml_metadata_workspace")

ml_metadata_workspace()

# Specify the minimum required bazel version.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("5.3.0")
