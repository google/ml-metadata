workspace(name = "ml_metadata")

load("//ml_metadata:repo.bzl", "mlmd_http_archive", "clean_dep")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# lts_20210324.2
http_archive(
    name = "com_google_absl",
    sha256 = "1764491a199eb9325b177126547f03d244f86b4ff28f16f206c7b3e7e4f777ec",
    strip_prefix = "abseil-cpp-278e0a071885a22dcd2fd1b5576cc44757299343",
    urls = [
        "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/278e0a071885a22dcd2fd1b5576cc44757299343.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/278e0a071885a22dcd2fd1b5576cc44757299343.tar.gz"
    ],
)

# rules_cc defines rules for generating C++ code from Protocol Buffers.
http_archive(
    name = "rules_cc",
    sha256 = "35f2fb4ea0b3e61ad64a369de284e4fbbdcdba71836a5555abb5e194cf119509",
    strip_prefix = "rules_cc-624b5d59dfb45672d4239422fa1e3de1822ee110",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/624b5d59dfb45672d4239422fa1e3de1822ee110.tar.gz",
        "https://github.com/bazelbuild/rules_cc/archive/624b5d59dfb45672d4239422fa1e3de1822ee110.tar.gz",
    ],
)

mlmd_http_archive(
    name = "boringssl",
    sha256 = "1188e29000013ed6517168600fc35a010d58c5d321846d6a6dfee74e4c788b45",
    strip_prefix = "boringssl-7f634429a04abc48e2eb041c81c5235816c96514",
    system_build_file = clean_dep("//ml_metadata/third_party/systemlibs:boringssl.BUILD"),
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
        "https://github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
    ],
)

mlmd_http_archive(
    name = "org_sqlite",
    build_file = clean_dep("//ml_metadata/third_party:sqlite.BUILD"),
    sha256 = "adf051d4c10781ea5cfabbbc4a2577b6ceca68590d23b58b8260a8e24cc5f081",
    strip_prefix = "sqlite-amalgamation-3300100",
    system_build_file = clean_dep("//ml_metadata/third_party/systemlibs:sqlite.BUILD"),
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/www.sqlite.org/2019/sqlite-amalgamation-3300100.zip",
        "https://www.sqlite.org/2019/sqlite-amalgamation-3300100.zip",
    ],
)

mlmd_http_archive(
    name = "com_google_googletest",
    sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
    strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
        "https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
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
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
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

PROTOBUF_COMMIT = "fde7cf7358ec7cd69e8db9be4f1fa6a5c431386a" # 3.13.0
http_archive(
    name = "com_google_protobuf",
    sha256 = "e589e39ef46fb2b3b476b3ca355bd324e5984cbdfac19f0e1625f0042e99c276",
    strip_prefix = "protobuf-%s" % PROTOBUF_COMMIT,
    urls = [
        "https://storage.googleapis.com/grpc-bazel-mirror/github.com/google/protobuf/archive/%s.tar.gz" % PROTOBUF_COMMIT,
        "https://github.com/google/protobuf/archive/%s.tar.gz" % PROTOBUF_COMMIT,
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# Needed by Protobuf.
http_archive(
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = ["https://zlib.net/zlib-1.2.11.tar.gz"],
)

# pybind11
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    strip_prefix = "pybind11-2.4.3",
    urls = ["https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz"],
)

# Bazel rules for pybind11
http_archive(
    name = "pybind11_bazel",
    strip_prefix = "pybind11_bazel-d5587e65fb8cbfc0015391a7616dc9c66f64a494",
    url = "https://github.com/pybind/pybind11_bazel/archive/d5587e65fb8cbfc0015391a7616dc9c66f64a494.zip",
    sha256 = "bf8e1f3ebde5ee37ad30c451377b03fbbe42b9d8f24c244aa8af2ccbaeca7e6c",
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

http_archive(
    name = "com_googlesource_code_re2",
    urls = [
        "https://github.com/google/re2/archive/d1394506654e0a19a92f3d8921e26f7c3f4de969.tar.gz",
    ],
    sha256 = "ac855fb93dfa6878f88bc1c399b9a2743fdfcb3dc24b94ea9a568a1c990b1212",
    strip_prefix = "re2-d1394506654e0a19a92f3d8921e26f7c3f4de969",
)

# gRPC. Official release 1.33.2. Name is required by Google APIs.
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "2060769f2d4b0d3535ba594b2ab614d7f68a492f786ab94b4318788d45e3278a",
    strip_prefix = "grpc-1.33.2",
    patches = ["//ml_metadata/third_party:grpc.patch"],
    urls = ["https://github.com/grpc/grpc/archive/v1.33.2.tar.gz"],
)
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

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
    sha256 = "d0833bc6dad817a367936a5f902a0c11318160b5e80a20ece35fb85a5675c886",
    strip_prefix = "rules_swift-3eeeb53cebda55b349d64c9fc144e18c5f7c0eb8",
    urls = ["https://github.com/bazelbuild/rules_swift/archive/3eeeb53cebda55b349d64c9fc144e18c5f7c0eb8.tar.gz"],
)

http_archive(
    name = "io_bazel_rules_go",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/rules_go/releases/download/v0.20.3/rules_go-v0.20.3.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.20.3/rules_go-v0.20.3.tar.gz",
    ],
    sha256 = "e88471aea3a3a4f19ec1310a55ba94772d087e9ce46e41ae38ecebe17935de7b",
)

load("@io_bazel_rules_go//go:deps.bzl", "go_rules_dependencies", "go_register_toolchains")

http_archive(
    name = "bazel_gazelle",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/bazel-gazelle/releases/download/v0.19.1/bazel-gazelle-v0.19.1.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.19.1/bazel-gazelle-v0.19.1.tar.gz",
    ],
    sha256 = "86c6d481b3f7aedc1d60c1c211c6f76da282ae197c3b3160f54bd3a8f847896f",
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
git_repository(
    name = "com_github_gflags_gflags",
    # v2.2.2
    commit = "e171aa2d15ed9eb17054558e0b3a6a413bb01067",
    remote = "https://github.com/gflags/gflags.git",
)

# BEGIN IFNDEF_WIN
ZETASQL_COMMIT = "5ccb05880e72ab9ff75dd6b05d7b0acce53f1ea2" # 04/22/2021  # windows
http_archive(  # windows
    name = "com_google_zetasql",  # windows
    urls = ["https://github.com/google/zetasql/archive/%s.zip" % ZETASQL_COMMIT],  # windows
    strip_prefix = "zetasql-%s" % ZETASQL_COMMIT,  # windows
    # patches = ["//ml_metadata/third_party:zetasql.patch"],  # windows
    sha256 = '4ca4e45f457926484822701ec15ca4d0172b01d7ce43c0b34c6f3ab98c95b241'  # windows
)  # windows

load("@com_google_zetasql//bazel:zetasql_deps_step_1.bzl", "zetasql_deps_step_1")  # windows
zetasql_deps_step_1()  # windows
load("@com_google_zetasql//bazel:zetasql_deps_step_2.bzl", "zetasql_deps_step_2")  # windows
zetasql_deps_step_2(  # windows
    analyzer_deps = True,  # windows
    evaluator_deps = True,  # windows
    tools_deps = False,  # windows
    java_deps = False,  # windows
    testing_deps = False)  # windows
# END IFNDEF_WIN


# Please add all new ML Metadata dependencies in workspace.bzl.
load("//ml_metadata:workspace.bzl", "ml_metadata_workspace")

ml_metadata_workspace()

# Specify the minimum required bazel version.
load("//ml_metadata:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("3.7.2")
