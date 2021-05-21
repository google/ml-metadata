workspace(name = "ml_metadata")

# To update TensorFlow to a new revision.
# 1. Update the 'git_commit' args below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.

load("//ml_metadata:repo.bzl", "mlmd_http_archive", "clean_dep")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# v1.15.2
# updated: 2020/09/10
_TENSORFLOW_GIT_COMMIT = "5d80e1e8e6ee999be7db39461e0e79c90403a2e4"

http_archive(
    name = "org_tensorflow",
    sha256 = "7e3c893995c221276e17ddbd3a1ff177593d00fc57805da56dcc30fdc4299632",
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
)

# Needed by tf_py_wrap_cc rule from Tensorflow.
# When upgrading tensorflow version, also check tensorflow/WORKSPACE for the
# version of this -- keep in sync.
http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

ABSL_COMMIT = "0f3bb466b868b523cf1dc9b2aaaed65c77b28862"  # lts_20200923.2
http_archive(
    name = "com_google_absl",
    urls = ["https://github.com/abseil/abseil-cpp/archive/%s.zip" % ABSL_COMMIT],
    sha256 = "9929f3662141bbb9c6c28accf68dcab34218c5ee2d83e6365d9cb2594b3f3171",
    strip_prefix = "abseil-cpp-%s" % ABSL_COMMIT,
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

# Please add all new ML Metadata dependencies in workspace.bzl.
load("//ml_metadata:workspace.bzl", "ml_metadata_workspace")

ml_metadata_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.24.1")
