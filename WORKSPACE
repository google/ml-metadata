workspace(name = "ml_metadata")

# To update TensorFlow to a new revision.
# 1. Update the 'git_commit' args below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//ml_metadata:repo.bzl", "tensorflow_http_archive")

# v1.11.0
tensorflow_http_archive(
    name = "org_tensorflow",
    sha256 = "025b47263af34475dc75da40c76a87934a70f69611e9b0b88445d65730f0fc73",
    git_commit = "c19e29306ce1777456b2dbb3a14f511edf7883a8",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
    ],
)

git_repository(
    name = "protobuf_bzl",
    # v3.4.0
    commit = "80a37e0782d2d702d52234b62dd4b9ec74fd2c95",
    remote = "https://github.com/google/protobuf.git",
)

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

check_bazel_version_at_least("0.15.0")
