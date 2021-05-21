# Glog has it's own BUILD file, but it enables gflags by default.
# Rather than add another dependency, this BUILD file is exactly
# the same but without gflags.

licenses(["notice"])

load(":bazel/glog.bzl", "glog_library")

glog_library(with_gflags = 0)
