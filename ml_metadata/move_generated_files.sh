#!/bin/bash
# Copyright 2020 Google LLC
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

# Convenience binary to build MLMD from source.
# Put wrapped c++ files in place
# Should be run in the directory containing WORKSPACE file. (workspace root)

function _is_macos() {
  [[ "$(uname -s | tr 'A-Z' 'a-z')" =~ darwin ]]
}

function mlmd::move_generated_files() {
  set -eux

  # Newer bazel does not create bazel-genfiles any more (see
  # https://github.com/bazelbuild/bazel/issues/6761). It's merged with
  # bazel-bin after Bazel 0.25.0. Latest bazel doesn't provide bazel-genfiles
  # symlink anymore, so if the symlink directory doesnt' exist, use bazel-bin.
  local bazel_genfiles
  if [[ -d "${BUILD_WORKSPACE_DIRECTORY}/bazel-genfiles" ]]; then
    bazel_genfiles="bazel-genfiles"
  else
    bazel_genfiles="bazel-bin"
  fi

  cp -f ${BUILD_WORKSPACE_DIRECTORY}/${bazel_genfiles}/ml_metadata/proto/metadata_store_pb2.py \
    ${BUILD_WORKSPACE_DIRECTORY}/ml_metadata/proto
  cp -f ${BUILD_WORKSPACE_DIRECTORY}/${bazel_genfiles}/ml_metadata/proto/metadata_store_service_pb2.py \
    ${BUILD_WORKSPACE_DIRECTORY}/ml_metadata/proto
  cp -f ${BUILD_WORKSPACE_DIRECTORY}/${bazel_genfiles}/ml_metadata/proto/metadata_store_service_pb2_grpc.py \
    ${BUILD_WORKSPACE_DIRECTORY}/ml_metadata/proto
  cp -f ${BUILD_WORKSPACE_DIRECTORY}/${bazel_genfiles}/ml_metadata/simple_types/proto/simple_types_pb2.py \
    ${BUILD_WORKSPACE_DIRECTORY}/ml_metadata/simple_types/proto


  MLMD_EXTENSION="ml_metadata/metadata_store/pywrap/metadata_store_extension.so"
  cp -f ${BUILD_WORKSPACE_DIRECTORY}/bazel-bin/${MLMD_EXTENSION} \
      ${BUILD_WORKSPACE_DIRECTORY}/${MLMD_EXTENSION}

  chmod +w "${BUILD_WORKSPACE_DIRECTORY}/${MLMD_EXTENSION}"
}

mlmd::move_generated_files
