#!/bin/bash
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

# Convenience binary to build TFDV from source.

# Put wrapped c++ files in place

set -u -x

cp -f ml_metadata/metadata_store/pywrap_metadata_store_serialized.py \
  ${BUILD_WORKSPACE_DIRECTORY}/ml_metadata/metadata_store
cp -f ml_metadata/proto/metadata_store_pb2.py \
  ${BUILD_WORKSPACE_DIRECTORY}/ml_metadata/proto
cp -f ml_metadata/proto/metadata_store_service_pb2.py \
  ${BUILD_WORKSPACE_DIRECTORY}/ml_metadata/proto

cp -f ml_metadata/metadata_store/_pywrap_metadata_store_serialized.so \
  ${BUILD_WORKSPACE_DIRECTORY}/ml_metadata/metadata_store

# Create the wheel
cd ${BUILD_WORKSPACE_DIRECTORY}

python setup.py bdist_wheel

# Cleanup
cd -
