# Copyright 2019 Google LLC
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


# Note that this Bazel version should be the same as the one specified in the
# WORKSPACE file.
FROM l.gcr.io/google/bazel:0.24.1 as builder

RUN apt-get update -y --option Acquire::Retries=3 && \
  apt-get install --no-install-recommends -y -q --option Acquire::Retries=3 \
  clang \
  cmake \
  make \
  musl-dev \
  openssl

ADD . /mlmd-src
WORKDIR /mlmd-src

RUN bazel build -c opt --action_env=PATH \
  --define=grpc_no_ares=true \
  //ml_metadata/metadata_store:metadata_store_server

# copying libmysqlclient source onto THIRD_PARTY folder.
RUN mkdir -p /mlmd-src/third_party
RUN cp -RL /mlmd-src/bazel-mlmd-src/external/libmysqlclient /mlmd-src/third_party/mariadb-connector-c

FROM ubuntu:18.04

COPY --from=builder /mlmd-src/bazel-bin/ml_metadata/metadata_store/metadata_store_server /bin/metadata_store_server
COPY --from=builder /mlmd-src/third_party /mlmd-src/third_party

ENV GRPC_PORT "8080"
ENV METADATA_STORE_SERVER_CONFIG_FILE ""

ENTRYPOINT \
  "/bin/metadata_store_server" \
  "--grpc_port=${GRPC_PORT}" \
  "--metadata_store_server_config_file=${METADATA_STORE_SERVER_CONFIG_FILE}"
