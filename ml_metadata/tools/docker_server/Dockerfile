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

# TODO(b/195701120) Introduces l.gcr.io/google/bazel:5.3.0 when it is available
# and makes sure that it uses ubuntu 20.04 as base image. Currently the lastest
# version only supports bazel 3.5.0.
FROM ubuntu:20.04 as builder

RUN export DEBIAN_FRONTEND=noninteractive && \
  apt-get update -y --option Acquire::Retries=3 && \
  apt-get install --no-install-recommends -y -q --option Acquire::Retries=3 \
  build-essential \
  clang \
  cmake \
  make \
  musl-dev \
  openssl \
  curl \
  ca-certificates \
  unzip \
  software-properties-common \
  git \
  python3-distutils \
  python-is-python3 \
  python3-dev

# Set up Bazel 5.3.0
ENV BAZEL_VERSION 5.3.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

ADD . /mlmd-src
WORKDIR /mlmd-src

# "-std=c++17" is needed in order to build with ZetaSQL.
RUN bazel build -c opt --action_env=PATH \
  --define=grpc_no_ares=true \
  //ml_metadata/metadata_store:metadata_store_server --cxxopt="-std=c++17"

# copying libmysqlclient source onto THIRD_PARTY folder.
RUN mkdir -p /mlmd-src/third_party
RUN cp -RL /mlmd-src/bazel-mlmd-src/external/libmysqlclient /mlmd-src/third_party/mariadb-connector-c

FROM ubuntu:20.04

COPY --from=builder /mlmd-src/bazel-bin/ml_metadata/metadata_store/metadata_store_server /bin/metadata_store_server
COPY --from=builder /mlmd-src/third_party /mlmd-src/third_party

ENV GRPC_PORT "8080"
ENV METADATA_STORE_SERVER_CONFIG_FILE ""

# Introduces tzdata package here to avoid LoadTimeZone check failed error in the metadata store server.
RUN export DEBIAN_FRONTEND=noninteractive && \
  apt-get update -y --option Acquire::Retries=3 && \
  apt-get install --no-install-recommends -y -q --option Acquire::Retries=3 \
  tzdata

ENTRYPOINT \
  "/bin/metadata_store_server" \
  "--grpc_port=${GRPC_PORT}" \
  "--metadata_store_server_config_file=${METADATA_STORE_SERVER_CONFIG_FILE}"
