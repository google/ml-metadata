/* Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// gRPC server binary, which supports methods to interact with ml.metadata store
// defined in third_party/ml_metadata/proto/metadata_store_service.proto.

#include <vector>

#include "gflags/gflags.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "absl/strings/str_cat.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/metadata_store_service_impl.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"

namespace {

// Creates SSL gRPC server credentials.
std::shared_ptr<::grpc::ServerCredentials> BuildSslServerCredentials(
    const ml_metadata::MetadataStoreServerConfig::SSLConfig& ssl_config) {
  ::grpc::SslServerCredentialsOptions ssl_ops;
  ssl_ops.client_certificate_request =
      ssl_config.client_verify()
          ? GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY
          : GRPC_SSL_DONT_REQUEST_CLIENT_CERTIFICATE;

  if (!ssl_config.custom_ca().empty())
    ssl_ops.pem_root_certs = ssl_config.custom_ca();

  ::grpc::SslServerCredentialsOptions::PemKeyCertPair keycert = {
      ssl_config.server_key(), ssl_config.server_cert()};
  ssl_ops.pem_key_cert_pairs.push_back(keycert);

  return ::grpc::SslServerCredentials(ssl_ops);
}

// Parses a comma separated list of gRPC channel arguments and adds the
// arguments to the gRPC service.
void AddGrpcChannelArgs(const string& channel_arguments_str,
                        ::grpc::ServerBuilder* builder) {
  const std::vector<string> channel_arguments =
      tensorflow::str_util::Split(channel_arguments_str, ",");
  for (const string& channel_argument : channel_arguments) {
    const std::vector<string> key_val =
        tensorflow::str_util::Split(channel_argument, "=");
    // gRPC accept arguments of two types, int and string. We will attempt to
    // parse each arg as int and pass it on as such if successful. Otherwise we
    // will pass it as a string. gRPC will log arguments that were not accepted.
    tensorflow::int32 value;
    if (tensorflow::strings::safe_strto32(key_val[1], &value)) {
      builder->AddChannelArgument(key_val[0], value);
    } else {
      builder->AddChannelArgument(key_val[0], key_val[1]);
    }
  }
}

// Parses an ascii MetadataStoreServerConfig protobuf from 'file'.
ml_metadata::MetadataStoreServerConfig ParseMetadataStoreServerConfig(
    const string& file) {
  ml_metadata::MetadataStoreServerConfig server_config;
  TF_CHECK_OK(tensorflow::ReadTextProto(tensorflow::Env::Default(), file,
                                        &server_config));
  return server_config;
}

}  // namespace

// gRPC server options
DEFINE_int32(grpc_port, 8080, "Port to listen on for gRPC API. (default 8080)");
DEFINE_string(grpc_channel_arguments, "",
              "A comma separated list of arguments to be passed to the grpc "
              "server. (e.g. grpc.max_connection_age_ms=2000)");

// metadata store server options
DEFINE_string(metadata_store_server_config_file, "",
              "If non-empty, read an ascii MetadataStoreServerConfig protobuf "
              "from the file name to connect to the specified metadata source "
              "and set up a secure gRPC channel");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_grpc_port <= 0) {
    LOG(ERROR) << "grpc_port is invalid: " << FLAGS_grpc_port;
    return -1;
  }

  ml_metadata::MetadataStoreServerConfig server_config;
  ml_metadata::ConnectionConfig connection_config;
  // try to create the server config from the given file
  if (!FLAGS_metadata_store_server_config_file.empty()) {
    server_config =
        ParseMetadataStoreServerConfig(FLAGS_metadata_store_server_config_file);
  }

  if (server_config.has_connection_config()) {
    connection_config = server_config.connection_config();
  } else {
    LOG(WARNING) << "The connection_config is not given. Using in memory fake "
                    "database, any metadata will not be persistent";
    connection_config.mutable_fake_database();
  }

  std::unique_ptr<ml_metadata::MetadataStore> metadata_store;
  TF_CHECK_OK(
      ml_metadata::CreateMetadataStore(connection_config, &metadata_store))
      << "MetadataStore cannot be created with the given connection config.";

  ml_metadata::MetadataStoreServiceImpl metadata_store_service(
      std::move(metadata_store));

  const string server_address = absl::StrCat("0.0.0.0:", FLAGS_grpc_port);
  ::grpc::ServerBuilder builder;

  std::shared_ptr<::grpc::ServerCredentials> credentials =
      ::grpc::InsecureServerCredentials();
  if (server_config.has_ssl_config()) {
    credentials = BuildSslServerCredentials(server_config.ssl_config());
  }

  builder.AddListeningPort(server_address, credentials);
  AddGrpcChannelArgs(FLAGS_grpc_channel_arguments, &builder);
  builder.RegisterService(&metadata_store_service);
  std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address;

  // keep the program running until the server shuts down.
  server->Wait();

  return 0;
}
