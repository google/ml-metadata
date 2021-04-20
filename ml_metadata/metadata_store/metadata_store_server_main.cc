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
#include "tensorflow/core/lib/core/status.h"
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

// Parses config file if provided and returns true if it is successful in
// populating service_config.
bool ParseMetadataStoreServerConfigOrDie(
    const std::string& filename,
    ml_metadata::MetadataStoreServerConfig* server_config) {
  if (filename.empty()) {
    return false;
  }

  TF_CHECK_OK(tensorflow::ReadTextProto(tensorflow::Env::Default(), filename,
                                        server_config));
  return true;
}

// Returns true if passed parameters were used to construct mysql connection
// config and set it to service_config. Returns false if host, port and database
// were set and dies if only some of them were provided.
bool ParseMySQLFlagsBasedServerConfigOrDie(
    const std::string& host, const int port, const std::string& database,
    const std::string& user, const std::string& password,
    const bool enable_database_upgrade, const int64 downgrade_db_schema_version,
    ml_metadata::MetadataStoreServerConfig* server_config) {
  if (host.empty() && database.empty() && port == 0) {
    return false;
  }

  CHECK(!host.empty() && !database.empty() && port > 0)
      << "To use mysql store, all of --mysql_config_host, "
         "--mysql_config_port, --mysql_config "
         "database needs to be provided";

  ml_metadata::ConnectionConfig* connection_config =
      server_config->mutable_connection_config();
  ml_metadata::MySQLDatabaseConfig* config = connection_config->mutable_mysql();
  config->set_host(host);
  config->set_port(port);
  config->set_database(database);
  config->set_user(user);
  config->set_password(password);

  CHECK(!enable_database_upgrade || downgrade_db_schema_version < 0)
      << "Both --enable_database_upgraded=true and downgrade_db_schema_version "
         ">= 0 cannot be set together. Only one of the flags needs to be set";

  if (enable_database_upgrade) {
    ml_metadata::MigrationOptions* migration_config =
        server_config->mutable_migration_options();
    migration_config->set_enable_upgrade_migration(enable_database_upgrade);
  }

  if (downgrade_db_schema_version >= 0) {
    ml_metadata::MigrationOptions* migration_config =
        server_config->mutable_migration_options();
    migration_config->set_downgrade_to_schema_version(
        downgrade_db_schema_version);
  }

  return true;
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
              "and set up a secure gRPC channel. If provided overrides the "
              "--mysql* configuration");
DEFINE_int32(
    metadata_store_connection_retries, 5,
    "The max number of retries when connecting to the given metadata source");

// MySQL config command line options
DEFINE_string(mysql_config_host, "",
              "The mysql hostname to use. If non-empty, works in conjunction "
              "with --mysql_config_port & "
              "--mysql_config_database, to provide MySQL configuration");
DEFINE_int32(mysql_config_port, 0,
             "The mysql port to use. If non-empty, works in conjunction with "
             "--mysql_config_host & "
             "--mysql_config_database, to provide mysql store configuration");
DEFINE_string(mysql_config_database, "",
              "The mysql database to use. If non-empty, works in conjunction "
              "with --mysql_config_host & "
              "--mysql_config_port, to provide MySQL configuration");
DEFINE_string(mysql_config_user, "",
              "The mysql user name to use (Optional parameter)");
DEFINE_string(mysql_config_password, "",
              "The mysql user password to use (Optional parameter)");
DEFINE_bool(
    enable_database_upgrade, false,
    "Flag specifying database upgrade option. If set to true, it enables "
    "database migration during initialization(Optional parameter");
DEFINE_int64(downgrade_db_schema_version, -1,
             "Database downgrade schema version value. If set the database "
             "schema version is downgraded to the set value during "
             "initialization(Optional Parameter)");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if ((FLAGS_grpc_port) <= 0) {
    LOG(ERROR) << "grpc_port is invalid: " << (FLAGS_grpc_port);
    return -1;
  }

  ml_metadata::MetadataStoreServerConfig server_config;
  ml_metadata::ConnectionConfig connection_config;

  if (!ParseMetadataStoreServerConfigOrDie(
          (FLAGS_metadata_store_server_config_file),
          &server_config) &&
      !ParseMySQLFlagsBasedServerConfigOrDie(
          (FLAGS_mysql_config_host),
          (FLAGS_mysql_config_port),
          (FLAGS_mysql_config_database),
          (FLAGS_mysql_config_user),
          (FLAGS_mysql_config_password),
          (FLAGS_enable_database_upgrade),
          (FLAGS_downgrade_db_schema_version), &server_config)) {
    LOG(WARNING) << "The connection_config is not given. Using in memory fake "
                    "database, any metadata will not be persistent";
    connection_config.mutable_fake_database();
  } else {
    connection_config = server_config.connection_config();
  }

  // Creates a metadata_store in the main thread and init schema if necessary.
  std::unique_ptr<ml_metadata::MetadataStore> metadata_store;
  tensorflow::Status status = ml_metadata::CreateMetadataStore(
      connection_config, server_config.migration_options(), &metadata_store);
  for (int i = 0; i < (FLAGS_metadata_store_connection_retries);
       i++) {
    if (status.ok() || !tensorflow::errors::IsAborted(status)) {
      break;
    }
    LOG(WARNING) << "Connection Aborted with error: " << status;
    LOG(INFO) << "Retry attempt " << i;
    status = ml_metadata::CreateMetadataStore(
        connection_config, server_config.migration_options(), &metadata_store);
  }
  TF_CHECK_OK(status)
      << "MetadataStore cannot be created with the given connection config.";
  // At this point, schema initialization and migration are done.
  metadata_store.reset();

  ml_metadata::MetadataStoreServiceImpl metadata_store_service(
      connection_config);

  const string server_address =
      absl::StrCat("0.0.0.0:", (FLAGS_grpc_port));
  ::grpc::ServerBuilder builder;

  std::shared_ptr<::grpc::ServerCredentials> credentials =
      ::grpc::InsecureServerCredentials();
  if (server_config.has_ssl_config()) {
    credentials = BuildSslServerCredentials(server_config.ssl_config());
  }

  builder.AddListeningPort(server_address, credentials);
  AddGrpcChannelArgs((FLAGS_grpc_channel_arguments), &builder);
  builder.RegisterService(&metadata_store_service);
  std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address;

  // keep the program running until the server shuts down.
  server->Wait();

  return 0;
}
