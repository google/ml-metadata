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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include <glog/logging.h>
#include "google/protobuf/text_format.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/metadata_store_service_impl.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

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
      absl::StrSplit(channel_arguments_str, ',', absl::SkipEmpty());
  for (const string& channel_argument : channel_arguments) {
    const std::vector<string> key_val =
        absl::StrSplit(channel_argument, '=', absl::SkipEmpty());
    // gRPC accept arguments of two types, int and string. We will attempt to
    // parse each arg as int and pass it on as such if successful. Otherwise we
    // will pass it as a string. gRPC will log arguments that were not accepted.
    int64_t value;
    if (absl::SimpleAtoi(key_val[1], &value)) {
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

  std::ifstream input_file_stream(std::string(filename).c_str());
  if (!input_file_stream) {
    return false;
  }

  google::protobuf::io::IstreamInputStream file_stream(&input_file_stream);
  if (!google::protobuf::TextFormat::Parse(&file_stream, server_config)) {
    return false;
  }
  return true;
}

// Returns true if passed parameters were used to construct mysql connection
// config and set it to service_config. Returns false if host, port and database
// were set and dies if only some of them were provided.
bool ParseMySQLFlagsBasedServerConfigOrDie(
    const std::string& host, const int port, const std::string& database,
    const std::string& user, const std::string& password,
    const std::string& sslcert, const std::string& sslkey,
    const std::string& sslrootcert, const std::string& sslcapath,
    const std::string& sslcipher, const bool verify_server_cert,
    const bool enable_database_upgrade,
    const int64_t downgrade_db_schema_version,
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
  bool has_ssl_config;
  if (!sslcert.empty()) {
    has_ssl_config = true;
    config->mutable_ssl_options()->set_cert(sslcert);
  }
  if (!sslkey.empty()) {
    has_ssl_config = true;
    config->mutable_ssl_options()->set_key(sslkey);
  }
  if (!sslrootcert.empty()) {
    has_ssl_config = true;
    config->mutable_ssl_options()->set_ca(sslrootcert);
  }
  if (!sslcapath.empty()) {
    has_ssl_config = true;
    config->mutable_ssl_options()->set_capath(sslcapath);
  }
  if (!sslcipher.empty()) {
    has_ssl_config = true;
    config->mutable_ssl_options()->set_cipher(sslcipher);
  }
  if (has_ssl_config) {
    config->mutable_ssl_options()->set_verify_server_cert(verify_server_cert);
  }

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

// Checks the postgresql flags are valid, used before constructing connection
// config.
bool CheckPostgreSQLConfig(const std::string& host, const std::string& hostaddr,
                           const std::string& port, const std::string& user,
                           const std::string& dbname) {
  if (host.empty() == hostaddr.empty()) {
    LOG(ERROR) << "exactly one of postgres_config_host or "
                  "postgres_config_hostaddr must be specified ";
    return false;
  }
  if (port.empty()) {
    LOG(ERROR) << "postgres_config_port must not be empty";
    return false;
  }
  if (dbname.empty()) {
    LOG(ERROR) << "postgres_config_dbname must not be empty";
    return false;
  }
  if (user.empty()) {
    LOG(ERROR) << "postgres_config_user parameter must not be empty";
    return false;
  }
  return true;
}

// Given all the arguments defined by postgresql connection config, constructs
// connection config within server_config parameter. If connection config
// construction is not successful, return false.
bool ParsePostgreSQLFlagsBasedServerConfigOrDie(
    const std::string& host, const std::string& hostaddr,
    const std::string& port, const std::string& user,
    const std::string& password, const std::string& passfile,
    const std::string& dbname, const bool skip_db_creation,
    const std::string& sslmode, const std::string& sslcert,
    const std::string& sslkey, const std::string& sslpassword,
    const std::string& sslrootcert, const bool enable_database_upgrade,
    const int64_t downgrade_db_schema_version,
    ml_metadata::MetadataStoreServerConfig* server_config) {
  if (!CheckPostgreSQLConfig(host, hostaddr, port, user, dbname)) {
    return false;
  }

  ml_metadata::ConnectionConfig* connection_config =
      server_config->mutable_connection_config();
  ml_metadata::PostgreSQLDatabaseConfig* config =
      connection_config->mutable_postgresql();
  if (!host.empty()) {
    config->set_host(host);
  }
  if (!hostaddr.empty()) {
    config->set_hostaddr(hostaddr);
  }
  if (!port.empty()) {
    config->set_port(port);
  }
  if (!user.empty()) {
    config->set_user(user);
  }
  if (!password.empty()) {
    config->set_password(password);
  }
  if (!passfile.empty()) {
    config->set_passfile(passfile);
  }
  if (!dbname.empty()) {
    config->set_dbname(dbname);
  }
  if (skip_db_creation) {
    config->set_skip_db_creation(skip_db_creation);
  }
  if (!sslmode.empty()) {
    config->mutable_ssloption()->set_sslmode(sslmode);
  }
  if (!sslkey.empty()) {
    config->mutable_ssloption()->set_sslkey(sslkey);
  }
  if (!sslcert.empty()) {
    config->mutable_ssloption()->set_sslcert(sslcert);
  }
  if (!sslpassword.empty()) {
    config->mutable_ssloption()->set_sslpassword(sslpassword);
  }
  if (!sslrootcert.empty()) {
    config->mutable_ssloption()->set_sslrootcert(sslrootcert);
  }

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

// A list of valid metadata source config types, each item has corresponded
// argument value defined by flag metadata_source_config_type.
enum class SourceConfigType {
  kDefaultType,
  kConfigFile,
  kMySql,
  kPostgreSql,
};

// Converts flag metadata_source_config_type value to actual enum
// SourceConfigType. Returns error if conversion is not successful.
absl::StatusOr<SourceConfigType> ConvertToSourceConfig(
    std::string metadata_source_config_type) {
  if (metadata_source_config_type == "default") {
    return SourceConfigType::kDefaultType;
  } else if (metadata_source_config_type == "config_file") {
    return SourceConfigType::kConfigFile;
  } else if (metadata_source_config_type == "mysql") {
    return SourceConfigType::kMySql;
  } else if (metadata_source_config_type == "postgresql") {
    return SourceConfigType::kPostgreSql;
  }

  return absl::InvalidArgumentError(
      "metadata_source_config_type is not valid, provide value in one of the "
      "followings: default, config_file, mysql, postgresql.");
}

// Configures Connection Option to choose approach to consume flags.
// Currently supported options are: default, config_file, mysql, postgresql.
DEFINE_string(metadata_source_config_type, "default",
              "Provide the source connection type to determine what flags will "
              "be used by Metadata store to connect to Database. For default "
              "option, metadata_store_server_config_file will be used first, "
              "if metadata_store_server_config_file doesn't exist, try to "
              "connect to mysql using mysql prefixed flags. Valid values "
              "for this flag are: default, config_file, mysql, postgresql.");

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
DEFINE_string(mysql_config_sslcert, "",
              "This parameter specifies the file name of the client SSL certificate.");
DEFINE_string(mysql_config_sslkey, "",
              "This parameter specifies the location for the secret key used for the "
              "client certificate.");
DEFINE_string(mysql_config_sslrootcert, "",
              "This parameter specifies the name of a file containing SSL "
              "certificate authority (CA) certificate(s).");
DEFINE_string(mysql_config_sslcapath, "",
              "This parameter specifies path name of the directory "
              "that contains trusted SSL CA certificates.");
DEFINE_string(mysql_config_sslcipher, "",
              "This parameter specifies the list of permissible ciphers for "
              "SSL encryption.");
DEFINE_bool(mysql_config_verify_server_cert, false,
              "This parameter enables verification of the server certificate "
              " against the host name used when connecting to the server.");

// PostgreSQL config command line options
DEFINE_string(postgres_config_host, "",
              "Name of host to connect to. If host name starts with /, it is "
              "taken as a Unix-domain socket in the abstract namespace.");
DEFINE_string(postgres_config_hostaddr, "",
              "Numeric IP address of host to connect to. If this field is "
              "provided, `host` field is ignored.");
DEFINE_string(
    postgres_config_port, "",
    "Port number to connect to at the server host, or socket file name"
    " extension for Unix-domain connections.");
DEFINE_string(postgres_config_user, "",
              "PostgreSQL user name to connect as. Defaults to be the same as "
              "the operating system name of the user running the application.");
DEFINE_string(postgres_config_password, "",
              "Password to be used if the server demands password "
              "authentication.");
DEFINE_string(postgres_config_passfile, "",
              "Specifies the name of the file used to store passwords.");
DEFINE_string(postgres_config_dbname, "",
              "The database name. Defaults to be the same as the user name.");
DEFINE_bool(postgres_config_skip_db_creation, false,
            "True if skipping database instance creation during ML Metadata "
            "service initialization. By default it is false.");
DEFINE_string(postgres_config_sslmode, "",
              "PostgreSQL sslmode setup. Values can be disable, allow, "
              "verify-ca, verify-full, etc.");
DEFINE_string(
    postgres_config_sslcert, "",
    "This parameter specifies the file name of the client SSL certificate, "
    "replacing the default ~/.postgresql/postgresql.crt.");
DEFINE_string(
    postgres_config_sslkey, "",
    "This parameter specifies the location for the secret key used for the "
    "client certificate. It can either specify a file name that will be used "
    "instead of the default ~/.postgresql/postgresql.key");
DEFINE_string(postgres_config_sslpassword, "",
              "This parameter specifies the password for the secret key "
              "specified in sslkey, allowing client certificate private keys "
              "to be stored in encrypted form on disk even when interactive "
              "passphrase input is not practical.");
DEFINE_string(postgres_config_sslrootcert, "",
              "This parameter specifies the name of a file containing SSL "
              "certificate authority (CA) certificate(s). ");

DEFINE_bool(
    enable_database_upgrade, false,
    "Flag specifying database upgrade option. If set to true, it enables "
    "database migration during initialization(Optional parameter");
DEFINE_int64(downgrade_db_schema_version, -1,
             "Database downgrade schema version value. If set the database "
             "schema version is downgraded to the set value during "
             "initialization(Optional Parameter)");

// Default connection option for metadata source. It will check for
// the existence of config file first, and check for mysql flags if
// config file doesn't exist. Otherwise, it will create fake database.
// @return server configuration that contains connection config set by flags. Or
// error status if failed.
absl::StatusOr<ml_metadata::MetadataStoreServerConfig>
BuildDefaultConnectionConfig() {
  ml_metadata::MetadataStoreServerConfig server_config;
  if (!ParseMetadataStoreServerConfigOrDie(
          (FLAGS_metadata_store_server_config_file),
          &server_config) &&
      !ParseMySQLFlagsBasedServerConfigOrDie(
          (FLAGS_mysql_config_host),
          (FLAGS_mysql_config_port),
          (FLAGS_mysql_config_database),
          (FLAGS_mysql_config_user),
          (FLAGS_mysql_config_password),
          (FLAGS_mysql_config_sslcert),
          (FLAGS_mysql_config_sslkey),
          (FLAGS_mysql_config_sslrootcert),
          (FLAGS_mysql_config_sslcapath),
          (FLAGS_mysql_config_sslcipher),
          (FLAGS_mysql_config_verify_server_cert),
          (FLAGS_enable_database_upgrade),
          (FLAGS_downgrade_db_schema_version), &server_config)) {
    LOG(WARNING) << "The connection_config is not given. Using in memory fake "
                    "database, any metadata will not be persistent";
    server_config.mutable_connection_config()->mutable_fake_database();
  }
  return server_config;
}

// config_file connection option for metadata source. It will check for
// the existence of config file, and try to construct connection config.
// @return server configuration that contains connection config set by config
// file. Or error status if failed.
absl::StatusOr<ml_metadata::MetadataStoreServerConfig>
BuildFileBasedConnectionConfig() {
  ml_metadata::MetadataStoreServerConfig server_config;
  if (ParseMetadataStoreServerConfigOrDie(
          (FLAGS_metadata_store_server_config_file),
          &server_config)) {
    return server_config;
  } else {
    LOG(ERROR) << "Unable to construct server config using config file.";
    return absl::InvalidArgumentError(
        "Unable to construct server config using config file.");
  }
}

// MySQL connection option for metadata source. It will check for
// the existence of mysql flags, and try to construct connection config.
// @return server configuration that contains connection config set by mysql
// flags. Or error status if failed.
absl::StatusOr<ml_metadata::MetadataStoreServerConfig>
BuildMySQLConnectionConfig() {
  ml_metadata::MetadataStoreServerConfig server_config;
  if (ParseMySQLFlagsBasedServerConfigOrDie(
          (FLAGS_mysql_config_host),
          (FLAGS_mysql_config_port),
          (FLAGS_mysql_config_database),
          (FLAGS_mysql_config_user),
          (FLAGS_mysql_config_password),
          (FLAGS_mysql_config_sslcert),
          (FLAGS_mysql_config_sslkey),
          (FLAGS_mysql_config_sslrootcert),
          (FLAGS_mysql_config_sslcapath),
          (FLAGS_mysql_config_sslcipher),
          (FLAGS_mysql_config_verify_server_cert),
          (FLAGS_enable_database_upgrade),
          (FLAGS_downgrade_db_schema_version), &server_config)) {
    return server_config;
  } else {
    LOG(ERROR) << "Unable to construct server config using config file.";
    return absl::InvalidArgumentError(
        "Unable to construct server config using config file.");
  }
}

// Constructs Connection Config for PostgreSQL database. Requires to
// set metadata_source_config_type as "postgresql", then provide necessary
// information in flags that have prefix of `postgres_`.
// Example run:
// sudo docker run --name "${MLMD_GRPC_CONTAINER}" \
//   -p ${MLMD_GRPC_PORT}:${MLMD_GRPC_PORT} \
//   --network="${GRPC_E2E_BRIDGE_NETWORK}"\
//   --entrypoint /bin/metadata_store_server -d "${MLMD_DOCKER_IMAGE}" \
//   --grpc_port=${MLMD_GRPC_PORT} \
//   --metadata_source_config_type="postgresql" \
//   --postgres_config_host=${POSTGRESQL_CONTAINER_NAME} \
//   --postgres_config_port="3456" \
//   --postgres_config_user="root" \
//   --postgres_config_password="${PWD}" \
//   --postgres_config_dbname="mlmd-db"
// @return server configuration that contains connection config set by
// postgresql flags. Or error status if failed.
absl::StatusOr<ml_metadata::MetadataStoreServerConfig>
BuildPostgreSQLConnectionConfig() {
  ml_metadata::MetadataStoreServerConfig server_config;
  if (ParsePostgreSQLFlagsBasedServerConfigOrDie(
          (FLAGS_postgres_config_host),
          (FLAGS_postgres_config_hostaddr),
          (FLAGS_postgres_config_port),
          (FLAGS_postgres_config_user),
          (FLAGS_postgres_config_password),
          (FLAGS_postgres_config_passfile),
          (FLAGS_postgres_config_dbname),
          (FLAGS_postgres_config_skip_db_creation),
          (FLAGS_postgres_config_sslmode),
          (FLAGS_postgres_config_sslcert),
          (FLAGS_postgres_config_sslkey),
          (FLAGS_postgres_config_sslpassword),
          (FLAGS_postgres_config_sslrootcert),
          (FLAGS_enable_database_upgrade),
          (FLAGS_downgrade_db_schema_version), &server_config)) {
    return server_config;
  } else {
    LOG(ERROR) << "Unable to construct server config using postgresql flags.";
    return absl::InvalidArgumentError(
        "Unable to construct server config using postgresql flags.");
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if ((FLAGS_grpc_port) <= 0) {
    LOG(ERROR) << "grpc_port is invalid: " << (FLAGS_grpc_port);
    return -1;
  }

  const std::string metadata_source_config_type =
      (FLAGS_metadata_source_config_type);
  absl::StatusOr<SourceConfigType> source_config =
      ConvertToSourceConfig(metadata_source_config_type);
  if (!source_config.ok()) {
    LOG(ERROR) << "metadata_source_config_type is invalid: "
               << source_config.status().message();
    return -1;
  }

  absl::StatusOr<ml_metadata::MetadataStoreServerConfig> server_config_status;
  if (source_config.value() == SourceConfigType::kDefaultType) {
    server_config_status = BuildDefaultConnectionConfig();
  } else if (source_config.value() == SourceConfigType::kConfigFile) {
    server_config_status = BuildFileBasedConnectionConfig();
  } else if (source_config.value() == SourceConfigType::kMySql) {
    server_config_status = BuildMySQLConnectionConfig();
  } else if (source_config.value() == SourceConfigType::kPostgreSql) {
    server_config_status = BuildPostgreSQLConnectionConfig();
  } else {
    LOG(ERROR) << "metadata_source_config_type is invalid: "
               << metadata_source_config_type;
    return -1;
  }
  if (!server_config_status.ok()) {
    LOG(ERROR) << "Unable to construct server config based on arguments. Error "
                  "message: "
               << server_config_status.status().message();
    return -1;
  }

  ml_metadata::MetadataStoreServerConfig server_config =
      server_config_status.value();
  ml_metadata::ConnectionConfig connection_config =
      server_config.connection_config();

  // Creates a metadata_store in the main thread and init schema if necessary.
  std::unique_ptr<ml_metadata::MetadataStore> metadata_store;
  absl::Status status = ml_metadata::CreateMetadataStore(
      connection_config, server_config.migration_options(), &metadata_store);
  for (int i = 0; i < (FLAGS_metadata_store_connection_retries);
       i++) {
    if (status.ok() || !absl::IsAborted(status)) {
      break;
    }
    LOG(WARNING) << "Connection Aborted with error: " << status;
    LOG(INFO) << "Retry attempt " << i;
    status = ml_metadata::CreateMetadataStore(
        connection_config, server_config.migration_options(), &metadata_store);
  }
  CHECK_EQ(absl::OkStatus(), status)
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
