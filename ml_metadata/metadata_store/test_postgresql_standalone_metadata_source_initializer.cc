/* Copyright 2023 Google LLC

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
#include <memory>

#include "gflags/gflags.h"
#include <glog/logging.h>

#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/postgresql_metadata_source.h"
#include "ml_metadata/metadata_store/test_postgresql_metadata_source_initializer.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include <libpq-fe.h>

DEFINE_string(db_name, "", "Name of PostgreSQL database to connect to");
DEFINE_string(user_name, "", "PostgreSQL login user name");
DEFINE_string(password, "",
              "Password for user_name. If empty, only PostgreSQL user ids that "
              "don't have a password set are allowed to connect.");
DEFINE_string(hostaddr, "", "Host IP address of the PostgreSQL server.");
DEFINE_string(
    port, "3456",
    "TCP port number that the PostgreSQL server accepts connection on. If "
    "not set, the default PostgreSQL port (3456) is used.");

namespace ml_metadata {
namespace testing {

// Provides a TestPostgreSQLMetadataSourceInitializer implementation that
// instantiates a PostgreSQL backed by a PostgreSQL server as specified by
// the flags listed above.
class TestPostgreSQLStandaloneMetadataSourceInitializer
    : public TestPostgreSQLMetadataSourceInitializer {
 public:
  TestPostgreSQLStandaloneMetadataSourceInitializer() = default;
  ~TestPostgreSQLStandaloneMetadataSourceInitializer() override = default;

  PostgreSQLMetadataSource* Init() override {
    PostgreSQLDatabaseConfig config;
    config.set_dbname((FLAGS_db_name));
    config.set_hostaddr((FLAGS_hostaddr));
    config.set_port((FLAGS_port));
    config.set_user((FLAGS_user_name));
    config.set_password((FLAGS_password));
    // In PostgreSQL, there is a default DB called postgres.
    // In order to CREATE DATABASE, MLMD needs to connect to
    // default DB first to run this command. For running unit
    // tests on PostgreSQL, setting skip_db_creation as false
    // so it can start from fresh DB in each test case.
    config.set_skip_db_creation(false);

    LOG(INFO) << "POSTGRESQL databaseconfig: " << config.DebugString();
    metadata_source_ = std::make_unique<PostgreSQLMetadataSource>(config);

    return metadata_source_.get();
  }

  void DropDB(std::string db_name) {
    const std::string default_db_name = "postgres";
    if (db_name.empty() || db_name == default_db_name) {
      LOG(ERROR) << "TestPostgreSQLStandaloneMetadataSourceInitializer: cannot "
                 << "delete Postgresql's default database: postgres.";
      return;
    }

    // Build connection config using command line flags, except that it is
    // connecting to default db.
    PostgreSQLDatabaseConfig config;
    config.set_dbname(default_db_name);
    config.set_hostaddr((FLAGS_hostaddr));
    config.set_port((FLAGS_port));
    config.set_user((FLAGS_user_name));
    config.set_password((FLAGS_password));
    bool use_default_db = false;
    std::string connection_config =
        buildConnectionConfig(config, use_default_db);

    // Establish connection with postgres db.
    PGconn* conn = PQconnectdb(connection_config.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
      LOG(ERROR) << "TestPostgreSQLStandaloneMetadataSourceInitializer: "
                 << "postgresql error: " << PQerrorMessage(conn);
      PQfinish(conn);
      return;
    } else {
      LOG(INFO) << "TestPostgreSQLStandaloneMetadataSourceInitializer: "
                << "Connection to database succeeds.";
    }

    // Remove any active connection to the target db, then drop database.
    // Note that Dropping DB cannot be executed within a transaction block.
    PGresult* res =
        PQexec(conn, ("SELECT pg_terminate_backend(pg_stat_activity.pid) FROM "
                      "pg_stat_activity WHERE pg_stat_activity.datname = '" +
                      db_name + "';")
                         .c_str());
    PQclear(res);
    res =
        PQexec(conn, absl::StrCat("DROP DATABASE IF EXISTS ", db_name).c_str());
    PQclear(res);
    PQfinish(conn);
  }

  void Cleanup() override {
    std::string db_name = (FLAGS_db_name);
    // In PostgreSQL, we need to create DB from another DB connection.
    // In order to enable fresh DB testing, we set skip_db_creation as false
    // to create new DB every time before each test case. However, since the
    // test DB has the same name across all test cases, the second test case
    // will fail if we don't drop the DB after first test case finishes.
    // Therefore, we need to drop DB during test case cleanup.
    DropDB(db_name);
    metadata_source_ = nullptr;
  }

 private:
  std::unique_ptr<PostgreSQLMetadataSource> metadata_source_;
};

std::unique_ptr<TestPostgreSQLMetadataSourceInitializer>
GetTestPostgreSQLMetadataSourceInitializer() {
  return std::make_unique<TestPostgreSQLStandaloneMetadataSourceInitializer>();
}

}  // namespace testing
}  // namespace ml_metadata
