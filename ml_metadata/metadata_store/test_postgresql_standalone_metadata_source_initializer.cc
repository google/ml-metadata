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

#include "ml_metadata/metadata_store/test_postgresql_metadata_source_initializer.h"
#include "ml_metadata/proto/metadata_store.pb.h"

DEFINE_string(db_name, "", "Name of PostgreSQL database to connect to");
DEFINE_string(user_name, "", "PostgreSQL login user name");
DEFINE_string(password, "",
              "Password for user_name. If empty, only PostgreSQL user ids that "
              "don't have a password set are allowed to connect.");
DEFINE_string(host_name, "",
              "Host name or IP address of the PostgreSQL server.");
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
    config.set_host((FLAGS_host_name));
    config.set_port((FLAGS_port));
    config.set_user((FLAGS_user_name));
    config.set_password((FLAGS_password));

    metadata_source_ = std::make_unique<PostgreSQLMetadataSource>(config);

    return metadata_source_.get();
  }

  void Cleanup() override { metadata_source_ = nullptr; }

 private:
  std::unique_ptr<PostgreSQLMetadataSource> metadata_source_;
};

std::unique_ptr<TestPostgreSQLMetadataSourceInitializer>
GetTestPostgreSQLMetadataSourceInitializer() {
  return std::make_unique<TestPostgreSQLStandaloneMetadataSourceInitializer>();
}

}  // namespace testing
}  // namespace ml_metadata
