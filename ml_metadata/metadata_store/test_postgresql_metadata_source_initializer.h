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

#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_TEST_POSTGRESQL_METADATA_SOURCE_INITIALIZER_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_TEST_POSTGRESQL_METADATA_SOURCE_INITIALIZER_H_

#include <memory>

#include "ml_metadata/metadata_store/postgresql_metadata_source.h"

namespace ml_metadata {
namespace testing {

// Parent class for PostgreSQL database creation and metadata_source
// management. Inherit this class to develop initialization of
// PostgreSQLMetadataSource based on the testing environment.
class TestPostgreSQLMetadataSourceInitializer {
 public:
  virtual ~TestPostgreSQLMetadataSourceInitializer() = default;

  // Creates or initializes a PostgreSQLMetadataSource.
  virtual PostgreSQLMetadataSource* Init() = 0;

  // Removes any existing PostgreSQLMetadataSource.
  virtual void Cleanup() = 0;
};

// Returns a TestPostgreSQLMetadataSourceInitializer to init/cleanup a
// PostgreSQLMetadataSource in tests.
std::unique_ptr<TestPostgreSQLMetadataSourceInitializer>
GetTestPostgreSQLMetadataSourceInitializer();

}  // namespace testing
}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_TEST_POSTGRESQL_METADATA_SOURCE_INITIALIZER_H_
