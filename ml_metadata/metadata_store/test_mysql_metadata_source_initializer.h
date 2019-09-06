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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_TEST_MYSQL_METADATA_SOURCE_INITIALIZER_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_TEST_MYSQL_METADATA_SOURCE_INITIALIZER_H_

#include <memory>

#include "ml_metadata/metadata_store/mysql_metadata_source.h"

namespace ml_metadata {
namespace testing {

// TestMySqlMetadataSourceInitializer provides an interface to
// initialize/cleanup a MySqlMetadataSource in tests.
//
// Typical usage:
//
// class FooTest : public ::testing::Test {
//  protected:
//   void SetUp() override {
//     metadata_source_initializer_ = GetTestMySqlMetadataSourceInitializer();
//     metadata_source_ = metadata_source_initializer_->Init(
//       TestMysqlMetadataSourceInitializer::ConnectionType::kTcp);
//   }
//   void TearDown() override { metadata_source_initializer_->Cleanup(); }
//
//   TestMySqlMetadataSourceInitializer* metadata_source_initializer_;
//   MySqlMetadataSource* metadata_source_;
// };
class TestMySqlMetadataSourceInitializer {
 public:
  enum class ConnectionType { kTcp, kSocket };

  virtual ~TestMySqlMetadataSourceInitializer() = default;

  // Creates or initializes a MySqlMetadataSource.
  virtual MySqlMetadataSource* Init(ConnectionType connection_type) = 0;

  // Removes any existing MySqlMetadataSource.
  virtual void Cleanup() = 0;
};

// Returns a TestMySqlMetadataSourceInitializer to init/cleanup a
// MySqlMetadataSource in tests.
std::unique_ptr<TestMySqlMetadataSourceInitializer>
GetTestMySqlMetadataSourceInitializer();

}  // namespace testing
}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_TEST_MYSQL_METADATA_SOURCE_INITIALIZER_H_
