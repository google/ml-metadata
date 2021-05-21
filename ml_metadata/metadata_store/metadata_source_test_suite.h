/* Copyright 2020 Google LLC

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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_SOURCE_TEST_SUITE_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_SOURCE_TEST_SUITE_H_

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"

namespace ml_metadata {
namespace testing {

// An Interface to create, retrieve and clean a MetadataSource.
class MetadataSourceContainer {
 public:
  MetadataSourceContainer() = default;
  virtual ~MetadataSourceContainer() = default;

  // MetadataSource is owned by MetadataSourceContainer.
  virtual MetadataSource* GetMetadataSource() = 0;

  // Creates a new table schema for testing.
  virtual void InitTestSchema() = 0;

  // Creates table schema and populates rows.
  virtual void InitSchemaAndPopulateRows() = 0;
};

// Represents the type of the Gunit Test param for the parameterized
// MetadataSourceTestSuite.
//
// Note: Since the Gunit Test param needs to be a copyable type, we use a
// std::function as opposed to directly using
// std::unique_ptr<MetadataSourceContainer> as the Gunit Test param type.
using MetadataSourceContainerFactory =
    std::function<std::unique_ptr<MetadataSourceContainer>()>;

class MetadataSourceTestSuite
    : public ::testing::TestWithParam<MetadataSourceContainerFactory> {
 protected:
  void SetUp() override {
    metadata_source_container_ = GetParam()();
    metadata_source_ = metadata_source_container_->GetMetadataSource();
  }

  void TearDown() override {
    metadata_source_ = nullptr;
    metadata_source_container_ = nullptr;
  }

  std::unique_ptr<MetadataSourceContainer> metadata_source_container_;
  // metadata_source_ is unowned.
  MetadataSource* metadata_source_;
};

}  // namespace testing
}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_SOURCE_TEST_SUITE_H_
