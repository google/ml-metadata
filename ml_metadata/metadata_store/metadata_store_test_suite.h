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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_STORE_TEST_SUITE_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_STORE_TEST_SUITE_H_

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ml_metadata/metadata_store/metadata_store.h"

namespace ml_metadata {
namespace testing {

// An Interface to create, retrieve and clean a MetadataStore object.
class MetadataStoreContainer {
 public:
  MetadataStoreContainer() = default;
  virtual ~MetadataStoreContainer() = default;

  // MetadataStore is owned by MetadataStoreContainer.
  virtual MetadataStore* GetMetadataStore() = 0;
};

// Represents the type of the Gunit Test param for the parameterized
// MetadataStoreTestSuite.
//
// Note: Since the Gunit Test param needs to be a copyable type, we use a
// std::function as opposed to directly using
// std::unique_ptr<MetadataStoreContainer> as the Gunit Test param type.
using MetadataStoreContainerFactory =
    std::function<std::unique_ptr<MetadataStoreContainer>()>;

class MetadataStoreTestSuite
    : public ::testing::TestWithParam<MetadataStoreContainerFactory> {
 protected:
  void SetUp() override {
    metadata_store_container_ = GetParam()();
    metadata_store_ = metadata_store_container_->GetMetadataStore();
  }

  void TearDown() override {
    metadata_store_ = nullptr;
    metadata_store_container_ = nullptr;
  }

  std::unique_ptr<MetadataStoreContainer> metadata_store_container_;
  MetadataStore* metadata_store_;
};

}  // namespace testing
}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_STORE_TEST_SUITE_H_
