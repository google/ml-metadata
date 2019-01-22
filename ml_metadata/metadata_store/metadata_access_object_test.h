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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_TEST_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_TEST_H_

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {

namespace testing {

// An Interface to generate and retrieve a MetadataAccessObject.
class MetadataAccessObjectContainer {
 public:
  virtual ~MetadataAccessObjectContainer() = default;

  // MetadataSource is owned by MetadataAccessObjectContainer.
  virtual MetadataSource* GetMetadataSource() = 0;

  // MetadataAccessObject is owned by MetadataAccessObjectContainer.
  virtual MetadataAccessObject* GetMetadataAccessObject() = 0;
};

// Represents the type of the Gunit Test param for the parameterized
// MetadataAccessObjectTest.
//
// Note: Since the Gunit Test param needs to be a copyable type, we use a
// std::function as opposed to directly using
// std::unique_ptr<MetadataAccessObjectContainer> as the Gunit Test param type.
using MetadataAccessObjectContainerFactory =
    std::function<std::unique_ptr<MetadataAccessObjectContainer>()>;

// A parameterized abstract test fixture to run tests for MetadataAccessObjects
// created with different MetadataSource types.
// See metadata_access_object_test.cc for list of test cases using this fixture.
//
// To run these tests for a MetadataAccessObject based on a newly added
// MetadataSource (say foo) , follow these steps:
// - Step #1: Define a new test file  foo_metadata_access_object_test.cc.
// - Step #2: Implement FooMetadataAccessObjectContainer.
//   class FooMetadataAccessObjectContainer : MetadataAccessObjectContainer {
//     ...
//   };
// - Step #3: Instantiate this parameterized test with a function that generates
//            a std::unique_ptr<FooMetadataAccessObjectContainer>
//   INSTANTIATE_TEST_CASE_P(
//       FooMetadataAccessObjectTest, MetadataAccessObjectTest,
//       ::testing::Values([]() {
//         return absl::make_unique<FakeMetadataAccessObjectContainer>();
//       }));
//
// See concrete metadata_access_object_test.cc for examples.
class MetadataAccessObjectTest
    : public ::testing::TestWithParam<MetadataAccessObjectContainerFactory> {
 protected:
  void SetUp() override {
    metadata_access_object_container_ = GetParam()();
    metadata_source_ = metadata_access_object_container_->GetMetadataSource();
    metadata_access_object_ =
        metadata_access_object_container_->GetMetadataAccessObject();
    TF_ASSERT_OK(metadata_source_->Begin());
  }
  void TearDown() override {
    TF_ASSERT_OK(metadata_source_->Commit());
    metadata_source_ = nullptr;
    metadata_access_object_ = nullptr;
    metadata_access_object_container_ = nullptr;
  }

  std::unique_ptr<MetadataAccessObjectContainer>
      metadata_access_object_container_;

  // metadata_source_ and metadata_access_object_ are unowned.
  MetadataSource* metadata_source_;
  MetadataAccessObject* metadata_access_object_;
};

}  // namespace testing

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_TEST_H_
