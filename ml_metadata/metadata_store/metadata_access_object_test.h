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
#include "ml_metadata/proto/metadata_source.pb.h"
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

  virtual tensorflow::Status Init() {
    return GetMetadataAccessObject()->InitMetadataSource();
  }

  // Tests if there is upgrade verification.
  virtual bool HasUpgradeVerification(int64 version) = 0;

  // Tests if there is upgrade verification.
  virtual bool HasDowngradeVerification(int64 version) = 0;

  // Tests if there is filter query support.
  virtual bool HasFilterQuerySupport() { return false; }

  // Tests if there is parent context query support.
  virtual bool HasParentContextSupport() { return false; }

  // Initializes the previous version of the database for downgrade.
  virtual tensorflow::Status SetupPreviousVersionForDowngrade(
      int64 version) = 0;

  // Verifies that a database has been downgraded to version.
  virtual tensorflow::Status DowngradeVerification(int64 version) = 0;

  // Initializes the previous version of the database for upgrade.
  virtual tensorflow::Status SetupPreviousVersionForUpgrade(int64 version) = 0;

  // Verifies that a database has been upgraded to version.
  virtual tensorflow::Status UpgradeVerification(int64 version) = 0;

  // Drops the type table (or some other table) to test the behavior of
  // InitMetadataSourceIfNotExists when a database is partially created.
  virtual tensorflow::Status DropTypeTable() = 0;

  // Drops the artiface table (or some other table) to test the behavior of
  // InitMetadataSourceIfNotExists when a database is partially created.
  virtual tensorflow::Status DropArtifactTable() = 0;

  // Deletes the schema version from MLMDVersion: this corrupts the database.
  virtual tensorflow::Status DeleteSchemaVersion() = 0;

  // Sets the schema version to an incompatible version in the future,
  virtual tensorflow::Status SetDatabaseVersionIncompatible() = 0;

  // Returns the minimum version to test for upgrades and downgrades.
  virtual int64 MinimumVersion() = 0;

  // If returns true, should perform tests that rely on some implementation
  // details of the database. Specifically, these tests rely on having a
  // MLMDEnv table, being able to not specify a schema version, and having
  // Init() be able to reset the database.
  virtual bool PerformExtendedTests() = 0;
};

// An Interface to generate and retrieve a MetadataAccessObject, where there
// is an associated MetadataSourceQueryConfig.
class QueryConfigMetadataAccessObjectContainer
    : public MetadataAccessObjectContainer {
 public:
  QueryConfigMetadataAccessObjectContainer(
      const MetadataSourceQueryConfig& config)
      : config_(config) {}

  virtual ~QueryConfigMetadataAccessObjectContainer() = default;

  bool HasUpgradeVerification(int64 version) final;

  bool HasDowngradeVerification(int64 version) final;

  bool HasFilterQuerySupport() final { return true; }

  bool HasParentContextSupport() final { return true; }

  tensorflow::Status SetupPreviousVersionForDowngrade(int64 version) final;

  tensorflow::Status DowngradeVerification(int64 version) final;

  tensorflow::Status SetupPreviousVersionForUpgrade(int64 version) final;

  tensorflow::Status UpgradeVerification(int64 version) final;

  tensorflow::Status DropTypeTable() final;

  tensorflow::Status DropArtifactTable() final;

  tensorflow::Status DeleteSchemaVersion() final;

  tensorflow::Status SetDatabaseVersionIncompatible() final;

  bool PerformExtendedTests() final { return true; }

  int64 MinimumVersion() final;

 private:
  // Get a migration scheme, or return NOT_FOUND.
  tensorflow::Status GetMigrationScheme(
      int64 version,
      MetadataSourceQueryConfig::MigrationScheme* migration_scheme);

  // Verify that a sequence of queries return true.
  tensorflow::Status Verification(
      const google::protobuf::RepeatedPtrField<MetadataSourceQueryConfig::TemplateQuery>&
          queries);

  MetadataSourceQueryConfig config_;
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
    TF_CHECK_OK(metadata_source_->Begin());
  }

  void TearDown() override {
    TF_CHECK_OK(metadata_source_->Commit());
    metadata_source_ = nullptr;
    metadata_access_object_ = nullptr;
    metadata_access_object_container_ = nullptr;
  }

  tensorflow::Status Init() {
    return metadata_access_object_container_->Init();
  }

  template <class NodeType>
  int64 InsertType(const std::string& type_name) {
    NodeType type;
    type.set_name(type_name);
    int64 type_id;
    TF_CHECK_OK(metadata_access_object_->CreateType(type, &type_id));
    return type_id;
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
