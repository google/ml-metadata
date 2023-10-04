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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/return_utils.h"

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

  // If the head library needs to support earlier schema version, this method
  // should be overridden to test against an earlier schema version.
  virtual std::optional<int64_t> GetSchemaVersion() { return absl::nullopt; }

  // Returns OK if DB schema passed verification.
  virtual absl::Status VerifyDbSchema(const int64_t version) {
    return absl::OkStatus();
  }

  // Init a test db environment. By default the testsuite is run against the
  // head schema. If GetSchemaVersion() is overridden, it prepares a
  // db at the particular schema version.
  virtual absl::Status Init() {
    MLMD_RETURN_IF_ERROR(GetMetadataAccessObject()->InitMetadataSource());
    // If the test suite indicates the library at head should be tested against
    // an existing db with a previous schema version, we downgrade the
    // initialized schema to setup the test environment.
    const std::optional<int64_t> earlier_schema_version = GetSchemaVersion();
    if (earlier_schema_version) {
      MLMD_RETURN_IF_ERROR(GetMetadataAccessObject()->DowngradeMetadataSource(
          *earlier_schema_version));
    }
    return absl::OkStatus();
  }

  // Skips schema migration tests if returned true.
  virtual bool SkipSchemaMigrationTests() { return false; }


  // Tests if there is upgrade verification.
  virtual bool HasUpgradeVerification(int64_t version) = 0;

  // Tests if there is upgrade verification.
  virtual bool HasDowngradeVerification(int64_t version) = 0;

  // Tests if there is filter query support.
  virtual bool HasFilterQuerySupport() { return false; }

  // Adds a commit point in the tests.
  // Default to be a no-op for SQLite, MySQL.
  virtual absl::Status AddCommitPoint() { return absl::OkStatus(); }

  // Resets the test transaction to the empty state.
  // Used after the test transaction commit fails.
  // Default to be a no-op for SQLite, MySQL.
  virtual absl::Status ResetForRetry() { return absl::OkStatus(); }

  // Checks the `unique_constraint_violation_status` is `AlreadyExistsError` and
  // reset the test transaction.
  // Returns InvalidArgumentError error, if `unique_constraint_violation_status`
  // is not `AlreadyExistsError`.
  // Returns detailed INTERNAL error, if the transaction reset fails.
  virtual absl::Status CheckUniqueConstraintAndResetTransaction(
      const absl::Status& unique_constraint_violation_status) {
    if (!absl::IsAlreadyExists(unique_constraint_violation_status)) {
      return absl::InvalidArgumentError(
          "Unique constraint violation status is not AlreadyExistsError!");
    }
    MLMD_RETURN_IF_ERROR(GetMetadataSource()->Rollback());
    MLMD_RETURN_IF_ERROR(GetMetadataSource()->Begin());
    return absl::OkStatus();
  }

  // Initializes the previous version of the database for downgrade.
  virtual absl::Status SetupPreviousVersionForDowngrade(int64_t version) = 0;

  // Verifies that a database has been downgraded to version.
  virtual absl::Status DowngradeVerification(int64_t version) = 0;

  // Initializes the previous version of the database for upgrade.
  virtual absl::Status SetupPreviousVersionForUpgrade(int64_t version) = 0;

  // Verifies that a database has been upgraded to version.
  virtual absl::Status UpgradeVerification(int64_t version) = 0;

  // Drops the type table (or some other table) to test the behavior of
  // InitMetadataSourceIfNotExists when a database is partially created.
  virtual absl::Status DropTypeTable() = 0;

  // Drops the artifact table (or some other table) to test the behavior of
  // InitMetadataSourceIfNotExists when a database is partially created.
  virtual absl::Status DropArtifactTable() = 0;

  // Deletes the schema version from MLMDVersion: this corrupts the database.
  virtual absl::Status DeleteSchemaVersion() = 0;

  // Determines whether or not the specified table is empty.
  // Returns `true` if the specified table is empty and `false` otherwise.
  // Returns an InternalError if there is an error executing the underlying
  // queries.
  virtual absl::StatusOr<bool> CheckTableEmpty(
      absl::string_view table_name) = 0;

  // Sets the schema version to an incompatible version in the future,
  virtual absl::Status SetDatabaseVersionIncompatible() = 0;

  // Returns the minimum version to test for upgrades and downgrades.
  virtual int64_t MinimumVersion() = 0;

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
  // By default the container returns a query config based MetadataAccessObject
  // that uses the query config at head. If earlier_schema_version is passed,
  // it creates a MetadataAccessObject that supports querying the earlier
  // db schema.
  QueryConfigMetadataAccessObjectContainer(
      const MetadataSourceQueryConfig& config,
      std::optional<int64_t> earlier_schema_version = absl::nullopt)
      : config_(config), testing_schema_version_(earlier_schema_version) {}

  virtual ~QueryConfigMetadataAccessObjectContainer() = default;

  std::optional<int64_t> GetSchemaVersion() final {
    return testing_schema_version_;
  }

  bool HasUpgradeVerification(int64_t version) final;

  bool HasDowngradeVerification(int64_t version) final;

  bool HasFilterQuerySupport() final { return true; }

  absl::Status VerifyDbSchema(int64_t version) final;

  absl::Status SetupPreviousVersionForDowngrade(int64_t version) final;

  absl::Status DowngradeVerification(int64_t version) final;

  absl::Status SetupPreviousVersionForUpgrade(int64_t version) final;

  absl::Status UpgradeVerification(int64_t version) final;

  absl::Status DropTypeTable() final;

  absl::Status DropArtifactTable() final;

  absl::Status DeleteSchemaVersion() final;

  absl::StatusOr<bool> CheckTableEmpty(absl::string_view table_name) final;

  absl::Status SetDatabaseVersionIncompatible() final;

  bool PerformExtendedTests() final { return true; }

  int64_t MinimumVersion() final;

  virtual std::string GetTableNumQuery() {
    return "select count(*) from sqlite_master where type='table' "
           "and name NOT LIKE 'sqlite_%' ;";
  }

  virtual std::string GetIndexNumQuery() {
    return "select count(*) from sqlite_master where type='index';";
  }

 private:
  // Get a migration scheme, or return NOT_FOUND.
  absl::Status GetMigrationScheme(
      int64_t version,
      MetadataSourceQueryConfig::MigrationScheme* migration_scheme);

  // Verify that a sequence of queries return true.
  absl::Status Verification(
      const google::protobuf::RepeatedPtrField<MetadataSourceQueryConfig::TemplateQuery>&
          queries);

  MetadataSourceQueryConfig config_;
  // If not set, by default, we test against the head version.
  std::optional<int64_t> testing_schema_version_;
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
//         return std::make_unique<FakeMetadataAccessObjectContainer>();
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
    if (EarlierSchemaEnabled()) {
      LOG(INFO) << "Test against the earlier schema version: "
                << *metadata_access_object_container_->GetSchemaVersion();
    }
    CHECK_EQ(absl::OkStatus(), metadata_source_->Begin());
  }

  void TearDown() override {
    CHECK_EQ(absl::OkStatus(), metadata_source_->Commit());
    metadata_source_ = nullptr;
    metadata_access_object_ = nullptr;
    metadata_access_object_container_ = nullptr;
  }

  absl::Status Init() { return metadata_access_object_container_->Init(); }

  // Uses to skip the tests that are not relevant to any earlier schema version.
  bool EarlierSchemaEnabled() {
    return metadata_access_object_container_->GetSchemaVersion().has_value();
  }

  // Uses to skip the schema migration related tests.
  bool SkipSchemaMigrationTests() {
    return metadata_access_object_container_->SkipSchemaMigrationTests();
  }


  // Uses to indicate the minimum expected schema version to run a test.
  bool SkipIfEarlierSchemaLessThan(int64_t min_schema_version) {
    const bool is_skip =
        EarlierSchemaEnabled() &&
        *metadata_access_object_container_->GetSchemaVersion() <
            min_schema_version;
    if (is_skip) {
      LOG(INFO) << "Skipping the test as it requires schema version at least: "
                << min_schema_version;
    }
    return is_skip;
  }

  // Uses as a condition for diverging different test behaviors for different
  // schema versions.
  bool IfSchemaLessThan(int64_t schema_version) {
    const bool is_true =
        EarlierSchemaEnabled() &&
        *metadata_access_object_container_->GetSchemaVersion() < schema_version;
    return is_true;
  }

  // Uses to a add commit point if needed in the tests.
  // Default to be a no-op for SQLite, MySQL.
  absl::Status AddCommitPointIfNeeded() {
    return metadata_access_object_container_->AddCommitPoint();
  }

  // Checks whether the unique constraint violation status is correct and reset
  // the test transaction.
  absl::Status CheckUniqueConstraintAndResetTransaction(
      const absl::Status& unique_constraint_violation_status) {
    return metadata_access_object_container_
        ->CheckUniqueConstraintAndResetTransaction(
            unique_constraint_violation_status);
  }

  template <class NodeType>
  int64_t InsertType(const std::string& type_name) {
    NodeType type;
    type.set_name(type_name);
    int64_t type_id;
    CHECK_EQ(absl::OkStatus(),
             metadata_access_object_->CreateType(type, &type_id));
    CHECK_EQ(absl::OkStatus(),
             metadata_access_object_container_->AddCommitPoint());
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
