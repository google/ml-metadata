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
#include <cstdint>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/substitute.h"
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/metadata_access_object_factory.h"
#include "ml_metadata/metadata_store/metadata_access_object_test.h"
#include "ml_metadata/metadata_store/postgresql_metadata_source.h"
#include "ml_metadata/metadata_store/test_postgresql_metadata_source_initializer.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"

namespace ml_metadata {
namespace testing {

namespace {

absl::StatusOr<int> GetCountQueryResult(const std::string& query,
                                        MetadataSource* metadata_source) {
  RecordSet record_set;
  int result;
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
      metadata_source->ExecuteQuery(query, &record_set), "query: ", query);
  if (record_set.records_size() != 1) {
    return absl::InternalError(
        absl::StrCat("Verification failed on query ", query));
  }
  if (!absl::SimpleAtoi(record_set.records(0).values(0), &result)) {
    return absl::InternalError(
        absl::StrCat("Value incorrect:", record_set.records(0).DebugString(),
                     " on query ", query));
  }
  return result;
}

class PostgresqlMetadataAccessObjectContainer
    : public MetadataAccessObjectContainer {
 public:
  PostgresqlMetadataAccessObjectContainer(
      std::optional<int64_t> earlier_schema_version = absl::nullopt)
      : MetadataAccessObjectContainer() {
    testing_schema_version_ = earlier_schema_version;
    config_ = util::GetPostgreSQLMetadataSourceQueryConfig();
    metadata_source_initializer_ = GetTestPostgreSQLMetadataSourceInitializer();
    metadata_source_ = metadata_source_initializer_->Init();
    CHECK_EQ(CreateMetadataAccessObject(config_, metadata_source_,
                                        earlier_schema_version,
                                        &metadata_access_object_),
             absl::OkStatus());
  }

  ~PostgresqlMetadataAccessObjectContainer() override {
    metadata_source_initializer_->Cleanup();
    metadata_source_initializer_ = nullptr;
    metadata_source_ = nullptr;
    metadata_access_object_ = nullptr;
  };
  int64_t MinimumVersion() { return 2; }

  MetadataSource* GetMetadataSource() override { return metadata_source_; }
  MetadataAccessObject* GetMetadataAccessObject() override {
    return metadata_access_object_.get();
  }

  std::optional<int64_t> GetSchemaVersion() final {
    return testing_schema_version_;
  }

  // Returns OK if DB schema passed verification.
  absl::Status VerifyDbSchema(const int64_t version) {
    MetadataSourceQueryConfig::MigrationScheme migration_scheme;
    if (!GetMigrationScheme(version, &migration_scheme).ok()) {
      return absl::InternalError(absl::StrCat("Migration scheme of version ",
                                              version, " is not found"));
    }
    if (!migration_scheme.has_db_verification()) {
      return absl::OkStatus();
    }
    const MetadataSourceQueryConfig::DbVerification& db_verification =
        migration_scheme.db_verification();
    RecordSet record_set;
    if (db_verification.total_num_tables() > 0) {
      ASSIGN_OR_RETURN(int result, GetCountQueryResult(GetTableNumQuery(),
                                                       GetMetadataSource()));
      if (result != db_verification.total_num_tables()) {
        return absl::InternalError(absl::StrCat(
            "Verification failed for version ", version,
            " as total number of tables mismatch, expected: ",
            db_verification.total_num_tables(), ", got: ", result));
      }
    }
    if (db_verification.total_num_indexes() > 0) {
      ASSIGN_OR_RETURN(int result, GetCountQueryResult(GetIndexNumQuery(),
                                                       GetMetadataSource()));
      if (result != db_verification.total_num_indexes()) {
        return absl::InternalError(absl::StrCat(
            "Verification failed for version ", version,
            " as total number of indexes mismatch, expected: ",
            db_verification.total_num_indexes(), ", got: ", result));
      }
    }
    return absl::OkStatus();
  }

  bool SkipSchemaMigrationTests() { return false; }

  // Tests if there is upgrade verification.
  bool HasUpgradeVerification(int64_t version) {
    MetadataSourceQueryConfig::MigrationScheme migration_scheme;
    if (!GetMigrationScheme(version, &migration_scheme).ok()) {
      return false;
    }
    return migration_scheme.has_upgrade_verification();
  }

  // Tests if there is downgrade verification.
  bool HasDowngradeVerification(int64_t version) {
    MetadataSourceQueryConfig::MigrationScheme migration_scheme;
    if (!GetMigrationScheme(version, &migration_scheme).ok()) {
      return false;
    }
    return migration_scheme.has_downgrade_verification();
  }

  // Adds a commit point in the tests.
  absl::Status AddCommitPoint() {
    MLMD_RETURN_IF_ERROR(metadata_source_->Commit());
    MLMD_RETURN_IF_ERROR(metadata_source_->Begin());
    return absl::OkStatus();
  }

  // Resets the test transaction to the empty state.
  // Used after the test transaction commit fails.
  absl::Status ResetForRetry() { return absl::OkStatus(); }

  absl::Status SetupPreviousVersionForDowngrade(int64_t version) {
    MetadataSourceQueryConfig::MigrationScheme migration_scheme;
    MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
        GetMigrationScheme(version, &migration_scheme),
        "Cannot find migration scheme for SetupPreviousVersionForDowngrade");
    for (const auto& query : migration_scheme.downgrade_verification()
                                 .previous_version_setup_queries()) {
      RecordSet dummy_record_set;
      MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
          GetMetadataSource()->ExecuteQuery(query.query(), &dummy_record_set),
          "SetupPreviousVersionForDowngrade query:", query.query());
    }
    return absl::OkStatus();
  }
  absl::Status SetupPreviousVersionForUpgrade(int64_t version) {
    MetadataSourceQueryConfig::MigrationScheme migration_scheme;
    MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
        GetMigrationScheme(version, &migration_scheme),
        "Cannot find migration scheme for SetupPreviousVersionForUpgrade");
    for (const auto& query : migration_scheme.upgrade_verification()
                                 .previous_version_setup_queries()) {
      RecordSet dummy_record_set;
      MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
          GetMetadataSource()->ExecuteQuery(query.query(), &dummy_record_set),
          "Cannot execute query in SetupPreviousVersionForUpgrade: ",
          query.query());
    }
    return absl::OkStatus();
  }
  absl::Status DowngradeVerification(int64_t version) {
    MetadataSourceQueryConfig::MigrationScheme migration_scheme;
    MLMD_RETURN_IF_ERROR(GetMigrationScheme(version, &migration_scheme));
    return Verification(migration_scheme.downgrade_verification()
                            .post_migration_verification_queries());
  }

  absl::Status UpgradeVerification(int64_t version) {
    MetadataSourceQueryConfig::MigrationScheme migration_scheme;
    MLMD_RETURN_IF_ERROR(GetMigrationScheme(version, &migration_scheme));
    return Verification(migration_scheme.upgrade_verification()
                            .post_migration_verification_queries());
  }

  absl::Status DropTypeTable() {
    RecordSet record_set;
    return GetMetadataSource()->ExecuteQuery("DROP TABLE IF EXISTS Type;",
                                             &record_set);
  }

  absl::Status DropArtifactTable() {
    RecordSet record_set;
    return GetMetadataSource()->ExecuteQuery("DROP TABLE Artifact;",
                                             &record_set);
  }

  absl::Status DeleteSchemaVersion() {
    RecordSet record_set;
    return GetMetadataSource()->ExecuteQuery("DELETE FROM MLMDEnv;",
                                             &record_set);
  }
  absl::StatusOr<bool> CheckTableEmpty(absl::string_view table_name) {
    absl::string_view query = R"(SELECT EXISTS (SELECT 1 FROM $0);)";
    RecordSet record_set;
    MLMD_RETURN_IF_ERROR(GetMetadataSource()->ExecuteQuery(
        absl::Substitute(query, table_name), &record_set));

    if (record_set.records_size() != 1) {
      return absl::InternalError(
          absl::StrCat("Failed to check if table ", table_name,
                       " is empty when running query ", query));
    }

    bool result;
    if (!absl::SimpleAtob(record_set.records(0).values(0), &result)) {
      return absl::InternalError(
          absl::StrCat("Value incorrect: ", record_set.records(0).DebugString(),
                       " on query ", query));
    }
    return absl::StatusOr<bool>(result == 0);
  }

  absl::Status SetDatabaseVersionIncompatible() {
    RecordSet record_set;
    MLMD_RETURN_IF_ERROR(GetMetadataSource()->ExecuteQuery(
        "UPDATE MLMDEnv SET schema_version = schema_version + 1;",
        &record_set));
    return absl::OkStatus();
  }
  bool PerformExtendedTests() final { return false; }

  std::string GetTableNumQuery() {
    return " SELECT count(*) FROM information_schema.tables "
           " WHERE table_schema='public';";
  }

  std::string GetIndexNumQuery() {
    return " SELECT count(*) FROM pg_indexes "
           " WHERE schemaname='public';";
  }

 private:
  // Get a migration scheme, or return NOT_FOUND.
  absl::Status GetMigrationScheme(
      int64_t version,
      MetadataSourceQueryConfig::MigrationScheme* migration_scheme) {
    if (config_.migration_schemes().find(version) ==
        config_.migration_schemes().end()) {
      LOG(ERROR) << "Could not find migration scheme for version " << version;
      return absl::NotFoundError(absl::StrCat(
          "Could not find migration scheme for version ", version));
    }
    *migration_scheme = config_.migration_schemes().at(version);
    return absl::OkStatus();
  }

  absl::Status Verification(
      const google::protobuf::RepeatedPtrField<MetadataSourceQueryConfig::TemplateQuery>&
          queries) {
    for (const auto& query : queries) {
      RecordSet record_set;
      MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
          GetMetadataSource()->ExecuteQuery(query.query(), &record_set),
          "query: ", query.query());
      if (record_set.records_size() != 1) {
        return absl::InternalError(
            absl::StrCat("Verification failed on query ", query.query()));
      }
      bool result = false;
      if (!absl::SimpleAtob(record_set.records(0).values(0), &result)) {
        return absl::InternalError(absl::StrCat(
            "Value incorrect:", record_set.records(0).DebugString(),
            " on query ", query.query()));
      }
      if (!result) {
        return absl::InternalError(
            absl::StrCat("Value false ", record_set.records(0).DebugString(),
                         " on query ", query.query()));
      }
    }
    return absl::OkStatus();
  }

  // An unowned TestPostgresqlMetadataSourceInitializer from a call to
  // GetTestPostgresqlMetadataSourceInitializer().
  std::unique_ptr<TestPostgreSQLMetadataSourceInitializer>
      metadata_source_initializer_;
  // An unowned MySqlMetadataSource from a call to
  // metadata_source_initializer->Init().
  PostgreSQLMetadataSource* metadata_source_;
  std::unique_ptr<MetadataAccessObject> metadata_access_object_;

  MetadataSourceQueryConfig config_;
  // If not set, by default, we test against the head version.
  std::optional<int64_t> testing_schema_version_;
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    PostgresqlMetadataAccessObjectTest, MetadataAccessObjectTest,
    ::testing::Values([]() {
      return std::make_unique<PostgresqlMetadataAccessObjectContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata
