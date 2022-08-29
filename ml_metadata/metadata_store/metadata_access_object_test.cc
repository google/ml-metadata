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
#include "ml_metadata/metadata_store/metadata_access_object_test.h"

#include <memory>
#include <vector>

#include "gflags/gflags.h"
#include <glog/logging.h>
#include "google/protobuf/repeated_field.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {
namespace testing {
namespace {

// A utility macros for OSS files as ASSERT_OK is not available in OSS.
#define MLMD_ASSERT_OK(expr) ASSERT_EQ(absl::OkStatus(), expr)

// A utility macros for OSS files as EXPECT_OK is not available in OSS.
#define MLMD_EXPECT_OK(expr) EXPECT_EQ(absl::OkStatus(), expr)

absl::Status GetCountQueryResult(const std::string& query,
                                 MetadataSource* metadata_source, int* result) {
  RecordSet record_set;
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
      metadata_source->ExecuteQuery(query, &record_set), "query: ", query);
  if (record_set.records_size() != 1) {
    return absl::InternalError(
        absl::StrCat("Verification failed on query ", query));
  }
  if (!absl::SimpleAtoi(record_set.records(0).values(0), result)) {
    return absl::InternalError(
        absl::StrCat("Value incorrect:", record_set.records(0).DebugString(),
                     " on query ", query));
  }
  return absl::OkStatus();
}

}  // namespace

// Get a migration scheme, or return NOT_FOUND.
absl::Status QueryConfigMetadataAccessObjectContainer::GetMigrationScheme(
    int64 version,
    MetadataSourceQueryConfig::MigrationScheme* migration_scheme) {
  if (config_.migration_schemes().find(version) ==
      config_.migration_schemes().end()) {
    LOG(ERROR) << "Could not find migration scheme for version " << version;
    return absl::NotFoundError(
        absl::StrCat("Could not find migration scheme for version ", version));
  }
  *migration_scheme = config_.migration_schemes().at(version);
  return absl::OkStatus();
}

bool QueryConfigMetadataAccessObjectContainer::HasUpgradeVerification(
    int64 version) {
  MetadataSourceQueryConfig::MigrationScheme migration_scheme;
  if (!GetMigrationScheme(version, &migration_scheme).ok()) {
    return false;
  }
  return migration_scheme.has_upgrade_verification();
}

absl::Status QueryConfigMetadataAccessObjectContainer::VerifyDbSchema(
    const int64 version) {
  MetadataSourceQueryConfig::MigrationScheme migration_scheme;
  if (!GetMigrationScheme(version, &migration_scheme).ok()) {
    return absl::InternalError(
        absl::StrCat("Migration scheme of version ", version, " is not found"));
  }
  if (!migration_scheme.has_db_verification()) {
    return absl::OkStatus();
  }
  const MetadataSourceQueryConfig::DbVerification& db_verification =
      migration_scheme.db_verification();
  RecordSet record_set;
  if (db_verification.total_num_tables() > 0) {
    int result = 0;
    MLMD_RETURN_IF_ERROR(
        GetCountQueryResult(GetTableNumQuery(), GetMetadataSource(), &result));
    if (result != db_verification.total_num_tables()) {
      return absl::InternalError(
          absl::StrCat("Verification failed for version ", version,
                       " as total number of tables mismatch, expected: ",
                       db_verification.total_num_tables(), ", got: ", result));
    }
  }
  if (db_verification.total_num_indexes() > 0) {
    int result = 0;
    MLMD_RETURN_IF_ERROR(
        GetCountQueryResult(GetIndexNumQuery(), GetMetadataSource(), &result));
    if (result != db_verification.total_num_indexes()) {
      return absl::InternalError(
          absl::StrCat("Verification failed for version ", version,
                       " as total number of indexes mismatch, expected: ",
                       db_verification.total_num_indexes(), ", got: ", result));
    }
  }
  return absl::OkStatus();
}

bool QueryConfigMetadataAccessObjectContainer::HasDowngradeVerification(
    int64 version) {
  MetadataSourceQueryConfig::MigrationScheme migration_scheme;
  if (!GetMigrationScheme(version, &migration_scheme).ok()) {
    return false;
  }
  return migration_scheme.has_downgrade_verification();
}

absl::Status
QueryConfigMetadataAccessObjectContainer::SetupPreviousVersionForUpgrade(
    int64 version) {
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

absl::Status
QueryConfigMetadataAccessObjectContainer::SetupPreviousVersionForDowngrade(
    int64 version) {
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

absl::Status QueryConfigMetadataAccessObjectContainer::DowngradeVerification(
    int64 version) {
  MetadataSourceQueryConfig::MigrationScheme migration_scheme;
  MLMD_RETURN_IF_ERROR(GetMigrationScheme(version, &migration_scheme));
  return Verification(migration_scheme.downgrade_verification()
                          .post_migration_verification_queries());
}

absl::Status QueryConfigMetadataAccessObjectContainer::UpgradeVerification(
    int64 version) {
  MetadataSourceQueryConfig::MigrationScheme migration_scheme;
  MLMD_RETURN_IF_ERROR(GetMigrationScheme(version, &migration_scheme));
  return Verification(migration_scheme.upgrade_verification()
                          .post_migration_verification_queries());
}

absl::Status QueryConfigMetadataAccessObjectContainer::Verification(
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
      return absl::InternalError(
          absl::StrCat("Value incorrect:", record_set.records(0).DebugString(),
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

int64 QueryConfigMetadataAccessObjectContainer::MinimumVersion() { return 1; }

absl::Status QueryConfigMetadataAccessObjectContainer::DropTypeTable() {
  RecordSet record_set;
  return GetMetadataSource()->ExecuteQuery("DROP TABLE IF EXISTS `Type`;",
                                           &record_set);
}

absl::Status QueryConfigMetadataAccessObjectContainer::DropArtifactTable() {
  RecordSet record_set;
  return GetMetadataSource()->ExecuteQuery("DROP TABLE `Artifact`;",
                                           &record_set);
}

absl::Status QueryConfigMetadataAccessObjectContainer::DeleteSchemaVersion() {
  RecordSet record_set;
  return GetMetadataSource()->ExecuteQuery("DELETE FROM `MLMDEnv`;",
                                           &record_set);
}

absl::StatusOr<bool> QueryConfigMetadataAccessObjectContainer::CheckTableEmpty(
    absl::string_view table_name) {
  absl::string_view query = R"(SELECT EXISTS (SELECT 1 FROM $0);)";
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(GetMetadataSource()->ExecuteQuery(
      absl::Substitute(query, table_name), &record_set));

  if (record_set.records_size() != 1) {
    return absl::InternalError(
        absl::StrCat("Failed to check if table ", table_name,
                     " is empty when running query ", query));
  }

  int64 result;
  if (!absl::SimpleAtoi(record_set.records(0).values(0), &result)) {
    return absl::InternalError(
        absl::StrCat("Value incorrect: ", record_set.records(0).DebugString(),
                     " on query ", query));
  }
  return absl::StatusOr<bool>(result == 0);
}

absl::Status
QueryConfigMetadataAccessObjectContainer::SetDatabaseVersionIncompatible() {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(GetMetadataSource()->ExecuteQuery(
      "UPDATE `MLMDEnv` SET `schema_version` = `schema_version` + 1;",
      &record_set));
  return absl::OkStatus();
}

namespace {

using ::ml_metadata::testing::ParseTextProtoOrDie;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedPointwise;

// A utility method creates and stores a type based on the given text proto.
// Returns stored type proto with id.
template <class NodeType>
NodeType CreateTypeFromTextProto(const std::string& type_text_proto,
                                 MetadataAccessObject& metadata_access_object) {
  NodeType type = ParseTextProtoOrDie<NodeType>(type_text_proto);
  int64 type_id;
  CHECK_EQ(absl::OkStatus(), metadata_access_object.CreateType(type, &type_id));
  type.set_id(type_id);
  return type;
}

// Utilities that create and store a node with the given text proto.
// Returns stored node proto with id and timestamps.
template <class Node>
void CreateNodeFromTextProto(
    const std::string& node_text_proto, int64 type_id,
    MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Node& output);

template <>
void CreateNodeFromTextProto(
    const std::string& node_text_proto, int64 type_id,
    MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Artifact& output) {
  Artifact node = ParseTextProtoOrDie<Artifact>(node_text_proto);
  node.set_type_id(type_id);
  int64 node_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.CreateArtifact(node, &node_id));
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_container->AddCommitPoint());
  std::vector<Artifact> nodes;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.FindArtifactsById({node_id}, &nodes));
  ASSERT_THAT(nodes, SizeIs(1));
  output = nodes[0];
}

template <>
void CreateNodeFromTextProto(
    const std::string& node_text_proto, int64 type_id,
    MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Execution& output) {
  Execution node = ParseTextProtoOrDie<Execution>(node_text_proto);
  node.set_type_id(type_id);
  int64 node_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.CreateExecution(node, &node_id));
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_container->AddCommitPoint());
  std::vector<Execution> nodes;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.FindExecutionsById({node_id}, &nodes));
  ASSERT_THAT(nodes, SizeIs(1));
  output = nodes[0];
}

template <>
void CreateNodeFromTextProto(
    const std::string& node_text_proto, int64 type_id,
    MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Context& output) {
  Context node = ParseTextProtoOrDie<Context>(node_text_proto);
  node.set_type_id(type_id);
  int64 node_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.CreateContext(node, &node_id));
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_container->AddCommitPoint());
  std::vector<Context> nodes;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.FindContextsById({node_id}, &nodes));
  ASSERT_THAT(nodes, SizeIs(1));
  output = nodes[0];
}

void CreateEventFromTextProto(
    const std::string& event_text_proto, const Artifact& artifact,
    const Execution& execution, MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Event& output_event) {
  output_event = ParseTextProtoOrDie<Event>(event_text_proto);
  output_event.set_artifact_id(artifact.id());
  output_event.set_execution_id(execution.id());
  int64 dummy_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.CreateEvent(output_event, &dummy_id));
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_container->AddCommitPoint());
}

// Utilities that waits for a millisecond to update a node and returns stored
// node proto with updated timestamps.
template <class Node>
void UpdateAndReturnNode(
    const Node& updated_node, MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Node& output);

template <>
void UpdateAndReturnNode(
    const Artifact& updated_node, MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Artifact& output) {
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object.UpdateArtifact(updated_node));
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_container->AddCommitPoint());
  std::vector<Artifact> artifacts;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object.FindArtifactsById(
                                  {updated_node.id()}, &artifacts));
  ASSERT_THAT(artifacts, SizeIs(1));
  output = artifacts.at(0);
}

template <>
void UpdateAndReturnNode(
    const Execution& updated_node, MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Execution& output) {
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object.UpdateExecution(updated_node));
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_container->AddCommitPoint());
  std::vector<Execution> executions;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object.FindExecutionsById(
                                  {updated_node.id()}, &executions));
  ASSERT_THAT(executions, SizeIs(1));
  output = executions.at(0);
}

template <>
void UpdateAndReturnNode(
    const Context& updated_node, MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Context& output) {
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object.UpdateContext(updated_node));
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_container->AddCommitPoint());
  std::vector<Context> contexts;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object.FindContextsById(
                                  {updated_node.id()}, &contexts));
  ASSERT_THAT(contexts, SizeIs(1));
  output = contexts.at(0);
}

// Set up for FindTypesByIds() related tests.
// `type_1` and `type_2` are initilized, inserted into db and returned.
template <class Type>
absl::Status FindTypesByIdsSetup(MetadataAccessObject& metadata_access_object,
                                 Type& type_1, Type& type_2) {
  type_1 = ParseTextProtoOrDie<Type>(R"pb(
    name: 'test_type_1'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRUCT }
  )pb");
  type_2 = ParseTextProtoOrDie<Type>(R"pb(
    name: 'test_type_2'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
  )pb");
  int64 type_id_1;
  int64 type_id_2;
  MLMD_RETURN_IF_ERROR(metadata_access_object.CreateType(type_1, &type_id_1));
  MLMD_RETURN_IF_ERROR(metadata_access_object.CreateType(type_2, &type_id_2));
  type_1.set_id(type_id_1);
  type_2.set_id(type_id_2);

  return absl::OkStatus();
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceCheckSchemaVersion) {
  // Skip schema/library version consistency check for earlier schema version.
  if (EarlierSchemaEnabled()) { return; }
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 schema_version;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->GetSchemaVersion(&schema_version));
  int64 local_schema_version = metadata_access_object_->GetLibraryVersion();
  EXPECT_EQ(schema_version, local_schema_version);
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceIfNotExists) {
  // Skip empty db init tests for earlier schema version.
  if (EarlierSchemaEnabled()) { return; }
  // creates the schema and insert some records
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->InitMetadataSourceIfNotExists());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Commit());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
  ArtifactType want_type =
      ParseTextProtoOrDie<ArtifactType>("name: 'test_type'");
  int64 type_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type, &type_id));
  // all schema exists, the methods does nothing, check the stored type
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->InitMetadataSourceIfNotExists());
  ArtifactType got_type;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeById(type_id, &got_type));
  EXPECT_THAT(want_type, EqualsProto(got_type, /*ignore_fields=*/{"id"}));
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceIfNotExistsErrorAborted) {
  // Skip empty db init tests for earlier schema version.
  if (EarlierSchemaEnabled() || SkipSchemaMigrationTests()) {
    return;
  }
  // creates the schema and insert some records
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->InitMetadataSourceIfNotExists());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Commit());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
  {
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_container_->DropTypeTable());
    absl::Status s = metadata_access_object_->InitMetadataSourceIfNotExists();
    EXPECT_TRUE(absl::IsAborted(s))
        << "Expected ABORTED but got: " << s.message();
  }
}

TEST_P(MetadataAccessObjectTest, InitForReset) {
  // Skip empty db init tests for earlier schema version.
  if (EarlierSchemaEnabled() || SkipSchemaMigrationTests()) {
    return;
  }
  // Tests if Init() can reset a corrupted database.
  // Not applicable to all databases.
  if (!metadata_access_object_container_->PerformExtendedTests()) {
    return;
  }
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->InitMetadataSourceIfNotExists());
  {
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_container_->DropTypeTable());
  }
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->InitMetadataSource());
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceIfNotExistsErrorAborted2) {
  // Skip partial schema initialization for earlier schema version.
  if (EarlierSchemaEnabled() || SkipSchemaMigrationTests()) {
    return;
  }
  // Drop the artifact table (or artifact property table).
  EXPECT_EQ(absl::OkStatus(), Init());
  {
    // drop a table.
    RecordSet record_set;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_container_->DropArtifactTable());
    absl::Status s = metadata_access_object_->InitMetadataSourceIfNotExists();
    EXPECT_TRUE(absl::IsAborted(s))
        << "Expected ABORTED but got: " << s.message();
  }
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceSchemaVersionMismatch) {
  // Skip schema/library version consistency test for earlier schema version.
  if (EarlierSchemaEnabled() || SkipSchemaMigrationTests()) {
    return;
  }
  // Skip partial schema initialization for earlier schema version.
  if (!metadata_access_object_container_->PerformExtendedTests()) {
    return;
  }
  // creates the schema and insert some records
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->InitMetadataSourceIfNotExists());
  {
    // delete the schema version
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_container_->DeleteSchemaVersion());
    absl::Status s = metadata_access_object_->InitMetadataSourceIfNotExists();
    EXPECT_TRUE(absl::IsAborted(s))
        << "Expected ABORTED but got " << s.message();
  }
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceSchemaVersionMismatch2) {
  // Skip schema/library version consistency test for earlier schema version.
  if (EarlierSchemaEnabled() || SkipSchemaMigrationTests()) {
    return;
  }
  // reset the database by recreating all missing tables
  EXPECT_EQ(absl::OkStatus(), Init());
  {
    // Change the `schema_version` to be a newer version.
    // fails precondition, as older library cannot work with newer db.
    // Note: at present, Version 4 is compatible with Version 5, so I bump
    // this to Version 6.
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_container_->SetDatabaseVersionIncompatible());
    absl::Status s = metadata_access_object_->InitMetadataSourceIfNotExists();
    EXPECT_TRUE(absl::IsFailedPrecondition(s));
  }
}

TEST_P(MetadataAccessObjectTest,
       EarlierSchemaInitMetadataSourceIfNotExistErrorEmptyDB) {
  if (!EarlierSchemaEnabled()) { return; }
  const absl::Status status =
      metadata_access_object_->InitMetadataSourceIfNotExists();
  EXPECT_TRUE(absl::IsFailedPrecondition(status))
      << "Expected FAILED_PRECONDITION but got " << status;
}

TEST_P(MetadataAccessObjectTest,
       EarlierSchemaInitMetadataSourceIfNotExistErrorIncompatibleSchema) {
  if (!EarlierSchemaEnabled()) { return; }
  // Populates an existing db at an incompatible schema_version.
  ASSERT_EQ(absl::OkStatus(), Init());
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_container_->SetDatabaseVersionIncompatible());
  const absl::Status status =
      metadata_access_object_->InitMetadataSourceIfNotExists();
  EXPECT_TRUE(absl::IsFailedPrecondition(status))
      << "Expected FAILED_PRECONDITION but got " << status;
}

TEST_P(MetadataAccessObjectTest, CreateParentTypeInheritanceLink) {
  ASSERT_EQ(absl::OkStatus(), Init());

  {
    // Test: create artifact parent type inheritance link
    const ArtifactType type1 = CreateTypeFromTextProto<ArtifactType>(
        "name: 't1'", *metadata_access_object_);
    const ArtifactType type2 = CreateTypeFromTextProto<ArtifactType>(
        "name: 't2'", *metadata_access_object_);
    // create parent type is ok.
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));
    // recreate the same parent type returns AlreadyExists
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2);
    EXPECT_TRUE(absl::IsAlreadyExists(status));
  }

  {
    // Test: create execution parent type inheritance link
    const ExecutionType type1 = CreateTypeFromTextProto<ExecutionType>(
        "name: 't1'", *metadata_access_object_);
    const ExecutionType type2 = CreateTypeFromTextProto<ExecutionType>(
        "name: 't2'", *metadata_access_object_);
    // create parent type is ok.
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));
    // recreate the same parent type returns AlreadyExists
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2);
    EXPECT_TRUE(absl::IsAlreadyExists(status));
  }

  {
    // Test: create context parent type inheritance link
    const ContextType type1 = CreateTypeFromTextProto<ContextType>(
        "name: 't1'", *metadata_access_object_);
    const ContextType type2 = CreateTypeFromTextProto<ContextType>(
        "name: 't2'", *metadata_access_object_);
    // create parent type is ok.
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));
    // recreate the same parent type returns AlreadyExists
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2);
    EXPECT_TRUE(absl::IsAlreadyExists(status));
  }
}

TEST_P(MetadataAccessObjectTest,
       CreateParentTypeInheritanceLinkInvalidTypeIdError) {
  ASSERT_EQ(absl::OkStatus(), Init());
  const ArtifactType stored_type1 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't1'", *metadata_access_object_);
  const ArtifactType no_id_type1, no_id_type2;

  {
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(no_id_type1,
                                                                 no_id_type2);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  {
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(stored_type1,
                                                                 no_id_type2);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  {
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(no_id_type1,
                                                                 stored_type1);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }
}

TEST_P(MetadataAccessObjectTest, CreateParentTypeInheritanceLinkWithCycle) {
  ASSERT_EQ(absl::OkStatus(), Init());
  const ArtifactType type1 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't1'", *metadata_access_object_);
  const ArtifactType type2 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't2'", *metadata_access_object_);
  const ArtifactType type3 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't3'", *metadata_access_object_);
  const ArtifactType type4 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't4'", *metadata_access_object_);
  const ArtifactType type5 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't4'", *metadata_access_object_);

  {
    // cannot add self as parent.
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type1);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  // type1 -> type2
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));

  {
    // cannot have bi-direction parent
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type2, type1);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  // type1 -> type2 -> type3
  //      \-> type4 -> type5
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type2, type3));
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type1, type4));
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type4, type5));

  {
    // cannot have transitive parent
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type3, type1);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  {
    // cannot have transitive parent
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type5, type1);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }
}

TEST_P(MetadataAccessObjectTest, FindParentTypesByTypeId) {
  ASSERT_EQ(absl::OkStatus(), Init());
  // Setup: init the store with the following types and inheritance links
  // ArtifactType:  type1 -> type2
  //                type3 -> type2
  // ExecutionType: type4 -> type5
  // ContextType:   type6 -> type7
  //                type8
  const ArtifactType type1 = CreateTypeFromTextProto<ArtifactType>(R"(
          name: 't1'
          properties { key: 'property_1' value: STRING }
      )", *metadata_access_object_);
  ArtifactType type2 =
      CreateTypeFromTextProto<ArtifactType>(R"(
          name: 't2'
          properties { key: 'property_2' value: INT }
      )",
                                            *metadata_access_object_);
  ArtifactType type3 =
      CreateTypeFromTextProto<ArtifactType>(R"(
          name: 't3'
          properties { key: 'property_3' value: DOUBLE }
      )",
                                            *metadata_access_object_);
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type3, type2));

  const ExecutionType type4 = CreateTypeFromTextProto<ExecutionType>(R"(
          name: 't4'
          properties { key: 'property_4' value: STRING }
      )", *metadata_access_object_);
  const ExecutionType type5 = CreateTypeFromTextProto<ExecutionType>(R"(
            name: 't5'
        )", *metadata_access_object_);
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type4, type5));

  const ContextType type6 = CreateTypeFromTextProto<ContextType>(R"(
          name: 't6'
          properties { key: 'property_5' value: INT }
          properties { key: 'property_6' value: DOUBLE }
      )", *metadata_access_object_);
  const ContextType type7 = CreateTypeFromTextProto<ContextType>(
      "name: 't7'", *metadata_access_object_);
  const ContextType type8 = CreateTypeFromTextProto<ContextType>(
      "name: 't8'", *metadata_access_object_);
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type6, type7));

  // verify artifact types
  {
    absl::flat_hash_map<int64, ArtifactType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id(), type3.id()}, parent_types));
    // Type properties will not be retrieved in FindParentTypesByTypeId.
    type2.clear_properties();
    type3.clear_properties();
    ASSERT_EQ(parent_types.size(), 2);
    EXPECT_THAT(parent_types[type1.id()], EqualsProto(type2));
    EXPECT_THAT(parent_types[type3.id()], EqualsProto(type2));
  }

  {
    absl::flat_hash_map<int64, ArtifactType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId({type2.id()},
                                                               parent_types));
    EXPECT_THAT(parent_types, IsEmpty());
  }

  // verify execution types
  {
    absl::flat_hash_map<int64, ExecutionType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId({type4.id()},
                                                               parent_types));
    EXPECT_THAT(parent_types[type4.id()], EqualsProto(type5));
  }

  {
    absl::flat_hash_map<int64, ExecutionType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId({type5.id()},
                                                               parent_types));
    EXPECT_THAT(parent_types, IsEmpty());
  }

  // verify context types
  {
    absl::flat_hash_map<int64, ContextType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId({type6.id()},
                                                               parent_types));
    EXPECT_THAT(parent_types[type6.id()], EqualsProto(type7));
  }

  {
    absl::flat_hash_map<int64, ContextType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId({type7.id()},
                                                               parent_types));
    EXPECT_THAT(parent_types, IsEmpty());
  }

  {
    absl::flat_hash_map<int64, ContextType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId({type8.id()},
                                                               parent_types));
    EXPECT_THAT(parent_types, IsEmpty());
  }

  // verify mixed type ids
  {
    absl::flat_hash_map<int64, ArtifactType> parent_types;
    // A mixture of context, exectuion and artifact type ids.
    const auto status = metadata_access_object_->FindParentTypesByTypeId(
        {type1.id(), type4.id(), type6.id()}, parent_types);
    // NOT_FOUND status was returned because `type4` and `type6` are not
    // artifact types and hence will not be found by FindTypesImpl.
    EXPECT_TRUE(absl::IsNotFound(status));
  }
}

TEST_P(MetadataAccessObjectTest, CreateType) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type1 = ParseTextProtoOrDie<ArtifactType>("name: 'test_type'");
  int64 type1_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type1, &type1_id));

  ArtifactType type2 = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type2'
    properties { key: 'property_1' value: STRING })");
  int64 type2_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type2, &type2_id));
  EXPECT_NE(type1_id, type2_id);

  ExecutionType type3 = ParseTextProtoOrDie<ExecutionType>(
      R"(name: 'test_type'
         properties { key: 'property_2' value: INT }
         input_type: { any: {} }
         output_type: { none: {} }
      )");
  int64 type3_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type3, &type3_id));
  EXPECT_NE(type1_id, type3_id);
  EXPECT_NE(type2_id, type3_id);

  ContextType type4 = ParseTextProtoOrDie<ContextType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: STRING })");
  int64 type4_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type4, &type4_id));
  EXPECT_NE(type1_id, type4_id);
  EXPECT_NE(type2_id, type4_id);
  EXPECT_NE(type3_id, type4_id);
}

TEST_P(MetadataAccessObjectTest, StoreTypeWithVersionAndDescriptions) {
  ASSERT_EQ(absl::OkStatus(), Init());
  static char kTypeStr[] = R"(
    name: 'test_type'
    version: 'v1'
    description: 'the type description'
    properties { key: 'stored_property' value: STRING })";

  {
    const ArtifactType want_artifact_type =
        CreateTypeFromTextProto<ArtifactType>(kTypeStr,
                                              *metadata_access_object_);
    ArtifactType got_artifact_type;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindTypeById(want_artifact_type.id(),
                                                    &got_artifact_type));
    EXPECT_THAT(want_artifact_type, EqualsProto(got_artifact_type));
  }

  {
    const ExecutionType want_execution_type =
        CreateTypeFromTextProto<ExecutionType>(kTypeStr,
                                               *metadata_access_object_);
    ExecutionType got_execution_type;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindTypeByNameAndVersion(
                  want_execution_type.name(), want_execution_type.version(),
                  &got_execution_type));
    EXPECT_THAT(want_execution_type, EqualsProto(got_execution_type));
  }

  {
    const ContextType want_context_type = CreateTypeFromTextProto<ContextType>(
        kTypeStr, *metadata_access_object_);
    std::vector<ContextType> got_context_types;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindTypes(&got_context_types));
    EXPECT_THAT(got_context_types, SizeIs(1));
    EXPECT_THAT(want_context_type, EqualsProto(got_context_types[0]));
  }
}

TEST_P(MetadataAccessObjectTest, StoreTypeWithEmptyVersion) {
  ASSERT_EQ(absl::OkStatus(), Init());
  // When the input version = empty string, it is treated as unset.
  static constexpr char kEmptyStringVersionTypeStr[] =
      "name: 'test_type' version: ''";

  {
    const ArtifactType want_artifact_type =
        CreateTypeFromTextProto<ArtifactType>(kEmptyStringVersionTypeStr,
                                              *metadata_access_object_);
    ArtifactType got_artifact_type;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindTypeById(want_artifact_type.id(),
                                                    &got_artifact_type));
    EXPECT_FALSE(got_artifact_type.has_version());
    EXPECT_THAT(want_artifact_type,
                EqualsProto(got_artifact_type, /*ignore_fields=*/{"version"}));
  }

  {
    const ExecutionType want_execution_type =
        CreateTypeFromTextProto<ExecutionType>(kEmptyStringVersionTypeStr,
                                               *metadata_access_object_);
    ExecutionType got_execution_type;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindTypeByNameAndVersion(
                  want_execution_type.name(), want_execution_type.version(),
                  &got_execution_type));
    EXPECT_FALSE(got_execution_type.has_version());
    EXPECT_THAT(want_execution_type,
                EqualsProto(got_execution_type, /*ignore_fields=*/{"version"}));
  }

  {
    const ContextType want_context_type = CreateTypeFromTextProto<ContextType>(
        kEmptyStringVersionTypeStr, *metadata_access_object_);
    std::vector<ContextType> got_context_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindTypes(&got_context_types));
    ASSERT_THAT(got_context_types, SizeIs(1));
    EXPECT_FALSE(got_context_types[0].has_version());
    EXPECT_THAT(want_context_type, EqualsProto(got_context_types[0],
                                               /*ignore_fields=*/{"version"}));
  }
}

TEST_P(MetadataAccessObjectTest, CreateTypeError) {
  ASSERT_EQ(absl::OkStatus(), Init());
  {
    ArtifactType wrong_type;
    int64 type_id;
    // Types must at least have a name.
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->CreateType(wrong_type, &type_id)));
  }
  {
    ArtifactType wrong_type = ParseTextProtoOrDie<ArtifactType>(R"(
      name: 'test_type2'
      properties { key: 'property_1' value: UNKNOWN })");
    int64 type_id;
    // Properties must have type either STRING, DOUBLE, or INT. UNKNOWN
    // is not allowed.
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->CreateType(wrong_type, &type_id)));
  }
}


TEST_P(MetadataAccessObjectTest, UpdateType) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type1 = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'type1'
    properties { key: 'stored_property' value: STRING })");
  int64 type1_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type1, &type1_id));

  ExecutionType type2 = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'type2'
    properties { key: 'stored_property' value: STRING })");
  int64 type2_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type2, &type2_id));

  ContextType type3 = ParseTextProtoOrDie<ContextType>(R"(
    name: 'type3'
    properties { key: 'stored_property' value: STRING })");
  int64 type3_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type3, &type3_id));

  ArtifactType want_type1;
  want_type1.set_id(type1_id);
  want_type1.set_name("type1");
  (*want_type1.mutable_properties())["stored_property"] = STRING;
  (*want_type1.mutable_properties())["new_property"] = INT;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->UpdateType(want_type1));

  ArtifactType got_type1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeById(type1_id, &got_type1));
  EXPECT_THAT(want_type1, EqualsProto(got_type1));

  // update properties may not include all existing properties
  ExecutionType want_type2;
  want_type2.set_name("type2");
  (*want_type2.mutable_properties())["new_property"] = DOUBLE;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->UpdateType(want_type2));

  ExecutionType got_type2;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeById(type2_id, &got_type2));
  (*want_type2.mutable_properties())["stored_property"] = STRING;
  EXPECT_THAT(want_type2, EqualsProto(got_type2, /*ignore_fields=*/{"id"}));

  // update context type
  ContextType want_type3;
  want_type3.set_name("type3");
  (*want_type3.mutable_properties())["new_property"] = STRING;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->UpdateType(want_type3));
  ContextType got_type3;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeById(type3_id, &got_type3));
  (*want_type3.mutable_properties())["stored_property"] = STRING;
  EXPECT_THAT(want_type3, EqualsProto(got_type3, /*ignore_fields=*/{"id"}));
}

TEST_P(MetadataAccessObjectTest, UpdateTypeError) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'stored_type'
    properties { key: 'stored_property' value: STRING })");
  int64 type_id;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));
  {
    ArtifactType type_without_name;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->UpdateType(type_without_name)));
  }
  {
    ArtifactType type_with_wrong_id;
    type_with_wrong_id.set_name("stored_type");
    type_with_wrong_id.set_id(type_id + 1);
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->UpdateType(type_with_wrong_id)));
  }
  {
    ArtifactType type_with_modified_property_type;
    type_with_modified_property_type.set_id(type_id);
    type_with_modified_property_type.set_name("stored_type");
    (*type_with_modified_property_type
          .mutable_properties())["stored_property"] = INT;
    EXPECT_TRUE(absl::IsAlreadyExists(
        metadata_access_object_->UpdateType(type_with_modified_property_type)));
  }
  {
    ArtifactType type_with_unknown_type_property;
    type_with_unknown_type_property.set_id(type_id);
    type_with_unknown_type_property.set_name("stored_type");
    (*type_with_unknown_type_property.mutable_properties())["new_property"] =
        UNKNOWN;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->UpdateType(type_with_unknown_type_property)));
  }
}

TEST_P(MetadataAccessObjectTest, FindTypeByIdArtifact) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType want_type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRUCT }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type, &type_id));

  ArtifactType got_type;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeById(type_id, &got_type));
  EXPECT_THAT(want_type, EqualsProto(got_type, /*ignore_fields=*/{"id"}));

  // type_id is for an artifact type, not an execution/context type.
  ExecutionType execution_type;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindTypeById(type_id, &execution_type)));
  ContextType context_type;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindTypeById(type_id, &context_type)));
}

TEST_P(MetadataAccessObjectTest, FindTypeByIdContext) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType want_type = ParseTextProtoOrDie<ContextType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type, &type_id));

  ContextType got_type;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeById(type_id, &got_type));
  EXPECT_THAT(want_type, EqualsProto(got_type, /*ignore_fields=*/{"id"}));

  // type_id is for a context type, not an artifact/execution type.
  ArtifactType artifact_type;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindTypeById(type_id, &artifact_type)));
  ExecutionType execution_type;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindTypeById(type_id, &execution_type)));
}

TEST_P(MetadataAccessObjectTest, FindTypeByIdExecution) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType want_type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    input_type: { any: {} }
    output_type: { none: {} }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type, &type_id));

  ExecutionType got_type;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeById(type_id, &got_type));
  EXPECT_THAT(want_type, EqualsProto(got_type, /*ignore_fields=*/{"id"}));

  // This type_id is an execution type, not an artifact/context type.
  ArtifactType artifact_type;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindTypeById(type_id, &artifact_type)));
  ContextType context_type;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindTypeById(type_id, &context_type)));
}

TEST_P(MetadataAccessObjectTest, FindTypeByIdExecutionUnicode) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType want_type;
  want_type.set_name("пример_типа");
  (*want_type.mutable_properties())["привет"] = INT;
  (*want_type.mutable_input_type()
        ->mutable_dict()
        ->mutable_properties())["пример"]
      .mutable_any();
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type, &type_id));

  ExecutionType got_type;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeById(type_id, &got_type));
  EXPECT_THAT(want_type, EqualsProto(got_type, /*ignore_fields=*/{"id"}));

  // This type_id is an execution type, not an artifact/context type.
  ArtifactType artifact_type;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindTypeById(type_id, &artifact_type)));
  ContextType context_type;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindTypeById(type_id, &context_type)));
}

// Test if an execution type can be stored without input_type and output_type.
TEST_P(MetadataAccessObjectTest, FindTypeByIdExecutionNoSignature) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType want_type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type, &type_id));

  ExecutionType got_type;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeById(type_id, &got_type));
  EXPECT_THAT(want_type, EqualsProto(got_type, /*ignore_fields=*/{"id"}));

  // This type_id is an execution type, not an artifact/context type.
  ArtifactType artifact_type;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindTypeById(type_id, &artifact_type)));
  ContextType context_type;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindTypeById(type_id, &context_type)));
}

TEST_P(MetadataAccessObjectTest, FindTypeByName) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType want_type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    input_type: { any: {} }
    output_type: { none: {} }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type, &type_id));

  ExecutionType got_type;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeByNameAndVersion(
                "test_type", /*version=*/absl::nullopt, &got_type));
  EXPECT_THAT(want_type, EqualsProto(got_type, /*ignore_fields=*/{"id"}));

  // The type with this name is an execution type, not an artifact/context type.
  ArtifactType artifact_type;
  EXPECT_TRUE(
      absl::IsNotFound(metadata_access_object_->FindTypeByNameAndVersion(
          "test_type", /*version=*/absl::nullopt, &artifact_type)));
  ContextType context_type;
  EXPECT_TRUE(
      absl::IsNotFound(metadata_access_object_->FindTypeByNameAndVersion(
          "test_type", /*version=*/absl::nullopt, &context_type)));
}


TEST_P(MetadataAccessObjectTest, FindTypeIdByNameAndVersion) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType want_type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )");
  int64 v0_type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type, &v0_type_id));

  int64 v0_got_type_id;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeIdByNameAndVersion(
                "test_type", /*version=*/absl::nullopt,
                TypeKind::EXECUTION_TYPE, &v0_got_type_id));
  EXPECT_EQ(v0_got_type_id, v0_type_id);

  want_type.set_version("v1");
  int64 v1_type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type, &v1_type_id));
  int64 v1_got_type_id;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeIdByNameAndVersion(
                "test_type", "v1",
                TypeKind::EXECUTION_TYPE, &v1_got_type_id));
  EXPECT_EQ(v1_got_type_id, v1_type_id);


  // The type with this name is an execution type, not an artifact/context type.
  int64 got_type_id;
  EXPECT_TRUE(
      absl::IsNotFound(metadata_access_object_->FindTypeIdByNameAndVersion(
          "test_type", /*version=*/absl::nullopt, TypeKind::ARTIFACT_TYPE,
          &got_type_id)));
  EXPECT_TRUE(
      absl::IsNotFound(metadata_access_object_->FindTypeIdByNameAndVersion(
          "test_type", /*version=*/absl::nullopt, TypeKind::CONTEXT_TYPE,
          &got_type_id)));
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsArtifactSuccess) {
  MLMD_ASSERT_OK(Init());
  ArtifactType want_type_1;
  ArtifactType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));

  std::vector<ArtifactType> got_types;
  MLMD_ASSERT_OK(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, got_types));
  EXPECT_THAT(got_types,
              ElementsAre(EqualsProto(want_type_1, /*ignore_fields=*/{"id"}),
                          EqualsProto(want_type_2, /*ignore_fields=*/{"id"})));
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsArtifactInvalidInput) {
  MLMD_ASSERT_OK(Init());
  ArtifactType want_type_1;
  ArtifactType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));
  std::vector<ArtifactType> got_types;

  // Returns INVALID_ARGUMENT error if `type_ids` is empty or `a/e/c_types` is
  // not empty.
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->FindTypesByIds({}, got_types)));
  EXPECT_THAT(got_types, IsEmpty());
  // Makes `a/e/c_types` to be not empty.
  got_types.push_back(want_type_1);
  EXPECT_TRUE(absl::IsInvalidArgument(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, got_types)));
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsArtifactNotFound) {
  MLMD_ASSERT_OK(Init());
  ArtifactType want_type_1;
  ArtifactType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));
  std::vector<ArtifactType> got_types;

  // Returns NOT_FOUND error if any of the id cannot be found.
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id() + 123, want_type_2.id()}, got_types)));
  got_types.clear();
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id() + 123}, got_types)));
}
TEST_P(MetadataAccessObjectTest, FindTypesByIdsArtifactInvalidTypeKind) {
  MLMD_ASSERT_OK(Init());
  ArtifactType want_type_1;
  ArtifactType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));

  // type_ids are for artifact types, not execution/context types.
  std::vector<ExecutionType> execution_types;
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, execution_types)));
  EXPECT_THAT(execution_types, IsEmpty());
  std::vector<ContextType> context_types;
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, context_types)));
  EXPECT_THAT(context_types, IsEmpty());
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsExecutionSuccess) {
  MLMD_ASSERT_OK(Init());
  ExecutionType want_type_1;
  ExecutionType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));

  std::vector<ExecutionType> got_types;
  MLMD_ASSERT_OK(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, got_types));
  EXPECT_THAT(got_types,
              ElementsAre(EqualsProto(want_type_1, /*ignore_fields=*/{"id"}),
                          EqualsProto(want_type_2, /*ignore_fields=*/{"id"})));
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsExecutionInvalidInput) {
  MLMD_ASSERT_OK(Init());
  ExecutionType want_type_1;
  ExecutionType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));
  std::vector<ExecutionType> got_types;

  // Returns INVALID_ARGUMENT error if `type_ids` is empty or `a/e/c_types` is
  // not empty.
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->FindTypesByIds({}, got_types)));
  EXPECT_THAT(got_types, IsEmpty());
  // Makes `a/e/c_types` to be not empty.
  got_types.push_back(want_type_1);
  EXPECT_TRUE(absl::IsInvalidArgument(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, got_types)));
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsExecutionNotFound) {
  MLMD_ASSERT_OK(Init());
  ExecutionType want_type_1;
  ExecutionType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));
  std::vector<ExecutionType> got_types;

  // Returns NOT_FOUND error if any of the id cannot be found.
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id() + 123, want_type_2.id()}, got_types)));
  got_types.clear();
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id() + 123}, got_types)));
}
TEST_P(MetadataAccessObjectTest, FindTypesByIdsExecutionInvalidTypeKind) {
  MLMD_ASSERT_OK(Init());
  ExecutionType want_type_1;
  ExecutionType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));

  // type_ids are for execution types, not artifact/context types.
  std::vector<ArtifactType> artifact_types;
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, artifact_types)));
  EXPECT_THAT(artifact_types, IsEmpty());
  std::vector<ContextType> context_types;
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, context_types)));
  EXPECT_THAT(context_types, IsEmpty());
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsContextSuccess) {
  MLMD_ASSERT_OK(Init());
  ContextType want_type_1;
  ContextType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));

  std::vector<ContextType> got_types;
  MLMD_ASSERT_OK(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, got_types));
  EXPECT_THAT(got_types,
              ElementsAre(EqualsProto(want_type_1, /*ignore_fields=*/{"id"}),
                          EqualsProto(want_type_2, /*ignore_fields=*/{"id"})));
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsContextInvalidInput) {
  MLMD_ASSERT_OK(Init());
  ContextType want_type_1;
  ContextType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));
  std::vector<ContextType> got_types;

  // Returns INVALID_ARGUMENT error if `type_ids` is empty or `a/e/c_types` is
  // not empty.
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->FindTypesByIds({}, got_types)));
  EXPECT_THAT(got_types, IsEmpty());
  // Makes `a/e/c_types` to be not empty.
  got_types.push_back(want_type_1);
  EXPECT_TRUE(absl::IsInvalidArgument(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, got_types)));
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsContextNotFound) {
  MLMD_ASSERT_OK(Init());
  ContextType want_type_1;
  ContextType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));
  std::vector<ContextType> got_types;

  // Returns NOT_FOUND error if any of the id cannot be found.
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id() + 123, want_type_2.id()}, got_types)));
  got_types.clear();
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id() + 123}, got_types)));
}
TEST_P(MetadataAccessObjectTest, FindTypesByIdsContextInvalidTypeKind) {
  MLMD_ASSERT_OK(Init());
  ContextType want_type_1;
  ContextType want_type_2;
  MLMD_ASSERT_OK(
      FindTypesByIdsSetup(*metadata_access_object_, want_type_1, want_type_2));

  // type_ids are for context types, not artifact/execution types.
  std::vector<ArtifactType> artifact_types;
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, artifact_types)));
  EXPECT_THAT(artifact_types, IsEmpty());
  std::vector<ExecutionType> execution_types;
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, execution_types)));
  EXPECT_THAT(execution_types, IsEmpty());
}

// Test if an execution type can be stored without input_type and output_type.
TEST_P(MetadataAccessObjectTest, FindTypeByNameNoSignature) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType want_type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type, &type_id));
  want_type.set_id(type_id);

  ExecutionType got_type;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindTypeByNameAndVersion(
                "test_type", /*version=*/absl::nullopt, &got_type));
  EXPECT_THAT(want_type, EqualsProto(got_type, /*ignore_fields=*/{"id"}));

  // The type with this name is an execution type, not an artifact/context type.
  ArtifactType artifact_type;
  EXPECT_TRUE(
      absl::IsNotFound(metadata_access_object_->FindTypeByNameAndVersion(
          "test_type", /*version=*/absl::nullopt, &artifact_type)));
  ContextType context_type;
  EXPECT_TRUE(
      absl::IsNotFound(metadata_access_object_->FindTypeByNameAndVersion(
          "test_type", /*version=*/absl::nullopt, &context_type)));
}

TEST_P(MetadataAccessObjectTest, FindAllArtifactTypes) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType want_type_1 = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type_1'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type_1, &type_id));
  want_type_1.set_id(type_id);

  ArtifactType want_type_2 = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type_2'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_5' value: STRING }
  )");
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type_2, &type_id));
  want_type_2.set_id(type_id);

  // No properties.
  ArtifactType want_type_3 = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'no_properties_type'
  )");
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type_3, &type_id));
  want_type_3.set_id(type_id);

  std::vector<ArtifactType> got_types;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindTypes(&got_types));
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2),
                                              EqualsProto(want_type_3)));
}

TEST_P(MetadataAccessObjectTest, FindAllExecutionTypes) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType want_type_1 = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type_1'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type_1, &type_id));
  want_type_1.set_id(type_id);

  ExecutionType want_type_2 = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type_2'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_5' value: STRING }
  )");
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type_2, &type_id));
  want_type_2.set_id(type_id);

  // No properties.
  ExecutionType want_type_3 = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'no_properties_type'
  )");
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type_3, &type_id));
  want_type_3.set_id(type_id);

  std::vector<ExecutionType> got_types;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindTypes(&got_types));
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2),
                                              EqualsProto(want_type_3)));
}

TEST_P(MetadataAccessObjectTest, FindAllContextTypes) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType want_type_1 = ParseTextProtoOrDie<ContextType>(R"(
    name: 'test_type_1'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type_1, &type_id));
  want_type_1.set_id(type_id);

  ContextType want_type_2 = ParseTextProtoOrDie<ContextType>(R"(
    name: 'test_type_2'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_5' value: STRING }
  )");
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type_2, &type_id));
  want_type_2.set_id(type_id);

  // No properties.
  ContextType want_type_3 = ParseTextProtoOrDie<ContextType>(R"(
    name: 'no_properties_type'
  )");
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(want_type_3, &type_id));
  want_type_3.set_id(type_id);

  std::vector<ContextType> got_types;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindTypes(&got_types));
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2),
                                              EqualsProto(want_type_3)));
}

TEST_P(MetadataAccessObjectTest, CreateArtifact) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type_with_predefined_property'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRUCT }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    properties {
      key: 'property_4'
      value: {
        struct_value {
          fields {
            key: "json number"
            value { number_value: 1234 }
          }
          fields {
            key: "json object"
            value {
              struct_value {
                fields {
                  key: "nested json key"
                  value { string_value: "string value" }
                }
              }
            }
          }
        }
      }
    }
  )");
  artifact.set_type_id(type_id);

  int64 artifact1_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact, &artifact1_id));
  int64 artifact2_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact, &artifact2_id));
  EXPECT_NE(artifact1_id, artifact2_id);
}

TEST_P(MetadataAccessObjectTest, CreateArtifactWithCustomProperty) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 type_id = InsertType<ArtifactType>("test_type_with_custom_property");
  Artifact artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri'
    custom_properties {
      key: 'custom_property_1'
      value: { int_value: 3 }
    }
    custom_properties {
      key: 'custom_property_2'
      value: { double_value: 3.0 }
    }
    custom_properties {
      key: 'custom_property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'property_4'
      value: {
        struct_value {
          fields {
            key: "json number"
            value { number_value: 1234 }
          }
          fields {
            key: "json object"
            value {
              struct_value {
                fields {
                  key: "nested json key"
                  value { string_value: "string value" }
                }
              }
            }
          }
        }
      }
    }
  )");
  artifact.set_type_id(type_id);

  int64 artifact1_id, artifact2_id;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact, &artifact1_id));
  EXPECT_EQ(artifact1_id, 1);
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact, &artifact2_id));
  EXPECT_EQ(artifact2_id, 2);
}

TEST_P(MetadataAccessObjectTest, CreateArtifactError) {
  ASSERT_EQ(absl::OkStatus(), Init());

  // unknown type specified
  Artifact artifact;
  int64 artifact_id;
  absl::Status s =
      metadata_access_object_->CreateArtifact(artifact, &artifact_id);
  EXPECT_TRUE(absl::IsInvalidArgument(s));

  artifact.set_type_id(1);
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->CreateArtifact(artifact, &artifact_id)));
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type_disallow_custom_property'
    properties { key: 'property_1' value: INT }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  // type mismatch
  Artifact artifact3;
  artifact3.set_type_id(type_id);
  (*artifact3.mutable_properties())["property_1"].set_string_value("3");
  int64 artifact3_id;
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->CreateArtifact(artifact3, &artifact3_id)));
}

TEST_P(MetadataAccessObjectTest, CreateArtifactWithDuplicatedNameError) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type;
  type.set_name("test_type");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact artifact;
  artifact.set_type_id(type_id);
  artifact.set_name("test artifact name");
  int64 artifact_id;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact, &artifact_id));
  // insert the same artifact again to check the unique constraint
  absl::Status unique_constraint_violation_status =
      metadata_access_object_->CreateArtifact(artifact, &artifact_id);
  EXPECT_EQ(CheckUniqueConstraintAndResetTransaction(
                unique_constraint_violation_status),
            absl::OkStatus());
}


TEST_P(MetadataAccessObjectTest, CreateArtifactWithoutValidation) {
  MLMD_ASSERT_OK(Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64 type_id;
  MLMD_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  // Inserts artifact without validation since the type are known to exist and
  // the artifact's properties are matched with its type.
  Artifact artifact = ParseTextProtoOrDie<Artifact>(R"pb(
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { string_value: '3' }
    }
  )pb");
  artifact.set_type_id(type_id);
  int64 artifact_id;
  EXPECT_EQ(
      metadata_access_object_->CreateArtifact(
          artifact, /*skip_type_and_property_validation=*/true, &artifact_id),
      absl::OkStatus());

  // Inserts artifact with invalid type id without validation.
  // TODO(b/197686185) this test would fail once the foreigen key contrainsts
  // are introduced, remove it at that time.
  Artifact artifact_with_invalid_type = ParseTextProtoOrDie<Artifact>(R"pb(
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { string_value: '3' }
    }
  )pb");
  artifact_with_invalid_type.set_type_id(type_id + 123);
  int64 artifact_id_with_invalid_type;
  EXPECT_EQ(metadata_access_object_->CreateArtifact(
                artifact_with_invalid_type,
                /*skip_type_and_property_validation=*/true,
                &artifact_id_with_invalid_type),
            absl::OkStatus());

  // Inserts artifact with unmatched property with its type(e.g., inserting a
  // double to a string property defined in type) without validation.
  Artifact artifact_with_unmatched_property =
      ParseTextProtoOrDie<Artifact>(R"pb(
        properties {
          key: 'property_1'
          value: { int_value: 3 }
        }
        properties {
          key: 'property_2'
          value: { double_value: 3.0 }
        }
      )pb");
  artifact_with_unmatched_property.set_type_id(type_id);
  int64 artifact_id_with_unmatched_property;
  EXPECT_EQ(metadata_access_object_->CreateArtifact(
                artifact_with_unmatched_property,
                /*skip_type_and_property_validation=*/true,
                &artifact_id_with_unmatched_property),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateArtifactWithCustomTimestamp) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type_with_predefined_property'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRUCT }
  )pb");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact artifact = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    properties {
      key: 'property_4'
      value: {
        struct_value {
          fields {
            key: "json number"
            value { number_value: 1234 }
          }
          fields {
            key: "json object"
            value {
              struct_value {
                fields {
                  key: "nested json key"
                  value { string_value: "string value" }
                }
              }
            }
          }
        }
      }
    }
  )pb");
  artifact.set_type_id(type_id);

  int64 artifact_id = -1;
  absl::Time create_time = absl::InfinitePast();

  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(
                artifact, /*skip_type_and_property_validation=*/false,
                create_time, &artifact_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Artifact got_artifact;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsById(
                                    {artifact_id}, &artifacts));
    got_artifact = artifacts.at(0);
  }

  EXPECT_EQ(got_artifact.create_time_since_epoch(),
            absl::ToUnixMillis(create_time));
}

TEST_P(MetadataAccessObjectTest, CreateExecutionWithoutValidation) {
  MLMD_ASSERT_OK(Init());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64 type_id;
  MLMD_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  // Inserts execution without validation since the type are known to exist and
  // the execution's properties are matched with its type.
  Execution execution = ParseTextProtoOrDie<Execution>(R"pb(
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { string_value: '3' }
    }
  )pb");
  execution.set_type_id(type_id);
  int64 execution_id;
  EXPECT_EQ(
      metadata_access_object_->CreateExecution(
          execution, /*skip_type_and_property_validation=*/true, &execution_id),
      absl::OkStatus());

  // Inserts execution with invalid type id without validation.
  // TODO(b/197686185) this test would fail once the foreigen key contrainsts
  // are introduced, remove it at that time.
  Execution execution_with_invalid_type = ParseTextProtoOrDie<Execution>(R"pb(
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { string_value: '3' }
    }
  )pb");
  execution_with_invalid_type.set_type_id(type_id + 123);
  int64 execution_id_with_invalid_type;
  EXPECT_EQ(metadata_access_object_->CreateExecution(
                execution_with_invalid_type,
                /*skip_type_and_property_validation=*/true,
                &execution_id_with_invalid_type),
            absl::OkStatus());

  // Inserts execution with unmatched property with its type(e.g., inserting a
  // double to a string property defined in type) without validation.
  Execution execution_with_unmatched_property =
      ParseTextProtoOrDie<Execution>(R"pb(
        properties {
          key: 'property_1'
          value: { int_value: 3 }
        }
        properties {
          key: 'property_2'
          value: { double_value: 3.0 }
        }
      )pb");
  execution_with_unmatched_property.set_type_id(type_id);
  int64 execution_id_with_unmatched_property;
  EXPECT_EQ(metadata_access_object_->CreateExecution(
                execution_with_unmatched_property,
                /*skip_type_and_property_validation=*/true,
                &execution_id_with_unmatched_property),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateExecutionWithCustomTimestamp) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64 type_id;
  MLMD_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  Execution execution = ParseTextProtoOrDie<Execution>(R"pb(
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { string_value: '3' }
    }
  )pb");
  execution.set_type_id(type_id);

  int64 execution_id = -1;
  absl::Time create_time = absl::InfinitePast();

  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateExecution(
                execution, /*skip_type_and_property_validation=*/false,
                create_time, &execution_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Execution got_execution;
  {
    std::vector<Execution> executions;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsById(
                                    {execution_id}, &executions));
    got_execution = executions.at(0);
  }

  EXPECT_EQ(got_execution.create_time_since_epoch(),
            absl::ToUnixMillis(create_time));
}

TEST_P(MetadataAccessObjectTest, CreateContextWithoutValidation) {
  MLMD_ASSERT_OK(Init());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64 type_id;
  MLMD_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  // Inserts context without validation since the type are known to exist and
  // the context's properties are matched with its type.
  Context context = ParseTextProtoOrDie<Context>(R"pb(
    name: 'test_context_1'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { string_value: '3' }
    }
  )pb");
  context.set_type_id(type_id);
  int64 context_id;
  EXPECT_EQ(
      metadata_access_object_->CreateContext(
          context, /*skip_type_and_property_validation=*/true, &context_id),
      absl::OkStatus());

  // Inserts context with invalid type id without validation.
  // TODO(b/197686185) this test would fail once the foreigen key contrainsts
  // are introduced, remove it at that time.
  Context context_with_invalid_type = ParseTextProtoOrDie<Context>(R"pb(
    name: 'test_context_2'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { string_value: '3' }
    }
  )pb");
  context_with_invalid_type.set_type_id(type_id + 123);
  int64 context_id_with_invalid_type;
  EXPECT_EQ(metadata_access_object_->CreateContext(
                context_with_invalid_type,
                /*skip_type_and_property_validation=*/true,
                &context_id_with_invalid_type),
            absl::OkStatus());

  // Inserts context with unmatched property with its type(e.g., inserting a
  // double to a string property defined in type) without validation.
  Context context_with_unmatched_property = ParseTextProtoOrDie<Context>(R"pb(
    name: 'test_context_3'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
  )pb");
  context_with_unmatched_property.set_type_id(type_id);
  int64 context_id_with_unmatched_property;
  EXPECT_EQ(metadata_access_object_->CreateContext(
                context_with_unmatched_property,
                /*skip_type_and_property_validation=*/true,
                &context_id_with_unmatched_property),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateContextWithCustomTimestamp) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64 type_id;
  MLMD_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  Context context = ParseTextProtoOrDie<Context>(R"pb(
    name: 'test_context_1'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { string_value: '3' }
    }
  )pb");
  context.set_type_id(type_id);

  int64 context_id = -1;
  absl::Time create_time = absl::InfinitePast();

  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(
                context, /*skip_type_and_property_validation=*/false,
                create_time, &context_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Context got_context;
  {
    std::vector<Context> contexts;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {context_id}, &contexts));
    got_context = contexts.at(0);
  }

  EXPECT_EQ(got_context.create_time_since_epoch(),
            absl::ToUnixMillis(create_time));
}

TEST_P(MetadataAccessObjectTest, FindArtifactById) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact want_artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )");
  want_artifact.set_type_id(type_id);
  int64 artifact_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  want_artifact, &artifact_id));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Artifact got_artifact;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsById(
                                    {artifact_id}, &artifacts));
    got_artifact = artifacts.at(0);
  }
  EXPECT_THAT(got_artifact, EqualsProto(want_artifact, /*ignore_fields=*/{
                                            "id", "create_time_since_epoch",
                                            "last_update_time_since_epoch"}));
  EXPECT_GT(got_artifact.create_time_since_epoch(), 0);
  EXPECT_GT(got_artifact.last_update_time_since_epoch(), 0);
  EXPECT_LE(got_artifact.last_update_time_since_epoch(),
            absl::ToUnixMillis(absl::Now()));
  EXPECT_GE(got_artifact.last_update_time_since_epoch(),
            got_artifact.create_time_since_epoch());
}

TEST_P(MetadataAccessObjectTest, FindArtifacts) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  constexpr absl::string_view kArtifactTemplate = R"(
    uri: 'testuri://testing/uri'
    type_id: %d
    properties {
      key: 'property_1'
      value: { int_value: %d }
    }
    properties {
      key: 'property_2'
      value: { double_value: %f }
    }
    properties {
      key: 'property_3'
      value: { string_value: '%s' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '%s' }
    }
  )";

  Artifact want_artifact1 = ParseTextProtoOrDie<Artifact>(
      absl::StrFormat(kArtifactTemplate, type_id, 1, 2.0, "3", "4"));
  {
    int64 artifact1_id;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                    want_artifact1, &artifact1_id));
    want_artifact1.set_id(artifact1_id);
  }

  Artifact want_artifact2 = ParseTextProtoOrDie<Artifact>(
      absl::StrFormat(kArtifactTemplate, type_id, 11, 12.0, "13", "14"));
  {
    int64 artifact2_id;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                    want_artifact2, &artifact2_id));
    want_artifact2.set_id(artifact2_id);
  }
  ASSERT_NE(want_artifact1.id(), want_artifact2.id());

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Test: retrieve by empty ids
  {
    std::vector<Artifact> got_artifacts;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindArtifactsById({}, &got_artifacts));
    EXPECT_THAT(got_artifacts, IsEmpty());
  }
  // Test: retrieve by unknown id
  const int64 unknown_id = want_artifact1.id() + want_artifact2.id() + 1;
  {
    std::vector<Artifact> got_artifacts;
    EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindArtifactsById(
        {unknown_id}, &got_artifacts)));
  }
  {
    std::vector<Artifact> got_artifacts;
    EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindArtifactsById(
        {unknown_id, want_artifact1.id()}, &got_artifacts)));
  }
  // Test: retrieve by id(s)
  {
    std::vector<Artifact> got_artifacts;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsById(
                                    {want_artifact1.id()}, &got_artifacts));
    EXPECT_THAT(got_artifacts,
                ElementsAre(EqualsProto(
                    want_artifact1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  {
    std::vector<Artifact> got_artifacts;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsById(
                                    {want_artifact2.id()}, &got_artifacts));
    EXPECT_THAT(got_artifacts,
                ElementsAre(EqualsProto(
                    want_artifact2,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  {
    std::vector<Artifact> got_artifacts;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindArtifactsById(
                  {want_artifact1.id(), want_artifact2.id()}, &got_artifacts));
    EXPECT_THAT(
        got_artifacts,
        UnorderedElementsAre(
            EqualsProto(want_artifact1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(want_artifact2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  // Test: Get all artifacts
  {
    std::vector<Artifact> got_artifacts;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindArtifacts(&got_artifacts));
    EXPECT_THAT(
        got_artifacts,
        UnorderedElementsAre(
            EqualsProto(want_artifact1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(want_artifact2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
}

TEST_P(MetadataAccessObjectTest, ListArtifactsInvalidPageSize) {
  ASSERT_EQ(absl::OkStatus(), Init());
  const ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: -1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  std::vector<Artifact> unused_artifacts;
  std::string unused_next_page_token;
  EXPECT_TRUE(absl::IsInvalidArgument(metadata_access_object_->ListArtifacts(
      list_options, &unused_artifacts, &unused_next_page_token)));
}

// A util to test ListArtifact/Execution/Context with filter query.
template <class Node>
void ListNode(MetadataAccessObject& metadata_access_object,
              const ListOperationOptions& options, std::vector<Node>& nodes,
              std::string& next_page_token);
template <>
void ListNode(MetadataAccessObject& metadata_access_object,
              const ListOperationOptions& options, std::vector<Artifact>& nodes,
              std::string& next_page_token) {
  ASSERT_EQ(absl::OkStatus(), metadata_access_object.ListArtifacts(
                                  options, &nodes, &next_page_token));
}

template <>
void ListNode(MetadataAccessObject& metadata_access_object,
              const ListOperationOptions& options,
              std::vector<Execution>& nodes, std::string& next_page_token) {
  ASSERT_EQ(absl::OkStatus(), metadata_access_object.ListExecutions(
                                  options, &nodes, &next_page_token));
}

template <>
void ListNode(MetadataAccessObject& metadata_access_object,
              const ListOperationOptions& options, std::vector<Context>& nodes,
              std::string& next_page_token) {
  ASSERT_EQ(absl::OkStatus(), metadata_access_object.ListContexts(
                                  options, &nodes, &next_page_token));
}

// Apply the list options to list the `Node`, and compare with the `want_nodes`
// point-wise in order.
template <class Node>
void VerifyListOptions(const std::string& list_option_text_proto,
                       MetadataAccessObject& metadata_access_object,
                       const std::vector<Node>& want_nodes) {
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(list_option_text_proto);
  LOG(INFO) << "Testing with list options: " << list_options.DebugString();
  std::string next_page_token;
  std::vector<Node> got_nodes;
  ListNode(metadata_access_object, list_options, got_nodes, next_page_token);
  EXPECT_THAT(got_nodes, Pointwise(EqualsProto<Node>(), want_nodes));
}

TEST_P(MetadataAccessObjectTest, ListArtifactsFilterAttributeQuery) {
  ASSERT_EQ(absl::OkStatus(), Init());
  const ArtifactType type = CreateTypeFromTextProto<ArtifactType>(
      "name: 't1'", *metadata_access_object_);
  std::vector<Artifact> want_artifacts(3);
  for (int i = 0; i < 3; i++) {
    absl::SleepFor(absl::Milliseconds(1));
    std::string base_text_proto_string;

    base_text_proto_string =
        R"( uri: 'uri_$0',
            name: 'artifact_$0')";
    CreateNodeFromTextProto(absl::Substitute(base_text_proto_string, i),
                            type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_artifacts[i]);
  }

  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: " NOT(id = $0) "
    )", want_artifacts[1].id()),
      *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[2], want_artifacts[0]});

  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri = 'uri_1' and type_id = $0 and \n"
        " create_time_since_epoch = $1"
    )", type.id(), want_artifacts[1].create_time_since_epoch()),
      *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[1]});

  VerifyListOptions<Artifact>(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' OR state = LIVE"
    )", *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[2], want_artifacts[1], want_artifacts[0]});

  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type_id = $0 AND state IS NULL"
    )", type.id()),
      *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[2], want_artifacts[1], want_artifacts[0]});

  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type = '$0'"
    )", type.name()), *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[2], want_artifacts[1], want_artifacts[0]});

  const int64 old_update_time =
      want_artifacts[2].last_update_time_since_epoch();
  Artifact old_artifact = want_artifacts[2];

  old_artifact.set_state(Artifact::LIVE);
  Artifact updated_artifact;
  UpdateAndReturnNode<Artifact>(old_artifact, *metadata_access_object_,
                                metadata_access_object_container_.get(),
                                updated_artifact);

  VerifyListOptions<Artifact>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type = '$0' \n"
        " AND last_update_time_since_epoch > $1"
    )", type.name(), old_update_time), *metadata_access_object_,
      /*want_nodes=*/{updated_artifact});

  VerifyListOptions<Artifact>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type = '$0' \n"
        " AND state = LIVE"
    )", type.name()), *metadata_access_object_,
      /*want_nodes=*/{updated_artifact});

  VerifyListOptions<Artifact>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type = '$0' \n"
        " AND (state = LIVE OR state = DELETED) "
    )", type.name()), *metadata_access_object_,
      /*want_nodes=*/{updated_artifact});

  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type = '$0' and state != LIVE"
    )", type.name()), *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[1], want_artifacts[0]});

  VerifyListOptions<Artifact>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type = '$0' and name = 'artifact_0'"
    )", type.name()), *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[0]});
}

TEST_P(MetadataAccessObjectTest, ListNodesFilterEventQuery) {
  ASSERT_EQ(absl::OkStatus(), Init());
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'at1'", *metadata_access_object_);
  std::vector<Artifact> want_artifacts(3);
  for (int i = 0; i < 3; i++) {
    absl::SleepFor(absl::Milliseconds(1));
    CreateNodeFromTextProto(
        absl::Substitute("uri: 'uri_$0' name: 'artifact_$0'", i),
        artifact_type.id(), *metadata_access_object_,
        metadata_access_object_container_.get(), want_artifacts[i]);
  }

  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'et1'", *metadata_access_object_);
  Execution want_execution;
  CreateNodeFromTextProto(
      "name: 'execution_0' ", execution_type.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), want_execution);

  // Test Setup: a0 -INPUT-> e0 -OUTPUT-> a1
  //                          \ -OUTPUT-> a2
  std::vector<Event> want_events(3);
  CreateEventFromTextProto("type: INPUT", want_artifacts[0], want_execution,
                           *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[0]);
  CreateEventFromTextProto("type: OUTPUT", want_artifacts[1], want_execution,
                           *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[1]);
  CreateEventFromTextProto(
      R"(
      type: OUTPUT,
      milliseconds_since_epoch: 1)",
      want_artifacts[2], want_execution, *metadata_access_object_,
      metadata_access_object_container_.get(), want_events[2]);

  // Filter Artifacts based on Events
  VerifyListOptions<Artifact>(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "events_0.type = INPUT"
    )", *metadata_access_object_, /*want_nodes=*/{want_artifacts[0]});

  VerifyListOptions<Artifact>(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: true }
      filter_query: "events_0.type = INPUT OR events_0.type = OUTPUT"
    )", *metadata_access_object_,
      /*want_nodes=*/want_artifacts);

  VerifyListOptions<Artifact>(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri = 'uri_0' AND events_0.type = INPUT"
    )", *metadata_access_object_, /*want_nodes=*/{want_artifacts[0]});

  VerifyListOptions<Artifact>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri = 'uri_0' AND events_0.execution_id = $0"
    )", want_execution.id()), *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[0]});

  VerifyListOptions<Artifact>(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query:
        "events_0.type = OUTPUT AND events_0.milliseconds_since_epoch = 1"
    )", *metadata_access_object_, /*want_nodes=*/{want_artifacts[2]});

  // Filter Executions based on Events
  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "events_0.artifact_id = $0"
    )", want_artifacts[0].id()), *metadata_access_object_,
      /*want_nodes=*/{want_execution});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "events_0.artifact_id = $0 AND events_0.type = OUTPUT"
    )", want_artifacts[0].id()), *metadata_access_object_, /*want_nodes=*/{});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "events_0.artifact_id = $0 AND events_0.type = OUTPUT"
    )", want_artifacts[1].id()), *metadata_access_object_,
      /*want_nodes=*/{want_execution});
}

TEST_P(MetadataAccessObjectTest, ListExecutionsFilterAttributeQuery) {
  ASSERT_EQ(absl::OkStatus(), Init());
  const ExecutionType type = CreateTypeFromTextProto<ExecutionType>(
      "name: 't1'", *metadata_access_object_);
  std::vector<Execution> want_executions(3);
  for (int i = 0; i < 3; i++) {
    absl::SleepFor(absl::Milliseconds(1));
    std::string base_text_proto_string;

    base_text_proto_string =
        R"(
          name: 'execution_$0')";
    CreateNodeFromTextProto(absl::Substitute(base_text_proto_string, i),
                            type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_executions[i]);
  }
  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "id = $0"
    )", want_executions[2].id()), *metadata_access_object_,
      /*want_nodes=*/{want_executions[2]});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "id != $0 OR last_known_state = COMPLETE"
    )", want_executions[2].id()), *metadata_access_object_,
      /*want_nodes=*/{want_executions[1], want_executions[0]});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type_id = $0 AND last_known_state != COMPLETE"
    )", type.id()),
      *metadata_access_object_,
      /*want_nodes=*/
      {want_executions[2], want_executions[1], want_executions[0]});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type_id = $0 AND create_time_since_epoch = $1"
    )", type.id(), want_executions[2].create_time_since_epoch()),
      *metadata_access_object_,
      /*want_nodes=*/
      {want_executions[2]});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1"
    )", type.name(), type.id()), *metadata_access_object_,
      /*want_nodes=*/
      {want_executions[2], want_executions[1], want_executions[0]});

  const int64 old_update_time =
      want_executions[2].last_update_time_since_epoch();
  Execution old_execution = want_executions[2];

  old_execution.set_last_known_state(Execution::COMPLETE);
  Execution updated_execution;
  UpdateAndReturnNode<Execution>(old_execution, *metadata_access_object_,
                                 metadata_access_object_container_.get(),
                                 updated_execution);

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 AND \n"
        " last_update_time_since_epoch > $2"
    )", type.name(), type.id(), old_update_time), *metadata_access_object_,
      /*want_nodes=*/{updated_execution});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 AND \n"
        " last_known_state = COMPLETE"
    )", type.name(), type.id()), *metadata_access_object_,
      /*want_nodes=*/{updated_execution});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 AND \n"
        " (last_known_state = COMPLETE OR last_known_state = NEW) "
    )", type.name(), type.id()), *metadata_access_object_,
      /*want_nodes=*/{updated_execution});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 AND \n"
        " last_known_state != COMPLETE"
    )", type.name(), type.id()), *metadata_access_object_,
      /*want_nodes=*/{want_executions[1], want_executions[0]});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 AND \n"
        " name = 'execution_0'"
    )", type.name(), type.id()), *metadata_access_object_,
      /*want_nodes=*/{want_executions[0]});
}

TEST_P(MetadataAccessObjectTest, ListContextsFilterAttributeQuery) {
  ASSERT_EQ(absl::OkStatus(), Init());
  const ContextType type =
      CreateTypeFromTextProto<ContextType>(R"(
      name: 't1'
      properties { key: 'p1' value: INT })",
                                           *metadata_access_object_);
  std::vector<Context> want_contexts(3);
  for (int i = 0; i < 3; i++) {
    absl::SleepFor(absl::Milliseconds(1));
    std::string base_text_proto_string;

    base_text_proto_string =
        R"(
          name: 'c$0')";
    CreateNodeFromTextProto(absl::Substitute(base_text_proto_string, i),
                            type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_contexts[i]);
  }

  VerifyListOptions<Context>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "id = $0 "
    )", want_contexts[0].id()), *metadata_access_object_,
      /*want_nodes=*/{want_contexts[0]});

  VerifyListOptions<Context>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type_id = $0 AND name IS NOT NULL"
    )", type.id()), *metadata_access_object_,
      /*want_nodes=*/{want_contexts[2], want_contexts[1], want_contexts[0]});

  // The queries below require the `IN` operator support.
  VerifyListOptions<Context>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "id IN ($0, $1) "
    )",
                       want_contexts[0].id(), want_contexts[2].id()),
      *metadata_access_object_,
      /*want_nodes=*/{want_contexts[2], want_contexts[0]});

  VerifyListOptions<Context>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1"
    )", type.name(), type.id()), *metadata_access_object_,
      /*want_nodes=*/
      {want_contexts[2], want_contexts[1], want_contexts[0]});

  VerifyListOptions<Context>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id != $1"
    )", type.name(), type.id()), *metadata_access_object_,
      /*want_nodes=*/{});

  VerifyListOptions<Context>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: '(type = \'$0\' AND name = \'$1\') OR '
                    '(type = \'$2\' AND name = \'$3\')'
    )",
        type.name(), want_contexts[2].name(),
        type.name(), want_contexts[1].name()),
      *metadata_access_object_,
      /*want_nodes=*/{want_contexts[2], want_contexts[1]});

  Context old_context = want_contexts[2];
  const int64 old_update_time = want_contexts[2].last_update_time_since_epoch();
  (*old_context.mutable_properties())["p1"].set_int_value(1);
  Context updated_context;
  UpdateAndReturnNode<Context>(old_context, *metadata_access_object_,
                               metadata_access_object_container_.get(),
                               updated_context);
  VerifyListOptions<Context>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 \n"
      " AND last_update_time_since_epoch > $2"
    )", type.name(), type.id(), old_update_time), *metadata_access_object_,
      /*want_nodes=*/{updated_context});

  VerifyListOptions<Context>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 \n"
      " AND name = 'c0'"
    )", type.name(), type.id()), *metadata_access_object_,
      /*want_nodes=*/{want_contexts[0]});
}

TEST_P(MetadataAccessObjectTest, ListNodesFilterContextNeighborQuery) {
  ASSERT_EQ(absl::OkStatus(), Init());
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_);
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_);
  const ContextType context_type = CreateTypeFromTextProto<ContextType>(
      "name: 'context_type'", *metadata_access_object_);

  // Test setup: creates 3 artifacts, 3 executions and 3 contexts, and attached:
  // [artifact_1, execution_2] to context_0,
  // [artifact_0, execution_0] to [context_0, context_1, context_2]
  std::vector<Artifact> want_artifacts(3);
  std::vector<Execution> want_executions(3);
  std::vector<Context> contexts(3);
  for (int i = 0; i < 3; i++) {
    CreateNodeFromTextProto(absl::Substitute("uri: 'uri_$0'", i),
                            artifact_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_artifacts[i]);
    CreateNodeFromTextProto("", execution_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_executions[i]);
    CreateNodeFromTextProto(absl::Substitute("name: 'c$0'", i),
                            context_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            contexts[i]);
    absl::SleepFor(absl::Milliseconds(1));
  }
  Attribution attribution;
  attribution.set_artifact_id(want_artifacts[1].id());
  attribution.set_context_id(contexts[0].id());
  int64 attid;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateAttribution(attribution, &attid));
  Association association;
  association.set_execution_id(want_executions[2].id());
  association.set_context_id(contexts[0].id());
  int64 assid;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateAssociation(association, &assid));
  for (int i = 0; i < 3; i++) {
    Attribution attribution;
    attribution.set_artifact_id(want_artifacts[0].id());
    attribution.set_context_id(contexts[i].id());
    int64 attid;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateAttribution(attribution, &attid));
    Association association;
    association.set_execution_id(want_executions[0].id());
    association.set_context_id(contexts[i].id());
    int64 assid;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateAssociation(association, &assid));
  }
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: true }
      filter_query: "uri IS NOT NULL AND type = '$0' AND \n"
                    "contexts_c.name = '$1' AND contexts_c.type = '$2' \n"
    )",
                       artifact_type.name(), contexts[0].name(),
                       context_type.name()),
      *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[0], want_artifacts[1]});

  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "contexts_c.name LIKE '%' AND contexts_c.type = '$0' \n"
    )", context_type.name()), *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[1], want_artifacts[0]});


  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "contexts_c.create_time_since_epoch = $0 AND \n"
                    "contexts_c.last_update_time_since_epoch = $1")",
                       contexts[1].create_time_since_epoch(),
                       contexts[1].last_update_time_since_epoch()),
      *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[0]});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND contexts_c.id = $1 AND \n"
                    "contexts_c.name = '$2' AND contexts_c.type = '$3' \n"
    )",
                       execution_type.name(), contexts[0].id(),
                       contexts[0].name(), context_type.name()),
      *metadata_access_object_,
      /*want_nodes=*/{want_executions[2], want_executions[0]});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "contexts_c.create_time_since_epoch < $0 AND \n"
                    "contexts_c.last_update_time_since_epoch < $1")",
                       contexts[1].create_time_since_epoch(),
                       contexts[1].last_update_time_since_epoch()),
      *metadata_access_object_,
      /*want_nodes=*/{want_executions[2], want_executions[0]});

  // Add another Association and make the Association relationship as follows:
  // contexts[0] <-> want_executions[0, 2]
  // contexts[1] <-> want_executions[0, 1]
  // contexts[2] <-> want_executions[0]
  Association additional_association;
  additional_association.set_execution_id(want_executions[1].id());
  additional_association.set_context_id(contexts[1].id());
  int64 dummy_assid;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateAssociation(
                                  additional_association, &dummy_assid));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: " contexts_a.id = $0 OR contexts_a.id = $1")",
                       contexts[0].id(),
                       contexts[1].id()),
      *metadata_access_object_,
      /*want_nodes=*/{want_executions[2], want_executions[1],
                      want_executions[0]});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: '(contexts_1.type = \'$0\' AND contexts_1.name = \'$1\') OR '
                    '(contexts_2.type = \'$2\' AND contexts_2.name = \'$3\')'
    )",
        context_type.name(), contexts[2].name(),
        context_type.name(), contexts[1].name()),
      *metadata_access_object_,
      /*want_nodes=*/{want_executions[1], want_executions[0]});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: '(contexts_1.type=\'$0\' AND contexts_1.name=\'$1\') AND '
                    '(contexts_2.type=\'$2\' AND contexts_2.name=\'$3\') AND '
                    '(contexts_3.type=\'$4\' AND contexts_3.name=\'$5\') '
    )",
        context_type.name(), contexts[2].name(),
        context_type.name(), contexts[1].name(),
        context_type.name(), contexts[0].name()),
      *metadata_access_object_,
      /*want_nodes=*/{want_executions[0]});
}

TEST_P(MetadataAccessObjectTest, ListContextNodesWithParentChildQuery) {
  ASSERT_EQ(absl::OkStatus(), Init());
  const ContextType parent_context_type_1 =
      CreateTypeFromTextProto<ContextType>("name: 'parent_context_type_1'",
                                           *metadata_access_object_);
  const ContextType parent_context_type_2 =
      CreateTypeFromTextProto<ContextType>("name: 'parent_context_type_2'",
                                           *metadata_access_object_);

  const ContextType child_context_type = CreateTypeFromTextProto<ContextType>(
      "name: 'child_context_type'", *metadata_access_object_);

  // Test Setup: Creates 3 context nodes with same parent and 2 of the 3
  // context nodes with same different parent.
  Context parent_context_1;
  CreateNodeFromTextProto(
      "name: 'p1'", parent_context_type_1.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), parent_context_1);

  Context parent_context_2;
  CreateNodeFromTextProto(
      "name: 'p2'", parent_context_type_2.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), parent_context_2);

  std::vector<Context> child_contexts;
  for (int i = 0; i < 3; i++) {
    Context child_context;
    CreateNodeFromTextProto(absl::Substitute("name: 'c$0'", i + 1),
                            child_context_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            child_context);
    child_contexts.push_back(child_context);

    ParentContext parent_context;
    parent_context.set_parent_id(parent_context_1.id());
    parent_context.set_child_id(child_context.id());
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateParentContext(parent_context));
  }

  for (int i = 0; i < 2; i++) {
    ParentContext parent_context;
    parent_context.set_parent_id(parent_context_2.id());
    parent_context.set_child_id(child_contexts[i].id());
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateParentContext(parent_context));
  }

  // Query on ParentContext.type
  VerifyListOptions<Context>(
      absl::Substitute(
          R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "parent_contexts_c.type = '$0'")",
          "parent_context_type_1"),
      *metadata_access_object_,
      /*want_nodes=*/{child_contexts[2], child_contexts[1], child_contexts[0]});

  // Query on ParentContext.type
  VerifyListOptions<Context>(
      absl::Substitute(
          R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "parent_contexts_c.type = '$0'")",
          "parent_context_type_2"),
      *metadata_access_object_,
      /*want_nodes=*/{child_contexts[1], child_contexts[0]});

  // Query on ParentContext.name
  VerifyListOptions<Context>(
      R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "parent_contexts_c.name = 'p1'")",
      *metadata_access_object_,
      /*want_nodes=*/{child_contexts[2], child_contexts[1], child_contexts[0]});

  // Query on ParentContext.name
  VerifyListOptions<Context>(
      R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "parent_contexts_c.name = 'p2'")",
      *metadata_access_object_,
      /*want_nodes=*/{child_contexts[1], child_contexts[0]});

  // Query on ChildContext.type
  VerifyListOptions<Context>(
      absl::Substitute(
          R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "child_contexts_c.type = '$0'")",
          "child_context_type"),
      *metadata_access_object_,
      /*want_nodes=*/{parent_context_2, parent_context_1});

  // Query on ChildContext.name
  VerifyListOptions<Context>(
      absl::Substitute(
          R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "child_contexts_c.name = '$0'")",
          child_contexts[0].name()),
      *metadata_access_object_,
      /*want_nodes=*/{parent_context_2, parent_context_1});

  // Query on ChildContext.name
  VerifyListOptions<Context>(absl::Substitute(
                                 R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "child_contexts_c.name = '$0'")",
                                 child_contexts[2].name()),
                             *metadata_access_object_,
                             /*want_nodes=*/{parent_context_1});
}

TEST_P(MetadataAccessObjectTest,
       ListContextNodesWithParentChildAndPropertiesQuery) {
  ASSERT_EQ(absl::OkStatus(), Init());
  const ContextType parent_context_type_1 =
      CreateTypeFromTextProto<ContextType>("name: 'parent_context_type_1'",
                                           *metadata_access_object_);

  const ContextType child_context_type = CreateTypeFromTextProto<ContextType>(
      "name: 'child_context_type'", *metadata_access_object_);

  Context parent_context_1;
  CreateNodeFromTextProto(
      "name: 'p1'", parent_context_type_1.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), parent_context_1);

  Context child_context_1;
  CreateNodeFromTextProto(absl::StrFormat(R"(
          type_id: %d
          name: 'c1'
          custom_properties {
            key: 'custom_property_1' value: { string_value: 'foo' }
        })",
                                          child_context_type.id()),
                          child_context_type.id(), *metadata_access_object_,
                          metadata_access_object_container_.get(),
                          child_context_1);

  Context child_context_2;
  CreateNodeFromTextProto(absl::StrFormat(R"(
          type_id: %d
          name: 'c2'
          custom_properties {
            key: 'custom_property_1' value: { string_value: 'foo' }
        })",
                                          child_context_type.id()),
                          child_context_type.id(), *metadata_access_object_,
                          metadata_access_object_container_.get(),
                          child_context_2);

  Context child_context_3;
  CreateNodeFromTextProto(absl::StrFormat(R"(
          type_id: %d
          name: 'c3'
          custom_properties {
            key: 'custom_property_1' value: { string_value: 'bar' }
        })",
                                          child_context_type.id()),
                          child_context_type.id(), *metadata_access_object_,
                          metadata_access_object_container_.get(),
                          child_context_3);

  ParentContext parent_child_context_1;
  parent_child_context_1.set_parent_id(parent_context_1.id());
  parent_child_context_1.set_child_id(child_context_1.id());
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateParentContext(
                                  parent_child_context_1));
  ParentContext parent_child_context_2;
  parent_child_context_2.set_parent_id(parent_context_1.id());
  parent_child_context_2.set_child_id(child_context_3.id());
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateParentContext(
                                  parent_child_context_2));

  // Query on ParentContext and custom property.
  VerifyListOptions<Context>(R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "parent_contexts_c.type = 'parent_context_type_1' \n"
             " AND custom_properties.custom_property_1.string_value = 'foo'")",
                             *metadata_access_object_,
                             /*want_nodes=*/{child_context_1});
}

template <class NodeType, class Node>
void TestFilteringWithListOptionsImpl(
    MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container) {
  const NodeType type = CreateTypeFromTextProto<NodeType>(R"(
    name: 'test_type'
    properties { key: 'p1' value: INT }
    properties { key: 'p2' value: DOUBLE }
    properties { key: 'p3' value: STRING }
  )", metadata_access_object);

  // Setup: 5 nodes of `test_type`
  // node_$i has a custom property custom_property_$i which is not NULL.
  // node_$i also has p1 = $i, p2 = $i.0, p3 = '$i'
  std::vector<Node> want_nodes(5);
  for (int i = 0; i < want_nodes.size(); i++) {
    CreateNodeFromTextProto(absl::StrFormat(R"(
          type_id: %d
          name: 'test_%d'
          properties { key: 'p1' value: { int_value: %d } }
          properties { key: 'p2' value: { double_value: %f } }
          properties { key: 'p3' value: { string_value: '%s' }
        }
        custom_properties {
          key: 'custom_property_%d' value: { string_value: 'foo' }
        }
        custom_properties {
          key: 'custom_property %d' value: { double_value: 1.0 }
        }
        custom_properties {
          key: 'custom_property:%d' value: { int_value: 1 }
        })",
                                            type.id(), i, i, 1.0 * i,
                                            absl::StrCat("0", i), i, i, i),
                            type.id(), metadata_access_object,
                            metadata_access_object_container, want_nodes[i]);
  }

  static constexpr char kListOption[] = R"(
            max_result_size: 10,
            order_by_field: { field: CREATE_TIME is_asc: false }
            filter_query: "$0")";

  // test property and custom property queries
  // verify all documented columns can be used.
  VerifyListOptions<Node>(
      absl::Substitute(kListOption, "properties.p1.int_value = 0"),
      metadata_access_object, /*want_nodes=*/{want_nodes[0]});

  VerifyListOptions<Node>(
      absl::Substitute(kListOption,
                       "(properties.p1.int_value + 2) * 1 = 2"),
      metadata_access_object, /*want_nodes=*/{want_nodes[0]});

  VerifyListOptions<Node>(
      absl::Substitute(kListOption, "properties.p2.double_value > 2.0"),
      metadata_access_object,
      /*want_nodes=*/{want_nodes[4], want_nodes[3]});

  VerifyListOptions<Node>(
      absl::Substitute(kListOption,
                       "properties.p2.double_value > 2.0 / 1.0 - (-0.5)"),
      metadata_access_object,
      /*want_nodes=*/{want_nodes[4], want_nodes[3]});

  VerifyListOptions<Node>(
      absl::Substitute(kListOption, "properties.p3.string_value LIKE '0%'"),
      metadata_access_object,
      /*want_nodes=*/
      {want_nodes[4], want_nodes[3], want_nodes[2], want_nodes[1],
       want_nodes[0]});

  VerifyListOptions<Node>(
      absl::Substitute(
          kListOption,
          "custom_properties.custom_property_0.int_value IS NULL AND "
          "custom_properties.custom_property_0.double_value IS NULL AND "
          "custom_properties.custom_property_0.string_value = 'foo'"),
      metadata_access_object,
      /*want_nodes=*/{want_nodes[0]});

  // verify unknown property name can be used
  VerifyListOptions<Node>(
      absl::Substitute(kListOption,
                       "properties.unknown.int_value > 0 OR "
                       "custom_properties.some_property.string_value = '0'"),
      metadata_access_object,
      /*want_nodes=*/{});

  // verify properties with backquoted string can be used
  VerifyListOptions<Node>(
      absl::Substitute(
          kListOption,
          "properties.p1.int_value = 0 AND "
          "custom_properties.`custom_property:0`.int_value > 0 AND "
          "custom_properties.`custom_property 0`.double_value > 0 "),
      metadata_access_object,
      /*want_nodes=*/{want_nodes[0]});
}

TEST_P(MetadataAccessObjectTest, ListArtifactsFilterPropertyQuery) {
  ASSERT_EQ(absl::OkStatus(), Init());
  TestFilteringWithListOptionsImpl<ArtifactType, Artifact>(
      *metadata_access_object_, metadata_access_object_container_.get());
}

TEST_P(MetadataAccessObjectTest, ListExecutionsFilterPropertyQuery) {
  ASSERT_EQ(absl::OkStatus(), Init());
  TestFilteringWithListOptionsImpl<ExecutionType, Execution>(
      *metadata_access_object_, metadata_access_object_container_.get());
}

TEST_P(MetadataAccessObjectTest, LisContextsFilterPropertyQuery) {
  ASSERT_EQ(absl::OkStatus(), Init());
  TestFilteringWithListOptionsImpl<ContextType, Context>(
      *metadata_access_object_, metadata_access_object_container_.get());
}

TEST_P(MetadataAccessObjectTest, ListNodesFilterWithErrors) {
  ASSERT_EQ(absl::OkStatus(), Init());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 10,
        order_by_field: { field: CREATE_TIME is_asc: false }
        filter_query: "unknown_field = 'uri_3' and type_id = 1"
      )");

  std::string next_page_token;
  {
    std::vector<Artifact> got_artifacts;
    const absl::Status status = metadata_access_object_->ListArtifacts(
        list_options, &got_artifacts, &next_page_token);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  {
    std::vector<Execution> got_executions;
    const absl::Status status = metadata_access_object_->ListExecutions(
        list_options, &got_executions, &next_page_token);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  {
    std::vector<Context> got_contexts;
    const absl::Status status = metadata_access_object_->ListContexts(
        list_options, &got_contexts, &next_page_token);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }
}

// Apply the list options to list the `Node`, and compare with the `want_nodes`
// point-wise in order.
void VerifyLineageGraph(const LineageGraph& subgraph,
                        const std::vector<Artifact>& artifacts,
                        const std::vector<Execution>& executions,
                        const std::vector<Event>& events,
                        MetadataAccessObject& metadata_access_object) {
  // Compare nodes and edges.
  EXPECT_THAT(subgraph.artifacts(),
              UnorderedPointwise(EqualsProto<Artifact>(), artifacts));
  EXPECT_THAT(subgraph.executions(),
              UnorderedPointwise(EqualsProto<Execution>(), executions));
  EXPECT_THAT(subgraph.events(),
              UnorderedPointwise(EqualsProto<Event>(/*ignore_fields=*/{
                                     "milliseconds_since_epoch",
                                 }),
                                 events));
  // Compare types.
  std::vector<ArtifactType> artifact_types;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.FindTypes(&artifact_types));
  EXPECT_THAT(subgraph.artifact_types(),
              UnorderedPointwise(EqualsProto<ArtifactType>(), artifact_types));
  std::vector<ExecutionType> execution_types;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.FindTypes(&execution_types));
  EXPECT_THAT(
      subgraph.execution_types(),
      UnorderedPointwise(EqualsProto<ExecutionType>(), execution_types));
  std::vector<ContextType> context_types;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object.FindTypes(&context_types));
  EXPECT_THAT(subgraph.context_types(),
              UnorderedPointwise(EqualsProto<ContextType>(), context_types));
}

TEST_P(MetadataAccessObjectTest, QueryLineageGraph) {
  ASSERT_EQ(absl::OkStatus(), Init());
  // Test setup: use a simple graph with multiple paths between (a1, e2).
  // a1 -> e1 -> a2
  //  \            \
  //   \------------> e2
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_);
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_);
  std::vector<Artifact> want_artifacts(2);
  std::vector<Execution> want_executions(2);
  for (int i = 0; i < 2; i++) {
    CreateNodeFromTextProto(absl::Substitute("uri: 'uri_$0'", i),
                            artifact_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_artifacts[i]);
  }
  for (int i = 0; i < 2; i++) {
    CreateNodeFromTextProto("", execution_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_executions[i]);
  }
  std::vector<Event> want_events(4);
  CreateEventFromTextProto("type: INPUT", want_artifacts[0], want_executions[0],
                           *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[0]);
  CreateEventFromTextProto("type: INPUT", want_artifacts[0], want_executions[1],
                           *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[1]);
  CreateEventFromTextProto("type: OUTPUT", want_artifacts[1],
                           want_executions[0], *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[2]);
  CreateEventFromTextProto("type: INPUT", want_artifacts[1], want_executions[1],
                           *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[3]);

  {
    // Empty query nodes is OK.
    LineageGraph output_graph;
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->QueryLineageGraph(
            /*query_nodes=*/{}, /*max_num_hops=*/0, /*max_nodes=*/absl::nullopt,
            /*boundary_artifacts=*/absl::nullopt,
            /*boundary_executions=*/absl::nullopt, output_graph));
    VerifyLineageGraph(output_graph, /*artifacts=*/{}, /*executions=*/{},
                       /*events=*/{}, *metadata_access_object_);
  }

  {
    // Query a1 with 1 hop.
    LineageGraph output_graph;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/1,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/absl::nullopt, output_graph));
    VerifyLineageGraph(
        output_graph, /*artifacts=*/{want_artifacts[0]}, want_executions,
        /*events=*/{want_events[0], want_events[1]}, *metadata_access_object_);
  }

  {
    // Query a1 with 2 hop. It returns all nodes with no duplicate.
    LineageGraph output_graph;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/2,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/absl::nullopt, output_graph));
    VerifyLineageGraph(output_graph, want_artifacts, want_executions,
                       want_events, *metadata_access_object_);
  }

  {
    // Query a1 with 2 hop and max_nodes of 2. It returns a1 and one of e1 or
    // e2.
    LineageGraph output_graph;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/2,
                  /*max_nodes=*/2,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/absl::nullopt, output_graph));

    // Compare nodes and edges.
    EXPECT_THAT(
        output_graph.artifacts(),
        UnorderedPointwise(EqualsProto<Artifact>(), {want_artifacts[0]}));
    EXPECT_EQ(output_graph.executions().size(), 1);
    EXPECT_EQ(output_graph.events().size(), 1);
  }

  {
    // Query a1 with 2 hop and max_nodes of 3.
    LineageGraph output_graph;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/2,
                  /*max_nodes=*/3,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/absl::nullopt, output_graph));
    VerifyLineageGraph(
        output_graph, /*artifacts=*/{want_artifacts[0]}, want_executions,
        /*events=*/{want_events[0], want_events[1]}, *metadata_access_object_);
  }

  {
    // Query a1 with 2 hop and max_nodes of 4. It returns all nodes with no
    // duplicate.
    LineageGraph output_graph;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/2,
                  /*max_nodes=*/4,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/absl::nullopt, output_graph));
    VerifyLineageGraph(output_graph, want_artifacts, want_executions,
                       want_events, *metadata_access_object_);
  }

  {
    // With multiple query nodes with 0 hop.
    LineageGraph output_graph;
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->QueryLineageGraph(
            want_artifacts, /*max_num_hops=*/0, /*max_nodes=*/absl::nullopt,
            /*boundary_artifacts=*/absl::nullopt,
            /*boundary_executions=*/absl::nullopt, output_graph));
    VerifyLineageGraph(output_graph, want_artifacts, /*executions=*/{},
                       /*events=*/{}, *metadata_access_object_);
  }

  {
    // Query multiple nodes with a large hop.
    // It returns all nodes with no duplicate.
    LineageGraph output_graph;
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->QueryLineageGraph(
            want_artifacts, /*max_num_hops=*/5, /*max_nodes=*/absl::nullopt,
            /*boundary_artifacts=*/absl::nullopt,
            /*boundary_executions=*/absl::nullopt, output_graph));
    VerifyLineageGraph(output_graph, want_artifacts, want_executions,
                       want_events, *metadata_access_object_);
  }
}

TEST_P(MetadataAccessObjectTest, QueryLineageGraphArtifactsOnly) {
  ASSERT_EQ(absl::OkStatus(), Init());
  // Test setup: only set up an artifact type and 2 artifacts.
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_);
  std::vector<Artifact> want_artifacts(2);
  for (int i = 0; i < 2; i++) {
    CreateNodeFromTextProto(absl::Substitute("uri: 'uri_$0'", i),
                            artifact_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_artifacts[i]);
  }

  LineageGraph output_graph;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->QueryLineageGraph(
                want_artifacts, /*max_num_hops=*/1, /*max_nodes=*/absl::nullopt,
                /*boundary_artifacts=*/absl::nullopt,
                /*boundary_executions=*/absl::nullopt, output_graph));
  VerifyLineageGraph(output_graph, want_artifacts, /*executions=*/{},
                     /*events=*/{}, *metadata_access_object_);
}

TEST_P(MetadataAccessObjectTest, QueryLineageGraphWithBoundaryConditions) {
  ASSERT_EQ(absl::OkStatus(), Init());
  // Test setup: use a high fan-out graph to test the boundaries cases
  // a0 -> e1 -> a1 -> e0
  //   \-> e2
  //   \-> ...
  //   \-> e250
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_);
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_);
  std::vector<Artifact> want_artifacts(2);
  std::vector<Execution> want_executions(251);
  for (int i = 0; i < 2; i++) {
    CreateNodeFromTextProto(absl::Substitute("uri: 'uri_$0'", i),
                            artifact_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_artifacts[i]);
  }
  for (int i = 0; i < 251; i++) {
    CreateNodeFromTextProto(absl::Substitute("name: 'e$0'", i),
                            execution_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_executions[i]);
  }
  std::vector<Event> all_events(252);
  for (int i = 1; i < 251; i++) {
    CreateEventFromTextProto("type: INPUT", want_artifacts[0],
                             want_executions[i], *metadata_access_object_,
                             metadata_access_object_container_.get(),
                             all_events[i - 1]);
  }
  const Event a0e1 = all_events[0];
  CreateEventFromTextProto("type: INPUT", want_artifacts[1], want_executions[0],
                           *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           all_events[250]);
  CreateEventFromTextProto("type: OUTPUT", want_artifacts[1],
                           want_executions[1], *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           all_events[251]);
  const Event a1e0 = all_events[250];
  const Event a1e1 = all_events[251];

  {
    // boundary execution condition with max num hops
    LineageGraph output_graph;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/2,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/"name != 'e0'", output_graph));
    VerifyLineageGraph(output_graph, want_artifacts,
                       /*executions=*/{want_executions[1]},
                       /*events=*/{a1e1, a0e1}, *metadata_access_object_);
  }

  {
    // boundary execution condition with max num hops and max_nodes of 3.
    LineageGraph output_graph;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/2,
                  /*max_nodes=*/3,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/"name != 'e0'", output_graph));
    VerifyLineageGraph(output_graph, want_artifacts,
                       /*executions=*/{want_executions[1]},
                       /*events=*/{a1e1, a0e1}, *metadata_access_object_);
  }

  {
    // boundary artifact condition with max_num_hops.
    LineageGraph output_graph;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/3,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/"uri != 'unknown_uri'",
                  /*boundary_executions=*/absl::nullopt, output_graph));
    VerifyLineageGraph(output_graph, want_artifacts, want_executions,
                       all_events, *metadata_access_object_);
  }

  {
    // boundary artifact condition with max_num_hops and max nodes of 10.
    // It should return both artifacts and 8 exeuctions.
    LineageGraph output_graph;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/3,
                  /*max_nodes=*/10,
                  /*boundary_artifacts=*/"uri != 'unknown_uri'",
                  /*boundary_executions=*/absl::nullopt, output_graph));
    // Compare nodes and edges.
    EXPECT_THAT(output_graph.artifacts(),
                UnorderedPointwise(EqualsProto<Artifact>(), want_artifacts));
    EXPECT_EQ(output_graph.executions().size(), 8);
    EXPECT_EQ(output_graph.events().size(), 9);
  }

  {
    // boundary artifact and execution condition with max num hops
    LineageGraph output_graph;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/2,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/"uri != 'uri_0'",
                  /*boundary_executions=*/"name != 'e0'", output_graph));
    VerifyLineageGraph(output_graph, /*artifacts=*/{want_artifacts[1]},
                       /*executions=*/{want_executions[1]},
                       /*events=*/{a1e1}, *metadata_access_object_);
  }

  {
    // boundary condition rejects large number of nodes.
    LineageGraph output_graph;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/3,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/"name = 'e1'", output_graph));
    VerifyLineageGraph(output_graph, /*artifacts=*/want_artifacts,
                       /*executions=*/{want_executions[1]},
                       /*events=*/{a1e1, a0e1}, *metadata_access_object_);
  }
}

TEST_P(MetadataAccessObjectTest, DeleteArtifactsById) {
  ASSERT_EQ(absl::OkStatus(), Init());

  const ArtifactType type = CreateTypeFromTextProto<ArtifactType>(
      R"pb(
        name: 'test_type'
        properties { key: 'property_1' value: INT }
        properties { key: 'property_2' value: DOUBLE }
        properties { key: 'property_3' value: STRING }
      )pb",
      *metadata_access_object_);
  Artifact artifact;
  CreateNodeFromTextProto(
      R"pb(
        name: 'delete_artifacts_by_id_test'
        properties {
          key: 'property_1'
          value: { int_value: 3 }
        }
        properties {
          key: 'property_2'
          value: { double_value: 3.0 }
        }
        properties {
          key: 'property_3'
          value: { string_value: '3' }
        }
      )pb",
      type.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), artifact);
  // Test: empty ids
  {
    std::vector<Artifact> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteArtifactsById({}));
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsById(
                                    {artifact.id()}, &result));
    EXPECT_THAT(result.size(), 1);
    ASSERT_EQ(
        absl::StatusOr<bool>(false),
        metadata_access_object_container_->CheckTableEmpty("ArtifactProperty"));
  }
  // Test: actual deletion
  {
    std::vector<Artifact> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteArtifactsById({artifact.id()}));
    absl::Status status =
        metadata_access_object_->FindArtifactsById({artifact.id()}, &result);
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    EXPECT_THAT(result, IsEmpty());
    // We expect the properties table to be empty because we deleted the only
    // Artifact that was in our database.
    ASSERT_EQ(
        absl::StatusOr<bool>(true),
        metadata_access_object_container_->CheckTableEmpty("ArtifactProperty"));
  }
}

TEST_P(MetadataAccessObjectTest, DeleteExecutionsById) {
  ASSERT_EQ(absl::OkStatus(), Init());

  const ExecutionType type = CreateTypeFromTextProto<ExecutionType>(
      R"pb(
        name: 'test_type'
        properties { key: 'property_1' value: INT }
        properties { key: 'property_2' value: DOUBLE }
        properties { key: 'property_3' value: STRING }
      )pb",
      *metadata_access_object_);
  Execution execution;
  CreateNodeFromTextProto(
      R"pb(
        name: 'delete_executions_by_id_test'
        properties {
          key: 'property_1'
          value: { int_value: 3 }
        }
        properties {
          key: 'property_2'
          value: { double_value: 3.0 }
        }
        properties {
          key: 'property_3'
          value: { string_value: '3' }
        }
      )pb",
      type.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), execution);

  // Test: empty ids
  {
    std::vector<Execution> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteExecutionsById({}));
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsById(
                                    {execution.id()}, &result));
    EXPECT_THAT(result.size(), 1);
    ASSERT_EQ(absl::StatusOr<bool>(false),
              metadata_access_object_container_->CheckTableEmpty(
                  "ExecutionProperty"));
  }
  // Test: actual deletion
  {
    std::vector<Execution> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteExecutionsById({execution.id()}));
    absl::Status status =
        metadata_access_object_->FindExecutionsById({execution.id()}, &result);
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    EXPECT_THAT(result, IsEmpty());
    // We expect the properties table to be empty because we deleted the only
    // Execution that was in our database.
    ASSERT_EQ(absl::StatusOr<bool>(true),
              metadata_access_object_container_->CheckTableEmpty(
                  "ExecutionProperty"));
  }
}

TEST_P(MetadataAccessObjectTest, DeleteContextsById) {
  ASSERT_EQ(absl::OkStatus(), Init());

  const ContextType type = CreateTypeFromTextProto<ContextType>(
      R"pb(
        name: 'test_type'
        properties { key: 'property_1' value: INT }
        properties { key: 'property_2' value: DOUBLE }
        properties { key: 'property_3' value: STRING }
      )pb",
      *metadata_access_object_);
  Context context1, context2;
  CreateNodeFromTextProto("name: 'delete_contexts_by_id_test_1'", type.id(),
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context1);
  CreateNodeFromTextProto("name: 'delete_contexts_by_id_test_2'", type.id(),
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context2);

  // Test: empty ids
  {
    std::vector<Context> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteContextsById({}));
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {context1.id(), context2.id()}, &result));
    EXPECT_THAT(result.size(), 2);
  }
  // Test: actual deletion on context1
  {
    std::vector<Context> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteContextsById({context1.id()}));
    absl::Status status = metadata_access_object_->FindContextsById(
        {context1.id(), context2.id()}, &result);
    // context1 not found
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    // context2 remains
    EXPECT_THAT(result.size(), 1);
    EXPECT_EQ(result.front().id(), context2.id());
  }
  // Test: context id is wrong when deleting context2
  {
    std::vector<Context> result;
    // still returns OK status when context id is not found
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteContextsById({context2.id() + 1}));
    absl::Status status = metadata_access_object_->FindContextsById(
        {context1.id(), context2.id()}, &result);
    // context1 not found
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    // context2 remains because context id is wrong when deleting it
    EXPECT_THAT(result.size(), 1);
    EXPECT_EQ(result.front().id(), context2.id());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteEventsByArtifactsId) {
  ASSERT_EQ(absl::OkStatus(), Init());

  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64 execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Artifact input_artifact;
  CreateNodeFromTextProto(
      "name: 'input_artifact'", artifact_type_id, *metadata_access_object_,
      metadata_access_object_container_.get(), input_artifact);
  Artifact output_artifact;
  CreateNodeFromTextProto(
      "name: 'output_artifact'", artifact_type_id, *metadata_access_object_,
      metadata_access_object_container_.get(), output_artifact);
  Execution execution;
  CreateNodeFromTextProto("name: 'execution'", execution_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), execution);
  Event event1;
  CreateEventFromTextProto("type: INPUT", input_artifact, execution,
                           *metadata_access_object_,
                           metadata_access_object_container_.get(), event1);
  Event event2;
  CreateEventFromTextProto("type: OUTPUT", output_artifact, execution,
                           *metadata_access_object_,
                           metadata_access_object_container_.get(), event2);

  // Test: empty ids
  {
    std::vector<Event> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteEventsByArtifactsId({}));
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindEventsByArtifacts(
                  {input_artifact.id(), output_artifact.id()}, &result));
    EXPECT_THAT(result.size(), 2);
  }
  // Test: delete one event
  {
    std::vector<Event> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteEventsByArtifactsId(
                  {input_artifact.id()}));
    absl::Status status = metadata_access_object_->FindEventsByArtifacts(
        {input_artifact.id()}, &result);
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    EXPECT_THAT(result, IsEmpty());
    status = metadata_access_object_->FindEventsByArtifacts(
        {output_artifact.id()}, &result);
    EXPECT_THAT(result.size(), 1);
  }
}

TEST_P(MetadataAccessObjectTest, DeleteEventsByExecutionsId) {
  ASSERT_EQ(absl::OkStatus(), Init());

  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64 execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Artifact input_artifact;
  CreateNodeFromTextProto(
      "name: 'input_artifact'", artifact_type_id, *metadata_access_object_,
      metadata_access_object_container_.get(), input_artifact);
  Artifact output_artifact;
  CreateNodeFromTextProto(
      "name: 'output_artifact'", artifact_type_id, *metadata_access_object_,
      metadata_access_object_container_.get(), output_artifact);
  Execution execution;
  CreateNodeFromTextProto("name: 'execution'", execution_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), execution);
  Event event1;
  CreateEventFromTextProto("type: INPUT", input_artifact, execution,
                           *metadata_access_object_,
                           metadata_access_object_container_.get(), event1);
  Event event2;
  CreateEventFromTextProto("type: OUTPUT", output_artifact, execution,
                           *metadata_access_object_,
                           metadata_access_object_container_.get(), event2);

  // Test: empty ids
  {
    std::vector<Event> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteEventsByExecutionsId({}));
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindEventsByExecutions(
                                    {execution.id()}, &result));
    EXPECT_THAT(result.size(), 2);
  }
  // Test: delete both events
  {
    std::vector<Event> result;
    EXPECT_EQ(
        absl::OkStatus(),
        metadata_access_object_->DeleteEventsByExecutionsId({execution.id()}));
    absl::Status status = metadata_access_object_->FindEventsByExecutions(
        {execution.id()}, &result);
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    EXPECT_THAT(result, IsEmpty());
    status = metadata_access_object_->FindEventsByArtifacts(
        {input_artifact.id(), output_artifact.id()}, &result);
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteAssociationsByContextsId) {
  ASSERT_EQ(absl::OkStatus(), Init());

  int64 execution_type_id = InsertType<ExecutionType>("execution_type");
  int64 context_type_id = InsertType<ContextType>("context_type");
  Execution execution;
  CreateNodeFromTextProto("name: 'execution'", execution_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), execution);
  Context context;
  CreateNodeFromTextProto("name: 'context'", context_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context);

  Association association;
  association.set_execution_id(execution.id());
  association.set_context_id(context.id());

  int64 association_id;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAssociation(
                                  association, &association_id));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Test: empty ids
  {
    std::vector<Execution> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteAssociationsByContextsId({}));
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindExecutionsByContext(context.id(),
                                                               &result));
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete association
  {
    std::vector<Execution> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteAssociationsByContextsId(
                  {context.id()}));
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindExecutionsByContext(context.id(),
                                                               &result));
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteAssociationsByExecutionsId) {
  ASSERT_EQ(absl::OkStatus(), Init());

  int64 execution_type_id = InsertType<ExecutionType>("execution_type");
  int64 context_type_id = InsertType<ContextType>("context_type");
  Execution execution;
  CreateNodeFromTextProto("name: 'execution'", execution_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), execution);
  Context context;
  CreateNodeFromTextProto("name: 'context'", context_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context);

  Association association;
  association.set_execution_id(execution.id());
  association.set_context_id(context.id());

  int64 association_id;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAssociation(
                                  association, &association_id));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Test: empty ids
  {
    std::vector<Context> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteAssociationsByExecutionsId({}));
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindContextsByExecution(execution.id(),
                                                               &result));
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete association
  {
    std::vector<Context> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteAssociationsByExecutionsId(
                  {execution.id()}));
    // Semantics for empty here is different than in FindExecutionsByContext.
    // Here if there is none, a notFound error is returned.
    absl::Status status = metadata_access_object_->FindContextsByExecution(
        execution.id(), &result);
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteAttributionsByContextsId) {
  ASSERT_EQ(absl::OkStatus(), Init());

  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64 context_type_id = InsertType<ContextType>("test_context_type");
  Artifact artifact;
  CreateNodeFromTextProto("name: 'artifact'", artifact_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), artifact);
  Context context;
  CreateNodeFromTextProto("name: 'context'", context_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context);

  Attribution attribution;
  attribution.set_artifact_id(artifact.id());
  attribution.set_context_id(context.id());

  int64 attribution_id;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAttribution(
                                  attribution, &attribution_id));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Test: empty ids
  {
    std::vector<Artifact> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteAttributionsByContextsId({}));
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsByContext(
                                    context.id(), &result));
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete attribution
  {
    std::vector<Artifact> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteAttributionsByContextsId(
                  {context.id()}));
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsByContext(
                                    context.id(), &result));
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteAttributionsByArtifactsId) {
  ASSERT_EQ(absl::OkStatus(), Init());

  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64 context_type_id = InsertType<ContextType>("test_context_type");
  Artifact artifact;
  CreateNodeFromTextProto("name: 'artifact'", artifact_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), artifact);
  Context context;
  CreateNodeFromTextProto("name: 'context'", context_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context);

  Attribution attribution;
  attribution.set_artifact_id(artifact.id());
  attribution.set_context_id(context.id());

  int64 attribution_id;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAttribution(
                                  attribution, &attribution_id));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Test: empty ids
  {
    std::vector<Context> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteAttributionsByArtifactsId({}));
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsByArtifact(
                                    artifact.id(), &result));
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete attribution
  {
    std::vector<Context> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteAttributionsByArtifactsId(
                  {artifact.id()}));
    // Semantics for empty here is different than in FindArtifactsByContext.
    // Here if there is none, a notFound error is returned.
    absl::Status status =
        metadata_access_object_->FindContextsByArtifact(artifact.id(), &result);
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteParentType) {
  ASSERT_EQ(absl::OkStatus(), Init());

  {
    // Test: create and delete artifact parent type inheritance link
    const ArtifactType type1 = CreateTypeFromTextProto<ArtifactType>(
        "name: 't1'", *metadata_access_object_);
    const ArtifactType type2 = CreateTypeFromTextProto<ArtifactType>(
        "name: 't2'", *metadata_access_object_);
    const ArtifactType type3 = CreateTypeFromTextProto<ArtifactType>(
        "name: 't3'", *metadata_access_object_);

    // create parent type links ok.
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type3));
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type2, type3));

    absl::flat_hash_map<int64, ArtifactType> output_artifact_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id(), type2.id()}, output_artifact_types));
    ASSERT_EQ(output_artifact_types.size(), 2);

    // delete parent link (type1, type3)
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteParentTypeInheritanceLink(
                  type1.id(), type3.id()));

    output_artifact_types.clear();
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id(), type2.id()}, output_artifact_types));
    ASSERT_EQ(output_artifact_types.size(), 1);
    EXPECT_TRUE(output_artifact_types.contains(type2.id()));

    // delete parent link (type2, type3)
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteParentTypeInheritanceLink(
                  type2.id(), type3.id()));

    output_artifact_types.clear();
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id()}, output_artifact_types));
    EXPECT_THAT(output_artifact_types, IsEmpty());
  }

  {
    // Test: create and delete execution parent type inheritance link
    const ExecutionType type1 = CreateTypeFromTextProto<ExecutionType>(
        "name: 't1'", *metadata_access_object_);
    const ExecutionType type2 = CreateTypeFromTextProto<ExecutionType>(
        "name: 't2'", *metadata_access_object_);
    // create parent type link ok.
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));

    // delete parent link (type1, type2)
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteParentTypeInheritanceLink(
                  type1.id(), type2.id()));

    absl::flat_hash_map<int64, ExecutionType> output_execution_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id()}, output_execution_types));
    EXPECT_TRUE(output_execution_types.empty());
  }

  {
    // Test: create and delete context parent type inheritance link
    const ContextType type1 = CreateTypeFromTextProto<ContextType>(
        "name: 't1'", *metadata_access_object_);
    const ContextType type2 = CreateTypeFromTextProto<ContextType>(
        "name: 't2'", *metadata_access_object_);
    // create parent type link ok.
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));

    // delete parent link (type1, type2)
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteParentTypeInheritanceLink(
                  type1.id(), type2.id()));

    absl::flat_hash_map<int64, ContextType> output_context_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id()}, output_context_types));
    EXPECT_TRUE(output_context_types.empty());
  }

  {
    // Test: delete non-existing context parent type inheritance link
    const ContextType type1 = CreateTypeFromTextProto<ContextType>(
        "name: 't1'", *metadata_access_object_);
    const ContextType type2 = CreateTypeFromTextProto<ContextType>(
        "name: 't2'", *metadata_access_object_);

    // delete non-existing parent link (type1, type2) returns ok
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteParentTypeInheritanceLink(
                  type1.id(), type2.id()));
  }
}

TEST_P(MetadataAccessObjectTest, DeleteParentContextsByParentIds) {
  ASSERT_EQ(absl::OkStatus(), Init());

  int64 context_type_id = InsertType<ContextType>("test_context_type");

  Context context1;
  CreateNodeFromTextProto("name: 'parent_context'", context_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context1);
  Context context2;
  CreateNodeFromTextProto("name: 'child_context'", context_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context2);

  ParentContext parent_context;
  parent_context.set_parent_id(context1.id());
  parent_context.set_child_id(context2.id());
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateParentContext(parent_context));

  // Test: empty ids
  {
    std::vector<Context> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteParentContextsByParentIds({}));
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentContextsByContextId(
                  context2.id(), &result));
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete parent context
  {
    std::vector<Context> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteParentContextsByParentIds(
                  {context1.id()}));
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentContextsByContextId(
                  context2.id(), &result));
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteParentContextsByChildIds) {
  ASSERT_EQ(absl::OkStatus(), Init());

  int64 context_type_id = InsertType<ContextType>("test_context_type");

  Context context1;
  CreateNodeFromTextProto("name: 'parent_context'", context_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context1);
  Context context2;
  CreateNodeFromTextProto("name: 'child_context'", context_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context2);

  ParentContext parent_context;
  parent_context.set_parent_id(context1.id());
  parent_context.set_child_id(context2.id());
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateParentContext(parent_context));

  // Test: empty ids
  {
    std::vector<Context> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteParentContextsByChildIds({}));
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentContextsByContextId(
                  context2.id(), &result));
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete parent context
  {
    std::vector<Context> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->DeleteParentContextsByChildIds(
                  {context2.id()}));
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentContextsByContextId(
                  context2.id(), &result));
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteParentContextsByParentIdAndChildIds) {
  MLMD_ASSERT_OK(Init());

  int64 context_type_id = InsertType<ContextType>("test_context_type");

  Context context1;
  CreateNodeFromTextProto("name: 'parent_context'", context_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context1);
  Context context2;
  CreateNodeFromTextProto("name: 'child_context'", context_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context2);

  Context context3;
  CreateNodeFromTextProto("name: 'independent_context'", context_type_id,
                          *metadata_access_object_,
                          metadata_access_object_container_.get(), context2);

  ParentContext parent_context;
  parent_context.set_parent_id(context1.id());
  parent_context.set_child_id(context2.id());
  MLMD_EXPECT_OK(metadata_access_object_->CreateParentContext(parent_context));

  // Test: independent parent id
  {
    std::vector<Context> result;
    MLMD_EXPECT_OK(metadata_access_object_->
                  DeleteParentContextsByParentIdAndChildIds(
                     context3.id(), {context2.id()}));
    MLMD_ASSERT_OK(metadata_access_object_->FindParentContextsByContextId(
                  context2.id(), &result));
    EXPECT_THAT(result, SizeIs(1));
  }
  // Test: empty child ids
  {
    std::vector<Context> result;
    MLMD_EXPECT_OK(metadata_access_object_->
                  DeleteParentContextsByParentIdAndChildIds(context1.id(), {}));
    MLMD_ASSERT_OK(metadata_access_object_->FindParentContextsByContextId(
                  context2.id(), &result));
    EXPECT_THAT(result, SizeIs(1));
  }
  // Test: delete parent context
  {
    std::vector<Context> result;
    MLMD_EXPECT_OK(metadata_access_object_->
                  DeleteParentContextsByParentIdAndChildIds(
                      context1.id(), {context2.id()}));
    MLMD_ASSERT_OK(metadata_access_object_->FindParentContextsByContextId(
                  context2.id(), &result));
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, ListArtifactsWithNonIdFieldOptions) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact sample_artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )");
  sample_artifact.set_type_id(type_id);
  const int total_stored_artifacts = 6;
  int64 last_stored_artifact_id;

  for (int i = 0; i < total_stored_artifacts; i++) {
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                    sample_artifact, &last_stored_artifact_id));
  }
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 2,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  int64 expected_artifact_id = last_stored_artifact_id;
  std::string next_page_token;

  do {
    std::vector<Artifact> got_artifacts;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->ListArtifacts(
                  list_options, &got_artifacts, &next_page_token));
    EXPECT_TRUE(got_artifacts.size() <= page_size);
    for (const Artifact& artifact : got_artifacts) {
      sample_artifact.set_id(expected_artifact_id--);
      EXPECT_THAT(artifact, EqualsProto(sample_artifact, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"}));
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_EQ(expected_artifact_id, 0);
}

TEST_P(MetadataAccessObjectTest, ListArtifactsWithIdFieldOptions) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact sample_artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )");

  sample_artifact.set_type_id(type_id);
  int stored_artifacts_count = 0;
  int64 first_artifact_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  sample_artifact, &first_artifact_id));
  stored_artifacts_count++;

  for (int i = 0; i < 6; i++) {
    int64 unused_artifact_id;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                    sample_artifact, &unused_artifact_id));
  }
  stored_artifacts_count += 6;
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 2,
        order_by_field: { field: ID is_asc: true }
      )");

  std::string next_page_token;
  int64 expected_artifact_id = first_artifact_id;
  int seen_artifacts_count = 0;
  do {
    std::vector<Artifact> got_artifacts;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->ListArtifacts(
                  list_options, &got_artifacts, &next_page_token));
    EXPECT_TRUE(got_artifacts.size() <= page_size);
    for (const Artifact& artifact : got_artifacts) {
      sample_artifact.set_id(expected_artifact_id++);

      EXPECT_THAT(artifact, EqualsProto(sample_artifact, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"}));
      seen_artifacts_count++;
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_EQ(stored_artifacts_count, seen_artifacts_count);
}

TEST_P(MetadataAccessObjectTest, ListArtifactsOnLastUpdateTime) {
  if (!metadata_access_object_container_->PerformExtendedTests()) {
    return;
  }
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact sample_artifact = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )pb");
  sample_artifact.set_type_id(type_id);
  const int total_stored_artifacts = 6;
  std::vector<int64> stored_artifact_ids;
  for (int i = 0; i < total_stored_artifacts; i++) {
    int64 created_artifact_id;
    absl::SleepFor(absl::Milliseconds(1));
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                    sample_artifact, &created_artifact_id));
    stored_artifact_ids.push_back(created_artifact_id);
  }

  // Setting the expected list in the order [3, 2, 1, 6, 5, 4]
  std::list<int64> expected_artifact_ids;
  for (int i = 3; i < total_stored_artifacts; i++) {
    expected_artifact_ids.push_front(stored_artifact_ids[i]);
  }

  sample_artifact.set_state(ml_metadata::Artifact::State::Artifact_State_LIVE);
  for (int i = 0; i < 3; i++) {
    sample_artifact.set_id(stored_artifact_ids[i]);
    absl::SleepFor(absl::Milliseconds(1));
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->UpdateArtifact(sample_artifact));
    expected_artifact_ids.push_front(stored_artifact_ids[i]);
  }
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 2,
        order_by_field: { field: LAST_UPDATE_TIME is_asc: false }
      )pb");

  std::string next_page_token;
  do {
    std::vector<Artifact> got_artifacts;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->ListArtifacts(
                  list_options, &got_artifacts, &next_page_token));
    EXPECT_LE(got_artifacts.size(), page_size);
    for (const Artifact& artifact : got_artifacts) {
      sample_artifact.set_id(expected_artifact_ids.front());
      EXPECT_THAT(artifact, EqualsProto(sample_artifact, /*ignore_fields=*/{
                                            "state", "create_time_since_epoch",
                                            "last_update_time_since_epoch"}));
      expected_artifact_ids.pop_front();
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_THAT(expected_artifact_ids, IsEmpty());
}

TEST_P(MetadataAccessObjectTest, ListArtifactsWithChangedOptions) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact sample_artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
  )");

  sample_artifact.set_type_id(type_id);
  int64 last_stored_artifact_id;

  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  sample_artifact, &last_stored_artifact_id));
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  sample_artifact, &last_stored_artifact_id));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  std::string next_page_token_string;
  std::vector<Artifact> got_artifacts;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->ListArtifacts(list_options, &got_artifacts,
                                                   &next_page_token_string));
  EXPECT_EQ(got_artifacts.size(), 1);
  EXPECT_EQ(got_artifacts[0].id(), last_stored_artifact_id);

  ListOperationOptions updated_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: true }
      )");

  updated_options.set_next_page_token(next_page_token_string);
  std::vector<Artifact> unused_artifacts;
  std::string unused_next_page_token;
  EXPECT_TRUE(absl::IsInvalidArgument(metadata_access_object_->ListArtifacts(
      updated_options, &unused_artifacts, &unused_next_page_token)));
}

TEST_P(MetadataAccessObjectTest, ListArtifactsWithInvalidNextPageToken) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact sample_artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
  )");

  sample_artifact.set_type_id(type_id);
  int64 last_stored_artifact_id;

  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  sample_artifact, &last_stored_artifact_id));
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  sample_artifact, &last_stored_artifact_id));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  std::string next_page_token_string;
  std::vector<Artifact> got_artifacts;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->ListArtifacts(list_options, &got_artifacts,
                                                   &next_page_token_string));
  EXPECT_EQ(got_artifacts.size(), 1);
  EXPECT_EQ(got_artifacts[0].id(), last_stored_artifact_id);

  list_options.set_next_page_token("Invalid String");
  std::vector<Artifact> unused_artifacts;
  std::string unused_next_page_token;
  EXPECT_TRUE(absl::IsInvalidArgument(metadata_access_object_->ListArtifacts(
      list_options, &unused_artifacts, &unused_next_page_token)));
}

TEST_P(MetadataAccessObjectTest, ListExecutionsWithNonIdFieldOptions) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Execution sample_execution = ParseTextProtoOrDie<Execution>(R"(
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )");
  sample_execution.set_type_id(type_id);
  const int total_stored_executions = 6;
  int64 last_stored_execution_id;

  for (int i = 0; i < total_stored_executions; i++) {
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateExecution(
                  sample_execution, &last_stored_execution_id));
  }
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 2,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  int64 expected_execution_id = last_stored_execution_id;
  std::string next_page_token;

  do {
    std::vector<Execution> got_executions;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->ListExecutions(
                  list_options, &got_executions, &next_page_token));
    EXPECT_TRUE(got_executions.size() <= page_size);
    for (const Execution& execution : got_executions) {
      sample_execution.set_id(expected_execution_id--);

      EXPECT_THAT(execution, EqualsProto(sample_execution, /*ignore_fields=*/{
                                             "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_EQ(expected_execution_id, 0);
}

TEST_P(MetadataAccessObjectTest, ListExecutionsWithIdFieldOptions) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Execution sample_execution = ParseTextProtoOrDie<Execution>(R"(
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )");

  sample_execution.set_type_id(type_id);
  int stored_executions_count = 0;
  int64 first_execution_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateExecution(
                                  sample_execution, &first_execution_id));
  stored_executions_count++;

  for (int i = 0; i < 6; i++) {
    int64 unused_execution_id;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateExecution(
                                    sample_execution, &unused_execution_id));
  }
  stored_executions_count += 6;
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 2,
        order_by_field: { field: ID is_asc: true }
      )");

  std::string next_page_token;
  int64 expected_execution_id = first_execution_id;
  int seen_executions_count = 0;
  do {
    std::vector<Execution> got_executions;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->ListExecutions(
                  list_options, &got_executions, &next_page_token));
    EXPECT_TRUE(got_executions.size() <= page_size);
    for (const Execution& execution : got_executions) {
      sample_execution.set_id(expected_execution_id++);

      EXPECT_THAT(execution, EqualsProto(sample_execution, /*ignore_fields=*/{
                                             "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
      seen_executions_count++;
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_EQ(stored_executions_count, seen_executions_count);
}

TEST_P(MetadataAccessObjectTest, ListContextsWithNonIdFieldOptions) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Context sample_context = ParseTextProtoOrDie<Context>(R"(
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )");
  sample_context.set_type_id(type_id);
  int64 last_stored_context_id;
  int context_name_suffix = 0;
  sample_context.set_name("list_contexts_test-1");
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateContext(
                                  sample_context, &last_stored_context_id));

  context_name_suffix++;
  sample_context.set_name("list_contexts_test-2");
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateContext(
                                  sample_context, &last_stored_context_id));
  context_name_suffix++;
  sample_context.set_name("list_contexts_test-3");
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateContext(
                                  sample_context, &last_stored_context_id));
  context_name_suffix++;
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 2,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  int64 expected_context_id = last_stored_context_id;
  std::string next_page_token;

  do {
    std::vector<Context> got_contexts;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->ListContexts(list_options, &got_contexts,
                                                    &next_page_token));
    EXPECT_TRUE(got_contexts.size() <= page_size);
    for (const Context& context : got_contexts) {
      sample_context.set_name(
          absl::StrCat("list_contexts_test-", context_name_suffix--));
      sample_context.set_id(expected_context_id--);
      EXPECT_THAT(context, EqualsProto(sample_context, /*ignore_fields=*/{
                                           "create_time_since_epoch",
                                           "last_update_time_since_epoch"}));
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_EQ(expected_context_id, 0);
}

TEST_P(MetadataAccessObjectTest, ListContextsWithIdFieldOptions) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Context sample_context = ParseTextProtoOrDie<Context>(R"(
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )");

  sample_context.set_type_id(type_id);
  int stored_contexts_count = 0;
  int64 first_context_id;
  sample_context.set_name("list_contexts_test-1");
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateContext(
                                  sample_context, &first_context_id));

  int64 unused_context_id;
  stored_contexts_count++;
  sample_context.set_name("list_contexts_test-2");
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateContext(
                                  sample_context, &unused_context_id));
  stored_contexts_count++;
  sample_context.set_name("list_contexts_test-3");
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateContext(
                                  sample_context, &unused_context_id));
  stored_contexts_count++;
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 2,
        order_by_field: { field: ID is_asc: true }
      )");

  std::string next_page_token;
  int64 expected_context_id = first_context_id;
  int expected_context_name_suffix = 1;
  int seen_contexts_count = 0;
  do {
    std::vector<Context> got_contexts;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->ListContexts(list_options, &got_contexts,
                                                    &next_page_token));
    EXPECT_TRUE(got_contexts.size() <= page_size);
    for (const Context& context : got_contexts) {
      sample_context.set_name(
          absl::StrCat("list_contexts_test-", expected_context_name_suffix++));
      sample_context.set_id(expected_context_id++);

      EXPECT_THAT(context, EqualsProto(sample_context, /*ignore_fields=*/{
                                           "create_time_since_epoch",
                                           "last_update_time_since_epoch"}));
      seen_contexts_count++;
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_EQ(stored_contexts_count, seen_contexts_count);
}

TEST_P(MetadataAccessObjectTest, GetContextsById) {
  ASSERT_EQ(absl::OkStatus(), Init());

  // Setup: create the type for the context
  int64 type_id;
  {
    ContextType type = ParseTextProtoOrDie<ContextType>(R"(
      name: 'test_type'
      properties { key: 'property_1' value: INT }
    )");
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateType(type, &type_id));
  }

  // Setup: Add first context instance
  Context first_context;
  {
    first_context = ParseTextProtoOrDie<Context>(R"(
      properties {
        key: 'property_1'
        value: { int_value: 3 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { string_value: 'foo' }
      }
    )");
    int64 first_context_id;
    first_context.set_type_id(type_id);
    first_context.set_name("get_contexts_by_id_test-1");
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateContext(
                                    first_context, &first_context_id));
    first_context.set_id(first_context_id);
  }

  // Setup: Add second context instance
  Context second_context;
  {
    second_context = ParseTextProtoOrDie<Context>(R"(
      properties {
        key: 'property_1'
        value: { int_value: 5 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { string_value: 'bar' }
      }
    )");
    int64 second_context_id;
    second_context.set_type_id(type_id);
    second_context.set_name("get_contexts_by_id_test-2");
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateContext(
                                    second_context, &second_context_id));
    second_context.set_id(second_context_id);
  }

  // Setup: Add third context instance that does not have *any* properties
  Context third_context;
  {
    int64 third_context_id;
    third_context.set_type_id(type_id);
    third_context.set_name("get_contexts_by_id_test-3");
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateContext(
                                    third_context, &third_context_id));
    third_context.set_id(third_context_id);
  }

  const int64 unknown_id =
      first_context.id() + second_context.id() + third_context.id() + 1;

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Test: empty ids
  {
    std::vector<Context> result;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindContextsById({}, &result));
    EXPECT_THAT(result, IsEmpty());
  }
  // Test: no results
  {
    std::vector<Context> result;
    absl::Status status =
        metadata_access_object_->FindContextsById({unknown_id}, &result);
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    EXPECT_THAT(result, IsEmpty());
  }
  // Test: retrieve a single context at a time
  {
    std::vector<Context> result;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {first_context.id()}, &result));
    EXPECT_THAT(result,
                ElementsAre(EqualsProto(first_context, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));
  }
  {
    std::vector<Context> result;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {second_context.id()}, &result));
    EXPECT_THAT(result,
                ElementsAre(EqualsProto(second_context, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));
  }
  {
    std::vector<Context> result;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {third_context.id()}, &result));
    EXPECT_THAT(result,
                ElementsAre(EqualsProto(third_context, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));
  }
  // Test: retrieve multiple contexts at a time
  {
    std::vector<Context> result;
    const std::vector<int64> ids = {first_context.id(), second_context.id(),
                                    unknown_id};
    absl::Status status =
        metadata_access_object_->FindContextsById(ids, &result);
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    EXPECT_THAT(
        result,
        UnorderedElementsAre(
            EqualsProto(first_context,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(second_context,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  {
    std::vector<Context> result;
    const std::vector<int64> ids = {first_context.id(), second_context.id(),
                                    third_context.id()};
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindContextsById(ids, &result));
    EXPECT_THAT(
        result,
        UnorderedElementsAre(
            EqualsProto(first_context,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(second_context,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(third_context,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
}

TEST_P(MetadataAccessObjectTest, DefaultArtifactState) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>("name: 'test_type'");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  // artifact 1 does not set the state
  Artifact want_artifact1;
  want_artifact1.set_uri("uri: 'testuri://testing/uri/1'");
  want_artifact1.set_type_id(type_id);
  int64 id1;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(want_artifact1, &id1));
  // artifact 2 sets the state to default UNKNOWN
  Artifact want_artifact2;
  want_artifact2.set_type_id(type_id);
  want_artifact2.set_uri("uri: 'testuri://testing/uri/2'");
  want_artifact2.set_state(Artifact::UNKNOWN);
  int64 id2;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(want_artifact2, &id2));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  std::vector<Artifact> artifacts;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->FindArtifacts(&artifacts));
  ASSERT_EQ(artifacts.size(), 2);
  EXPECT_THAT(artifacts[0], EqualsProto(want_artifact1, /*ignore_fields=*/{
                                            "id", "create_time_since_epoch",
                                            "last_update_time_since_epoch"}));
  EXPECT_THAT(artifacts[1],
              EqualsProto(want_artifact2,
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
}

TEST_P(MetadataAccessObjectTest, FindArtifactsByTypeIds) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 type_id = InsertType<ArtifactType>("test_type");
  Artifact want_artifact1 =
      ParseTextProtoOrDie<Artifact>("uri: 'testuri://testing/uri1'");
  want_artifact1.set_type_id(type_id);
  int64 artifact1_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  want_artifact1, &artifact1_id));

  Artifact want_artifact2 =
      ParseTextProtoOrDie<Artifact>("uri: 'testuri://testing/uri2'");
  want_artifact2.set_type_id(type_id);
  int64 artifact2_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  want_artifact2, &artifact2_id));

  int64 type2_id = InsertType<ArtifactType>("test_type2");
  Artifact artifact3;
  artifact3.set_type_id(type2_id);
  int64 artifact3_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact3, &artifact3_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  std::vector<Artifact> got_artifacts;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindArtifactsByTypeId(
                type_id, absl::nullopt, &got_artifacts, nullptr));
  EXPECT_EQ(got_artifacts.size(), 2);
  EXPECT_THAT(want_artifact1, EqualsProto(got_artifacts[0], /*ignore_fields=*/{
                                              "id", "create_time_since_epoch",
                                              "last_update_time_since_epoch"}));
  EXPECT_THAT(want_artifact2, EqualsProto(got_artifacts[1], /*ignore_fields=*/{
                                              "id", "create_time_since_epoch",
                                              "last_update_time_since_epoch"}));
}

TEST_P(MetadataAccessObjectTest, FindArtifactByTypeIdAndArtifactName) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 type_id = InsertType<ArtifactType>("test_type");
  Artifact want_artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri1'
    name: 'artifact1')");
  want_artifact.set_type_id(type_id);
  int64 artifact1_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  want_artifact, &artifact1_id));
  want_artifact.set_id(artifact1_id);

  Artifact artifact2 = ParseTextProtoOrDie<Artifact>(
      "uri: 'testuri://testing/uri2' name: 'artifact2'");
  artifact2.set_type_id(type_id);
  int64 artifact2_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact2, &artifact2_id));

  int64 type2_id = InsertType<ArtifactType>("test_type2");
  Artifact artifact3;
  artifact3.set_type_id(type2_id);
  int64 artifact3_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact3, &artifact3_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Artifact got_artifact;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindArtifactByTypeIdAndArtifactName(
                type_id, "artifact1", &got_artifact));
  EXPECT_THAT(want_artifact, EqualsProto(got_artifact, /*ignore_fields=*/{
                                             "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  Artifact got_empty_artifact;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindArtifactByTypeIdAndArtifactName(
          type_id, "unknown", &got_empty_artifact)));
  EXPECT_THAT(got_empty_artifact, EqualsProto(Artifact()));
}

TEST_P(MetadataAccessObjectTest, FindArtifactsByURI) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 type_id = InsertType<ArtifactType>("test_type");
  Artifact want_artifact1 = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri1'
    name: 'artifact1')");
  want_artifact1.set_type_id(type_id);
  int64 artifact1_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  want_artifact1, &artifact1_id));
  want_artifact1.set_id(artifact1_id);

  Artifact artifact2 = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri2'
    name: 'artifact2')");
  artifact2.set_type_id(type_id);
  int64 artifact2_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact2, &artifact2_id));
  artifact2.set_id(artifact2_id);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  std::vector<Artifact> got_artifacts;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsByURI(
                                  "testuri://testing/uri1", &got_artifacts));
  ASSERT_EQ(got_artifacts.size(), 1);
  EXPECT_THAT(want_artifact1, EqualsProto(got_artifacts[0], /*ignore_fields=*/{
                                              "create_time_since_epoch",
                                              "last_update_time_since_epoch"}));
}


TEST_P(MetadataAccessObjectTest, UpdateArtifact) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact stored_artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
    state: LIVE
  )");
  stored_artifact.set_type_id(type_id);
  int64 artifact_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  stored_artifact, &artifact_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Artifact got_artifact_before_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsById(
                                    {artifact_id}, &artifacts));
    got_artifact_before_update = artifacts.at(0);
  }
  EXPECT_THAT(got_artifact_before_update,
              EqualsProto(stored_artifact,
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  // update `property_1`, add `property_2`, and drop `property_3`
  // change the value type of `custom_property_1`
  Artifact updated_artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://changed/uri'
    properties {
      key: 'property_1'
      value: { int_value: 5 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { int_value: 3 }
    }
  )");
  updated_artifact.set_id(artifact_id);
  updated_artifact.set_type_id(type_id);
  // sleep to verify the latest update time is updated.
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->UpdateArtifact(updated_artifact));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Artifact got_artifact_after_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsById(
                                    {artifact_id}, &artifacts));
    got_artifact_after_update = artifacts.at(0);
  }
  EXPECT_THAT(got_artifact_after_update,
              EqualsProto(updated_artifact,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_artifact_before_update.create_time_since_epoch(),
            got_artifact_after_update.create_time_since_epoch());
  EXPECT_LT(got_artifact_before_update.last_update_time_since_epoch(),
            got_artifact_after_update.last_update_time_since_epoch());

}

TEST_P(MetadataAccessObjectTest, UpdateArtifactWithCustomUpdateTime) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact stored_artifact = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
    state: LIVE
  )pb");
  stored_artifact.set_type_id(type_id);
  int64 artifact_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  stored_artifact, &artifact_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Artifact got_artifact_before_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsById(
                                    {artifact_id}, &artifacts));
    got_artifact_before_update = artifacts.at(0);
  }
  EXPECT_THAT(got_artifact_before_update,
              EqualsProto(stored_artifact,
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  // update `property_1`, add `property_2`, and drop `property_3`
  // change the value type of `custom_property_1`
  Artifact updated_artifact = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://changed/uri'
    properties {
      key: 'property_1'
      value: { int_value: 5 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { int_value: 3 }
    }
  )pb");
  updated_artifact.set_id(artifact_id);
  updated_artifact.set_type_id(type_id);
  absl::Time update_time = absl::InfiniteFuture();
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->UpdateArtifact(
                                  updated_artifact, update_time));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Artifact got_artifact_after_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsById(
                                    {artifact_id}, &artifacts));
    got_artifact_after_update = artifacts.at(0);
  }
  EXPECT_THAT(got_artifact_after_update,
              EqualsProto(updated_artifact,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_artifact_before_update.create_time_since_epoch(),
            got_artifact_after_update.create_time_since_epoch());
  EXPECT_EQ(got_artifact_after_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, UpdateNodeLastUpdateTimeSinceEpoch) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'p1' value: INT }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));
  // Create the original artifact before update.
  Artifact artifact;
  artifact.set_uri("testuri://changed/uri");
  artifact.set_type_id(type_id);
  int64 artifact_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact, &artifact_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Artifact curr_artifact;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsById(
                                    {artifact_id}, &artifacts));
    curr_artifact = artifacts.at(0);
  }

  // insert executions and links to artifacts
  auto update_and_get_last_update_time_since_epoch =
      [&](const Artifact& artifact) {
        // sleep to verify the latest update time is updated.
        absl::SleepFor(absl::Milliseconds(1));
        CHECK_EQ(absl::OkStatus(),
                 metadata_access_object_->UpdateArtifact(artifact));
        CHECK_EQ(absl::OkStatus(), AddCommitPointIfNeeded());
        Artifact got_artifact_after_update;
        {
          std::vector<Artifact> artifacts;
          CHECK_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsById(
                                         {artifact.id()}, &artifacts));
          got_artifact_after_update = artifacts.at(0);
        }
        EXPECT_THAT(got_artifact_after_update,
                    EqualsProto(artifact, /*ignore_fields=*/{
                                    "last_update_time_since_epoch"}));
        return got_artifact_after_update.last_update_time_since_epoch();
      };

  // no attribute or property change.
  const int64 t0 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_EQ(t0, curr_artifact.last_update_time_since_epoch());
  // update attributes
  curr_artifact.set_uri("new/uri");
  const int64 t1 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t1, t0);
  // set attributes
  curr_artifact.set_state(Artifact::LIVE);
  const int64 t2 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t2, t1);
  // add property
  (*curr_artifact.mutable_properties())["p1"].set_int_value(1);
  const int64 t3 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t3, t2);
  // modify property
  (*curr_artifact.mutable_properties())["p1"].set_int_value(2);
  const int64 t4 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t4, t3);
  // delete property
  curr_artifact.clear_properties();
  const int64 t5 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t5, t4);
  // set custom property
  (*curr_artifact.mutable_custom_properties())["custom"].set_string_value("1");
  const int64 t6 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t6, t5);
  // modify custom property
  (*curr_artifact.mutable_custom_properties())["custom"].set_string_value("2");
  const int64 t7 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t7, t6);
  // delete custom property
  curr_artifact.clear_custom_properties();
  const int64 t8 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t8, t7);
}

TEST_P(MetadataAccessObjectTest, UpdateArtifactError) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Artifact artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
  )");
  artifact.set_type_id(type_id);
  int64 artifact_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact, &artifact_id));
  artifact.set_id(artifact_id);
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // no artifact id given
  Artifact wrong_artifact;
  absl::Status s = metadata_access_object_->UpdateArtifact(wrong_artifact);
  EXPECT_TRUE(absl::IsInvalidArgument(s));

  // artifact id cannot be found
  int64 different_id = artifact_id + 1;
  wrong_artifact.set_id(different_id);
  s = metadata_access_object_->UpdateArtifact(wrong_artifact);
  EXPECT_TRUE(absl::IsInvalidArgument(s));

  // type_id if given is not aligned with the stored one
  wrong_artifact.set_id(artifact_id);
  int64 different_type_id = type_id + 1;
  wrong_artifact.set_type_id(different_type_id);
  s = metadata_access_object_->UpdateArtifact(wrong_artifact);
  EXPECT_TRUE(absl::IsInvalidArgument(s));

  // artifact has unknown property
  wrong_artifact.clear_type_id();
  (*wrong_artifact.mutable_properties())["unknown_property"].set_int_value(1);
  s = metadata_access_object_->UpdateArtifact(wrong_artifact);
  EXPECT_TRUE(absl::IsInvalidArgument(s));
}

TEST_P(MetadataAccessObjectTest, CreateAndFindExecution) {
  ASSERT_EQ(absl::OkStatus(), Init());
  // Creates execution 1 with type 1
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type_with_predefined_property'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Execution want_execution1 = ParseTextProtoOrDie<Execution>(R"(
    name: "my_execution1"
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { int_value: 3 }
    }
  )");
  want_execution1.set_type_id(type_id);
  {
    int64 execution_id = -1;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateExecution(
                                    want_execution1, &execution_id));
    want_execution1.set_id(execution_id);
  }
  // Creates execution 2 with type 2
  int64 type2_id = InsertType<ExecutionType>("test_type_with_no_property");
  Execution want_execution2 = ParseTextProtoOrDie<Execution>(R"(
    name: "my_execution2"
    last_known_state: RUNNING
  )");
  want_execution2.set_type_id(type2_id);
  {
    int64 execution_id = -1;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateExecution(
                                    want_execution2, &execution_id));
    want_execution2.set_id(execution_id);
  }

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());


  EXPECT_NE(want_execution1.id(), want_execution2.id());

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Test: retrieve one execution at a time
  Execution got_execution1;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsById(
                                    {want_execution1.id()}, &executions));
    got_execution1 = executions.at(0);
    EXPECT_THAT(
        want_execution1,
        EqualsProto(got_execution1,
                    /*ignore_fields=*/{"id", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
    EXPECT_GT(got_execution1.create_time_since_epoch(), 0);
    EXPECT_GT(got_execution1.last_update_time_since_epoch(), 0);
    EXPECT_LE(got_execution1.last_update_time_since_epoch(),
              absl::ToUnixMillis(absl::Now()));
    EXPECT_GE(got_execution1.last_update_time_since_epoch(),
              got_execution1.create_time_since_epoch());
  }

  Execution got_execution2;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsById(
                                    {want_execution2.id()}, &executions));
    got_execution2 = executions.at(0);
    EXPECT_THAT(
        got_execution2,
        EqualsProto(want_execution2,
                    /*ignore_fields=*/{"id", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
    EXPECT_GT(got_execution2.create_time_since_epoch(), 0);
    EXPECT_GT(got_execution2.last_update_time_since_epoch(), 0);
    EXPECT_LE(got_execution2.last_update_time_since_epoch(),
              absl::ToUnixMillis(absl::Now()));
    EXPECT_GE(got_execution2.last_update_time_since_epoch(),
              got_execution2.create_time_since_epoch());
  }

  // Test: empty input
  {
    std::vector<Execution> executions;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindExecutionsById({}, &executions));
    EXPECT_THAT(executions, IsEmpty());
  }

  // Test: unknown id
  const int64 unknown_id = want_execution1.id() + want_execution2.id() + 1;
  {
    std::vector<Execution> executions;
    EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindExecutionsById(
        {unknown_id}, &executions)));
  }

  // Test: retrieve multiple by ids
  {
    std::vector<Execution> got_executions;
    EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindExecutionsById(
        {unknown_id, want_execution1.id(), want_execution2.id()},
        &got_executions)));
  }
  {
    std::vector<Execution> got_executions;
    EXPECT_EQ(
        absl::OkStatus(),
        metadata_access_object_->FindExecutionsById(
            {want_execution1.id(), want_execution2.id()}, &got_executions));
    EXPECT_THAT(
        got_executions,
        UnorderedElementsAre(
            EqualsProto(want_execution1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(want_execution2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  // Test: retrieve all executions
  {
    std::vector<Execution> got_executions;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindExecutions(&got_executions));
    EXPECT_THAT(
        got_executions,
        UnorderedElementsAre(
            EqualsProto(want_execution1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(want_execution2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }

  // Test: retrieve by type
  {
    std::vector<Execution> type1_executions;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindExecutionsByTypeId(
                  type_id, absl::nullopt, &type1_executions, nullptr));
    EXPECT_EQ(type1_executions.size(), 1);
    EXPECT_THAT(type1_executions[0], EqualsProto(got_execution1));
  }

  // Test: retrieve by type and name
  {
    Execution got_execution_from_type_and_name1;
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->FindExecutionByTypeIdAndExecutionName(
            type_id, "my_execution1", &got_execution_from_type_and_name1));
    EXPECT_THAT(got_execution_from_type_and_name1, EqualsProto(got_execution1));

    Execution got_execution_from_type_and_name2;
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->FindExecutionByTypeIdAndExecutionName(
            type2_id, "my_execution2", &got_execution_from_type_and_name2));
    EXPECT_THAT(got_execution_from_type_and_name2, EqualsProto(got_execution2));

    Execution got_empty_execution;
    EXPECT_TRUE(absl::IsNotFound(
        metadata_access_object_->FindExecutionByTypeIdAndExecutionName(
            type_id, "my_execution2", &got_empty_execution)));
    EXPECT_THAT(got_empty_execution, EqualsProto(Execution()));
  }

}

TEST_P(MetadataAccessObjectTest, CreateExecutionWithDuplicatedNameError) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType type;
  type.set_name("test_type");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Execution execution;
  execution.set_type_id(type_id);
  execution.set_name("test execution name");
  int64 execution_id;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateExecution(execution, &execution_id));
  // insert the same execution again to check the unique constraint
  absl::Status unique_constraint_violation_status =
      metadata_access_object_->CreateExecution(execution, &execution_id);
  EXPECT_EQ(CheckUniqueConstraintAndResetTransaction(
                unique_constraint_violation_status),
            absl::OkStatus());
}


TEST_P(MetadataAccessObjectTest, UpdateExecution) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Execution stored_execution = ParseTextProtoOrDie<Execution>(R"(
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
    last_known_state: RUNNING
  )");
  stored_execution.set_type_id(type_id);
  int64 execution_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateExecution(
                                  stored_execution, &execution_id));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Execution got_execution_before_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsById(
                                    {execution_id}, &executions));
    got_execution_before_update = executions.at(0);
  }
  EXPECT_THAT(got_execution_before_update,
              EqualsProto(stored_execution,
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  // add `property_1` and update `property_3`, and drop `custom_property_1`
  Execution updated_execution = ParseTextProtoOrDie<Execution>(R"(
    properties {
      key: 'property_1'
      value: { int_value: 5 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '5' }
    }
  )");
  updated_execution.set_id(execution_id);
  updated_execution.set_type_id(type_id);
  // sleep to verify the latest update time is updated.
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->UpdateExecution(updated_execution));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Execution got_execution_after_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsById(
                                    {execution_id}, &executions));
    got_execution_after_update = executions.at(0);
  }
  EXPECT_THAT(got_execution_after_update,
              EqualsProto(updated_execution,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_execution_before_update.create_time_since_epoch(),
            got_execution_after_update.create_time_since_epoch());
  EXPECT_LT(got_execution_before_update.last_update_time_since_epoch(),
            got_execution_after_update.last_update_time_since_epoch());
}

TEST_P(MetadataAccessObjectTest, UpdateExecutionWithCustomUpdateTime) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Execution stored_execution = ParseTextProtoOrDie<Execution>(R"pb(
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
    last_known_state: RUNNING
  )pb");
  stored_execution.set_type_id(type_id);
  int64 execution_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateExecution(
                                  stored_execution, &execution_id));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Execution got_execution_before_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsById(
                                    {execution_id}, &executions));
    got_execution_before_update = executions.at(0);
  }
  EXPECT_THAT(got_execution_before_update,
              EqualsProto(stored_execution,
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  // add `property_1` and update `property_3`, and drop `custom_property_1`
  Execution updated_execution = ParseTextProtoOrDie<Execution>(R"pb(
    properties {
      key: 'property_1'
      value: { int_value: 5 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '5' }
    }
  )pb");
  updated_execution.set_id(execution_id);
  updated_execution.set_type_id(type_id);
  absl::Time update_time = absl::InfiniteFuture();
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->UpdateExecution(
                                  updated_execution, update_time));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Execution got_execution_after_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsById(
                                    {execution_id}, &executions));
    got_execution_after_update = executions.at(0);
  }
  EXPECT_THAT(got_execution_after_update,
              EqualsProto(updated_execution,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_execution_before_update.create_time_since_epoch(),
            got_execution_after_update.create_time_since_epoch());
  EXPECT_EQ(got_execution_after_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, CreateAndFindContext) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType type1 = ParseTextProtoOrDie<ContextType>(R"(
    name: 'test_type_with_predefined_property'
    properties { key: 'property_1' value: INT }
  )");
  int64 type1_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type1, &type1_id));

  ContextType type2 = ParseTextProtoOrDie<ContextType>(R"(
    name: 'test_type_with_no_property'
  )");
  int64 type2_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type2, &type2_id));

  // Creates two contexts of different types
  Context context1 = ParseTextProtoOrDie<Context>(R"(
    name: "my_context1"
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { int_value: 3 }
    }
  )");
  context1.set_type_id(type1_id);
  int64 context1_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context1, &context1_id));
  context1.set_id(context1_id);

  Context context2 = ParseTextProtoOrDie<Context>(R"(
    name: "my_context2")");
  context2.set_type_id(type2_id);
  int64 context2_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context2, &context2_id));
  context2.set_id(context2_id);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());


  EXPECT_NE(context1_id, context2_id);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Find contexts
  Context got_context1;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {context1_id}, &contexts));
    ASSERT_THAT(contexts, SizeIs(1));
    got_context1 = contexts[0];
  }
  EXPECT_THAT(context1, EqualsProto(got_context1, /*ignore_fields=*/{
                                        "create_time_since_epoch",
                                        "last_update_time_since_epoch"}));
  EXPECT_GT(got_context1.create_time_since_epoch(), 0);
  EXPECT_GT(got_context1.last_update_time_since_epoch(), 0);
  EXPECT_LE(got_context1.last_update_time_since_epoch(),
            absl::ToUnixMillis(absl::Now()));
  EXPECT_GE(got_context1.last_update_time_since_epoch(),
            got_context1.create_time_since_epoch());

  std::vector<Context> got_contexts;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindContexts(&got_contexts));
  EXPECT_EQ(got_contexts.size(), 2);
  EXPECT_THAT(context1, EqualsProto(got_contexts[0], /*ignore_fields=*/{
                                        "create_time_since_epoch",
                                        "last_update_time_since_epoch"}));
  EXPECT_THAT(context2, EqualsProto(got_contexts[1], /*ignore_fields=*/{
                                        "create_time_since_epoch",
                                        "last_update_time_since_epoch"}));

  std::vector<Context> got_type2_contexts;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindContextsByTypeId(
                type2_id, /*list_options=*/absl::nullopt, &got_type2_contexts,
                /*next_page_token=*/nullptr));
  EXPECT_EQ(got_type2_contexts.size(), 1);
  EXPECT_THAT(got_type2_contexts[0], EqualsProto(got_contexts[1]));

  Context got_context_from_type_and_name1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindContextByTypeIdAndContextName(
                type1_id, "my_context1", /*id_only=*/false,
                &got_context_from_type_and_name1));
  EXPECT_THAT(got_context_from_type_and_name1, EqualsProto(got_contexts[0]));

  Context got_context_from_type_and_name2;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindContextByTypeIdAndContextName(
                type2_id, "my_context2", /*id_only=*/false,
                &got_context_from_type_and_name2));
  EXPECT_THAT(got_context_from_type_and_name2, EqualsProto(got_contexts[1]));
  Context got_empty_context;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindContextByTypeIdAndContextName(
          type1_id, "my_context2", /*id_only=*/false, &got_empty_context)));
  EXPECT_THAT(got_empty_context, EqualsProto(Context()));

  Context got_context_from_type_and_name_with_only_id;
  Context expected_context_from_type_and_name_with_only_id;
  expected_context_from_type_and_name_with_only_id.set_id(got_contexts[0].id());
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindContextByTypeIdAndContextName(
                type1_id, "my_context1", /*id_only=*/true,
                &got_context_from_type_and_name_with_only_id));
  EXPECT_THAT(got_context_from_type_and_name_with_only_id,
              EqualsProto(expected_context_from_type_and_name_with_only_id));
}

TEST_P(MetadataAccessObjectTest, ListArtifactsByType) {
  ASSERT_EQ(absl::OkStatus(), Init());

  // Setup: create an artifact type and insert two instances.
  int64 type_id;
  {
    ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
      name: 'test_type_with_predefined_property'
      properties { key: 'property_1' value: INT }
    )pb");

    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateType(type, &type_id));
  }
  Artifact entity_1;
  {
    entity_1 = ParseTextProtoOrDie<Artifact>(R"pb(
      name: "1a"
      properties {
        key: 'property_1'
        value: { int_value: 1 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { int_value: 3 }
      }
    )pb");
    entity_1.set_type_id(type_id);
    int64 entity_id = -1;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateArtifact(entity_1, &entity_id));
    entity_1.set_id(entity_id);
  }
  Artifact entity_2;
  {
    entity_2 = ParseTextProtoOrDie<Artifact>(R"pb(
      name: "1b"
      properties {
        key: 'property_1'
        value: { int_value: 2 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { int_value: 4 }
      }
    )pb");
    entity_2.set_type_id(type_id);
    int64 entity_id = -1;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateArtifact(entity_2, &entity_id));
    entity_2.set_id(entity_id);
  }

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Test: List entities by default ordering -- ID.
  {
    ListOperationOptions options;
    options.set_max_result_size(1);

    std::vector<Artifact> entities;
    std::string next_page_token;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsByTypeId(
                                    type_id, absl::make_optional(options),
                                    &entities, &next_page_token));
    EXPECT_THAT(next_page_token, Not(IsEmpty()));
    EXPECT_THAT(entities,
                ElementsAre(EqualsProto(entity_1, /*ignore_fields=*/{
                                            "uri", "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));

    entities.clear();
    options.set_next_page_token(next_page_token);
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsByTypeId(
                                    type_id, absl::make_optional(options),
                                    &entities, &next_page_token));
    EXPECT_THAT(next_page_token, IsEmpty());
    EXPECT_THAT(entities,
                ElementsAre(EqualsProto(
                    entity_2,
                    /*ignore_fields=*/{"uri", "create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
}

TEST_P(MetadataAccessObjectTest, ListExecutionsByType) {
  ASSERT_EQ(absl::OkStatus(), Init());

  // Setup: create an excution type and insert two instances.
  int64 type_id;
  {
    ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
      name: 'test_type_with_predefined_property'
      properties { key: 'property_1' value: INT }
    )pb");

    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateType(type, &type_id));
  }
  Execution entity_1;
  {
    entity_1 = ParseTextProtoOrDie<Execution>(R"pb(
      name: "1a"
      properties {
        key: 'property_1'
        value: { int_value: 1 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { int_value: 3 }
      }
    )pb");
    entity_1.set_type_id(type_id);
    int64 entity_id = -1;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateExecution(entity_1, &entity_id));
    entity_1.set_id(entity_id);
  }
  Execution entity_2;
  {
    entity_2 = ParseTextProtoOrDie<Execution>(R"pb(
      name: "1b"
      properties {
        key: 'property_1'
        value: { int_value: 2 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { int_value: 4 }
      }
    )pb");
    entity_2.set_type_id(type_id);
    int64 entity_id = -1;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateExecution(entity_2, &entity_id));
    entity_2.set_id(entity_id);
  }

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Test: List entities by default ordering -- ID.
  {
    ListOperationOptions options;
    options.set_max_result_size(1);

    std::vector<Execution> entities;
    std::string next_page_token;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsByTypeId(
                                    type_id, absl::make_optional(options),
                                    &entities, &next_page_token));
    EXPECT_THAT(next_page_token, Not(IsEmpty()));
    EXPECT_THAT(entities,
                ElementsAre(EqualsProto(entity_1, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));

    entities.clear();
    options.set_next_page_token(next_page_token);
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsByTypeId(
                                    type_id, absl::make_optional(options),
                                    &entities, &next_page_token));
    EXPECT_THAT(next_page_token, IsEmpty());
    EXPECT_THAT(entities,
                ElementsAre(EqualsProto(
                    entity_2,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
}

TEST_P(MetadataAccessObjectTest, ListContextsByType) {
  ASSERT_EQ(absl::OkStatus(), Init());

  // Setup: create a context type and insert two instances.
  int64 type_id;
  {
    ContextType type = ParseTextProtoOrDie<ContextType>(R"(
      name: 'test_type_with_predefined_property'
      properties { key: 'property_1' value: INT }
    )");

    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateType(type, &type_id));
  }
  Context context_1;
  {
    context_1 = ParseTextProtoOrDie<Context>(R"(
      name: "context_1a"
      properties {
        key: 'property_1'
        value: { int_value: 1 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { int_value: 3 }
      }
    )");
    context_1.set_type_id(type_id);
    int64 context_id = -1;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateContext(context_1, &context_id));
    context_1.set_id(context_id);
  }
  Context context_2;
  {
    context_2 = ParseTextProtoOrDie<Context>(R"(
      name: "context_1b"
      properties {
        key: 'property_1'
        value: { int_value: 2 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { int_value: 4 }
      }
    )");
    context_2.set_type_id(type_id);
    int64 context_id = -1;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateContext(context_2, &context_id));
    context_2.set_id(context_id);
  }

  // Setup: insert one more context type and an additional instance. This is
  // additional data that will not retrieved by the test queries.
  {
    int64 type2_id;
    ContextType type2 = ParseTextProtoOrDie<ContextType>(R"(
      name: 'test_type_with_no_property'
    )");
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateType(type2, &type2_id));

    Context context = ParseTextProtoOrDie<Context>(R"(
      name: "my_context2")");
    context.set_type_id(type2_id);
    int64 context_id = -1;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateContext(context, &context_id));
  }

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Test: List contexts by default ordering -- ID.
  {
    ListOperationOptions options;
    options.set_max_result_size(1);

    std::vector<Context> contexts;
    std::string next_page_token;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsByTypeId(
                                    type_id, absl::make_optional(options),
                                    &contexts, &next_page_token));
    EXPECT_THAT(next_page_token, Not(IsEmpty()));
    EXPECT_THAT(contexts,
                ElementsAre(EqualsProto(context_1, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));

    contexts.clear();
    options.set_next_page_token(next_page_token);
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsByTypeId(
                                    type_id, absl::make_optional(options),
                                    &contexts, &next_page_token));
    EXPECT_THAT(next_page_token, IsEmpty());
    EXPECT_THAT(
        contexts,
        ElementsAre(
            EqualsProto(context_2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  // Test: List contexts by reverse default ordering (ID)
  {
    ListOperationOptions options;
    options.mutable_order_by_field()->set_is_asc(false);
    options.set_max_result_size(1);

    std::vector<Context> contexts;
    std::string next_page_token;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsByTypeId(
                                    type_id, absl::make_optional(options),
                                    &contexts, &next_page_token));
    EXPECT_THAT(next_page_token, Not(IsEmpty()));
    EXPECT_THAT(contexts,
                ElementsAre(EqualsProto(context_2, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));

    contexts.clear();
    options.set_next_page_token(next_page_token);
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsByTypeId(
                                    type_id, absl::make_optional(options),
                                    &contexts, &next_page_token));
    EXPECT_THAT(next_page_token, IsEmpty());
    EXPECT_THAT(
        contexts,
        ElementsAre(
            EqualsProto(context_1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  // Test: List contexts through a big max-result size.
  {
    ListOperationOptions options;
    options.set_max_result_size(100);

    std::vector<Context> contexts;
    std::string next_page_token;
    ASSERT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsByTypeId(
                                    type_id, absl::make_optional(options),
                                    &contexts, &next_page_token));
    EXPECT_THAT(next_page_token, IsEmpty());
    EXPECT_THAT(
        contexts,
        ElementsAre(
            EqualsProto(context_1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(context_2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
}

TEST_P(MetadataAccessObjectTest, CreateContextError) {
  ASSERT_EQ(absl::OkStatus(), Init());
  Context context;
  int64 context_id;

  // unknown type specified
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->CreateContext(context, &context_id)));

  context.set_type_id(1);
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->CreateContext(context, &context_id)));

  ContextType type = ParseTextProtoOrDie<ContextType>(R"(
    name: 'test_type_disallow_custom_property'
    properties { key: 'property_1' value: INT }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  // type mismatch
  context.set_type_id(type_id);
  (*context.mutable_properties())["property_1"].set_string_value("3");
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->CreateContext(context, &context_id)));

  // empty name
  (*context.mutable_properties())["property_1"].set_int_value(3);
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->CreateContext(context, &context_id)));
}

TEST_P(MetadataAccessObjectTest, CreateContextWithDuplicatedNameError) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType type;
  type.set_name("test_type");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Context context;
  context.set_type_id(type_id);
  context.set_name("test context name");
  int64 context_id;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context, &context_id));
  // insert the same context again to check the unique constraint
  absl::Status unique_constraint_violation_status =
      metadata_access_object_->CreateContext(context, &context_id);
  EXPECT_EQ(CheckUniqueConstraintAndResetTransaction(
                unique_constraint_violation_status),
            absl::OkStatus());
}


TEST_P(MetadataAccessObjectTest, UpdateContext) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Context context1 = ParseTextProtoOrDie<Context>(R"(
    name: "before update name"
    properties {
      key: 'property_1'
      value: { int_value: 2 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )");
  context1.set_type_id(type_id);
  int64 context_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context1, &context_id));
  Context got_context_before_update;

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  {
    std::vector<Context> contexts;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {context_id}, &contexts));
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_before_update = contexts[0];
  }

  // add `property_2` and update `property_1`, and drop `custom_property_1`
  Context want_context = ParseTextProtoOrDie<Context>(R"(
    name: "after update name"
    properties {
      key: 'property_1'
      value: { int_value: 5 }
    }
    properties {
      key: 'property_2'
      value: { string_value: 'test' }
    }
  )");
  want_context.set_id(context_id);
  want_context.set_type_id(type_id);
  // sleep to verify the latest update time is updated.
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->UpdateContext(want_context));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Context got_context_after_update;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {context_id}, &contexts));
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_after_update = contexts[0];
  }
  EXPECT_THAT(want_context,
              EqualsProto(got_context_after_update,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_context_before_update.create_time_since_epoch(),
            got_context_after_update.create_time_since_epoch());
  EXPECT_LT(got_context_before_update.last_update_time_since_epoch(),
            got_context_after_update.last_update_time_since_epoch());
}

TEST_P(MetadataAccessObjectTest, UpdateContextWithCustomUpdatetime) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(type, &type_id));

  Context context1 = ParseTextProtoOrDie<Context>(R"pb(
    name: "before update name"
    properties {
      key: 'property_1'
      value: { int_value: 2 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )pb");
  context1.set_type_id(type_id);
  int64 context_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context1, &context_id));
  Context got_context_before_update;

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  {
    std::vector<Context> contexts;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {context_id}, &contexts));
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_before_update = contexts[0];
  }

  // add `property_2` and update `property_1`, and drop `custom_property_1`
  Context want_context = ParseTextProtoOrDie<Context>(R"pb(
    name: "after update name"
    properties {
      key: 'property_1'
      value: { int_value: 5 }
    }
    properties {
      key: 'property_2'
      value: { string_value: 'test' }
    }
  )pb");
  want_context.set_id(context_id);
  want_context.set_type_id(type_id);
  absl::Time update_time = absl::InfiniteFuture();
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->UpdateContext(want_context, update_time));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Context got_context_after_update;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {context_id}, &contexts));
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_after_update = contexts[0];
  }
  EXPECT_THAT(want_context,
              EqualsProto(got_context_after_update,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_context_before_update.create_time_since_epoch(),
            got_context_after_update.create_time_since_epoch());
  EXPECT_EQ(got_context_after_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, CreateAndUseAssociation) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 execution_type_id = InsertType<ExecutionType>("execution_type");
  int64 context_type_id = InsertType<ContextType>("context_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  (*execution.mutable_custom_properties())["custom"].set_int_value(3);
  Context context = ParseTextProtoOrDie<Context>("name: 'context_instance'");
  context.set_type_id(context_type_id);

  int64 execution_id, context_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateExecution(execution, &execution_id));
  execution.set_id(execution_id);
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context, &context_id));
  context.set_id(context_id);

  Association association;
  association.set_execution_id(execution_id);
  association.set_context_id(context_id);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  int64 association_id;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAssociation(
                                  association, &association_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  std::vector<Context> got_contexts;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsByExecution(
                                  execution_id, &got_contexts));
  ASSERT_EQ(got_contexts.size(), 1);
  EXPECT_THAT(context, EqualsProto(got_contexts[0], /*ignore_fields=*/{
                                       "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));

  std::vector<Execution> got_executions;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsByContext(
                                  context_id, &got_executions));
  ASSERT_EQ(got_executions.size(), 1);
  EXPECT_THAT(execution, EqualsProto(got_executions[0], /*ignore_fields=*/{
                                         "create_time_since_epoch",
                                         "last_update_time_since_epoch"}));

  std::vector<Artifact> got_artifacts;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsByContext(
                                  context_id, &got_artifacts));
  EXPECT_EQ(got_artifacts.size(), 0);
}


TEST_P(MetadataAccessObjectTest, GetAssociationUsingPagination) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 execution_type_id = InsertType<ExecutionType>("execution_type");
  int64 context_type_id = InsertType<ContextType>("context_type");
  Execution execution1;
  execution1.set_type_id(execution_type_id);
  (*execution1.mutable_custom_properties())["custom"].set_int_value(3);
  Execution execution2;
  execution2.set_type_id(execution_type_id);
  (*execution2.mutable_custom_properties())["custom"].set_int_value(5);

  Context context = ParseTextProtoOrDie<Context>("name: 'context_instance'");
  context.set_type_id(context_type_id);

  int64 execution_id_1, execution_id_2, context_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateExecution(
                                  execution1, &execution_id_1));
  execution1.set_id(execution_id_1);
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateExecution(
                                  execution2, &execution_id_2));
  execution2.set_id(execution_id_2);

  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context, &context_id));
  context.set_id(context_id);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Association association1;
  association1.set_execution_id(execution_id_1);
  association1.set_context_id(context_id);

  Association association2;
  association2.set_execution_id(execution_id_2);
  association2.set_context_id(context_id);

  int64 association_id_1;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAssociation(
                                  association1, &association_id_1));
  int64 association_id_2;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAssociation(
                                  association2, &association_id_2));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  std::string next_page_token;
  std::vector<Execution> got_executions;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindExecutionsByContext(
                context_id, list_options, &got_executions, &next_page_token));
  EXPECT_THAT(got_executions, SizeIs(1));
  EXPECT_THAT(execution2, EqualsProto(got_executions[0], /*ignore_fields=*/{
                                          "create_time_since_epoch",
                                          "last_update_time_since_epoch"}));
  ASSERT_FALSE(next_page_token.empty());

  list_options.set_next_page_token(next_page_token);
  got_executions.clear();
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindExecutionsByContext(
                context_id, list_options, &got_executions, &next_page_token));
  EXPECT_THAT(got_executions, SizeIs(1));
  EXPECT_THAT(execution1, EqualsProto(got_executions[0], /*ignore_fields=*/{
                                          "create_time_since_epoch",
                                          "last_update_time_since_epoch"}));
  ASSERT_TRUE(next_page_token.empty());
}

TEST_P(MetadataAccessObjectTest, GetAttributionUsingPagination) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 artifact_type_id = InsertType<ArtifactType>("artifact_type");
  int64 context_type_id = InsertType<ContextType>("context_type");
  Artifact artifact1;
  artifact1.set_uri("http://some_uri");
  artifact1.set_type_id(artifact_type_id);
  (*artifact1.mutable_custom_properties())["custom"].set_int_value(3);

  Artifact artifact2;
  artifact2.set_uri("http://some_uri");
  artifact2.set_type_id(artifact_type_id);
  (*artifact2.mutable_custom_properties())["custom"].set_int_value(5);

  Context context = ParseTextProtoOrDie<Context>("name: 'context_instance'");
  context.set_type_id(context_type_id);

  int64 artifact_id_1, artifact_id_2, context_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact1, &artifact_id_1));
  artifact1.set_id(artifact_id_1);
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact2, &artifact_id_2));
  artifact2.set_id(artifact_id_2);

  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context, &context_id));
  context.set_id(context_id);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Attribution attribution1;
  attribution1.set_artifact_id(artifact_id_1);
  attribution1.set_context_id(context_id);

  Attribution attribution2;
  attribution2.set_artifact_id(artifact_id_2);
  attribution2.set_context_id(context_id);

  int64 attribution_id_1;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAttribution(
                                  attribution1, &attribution_id_1));
  int64 attribution_id_2;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAttribution(
                                  attribution2, &attribution_id_2));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  std::string next_page_token;
  std::vector<Artifact> got_artifacts;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindArtifactsByContext(
                context_id, list_options, &got_artifacts, &next_page_token));
  EXPECT_THAT(got_artifacts, SizeIs(1));
  EXPECT_THAT(artifact2, EqualsProto(got_artifacts[0], /*ignore_fields=*/{
                                         "create_time_since_epoch",
                                         "last_update_time_since_epoch"}));
  ASSERT_FALSE(next_page_token.empty());

  got_artifacts.clear();
  list_options.set_next_page_token(next_page_token);
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindArtifactsByContext(
                context_id, list_options, &got_artifacts, &next_page_token));
  EXPECT_THAT(got_artifacts, SizeIs(1));
  ASSERT_TRUE(next_page_token.empty());
  EXPECT_THAT(artifact1, EqualsProto(got_artifacts[0], /*ignore_fields=*/{
                                         "create_time_since_epoch",
                                         "last_update_time_since_epoch"}));
}

TEST_P(MetadataAccessObjectTest, GetEmptyAttributionAssociationWithPagination) {
  ASSERT_EQ(absl::OkStatus(), Init());
  const ContextType context_type = CreateTypeFromTextProto<ContextType>(
      "name: 't1'", *metadata_access_object_);
  Context context = ParseTextProtoOrDie<Context>("name: 'c1'");
  context.set_type_id(context_type.id());
  int64 context_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context, &context_id));
  context.set_id(context_id);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  const ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");
  {
    std::vector<Artifact> got_artifacts;
    std::string next_page_token;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindArtifactsByContext(
                  context_id, list_options, &got_artifacts, &next_page_token));
    EXPECT_THAT(got_artifacts, IsEmpty());
  }

  {
    std::vector<Execution> got_executions;
    std::string next_page_token;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindExecutionsByContext(
                  context_id, list_options, &got_executions, &next_page_token));
    EXPECT_THAT(got_executions, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, CreateAssociationError) {
  ASSERT_EQ(absl::OkStatus(), Init());

  // Create base association with
  // * valid context id
  // * valid execution id
  int64 context_type_id = InsertType<ContextType>("test_context_type");
  Context context;
  context.set_type_id(context_type_id);
  context.set_name("test_context");
  int64 context_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context, &context_id));

  int64 execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  int64 execution_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateExecution(
                                  execution, &execution_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Association base_association;
  base_association.set_context_id(context_id);
  base_association.set_execution_id(execution_id);

  // no context id
  {
    Association association = base_association;
    association.clear_context_id();
    int64 association_id;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_access_object_->CreateAssociation(
            association, &association_id)));
  }

  // no execution id
  {
    Association association = base_association;
    association.clear_execution_id();
    int64 association_id;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_access_object_->CreateAssociation(
            association, &association_id)));
  }

  // the context cannot be found
  {
    Association association = base_association;
    int64 unknown_id = 12345;
    association.set_context_id(unknown_id);
    int64 association_id;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_access_object_->CreateAssociation(
            association, &association_id)));
  }

  // the execution cannot be found

  {
    Association association = base_association;
    int64 unknown_id = 12345;
    association.set_execution_id(unknown_id);
    int64 association_id;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_access_object_->CreateAssociation(
            association, &association_id)));
  }
}

// TODO(b/197686185): Remove test once foreign keys schema is implemented for
// CreateAssociation
TEST_P(MetadataAccessObjectTest, CreateAssociationWithoutValidation) {
  ASSERT_EQ(absl::OkStatus(), Init());

  int64 context_type_id = InsertType<ContextType>("test_context_type");
  Context context;
  context.set_type_id(context_type_id);
  context.set_name("test_context");
  int64 context_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context, &context_id));

  int64 execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  int64 execution_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateExecution(execution, &execution_id));

  // create association without validation (since the nodes are known to exist)
  Association association;
  association.set_context_id(context_id);
  association.set_execution_id(execution_id);
  int64 association_id;
  absl::Status create_new_association_without_validation_status =
      metadata_access_object_->CreateAssociation(
          association, /*is_already_validated=*/true, &association_id);
  EXPECT_EQ(absl::OkStatus(), create_new_association_without_validation_status);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // create duplicate association without validation
  int64 duplicate_association_id;
  absl::Status create_duplicate_association_without_validation_status =
      metadata_access_object_->CreateAssociation(association,
                                                 /*is_already_validated=*/true,
                                                 &duplicate_association_id);
  EXPECT_TRUE(absl::IsAlreadyExists(
      create_duplicate_association_without_validation_status));

  // create invalid association without validation
  // NOTE: This is an invalid use case, but is intended to break once foreign
  // key support is implemented in the schema.
  Association invalid_association;
  invalid_association.set_context_id(context_id + 1);
  invalid_association.set_execution_id(execution_id + 1);
  int64 invalid_association_id;
  absl::Status create_invalid_association_without_validation_status =
      metadata_access_object_->CreateAssociation(invalid_association,
                                                 /*is_already_validated=*/true,
                                                 &invalid_association_id);
  EXPECT_EQ(absl::OkStatus(),
            create_invalid_association_without_validation_status);
}

TEST_P(MetadataAccessObjectTest, CreateAttributionError) {
  ASSERT_EQ(absl::OkStatus(), Init());

  // Create base attribution with
  // * valid context id
  // * valid artifact id
  int64 context_type_id = InsertType<ContextType>("test_context_type");
  Context context;
  context.set_type_id(context_type_id);
  context.set_name("test_context");
  int64 context_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateContext(
                                  context, &context_id));

  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  Artifact artifact;
  artifact.set_type_id(artifact_type_id);
  int64 artifact_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact, &artifact_id));

  Attribution base_attribution;
  base_attribution.set_context_id(context_id);
  base_attribution.set_artifact_id(artifact_id);

  // no context id
  {
    Attribution attribution = base_attribution;
    attribution.clear_context_id();
    int64 attribution_id;
    absl::Status s = metadata_access_object_->CreateAttribution(
        attribution, &attribution_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // no artifact id
  {
    Attribution attribution = base_attribution;
    attribution.clear_artifact_id();
    int64 attribution_id;
    absl::Status s = metadata_access_object_->CreateAttribution(
        attribution, &attribution_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // the context cannot be found
  {
    Attribution attribution = base_attribution;
    int64 unknown_id = 12345;
    attribution.set_context_id(unknown_id);
    int64 attribution_id;
    absl::Status s = metadata_access_object_->CreateAttribution(
        attribution, &attribution_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // the artifact cannot be found
  {
    Attribution attribution = base_attribution;
    int64 unknown_id = 12345;
    attribution.set_artifact_id(unknown_id);
    int64 attribution_id;
    absl::Status s = metadata_access_object_->CreateAttribution(
        attribution, &attribution_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }
}

// TODO(b/197686185): Remove test once foreign keys schema is implemented for
// CreateAttribution
TEST_P(MetadataAccessObjectTest, CreateAttributionWithoutValidation) {
  ASSERT_EQ(absl::OkStatus(), Init());

  int64 context_type_id = InsertType<ContextType>("test_context_type");
  Context context;
  context.set_type_id(context_type_id);
  context.set_name("test_context");
  int64 context_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context, &context_id));

  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  Artifact artifact;
  artifact.set_type_id(artifact_type_id);
  int64 artifact_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact, &artifact_id));

  // create attribution without validation (since the nodes are known to exist)
  Attribution attribution;
  attribution.set_context_id(context_id);
  attribution.set_artifact_id(artifact_id);
  int64 attribution_id;
  absl::Status create_new_attribution_without_validation_status =
      metadata_access_object_->CreateAttribution(
          attribution, /*is_already_validated=*/true, &attribution_id);
  EXPECT_EQ(absl::OkStatus(), create_new_attribution_without_validation_status);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // create duplicate attribution without validation
  int64 duplicate_attribution_id;
  absl::Status create_duplicate_attribution_without_validation_status =
      metadata_access_object_->CreateAttribution(attribution,
                                                 /*is_already_validated=*/true,
                                                 &duplicate_attribution_id);
  EXPECT_TRUE(absl::IsAlreadyExists(
      create_duplicate_attribution_without_validation_status));

  // create invalid attribution without validation
  // NOTE: This is an invalid use case, but is intended to break once foreign
  // key support is implemented in the schema.
  Attribution invalid_attribution;
  invalid_attribution.set_context_id(context_id + 1);
  invalid_attribution.set_artifact_id(artifact_id + 1);
  int64 invalid_attribution_id;
  absl::Status create_invalid_attribution_without_validation_status =
      metadata_access_object_->CreateAttribution(invalid_attribution,
                                                 /*is_already_validated=*/true,
                                                 &invalid_attribution_id);
  EXPECT_EQ(absl::OkStatus(),
            create_invalid_attribution_without_validation_status);
}

TEST_P(MetadataAccessObjectTest, CreateAssociationError2) {
  ASSERT_EQ(absl::OkStatus(), Init());
  Association association;
  int64 association_id;
  // duplicated association
  int64 execution_type_id = InsertType<ExecutionType>("execution_type");
  int64 context_type_id = InsertType<ContextType>("context_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  Context context = ParseTextProtoOrDie<Context>("name: 'context_instance'");
  context.set_type_id(context_type_id);
  int64 execution_id, context_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateExecution(execution, &execution_id));
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context, &context_id));
  association.set_execution_id(execution_id);
  association.set_context_id(context_id);
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // first insertion succeeds
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAssociation(
                                  association, &association_id));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());
  // second insertion fails
  EXPECT_TRUE(absl::IsAlreadyExists(metadata_access_object_->CreateAssociation(
      association, &association_id)));

  ASSERT_EQ(absl::OkStatus(), metadata_source_->Rollback());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
}

TEST_P(MetadataAccessObjectTest, CreateAndUseAttribution) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64 context_type_id = InsertType<ContextType>("test_context_type");

  Artifact artifact;
  artifact.set_uri("testuri");
  artifact.set_type_id(artifact_type_id);
  (*artifact.mutable_custom_properties())["custom"].set_string_value("str");
  Context context = ParseTextProtoOrDie<Context>("name: 'context_instance'");
  context.set_type_id(context_type_id);

  int64 artifact_id, context_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateArtifact(artifact, &artifact_id));
  artifact.set_id(artifact_id);
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context, &context_id));
  context.set_id(context_id);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  Attribution attribution;
  attribution.set_artifact_id(artifact_id);
  attribution.set_context_id(context_id);

  int64 attribution_id;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAttribution(
                                  attribution, &attribution_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  std::vector<Context> got_contexts;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsByArtifact(
                                  artifact_id, &got_contexts));
  ASSERT_EQ(got_contexts.size(), 1);
  EXPECT_THAT(context, EqualsProto(got_contexts[0], /*ignore_fields=*/{
                                       "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));

  std::vector<Artifact> got_artifacts;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsByContext(
                                  context_id, &got_artifacts));
  ASSERT_EQ(got_artifacts.size(), 1);
  EXPECT_THAT(artifact, EqualsProto(got_artifacts[0], /*ignore_fields=*/{
                                        "create_time_since_epoch",
                                        "last_update_time_since_epoch"}));

  std::vector<Execution> got_executions;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsByContext(
                                  context_id, &got_executions));
  EXPECT_EQ(got_executions.size(), 0);
}

TEST_P(MetadataAccessObjectTest, CreateAndFindEvent) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64 execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Artifact input_artifact;
  input_artifact.set_type_id(artifact_type_id);
  int64 input_artifact_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  input_artifact, &input_artifact_id));

  Artifact output_artifact;
  output_artifact.set_type_id(artifact_type_id);
  int64 output_artifact_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  output_artifact, &output_artifact_id));

  Execution execution;
  execution.set_type_id(execution_type_id);
  int64 execution_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateExecution(execution, &execution_id));
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // event1 with event paths
  Event event1 = ParseTextProtoOrDie<Event>("type: INPUT");
  event1.set_artifact_id(input_artifact_id);
  event1.set_execution_id(execution_id);
  event1.set_milliseconds_since_epoch(12345);
  event1.mutable_path()->add_steps()->set_index(1);
  event1.mutable_path()->add_steps()->set_key("key");
  int64 event1_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event1, &event1_id),
            absl::OkStatus());
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // event2 with optional fields
  Event event2 = ParseTextProtoOrDie<Event>("type: OUTPUT");
  event2.set_artifact_id(output_artifact_id);
  event2.set_execution_id(execution_id);
  int64 event2_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event2, &event2_id),
            absl::OkStatus());
  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  EXPECT_NE(event1_id, -1);
  EXPECT_NE(event2_id, -1);
  EXPECT_NE(event1_id, event2_id);

  // query the events
  std::vector<Event> events_with_artifacts;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindEventsByArtifacts(
                                  {input_artifact_id, output_artifact_id},
                                  &events_with_artifacts));
  EXPECT_EQ(events_with_artifacts.size(), 2);
  EXPECT_THAT(
      events_with_artifacts,
      UnorderedElementsAre(
          EqualsProto(event1),
          EqualsProto(event2, /*ignore_fields=*/{"milliseconds_since_epoch"})));

  std::vector<Event> events_with_execution;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindEventsByExecutions(
                                  {execution_id}, &events_with_execution));
  EXPECT_EQ(events_with_execution.size(), 2);
}

TEST_P(MetadataAccessObjectTest, FindEventsByArtifactsNotFound) {
  ASSERT_EQ(absl::OkStatus(), Init());
  std::vector<Event> events;
  const absl::Status no_artifact_ids_status =
      metadata_access_object_->FindEventsByArtifacts(
          /*artifact_ids=*/{}, &events);
  EXPECT_TRUE(absl::IsNotFound(no_artifact_ids_status));

  const absl::Status not_exist_id_status =
      metadata_access_object_->FindEventsByArtifacts(
          /*artifact_ids=*/{1}, &events);
  EXPECT_TRUE(absl::IsNotFound(not_exist_id_status));
}

TEST_P(MetadataAccessObjectTest, FindEventsByExecutionsNotFound) {
  ASSERT_EQ(absl::OkStatus(), Init());
  std::vector<Event> events;
  const absl::Status empty_execution_ids_status =
      metadata_access_object_->FindEventsByExecutions(
          /*execution_ids=*/{}, &events);
  EXPECT_TRUE(absl::IsNotFound(empty_execution_ids_status));

  const absl::Status not_exist_id_status =
      metadata_access_object_->FindEventsByExecutions(
          /*execution_ids=*/{1}, &events);
  EXPECT_TRUE(absl::IsNotFound(not_exist_id_status));
}

TEST_P(MetadataAccessObjectTest, CreateEventError) {
  ASSERT_EQ(Init(), absl::OkStatus());

  // Create base event with
  // * valid artifact id
  // * valid execution id
  // * event_type
  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  Artifact artifact;
  artifact.set_type_id(artifact_type_id);
  int64 artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());

  int64 execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  int64 execution_id;
  ASSERT_EQ(metadata_access_object_->CreateExecution(execution, &execution_id),
            absl::OkStatus());

  Event base_event;
  base_event.set_artifact_id(artifact_id);
  base_event.set_execution_id(execution_id);
  base_event.set_type(Event::INPUT);

  // no artifact id
  {
    Event event = base_event;
    event.clear_artifact_id();
    int64 event_id;
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // no execution id
  {
    Event event = base_event;
    event.clear_execution_id();
    int64 event_id;
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // no event type
  {
    Event event = base_event;
    event.clear_type();
    int64 event_id;
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // artifact cannot be found
  {
    Event event = base_event;
    int64 unknown_id = 12345;
    int64 event_id;
    event.set_artifact_id(unknown_id);
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // execution cannot be found
  {
    Event event = base_event;
    int64 unknown_id = 12345;
    int64 event_id;
    event.set_execution_id(unknown_id);
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }
}

// TODO(b/197686185): Remove test once foreign keys schema is implemented for
// CreateEvent
TEST_P(MetadataAccessObjectTest, CreateEventWithoutValidation) {
  ASSERT_EQ(Init(), absl::OkStatus());

  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  Artifact artifact;
  artifact.set_type_id(artifact_type_id);
  int64 artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());

  int64 execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  int64 execution_id;
  ASSERT_EQ(metadata_access_object_->CreateExecution(execution, &execution_id),
            absl::OkStatus());

  // insert event without validating (since the nodes are known to exist)
  Event event;
  event.set_artifact_id(artifact_id);
  event.set_execution_id(execution_id);
  event.set_type(Event::INPUT);
  int64 event_id;
  absl::Status create_new_event_without_validation_status =
      metadata_access_object_->CreateEvent(event, /*is_already_validated=*/true,
                                           &event_id);
  EXPECT_EQ(create_new_event_without_validation_status, absl::OkStatus());

  // insert invalid event without validation
  // NOTE: This is an invalid use case, but is intended to break once foreign
  // key support is implemented in the schema.
  Event invalid_event;
  int64 invalid_event_id;
  invalid_event.set_artifact_id(artifact_id + 1);
  invalid_event.set_execution_id(execution_id + 1);
  invalid_event.set_type(Event::INPUT);
  absl::Status create_invalid_event_without_validation_status =
      metadata_access_object_->CreateEvent(
          invalid_event, /*is_already_validated=*/true, &invalid_event_id);
  EXPECT_EQ(create_invalid_event_without_validation_status, absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, PutEventsWithPaths) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64 execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Artifact input_artifact;
  input_artifact.set_type_id(artifact_type_id);
  int64 input_artifact_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  input_artifact, &input_artifact_id));

  Artifact output_artifact;
  output_artifact.set_type_id(artifact_type_id);
  int64 output_artifact_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  output_artifact, &output_artifact_id));

  Execution execution;
  execution.set_type_id(execution_type_id);
  int64 execution_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateExecution(execution, &execution_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // event1 with event paths
  Event event1 = ParseTextProtoOrDie<Event>("type: INPUT");
  event1.set_artifact_id(input_artifact_id);
  event1.set_execution_id(execution_id);
  event1.set_milliseconds_since_epoch(12345);
  event1.mutable_path()->add_steps()->set_index(1);
  event1.mutable_path()->add_steps()->set_key("key");
  int64 event1_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event1, &event1_id),
            absl::OkStatus());

  // event2 with optional fields
  Event event2 = ParseTextProtoOrDie<Event>("type: OUTPUT");
  event2.set_artifact_id(output_artifact_id);
  event2.set_execution_id(execution_id);
  event2.mutable_path()->add_steps()->set_index(2);
  event2.mutable_path()->add_steps()->set_key("output_key");

  int64 event2_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event2, &event2_id),
            absl::OkStatus());

  EXPECT_NE(event1_id, -1);
  EXPECT_NE(event2_id, -1);
  EXPECT_NE(event1_id, event2_id);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // query the events
  std::vector<Event> events_with_artifacts;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindEventsByArtifacts(
                                  {input_artifact_id, output_artifact_id},
                                  &events_with_artifacts));
  EXPECT_EQ(events_with_artifacts.size(), 2);
  EXPECT_THAT(
      events_with_artifacts,
      UnorderedElementsAre(
          EqualsProto(event1),
          EqualsProto(event2, /*ignore_fields=*/{"milliseconds_since_epoch"})));

  std::vector<Event> events_with_execution;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindEventsByExecutions(
                                  {execution_id}, &events_with_execution));
  EXPECT_EQ(events_with_execution.size(), 2);
}

TEST_P(MetadataAccessObjectTest, CreateDuplicatedEvents) {
  // Support after Spanner upgrade schema to V8.
  if (!metadata_access_object_container_->HasFilterQuerySupport()) {
    return;
  }
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64 execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Artifact input_artifact;
  input_artifact.set_type_id(artifact_type_id);
  int64 input_artifact_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  input_artifact, &input_artifact_id));

  Artifact output_artifact;
  output_artifact.set_type_id(artifact_type_id);
  int64 output_artifact_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateArtifact(
                                  output_artifact, &output_artifact_id));

  Execution execution;
  execution.set_type_id(execution_type_id);
  int64 execution_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateExecution(execution, &execution_id));

  // event1 with event paths
  Event event1 = ParseTextProtoOrDie<Event>("type: INPUT");
  event1.set_artifact_id(input_artifact_id);
  event1.set_execution_id(execution_id);
  event1.set_milliseconds_since_epoch(12345);
  event1.mutable_path()->add_steps()->set_index(1);
  event1.mutable_path()->add_steps()->set_key("key");
  int64 event1_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event1, &event1_id),
            absl::OkStatus());
  EXPECT_NE(event1_id, -1);

  // event2 with same artifact_id, execution_id but different type.
  Event event2 = ParseTextProtoOrDie<Event>("type: DECLARED_INPUT");
  event2.set_artifact_id(input_artifact_id);
  event2.set_execution_id(execution_id);
  int64 event2_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event2, &event2_id),
            absl::OkStatus());
  EXPECT_NE(event2_id, -1);
  EXPECT_NE(event1_id, event2_id);

  // event3 with same artifact_id, execution_id and type.
  Event event3 = ParseTextProtoOrDie<Event>("type: INPUT");
  event3.set_artifact_id(input_artifact_id);
  event3.set_execution_id(execution_id);
  int64 unused_event3_id = -1;
  EXPECT_TRUE(absl::IsAlreadyExists(
      metadata_access_object_->CreateEvent(event3, &unused_event3_id)));

  // query the events
  std::vector<Event> events_with_artifacts;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindEventsByArtifacts(
                                  {input_artifact_id},
                                  &events_with_artifacts));
  EXPECT_EQ(events_with_artifacts.size(), 2);
  EXPECT_THAT(
      events_with_artifacts,
      UnorderedElementsAre(
          EqualsProto(event1),
          EqualsProto(event2, /*ignore_fields=*/{"milliseconds_since_epoch"})));

  std::vector<Event> events_with_execution;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindEventsByExecutions(
                                  {execution_id}, &events_with_execution));
  EXPECT_EQ(events_with_execution.size(), 2);
  EXPECT_THAT(
      events_with_execution,
      UnorderedElementsAre(
          EqualsProto(event1),
          EqualsProto(event2, /*ignore_fields=*/{"milliseconds_since_epoch"})));
}

TEST_P(MetadataAccessObjectTest, CreateParentContext) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType context_type;
  context_type.set_name("context_type_name");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(context_type, &type_id));
  Context context1, context2;
  context1.set_name("parent_context");
  context1.set_type_id(type_id);
  int64 context1_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context1, &context1_id));
  context2.set_name("child_context");
  context2.set_type_id(type_id);
  int64 context2_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateContext(context2, &context2_id));

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  ParentContext parent_context;
  parent_context.set_parent_id(context1_id);
  parent_context.set_child_id(context2_id);
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateParentContext(parent_context));

  // recreate the same context returns AlreadyExists
  const absl::Status status =
      metadata_access_object_->CreateParentContext(parent_context);
  EXPECT_TRUE(absl::IsAlreadyExists(status));
}

TEST_P(MetadataAccessObjectTest, CreateParentContextInvalidArgumentError) {
  // Prepare a stored context.
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType context_type;
  context_type.set_name("context_type_name");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(context_type, &type_id));
  Context context;
  context.set_name("parent_context");
  context.set_type_id(type_id);
  int64 stored_context_id;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateContext(
                                  context, &stored_context_id));
  int64 not_exist_context_id = stored_context_id + 1;
  int64 not_exist_context_id_2 = stored_context_id + 2;

  // Enumerate the case of parent context requests which are invalid
  auto verify_is_invalid_argument = [this](absl::string_view case_name,
                                           absl::optional<int64> parent_id,
                                           absl::optional<int64> child_id) {
    ParentContext parent_context;
    if (parent_id) {
      parent_context.set_parent_id(parent_id.value());
    }
    if (child_id) {
      parent_context.set_child_id(child_id.value());
    }
    const absl::Status status =
        metadata_access_object_->CreateParentContext(parent_context);
    EXPECT_TRUE(absl::IsInvalidArgument(status)) << case_name;
  };

  verify_is_invalid_argument(/*case_name=*/"no parent id, no child id",
                             /*parent_id=*/absl::nullopt,
                             /*child_id=*/absl::nullopt);
  verify_is_invalid_argument(/*case_name=*/"no parent id",
                             /*parent_id=*/stored_context_id,
                             /*child_id=*/absl::nullopt);
  verify_is_invalid_argument(/*case_name=*/"no child id",
                             /*parent_id=*/absl::nullopt,
                             /*child_id=*/stored_context_id);
  verify_is_invalid_argument(/*case_name=*/"both parent and child id not found",
                             /*parent_id=*/not_exist_context_id,
                             /*child_id=*/not_exist_context_id_2);
  verify_is_invalid_argument(/*case_name=*/"parent id not found",
                             /*parent_id=*/not_exist_context_id,
                             /*child_id=*/stored_context_id);
  verify_is_invalid_argument(/*case_name=*/"child id not found",
                             /*parent_id=*/stored_context_id,
                             /*child_id=*/not_exist_context_id);
}

TEST_P(MetadataAccessObjectTest, CreateAndFindParentContext) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType context_type;
  context_type.set_name("context_type_name");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(context_type, &type_id));
  // Create some contexts to insert parent context relationship.
  const int num_contexts = 5;
  std::vector<Context> contexts(num_contexts);
  for (int i = 0; i < num_contexts; i++) {
    contexts[i].set_name(absl::StrCat("context", i));
    contexts[i].set_type_id(type_id);
    int64 ctx_id;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateContext(contexts[i], &ctx_id));
    contexts[i].set_id(ctx_id);
  }

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Populate a list of parent contexts and capture expected results of number
  // of parents and children per context.
  absl::node_hash_map<int, std::vector<Context>> want_parents;
  std::unordered_map<int, std::vector<Context>> want_children;
  auto put_parent_context = [this, &contexts, &want_parents, &want_children](
                                int64 parent_idx, int64 child_idx) {
    ParentContext parent_context;
    parent_context.set_parent_id(contexts[parent_idx].id());
    parent_context.set_child_id(contexts[child_idx].id());
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateParentContext(parent_context));
    want_parents[child_idx].push_back(contexts[parent_idx]);
    want_children[parent_idx].push_back(contexts[child_idx]);
  };
  put_parent_context(/*parent_idx=*/0, /*child_idx=*/1);
  put_parent_context(/*parent_idx=*/0, /*child_idx=*/2);
  put_parent_context(/*parent_idx=*/0, /*child_idx=*/3);
  put_parent_context(/*parent_idx=*/2, /*child_idx=*/3);
  put_parent_context(/*parent_idx=*/4, /*child_idx=*/3);

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Verify the results by look up contexts
  for (int i = 0; i < num_contexts; i++) {
    const Context& curr_context = contexts[i];
    std::vector<Context> got_parents, got_children;
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentContextsByContextId(
                  curr_context.id(), &got_parents));
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->FindChildContextsByContextId(
                  curr_context.id(), &got_children));
    EXPECT_THAT(got_parents, SizeIs(want_parents[i].size()));
    EXPECT_THAT(got_children, SizeIs(want_children[i].size()));
    EXPECT_THAT(got_parents,
                UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                                       "create_time_since_epoch",
                                       "last_update_time_since_epoch"}),
                                   want_parents[i]));
  }
}

TEST_P(MetadataAccessObjectTest, CreateParentContextInheritanceLinkWithCycle) {
  ASSERT_EQ(absl::OkStatus(), Init());
  ContextType context_type;
  context_type.set_name("context_type_name");
  int64 type_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateType(context_type, &type_id));
  // Creates some contexts for parent context relationship.
  const int num_contexts = 5;
  std::vector<Context> contexts(num_contexts);
  for (int i = 0; i < num_contexts; i++) {
    contexts[i].set_name(absl::StrCat("context", i));
    contexts[i].set_type_id(type_id);
    int64 ctx_id;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateContext(contexts[i], &ctx_id));
    contexts[i].set_id(ctx_id);
  }

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  auto set_and_return_parent_context = [](int parent_id, int child_id) {
    ParentContext parent_context;
    parent_context.set_parent_id(parent_id);
    parent_context.set_child_id(child_id);
    return parent_context;
  };

  auto verify_insert_parent_context_is_invalid = [this](const Context& parent,
                                                        const Context& child) {
    ParentContext parent_context;
    parent_context.set_parent_id(parent.id());
    parent_context.set_child_id(child.id());
    const absl::Status status =
        metadata_access_object_->CreateParentContext(parent_context);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  };

  ASSERT_EQ(absl::OkStatus(), AddCommitPointIfNeeded());

  // Cannot add self as parent context.
  verify_insert_parent_context_is_invalid(
      /*parent=*/contexts[0], /*child=*/contexts[0]);

  // context0 -> context1
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateParentContext(
                                  set_and_return_parent_context(
                                      /*parent_id=*/contexts[0].id(),
                                      /*child_id=*/contexts[1].id())));

  // Cannot have bi-direction parent context.
  verify_insert_parent_context_is_invalid(
      /*parent=*/contexts[1], /*child=*/contexts[0]);

  // context0 -> context1 -> context2
  //         \-> context3 -> context4
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateParentContext(
                                  set_and_return_parent_context(
                                      /*parent_id=*/contexts[1].id(),
                                      /*child_id=*/contexts[2].id())));
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateParentContext(
                                  set_and_return_parent_context(
                                      /*parent_id=*/contexts[0].id(),
                                      /*child_id=*/contexts[3].id())));
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->CreateParentContext(
                                  set_and_return_parent_context(
                                      /*parent_id=*/contexts[3].id(),
                                      /*child_id=*/contexts[4].id())));

  // Cannot have cyclic parent context.
  verify_insert_parent_context_is_invalid(
      /*parent=*/contexts[2], /*child=*/contexts[0]);
  verify_insert_parent_context_is_invalid(
      /*parent=*/contexts[4], /*child=*/contexts[0]);
}

TEST_P(MetadataAccessObjectTest, MigrateToCurrentLibVersion) {
  // Skip upgrade/downgrade migration tests for earlier schema version.
  if (EarlierSchemaEnabled() || SkipSchemaMigrationTests()) {
    return;
  }
  // setup the database using the previous version.
  // Calling this with the minimum version sets up the original database.
  int64 lib_version = metadata_access_object_->GetLibraryVersion();
  for (int64 i = metadata_access_object_container_->MinimumVersion();
       i <= lib_version; i++) {
    if (!metadata_access_object_container_->HasUpgradeVerification(i)) {
      continue;
    }
    MLMD_ASSERT_OK(
        metadata_access_object_container_->SetupPreviousVersionForUpgrade(i));
    if (i > 1) continue;
    // when i = 0, it is v0.13.2. At that time, the MLMDEnv table does not
    // exist, GetSchemaVersion resolves the current version as 0.
    int64 v0_13_2_version = 100;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->GetSchemaVersion(&v0_13_2_version));
    ASSERT_EQ(0, v0_13_2_version);
  }

  // expect to have an error when connecting an older database version without
  // enabling upgrade migration
  absl::Status status =
      metadata_access_object_->InitMetadataSourceIfNotExists();
  ASSERT_TRUE(absl::IsFailedPrecondition(status))
      << "Error: " << status.message();

  ASSERT_EQ(absl::OkStatus(), metadata_source_->Commit());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());

  // then init the store and the migration queries runs.
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->InitMetadataSourceIfNotExists(
                /*enable_upgrade_migration=*/true));
  // at the end state, schema version should becomes the library version and
  // all migration queries should all succeed.
  int64 curr_version = 0;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->GetSchemaVersion(&curr_version));
  ASSERT_EQ(lib_version, curr_version);
  // check the verification queries in the previous version scheme
  if (metadata_access_object_container_->HasUpgradeVerification(lib_version)) {
    MLMD_ASSERT_OK(
        metadata_access_object_container_->UpgradeVerification(lib_version));
  }
  MLMD_ASSERT_OK(
      metadata_access_object_container_->VerifyDbSchema(lib_version));
}

TEST_P(MetadataAccessObjectTest, DowngradeToV0FromCurrentLibVersion) {
  // Skip upgrade/downgrade migration tests for earlier schema version.
  if (EarlierSchemaEnabled() || SkipSchemaMigrationTests()) {
    return;
  }
  // should not use downgrade when the database is empty.
  EXPECT_TRUE(
      absl::IsInvalidArgument(metadata_access_object_->DowngradeMetadataSource(
          /*to_schema_version=*/0)));
  // init the database to the current library version.
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->InitMetadataSourceIfNotExists());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Commit());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
  int64 lib_version = metadata_access_object_->GetLibraryVersion();
  MLMD_ASSERT_OK(
      metadata_access_object_container_->VerifyDbSchema(lib_version));
  int64 curr_version = 0;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->GetSchemaVersion(&curr_version));
  EXPECT_EQ(curr_version, lib_version);

  // downgrade one version at a time and verify the state.
  for (int i = lib_version - 1; i >= 0; i--) {
    // set the pre-migration states of i+1 version.
    if (!metadata_access_object_container_->HasDowngradeVerification(i)) {
      continue;
    }
    MLMD_ASSERT_OK(
        metadata_access_object_container_->SetupPreviousVersionForDowngrade(i));
    // downgrade
    MLMD_ASSERT_OK(metadata_access_object_->DowngradeMetadataSource(i));

    MLMD_ASSERT_OK(metadata_access_object_container_->DowngradeVerification(i));
    // verify the db schema version
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->GetSchemaVersion(&curr_version));
    EXPECT_EQ(curr_version, i);
    MLMD_ASSERT_OK(metadata_access_object_container_->VerifyDbSchema(i));
  }
}

TEST_P(MetadataAccessObjectTest, AutoMigrationTurnedOffByDefault) {
  // Skip upgrade/downgrade migration tests for earlier schema version.
  if (EarlierSchemaEnabled() || SkipSchemaMigrationTests()) {
    return;
  }
  // init the database to the current library version.
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->InitMetadataSourceIfNotExists());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Commit());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
  // downgrade when the database to version 0.
  int64 current_library_version = metadata_access_object_->GetLibraryVersion();
  if (current_library_version ==
      metadata_access_object_container_->MinimumVersion()) {
    return;
  }
  const int64 to_schema_version = current_library_version - 1;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object_->DowngradeMetadataSource(
                                  to_schema_version));
  int64 db_version = -1;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->GetSchemaVersion(&db_version));
  ASSERT_EQ(db_version, to_schema_version);
  // connect earlier version db by default should fail with FailedPrecondition.
  absl::Status status =
      metadata_access_object_->InitMetadataSourceIfNotExists();
  EXPECT_TRUE(absl::IsFailedPrecondition(status));
}


}  // namespace
}  // namespace testing
}  // namespace ml_metadata

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
