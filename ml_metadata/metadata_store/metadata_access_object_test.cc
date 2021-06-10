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

#include "gflags/gflags.h"
#include <glog/logging.h>
#include "google/protobuf/repeated_field.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {
namespace testing {

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
void CreateNodeFromTextProto(const std::string& node_text_proto, int64 type_id,
                             MetadataAccessObject& metadata_access_object,
                             Node& output);

template <>
void CreateNodeFromTextProto(const std::string& node_text_proto, int64 type_id,
                             MetadataAccessObject& metadata_access_object,
                             Artifact& output) {
  Artifact node = ParseTextProtoOrDie<Artifact>(node_text_proto);
  node.set_type_id(type_id);
  int64 node_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.CreateArtifact(node, &node_id));
  std::vector<Artifact> nodes;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.FindArtifactsById({node_id}, &nodes));
  ASSERT_THAT(nodes, SizeIs(1));
  output = nodes[0];
}

template <>
void CreateNodeFromTextProto(const std::string& node_text_proto, int64 type_id,
                             MetadataAccessObject& metadata_access_object,
                             Execution& output) {
  Execution node = ParseTextProtoOrDie<Execution>(node_text_proto);
  node.set_type_id(type_id);
  int64 node_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.CreateExecution(node, &node_id));
  std::vector<Execution> nodes;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.FindExecutionsById({node_id}, &nodes));
  ASSERT_THAT(nodes, SizeIs(1));
  output = nodes[0];
}

template <>
void CreateNodeFromTextProto(const std::string& node_text_proto, int64 type_id,
                             MetadataAccessObject& metadata_access_object,
                             Context& output) {
  Context node = ParseTextProtoOrDie<Context>(node_text_proto);
  node.set_type_id(type_id);
  int64 node_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.CreateContext(node, &node_id));
  std::vector<Context> nodes;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.FindContextsById({node_id}, &nodes));
  ASSERT_THAT(nodes, SizeIs(1));
  output = nodes[0];
}

void CreateEventFromTextProto(const std::string& event_text_proto,
                              const Artifact& artifact,
                              const Execution& execution,
                              MetadataAccessObject& metadata_access_object,
                              Event& output_event) {
  output_event = ParseTextProtoOrDie<Event>("type: INPUT");
  output_event.set_artifact_id(artifact.id());
  output_event.set_execution_id(execution.id());
  int64 dummy_id;
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object.CreateEvent(output_event, &dummy_id));
}

// Utilities that waits for a millisecond to update a node and returns stored
// node proto with updated timestamps.
template <class Node>
void UpdateAndReturnNode(const Node& updated_node,
                         MetadataAccessObject& metadata_access_object,
                         Node& output);

template <>
void UpdateAndReturnNode(const Artifact& updated_node,
                         MetadataAccessObject& metadata_access_object,
                         Artifact& output) {
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object.UpdateArtifact(updated_node));
  std::vector<Artifact> artifacts;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object.FindArtifactsById(
                                  {updated_node.id()}, &artifacts));
  ASSERT_THAT(artifacts, SizeIs(1));
  output = artifacts.at(0);
}

template <>
void UpdateAndReturnNode(const Execution& updated_node,
                         MetadataAccessObject& metadata_access_object,
                         Execution& output) {
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object.UpdateExecution(updated_node));
  std::vector<Execution> executions;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object.FindExecutionsById(
                                  {updated_node.id()}, &executions));
  ASSERT_THAT(executions, SizeIs(1));
  output = executions.at(0);
}

template <>
void UpdateAndReturnNode(const Context& updated_node,
                         MetadataAccessObject& metadata_access_object,
                         Context& output) {
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object.UpdateContext(updated_node));
  std::vector<Context> contexts;
  ASSERT_EQ(absl::OkStatus(), metadata_access_object.FindContextsById(
                                  {updated_node.id()}, &contexts));
  ASSERT_THAT(contexts, SizeIs(1));
  output = contexts.at(0);
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
  if (EarlierSchemaEnabled()) { return; }
  // creates the schema and insert some records
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->InitMetadataSourceIfNotExists());
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
  if (EarlierSchemaEnabled()) { return; }
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
  if (EarlierSchemaEnabled()) { return; }
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
  if (EarlierSchemaEnabled()) { return; }
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
  if (EarlierSchemaEnabled()) { return; }
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
  if (!metadata_access_object_container_->HasParentTypeSupport()) {
    return;
  }
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
  if (!metadata_access_object_container_->HasParentTypeSupport()) {
    return;
  }
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
  if (!metadata_access_object_container_->HasParentTypeSupport()) {
    return;
  }
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
  if (!metadata_access_object_container_->HasParentTypeSupport()) {
    return;
  }
  ASSERT_EQ(absl::OkStatus(), Init());
  // Setup: init the store with the following types and inheritance links
  // ArtifactType:  type1 -> type2
  //                     \-> type3
  // ExecutionType: type4 -> type5
  // ContextType:   type6 -> type7
  //                type8
  const ArtifactType type1 = CreateTypeFromTextProto<ArtifactType>(R"(
          name: 't1'
          properties { key: 'property_1' value: STRING }
      )", *metadata_access_object_);
  const ArtifactType type2 = CreateTypeFromTextProto<ArtifactType>(R"(
          name: 't2'
          properties { key: 'property_2' value: INT }
      )", *metadata_access_object_);
  const ArtifactType type3 = CreateTypeFromTextProto<ArtifactType>(R"(
          name: 't3'
          properties { key: 'property_3' value: DOUBLE }
      )", *metadata_access_object_);
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type1, type3));

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
    std::vector<ArtifactType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(type1.id(),
                                                               parent_types));
    EXPECT_THAT(parent_types,
                UnorderedElementsAre(EqualsProto(type2), EqualsProto(type3)));
  }

  {
    std::vector<ArtifactType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(type2.id(),
                                                               parent_types));
    EXPECT_THAT(parent_types, IsEmpty());
  }

  {
    std::vector<ArtifactType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(type3.id(),
                                                               parent_types));
    EXPECT_THAT(parent_types, IsEmpty());
  }

  // verify execution types
  {
    std::vector<ExecutionType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(type4.id(),
                                                               parent_types));
    EXPECT_THAT(parent_types,
                UnorderedPointwise(EqualsProto<ExecutionType>(), {type5}));
  }

  {
    std::vector<ExecutionType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(type5.id(),
                                                               parent_types));
    EXPECT_THAT(parent_types, IsEmpty());
  }

  // verify context types
  {
    std::vector<ContextType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(type6.id(),
                                                               parent_types));
    EXPECT_THAT(parent_types,
                UnorderedPointwise(EqualsProto<ContextType>(), {type7}));
  }

  {
    std::vector<ContextType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(type7.id(),
                                                               parent_types));
    EXPECT_THAT(parent_types, IsEmpty());
  }

  {
    std::vector<ContextType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->FindParentTypesByTypeId(type8.id(),
                                                               parent_types));
    EXPECT_THAT(parent_types, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, FindParentTypesByTypeIdError) {
  if (!metadata_access_object_container_->HasParentTypeSupport()) {
    return;
  }
  ASSERT_EQ(absl::OkStatus(), Init());
  const int64 stored_artifact_type_id = InsertType<ArtifactType>("t1");
  const int64 stored_execution_type_id = InsertType<ExecutionType>("t1");
  const int64 stored_context_type_id = InsertType<ContextType>("t1");
  const int64 unknown_artifact_type_id = stored_artifact_type_id + 1;
  const int64 unknown_execution_type_id = stored_execution_type_id + 1;
  const int64 unknown_context_type_id = stored_context_type_id + 1;

  {
    std::vector<ArtifactType> parent_artifact_types;
    const absl::Status status =
        metadata_access_object_->FindParentTypesByTypeId(
            unknown_artifact_type_id, parent_artifact_types);
    EXPECT_TRUE(absl::IsNotFound(status));
  }

  {
    std::vector<ExecutionType> parent_execution_types;
    const absl::Status status =
        metadata_access_object_->FindParentTypesByTypeId(
            unknown_execution_type_id, parent_execution_types);
    EXPECT_TRUE(absl::IsNotFound(status));
  }

  {
    std::vector<ContextType> parent_context_types;
    const absl::Status status =
        metadata_access_object_->FindParentTypesByTypeId(
            unknown_context_type_id, parent_context_types);
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

TEST_P(MetadataAccessObjectTest, FindTypeById) {
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
  ASSERT_TRUE(absl::IsAlreadyExists(
      metadata_access_object_->CreateArtifact(artifact, &artifact_id)));
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Rollback());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
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

  std::vector<Artifact> got_artifacts;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindArtifactsByTypeId(
                                  type_id, &got_artifacts));
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
  EXPECT_NE(want_execution1.id(), want_execution2.id());

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
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindExecutionsByTypeId(
                                    type_id, &type1_executions));
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
  ASSERT_TRUE(absl::IsAlreadyExists(
      metadata_access_object_->CreateExecution(execution, &execution_id)));

  ASSERT_EQ(absl::OkStatus(), metadata_source_->Rollback());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
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

  EXPECT_NE(context1_id, context2_id);

  // Find contexts
  Context got_context1;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {context1_id}, &contexts));
    ASSERT_THAT(contexts, ::testing::SizeIs(1));
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
                type1_id, "my_context1", &got_context_from_type_and_name1));
  EXPECT_THAT(got_context_from_type_and_name1, EqualsProto(got_contexts[0]));

  Context got_context_from_type_and_name2;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->FindContextByTypeIdAndContextName(
                type2_id, "my_context2", &got_context_from_type_and_name2));
  EXPECT_THAT(got_context_from_type_and_name2, EqualsProto(got_contexts[1]));
  Context got_empty_context;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindContextByTypeIdAndContextName(
          type1_id, "my_context2", &got_empty_context)));

  EXPECT_THAT(got_empty_context, EqualsProto(Context()));
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
  ASSERT_TRUE(absl::IsAlreadyExists(
      metadata_access_object_->CreateContext(context, &context_id)));

  ASSERT_EQ(absl::OkStatus(), metadata_source_->Rollback());
  ASSERT_EQ(absl::OkStatus(), metadata_source_->Begin());
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
  {
    std::vector<Context> contexts;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {context_id}, &contexts));
    ASSERT_THAT(contexts, ::testing::SizeIs(1));
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

  Context got_context_after_update;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(absl::OkStatus(), metadata_access_object_->FindContextsById(
                                    {context_id}, &contexts));
    ASSERT_THAT(contexts, ::testing::SizeIs(1));
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

  int64 association_id;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAssociation(
                                  association, &association_id));

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
  Association association;
  int64 association_id;
  // no context id
  EXPECT_TRUE(
      absl::IsInvalidArgument(metadata_access_object_->CreateAssociation(
          association, &association_id)));

  // no execution id
  association.set_context_id(100);
  EXPECT_TRUE(
      absl::IsInvalidArgument(metadata_access_object_->CreateAssociation(
          association, &association_id)));

  // the context or execution cannot be found
  association.set_execution_id(100);
  EXPECT_TRUE(
      absl::IsInvalidArgument(metadata_access_object_->CreateAssociation(
          association, &association_id)));
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

  // first insertion succeeds
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAssociation(
                                  association, &association_id));
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

  Attribution attribution;
  attribution.set_artifact_id(artifact_id);
  attribution.set_context_id(context_id);

  int64 attribution_id;
  EXPECT_EQ(absl::OkStatus(), metadata_access_object_->CreateAttribution(
                                  attribution, &attribution_id));

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

  // event1 with event paths
  Event event1 = ParseTextProtoOrDie<Event>("type: INPUT");
  event1.set_artifact_id(input_artifact_id);
  event1.set_execution_id(execution_id);
  event1.set_milliseconds_since_epoch(12345);
  event1.mutable_path()->add_steps()->set_index(1);
  event1.mutable_path()->add_steps()->set_key("key");
  int64 event1_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateEvent(event1, &event1_id));

  // event2 with optional fields
  Event event2 = ParseTextProtoOrDie<Event>("type: OUTPUT");
  event2.set_artifact_id(output_artifact_id);
  event2.set_execution_id(execution_id);
  int64 event2_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateEvent(event2, &event2_id));

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
  ASSERT_EQ(absl::OkStatus(), Init());

  // no artifact id
  {
    Event event;
    int64 event_id;
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // no execution id
  {
    Event event;
    int64 event_id;
    event.set_artifact_id(1);
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // no event type
  {
    Event event;
    int64 event_id;
    event.set_artifact_id(1);
    event.set_execution_id(1);
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // artifact or execution cannot be found
  {
    int64 artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
    Artifact artifact;
    artifact.set_type_id(artifact_type_id);
    int64 artifact_id;
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->CreateArtifact(artifact, &artifact_id));

    Event event;
    int64 event_id;
    event.set_artifact_id(artifact_id);
    int64 unknown_id = 12345;
    event.set_execution_id(unknown_id);
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }
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

  // event1 with event paths
  Event event1 = ParseTextProtoOrDie<Event>("type: INPUT");
  event1.set_artifact_id(input_artifact_id);
  event1.set_execution_id(execution_id);
  event1.set_milliseconds_since_epoch(12345);
  event1.mutable_path()->add_steps()->set_index(1);
  event1.mutable_path()->add_steps()->set_key("key");
  int64 event1_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateEvent(event1, &event1_id));

  // event2 with optional fields
  Event event2 = ParseTextProtoOrDie<Event>("type: OUTPUT");
  event2.set_artifact_id(output_artifact_id);
  event2.set_execution_id(execution_id);
  event2.mutable_path()->add_steps()->set_index(2);
  event2.mutable_path()->add_steps()->set_key("output_key");

  int64 event2_id = -1;
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->CreateEvent(event2, &event2_id));

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

TEST_P(MetadataAccessObjectTest, MigrateToCurrentLibVersion) {
  // Skip upgrade/downgrade migration tests for earlier schema version.
  if (EarlierSchemaEnabled()) { return; }
  // setup the database using the previous version.
  // Calling this with the minimum version sets up the original database.
  int64 lib_version = metadata_access_object_->GetLibraryVersion();
  for (int64 i = metadata_access_object_container_->MinimumVersion();
       i <= lib_version; i++) {
    if (!metadata_access_object_container_->HasUpgradeVerification(i)) {
      continue;
    }
    ASSERT_EQ(
        absl::OkStatus(),
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
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_container_->UpgradeVerification(lib_version));
  }
}

TEST_P(MetadataAccessObjectTest, DowngradeToV0FromCurrentLibVersion) {
  // Skip upgrade/downgrade migration tests for earlier schema version.
  if (EarlierSchemaEnabled()) { return; }
  // should not use downgrade when the database is empty.
  EXPECT_TRUE(
      absl::IsInvalidArgument(metadata_access_object_->DowngradeMetadataSource(
          /*to_schema_version=*/0)));
  // init the database to the current library version.
  EXPECT_EQ(absl::OkStatus(),
            metadata_access_object_->InitMetadataSourceIfNotExists());
  int64 lib_version = metadata_access_object_->GetLibraryVersion();
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
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_container_->SetupPreviousVersionForDowngrade(i));
    // downgrade
    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_->DowngradeMetadataSource(i));

    ASSERT_EQ(absl::OkStatus(),
              metadata_access_object_container_->DowngradeVerification(i));
    // verify the db schema version
    EXPECT_EQ(absl::OkStatus(),
              metadata_access_object_->GetSchemaVersion(&curr_version));
    EXPECT_EQ(curr_version, i);
  }
}

TEST_P(MetadataAccessObjectTest, AutoMigrationTurnedOffByDefault) {
  // Skip upgrade/downgrade migration tests for earlier schema version.
  if (EarlierSchemaEnabled()) { return; }
  // init the database to the current library version.
  ASSERT_EQ(absl::OkStatus(),
            metadata_access_object_->InitMetadataSourceIfNotExists());
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
