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

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include <glog/logging.h>
#include "google/protobuf/any.pb.h"
#include "google/protobuf/field_mask.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/testing/mock.pb.h"
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
    int64_t version,
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
    int64_t version) {
  MetadataSourceQueryConfig::MigrationScheme migration_scheme;
  if (!GetMigrationScheme(version, &migration_scheme).ok()) {
    return false;
  }
  return migration_scheme.has_upgrade_verification();
}

absl::Status QueryConfigMetadataAccessObjectContainer::VerifyDbSchema(
    const int64_t version) {
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
    int64_t version) {
  MetadataSourceQueryConfig::MigrationScheme migration_scheme;
  if (!GetMigrationScheme(version, &migration_scheme).ok()) {
    return false;
  }
  return migration_scheme.has_downgrade_verification();
}

absl::Status
QueryConfigMetadataAccessObjectContainer::SetupPreviousVersionForUpgrade(
    int64_t version) {
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
    int64_t version) {
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
    int64_t version) {
  MetadataSourceQueryConfig::MigrationScheme migration_scheme;
  MLMD_RETURN_IF_ERROR(GetMigrationScheme(version, &migration_scheme));
  return Verification(migration_scheme.downgrade_verification()
                          .post_migration_verification_queries());
}

absl::Status QueryConfigMetadataAccessObjectContainer::UpgradeVerification(
    int64_t version) {
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

int64_t QueryConfigMetadataAccessObjectContainer::MinimumVersion() { return 1; }

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

  int64_t result;
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
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedPointwise;

// A utility method creates and stores a type based on the given text proto.
// Returns stored type proto with id.
template <class NodeType>
NodeType CreateTypeFromTextProto(
    const std::string& type_text_proto,
    MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container) {
  NodeType type = ParseTextProtoOrDie<NodeType>(type_text_proto);
  int64_t type_id;
  CHECK_EQ(metadata_access_object.CreateType(type, &type_id), absl::OkStatus());
  CHECK_EQ(metadata_access_object_container->AddCommitPoint(),
           absl::OkStatus());
  type.set_id(type_id);
  return type;
}

// Utilities that create and store a node with the given text proto.
// Returns stored node proto with id and timestamps.
template <class Node>
void CreateNodeFromTextProto(
    const std::string& node_text_proto, int64_t type_id,
    MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Node& output);

template <>
void CreateNodeFromTextProto(
    const std::string& node_text_proto, int64_t type_id,
    MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Artifact& output) {
  Artifact node = ParseTextProtoOrDie<Artifact>(node_text_proto);
  node.set_type_id(type_id);
  int64_t node_id;
  ASSERT_EQ(metadata_access_object.CreateArtifact(node, &node_id),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_container->AddCommitPoint(),
            absl::OkStatus());
  std::vector<Artifact> nodes;
  ASSERT_EQ(metadata_access_object.FindArtifactsById({node_id}, &nodes),
            absl::OkStatus());
  ASSERT_THAT(nodes, SizeIs(1));
  output = nodes[0];
}

template <>
void CreateNodeFromTextProto(
    const std::string& node_text_proto, int64_t type_id,
    MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Execution& output) {
  Execution node = ParseTextProtoOrDie<Execution>(node_text_proto);
  node.set_type_id(type_id);
  int64_t node_id;
  ASSERT_EQ(metadata_access_object.CreateExecution(node, &node_id),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_container->AddCommitPoint(),
            absl::OkStatus());
  std::vector<Execution> nodes;
  ASSERT_EQ(metadata_access_object.FindExecutionsById({node_id}, &nodes),
            absl::OkStatus());
  ASSERT_THAT(nodes, SizeIs(1));
  output = nodes[0];
}

template <>
void CreateNodeFromTextProto(
    const std::string& node_text_proto, int64_t type_id,
    MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Context& output) {
  Context node = ParseTextProtoOrDie<Context>(node_text_proto);
  node.set_type_id(type_id);
  int64_t node_id;
  ASSERT_EQ(metadata_access_object.CreateContext(node, &node_id),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_container->AddCommitPoint(),
            absl::OkStatus());
  std::vector<Context> nodes;
  ASSERT_EQ(metadata_access_object.FindContextsById({node_id}, &nodes),
            absl::OkStatus());
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
  int64_t dummy_id;
  ASSERT_EQ(metadata_access_object.CreateEvent(output_event, &dummy_id),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_container->AddCommitPoint(),
            absl::OkStatus());
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
  EXPECT_EQ(metadata_access_object.UpdateArtifact(updated_node),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_container->AddCommitPoint(),
            absl::OkStatus());
  std::vector<Artifact> artifacts;
  ASSERT_EQ(
      metadata_access_object.FindArtifactsById({updated_node.id()}, &artifacts),
      absl::OkStatus());
  ASSERT_THAT(artifacts, SizeIs(1));
  output = artifacts.at(0);
}

template <>
void UpdateAndReturnNode(
    const Execution& updated_node, MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Execution& output) {
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(metadata_access_object.UpdateExecution(updated_node),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_container->AddCommitPoint(),
            absl::OkStatus());
  std::vector<Execution> executions;
  ASSERT_EQ(metadata_access_object.FindExecutionsById({updated_node.id()},
                                                      &executions),
            absl::OkStatus());
  ASSERT_THAT(executions, SizeIs(1));
  output = executions.at(0);
}

template <>
void UpdateAndReturnNode(
    const Context& updated_node, MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
    Context& output) {
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(metadata_access_object.UpdateContext(updated_node),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_container->AddCommitPoint(),
            absl::OkStatus());
  std::vector<Context> contexts;
  ASSERT_EQ(
      metadata_access_object.FindContextsById({updated_node.id()}, &contexts),
      absl::OkStatus());
  ASSERT_THAT(contexts, SizeIs(1));
  output = contexts.at(0);
}

// Set up for FindTypesByIds() related tests.
// `type_1` and `type_2` are initilized, inserted into db and returned.
template <class Type>
absl::Status FindTypesByIdsSetup(
    MetadataAccessObject& metadata_access_object,
    MetadataAccessObjectContainer* metadata_access_object_container,
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
  int64_t type_id_1;
  int64_t type_id_2;
  MLMD_RETURN_IF_ERROR(metadata_access_object.CreateType(type_1, &type_id_1));
  MLMD_RETURN_IF_ERROR(metadata_access_object.CreateType(type_2, &type_id_2));
  MLMD_RETURN_IF_ERROR(metadata_access_object_container->AddCommitPoint());
  type_1.set_id(type_id_1);
  type_2.set_id(type_id_2);

  return absl::OkStatus();
}

TEST_P(MetadataAccessObjectTest, ParsePackedMockProto) {
  // Tests that explicitly parsing MockProto and packing it into a protobuf::Any
  // yields the same result as parsing protobuf::Any from a dynamically resolved
  // [MockProto]{...} definition.
  //
  // Note: The presence of this test is necessary to build and test on MacOS.
  // On MacOS, this explicit reference to MockProto prevents the compiler from
  // removing the definition that ParseTextProtoOrDie needs in order to
  // dynamically resolve the MockProto type.
  MockProto mock_proto = ParseTextProtoOrDie<MockProto>(R"pb(
    string_value: '3'
    double_value: 3.0
  )pb");
  google::protobuf::Any any_proto_1;
  any_proto_1.PackFrom(mock_proto);
  google::protobuf::Any any_proto_2 =
      ParseTextProtoOrDie<google::protobuf::Any>(R"pb(
        [type.googleapis.com/ml_metadata.testing.MockProto] {
          string_value: '3'
          double_value: 3.0
        })pb");
  EXPECT_THAT(any_proto_1, EqualsProto(any_proto_2));
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceCheckSchemaVersion) {
  // Skip schema/library version consistency check for earlier schema version.
  if (EarlierSchemaEnabled()) {
    return;
  }
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t schema_version;
  ASSERT_EQ(metadata_access_object_->GetSchemaVersion(&schema_version),
            absl::OkStatus());
  int64_t local_schema_version = metadata_access_object_->GetLibraryVersion();
  EXPECT_EQ(schema_version, local_schema_version);
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceIfNotExists) {
  // Skip empty db init tests for earlier schema version.
  if (EarlierSchemaEnabled()) {
    return;
  }
  // creates the schema and insert some records
  EXPECT_EQ(metadata_access_object_->InitMetadataSourceIfNotExists(),
            absl::OkStatus());
  ASSERT_EQ(metadata_source_->Commit(), absl::OkStatus());
  ASSERT_EQ(metadata_source_->Begin(), absl::OkStatus());
  ArtifactType want_type =
      ParseTextProtoOrDie<ArtifactType>("name: 'test_type'");
  int64_t type_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateType(want_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  // all schema exists, the methods does nothing, check the stored type
  EXPECT_EQ(metadata_access_object_->InitMetadataSourceIfNotExists(),
            absl::OkStatus());
  ArtifactType got_type;
  EXPECT_EQ(metadata_access_object_->FindTypeById(type_id, &got_type),
            absl::OkStatus());
  EXPECT_THAT(want_type, EqualsProto(got_type, /*ignore_fields=*/{"id"}));
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceIfNotExistsErrorAborted) {
  // Skip empty db init tests for earlier schema version.
  if (EarlierSchemaEnabled() || SkipSchemaMigrationTests()) {
    return;
  }
  // creates the schema and insert some records
  ASSERT_EQ(metadata_access_object_->InitMetadataSourceIfNotExists(),
            absl::OkStatus());
  ASSERT_EQ(metadata_source_->Commit(), absl::OkStatus());
  ASSERT_EQ(metadata_source_->Begin(), absl::OkStatus());
  {
    ASSERT_EQ(metadata_access_object_container_->DropTypeTable(),
              absl::OkStatus());
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
  ASSERT_EQ(metadata_access_object_->InitMetadataSourceIfNotExists(),
            absl::OkStatus());
  {
    ASSERT_EQ(metadata_access_object_container_->DropTypeTable(),
              absl::OkStatus());
  }
  EXPECT_EQ(metadata_access_object_->InitMetadataSource(), absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceIfNotExistsErrorAborted2) {
  // Skip partial schema initialization for earlier schema version.
  if (EarlierSchemaEnabled() || SkipSchemaMigrationTests()) {
    return;
  }
  // Drop the artifact table (or artifact property table).
  EXPECT_EQ(Init(), absl::OkStatus());
  {
    // drop a table.
    RecordSet record_set;
    ASSERT_EQ(metadata_access_object_container_->DropArtifactTable(),
              absl::OkStatus());
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
  ASSERT_EQ(metadata_access_object_->InitMetadataSourceIfNotExists(),
            absl::OkStatus());
  {
    // delete the schema version
    ASSERT_EQ(metadata_access_object_container_->DeleteSchemaVersion(),
              absl::OkStatus());
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
  EXPECT_EQ(Init(), absl::OkStatus());
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
  if (!EarlierSchemaEnabled()) {
    return;
  }
  const absl::Status status =
      metadata_access_object_->InitMetadataSourceIfNotExists();
  EXPECT_TRUE(absl::IsFailedPrecondition(status))
      << "Expected FAILED_PRECONDITION but got " << status;
}

TEST_P(MetadataAccessObjectTest,
       EarlierSchemaInitMetadataSourceIfNotExistErrorIncompatibleSchema) {
  if (!EarlierSchemaEnabled()) {
    return;
  }
  // Populates an existing db at an incompatible schema_version.
  ASSERT_EQ(Init(), absl::OkStatus());
  ASSERT_EQ(metadata_access_object_container_->SetDatabaseVersionIncompatible(),
            absl::OkStatus());
  const absl::Status status =
      metadata_access_object_->InitMetadataSourceIfNotExists();
  EXPECT_TRUE(absl::IsFailedPrecondition(status))
      << "Expected FAILED_PRECONDITION but got " << status;
}

TEST_P(MetadataAccessObjectTest, CreateParentTypeInheritanceLink) {
  ASSERT_EQ(Init(), absl::OkStatus());

  {
    // Test: create artifact parent type inheritance link
    const ArtifactType type1 = CreateTypeFromTextProto<ArtifactType>(
        "name: 't1'", *metadata_access_object_,
        metadata_access_object_container_.get());
    const ArtifactType type2 = CreateTypeFromTextProto<ArtifactType>(
        "name: 't2'", *metadata_access_object_,
        metadata_access_object_container_.get());
    // create parent type is ok.
    ASSERT_EQ(
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2),
        absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    // recreate the same parent type returns AlreadyExists
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2);
    EXPECT_TRUE(absl::IsAlreadyExists(status));
  }

  {
    // Test: create execution parent type inheritance link
    const ExecutionType type1 = CreateTypeFromTextProto<ExecutionType>(
        "name: 't1'", *metadata_access_object_,
        metadata_access_object_container_.get());
    const ExecutionType type2 = CreateTypeFromTextProto<ExecutionType>(
        "name: 't2'", *metadata_access_object_,
        metadata_access_object_container_.get());
    // create parent type is ok.
    ASSERT_EQ(
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2),
        absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    // recreate the same parent type returns AlreadyExists
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2);
    EXPECT_TRUE(absl::IsAlreadyExists(status));
  }

  {
    // Test: create context parent type inheritance link
    const ContextType type1 = CreateTypeFromTextProto<ContextType>(
        "name: 't1'", *metadata_access_object_,
        metadata_access_object_container_.get());
    const ContextType type2 = CreateTypeFromTextProto<ContextType>(
        "name: 't2'", *metadata_access_object_,
        metadata_access_object_container_.get());
    // create parent type is ok.
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    // recreate the same parent type returns AlreadyExists
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2);
    EXPECT_TRUE(absl::IsAlreadyExists(status));
  }
}

TEST_P(MetadataAccessObjectTest,
       CreateParentTypeInheritanceLinkInvalidTypeIdError) {
  ASSERT_EQ(Init(), absl::OkStatus());
  const ArtifactType stored_type1 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't1'", *metadata_access_object_,
      metadata_access_object_container_.get());
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
  ASSERT_EQ(Init(), absl::OkStatus());
  const ArtifactType type1 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't1'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ArtifactType type2 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't2'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ArtifactType type3 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't3'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ArtifactType type4 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't4'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ArtifactType type5 = CreateTypeFromTextProto<ArtifactType>(
      "name: 't4'", *metadata_access_object_,
      metadata_access_object_container_.get());

  {
    // cannot add self as parent.
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type1);
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  // type1 -> type2
  ASSERT_EQ(

      metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  {
    // cannot have bi-direction parent
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type2, type1);
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  // type1 -> type2 -> type3
  //      \-> type4 -> type5
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type2, type3));
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type1, type4));
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type4, type5));
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  {
    // cannot have transitive parent
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type3, type1);
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  {
    // cannot have transitive parent
    const absl::Status status =
        metadata_access_object_->CreateParentTypeInheritanceLink(type5, type1);
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }
}

TEST_P(MetadataAccessObjectTest, FindParentTypesByTypeId) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Setup: init the store with the following types and inheritance links
  // ArtifactType:  type1 -> type2
  //                type3 -> type2
  // ExecutionType: type4 -> type5
  // ContextType:   type6 -> type7
  //                type8
  const ArtifactType type1 = CreateTypeFromTextProto<ArtifactType>(
      R"(
          name: 't1'
          properties { key: 'property_1' value: STRING }
      )",
      *metadata_access_object_, metadata_access_object_container_.get());
  ArtifactType type2 = CreateTypeFromTextProto<ArtifactType>(
      R"(
          name: 't2'
          properties { key: 'property_2' value: INT }
      )",
      *metadata_access_object_, metadata_access_object_container_.get());
  ArtifactType type3 = CreateTypeFromTextProto<ArtifactType>(
      R"(
          name: 't3'
          properties { key: 'property_3' value: DOUBLE }
      )",
      *metadata_access_object_, metadata_access_object_container_.get());
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type3, type2));

  const ExecutionType type4 = CreateTypeFromTextProto<ExecutionType>(
      R"(
          name: 't4'
          properties { key: 'property_4' value: STRING }
      )",
      *metadata_access_object_, metadata_access_object_container_.get());
  const ExecutionType type5 = CreateTypeFromTextProto<ExecutionType>(
      R"(
            name: 't5'
        )",
      *metadata_access_object_, metadata_access_object_container_.get());
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type4, type5));

  const ContextType type6 = CreateTypeFromTextProto<ContextType>(
      R"(
          name: 't6'
          properties { key: 'property_5' value: INT }
          properties { key: 'property_6' value: DOUBLE }
      )",
      *metadata_access_object_, metadata_access_object_container_.get());
  const ContextType type7 = CreateTypeFromTextProto<ContextType>(
      "name: 't7'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ContextType type8 = CreateTypeFromTextProto<ContextType>(
      "name: 't8'", *metadata_access_object_,
      metadata_access_object_container_.get());
  ASSERT_EQ(
      absl::OkStatus(),
      metadata_access_object_->CreateParentTypeInheritanceLink(type6, type7));

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // verify artifact types
  {
    absl::flat_hash_map<int64_t, ArtifactType> parent_types;
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id(), type3.id()}, parent_types),
              absl::OkStatus());
    // Type properties will not be retrieved in FindParentTypesByTypeId.
    type2.clear_properties();
    type3.clear_properties();
    ASSERT_EQ(parent_types.size(), 2);
    EXPECT_THAT(parent_types[type1.id()], EqualsProto(type2));
    EXPECT_THAT(parent_types[type3.id()], EqualsProto(type2));
  }

  {
    absl::flat_hash_map<int64_t, ArtifactType> parent_types;
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId({type2.id()},
                                                               parent_types),
              absl::OkStatus());
    EXPECT_THAT(parent_types, IsEmpty());
  }

  // verify execution types
  {
    absl::flat_hash_map<int64_t, ExecutionType> parent_types;
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId({type4.id()},
                                                               parent_types),
              absl::OkStatus());
    EXPECT_THAT(parent_types[type4.id()], EqualsProto(type5));
  }

  {
    absl::flat_hash_map<int64_t, ExecutionType> parent_types;
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId({type5.id()},
                                                               parent_types),
              absl::OkStatus());
    EXPECT_THAT(parent_types, IsEmpty());
  }

  // verify context types
  {
    absl::flat_hash_map<int64_t, ContextType> parent_types;
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId({type6.id()},
                                                               parent_types),
              absl::OkStatus());
    EXPECT_THAT(parent_types[type6.id()], EqualsProto(type7));
  }

  {
    absl::flat_hash_map<int64_t, ContextType> parent_types;
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId({type7.id()},
                                                               parent_types),
              absl::OkStatus());
    EXPECT_THAT(parent_types, IsEmpty());
  }

  {
    absl::flat_hash_map<int64_t, ContextType> parent_types;
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId({type8.id()},
                                                               parent_types),
              absl::OkStatus());
    EXPECT_THAT(parent_types, IsEmpty());
  }

  // verify mixed type ids
  {
    absl::flat_hash_map<int64_t, ArtifactType> parent_types;
    // A mixture of context, exectuion and artifact type ids.
    const auto status = metadata_access_object_->FindParentTypesByTypeId(
        {type1.id(), type4.id(), type6.id()}, parent_types);
    // NOT_FOUND status was returned because `type4` and `type6` are not
    // artifact types and hence will not be found by FindTypesImpl.
    EXPECT_TRUE(absl::IsNotFound(status));
  }
}

TEST_P(MetadataAccessObjectTest, CreateType) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type1 = ParseTextProtoOrDie<ArtifactType>("name: 'test_type'");
  int64_t type1_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateType(type1, &type1_id),
            absl::OkStatus());

  ArtifactType type2 = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type2'
    properties { key: 'property_1' value: STRING })pb");
  int64_t type2_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateType(type2, &type2_id),
            absl::OkStatus());
  EXPECT_NE(type1_id, type2_id);

  ExecutionType type3 = ParseTextProtoOrDie<ExecutionType>(
      R"pb(name: 'test_type'
           properties { key: 'property_2' value: INT }
           input_type: { any: {} }
           output_type: { none: {} }
      )pb");
  int64_t type3_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateType(type3, &type3_id),
            absl::OkStatus());
  EXPECT_NE(type1_id, type3_id);
  EXPECT_NE(type2_id, type3_id);

  ContextType type4 = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: STRING })pb");
  int64_t type4_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateType(type4, &type4_id),
            absl::OkStatus());
  EXPECT_NE(type1_id, type4_id);
  EXPECT_NE(type2_id, type4_id);
  EXPECT_NE(type3_id, type4_id);
}

TEST_P(MetadataAccessObjectTest, StoreTypeWithVersionAndDescriptions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  static char kTypeStr[] = R"(
    name: 'test_type'
    version: 'v1'
    description: 'the type description'
    properties { key: 'stored_property' value: STRING })";

  {
    const ArtifactType want_artifact_type =
        CreateTypeFromTextProto<ArtifactType>(
            kTypeStr, *metadata_access_object_,
            metadata_access_object_container_.get());
    ArtifactType got_artifact_type;
    EXPECT_EQ(metadata_access_object_->FindTypeById(want_artifact_type.id(),
                                                    &got_artifact_type),
              absl::OkStatus());
    EXPECT_THAT(want_artifact_type, EqualsProto(got_artifact_type));
  }

  {
    const ExecutionType want_execution_type =
        CreateTypeFromTextProto<ExecutionType>(
            kTypeStr, *metadata_access_object_,
            metadata_access_object_container_.get());
    ExecutionType got_execution_type;
    EXPECT_EQ(metadata_access_object_->FindTypeByNameAndVersion(
                  want_execution_type.name(), want_execution_type.version(),
                  &got_execution_type),
              absl::OkStatus());
    EXPECT_THAT(want_execution_type, EqualsProto(got_execution_type));
  }

  {
    const ContextType want_context_type = CreateTypeFromTextProto<ContextType>(
        kTypeStr, *metadata_access_object_,
        metadata_access_object_container_.get());
    std::vector<ContextType> got_context_types;
    EXPECT_EQ(metadata_access_object_->FindTypes(&got_context_types),
              absl::OkStatus());
    EXPECT_THAT(got_context_types, SizeIs(1));
    EXPECT_THAT(want_context_type, EqualsProto(got_context_types[0]));
  }
}

TEST_P(MetadataAccessObjectTest, StoreTypeWithEmptyVersion) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // When the input version = empty string, it is treated as unset.
  static constexpr absl::string_view kEmptyStringVersionTypeStr =
      "name: 'test_type' version: ''";

  {
    const ArtifactType want_artifact_type =
        CreateTypeFromTextProto<ArtifactType>(
            kEmptyStringVersionTypeStr.data(), *metadata_access_object_,
            metadata_access_object_container_.get());
    ArtifactType got_artifact_type;
    ASSERT_EQ(metadata_access_object_->FindTypeById(want_artifact_type.id(),
                                                    &got_artifact_type),
              absl::OkStatus());
    EXPECT_FALSE(got_artifact_type.has_version());
    EXPECT_THAT(want_artifact_type,
                EqualsProto(got_artifact_type, /*ignore_fields=*/{"version"}));
  }

  {
    const ExecutionType want_execution_type =
        CreateTypeFromTextProto<ExecutionType>(
            kEmptyStringVersionTypeStr.data(), *metadata_access_object_,
            metadata_access_object_container_.get());
    ExecutionType got_execution_type;
    ASSERT_EQ(metadata_access_object_->FindTypeByNameAndVersion(
                  want_execution_type.name(), want_execution_type.version(),
                  &got_execution_type),
              absl::OkStatus());
    EXPECT_FALSE(got_execution_type.has_version());
    EXPECT_THAT(want_execution_type,
                EqualsProto(got_execution_type, /*ignore_fields=*/{"version"}));
  }

  {
    const ContextType want_context_type = CreateTypeFromTextProto<ContextType>(
        kEmptyStringVersionTypeStr.data(), *metadata_access_object_,
        metadata_access_object_container_.get());
    std::vector<ContextType> got_context_types;
    ASSERT_EQ(metadata_access_object_->FindTypes(&got_context_types),
              absl::OkStatus());
    ASSERT_THAT(got_context_types, SizeIs(1));
    EXPECT_FALSE(got_context_types[0].has_version());
    EXPECT_THAT(want_context_type, EqualsProto(got_context_types[0],
                                               /*ignore_fields=*/{"version"}));
  }
}

TEST_P(MetadataAccessObjectTest, CreateTypeError) {
  ASSERT_EQ(Init(), absl::OkStatus());
  {
    ArtifactType wrong_type;
    int64_t type_id;
    // Types must at least have a name.
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->CreateType(wrong_type, &type_id)));
  }
  {
    ArtifactType wrong_type = ParseTextProtoOrDie<ArtifactType>(R"pb(
      name: 'test_type2'
      properties { key: 'property_1' value: UNKNOWN })pb");
    int64_t type_id;
    // Properties must have type either STRING, DOUBLE, or INT. UNKNOWN
    // is not allowed.
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->CreateType(wrong_type, &type_id)));
  }
}

TEST_P(MetadataAccessObjectTest,
       CreateArtifactTypeWithDuplicatedExternalIdError) {
  if (SkipIfEarlierSchemaLessThan(/*min_schema_version=*/9)) {
    return;
  }
  MLMD_ASSERT_OK(Init());

  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type1'
    external_id: 'artifact_type1'
  )pb");
  int64_t type_id = -1;
  MLMD_EXPECT_OK(metadata_access_object_->CreateType(type, &type_id));

  MLMD_ASSERT_OK(AddCommitPointIfNeeded());

  // Insert the same type again to check the unique constraint.
  absl::Status unique_constraint_violation_status =
      metadata_access_object_->CreateType(type, &type_id);
  EXPECT_EQ(CheckUniqueConstraintAndResetTransaction(
                unique_constraint_violation_status),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest,
       CreateExecutionTypeWithDuplicatedExternalIdError) {
  if (SkipIfEarlierSchemaLessThan(/*min_schema_version=*/9)) {
    return;
  }
  MLMD_ASSERT_OK(Init());

  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type1'
    external_id: 'execution_type1'
  )pb");
  int64_t type_id = -1;
  MLMD_EXPECT_OK(metadata_access_object_->CreateType(type, &type_id));

  MLMD_ASSERT_OK(AddCommitPointIfNeeded());

  // Insert the same type again to check the unique constraint.
  absl::Status unique_constraint_violation_status =
      metadata_access_object_->CreateType(type, &type_id);
  EXPECT_EQ(CheckUniqueConstraintAndResetTransaction(
                unique_constraint_violation_status),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest,
       CreateContextTypeWithDuplicatedExternalIdError) {
  if (SkipIfEarlierSchemaLessThan(/*min_schema_version=*/9)) {
    return;
  }
  MLMD_ASSERT_OK(Init());

  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type1'
    external_id: 'context_type1'
  )pb");
  int64_t type_id = -1;
  MLMD_EXPECT_OK(metadata_access_object_->CreateType(type, &type_id));

  MLMD_ASSERT_OK(AddCommitPointIfNeeded());

  absl::Status unique_constraint_violation_status =
      metadata_access_object_->CreateType(type, &type_id);
  EXPECT_EQ(CheckUniqueConstraintAndResetTransaction(
                unique_constraint_violation_status),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, UpdateType) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type1 = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'type1'
    properties { key: 'stored_property' value: STRING })pb");
  int64_t type1_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateType(type1, &type1_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ExecutionType type2 = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'type2'
    properties { key: 'stored_property' value: STRING })pb");
  int64_t type2_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateType(type2, &type2_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ContextType type3 = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'type3'
    properties { key: 'stored_property' value: STRING })pb");
  int64_t type3_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateType(type3, &type3_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ArtifactType want_type1;
  want_type1.set_id(type1_id);
  want_type1.set_name("type1");
  (*want_type1.mutable_properties())["stored_property"] = STRING;
  (*want_type1.mutable_properties())["new_property"] = INT;
  EXPECT_EQ(metadata_access_object_->UpdateType(want_type1), absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ArtifactType got_type1;
  EXPECT_EQ(metadata_access_object_->FindTypeById(type1_id, &got_type1),
            absl::OkStatus());
  EXPECT_THAT(want_type1, EqualsProto(got_type1));

  // update properties may not include all existing properties
  ExecutionType want_type2;
  want_type2.set_name("type2");
  (*want_type2.mutable_properties())["new_property"] = DOUBLE;
  EXPECT_EQ(metadata_access_object_->UpdateType(want_type2), absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ExecutionType got_type2;
  EXPECT_EQ(metadata_access_object_->FindTypeById(type2_id, &got_type2),
            absl::OkStatus());
  (*want_type2.mutable_properties())["stored_property"] = STRING;
  EXPECT_THAT(want_type2, EqualsProto(got_type2, /*ignore_fields=*/{"id"}));

  // update context type
  ContextType want_type3;
  want_type3.set_name("type3");
  (*want_type3.mutable_properties())["new_property"] = STRING;
  EXPECT_EQ(metadata_access_object_->UpdateType(want_type3), absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  ContextType got_type3;
  EXPECT_EQ(metadata_access_object_->FindTypeById(type3_id, &got_type3),
            absl::OkStatus());
  (*want_type3.mutable_properties())["stored_property"] = STRING;
  EXPECT_THAT(want_type3, EqualsProto(got_type3, /*ignore_fields=*/{"id"}));
}

TEST_P(MetadataAccessObjectTest, UpdateTypeError) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'stored_type'
    properties { key: 'stored_property' value: STRING })pb");
  int64_t type_id;
  EXPECT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  {
    ArtifactType type_without_name;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->UpdateType(type_without_name)));
  }
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  {
    ArtifactType type_with_wrong_id;
    type_with_wrong_id.set_name("stored_type");
    type_with_wrong_id.set_id(type_id + 1);
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->UpdateType(type_with_wrong_id)));
  }
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  {
    ArtifactType type_with_modified_property_type;
    type_with_modified_property_type.set_id(type_id);
    type_with_modified_property_type.set_name("stored_type");
    (*type_with_modified_property_type
          .mutable_properties())["stored_property"] = INT;
    EXPECT_TRUE(absl::IsAlreadyExists(
        metadata_access_object_->UpdateType(type_with_modified_property_type)));
  }
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType want_type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRUCT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ArtifactType got_type;
  EXPECT_EQ(metadata_access_object_->FindTypeById(type_id, &got_type),
            absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType want_type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ContextType got_type;
  EXPECT_EQ(metadata_access_object_->FindTypeById(type_id, &got_type),
            absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType want_type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    input_type: { any: {} }
    output_type: { none: {} }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ExecutionType got_type;
  EXPECT_EQ(metadata_access_object_->FindTypeById(type_id, &got_type),
            absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType want_type;
  want_type.set_name("пример_типа");
  (*want_type.mutable_properties())["привет"] = INT;
  (*want_type.mutable_input_type()
        ->mutable_dict()
        ->mutable_properties())["пример"]
      .mutable_any();
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ExecutionType got_type;
  EXPECT_EQ(metadata_access_object_->FindTypeById(type_id, &got_type),
            absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType want_type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ExecutionType got_type;
  EXPECT_EQ(metadata_access_object_->FindTypeById(type_id, &got_type),
            absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType want_type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    input_type: { any: {} }
    output_type: { none: {} }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ExecutionType got_type;
  EXPECT_EQ(metadata_access_object_->FindTypeByNameAndVersion(
                "test_type", /*version=*/absl::nullopt, &got_type),
            absl::OkStatus());
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

TEST_P(MetadataAccessObjectTest, FindArtifactTypesByExternalIds) {
  if (SkipIfEarlierSchemaLessThan(/*min_schema_version=*/9)) {
    return;
  }
  MLMD_ASSERT_OK(Init());
  ArtifactType want_artifact_type_1;
  ArtifactType want_artifact_type_2;
  MLMD_ASSERT_OK(FindTypesByIdsSetup(
      *metadata_access_object_, metadata_access_object_container_.get(),
      want_artifact_type_1, want_artifact_type_2));

  // Test: update external_id, which also prepares for retrieving by external_id
  EXPECT_TRUE(want_artifact_type_1.external_id().empty());
  EXPECT_TRUE(want_artifact_type_2.external_id().empty());

  want_artifact_type_1.set_external_id("want_artifact_type_1");
  want_artifact_type_2.set_external_id("want_artifact_type_2");
  MLMD_ASSERT_OK(metadata_access_object_->UpdateType(want_artifact_type_1));
  MLMD_ASSERT_OK(metadata_access_object_->UpdateType(want_artifact_type_2));

  MLMD_ASSERT_OK(metadata_access_object_container_.get()->AddCommitPoint());

  // Test 1: artifact types can be retrieved by external_ids.
  std::vector<ArtifactType> got_types;
  std::vector<absl::string_view> external_ids = {
      want_artifact_type_1.external_id(), want_artifact_type_2.external_id()};
  MLMD_ASSERT_OK(metadata_access_object_->FindTypesByExternalIds(
      absl::MakeSpan(external_ids), got_types));
  EXPECT_THAT(got_types,
              UnorderedElementsAre(
                  EqualsProto(want_artifact_type_1, /*ignore_fields=*/{"id"}),
                  EqualsProto(want_artifact_type_2, /*ignore_fields=*/{"id"})));

  // Test 2: will return NOT_FOUND error when finding artifact types by
  // non-existing external_id.
  std::vector<absl::string_view> external_ids_absent = {
      "want_artifact_type_absent"};
  std::vector<ArtifactType> got_artifact_types_absent;
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByExternalIds(
      absl::MakeSpan(external_ids_absent), got_artifact_types_absent)));

  // Test 3: will return whatever found when a part of external_ids is
  // non-existing.
  external_ids_absent.push_back(absl::string_view("want_artifact_type_1"));
  MLMD_EXPECT_OK(metadata_access_object_->FindTypesByExternalIds(
      absl::MakeSpan(external_ids_absent), got_artifact_types_absent));
  EXPECT_THAT(got_artifact_types_absent,
              UnorderedElementsAre(
                  EqualsProto(want_artifact_type_1, /*ignore_fields=*/{"id"})));

  // Test 4: will return INVALID_ARGUMENT error when any of the external_ids is
  // empty.
  std::vector<absl::string_view> external_ids_empty = {""};
  std::vector<ArtifactType> got_artifact_types_from_empty_external_ids;
  EXPECT_TRUE(
      absl::IsInvalidArgument(metadata_access_object_->FindTypesByExternalIds(
          absl::MakeSpan(external_ids_empty),
          got_artifact_types_from_empty_external_ids)));
}

TEST_P(MetadataAccessObjectTest, FindExecutionTypesByExternalIds) {
  if (SkipIfEarlierSchemaLessThan(/*min_schema_version=*/9)) {
    return;
  }
  MLMD_ASSERT_OK(Init());
  ExecutionType want_execution_type_1;
  ExecutionType want_execution_type_2;
  MLMD_ASSERT_OK(FindTypesByIdsSetup(
      *metadata_access_object_, metadata_access_object_container_.get(),
      want_execution_type_1, want_execution_type_2));

  // Test: update external_id, which also prepares for retrieving by external_id
  EXPECT_TRUE(want_execution_type_1.external_id().empty());
  EXPECT_TRUE(want_execution_type_2.external_id().empty());

  want_execution_type_1.set_external_id("want_execution_type_1");
  want_execution_type_2.set_external_id("want_execution_type_2");
  MLMD_ASSERT_OK(metadata_access_object_->UpdateType(want_execution_type_1));
  MLMD_ASSERT_OK(metadata_access_object_->UpdateType(want_execution_type_2));

  MLMD_ASSERT_OK(metadata_access_object_container_.get()->AddCommitPoint());

  // Test 1: execution types can be retrieved by external_ids.
  std::vector<ExecutionType> got_types;
  std::vector<absl::string_view> external_ids = {
      want_execution_type_1.external_id(), want_execution_type_2.external_id()};
  MLMD_ASSERT_OK(metadata_access_object_->FindTypesByExternalIds(
      absl::MakeSpan(external_ids), got_types));
  EXPECT_THAT(
      got_types,
      UnorderedElementsAre(
          EqualsProto(want_execution_type_1, /*ignore_fields=*/{"id"}),
          EqualsProto(want_execution_type_2, /*ignore_fields=*/{"id"})));

  // Test 2: will return NOT_FOUND error when finding execution types by
  // non-existing external_id.
  std::vector<absl::string_view> external_ids_absent = {
      "want_execution_type_absent"};
  std::vector<ExecutionType> got_execution_types_absent;
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByExternalIds(
      absl::MakeSpan(external_ids_absent), got_execution_types_absent)));

  // Test 3: will return whatever found when a part of external_ids is
  // non-existing.
  external_ids_absent.push_back(absl::string_view("want_execution_type_1"));
  MLMD_EXPECT_OK(metadata_access_object_->FindTypesByExternalIds(
      absl::MakeSpan(external_ids_absent), got_execution_types_absent));
  EXPECT_THAT(got_execution_types_absent,
              UnorderedElementsAre(EqualsProto(want_execution_type_1,
                                               /*ignore_fields=*/{"id"})));

  // Test 4: will return INVALID_ARGUMENT error when any of the external_ids is
  // empty.
  std::vector<absl::string_view> external_ids_empty = {""};
  std::vector<ExecutionType> got_execution_types_from_empty_external_ids;
  EXPECT_TRUE(
      absl::IsInvalidArgument(metadata_access_object_->FindTypesByExternalIds(
          absl::MakeSpan(external_ids_empty),
          got_execution_types_from_empty_external_ids)));
}

TEST_P(MetadataAccessObjectTest, FindContextTypesByExternalIds) {
  if (SkipIfEarlierSchemaLessThan(/*min_schema_version=*/9)) {
    return;
  }
  MLMD_ASSERT_OK(Init());
  ContextType want_context_type_1;
  ContextType want_context_type_2;
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_context_type_1, want_context_type_2));

  // Test: update external_id, which also prepares for retrieving by external_id
  EXPECT_TRUE(want_context_type_1.external_id().empty());
  EXPECT_TRUE(want_context_type_2.external_id().empty());

  want_context_type_1.set_external_id("want_context_type_1");
  want_context_type_2.set_external_id("want_context_type_2");
  MLMD_ASSERT_OK(metadata_access_object_->UpdateType(want_context_type_1));
  MLMD_ASSERT_OK(metadata_access_object_->UpdateType(want_context_type_2));

  MLMD_ASSERT_OK(metadata_access_object_container_.get()->AddCommitPoint());

  // Test 1: context types can be retrieved by external_ids.
  std::vector<ContextType> got_types;
  std::vector<absl::string_view> external_ids = {
      want_context_type_1.external_id(), want_context_type_2.external_id()};
  MLMD_ASSERT_OK(metadata_access_object_->FindTypesByExternalIds(
      absl::MakeSpan(external_ids), got_types));
  EXPECT_THAT(got_types,
              UnorderedElementsAre(
                  EqualsProto(want_context_type_1, /*ignore_fields=*/{"id"}),
                  EqualsProto(want_context_type_2, /*ignore_fields=*/{"id"})));

  // Test 2: will return NOT_FOUND error when finding context types by
  // non-existing external_id.
  std::vector<absl::string_view> external_ids_absent = {
      "want_context_type_absent"};
  std::vector<ContextType> got_context_types_absent;
  EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindTypesByExternalIds(
      absl::MakeSpan(external_ids_absent), got_context_types_absent)));

  // Test 3: will return whatever found when a part of external_ids is
  // non-existing.
  external_ids_absent.push_back(absl::string_view("want_context_type_1"));
  MLMD_EXPECT_OK(metadata_access_object_->FindTypesByExternalIds(
      absl::MakeSpan(external_ids_absent), got_context_types_absent));
  EXPECT_THAT(got_context_types_absent,
              UnorderedElementsAre(
                  EqualsProto(want_context_type_1, /*ignore_fields=*/{"id"})));

  // Test 4: will return INVALID_ARGUMENT error when any of the external_ids is
  // empty.
  std::vector<absl::string_view> external_ids_empty = {""};
  std::vector<ContextType> got_context_types_from_empty_external_ids;
  EXPECT_TRUE(
      absl::IsInvalidArgument(metadata_access_object_->FindTypesByExternalIds(
          absl::MakeSpan(external_ids_empty),
          got_context_types_from_empty_external_ids)));
}

TEST_P(MetadataAccessObjectTest, FindTypeIdByNameAndVersion) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType want_type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )pb");
  int64_t v0_type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type, &v0_type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  int64_t v0_got_type_id;
  EXPECT_EQ(metadata_access_object_->FindTypeIdByNameAndVersion(
                "test_type", /*version=*/absl::nullopt,
                TypeKind::EXECUTION_TYPE, &v0_got_type_id),
            absl::OkStatus());
  EXPECT_EQ(v0_got_type_id, v0_type_id);

  want_type.set_version("v1");
  int64_t v1_type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type, &v1_type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  int64_t v1_got_type_id;
  EXPECT_EQ(metadata_access_object_->FindTypeIdByNameAndVersion(
                "test_type", "v1", TypeKind::EXECUTION_TYPE, &v1_got_type_id),
            absl::OkStatus());
  EXPECT_EQ(v1_got_type_id, v1_type_id);

  // The type with this name is an execution type, not an artifact/context type.
  int64_t got_type_id;
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
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));

  std::vector<ArtifactType> got_types;
  MLMD_ASSERT_OK(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, got_types));
  EXPECT_THAT(
      got_types,
      UnorderedElementsAre(EqualsProto(want_type_1, /*ignore_fields=*/{"id"}),
                           EqualsProto(want_type_2, /*ignore_fields=*/{"id"})));
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsArtifactInvalidInput) {
  MLMD_ASSERT_OK(Init());
  ArtifactType want_type_1;
  ArtifactType want_type_2;
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));
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
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));
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
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));

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
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));

  std::vector<ExecutionType> got_types;
  MLMD_ASSERT_OK(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, got_types));
  EXPECT_THAT(
      got_types,
      UnorderedElementsAre(EqualsProto(want_type_1, /*ignore_fields=*/{"id"}),
                           EqualsProto(want_type_2, /*ignore_fields=*/{"id"})));
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsExecutionInvalidInput) {
  MLMD_ASSERT_OK(Init());
  ExecutionType want_type_1;
  ExecutionType want_type_2;
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));
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
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));
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
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));

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
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));

  std::vector<ContextType> got_types;
  MLMD_ASSERT_OK(metadata_access_object_->FindTypesByIds(
      {want_type_1.id(), want_type_2.id()}, got_types));
  EXPECT_THAT(
      got_types,
      UnorderedElementsAre(EqualsProto(want_type_1, /*ignore_fields=*/{"id"}),
                           EqualsProto(want_type_2, /*ignore_fields=*/{"id"})));
}

TEST_P(MetadataAccessObjectTest, FindTypesByIdsContextInvalidInput) {
  MLMD_ASSERT_OK(Init());
  ContextType want_type_1;
  ContextType want_type_2;
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));
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
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));
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
  MLMD_ASSERT_OK(FindTypesByIdsSetup(*metadata_access_object_,
                                     metadata_access_object_container_.get(),
                                     want_type_1, want_type_2));

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
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType want_type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  want_type.set_id(type_id);

  ExecutionType got_type;
  EXPECT_EQ(metadata_access_object_->FindTypeByNameAndVersion(
                "test_type", /*version=*/absl::nullopt, &got_type),
            absl::OkStatus());
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

TEST_P(MetadataAccessObjectTest, FindArtifactTypesByNamesAndVersions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType want_type_1 = CreateTypeFromTextProto<ArtifactType>(
      R"(
    name: 'artifact_type_1'
  )",
      *metadata_access_object_, metadata_access_object_container_.get());
  ArtifactType want_type_2 = CreateTypeFromTextProto<ArtifactType>(
      R"(
    name: 'artifact_type_2'
    version: 'test_version'
  )",
      *metadata_access_object_, metadata_access_object_container_.get());
  // Test 1: artifact types can be retrieved by names_and_versions.
  std::vector<std::pair<std::string, std::string>> names_and_versions = {
      {"artifact_type_1", ""}, {"artifact_type_2", "test_version"}};
  std::vector<ArtifactType> got_types;
  ASSERT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_types),
            absl::OkStatus());
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2)));

  // Test 2: The types in the names and versions are artifact types, not
  // execution/context types.
  std::vector<ExecutionType> got_execution_types;
  ASSERT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_execution_types),
            absl::OkStatus());
  EXPECT_THAT(got_execution_types, IsEmpty());
  std::vector<ContextType> got_context_types;
  ASSERT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_context_types),
            absl::OkStatus());
  EXPECT_THAT(got_context_types, IsEmpty());

  // Test 3: return whatever found when a part of names and versions is
  // non-existing.
  names_and_versions.emplace_back("artifact_type_absent", "");
  std::vector<ArtifactType> got_types_absent;
  ASSERT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_types_absent),
            absl::OkStatus());
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2)));

  // Test 4: return INVALID_ARGUMENT error if `a/e/c_types` is not empty or
  // `names_and_versions` is empty.
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->FindTypesByNamesAndVersions(
          absl::MakeSpan(names_and_versions), got_types)));

  names_and_versions.clear();
  got_types.clear();
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->FindTypesByNamesAndVersions({}, got_types)));
}

TEST_P(MetadataAccessObjectTest, FindExecutionTypesByNamesAndVersions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType want_type_1 = CreateTypeFromTextProto<ExecutionType>(
      R"(
    name: 'execution_type_1'
  )",
      *metadata_access_object_, metadata_access_object_container_.get());
  ExecutionType want_type_2 = CreateTypeFromTextProto<ExecutionType>(
      R"(
    name: 'execution_type_2'
    version: 'test_version'
  )",
      *metadata_access_object_, metadata_access_object_container_.get());

  // Test 1: artifact types can be retrieved by names_and_versions.
  std::vector<std::pair<std::string, std::string>> names_and_versions = {
      {"execution_type_1", ""}, {"execution_type_2", "test_version"}};
  std::vector<ExecutionType> got_types;
  ASSERT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_types),
            absl::OkStatus());
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2)));

  // Test 2: The types in the names and versions are execution types, not
  // artifact/context types.
  std::vector<ArtifactType> got_execution_types;
  ASSERT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_execution_types),
            absl::OkStatus());
  EXPECT_THAT(got_execution_types, IsEmpty());
  std::vector<ContextType> got_context_types;
  EXPECT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_context_types),
            absl::OkStatus());
  EXPECT_THAT(got_context_types, IsEmpty());

  // Test 3: return whatever found when a part of names and versions is
  // non-existing.
  names_and_versions.emplace_back("execution_type_absent", "");
  std::vector<ArtifactType> got_types_absent;
  ASSERT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_types_absent),
            absl::OkStatus());
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2)));

  // Test 4: return INVALID_ARGUMENT error if `a/e/c_types` is not empty or
  // `names_and_versions` is empty.
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->FindTypesByNamesAndVersions(
          absl::MakeSpan(names_and_versions), got_types)));

  names_and_versions.clear();
  got_types.clear();
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->FindTypesByNamesAndVersions({}, got_types)));
}

TEST_P(MetadataAccessObjectTest, FindContextTypesByNamesAndVersions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType want_type_1 = CreateTypeFromTextProto<ContextType>(
      R"(
    name: 'context_type_1'
  )",
      *metadata_access_object_, metadata_access_object_container_.get());
  ContextType want_type_2 = CreateTypeFromTextProto<ContextType>(
      R"(
    name: 'context_type_2'
    version: 'test_version'
  )",
      *metadata_access_object_, metadata_access_object_container_.get());

  // Test 1: artifact types can be retrieved by names_and_versions.
  std::vector<std::pair<std::string, std::string>> names_and_versions = {
      {"context_type_1", ""}, {"context_type_2", "test_version"}};
  std::vector<ContextType> got_types;
  ASSERT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_types),
            absl::OkStatus());
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2)));

  // Test 2: The types in the names and versions are context types, not
  // artifact/execution types.
  std::vector<ArtifactType> got_execution_types;
  ASSERT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_execution_types),
            absl::OkStatus());
  EXPECT_THAT(got_execution_types, IsEmpty());
  std::vector<ExecutionType> got_context_types;
  ASSERT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_context_types),
            absl::OkStatus());
  EXPECT_THAT(got_context_types, IsEmpty());

  // Test 3: return whatever found when a part of names and versions is
  // non-existing.
  names_and_versions.emplace_back("context_type_absent", "");
  std::vector<ContextType> got_types_absent;
  ASSERT_EQ(metadata_access_object_->FindTypesByNamesAndVersions(
                absl::MakeSpan(names_and_versions), got_types_absent),
            absl::OkStatus());
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2)));

  // Test 4: return INVALID_ARGUMENT error if `a/e/c_types` is not empty or
  // `names_and_versions` is empty.
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->FindTypesByNamesAndVersions(
          absl::MakeSpan(names_and_versions), got_types)));

  names_and_versions.clear();
  got_types.clear();
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->FindTypesByNamesAndVersions({}, got_types)));
}

TEST_P(MetadataAccessObjectTest, FindAllArtifactTypes) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType want_type_1 = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type_1'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type_1, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  want_type_1.set_id(type_id);

  ArtifactType want_type_2 = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type_2'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_5' value: STRING }
  )pb");
  ASSERT_EQ(metadata_access_object_->CreateType(want_type_2, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  want_type_2.set_id(type_id);

  // No properties.
  ArtifactType want_type_3 = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'no_properties_type'
  )pb");
  ASSERT_EQ(metadata_access_object_->CreateType(want_type_3, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  want_type_3.set_id(type_id);

  std::vector<ArtifactType> got_types;
  EXPECT_EQ(metadata_access_object_->FindTypes(&got_types), absl::OkStatus());
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2),
                                              EqualsProto(want_type_3)));
}

TEST_P(MetadataAccessObjectTest, FindAllExecutionTypes) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType want_type_1 = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type_1'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type_1, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  want_type_1.set_id(type_id);

  ExecutionType want_type_2 = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type_2'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_5' value: STRING }
  )pb");
  ASSERT_EQ(metadata_access_object_->CreateType(want_type_2, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  want_type_2.set_id(type_id);

  // No properties.
  ExecutionType want_type_3 = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'no_properties_type'
  )pb");
  ASSERT_EQ(metadata_access_object_->CreateType(want_type_3, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  want_type_3.set_id(type_id);

  std::vector<ExecutionType> got_types;
  EXPECT_EQ(metadata_access_object_->FindTypes(&got_types), absl::OkStatus());
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2),
                                              EqualsProto(want_type_3)));
}

TEST_P(MetadataAccessObjectTest, FindAllContextTypes) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType want_type_1 = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type_1'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(want_type_1, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  want_type_1.set_id(type_id);

  ContextType want_type_2 = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type_2'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_5' value: STRING }
  )pb");
  ASSERT_EQ(metadata_access_object_->CreateType(want_type_2, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  want_type_2.set_id(type_id);

  // No properties.
  ContextType want_type_3 = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'no_properties_type'
  )pb");
  ASSERT_EQ(metadata_access_object_->CreateType(want_type_3, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  want_type_3.set_id(type_id);

  std::vector<ContextType> got_types;
  EXPECT_EQ(metadata_access_object_->FindTypes(&got_types), absl::OkStatus());
  EXPECT_THAT(got_types, UnorderedElementsAre(EqualsProto(want_type_1),
                                              EqualsProto(want_type_2),
                                              EqualsProto(want_type_3)));
}

TEST_P(MetadataAccessObjectTest, CreateArtifact) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(
      absl::StrCat(R"(
    name: 'test_type_with_predefined_property'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRUCT }
  )",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties { key: 'property_5' value: PROTO }
                     properties { key: 'property_6' value: BOOLEAN }
                                        )pb",
                   ""));
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  Artifact artifact = ParseTextProtoOrDie<Artifact>(
      absl::StrCat(R"(
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
  )",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties {
                       key: 'property_5'
                       value {
                         proto_value {
                           [type.googleapis.com/ml_metadata.testing.MockProto] {
                             string_value: '3'
                             double_value: 3.0
                           }
                         }
                       }
                     }
                     properties {
                       key: 'property_6'
                       value { bool_value: true }
                     }
                                        )pb",
                   ""));
  artifact.set_type_id(type_id);
  int64_t artifact1_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact1_id),
            absl::OkStatus());
  int64_t artifact2_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact2_id),
            absl::OkStatus());
  EXPECT_NE(artifact1_id, artifact2_id);
}

TEST_P(MetadataAccessObjectTest, CreateArtifactWithCustomProperty) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t type_id = InsertType<ArtifactType>("test_type_with_custom_property");
  Artifact artifact = ParseTextProtoOrDie<Artifact>(
      absl::StrCat(R"(
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
  )",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"(
    custom_properties {
      key: 'custom_property_5'
      value {
        proto_value {
          [type.googleapis.com/ml_metadata.testing.MockProto] {
            string_value: '3'
            double_value: 3.0
          }
        }
      }
    }
    custom_properties {
      key: 'custom_property_6'
      value { bool_value: true }
    }
  )",
                   ""));
  artifact.set_type_id(type_id);

  int64_t artifact1_id, artifact2_id;
  EXPECT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact1_id),
            absl::OkStatus());
  EXPECT_EQ(artifact1_id, 1);
  EXPECT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact2_id),
            absl::OkStatus());
  EXPECT_EQ(artifact2_id, 2);
}

TEST_P(MetadataAccessObjectTest, CreateArtifactError) {
  ASSERT_EQ(Init(), absl::OkStatus());

  // unknown type specified
  Artifact artifact;
  int64_t artifact_id;
  absl::Status s =
      metadata_access_object_->CreateArtifact(artifact, &artifact_id);
  EXPECT_TRUE(absl::IsInvalidArgument(s));

  artifact.set_type_id(1);
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->CreateArtifact(artifact, &artifact_id)));
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type_disallow_custom_property'
    properties { key: 'property_1' value: INT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // type mismatch
  Artifact artifact3;
  artifact3.set_type_id(type_id);
  (*artifact3.mutable_properties())["property_1"].set_string_value("3");
  int64_t artifact3_id;
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->CreateArtifact(artifact3, &artifact3_id)));
}

TEST_P(MetadataAccessObjectTest, CreateArtifactWithDuplicatedNameError) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type;
  type.set_name("test_type");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact artifact;
  artifact.set_type_id(type_id);
  artifact.set_name("test artifact name");
  int64_t artifact_id;
  EXPECT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());
  // insert the same artifact again to check the unique constraint
  absl::Status unique_constraint_violation_status =
      metadata_access_object_->CreateArtifact(artifact, &artifact_id);
  EXPECT_EQ(CheckUniqueConstraintAndResetTransaction(
                unique_constraint_violation_status),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateArtifactWithDuplicatedExternalIdError) {
  if (SkipIfEarlierSchemaLessThan(/*min_schema_version=*/9)) {
    return;
  }
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type;
  type.set_name("test_type");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact artifact;
  artifact.set_type_id(type_id);
  artifact.set_external_id("artifact_1");
  int64_t artifact_id;
  EXPECT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());

  // Add a commit point here, because a read operation will be performed to
  // check the uniqueness of external_id first when calling CreateArtifact().
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  // Insert the same artifact again to check the unique constraint
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
  int64_t type_id;
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
  int64_t artifact_id;
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
  int64_t artifact_id_with_invalid_type;
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
  int64_t artifact_id_with_unmatched_property;
  EXPECT_EQ(metadata_access_object_->CreateArtifact(
                artifact_with_unmatched_property,
                /*skip_type_and_property_validation=*/true,
                &artifact_id_with_unmatched_property),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateArtifactWithCustomTimestamp) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type_with_predefined_property'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
    properties { key: 'property_4' value: STRUCT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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

  int64_t artifact_id = -1;
  absl::Time create_time = absl::InfinitePast();

  ASSERT_EQ(metadata_access_object_->CreateArtifact(
                artifact, /*skip_type_and_property_validation=*/false,
                create_time, &artifact_id),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact = artifacts.at(0);
  }

  EXPECT_EQ(got_artifact.create_time_since_epoch(),
            absl::ToUnixMillis(create_time));
}

TEST_P(MetadataAccessObjectTest, CreateExecutionWithoutValidation) {
  MLMD_ASSERT_OK(Init());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(
      absl::StrCat(R"pb(
                     name: 'test_type'
                     properties { key: 'property_1' value: INT }
                     properties { key: 'property_2' value: STRING }
                   )pb",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties { key: 'property_3' value: PROTO }
                     properties { key: 'property_4' value: BOOLEAN }
                                        )pb",
                   ""));
  int64_t type_id;
  MLMD_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  // Inserts execution without validation since the type are known to exist and
  // the execution's properties are matched with its type.
  Execution execution = ParseTextProtoOrDie<Execution>(
      absl::StrCat(R"pb(
                     properties {
                       key: 'property_1'
                       value: { int_value: 3 }
                     }
                     properties {
                       key: 'property_2'
                       value: { string_value: '3' }
                     }
                   )pb",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties {
                       key: 'property_3'
                       value {
                         proto_value {
                           [type.googleapis.com/ml_metadata.testing.MockProto] {
                             string_value: '3'
                             double_value: 3.0
                           }
                         }
                       }
                     }
                     properties {
                       key: 'property_4'
                       value { bool_value: true }
                     }
                                        )pb",
                   ""));
  execution.set_type_id(type_id);
  int64_t execution_id;
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
  int64_t execution_id_with_invalid_type;
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
  int64_t execution_id_with_unmatched_property;
  EXPECT_EQ(metadata_access_object_->CreateExecution(
                execution_with_unmatched_property,
                /*skip_type_and_property_validation=*/true,
                &execution_id_with_unmatched_property),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateExecutionWithCustomTimestamp) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64_t type_id;
  MLMD_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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

  int64_t execution_id = -1;
  absl::Time create_time = absl::InfinitePast();

  ASSERT_EQ(metadata_access_object_->CreateExecution(
                execution, /*skip_type_and_property_validation=*/false,
                create_time, &execution_id),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution;
  {
    std::vector<Execution> executions;
    ASSERT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
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
  int64_t type_id;
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
  int64_t context_id;
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
  int64_t context_id_with_invalid_type;
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
  int64_t context_id_with_unmatched_property;
  EXPECT_EQ(metadata_access_object_->CreateContext(
                context_with_unmatched_property,
                /*skip_type_and_property_validation=*/true,
                &context_id_with_unmatched_property),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateContextWithCustomTimestamp) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64_t type_id;
  MLMD_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
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

  int64_t context_id = -1;
  absl::Time create_time = absl::InfinitePast();

  ASSERT_EQ(metadata_access_object_->CreateContext(
                context, /*skip_type_and_property_validation=*/false,
                create_time, &context_id),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context got_context;
  {
    std::vector<Context> contexts;
    ASSERT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
    got_context = contexts.at(0);
  }

  EXPECT_EQ(got_context.create_time_since_epoch(),
            absl::ToUnixMillis(create_time));
}

TEST_P(MetadataAccessObjectTest, FindArtifactById) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact want_artifact = ParseTextProtoOrDie<Artifact>(R"pb(
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
  want_artifact.set_type_id(type_id);
  want_artifact.set_type("test_type");

  int64_t artifact_id;
  ASSERT_EQ(
      metadata_access_object_->CreateArtifact(want_artifact, &artifact_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
    int64_t artifact1_id;
    ASSERT_EQ(
        metadata_access_object_->CreateArtifact(want_artifact1, &artifact1_id),
        absl::OkStatus());
    want_artifact1.set_id(artifact1_id);
  }

  Artifact want_artifact2 = ParseTextProtoOrDie<Artifact>(
      absl::StrFormat(kArtifactTemplate, type_id, 11, 12.0, "13", "14"));
  {
    int64_t artifact2_id;
    ASSERT_EQ(
        metadata_access_object_->CreateArtifact(want_artifact2, &artifact2_id),
        absl::OkStatus());
    want_artifact2.set_id(artifact2_id);
  }

  ASSERT_NE(want_artifact1.id(), want_artifact2.id());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  want_artifact1.set_type(type.name());
  want_artifact2.set_type(type.name());

  // Test: retrieve by empty ids
  {
    std::vector<Artifact> got_artifacts;
    EXPECT_EQ(metadata_access_object_->FindArtifactsById({}, &got_artifacts),
              absl::OkStatus());
    EXPECT_THAT(got_artifacts, IsEmpty());
  }
  // Test: retrieve by unknown id
  const int64_t unknown_id = want_artifact1.id() + want_artifact2.id() + 1;
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
    EXPECT_EQ(metadata_access_object_->FindArtifactsById({want_artifact1.id()},
                                                         &got_artifacts),
              absl::OkStatus());
    EXPECT_THAT(got_artifacts,
                ElementsAre(EqualsProto(
                    want_artifact1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  {
    std::vector<Artifact> got_artifacts;
    EXPECT_EQ(metadata_access_object_->FindArtifactsById({want_artifact2.id()},
                                                         &got_artifacts),
              absl::OkStatus());
    EXPECT_THAT(got_artifacts,
                ElementsAre(EqualsProto(
                    want_artifact2,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  {
    std::vector<Artifact> got_artifacts;
    EXPECT_EQ(metadata_access_object_->FindArtifactsById(
                  {want_artifact1.id(), want_artifact2.id()}, &got_artifacts),
              absl::OkStatus());
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
    EXPECT_EQ(metadata_access_object_->FindArtifacts(&got_artifacts),
              absl::OkStatus());
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

TEST_P(MetadataAccessObjectTest,
       FindArtifactsByIdReturnsBothArtifactsAndTypes) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  if (!IfSchemaLessThan(9)) {
    type.set_external_id("test_type_external_id");
  }
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  type.set_id(type_id);
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
    int64_t artifact1_id;
    ASSERT_EQ(
        metadata_access_object_->CreateArtifact(want_artifact1, &artifact1_id),
        absl::OkStatus());
    want_artifact1.set_id(artifact1_id);
  }

  Artifact want_artifact2 = ParseTextProtoOrDie<Artifact>(
      absl::StrFormat(kArtifactTemplate, type_id, 11, 12.0, "13", "14"));
  {
    int64_t artifact2_id;
    ASSERT_EQ(
        metadata_access_object_->CreateArtifact(want_artifact2, &artifact2_id),
        absl::OkStatus());
    want_artifact2.set_id(artifact2_id);
  }

  ASSERT_NE(want_artifact1.id(), want_artifact2.id());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  want_artifact1.set_type(type.name());
  want_artifact2.set_type(type.name());

  // Test: retrieve by empty ids
  {
    std::vector<Artifact> got_artifacts;
    std::vector<ArtifactType> got_artifact_types;
    EXPECT_EQ(metadata_access_object_->FindArtifactsById({}, got_artifacts,
                                                         got_artifact_types),
              absl::OkStatus());
    EXPECT_THAT(got_artifacts, IsEmpty());
    EXPECT_THAT(got_artifact_types, IsEmpty());
  }
  // Test: retrieve by unknown id
  const int64_t unknown_id = want_artifact1.id() + want_artifact2.id() + 1;
  {
    std::vector<Artifact> got_artifacts;
    std::vector<ArtifactType> got_artifact_types;
    EXPECT_TRUE(absl::IsNotFound(metadata_access_object_->FindArtifactsById(
        {unknown_id}, got_artifacts, got_artifact_types)));
    EXPECT_THAT(got_artifacts, IsEmpty());
    EXPECT_THAT(got_artifact_types, IsEmpty());
  }
  {
    std::vector<Artifact> got_artifacts;
    std::vector<ArtifactType> got_artifact_types;
    absl::Status status = metadata_access_object_->FindArtifactsById(
        {unknown_id, want_artifact1.id()}, got_artifacts, got_artifact_types);
    EXPECT_TRUE(absl::IsNotFound(status));
    EXPECT_THAT(
        string(status.message()),
        AllOf(HasSubstr(absl::StrCat("Results missing for ids: {", unknown_id,
                                     ",", want_artifact1.id(), "}")),
              HasSubstr(absl::StrCat("Found results for {", want_artifact1.id(),
                                     "}"))));

    EXPECT_THAT(got_artifacts,
                ElementsAre(EqualsProto(
                    want_artifact1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
    EXPECT_THAT(got_artifact_types, ElementsAre(EqualsProto(type)));
  }
  // Test: retrieve by id(s)
  {
    std::vector<Artifact> got_artifacts;
    std::vector<ArtifactType> got_artifact_types;
    EXPECT_EQ(metadata_access_object_->FindArtifactsById(
                  {want_artifact1.id()}, got_artifacts, got_artifact_types),
              absl::OkStatus());
    EXPECT_THAT(got_artifacts,
                ElementsAre(EqualsProto(
                    want_artifact1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
    EXPECT_THAT(got_artifact_types, ElementsAre(EqualsProto(type)));
  }
  {
    std::vector<Artifact> got_artifacts;
    std::vector<ArtifactType> got_artifact_types;
    EXPECT_EQ(metadata_access_object_->FindArtifactsById(
                  {want_artifact2.id()}, got_artifacts, got_artifact_types),
              absl::OkStatus());
    EXPECT_THAT(got_artifacts,
                ElementsAre(EqualsProto(
                    want_artifact2,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
    EXPECT_THAT(got_artifact_types, ElementsAre(EqualsProto(type)));
  }
  {
    std::vector<Artifact> got_artifacts;
    std::vector<ArtifactType> got_artifact_types;
    EXPECT_EQ(metadata_access_object_->FindArtifactsById(
                  {want_artifact1.id(), want_artifact2.id()}, got_artifacts,
                  got_artifact_types),
              absl::OkStatus());
    EXPECT_THAT(
        got_artifacts,
        UnorderedElementsAre(
            EqualsProto(want_artifact1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(want_artifact2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
    EXPECT_THAT(got_artifact_types, ElementsAre(EqualsProto(type)));
  }
}

TEST_P(MetadataAccessObjectTest, ListArtifactsInvalidPageSize) {
  ASSERT_EQ(Init(), absl::OkStatus());
  const ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: -1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )pb");

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
  ASSERT_EQ(
      metadata_access_object.ListArtifacts(options, &nodes, &next_page_token),
      absl::OkStatus());
}

template <>
void ListNode(MetadataAccessObject& metadata_access_object,
              const ListOperationOptions& options,
              std::vector<Execution>& nodes, std::string& next_page_token) {
  ASSERT_EQ(
      metadata_access_object.ListExecutions(options, &nodes, &next_page_token),
      absl::OkStatus());
}

template <>
void ListNode(MetadataAccessObject& metadata_access_object,
              const ListOperationOptions& options, std::vector<Context>& nodes,
              std::string& next_page_token) {
  ASSERT_EQ(
      metadata_access_object.ListContexts(options, &nodes, &next_page_token),
      absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());
  const ArtifactType type = CreateTypeFromTextProto<ArtifactType>(
      "name: 't1'", *metadata_access_object_,
      metadata_access_object_container_.get());
  std::vector<Artifact> want_artifacts(3);
  for (int i = 0; i < 3; i++) {
    absl::SleepFor(absl::Milliseconds(1));
    std::string base_text_proto_string;

    base_text_proto_string = !IfSchemaLessThan(/*schema_version=*/9)
                                 ?
                                 R"( uri: 'uri_$0',
            name: 'artifact_$0',
            external_id: 'artifact_$0')"
                                 :
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
    )",
                       want_artifacts[1].id()),
      *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[2], want_artifacts[0]});

  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri = 'uri_1' and type_id = $0 and \n"
        " create_time_since_epoch = $1"
    )",
                       type.id(), want_artifacts[1].create_time_since_epoch()),
      *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[1]});

  VerifyListOptions<Artifact>(
      R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' OR state = LIVE"
    )",
      *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[2], want_artifacts[1], want_artifacts[0]});

  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type_id = $0 AND state IS NULL"
    )",
                       type.id()),
      *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[2], want_artifacts[1], want_artifacts[0]});

  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type = '$0'"
    )",
                       type.name()),
      *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[2], want_artifacts[1], want_artifacts[0]});

  const int64_t old_update_time =
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
    )",
                                               type.name(), old_update_time),
                              *metadata_access_object_,
                              /*want_nodes=*/{updated_artifact});

  VerifyListOptions<Artifact>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type = '$0' \n"
        " AND state = LIVE"
    )",
                                               type.name()),
                              *metadata_access_object_,
                              /*want_nodes=*/{updated_artifact});

  VerifyListOptions<Artifact>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type = '$0' \n"
        " AND (state = LIVE OR state = DELETED) "
    )",
                                               type.name()),
                              *metadata_access_object_,
                              /*want_nodes=*/{updated_artifact});

  VerifyListOptions<Artifact>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type = '$0' and state != LIVE"
    )",
                       type.name()),
      *metadata_access_object_,
      /*want_nodes=*/{want_artifacts[1], want_artifacts[0]});

  VerifyListOptions<Artifact>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri LIKE 'uri_%' and type = '$0' and name = 'artifact_0'"
    )",
                                               type.name()),
                              *metadata_access_object_,
                              /*want_nodes=*/{want_artifacts[0]});
  if (!IfSchemaLessThan(/*schema_version=*/9)) {
    VerifyListOptions<Artifact>(
        absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "external_id = '$0'"
    )",
                         want_artifacts[0].external_id()),
        *metadata_access_object_,
        /*want_nodes=*/{want_artifacts[0]});
  }
}

TEST_P(MetadataAccessObjectTest, ListNodesFilterEventQuery) {
  ASSERT_EQ(Init(), absl::OkStatus());
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'at1'", *metadata_access_object_,
      metadata_access_object_container_.get());
  std::vector<Artifact> want_artifacts(3);
  for (int i = 0; i < 3; i++) {
    absl::SleepFor(absl::Milliseconds(1));
    CreateNodeFromTextProto(
        absl::Substitute("uri: 'uri_$0' name: 'artifact_$0'", i),
        artifact_type.id(), *metadata_access_object_,
        metadata_access_object_container_.get(), want_artifacts[i]);
  }

  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'et1'", *metadata_access_object_,
      metadata_access_object_container_.get());
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
    )",
                              *metadata_access_object_,
                              /*want_nodes=*/{want_artifacts[0]});

  VerifyListOptions<Artifact>(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: true }
      filter_query: "events_0.type = INPUT OR events_0.type = OUTPUT"
    )",
                              *metadata_access_object_,
                              /*want_nodes=*/want_artifacts);

  VerifyListOptions<Artifact>(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri = 'uri_0' AND events_0.type = INPUT"
    )",
                              *metadata_access_object_,
                              /*want_nodes=*/{want_artifacts[0]});

  VerifyListOptions<Artifact>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "uri = 'uri_0' AND events_0.execution_id = $0"
    )",
                                               want_execution.id()),
                              *metadata_access_object_,
                              /*want_nodes=*/{want_artifacts[0]});

  VerifyListOptions<Artifact>(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query:
        "events_0.type = OUTPUT AND events_0.milliseconds_since_epoch = 1"
    )",
                              *metadata_access_object_,
                              /*want_nodes=*/{want_artifacts[2]});

  // Filter Executions based on Events
  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "events_0.artifact_id = $0"
    )",
                                                want_artifacts[0].id()),
                               *metadata_access_object_,
                               /*want_nodes=*/{want_execution});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "events_0.artifact_id = $0 AND events_0.type = OUTPUT"
    )",
                                                want_artifacts[0].id()),
                               *metadata_access_object_, /*want_nodes=*/{});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "events_0.artifact_id = $0 AND events_0.type = OUTPUT"
    )",
                                                want_artifacts[1].id()),
                               *metadata_access_object_,
                               /*want_nodes=*/{want_execution});
}

TEST_P(MetadataAccessObjectTest, ListExecutionsFilterAttributeQuery) {
  ASSERT_EQ(Init(), absl::OkStatus());
  const ExecutionType type = CreateTypeFromTextProto<ExecutionType>(
      "name: 't1'", *metadata_access_object_,
      metadata_access_object_container_.get());
  std::vector<Execution> want_executions(3);
  for (int i = 0; i < 3; i++) {
    absl::SleepFor(absl::Milliseconds(1));
    std::string base_text_proto_string;

    base_text_proto_string = !IfSchemaLessThan(/*schema_version=*/9) ?
                                                                     R"(
          name: 'execution_$0', external_id: 'execution_$0')"
                                                                     :
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
    )",
                                                want_executions[2].id()),
                               *metadata_access_object_,
                               /*want_nodes=*/{want_executions[2]});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "id != $0 OR last_known_state = COMPLETE"
    )",
                       want_executions[2].id()),
      *metadata_access_object_,
      /*want_nodes=*/{want_executions[1], want_executions[0]});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type_id = $0 AND last_known_state != COMPLETE"
    )",
                       type.id()),
      *metadata_access_object_,
      /*want_nodes=*/
      {want_executions[2], want_executions[1], want_executions[0]});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type_id = $0 AND create_time_since_epoch = $1"
    )",
                       type.id(), want_executions[2].create_time_since_epoch()),
      *metadata_access_object_,
      /*want_nodes=*/
      {want_executions[2]});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1"
    )",
                       type.name(), type.id()),
      *metadata_access_object_,
      /*want_nodes=*/
      {want_executions[2], want_executions[1], want_executions[0]});

  const int64_t old_update_time =
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
    )",
                       type.name(), type.id(), old_update_time),
      *metadata_access_object_,
      /*want_nodes=*/{updated_execution});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 AND \n"
        " last_known_state = COMPLETE"
    )",
                                                type.name(), type.id()),
                               *metadata_access_object_,
                               /*want_nodes=*/{updated_execution});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 AND \n"
        " (last_known_state = COMPLETE OR last_known_state = NEW) "
    )",
                                                type.name(), type.id()),
                               *metadata_access_object_,
                               /*want_nodes=*/{updated_execution});

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 AND \n"
        " last_known_state != COMPLETE"
    )",
                       type.name(), type.id()),
      *metadata_access_object_,
      /*want_nodes=*/{want_executions[1], want_executions[0]});

  VerifyListOptions<Execution>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 AND \n"
        " name = 'execution_0'"
    )",
                                                type.name(), type.id()),
                               *metadata_access_object_,
                               /*want_nodes=*/{want_executions[0]});
  if (!IfSchemaLessThan(/*schema_version=*/9)) {
    VerifyListOptions<Execution>(
        absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "external_id = '$0'"
    )",
                         want_executions[0].external_id()),
        *metadata_access_object_,
        /*want_nodes=*/{want_executions[0]});
  }
}

TEST_P(MetadataAccessObjectTest, ListContextsFilterAttributeQuery) {
  ASSERT_EQ(Init(), absl::OkStatus());
  const ContextType type = CreateTypeFromTextProto<ContextType>(
      R"(
      name: 't1'
      properties { key: 'p1' value: INT })",
      *metadata_access_object_, metadata_access_object_container_.get());
  std::vector<Context> want_contexts(3);
  for (int i = 0; i < 3; i++) {
    absl::SleepFor(absl::Milliseconds(1));
    std::string base_text_proto_string;

    base_text_proto_string = !IfSchemaLessThan(/*schema_version=*/9) ?
                                                                     R"(
          name: 'c$0', external_id: 'c$0')"
                                                                     :
                                                                     R"(
          name: 'c$0')";
    CreateNodeFromTextProto(absl::Substitute(base_text_proto_string, i),
                            type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_contexts[i]);
  }

  VerifyListOptions<Context>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "id = $0 "
    )",
                                              want_contexts[0].id()),
                             *metadata_access_object_,
                             /*want_nodes=*/{want_contexts[0]});

  VerifyListOptions<Context>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type_id = $0 AND name IS NOT NULL"
    )",
                       type.id()),
      *metadata_access_object_,
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
    )",
                       type.name(), type.id()),
      *metadata_access_object_,
      /*want_nodes=*/
      {want_contexts[2], want_contexts[1], want_contexts[0]});

  VerifyListOptions<Context>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id != $1"
    )",
                                              type.name(), type.id()),
                             *metadata_access_object_,
                             /*want_nodes=*/{});

  VerifyListOptions<Context>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: '(type = \'$0\' AND name = \'$1\') OR '
                    '(type = \'$2\' AND name = \'$3\')'
    )",
                       type.name(), want_contexts[2].name(), type.name(),
                       want_contexts[1].name()),
      *metadata_access_object_,
      /*want_nodes=*/{want_contexts[2], want_contexts[1]});

  Context old_context = want_contexts[2];
  const int64_t old_update_time =
      want_contexts[2].last_update_time_since_epoch();
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
    )",
                       type.name(), type.id(), old_update_time),
      *metadata_access_object_,
      /*want_nodes=*/{updated_context});

  VerifyListOptions<Context>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "type = '$0' AND type_id = $1 \n"
      " AND name = 'c0'"
    )",
                                              type.name(), type.id()),
                             *metadata_access_object_,
                             /*want_nodes=*/{want_contexts[0]});
  if (!IfSchemaLessThan(/*schema_version=*/9)) {
    VerifyListOptions<Context>(absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "external_id = '$0' "
    )",
                                                want_contexts[0].external_id()),
                               *metadata_access_object_,
                               /*want_nodes=*/{want_contexts[0]});
  }
}

TEST_P(MetadataAccessObjectTest, ListNodesFilterContextNeighborQuery) {
  ASSERT_EQ(Init(), absl::OkStatus());
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ContextType context_type = CreateTypeFromTextProto<ContextType>(
      "name: 'context_type'", *metadata_access_object_,
      metadata_access_object_container_.get());

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
  int64_t attid;
  ASSERT_EQ(metadata_access_object_->CreateAttribution(attribution, &attid),
            absl::OkStatus());
  Association association;
  association.set_execution_id(want_executions[2].id());
  association.set_context_id(contexts[0].id());
  int64_t assid;
  ASSERT_EQ(metadata_access_object_->CreateAssociation(association, &assid),
            absl::OkStatus());
  for (int i = 0; i < 3; i++) {
    Attribution attribution;
    attribution.set_artifact_id(want_artifacts[0].id());
    attribution.set_context_id(contexts[i].id());
    int64_t attid;
    ASSERT_EQ(metadata_access_object_->CreateAttribution(attribution, &attid),
              absl::OkStatus());
    Association association;
    association.set_execution_id(want_executions[0].id());
    association.set_context_id(contexts[i].id());
    int64_t assid;
    ASSERT_EQ(metadata_access_object_->CreateAssociation(association, &assid),
              absl::OkStatus());
  }
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // TODO(b/251486486) Use R"pb(...)pb" for protos in the entire file.
  VerifyListOptions<Context>(
      absl::Substitute(R"pb(
                         max_result_size: 10,
                         order_by_field: { field: CREATE_TIME is_asc: true }
                         filter_query: "artifacts_c.id = $0")pb",
                       want_artifacts[1].id()),
      *metadata_access_object_,
      /*want_nodes=*/{contexts[0]});

  VerifyListOptions<Context>(
      absl::Substitute(R"pb(
                         max_result_size: 10,
                         order_by_field: { field: CREATE_TIME is_asc: true }
                         filter_query: "artifacts_c.id = $0")pb",
                       want_artifacts[0].id()),
      *metadata_access_object_,
      /*want_nodes=*/{contexts[0], contexts[1], contexts[2]});

  VerifyListOptions<Context>(
      absl::Substitute(
          R"pb(
            max_result_size: 10,
            order_by_field: { field: CREATE_TIME is_asc: true }
            filter_query: "artifacts_c.id = $0 and name = '$1'")pb",
          want_artifacts[0].id(), contexts[0].name()),
      *metadata_access_object_,
      /*want_nodes=*/{contexts[0]});

  VerifyListOptions<Context>(
      absl::Substitute(R"pb(
                         max_result_size: 10,
                         order_by_field: { field: CREATE_TIME is_asc: true }
                         filter_query: "executions_c.id = $0")pb",
                       want_executions[2].id()),
      *metadata_access_object_,
      /*want_nodes=*/{contexts[0]});

  VerifyListOptions<Context>(
      absl::Substitute(R"pb(
                         max_result_size: 10,
                         order_by_field: { field: CREATE_TIME is_asc: true }
                         filter_query: "executions_c.id = $0")pb",
                       want_executions[0].id()),
      *metadata_access_object_,
      /*want_nodes=*/{contexts[0], contexts[1], contexts[2]});

  VerifyListOptions<Context>(
      absl::Substitute(
          R"pb(
            max_result_size: 10,
            order_by_field: { field: CREATE_TIME is_asc: true }
            filter_query: "executions_c.id = $0 and name = '$1'")pb",
          want_executions[0].id(), contexts[0].name()),
      *metadata_access_object_,
      /*want_nodes=*/{contexts[0]});

  VerifyListOptions<Context>(
      absl::Substitute(
          R"pb(
            max_result_size: 10,
            order_by_field: { field: CREATE_TIME is_asc: true }
            filter_query: "executions_c.id = $0 and artifacts_a.id = $1")pb",
          want_executions[0].id(), want_artifacts[1].id()),
      *metadata_access_object_,
      /*want_nodes=*/{contexts[0]});

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
    )",
                       context_type.name()),
      *metadata_access_object_,
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
  int64_t dummy_assid;
  ASSERT_EQ(metadata_access_object_->CreateAssociation(additional_association,
                                                       &dummy_assid),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  VerifyListOptions<Execution>(
      absl::Substitute(R"(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: " contexts_a.id = $0 OR contexts_a.id = $1")",
                       contexts[0].id(), contexts[1].id()),
      *metadata_access_object_,
      /*want_nodes=*/
      {want_executions[2], want_executions[1], want_executions[0]});

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
  ASSERT_EQ(Init(), absl::OkStatus());
  const ContextType parent_context_type_1 =
      CreateTypeFromTextProto<ContextType>(
          "name: 'parent_context_type_1'", *metadata_access_object_,
          metadata_access_object_container_.get());
  const ContextType parent_context_type_2 =
      CreateTypeFromTextProto<ContextType>(
          "name: 'parent_context_type_2'", *metadata_access_object_,
          metadata_access_object_container_.get());

  const ContextType child_context_type = CreateTypeFromTextProto<ContextType>(
      "name: 'child_context_type'", *metadata_access_object_,
      metadata_access_object_container_.get());

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
    EXPECT_EQ(metadata_access_object_->CreateParentContext(parent_context),
              absl::OkStatus());
  }

  for (int i = 0; i < 2; i++) {
    ParentContext parent_context;
    parent_context.set_parent_id(parent_context_2.id());
    parent_context.set_child_id(child_contexts[i].id());
    EXPECT_EQ(metadata_access_object_->CreateParentContext(parent_context),
              absl::OkStatus());
  }

  // Query on ParentContext.id
  VerifyListOptions<Context>(
      absl::Substitute(
          R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "parent_contexts_c.id = $0")",
          absl::StrCat(parent_context_1.id())),
      *metadata_access_object_,
      /*want_nodes=*/{child_contexts[2], child_contexts[1], child_contexts[0]});

  // Query on ParentContext.id
  VerifyListOptions<Context>(
      absl::Substitute(
          R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "parent_contexts_c.id = $0")",
          absl::StrCat(parent_context_2.id())),
      *metadata_access_object_,
      /*want_nodes=*/{child_contexts[1], child_contexts[0]});

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

  // Query on ChildContext.id
  VerifyListOptions<Context>(
      absl::Substitute(
      R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "child_contexts_c.id = $0")",
          absl::StrCat(child_contexts[0].id())),
      *metadata_access_object_,
      /*want_nodes=*/{parent_context_2, parent_context_1});

  // Query on ChildContext.id
  VerifyListOptions<Context>(
      absl::Substitute(
      R"(max_result_size: 10,
          order_by_field: { field: CREATE_TIME is_asc: false }
          filter_query: "child_contexts_c.id = $0")",
          absl::StrCat(child_contexts[2].id())),
      *metadata_access_object_,
      /*want_nodes=*/{parent_context_1});

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
  ASSERT_EQ(Init(), absl::OkStatus());
  const ContextType parent_context_type_1 =
      CreateTypeFromTextProto<ContextType>(
          "name: 'parent_context_type_1'", *metadata_access_object_,
          metadata_access_object_container_.get());

  const ContextType child_context_type = CreateTypeFromTextProto<ContextType>(
      "name: 'child_context_type'", *metadata_access_object_,
      metadata_access_object_container_.get());

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
  EXPECT_EQ(
      metadata_access_object_->CreateParentContext(parent_child_context_1),
      absl::OkStatus());
  ParentContext parent_child_context_2;
  parent_child_context_2.set_parent_id(parent_context_1.id());
  parent_child_context_2.set_child_id(child_context_3.id());
  EXPECT_EQ(
      metadata_access_object_->CreateParentContext(parent_child_context_2),
      absl::OkStatus());

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
    MetadataAccessObjectContainer* metadata_access_object_container,
    bool schema_less_than_10) {
  const NodeType type = CreateTypeFromTextProto<NodeType>(
      R"(
    name: 'test_type'
    properties { key: 'p1' value: INT }
    properties { key: 'p2' value: DOUBLE }
    properties { key: 'p3' value: STRING }
    properties { key: 'p4' value: BOOLEAN }
  )",
      metadata_access_object, metadata_access_object_container);

  // Setup: 5 nodes of `test_type`
  // node_$i has a custom property custom_property_$i which is not NULL.
  // node_$i also has p1 = $i, p2 = $i.0, p3 = '$i',
  //   and if schema_version >= 10, has p4 = ((i % 2) == 0)
  std::vector<Node> want_nodes(5);
  for (int i = 0; i < want_nodes.size(); i++) {
    CreateNodeFromTextProto(
        schema_less_than_10
            ? absl::StrFormat(
                  R"pb(
                    type_id: %d
                    name: 'test_%d'
                    properties {
                      key: 'p1'
                      value: { int_value: %d }
                    }
                    properties {
                      key: 'p2'
                      value: { double_value: %f }
                    }
                    properties {
                      key: 'p3'
                      value: { string_value: '%s' }
                    }
                    custom_properties {
                      key: 'custom_property_%d'
                      value: { string_value: 'foo' }
                    }
                    custom_properties {
                      key: 'custom_property %d'
                      value: { double_value: 1.0 }
                    }
                    custom_properties {
                      key: 'custom_property:%d'
                      value: { int_value: 1 }
                    }
                  )pb",
                  /*type_id, name_suffix=*/type.id(), i,
                  /*int_value=*/i,
                  /*double_value=*/1.0 * i,
                  /*string_value=*/absl::StrCat("0", i), i, i, i)
            : absl::StrFormat(
                  R"pb(
                    type_id: %d
                    name: 'test_%d'
                    properties {
                      key: 'p1'
                      value: { int_value: %d }
                    }
                    properties {
                      key: 'p2'
                      value: { double_value: %f }
                    }
                    properties {
                      key: 'p3'
                      value: { string_value: '%s' }
                    }
                    properties {
                      key: 'p4'
                      value: { bool_value: %s }
                    }
                    custom_properties {
                      key: 'custom_property_%d'
                      value: { string_value: 'foo' }
                    }
                    custom_properties {
                      key: 'custom_property %d'
                      value: { double_value: 1.0 }
                    }
                    custom_properties {
                      key: 'custom_property:%d'
                      value: { int_value: 1 }
                    }
                    custom_properties {
                      key: 'custom_property-%d'
                      value: { bool_value: true }
                    }
                  )pb",
                  /*type_id, name_suffix=*/type.id(), i,
                  /*int_value=*/i,
                  /*double_value=*/1.0 * i,
                  /*string_value=*/absl::StrCat("0", i),
                  /*bool_value=*/(i % 2) == 0 ? "true" : "false", i, i, i, i),
        type.id(), metadata_access_object, metadata_access_object_container,
        want_nodes[i]);
  }

  static constexpr absl::string_view kListOption = R"(
            max_result_size: 10,
            order_by_field: { field: CREATE_TIME is_asc: false }
            filter_query: "$0")";

  // test property and custom property queries
  // verify all documented columns can be used.
  VerifyListOptions<Node>(
      absl::Substitute(kListOption, "properties.p1.int_value = 0"),
      metadata_access_object, /*want_nodes=*/{want_nodes[0]});

  VerifyListOptions<Node>(
      absl::Substitute(kListOption, "(properties.p1.int_value + 2) * 1 = 2"),
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

  if (!schema_less_than_10) {
    VerifyListOptions<Node>(
        absl::Substitute(kListOption, "properties.p4.bool_value = true"),
        metadata_access_object,
        /*want_nodes=*/
        {want_nodes[4], want_nodes[2], want_nodes[0]}
    );
  }

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
  ASSERT_EQ(Init(), absl::OkStatus());
  TestFilteringWithListOptionsImpl<ArtifactType, Artifact>(
      *metadata_access_object_, metadata_access_object_container_.get(),
      /*schema_less_than_10=*/IfSchemaLessThan(10));
}

TEST_P(MetadataAccessObjectTest, ListExecutionsFilterPropertyQuery) {
  ASSERT_EQ(Init(), absl::OkStatus());
  TestFilteringWithListOptionsImpl<ExecutionType, Execution>(
      *metadata_access_object_, metadata_access_object_container_.get(),
      /*schema_less_than_10=*/IfSchemaLessThan(10));
}

TEST_P(MetadataAccessObjectTest, LisContextsFilterPropertyQuery) {
  ASSERT_EQ(Init(), absl::OkStatus());
  TestFilteringWithListOptionsImpl<ContextType, Context>(
      *metadata_access_object_, metadata_access_object_container_.get(),
      /*schema_less_than_10=*/IfSchemaLessThan(10));
}

TEST_P(MetadataAccessObjectTest, ListNodesFilterWithErrors) {
  ASSERT_EQ(Init(), absl::OkStatus());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 10,
        order_by_field: { field: CREATE_TIME is_asc: false }
        filter_query: "unknown_field = 'uri_3' and type_id = 1"
      )pb");

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
  // testing filter query type mismatch
  {
    list_options = ParseTextProtoOrDie<ListOperationOptions>(R"pb(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "executions_c.id = 123"
    )pb");
    std::vector<Artifact> got_artifacts;
    const absl::Status status = metadata_access_object_->ListArtifacts(
        list_options, &got_artifacts, &next_page_token);
    EXPECT_TRUE(absl::IsInvalidArgument(status))
        << " status is: " << status.code()
        << " status Error is:" << status.message();
  }

  {
    list_options = ParseTextProtoOrDie<ListOperationOptions>(R"pb(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "artifacts_c.id = 123"
    )pb");
    std::vector<Artifact> got_artifacts;
    const absl::Status status = metadata_access_object_->ListArtifacts(
        list_options, &got_artifacts, &next_page_token);
    EXPECT_TRUE(absl::IsInvalidArgument(status))
        << " status is: " << status.code()
        << " status Error is:" << status.message();
  }
  {
    list_options = ParseTextProtoOrDie<ListOperationOptions>(R"pb(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "executions_c.id = 123"
    )pb");
    std::vector<Execution> got_executions;
    const absl::Status status = metadata_access_object_->ListExecutions(
        list_options, &got_executions, &next_page_token);
    EXPECT_TRUE(absl::IsInvalidArgument(status))
        << " status is: " << status.code()
        << " status Error is:" << status.message();
  }
  {
    list_options = ParseTextProtoOrDie<ListOperationOptions>(R"pb(
      max_result_size: 10,
      order_by_field: { field: CREATE_TIME is_asc: false }
      filter_query: "artifacts_c.id = 123"
    )pb");
    std::vector<Execution> got_executions;
    const absl::Status status = metadata_access_object_->ListExecutions(
        list_options, &got_executions, &next_page_token);
    EXPECT_TRUE(absl::IsInvalidArgument(status))
        << " status is: " << status.code()
        << " status Error is:" << status.message();
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
  ASSERT_EQ(metadata_access_object.FindTypes(&artifact_types),
            absl::OkStatus());
  EXPECT_THAT(subgraph.artifact_types(),
              UnorderedPointwise(EqualsProto<ArtifactType>(), artifact_types));
  std::vector<ExecutionType> execution_types;
  ASSERT_EQ(metadata_access_object.FindTypes(&execution_types),
            absl::OkStatus());
  EXPECT_THAT(
      subgraph.execution_types(),
      UnorderedPointwise(EqualsProto<ExecutionType>(), execution_types));
  std::vector<ContextType> context_types;
  ASSERT_EQ(metadata_access_object.FindTypes(&context_types), absl::OkStatus());
  EXPECT_THAT(subgraph.context_types(),
              UnorderedPointwise(EqualsProto<ContextType>(), context_types));
}

void VerifyLineageGraphSkeleton(
    const LineageGraph& skeleton,
    absl::Span<const int64_t> expected_artifact_ids,
    absl::Span<const int64_t> expected_execution_ids,
    const std::vector<Event>& events) {
  EXPECT_THAT(skeleton.artifacts(),
              UnorderedPointwise(IdEquals(), expected_artifact_ids));
  EXPECT_THAT(skeleton.executions(),
              UnorderedPointwise(IdEquals(), expected_execution_ids));
  EXPECT_THAT(skeleton.events(),
              UnorderedPointwise(EqualsProto<Event>(/*ignore_fields=*/{
                                     "milliseconds_since_epoch",
                                 }),
                                 events));
}

TEST_P(MetadataAccessObjectTest, QueryLineageGraph) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Test setup: use a simple graph with multiple paths between (a1, e2).
  // a1 -> e1 -> a2
  //  \            \
  //   \------------> e2
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
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
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/1,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/absl::nullopt, output_graph),
              absl::OkStatus());
    VerifyLineageGraph(
        output_graph, /*artifacts=*/{want_artifacts[0]}, want_executions,
        /*events=*/{want_events[0], want_events[1]}, *metadata_access_object_);
  }

  {
    // Query a1 with 2 hop. It returns all nodes with no duplicate.
    LineageGraph output_graph;
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/2,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/absl::nullopt, output_graph),
              absl::OkStatus());
    VerifyLineageGraph(output_graph, want_artifacts, want_executions,
                       want_events, *metadata_access_object_);
  }

  {
    // Query a1 with 2 hop and max_nodes of 2. It returns a1 and one of e1 or
    // e2.
    LineageGraph output_graph;
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/2,
                  /*max_nodes=*/2,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/absl::nullopt, output_graph),
              absl::OkStatus());

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
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/2,
                  /*max_nodes=*/3,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/absl::nullopt, output_graph),
              absl::OkStatus());
    VerifyLineageGraph(
        output_graph, /*artifacts=*/{want_artifacts[0]}, want_executions,
        /*events=*/{want_events[0], want_events[1]}, *metadata_access_object_);
  }

  {
    // Query a1 with 2 hop and max_nodes of 4. It returns all nodes with no
    // duplicate.
    LineageGraph output_graph;
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/2,
                  /*max_nodes=*/4,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/absl::nullopt, output_graph),
              absl::OkStatus());
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

TEST_P(MetadataAccessObjectTest,
       QueryLineageGrapWithContextAliasInBoundaryArtifacts) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Test setup: use a simple graph with multiple paths between (a1, e2). (a1,
  // a2) are attributed to c1.
  // c1(a1) -> e1 -> c1(a2)
  //     \               \
  //      \------------> e2
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ContextType context_type = CreateTypeFromTextProto<ContextType>(
      "name: 'context_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  std::vector<Artifact> want_artifacts(2);
  std::vector<Execution> want_executions(2);
  std::vector<Context> want_contexts(1);
  CreateNodeFromTextProto(
      "name: 'context_1'", context_type.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), want_contexts[0]);
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
  Attribution attribution;
  attribution.set_artifact_id(want_artifacts[0].id());
  attribution.set_context_id(want_contexts[0].id());
  int64_t attribution_id;
  // Note using ASSERT_EQ as *_OK is not well supported in OSS
  ASSERT_EQ(
      metadata_access_object_->CreateAttribution(attribution, &attribution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  attribution.set_artifact_id(want_artifacts[1].id());
  ASSERT_EQ(
      metadata_access_object_->CreateAttribution(attribution, &attribution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Query a1 with 2 hop, with artifacts boundary filter query using Context
  // alias. It should return only a1 , e1, e2 and two event nodes connecting
  // them. without the filter it would return all nodes with no duplicate,
  // same as above.
  LineageGraph output_graph;
  const std::string boundary_artifacts =
      absl::Substitute("NOT(contexts_0.id = $0)", want_contexts[0].id());
  ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/2,
                /*max_nodes=*/absl::nullopt,
                /*boundary_artifacts=*/boundary_artifacts,
                /*boundary_executions=*/absl::nullopt, output_graph),
            absl::OkStatus());
  VerifyLineageGraph(output_graph, {want_artifacts[0]}, want_executions,
                     /*events=*/{want_events[0], want_events[1]},
                     *metadata_access_object_);
}

TEST_P(MetadataAccessObjectTest,
       QueryLineageGrapWithEventAliasInBoundaryArtifacts) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Test setup: use a simple graph with multiple paths between (a1, e2). (a1,
  // a2) are attributed to c1.
  // c1(a1) -> e1 -> c1(a2)
  //     \               \
  //      \------------> e2 -> a3
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ContextType context_type = CreateTypeFromTextProto<ContextType>(
      "name: 'context_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  std::vector<Artifact> want_artifacts(3);
  std::vector<Execution> want_executions(2);
  std::vector<Context> want_contexts(1);
  CreateNodeFromTextProto(
      "name: 'context_1'", context_type.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), want_contexts[0]);
  for (int i = 0; i < 3; i++) {
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
  std::vector<Event> want_events(5);
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
  CreateEventFromTextProto("type: OUTPUT", want_artifacts[2],
                           want_executions[1], *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[4]);
  Attribution attribution;
  attribution.set_artifact_id(want_artifacts[0].id());
  attribution.set_context_id(want_contexts[0].id());
  int64_t attribution_id;
  // Note using ASSERT_EQ as *_OK is not well supported in OSS
  ASSERT_EQ(
      metadata_access_object_->CreateAttribution(attribution, &attribution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  attribution.set_artifact_id(want_artifacts[1].id());
  ASSERT_EQ(
      metadata_access_object_->CreateAttribution(attribution, &attribution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Query a1 with 2 hop, with artifacts boundary filter query using Event
  // alias. It will return a1, e1, e2 and four events connecting
  // them. Without the filter it would return all nodes with no duplicate,
  // same as above.
  {
    LineageGraph output_graph;
    const std::string boundary_artifacts = absl::Substitute(
        "NOT(events_0.execution_id = $0)", want_executions[1].id());
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/2,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/boundary_artifacts,
                  /*boundary_executions=*/absl::nullopt, output_graph),
              absl::OkStatus());
    VerifyLineageGraph(
        output_graph, {want_artifacts[0], want_artifacts[1]}, want_executions,
        /*events=*/
        {want_events[0], want_events[1], want_events[2], want_events[3]},
        *metadata_access_object_);
  }

  // Query a1 with 1 hop, with executions boundary filter query on execution ID.
  // It will return a1, e2 and 1 event connecting them; e1 is filtered out.
  {
    LineageGraph output_graph;
    const std::string boundary_executions =
        absl::Substitute("id != $0", want_executions[0].id());
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/1,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/boundary_executions, output_graph),
              absl::OkStatus());
    VerifyLineageGraph(output_graph, {want_artifacts[0]}, {want_executions[1]},
                       {want_events[1]}, *metadata_access_object_);
  }

  // Query a1 with 1 hop, with executions boundary filter query using Event
  // alias to check existance of an edge from the executions to artifact a2.
  // It will return a1, e1, e2 and 2 events connecting them.
  {
    LineageGraph output_graph;
    const std::string boundary_executions =
        absl::Substitute("events_0.artifact_id = $0", want_artifacts[1].id());
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/1,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/boundary_executions, output_graph),
              absl::OkStatus());
    VerifyLineageGraph(output_graph, {want_artifacts[0]},
                       {want_executions[0], want_executions[1]},
                       {want_events[0], want_events[1]},
                       *metadata_access_object_);
  }

  // Query a1 with 1 hop, with executions boundary filter query using Event
  // alias and exeuction ID check. It will return a1, e1, e2 and 2 events
  // connecting them because the OR condition and the given condition on events
  // should return all executions.
  {
    LineageGraph output_graph;
    const std::string boundary_executions =
        absl::Substitute("id != $0 OR events_0.artifact_id = $1",
                         want_executions[0].id(), want_artifacts[1].id());
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[0]}, /*max_num_hops=*/1,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/boundary_executions, output_graph),
              absl::OkStatus());
    VerifyLineageGraph(output_graph, {want_artifacts[0]},
                       {want_executions[0], want_executions[1]},
                       {want_events[0], want_events[1]},
                       *metadata_access_object_);
  }
}

TEST_P(MetadataAccessObjectTest, QueryLineageGraphArtifactsOnly) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Test setup: only set up an artifact type and 2 artifacts.
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  std::vector<Artifact> want_artifacts(2);
  for (int i = 0; i < 2; i++) {
    CreateNodeFromTextProto(absl::Substitute("uri: 'uri_$0'", i),
                            artifact_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_artifacts[i]);
  }

  LineageGraph output_graph;
  ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                want_artifacts, /*max_num_hops=*/1, /*max_nodes=*/absl::nullopt,
                /*boundary_artifacts=*/absl::nullopt,
                /*boundary_executions=*/absl::nullopt, output_graph),
            absl::OkStatus());
  VerifyLineageGraph(output_graph, want_artifacts, /*executions=*/{},
                     /*events=*/{}, *metadata_access_object_);
}

TEST_P(MetadataAccessObjectTest, QueryLineageGraphWithBoundaryConditions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Test setup: use a high fan-out graph to test the boundaries cases
  // a0 -> e1 -> a1 -> e0
  //   \-> e2
  //   \-> ...
  //   \-> e250
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
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
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/2,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/"name != 'e0'", output_graph),
              absl::OkStatus());
    VerifyLineageGraph(output_graph, want_artifacts,
                       /*executions=*/{want_executions[1]},
                       /*events=*/{a1e1, a0e1}, *metadata_access_object_);
  }

  {
    // boundary execution condition with max num hops and max_nodes of 3.
    LineageGraph output_graph;
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/2,
                  /*max_nodes=*/3,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/"name != 'e0'", output_graph),
              absl::OkStatus());
    VerifyLineageGraph(output_graph, want_artifacts,
                       /*executions=*/{want_executions[1]},
                       /*events=*/{a1e1, a0e1}, *metadata_access_object_);
  }

  {
    // boundary artifact condition with max_num_hops.
    LineageGraph output_graph;
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/3,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/"uri != 'unknown_uri'",
                  /*boundary_executions=*/absl::nullopt, output_graph),
              absl::OkStatus());
    VerifyLineageGraph(output_graph, want_artifacts, want_executions,
                       all_events, *metadata_access_object_);
  }

  {
    // boundary artifact condition with max_num_hops and max nodes of 10.
    // It should return both artifacts and 8 exeuctions.
    LineageGraph output_graph;
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/3,
                  /*max_nodes=*/10,
                  /*boundary_artifacts=*/"uri != 'unknown_uri'",
                  /*boundary_executions=*/absl::nullopt, output_graph),
              absl::OkStatus());
    // Compare nodes and edges.
    EXPECT_THAT(output_graph.artifacts(),
                UnorderedPointwise(EqualsProto<Artifact>(), want_artifacts));
    EXPECT_EQ(output_graph.executions().size(), 8);
    EXPECT_EQ(output_graph.events().size(), 9);
  }

  {
    // boundary artifact and execution condition with max num hops
    LineageGraph output_graph;
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/2,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/"uri != 'uri_0'",
                  /*boundary_executions=*/"name != 'e0'", output_graph),
              absl::OkStatus());
    VerifyLineageGraph(output_graph, /*artifacts=*/{want_artifacts[1]},
                       /*executions=*/{want_executions[1]},
                       /*events=*/{a1e1}, *metadata_access_object_);
  }

  {
    // boundary condition rejects large number of nodes.
    LineageGraph output_graph;
    ASSERT_EQ(metadata_access_object_->QueryLineageGraph(
                  /*query_nodes=*/{want_artifacts[1]}, /*max_num_hops=*/3,
                  /*max_nodes=*/absl::nullopt,
                  /*boundary_artifacts=*/absl::nullopt,
                  /*boundary_executions=*/"name = 'e1'", output_graph),
              absl::OkStatus());
    VerifyLineageGraph(output_graph, /*artifacts=*/want_artifacts,
                       /*executions=*/{want_executions[1]},
                       /*events=*/{a1e1, a0e1}, *metadata_access_object_);
  }
}

TEST_P(MetadataAccessObjectTest, QueryLineageSubgraph) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Test setup: use a simple graph with multiple paths between (a1, e2).
  // a1 -> e1 -> a2
  //  \            \
  //   \------------> e2
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
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
    // Return invalid argument error if no starting nodes are provided.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options;
    google::protobuf::FieldMask read_mask =
        ParseTextProtoOrDie<google::protobuf::FieldMask>(
            R"pb(
              paths: "artifacts" paths: "executions" paths: "events"
            )pb");
    options.mutable_starting_artifacts()->set_filter_query("");
    options.set_max_num_hops(0);
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_access_object_->QueryLineageSubgraph(
            options, read_mask, output_subgraph)));

    options.mutable_starting_executions()->set_filter_query("");
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_access_object_->QueryLineageSubgraph(
            options, read_mask, output_subgraph)));
  }

  {
    // Query a1 with 1 hop.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options;
    google::protobuf::FieldMask read_mask =
        ParseTextProtoOrDie<google::protobuf::FieldMask>(
            R"pb(
              paths: "artifacts" paths: "executions" paths: "events"
            )pb");
    options.mutable_starting_artifacts()->set_filter_query(
        absl::Substitute("id = $0", want_artifacts[0].id()));
    options.set_max_num_hops(1);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[0].id()},
        {want_executions[0].id(), want_executions[1].id()},
        /*events=*/{want_events[0], want_events[1]});
  }
  {
    // Query a1 with 2 hops. It returns all nodes with no duplicate.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options;
    google::protobuf::FieldMask read_mask =
        ParseTextProtoOrDie<google::protobuf::FieldMask>(
            R"pb(
              paths: "artifacts" paths: "executions" paths: "events"
            )pb");
    options.mutable_starting_artifacts()->set_filter_query(
        absl::Substitute("id = $0", want_artifacts[0].id()));
    options.set_max_num_hops(2);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[0].id(), want_artifacts[1].id()},
        {want_executions[0].id(), want_executions[1].id()}, want_events);
  }

  {
    // With multiple query nodes with 0 hop.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options;
    google::protobuf::FieldMask read_mask =
        ParseTextProtoOrDie<google::protobuf::FieldMask>(
            R"pb(
              paths: "artifacts" paths: "executions" paths: "events"
            )pb");
    options.mutable_starting_artifacts()->set_filter_query(absl::Substitute(
        "id = $0 OR id = $1", want_artifacts[0].id(), want_artifacts[1].id()));
    options.set_max_num_hops(0);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(output_subgraph,
                               {want_artifacts[0].id(), want_artifacts[1].id()},
                               /*expected_execution_ids=*/{},
                               /*events=*/{});
  }

  {
    // Query multiple nodes with a large hop.
    // It returns all nodes with no duplicate.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options;
    google::protobuf::FieldMask read_mask =
        ParseTextProtoOrDie<google::protobuf::FieldMask>(
            R"pb(
              paths: "artifacts" paths: "executions" paths: "events"
            )pb");
    options.mutable_starting_artifacts()->set_filter_query(absl::Substitute(
        "id = $0 OR id = $1", want_artifacts[0].id(), want_artifacts[1].id()));
    options.set_max_num_hops(5);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[0].id(), want_artifacts[1].id()},
        {want_executions[0].id(), want_executions[1].id()}, want_events);
  }
  {
    // Query from e2 with 1 hop
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options;
    google::protobuf::FieldMask read_mask =
        ParseTextProtoOrDie<google::protobuf::FieldMask>(
            R"pb(
              paths: "artifacts" paths: "executions" paths: "events"
            )pb");
    options.mutable_starting_executions()->set_filter_query(
        absl::Substitute("id = $0", want_executions[1].id()));
    options.set_max_num_hops(1);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(output_subgraph,
                               {want_artifacts[0].id(), want_artifacts[1].id()},
                               {want_executions[1].id()},
                               /*events=*/{want_events[1], want_events[3]});
  }
  {
    // Query from e2 with 2 hops
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options;
    google::protobuf::FieldMask read_mask =
        ParseTextProtoOrDie<google::protobuf::FieldMask>(
            R"pb(
              paths: "artifacts" paths: "executions" paths: "events"
            )pb");
    options.mutable_starting_executions()->set_filter_query(
        absl::Substitute("id = $0", want_executions[1].id()));
    options.set_max_num_hops(2);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[0].id(), want_artifacts[1].id()},
        {want_executions[1].id(), want_executions[0].id()},
        /*events=*/want_events);
  }
  {
    // With multiple query nodes with 0 hop.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options;
    google::protobuf::FieldMask read_mask =
        ParseTextProtoOrDie<google::protobuf::FieldMask>(
            R"pb(
              paths: "artifacts" paths: "executions" paths: "events"
            )pb");
    options.mutable_starting_executions()->set_filter_query(
        absl::Substitute("id = $0 OR id = $1", want_executions[0].id(),
                         want_executions[1].id()));
    options.set_max_num_hops(0);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, /*expected_artifact_ids=*/
        {}, {want_executions[0].id(), want_executions[1].id()},
        /*events=*/{});
  }
}

TEST_P(MetadataAccessObjectTest, QueryLineageSubgraphWithEndingNodes) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Test setup: use a simple graph with multiple paths between (a1, e2).
  // a1 -> e1 -> a2
  //  \            \
  //   \------------> e2
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
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

  LineageSubgraphQueryOptions base_options;
  google::protobuf::FieldMask read_mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(
          R"pb(
            paths: "artifacts" paths: "executions" paths: "events"
          )pb");
  base_options.mutable_starting_artifacts()->set_filter_query(
      absl::Substitute("id = $0", want_artifacts[0].id()));
  base_options.mutable_ending_executions()->set_filter_query(
      absl::Substitute("id = $0", want_executions[0].id()));
  base_options.set_max_num_hops(2);

  {
    // Query a1 with 2 hops with e1 as ending node and include ending nodes.
    // Returned graph will be:
    // a1 -> e1    a2
    //  \            \
    //   \------------> e2
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.mutable_ending_executions()->set_include_ending_nodes(true);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[0].id(), want_artifacts[1].id()},
        {want_executions[0].id(), want_executions[1].id()},
        /*events=*/{want_events[0], want_events[1], want_events[3]});
  }
  {
    // Query a1 with 2 hops with e1 as ending node and don't include ending
    // nodes.
    // Returned graph will be:
    // a1          a2
    //  \            \
    //   \------------> e2
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(output_subgraph,
                               {want_artifacts[0].id(), want_artifacts[1].id()},
                               {want_executions[1].id()},
                               /*events=*/{want_events[1], want_events[3]});
  }
  {
    // Query a1 with 3 hops with e1 as ending node and include ending nodes.
    // Returned graph will be:
    // a1 -> e1 -> a2
    //  \            \
    //   \------------> e2
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.mutable_ending_executions()->set_include_ending_nodes(true);
    options.set_max_num_hops(3);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[0].id(), want_artifacts[1].id()},
        {want_executions[0].id(), want_executions[1].id()},
        /*events=*/
        {want_events[0], want_events[1], want_events[2], want_events[3]});
  }
  {
    // Query a1 with 3 hops with e1 as ending node and don't include ending
    // nodes.
    // Returned graph will be:
    // a1          a2
    //  \            \
    //   \------------> e2
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(3);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(output_subgraph,
                               {want_artifacts[0].id(), want_artifacts[1].id()},
                               {want_executions[1].id()},
                               /*events=*/{want_events[1], want_events[3]});
  }
  {
    // Query a1 with 3 hops in downstream direction with e1 as ending node and
    // include ending nodes.
    // Returned graph will be:
    // a1 -> e1
    //  \
    //   \------------> e2
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(3);
    options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
    options.mutable_ending_executions()->set_include_ending_nodes(true);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[0].id()},
        {want_executions[0].id(), want_executions[1].id()},
        /*events=*/{want_events[0], want_events[1]});
  }
  {
    // Query a1 with 3 hops in downstream direction with e1 as ending node and
    // don't include ending nodes.
    // Returned graph will be:
    // a1
    //  \
    //   \------------> e2
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(3);
    options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(output_subgraph, {want_artifacts[0].id()},
                               {want_executions[1].id()},
                               /*events=*/{want_events[1]});
  }
  {
    // Query a1 with 2 hops in downstream direction with a2 and e2 as ending
    // node, include a2 and don't include e2. Returned graph will be:
    // a1 -> e1 -> a2
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(3);
    options.mutable_ending_artifacts()->set_filter_query(
        absl::Substitute("id = $0", want_artifacts[1].id()));
    options.mutable_ending_artifacts()->set_include_ending_nodes(true);
    options.mutable_ending_executions()->set_filter_query(
        absl::Substitute("id = $0", want_executions[1].id()));
    options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(output_subgraph,
                               {want_artifacts[0].id(), want_artifacts[1].id()},
                               {want_executions[0].id()},
                               /*events=*/{want_events[0], want_events[2]});
  }
  {
    // Query a1 with 2 hops in downstream direction with a2 and e2 as ending
    // node, include e2 and don't include a2. Returned graph will be:
    // a1 -> e1
    //  \
    //   \------------> e2
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(3);
    options.mutable_ending_artifacts()->set_filter_query(
        absl::Substitute("id = $0", want_artifacts[1].id()));
    options.mutable_ending_executions()->set_filter_query(
        absl::Substitute("id = $0", want_executions[1].id()));
    options.mutable_ending_executions()->set_include_ending_nodes(true);
    options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[0].id()},
        {want_executions[0].id(), want_executions[1].id()},
        /*events=*/{want_events[0], want_events[1]});
  }
  {
    // Query e1 with 0 hop with e1 as ending node and don't include ending
    // nodes.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.clear_starting_artifacts();
    options.mutable_starting_executions()->set_filter_query(
        absl::Substitute("id = $0", want_executions[0].id()));
    options.set_max_num_hops(0);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(output_subgraph, {}, {}, {});
  }
  {
    // Query e1 with 0 hop with e1 as ending node and include ending nodes
    // nodes.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.clear_starting_artifacts();
    options.mutable_starting_executions()->set_filter_query(
        absl::Substitute("id = $0", want_executions[0].id()));
    options.set_max_num_hops(0);
    options.mutable_ending_executions()->set_include_ending_nodes(true);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(output_subgraph, {}, {want_executions[0].id()},
                               {});
  }
}

TEST_P(MetadataAccessObjectTest, QueryLineageSubgraphArtifactsOnly) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Test setup: only set up an artifact type and 2 artifacts.
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  std::vector<Artifact> want_artifacts(2);
  for (int i = 0; i < 2; i++) {
    CreateNodeFromTextProto(absl::Substitute("uri: 'uri_$0'", i),
                            artifact_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_artifacts[i]);
  }

  LineageGraph output_subgraph;
  LineageSubgraphQueryOptions options;
  google::protobuf::FieldMask read_mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(
          R"pb(
            paths: "artifacts" paths: "executions" paths: "events"
          )pb");
  options.mutable_starting_artifacts()->set_filter_query(absl::Substitute(
      "id = $0 OR id = $1", want_artifacts[0].id(), want_artifacts[1].id()));
  options.set_max_num_hops(1);
  ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                          output_subgraph),
            absl::OkStatus());
  VerifyLineageGraphSkeleton(output_subgraph,
                             {want_artifacts[0].id(), want_artifacts[1].id()},
                             /*expected_execution_ids=*/{},
                             /*events=*/{});
}

TEST_P(MetadataAccessObjectTest, QueryLineageSubgraphFromFilteredExecutions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Test setup: use a simple graph with multiple paths between (a1, e2).
  // c1(a1) -> e1(a1)-> c1(a2)
  //     \               \
  //      \------------> e2(a2)
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ContextType context_type = CreateTypeFromTextProto<ContextType>(
      "name: 'context_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  std::vector<Artifact> want_artifacts(2);
  std::vector<Execution> want_executions(2);
  std::vector<Context> want_contexts(2);
  CreateNodeFromTextProto(
      "name: 'context_1'", context_type.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), want_contexts[0]);
  CreateNodeFromTextProto(
      "name: 'context_2'", context_type.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), want_contexts[1]);
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
  Attribution attribution;
  attribution.set_artifact_id(want_artifacts[0].id());
  attribution.set_context_id(want_contexts[0].id());
  int64_t attribution_id;
  // Note using ASSERT_EQ as *_OK is not well supported in OSS
  ASSERT_EQ(
      metadata_access_object_->CreateAttribution(attribution, &attribution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  attribution.set_context_id(want_contexts[1].id());
  attribution.set_artifact_id(want_artifacts[1].id());
  ASSERT_EQ(
      metadata_access_object_->CreateAttribution(attribution, &attribution_id),
      absl::OkStatus());
  Association association;
  association.set_context_id(want_contexts[0].id());
  association.set_execution_id(want_executions[0].id());
  int64_t association_id;
  ASSERT_EQ(
      metadata_access_object_->CreateAssociation(association, &association_id),
      absl::OkStatus());
  association.set_context_id(want_contexts[1].id());
  association.set_execution_id(want_executions[1].id());
  ASSERT_EQ(
      metadata_access_object_->CreateAssociation(association, &association_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  LineageGraph output_subgraph;
  LineageSubgraphQueryOptions options;
  google::protobuf::FieldMask read_mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(
          R"pb(
            paths: "artifacts" paths: "executions" paths: "events"
          )pb");
  options.mutable_starting_executions()->set_filter_query(
      absl::Substitute("contexts_a.id = $0", want_contexts[0].id()));
  options.set_max_num_hops(1);
  ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                          output_subgraph),
            absl::OkStatus());
  VerifyLineageGraphSkeleton(
      output_subgraph, {want_artifacts[0].id(), want_artifacts[1].id()},
      /*expected_execution_ids=*/{want_executions[0].id()},
      /*events=*/{want_events[0], want_events[2]});
}

TEST_P(MetadataAccessObjectTest, QueryLineageSubgraphWithFieldMask) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Test setup: use a simple graph with multiple paths between (a1, e2).
  // a1(c1) -> e1(c1)-> a2(c2)
  //     \               \
  //      \------------> e2(c2)
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ContextType context_type = CreateTypeFromTextProto<ContextType>(
      "name: 'context_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  std::vector<Artifact> want_artifacts(2);
  std::vector<Execution> want_executions(2);
  std::vector<Context> want_contexts(2);
  CreateNodeFromTextProto(
      "name: 'context_1'", context_type.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), want_contexts[0]);
  CreateNodeFromTextProto(
      "name: 'context_2'", context_type.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), want_contexts[1]);
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
  Attribution attribution;
  attribution.set_artifact_id(want_artifacts[0].id());
  attribution.set_context_id(want_contexts[0].id());
  int64_t attribution_id;
  // Note using ASSERT_EQ as *_OK is not well supported in OSS, use
  // ASSERT_EQ(..., absl::OkStatus()) instead.
  ASSERT_EQ(
      metadata_access_object_->CreateAttribution(attribution, &attribution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  attribution.set_context_id(want_contexts[1].id());
  attribution.set_artifact_id(want_artifacts[1].id());
  ASSERT_EQ(
      metadata_access_object_->CreateAttribution(attribution, &attribution_id),
      absl::OkStatus());
  Association association;
  association.set_context_id(want_contexts[0].id());
  association.set_execution_id(want_executions[0].id());
  int64_t association_id;
  ASSERT_EQ(
      metadata_access_object_->CreateAssociation(association, &association_id),
      absl::OkStatus());
  association.set_context_id(want_contexts[1].id());
  association.set_execution_id(want_executions[1].id());
  ASSERT_EQ(
      metadata_access_object_->CreateAssociation(association, &association_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  LineageSubgraphQueryOptions base_options;
  google::protobuf::FieldMask base_read_mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(
          R"pb(
            paths: "artifacts"
            paths: "executions"
            paths: "contexts"
            paths: "events"
            paths: "artifact_types"
            paths: "execution_types"
            paths: "context_types"
          )pb");
  base_options.mutable_starting_executions()->set_filter_query(
      absl::Substitute("contexts_a.id = $0", want_contexts[0].id()));
  base_options.set_max_num_hops(1);
  {
    LineageSubgraphQueryOptions options = base_options;
    google::protobuf::FieldMask read_mask = base_read_mask;
    LineageGraph output_subgraph;
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    ASSERT_EQ(output_subgraph.artifacts_size(), 2);
    ASSERT_EQ(output_subgraph.executions_size(), 1);
    ASSERT_EQ(output_subgraph.contexts_size(), 2);
    ASSERT_EQ(output_subgraph.events_size(), 2);
    ASSERT_EQ(output_subgraph.artifact_types_size(), 1);
    ASSERT_EQ(output_subgraph.execution_types_size(), 1);
    ASSERT_EQ(output_subgraph.context_types_size(), 1);
  }
  {
    // Test: Get graph assets in 1 hop provenance.
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(2);
    google::protobuf::FieldMask read_mask = base_read_mask;
    LineageGraph output_subgraph;
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    ASSERT_EQ(output_subgraph.artifacts_size(), 2);
    ASSERT_EQ(output_subgraph.executions_size(), 2);
    ASSERT_EQ(output_subgraph.contexts_size(), 2);
    ASSERT_EQ(output_subgraph.events_size(), 4);
    ASSERT_EQ(output_subgraph.artifact_types_size(), 1);
    ASSERT_EQ(output_subgraph.execution_types_size(), 1);
    ASSERT_EQ(output_subgraph.context_types_size(), 1);
  }
  {
    // Test: Get lineage subgraph without read mask returns invalid argument
    // error.
    LineageSubgraphQueryOptions options = base_options;
    google::protobuf::FieldMask read_mask = base_read_mask;
    read_mask.mutable_paths()->Clear();

    LineageGraph output_subgraph;
    ASSERT_TRUE(
        absl::IsInvalidArgument(metadata_access_object_->QueryLineageSubgraph(
            options, read_mask, output_subgraph)));
  }
  {
    // Test: Get lineage subgraph with nodes and events.
    LineageSubgraphQueryOptions options = base_options;
    google::protobuf::FieldMask read_mask =
        ParseTextProtoOrDie<google::protobuf::FieldMask>(
            R"pb(
              paths: "artifacts"
              paths: "executions"
              paths: "contexts"
              paths: "events"
            )pb");

    LineageGraph output_subgraph;
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    ASSERT_EQ(output_subgraph.artifacts_size(), 2);
    ASSERT_EQ(output_subgraph.executions_size(), 1);
    ASSERT_EQ(output_subgraph.contexts_size(), 2);
    ASSERT_EQ(output_subgraph.events_size(), 2);
    ASSERT_TRUE(output_subgraph.artifact_types().empty());
    ASSERT_TRUE(output_subgraph.execution_types().empty());
    ASSERT_TRUE(output_subgraph.context_types().empty());
  }
  {
    // Test: Get lineage subgraph with artifacts and artifact types.
    LineageSubgraphQueryOptions options = base_options;
    google::protobuf::FieldMask read_mask =
        ParseTextProtoOrDie<google::protobuf::FieldMask>(
            R"pb(
              paths: "artifacts" paths: "artifact_types"
            )pb");

    LineageGraph output_subgraph;
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    ASSERT_EQ(output_subgraph.artifacts_size(), 2);
    ASSERT_TRUE(output_subgraph.executions().empty());
    ASSERT_TRUE(output_subgraph.contexts().empty());
    ASSERT_TRUE(output_subgraph.events().empty());
    ASSERT_EQ(output_subgraph.artifact_types_size(), 1);
    ASSERT_TRUE(output_subgraph.execution_types().empty());
    ASSERT_TRUE(output_subgraph.context_types().empty());
  }
}

TEST_P(MetadataAccessObjectTest, QueryLineageSubgraphDirectional) {
  // TODO (b/294894126): Increase test coverage on more test cases.
  ASSERT_EQ(Init(), absl::OkStatus());
  // Test setup: query from e2 in the following graph.
  // a1 -(v1)-> e1 -(v2)-> a2
  //  \                      \(v4)
  //   \---------(v3)---------> e2 -(v5)-> a3 -(v6)-> e3
  //                              \                     \(v7)
  //                               \---------(v8)--------> a4
  const ArtifactType artifact_type = CreateTypeFromTextProto<ArtifactType>(
      "name: 'artifact_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  const ExecutionType execution_type = CreateTypeFromTextProto<ExecutionType>(
      "name: 'execution_type'", *metadata_access_object_,
      metadata_access_object_container_.get());
  std::vector<Artifact> want_artifacts(4);
  std::vector<Execution> want_executions(3);
  for (int i = 0; i < 4; i++) {
    CreateNodeFromTextProto(absl::Substitute("uri: 'uri_$0'", i),
                            artifact_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_artifacts[i]);
  }
  for (int i = 0; i < 3; i++) {
    CreateNodeFromTextProto("", execution_type.id(), *metadata_access_object_,
                            metadata_access_object_container_.get(),
                            want_executions[i]);
  }
  std::vector<Event> want_events(8);
  CreateEventFromTextProto("type: INPUT", want_artifacts[0], want_executions[0],
                           *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[0]);
  CreateEventFromTextProto("type: OUTPUT", want_artifacts[1],
                           want_executions[0], *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[1]);
  CreateEventFromTextProto("type: INPUT", want_artifacts[0], want_executions[1],
                           *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[2]);
  CreateEventFromTextProto("type: INPUT", want_artifacts[1], want_executions[1],
                           *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[3]);
  CreateEventFromTextProto("type: OUTPUT", want_artifacts[2],
                           want_executions[1], *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[4]);
  CreateEventFromTextProto("type: INPUT", want_artifacts[2], want_executions[2],
                           *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[5]);
  CreateEventFromTextProto("type: OUTPUT", want_artifacts[3],
                           want_executions[2], *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[6]);
  CreateEventFromTextProto("type: OUTPUT", want_artifacts[3],
                           want_executions[1], *metadata_access_object_,
                           metadata_access_object_container_.get(),
                           want_events[7]);
  LineageSubgraphQueryOptions base_options;
  base_options.mutable_starting_executions()->set_filter_query(
      absl::Substitute("id = $0", want_executions[1].id()));
  google::protobuf::FieldMask read_mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(
          R"pb(
            paths: "artifacts" paths: "executions" paths: "events"
          )pb");
  {
    // Query from e2 with 1 hop.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(1);
    // Tracing downstream, returned subgraph will be:
    // e2 -(v5)-> a3
    //  \
    //   \---------(v8)--------> a4
    options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(output_subgraph,
                               {want_artifacts[2].id(), want_artifacts[3].id()},
                               {want_executions[1].id()},
                               /*events=*/{want_events[4], want_events[7]});
    // Tracing upstream, returned subgraph will be:
    // a1                    a2
    //  \                      \(v4)
    //   \---------(v3)---------> e2
    options.set_direction(LineageSubgraphQueryOptions::UPSTREAM);
    output_subgraph.Clear();
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(output_subgraph,
                               {want_artifacts[0].id(), want_artifacts[1].id()},
                               {want_executions[1].id()},
                               /*events=*/{want_events[2], want_events[3]});
  }
  {
    // Query from e2 with 2 hops.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(2);
    // Tracing downstream, returned subgraph will be:
    // e2 -(v5)-> a3 -(v6)-> e3
    //   \
    //    \---------(v8)--------> a4
    options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[2].id(), want_artifacts[3].id()},
        {want_executions[1].id(), want_executions[2].id()},
        {want_events[4], want_events[5], want_events[7]});
    // Tracing upstream, returned subgraph will be:
    // a1          e1 -(v2)-> a2
    //  \                      \(v4)
    //   \---------(v3)---------> e2
    options.set_direction(LineageSubgraphQueryOptions::UPSTREAM);
    output_subgraph.Clear();
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[0].id(), want_artifacts[1].id()},
        {want_executions[0].id(), want_executions[1].id()},
        /*events=*/{want_events[1], want_events[2], want_events[3]});
  }

  {
    // Query from e2 with a large hop.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(100);
    // Tracing downstream, returned subgraph will be:
    // e2 -(v5)-> a3 -(v6)-> e3
    //   \                     \(v7)
    //    \---------(v8)--------> a4
    options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[2].id(), want_artifacts[3].id()},
        {want_executions[1].id(), want_executions[2].id()},
        {want_events[4], want_events[5], want_events[6], want_events[7]});
    // Tracing upstream, returned subgraph will be:
    // a1 -(v1)-> e1 -(v2)-> a2
    //  \                      \(v4)
    //   \---------(v3)---------> e2
    options.set_direction(LineageSubgraphQueryOptions::UPSTREAM);
    output_subgraph.Clear();
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, {want_artifacts[0].id(), want_artifacts[1].id()},
        {want_executions[0].id(), want_executions[1].id()},
        /*events=*/
        {want_events[0], want_events[1], want_events[2], want_events[3]});
  }
  {
    // With multiple query nodes with 0 hop.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.mutable_starting_executions()->set_filter_query(absl::Substitute(
        "id = $0 OR id = $1 OR id = $2", want_executions[0].id(),
        want_executions[1].id(), want_executions[2].id()));
    options.set_max_num_hops(0);

    options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, /*expected_artifact_ids=*/
        {},
        {want_executions[0].id(), want_executions[1].id(),
         want_executions[2].id()},
        /*events=*/{});

    options.set_direction(LineageSubgraphQueryOptions::UPSTREAM);
    output_subgraph.Clear();
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, /*expected_artifact_ids=*/
        {},
        {want_executions[0].id(), want_executions[1].id(),
         want_executions[2].id()},
        /*events=*/{});
  }
  {
    // With multiple query nodes with 1 hop.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.mutable_starting_executions()->set_filter_query(absl::Substitute(
        "id = $0 OR id = $1 OR id = $2", want_executions[0].id(),
        want_executions[1].id(), want_executions[2].id()));
    options.set_max_num_hops(1);
    // Tracing downstream, returned subgraph will be:
    //             e1 -(v2)-> a2
    //
    //.                            e2 -(v5)-> a3         e3
    //                              \                     \(v7)
    //                               \---------(v8)--------> a4
    options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, /*expected_artifact_ids=*/
        {want_artifacts[1].id(), want_artifacts[2].id(),
         want_artifacts[3].id()},
        {want_executions[0].id(), want_executions[1].id(),
         want_executions[2].id()},
        /*events=*/
        {want_events[1], want_events[4], want_events[6], want_events[7]});

    // Tracing upstream, returned subgraph will be:
    // a1 -(v1)-> e1          a2
    //  \                      \(v4)
    //   \---------(v3)---------> e2          a3 -(v6)-> e3
    options.set_direction(LineageSubgraphQueryOptions::UPSTREAM);
    output_subgraph.Clear();
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, /*expected_artifact_ids=*/
        {want_artifacts[0].id(), want_artifacts[1].id(),
         want_artifacts[2].id()},
        {want_executions[0].id(), want_executions[1].id(),
         want_executions[2].id()},
        /*events=*/
        {want_events[0], want_events[2], want_events[3], want_events[5]});
  }
  {
    // With multiple query nodes with more hops.
    LineageGraph output_subgraph;
    LineageSubgraphQueryOptions options = base_options;
    options.mutable_starting_executions()->set_filter_query(absl::Substitute(
        "id = $0 OR id = $1 OR id = $2", want_executions[0].id(),
        want_executions[1].id(), want_executions[2].id()));
    options.set_max_num_hops(100);
    // Tracing downstream, returned subgraph will be:
    //             e1 -(v2)-> a2
    //                          \(v4)
    //                            e2 -(v5)-> a3 -(v6)-> e3
    //                              \                     \(v7)
    //                               \---------(v8)--------> a4
    options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, /*expected_artifact_ids=*/
        {want_artifacts[1].id(), want_artifacts[2].id(),
         want_artifacts[3].id()},
        {want_executions[0].id(), want_executions[1].id(),
         want_executions[2].id()},
        /*events=*/
        {want_events[1], want_events[3], want_events[4], want_events[5],
         want_events[6], want_events[7]});

    // Tracing upstream, returned subgraph will be:
    // a1 -(v1)-> e1 -(v2)-> a2
    //  \                      \(v4)
    //   \---------(v3)---------> e2 -(v5)-> a3 -(v6)-> e3
    options.set_direction(LineageSubgraphQueryOptions::UPSTREAM);
    output_subgraph.Clear();
    ASSERT_EQ(metadata_access_object_->QueryLineageSubgraph(options, read_mask,
                                                            output_subgraph),
              absl::OkStatus());
    VerifyLineageGraphSkeleton(
        output_subgraph, /*expected_artifact_ids=*/
        {want_artifacts[0].id(), want_artifacts[1].id(),
         want_artifacts[2].id()},
        {want_executions[0].id(), want_executions[1].id(),
         want_executions[2].id()},
        /*events=*/
        {want_events[0], want_events[1], want_events[2], want_events[3],
         want_events[4], want_events[5]});
  }
}

TEST_P(MetadataAccessObjectTest, DeleteArtifactsById) {
  ASSERT_EQ(Init(), absl::OkStatus());

  const ArtifactType type = CreateTypeFromTextProto<ArtifactType>(
      absl::StrCat(
          R"pb(
            name: 'test_type'
            properties { key: 'property_1' value: INT }
            properties { key: 'property_2' value: DOUBLE }
            properties { key: 'property_3' value: STRING }
          )pb",
          // TODO(b/257334039): cleanup fat client
          IfSchemaLessThan(10) ? "" :
                               R"pb(
            properties { key: 'property_4' value: PROTO }
            properties { key: 'property_5' value: BOOLEAN }
                               )pb",
          ""),
      *metadata_access_object_, metadata_access_object_container_.get());
  Artifact artifact;
  CreateNodeFromTextProto(
      absl::StrCat(
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
          // TODO(b/257334039): cleanup fat client
          IfSchemaLessThan(10) ? "" :
                               R"pb(
            properties {
              key: 'property_4'
              value {
                proto_value {
                  [type.googleapis.com/ml_metadata.testing.MockProto] {
                    string_value: '3'
                    double_value: 3.0
                  }
                }
              }
            }
            properties {
              key: 'property_5'
              value { bool_value: true }
            }
                               )pb",
          ""),
      type.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), artifact);
  // Test: empty ids
  {
    std::vector<Artifact> result;
    EXPECT_EQ(metadata_access_object_->DeleteArtifactsById({}),
              absl::OkStatus());
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact.id()}, &result),
        absl::OkStatus());
    EXPECT_THAT(result.size(), 1);
    ASSERT_EQ(
        absl::StatusOr<bool>(false),
        metadata_access_object_container_->CheckTableEmpty("ArtifactProperty"));
  }
  // Test: actual deletion
  {
    std::vector<Artifact> result;
    EXPECT_EQ(metadata_access_object_->DeleteArtifactsById({artifact.id()}),
              absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());

  const ExecutionType type = CreateTypeFromTextProto<ExecutionType>(
      absl::StrCat(
          R"pb(
            name: 'test_type'
            properties { key: 'property_1' value: INT }
            properties { key: 'property_2' value: DOUBLE }
            properties { key: 'property_3' value: STRING }
          )pb",
          // TODO(b/257334039): cleanup fat client
          IfSchemaLessThan(10) ? "" :
                               R"pb(
            properties { key: 'property_4' value: PROTO }
            properties { key: 'property_5' value: BOOLEAN }
                               )pb",
          ""),
      *metadata_access_object_, metadata_access_object_container_.get());
  Execution execution;
  CreateNodeFromTextProto(
      absl::StrCat(
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
          // TODO(b/257334039): cleanup fat client
          IfSchemaLessThan(10) ? "" :
                               R"pb(
            properties {
              key: 'property_4'
              value {
                proto_value {
                  [type.googleapis.com/ml_metadata.testing.MockProto] {
                    string_value: '3'
                    double_value: 3.0
                  }
                }
              }
            }
            properties {
              key: 'property_5'
              value { bool_value: true }
            }
                               )pb",
          ""),
      type.id(), *metadata_access_object_,
      metadata_access_object_container_.get(), execution);

  // Test: empty ids
  {
    std::vector<Execution> result;
    EXPECT_EQ(metadata_access_object_->DeleteExecutionsById({}),
              absl::OkStatus());
    ASSERT_EQ(
        metadata_access_object_->FindExecutionsById({execution.id()}, &result),
        absl::OkStatus());
    EXPECT_THAT(result.size(), 1);
    ASSERT_EQ(absl::StatusOr<bool>(false),
              metadata_access_object_container_->CheckTableEmpty(
                  "ExecutionProperty"));
  }
  // Test: actual deletion
  {
    std::vector<Execution> result;
    EXPECT_EQ(metadata_access_object_->DeleteExecutionsById({execution.id()}),
              absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());

  const ContextType type = CreateTypeFromTextProto<ContextType>(
      absl::StrCat(
          R"pb(
            name: 'test_type'
            properties { key: 'property_1' value: INT }
            properties { key: 'property_2' value: DOUBLE }
            properties { key: 'property_3' value: STRING }
          )pb",
          // TODO(b/257334039): cleanup fat client
          IfSchemaLessThan(10) ? "" :
                               R"pb(
            properties { key: 'property_4' value: PROTO }
            properties { key: 'property_5' value: BOOLEAN }
                               )pb",
          ""),
      *metadata_access_object_, metadata_access_object_container_.get());
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
    EXPECT_EQ(metadata_access_object_->DeleteContextsById({}),
              absl::OkStatus());
    ASSERT_EQ(metadata_access_object_->FindContextsById(
                  {context1.id(), context2.id()}, &result),
              absl::OkStatus());
    EXPECT_THAT(result.size(), 2);
  }
  // Test: actual deletion on context1
  {
    std::vector<Context> result;
    EXPECT_EQ(metadata_access_object_->DeleteContextsById({context1.id()}),
              absl::OkStatus());
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
    EXPECT_EQ(metadata_access_object_->DeleteContextsById({context2.id() + 1}),
              absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());

  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64_t execution_type_id = InsertType<ExecutionType>("test_execution_type");
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
    EXPECT_EQ(metadata_access_object_->DeleteEventsByArtifactsId({}),
              absl::OkStatus());
    ASSERT_EQ(metadata_access_object_->FindEventsByArtifacts(
                  {input_artifact.id(), output_artifact.id()}, &result),
              absl::OkStatus());
    EXPECT_THAT(result.size(), 2);
  }
  // Test: delete one event
  {
    std::vector<Event> result;
    EXPECT_EQ(metadata_access_object_->DeleteEventsByArtifactsId(
                  {input_artifact.id()}),
              absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());

  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64_t execution_type_id = InsertType<ExecutionType>("test_execution_type");
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
    EXPECT_EQ(metadata_access_object_->DeleteEventsByExecutionsId({}),
              absl::OkStatus());
    ASSERT_EQ(metadata_access_object_->FindEventsByExecutions({execution.id()},
                                                              &result),
              absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());

  int64_t execution_type_id = InsertType<ExecutionType>("execution_type");
  int64_t context_type_id = InsertType<ContextType>("context_type");
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

  int64_t association_id;
  EXPECT_EQ(
      metadata_access_object_->CreateAssociation(association, &association_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: empty ids
  {
    std::vector<Execution> result;
    EXPECT_EQ(metadata_access_object_->DeleteAssociationsByContextsId({}),
              absl::OkStatus());
    ASSERT_EQ(
        metadata_access_object_->FindExecutionsByContext(context.id(), &result),
        absl::OkStatus());
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete association
  {
    std::vector<Execution> result;
    EXPECT_EQ(
        metadata_access_object_->DeleteAssociationsByContextsId({context.id()}),
        absl::OkStatus());
    ASSERT_EQ(
        metadata_access_object_->FindExecutionsByContext(context.id(), &result),
        absl::OkStatus());
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteAssociationsByExecutionsId) {
  ASSERT_EQ(Init(), absl::OkStatus());

  int64_t execution_type_id = InsertType<ExecutionType>("execution_type");
  int64_t context_type_id = InsertType<ContextType>("context_type");

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

  int64_t association_id;
  EXPECT_EQ(
      metadata_access_object_->CreateAssociation(association, &association_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: empty ids
  {
    std::vector<Context> result;
    EXPECT_EQ(metadata_access_object_->DeleteAssociationsByExecutionsId({}),
              absl::OkStatus());
    ASSERT_EQ(metadata_access_object_->FindContextsByExecution(execution.id(),
                                                               &result),
              absl::OkStatus());
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete association
  {
    std::vector<Context> result;
    EXPECT_EQ(metadata_access_object_->DeleteAssociationsByExecutionsId(
                  {execution.id()}),
              absl::OkStatus());
    // Semantics for empty here is different than in FindExecutionsByContext.
    // Here if there is none, a notFound error is returned.
    absl::Status status = metadata_access_object_->FindContextsByExecution(
        execution.id(), &result);
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteAttributionsByContextsId) {
  ASSERT_EQ(Init(), absl::OkStatus());

  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64_t context_type_id = InsertType<ContextType>("test_context_type");
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

  int64_t attribution_id;
  EXPECT_EQ(
      metadata_access_object_->CreateAttribution(attribution, &attribution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: empty ids
  {
    std::vector<Artifact> result;
    EXPECT_EQ(metadata_access_object_->DeleteAttributionsByContextsId({}),
              absl::OkStatus());
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsByContext(context.id(), &result),
        absl::OkStatus());
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete attribution
  {
    std::vector<Artifact> result;
    EXPECT_EQ(
        metadata_access_object_->DeleteAttributionsByContextsId({context.id()}),
        absl::OkStatus());
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsByContext(context.id(), &result),
        absl::OkStatus());
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteAttributionsByArtifactsId) {
  ASSERT_EQ(Init(), absl::OkStatus());

  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64_t context_type_id = InsertType<ContextType>("test_context_type");
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

  int64_t attribution_id;
  EXPECT_EQ(
      metadata_access_object_->CreateAttribution(attribution, &attribution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: empty ids
  {
    std::vector<Context> result;
    EXPECT_EQ(metadata_access_object_->DeleteAttributionsByArtifactsId({}),
              absl::OkStatus());
    ASSERT_EQ(
        metadata_access_object_->FindContextsByArtifact(artifact.id(), &result),
        absl::OkStatus());
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete attribution
  {
    std::vector<Context> result;
    EXPECT_EQ(metadata_access_object_->DeleteAttributionsByArtifactsId(
                  {artifact.id()}),
              absl::OkStatus());
    // Semantics for empty here is different than in FindArtifactsByContext.
    // Here if there is none, a notFound error is returned.
    absl::Status status =
        metadata_access_object_->FindContextsByArtifact(artifact.id(), &result);
    EXPECT_TRUE(absl::IsNotFound(status)) << status;
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteParentType) {
  ASSERT_EQ(Init(), absl::OkStatus());

  {
    // Test: create and delete artifact parent type inheritance link
    const ArtifactType type1 = CreateTypeFromTextProto<ArtifactType>(
        "name: 't1'", *metadata_access_object_,
        metadata_access_object_container_.get());
    const ArtifactType type2 = CreateTypeFromTextProto<ArtifactType>(
        "name: 't2'", *metadata_access_object_,
        metadata_access_object_container_.get());
    const ArtifactType type3 = CreateTypeFromTextProto<ArtifactType>(
        "name: 't3'", *metadata_access_object_,
        metadata_access_object_container_.get());

    // create parent type links ok.
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type3));
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type2, type3));
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

    absl::flat_hash_map<int64_t, ArtifactType> output_artifact_types;
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id(), type2.id()}, output_artifact_types),
              absl::OkStatus());
    ASSERT_EQ(output_artifact_types.size(), 2);

    // delete parent link (type1, type3)
    ASSERT_EQ(metadata_access_object_->DeleteParentTypeInheritanceLink(
                  type1.id(), type3.id()),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

    output_artifact_types.clear();
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id(), type2.id()}, output_artifact_types),
              absl::OkStatus());
    ASSERT_EQ(output_artifact_types.size(), 1);
    EXPECT_TRUE(output_artifact_types.contains(type2.id()));

    // delete parent link (type2, type3)
    ASSERT_EQ(metadata_access_object_->DeleteParentTypeInheritanceLink(
                  type2.id(), type3.id()),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

    output_artifact_types.clear();
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id()}, output_artifact_types),
              absl::OkStatus());
    EXPECT_THAT(output_artifact_types, IsEmpty());
  }

  {
    // Test: create and delete execution parent type inheritance link
    const ExecutionType type1 = CreateTypeFromTextProto<ExecutionType>(
        "name: 't1'", *metadata_access_object_,
        metadata_access_object_container_.get());
    const ExecutionType type2 = CreateTypeFromTextProto<ExecutionType>(
        "name: 't2'", *metadata_access_object_,
        metadata_access_object_container_.get());
    // create parent type link ok.
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

    // delete parent link (type1, type2)
    ASSERT_EQ(metadata_access_object_->DeleteParentTypeInheritanceLink(
                  type1.id(), type2.id()),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

    absl::flat_hash_map<int64_t, ExecutionType> output_execution_types;
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id()}, output_execution_types),
              absl::OkStatus());
    EXPECT_TRUE(output_execution_types.empty());
  }

  {
    // Test: create and delete context parent type inheritance link
    const ContextType type1 = CreateTypeFromTextProto<ContextType>(
        "name: 't1'", *metadata_access_object_,
        metadata_access_object_container_.get());
    const ContextType type2 = CreateTypeFromTextProto<ContextType>(
        "name: 't2'", *metadata_access_object_,
        metadata_access_object_container_.get());
    // create parent type link ok.
    ASSERT_EQ(
        absl::OkStatus(),
        metadata_access_object_->CreateParentTypeInheritanceLink(type1, type2));
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

    // delete parent link (type1, type2)
    ASSERT_EQ(metadata_access_object_->DeleteParentTypeInheritanceLink(
                  type1.id(), type2.id()),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

    absl::flat_hash_map<int64_t, ContextType> output_context_types;
    ASSERT_EQ(metadata_access_object_->FindParentTypesByTypeId(
                  {type1.id()}, output_context_types),
              absl::OkStatus());
    EXPECT_TRUE(output_context_types.empty());
  }

  {
    // Test: delete non-existing context parent type inheritance link
    const ContextType type1 = CreateTypeFromTextProto<ContextType>(
        "name: 't1'", *metadata_access_object_,
        metadata_access_object_container_.get());
    const ContextType type2 = CreateTypeFromTextProto<ContextType>(
        "name: 't2'", *metadata_access_object_,
        metadata_access_object_container_.get());

    // delete non-existing parent link (type1, type2) returns ok
    ASSERT_EQ(metadata_access_object_->DeleteParentTypeInheritanceLink(
                  type1.id(), type2.id()),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteParentContextsByParentIds) {
  ASSERT_EQ(Init(), absl::OkStatus());

  int64_t context_type_id = InsertType<ContextType>("test_context_type");

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
  EXPECT_EQ(metadata_access_object_->CreateParentContext(parent_context),
            absl::OkStatus());

  // Test: empty ids
  {
    std::vector<Context> result;
    EXPECT_EQ(metadata_access_object_->DeleteParentContextsByParentIds({}),
              absl::OkStatus());
    ASSERT_EQ(metadata_access_object_->FindParentContextsByContextId(
                  context2.id(), &result),
              absl::OkStatus());
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete parent context
  {
    std::vector<Context> result;
    EXPECT_EQ(metadata_access_object_->DeleteParentContextsByParentIds(
                  {context1.id()}),
              absl::OkStatus());
    ASSERT_EQ(metadata_access_object_->FindParentContextsByContextId(
                  context2.id(), &result),
              absl::OkStatus());
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteParentContextsByChildIds) {
  ASSERT_EQ(Init(), absl::OkStatus());

  int64_t context_type_id = InsertType<ContextType>("test_context_type");

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
  EXPECT_EQ(metadata_access_object_->CreateParentContext(parent_context),
            absl::OkStatus());

  // Test: empty ids
  {
    std::vector<Context> result;
    EXPECT_EQ(metadata_access_object_->DeleteParentContextsByChildIds({}),
              absl::OkStatus());
    ASSERT_EQ(metadata_access_object_->FindParentContextsByContextId(
                  context2.id(), &result),
              absl::OkStatus());
    EXPECT_THAT(result.size(), 1);
  }
  // Test: delete parent context
  {
    std::vector<Context> result;
    EXPECT_EQ(metadata_access_object_->DeleteParentContextsByChildIds(
                  {context2.id()}),
              absl::OkStatus());
    ASSERT_EQ(metadata_access_object_->FindParentContextsByContextId(
                  context2.id(), &result),
              absl::OkStatus());
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, DeleteParentContextsByParentIdAndChildIds) {
  MLMD_ASSERT_OK(Init());

  int64_t context_type_id = InsertType<ContextType>("test_context_type");

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
    MLMD_EXPECT_OK(
        metadata_access_object_->DeleteParentContextsByParentIdAndChildIds(
            context3.id(), {context2.id()}));
    MLMD_ASSERT_OK(metadata_access_object_->FindParentContextsByContextId(
        context2.id(), &result));
    EXPECT_THAT(result, SizeIs(1));
  }
  // Test: empty child ids
  {
    std::vector<Context> result;
    MLMD_EXPECT_OK(
        metadata_access_object_->DeleteParentContextsByParentIdAndChildIds(
            context1.id(), {}));
    MLMD_ASSERT_OK(metadata_access_object_->FindParentContextsByContextId(
        context2.id(), &result));
    EXPECT_THAT(result, SizeIs(1));
  }
  // Test: delete parent context
  {
    std::vector<Context> result;
    MLMD_EXPECT_OK(
        metadata_access_object_->DeleteParentContextsByParentIdAndChildIds(
            context1.id(), {context2.id()}));
    MLMD_ASSERT_OK(metadata_access_object_->FindParentContextsByContextId(
        context2.id(), &result));
    EXPECT_THAT(result, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, ListArtifactsWithNonIdFieldOptions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t last_stored_artifact_id;

  for (int i = 0; i < total_stored_artifacts; i++) {
    ASSERT_EQ(metadata_access_object_->CreateArtifact(sample_artifact,
                                                      &last_stored_artifact_id),
              absl::OkStatus());
  }
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 2,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )pb");

  int64_t expected_artifact_id = last_stored_artifact_id;
  std::string next_page_token;

  do {
    std::vector<Artifact> got_artifacts;
    ASSERT_EQ(metadata_access_object_->ListArtifacts(
                  list_options, &got_artifacts, &next_page_token),
              absl::OkStatus());
    EXPECT_TRUE(got_artifacts.size() <= page_size);
    for (const Artifact& artifact : got_artifacts) {
      sample_artifact.set_id(expected_artifact_id--);
      EXPECT_THAT(artifact, EqualsProto(sample_artifact, /*ignore_fields=*/{
                                            "type", "create_time_since_epoch",
                                            "last_update_time_since_epoch"}));
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_EQ(expected_artifact_id, 0);
}

TEST_P(MetadataAccessObjectTest, ListArtifactsWithIdFieldOptions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact sample_artifact = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )pb");

  sample_artifact.set_type_id(type_id);
  int stored_artifacts_count = 0;
  int64_t first_artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(sample_artifact,
                                                    &first_artifact_id),
            absl::OkStatus());
  stored_artifacts_count++;

  for (int i = 0; i < 6; i++) {
    int64_t unused_artifact_id;
    ASSERT_EQ(metadata_access_object_->CreateArtifact(sample_artifact,
                                                      &unused_artifact_id),
              absl::OkStatus());
  }
  stored_artifacts_count += 6;
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 2,
        order_by_field: { field: ID is_asc: true }
      )pb");

  std::string next_page_token;
  int64_t expected_artifact_id = first_artifact_id;
  int seen_artifacts_count = 0;
  do {
    std::vector<Artifact> got_artifacts;
    ASSERT_EQ(metadata_access_object_->ListArtifacts(
                  list_options, &got_artifacts, &next_page_token),
              absl::OkStatus());
    EXPECT_TRUE(got_artifacts.size() <= page_size);
    for (const Artifact& artifact : got_artifacts) {
      sample_artifact.set_id(expected_artifact_id++);

      EXPECT_THAT(artifact, EqualsProto(sample_artifact, /*ignore_fields=*/{
                                            "type", "create_time_since_epoch",
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
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  std::vector<int64_t> stored_artifact_ids;
  for (int i = 0; i < total_stored_artifacts; i++) {
    int64_t created_artifact_id;
    absl::SleepFor(absl::Milliseconds(1));
    ASSERT_EQ(metadata_access_object_->CreateArtifact(sample_artifact,
                                                      &created_artifact_id),
              absl::OkStatus());
    stored_artifact_ids.push_back(created_artifact_id);
  }

  // Setting the expected list in the order [3, 2, 1, 6, 5, 4]
  std::list<int64_t> expected_artifact_ids;
  for (int i = 3; i < total_stored_artifacts; i++) {
    expected_artifact_ids.push_front(stored_artifact_ids[i]);
  }

  sample_artifact.set_state(ml_metadata::Artifact::State::Artifact_State_LIVE);
  for (int i = 0; i < 3; i++) {
    sample_artifact.set_id(stored_artifact_ids[i]);
    absl::SleepFor(absl::Milliseconds(1));
    ASSERT_EQ(metadata_access_object_->UpdateArtifact(sample_artifact),
              absl::OkStatus());
    expected_artifact_ids.push_front(stored_artifact_ids[i]);
  }
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 2,
        order_by_field: { field: LAST_UPDATE_TIME is_asc: false }
      )pb");

  std::string next_page_token;
  do {
    std::vector<Artifact> got_artifacts;
    ASSERT_EQ(metadata_access_object_->ListArtifacts(
                  list_options, &got_artifacts, &next_page_token),
              absl::OkStatus());
    EXPECT_LE(got_artifacts.size(), page_size);
    for (const Artifact& artifact : got_artifacts) {
      sample_artifact.set_id(expected_artifact_ids.front());
      EXPECT_THAT(artifact,
                  EqualsProto(sample_artifact, /*ignore_fields=*/{
                                  "type", "state", "create_time_since_epoch",
                                  "last_update_time_since_epoch"}));
      expected_artifact_ids.pop_front();
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_THAT(expected_artifact_ids, IsEmpty());
}

TEST_P(MetadataAccessObjectTest, ListArtifactsWithChangedOptions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact sample_artifact = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
  )pb");

  sample_artifact.set_type_id(type_id);
  int64_t last_stored_artifact_id;

  ASSERT_EQ(metadata_access_object_->CreateArtifact(sample_artifact,
                                                    &last_stored_artifact_id),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_->CreateArtifact(sample_artifact,
                                                    &last_stored_artifact_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )pb");

  std::string next_page_token_string;
  std::vector<Artifact> got_artifacts;
  ASSERT_EQ(metadata_access_object_->ListArtifacts(list_options, &got_artifacts,
                                                   &next_page_token_string),
            absl::OkStatus());
  EXPECT_EQ(got_artifacts.size(), 1);
  EXPECT_EQ(got_artifacts[0].id(), last_stored_artifact_id);

  ListOperationOptions updated_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: true }
      )pb");

  updated_options.set_next_page_token(next_page_token_string);
  std::vector<Artifact> unused_artifacts;
  std::string unused_next_page_token;
  EXPECT_TRUE(absl::IsInvalidArgument(metadata_access_object_->ListArtifacts(
      updated_options, &unused_artifacts, &unused_next_page_token)));
}

TEST_P(MetadataAccessObjectTest, ListArtifactsWithInvalidNextPageToken) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  Artifact sample_artifact = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
  )pb");

  sample_artifact.set_type_id(type_id);
  int64_t last_stored_artifact_id;

  ASSERT_EQ(metadata_access_object_->CreateArtifact(sample_artifact,
                                                    &last_stored_artifact_id),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_->CreateArtifact(sample_artifact,
                                                    &last_stored_artifact_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )pb");

  std::string next_page_token_string;
  std::vector<Artifact> got_artifacts;
  ASSERT_EQ(metadata_access_object_->ListArtifacts(list_options, &got_artifacts,
                                                   &next_page_token_string),
            absl::OkStatus());
  EXPECT_EQ(got_artifacts.size(), 1);
  EXPECT_EQ(got_artifacts[0].id(), last_stored_artifact_id);

  list_options.set_next_page_token("Invalid String");
  std::vector<Artifact> unused_artifacts;
  std::string unused_next_page_token;
  EXPECT_TRUE(absl::IsInvalidArgument(metadata_access_object_->ListArtifacts(
      list_options, &unused_artifacts, &unused_next_page_token)));
}

TEST_P(MetadataAccessObjectTest, ListExecutionsWithNonIdFieldOptions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution sample_execution = ParseTextProtoOrDie<Execution>(R"pb(
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
  sample_execution.set_type_id(type_id);
  const int total_stored_executions = 6;
  int64_t last_stored_execution_id;

  for (int i = 0; i < total_stored_executions; i++) {
    ASSERT_EQ(metadata_access_object_->CreateExecution(
                  sample_execution, &last_stored_execution_id),
              absl::OkStatus());
  }
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 2,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )pb");

  int64_t expected_execution_id = last_stored_execution_id;
  std::string next_page_token;

  do {
    std::vector<Execution> got_executions;
    ASSERT_EQ(metadata_access_object_->ListExecutions(
                  list_options, &got_executions, &next_page_token),
              absl::OkStatus());
    EXPECT_TRUE(got_executions.size() <= page_size);
    for (const Execution& execution : got_executions) {
      sample_execution.set_id(expected_execution_id--);

      EXPECT_THAT(execution, EqualsProto(sample_execution, /*ignore_fields=*/{
                                             "type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_EQ(expected_execution_id, 0);
}

TEST_P(MetadataAccessObjectTest, ListExecutionsWithIdFieldOptions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution sample_execution = ParseTextProtoOrDie<Execution>(R"pb(
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )pb");

  sample_execution.set_type_id(type_id);
  int stored_executions_count = 0;
  int64_t first_execution_id;
  ASSERT_EQ(metadata_access_object_->CreateExecution(sample_execution,
                                                     &first_execution_id),
            absl::OkStatus());
  stored_executions_count++;

  for (int i = 0; i < 6; i++) {
    int64_t unused_execution_id;
    ASSERT_EQ(metadata_access_object_->CreateExecution(sample_execution,
                                                       &unused_execution_id),
              absl::OkStatus());
  }
  stored_executions_count += 6;
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 2,
        order_by_field: { field: ID is_asc: true }
      )pb");

  std::string next_page_token;
  int64_t expected_execution_id = first_execution_id;
  int seen_executions_count = 0;
  do {
    std::vector<Execution> got_executions;
    ASSERT_EQ(metadata_access_object_->ListExecutions(
                  list_options, &got_executions, &next_page_token),
              absl::OkStatus());
    EXPECT_TRUE(got_executions.size() <= page_size);
    for (const Execution& execution : got_executions) {
      sample_execution.set_id(expected_execution_id++);

      EXPECT_THAT(execution, EqualsProto(sample_execution, /*ignore_fields=*/{
                                             "type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
      seen_executions_count++;
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_EQ(stored_executions_count, seen_executions_count);
}

TEST_P(MetadataAccessObjectTest, ListContextsWithNonIdFieldOptions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context sample_context = ParseTextProtoOrDie<Context>(R"pb(
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
  sample_context.set_type_id(type_id);
  int64_t last_stored_context_id;
  int context_name_suffix = 0;
  sample_context.set_name("list_contexts_test-1");
  ASSERT_EQ(metadata_access_object_->CreateContext(sample_context,
                                                   &last_stored_context_id),
            absl::OkStatus());

  context_name_suffix++;
  sample_context.set_name("list_contexts_test-2");
  ASSERT_EQ(metadata_access_object_->CreateContext(sample_context,
                                                   &last_stored_context_id),
            absl::OkStatus());
  context_name_suffix++;
  sample_context.set_name("list_contexts_test-3");
  ASSERT_EQ(metadata_access_object_->CreateContext(sample_context,
                                                   &last_stored_context_id),
            absl::OkStatus());
  context_name_suffix++;
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 2,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )pb");

  int64_t expected_context_id = last_stored_context_id;
  std::string next_page_token;

  do {
    std::vector<Context> got_contexts;
    ASSERT_EQ(metadata_access_object_->ListContexts(list_options, &got_contexts,
                                                    &next_page_token),
              absl::OkStatus());
    EXPECT_TRUE(got_contexts.size() <= page_size);
    for (const Context& context : got_contexts) {
      sample_context.set_name(
          absl::StrCat("list_contexts_test-", context_name_suffix--));
      sample_context.set_id(expected_context_id--);
      EXPECT_THAT(context, EqualsProto(sample_context, /*ignore_fields=*/{
                                           "type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}));
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_EQ(expected_context_id, 0);
}

TEST_P(MetadataAccessObjectTest, ListContextsWithIdFieldOptions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context sample_context = ParseTextProtoOrDie<Context>(R"pb(
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )pb");

  sample_context.set_type_id(type_id);
  int stored_contexts_count = 0;
  int64_t first_context_id;
  sample_context.set_name("list_contexts_test-1");
  ASSERT_EQ(
      metadata_access_object_->CreateContext(sample_context, &first_context_id),
      absl::OkStatus());

  int64_t unused_context_id;
  stored_contexts_count++;
  sample_context.set_name("list_contexts_test-2");
  ASSERT_EQ(metadata_access_object_->CreateContext(sample_context,
                                                   &unused_context_id),
            absl::OkStatus());
  stored_contexts_count++;
  sample_context.set_name("list_contexts_test-3");
  ASSERT_EQ(metadata_access_object_->CreateContext(sample_context,
                                                   &unused_context_id),
            absl::OkStatus());
  stored_contexts_count++;
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  const int page_size = 2;
  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 2,
        order_by_field: { field: ID is_asc: true }
      )pb");

  std::string next_page_token;
  int64_t expected_context_id = first_context_id;
  int expected_context_name_suffix = 1;
  int seen_contexts_count = 0;
  do {
    std::vector<Context> got_contexts;
    ASSERT_EQ(metadata_access_object_->ListContexts(list_options, &got_contexts,
                                                    &next_page_token),
              absl::OkStatus());
    EXPECT_TRUE(got_contexts.size() <= page_size);
    for (const Context& context : got_contexts) {
      sample_context.set_name(
          absl::StrCat("list_contexts_test-", expected_context_name_suffix++));
      sample_context.set_id(expected_context_id++);

      EXPECT_THAT(context, EqualsProto(sample_context, /*ignore_fields=*/{
                                           "type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}));
      seen_contexts_count++;
    }
    list_options.set_next_page_token(next_page_token);
  } while (!next_page_token.empty());

  EXPECT_EQ(stored_contexts_count, seen_contexts_count);
}

TEST_P(MetadataAccessObjectTest, GetContextsById) {
  ASSERT_EQ(Init(), absl::OkStatus());

  // Setup: create the type for the context
  int64_t type_id;
  std::string test_type_name = "test_type";
  {
    ContextType type = ParseTextProtoOrDie<ContextType>(
        absl::StrFormat(R"pb(
                          name: '%s'
                          properties { key: 'property_1' value: INT }
                        )pb",
                        test_type_name));
    ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  }

  // Setup: Add first context instance
  Context first_context;
  {
    first_context = ParseTextProtoOrDie<Context>(R"pb(
      properties {
        key: 'property_1'
        value: { int_value: 3 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { string_value: 'foo' }
      }
    )pb");
    int64_t first_context_id;
    first_context.set_type_id(type_id);
    first_context.set_name("get_contexts_by_id_test-1");
    ASSERT_EQ(metadata_access_object_->CreateContext(first_context,
                                                     &first_context_id),
              absl::OkStatus());
    first_context.set_id(first_context_id);
    first_context.set_type(test_type_name);
  }

  // Setup: Add second context instance
  Context second_context;
  {
    second_context = ParseTextProtoOrDie<Context>(R"pb(
      properties {
        key: 'property_1'
        value: { int_value: 5 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { string_value: 'bar' }
      }
    )pb");
    int64_t second_context_id;
    second_context.set_type_id(type_id);
    second_context.set_name("get_contexts_by_id_test-2");
    ASSERT_EQ(metadata_access_object_->CreateContext(second_context,
                                                     &second_context_id),
              absl::OkStatus());
    second_context.set_id(second_context_id);
    second_context.set_type(test_type_name);
  }

  // Setup: Add third context instance that does not have *any* properties
  Context third_context;
  {
    int64_t third_context_id;
    third_context.set_type_id(type_id);
    third_context.set_name("get_contexts_by_id_test-3");
    ASSERT_EQ(metadata_access_object_->CreateContext(third_context,
                                                     &third_context_id),
              absl::OkStatus());
    third_context.set_id(third_context_id);
    third_context.set_type(test_type_name);
  }

  const int64_t unknown_id =
      first_context.id() + second_context.id() + third_context.id() + 1;

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: empty ids
  {
    std::vector<Context> result;
    EXPECT_EQ(metadata_access_object_->FindContextsById({}, &result),
              absl::OkStatus());
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
    ASSERT_EQ(metadata_access_object_->FindContextsById({first_context.id()},
                                                        &result),
              absl::OkStatus());
    EXPECT_THAT(result,
                ElementsAre(EqualsProto(first_context, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));
  }
  {
    std::vector<Context> result;
    ASSERT_EQ(metadata_access_object_->FindContextsById({second_context.id()},
                                                        &result),
              absl::OkStatus());
    EXPECT_THAT(result,
                ElementsAre(EqualsProto(second_context, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));
  }
  {
    std::vector<Context> result;
    ASSERT_EQ(metadata_access_object_->FindContextsById({third_context.id()},
                                                        &result),
              absl::OkStatus());
    EXPECT_THAT(result,
                ElementsAre(EqualsProto(third_context, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));
  }
  // Test: retrieve multiple contexts at a time
  {
    std::vector<Context> result;
    const std::vector<int64_t> ids = {first_context.id(), second_context.id(),
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
    const std::vector<int64_t> ids = {first_context.id(), second_context.id(),
                                      third_context.id()};
    ASSERT_EQ(metadata_access_object_->FindContextsById(ids, &result),
              absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>("name: 'test_type'");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // artifact 1 does not set the state
  Artifact want_artifact1;
  want_artifact1.set_uri("uri: 'testuri://testing/uri/1'");
  want_artifact1.set_type_id(type_id);
  int64_t id1;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(want_artifact1, &id1),
            absl::OkStatus());
  // artifact 2 sets the state to default UNKNOWN
  Artifact want_artifact2;
  want_artifact2.set_type_id(type_id);
  want_artifact2.set_uri("uri: 'testuri://testing/uri/2'");
  want_artifact2.set_state(Artifact::UNKNOWN);
  int64_t id2;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(want_artifact2, &id2),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  std::vector<Artifact> artifacts;
  ASSERT_EQ(metadata_access_object_->FindArtifacts(&artifacts),
            absl::OkStatus());
  ASSERT_EQ(artifacts.size(), 2);
  EXPECT_THAT(
      artifacts,
      UnorderedElementsAre(
          EqualsProto(
              want_artifact1,
              /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                 "last_update_time_since_epoch"}),
          EqualsProto(want_artifact2, /*ignore_fields=*/{
                          "id", "type", "create_time_since_epoch",
                          "last_update_time_since_epoch"})));
}

TEST_P(MetadataAccessObjectTest, FindArtifactsByTypeIds) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t type_id = InsertType<ArtifactType>("test_type");
  Artifact want_artifact1 =
      ParseTextProtoOrDie<Artifact>("uri: 'testuri://testing/uri1'");
  want_artifact1.set_type_id(type_id);
  int64_t artifact1_id;
  ASSERT_EQ(
      metadata_access_object_->CreateArtifact(want_artifact1, &artifact1_id),
      absl::OkStatus());

  Artifact want_artifact2 =
      ParseTextProtoOrDie<Artifact>("uri: 'testuri://testing/uri2'");
  want_artifact2.set_type_id(type_id);
  int64_t artifact2_id;
  ASSERT_EQ(
      metadata_access_object_->CreateArtifact(want_artifact2, &artifact2_id),
      absl::OkStatus());

  int64_t type2_id = InsertType<ArtifactType>("test_type2");
  Artifact artifact3;
  artifact3.set_type_id(type2_id);
  int64_t artifact3_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact3, &artifact3_id),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  std::vector<Artifact> got_artifacts;
  EXPECT_EQ(metadata_access_object_->FindArtifactsByTypeId(
                type_id, absl::nullopt, &got_artifacts, nullptr),
            absl::OkStatus());
  EXPECT_EQ(got_artifacts.size(), 2);
  // Should perform unordered elements check if list option is not specified and
  // passed to `FindArtifactsByTypeId`
  EXPECT_THAT(
      got_artifacts,
      UnorderedElementsAre(
          EqualsProto(
              want_artifact1,
              /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                 "last_update_time_since_epoch"}),
          EqualsProto(want_artifact2, /*ignore_fields=*/{
                          "id", "type", "create_time_since_epoch",
                          "last_update_time_since_epoch"})));
}

TEST_P(MetadataAccessObjectTest, FindArtifactsByTypeIdsFilterPropertyQuery) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t type1_id = InsertType<ArtifactType>("test_type1");
  Artifact artifact1 = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri1'
    custom_properties {
      key: 'custom_property_1'
      value: { int_value: 1 }
    }
  )pb");
  artifact1.set_type_id(type1_id);
  int64_t artifact1_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact1, &artifact1_id),
            absl::OkStatus());

  Artifact artifact2 = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri1'
    custom_properties {
      key: 'custom_property_1'
      value: { int_value: 2 }
    }
  )pb");
  artifact2.set_type_id(type1_id);
  int64_t artifact2_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact2, &artifact2_id),
            absl::OkStatus());

  int64_t type2_id = InsertType<ArtifactType>("test_type2");
  Artifact artifact3 = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri1'
    custom_properties {
      key: 'custom_property_1'
      value: { int_value: 3 }
    }
  )pb");
  artifact3.set_type_id(type2_id);
  int64_t artifact3_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact3, &artifact3_id),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 10,
        order_by_field: { field: CREATE_TIME is_asc: true }
        filter_query: "custom_properties.custom_property_1.int_value = 1 OR "
                      "custom_properties.custom_property_1.int_value = 2 OR "
                      "custom_properties.custom_property_1.int_value = 3"
      )pb");

  std::vector<Artifact> got_artifacts;
  std::string next_page_token;
  ASSERT_EQ(metadata_access_object_->FindArtifactsByTypeId(
                type1_id, absl::make_optional(list_options), &got_artifacts,
                &next_page_token),
            absl::OkStatus());
  ASSERT_THAT(got_artifacts, SizeIs(2));
  // Given `list_options`, the result artifacts should sorted in ascending order
  // in terms of create_time.
  EXPECT_THAT(
      got_artifacts,
      ElementsAre(
          EqualsProto(
              artifact1,
              /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                 "last_update_time_since_epoch"}),
          EqualsProto(
              artifact2,
              /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                 "last_update_time_since_epoch"})));

  got_artifacts.clear();
  ASSERT_EQ(metadata_access_object_->FindArtifactsByTypeId(
                type2_id, absl::make_optional(list_options), &got_artifacts,
                &next_page_token),
            absl::OkStatus());
  ASSERT_THAT(got_artifacts, SizeIs(1));
  EXPECT_THAT(artifact3,
              EqualsProto(got_artifacts[0], /*ignore_fields=*/{
                              "id", "type", "create_time_since_epoch",
                              "last_update_time_since_epoch"}));
}

TEST_P(MetadataAccessObjectTest, FindArtifactByTypeIdAndArtifactName) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t type_id = InsertType<ArtifactType>("test_type");
  Artifact want_artifact = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri1'
    name: 'artifact1')pb");
  want_artifact.set_type_id(type_id);
  int64_t artifact1_id;
  ASSERT_EQ(
      metadata_access_object_->CreateArtifact(want_artifact, &artifact1_id),
      absl::OkStatus());
  want_artifact.set_id(artifact1_id);

  Artifact artifact2 = ParseTextProtoOrDie<Artifact>(
      "uri: 'testuri://testing/uri2' name: 'artifact2'");
  artifact2.set_type_id(type_id);
  int64_t artifact2_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact2, &artifact2_id),
            absl::OkStatus());

  int64_t type2_id = InsertType<ArtifactType>("test_type2");
  Artifact artifact3;
  artifact3.set_type_id(type2_id);
  int64_t artifact3_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact3, &artifact3_id),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact;
  EXPECT_EQ(metadata_access_object_->FindArtifactByTypeIdAndArtifactName(
                type_id, "artifact1", &got_artifact),
            absl::OkStatus());
  EXPECT_THAT(want_artifact, EqualsProto(got_artifact, /*ignore_fields=*/{
                                             "type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  Artifact got_empty_artifact;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindArtifactByTypeIdAndArtifactName(
          type_id, "unknown", &got_empty_artifact)));
  EXPECT_THAT(got_empty_artifact, EqualsProto(Artifact()));
}

TEST_P(MetadataAccessObjectTest, FindArtifactsByURI) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t type_id = InsertType<ArtifactType>("test_type");
  Artifact want_artifact1 = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri1'
    name: 'artifact1')pb");
  want_artifact1.set_type_id(type_id);
  int64_t artifact1_id;
  ASSERT_EQ(
      metadata_access_object_->CreateArtifact(want_artifact1, &artifact1_id),
      absl::OkStatus());
  want_artifact1.set_id(artifact1_id);

  Artifact artifact2 = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri2'
    name: 'artifact2')pb");
  artifact2.set_type_id(type_id);
  int64_t artifact2_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact2, &artifact2_id),
            absl::OkStatus());
  artifact2.set_id(artifact2_id);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  std::vector<Artifact> got_artifacts;
  EXPECT_EQ(metadata_access_object_->FindArtifactsByURI(
                "testuri://testing/uri1", &got_artifacts),
            absl::OkStatus());
  ASSERT_EQ(got_artifacts.size(), 1);
  EXPECT_THAT(want_artifact1, EqualsProto(got_artifacts[0], /*ignore_fields=*/{
                                              "type", "create_time_since_epoch",
                                              "last_update_time_since_epoch"}));
}

TEST_P(MetadataAccessObjectTest, FindArtifactsByExternalIds) {
  if (SkipIfEarlierSchemaLessThan(/*min_schema_version=*/9)) {
    return;
  }
  MLMD_ASSERT_OK(Init());
  // Setup: Create two arifacts with external_id.
  const string test_type_name = "test_type";
  int64_t type_id = InsertType<ArtifactType>(test_type_name);
  std::vector<absl::string_view> external_ids = {"artifact1", "artifact2"};
  Artifact artifact1 = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri1'
    name: 'artifact1')pb");
  artifact1.set_type_id(type_id);
  artifact1.set_external_id("artifact1");
  int64_t artifact1_id;
  MLMD_ASSERT_OK(
      metadata_access_object_->CreateArtifact(artifact1, &artifact1_id));
  artifact1.set_id(artifact1_id);

  Artifact artifact2 = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri2'
    name: 'artifact2')pb");
  artifact2.set_type_id(type_id);
  artifact2.set_external_id("artifact2");
  int64_t artifact2_id;
  MLMD_ASSERT_OK(
      metadata_access_object_->CreateArtifact(artifact2, &artifact2_id));
  artifact2.set_id(artifact2_id);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  artifact1.set_type(test_type_name);
  artifact2.set_type(test_type_name);

  // Test 1: artifacts can be retrieved by external_ids.
  std::vector<Artifact> got_artifacts_from_external_ids;
  EXPECT_EQ(
      absl::OkStatus(),
      metadata_access_object_->FindArtifactsByExternalIds(
          absl::MakeSpan(external_ids), &got_artifacts_from_external_ids));
  ASSERT_EQ(got_artifacts_from_external_ids.size(), 2);
  EXPECT_THAT(
      got_artifacts_from_external_ids,
      UnorderedElementsAre(
          EqualsProto(artifact1,
                      /*ignore_fields=*/{"type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"}),
          EqualsProto(artifact2,
                      /*ignore_fields=*/{"type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"})));

  // Test 2: will return NOT_FOUND error when finding artifacts by non-exsiting
  // external_id.
  std::vector<absl::string_view> external_ids_absent = {"artifact_absent"};
  std::vector<Artifact> got_artifacts_by_external_ids_absent;
  EXPECT_TRUE(
      absl::IsNotFound(metadata_access_object_->FindArtifactsByExternalIds(
          absl::MakeSpan(external_ids_absent),
          &got_artifacts_by_external_ids_absent)));

  // Test 3: will return whatever found when a part of external_ids is
  // non-existing, e.g., the input vector is {"artifact_absent", "artifact1"}.
  external_ids_absent.push_back(absl::string_view("artifact1"));
  EXPECT_EQ(metadata_access_object_->FindArtifactsByExternalIds(
                absl::MakeSpan(external_ids_absent),
                &got_artifacts_by_external_ids_absent),
            absl::OkStatus());
  ASSERT_EQ(got_artifacts_by_external_ids_absent.size(), 1);
  EXPECT_THAT(got_artifacts_by_external_ids_absent,
              UnorderedElementsAre(EqualsProto(
                  artifact1,
                  /*ignore_fields=*/{"type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"})));

  // Test 4: will return INVALID_ARGUMENT error when any of the external_ids is
  // empty.
  std::vector<absl::string_view> external_ids_empty = {""};
  std::vector<Artifact> got_artifacts_by_empty_external_id;
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->FindArtifactsByExternalIds(
          absl::MakeSpan(external_ids_empty),
          &got_artifacts_by_empty_external_id)));
}

TEST_P(MetadataAccessObjectTest, UpdateArtifact) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(
      absl::StrCat(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties { key: 'property_4' value: PROTO }
                     properties { key: 'property_5' value: BOOLEAN }
                                        )pb",
                   ""));
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
      value: { int_value: 5 }
    }
    state: LIVE
  )pb");
  stored_artifact.set_type_id(type_id);
  int64_t artifact_id;
  ASSERT_EQ(
      metadata_access_object_->CreateArtifact(stored_artifact, &artifact_id),
      absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_before_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_before_update = artifacts.at(0);
  }
  EXPECT_THAT(
      got_artifact_before_update,
      EqualsProto(stored_artifact,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));

  // update `property_1`, add `property_2`, and drop `property_3`
  // change the value type of `custom_property_1`
  // If schema version is at least 10, add `property_4` and `property_5`
  // Update `uri` and `state`
  Artifact updated_artifact = ParseTextProtoOrDie<Artifact>(
      absl::StrCat(R"(
    uri: 'testuri://changed/uri'
    properties {
      key: 'property_1'
      value: { int_value: 5 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
  )",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"(
    properties {
      key: 'property_4'
      value {
        proto_value {
          [type.googleapis.com/ml_metadata.testing.MockProto] {
            string_value: '3'
            double_value: 3.0
          }
        }
      }
    }
    properties {
      key: 'property_5'
      value { bool_value: true }
    }
  )",
                   R"(
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '3' }
    }
  )"));
  updated_artifact.set_id(artifact_id);
  updated_artifact.clear_type_id();

  // sleep to verify the latest update time is updated.
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(metadata_access_object_->UpdateArtifact(updated_artifact),
            absl::OkStatus());
  updated_artifact.set_type_id(type_id);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_after_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_after_update = artifacts.at(0);
  }
  EXPECT_THAT(got_artifact_after_update,
              EqualsProto(updated_artifact,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_artifact_before_update.create_time_since_epoch(),
            got_artifact_after_update.create_time_since_epoch());
  EXPECT_LT(got_artifact_before_update.last_update_time_since_epoch(),
            got_artifact_after_update.last_update_time_since_epoch());

  if (!IfSchemaLessThan(/*schema_version=*/9)) {
    EXPECT_TRUE(got_artifact_before_update.external_id().empty());
    updated_artifact.set_external_id("artifact_1");
    EXPECT_EQ(metadata_access_object_->UpdateArtifact(updated_artifact),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_after_update = artifacts.at(0);
    EXPECT_EQ(got_artifact_after_update.external_id(), "artifact_1");
  }
}

TEST_P(MetadataAccessObjectTest, UpdateArtifactWithCustomUpdateTime) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(
      absl::StrCat(R"pb(
                     name: 'test_type'
                     properties { key: 'property_1' value: INT }
                     properties { key: 'property_2' value: DOUBLE }
                     properties { key: 'property_3' value: STRING }
                   )pb",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties { key: 'property_4' value: PROTO }
                     properties { key: 'property_5' value: BOOLEAN }
                                        )pb",
                   ""));
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact stored_artifact = ParseTextProtoOrDie<Artifact>(
      absl::StrCat(R"pb(
                     uri: 'testuri://testing/uri'
                     properties {
                       key: 'property_1'
                       value: { int_value: 3 }
                     }
                     properties {
                       key: 'property_3'
                       value: { string_value: '3' }
                     }
                   )pb",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties {
                       key: 'property_4'
                       value {
                         proto_value {
                           [type.googleapis.com/ml_metadata.testing.MockProto] {
                             string_value: '3'
                             double_value: 3.0
                           }
                         }
                       }
                     }
                     properties {
                       key: 'property_5'
                       value { bool_value: true }
                     }
                                        )pb",
                   R"pb(
                     custom_properties {
                       key: 'custom_property_1'
                       value: { string_value: '5' }
                     }
                     state: LIVE
                   )pb"));
  stored_artifact.set_type_id(type_id);
  int64_t artifact_id;
  ASSERT_EQ(
      metadata_access_object_->CreateArtifact(stored_artifact, &artifact_id),
      absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_before_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_before_update = artifacts.at(0);
  }
  EXPECT_THAT(
      got_artifact_before_update,
      EqualsProto(stored_artifact,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));

  // update `property_1`, add `property_2`, and drop `property_3`
  // change the value type of `custom_property_1`
  Artifact updated_artifact = ParseTextProtoOrDie<Artifact>(
      absl::StrCat(R"pb(
                     uri: 'testuri://changed/uri'
                     properties {
                       key: 'property_1'
                       value: { int_value: 5 }
                     }
                     properties {
                       key: 'property_2'
                       value: { double_value: 3.0 }
                     }
                   )pb",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties {
                       key: 'property_4'
                       value {
                         proto_value {
                           [type.googleapis.com/ml_metadata.testing.MockProto] {
                             string_value: '1'
                             double_value: 1.0
                           }
                         }
                       }
                     }
                     properties {
                       key: 'property_5'
                       value { bool_value: true }
                     }
                                        )pb",
                   R"pb(
                     custom_properties {
                       key: 'custom_property_1'
                       value: { int_value: 3 }
                     }
                   )pb"));
  updated_artifact.set_id(artifact_id);
  updated_artifact.set_type_id(type_id);
  absl::Time update_time = absl::InfiniteFuture();
  ASSERT_EQ(metadata_access_object_->UpdateArtifact(
                updated_artifact, update_time, /*force_update_time=*/false),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_after_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_after_update = artifacts.at(0);
  }
  EXPECT_THAT(got_artifact_after_update,
              EqualsProto(updated_artifact,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_artifact_before_update.create_time_since_epoch(),
            got_artifact_after_update.create_time_since_epoch());
  EXPECT_EQ(got_artifact_after_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, UpdateArtifactWithForceUpdateTimeEnabled) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t artifact_id;
  ASSERT_EQ(
      metadata_access_object_->CreateArtifact(stored_artifact, &artifact_id),
      absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_before_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_before_update = artifacts.at(0);
  }
  EXPECT_THAT(
      got_artifact_before_update,
      EqualsProto(stored_artifact,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));

  // Update with no changes and force_update_time disabled.
  absl::Time update_time = absl::InfiniteFuture();
  ASSERT_EQ(metadata_access_object_->UpdateArtifact(
                got_artifact_before_update, update_time,
                /*force_update_time=*/false),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_after_1st_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_after_1st_update = artifacts.at(0);
  }
  // Expect no changes for the updated resource.
  EXPECT_THAT(got_artifact_after_1st_update,
              EqualsProto(got_artifact_before_update));

  // Update with no changes again but with force_update_time set to true.
  ASSERT_EQ(metadata_access_object_->UpdateArtifact(
                got_artifact_after_1st_update, update_time,
                /*force_update_time=*/true),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_after_2nd_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_after_2nd_update = artifacts.at(0);
  }
  // Expect no changes for the updated resource other than
  // `last_update_time_since_epoch`.
  EXPECT_THAT(
      got_artifact_after_2nd_update,
      EqualsProto(got_artifact_after_1st_update,
                  /*ignore_fields=*/{"type", "last_update_time_since_epoch"}));
  EXPECT_NE(got_artifact_after_2nd_update.last_update_time_since_epoch(),
            got_artifact_after_1st_update.last_update_time_since_epoch());
  EXPECT_EQ(got_artifact_after_2nd_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, UpdateArtifactWithMasking) {
  // Set up
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(
      absl::StrCat(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties { key: 'property_4' value: PROTO }
                     properties { key: 'property_5' value: BOOLEAN }
                                        )pb",
                   ""));
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
      value: { int_value: 5 }
    }
    custom_properties {
      key: 'custom_property_2'
      value: { int_value: 5 }
    }
    state: LIVE
  )pb");
  stored_artifact.set_type_id(type_id);
  int64_t artifact_id;
  ASSERT_EQ(
      metadata_access_object_->CreateArtifact(stored_artifact, &artifact_id),
      absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_before_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_before_update = artifacts.at(0);
  }
  EXPECT_THAT(
      got_artifact_before_update,
      EqualsProto(stored_artifact,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));

  // Test 1:
  // Update `property_1`, add `property_2`, and drop `property_3`
  // Change the value type of `custom_property_1`, drop `custom_property_2`
  // If schema version is at least 10, add `property_4` and `property_5`
  // Update `uri`
  // Add "invalid_path" in mask, which will be ignored during update.
  google::protobuf::FieldMask mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(R"pb(
        paths: 'uri'
        paths: 'properties.property_1'
        paths: 'properties.property_2'
        paths: 'properties.property_3'
        paths: 'custom_properties.custom_property_1'
        paths: 'custom_properties.custom_property_2'
        paths: 'invalid_path'
      )pb");
  if (!IfSchemaLessThan(10)) {
    mask.add_paths("properties.property_4");
    mask.add_paths("properties.property_5");
  }

  Artifact updated_artifact =
      ParseTextProtoOrDie<Artifact>(absl::StrCat(R"(
    uri: 'testuri://changed/uri'
    properties {
      key: 'property_1'
      value: { int_value: 5 }
    }
    properties {
      key: 'property_2'
      value: { double_value: 3.0 }
    }
  )",
                                                 IfSchemaLessThan(10) ? "" :
                                                                      R"(
    properties {
      key: 'property_4'
      value {
        proto_value {
          [type.googleapis.com/ml_metadata.testing.MockProto] {
            string_value: '3'
            double_value: 3.0
          }
        }
      }
    }
    properties {
      key: 'property_5'
      value { bool_value: true }
    }
  )",
                                                 R"(
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '3' }
    }
  )"));
  updated_artifact.set_id(artifact_id);
  updated_artifact.set_type_id(type_id);
  updated_artifact.set_state(Artifact::LIVE);

  {
    // sleep to verify the latest update time is updated.
    absl::SleepFor(absl::Milliseconds(1));
    EXPECT_EQ(metadata_access_object_->UpdateArtifact(updated_artifact, mask),
              absl::OkStatus());

    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

    Artifact got_artifact_after_update;
    {
      std::vector<Artifact> artifacts;
      ASSERT_EQ(
          metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
          absl::OkStatus());
      got_artifact_after_update = artifacts.at(0);
    }
    EXPECT_THAT(
        got_artifact_after_update,
        EqualsProto(updated_artifact,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
    EXPECT_EQ(got_artifact_before_update.create_time_since_epoch(),
              got_artifact_after_update.create_time_since_epoch());
    EXPECT_LT(got_artifact_before_update.last_update_time_since_epoch(),
              got_artifact_after_update.last_update_time_since_epoch());
  }

  // Test 2:
  // Update 'property_1' and delete 'property_2'.
  // Add both "properties" and "properties.property_1" in mask, this means
  // making for "properties.property_1" is ignored and `properties` will be
  // updated as a whole.
  updated_artifact.mutable_properties()->at("property_1").set_int_value(6);
  updated_artifact.mutable_properties()->erase("property_2");
  {
    // sleep to verify the latest update time is updated.
    absl::SleepFor(absl::Milliseconds(1));
    mask.Clear();
    // Add field path: "properties" to mask, which means diffing and updating
    // `properties` as a whole.
    mask.add_paths("properties");
    // Add field path: "properties" to mask, which means diffing and updating
    // `custom_properties` as a whole.
    mask.add_paths("custom_properties");
    // "properties.property_1" will be ignored in the mask during update.
    mask.add_paths("properties.property_1");
    // Add an empty path to mask.
    mask.add_paths("");
    EXPECT_EQ(metadata_access_object_->UpdateArtifact(updated_artifact, mask),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    Artifact got_artifact_after_update;
    {
      std::vector<Artifact> artifacts;
      ASSERT_EQ(
          metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
          absl::OkStatus());
      got_artifact_after_update = artifacts.at(0);
    }
    EXPECT_THAT(
        got_artifact_after_update,
        EqualsProto(updated_artifact,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
    EXPECT_EQ(got_artifact_before_update.create_time_since_epoch(),
              got_artifact_after_update.create_time_since_epoch());
    EXPECT_LT(got_artifact_before_update.last_update_time_since_epoch(),
              got_artifact_after_update.last_update_time_since_epoch());
  }

  // Test 3:
  // Set `external_id`
  if (!IfSchemaLessThan(/*schema_version=*/9)) {
    EXPECT_TRUE(got_artifact_before_update.external_id().empty());
    mask.Clear();
    mask.add_paths("external_id");
    updated_artifact.set_external_id("artifact_1");
    EXPECT_EQ(metadata_access_object_->UpdateArtifact(updated_artifact, mask),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    EXPECT_EQ(artifacts.at(0).external_id(), "artifact_1");
  }

  // Test 4:
  // Update with an empty mask, which means the artifact will be updated as a
  // whole.
  {
    mask.Clear();
    updated_artifact.mutable_properties()->at("property_1").set_int_value(9);
    EXPECT_EQ(metadata_access_object_->UpdateArtifact(updated_artifact, mask),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
    Artifact got_artifact_after_update;
    {
      std::vector<Artifact> artifacts;
      ASSERT_EQ(
          metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
          absl::OkStatus());
      got_artifact_after_update = artifacts.at(0);
    }
    EXPECT_THAT(
        got_artifact_after_update,
        EqualsProto(updated_artifact,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
  }
}

TEST_P(MetadataAccessObjectTest, UpdateArtifactWithMaskingError) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact artifact = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
  )pb");
  artifact.set_type_id(type_id);
  int64_t artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());
  artifact.set_id(artifact_id);
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  google::protobuf::FieldMask mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(R"pb(
        paths: 'uri'
        paths: 'properties.property_1'
      )pb");
  // no artifact id given
  Artifact wrong_artifact;
  absl::Status s =
      metadata_access_object_->UpdateArtifact(wrong_artifact, mask);
  EXPECT_TRUE(absl::IsInvalidArgument(s));

  // artifact id cannot be found
  int64_t different_id = artifact_id + 1;
  wrong_artifact.set_id(different_id);
  s = metadata_access_object_->UpdateArtifact(wrong_artifact, mask);
  EXPECT_TRUE(absl::IsInvalidArgument(s));

  // type_id if given is not aligned with the stored one
  wrong_artifact.set_id(artifact_id);
  int64_t different_type_id = type_id + 1;
  wrong_artifact.set_type_id(different_type_id);
  s = metadata_access_object_->UpdateArtifact(wrong_artifact, mask);
  EXPECT_TRUE(absl::IsInvalidArgument(s));

  // artifact has unknown property
  mask.add_paths("properties.unknown_property");
  wrong_artifact.clear_type_id();
  (*wrong_artifact.mutable_properties())["unknown_property"].set_int_value(1);
  s = metadata_access_object_->UpdateArtifact(wrong_artifact, mask);
  EXPECT_TRUE(absl::IsInvalidArgument(s));
}

TEST_P(MetadataAccessObjectTest, UpdateArtifactWithCustomUpdateTimeAndMasking) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(
      absl::StrCat(R"pb(
                     name: 'test_type'
                     properties { key: 'property_1' value: INT }
                     properties { key: 'property_2' value: DOUBLE }
                     properties { key: 'property_3' value: STRING }
                   )pb",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties { key: 'property_4' value: PROTO }
                     properties { key: 'property_5' value: BOOLEAN }
                                        )pb",
                   ""));
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact stored_artifact = ParseTextProtoOrDie<Artifact>(
      absl::StrCat(R"pb(
                     uri: 'testuri://testing/uri'
                     properties {
                       key: 'property_1'
                       value: { int_value: 3 }
                     }
                     properties {
                       key: 'property_3'
                       value: { string_value: '3' }
                     }
                   )pb",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties {
                       key: 'property_4'
                       value {
                         proto_value {
                           [type.googleapis.com/ml_metadata.testing.MockProto] {
                             string_value: '3'
                             double_value: 3.0
                           }
                         }
                       }
                     }
                     properties {
                       key: 'property_5'
                       value { bool_value: true }
                     }
                                        )pb",
                   R"pb(
                     custom_properties {
                       key: 'custom_property_1'
                       value: { string_value: '5' }
                     }
                     state: LIVE
                   )pb"));
  stored_artifact.set_type_id(type_id);
  int64_t artifact_id;
  ASSERT_EQ(
      metadata_access_object_->CreateArtifact(stored_artifact, &artifact_id),
      absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_before_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_before_update = artifacts.at(0);
  }
  EXPECT_THAT(
      got_artifact_before_update,
      EqualsProto(stored_artifact,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));

  // Update `property_1`, add `property_2`, and drop `property_3`
  // Change the value type of `custom_property_1`
  // If schema version is at least 10, update `property_4`
  // Update `uri`
  google::protobuf::FieldMask mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(R"pb(
        paths: 'uri'
        paths: 'properties.property_1'
        paths: 'properties.property_2'
        paths: 'properties.property_3'
        paths: 'custom_properties.custom_property_1'
      )pb");
  if (!IfSchemaLessThan(10)) {
    mask.add_paths("properties.property_4");
  }
  Artifact updated_artifact = ParseTextProtoOrDie<Artifact>(
      absl::StrCat(R"pb(
                     uri: 'testuri://changed/uri'
                     properties {
                       key: 'property_1'
                       value: { int_value: 5 }
                     }
                     properties {
                       key: 'property_2'
                       value: { double_value: 3.0 }
                     }
                   )pb",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties {
                       key: 'property_4'
                       value {
                         proto_value {
                           [type.googleapis.com/ml_metadata.testing.MockProto] {
                             string_value: '1'
                             double_value: 1.0
                           }
                         }
                       }
                     }
                     properties {
                       key: 'property_5'
                       value { bool_value: false }
                     }
                                        )pb",
                   R"pb(
                     custom_properties {
                       key: 'custom_property_1'
                       value: { int_value: 3 }
                     }
                     state: LIVE
                   )pb"));
  updated_artifact.set_id(artifact_id);
  updated_artifact.set_type_id(type_id);
  absl::Time update_time = absl::InfiniteFuture();
  ASSERT_EQ(metadata_access_object_->UpdateArtifact(
                updated_artifact, update_time,
                /*force_update_time=*/false, mask),
            absl::OkStatus());
  if (!IfSchemaLessThan(10)) {
    // Change `property_5` back to its stored bool value before comparing
    // protos.
    updated_artifact.mutable_properties()
        ->at("property_5")
        .set_bool_value(true);
  }

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_after_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_after_update = artifacts.at(0);
  }
  EXPECT_THAT(got_artifact_after_update,
              EqualsProto(updated_artifact,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_artifact_before_update.create_time_since_epoch(),
            got_artifact_after_update.create_time_since_epoch());
  EXPECT_EQ(got_artifact_after_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest,
       UpdateArtifactWithForceUpdateTimeEnabledAndMasking) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t artifact_id;
  ASSERT_EQ(
      metadata_access_object_->CreateArtifact(stored_artifact, &artifact_id),
      absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_before_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_before_update = artifacts.at(0);
  }
  EXPECT_THAT(
      got_artifact_before_update,
      EqualsProto(stored_artifact,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));

  // Update with no changes and force_update_time disabled.
  google::protobuf::FieldMask mask;
  mask.add_paths("name");
  mask.add_paths("external_id");
  mask.add_paths("");
  absl::Time update_time = absl::InfiniteFuture();
  ASSERT_EQ(metadata_access_object_->UpdateArtifact(
                got_artifact_before_update, update_time,
                /*force_update_time=*/false, mask),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_after_1st_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_after_1st_update = artifacts.at(0);
  }
  // Expect no changes for the updated resource.
  EXPECT_THAT(got_artifact_after_1st_update,
              EqualsProto(got_artifact_before_update));

  got_artifact_after_1st_update.set_uri("testuri://testing/new_uri");
  ASSERT_EQ(metadata_access_object_->UpdateArtifact(
                got_artifact_after_1st_update, update_time,
                /*force_update_time=*/true, google::protobuf::FieldMask()),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  got_artifact_after_1st_update.mutable_properties()
      ->at("property_1")
      .set_int_value(6);
  got_artifact_after_1st_update.set_state(Artifact::MARKED_FOR_DELETION);
  got_artifact_after_1st_update.set_uri("invalid_uri");
  mask.clear_paths();
  mask.add_paths("properties.property_1");
  mask.add_paths("state");
  ASSERT_EQ(metadata_access_object_->UpdateArtifact(
                got_artifact_after_1st_update, update_time,
                /*force_update_time=*/true, mask),
            absl::OkStatus());
  // Set back uri value for later comparison
  got_artifact_after_1st_update.set_uri("testuri://testing/new_uri");

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact got_artifact_after_2nd_update;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    got_artifact_after_2nd_update = artifacts.at(0);
  }

  EXPECT_THAT(
      got_artifact_after_2nd_update,
      EqualsProto(got_artifact_after_1st_update,
                  /*ignore_fields=*/{"type", "last_update_time_since_epoch"}));
  EXPECT_NE(got_artifact_after_2nd_update.last_update_time_since_epoch(),
            got_artifact_after_1st_update.last_update_time_since_epoch());
  EXPECT_EQ(got_artifact_after_2nd_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, UpdateNodeLastUpdateTimeSinceEpoch) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'p1' value: INT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  // Create the original artifact before update.
  Artifact artifact;
  artifact.set_uri("testuri://changed/uri");
  artifact.set_type_id(type_id);
  int64_t artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact curr_artifact;
  {
    std::vector<Artifact> artifacts;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsById({artifact_id}, &artifacts),
        absl::OkStatus());
    curr_artifact = artifacts.at(0);
  }

  // insert executions and links to artifacts
  auto update_and_get_last_update_time_since_epoch =
      [&](const Artifact& artifact) {
        // sleep to verify the latest update time is updated.
        absl::SleepFor(absl::Milliseconds(1));
        CHECK_EQ(metadata_access_object_->UpdateArtifact(artifact),
                 absl::OkStatus());
        CHECK_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
        Artifact got_artifact_after_update;
        {
          std::vector<Artifact> artifacts;
          CHECK_EQ(metadata_access_object_->FindArtifactsById({artifact.id()},
                                                              &artifacts),
                   absl::OkStatus());
          got_artifact_after_update = artifacts.at(0);
        }
        EXPECT_THAT(got_artifact_after_update,
                    EqualsProto(artifact, /*ignore_fields=*/{
                                    "type", "last_update_time_since_epoch"}));
        return got_artifact_after_update.last_update_time_since_epoch();
      };

  // no attribute or property change.
  const int64_t t0 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_EQ(t0, curr_artifact.last_update_time_since_epoch());
  // update attributes
  curr_artifact.set_uri("new/uri");
  const int64_t t1 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t1, t0);
  // set attributes
  curr_artifact.set_state(Artifact::LIVE);
  const int64_t t2 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t2, t1);
  // add property
  (*curr_artifact.mutable_properties())["p1"].set_int_value(1);
  const int64_t t3 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t3, t2);
  // modify property
  (*curr_artifact.mutable_properties())["p1"].set_int_value(2);
  const int64_t t4 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t4, t3);
  // delete property
  curr_artifact.clear_properties();
  const int64_t t5 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t5, t4);
  // set custom property
  (*curr_artifact.mutable_custom_properties())["custom"].set_string_value("1");
  const int64_t t6 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t6, t5);
  // modify custom property
  (*curr_artifact.mutable_custom_properties())["custom"].set_string_value("2");
  const int64_t t7 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t7, t6);
  // delete custom property
  curr_artifact.clear_custom_properties();
  const int64_t t8 = update_and_get_last_update_time_since_epoch(curr_artifact);
  EXPECT_GT(t8, t7);
}

TEST_P(MetadataAccessObjectTest, UpdateArtifactError) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Artifact artifact = ParseTextProtoOrDie<Artifact>(R"pb(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
  )pb");
  artifact.set_type_id(type_id);
  int64_t artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());
  artifact.set_id(artifact_id);
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // no artifact id given
  Artifact wrong_artifact;
  absl::Status s = metadata_access_object_->UpdateArtifact(wrong_artifact);
  EXPECT_TRUE(absl::IsInvalidArgument(s));

  // artifact id cannot be found
  int64_t different_id = artifact_id + 1;
  wrong_artifact.set_id(different_id);
  s = metadata_access_object_->UpdateArtifact(wrong_artifact);
  EXPECT_TRUE(absl::IsInvalidArgument(s));

  // type_id if given is not aligned with the stored one
  wrong_artifact.set_id(artifact_id);
  int64_t different_type_id = type_id + 1;
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
  ASSERT_EQ(Init(), absl::OkStatus());
  // Creates execution 1 with type 1
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(
      absl::StrCat(R"(
    name: 'test_type_with_predefined_property'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )",
                   // TODO(b/257334039): cleanup fat client
                   IfSchemaLessThan(10) ? "" :
                                        R"pb(
                     properties { key: 'property_4' value: PROTO }
                     properties { key: 'property_5' value: BOOLEAN }
                                        )pb",
                   ""));
  const string test_type_name_with_no_property = "test_type_with_no_property";
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution want_execution1 =
      ParseTextProtoOrDie<Execution>(absl::StrCat(R"(
    name: "my_execution1"
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
  )",  // TODO(b/257334039): cleanup fat client
                                                  IfSchemaLessThan(10) ? "" :
                                                                       R"(
    properties {
      key: 'property_4'
      value {
        proto_value {
          [type.googleapis.com/ml_metadata.testing.MockProto] {
            string_value: '3'
            double_value: 3.0
          }
        }
      }
    }
    properties {
      key: 'property_5'
      value { bool_value: true }
    }
  )",
                                                  R"(
    custom_properties {
      key: 'custom_property_1'
      value: { int_value: 3 }
    }
  )"));
  want_execution1.set_type_id(type_id);
  {
    int64_t execution_id = -1;
    ASSERT_EQ(metadata_access_object_->CreateExecution(want_execution1,
                                                       &execution_id),
              absl::OkStatus());
    want_execution1.set_id(execution_id);
  }
  // Creates execution 2 with type 2
  int64_t type2_id = InsertType<ExecutionType>(test_type_name_with_no_property);
  Execution want_execution2 = ParseTextProtoOrDie<Execution>(R"pb(
    name: "my_execution2"
    last_known_state: RUNNING
  )pb");
  want_execution2.set_type_id(type2_id);
  {
    int64_t execution_id = -1;
    ASSERT_EQ(metadata_access_object_->CreateExecution(want_execution2,
                                                       &execution_id),
              absl::OkStatus());
    want_execution2.set_id(execution_id);
  }

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: update external_id, which also prepares for retrieving by external_id
  if (!IfSchemaLessThan(/*schema_version=*/9)) {
    ASSERT_TRUE(want_execution1.external_id().empty());
    ASSERT_TRUE(want_execution2.external_id().empty());

    want_execution1.set_external_id("my_execution1");
    want_execution2.set_external_id("my_execution2");

    Execution want_execution1_after_update, want_execution2_after_update;
    UpdateAndReturnNode<Execution>(want_execution1, *metadata_access_object_,
                                   metadata_access_object_container_.get(),
                                   want_execution1_after_update);
    UpdateAndReturnNode<Execution>(want_execution2, *metadata_access_object_,
                                   metadata_access_object_container_.get(),
                                   want_execution2_after_update);

    EXPECT_EQ(want_execution1_after_update.external_id(),
              want_execution1.external_id());
    EXPECT_EQ(want_execution2_after_update.external_id(),
              want_execution2.external_id());
  }

  EXPECT_NE(want_execution1.id(), want_execution2.id());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  want_execution1.set_type(type.name());
  want_execution2.set_type(test_type_name_with_no_property);
  // Test: retrieve one execution at a time
  Execution got_execution1;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById(
                  {want_execution1.id()}, &executions),
              absl::OkStatus());
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
    EXPECT_EQ(metadata_access_object_->FindExecutionsById(
                  {want_execution2.id()}, &executions),
              absl::OkStatus());
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
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({}, &executions),
              absl::OkStatus());
    EXPECT_THAT(executions, IsEmpty());
  }

  // Test: unknown id
  const int64_t unknown_id = want_execution1.id() + want_execution2.id() + 1;
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
    EXPECT_EQ(metadata_access_object_->FindExecutions(&got_executions),
              absl::OkStatus());
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
    EXPECT_EQ(metadata_access_object_->FindExecutionsByTypeId(
                  type_id, absl::nullopt, &type1_executions, nullptr),
              absl::OkStatus());
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

  if (!IfSchemaLessThan(/*schema_version=*/9)) {
    // Test: retrieve by external_id
    // Test 1: executions can be retrieved by external_ids.
    std::vector<Execution> got_executions_from_external_ids;
    std::vector<absl::string_view> external_ids = {"my_execution1",
                                                   "my_execution2"};
    MLMD_EXPECT_OK(metadata_access_object_->FindExecutionsByExternalIds(
        absl::MakeSpan(external_ids), &got_executions_from_external_ids));
    EXPECT_THAT(
        got_executions_from_external_ids,
        UnorderedElementsAre(
            EqualsProto(want_execution1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(want_execution2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));

    // Test 2: will return NOT_FOUND error when finding executions by
    // non-exsiting external_id.
    std::vector<absl::string_view> external_ids_absent = {
        "my_execution_absent"};
    std::vector<Execution> got_executions_from_external_ids_absent;
    EXPECT_TRUE(
        absl::IsNotFound(metadata_access_object_->FindExecutionsByExternalIds(
            absl::MakeSpan(external_ids_absent),
            &got_executions_from_external_ids_absent)));

    // Test 3: will return whatever found when a part of external_ids is
    // non-existing, e.g., the input error vector is {"my_execution1",
    // "my_execution_absent"}.
    external_ids_absent.push_back(absl::string_view("my_execution1"));
    MLMD_EXPECT_OK(metadata_access_object_->FindExecutionsByExternalIds(
        absl::MakeSpan(external_ids_absent),
        &got_executions_from_external_ids_absent));
    ASSERT_EQ(got_executions_from_external_ids_absent.size(), 1);
    EXPECT_THAT(
        got_executions_from_external_ids_absent,
        UnorderedElementsAre(EqualsProto(
            want_execution1, /*ignore_fields=*/{
                "create_time_since_epoch", "last_update_time_since_epoch"})));

    // Test 4: will return INVALID_ARGUMENT error when any of the external_ids
    // is empty.
    std::vector<absl::string_view> external_ids_empty = {""};
    std::vector<Execution> got_executions_from_empty_external_ids;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->FindExecutionsByExternalIds(
            absl::MakeSpan(external_ids_empty),
            &got_executions_from_empty_external_ids)));
  }
}

TEST_P(MetadataAccessObjectTest, CreateExecutionWithDuplicatedNameError) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType type;
  type.set_name("test_type");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution execution;
  execution.set_type_id(type_id);
  execution.set_name("test execution name");
  int64_t execution_id;
  EXPECT_EQ(metadata_access_object_->CreateExecution(execution, &execution_id),
            absl::OkStatus());
  // insert the same execution again to check the unique constraint
  absl::Status unique_constraint_violation_status =
      metadata_access_object_->CreateExecution(execution, &execution_id);
  EXPECT_EQ(CheckUniqueConstraintAndResetTransaction(
                unique_constraint_violation_status),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateExecutionWithDuplicatedExternalIdError) {
  if (SkipIfEarlierSchemaLessThan(/*min_schema_version=*/9)) {
    return;
  }
  MLMD_ASSERT_OK(Init());
  ExecutionType type;
  type.set_name("test_type");
  int64_t type_id;
  MLMD_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution execution;
  execution.set_type_id(type_id);
  execution.set_external_id("execution_1");
  int64_t execution_id;
  MLMD_EXPECT_OK(
      metadata_access_object_->CreateExecution(execution, &execution_id));

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Insert the same execution again to check the unique constraint
  absl::Status unique_constraint_violation_status =
      metadata_access_object_->CreateExecution(execution, &execution_id);
  MLMD_EXPECT_OK(CheckUniqueConstraintAndResetTransaction(
      unique_constraint_violation_status));
}

TEST_P(MetadataAccessObjectTest, UpdateExecution) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t execution_id;
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(stored_execution, &execution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_before_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_before_update = executions.at(0);
  }
  EXPECT_THAT(
      got_execution_before_update,
      EqualsProto(stored_execution,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
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
  updated_execution.clear_type_id();
  // sleep to verify the latest update time is updated.
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(metadata_access_object_->UpdateExecution(updated_execution),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  updated_execution.set_type_id(type_id);
  Execution got_execution_after_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_after_update = executions.at(0);
  }
  EXPECT_THAT(got_execution_after_update,
              EqualsProto(updated_execution,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_execution_before_update.create_time_since_epoch(),
            got_execution_after_update.create_time_since_epoch());
  EXPECT_LT(got_execution_before_update.last_update_time_since_epoch(),
            got_execution_after_update.last_update_time_since_epoch());
}

TEST_P(MetadataAccessObjectTest, UpdateExecutionWithCustomUpdateTime) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t execution_id;
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(stored_execution, &execution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_before_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_before_update = executions.at(0);
  }
  EXPECT_THAT(
      got_execution_before_update,
      EqualsProto(stored_execution,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
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
  ASSERT_EQ(metadata_access_object_->UpdateExecution(
                updated_execution, update_time, /*force_update_time=*/false),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_after_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_after_update = executions.at(0);
  }
  EXPECT_THAT(got_execution_after_update,
              EqualsProto(updated_execution,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_execution_before_update.create_time_since_epoch(),
            got_execution_after_update.create_time_since_epoch());
  EXPECT_EQ(got_execution_after_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, UpdateExecutionWithForceUpdateTimeEnabled) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t execution_id;
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(stored_execution, &execution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_before_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_before_update = executions.at(0);
  }
  EXPECT_THAT(
      got_execution_before_update,
      EqualsProto(stored_execution,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));
  // Update with no changes and force_update_time disabled.
  absl::Time update_time = absl::InfiniteFuture();
  ASSERT_EQ(metadata_access_object_->UpdateExecution(
                got_execution_before_update, update_time,
                /*force_update_time=*/false),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_after_1st_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_after_1st_update = executions.at(0);
  }
  // Expect no changes for the updated resource.
  EXPECT_THAT(got_execution_after_1st_update,
              EqualsProto(got_execution_before_update));

  // Update with no changes again but with force_update_time set to true.
  ASSERT_EQ(metadata_access_object_->UpdateExecution(
                got_execution_after_1st_update, update_time,
                /*force_update_time=*/true),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_after_2nd_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_after_2nd_update = executions.at(0);
  }
  // Expect no changes for the updated resource other than
  EXPECT_THAT(got_execution_after_2nd_update,
              EqualsProto(got_execution_after_1st_update,
                          /*ignore_fields=*/{"last_update_time_since_epoch"}));
  EXPECT_NE(got_execution_after_2nd_update.last_update_time_since_epoch(),
            got_execution_after_1st_update.last_update_time_since_epoch());
  EXPECT_EQ(got_execution_after_2nd_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, UpdateExecutionWithMasking) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t execution_id;
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(stored_execution, &execution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_before_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_before_update = executions.at(0);
  }
  EXPECT_THAT(
      got_execution_before_update,
      EqualsProto(stored_execution,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
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
  google::protobuf::FieldMask mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(R"pb(
        paths: 'properties.property_1'
        paths: 'properties.property_3'
        paths: 'custom_properties.custom_property_1'
      )pb");
  // sleep to verify the latest update time is updated.
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(metadata_access_object_->UpdateExecution(
                updated_execution, /*force_update_time=*/false, mask),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_after_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_after_update = executions.at(0);
  }
  updated_execution.set_last_known_state(stored_execution.last_known_state());
  EXPECT_THAT(got_execution_after_update,
              EqualsProto(updated_execution,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_execution_before_update.create_time_since_epoch(),
            got_execution_after_update.create_time_since_epoch());
  EXPECT_LT(got_execution_before_update.last_update_time_since_epoch(),
            got_execution_after_update.last_update_time_since_epoch());
}

TEST_P(MetadataAccessObjectTest,
       UpdateExecutionWithCustomUpdateTimeAndMasking) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t execution_id;
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(stored_execution, &execution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_before_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_before_update = executions.at(0);
  }
  EXPECT_THAT(
      got_execution_before_update,
      EqualsProto(stored_execution,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
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
  google::protobuf::FieldMask mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(R"pb(
        paths: 'properties.property_1'
        paths: 'properties.property_3'
        paths: 'custom_properties.custom_property_1'
      )pb");
  absl::Time update_time = absl::InfiniteFuture();
  ASSERT_EQ(
      metadata_access_object_->UpdateExecution(
          updated_execution, update_time, /*force_update_time=*/false, mask),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_after_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_after_update = executions.at(0);
  }
  updated_execution.set_last_known_state(stored_execution.last_known_state());
  EXPECT_THAT(got_execution_after_update,
              EqualsProto(updated_execution,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_execution_before_update.create_time_since_epoch(),
            got_execution_after_update.create_time_since_epoch());
  EXPECT_EQ(got_execution_after_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest,
       UpdateExecutionWithForceUpdateTimeEnabledAndMasking) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t execution_id;
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(stored_execution, &execution_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_before_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_before_update = executions.at(0);
  }
  EXPECT_THAT(
      got_execution_before_update,
      EqualsProto(stored_execution,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));
  google::protobuf::FieldMask mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(R"pb(
        paths: 'properties.property_1'
        paths: 'properties.property_3'
        paths: 'custom_properties.custom_property_1'
      )pb");
  // Update with no changes and force_update_time disabled.
  absl::Time update_time = absl::InfiniteFuture();
  ASSERT_EQ(metadata_access_object_->UpdateExecution(
                got_execution_before_update, update_time,
                /*force_update_time=*/false, mask),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_after_1st_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_after_1st_update = executions.at(0);
  }
  // Expect no changes for the updated resource.
  EXPECT_THAT(got_execution_after_1st_update,
              EqualsProto(got_execution_before_update));

  // Update with no changes again but with force_update_time set to true and
  // update_time default set to now.
  ASSERT_EQ(metadata_access_object_->UpdateExecution(
                got_execution_after_1st_update,
                /*force_update_time=*/true, mask),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_after_2nd_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_after_2nd_update = executions.at(0);
  }
  // Expect no changes for the updated resource other than last_update_time.
  // last_update_time should be no less than the 1st update time.
  EXPECT_THAT(got_execution_after_2nd_update,
              EqualsProto(got_execution_after_1st_update,
                          /*ignore_fields=*/{"last_update_time_since_epoch"}));
  EXPECT_GE(got_execution_after_2nd_update.last_update_time_since_epoch(),
            got_execution_after_1st_update.last_update_time_since_epoch());

  // Update with no changes again but with force_update_time set to true and
  // update_time set to infinite_future.
  ASSERT_EQ(metadata_access_object_->UpdateExecution(
                got_execution_after_1st_update, update_time,
                /*force_update_time=*/true, mask),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Execution got_execution_after_3nd_update;
  {
    std::vector<Execution> executions;
    EXPECT_EQ(metadata_access_object_->FindExecutionsById({execution_id},
                                                          &executions),
              absl::OkStatus());
    got_execution_after_3nd_update = executions.at(0);
  }
  // Expect no changes for the updated resource other than last_update_time.
  EXPECT_THAT(got_execution_after_3nd_update,
              EqualsProto(got_execution_after_1st_update,
                          /*ignore_fields=*/{"last_update_time_since_epoch"}));
  EXPECT_NE(got_execution_after_3nd_update.last_update_time_since_epoch(),
            got_execution_after_2nd_update.last_update_time_since_epoch());
  EXPECT_EQ(got_execution_after_3nd_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, CreateAndFindContext) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType type1 = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type_with_predefined_property'
    properties { key: 'property_1' value: INT }
  )pb");
  int64_t type1_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type1, &type1_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ContextType type2 = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type_with_no_property'
  )pb");
  int64_t type2_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type2, &type2_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Creates two contexts of different types
  Context context1 = ParseTextProtoOrDie<Context>(R"pb(
    name: "my_context1"
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { int_value: 3 }
    }
  )pb");
  context1.set_type_id(type1_id);
  int64_t context1_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateContext(context1, &context1_id),
            absl::OkStatus());
  context1.set_id(context1_id);
  context1.set_type("test_type_with_predefined_property");

  Context context2 = ParseTextProtoOrDie<Context>(R"pb(
    name: "my_context2")pb");
  context2.set_type_id(type2_id);

  int64_t context2_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateContext(context2, &context2_id),
            absl::OkStatus());
  context2.set_id(context2_id);
  context2.set_type("test_type_with_no_property");

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  if (!IfSchemaLessThan(/*schema_version=*/9)) {
    ASSERT_TRUE(context1.external_id().empty());
    ASSERT_TRUE(context2.external_id().empty());

    context1.set_external_id("my_context1");
    context2.set_external_id("my_context2");

    Context want_context1_after_update, want_context2_after_update;
    UpdateAndReturnNode<Context>(context1, *metadata_access_object_,
                                 metadata_access_object_container_.get(),
                                 want_context1_after_update);
    UpdateAndReturnNode<Context>(context2, *metadata_access_object_,
                                 metadata_access_object_container_.get(),
                                 want_context2_after_update);

    EXPECT_EQ(want_context1_after_update.external_id(), context1.external_id());
    EXPECT_EQ(want_context2_after_update.external_id(), context2.external_id());
  }

  EXPECT_NE(context1_id, context2_id);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Find contexts
  Context got_context1;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context1_id}, &contexts),
        absl::OkStatus());
    ASSERT_THAT(contexts, SizeIs(1));
    got_context1 = contexts[0];
  }
  EXPECT_THAT(got_context1, EqualsProto(context1, /*ignore_fields=*/{
                                            "create_time_since_epoch",
                                            "last_update_time_since_epoch"}));
  EXPECT_GT(got_context1.create_time_since_epoch(), 0);
  EXPECT_GT(got_context1.last_update_time_since_epoch(), 0);
  EXPECT_LE(got_context1.last_update_time_since_epoch(),
            absl::ToUnixMillis(absl::Now()));
  EXPECT_GE(got_context1.last_update_time_since_epoch(),
            got_context1.create_time_since_epoch());

  std::vector<Context> got_contexts;
  EXPECT_EQ(metadata_access_object_->FindContexts(&got_contexts),
            absl::OkStatus());
  EXPECT_EQ(got_contexts.size(), 2);
  EXPECT_THAT(
      got_contexts,
      UnorderedElementsAre(
          EqualsProto(context1,
                      /*ignore_fields=*/{"create_time_since_epoch",
                                         "last_update_time_since_epoch"}),
          EqualsProto(context2,
                      /*ignore_fields=*/{"create_time_since_epoch",
                                         "last_update_time_since_epoch"})));

  std::vector<Context> got_type2_contexts;
  EXPECT_EQ(metadata_access_object_->FindContextsByTypeId(
                type2_id, /*list_options=*/absl::nullopt, &got_type2_contexts,
                /*next_page_token=*/nullptr),
            absl::OkStatus());
  EXPECT_EQ(got_type2_contexts.size(), 1);
  EXPECT_THAT(got_type2_contexts[0],
              EqualsProto(context2,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  Context got_context_from_type_and_name1;
  EXPECT_EQ(metadata_access_object_->FindContextByTypeIdAndContextName(
                type1_id, "my_context1", /*id_only=*/false,
                &got_context_from_type_and_name1),
            absl::OkStatus());
  EXPECT_THAT(got_context_from_type_and_name1,
              EqualsProto(context1,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  Context got_context_from_type_and_name2;
  EXPECT_EQ(metadata_access_object_->FindContextByTypeIdAndContextName(
                type2_id, "my_context2", /*id_only=*/false,
                &got_context_from_type_and_name2),
            absl::OkStatus());
  EXPECT_THAT(got_context_from_type_and_name2,
              EqualsProto(context2,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  Context got_empty_context;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->FindContextByTypeIdAndContextName(
          type1_id, "my_context2", /*id_only=*/false, &got_empty_context)));
  EXPECT_THAT(got_empty_context, EqualsProto(Context()));

  Context got_context_from_type_and_name_with_only_id;
  Context expected_context_from_type_and_name_with_only_id;
  expected_context_from_type_and_name_with_only_id.set_id(context1.id());
  EXPECT_EQ(metadata_access_object_->FindContextByTypeIdAndContextName(
                type1_id, "my_context1", /*id_only=*/true,
                &got_context_from_type_and_name_with_only_id),
            absl::OkStatus());
  EXPECT_THAT(got_context_from_type_and_name_with_only_id,
              EqualsProto(expected_context_from_type_and_name_with_only_id));

  if (!IfSchemaLessThan(/*schema_version=*/9)) {
    // Test 1: contexts can be retrieved by external_ids.
    std::vector<Context> got_contexts_from_external_ids;
    std::vector<absl::string_view> external_ids = {"my_context1",
                                                   "my_context2"};
    MLMD_EXPECT_OK(metadata_access_object_->FindContextsByExternalIds(
        absl::MakeSpan(external_ids), &got_contexts_from_external_ids));
    EXPECT_THAT(
        got_contexts_from_external_ids,
        UnorderedElementsAre(
            EqualsProto(context1,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(context2,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));

    // Test 2: will return NOT_FOUND error when finding contexts by
    // non-existing external_id.
    std::vector<Context> got_contexts_from_external_ids_absent;
    std::vector<absl::string_view> external_ids_absent = {"my_context_absent"};
    EXPECT_TRUE(
        absl::IsNotFound(metadata_access_object_->FindContextsByExternalIds(
            absl::MakeSpan(external_ids_absent),
            &got_contexts_from_external_ids_absent)));

    // Test 3: will return whatever found when a part of external_ids is
    // non-existing, e.g., the input error vector is {"my_context1",
    // "my_context_absent"}.
    external_ids_absent.push_back(absl::string_view("my_context1"));
    MLMD_EXPECT_OK(metadata_access_object_->FindContextsByExternalIds(
        absl::MakeSpan(external_ids_absent),
        &got_contexts_from_external_ids_absent));
    ASSERT_EQ(got_contexts_from_external_ids_absent.size(), 1);
    EXPECT_THAT(
        got_contexts_from_external_ids_absent,
        UnorderedElementsAre(EqualsProto(
            context1, /*ignore_fields=*/{"type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"})));

    // Test 4: will return INVALID_ARGUMENT error when any of the external_ids
    // is empty.
    std::vector<absl::string_view> external_ids_empty = {""};
    std::vector<Context> got_contexts_from_empty_external_ids;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->FindContextsByExternalIds(
            absl::MakeSpan(external_ids_empty),
            &got_contexts_from_empty_external_ids)));
  }
}

TEST_P(MetadataAccessObjectTest, ListArtifactsByType) {
  ASSERT_EQ(Init(), absl::OkStatus());

  // Setup: create an artifact type and insert two instances.
  int64_t type_id;
  {
    ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"pb(
      name: 'test_type_with_predefined_property'
      properties { key: 'property_1' value: INT }
    )pb");

    ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
              absl::OkStatus());
  }
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
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
    int64_t entity_id = -1;
    EXPECT_EQ(metadata_access_object_->CreateArtifact(entity_1, &entity_id),
              absl::OkStatus());
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
    int64_t entity_id = -1;
    EXPECT_EQ(metadata_access_object_->CreateArtifact(entity_2, &entity_id),
              absl::OkStatus());
    entity_2.set_id(entity_id);
  }

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: List entities by default ordering -- ID.
  {
    ListOperationOptions options;
    options.set_max_result_size(1);

    std::vector<Artifact> entities;
    std::string next_page_token;
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsByTypeId(
            type_id, absl::make_optional(options), &entities, &next_page_token),
        absl::OkStatus());
    EXPECT_THAT(next_page_token, Not(IsEmpty()));
    EXPECT_THAT(entities, ElementsAre(EqualsProto(
                              entity_1, /*ignore_fields=*/{
                                  "uri", "type", "create_time_since_epoch",
                                  "last_update_time_since_epoch"})));

    entities.clear();
    options.set_next_page_token(next_page_token);
    ASSERT_EQ(
        metadata_access_object_->FindArtifactsByTypeId(
            type_id, absl::make_optional(options), &entities, &next_page_token),
        absl::OkStatus());
    EXPECT_THAT(next_page_token, IsEmpty());
    EXPECT_THAT(entities,
                ElementsAre(EqualsProto(
                    entity_2,
                    /*ignore_fields=*/{"uri", "type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
}

TEST_P(MetadataAccessObjectTest, ListExecutionsByType) {
  ASSERT_EQ(Init(), absl::OkStatus());

  // Setup: create an excution type and insert two instances.
  int64_t type_id;
  {
    ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"pb(
      name: 'test_type_with_predefined_property'
      properties { key: 'property_1' value: INT }
    )pb");

    ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
              absl::OkStatus());
  }
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
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
    int64_t entity_id = -1;
    EXPECT_EQ(metadata_access_object_->CreateExecution(entity_1, &entity_id),
              absl::OkStatus());
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
    int64_t entity_id = -1;
    EXPECT_EQ(metadata_access_object_->CreateExecution(entity_2, &entity_id),
              absl::OkStatus());
    entity_2.set_id(entity_id);
  }

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: List entities by default ordering -- ID.
  {
    ListOperationOptions options;
    options.set_max_result_size(1);

    std::vector<Execution> entities;
    std::string next_page_token;
    ASSERT_EQ(
        metadata_access_object_->FindExecutionsByTypeId(
            type_id, absl::make_optional(options), &entities, &next_page_token),
        absl::OkStatus());
    EXPECT_THAT(next_page_token, Not(IsEmpty()));
    EXPECT_THAT(entities,
                ElementsAre(EqualsProto(entity_1, /*ignore_fields=*/{
                                            "type", "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));

    entities.clear();
    options.set_next_page_token(next_page_token);
    ASSERT_EQ(
        metadata_access_object_->FindExecutionsByTypeId(
            type_id, absl::make_optional(options), &entities, &next_page_token),
        absl::OkStatus());
    EXPECT_THAT(next_page_token, IsEmpty());
    EXPECT_THAT(entities,
                ElementsAre(EqualsProto(
                    entity_2,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
}

TEST_P(MetadataAccessObjectTest, ListContextsByType) {
  ASSERT_EQ(Init(), absl::OkStatus());

  // Setup: create a context type and insert two instances.
  int64_t type_id;
  {
    ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
      name: 'test_type_with_predefined_property'
      properties { key: 'property_1' value: INT }
    )pb");

    ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  }
  Context context_1;
  {
    context_1 = ParseTextProtoOrDie<Context>(R"pb(
      name: "context_1a"
      properties {
        key: 'property_1'
        value: { int_value: 1 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { int_value: 3 }
      }
    )pb");
    context_1.set_type_id(type_id);
    int64_t context_id = -1;
    EXPECT_EQ(metadata_access_object_->CreateContext(context_1, &context_id),
              absl::OkStatus());
    context_1.set_id(context_id);
  }
  Context context_2;
  {
    context_2 = ParseTextProtoOrDie<Context>(R"pb(
      name: "context_1b"
      properties {
        key: 'property_1'
        value: { int_value: 2 }
      }
      custom_properties {
        key: 'custom_property_1'
        value: { int_value: 4 }
      }
    )pb");
    context_2.set_type_id(type_id);
    int64_t context_id = -1;
    EXPECT_EQ(metadata_access_object_->CreateContext(context_2, &context_id),
              absl::OkStatus());
    context_2.set_id(context_id);
  }

  // Setup: insert one more context type and an additional instance. This is
  // additional data that will not retrieved by the test queries.
  {
    int64_t type2_id;
    ContextType type2 = ParseTextProtoOrDie<ContextType>(R"pb(
      name: 'test_type_with_no_property'
    )pb");
    ASSERT_EQ(metadata_access_object_->CreateType(type2, &type2_id),
              absl::OkStatus());
    ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

    Context context = ParseTextProtoOrDie<Context>(R"pb(
      name: "my_context2")pb");
    context.set_type_id(type2_id);
    int64_t context_id = -1;
    EXPECT_EQ(metadata_access_object_->CreateContext(context, &context_id),
              absl::OkStatus());
  }

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: List contexts by default ordering -- ID.
  {
    ListOperationOptions options;
    options.set_max_result_size(1);

    std::vector<Context> contexts;
    std::string next_page_token;
    ASSERT_EQ(
        metadata_access_object_->FindContextsByTypeId(
            type_id, absl::make_optional(options), &contexts, &next_page_token),
        absl::OkStatus());
    EXPECT_THAT(next_page_token, Not(IsEmpty()));
    EXPECT_THAT(contexts,
                ElementsAre(EqualsProto(context_1, /*ignore_fields=*/{
                                            "type", "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));

    contexts.clear();
    options.set_next_page_token(next_page_token);
    ASSERT_EQ(
        metadata_access_object_->FindContextsByTypeId(
            type_id, absl::make_optional(options), &contexts, &next_page_token),
        absl::OkStatus());
    EXPECT_THAT(next_page_token, IsEmpty());
    EXPECT_THAT(contexts,
                ElementsAre(EqualsProto(
                    context_2,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  // Test: List contexts by reverse default ordering (ID)
  {
    ListOperationOptions options;
    options.mutable_order_by_field()->set_is_asc(false);
    options.set_max_result_size(1);

    std::vector<Context> contexts;
    std::string next_page_token;
    ASSERT_EQ(
        metadata_access_object_->FindContextsByTypeId(
            type_id, absl::make_optional(options), &contexts, &next_page_token),
        absl::OkStatus());
    EXPECT_THAT(next_page_token, Not(IsEmpty()));
    EXPECT_THAT(contexts,
                ElementsAre(EqualsProto(context_2, /*ignore_fields=*/{
                                            "type", "create_time_since_epoch",
                                            "last_update_time_since_epoch"})));

    contexts.clear();
    options.set_next_page_token(next_page_token);
    ASSERT_EQ(
        metadata_access_object_->FindContextsByTypeId(
            type_id, absl::make_optional(options), &contexts, &next_page_token),
        absl::OkStatus());
    EXPECT_THAT(next_page_token, IsEmpty());
    EXPECT_THAT(contexts,
                ElementsAre(EqualsProto(
                    context_1,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  // Test: List contexts through a big max-result size.
  {
    ListOperationOptions options;
    options.set_max_result_size(100);

    std::vector<Context> contexts;
    std::string next_page_token;
    ASSERT_EQ(
        metadata_access_object_->FindContextsByTypeId(
            type_id, absl::make_optional(options), &contexts, &next_page_token),
        absl::OkStatus());
    EXPECT_THAT(next_page_token, IsEmpty());
    EXPECT_THAT(
        contexts,
        ElementsAre(
            EqualsProto(context_1,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(context_2,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
}

TEST_P(MetadataAccessObjectTest, CreateContextError) {
  ASSERT_EQ(Init(), absl::OkStatus());
  Context context;
  int64_t context_id;

  // unknown type specified
  EXPECT_TRUE(absl::IsInvalidArgument(
      metadata_access_object_->CreateContext(context, &context_id)));

  context.set_type_id(1);
  EXPECT_TRUE(absl::IsNotFound(
      metadata_access_object_->CreateContext(context, &context_id)));

  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type_disallow_custom_property'
    properties { key: 'property_1' value: INT }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType type;
  type.set_name("test_type");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context context;
  context.set_type_id(type_id);
  context.set_name("test context name");
  int64_t context_id;
  EXPECT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());
  // insert the same context again to check the unique constraint
  absl::Status unique_constraint_violation_status =
      metadata_access_object_->CreateContext(context, &context_id);
  EXPECT_EQ(CheckUniqueConstraintAndResetTransaction(
                unique_constraint_violation_status),
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateContextWithDuplicatedExternalIdError) {
  if (SkipIfEarlierSchemaLessThan(/*min_schema_version=*/9)) {
    return;
  }
  MLMD_ASSERT_OK(Init());
  ContextType type;
  type.set_name("test_type");
  int64_t type_id;
  MLMD_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context context;
  context.set_type_id(type_id);
  context.set_external_id("context1");
  context.set_name("test context name");
  int64_t context_id;

  MLMD_EXPECT_OK(metadata_access_object_->CreateContext(context, &context_id));

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Insert the same context again to check the unique constraint
  absl::Status unique_constraint_violation_status =
      metadata_access_object_->CreateContext(context, &context_id);
  MLMD_EXPECT_OK(CheckUniqueConstraintAndResetTransaction(
      unique_constraint_violation_status));
}

TEST_P(MetadataAccessObjectTest, UpdateContext) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
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
  int64_t context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context1, &context_id),
            absl::OkStatus());
  Context got_context_before_update;

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
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
  want_context.clear_type_id();
  // sleep to verify the latest update time is updated.
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(metadata_access_object_->UpdateContext(want_context),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  want_context.set_type_id(type_id);
  Context got_context_after_update;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_after_update = contexts[0];
  }
  EXPECT_THAT(want_context,
              EqualsProto(got_context_after_update,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_context_before_update.create_time_since_epoch(),
            got_context_after_update.create_time_since_epoch());
  EXPECT_LT(got_context_before_update.last_update_time_since_epoch(),
            got_context_after_update.last_update_time_since_epoch());
}

TEST_P(MetadataAccessObjectTest, UpdateContextWithCustomUpdatetime) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context1, &context_id),
            absl::OkStatus());
  Context got_context_before_update;

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
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
  EXPECT_EQ(metadata_access_object_->UpdateContext(want_context, update_time,
                                                   /*force_update_time=*/false),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context got_context_after_update;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_after_update = contexts[0];
  }
  EXPECT_THAT(want_context,
              EqualsProto(got_context_after_update,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_context_before_update.create_time_since_epoch(),
            got_context_after_update.create_time_since_epoch());
  EXPECT_EQ(got_context_after_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, UpdateContextWithForceUpdateTimeEnabled) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context1, &context_id),
            absl::OkStatus());
  Context got_context_before_update;

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_before_update = contexts[0];
  }

  // Update with no changes and force_update_time disabled.
  absl::Time update_time = absl::InfiniteFuture();
  ASSERT_EQ(metadata_access_object_->UpdateContext(got_context_before_update,
                                                   update_time,
                                                   /*force_update_time=*/false),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context got_context_after_1st_update;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_after_1st_update = contexts[0];
  }
  // Expect no changes for the updated resource.
  EXPECT_THAT(got_context_before_update,
              EqualsProto(got_context_after_1st_update));

  // Update with no changes again but with force_update_time set to true.
  ASSERT_EQ(metadata_access_object_->UpdateContext(got_context_after_1st_update,
                                                   update_time,
                                                   /*force_update_time=*/true),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context got_context_after_2nd_update;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_after_2nd_update = contexts[0];
  }
  // Expect no changes for the updated resource other than
  // `last_update_time_since_epoch`.
  EXPECT_THAT(
      got_context_after_2nd_update,
      EqualsProto(got_context_after_1st_update,
                  /*ignore_fields=*/{"type", "last_update_time_since_epoch"}));
  EXPECT_NE(got_context_after_2nd_update.last_update_time_since_epoch(),
            got_context_after_1st_update.last_update_time_since_epoch());
  EXPECT_EQ(got_context_after_2nd_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, UpdateContextWithMasking) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
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
  int64_t context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context1, &context_id),
            absl::OkStatus());
  Context got_context_before_update;

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
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

  google::protobuf::FieldMask mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(R"pb(
        paths: 'name'
        paths: 'properties.property_1'
        paths: 'properties.property_2'
        paths: 'custom_properties.custom_property_1'
      )pb");

  // sleep to verify the latest update time is updated.
  absl::SleepFor(absl::Milliseconds(1));
  EXPECT_EQ(metadata_access_object_->UpdateContext(want_context, mask),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context got_context_after_update;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_after_update = contexts[0];
  }
  EXPECT_THAT(want_context,
              EqualsProto(got_context_after_update,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_context_before_update.create_time_since_epoch(),
            got_context_after_update.create_time_since_epoch());
  EXPECT_LT(got_context_before_update.last_update_time_since_epoch(),
            got_context_after_update.last_update_time_since_epoch());
}

TEST_P(MetadataAccessObjectTest, UpdateContextWithCustomUpdatetimeAndMasking) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context1, &context_id),
            absl::OkStatus());
  Context got_context_before_update;

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
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
  google::protobuf::FieldMask mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(R"pb(
        paths: 'name'
        paths: 'properties.property_1'
        paths: 'properties.property_2'
        paths: 'custom_properties.custom_property_1'
      )pb");
  absl::Time update_time = absl::InfiniteFuture();
  EXPECT_EQ(
      metadata_access_object_->UpdateContext(want_context, update_time,
                                             /*force_update_time=*/false, mask),
      absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context got_context_after_update;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_after_update = contexts[0];
  }
  EXPECT_THAT(want_context,
              EqualsProto(got_context_after_update,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_EQ(got_context_before_update.create_time_since_epoch(),
            got_context_after_update.create_time_since_epoch());
  EXPECT_EQ(got_context_after_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest,
       UpdateContextWithForceUpdateTimeEnabledAndMasking) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType type = ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: STRING }
  )pb");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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
  int64_t context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context1, &context_id),
            absl::OkStatus());
  Context got_context_before_update;

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_before_update = contexts[0];
  }
  google::protobuf::FieldMask mask =
      ParseTextProtoOrDie<google::protobuf::FieldMask>(R"pb(
        paths: 'name'
        paths: 'properties.property_1'
        paths: 'properties.property_2'
        paths: 'custom_properties.custom_property_1'
      )pb");
  // Update with no changes and force_update_time disabled.
  absl::Time update_time = absl::InfiniteFuture();
  ASSERT_EQ(metadata_access_object_->UpdateContext(
                got_context_before_update, update_time,
                /*force_update_time=*/false, mask),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context got_context_after_1st_update;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_after_1st_update = contexts[0];
  }
  // Expect no changes for the updated resource.
  EXPECT_THAT(got_context_before_update,
              EqualsProto(got_context_after_1st_update));

  // Update with no changes again but with force_update_time set to true.
  ASSERT_EQ(metadata_access_object_->UpdateContext(
                got_context_after_1st_update, update_time,
                /*force_update_time=*/true, mask),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Context got_context_after_2nd_update;
  {
    std::vector<Context> contexts;
    EXPECT_EQ(
        metadata_access_object_->FindContextsById({context_id}, &contexts),
        absl::OkStatus());
    ASSERT_THAT(contexts, SizeIs(1));
    got_context_after_2nd_update = contexts[0];
  }
  // Expect no changes for the updated resource other than
  // `last_update_time_since_epoch`.
  EXPECT_THAT(
      got_context_after_2nd_update,
      EqualsProto(got_context_after_1st_update,
                  /*ignore_fields=*/{"type", "last_update_time_since_epoch"}));
  EXPECT_NE(got_context_after_2nd_update.last_update_time_since_epoch(),
            got_context_after_1st_update.last_update_time_since_epoch());
  EXPECT_EQ(got_context_after_2nd_update.last_update_time_since_epoch(),
            absl::ToUnixMillis(update_time));
}

TEST_P(MetadataAccessObjectTest, CreateAndUseAssociation) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t execution_type_id = InsertType<ExecutionType>("execution_type");
  int64_t context_type_id = InsertType<ContextType>("context_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  (*execution.mutable_custom_properties())["custom"].set_int_value(3);
  Context context = ParseTextProtoOrDie<Context>("name: 'context_instance'");
  context.set_type_id(context_type_id);

  int64_t execution_id, context_id;
  ASSERT_EQ(metadata_access_object_->CreateExecution(execution, &execution_id),
            absl::OkStatus());
  execution.set_id(execution_id);
  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());
  context.set_id(context_id);

  Association association;
  association.set_execution_id(execution_id);
  association.set_context_id(context_id);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  int64_t association_id;
  EXPECT_EQ(
      metadata_access_object_->CreateAssociation(association, &association_id),
      absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  std::vector<Context> got_contexts;
  EXPECT_EQ(metadata_access_object_->FindContextsByExecution(execution_id,
                                                             &got_contexts),
            absl::OkStatus());
  ASSERT_EQ(got_contexts.size(), 1);
  EXPECT_THAT(context, EqualsProto(got_contexts[0], /*ignore_fields=*/{
                                       "type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));

  std::vector<Execution> got_executions;
  EXPECT_EQ(metadata_access_object_->FindExecutionsByContext(context_id,
                                                             &got_executions),
            absl::OkStatus());
  ASSERT_EQ(got_executions.size(), 1);
  EXPECT_THAT(execution, EqualsProto(got_executions[0], /*ignore_fields=*/{
                                         "type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"}));

  std::vector<Artifact> got_artifacts;
  EXPECT_EQ(metadata_access_object_->FindArtifactsByContext(context_id,
                                                            &got_artifacts),
            absl::OkStatus());
  EXPECT_EQ(got_artifacts.size(), 0);
}

TEST_P(MetadataAccessObjectTest, GetAssociationsByContexts) {
  ASSERT_EQ(Init(), absl::OkStatus());
  // Setup: Prepare associations.
  int64_t execution_type_id = InsertType<ExecutionType>("execution_type");
  int64_t context_type_id = InsertType<ContextType>("context_type");
  Execution execution1;
  execution1.set_type_id(execution_type_id);
  (*execution1.mutable_custom_properties())["custom"].set_int_value(3);
  Execution execution2;
  execution2.set_type_id(execution_type_id);
  (*execution2.mutable_custom_properties())["custom"].set_int_value(5);

  Context context = ParseTextProtoOrDie<Context>("name: 'context_instance'");
  context.set_type_id(context_type_id);

  int64_t execution_id_1, execution_id_2, context_id;
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(execution1, &execution_id_1),
      absl::OkStatus());
  execution1.set_id(execution_id_1);
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(execution2, &execution_id_2),
      absl::OkStatus());
  execution2.set_id(execution_id_2);

  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());
  context.set_id(context_id);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Association association1;
  association1.set_execution_id(execution_id_1);
  association1.set_context_id(context_id);

  Association association2;
  association2.set_execution_id(execution_id_2);
  association2.set_context_id(context_id);

  int64_t association_id_1;
  ASSERT_EQ(metadata_access_object_->CreateAssociation(association1,
                                                       &association_id_1),
            absl::OkStatus());
  int64_t association_id_2;
  ASSERT_EQ(metadata_access_object_->CreateAssociation(association2,
                                                       &association_id_2),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: associations can be found by the context.
  std::vector<Association> got_associations;
  ASSERT_EQ(metadata_access_object_->FindAssociationsByContexts(
                {context_id}, &got_associations),
            absl::OkStatus());
  ASSERT_THAT(got_associations, SizeIs(2));
  EXPECT_THAT(
      got_associations,
      UnorderedElementsAre(EqualsProto(association1, /*ignore_fields=*/{}),
                           EqualsProto(association2, /*ignore_fields=*/{})));

  // Test: Existing + Unexisting context_ids returns exsisting associations.
  const int64_t kInvalidContextId = 12345678;
  got_associations.clear();
  ASSERT_EQ(metadata_access_object_->FindAssociationsByContexts(
                {context_id, kInvalidContextId}, &got_associations),
            absl::OkStatus());
  ASSERT_THAT(got_associations, SizeIs(2));
  EXPECT_THAT(
      got_associations,
      UnorderedElementsAre(EqualsProto(association1, /*ignore_fields=*/{}),
                           EqualsProto(association2, /*ignore_fields=*/{})));

  // Test: Unexisting context_id returns nothing.
  got_associations.clear();
  EXPECT_TRUE(
      absl::IsNotFound(metadata_access_object_->FindAssociationsByContexts(
          {kInvalidContextId}, &got_associations)));
}

TEST_P(MetadataAccessObjectTest, GetAssociationsByExecutions) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t execution_type_id = InsertType<ExecutionType>("execution_type");
  int64_t context_type_id = InsertType<ContextType>("context_type");

  Execution execution_1;
  execution_1.set_type_id(execution_type_id);
  Execution execution_2;
  execution_2.set_type_id(execution_type_id);

  Context context = ParseTextProtoOrDie<Context>("name: 'associated_context'");
  context.set_type_id(context_type_id);

  int64_t execution_id_1, execution_id_2, context_id;
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(execution_1, &execution_id_1),
      absl::OkStatus());
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(execution_2, &execution_id_2),
      absl::OkStatus());
  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Association association_1 = ParseTextProtoOrDie<Association>(absl::Substitute(
      R"pb(execution_id: $0 context_id: $1)pb", execution_id_1, context_id));
  Association association_2 = ParseTextProtoOrDie<Association>(absl::Substitute(
      R"pb(execution_id: $0 context_id: $1)pb", execution_id_2, context_id));
  int64_t association_id_1, association_id_2;
  ASSERT_EQ(metadata_access_object_->CreateAssociation(association_1,
                                                       &association_id_1),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_->CreateAssociation(association_2,
                                                       &association_id_2),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: got associations for execution_ids = {execution_id_1,
  // execution_id_2}.
  {
    std::vector<Association> got_associations;
    ASSERT_EQ(metadata_access_object_->FindAssociationsByExecutions(
                  {execution_id_1, execution_id_2}, &got_associations),
              absl::OkStatus());
    ASSERT_THAT(got_associations, SizeIs(2));
    EXPECT_THAT(got_associations,
                UnorderedPointwise(EqualsProto<Association>(),
                                   {association_1, association_2}));
  }

  // Test: got `association_1` for execution_ids = {execution_id_1,
  // invalid_execution_id}.
  {
    std::vector<Association> got_associations;
    int64_t invalid_execution_id = execution_id_1 + execution_id_2;
    ASSERT_EQ(metadata_access_object_->FindAssociationsByExecutions(
                  {execution_id_1, invalid_execution_id}, &got_associations),
              absl::OkStatus());
    ASSERT_THAT(got_associations, SizeIs(1));
    EXPECT_THAT(got_associations, UnorderedPointwise(EqualsProto<Association>(),
                                                     {association_1}));
  }

  // Test: got empty list of associations for empty `execution_ids`.
  {
    std::vector<Association> got_associations;
    ASSERT_EQ(metadata_access_object_->FindAssociationsByExecutions(
                  {}, &got_associations),
              absl::OkStatus());
    EXPECT_TRUE(got_associations.empty());
  }

  // Test: returns INVALID_ARGUMENT error if `associations` is null.
  {
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->FindAssociationsByExecutions(
            {execution_id_1, execution_id_2}, nullptr)));
  }
}

TEST_P(MetadataAccessObjectTest, GetAttributionsByArtifacts) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t artifact_type_id = InsertType<ArtifactType>("artifact_type");
  int64_t context_type_id = InsertType<ContextType>("context_type");

  Artifact artifact_1;
  artifact_1.set_type_id(artifact_type_id);
  Artifact artifact_2;
  artifact_2.set_type_id(artifact_type_id);

  Context context = ParseTextProtoOrDie<Context>("name: 'attributed_context'");
  context.set_type_id(context_type_id);

  int64_t artifact_id_1, artifact_id_2, context_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact_1, &artifact_id_1),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact_2, &artifact_id_2),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Attribution attribution_1 = ParseTextProtoOrDie<Attribution>(absl::Substitute(
      R"pb(artifact_id: $0 context_id: $1)pb", artifact_id_1, context_id));
  Attribution attribution_2 = ParseTextProtoOrDie<Attribution>(absl::Substitute(
      R"pb(artifact_id: $0 context_id: $1)pb", artifact_id_2, context_id));
  int64_t attribution_id_1, attribution_id_2;
  ASSERT_EQ(metadata_access_object_->CreateAttribution(attribution_1,
                                                       &attribution_id_1),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_->CreateAttribution(attribution_2,
                                                       &attribution_id_2),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Test: got attributions for artifact_ids = {artifact_id_1, artifact_id_2}.
  {
    std::vector<Attribution> got_attributions;
    ASSERT_EQ(metadata_access_object_->FindAttributionsByArtifacts(
                  {artifact_id_1, artifact_id_2}, &got_attributions),
              absl::OkStatus());
    ASSERT_THAT(got_attributions, SizeIs(2));
    EXPECT_THAT(got_attributions,
                UnorderedPointwise(EqualsProto<Attribution>(),
                                   {attribution_1, attribution_2}));
  }

  // Test: got attribution_1 for artifact_ids = {artifact_id_1,
  // invalid_artifact_id}.
  {
    std::vector<Attribution> got_attributions;
    int64_t invalid_artifact_id = artifact_id_1 + artifact_id_2;
    ASSERT_EQ(metadata_access_object_->FindAttributionsByArtifacts(
                  {artifact_id_1, invalid_artifact_id}, &got_attributions),
              absl::OkStatus());
    ASSERT_THAT(got_attributions, SizeIs(1));
    EXPECT_THAT(got_attributions, UnorderedPointwise(EqualsProto<Attribution>(),
                                                     {attribution_1}));
  }

  // Test: got empty list for empty artifact ids.
  {
    std::vector<Attribution> got_attributions;
    ASSERT_EQ(metadata_access_object_->FindAttributionsByArtifacts(
                  {}, &got_attributions),
              absl::OkStatus());
    ASSERT_THAT(got_attributions, IsEmpty());
  }

  // Test: returns INVALID_ARGUMENT error if `attributions` is null.
  {
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_access_object_->FindAttributionsByArtifacts(
            {artifact_id_1, artifact_id_2}, nullptr)));
  }
}

TEST_P(MetadataAccessObjectTest, GetAssociationUsingPagination) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t execution_type_id = InsertType<ExecutionType>("execution_type");
  int64_t context_type_id = InsertType<ContextType>("context_type");
  Execution execution1;
  execution1.set_type_id(execution_type_id);
  (*execution1.mutable_custom_properties())["custom"].set_int_value(3);
  Execution execution2;
  execution2.set_type_id(execution_type_id);
  (*execution2.mutable_custom_properties())["custom"].set_int_value(5);

  Context context = ParseTextProtoOrDie<Context>("name: 'context_instance'");
  context.set_type_id(context_type_id);

  int64_t execution_id_1, execution_id_2, context_id;
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(execution1, &execution_id_1),
      absl::OkStatus());
  execution1.set_id(execution_id_1);
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(execution2, &execution_id_2),
      absl::OkStatus());
  execution2.set_id(execution_id_2);

  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());
  context.set_id(context_id);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Association association1;
  association1.set_execution_id(execution_id_1);
  association1.set_context_id(context_id);

  Association association2;
  association2.set_execution_id(execution_id_2);
  association2.set_context_id(context_id);

  int64_t association_id_1;
  EXPECT_EQ(metadata_access_object_->CreateAssociation(association1,
                                                       &association_id_1),
            absl::OkStatus());
  int64_t association_id_2;
  EXPECT_EQ(metadata_access_object_->CreateAssociation(association2,
                                                       &association_id_2),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )pb");

  std::string next_page_token;
  std::vector<Execution> got_executions;
  EXPECT_EQ(metadata_access_object_->FindExecutionsByContext(
                context_id, list_options, &got_executions, &next_page_token),
            absl::OkStatus());
  EXPECT_THAT(got_executions, SizeIs(1));
  EXPECT_THAT(execution2, EqualsProto(got_executions[0], /*ignore_fields=*/{
                                          "type", "create_time_since_epoch",
                                          "last_update_time_since_epoch"}));
  ASSERT_FALSE(next_page_token.empty());

  list_options.set_next_page_token(next_page_token);
  got_executions.clear();
  EXPECT_EQ(metadata_access_object_->FindExecutionsByContext(
                context_id, list_options, &got_executions, &next_page_token),
            absl::OkStatus());
  EXPECT_THAT(got_executions, SizeIs(1));
  EXPECT_THAT(execution1, EqualsProto(got_executions[0], /*ignore_fields=*/{
                                          "type", "create_time_since_epoch",
                                          "last_update_time_since_epoch"}));
  ASSERT_TRUE(next_page_token.empty());
}

TEST_P(MetadataAccessObjectTest, GetAssociationFilterStateQuery) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t execution_type_id = InsertType<ExecutionType>("execution_type");
  int64_t context_type_id = InsertType<ContextType>("context_type");
  Execution execution1;
  execution1.set_type_id(execution_type_id);
  execution1.set_last_known_state(Execution::NEW);
  Execution execution2;
  execution2.set_type_id(execution_type_id);
  execution2.set_last_known_state(Execution::RUNNING);

  Context context1 = ParseTextProtoOrDie<Context>("name: 'context_1'");
  context1.set_type_id(context_type_id);
  Context context2 = ParseTextProtoOrDie<Context>("name: 'context_2'");
  context2.set_type_id(context_type_id);

  int64_t execution_id_1, execution_id_2, context_id_1, context_id_2;
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(execution1, &execution_id_1),
      absl::OkStatus());
  execution1.set_id(execution_id_1);
  ASSERT_EQ(
      metadata_access_object_->CreateExecution(execution2, &execution_id_2),
      absl::OkStatus());
  execution2.set_id(execution_id_2);

  ASSERT_EQ(metadata_access_object_->CreateContext(context1, &context_id_1),
            absl::OkStatus());
  context1.set_id(context_id_1);
  ASSERT_EQ(metadata_access_object_->CreateContext(context2, &context_id_2),
            absl::OkStatus());
  context2.set_id(context_id_2);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Association association1;
  association1.set_execution_id(execution_id_1);
  association1.set_context_id(context_id_1);

  Association association2;
  association2.set_execution_id(execution_id_2);
  association2.set_context_id(context_id_2);

  int64_t association_id_1;
  EXPECT_EQ(metadata_access_object_->CreateAssociation(association1,
                                                       &association_id_1),
            absl::OkStatus());
  int64_t association_id_2;
  EXPECT_EQ(metadata_access_object_->CreateAssociation(association2,
                                                       &association_id_2),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  std::string next_page_token;
  std::vector<Execution> got_executions;
  got_executions.clear();
  ASSERT_EQ(metadata_access_object_->FindExecutionsByContext(
                context_id_1, absl::nullopt, &got_executions, &next_page_token),
            absl::OkStatus());
  ASSERT_THAT(got_executions, SizeIs(1));
  EXPECT_THAT(execution1, EqualsProto(got_executions[0], /*ignore_fields=*/{
                                          "type", "create_time_since_epoch",
                                          "last_update_time_since_epoch"}));
  ASSERT_TRUE(next_page_token.empty());

  got_executions.clear();
  ASSERT_EQ(metadata_access_object_->FindExecutionsByContext(
                context_id_2, absl::nullopt, &got_executions, &next_page_token),
            absl::OkStatus());
  ASSERT_THAT(got_executions, SizeIs(1));
  EXPECT_THAT(execution2, EqualsProto(got_executions[0], /*ignore_fields=*/{
                                          "type", "create_time_since_epoch",
                                          "last_update_time_since_epoch"}));
  ASSERT_TRUE(next_page_token.empty());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
        filter_query: "last_known_state = NEW OR last_known_state = RUNNING"
      )pb");

  got_executions.clear();
  ASSERT_EQ(metadata_access_object_->FindExecutionsByContext(
                context_id_1, list_options, &got_executions, &next_page_token),
            absl::OkStatus());
  ASSERT_THAT(got_executions, SizeIs(1));
  EXPECT_THAT(execution1, EqualsProto(got_executions[0], /*ignore_fields=*/{
                                          "type", "create_time_since_epoch",
                                          "last_update_time_since_epoch"}));
  ASSERT_TRUE(next_page_token.empty());

  got_executions.clear();
  ASSERT_EQ(metadata_access_object_->FindExecutionsByContext(
                context_id_2, list_options, &got_executions, &next_page_token),
            absl::OkStatus());
  ASSERT_THAT(got_executions, SizeIs(1));
  EXPECT_THAT(execution2, EqualsProto(got_executions[0], /*ignore_fields=*/{
                                          "type", "create_time_since_epoch",
                                          "last_update_time_since_epoch"}));
  ASSERT_TRUE(next_page_token.empty());
}

TEST_P(MetadataAccessObjectTest, GetAttributionUsingPagination) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t artifact_type_id = InsertType<ArtifactType>("artifact_type");
  int64_t context_type_id = InsertType<ContextType>("context_type");
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

  int64_t artifact_id_1, artifact_id_2, context_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact1, &artifact_id_1),
            absl::OkStatus());
  artifact1.set_id(artifact_id_1);
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact2, &artifact_id_2),
            absl::OkStatus());
  artifact2.set_id(artifact_id_2);

  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());
  context.set_id(context_id);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Attribution attribution1;
  attribution1.set_artifact_id(artifact_id_1);
  attribution1.set_context_id(context_id);

  Attribution attribution2;
  attribution2.set_artifact_id(artifact_id_2);
  attribution2.set_context_id(context_id);

  int64_t attribution_id_1;
  EXPECT_EQ(metadata_access_object_->CreateAttribution(attribution1,
                                                       &attribution_id_1),
            absl::OkStatus());
  int64_t attribution_id_2;
  EXPECT_EQ(metadata_access_object_->CreateAttribution(attribution2,
                                                       &attribution_id_2),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )pb");

  std::string next_page_token;
  std::vector<Artifact> got_artifacts;
  EXPECT_EQ(metadata_access_object_->FindArtifactsByContext(
                context_id, list_options, &got_artifacts, &next_page_token),
            absl::OkStatus());
  EXPECT_THAT(got_artifacts, SizeIs(1));
  EXPECT_THAT(artifact2, EqualsProto(got_artifacts[0], /*ignore_fields=*/{
                                         "type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"}));
  ASSERT_FALSE(next_page_token.empty());

  got_artifacts.clear();
  list_options.set_next_page_token(next_page_token);
  EXPECT_EQ(metadata_access_object_->FindArtifactsByContext(
                context_id, list_options, &got_artifacts, &next_page_token),
            absl::OkStatus());
  EXPECT_THAT(got_artifacts, SizeIs(1));
  ASSERT_TRUE(next_page_token.empty());
  EXPECT_THAT(artifact1, EqualsProto(got_artifacts[0], /*ignore_fields=*/{
                                         "type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"}));
}

TEST_P(MetadataAccessObjectTest, GetEmptyAttributionAssociationWithPagination) {
  ASSERT_EQ(Init(), absl::OkStatus());
  const ContextType context_type = CreateTypeFromTextProto<ContextType>(
      "name: 't1'", *metadata_access_object_,
      metadata_access_object_container_.get());
  Context context = ParseTextProtoOrDie<Context>("name: 'c1'");
  context.set_type_id(context_type.id());
  int64_t context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());
  context.set_id(context_id);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  const ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"pb(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )pb");
  {
    std::vector<Artifact> got_artifacts;
    std::string next_page_token;
    EXPECT_EQ(metadata_access_object_->FindArtifactsByContext(
                  context_id, list_options, &got_artifacts, &next_page_token),
              absl::OkStatus());
    EXPECT_THAT(got_artifacts, IsEmpty());
  }

  {
    std::vector<Execution> got_executions;
    std::string next_page_token;
    EXPECT_EQ(metadata_access_object_->FindExecutionsByContext(
                  context_id, list_options, &got_executions, &next_page_token),
              absl::OkStatus());
    EXPECT_THAT(got_executions, IsEmpty());
  }
}

TEST_P(MetadataAccessObjectTest, CreateAssociationError) {
  ASSERT_EQ(Init(), absl::OkStatus());

  // Create base association with
  // * valid context id
  // * valid execution id
  int64_t context_type_id = InsertType<ContextType>("test_context_type");
  Context context;
  context.set_type_id(context_type_id);
  context.set_name("test_context");
  int64_t context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());

  int64_t execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  int64_t execution_id;
  ASSERT_EQ(metadata_access_object_->CreateExecution(execution, &execution_id),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Association base_association;
  base_association.set_context_id(context_id);
  base_association.set_execution_id(execution_id);

  // no context id
  {
    Association association = base_association;
    association.clear_context_id();
    int64_t association_id;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_access_object_->CreateAssociation(
            association, &association_id)));
  }

  // no execution id
  {
    Association association = base_association;
    association.clear_execution_id();
    int64_t association_id;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_access_object_->CreateAssociation(
            association, &association_id)));
  }

  // the context cannot be found
  {
    Association association = base_association;
    int64_t unknown_id = 12345;
    association.set_context_id(unknown_id);
    int64_t association_id;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_access_object_->CreateAssociation(
            association, &association_id)));
  }

  // the execution cannot be found

  {
    Association association = base_association;
    int64_t unknown_id = 12345;
    association.set_execution_id(unknown_id);
    int64_t association_id;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_access_object_->CreateAssociation(
            association, &association_id)));
  }
}

// TODO(b/197686185): Remove test once foreign keys schema is implemented for
// CreateAssociation
TEST_P(MetadataAccessObjectTest, CreateAssociationWithoutValidation) {
  ASSERT_EQ(Init(), absl::OkStatus());

  int64_t context_type_id = InsertType<ContextType>("test_context_type");
  Context context;
  context.set_type_id(context_type_id);
  context.set_name("test_context");
  int64_t context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());

  int64_t execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  int64_t execution_id;
  ASSERT_EQ(metadata_access_object_->CreateExecution(execution, &execution_id),
            absl::OkStatus());

  // create association without validation (since the nodes are known to exist)
  Association association;
  association.set_context_id(context_id);
  association.set_execution_id(execution_id);
  int64_t association_id;
  absl::Status create_new_association_without_validation_status =
      metadata_access_object_->CreateAssociation(
          association, /*is_already_validated=*/true, &association_id);
  EXPECT_EQ(create_new_association_without_validation_status, absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // create duplicate association without validation
  int64_t duplicate_association_id;
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
  int64_t invalid_association_id;
  absl::Status create_invalid_association_without_validation_status =
      metadata_access_object_->CreateAssociation(invalid_association,
                                                 /*is_already_validated=*/true,
                                                 &invalid_association_id);
  EXPECT_EQ(create_invalid_association_without_validation_status,
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateAttributionError) {
  ASSERT_EQ(Init(), absl::OkStatus());

  // Create base attribution with
  // * valid context id
  // * valid artifact id
  int64_t context_type_id = InsertType<ContextType>("test_context_type");
  Context context;
  context.set_type_id(context_type_id);
  context.set_name("test_context");
  int64_t context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());

  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  Artifact artifact;
  artifact.set_type_id(artifact_type_id);
  int64_t artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());

  Attribution base_attribution;
  base_attribution.set_context_id(context_id);
  base_attribution.set_artifact_id(artifact_id);

  // no context id
  {
    Attribution attribution = base_attribution;
    attribution.clear_context_id();
    int64_t attribution_id;
    absl::Status s = metadata_access_object_->CreateAttribution(
        attribution, &attribution_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // no artifact id
  {
    Attribution attribution = base_attribution;
    attribution.clear_artifact_id();
    int64_t attribution_id;
    absl::Status s = metadata_access_object_->CreateAttribution(
        attribution, &attribution_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // the context cannot be found
  {
    Attribution attribution = base_attribution;
    int64_t unknown_id = 12345;
    attribution.set_context_id(unknown_id);
    int64_t attribution_id;
    absl::Status s = metadata_access_object_->CreateAttribution(
        attribution, &attribution_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // the artifact cannot be found
  {
    Attribution attribution = base_attribution;
    int64_t unknown_id = 12345;
    attribution.set_artifact_id(unknown_id);
    int64_t attribution_id;
    absl::Status s = metadata_access_object_->CreateAttribution(
        attribution, &attribution_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }
}

// TODO(b/197686185): Remove test once foreign keys schema is implemented for
// CreateAttribution
TEST_P(MetadataAccessObjectTest, CreateAttributionWithoutValidation) {
  ASSERT_EQ(Init(), absl::OkStatus());

  int64_t context_type_id = InsertType<ContextType>("test_context_type");
  Context context;
  context.set_type_id(context_type_id);
  context.set_name("test_context");
  int64_t context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());

  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  Artifact artifact;
  artifact.set_type_id(artifact_type_id);
  int64_t artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());

  // create attribution without validation (since the nodes are known to exist)
  Attribution attribution;
  attribution.set_context_id(context_id);
  attribution.set_artifact_id(artifact_id);
  int64_t attribution_id;
  absl::Status create_new_attribution_without_validation_status =
      metadata_access_object_->CreateAttribution(
          attribution, /*is_already_validated=*/true, &attribution_id);
  EXPECT_EQ(create_new_attribution_without_validation_status, absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // create duplicate attribution without validation
  int64_t duplicate_attribution_id;
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
  int64_t invalid_attribution_id;
  absl::Status create_invalid_attribution_without_validation_status =
      metadata_access_object_->CreateAttribution(invalid_attribution,
                                                 /*is_already_validated=*/true,
                                                 &invalid_attribution_id);
  EXPECT_EQ(create_invalid_attribution_without_validation_status,
            absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateAssociationError2) {
  ASSERT_EQ(Init(), absl::OkStatus());
  Association association;
  int64_t association_id;
  // duplicated association
  int64_t execution_type_id = InsertType<ExecutionType>("execution_type");
  int64_t context_type_id = InsertType<ContextType>("context_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  Context context = ParseTextProtoOrDie<Context>("name: 'context_instance'");
  context.set_type_id(context_type_id);
  int64_t execution_id, context_id;
  ASSERT_EQ(metadata_access_object_->CreateExecution(execution, &execution_id),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());
  association.set_execution_id(execution_id);
  association.set_context_id(context_id);
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // first insertion succeeds
  EXPECT_EQ(
      metadata_access_object_->CreateAssociation(association, &association_id),
      absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  // second insertion fails
  EXPECT_TRUE(absl::IsAlreadyExists(metadata_access_object_->CreateAssociation(
      association, &association_id)));
  ASSERT_EQ(metadata_source_->Rollback(), absl::OkStatus());
  ASSERT_EQ(metadata_source_->Begin(), absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, CreateAndUseAttribution) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64_t context_type_id = InsertType<ContextType>("test_context_type");

  Artifact artifact;
  artifact.set_uri("testuri");
  artifact.set_type_id(artifact_type_id);
  (*artifact.mutable_custom_properties())["custom"].set_string_value("str");
  Context context = ParseTextProtoOrDie<Context>("name: 'context_instance'");
  context.set_type_id(context_type_id);

  int64_t artifact_id, context_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());
  artifact.set_id(artifact_id);
  ASSERT_EQ(metadata_access_object_->CreateContext(context, &context_id),
            absl::OkStatus());
  context.set_id(context_id);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  Attribution attribution;
  attribution.set_artifact_id(artifact_id);
  attribution.set_context_id(context_id);

  int64_t attribution_id;
  EXPECT_EQ(
      metadata_access_object_->CreateAttribution(attribution, &attribution_id),
      absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  std::vector<Context> got_contexts;
  EXPECT_EQ(metadata_access_object_->FindContextsByArtifact(artifact_id,
                                                            &got_contexts),
            absl::OkStatus());
  ASSERT_EQ(got_contexts.size(), 1);
  EXPECT_THAT(context, EqualsProto(got_contexts[0], /*ignore_fields=*/{
                                       "type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));

  std::vector<Artifact> got_artifacts;
  EXPECT_EQ(metadata_access_object_->FindArtifactsByContext(context_id,
                                                            &got_artifacts),
            absl::OkStatus());
  ASSERT_EQ(got_artifacts.size(), 1);
  EXPECT_THAT(artifact, EqualsProto(got_artifacts[0], /*ignore_fields=*/{
                                        "type", "create_time_since_epoch",
                                        "last_update_time_since_epoch"}));

  std::vector<Execution> got_executions;
  EXPECT_EQ(metadata_access_object_->FindExecutionsByContext(context_id,
                                                             &got_executions),
            absl::OkStatus());
  EXPECT_EQ(got_executions.size(), 0);
}

TEST_P(MetadataAccessObjectTest, CreateAndFindEvent) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64_t execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Artifact input_artifact;
  input_artifact.set_type_id(artifact_type_id);
  int64_t input_artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(input_artifact,
                                                    &input_artifact_id),
            absl::OkStatus());

  Artifact output_artifact;
  output_artifact.set_type_id(artifact_type_id);
  int64_t output_artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(output_artifact,
                                                    &output_artifact_id),
            absl::OkStatus());

  Execution execution;
  execution.set_type_id(execution_type_id);
  int64_t execution_id;
  ASSERT_EQ(metadata_access_object_->CreateExecution(execution, &execution_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // event1 with event paths
  Event event1 = ParseTextProtoOrDie<Event>("type: INPUT");
  event1.set_artifact_id(input_artifact_id);
  event1.set_execution_id(execution_id);
  event1.set_milliseconds_since_epoch(12345);
  event1.mutable_path()->add_steps()->set_index(1);
  event1.mutable_path()->add_steps()->set_key("key");
  int64_t event1_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event1, &event1_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // event2 with optional fields
  Event event2 = ParseTextProtoOrDie<Event>("type: OUTPUT");
  event2.set_artifact_id(output_artifact_id);
  event2.set_execution_id(execution_id);
  int64_t event2_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event2, &event2_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  EXPECT_NE(event1_id, -1);
  EXPECT_NE(event2_id, -1);
  EXPECT_NE(event1_id, event2_id);

  // query the events
  std::vector<Event> events_with_artifacts;
  EXPECT_EQ(
      metadata_access_object_->FindEventsByArtifacts(
          {input_artifact_id, output_artifact_id}, &events_with_artifacts),
      absl::OkStatus());
  EXPECT_EQ(events_with_artifacts.size(), 2);
  EXPECT_THAT(
      events_with_artifacts,
      UnorderedElementsAre(
          EqualsProto(event1),
          EqualsProto(event2, /*ignore_fields=*/{"milliseconds_since_epoch"})));

  std::vector<Event> events_with_execution;
  EXPECT_EQ(metadata_access_object_->FindEventsByExecutions(
                {execution_id}, &events_with_execution),
            absl::OkStatus());
  EXPECT_EQ(events_with_execution.size(), 2);
}

TEST_P(MetadataAccessObjectTest, FindEventsByArtifactsNotFound) {
  ASSERT_EQ(Init(), absl::OkStatus());
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
  ASSERT_EQ(Init(), absl::OkStatus());
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
  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  Artifact artifact;
  artifact.set_type_id(artifact_type_id);
  int64_t artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());

  int64_t execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  int64_t execution_id;
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
    int64_t event_id;
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // no execution id
  {
    Event event = base_event;
    event.clear_execution_id();
    int64_t event_id;
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // no event type
  {
    Event event = base_event;
    event.clear_type();
    int64_t event_id;
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // artifact cannot be found
  {
    Event event = base_event;
    int64_t unknown_id = 12345;
    int64_t event_id;
    event.set_artifact_id(unknown_id);
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }

  // execution cannot be found
  {
    Event event = base_event;
    int64_t unknown_id = 12345;
    int64_t event_id;
    event.set_execution_id(unknown_id);
    absl::Status s = metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
  }
}

// TODO(b/197686185): Remove test once foreign keys schema is implemented for
// CreateEvent
TEST_P(MetadataAccessObjectTest, CreateEventWithoutValidation) {
  ASSERT_EQ(Init(), absl::OkStatus());

  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  Artifact artifact;
  artifact.set_type_id(artifact_type_id);
  int64_t artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(artifact, &artifact_id),
            absl::OkStatus());

  int64_t execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Execution execution;
  execution.set_type_id(execution_type_id);
  int64_t execution_id;
  ASSERT_EQ(metadata_access_object_->CreateExecution(execution, &execution_id),
            absl::OkStatus());

  // insert event without validating (since the nodes are known to exist)
  Event event;
  event.set_artifact_id(artifact_id);
  event.set_execution_id(execution_id);
  event.set_type(Event::INPUT);
  int64_t event_id;
  absl::Status create_new_event_without_validation_status =
      metadata_access_object_->CreateEvent(event, /*is_already_validated=*/true,
                                           &event_id);
  EXPECT_EQ(create_new_event_without_validation_status, absl::OkStatus());

  // insert invalid event without validation
  // NOTE: This is an invalid use case, but is intended to break once foreign
  // key support is implemented in the schema.
  Event invalid_event;
  int64_t invalid_event_id;
  invalid_event.set_artifact_id(artifact_id + 1);
  invalid_event.set_execution_id(execution_id + 1);
  invalid_event.set_type(Event::INPUT);
  absl::Status create_invalid_event_without_validation_status =
      metadata_access_object_->CreateEvent(
          invalid_event, /*is_already_validated=*/true, &invalid_event_id);
  EXPECT_EQ(create_invalid_event_without_validation_status, absl::OkStatus());
}

TEST_P(MetadataAccessObjectTest, PutEventsWithPaths) {
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64_t execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Artifact input_artifact;
  input_artifact.set_type_id(artifact_type_id);
  int64_t input_artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(input_artifact,
                                                    &input_artifact_id),
            absl::OkStatus());

  Artifact output_artifact;
  output_artifact.set_type_id(artifact_type_id);
  int64_t output_artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(output_artifact,
                                                    &output_artifact_id),
            absl::OkStatus());

  Execution execution;
  execution.set_type_id(execution_type_id);
  int64_t execution_id;
  ASSERT_EQ(metadata_access_object_->CreateExecution(execution, &execution_id),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // event1 with event paths
  Event event1 = ParseTextProtoOrDie<Event>("type: INPUT");
  event1.set_artifact_id(input_artifact_id);
  event1.set_execution_id(execution_id);
  event1.set_milliseconds_since_epoch(12345);
  event1.mutable_path()->add_steps()->set_index(1);
  event1.mutable_path()->add_steps()->set_key("key");
  int64_t event1_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event1, &event1_id),
            absl::OkStatus());

  // event2 with optional fields
  Event event2 = ParseTextProtoOrDie<Event>("type: OUTPUT");
  event2.set_artifact_id(output_artifact_id);
  event2.set_execution_id(execution_id);
  event2.mutable_path()->add_steps()->set_index(2);
  event2.mutable_path()->add_steps()->set_key("output_key");

  int64_t event2_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event2, &event2_id),
            absl::OkStatus());

  EXPECT_NE(event1_id, -1);
  EXPECT_NE(event2_id, -1);
  EXPECT_NE(event1_id, event2_id);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // query the events
  std::vector<Event> events_with_artifacts;
  EXPECT_EQ(
      metadata_access_object_->FindEventsByArtifacts(
          {input_artifact_id, output_artifact_id}, &events_with_artifacts),
      absl::OkStatus());
  EXPECT_EQ(events_with_artifacts.size(), 2);
  EXPECT_THAT(
      events_with_artifacts,
      UnorderedElementsAre(
          EqualsProto(event1),
          EqualsProto(event2, /*ignore_fields=*/{"milliseconds_since_epoch"})));

  std::vector<Event> events_with_execution;
  EXPECT_EQ(metadata_access_object_->FindEventsByExecutions(
                {execution_id}, &events_with_execution),
            absl::OkStatus());
  EXPECT_EQ(events_with_execution.size(), 2);
}

TEST_P(MetadataAccessObjectTest, CreateDuplicatedEvents) {
  // Support after Spanner upgrade schema to V8.
  if (!metadata_access_object_container_->HasFilterQuerySupport()) {
    return;
  }
  ASSERT_EQ(Init(), absl::OkStatus());
  int64_t artifact_type_id = InsertType<ArtifactType>("test_artifact_type");
  int64_t execution_type_id = InsertType<ExecutionType>("test_execution_type");
  Artifact input_artifact;
  input_artifact.set_type_id(artifact_type_id);
  int64_t input_artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(input_artifact,
                                                    &input_artifact_id),
            absl::OkStatus());

  Artifact output_artifact;
  output_artifact.set_type_id(artifact_type_id);
  int64_t output_artifact_id;
  ASSERT_EQ(metadata_access_object_->CreateArtifact(output_artifact,
                                                    &output_artifact_id),
            absl::OkStatus());

  Execution execution;
  execution.set_type_id(execution_type_id);
  int64_t execution_id;
  ASSERT_EQ(metadata_access_object_->CreateExecution(execution, &execution_id),
            absl::OkStatus());

  // event1 with event paths
  Event event1 = ParseTextProtoOrDie<Event>("type: INPUT");
  event1.set_artifact_id(input_artifact_id);
  event1.set_execution_id(execution_id);
  event1.set_milliseconds_since_epoch(12345);
  event1.mutable_path()->add_steps()->set_index(1);
  event1.mutable_path()->add_steps()->set_key("key");
  int64_t event1_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event1, &event1_id),
            absl::OkStatus());
  EXPECT_NE(event1_id, -1);

  // event2 with same artifact_id, execution_id but different type.
  Event event2 = ParseTextProtoOrDie<Event>("type: DECLARED_INPUT");
  event2.set_artifact_id(input_artifact_id);
  event2.set_execution_id(execution_id);
  int64_t event2_id = -1;
  EXPECT_EQ(metadata_access_object_->CreateEvent(event2, &event2_id),
            absl::OkStatus());
  EXPECT_NE(event2_id, -1);
  EXPECT_NE(event1_id, event2_id);

  // event3 with same artifact_id, execution_id and type.
  Event event3 = ParseTextProtoOrDie<Event>("type: INPUT");
  event3.set_artifact_id(input_artifact_id);
  event3.set_execution_id(execution_id);
  int64_t unused_event3_id = -1;
  // TODO(b/248836219): Cleanup the fat-client after fully migrated to V9+.
  // At schema version 7, the unique constraint on Event table on (artifact_id,
  // execution_id, type) is not introduced.
  if (metadata_access_object_container_->GetSchemaVersion() == 7) {
    EXPECT_EQ(metadata_access_object_->CreateEvent(event3, &unused_event3_id),
              absl::OkStatus());
  } else {
    EXPECT_TRUE(absl::IsAlreadyExists(
        metadata_access_object_->CreateEvent(event3, &unused_event3_id)));
  }

  // query the events
  std::vector<Event> events_with_artifacts;
  EXPECT_EQ(metadata_access_object_->FindEventsByArtifacts(
                {input_artifact_id}, &events_with_artifacts),
            absl::OkStatus());
  if (metadata_access_object_container_->GetSchemaVersion() == 7) {
    EXPECT_EQ(events_with_artifacts.size(), 3);
    EXPECT_THAT(
        events_with_artifacts,
        UnorderedElementsAre(
            EqualsProto(event1),
            EqualsProto(event2, /*ignore_fields=*/{"milliseconds_since_epoch"}),
            EqualsProto(event3,
                        /*ignore_fields=*/{"milliseconds_since_epoch"})));
  } else {
    EXPECT_EQ(events_with_artifacts.size(), 2);
    EXPECT_THAT(
        events_with_artifacts,
        UnorderedElementsAre(EqualsProto(event1),
                             EqualsProto(event2, /*ignore_fields=*/{
                                             "milliseconds_since_epoch"})));
  }

  std::vector<Event> events_with_execution;
  EXPECT_EQ(metadata_access_object_->FindEventsByExecutions(
                {execution_id}, &events_with_execution),
            absl::OkStatus());
  if (metadata_access_object_container_->GetSchemaVersion() == 7) {
    EXPECT_EQ(events_with_execution.size(), 3);
    EXPECT_THAT(
        events_with_execution,
        UnorderedElementsAre(
            EqualsProto(event1),
            EqualsProto(event2, /*ignore_fields=*/{"milliseconds_since_epoch"}),
            EqualsProto(event3,
                        /*ignore_fields=*/{"milliseconds_since_epoch"})));
  } else {
    EXPECT_EQ(events_with_execution.size(), 2);
    EXPECT_THAT(
        events_with_execution,
        UnorderedElementsAre(EqualsProto(event1),
                             EqualsProto(event2, /*ignore_fields=*/{
                                             "milliseconds_since_epoch"})));
  }
}

TEST_P(MetadataAccessObjectTest, CreateParentContext) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType context_type;
  context_type.set_name("context_type_name");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(context_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  Context context1, context2;
  context1.set_name("parent_context");
  context1.set_type_id(type_id);
  int64_t context1_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context1, &context1_id),
            absl::OkStatus());
  context2.set_name("child_context");
  context2.set_type_id(type_id);
  int64_t context2_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context2, &context2_id),
            absl::OkStatus());

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  ParentContext parent_context;
  parent_context.set_parent_id(context1_id);
  parent_context.set_child_id(context2_id);
  EXPECT_EQ(metadata_access_object_->CreateParentContext(parent_context),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // recreate the same context returns AlreadyExists
  const absl::Status status =
      metadata_access_object_->CreateParentContext(parent_context);
  EXPECT_TRUE(absl::IsAlreadyExists(status));
}

TEST_P(MetadataAccessObjectTest, CreateParentContextInvalidArgumentError) {
  // Prepare a stored context.
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType context_type;
  context_type.set_name("context_type_name");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(context_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  Context context;
  context.set_name("parent_context");
  context.set_type_id(type_id);
  int64_t stored_context_id;
  ASSERT_EQ(metadata_access_object_->CreateContext(context, &stored_context_id),
            absl::OkStatus());
  int64_t not_exist_context_id = stored_context_id + 1;
  int64_t not_exist_context_id_2 = stored_context_id + 2;

  // Enumerate the case of parent context requests which are invalid
  auto verify_is_invalid_argument = [this](absl::string_view case_name,
                                           std::optional<int64_t> parent_id,
                                           std::optional<int64_t> child_id) {
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
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType context_type;
  context_type.set_name("context_type_name");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(context_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  // Create some contexts to insert parent context relationship.
  const int num_contexts = 5;
  std::vector<Context> contexts(num_contexts);
  for (int i = 0; i < num_contexts; i++) {
    contexts[i].set_name(absl::StrCat("context", i));
    contexts[i].set_type_id(type_id);
    int64_t ctx_id;
    ASSERT_EQ(metadata_access_object_->CreateContext(contexts[i], &ctx_id),
              absl::OkStatus());
    contexts[i].set_id(ctx_id);
  }

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Populate a list of parent contexts and capture expected results of number
  // of parents and children per context.
  absl::node_hash_map<int, std::vector<Context>> want_parents;
  std::unordered_map<int, std::vector<Context>> want_children;
  auto put_parent_context = [this, &contexts, &want_parents, &want_children](
                                int64_t parent_idx, int64_t child_idx) {
    ParentContext parent_context;
    parent_context.set_parent_id(contexts[parent_idx].id());
    parent_context.set_child_id(contexts[child_idx].id());
    ASSERT_EQ(metadata_access_object_->CreateParentContext(parent_context),
              absl::OkStatus());
    want_parents[child_idx].push_back(contexts[parent_idx]);
    want_children[parent_idx].push_back(contexts[child_idx]);
  };
  put_parent_context(/*parent_idx=*/0, /*child_idx=*/1);
  put_parent_context(/*parent_idx=*/0, /*child_idx=*/2);
  put_parent_context(/*parent_idx=*/0, /*child_idx=*/3);
  put_parent_context(/*parent_idx=*/2, /*child_idx=*/3);
  put_parent_context(/*parent_idx=*/4, /*child_idx=*/3);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Verify the results by look up contexts
  for (int i = 0; i < num_contexts; i++) {
    const Context& curr_context = contexts[i];
    std::vector<Context> got_parents, got_children;
    EXPECT_EQ(metadata_access_object_->FindParentContextsByContextId(
                  curr_context.id(), &got_parents),
              absl::OkStatus());
    EXPECT_EQ(metadata_access_object_->FindChildContextsByContextId(
                  curr_context.id(), &got_children),
              absl::OkStatus());
    EXPECT_THAT(got_parents, SizeIs(want_parents[i].size()));
    EXPECT_THAT(got_children, SizeIs(want_children[i].size()));
    EXPECT_THAT(got_parents,
                UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                                       "type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}),
                                   want_parents[i]));
  }
}

TEST_P(MetadataAccessObjectTest, FindParentandChildContextsByContextIds) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType context_type;
  context_type.set_name("context_type_name");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(context_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  // Setup: create some contexts to insert parent context relationship.
  const int num_contexts = 5;
  std::vector<Context> contexts(num_contexts);
  std::vector<int64_t> context_ids;
  for (int i = 0; i < num_contexts; i++) {
    contexts[i].set_name(absl::StrCat("context", i));
    contexts[i].set_type_id(type_id);
    int64_t ctx_id;
    ASSERT_EQ(metadata_access_object_->CreateContext(contexts[i], &ctx_id),
              absl::OkStatus());
    contexts[i].set_id(ctx_id);
    context_ids.push_back(ctx_id);
  }

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Setup: populate a list of parent contexts and capture expected results of
  // number of parents and children per context.
  absl::node_hash_map<int64_t, std::vector<Context>> want_parents,
      want_children;
  auto put_parent_context = [this, &contexts, &want_parents, &want_children](
                                int64_t parent_idx, int64_t child_idx) {
    ParentContext parent_context;
    parent_context.set_parent_id(contexts[parent_idx].id());
    parent_context.set_child_id(contexts[child_idx].id());
    ASSERT_EQ(metadata_access_object_->CreateParentContext(parent_context),
              absl::OkStatus());
    want_parents[contexts[child_idx].id()].push_back(contexts[parent_idx]);
    want_children[contexts[parent_idx].id()].push_back(contexts[child_idx]);
  };
  put_parent_context(/*parent_idx=*/0, /*child_idx=*/1);
  put_parent_context(/*parent_idx=*/0, /*child_idx=*/2);
  put_parent_context(/*parent_idx=*/0, /*child_idx=*/3);
  put_parent_context(/*parent_idx=*/2, /*child_idx=*/3);
  put_parent_context(/*parent_idx=*/4, /*child_idx=*/3);

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  {
    // Act: call FindParent/ChildContextsByContextIds with empty context_ids.
    absl::node_hash_map<int64_t, std::vector<Context>> got_children,
        got_parents;
    absl::Status status =
        metadata_access_object_->FindChildContextsByContextIds({},
                                                               got_children);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
    status = metadata_access_object_->FindParentContextsByContextIds(
        {}, got_parents);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }
  {
    // Act: call FindParent/ChildContextsByContextIds on all the context_ids.
    absl::node_hash_map<int64_t, std::vector<Context>> got_children,
        got_parents;
    ASSERT_EQ(metadata_access_object_->FindChildContextsByContextIds(
                  context_ids, got_children),
              absl::OkStatus());
    ASSERT_EQ(metadata_access_object_->FindParentContextsByContextIds(
                  context_ids, got_parents),
              absl::OkStatus());

    // Verify the results
    ASSERT_THAT(got_parents, SizeIs(want_parents.size()));
    for (const int64_t context_id : context_ids) {
      ASSERT_EQ(got_parents.contains(context_id),
                want_parents.contains(context_id));
      if (!got_parents.contains(context_id)) continue;
      EXPECT_THAT(got_parents.at(context_id),
                  UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                                         "type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"}),
                                     want_parents[context_id]));
    }
    ASSERT_THAT(got_children, SizeIs(want_children.size()));
    for (const int64_t context_id : context_ids) {
      ASSERT_EQ(got_children.contains(context_id),
                want_children.contains(context_id));
      if (!got_children.contains(context_id)) continue;
      EXPECT_THAT(got_children.at(context_id),
                  UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                                         "type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"}),
                                     want_children[context_id]));
    }
  }
}

TEST_P(MetadataAccessObjectTest, CreateParentContextInheritanceLinkWithCycle) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ContextType context_type;
  context_type.set_name("context_type_name");
  int64_t type_id;
  ASSERT_EQ(metadata_access_object_->CreateType(context_type, &type_id),
            absl::OkStatus());
  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());
  // Creates some contexts for parent context relationship.
  const int num_contexts = 5;
  std::vector<Context> contexts(num_contexts);
  for (int i = 0; i < num_contexts; i++) {
    contexts[i].set_name(absl::StrCat("context", i));
    contexts[i].set_type_id(type_id);
    int64_t ctx_id;
    ASSERT_EQ(metadata_access_object_->CreateContext(contexts[i], &ctx_id),
              absl::OkStatus());
    contexts[i].set_id(ctx_id);
  }

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

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

  ASSERT_EQ(AddCommitPointIfNeeded(), absl::OkStatus());

  // Cannot add self as parent context.
  verify_insert_parent_context_is_invalid(
      /*parent=*/contexts[0], /*child=*/contexts[0]);

  // context0 -> context1
  ASSERT_EQ(metadata_access_object_->CreateParentContext(
                set_and_return_parent_context(
                    /*parent_id=*/contexts[0].id(),
                    /*child_id=*/contexts[1].id())),
            absl::OkStatus());

  // Cannot have bi-direction parent context.
  verify_insert_parent_context_is_invalid(
      /*parent=*/contexts[1], /*child=*/contexts[0]);

  // context0 -> context1 -> context2
  //         \-> context3 -> context4
  ASSERT_EQ(metadata_access_object_->CreateParentContext(
                set_and_return_parent_context(
                    /*parent_id=*/contexts[1].id(),
                    /*child_id=*/contexts[2].id())),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_->CreateParentContext(
                set_and_return_parent_context(
                    /*parent_id=*/contexts[0].id(),
                    /*child_id=*/contexts[3].id())),
            absl::OkStatus());
  ASSERT_EQ(metadata_access_object_->CreateParentContext(
                set_and_return_parent_context(
                    /*parent_id=*/contexts[3].id(),
                    /*child_id=*/contexts[4].id())),
            absl::OkStatus());

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
  int64_t lib_version = metadata_access_object_->GetLibraryVersion();
  for (int64_t i = metadata_access_object_container_->MinimumVersion();
       i <= lib_version; i++) {
    if (!metadata_access_object_container_->HasUpgradeVerification(i)) {
      continue;
    }
    MLMD_ASSERT_OK(
        metadata_access_object_container_->SetupPreviousVersionForUpgrade(i));
    if (i > 1) continue;
    // when i = 0, it is v0.13.2. At that time, the MLMDEnv table does not
    // exist, GetSchemaVersion resolves the current version as 0.
    int64_t v0_13_2_version = 100;
    ASSERT_EQ(metadata_access_object_->GetSchemaVersion(&v0_13_2_version),
              absl::OkStatus());
    ASSERT_EQ(0, v0_13_2_version);
  }

  // expect to have an error when connecting an older database version without
  // enabling upgrade migration
  absl::Status status =
      metadata_access_object_->InitMetadataSourceIfNotExists();
  ASSERT_TRUE(absl::IsFailedPrecondition(status))
      << "Error: " << status.message();

  ASSERT_EQ(metadata_source_->Commit(), absl::OkStatus());
  ASSERT_EQ(metadata_source_->Begin(), absl::OkStatus());

  // then init the store and the migration queries runs.
  ASSERT_EQ(metadata_access_object_->InitMetadataSourceIfNotExists(
                /*enable_upgrade_migration=*/true),
            absl::OkStatus());
  // at the end state, schema version should becomes the library version and
  // all migration queries should all succeed.
  int64_t curr_version = 0;
  ASSERT_EQ(metadata_access_object_->GetSchemaVersion(&curr_version),
            absl::OkStatus());
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
  EXPECT_EQ(metadata_access_object_->InitMetadataSourceIfNotExists(),
            absl::OkStatus());
  ASSERT_EQ(metadata_source_->Commit(), absl::OkStatus());
  ASSERT_EQ(metadata_source_->Begin(), absl::OkStatus());
  int64_t lib_version = metadata_access_object_->GetLibraryVersion();
  MLMD_ASSERT_OK(
      metadata_access_object_container_->VerifyDbSchema(lib_version));
  int64_t curr_version = 0;
  EXPECT_EQ(metadata_access_object_->GetSchemaVersion(&curr_version),
            absl::OkStatus());
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
    EXPECT_EQ(metadata_access_object_->GetSchemaVersion(&curr_version),
              absl::OkStatus());
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
  ASSERT_EQ(metadata_access_object_->InitMetadataSourceIfNotExists(),
            absl::OkStatus());
  ASSERT_EQ(metadata_source_->Commit(), absl::OkStatus());
  ASSERT_EQ(metadata_source_->Begin(), absl::OkStatus());
  // downgrade when the database to version 0.
  int64_t current_library_version =
      metadata_access_object_->GetLibraryVersion();
  if (current_library_version ==
      metadata_access_object_container_->MinimumVersion()) {
    return;
  }
  const int64_t to_schema_version = current_library_version - 1;
  ASSERT_EQ(metadata_access_object_->DowngradeMetadataSource(to_schema_version),
            absl::OkStatus());
  int64_t db_version = -1;
  ASSERT_EQ(metadata_access_object_->GetSchemaVersion(&db_version),
            absl::OkStatus());
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
