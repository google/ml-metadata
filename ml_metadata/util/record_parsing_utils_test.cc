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
#include "ml_metadata/util/record_parsing_utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {
namespace testing {
namespace {

using ::ml_metadata::testing::EqualsProto;
using ::ml_metadata::testing::ParseTextProtoOrDie;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(ParseRecordSetTest, ParseRecordSetToArtifactArraySuccess) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(
        column_names: 'id'
        column_names: 'type_id'
        column_names: 'uri'
        column_names: 'state'
        column_names: 'name'
        column_names: 'external_id'
        column_names: 'create_time_since_epoch'
        column_names: 'last_update_time_since_epoch'
        records {
          values: '1'
          values: '1'
          values: '/fake/uri'
          values: '1'
          values: '1a'
          values: '__MLMD_NULL__'
          values: '1677288915393'
          values: '1677288915393'
        }
        records {
          values: '2'
          values: '1'
          values: ''
          values: '__MLMD_NULL__'
          values: '1b'
          values: 'test_id'
          values: '1677288915393'
          values: '1677288915393'
        }
      )pb");

  std::vector<Artifact> artifacts;
  absl::Status status = ParseRecordSetToNodeArray(record_set, artifacts);
  EXPECT_EQ(status, absl::OkStatus());
  EXPECT_THAT(artifacts,
              ElementsAre(EqualsProto(ParseTextProtoOrDie<Artifact>(R"pb(
                            id: 1
                            type_id: 1
                            uri: '/fake/uri'
                            state: PENDING
                            name: '1a'
                            create_time_since_epoch: 1677288915393
                            last_update_time_since_epoch: 1677288915393
                          )pb")),
                          EqualsProto(ParseTextProtoOrDie<Artifact>(R"pb(
                            id: 2
                            type_id: 1
                            uri: ''
                            name: '1b'
                            external_id: 'test_id'
                            create_time_since_epoch: 1677288915393
                            last_update_time_since_epoch: 1677288915393
                          )pb"))));
}

TEST(ParseRecordSetTest, ParseRecordSetToExecutionArraySuccess) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(
        column_names: 'id'
        column_names: 'type_id'
        column_names: 'last_known_state'
        column_names: 'name'
        column_names: 'external_id'
        column_names: 'create_time_since_epoch'
        column_names: 'last_update_time_since_epoch'
        records {
          values: '1'
          values: '2'
          values: '2'
          values: 'excution_name'
          values: '__MLMD_NULL__'
          values: '1677288907995'
          values: '1677288907995'
        }
        records {
          values: '2'
          values: '2'
          values: '__MLMD_NULL__'
          values: '__MLMD_NULL__'
          values: 'test_external_id'
          values: '1677288907996'
          values: '1677288907996'
        }
      )pb");

  std::vector<Execution> executions;
  absl::Status status = ParseRecordSetToNodeArray(record_set, executions);
  EXPECT_EQ(status, absl::OkStatus());
  EXPECT_THAT(executions,
              ElementsAre(EqualsProto(ParseTextProtoOrDie<Execution>(R"pb(
                            id: 1
                            type_id: 2
                            name: 'excution_name'
                            last_known_state: RUNNING
                            create_time_since_epoch: 1677288907995
                            last_update_time_since_epoch: 1677288907995
                          )pb")),
                          EqualsProto(ParseTextProtoOrDie<Execution>(R"pb(
                            id: 2
                            type_id: 2
                            create_time_since_epoch: 1677288907996
                            last_update_time_since_epoch: 1677288907996
                            external_id: 'test_external_id'
                          )pb"))));
}

TEST(ParseRecordSetTest, ParseRecordSetToContextArraySuccess) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(
        column_names: 'id'
        column_names: 'type_id'
        column_names: 'name'
        column_names: 'external_id'
        column_names: 'create_time_since_epoch'
        column_names: 'last_update_time_since_epoch'
        records {
          values: '1'
          values: '1'
          values: 'delete_contexts_by_id_test_1'
          values: 'test_id'
          values: '1677288909912'
          values: '1677288909912'
        }
        records {
          values: '2'
          values: '2'
          values: '__MLMD_NULL__'
          values: '__MLMD_NULL__'
          values: '1677288907996'
          values: '1677288907996'
        }
      )pb");

  std::vector<Context> contexts;
  absl::Status status = ParseRecordSetToNodeArray(record_set, contexts);
  EXPECT_EQ(status, absl::OkStatus());
  EXPECT_THAT(contexts,
              ElementsAre(EqualsProto(ParseTextProtoOrDie<Context>(R"pb(
                            id: 1
                            type_id: 1
                            name: 'delete_contexts_by_id_test_1'
                            external_id: 'test_id'
                            create_time_since_epoch: 1677288909912
                            last_update_time_since_epoch: 1677288909912
                          )pb")),
                          EqualsProto(ParseTextProtoOrDie<Context>(R"pb(
                            id: 2
                            type_id: 2
                            create_time_since_epoch: 1677288907996
                            last_update_time_since_epoch: 1677288907996
                          )pb"))));
}

TEST(ParseRecordSetTest, ParseRecordSetToEventArraySuccess) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(

        column_names: 'id'
        column_names: 'artifact_id'
        column_names: 'execution_id'
        column_names: 'type'
        column_names: 'milliseconds_since_epoch'
        records {
          values: '1'
          values: '1'
          values: '1'
          values: '3'
          values: '12345'
        }
        records {
          values: '2'
          values: '2'
          values: '1'
          values: '4'
          values: '1677288953794'
        }
      )pb");

  std::vector<Event> events;
  absl::Status status = ParseRecordSetToEdgeArray(record_set, events);
  EXPECT_EQ(status, absl::OkStatus());
  EXPECT_THAT(events, ElementsAre(EqualsProto(ParseTextProtoOrDie<Event>(R"pb(
                                    artifact_id: 1
                                    execution_id: 1
                                    type: INPUT
                                    milliseconds_since_epoch: 12345
                                  )pb")),
                                  EqualsProto(ParseTextProtoOrDie<Event>(R"pb(
                                    artifact_id: 2
                                    execution_id: 1
                                    type: OUTPUT
                                    milliseconds_since_epoch: 1677288953794
                                  )pb"))));
}

TEST(ParseRecordSetTest, ParseRecordSetToAssociationArraySuccess) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(
        column_names: 'execution_id'
        column_names: 'context_id'
        records { values: '1' values: '1' }
        records { values: '2' values: '2' }
      )pb");

  std::vector<Association> associations;
  absl::Status status = ParseRecordSetToEdgeArray(record_set, associations);
  EXPECT_EQ(status, absl::OkStatus());
  EXPECT_THAT(associations,
              ElementsAre(EqualsProto(ParseTextProtoOrDie<Association>(R"pb(
                            execution_id: 1
                            context_id: 1
                          )pb")),
                          EqualsProto(ParseTextProtoOrDie<Association>(R"pb(
                            execution_id: 2
                            context_id: 2
                          )pb"))));
}

TEST(ParseRecordSetTest, ParseRecordSetToAttributionArraySuccess) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(
        column_names: 'artifact_id'
        column_names: 'context_id'
        records { values: '1' values: '1' }
        records { values: '2' values: '2' }
      )pb");

  std::vector<Attribution> attributions;
  absl::Status status = ParseRecordSetToEdgeArray(record_set, attributions);
  EXPECT_EQ(status, absl::OkStatus());
  EXPECT_THAT(attributions,
              ElementsAre(EqualsProto(ParseTextProtoOrDie<Attribution>(R"pb(
                            artifact_id: 1
                            context_id: 1
                          )pb")),
                          EqualsProto(ParseTextProtoOrDie<Attribution>(R"pb(
                            artifact_id: 2
                            context_id: 2
                          )pb"))));
}

TEST(ParseRecordSetTest, ParseRecordSetToParentContextArraySuccess) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(
        column_names: 'parent_id'
        column_names: 'child_id'
        records { values: '123' values: '321' }
        records { values: '456' values: '654' }
      )pb");

  std::vector<ParentContext> parent_contexts;
  absl::Status status = ParseRecordSetToEdgeArray(record_set, parent_contexts);
  EXPECT_EQ(status, absl::OkStatus());
  EXPECT_THAT(parent_contexts,
              ElementsAre(EqualsProto(ParseTextProtoOrDie<ParentContext>(R"pb(
                            parent_id: 123 child_id: 321
                          )pb")),
                          EqualsProto(ParseTextProtoOrDie<ParentContext>(R"pb(
                            parent_id: 456 child_id: 654
                          )pb"))));
}

TEST(ParseRecordSetTest, ParseRecordSetToParentContextArrayEmptySuccess) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(
        column_names: 'parent_id'
        column_names: 'child_id'
      )pb");

  std::vector<ParentContext> parent_contexts;
  absl::Status status = ParseRecordSetToEdgeArray(record_set, parent_contexts);
  EXPECT_EQ(status, absl::OkStatus());
  EXPECT_THAT(parent_contexts, IsEmpty());
}

TEST(ParseRecordSetTest, MismatchRecordAndFieldNameIgnored) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(
        column_names: 'id'
        column_names: 'invalid_field_name'
        records { values: '1' values: 'some_value' }
      )pb");

  std::vector<Artifact> artifacts;
  absl::Status status = ParseRecordSetToNodeArray(record_set, artifacts);
  EXPECT_EQ(status, absl::OkStatus());
  EXPECT_THAT(artifacts,
              ElementsAre(EqualsProto(ParseTextProtoOrDie<Artifact>(R"pb(
                id: 1
              )pb"))));
}

TEST(ParseRecordSetTest, ParseNodeRecordSetToDedupedArtifactTypeArraySuccess) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(
        column_names: 'id'
        column_names: 'type_id'
        column_names: 'uri'
        column_names: 'state'
        column_names: 'name'
        column_names: 'external_id'
        column_names: 'create_time_since_epoch'
        column_names: 'last_update_time_since_epoch'
        column_names: 'type'
        column_names: 'type_version'
        column_names: 'type_description'
        column_names: 'type_external_id'
        records {
          values: '1'
          values: '1'
          values: '/fake/uri'
          values: '1'
          values: '1a'
          values: '__MLMD_NULL__'
          values: '1677288915393'
          values: '1677288915393'
          values: 'artifact_type_1'
          values: 'v1'
          values: 'artifact_type_description'
          values: 'artifact_type_1'
        }
        records {
          values: '2'
          values: '1'
          values: ''
          values: '__MLMD_NULL__'
          values: '1b'
          values: 'test_id'
          values: '1677288915393'
          values: '1677288915393'
          values: 'artifact_type_1'
          values: 'v1'
          values: 'artifact_type_description'
          values: 'artifact_type_1'
        }
      )pb");
  std::vector<ArtifactType> artifact_types;
  EXPECT_EQ(ParseNodeRecordSetToDedupedTypes(record_set, artifact_types),
            absl::OkStatus());
  EXPECT_THAT(artifact_types,
              ElementsAre(EqualsProto(ParseTextProtoOrDie<ArtifactType>(R"pb(
                id: 1
                name: 'artifact_type_1'
                version: 'v1'
                description: 'artifact_type_description'
                external_id: 'artifact_type_1'
              )pb"))));
}

TEST(ParseRecordSetTest, ParseRecordSetToDedupedExecutionTypeArraySuccess) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(
        column_names: 'id'
        column_names: 'type_id'
        column_names: 'last_known_state'
        column_names: 'name'
        column_names: 'external_id'
        column_names: 'create_time_since_epoch'
        column_names: 'last_update_time_since_epoch'
        column_names: 'type'
        column_names: 'type_version'
        column_names: 'type_description'
        column_names: 'type_external_id'
        records {
          values: '1'
          values: '1'
          values: '2'
          values: 'excution_name'
          values: '__MLMD_NULL__'
          values: '1677288907995'
          values: '1677288907995'
          values: 'execution_type_1'
          values: 'v1'
          values: 'execution_type_description'
          values: 'execution_type_1'
        }
        records {
          values: '2'
          values: '2'
          values: '__MLMD_NULL__'
          values: '__MLMD_NULL__'
          values: 'test_external_id'
          values: '1677288907996'
          values: '1677288907996'
          values: 'execution_type_2'
          values: 'v1'
          values: 'execution_type_description'
          values: 'execution_type_2'
        }
      )pb");

  std::vector<ExecutionType> execution_types;
  EXPECT_EQ(ParseNodeRecordSetToDedupedTypes(record_set, execution_types),
            absl::OkStatus());
  EXPECT_THAT(execution_types,
              ElementsAre(EqualsProto(ParseTextProtoOrDie<ExecutionType>(R"pb(
                            id: 1
                            name: 'execution_type_1'
                            version: 'v1'
                            description: 'execution_type_description'
                            external_id: 'execution_type_1'
                          )pb")),
                          EqualsProto(ParseTextProtoOrDie<ExecutionType>(R"pb(
                            id: 2
                            name: 'execution_type_2'
                            version: 'v1'
                            description: 'execution_type_description'
                            external_id: 'execution_type_2'
                          )pb"))));
}

TEST(ParseRecordSetTest, ParseRecordSetToDedupedContextTypeArraySuccess) {
  RecordSet record_set = ParseTextProtoOrDie<RecordSet>(
      R"pb(
        column_names: 'id'
        column_names: 'type_id'
        column_names: 'name'
        column_names: 'external_id'
        column_names: 'create_time_since_epoch'
        column_names: 'last_update_time_since_epoch'
        column_names: 'type'
        column_names: 'type_version'
        column_names: 'type_description'
        column_names: 'type_external_id'
        records {
          values: '1'
          values: '1'
          values: 'delete_contexts_by_id_test_1'
          values: 'test_id'
          values: '1677288909912'
          values: '1677288909912'
          values: 'context_type_1'
          values: '__MLMD_NULL__'
          values: '__MLMD_NULL__'
          values: '__MLMD_NULL__'
        }
        records {
          values: '2'
          values: '1'
          values: '__MLMD_NULL__'
          values: '__MLMD_NULL__'
          values: '1677288907996'
          values: '1677288907996'
          values: 'context_type_1'
          values: '__MLMD_NULL__'
          values: '__MLMD_NULL__'
          values: '__MLMD_NULL__'
        }
      )pb");

  std::vector<ContextType> context_types;
  EXPECT_EQ(ParseNodeRecordSetToDedupedTypes(record_set, context_types),
            absl::OkStatus());
  EXPECT_THAT(context_types,
              ElementsAre(EqualsProto(ParseTextProtoOrDie<ContextType>(R"pb(
                id: 1
                name: 'context_type_1'
              )pb"))));
}

}  // namespace
}  // namespace testing
}  // namespace ml_metadata
