/* Copyright 2022 Google LLC

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
#include "ml_metadata/metadata_store/rdbms_metadata_access_object_test.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {

using ::ml_metadata::testing::EqualsProto;
using ::ml_metadata::testing::ParseTextProtoOrDie;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

constexpr absl::string_view kArtifactTypeRecordSet =
    R"pb(column_names: "id"
         column_names: "name"
         column_names: "type_kind"
         records: { values: "1" values: "artifact_type_1" values: "1" }
         records: { values: "2" values: "artifact_type_2" values: "1" }
    )pb";
constexpr absl::string_view kExecutionTypeRecordSet =
    R"pb(column_names: "id"
         column_names: "name"
         column_names: "type_kind"
         records: { values: "1" values: "execution_type_1" values: "0" }
         records: { values: "2" values: "execution_type_2" values: "0" }
    )pb";
constexpr absl::string_view kContextTypeRecordSet =
    R"pb(column_names: "id"
         column_names: "name"
         column_names: "type_kind"
         records: { values: "1" values: "context_type_1" values: "2" }
    )pb";

template <typename MessageType>
void RDBMSMetadataAccessObjectTest::VerifyFindTypesFromRecordSet(
    const RecordSet& records, std::vector<MessageType> expected_types) {
  // Test when `get_properties` is true.
  std::vector<MessageType> got_types;
  ASSERT_EQ(absl::OkStatus(), FindTypesFromRecordSet(records, &got_types,
                                                     /*get_properties=*/true));
  ASSERT_EQ(got_types.size(), expected_types.size());
  for (int i = 0; i < got_types.size(); ++i) {
    EXPECT_THAT(got_types[i], EqualsProto(expected_types[i]));
  }

  // Test when `get_properties` is false.
  got_types.clear();
  ASSERT_EQ(absl::OkStatus(), FindTypesFromRecordSet(records, &got_types,
                                                     /*get_properties=*/false));
  ASSERT_EQ(got_types.size(), expected_types.size());
  for (int i = 0; i < got_types.size(); ++i) {
    expected_types[i].clear_properties();
    EXPECT_THAT(got_types[i], EqualsProto(expected_types[i]));
  }
}

TEST_P(RDBMSMetadataAccessObjectTest, FindArtifactTypesFromRecordSet) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64_t type_id_1, type_id_2;
  ArtifactType type_1 = testing::ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'artifact_type_1'
    properties { key: 'property_1' value: STRING }
    properties { key: 'property_2' value: DOUBLE }
  )pb");
  ArtifactType type_2 = testing::ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'artifact_type_2'
    properties { key: 'property_1' value: INT }
  )pb");
  ASSERT_EQ(absl::OkStatus(), CreateType(type_1, &type_id_1));
  type_1.set_id(type_id_1);
  ASSERT_EQ(absl::OkStatus(), CreateType(type_2, &type_id_2));
  type_2.set_id(type_id_2);

  RecordSet records = testing::ParseTextProtoOrDie<RecordSet>(
      std::string(kArtifactTypeRecordSet));
  records.mutable_records(0)->set_values(0, std::to_string(type_id_1));
  records.mutable_records(1)->set_values(0, std::to_string(type_id_2));

  std::vector<ArtifactType> expected_types = {type_1, type_2};
  VerifyFindTypesFromRecordSet(records, expected_types);
}

TEST_P(RDBMSMetadataAccessObjectTest, FindExecutionTypesFromRecordSet) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64_t type_id_1, type_id_2;
  ExecutionType type_1 = testing::ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'execution_type_1'
    properties { key: 'property_1' value: STRING }
    properties { key: 'property_2' value: DOUBLE }
  )pb");
  ExecutionType type_2 = testing::ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'execution_type_2'
  )pb");
  ASSERT_EQ(absl::OkStatus(), CreateType(type_1, &type_id_1));
  type_1.set_id(type_id_1);
  ASSERT_EQ(absl::OkStatus(), CreateType(type_2, &type_id_2));
  type_2.set_id(type_id_2);

  RecordSet records = testing::ParseTextProtoOrDie<RecordSet>(
      std::string(kExecutionTypeRecordSet));
  records.mutable_records(0)->set_values(0, std::to_string(type_id_1));
  records.mutable_records(1)->set_values(0, std::to_string(type_id_2));

  std::vector<ExecutionType> expected_types = {type_1, type_2};
  VerifyFindTypesFromRecordSet(records, expected_types);
}

TEST_P(RDBMSMetadataAccessObjectTest, FindContextTypesFromRecordSet) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64_t type_id_1;
  ContextType type_1 = testing::ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'context_type_1'
  )pb");
  ASSERT_EQ(absl::OkStatus(), CreateType(type_1, &type_id_1));
  type_1.set_id(type_id_1);

  RecordSet records = testing::ParseTextProtoOrDie<RecordSet>(
      std::string(kContextTypeRecordSet));
  records.mutable_records(0)->set_values(0, std::to_string(type_id_1));

  std::vector<ContextType> expected_types = {type_1};
  VerifyFindTypesFromRecordSet(records, expected_types);
}

TEST_P(RDBMSMetadataAccessObjectTest, FindTypesImpl) {
  ASSERT_EQ(absl::OkStatus(), Init());

  // Setup: Create context type.
  int64_t context_type_id;
  ContextType context_type = testing::ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'context_type'
  )pb");
  ASSERT_EQ(absl::OkStatus(), CreateType(context_type, &context_type_id));
  context_type.set_id(context_type_id);

  // Setup: Create artifact types.
  int64_t artifact_type_id_1, artifact_type_id_2;
  ArtifactType artifact_type_1 =
      testing::ParseTextProtoOrDie<ArtifactType>(R"pb(
        name: 'artifact_type_1'
        properties { key: 'property_1' value: STRING }
        properties { key: 'property_2' value: DOUBLE }
      )pb");
  ArtifactType artifact_type_2 =
      testing::ParseTextProtoOrDie<ArtifactType>(R"pb(
        name: 'artifact_type_2'
        properties { key: 'property_1' value: INT }
      )pb");
  ASSERT_EQ(absl::OkStatus(), CreateType(artifact_type_1, &artifact_type_id_1));
  artifact_type_1.set_id(artifact_type_id_1);
  ASSERT_EQ(absl::OkStatus(), CreateType(artifact_type_2, &artifact_type_id_2));
  artifact_type_2.set_id(artifact_type_id_2);

  // Setup: Create execution type.
  int64_t execution_type_id;
  ExecutionType execution_type =
      testing::ParseTextProtoOrDie<ExecutionType>(R"pb(
        name: 'execution_type'
      )pb");
  ASSERT_EQ(absl::OkStatus(), CreateType(execution_type, &execution_type_id));
  execution_type.set_id(execution_type_id);

  // Test: empty ids
  {
    std::vector<ArtifactType> types;
    EXPECT_TRUE(absl::IsInvalidArgument(
        FindTypesImpl({}, /*get_properties=*/false, types)));
  }
  // Test: get types succeeded when `get_properties` is set to true.
  {
    std::vector<ArtifactType> artifact_types;
    ASSERT_EQ(absl::OkStatus(),
              FindTypesImpl({artifact_type_id_1, artifact_type_id_2},
                            /*get_properties=*/true, artifact_types));
    // Verify: type properties WILL be retrieved by FindTypesImpl if
    // `get_properties` is true.
    EXPECT_THAT(artifact_types,
                UnorderedElementsAre(EqualsProto(artifact_type_1),
                                     EqualsProto(artifact_type_2)));
    std::vector<ExecutionType> execution_types;
    ASSERT_EQ(absl::OkStatus(),
              FindTypesImpl({execution_type_id}, /*get_properties=*/true,
                            execution_types));
    ASSERT_EQ(execution_types.size(), 1);
    EXPECT_THAT(execution_types[0], EqualsProto(execution_type));

    std::vector<ContextType> context_types;
    ASSERT_EQ(absl::OkStatus(),
              FindTypesImpl({context_type_id}, /*get_properties=*/true,
                            context_types));
    ASSERT_EQ(context_types.size(), 1);
    EXPECT_THAT(context_types[0], EqualsProto(context_type));
  }
  // Test: get types succeeded when `get_properties` is set to false.
  {
    std::vector<ArtifactType> artifact_types;
    ASSERT_EQ(absl::OkStatus(),
              FindTypesImpl({artifact_type_id_1, artifact_type_id_2},
                            /*get_properties=*/false, artifact_types));
    // Verify: type properties will NOT be retrieved by FindTypesImpl if
    // `get_properties` is false.
    artifact_type_1.clear_properties();
    artifact_type_2.clear_properties();
    EXPECT_THAT(artifact_types,
                UnorderedElementsAre(EqualsProto(artifact_type_1),
                                     EqualsProto(artifact_type_2)));
    std::vector<ExecutionType> execution_types;
    ASSERT_EQ(absl::OkStatus(),
              FindTypesImpl({execution_type_id}, /*get_properties=*/false,
                            execution_types));
    ASSERT_EQ(execution_types.size(), 1);
    EXPECT_THAT(execution_types[0], EqualsProto(execution_type));

    std::vector<ContextType> context_types;
    ASSERT_EQ(absl::OkStatus(),
              FindTypesImpl({context_type_id}, /*get_properties=*/false,
                            context_types));
    ASSERT_EQ(context_types.size(), 1);
    EXPECT_THAT(context_types[0], EqualsProto(context_type));
  }
  // Test: mixed ids of different types
  {
    std::vector<ArtifactType> artifact_types;
    // Verify: NOT_FOUND error was returned because results were missing for
    // execution type id.
    EXPECT_TRUE(absl::IsNotFound(
        FindTypesImpl({artifact_type_id_1, execution_type_id},
                      /*get_properties=*/false, artifact_types)));
  }
}

TEST_P(RDBMSMetadataAccessObjectTest, FindParentTypesByTypeIdImpl) {
  ASSERT_EQ(absl::OkStatus(), Init());

  // Setup: init the database with the following types and inheritance links
  // ArtifactType:  type1 -> type3
  //                type2 -> type3
  //                type4 -> type5
  // ExecutionType: type6 -> type7
  // ContextType:   type8

  // Setup: Create artifact types and links.
  int64_t type_id_1, type_id_2, type_id_3, type_id_4, type_id_5;
  ArtifactType type_1 = testing::ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'artifact_type_1'
  )pb");
  ArtifactType type_2 = testing::ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'artifact_type_2'
  )pb");
  ArtifactType type_3 = testing::ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'artifact_type_3'
    properties { key: 'property_1' value: INT }
  )pb");
  ArtifactType type_4 = testing::ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'artifact_type_4'
  )pb");
  ArtifactType type_5 = testing::ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'artifact_type_5'
  )pb");
  ASSERT_EQ(absl::OkStatus(), CreateType(type_1, &type_id_1));
  type_1.set_id(type_id_1);
  ASSERT_EQ(absl::OkStatus(), CreateType(type_2, &type_id_2));
  type_2.set_id(type_id_2);
  ASSERT_EQ(absl::OkStatus(), CreateType(type_3, &type_id_3));
  type_3.set_id(type_id_3);
  ASSERT_EQ(absl::OkStatus(), CreateType(type_4, &type_id_4));
  type_4.set_id(type_id_4);
  ASSERT_EQ(absl::OkStatus(), CreateType(type_5, &type_id_5));
  type_5.set_id(type_id_5);
  ASSERT_EQ(absl::OkStatus(),
            rdbms_metadata_access_object_->CreateParentTypeInheritanceLink(
                type_1, type_3));
  ASSERT_EQ(absl::OkStatus(),
            rdbms_metadata_access_object_->CreateParentTypeInheritanceLink(
                type_2, type_3));
  ASSERT_EQ(absl::OkStatus(),
            rdbms_metadata_access_object_->CreateParentTypeInheritanceLink(
                type_4, type_5));

  // Setup: Create execution types and link.
  int64_t type_id_6, type_id_7;
  ExecutionType type_6 = testing::ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'execution_type_6'
  )pb");
  ExecutionType type_7 = testing::ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'execution_type_7'
  )pb");
  ASSERT_EQ(absl::OkStatus(), CreateType(type_6, &type_id_6));
  type_6.set_id(type_id_6);
  ASSERT_EQ(absl::OkStatus(), CreateType(type_7, &type_id_7));
  type_7.set_id(type_id_7);
  ASSERT_EQ(absl::OkStatus(),
            rdbms_metadata_access_object_->CreateParentTypeInheritanceLink(
                type_6, type_7));

  // Setup: Create context type.
  int64_t type_id_8;
  ContextType type_8 = testing::ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'context_type_8'
  )pb");
  ASSERT_EQ(absl::OkStatus(), CreateType(type_8, &type_id_8));
  type_8.set_id(type_id_8);

  // Test: empty ids.
  {
    absl::flat_hash_map<int64_t, ArtifactType> parent_types;
    EXPECT_TRUE(
        absl::IsInvalidArgument(FindParentTypesByTypeIdImpl({}, parent_types)));
  }
  // Test: get artifact parent types.
  {
    absl::flat_hash_map<int64_t, ArtifactType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              FindParentTypesByTypeIdImpl(
                  {type_id_1, type_id_2, type_id_3, type_id_4, type_id_5},
                  parent_types));
    // Verify: only type_1 and type_4 have parent types.
    ASSERT_EQ(parent_types.size(), 3);
    // Verify: type properties will NOT be retrieved by FindTypesImpl in
    // FindParentTypesByTypeIdImpl.
    type_3.clear_properties();
    EXPECT_THAT(parent_types[type_id_1], EqualsProto(type_3));
    EXPECT_THAT(parent_types[type_id_2], EqualsProto(type_3));
    EXPECT_THAT(parent_types[type_id_4], EqualsProto(type_5));
  }
  // Test: get execution parent types.
  {
    absl::flat_hash_map<int64_t, ExecutionType> parent_types;
    ASSERT_EQ(absl::OkStatus(), FindParentTypesByTypeIdImpl(
                                    {type_id_6, type_id_7}, parent_types));
    // Verify: only type_6 have 1 parent type.
    ASSERT_EQ(parent_types.size(), 1);
    EXPECT_THAT(parent_types[type_id_6], EqualsProto(type_7));
  }
  // Test: get context parent types.
  {
    absl::flat_hash_map<int64_t, ContextType> parent_types;
    ASSERT_EQ(absl::OkStatus(),
              FindParentTypesByTypeIdImpl({type_id_8}, parent_types));
    // Verify: type_8 do not have parent types.
    EXPECT_EQ(parent_types.size(), 0);
  }
  // Test: mixed ids of different child types.
  {
    absl::flat_hash_map<int64_t, ArtifactType> parent_types;
    // Verify: NOT_FOUND error was returned because results were missing for
    // `type_id_6` and `type_id_8`.
    EXPECT_TRUE(absl::IsNotFound(FindParentTypesByTypeIdImpl(
        {type_id_1, type_id_6, type_id_8}, parent_types)));
  }
}

TEST_P(RDBMSMetadataAccessObjectTest, FindNodesWithTypesImpl) {
  ASSERT_EQ(Init(), absl::OkStatus());
  ArtifactType type_1 = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'artifact_type_1'
    version: 'v1'
    description: 'artifact_type_description'
    external_id: 'artifact_type_1'
  )pb");
  ArtifactType type_2 = ParseTextProtoOrDie<ArtifactType>(R"pb(
    name: 'artifact_type_2'
    version: 'v1'
    description: 'artifact_type_description'
    external_id: 'artifact_type_2'
    properties { key: 'property' value: STRING }
  )pb");
  int64_t type_id_1, type_id_2;

  ASSERT_EQ(CreateType(type_1, &type_id_1), absl::OkStatus());
  type_1.set_id(type_id_1);
  ASSERT_EQ(CreateType(type_2, &type_id_2), absl::OkStatus());
  type_2.set_id(type_id_2);

  Artifact artifact_1 = ParseTextProtoOrDie<Artifact>(absl::Substitute(
      R"pb(
        type_id: $0 uri: 'testuri://testing/uri'
      )pb",
      type_id_1));
  Artifact artifact_2 = ParseTextProtoOrDie<Artifact>(absl::Substitute(
      R"pb(
        type_id: $0
        uri: 'testuri://testing/uri'
        properties {
          key: 'property'
          value: { string_value: '$1' }
        }
      )pb",
      type_id_2, "2"));
  int64_t artifact_id_1, artifact_id_2;
  ASSERT_EQ(CreateNodeImpl<Artifact>(artifact_1, type_1, &artifact_id_1),
            absl::OkStatus());
  ASSERT_EQ(CreateNodeImpl<Artifact>(artifact_2, type_2, &artifact_id_2),
            absl::OkStatus());
  artifact_1.set_id(artifact_id_1);
  artifact_2.set_id(artifact_id_2);

  // Test: Find two artifacts with artifact_types index aligned with them.
  {
    std::vector<Artifact> artifacts;
    std::vector<ArtifactType> types;
    ASSERT_EQ(
        FindNodesWithTypeImpl({artifact_id_2, artifact_id_1}, artifacts, types),
        absl::OkStatus());
    EXPECT_THAT(
        artifacts,
        UnorderedElementsAre(
            EqualsProto(artifact_1,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(artifact_2,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
    EXPECT_THAT(types,
                UnorderedElementsAre(EqualsProto(type_1), EqualsProto(type_2)));
  }
  // Test: Finding artifacts with type fails with INVALID_ARGUMENT error.
  {
    std::vector<Artifact> artifacts = {artifact_1};
    std::vector<ArtifactType> types;
    EXPECT_TRUE(absl::IsInvalidArgument(
        FindNodesWithTypeImpl({artifact_id_2}, artifacts, types)));
  }
  // Test: Finding artifacts with empty id list fails with INVALID_ARGUMENT
  // error.
  {
    std::vector<Artifact> artifacts;
    std::vector<ArtifactType> types;
    EXPECT_TRUE(
        absl::IsInvalidArgument(FindNodesWithTypeImpl({}, artifacts, types)));
  }
  // Test: Finding artifacts with type succeeds with unknown artifact_id.
  {
    std::vector<Artifact> artifacts;
    std::vector<ArtifactType> types;
    int64_t unknown_artifact_id = artifact_id_1 + artifact_id_2;
    absl::Status status = FindNodesWithTypeImpl(
        {artifact_id_1, artifact_id_2, unknown_artifact_id}, artifacts, types);
    EXPECT_TRUE(absl::IsNotFound(status));
    EXPECT_THAT(
        string(status.message()),
        AllOf(HasSubstr(absl::StrCat("Results missing for ids: {",
                                     artifact_id_1, ",", artifact_id_2, ",",
                                     unknown_artifact_id, "}")),
              AnyOf(HasSubstr(absl::StrCat("Found results for {", artifact_id_1,
                                           ",", artifact_id_2, "}")),
                    HasSubstr(absl::StrCat("Found results for {", artifact_id_2,
                                           ",", artifact_id_1, "}")))));
  }
}

}  // namespace ml_metadata
