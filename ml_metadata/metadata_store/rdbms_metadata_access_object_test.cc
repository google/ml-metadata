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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {

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
    EXPECT_THAT(got_types[i], testing::EqualsProto(expected_types[i]));
  }

  // Test when `get_properties` is false.
  got_types.clear();
  ASSERT_EQ(absl::OkStatus(), FindTypesFromRecordSet(records, &got_types,
                                                     /*get_properties=*/false));
  ASSERT_EQ(got_types.size(), expected_types.size());
  for (int i = 0; i < got_types.size(); ++i) {
    expected_types[i].clear_properties();
    EXPECT_THAT(got_types[i], testing::EqualsProto(expected_types[i]));
  }
}

TEST_P(RDBMSMetadataAccessObjectTest, FindArtifactTypesFromRecordSet) {
  ASSERT_EQ(absl::OkStatus(), Init());
  int64 type_id_1, type_id_2;
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
  int64 type_id_1, type_id_2;
  ExecutionType type_1 = testing::ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'execution_type_1'
    properties { key: 'property_1' value: STRING }
    properties { key: 'property_2' value: DOUBLE }
  )pb");
  ExecutionType type_2 = testing::ParseTextProtoOrDie<ExecutionType>(R"pb(
    name: 'execution_type_2'
    properties { key: 'property_1' value: INT }
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
  int64 type_id_1;
  ContextType type_1 = testing::ParseTextProtoOrDie<ContextType>(R"pb(
    name: 'context_type_1'
    properties { key: 'property_1' value: STRING }
    properties { key: 'property_2' value: DOUBLE }
  )pb");
  ASSERT_EQ(absl::OkStatus(), CreateType(type_1, &type_id_1));
  type_1.set_id(type_id_1);

  RecordSet records = testing::ParseTextProtoOrDie<RecordSet>(
      std::string(kContextTypeRecordSet));
  records.mutable_records(0)->set_values(0, std::to_string(type_id_1));

  std::vector<ContextType> expected_types = {type_1};
  VerifyFindTypesFromRecordSet(records, expected_types);
}

}  // namespace ml_metadata
