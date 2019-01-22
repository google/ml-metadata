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

#include "absl/time/time.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace testing {

namespace {
using testing::ParseTextProtoOrDie;
}  // namespace

TEST_P(MetadataAccessObjectTest, InitMetadataSource) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceIfNotExists) {
  // creates the schema and insert some records
  TF_EXPECT_OK(metadata_access_object_->InitMetadataSourceIfNotExists());
  ArtifactType want_type =
      ParseTextProtoOrDie<ArtifactType>("name: 'test_type'");
  int64 type_id = -1;
  TF_EXPECT_OK(metadata_access_object_->CreateType(want_type, &type_id));
  want_type.set_id(type_id);

  // all schema exists, the methods does nothing, check the stored type
  TF_EXPECT_OK(metadata_access_object_->InitMetadataSourceIfNotExists());
  ArtifactType type;
  TF_EXPECT_OK(metadata_access_object_->FindTypeById(type_id, &type));
  EXPECT_THAT(type, EqualsProto(want_type));
}

TEST_P(MetadataAccessObjectTest, InitMetadataSourceIfNotExistsErrorDataLoss) {
  // creates the schema and insert some records
  TF_EXPECT_OK(metadata_access_object_->InitMetadataSourceIfNotExists());

  {
    // delete some table
    RecordSet record_set;
    TF_EXPECT_OK(metadata_access_object_->metadata_source()->ExecuteQuery(
        "DROP TABLE IF EXISTS `Type`;", &record_set));
    tensorflow::Status s =
        metadata_access_object_->InitMetadataSourceIfNotExists();
    EXPECT_EQ(s.code(), tensorflow::error::DATA_LOSS);
  }

  // reset the database by drop and recreate all tables
  TF_EXPECT_OK(metadata_access_object_->InitMetadataSource());

  {
    // rename expected column from a table
    RecordSet record_set;
    tensorflow::Status alter_table_status =
        metadata_access_object_->metadata_source()->ExecuteQuery(
            "ALTER TABLE `Artifact` DROP COLUMN `uri`;", &record_set);
    // sqlite3 supports limited set of alter table syntax
    if (!alter_table_status.ok())
      TF_ASSERT_OK(metadata_access_object_->metadata_source()->ExecuteQuery(
          "ALTER TABLE `Artifact` RENAME COLUMN `uri` TO `column1`;",
          &record_set));

    tensorflow::Status s =
        metadata_access_object_->InitMetadataSourceIfNotExists();
    EXPECT_EQ(s.code(), tensorflow::error::DATA_LOSS);
  }
}

TEST_P(MetadataAccessObjectTest, CreateType) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ArtifactType type1 = ParseTextProtoOrDie<ArtifactType>("name: 'test_type'");
  int64 type1_id = -1;
  TF_EXPECT_OK(metadata_access_object_->CreateType(type1, &type1_id));

  ArtifactType type2 = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type2'
    properties { key: 'property_1' value: STRING })");
  int64 type2_id = -1;
  TF_EXPECT_OK(metadata_access_object_->CreateType(type2, &type2_id));
  EXPECT_NE(type1_id, type2_id);

  ExecutionType type3 = ParseTextProtoOrDie<ExecutionType>("name: 'test_type'");
  int64 type3_id = -1;
  TF_EXPECT_OK(metadata_access_object_->CreateType(type3, &type3_id));
  EXPECT_NE(type1_id, type3_id);
  EXPECT_NE(type2_id, type3_id);
}

TEST_P(MetadataAccessObjectTest, CreateTypeError) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  {
    ArtifactType wrong_type;
    int64 type_id;
    tensorflow::Status s =
        metadata_access_object_->CreateType(wrong_type, &type_id);
    EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);
  }
  {
    ArtifactType wrong_type = ParseTextProtoOrDie<ArtifactType>(R"(
      name: 'test_type2'
      properties { key: 'property_1' value: UNKNOWN })");
    int64 type_id;
    tensorflow::Status s =
        metadata_access_object_->CreateType(wrong_type, &type_id);
    EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);
  }
}

TEST_P(MetadataAccessObjectTest, FindTypeById) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ArtifactType want_type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(want_type, &type_id));
  want_type.set_id(type_id);

  ArtifactType type;
  TF_EXPECT_OK(metadata_access_object_->FindTypeById(type_id, &type));
  EXPECT_THAT(type, EqualsProto(want_type));

  ExecutionType execution_type;
  tensorflow::Status s =
      metadata_access_object_->FindTypeById(type_id, &execution_type);
  EXPECT_EQ(s.code(), tensorflow::error::NOT_FOUND);
}

TEST_P(MetadataAccessObjectTest, FindTypeByName) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ExecutionType want_type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(want_type, &type_id));
  want_type.set_id(type_id);

  ExecutionType type;
  TF_EXPECT_OK(metadata_access_object_->FindTypeByName("test_type", &type));
  EXPECT_THAT(type, EqualsProto(want_type));

  ArtifactType artifact_type;
  tensorflow::Status s =
      metadata_access_object_->FindTypeByName("test_type", &artifact_type);
  EXPECT_EQ(s.code(), tensorflow::error::NOT_FOUND);
}

TEST_P(MetadataAccessObjectTest, CreateArtifact) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type_with_predefined_property'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  Artifact artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://test/uri'
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
  )");
  artifact.set_type_id(type_id);

  int64 artifact1_id = -1;
  TF_EXPECT_OK(
      metadata_access_object_->CreateArtifact(artifact, &artifact1_id));
  int64 artifact2_id = -1;
  TF_EXPECT_OK(
      metadata_access_object_->CreateArtifact(artifact, &artifact2_id));
  EXPECT_NE(artifact1_id, artifact2_id);
}

TEST_P(MetadataAccessObjectTest, CreateArtifactWithCustomProperty) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type_with_custom_property'
  )");
  int64 type_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  Artifact artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://test/uri'
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
  )");
  artifact.set_type_id(type_id);

  int64 artifact1_id, artifact2_id;
  TF_EXPECT_OK(
      metadata_access_object_->CreateArtifact(artifact, &artifact1_id));
  EXPECT_EQ(artifact1_id, 1);
  TF_EXPECT_OK(
      metadata_access_object_->CreateArtifact(artifact, &artifact2_id));
  EXPECT_EQ(artifact2_id, 2);
}

TEST_P(MetadataAccessObjectTest, CreateArtifactError) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());

  // unknown type specified
  Artifact artifact;
  int64 artifact_id;
  tensorflow::Status s =
      metadata_access_object_->CreateArtifact(artifact, &artifact_id);
  EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);

  artifact.set_type_id(1);
  s = metadata_access_object_->CreateArtifact(artifact, &artifact_id);
  EXPECT_EQ(s.code(), tensorflow::error::NOT_FOUND);

  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type_disallow_custom_property'
    properties { key: 'property_1' value: INT }
  )");
  int64 type_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  // type mismatch
  Artifact artifact3;
  artifact3.set_type_id(type_id);
  (*artifact3.mutable_properties())["property_1"].set_string_value("3");
  int64 artifact3_id;
  s = metadata_access_object_->CreateArtifact(artifact3, &artifact3_id);
  EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);
}

TEST_P(MetadataAccessObjectTest, FindArtifactById) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  Artifact want_artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://test/uri'
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
  TF_ASSERT_OK(
      metadata_access_object_->CreateArtifact(want_artifact, &artifact_id));
  want_artifact.set_id(artifact_id);

  Artifact artifact;
  TF_EXPECT_OK(
      metadata_access_object_->FindArtifactById(artifact_id, &artifact));
  EXPECT_THAT(artifact, EqualsProto(want_artifact));
}

TEST_P(MetadataAccessObjectTest, FindAllArtifacts) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  Artifact want_artifact1 = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://test/uri'
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
  want_artifact1.set_type_id(type_id);

  int64 artifact1_id;
  TF_ASSERT_OK(
      metadata_access_object_->CreateArtifact(want_artifact1, &artifact1_id));
  want_artifact1.set_id(artifact1_id);

  Artifact want_artifact2 = want_artifact1;
  int64 artifact2_id;
  TF_ASSERT_OK(
      metadata_access_object_->CreateArtifact(want_artifact2, &artifact2_id));
  want_artifact2.set_id(artifact2_id);
  ASSERT_NE(artifact1_id, artifact2_id);

  std::vector<Artifact> artifacts;
  TF_EXPECT_OK(metadata_access_object_->FindArtifacts(&artifacts));
  EXPECT_EQ(artifacts.size(), 2);
  EXPECT_THAT(artifacts[0], EqualsProto(want_artifact1));
  EXPECT_THAT(artifacts[1], EqualsProto(want_artifact2));
}

TEST_P(MetadataAccessObjectTest, UpdateArtifact) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  Artifact stored_artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://test/uri'
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
  )");
  stored_artifact.set_type_id(type_id);
  int64 artifact_id;
  TF_ASSERT_OK(
      metadata_access_object_->CreateArtifact(stored_artifact, &artifact_id));

  // update `property_1`, add `property_2`, and drop `property_3`
  // change the value type of `custom_property_1`
  Artifact want_artifact = ParseTextProtoOrDie<Artifact>(R"(
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
  want_artifact.set_id(artifact_id);
  want_artifact.set_type_id(type_id);
  TF_EXPECT_OK(metadata_access_object_->UpdateArtifact(want_artifact));

  Artifact artifact;
  TF_EXPECT_OK(
      metadata_access_object_->FindArtifactById(artifact_id, &artifact));
  EXPECT_THAT(artifact, EqualsProto(want_artifact));
}

TEST_P(MetadataAccessObjectTest, UpdateArtifactError) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ArtifactType type = ParseTextProtoOrDie<ArtifactType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
  )");
  int64 type_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  Artifact artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://test/uri'
    properties {
      key: 'property_1'
      value: { int_value: 3 }
    }
  )");
  artifact.set_type_id(type_id);
  int64 artifact_id;
  TF_ASSERT_OK(metadata_access_object_->CreateArtifact(artifact, &artifact_id));
  artifact.set_id(artifact_id);

  // no artifact id given
  Artifact wrong_artifact;
  tensorflow::Status s =
      metadata_access_object_->UpdateArtifact(wrong_artifact);
  EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);

  // artifact id cannot be found
  int64 different_id = artifact_id + 1;
  wrong_artifact.set_id(different_id);
  s = metadata_access_object_->UpdateArtifact(wrong_artifact);
  EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);

  // type_id if given is not aligned with the stored one
  wrong_artifact.set_id(artifact_id);
  int64 different_type_id = type_id + 1;
  wrong_artifact.set_type_id(different_type_id);
  s = metadata_access_object_->UpdateArtifact(wrong_artifact);
  EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);

  // artifact has unknown property
  wrong_artifact.clear_type_id();
  (*wrong_artifact.mutable_properties())["unknown_property"].set_int_value(1);
  s = metadata_access_object_->UpdateArtifact(wrong_artifact);
  EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);
}

TEST_P(MetadataAccessObjectTest, CreateAndFindExecution) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type_with_predefined_property'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  Execution execution1 = ParseTextProtoOrDie<Execution>(R"(
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
  execution1.set_type_id(type_id);

  int64 execution1_id = -1;
  TF_EXPECT_OK(
      metadata_access_object_->CreateExecution(execution1, &execution1_id));
  execution1.set_id(execution1_id);

  ExecutionType name_only_type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type_with_no_property'
  )");
  int64 type2_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(name_only_type, &type2_id));
  Execution execution2;
  execution2.set_type_id(type2_id);

  int64 execution2_id = -1;
  TF_EXPECT_OK(
      metadata_access_object_->CreateExecution(execution2, &execution2_id));
  execution2.set_id(execution2_id);

  EXPECT_NE(execution1_id, execution2_id);

  Execution want_execution1;
  TF_EXPECT_OK(metadata_access_object_->FindExecutionById(execution1_id,
                                                          &want_execution1));
  EXPECT_THAT(execution1, EqualsProto(want_execution1));

  Execution want_execution2;
  TF_EXPECT_OK(metadata_access_object_->FindExecutionById(execution2_id,
                                                          &want_execution2));
  EXPECT_THAT(execution2, EqualsProto(want_execution2));

  std::vector<Execution> executions;
  TF_EXPECT_OK(metadata_access_object_->FindExecutions(&executions));
  EXPECT_EQ(executions.size(), 2);
  EXPECT_THAT(executions[0], EqualsProto(want_execution1));
  EXPECT_THAT(executions[1], EqualsProto(want_execution2));
}

TEST_P(MetadataAccessObjectTest, UpdateExecution) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ExecutionType type = ParseTextProtoOrDie<ExecutionType>(R"(
    name: 'test_type'
    properties { key: 'property_1' value: INT }
    properties { key: 'property_2' value: DOUBLE }
    properties { key: 'property_3' value: STRING }
  )");
  int64 type_id;
  TF_ASSERT_OK(metadata_access_object_->CreateType(type, &type_id));

  Execution stored_execution = ParseTextProtoOrDie<Execution>(R"(
    properties {
      key: 'property_3'
      value: { string_value: '3' }
    }
    custom_properties {
      key: 'custom_property_1'
      value: { string_value: '5' }
    }
  )");
  stored_execution.set_type_id(type_id);
  int64 execution_id;
  TF_ASSERT_OK(metadata_access_object_->CreateExecution(stored_execution,
                                                        &execution_id));

  // add `property_1` and update `property_3`, and drop `custom_property_1`
  Execution want_execution = ParseTextProtoOrDie<Execution>(R"(
    properties {
      key: 'property_1'
      value: { int_value: 5 }
    }
    properties {
      key: 'property_3'
      value: { string_value: '5' }
    }
  )");
  want_execution.set_id(execution_id);
  want_execution.set_type_id(type_id);
  TF_EXPECT_OK(metadata_access_object_->UpdateExecution(want_execution));

  Execution execution;
  TF_EXPECT_OK(
      metadata_access_object_->FindExecutionById(execution_id, &execution));
  EXPECT_THAT(execution, EqualsProto(want_execution));
}

TEST_P(MetadataAccessObjectTest, CreateAndFindEvent) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());
  ArtifactType artifact_type;
  artifact_type.set_name("test_artifact_type");
  int64 artifact_type_id;
  TF_ASSERT_OK(
      metadata_access_object_->CreateType(artifact_type, &artifact_type_id));
  ExecutionType execution_type;
  execution_type.set_name("test_execution_type");
  int64 execution_type_id;
  TF_ASSERT_OK(
      metadata_access_object_->CreateType(execution_type, &execution_type_id));
  Artifact input_artifact;
  input_artifact.set_type_id(artifact_type_id);
  int64 input_artifact_id;
  TF_ASSERT_OK(metadata_access_object_->CreateArtifact(input_artifact,
                                                       &input_artifact_id));

  Artifact output_artifact;
  output_artifact.set_type_id(artifact_type_id);
  int64 output_artifact_id;
  TF_ASSERT_OK(metadata_access_object_->CreateArtifact(output_artifact,
                                                       &output_artifact_id));

  Execution execution;
  execution.set_type_id(execution_type_id);
  int64 execution_id;
  TF_ASSERT_OK(
      metadata_access_object_->CreateExecution(execution, &execution_id));

  // event1 with event paths
  Event event1 = ParseTextProtoOrDie<Event>("type: INPUT");
  event1.set_artifact_id(input_artifact_id);
  event1.set_execution_id(execution_id);
  event1.set_milliseconds_since_epoch(12345);
  event1.mutable_path()->add_steps()->set_index(1);
  event1.mutable_path()->add_steps()->set_key("key");
  int64 event1_id = -1;
  TF_EXPECT_OK(metadata_access_object_->CreateEvent(event1, &event1_id));

  // event2 with optional fields
  Event event2 = ParseTextProtoOrDie<Event>("type: OUTPUT");
  event2.set_artifact_id(output_artifact_id);
  event2.set_execution_id(execution_id);
  int64 event2_id = -1;
  TF_EXPECT_OK(metadata_access_object_->CreateEvent(event2, &event2_id));

  EXPECT_NE(event1_id, -1);
  EXPECT_NE(event2_id, -1);
  EXPECT_NE(event1_id, event2_id);

  // query the executions
  std::vector<Event> events_with_input_artifact;
  TF_EXPECT_OK(metadata_access_object_->FindEventsByArtifact(
      input_artifact_id, &events_with_input_artifact));
  EXPECT_EQ(events_with_input_artifact.size(), 1);
  EXPECT_THAT(events_with_input_artifact[0], EqualsProto(event1));

  std::vector<Event> events_with_output_artifact;
  TF_EXPECT_OK(metadata_access_object_->FindEventsByArtifact(
      output_artifact_id, &events_with_output_artifact));
  EXPECT_EQ(events_with_output_artifact.size(), 1);
  event2.set_milliseconds_since_epoch(
      events_with_output_artifact[0].milliseconds_since_epoch());
  EXPECT_THAT(events_with_output_artifact[0], EqualsProto(event2));

  std::vector<Event> events_with_execution;
  TF_EXPECT_OK(metadata_access_object_->FindEventsByExecution(
      execution_id, &events_with_execution));
  EXPECT_EQ(events_with_execution.size(), 2);
}

TEST_P(MetadataAccessObjectTest, CreateEventError) {
  TF_ASSERT_OK(metadata_access_object_->InitMetadataSource());

  // no artifact id
  {
    Event event;
    int64 event_id;
    tensorflow::Status s =
        metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);
  }

  // no execution id
  {
    Event event;
    int64 event_id;
    event.set_artifact_id(1);
    tensorflow::Status s =
        metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);
  }

  // no event type
  {
    Event event;
    int64 event_id;
    event.set_artifact_id(1);
    event.set_execution_id(1);
    tensorflow::Status s =
        metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);
  }

  // artifact or execution cannot be found
  {
    ArtifactType artifact_type;
    artifact_type.set_name("test_artifact_type");
    int64 artifact_type_id;
    TF_ASSERT_OK(
        metadata_access_object_->CreateType(artifact_type, &artifact_type_id));
    Artifact artifact;
    artifact.set_type_id(artifact_type_id);
    int64 artifact_id;
    TF_ASSERT_OK(
        metadata_access_object_->CreateArtifact(artifact, &artifact_id));

    Event event;
    int64 event_id;
    event.set_artifact_id(artifact_id);
    int64 unknown_id = 12345;
    event.set_execution_id(unknown_id);
    tensorflow::Status s =
        metadata_access_object_->CreateEvent(event, &event_id);
    EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace testing
}  // namespace ml_metadata
