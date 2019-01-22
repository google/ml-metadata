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
#include "ml_metadata/metadata_store/metadata_store.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {
using std::unique_ptr;
using testing::ParseTextProtoOrDie;

class MetadataStoreTest : public ::testing::Test {
 protected:
  MetadataStoreTest() {
    TF_CHECK_OK(MetadataStore::Create(
        util::GetSqliteMetadataSourceQueryConfig(),
        absl::make_unique<SqliteMetadataSource>(SqliteMetadataSourceConfig()),
        &metadata_store_));
    TF_CHECK_OK(metadata_store_->InitMetadataStore());
  }

  // MetadataStore that is initialized at construction time.
  std::unique_ptr<MetadataStore> metadata_store_;
};

TEST_F(MetadataStoreTest, InitMetadataStoreIfNotExists) {
  TF_ASSERT_OK(metadata_store_->InitMetadataStoreIfNotExists());
  // This is just to check that the metadata store was initialized.
  PutArtifactTypeRequest put_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  TF_ASSERT_OK(metadata_store_->InitMetadataStoreIfNotExists());
  GetArtifactTypeRequest get_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetArtifactTypeResponse get_response;
  TF_ASSERT_OK(metadata_store_->GetArtifactType(get_request, &get_response));
  EXPECT_EQ(put_response.type_id(), get_response.artifact_type().id())
      << "Type ID should be the same as the type created.";
  EXPECT_EQ("test_type2", get_response.artifact_type().name())
      << "The name should be the same as the one returned.";
}

TEST_F(MetadataStoreTest, PutArtifactTypeGetArtifactType) {
  PutArtifactTypeRequest put_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  GetArtifactTypeRequest get_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetArtifactTypeResponse get_response;
  TF_ASSERT_OK(metadata_store_->GetArtifactType(get_request, &get_response));
  EXPECT_EQ(put_response.type_id(), get_response.artifact_type().id())
      << "Type ID should be the same as the type created.";
  EXPECT_EQ("test_type2", get_response.artifact_type().name())
      << "The name should be the same as the one returned.";
  // Don't test all the properties, to make the serialization of the type
  // more flexible. This can be tested at other layers.
}

// Create an artifact, then try to create it again with an added property.
TEST_F(MetadataStoreTest, PutArtifactTypeTwiceChangedAddedProperty) {
  PutArtifactTypeRequest request_1 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse response_1;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(request_1, &response_1));

  PutArtifactTypeRequest request_2 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
              properties { key: 'property_2' value: STRING }
            }
          )");
  PutArtifactTypeResponse response_2;
  EXPECT_FALSE(metadata_store_->PutArtifactType(request_2, &response_2).ok());
}

TEST_F(MetadataStoreTest, PutArtifactTypeTwiceChangedRemovedProperty) {
  PutArtifactTypeRequest request_1 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
              properties { key: 'property_2' value: STRING }
            }
          )");
  PutArtifactTypeResponse response_1;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(request_1, &response_1));

  PutArtifactTypeRequest request_2 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse response_2;
  EXPECT_FALSE(metadata_store_->PutArtifactType(request_2, &response_2).ok());
}

TEST_F(MetadataStoreTest, PutArtifactTypeTwiceChangedPropertyType) {
  PutArtifactTypeRequest request_1 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse response_1;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(request_1, &response_1));

  PutArtifactTypeRequest request_2 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: INT }
            }
          )");
  PutArtifactTypeResponse response_2;
  EXPECT_FALSE(metadata_store_->PutArtifactType(request_2, &response_2).ok());
}

TEST_F(MetadataStoreTest, PutArtifactTypeSame) {
  PutArtifactTypeRequest request_1 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse response_1;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(request_1, &response_1));

  PutArtifactTypeRequest request_2 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse response_2;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(request_2, &response_2));
  EXPECT_EQ(response_1.type_id(), response_2.type_id());
}

// Test for failure.
TEST_F(MetadataStoreTest, GetArtifactTypeMissing) {
  GetArtifactTypeRequest get_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetArtifactTypeResponse get_response;
  EXPECT_FALSE(
      metadata_store_->GetArtifactType(get_request, &get_response).ok());
}

TEST_F(MetadataStoreTest, PutArtifactTypeGetArtifactTypesByID) {
  PutArtifactTypeRequest put_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  GetArtifactTypesByIDRequest get_request;
  GetArtifactTypesByIDResponse get_response;
  get_request.add_type_ids(put_response.type_id());
  TF_ASSERT_OK(
      metadata_store_->GetArtifactTypesByID(get_request, &get_response));
  ASSERT_EQ(get_response.artifact_types_size(), 1);
  const ArtifactType& result = get_response.artifact_types(0);
  EXPECT_EQ(put_response.type_id(), result.id())
      << "Type ID should be the same as the type created.";
  ArtifactType expected_result = put_request.artifact_type();
  expected_result.set_id(put_response.type_id());
  EXPECT_THAT(result, testing::EqualsProto(expected_result))
      << "The type should be the same as the one given.";
}

TEST_F(MetadataStoreTest, GetArtifactTypesByIDMissing) {
  // Returns an empty list.
  GetArtifactTypesByIDRequest get_request;
  GetArtifactTypesByIDResponse get_response;
  // There are no artifact types: this one is just made up.
  get_request.add_type_ids(12);
  TF_ASSERT_OK(
      metadata_store_->GetArtifactTypesByID(get_request, &get_response));
  ASSERT_EQ(get_response.artifact_types_size(), 0);
}

TEST_F(MetadataStoreTest, PutArtifactTypeGetArtifactTypesByIDTwo) {
  // Check that two artifact types can be retrieved.
  PutArtifactTypeRequest put_request_1 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type1'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse put_response_1;
  TF_ASSERT_OK(
      metadata_store_->PutArtifactType(put_request_1, &put_response_1));
  ASSERT_TRUE(put_response_1.has_type_id());
  PutArtifactTypeRequest put_request_2 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse put_response_2;
  TF_ASSERT_OK(
      metadata_store_->PutArtifactType(put_request_2, &put_response_2));

  GetArtifactTypesByIDRequest get_request;
  GetArtifactTypesByIDResponse get_response;
  get_request.add_type_ids(put_response_1.type_id());
  get_request.add_type_ids(put_response_2.type_id());
  TF_ASSERT_OK(
      metadata_store_->GetArtifactTypesByID(get_request, &get_response));
  ASSERT_EQ(get_response.artifact_types_size(), 2);
  const ArtifactType& result_1 = get_response.artifact_types(0);
  const ArtifactType& result_2 = get_response.artifact_types(1);
  ArtifactType expected_result_1 = put_request_1.artifact_type();
  ArtifactType expected_result_2 = put_request_2.artifact_type();
  expected_result_1.set_id(put_response_1.type_id());
  expected_result_2.set_id(put_response_2.type_id());

  EXPECT_THAT(result_1, testing::EqualsProto(expected_result_1))
      << "Type ID should be the same as the type created.";
  EXPECT_THAT(result_2, testing::EqualsProto(expected_result_2))
      << "The name should be the same as the one returned.";
}

TEST_F(MetadataStoreTest, PutExecutionTypeGetExecutionTypesByID) {
  PutExecutionTypeRequest put_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutExecutionTypeResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  GetExecutionTypesByIDRequest get_request;
  GetExecutionTypesByIDResponse get_response;
  get_request.add_type_ids(put_response.type_id());
  TF_ASSERT_OK(
      metadata_store_->GetExecutionTypesByID(get_request, &get_response));
  ASSERT_EQ(get_response.execution_types_size(), 1);
  const ExecutionType& result = get_response.execution_types(0);
  EXPECT_EQ(put_response.type_id(), result.id())
      << "Type ID should be the same as the type created.";
  ExecutionType expected_result = put_request.execution_type();
  expected_result.set_id(put_response.type_id());
  EXPECT_THAT(result, testing::EqualsProto(expected_result))
      << "The type should be the same as the one given.";
}

TEST_F(MetadataStoreTest, GetExecutionTypesByIDMissing) {
  // Returns an empty list.
  GetExecutionTypesByIDRequest get_request;
  GetExecutionTypesByIDResponse get_response;
  // There are no execution types: this one is just made up.
  get_request.add_type_ids(12);
  TF_ASSERT_OK(
      metadata_store_->GetExecutionTypesByID(get_request, &get_response));
  ASSERT_EQ(get_response.execution_types_size(), 0);
}

TEST_F(MetadataStoreTest, PutExecutionTypeGetExecutionTypesByIDTwo) {
  // Check that two execution types can be retrieved.
  PutExecutionTypeRequest put_request_1 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type1'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutExecutionTypeResponse put_response_1;
  TF_ASSERT_OK(
      metadata_store_->PutExecutionType(put_request_1, &put_response_1));
  ASSERT_TRUE(put_response_1.has_type_id());
  PutExecutionTypeRequest put_request_2 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutExecutionTypeResponse put_response_2;
  TF_ASSERT_OK(
      metadata_store_->PutExecutionType(put_request_2, &put_response_2));

  GetExecutionTypesByIDRequest get_request;
  GetExecutionTypesByIDResponse get_response;
  get_request.add_type_ids(put_response_1.type_id());
  get_request.add_type_ids(put_response_2.type_id());
  TF_ASSERT_OK(
      metadata_store_->GetExecutionTypesByID(get_request, &get_response));
  ASSERT_EQ(get_response.execution_types_size(), 2);
  const ExecutionType& result_1 = get_response.execution_types(0);
  const ExecutionType& result_2 = get_response.execution_types(1);
  ExecutionType expected_result_1 = put_request_1.execution_type();
  ExecutionType expected_result_2 = put_request_2.execution_type();
  expected_result_1.set_id(put_response_1.type_id());
  expected_result_2.set_id(put_response_2.type_id());

  EXPECT_THAT(result_1, testing::EqualsProto(expected_result_1))
      << "Type ID should be the same as the type created.";
  EXPECT_THAT(result_2, testing::EqualsProto(expected_result_2))
      << "The name should be the same as the one returned.";
}

TEST_F(MetadataStoreTest, PutArtifactsGetArtifactsByID) {
  const PutArtifactTypeRequest put_artifact_type_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property' value: STRING }
            }
          )");
  PutArtifactTypeResponse put_artifact_type_response;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(put_artifact_type_request,
                                                &put_artifact_type_response));
  ASSERT_TRUE(put_artifact_type_response.has_type_id());

  const int64 type_id = put_artifact_type_response.type_id();

  PutArtifactsRequest put_artifacts_request =
      ParseTextProtoOrDie<PutArtifactsRequest>(R"(
        artifacts: {
          uri: 'testuri://test/uri'
          properties {
            key: 'property'
            value: { string_value: '3' }
          }
        }
      )");
  put_artifacts_request.mutable_artifacts(0)->set_type_id(type_id);
  PutArtifactsResponse put_artifacts_response;

  TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request,
                                             &put_artifacts_response));
  ASSERT_EQ(put_artifacts_response.artifact_ids_size(), 1);
  const int64 artifact_id = put_artifacts_response.artifact_ids(0);
  GetArtifactsByIDRequest get_artifacts_by_id_request;
  get_artifacts_by_id_request.add_artifact_ids(artifact_id);
  GetArtifactsByIDResponse get_artifacts_by_id_response;
  TF_ASSERT_OK(metadata_store_->GetArtifactsByID(
      get_artifacts_by_id_request, &get_artifacts_by_id_response));
  GetArtifactsByIDResponse expected;
  *expected.mutable_artifacts() = put_artifacts_request.artifacts();
  expected.mutable_artifacts(0)->set_id(artifact_id);
  EXPECT_THAT(get_artifacts_by_id_response, testing::EqualsProto(expected));
}

// Test creating an artifact and then updating one of its properties.
TEST_F(MetadataStoreTest, PutArtifactsUpdateGetArtifactsByID) {
  const PutArtifactTypeRequest put_artifact_type_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property' value: STRING }
            }
          )");
  PutArtifactTypeResponse put_artifact_type_response;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(put_artifact_type_request,
                                                &put_artifact_type_response));
  ASSERT_TRUE(put_artifact_type_response.has_type_id());

  const int64 type_id = put_artifact_type_response.type_id();

  PutArtifactsRequest put_artifacts_request =
      ParseTextProtoOrDie<PutArtifactsRequest>(R"(
        artifacts: {
          uri: 'testuri://test/uri'
          properties {
            key: 'property'
            value: { string_value: '3' }
          }
        }
      )");
  put_artifacts_request.mutable_artifacts(0)->set_type_id(type_id);
  PutArtifactsResponse put_artifacts_response;
  TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request,
                                             &put_artifacts_response));
  ASSERT_EQ(put_artifacts_response.artifact_ids_size(), 1);
  const int64 artifact_id = put_artifacts_response.artifact_ids(0);

  // Now we change 3 to 2.
  PutArtifactsRequest put_artifacts_request_2 =
      ParseTextProtoOrDie<PutArtifactsRequest>(R"(
        artifacts: {
          uri: 'testuri://test/uri'
          properties {
            key: 'property'
            value: { string_value: '2' }
          }
        }
      )");

  put_artifacts_request_2.mutable_artifacts(0)->set_type_id(type_id);
  put_artifacts_request_2.mutable_artifacts(0)->set_id(artifact_id);
  PutArtifactsResponse put_artifacts_response_2;
  TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request_2,
                                             &put_artifacts_response_2));

  GetArtifactsByIDRequest get_artifacts_by_id_request;
  get_artifacts_by_id_request.add_artifact_ids(artifact_id);
  GetArtifactsByIDResponse get_artifacts_by_id_response;
  TF_ASSERT_OK(metadata_store_->GetArtifactsByID(
      get_artifacts_by_id_request, &get_artifacts_by_id_response));
  ASSERT_EQ(get_artifacts_by_id_response.artifacts_size(), 1);
  EXPECT_THAT(get_artifacts_by_id_response.artifacts(0),
              testing::EqualsProto(put_artifacts_request_2.artifacts(0)));
}
// Test creating an execution and then updating one of its properties.
TEST_F(MetadataStoreTest, PutExecutionsUpdateGetExecutionsByID) {
  const PutExecutionTypeRequest put_execution_type_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property' value: STRING }
            }
          )");
  PutExecutionTypeResponse put_execution_type_response;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(put_execution_type_request,
                                                 &put_execution_type_response));
  ASSERT_TRUE(put_execution_type_response.has_type_id());

  const int64 type_id = put_execution_type_response.type_id();

  PutExecutionsRequest put_executions_request =
      ParseTextProtoOrDie<PutExecutionsRequest>(R"(
        executions: {
          properties {
            key: 'property'
            value: { string_value: '3' }
          }
        }
      )");
  put_executions_request.mutable_executions(0)->set_type_id(type_id);
  PutExecutionsResponse put_executions_response;
  TF_ASSERT_OK(metadata_store_->PutExecutions(put_executions_request,
                                              &put_executions_response));
  ASSERT_EQ(put_executions_response.execution_ids_size(), 1);
  const int64 execution_id = put_executions_response.execution_ids(0);

  // Now we change 3 to 2.
  PutExecutionsRequest put_executions_request_2 =
      ParseTextProtoOrDie<PutExecutionsRequest>(R"(
        executions: {
          properties {
            key: 'property'
            value: { string_value: '2' }
          }
        }
      )");

  put_executions_request_2.mutable_executions(0)->set_type_id(type_id);
  put_executions_request_2.mutable_executions(0)->set_id(execution_id);
  PutExecutionsResponse put_executions_response_2;
  TF_ASSERT_OK(metadata_store_->PutExecutions(put_executions_request_2,
                                              &put_executions_response_2));

  GetExecutionsByIDRequest get_executions_by_id_request;
  get_executions_by_id_request.add_execution_ids(execution_id);
  GetExecutionsByIDResponse get_executions_by_id_response;
  TF_ASSERT_OK(metadata_store_->GetExecutionsByID(
      get_executions_by_id_request, &get_executions_by_id_response));

  GetExecutionsByIDResponse expected_response =
      ParseTextProtoOrDie<GetExecutionsByIDResponse>(R"(
        executions: {
          properties {
            key: 'property'
            value: { string_value: '2' }
          }
        }
      )");
  expected_response.mutable_executions(0)->set_id(execution_id);
  expected_response.mutable_executions(0)->set_type_id(type_id);

  EXPECT_THAT(get_executions_by_id_response,
              testing::EqualsProto(expected_response));
}

TEST_F(MetadataStoreTest, PutExecutionTypeGetExecutionType) {
  PutExecutionTypeRequest put_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutExecutionTypeResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  GetExecutionTypeRequest get_request =
      ParseTextProtoOrDie<GetExecutionTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetExecutionTypeResponse get_response;
  TF_ASSERT_OK(metadata_store_->GetExecutionType(get_request, &get_response));
  ExecutionType expected = put_request.execution_type();
  expected.set_id(put_response.type_id());
  EXPECT_THAT(get_response.execution_type(), testing::EqualsProto(expected));
}

TEST_F(MetadataStoreTest, PutExecutionTypeTwiceChangedPropertyType) {
  PutExecutionTypeRequest request_1 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutExecutionTypeResponse response_1;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(request_1, &response_1));

  PutExecutionTypeRequest request_2 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: INT }
            }
          )");
  PutExecutionTypeResponse response_2;
  tensorflow::Status status =
      metadata_store_->PutExecutionType(request_2, &response_2);
  EXPECT_EQ(tensorflow::error::ALREADY_EXISTS, status.code())
      << status.ToString();
}

TEST_F(MetadataStoreTest, PutExecutionTypeSame) {
  PutExecutionTypeRequest request_1 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutExecutionTypeResponse response_1;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(request_1, &response_1));

  PutExecutionTypeRequest request_2 = request_1;
  PutExecutionTypeResponse response_2;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(request_2, &response_2));
  EXPECT_EQ(response_1.type_id(), response_2.type_id());
}

// Test for failure.
TEST_F(MetadataStoreTest, GetExecutionTypeMissing) {
  GetExecutionTypeRequest get_request =
      ParseTextProtoOrDie<GetExecutionTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetExecutionTypeResponse get_response;
  EXPECT_EQ(
      tensorflow::error::NOT_FOUND,
      metadata_store_->GetExecutionType(get_request, &get_response).code());
}

TEST_F(MetadataStoreTest, PutExecutionsGetExecutionByID) {
  const PutExecutionTypeRequest put_execution_type_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property' value: STRING }
            }
          )");
  PutExecutionTypeResponse put_execution_type_response;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(put_execution_type_request,
                                                 &put_execution_type_response));
  ASSERT_TRUE(put_execution_type_response.has_type_id());

  const int64 type_id = put_execution_type_response.type_id();

  PutExecutionsRequest put_executions_request =
      ParseTextProtoOrDie<PutExecutionsRequest>(R"(
        executions: {
          properties {
            key: 'property'
            value: { string_value: '3' }
          }
        }
        executions: {
          properties {
            key: 'property'
            value: { string_value: '2' }
          }
        }
      )");
  put_executions_request.mutable_executions(0)->set_type_id(type_id);
  put_executions_request.mutable_executions(1)->set_type_id(type_id);
  PutExecutionsResponse put_executions_response;

  TF_ASSERT_OK(metadata_store_->PutExecutions(put_executions_request,
                                              &put_executions_response));
  ASSERT_EQ(put_executions_response.execution_ids_size(), 2);
  const int64 execution_id_0 = put_executions_response.execution_ids(0);
  const int64 execution_id_1 = put_executions_response.execution_ids(1);

  GetExecutionsByIDRequest get_executions_by_id_request;
  get_executions_by_id_request.add_execution_ids(execution_id_0);
  get_executions_by_id_request.add_execution_ids(execution_id_1);
  GetExecutionsByIDResponse get_executions_by_id_response;
  TF_ASSERT_OK(metadata_store_->GetExecutionsByID(
      get_executions_by_id_request, &get_executions_by_id_response));
  ASSERT_EQ(get_executions_by_id_response.executions_size(), 2);
  GetExecutionsByIDResponse expected;
  *expected.mutable_executions() = put_executions_request.executions();
  expected.mutable_executions(0)->set_id(execution_id_0);
  expected.mutable_executions(1)->set_id(execution_id_1);
  EXPECT_THAT(get_executions_by_id_response, testing::EqualsProto(expected));
}

TEST_F(MetadataStoreTest, PutExecutionsGetExecutionsWithEmptyExecution) {
  const PutExecutionTypeRequest put_execution_type_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: { name: 'test_type2' }
          )");
  PutExecutionTypeResponse put_execution_type_response;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(put_execution_type_request,
                                                 &put_execution_type_response));
  ASSERT_TRUE(put_execution_type_response.has_type_id());

  const int64 type_id = put_execution_type_response.type_id();

  PutExecutionsRequest put_executions_request =
      ParseTextProtoOrDie<PutExecutionsRequest>(R"(
        executions: {}
      )");
  put_executions_request.mutable_executions(0)->set_type_id(type_id);
  PutExecutionsResponse put_executions_response;

  TF_ASSERT_OK(metadata_store_->PutExecutions(put_executions_request,
                                              &put_executions_response));
  ASSERT_EQ(put_executions_response.execution_ids_size(), 1);
  const int64 execution_id = put_executions_response.execution_ids(0);
  GetExecutionsRequest get_executions_request;
  GetExecutionsResponse get_executions_response;
  TF_ASSERT_OK(metadata_store_->GetExecutions(get_executions_request,
                                              &get_executions_response));
  GetExecutionsResponse expected;
  *expected.mutable_executions() = put_executions_request.executions();
  expected.mutable_executions(0)->set_id(execution_id);
  EXPECT_THAT(get_executions_response, testing::EqualsProto(expected));
}

TEST_F(MetadataStoreTest, PutArtifactsGetArtifactsWithEmptyArtifact) {
  const PutArtifactTypeRequest put_artifact_type_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: { name: 'test_type2' }
          )");
  PutArtifactTypeResponse put_artifact_type_response;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(put_artifact_type_request,
                                                &put_artifact_type_response));
  ASSERT_TRUE(put_artifact_type_response.has_type_id());

  const int64 type_id = put_artifact_type_response.type_id();

  PutArtifactsRequest put_artifacts_request =
      ParseTextProtoOrDie<PutArtifactsRequest>(R"(
        artifacts: {}
      )");
  put_artifacts_request.mutable_artifacts(0)->set_type_id(type_id);
  PutArtifactsResponse put_artifacts_response;

  TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request,
                                             &put_artifacts_response));
  ASSERT_EQ(put_artifacts_response.artifact_ids_size(), 1);
  const int64 artifact_id = put_artifacts_response.artifact_ids(0);
  GetArtifactsRequest get_artifacts_request;
  GetArtifactsResponse get_artifacts_response;
  TF_ASSERT_OK(metadata_store_->GetArtifacts(get_artifacts_request,
                                             &get_artifacts_response));
  ASSERT_EQ(get_artifacts_response.artifacts_size(), 1);
  EXPECT_EQ(get_artifacts_response.artifacts(0).id(), artifact_id);
}

TEST_F(MetadataStoreTest, PutExecutionTypeTwiceChangedRemovedProperty) {
  PutExecutionTypeRequest request_1 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
              properties { key: 'property_2' value: STRING }
            }
          )");
  PutExecutionTypeResponse response_1;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(request_1, &response_1));

  PutExecutionTypeRequest request_2 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutExecutionTypeResponse response_2;
  EXPECT_EQ(tensorflow::error::ALREADY_EXISTS,
            metadata_store_->PutExecutionType(request_2, &response_2).code());
}

TEST_F(MetadataStoreTest, PutEventGetEvents) {
  PutExecutionTypeRequest put_execution_type_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: { name: 'test_type' }
          )");
  PutExecutionTypeResponse put_execution_type_response;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(put_execution_type_request,
                                                 &put_execution_type_response));
  ASSERT_TRUE(put_execution_type_response.has_type_id());

  PutExecutionsRequest put_executions_request =
      ParseTextProtoOrDie<PutExecutionsRequest>(R"(
        executions: {}
      )");
  put_executions_request.mutable_executions(0)->set_type_id(
      put_execution_type_response.type_id());
  PutExecutionsResponse put_executions_response;
  TF_ASSERT_OK(metadata_store_->PutExecutions(put_executions_request,
                                              &put_executions_response));
  ASSERT_EQ(put_executions_response.execution_ids_size(), 1);

  PutArtifactTypeRequest put_artifact_type_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: { name: 'test_type' }
          )");
  PutArtifactTypeResponse put_artifact_type_response;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(put_artifact_type_request,
                                                &put_artifact_type_response));
  ASSERT_TRUE(put_artifact_type_response.has_type_id());
  PutArtifactsRequest put_artifacts_request =
      ParseTextProtoOrDie<PutArtifactsRequest>(R"(
        artifacts: {}
      )");
  put_artifacts_request.mutable_artifacts(0)->set_type_id(
      put_artifact_type_response.type_id());
  PutArtifactsResponse put_artifacts_response;
  TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request,
                                             &put_artifacts_response));
  ASSERT_EQ(put_artifacts_response.artifact_ids_size(), 1);

  PutEventsRequest put_events_request = ParseTextProtoOrDie<PutEventsRequest>(
      R"(
        events: {}
      )");
  put_events_request.mutable_events(0)->set_artifact_id(
      put_artifacts_response.artifact_ids(0));
  put_events_request.mutable_events(0)->set_execution_id(
      put_executions_response.execution_ids(0));
  put_events_request.mutable_events(0)->set_type(Event::DECLARED_OUTPUT);
  PutEventsResponse put_events_response;
  TF_ASSERT_OK(
      metadata_store_->PutEvents(put_events_request, &put_events_response));

  GetEventsByArtifactIDsRequest get_events_by_artifact_ids_request;
  get_events_by_artifact_ids_request.add_artifact_ids(
      put_artifacts_response.artifact_ids(0));
  GetEventsByArtifactIDsResponse get_events_by_artifact_ids_response;
  TF_ASSERT_OK(metadata_store_->GetEventsByArtifactIDs(
      get_events_by_artifact_ids_request,
      &get_events_by_artifact_ids_response));
  ASSERT_EQ(get_events_by_artifact_ids_response.events_size(), 1);
  ASSERT_EQ(get_events_by_artifact_ids_response.events(0).execution_id(),
            put_executions_response.execution_ids(0));

  GetEventsByExecutionIDsRequest get_events_by_execution_ids_request;
  get_events_by_execution_ids_request.add_execution_ids(
      put_executions_response.execution_ids(0));
  GetEventsByExecutionIDsResponse get_events_by_execution_ids_response;
  TF_ASSERT_OK(metadata_store_->GetEventsByExecutionIDs(
      get_events_by_execution_ids_request,
      &get_events_by_execution_ids_response));
  ASSERT_EQ(get_events_by_execution_ids_response.events_size(), 1);
  EXPECT_EQ(get_events_by_artifact_ids_response.events(0).artifact_id(),
            put_artifacts_response.artifact_ids(0));
}

}  // namespace
}  // namespace ml_metadata
