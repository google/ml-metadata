/* Copyright 2020 Google LLC

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
#include "ml_metadata/metadata_store/metadata_store_test_suite.h"

#include <memory>
#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace testing {
namespace {

using ::ml_metadata::testing::EqualsProto;
using ::ml_metadata::testing::ParseTextProtoOrDie;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// A list of test utils for inserting types and nodes.
template <typename T>
void InsertTypeAndSetTypeID(MetadataStore* metadata_store, T& curr_type);

template <>
void InsertTypeAndSetTypeID(MetadataStore* metadata_store,
                            ArtifactType& curr_type) {
  PutArtifactTypeRequest put_type_request;
  *put_type_request.mutable_artifact_type() = curr_type;
  PutArtifactTypeResponse put_type_response;
  TF_ASSERT_OK(
      metadata_store->PutArtifactType(put_type_request, &put_type_response));
  ASSERT_TRUE(put_type_response.has_type_id());
  curr_type.set_id(put_type_response.type_id());
}

template <>
void InsertTypeAndSetTypeID(MetadataStore* metadata_store,
                            ExecutionType& curr_type) {
  PutExecutionTypeRequest put_type_request;
  *put_type_request.mutable_execution_type() = curr_type;
  PutExecutionTypeResponse put_type_response;
  TF_ASSERT_OK(
      metadata_store->PutExecutionType(put_type_request, &put_type_response));
  ASSERT_TRUE(put_type_response.has_type_id());
  curr_type.set_id(put_type_response.type_id());
}

template <>
void InsertTypeAndSetTypeID(MetadataStore* metadata_store,
                            ContextType& curr_type) {
  PutContextTypeRequest put_type_request;
  *put_type_request.mutable_context_type() = curr_type;
  PutContextTypeResponse put_type_response;
  TF_ASSERT_OK(
      metadata_store->PutContextType(put_type_request, &put_type_response));
  ASSERT_TRUE(put_type_response.has_type_id());
  curr_type.set_id(put_type_response.type_id());
}

template <typename NT>
void InsertNodeAndSetNodeID(MetadataStore* metadata_store,
                            std::vector<NT>& nodes);

template <>
void InsertNodeAndSetNodeID(MetadataStore* metadata_store,
                            std::vector<Artifact>& nodes) {
  PutArtifactsRequest put_nodes_request;
  absl::c_copy(nodes, google::protobuf::RepeatedFieldBackInserter(
                          put_nodes_request.mutable_artifacts()));
  PutArtifactsResponse put_nodes_response;
  TF_ASSERT_OK(
      metadata_store->PutArtifacts(put_nodes_request, &put_nodes_response));
  for (size_t i = 0; i < put_nodes_response.artifact_ids_size(); ++i) {
    nodes[i].set_id(put_nodes_response.artifact_ids(i));
  }
}

template <>
void InsertNodeAndSetNodeID(MetadataStore* metadata_store,
                            std::vector<Execution>& nodes) {
  PutExecutionsRequest put_nodes_request;
  absl::c_copy(nodes, google::protobuf::RepeatedFieldBackInserter(
                          put_nodes_request.mutable_executions()));
  PutExecutionsResponse put_nodes_response;
  TF_ASSERT_OK(
      metadata_store->PutExecutions(put_nodes_request, &put_nodes_response));
  for (size_t i = 0; i < put_nodes_response.execution_ids_size(); ++i) {
    nodes[i].set_id(put_nodes_response.execution_ids(i));
  }
}

template <>
void InsertNodeAndSetNodeID(MetadataStore* metadata_store,
                            std::vector<Context>& nodes) {
  PutContextsRequest put_nodes_request;
  absl::c_copy(nodes, google::protobuf::RepeatedFieldBackInserter(
                          put_nodes_request.mutable_contexts()));
  PutContextsResponse put_nodes_response;
  TF_ASSERT_OK(
      metadata_store->PutContexts(put_nodes_request, &put_nodes_response));
  for (size_t i = 0; i < put_nodes_response.context_ids_size(); ++i) {
    nodes[i].set_id(put_nodes_response.context_ids(i));
  }
}

// The utility function to prepare types and nodes for list node through type
// test cases.
template <typename T, typename NT>
void PrepareTypesAndNodesForListNodeThroughType(MetadataStore* metadata_store,
                                                std::vector<T>& types,
                                                std::vector<NT>& nodes) {
  // Setup: Prepares a list of type pbtxt and a list of node pbtxt.
  const std::vector<absl ::string_view> type_definitions = {
      R"( name: 'test_type'
        properties { key: 'test_property' value: STRING })",
      R"( name: 'test_type'
        version: 'v1'
        properties { key: 'test_property' value: STRING })",
  };
  const std::vector<absl::string_view> node_definitions = {
      R"( name: 'test_node_name_for_node_0_and_node_2'
        properties {
            key: 'test_property'
            value: { string_value: 'foo' }
          })",
      R"( name: 'test_node_name_for_node_1'
        properties {
            key: 'test_property'
            value: { string_value: 'bar' }
          })",
  };

  // Insert types for the later nodes insertion.
  for (absl::string_view type_definition : type_definitions) {
    T curr_type = ParseTextProtoOrDie<T>(std::string(type_definition));
    InsertTypeAndSetTypeID(metadata_store, curr_type);
    types.push_back(curr_type);
  }

  // Insert nodes under the previous types
  // `nodes[0]` and `nodes[2]` will have the same name.
  // `nodes[0]` is inserted under `types[0]` while `nodes[1]` and `nodes[2]` are
  // inserted under `types[1]`.
  nodes[0] = ParseTextProtoOrDie<NT>(std::string(node_definitions[0]));
  nodes[0].set_type_id(types[0].id());
  nodes[1] = ParseTextProtoOrDie<NT>(std::string(node_definitions[1]));
  nodes[1].set_type_id(types[1].id());
  nodes[2] = ParseTextProtoOrDie<NT>(std::string(node_definitions[0]));
  nodes[2].set_type_id(types[1].id());
  InsertNodeAndSetNodeID(metadata_store, nodes);
}

TEST_P(MetadataStoreTestSuite, InitMetadataStoreIfNotExists) {
  TF_ASSERT_OK(metadata_store_->InitMetadataStoreIfNotExists());
  // This is just to check that the metadata store was initialized.
  const PutArtifactTypeRequest put_request =
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
  const GetArtifactTypeRequest get_request =
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

TEST_P(MetadataStoreTestSuite, PutArtifactTypeGetArtifactType) {
  const PutArtifactTypeRequest put_request =
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
  const GetArtifactTypeRequest get_request =
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

TEST_P(MetadataStoreTestSuite, PutArtifactTypesGetArtifactTypes) {
  const PutArtifactTypeRequest put_request_1 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type_1'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(put_request_1, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ArtifactType type_1 = ParseTextProtoOrDie<ArtifactType>(
      R"(
        name: 'test_type_1'
        properties { key: 'property_1' value: STRING }
      )");
  type_1.set_id(put_response.type_id());

  const PutArtifactTypeRequest put_request_2 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type_2'
              properties { key: 'property_2' value: INT }
            }
          )");
  TF_ASSERT_OK(metadata_store_->PutArtifactType(put_request_2, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ArtifactType type_2 = ParseTextProtoOrDie<ArtifactType>(
      R"(
        name: 'test_type_2'
        properties { key: 'property_2' value: INT }
      )");
  type_2.set_id(put_response.type_id());

  GetArtifactTypesRequest get_request;
  GetArtifactTypesResponse got_response;
  TF_ASSERT_OK(metadata_store_->GetArtifactTypes(get_request, &got_response));
  GetArtifactTypesResponse want_response;
  *want_response.add_artifact_types() = type_1;
  *want_response.add_artifact_types() = type_2;
  EXPECT_THAT(got_response, EqualsProto(want_response));
}

TEST_P(MetadataStoreTestSuite, GetArtifactTypesWhenNoneExist) {
  GetArtifactTypesRequest get_request;
  GetArtifactTypesResponse got_response;

  // Expect OK status and empty response.
  TF_ASSERT_OK(metadata_store_->GetArtifactTypes(get_request, &got_response));
  const GetArtifactTypesResponse want_response;
  EXPECT_THAT(got_response, EqualsProto(want_response));
}

// Create an artifact, then try to create it again with an added property.
TEST_P(MetadataStoreTestSuite, PutArtifactTypeTwiceChangedAddedProperty) {
  const PutArtifactTypeRequest request_1 =
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

  const PutArtifactTypeRequest request_2 =
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

TEST_P(MetadataStoreTestSuite, PutArtifactTypeTwiceChangedRemovedProperty) {
  const PutArtifactTypeRequest request_1 =
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

  const PutArtifactTypeRequest request_2 =
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

TEST_P(MetadataStoreTestSuite, PutArtifactTypeTwiceChangedPropertyType) {
  const PutArtifactTypeRequest request_1 =
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

  const PutArtifactTypeRequest request_2 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: INT }
            }
          )");
  PutArtifactTypeResponse response_2;
  EXPECT_FALSE(metadata_store_->PutArtifactType(request_2, &response_2).ok());
}

TEST_P(MetadataStoreTestSuite, PutArtifactTypeMultipleTimesWithUpdate) {
  PutArtifactTypeRequest request_1 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse response_1;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(request_1, &response_1));

  PutArtifactTypeRequest request_2 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            can_add_fields: true
            artifact_type: {
              name: 'test_type'
              properties { key: 'property_1' value: STRING }
              properties { key: 'property_2' value: INT }
            }
          )");
  PutArtifactTypeResponse response_2;
  TF_EXPECT_OK(metadata_store_->PutArtifactType(request_2, &response_2));
  EXPECT_EQ(response_2.type_id(), response_1.type_id());
}

TEST_P(MetadataStoreTestSuite, PutArtifactTypeWithUpdateErrors) {
  PutArtifactTypeRequest request_1 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: {
              name: 'test_type'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutArtifactTypeResponse response_1;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(request_1, &response_1));
  const int64 type_id = response_1.type_id();

  {
    // can_add_fields is not set to true
    PutArtifactTypeRequest wrong_request =
        ParseTextProtoOrDie<PutArtifactTypeRequest>(
            R"(
              all_fields_match: true
              artifact_type: {
                name: 'test_type'
                properties { key: 'property_1' value: STRING }
                properties { key: 'property_2' value: INT }
              }
            )");
    PutArtifactTypeResponse response;
    EXPECT_EQ(metadata_store_->PutArtifactType(wrong_request, &response).code(),
              tensorflow::error::ALREADY_EXISTS);
  }

  {
    // cannot update an existing property
    PutArtifactTypeRequest wrong_request =
        ParseTextProtoOrDie<PutArtifactTypeRequest>(
            R"(
              all_fields_match: true
              can_add_fields: true
              artifact_type: {
                name: 'test_type'
                properties { key: 'property_1' value: DOUBLE }
              }
            )");
    wrong_request.mutable_artifact_type()->set_id(type_id);
    PutArtifactTypeResponse response;
    EXPECT_EQ(metadata_store_->PutArtifactType(wrong_request, &response).code(),
              tensorflow::error::ALREADY_EXISTS);
  }

  {
    // should provide a name
    PutArtifactTypeRequest wrong_request =
        ParseTextProtoOrDie<PutArtifactTypeRequest>(
            R"(
              all_fields_match: true
              can_add_fields: true
              artifact_type: { properties { key: 'property_2' value: INT } }
            )");
    wrong_request.mutable_artifact_type()->set_id(type_id);
    PutArtifactTypeResponse response;
    EXPECT_EQ(metadata_store_->PutArtifactType(wrong_request, &response).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }

  {
    // all stored fields should be matched
    PutArtifactTypeRequest wrong_request =
        ParseTextProtoOrDie<PutArtifactTypeRequest>(
            R"(
              all_fields_match: true
              can_add_fields: true
              artifact_type: {
                name: 'test_type'
                properties { key: 'property_2' value: INT }
              }
            )");
    wrong_request.mutable_artifact_type()->set_id(type_id);
    PutArtifactTypeResponse response;
    EXPECT_EQ(metadata_store_->PutArtifactType(wrong_request, &response).code(),
              tensorflow::error::ALREADY_EXISTS);
  }
}

TEST_P(MetadataStoreTestSuite, PutArtifactTypeSame) {
  const PutArtifactTypeRequest request_1 =
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

  const PutArtifactTypeRequest request_2 =
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

TEST_P(MetadataStoreTestSuite, PutArtifactTypeCanOmitFields) {
  PutArtifactTypeRequest request_1 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            artifact_type: {
              name: 'test_type'
              properties { key: 'property_1' value: INT }
              properties { key: 'property_2' value: STRING }
            }
          )");
  PutArtifactTypeResponse response_1;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(request_1, &response_1));
  ArtifactType stored_type = request_1.artifact_type();
  stored_type.set_id(response_1.type_id());

  // Do a list of updates with different options and verify the stored type.
  auto verify_stored_type_equals = [this](const ArtifactType& want_type) {
    GetArtifactTypesRequest get_request;
    GetArtifactTypesResponse got_response;
    TF_ASSERT_OK(metadata_store_->GetArtifactTypes(get_request, &got_response));
    EXPECT_THAT(got_response.artifact_types(), SizeIs(1));
    EXPECT_THAT(got_response.artifact_types(0), EqualsProto(want_type));
  };

  {
    // can_omit_field is false
    PutArtifactTypeRequest wrong_request =
        ParseTextProtoOrDie<PutArtifactTypeRequest>(
            R"(
              artifact_type: {
                name: 'test_type'
                properties { key: 'property_2' value: STRING }
              }
            )");
    PutArtifactTypeResponse response;
    EXPECT_EQ(metadata_store_->PutArtifactType(wrong_request, &response).code(),
              tensorflow::error::ALREADY_EXISTS);
    verify_stored_type_equals(stored_type);
  }

  {
    // can_omit_field is set to true
    PutArtifactTypeRequest correct_request =
        ParseTextProtoOrDie<PutArtifactTypeRequest>(
            R"(
              can_omit_fields: true
              artifact_type: {
                name: 'test_type'
                properties { key: 'property_2' value: STRING }
              }
            )");
    PutArtifactTypeResponse response;
    TF_EXPECT_OK(metadata_store_->PutArtifactType(correct_request, &response));
    EXPECT_EQ(response_1.type_id(), response.type_id());
    verify_stored_type_equals(stored_type);
  }

  {
    // can_omit_fields = true and can_add_fields = false
    // the new properties cannot be inserted.
    PutArtifactTypeRequest wrong_request =
        ParseTextProtoOrDie<PutArtifactTypeRequest>(
            R"(
              can_omit_fields: true
              artifact_type: {
                name: 'test_type'
                properties { key: 'property_3' value: DOUBLE }
              }
            )");
    PutArtifactTypeResponse response;
    EXPECT_EQ(metadata_store_->PutArtifactType(wrong_request, &response).code(),
              tensorflow::error::ALREADY_EXISTS);
    verify_stored_type_equals(stored_type);
  }

  {
    // can_omit_fields = true and can_add_fields = true
    // the new properties can be inserted.
    PutArtifactTypeRequest correct_request =
        ParseTextProtoOrDie<PutArtifactTypeRequest>(
            R"(
              can_add_fields: true
              can_omit_fields: true
              artifact_type: {
                name: 'test_type'
                properties { key: 'property_3' value: DOUBLE }
              }
            )");
    PutArtifactTypeResponse response;
    TF_EXPECT_OK(metadata_store_->PutArtifactType(correct_request, &response));
    EXPECT_EQ(response_1.type_id(), response.type_id());

    ArtifactType want_type = stored_type;
    (*want_type.mutable_properties())["property_3"] = ml_metadata::DOUBLE;
    verify_stored_type_equals(want_type);
  }
}

// Test for failure.
TEST_P(MetadataStoreTestSuite, GetArtifactTypeMissing) {
  const GetArtifactTypeRequest get_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetArtifactTypeResponse get_response;
  EXPECT_FALSE(
      metadata_store_->GetArtifactType(get_request, &get_response).ok());
}

TEST_P(MetadataStoreTestSuite, PutArtifactTypeGetArtifactTypesByID) {
  const PutArtifactTypeRequest put_request =
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
  ASSERT_THAT(get_response.artifact_types(), SizeIs(1));
  const ArtifactType& result = get_response.artifact_types(0);
  EXPECT_EQ(put_response.type_id(), result.id())
      << "Type ID should be the same as the type created.";
  ArtifactType expected_result = put_request.artifact_type();
  expected_result.set_id(put_response.type_id());
  EXPECT_THAT(result, EqualsProto(expected_result))
      << "The type should be the same as the one given.";
}

TEST_P(MetadataStoreTestSuite, GetArtifactTypesByIDMissing) {
  // Returns an empty list.
  GetArtifactTypesByIDRequest get_request;
  GetArtifactTypesByIDResponse get_response;
  // There are no artifact types: this one is just made up.
  get_request.add_type_ids(12);
  TF_ASSERT_OK(
      metadata_store_->GetArtifactTypesByID(get_request, &get_response));
  ASSERT_THAT(get_response.artifact_types(), SizeIs(0));
}

TEST_P(MetadataStoreTestSuite, PutArtifactTypeGetArtifactTypesByIDTwo) {
  // Check that two artifact types can be retrieved.
  const PutArtifactTypeRequest put_request_1 =
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
  const PutArtifactTypeRequest put_request_2 =
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
  ASSERT_THAT(get_response.artifact_types(), SizeIs(2));
  const ArtifactType& result_1 = get_response.artifact_types(0);
  const ArtifactType& result_2 = get_response.artifact_types(1);
  ArtifactType expected_result_1 = put_request_1.artifact_type();
  ArtifactType expected_result_2 = put_request_2.artifact_type();
  expected_result_1.set_id(put_response_1.type_id());
  expected_result_2.set_id(put_response_2.type_id());

  EXPECT_THAT(result_1, EqualsProto(expected_result_1))
      << "Type ID should be the same as the type created.";
  EXPECT_THAT(result_2, EqualsProto(expected_result_2))
      << "The name should be the same as the one returned.";
}

TEST_P(MetadataStoreTestSuite, PutExecutionTypeGetExecutionTypesByID) {
  const PutExecutionTypeRequest put_request =
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
  ASSERT_THAT(get_response.execution_types(), SizeIs(1));
  const ExecutionType& result = get_response.execution_types(0);
  EXPECT_EQ(put_response.type_id(), result.id())
      << "Type ID should be the same as the type created.";
  ExecutionType expected_result = put_request.execution_type();
  expected_result.set_id(put_response.type_id());
  EXPECT_THAT(result, EqualsProto(expected_result))
      << "The type should be the same as the one given.";
}

TEST_P(MetadataStoreTestSuite, GetExecutionTypesByIDMissing) {
  // Returns an empty list.
  GetExecutionTypesByIDRequest get_request;
  GetExecutionTypesByIDResponse get_response;
  // There are no execution types: this one is just made up.
  get_request.add_type_ids(12);
  TF_ASSERT_OK(
      metadata_store_->GetExecutionTypesByID(get_request, &get_response));
  ASSERT_THAT(get_response.execution_types(), SizeIs(0));
}

TEST_P(MetadataStoreTestSuite, PutExecutionTypeGetExecutionTypesByIDTwo) {
  // Check that two execution types can be retrieved.
  const PutExecutionTypeRequest put_request_1 =
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
  const PutExecutionTypeRequest put_request_2 =
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
  ASSERT_THAT(get_response.execution_types(), SizeIs(2));
  const ExecutionType& result_1 = get_response.execution_types(0);
  const ExecutionType& result_2 = get_response.execution_types(1);
  ExecutionType expected_result_1 = put_request_1.execution_type();
  ExecutionType expected_result_2 = put_request_2.execution_type();
  expected_result_1.set_id(put_response_1.type_id());
  expected_result_2.set_id(put_response_2.type_id());

  EXPECT_THAT(result_1, EqualsProto(expected_result_1))
      << "Type ID should be the same as the type created.";
  EXPECT_THAT(result_2, EqualsProto(expected_result_2))
      << "The name should be the same as the one returned.";
}

TEST_P(MetadataStoreTestSuite, PutTypeWithVersionsGetType) {
  // Setup: a list of type pbtxt used to upsert types in an order.
  // The list of types share the same name with different versions, and the
  // expected output is three different types.
  const std::vector<absl::string_view> type_definitions = {
      R"( name: 'test_type'
        properties { key: 'property_1' value: STRING })",
      R"( name: 'test_type'
        version: '1'
        properties { key: 'property_1' value: INT })",
      R"( name: 'test_type'
        version: '2'
        properties { key: 'property_1' value: DOUBLE })",
  };
  std::vector<ArtifactType> want_types;
  for (absl::string_view type_definition : type_definitions) {
    PutArtifactTypeRequest put_request;
    *put_request.mutable_artifact_type() =
        ParseTextProtoOrDie<ArtifactType>(std::string(type_definition));
    PutArtifactTypeResponse put_response;
    TF_ASSERT_OK(metadata_store_->PutArtifactType(put_request, &put_response));
    ASSERT_TRUE(put_response.has_type_id());
    want_types.push_back(put_request.artifact_type());
    want_types.back().set_id(put_response.type_id());
  }
  std::vector<ArtifactType> got_types;
  for (const ArtifactType& want_type : want_types) {
    GetArtifactTypeRequest get_request;
    get_request.set_type_name(want_type.name());
    if (want_type.has_version()) {
      get_request.set_type_version(want_type.version());
    }
    GetArtifactTypeResponse get_response;
    TF_ASSERT_OK(metadata_store_->GetArtifactType(get_request, &get_response));
    got_types.push_back(get_response.artifact_type());
  }
  EXPECT_THAT(got_types, Pointwise(EqualsProto<ArtifactType>(), want_types));
}

TEST_P(MetadataStoreTestSuite, EvolveTypeWithVersionsGetType) {
  // Setup: a list of type pbtxt used to upsert types in an order.
  // The list of types share the same name and version, and evolved by adding
  // more properties. The expected output is a single type with 3 properties.
  const std::vector<absl::string_view> type_definitions = {
      R"( name: 'test_type'
        version: '1'
        properties { key: 'property_1' value: STRING })",
      R"( name: 'test_type'
        version: '1'
        properties { key: 'property_1' value: STRING }
        properties { key: 'property_2' value: INT })",
      R"( name: 'test_type'
        version: '1'
        properties { key: 'property_3' value: DOUBLE })",
  };
  // Create the first version of the type with version.
  {
    PutExecutionTypeRequest put_request;
    *put_request.mutable_execution_type() =
        ParseTextProtoOrDie<ExecutionType>(std::string(type_definitions[0]));
    PutExecutionTypeResponse put_response;
    TF_ASSERT_OK(metadata_store_->PutExecutionType(put_request, &put_response));
    ASSERT_TRUE(put_response.has_type_id());
    ExecutionType want_type = put_request.execution_type();
    want_type.set_id(put_response.type_id());
    GetExecutionTypesResponse get_response;
    TF_ASSERT_OK(metadata_store_->GetExecutionTypes({}, &get_response));
    EXPECT_THAT(get_response.execution_types(),
                Pointwise(EqualsProto<ExecutionType>(), {want_type}));
  }
  // Update the stored type with version by adding properties
  {
    PutExecutionTypeRequest put_request;
    *put_request.mutable_execution_type() =
        ParseTextProtoOrDie<ExecutionType>(std::string(type_definitions[1]));
    PutExecutionTypeResponse put_response;
    // Update the type with the same name and version fails.
    EXPECT_TRUE(tensorflow::errors::IsAlreadyExists(
        metadata_store_->PutExecutionType(put_request, &put_response)));
    // The type evolution succeeds for types with versions
    put_request.set_can_add_fields(true);
    TF_ASSERT_OK(metadata_store_->PutExecutionType(put_request, &put_response));
    ExecutionType want_type = put_request.execution_type();
    want_type.set_id(put_response.type_id());
    GetExecutionTypesResponse get_response;
    TF_ASSERT_OK(metadata_store_->GetExecutionTypes({}, &get_response));
    EXPECT_THAT(get_response.execution_types(),
                Pointwise(EqualsProto<ExecutionType>(), {want_type}));
  }
  // Update the stored type with version by omitting properties
  {
    PutExecutionTypeRequest put_request;
    *put_request.mutable_execution_type() =
        ParseTextProtoOrDie<ExecutionType>(std::string(type_definitions[2]));
    // Update the type with the same name and version fails.
    put_request.set_can_add_fields(true);
    PutExecutionTypeResponse put_response;
    EXPECT_TRUE(tensorflow::errors::IsAlreadyExists(
        metadata_store_->PutExecutionType(put_request, &put_response)));
    // The type evolution succeeds for types with versions
    put_request.set_can_omit_fields(true);
    TF_ASSERT_OK(metadata_store_->PutExecutionType(put_request, &put_response));
    GetExecutionTypesResponse get_response;
    TF_ASSERT_OK(metadata_store_->GetExecutionTypes({}, &get_response));
    ExecutionType want_type = put_request.execution_type();
    want_type.set_id(put_response.type_id());
    (*want_type.mutable_properties())["property_1"] = STRING;
    (*want_type.mutable_properties())["property_2"] = INT;
    EXPECT_THAT(get_response.execution_types(),
                Pointwise(EqualsProto<ExecutionType>(), {want_type}));
  }
}

TEST_P(MetadataStoreTestSuite, TypeWithNullAndEmptyStringVersionsGetType) {
  // Test the behavior of registering types with NULL version and empty string
  // version names. The expected behavior is that two types are registered.
  const std::vector<absl::string_view> type_definitions = {
      R"( name: 'test_type'
        properties { key: 'property_1' value: STRING })",
      R"( name: 'test_type'
        version: ''
        properties { key: 'property_1' value: STRING })",
      R"( name: 'test_type_2'
        version: ''
        properties { key: 'property_1' value: STRING })",
      R"( name: 'test_type_2'
        properties { key: 'property_1' value: STRING })",
  };

  std::vector<ContextType> want_types;
  for (int i = 0; i < 4; i++) {
    PutContextTypeRequest put_request;
    *put_request.mutable_context_type() =
        ParseTextProtoOrDie<ContextType>(std::string(type_definitions[i]));
    want_types.push_back(put_request.context_type());
    PutContextTypeResponse put_response;
    const tensorflow::Status status =
        metadata_store_->PutContextType(put_request, &put_response);
    if (status.ok()) {
      ASSERT_TRUE(put_response.has_type_id());
      want_types.back().set_id(put_response.type_id());
    } else {
      EXPECT_TRUE(tensorflow::errors::IsAlreadyExists(status));
    }
  }
  GetContextTypesResponse get_response;
  TF_ASSERT_OK(metadata_store_->GetContextTypes({}, &get_response));
  EXPECT_THAT(get_response.context_types(), SizeIs(2));
  EXPECT_THAT(get_response.context_types(0), EqualsProto(want_types[0]));
  EXPECT_FALSE(get_response.context_types(1).has_version());
  EXPECT_THAT(get_response.context_types(1),
              EqualsProto(want_types[2], /*ignore_fields=*/{"version"}));
}

TEST_P(MetadataStoreTestSuite, PutTypesAndArtifactsGetArtifactsThroughType) {
  std::vector<ArtifactType> types;
  std::vector<Artifact> nodes(3);
  PrepareTypesAndNodesForListNodeThroughType(metadata_store_, types, nodes);

  auto verify_get_artifacts_by_type =
      [this](string type_name, absl::optional<string> version,
             std::vector<Artifact> want_artifacts) {
        GetArtifactsByTypeRequest get_nodes_request;
        get_nodes_request.set_type_name(type_name);
        if (version) {
          get_nodes_request.set_type_version(*version);
        }
        GetArtifactsByTypeResponse get_nodes_response;
        TF_ASSERT_OK(metadata_store_->GetArtifactsByType(get_nodes_request,
                                                         &get_nodes_response));
        EXPECT_THAT(get_nodes_response.artifacts(),
                    UnorderedPointwise(EqualsProto<Artifact>(/*ignore_fields=*/{
                                           "uri", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
                                       want_artifacts));
      };

  auto verify_get_artifact_by_type_and_name =
      [this](string type_name, absl::optional<string> version,
             string artifact_name, Artifact want_artifact) {
        GetArtifactByTypeAndNameRequest get_node_request;
        get_node_request.set_type_name(type_name);
        if (version) {
          get_node_request.set_type_version(*version);
        }
        get_node_request.set_artifact_name(artifact_name);
        GetArtifactByTypeAndNameResponse get_node_response;
        TF_ASSERT_OK(metadata_store_->GetArtifactByTypeAndName(
            get_node_request, &get_node_response));
        EXPECT_THAT(get_node_response.artifact(),
                    EqualsProto<Artifact>(want_artifact, /*ignore_fields=*/{
                                              "uri", "create_time_since_epoch",
                                              "last_update_time_since_epoch"}));
      };

  // Fetches the node according through the types.
  // Test no.1: lists the nodes by `types[0]` without version, expecting
  // `nodes[0]`.
  verify_get_artifacts_by_type(types[0].name(), types[0].version(), {nodes[0]});

  // Test no.2: lists the nodes by `types[1]` with version, expecting `nodes[1]`
  // and `nodes[2]`.
  verify_get_artifacts_by_type(types[1].name(), types[1].version(),
                               {nodes[1], nodes[2]});

  // Test no.3: lists the node by `test_node_name_for_node_0_and_node_2` and
  // `types[0]` without version, expecting `nodes[0]`.
  verify_get_artifact_by_type_and_name(types[0].name(), types[0].version(),
                                       nodes[2].name(), nodes[0]);

  // Test no.4: lists the node by `test_node_name_for_node_0_and_node_2` and
  // `types[1]` with version, expecting `nodes[2]`.
  verify_get_artifact_by_type_and_name(types[1].name(), types[1].version(),
                                       nodes[0].name(), nodes[2]);

  // Test no.5: Unknown type, expecting empty nodes / node result.
  verify_get_artifacts_by_type("Unknown_type_name",
                               absl::make_optional("Unknown_type_version"), {});
  verify_get_artifact_by_type_and_name(
      "Unknown_type_name", absl::make_optional("Unknown_type_version"),
      nodes[0].name(), {});
}

TEST_P(MetadataStoreTestSuite, PutTypesAndExecutionsGetExecutionsThroughType) {
  std::vector<ExecutionType> types;
  std::vector<Execution> nodes(3);
  PrepareTypesAndNodesForListNodeThroughType(metadata_store_, types, nodes);

  auto verify_get_executions_by_type =
      [this](string type_name, absl::optional<string> version,
             std::vector<Execution> want_executions) {
        GetExecutionsByTypeRequest get_nodes_request;
        get_nodes_request.set_type_name(type_name);
        if (version) {
          get_nodes_request.set_type_version(*version);
        }
        GetExecutionsByTypeResponse get_nodes_response;
        TF_ASSERT_OK(metadata_store_->GetExecutionsByType(get_nodes_request,
                                                          &get_nodes_response));
        EXPECT_THAT(
            get_nodes_response.executions(),
            UnorderedPointwise(
                EqualsProto<Execution>(/*ignore_fields=*/{
                    "create_time_since_epoch", "last_update_time_since_epoch"}),
                want_executions));
      };

  auto verify_get_execution_by_type_and_name =
      [this](string type_name, absl::optional<string> version,
             string execution_name, Execution want_execution) {
        GetExecutionByTypeAndNameRequest get_node_request;
        get_node_request.set_type_name(type_name);
        if (version) {
          get_node_request.set_type_version(*version);
        }
        get_node_request.set_execution_name(execution_name);
        GetExecutionByTypeAndNameResponse get_node_response;
        TF_ASSERT_OK(metadata_store_->GetExecutionByTypeAndName(
            get_node_request, &get_node_response));
        EXPECT_THAT(
            get_node_response.execution(),
            EqualsProto<Execution>(want_execution, /*ignore_fields=*/{
                                       "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
      };

  // Fetches the node according through the types.
  // Test no.1: lists the nodes by `types[0]` without version, expecting
  // `nodes[0]`.
  verify_get_executions_by_type(types[0].name(), types[0].version(),
                                {nodes[0]});

  // Test no.2: lists the nodes by `types[1]` with version, expecting `nodes[1]`
  // and `nodes[2]`.
  verify_get_executions_by_type(types[1].name(), types[1].version(),
                                {nodes[1], nodes[2]});

  // Test no.3: lists the node by `test_node_name_for_node_0_and_node_2` and
  // `types[0]` without version, expecting `nodes[0]`.
  verify_get_execution_by_type_and_name(types[0].name(), types[0].version(),
                                        nodes[2].name(), nodes[0]);

  // Test no.4: lists the node by `test_node_name_for_node_0_and_node_2` and
  // `types[1]` with version, expecting `nodes[2]`.
  verify_get_execution_by_type_and_name(types[1].name(), types[1].version(),
                                        nodes[0].name(), nodes[2]);

  // Test no.5: Unknown type, expecting empty nodes / node result.
  verify_get_executions_by_type(
      "Unknown_type_name", absl::make_optional("Unknown_type_version"), {});
  verify_get_execution_by_type_and_name(
      "Unknown_type_name", absl::make_optional("Unknown_type_version"),
      nodes[0].name(), {});
}

TEST_P(MetadataStoreTestSuite, PutTypesAndContextsGetContextsThroughType) {
  std::vector<ContextType> types;
  std::vector<Context> nodes(3);
  PrepareTypesAndNodesForListNodeThroughType(metadata_store_, types, nodes);

  auto verify_get_contexts_by_type =
      [this](string type_name, absl::optional<string> version,
             std::vector<Context> want_contexts) {
        GetContextsByTypeRequest get_nodes_request;
        get_nodes_request.set_type_name(type_name);
        if (version) {
          get_nodes_request.set_type_version(*version);
        }
        GetContextsByTypeResponse get_nodes_response;
        TF_ASSERT_OK(metadata_store_->GetContextsByType(get_nodes_request,
                                                        &get_nodes_response));
        EXPECT_THAT(get_nodes_response.contexts(),
                    UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                                           "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
                                       want_contexts));
      };

  auto verify_get_context_by_type_and_name =
      [this](string type_name, absl::optional<string> version,
             string context_name, Context want_context) {
        GetContextByTypeAndNameRequest get_node_request;
        get_node_request.set_type_name(type_name);
        if (version) {
          get_node_request.set_type_version(*version);
        }
        get_node_request.set_context_name(context_name);
        GetContextByTypeAndNameResponse get_node_response;
        TF_ASSERT_OK(metadata_store_->GetContextByTypeAndName(
            get_node_request, &get_node_response));
        EXPECT_THAT(get_node_response.context(),
                    EqualsProto<Context>(want_context, /*ignore_fields=*/{
                                             "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
      };

  // Fetches the node according through the types.
  // Test no.1: lists the nodes by `types[0]` without version, expecting
  // `nodes[0]`.
  verify_get_contexts_by_type(types[0].name(), types[0].version(), {nodes[0]});

  // Test no.2: lists the nodes by `types[1]` with version, expecting `nodes[1]`
  // and `nodes[2]`.
  verify_get_contexts_by_type(types[1].name(), types[1].version(),
                              {nodes[1], nodes[2]});

  // Test no.3: lists the node by `test_node_name_for_node_0_and_node_2` and
  // `types[0]` without version, expecting `nodes[0]`.
  verify_get_context_by_type_and_name(types[0].name(), types[0].version(),
                                      nodes[2].name(), nodes[0]);

  // Test no.4: lists the node by `test_node_name_for_node_0_and_node_2` and
  // `types[1]` with version, expecting `nodes[2]`.
  verify_get_context_by_type_and_name(types[1].name(), types[1].version(),
                                      nodes[0].name(), nodes[2]);

  // Test no.5: Unknown type, expecting empty nodes / node result.
  verify_get_contexts_by_type("Unknown_type_name",
                              absl::make_optional("Unknown_type_version"), {});
  verify_get_context_by_type_and_name(
      "Unknown_type_name", absl::make_optional("Unknown_type_version"),
      nodes[0].name(), {});
}

TEST_P(MetadataStoreTestSuite, PutArtifactsGetArtifactsByID) {
  int64 type_id;
  // Create the type
  {
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

    type_id = put_artifact_type_response.type_id();
  }

  // Put in two artifacts
  constexpr absl::string_view kArtifactTemplate = R"(
          type_id: %d
          uri: 'testuri://testing/uri'
          properties {
            key: 'property'
            value: { string_value: '%s' }
          }
      )";
  Artifact artifact1 = ParseTextProtoOrDie<Artifact>(
      absl::StrFormat(kArtifactTemplate, type_id, "1"));
  Artifact artifact2 = ParseTextProtoOrDie<Artifact>(
      absl::StrFormat(kArtifactTemplate, type_id, "2"));

  {
    PutArtifactsRequest put_artifacts_request;
    *put_artifacts_request.mutable_artifacts()->Add() = artifact1;
    *put_artifacts_request.mutable_artifacts()->Add() = artifact2;
    PutArtifactsResponse put_artifacts_response;
    TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request,
                                               &put_artifacts_response));
    ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(2));
    artifact1.set_id(put_artifacts_response.artifact_ids(0));
    artifact2.set_id(put_artifacts_response.artifact_ids(1));
  }

  // Test: retrieve by one id
  {
    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(artifact1.id());
    GetArtifactsByIDResponse get_artifacts_by_id_response;
    TF_ASSERT_OK(metadata_store_->GetArtifactsByID(
        get_artifacts_by_id_request, &get_artifacts_by_id_response));
    ASSERT_THAT(get_artifacts_by_id_response.artifacts(),
                ElementsAre(EqualsProto(
                    artifact1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  // Test: retrieve by one id
  const int64 unknown_id = artifact1.id() + artifact2.id() + 1;
  {
    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(unknown_id);
    GetArtifactsByIDResponse get_artifacts_by_id_response;
    TF_ASSERT_OK(metadata_store_->GetArtifactsByID(
        get_artifacts_by_id_request, &get_artifacts_by_id_response));
    ASSERT_THAT(get_artifacts_by_id_response.artifacts(), IsEmpty());
  }
  // Test: retrieve by multiple ids
  {
    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(unknown_id);
    get_artifacts_by_id_request.add_artifact_ids(artifact1.id());
    get_artifacts_by_id_request.add_artifact_ids(artifact2.id());
    GetArtifactsByIDResponse get_artifacts_by_id_response;
    TF_ASSERT_OK(metadata_store_->GetArtifactsByID(
        get_artifacts_by_id_request, &get_artifacts_by_id_response));
    ASSERT_THAT(
        get_artifacts_by_id_response.artifacts(),
        UnorderedElementsAre(
            EqualsProto(artifact1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(artifact2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
}

// Test creating an artifact and then updating one of its properties.
TEST_P(MetadataStoreTestSuite, PutArtifactsUpdateGetArtifactsByID) {
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
          uri: 'testuri://testing/uri'
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
  ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(1));
  const int64 artifact_id = put_artifacts_response.artifact_ids(0);

  // Now we change 3 to 2 and adds the state
  PutArtifactsRequest put_artifacts_request_2 =
      ParseTextProtoOrDie<PutArtifactsRequest>(R"(
        artifacts: {
          uri: 'testuri://testing/uri'
          properties {
            key: 'property'
            value: { string_value: '2' }
          }
          state: LIVE
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
  ASSERT_THAT(get_artifacts_by_id_response.artifacts(), SizeIs(1));
  EXPECT_THAT(get_artifacts_by_id_response.artifacts(0),
              EqualsProto(put_artifacts_request_2.artifacts(0),
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
}

TEST_P(MetadataStoreTestSuite, PutArtifactsGetArtifactsWithListOptions) {
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

  Artifact artifact = ParseTextProtoOrDie<Artifact>(R"(
    uri: 'testuri://testing/uri'
    properties {
      key: 'property'
      value: { string_value: '3' }
    }
  )");

  artifact.set_type_id(type_id);

  PutArtifactsRequest put_artifacts_request;
  // Creating 2 artifacts.
  *put_artifacts_request.add_artifacts() = artifact;
  *put_artifacts_request.add_artifacts() = artifact;
  PutArtifactsResponse put_artifacts_response;

  TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request,
                                             &put_artifacts_response));
  ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(2));
  const int64 first_artifact_id = put_artifacts_response.artifact_ids(0);
  const int64 second_artifact_id = put_artifacts_response.artifact_ids(1);

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  GetArtifactsRequest get_artifacts_request;
  *get_artifacts_request.mutable_options() = list_options;

  GetArtifactsResponse get_artifacts_response;
  TF_ASSERT_OK(metadata_store_->GetArtifacts(get_artifacts_request,
                                             &get_artifacts_response));
  EXPECT_THAT(get_artifacts_response.artifacts(), SizeIs(1));
  EXPECT_THAT(get_artifacts_response.next_page_token(), Not(IsEmpty()));
  EXPECT_EQ(get_artifacts_response.artifacts(0).id(), second_artifact_id);

  EXPECT_THAT(get_artifacts_response.artifacts(0),
              EqualsProto(put_artifacts_request.artifacts(1),
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  list_options.set_next_page_token(get_artifacts_response.next_page_token());
  *get_artifacts_request.mutable_options() = list_options;
  TF_ASSERT_OK(metadata_store_->GetArtifacts(get_artifacts_request,
                                             &get_artifacts_response));
  EXPECT_THAT(get_artifacts_response.artifacts(), SizeIs(1));
  EXPECT_THAT(get_artifacts_response.next_page_token(), IsEmpty());
  EXPECT_EQ(get_artifacts_response.artifacts(0).id(), first_artifact_id);
  EXPECT_THAT(get_artifacts_response.artifacts(0),
              EqualsProto(put_artifacts_request.artifacts(0),
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
}

TEST_P(MetadataStoreTestSuite, PutArtifactsWhenLatestUpdatedTimeChanged) {
  PutArtifactTypeRequest put_type_request;
  put_type_request.mutable_artifact_type()->set_name("test_type");
  PutArtifactTypeResponse put_type_response;
  TF_ASSERT_OK(
      metadata_store_->PutArtifactType(put_type_request, &put_type_response));
  const int64 type_id = put_type_response.type_id();
  PutArtifactsRequest put_artifacts_request;
  put_artifacts_request.add_artifacts()->set_type_id(type_id);
  PutArtifactsResponse put_artifacts_response;
  TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request,
                                             &put_artifacts_response));

  // Reads the stored artifact, and prepares update.
  GetArtifactsResponse get_artifacts_response;
  TF_ASSERT_OK(metadata_store_->GetArtifacts({}, &get_artifacts_response));
  ASSERT_THAT(get_artifacts_response.artifacts(), SizeIs(1));
  const Artifact& stored_artifact = get_artifacts_response.artifacts(0);

  // `latest_updated_time` match with the stored one. The update succeeds.
  Artifact updated_artifact = stored_artifact;
  updated_artifact.set_state(Artifact::LIVE);
  ASSERT_GT(updated_artifact.last_update_time_since_epoch(), 0);
  PutArtifactsRequest update_artifact_request;
  *update_artifact_request.add_artifacts() = updated_artifact;
  update_artifact_request.mutable_options()
      ->set_abort_if_latest_updated_time_changed(true);
  PutArtifactsResponse update_artifact_response;
  TF_EXPECT_OK(metadata_store_->PutArtifacts(update_artifact_request,
                                             &update_artifact_response));

  // If update it again with the old `latest_updated_time`, the call fails
  // with FailedPrecondition error.
  tensorflow::Status status = metadata_store_->PutArtifacts(
      update_artifact_request, &update_artifact_response);
  EXPECT_EQ(status.code(), tensorflow::error::FAILED_PRECONDITION);
  EXPECT_THAT(update_artifact_response.artifact_ids(), SizeIs(0));
}

// Test creating an execution and then updating one of its properties.
TEST_P(MetadataStoreTestSuite, PutExecutionsUpdateGetExecutionsByID) {
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
          last_known_state: RUNNING
        }
      )");
  put_executions_request.mutable_executions(0)->set_type_id(type_id);
  PutExecutionsResponse put_executions_response;
  TF_ASSERT_OK(metadata_store_->PutExecutions(put_executions_request,
                                              &put_executions_response));
  ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(1));
  const int64 execution_id = put_executions_response.execution_ids(0);

  // Now we change 3 to 2, and drop the state.
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

  EXPECT_THAT(get_executions_by_id_response.executions(0),
              EqualsProto(put_executions_request_2.executions(0),
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
}

TEST_P(MetadataStoreTestSuite, PutExecutionTypeGetExecutionType) {
  const PutExecutionTypeRequest put_request =
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
  const GetExecutionTypeRequest get_request =
      ParseTextProtoOrDie<GetExecutionTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetExecutionTypeResponse get_response;
  TF_ASSERT_OK(metadata_store_->GetExecutionType(get_request, &get_response));
  ExecutionType expected = put_request.execution_type();
  expected.set_id(put_response.type_id());
  EXPECT_THAT(get_response.execution_type(), EqualsProto(expected));
}

TEST_P(MetadataStoreTestSuite, PutExecutionTypesGetExecutionTypes) {
  const PutExecutionTypeRequest put_request_1 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type_1'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutExecutionTypeResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(put_request_1, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ExecutionType type_1 = ParseTextProtoOrDie<ExecutionType>(
      R"(
        name: 'test_type_1'
        properties { key: 'property_1' value: STRING }
      )");
  type_1.set_id(put_response.type_id());

  const PutExecutionTypeRequest put_request_2 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type_2'
              properties { key: 'property_2' value: INT }
            }
          )");
  TF_ASSERT_OK(metadata_store_->PutExecutionType(put_request_2, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ExecutionType type_2 = ParseTextProtoOrDie<ExecutionType>(
      R"(
        name: 'test_type_2'
        properties { key: 'property_2' value: INT }
      )");
  type_2.set_id(put_response.type_id());

  GetExecutionTypesRequest get_request;
  GetExecutionTypesResponse got_response;
  TF_ASSERT_OK(metadata_store_->GetExecutionTypes(get_request, &got_response));
  GetExecutionTypesResponse want_response;
  *want_response.add_execution_types() = type_1;
  *want_response.add_execution_types() = type_2;
  EXPECT_THAT(got_response, EqualsProto(want_response));
}

TEST_P(MetadataStoreTestSuite, GetExecutionTypesWhenNoneExist) {
  GetExecutionTypesRequest get_request;
  GetExecutionTypesResponse got_response;

  // Expect OK status and empty response.
  TF_ASSERT_OK(metadata_store_->GetExecutionTypes(get_request, &got_response));
  const GetExecutionTypesResponse want_response;
  EXPECT_THAT(got_response, EqualsProto(want_response));
}

TEST_P(MetadataStoreTestSuite, PutExecutionTypeTwiceChangedPropertyType) {
  const PutExecutionTypeRequest request_1 =
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

  const PutExecutionTypeRequest request_2 =
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

TEST_P(MetadataStoreTestSuite, PutExecutionTypeMultipleTimesWithUpdate) {
  PutExecutionTypeRequest request_1 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: {
              name: 'test_type'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutExecutionTypeResponse response_1;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(request_1, &response_1));

  PutExecutionTypeRequest request_2 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            can_add_fields: true
            execution_type: {
              name: 'test_type'
              properties { key: 'property_1' value: STRING }
              properties { key: 'property_2' value: INT }
            }
          )");
  request_2.mutable_execution_type()->set_id(response_1.type_id());
  PutExecutionTypeResponse response_2;
  TF_EXPECT_OK(metadata_store_->PutExecutionType(request_2, &response_2));
  EXPECT_EQ(response_2.type_id(), response_1.type_id());
}

TEST_P(MetadataStoreTestSuite, PutExecutionTypeSame) {
  const PutExecutionTypeRequest request_1 =
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

  const PutExecutionTypeRequest request_2 = request_1;
  PutExecutionTypeResponse response_2;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(request_2, &response_2));
  EXPECT_EQ(response_1.type_id(), response_2.type_id());
}

// Test for failure.
TEST_P(MetadataStoreTestSuite, GetExecutionTypeMissing) {
  const GetExecutionTypeRequest get_request =
      ParseTextProtoOrDie<GetExecutionTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetExecutionTypeResponse get_response;
  EXPECT_EQ(
      tensorflow::error::NOT_FOUND,
      metadata_store_->GetExecutionType(get_request, &get_response).code());
}

TEST_P(MetadataStoreTestSuite, PutExecutionsGetExecutionByID) {
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

  // Setup: Insert two executions
  Execution execution1;
  Execution execution2;
  {
    PutExecutionsRequest put_executions_request =
        ParseTextProtoOrDie<PutExecutionsRequest>(R"(
          executions: {
            properties {
              key: 'property'
              value: { string_value: '3' }
            }
            last_known_state: RUNNING
          }
          executions: {
            properties {
              key: 'property'
              value: { string_value: '2' }
            }
            last_known_state: CANCELED
          }
        )");
    put_executions_request.mutable_executions(0)->set_type_id(type_id);
    put_executions_request.mutable_executions(1)->set_type_id(type_id);
    PutExecutionsResponse put_executions_response;

    TF_ASSERT_OK(metadata_store_->PutExecutions(put_executions_request,
                                                &put_executions_response));
    ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(2));
    execution1 = put_executions_request.executions(0);
    execution1.set_id(put_executions_response.execution_ids(0));
    execution2 = put_executions_request.executions(1);
    execution2.set_id(put_executions_response.execution_ids(1));
  }
  // Test: Get by a single id
  {
    GetExecutionsByIDRequest get_executions_by_id_request;
    get_executions_by_id_request.add_execution_ids(execution1.id());
    GetExecutionsByIDResponse get_executions_by_id_response;
    TF_ASSERT_OK(metadata_store_->GetExecutionsByID(
        get_executions_by_id_request, &get_executions_by_id_response));
    EXPECT_THAT(get_executions_by_id_response.executions(),
                ElementsAre(EqualsProto(
                    execution1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  // Test: Get by a single unknown id
  const int64 unknown_id = execution1.id() + execution2.id() + 1;
  {
    GetExecutionsByIDRequest get_executions_by_id_request;
    get_executions_by_id_request.add_execution_ids(unknown_id);
    GetExecutionsByIDResponse get_executions_by_id_response;
    TF_ASSERT_OK(metadata_store_->GetExecutionsByID(
        get_executions_by_id_request, &get_executions_by_id_response));
    EXPECT_THAT(get_executions_by_id_response.executions(), IsEmpty());
  }
  // Test: Get by a several ids
  {
    GetExecutionsByIDRequest get_executions_by_id_request;
    get_executions_by_id_request.add_execution_ids(execution1.id());
    get_executions_by_id_request.add_execution_ids(execution2.id());
    get_executions_by_id_request.add_execution_ids(unknown_id);
    GetExecutionsByIDResponse get_executions_by_id_response;
    TF_ASSERT_OK(metadata_store_->GetExecutionsByID(
        get_executions_by_id_request, &get_executions_by_id_response));
    EXPECT_THAT(
        get_executions_by_id_response.executions(),
        UnorderedElementsAre(
            EqualsProto(execution1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(execution2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
}

TEST_P(MetadataStoreTestSuite, PutExecutionsGetExecutionsWithEmptyExecution) {
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
  ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(1));
  const int64 execution_id = put_executions_response.execution_ids(0);
  const GetExecutionsRequest get_executions_request;
  GetExecutionsResponse get_executions_response;
  TF_ASSERT_OK(metadata_store_->GetExecutions(get_executions_request,
                                              &get_executions_response));
  ASSERT_THAT(get_executions_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_response.executions(0),
              EqualsProto(put_executions_request.executions(0),
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  GetExecutionsByTypeRequest get_executions_by_type_request;
  GetExecutionsByTypeResponse get_executions_by_type_response;
  get_executions_by_type_request.set_type_name("test_type2");
  TF_ASSERT_OK(metadata_store_->GetExecutionsByType(
      get_executions_by_type_request, &get_executions_by_type_response));
  ASSERT_THAT(get_executions_by_type_response.executions(), SizeIs(1));
  EXPECT_EQ(get_executions_by_type_response.executions(0).id(), execution_id);

  GetExecutionsByTypeRequest get_executions_by_not_exist_type_request;
  GetExecutionsByTypeResponse get_executions_by_not_exist_type_response;
  get_executions_by_not_exist_type_request.set_type_name("not_exist_type");
  TF_ASSERT_OK(metadata_store_->GetExecutionsByType(
      get_executions_by_not_exist_type_request,
      &get_executions_by_not_exist_type_response));
  EXPECT_THAT(get_executions_by_not_exist_type_response.executions(),
              SizeIs(0));
}

TEST_P(MetadataStoreTestSuite, PutExecutionsGetExecutionsWithListOptions) {
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

  Execution execution = ParseTextProtoOrDie<Execution>(R"(
    properties {
      key: 'property'
      value: { string_value: '3' }
    }
    last_known_state: RUNNING
  )");

  execution.set_type_id(type_id);

  PutExecutionsRequest put_executions_request;
  // Creating 2 executions.
  *put_executions_request.add_executions() = execution;
  *put_executions_request.add_executions() = execution;
  PutExecutionsResponse put_executions_response;

  TF_ASSERT_OK(metadata_store_->PutExecutions(put_executions_request,
                                              &put_executions_response));
  ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(2));
  const int64 execution_id_0 = put_executions_response.execution_ids(0);
  const int64 execution_id_1 = put_executions_response.execution_ids(1);

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  GetExecutionsRequest get_executions_request;
  *get_executions_request.mutable_options() = list_options;

  GetExecutionsResponse get_executions_response;
  TF_ASSERT_OK(metadata_store_->GetExecutions(get_executions_request,
                                              &get_executions_response));
  EXPECT_THAT(get_executions_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_response.next_page_token(), Not(IsEmpty()));
  EXPECT_EQ(get_executions_response.executions(0).id(), execution_id_1);

  EXPECT_THAT(get_executions_response.executions(0),
              EqualsProto(put_executions_request.executions(1),
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  list_options.set_next_page_token(get_executions_response.next_page_token());
  *get_executions_request.mutable_options() = list_options;
  TF_ASSERT_OK(metadata_store_->GetExecutions(get_executions_request,
                                              &get_executions_response));
  EXPECT_THAT(get_executions_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_response.next_page_token(), IsEmpty());
  EXPECT_EQ(get_executions_response.executions(0).id(), execution_id_0);
  EXPECT_THAT(get_executions_response.executions(0),
              EqualsProto(put_executions_request.executions(0),
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
}

TEST_P(MetadataStoreTestSuite,
       GetArtifactAndExecutionByTypesWithEmptyDatabase) {
  GetArtifactsByTypeRequest get_artifacts_by_not_exist_type_request;
  GetArtifactsByTypeResponse get_artifacts_by_not_exist_type_response;
  get_artifacts_by_not_exist_type_request.set_type_name("artifact_type");
  TF_ASSERT_OK(metadata_store_->GetArtifactsByType(
      get_artifacts_by_not_exist_type_request,
      &get_artifacts_by_not_exist_type_response));
  EXPECT_THAT(get_artifacts_by_not_exist_type_response.artifacts(), SizeIs(0));

  GetExecutionsByTypeRequest get_executions_by_not_exist_type_request;
  GetExecutionsByTypeResponse get_executions_by_not_exist_type_response;
  get_executions_by_not_exist_type_request.set_type_name("execution_type");
  TF_ASSERT_OK(metadata_store_->GetExecutionsByType(
      get_executions_by_not_exist_type_request,
      &get_executions_by_not_exist_type_response));
  EXPECT_THAT(get_executions_by_not_exist_type_response.executions(),
              SizeIs(0));
}

TEST_P(MetadataStoreTestSuite, GetArtifactAndExecutionByTypesWithEmptyType) {
  const PutArtifactTypeRequest put_artifact_type_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: { name: 'empty_artifact_type' }
          )");
  PutArtifactTypeResponse put_artifact_type_response;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(put_artifact_type_request,
                                                &put_artifact_type_response));
  GetArtifactsByTypeRequest get_artifacts_by_empty_type_request;
  GetArtifactsByTypeResponse get_artifacts_by_empty_type_response;
  get_artifacts_by_empty_type_request.set_type_name("empty_artifact_type");
  TF_ASSERT_OK(metadata_store_->GetArtifactsByType(
      get_artifacts_by_empty_type_request,
      &get_artifacts_by_empty_type_response));
  EXPECT_THAT(get_artifacts_by_empty_type_response.artifacts(), SizeIs(0));

  const PutExecutionTypeRequest put_execution_type_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: { name: 'empty_execution_type' }
          )");
  PutExecutionTypeResponse put_execution_type_response;
  TF_ASSERT_OK(metadata_store_->PutExecutionType(put_execution_type_request,
                                                 &put_execution_type_response));
  GetExecutionsByTypeRequest get_executions_by_empty_type_request;
  GetExecutionsByTypeResponse get_executions_by_empty_type_response;
  get_executions_by_empty_type_request.set_type_name("empty_execution_type");
  TF_ASSERT_OK(metadata_store_->GetExecutionsByType(
      get_executions_by_empty_type_request,
      &get_executions_by_empty_type_response));
  EXPECT_THAT(get_executions_by_empty_type_response.executions(), SizeIs(0));
}

TEST_P(MetadataStoreTestSuite, GetArtifactByURI) {
  const PutArtifactTypeRequest put_artifact_type_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(all_fields_match: true
             artifact_type: { name: 'artifact_type' })");
  PutArtifactTypeResponse put_artifact_type_response;
  TF_ASSERT_OK(metadata_store_->PutArtifactType(put_artifact_type_request,
                                                &put_artifact_type_response));
  const int64 type_id = put_artifact_type_response.type_id();

  const GetArtifactsByURIRequest get_artifacts_by_uri_empty_db_request;
  GetArtifactsByURIResponse get_artifacts_by_uri_empty_db_response;
  TF_ASSERT_OK(metadata_store_->GetArtifactsByURI(
      get_artifacts_by_uri_empty_db_request,
      &get_artifacts_by_uri_empty_db_response));
  EXPECT_THAT(get_artifacts_by_uri_empty_db_response.artifacts(), SizeIs(0));

  PutArtifactsRequest put_artifacts_request =
      ParseTextProtoOrDie<PutArtifactsRequest>(R"(
        artifacts: { uri: 'testuri://with_one_artifact' }
        artifacts: { uri: 'testuri://with_multiple_artifacts' }
        artifacts: { uri: 'testuri://with_multiple_artifacts' }
        artifacts: {}
        artifacts: {}
        artifacts: {}
      )");
  for (int i = 0; i < put_artifacts_request.artifacts_size(); i++) {
    put_artifacts_request.mutable_artifacts(i)->set_type_id(type_id);
  }
  PutArtifactsResponse put_artifacts_response;
  TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request,
                                             &put_artifacts_response));
  ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(6));

  {
    GetArtifactsByURIRequest get_artifacts_by_uri_request;
    GetArtifactsByURIResponse get_artifacts_by_uri_response;
    get_artifacts_by_uri_request.add_uris("testuri://with_one_artifact");
    TF_ASSERT_OK(metadata_store_->GetArtifactsByURI(
        get_artifacts_by_uri_request, &get_artifacts_by_uri_response));
    EXPECT_THAT(get_artifacts_by_uri_response.artifacts(), SizeIs(1));
  }

  {
    GetArtifactsByURIRequest get_artifacts_by_uri_request;
    GetArtifactsByURIResponse get_artifacts_by_uri_response;
    get_artifacts_by_uri_request.add_uris("testuri://with_multiple_artifacts");
    TF_ASSERT_OK(metadata_store_->GetArtifactsByURI(
        get_artifacts_by_uri_request, &get_artifacts_by_uri_response));
    EXPECT_THAT(get_artifacts_by_uri_response.artifacts(), SizeIs(2));
  }

  {
    // empty uri
    GetArtifactsByURIRequest get_artifacts_by_uri_request;
    get_artifacts_by_uri_request.add_uris("");
    GetArtifactsByURIResponse get_artifacts_by_uri_response;
    TF_ASSERT_OK(metadata_store_->GetArtifactsByURI(
        get_artifacts_by_uri_request, &get_artifacts_by_uri_response));
    EXPECT_THAT(get_artifacts_by_uri_response.artifacts(), SizeIs(3));
  }

  {
    // query uri that does not exist
    GetArtifactsByURIRequest get_artifacts_by_uri_request;
    GetArtifactsByURIResponse get_artifacts_by_uri_response;
    get_artifacts_by_uri_request.add_uris("unknown_uri");
    TF_ASSERT_OK(metadata_store_->GetArtifactsByURI(
        get_artifacts_by_uri_request, &get_artifacts_by_uri_response));
    EXPECT_THAT(get_artifacts_by_uri_response.artifacts(), SizeIs(0));
  }

  {
    // query multiple uris with duplicates
    GetArtifactsByURIRequest get_artifacts_by_uri_request;
    GetArtifactsByURIResponse get_artifacts_by_uri_response;
    get_artifacts_by_uri_request.add_uris("unknown_uri");
    get_artifacts_by_uri_request.add_uris("testuri://with_one_artifact");
    get_artifacts_by_uri_request.add_uris("testuri://with_one_artifact");
    get_artifacts_by_uri_request.add_uris("unknown_uri_2");
    get_artifacts_by_uri_request.add_uris("testuri://with_multiple_artifacts");
    get_artifacts_by_uri_request.add_uris("");

    TF_ASSERT_OK(metadata_store_->GetArtifactsByURI(
        get_artifacts_by_uri_request, &get_artifacts_by_uri_response));
    EXPECT_THAT(get_artifacts_by_uri_response.artifacts(), SizeIs(6));
  }

  {
    GetArtifactsByURIRequest request;
    const google::protobuf::Reflection* reflection = request.GetReflection();
    google::protobuf::UnknownFieldSet* fs = reflection->MutableUnknownFields(&request);
    std::string uri = "deprecated_uri_field_value";
    fs->AddLengthDelimited(1)->assign(uri);

    GetArtifactsByURIResponse response;
    tensorflow::Status s =
        metadata_store_->GetArtifactsByURI(request, &response);
    EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);
    EXPECT_TRUE(absl::StrContains(s.error_message(),
                                  "The request contains deprecated field"));
  }
}

TEST_P(MetadataStoreTestSuite, PutArtifactsGetArtifactsWithEmptyArtifact) {
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
  ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(1));
  const int64 artifact_id = put_artifacts_response.artifact_ids(0);
  GetArtifactsRequest get_artifacts_request;
  GetArtifactsResponse get_artifacts_response;
  TF_ASSERT_OK(metadata_store_->GetArtifacts(get_artifacts_request,
                                             &get_artifacts_response));
  ASSERT_THAT(get_artifacts_response.artifacts(), SizeIs(1));
  EXPECT_EQ(get_artifacts_response.artifacts(0).id(), artifact_id);

  GetArtifactsByTypeRequest get_artifacts_by_type_request;
  GetArtifactsByTypeResponse get_artifacts_by_type_response;
  get_artifacts_by_type_request.set_type_name("test_type2");
  TF_ASSERT_OK(metadata_store_->GetArtifactsByType(
      get_artifacts_by_type_request, &get_artifacts_by_type_response));
  ASSERT_THAT(get_artifacts_by_type_response.artifacts(), SizeIs(1));
  EXPECT_EQ(get_artifacts_by_type_response.artifacts(0).id(), artifact_id);

  GetArtifactsByTypeRequest get_artifacts_by_not_exist_type_request;
  GetArtifactsByTypeResponse get_artifacts_by_not_exist_type_response;
  get_artifacts_by_not_exist_type_request.set_type_name("not_exist_type");
  TF_ASSERT_OK(metadata_store_->GetArtifactsByType(
      get_artifacts_by_not_exist_type_request,
      &get_artifacts_by_not_exist_type_response));
  EXPECT_THAT(get_artifacts_by_not_exist_type_response.artifacts(), SizeIs(0));
}

TEST_P(MetadataStoreTestSuite, PutExecutionTypeTwiceChangedRemovedProperty) {
  const PutExecutionTypeRequest request_1 =
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

  const PutExecutionTypeRequest request_2 =
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

TEST_P(MetadataStoreTestSuite, PutEventGetEvents) {
  const PutExecutionTypeRequest put_execution_type_request =
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
  ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(1));

  const PutArtifactTypeRequest put_artifact_type_request =
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
  ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(1));

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
  ASSERT_THAT(get_events_by_artifact_ids_response.events(), SizeIs(1));
  ASSERT_EQ(get_events_by_artifact_ids_response.events(0).execution_id(),
            put_executions_response.execution_ids(0));

  GetEventsByExecutionIDsRequest get_events_by_execution_ids_request;
  get_events_by_execution_ids_request.add_execution_ids(
      put_executions_response.execution_ids(0));
  GetEventsByExecutionIDsResponse get_events_by_execution_ids_response;
  TF_ASSERT_OK(metadata_store_->GetEventsByExecutionIDs(
      get_events_by_execution_ids_request,
      &get_events_by_execution_ids_response));
  ASSERT_THAT(get_events_by_execution_ids_response.events(), SizeIs(1));
  EXPECT_EQ(get_events_by_artifact_ids_response.events(0).artifact_id(),
            put_artifacts_response.artifact_ids(0));
}

TEST_P(MetadataStoreTestSuite, PutTypesGetTypes) {
  const PutTypesRequest put_request = ParseTextProtoOrDie<PutTypesRequest>(
      R"(
        artifact_types: {
          name: 'test_type1'
          properties { key: 'property_1' value: STRING }
        }
        artifact_types: {
          name: 'test_type1'
          properties { key: 'property_1' value: STRING }
        }
        execution_types: {
          name: 'test_type1'
          properties { key: 'property_1' value: STRING }
        }
        execution_types: {
          name: 'test_type2'
          properties { key: 'property_1' value: DOUBLE }
        }
        context_types: {
          name: 'test_type1'
          properties { key: 'property_1' value: INT }
        }
      )");
  PutTypesResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutTypes(put_request, &put_response));
  ASSERT_THAT(put_response.artifact_type_ids(), SizeIs(2));
  // Two identical artifact types are inserted. The returned ids are the same.
  EXPECT_EQ(put_response.artifact_type_ids(0),
            put_response.artifact_type_ids(1));
  ASSERT_THAT(put_response.execution_type_ids(), SizeIs(2));
  // Two different execution types are inserted. The returned ids are different.
  EXPECT_NE(put_response.execution_type_ids(0),
            put_response.execution_type_ids(1));
  // Context type can be inserted too.
  EXPECT_THAT(put_response.context_type_ids(), SizeIs(1));

  const GetArtifactTypeRequest get_artifact_type_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>("type_name: 'test_type1'");
  GetArtifactTypeResponse get_artifact_type_response;
  TF_ASSERT_OK(metadata_store_->GetArtifactType(get_artifact_type_request,
                                                &get_artifact_type_response));
  EXPECT_EQ(put_response.artifact_type_ids(0),
            get_artifact_type_response.artifact_type().id());

  GetExecutionTypeRequest get_execution_type_request =
      ParseTextProtoOrDie<GetExecutionTypeRequest>("type_name: 'test_type2'");
  GetExecutionTypeResponse get_execution_type_response;
  TF_ASSERT_OK(metadata_store_->GetExecutionType(get_execution_type_request,
                                                 &get_execution_type_response));
  EXPECT_EQ(put_response.execution_type_ids(1),
            get_execution_type_response.execution_type().id());

  const GetContextTypeRequest get_context_type_request =
      ParseTextProtoOrDie<GetContextTypeRequest>("type_name: 'test_type1'");
  GetContextTypeResponse get_context_type_response;
  TF_ASSERT_OK(metadata_store_->GetContextType(get_context_type_request,
                                               &get_context_type_response));
  EXPECT_EQ(put_response.context_type_ids(0),
            get_context_type_response.context_type().id());
}

TEST_P(MetadataStoreTestSuite, PutTypesUpdateTypes) {
  // Insert a type first, then update it.
  const PutTypesRequest put_request = ParseTextProtoOrDie<PutTypesRequest>(
      R"(
        artifact_types: {
          name: 'test_type1'
          properties { key: 'property_1' value: STRING }
        }
      )");
  PutTypesResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutTypes(put_request, &put_response));
  ASSERT_THAT(put_response.artifact_type_ids(), SizeIs(1));

  const PutTypesRequest update_request = ParseTextProtoOrDie<PutTypesRequest>(
      R"(
        artifact_types: {
          name: 'test_type1'
          properties { key: 'property_1' value: STRING }
          properties { key: 'property_2' value: STRING }
        }
        can_add_fields: true
      )");
  PutTypesResponse update_response;
  TF_ASSERT_OK(metadata_store_->PutTypes(update_request, &update_response));
  ASSERT_THAT(update_response.artifact_type_ids(), SizeIs(1));
  EXPECT_EQ(update_response.artifact_type_ids(0),
            put_response.artifact_type_ids(0));

  const GetArtifactTypeRequest get_artifact_type_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>("type_name: 'test_type1'");
  GetArtifactTypeResponse get_artifact_type_response;
  TF_ASSERT_OK(metadata_store_->GetArtifactType(get_artifact_type_request,
                                                &get_artifact_type_response));
  ArtifactType want_artifact_type = update_request.artifact_types(0);
  want_artifact_type.set_id(update_response.artifact_type_ids(0));
  EXPECT_THAT(get_artifact_type_response.artifact_type(),
              EqualsProto(want_artifact_type));
}

TEST_P(MetadataStoreTestSuite, PutAndGetExecution) {
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"(
    artifact_types: { name: 'artifact_type' }
    execution_types: {
      name: 'execution_type'
      properties { key: 'running_status' value: STRING }
    })");
  PutTypesResponse put_types_response;
  TF_ASSERT_OK(
      metadata_store_->PutTypes(put_types_request, &put_types_response));
  int64 artifact_type_id = put_types_response.artifact_type_ids(0);
  int64 execution_type_id = put_types_response.execution_type_ids(0);

  // 1. Insert an execution first time without any artifact and event pair.
  Execution execution;
  execution.set_type_id(execution_type_id);
  (*execution.mutable_properties())["running_status"].set_string_value("INIT");
  execution.set_last_known_state(Execution::NEW);

  PutExecutionRequest put_execution_request_1;
  *put_execution_request_1.mutable_execution() = execution;
  PutExecutionResponse put_execution_response_1;
  TF_ASSERT_OK(metadata_store_->PutExecution(put_execution_request_1,
                                             &put_execution_response_1));
  execution.set_id(put_execution_response_1.execution_id());
  EXPECT_THAT(put_execution_response_1.artifact_ids(), SizeIs(0));

  // 2. Update an existing execution with an input artifact but no event
  PutExecutionRequest put_execution_request_2;
  (*execution.mutable_properties())["running_status"].set_string_value("RUN");
  execution.set_last_known_state(Execution::RUNNING);
  *put_execution_request_2.mutable_execution() = execution;
  Artifact artifact_1;
  artifact_1.set_uri("uri://an_input_artifact");
  artifact_1.set_type_id(artifact_type_id);

  *put_execution_request_2.add_artifact_event_pairs()->mutable_artifact() =
      artifact_1;
  PutExecutionResponse put_execution_response_2;
  TF_ASSERT_OK(metadata_store_->PutExecution(put_execution_request_2,
                                             &put_execution_response_2));
  // The persistent id of the execution should be the same.
  EXPECT_EQ(put_execution_response_2.execution_id(), execution.id());
  EXPECT_THAT(put_execution_response_2.artifact_ids(), SizeIs(1));
  artifact_1.set_id(put_execution_response_2.artifact_ids(0));

  // 3. Update an existing execution with existing/new artifacts with events.
  PutExecutionRequest put_execution_request_3;
  (*execution.mutable_properties())["running_status"].set_string_value("DONE");
  execution.set_last_known_state(Execution::COMPLETE);

  *put_execution_request_3.mutable_execution() = execution;
  // add an existing artifact as input, and event has artifact/execution ids
  Event event_1;
  event_1.set_artifact_id(artifact_1.id());
  event_1.set_execution_id(execution.id());
  event_1.set_type(Event::DECLARED_INPUT);
  *put_execution_request_3.add_artifact_event_pairs()->mutable_event() =
      event_1;
  // add a new artifact as output, and event has no artifact/execution ids
  Artifact artifact_2;
  artifact_2.set_uri("uri://an_output_artifact");
  artifact_2.set_type_id(artifact_type_id);
  artifact_2.set_state(Artifact::LIVE);

  Event event_2;
  event_2.set_type(Event::DECLARED_OUTPUT);
  *put_execution_request_3.add_artifact_event_pairs()->mutable_artifact() =
      artifact_2;
  *put_execution_request_3.mutable_artifact_event_pairs(1)->mutable_event() =
      event_2;
  PutExecutionResponse put_execution_response_3;
  TF_ASSERT_OK(metadata_store_->PutExecution(put_execution_request_3,
                                             &put_execution_response_3));
  EXPECT_EQ(put_execution_response_3.execution_id(), execution.id());
  EXPECT_THAT(put_execution_response_3.artifact_ids(), SizeIs(2));
  EXPECT_EQ(put_execution_response_3.artifact_ids(0), artifact_1.id());
  artifact_2.set_id(put_execution_response_3.artifact_ids(1));

  // In the end, there should be 2 artifacts, 1 execution and 2 events.
  GetArtifactsRequest get_artifacts_request;
  GetArtifactsResponse get_artifacts_response;
  TF_ASSERT_OK(metadata_store_->GetArtifacts(get_artifacts_request,
                                             &get_artifacts_response));
  ASSERT_THAT(get_artifacts_response.artifacts(), SizeIs(2));
  EXPECT_THAT(get_artifacts_response.artifacts(0),
              EqualsProto(artifact_1,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  EXPECT_THAT(get_artifacts_response.artifacts(1),
              EqualsProto(artifact_2,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  GetExecutionsRequest get_executions_request;
  GetExecutionsResponse get_executions_response;
  TF_ASSERT_OK(metadata_store_->GetExecutions(get_executions_request,
                                              &get_executions_response));
  ASSERT_THAT(get_executions_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_response.executions(0),
              EqualsProto(execution,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  GetEventsByExecutionIDsRequest get_events_request;
  get_events_request.add_execution_ids(execution.id());
  GetEventsByExecutionIDsResponse get_events_response;
  TF_ASSERT_OK(metadata_store_->GetEventsByExecutionIDs(get_events_request,
                                                        &get_events_response));
  ASSERT_THAT(get_events_response.events(), SizeIs(2));
  EXPECT_EQ(get_events_response.events(0).artifact_id(), artifact_1.id());
  EXPECT_EQ(get_events_response.events(1).artifact_id(), artifact_2.id());
}

// Put an execution with contexts.
// Setup: the test prepares a subgraph of 1 execution, 1 artifact and 2
//   contexts, then inserts them atomically with PutExecution.
// Expectation: the end state has 2 associations and 2 attributions.
TEST_P(MetadataStoreTestSuite, PutAndGetExecutionWithContext) {
  // prepares input that consists of 1 execution, 1 artifact and 2 contexts,
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"(
    artifact_types: { name: 'artifact_type' }
    context_types: { name: 'context_type' }
    execution_types: { name: 'execution_type' })");
  PutTypesResponse put_types_response;
  TF_ASSERT_OK(
      metadata_store_->PutTypes(put_types_request, &put_types_response));
  Context context1;
  context1.set_type_id(put_types_response.context_type_ids(0));
  context1.set_name("context1");
  Context context2;
  context2.set_type_id(put_types_response.context_type_ids(0));
  context2.set_name("context2");

  PutExecutionRequest put_execution_request;
  put_execution_request.mutable_execution()->set_type_id(
      put_types_response.execution_type_ids(0));
  put_execution_request.add_artifact_event_pairs()
      ->mutable_artifact()
      ->set_type_id(put_types_response.artifact_type_ids(0));
  *put_execution_request.add_contexts() = context1;
  *put_execution_request.add_contexts() = context2;

  // calls PutExecution and test end states.
  PutExecutionResponse put_execution_response;
  TF_ASSERT_OK(metadata_store_->PutExecution(put_execution_request,
                                             &put_execution_response));
  // check the nodes of the end state graph
  ASSERT_GE(put_execution_response.execution_id(), 0);
  ASSERT_THAT(put_execution_response.artifact_ids(), SizeIs(1));
  ASSERT_THAT(put_execution_response.context_ids(), SizeIs(2));
  context1.set_id(put_execution_response.context_ids(0));
  context2.set_id(put_execution_response.context_ids(1));
  GetContextsResponse get_contexts_response;
  TF_ASSERT_OK(metadata_store_->GetContexts({}, &get_contexts_response));
  EXPECT_THAT(
      get_contexts_response.contexts(),
      ElementsAre(
          EqualsProto(context1,
                      /*ignore_fields=*/{"create_time_since_epoch",
                                         "last_update_time_since_epoch"}),
          EqualsProto(context2,
                      /*ignore_fields=*/{"create_time_since_epoch",
                                         "last_update_time_since_epoch"})));

  // check attributions and associations of each context.
  for (const int64 context_id : put_execution_response.context_ids()) {
    GetArtifactsByContextRequest get_artifacts_by_context_request;
    get_artifacts_by_context_request.set_context_id(context_id);
    GetArtifactsByContextResponse get_artifacts_by_context_response;
    TF_ASSERT_OK(metadata_store_->GetArtifactsByContext(
        get_artifacts_by_context_request, &get_artifacts_by_context_response));
    ASSERT_THAT(get_artifacts_by_context_response.artifacts(), SizeIs(1));
    EXPECT_EQ(get_artifacts_by_context_response.artifacts(0).id(),
              put_execution_response.artifact_ids(0));

    GetExecutionsByContextRequest get_executions_by_context_request;
    get_executions_by_context_request.set_context_id(context_id);
    GetExecutionsByContextResponse get_executions_by_context_response;
    TF_ASSERT_OK(metadata_store_->GetExecutionsByContext(
        get_executions_by_context_request,
        &get_executions_by_context_response));
    ASSERT_THAT(get_executions_by_context_response.executions(), SizeIs(1));
    EXPECT_EQ(get_executions_by_context_response.executions(0).id(),
              put_execution_response.execution_id());
  }
}

// Tests pagination with GetExecutionsWithContext API.
TEST_P(MetadataStoreTestSuite, PutAndGetExecutionsWithContextUsingListOptions) {
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"(
    context_types: { name: 'context_type' }
    execution_types: { name: 'execution_type' })");
  PutTypesResponse put_types_response;
  TF_ASSERT_OK(
      metadata_store_->PutTypes(put_types_request, &put_types_response));
  Context context1;
  context1.set_type_id(put_types_response.context_type_ids(0));
  context1.set_name("context1");

  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = context1;
  PutContextsResponse put_contexts_response;

  TF_ASSERT_OK(metadata_store_->PutContexts(put_contexts_request,
                                            &put_contexts_response));

  ASSERT_EQ(put_contexts_response.context_ids().size(), 1);
  int64 context_id = put_contexts_response.context_ids(0);

  PutExecutionRequest put_execution_request;
  put_execution_request.mutable_execution()->set_type_id(
      put_types_response.execution_type_ids(0));
  // calls PutExecution and test end states.
  PutExecutionResponse put_execution_response;
  TF_ASSERT_OK(metadata_store_->PutExecution(put_execution_request,
                                             &put_execution_response));
  // check the nodes of the end state graph
  ASSERT_GE(put_execution_response.execution_id(), 0);
  int64 execution_id_1 = put_execution_response.execution_id();

  TF_ASSERT_OK(metadata_store_->PutExecution(put_execution_request,
                                             &put_execution_response));
  // check the nodes of the end state graph
  ASSERT_GE(put_execution_response.execution_id(), 1);
  int64 execution_id_2 = put_execution_response.execution_id();

  Association association1;
  association1.set_context_id(context_id);
  association1.set_execution_id(execution_id_1);

  Association association2;
  association2.set_context_id(context_id);
  association2.set_execution_id(execution_id_2);

  PutAttributionsAndAssociationsRequest put_attributions_associations_request;
  *put_attributions_associations_request.add_associations() = association1;
  *put_attributions_associations_request.add_associations() = association2;
  PutAttributionsAndAssociationsResponse put_attributions_associations_response;

  TF_ASSERT_OK(metadata_store_->PutAttributionsAndAssociations(
      put_attributions_associations_request,
      &put_attributions_associations_response));

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  GetExecutionsByContextRequest get_executions_by_context_request;
  get_executions_by_context_request.set_context_id(context_id);
  *get_executions_by_context_request.mutable_options() = list_options;

  GetExecutionsByContextResponse get_executions_by_context_response;
  TF_ASSERT_OK(metadata_store_->GetExecutionsByContext(
      get_executions_by_context_request, &get_executions_by_context_response));

  EXPECT_THAT(get_executions_by_context_response.executions(), SizeIs(1));
  ASSERT_EQ(get_executions_by_context_response.executions(0).id(),
            execution_id_2);
  ASSERT_FALSE(get_executions_by_context_response.next_page_token().empty());

  list_options.set_next_page_token(
      get_executions_by_context_response.next_page_token());
  *get_executions_by_context_request.mutable_options() = list_options;
  TF_ASSERT_OK(metadata_store_->GetExecutionsByContext(
      get_executions_by_context_request, &get_executions_by_context_response));

  EXPECT_THAT(get_executions_by_context_response.executions(), SizeIs(1));
  ASSERT_EQ(get_executions_by_context_response.executions(0).id(),
            execution_id_1);
  ASSERT_TRUE(get_executions_by_context_response.next_page_token().empty());
}

// Tests pagination with GetArtifactsWithContext API.
TEST_P(MetadataStoreTestSuite, PutAndGetArtifactsWithContextUsingListOptions) {
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"(
    context_types: { name: 'context_type' }
    artifact_types: { name: 'artifact_type' })");
  PutTypesResponse put_types_response;
  TF_ASSERT_OK(
      metadata_store_->PutTypes(put_types_request, &put_types_response));
  Context context1;
  context1.set_type_id(put_types_response.context_type_ids(0));
  context1.set_name("context1");

  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = context1;
  PutContextsResponse put_contexts_response;

  TF_ASSERT_OK(metadata_store_->PutContexts(put_contexts_request,
                                            &put_contexts_response));

  ASSERT_EQ(put_contexts_response.context_ids().size(), 1);
  int64 context_id = put_contexts_response.context_ids(0);

  PutArtifactsRequest put_artifacts_request_1;
  Artifact artifact1;
  artifact1.set_type_id(put_types_response.artifact_type_ids(0));
  Artifact artifact2;
  artifact2.set_type_id(put_types_response.artifact_type_ids(0));
  *put_artifacts_request_1.add_artifacts() = artifact1;
  *put_artifacts_request_1.add_artifacts() = artifact2;

  PutArtifactsResponse put_artifacts_response;
  TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request_1,
                                             &put_artifacts_response));
  ASSERT_EQ(put_artifacts_response.artifact_ids().size(), 2);
  int64 artifact_id_1 = put_artifacts_response.artifact_ids(0);
  int64 artifact_id_2 = put_artifacts_response.artifact_ids(1);

  Attribution attribution1;
  attribution1.set_context_id(context_id);
  attribution1.set_artifact_id(artifact_id_1);

  Attribution attribution2;
  attribution2.set_context_id(context_id);
  attribution2.set_artifact_id(artifact_id_2);

  PutAttributionsAndAssociationsRequest put_attributions_associations_request;
  *put_attributions_associations_request.add_attributions() = attribution1;
  *put_attributions_associations_request.add_attributions() = attribution2;
  PutAttributionsAndAssociationsResponse put_attributions_associations_response;

  TF_ASSERT_OK(metadata_store_->PutAttributionsAndAssociations(
      put_attributions_associations_request,
      &put_attributions_associations_response));

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  GetArtifactsByContextRequest get_artifacts_by_context_request;
  get_artifacts_by_context_request.set_context_id(context_id);
  *get_artifacts_by_context_request.mutable_options() = list_options;

  GetArtifactsByContextResponse get_artifacts_by_context_response;
  TF_ASSERT_OK(metadata_store_->GetArtifactsByContext(
      get_artifacts_by_context_request, &get_artifacts_by_context_response));

  EXPECT_THAT(get_artifacts_by_context_response.artifacts(), SizeIs(1));
  ASSERT_EQ(get_artifacts_by_context_response.artifacts(0).id(), artifact_id_2);
  ASSERT_FALSE(get_artifacts_by_context_response.next_page_token().empty());

  list_options.set_next_page_token(
      get_artifacts_by_context_response.next_page_token());
  *get_artifacts_by_context_request.mutable_options() = list_options;
  TF_ASSERT_OK(metadata_store_->GetArtifactsByContext(
      get_artifacts_by_context_request, &get_artifacts_by_context_response));

  EXPECT_THAT(get_artifacts_by_context_response.artifacts(), SizeIs(1));
  ASSERT_EQ(get_artifacts_by_context_response.artifacts(0).id(), artifact_id_1);
  ASSERT_TRUE(get_artifacts_by_context_response.next_page_token().empty());
}

// Call PutExecution with a new context multiple times. If
// `reuse_context_if_already_exist` is set, the call succeeds without
// `AlreadyExist` error.
TEST_P(MetadataStoreTestSuite, PutAndGetExecutionWithContextReuseOption) {
  // prepares input that consists of 1 execution and 1 context,
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"(
    context_types: { name: 'context_type' }
    execution_types: { name: 'execution_type' })");
  PutTypesResponse put_types_response;
  TF_ASSERT_OK(
      metadata_store_->PutTypes(put_types_request, &put_types_response));
  Context context;
  context.set_type_id(put_types_response.context_type_ids(0));
  context.set_name("context");
  PutExecutionRequest request;
  request.mutable_execution()->set_type_id(
      put_types_response.execution_type_ids(0));
  *request.add_contexts() = context;

  // The first call PutExecution succeeds.
  PutExecutionResponse response;
  TF_ASSERT_OK(metadata_store_->PutExecution(request, &response));
  // A call with the same request fails, as the context already exists.
  const tensorflow::Status duplicate_update_status =
      metadata_store_->PutExecution(request, &response);
  EXPECT_EQ(duplicate_update_status.code(), tensorflow::error::ALREADY_EXISTS);
  // If set `reuse_context_if_already_exist`, it succeeds.
  request.mutable_options()->set_reuse_context_if_already_exist(true);
  TF_ASSERT_OK(metadata_store_->PutExecution(request, &response));

  // Check the stored nodes, there should be 1 context and 2 executions.
  GetContextsResponse get_contexts_response;
  TF_ASSERT_OK(metadata_store_->GetContexts({}, &get_contexts_response));
  ASSERT_THAT(get_contexts_response.contexts(), SizeIs(1));
  const Context& stored_context = get_contexts_response.contexts(0);
  EXPECT_THAT(stored_context, EqualsProto(context, /*ignore_fields=*/{
                                              "id", "create_time_since_epoch",
                                              "last_update_time_since_epoch"}));
  GetExecutionsByContextRequest get_executions_by_context_request;
  get_executions_by_context_request.set_context_id(stored_context.id());
  GetExecutionsByContextResponse get_executions_by_context_response;
  TF_ASSERT_OK(metadata_store_->GetExecutionsByContext(
      get_executions_by_context_request, &get_executions_by_context_response));
  ASSERT_THAT(get_executions_by_context_response.executions(), SizeIs(2));
  EXPECT_THAT(
      get_executions_by_context_response.executions(),
      ElementsAre(
          EqualsProto(request.execution(),
                      /*ignore_fields=*/{"id", "create_time_since_epoch",
                                         "last_update_time_since_epoch"}),
          EqualsProto(request.execution(),
                      /*ignore_fields=*/{"id", "create_time_since_epoch",
                                         "last_update_time_since_epoch"})));
}

TEST_P(MetadataStoreTestSuite, PutContextTypeGetContextType) {
  const PutContextTypeRequest put_request =
      ParseTextProtoOrDie<PutContextTypeRequest>(
          R"(
            all_fields_match: true
            context_type: {
              name: 'test_type'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutContextTypeResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutContextType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());

  GetContextTypeRequest get_request =
      ParseTextProtoOrDie<GetContextTypeRequest>("type_name: 'test_type'");
  GetContextTypeResponse get_response;
  TF_ASSERT_OK(metadata_store_->GetContextType(get_request, &get_response));
  EXPECT_EQ(put_response.type_id(), get_response.context_type().id())
      << "Type ID should be the same as the type created.";
  EXPECT_EQ("test_type", get_response.context_type().name())
      << "The name should be the same as the one returned.";
}

TEST_P(MetadataStoreTestSuite, PutContextTypesGetContextTypes) {
  const PutContextTypeRequest put_request_1 =
      ParseTextProtoOrDie<PutContextTypeRequest>(
          R"(
            all_fields_match: true
            context_type: {
              name: 'test_type_1'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutContextTypeResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutContextType(put_request_1, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ContextType type_1 = ParseTextProtoOrDie<ContextType>(
      R"(
        name: 'test_type_1'
        properties { key: 'property_1' value: STRING }
      )");
  type_1.set_id(put_response.type_id());

  const PutContextTypeRequest put_request_2 =
      ParseTextProtoOrDie<PutContextTypeRequest>(
          R"(
            all_fields_match: true
            context_type: {
              name: 'test_type_2'
              properties { key: 'property_2' value: INT }
            }
          )");
  TF_ASSERT_OK(metadata_store_->PutContextType(put_request_2, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ContextType type_2 = ParseTextProtoOrDie<ContextType>(
      R"(
        name: 'test_type_2'
        properties { key: 'property_2' value: INT }
      )");
  type_2.set_id(put_response.type_id());

  GetContextTypesRequest get_request;
  GetContextTypesResponse got_response;
  TF_ASSERT_OK(metadata_store_->GetContextTypes(get_request, &got_response));
  GetContextTypesResponse want_response;
  *want_response.add_context_types() = type_1;
  *want_response.add_context_types() = type_2;
  EXPECT_THAT(got_response, EqualsProto(want_response));
}

TEST_P(MetadataStoreTestSuite, GetContextTypesWhenNoneExist) {
  GetContextTypesRequest get_request;
  GetContextTypesResponse got_response;

  // Expect OK status and empty response.
  TF_ASSERT_OK(metadata_store_->GetContextTypes(get_request, &got_response));
  const GetContextTypesResponse want_response;
  EXPECT_THAT(got_response, EqualsProto(want_response));
}

TEST_P(MetadataStoreTestSuite, PutContextTypeGetContextTypesByID) {
  const PutContextTypeRequest put_request =
      ParseTextProtoOrDie<PutContextTypeRequest>(
          R"(
            all_fields_match: true
            context_type: {
              name: 'test_type'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutContextTypeResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutContextType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());

  // Get types by exist and non-exist ids.
  GetContextTypesByIDRequest get_request;
  get_request.add_type_ids(put_response.type_id());
  get_request.add_type_ids(put_response.type_id() + 100);
  GetContextTypesByIDResponse get_response;
  TF_ASSERT_OK(
      metadata_store_->GetContextTypesByID(get_request, &get_response));
  ASSERT_THAT(get_response.context_types(), SizeIs(1));
  const ContextType& result = get_response.context_types(0);
  EXPECT_EQ(put_response.type_id(), result.id())
      << "Type ID should be the same as the type created.";
  ContextType expected_result = put_request.context_type();
  expected_result.set_id(put_response.type_id());
  EXPECT_THAT(result, EqualsProto(expected_result))
      << "The type should be the same as the one given.";
}

TEST_P(MetadataStoreTestSuite, PutContextsGetContextsWithListOptions) {
  const PutContextTypeRequest put_context_type_request =
      ParseTextProtoOrDie<PutContextTypeRequest>(
          R"(
            all_fields_match: true
            context_type: {
              name: 'test_type'
              properties { key: 'property_1' value: STRING }
            }
          )");
  PutContextTypeResponse put_context_type_response;
  TF_ASSERT_OK(metadata_store_->PutContextType(put_context_type_request,
                                               &put_context_type_response));
  ASSERT_TRUE(put_context_type_response.has_type_id());

  const int64 type_id = put_context_type_response.type_id();

  Context context = ParseTextProtoOrDie<Context>(R"(
    name: 'test_type_1'
    properties {
      key: 'property_1'
      value: { string_value: '3' }
    }
  )");

  context.set_type_id(type_id);

  PutContextsRequest put_contexts_request;
  // Creating 2 contexts.
  *put_contexts_request.add_contexts() = context;
  context.set_name("test_type_2");
  *put_contexts_request.add_contexts() = context;
  PutContextsResponse put_contexts_response;

  TF_ASSERT_OK(metadata_store_->PutContexts(put_contexts_request,
                                            &put_contexts_response));
  ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(2));
  const int64 context_id_0 = put_contexts_response.context_ids(0);
  const int64 context_id_1 = put_contexts_response.context_ids(1);

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  GetContextsRequest get_contexts_request;
  *get_contexts_request.mutable_options() = list_options;

  GetContextsResponse get_contexts_response;
  TF_ASSERT_OK(metadata_store_->GetContexts(get_contexts_request,
                                            &get_contexts_response));
  EXPECT_THAT(get_contexts_response.contexts(), SizeIs(1));
  EXPECT_THAT(get_contexts_response.next_page_token(), Not(IsEmpty()));
  EXPECT_EQ(get_contexts_response.contexts(0).id(), context_id_1);

  EXPECT_THAT(get_contexts_response.contexts(0),
              EqualsProto(put_contexts_request.contexts(1),
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  list_options.set_next_page_token(get_contexts_response.next_page_token());
  *get_contexts_request.mutable_options() = list_options;
  TF_ASSERT_OK(metadata_store_->GetContexts(get_contexts_request,
                                            &get_contexts_response));
  EXPECT_THAT(get_contexts_response.contexts(), SizeIs(1));
  EXPECT_THAT(get_contexts_response.next_page_token(), IsEmpty());
  EXPECT_EQ(get_contexts_response.contexts(0).id(), context_id_0);
  EXPECT_THAT(get_contexts_response.contexts(0),
              EqualsProto(put_contexts_request.contexts(0),
                          /*ignore_fields=*/{"id", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
}

TEST_P(MetadataStoreTestSuite, PutContextTypeUpsert) {
  const PutContextTypeRequest put_request =
      ParseTextProtoOrDie<PutContextTypeRequest>(
          R"(
            all_fields_match: true
            context_type: {
              name: 'test_type'
              properties { key: 'property_1' value: STRING }
              properties { key: 'property_2' value: INT }
            }
          )");
  PutContextTypeResponse put_response;
  TF_ASSERT_OK(metadata_store_->PutContextType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());

  // Put the same request again, the upsert returns the same id
  {
    const PutContextTypeRequest same_put_request = put_request;
    PutContextTypeResponse same_put_response;
    TF_ASSERT_OK(
        metadata_store_->PutContextType(same_put_request, &same_put_response));
    ASSERT_TRUE(same_put_response.has_type_id());
    EXPECT_EQ(same_put_response.type_id(), put_response.type_id());
  }

  // Add property when can_add_fields is set
  {
    const PutContextTypeRequest add_property_put_request =
        ParseTextProtoOrDie<PutContextTypeRequest>(
            R"(
              all_fields_match: true
              can_add_fields: true
              context_type: {
                name: 'test_type'
                properties { key: 'property_1' value: STRING }
                properties { key: 'property_2' value: INT }
                properties { key: 'new_property' value: DOUBLE }
              }
            )");
    PutContextTypeResponse response;
    TF_ASSERT_OK(
        metadata_store_->PutContextType(add_property_put_request, &response));
    ASSERT_TRUE(response.has_type_id());
    EXPECT_EQ(response.type_id(), put_response.type_id());
  }

  // Upsert fails if the type definition is changed by adding, removing, or
  // changing property type.

  // Add property with the same type name
  {
    const PutContextTypeRequest add_property_put_request =
        ParseTextProtoOrDie<PutContextTypeRequest>(
            R"(
              all_fields_match: true
              context_type: {
                name: 'test_type'
                properties { key: 'property_1' value: STRING }
                properties { key: 'property_2' value: INT }
                properties { key: 'property_3' value: DOUBLE }
              }
            )");
    PutContextTypeResponse response;
    EXPECT_FALSE(
        metadata_store_->PutContextType(add_property_put_request, &response)
            .ok());
  }

  // Remove property with the same type name
  {
    const PutContextTypeRequest missing_property_put_request =
        ParseTextProtoOrDie<PutContextTypeRequest>(
            R"(
              all_fields_match: true
              context_type: {
                name: 'test_type'
                properties { key: 'property_1' value: STRING }
              }
            )");
    PutContextTypeResponse response;
    EXPECT_FALSE(
        metadata_store_->PutContextType(missing_property_put_request, &response)
            .ok());
  }

  // Change property type with the same type name
  {
    const PutContextTypeRequest change_property_type_put_request =
        ParseTextProtoOrDie<PutContextTypeRequest>(
            R"(
              all_fields_match: true
              context_type: {
                name: 'test_type'
                properties { key: 'property_1' value: STRING }
                properties { key: 'property_2' value: STRING }
              }
            )");
    PutContextTypeResponse response;
    EXPECT_FALSE(
        metadata_store_
            ->PutContextType(change_property_type_put_request, &response)
            .ok());
  }
}

// Test creating a context and then updating one of its properties.
TEST_P(MetadataStoreTestSuite, PutContextsUpdateGetContexts) {
  // Create two context types
  const PutContextTypeRequest put_context_type_request =
      ParseTextProtoOrDie<PutContextTypeRequest>(R"(
        all_fields_match: true
        context_type: {
          name: 'test_type'
          properties { key: 'property' value: STRING }
        }
      )");
  PutContextTypeResponse put_context_type_response;
  TF_ASSERT_OK(metadata_store_->PutContextType(put_context_type_request,
                                               &put_context_type_response));
  ASSERT_TRUE(put_context_type_response.has_type_id());
  const int64 type_id = put_context_type_response.type_id();

  ContextType type2;
  type2.set_name("type2_name");
  PutContextTypeRequest put_context_type_request2;
  put_context_type_request2.set_all_fields_match(true);
  *put_context_type_request2.mutable_context_type() = type2;
  PutContextTypeResponse put_context_type_response2;
  TF_ASSERT_OK(metadata_store_->PutContextType(put_context_type_request2,
                                               &put_context_type_response2));
  ASSERT_TRUE(put_context_type_response2.has_type_id());
  const int64 type2_id = put_context_type_response2.type_id();

  PutContextsRequest put_contexts_request =
      ParseTextProtoOrDie<PutContextsRequest>(R"(
        contexts: {
          name: 'context1'
          properties {
            key: 'property'
            value: { string_value: '1' }
          }
        }
        contexts: {
          name: 'context2'
          custom_properties {
            key: 'custom'
            value: { int_value: 2 }
          }
        }
      )");
  put_contexts_request.mutable_contexts(0)->set_type_id(type_id);
  put_contexts_request.mutable_contexts(1)->set_type_id(type_id);
  PutContextsResponse put_contexts_response;
  TF_ASSERT_OK(metadata_store_->PutContexts(put_contexts_request,
                                            &put_contexts_response));
  ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(2));
  const int64 id1 = put_contexts_response.context_ids(0);
  const int64 id2 = put_contexts_response.context_ids(1);

  // Now we update context1's string value from 1 to 2.
  // and context2's int value from 2 to 3, and add a new context with type2.
  Context want_context1 = *put_contexts_request.mutable_contexts(0);
  want_context1.set_id(id1);
  (*want_context1.mutable_properties())["property"].set_string_value("2");
  Context want_context2 = *put_contexts_request.mutable_contexts(1);
  want_context2.set_id(id2);
  (*want_context2.mutable_custom_properties())["custom"].set_int_value(2);
  Context want_context3;
  want_context3.set_type_id(type2_id);
  want_context3.set_name("context3");

  PutContextsRequest put_contexts_request2;
  *put_contexts_request2.add_contexts() = want_context1;
  *put_contexts_request2.add_contexts() = want_context2;
  *put_contexts_request2.add_contexts() = want_context3;
  PutContextsResponse put_contexts_response2;
  TF_ASSERT_OK(metadata_store_->PutContexts(put_contexts_request2,
                                            &put_contexts_response2));
  ASSERT_THAT(put_contexts_response2.context_ids(), SizeIs(3));
  want_context3.set_id(put_contexts_response2.context_ids(2));

  // Test: GetContextsByID: one id
  {
    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(id1);
    GetContextsByIDResponse get_contexts_by_id_response;
    TF_ASSERT_OK(metadata_store_->GetContextsByID(
        get_contexts_by_id_request, &get_contexts_by_id_response));
    EXPECT_THAT(get_contexts_by_id_response.contexts(),
                ElementsAre(EqualsProto(
                    want_context1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  // Test: GetContextsByID: many ids + unknown id
  const int64 unknown_id =
      want_context1.id() + want_context2.id() + want_context3.id() + 1;
  {
    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(id1);
    get_contexts_by_id_request.add_context_ids(id2);
    get_contexts_by_id_request.add_context_ids(unknown_id);
    GetContextsByIDResponse get_contexts_by_id_response;
    TF_ASSERT_OK(metadata_store_->GetContextsByID(
        get_contexts_by_id_request, &get_contexts_by_id_response));
    EXPECT_THAT(
        get_contexts_by_id_response.contexts(),
        UnorderedElementsAre(
            EqualsProto(want_context1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(want_context2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  // Test: GetContextsByID: single unknown id
  {
    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(unknown_id);
    GetContextsByIDResponse get_contexts_by_id_response;
    TF_ASSERT_OK(metadata_store_->GetContextsByID(
        get_contexts_by_id_request, &get_contexts_by_id_response));
    EXPECT_THAT(get_contexts_by_id_response.contexts(), IsEmpty());
  }
  // Test: GetContextsByType
  {
    GetContextsByTypeRequest get_contexts_by_type_request;
    get_contexts_by_type_request.set_type_name("type2_name");
    GetContextsByTypeResponse get_contexts_by_type_response;
    TF_ASSERT_OK(metadata_store_->GetContextsByType(
        get_contexts_by_type_request, &get_contexts_by_type_response));
    ASSERT_THAT(get_contexts_by_type_response.contexts(), SizeIs(1));
    EXPECT_THAT(
        get_contexts_by_type_response.contexts(0),
        EqualsProto(want_context3,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
  }
  // Test: GetContextsByType with list options
  {
    GetContextsByTypeRequest request;
    request.set_type_name("test_type");
    request.mutable_options()->set_max_result_size(1);
    GetContextsByTypeResponse response;
    TF_ASSERT_OK(metadata_store_->GetContextsByType(request, &response));
    EXPECT_THAT(response.contexts(),
                ElementsAre(EqualsProto(
                    want_context1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
    ASSERT_THAT(response.next_page_token(), Not(IsEmpty()));
    request.mutable_options()->set_next_page_token(response.next_page_token());
    response.Clear();
    TF_ASSERT_OK(metadata_store_->GetContextsByType(request, &response));
    EXPECT_THAT(response.contexts(),
                ElementsAre(EqualsProto(
                    want_context2,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
    EXPECT_THAT(response.next_page_token(), IsEmpty());
  }
  // Test: GetContexts
  {
    GetContextsRequest get_contexts_request;
    GetContextsResponse get_contexts_response;
    TF_ASSERT_OK(metadata_store_->GetContexts(get_contexts_request,
                                              &get_contexts_response));
    ASSERT_THAT(get_contexts_response.contexts(), SizeIs(3));
    EXPECT_THAT(
        get_contexts_response.contexts(0),
        EqualsProto(want_context1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
    EXPECT_THAT(
        get_contexts_response.contexts(1),
        EqualsProto(want_context2,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
    EXPECT_THAT(
        get_contexts_response.contexts(2),
        EqualsProto(want_context3,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
  }
}

TEST_P(MetadataStoreTestSuite, PutAndUseAttributionsAndAssociations) {
  const PutTypesRequest put_types_request =
      ParseTextProtoOrDie<PutTypesRequest>(R"(
        artifact_types: { name: 'artifact_type' }
        execution_types: {
          name: 'execution_type'
          properties { key: 'property' value: STRING }
        })");
  PutTypesResponse put_types_response;
  TF_ASSERT_OK(
      metadata_store_->PutTypes(put_types_request, &put_types_response));
  int64 artifact_type_id = put_types_response.artifact_type_ids(0);
  int64 execution_type_id = put_types_response.execution_type_ids(0);

  const PutContextTypeRequest put_context_type_request =
      ParseTextProtoOrDie<PutContextTypeRequest>(R"(
        all_fields_match: true
        context_type: { name: 'context_type' }
      )");
  PutContextTypeResponse put_context_type_response;
  TF_ASSERT_OK(metadata_store_->PutContextType(put_context_type_request,
                                               &put_context_type_response));
  int64 context_type_id = put_context_type_response.type_id();

  Execution want_execution;
  want_execution.set_type_id(execution_type_id);
  (*want_execution.mutable_properties())["property"].set_string_value("1");
  PutExecutionsRequest put_executions_request;
  *put_executions_request.add_executions() = want_execution;
  PutExecutionsResponse put_executions_response;
  TF_ASSERT_OK(metadata_store_->PutExecutions(put_executions_request,
                                              &put_executions_response));
  ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(1));
  want_execution.set_id(put_executions_response.execution_ids(0));

  Artifact want_artifact;
  want_artifact.set_uri("testuri");
  want_artifact.set_type_id(artifact_type_id);
  (*want_artifact.mutable_custom_properties())["custom"].set_int_value(1);
  PutArtifactsRequest put_artifacts_request;
  *put_artifacts_request.add_artifacts() = want_artifact;
  PutArtifactsResponse put_artifacts_response;
  TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request,
                                             &put_artifacts_response));
  ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(1));
  want_artifact.set_id(put_artifacts_response.artifact_ids(0));

  Context want_context;
  want_context.set_name("context");
  want_context.set_type_id(context_type_id);
  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = want_context;
  PutContextsResponse put_contexts_response;
  TF_ASSERT_OK(metadata_store_->PutContexts(put_contexts_request,
                                            &put_contexts_response));
  ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(1));
  want_context.set_id(put_contexts_response.context_ids(0));

  // insert an attribution
  PutAttributionsAndAssociationsRequest request;
  Attribution* attribution = request.add_attributions();
  attribution->set_artifact_id(want_artifact.id());
  attribution->set_context_id(want_context.id());
  PutAttributionsAndAssociationsResponse response;
  TF_EXPECT_OK(
      metadata_store_->PutAttributionsAndAssociations(request, &response));

  GetContextsByArtifactRequest get_contexts_by_artifact_request;
  get_contexts_by_artifact_request.set_artifact_id(want_artifact.id());
  GetContextsByArtifactResponse get_contexts_by_artifact_response;
  TF_EXPECT_OK(metadata_store_->GetContextsByArtifact(
      get_contexts_by_artifact_request, &get_contexts_by_artifact_response));
  ASSERT_THAT(get_contexts_by_artifact_response.contexts(), SizeIs(1));
  EXPECT_THAT(get_contexts_by_artifact_response.contexts(0),
              EqualsProto(want_context,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  GetArtifactsByContextRequest get_artifacts_by_context_request;
  get_artifacts_by_context_request.set_context_id(want_context.id());
  GetArtifactsByContextResponse get_artifacts_by_context_response;
  TF_EXPECT_OK(metadata_store_->GetArtifactsByContext(
      get_artifacts_by_context_request, &get_artifacts_by_context_response));
  ASSERT_THAT(get_artifacts_by_context_response.artifacts(), SizeIs(1));
  EXPECT_THAT(get_artifacts_by_context_response.artifacts(0),
              EqualsProto(want_artifact,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  // append the association and reinsert the existing attribution.
  Association* association = request.add_associations();
  association->set_execution_id(want_execution.id());
  association->set_context_id(want_context.id());
  TF_ASSERT_OK(
      metadata_store_->PutAttributionsAndAssociations(request, &response));

  GetContextsByExecutionRequest get_contexts_by_execution_request;
  get_contexts_by_execution_request.set_execution_id(want_execution.id());
  GetContextsByExecutionResponse get_contexts_by_execution_response;
  TF_ASSERT_OK(metadata_store_->GetContextsByExecution(
      get_contexts_by_execution_request, &get_contexts_by_execution_response));
  ASSERT_THAT(get_contexts_by_execution_response.contexts(), SizeIs(1));
  EXPECT_THAT(get_contexts_by_execution_response.contexts(0),
              EqualsProto(want_context,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  GetExecutionsByContextRequest get_executions_by_context_request;
  get_executions_by_context_request.set_context_id(want_context.id());
  GetExecutionsByContextResponse get_executions_by_context_response;
  TF_ASSERT_OK(metadata_store_->GetExecutionsByContext(
      get_executions_by_context_request, &get_executions_by_context_response));
  ASSERT_THAT(get_executions_by_context_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_by_context_response.executions(0),
              EqualsProto(want_execution,
                          /*ignore_fields=*/{"create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  // Add another artifact.
  Artifact want_artifact_2;
  want_artifact_2.set_uri("testuri2");
  want_artifact_2.set_type_id(artifact_type_id);
  PutArtifactsRequest put_artifacts_request_2;
  *put_artifacts_request_2.add_artifacts() = want_artifact_2;
  PutArtifactsResponse put_artifacts_response_2;
  TF_ASSERT_OK(metadata_store_->PutArtifacts(put_artifacts_request_2,
                                             &put_artifacts_response_2));
  ASSERT_THAT(put_artifacts_response_2.artifact_ids(), SizeIs(1));
  want_artifact_2.set_id(put_artifacts_response_2.artifact_ids(0));

  // Reinsert association and append another attribution for the new Artifact.
  // Notice that the request now contains one existing association, one existing
  // attribution and a new attribution.
  Attribution* attribution_2 = request.add_attributions();
  attribution_2->set_artifact_id(want_artifact_2.id());
  attribution_2->set_context_id(want_context.id());
  TF_EXPECT_OK(
      metadata_store_->PutAttributionsAndAssociations(request, &response));

  // The new Artifact can also be retrieved.
  GetArtifactsByContextResponse get_artifacts_by_context_response_2;
  TF_EXPECT_OK(metadata_store_->GetArtifactsByContext(
      get_artifacts_by_context_request, &get_artifacts_by_context_response_2));
  ASSERT_THAT(get_artifacts_by_context_response_2.artifacts(), SizeIs(2));
  EXPECT_THAT(
      get_artifacts_by_context_response_2.artifacts(),
      UnorderedElementsAre(
          EqualsProto(want_artifact,
                      /*ignore_fields=*/{"create_time_since_epoch",
                                         "last_update_time_since_epoch"}),
          EqualsProto(want_artifact_2,
                      /*ignore_fields=*/{"create_time_since_epoch",
                                         "last_update_time_since_epoch"})));
}

TEST_P(MetadataStoreTestSuite, PutParentContextsAlreadyExistsError) {
  // Inserts a context type.
  ContextType context_type;
  context_type.set_name("context_type_name");
  PutContextTypeRequest put_type_request;
  *put_type_request.mutable_context_type() = context_type;
  PutContextTypeResponse put_type_response;
  TF_ASSERT_OK(
      metadata_store_->PutContextType(put_type_request, &put_type_response));
  context_type.set_id(put_type_response.type_id());

  // Inserts two connected contexts.
  Context context_1, context_2;
  context_1.set_name("child_context");
  context_1.set_type_id(context_type.id());
  context_2.set_name("parent_context");
  context_2.set_type_id(context_type.id());
  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = context_1;
  *put_contexts_request.add_contexts() = context_2;
  PutContextsResponse put_contexts_response;
  TF_ASSERT_OK(metadata_store_->PutContexts(put_contexts_request,
                                            &put_contexts_response));
  context_1.set_id(put_contexts_response.context_ids(0));
  context_2.set_id(put_contexts_response.context_ids(1));

  // Inserts a parent context.
  ParentContext parent_context;
  parent_context.set_parent_id(context_2.id());
  parent_context.set_child_id(context_1.id());
  PutParentContextsRequest put_parent_contexts_request;
  *put_parent_contexts_request.add_parent_contexts() = parent_context;
  PutParentContextsResponse put_parent_contexts_response;
  TF_EXPECT_OK(metadata_store_->PutParentContexts(
      put_parent_contexts_request, &put_parent_contexts_response));

  // Recreates the same parent context should returns AlreadyExists error.
  const tensorflow::Status status = metadata_store_->PutParentContexts(
      put_parent_contexts_request, &put_parent_contexts_response);
  EXPECT_EQ(status.code(), tensorflow::error::ALREADY_EXISTS);
}

TEST_P(MetadataStoreTestSuite, PutParentContextsInvalidArgumentError) {
  // Inserts a context type.
  ContextType context_type;
  context_type.set_name("context_type_name");
  PutContextTypeRequest put_type_request;
  *put_type_request.mutable_context_type() = context_type;
  PutContextTypeResponse put_type_response;
  TF_ASSERT_OK(
      metadata_store_->PutContextType(put_type_request, &put_type_response));
  context_type.set_id(put_type_response.type_id());

  // Creates two not exist context ids.
  Context stored_context;
  stored_context.set_name("stored_context");
  stored_context.set_type_id(context_type.id());
  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = stored_context;
  PutContextsResponse put_contexts_response;
  TF_ASSERT_OK(metadata_store_->PutContexts(put_contexts_request,
                                            &put_contexts_response));
  int stored_context_id = put_contexts_response.context_ids(0);
  int64 not_exist_context_id = stored_context_id + 1;
  int64 not_exist_context_id_2 = stored_context_id + 2;

  // Enumerates the case of creating parent context with invalid argument
  // (context id cannot be found in the database).
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
    PutParentContextsRequest put_parent_contexts_request;
    *put_parent_contexts_request.add_parent_contexts() = parent_context;
    PutParentContextsResponse put_parent_contexts_response;
    const tensorflow::Status status = metadata_store_->PutParentContexts(
        put_parent_contexts_request, &put_parent_contexts_response);
    EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT) << case_name;
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
  verify_is_invalid_argument(
      /*case_name=*/"both parent and child id are not valid",
      /*parent_id=*/not_exist_context_id,
      /*child_id=*/not_exist_context_id_2);
  verify_is_invalid_argument(/*case_name=*/"parent id is not valid",
                             /*parent_id=*/not_exist_context_id,
                             /*child_id=*/stored_context_id);
  verify_is_invalid_argument(/*case_name=*/"child id is not valid",
                             /*parent_id=*/stored_context_id,
                             /*child_id=*/not_exist_context_id);
}

TEST_P(MetadataStoreTestSuite, PutParentContextsAndGetLinkedContextByContext) {
  // Inserts a context type.
  ContextType context_type;
  context_type.set_name("context_type_name");
  PutContextTypeRequest put_type_request;
  *put_type_request.mutable_context_type() = context_type;
  PutContextTypeResponse put_type_response;
  TF_ASSERT_OK(
      metadata_store_->PutContextType(put_type_request, &put_type_response));
  context_type.set_id(put_type_response.type_id());

  // Creates some contexts to be inserted into the later parent context
  // relationship.
  const int num_contexts = 7;
  std::vector<Context> contexts(num_contexts);
  PutContextsRequest put_contexts_request;
  for (int i = 0; i < num_contexts; i++) {
    contexts[i].set_name(absl::StrCat("context_", i));
    contexts[i].set_type_id(context_type.id());
    *put_contexts_request.add_contexts() = contexts[i];
  }
  PutContextsResponse put_contexts_response;
  TF_ASSERT_OK(metadata_store_->PutContexts(put_contexts_request,
                                            &put_contexts_response));
  for (int i = 0; i < num_contexts; i++) {
    contexts[i].set_id(put_contexts_response.context_ids(i));
  }

  // Prepares a list of parent contexts and stores every parent context
  // relationship for each context.
  std::unordered_map<int, std::vector<Context>> want_parents;
  std::unordered_map<int, std::vector<Context>> want_children;
  PutParentContextsRequest put_parent_contexts_request;

  auto put_parent_context = [&contexts, &want_parents, &want_children,
                             &put_parent_contexts_request](int64 parent_idx,
                                                           int64 child_idx) {
    ParentContext parent_context;
    parent_context.set_parent_id(contexts[parent_idx].id());
    parent_context.set_child_id(contexts[child_idx].id());
    put_parent_contexts_request.add_parent_contexts()->CopyFrom(parent_context);
    want_parents[child_idx].push_back(contexts[parent_idx]);
    want_children[parent_idx].push_back(contexts[child_idx]);
  };

  put_parent_context(/*parent_idx=*/0, /*child_idx=*/1);
  put_parent_context(/*parent_idx=*/0, /*child_idx=*/2);
  put_parent_context(/*parent_idx=*/2, /*child_idx=*/3);
  put_parent_context(/*parent_idx=*/1, /*child_idx=*/6);
  put_parent_context(/*parent_idx=*/4, /*child_idx=*/5);
  put_parent_context(/*parent_idx=*/5, /*child_idx=*/6);

  PutParentContextsResponse put_parent_contexts_response;
  TF_ASSERT_OK(metadata_store_->PutParentContexts(
      put_parent_contexts_request, &put_parent_contexts_response));

  // Verifies the parent contexts by looking up and stored result.
  for (int i = 0; i < num_contexts; i++) {
    GetParentContextsByContextRequest get_parents_request;
    get_parents_request.set_context_id(contexts[i].id());
    GetParentContextsByContextResponse get_parents_response;
    TF_ASSERT_OK(metadata_store_->GetParentContextsByContext(
        get_parents_request, &get_parents_response));
    GetChildrenContextsByContextRequest get_children_request;
    get_children_request.set_context_id(contexts[i].id());
    GetChildrenContextsByContextResponse get_children_response;
    TF_ASSERT_OK(metadata_store_->GetChildrenContextsByContext(
        get_children_request, &get_children_response));
    EXPECT_THAT(get_parents_response.contexts(),
                SizeIs(want_parents[i].size()));
    EXPECT_THAT(get_children_response.contexts(),
                SizeIs(want_children[i].size()));
    EXPECT_THAT(get_parents_response.contexts(),
                UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                                       "create_time_since_epoch",
                                       "last_update_time_since_epoch"}),
                                   want_parents[i]));
    EXPECT_THAT(get_children_response.contexts(),
                UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                                       "create_time_since_epoch",
                                       "last_update_time_since_epoch"}),
                                   want_children[i]));
  }
}

}  // namespace
}  // namespace testing
}  // namespace ml_metadata
