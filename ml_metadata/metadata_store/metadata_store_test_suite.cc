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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/simple_types_util.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/simple_types/proto/simple_types.pb.h"

namespace ml_metadata {
namespace testing {
namespace {

using ::ml_metadata::testing::EqualsProto;
using ::ml_metadata::testing::ParseTextProtoOrDie;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
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
  ASSERT_EQ(absl::OkStatus(), metadata_store->PutArtifactType(
                                  put_type_request, &put_type_response));
  ASSERT_TRUE(put_type_response.has_type_id());
  curr_type.set_id(put_type_response.type_id());
}

template <>
void InsertTypeAndSetTypeID(MetadataStore* metadata_store,
                            ExecutionType& curr_type) {
  PutExecutionTypeRequest put_type_request;
  *put_type_request.mutable_execution_type() = curr_type;
  PutExecutionTypeResponse put_type_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store->PutExecutionType(
                                  put_type_request, &put_type_response));
  ASSERT_TRUE(put_type_response.has_type_id());
  curr_type.set_id(put_type_response.type_id());
}

template <>
void InsertTypeAndSetTypeID(MetadataStore* metadata_store,
                            ContextType& curr_type) {
  PutContextTypeRequest put_type_request;
  *put_type_request.mutable_context_type() = curr_type;
  PutContextTypeResponse put_type_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store->PutContextType(
                                  put_type_request, &put_type_response));
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
  for (const Artifact& artifact : nodes) {
    *put_nodes_request.add_artifacts() = artifact;
  }
  PutArtifactsResponse put_nodes_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store->PutArtifacts(
                                  put_nodes_request, &put_nodes_response));
  for (size_t i = 0; i < put_nodes_response.artifact_ids_size(); ++i) {
    nodes[i].set_id(put_nodes_response.artifact_ids(i));
  }
}

template <>
void InsertNodeAndSetNodeID(MetadataStore* metadata_store,
                            std::vector<Execution>& nodes) {
  PutExecutionsRequest put_nodes_request;
  for (const Execution& execution : nodes) {
    *put_nodes_request.add_executions() = execution;
  }
  PutExecutionsResponse put_nodes_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store->PutExecutions(
                                  put_nodes_request, &put_nodes_response));
  for (size_t i = 0; i < put_nodes_response.execution_ids_size(); ++i) {
    nodes[i].set_id(put_nodes_response.execution_ids(i));
  }
}

template <>
void InsertNodeAndSetNodeID(MetadataStore* metadata_store,
                            std::vector<Context>& nodes) {
  PutContextsRequest put_nodes_request;
  for (const Context& context : nodes) {
    *put_nodes_request.add_contexts() = context;
  }
  PutContextsResponse put_nodes_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store->PutContexts(put_nodes_request,
                                                          &put_nodes_response));
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

// The utility function to verify simple types creation after metadata store
// initialization.
void VerifySimpleTypesCreation(MetadataStore* metadata_store) {
  SimpleTypes simple_types;
  ASSERT_EQ(LoadSimpleTypes(simple_types), absl::OkStatus());

  for (const ArtifactType& artifact_type : simple_types.artifact_types()) {
    GetArtifactTypeRequest request;
    request.set_type_name(artifact_type.name());
    GetArtifactTypeResponse response;
    ASSERT_EQ(metadata_store->GetArtifactType(request, &response),
              absl::OkStatus());
    EXPECT_THAT(artifact_type, EqualsProto(response.artifact_type(),
                                           /*ignore_fields=*/{"id"}));
  }

  for (const ExecutionType& execution_type : simple_types.execution_types()) {
    GetExecutionTypeRequest request;
    request.set_type_name(execution_type.name());
    GetExecutionTypeResponse response;
    ASSERT_EQ(metadata_store->GetExecutionType(request, &response),
              absl::OkStatus());
    EXPECT_THAT(execution_type, EqualsProto(response.execution_type(),
                                            /*ignore_fields=*/{"id"}));
  }
}

TEST_P(MetadataStoreTestSuite, InitMetadataStoreIfNotExists) {
  ASSERT_EQ(absl::OkStatus(), metadata_store_->InitMetadataStoreIfNotExists());
  VerifySimpleTypesCreation(metadata_store_);
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ASSERT_EQ(absl::OkStatus(), metadata_store_->InitMetadataStoreIfNotExists());
  VerifySimpleTypesCreation(metadata_store_);
  const GetArtifactTypeRequest get_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetArtifactTypeResponse get_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifactType(get_request, &get_response));
  EXPECT_EQ(put_response.type_id(), get_response.artifact_type().id())
      << "Type ID should be the same as the type created.";
  EXPECT_EQ("test_type2", get_response.artifact_type().name())
      << "The name should be the same as the one returned.";
}

TEST_P(MetadataStoreTestSuite, PutArtifactTypeGetArtifactType) {
  const PutArtifactTypeRequest put_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"pb(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
              base_type: MODEL
            }
          )pb");
  PutArtifactTypeResponse put_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  const GetArtifactTypeRequest get_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetArtifactTypeResponse get_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifactType(get_request, &get_response));
  EXPECT_EQ(put_response.type_id(), get_response.artifact_type().id())
      << "Type ID should be the same as the type created.";
  EXPECT_EQ("test_type2", get_response.artifact_type().name())
      << "The name should be the same as the one returned.";
  EXPECT_EQ(put_request.artifact_type().base_type(),
            get_response.artifact_type().base_type())
      << "The base type should be the same as the one returned.";
  // Don't test all the properties, to make the serialization of the type
  // more flexible. This can be tested at other layers.
}

TEST_P(MetadataStoreTestSuite, PutArtifactTypeInsertTypeLink) {
  absl ::string_view type_definition =
      R"( name: 'test_type2'
              properties { key: 'property_1' value: STRING }
              base_type: MODEL)";
  ArtifactType expected =
      ParseTextProtoOrDie<ArtifactType>(std::string(type_definition));
  InsertTypeAndSetTypeID(metadata_store_, expected);

  const GetArtifactTypeRequest get_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>(
          R"pb(
            type_name: 'test_type2'
          )pb");
  GetArtifactTypeResponse get_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifactType(get_request, &get_response));
  EXPECT_THAT(get_response.artifact_type(), EqualsProto(expected));

  PutArtifactTypeRequest type_request;
  *type_request.mutable_artifact_type() = expected;
  {
    PutArtifactTypeRequest update_base_type_request = type_request;
    update_base_type_request.mutable_artifact_type()->set_base_type(
        ArtifactType::DATASET);
    PutArtifactTypeResponse update_base_type_response;
    ASSERT_EQ(absl::UnimplementedError("base_type update is not supported yet"),
              metadata_store_->PutArtifactType(update_base_type_request,
                                               &update_base_type_response));
  }
  {
    PutArtifactTypeRequest null_base_type_request = type_request;
    null_base_type_request.mutable_artifact_type()->clear_base_type();
    PutArtifactTypeResponse null_base_type_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutArtifactType(null_base_type_request,
                                               &null_base_type_response));
  }
  {
    PutArtifactTypeRequest delete_base_type_request = type_request;
    delete_base_type_request.mutable_artifact_type()->set_base_type(
        ArtifactType::UNSET);
    PutArtifactTypeResponse delete_base_type_response;
    ASSERT_EQ(
        absl::UnimplementedError("base_type deletion is not supported yet"),
        metadata_store_->PutArtifactType(delete_base_type_request,
                                         &delete_base_type_response));
  }
}

TEST_P(MetadataStoreTestSuite, PutArtifactTypesGetArtifactTypes) {
  const PutArtifactTypeRequest put_request_1 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"pb(
            all_fields_match: true
            artifact_type: {
              name: 'test_type_1'
              properties { key: 'property_1' value: STRING }
              base_type: DATASET
            }
          )pb");
  PutArtifactTypeResponse put_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_request_1, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ArtifactType type_1 = ParseTextProtoOrDie<ArtifactType>(
      R"pb(
        name: 'test_type_1'
        properties { key: 'property_1' value: STRING }
        base_type: DATASET
      )pb");
  type_1.set_id(put_response.type_id());

  const PutArtifactTypeRequest put_request_2 =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"pb(
            all_fields_match: true
            artifact_type: {
              name: 'test_type_2'
              properties { key: 'property_2' value: INT }
              base_type: MODEL
            }
          )pb");
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_request_2, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ArtifactType type_2 = ParseTextProtoOrDie<ArtifactType>(
      R"pb(
        name: 'test_type_2'
        properties { key: 'property_2' value: INT }
        base_type: MODEL
      )pb");
  type_2.set_id(put_response.type_id());

  GetArtifactTypesRequest get_request;
  GetArtifactTypesResponse got_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifactTypes(get_request, &got_response));

  EXPECT_THAT(got_response.artifact_types(),
              UnorderedElementsAre(EqualsProto(type_1), EqualsProto(type_2)));
}

TEST_P(MetadataStoreTestSuite, PutArtifactTypesGetArtifactTypesByExternalIds) {
  constexpr absl::string_view kArtifactTypeTemplate = R"pb(
    all_fields_match: true
    artifact_type: {
      name: '%s'
      external_id: '%s'
      properties { key: 'property' value: STRING }
    }
  )pb";
  const PutArtifactTypeRequest put_artifact_type_1_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          absl::StrFormat(kArtifactTypeTemplate, "test_artifact_type_1",
                          "artifact_type_external_id_1"));
  const PutArtifactTypeRequest put_artifact_type_2_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          absl::StrFormat(kArtifactTypeTemplate, "test_artifact_type_2",
                          "artifact_type_external_id_2"));

  // Create the types

  PutArtifactTypeResponse put_artifact_type_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_artifact_type_1_request,
                                             &put_artifact_type_response));
  ASSERT_TRUE(put_artifact_type_response.has_type_id());
  ArtifactType artifact_type1 =
      ParseTextProtoOrDie<ArtifactType>(absl::StrFormat(
          R"pb(
            name: '%s'
            external_id: '%s'
            properties { key: 'property' value: STRING }
          )pb",
          "test_artifact_type_1", "artifact_type_external_id_1"));
  artifact_type1.set_id(put_artifact_type_response.type_id());

  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_artifact_type_2_request,
                                             &put_artifact_type_response));
  ASSERT_TRUE(put_artifact_type_response.has_type_id());
  ArtifactType artifact_type2 =
      ParseTextProtoOrDie<ArtifactType>(absl::StrFormat(
          R"pb(
            name: '%s'
            external_id: '%s'
            properties { key: 'property' value: STRING }
          )pb",
          "test_artifact_type_2", "artifact_type_external_id_2"));
  artifact_type2.set_id(put_artifact_type_response.type_id());

  // Test: retrieve by one external id
  {
    GetArtifactTypesByExternalIdsRequest
        get_artifact_types_by_external_ids_request;
    get_artifact_types_by_external_ids_request.add_external_ids(
        artifact_type1.external_id());
    GetArtifactTypesByExternalIdsResponse
        get_artifact_types_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(),
              metadata_store_->GetArtifactTypesByExternalIds(
                  get_artifact_types_by_external_ids_request,
                  &get_artifact_types_by_external_ids_response));
    EXPECT_THAT(get_artifact_types_by_external_ids_response.artifact_types(),
                ElementsAre(EqualsProto(artifact_type1)));
  }
  // Test: retrieve by one non-existing external id
  {
    GetArtifactTypesByExternalIdsRequest
        get_artifact_types_by_external_ids_request;
    get_artifact_types_by_external_ids_request.add_external_ids(
        "artifact_type_external_id_absent");
    GetArtifactTypesByExternalIdsResponse
        get_artifact_types_by_external_ids_response;
    EXPECT_TRUE(absl::IsNotFound(metadata_store_->GetArtifactTypesByExternalIds(
        get_artifact_types_by_external_ids_request,
        &get_artifact_types_by_external_ids_response)));
  }
  // Test: retrieve by multiple external ids
  {
    GetArtifactTypesByExternalIdsRequest
        get_artifact_types_by_external_ids_request;

    // Can retrieve ArtifactTypes by multiple external ids
    get_artifact_types_by_external_ids_request.add_external_ids(
        artifact_type1.external_id());
    get_artifact_types_by_external_ids_request.add_external_ids(
        artifact_type2.external_id());
    GetArtifactTypesByExternalIdsResponse
        get_artifact_types_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(),
              metadata_store_->GetArtifactTypesByExternalIds(
                  get_artifact_types_by_external_ids_request,
                  &get_artifact_types_by_external_ids_response));
    EXPECT_THAT(get_artifact_types_by_external_ids_response.artifact_types(),
                UnorderedElementsAre(EqualsProto(artifact_type1),
                                     EqualsProto(artifact_type2)));

    // Will return whatever found if some of the external ids is absent
    get_artifact_types_by_external_ids_request.add_external_ids(
        "artifact_type_external_id_absent");
    EXPECT_EQ(absl::OkStatus(),
              metadata_store_->GetArtifactTypesByExternalIds(
                  get_artifact_types_by_external_ids_request,
                  &get_artifact_types_by_external_ids_response));
    EXPECT_THAT(get_artifact_types_by_external_ids_response.artifact_types(),
                UnorderedElementsAre(EqualsProto(artifact_type1),
                                     EqualsProto(artifact_type2)));
  }

  // Test retrieve by empty external id
  {
    GetArtifactTypesByExternalIdsRequest
        get_artifact_types_by_external_ids_request;
    get_artifact_types_by_external_ids_request.add_external_ids("");
    GetArtifactTypesByExternalIdsResponse
        get_artifact_types_by_external_ids_response;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_store_->GetArtifactTypesByExternalIds(
            get_artifact_types_by_external_ids_request,
            &get_artifact_types_by_external_ids_response)));
  }
}

TEST_P(MetadataStoreTestSuite, GetArtifactTypesWhenNoneExist) {
  GetArtifactTypesRequest get_request;
  GetArtifactTypesResponse got_response;

  // Expect OK status and empty response.
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifactTypes(get_request, &got_response));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(request_1, &response_1));

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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(request_1, &response_1));

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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(request_1, &response_1));

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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(request_1, &response_1));

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
  EXPECT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(request_2, &response_2));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(request_1, &response_1));
  const int64_t type_id = response_1.type_id();

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
    EXPECT_TRUE(absl::IsAlreadyExists(
        metadata_store_->PutArtifactType(wrong_request, &response)));
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
    EXPECT_TRUE(absl::IsAlreadyExists(
        metadata_store_->PutArtifactType(wrong_request, &response)));
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
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_store_->PutArtifactType(wrong_request, &response)));
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
    EXPECT_TRUE(absl::IsAlreadyExists(
        metadata_store_->PutArtifactType(wrong_request, &response)));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(request_1, &response_1));

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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(request_2, &response_2));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(request_1, &response_1));
  ArtifactType stored_type = request_1.artifact_type();
  stored_type.set_id(response_1.type_id());

  // Do a list of updates with different options and verify the stored type.
  auto verify_stored_type_equals = [this](const ArtifactType& want_type) {
    GetArtifactTypesRequest get_request;
    GetArtifactTypesResponse got_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetArtifactTypes(get_request, &got_response));
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
    EXPECT_TRUE(absl::IsAlreadyExists(
        metadata_store_->PutArtifactType(wrong_request, &response)));
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
    EXPECT_EQ(absl::OkStatus(),
              metadata_store_->PutArtifactType(correct_request, &response));
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
    EXPECT_TRUE(absl::IsAlreadyExists(
        metadata_store_->PutArtifactType(wrong_request, &response)));

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
    EXPECT_EQ(absl::OkStatus(),
              metadata_store_->PutArtifactType(correct_request, &response));
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
          R"pb(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
              base_type: MODEL
            }
          )pb");
  PutArtifactTypeResponse put_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  GetArtifactTypesByIDRequest get_request;
  GetArtifactTypesByIDResponse get_response;
  get_request.add_type_ids(put_response.type_id());
  ASSERT_EQ(absl::OkStatus(),
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
  ASSERT_EQ(absl::OkStatus(),
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
  ASSERT_EQ(absl::OkStatus(),
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_request_2, &put_response_2));

  GetArtifactTypesByIDRequest get_request;
  GetArtifactTypesByIDResponse get_response;
  get_request.add_type_ids(put_response_1.type_id());
  get_request.add_type_ids(put_response_2.type_id());
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifactTypesByID(get_request, &get_response));
  ASSERT_THAT(get_response.artifact_types(), SizeIs(2));

  ArtifactType expected_result_1 = put_request_1.artifact_type();
  ArtifactType expected_result_2 = put_request_2.artifact_type();
  expected_result_1.set_id(put_response_1.type_id());
  expected_result_2.set_id(put_response_2.type_id());

  EXPECT_THAT(get_response.artifact_types(),
              UnorderedElementsAre(EqualsProto(expected_result_1),
                                   EqualsProto(expected_result_2)));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  GetExecutionTypesByIDRequest get_request;
  GetExecutionTypesByIDResponse get_response;
  get_request.add_type_ids(put_response.type_id());
  ASSERT_EQ(absl::OkStatus(),
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

TEST_P(MetadataStoreTestSuite,
       PutExecutionTypesGetExecutionTypesByExternalIds) {
  constexpr absl::string_view kExecutionTypeTemplate = R"pb(
    all_fields_match: true
    execution_type: {
      name: '%s'
      external_id: '%s'
      properties { key: 'property' value: STRING }
    }
  )pb";
  const PutExecutionTypeRequest put_execution_type_1_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          absl::StrFormat(kExecutionTypeTemplate, "test_execution_type_1",
                          "test_execution_type_external_id_1"));
  const PutExecutionTypeRequest put_execution_type_2_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          absl::StrFormat(kExecutionTypeTemplate, "test_execution_type_2",
                          "test_execution_type_external_id_2"));

  // Create the types
  PutExecutionTypeResponse put_execution_type_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_execution_type_1_request,
                                              &put_execution_type_response));
  ASSERT_TRUE(put_execution_type_response.has_type_id());
  ExecutionType execution_type1 =
      ParseTextProtoOrDie<ExecutionType>(absl::StrFormat(
          R"pb(
            name: '%s'
            external_id: '%s'
            properties { key: 'property' value: STRING }
          )pb",
          "test_execution_type_1", "test_execution_type_external_id_1"));
  execution_type1.set_id(put_execution_type_response.type_id());

  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_execution_type_2_request,
                                              &put_execution_type_response));
  ASSERT_TRUE(put_execution_type_response.has_type_id());
  ExecutionType execution_type2 =
      ParseTextProtoOrDie<ExecutionType>(absl::StrFormat(
          R"pb(
            name: '%s'
            external_id: '%s'
            properties { key: 'property' value: STRING }
          )pb",
          "test_execution_type_2", "test_execution_type_external_id_2"));
  execution_type2.set_id(put_execution_type_response.type_id());

  // Test: retrieve by one external id
  {
    GetExecutionTypesByExternalIdsRequest
        get_execution_types_by_external_ids_request;
    get_execution_types_by_external_ids_request.add_external_ids(
        execution_type1.external_id());
    GetExecutionTypesByExternalIdsResponse
        get_execution_types_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(),
              metadata_store_->GetExecutionTypesByExternalIds(
                  get_execution_types_by_external_ids_request,
                  &get_execution_types_by_external_ids_response));
    EXPECT_THAT(get_execution_types_by_external_ids_response.execution_types(),
                ElementsAre(EqualsProto(execution_type1)));
  }
  // Test: retrieve by one non-existing external id
  {
    GetExecutionTypesByExternalIdsRequest
        get_execution_types_by_external_ids_request;
    get_execution_types_by_external_ids_request.add_external_ids(
        "execution_type_external_id_absent");
    GetExecutionTypesByExternalIdsResponse
        get_execution_types_by_external_ids_response;
    EXPECT_TRUE(
        absl::IsNotFound(metadata_store_->GetExecutionTypesByExternalIds(
            get_execution_types_by_external_ids_request,
            &get_execution_types_by_external_ids_response)));
  }
  // Test: retrieve by multiple external ids
  {
    GetExecutionTypesByExternalIdsRequest
        get_execution_types_by_external_ids_request;

    // Can retrieve ExecutionTypes by multiple external ids
    get_execution_types_by_external_ids_request.add_external_ids(
        execution_type1.external_id());
    get_execution_types_by_external_ids_request.add_external_ids(
        execution_type2.external_id());
    GetExecutionTypesByExternalIdsResponse
        get_execution_types_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(),
              metadata_store_->GetExecutionTypesByExternalIds(
                  get_execution_types_by_external_ids_request,
                  &get_execution_types_by_external_ids_response));
    EXPECT_THAT(get_execution_types_by_external_ids_response.execution_types(),
                UnorderedElementsAre(EqualsProto(execution_type1),
                                     EqualsProto(execution_type2)));

    // Will return whatever found if some of the external ids is absent
    get_execution_types_by_external_ids_request.add_external_ids(
        "execution_type_external_id_absent");
    EXPECT_EQ(absl::OkStatus(),
              metadata_store_->GetExecutionTypesByExternalIds(
                  get_execution_types_by_external_ids_request,
                  &get_execution_types_by_external_ids_response));
    EXPECT_THAT(get_execution_types_by_external_ids_response.execution_types(),
                UnorderedElementsAre(EqualsProto(execution_type1),
                                     EqualsProto(execution_type2)));
  }

  // Test retrieve by empty external id
  {
    GetExecutionTypesByExternalIdsRequest
        get_execution_types_by_external_ids_request;
    get_execution_types_by_external_ids_request.add_external_ids("");
    GetExecutionTypesByExternalIdsResponse
        get_execution_types_by_external_ids_response;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_store_->GetExecutionTypesByExternalIds(
            get_execution_types_by_external_ids_request,
            &get_execution_types_by_external_ids_response)));
  }
}

TEST_P(MetadataStoreTestSuite, GetExecutionTypesByIDMissing) {
  // Returns an empty list.
  GetExecutionTypesByIDRequest get_request;
  GetExecutionTypesByIDResponse get_response;
  // There are no execution types: this one is just made up.
  get_request.add_type_ids(12);
  ASSERT_EQ(absl::OkStatus(),
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_request_1, &put_response_1));
  ASSERT_TRUE(put_response_1.has_type_id());
  const PutExecutionTypeRequest put_request_2 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"pb(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
              base_type: TRAIN
            }
          )pb");
  PutExecutionTypeResponse put_response_2;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_request_2, &put_response_2));

  GetExecutionTypesByIDRequest get_request;
  GetExecutionTypesByIDResponse get_response;
  get_request.add_type_ids(put_response_1.type_id());
  get_request.add_type_ids(put_response_2.type_id());
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutionTypesByID(get_request, &get_response));
  ASSERT_THAT(get_response.execution_types(), SizeIs(2));
  ExecutionType expected_result_1 = put_request_1.execution_type();
  ExecutionType expected_result_2 = put_request_2.execution_type();
  expected_result_1.set_id(put_response_1.type_id());
  expected_result_2.set_id(put_response_2.type_id());

  EXPECT_THAT(get_response.execution_types(),
              UnorderedElementsAre(EqualsProto(expected_result_1),
                                   EqualsProto(expected_result_2)));
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
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutArtifactType(put_request, &put_response));
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
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetArtifactType(get_request, &get_response));
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
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutExecutionType(put_request, &put_response));
    ASSERT_TRUE(put_response.has_type_id());
    ExecutionType want_type = put_request.execution_type();
    want_type.set_id(put_response.type_id());
    GetExecutionTypesResponse get_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetExecutionTypes({}, &get_response));
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
    EXPECT_TRUE(absl::IsAlreadyExists(
        metadata_store_->PutExecutionType(put_request, &put_response)));
    // The type evolution succeeds for types with versions
    put_request.set_can_add_fields(true);
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutExecutionType(put_request, &put_response));
    ExecutionType want_type = put_request.execution_type();
    want_type.set_id(put_response.type_id());
    GetExecutionTypesResponse get_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetExecutionTypes({}, &get_response));
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
    EXPECT_TRUE(absl::IsAlreadyExists(
        metadata_store_->PutExecutionType(put_request, &put_response)));
    // The type evolution succeeds for types with versions
    put_request.set_can_omit_fields(true);
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutExecutionType(put_request, &put_response));
    GetExecutionTypesResponse get_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetExecutionTypes({}, &get_response));
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
    const absl::Status status =
        metadata_store_->PutContextType(put_request, &put_response);
    if (status.ok()) {
      ASSERT_TRUE(put_response.has_type_id());
      want_types.back().set_id(put_response.type_id());
    } else {
      EXPECT_TRUE(absl::IsAlreadyExists(status));
    }
  }
  GetContextTypesResponse get_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetContextTypes({}, &get_response));

  EXPECT_THAT(get_response.context_types(),
              UnorderedElementsAre(EqualsProto(want_types[0]),
                                   EqualsProto(want_types[3])));
}

TEST_P(MetadataStoreTestSuite, PutTypesAndArtifactsGetArtifactsThroughType) {
  std::vector<ArtifactType> types;
  std::vector<Artifact> nodes(3);
  PrepareTypesAndNodesForListNodeThroughType(metadata_store_, types, nodes);

  auto verify_get_artifacts_by_type =
      [this](string type_name, std::optional<string> version,
             std::vector<Artifact> want_artifacts) {
        GetArtifactsByTypeRequest get_nodes_request;
        get_nodes_request.set_type_name(type_name);
        if (version) {
          get_nodes_request.set_type_version(*version);
        }
        GetArtifactsByTypeResponse get_nodes_response;
        ASSERT_EQ(absl::OkStatus(),
                  metadata_store_->GetArtifactsByType(get_nodes_request,
                                                      &get_nodes_response));
        EXPECT_THAT(
            get_nodes_response.artifacts(),
            UnorderedPointwise(EqualsProto<Artifact>(/*ignore_fields=*/{
                                   "uri", "type", "create_time_since_epoch",
                                   "last_update_time_since_epoch"}),
                               want_artifacts));
      };

  auto verify_get_artifact_by_type_and_name =
      [this](string type_name, std::optional<string> version,
             string artifact_name, Artifact want_artifact) {
        GetArtifactByTypeAndNameRequest get_node_request;
        get_node_request.set_type_name(type_name);
        if (version) {
          get_node_request.set_type_version(*version);
        }
        get_node_request.set_artifact_name(artifact_name);
        GetArtifactByTypeAndNameResponse get_node_response;
        ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactByTypeAndName(
                                        get_node_request, &get_node_response));
        EXPECT_THAT(
            get_node_response.artifact(),
            EqualsProto<Artifact>(want_artifact, /*ignore_fields=*/{
                                      "uri", "type", "create_time_since_epoch",
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

TEST_P(MetadataStoreTestSuite,
       PutTypesAndArtifactsGetArtifactsThroughTypeWithOptions) {
  const int kNumNodes = 110;
  std::vector<Artifact> nodes(kNumNodes);
  ArtifactType artifact_type;
  artifact_type.set_name("test_type");
  InsertTypeAndSetTypeID(metadata_store_, artifact_type);
  Artifact artifact;
  artifact.set_type_id(artifact_type.id());
  for (int i = 0; i < kNumNodes; ++i) {
    nodes[i] = artifact;
  }
  InsertNodeAndSetNodeID(metadata_store_, nodes);

  auto call_get_artifacts_by_type =
      [this](GetArtifactsByTypeRequest get_nodes_request,
             absl::Span<Artifact> want_artifacts,
             GetArtifactsByTypeResponse& get_nodes_response) {
        ASSERT_EQ(absl::OkStatus(),
                  metadata_store_->GetArtifactsByType(get_nodes_request,
                                                      &get_nodes_response));
        ASSERT_EQ(want_artifacts.size(), get_nodes_response.artifacts_size());
        EXPECT_THAT(
            get_nodes_response.artifacts(),
            UnorderedPointwise(EqualsProto<Artifact>(/*ignore_fields=*/{
                                   "uri", "type", "create_time_since_epoch",
                                   "last_update_time_since_epoch"}),
                               want_artifacts));
      };

  GetArtifactsByTypeRequest get_nodes_request;
  get_nodes_request.set_type_name("test_type");
  GetArtifactsByTypeResponse get_nodes_response;

  // Fetches the node according through the types.
  // Test: lists the nodes without options, expecting all nodes are
  // returned.
  call_get_artifacts_by_type(get_nodes_request, absl::MakeSpan(nodes),
                             get_nodes_response);
  EXPECT_TRUE(get_nodes_response.next_page_token().empty());

  // Test: list the nodes with options.max_result_size >= 101
  // nodes, expect top 101 nodes are returned.
  // TODO(b/197879364): Consider a better solution for boundary cases.
  get_nodes_request.mutable_options()->Clear();
  get_nodes_request.mutable_options()->set_max_result_size(102);
  call_get_artifacts_by_type(
      get_nodes_request, absl::MakeSpan(nodes.data(), 101), get_nodes_response);
  EXPECT_TRUE(get_nodes_response.next_page_token().empty());

  // Test: list the nodes with options.max_result_size < 101.
  get_nodes_request.mutable_options()->Clear();
  get_nodes_request.mutable_options()->set_max_result_size(100);
  call_get_artifacts_by_type(
      get_nodes_request, absl::MakeSpan(nodes.data(), 100), get_nodes_response);
  EXPECT_FALSE(get_nodes_response.next_page_token().empty());

  // Test: lists the nodes with next page token.
  get_nodes_request.mutable_options()->Clear();
  get_nodes_request.mutable_options()->set_next_page_token(
      get_nodes_response.next_page_token());
  call_get_artifacts_by_type(get_nodes_request,
                             absl::MakeSpan(nodes.data() + 100, 10),
                             get_nodes_response);
  EXPECT_TRUE(get_nodes_response.next_page_token().empty());
}

TEST_P(MetadataStoreTestSuite, PutTypesAndExecutionsGetExecutionsThroughType) {
  std::vector<ExecutionType> types;
  std::vector<Execution> nodes(3);
  PrepareTypesAndNodesForListNodeThroughType(metadata_store_, types, nodes);

  auto verify_get_executions_by_type =
      [this](string type_name, std::optional<string> version,
             std::vector<Execution> want_executions) {
        GetExecutionsByTypeRequest get_nodes_request;
        get_nodes_request.set_type_name(type_name);
        if (version) {
          get_nodes_request.set_type_version(*version);
        }
        GetExecutionsByTypeResponse get_nodes_response;
        ASSERT_EQ(absl::OkStatus(),
                  metadata_store_->GetExecutionsByType(get_nodes_request,
                                                       &get_nodes_response));
        EXPECT_THAT(
            get_nodes_response.executions(),
            UnorderedPointwise(EqualsProto<Execution>(/*ignore_fields=*/{
                                   "type", "create_time_since_epoch",
                                   "last_update_time_since_epoch"}),
                               want_executions));
      };

  auto verify_get_execution_by_type_and_name =
      [this](string type_name, std::optional<string> version,
             string execution_name, Execution want_execution) {
        GetExecutionByTypeAndNameRequest get_node_request;
        get_node_request.set_type_name(type_name);
        if (version) {
          get_node_request.set_type_version(*version);
        }
        get_node_request.set_execution_name(execution_name);
        GetExecutionByTypeAndNameResponse get_node_response;
        ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionByTypeAndName(
                                        get_node_request, &get_node_response));
        EXPECT_THAT(
            get_node_response.execution(),
            EqualsProto<Execution>(want_execution, /*ignore_fields=*/{
                                       "type", "create_time_since_epoch",
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
      [this](string type_name, std::optional<string> version,
             std::vector<Context> want_contexts) {
        GetContextsByTypeRequest get_nodes_request;
        get_nodes_request.set_type_name(type_name);
        if (version) {
          get_nodes_request.set_type_version(*version);
        }
        GetContextsByTypeResponse get_nodes_response;
        ASSERT_EQ(absl::OkStatus(),
                  metadata_store_->GetContextsByType(get_nodes_request,
                                                     &get_nodes_response));
        EXPECT_THAT(get_nodes_response.contexts(),
                    UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                                           "type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
                                       want_contexts));
      };

  auto verify_get_context_by_type_and_name =
      [this](string type_name, std::optional<string> version,
             string context_name, Context want_context) {
        GetContextByTypeAndNameRequest get_node_request;
        get_node_request.set_type_name(type_name);
        if (version) {
          get_node_request.set_type_version(*version);
        }
        get_node_request.set_context_name(context_name);
        GetContextByTypeAndNameResponse get_node_response;
        ASSERT_EQ(absl::OkStatus(), metadata_store_->GetContextByTypeAndName(
                                        get_node_request, &get_node_response));
        EXPECT_THAT(get_node_response.context(),
                    EqualsProto<Context>(want_context, /*ignore_fields=*/{
                                             "type", "create_time_since_epoch",
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
  ArtifactType type;
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
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutArtifactType(put_artifact_type_request,
                                               &put_artifact_type_response));
    ASSERT_TRUE(put_artifact_type_response.has_type_id());

    type = put_artifact_type_request.artifact_type();
    type.set_id(put_artifact_type_response.type_id());
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
      absl::StrFormat(kArtifactTemplate, type.id(), "1"));
  Artifact artifact2 = ParseTextProtoOrDie<Artifact>(
      absl::StrFormat(kArtifactTemplate, type.id(), "2"));

  {
    PutArtifactsRequest put_artifacts_request;
    *put_artifacts_request.mutable_artifacts()->Add() = artifact1;
    *put_artifacts_request.mutable_artifacts()->Add() = artifact2;
    PutArtifactsResponse put_artifacts_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutArtifacts(put_artifacts_request,
                                            &put_artifacts_response));
    ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(2));
    artifact1.set_id(put_artifacts_response.artifact_ids(0));
    artifact2.set_id(put_artifacts_response.artifact_ids(1));
    artifact1.set_type("test_type2");
    artifact2.set_type("test_type2");
  }

  // Test: retrieve by one id
  {
    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(artifact1.id());
    GetArtifactsByIDResponse get_artifacts_by_id_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                                &get_artifacts_by_id_response));
    ASSERT_THAT(get_artifacts_by_id_response.artifacts(),
                ElementsAre(EqualsProto(
                    artifact1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
    ASSERT_THAT(get_artifacts_by_id_response.artifact_types(), IsEmpty());
  }
  // Test: retrieve by one id
  const int64_t unknown_id = artifact1.id() + artifact2.id() + 1;
  {
    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(unknown_id);
    GetArtifactsByIDResponse get_artifacts_by_id_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                                &get_artifacts_by_id_response));
    ASSERT_THAT(get_artifacts_by_id_response.artifacts(), IsEmpty());
    ASSERT_THAT(get_artifacts_by_id_response.artifact_types(), IsEmpty());
  }
  // Test: retrieve by multiple ids
  {
    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(unknown_id);
    get_artifacts_by_id_request.add_artifact_ids(artifact1.id());
    get_artifacts_by_id_request.add_artifact_ids(artifact2.id());
    GetArtifactsByIDResponse get_artifacts_by_id_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                                &get_artifacts_by_id_response));
    ASSERT_THAT(
        get_artifacts_by_id_response.artifacts(),
        UnorderedElementsAre(
            EqualsProto(artifact1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(artifact2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
    ASSERT_THAT(get_artifacts_by_id_response.artifact_types(), IsEmpty());
  }
}

TEST_P(MetadataStoreTestSuite, PutArtifactsGetArtifactsByIDAndPopulateType) {
  std::vector<ArtifactType> types;
  // Create the types
  {
    for (int64_t i = 0; i < 2; i++) {
      PutArtifactTypeRequest put_artifact_type_request =
          ParseTextProtoOrDie<PutArtifactTypeRequest>(
              absl::Substitute(R"(
              all_fields_match: true
              artifact_type: {
                name: 'test_type_$0'
                version: 'v1'
                description: 'artifact_type_description'
                external_id: 'test_type_$1'
                properties { key: 'property_$2' value: STRING }
              }
            )",
                               i + 1, i + 1, i + 1));
      PutArtifactTypeResponse put_artifact_type_response;
      ASSERT_EQ(absl::OkStatus(),
                metadata_store_->PutArtifactType(put_artifact_type_request,
                                                 &put_artifact_type_response));
      ASSERT_TRUE(put_artifact_type_response.has_type_id());

      types.push_back(put_artifact_type_request.artifact_type());
      types[i].set_id(put_artifact_type_response.type_id());
    }
    // Create an ArtifactType without properties.
    {
      PutArtifactTypeRequest put_artifact_type_request =
          ParseTextProtoOrDie<PutArtifactTypeRequest>(absl::Substitute(R"(
              all_fields_match: true
              artifact_type: {
                name: 'test_type_$0'
              }
            )",
                                                                       3));
      PutArtifactTypeResponse put_artifact_type_response;
      ASSERT_EQ(absl::OkStatus(),
                metadata_store_->PutArtifactType(put_artifact_type_request,
                                                 &put_artifact_type_response));
      ASSERT_TRUE(put_artifact_type_response.has_type_id());

      types.push_back(put_artifact_type_request.artifact_type());
      types.back().set_id(put_artifact_type_response.type_id());
    }
  }

  // Test: Put in two artifacts with the same type, retrieve Artifacts and
  // populate ArtifactTypes.
  {
    // Put in two artifacts.
    constexpr absl::string_view kArtifactTemplate = R"(
          type_id: %d
          uri: 'testuri://testing/uri'
          properties {
            key: 'property_%d'
            value: { string_value: '%s' }
          }
      )";
    Artifact artifact1 = ParseTextProtoOrDie<Artifact>(
        absl::StrFormat(kArtifactTemplate, types[0].id(), 1, "1"));
    Artifact artifact2 = ParseTextProtoOrDie<Artifact>(
        absl::StrFormat(kArtifactTemplate, types[0].id(), 1, "2"));
    PutArtifactsRequest put_artifacts_request;
    *put_artifacts_request.mutable_artifacts()->Add() = artifact1;
    *put_artifacts_request.mutable_artifacts()->Add() = artifact2;
    PutArtifactsResponse put_artifacts_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutArtifacts(put_artifacts_request,
                                            &put_artifacts_response));
    ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(2));
    artifact1.set_id(put_artifacts_response.artifact_ids(0));
    artifact2.set_id(put_artifacts_response.artifact_ids(1));
    artifact1.set_type(types[0].name());
    artifact2.set_type(types[0].name());

    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(artifact1.id());
    get_artifacts_by_id_request.add_artifact_ids(artifact2.id());
    get_artifacts_by_id_request.set_populate_artifact_types(true);
    GetArtifactsByIDResponse get_artifacts_by_id_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                                &get_artifacts_by_id_response));
    ASSERT_THAT(
        get_artifacts_by_id_response.artifacts(),
        UnorderedElementsAre(
            EqualsProto(artifact1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(artifact2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
    ASSERT_THAT(
        get_artifacts_by_id_response.artifact_types(),
        UnorderedElementsAre(EqualsProto(types[0])));
  }

  // Test: Put in three artifacts with different types, retrieve Artifacts and
  // populate ArtifactTypes.
  {
    // Put in two artifacts with properties.
    constexpr absl::string_view kArtifactTemplate = R"(
          type_id: %d
          uri: 'testuri://testing/uri'
          properties {
            key: 'property_%d'
            value: { string_value: '%s' }
          }
      )";
    Artifact artifact1 = ParseTextProtoOrDie<Artifact>(
        absl::StrFormat(kArtifactTemplate, types[0].id(), 1, "1"));
    Artifact artifact2 = ParseTextProtoOrDie<Artifact>(
        absl::StrFormat(kArtifactTemplate, types[1].id(), 2, "2"));
    // Put in an artifact without property.
    Artifact artifact3 = ParseTextProtoOrDie<Artifact>(absl::StrFormat(
        R"(
          type_id: %d
          uri: 'testuri://testing/uri'
      )",
        types[2].id()));
    PutArtifactsRequest put_artifacts_request;
    *put_artifacts_request.mutable_artifacts()->Add() = artifact1;
    *put_artifacts_request.mutable_artifacts()->Add() = artifact2;
    *put_artifacts_request.mutable_artifacts()->Add() = artifact3;
    PutArtifactsResponse put_artifacts_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutArtifacts(put_artifacts_request,
                                            &put_artifacts_response));
    ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(3));
    artifact1.set_id(put_artifacts_response.artifact_ids(0));
    artifact2.set_id(put_artifacts_response.artifact_ids(1));
    artifact3.set_id(put_artifacts_response.artifact_ids(2));
    artifact1.set_type(types[0].name());
    artifact2.set_type(types[1].name());
    artifact3.set_type(types[2].name());

    GetArtifactsByIDRequest get_artifacts_by_id_request;
    const int64_t kIrrelevantArtifactId = 4;
    get_artifacts_by_id_request.add_artifact_ids(kIrrelevantArtifactId);
    get_artifacts_by_id_request.add_artifact_ids(artifact1.id());
    get_artifacts_by_id_request.add_artifact_ids(artifact2.id());
    get_artifacts_by_id_request.add_artifact_ids(artifact3.id());
    get_artifacts_by_id_request.set_populate_artifact_types(true);
    GetArtifactsByIDResponse get_artifacts_by_id_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                                &get_artifacts_by_id_response));

    ASSERT_THAT(
        get_artifacts_by_id_response.artifacts(),
        UnorderedElementsAre(
            EqualsProto(artifact1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(artifact2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(artifact3,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
    ASSERT_THAT(
        get_artifacts_by_id_response.artifact_types(),
        UnorderedElementsAre(EqualsProto(types[0]), EqualsProto(types[1]),
                             EqualsProto(types[2])));
  }
}

TEST_P(MetadataStoreTestSuite, PutArtifactsGetArtifactsByExternalIds) {
  int64_t type_id;
  // Create the type
  {
    const PutArtifactTypeRequest put_artifact_type_request =
        ParseTextProtoOrDie<PutArtifactTypeRequest>(
            R"pb(
              all_fields_match: true
              artifact_type: {
                name: 'test_type2'
                properties { key: 'property' value: STRING }
              }
            )pb");
    PutArtifactTypeResponse put_artifact_type_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutArtifactType(put_artifact_type_request,
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
          external_id: '%s'
      )";
  Artifact artifact1 = ParseTextProtoOrDie<Artifact>(
      absl::StrFormat(kArtifactTemplate, type_id, "1", "artifact_1"));
  Artifact artifact2 = ParseTextProtoOrDie<Artifact>(
      absl::StrFormat(kArtifactTemplate, type_id, "2", "artifact_2"));

  {
    PutArtifactsRequest put_artifacts_request;
    *put_artifacts_request.mutable_artifacts()->Add() = artifact1;
    *put_artifacts_request.mutable_artifacts()->Add() = artifact2;
    PutArtifactsResponse put_artifacts_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutArtifacts(put_artifacts_request,
                                            &put_artifacts_response));
    ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(2));
    artifact1.set_id(put_artifacts_response.artifact_ids(0));
    artifact2.set_id(put_artifacts_response.artifact_ids(1));
    artifact1.set_type("test_type2");
    artifact2.set_type("test_type2");
  }

  // Test: retrieve by one external id
  {
    GetArtifactsByExternalIdsRequest get_artifacts_by_external_ids_request;
    get_artifacts_by_external_ids_request.add_external_ids(
        artifact1.external_id());
    GetArtifactsByExternalIdsResponse get_artifacts_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByExternalIds(
                                    get_artifacts_by_external_ids_request,
                                    &get_artifacts_by_external_ids_response));
    EXPECT_THAT(get_artifacts_by_external_ids_response.artifacts(),
                ElementsAre(EqualsProto(
                    artifact1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  // Test: retrieve by one non-existing external id
  {
    GetArtifactsByExternalIdsRequest get_artifacts_by_external_ids_request;
    get_artifacts_by_external_ids_request.add_external_ids("artifact_absent");
    GetArtifactsByExternalIdsResponse get_artifacts_by_external_ids_response;
    EXPECT_TRUE(absl::IsNotFound(metadata_store_->GetArtifactsByExternalIds(
        get_artifacts_by_external_ids_request,
        &get_artifacts_by_external_ids_response)));
  }
  // Test: retrieve by multiple external ids
  {
    GetArtifactsByExternalIdsRequest get_artifacts_by_external_ids_request;

    // Can retrieve Artifacts by multiple external ids
    get_artifacts_by_external_ids_request.add_external_ids(
        artifact1.external_id());
    get_artifacts_by_external_ids_request.add_external_ids(
        artifact2.external_id());
    GetArtifactsByExternalIdsResponse get_artifacts_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByExternalIds(
                                    get_artifacts_by_external_ids_request,
                                    &get_artifacts_by_external_ids_response));
    EXPECT_THAT(
        get_artifacts_by_external_ids_response.artifacts(),
        UnorderedElementsAre(
            EqualsProto(artifact1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(artifact2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));

    // Will return whatever found if some of the external ids is absent
    get_artifacts_by_external_ids_request.add_external_ids("artifact_absent");
    EXPECT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByExternalIds(
                                    get_artifacts_by_external_ids_request,
                                    &get_artifacts_by_external_ids_response));
    EXPECT_THAT(
        get_artifacts_by_external_ids_response.artifacts(),
        UnorderedElementsAre(
            EqualsProto(artifact1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(artifact2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }

  // Test retrieve by empty external id
  {
    GetArtifactsByExternalIdsRequest get_artifacts_by_external_ids_request;
    get_artifacts_by_external_ids_request.add_external_ids("");
    GetArtifactsByExternalIdsResponse get_artifacts_by_external_ids_response;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_store_->GetArtifactsByExternalIds(
            get_artifacts_by_external_ids_request,
            &get_artifacts_by_external_ids_response)));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_artifact_type_request,
                                             &put_artifact_type_response));
  ASSERT_TRUE(put_artifact_type_response.has_type_id());

  const int64_t type_id = put_artifact_type_response.type_id();

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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request,
                                          &put_artifacts_response));
  ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(1));
  const int64_t artifact_id = put_artifacts_response.artifact_ids(0);

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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request_2,
                                          &put_artifacts_response_2));

  GetArtifactsByIDRequest get_artifacts_by_id_request;
  get_artifacts_by_id_request.add_artifact_ids(artifact_id);
  GetArtifactsByIDResponse get_artifacts_by_id_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                              &get_artifacts_by_id_response));
  ASSERT_THAT(get_artifacts_by_id_response.artifacts(), SizeIs(1));
  EXPECT_THAT(get_artifacts_by_id_response.artifacts(0),
              EqualsProto(put_artifacts_request_2.artifacts(0),
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
}

// Test creating an artifact and then updating one of its properties.
TEST_P(MetadataStoreTestSuite, UpdateArtifactWithMasking) {
  const PutArtifactTypeRequest put_artifact_type_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"pb(
            all_fields_match: true
            artifact_type: {
              name: 'test_type2'
              properties { key: 'property' value: STRING }
            }
          )pb");
  PutArtifactTypeResponse put_artifact_type_response;
  ASSERT_EQ(metadata_store_->PutArtifactType(put_artifact_type_request,
                                             &put_artifact_type_response),
            absl::OkStatus());
  ASSERT_TRUE(put_artifact_type_response.has_type_id());

  const int64_t type_id = put_artifact_type_response.type_id();

  // Add two artifacts, one with a `properties` pair <'property': '3'>, one
  // without.
  PutArtifactsRequest put_artifacts_request =
      ParseTextProtoOrDie<PutArtifactsRequest>(R"pb(
        artifacts: {
          uri: 'testuri://testing/uri1'
          properties {
            key: 'property'
            value: { string_value: '3' }
          }
        }
        artifacts: { uri: 'testuri://testing/uri2' }
      )pb");
  put_artifacts_request.mutable_artifacts(0)->set_type_id(type_id);
  put_artifacts_request.mutable_artifacts(1)->set_type_id(type_id);
  PutArtifactsResponse put_artifacts_response;
  {
    // Test 1: a complex test case for updating fields and properties for both
    // artifacts.
    ASSERT_EQ(metadata_store_->PutArtifacts(put_artifacts_request,
                                            &put_artifacts_response),
              absl::OkStatus());
    ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(2));
    const int64_t artifact_id1 = put_artifacts_response.artifact_ids(0);
    const int64_t artifact_id2 = put_artifacts_response.artifact_ids(1);
    // Add `state` for both artifacts. `uri` for both artifacts will remain
    // unchanged.
    // Change string value of key `property` from '3' to '1' in the first
    // artifact.
    // Add `properties` pair <'property': '2'> in the second artifact.
    PutArtifactsRequest update_artifacts_request =
        ParseTextProtoOrDie<PutArtifactsRequest>(R"pb(
          artifacts: {
            properties {
              key: 'property'
              value: { string_value: '1' }
            }
            state: LIVE
          }
          artifacts: {
            properties {
              key: 'property'
              value: { string_value: '2' }
            }
            state: LIVE
          }
          update_mask: {
            paths: 'properties.property'
            paths: 'state'
            paths: 'an_invalid_field_path_having_no_effect'
          }
        )pb");
    update_artifacts_request.mutable_artifacts(0)->set_type_id(type_id);
    update_artifacts_request.mutable_artifacts(0)->set_id(artifact_id1);
    update_artifacts_request.mutable_artifacts(1)->set_type_id(type_id);
    update_artifacts_request.mutable_artifacts(1)->set_id(artifact_id2);
    PutArtifactsResponse update_artifacts_response;
    ASSERT_EQ(metadata_store_->PutArtifacts(update_artifacts_request,
                                            &update_artifacts_response),
              absl::OkStatus());

    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(artifact_id1);
    get_artifacts_by_id_request.add_artifact_ids(artifact_id2);

    GetArtifactsByIDResponse get_artifacts_by_id_response;
    ASSERT_EQ(metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                                &get_artifacts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_artifacts_by_id_response.artifacts(), SizeIs(2));

    update_artifacts_request.mutable_artifacts(0)->set_uri(
        "testuri://testing/uri1");
    update_artifacts_request.mutable_artifacts(1)->set_uri(
        "testuri://testing/uri2");
    EXPECT_THAT(
        get_artifacts_by_id_response.artifacts(),
        UnorderedElementsAre(
            EqualsProto(update_artifacts_request.artifacts(0),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(update_artifacts_request.artifacts(1),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  {
    // Test 2: insert two new artifacts and update fields for both artifacts.
    ASSERT_EQ(metadata_store_->PutArtifacts(put_artifacts_request,
                                            &put_artifacts_response),
              absl::OkStatus());
    ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(2));
    const int64_t artifact_id3 = put_artifacts_response.artifact_ids(0);
    const int64_t artifact_id4 = put_artifacts_response.artifact_ids(1);
    // Set `external_id` and `uri` for both artifacts.
    // `properties` for both artifacts will remain unchanged.
    PutArtifactsRequest update_artifacts_request =
        ParseTextProtoOrDie<PutArtifactsRequest>(R"pb(
          artifacts: { external_id: 'artifact_3' uri: 'testuri://testing/uri3' }
          artifacts: { external_id: 'artifact_4' uri: 'testuri://testing/uri4' }
          update_mask: { paths: 'external_id' paths: 'uri' }
        )pb");
    update_artifacts_request.mutable_artifacts(0)->set_type_id(type_id);
    update_artifacts_request.mutable_artifacts(0)->set_id(artifact_id3);
    update_artifacts_request.mutable_artifacts(1)->set_type_id(type_id);
    update_artifacts_request.mutable_artifacts(1)->set_id(artifact_id4);
    PutArtifactsResponse update_artifacts_response;
    ASSERT_EQ(metadata_store_->PutArtifacts(update_artifacts_request,
                                            &update_artifacts_response),
              absl::OkStatus());

    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(artifact_id3);
    get_artifacts_by_id_request.add_artifact_ids(artifact_id4);

    GetArtifactsByIDResponse get_artifacts_by_id_response;
    ASSERT_EQ(metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                                &get_artifacts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_artifacts_by_id_response.artifacts(), SizeIs(2));

    EXPECT_THAT(
        get_artifacts_by_id_response.artifacts(),
        UnorderedElementsAre(
            EqualsProto(update_artifacts_request.artifacts(0),
                        /*ignore_fields=*/{"type", "properties",
                                           "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(update_artifacts_request.artifacts(1),
                        /*ignore_fields=*/{"type", "properties",
                                           "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  Artifact artifact_for_test_3_4_and_5;
  {
    // Test 3: insert two new artifacts and update `properties` and
    // `custom_properties` for both artifacts.
    ASSERT_EQ(metadata_store_->PutArtifacts(put_artifacts_request,
                                            &put_artifacts_response),
              absl::OkStatus());
    ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(2));
    const int64_t artifact_id5 = put_artifacts_response.artifact_ids(0);
    const int64_t artifact_id6 = put_artifacts_response.artifact_ids(1);
    // `uri` for both artifacts will remain unchanged.
    // Delete `properties` pair <'property': '3'> in the first artifact.
    // Add `custom_properties` pair <'custom_property': true> for `artifact_5`.
    // Add `custom_properties` pair <'custom_property': false> for `artifact_6`.
    PutArtifactsRequest update_artifacts_request =
        ParseTextProtoOrDie<PutArtifactsRequest>(R"pb(
          artifacts: {
            custom_properties {
              key: 'custom_property'
              value: { bool_value: true }
            }
          }
          artifacts: {
            custom_properties {
              key: 'custom_property'
              value: { bool_value: false }
            }
          }
          update_mask: {
            paths: 'properties.property'
            paths: 'custom_properties.custom_property'
          }
        )pb");
    update_artifacts_request.mutable_artifacts(0)->set_type_id(type_id);
    update_artifacts_request.mutable_artifacts(0)->set_id(artifact_id5);
    update_artifacts_request.mutable_artifacts(1)->set_type_id(type_id);
    update_artifacts_request.mutable_artifacts(1)->set_id(artifact_id6);
    PutArtifactsResponse update_artifacts_response;
    ASSERT_EQ(metadata_store_->PutArtifacts(update_artifacts_request,
                                            &update_artifacts_response),
              absl::OkStatus());

    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(artifact_id5);
    get_artifacts_by_id_request.add_artifact_ids(artifact_id6);

    GetArtifactsByIDResponse get_artifacts_by_id_response;
    ASSERT_EQ(metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                                &get_artifacts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_artifacts_by_id_response.artifacts(), SizeIs(2));

    update_artifacts_request.mutable_artifacts(0)->set_uri(
        "testuri://testing/uri1");
    update_artifacts_request.mutable_artifacts(1)->set_uri(
        "testuri://testing/uri2");
    artifact_for_test_3_4_and_5 = update_artifacts_request.artifacts(1);
    EXPECT_THAT(
        get_artifacts_by_id_response.artifacts(),
        UnorderedElementsAre(
            EqualsProto(update_artifacts_request.artifacts(0),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(update_artifacts_request.artifacts(1),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  {
    // Test 4: insert a new artifact and update an existing artifact in the
    // same request under masking. The mask is expected to have no effect on
    // insertion but to protect fields for update.
    PutArtifactsRequest upsert_artifacts_request =
        ParseTextProtoOrDie<PutArtifactsRequest>(R"pb(
          artifacts: {
            external_id: 'artifact_6'
            custom_properties {
              key: 'custom_property'
              value: { bool_value: true }
            }
          }
          artifacts: {
            uri: 'testuri://testing/uri7'
            external_id: 'artifact_7'
            custom_properties {
              key: 'custom_property'
              value: { bool_value: true }
            }
          }
          update_mask: { paths: 'external_id' }
        )pb");
    upsert_artifacts_request.mutable_artifacts(0)->set_type_id(type_id);
    upsert_artifacts_request.mutable_artifacts(0)->set_id(
        artifact_for_test_3_4_and_5.id());
    upsert_artifacts_request.mutable_artifacts(1)->set_type_id(type_id);
    PutArtifactsResponse upsert_artifacts_response;
    ASSERT_EQ(metadata_store_->PutArtifacts(upsert_artifacts_request,
                                            &upsert_artifacts_response),
              absl::OkStatus());
    const int64_t artifact_id7 = upsert_artifacts_response.artifact_ids(1);

    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(
        artifact_for_test_3_4_and_5.id());
    get_artifacts_by_id_request.add_artifact_ids(artifact_id7);

    GetArtifactsByIDResponse get_artifacts_by_id_response;
    ASSERT_EQ(metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                                &get_artifacts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_artifacts_by_id_response.artifacts(), SizeIs(2));

    // If put update is successful, one of the obtained artifacts should be the
    // updated artifact, one of the obtained artifacts should be the inserted
    // artifact.
    artifact_for_test_3_4_and_5.set_external_id("artifact_6");
    upsert_artifacts_request.mutable_artifacts(1)->set_id(artifact_id7);
    EXPECT_THAT(
        get_artifacts_by_id_response.artifacts(),
        UnorderedElementsAre(
            EqualsProto(artifact_for_test_3_4_and_5,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(upsert_artifacts_request.artifacts(1),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  {
    // Test 5: updating an artifact with a mask containing only invalid mask
    // paths has no effect.
    PutArtifactsRequest upsert_artifacts_request =
        ParseTextProtoOrDie<PutArtifactsRequest>(R"pb(
          artifacts: {
            external_id: 'unimportant_exeternal_id_value'
            custom_properties {
              key: 'unimportant_property_key'
              value: { bool_value: true }
            }
          }
          update_mask: { paths: 'an_invalid_field_path' }
        )pb");
    upsert_artifacts_request.mutable_artifacts(0)->set_type_id(type_id);
    upsert_artifacts_request.mutable_artifacts(0)->set_id(
        artifact_for_test_3_4_and_5.id());
    PutArtifactsResponse upsert_artifacts_response;
    ASSERT_EQ(metadata_store_->PutArtifacts(upsert_artifacts_request,
                                            &upsert_artifacts_response),
              absl::OkStatus());

    GetArtifactsByIDRequest get_artifacts_by_id_request;
    get_artifacts_by_id_request.add_artifact_ids(
        artifact_for_test_3_4_and_5.id());

    GetArtifactsByIDResponse get_artifacts_by_id_response;
    ASSERT_EQ(metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                                &get_artifacts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_artifacts_by_id_response.artifacts(), SizeIs(1));

    EXPECT_THAT(
        get_artifacts_by_id_response.artifacts(0),
        EqualsProto(
            artifact_for_test_3_4_and_5,
            /*ignore_fields=*/{"type", "external_id", "create_time_since_epoch",
                               "last_update_time_since_epoch"}));
  }
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_artifact_type_request,
                                             &put_artifact_type_response));
  ASSERT_TRUE(put_artifact_type_response.has_type_id());

  const int64_t type_id = put_artifact_type_response.type_id();

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

  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request,
                                          &put_artifacts_response));
  ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(2));
  const int64_t first_artifact_id = put_artifacts_response.artifact_ids(0);
  const int64_t second_artifact_id = put_artifacts_response.artifact_ids(1);

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  GetArtifactsRequest get_artifacts_request;
  *get_artifacts_request.mutable_options() = list_options;

  GetArtifactsResponse get_artifacts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifacts(get_artifacts_request,
                                          &get_artifacts_response));
  EXPECT_THAT(get_artifacts_response.artifacts(), SizeIs(1));
  EXPECT_THAT(get_artifacts_response.next_page_token(), Not(IsEmpty()));
  EXPECT_EQ(get_artifacts_response.artifacts(0).id(), second_artifact_id);

  EXPECT_THAT(
      get_artifacts_response.artifacts(0),
      EqualsProto(put_artifacts_request.artifacts(1),
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));

  list_options.set_next_page_token(get_artifacts_response.next_page_token());
  *get_artifacts_request.mutable_options() = list_options;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifacts(get_artifacts_request,
                                          &get_artifacts_response));
  EXPECT_THAT(get_artifacts_response.artifacts(), SizeIs(1));
  EXPECT_THAT(get_artifacts_response.next_page_token(), IsEmpty());
  EXPECT_EQ(get_artifacts_response.artifacts(0).id(), first_artifact_id);
  EXPECT_THAT(
      get_artifacts_response.artifacts(0),
      EqualsProto(put_artifacts_request.artifacts(0),
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));
}

TEST_P(MetadataStoreTestSuite, PutArtifactsWhenLatestUpdatedTimeChanged) {
  PutArtifactTypeRequest put_type_request;
  put_type_request.mutable_artifact_type()->set_name("test_type");
  PutArtifactTypeResponse put_type_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->PutArtifactType(
                                  put_type_request, &put_type_response));
  const int64_t type_id = put_type_response.type_id();
  PutArtifactsRequest put_artifacts_request;
  put_artifacts_request.add_artifacts()->set_type_id(type_id);
  PutArtifactsResponse put_artifacts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request,
                                          &put_artifacts_response));

  // Reads the stored artifact, and prepares update.
  GetArtifactsResponse get_artifacts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifacts({}, &get_artifacts_response));
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
  EXPECT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(update_artifact_request,
                                          &update_artifact_response));

  // If update it again with the old `latest_updated_time`, the call fails
  // with FailedPrecondition error.
  absl::Status status = metadata_store_->PutArtifacts(
      update_artifact_request, &update_artifact_response);
  EXPECT_TRUE(absl::IsFailedPrecondition(status));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_execution_type_request,
                                              &put_execution_type_response));
  ASSERT_TRUE(put_execution_type_response.has_type_id());

  const int64_t type_id = put_execution_type_response.type_id();

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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutions(put_executions_request,
                                           &put_executions_response));
  ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(1));
  const int64_t execution_id = put_executions_response.execution_ids(0);

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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutions(put_executions_request_2,
                                           &put_executions_response_2));

  GetExecutionsByIDRequest get_executions_by_id_request;
  get_executions_by_id_request.add_execution_ids(execution_id);
  GetExecutionsByIDResponse get_executions_by_id_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutionsByID(get_executions_by_id_request,
                                               &get_executions_by_id_response));

  EXPECT_THAT(get_executions_by_id_response.executions(0),
              EqualsProto(put_executions_request_2.executions(0),
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
}

TEST_P(MetadataStoreTestSuite, PutExecutionTypeGetExecutionType) {
  const PutExecutionTypeRequest put_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"pb(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property_1' value: STRING }
              base_type: TRAIN
            }
          )pb");
  PutExecutionTypeResponse put_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  const GetExecutionTypeRequest get_request =
      ParseTextProtoOrDie<GetExecutionTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetExecutionTypeResponse get_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutionType(get_request, &get_response));
  ExecutionType expected = put_request.execution_type();
  expected.set_id(put_response.type_id());
  EXPECT_THAT(get_response.execution_type(), EqualsProto(expected));
}

TEST_P(MetadataStoreTestSuite, PutExecutionTypeInsertTypeLink) {
  absl ::string_view type_definition =
      R"( name: 'test_type2'
              properties { key: 'property_1' value: STRING }
              base_type: TRAIN)";
  ExecutionType expected =
      ParseTextProtoOrDie<ExecutionType>(std::string(type_definition));
  InsertTypeAndSetTypeID(metadata_store_, expected);

  const GetExecutionTypeRequest get_request =
      ParseTextProtoOrDie<GetExecutionTypeRequest>(
          R"pb(
            type_name: 'test_type2'
          )pb");
  GetExecutionTypeResponse get_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutionType(get_request, &get_response));
  EXPECT_THAT(get_response.execution_type(), EqualsProto(expected));

  PutExecutionTypeRequest type_request;
  *type_request.mutable_execution_type() = expected;
  {
    PutExecutionTypeRequest update_base_type_request = type_request;
    update_base_type_request.mutable_execution_type()->set_base_type(
        ExecutionType::EVALUATE);
    PutExecutionTypeResponse update_base_type_response;
    ASSERT_EQ(absl::UnimplementedError("base_type update is not supported yet"),
              metadata_store_->PutExecutionType(update_base_type_request,
                                                &update_base_type_response));
  }
  {
    PutExecutionTypeRequest null_base_type_request = type_request;
    null_base_type_request.mutable_execution_type()->clear_base_type();
    PutExecutionTypeResponse null_base_type_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutExecutionType(null_base_type_request,
                                                &null_base_type_response));
  }
  {
    PutExecutionTypeRequest delete_base_type_request = type_request;
    delete_base_type_request.mutable_execution_type()->set_base_type(
        ExecutionType::UNSET);
    PutExecutionTypeResponse delete_base_type_response;
    ASSERT_EQ(
        absl::UnimplementedError("base_type deletion is not supported yet"),
        metadata_store_->PutExecutionType(delete_base_type_request,
                                          &delete_base_type_response));
  }
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_request_1, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ExecutionType type_1 = ParseTextProtoOrDie<ExecutionType>(
      R"(
        name: 'test_type_1'
        properties { key: 'property_1' value: STRING }
      )");
  type_1.set_id(put_response.type_id());

  const PutExecutionTypeRequest put_request_2 =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"pb(
            all_fields_match: true
            execution_type: {
              name: 'test_type_2'
              properties { key: 'property_2' value: INT }
              base_type: PROCESS
            }
          )pb");
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_request_2, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ExecutionType type_2 = ParseTextProtoOrDie<ExecutionType>(
      R"pb(
        name: 'test_type_2'
        properties { key: 'property_2' value: INT }
        base_type: PROCESS
      )pb");
  type_2.set_id(put_response.type_id());

  GetExecutionTypesRequest get_request;
  GetExecutionTypesResponse got_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutionTypes(get_request, &got_response));

  EXPECT_THAT(got_response.execution_types(),
              UnorderedElementsAre(EqualsProto(type_1), EqualsProto(type_2)));
}

TEST_P(MetadataStoreTestSuite, GetExecutionTypesWhenNoneExist) {
  GetExecutionTypesRequest get_request;
  GetExecutionTypesResponse got_response;

  // Expect OK status and empty response.
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutionTypes(get_request, &got_response));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(request_1, &response_1));

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
  absl::Status status =
      metadata_store_->PutExecutionType(request_2, &response_2);
  EXPECT_TRUE(absl::IsAlreadyExists(status)) << status.ToString();
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(request_1, &response_1));

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
  EXPECT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(request_2, &response_2));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(request_1, &response_1));

  const PutExecutionTypeRequest request_2 = request_1;
  PutExecutionTypeResponse response_2;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(request_2, &response_2));
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
  EXPECT_TRUE(absl::IsNotFound(
      metadata_store_->GetExecutionType(get_request, &get_response)));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_execution_type_request,
                                              &put_execution_type_response));
  ASSERT_TRUE(put_execution_type_response.has_type_id());

  const int64_t type_id = put_execution_type_response.type_id();

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

    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutExecutions(put_executions_request,
                                             &put_executions_response));
    ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(2));
    execution1 = put_executions_request.executions(0);
    execution1.set_id(put_executions_response.execution_ids(0));
    execution2 = put_executions_request.executions(1);
    execution2.set_id(put_executions_response.execution_ids(1));
    execution1.set_type("test_type2");
    execution2.set_type("test_type2");
  }
  // Test: Get by a single id
  {
    GetExecutionsByIDRequest get_executions_by_id_request;
    get_executions_by_id_request.add_execution_ids(execution1.id());
    GetExecutionsByIDResponse get_executions_by_id_response;
    ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByID(
                                    get_executions_by_id_request,
                                    &get_executions_by_id_response));
    EXPECT_THAT(get_executions_by_id_response.executions(),
                ElementsAre(EqualsProto(
                    execution1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  // Test: Get by a single unknown id
  const int64_t unknown_id = execution1.id() + execution2.id() + 1;
  {
    GetExecutionsByIDRequest get_executions_by_id_request;
    get_executions_by_id_request.add_execution_ids(unknown_id);
    GetExecutionsByIDResponse get_executions_by_id_response;
    ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByID(
                                    get_executions_by_id_request,
                                    &get_executions_by_id_response));
    EXPECT_THAT(get_executions_by_id_response.executions(), IsEmpty());
  }
  // Test: Get by a several ids
  {
    GetExecutionsByIDRequest get_executions_by_id_request;
    get_executions_by_id_request.add_execution_ids(execution1.id());
    get_executions_by_id_request.add_execution_ids(execution2.id());
    get_executions_by_id_request.add_execution_ids(unknown_id);
    GetExecutionsByIDResponse get_executions_by_id_response;
    ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByID(
                                    get_executions_by_id_request,
                                    &get_executions_by_id_response));
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

TEST_P(MetadataStoreTestSuite, PutExecutionsGetExecutionsByExternalIds) {
  int64_t type_id;
  // Create the type
  {
    const PutExecutionTypeRequest put_execution_type_request =
        ParseTextProtoOrDie<PutExecutionTypeRequest>(
            R"pb(
              all_fields_match: true
              execution_type: {
                name: 'test_execution_type'
                properties { key: 'property' value: STRING }
              }
            )pb");
    PutExecutionTypeResponse put_execution_type_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutExecutionType(put_execution_type_request,
                                                &put_execution_type_response));
    ASSERT_TRUE(put_execution_type_response.has_type_id());

    type_id = put_execution_type_response.type_id();
  }

  // Put in two executions
  constexpr absl::string_view kExecutionTemplate = R"(
          type_id: %d
          properties {
            key: 'property'
            value: { string_value: '%s' }
          }
          external_id: '%s'
      )";
  Execution execution1 = ParseTextProtoOrDie<Execution>(absl::StrFormat(
      kExecutionTemplate, type_id, "1", "execution_external_id_1"));
  Execution execution2 = ParseTextProtoOrDie<Execution>(absl::StrFormat(
      kExecutionTemplate, type_id, "2", "execution_external_id_2"));

  {
    PutExecutionsRequest put_executions_request;
    *put_executions_request.mutable_executions()->Add() = execution1;
    *put_executions_request.mutable_executions()->Add() = execution2;
    PutExecutionsResponse put_executions_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutExecutions(put_executions_request,
                                             &put_executions_response));
    ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(2));
    execution1.set_id(put_executions_response.execution_ids(0));
    execution2.set_id(put_executions_response.execution_ids(1));
    execution1.set_type("test_execution_type");
    execution2.set_type("test_execution_type");
  }

  // Test: retrieve by one external id
  {
    GetExecutionsByExternalIdsRequest get_executions_by_external_ids_request;
    get_executions_by_external_ids_request.add_external_ids(
        execution1.external_id());
    GetExecutionsByExternalIdsResponse get_executions_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByExternalIds(
                                    get_executions_by_external_ids_request,
                                    &get_executions_by_external_ids_response));
    EXPECT_THAT(get_executions_by_external_ids_response.executions(),
                ElementsAre(EqualsProto(
                    execution1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  // Test: retrieve by one non-existing external id
  {
    GetExecutionsByExternalIdsRequest get_executions_by_external_ids_request;
    get_executions_by_external_ids_request.add_external_ids(
        "execution_absent_external_id");
    GetExecutionsByExternalIdsResponse get_executions_by_external_ids_response;
    EXPECT_TRUE(absl::IsNotFound(metadata_store_->GetExecutionsByExternalIds(
        get_executions_by_external_ids_request,
        &get_executions_by_external_ids_response)));
  }
  // Test: retrieve by multiple external ids
  {
    GetExecutionsByExternalIdsRequest get_executions_by_external_ids_request;

    // Can retrieve Executions by multiple external ids
    get_executions_by_external_ids_request.add_external_ids(
        execution1.external_id());
    get_executions_by_external_ids_request.add_external_ids(
        execution2.external_id());
    GetExecutionsByExternalIdsResponse get_executions_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByExternalIds(
                                    get_executions_by_external_ids_request,
                                    &get_executions_by_external_ids_response));
    EXPECT_THAT(
        get_executions_by_external_ids_response.executions(),
        UnorderedElementsAre(
            EqualsProto(execution1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(execution2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));

    // Will return whatever found if some of the external ids is absent
    get_executions_by_external_ids_request.add_external_ids(
        "execution_absent_external_id");
    EXPECT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByExternalIds(
                                    get_executions_by_external_ids_request,
                                    &get_executions_by_external_ids_response));
    EXPECT_THAT(
        get_executions_by_external_ids_response.executions(),
        UnorderedElementsAre(
            EqualsProto(execution1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(execution2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }

  // Test retrieve by empty external id
  {
    GetExecutionsByExternalIdsRequest get_executions_by_external_ids_request;
    get_executions_by_external_ids_request.add_external_ids("");
    GetExecutionsByExternalIdsResponse get_executions_by_external_ids_response;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_store_->GetExecutionsByExternalIds(
            get_executions_by_external_ids_request,
            &get_executions_by_external_ids_response)));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_execution_type_request,
                                              &put_execution_type_response));
  ASSERT_TRUE(put_execution_type_response.has_type_id());

  const int64_t type_id = put_execution_type_response.type_id();

  PutExecutionsRequest put_executions_request =
      ParseTextProtoOrDie<PutExecutionsRequest>(R"(
        executions: {}
      )");
  put_executions_request.mutable_executions(0)->set_type_id(type_id);
  PutExecutionsResponse put_executions_response;

  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutions(put_executions_request,
                                           &put_executions_response));
  ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(1));
  const int64_t execution_id = put_executions_response.execution_ids(0);
  const GetExecutionsRequest get_executions_request;
  GetExecutionsResponse get_executions_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutions(get_executions_request,
                                           &get_executions_response));
  ASSERT_THAT(get_executions_response.executions(), SizeIs(1));
  EXPECT_THAT(
      get_executions_response.executions(0),
      EqualsProto(put_executions_request.executions(0),
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));

  GetExecutionsByTypeRequest get_executions_by_type_request;
  GetExecutionsByTypeResponse get_executions_by_type_response;
  get_executions_by_type_request.set_type_name("test_type2");
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByType(
                                  get_executions_by_type_request,
                                  &get_executions_by_type_response));
  ASSERT_THAT(get_executions_by_type_response.executions(), SizeIs(1));
  EXPECT_EQ(get_executions_by_type_response.executions(0).id(), execution_id);

  GetExecutionsByTypeRequest get_executions_by_not_exist_type_request;
  GetExecutionsByTypeResponse get_executions_by_not_exist_type_response;
  get_executions_by_not_exist_type_request.set_type_name("not_exist_type");
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByType(
                                  get_executions_by_not_exist_type_request,
                                  &get_executions_by_not_exist_type_response));
  EXPECT_THAT(get_executions_by_not_exist_type_response.executions(),
              SizeIs(0));
}

// Test creating an execution and then updating one of its properties.
TEST_P(MetadataStoreTestSuite, UpdateExecutionWithMasking) {
  const PutExecutionTypeRequest put_execution_type_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"pb(
            all_fields_match: true
            execution_type: {
              name: 'test_type2'
              properties { key: 'property' value: STRING }
            }
          )pb");
  PutExecutionTypeResponse put_execution_type_response;
  ASSERT_EQ(metadata_store_->PutExecutionType(put_execution_type_request,
                                              &put_execution_type_response),
            absl::OkStatus());
  ASSERT_TRUE(put_execution_type_response.has_type_id());

  const int64_t type_id = put_execution_type_response.type_id();

  // Add two executions, one with a `properties` pair <'property': '3'>, one
  // without.
  PutExecutionsRequest put_executions_request =
      ParseTextProtoOrDie<PutExecutionsRequest>(R"pb(
        executions: {
          properties {
            key: 'property'
            value: { string_value: '3' }
          }

        }
        executions: {}
      )pb");
  put_executions_request.mutable_executions(0)->set_type_id(type_id);
  put_executions_request.mutable_executions(1)->set_type_id(type_id);
  PutExecutionsResponse put_executions_response;
  {
    // Test 1: a complex test case for updating fields and properties for both
    // executions.
    ASSERT_EQ(metadata_store_->PutExecutions(put_executions_request,
                                             &put_executions_response),
              absl::OkStatus());
    ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(2));
    const int64_t execution_id1 = put_executions_response.execution_ids(0);
    const int64_t execution_id2 = put_executions_response.execution_ids(1);
    // Add `last_known_state` for both executions.
    // Change string value of key `property` from '3' to '1' in the first
    // execution.
    // Add `properties` pair <'property': '2'> in the second execution.
    PutExecutionsRequest update_executions_request =
        ParseTextProtoOrDie<PutExecutionsRequest>(R"pb(
          executions: {
            properties {
              key: 'property'
              value: { string_value: '1' }
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
          update_mask: {
            paths: 'properties.property'
            paths: 'last_known_state'
            paths: 'an_invalid_field_path_having_no_effect'
          }
        )pb");
    update_executions_request.mutable_executions(0)->set_type_id(type_id);
    update_executions_request.mutable_executions(0)->set_id(execution_id1);
    update_executions_request.mutable_executions(1)->set_type_id(type_id);
    update_executions_request.mutable_executions(1)->set_id(execution_id2);
    PutExecutionsResponse update_executions_response;
    ASSERT_EQ(metadata_store_->PutExecutions(update_executions_request,
                                             &update_executions_response),
              absl::OkStatus());

    GetExecutionsByIDRequest get_executions_by_id_request;
    get_executions_by_id_request.add_execution_ids(execution_id1);
    get_executions_by_id_request.add_execution_ids(execution_id2);

    GetExecutionsByIDResponse get_executions_by_id_response;
    ASSERT_EQ(metadata_store_->GetExecutionsByID(
                  get_executions_by_id_request, &get_executions_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_executions_by_id_response.executions(), SizeIs(2));

    EXPECT_THAT(
        get_executions_by_id_response.executions(),
        UnorderedElementsAre(
            EqualsProto(update_executions_request.executions(0),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(update_executions_request.executions(1),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  {
    // Test 2: insert two new executions and update fields for both executions.
    ASSERT_EQ(metadata_store_->PutExecutions(put_executions_request,
                                             &put_executions_response),
              absl::OkStatus());
    ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(2));
    const int64_t execution_id3 = put_executions_response.execution_ids(0);
    const int64_t execution_id4 = put_executions_response.execution_ids(1);
    // Set `external_id` and `last_known_state` for both executions.
    // `properties` for both executions will remain unchanged.
    PutExecutionsRequest update_executions_request =
        ParseTextProtoOrDie<PutExecutionsRequest>(R"pb(
          executions: { external_id: 'execution_3' last_known_state: RUNNING }
          executions: { external_id: 'execution_4' last_known_state: CANCELED }
          update_mask: { paths: 'external_id' paths: 'last_known_state' }
        )pb");
    update_executions_request.mutable_executions(0)->set_type_id(type_id);
    update_executions_request.mutable_executions(0)->set_id(execution_id3);
    update_executions_request.mutable_executions(1)->set_type_id(type_id);
    update_executions_request.mutable_executions(1)->set_id(execution_id4);
    PutExecutionsResponse update_executions_response;
    ASSERT_EQ(metadata_store_->PutExecutions(update_executions_request,
                                             &update_executions_response),
              absl::OkStatus());

    GetExecutionsByIDRequest get_executions_by_id_request;
    get_executions_by_id_request.add_execution_ids(execution_id3);
    get_executions_by_id_request.add_execution_ids(execution_id4);

    GetExecutionsByIDResponse get_executions_by_id_response;
    ASSERT_EQ(metadata_store_->GetExecutionsByID(
                  get_executions_by_id_request, &get_executions_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_executions_by_id_response.executions(), SizeIs(2));

    EXPECT_THAT(
        get_executions_by_id_response.executions(),
        UnorderedElementsAre(
            EqualsProto(update_executions_request.executions(0),
                        /*ignore_fields=*/{"type", "properties",
                                           "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(update_executions_request.executions(1),
                        /*ignore_fields=*/{"type", "properties",
                                           "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  Execution execution_for_test_3_4_and_5;
  {
    // Test 3: insert two new executions and update `properties` and
    // `custom_properties` for both executions.
    ASSERT_EQ(metadata_store_->PutExecutions(put_executions_request,
                                             &put_executions_response),
              absl::OkStatus());
    ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(2));
    const int64_t execution_id5 = put_executions_response.execution_ids(0);
    const int64_t execution_id6 = put_executions_response.execution_ids(1);
    // Delete `properties` pair <'property': '3'> in the first execution.
    // Add `custom_properties` pair <'custom_property': true> for `execution_5`.
    // Add `custom_properties` pair <'custom_property': false> for
    // `execution_6`.
    PutExecutionsRequest update_executions_request =
        ParseTextProtoOrDie<PutExecutionsRequest>(R"pb(
          executions: {
            custom_properties {
              key: 'custom_property'
              value: { bool_value: true }
            }
          }
          executions: {
            custom_properties {
              key: 'custom_property'
              value: { bool_value: false }
            }
          }
          update_mask: {
            paths: 'properties.property'
            paths: 'custom_properties.custom_property'
          }
        )pb");
    update_executions_request.mutable_executions(0)->set_type_id(type_id);
    update_executions_request.mutable_executions(0)->set_id(execution_id5);
    update_executions_request.mutable_executions(1)->set_type_id(type_id);
    update_executions_request.mutable_executions(1)->set_id(execution_id6);
    PutExecutionsResponse update_executions_response;
    ASSERT_EQ(metadata_store_->PutExecutions(update_executions_request,
                                             &update_executions_response),
              absl::OkStatus());
    execution_for_test_3_4_and_5 = update_executions_request.executions(1);

    GetExecutionsByIDRequest get_executions_by_id_request;
    get_executions_by_id_request.add_execution_ids(execution_id5);
    get_executions_by_id_request.add_execution_ids(execution_id6);

    GetExecutionsByIDResponse get_executions_by_id_response;
    ASSERT_EQ(metadata_store_->GetExecutionsByID(
                  get_executions_by_id_request, &get_executions_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_executions_by_id_response.executions(), SizeIs(2));

    EXPECT_THAT(
        get_executions_by_id_response.executions(),
        UnorderedElementsAre(
            EqualsProto(update_executions_request.executions(0),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(update_executions_request.executions(1),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  {
    // Test 4: insert a new execution and update an existing execution in the
    // same request under masking. The mask is expected to have no effect on
    // insertion but to protect fields for update.
    PutExecutionsRequest upsert_executions_request =
        ParseTextProtoOrDie<PutExecutionsRequest>(R"pb(
          executions: {
            external_id: 'execution_6'
            custom_properties {
              key: 'custom_property'
              value: { bool_value: true }
            }
          }
          executions: {
            external_id: 'execution_7'
            custom_properties {
              key: 'custom_property'
              value: { bool_value: true }
            }
          }
          update_mask: { paths: 'external_id' }
        )pb");
    upsert_executions_request.mutable_executions(0)->set_type_id(type_id);
    upsert_executions_request.mutable_executions(0)->set_id(
        execution_for_test_3_4_and_5.id());
    upsert_executions_request.mutable_executions(1)->set_type_id(type_id);
    PutExecutionsResponse upsert_executions_response;
    ASSERT_EQ(metadata_store_->PutExecutions(upsert_executions_request,
                                             &upsert_executions_response),
              absl::OkStatus());
    const int64_t execution_id7 = upsert_executions_response.execution_ids(1);

    GetExecutionsByIDRequest get_executions_by_id_request;
    get_executions_by_id_request.add_execution_ids(
        execution_for_test_3_4_and_5.id());
    get_executions_by_id_request.add_execution_ids(execution_id7);

    GetExecutionsByIDResponse get_executions_by_id_response;
    ASSERT_EQ(metadata_store_->GetExecutionsByID(
                  get_executions_by_id_request, &get_executions_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_executions_by_id_response.executions(), SizeIs(2));

    // If put update is successful, one of the obtained executions should be the
    // updated execution, one of the obtained executions should be the inserted
    // execution.
    execution_for_test_3_4_and_5.set_external_id("execution_6");
    upsert_executions_request.mutable_executions(1)->set_id(execution_id7);
    EXPECT_THAT(
        get_executions_by_id_response.executions(),
        UnorderedElementsAre(
            EqualsProto(execution_for_test_3_4_and_5,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(upsert_executions_request.executions(1),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  {
    // Test 5: updating an execution with a mask containing only invalid mask
    // paths has no effect.
    PutExecutionsRequest upsert_executions_request =
        ParseTextProtoOrDie<PutExecutionsRequest>(R"pb(
          executions: {
            external_id: 'unimportant_exeternal_id_value'
            custom_properties {
              key: 'unimportant_property_key'
              value: { bool_value: true }
            }
          }
          update_mask: { paths: 'an_invalid_field_path' }
        )pb");
    upsert_executions_request.mutable_executions(0)->set_type_id(type_id);
    upsert_executions_request.mutable_executions(0)->set_id(
        execution_for_test_3_4_and_5.id());
    PutExecutionsResponse upsert_executions_response;
    ASSERT_EQ(metadata_store_->PutExecutions(upsert_executions_request,
                                             &upsert_executions_response),
              absl::OkStatus());

    GetExecutionsByIDRequest get_executions_by_id_request;
    get_executions_by_id_request.add_execution_ids(
        execution_for_test_3_4_and_5.id());

    GetExecutionsByIDResponse get_executions_by_id_response;
    ASSERT_EQ(metadata_store_->GetExecutionsByID(
                  get_executions_by_id_request, &get_executions_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_executions_by_id_response.executions(), SizeIs(1));

    EXPECT_THAT(
        get_executions_by_id_response.executions(0),
        EqualsProto(
            execution_for_test_3_4_and_5,
            /*ignore_fields=*/{"type", "external_id", "create_time_since_epoch",
                               "last_update_time_since_epoch"}));
  }
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_execution_type_request,
                                              &put_execution_type_response));
  ASSERT_TRUE(put_execution_type_response.has_type_id());

  const int64_t type_id = put_execution_type_response.type_id();

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

  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutions(put_executions_request,
                                           &put_executions_response));
  ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(2));
  const int64_t execution_id_0 = put_executions_response.execution_ids(0);
  const int64_t execution_id_1 = put_executions_response.execution_ids(1);

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  GetExecutionsRequest get_executions_request;
  *get_executions_request.mutable_options() = list_options;

  GetExecutionsResponse get_executions_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutions(get_executions_request,
                                           &get_executions_response));
  EXPECT_THAT(get_executions_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_response.next_page_token(), Not(IsEmpty()));
  EXPECT_EQ(get_executions_response.executions(0).id(), execution_id_1);

  EXPECT_THAT(
      get_executions_response.executions(0),
      EqualsProto(put_executions_request.executions(1),
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));

  list_options.set_next_page_token(get_executions_response.next_page_token());
  *get_executions_request.mutable_options() = list_options;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutions(get_executions_request,
                                           &get_executions_response));
  EXPECT_THAT(get_executions_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_response.next_page_token(), IsEmpty());
  EXPECT_EQ(get_executions_response.executions(0).id(), execution_id_0);
  EXPECT_THAT(
      get_executions_response.executions(0),
      EqualsProto(put_executions_request.executions(0),
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));
}

TEST_P(MetadataStoreTestSuite,
       GetArtifactAndExecutionByTypesWithEmptyDatabase) {
  GetArtifactsByTypeRequest get_artifacts_by_not_exist_type_request;
  GetArtifactsByTypeResponse get_artifacts_by_not_exist_type_response;
  get_artifacts_by_not_exist_type_request.set_type_name("artifact_type");
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByType(
                                  get_artifacts_by_not_exist_type_request,
                                  &get_artifacts_by_not_exist_type_response));
  EXPECT_THAT(get_artifacts_by_not_exist_type_response.artifacts(), SizeIs(0));

  GetExecutionsByTypeRequest get_executions_by_not_exist_type_request;
  GetExecutionsByTypeResponse get_executions_by_not_exist_type_response;
  get_executions_by_not_exist_type_request.set_type_name("execution_type");
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByType(
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_artifact_type_request,
                                             &put_artifact_type_response));
  GetArtifactsByTypeRequest get_artifacts_by_empty_type_request;
  GetArtifactsByTypeResponse get_artifacts_by_empty_type_response;
  get_artifacts_by_empty_type_request.set_type_name("empty_artifact_type");
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByType(
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_execution_type_request,
                                              &put_execution_type_response));
  GetExecutionsByTypeRequest get_executions_by_empty_type_request;
  GetExecutionsByTypeResponse get_executions_by_empty_type_response;
  get_executions_by_empty_type_request.set_type_name("empty_execution_type");
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByType(
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_artifact_type_request,
                                             &put_artifact_type_response));
  const int64_t type_id = put_artifact_type_response.type_id();

  const GetArtifactsByURIRequest get_artifacts_by_uri_empty_db_request;
  GetArtifactsByURIResponse get_artifacts_by_uri_empty_db_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByURI(
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request,
                                          &put_artifacts_response));
  ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(6));

  {
    GetArtifactsByURIRequest get_artifacts_by_uri_request;
    GetArtifactsByURIResponse get_artifacts_by_uri_response;
    get_artifacts_by_uri_request.add_uris("testuri://with_one_artifact");
    ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByURI(
                                    get_artifacts_by_uri_request,
                                    &get_artifacts_by_uri_response));
    EXPECT_THAT(get_artifacts_by_uri_response.artifacts(), SizeIs(1));
  }

  {
    GetArtifactsByURIRequest get_artifacts_by_uri_request;
    GetArtifactsByURIResponse get_artifacts_by_uri_response;
    get_artifacts_by_uri_request.add_uris("testuri://with_multiple_artifacts");
    ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByURI(
                                    get_artifacts_by_uri_request,
                                    &get_artifacts_by_uri_response));
    EXPECT_THAT(get_artifacts_by_uri_response.artifacts(), SizeIs(2));
  }

  {
    // empty uri
    GetArtifactsByURIRequest get_artifacts_by_uri_request;
    get_artifacts_by_uri_request.add_uris("");
    GetArtifactsByURIResponse get_artifacts_by_uri_response;
    ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByURI(
                                    get_artifacts_by_uri_request,
                                    &get_artifacts_by_uri_response));
    EXPECT_THAT(get_artifacts_by_uri_response.artifacts(), SizeIs(3));
  }

  {
    // query uri that does not exist
    GetArtifactsByURIRequest get_artifacts_by_uri_request;
    GetArtifactsByURIResponse get_artifacts_by_uri_response;
    get_artifacts_by_uri_request.add_uris("unknown_uri");
    ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByURI(
                                    get_artifacts_by_uri_request,
                                    &get_artifacts_by_uri_response));
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

    ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByURI(
                                    get_artifacts_by_uri_request,
                                    &get_artifacts_by_uri_response));
    EXPECT_THAT(get_artifacts_by_uri_response.artifacts(), SizeIs(6));
  }

  {
    GetArtifactsByURIRequest request;
    const google::protobuf::Reflection* reflection = request.GetReflection();
    google::protobuf::UnknownFieldSet* fs = reflection->MutableUnknownFields(&request);
    std::string uri = "deprecated_uri_field_value";
    fs->AddLengthDelimited(1)->assign(uri);

    GetArtifactsByURIResponse response;
    absl::Status s = metadata_store_->GetArtifactsByURI(request, &response);
    EXPECT_TRUE(absl::IsInvalidArgument(s));
    EXPECT_TRUE(absl::StrContains(std::string(s.message()),
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_artifact_type_request,
                                             &put_artifact_type_response));
  ASSERT_TRUE(put_artifact_type_response.has_type_id());

  const int64_t type_id = put_artifact_type_response.type_id();

  PutArtifactsRequest put_artifacts_request =
      ParseTextProtoOrDie<PutArtifactsRequest>(R"(
        artifacts: {}
      )");
  put_artifacts_request.mutable_artifacts(0)->set_type_id(type_id);
  PutArtifactsResponse put_artifacts_response;

  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request,
                                          &put_artifacts_response));
  ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(1));
  const int64_t artifact_id = put_artifacts_response.artifact_ids(0);
  GetArtifactsRequest get_artifacts_request;
  GetArtifactsResponse get_artifacts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifacts(get_artifacts_request,
                                          &get_artifacts_response));
  ASSERT_THAT(get_artifacts_response.artifacts(), SizeIs(1));
  EXPECT_EQ(get_artifacts_response.artifacts(0).id(), artifact_id);

  GetArtifactsByTypeRequest get_artifacts_by_type_request;
  GetArtifactsByTypeResponse get_artifacts_by_type_response;
  get_artifacts_by_type_request.set_type_name("test_type2");
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByType(
                                  get_artifacts_by_type_request,
                                  &get_artifacts_by_type_response));
  ASSERT_THAT(get_artifacts_by_type_response.artifacts(), SizeIs(1));
  EXPECT_EQ(get_artifacts_by_type_response.artifacts(0).id(), artifact_id);

  GetArtifactsByTypeRequest get_artifacts_by_not_exist_type_request;
  GetArtifactsByTypeResponse get_artifacts_by_not_exist_type_response;
  get_artifacts_by_not_exist_type_request.set_type_name("not_exist_type");
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByType(
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(request_1, &response_1));

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
  EXPECT_TRUE(absl::IsAlreadyExists(
      metadata_store_->PutExecutionType(request_2, &response_2)));
}

TEST_P(MetadataStoreTestSuite, PutEventGetEvents) {
  const PutExecutionTypeRequest put_execution_type_request =
      ParseTextProtoOrDie<PutExecutionTypeRequest>(
          R"(
            all_fields_match: true
            execution_type: { name: 'test_type' }
          )");
  PutExecutionTypeResponse put_execution_type_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutionType(put_execution_type_request,
                                              &put_execution_type_response));
  ASSERT_TRUE(put_execution_type_response.has_type_id());

  PutExecutionsRequest put_executions_request =
      ParseTextProtoOrDie<PutExecutionsRequest>(R"(
        executions: {}
      )");
  put_executions_request.mutable_executions(0)->set_type_id(
      put_execution_type_response.type_id());
  PutExecutionsResponse put_executions_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutions(put_executions_request,
                                           &put_executions_response));
  ASSERT_THAT(put_executions_response.execution_ids(), SizeIs(1));

  const PutArtifactTypeRequest put_artifact_type_request =
      ParseTextProtoOrDie<PutArtifactTypeRequest>(
          R"(
            all_fields_match: true
            artifact_type: { name: 'test_type' }
          )");
  PutArtifactTypeResponse put_artifact_type_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifactType(put_artifact_type_request,
                                             &put_artifact_type_response));
  ASSERT_TRUE(put_artifact_type_response.has_type_id());
  PutArtifactsRequest put_artifacts_request =
      ParseTextProtoOrDie<PutArtifactsRequest>(R"(
        artifacts: {}
      )");
  put_artifacts_request.mutable_artifacts(0)->set_type_id(
      put_artifact_type_response.type_id());
  PutArtifactsResponse put_artifacts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request,
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
  ASSERT_EQ(absl::OkStatus(), metadata_store_->PutEvents(put_events_request,
                                                         &put_events_response));

  GetEventsByArtifactIDsRequest get_events_by_artifact_ids_request;
  get_events_by_artifact_ids_request.add_artifact_ids(
      put_artifacts_response.artifact_ids(0));
  GetEventsByArtifactIDsResponse get_events_by_artifact_ids_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetEventsByArtifactIDs(
                                  get_events_by_artifact_ids_request,
                                  &get_events_by_artifact_ids_response));
  ASSERT_THAT(get_events_by_artifact_ids_response.events(), SizeIs(1));
  ASSERT_EQ(get_events_by_artifact_ids_response.events(0).execution_id(),
            put_executions_response.execution_ids(0));

  GetEventsByExecutionIDsRequest get_events_by_execution_ids_request;
  get_events_by_execution_ids_request.add_execution_ids(
      put_executions_response.execution_ids(0));
  GetEventsByExecutionIDsResponse get_events_by_execution_ids_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetEventsByExecutionIDs(
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutTypes(put_request, &put_response));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifactType(get_artifact_type_request,
                                             &get_artifact_type_response));
  EXPECT_EQ(put_response.artifact_type_ids(0),
            get_artifact_type_response.artifact_type().id());

  GetExecutionTypeRequest get_execution_type_request =
      ParseTextProtoOrDie<GetExecutionTypeRequest>("type_name: 'test_type2'");
  GetExecutionTypeResponse get_execution_type_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutionType(get_execution_type_request,
                                              &get_execution_type_response));
  EXPECT_EQ(put_response.execution_type_ids(1),
            get_execution_type_response.execution_type().id());

  const GetContextTypeRequest get_context_type_request =
      ParseTextProtoOrDie<GetContextTypeRequest>("type_name: 'test_type1'");
  GetContextTypeResponse get_context_type_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetContextType(get_context_type_request,
                                            &get_context_type_response));
  EXPECT_EQ(put_response.context_type_ids(0),
            get_context_type_response.context_type().id());
}

TEST_P(MetadataStoreTestSuite, PutTypesInsertTypeLink) {
  const PutTypesRequest put_request = ParseTextProtoOrDie<PutTypesRequest>(
      R"pb(
        artifact_types: {
          name: 'test_type1'
          properties { key: 'property_1' value: STRING }
          base_type: MODEL
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
          base_type: TRAIN
        }
      )pb");
  PutTypesResponse put_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutTypes(put_request, &put_response));
  ASSERT_THAT(put_response.artifact_type_ids(), SizeIs(2));
  // Two artifact types with the same type name are inserted. The second
  // artifact type has no base_type, and thus no-op for creating type link.
  // The returned ids are the same.
  EXPECT_EQ(put_response.artifact_type_ids(0),
            put_response.artifact_type_ids(1));
  ASSERT_THAT(put_response.execution_type_ids(), SizeIs(2));
  // Two different execution types are inserted. The returned ids are different.
  EXPECT_NE(put_response.execution_type_ids(0),
            put_response.execution_type_ids(1));

  const GetArtifactTypeRequest get_artifact_type_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>("type_name: 'test_type1'");
  GetArtifactTypeResponse get_artifact_type_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifactType(get_artifact_type_request,
                                             &get_artifact_type_response));
  EXPECT_EQ(put_response.artifact_type_ids(0),
            get_artifact_type_response.artifact_type().id());
  // Verifies that GetArtifactType('test_type1') returns the first artifact type
  // with base_type = MODEL.
  ArtifactType expected_artifact_type = put_request.artifact_types()[0];
  expected_artifact_type.set_id(put_response.artifact_type_ids(0));
  EXPECT_THAT(get_artifact_type_response.artifact_type(),
              EqualsProto(expected_artifact_type));

  const GetExecutionTypeRequest get_execution_type_request =
      ParseTextProtoOrDie<GetExecutionTypeRequest>("type_name: 'test_type2'");
  GetExecutionTypeResponse get_execution_type_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutionType(get_execution_type_request,
                                              &get_execution_type_response));
  EXPECT_EQ(put_response.execution_type_ids(1),
            get_execution_type_response.execution_type().id());
  // Verifies that GetExecutionType('test_type2') returns the second execution
  // type with base_type = TRAIN.
  ExecutionType expected_execution_type = put_request.execution_types()[1];
  expected_execution_type.set_id(put_response.execution_type_ids(1));
  EXPECT_THAT(get_execution_type_response.execution_type(),
              EqualsProto(expected_execution_type));
}

TEST_P(MetadataStoreTestSuite, PutTypesUpdateTypes) {
  // Insert types first, then update them.
  const PutTypesRequest put_request = ParseTextProtoOrDie<PutTypesRequest>(
      R"pb(
        artifact_types: {
          name: 'artifact_type'
          properties { key: 'property_1' value: STRING }
        }
        execution_types: {
          name: 'execution_type'
          properties { key: 'property_1' value: STRING }
        }
        context_types: {
          name: 'context_type'
          properties { key: 'property_1' value: STRING }
        }
      )pb");
  PutTypesResponse put_response;
  ASSERT_EQ(metadata_store_->PutTypes(put_request, &put_response),
            absl::OkStatus());
  ASSERT_THAT(put_response.artifact_type_ids(), SizeIs(1));

  const PutTypesRequest update_request = ParseTextProtoOrDie<PutTypesRequest>(
      R"pb(
        artifact_types: {
          name: 'artifact_type'
          properties { key: 'property_1' value: STRING }
          properties { key: 'property_2' value: STRING }
        }
        execution_types: {
          name: 'execution_type'
          properties { key: 'property_1' value: STRING }
          properties { key: 'property_2' value: STRING }
        }
        context_types: {
          name: 'context_type'
          properties { key: 'property_1' value: STRING }
          properties { key: 'property_2' value: STRING }
        }
        can_add_fields: true
      )pb");
  PutTypesResponse update_response;
  ASSERT_EQ(metadata_store_->PutTypes(update_request, &update_response),
            absl::OkStatus());
  ASSERT_THAT(update_response.artifact_type_ids(), SizeIs(1));
  EXPECT_EQ(update_response.artifact_type_ids(0),
            put_response.artifact_type_ids(0));

  const GetArtifactTypeRequest get_artifact_type_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>("type_name: 'artifact_type'");
  GetArtifactTypeResponse get_artifact_type_response;
  ASSERT_EQ(metadata_store_->GetArtifactType(get_artifact_type_request,
                                             &get_artifact_type_response),
            absl::OkStatus());
  ArtifactType want_artifact_type = update_request.artifact_types(0);
  want_artifact_type.set_id(update_response.artifact_type_ids(0));
  EXPECT_THAT(get_artifact_type_response.artifact_type(),
              EqualsProto(want_artifact_type));

  const GetExecutionTypeRequest get_execution_type_request =
      ParseTextProtoOrDie<GetExecutionTypeRequest>(
          "type_name: 'execution_type'");
  GetExecutionTypeResponse get_execution_type_response;
  ASSERT_EQ(metadata_store_->GetExecutionType(get_execution_type_request,
                                              &get_execution_type_response),
            absl::OkStatus());
  ExecutionType want_execution_type = update_request.execution_types(0);
  want_execution_type.set_id(update_response.execution_type_ids(0));
  EXPECT_THAT(get_execution_type_response.execution_type(),
              EqualsProto(want_execution_type));

  const GetContextTypeRequest get_context_type_request =
      ParseTextProtoOrDie<GetContextTypeRequest>("type_name: 'context_type'");
  GetContextTypeResponse get_context_type_response;
  ASSERT_EQ(metadata_store_->GetContextType(get_context_type_request,
                                            &get_context_type_response),
            absl::OkStatus());
  ContextType want_context_type = update_request.context_types(0);
  want_context_type.set_id(update_response.context_type_ids(0));
  EXPECT_THAT(get_context_type_response.context_type(),
              EqualsProto(want_context_type));
}

TEST_P(MetadataStoreTestSuite, PutTypesUpdateTypesAlreadyExistsError) {
  // Insert types first, then update them.
  const PutTypesRequest put_request = ParseTextProtoOrDie<PutTypesRequest>(
      R"pb(
        artifact_types: {
          name: 'artifact_type'
          properties { key: 'property_1' value: STRING }
        }
        execution_types: {
          name: 'execution_type'
          properties { key: 'property_1' value: STRING }
        }
        context_types: {
          name: 'context_type'
          properties { key: 'property_1' value: STRING }
        }
      )pb");
  PutTypesResponse put_response;
  ASSERT_EQ(metadata_store_->PutTypes(put_request, &put_response),
            absl::OkStatus());
  ASSERT_THAT(put_response.artifact_type_ids(), SizeIs(1));

  // Return ALREADY_EXIST error if can_add_fields is false.
  PutTypesRequest update_request = ParseTextProtoOrDie<PutTypesRequest>(
      R"pb(
        artifact_types: {
          name: 'test_type1'
          properties { key: 'property_1' value: STRING }
          properties { key: 'property_2' value: STRING }
        }
        execution_types: {
          name: 'execution_type'
          properties { key: 'property_1' value: STRING }
          properties { key: 'property_2' value: STRING }
        }
        context_types: {
          name: 'context_type'
          properties { key: 'property_1' value: STRING }
          properties { key: 'property_2' value: STRING }
        }
        can_add_fields: false
        can_omit_fields: false
      )pb");
  PutTypesResponse update_response;
  EXPECT_TRUE(absl::IsAlreadyExists(
      metadata_store_->PutTypes(update_request, &update_response)));

  // Return ALREADY_EXIST error if can_omit_fields is false.
  update_request.mutable_artifact_types(0)->mutable_properties()->erase(
      "property_1");
  update_request.mutable_execution_types(0)->mutable_properties()->erase(
      "property_1");
  update_request.mutable_context_types(0)->mutable_properties()->erase(
      "property_1");
  update_request.set_can_add_fields(true);
  EXPECT_TRUE(absl::IsAlreadyExists(
      metadata_store_->PutTypes(update_request, &update_response)));

  // Return ALREADY_EXIST error if property value type changed.
  update_request = ParseTextProtoOrDie<PutTypesRequest>(
      R"pb(
        artifact_types: {
          name: 'test_type1'
          properties { key: 'property_1' value: INT }
        }
        execution_types: {
          name: 'execution_type'
          properties { key: 'property_1' value: INT }
        }
        context_types: {
          name: 'context_type'
          properties { key: 'property_1' value: INT }
        }
        can_add_fields: true
      )pb");
  EXPECT_TRUE(absl::IsAlreadyExists(
      metadata_store_->PutTypes(update_request, &update_response)));

  // Verify the stored types are still the same as the initial put types.
  const GetArtifactTypeRequest get_artifact_type_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>("type_name: 'artifact_type'");
  GetArtifactTypeResponse get_artifact_type_response;
  ASSERT_EQ(metadata_store_->GetArtifactType(get_artifact_type_request,
                                             &get_artifact_type_response),
            absl::OkStatus());
  ArtifactType want_artifact_type = put_request.artifact_types(0);
  want_artifact_type.set_id(put_response.artifact_type_ids(0));
  EXPECT_THAT(get_artifact_type_response.artifact_type(),
              EqualsProto(want_artifact_type));

  const GetExecutionTypeRequest get_execution_type_request =
      ParseTextProtoOrDie<GetExecutionTypeRequest>(
          "type_name: 'execution_type'");
  GetExecutionTypeResponse get_execution_type_response;
  ASSERT_EQ(metadata_store_->GetExecutionType(get_execution_type_request,
                                              &get_execution_type_response),
            absl::OkStatus());
  ExecutionType want_execution_type = put_request.execution_types(0);
  want_execution_type.set_id(put_response.execution_type_ids(0));
  EXPECT_THAT(get_execution_type_response.execution_type(),
              EqualsProto(want_execution_type));

  const GetContextTypeRequest get_context_type_request =
      ParseTextProtoOrDie<GetContextTypeRequest>("type_name: 'context_type'");
  GetContextTypeResponse get_context_type_response;
  ASSERT_EQ(metadata_store_->GetContextType(get_context_type_request,
                                            &get_context_type_response),
            absl::OkStatus());
  ContextType want_context_type = put_request.context_types(0);
  want_context_type.set_id(put_response.context_type_ids(0));
  EXPECT_THAT(get_context_type_response.context_type(),
              EqualsProto(want_context_type));
}

TEST_P(MetadataStoreTestSuite, PutAndGetExecution) {
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"(
    artifact_types: { name: 'artifact_type' }
    execution_types: {
      name: 'execution_type'
      properties { key: 'running_status' value: STRING }
    })");
  PutTypesResponse put_types_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutTypes(put_types_request, &put_types_response));
  int64_t artifact_type_id = put_types_response.artifact_type_ids(0);
  int64_t execution_type_id = put_types_response.execution_type_ids(0);

  // 1. Insert an execution first time without any artifact and event pair.
  Execution execution;
  execution.set_type_id(execution_type_id);
  (*execution.mutable_properties())["running_status"].set_string_value("INIT");
  execution.set_last_known_state(Execution::NEW);

  PutExecutionRequest put_execution_request_1;
  *put_execution_request_1.mutable_execution() = execution;
  PutExecutionResponse put_execution_response_1;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request_1,
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request_2,
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request_3,
                                          &put_execution_response_3));
  EXPECT_EQ(put_execution_response_3.execution_id(), execution.id());
  EXPECT_THAT(put_execution_response_3.artifact_ids(), SizeIs(2));
  EXPECT_EQ(put_execution_response_3.artifact_ids(0), artifact_1.id());
  artifact_2.set_id(put_execution_response_3.artifact_ids(1));

  // Test empty artifact and event pairs.
  PutExecutionRequest put_execution_request_4;
  *put_execution_request_4.mutable_execution() = execution;
  put_execution_request_4.add_artifact_event_pairs();
  put_execution_request_4.add_artifact_event_pairs();
  PutExecutionResponse put_execution_response_4;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request_4,
                                          &put_execution_response_4));
  EXPECT_THAT(put_execution_response_4.artifact_ids(), SizeIs(2));
  EXPECT_THAT(put_execution_response_4.artifact_ids(), Each(Eq(-1)));

  // In the end, there should be 2 artifacts, 1 execution and 2 events.
  GetArtifactsRequest get_artifacts_request;
  GetArtifactsResponse get_artifacts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifacts(get_artifacts_request,
                                          &get_artifacts_response));
  ASSERT_THAT(get_artifacts_response.artifacts(), SizeIs(2));
  EXPECT_THAT(
      get_artifacts_response.artifacts(),
      UnorderedElementsAre(
          EqualsProto(artifact_1,
                      /*ignore_fields=*/{"type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"}),
          EqualsProto(artifact_2,
                      /*ignore_fields=*/{"type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"})));

  GetExecutionsRequest get_executions_request;
  GetExecutionsResponse get_executions_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutions(get_executions_request,
                                           &get_executions_response));
  ASSERT_THAT(get_executions_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_response.executions(0),
              EqualsProto(execution,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));
  GetEventsByExecutionIDsRequest get_events_request;
  get_events_request.add_execution_ids(execution.id());
  GetEventsByExecutionIDsResponse get_events_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetEventsByExecutionIDs(
                                  get_events_request, &get_events_response));
  ASSERT_THAT(get_events_response.events(), SizeIs(2));
  std::vector<int64_t> got_events_artifact_ids = {
      get_events_response.events(0).artifact_id(),
      get_events_response.events(1).artifact_id()};
  EXPECT_THAT(got_events_artifact_ids,
              UnorderedElementsAre(artifact_1.id(), artifact_2.id()));

  // Check that execution's update time is equal or larger than the create time
  // of all the artifacts.
  std::vector<int64_t> artifact_create_time;
  for (const Artifact& artifact : get_artifacts_response.artifacts()) {
    artifact_create_time.push_back(artifact.create_time_since_epoch());
  }
  int64_t max_artifact_create_time = *std::max_element(
      artifact_create_time.begin(), artifact_create_time.end());
  EXPECT_GT(max_artifact_create_time, 0);
  for (const Execution& execution : get_executions_response.executions()) {
    EXPECT_GE(execution.last_update_time_since_epoch(),
              max_artifact_create_time);
  }
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
  ASSERT_EQ(absl::OkStatus(),
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response));
  // check the nodes of the end state graph
  ASSERT_GE(put_execution_response.execution_id(), 0);
  ASSERT_THAT(put_execution_response.artifact_ids(), SizeIs(1));
  ASSERT_THAT(put_execution_response.context_ids(), SizeIs(2));
  context1.set_id(put_execution_response.context_ids(0));
  context2.set_id(put_execution_response.context_ids(1));
  GetContextsResponse get_contexts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetContexts({}, &get_contexts_response));
  EXPECT_THAT(
      get_contexts_response.contexts(),
      UnorderedElementsAre(
          EqualsProto(context1,
                      /*ignore_fields=*/{"type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"}),
          EqualsProto(context2,
                      /*ignore_fields=*/{"type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"})));

  // check attributions and associations of each context.
  for (const int64_t context_id : put_execution_response.context_ids()) {
    GetArtifactsByContextRequest get_artifacts_by_context_request;
    get_artifacts_by_context_request.set_context_id(context_id);
    GetArtifactsByContextResponse get_artifacts_by_context_response;
    ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByContext(
                                    get_artifacts_by_context_request,
                                    &get_artifacts_by_context_response));
    ASSERT_THAT(get_artifacts_by_context_response.artifacts(), SizeIs(1));
    EXPECT_EQ(get_artifacts_by_context_response.artifacts(0).id(),
              put_execution_response.artifact_ids(0));

    GetExecutionsByContextRequest get_executions_by_context_request;
    get_executions_by_context_request.set_context_id(context_id);
    GetExecutionsByContextResponse get_executions_by_context_response;
    ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByContext(
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutTypes(put_types_request, &put_types_response));
  Context context1;
  context1.set_type_id(put_types_response.context_type_ids(0));
  context1.set_name("context1");

  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = context1;
  PutContextsResponse put_contexts_response;

  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request,
                                         &put_contexts_response));

  ASSERT_EQ(put_contexts_response.context_ids().size(), 1);
  int64_t context_id = put_contexts_response.context_ids(0);

  PutExecutionRequest put_execution_request;
  put_execution_request.mutable_execution()->set_type_id(
      put_types_response.execution_type_ids(0));
  // calls PutExecution and test end states.
  PutExecutionResponse put_execution_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response));
  // check the nodes of the end state graph
  ASSERT_GE(put_execution_response.execution_id(), 0);
  int64_t execution_id_1 = put_execution_response.execution_id();

  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response));
  // check the nodes of the end state graph
  ASSERT_GE(put_execution_response.execution_id(), 1);
  int64_t execution_id_2 = put_execution_response.execution_id();

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

  ASSERT_EQ(absl::OkStatus(), metadata_store_->PutAttributionsAndAssociations(
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
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByContext(
                                  get_executions_by_context_request,
                                  &get_executions_by_context_response));

  EXPECT_THAT(get_executions_by_context_response.executions(), SizeIs(1));
  ASSERT_EQ(get_executions_by_context_response.executions(0).id(),
            execution_id_2);
  ASSERT_FALSE(get_executions_by_context_response.next_page_token().empty());

  list_options.set_next_page_token(
      get_executions_by_context_response.next_page_token());
  *get_executions_by_context_request.mutable_options() = list_options;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByContext(
                                  get_executions_by_context_request,
                                  &get_executions_by_context_response));

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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutTypes(put_types_request, &put_types_response));
  Context context1;
  context1.set_type_id(put_types_response.context_type_ids(0));
  context1.set_name("context1");

  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = context1;
  PutContextsResponse put_contexts_response;

  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request,
                                         &put_contexts_response));

  ASSERT_EQ(put_contexts_response.context_ids().size(), 1);
  int64_t context_id = put_contexts_response.context_ids(0);

  PutArtifactsRequest put_artifacts_request_1;
  Artifact artifact1;
  artifact1.set_type_id(put_types_response.artifact_type_ids(0));
  Artifact artifact2;
  artifact2.set_type_id(put_types_response.artifact_type_ids(0));
  *put_artifacts_request_1.add_artifacts() = artifact1;
  *put_artifacts_request_1.add_artifacts() = artifact2;

  PutArtifactsResponse put_artifacts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request_1,
                                          &put_artifacts_response));
  ASSERT_EQ(put_artifacts_response.artifact_ids().size(), 2);
  int64_t artifact_id_1 = put_artifacts_response.artifact_ids(0);
  int64_t artifact_id_2 = put_artifacts_response.artifact_ids(1);

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

  ASSERT_EQ(absl::OkStatus(), metadata_store_->PutAttributionsAndAssociations(
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
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByContext(
                                  get_artifacts_by_context_request,
                                  &get_artifacts_by_context_response));

  EXPECT_THAT(get_artifacts_by_context_response.artifacts(), SizeIs(1));
  ASSERT_EQ(get_artifacts_by_context_response.artifacts(0).id(), artifact_id_2);
  ASSERT_FALSE(get_artifacts_by_context_response.next_page_token().empty());

  list_options.set_next_page_token(
      get_artifacts_by_context_response.next_page_token());
  *get_artifacts_by_context_request.mutable_options() = list_options;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByContext(
                                  get_artifacts_by_context_request,
                                  &get_artifacts_by_context_response));

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
  ASSERT_EQ(absl::OkStatus(),
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(request, &response));
  // A call with the same request fails, as the context already exists.
  const absl::Status duplicate_update_status =
      metadata_store_->PutExecution(request, &response);
  EXPECT_TRUE(absl::IsAlreadyExists(duplicate_update_status));
  // If set `reuse_context_if_already_exist`, it succeeds.
  request.mutable_options()->set_reuse_context_if_already_exist(true);
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(request, &response));

  // Check the stored nodes, there should be 1 context and 2 executions.
  GetContextsResponse get_contexts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetContexts({}, &get_contexts_response));
  ASSERT_THAT(get_contexts_response.contexts(), SizeIs(1));
  const Context& stored_context = get_contexts_response.contexts(0);
  EXPECT_THAT(stored_context,
              EqualsProto(context, /*ignore_fields=*/{
                              "id", "type", "create_time_since_epoch",
                              "last_update_time_since_epoch"}));
  GetExecutionsByContextRequest get_executions_by_context_request;
  get_executions_by_context_request.set_context_id(stored_context.id());
  GetExecutionsByContextResponse get_executions_by_context_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByContext(
                                  get_executions_by_context_request,
                                  &get_executions_by_context_response));
  ASSERT_THAT(get_executions_by_context_response.executions(), SizeIs(2));
  EXPECT_THAT(
      get_executions_by_context_response.executions(),
      UnorderedElementsAre(
          EqualsProto(
              request.execution(),
              /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                 "last_update_time_since_epoch"}),
          EqualsProto(
              request.execution(),
              /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                 "last_update_time_since_epoch"})));
}

// Call PutExecution with `force_reuse_context`, and verify that the context
// is not updated.
TEST_P(MetadataStoreTestSuite, PutExecutionWithForceReuseContextOption) {
  // Prepares input that consists of 1 execution and 1 context,
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    context_types: { name: 'context_type' }
    execution_types: { name: 'execution_type' })pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(metadata_store_->PutTypes(put_types_request, &put_types_response),
            absl::OkStatus());
  const int64_t context_type_id = put_types_response.context_type_ids(0);
  const int64_t execution_type_id = put_types_response.execution_type_ids(0);

  Context original_context;
  original_context.set_type_id(context_type_id);
  original_context.set_name("context");
  (*original_context.mutable_custom_properties())["property1"].set_string_value(
      "value1");

  {
    PutContextsRequest put_contexts_request;
    *put_contexts_request.add_contexts() = original_context;
    PutContextsResponse put_contexts_response;
    ASSERT_EQ(metadata_store_->PutContexts(put_contexts_request,
                                           &put_contexts_response),
              absl::OkStatus());
    ASSERT_EQ(put_contexts_response.context_ids().size(), 1);
    original_context.set_id(put_contexts_response.context_ids(0));
  }

  Context modified_context;
  modified_context.CopyFrom(original_context);
  (*modified_context.mutable_custom_properties())["property1"].set_string_value(
      "changed");
  (*modified_context.mutable_custom_properties())["property2"].set_string_value(
      "added");

  PutExecutionRequest request;
  request.mutable_execution()->set_type_id(execution_type_id);
  *request.add_contexts() = modified_context;
  request.mutable_options()->set_force_reuse_context(true);

  PutExecutionResponse response;
  ASSERT_EQ(metadata_store_->PutExecution(request, &response),
            absl::OkStatus());

  // Check that the context was not modified.
  GetContextsResponse get_contexts_response;
  ASSERT_EQ(metadata_store_->GetContexts({}, &get_contexts_response),
            absl::OkStatus());
  ASSERT_THAT(get_contexts_response.contexts(), SizeIs(1));
  const Context& stored_context = get_contexts_response.contexts(0);
  EXPECT_THAT(stored_context, EqualsProto(original_context, /*ignore_fields=*/{
                                              "type", "create_time_since_epoch",
                                              "last_update_time_since_epoch"}));
}

// Call PutExecution with `force_reuse_context`, when the context does not
// exist.
TEST_P(MetadataStoreTestSuite,
       PutExecutionWithForceReuseContextOptionContextDoesNotExist) {
  // Prepares input that consists of 1 execution and 1 context,
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    context_types: { name: 'context_type' }
    execution_types: { name: 'execution_type' })pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(metadata_store_->PutTypes(put_types_request, &put_types_response),
            absl::OkStatus());
  const int64_t context_type_id = put_types_response.context_type_ids(0);
  const int64_t execution_type_id = put_types_response.execution_type_ids(0);

  Context context;
  context.set_id(12345);  // Arbitrary ID. This context doesn't exist.
  context.set_type_id(context_type_id);
  context.set_name("context");

  PutExecutionRequest request;
  request.mutable_execution()->set_type_id(execution_type_id);
  *request.add_contexts() = context;
  request.mutable_options()->set_force_reuse_context(true);

  PutExecutionResponse response;
  EXPECT_TRUE(absl::IsNotFound(
      metadata_store_->PutExecution(request, &response)));
}

// Call PutExecution with both `force_reuse_context` and
// `force_reuse_context_if_exists`.
TEST_P(MetadataStoreTestSuite,
       PutExecutionWithForceReuseContextAndReuseContextIfAlreadyExistOptions) {
  // Prepares input that consists of 1 execution and 1 context,
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    context_types: { name: 'context_type' }
    execution_types: { name: 'execution_type' })pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(metadata_store_->PutTypes(put_types_request, &put_types_response),
            absl::OkStatus());
  const int64_t context_type_id = put_types_response.context_type_ids(0);
  const int64_t execution_type_id = put_types_response.execution_type_ids(0);

  Context context_one;
  context_one.set_type_id(context_type_id);
  context_one.set_name("one");
  (*context_one.mutable_custom_properties())["property1"].set_string_value(
      "value1");

  Context context_two;
  context_two.set_type_id(context_type_id);
  context_two.set_name("two");
  (*context_two.mutable_custom_properties())["property2"].set_string_value(
      "value2");

  int64_t context_one_id;
  int64_t context_two_id;
  {
    PutContextsRequest put_contexts_request;
    *put_contexts_request.add_contexts() = context_one;
    *put_contexts_request.add_contexts() = context_two;
    PutContextsResponse put_contexts_response;
    ASSERT_EQ(metadata_store_->PutContexts(put_contexts_request,
                                           &put_contexts_response),
              absl::OkStatus());
    ASSERT_EQ(put_contexts_response.context_ids().size(), 2);
    context_one_id = put_contexts_response.context_ids(0);
    context_two_id = put_contexts_response.context_ids(1);
  }

  // modified_context_one has a context ID, so it won't be updated.
  // modified_context_two has no context ID, but reuse_context_if_already_exist
  // is set, and it matches the name and type ID of an exiting context, so it
  // also won't be updated.
  Context modified_context_one;
  modified_context_one.CopyFrom(context_one);
  modified_context_one.set_id(context_one_id);
  (*modified_context_one.mutable_custom_properties())["property1"]
      .set_string_value("changed1");
  Context modified_context_two;
  modified_context_two.CopyFrom(context_two);
  (*modified_context_one.mutable_custom_properties())["property2"]
      .set_string_value("changed2");

  PutExecutionRequest request;
  request.mutable_execution()->set_type_id(execution_type_id);
  *request.add_contexts() = modified_context_one;
  *request.add_contexts() = modified_context_two;
  request.mutable_options()->set_force_reuse_context(true);
  request.mutable_options()->set_reuse_context_if_already_exist(true);

  PutExecutionResponse response;
  ASSERT_EQ(metadata_store_->PutExecution(request, &response),
            absl::OkStatus());

  // Check that the contexts were not modified.
  GetContextsResponse get_contexts_response;
  ASSERT_EQ(metadata_store_->GetContexts({}, &get_contexts_response),
            absl::OkStatus());
  const std::vector<std::string> ignore_fields = {
      "type", "create_time_since_epoch", "last_update_time_since_epoch"};
  // Set the IDs for comparison.
  context_one.set_id(context_one_id);
  context_two.set_id(context_two_id);
  EXPECT_THAT(get_contexts_response.contexts(),
              UnorderedElementsAre(EqualsProto(context_one, ignore_fields),
                                   EqualsProto(context_two, ignore_fields)));
}

TEST_P(MetadataStoreTestSuite, PutAndGetExecutionWithArtifactReuseOption) {
  // Setup: Prepares input that consists of 1 execution and 1 artifact,
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    artifact_types: { name: 'artifact_type' }
    execution_types: { name: 'execution_type' })pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(metadata_store_->PutTypes(put_types_request, &put_types_response),
            absl::OkStatus());
  Artifact artifact = ParseTextProtoOrDie<Artifact>(
      R"pb(
        external_id: 'artifact_reference_str'
        name: 'artifact_name'
      )pb");
  artifact.set_type_id(put_types_response.artifact_type_ids(0));
  PutExecutionRequest put_execution_request;
  put_execution_request.mutable_execution()->set_type_id(
      put_types_response.execution_type_ids(0));
  *put_execution_request.add_artifact_event_pairs()->mutable_artifact() =
      artifact;

  // Test case 1: id not exists and option=false. Expect regular insertion
  // behavior holds
  // Test 1.1: The first call succeeds.
  PutExecutionResponse put_execution_response;
  ASSERT_EQ(metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response),
            absl::OkStatus());
  ASSERT_EQ(put_execution_response.artifact_ids_size(), 1);
  const int64_t artifact_id = put_execution_response.artifact_ids(0);

  // Test 1.2: Try to insert another artifact with the same external_id but a
  // new uri fails as the artifact with the same external_id already exists.
  const std::string kUriStr = "new_uri";
  put_execution_request.mutable_artifact_event_pairs(0)
      ->mutable_artifact()
      ->set_uri(kUriStr);
  EXPECT_TRUE(absl::IsAlreadyExists(metadata_store_->PutExecution(
      put_execution_request, &put_execution_response)));

  // Test case 2: id not exists and option=true. Expect update.
  // Test 2.1: Same request succeeds if
  // `set_reuse_artifact_if_already_exist_by_external_id=true`.
  put_execution_request.mutable_options()
      ->set_reuse_artifact_if_already_exist_by_external_id(true);
  ASSERT_EQ(metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response),
            absl::OkStatus());
  GetArtifactsByIDRequest get_artifact_request;
  get_artifact_request.add_artifact_ids(artifact_id);
  GetArtifactsByIDResponse get_artifact_response;
  ASSERT_EQ(metadata_store_->GetArtifactsByID(get_artifact_request,
                                              &get_artifact_response),
            absl::OkStatus());
  ASSERT_EQ(get_artifact_response.artifacts_size(), 1);
  EXPECT_EQ(
      get_artifact_response.artifacts(0).uri(), kUriStr);

  // Test 2.2: Put input artifact which has the same <type_id, name> pair
  // but no external_id with
  // `set_reuse_artifact_if_already_exist_by_external_id=true`.
  // Expect insert instead of updaate but failed with <type_id, name> unique
  // index violation.
  put_execution_request.mutable_artifact_event_pairs(0)
      ->mutable_artifact()
      ->clear_external_id();
  EXPECT_TRUE(absl::IsAlreadyExists(metadata_store_->PutExecution(
      put_execution_request, &put_execution_response)));

  // Test case 3: id exists and option=true. Expect update applies to
  // external_id as well.
  // Test 3: external_id updated if external_id is changed and
  // `set_reuse_artifact_if_already_exist_by_external_id=true`.
  const std::string kNewExternalIdStr = "new_external_id";
  put_execution_request.mutable_artifact_event_pairs(0)
      ->mutable_artifact()
      ->set_external_id(kNewExternalIdStr);
  put_execution_request.mutable_artifact_event_pairs(0)
      ->mutable_artifact()
      ->set_id(artifact_id);
  ASSERT_EQ(metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response),
            absl::OkStatus());
  ASSERT_EQ(metadata_store_->GetArtifactsByID(get_artifact_request,
                                              &get_artifact_response),
            absl::OkStatus());
  ASSERT_EQ(get_artifact_response.artifacts_size(), 1);
  EXPECT_EQ(get_artifact_response.artifacts(0).external_id(),
            kNewExternalIdStr);
}

TEST_P(MetadataStoreTestSuite, PutAndGetExecutionWithDuplicatedArtifacts) {
  // Setup: Insert different types.
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    artifact_types: { name: 'artifact_type' }
    execution_types: { name: 'execution_type' }
    context_types: { name: 'context_type' }
  )pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(metadata_store_->PutTypes(put_types_request, &put_types_response),
            absl::OkStatus());
  // Setup: Insert 1 context.
  Context context = ParseTextProtoOrDie<Context>(
      R"pb(
        name: 'context_name'
      )pb");
  context.set_type_id(put_types_response.context_type_ids(0));
  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = context;
  PutContextsResponse put_contexts_response;
  ASSERT_EQ(metadata_store_->PutContexts(put_contexts_request,
                                         &put_contexts_response),
            absl::OkStatus());
  context.set_id(put_contexts_response.context_ids(0));
  // Setup: Insert 1 artifact.
  Artifact artifact = ParseTextProtoOrDie<Artifact>(
      R"pb(
        external_id: 'artifact_reference_str' name: 'artifact_name'
      )pb");
  artifact.set_type_id(put_types_response.artifact_type_ids(0));

  PutArtifactsRequest put_artifact_request;
  *put_artifact_request.add_artifacts() = artifact;
  PutArtifactsResponse put_artifact_response;
  ASSERT_EQ(metadata_store_->PutArtifacts(put_artifact_request,
                                          &put_artifact_response),
            absl::OkStatus());

  // Ack: Call put execution with duplicate artifacts.
  // Expect call succeeds without error due to built-in artifact de-dup.
  PutExecutionRequest put_execution_request;
  put_execution_request.mutable_execution()->set_type_id(
      put_types_response.execution_type_ids(0));
  *put_execution_request.add_artifact_event_pairs()->mutable_artifact() =
      artifact;
  *put_execution_request.add_artifact_event_pairs()->mutable_artifact() =
      artifact;
  *put_execution_request.add_contexts() = context;
  put_execution_request.mutable_options()
      ->set_reuse_artifact_if_already_exist_by_external_id(true);
  put_execution_request.mutable_options()->set_reuse_context_if_already_exist(
      true);
  PutExecutionResponse put_execution_response;
  ASSERT_EQ(metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response),
            absl::OkStatus());

  // Ack: Call GetArtifactsByContext to confirm that attribution relationship
  // is correctly created between context and artifact.
  GetArtifactsByContextRequest get_artifact_by_context_request;
  get_artifact_by_context_request.set_context_id(context.id());
  GetArtifactsByContextResponse get_artifact_by_context_response;
  ASSERT_EQ(
      metadata_store_->GetArtifactsByContext(get_artifact_by_context_request,
                                             &get_artifact_by_context_response),
      absl::OkStatus());
  EXPECT_THAT(
      get_artifact_by_context_response.artifacts(),
      UnorderedElementsAre(EqualsProto(
          artifact,
          /*ignore_fields=*/{"id", "type", "uri", "create_time_since_epoch",
                             "last_update_time_since_epoch"})));
}

TEST_P(MetadataStoreTestSuite,
       PutAndGetLineageSubgraphWithArtifactReuseOption) {
  // Setup: Prepares input that consists of 1 execution and 1 artifact,
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    artifact_types: { name: 'artifact_type' }
    execution_types: { name: 'execution_type' })pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(metadata_store_->PutTypes(put_types_request, &put_types_response),
            absl::OkStatus());
  Artifact artifact = ParseTextProtoOrDie<Artifact>(
      R"pb(
        external_id: 'artifact_reference_str'
        name: 'artifact_name'
      )pb");
  artifact.set_type_id(put_types_response.artifact_type_ids(0));
  PutLineageSubgraphRequest put_subgraph_request;
  put_subgraph_request.add_executions()->set_type_id(
      put_types_response.execution_type_ids(0));
  *put_subgraph_request.add_artifacts() = artifact;

  // Test case 1: id not exists and option=false. Expect regular insertion
  // behavior holds
  // Test 1.1: The first call succeeds.
  PutLineageSubgraphResponse put_subgraph_response;
  ASSERT_EQ(metadata_store_->PutLineageSubgraph(put_subgraph_request,
                                                &put_subgraph_response),
            absl::OkStatus());
  ASSERT_EQ(put_subgraph_response.artifact_ids_size(), 1);
  const int64_t artifact_id = put_subgraph_response.artifact_ids(0);

  // Test 1.2: Try to insert another artifact with the same external_id but a
  // new uri fails as the artifact with the same external_id already exists.
  const std::string kUriStr = "new_uri";
  put_subgraph_request.mutable_artifacts(0)->set_uri(kUriStr);
  EXPECT_TRUE(absl::IsAlreadyExists(metadata_store_->PutLineageSubgraph(
      put_subgraph_request, &put_subgraph_response)));

  // Test case 2: id not exists and option=true. Expect update.
  // Test 2.1: Same request succeeds if
  // `set_reuse_artifact_if_already_exist_by_external_id=true`.
  put_subgraph_request.mutable_options()
      ->set_reuse_artifact_if_already_exist_by_external_id(true);
  ASSERT_EQ(metadata_store_->PutLineageSubgraph(put_subgraph_request,
                                                &put_subgraph_response),
            absl::OkStatus());
  GetArtifactsByIDRequest get_artifact_request;
  get_artifact_request.add_artifact_ids(artifact_id);
  GetArtifactsByIDResponse get_artifact_response;
  ASSERT_EQ(metadata_store_->GetArtifactsByID(get_artifact_request,
                                              &get_artifact_response),
            absl::OkStatus());
  ASSERT_EQ(get_artifact_response.artifacts_size(), 1);
  EXPECT_EQ(get_artifact_response.artifacts(0).uri(), kUriStr);

  // Test 2.2: Put input artifact which has the same <type_id, name> pair
  // but no external_id with
  // `set_reuse_artifact_if_already_exist_by_external_id=true`.
  // Expect insert instead of update but failed with <type_id, name> unique
  // index violation.
  put_subgraph_request.mutable_artifacts(0)->clear_external_id();
  EXPECT_TRUE(absl::IsAlreadyExists(metadata_store_->PutLineageSubgraph(
      put_subgraph_request, &put_subgraph_response)));

  // Test case 3: id exists and option=true. Expect update applies to
  // external_id as well.
  // Test 3: external_id updated if external_id is changed and
  // `set_reuse_artifact_if_already_exist_by_external_id=true`.
  const std::string kNewExternalIdStr = "new_external_id";
  put_subgraph_request.mutable_artifacts(0)->set_external_id(kNewExternalIdStr);
  put_subgraph_request.mutable_artifacts(0)->set_id(artifact_id);
  ASSERT_EQ(metadata_store_->PutLineageSubgraph(put_subgraph_request,
                                                &put_subgraph_response),
            absl::OkStatus());
  ASSERT_EQ(metadata_store_->GetArtifactsByID(get_artifact_request,
                                              &get_artifact_response),
            absl::OkStatus());
  ASSERT_EQ(get_artifact_response.artifacts_size(), 1);
  EXPECT_EQ(get_artifact_response.artifacts(0).external_id(),
            kNewExternalIdStr);

  // Test case 4: Try to insert a new artifact with a new external_id that does
  // not exist and option=true. Expect regular insertion behavior holds.
  Artifact another_artifact = ParseTextProtoOrDie<Artifact>(
      R"pb(
        external_id: 'another_artifact_reference_str'
        name: 'another_artifact_name'
      )pb");
  another_artifact.set_type_id(put_types_response.artifact_type_ids(0));
  put_subgraph_request.mutable_artifacts(0)->Swap(&another_artifact);
  ASSERT_EQ(metadata_store_->PutLineageSubgraph(put_subgraph_request,
                                                &put_subgraph_response),
            absl::OkStatus());
  get_artifact_request.clear_artifact_ids();
  get_artifact_request.add_artifact_ids(put_subgraph_response.artifact_ids(0));
  ASSERT_EQ(metadata_store_->GetArtifactsByID(get_artifact_request,
                                              &get_artifact_response),
            absl::OkStatus());
  ASSERT_EQ(get_artifact_response.artifacts_size(), 1);
  EXPECT_EQ(get_artifact_response.artifacts(0).external_id(),
            "another_artifact_reference_str");
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContextType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());

  GetContextTypeRequest get_request =
      ParseTextProtoOrDie<GetContextTypeRequest>("type_name: 'test_type'");
  GetContextTypeResponse get_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetContextType(get_request, &get_response));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContextType(put_request_1, &put_response));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContextType(put_request_2, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  ContextType type_2 = ParseTextProtoOrDie<ContextType>(
      R"(
        name: 'test_type_2'
        properties { key: 'property_2' value: INT }
      )");
  type_2.set_id(put_response.type_id());

  GetContextTypesRequest get_request;
  GetContextTypesResponse got_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetContextTypes(get_request, &got_response));
  GetContextTypesResponse want_response;
  *want_response.add_context_types() = type_1;
  *want_response.add_context_types() = type_2;
  EXPECT_THAT(got_response, EqualsProto(want_response));
}

TEST_P(MetadataStoreTestSuite, PutContextTypesGetContextTypesByExternalIds) {
  constexpr absl::string_view kContextTypeTemplate = R"pb(
    all_fields_match: true
    context_type: {
      name: '%s'
      external_id: '%s'
      properties { key: 'property' value: STRING }
    }
  )pb";
  const PutContextTypeRequest put_context_type_1_request =
      ParseTextProtoOrDie<PutContextTypeRequest>(
          absl::StrFormat(kContextTypeTemplate, "test_context_type_1",
                          "test_context_type_external_id_1"));
  const PutContextTypeRequest put_context_type_2_request =
      ParseTextProtoOrDie<PutContextTypeRequest>(
          absl::StrFormat(kContextTypeTemplate, "test_context_type_2",
                          "test_context_type_external_id_2"));
  ContextType context_type1, context_type2;
  // Create the types
  {
    PutContextTypeResponse put_context_type_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutContextType(put_context_type_1_request,
                                              &put_context_type_response));
    ASSERT_TRUE(put_context_type_response.has_type_id());
    context_type1 = ParseTextProtoOrDie<ContextType>(absl::StrFormat(
        R"pb(
          name: '%s'
          external_id: '%s'
          properties { key: 'property' value: STRING }
        )pb",
        "test_context_type_1", "test_context_type_external_id_1"));
    context_type1.set_id(put_context_type_response.type_id());

    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutContextType(put_context_type_2_request,
                                              &put_context_type_response));
    ASSERT_TRUE(put_context_type_response.has_type_id());
    context_type2 = ParseTextProtoOrDie<ContextType>(absl::StrFormat(
        R"pb(
          name: '%s'
          external_id: '%s'
          properties { key: 'property' value: STRING }
        )pb",
        "test_context_type_2", "test_context_type_external_id_2"));
    context_type2.set_id(put_context_type_response.type_id());
  }
  // Test: retrieve by one external id
  {
    GetContextTypesByExternalIdsRequest
        get_context_types_by_external_ids_request;
    get_context_types_by_external_ids_request.add_external_ids(
        context_type1.external_id());
    GetContextTypesByExternalIdsResponse
        get_context_types_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(),
              metadata_store_->GetContextTypesByExternalIds(
                  get_context_types_by_external_ids_request,
                  &get_context_types_by_external_ids_response));
    EXPECT_THAT(get_context_types_by_external_ids_response.context_types(),
                ElementsAre(EqualsProto(context_type1)));
  }
  // Test: retrieve by one non-existing external id
  {
    GetContextTypesByExternalIdsRequest
        get_context_types_by_external_ids_request;
    get_context_types_by_external_ids_request.add_external_ids(
        "context_type_absent_external_id");
    GetContextTypesByExternalIdsResponse
        get_context_types_by_external_ids_response;
    EXPECT_TRUE(absl::IsNotFound(metadata_store_->GetContextTypesByExternalIds(
        get_context_types_by_external_ids_request,
        &get_context_types_by_external_ids_response)));
  }
  // Test: retrieve by multiple external ids
  {
    GetContextTypesByExternalIdsRequest
        get_context_types_by_external_ids_request;

    // Can retrieve ContextTypes by multiple external ids
    get_context_types_by_external_ids_request.add_external_ids(
        context_type1.external_id());
    get_context_types_by_external_ids_request.add_external_ids(
        context_type2.external_id());
    GetContextTypesByExternalIdsResponse
        get_context_types_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(),
              metadata_store_->GetContextTypesByExternalIds(
                  get_context_types_by_external_ids_request,
                  &get_context_types_by_external_ids_response));
    EXPECT_THAT(get_context_types_by_external_ids_response.context_types(),
                UnorderedElementsAre(EqualsProto(context_type1),
                                     EqualsProto(context_type2)));

    // Will return whatever found if some of the external ids is absent
    get_context_types_by_external_ids_request.add_external_ids(
        "context_type_absent_external_id");
    EXPECT_EQ(absl::OkStatus(),
              metadata_store_->GetContextTypesByExternalIds(
                  get_context_types_by_external_ids_request,
                  &get_context_types_by_external_ids_response));
    EXPECT_THAT(get_context_types_by_external_ids_response.context_types(),
                UnorderedElementsAre(EqualsProto(context_type1),
                                     EqualsProto(context_type2)));
  }

  // Test retrieve by empty external id
  {
    GetContextTypesByExternalIdsRequest
        get_context_types_by_external_ids_request;
    get_context_types_by_external_ids_request.add_external_ids("");
    GetContextTypesByExternalIdsResponse
        get_context_types_by_external_ids_response;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_store_->GetContextTypesByExternalIds(
            get_context_types_by_external_ids_request,
            &get_context_types_by_external_ids_response)));
  }
}

TEST_P(MetadataStoreTestSuite, GetContextTypesWhenNoneExist) {
  GetContextTypesRequest get_request;
  GetContextTypesResponse got_response;

  // Expect OK status and empty response.
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetContextTypes(get_request, &got_response));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContextType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());

  // Get types by exist and non-exist ids.
  GetContextTypesByIDRequest get_request;
  get_request.add_type_ids(put_response.type_id());
  get_request.add_type_ids(put_response.type_id() + 100);
  GetContextTypesByIDResponse get_response;
  ASSERT_EQ(absl::OkStatus(),
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

TEST_P(MetadataStoreTestSuite, UpdateContextPropertiesAndCustomProperties) {
  const PutContextTypeRequest put_context_type_request =
      ParseTextProtoOrDie<PutContextTypeRequest>(
          R"pb(
            all_fields_match: true
            context_type: {
              name: 'test_type'
              properties { key: 'property_str' value: STRING }
            }
          )pb");
  PutContextTypeResponse put_context_type_response;
  ASSERT_EQ(metadata_store_->PutContextType(put_context_type_request,
                                            &put_context_type_response),
            absl::OkStatus());
  ASSERT_TRUE(put_context_type_response.has_type_id());
  const int64_t type_id = put_context_type_response.type_id();

  PutContextsRequest put_contexts_request =
      ParseTextProtoOrDie<PutContextsRequest>(R"pb(
        contexts: {
          properties {
            key: 'property_str'
            value: { string_value: '1' }
          }
          custom_properties {
            key: 'custom_property_str_to_int'
            value: { string_value: '1' }
          }
          custom_properties {
            key: 'custom_property_int_to_str'
            value: { int_value: 1 }
          }
          custom_properties {
            key: 'custom_property_double_to_bool'
            value: { double_value: 1.2345 }
          }
          custom_properties {
            key: 'custom_property_bool_to_double'
            value: { bool_value: true }
          }
          name: 'context_name'
        }
      )pb");
  put_contexts_request.mutable_contexts(0)->set_type_id(type_id);

  PutContextsResponse put_contexts_response;
  {
    // Test 1: update custom_properties.
    ASSERT_EQ(metadata_store_->PutContexts(put_contexts_request,
                                           &put_contexts_response),
              absl::OkStatus());
    ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(1));
    const int64_t context_id1 = put_contexts_response.context_ids(0);

    PutContextsRequest update_contexts_request =
        ParseTextProtoOrDie<PutContextsRequest>(R"pb(
          contexts: {
            properties {
              key: 'property_str'
              value: { string_value: '1' }
            }
            custom_properties {
              key: 'custom_property_str_to_int'
              value: { int_value: 1 }
            }
            custom_properties {
              key: 'custom_property_int_to_str'
              value: { string_value: '1' }
            }
            custom_properties {
              key: 'custom_property_double_to_bool'
              value: { bool_value: true }
            }
            custom_properties {
              key: 'custom_property_bool_to_double'
              value: { double_value: 1.2345 }
            }
            name: 'context_name'
          }
        )pb");
    update_contexts_request.mutable_contexts(0)->set_type_id(type_id);
    update_contexts_request.mutable_contexts(0)->set_id(context_id1);

    PutContextsResponse update_contexts_response;
    ASSERT_EQ(metadata_store_->PutContexts(update_contexts_request,
                                           &update_contexts_response),
              absl::OkStatus());

    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(context_id1);

    GetContextsByIDResponse get_contexts_by_id_response;
    ASSERT_EQ(metadata_store_->GetContextsByID(get_contexts_by_id_request,
                                               &get_contexts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_contexts_by_id_response.contexts(), SizeIs(1));

    EXPECT_THAT(get_contexts_by_id_response.contexts(),
                ElementsAre(EqualsProto(
                    update_contexts_request.contexts(0),
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  {
    // Test 2: update custom_properties under masking.
    put_contexts_request.mutable_contexts(0)->set_name("context_name2");
    ASSERT_EQ(metadata_store_->PutContexts(put_contexts_request,
                                           &put_contexts_response),
              absl::OkStatus());
    ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(1));
    const int64_t context_id2 = put_contexts_response.context_ids(0);

    PutContextsRequest update_contexts_request =
        ParseTextProtoOrDie<PutContextsRequest>(R"pb(
          contexts: {
            custom_properties {
              key: 'custom_property_str_to_int'
              value: { int_value: 1 }
            }
            custom_properties {
              key: 'custom_property_int_to_str'
              value: { string_value: '1' }
            }
            custom_properties {
              key: 'custom_property_double_to_bool'
              value: { bool_value: true }
            }
            custom_properties {
              key: 'custom_property_bool_to_double'
              value: { double_value: 1.2345 }
            }
          }
          update_mask: {
            paths: 'custom_properties.custom_property_str_to_int'
            paths: 'custom_properties.custom_property_int_to_str'
          }
        )pb");
    update_contexts_request.mutable_contexts(0)->set_type_id(type_id);
    update_contexts_request.mutable_contexts(0)->set_id(context_id2);

    PutContextsResponse update_contexts_response;
    ASSERT_EQ(metadata_store_->PutContexts(update_contexts_request,
                                           &update_contexts_response),
              absl::OkStatus());

    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(context_id2);

    GetContextsByIDResponse get_contexts_by_id_response;
    ASSERT_EQ(metadata_store_->GetContextsByID(get_contexts_by_id_request,
                                               &get_contexts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_contexts_by_id_response.contexts(), SizeIs(1));

    Context wanted_context = ParseTextProtoOrDie<Context>(R"pb(
      properties {
        key: 'property_str'
        value: { string_value: '1' }
      }
      custom_properties {
        key: 'custom_property_str_to_int'
        value: { int_value: 1 }
      }
      custom_properties {
        key: 'custom_property_int_to_str'
        value: { string_value: '1' }
      }
      custom_properties {
        key: 'custom_property_double_to_bool'
        value: { double_value: 1.2345 }
      }
      custom_properties {
        key: 'custom_property_bool_to_double'
        value: { bool_value: true }
      }
      name: 'context_name2'
    )pb");

    EXPECT_THAT(get_contexts_by_id_response.contexts(),
                ElementsAre(EqualsProto(
                    wanted_context,
                    /*ignore_fields=*/{"id", "type_id", "type",
                                       "create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  {
    // Test 3: updating property's value_type fails.
    put_contexts_request.mutable_contexts(0)->set_name("context_name3");
    ASSERT_EQ(metadata_store_->PutContexts(put_contexts_request,
                                           &put_contexts_response),
              absl::OkStatus());
    ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(1));
    const int64_t context_id3 = put_contexts_response.context_ids(0);

    PutContextsRequest update_contexts_request =
        ParseTextProtoOrDie<PutContextsRequest>(R"pb(
          contexts: {
            properties {
              key: 'property_str'
              value: { int_value: 1 }
            }
          }
          update_mask: { paths: 'properties.property_str' }
        )pb");
    PutContextsResponse update_contexts_response;
    update_contexts_request.mutable_contexts(0)->set_type_id(type_id);
    update_contexts_request.mutable_contexts(0)->set_id(context_id3);
    EXPECT_TRUE(absl::IsInvalidArgument(metadata_store_->PutContexts(
        update_contexts_request, &update_contexts_response)));
    EXPECT_TRUE(absl::StrContains(
        metadata_store_
            ->PutContexts(update_contexts_request, &update_contexts_response)
            .message(),
        "unmatched property type"));
  }
}

// Test creating an context and then updating one of its properties.
TEST_P(MetadataStoreTestSuite, UpdateContextWithMasking) {
  const PutContextTypeRequest put_context_type_request =
      ParseTextProtoOrDie<PutContextTypeRequest>(
          R"pb(
            all_fields_match: true
            context_type: {
              name: 'test_type'
              properties { key: 'property' value: STRING }
            }
          )pb");
  PutContextTypeResponse put_context_type_response;
  ASSERT_EQ(metadata_store_->PutContextType(put_context_type_request,
                                            &put_context_type_response),
            absl::OkStatus());
  ASSERT_TRUE(put_context_type_response.has_type_id());
  const int64_t type_id = put_context_type_response.type_id();

  // Add two contexts, one with a `properties` pair <'property': '3'>, one
  // without.
  PutContextsRequest put_contexts_request =
      ParseTextProtoOrDie<PutContextsRequest>(R"pb(
        contexts: {
          properties {
            key: 'property'
            value: { string_value: '1' }
          }
          name: 'context_name_old_1'
        }
        contexts: { name: 'context_name_old_2' }
      )pb");
  put_contexts_request.mutable_contexts(0)->set_type_id(type_id);
  put_contexts_request.mutable_contexts(1)->set_type_id(type_id);
  PutContextsResponse put_contexts_response;
  {
    // Test 1: a complex test case for updating fields and properties for both
    // contexts.
    ASSERT_EQ(metadata_store_->PutContexts(put_contexts_request,
                                           &put_contexts_response),
              absl::OkStatus());
    ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(2));
    const int64_t context_id1 = put_contexts_response.context_ids(0);
    const int64_t context_id2 = put_contexts_response.context_ids(1);
    // Add `name` for both contexts.
    // Change string value of key `property` from '3' to '1' in the first
    // context.
    // Add `properties` pair <'property': '2'> in the second context.
    PutContextsRequest update_contexts_request =
        ParseTextProtoOrDie<PutContextsRequest>(R"pb(
          contexts: {
            properties {
              key: 'property'
              value: { string_value: '1' }
            }
            name: 'context_1'
          }
          contexts: {
            properties {
              key: 'property'
              value: { string_value: '2' }
            }
            name: 'context_2'
          }
          update_mask: {
            paths: 'properties.property'
            paths: 'name'
            paths: 'an_invalid_field_path_having_no_effect'
          }
        )pb");
    update_contexts_request.mutable_contexts(0)->set_type_id(type_id);
    update_contexts_request.mutable_contexts(0)->set_id(context_id1);
    update_contexts_request.mutable_contexts(1)->set_type_id(type_id);
    update_contexts_request.mutable_contexts(1)->set_id(context_id2);
    PutContextsResponse update_contexts_response;
    ASSERT_EQ(metadata_store_->PutContexts(update_contexts_request,
                                           &update_contexts_response),
              absl::OkStatus());

    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(context_id1);
    get_contexts_by_id_request.add_context_ids(context_id2);

    GetContextsByIDResponse get_contexts_by_id_response;
    ASSERT_EQ(metadata_store_->GetContextsByID(get_contexts_by_id_request,
                                               &get_contexts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_contexts_by_id_response.contexts(), SizeIs(2));

    EXPECT_THAT(
        get_contexts_by_id_response.contexts(),
        UnorderedElementsAre(
            EqualsProto(update_contexts_request.contexts(0),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(update_contexts_request.contexts(1),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  {
    // Test 2: insert two new contexts and update fields for both contexts.
    ASSERT_EQ(metadata_store_->PutContexts(put_contexts_request,
                                           &put_contexts_response),
              absl::OkStatus());
    ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(2));
    const int64_t context_id3 = put_contexts_response.context_ids(0);
    const int64_t context_id4 = put_contexts_response.context_ids(1);
    // Set `external_id` and `name` for both contexts.
    // `properties` for both contexts will remain unchanged.
    PutContextsRequest update_contexts_request =
        ParseTextProtoOrDie<PutContextsRequest>(R"pb(
          contexts: { external_id: 'context_3' name: 'context_3' }
          contexts: { external_id: 'context_4' name: 'context_4' }
          update_mask: { paths: 'external_id' paths: 'name' }
        )pb");
    update_contexts_request.mutable_contexts(0)->set_type_id(type_id);
    update_contexts_request.mutable_contexts(0)->set_id(context_id3);
    update_contexts_request.mutable_contexts(1)->set_type_id(type_id);
    update_contexts_request.mutable_contexts(1)->set_id(context_id4);
    PutContextsResponse update_contexts_response;
    ASSERT_EQ(metadata_store_->PutContexts(update_contexts_request,
                                           &update_contexts_response),
              absl::OkStatus());

    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(context_id3);
    get_contexts_by_id_request.add_context_ids(context_id4);

    GetContextsByIDResponse get_contexts_by_id_response;
    ASSERT_EQ(metadata_store_->GetContextsByID(get_contexts_by_id_request,
                                               &get_contexts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_contexts_by_id_response.contexts(), SizeIs(2));

    EXPECT_THAT(
        get_contexts_by_id_response.contexts(),
        UnorderedElementsAre(
            EqualsProto(update_contexts_request.contexts(0),
                        /*ignore_fields=*/{"type", "properties",
                                           "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(update_contexts_request.contexts(1),
                        /*ignore_fields=*/{"type", "properties",
                                           "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  Context context_for_test_3_4_and_5;
  {
    // Test 3: insert two new contexts and update `properties` and
    // `custom_properties` for both contexts.
    ASSERT_EQ(metadata_store_->PutContexts(put_contexts_request,
                                           &put_contexts_response),
              absl::OkStatus());
    ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(2));
    const int64_t context_id5 = put_contexts_response.context_ids(0);
    const int64_t context_id6 = put_contexts_response.context_ids(1);
    // Delete `properties` pair <'property': '3'> in the first context.
    // Add `custom_properties` pair <'custom_property': true> for `context_5`.
    // Add `custom_properties` pair <'custom_property': false> for `context_6`.
    // The `name` for both contexts will remain unchanged.
    PutContextsRequest update_contexts_request =
        ParseTextProtoOrDie<PutContextsRequest>(R"pb(
          contexts: {
            custom_properties {
              key: 'custom_property'
              value: { bool_value: true }
            }
            name: 'context_name_old_1'
          }
          contexts: {
            custom_properties {
              key: 'custom_property'
              value: { bool_value: false }
            }
            name: 'context_name_old_2'
          }
          update_mask: {
            paths: 'properties.property'
            paths: 'custom_properties.custom_property'
          }
        )pb");
    update_contexts_request.mutable_contexts(0)->set_type_id(type_id);
    update_contexts_request.mutable_contexts(0)->set_id(context_id5);
    update_contexts_request.mutable_contexts(1)->set_type_id(type_id);
    update_contexts_request.mutable_contexts(1)->set_id(context_id6);
    PutContextsResponse update_contexts_response;
    ASSERT_EQ(metadata_store_->PutContexts(update_contexts_request,
                                           &update_contexts_response),
              absl::OkStatus());
    context_for_test_3_4_and_5 = update_contexts_request.contexts(1);

    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(context_id5);
    get_contexts_by_id_request.add_context_ids(context_id6);

    GetContextsByIDResponse get_contexts_by_id_response;
    ASSERT_EQ(metadata_store_->GetContextsByID(get_contexts_by_id_request,
                                               &get_contexts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_contexts_by_id_response.contexts(), SizeIs(2));

    EXPECT_THAT(
        get_contexts_by_id_response.contexts(),
        UnorderedElementsAre(
            EqualsProto(update_contexts_request.contexts(0),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(update_contexts_request.contexts(1),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  {
    // Test 4: insert a new context and update an existing context in the
    // same request under masking. The mask is expected to have no effect on
    // insertion but to protect fields for update.
    PutContextsRequest upsert_contexts_request =
        ParseTextProtoOrDie<PutContextsRequest>(R"pb(
          contexts: {
            external_id: 'context_6'
            custom_properties {
              key: 'custom_property'
              value: { bool_value: true }
            }
          }
          contexts: {
            name: 'context_7'
            external_id: 'context_7'
            custom_properties {
              key: 'custom_property'
              value: { bool_value: true }
            }
          }
          update_mask: { paths: 'external_id' }
        )pb");
    upsert_contexts_request.mutable_contexts(0)->set_type_id(type_id);
    upsert_contexts_request.mutable_contexts(0)->set_id(
        context_for_test_3_4_and_5.id());
    upsert_contexts_request.mutable_contexts(1)->set_type_id(type_id);
    PutContextsResponse upsert_contexts_response;
    ASSERT_EQ(metadata_store_->PutContexts(upsert_contexts_request,
                                           &upsert_contexts_response),
              absl::OkStatus());
    const int64_t context_id7 = upsert_contexts_response.context_ids(1);

    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(context_for_test_3_4_and_5.id());
    get_contexts_by_id_request.add_context_ids(context_id7);

    GetContextsByIDResponse get_contexts_by_id_response;
    ASSERT_EQ(metadata_store_->GetContextsByID(get_contexts_by_id_request,
                                               &get_contexts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_contexts_by_id_response.contexts(), SizeIs(2));

    // If put update is successful, one of the obtained contexts should be the
    // updated context, one of the obtained contexts should be the inserted
    // context.
    context_for_test_3_4_and_5.set_external_id("context_6");
    upsert_contexts_request.mutable_contexts(1)->set_id(context_id7);
    EXPECT_THAT(
        get_contexts_by_id_response.contexts(),
        UnorderedElementsAre(
            EqualsProto(context_for_test_3_4_and_5,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(upsert_contexts_request.contexts(1),
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }
  {
    // Test 5: updating an context with a mask containing only invalid mask
    // paths has no effect.
    PutContextsRequest upsert_contexts_request =
        ParseTextProtoOrDie<PutContextsRequest>(R"pb(
          contexts: {
            external_id: 'unimportant_exeternal_id_value'
            custom_properties {
              key: 'unimportant_property_key'
              value: { bool_value: true }
            }
          }
          update_mask: { paths: 'an_invalid_field_path' }
        )pb");
    upsert_contexts_request.mutable_contexts(0)->set_type_id(type_id);
    upsert_contexts_request.mutable_contexts(0)->set_id(
        context_for_test_3_4_and_5.id());
    PutContextsResponse upsert_contexts_response;
    ASSERT_EQ(metadata_store_->PutContexts(upsert_contexts_request,
                                           &upsert_contexts_response),
              absl::OkStatus());

    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(context_for_test_3_4_and_5.id());

    GetContextsByIDResponse get_contexts_by_id_response;
    ASSERT_EQ(metadata_store_->GetContextsByID(get_contexts_by_id_request,
                                               &get_contexts_by_id_response),
              absl::OkStatus());
    ASSERT_THAT(get_contexts_by_id_response.contexts(), SizeIs(1));

    EXPECT_THAT(
        get_contexts_by_id_response.contexts(0),
        EqualsProto(
            context_for_test_3_4_and_5,
            /*ignore_fields=*/{"type", "external_id", "create_time_since_epoch",
                               "last_update_time_since_epoch"}));
  }
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContextType(put_context_type_request,
                                            &put_context_type_response));
  ASSERT_TRUE(put_context_type_response.has_type_id());

  const int64_t type_id = put_context_type_response.type_id();

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

  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request,
                                         &put_contexts_response));
  ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(2));
  const int64_t context_id_0 = put_contexts_response.context_ids(0);
  const int64_t context_id_1 = put_contexts_response.context_ids(1);

  ListOperationOptions list_options =
      ParseTextProtoOrDie<ListOperationOptions>(R"(
        max_result_size: 1,
        order_by_field: { field: CREATE_TIME is_asc: false }
      )");

  GetContextsRequest get_contexts_request;
  *get_contexts_request.mutable_options() = list_options;

  GetContextsResponse get_contexts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetContexts(get_contexts_request,
                                         &get_contexts_response));
  EXPECT_THAT(get_contexts_response.contexts(), SizeIs(1));
  EXPECT_THAT(get_contexts_response.next_page_token(), Not(IsEmpty()));
  EXPECT_EQ(get_contexts_response.contexts(0).id(), context_id_1);

  EXPECT_THAT(
      get_contexts_response.contexts(0),
      EqualsProto(put_contexts_request.contexts(1),
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}));

  list_options.set_next_page_token(get_contexts_response.next_page_token());
  *get_contexts_request.mutable_options() = list_options;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetContexts(get_contexts_request,
                                         &get_contexts_response));
  EXPECT_THAT(get_contexts_response.contexts(), SizeIs(1));
  EXPECT_THAT(get_contexts_response.next_page_token(), IsEmpty());
  EXPECT_EQ(get_contexts_response.contexts(0).id(), context_id_0);
  EXPECT_THAT(
      get_contexts_response.contexts(0),
      EqualsProto(put_contexts_request.contexts(0),
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContextType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());

  // Put the same request again, the upsert returns the same id
  {
    const PutContextTypeRequest same_put_request = put_request;
    PutContextTypeResponse same_put_response;
    ASSERT_EQ(absl::OkStatus(), metadata_store_->PutContextType(
                                    same_put_request, &same_put_response));
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
    ASSERT_EQ(absl::OkStatus(), metadata_store_->PutContextType(
                                    add_property_put_request, &response));
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

TEST_P(MetadataStoreTestSuite, PutContextsGetContextsByExternalIds) {
  int64_t type_id;
  // Create the type
  {
    const PutContextTypeRequest put_context_type_request =
        ParseTextProtoOrDie<PutContextTypeRequest>(
            R"pb(
              all_fields_match: true
              context_type: {
                name: 'test_context_type'
                properties { key: 'property' value: STRING }
              }
            )pb");
    PutContextTypeResponse put_context_type_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutContextType(put_context_type_request,
                                              &put_context_type_response));
    ASSERT_TRUE(put_context_type_response.has_type_id());

    type_id = put_context_type_response.type_id();
  }

  // Put in two contexts
  constexpr absl::string_view kContextTemplate = R"(
          type_id: %d
          name: '%s'
          properties {
            key: 'property'
            value: { string_value: '%s' }
          }
          external_id: '%s'
      )";
  Context context1 = ParseTextProtoOrDie<Context>(absl::StrFormat(
      kContextTemplate, type_id, "context_1", "1", "context_external_id_1"));
  Context context2 = ParseTextProtoOrDie<Context>(absl::StrFormat(
      kContextTemplate, type_id, "context_2", "2", "context_external_id_2"));

  {
    PutContextsRequest put_contexts_request;
    *put_contexts_request.mutable_contexts()->Add() = context1;
    *put_contexts_request.mutable_contexts()->Add() = context2;
    PutContextsResponse put_contexts_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->PutContexts(put_contexts_request,
                                           &put_contexts_response));
    ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(2));
    context1.set_id(put_contexts_response.context_ids(0));
    context2.set_id(put_contexts_response.context_ids(1));
  }

  // Test: retrieve by one external id
  {
    GetContextsByExternalIdsRequest get_contexts_by_external_ids_request;
    get_contexts_by_external_ids_request.add_external_ids(
        context1.external_id());
    GetContextsByExternalIdsResponse get_contexts_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(), metadata_store_->GetContextsByExternalIds(
                                    get_contexts_by_external_ids_request,
                                    &get_contexts_by_external_ids_response));
    EXPECT_THAT(get_contexts_by_external_ids_response.contexts(),
                ElementsAre(EqualsProto(
                    context1,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  // Test: retrieve by one non-existing external id
  {
    GetContextsByExternalIdsRequest get_contexts_by_external_ids_request;
    get_contexts_by_external_ids_request.add_external_ids(
        "context_absent_external_id");
    GetContextsByExternalIdsResponse get_contexts_by_external_ids_response;
    EXPECT_TRUE(absl::IsNotFound(metadata_store_->GetContextsByExternalIds(
        get_contexts_by_external_ids_request,
        &get_contexts_by_external_ids_response)));
  }
  // Test: retrieve by multiple external ids
  {
    GetContextsByExternalIdsRequest get_contexts_by_external_ids_request;

    // Can retrieve Contexts by multiple external ids
    get_contexts_by_external_ids_request.add_external_ids(
        context1.external_id());
    get_contexts_by_external_ids_request.add_external_ids(
        context2.external_id());
    GetContextsByExternalIdsResponse get_contexts_by_external_ids_response;
    EXPECT_EQ(absl::OkStatus(), metadata_store_->GetContextsByExternalIds(
                                    get_contexts_by_external_ids_request,
                                    &get_contexts_by_external_ids_response));
    EXPECT_THAT(
        get_contexts_by_external_ids_response.contexts(),
        UnorderedElementsAre(
            EqualsProto(context1,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(context2,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));

    // Will return whatever found if some of the external ids is absent
    get_contexts_by_external_ids_request.add_external_ids(
        "context_absent_external_id");
    EXPECT_EQ(absl::OkStatus(), metadata_store_->GetContextsByExternalIds(
                                    get_contexts_by_external_ids_request,
                                    &get_contexts_by_external_ids_response));
    EXPECT_THAT(
        get_contexts_by_external_ids_response.contexts(),
        UnorderedElementsAre(
            EqualsProto(context1,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(context2,
                        /*ignore_fields=*/{"type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
  }

  // Test retrieve by empty external id
  {
    GetContextsByExternalIdsRequest get_contexts_by_external_ids_request;
    get_contexts_by_external_ids_request.add_external_ids("");
    GetContextsByExternalIdsResponse get_contexts_by_external_ids_response;
    EXPECT_TRUE(
        absl::IsInvalidArgument(metadata_store_->GetContextsByExternalIds(
            get_contexts_by_external_ids_request,
            &get_contexts_by_external_ids_response)));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContextType(put_context_type_request,
                                            &put_context_type_response));
  ASSERT_TRUE(put_context_type_response.has_type_id());
  const int64_t type_id = put_context_type_response.type_id();

  ContextType type2;
  type2.set_name("type2_name");
  PutContextTypeRequest put_context_type_request2;
  put_context_type_request2.set_all_fields_match(true);
  *put_context_type_request2.mutable_context_type() = type2;
  PutContextTypeResponse put_context_type_response2;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContextType(put_context_type_request2,
                                            &put_context_type_response2));
  ASSERT_TRUE(put_context_type_response2.has_type_id());
  const int64_t type2_id = put_context_type_response2.type_id();

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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request,
                                         &put_contexts_response));
  ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(2));
  const int64_t id1 = put_contexts_response.context_ids(0);
  const int64_t id2 = put_contexts_response.context_ids(1);

  // Now we update context1's string value from 1 to 2.
  // and context2's int value from 2 to 3, and add a new context with type2.
  Context want_context1 = *put_contexts_request.mutable_contexts(0);
  want_context1.set_id(id1);
  want_context1.set_type(put_context_type_request.context_type().name());
  (*want_context1.mutable_properties())["property"].set_string_value("2");
  Context want_context2 = *put_contexts_request.mutable_contexts(1);
  want_context2.set_id(id2);
  want_context2.set_type(put_context_type_request.context_type().name());
  (*want_context2.mutable_custom_properties())["custom"].set_int_value(2);
  Context want_context3;
  want_context3.set_type_id(type2_id);
  want_context3.set_name("context3");
  want_context3.set_type(put_context_type_request2.context_type().name());

  PutContextsRequest put_contexts_request2;
  *put_contexts_request2.add_contexts() = want_context1;
  *put_contexts_request2.add_contexts() = want_context2;
  *put_contexts_request2.add_contexts() = want_context3;
  PutContextsResponse put_contexts_response2;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request2,
                                         &put_contexts_response2));
  ASSERT_THAT(put_contexts_response2.context_ids(), SizeIs(3));
  want_context3.set_id(put_contexts_response2.context_ids(2));

  // Test: GetContextsByID: one id
  {
    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(id1);
    GetContextsByIDResponse get_contexts_by_id_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetContextsByID(get_contexts_by_id_request,
                                               &get_contexts_by_id_response));
    EXPECT_THAT(get_contexts_by_id_response.contexts(),
                ElementsAre(EqualsProto(
                    want_context1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
  }
  // Test: GetContextsByID: many ids + unknown id
  const int64_t unknown_id =
      want_context1.id() + want_context2.id() + want_context3.id() + 1;
  {
    GetContextsByIDRequest get_contexts_by_id_request;
    get_contexts_by_id_request.add_context_ids(id1);
    get_contexts_by_id_request.add_context_ids(id2);
    get_contexts_by_id_request.add_context_ids(unknown_id);
    GetContextsByIDResponse get_contexts_by_id_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetContextsByID(get_contexts_by_id_request,
                                               &get_contexts_by_id_response));
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
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetContextsByID(get_contexts_by_id_request,
                                               &get_contexts_by_id_response));
    EXPECT_THAT(get_contexts_by_id_response.contexts(), IsEmpty());
  }
  // Test: GetContextsByType
  {
    GetContextsByTypeRequest get_contexts_by_type_request;
    get_contexts_by_type_request.set_type_name("type2_name");
    GetContextsByTypeResponse get_contexts_by_type_response;
    ASSERT_EQ(absl::OkStatus(), metadata_store_->GetContextsByType(
                                    get_contexts_by_type_request,
                                    &get_contexts_by_type_response));
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
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetContextsByType(request, &response));
    EXPECT_THAT(response.contexts(),
                ElementsAre(EqualsProto(
                    want_context1,
                    /*ignore_fields=*/{"create_time_since_epoch",
                                       "last_update_time_since_epoch"})));
    ASSERT_THAT(response.next_page_token(), Not(IsEmpty()));
    request.mutable_options()->set_next_page_token(response.next_page_token());
    response.Clear();
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetContextsByType(request, &response));
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
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetContexts(get_contexts_request,
                                           &get_contexts_response));
    ASSERT_THAT(get_contexts_response.contexts(), SizeIs(3));
    EXPECT_THAT(
        get_contexts_response.contexts(),
        UnorderedElementsAre(

            EqualsProto(want_context1,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(want_context2,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
            EqualsProto(want_context3,
                        /*ignore_fields=*/{"create_time_since_epoch",
                                           "last_update_time_since_epoch"})));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutTypes(put_types_request, &put_types_response));
  int64_t artifact_type_id = put_types_response.artifact_type_ids(0);
  int64_t execution_type_id = put_types_response.execution_type_ids(0);

  const PutContextTypeRequest put_context_type_request =
      ParseTextProtoOrDie<PutContextTypeRequest>(R"(
        all_fields_match: true
        context_type: { name: 'context_type' }
      )");
  PutContextTypeResponse put_context_type_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContextType(put_context_type_request,
                                            &put_context_type_response));
  int64_t context_type_id = put_context_type_response.type_id();

  Execution want_execution;
  want_execution.set_type_id(execution_type_id);
  (*want_execution.mutable_properties())["property"].set_string_value("1");
  PutExecutionsRequest put_executions_request;
  *put_executions_request.add_executions() = want_execution;
  PutExecutionsResponse put_executions_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecutions(put_executions_request,
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request,
                                          &put_artifacts_response));
  ASSERT_THAT(put_artifacts_response.artifact_ids(), SizeIs(1));
  want_artifact.set_id(put_artifacts_response.artifact_ids(0));

  Context want_context;
  want_context.set_name("context");
  want_context.set_type_id(context_type_id);
  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = want_context;
  PutContextsResponse put_contexts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request,
                                         &put_contexts_response));
  ASSERT_THAT(put_contexts_response.context_ids(), SizeIs(1));
  want_context.set_id(put_contexts_response.context_ids(0));

  // insert an attribution
  PutAttributionsAndAssociationsRequest request;
  Attribution* attribution = request.add_attributions();
  attribution->set_artifact_id(want_artifact.id());
  attribution->set_context_id(want_context.id());
  PutAttributionsAndAssociationsResponse response;
  EXPECT_EQ(absl::OkStatus(), metadata_store_->PutAttributionsAndAssociations(
                                  request, &response));

  GetContextsByArtifactRequest get_contexts_by_artifact_request;
  get_contexts_by_artifact_request.set_artifact_id(want_artifact.id());
  GetContextsByArtifactResponse get_contexts_by_artifact_response;
  EXPECT_EQ(absl::OkStatus(), metadata_store_->GetContextsByArtifact(
                                  get_contexts_by_artifact_request,
                                  &get_contexts_by_artifact_response));
  ASSERT_THAT(get_contexts_by_artifact_response.contexts(), SizeIs(1));
  EXPECT_THAT(get_contexts_by_artifact_response.contexts(0),
              EqualsProto(want_context,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  GetArtifactsByContextRequest get_artifacts_by_context_request;
  get_artifacts_by_context_request.set_context_id(want_context.id());
  GetArtifactsByContextResponse get_artifacts_by_context_response;
  EXPECT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByContext(
                                  get_artifacts_by_context_request,
                                  &get_artifacts_by_context_response));
  ASSERT_THAT(get_artifacts_by_context_response.artifacts(), SizeIs(1));
  EXPECT_THAT(get_artifacts_by_context_response.artifacts(0),
              EqualsProto(want_artifact,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  // append the association and reinsert the existing attribution.
  Association* association = request.add_associations();
  association->set_execution_id(want_execution.id());
  association->set_context_id(want_context.id());
  ASSERT_EQ(absl::OkStatus(), metadata_store_->PutAttributionsAndAssociations(
                                  request, &response));

  GetContextsByExecutionRequest get_contexts_by_execution_request;
  get_contexts_by_execution_request.set_execution_id(want_execution.id());
  GetContextsByExecutionResponse get_contexts_by_execution_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetContextsByExecution(
                                  get_contexts_by_execution_request,
                                  &get_contexts_by_execution_response));
  ASSERT_THAT(get_contexts_by_execution_response.contexts(), SizeIs(1));
  EXPECT_THAT(get_contexts_by_execution_response.contexts(0),
              EqualsProto(want_context,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  GetExecutionsByContextRequest get_executions_by_context_request;
  get_executions_by_context_request.set_context_id(want_context.id());
  GetExecutionsByContextResponse get_executions_by_context_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByContext(
                                  get_executions_by_context_request,
                                  &get_executions_by_context_response));
  ASSERT_THAT(get_executions_by_context_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_by_context_response.executions(0),
              EqualsProto(want_execution,
                          /*ignore_fields=*/{"type", "create_time_since_epoch",
                                             "last_update_time_since_epoch"}));

  // Add another artifact.
  Artifact want_artifact_2;
  want_artifact_2.set_uri("testuri2");
  want_artifact_2.set_type_id(artifact_type_id);
  PutArtifactsRequest put_artifacts_request_2;
  *put_artifacts_request_2.add_artifacts() = want_artifact_2;
  PutArtifactsResponse put_artifacts_response_2;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request_2,
                                          &put_artifacts_response_2));
  ASSERT_THAT(put_artifacts_response_2.artifact_ids(), SizeIs(1));
  want_artifact_2.set_id(put_artifacts_response_2.artifact_ids(0));

  // Reinsert association and append another attribution for the new Artifact.
  // Notice that the request now contains one existing association, one existing
  // attribution and a new attribution.
  Attribution* attribution_2 = request.add_attributions();
  attribution_2->set_artifact_id(want_artifact_2.id());
  attribution_2->set_context_id(want_context.id());
  EXPECT_EQ(absl::OkStatus(), metadata_store_->PutAttributionsAndAssociations(
                                  request, &response));

  // The new Artifact can also be retrieved.
  GetArtifactsByContextResponse get_artifacts_by_context_response_2;
  EXPECT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByContext(
                                  get_artifacts_by_context_request,
                                  &get_artifacts_by_context_response_2));
  ASSERT_THAT(get_artifacts_by_context_response_2.artifacts(), SizeIs(2));
  EXPECT_THAT(
      get_artifacts_by_context_response_2.artifacts(),
      UnorderedElementsAre(
          EqualsProto(want_artifact,
                      /*ignore_fields=*/{"type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"}),
          EqualsProto(want_artifact_2,
                      /*ignore_fields=*/{"type", "create_time_since_epoch",
                                         "last_update_time_since_epoch"})));
}

TEST_P(MetadataStoreTestSuite, PutParentContextsAlreadyExistsError) {
  // Inserts a context type.
  ContextType context_type;
  context_type.set_name("context_type_name");
  PutContextTypeRequest put_type_request;
  *put_type_request.mutable_context_type() = context_type;
  PutContextTypeResponse put_type_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->PutContextType(
                                  put_type_request, &put_type_response));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request,
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
  EXPECT_EQ(absl::OkStatus(),
            metadata_store_->PutParentContexts(put_parent_contexts_request,
                                               &put_parent_contexts_response));

  // Recreates the same parent context should returns AlreadyExists error.
  const absl::Status status = metadata_store_->PutParentContexts(
      put_parent_contexts_request, &put_parent_contexts_response);
  EXPECT_TRUE(absl::IsAlreadyExists(status));
}

TEST_P(MetadataStoreTestSuite, PutParentContextsInvalidArgumentError) {
  // Inserts a context type.
  ContextType context_type;
  context_type.set_name("context_type_name");
  PutContextTypeRequest put_type_request;
  *put_type_request.mutable_context_type() = context_type;
  PutContextTypeResponse put_type_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->PutContextType(
                                  put_type_request, &put_type_response));
  context_type.set_id(put_type_response.type_id());

  // Creates two not exist context ids.
  Context stored_context;
  stored_context.set_name("stored_context");
  stored_context.set_type_id(context_type.id());
  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = stored_context;
  PutContextsResponse put_contexts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request,
                                         &put_contexts_response));
  int stored_context_id = put_contexts_response.context_ids(0);
  int64_t not_exist_context_id = stored_context_id + 1;
  int64_t not_exist_context_id_2 = stored_context_id + 2;

  // Enumerates the case of creating parent context with invalid argument
  // (context id cannot be found in the database).
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
    PutParentContextsRequest put_parent_contexts_request;
    *put_parent_contexts_request.add_parent_contexts() = parent_context;
    PutParentContextsResponse put_parent_contexts_response;
    const absl::Status status = metadata_store_->PutParentContexts(
        put_parent_contexts_request, &put_parent_contexts_response);
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
  ASSERT_EQ(absl::OkStatus(), metadata_store_->PutContextType(
                                  put_type_request, &put_type_response));
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request,
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
                             &put_parent_contexts_request](int64_t parent_idx,
                                                           int64_t child_idx) {
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
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutParentContexts(put_parent_contexts_request,
                                               &put_parent_contexts_response));

  // Verifies the parent contexts by looking up and stored result.
  for (int i = 0; i < num_contexts; i++) {
    GetParentContextsByContextRequest get_parents_request;
    get_parents_request.set_context_id(contexts[i].id());
    GetParentContextsByContextResponse get_parents_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetParentContextsByContext(
                  get_parents_request, &get_parents_response));
    GetChildrenContextsByContextRequest get_children_request;
    get_children_request.set_context_id(contexts[i].id());
    GetChildrenContextsByContextResponse get_children_response;
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetChildrenContextsByContext(
                  get_children_request, &get_children_response));
    EXPECT_THAT(get_parents_response.contexts(),
                SizeIs(want_parents[i].size()));
    EXPECT_THAT(get_children_response.contexts(),
                SizeIs(want_children[i].size()));
    EXPECT_THAT(get_parents_response.contexts(),
                UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                                       "type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}),
                                   want_parents[i]));
    EXPECT_THAT(get_children_response.contexts(),
                UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                                       "type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}),
                                   want_children[i]));
  }
}

TEST_P(MetadataStoreTestSuite,
       PutParentContextsAndGetLinkedContextsByContexts) {
  // Inserts a context type.
  ContextType context_type;
  context_type.set_name("context_type_name");
  PutContextTypeRequest put_type_request;
  *put_type_request.mutable_context_type() = context_type;
  PutContextTypeResponse put_type_response;
  ASSERT_EQ(
      metadata_store_->PutContextType(put_type_request, &put_type_response),
      absl::OkStatus());
  context_type.set_id(put_type_response.type_id());

  // Creates some contexts to be inserted into the later parent context
  // relationship.
  const int num_contexts = 7;
  std::vector<Context> contexts(num_contexts);
  std::vector<int64_t> context_ids(num_contexts);
  PutContextsRequest put_contexts_request;
  for (int i = 0; i < num_contexts; i++) {
    contexts[i].set_name(absl::StrCat("context_", i));
    contexts[i].set_type_id(context_type.id());
    *put_contexts_request.add_contexts() = contexts[i];
  }
  PutContextsResponse put_contexts_response;
  ASSERT_EQ(metadata_store_->PutContexts(put_contexts_request,
                                         &put_contexts_response),
            absl::OkStatus());
  for (int i = 0; i < num_contexts; i++) {
    contexts[i].set_id(put_contexts_response.context_ids(i));
    context_ids[i] = contexts[i].id();
  }

  // Prepares a list of parent contexts and stores every parent context
  // relationship for each context.
  std::unordered_map<int, std::vector<Context>> want_parents;
  std::unordered_map<int, std::vector<Context>> want_children;
  PutParentContextsRequest put_parent_contexts_request;

  auto put_parent_context = [&](int64_t parent_idx, int64_t child_idx) {
    ParentContext parent_context;
    parent_context.set_parent_id(contexts[parent_idx].id());
    parent_context.set_child_id(contexts[child_idx].id());
    put_parent_contexts_request.add_parent_contexts()->CopyFrom(parent_context);
    want_parents[contexts[child_idx].id()].push_back(contexts[parent_idx]);
    want_children[contexts[parent_idx].id()].push_back(contexts[child_idx]);
  };

  put_parent_context(/*parent_idx=*/0, /*child_idx=*/1);
  put_parent_context(/*parent_idx=*/0, /*child_idx=*/2);
  put_parent_context(/*parent_idx=*/2, /*child_idx=*/3);
  put_parent_context(/*parent_idx=*/1, /*child_idx=*/6);
  put_parent_context(/*parent_idx=*/4, /*child_idx=*/5);
  put_parent_context(/*parent_idx=*/5, /*child_idx=*/6);

  PutParentContextsResponse put_parent_contexts_response;
  ASSERT_EQ(metadata_store_->PutParentContexts(put_parent_contexts_request,
                                               &put_parent_contexts_response),
            absl::OkStatus());

  // Verifies the parent/child contexts by looking up and stored result.
  GetParentContextsByContextsRequest get_parents_request;
  get_parents_request.mutable_context_ids()->Reserve(context_ids.size());
  for (const int64_t context_id : context_ids) {
    get_parents_request.add_context_ids(context_id);
  }
  GetParentContextsByContextsResponse get_parents_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetParentContextsByContexts(
                                  get_parents_request, &get_parents_response));
  GetChildrenContextsByContextsRequest get_children_request;
  get_children_request.mutable_context_ids()->Reserve(context_ids.size());
  for (const int64_t context_id : context_ids) {
    get_children_request.add_context_ids(context_id);
  }
  GetChildrenContextsByContextsResponse get_children_response;
  ASSERT_EQ(metadata_store_->GetChildrenContextsByContexts(
                get_children_request, &get_children_response),
            absl::OkStatus());

  ASSERT_THAT(get_parents_response.contexts(), SizeIs(want_parents.size()));
  for (const int64_t context_id : context_ids) {
    ASSERT_EQ(get_parents_response.contexts().contains(context_id),
              want_parents.count(context_id) > 0);
    if (!get_parents_response.contexts().contains(context_id)) continue;
    EXPECT_THAT(
        get_parents_response.contexts().at(context_id).parent_contexts(),
        UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                               "type", "create_time_since_epoch",
                               "last_update_time_since_epoch"}),
                           want_parents[context_id]));
  }

  ASSERT_THAT(get_children_response.contexts(), SizeIs(want_children.size()));
  for (const int64_t context_id : context_ids) {
    ASSERT_EQ(get_children_response.contexts().contains(context_id),
              want_children.count(context_id) > 0);
    if (!get_children_response.contexts().contains(context_id)) continue;
    EXPECT_THAT(
        get_children_response.contexts().at(context_id).children_contexts(),
        UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                               "type", "create_time_since_epoch",
                               "last_update_time_since_epoch"}),
                           want_children[context_id]));
  }
}


TEST_P(MetadataStoreTestSuite,
       PutTypesAndContextsGetContextsThroughTypeWithOptions) {
  const int kNumNodes = 110;
  std::vector<Context> nodes(kNumNodes);
  ContextType context_type;
  context_type.set_name("test_type");
  InsertTypeAndSetTypeID(metadata_store_, context_type);
  Context context;
  context.set_type_id(context_type.id());
  for (int i = 0; i < kNumNodes; ++i) {
    context.set_name(absl::StrFormat("name_%d", i));
    nodes[i] = context;
  }
  InsertNodeAndSetNodeID(metadata_store_, nodes);

  auto call_get_contexts_by_type =
      [this](GetContextsByTypeRequest get_nodes_request,
             absl::Span<Context> want_contexts,
             GetContextsByTypeResponse& get_nodes_response) {
        ASSERT_EQ(absl::OkStatus(),
                  metadata_store_->GetContextsByType(get_nodes_request,
                                                     &get_nodes_response));
        ASSERT_EQ(want_contexts.size(), get_nodes_response.contexts_size());
        EXPECT_THAT(get_nodes_response.contexts(),
                    UnorderedPointwise(EqualsProto<Context>(/*ignore_fields=*/{
                                           "type", "create_time_since_epoch",
                                           "last_update_time_since_epoch"}),
                                       want_contexts));
      };

  GetContextsByTypeRequest get_nodes_request;
  get_nodes_request.set_type_name("test_type");
  GetContextsByTypeResponse get_nodes_response;

  // Fetches the node according through the types.
  // Test: lists the nodes without options, expecting all nodes are
  // returned.
  call_get_contexts_by_type(get_nodes_request, absl::MakeSpan(nodes),
                            get_nodes_response);
  EXPECT_TRUE(get_nodes_response.next_page_token().empty());

  // Test: list the nodes with options.max_result_size >= 101
  // nodes, expect top 101 nodes are returned.
  get_nodes_request.mutable_options()->Clear();
  get_nodes_request.mutable_options()->set_max_result_size(102);
  call_get_contexts_by_type(
      get_nodes_request, absl::MakeSpan(nodes.data(), 101), get_nodes_response);
  EXPECT_TRUE(get_nodes_response.next_page_token().empty());

  // Test: list the nodes with options.max_result_size < 101.
  get_nodes_request.mutable_options()->Clear();
  get_nodes_request.mutable_options()->set_max_result_size(100);
  call_get_contexts_by_type(
      get_nodes_request, absl::MakeSpan(nodes.data(), 100), get_nodes_response);
  EXPECT_FALSE(get_nodes_response.next_page_token().empty());

  // Test: lists the nodes with next page token.
  get_nodes_request.mutable_options()->Clear();
  get_nodes_request.mutable_options()->set_next_page_token(
      get_nodes_response.next_page_token());
  call_get_contexts_by_type(get_nodes_request,
                            absl::MakeSpan(nodes.data() + 100, 10),
                            get_nodes_response);
  EXPECT_TRUE(get_nodes_response.next_page_token().empty());
}

TEST_P(MetadataStoreTestSuite, GetExecutionsByContextWithFilterStateQuery) {
  // Insert context and execution types.
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    context_types: { name: 'context_type' }
    execution_types: { name: 'execution_type' }
  )pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(metadata_store_->PutTypes(put_types_request, &put_types_response),
            absl::OkStatus());
  int64_t context_type_id = put_types_response.context_type_ids(0);
  int64_t execution_type_id = put_types_response.execution_type_ids(0);

  // Insert two contexts.
  Context context1, context2;
  context1.set_type_id(context_type_id);
  context1.set_name("context1");
  context2.set_type_id(context_type_id);
  context2.set_name("context2");
  PutContextsRequest put_context_request;
  *put_context_request.add_contexts() = context1;
  *put_context_request.add_contexts() = context2;
  PutContextsResponse put_context_response;
  ASSERT_EQ(
      metadata_store_->PutContexts(put_context_request, &put_context_response),
      absl::OkStatus());
  context1.set_id(put_context_response.context_ids(0));
  context2.set_id(put_context_response.context_ids(1));

  // Insert two executions associated with each context.
  Execution execution1, execution2;
  execution1.set_type_id(execution_type_id);
  execution1.set_last_known_state(Execution::NEW);
  execution1.set_name("execution1");
  execution2.set_type_id(execution_type_id);
  execution2.set_last_known_state(Execution::RUNNING);
  execution2.set_name("execution2");
  PutExecutionRequest put_execution_request1;
  *put_execution_request1.mutable_execution() = execution1;
  *put_execution_request1.add_contexts() = context1;
  PutExecutionResponse put_execution_response1;
  ASSERT_EQ(metadata_store_->PutExecution(put_execution_request1,
                                          &put_execution_response1),
            absl::OkStatus());
  execution1.set_id(put_execution_response1.execution_id());

  PutExecutionRequest put_execution_request2;
  *put_execution_request2.mutable_execution() = execution2;
  *put_execution_request2.add_contexts() = context2;
  PutExecutionResponse put_execution_response2;
  ASSERT_EQ(metadata_store_->PutExecution(put_execution_request2,
                                          &put_execution_response2),
            absl::OkStatus());
  execution2.set_id(put_execution_response2.execution_id());

  // Test: GetExecutionsByContext for one context should only return the
  // executions associated with that context even when filter query matches the
  // executions for another context.
  {
    GetExecutionsByContextRequest get_executions_request;
    GetExecutionsByContextResponse get_executions_response;
    get_executions_request.mutable_options()->set_filter_query(
        R"( last_known_state = NEW OR last_known_state = RUNNING )");
    get_executions_request.set_context_id(context1.id());
    ASSERT_EQ(metadata_store_->GetExecutionsByContext(get_executions_request,
                                                      &get_executions_response),
              absl::OkStatus());
    ASSERT_THAT(get_executions_response.executions(), SizeIs(1));
    EXPECT_THAT(
        get_executions_response.executions(0),
        EqualsProto(execution1,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
  }

  {
    GetExecutionsByContextRequest get_executions_request;
    GetExecutionsByContextResponse get_executions_response;
    get_executions_request.mutable_options()->set_filter_query(
        R"( last_known_state = NEW OR last_known_state = RUNNING )");
    get_executions_request.set_context_id(context2.id());
    ASSERT_EQ(metadata_store_->GetExecutionsByContext(get_executions_request,
                                                      &get_executions_response),
              absl::OkStatus());
    ASSERT_THAT(get_executions_response.executions(), SizeIs(1));
    EXPECT_THAT(
        get_executions_response.executions(0),
        EqualsProto(execution2,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
  }
}

TEST_P(MetadataStoreTestSuite, GetExecutionFilterWithSpecialChars) {
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    context_types: { name: 'context_type' }
    execution_types: { name: 'execution_type' }
  )pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutTypes(put_types_request, &put_types_response));
  int64_t context_type_id = put_types_response.context_type_ids(0);
  int64_t execution_type_id = put_types_response.execution_type_ids(0);

  // Setup: Insert an execution, a context and an association.
  Context context;
  context.set_type_id(context_type_id);
  context.set_name("context_name_with_'");
  PutContextsRequest put_context_request;
  *put_context_request.add_contexts() = context;
  PutContextsResponse put_context_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_context_request,
                                         &put_context_response));
  context.set_id(put_context_response.context_ids(0));

  Execution execution;
  execution.set_type_id(execution_type_id);
  execution.set_name("exe_name_with_'");
  PutExecutionRequest put_execution_request;
  *put_execution_request.mutable_execution() = execution;
  *put_execution_request.add_contexts() = context;
  PutExecutionResponse put_execution_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response));
  execution.set_id(put_execution_response.execution_id());

  // Test: GetExecutions with filter query on execution.name could obtain the
  // wanted execution.
  {
    GetExecutionsRequest get_executions_request;
    GetExecutionsResponse get_executions_response;
    get_executions_request.mutable_options()->set_filter_query(
        R"( name = "exe_name_with_'" )");
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetExecutions(get_executions_request,
                                             &get_executions_response));
    ASSERT_THAT(get_executions_response.executions(), SizeIs(1));
    EXPECT_THAT(
        get_executions_response.executions(0),
        EqualsProto(execution,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
  }

  // Test: GetExecutions with filter query on context.name could obtain the
  // wanted execution.
  {
    GetExecutionsRequest get_executions_request;
    GetExecutionsResponse get_executions_response;
    get_executions_request.mutable_options()->set_filter_query(
        R"( contexts_1.name = "context_name_with_'" )");
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetExecutions(get_executions_request,
                                             &get_executions_response));
    ASSERT_THAT(get_executions_response.executions(), SizeIs(1));
    EXPECT_THAT(
        get_executions_response.executions(0),
        EqualsProto(execution,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
  }

  // Test: GetExecutions with filter query on both execution.name and
  // context.name could obtain the wanted execution.
  {
    GetExecutionsRequest get_executions_request;
    GetExecutionsResponse get_executions_response;
    get_executions_request.mutable_options()->set_filter_query(R"(
      contexts_1.name = "context_name_with_'" AND name = "exe_name_with_'")");
    ASSERT_EQ(absl::OkStatus(),
              metadata_store_->GetExecutions(get_executions_request,
                                             &get_executions_response));
    ASSERT_THAT(get_executions_response.executions(), SizeIs(1));
    EXPECT_THAT(
        get_executions_response.executions(0),
        EqualsProto(execution,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
  }

  // Test: If single quote is exist in a pair of single quotes, return error.
  {
    GetExecutionsRequest get_executions_request;
    GetExecutionsResponse get_executions_response;
    get_executions_request.mutable_options()->set_filter_query(R"(
      name = 'exe_name_with_'')");
    EXPECT_TRUE(absl::IsInvalidArgument(metadata_store_->GetExecutions(
        get_executions_request, &get_executions_response)));
  }
}

TEST_P(MetadataStoreTestSuite, GetExecutionWithFilterContextQuery) {
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    context_types: { name: 'context_type' }
    execution_types: { name: 'execution_type' }
  )pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(metadata_store_->PutTypes(put_types_request, &put_types_response),
            absl::OkStatus());
  int64_t context_type_id = put_types_response.context_type_ids(0);
  int64_t execution_type_id = put_types_response.execution_type_ids(0);

  // Setup: Insert an execution, a context and an association.
  Context context;
  context.set_type_id(context_type_id);
  // context name has a form similar to properties filter query syntax.
  context.set_name("properties.component.component");
  PutContextsRequest put_context_request;
  *put_context_request.add_contexts() = context;
  PutContextsResponse put_context_response;
  ASSERT_EQ(
      metadata_store_->PutContexts(put_context_request, &put_context_response),
      absl::OkStatus());
  context.set_id(put_context_response.context_ids(0));

  Execution execution;
  execution.set_type_id(execution_type_id);
  execution.set_name("execution_name'");
  PutExecutionRequest put_execution_request;
  *put_execution_request.mutable_execution() = execution;
  *put_execution_request.add_contexts() = context;
  PutExecutionResponse put_execution_response;
  ASSERT_EQ(metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response),
            absl::OkStatus());
  execution.set_id(put_execution_response.execution_id());

  // Test: GetExecutions with filter query on context.type and context.name
  // could obtain the wanted execution.
  {
    GetExecutionsRequest get_executions_request;
    GetExecutionsResponse get_executions_response;
    get_executions_request.mutable_options()->set_filter_query(
        R"((contexts_0.type = 'context_type') AND (contexts_0.name = 'properties.component.component'))");
    ASSERT_EQ(metadata_store_->GetExecutions(get_executions_request,
                                             &get_executions_response),
              absl::OkStatus());
    ASSERT_THAT(get_executions_response.executions(), SizeIs(1));
    EXPECT_THAT(
        get_executions_response.executions(0),
        EqualsProto(execution,
                    /*ignore_fields=*/{"type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"}));
  }
}

// Test that PutLineageSubgraph adds entire subgraph.
TEST_P(MetadataStoreTestSuite, PutLineageSubgraphAndVerifyLineageGraph) {
  // Prepare the metadata store with types
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    context_types: {
      name: 'context_type'
      properties { key: 'property_1' value: INT }
    }

    execution_types: {
      name: 'execution_type'
      properties { key: 'property_1' value: DOUBLE }
    }
    artifact_types: {
      name: 'artifact_type'
      properties { key: 'property_1' value: STRING }
    }
  )pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutTypes(put_types_request, &put_types_response));

  // Prepare the metadata store with existing data to verify input validity
  Context context;
  context.set_type_id(put_types_response.context_type_ids(0));
  context.set_name("context");
  (*context.mutable_properties())["property_1"].set_int_value(1);
  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = context;
  PutContextsResponse put_contexts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request,
                                         &put_contexts_response));

  Execution execution;
  execution.set_type_id(put_types_response.execution_type_ids(0));
  (*execution.mutable_properties())["property_1"].set_double_value(1.0);
  PutExecutionRequest put_execution_request;
  *put_execution_request.mutable_execution() = execution;
  PutExecutionResponse put_execution_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response));
  execution.set_id(put_execution_response.execution_id());

  Artifact artifact;
  artifact.set_type_id(put_types_response.artifact_type_ids(0));
  artifact.set_uri("testuri");
  (*artifact.mutable_properties())["property_1"].set_string_value("1");
  PutArtifactsRequest put_artifacts_request;
  *put_artifacts_request.add_artifacts() = artifact;
  PutArtifactsResponse put_artifacts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request,
                                          &put_artifacts_response));
  artifact.set_id(put_artifacts_response.artifact_ids(0));

  // Prepare the PutLineageSubgraph request
  PutLineageSubgraphRequest put_lineage_subgraph_request;
  *put_lineage_subgraph_request.add_executions() = execution;
  *put_lineage_subgraph_request.add_artifacts() = artifact;
  *put_lineage_subgraph_request.add_contexts() = context;
  put_lineage_subgraph_request.mutable_options()
      ->set_reuse_context_if_already_exist(true);

  // Prepare event_edge with execution_index and artifact_index but no
  // execution_id and no artifact_id
  Event event_1;
  event_1.set_type(Event::OUTPUT);
  PutLineageSubgraphRequest::EventEdge* event_edge_1 =
      put_lineage_subgraph_request.add_event_edges();
  event_edge_1->set_execution_index(0);
  event_edge_1->set_artifact_index(0);
  *event_edge_1->mutable_event() = event_1;

  // Prepare event with execution_index and artifact_index and matching
  // execution_id and matching artifact_id
  Event event_2;
  event_2.set_type(Event::INPUT);
  event_2.set_execution_id(put_execution_response.execution_id());
  event_2.set_artifact_id(put_artifacts_response.artifact_ids(0));
  PutLineageSubgraphRequest::EventEdge* event_edge_2 =
      put_lineage_subgraph_request.add_event_edges();
  event_edge_2->set_execution_index(0);
  event_edge_2->set_artifact_index(0);
  *event_edge_2->mutable_event() = event_2;

  // Prepare event with execution_id and artifact_id but no execution_index
  // and no artifact_index
  Event event_3;
  event_3.set_type(Event::DECLARED_INPUT);
  event_3.set_execution_id(put_execution_response.execution_id());
  event_3.set_artifact_id(put_artifacts_response.artifact_ids(0));
  PutLineageSubgraphRequest::EventEdge* event_edge_3 =
      put_lineage_subgraph_request.add_event_edges();
  *event_edge_3->mutable_event() = event_3;

  PutLineageSubgraphResponse put_lineage_subgraph_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutLineageSubgraph(
                put_lineage_subgraph_request, &put_lineage_subgraph_response));

  // Verify lineage subgraph is inserted correctly
  GetExecutionsByContextRequest get_executions_by_context_request;
  get_executions_by_context_request.set_context_id(
      put_lineage_subgraph_response.context_ids(0));
  GetExecutionsByContextResponse get_executions_by_context_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByContext(
                                  get_executions_by_context_request,
                                  &get_executions_by_context_response));
  ASSERT_THAT(get_executions_by_context_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_by_context_response.executions(),
              ElementsAre(EqualsProto(
                  execution,
                  /*ignore_fields=*/{"type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"})));

  GetArtifactsByContextRequest get_artifacts_by_context_request;
  get_artifacts_by_context_request.set_context_id(
      put_lineage_subgraph_response.context_ids(0));
  GetArtifactsByContextResponse get_artifacts_by_context_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByContext(
                                  get_artifacts_by_context_request,
                                  &get_artifacts_by_context_response));
  ASSERT_THAT(get_artifacts_by_context_response.artifacts(), SizeIs(1));
  EXPECT_THAT(get_artifacts_by_context_response.artifacts(),
              ElementsAre(EqualsProto(
                  artifact,
                  /*ignore_fields=*/{"type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"})));

  GetContextsRequest get_contexts_request;
  GetContextsResponse get_contexts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetContexts(get_contexts_request,
                                         &get_contexts_response));
  ASSERT_THAT(get_contexts_response.contexts(), SizeIs(1));
  EXPECT_EQ(get_contexts_response.contexts(0).id(),
            put_lineage_subgraph_response.context_ids(0));
  EXPECT_THAT(get_contexts_response.contexts(),
              ElementsAre(EqualsProto(context,
                                      /*ignore_fields=*/
                                      {"id", "type", "create_time_since_epoch",
                                       "last_update_time_since_epoch"})));

  GetEventsByExecutionIDsRequest get_events_by_execution_ids_request;
  get_events_by_execution_ids_request.add_execution_ids(
      put_lineage_subgraph_response.execution_ids(0));
  GetEventsByExecutionIDsResponse get_events_by_execution_ids_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetEventsByExecutionIDs(
                                  get_events_by_execution_ids_request,
                                  &get_events_by_execution_ids_response));
  ASSERT_THAT(get_events_by_execution_ids_response.events(), SizeIs(3));
  EXPECT_EQ(get_events_by_execution_ids_response.events(0).execution_id(),
            put_lineage_subgraph_response.execution_ids(0));
  EXPECT_EQ(get_events_by_execution_ids_response.events(0).artifact_id(),
            put_lineage_subgraph_response.artifact_ids(0));
  EXPECT_THAT(get_events_by_execution_ids_response.events(),
              UnorderedElementsAre(
                  EqualsProto(event_1,
                              /*ignore_fields=*/{"artifact_id", "execution_id",
                                                 "milliseconds_since_epoch"}),
                  EqualsProto(event_2,
                              /*ignore_fields=*/{"milliseconds_since_epoch"}),
                  EqualsProto(event_3,
                              /*ignore_fields=*/{"milliseconds_since_epoch"})));
}

// Test that PutLineageSubgraph Upsert artifacts before execution.
TEST_P(MetadataStoreTestSuite, PutLineageSubgraphWithNewArtifactsExecutions) {
  // Prepare types and write them to MLMD.
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    context_types: {
      name: 'context_type'
      properties { key: 'property_1' value: INT }
    }

    execution_types: {
      name: 'execution_type'
      properties { key: 'property_1' value: DOUBLE }
    }
    artifact_types: {
      name: 'artifact_type'
      properties { key: 'property_1' value: STRING }
    }
  )pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutTypes(put_types_request, &put_types_response));

  // Prepare a context and write it to MLMD.
  Context context;
  context.set_type_id(put_types_response.context_type_ids(0));
  context.set_name("context");
  (*context.mutable_properties())["property_1"].set_int_value(1);
  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = context;
  PutContextsResponse put_contexts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request,
                                         &put_contexts_response));

  // Prepare a new execution and write it to MLMD.
  Execution execution;
  execution.set_type_id(put_types_response.execution_type_ids(0));
  (*execution.mutable_properties())["property_1"].set_double_value(1.0);
  PutExecutionRequest put_execution_request;
  *put_execution_request.mutable_execution() = execution;
  PutExecutionResponse put_execution_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response));
  execution.set_id(put_execution_response.execution_id());

  // Prepare a new artifact, but don't write it to MLMD.
  Artifact artifact;
  artifact.set_type_id(put_types_response.artifact_type_ids(0));
  artifact.set_uri("testuri");
  (*artifact.mutable_properties())["property_1"].set_string_value("1");

  // Prepare the PutLineageSubgraph request
  PutLineageSubgraphRequest put_lineage_subgraph_request;
  *put_lineage_subgraph_request.add_executions() = execution;
  *put_lineage_subgraph_request.add_artifacts() = artifact;
  *put_lineage_subgraph_request.add_contexts() = context;
  put_lineage_subgraph_request.mutable_options()
      ->set_reuse_context_if_already_exist(true);

  // Prepare event_edge with execution_index and artifact_index but no
  // execution_id and no artifact_id
  Event event;
  event.set_type(Event::OUTPUT);
  PutLineageSubgraphRequest::EventEdge* event_edge =
      put_lineage_subgraph_request.add_event_edges();
  event_edge->set_execution_index(0);
  event_edge->set_artifact_index(0);
  *event_edge->mutable_event() = event;

  PutLineageSubgraphResponse put_lineage_subgraph_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutLineageSubgraph(
                put_lineage_subgraph_request, &put_lineage_subgraph_response));

  // Verify that new artifact and execution are correctly written.
  GetExecutionsByContextRequest get_executions_by_context_request;
  get_executions_by_context_request.set_context_id(
      put_lineage_subgraph_response.context_ids(0));
  GetExecutionsByContextResponse get_executions_by_context_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetExecutionsByContext(
                                  get_executions_by_context_request,
                                  &get_executions_by_context_response));
  ASSERT_THAT(get_executions_by_context_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_by_context_response.executions(),
              ElementsAre(EqualsProto(
                  execution,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"})));

  GetArtifactsByContextRequest get_artifacts_by_context_request;
  get_artifacts_by_context_request.set_context_id(
      put_lineage_subgraph_response.context_ids(0));
  GetArtifactsByContextResponse get_artifacts_by_context_response;
  ASSERT_EQ(absl::OkStatus(), metadata_store_->GetArtifactsByContext(
                                  get_artifacts_by_context_request,
                                  &get_artifacts_by_context_response));
  ASSERT_THAT(get_artifacts_by_context_response.artifacts(), SizeIs(1));
  EXPECT_THAT(get_artifacts_by_context_response.artifacts(),
              ElementsAre(EqualsProto(
                  artifact,
                  /*ignore_fields=*/{"id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"})));

  // Verify that the update time of execution is larger than the creation time
  // of the artifact.
  EXPECT_GE(
      get_executions_by_context_response.executions(0)
          .last_update_time_since_epoch(),
      get_artifacts_by_context_response.artifacts(0).create_time_since_epoch());
}

TEST_P(MetadataStoreTestSuite,
       PutLineageSubgraphUpdatesArtifactsAndExecutions) {
  // Prepare the metadata store with types
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    execution_types: {
      name: 'execution_type'
      properties { key: 'property_1' value: DOUBLE }
    }
    artifact_types: {
      name: 'artifact_type'
      properties { key: 'property_1' value: STRING }
    }
  )pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutTypes(put_types_request, &put_types_response));

  // Prepare the metadata store with existing data to verify input validity
  Execution execution;
  execution.set_last_known_state(Execution::RUNNING);
  execution.set_type_id(put_types_response.execution_type_ids(0));
  (*execution.mutable_properties())["property_1"].set_double_value(1.0);
  (*execution.mutable_custom_properties())["property_2"].set_double_value(2.0);
  PutExecutionRequest put_execution_request;
  *put_execution_request.mutable_execution() = execution;
  PutExecutionResponse put_execution_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response));
  execution.set_id(put_execution_response.execution_id());

  Artifact artifact;
  artifact.set_state(Artifact::PENDING);
  artifact.set_type_id(put_types_response.artifact_type_ids(0));
  artifact.set_uri("testuri");
  (*artifact.mutable_properties())["property_1"].set_string_value("1");
  (*artifact.mutable_custom_properties())["property_2"].set_string_value("2");
  PutArtifactsRequest put_artifacts_request;
  *put_artifacts_request.add_artifacts() = artifact;
  PutArtifactsResponse put_artifacts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request,
                                          &put_artifacts_response));
  artifact.set_id(put_artifacts_response.artifact_ids(0));

  // Modify existing execution and artifact nodes for verifying update in API
  // call.
  execution.set_last_known_state(Execution::COMPLETE);
  (*execution.mutable_properties())["property_1"].set_double_value(2.0);
  (*execution.mutable_custom_properties())["property_2"].set_double_value(3.0);
  artifact.set_state(Artifact::LIVE);
  (*artifact.mutable_properties())["property_1"].set_string_value("2");
  (*artifact.mutable_custom_properties())["property_2"].set_string_value("3");

  // Prepare the PutLineageSubgraph request
  PutLineageSubgraphRequest put_lineage_subgraph_request;
  *put_lineage_subgraph_request.add_executions() = execution;
  *put_lineage_subgraph_request.add_artifacts() = artifact;

  // Verify API call is successful
  PutLineageSubgraphResponse put_lineage_subgraph_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutLineageSubgraph(
                put_lineage_subgraph_request, &put_lineage_subgraph_response));

  // Verify lineage subgraph is updated
  GetExecutionsByIDRequest get_executions_by_id_request;
  get_executions_by_id_request.mutable_execution_ids()->CopyFrom(
      put_lineage_subgraph_response.execution_ids());
  GetExecutionsByIDResponse get_executions_by_id_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetExecutionsByID(get_executions_by_id_request,
                                               &get_executions_by_id_response));
  ASSERT_THAT(get_executions_by_id_response.executions(), SizeIs(1));
  EXPECT_THAT(get_executions_by_id_response.executions(),
              ElementsAre(EqualsProto(
                  execution,
                  /*ignore_fields=*/{"type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"})));

  GetArtifactsByIDRequest get_artifacts_by_id_request;
  get_artifacts_by_id_request.mutable_artifact_ids()->CopyFrom(
      put_lineage_subgraph_response.artifact_ids());
  GetArtifactsByIDResponse get_artifacts_by_id_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->GetArtifactsByID(get_artifacts_by_id_request,
                                              &get_artifacts_by_id_response));
  ASSERT_THAT(get_artifacts_by_id_response.artifacts(), SizeIs(1));
  EXPECT_THAT(get_artifacts_by_id_response.artifacts(),
              ElementsAre(EqualsProto(
                  artifact,
                  /*ignore_fields=*/{"type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"})));
}

// Test that PutLineageSubgraph fails on invalid inputs.
TEST_P(MetadataStoreTestSuite, PutLineageSubgraphFailsWithInvalidEventEdge) {
  // Prepare the metadata store with types
  PutTypesRequest put_types_request = ParseTextProtoOrDie<PutTypesRequest>(R"pb(
    context_types: { name: 'context_type' }
    execution_types: { name: 'execution_type' }
    artifact_types: { name: 'artifact_type' }
  )pb");
  PutTypesResponse put_types_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutTypes(put_types_request, &put_types_response));

  Execution execution;
  execution.set_type_id(put_types_response.execution_type_ids(0));
  PutExecutionRequest put_execution_request;
  *put_execution_request.mutable_execution() = execution;
  PutExecutionResponse put_execution_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutExecution(put_execution_request,
                                          &put_execution_response));
  execution.set_id(put_execution_response.execution_id());

  Artifact artifact;
  artifact.set_type_id(put_types_response.artifact_type_ids(0));
  artifact.set_uri("testuri");
  PutArtifactsRequest put_artifacts_request;
  *put_artifacts_request.add_artifacts() = artifact;
  PutArtifactsResponse put_artifacts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutArtifacts(put_artifacts_request,
                                          &put_artifacts_response));
  artifact.set_id(put_artifacts_response.artifact_ids(0));

  Context context;
  context.set_type_id(put_types_response.context_type_ids(0));
  context.set_name("context");
  PutContextsRequest put_contexts_request;
  *put_contexts_request.add_contexts() = context;
  PutContextsResponse put_contexts_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store_->PutContexts(put_contexts_request,
                                         &put_contexts_response));
  // Setup `base_request` that has
  //   1) 1 Artifact proto with id populated
  //   2) 1 Execution proto with id populated
  //   3) 1 EventEdge proto with indices populated but without artifact_id and
  //      execution_id in the event.
  PutLineageSubgraphRequest base_request;
  *base_request.add_artifacts() = artifact;
  *base_request.add_executions() = execution;
  PutLineageSubgraphRequest::EventEdge* base_event_edge =
      base_request.add_event_edges();
  base_event_edge->set_execution_index(0);
  base_event_edge->set_artifact_index(0);
  base_event_edge->mutable_event()->set_type(Event::INPUT);

  // Test failure on event_edge with no Event
  {
    PutLineageSubgraphRequest request = base_request;
    request.mutable_event_edges(0)->clear_event();

    PutLineageSubgraphResponse response;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_store_->PutLineageSubgraph(request, &response)));
  }

  // Test failure on event_edge with no execution_index and no execution id
  {
    PutLineageSubgraphRequest request = base_request;
    request.mutable_event_edges(0)->clear_execution_index();

    PutLineageSubgraphResponse response;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_store_->PutLineageSubgraph(request, &response)));
  }

  // Test failure on event_edge with no artifact_index and no artifact id
  {
    PutLineageSubgraphRequest request = base_request;
    request.mutable_event_edges(0)->clear_artifact_index();

    PutLineageSubgraphResponse response;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_store_->PutLineageSubgraph(request, &response)));
  }

  // Test failure on event_edge with out of range execution_index
  {
    PutLineageSubgraphRequest request = base_request;
    request.mutable_executions()->Clear();

    PutLineageSubgraphResponse response;
    EXPECT_TRUE(absl::IsOutOfRange(
        metadata_store_->PutLineageSubgraph(request, &response)));
  }

  // Test failure on event_edge with out of range artifact_index
  {
    PutLineageSubgraphRequest request = base_request;
    request.mutable_artifacts()->Clear();

    PutLineageSubgraphResponse response;
    EXPECT_TRUE(absl::IsOutOfRange(
        metadata_store_->PutLineageSubgraph(request, &response)));
  }

  // Test failure on inserting with already existing context
  {
    PutLineageSubgraphRequest request = base_request;
    *request.add_contexts() = context;

    PutLineageSubgraphResponse response;
    EXPECT_TRUE(absl::IsAlreadyExists(
        metadata_store_->PutLineageSubgraph(request, &response)));
  }

  // Test failure on inserting event with non-matching execution ID
  {
    PutLineageSubgraphRequest request = base_request;
    request.mutable_event_edges(0)->mutable_event()
        ->set_execution_id(execution.id() + 1);

    PutLineageSubgraphResponse response;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_store_->PutLineageSubgraph(request, &response)));
  }

  // Test failure on inserting event with non-matching artifact ID
  {
    PutLineageSubgraphRequest request = base_request;
    request.mutable_event_edges(0)->mutable_event()
        ->set_artifact_id(artifact.id() + 1);

    PutLineageSubgraphResponse response;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_store_->PutLineageSubgraph(request, &response)));
  }

  // Test failure on event_edge with event.execution_id and execution_index that
  // points to an execution without an ID
  {
    PutLineageSubgraphRequest request = base_request;
    request.mutable_event_edges(0)->mutable_event()->set_execution_id(
        execution.id());
    request.mutable_executions(0)->clear_id();

    PutLineageSubgraphResponse response;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_store_->PutLineageSubgraph(request, &response)));
  }

  // Test failure on event_edge with event.artifact_id and artifact_index that
  // points to an artifact without an ID
  {
    PutLineageSubgraphRequest request = base_request;
    request.mutable_event_edges(0)->mutable_event()->set_artifact_id(
        artifact.id());
    request.mutable_artifacts(0)->clear_id();

    PutLineageSubgraphResponse response;
    EXPECT_TRUE(absl::IsInvalidArgument(
        metadata_store_->PutLineageSubgraph(request, &response)));
  }
}

}  // namespace
}  // namespace testing
}  // namespace ml_metadata
