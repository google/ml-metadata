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
#include "ml_metadata/tools/mlmd_bench/util.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

constexpr int kNumberOfInsertedArtifactTypes = 51;
constexpr int kNumberOfInsertedExecutionTypes = 52;
constexpr int kNumberOfInsertedContextTypes = 53;

constexpr int kNumberOfInsertedArtifacts = 101;
constexpr int kNumberOfInsertedExecutions = 102;
constexpr int kNumberOfInsertedContexts = 103;

// Tests InsertTypesInDb().
TEST(UtilInsertTest, InsertTypesTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));

  GetArtifactTypesResponse get_artifact_types_response;
  TF_ASSERT_OK(store->GetArtifactTypes(
      /*request=*/{}, &get_artifact_types_response));
  GetExecutionTypesResponse get_execution_types_response;
  TF_ASSERT_OK(store->GetExecutionTypes(
      /*request=*/{}, &get_execution_types_response));
  GetContextTypesResponse get_context_types_response;
  TF_ASSERT_OK(store->GetContextTypes(
      /*request=*/{}, &get_context_types_response));

  EXPECT_THAT(get_artifact_types_response.artifact_types(),
              ::testing::SizeIs(kNumberOfInsertedArtifactTypes));
  EXPECT_THAT(get_execution_types_response.execution_types(),
              ::testing::SizeIs(kNumberOfInsertedExecutionTypes));
  EXPECT_THAT(get_context_types_response.context_types(),
              ::testing::SizeIs(kNumberOfInsertedContextTypes));
}

// Tests InsertNodesInDb().
TEST(UtilInsertTest, InsertNodesTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfInsertedArtifacts,
      /*num_execution_nodes=*/kNumberOfInsertedExecutions,
      /*num_context_nodes=*/kNumberOfInsertedContexts, *store));

  GetArtifactsResponse get_artifacts_response;
  TF_ASSERT_OK(store->GetArtifacts(
      /*request=*/{}, &get_artifacts_response));
  GetExecutionsResponse get_executions_response;
  TF_ASSERT_OK(store->GetExecutions(
      /*request=*/{}, &get_executions_response));
  GetContextsResponse get_contexts_response;
  TF_ASSERT_OK(store->GetContexts(
      /*request=*/{}, &get_contexts_response));

  EXPECT_THAT(get_artifacts_response.artifacts(),
              ::testing::SizeIs(kNumberOfInsertedArtifacts));
  EXPECT_THAT(get_executions_response.executions(),
              ::testing::SizeIs(kNumberOfInsertedExecutions));
  EXPECT_THAT(get_contexts_response.contexts(),
              ::testing::SizeIs(kNumberOfInsertedContexts));
}

// Tests GetExistingTypes() with FillTypesConfig as input.
TEST(UtilGetTest, GetTypesWithFillTypesConfigTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));

  {
    std::vector<Type> exisiting_types;
    FillTypesConfig fill_types_config;
    fill_types_config.set_specification(FillTypesConfig::ARTIFACT_TYPE);
    TF_ASSERT_OK(GetExistingTypes(fill_types_config, *store, exisiting_types));
    EXPECT_THAT(exisiting_types,
                ::testing::SizeIs(kNumberOfInsertedArtifactTypes));
  }

  {
    std::vector<Type> exisiting_types;
    FillTypesConfig fill_types_config;
    fill_types_config.set_specification(FillTypesConfig::EXECUTION_TYPE);
    TF_ASSERT_OK(GetExistingTypes(fill_types_config, *store, exisiting_types));
    EXPECT_THAT(exisiting_types,
                ::testing::SizeIs(kNumberOfInsertedExecutionTypes));
  }

  {
    std::vector<Type> exisiting_types;
    FillTypesConfig fill_types_config;
    fill_types_config.set_specification(FillTypesConfig::CONTEXT_TYPE);
    TF_ASSERT_OK(GetExistingTypes(fill_types_config, *store, exisiting_types));
    EXPECT_THAT(exisiting_types,
                ::testing::SizeIs(kNumberOfInsertedContextTypes));
  }
}

// Tests GetExistingTypes() with FillNodesConfig as input.
TEST(UtilGetTest, GetTypesWithFillNodesConfigTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));

  {
    std::vector<Type> exisiting_types;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::ARTIFACT);
    TF_ASSERT_OK(GetExistingTypes(fill_nodes_config, *store, exisiting_types));
    EXPECT_THAT(exisiting_types,
                ::testing::SizeIs(kNumberOfInsertedArtifactTypes));
  }

  {
    std::vector<Type> exisiting_types;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::EXECUTION);
    TF_ASSERT_OK(GetExistingTypes(fill_nodes_config, *store, exisiting_types));
    EXPECT_THAT(exisiting_types,
                ::testing::SizeIs(kNumberOfInsertedExecutionTypes));
  }

  {
    std::vector<Type> exisiting_types;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::CONTEXT);
    TF_ASSERT_OK(GetExistingTypes(fill_nodes_config, *store, exisiting_types));
    EXPECT_THAT(exisiting_types,
                ::testing::SizeIs(kNumberOfInsertedContextTypes));
  }
}

// Tests GetExistingTypes() with ReadTypesConfig as input.
TEST(UtilGetTest, GetTypesWithReadTypesConfigTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));

  std::vector<std::pair<ReadTypesConfig::Specification, int>> specification{
      {ReadTypesConfig::ALL_ARTIFACT_TYPES, kNumberOfInsertedArtifactTypes},
      {ReadTypesConfig::ARTIFACT_TYPES_BY_IDs, kNumberOfInsertedArtifactTypes},
      {ReadTypesConfig::ARTIFACT_TYPE_BY_NAME, kNumberOfInsertedArtifactTypes},
      {ReadTypesConfig::ALL_EXECUTION_TYPES, kNumberOfInsertedExecutionTypes},
      {ReadTypesConfig::EXECUTION_TYPES_BY_IDs,
       kNumberOfInsertedExecutionTypes},
      {ReadTypesConfig::EXECUTION_TYPE_BY_NAME,
       kNumberOfInsertedExecutionTypes},
      {ReadTypesConfig::ALL_CONTEXT_TYPES, kNumberOfInsertedContextTypes},
      {ReadTypesConfig::CONTEXT_TYPES_BY_IDs, kNumberOfInsertedContextTypes},
      {ReadTypesConfig::CONTEXT_TYPE_BY_NAME, kNumberOfInsertedContextTypes}};

  for (int i = 0; i < 9; ++i) {
    std::vector<Type> exisiting_types;
    ReadTypesConfig read_types_config;
    read_types_config.set_specification(specification[i].first);
    TF_ASSERT_OK(GetExistingTypes(read_types_config, *store, exisiting_types));
    EXPECT_THAT(exisiting_types, ::testing::SizeIs(specification[i].second));
  }
}

// Tests GetExistingNodes() with FillNodesConfig as input.
TEST(UtilGetTest, GetNodesWithFillNodesConfigTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfInsertedArtifacts,
      /*num_execution_nodes=*/kNumberOfInsertedExecutions,
      /*num_context_nodes=*/kNumberOfInsertedContexts, *store));

  {
    std::vector<Node> exisiting_nodes;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::ARTIFACT);
    TF_ASSERT_OK(GetExistingNodes(fill_nodes_config, *store, exisiting_nodes));
    EXPECT_THAT(exisiting_nodes, ::testing::SizeIs(kNumberOfInsertedArtifacts));
  }

  {
    std::vector<Node> exisiting_nodes;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::EXECUTION);
    TF_ASSERT_OK(GetExistingNodes(fill_nodes_config, *store, exisiting_nodes));
    EXPECT_THAT(exisiting_nodes,
                ::testing::SizeIs(kNumberOfInsertedExecutions));
  }

  {
    std::vector<Node> exisiting_nodes;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::CONTEXT);
    TF_ASSERT_OK(GetExistingNodes(fill_nodes_config, *store, exisiting_nodes));
    EXPECT_THAT(exisiting_nodes, ::testing::SizeIs(kNumberOfInsertedContexts));
  }
}

// Tests GetExistingNodes() with FillContextEdgesConfig as input.
TEST(UtilGetTest, GetNodesWithFillContextEdgesConfigTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfInsertedArtifacts,
      /*num_execution_nodes=*/kNumberOfInsertedExecutions,
      /*num_context_nodes=*/kNumberOfInsertedContexts, *store));

  {
    std::vector<Node> existing_non_context_nodes;
    std::vector<Node> existing_context_nodes;
    FillContextEdgesConfig fill_context_edges_config;
    fill_context_edges_config.set_specification(
        FillContextEdgesConfig::ATTRIBUTION);
    TF_ASSERT_OK(GetExistingNodes(fill_context_edges_config, *store,
                                  existing_non_context_nodes,
                                  existing_context_nodes));
    EXPECT_EQ(kNumberOfInsertedArtifacts, existing_non_context_nodes.size());
    EXPECT_EQ(kNumberOfInsertedContexts, existing_context_nodes.size());
  }

  {
    std::vector<Node> existing_non_context_nodes;
    std::vector<Node> existing_context_nodes;
    FillContextEdgesConfig fill_context_edges_config;
    fill_context_edges_config.set_specification(
        FillContextEdgesConfig::ASSOCIATION);
    TF_ASSERT_OK(GetExistingNodes(fill_context_edges_config, *store,
                                  existing_non_context_nodes,
                                  existing_context_nodes));
    EXPECT_EQ(kNumberOfInsertedExecutions, existing_non_context_nodes.size());
    EXPECT_EQ(kNumberOfInsertedContexts, existing_context_nodes.size());
  }
}

// Tests GetExistingNodes() with FillEventsConfig as input.
TEST(UtilGetTest, GetNodesWithFillEventsConfigTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfInsertedArtifacts,
      /*num_execution_nodes=*/kNumberOfInsertedExecutions,
      /*num_context_nodes=*/kNumberOfInsertedContexts, *store));

  std::vector<Node> existing_artifact_nodes;
  std::vector<Node> existing_execution_nodes;
  FillEventsConfig fill_events_config;
  TF_ASSERT_OK(GetExistingNodes(fill_events_config, *store,
                                existing_artifact_nodes,
                                existing_execution_nodes));
  EXPECT_EQ(kNumberOfInsertedArtifacts, existing_artifact_nodes.size());
  EXPECT_EQ(kNumberOfInsertedExecutions, existing_execution_nodes.size());
}

// Tests GetExistingNodes() with ReadNodesByPropertiesConfig as input.
TEST(UtilGetTest, GetNodesWithReadNodesByPropertiesConfigTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfInsertedArtifacts,
      /*num_execution_nodes=*/kNumberOfInsertedExecutions,
      /*num_context_nodes=*/kNumberOfInsertedContexts, *store));

  std::vector<ReadNodesByPropertiesConfig::Specification> specification{
      ReadNodesByPropertiesConfig::ARTIFACTS_BY_IDs,
      ReadNodesByPropertiesConfig::ARTIFACTS_BY_TYPE,
      ReadNodesByPropertiesConfig::ARTIFACT_BY_TYPE_AND_NAME,
      ReadNodesByPropertiesConfig::ARTIFACTS_BY_URIs,
      ReadNodesByPropertiesConfig::EXECUTIONS_BY_IDs,
      ReadNodesByPropertiesConfig::EXECUTIONS_BY_TYPE,
      ReadNodesByPropertiesConfig::EXECUTION_BY_TYPE_AND_NAME,
      ReadNodesByPropertiesConfig::CONTEXTS_BY_IDs,
      ReadNodesByPropertiesConfig::CONTEXTS_BY_TYPE,
      ReadNodesByPropertiesConfig::CONTEXT_BY_TYPE_AND_NAME};

  std::vector<int> size{
      kNumberOfInsertedArtifacts,  kNumberOfInsertedArtifacts,
      kNumberOfInsertedArtifacts,  kNumberOfInsertedArtifacts,
      kNumberOfInsertedExecutions, kNumberOfInsertedExecutions,
      kNumberOfInsertedExecutions, kNumberOfInsertedContexts,
      kNumberOfInsertedContexts,   kNumberOfInsertedContexts};

  for (int i = 0; i < size.size(); ++i) {
    std::vector<Node> exisiting_nodes;
    ReadNodesByPropertiesConfig read_nodes_by_properties_config;
    read_nodes_by_properties_config.set_specification(specification[i]);
    TF_ASSERT_OK(GetExistingNodes(read_nodes_by_properties_config, *store,
                                  exisiting_nodes));
    EXPECT_THAT(exisiting_nodes, ::testing::SizeIs(size[i]));
  }
}

// Tests GetExistingNodes() with ReadNodesViaContextEdgesConfig as input.
TEST(UtilGetTest, GetNodesWithReadNodesViaContextEdgesConfigTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfInsertedArtifacts,
      /*num_execution_nodes=*/kNumberOfInsertedExecutions,
      /*num_context_nodes=*/kNumberOfInsertedContexts, *store));

  std::vector<ReadNodesViaContextEdgesConfig::Specification> specification{
      ReadNodesViaContextEdgesConfig::ARTIFACTS_BY_CONTEXT,
      ReadNodesViaContextEdgesConfig::EXECUTIONS_BY_CONTEXT,
      ReadNodesViaContextEdgesConfig::CONTEXTS_BY_ARTIFACT,
      ReadNodesViaContextEdgesConfig::CONTEXTS_BY_EXECUTION};

  std::vector<int> size{kNumberOfInsertedContexts, kNumberOfInsertedContexts,
                        kNumberOfInsertedArtifacts,
                        kNumberOfInsertedExecutions};

  for (int i = 0; i < size.size(); ++i) {
    std::vector<Node> exisiting_nodes;
    ReadNodesViaContextEdgesConfig read_nodes_via_context_edges_config;
    read_nodes_via_context_edges_config.set_specification(specification[i]);
    TF_ASSERT_OK(GetExistingNodes(read_nodes_via_context_edges_config, *store,
                                  exisiting_nodes));
    EXPECT_THAT(exisiting_nodes, ::testing::SizeIs(size[i]));
  }
}

// Tests GetExistingNodes() with ReadEventsConfig as input.
TEST(UtilGetTest, GetNodesWithReadEventsConfigTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_nodes=*/kNumberOfInsertedArtifacts,
      /*num_execution_nodes=*/kNumberOfInsertedExecutions,
      /*num_context_nodes=*/kNumberOfInsertedContexts, *store));

  {
    std::vector<Node> existing_nodes;
    ReadEventsConfig read_events_config;
    read_events_config.set_specification(
        ReadEventsConfig::EVENTS_BY_ARTIFACT_IDS);
    TF_ASSERT_OK(GetExistingNodes(read_events_config, *store, existing_nodes));
    EXPECT_EQ(kNumberOfInsertedArtifacts, existing_nodes.size());
  }

  {
    std::vector<Node> existing_nodes;
    ReadEventsConfig read_events_config;
    read_events_config.set_specification(
        ReadEventsConfig::EVENTS_BY_EXECUTION_IDS);
    TF_ASSERT_OK(GetExistingNodes(read_events_config, *store, existing_nodes));
    EXPECT_EQ(kNumberOfInsertedExecutions, existing_nodes.size());
  }
}

}  // namespace
}  // namespace ml_metadata
