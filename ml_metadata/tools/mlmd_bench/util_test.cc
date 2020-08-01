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

  ASSERT_EQ(kNumberOfInsertedArtifactTypes,
            get_artifact_types_response.artifact_types_size());
  ASSERT_EQ(kNumberOfInsertedExecutionTypes,
            get_execution_types_response.execution_types_size());
  ASSERT_EQ(kNumberOfInsertedContextTypes,
            get_context_types_response.context_types_size());
}

// Tests InsertNodesInDb().
TEST(UtilInsertTest, InsertNodesTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  // InsertTypesInDb() has passed the tests.
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

  ASSERT_EQ(kNumberOfInsertedArtifacts,
            get_artifacts_response.artifacts_size());
  ASSERT_EQ(kNumberOfInsertedExecutions,
            get_executions_response.executions_size());
  ASSERT_EQ(kNumberOfInsertedContexts, get_contexts_response.contexts_size());
}

// Tests GetExistingTypes() with FillTypesConfig as input.
TEST(UtilGetTest, GetTypesWithFillTypesConfigTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  // InsertTypesInDb() has passed the tests.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));

  {
    std::vector<Type> exisiting_types;
    FillTypesConfig fill_types_config;
    fill_types_config.set_specification(FillTypesConfig::ARTIFACT_TYPE);
    TF_ASSERT_OK(GetExistingTypes(fill_types_config, *store, exisiting_types));
    EXPECT_EQ(kNumberOfInsertedArtifactTypes, exisiting_types.size());
  }

  {
    std::vector<Type> exisiting_types;
    FillTypesConfig fill_types_config;
    fill_types_config.set_specification(FillTypesConfig::EXECUTION_TYPE);
    TF_ASSERT_OK(GetExistingTypes(fill_types_config, *store, exisiting_types));
    EXPECT_EQ(kNumberOfInsertedExecutionTypes, exisiting_types.size());
  }

  {
    std::vector<Type> exisiting_types;
    FillTypesConfig fill_types_config;
    fill_types_config.set_specification(FillTypesConfig::CONTEXT_TYPE);
    TF_ASSERT_OK(GetExistingTypes(fill_types_config, *store, exisiting_types));
    EXPECT_EQ(kNumberOfInsertedContextTypes, exisiting_types.size());
  }
}

// Tests GetExistingTypes() with FillNodesConfig as input.
TEST(UtilGetTest, GetTypesWithFillNodesConfigTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  // InsertTypesInDb() has passed the tests.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));

  {
    std::vector<Type> exisiting_types;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::ARTIFACT);
    TF_ASSERT_OK(GetExistingTypes(fill_nodes_config, *store, exisiting_types));
    EXPECT_EQ(kNumberOfInsertedArtifactTypes, exisiting_types.size());
  }

  {
    std::vector<Type> exisiting_types;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::EXECUTION);
    TF_ASSERT_OK(GetExistingTypes(fill_nodes_config, *store, exisiting_types));
    EXPECT_EQ(kNumberOfInsertedExecutionTypes, exisiting_types.size());
  }

  {
    std::vector<Type> exisiting_types;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::CONTEXT);
    TF_ASSERT_OK(GetExistingTypes(fill_nodes_config, *store, exisiting_types));
    EXPECT_EQ(kNumberOfInsertedContextTypes, exisiting_types.size());
  }
}

// Tests GetExistingNodes().
TEST(UtilGetTest, GetNodesTest) {
  std::unique_ptr<MetadataStore> store;
  ConnectionConfig mlmd_config;
  // Uses a fake in-memory SQLite database for testing.
  mlmd_config.mutable_fake_database();
  TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
  // InsertTypesInDb() has passed the tests.
  TF_ASSERT_OK(InsertTypesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifactTypes,
      /*num_execution_types=*/kNumberOfInsertedExecutionTypes,
      /*num_context_types=*/kNumberOfInsertedContextTypes, *store));
  // InsertNodesInDb() has passed the tests.
  TF_ASSERT_OK(InsertNodesInDb(
      /*num_artifact_types=*/kNumberOfInsertedArtifacts,
      /*num_execution_types=*/kNumberOfInsertedExecutions,
      /*num_context_types=*/kNumberOfInsertedContexts, *store));

  {
    std::vector<Node> exisiting_nodes;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::ARTIFACT);
    TF_ASSERT_OK(GetExistingNodes(fill_nodes_config, *store, exisiting_nodes));
    EXPECT_EQ(kNumberOfInsertedArtifacts, exisiting_nodes.size());
  }

  {
    std::vector<Node> exisiting_nodes;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::EXECUTION);
    TF_ASSERT_OK(GetExistingNodes(fill_nodes_config, *store, exisiting_nodes));
    EXPECT_EQ(kNumberOfInsertedExecutions, exisiting_nodes.size());
  }

  {
    std::vector<Node> exisiting_nodes;
    FillNodesConfig fill_nodes_config;
    fill_nodes_config.set_specification(FillNodesConfig::CONTEXT);
    TF_ASSERT_OK(GetExistingNodes(fill_nodes_config, *store, exisiting_nodes));
    EXPECT_EQ(kNumberOfInsertedContexts, exisiting_nodes.size());
  }
}

}  // namespace
}  // namespace ml_metadata
