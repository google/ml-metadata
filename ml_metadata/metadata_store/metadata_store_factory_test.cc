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
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/metadata_store.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"


namespace ml_metadata {
namespace {
using testing::ParseTextProtoOrDie;

// Utility method to test CreateMetadataStore.
// Given a ConnectionConfig, it creates the store, puts and gets an
// artifact type.
void TestPutAndGetArtifactType(const ConnectionConfig& connection_config) {
  std::unique_ptr<MetadataStore> store;
  TF_ASSERT_OK(CreateMetadataStore(connection_config, &store));

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
  TF_ASSERT_OK(store->PutArtifactType(put_request, &put_response));
  ASSERT_TRUE(put_response.has_type_id());
  GetArtifactTypeRequest get_request =
      ParseTextProtoOrDie<GetArtifactTypeRequest>(
          R"(
            type_name: 'test_type2'
          )");
  GetArtifactTypeResponse get_response;
  TF_ASSERT_OK(store->GetArtifactType(get_request, &get_response));
  ArtifactType expected = put_request.artifact_type();
  expected.set_id(put_response.type_id());
  EXPECT_THAT(get_response.artifact_type(), testing::EqualsProto(expected))
      << "The type should be the same as the one given.";
}


TEST(MetadataStoreFactoryTest, CreateSQLiteMetadataStore) {
  ConnectionConfig connection_config;
  connection_config.mutable_sqlite();
  TestPutAndGetArtifactType(connection_config);
}

}  // namespace
}  // namespace ml_metadata
