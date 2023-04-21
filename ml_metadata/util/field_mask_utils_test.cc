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
#include "ml_metadata/util/field_mask_utils.h"

#include "google/protobuf/field_mask.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {
namespace {
using ::testing::UnorderedElementsAre;

TEST(FieldMaskUtils, GetArtifactPropertyNamesFromMaskSucceeds) {
  google::protobuf::FieldMask mask;
  mask.add_paths("external_id");
  mask.add_paths("name");
  mask.add_paths("properties.key1");
  mask.add_paths("properties.key2");

  Artifact artifact;
  // Mask:
  //   {"external_id", "name", "properties.key1",
  //    "properties.key2"}
  // Get by "properties"
  // Expected results:
  //   property_names = {"key1", "key2"}
  absl::StatusOr<absl::flat_hash_set<absl::string_view>> result =
      GetPropertyNamesFromMask(mask, /*is_custom_properties=*/false,
                               artifact.GetDescriptor());

  EXPECT_EQ(result.status(), absl::OkStatus());
  EXPECT_THAT(result.value(), UnorderedElementsAre("key1", "key2"));
}

TEST(FieldMaskUtils,
     GetNamesFromMaskWithPropertiesAndPropertyNamesReturnsInternalError) {
  google::protobuf::FieldMask mask;
  mask.add_paths("external_id");
  mask.add_paths("name");
  mask.add_paths("properties");
  mask.add_paths("properties.key1");

  Artifact artifact;
  // Mask:
  //   {"external_id", "name", "properties.key1", "properties"}
  // Get by "properties"
  // Expected results:
  //   INTERNAL error
  absl::StatusOr<absl::flat_hash_set<absl::string_view>> result =
      GetPropertyNamesFromMask(mask, /*is_custom_properties=*/false,
                               artifact.GetDescriptor());

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(result.status().message(),
                                "Cannot split property names"));
}

TEST(FieldMaskUtils,
     GetArtifactPropertyNamesAndCustomPropertyNamesFromMaskSucceeds) {
  google::protobuf::FieldMask mask;
  mask.add_paths("external_id");
  mask.add_paths("name");
  mask.add_paths("properties.key1");
  mask.add_paths("properties.key2");
  mask.add_paths("custom_properties.key3");
  mask.add_paths("custom_properties.key4");

  Artifact artifact;
  // Mask:
  //   {"external_id", "name", "properties.key1",
  //    "properties.key2","custom_properties.key3","custom_properties.key4"}
  // Get by "properties"
  // Expected results:
  //   property_names = {"key1", "key2"}
  absl::StatusOr<absl::flat_hash_set<absl::string_view>> result =
      GetPropertyNamesFromMask(mask, /*is_custom_properties=*/false,
                               artifact.GetDescriptor());

  EXPECT_EQ(result.status(), absl::OkStatus());
  EXPECT_THAT(result.value(), UnorderedElementsAre("key1", "key2"));

  // Get by "custom_properties"
  // Expected results:
  //   custom_property_names = {"key3", "key4"}
  result = GetPropertyNamesFromMask(mask, /*is_custom_properties=*/true,
                                    artifact.GetDescriptor());

  EXPECT_EQ(result.status(), absl::OkStatus());
  EXPECT_THAT(result.value(), UnorderedElementsAre("key3", "key4"));
}

TEST(FieldMaskUtils,
     GetPropertyNamesFromMaskWithNoPropertyNamesReturnsEmptySplitedMask) {
  google::protobuf::FieldMask mask;
  mask.add_paths("absent_field_name");

  Artifact artifact;
  // Mask: {"absent_field_name"}
  // Get by "properties"
  // Expected results:
  //   property_names = {}
  absl::StatusOr<absl::flat_hash_set<absl::string_view>> result =
      GetPropertyNamesFromMask(mask, /*is_custom_properties=*/false,
                               artifact.GetDescriptor());

  EXPECT_EQ(result.status(), absl::OkStatus());
  EXPECT_EQ(result.value().size(), 0);
}

TEST(FieldMaskUtils,
     GetPropertyNamesFromMaskWithDuplicatedFieldsReturnsDedupedPropertyNames) {
  google::protobuf::FieldMask mask;
  mask.add_paths("external_id");
  mask.add_paths("properties.key1");
  mask.add_paths("properties.key1");
  mask.add_paths("properties.key2");
  mask.add_paths("properties.key2");

  Artifact artifact;
  // Mask:
  //   {"external_id", "properties.key1","properties.key1",
  //    "properties.key2","properties.key2"}
  // Get by "properties"
  // Expected results:
  //   property_names = {"key1", "key2"}
  absl::StatusOr<absl::flat_hash_set<absl::string_view>> result =
      GetPropertyNamesFromMask(mask, /*is_custom_properties=*/false,
                               artifact.GetDescriptor());

  EXPECT_EQ(result.status(), absl::OkStatus());
  EXPECT_THAT(result.value(), UnorderedElementsAre("key1", "key2"));
}

TEST(FieldMaskUtils, GetFieldSubmaskFromMaskFiltersPropertiesAndInvalidPaths) {
  google::protobuf::FieldMask mask;
  mask.add_paths("external_id");
  mask.add_paths("properties.key1");
  mask.add_paths("properties.key2");
  mask.add_paths("custom_properties.key1");
  mask.add_paths("field_not_in_artifact");
  Artifact artifact;
  // Mask:
  //   {"external_id", "properties.key1","properties.key2",
  //    "custom_properties.key1", "field_not_in_artifact"}
  // Get field submask
  // Expected results:
  //   submask: "external_id"
  absl::StatusOr<google::protobuf::FieldMask> result =
      GetFieldsSubMaskFromMask(mask, artifact.GetDescriptor());
  EXPECT_EQ(result.status(), absl::OkStatus());
  EXPECT_THAT(result.value().paths(), UnorderedElementsAre("external_id"));
}

TEST(FieldMaskUtils, GetFromMaskWithNullDescriptorReturnsInvalidArgumentError) {
  google::protobuf::FieldMask mask;
  mask.add_paths("external_id");
  mask.add_paths("properties.key1");
  mask.add_paths("properties.key2");

  // Expected results for util functions:
  //   INVALID_ARTUMENT
  absl::StatusOr<absl::flat_hash_set<absl::string_view>>
      result_get_property_names = GetPropertyNamesFromMask(
          mask, /*is_custom_properties=*/false, nullptr);
  EXPECT_EQ(result_get_property_names.status().code(),
            absl::StatusCode::kInvalidArgument);
  absl::StatusOr<google::protobuf::FieldMask> result_get_fields_submask =
      GetFieldsSubMaskFromMask(mask, nullptr);
  EXPECT_EQ(result_get_fields_submask.status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(FieldMaskUtils, GetFromEmptyMaskReturnsEmptyResults) {
  google::protobuf::FieldMask mask;

  Artifact artifact;
  // Mask: {}
  // Get by "properties"
  // Expected results:
  //   property_names = {}
  // Get field submask
  // Expected results: {}
  absl::StatusOr<absl::flat_hash_set<absl::string_view>> result =
      GetPropertyNamesFromMask(mask, /*is_custom_properties=*/false,
                               artifact.GetDescriptor());

  EXPECT_EQ(result.status(), absl::OkStatus());
  EXPECT_EQ(result.value().size(), 0);

  absl::StatusOr<google::protobuf::FieldMask> submask_or_error =
      GetFieldsSubMaskFromMask(mask, artifact.GetDescriptor());
  EXPECT_EQ(submask_or_error.value().paths().size(), 0);
}
}  // namespace
}  // namespace ml_metadata
