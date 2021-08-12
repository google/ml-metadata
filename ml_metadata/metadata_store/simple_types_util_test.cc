/* Copyright 2021 Google LLC

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
#include "ml_metadata/metadata_store/simple_types_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {
namespace testing {
namespace {

TEST(SimpleTypesUtilTest, GetArtifactSystemTypeExtensionAndEnum) {
  ArtifactType type;
  std::vector<ArtifactType::SystemDefinedBaseType> want_enums = {
      ArtifactType::DATASET, ArtifactType::MODEL, ArtifactType::METRICS,
      ArtifactType::STATISTICS};
  for (const auto& want_enum : want_enums) {
    type.set_base_type(want_enum);
    SystemTypeExtension extension;
    ASSERT_EQ(absl::OkStatus(), GetSystemTypeExtension(type, extension));
    ArtifactType::SystemDefinedBaseType got_enum;
    ASSERT_EQ(absl::OkStatus(), GetSystemTypeEnum(extension, got_enum));
    EXPECT_EQ(want_enum, got_enum);
  }
}

TEST(SimpleTypesUtilTest, GetExecutionSystemTypeExtensionAndEnum) {
  ExecutionType type;
  std::vector<ExecutionType::SystemDefinedBaseType> want_enums = {
      ExecutionType::TRAIN, ExecutionType::EVALUATE, ExecutionType::DEPLOY,
      ExecutionType::TRANSFORM, ExecutionType::PROCESS};
  for (const auto& want_enum : want_enums) {
    type.set_base_type(want_enum);
    SystemTypeExtension extension;
    ASSERT_EQ(absl::OkStatus(), GetSystemTypeExtension(type, extension));
    ExecutionType::SystemDefinedBaseType got_enum;
    ASSERT_EQ(absl::OkStatus(), GetSystemTypeEnum(extension, got_enum));
    EXPECT_EQ(want_enum, got_enum);
  }
}

TEST(SimpleTypesUtilTest, TypeHasNoBaseType) {
  {
    // artifact type has no base type
    SystemTypeExtension extension;
    ArtifactType type;
    EXPECT_TRUE(absl::IsNotFound(GetSystemTypeExtension(type, extension)));
  }
  {
    // execution type has no base type
    SystemTypeExtension extension;
    ExecutionType type;
    EXPECT_TRUE(absl::IsNotFound(GetSystemTypeExtension(type, extension)));
  }
}

TEST(SimpleTypesUtilTest, BaseTypeIsUnset) {
  {
    // unset artifact type as base_type
    SystemTypeExtension extension;
    ArtifactType type;
    type.set_base_type(ArtifactType::UNSET);
    ASSERT_EQ(absl::OkStatus(), GetSystemTypeExtension(type, extension));
    EXPECT_TRUE(IsUnsetBaseType(extension));

    ArtifactType::SystemDefinedBaseType got_enum;
    ASSERT_EQ(absl::OkStatus(), GetSystemTypeEnum(extension, got_enum));
    EXPECT_THAT(ArtifactType::UNSET, got_enum);
  }
  {
    // unset execution type as base_type
    SystemTypeExtension extension;
    ExecutionType type;
    type.set_base_type(ExecutionType::UNSET);
    ASSERT_EQ(absl::OkStatus(), GetSystemTypeExtension(type, extension));
    EXPECT_TRUE(IsUnsetBaseType(extension));

    ExecutionType::SystemDefinedBaseType got_enum;
    ASSERT_EQ(absl::OkStatus(), GetSystemTypeEnum(extension, got_enum));
    EXPECT_EQ(ExecutionType::UNSET, got_enum);
  }
}

}  // namespace
}  // namespace testing
}  // namespace ml_metadata
