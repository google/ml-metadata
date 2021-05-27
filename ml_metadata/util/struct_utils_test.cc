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
#include "ml_metadata/util/struct_utils.h"

#include <glog/logging.h>
#include "google/protobuf/struct.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ml_metadata/metadata_store/test_util.h"

namespace ml_metadata {
namespace {

using ::ml_metadata::testing::EqualsProto;

TEST(StructUtils, IsStructStringWorks) {
  EXPECT_TRUE(
      IsStructSerializedString("mlmd-struct::<serialized-string-goes-here>"));
  EXPECT_FALSE(
      IsStructSerializedString("mlmd-struct:<serialized-string-goes-here>"));
  EXPECT_FALSE(
      IsStructSerializedString("mlmd--struct::<serialized-string-goes-here>"));
  EXPECT_FALSE(IsStructSerializedString("<serialized-string-goes-here>"));
}

TEST(StructUtils, SerializeAndDeserializeStructWorks) {
  const auto struct_value =
      ml_metadata::testing::ParseTextProtoOrDie<google::protobuf::Struct>(R"pb(
        fields {
          key: "json number"
          value { number_value: 1234 }
        }
        fields {
          key: "json object"
          value {
            struct_value {
              fields {
                key: "nested json key"
                value { string_value: "string value" }
              }
            }
          }
        }
      )pb");

  const std::string serialized_value = StructToString(struct_value);
  google::protobuf::Struct got_value;
  ASSERT_EQ(absl::OkStatus(), StringToStruct(serialized_value, got_value));
  EXPECT_THAT(got_value, EqualsProto(struct_value));
}

TEST(StructUtils, DeserializeInvalidPrefix) {
  google::protobuf::Struct got_value;
  const absl::Status status = StringToStruct("mlmd-struct--", got_value);
  EXPECT_TRUE(absl::IsInvalidArgument(status));
}

TEST(StructUtils, DeserializeNonBase64) {
  google::protobuf::Struct got_value;
  const absl::Status status = StringToStruct("mlmd-struct::garbage", got_value);
  EXPECT_TRUE(absl::IsInvalidArgument(status));
}

TEST(StructUtils, DeserializeNonStructSerializedString) {
  const google::protobuf::Struct stored_value =
      ml_metadata::testing::ParseTextProtoOrDie<google::protobuf::Struct>(R"pb(
        fields {
          key: "json number"
          value { number_value: 1234 }
        })pb");
  google::protobuf::Struct got_value;
  const absl::Status status = StringToStruct(
      absl::StrCat("mlmd-struct::", stored_value.SerializeAsString()),
      got_value);
  EXPECT_TRUE(absl::IsInvalidArgument(status));
}

}  // namespace
}  // namespace ml_metadata
