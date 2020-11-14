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

#include "ml_metadata/metadata_store/test_util.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {
namespace testing {
namespace {

TEST(TestUtilTest, BasicEqualsProto) {
  Artifact want_artifact;
  want_artifact.set_id(1);
  want_artifact.set_uri("some_uri");
  Artifact got_artifact = want_artifact;
  EXPECT_THAT(got_artifact, EqualsProto(want_artifact));

  Artifact other_artifact;
  EXPECT_THAT(other_artifact, ::testing::Not(EqualsProto(want_artifact)));
}

TEST(TestUtilTest, EqualsProtoWithIgnoreFields) {
  Artifact want_artifact;
  want_artifact.set_id(1);
  want_artifact.set_uri("some_uri");
  Artifact empty_artifact;
  EXPECT_THAT(empty_artifact,
              EqualsProto(want_artifact, /*ignore_fields=*/{"id", "uri"}));
}

TEST(TestUtilTest, EqualsProtoWithUnorderedPointwise) {
  std::vector<Artifact> want_artifacts(2);
  want_artifacts[0].set_id(1);
  want_artifacts[1].set_uri("2");
  std::vector<Artifact> got_artifacts(2);
  got_artifacts[0].set_uri("2");
  got_artifacts[1].set_id(1);
  EXPECT_THAT(want_artifacts, ::testing::UnorderedPointwise(
                                  EqualsProto<Artifact>(), got_artifacts));
  got_artifacts[0].set_uri("1");
  EXPECT_THAT(want_artifacts, ::testing::Not(::testing::UnorderedPointwise(
                                  EqualsProto<Artifact>(), got_artifacts)));
  EXPECT_THAT(
      want_artifacts,
      ::testing::UnorderedPointwise(
          EqualsProto<Artifact>(/*ignore_fields=*/{"uri"}), got_artifacts));
}


}  // namespace
}  // namespace testing
}  // namespace ml_metadata
