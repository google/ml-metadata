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

#ifndef ML_METADATA_METADATA_STORE_TEST_UTIL_H_
#define ML_METADATA_METADATA_STORE_TEST_UTIL_H_

#include <string>
#include <vector>

#include "google/protobuf/util/message_differencer.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/platform/protobuf.h"

namespace ml_metadata {
namespace testing {

// Simple implementation of a proto matcher comparing string representations.
//
// IMPORTANT: Only use this for protos whose textual representation is
// deterministic (that may not be the case for the map collection type).

template <typename Message>
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(
      const Message& expected,
      const std::vector<std::string>& ignore_fields = {})
      : expected_(expected), ignore_fields_(ignore_fields) {}

  bool MatchAndExplain(const Message& p,
                       ::testing::MatchResultListener* /* listener */) const;

  void DescribeTo(::std::ostream* os) const { *os << expected_.DebugString(); }

  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_.DebugString();
  }

 private:
  const Message expected_;
  const std::vector<std::string>& ignore_fields_;
};

template <typename Message>
bool ProtoStringMatcher<Message>::MatchAndExplain(
    const Message& p, ::testing::MatchResultListener* /* listener */) const {
  google::protobuf::util::MessageDifferencer diff;
  for (const std::string& field_name : ignore_fields_) {
    const google::protobuf::FieldDescriptor* field_descriptor =
        p.descriptor()->FindFieldByName(field_name);
    diff.IgnoreField(field_descriptor);
  }
  return diff.Compare(p, expected_);
}

// Polymorphic matcher to compare any two protos.
template <typename Message>
inline ::testing::PolymorphicMatcher<ProtoStringMatcher<Message>> EqualsProto(
    const Message& x, const std::vector<std::string>& ignore_fields = {}) {
  return ::testing::MakePolymorphicMatcher(
      ProtoStringMatcher<Message>(x, ignore_fields));
}

// Parse input string as a protocol buffer.
// TODO(b/143236826) Drops TF dependency.
template <typename T>
T ParseTextProtoOrDie(const std::string& input) {
  T result;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(input, &result))
      << "Failed to parse: " << input;
  return result;
}

}  // namespace testing
}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_TEST_UTIL_H_
