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

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "google/protobuf/util/message_differencer.h"
#include <gmock/gmock.h>
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "ml_metadata/metadata_store/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace ml_metadata {
namespace testing {

using tensorflow::protobuf::TextFormat;

// Simple implementation of a proto matcher comparing string representations.
//
// IMPORTANT: Only use this for protos whose textual representation is
// deterministic (that may not be the case for the map collection type).

class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const string& expected);
  explicit ProtoStringMatcher(const ::tensorflow::protobuf::Message& expected);

  template <typename Message>
  bool MatchAndExplain(const Message& p,
                       ::testing::MatchResultListener* /* listener */) const;

  void DescribeTo(::std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const string expected_;
};

template <typename T>
T CreateProto(const string& textual_proto) {
  T proto;
  CHECK(TextFormat::ParseFromString(textual_proto, &proto));
  return proto;
}

template <typename Message>
bool ProtoStringMatcher::MatchAndExplain(
    const Message& p, ::testing::MatchResultListener* /* listener */) const {
  return google::protobuf::util::MessageDifferencer::Equals(
      p, CreateProto<Message>(expected_));
}

// Polymorphic matcher to compare any two protos.
inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const string& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

// Polymorphic matcher to compare any two protos.
inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const ::tensorflow::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

// Parse input string as a protocol buffer.
template <typename T>
T ParseTextProtoOrDie(const string& input) {
  T result;
  CHECK(TextFormat::ParseFromString(input, &result))
      << "Failed to parse: " << input;
  return result;
}

}  // namespace testing
}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_TEST_UTIL_H_
