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
#include <tuple>
#include <vector>

#include <glog/logging.h>
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace ml_metadata {
namespace testing {

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

template <typename Message, typename Tuple>
class TupleProtoMatcher : public ::testing::MatcherInterface<Tuple> {
 public:
  explicit TupleProtoMatcher(const std::vector<std::string>& ignore_fields)
      : ignore_fields_(ignore_fields) {}

  bool MatchAndExplain(
      Tuple args,
      ::testing::MatchResultListener* /* listener */) const override {
    return ProtoStringMatcher<Message>(std::get<0>(args), ignore_fields_)
        .MatchAndExplain(std::get<1>(args), nullptr);
  }

  void DescribeTo(::std::ostream* os) const override { *os << "are equal"; }

  void DescribeNegationTo(::std::ostream* os) const override {
    *os << "are not equal";
  }

 private:
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

// A polymorphic matcher for a 2-tuple where first.Equals(second) = true.
template <typename Message>
inline ::testing::Matcher<::testing::tuple<Message, Message>> EqualsProto(
    const std::vector<std::string>& ignore_fields = {}) {
  return ::testing::MakeMatcher(
      new TupleProtoMatcher<Message, ::testing::tuple<Message, Message>>(
          ignore_fields));
}

// Parse input string as a protocol buffer.
template <typename T>
T ParseTextProtoOrDie(const std::string& input) {
  T result;
  CHECK(google::protobuf::TextFormat::ParseFromString(input, &result))
      << "Failed to parse: " << input;
  return result;
}

// Checks if the id of a proto matches an expected value.
MATCHER(IdEquals, "") {
  auto& lhs = ::testing::get<0>(arg);
  auto& rhs = ::testing::get<1>(arg);
  return ::testing::Matches(lhs.id())(rhs);
}

}  // namespace testing
}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_TEST_UTIL_H_
