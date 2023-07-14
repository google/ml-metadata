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
#include "ml_metadata/query/filter_query_ast_resolver.h"

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {
namespace {

using ::testing::HasSubstr;
using ::testing::ValuesIn;

struct QueryTestCase {
  const char* query_string;
  const char* expected_error;

  // Note gtest has limitation to support parametrized type and value together.
  // We use a test_case_nodes in QueryTestCase to implement parameterized tests
  // and share test cases for both types {Artifact, Execution, Context} and
  // query string values.
  // Each {`query_string`, `expected_error`} is tested on all three node types
  // unless that node type is set to false in `test_case_nodes`.
  struct TestOnNodes {
    const bool artifact = true;
    const bool execution = true;
    const bool context = true;
  };
  const TestOnNodes test_case_nodes;
};

constexpr QueryTestCase::TestOnNodes artifact_only = {true, false, false};
constexpr QueryTestCase::TestOnNodes exclude_artifact = {false, true, true};
constexpr QueryTestCase::TestOnNodes exclude_context = {true, true, false};
constexpr QueryTestCase::TestOnNodes context_only = {false, false, true};
constexpr QueryTestCase::TestOnNodes execution_only = {false, true, false};

// A list of invalid queries with expected error sub-messages.
constexpr QueryTestCase kInValidTestQueriesWithErrors[] = {
    // invalid expression
    {"", "Syntax error"},
    // wrong logical operator
    {"a == 'a'", "Syntax error"},
    // `a` is a unknown column to mlmd filtering query
    {"a", "Unrecognized name"},
    // a boolean expression with wrong logical operator
    {"a = 'a'", "Unrecognized name"},
    // not a boolean expression
    {"'a'", "not a valid boolean expression"},
    // mentioned artifact's type_id, not a boolean expression
    {"type_id", "not a valid boolean expression"},
    // mentioned correct columns, but the value type is incorrect
    {"id = 'a'", "No matching signature for operator = for argument types"},
    {"type_id = 'a'",
     "No matching signature for operator = for argument types"},
    {"type = 1", "No matching signature for operator = for argument types"},
    {"uri = 1", "No matching signature for operator = for argument types",
     artifact_only},
    {"uri = 1", "Unrecognized name", exclude_artifact},
    {"create_time_since_epoch = 'a'",
     "No matching signature for operator = for argument types"},
    {"create_time_since_epoch = a", "Unrecognized name"},
    {"last_update_time_since_epoch = 'a'",
     "No matching signature for operator = for argument types"},
    {"last_update_time_since_epoch = a", "Unrecognized name"},
    {"name = 1", "No matching signature for operator = for argument types"},
    {"name = a", "Unrecognized name"},
    {"external_id = 1",
     "No matching signature for operator = for argument types"},
    {"external_id = a", "Unrecognized name"},
    // invalid context expressions
    {"contexts", "Unrecognized name"},
    {"contexts_0", "Unrecognized name"},
    {"contexts_0.foo", "Field name foo does not exist", exclude_context},
    {"contexts_0.id.", "Unexpected end of expression", exclude_context},
    {"contexts_0.id.bar", "Cannot access field bar", exclude_context},
    // invalid artifact expressions
    {"artifacts", "Unrecognized name"},
    {"artifacts_0", "Unrecognized name"},
    {"artifacts_0.foo", "Field name foo does not exist", context_only},
    {"artifacts_0.id.", "Unexpected end of expression", context_only},
    {"artifacts_0.id.bar", "Cannot access field bar", context_only},
    // invalid execution expressions
    {"executions", "Unrecognized name"},
    {"executions_0", "Unrecognized name"},
    {"executions_0.foo", "Field name foo does not exist", context_only},
    {"executions_0.id.", "Unexpected end of expression", context_only},
    {"executions_0.id.bar", "Cannot access field bar", context_only},
    // invalid property expressions
    {"properties", "Unrecognized name"},
    {"properties.foo", "Unrecognized name"},
    {"properties.bar.", "Unexpected end of expression"},
    {"properties.baz.int", "Field name int does not exist"},
    {"properties.0:b.int_value = 1", "Syntax error"},
    // invalid custom property expressions
    {"custom_properties", "Unrecognized name"},
    {"custom_properties.foo", "Unrecognized name"},
    {"custom_properties.bar.", "Unexpected end of expression"},
    {"custom_properties.baz.string", "Field name string does not exist"},
    {"custom_properties.0:b.string_value", "Syntax error"},
    // parent/child context expressions
    {"child_contexts", "Unrecognized name", context_only},
    {"child_contexts_0", "Unrecognized name", context_only},
    {"child_contexts_0.foo", "Field name foo does not exist", context_only},
    {"child_contexts_0.id.", "Unexpected end of expression", context_only},
    {"child_contexts_0.id.bar", "Cannot access field bar", context_only},
    {"parent_contexts", "Unrecognized name", context_only},
    {"parent_contexts_0", "Unrecognized name", context_only},
    {"parent_contexts_0.foo", "Field name foo does not exist", context_only},
    {"parent_contexts_0.id.", "Unexpected end of expression", context_only},
    {"parent_contexts_0.id.bar", "Cannot access field bar", context_only},
    // state related expressions
    {"state", "not a valid boolean expression", artifact_only},
    {"state = 'LIVE'",
     "No matching signature for operator = for argument types", artifact_only},
    {"state = 1", "Unsupported enum value specified", artifact_only},
    {"last_known_state", "not a valid boolean expression", execution_only},
    {"last_known_state = 'RUNNING'", "No matching signature for operator",
     execution_only},
    {"last_known_state = 1", "Unsupported enum value specified",
     execution_only},
    {"state = RUNNING", "Unsupported enum value specified", artifact_only},
    {"state IN RUNNING",
     "Expected a list of enum values enclosed in parentheses", artifact_only},
    {"last_known_state = LIVE", "Unsupported enum value specified",
     execution_only},
    {"last_known_state IN ()", "Syntax error", execution_only},
    {"last_known_state NOT IN (PENDING)", "Unsupported enum value specified",
     execution_only},
    {"events_0.field_1 = '1'", "Field name field_1 does not exist",
     exclude_context},
    {"events", "Unrecognized name", exclude_context},
    {"events_0.execution_id.", "Unexpected end of expression", exclude_context},
    {"events_0.type = SOME_ENUM",
     "Unsupported enum value specified in the query: SOME_ENUM",
     exclude_context},
    {"events_0.type IN (SOME_ENUM)",
     "Unsupported enum value specified in the query: SOME_ENUM",
     exclude_context},
    {"events_0.artifact_id = 1", "Field name artifact_id does not exist",
     artifact_only},
    {"events_0.execution_id = 1", "Field name execution_id does not exist",
     execution_only},
    {"events_0.artifact_id = 1", "Unrecognized name", context_only},
    {"events_0.execution_id = 1", "Unrecognized name", context_only}};

// A list of valid queries.
constexpr QueryTestCase kValidTestQueries[] = {
    // artifact attributes with boolean operators
    {"id != 1", ""},
    {"id IN (1, 3)", ""},
    {"type_id = 1", ""},
    {"type_id > 1", ""},
    {"type_id + 1 > 1", ""},
    {"type = 'Dataset'", ""},
    {"type IS NOT NULL", ""},
    {"uri = 'abc'", "", artifact_only},
    {"uri like 'abc'", "", artifact_only},
    {"uri LIKE 'abc'", "", artifact_only},
    {"uri IS NOT NULL", "", artifact_only},
    {"name = 'a'", ""},
    {"external_id = 'a'", ""},
    {"create_time_since_epoch = 1", ""},
    {"create_time_since_epoch > 1", ""},
    {"create_time_since_epoch + 1 > 1", ""},
    {"last_update_time_since_epoch = 1", ""},
    {"last_update_time_since_epoch > 1", ""},
    {"last_update_time_since_epoch + 1 > 1", ""},
    // context related expressions
    {"contexts_0.id = 1", "", exclude_context},
    {"contexts_2.name = '1'", "", exclude_context},
    {"contexts_abc.type = 'foo'", "", exclude_context},
    {"contexts_abc.create_time_since_epoch = 1", "", exclude_context},
    {"contexts_abc.last_update_time_since_epoch = 1", "", exclude_context},
    // artifact related expressions
    {"artifacts_0.id = 1", "", context_only},
    {"artifacts_2.name = '1'", "", context_only},
    {"artifacts_abc.type = 'foo'", "", context_only},
    {"artifacts_abc.create_time_since_epoch = 1", "", context_only},
    {"artifacts_abc.last_update_time_since_epoch = 1", "", context_only},
    // context related expressions
    {"executions_0.id = 1", "", context_only},
    {"executions_2.name = '1'", "", context_only},
    {"executions_abc.type = 'foo'", "", context_only},
    {"executions_abc.create_time_since_epoch = 1", "", context_only},
    {"executions_abc.last_update_time_since_epoch = 1", "", context_only},
    // logical operators
    {"type_id = 1 AND uri != 'abc'", "", artifact_only},
    {"type_id = 1 OR uri != 'abc'", "", artifact_only},
    {"NOT(type_id = 1)", ""},
    {"NOT(type_id = 1) AND (contexts_foo.type = 'bar' OR uri like 'x%')", "",
     artifact_only},
    // properties related expressions
    {"properties.0.int_value = 1", ""},
    {"properties.foo.string_value = '2'", ""},
    {"properties.bar_baz.double_value IS NOT NULL", ""},
    {"properties.qux.bool_value = true", ""},
    // custom properties related expressions
    {"custom_properties.0.double_value > 1.0", ""},
    {"custom_properties.1.string_value = '2'", ""},
    {"custom_properties.2.int_value != 0", ""},
    {"custom_properties.3.bool_value != false", ""},
    // the property name with arbitrary strings
    {"properties.`0:b`.int_value = 1", ""},
    {"custom_properties.`0 b`.string_value != '1'", ""},
    {"properties.`0:b`.int_value = 1 AND "
     "properties.foo.double_value = 1 AND "
     "custom_properties.`0 b`.string_value != '1'",
     ""},
    // mixed conditions with property and custom properties
    {"properties.0.int_value > 1 OR custom_properties.0.int_value > 1 ", ""},
    // mixed conditions with attributes, context, and properties
    {"type_id = 1 OR type = 'Dataset' AND contexts_0.name = '1' AND "
     "(properties.0.int_value > 1 OR custom_properties.0.int_value > 1) AND "
     "custom_properties.1.string_value IS NOT NULL",
     "", exclude_context},
    // parent/child context related expressions
    {"child_contexts_0.id = 1", "", context_only},
    {"child_contexts_2.name = '1'", "", context_only},
    {"child_contexts_abc.type = 'foo'", "", context_only},
    {"parent_contexts_0.id = 1", "", context_only},
    {"parent_contexts_2.name = '1'", "", context_only},
    {"parent_contexts_abc.type = 'foo'", "", context_only},
    // state related expressions
    {"state = PENDING", "", artifact_only},
    {"state=PENDING", "", artifact_only},
    {"state != PENDING", "", artifact_only},
    {"state = LIVE", "", artifact_only},
    {"state = PENDING AND state = LIVE", "", artifact_only},
    {"state IN (PENDING, LIVE)", "", artifact_only},
    {"state NOT IN (DELETED)", "", artifact_only},
    {"last_known_state = RUNNING", "", execution_only},
    {"last_known_state != RUNNING", "", execution_only},
    {"last_known_state = CANCELED", "", execution_only},
    {"last_known_state!=COMPLETE", "", execution_only},
    {"last_known_state IN (COMPLETE, RUNNING)", "", execution_only},
    {"state IS NULL", "", artifact_only},
    {"last_known_state IS NOT NULL", "", execution_only},
    // event related expressions
    {"events_0.execution_id = 1", "", artifact_only},
    {"events_0.artifact_id = 1", "", execution_only},
    {"events_0.milliseconds_since_epoch = 1", "", exclude_context},
    {"events_0.type = INPUT", "", execution_only},
    {"events_0.type=OUTPUT", "", execution_only},
    {"events_0.type IN (OUTPUT, DECLARED_OUTPUT)", "", execution_only},
};

class InValidQueryTest : public ::testing::TestWithParam<QueryTestCase> {
 protected:
  template <typename T>
  void VerifyInvalidQuery() {
    LOG(INFO) << "Testing invalid query string: " << GetParam().query_string;
    FilterQueryAstResolver<T> ast_resolver(GetParam().query_string);
    absl::Status s = ast_resolver.Resolve();
    EXPECT_TRUE(absl::IsInvalidArgument(s));
    EXPECT_THAT(std::string(s.message()), HasSubstr(GetParam().expected_error));
    EXPECT_EQ(ast_resolver.GetAst(), nullptr);
  }
};

TEST_P(InValidQueryTest, Artifact) {
  if (GetParam().test_case_nodes.artifact) {
    VerifyInvalidQuery<Artifact>();
  }
}

TEST_P(InValidQueryTest, Execution) {
  if (GetParam().test_case_nodes.execution) {
    VerifyInvalidQuery<Execution>();
  }
}

TEST_P(InValidQueryTest, Context) {
  if (GetParam().test_case_nodes.context) {
    VerifyInvalidQuery<Context>();
  }
}

class ValidQueryTest : public ::testing::TestWithParam<QueryTestCase> {
 protected:
  template <typename T>
  void VerifyValidQuery() {
    LOG(INFO) << "Testing valid query string: " << GetParam().query_string;
    FilterQueryAstResolver<T> ast_resolver(GetParam().query_string);
    EXPECT_EQ(absl::OkStatus(), ast_resolver.Resolve());
    EXPECT_NE(ast_resolver.GetAst(), nullptr);
  }
};

TEST_P(ValidQueryTest, Artifact) {
  if (GetParam().test_case_nodes.artifact) {
    VerifyValidQuery<Artifact>();
  }
}

TEST_P(ValidQueryTest, Execution) {
  if (GetParam().test_case_nodes.execution) {
    VerifyValidQuery<Execution>();
  }
}

TEST_P(ValidQueryTest, Context) {
  if (GetParam().test_case_nodes.context) {
    VerifyValidQuery<Context>();
  }
}

INSTANTIATE_TEST_SUITE_P(FilterQueryASTResolverTest, InValidQueryTest,
                         ValuesIn(kInValidTestQueriesWithErrors));

INSTANTIATE_TEST_SUITE_P(FilterQueryASTResolverTest, ValidQueryTest,
                         ValuesIn(kValidTestQueries));

}  // namespace
}  // namespace ml_metadata
