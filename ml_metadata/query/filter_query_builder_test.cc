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
#include "ml_metadata/query/filter_query_builder.h"

#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/query/filter_query_ast_resolver.h"

namespace ml_metadata {
namespace {

using ::testing::ValuesIn;

// A property mention consists of a tuple (base table alias, property name).
using PropertyMention = std::pair<absl::string_view, absl::string_view>;

// A tuple of (user-query, from clause, where clause)
struct QueryTupleTestCase {
  const std::string user_query;
  // The from clause depends on the base_table of the template Node type
  // (Artifact/Execution/Context). The `join_mentions` describes the expected
  // table alias of related neighbors.
  // Use GetFromClause<T> to test the resolved from_clause with the test case.
  struct MentionedNeighbors {
    std::vector<absl::string_view> types;
    std::vector<absl::string_view> contexts;
    std::vector<PropertyMention> properties;
    std::vector<PropertyMention> custom_properties;
    std::vector<absl::string_view> parent_contexts;
    std::vector<absl::string_view> child_contexts;
    std::vector<absl::string_view> events;
    std::vector<absl::string_view> artifacts;
    std::vector<absl::string_view> executions;
  };
  const MentionedNeighbors join_mentions;
  const std::string where_clause;

  // Note gtest has limitation to support parametrized type and value together.
  // We use a test_case_nodes in QueryTupleTestCase to implement parameterized
  // tests and share test cases for both types {Artifact, Execution, Context}
  // and query tuple values.
  // Each {`user_query`, `from_clause`, `where_clause`} is tested on all three
  // node types unless that node type is set to false in `test_case_nodes`.
  struct TestOnNodes {
    const bool artifact = true;
    const bool execution = true;
    const bool context = true;
  };
  const TestOnNodes test_case_nodes;

  // Utility method to test the resolved from clause with the testcase instance.
  // TODO(b/257334039): remove query_version parameter
  template <typename Node>
  std::string GetFromClause(int64_t query_version) const {
    absl::string_view base_alias = FilterQueryBuilder<Node>::kBaseTableAlias;
    std::string from_clause =
        FilterQueryBuilder<Node>::GetBaseNodeTable(base_alias);
    for (absl::string_view type_alias : join_mentions.types) {
      from_clause +=
          FilterQueryBuilder<Node>::GetTypeJoinTable(base_alias, type_alias);
    }
    for (absl::string_view context_alias : join_mentions.contexts) {
      from_clause += FilterQueryBuilder<Node>::GetContextJoinTable(
          base_alias, context_alias);
    }
    for (absl::string_view artifact_alias : join_mentions.artifacts) {
      // TODO(b/248836219): remove query_version parameter
      from_clause += FilterQueryBuilder<Node>::GetArtifactJoinTable(
          base_alias, artifact_alias, query_version);
    }
    for (absl::string_view execution_alias : join_mentions.executions) {
      from_clause += FilterQueryBuilder<Node>::GetExecutionJoinTable(
          base_alias, execution_alias);
    }
    for (const PropertyMention& property_mention : join_mentions.properties) {
      // TODO(b/257334039): remove query_version parameter
      from_clause += FilterQueryBuilder<Node>::GetPropertyJoinTable(
          base_alias, property_mention.first, property_mention.second,
          query_version);
    }
    for (const PropertyMention& property_mention :
         join_mentions.custom_properties) {
      // TODO(b/257334039): remove query_version parameter
      from_clause += FilterQueryBuilder<Node>::GetCustomPropertyJoinTable(
          base_alias, property_mention.first, property_mention.second,
          query_version);
    }
    for (absl::string_view parent_context_alias :
         join_mentions.parent_contexts) {
      from_clause += FilterQueryBuilder<Node>::GetParentContextJoinTable(
          base_alias, parent_context_alias);
    }
    for (absl::string_view child_context_alias : join_mentions.child_contexts) {
      from_clause += FilterQueryBuilder<Node>::GetChildContextJoinTable(
          base_alias, child_context_alias);
    }
    for (absl::string_view event_alias : join_mentions.events) {
      from_clause +=
          FilterQueryBuilder<Node>::GetEventJoinTable(base_alias, event_alias);
    }
    return from_clause;
  }
};

constexpr QueryTupleTestCase::TestOnNodes artifact_only = {true, false, false};
constexpr QueryTupleTestCase::TestOnNodes execution_only = {false, true, false};
constexpr QueryTupleTestCase::TestOnNodes exclude_context = {true, true, false};
constexpr QueryTupleTestCase::TestOnNodes context_only = {false, false, true};

// A list of utilities to write the mentioned tables for the test cases.
QueryTupleTestCase::MentionedNeighbors NoJoin() {
  return QueryTupleTestCase::MentionedNeighbors();
}

QueryTupleTestCase::MentionedNeighbors JoinWith(
    std::vector<absl::string_view> types = {},
    std::vector<absl::string_view> contexts = {},
    std::vector<PropertyMention> properties = {},
    std::vector<PropertyMention> custom_properties = {},
    std::vector<absl::string_view> parent_contexts = {},
    std::vector<absl::string_view> child_contexts = {},
    std::vector<absl::string_view> events = {},
    std::vector<absl::string_view> artifacts = {},
    std::vector<absl::string_view> executions = {}) {
  return {
      types,          contexts, properties, custom_properties, parent_contexts,
      child_contexts, events,   artifacts,  executions,
  };
}

QueryTupleTestCase::MentionedNeighbors JoinWithType(
    absl::string_view type_table_alias) {
  return JoinWith(/*types=*/{type_table_alias}, /*contexts=*/{});
}

QueryTupleTestCase::MentionedNeighbors JoinWithContexts(
    std::vector<absl::string_view> context_table_alias) {
  return JoinWith(/*types=*/{}, context_table_alias);
}

QueryTupleTestCase::MentionedNeighbors JoinWithProperty(
    absl::string_view property_table_alias, absl::string_view property_name) {
  return JoinWith(/*types=*/{}, /*contexts=*/{},
                  {{property_table_alias, property_name}});
}

QueryTupleTestCase::MentionedNeighbors JoinWithCustomProperty(
    absl::string_view property_table_alias, absl::string_view property_name) {
  return JoinWith(/*types=*/{}, /*contexts=*/{},
                  /*properties=*/{}, {{property_table_alias, property_name}});
}

QueryTupleTestCase::MentionedNeighbors JoinWithProperties(
    std::vector<PropertyMention> properties,
    std::vector<PropertyMention> custom_properties) {
  return JoinWith(/*types=*/{}, /*contexts=*/{}, properties, custom_properties);
}

QueryTupleTestCase::MentionedNeighbors JoinWithParentContexts(
    std::vector<absl::string_view> parent_context_table_alias) {
  return JoinWith(/*types=*/{}, /*contexts=*/{}, /*properties=*/{},
                  /*custom_properties=*/{}, parent_context_table_alias,
                  /*child_contexts=*/{});
}

QueryTupleTestCase::MentionedNeighbors JoinWithChildContexts(
    std::vector<absl::string_view> child_context_table_alias) {
  return JoinWith(/*types=*/{}, /*contexts=*/{}, /*properties=*/{},
                  /*custom_properties=*/{}, /*parent_contexts=*/{},
                  child_context_table_alias);
}

QueryTupleTestCase::MentionedNeighbors JoinWithEvents(
    std::vector<absl::string_view> events_alias) {
  return JoinWith(/*types=*/{}, /*contexts=*/{}, /*properties=*/{},
                  /*custom_properties=*/{}, /*parent_contexts=*/{},
                  /*child_contexts=*/{}, events_alias);
}

QueryTupleTestCase::MentionedNeighbors JoinWithArtifacts(
    std::vector<absl::string_view> artifacts_alias) {
  return JoinWith(/*types=*/{}, /*contexts=*/{}, /*properties=*/{},
                  /*custom_properties=*/{}, /*parent_contexts=*/{},
                  /*child_contexts=*/{}, /*events=*/{}, artifacts_alias);
}

QueryTupleTestCase::MentionedNeighbors JoinWithExecutions(
    std::vector<absl::string_view> executions_alias) {
  return JoinWith(/*types=*/{}, /*contexts=*/{}, /*properties=*/{},
                  /*custom_properties=*/{}, /*parent_contexts=*/{},
                  /*child_contexts=*/{}, /*events=*/{}, /*artifacts=*/{},
                  executions_alias);
}

std::vector<QueryTupleTestCase> GetTestQueryTuples() {
  return {
      // basic type attributes conditions
      {"type_id = 1", NoJoin(), "((table_0.type_id) = 1)"},
      {"NOT(type_id = 1)", NoJoin(), "(NOT ((table_0.type_id) = 1))"},
      {"type = 'foo'", JoinWithType("table_1"), "((table_1.type) = (\"foo\"))"},
      // artifact-only attributes
      {"uri like 'abc'", NoJoin(), "((table_0.uri) LIKE (\"abc\"))",
       artifact_only},
      {"state = LIVE AND state = DELETED", NoJoin(),
       "(((table_0.state) = 2) AND ((table_0.state) = 4))", artifact_only},
      {"state IN (LIVE, PENDING)", NoJoin(), "((table_0.state) IN (2, 1))",
       artifact_only},
      {"state NOT IN (DELETED)", NoJoin(),
       "((NOT ((table_0.state) IN (4))) OR ((table_0.state) IS NULL))",
       artifact_only},
      // execution-only attributes
      {"last_known_state = NEW OR last_known_state = COMPLETE", NoJoin(),
       "(((table_0.last_known_state) = 1) OR ((table_0.last_known_state) = 3))",
       execution_only},
      {"last_known_state IN (NEW, RUNNING)", NoJoin(),
       "((table_0.last_known_state) IN (1, 2))", execution_only},
      // mention context (the neighbor only applies to artifact/execution)
      {"contexts_0.id = 1", JoinWithContexts({"table_1"}), "((table_1.id) = 1)",
       exclude_context},
      {"contexts_0.name = 'properties.node.node'",
       JoinWithContexts({"table_1"}),
       "((table_1.name) = (\"properties.node.node\"))", exclude_context},
      // use multiple conditions on the same context
      {"contexts_0.id = 1 AND contexts_0.name LIKE 'foo%'",
       JoinWithContexts({"table_1"}),
       "(((table_1.id) = 1) AND ((table_1.name) LIKE (\"foo%\")))",
       exclude_context},
      // use multiple conditions(including date fields) on the same context
      {"contexts_0.id = 1 AND contexts_0.create_time_since_epoch > 1",
       JoinWithContexts({"table_1"}),
       "(((table_1.id) = 1) AND ((table_1.create_time_since_epoch) > 1))",
       exclude_context},
      // use multiple conditions on different contexts
      {"contexts_0.id = 1 AND contexts_1.id != 2",
       JoinWithContexts({"table_1", "table_2"}),
       "(((table_1.id) = 1) AND ((table_2.id) != 2))", exclude_context},
      // use multiple conditions on different contexts
      {"contexts_0.id = 1 AND contexts_0.last_update_time_since_epoch < 1 AND "
       "contexts_1.id != 2",
       JoinWithContexts({"table_1", "table_2"}),
       "(((table_1.id) = 1) AND ((table_1.last_update_time_since_epoch) < 1) "
       "AND ((table_2.id) != 2))",
       exclude_context},
      // mix attributes and context together
      {"type_id = 1 AND contexts_0.id = 1", JoinWithContexts({"table_1"}),
       "(((table_0.type_id) = 1) AND ((table_1.id) = 1))", exclude_context},
      // mix attributes (including type) and context together
      {"(type_id = 1 OR type != 'foo') AND contexts_0.id = 1",
       JoinWith(/*types=*/{"table_1"}, /*contexts=*/{"table_2"}),
       "((((table_0.type_id) = 1) OR ((table_1.type) != (\"foo\"))) AND "
       "((table_2.id) = 1))",
       exclude_context},
      // mention artifact (the neighbor only applies to context)
      {"artifacts_0.id = 1", JoinWithArtifacts({"table_1"}),
       "((table_1.id) = 1)", context_only},
      {"artifacts_0.uri like 'ab_c%'", JoinWithArtifacts({"table_1"}),
       "((table_1.uri) LIKE (\"ab_c%\"))", context_only},
      {"artifacts_0.state = LIVE", JoinWithArtifacts({"table_1"}),
       "((table_1.state) = 2)", context_only},
      {"artifacts_0.state IN (PENDING, LIVE)", JoinWithArtifacts({"table_1"}),
       "((table_1.state) IN (1, 2))", context_only},
      // use multiple conditions on the same artifact
      {"artifacts_0.id = 1 AND artifacts_0.name LIKE 'foo%'",
       JoinWithArtifacts({"table_1"}),
       "(((table_1.id) = 1) AND ((table_1.name) LIKE (\"foo%\")))",
       context_only},
      // use multiple conditions(including date fields) on the same artifact
      {"artifacts_0.id = 1 AND artifacts_0.create_time_since_epoch > 1",
       JoinWithArtifacts({"table_1"}),
       "(((table_1.id) = 1) AND ((table_1.create_time_since_epoch) > 1))",
       context_only},
      // use multiple conditions on different artifacts
      {"artifacts_0.id = 1 AND artifacts_1.id != 2",
       JoinWithArtifacts({"table_1", "table_2"}),
       "(((table_1.id) = 1) AND ((table_2.id) != 2))", context_only},
      // use multiple conditions on different artifacts
      {"artifacts_0.id = 1 AND artifacts_0.last_update_time_since_epoch < 1 "
       "AND artifacts_1.id != 2",
       JoinWithArtifacts({"table_1", "table_2"}),
       "(((table_1.id) = 1) AND ((table_1.last_update_time_since_epoch) < 1) "
       "AND ((table_2.id) != 2))",
       context_only},
      // mix attributes and artifact together
      {"type_id = 1 AND artifacts_0.id = 1", JoinWithArtifacts({"table_1"}),
       "(((table_0.type_id) = 1) AND ((table_1.id) = 1))", context_only},
      // mix attributes (including type) and artifact together
      {"(type_id = 1 OR type != 'foo') AND artifacts_0.id = 1",
       JoinWith(/*types=*/{"table_1"}, {}, {}, {}, {}, {}, {},
                /*artifacts=*/{"table_2"}),
       "((((table_0.type_id) = 1) OR ((table_1.type) != (\"foo\"))) AND "
       "((table_2.id) = 1))",
       context_only},
      // mention execution (the neighbor only applies to context)
      {"executions_0.id = 1", JoinWithExecutions({"table_1"}),
       "((table_1.id) = 1)", context_only},
      // use multiple conditions on the same execution
      {"executions_0.id = 1 AND executions_0.name LIKE 'foo%'",
       JoinWithExecutions({"table_1"}),
       "(((table_1.id) = 1) AND ((table_1.name) LIKE (\"foo%\")))",
       context_only},
      // use multiple conditions(including date fields) on the same execution
      {"executions_0.id = 1 AND executions_0.create_time_since_epoch > 1",
       JoinWithExecutions({"table_1"}),
       "(((table_1.id) = 1) AND ((table_1.create_time_since_epoch) > 1))",
       context_only},
      // use multiple conditions on different executions
      {"executions_0.id = 1 AND executions_1.id != 2",
       JoinWithExecutions({"table_1", "table_2"}),
       "(((table_1.id) = 1) AND ((table_2.id) != 2))", context_only},
      // use multiple conditions on different executions
      {"executions_0.id = 1 AND executions_0.last_update_time_since_epoch < 1 "
       "AND "
       "executions_1.id != 2",
       JoinWithExecutions({"table_1", "table_2"}),
       "(((table_1.id) = 1) AND ((table_1.last_update_time_since_epoch) < 1) "
       "AND ((table_2.id) != 2))",
       context_only},
      // mix attributes and execution together
      {"type_id = 1 AND executions_0.id = 1", JoinWithExecutions({"table_1"}),
       "(((table_0.type_id) = 1) AND ((table_1.id) = 1))", context_only},
      // mix attributes (including type) and execution together
      {"(type_id = 1 OR type != 'foo') AND executions_0.id = 1",
       JoinWith(/*types=*/{"table_1"}, {}, {}, {}, {}, {}, {}, {},
                /*executions=*/{"table_2"}),
       "((((table_0.type_id) = 1) OR ((table_1.type) != (\"foo\"))) AND "
       "((table_2.id) = 1))",
       context_only},
      {"executions_0.last_known_state = COMPLETE",
       JoinWithExecutions({"table_1"}), "((table_1.last_known_state) = 3)",
       context_only},
      {"executions_0.last_known_state IN (COMPLETE, RUNNING)",
       JoinWithExecutions({"table_1"}),
       "((table_1.last_known_state) IN (3, 2))", context_only},
      // mention properties
      {"properties.p0.int_value = 1", JoinWithProperty("table_1", "p0"),
       "((table_1.int_value) = 1)"},
      // properties with backquoted names
      {"properties.`0:b`.int_value = 1", JoinWithProperty("table_1", "0:b"),
       "((table_1.int_value) = 1)"},
      {"custom_properties.`0 b`.string_value != '1'",
       JoinWithCustomProperty("table_1", "0 b"),
       "((table_1.string_value) != (\"1\"))"},
      {"properties.`0:b`.int_value = 1 AND "
       "properties.foo.double_value > 1 AND "
       "custom_properties.`0 b`.string_value != '1'",
       JoinWithProperties(
           /*properties=*/{{"table_1", "0:b"}, {"table_2", "foo"}},
           /*custom_properties=*/{{"table_3", "0 b"}}),
       "(((table_1.int_value) = 1) AND ((table_2.double_value) > (1.0)) AND "
       "((table_3.string_value) != (\"1\")))"},
      // use multiple conditions on the same property
      {"properties.p0.int_value = 1 OR properties.p0.string_value = '1' ",
       JoinWithProperty("table_1", "p0"),
       "(((table_1.int_value) = 1) OR ((table_1.string_value) = (\"1\")))"},
      // mention property and custom property with the same property name
      {"properties.p0.int_value > 1 OR custom_properties.p0.int_value > 1",
       JoinWithProperties(/*properties=*/{{"table_1", "p0"}},
                          /*custom_properties=*/{{"table_2", "p0"}}),
       "(((table_1.int_value) > 1) OR ((table_2.int_value) > 1))"},
      // use multiple properties and custom properties
      {"(properties.p0.int_value > 1 OR custom_properties.p0.int_value > 1) "
       "AND "
       "properties.p1.double_value > 0.95 AND "
       "custom_properties.p2.string_value = 'name'",
       JoinWithProperties(
           /*properties=*/{{"table_1", "p0"}, {"table_3", "p1"}},
           /*custom_properties=*/{{"table_2", "p0"}, {"table_4", "p2"}}),
       "((((table_1.int_value) > 1) OR ((table_2.int_value) > 1)) AND "
       "((table_3.double_value) > (0.95)) AND "
       "((table_4.string_value) = (\"name\")))"},
      // use attributes, contexts, properties and custom properties
      {"type = 'dataset' AND "
       "(contexts_0.name = 'my_run' AND contexts_0.type = 'exp') AND "
       "(properties.p0.int_value > 1 OR custom_properties.p1.double_value > "
       "0.9)",
       JoinWith(/*types=*/{"table_1"},
                /*contexts=*/{"table_2"},
                /*properties=*/{{"table_3", "p0"}},
                /*custom_properties=*/{{"table_4", "p1"}}),
       "(((table_1.type) = (\"dataset\")) AND (((table_2.name) = (\"my_run\")) "
       "AND ((table_2.type) = (\"exp\"))) AND (((table_3.int_value) > 1) OR "
       "((table_4.double_value) > (0.9))))",
       exclude_context},
      // Parent context queries.
      // mention context (the neighbor only applies to contexts)
      {"parent_contexts_0.id = 1", JoinWithParentContexts({"table_1"}),
       "((table_1.id) = 1)", context_only},
      // use multiple conditions on the same parent context
      {"parent_contexts_0.id = 1 AND parent_contexts_0.name LIKE 'foo%'",
       JoinWithParentContexts({"table_1"}),
       "(((table_1.id) = 1) AND ((table_1.name) LIKE (\"foo%\")))",
       context_only},
      // use multiple conditions on different parent contexts
      {"parent_contexts_0.id = 1 AND parent_contexts_1.id != 2",
       JoinWithParentContexts({"table_1", "table_2"}),
       "(((table_1.id) = 1) AND ((table_2.id) != 2))", context_only},
      // // mix attributes and parent context together
      {"type_id = 1 AND parent_contexts_0.id = 1",
       JoinWithParentContexts({"table_1"}),
       "(((table_0.type_id) = 1) AND ((table_1.id) = 1))", context_only},
      // mix attributes (including type) and parent context together
      {"(type_id = 1 OR type != 'foo') AND parent_contexts_0.id = 1",
       JoinWith(/*types=*/{"table_1"}, /*contexts=*/{}, /*properties=*/{},
                /*custom_properties=*/{},
                /*parent_contexts=*/{"table_2"}),
       "((((table_0.type_id) = 1) OR ((table_1.type) != (\"foo\"))) AND "
       "((table_2.id) = 1))",
       context_only},
      // use attributes, parent contexts, properties and custom properties
      {"type = 'pipeline_run' AND (properties.p0.int_value > 1 OR "
       "custom_properties.p1.double_value > 0.9) AND (parent_contexts_0.name = "
       "'pipeline_context' AND parent_contexts_0.type = 'pipeline')",
       JoinWith(/*types=*/{"table_1"},
                /*contexts=*/{},
                /*properties=*/{{"table_2", "p0"}},
                /*custom_properties=*/{{"table_3", "p1"}},
                /*parent_contexts=*/{"table_4"}),
       "(((table_1.type) = (\"pipeline_run\")) AND (((table_2.int_value) > 1) "
       "OR ((table_3.double_value) > (0.9))) AND (((table_4.name) = "
       "(\"pipeline_context\")) AND ((table_4.type) = (\"pipeline\"))))",
       context_only},
      // Child context queries.
      // mention context (the neighbor only applies to contexts)
      {"child_contexts_0.id = 1", JoinWithChildContexts({"table_1"}),
       "((table_1.id) = 1)", context_only},
      // use multiple conditions on the same child context
      {"child_contexts_0.id = 1 AND child_contexts_0.name LIKE 'foo%'",
       JoinWithChildContexts({"table_1"}),
       "(((table_1.id) = 1) AND ((table_1.name) LIKE (\"foo%\")))",
       context_only},
      // use multiple conditions on different child contexts
      {"child_contexts_0.id = 1 AND child_contexts_1.id != 2",
       JoinWithChildContexts({"table_1", "table_2"}),
       "(((table_1.id) = 1) AND ((table_2.id) != 2))", context_only},
      // // mix attributes and child context together
      {"type_id = 1 AND child_contexts_0.id = 1",
       JoinWithChildContexts({"table_1"}),
       "(((table_0.type_id) = 1) AND ((table_1.id) = 1))", context_only},
      // mix attributes (including type) and child context together
      {"(type_id = 1 OR type != 'foo') AND child_contexts_0.id = 1",
       JoinWith(/*types=*/{"table_1"}, /*contexts=*/{}, /*properties=*/{},
                /*custom_properties=*/{}, /*parent_contexts=*/{},
                /*child_contexts=*/{"table_2"}),
       "((((table_0.type_id) = 1) OR ((table_1.type) != (\"foo\"))) AND "
       "((table_2.id) = 1))",
       context_only},
      // use attributes, child contexts, properties and custom properties
      {"type = 'pipeline' AND (properties.p0.int_value > 1 OR "
       "custom_properties.p1.double_value > 0.9) AND (child_contexts_0.name = "
       "'pipeline_run' AND child_contexts_0.type = 'runs')",
       JoinWith(/*types=*/{"table_1"},
                /*contexts=*/{},
                /*properties=*/{{"table_2", "p0"}},
                /*custom_properties=*/{{"table_3", "p1"}},
                /*parent_contexts=*/{},
                /*child_contexts=*/{"table_4"}),
       "(((table_1.type) = (\"pipeline\")) AND (((table_2.int_value) > 1) "
       "OR ((table_3.double_value) > (0.9))) AND (((table_4.name) = "
       "(\"pipeline_run\")) AND ((table_4.type) = (\"runs\"))))",
       context_only},
      // use attributes, parent context, child contexts, properties and custom
      // properties
      {"type = 'pipeline' AND (properties.p0.int_value > 1 OR "
       "custom_properties.p1.double_value > 0.9) AND (parent_contexts_0.name = "
       "'parent_context1' AND parent_contexts_0.type = 'parent_context_type') "
       "AND (child_contexts_0.name = 'pipeline_run' AND child_contexts_0.type "
       "= 'runs')",
       JoinWith(/*types=*/{"table_1"},
                /*contexts=*/{},
                /*properties=*/{{"table_2", "p0"}},
                /*custom_properties=*/{{"table_3", "p1"}},
                /*parent_contexts=*/{"table_4"},
                /*child_contexts=*/{"table_5"}),
       "(((table_1.type) = (\"pipeline\")) AND (((table_2.int_value) > 1) "
       "OR ((table_3.double_value) > (0.9))) AND (((table_4.name) = "
       "(\"parent_context1\")) AND ((table_4.type) = "
       "(\"parent_context_type\"))) AND (((table_5.name) = (\"pipeline_run\")) "
       "AND ((table_5.type) = (\"runs\"))))",
       context_only},
      {"events_0.execution_id = 1", JoinWithEvents({"table_1"}),
       "((table_1.execution_id) = 1)", artifact_only},
      {"events_0.type = INPUT", JoinWithEvents({"table_1"}),
       "((table_1.type) = 3)", exclude_context},
      {"events_0.type = INPUT OR events_0.type = OUTPUT",
       JoinWithEvents({"table_1"}),
       "(((table_1.type) = 3) OR ((table_1.type) = 4))", exclude_context},
      {"events_0.type IN (INPUT, DECLARED_INPUT)", JoinWithEvents({"table_1"}),
       "((table_1.type) IN (3, 2))", exclude_context},
      {"uri = 'http://some_path' AND events_0.type = INPUT",
       JoinWithEvents({"table_1"}),
       "(((table_0.uri) = (\"http://some_path\")) AND ((table_1.type) = 3))",
       artifact_only}};
}

class SQLGenerationTest : public ::testing::TestWithParam<QueryTupleTestCase> {
 protected:
  template <typename T>
  // TODO(b/257334039): remove query_version parameter
  void VerifyQueryTuple(int64_t query_version) {
    LOG(INFO) << "Testing valid query string: " << GetParam().user_query;
    FilterQueryAstResolver<T> ast_resolver(GetParam().user_query);
    ASSERT_EQ(absl::OkStatus(), ast_resolver.Resolve());
    ASSERT_NE(ast_resolver.GetAst(), nullptr);
    FilterQueryBuilder<T> query_builder;
    ASSERT_EQ(absl::OkStatus(), ast_resolver.GetAst()->Accept(&query_builder));
    // Ensures the base table alias constant does not violate the test strings
    // used in the expected where clause.
    ASSERT_EQ(FilterQueryBuilder<T>::kBaseTableAlias, "table_0");
    EXPECT_EQ(query_builder.GetFromClause(query_version),
              GetParam().GetFromClause<T>(query_version));
    EXPECT_EQ(query_builder.GetWhereClause(), GetParam().where_clause);
  }
};

TEST_P(SQLGenerationTest, Artifact) {
  if (GetParam().test_case_nodes.artifact) {
    VerifyQueryTuple<Artifact>(/*query_version=*/10);
  }
}

// TODO(b/257334039): cleanup after migration to v10+
TEST_P(SQLGenerationTest, ArtifactV7) {
  if (GetParam().test_case_nodes.artifact) {
    VerifyQueryTuple<Artifact>(/*query_version=*/7);
  }
}
TEST_P(SQLGenerationTest, ArtifactV8) {
  if (GetParam().test_case_nodes.artifact) {
    VerifyQueryTuple<Artifact>(/*query_version=*/8);
  }
}
TEST_P(SQLGenerationTest, ArtifactV9) {
  if (GetParam().test_case_nodes.artifact) {
    VerifyQueryTuple<Artifact>(/*query_version=*/9);
  }
}

TEST_P(SQLGenerationTest, Execution) {
  if (GetParam().test_case_nodes.execution) {
    VerifyQueryTuple<Execution>(/*query_version=*/10);
  }
}

// TODO(b/257334039): cleanup after migration to v10+
TEST_P(SQLGenerationTest, ExecutionV7) {
  if (GetParam().test_case_nodes.execution) {
    VerifyQueryTuple<Execution>(/*query_version=*/7);
  }
}
TEST_P(SQLGenerationTest, ExecutionV8) {
  if (GetParam().test_case_nodes.execution) {
    VerifyQueryTuple<Execution>(/*query_version=*/8);
  }
}
TEST_P(SQLGenerationTest, ExecutionV9) {
  if (GetParam().test_case_nodes.execution) {
    VerifyQueryTuple<Execution>(/*query_version=*/9);
  }
}

TEST_P(SQLGenerationTest, Context) {
  if (GetParam().test_case_nodes.context) {
    VerifyQueryTuple<Context>(/*query_version=*/10);
  }
}

// TODO(b/257334039): cleanup after migration to v10+
TEST_P(SQLGenerationTest, ContextV7) {
  if (GetParam().test_case_nodes.context) {
    VerifyQueryTuple<Context>(/*query_version=*/7);
  }
}
TEST_P(SQLGenerationTest, ContextV8) {
  if (GetParam().test_case_nodes.context) {
    VerifyQueryTuple<Context>(/*query_version=*/8);
  }
}
TEST_P(SQLGenerationTest, ContextV9) {
  if (GetParam().test_case_nodes.context) {
    VerifyQueryTuple<Context>(/*query_version=*/9);
  }
}

INSTANTIATE_TEST_SUITE_P(FilterQueryBuilderTest, SQLGenerationTest,
                         ValuesIn(GetTestQueryTuples()));

}  // namespace
}  // namespace ml_metadata
