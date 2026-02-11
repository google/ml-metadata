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
#ifndef ML_METADATA_GOOGLE_QUERY_FILTER_QUERY_BUILDER_H
#define ML_METADATA_GOOGLE_QUERY_FILTER_QUERY_BUILDER_H

#include "zetasql/resolved_ast/sql_builder.h"
#include "absl/container/btree_map.h"
#include "absl/status/status.h"

namespace ml_metadata {

// DEPRECATED: This class and its associated ZetaSQL-based filter_query
// functionality is deprecated and will be removed in version 1.18.0.
// ZetaSQL dependency is being phased out from ML Metadata.
// Please migrate to alternative filtering approaches.
//
// FilterQueryBuilder is a ZetaSQL AST Visitor. It walks through a ZetaSQL
// boolean expression AST parsed from a filtering query string and generates
// FROM clauses and WHERE clauses which can be used by MLMD query executors. It
// can be instantiated with MLMD nodes types: Artifact, Execution and Context.
template <typename T>
class FilterQueryBuilder : public zetasql::SQLBuilder {
 public:
  FilterQueryBuilder();

  // Not copyable or movable
  FilterQueryBuilder(const FilterQueryBuilder&) = delete;
  FilterQueryBuilder& operator=(const FilterQueryBuilder&) = delete;

  // Returns the SQL string that can be used in MLMD node listing WHERE clause.
  std::string GetWhereClause();

  // Returns the SQL string that can be used in MLMD node listing FROM clause.
  // TODO(b/257334039): remove query_version parameter
  std::string GetFromClause(int64_t query_version);

  // The alias for the node table used in the query builder implementation.
  static constexpr absl::string_view kBaseTableAlias = "table_0";

  // Test-use only: to share the join rule details in the implementation.
  // Returns part of the join clause depending on the neighborhood.
  static std::string GetBaseNodeTable(absl::string_view base_alias);

  static std::string GetTypeJoinTable(absl::string_view base_alias,
                                      absl::string_view type_alias);

  static std::string GetContextJoinTable(absl::string_view base_alias,
                                         absl::string_view context_alias);

  // TODO(b/257334039): remove query_version parameter
  static std::string GetPropertyJoinTable(absl::string_view base_alias,
                                          absl::string_view property_alias,
                                          absl::string_view property_name,
                                          int64_t query_version);

  static std::string GetParentContextJoinTable(
      absl::string_view base_alias, absl::string_view parent_context_alias);

  static std::string GetChildContextJoinTable(
      absl::string_view base_alias, absl::string_view child_context_alias);

  // TODO(b/257334039): remove query_version parameter
  static std::string GetCustomPropertyJoinTable(
      absl::string_view base_alias, absl::string_view property_alias,
      absl::string_view property_name, int64_t query_version);

  static std::string GetEventJoinTable(absl::string_view base_alias,
                                       absl::string_view event_alias);

  // TODO(b/248836219): remove query_version parameter
  static std::string GetArtifactJoinTable(absl::string_view base_alias,
                                          absl::string_view artifact_alias,
                                          int64_t query_version);

  static std::string GetExecutionJoinTable(absl::string_view base_alias,
                                           absl::string_view execution_alias);

 protected:
  // Implementation details. API users need not look below.
  //
  // When navigating to the AST ExpressionColumn nodes, we generate a SQL query
  // segmentation by rewriting the parsed node name with proper alias, which is
  // later used in FROM clauses against the physical schema.
  //
  // In filtering query, the expression columns could be
  // a) node attributes of the main node
  // b) node neighborhood, such as other nodes, events, properties.
  //
  // For case a), we simply decorate the expression column with the main table
  // alias, for b) and c), we follow different rules by adding prefix to the
  // the join temporary tables.
  //
  // For example, the following AST ExpressionColumn node is about an attribute
  //
  // +-FunctionCall(ZetaSQL:$equal(INT64, INT64) -> BOOL)
  // | +-ExpressionColumn(type=INT64, name="type_id")
  // | +-Literal(type=INT64, value=6)
  //
  // The function rewrites `type_id` to `table_0.type_id`, where table_0 is the
  // table reference in the physical schema.
  //
  // For case b), the AST ExpressionColumn node is about a neighbor context:
  //
  // +-FunctionCall(ZetaSQL:$equal(STRING, STRING) -> BOOL)
  // | +-GetStructField
  // | | +-type=STRING
  // | | +-expr=
  // | | | +-ExpressionColumn(type=STRUCT<id INT64, name STRING, type STRING>,
  //                          name="contexts_pipeline")
  // | | +-field_idx=1
  // | +-Literal(type=STRING, value="taxi")
  //
  // The function substitutes the contexts_pipelines.type to table_1.type on
  // the context table to be joined.
  //
  // Returns UnimplementedError if any Struct of unknown neighbor is visited.
  absl::Status VisitResolvedExpressionColumn(
      const zetasql::ResolvedExpressionColumn* node) final;

 private:
  // For each mentioned expression column, we maintain a mapping of unique
  // table alias, which are used for FROM and WHERE clause query generation.
  enum class AtomType {
    ATTRIBUTE,
    CONTEXT,
    PROPERTY,
    CUSTOM_PROPERTY,
    PARENT_CONTEXT,
    CHILD_CONTEXT,
    EVENT,
    ARTIFACT,
    EXECUTION,
  };

  using JoinTableAlias =
      absl::btree_map<AtomType, absl::btree_map<std::string, std::string>>;

  // A routine maintains the mentioned_alias when constructing the query. It
  // auto-increase the alias_index_ when a `atom_type` and `concept_name` is
  // first seen when walking through the AST.
  std::string GetTableAlias(AtomType atom_type, absl::string_view concept_name);

  // The alias names of mentioned tables.
  JoinTableAlias mentioned_alias_;

  // Auto increased indices used as suffix of different table alias.
  // Should always be modified by using GetTableAlias.
  int alias_index_ = 0;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_GOOGLE_QUERY_FILTER_QUERY_BUILDER_H
