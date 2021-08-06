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

#include "zetasql/public/analyzer.h"
#include "zetasql/public/simple_catalog.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/util/return_utils.h"
#include "re2/re2.h"

namespace ml_metadata {
namespace {

using ::zetasql::types::DoubleType;
using ::zetasql::types::Int64Type;
using ::zetasql::types::StringType;

// A regular expression that for mentioned contexts in query.
constexpr char kContextRE[] = "\\b(contexts_[[:word:]]+)\\.";
constexpr char kPropertyRE[] = "\\bproperties\\.(:?([[:word:]]+)|(`[^`]+`))\\.";
constexpr char kCustomPropertyRE[] =
    "\\bcustom_properties\\.(:?([[:word:]]+)|(`[^`]+`))\\.";
constexpr char kChildContextRE[] = "\\b(child_contexts_[[:word:]]+)\\.";
constexpr char kParentContextRE[] = "\\b(parent_contexts_[[:word:]]+)\\.";
constexpr char kArtifactStatePredicateRE[] =
    "\\b(state)[[:space:]]*(=|!=)[[:space:]]*([[:word:]]+)\\b";
constexpr char kExecutionStatePredicateRE[] =
    "\\b(last_known_state)[[:space:]]*(=|!=)[[:space:]]*([[:word:]]+)\\b";

// Parses query for state predicates e.g. state = LIVE and re-writes the query
// into form state = <int> based on the `state_value_mapping` provided by the
// caller. The `state_regex` provides the expected state predicate in the query.
zetasql_base::StatusOr<std::string> ParseStatePredicateAndTransformQuery(
    absl::string_view query_string, absl::string_view state_regex,
    const absl::flat_hash_map<std::string, int>& state_value_mapping) {
  std::string rewritten_query = std::string(query_string);
  std::string state_string_literal, operator_literal, value_literal;
  while (RE2::FindAndConsume(&query_string, state_regex, &state_string_literal,
                             &operator_literal, &value_literal)) {
    if (!state_value_mapping.contains(value_literal)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported state predicate specified in the query: ",
                       value_literal));
    }

    std::string query_subtitute;
    if (operator_literal == "!=") {
      query_subtitute = absl::Substitute(" (($0 $1 $2) OR ($0 IS NULL)) ",
                                         state_string_literal, operator_literal,
                                         state_value_mapping.at(value_literal));
    } else {
      query_subtitute =
          absl::Substitute(" $0 $1 $2 ", state_string_literal, operator_literal,
                           state_value_mapping.at(value_literal));
    }

    if (!RE2::GlobalReplace(&rewritten_query, state_regex, query_subtitute)) {
      return absl::InternalError(absl::Substitute(
          "Query cannot be rewritten successfully for matched state predicate: "
          "$0 $1 $2",
          state_string_literal, operator_literal, value_literal));
    }
  }

  return rewritten_query;
}

zetasql_base::StatusOr<std::string> AddArtifactStateAndTransformQuery(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts) {
  MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn("state", Int64Type()));
  static const auto& state_value_mapping =
      *new absl::flat_hash_map<std::string, int>(
          {{Artifact::State_Name(Artifact::UNKNOWN), Artifact::UNKNOWN},
           {Artifact::State_Name(Artifact::PENDING), Artifact::PENDING},
           {Artifact::State_Name(Artifact::LIVE), Artifact::LIVE},
           {Artifact::State_Name(Artifact::MARKED_FOR_DELETION),
            Artifact::MARKED_FOR_DELETION},
           {Artifact::State_Name(Artifact::DELETED), Artifact::DELETED}});

  return ParseStatePredicateAndTransformQuery(
      query_string, kArtifactStatePredicateRE, state_value_mapping);
}

zetasql_base::StatusOr<std::string> AddExecutionLastKnownStateAndTransformQuery(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts) {
  MLMD_RETURN_IF_ERROR(
      analyzer_opts.AddExpressionColumn("last_known_state", Int64Type()));
  static const auto& state_value_mapping =
      *new absl::flat_hash_map<std::string, int>(
          {{Execution::State_Name(Execution::UNKNOWN), Execution::UNKNOWN},
           {Execution::State_Name(Execution::NEW), Execution::NEW},
           {Execution::State_Name(Execution::RUNNING), Execution::RUNNING},
           {Execution::State_Name(Execution::COMPLETE), Execution::COMPLETE},
           {Execution::State_Name(Execution::FAILED), Execution::FAILED},
           {Execution::State_Name(Execution::CACHED), Execution::CACHED},
           {Execution::State_Name(Execution::CANCELED), Execution::CANCELED}});

  return ParseStatePredicateAndTransformQuery(
      query_string, kExecutionStatePredicateRE, state_value_mapping);
}

// Adds a list of columns corresponding to the node attributes which are
// allowed in the query to the analyzer. MLMD nodes share a list of common
// attributes, such as id, type and name, while each node type has its own
// specific fields, e.g., Artifact.uri, Execution.last_known_state.
template <typename T>
zetasql_base::StatusOr<std::string> AddSpecificAttributesAndTransformQuery(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts) {
  return std::string(query_string);
}

template <>
zetasql_base::StatusOr<std::string> AddSpecificAttributesAndTransformQuery<Artifact>(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts) {
  MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn("uri", StringType()));
  return AddArtifactStateAndTransformQuery(query_string, analyzer_opts);
}

template <>
zetasql_base::StatusOr<std::string> AddSpecificAttributesAndTransformQuery<Execution>(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts) {
  return AddExecutionLastKnownStateAndTransformQuery(query_string,
                                                     analyzer_opts);
}

template <typename T>
zetasql_base::StatusOr<std::string> AddAttributesAndTransformQuery(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts) {
  // TODO(b/145945460) Consider to expose configurable names to extend ast.
  MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn("id", Int64Type()));
  MLMD_RETURN_IF_ERROR(
      analyzer_opts.AddExpressionColumn("type_id", Int64Type()));
  MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn("type", StringType()));
  MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn(
      "create_time_since_epoch", Int64Type()));
  MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn(
      "last_update_time_since_epoch", Int64Type()));
  MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn("name", StringType()));
  return AddSpecificAttributesAndTransformQuery<T>(query_string, analyzer_opts);
}

// Adds neighborhood nodes allowed in the query to the analyzer as StructType
// columns. For example, for artifact and execution, 1-hop contexts and events
// are supported in the filtering query; for context, the attributed artifacts
// and associated executions can be used.
template <typename T>
absl::Status AddNeighborhoodNodes(absl::string_view query_string,
                                  zetasql::AnalyzerOptions& analyzer_opts,
                                  zetasql::TypeFactory& type_factory);

// Adds the Contexts used in the query to the analyzer as StructType column.
absl::Status AddContextsImpl(absl::string_view query_string,
                             const absl::string_view context_prefix,
                             zetasql::AnalyzerOptions& analyzer_opts,
                             zetasql::TypeFactory& type_factory) {
  RE2 context_re(context_prefix);
  std::string matched_context;
  absl::flat_hash_set<std::string> mentioned_contexts;
  while (RE2::FindAndConsume(&query_string, context_re, &matched_context)) {
    // Each mentioned context is defined only once.
    if (mentioned_contexts.contains(matched_context)) {
      continue;
    }
    mentioned_contexts.insert(matched_context);
    const zetasql::Type* context_struct_type;
    // TODO(b/145945460) Adds other context proto message fields.
    MLMD_RETURN_IF_ERROR(type_factory.MakeStructType(
        {{"id", Int64Type()},
         {"name", StringType()},
         {"type", StringType()},
         {"create_time_since_epoch", Int64Type()},
         {"last_update_time_since_epoch", Int64Type()}},
        &context_struct_type));
    MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn(
        matched_context, context_struct_type));
  }
  return absl::OkStatus();
}

absl::Status AddContexts(absl::string_view query_string,
                         zetasql::AnalyzerOptions& analyzer_opts,
                         zetasql::TypeFactory& type_factory) {
  return AddContextsImpl(query_string, kContextRE, analyzer_opts, type_factory);
}
absl::Status AddParentContexts(absl::string_view query_string,
                               zetasql::AnalyzerOptions& analyzer_opts,
                               zetasql::TypeFactory& type_factory) {
  return AddContextsImpl(query_string, kParentContextRE, analyzer_opts,
                         type_factory);
}

absl::Status AddChildContexts(absl::string_view query_string,
                              zetasql::AnalyzerOptions& analyzer_opts,
                              zetasql::TypeFactory& type_factory) {
  return AddContextsImpl(query_string, kChildContextRE, analyzer_opts,
                         type_factory);
}

template <>
absl::Status AddNeighborhoodNodes<Artifact>(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory) {
  // TODO(b/145945460) Support events for artifacts.
  MLMD_RETURN_IF_ERROR(AddContexts(query_string, analyzer_opts, type_factory));
  return absl::OkStatus();
}

template <>
absl::Status AddNeighborhoodNodes<Execution>(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory) {
  // TODO(b/145945460) Support events for executions.
  MLMD_RETURN_IF_ERROR(AddContexts(query_string, analyzer_opts, type_factory));
  return absl::OkStatus();
}

template <>
absl::Status AddNeighborhoodNodes<Context>(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory) {
  MLMD_RETURN_IF_ERROR(
      AddChildContexts(query_string, analyzer_opts, type_factory));
  MLMD_RETURN_IF_ERROR(
      AddParentContexts(query_string, analyzer_opts, type_factory));
  return absl::OkStatus();
}

// Adds the (custom)properties used in the query to the analyzer as StructType
// column. It uses `property_re` to match the mentioned (custom)properties.
// Rewrites the user-facing syntax `column_prefix.name.` to valid
// ZetaSQL syntax `column_prefix_name.`, where `column_prefix` is either
// `properties` or `custom_properties`.
zetasql_base::StatusOr<std::string> AddPropertiesAndTransformQueryImpl(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory, const RE2& property_re,
    absl::string_view column_prefix) {
  std::string matched_property;
  absl::flat_hash_set<std::string> mentioned_properties;
  std::string rewritten_query(query_string);
  while (RE2::FindAndConsume(&query_string, property_re, &matched_property)) {
    const bool is_matched_property_backquoted = matched_property[0] == '`';
    const std::string ast_property = absl::StrCat(
        column_prefix, "_",
        is_matched_property_backquoted
            ? matched_property.substr(1, matched_property.size() - 2)
            : matched_property);
    // Each mentioned (custom)property is defined only once.
    if (mentioned_properties.contains(ast_property)) {
      continue;
    }
    mentioned_properties.insert(ast_property);
    // rewrite the query string if it encounters the mentioned property for
    // the first time. If the query rewriting replacement returns error, the
    // RE used for matching may be updated and not in-sync with the
    // replacement.
    if (!RE2::GlobalReplace(
            &rewritten_query,
            absl::StrCat("\\b", column_prefix, "\\.", matched_property, "\\."),
            is_matched_property_backquoted
                ? absl::StrCat("`", ast_property, "`.")
                : absl::StrCat(ast_property, "."))) {
      return absl::InternalError(absl::StrCat(
          "Query cannot be rewritten successfully: matched_property: ",
          column_prefix, ".", matched_property,
          ", current query: ", ast_property));
    }
    const zetasql::Type* property_struct_type;
    MLMD_RETURN_IF_ERROR(
        type_factory.MakeStructType({{"int_value", Int64Type()},
                                     {"double_value", DoubleType()},
                                     {"string_value", StringType()}},
                                    &property_struct_type));
    MLMD_RETURN_IF_ERROR(
        analyzer_opts.AddExpressionColumn(ast_property, property_struct_type));
  }
  return rewritten_query;
}

// Adds the properties used in the query to the analyzer as StructType
// column and rewrite the user-facing syntax using ZetaSQL syntax.
zetasql_base::StatusOr<std::string> AddPropertiesAndTransformQuery(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory) {
  static LazyRE2 property_re = {kPropertyRE};
  return AddPropertiesAndTransformQueryImpl(query_string, analyzer_opts,
                                            type_factory, *property_re,
                                            /*column_prefix=*/"properties");
}

// Adds the custom properties used in the query to the analyzer as StructType
// column and rewrite the user-facing syntax using ZetaSQL syntax.
zetasql_base::StatusOr<std::string> AddCustomPropertiesAndTransformQuery(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory) {
  static LazyRE2 property_re = {kCustomPropertyRE};
  return AddPropertiesAndTransformQueryImpl(
      query_string, analyzer_opts, type_factory, *property_re,
      /*column_prefix=*/"custom_properties");
}

}  // namespace

template <typename T>
FilterQueryAstResolver<T>::FilterQueryAstResolver(
    const std::string& query_string)
    : raw_query_(query_string), catalog_("default", &type_factory_) {}

template <typename T>
absl::Status FilterQueryAstResolver<T>::Resolve() {
  // return if the query is already resolved.
  if (output_) {
    return absl::OkStatus();
  }

  // Scan the query string and prepare the type factory.
  ZETASQL_ASSIGN_OR_RETURN(
      std::string transformed_query,
      AddAttributesAndTransformQuery<T>(raw_query_, analyzer_opts_));
  MLMD_RETURN_IF_ERROR(AddNeighborhoodNodes<T>(transformed_query,
                                               analyzer_opts_, type_factory_));
  ZETASQL_ASSIGN_OR_RETURN(transformed_query,
                   AddPropertiesAndTransformQuery(
                       transformed_query, analyzer_opts_, type_factory_));
  ZETASQL_ASSIGN_OR_RETURN(transformed_query,
                   AddCustomPropertiesAndTransformQuery(
                       transformed_query, analyzer_opts_, type_factory_));
  // Parse the query string with the type factory and return errors if invalid.
  catalog_.AddZetaSQLFunctions(analyzer_opts_.language());
  MLMD_RETURN_IF_ERROR(zetasql::AnalyzeExpression(
      transformed_query, analyzer_opts_, &catalog_, &type_factory_, &output_));
  if (output_->resolved_expr()->type()->kind() !=
      zetasql::TypeKind::TYPE_BOOL) {
    output_.reset();
    return absl::InvalidArgumentError(absl::StrCat(
        "Given query string is not a valid boolean expression: ", raw_query_));
  }
  return absl::OkStatus();
}

template <typename T>
const zetasql::ResolvedExpr* FilterQueryAstResolver<T>::GetAst() {
  if (!output_) {
    return nullptr;
  }
  return output_->resolved_expr();
}

// Explicit template instantiation for supported node types.
template class FilterQueryAstResolver<Artifact>;
template class FilterQueryAstResolver<Execution>;
template class FilterQueryAstResolver<Context>;

}  // namespace ml_metadata
