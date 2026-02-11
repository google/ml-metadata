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

// DEPRECATED: This file and its associated ZetaSQL-based filter_query
// functionality is deprecated and will be removed in version 1.18.0.
// ZetaSQL dependency is being phased out from ML Metadata.
// Please migrate to alternative filtering approaches.

#include "ml_metadata/query/filter_query_ast_resolver.h"
#include <vector>

#include "zetasql/public/analyzer.h"
#include "zetasql/public/simple_catalog.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/util/return_utils.h"
#include "re2/re2.h"

namespace ml_metadata {
namespace {

using ::zetasql::types::DoubleType;
using ::zetasql::types::Int64Type;
using ::zetasql::types::StringType;
using ::zetasql::types::BoolType;

// A regular expression that for mentioned contexts in query.
constexpr absl::string_view kContextRE = "\\b(contexts_[[:word:]]+)\\.";
constexpr absl::string_view kArtifactRE = "\\b(artifacts_[[:word:]]+)\\.";
constexpr absl::string_view kExecutionRE = "\\b(executions_[[:word:]]+)\\.";
constexpr absl::string_view kPropertyRE =
    "\\bproperties\\.(:?([[:word:]]+)|(`[^`]+`))\\.(?:int|double|string|bool)";
constexpr absl::string_view kCustomPropertyRE =
    "\\bcustom_properties\\.(:?([[:word:]]+)|(`[^`]+`))\\."
    "(?:int|double|string|bool)";
constexpr absl::string_view kChildContextRE =
    "\\b(child_contexts_[[:word:]]+)\\.";
constexpr absl::string_view kParentContextRE =
    "\\b(parent_contexts_[[:word:]]+)\\.";
constexpr absl::string_view kEventRE = "\\b(events_[[:word:]]+)\\.";

constexpr absl::string_view kArtifactStatePredicateRE =
    "\\b(state)[[:space:]]*(=|!=|"
    "(?i)[[:space:]]NOT[[:space:]]*IN[[:space:]]|(?i)[[:space:]]IN[[:space:]])"
    "[[:space:]]*([[:word:]]+\\b|\\(([[:space:]]*[[:word:]]+[[:space:]]*[,]*"
    "[[:space:]]*[[:word:]]+[[:space:]]*)*\\))";
constexpr absl::string_view kExecutionStatePredicateRE =
    "\\b(last_known_state)[[:space:]]*(=|!=|"
    "(?i)[[:space:]]NOT[[:space:]]*IN[[:space:]]|(?i)[[:space:]]IN[[:space:]])"
    "[[:space:]]*([[:word:]]+\\b|\\(([[:space:]]*[[:word:]]+[[:space:]]*[,]*"
    "[[:space:]]*[[:word:]]+[[:space:]]*)*\\))";
constexpr absl::string_view kEventTypePredicateRE =
    "\\b(events_[[:word:]]+\\.type)[[:space:]]*(=|!=|"
    "(?i)[[:space:]]NOT[[:space:]]*IN[[:space:]]|(?i)[[:space:]]IN[[:space:]])"
    "[[:space:]]*([[:word:]]+\\b|\\(([[:space:]]*[[:word:]]+[[:space:]]*[,]*"
    "[[:space:]]*[[:word:]]+[[:space:]]*)*\\))";

// Returns a map of Artifact state Enums to their corresponding int values. See
// go/totw/110#the-fix-safe-initialization-no-destruction for more information
// on why we are using a function instead of a direct variable declaration.
// Even though gtl::fixed_flat_map_of is preferred, here we use an alternative
// approach because gtl libraries are not available in OSS.
static const absl::flat_hash_map<std::string, int>
GetArtifactStateValueMapping() {
  static const absl::flat_hash_map<std::string, int>& mapping =
      *new absl::flat_hash_map<std::string, int>(
          {{Artifact::State_Name(Artifact::UNKNOWN), Artifact::UNKNOWN},
           {Artifact::State_Name(Artifact::PENDING), Artifact::PENDING},
           {Artifact::State_Name(Artifact::LIVE), Artifact::LIVE},
           {Artifact::State_Name(Artifact::MARKED_FOR_DELETION),
            Artifact::MARKED_FOR_DELETION},
           {Artifact::State_Name(Artifact::DELETED), Artifact::DELETED}});
  return mapping;
}

// Returns a map of Execution state Enums to their corresponding int values. See
// go/totw/110#the-fix-safe-initialization-no-destruction for more information
// on why we are using a function instead of a direct variable declaration.
// Even though gtl::fixed_flat_map_of is preferred, here we use an alternative
// approach because gtl libraries are not available in OSS.
static const absl::flat_hash_map<std::string, int>
GetExecutionStateValueMapping() {
  static const absl::flat_hash_map<std::string, int>& mapping =
      *new absl::flat_hash_map<std::string, int>(
          {{Execution::State_Name(Execution::UNKNOWN), Execution::UNKNOWN},
           {Execution::State_Name(Execution::NEW), Execution::NEW},
           {Execution::State_Name(Execution::RUNNING), Execution::RUNNING},
           {Execution::State_Name(Execution::COMPLETE), Execution::COMPLETE},
           {Execution::State_Name(Execution::FAILED), Execution::FAILED},
           {Execution::State_Name(Execution::CACHED), Execution::CACHED},
           {Execution::State_Name(Execution::CANCELED), Execution::CANCELED}});
  return mapping;
}

// Returns a map of Event state Enums to their corresponding int values. See
// go/totw/110#the-fix-safe-initialization-no-destruction for more information
// on why we are using a function instead of a direct variable declaration.
// Even though gtl::fixed_flat_map_of is preferred, here we use an alternative
// approach because gtl libraries are not available in OSS.
static const absl::flat_hash_map<std::string, int> GetEventStateValueMapping() {
  static const absl::flat_hash_map<std::string, int>& mapping =
      *new absl::flat_hash_map<std::string, int>({
          {Event::Type_Name(Event::INPUT), Event::INPUT},
          {Event::Type_Name(Event::OUTPUT), Event::OUTPUT},
          {Event::Type_Name(Event::DECLARED_INPUT), Event::DECLARED_INPUT},
          {Event::Type_Name(Event::DECLARED_OUTPUT), Event::DECLARED_OUTPUT},
          {Event::Type_Name(Event::INTERNAL_INPUT), Event::INTERNAL_INPUT},
          {Event::Type_Name(Event::INTERNAL_OUTPUT), Event::INTERNAL_OUTPUT},
      });
  return mapping;
}

// Parses query for enum predicates e.g. state = LIVE and re-writes the query
// into form state = <int> based on the `enum_value_mapping` provided by the
// caller. The `enum_predicate_regex` provides the expected state predicate in
// the query.
absl::StatusOr<std::string> ParseEnumPredicateAndTransformQuery(
    absl::string_view query_string, absl::string_view enum_predicate_regex,
    const absl::flat_hash_map<std::string, int>& enum_value_mapping) {
  std::string rewritten_query = std::string(query_string);
  std::string state_string_literal, operator_literal, value_literal;
  while (RE2::FindAndConsume(&query_string, enum_predicate_regex,
                             &state_string_literal, &operator_literal,
                             &value_literal)) {
    std::string query_subtitute;
    if (operator_literal == "=" || operator_literal == "!=") {
      if (!enum_value_mapping.contains(value_literal)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported enum value specified in the query: ",
                        value_literal));
      }

      if (operator_literal == "!=") {
        query_subtitute = absl::Substitute(
            " (($0 $1 $2) OR ($0 IS NULL)) ", state_string_literal,
            operator_literal, enum_value_mapping.at(value_literal));
      } else {
        query_subtitute = absl::Substitute(
            " $0 $1 $2 ", state_string_literal, operator_literal,
            enum_value_mapping.at(value_literal));
      }
    } else {
      if (!absl::StartsWith(value_literal, "(") &&
          !absl::EndsWith(value_literal, ")")) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected a list of enum values enclosed in parentheses but got ",
            value_literal));
      }
      std::string value_literal_no_parens =
          value_literal.substr(1, value_literal.size() - 2);
      if (value_literal_no_parens.empty()) { continue; }
      std::vector<std::string> value_literals =
          absl::StrSplit(value_literal_no_parens, ',');
      std::vector<int> enum_values;
      for (std::string& literal : value_literals) {
        absl::StripAsciiWhitespace(&literal);
        if (!enum_value_mapping.contains(literal)) {
          return absl::InvalidArgumentError(
              absl::StrCat("Unsupported enum value specified in the query: ",
                           literal));
        }
        enum_values.push_back(enum_value_mapping.at(literal));
      }
      std::string enum_literal = absl::StrJoin(enum_values, ",");
      absl::StripAsciiWhitespace(&operator_literal);
      if (absl::EqualsIgnoreCase(operator_literal, "IN")) {
        query_subtitute = absl::Substitute(" $0 $1 ($2) ", state_string_literal,
                                           operator_literal, enum_literal);
      } else {
        query_subtitute = absl::Substitute(" (($0 $1 ($2)) OR ($0 IS NULL)) ",
                                           state_string_literal,
                                           operator_literal, enum_literal);
      }
    }
    value_literal =
        absl::StrReplaceAll(value_literal, {{"(", "\\("}, {")", "\\)"}});
    std::string replace_regex = absl::StrReplaceAll(
        enum_predicate_regex,
        {{"([[:word:]]+\\b|\\(([[:space:]]*[[:word:]]+[[:space:]]*[,]*"
          "[[:space:]]*[[:word:]]+[[:space:]]*)*\\))",
          value_literal}});
    if (!RE2::GlobalReplace(&rewritten_query, replace_regex, query_subtitute)) {
      return absl::InternalError(absl::Substitute(
          "Query cannot be rewritten successfully for matched enum predicate: "
          "$0 $1 $2",
          state_string_literal, operator_literal, value_literal));
    }
  }

  return rewritten_query;
}

absl::StatusOr<std::string> AddArtifactStateAndTransformQuery(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts) {
  MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn("state", Int64Type()));
  return ParseEnumPredicateAndTransformQuery(
      query_string, kArtifactStatePredicateRE, GetArtifactStateValueMapping());
}

absl::StatusOr<std::string> AddExecutionLastKnownStateAndTransformQuery(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts) {
  MLMD_RETURN_IF_ERROR(
      analyzer_opts.AddExpressionColumn("last_known_state", Int64Type()));
  return ParseEnumPredicateAndTransformQuery(query_string,
                                             kExecutionStatePredicateRE,
                                             GetExecutionStateValueMapping());
}

// Adds a list of columns corresponding to the node attributes which are
// allowed in the query to the analyzer. MLMD nodes share a list of common
// attributes, such as id, type and name, while each node type has its own
// specific fields, e.g., Artifact.uri, Execution.last_known_state.
template <typename T>
absl::StatusOr<std::string> AddSpecificAttributesAndTransformQuery(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts) {
  return std::string(query_string);
}

template <>
absl::StatusOr<std::string> AddSpecificAttributesAndTransformQuery<Artifact>(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts) {
  MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn("uri", StringType()));
  return AddArtifactStateAndTransformQuery(query_string, analyzer_opts);
}

template <>
absl::StatusOr<std::string> AddSpecificAttributesAndTransformQuery<Execution>(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts) {
  return AddExecutionLastKnownStateAndTransformQuery(query_string,
                                                     analyzer_opts);
}

template <typename T>
absl::StatusOr<std::string> AddAttributesAndTransformQuery(
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
  MLMD_RETURN_IF_ERROR(
      analyzer_opts.AddExpressionColumn("external_id", StringType()));
  return AddSpecificAttributesAndTransformQuery<T>(query_string, analyzer_opts);
}

// Adds neighborhood nodes allowed in the query to the analyzer as StructType
// columns. For example, for artifact and execution, 1-hop contexts and events
// are supported in the filtering query; for context, the attributed artifacts
// and associated executions can be used.
template <typename T>
absl::StatusOr<std::string> AddNeighborhoodNodes(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory);

// Adds the Contexts used in the query to the analyzer as StructType column.
absl::Status AddContextsImpl(absl::string_view query_string,
                             absl::string_view context_prefix,
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

// Adds the Artifacts used in the query to the analyzer as StructType column.
absl::Status AddArtifactsImpl(absl::string_view query_string,
                              absl::string_view artifact_prefix,
                              zetasql::AnalyzerOptions& analyzer_opts,
                              zetasql::TypeFactory& type_factory) {
  RE2 artifact_re(artifact_prefix);
  std::string matched_artifact;
  absl::flat_hash_set<std::string> mentioned_artifacts;
  while (RE2::FindAndConsume(&query_string, artifact_re, &matched_artifact)) {
    // Each mentioned artifact is defined only once.
    if (mentioned_artifacts.contains(matched_artifact)) {
      continue;
    }
    mentioned_artifacts.insert(matched_artifact);
    const zetasql::Type* artifact_struct_type;
    MLMD_RETURN_IF_ERROR(type_factory.MakeStructType(
        {{"id", Int64Type()},
         {"name", StringType()},
         {"type", StringType()},
         {"state", Int64Type()},
         {"uri", StringType()},
         {"external_id", StringType()},
         {"create_time_since_epoch", Int64Type()},
         {"last_update_time_since_epoch", Int64Type()}},
        &artifact_struct_type));
    MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn(
        matched_artifact, artifact_struct_type));
  }
  return absl::OkStatus();
}

// Adds the Executions used in the query to the analyzer as StructType column.
absl::Status AddExecutionsImpl(absl::string_view query_string,
                               absl::string_view execution_prefix,
                               zetasql::AnalyzerOptions& analyzer_opts,
                               zetasql::TypeFactory& type_factory) {
  RE2 execution_re(execution_prefix);
  std::string matched_execution;
  absl::flat_hash_set<std::string> mentioned_executions;
  while (RE2::FindAndConsume(&query_string, execution_re, &matched_execution)) {
    // Each mentioned execution is defined only once.
    if (mentioned_executions.contains(matched_execution)) {
      continue;
    }
    mentioned_executions.insert(matched_execution);
    const zetasql::Type* execution_struct_type;
    MLMD_RETURN_IF_ERROR(type_factory.MakeStructType(
        {{"id", Int64Type()},
         {"name", StringType()},
         {"type", StringType()},
         {"last_known_state", Int64Type()},
         {"external_id", StringType()},
         {"create_time_since_epoch", Int64Type()},
         {"last_update_time_since_epoch", Int64Type()}},
        &execution_struct_type));
    MLMD_RETURN_IF_ERROR(analyzer_opts.AddExpressionColumn(
        matched_execution, execution_struct_type));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> AddEvents(absl::string_view query_string,
                                      absl::string_view neighbor_node_id,
                                      zetasql::AnalyzerOptions& analyzer_opts,
                                      zetasql::TypeFactory& type_factory) {
  static LazyRE2 event_re = {kEventRE.data()};
  std::string matched_event;
  absl::flat_hash_set<std::string> mentioned_events;
  std::string original_query = std::string(query_string);
  while (RE2::FindAndConsume(&query_string, *event_re, &matched_event)) {
    // Each mentioned event is defined only once.
    if (mentioned_events.contains(matched_event)) {
      continue;
    }
    mentioned_events.insert(matched_event);
    const zetasql::Type* event_struct_type;
    MLMD_RETURN_IF_ERROR(type_factory.MakeStructType(
        {{"type", Int64Type()},
         {"milliseconds_since_epoch", Int64Type()},
         {std::string(neighbor_node_id), Int64Type()}},
        &event_struct_type));

    MLMD_RETURN_IF_ERROR(
        analyzer_opts.AddExpressionColumn(matched_event, event_struct_type));
  }

  return ParseEnumPredicateAndTransformQuery(
      original_query, kEventTypePredicateRE, GetEventStateValueMapping());
}

absl::Status AddContexts(absl::string_view query_string,
                         zetasql::AnalyzerOptions& analyzer_opts,
                         zetasql::TypeFactory& type_factory) {
  return AddContextsImpl(query_string, kContextRE, analyzer_opts, type_factory);
}

absl::Status AddExecutions(absl::string_view query_string,
                           zetasql::AnalyzerOptions& analyzer_opts,
                           zetasql::TypeFactory& type_factory) {
  return AddExecutionsImpl(query_string, kExecutionRE, analyzer_opts,
                           type_factory);
}

absl::Status AddArtifacts(absl::string_view query_string,
                          zetasql::AnalyzerOptions& analyzer_opts,
                          zetasql::TypeFactory& type_factory) {
  return AddArtifactsImpl(query_string, kArtifactRE, analyzer_opts,
                          type_factory);
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
absl::StatusOr<std::string> AddNeighborhoodNodes<Artifact>(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory) {
  MLMD_RETURN_IF_ERROR(AddContexts(query_string, analyzer_opts, type_factory));
  return AddEvents(query_string, "execution_id", analyzer_opts, type_factory);
}

template <>
absl::StatusOr<std::string> AddNeighborhoodNodes<Execution>(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory) {
  MLMD_RETURN_IF_ERROR(AddContexts(query_string, analyzer_opts, type_factory));
  return AddEvents(query_string, "artifact_id", analyzer_opts, type_factory);
}

template <>
absl::StatusOr<std::string> AddNeighborhoodNodes<Context>(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory) {
  MLMD_RETURN_IF_ERROR(
      AddChildContexts(query_string, analyzer_opts, type_factory));
  MLMD_RETURN_IF_ERROR(
      AddParentContexts(query_string, analyzer_opts, type_factory));
  MLMD_RETURN_IF_ERROR(AddArtifacts(query_string, analyzer_opts, type_factory));
  MLMD_RETURN_IF_ERROR(
      AddExecutions(query_string, analyzer_opts, type_factory));
  MLMD_ASSIGN_OR_RETURN(std::string modified_query_string,
                        ParseEnumPredicateAndTransformQuery(
                            query_string, kArtifactStatePredicateRE,
                            GetArtifactStateValueMapping()));
  MLMD_ASSIGN_OR_RETURN(modified_query_string,
                        ParseEnumPredicateAndTransformQuery(
                            modified_query_string, kExecutionStatePredicateRE,
                            GetExecutionStateValueMapping()));
  return modified_query_string;
}

// Adds the (custom)properties used in the query to the analyzer as StructType
// column. It uses `property_re` to match the mentioned (custom)properties.
// Rewrites the user-facing syntax `column_prefix.name.` to valid
// ZetaSQL syntax `column_prefix_name.`, where `column_prefix` is either
// `properties` or `custom_properties`.
absl::StatusOr<std::string> AddPropertiesAndTransformQueryImpl(
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
                                     {"string_value", StringType()},
                                     {"bool_value", BoolType()}},
                                    &property_struct_type));
    MLMD_RETURN_IF_ERROR(
        analyzer_opts.AddExpressionColumn(ast_property, property_struct_type));
  }
  return rewritten_query;
}

// Adds the properties used in the query to the analyzer as StructType
// column and rewrite the user-facing syntax using ZetaSQL syntax.
absl::StatusOr<std::string> AddPropertiesAndTransformQuery(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory) {
  static LazyRE2 property_re = {kPropertyRE.data()};
  return AddPropertiesAndTransformQueryImpl(query_string, analyzer_opts,
                                            type_factory, *property_re,
                                            /*column_prefix=*/"properties");
}

// Adds the custom properties used in the query to the analyzer as StructType
// column and rewrite the user-facing syntax using ZetaSQL syntax.
absl::StatusOr<std::string> AddCustomPropertiesAndTransformQuery(
    absl::string_view query_string, zetasql::AnalyzerOptions& analyzer_opts,
    zetasql::TypeFactory& type_factory) {
  static LazyRE2 property_re = {kCustomPropertyRE.data()};
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
  MLMD_ASSIGN_OR_RETURN(
      std::string transformed_query,
      AddAttributesAndTransformQuery<T>(raw_query_, analyzer_opts_));
  MLMD_ASSIGN_OR_RETURN(transformed_query,
                        (AddNeighborhoodNodes<T>(
                            transformed_query, analyzer_opts_, type_factory_)));
  MLMD_ASSIGN_OR_RETURN(transformed_query,
                        AddPropertiesAndTransformQuery(
                            transformed_query, analyzer_opts_, type_factory_));
  MLMD_ASSIGN_OR_RETURN(transformed_query,
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
