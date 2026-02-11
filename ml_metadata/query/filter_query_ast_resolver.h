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
#ifndef ML_METADATA_GOOGLE_QUERY_FILTER_QUERY_AST_RESOLVER_H
#define ML_METADATA_GOOGLE_QUERY_FILTER_QUERY_AST_RESOLVER_H

#include "zetasql/public/analyzer.h"
#include "zetasql/public/simple_catalog.h"
#include "absl/status/status.h"
#include "re2/re2.h"

namespace ml_metadata {

// DEPRECATED: This class and its associated ZetaSQL-based filter_query
// functionality is deprecated and will be removed in version 1.18.0.
// ZetaSQL dependency is being phased out from ML Metadata.
// Please migrate to alternative filtering approaches.
//
// FilterQueryAstResolver parses the MLMD filtering query string and generates
// an AST via ZetaSQL analyzer. It can be instantiated with MLMD nodes types:
// Artifact, Execution and Context.
template <typename T>
class FilterQueryAstResolver {
 public:
  // Constructs an AST resolver for a given query_string. Always use a new AST
  // resolver instance to process a query string, as the temporary types for
  // each query string is not sharable.
  explicit FilterQueryAstResolver(const std::string& query_string);

  // Not copyable or movable
  FilterQueryAstResolver(const FilterQueryAstResolver&) = delete;
  FilterQueryAstResolver& operator=(const FilterQueryAstResolver&) = delete;

  // Processes the user query string and generates an ast.
  // It reads the user query string, creates a list of zetasql types
  // and validates the query.
  // Returns InvalidArgument if the given query is not a boolean expression.
  // Returns InvalidArgument errors if the query string contains symbols that
  // is not supported by the filter query syntax and semantics.
  absl::Status Resolve();

  // Returns the parsed AST of the giving query string. The AST is owned by
  // the instance of FilterQueryAstResolver and it is immutable.
  // The AST can be used with FilterQueryBuilder to generate actual query.
  // Returns nullptr if the Resolve() failed with errors.
  const zetasql::ResolvedExpr* GetAst();

 private:
  // The user query.
  const std::string raw_query_;
  // The type factory to keep the temporary types used in the query.
  zetasql::TypeFactory type_factory_;
  // The analyzer options that bridge the alias and the type.
  zetasql::AnalyzerOptions analyzer_opts_;
  // The simple catalog used by the analyzer for resolving functions.
  zetasql::SimpleCatalog catalog_;
  // The output that holds the resolved query.
  std::unique_ptr<const zetasql::AnalyzerOutput> output_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_GOOGLE_QUERY_FILTER_QUERY_AST_RESOLVER_H
