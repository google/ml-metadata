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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_CONSTANTS_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_CONSTANTS_H_

#include "absl/strings/string_view.h"
namespace ml_metadata {

// The in-memory encoding of the NULL value in RecordSet proto returned from
// any MetadataSource.
static constexpr absl::string_view kMetadataSourceNull = "__MLMD_NULL__";

// The node type_kind enum values used for internal storage. The enum value
// should not be modified, in order to be backward compatible with stored types.
// LINT.IfChange
enum class TypeKind { EXECUTION_TYPE = 0, ARTIFACT_TYPE = 1, CONTEXT_TYPE = 2 };
// LINT.ThenChange(../util/metadata_source_query_config.cc)

// Default maximum number of returned resources for List operation.
constexpr int kDefaultMaxListOperationResultSize = 100;

constexpr int kPropertyRecordSetSize = 8;

constexpr int kQueryLineageSubgraphMaxNumHops = 100;

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_CONSTANTS_H_
