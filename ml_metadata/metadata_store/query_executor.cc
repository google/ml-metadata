/* Copyright 2021 Google LLC

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
#include "ml_metadata/metadata_store/query_executor.h"

#include <optional>

#include <glog/logging.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {

// The kSupportedEarlierQueryVersion can be the same with the current library
// schema version or a schema version before hand. The latter is for
// supporting migration for MLMD online services.
static constexpr int64_t kSupportedEarlierQueryVersion = 7;

QueryExecutor::QueryExecutor(std::optional<int64_t> query_schema_version)
    : query_schema_version_(query_schema_version) {
  if (query_schema_version_) {
    CHECK_GE(*query_schema_version_, kSupportedEarlierQueryVersion)
        << "The query config executor does not support other earlier query "
           "version other than "
        << kSupportedEarlierQueryVersion
        << "; query_schema_version: " << *query_schema_version_;
  }
}

absl::Status QueryExecutor::VerifyCurrentQueryVersionIsAtLeast(
    int64_t min_schema_version) const {
  return query_schema_version_ && *query_schema_version_ < min_schema_version
             ? absl::FailedPreconditionError(absl::StrCat(
                   "The query executor method requires query_schema_version "
                   ">= ",
                   min_schema_version,
                   "; current query version: ", *query_schema_version_))
             : absl::OkStatus();
}

absl::Status QueryExecutor::CheckSchemaVersionAlignsWithQueryVersion() {
  if (!query_schema_version_) {
    return absl::InvalidArgumentError(
        "When query_schema_version_ is set, the method checks the given db has "
        "already initialized with the schema version.");
  }
  int64_t db_version = -1;
  const absl::Status existing_schema_version_status =
      GetSchemaVersion(&db_version);
  if (absl::IsNotFound(existing_schema_version_status)) {
    return absl::FailedPreconditionError(
        "When using the query executor with query_version other than head, "
        "the db should already exists. For empty db, init it using head "
        "version with the default constructor");
  }
  MLMD_RETURN_IF_ERROR(existing_schema_version_status);
  if (db_version != *query_schema_version_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "The query executor is configured with a query_version: ",
        *query_schema_version_,
        ", however the underlying db is at schema_version: ", db_version));
  }
  return absl::OkStatus();
}

bool QueryExecutor::IsQuerySchemaVersionEquals(int64_t schema_version) const {
  return query_schema_version_ && *query_schema_version_ == schema_version;
}

}  // namespace ml_metadata
