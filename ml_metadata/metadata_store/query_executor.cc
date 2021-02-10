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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

static constexpr int64 kSupportedEarlierQueryVersion = 5;

QueryExecutor::QueryExecutor(absl::optional<int64> query_schema_version)
    : query_schema_version_(query_schema_version) {
  if (query_schema_version_) {
    CHECK_EQ(*query_schema_version_, kSupportedEarlierQueryVersion)
        << "The query config executor does not support other earlier query "
           "version other than "
        << kSupportedEarlierQueryVersion
        << "; query_schema_version: " << *query_schema_version_;
  }
}

tensorflow::Status QueryExecutor::VerifyCurrentQueryVersionIsAtLeast(
    int64 min_schema_version) const {
  return query_schema_version_ && *query_schema_version_ < min_schema_version
             ? tensorflow::errors::FailedPrecondition(
                   "The query executor method requires query_schema_version "
                   ">= ", min_schema_version,
                   "; current query version: ", *query_schema_version_)
             : tensorflow::Status::OK();
}

tensorflow::Status QueryExecutor::CheckSchemaVersionAlignsWithQueryVersion() {
  if (!query_schema_version_) {
    return tensorflow::errors::InvalidArgument(
        "When query_schema_version_ is set, the method checks the given db has "
        "already initialized with the schema version.");
  }
  int64 db_version = -1;
  const tensorflow::Status existing_schema_version_status =
      GetSchemaVersion(&db_version);
  if (tensorflow::errors::IsNotFound(existing_schema_version_status)) {
    return tensorflow::errors::FailedPrecondition(
        "When using the query executor with query_version other than head, "
        "the db should already exists. For empty db, init it using head "
        "version with the default constructor");
  }
  TF_RETURN_IF_ERROR(existing_schema_version_status);
  if (db_version != *query_schema_version_) {
    return tensorflow::errors::FailedPrecondition(
        "The query executor is configured with a query_version: ",
        *query_schema_version_,
        ", however the underlying db is at schema_version: ", db_version);
  }
  return tensorflow::Status::OK();
}

bool QueryExecutor::IsQuerySchemaVersionEquals(int64 schema_version) const {
  return query_schema_version_ && *query_schema_version_ == schema_version;
}

}  // namespace ml_metadata
