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
#include "ml_metadata/metadata_store/metadata_source.h"

#include "absl/status/status.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {

absl::Status MetadataSource::Connect() {
  if (is_connected_)
    return absl::FailedPreconditionError(
        "The connection has been opened. Close() the current connection before "
        "Connect() again.");
  MLMD_RETURN_IF_ERROR(ConnectImpl());
  is_connected_ = true;
  return absl::OkStatus();
}

absl::Status MetadataSource::Close() {
  if (!is_connected_)
    return absl::FailedPreconditionError(
        "No connection is opened when calling Close().");
  MLMD_RETURN_IF_ERROR(CloseImpl());
  is_connected_ = false;
  return absl::OkStatus();
}

absl::Status MetadataSource::ExecuteQuery(const std::string& query,
                                          RecordSet* results) {
  if (!is_connected_)
    return absl::FailedPreconditionError("No opened connection for querying.");
  if (!transaction_open_)
    return absl::FailedPreconditionError("Transaction not open.");
  return ExecuteQueryImpl(query, results);
}

absl::Status MetadataSource::Begin() {
  if (!is_connected_)
    return absl::FailedPreconditionError("No opened connection for querying.");
  if (transaction_open_)
    return absl::FailedPreconditionError("Transaction already open.");
  MLMD_RETURN_IF_ERROR(BeginImpl());
  transaction_open_ = true;
  return absl::OkStatus();
}


absl::Status MetadataSource::Commit() {
  if (!is_connected_)
    return absl::FailedPreconditionError("No opened connection for querying.");
  if (!transaction_open_)
    return absl::FailedPreconditionError("Transaction not open.");
  MLMD_RETURN_IF_ERROR(CommitImpl());
  transaction_open_ = false;
  return absl::OkStatus();
}

absl::Status MetadataSource::Rollback() {
  if (!is_connected_)
    return absl::FailedPreconditionError("No opened connection for querying.");
  if (!transaction_open_)
    return absl::FailedPreconditionError("Transaction not open.");
  MLMD_RETURN_IF_ERROR(RollbackImpl());
  transaction_open_ = false;
  return absl::OkStatus();
}

}  // namespace ml_metadata
