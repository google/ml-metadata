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

#include "ml_metadata/metadata_store/transaction_executor.h"

#include "absl/status/status.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {

absl::Status RdbmsTransactionExecutor::Execute(
    const std::function<absl::Status()>& txn_body,
    const TransactionOptions& transaction_options) const {
  if (metadata_source_ == nullptr || !metadata_source_->is_connected()) {
    return absl::FailedPreconditionError(
        "To use ExecuteTransaction, the metadata_source should be created and "
        "connected");
  }

  MLMD_RETURN_IF_ERROR(metadata_source_->Begin());

  absl::Status transaction_status = txn_body();
  if (transaction_status.ok()) {
    transaction_status.Update(metadata_source_->Commit());
  }
  // Commit may fail as well, if so, we do rollback to allow the caller retry.
  if (!transaction_status.ok()) {
    transaction_status.Update(metadata_source_->Rollback());
  }
  return transaction_status;
}

}  // namespace ml_metadata
