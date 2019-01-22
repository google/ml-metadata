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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

tensorflow::Status MetadataSource::Connect() {
  if (is_connected_)
    return tensorflow::errors::FailedPrecondition(
        "The connection has been opened. Close() the current connection before "
        "Connect() again.");
  TF_RETURN_IF_ERROR(ConnectImpl());
  is_connected_ = true;
  return tensorflow::Status::OK();
}

tensorflow::Status MetadataSource::Close() {
  if (!is_connected_)
    return tensorflow::errors::FailedPrecondition(
        "No connection is opened when calling Close().");
  TF_RETURN_IF_ERROR(CloseImpl());
  is_connected_ = false;
  return tensorflow::Status::OK();
}

tensorflow::Status MetadataSource::ExecuteQuery(const string& query,
                                                RecordSet* results) {
  if (!is_connected_)
    return tensorflow::errors::FailedPrecondition(
        "No opened connection for querying.");
  if (!transaction_open_)
    return tensorflow::errors::FailedPrecondition("Transaction not open.");
  return ExecuteQueryImpl(query, results);
}

tensorflow::Status MetadataSource::Begin() {
  if (!is_connected_)
    return tensorflow::errors::FailedPrecondition(
        "No opened connection for querying.");
  if (transaction_open_)
    return tensorflow::errors::FailedPrecondition("Transaction already open.");
  TF_RETURN_IF_ERROR(BeginImpl());
  transaction_open_ = true;
  return tensorflow::Status::OK();
}

tensorflow::Status MetadataSource::Commit() {
  if (!is_connected_)
    return tensorflow::errors::FailedPrecondition(
        "No opened connection for querying.");
  if (!transaction_open_)
    return tensorflow::errors::FailedPrecondition("Transaction not open.");
  TF_RETURN_IF_ERROR(CommitImpl());
  transaction_open_ = false;
  return tensorflow::Status::OK();
}

tensorflow::Status MetadataSource::Rollback() {
  if (!is_connected_)
    return tensorflow::errors::FailedPrecondition(
        "No opened connection for querying.");
  if (!transaction_open_)
    return tensorflow::errors::FailedPrecondition("Transaction not open.");
  TF_RETURN_IF_ERROR(RollbackImpl());
  transaction_open_ = false;
  return tensorflow::Status::OK();
}

ScopedTransaction::ScopedTransaction(MetadataSource* metadata_source)
    : committed_(false), metadata_source_(metadata_source) {
  CHECK(metadata_source->is_connected());
  TF_CHECK_OK(metadata_source->Begin());
}

ScopedTransaction::~ScopedTransaction() {
  if (!committed_) {
    TF_CHECK_OK(metadata_source_->Rollback());
  }
}

// Commit the transaction.
// If there is a failure during the commit, the commit_ flag is not
// set, resulting in a Rollback().
// Should be called no more than once on a transaction.
tensorflow::Status ScopedTransaction::Commit() {
  if (committed_) {
    // Note: this doesn't catch if the user calls Commit() on the
    // metadata_source_ directly.
    return ::tensorflow::errors::FailedPrecondition(
        "Cannot commit a transaction twice");
  }
  TF_CHECK_OK(metadata_source_->Commit());
  committed_ = true;
  return tensorflow::Status::OK();
}

}  // namespace ml_metadata
