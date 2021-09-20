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

#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_TRANSACTION_EXECUTOR_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_TRANSACTION_EXECUTOR_H_

#include "absl/status/status.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {

// Pure virtual interface for MetadataStore to execute a transaction.
//
// Example usage:
//    TransactionExecutor* txn_executor;
//    ...
//    txn_executor->Execute(
//     [&metadata_access_object]() -> absl::Status {
//        return metadata_access_object->InitMetadataSource();
//     });
class TransactionExecutor {
 public:
  virtual ~TransactionExecutor() = default;

  // Runs txn_body and return the transaction status.
  virtual absl::Status Execute(
      const std::function<absl::Status()>& txn_body,
      const TransactionOptions& transaction_options = TransactionOptions())
      const = 0;
};

// An implementation of TransactionExecutor.
// It contains a method to execute the transaction body and tries to commit
// the execution result in the database by using Begin/Commit/Rollback
// methods in MetadataSource.
class RdbmsTransactionExecutor : public TransactionExecutor {
 public:
  explicit RdbmsTransactionExecutor(MetadataSource* metadata_source)
      : metadata_source_(metadata_source) {}
  ~RdbmsTransactionExecutor() override = default;

  // Tries to commit the execution result of txn_body.
  // When the txn_body returns OK, it calls Commit, otherwise it calls Rollback.
  //
  // Returns FAILED_PRECONDITION if metadata_source is null or not connected.
  // Returns detailed internal errors of transaction, i.e.
  //   Begin, Rollback and Commit.
  absl::Status Execute(const std::function<absl::Status()>& txn_body,
                       const TransactionOptions& transaction_options =
                           TransactionOptions()) const override;

 private:
  // The MetadataSource which has the connection to a database.
  // It also supports other database primitves like Commit and Abort.
  // Not owned by this class.
  MetadataSource* metadata_source_;
};

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_TRANSACTION_EXECUTOR_H_
