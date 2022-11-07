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

#include <functional>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ml_metadata/metadata_store/metadata_source.h"

namespace ml_metadata {
namespace {

using ::testing::Return;

class MockMetadataSource : public MetadataSource {
 public:
  MOCK_METHOD(absl::Status, BeginImpl, (), (override));
  MOCK_METHOD(absl::Status, ConnectImpl, (), (override));
  MOCK_METHOD(absl::Status, CloseImpl, (), (override));
  MOCK_METHOD(absl::Status, RollbackImpl, (), (override));
  MOCK_METHOD(absl::Status, CommitImpl, (), ());
  MOCK_METHOD(absl::Status, ExecuteQueryImpl,
              (const std::string& query, RecordSet* results), (override));
  MOCK_METHOD(std::string, EscapeString, (absl::string_view value),
              (const, override));
  MOCK_METHOD(std::string, EncodeBytes, (absl::string_view value), (const));
  MOCK_METHOD(absl::StatusOr<std::string>, DecodeBytes,
              (absl::string_view value), (const));
};

// Fake Errors.
const absl::Status kTfFuncErrorStatus =
    absl::InternalError("Fake txn body error.");
const absl::Status kTfConnectErrorStatus =
    absl::InternalError("Fake connection error.");
const absl::Status kTfCommitErrorStatus =
    absl::InternalError("Fake commit error.");
const absl::Status kTfRollbackErrorStatus =
    absl::InternalError("Fake rollback error.");
const absl::Status kTfBeginErrorStatus =
    absl::InternalError("Fake begin error.");

// Fake transaction body that always return OK status.
const std::function<absl::Status()> kFuncReturnOk = []() -> absl::Status {
  return absl::OkStatus();
};
// Fake transaction body that always return Internal error status.
const std::function<absl::Status()> kFuncReturnInternalError =
    []() -> absl::Status { return kTfFuncErrorStatus; };

TEST(TransactionExecutorTest, ReturnOkWhenBothTxnBodyAndCommitOk) {
  MockMetadataSource mock_metadata_source;
  // These calls should be called once and only once.
  EXPECT_CALL(mock_metadata_source, BeginImpl())
      .Times(1)
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_metadata_source, ConnectImpl())
      .Times(1)
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_metadata_source, CommitImpl())
      .Times(1)
      .WillOnce(Return(absl::OkStatus()));
  // These methods should not be called.
  EXPECT_CALL(mock_metadata_source, RollbackImpl())
      .Times(0);
  EXPECT_CALL(mock_metadata_source, CloseImpl())
      .Times(0);

  // Initialize the mock_metadata_source.
  ASSERT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  RdbmsTransactionExecutor txn_executor(&mock_metadata_source);

  EXPECT_EQ(absl::OkStatus(), txn_executor.Execute(kFuncReturnOk));
}

TEST(TransactionExecutorTest, ReturnErrorWhenTxnBodyReturnsError) {
  MockMetadataSource mock_metadata_source;
  // These calls should be called once and only once.
  EXPECT_CALL(mock_metadata_source, BeginImpl())
      .Times(1)
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_metadata_source, ConnectImpl())
      .Times(1)
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_metadata_source, RollbackImpl())
      .Times(1)
      .WillOnce(Return(absl::OkStatus()));
  // These methods should not be called.
  EXPECT_CALL(mock_metadata_source, CommitImpl())
      .Times(0);
  EXPECT_CALL(mock_metadata_source, CloseImpl())
      .Times(0);

  // Initialize the mock_metadata_source.
  ASSERT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  RdbmsTransactionExecutor txn_executor(&mock_metadata_source);

  EXPECT_EQ(txn_executor.Execute(kFuncReturnInternalError), kTfFuncErrorStatus);
}

TEST(TransactionExecutorTest, ReturnErrorWhenTxnBodyReturnsOkButCommitFails) {
  MockMetadataSource mock_metadata_source;
  // These calls should be called once and only once.
  EXPECT_CALL(mock_metadata_source, CommitImpl())
      .Times(1)
      .WillOnce(Return(kTfCommitErrorStatus));
  EXPECT_CALL(mock_metadata_source, BeginImpl())
      .Times(1)
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_metadata_source, ConnectImpl())
      .Times(1)
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(mock_metadata_source, CloseImpl())
      .Times(0);
  EXPECT_CALL(mock_metadata_source, RollbackImpl())
      .Times(1)
      .WillOnce(Return(kTfRollbackErrorStatus));

  // Initialize the mock_metadata_source.
  ASSERT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  RdbmsTransactionExecutor txn_executor(&mock_metadata_source);

  // Return commit error even rollback fails.
  EXPECT_EQ(txn_executor.Execute(kFuncReturnOk), kTfCommitErrorStatus);
}

TEST(TransactionExecutorTest, ReturnConnectErrorWhenConnectFails) {
  MockMetadataSource mock_metadata_source;
  // These calls should be called once and only once.
  EXPECT_CALL(mock_metadata_source, ConnectImpl())
      .Times(1)
      .WillOnce(Return(kTfConnectErrorStatus));
  // These methods should not be called.
  EXPECT_CALL(mock_metadata_source, CommitImpl())
      .Times(0);
  EXPECT_CALL(mock_metadata_source, RollbackImpl())
      .Times(0);
  EXPECT_CALL(mock_metadata_source, CloseImpl())
      .Times(0);

  // Initialize the mock_metadata_source.
  EXPECT_EQ(mock_metadata_source.Connect(), kTfConnectErrorStatus);
  RdbmsTransactionExecutor txn_executor(&mock_metadata_source);

  // Execute of txn_body will fail when the connection fails.
  EXPECT_FALSE(txn_executor.Execute(kFuncReturnOk).ok());
  EXPECT_EQ(txn_executor.Execute(kFuncReturnOk),
            absl::FailedPreconditionError(
                "To use ExecuteTransaction, the metadata_source should be"
                " created and connected"));
}

}  // namespace
}  // namespace ml_metadata
