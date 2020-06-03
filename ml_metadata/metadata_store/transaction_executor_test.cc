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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ml_metadata/metadata_store/metadata_source.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

using ::testing::Return;

class MockMetadataSource : public MetadataSource {
 public:
  MOCK_METHOD(tensorflow::Status, BeginImpl, (), (override));
  MOCK_METHOD(tensorflow::Status, ConnectImpl, (), (override));
  MOCK_METHOD(tensorflow::Status, CloseImpl, (), (override));
  MOCK_METHOD(tensorflow::Status, RollbackImpl, (), (override));
  MOCK_METHOD(tensorflow::Status, CommitImpl, (), ());
  MOCK_METHOD(tensorflow::Status, ExecuteQueryImpl,
              (const std::string& query, RecordSet* results), (override));
  MOCK_METHOD(std::string, EscapeString, (absl::string_view value),
              (const, override));
};

// Fake Errors.
const tensorflow::Status kTfFuncErrorStatus =
    tensorflow::errors::Internal("Fake txn body error.");
const tensorflow::Status kTfConnectErrorStatus =
    tensorflow::errors::Internal("Fake connection error.");
const tensorflow::Status kTfCommitErrorStatus =
    tensorflow::errors::Internal("Fake commit error.");
const tensorflow::Status kTfRollbackErrorStatus =
    tensorflow::errors::Internal("Fake rollback error.");
const tensorflow::Status kTfBeginErrorStatus =
    tensorflow::errors::Internal("Fake begin error.");

// Fake transaction body that always return OK status.
const std::function<tensorflow::Status()> kFuncReturnOk =
    []() -> tensorflow::Status { return tensorflow::Status::OK(); };
// Fake transaction body that always return Internal error status.
const std::function<tensorflow::Status()> kFuncReturnInternalError =
    []() -> tensorflow::Status { return kTfFuncErrorStatus; };

TEST(TransactionExecutorTest, ReturnOkWhenBothTxnBodyAndCommitOk) {
  MockMetadataSource mock_metadata_source;
  // These calls should be called once and only once.
  EXPECT_CALL(mock_metadata_source, BeginImpl())
      .Times(1)
      .WillOnce(Return(tensorflow::Status::OK()));
  EXPECT_CALL(mock_metadata_source, ConnectImpl())
      .Times(1)
      .WillOnce(Return(tensorflow::Status::OK()));
  EXPECT_CALL(mock_metadata_source, CommitImpl())
      .Times(1)
      .WillOnce(Return(tensorflow::Status::OK()));
  // These methods should not be called.
  EXPECT_CALL(mock_metadata_source, RollbackImpl())
      .Times(0);
  EXPECT_CALL(mock_metadata_source, CloseImpl())
      .Times(0);

  // Initialize the mock_metadata_source.
  TF_ASSERT_OK(mock_metadata_source.Connect());
  RdbmsTransactionExecutor txn_executor(&mock_metadata_source);

  TF_EXPECT_OK(txn_executor.Execute(kFuncReturnOk));
}

TEST(TransactionExecutorTest, ReturnErrorWhenTxnBodyReturnsError) {
  MockMetadataSource mock_metadata_source;
  // These calls should be called once and only once.
  EXPECT_CALL(mock_metadata_source, BeginImpl())
      .Times(1)
      .WillOnce(Return(tensorflow::Status::OK()));
  EXPECT_CALL(mock_metadata_source, ConnectImpl())
      .Times(1)
      .WillOnce(Return(tensorflow::Status::OK()));
  EXPECT_CALL(mock_metadata_source, RollbackImpl())
      .Times(1)
      .WillOnce(Return(tensorflow::Status::OK()));
  // These methods should not be called.
  EXPECT_CALL(mock_metadata_source, CommitImpl())
      .Times(0);
  EXPECT_CALL(mock_metadata_source, CloseImpl())
      .Times(0);

  // Initialize the mock_metadata_source.
  TF_ASSERT_OK(mock_metadata_source.Connect());
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
      .WillOnce(Return(tensorflow::Status::OK()));
  EXPECT_CALL(mock_metadata_source, ConnectImpl())
      .Times(1)
      .WillOnce(Return(tensorflow::Status::OK()));
  EXPECT_CALL(mock_metadata_source, CloseImpl())
      .Times(0);
  EXPECT_CALL(mock_metadata_source, RollbackImpl())
      .Times(1)
      .WillOnce(Return(kTfRollbackErrorStatus));

  // Initialize the mock_metadata_source.
  TF_ASSERT_OK(mock_metadata_source.Connect());
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
            tensorflow::errors::FailedPrecondition(
                "To use ExecuteTransaction, the metadata_source should be"
                " created and connected"));
}

}  // namespace
}  // namespace ml_metadata
