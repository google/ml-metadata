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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ml_metadata/proto/metadata_source.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {

class MockMetadataSource : public MetadataSource {
 public:
  MOCK_METHOD0(ConnectImpl, tensorflow::Status());
  MOCK_METHOD0(CloseImpl, tensorflow::Status());
  MOCK_METHOD0(BeginImpl, tensorflow::Status());
  MOCK_METHOD2(ExecuteQueryImpl,
               tensorflow::Status(const string& query, RecordSet* results));
  MOCK_METHOD0(CommitImpl, tensorflow::Status());
  MOCK_METHOD0(RollbackImpl, tensorflow::Status());
  MOCK_CONST_METHOD1(EscapeString, string(absl::string_view value));
};

TEST(MetadataSourceTest, ConnectAgainWithoutClose) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, ConnectImpl()).Times(1);

  TF_EXPECT_OK(mock_metadata_source.Connect());
  tensorflow::Status s = mock_metadata_source.Connect();
  EXPECT_EQ(s.code(), tensorflow::error::FAILED_PRECONDITION);
}

TEST(MetadataSourceTest, CloseWithoutConnect) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, CloseImpl()).Times(0);

  tensorflow::Status s = mock_metadata_source.Close();
  EXPECT_EQ(s.code(), tensorflow::error::FAILED_PRECONDITION);
}

TEST(MetadataSourceTest, ConnectThenCloseThenConnectAgain) {
  MockMetadataSource mock_metadata_source;
  {
    ::testing::InSequence call_seq;
    EXPECT_CALL(mock_metadata_source, ConnectImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, CloseImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, ConnectImpl()).Times(1);
  }

  TF_EXPECT_OK(mock_metadata_source.Connect());
  TF_EXPECT_OK(mock_metadata_source.Close());
  TF_EXPECT_OK(mock_metadata_source.Connect());
}

TEST(MetadataSourceTest, TestExecuteQueryWithoutConnect) {
  MockMetadataSource mock_metadata_source;
  string query = "some query";
  RecordSet result;
  EXPECT_CALL(mock_metadata_source, ExecuteQueryImpl(query, &result)).Times(0);
  tensorflow::Status s = mock_metadata_source.ExecuteQuery(query, &result);
  EXPECT_EQ(s.code(), tensorflow::error::FAILED_PRECONDITION);
}

TEST(MetadataSourceTest, TestExecuteQueryWithoutBegin) {
  MockMetadataSource mock_metadata_source;
  string query = "some query";
  RecordSet result;
  EXPECT_CALL(mock_metadata_source, ExecuteQueryImpl(query, &result)).Times(0);
  TF_EXPECT_OK(mock_metadata_source.Connect());
  tensorflow::Status s = mock_metadata_source.ExecuteQuery(query, &result);
  EXPECT_EQ(s.code(), tensorflow::error::FAILED_PRECONDITION);
}

TEST(MetadataSourceTest, TestBeginAndCommit) {
  MockMetadataSource mock_metadata_source;
  {
    ::testing::InSequence call_seq;
    EXPECT_CALL(mock_metadata_source, BeginImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, CommitImpl()).Times(1);
  }
  TF_EXPECT_OK(mock_metadata_source.Connect());
  TF_EXPECT_OK(mock_metadata_source.Begin());
  TF_EXPECT_OK(mock_metadata_source.Commit());
}
TEST(MetadataSourceTest, TestBeginAndCommitTwice) {
  MockMetadataSource mock_metadata_source;
  {
    ::testing::InSequence call_seq;
    EXPECT_CALL(mock_metadata_source, BeginImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, CommitImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, BeginImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, CommitImpl()).Times(1);
  }
  TF_EXPECT_OK(mock_metadata_source.Connect());
  TF_EXPECT_OK(mock_metadata_source.Begin());
  TF_EXPECT_OK(mock_metadata_source.Commit());
  TF_EXPECT_OK(mock_metadata_source.Begin());
  TF_EXPECT_OK(mock_metadata_source.Commit());
}

TEST(MetadataSourceTest, TestCommitWithoutConnect) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, CommitImpl()).Times(0);
  tensorflow::Status s = mock_metadata_source.Commit();
  EXPECT_EQ(s.code(), tensorflow::error::FAILED_PRECONDITION);
}

TEST(MetadataSourceTest, TestCommitWithoutBegin) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, CommitImpl()).Times(0);
  TF_EXPECT_OK(mock_metadata_source.Connect());
  tensorflow::Status s = mock_metadata_source.Commit();
  EXPECT_EQ(s.code(), tensorflow::error::FAILED_PRECONDITION);
}

TEST(MetadataSourceTest, TestBeginAndRollback) {
  MockMetadataSource mock_metadata_source;
  {
    ::testing::InSequence call_seq;
    EXPECT_CALL(mock_metadata_source, BeginImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, RollbackImpl()).Times(1);
  }
  TF_EXPECT_OK(mock_metadata_source.Connect());
  TF_EXPECT_OK(mock_metadata_source.Begin());
  TF_EXPECT_OK(mock_metadata_source.Rollback());
}

TEST(MetadataSourceTest, TestBeginAndRollbackTwice) {
  MockMetadataSource mock_metadata_source;
  {
    ::testing::InSequence call_seq;
    EXPECT_CALL(mock_metadata_source, BeginImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, RollbackImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, BeginImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, RollbackImpl()).Times(1);
  }

  TF_EXPECT_OK(mock_metadata_source.Connect());
  TF_EXPECT_OK(mock_metadata_source.Begin());
  TF_EXPECT_OK(mock_metadata_source.Rollback());
  TF_EXPECT_OK(mock_metadata_source.Begin());
  TF_EXPECT_OK(mock_metadata_source.Rollback());
}

TEST(MetadataSourceTest, TestRollbackWithoutBegin) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, RollbackImpl()).Times(0);
  TF_EXPECT_OK(mock_metadata_source.Connect());
  tensorflow::Status s = mock_metadata_source.Rollback();
  EXPECT_EQ(s.code(), tensorflow::error::FAILED_PRECONDITION);
}

TEST(MetadataSourceTest, TestRollbackWithoutConnect) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, RollbackImpl()).Times(0);
  tensorflow::Status s = mock_metadata_source.Rollback();
  EXPECT_EQ(s.code(), tensorflow::error::FAILED_PRECONDITION);
}

TEST(MetadataSourceTest, TestBeginWithoutConnect) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, BeginImpl()).Times(0);
  tensorflow::Status s = mock_metadata_source.Begin();
  EXPECT_EQ(s.code(), tensorflow::error::FAILED_PRECONDITION);
}

}  // namespace ml_metadata
