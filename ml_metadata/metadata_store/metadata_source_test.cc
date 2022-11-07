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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ml_metadata/proto/metadata_source.pb.h"

namespace ml_metadata {

class MockMetadataSource : public MetadataSource {
 public:
  MOCK_METHOD(absl::Status, ConnectImpl, (), (override));
  MOCK_METHOD(absl::Status, CloseImpl, (), (override));
  MOCK_METHOD(absl::Status, BeginImpl, (), (override));
  MOCK_METHOD(absl::Status, ExecuteQueryImpl,
              (const std::string& query, RecordSet* results), (override));
  MOCK_METHOD(absl::Status, CommitImpl, (), (override));
  MOCK_METHOD(absl::Status, RollbackImpl, (), (override));
  MOCK_METHOD(std::string, EscapeString, (absl::string_view value),
              (const, override));
  MOCK_METHOD(std::string, EncodeBytes, (absl::string_view value), (const));
  MOCK_METHOD(absl::StatusOr<std::string>, DecodeBytes,
              (absl::string_view value), (const));
};

TEST(MetadataSourceTest, ConnectAgainWithoutClose) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, ConnectImpl()).Times(1);
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  absl::Status s = mock_metadata_source.Connect();
  EXPECT_TRUE(absl::IsFailedPrecondition(s));
}

TEST(MetadataSourceTest, CloseWithoutConnect) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, CloseImpl()).Times(0);

  absl::Status s = mock_metadata_source.Close();
  EXPECT_TRUE(absl::IsFailedPrecondition(s));
}

TEST(MetadataSourceTest, ConnectThenCloseThenConnectAgain) {
  MockMetadataSource mock_metadata_source;
  {
    ::testing::InSequence call_seq;
    EXPECT_CALL(mock_metadata_source, ConnectImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, CloseImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, ConnectImpl()).Times(1);
  }

  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Close());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
}

TEST(MetadataSourceTest, TestExecuteQueryWithoutConnect) {
  MockMetadataSource mock_metadata_source;
  std::string query = "some query";
  RecordSet result;
  EXPECT_CALL(mock_metadata_source, ExecuteQueryImpl(query, &result)).Times(0);
  absl::Status s = mock_metadata_source.ExecuteQuery(query, &result);
  EXPECT_TRUE(absl::IsFailedPrecondition(s));
}

TEST(MetadataSourceTest, TestExecuteQueryWithoutBegin) {
  MockMetadataSource mock_metadata_source;
  std::string query = "some query";
  RecordSet result;
  EXPECT_CALL(mock_metadata_source, ExecuteQueryImpl(query, &result)).Times(0);
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  absl::Status s = mock_metadata_source.ExecuteQuery(query, &result);
  EXPECT_TRUE(absl::IsFailedPrecondition(s));
}

TEST(MetadataSourceTest, TestBeginAndCommit) {
  MockMetadataSource mock_metadata_source;
  {
    ::testing::InSequence call_seq;
    EXPECT_CALL(mock_metadata_source, BeginImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, CommitImpl()).Times(1);
  }
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Begin());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Commit());
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
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Begin());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Commit());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Begin());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Commit());
}


TEST(MetadataSourceTest, TestCommitWithoutConnect) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, CommitImpl()).Times(0);
  absl::Status s = mock_metadata_source.Commit();
  EXPECT_TRUE(absl::IsFailedPrecondition(s));
}

TEST(MetadataSourceTest, TestCommitWithoutBegin) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, CommitImpl()).Times(0);
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  absl::Status s = mock_metadata_source.Commit();
  EXPECT_TRUE(absl::IsFailedPrecondition(s));
}

TEST(MetadataSourceTest, TestBeginAndRollback) {
  MockMetadataSource mock_metadata_source;
  {
    ::testing::InSequence call_seq;
    EXPECT_CALL(mock_metadata_source, BeginImpl()).Times(1);
    EXPECT_CALL(mock_metadata_source, RollbackImpl()).Times(1);
  }
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Begin());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Rollback());
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

  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Begin());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Rollback());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Begin());
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Rollback());
}

TEST(MetadataSourceTest, TestRollbackWithoutBegin) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, RollbackImpl()).Times(0);
  EXPECT_EQ(absl::OkStatus(), mock_metadata_source.Connect());
  absl::Status s = mock_metadata_source.Rollback();
  EXPECT_TRUE(absl::IsFailedPrecondition(s));
}

TEST(MetadataSourceTest, TestRollbackWithoutConnect) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, RollbackImpl()).Times(0);
  absl::Status s = mock_metadata_source.Rollback();
  EXPECT_TRUE(absl::IsFailedPrecondition(s));
}

TEST(MetadataSourceTest, TestBeginWithoutConnect) {
  MockMetadataSource mock_metadata_source;
  EXPECT_CALL(mock_metadata_source, BeginImpl()).Times(0);
  absl::Status s = mock_metadata_source.Begin();
  EXPECT_TRUE(absl::IsFailedPrecondition(s));
}


}  // namespace ml_metadata
