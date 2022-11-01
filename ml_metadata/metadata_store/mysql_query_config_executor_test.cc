/* Copyright 2022 Google LLC

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
// Test suite for a mysql query config-based QueryExecutor.
#include <memory>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/query_config_executor.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/metadata_store/query_executor_test.h"
#include "ml_metadata/metadata_store/test_mysql_metadata_source_initializer.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"

namespace ml_metadata {
namespace testing {

namespace {
// MySqlQueryConfigExecutorContainer implements
// QueryConfigExecutorContainer to generate and retrieve a
// QueryExecutor based on a MySqlMetadataSource.
class MySqlQueryConfigExecutorContainer : public QueryConfigExecutorContainer {
 public:
  MySqlQueryConfigExecutorContainer()
      : QueryConfigExecutorContainer(
            util::GetMySqlMetadataSourceQueryConfig()) {
    metadata_source_initializer_ = GetTestMySqlMetadataSourceInitializer();
    metadata_source_ = metadata_source_initializer_->Init(
        TestMySqlMetadataSourceInitializer::ConnectionType::kTcp);
    if (!metadata_source_->is_connected())
      CHECK_EQ(absl::OkStatus(), metadata_source_->Connect());
    query_executor_ = absl::WrapUnique(new QueryConfigExecutor(
        util::GetMySqlMetadataSourceQueryConfig(), metadata_source_));
  }

  ~MySqlQueryConfigExecutorContainer() override = default;

  MetadataSource* GetMetadataSource() override {
    return metadata_source_;
  }
  QueryExecutor* GetQueryExecutor() override { return query_executor_.get(); }

 private:
  // An unowned TestMySqlMetadataSourceInitializer from a call to
  // GetTestMySqlMetadataSourceInitializer().
  std::unique_ptr<TestMySqlMetadataSourceInitializer>
      metadata_source_initializer_;
  // An unowned MySqlMetadataSource from a call to
  // metadata_source_initializer->Init().
  MySqlMetadataSource* metadata_source_;
  std::unique_ptr<QueryExecutor> query_executor_;
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    MySqlQueryConfigExecutorContainer, QueryExecutorTest,
    ::testing::Values([]() {
      return std::make_unique<MySqlQueryConfigExecutorContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata
