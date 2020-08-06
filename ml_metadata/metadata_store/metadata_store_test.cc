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
#include "ml_metadata/metadata_store/metadata_store.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_store_test_suite.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"

namespace ml_metadata {
namespace testing {
namespace {

std::unique_ptr<MetadataStore> CreateMetadataStore() {
  auto metadata_source =
      absl::make_unique<SqliteMetadataSource>(SqliteMetadataSourceConfig());
  auto transaction_executor =
      absl::make_unique<RdbmsTransactionExecutor>(metadata_source.get());

  std::unique_ptr<MetadataStore> metadata_store;
  TF_CHECK_OK(MetadataStore::Create(util::GetSqliteMetadataSourceQueryConfig(),
                                    {}, std::move(metadata_source),
                                    std::move(transaction_executor),
                                    &metadata_store));
  TF_CHECK_OK(metadata_store->InitMetadataStore());
  return metadata_store;
}

class RDBMSMetadataStoreContainer : public MetadataStoreContainer {
 public:
  RDBMSMetadataStoreContainer() : MetadataStoreContainer() {
    metadata_store_ = CreateMetadataStore();
  }

  ~RDBMSMetadataStoreContainer() override = default;

  MetadataStore* GetMetadataStore() override { return metadata_store_.get(); }

 private:
  // MetadataStore that is initialized at RDBMSMetadataStoreContainer
  // construction time.
  std::unique_ptr<MetadataStore> metadata_store_;
};

TEST(MetadataStoreExtendedTest, SpecifyDowngradeMigrationWhenCreate) {
  // create the metadata store first, and init the schema to
  // the library version
  const MetadataSourceQueryConfig& query_config =
      util::GetSqliteMetadataSourceQueryConfig();
  std::string filename_uri =
      absl::StrCat(::testing::TempDir(), "test_shared.db");
  SqliteMetadataSourceConfig connection_config;
  connection_config.set_filename_uri(filename_uri);

  {
    std::unique_ptr<MetadataStore> metadata_store;
    auto metadata_source =
        absl::make_unique<SqliteMetadataSource>(connection_config);
    auto transaction_executor =
        absl::make_unique<RdbmsTransactionExecutor>(metadata_source.get());
    TF_EXPECT_OK(MetadataStore::Create(
        query_config, {}, std::move(metadata_source),
        std::move(transaction_executor), &metadata_store));
    TF_ASSERT_OK(metadata_store->InitMetadataStore());
  }

  // Create another metadata store, and test when migration_options are given
  {
    std::unique_ptr<MetadataStore> other_metadata_store;
    auto metadata_source =
        absl::make_unique<SqliteMetadataSource>(connection_config);
    auto transaction_executor =
        absl::make_unique<RdbmsTransactionExecutor>(metadata_source.get());
    MigrationOptions options;
    options.set_downgrade_to_schema_version(query_config.schema_version() + 1);
    tensorflow::Status s = MetadataStore::Create(
        query_config, options, std::move(metadata_source),
        std::move(transaction_executor), &other_metadata_store);
    EXPECT_EQ(s.code(), tensorflow::error::INVALID_ARGUMENT);
    EXPECT_EQ(other_metadata_store, nullptr);
  }

  {
    std::unique_ptr<MetadataStore> other_metadata_store;
    auto metadata_source =
        absl::make_unique<SqliteMetadataSource>(connection_config);
    auto transaction_executor =
        absl::make_unique<RdbmsTransactionExecutor>(metadata_source.get());
    MigrationOptions options;
    options.set_downgrade_to_schema_version(0);
    tensorflow::Status s = MetadataStore::Create(
        query_config, options, std::move(metadata_source),
        std::move(transaction_executor), &other_metadata_store);
    EXPECT_EQ(s.code(), tensorflow::error::CANCELLED);
    EXPECT_TRUE(absl::StrContains(s.error_message(),
                                  "Downgrade migration was performed."));
    EXPECT_EQ(other_metadata_store, nullptr);
  }
  TF_EXPECT_OK(tensorflow::Env::Default()->DeleteFile(filename_uri));
}


}  // namespace

INSTANTIATE_TEST_CASE_P(
    MetadataStoreTest, MetadataStoreTestSuite, ::testing::Values([]() {
      return absl::make_unique<RDBMSMetadataStoreContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata
