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

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "ml_metadata/metadata_store/metadata_store_test_suite.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {
namespace testing {
namespace {

using ::testing::SizeIs;
using ::testing::UnorderedElementsAreArray;
using ::testing::UnorderedPointwise;

std::unique_ptr<MetadataStore> CreateMetadataStore() {
  auto metadata_source =
      absl::make_unique<SqliteMetadataSource>(SqliteMetadataSourceConfig());
  auto transaction_executor =
      absl::make_unique<RdbmsTransactionExecutor>(metadata_source.get());

  std::unique_ptr<MetadataStore> metadata_store;
  CHECK_EQ(
      absl::OkStatus(),
      MetadataStore::Create(util::GetSqliteMetadataSourceQueryConfig(), {},
                            std::move(metadata_source),
                            std::move(transaction_executor), &metadata_store));
  CHECK_EQ(absl::OkStatus(), metadata_store->InitMetadataStore());
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



}  // namespace

INSTANTIATE_TEST_SUITE_P(
    MetadataStoreTest, MetadataStoreTestSuite, ::testing::Values([]() {
      return absl::make_unique<RDBMSMetadataStoreContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata
