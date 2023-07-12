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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_RDBMS_METADATA_ACCESS_OBJECT_TEST_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_RDBMS_METADATA_ACCESS_OBJECT_TEST_H_

#include <memory>
#include <vector>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "ml_metadata/metadata_store/rdbms_metadata_access_object.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {

// An Interface to generate and retrieve a RDBMSMetadataAccessObject.
class RDBMSMetadataAccessObjectContainer {
 public:
  virtual ~RDBMSMetadataAccessObjectContainer() = default;

  // MetadataSource is owned by RDBMSMetadataAccessObjectContainer.
  virtual MetadataSource* GetMetadataSource() = 0;

  // MetadataAccessObject is owned by MetadataAccessObjectContainer.
  virtual RDBMSMetadataAccessObject* GetRDBMSMetadataAccessObject() = 0;

  // Init a test db environment. By default the testsuite is run against the
  // head schema. If GetSchemaVersion() is overridden, it prepares a
  // db at tht particular schema version.
  virtual absl::Status Init() {
    MLMD_RETURN_IF_ERROR(GetRDBMSMetadataAccessObject()->InitMetadataSource());
    return absl::OkStatus();
  }
};

// An Interface to generate and retrieve a RDBMSMetadataAccessObject, where
// there is an associated MetadataSourceQueryConfig.
class QueryConfigRDBMSMetadataAccessObjectContainer
    : public RDBMSMetadataAccessObjectContainer {
 public:
  // By default the container returns a query config-based
  // RDBMSMetadataAccessObject that uses the query config at head.
  QueryConfigRDBMSMetadataAccessObjectContainer(
      const MetadataSourceQueryConfig& config)
      : config_(config) {}

  virtual ~QueryConfigRDBMSMetadataAccessObjectContainer() = default;

 private:
  MetadataSourceQueryConfig config_;
};

// Represents the type of the Gunit Test param for the parameterized
// RDBMSMetadataAccessObjectTest.
using RDBMSMetadataAccessObjectContainerFactory =
    std::function<std::unique_ptr<RDBMSMetadataAccessObjectContainer>()>;

// A parameterized abstract test fixture to run tests for
// RDBMSMetadataAccessObject created with different MetadataSource types.
// See rdbms_metadata_access_object_test.cc for list of test cases using this
// fixture.
class RDBMSMetadataAccessObjectTest
    : public ::testing::TestWithParam<
          RDBMSMetadataAccessObjectContainerFactory> {
 protected:
  void SetUp() override {
    metadata_access_object_container_ = GetParam()();
    metadata_source_ = metadata_access_object_container_->GetMetadataSource();
    rdbms_metadata_access_object_ =
        metadata_access_object_container_->GetRDBMSMetadataAccessObject();
    CHECK_EQ(absl::OkStatus(), metadata_source_->Begin());
  }

  void TearDown() override {
    CHECK_EQ(absl::OkStatus(), metadata_source_->Commit());
    metadata_source_ = nullptr;
    rdbms_metadata_access_object_ = nullptr;
    metadata_access_object_container_ = nullptr;
  }

  absl::Status Init() { return metadata_access_object_container_->Init(); }

  template <typename MessageType>
  void VerifyFindTypesFromRecordSet(const RecordSet& records,
                                    std::vector<MessageType> expected_types);

  template <typename MessageType>
  absl::Status FindTypesFromRecordSet(const RecordSet& type_record_set,
                                      std::vector<MessageType>* types,
                                      bool get_properties) {
    return rdbms_metadata_access_object_->FindTypesFromRecordSet(
        type_record_set, types, get_properties);
  }

  template <typename MessageType>
  absl::Status FindTypesImpl(absl::Span<const int64_t> type_ids,
                             bool get_properties,
                             std::vector<MessageType>& types) {
    return rdbms_metadata_access_object_->FindTypesImpl(type_ids,
                                                        get_properties, types);
  }

  template <typename Type>
  absl::Status FindParentTypesByTypeIdImpl(
      absl::Span<const int64_t> type_ids,
      absl::flat_hash_map<int64_t, Type>& output_parent_types) {
    return rdbms_metadata_access_object_->FindParentTypesByTypeIdImpl(
        type_ids, output_parent_types);
  }

  template <typename MessageType>
  absl::Status CreateType(const MessageType& type, int64_t* type_id) {
    return rdbms_metadata_access_object_->CreateType(type, type_id);
  }

  template <typename Node, typename NodeType>
  absl::Status CreateNodeImpl(const Node& node, const NodeType& node_type,
                              int64_t* node_id) {
    return rdbms_metadata_access_object_->CreateNodeImpl<Node, NodeType>(
        node, /*skip_type_and_property_validation=*/false,
        /*create_timestamp=*/absl::Now(), node_id);
  }

  template <typename Node, typename NodeType>
  absl::Status FindNodesWithTypeImpl(absl::Span<const int64_t> node_ids,
                                     std::vector<Node>& nodes,
                                     std::vector<NodeType>& node_types) {
    return rdbms_metadata_access_object_->FindNodesWithTypesImpl(
        node_ids, nodes, node_types);
  }

  std::unique_ptr<RDBMSMetadataAccessObjectContainer>
      metadata_access_object_container_;

  // metadata_source_ and rdbms_metadata_access_object_ are unowned.
  MetadataSource* metadata_source_;
  RDBMSMetadataAccessObject* rdbms_metadata_access_object_;
};

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_RDBMS_METADATA_ACCESS_OBJECT_TEST_H_
