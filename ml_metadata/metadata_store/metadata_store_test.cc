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
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "ml_metadata/metadata_store/metadata_store_test_suite.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
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
      std::make_unique<SqliteMetadataSource>(SqliteMetadataSourceConfig());
  auto transaction_executor =
      std::make_unique<RdbmsTransactionExecutor>(metadata_source.get());

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



// Creates the following lineage graph for deleting lineage.
//  a_0 a_1     a_3
//   |    \   /     \
//  e_0    e_1        e_3
//                 /       \
//     a_2     a_4(LIVE)    a_5
//        \    /      |
//         e_2        |
// -------------------t --------->
// Returns `creation_time_threshold` that is greater than the creation time of
// {a_0, a_1, a_2, a_3, a_4}. In all artifacts, only a_4 is LIVE.
absl::Status CreateLineageGraph(MetadataStore& metadata_store,
                                int64_t& creation_time_threshold,
                                std::vector<Artifact>& want_artifacts,
                                std::vector<Execution>& want_executions) {
  const PutTypesRequest put_types_req =
      ParseTextProtoOrDie<PutTypesRequest>(R"(
        artifact_types: { name: 't1' properties { key: 'p1' value: STRING } }
        execution_types: { name: 't2' properties { key: 'p2' value: STRING } }
        context_types: { name: 't3' properties { key: 'p3' value: STRING } }
      )");
  PutTypesResponse put_types_resp;
  MLMD_RETURN_IF_ERROR(metadata_store.PutTypes(put_types_req, &put_types_resp));

  // insert artifacts
  auto put_artifact = [&metadata_store, &put_types_resp, &want_artifacts](
                          const std::string& label,
                          const absl::optional<Artifact::State>& state) {
    PutArtifactsRequest put_artifacts_req;
    Artifact* artifact = put_artifacts_req.add_artifacts();
    artifact->set_uri(absl::StrCat("uri://foo/", label));
    artifact->set_type_id(put_types_resp.artifact_type_ids(0));
    (*artifact->mutable_properties())["p1"].set_string_value(label);
    if (state) {
      artifact->set_state(state.value());
    }
    PutArtifactsResponse resp;
    CHECK_EQ(absl::OkStatus(),
             metadata_store.PutArtifacts(put_artifacts_req, &resp));
    artifact->set_id(resp.artifact_ids(0));
    want_artifacts.push_back(*artifact);
    absl::SleepFor(absl::Milliseconds(1));
  };

  put_artifact(/*label=*/"a0", /*state=*/absl::nullopt);
  put_artifact(/*label=*/"a1", /*state=*/absl::nullopt);
  put_artifact(/*label=*/"a2", /*state=*/Artifact::UNKNOWN);
  put_artifact(/*label=*/"a3", /*state=*/Artifact::DELETED);
  put_artifact(/*label=*/"a4", /*state=*/Artifact::LIVE);

  // insert executions and links to artifacts
  auto put_execution = [&metadata_store, &put_types_resp, &want_executions](
                           const std::string& label,
                           const std::vector<int64_t>& input_ids,
                           const std::vector<int64_t>& output_ids) {
    PutExecutionRequest req;
    Execution* execution = req.mutable_execution();
    execution->set_type_id(put_types_resp.execution_type_ids(0));
    (*execution->mutable_properties())["p2"].set_string_value(label);
    want_executions.push_back(*execution);
    // database id starts from 1.
    for (int64_t id : input_ids) {
      Event* event = req.add_artifact_event_pairs()->mutable_event();
      event->set_artifact_id(id + 1);
      event->set_type(Event::INPUT);
    }
    for (int64_t id : output_ids) {
      Event* event = req.add_artifact_event_pairs()->mutable_event();
      event->set_artifact_id(id + 1);
      event->set_type(Event::OUTPUT);
    }
    PutExecutionResponse resp;
    ASSERT_EQ(absl::OkStatus(), metadata_store.PutExecution(req, &resp));
  };

  // Creates executions and connects lineage
  put_execution(/*label=*/"e0", /*input_ids=*/{0}, /*output_ids=*/{});
  put_execution(/*label=*/"e1", /*input_ids=*/{1}, /*output_ids=*/{3});
  put_execution(/*label=*/"e2", /*input_ids=*/{2}, /*output_ids=*/{4});

  // e3 and a5 are generated after the timestamp.
  creation_time_threshold = absl::ToUnixMillis(absl::Now());
  absl::SleepFor(absl::Milliseconds(1));
  put_artifact(/*label=*/"a5", /*state=*/absl::nullopt);
  put_execution(/*label=*/"e3", /*input_ids=*/{3, 4}, /*output_ids=*/{5});
  return absl::OkStatus();
}


// The utilities for testing the GetLineageGraph.
// Verify the `subgraph` contains the specified nodes and relations.
void VerifySubgraph(const LineageGraph& subgraph,
                    const std::vector<Artifact>& want_artifacts,
                    const std::vector<Execution>& want_executions,
                    const std::vector<std::pair<int64_t, int64_t>>& want_events,
                    std::unique_ptr<MetadataStore>& metadata_store) {
  // Compare nodes and edges.
  EXPECT_THAT(subgraph.artifacts(),
              UnorderedPointwise(EqualsProto<Artifact>(/*ignore_fields=*/{
                                     "id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}),
                                 want_artifacts));

  EXPECT_THAT(subgraph.executions(),
              UnorderedPointwise(EqualsProto<Execution>(/*ignore_fields=*/{
                                     "id", "type", "create_time_since_epoch",
                                     "last_update_time_since_epoch"}),
                                 want_executions));
  EXPECT_THAT(subgraph.events(), SizeIs(want_events.size()));
  std::vector<std::pair<int64_t, int64_t>> got_events;
  for (const Event& event : subgraph.events()) {
    got_events.push_back({event.artifact_id() - 1, event.execution_id() - 1});
  }
  EXPECT_THAT(got_events, UnorderedElementsAreArray(want_events));
  // Compare types.
  GetArtifactTypesResponse artifact_types_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store->GetArtifactTypes({}, &artifact_types_response));
  EXPECT_THAT(subgraph.artifact_types(),
              UnorderedPointwise(EqualsProto<ArtifactType>(),
                                 artifact_types_response.artifact_types()));
  GetExecutionTypesResponse execution_types_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store->GetExecutionTypes({}, &execution_types_response));
  EXPECT_THAT(subgraph.execution_types(),
              UnorderedPointwise(EqualsProto<ExecutionType>(),
                                 execution_types_response.execution_types()));
  GetContextTypesResponse context_types_response;
  ASSERT_EQ(absl::OkStatus(),
            metadata_store->GetContextTypes({}, &context_types_response));
  EXPECT_THAT(subgraph.context_types(),
              UnorderedPointwise(EqualsProto<ContextType>(),
                                 context_types_response.context_types()));
}

// Test valid query options when using GetLineageGraph on the lineage graph
// created with `CreateLineageGraph`.
TEST(MetadataStoreExtendedTest, GetLineageGraphWithMaxHops) {
  // Prepare a store with the lineage graph
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  int64_t min_creation_time;
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  ASSERT_EQ(absl::OkStatus(),
            CreateLineageGraph(*metadata_store, min_creation_time,
                               want_artifacts, want_executions));

  // Verify the query results with the specified max_num_hop
  auto verify_lineage_graph_with_max_num_hop =
      [&metadata_store](
          absl::optional<int64_t> max_num_hop,
          const std::vector<Artifact>& want_artifacts,
          const std::vector<Execution>& want_executions,
          const std::vector<std::pair<int64_t, int64_t>>& want_events) {
        GetLineageGraphRequest req;
        GetLineageGraphResponse resp;
        req.mutable_options()->mutable_artifacts_options()->set_filter_query(
            "uri = 'uri://foo/a4'");
        if (max_num_hop) {
          LOG(INFO) << "Test when max_num_hops = " << *max_num_hop;
          req.mutable_options()->mutable_stop_conditions()->set_max_num_hops(
              *max_num_hop);
        } else {
          LOG(INFO) << "Test when max_num_hops is unset.";
        }
        EXPECT_EQ(absl::OkStatus(),
                  metadata_store->GetLineageGraph(req, &resp));

        VerifySubgraph(resp.subgraph(), want_artifacts, want_executions,
                       want_events, metadata_store);
      };

  // Verify the lineage graph query results by increasing the max_num_hops.
  verify_lineage_graph_with_max_num_hop(/*max_num_hop=*/0,
                                        /*want_artifacts=*/{want_artifacts[4]},
                                        /*want_executions=*/{},
                                        /*want_events=*/{});

  verify_lineage_graph_with_max_num_hop(/*max_num_hop=*/1,
      /*want_artifacts=*/{want_artifacts[4]},
      /*want_executions=*/{want_executions[2], want_executions[3]},
      /*want_events=*/{{4, 2}, {4, 3}});

  verify_lineage_graph_with_max_num_hop(/*max_num_hop=*/2,
      /*want_artifacts=*/{want_artifacts[2], want_artifacts[3],
                          want_artifacts[4], want_artifacts[5]},
      /*want_executions=*/{want_executions[2], want_executions[3]},
      /*want_events=*/{{4, 2}, {4, 3}, {2, 2}, {3, 3}, {5, 3}});

  verify_lineage_graph_with_max_num_hop(/*max_num_hop=*/3,
      /*want_artifacts=*/
      {want_artifacts[2], want_artifacts[3], want_artifacts[4],
       want_artifacts[5]},
      /*want_executions=*/
      {want_executions[1], want_executions[2], want_executions[3]},
      /*want_events=*/{{4, 2}, {4, 3}, {2, 2}, {3, 3}, {5, 3}, {3, 1}});

  verify_lineage_graph_with_max_num_hop(/*max_num_hop=*/4,
      /*want_artifacts=*/
      {want_artifacts[1], want_artifacts[2], want_artifacts[3],
       want_artifacts[4], want_artifacts[5]},
      /*want_executions=*/
      {want_executions[1], want_executions[2], want_executions[3]},
      /*want_events=*/{{4, 2}, {4, 3}, {2, 2}, {3, 3}, {5, 3}, {3, 1}, {1, 1}});

  verify_lineage_graph_with_max_num_hop(/*max_num_hop=*/absl::nullopt,
      /*want_artifacts=*/
      {want_artifacts[1], want_artifacts[2], want_artifacts[3],
       want_artifacts[4], want_artifacts[5]},
      /*want_executions=*/
      {want_executions[1], want_executions[2], want_executions[3]},
      /*want_events=*/{{4, 2}, {4, 3}, {2, 2}, {3, 3}, {5, 3}, {3, 1}, {1, 1}});
}

// Test valid query options when using GetLineageGraph on the lineage graph
// created with `CreateLineageGraph`.
TEST(MetadataStoreExtendedTest, GetLineageGraphWithMaxNodeSize) {
  // Prepare a store with the lineage graph
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  int64_t min_creation_time;
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  ASSERT_EQ(absl::OkStatus(),
            CreateLineageGraph(*metadata_store, min_creation_time,
                               want_artifacts, want_executions));

  // Verify the query results with the specified max_node_size
  auto verify_lineage_graph_with_max_node_size =
      [&metadata_store](
          absl::optional<int64_t> max_node_size,
          const std::vector<Artifact>& want_artifacts,
          const std::vector<Execution>& want_executions,
          const std::vector<std::pair<int64_t, int64_t>>& want_events,
          bool artifact_requires_live_state = false) {
        GetLineageGraphRequest req;
        GetLineageGraphResponse resp;
        if (artifact_requires_live_state) {
          req.mutable_options()->mutable_artifacts_options()->set_filter_query(
              "type = 't1' and state = LIVE");
        } else {
          req.mutable_options()->mutable_artifacts_options()->set_filter_query(
              "type = 't1'");
        }
        req.mutable_options()
            ->mutable_artifacts_options()
            ->mutable_order_by_field()
            ->set_is_asc(true);

        if (max_node_size.has_value()) {
          LOG(INFO) << "Test when max_node_size = " << *max_node_size;
          req.mutable_options()->set_max_node_size(*max_node_size);
        } else {
          LOG(INFO) << "Test when max_node_size is unset.";
        }
        EXPECT_EQ(absl::OkStatus(),
                  metadata_store->GetLineageGraph(req, &resp));

        VerifySubgraph(resp.subgraph(), want_artifacts, want_executions,
                       want_events, metadata_store);
      };

  // Verify the lineage graph query results by increasing the max_node_size.
  // Return every related node if max_node_size is absl::nullopt.
  verify_lineage_graph_with_max_node_size(
      /*max_node_size=*/absl::nullopt,
      /*want_artifacts=*/want_artifacts,
      /*want_executions=*/want_executions,
      /*want_events=*/
      {{4, 2}, {0, 0}, {4, 3}, {2, 2}, {3, 3}, {5, 3}, {3, 1}, {1, 1}});

  // Return every related node if max_node_size <= 0.
  verify_lineage_graph_with_max_node_size(
      /*max_node_size=*/0,
      /*want_artifacts=*/want_artifacts,
      /*want_executions=*/want_executions,
      /*want_events=*/
      {{4, 2}, {0, 0}, {4, 3}, {2, 2}, {3, 3}, {5, 3}, {3, 1}, {1, 1}});

  verify_lineage_graph_with_max_node_size(
      /*max_node_size=*/-1,
      /*want_artifacts=*/want_artifacts,
      /*want_executions=*/want_executions,
      /*want_events=*/
      {{4, 2}, {0, 0}, {4, 3}, {2, 2}, {3, 3}, {5, 3}, {3, 1}, {1, 1}});

  verify_lineage_graph_with_max_node_size(
      /*max_node_size=*/1,
      /*want_artifacts=*/{want_artifacts[0]},
      /*want_executions=*/{},
      /*want_events=*/{});

  verify_lineage_graph_with_max_node_size(
      /*max_node_size=*/6,
      /*want_artifacts=*/want_artifacts,
      /*want_executions=*/{},
      /*want_events=*/{});

  verify_lineage_graph_with_max_node_size(
      /*max_node_size=*/3,
      /*want_artifacts=*/{want_artifacts[4]},
      /*want_executions=*/{want_executions[2], want_executions[3]},
      /*want_events=*/{{4, 2}, {4, 3}},
      /*artifact_requires_live_state=*/true);

  verify_lineage_graph_with_max_node_size(
      /*max_num_hop=*/20,
      /*want_artifacts=*/want_artifacts,
      /*want_executions=*/want_executions,
      /*want_events=*/
      {{4, 2}, {0, 0}, {4, 3}, {2, 2}, {3, 3}, {5, 3}, {3, 1}, {1, 1}});
}

TEST(MetadataStoreExtendedTest, GetLineageGraphWithBoundaryConditions) {
  // Prepare a store with the lineage graph
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  int64_t min_creation_time;
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  ASSERT_EQ(absl::OkStatus(),
            CreateLineageGraph(*metadata_store, min_creation_time,
                               want_artifacts, want_executions));

  // Verify the query results about a4 with the boundary conditions. The
  // max_num_hops is set to 3. a_1, a_0, e_0 will not be included in the result.
  auto verify_lineage_graph_with_boundary_conditions =
      [&metadata_store](
          absl::optional<std::string> boundary_artifacts,
          absl::optional<std::string> boundary_executions,
          const std::vector<Artifact>& want_artifacts,
          const std::vector<Execution>& want_executions,
          const std::vector<std::pair<int64_t, int64_t>>& want_events) {
        GetLineageGraphRequest req;
        GetLineageGraphResponse resp;
        req.mutable_options()->mutable_artifacts_options()->set_filter_query(
            "uri = 'uri://foo/a4'");
        req.mutable_options()->mutable_stop_conditions()->set_max_num_hops(3);
        if (boundary_artifacts) {
          req.mutable_options()
              ->mutable_stop_conditions()
              ->set_boundary_artifacts(*boundary_artifacts);
          LOG(INFO) << "Test when boundary_artifacts = " << *boundary_artifacts;
        }
        if (boundary_executions) {
          req.mutable_options()
              ->mutable_stop_conditions()
              ->set_boundary_executions(*boundary_executions);
          LOG(INFO) << "Test when boundary_executions = "
                    << *boundary_executions;
        }
        EXPECT_EQ(absl::OkStatus(),
                  metadata_store->GetLineageGraph(req, &resp));
        VerifySubgraph(resp.subgraph(), want_artifacts, want_executions,
                       want_events, metadata_store);
      };

  // When no boundary is set, all nodes can be reached.
  verify_lineage_graph_with_boundary_conditions(
      /*boundary_artifacts=*/absl::nullopt,
      /*boundary_executions=*/absl::nullopt,
      /*want_artifacts=*/
      {want_artifacts[2], want_artifacts[3], want_artifacts[4],
       want_artifacts[5]},
      /*want_executions=*/
      {want_executions[1], want_executions[2], want_executions[3]},
      /*want_events=*/{{4, 2}, {4, 3}, {2, 2}, {3, 3}, {5, 3}, {3, 1}});

  // Set boundary condition at a3, {a3, e1, a1} are excluded.
  verify_lineage_graph_with_boundary_conditions(
      /*boundary_artifacts=*/"uri != 'uri://foo/a3'",
      /*boundary_executions=*/absl::nullopt,
      /*want_artifacts=*/
      {want_artifacts[2], want_artifacts[4], want_artifacts[5]},
      /*want_executions=*/
      {want_executions[2], want_executions[3]},
      /*want_events=*/{{4, 2}, {4, 3}, {2, 2}, {5, 3}});

  // Set boundary condition at a3 and e3, {a3, e1, a1, e3, a5} are excluded.
  verify_lineage_graph_with_boundary_conditions(
      /*boundary_artifacts=*/"uri != 'uri://foo/a3'",
      /*boundary_executions=*/"properties.p2.string_value != 'e3'",
      /*want_artifacts=*/{want_artifacts[2], want_artifacts[4]},
      /*want_executions=*/{want_executions[2]},
      /*want_events=*/{{4, 2}, {2, 2}});

  // Set boundary condition at a3 and e2, {a3, e1, a1, e2, a2} are excluded.
  verify_lineage_graph_with_boundary_conditions(
      /*boundary_artifacts=*/"uri != 'uri://foo/a3'",
      /*boundary_executions=*/"properties.p2.string_value != 'e2'",
      /*want_artifacts=*/{want_artifacts[4], want_artifacts[5]},
      /*want_executions=*/{want_executions[3]},
      /*want_events=*/{{4, 3}, {5, 3}});
}

TEST(MetadataStoreExtendedTest, GetLineageGraphErrors) {
  // Prepare a store with the lineage graph
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  int64_t min_creation_time;
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  ASSERT_EQ(absl::OkStatus(),
            CreateLineageGraph(*metadata_store, min_creation_time,
                               want_artifacts, want_executions));

  GetLineageGraphResponse resp;
  {
    // No query_nodes conditions.
    GetLineageGraphRequest req;
    absl::Status status = metadata_store->GetLineageGraph(req, &resp);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  {
    // invalid query syntax
    GetLineageGraphRequest req;
    req.mutable_options()->mutable_artifacts_options()->set_filter_query(
        "invalid query syntax");
    absl::Status status = metadata_store->GetLineageGraph(req, &resp);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  {
    // invalid max_num_hops.
    GetLineageGraphRequest req;
    req.mutable_options()->mutable_artifacts_options()->set_filter_query(
        "uri = 'uri://foo/bar/'");
    req.mutable_options()->mutable_stop_conditions()->set_max_num_hops(-1);
    absl::Status status = metadata_store->GetLineageGraph(req, &resp);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  {
    // query_nodes does not match any nodes.
    GetLineageGraphRequest req;
    req.mutable_options()->mutable_artifacts_options()->set_filter_query(
        "uri = 'uri://foo/bar/'");
    absl::Status status = metadata_store->GetLineageGraph(req, &resp);
    EXPECT_TRUE(absl::IsNotFound(status));
  }

  {
    // invalid boundary artifacts.
    GetLineageGraphRequest req;
    req.mutable_options()->mutable_artifacts_options()->set_filter_query(
        "uri = 'uri://foo/a4'");
    req.mutable_options()->mutable_stop_conditions()->set_boundary_artifacts(
        "invalid boundary query");
    absl::Status status = metadata_store->GetLineageGraph(req, &resp);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  {
    // invalid boundary executions.
    GetLineageGraphRequest req;
    req.mutable_options()->mutable_artifacts_options()->set_filter_query(
        "uri = 'uri://foo/a4'");
    req.mutable_options()->mutable_stop_conditions()->set_boundary_executions(
        "invalid boundary query");
    absl::Status status = metadata_store->GetLineageGraph(req, &resp);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    MetadataStoreTest, MetadataStoreTestSuite, ::testing::Values([]() {
      return std::make_unique<RDBMSMetadataStoreContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata
