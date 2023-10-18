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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/metadata_store_test_suite.h"
#include "ml_metadata/metadata_store/sqlite_metadata_source.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/util/metadata_source_query_config.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {
namespace testing {
namespace {

constexpr int64_t kTestNumArtifactsInLargeLineageGraph = 102;
constexpr int64_t kTestNumExecutionsInLargeLineageGraph = 100;
constexpr int64_t kTestNumContextsInLargeLineageGraph = 2;

constexpr int64_t kTestNumArtifactsInLongLineageGraph = 4;
constexpr int64_t kTestNumExecutionsInLongLineageGraph = 3;
constexpr int64_t kTestNumContextsInLongLineageGraph = 3;

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



// Creates the following lineage graph.
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
                          const std::optional<Artifact::State>& state) {
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
                           absl::Span<const int64_t> input_ids,
                           absl::Span<const int64_t> output_ids) {
    PutExecutionRequest req;
    Execution* execution = req.mutable_execution();
    execution->set_type_id(put_types_resp.execution_type_ids(0));
    (*execution->mutable_properties())["p2"].set_string_value(label);
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
    execution->set_id(resp.execution_id());
    want_executions.push_back(*execution);
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

// Creates a large lineage graph, with kTestNumArtifactsInLargeLineageGraph
// artifacts and 2 * kTestNumExecutionsInLargeLineageGraph executions.
// Each path will look like:
// a_0 -> ... -> e_i -> a_i+1 -> e_i+kTestNumExecutionsInLargeLineageGraph
// -> ... -> a_kTestNumArtifactsInLargeLineageGraph
// TODO(b/283852485): Extract lineage graph creation util functions.
absl::Status CreateLargeLineageGraph(MetadataStore& metadata_store,
                                     std::vector<Artifact>& want_artifacts,
                                     std::vector<Execution>& want_executions,
                                     std::vector<Context>& want_contexts) {
  const PutTypesRequest put_types_req =
      ParseTextProtoOrDie<PutTypesRequest>(R"pb(
        artifact_types: {
          name: 'artifact_type'
          properties { key: 'p1' value: STRING }
        }
        execution_types: {
          name: 'execution_type'
          properties { key: 'p2' value: STRING }
        }
        context_types: {
          name: 'context_type'
          properties { key: 'p3' value: STRING }
        }
      )pb");
  PutTypesResponse put_types_resp;
  MLMD_RETURN_IF_ERROR(metadata_store.PutTypes(put_types_req, &put_types_resp));

  // Insert artifacts
  auto put_artifact = [&](absl::string_view label) {
    PutArtifactsRequest put_artifacts_req;
    Artifact* artifact = put_artifacts_req.add_artifacts();
    artifact->set_uri(absl::StrCat("uri://foo/", string(label)));
    artifact->set_type_id(put_types_resp.artifact_type_ids(0));
    (*artifact->mutable_properties())["p1"].set_string_value(string(label));
    PutArtifactsResponse resp;
    CHECK_EQ(metadata_store.PutArtifacts(put_artifacts_req, &resp),
             absl::OkStatus());
    artifact->set_id(resp.artifact_ids(0));
    want_artifacts.push_back(*artifact);
  };

  for (int i = 0; i < kTestNumArtifactsInLargeLineageGraph; i++) {
    put_artifact(/*label=*/absl::StrCat("a", i));
  }

  auto put_context = [&](absl::string_view label) {
    PutContextsRequest put_context_req;
    Context* context = put_context_req.add_contexts();
    context->set_type_id(put_types_resp.context_type_ids(0));
    context->set_name(string(label));
    (*context->mutable_properties())["p3"].set_string_value(string(label));
    PutContextsResponse resp;
    CHECK_EQ(metadata_store.PutContexts(put_context_req, &resp),
             absl::OkStatus());
    context->set_id(resp.context_ids(0));
    want_contexts.push_back(*context);
  };

  for (int i = 0; i < kTestNumContextsInLargeLineageGraph; i++) {
    put_context(/*label=*/absl::StrCat("c", i));
  }

  // Insert executions and links to artifacts
  auto put_execution = [&](absl::string_view label,
                           absl::Span<const int64_t> input_artifact_ids,
                           absl::Span<const int64_t> output_artifact_ids,
                           int64_t context_index) {
    PutExecutionRequest req;
    Execution* execution = req.mutable_execution();
    execution->set_type_id(put_types_resp.execution_type_ids(0));
    (*execution->mutable_properties())["p2"].set_string_value(string(label));
    for (int64_t id : input_artifact_ids) {
      Event* event = req.add_artifact_event_pairs()->mutable_event();
      event->set_artifact_id(id);
      event->set_type(Event::INPUT);
    }
    for (int64_t id : output_artifact_ids) {
      Event* event = req.add_artifact_event_pairs()->mutable_event();
      event->set_artifact_id(id);
      event->set_type(Event::OUTPUT);
    }
    req.mutable_contexts()->Add()->CopyFrom(want_contexts[context_index]);
    PutExecutionResponse resp;
    ASSERT_EQ(metadata_store.PutExecution(req, &resp), absl::OkStatus());
    execution->set_id(resp.execution_id());
    want_executions.push_back(*execution);
  };

  // Create executions, edges and contexts.
  for (int i = 0; i < kTestNumExecutionsInLargeLineageGraph; i++) {
    put_execution(/*label=*/absl::StrCat("e", i),
                  /*input_artifact_ids=*/{want_artifacts[0].id()},
                  /*output_artifact_ids=*/{want_artifacts[i + 1].id()},
                  /*context_index=*/0);
  }
  for (int i = 0; i < kTestNumExecutionsInLargeLineageGraph; i++) {
    put_execution(
        /*label=*/absl::StrCat("e", i),
        /*input_artifact_ids=*/{want_artifacts[i + 1].id()},
        /*output_artifact_ids=*/
        {want_artifacts[kTestNumArtifactsInLargeLineageGraph - 1].id()},
        /*context_index=*/1);
  }
  return absl::OkStatus();
}

// Create a lineage subgraph with the following setup:
// a0 (c0) -> e0 (c0) -> a1 (c0, c1) -> e1 (c1) -> a2 (c1, c2) -> e2 (c2) ->
// a3 (c2) -> ... -> a_kTestNumArtifactsInLongLineageGraph
// (c_kTestNumContextsInLongLineageGraph);
// TODO(b/283852485): Extract lineage graph creation util functions.
absl::Status CreateLongLineageGraph(MetadataStore& metadata_store,
                                    std::vector<Artifact>& want_artifacts,
                                    std::vector<Execution>& want_executions,
                                    std::vector<Context>& want_contexts) {
  const PutTypesRequest put_types_req =
      ParseTextProtoOrDie<PutTypesRequest>(R"pb(
        artifact_types: {
          name: 't1'
          properties { key: 'p1' value: STRING }
        }
        execution_types: {
          name: 't2'
          properties { key: 'p2' value: STRING }
        }
        context_types: {
          name: 't3'
          properties { key: 'p3' value: STRING }
        }
      )pb");
  PutTypesResponse put_types_resp;
  MLMD_RETURN_IF_ERROR(metadata_store.PutTypes(put_types_req, &put_types_resp));

  // insert artifacts
  auto put_artifact = [&](absl::string_view label) {
    PutArtifactsRequest put_artifacts_req;
    Artifact* artifact = put_artifacts_req.add_artifacts();
    artifact->set_uri(absl::StrCat("uri://foo/", string(label)));
    artifact->set_type_id(put_types_resp.artifact_type_ids(0));
    (*artifact->mutable_properties())["p1"].set_string_value(string(label));
    PutArtifactsResponse resp;
    CHECK_EQ(metadata_store.PutArtifacts(put_artifacts_req, &resp),
             absl::OkStatus());
    artifact->set_id(resp.artifact_ids(0));
    want_artifacts.push_back(*artifact);
  };

  for (int i = 0; i < kTestNumArtifactsInLongLineageGraph; i++) {
    put_artifact(/*label=*/absl::StrCat("a", i));
  }

  auto put_context = [&](absl::string_view label) {
    PutContextsRequest put_context_req;
    Context* context = put_context_req.add_contexts();
    context->set_type_id(put_types_resp.context_type_ids(0));
    context->set_name(string(label));
    (*context->mutable_properties())["p3"].set_string_value(string(label));
    PutContextsResponse resp;
    CHECK_EQ(metadata_store.PutContexts(put_context_req, &resp),
             absl::OkStatus());
    context->set_id(resp.context_ids(0));
    want_contexts.push_back(*context);
  };

  for (int i = 0; i < kTestNumContextsInLongLineageGraph; i++) {
    put_context(/*label=*/absl::StrCat("c", i));
  }

  // Insert executions and links to artifacts
  auto put_execution = [&](absl::string_view label,
                           absl::Span<const int64_t> input_artifact_ids,
                           absl::Span<const int64_t> output_artifact_ids,
                           int64_t context_index) {
    PutExecutionRequest req;
    Execution* execution = req.mutable_execution();
    execution->set_type_id(put_types_resp.execution_type_ids(0));
    (*execution->mutable_properties())["p2"].set_string_value(string(label));
    // database id starts from 1.
    for (int64_t id : input_artifact_ids) {
      Event* event = req.add_artifact_event_pairs()->mutable_event();
      event->set_artifact_id(id);
      event->set_type(Event::INPUT);
    }
    for (int64_t id : output_artifact_ids) {
      Event* event = req.add_artifact_event_pairs()->mutable_event();
      event->set_artifact_id(id);
      event->set_type(Event::OUTPUT);
    }
    req.mutable_contexts()->Add()->CopyFrom(want_contexts[context_index]);
    PutExecutionResponse resp;
    ASSERT_EQ(metadata_store.PutExecution(req, &resp), absl::OkStatus());
    execution->set_id(resp.execution_id());
    want_executions.push_back(*execution);
  };

  // Create executions, edges and contexts.
  for (int i = 0; i < kTestNumExecutionsInLongLineageGraph; i++) {
    put_execution(/*label=*/absl::StrCat("e", i),
                  /*input_artifact_ids=*/{want_artifacts[i].id()},
                  /*output_artifact_ids=*/{want_artifacts[i + 1].id()},
                  /*context_index=*/i);
  }
  return absl::OkStatus();
}


// The utilities for testing the GetLineageSubgraph.
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

void VerifySubgraphSkeleton(const LineageGraph& skeleton,
                            absl::Span<const int64_t> expected_artifact_ids,
                            absl::Span<const int64_t> expected_execution_ids,
                            absl::Span<const std::pair<int64_t, int64_t>>
                                expected_node_id_pairs_in_events) {
  EXPECT_THAT(skeleton.artifacts(),
              UnorderedPointwise(IdEquals(), expected_artifact_ids));
  EXPECT_THAT(skeleton.executions(),
              UnorderedPointwise(IdEquals(), expected_execution_ids));
  EXPECT_THAT(skeleton.events(),
              SizeIs(expected_node_id_pairs_in_events.size()));
  std::vector<std::pair<int64_t, int64_t>> got_node_id_pairs_in_events;
  for (const Event& event : skeleton.events()) {
    got_node_id_pairs_in_events.push_back(
        {event.artifact_id(), event.execution_id()});
  }

  EXPECT_THAT(got_node_id_pairs_in_events,
              UnorderedElementsAreArray(expected_node_id_pairs_in_events));
}

TEST(MetadataStoreExtendedTest, GetLineageSubgraphFromArtifactsWithMaxHops) {
  // Prepare a store with the lineage graph
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  int64_t min_creation_time;
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  ASSERT_EQ(CreateLineageGraph(*metadata_store, min_creation_time,
                               want_artifacts, want_executions),
            absl::OkStatus());

  // Verify the query results with the specified max_num_hop
  auto verify_lineage_graph_with_max_num_hop =
      [&](std::optional<int64_t> max_num_hop,
          absl::Span<const int64_t> expected_artifact_ids,
          absl::Span<const int64_t> expected_execution_ids,
          absl::Span<const std::pair<int64_t, int64_t>>
              expected_node_index_pairs_in_events) {
        GetLineageSubgraphRequest req;
        GetLineageSubgraphResponse resp;
        req.mutable_lineage_subgraph_query_options()
            ->mutable_starting_artifacts()
            ->set_filter_query("uri = 'uri://foo/a4'");
        if (max_num_hop) {
          LOG(INFO) << "Test when max_num_hops = " << *max_num_hop;
          req.mutable_lineage_subgraph_query_options()->set_max_num_hops(
              *max_num_hop);
        } else {
          LOG(INFO) << "Test when max_num_hops is unset.";
        }
        EXPECT_EQ(metadata_store->GetLineageSubgraph(req, &resp),
                  absl::OkStatus());
        std::vector<std::pair<int64_t, int64_t>>
            expected_node_id_pairs_in_events;
        for (const auto& [artifact_index, execution_index] :
             expected_node_index_pairs_in_events) {
          expected_node_id_pairs_in_events.push_back(
              {want_artifacts.at(artifact_index).id(),
               want_executions.at(execution_index).id()});
        }
        VerifySubgraphSkeleton(resp.lineage_subgraph(), expected_artifact_ids,
                               expected_execution_ids,
                               expected_node_id_pairs_in_events);
      };

  // Verify the lineage graph query results by increasing the max_num_hops.
  verify_lineage_graph_with_max_num_hop(
      /*max_num_hop=*/0,
      /*want_artifacts=*/{want_artifacts[4].id()},
      /*want_executions=*/{},
      /*want_events=*/{});

  verify_lineage_graph_with_max_num_hop(
      /*max_num_hop=*/1,
      /*want_artifacts=*/{want_artifacts[4].id()},
      /*want_executions=*/{want_executions[2].id(), want_executions[3].id()},
      /*want_events=*/{{4, 2}, {4, 3}});

  verify_lineage_graph_with_max_num_hop(
      /*max_num_hop=*/2,
      /*want_artifacts=*/
      {want_artifacts[2].id(), want_artifacts[3].id(), want_artifacts[4].id(),
       want_artifacts[5].id()},
      /*want_executions=*/{want_executions[2].id(), want_executions[3].id()},
      /*want_events=*/{{4, 2}, {4, 3}, {2, 2}, {3, 3}, {5, 3}});

  verify_lineage_graph_with_max_num_hop(
      /*max_num_hop=*/3,
      /*want_artifacts=*/
      {want_artifacts[2].id(), want_artifacts[3].id(), want_artifacts[4].id(),
       want_artifacts[5].id()},
      /*want_executions=*/
      {want_executions[1].id(), want_executions[2].id(),
       want_executions[3].id()},
      /*want_events=*/{{4, 2}, {4, 3}, {2, 2}, {3, 3}, {5, 3}, {3, 1}});

  verify_lineage_graph_with_max_num_hop(
      /*max_num_hop=*/4,
      /*want_artifacts=*/
      {want_artifacts[1].id(), want_artifacts[2].id(), want_artifacts[3].id(),
       want_artifacts[4].id(), want_artifacts[5].id()},
      /*want_executions=*/
      {want_executions[1].id(), want_executions[2].id(),
       want_executions[3].id()},
      /*want_events=*/{{4, 2}, {4, 3}, {2, 2}, {3, 3}, {5, 3}, {3, 1}, {1, 1}});

  verify_lineage_graph_with_max_num_hop(
      /*max_num_hop=*/absl::nullopt,
      /*want_artifacts=*/
      {want_artifacts[1].id(), want_artifacts[2].id(), want_artifacts[3].id(),
       want_artifacts[4].id(), want_artifacts[5].id()},
      /*want_executions=*/
      {want_executions[1].id(), want_executions[2].id(),
       want_executions[3].id()},
      /*want_events=*/{{4, 2}, {4, 3}, {2, 2}, {3, 3}, {5, 3}, {3, 1}, {1, 1}});
}

TEST(MetadataStoreExtendedTest, GetLineageSubgraphFromExecutionsWithMaxHops) {
  // Prepare a store with the lineage graph
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  int64_t min_creation_time;
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  // Creates the following lineage graph for deleting lineage.
  //  a_0 a_1     a_3
  //   |    \   /     \
  //  e_0    e_1        e_3
  //                 /       \
  //     a_2     a_4(LIVE)    a_5
  //        \    /
  //         e_2

  ASSERT_EQ(CreateLineageGraph(*metadata_store, min_creation_time,
                               want_artifacts, want_executions),
            absl::OkStatus());

  // Verify the query results with the specified max_num_hop.
  // Starting from execution_0, execution_1 and execution_2.
  auto verify_lineage_graph_with_max_num_hop =
      [&](std::optional<int64_t> max_num_hop,
          absl::Span<const int64_t> expected_artifact_ids,
          absl::Span<const int64_t> expected_execution_ids,
          absl::Span<const std::pair<int64_t, int64_t>>
              expected_node_index_pairs_in_events) {
        GetLineageSubgraphRequest req;
        GetLineageSubgraphResponse resp;
        req.mutable_lineage_subgraph_query_options()
            ->mutable_starting_executions()
            ->set_filter_query(
                "properties.p2.string_value = 'e0' OR "
                "properties.p2.string_value = 'e1' OR "
                "properties.p2.string_value = 'e2'");
        if (max_num_hop) {
          LOG(INFO) << "Test when max_num_hops = " << *max_num_hop;
          req.mutable_lineage_subgraph_query_options()->set_max_num_hops(
              *max_num_hop);
        } else {
          LOG(INFO) << "Test when max_num_hops is unset.";
        }
        EXPECT_EQ(metadata_store->GetLineageSubgraph(req, &resp),
                  absl::OkStatus());
        std::vector<std::pair<int64_t, int64_t>>
            expected_node_id_pairs_in_events;
        for (const auto& [artifact_index, execution_index] :
             expected_node_index_pairs_in_events) {
          expected_node_id_pairs_in_events.push_back(
              {want_artifacts.at(artifact_index).id(),
               want_executions.at(execution_index).id()});
        }
        VerifySubgraphSkeleton(resp.lineage_subgraph(), expected_artifact_ids,
                               expected_execution_ids,
                               expected_node_id_pairs_in_events);
      };

  // Verify the lineage graph query results by increasing the max_num_hops.
  verify_lineage_graph_with_max_num_hop(
      /*max_num_hop=*/0,
      /*want_artifacts=*/{},
      /*want_executions=*/
      {want_executions[0].id(), want_executions[1].id(),
       want_executions[2].id()},
      /*want_events=*/{});

  verify_lineage_graph_with_max_num_hop(
      /*max_num_hop=*/1,
      /*want_artifacts=*/
      {want_artifacts[0].id(), want_artifacts[1].id(), want_artifacts[2].id(),
       want_artifacts[3].id(), want_artifacts[4].id()},
      /*want_executions=*/
      {want_executions[0].id(), want_executions[1].id(),
       want_executions[2].id()},
      /*want_events=*/{{0, 0}, {1, 1}, {3, 1}, {2, 2}, {4, 2}});

  verify_lineage_graph_with_max_num_hop(
      /*max_num_hop=*/2,
      /*want_artifacts=*/
      {want_artifacts[0].id(), want_artifacts[1].id(), want_artifacts[2].id(),
       want_artifacts[3].id(), want_artifacts[4].id()},
      /*want_executions=*/
      {want_executions[0].id(), want_executions[1].id(),
       want_executions[2].id(), want_executions[3].id()},
      /*want_events=*/{{0, 0}, {1, 1}, {3, 1}, {2, 2}, {4, 2}, {3, 3}, {4, 3}});

  verify_lineage_graph_with_max_num_hop(
      /*max_num_hop=*/3,
      /*want_artifacts=*/
      {want_artifacts[0].id(), want_artifacts[1].id(), want_artifacts[2].id(),
       want_artifacts[3].id(), want_artifacts[4].id(), want_artifacts[5].id()},
      /*want_executions=*/
      {want_executions[0].id(), want_executions[1].id(),
       want_executions[2].id(), want_executions[3].id()},

      /*want_events=*/
      {{0, 0}, {1, 1}, {3, 1}, {2, 2}, {4, 2}, {3, 3}, {4, 3}, {5, 3}});

  verify_lineage_graph_with_max_num_hop(
      /*max_num_hop=*/absl::nullopt,
      /*want_artifacts=*/
      {want_artifacts[0].id(), want_artifacts[1].id(), want_artifacts[2].id(),
       want_artifacts[3].id(), want_artifacts[4].id(), want_artifacts[5].id()},
      /*want_executions=*/
      {want_executions[0].id(), want_executions[1].id(),
       want_executions[2].id(), want_executions[3].id()},

      /*want_events=*/
      {{0, 0}, {1, 1}, {3, 1}, {2, 2}, {4, 2}, {3, 3}, {4, 3}, {5, 3}});
}

TEST(MetadataStoreExtendedTest, GetLineageSubgraphFromArtifactsWithDirection) {
  // Prepare a store with the lineage graph
  // a0 -> e0
  // a1 -> e1 -> a3 --> e3 -> a5
  //                 /
  // a2 -> e2 -> a4 -
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  int64_t min_creation_time;
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  ASSERT_EQ(CreateLineageGraph(*metadata_store, min_creation_time,
                               want_artifacts, want_executions),
            absl::OkStatus());

  // Verify the query results with the specified direction
  auto verify_lineage_graph_with_direction =
      [&](const LineageSubgraphQueryOptions::Direction direction,
          const int64_t max_num_hop,
          absl::Span<const int64_t> expected_artifact_ids,
          absl::Span<const int64_t> expected_execution_ids,
          absl::Span<const std::pair<int64_t, int64_t>>
              expected_node_index_pairs_in_events) {
        GetLineageSubgraphRequest req;
        GetLineageSubgraphResponse resp;
        req.mutable_lineage_subgraph_query_options()
            ->mutable_starting_artifacts()
            ->set_filter_query("uri = 'uri://foo/a4'");
        req.mutable_lineage_subgraph_query_options()->set_max_num_hops(
            max_num_hop);
        req.mutable_lineage_subgraph_query_options()->set_direction(direction);
        EXPECT_EQ(metadata_store->GetLineageSubgraph(req, &resp),
                  absl::OkStatus());
        std::vector<std::pair<int64_t, int64_t>>
            expected_node_id_pairs_in_events;
        for (const auto& [artifact_index, execution_index] :
             expected_node_index_pairs_in_events) {
          expected_node_id_pairs_in_events.push_back(
              {want_artifacts.at(artifact_index).id(),
               want_executions.at(execution_index).id()});
        }
        VerifySubgraphSkeleton(resp.lineage_subgraph(), expected_artifact_ids,
                               expected_execution_ids,
                               expected_node_id_pairs_in_events);
      };

  // Verify the lineage graph query results by increasing the max_num_hops.
  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::DOWNSTREAM,
      /*max_num_hop=*/1,
      /*want_artifacts=*/{want_artifacts[4].id()},
      /*want_executions=*/{want_executions[3].id()},
      /*want_events=*/{{4, 3}});

  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::UPSTREAM,
      /*max_num_hop=*/1,
      /*want_artifacts=*/{want_artifacts[4].id()},
      /*want_executions=*/{want_executions[2].id()},
      /*want_events=*/{{4, 2}});

  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::DOWNSTREAM,
      /*max_num_hop=*/2,
      /*want_artifacts=*/
      {want_artifacts[4].id(), want_artifacts[5].id()},
      /*want_executions=*/{want_executions[3].id()},
      /*want_events=*/{{4, 3}, {5, 3}});

  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::UPSTREAM,
      /*max_num_hop=*/2,
      /*want_artifacts=*/
      {want_artifacts[2].id(), want_artifacts[4].id()},
      /*want_executions=*/{want_executions[2].id()},
      /*want_events=*/{{4, 2}, {2, 2}});

  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::DOWNSTREAM,
      /*max_num_hop=*/100,
      /*want_artifacts=*/
      {want_artifacts[4].id(), want_artifacts[5].id()},
      /*want_executions=*/{want_executions[3].id()},
      /*want_events=*/{{4, 3}, {5, 3}});

  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::UPSTREAM,
      /*max_num_hop=*/100,
      /*want_artifacts=*/
      {want_artifacts[2].id(), want_artifacts[4].id()},
      /*want_executions=*/{want_executions[2].id()},
      /*want_events=*/{{4, 2}, {2, 2}});
}

TEST(MetadataStoreExtendedTest, GetLineageSubgraphFromExecutionsWithDirection) {
  // Prepare a store with the lineage graph
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  int64_t min_creation_time;
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  // Creates the following lineage graph for deleting lineage.
  // a0 -> e0
  // a1 -> e1 -> a3 --> e3 -> a5
  //                 /
  // a2 -> e2 -> a4 -

  ASSERT_EQ(CreateLineageGraph(*metadata_store, min_creation_time,
                               want_artifacts, want_executions),
            absl::OkStatus());

  // Verify the query results with the specified max_num_hop.
  // Starting from execution_0, execution_1 and execution_2.
  auto verify_lineage_graph_with_direction =
      [&](LineageSubgraphQueryOptions::Direction direction, int64_t max_num_hop,
          absl::Span<const int64_t> expected_artifact_ids,
          absl::Span<const int64_t> expected_execution_ids,
          absl::Span<const std::pair<int64_t, int64_t>>
              expected_node_index_pairs_in_events) {
        GetLineageSubgraphRequest req;
        GetLineageSubgraphResponse resp;
        req.mutable_lineage_subgraph_query_options()
            ->mutable_starting_executions()
            ->set_filter_query(
                "properties.p2.string_value = 'e0' OR "
                "properties.p2.string_value = 'e3'");
        req.mutable_lineage_subgraph_query_options()->set_direction(direction);
        req.mutable_lineage_subgraph_query_options()->set_max_num_hops(
            max_num_hop);
        EXPECT_EQ(metadata_store->GetLineageSubgraph(req, &resp),
                  absl::OkStatus());
        std::vector<std::pair<int64_t, int64_t>>
            expected_node_id_pairs_in_events;
        for (const auto& [artifact_index, execution_index] :
             expected_node_index_pairs_in_events) {
          expected_node_id_pairs_in_events.push_back(
              {want_artifacts.at(artifact_index).id(),
               want_executions.at(execution_index).id()});
        }
        VerifySubgraphSkeleton(resp.lineage_subgraph(), expected_artifact_ids,
                               expected_execution_ids,
                               expected_node_id_pairs_in_events);
      };

  // Verify the lineage graph query results by increasing the max_num_hops.
  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::DOWNSTREAM, /*max_num_hop=*/1,
      /*want_artifacts=*/{want_artifacts[5].id()},
      /*want_executions=*/
      {want_executions[0].id(), want_executions[3].id()},
      /*want_events=*/{{5, 3}});

  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::UPSTREAM, /*max_num_hop=*/1,
      /*want_artifacts=*/
      {want_artifacts[0].id(), want_artifacts[3].id(), want_artifacts[4].id()},
      /*want_executions=*/
      {want_executions[0].id(), want_executions[3].id()},
      /*want_events=*/{{0, 0}, {3, 3}, {4, 3}});

  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::DOWNSTREAM, /*max_num_hop=*/2,
      /*want_artifacts=*/{want_artifacts[5].id()},
      /*want_executions=*/
      {want_executions[0].id(), want_executions[3].id()},
      /*want_events=*/{{5, 3}});

  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::UPSTREAM, /*max_num_hop=*/2,
      /*want_artifacts=*/
      {want_artifacts[0].id(), want_artifacts[3].id(), want_artifacts[4].id()},
      /*want_executions=*/
      {want_executions[0].id(), want_executions[1].id(),
       want_executions[2].id(), want_executions[3].id()},
      /*want_events=*/{{0, 0}, {3, 3}, {4, 3}, {3, 1}, {4, 2}});

  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::DOWNSTREAM, /*max_num_hop=*/100,
      /*want_artifacts=*/{want_artifacts[5].id()},
      /*want_executions=*/
      {want_executions[0].id(), want_executions[3].id()},
      /*want_events=*/{{5, 3}});

  verify_lineage_graph_with_direction(
      LineageSubgraphQueryOptions::UPSTREAM, /*max_num_hop=*/100,
      /*want_artifacts=*/
      {want_artifacts[0].id(), want_artifacts[1].id(), want_artifacts[2].id(),
       want_artifacts[3].id(), want_artifacts[4].id()},
      /*want_executions=*/
      {want_executions[0].id(), want_executions[1].id(),
       want_executions[2].id(), want_executions[3].id()},
      /*want_events=*/{{0, 0}, {3, 3}, {4, 3}, {3, 1}, {4, 2}, {1, 1}, {2, 2}});
}

TEST(MetadataStoreExtendedTest, GetLineageSubgraphFromWithEndingNodes) {
  // Prepare a store with the lineage graph
  // a0 -> e0
  // a1 -> e1 -> a3 --> e3 -> a5
  //                 /
  // a2 -> e2 -> a4 -
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  int64_t min_creation_time;
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  ASSERT_EQ(CreateLineageGraph(*metadata_store, min_creation_time,
                               want_artifacts, want_executions),
            absl::OkStatus());

  // Verify the query results with ending nodes filtering
  auto verify_lineage_graph_with_ending_nodes =
      [&](const LineageSubgraphQueryOptions::Direction direction,
          absl::string_view ending_artifact_filter_query,
          absl::string_view ending_execution_filter_query,
          const bool include_ending_nodes, const int64_t max_num_hop,
          absl::Span<const int64_t> expected_artifact_ids,
          absl::Span<const int64_t> expected_execution_ids,
          absl::Span<const std::pair<int64_t, int64_t>>
              expected_node_index_pairs_in_events) {
        GetLineageSubgraphRequest req;
        GetLineageSubgraphResponse resp;
        req.mutable_lineage_subgraph_query_options()
            ->mutable_starting_artifacts()
            ->set_filter_query("uri = 'uri://foo/a4'");
        req.mutable_lineage_subgraph_query_options()->set_max_num_hops(
            max_num_hop);
        req.mutable_lineage_subgraph_query_options()->set_direction(direction);
        req.mutable_lineage_subgraph_query_options()
            ->mutable_ending_artifacts()
            ->set_filter_query(std::string(ending_artifact_filter_query));
        req.mutable_lineage_subgraph_query_options()
            ->mutable_ending_executions()
            ->set_filter_query(std::string(ending_execution_filter_query));
        req.mutable_lineage_subgraph_query_options()
            ->mutable_ending_artifacts()
            ->set_include_ending_nodes(include_ending_nodes);
        req.mutable_lineage_subgraph_query_options()
            ->mutable_ending_executions()
            ->set_include_ending_nodes(include_ending_nodes);
        EXPECT_EQ(metadata_store->GetLineageSubgraph(req, &resp),
                  absl::OkStatus());
        std::vector<std::pair<int64_t, int64_t>>
            expected_node_id_pairs_in_events;
        for (const auto& [artifact_index, execution_index] :
             expected_node_index_pairs_in_events) {
          expected_node_id_pairs_in_events.push_back(
              {want_artifacts.at(artifact_index).id(),
               want_executions.at(execution_index).id()});
        }
        VerifySubgraphSkeleton(resp.lineage_subgraph(), expected_artifact_ids,
                               expected_execution_ids,
                               expected_node_id_pairs_in_events);
      };

  verify_lineage_graph_with_ending_nodes(
      LineageSubgraphQueryOptions::DOWNSTREAM,
      /*ending_artifact_filter_query=*/"uri = 'uri://foo/a4'",
      /*ending_execution_filter_query=*/"",
      /*include_ending_nodes=*/true,
      /*max_num_hop=*/1,
      /*want_artifacts=*/{want_artifacts[4].id()},
      /*want_executions=*/{},
      /*want_events=*/{});

  verify_lineage_graph_with_ending_nodes(
      LineageSubgraphQueryOptions::DOWNSTREAM,
      /*ending_artifact_filter_query=*/"uri = 'uri://foo/a4'",
      /*ending_execution_filter_query=*/"",
      /*include_ending_nodes=*/false,
      /*max_num_hop=*/1,
      /*want_artifacts=*/{},
      /*want_executions=*/{},
      /*want_events=*/{});

  verify_lineage_graph_with_ending_nodes(
      LineageSubgraphQueryOptions::DOWNSTREAM,
      /*ending_artifact_filter_query=*/"uri = 'uri://foo/a5'",
      /*ending_execution_filter_query=*/
      absl::Substitute("id = $0", want_executions[3].id()),
      /*include_ending_nodes=*/true,
      /*max_num_hop=*/1,
      /*want_artifacts=*/{want_artifacts[4].id()},
      /*want_executions=*/{want_executions[3].id()},
      /*want_events=*/{{4, 3}});

  verify_lineage_graph_with_ending_nodes(
      LineageSubgraphQueryOptions::DOWNSTREAM,
      /*ending_artifact_filter_query=*/"uri = 'uri://foo/a5'",
      /*ending_execution_filter_query=*/
      absl::Substitute("id = $0", want_executions[3].id()),
      /*include_ending_nodes=*/false,
      /*max_num_hop=*/1,
      /*want_artifacts=*/{want_artifacts[4].id()},
      /*want_executions=*/{},
      /*want_events=*/{});

  verify_lineage_graph_with_ending_nodes(
      LineageSubgraphQueryOptions::BIDIRECTIONAL,
      /*ending_artifact_filter_query=*/"uri = 'uri://foo/a5'",
      /*ending_execution_filter_query=*/
      absl::Substitute("id = $0 OR id = $1", want_executions[1].id(),
                       want_executions[2].id()),
      /*include_ending_nodes=*/true,
      /*max_num_hop=*/20,
      /*want_artifacts=*/
      {want_artifacts[3].id(), want_artifacts[4].id(), want_artifacts[5].id()},
      /*want_executions=*/
      {want_executions[1].id(), want_executions[2].id(),
       want_executions[3].id()},
      /*want_events=*/{{4, 3}, {3, 3}, {3, 1}, {4, 2}, {5, 3}});

  verify_lineage_graph_with_ending_nodes(
      LineageSubgraphQueryOptions::BIDIRECTIONAL,
      /*ending_artifact_filter_query=*/"uri = 'uri://foo/a5'",
      /*ending_execution_filter_query=*/
      absl::Substitute("id = $0 OR id = $1", want_executions[1].id(),
                       want_executions[2].id()),
      /*include_ending_nodes=*/false,
      /*max_num_hop=*/20,
      /*want_artifacts=*/{want_artifacts[3].id(), want_artifacts[4].id()},
      /*want_executions=*/{want_executions[3].id()},
      /*want_events=*/{{4, 3}, {3, 3}});
}

TEST(MetadataStoreExtendedTest, GetLineageSubgraphWithContexts) {
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  std::vector<Context> want_contexts;

  ASSERT_EQ(CreateLongLineageGraph(*metadata_store, want_artifacts,
                                   want_executions, want_contexts),
            absl::OkStatus());

  auto verify_lineage_subgraph_with_contexts =
      [&](LineageSubgraphQueryOptions& options,
          absl::Span<const int64_t> expected_artifact_ids,
          absl::Span<const int64_t> expected_execution_ids,
          absl::Span<const int64_t> expected_context_ids,
          absl::Span<const std::pair<int64_t, int64_t>>
              expected_node_index_pairs_in_events,
          absl::Span<const std::pair<int64_t, int64_t>>
              expected_node_index_pairs_in_attributions,
          absl::Span<const std::pair<int64_t, int64_t>>
              expected_node_index_pairs_in_associations) {
        GetLineageSubgraphRequest req;
        GetLineageSubgraphResponse resp;
        req.mutable_lineage_subgraph_query_options()->Swap(&options);
        EXPECT_EQ(metadata_store->GetLineageSubgraph(req, &resp),
                  absl::OkStatus());
        std::vector<std::pair<int64_t, int64_t>>
            expected_node_id_pairs_in_events;
        for (const auto& [artifact_index, execution_index] :
             expected_node_index_pairs_in_events) {
          expected_node_id_pairs_in_events.push_back(
              {want_artifacts.at(artifact_index).id(),
               want_executions.at(execution_index).id()});
        }
        VerifySubgraphSkeleton(resp.lineage_subgraph(), expected_artifact_ids,
                               expected_execution_ids,
                               expected_node_id_pairs_in_events);
        EXPECT_THAT(resp.lineage_subgraph().contexts(),
                    UnorderedPointwise(IdEquals(), expected_context_ids));

        std::vector<Attribution> expected_attributions;
        for (const auto& [artifact_index, context_index] :
             expected_node_index_pairs_in_attributions) {
          Attribution attribution;
          attribution.set_artifact_id(want_artifacts.at(artifact_index).id());
          attribution.set_context_id(want_contexts.at(context_index).id());
          expected_attributions.push_back(attribution);
        }
        EXPECT_THAT(resp.lineage_subgraph().attributions(),
                    UnorderedPointwise(EqualsProto<Attribution>(),
                                       expected_attributions));

        std::vector<Association> expected_associations;
        for (const auto& [execution_index, context_index] :
             expected_node_index_pairs_in_associations) {
          Association association;
          association.set_execution_id(
              want_executions.at(execution_index).id());
          association.set_context_id(want_contexts.at(context_index).id());
          expected_associations.push_back(association);
        }
        EXPECT_THAT(resp.lineage_subgraph().associations(),
                    UnorderedPointwise(EqualsProto<Association>(),
                                       expected_associations));
      };
  LineageSubgraphQueryOptions base_options;
  base_options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
  base_options.mutable_starting_artifacts()->set_filter_query(
      absl::Substitute(" contexts_0.name = 'c$0' ", 0));
  base_options.set_max_num_hops(0);
  {
    // Start from artifacts in context_0 and trace towards downstream in 0 hops.
    LineageSubgraphQueryOptions options = base_options;

    verify_lineage_subgraph_with_contexts(
        options, {want_artifacts[0].id(), want_artifacts[1].id()}, {},
        {want_contexts[0].id(), want_contexts[1].id()}, {},
        {{0, 0}, {1, 0}, {1, 1}}, {});
  }
  {
    // Start from context_0 and trace towards downstream in 1 hop.
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(1);

    verify_lineage_subgraph_with_contexts(
        options, {want_artifacts[0].id(), want_artifacts[1].id()},
        {want_executions[0].id(), want_executions[1].id()},
        {want_contexts[0].id(), want_contexts[1].id()}, {{0, 0}, {1, 1}},
        {{0, 0}, {1, 0}, {1, 1}}, {{0, 0}, {1, 1}});
  }
  {
    // Start from artifacts in context_0 and trace towards downstream in 2 hops
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(2);

    verify_lineage_subgraph_with_contexts(
        options,
        {want_artifacts[0].id(), want_artifacts[1].id(),
         want_artifacts[2].id()},
        {want_executions[0].id(), want_executions[1].id()},
        {want_contexts[0].id(), want_contexts[1].id(), want_contexts[2].id()},
        {{0, 0}, {1, 0}, {1, 1}, {2, 1}},
        {{0, 0}, {1, 0}, {1, 1}, {2, 1}, {2, 2}}, {{0, 0}, {1, 1}});
  }
  {
    // Start from artifacts in context_0 and trace towards downstream in 20 hops
    // with ending nodes as nodes in c2. Include ending nodes.
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(20);
    options.mutable_ending_artifacts()->set_filter_query(
        absl::Substitute(" contexts_0.name = 'c$0' ", 2));
    options.mutable_ending_artifacts()->set_include_ending_nodes(true);
    options.mutable_ending_executions()->set_filter_query(
        absl::Substitute(" contexts_0.name = 'c$0' ", 2));
    options.mutable_ending_executions()->set_include_ending_nodes(true);

    verify_lineage_subgraph_with_contexts(
        options,
        {want_artifacts[0].id(), want_artifacts[1].id(),
         want_artifacts[2].id()},
        {want_executions[0].id(), want_executions[1].id()},
        {want_contexts[0].id(), want_contexts[1].id(), want_contexts[2].id()},
        {{0, 0}, {1, 0}, {1, 1}, {2, 1}},
        {{0, 0}, {1, 0}, {1, 1}, {2, 1}, {2, 2}}, {{0, 0}, {1, 1}});
  }
  {
    // Start from artifacts in context_0 and trace towards downstream in 20 hops
    // with ending nodes as nodes in c2. Don't include ending nodes.
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(20);
    options.mutable_ending_artifacts()->set_filter_query(
        absl::Substitute(" contexts_0.name = 'c$0' ", 2));
    options.mutable_ending_executions()->set_filter_query(
        absl::Substitute(" contexts_0.name = 'c$0' ", 2));

    verify_lineage_subgraph_with_contexts(
        options, {want_artifacts[0].id(), want_artifacts[1].id()},
        {want_executions[0].id(), want_executions[1].id()},
        {want_contexts[0].id(), want_contexts[1].id()},
        {{0, 0}, {1, 0}, {1, 1}}, {{0, 0}, {1, 0}, {1, 1}}, {{0, 0}, {1, 1}});
  }
  {
    // Start from artifacts in context_0 OR context_2 and trace towards
    // downstream in 0 hops.
    LineageSubgraphQueryOptions options = base_options;
    options.mutable_starting_artifacts()->set_filter_query(absl::Substitute(
        " contexts_0.name = 'c$0' OR contexts_1.name = 'c$1' ", 0, 2));

    verify_lineage_subgraph_with_contexts(
        options,
        {want_artifacts[0].id(), want_artifacts[1].id(), want_artifacts[2].id(),
         want_artifacts[3].id()},
        {},
        {want_contexts[0].id(), want_contexts[1].id(), want_contexts[2].id()},
        {}, {{0, 0}, {1, 0}, {1, 1}, {2, 1}, {2, 2}, {3, 2}}, {});
  }
  {
    // Start from artifacts in context_0 OR context_2 and trace towards
    // downstream in 1 hop.
    LineageSubgraphQueryOptions options = base_options;
    options.mutable_starting_artifacts()->set_filter_query(absl::Substitute(
        " contexts_0.name = 'c$0' OR contexts_1.name = 'c$1'", 0, 2));
    options.set_max_num_hops(1);
    verify_lineage_subgraph_with_contexts(
        options,
        {want_artifacts[0].id(), want_artifacts[1].id(), want_artifacts[2].id(),
         want_artifacts[3].id()},
        {want_executions[0].id(), want_executions[1].id(),
         want_executions[2].id()},
        {want_contexts[0].id(), want_contexts[1].id(), want_contexts[2].id()},
        {{0, 0}, {1, 1}, {2, 2}},
        {{0, 0}, {1, 0}, {1, 1}, {2, 1}, {2, 2}, {3, 2}},
        {{0, 0}, {1, 1}, {2, 2}});
  }
  {
    // Start from artifacts in context_0 OR context_2 and trace towards
    // downstream in 20 hops.
    LineageSubgraphQueryOptions options = base_options;
    options.mutable_starting_artifacts()->set_filter_query(absl::Substitute(
        " contexts_0.name = 'c$0' OR contexts_1.name = 'c$1'", 0, 2));
    options.set_max_num_hops(20);
    verify_lineage_subgraph_with_contexts(
        options,
        {want_artifacts[0].id(), want_artifacts[1].id(), want_artifacts[2].id(),
         want_artifacts[3].id()},
        {want_executions[0].id(), want_executions[1].id(),
         want_executions[2].id()},
        {want_contexts[0].id(), want_contexts[1].id(), want_contexts[2].id()},
        {{0, 0}, {1, 0}, {1, 1}, {2, 1}, {2, 2}, {3, 2}},
        {{0, 0}, {1, 0}, {1, 1}, {2, 1}, {2, 2}, {3, 2}},
        {{0, 0}, {1, 1}, {2, 2}});
  }
  {
    // Start from artifacts in context_0 AND context_1 and trace towards
    // downstream in 0 hops.
    LineageSubgraphQueryOptions options = base_options;
    options.mutable_starting_artifacts()->set_filter_query(absl::Substitute(
        " contexts_0.name = 'c$0' AND contexts_1.name = 'c$1' ", 0, 1));

    verify_lineage_subgraph_with_contexts(
        options, {want_artifacts[1].id()}, {},
        {want_contexts[0].id(), want_contexts[1].id()}, {}, {{1, 0}, {1, 1}},
        {});
  }
}

TEST(MetadataStoreExtendedTest, GetLineageSubgraphOnLargeGraphWithDirection) {
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  std::vector<Context> want_contexts;
  ASSERT_EQ(CreateLargeLineageGraph(*metadata_store, want_artifacts,
                                    want_executions, want_contexts),
            absl::OkStatus());

  auto verify_lineage_subgraph_with_direction =
      [&](LineageSubgraphQueryOptions& options,
          absl::Span<const int64_t> expected_artifact_ids,
          absl::Span<const int64_t> expected_execution_ids,
          absl::Span<const int64_t> expected_context_ids,
          absl::Span<const std::pair<int64_t, int64_t>>
              expected_node_index_pairs_in_events) {
        GetLineageSubgraphRequest req;
        GetLineageSubgraphResponse resp;
        req.mutable_lineage_subgraph_query_options()->Swap(&options);
        EXPECT_EQ(metadata_store->GetLineageSubgraph(req, &resp),
                  absl::OkStatus());
        std::vector<std::pair<int64_t, int64_t>>
            expected_node_id_pairs_in_events;
        for (const auto& [artifact_index, execution_index] :
             expected_node_index_pairs_in_events) {
          expected_node_id_pairs_in_events.push_back(
              {want_artifacts.at(artifact_index).id(),
               want_executions.at(execution_index).id()});
        }
        VerifySubgraphSkeleton(resp.lineage_subgraph(), expected_artifact_ids,
                               expected_execution_ids,
                               expected_node_id_pairs_in_events);
        EXPECT_THAT(resp.lineage_subgraph().contexts(),
                    UnorderedPointwise(IdEquals(), expected_context_ids));
      };

  LineageSubgraphQueryOptions base_options;
  base_options.set_direction(LineageSubgraphQueryOptions::DOWNSTREAM);
  base_options.set_max_num_hops(1);
  base_options.mutable_starting_artifacts()->set_filter_query(
      " properties.p1.string_value = 'a0' ");
  {
    // Query from a_0 towards downstream with max_num_hops = 1.
    LineageSubgraphQueryOptions options = base_options;
    std::vector<int64_t> want_artifact_ids = {want_artifacts[0].id()};
    std::vector<int64_t> want_execution_ids;
    for (int i = 0; i < kTestNumExecutionsInLargeLineageGraph; i++) {
      want_execution_ids.push_back(want_executions[i].id());
    }
    std::vector<int64_t> want_context_ids = {want_contexts[0].id()};
    std::vector<std::pair<int64_t, int64_t>> want_node_index_pairs_in_events;
    for (int i = 0; i < kTestNumExecutionsInLargeLineageGraph; i++) {
      want_node_index_pairs_in_events.push_back({0, i});
    }
    verify_lineage_subgraph_with_direction(options, want_artifact_ids,
                                           want_execution_ids, want_context_ids,
                                           want_node_index_pairs_in_events);
  }
  {
    // Query from a_0 towards downstream with max_num_hops = 2.
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(2);
    std::vector<int64_t> want_artifact_ids;
    for (int i = 0; i < kTestNumArtifactsInLargeLineageGraph - 1; i++) {
      want_artifact_ids.push_back(want_artifacts[i].id());
    }
    std::vector<int64_t> want_execution_ids;
    for (int i = 0; i < kTestNumExecutionsInLargeLineageGraph; i++) {
      want_execution_ids.push_back(want_executions[i].id());
    }
    std::vector<int64_t> want_context_ids = {want_contexts[0].id(),
                                             want_contexts[1].id()};
    std::vector<std::pair<int64_t, int64_t>> want_node_index_pairs_in_events;
    for (int i = 0; i < kTestNumExecutionsInLargeLineageGraph; i++) {
      want_node_index_pairs_in_events.push_back({0, i});
      want_node_index_pairs_in_events.push_back({i + 1, i});
    }
    verify_lineage_subgraph_with_direction(options, want_artifact_ids,
                                           want_execution_ids, want_context_ids,
                                           want_node_index_pairs_in_events);
  }
  {
    // Query from a_0 towards downstream with max_num_hops = 3.
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(3);
    std::vector<int64_t> want_artifact_ids;
    for (int i = 0; i < kTestNumArtifactsInLargeLineageGraph - 1; i++) {
      want_artifact_ids.push_back(want_artifacts[i].id());
    }
    std::vector<int64_t> want_execution_ids;
    for (int i = 0; i < 2 * kTestNumExecutionsInLargeLineageGraph; i++) {
      want_execution_ids.push_back(want_executions[i].id());
    }
    std::vector<int64_t> want_context_ids = {want_contexts[0].id(),
                                             want_contexts[1].id()};
    std::vector<std::pair<int64_t, int64_t>> want_node_index_pairs_in_events;
    for (int i = 0; i < kTestNumExecutionsInLargeLineageGraph; i++) {
      want_node_index_pairs_in_events.push_back({0, i});
      want_node_index_pairs_in_events.push_back({i + 1, i});
      want_node_index_pairs_in_events.push_back(
          {i + 1, i + kTestNumExecutionsInLargeLineageGraph});
    }
    verify_lineage_subgraph_with_direction(options, want_artifact_ids,
                                           want_execution_ids, want_context_ids,
                                           want_node_index_pairs_in_events);
  }
  {
    // Query from a_0 towards downstream with max_num_hops = 4.
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(4);
    std::vector<int64_t> want_artifact_ids;
    for (int i = 0; i < kTestNumArtifactsInLargeLineageGraph; i++) {
      want_artifact_ids.push_back(want_artifacts[i].id());
    }
    std::vector<int64_t> want_execution_ids;
    for (int i = 0; i < 2 * kTestNumExecutionsInLargeLineageGraph; i++) {
      want_execution_ids.push_back(want_executions[i].id());
    }
    std::vector<int64_t> want_context_ids = {want_contexts[0].id(),
                                             want_contexts[1].id()};
    std::vector<std::pair<int64_t, int64_t>> want_node_index_pairs_in_events;
    for (int i = 0; i < kTestNumExecutionsInLargeLineageGraph; i++) {
      want_node_index_pairs_in_events.push_back({0, i});
      want_node_index_pairs_in_events.push_back({i + 1, i});
      want_node_index_pairs_in_events.push_back(
          {kTestNumArtifactsInLargeLineageGraph - 1,
           i + kTestNumExecutionsInLargeLineageGraph});
      want_node_index_pairs_in_events.push_back(
          {i + 1, i + kTestNumExecutionsInLargeLineageGraph});
    }
    verify_lineage_subgraph_with_direction(options, want_artifact_ids,
                                           want_execution_ids, want_context_ids,
                                           want_node_index_pairs_in_events);
  }
  {
    // Query from a_0 bidirectionally with max_num_hops = 3.
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(3);
    options.set_direction(LineageSubgraphQueryOptions::BIDIRECTIONAL);
    std::vector<int64_t> want_artifact_ids;
    for (int i = 0; i < kTestNumArtifactsInLargeLineageGraph - 1; i++) {
      want_artifact_ids.push_back(want_artifacts[i].id());
    }
    std::vector<int64_t> want_execution_ids;
    for (int i = 0; i < 2 * kTestNumExecutionsInLargeLineageGraph; i++) {
      want_execution_ids.push_back(want_executions[i].id());
    }
    std::vector<int64_t> want_context_ids = {want_contexts[0].id(),
                                             want_contexts[1].id()};
    std::vector<std::pair<int64_t, int64_t>> want_node_index_pairs_in_events;
    for (int i = 0; i < kTestNumExecutionsInLargeLineageGraph; i++) {
      want_node_index_pairs_in_events.push_back({0, i});
      want_node_index_pairs_in_events.push_back({i + 1, i});
      want_node_index_pairs_in_events.push_back(
          {i + 1, i + kTestNumExecutionsInLargeLineageGraph});
    }
    verify_lineage_subgraph_with_direction(options, want_artifact_ids,
                                           want_execution_ids, want_context_ids,
                                           want_node_index_pairs_in_events);
  }
  {
    // Query from a_0 bidirectionally with max_num_hops = 4.
    LineageSubgraphQueryOptions options = base_options;
    options.set_max_num_hops(4);
    options.set_direction(LineageSubgraphQueryOptions::BIDIRECTIONAL);
    std::vector<int64_t> want_artifact_ids;
    for (int i = 0; i < kTestNumArtifactsInLargeLineageGraph; i++) {
      want_artifact_ids.push_back(want_artifacts[i].id());
    }
    std::vector<int64_t> want_execution_ids;
    for (int i = 0; i < 2 * kTestNumExecutionsInLargeLineageGraph; i++) {
      want_execution_ids.push_back(want_executions[i].id());
    }
    std::vector<int64_t> want_context_ids = {want_contexts[0].id(),
                                             want_contexts[1].id()};
    std::vector<std::pair<int64_t, int64_t>> want_node_index_pairs_in_events;
    for (int i = 0; i < kTestNumExecutionsInLargeLineageGraph; i++) {
      want_node_index_pairs_in_events.push_back({0, i});
      want_node_index_pairs_in_events.push_back({i + 1, i});
      want_node_index_pairs_in_events.push_back(
          {kTestNumArtifactsInLargeLineageGraph - 1,
           i + kTestNumExecutionsInLargeLineageGraph});
      want_node_index_pairs_in_events.push_back(
          {i + 1, i + kTestNumExecutionsInLargeLineageGraph});
    }
    verify_lineage_subgraph_with_direction(options, want_artifact_ids,
                                           want_execution_ids, want_context_ids,
                                           want_node_index_pairs_in_events);
  }
}

TEST(MetadataStoreExtendedTest, GetLineageSubgraphErrors) {
  // Prepare a store with the lineage graph
  std::unique_ptr<MetadataStore> metadata_store = CreateMetadataStore();
  int64_t min_creation_time;
  std::vector<Artifact> want_artifacts;
  std::vector<Execution> want_executions;
  ASSERT_EQ(CreateLineageGraph(*metadata_store, min_creation_time,
                               want_artifacts, want_executions),
            absl::OkStatus());

  // Set up a valid `base_request` for the rest of test with:
  // 1. `starting_artifacts.filter_query` specified with valid syntax.
  // 2. `max_num_hops` set to 20.
  GetLineageSubgraphRequest base_request = GetLineageSubgraphRequest();
  base_request.mutable_lineage_subgraph_query_options()
      ->mutable_starting_artifacts()
      ->set_filter_query("uri = 'uri://foo/a4'");
  base_request.mutable_lineage_subgraph_query_options()->set_max_num_hops(20);

  GetLineageSubgraphResponse resp;
  {
    // No starting_nodes specified.
    GetLineageSubgraphRequest req = base_request;
    req.mutable_lineage_subgraph_query_options()->clear_starting_nodes();
    absl::Status status = metadata_store->GetLineageSubgraph(req, &resp);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
    EXPECT_TRUE(absl::StrContains(
        status.message(), "Missing arguments for listing starting nodes."));
  }

  {
    // No starting_nodes.filter_query.
    GetLineageSubgraphRequest req = base_request;
    req.mutable_lineage_subgraph_query_options()
        ->mutable_starting_artifacts()
        ->clear_filter_query();
    absl::Status status = metadata_store->GetLineageSubgraph(req, &resp);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
    EXPECT_TRUE(absl::StrContains(
        status.message(),
        "Cannot list starting nodes if `filter_query` is unspecified."));
  }

  {
    // invalid query syntax
    GetLineageSubgraphRequest req = base_request;
    req.mutable_lineage_subgraph_query_options()
        ->mutable_starting_artifacts()
        ->set_filter_query("invalid query syntax");
    absl::Status status = metadata_store->GetLineageSubgraph(req, &resp);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
  }

  {
    // invalid max_num_hops.
    GetLineageSubgraphRequest req = base_request;
    req.mutable_lineage_subgraph_query_options()->set_max_num_hops(-1);
    absl::Status status = metadata_store->GetLineageSubgraph(req, &resp);
    EXPECT_TRUE(absl::IsInvalidArgument(status));
    EXPECT_TRUE(
        absl::StrContains(status.message(), "max_num_hops cannot be negative"));
  }

  {
    // query_nodes does not match any nodes.
    GetLineageSubgraphRequest req = base_request;
    req.mutable_lineage_subgraph_query_options()
        ->mutable_starting_artifacts()
        ->set_filter_query("name = 'non_existing_artifact'");
    absl::Status status = metadata_store->GetLineageSubgraph(req, &resp);
    EXPECT_TRUE(absl::IsNotFound(status));
  }
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    MetadataStoreTest, MetadataStoreTestSuite, ::testing::Values([]() {
      return std::make_unique<RDBMSMetadataStoreContainer>();
    }));

}  // namespace testing
}  // namespace ml_metadata
