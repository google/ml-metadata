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
#include "ml_metadata/tools/mlmd_bench/read_events_workload.h"

#include <random>
#include <vector>

#include "absl/time/clock.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {
namespace {

constexpr int64 kInt64IdSize = 8;
constexpr int64 kNumNodeIdsPerEdge = 2;
constexpr int64 kEventTypeSize = 1;
constexpr int64 kTimeEpochSize = 1;

template <typename T>
void InitializeReadRequest(ReadEventsWorkItemType& read_request) {
  read_request.emplace<T>();
}

// Calculates and returns the transferred bytes of the `event`.
int64 GetTransferredBytesForEvent(const Event& event) {
  // Includes the transferred bytes for `artifact_id`, `execution_id`, `type`
  // and `last_update_time_since_epoch` for current event.
  int64 bytes =
      kInt64IdSize * kNumNodeIdsPerEdge + kEventTypeSize + kTimeEpochSize;
  // Includes the transferred bytes for event's path.
  for (const auto& step : event.path().steps()) {
    bytes += step.key().size();
  }
  return bytes;
}

tensorflow::Status GetTransferredBytes(
    const ReadEventsConfig& read_events_config,
    const ReadEventsWorkItemType& request, MetadataStore& store,
    int64& curr_bytes) {
  switch (read_events_config.specification()) {
    case ReadEventsConfig::EVENTS_BY_ARTIFACT_IDS: {
      GetEventsByArtifactIDsResponse response;
      TF_RETURN_IF_ERROR(store.GetEventsByArtifactIDs(
          absl::get<GetEventsByArtifactIDsRequest>(request), &response));
      for (const Event& event : response.events()) {
        curr_bytes += GetTransferredBytesForEvent(event);
      }
      break;
    }
    case ReadEventsConfig::EVENTS_BY_EXECUTION_IDS: {
      GetEventsByExecutionIDsResponse response;
      TF_RETURN_IF_ERROR(store.GetEventsByExecutionIDs(
          absl::get<GetEventsByExecutionIDsRequest>(request), &response));
      for (const Event& event : response.events()) {
        curr_bytes += GetTransferredBytesForEvent(event);
      }
      break;
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
  return tensorflow::Status::OK();
}

}  // namespace

ReadEvents::ReadEvents(const ReadEventsConfig& read_events_config,
                       const int64 num_operations)
    : read_events_config_(read_events_config),
      num_operations_(num_operations),
      name_(absl::StrCat("READ_", read_events_config_.Specification_Name(
                                      read_events_config_.specification()))) {}

tensorflow::Status ReadEvents::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  std::vector<Node> existing_nodes;
  TF_RETURN_IF_ERROR(
      GetExistingNodes(read_events_config_, *store, existing_nodes));
  std::uniform_int_distribution<int64> node_index_dist{
      0, (int64)(existing_nodes.size() - 1)};

  UniformDistribution num_ids_proto_dist = read_events_config_.num_ids();
  std::uniform_int_distribution<int64> num_ids_dist{
      num_ids_proto_dist.minimum(), num_ids_proto_dist.maximum()};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  for (int64 i = 0; i < num_operations_; ++i) {
    int64 curr_bytes = 0;
    ReadEventsWorkItemType request;
    const int64 num_ids = num_ids_dist(gen);
    for (int64 j = 0; j < num_ids; ++j) {
      const int64 node_index = node_index_dist(gen);
      switch (read_events_config_.specification()) {
        case ReadEventsConfig::EVENTS_BY_ARTIFACT_IDS: {
          InitializeReadRequest<GetEventsByArtifactIDsRequest>(request);
          absl::get<GetEventsByArtifactIDsRequest>(request).add_artifact_ids(
              absl::get<Artifact>(existing_nodes[node_index]).id());
          break;
        }
        case ReadEventsConfig::EVENTS_BY_EXECUTION_IDS: {
          InitializeReadRequest<GetEventsByExecutionIDsRequest>(request);
          absl::get<GetEventsByExecutionIDsRequest>(request).add_execution_ids(
              absl::get<Execution>(existing_nodes[node_index]).id());
          break;
        }
        default:
          LOG(FATAL) << "Wrong specification for ReadEvents!";
      }
    }
    TF_RETURN_IF_ERROR(
        GetTransferredBytes(read_events_config_, request, *store, curr_bytes));
    work_items_.emplace_back(request, curr_bytes);
  }

  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status ReadEvents::RunOpImpl(const int64 work_items_index,
                                         MetadataStore* store) {
  switch (read_events_config_.specification()) {
    case ReadEventsConfig::EVENTS_BY_ARTIFACT_IDS: {
      GetEventsByArtifactIDsRequest request =
          absl::get<GetEventsByArtifactIDsRequest>(
              work_items_[work_items_index].first);
      GetEventsByArtifactIDsResponse response;
      return store->GetEventsByArtifactIDs(request, &response);
      break;
    }
    case ReadEventsConfig::EVENTS_BY_EXECUTION_IDS: {
      GetEventsByExecutionIDsRequest request =
          absl::get<GetEventsByExecutionIDsRequest>(
              work_items_[work_items_index].first);
      GetEventsByExecutionIDsResponse response;
      return store->GetEventsByExecutionIDs(request, &response);
      break;
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
}

tensorflow::Status ReadEvents::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string ReadEvents::GetName() { return name_; }

}  // namespace ml_metadata
