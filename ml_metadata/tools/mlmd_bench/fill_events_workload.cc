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
#include "ml_metadata/tools/mlmd_bench/fill_events_workload.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace ml_metadata {
namespace {

constexpr int64 kInt64IdSize = 8;
constexpr int64 kNumNodeIdsPerEdge = 2;
constexpr int64 kEventTypeSize = 1;

// Validates correct popularity distribution for artifacts under input / output
// events. Returns INVALID_ARGUMENT error if the specified distribution is not
// correct or not set.
tensorflow::Status ValidateCorrectArtifactsDistributionType(
    const FillEventsConfig& fill_events_config) {
  switch (fill_events_config.artifact_node_popularity_case()) {
    case FillEventsConfig::kArtifactNodePopularityCategorical: {
      if (fill_events_config.specification() == FillEventsConfig::INPUT) {
        return tensorflow::errors::InvalidArgument(
            "Input event should has a zipf distribution popularity for "
            "artifacts.");
      }
      break;
    }
    case FillEventsConfig::kArtifactNodePopularityZipf: {
      if (fill_events_config.specification() == FillEventsConfig::OUTPUT) {
        return tensorflow::errors::InvalidArgument(
            "Output event should has a categorical distribution popularity for "
            "artifacts.");
      }
      break;
    }
    default:
      return tensorflow::errors::InvalidArgument(
          "Artifacts' popularity is not set!");
  }
  return tensorflow::Status::OK();
}

// Generates and returns a categorical distribution with Dirichlet Prior
// specified by `dist`.
std::discrete_distribution<int64>
GenerateCategoricalDistributionWithDirichletPrior(
    const int64 sample_size, const CategoricalDistribution& dist,
    std::minstd_rand0& gen) {
  // With a source of Gamma-distributed random variates, draws `sample_size`
  // independent random samples and store in `weights`.
  std::gamma_distribution<double> gamma_distribution(dist.dirichlet_alpha(),
                                                     1.0);
  std::vector<double> weights(sample_size);
  for (int64 i = 0; i < sample_size; ++i) {
    weights[i] = gamma_distribution(gen);
  }
  // Uses these random number generated w.r.t. a Dirichlet distribution with
  // `concentration_param` to represent the possibility of being chosen for each
  // integer within [0, sample_size) in a discrete distribution.
  return std::discrete_distribution<int64>{weights.begin(), weights.end()};
}

// Generates and returns a zipf distribution with a configurable `skew`
// specified by `dist`.
std::discrete_distribution<int64> GenerateZipfDistributionWithConfigurableSkew(
    const int64 sample_size, const ZipfDistribution& dist,
    std::minstd_rand0& gen) {
  std::vector<double> weights(sample_size);
  for (int64 i = 0; i < sample_size; ++i) {
    const int64 rank = i + 1;
    // Here, we discard the normalize factor since the `discrete_distribution`
    // will perform the normalization for us.
    weights[i] = 1 / pow(rank, dist.skew());
  }
  // Random shuffles the weight vector.
  std::shuffle(std::begin(weights), std::end(weights), gen);
  // Uses these random number generated w.r.t. a zipf distribution with a
  // configurable `skew` to represent the possibility of being chosen for each
  // integer within [0, sample_size) in a discrete distribution.
  return std::discrete_distribution<int64>{weights.begin(), weights.end()};
}

// Generates and returns the artifact popularity distribution according to the
// type of event.
std::discrete_distribution<int64> GeneratePopularityDistributionForArtifacts(
    const FillEventsConfig& fill_events_config, const int64 sample_size,
    std::minstd_rand0& gen) {
  switch (fill_events_config.specification()) {
    case FillEventsConfig::INPUT: {
      return GenerateZipfDistributionWithConfigurableSkew(
          sample_size, fill_events_config.artifact_node_popularity_zipf(), gen);
    }
    case FillEventsConfig::OUTPUT: {
      return GenerateCategoricalDistributionWithDirichletPrior(
          sample_size,
          fill_events_config.artifact_node_popularity_categorical(), gen);
    }
    default:
      LOG(FATAL) << "Wrong specification for FillEvents!";
  }
}

// Checks if `output_artifact_id` has been outputted in current setup(all the
// previous outputted artifact ids have been stored inside
// `output_artifact_ids`). Returns true if `output_artifact_id` has been
// outputted before, otherwise, returns false.
bool CheckArtifactNotAlreadyBeenOutputtedInSetUpAndRecordItsId(
    const int64 output_artifact_id,
    absl::flat_hash_set<int64>& output_artifact_ids) {
  if (output_artifact_ids.find(output_artifact_id) !=
      output_artifact_ids.end()) {
    return false;
  }
  output_artifact_ids.insert(output_artifact_id);
  return true;
}

// Checks if current artifact whose id is `output_artifact_id` has been
// outputted by other events inside db before. Returns ALREADY_EXISTS error, if
// the current artifact has been outputted before. Returns detailed error if
// query executions failed.
tensorflow::Status CheckArtifactNotAlreadyBeenOutputtedInDb(
    const int64 output_artifact_id, MetadataStore& store) {
  GetEventsByArtifactIDsRequest request;
  request.add_artifact_ids(output_artifact_id);
  GetEventsByArtifactIDsResponse response;
  TF_RETURN_IF_ERROR(store.GetEventsByArtifactIDs(request, &response));
  for (const auto& event : response.events()) {
    if (event.type() == Event::OUTPUT) {
      return tensorflow::errors::AlreadyExists(
          ("The current artifact has been outputted by another output event "
           "inside db!"));
    }
  }
  return tensorflow::Status::OK();
}

// Calculates and returns the transferred bytes of the `event`.
int64 GetTransferredBytes(const Event& event) {
  // Includes the transferred bytes for `artifact_id`, `execution_id` and `type`
  // for current event(the type of id is int64 that takes eight bytes while the
  // type is an enum that takes one bytes).
  int64 bytes = kInt64IdSize * kNumNodeIdsPerEdge + kEventTypeSize;
  // Includes the transferred bytes for event's path.
  for (const auto& step : event.path().steps()) {
    bytes += step.key().size();
  }
  return bytes;
}

// Sets the properties of current event while recording the transferred bytes
// for it.
void SetEvent(const FillEventsConfig& fill_events_config,
              const int64 artifact_id, const int64 execution_id, Event& event,
              int64& curr_bytes) {
  switch (fill_events_config.specification()) {
    case FillEventsConfig::INPUT: {
      event.set_type(Event::INPUT);
      break;
    }
    case FillEventsConfig::OUTPUT: {
      event.set_type(Event::OUTPUT);
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillEvents!";
  }
  event.set_artifact_id(artifact_id);
  event.set_execution_id(execution_id);
  // TODO(briansong) Adds an additional field in protocol message in
  // FillEventsConfig that describe the string length of key and supports
  // index for step value as well.
  event.mutable_path()->add_steps()->set_key("foo");
  curr_bytes += GetTransferredBytes(event);
}

// Generates `num_events` events within `request`.
// Selects `num_events` artifact and execution node pairs from
// `existing_artifact_nodes` and `existing_execution_nodes` w.r.t. to the
// node popularity distribution. Also, performs rejection sampling to ensure
// that each artifact is only be outputted once.
// Returns detailed error if query executions failed.
tensorflow::Status GenerateEvent(
    const FillEventsConfig& fill_events_config,
    const std::vector<Node>& existing_artifact_nodes,
    const std::vector<Node>& existing_execution_nodes, const int64 num_events,
    std::discrete_distribution<int64>& artifact_index_dist,
    std::discrete_distribution<int64>& execution_index_dist,
    std::minstd_rand0& gen, MetadataStore& store,
    absl::flat_hash_set<int64>& output_artifact_ids, PutEventsRequest& request,
    int64& curr_bytes) {
  int64 i = 0;
  while (i < num_events) {
    const int64 artifact_id =
        absl::get<Artifact>(existing_artifact_nodes[artifact_index_dist(gen)])
            .id();
    const int64 execution_id =
        absl::get<Execution>(
            existing_execution_nodes[execution_index_dist(gen)])
            .id();
    if (fill_events_config.specification() == FillEventsConfig::OUTPUT) {
      const bool artifact_not_already_been_outputted_in_setup =
          CheckArtifactNotAlreadyBeenOutputtedInSetUpAndRecordItsId(
              artifact_id, output_artifact_ids);
      const tensorflow::Status status =
          CheckArtifactNotAlreadyBeenOutputtedInDb(artifact_id, store);
      if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
        return status;
      }
      // Rejection sampling.
      if (!artifact_not_already_been_outputted_in_setup || !status.ok()) {
        continue;
      }
    }
    SetEvent(fill_events_config, artifact_id, execution_id,
             *request.add_events(), curr_bytes);
    i++;
  }
  return tensorflow::Status::OK();
}

}  // namespace

FillEvents::FillEvents(const FillEventsConfig& fill_events_config,
                       int64 num_operations)
    : fill_events_config_(fill_events_config),
      num_operations_(num_operations),
      name_(absl::StrCat("FILL_",
                         fill_events_config_.Specification_Name(
                             fill_events_config_.specification()),
                         "_EVENT")) {
  TF_CHECK_OK(ValidateCorrectArtifactsDistributionType(fill_events_config_));
}

tensorflow::Status FillEvents::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  std::uniform_int_distribution<int64> num_events_dist{
      fill_events_config_.num_events().minimum(),
      fill_events_config_.num_events().maximum()};

  std::vector<Node> existing_artifact_nodes;
  std::vector<Node> existing_execution_nodes;
  TF_RETURN_IF_ERROR(GetExistingNodes(fill_events_config_, *store,
                                      existing_artifact_nodes,
                                      existing_execution_nodes));

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  std::discrete_distribution<int64> execution_index_dist =
      GenerateCategoricalDistributionWithDirichletPrior(
          existing_execution_nodes.size(),
          fill_events_config_.execution_node_popularity(), gen);
  std::discrete_distribution<int64> artifact_index_dist =
      GeneratePopularityDistributionForArtifacts(
          fill_events_config_, existing_artifact_nodes.size(), gen);

  for (int64 i = 0; i < num_operations_; ++i) {
    int64 curr_bytes = 0;
    PutEventsRequest put_request;
    const int64 num_events = num_events_dist(gen);
    TF_RETURN_IF_ERROR(GenerateEvent(
        fill_events_config_, existing_artifact_nodes, existing_execution_nodes,
        num_events, artifact_index_dist, execution_index_dist, gen, *store,
        output_artifact_ids_, put_request, curr_bytes));
    work_items_.emplace_back(put_request, curr_bytes);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status FillEvents::RunOpImpl(const int64 work_items_index,
                                         MetadataStore* store) {
  PutEventsRequest put_request = work_items_[work_items_index].first;
  PutEventsResponse put_response;
  return store->PutEvents(put_request, &put_response);
}

tensorflow::Status FillEvents::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string FillEvents::GetName() { return name_; }

}  // namespace ml_metadata
