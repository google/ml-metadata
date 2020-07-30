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
#include "ml_metadata/tools/mlmd_bench/util.h"

#include <random>
#include <vector>

#include "absl/time/clock.h"
#include "absl/types/variant.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {
namespace {

template <typename T, typename NT>
void PrepareNode(const std::string& node_name, const T& curr_type,
                 NT& curr_node) {
  curr_node.set_type_id(curr_type.id());
  curr_node.set_name(node_name);
  (*curr_node.mutable_properties())["property"].set_string_value("*");
}

// Sets additional fields(uri, state) for artifacts.
void SetArtifactAdditionalFields(const string& node_name, Artifact& node) {
  node.set_uri(absl::StrCat(node_name, "_uri"));
  node.set_state(Artifact::UNKNOWN);
}

// Sets additional field(state) for executions.
void SetExecutionAdditionalFields(Execution& node) {
  node.set_last_known_state(Execution::UNKNOWN);
}

}  // namespace

tensorflow::Status GetExistingTypes(const int specification,
                                    MetadataStore* store,
                                    std::vector<Type>& existing_types) {
  switch (specification) {
    // Gets ArtifactTypes.
    case 0: {
      GetArtifactTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetArtifactTypes(
          /*request=*/{}, &get_response));
      for (auto& artifact_type : get_response.artifact_types()) {
        existing_types.push_back(artifact_type);
      }
      break;
    }
    // Gets ExecutionTypes.
    case 1: {
      GetExecutionTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetExecutionTypes(
          /*request=*/{}, &get_response));
      for (auto& execution_type : get_response.execution_types()) {
        existing_types.push_back(execution_type);
      }
      break;
    }
    // Gets ContextTypes.
    case 2: {
      GetContextTypesResponse get_response;
      TF_RETURN_IF_ERROR(store->GetContextTypes(
          /*request=*/{}, &get_response));
      for (auto& context_type : get_response.context_types()) {
        existing_types.push_back(context_type);
      }
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for getting types in db!";
  }
  return tensorflow::Status::OK();
}

tensorflow::Status GetExistingNodes(const int specification,
                                    MetadataStore* store,
                                    std::vector<NodeType>& existing_nodes) {
  switch (specification) {
    // Gets Artifacts.
    case 0: {
      GetArtifactsResponse get_response;
      TF_RETURN_IF_ERROR(store->GetArtifacts(
          /*request=*/{}, &get_response));
      for (auto& artifact : get_response.artifacts()) {
        existing_nodes.push_back(artifact);
      }
      break;
    }
    // Gets Executions.
    case 1: {
      GetExecutionsResponse get_response;
      TF_RETURN_IF_ERROR(store->GetExecutions(
          /*request=*/{}, &get_response));
      for (auto& execution : get_response.executions()) {
        existing_nodes.push_back(execution);
      }
      break;
    }
    // Gets Contexts.
    case 2: {
      GetContextsResponse get_response;
      TF_RETURN_IF_ERROR(store->GetContexts(
          /*request=*/{}, &get_response));
      for (auto& context : get_response.contexts()) {
        existing_nodes.push_back(context);
      }
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for getting nodes in db!";
  }
  return tensorflow::Status::OK();
}

tensorflow::Status InsertTypesInDb(const int64 num_artifact_types,
                                   const int64 num_execution_types,
                                   const int64 num_context_types,
                                   MetadataStore* store) {
  PutTypesRequest put_request;
  PutTypesResponse put_response;

  for (int64 i = 0; i < num_artifact_types; i++) {
    ArtifactType* curr_type = put_request.add_artifact_types();
    curr_type->set_name(absl::StrCat("pre_insert_artifact_type-", i));
    (*curr_type->mutable_properties())["property"] = STRING;
  }

  for (int64 i = 0; i < num_execution_types; i++) {
    ExecutionType* curr_type = put_request.add_execution_types();
    curr_type->set_name(absl::StrCat("pre_insert_execution_type-", i));
    (*curr_type->mutable_properties())["property"] = STRING;
  }

  for (int64 i = 0; i < num_context_types; i++) {
    ContextType* curr_type = put_request.add_context_types();
    curr_type->set_name(absl::StrCat("pre_insert_context_type-", i));
    (*curr_type->mutable_properties())["property"] = STRING;
  }

  return store->PutTypes(put_request, &put_response);
}

tensorflow::Status InsertNodesInDb(const int64 num_artifact_nodes,
                                   const int64 num_execution_nodes,
                                   const int64 num_context_nodes,
                                   MetadataStore* store) {
  std::vector<Type> existing_artifact_types;
  GetExistingTypes(/*specification=*/0, store, existing_artifact_types);
  std::vector<Type> existing_execution_types;
  GetExistingTypes(/*specification=*/1, store, existing_execution_types);
  std::vector<Type> existing_context_types;
  GetExistingTypes(/*specification=*/2, store, existing_context_types);

  std::uniform_int_distribution<int64> uniform_dist_artifact_type_index{
      0, (int64)(existing_artifact_types.size() - 1)};
  std::uniform_int_distribution<int64> uniform_dist_execution_type_index{
      0, (int64)(existing_execution_types.size() - 1)};
  std::uniform_int_distribution<int64> uniform_dist_context_type_index{
      0, (int64)(existing_context_types.size() - 1)};

  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  for (int64 i = 0; i < num_artifact_nodes; i++) {
    const int64 type_index = uniform_dist_artifact_type_index(gen);
    const string& node_name = absl::StrCat("pre_insert_artifact-", i);
    PutArtifactsRequest put_request;
    Artifact* curr_node = put_request.add_artifacts();
    PrepareNode<ArtifactType, Artifact>(
        node_name, absl::get<ArtifactType>(existing_artifact_types[type_index]),
        *curr_node);
    SetArtifactAdditionalFields(node_name, *curr_node);
    PutArtifactsResponse put_response;
    TF_RETURN_IF_ERROR(store->PutArtifacts(put_request, &put_response));
  }

  for (int64 i = 0; i < num_execution_nodes; i++) {
    const int64 type_index = uniform_dist_execution_type_index(gen);
    const string& node_name = absl::StrCat("pre_insert_execution-", i);
    PutExecutionsRequest put_request;
    Execution* curr_node = put_request.add_executions();
    PrepareNode<ExecutionType, Execution>(
        node_name,
        absl::get<ExecutionType>(existing_execution_types[type_index]),
        *curr_node);
    SetExecutionAdditionalFields(*curr_node);
    PutExecutionsResponse put_response;
    TF_RETURN_IF_ERROR(store->PutExecutions(put_request, &put_response));
  }

  for (int64 i = 0; i < num_context_nodes; i++) {
    const int64 type_index = uniform_dist_context_type_index(gen);
    const string& node_name = absl::StrCat("pre_insert_context-", i);
    PutContextsRequest put_request;
    PrepareNode<ContextType, Context>(
        node_name, absl::get<ContextType>(existing_context_types[type_index]),
        *put_request.add_contexts());
    PutContextsResponse put_response;
    TF_RETURN_IF_ERROR(store->PutContexts(put_request, &put_response));
  }

  return tensorflow::Status::OK();
}

int64 GenerateRandomNumberFromUD(const UniformDistribution& dist,
                                 std::minstd_rand0& gen) {
  std::uniform_int_distribution<int64> uniform_dist{dist.minimum(),
                                                    dist.maximum()};
  return uniform_dist(gen);
}

}  // namespace ml_metadata
