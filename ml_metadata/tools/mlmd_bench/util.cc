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

#include <vector>

#include "absl/types/variant.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

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

}  // namespace ml_metadata
