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
#include "ml_metadata/tools/mlmd_bench/fill_types_workload.h"

#include <random>

#include "absl/strings/substitute.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {
namespace {

// A template function where the Type can be ArtifactType / ExecutionType /
// ContextType. It takes a `type_name` to generate a type and generates number
// of properties w.r.t. to the uniform distribution.
template <typename Type>
void GenerateRandomType(const std::string& type_name,
                        std::uniform_int_distribution<int64>& uniform_dist,
                        std::minstd_rand0& gen, Type* type, int64* curr_bytes) {
  // The random type name will be a random number.
  type->set_name(type_name);
  // The curr_bytes records the total transferred bytes for executing each work
  // item.
  *curr_bytes += type->name().size();
  // Generates the number of properties for each type
  // w.r.t. the uniform distribution
  const int64 num_properties = uniform_dist(gen);
  for (int64 i = 0; i < num_properties; i++) {
    (*type->mutable_properties())[absl::StrCat("p-", i)] = STRING;
    *curr_bytes += absl::StrCat("p-", i).size();
  }
}

}  // namespace

FillTypes::FillTypes(const FillTypesConfig& fill_types_config,
                     int64 num_operations)
    : fill_types_config_(fill_types_config), num_operations_(num_operations) {
  switch (fill_types_config_.specification()) {
    case FillTypesConfig::ARTIFACT_TYPE: {
      name_ = "fill_artifact_type";
      break;
    }
    case FillTypesConfig::EXECUTION_TYPE: {
      name_ = "fill_execution_type";
      break;
    }
    case FillTypesConfig::CONTEXT_TYPE: {
      name_ = "fill_context_type";
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillTypes!";
  }
}

tensorflow::Status FillTypes::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  int64 curr_bytes = 0;
  // Uniform distribution that describes the number of properties for each
  // generated types.
  UniformDistribution num_properties = fill_types_config_.num_properties();
  int64 min = num_properties.minimum();
  int64 max = num_properties.maximum();
  std::uniform_int_distribution<int64> uniform_dist{min, max};
  // The seed for the random generator is the time when the FillTypes is
  // created.
  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  // TODO(briansong): Add update support.
  for (int64 i = 0; i < num_operations_; i++) {
    curr_bytes = 0;
    FillTypeWorkItemType put_request;
    const std::string type_name = absl::StrCat("type_", i);
    switch (fill_types_config_.specification()) {
      case FillTypesConfig::ARTIFACT_TYPE: {
        put_request.emplace<PutArtifactTypeRequest>();
        GenerateRandomType<ArtifactType>(
            type_name, uniform_dist, gen,
            absl::get<PutArtifactTypeRequest>(put_request)
                .mutable_artifact_type(),
            &curr_bytes);
        break;
      }
      case FillTypesConfig::EXECUTION_TYPE: {
        put_request.emplace<PutExecutionTypeRequest>();
        GenerateRandomType<ExecutionType>(
            type_name, uniform_dist, gen,
            absl::get<PutExecutionTypeRequest>(put_request)
                .mutable_execution_type(),
            &curr_bytes);
        break;
      }
      case FillTypesConfig::CONTEXT_TYPE: {
        put_request.emplace<PutContextTypeRequest>();
        GenerateRandomType<ContextType>(
            type_name, uniform_dist, gen,
            absl::get<PutContextTypeRequest>(put_request)
                .mutable_context_type(),
            &curr_bytes);
        break;
      }
      default:
        return tensorflow::errors::InvalidArgument("Wrong specification!");
    }
    work_items_.emplace_back(put_request, curr_bytes);
  }
  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status FillTypes::RunOpImpl(int64 i, MetadataStore* store) {
  switch (fill_types_config_.specification()) {
    case FillTypesConfig::ARTIFACT_TYPE: {
      PutArtifactTypeRequest put_request =
          absl::get<PutArtifactTypeRequest>(work_items_[i].first);
      PutArtifactTypeResponse put_response;
      return store->PutArtifactType(put_request, &put_response);
    }
    case FillTypesConfig::EXECUTION_TYPE: {
      PutExecutionTypeRequest put_request =
          absl::get<PutExecutionTypeRequest>(work_items_[i].first);
      PutExecutionTypeResponse put_response;
      return store->PutExecutionType(put_request, &put_response);
    }
    case FillTypesConfig::CONTEXT_TYPE: {
      PutContextTypeRequest put_request =
          absl::get<PutContextTypeRequest>(work_items_[i].first);
      PutContextTypeResponse put_response;
      return store->PutContextType(put_request, &put_response);
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
  return tensorflow::errors::InvalidArgument(
      "Cannot execute the query due to wrong specification!");
}

tensorflow::Status FillTypes::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string FillTypes::GetName() { return name_; }

}  // namespace ml_metadata
