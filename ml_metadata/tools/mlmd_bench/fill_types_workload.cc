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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
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

// Gets the number of current types(`num_curr_type`) and total
// types(`num_total_type`) for later insert or update. Also updates the
// `get_response`. Returns detailed error if query executions failed.
tensorflow::Status GetNumberOfTypes(const FillTypesConfig& fill_types_config,
                                    MetadataStore* store, int64& num_curr_type,
                                    int64& num_total_type,
                                    GetTypesResponseType& get_response) {
  GetArtifactTypesResponse get_artifact_type_response;
  TF_RETURN_IF_ERROR(store->GetArtifactTypes(
      /*request=*/{}, &get_artifact_type_response));
  int64 num_artifact_type = get_artifact_type_response.artifact_types_size();

  GetExecutionTypesResponse get_execution_type_response;
  TF_RETURN_IF_ERROR(store->GetExecutionTypes(
      /*request=*/{}, &get_execution_type_response));
  int64 num_execution_type = get_execution_type_response.execution_types_size();

  GetContextTypesResponse get_context_type_response;
  TF_RETURN_IF_ERROR(store->GetContextTypes(
      /*request=*/{}, &get_context_type_response));
  int64 num_context_type = get_context_type_response.context_types_size();

  switch (fill_types_config.specification()) {
    case FillTypesConfig::ARTIFACT_TYPE: {
      num_curr_type = num_artifact_type;
      get_response.emplace<GetArtifactTypesResponse>(
          get_artifact_type_response);
      break;
    }
    case FillTypesConfig::EXECUTION_TYPE: {
      num_curr_type = num_execution_type;
      get_response.emplace<GetExecutionTypesResponse>(
          get_execution_type_response);
      break;
    }
    case FillTypesConfig::CONTEXT_TYPE: {
      num_curr_type = num_context_type;
      get_response.emplace<GetContextTypesResponse>(get_context_type_response);
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillTypes!";
  }

  num_total_type = num_artifact_type + num_execution_type + num_context_type;
  return tensorflow::Status::OK();
}

// Inserts new types into the db if the current types inside db is not enough
// for update. Returns detailed error if query executions failed.
tensorflow::Status MakeUpTypesForUpdate(
    const FillTypesConfig& fill_types_config, MetadataStore* store,
    const int64 num_type_to_make_up) {
  FillTypesConfig make_up_fill_types_config = fill_types_config;
  make_up_fill_types_config.set_update(false);
  std::unique_ptr<FillTypes> make_up_fill_types = absl::make_unique<FillTypes>(
      FillTypes(make_up_fill_types_config, num_type_to_make_up));
  TF_RETURN_IF_ERROR(make_up_fill_types->SetUp(store));
  for (int64 i = 0; i < num_type_to_make_up; ++i) {
    OpStats op_stats;
    TF_RETURN_IF_ERROR(make_up_fill_types->RunOp(i, store, op_stats));
  }
  return tensorflow::Status::OK();
}

// Initializes the properties of the `put_request` according to
// `fill_types_config`.
void InitializePutRequest(const FillTypesConfig& fill_types_config,
                          FillTypeWorkItemType& put_request) {
  switch (fill_types_config.specification()) {
    case FillTypesConfig::ARTIFACT_TYPE: {
      put_request.emplace<PutArtifactTypeRequest>();
      if (fill_types_config.update()) {
        absl::get<PutArtifactTypeRequest>(put_request).set_can_add_fields(true);
      }
      break;
    }
    case FillTypesConfig::EXECUTION_TYPE: {
      put_request.emplace<PutExecutionTypeRequest>();
      if (fill_types_config.update()) {
        absl::get<PutExecutionTypeRequest>(put_request)
            .set_can_add_fields(true);
      }
      break;
    }
    case FillTypesConfig::CONTEXT_TYPE: {
      put_request.emplace<PutContextTypeRequest>();
      if (fill_types_config.update()) {
        absl::get<PutContextTypeRequest>(put_request).set_can_add_fields(true);
      }
      break;
    }
    default:
      LOG(FATAL) << "Wrong specification for FillTypes!";
  }
}

// A template function where the Type can be ArtifactType / ExecutionType /
// ContextType.
// For insert cases, it takes a `type_name` to generate a type and generates
// `number_properties` properties. It also increase the stored bytes for the
// `type` in `curr_bytes`.
// For update cases, it takes a `existed_type` to generate a `type` with
// additional `number_properties` properties. It also increase the stored bytes
// for the `type` in `curr_bytes`.
template <typename Type>
void GenerateOrUpdateType(const FillTypesConfig& fill_types_config,
                          const std::string& type_name,
                          const int64 num_properties, const Type& existed_type,
                          Type& type, int64& curr_bytes) {
  if (fill_types_config.update()) {
    // Update cases.
    // Except the new added fields, `type` will the same as `existed_type`.
    type = existed_type;
    // The `curr_bytes` records the total transferred bytes for executing each
    // work item.
    curr_bytes += existed_type.name().size();
    for (auto& pair : existed_type.properties()) {
      // pair.first is the property name of `existed_type`.
      curr_bytes += pair.first.size();
    }
    for (int64 i = 0; i < num_properties; i++) {
      (*type.mutable_properties())[absl::StrCat("add_p-", i)] = STRING;
      curr_bytes += absl::StrCat("add_p-", i).size();
    }
  } else {
    // Insert cases.
    type.set_name(type_name);
    curr_bytes += type.name().size();
    for (int64 i = 0; i < num_properties; i++) {
      (*type.mutable_properties())[absl::StrCat("p-", i)] = STRING;
      curr_bytes += absl::StrCat("p-", i).size();
    }
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
  if (fill_types_config_.update()) {
    name_ += "(update)";
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

  // Gets the number of current types(`num_curr_type`) and total
  // types(`num_total_type`) for later insert or update.
  int64 num_curr_type = 0, num_total_type = 0;
  GetTypesResponseType get_response;
  TF_RETURN_IF_ERROR(GetNumberOfTypes(fill_types_config_, store, num_curr_type,
                                      num_total_type, get_response));

  // If the number of current types is less than the update number of
  // operations, calls MakeUpTypesForUpdate() for inserting new types into the
  // db for later update.
  if (fill_types_config_.update() && num_curr_type < num_operations_) {
    const int64 num_type_to_make_up = num_operations_ - num_curr_type;
    TF_RETURN_IF_ERROR(
        MakeUpTypesForUpdate(fill_types_config_, store, num_type_to_make_up));
    // Updates the get_response to contain the most up-to-date types inside db.
    TF_RETURN_IF_ERROR(GetNumberOfTypes(fill_types_config_, store,
                                        num_curr_type, num_total_type,
                                        get_response));
  }

  for (int64 i = num_total_type; i < num_total_type + num_operations_; i++) {
    curr_bytes = 0;
    FillTypeWorkItemType put_request;
    InitializePutRequest(fill_types_config_, put_request);
    // Updates the first `num_operations_` types inside db.
    const int64 update_type_index = i - num_total_type;
    const std::string type_name = absl::StrCat("type_", i);
    const int64 num_properties = uniform_dist(gen);
    switch (fill_types_config_.specification()) {
      case FillTypesConfig::ARTIFACT_TYPE: {
        GenerateOrUpdateType<ArtifactType>(
            fill_types_config_, type_name, num_properties,
            absl::get<GetArtifactTypesResponse>(get_response)
                .artifact_types()[update_type_index],
            *absl::get<PutArtifactTypeRequest>(put_request)
                 .mutable_artifact_type(),
            curr_bytes);
        break;
      }
      case FillTypesConfig::EXECUTION_TYPE: {
        GenerateOrUpdateType<ExecutionType>(
            fill_types_config_, type_name, num_properties,
            absl::get<GetExecutionTypesResponse>(get_response)
                .execution_types()[update_type_index],
            *absl::get<PutExecutionTypeRequest>(put_request)
                 .mutable_execution_type(),
            curr_bytes);
        break;
      }
      case FillTypesConfig::CONTEXT_TYPE: {
        GenerateOrUpdateType<ContextType>(
            fill_types_config_, type_name, num_properties,
            absl::get<GetContextTypesResponse>(get_response)
                .context_types()[update_type_index],
            *absl::get<PutContextTypeRequest>(put_request)
                 .mutable_context_type(),
            curr_bytes);
        break;
      }
      default:
        return tensorflow::errors::InvalidArgument("Wrong specification!");
    }
    // Updates work_items_.
    work_items_.emplace_back(put_request, curr_bytes);
  }
  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status FillTypes::RunOpImpl(const int64 work_items_index,
                                        MetadataStore* store) {
  const int64 i = work_items_index;
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
