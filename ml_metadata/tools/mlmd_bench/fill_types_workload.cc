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
#include <type_traits>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
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

// Template function that initializes the properties of the `put_request`.
template <typename T>
void InitializePutRequest(const FillTypesConfig& fill_types_config,
                          FillTypesWorkItemType& put_request) {
  put_request.emplace<T>();
  if (fill_types_config.update()) {
    absl::get<T>(put_request).set_can_add_fields(true);
  }
}

// Template function for setting name and properties for insert types.
template <typename T>
void SetInsertType(const std::string& type_name, const int64 num_properties,
                   T& type) {
  type.set_name(type_name);
  for (int64 i = 0; i < num_properties; i++) {
    (*type.mutable_properties())[absl::StrCat("p-", i)] = STRING;
  }
}

// Template function for setting additional properties for update types.
template <typename T>
void SetUpdateType(const T& existed_type, const int64 num_properties, T& type) {
  type = existed_type;
  for (int64 i = 0; i < num_properties; i++) {
    (*type.mutable_properties())[absl::StrCat("add_p-", i)] = STRING;
  }
}

// Prepares update types.
// If `type_index` is less than the length of the `existing_types`, return the
// existed type. Otherwise, the current existing types inside db are not enough
// for update. Then, makes up new type and inserts it into the db and return
// this make-up type. Returns detailed error if query executions failed.
template <typename T>
tensorflow::Status PrepareTypeForUpdate(const int64 type_index,
                                        const std::string& type_name,
                                        const int64 num_properties,
                                        const std::vector<Type>& existing_types,
                                        MetadataStore* store,
                                        T& existing_type) {
  if (type_index < existing_types.size()) {
    existing_type = absl::get<T>(existing_types[type_index]);
    return tensorflow::Status::OK();
  }

  PutTypesRequest put_request;
  PutTypesResponse put_response;
  SetInsertType<T>(type_name, num_properties, existing_type);
  if (std::is_same<T, ArtifactType>::value) {
    (*put_request.add_artifact_types()).CopyFrom(existing_type);
    TF_RETURN_IF_ERROR(store->PutTypes(put_request, &put_response));
    existing_type.set_id(put_response.artifact_type_ids(0));
  } else if (std::is_same<T, ExecutionType>::value) {
    (*put_request.add_execution_types()).CopyFrom(existing_type);
    TF_RETURN_IF_ERROR(store->PutTypes(put_request, &put_response));
    existing_type.set_id(put_response.execution_type_ids(0));
  } else if (std::is_same<T, ContextType>::value) {
    (*put_request.add_context_types()).CopyFrom(existing_type);
    TF_RETURN_IF_ERROR(store->PutTypes(put_request, &put_response));
    existing_type.set_id(put_response.context_type_ids(0));
  } else {
    LOG(FATAL) << "Unexpected types used for the workload.";
  }
  return tensorflow::Status::OK();
}

// Calculates the transferred bytes for each types that will be inserted or
// updated later.
template <typename T>
tensorflow::Status GetTransferredBytes(const T& type, int64& curr_bytes) {
  curr_bytes += type.name().size();
  for (auto& pair : type.properties()) {
    // Includes the bytes for properties' name size.
    curr_bytes += pair.first.size();
    // Includes the bytes for properties' value enumeration size.
    if (pair.second == PropertyType::UNKNOWN) {
      return tensorflow::errors::InvalidArgument("Invalid PropertyType!");
    }
    // As we uses a TINYINT to store the enum.
    curr_bytes += 1;
  }
  return tensorflow::Status::OK();
}

// Generates insert / update type.
// For insert cases, it takes a `type_name` and `number_properties`to set the
// insert type. For update cases, it prepares the `existing_type`, takes
// it and `number_properties`to set the update type. Gets the transferred
// bytes for both cases at the end. Returns detailed error if query executions
// failed.
template <typename T>
tensorflow::Status GenerateType(const FillTypesConfig& fill_types_config,
                                const int64 update_type_index,
                                const std::string& type_name,
                                const int64 num_properties,
                                const std::vector<Type>& existing_types,
                                MetadataStore* store, T& type,
                                int64& curr_bytes) {
  CHECK((std::is_same<T, ArtifactType>::value ||
         std::is_same<T, ExecutionType>::value ||
         std::is_same<T, ContextType>::value))
      << "Unexpected Types";
  if (fill_types_config.update()) {
    // Update cases.
    T existing_type;
    TF_RETURN_IF_ERROR(PrepareTypeForUpdate<T>(update_type_index, type_name,
                                               num_properties, existing_types,
                                               store, existing_type));
    SetUpdateType<T>(existing_type, num_properties, type);
  } else {
    // Insert cases.
    SetInsertType<T>(type_name, num_properties, type);
  }
  return GetTransferredBytes(type, curr_bytes);
}

}  // namespace

FillTypes::FillTypes(const FillTypesConfig& fill_types_config,
                     const int64 num_operations)
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
  std::uniform_int_distribution<int64> uniform_dist{num_properties.minimum(),
                                                    num_properties.maximum()};
  // The seed for the random generator is the current time.
  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  // Gets all the existing current types inside db for later update cases.
  std::vector<Type> existing_types;
  TF_RETURN_IF_ERROR(GetExistingTypes(fill_types_config_.specification(), store,
                                      existing_types));

  for (int64 i = 0; i < num_operations_; i++) {
    curr_bytes = 0;
    FillTypesWorkItemType put_request;
    const std::string type_name =
        absl::StrCat("type_", absl::FormatTime(absl::Now()), "_", i);
    const int64 num_properties = uniform_dist(gen);
    switch (fill_types_config_.specification()) {
      case FillTypesConfig::ARTIFACT_TYPE: {
        InitializePutRequest<PutArtifactTypeRequest>(fill_types_config_,
                                                     put_request);
        TF_RETURN_IF_ERROR(GenerateType<ArtifactType>(
            fill_types_config_, i, type_name, num_properties, existing_types,
            store,
            *absl::get<PutArtifactTypeRequest>(put_request)
                 .mutable_artifact_type(),
            curr_bytes));
        break;
      }
      case FillTypesConfig::EXECUTION_TYPE: {
        InitializePutRequest<PutExecutionTypeRequest>(fill_types_config_,
                                                      put_request);
        TF_RETURN_IF_ERROR(GenerateType<ExecutionType>(
            fill_types_config_, i, type_name, num_properties, existing_types,
            store,
            *absl::get<PutExecutionTypeRequest>(put_request)
                 .mutable_execution_type(),
            curr_bytes));
        break;
      }
      case FillTypesConfig::CONTEXT_TYPE: {
        InitializePutRequest<PutContextTypeRequest>(fill_types_config_,
                                                    put_request);
        TF_RETURN_IF_ERROR(GenerateType<ContextType>(
            fill_types_config_, i, type_name, num_properties, existing_types,
            store,
            *absl::get<PutContextTypeRequest>(put_request)
                 .mutable_context_type(),
            curr_bytes));
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
