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
#include "ml_metadata/tools/mlmd_bench/read_types_workload.h"

#include <random>
#include <vector>

#include "absl/strings/str_cat.h"
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
constexpr int64 kPropertyTypeSize = 1;

// Gets all types inside db. Returns detailed error if query executions failed.
// Returns FAILED_PRECONDITION if there is no types inside db to read from.
tensorflow::Status GetAndValidateExistingTypes(
    const ReadTypesConfig& read_types_config, MetadataStore& store,
    std::vector<Type>& existing_types) {
  TF_RETURN_IF_ERROR(
      GetExistingTypes(read_types_config, store, existing_types));
  if (existing_types.empty()) {
    return tensorflow::errors::FailedPrecondition(
        "There are no types inside db to read from!");
  }
  return tensorflow::Status::OK();
}

// Calculates the transferred bytes for each types that will be read from later.
// Returns INVALID_ARGUMENT error if any property of current type is UNKNOWN.
template <typename T>
tensorflow::Status GetTransferredBytes(const T& type, int64& curr_bytes) {
  // Includes the id of current type(int64 takes 8 bytes).
  curr_bytes += kInt64IdSize;
  curr_bytes += type.name().size();
  for (const auto& pair : type.properties()) {
    // Includes the bytes for properties' name size.
    curr_bytes += pair.first.size();
    // Includes the bytes for properties' value enumeration size.
    if (pair.second == PropertyType::UNKNOWN) {
      return tensorflow::errors::InvalidArgument("Invalid PropertyType!");
    }
    // As we uses a TINYINT to store the enum.
    curr_bytes += kPropertyTypeSize;
  }
  return tensorflow::Status::OK();
}

// Calculates the transferred bytes for all types inside db that will be read
// from later.
template <typename T>
tensorflow::Status GetTransferredBytesForAllTypes(
    const std::vector<Type>& existing_types, int64& curr_bytes) {
  // Loops over all the types and get its transferred bytes one by one.
  for (const auto& type : existing_types) {
    TF_RETURN_IF_ERROR(GetTransferredBytes(absl::get<T>(type), curr_bytes));
  }
  return tensorflow::Status::OK();
}

// SetUpImpl() for the specifications to read all types in db.
// Returns detailed error if query executions failed.
tensorflow::Status SetUpImplForReadAllTypes(
    const ReadTypesConfig& read_types_config,
    const std::vector<Type>& existing_types, ReadTypesWorkItemType& request,
    int64& curr_bytes) {
  switch (read_types_config.specification()) {
    case ReadTypesConfig::ALL_ARTIFACT_TYPES: {
      request = GetArtifactTypesRequest();
      return GetTransferredBytesForAllTypes<ArtifactType>(existing_types,
                                                          curr_bytes);
    }
    case ReadTypesConfig::ALL_EXECUTION_TYPES: {
      request = GetExecutionTypesRequest();
      return GetTransferredBytesForAllTypes<ExecutionType>(existing_types,
                                                           curr_bytes);
    }
    case ReadTypesConfig::ALL_CONTEXT_TYPES: {
      request = GetContextTypesRequest();
      return GetTransferredBytesForAllTypes<ContextType>(existing_types,
                                                         curr_bytes);
    }
    default:
      return tensorflow::errors::FailedPrecondition(
          "Wrong ReadTypesConfig specification for read all types in db.");
  }
}

// SetUpImpl() for the specifications to read types by a list of ids in db.
// Returns detailed error if query executions failed.
tensorflow::Status SetUpImplForReadTypesByIds(
    const ReadTypesConfig& read_types_config,
    const std::vector<Type>& existing_types,
    std::uniform_int_distribution<int64>& type_index_dist,
    std::minstd_rand0& gen, ReadTypesWorkItemType& request, int64& curr_bytes) {
  // The number of ids per request will be generated w.r.t. the uniform
  // distribution `maybe_num_ids`.
  UniformDistribution num_ids_proto_dist = read_types_config.maybe_num_ids();
  std::uniform_int_distribution<int64> num_ids_dist{
      num_ids_proto_dist.minimum(), num_ids_proto_dist.maximum()};
  const int64 num_ids = num_ids_dist(gen);
  for (int64 i = 0; i < num_ids; ++i) {
    // Selects from existing types uniformly.
    const int64 type_index = type_index_dist(gen);
    switch (read_types_config.specification()) {
      case ReadTypesConfig::ARTIFACT_TYPES_BY_ID: {
        request = GetArtifactTypesByIDRequest();
        absl::get<GetArtifactTypesByIDRequest>(request).add_type_ids(
            absl::get<ArtifactType>(existing_types[type_index]).id());
        TF_RETURN_IF_ERROR(GetTransferredBytes(
            absl::get<ArtifactType>(existing_types[type_index]), curr_bytes));
        break;
      }
      case ReadTypesConfig::EXECUTION_TYPES_BY_ID: {
        request = GetExecutionTypesByIDRequest();
        absl::get<GetExecutionTypesByIDRequest>(request).add_type_ids(
            absl::get<ExecutionType>(existing_types[type_index]).id());
        TF_RETURN_IF_ERROR(GetTransferredBytes(
            absl::get<ExecutionType>(existing_types[type_index]), curr_bytes));
        break;
      }
      case ReadTypesConfig::CONTEXT_TYPES_BY_ID: {
        request = GetContextTypesByIDRequest();
        absl::get<GetContextTypesByIDRequest>(request).add_type_ids(
            absl::get<ContextType>(existing_types[type_index]).id());
        TF_RETURN_IF_ERROR(GetTransferredBytes(
            absl::get<ContextType>(existing_types[type_index]), curr_bytes));
        break;
      }
      default:
        return tensorflow::errors::Unimplemented(
            "Wrong ReadTypesConfig specification for read types by ids in "
            "db.");
    }
  }
  return tensorflow::Status::OK();
}

// SetUpImpl() for the specifications to read type by name in db.
// Returns detailed error if query executions failed.
tensorflow::Status SetUpImplForReadTypeByName(
    const ReadTypesConfig& read_types_config,
    const std::vector<Type>& existing_types,
    std::uniform_int_distribution<int64>& type_index_dist,
    std::minstd_rand0& gen, ReadTypesWorkItemType& request, int64& curr_bytes) {
  // Selects from existing types uniformly.
  const int64 type_index = type_index_dist(gen);
  switch (read_types_config.specification()) {
    case ReadTypesConfig::ARTIFACT_TYPE_BY_NAME: {
      request = GetArtifactTypeRequest();
      absl::get<GetArtifactTypeRequest>(request).set_type_name(
          absl::get<ArtifactType>(existing_types[type_index]).name());
      return GetTransferredBytes(
          absl::get<ArtifactType>(existing_types[type_index]), curr_bytes);
    }
    case ReadTypesConfig::EXECUTION_TYPE_BY_NAME: {
      request = GetExecutionTypeRequest();
      absl::get<GetExecutionTypeRequest>(request).set_type_name(
          absl::get<ExecutionType>(existing_types[type_index]).name());
      return GetTransferredBytes(
          absl::get<ExecutionType>(existing_types[type_index]), curr_bytes);
    }
    case ReadTypesConfig::CONTEXT_TYPE_BY_NAME: {
      request = GetContextTypeRequest();
      absl::get<GetContextTypeRequest>(request).set_type_name(
          absl::get<ContextType>(existing_types[type_index]).name());
      return GetTransferredBytes(
          absl::get<ContextType>(existing_types[type_index]), curr_bytes);
    }
    default:
      return tensorflow::errors::Unimplemented(
          "Wrong ReadTypesConfig specification for read type by name in db.");
  }
}

}  // namespace

ReadTypes::ReadTypes(const ReadTypesConfig& read_types_config,
                     const int64 num_operations)
    : read_types_config_(read_types_config),
      num_operations_(num_operations),
      name_(absl::StrCat("READ_", read_types_config.Specification_Name(
                                      read_types_config.specification()))) {}

tensorflow::Status ReadTypes::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";
  int64 curr_bytes = 0;

  // Gets all the specific types in db to choose from when reading types.
  // If there's no types in the store, returns FAILED_PRECONDITION error.
  std::vector<Type> existing_types;
  TF_RETURN_IF_ERROR(
      GetAndValidateExistingTypes(read_types_config_, *store, existing_types));
  // Uniform distribution to select existing types uniformly.
  std::uniform_int_distribution<int64> type_index_dist{
      0, (int64)(existing_types.size() - 1)};
  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  for (int64 i = 0; i < num_operations_; ++i) {
    curr_bytes = 0;
    ReadTypesWorkItemType read_request;
    switch (read_types_config_.specification()) {
      case ReadTypesConfig::ALL_ARTIFACT_TYPES:
      case ReadTypesConfig::ALL_EXECUTION_TYPES:
      case ReadTypesConfig::ALL_CONTEXT_TYPES:
        TF_RETURN_IF_ERROR(SetUpImplForReadAllTypes(
            read_types_config_, existing_types, read_request, curr_bytes));
        break;
      case ReadTypesConfig::ARTIFACT_TYPES_BY_ID:
      case ReadTypesConfig::EXECUTION_TYPES_BY_ID:
      case ReadTypesConfig::CONTEXT_TYPES_BY_ID:
        TF_RETURN_IF_ERROR(SetUpImplForReadTypesByIds(
            read_types_config_, existing_types, type_index_dist, gen,
            read_request, curr_bytes));
        break;
      case ReadTypesConfig::ARTIFACT_TYPE_BY_NAME:
      case ReadTypesConfig::EXECUTION_TYPE_BY_NAME:
      case ReadTypesConfig::CONTEXT_TYPE_BY_NAME:
        TF_RETURN_IF_ERROR(SetUpImplForReadTypeByName(
            read_types_config_, existing_types, type_index_dist, gen,
            read_request, curr_bytes));
        break;
      default:
        LOG(FATAL) << "Wrong specification for ReadTypes!";
    }
    work_items_.emplace_back(read_request, curr_bytes);
  }
  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status ReadTypes::RunOpImpl(const int64 work_items_index,
                                        MetadataStore* store) {
  switch (read_types_config_.specification()) {
    case ReadTypesConfig::ALL_ARTIFACT_TYPES: {
      auto request = absl::get<GetArtifactTypesRequest>(
          work_items_[work_items_index].first);
      GetArtifactTypesResponse response;
      return store->GetArtifactTypes(request, &response);
    }
    case ReadTypesConfig::ALL_EXECUTION_TYPES: {
      auto request = absl::get<GetExecutionTypesRequest>(
          work_items_[work_items_index].first);
      GetExecutionTypesResponse response;
      return store->GetExecutionTypes(request, &response);
    }
    case ReadTypesConfig::ALL_CONTEXT_TYPES: {
      auto request = absl::get<GetContextTypesRequest>(
          work_items_[work_items_index].first);
      GetContextTypesResponse response;
      return store->GetContextTypes(request, &response);
    }
    case ReadTypesConfig::ARTIFACT_TYPES_BY_ID: {
      auto request = absl::get<GetArtifactTypesByIDRequest>(
          work_items_[work_items_index].first);
      GetArtifactTypesByIDResponse response;
      return store->GetArtifactTypesByID(request, &response);
    }
    case ReadTypesConfig::EXECUTION_TYPES_BY_ID: {
      auto request = absl::get<GetExecutionTypesByIDRequest>(
          work_items_[work_items_index].first);
      GetExecutionTypesByIDResponse response;
      return store->GetExecutionTypesByID(request, &response);
    }
    case ReadTypesConfig::CONTEXT_TYPES_BY_ID: {
      auto request = absl::get<GetContextTypesByIDRequest>(
          work_items_[work_items_index].first);
      GetContextTypesByIDResponse response;
      return store->GetContextTypesByID(request, &response);
    }
    case ReadTypesConfig::ARTIFACT_TYPE_BY_NAME: {
      auto request = absl::get<GetArtifactTypeRequest>(
          work_items_[work_items_index].first);
      GetArtifactTypeResponse response;
      return store->GetArtifactType(request, &response);
    }
    case ReadTypesConfig::EXECUTION_TYPE_BY_NAME: {
      auto request = absl::get<GetExecutionTypeRequest>(
          work_items_[work_items_index].first);
      GetExecutionTypeResponse response;
      return store->GetExecutionType(request, &response);
    }
    case ReadTypesConfig::CONTEXT_TYPE_BY_NAME: {
      auto request =
          absl::get<GetContextTypeRequest>(work_items_[work_items_index].first);
      GetContextTypeResponse response;
      return store->GetContextType(request, &response);
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
}

tensorflow::Status ReadTypes::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

std::string ReadTypes::GetName() { return name_; }

}  // namespace ml_metadata
