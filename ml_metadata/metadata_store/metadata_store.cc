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

#include <algorithm>
#include <iterator>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/repeated_field.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "ml_metadata/metadata_store/metadata_access_object_factory.h"
#include "ml_metadata/metadata_store/simple_types_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/simple_types/proto/simple_types.pb.h"
#include "ml_metadata/simple_types/simple_types_constants.h"
#include "ml_metadata/util/return_utils.h"
#include "ml_metadata/util/status_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {
namespace {
using std::unique_ptr;

// Checks if the `stored_type` and `other_type` have the same names.
// In addition, it checks whether the types are inconsistent:
// a) `stored_type` and `other_type` have conflicting property value type
// b) `can_omit_fields` is false, while `stored_type`/`other_type` is not empty.
// c) `can_add_fields` is false, while `other_type`/`store_type` is not empty.
// Returns OK if the types are consistent and an output_type that contains the
// union of the properties in stored_type and other_type.
// Returns FAILED_PRECONDITION, if the types are inconsistent.
template <typename T>
tensorflow::Status CheckFieldsConsistent(const T& stored_type,
                                         const T& other_type,
                                         bool can_add_fields,
                                         bool can_omit_fields, T& output_type) {
  if (stored_type.name() != other_type.name()) {
    return tensorflow::errors::FailedPrecondition(
        "Conflicting type name found in stored and given types: "
        "stored type: ",
        stored_type.DebugString(), "; given type: ", other_type.DebugString());
  }
  // Make sure every property in stored_type matches with the one in other_type
  // unless can_omit_fields is set to true.
  int omitted_fields_count = 0;
  for (const auto& pair : stored_type.properties()) {
    const std::string& key = pair.first;
    const PropertyType value = pair.second;
    const auto other_iter = other_type.properties().find(key);
    if (other_iter == other_type.properties().end()) {
      omitted_fields_count++;
    } else if (other_iter->second != value) {
      return tensorflow::errors::FailedPrecondition(
          "Conflicting property value type found in stored and given types: "
          "stored_type: ",
          stored_type.DebugString(),
          "; other_type: ", other_type.DebugString());
    }
    if (omitted_fields_count > 0 && !can_omit_fields) {
      return tensorflow::errors::FailedPrecondition(
          "can_omit_fields is false while stored type has more properties: "
          "stored type: ",
          stored_type.DebugString(),
          "; given type: ", other_type.DebugString());
    }
  }
  if (stored_type.properties_size() - omitted_fields_count ==
      other_type.properties_size()) {
    output_type = stored_type;
    return tensorflow::Status::OK();
  }
  if (!can_add_fields) {
    return tensorflow::errors::FailedPrecondition(
        "can_add_fields is false while the given type has more properties: "
        "stored_type: ",
        stored_type.DebugString(), "; other_type: ", other_type.DebugString());
  }
  // add new properties to output_types if can_add_fields is true.
  output_type = stored_type;
  for (const auto& pair : other_type.properties()) {
    const std::string& property_name = pair.first;
    const PropertyType value = pair.second;
    if (stored_type.properties().find(property_name) ==
        stored_type.properties().end()) {
      (*output_type.mutable_properties())[property_name] = value;
    }
  }
  return tensorflow::Status::OK();
}

// If there is no type having the same name and version, then inserts a new
// type. If a type with the same name and version already exists
// (let's call it `old_type`), it checks the consistency of `type` and
// `old_type` as described in CheckFieldsConsistent according to
// can_add_fields and can_omit_fields.
// It returns ALREADY_EXISTS if:
//  a) any property in `type` has different value from the one in `old_type`
//  b) can_add_fields = false, `type` has more properties than `old_type`
//  c) can_omit_fields = false, `type` has less properties than `old_type`
// If `type` is a valid update, then new fields in `type` are added.
// Returns INVALID_ARGUMENT error, if name field in `type` is not given.
// Returns INVALID_ARGUMENT error, if any property type in `type` is unknown.
// Returns detailed INTERNAL error, if query execution fails.
template <typename T>
absl::Status UpsertType(const T& type, bool can_add_fields,
                        bool can_omit_fields,
                        MetadataAccessObject* metadata_access_object,
                        int64* type_id) {
  T stored_type;
  const absl::Status status = metadata_access_object->FindTypeByNameAndVersion(
      type.name(), type.version(), &stored_type);
  if (!status.ok() && !absl::IsNotFound(status)) {
    return status;
  }
  // if not found, then it creates a type. `can_add_fields` is ignored.
  if (absl::IsNotFound(status)) {
    return metadata_access_object->CreateType(type, type_id);
  }
  // otherwise it updates the type.
  *type_id = stored_type.id();
  // all properties in stored_type must match the given type.
  // if `can_add_fields` is set, then new properties can be added
  // if `can_omit_fields` is set, then existing properties can be missing.
  T output_type;
  const tensorflow::Status check_status = CheckFieldsConsistent(
      stored_type, type, can_add_fields, can_omit_fields, output_type);
  if (!check_status.ok()) {
    return absl::AlreadyExistsError(
        absl::StrCat("Type already exists with different properties: ",
                     check_status.error_message()));
  }
  return metadata_access_object->UpdateType(output_type);
}

// Inserts or updates all the types in the argument list. 'can_add_fields' and
// 'can_omit_fields' are both enabled. Type ids are inserted into the
// PutTypesResponse 'response'.
absl::Status UpsertTypes(
    const google::protobuf::RepeatedPtrField<ArtifactType>& artifact_types,
    const google::protobuf::RepeatedPtrField<ExecutionType>& execution_types,
    const google::protobuf::RepeatedPtrField<ContextType>& context_types,
    const bool can_add_fields, const bool can_omit_fields,
    MetadataAccessObject* metadata_access_object, PutTypesResponse* response) {
  for (const ArtifactType& artifact_type : artifact_types) {
    int64 artifact_type_id;
    MLMD_RETURN_IF_ERROR(UpsertType(artifact_type, can_add_fields,
                                    can_omit_fields, metadata_access_object,
                                    &artifact_type_id));
    response->add_artifact_type_ids(artifact_type_id);
  }
  for (const ExecutionType& execution_type : execution_types) {
    int64 execution_type_id;
    MLMD_RETURN_IF_ERROR(UpsertType(execution_type, can_add_fields,
                                    can_omit_fields, metadata_access_object,
                                    &execution_type_id));
    response->add_execution_type_ids(execution_type_id);
  }
  for (const ContextType& context_type : context_types) {
    int64 context_type_id;
    MLMD_RETURN_IF_ERROR(UpsertType(context_type, can_add_fields,
                                    can_omit_fields, metadata_access_object,
                                    &context_type_id));
    response->add_context_type_ids(context_type_id);
  }
  return absl::OkStatus();
}

// Loads SimpleTypes proto from string and updates or inserts it into database.
absl::Status UpsertSimpleTypes(MetadataAccessObject* metadata_access_object) {
  SimpleTypes simple_types;
  PutTypesResponse response;
  MLMD_RETURN_IF_ERROR(LoadSimpleTypes(simple_types));
  return UpsertTypes(
      simple_types.artifact_types(), simple_types.execution_types(),
      simple_types.context_types(), /*can_add_fields=*/true,
      /*can_omit_fields=*/true, metadata_access_object, &response);
}

// Updates or inserts an artifact. If the artifact.id is given, it updates the
// stored artifact, otherwise, it creates a new artifact.
absl::Status UpsertArtifact(const Artifact& artifact,
                            MetadataAccessObject* metadata_access_object,
                            int64* artifact_id) {
  CHECK(artifact_id) << "artifact_id should not be null";
  if (artifact.has_id()) {
    MLMD_RETURN_IF_ERROR(metadata_access_object->UpdateArtifact(artifact));
    *artifact_id = artifact.id();
  } else {
    MLMD_RETURN_IF_ERROR(
        metadata_access_object->CreateArtifact(artifact, artifact_id));
  }
  return absl::OkStatus();
}

// Updates or inserts an execution. If the execution.id is given, it updates the
// stored execution, otherwise, it creates a new execution.
absl::Status UpsertExecution(const Execution& execution,
                             MetadataAccessObject* metadata_access_object,
                             int64* execution_id) {
  CHECK(execution_id) << "execution_id should not be null";
  if (execution.has_id()) {
    MLMD_RETURN_IF_ERROR(metadata_access_object->UpdateExecution(execution));
    *execution_id = execution.id();
  } else {
    MLMD_RETURN_IF_ERROR(
        metadata_access_object->CreateExecution(execution, execution_id));
  }
  return absl::OkStatus();
}

// Updates or inserts a context. If the context.id is given, it updates the
// stored context, otherwise, it creates a new context.
absl::Status UpsertContext(const Context& context,
                           MetadataAccessObject* metadata_access_object,
                           int64* context_id) {
  CHECK(context_id) << "context_id should not be null";
  if (context.has_id()) {
    MLMD_RETURN_IF_ERROR(metadata_access_object->UpdateContext(context));
    *context_id = context.id();
  } else {
    MLMD_RETURN_IF_ERROR(
        metadata_access_object->CreateContext(context, context_id));
  }
  return absl::OkStatus();
}

// Inserts an association. If the association already exists it returns OK.
absl::Status InsertAssociationIfNotExist(
    int64 context_id, int64 execution_id,
    MetadataAccessObject* metadata_access_object) {
  Association association;
  association.set_execution_id(execution_id);
  association.set_context_id(context_id);
  int64 dummy_assocation_id;
  absl::Status status = metadata_access_object->CreateAssociation(
      association, &dummy_assocation_id);
  if (!status.ok() && !absl::IsAlreadyExists(status)) {
    return status;
  }
  return absl::OkStatus();
}

// Inserts an attribution. If the attribution already exists it returns OK.
absl::Status InsertAttributionIfNotExist(
    int64 context_id, int64 artifact_id,
    MetadataAccessObject* metadata_access_object) {
  Attribution attribution;
  attribution.set_artifact_id(artifact_id);
  attribution.set_context_id(context_id);
  int64 dummy_attribution_id;
  absl::Status status = metadata_access_object->CreateAttribution(
      attribution, &dummy_attribution_id);
  if (!status.ok() && !absl::IsAlreadyExists(status)) {
    return status;
  }
  return absl::OkStatus();
}

// Updates or inserts a pair of {Artifact, Event}. If artifact is not given,
// the event.artifact_id must exist, and it inserts the event, and returns the
// artifact_id. Otherwise if artifact is given, event.artifact_id is optional,
// if set, then artifact.id and event.artifact_id must align.
absl::Status UpsertArtifactAndEvent(
    const PutExecutionRequest::ArtifactAndEvent& artifact_and_event,
    MetadataAccessObject* metadata_access_object, int64* artifact_id) {
  CHECK(artifact_id) << "The output artifact_id pointer should not be null";
  if (!artifact_and_event.has_artifact() && !artifact_and_event.has_event()) {
    return absl::OkStatus();
  }
  // validate event and artifact's id aligns.
  // if artifact is not given, the event.artifact_id must exist
  absl::optional<int64> maybe_event_artifact_id =
      artifact_and_event.has_event() &&
              artifact_and_event.event().has_artifact_id()
          ? absl::make_optional<int64>(artifact_and_event.event().artifact_id())
          : absl::nullopt;
  if (!artifact_and_event.has_artifact() && !maybe_event_artifact_id) {
    return absl::InvalidArgumentError(absl::StrCat(
        "If no artifact is present, given event must have an artifact_id: ",
        artifact_and_event.DebugString()));
  }
  // if artifact and event.artifact_id is given, then artifact.id and
  // event.artifact_id must align.
  absl::optional<int64> maybe_artifact_id =
      artifact_and_event.has_artifact() &&
              artifact_and_event.artifact().has_id()
          ? absl::make_optional<int64>(artifact_and_event.artifact().id())
          : absl::nullopt;
  if (artifact_and_event.has_artifact() && maybe_event_artifact_id &&
      maybe_artifact_id != maybe_event_artifact_id) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Given event.artifact_id is not aligned with the artifact: ",
        artifact_and_event.DebugString()));
  }
  // upsert artifact if present.
  if (artifact_and_event.has_artifact()) {
    MLMD_RETURN_IF_ERROR(UpsertArtifact(artifact_and_event.artifact(),
                                        metadata_access_object, artifact_id));
  }
  // insert event if any.
  if (!artifact_and_event.has_event()) {
    return absl::OkStatus();
  }
  Event event = artifact_and_event.event();
  if (artifact_and_event.has_artifact()) {
    event.set_artifact_id(*artifact_id);
  } else {
    *artifact_id = event.artifact_id();
  }
  int64 dummy_event_id = -1;
  return metadata_access_object->CreateEvent(event, &dummy_event_id);
}

// A util to handle type_version in type read/write API requests.
template <typename T>
absl::optional<std::string> GetRequestTypeVersion(const T& type_request) {
  return type_request.has_type_version() && !type_request.type_version().empty()
             ? absl::make_optional(type_request.type_version())
             : absl::nullopt;
}

}  // namespace

tensorflow::Status MetadataStore::InitMetadataStore() {
  TF_RETURN_IF_ERROR(
      FromABSLStatus(transaction_executor_->Execute([this]() -> absl::Status {
        return metadata_access_object_->InitMetadataSource();
      })));
  return FromABSLStatus(
      transaction_executor_->Execute([this]() -> absl::Status {
        return UpsertSimpleTypes(metadata_access_object_.get());
      }));
}

// TODO(b/187357155): duplicated results when inserting simple types
// concurrently
tensorflow::Status MetadataStore::InitMetadataStoreIfNotExists(
    const bool enable_upgrade_migration) {
  TF_RETURN_IF_ERROR(FromABSLStatus(transaction_executor_->Execute(
      [this, &enable_upgrade_migration]() -> absl::Status {
        return metadata_access_object_->InitMetadataSourceIfNotExists(
            enable_upgrade_migration);
      })));
  return FromABSLStatus(
      transaction_executor_->Execute([this]() -> absl::Status {
        return UpsertSimpleTypes(metadata_access_object_.get());
      }));
}

tensorflow::Status MetadataStore::PutTypes(const PutTypesRequest& request,
                                           PutTypesResponse* response) {
  if (!request.all_fields_match()) {
    return tensorflow::errors::Unimplemented("Must match all fields.");
  }
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        return UpsertTypes(request.artifact_types(), request.execution_types(),
                           request.context_types(), request.can_add_fields(),
                           request.can_omit_fields(),
                           metadata_access_object_.get(), response);
      }));
}

tensorflow::Status MetadataStore::PutArtifactType(
    const PutArtifactTypeRequest& request, PutArtifactTypeResponse* response) {
  if (!request.all_fields_match()) {
    return tensorflow::errors::Unimplemented("Must match all fields.");
  }
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        int64 type_id;
        MLMD_RETURN_IF_ERROR(
            UpsertType(request.artifact_type(), request.can_add_fields(),
                       request.can_omit_fields(), metadata_access_object_.get(),
                       &type_id));
        response->set_type_id(type_id);
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::PutExecutionType(
    const PutExecutionTypeRequest& request,
    PutExecutionTypeResponse* response) {
  if (!request.all_fields_match()) {
    return tensorflow::errors::Unimplemented("Must match all fields.");
  }
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        int64 type_id;
        MLMD_RETURN_IF_ERROR(
            UpsertType(request.execution_type(), request.can_add_fields(),
                       request.can_omit_fields(), metadata_access_object_.get(),
                       &type_id));
        response->set_type_id(type_id);
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::PutContextType(
    const PutContextTypeRequest& request, PutContextTypeResponse* response) {
  if (!request.all_fields_match()) {
    return tensorflow::errors::Unimplemented("Must match all fields.");
  }
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        int64 type_id;
        MLMD_RETURN_IF_ERROR(
            UpsertType(request.context_type(), request.can_add_fields(),
                       request.can_omit_fields(), metadata_access_object_.get(),
                       &type_id));
        response->set_type_id(type_id);
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetArtifactType(
    const GetArtifactTypeRequest& request, GetArtifactTypeResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        return metadata_access_object_->FindTypeByNameAndVersion(
            request.type_name(), GetRequestTypeVersion(request),
            response->mutable_artifact_type());
      }));
}

tensorflow::Status MetadataStore::GetExecutionType(
    const GetExecutionTypeRequest& request,
    GetExecutionTypeResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        return metadata_access_object_->FindTypeByNameAndVersion(
            request.type_name(), GetRequestTypeVersion(request),
            response->mutable_execution_type());
      }));
}

tensorflow::Status MetadataStore::GetContextType(
    const GetContextTypeRequest& request, GetContextTypeResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        return metadata_access_object_->FindTypeByNameAndVersion(
            request.type_name(), GetRequestTypeVersion(request),
            response->mutable_context_type());
      }));
}

tensorflow::Status MetadataStore::GetArtifactTypesByID(
    const GetArtifactTypesByIDRequest& request,
    GetArtifactTypesByIDResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const int64 type_id : request.type_ids()) {
          ArtifactType artifact_type;
          const absl::Status status =
              metadata_access_object_->FindTypeById(type_id, &artifact_type);
          if (status.ok()) {
            *response->mutable_artifact_types()->Add() = artifact_type;
          } else if (!absl::IsNotFound(status)) {
            return status;
          }
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetExecutionTypesByID(
    const GetExecutionTypesByIDRequest& request,
    GetExecutionTypesByIDResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const int64 type_id : request.type_ids()) {
          ExecutionType execution_type;
          const absl::Status status =
              metadata_access_object_->FindTypeById(type_id, &execution_type);
          if (status.ok()) {
            *response->mutable_execution_types()->Add() = execution_type;
          } else if (!absl::IsNotFound(status)) {
            return status;
          }
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetContextTypesByID(
    const GetContextTypesByIDRequest& request,
    GetContextTypesByIDResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const int64 type_id : request.type_ids()) {
          ContextType context_type;
          const absl::Status status =
              metadata_access_object_->FindTypeById(type_id, &context_type);
          if (status.ok()) {
            *response->mutable_context_types()->Add() = context_type;
          } else if (!absl::IsNotFound(status)) {
            return status;
          }
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetArtifactsByID(
    const GetArtifactsByIDRequest& request,
    GetArtifactsByIDResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Artifact> artifacts;
        const std::vector<int64> ids(request.artifact_ids().begin(),
                                     request.artifact_ids().end());
        const absl::Status status =
            metadata_access_object_->FindArtifactsById(ids, &artifacts);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        absl::c_copy(artifacts, google::protobuf::RepeatedPtrFieldBackInserter(
                                    response->mutable_artifacts()));
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetExecutionsByID(
    const GetExecutionsByIDRequest& request,
    GetExecutionsByIDResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Execution> executions;
        const std::vector<int64> ids(request.execution_ids().begin(),
                                     request.execution_ids().end());
        const absl::Status status =
            metadata_access_object_->FindExecutionsById(ids, &executions);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        absl::c_copy(executions, google::protobuf::RepeatedPtrFieldBackInserter(
                                     response->mutable_executions()));
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetContextsByID(
    const GetContextsByIDRequest& request, GetContextsByIDResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> contexts;
        const std::vector<int64> ids(request.context_ids().begin(),
                                     request.context_ids().end());
        const absl::Status status =
            metadata_access_object_->FindContextsById(ids, &contexts);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        absl::c_copy(contexts, google::protobuf::RepeatedFieldBackInserter(
                                   response->mutable_contexts()));
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::PutArtifacts(
    const PutArtifactsRequest& request, PutArtifactsResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute([this, &request,
                                                        &response]()
                                                           -> absl::Status {
    response->Clear();
    for (const Artifact& artifact : request.artifacts()) {
      int64 artifact_id = -1;
      // Verify the latest_updated_time before upserting the artifact.
      if (artifact.has_id() &&
          request.options().abort_if_latest_updated_time_changed()) {
        Artifact existing_artifact;
        absl::Status status;
        {
          std::vector<Artifact> artifacts;
          status = metadata_access_object_->FindArtifactsById({artifact.id()},
                                                              &artifacts);
          if (status.ok()) {
            existing_artifact = artifacts.at(0);
          }
        }
        if (!absl::IsNotFound(status)) {
          MLMD_RETURN_IF_ERROR(status);
          if (artifact.last_update_time_since_epoch() !=
              existing_artifact.last_update_time_since_epoch()) {
            return absl::FailedPreconditionError(absl::StrCat(
                "`abort_if_latest_updated_time_changed` is set, and the stored "
                "artifact with id = ",
                artifact.id(),
                " has a different last_update_time_since_epoch: ",
                existing_artifact.last_update_time_since_epoch(),
                " from the one in the given artifact: ",
                artifact.last_update_time_since_epoch()));
          }
          // If set the option and all check succeeds, we make sure the
          // timestamp after the update increases.
          absl::SleepFor(absl::Milliseconds(1));
        }
      }
      MLMD_RETURN_IF_ERROR(UpsertArtifact(
          artifact, metadata_access_object_.get(), &artifact_id));
      response->add_artifact_ids(artifact_id);
    }
    return absl::OkStatus();
  }));
}

tensorflow::Status MetadataStore::PutExecutions(
    const PutExecutionsRequest& request, PutExecutionsResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const Execution& execution : request.executions()) {
          int64 execution_id = -1;
          MLMD_RETURN_IF_ERROR(UpsertExecution(
              execution, metadata_access_object_.get(), &execution_id));
          response->add_execution_ids(execution_id);
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::PutContexts(const PutContextsRequest& request,
                                              PutContextsResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const Context& context : request.contexts()) {
          int64 context_id = -1;
          MLMD_RETURN_IF_ERROR(UpsertContext(
              context, metadata_access_object_.get(), &context_id));
          response->add_context_ids(context_id);
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::Create(
    const MetadataSourceQueryConfig& query_config,
    const MigrationOptions& migration_options,
    unique_ptr<MetadataSource> metadata_source,
    unique_ptr<TransactionExecutor> transaction_executor,
    unique_ptr<MetadataStore>* result) {
  unique_ptr<MetadataAccessObject> metadata_access_object;
  TF_RETURN_IF_ERROR(FromABSLStatus(CreateMetadataAccessObject(
      query_config, metadata_source.get(), &metadata_access_object)));
  // if downgrade migration is specified
  if (migration_options.downgrade_to_schema_version() >= 0) {
    TF_RETURN_IF_ERROR(FromABSLStatus(transaction_executor->Execute(
        [&migration_options, &metadata_access_object]() -> absl::Status {
          return metadata_access_object->DowngradeMetadataSource(
              migration_options.downgrade_to_schema_version());
        })));
    return tensorflow::errors::Cancelled(
        "Downgrade migration was performed. Connection to the downgraded "
        "database is Cancelled. Now the database is at schema version ",
        migration_options.downgrade_to_schema_version(),
        ". Please refer to the migration guide and use lower version of the "
        "library to connect to the metadata store.");
  }
  *result = absl::WrapUnique(new MetadataStore(
      std::move(metadata_source), std::move(metadata_access_object),
      std::move(transaction_executor)));
  return tensorflow::Status::OK();
}

tensorflow::Status MetadataStore::PutEvents(const PutEventsRequest& request,
                                            PutEventsResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const Event& event : request.events()) {
          int64 dummy_event_id = -1;
          MLMD_RETURN_IF_ERROR(
              metadata_access_object_->CreateEvent(event, &dummy_event_id));
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::PutExecution(
    const PutExecutionRequest& request, PutExecutionResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute([this, &request,
                                                        &response]()
                                                           -> absl::Status {
    response->Clear();
    if (!request.has_execution()) {
      return absl::InvalidArgumentError(
          absl::StrCat("No execution is found: ", request.DebugString()));
    }
    // 1. Upsert Execution
    const Execution& execution = request.execution();
    int64 execution_id = -1;
    MLMD_RETURN_IF_ERROR(UpsertExecution(
        execution, metadata_access_object_.get(), &execution_id));
    response->set_execution_id(execution_id);
    // 2. Upsert Artifacts and insert events
    for (PutExecutionRequest::ArtifactAndEvent artifact_and_event :
         request.artifact_event_pairs()) {
      // validate execution and event if given
      if (artifact_and_event.has_event()) {
        Event* event = artifact_and_event.mutable_event();
        if (event->has_execution_id() &&
            (!execution.has_id() || execution.id() != event->execution_id())) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Request's event.execution_id does not match with the given "
              "execution: ",
              request.DebugString()));
        }
        event->set_execution_id(execution_id);
      }
      int64 artifact_id = -1;
      MLMD_RETURN_IF_ERROR(UpsertArtifactAndEvent(
          artifact_and_event, metadata_access_object_.get(), &artifact_id));
      response->add_artifact_ids(artifact_id);
    }
    // 3. Upsert contexts and insert associations and attributions.
    for (const Context& context : request.contexts()) {
      int64 context_id = -1;
      // Try to reuse existing context if the options is set.
      if (request.options().reuse_context_if_already_exist() &&
          !context.has_id()) {
        Context existing_context;
        const absl::Status status =
            metadata_access_object_->FindContextByTypeIdAndContextName(
                context.type_id(), context.name(), &existing_context);
        if (!absl::IsNotFound(status)) {
          MLMD_RETURN_IF_ERROR(status);
          context_id = existing_context.id();
        }
      }
      if (context_id == -1) {
        const absl::Status status =
            UpsertContext(context, metadata_access_object_.get(), &context_id);
        // When `reuse_context_if_already_exist`, there are concurrent timelines
        // to create the same new context. If use the option, let client side
        // to retry the failed transaction safely.
        if (request.options().reuse_context_if_already_exist() &&
            absl::IsAlreadyExists(status)) {
          return absl::AbortedError(absl::StrCat(
              "Concurrent creation of the same context at the first time. "
              "Retry the transaction to reuse the context: ",
              context.DebugString()));
        }
        MLMD_RETURN_IF_ERROR(status);
      }
      response->add_context_ids(context_id);
      MLMD_RETURN_IF_ERROR(InsertAssociationIfNotExist(
          context_id, response->execution_id(), metadata_access_object_.get()));
      for (const int64 artifact_id : response->artifact_ids()) {
        MLMD_RETURN_IF_ERROR(InsertAttributionIfNotExist(
            context_id, artifact_id, metadata_access_object_.get()));
      }
    }
    return absl::OkStatus();
  }));
}

tensorflow::Status MetadataStore::GetEventsByExecutionIDs(
    const GetEventsByExecutionIDsRequest& request,
    GetEventsByExecutionIDsResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Event> events;
        const absl::Status status =
            metadata_access_object_->FindEventsByExecutions(
                std::vector<int64>(request.execution_ids().begin(),
                                   request.execution_ids().end()),
                &events);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const Event& event : events) {
          *response->mutable_events()->Add() = event;
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetEventsByArtifactIDs(
    const GetEventsByArtifactIDsRequest& request,
    GetEventsByArtifactIDsResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Event> events;
        const absl::Status status =
            metadata_access_object_->FindEventsByArtifacts(
                std::vector<int64>(request.artifact_ids().begin(),
                                   request.artifact_ids().end()),
                &events);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const Event& event : events) {
          *response->mutable_events()->Add() = event;
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetExecutions(
    const GetExecutionsRequest& request, GetExecutionsResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Execution> executions;
        absl::Status status;
        std::string next_page_token;
        if (request.has_options()) {
          status = metadata_access_object_->ListExecutions(
              request.options(), &executions, &next_page_token);
        } else {
          status = metadata_access_object_->FindExecutions(&executions);
        }

        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }

        for (const Execution& execution : executions) {
          *response->mutable_executions()->Add() = execution;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetArtifacts(
    const GetArtifactsRequest& request, GetArtifactsResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Artifact> artifacts;
        absl::Status status;
        std::string next_page_token;
        if (request.has_options()) {
          status = metadata_access_object_->ListArtifacts(
              request.options(), &artifacts, &next_page_token);
        } else {
          status = metadata_access_object_->FindArtifacts(&artifacts);
        }

        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }

        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }

        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetContexts(const GetContextsRequest& request,
                                              GetContextsResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> contexts;
        absl::Status status;
        std::string next_page_token;
        if (request.has_options()) {
          status = metadata_access_object_->ListContexts(
              request.options(), &contexts, &next_page_token);
        } else {
          status = metadata_access_object_->FindContexts(&contexts);
        }

        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }

        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }

        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetArtifactTypes(
    const GetArtifactTypesRequest& request,
    GetArtifactTypesResponse* response) {
  return FromABSLStatus(
      transaction_executor_->Execute([this, &response]() -> absl::Status {
        response->Clear();
        std::vector<ArtifactType> artifact_types;
        const absl::Status status =
            metadata_access_object_->FindTypes(&artifact_types);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const ArtifactType& artifact_type : artifact_types) {
          // Simple types will not be returned by Get*Types APIs because they
          // are invisible to users.
          const bool is_simple_type =
              std::find(kSimpleTypeNames.begin(), kSimpleTypeNames.end(),
                        artifact_type.name()) != kSimpleTypeNames.end();
          if (!is_simple_type) {
            *response->mutable_artifact_types()->Add() = artifact_type;
          }
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetExecutionTypes(
    const GetExecutionTypesRequest& request,
    GetExecutionTypesResponse* response) {
  return FromABSLStatus(
      transaction_executor_->Execute([this, &response]() -> absl::Status {
        response->Clear();
        std::vector<ExecutionType> execution_types;
        const absl::Status status =
            metadata_access_object_->FindTypes(&execution_types);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const ExecutionType& execution_type : execution_types) {
          // Simple types will not be returned by Get*Types APIs because they
          // are invisible to users.
          const bool is_simple_type =
              std::find(kSimpleTypeNames.begin(), kSimpleTypeNames.end(),
                        execution_type.name()) != kSimpleTypeNames.end();
          if (!is_simple_type) {
            *response->mutable_execution_types()->Add() = execution_type;
          }
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetContextTypes(
    const GetContextTypesRequest& request, GetContextTypesResponse* response) {
  return FromABSLStatus(
      transaction_executor_->Execute([this, &response]() -> absl::Status {
        response->Clear();
        std::vector<ContextType> context_types;
        const absl::Status status =
            metadata_access_object_->FindTypes(&context_types);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const ContextType& context_type : context_types) {
          *response->mutable_context_types()->Add() = context_type;
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetArtifactsByURI(
    const GetArtifactsByURIRequest& request,
    GetArtifactsByURIResponse* response) {
  // Validate if there's already deprecated optional string uri = 1 field.
  const google::protobuf::UnknownFieldSet& unknown_field_set =
      request.GetReflection()->GetUnknownFields(request);
  for (int i = 0; i < unknown_field_set.field_count(); i++) {
    const google::protobuf::UnknownField& unknown_field = unknown_field_set.field(i);
    if (unknown_field.number() == 1) {
      return tensorflow::errors::InvalidArgument(
          "The request contains deprecated field `uri`. Please upgrade the "
          "client library version above 0.21.0. GetArtifactsByURIRequest: ",
          request.DebugString());
    }
  }
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        absl::flat_hash_set<std::string> uris(request.uris().begin(),
                                              request.uris().end());
        for (const std::string& uri : uris) {
          std::vector<Artifact> artifacts;
          const absl::Status status =
              metadata_access_object_->FindArtifactsByURI(uri, &artifacts);
          if (!status.ok() && !absl::IsNotFound(status)) {
            // If any none NotFound error returned, we do early stopping as
            // the query execution has internal db errors.
            return status;
          }
          for (const Artifact& artifact : artifacts) {
            *response->mutable_artifacts()->Add() = artifact;
          }
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetArtifactsByType(
    const GetArtifactsByTypeRequest& request,
    GetArtifactsByTypeResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        ArtifactType artifact_type;
        absl::Status status = metadata_access_object_->FindTypeByNameAndVersion(
            request.type_name(), GetRequestTypeVersion(request),
            &artifact_type);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        std::vector<Artifact> artifacts;
        status = metadata_access_object_->FindArtifactsByTypeId(
            artifact_type.id(), &artifacts);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetArtifactByTypeAndName(
    const GetArtifactByTypeAndNameRequest& request,
    GetArtifactByTypeAndNameResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        ArtifactType artifact_type;
        absl::Status status = metadata_access_object_->FindTypeByNameAndVersion(
            request.type_name(), GetRequestTypeVersion(request),
            &artifact_type);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        Artifact artifact;
        status = metadata_access_object_->FindArtifactByTypeIdAndArtifactName(
            artifact_type.id(), request.artifact_name(), &artifact);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        *response->mutable_artifact() = artifact;
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetExecutionsByType(
    const GetExecutionsByTypeRequest& request,
    GetExecutionsByTypeResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        ExecutionType execution_type;
        absl::Status status = metadata_access_object_->FindTypeByNameAndVersion(
            request.type_name(), GetRequestTypeVersion(request),
            &execution_type);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        std::vector<Execution> executions;
        status = metadata_access_object_->FindExecutionsByTypeId(
            execution_type.id(), &executions);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const Execution& execution : executions) {
          *response->mutable_executions()->Add() = execution;
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetExecutionByTypeAndName(
    const GetExecutionByTypeAndNameRequest& request,
    GetExecutionByTypeAndNameResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        ExecutionType execution_type;
        absl::Status status = metadata_access_object_->FindTypeByNameAndVersion(
            request.type_name(), GetRequestTypeVersion(request),
            &execution_type);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        Execution execution;
        status = metadata_access_object_->FindExecutionByTypeIdAndExecutionName(
            execution_type.id(), request.execution_name(), &execution);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        *response->mutable_execution() = execution;
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetContextsByType(
    const GetContextsByTypeRequest& request,
    GetContextsByTypeResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        ContextType context_type;
        {
          absl::Status status =
              metadata_access_object_->FindTypeByNameAndVersion(
                  request.type_name(), GetRequestTypeVersion(request),
                  &context_type);
          if (absl::IsNotFound(status)) {
            return absl::OkStatus();
          } else if (!status.ok()) {
            return status;
          }
        }
        std::vector<Context> contexts;
        std::string next_page_token;
        {
          absl::Status status;
          status = metadata_access_object_->FindContextsByTypeId(
              context_type.id(),
              (request.has_options() ? absl::make_optional(request.options())
                                     : absl::nullopt),
              &contexts, &next_page_token);
          if (absl::IsNotFound(status)) {
            return absl::OkStatus();
          } else if (!status.ok()) {
            return status;
          }
        }
        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }
        if (request.has_options()) {
          response->set_next_page_token(next_page_token);
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetContextByTypeAndName(
    const GetContextByTypeAndNameRequest& request,
    GetContextByTypeAndNameResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        ContextType context_type;
        absl::Status status = metadata_access_object_->FindTypeByNameAndVersion(
            request.type_name(), GetRequestTypeVersion(request), &context_type);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        Context context;
        status = metadata_access_object_->FindContextByTypeIdAndContextName(
            context_type.id(), request.context_name(), &context);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        *response->mutable_context() = context;
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::PutAttributionsAndAssociations(
    const PutAttributionsAndAssociationsRequest& request,
    PutAttributionsAndAssociationsResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const Attribution& attribution : request.attributions()) {
          MLMD_RETURN_IF_ERROR(InsertAttributionIfNotExist(
              attribution.context_id(), attribution.artifact_id(),
              metadata_access_object_.get()));
        }
        for (const Association& association : request.associations()) {
          MLMD_RETURN_IF_ERROR(InsertAssociationIfNotExist(
              association.context_id(), association.execution_id(),
              metadata_access_object_.get()));
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::PutParentContexts(
    const PutParentContextsRequest& request,
    PutParentContextsResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const ParentContext& parent_context : request.parent_contexts()) {
          MLMD_RETURN_IF_ERROR(
              metadata_access_object_->CreateParentContext(parent_context));
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetContextsByArtifact(
    const GetContextsByArtifactRequest& request,
    GetContextsByArtifactResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> contexts;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindContextsByArtifact(
            request.artifact_id(), &contexts));
        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetContextsByExecution(
    const GetContextsByExecutionRequest& request,
    GetContextsByExecutionResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> contexts;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindContextsByExecution(
            request.execution_id(), &contexts));
        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetArtifactsByContext(
    const GetArtifactsByContextRequest& request,
    GetArtifactsByContextResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Artifact> artifacts;
        std::string next_page_token;
        auto list_options = request.has_options()
                                ? absl::make_optional(request.options())
                                : absl::nullopt;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindArtifactsByContext(
            request.context_id(), list_options, &artifacts, &next_page_token));

        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }

        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetExecutionsByContext(
    const GetExecutionsByContextRequest& request,
    GetExecutionsByContextResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Execution> executions;
        std::string next_page_token;
        auto list_options = request.has_options()
                                ? absl::make_optional(request.options())
                                : absl::nullopt;

        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindExecutionsByContext(
            request.context_id(), list_options, &executions, &next_page_token));

        for (const Execution& execution : executions) {
          *response->mutable_executions()->Add() = execution;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }

        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetParentContextsByContext(
    const GetParentContextsByContextRequest& request,
    GetParentContextsByContextResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> parent_contexts;
        const absl::Status status =
            metadata_access_object_->FindParentContextsByContextId(
                request.context_id(), &parent_contexts);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        absl::c_copy(parent_contexts, google::protobuf::RepeatedPtrFieldBackInserter(
                                          response->mutable_contexts()));
        return absl::OkStatus();
      }));
}

tensorflow::Status MetadataStore::GetChildrenContextsByContext(
    const GetChildrenContextsByContextRequest& request,
    GetChildrenContextsByContextResponse* response) {
  return FromABSLStatus(transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> child_contexts;
        const absl::Status status =
            metadata_access_object_->FindChildContextsByContextId(
                request.context_id(), &child_contexts);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        absl::c_copy(child_contexts, google::protobuf::RepeatedPtrFieldBackInserter(
                                         response->mutable_contexts()));
        return absl::OkStatus();
      }));
}


MetadataStore::MetadataStore(
    std::unique_ptr<MetadataSource> metadata_source,
    std::unique_ptr<MetadataAccessObject> metadata_access_object,
    std::unique_ptr<TransactionExecutor> transaction_executor)
    : metadata_source_(std::move(metadata_source)),
      metadata_access_object_(std::move(metadata_access_object)),
      transaction_executor_(std::move(transaction_executor)) {}

}  // namespace ml_metadata
