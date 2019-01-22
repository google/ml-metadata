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

#include "absl/memory/memory.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ml_metadata {
namespace {
using std::unique_ptr;

template <typename T>
// Test if two types have identical names and properties.
// Works for ArtifactType and ExecutionType.
bool CheckIdentical(const T& type_a, const T& type_b) {
  if (type_a.name() != type_b.name()) {
    return false;
  }
  // Make sure every property in a is in b, and has the same type.
  for (const auto& pair : type_a.properties()) {
    const string& key = pair.first;
    const PropertyType value = pair.second;
    const auto other_iter = type_b.properties().find(key);
    if (other_iter == type_b.properties().end()) {
      return false;
    }
    if (other_iter->second != value) {
      return false;
    }
  }
  // If every property that is in a is in b, and the size is the same, then
  // the properties are the same.
  return type_a.properties_size() == type_b.properties_size();
}
}  // namespace

tensorflow::Status MetadataStore::InitMetadataStore() {
  ScopedTransaction transaction(metadata_source_.get());
  TF_RETURN_IF_ERROR(metadata_access_object_->InitMetadataSource());
  return transaction.Commit();
}

tensorflow::Status MetadataStore::InitMetadataStoreIfNotExists() {
  ScopedTransaction transaction(metadata_source_.get());
  TF_RETURN_IF_ERROR(metadata_access_object_->InitMetadataSourceIfNotExists());
  return transaction.Commit();
}

tensorflow::Status MetadataStore::PutArtifactType(
    const PutArtifactTypeRequest& request, PutArtifactTypeResponse* response) {
  if (request.can_add_fields()) {
    return tensorflow::errors::Unimplemented("Cannot add fields.");
  }
  if (request.can_delete_fields()) {
    return tensorflow::errors::Unimplemented("Cannot remove fields.");
  }
  if (!request.all_fields_match()) {
    return tensorflow::errors::Unimplemented("Must match all fields.");
  }
  if (!request.artifact_type().has_id()) {
    ScopedTransaction transaction(metadata_source_.get());
    int64 type_id;

    ArtifactType current;
    tensorflow::Status status = metadata_access_object_->FindTypeByName(
        request.artifact_type().name(), &current);
    if (status.ok()) {
      if (!CheckIdentical(current, request.artifact_type())) {
        return tensorflow::errors::AlreadyExists(
            "Type already exists with different properties.");
      }
      response->set_type_id(current.id());
      return transaction.Commit();
    } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
      return status;
    }
    // If the type does not exist, create it.
    TF_RETURN_IF_ERROR(
        metadata_access_object_->CreateType(request.artifact_type(), &type_id));
    response->set_type_id(type_id);
    return transaction.Commit();
  } else {
    return tensorflow::errors::Unimplemented(
        "Updating type by ID not implemented.");
  }
}

tensorflow::Status MetadataStore::InsertExecutionType(
    const PutExecutionTypeRequest& request,
    PutExecutionTypeResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  int64 type_id;

  ExecutionType current;
  tensorflow::Status status = metadata_access_object_->FindTypeByName(
      request.execution_type().name(), &current);
  if (status.ok()) {
    if (!CheckIdentical(current, request.execution_type())) {
      return tensorflow::errors::AlreadyExists(
          "Type already exists with different properties.");
    }
    response->set_type_id(current.id());
    return transaction.Commit();
  } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
    return status;
  }
  // If the type does not exist, create it.
  TF_RETURN_IF_ERROR(
      metadata_access_object_->CreateType(request.execution_type(), &type_id));
  response->set_type_id(type_id);
  return transaction.Commit();
}

tensorflow::Status MetadataStore::PutExecutionType(
    const PutExecutionTypeRequest& request,
    PutExecutionTypeResponse* response) {
  if (request.can_add_fields()) {
    return tensorflow::errors::Unimplemented("Cannot add fields.");
  }
  if (request.can_delete_fields()) {
    return tensorflow::errors::Unimplemented("Cannot remove fields.");
  }
  if (!request.all_fields_match()) {
    return tensorflow::errors::Unimplemented("Must match all fields.");
  }
  if (!request.execution_type().has_id()) {
    return InsertExecutionType(request, response);
  } else {
    return tensorflow::errors::Unimplemented(
        "Updating type by ID not implemented.");
  }
}

tensorflow::Status MetadataStore::GetArtifactType(
    const GetArtifactTypeRequest& request, GetArtifactTypeResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  TF_RETURN_IF_ERROR(metadata_access_object_->FindTypeByName(
      request.type_name(), response->mutable_artifact_type()));
  return transaction.Commit();
}

tensorflow::Status MetadataStore::GetExecutionType(
    const GetExecutionTypeRequest& request,
    GetExecutionTypeResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  tensorflow::Status result = metadata_access_object_->FindTypeByName(
      request.type_name(), response->mutable_execution_type());
  TF_RETURN_IF_ERROR(result);
  return transaction.Commit();
}

tensorflow::Status MetadataStore::GetArtifactsByID(
    const GetArtifactsByIDRequest& request,
    GetArtifactsByIDResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  for (const int64 artifact_id : request.artifact_ids()) {
    Artifact artifact;
    tensorflow::Status status =
        metadata_access_object_->FindArtifactById(artifact_id, &artifact);
    if (status.ok()) {
      *response->mutable_artifacts()->Add() = artifact;
    } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
      return status;
    }
  }
  return transaction.Commit();
}

tensorflow::Status MetadataStore::GetArtifactTypesByID(
    const GetArtifactTypesByIDRequest& request,
    GetArtifactTypesByIDResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  for (const int64 type_id : request.type_ids()) {
    ArtifactType artifact_type;
    tensorflow::Status status =
        metadata_access_object_->FindTypeById(type_id, &artifact_type);
    if (status.ok()) {
      *response->mutable_artifact_types()->Add() = artifact_type;
    } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
      return status;
    }
  }
  return transaction.Commit();
}

tensorflow::Status MetadataStore::GetExecutionTypesByID(
    const GetExecutionTypesByIDRequest& request,
    GetExecutionTypesByIDResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  for (const int64 type_id : request.type_ids()) {
    ExecutionType execution_type;
    tensorflow::Status status =
        metadata_access_object_->FindTypeById(type_id, &execution_type);
    if (status.ok()) {
      *response->mutable_execution_types()->Add() = execution_type;
    } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
      return status;
    }
  }
  return transaction.Commit();
}

tensorflow::Status MetadataStore::GetExecutionsByID(
    const GetExecutionsByIDRequest& request,
    GetExecutionsByIDResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  for (const int64 execution_id : request.execution_ids()) {
    Execution execution;
    tensorflow::Status status =
        metadata_access_object_->FindExecutionById(execution_id, &execution);
    if (status.ok()) {
      *response->mutable_executions()->Add() = execution;
    } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
      return status;
    }
  }
  return transaction.Commit();
}

tensorflow::Status MetadataStore::PutArtifacts(
    const PutArtifactsRequest& request, PutArtifactsResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  for (const Artifact& artifact : request.artifacts()) {
    if (artifact.has_id()) {
      TF_RETURN_IF_ERROR(metadata_access_object_->UpdateArtifact(artifact));
      response->add_artifact_ids(artifact.id());
    } else {
      int64 artifact_id;
      TF_RETURN_IF_ERROR(
          metadata_access_object_->CreateArtifact(artifact, &artifact_id));
      response->add_artifact_ids(artifact_id);
    }
  }
  return transaction.Commit();
}

tensorflow::Status MetadataStore::PutExecutions(
    const PutExecutionsRequest& request, PutExecutionsResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  for (const Execution& execution : request.executions()) {
    if (execution.has_id()) {
      TF_RETURN_IF_ERROR(metadata_access_object_->UpdateExecution(execution));
      response->add_execution_ids(execution.id());
    } else {
      int64 execution_id;
      TF_RETURN_IF_ERROR(
          metadata_access_object_->CreateExecution(execution, &execution_id));
      response->add_execution_ids(execution_id);
    }
  }
  return transaction.Commit();
}

tensorflow::Status MetadataStore::Create(
    const MetadataSourceQueryConfig& query_config,
    unique_ptr<MetadataSource> metadata_source,
    unique_ptr<MetadataStore>* result) {
  unique_ptr<MetadataAccessObject> metadata_access_object;
  TF_RETURN_IF_ERROR(MetadataAccessObject::Create(
      query_config, metadata_source.get(), &metadata_access_object));
  *result = absl::WrapUnique(new MetadataStore(
      std::move(metadata_source), std::move(metadata_access_object)));
  return tensorflow::Status::OK();
}

tensorflow::Status MetadataStore::PutEvents(const PutEventsRequest& request,
                                            PutEventsResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  for (const Event& event : request.events()) {
    int64 dummy_event_id;
    TF_RETURN_IF_ERROR(
        metadata_access_object_->CreateEvent(event, &dummy_event_id));
  }
  return transaction.Commit();
}

tensorflow::Status MetadataStore::GetEventsByExecutionIDs(
    const GetEventsByExecutionIDsRequest& request,
    GetEventsByExecutionIDsResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());

  for (const int64 execution_id : request.execution_ids()) {
    std::vector<Event> events;
    tensorflow::Status status =
        metadata_access_object_->FindEventsByExecution(execution_id, &events);
    if (status.ok()) {
      for (const Event& event : events) {
        *response->mutable_events()->Add() = event;
      }
    } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
      return status;
    }
  }
  return transaction.Commit();
}

tensorflow::Status MetadataStore::GetEventsByArtifactIDs(
    const GetEventsByArtifactIDsRequest& request,
    GetEventsByArtifactIDsResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());

  for (const int64 artifact_id : request.artifact_ids()) {
    std::vector<Event> events;
    tensorflow::Status status =
        metadata_access_object_->FindEventsByArtifact(artifact_id, &events);
    if (status.ok()) {
      for (const Event& event : events) {
        *response->mutable_events()->Add() = event;
      }
    } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
      return status;
    }
  }
  return transaction.Commit();
}

tensorflow::Status MetadataStore::GetExecutions(
    const GetExecutionsRequest& request, GetExecutionsResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  std::vector<Execution> executions;
  TF_RETURN_IF_ERROR(metadata_access_object_->FindExecutions(&executions));
  for (const Execution& execution : executions) {
    *response->mutable_executions()->Add() = execution;
  }
  return transaction.Commit();
}

tensorflow::Status MetadataStore::GetArtifacts(
    const GetArtifactsRequest& request, GetArtifactsResponse* response) {
  ScopedTransaction transaction(metadata_source_.get());
  std::vector<Artifact> artifacts;
  TF_RETURN_IF_ERROR(metadata_access_object_->FindArtifacts(&artifacts));
  for (const Artifact& artifact : artifacts) {
    *response->mutable_artifacts()->Add() = artifact;
  }
  return transaction.Commit();
}

MetadataStore::MetadataStore(
    std::unique_ptr<MetadataSource> metadata_source,
    std::unique_ptr<MetadataAccessObject> metadata_access_object)
    : metadata_source_(std::move(metadata_source)),
      metadata_access_object_(std::move(metadata_access_object)) {}

}  // namespace ml_metadata
