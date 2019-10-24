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
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ml_metadata {
namespace {
using std::unique_ptr;

// Checks if the `other_type` have the same names and all list of properties.
// Returns true if the types are consistent.
// For a type to be consistent:
// - all properties in stored_type are in other_type, and have the same type.
// - either can_add_fields is true or the set of properties in both types
//   are identical.
template <typename T>
bool CheckFieldsConsistent(const T& stored_type, const T& other_type,
                           bool can_add_fields) {
  if (stored_type.name() != other_type.name()) {
    return false;
  }
  // Make sure every property in a is in b, and has the same type.
  for (const auto& pair : stored_type.properties()) {
    const string& key = pair.first;
    const PropertyType value = pair.second;
    const auto other_iter = other_type.properties().find(key);
    if (other_iter == other_type.properties().end()) {
      return false;
    }
    if (other_iter->second != value) {
      return false;
    }
  }
  return can_add_fields ||
         stored_type.properties_size() == other_type.properties_size();
}

// If a type with the same name already exists (let's call it `old_type`), it
// checks the consistency of `type` and `old_type` as described in
// CheckFieldsConsistent according to can_add_fields.
// If there are inconsistent, it returns ALREADY_EXISTS. If they are consistent
// and the types are identical, it returns the old type_id. If they are
// consistent and there are new fields in `type`, then those fields are added.
// If there is no type having the same name, then insert a new type.
// Returns INVALID_ARGUMENT error, if name field in `type` is not given.
// Returns INVALID_ARGUMENT error, if any property type in `type` is unknown.
// Returns detailed INTERNAL error, if query execution fails.
template <typename T>
tensorflow::Status UpsertType(MetadataAccessObject* metadata_access_object,
                              const T& type, bool can_add_fields,
                              int64* type_id) {
  T stored_type;
  const tensorflow::Status status =
      metadata_access_object->FindTypeByName(type.name(), &stored_type);
  if (!status.ok() && !tensorflow::errors::IsNotFound(status)) {
    return status;
  }
  // if not found, then it creates a type. `can_add_fields` is ignored.
  if (tensorflow::errors::IsNotFound(status)) {
    return metadata_access_object->CreateType(type, type_id);
  }
  // otherwise it is update type.
  *type_id = stored_type.id();
  // all properties in stored_type must match the given type.
  // if `can_add_fields` is set, then new properties can be added
  if (!CheckFieldsConsistent(stored_type, type, can_add_fields)) {
    return tensorflow::errors::AlreadyExists(
        "Type already exists with different properties.");
  }
  return metadata_access_object->UpdateType(type);
}

}  // namespace

tensorflow::Status MetadataStore::InitMetadataStore() {
  return ExecuteTransaction(
      metadata_source_.get(), [this]() -> tensorflow::Status {
        return metadata_access_object_->InitMetadataSource();
      });
}

tensorflow::Status MetadataStore::InitMetadataStoreIfNotExists() {
  return ExecuteTransaction(
      metadata_source_.get(), [this]() -> tensorflow::Status {
        return metadata_access_object_->InitMetadataSourceIfNotExists();
      });
}

tensorflow::Status MetadataStore::PutTypes(const PutTypesRequest& request,
                                           PutTypesResponse* response) {
  if (request.can_delete_fields()) {
    return tensorflow::errors::Unimplemented("Cannot remove fields.");
  }
  if (!request.all_fields_match()) {
    return tensorflow::errors::Unimplemented("Must match all fields.");
  }
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const ArtifactType& artifact_type : request.artifact_types()) {
          int64 artifact_type_id;
          TF_RETURN_IF_ERROR(UpsertType(metadata_access_object_.get(),
                                        artifact_type, request.can_add_fields(),
                                        &artifact_type_id));
          response->add_artifact_type_ids(artifact_type_id);
        }
        for (const ExecutionType& execution_type : request.execution_types()) {
          int64 execution_type_id;
          TF_RETURN_IF_ERROR(
              UpsertType(metadata_access_object_.get(), execution_type,
                         request.can_add_fields(), &execution_type_id));
          response->add_execution_type_ids(execution_type_id);
        }
        for (const ContextType& context_type : request.context_types()) {
          int64 context_type_id;
          TF_RETURN_IF_ERROR(UpsertType(metadata_access_object_.get(),
                                        context_type, request.can_add_fields(),
                                        &context_type_id));
          response->add_context_type_ids(context_type_id);
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutArtifactType(
    const PutArtifactTypeRequest& request, PutArtifactTypeResponse* response) {
  if (request.can_delete_fields()) {
    return tensorflow::errors::Unimplemented("Cannot remove fields.");
  }
  if (!request.all_fields_match()) {
    return tensorflow::errors::Unimplemented("Must match all fields.");
  }
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        int64 type_id;
        TF_RETURN_IF_ERROR(UpsertType(metadata_access_object_.get(),
                                      request.artifact_type(),
                                      request.can_add_fields(), &type_id));
        response->set_type_id(type_id);
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutExecutionType(
    const PutExecutionTypeRequest& request,
    PutExecutionTypeResponse* response) {
  if (request.can_delete_fields()) {
    return tensorflow::errors::Unimplemented("Cannot remove fields.");
  }
  if (!request.all_fields_match()) {
    return tensorflow::errors::Unimplemented("Must match all fields.");
  }
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        int64 type_id;
        TF_RETURN_IF_ERROR(UpsertType(metadata_access_object_.get(),
                                      request.execution_type(),
                                      request.can_add_fields(), &type_id));
        response->set_type_id(type_id);
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutContextType(
    const PutContextTypeRequest& request, PutContextTypeResponse* response) {
  if (request.can_delete_fields()) {
    return tensorflow::errors::Unimplemented("Cannot remove fields.");
  }
  if (!request.all_fields_match()) {
    return tensorflow::errors::Unimplemented("Must match all fields.");
  }
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        int64 type_id;
        TF_RETURN_IF_ERROR(UpsertType(metadata_access_object_.get(),
                                      request.context_type(),
                                      request.can_add_fields(), &type_id));
        response->set_type_id(type_id);
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifactType(
    const GetArtifactTypeRequest& request, GetArtifactTypeResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        return metadata_access_object_->FindTypeByName(
            request.type_name(), response->mutable_artifact_type());
      });
}

tensorflow::Status MetadataStore::GetExecutionType(
    const GetExecutionTypeRequest& request,
    GetExecutionTypeResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        return metadata_access_object_->FindTypeByName(
            request.type_name(), response->mutable_execution_type());
      });
}

tensorflow::Status MetadataStore::GetContextType(
    const GetContextTypeRequest& request, GetContextTypeResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        return metadata_access_object_->FindTypeByName(
            request.type_name(), response->mutable_context_type());
      });
}

tensorflow::Status MetadataStore::GetArtifactTypesByID(
    const GetArtifactTypesByIDRequest& request,
    GetArtifactTypesByIDResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const int64 type_id : request.type_ids()) {
          ArtifactType artifact_type;
          const tensorflow::Status status =
              metadata_access_object_->FindTypeById(type_id, &artifact_type);
          if (status.ok()) {
            *response->mutable_artifact_types()->Add() = artifact_type;
          } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetExecutionTypesByID(
    const GetExecutionTypesByIDRequest& request,
    GetExecutionTypesByIDResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const int64 type_id : request.type_ids()) {
          ExecutionType execution_type;
          const tensorflow::Status status =
              metadata_access_object_->FindTypeById(type_id, &execution_type);
          if (status.ok()) {
            *response->mutable_execution_types()->Add() = execution_type;
          } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetContextTypesByID(
    const GetContextTypesByIDRequest& request,
    GetContextTypesByIDResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const int64 type_id : request.type_ids()) {
          ContextType context_type;
          const tensorflow::Status status =
              metadata_access_object_->FindTypeById(type_id, &context_type);
          if (status.ok()) {
            *response->mutable_context_types()->Add() = context_type;
          } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifactsByID(
    const GetArtifactsByIDRequest& request,
    GetArtifactsByIDResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const int64 artifact_id : request.artifact_ids()) {
          Artifact artifact;
          const tensorflow::Status status =
              metadata_access_object_->FindArtifactById(artifact_id, &artifact);
          if (status.ok()) {
            *response->mutable_artifacts()->Add() = artifact;
          } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetExecutionsByID(
    const GetExecutionsByIDRequest& request,
    GetExecutionsByIDResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const int64 execution_id : request.execution_ids()) {
          Execution execution;
          const tensorflow::Status status =
              metadata_access_object_->FindExecutionById(execution_id,
                                                         &execution);
          if (status.ok()) {
            *response->mutable_executions()->Add() = execution;
          } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetContextsByID(
    const GetContextsByIDRequest& request, GetContextsByIDResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const int64 context_id : request.context_ids()) {
          Context context;
          const tensorflow::Status status =
              metadata_access_object_->FindContextById(context_id, &context);
          if (status.ok()) {
            *response->mutable_contexts()->Add() = context;
          } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutArtifacts(
    const PutArtifactsRequest& request, PutArtifactsResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const Artifact& artifact : request.artifacts()) {
          if (artifact.has_id()) {
            TF_RETURN_IF_ERROR(
                metadata_access_object_->UpdateArtifact(artifact));
            response->add_artifact_ids(artifact.id());
          } else {
            int64 artifact_id;
            TF_RETURN_IF_ERROR(metadata_access_object_->CreateArtifact(
                artifact, &artifact_id));
            response->add_artifact_ids(artifact_id);
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutExecutions(
    const PutExecutionsRequest& request, PutExecutionsResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const Execution& execution : request.executions()) {
          if (execution.has_id()) {
            TF_RETURN_IF_ERROR(
                metadata_access_object_->UpdateExecution(execution));
            response->add_execution_ids(execution.id());
          } else {
            int64 execution_id;
            TF_RETURN_IF_ERROR(metadata_access_object_->CreateExecution(
                execution, &execution_id));
            response->add_execution_ids(execution_id);
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutContexts(const PutContextsRequest& request,
                                              PutContextsResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const Context& context : request.contexts()) {
          if (context.has_id()) {
            TF_RETURN_IF_ERROR(metadata_access_object_->UpdateContext(context));
            response->add_context_ids(context.id());
          } else {
            int64 context_id;
            TF_RETURN_IF_ERROR(
                metadata_access_object_->CreateContext(context, &context_id));
            response->add_context_ids(context_id);
          }
        }
        return tensorflow::Status::OK();
      });
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
  return ExecuteTransaction(
      metadata_source_.get(), [this, &request]() -> tensorflow::Status {
        for (const Event& event : request.events()) {
          int64 dummy_event_id = -1;
          TF_RETURN_IF_ERROR(
              metadata_access_object_->CreateEvent(event, &dummy_event_id));
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutExecution(
    const PutExecutionRequest& request, PutExecutionResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        if (!request.has_execution()) {
          return tensorflow::errors::InvalidArgument("No execution is found: ",
                                                     request.DebugString());
        }
        // 1. Upsert Execution
        const Execution& execution = request.execution();
        int64 execution_id = -1;
        if (execution.has_id()) {
          TF_RETURN_IF_ERROR(
              metadata_access_object_->UpdateExecution(execution));
          execution_id = execution.id();
        } else {
          TF_RETURN_IF_ERROR(metadata_access_object_->CreateExecution(
              execution, &execution_id));
        }
        response->set_execution_id(execution_id);
        // 2. Upsert Artifacts and insert events
        for (const PutExecutionRequest::ArtifactAndEvent& artifact_and_event :
             request.artifact_event_pairs()) {
          if (!artifact_and_event.has_artifact()) {
            return tensorflow::errors::InvalidArgument(
                "Request has no artifact: ", request.DebugString());
          }
          const Artifact& artifact = artifact_and_event.artifact();
          int64 artifact_id = -1;
          if (artifact.has_id()) {
            TF_RETURN_IF_ERROR(
                metadata_access_object_->UpdateArtifact(artifact));
            artifact_id = artifact.id();
          } else {
            TF_RETURN_IF_ERROR(metadata_access_object_->CreateArtifact(
                artifact, &artifact_id));
          }
          response->add_artifact_ids(artifact_id);
          // insert event if any
          if (!artifact_and_event.has_event()) {
            continue;
          }
          Event event = artifact_and_event.event();
          if ((event.has_artifact_id() && !artifact.has_id()) ||
              (event.has_artifact_id() && artifact_id != event.artifact_id())) {
            return tensorflow::errors::InvalidArgument(
                "Request's event.artifact_id does not match with the given "
                "artifact: ",
                request.DebugString());
          }
          event.set_artifact_id(artifact_id);
          if ((event.has_execution_id() && !execution.has_id()) ||
              (event.has_execution_id() &&
               execution_id != event.execution_id())) {
            return tensorflow::errors::InvalidArgument(
                "Request's event.execution_id does not match with the given "
                "execution: ",
                request.DebugString());
          }
          event.set_execution_id(execution_id);
          int64 dummy_event_id = -1;
          TF_RETURN_IF_ERROR(
              metadata_access_object_->CreateEvent(event, &dummy_event_id));
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetEventsByExecutionIDs(
    const GetEventsByExecutionIDsRequest& request,
    GetEventsByExecutionIDsResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const int64 execution_id : request.execution_ids()) {
          std::vector<Event> events;
          const tensorflow::Status status =
              metadata_access_object_->FindEventsByExecution(execution_id,
                                                             &events);
          if (status.ok()) {
            for (const Event& event : events) {
              *response->mutable_events()->Add() = event;
            }
          } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetEventsByArtifactIDs(
    const GetEventsByArtifactIDsRequest& request,
    GetEventsByArtifactIDsResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        for (const int64 artifact_id : request.artifact_ids()) {
          std::vector<Event> events;
          const tensorflow::Status status =
              metadata_access_object_->FindEventsByArtifact(artifact_id,
                                                            &events);
          if (status.ok()) {
            for (const Event& event : events) {
              *response->mutable_events()->Add() = event;
            }
          } else if (status.code() != ::tensorflow::error::NOT_FOUND) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetExecutions(
    const GetExecutionsRequest& request, GetExecutionsResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(), [this, &response]() -> tensorflow::Status {
        std::vector<Execution> executions;
        TF_RETURN_IF_ERROR(
            metadata_access_object_->FindExecutions(&executions));
        for (const Execution& execution : executions) {
          *response->mutable_executions()->Add() = execution;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifacts(
    const GetArtifactsRequest& request, GetArtifactsResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(), [this, &response]() -> tensorflow::Status {
        std::vector<Artifact> artifacts;
        TF_RETURN_IF_ERROR(metadata_access_object_->FindArtifacts(&artifacts));
        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetContexts(const GetContextsRequest& request,
                                              GetContextsResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(), [this, &response]() -> tensorflow::Status {
        std::vector<Context> contexts;
        TF_RETURN_IF_ERROR(metadata_access_object_->FindContexts(&contexts));
        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifactTypes(
    const GetArtifactTypesRequest& request,
    GetArtifactTypesResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(), [this, &response]() -> tensorflow::Status {
        std::vector<ArtifactType> artifact_types;
        const tensorflow::Status status =
            metadata_access_object_->FindTypes(&artifact_types);
        if (status.code() == ::tensorflow::error::NOT_FOUND) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        for (const ArtifactType& artifact_type : artifact_types) {
          *response->mutable_artifact_types()->Add() = artifact_type;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetExecutionTypes(
    const GetExecutionTypesRequest& request,
    GetExecutionTypesResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(), [this, &response]() -> tensorflow::Status {
        std::vector<ExecutionType> execution_types;
        const tensorflow::Status status =
            metadata_access_object_->FindTypes(&execution_types);
        if (status.code() == ::tensorflow::error::NOT_FOUND) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        for (const ExecutionType& execution_type : execution_types) {
          *response->mutable_execution_types()->Add() = execution_type;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetContextTypes(
    const GetContextTypesRequest& request, GetContextTypesResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(), [this, &response]() -> tensorflow::Status {
        std::vector<ContextType> context_types;
        const tensorflow::Status status =
            metadata_access_object_->FindTypes(&context_types);
        if (status.code() == ::tensorflow::error::NOT_FOUND) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        for (const ContextType& context_type : context_types) {
          *response->mutable_context_types()->Add() = context_type;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifactsByURI(
    const GetArtifactsByURIRequest& request,
    GetArtifactsByURIResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        std::vector<Artifact> artifacts;
        const tensorflow::Status status =
            metadata_access_object_->FindArtifactsByURI(request.uri(),
                                                        &artifacts);
        if (status.code() == ::tensorflow::error::NOT_FOUND) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifactsByType(
    const GetArtifactsByTypeRequest& request,
    GetArtifactsByTypeResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        ArtifactType artifact_type;
        tensorflow::Status status = metadata_access_object_->FindTypeByName(
            request.type_name(), &artifact_type);
        if (status.code() == ::tensorflow::error::NOT_FOUND) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        std::vector<Artifact> artifacts;
        status = metadata_access_object_->FindArtifactsByTypeId(
            artifact_type.id(), &artifacts);
        if (status.code() == ::tensorflow::error::NOT_FOUND) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetExecutionsByType(
    const GetExecutionsByTypeRequest& request,
    GetExecutionsByTypeResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        ExecutionType execution_type;
        tensorflow::Status status = metadata_access_object_->FindTypeByName(
            request.type_name(), &execution_type);
        if (status.code() == ::tensorflow::error::NOT_FOUND) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        std::vector<Execution> executions;
        status = metadata_access_object_->FindExecutionsByTypeId(
            execution_type.id(), &executions);
        if (status.code() == ::tensorflow::error::NOT_FOUND) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        for (const Execution& execution : executions) {
          *response->mutable_executions()->Add() = execution;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetContextsByType(
    const GetContextsByTypeRequest& request,
    GetContextsByTypeResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        ContextType context_type;
        tensorflow::Status status = metadata_access_object_->FindTypeByName(
            request.type_name(), &context_type);
        if (status.code() == ::tensorflow::error::NOT_FOUND) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        std::vector<Context> contexts;
        status = metadata_access_object_->FindContextsByTypeId(
            context_type.id(), &contexts);
        if (status.code() == ::tensorflow::error::NOT_FOUND) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutAttributionsAndAssociations(
    const PutAttributionsAndAssociationsRequest& request,
    PutAttributionsAndAssociationsResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(), [this, &request]() -> tensorflow::Status {
        for (const Attribution& attribution : request.attributions()) {
          int64 dummy_attribution_id;
          tensorflow::Status status =
              metadata_access_object_->CreateAttribution(attribution,
                                                         &dummy_attribution_id);
          if (tensorflow::errors::IsAlreadyExists(status)) continue;
          TF_RETURN_IF_ERROR(status);
        }
        for (const Association& association : request.associations()) {
          int64 dummy_assocation_id;
          tensorflow::Status status =
              metadata_access_object_->CreateAssociation(association,
                                                         &dummy_assocation_id);
          if (tensorflow::errors::IsAlreadyExists(status)) continue;
          TF_RETURN_IF_ERROR(status);
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetContextsByArtifact(
    const GetContextsByArtifactRequest& request,
    GetContextsByArtifactResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        std::vector<Context> contexts;
        TF_RETURN_IF_ERROR(metadata_access_object_->FindContextsByArtifact(
            request.artifact_id(), &contexts));
        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetContextsByExecution(
    const GetContextsByExecutionRequest& request,
    GetContextsByExecutionResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        std::vector<Context> contexts;
        TF_RETURN_IF_ERROR(metadata_access_object_->FindContextsByExecution(
            request.execution_id(), &contexts));
        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifactsByContext(
    const GetArtifactsByContextRequest& request,
    GetArtifactsByContextResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        std::vector<Artifact> artifacts;
        TF_RETURN_IF_ERROR(metadata_access_object_->FindArtifactsByContext(
            request.context_id(), &artifacts));
        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetExecutionsByContext(
    const GetExecutionsByContextRequest& request,
    GetExecutionsByContextResponse* response) {
  return ExecuteTransaction(
      metadata_source_.get(),
      [this, &request, &response]() -> tensorflow::Status {
        std::vector<Execution> executions;
        TF_RETURN_IF_ERROR(metadata_access_object_->FindExecutionsByContext(
            request.context_id(), &executions));
        for (const Execution& execution : executions) {
          *response->mutable_executions()->Add() = execution;
        }
        return tensorflow::Status::OK();
      });
}

MetadataStore::MetadataStore(
    std::unique_ptr<MetadataSource> metadata_source,
    std::unique_ptr<MetadataAccessObject> metadata_access_object)
    : metadata_source_(std::move(metadata_source)),
      metadata_access_object_(std::move(metadata_access_object)) {}

}  // namespace ml_metadata
