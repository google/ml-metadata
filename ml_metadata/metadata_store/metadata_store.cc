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

#include "google/protobuf/descriptor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_access_object_factory.h"
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
    const std::string& key = pair.first;
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
tensorflow::Status UpsertType(const T& type, bool can_add_fields,
                              MetadataAccessObject* metadata_access_object,
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

// Updates or inserts an artifact. If the artifact.id is given, it updates the
// stored artifact, otherwise, it creates a new artifact.
tensorflow::Status UpsertArtifact(const Artifact& artifact,
                                  MetadataAccessObject* metadata_access_object,
                                  int64* artifact_id) {
  CHECK(artifact_id) << "artifact_id should not be null";
  if (artifact.has_id()) {
    TF_RETURN_IF_ERROR(metadata_access_object->UpdateArtifact(artifact));
    *artifact_id = artifact.id();
  } else {
    TF_RETURN_IF_ERROR(
        metadata_access_object->CreateArtifact(artifact, artifact_id));
  }
  return tensorflow::Status::OK();
}

// Updates or inserts an execution. If the execution.id is given, it updates the
// stored execution, otherwise, it creates a new execution.
tensorflow::Status UpsertExecution(const Execution& execution,
                                   MetadataAccessObject* metadata_access_object,
                                   int64* execution_id) {
  CHECK(execution_id) << "execution_id should not be null";
  if (execution.has_id()) {
    TF_RETURN_IF_ERROR(metadata_access_object->UpdateExecution(execution));
    *execution_id = execution.id();
  } else {
    TF_RETURN_IF_ERROR(
        metadata_access_object->CreateExecution(execution, execution_id));
  }
  return tensorflow::Status::OK();
}

// Updates or inserts a context. If the context.id is given, it updates the
// stored context, otherwise, it creates a new context.
tensorflow::Status UpsertContext(const Context& context,
                                 MetadataAccessObject* metadata_access_object,
                                 int64* context_id) {
  CHECK(context_id) << "context_id should not be null";
  if (context.has_id()) {
    TF_RETURN_IF_ERROR(metadata_access_object->UpdateContext(context));
    *context_id = context.id();
  } else {
    TF_RETURN_IF_ERROR(
        metadata_access_object->CreateContext(context, context_id));
  }
  return tensorflow::Status::OK();
}

// Inserts an association. If the association already exists it returns OK.
tensorflow::Status InsertAssociationIfNotExist(
    int64 context_id, int64 execution_id,
    MetadataAccessObject* metadata_access_object) {
  Association association;
  association.set_execution_id(execution_id);
  association.set_context_id(context_id);
  int64 dummy_assocation_id;
  tensorflow::Status status = metadata_access_object->CreateAssociation(
      association, &dummy_assocation_id);
  if (!status.ok() && !tensorflow::errors::IsAlreadyExists(status)) {
    return status;
  }
  return tensorflow::Status::OK();
}

// Inserts an attribution. If the attribution already exists it returns OK.
tensorflow::Status InsertAttributionIfNotExist(
    int64 context_id, int64 artifact_id,
    MetadataAccessObject* metadata_access_object) {
  Attribution attribution;
  attribution.set_artifact_id(artifact_id);
  attribution.set_context_id(context_id);
  int64 dummy_attribution_id;
  tensorflow::Status status = metadata_access_object->CreateAttribution(
      attribution, &dummy_attribution_id);
  if (!status.ok() && !tensorflow::errors::IsAlreadyExists(status)) {
    return status;
  }
  return tensorflow::Status::OK();
}

// Updates or inserts a pair of {Artifact, Event}. If artifact is not given,
// the event.artifact_id must exist, and it inserts the event, and returns the
// artifact_id. Otherwise if artifact is given, event.artifact_id is optional,
// if set, then artifact.id and event.artifact_id must align.
tensorflow::Status UpsertArtifactAndEvent(
    const PutExecutionRequest::ArtifactAndEvent& artifact_and_event,
    MetadataAccessObject* metadata_access_object, int64* artifact_id) {
  CHECK(artifact_id) << "The output artifact_id pointer should not be null";
  if (!artifact_and_event.has_artifact() && !artifact_and_event.has_event()) {
    return tensorflow::Status::OK();
  }
  // validate event and artifact's id aligns.
  // if artifact is not given, the event.artifact_id must exist
  absl::optional<int64> maybe_event_artifact_id =
      artifact_and_event.has_event() &&
              artifact_and_event.event().has_artifact_id()
          ? absl::make_optional<int64>(artifact_and_event.event().artifact_id())
          : absl::nullopt;
  if (!artifact_and_event.has_artifact() && !maybe_event_artifact_id) {
    return tensorflow::errors::InvalidArgument(absl::StrCat(
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
    return tensorflow::errors::InvalidArgument(absl::StrCat(
        "Given event.artifact_id is not aligned with the artifact: ",
        artifact_and_event.DebugString()));
  }
  // upsert artifact if present.
  if (artifact_and_event.has_artifact()) {
    TF_RETURN_IF_ERROR(UpsertArtifact(artifact_and_event.artifact(),
                                      metadata_access_object, artifact_id));
  }
  // insert event if any.
  if (!artifact_and_event.has_event()) {
    return tensorflow::Status::OK();
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

}  // namespace

tensorflow::Status MetadataStore::InitMetadataStore() {
  return transaction_executor_->Execute([this]() -> tensorflow::Status {
    return metadata_access_object_->InitMetadataSource();
  });
}

tensorflow::Status MetadataStore::InitMetadataStoreIfNotExists(
    const bool enable_upgrade_migration) {
  return transaction_executor_->Execute(
      [this, &enable_upgrade_migration]() -> tensorflow::Status {
        return metadata_access_object_->InitMetadataSourceIfNotExists(
            enable_upgrade_migration);
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
  return transaction_executor_->Execute([this, &request,
                                         &response]() -> tensorflow::Status {
    response->Clear();
    for (const ArtifactType& artifact_type : request.artifact_types()) {
      int64 artifact_type_id;
      TF_RETURN_IF_ERROR(UpsertType(artifact_type, request.can_add_fields(),
                                    metadata_access_object_.get(),
                                    &artifact_type_id));
      response->add_artifact_type_ids(artifact_type_id);
    }
    for (const ExecutionType& execution_type : request.execution_types()) {
      int64 execution_type_id;
      TF_RETURN_IF_ERROR(UpsertType(execution_type, request.can_add_fields(),
                                    metadata_access_object_.get(),
                                    &execution_type_id));
      response->add_execution_type_ids(execution_type_id);
    }
    for (const ContextType& context_type : request.context_types()) {
      int64 context_type_id;
      TF_RETURN_IF_ERROR(UpsertType(context_type, request.can_add_fields(),
                                    metadata_access_object_.get(),
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
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        int64 type_id;
        TF_RETURN_IF_ERROR(UpsertType(request.artifact_type(),
                                      request.can_add_fields(),
                                      metadata_access_object_.get(), &type_id));
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
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        int64 type_id;
        TF_RETURN_IF_ERROR(UpsertType(request.execution_type(),
                                      request.can_add_fields(),
                                      metadata_access_object_.get(), &type_id));
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
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        int64 type_id;
        TF_RETURN_IF_ERROR(UpsertType(request.context_type(),
                                      request.can_add_fields(),
                                      metadata_access_object_.get(), &type_id));
        response->set_type_id(type_id);
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifactType(
    const GetArtifactTypeRequest& request, GetArtifactTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        return metadata_access_object_->FindTypeByName(
            request.type_name(), response->mutable_artifact_type());
      });
}

tensorflow::Status MetadataStore::GetExecutionType(
    const GetExecutionTypeRequest& request,
    GetExecutionTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        return metadata_access_object_->FindTypeByName(
            request.type_name(), response->mutable_execution_type());
      });
}

tensorflow::Status MetadataStore::GetContextType(
    const GetContextTypeRequest& request, GetContextTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        return metadata_access_object_->FindTypeByName(
            request.type_name(), response->mutable_context_type());
      });
}

tensorflow::Status MetadataStore::GetArtifactTypesByID(
    const GetArtifactTypesByIDRequest& request,
    GetArtifactTypesByIDResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        for (const int64 type_id : request.type_ids()) {
          ArtifactType artifact_type;
          const tensorflow::Status status =
              metadata_access_object_->FindTypeById(type_id, &artifact_type);
          if (status.ok()) {
            *response->mutable_artifact_types()->Add() = artifact_type;
          } else if (!tensorflow::errors::IsNotFound(status)) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetExecutionTypesByID(
    const GetExecutionTypesByIDRequest& request,
    GetExecutionTypesByIDResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        for (const int64 type_id : request.type_ids()) {
          ExecutionType execution_type;
          const tensorflow::Status status =
              metadata_access_object_->FindTypeById(type_id, &execution_type);
          if (status.ok()) {
            *response->mutable_execution_types()->Add() = execution_type;
          } else if (!tensorflow::errors::IsNotFound(status)) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetContextTypesByID(
    const GetContextTypesByIDRequest& request,
    GetContextTypesByIDResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        for (const int64 type_id : request.type_ids()) {
          ContextType context_type;
          const tensorflow::Status status =
              metadata_access_object_->FindTypeById(type_id, &context_type);
          if (status.ok()) {
            *response->mutable_context_types()->Add() = context_type;
          } else if (!tensorflow::errors::IsNotFound(status)) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifactsByID(
    const GetArtifactsByIDRequest& request,
    GetArtifactsByIDResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        for (const int64 artifact_id : request.artifact_ids()) {
          Artifact artifact;
          const tensorflow::Status status =
              metadata_access_object_->FindArtifactById(artifact_id, &artifact);
          if (status.ok()) {
            *response->mutable_artifacts()->Add() = artifact;
          } else if (!tensorflow::errors::IsNotFound(status)) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetExecutionsByID(
    const GetExecutionsByIDRequest& request,
    GetExecutionsByIDResponse* response) {
  return transaction_executor_->Execute([this, &request,
                                         &response]() -> tensorflow::Status {
    response->Clear();
    for (const int64 execution_id : request.execution_ids()) {
      Execution execution;
      const tensorflow::Status status =
          metadata_access_object_->FindExecutionById(execution_id, &execution);
      if (status.ok()) {
        *response->mutable_executions()->Add() = execution;
      } else if (!tensorflow::errors::IsNotFound(status)) {
        return status;
      }
    }
    return tensorflow::Status::OK();
  });
}

tensorflow::Status MetadataStore::GetContextsByID(
    const GetContextsByIDRequest& request, GetContextsByIDResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        for (const int64 context_id : request.context_ids()) {
          Context context;
          const tensorflow::Status status =
              metadata_access_object_->FindContextById(context_id, &context);
          if (status.ok()) {
            *response->mutable_contexts()->Add() = context;
          } else if (!tensorflow::errors::IsNotFound(status)) {
            return status;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutArtifacts(
    const PutArtifactsRequest& request, PutArtifactsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        for (const Artifact& artifact : request.artifacts()) {
          int64 artifact_id = -1;
          TF_RETURN_IF_ERROR(UpsertArtifact(
              artifact, metadata_access_object_.get(), &artifact_id));
          response->add_artifact_ids(artifact_id);
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutExecutions(
    const PutExecutionsRequest& request, PutExecutionsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        for (const Execution& execution : request.executions()) {
          int64 execution_id = -1;
          TF_RETURN_IF_ERROR(UpsertExecution(
              execution, metadata_access_object_.get(), &execution_id));
          response->add_execution_ids(execution_id);
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutContexts(const PutContextsRequest& request,
                                              PutContextsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        for (const Context& context : request.contexts()) {
          int64 context_id = -1;
          TF_RETURN_IF_ERROR(UpsertContext(
              context, metadata_access_object_.get(), &context_id));
          response->add_context_ids(context_id);
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::Create(
    const MetadataSourceQueryConfig& query_config,
    const MigrationOptions& migration_options,
    unique_ptr<MetadataSource> metadata_source,
    unique_ptr<TransactionExecutor> transaction_executor,
    unique_ptr<MetadataStore>* result) {
  unique_ptr<MetadataAccessObject> metadata_access_object;
  TF_RETURN_IF_ERROR(CreateMetadataAccessObject(
      query_config, metadata_source.get(), &metadata_access_object));
  // if downgrade migration is specified
  if (migration_options.downgrade_to_schema_version() >= 0) {
    TF_RETURN_IF_ERROR(transaction_executor->Execute(
        [&migration_options, &metadata_access_object]() -> tensorflow::Status {
          return metadata_access_object->DowngradeMetadataSource(
              migration_options.downgrade_to_schema_version());
        }));
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
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
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
  return transaction_executor_->Execute([this, &request,
                                         &response]() -> tensorflow::Status {
    response->Clear();
    if (!request.has_execution()) {
      return tensorflow::errors::InvalidArgument("No execution is found: ",
                                                 request.DebugString());
    }
    // 1. Upsert Execution
    const Execution& execution = request.execution();
    int64 execution_id = -1;
    TF_RETURN_IF_ERROR(UpsertExecution(execution, metadata_access_object_.get(),
                                       &execution_id));
    response->set_execution_id(execution_id);
    // 2. Upsert Artifacts and insert events
    for (PutExecutionRequest::ArtifactAndEvent artifact_and_event :
         request.artifact_event_pairs()) {
      // validate execution and event if given
      if (artifact_and_event.has_event()) {
        Event* event = artifact_and_event.mutable_event();
        if (event->has_execution_id() &&
            (!execution.has_id() || execution.id() != event->execution_id())) {
          return tensorflow::errors::InvalidArgument(
              "Request's event.execution_id does not match with the given "
              "execution: ",
              request.DebugString());
        }
        event->set_execution_id(execution_id);
      }
      int64 artifact_id = -1;
      TF_RETURN_IF_ERROR(UpsertArtifactAndEvent(
          artifact_and_event, metadata_access_object_.get(), &artifact_id));
      response->add_artifact_ids(artifact_id);
    }
    // 3. Upsert contexts and insert associations and attributions.
    for (const Context& context : request.contexts()) {
      int64 context_id = -1;
      TF_RETURN_IF_ERROR(
          UpsertContext(context, metadata_access_object_.get(), &context_id));
      response->add_context_ids(context_id);
      TF_RETURN_IF_ERROR(InsertAssociationIfNotExist(
          context_id, response->execution_id(), metadata_access_object_.get()));
      for (const int64 artifact_id : response->artifact_ids()) {
        TF_RETURN_IF_ERROR(InsertAttributionIfNotExist(
            context_id, artifact_id, metadata_access_object_.get()));
      }
    }
    return tensorflow::Status::OK();
  });
}

tensorflow::Status MetadataStore::GetEventsByExecutionIDs(
    const GetEventsByExecutionIDsRequest& request,
    GetEventsByExecutionIDsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        std::vector<Event> events;
        const tensorflow::Status status =
            metadata_access_object_->FindEventsByExecutions(
                std::vector<int64>(request.execution_ids().begin(),
                              request.execution_ids().end()),
                &events);
        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        for (const Event& event : events) {
          *response->mutable_events()->Add() = event;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetEventsByArtifactIDs(
    const GetEventsByArtifactIDsRequest& request,
    GetEventsByArtifactIDsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        std::vector<Event> events;
        const tensorflow::Status status =
            metadata_access_object_->FindEventsByArtifacts(
                std::vector<int64>(request.artifact_ids().begin(),
                              request.artifact_ids().end()),
                &events);
        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        for (const Event& event : events) {
          *response->mutable_events()->Add() = event;
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetExecutions(
    const GetExecutionsRequest& request, GetExecutionsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        std::vector<Execution> executions;
        tensorflow::Status status;
        std::string next_page_token;
        if (request.has_options()) {
          status = metadata_access_object_->ListExecutions(
              request.options(), &executions, &next_page_token);
        } else {
          status = metadata_access_object_->FindExecutions(&executions);
        }

        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }

        for (const Execution& execution : executions) {
          *response->mutable_executions()->Add() = execution;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifacts(
    const GetArtifactsRequest& request, GetArtifactsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        std::vector<Artifact> artifacts;
        tensorflow::Status status;
        std::string next_page_token;
        if (request.has_options()) {
          status = metadata_access_object_->ListArtifacts(
              request.options(), &artifacts, &next_page_token);
        } else {
          status = metadata_access_object_->FindArtifacts(&artifacts);
        }

        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }

        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }

        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetContexts(const GetContextsRequest& request,
                                              GetContextsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        std::vector<Context> contexts;
        tensorflow::Status status;
        std::string next_page_token;
        if (request.has_options()) {
          status = metadata_access_object_->ListContexts(
              request.options(), &contexts, &next_page_token);
        } else {
          status = metadata_access_object_->FindContexts(&contexts);
        }

        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }

        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }

        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifactTypes(
    const GetArtifactTypesRequest& request,
    GetArtifactTypesResponse* response) {
  return transaction_executor_->Execute(
      [this, &response]() -> tensorflow::Status {
        response->Clear();
        std::vector<ArtifactType> artifact_types;
        const tensorflow::Status status =
            metadata_access_object_->FindTypes(&artifact_types);
        if (tensorflow::errors::IsNotFound(status)) {
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
  return transaction_executor_->Execute(
      [this, &response]() -> tensorflow::Status {
        response->Clear();
        std::vector<ExecutionType> execution_types;
        const tensorflow::Status status =
            metadata_access_object_->FindTypes(&execution_types);
        if (tensorflow::errors::IsNotFound(status)) {
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
  return transaction_executor_->Execute(
      [this, &response]() -> tensorflow::Status {
        response->Clear();
        std::vector<ContextType> context_types;
        const tensorflow::Status status =
            metadata_access_object_->FindTypes(&context_types);
        if (tensorflow::errors::IsNotFound(status)) {
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
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        absl::flat_hash_set<std::string> uris(request.uris().begin(),
                                              request.uris().end());
        for (const std::string& uri : uris) {
          std::vector<Artifact> artifacts;
          const tensorflow::Status status =
              metadata_access_object_->FindArtifactsByURI(uri, &artifacts);
          if (!status.ok() && !tensorflow::errors::IsNotFound(status)) {
            // If any none NotFound error returned, we do early stopping as
            // the query execution has internal db errors.
            return status;
          }
          for (const Artifact& artifact : artifacts) {
            *response->mutable_artifacts()->Add() = artifact;
          }
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetArtifactsByType(
    const GetArtifactsByTypeRequest& request,
    GetArtifactsByTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        ArtifactType artifact_type;
        tensorflow::Status status = metadata_access_object_->FindTypeByName(
            request.type_name(), &artifact_type);
        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        std::vector<Artifact> artifacts;
        status = metadata_access_object_->FindArtifactsByTypeId(
            artifact_type.id(), &artifacts);
        if (tensorflow::errors::IsNotFound(status)) {
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

tensorflow::Status MetadataStore::GetArtifactByTypeAndName(
    const GetArtifactByTypeAndNameRequest& request,
    GetArtifactByTypeAndNameResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        ArtifactType artifact_type;
        tensorflow::Status status = metadata_access_object_->FindTypeByName(
            request.type_name(), &artifact_type);
        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        Artifact artifact;
        status = metadata_access_object_->FindArtifactByTypeIdAndArtifactName(
            artifact_type.id(), request.artifact_name(), &artifact);
        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        *response->mutable_artifact() = artifact;
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetExecutionsByType(
    const GetExecutionsByTypeRequest& request,
    GetExecutionsByTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        ExecutionType execution_type;
        tensorflow::Status status = metadata_access_object_->FindTypeByName(
            request.type_name(), &execution_type);
        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        std::vector<Execution> executions;
        status = metadata_access_object_->FindExecutionsByTypeId(
            execution_type.id(), &executions);
        if (tensorflow::errors::IsNotFound(status)) {
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

tensorflow::Status MetadataStore::GetExecutionByTypeAndName(
    const GetExecutionByTypeAndNameRequest& request,
    GetExecutionByTypeAndNameResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        ExecutionType execution_type;
        tensorflow::Status status = metadata_access_object_->FindTypeByName(
            request.type_name(), &execution_type);
        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        Execution execution;
        status = metadata_access_object_->FindExecutionByTypeIdAndExecutionName(
            execution_type.id(), request.execution_name(), &execution);
        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        *response->mutable_execution() = execution;
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetContextsByType(
    const GetContextsByTypeRequest& request,
    GetContextsByTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        ContextType context_type;
        tensorflow::Status status = metadata_access_object_->FindTypeByName(
            request.type_name(), &context_type);
        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        std::vector<Context> contexts;
        status = metadata_access_object_->FindContextsByTypeId(
            context_type.id(), &contexts);
        if (tensorflow::errors::IsNotFound(status)) {
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

tensorflow::Status MetadataStore::GetContextByTypeAndName(
    const GetContextByTypeAndNameRequest& request,
    GetContextByTypeAndNameResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        ContextType context_type;
        tensorflow::Status status = metadata_access_object_->FindTypeByName(
            request.type_name(), &context_type);
        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        Context context;
        status = metadata_access_object_->FindContextByTypeIdAndContextName(
            context_type.id(), request.context_name(), &context);
        if (tensorflow::errors::IsNotFound(status)) {
          return tensorflow::Status::OK();
        } else if (!status.ok()) {
          return status;
        }
        *response->mutable_context() = context;
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::PutAttributionsAndAssociations(
    const PutAttributionsAndAssociationsRequest& request,
    PutAttributionsAndAssociationsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
        for (const Attribution& attribution : request.attributions()) {
          TF_RETURN_IF_ERROR(InsertAttributionIfNotExist(
              attribution.context_id(), attribution.artifact_id(),
              metadata_access_object_.get()));
        }
        for (const Association& association : request.associations()) {
          TF_RETURN_IF_ERROR(InsertAssociationIfNotExist(
              association.context_id(), association.execution_id(),
              metadata_access_object_.get()));
        }
        return tensorflow::Status::OK();
      });
}

tensorflow::Status MetadataStore::GetContextsByArtifact(
    const GetContextsByArtifactRequest& request,
    GetContextsByArtifactResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
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
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
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
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
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
  return transaction_executor_->Execute(
      [this, &request, &response]() -> tensorflow::Status {
        response->Clear();
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
    std::unique_ptr<MetadataAccessObject> metadata_access_object,
    std::unique_ptr<TransactionExecutor> transaction_executor)
    : metadata_source_(std::move(metadata_source)),
      metadata_access_object_(std::move(metadata_access_object)),
      transaction_executor_(std::move(transaction_executor)) {}

}  // namespace ml_metadata
