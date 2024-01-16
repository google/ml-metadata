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
#ifndef ML_METADATA_METADATA_STORE_METADATA_STORE_H_
#define ML_METADATA_METADATA_STORE_METADATA_STORE_H_

#include <memory>

#include "absl/status/status.h"
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/metadata_store_service_interface.h"
#include "ml_metadata/metadata_store/transaction_executor.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"

namespace ml_metadata {

// A metadata store.
// Implements the API specified in MetadataStoreService.
// Each method is an atomic operation.
class MetadataStore : public MetadataStoreServiceInterface {
 public:
  // Factory method that creates a MetadataStore in result. The result is owned
  // by the caller, and metadata_source is owned by result.
  // If the return value is ok, 'result' is populated with an object that can be
  // used to access metadata with the given config and metadata_source.
  // Returns INVALID_ARGUMENT error, if query_config is not valid.
  // Returns INVALID_ARGUMENT error, if migration options are invalid.
  // Returns CANCELLED error, if downgrade migration is performed.
  // Returns detailed INTERNAL error, if the MetadataSource cannot be connected.
  static absl::Status Create(
      const MetadataSourceQueryConfig& query_config,
      const MigrationOptions& migration_options,
      std::unique_ptr<MetadataSource> metadata_source,
      std::unique_ptr<TransactionExecutor> transaction_executor,
      std::unique_ptr<MetadataStore>* result);

  // Initializes the metadata source and creates schema. Any existing data in
  // the metadata is dropped.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status InitMetadataStore();

  // Initializes the metadata source and creates schema if it does not exist.
  // Returns OK and does nothing, if all required schema exist.
  // Returns OK and creates schema, if no schema exists yet.
  // Returns DATA_LOSS error, if any required schema is missing.
  // Returns FAILED_PRECONDITION error, if library and db have incompatible
  //   schema versions, and upgrade migrations are not enabled.
  // Returns detailed INTERNAL error, if create schema query execution fails.
  absl::Status InitMetadataStoreIfNotExists(
      bool enable_upgrade_migration = false);



  // Inserts or updates a ArtifactType/ExecutionType/ContextType.
  //
  // A type has a set of strong typed properties describing the schema of any
  // stored instance associated with that type. A type is identified by a name
  // and an optional version.
  //
  // Type Creation:
  // If no type exists in the database with the given identifier
  // (name, version), it creates a new type and returns the type_id.
  //
  // Type Evolution:
  // If the request type with the same (name, version) already exists
  // (let's call it stored_type), the method enforces the stored_type can be
  // updated only when the request type is backward compatible for the already
  // stored instances.
  //
  // Backwards compatibility is violated if:
  //
  //   a) there is a property where the request type and stored_type have
  //      different value type (e.g., int vs. string)
  //   b) `can_add_fields = false` and the request type has a new property that
  //      is not stored.
  //   c) `can_omit_fields = false` and stored_type has an existing property
  //      that is not provided in the request type.
  //
  // If non-backward type change is required in the application, e.g.,
  // deprecate properties, re-purpose property name, change value types,
  // a new type can be created with a different (name, version) identifier.
  // Note the type version is optional, and a version value with empty string
  // is treated as unset.
  //
  // For the fields of PutArtifactTypeRequest:
  //
  //   can_add_fields:
  //     when set to true, new properties can be added;
  //     when set to false, returns ALREADY_EXISTS if the request type has
  //     properties that are not in stored_type.
  //
  //   can_omit_fields:
  //     when set to true, stored properties can be omitted in the request type;
  //     when set to false, returns ALREADY_EXISTS if the stored_type has
  //     properties not in the request type.
  //
  // Returns ALREADY_EXISTS error in the case listed above.
  // Returns INVALID_ARGUMENT error, if the given type has no name, or any
  //     property value type is unknown.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status PutArtifactType(const PutArtifactTypeRequest& request,
                               PutArtifactTypeResponse* response) override;

  absl::Status PutExecutionType(const PutExecutionTypeRequest& request,
                                PutExecutionTypeResponse* response) override;

  absl::Status PutContextType(const PutContextTypeRequest& request,
                              PutContextTypeResponse* response) override;

  // Gets an ArtifactType/ExecutionType/ContextType by name.
  // If no type exists, returns a NOT_FOUND error.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetArtifactType(const GetArtifactTypeRequest& request,
                               GetArtifactTypeResponse* response) override;

  absl::Status GetExecutionType(const GetExecutionTypeRequest& request,
                                GetExecutionTypeResponse* response) override;

  absl::Status GetContextType(const GetContextTypeRequest& request,
                              GetContextTypeResponse* response) override;

  // Bulk inserts a list of types atomically. The types could be ArtifactType,
  // ExecutionType, or ContextType.
  //
  // Returns ALREADY_EXISTS if any type update is backward incompatible. Please
  //     see comments in PutArtifactType for the detailed conditions.
  // Returns INVALID_ARGUMENT if any given type message has no name, or its
  //     property value type is unknown.
  // Returns detailed INTERNAL error, if query execution fails.
  // TODO(huimiao) Surface the API in python/Go/gRPC.
  absl::Status PutTypes(const PutTypesRequest& request,
                        PutTypesResponse* response) override;

  // Gets all ArtifactTypes/ExecutionTypes/ContextTypes.
  // If no types found, it returns OK and empty response.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetArtifactTypes(const GetArtifactTypesRequest& request,
                                GetArtifactTypesResponse* response) override;

  absl::Status GetExecutionTypes(const GetExecutionTypesRequest& request,
                                 GetExecutionTypesResponse* response) override;

  absl::Status GetContextTypes(const GetContextTypesRequest& request,
                               GetContextTypesResponse* response) override;

  // Gets a list of ArtifactTypes/ExecutionTypes/ContextTypes by ids.
  // If no types with an ID exists, the type is skipped.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetArtifactTypesByID(
      const GetArtifactTypesByIDRequest& request,
      GetArtifactTypesByIDResponse* response) override;

  absl::Status GetExecutionTypesByID(
      const GetExecutionTypesByIDRequest& request,
      GetExecutionTypesByIDResponse* response) override;

  absl::Status GetContextTypesByID(
      const GetContextTypesByIDRequest& request,
      GetContextTypesByIDResponse* response) override;

  // Gets a list of types using their external ids in `external_ids` and
  // appends the types to `artifact_types`/`execution_types`/`context_types`.
  // Returns absl::OkStatus() if types are fetched successfully.
  // Note: The `types` has been deduped.
  // Returns whatever found when a part of `external_ids` is non-existing.
  // Returns NOT_FOUND error if all the given `external_ids` are not found.
  // Returns INVALID_ARGUMENT if `external_ids` is empty or
  // `artifact_types`/`execution_types`/`context_types` is not empty.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetArtifactTypesByExternalIds(
      const GetArtifactTypesByExternalIdsRequest& request,
      GetArtifactTypesByExternalIdsResponse* response) override;

  absl::Status GetExecutionTypesByExternalIds(
      const GetExecutionTypesByExternalIdsRequest& request,
      GetExecutionTypesByExternalIdsResponse* response) override;

  absl::Status GetContextTypesByExternalIds(
      const GetContextTypesByExternalIdsRequest& request,
      GetContextTypesByExternalIdsResponse* response) override;

  // Inserts and updates artifacts in request into the database.
  // If artifact_id is specified, an existing artifact is updated.
  // If artifact_id is not specified, a new artifact is created.
  // It is not guaranteed that the created or updated artifacts will share the
  // same `create_time_since_epoch` or `last_update_time_since_epoch` timestamp.
  //
  // Returns a list of artifact ids index-aligned with the input.
  // Returns INVALID_ARGUMENT error, if no artifact is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ArtifactType on file.
  // Returns FAILED_PRECONDITION error, if the request set
  // options.abort_if_latest_updated_time_changed, and the stored artifact has
  // different latest_updated_time.
  // Returns ALREADY_EXISTS error, if the name exists in the artifact_type.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status PutArtifacts(const PutArtifactsRequest& request,
                            PutArtifactsResponse* response) override;

  // Inserts and updates executions in the request into the database.
  // If execution_id is specified, an existing execution is updated.
  // If execution_id is not specified, a new execution is created.
  // It is not guaranteed that the created or updated executions will share the
  // same `create_time_since_epoch` or `last_update_time_since_epoch`
  // timestamps.
  //
  // Returns a list of execution ids index-aligned with the input.
  // Returns INVALID_ARGUMENT error, if no execution is found with a given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ExecutionType on file.
  // Returns ALREADY_EXISTS error, if the name exists in the artifact_type.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status PutExecutions(const PutExecutionsRequest& request,
                             PutExecutionsResponse* response) override;

  // Inserts and updates contexts in the request into the database. Context
  // must have none empty name, and it should be unique of a ContextType.
  // If context_id is specified, an existing context is updated.
  // If context_id is not specified, a new execution is created.
  // It is not guaranteed that the created or updated contexts will share the
  // same `create_time_since_epoch` or `last_update_time_since_epoch`
  // timestamps.
  //
  // Returns a list of context ids index-aligned with the input.
  // Returns INVALID_ARGUMENT error, if no context is found with a given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if name is empty.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ContextType on file.
  // Returns ALREADY_EXISTS error, if the name exists in the context_type.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status PutContexts(const PutContextsRequest& request,
                           PutContextsResponse* response) override;

  // Inserts events into the database.
  //
  // The execution_id and artifact_id must already exist.
  // Once created, events cannot be modified.
  // If the event timestamp is not set, it will be set to the current time.
  // request.events is a list of events to insert or update.
  // response is empty.
  // It is not guaranteed that the created or updated events will share the
  // same `milliseconds_since_epoch` timestamps.
  //
  // Returns INVALID_ARGUMENT error, if no artifact matches the artifact_id.
  // Returns INVALID_ARGUMENT error, if no execution matches the execution_id.
  // Returns INVALID_ARGUMENT error, if the type field is UNKNOWN.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status PutEvents(const PutEventsRequest& request,
                         PutEventsResponse* response) override;

  // Inserts or updates an Execution and its input and output artifacts and
  // events atomically. The request includes the state changes of the Artifacts
  // used or generated by the Execution, as well as the input/output Event.
  //
  // If an execution_id, artifact_id or context_id is specified, it is an update
  // , otherwise it does an insertion. For insertion, type must be specified.
  // If the event timestamp is not set, it will be set to the current time.
  // It is not guaranteed that the created or updated executions, artifacts,
  // contexts and events will share the same `create_time_since_epoch`,
  // `last_update_time_since_epoch`, or `milliseconds_since_epoch` timestamps.
  //
  // Returns a list of execution, artifact, and context ids index-aligned with
  // the input.
  // Returns INVALID_ARGUMENT error, if no artifact, execution, or context
  // matches the id.
  // Returns INVALID_ARGUMENT error, if type_id is different from stored one.
  // Returns INVALID_ARGUMENT error, if property names and types do not align.
  // Returns INVALID_ARGUMENT error, if the event.type field is UNKNOWN.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status PutExecution(const PutExecutionRequest& request,
                            PutExecutionResponse* response) override;

  // Inserts or updates a lineage subgraph (i.e. a collection of event edges
  // and its executions, artifacts, and related contexts) atomically. All
  // contexts will be associated to every execution and attributed to every
  // artifact provided in the request.
  //
  // It is not guaranteed that the created or updated executions, artifacts,
  // contexts and events will share the same `create_time_since_epoch`,
  // `last_update_time_since_epoch`, or `milliseconds_since_epoch` timestamps.
  //
  // Returns a list of execution, artifact, and context ids index-aligned with
  // the input.
  // Returns INVALID_ARGUMENT error, if no artifact, execution, or context
  // matches the id.
  // Returns INVALID_ARGUMENT error, if event_edge has no event.
  // Returns INVALID_ARGUMENT error, if type_id is different from stored one.
  // Returns INVALID_ARGUMENT error, if the event.type field is UNKNOWN.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns OUT_OF_BOUND error, if event_edge has an out of bound
  // execution_index or artifact_index.
  // TODO(b/217390865): OSS after this API is tested and stable.
  absl::Status PutLineageSubgraph(
      const PutLineageSubgraphRequest& request,
      PutLineageSubgraphResponse* response) override;


  // Gets all events with matching execution ids.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetEventsByExecutionIDs(
      const GetEventsByExecutionIDsRequest& request,
      GetEventsByExecutionIDsResponse* response) override;

  // Gets all events with matching artifact ids.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetEventsByArtifactIDs(
      const GetEventsByArtifactIDsRequest& request,
      GetEventsByArtifactIDsResponse* response) override;

  // Gets a list of artifacts by ID.
  // If no artifact with an ID exists, the artifact is skipped.
  // Sets the error field if any other internal errors are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetArtifactsByID(const GetArtifactsByIDRequest& request,
                                GetArtifactsByIDResponse* response) override;

  // Gets Artifacts using list options.
  // If option is not set in the request, then all Artifacts are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetArtifacts(const GetArtifactsRequest& request,
                            GetArtifactsResponse* response) override;

  // Gets a list of artifacts using external_ids.
  // |external_ids| is a list of non-null strings for the given external ids.
  // Returns INVALID_ARGUMENT error, if any of the |external_ids| is empty.
  // Returns NOT_FOUND error, if the given external_ids are not found.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetArtifactsByExternalIds(
      const GetArtifactsByExternalIdsRequest& request,
      GetArtifactsByExternalIdsResponse* response) override;

  // Gets all the artifacts of a given type. If no artifacts found, it returns
  // OK and empty response.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetArtifactsByType(
      const GetArtifactsByTypeRequest& request,
      GetArtifactsByTypeResponse* response) override;

  // Gets the artifact of a given type and name. If no artifact found, it
  // returns OK and empty response.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetArtifactByTypeAndName(
      const GetArtifactByTypeAndNameRequest& request,
      GetArtifactByTypeAndNameResponse* response) override;

  // Gets all the artifacts matching the given URIs. If no artifacts found, it
  // returns OK and empty response.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetArtifactsByURI(const GetArtifactsByURIRequest& request,
                                 GetArtifactsByURIResponse* response) override;

  // Gets a list of executions by ID.
  // If no execution with an ID exists, the execution is skipped.
  // Sets the error field if any other internal errors are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetExecutionsByID(const GetExecutionsByIDRequest& request,
                                 GetExecutionsByIDResponse* response) override;

  // Gets Executions using list options.
  // If option is not set in the request, then all Executions are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetExecutions(const GetExecutionsRequest& request,
                             GetExecutionsResponse* response) override;

  // Gets all the executions of a given type. If no executions found, it returns
  // OK and empty response.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetExecutionsByType(
      const GetExecutionsByTypeRequest& request,
      GetExecutionsByTypeResponse* response) override;

  // Gets the execution of a given type and name. If no execution found, it
  // returns OK and empty response.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetExecutionByTypeAndName(
      const GetExecutionByTypeAndNameRequest& request,
      GetExecutionByTypeAndNameResponse* response) override;

  // Gets a list of executions using external_ids.
  // |external_ids| is a list of non-null strings for the given external ids.
  // Returns INVALID_ARGUMENT error, if any of the |external_ids| is empty.
  // Returns NOT_FOUND error, if the given external_ids are not found.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetExecutionsByExternalIds(
      const GetExecutionsByExternalIdsRequest& request,
      GetExecutionsByExternalIdsResponse* response) override;

  // Gets a list of contexts by ID.
  // If no context with an ID exists, the context is skipped.
  // Sets the error field if any other internal errors are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetContextsByID(const GetContextsByIDRequest& request,
                               GetContextsByIDResponse* response) override;

  // Retrieve Contexts using list options.
  // If option is not set in the request, then all Contexts are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetContexts(const GetContextsRequest& request,
                           GetContextsResponse* response) override;

  // Gets all the contexts of a given type. If no contexts found, it returns
  // OK and empty response.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetContextsByType(const GetContextsByTypeRequest& request,
                                 GetContextsByTypeResponse* response) override;

  // Gets the context of a given type and name. If no context found, it returns
  // OK and empty response. If multiple contexts are found with the same type
  // and name, the query execution fails.
  absl::Status GetContextByTypeAndName(
      const GetContextByTypeAndNameRequest& request,
      GetContextByTypeAndNameResponse* response) override;

  // Gets a list of contexts using external_ids.
  // |external_ids| is a list of non-null strings for the given external ids.
  // Returns INVALID_ARGUMENT error, if any of the |external_ids| is empty.
  // Returns NOT_FOUND error, if the given external_ids are not found.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetContextsByExternalIds(
      const GetContextsByExternalIdsRequest& request,
      GetContextsByExternalIdsResponse* response) override;

  // Inserts attribution and association relationships in the database.
  // The context_id, artifact_id, and execution_id must already exist.
  // If the relationship exists, this call does nothing. Once added, the
  // relationships cannot be modified.
  //
  // Returns INVALID_ARGUMENT error, if no artifact, execution or context
  //   matches the given id.
  absl::Status PutAttributionsAndAssociations(
      const PutAttributionsAndAssociationsRequest& request,
      PutAttributionsAndAssociationsResponse* response) override;

  // Inserts parent contexts in the database.
  // The `child_id` and `parent_id` in every parent context must already exist.
  // Returns INVALID_ARGUMENT error, if no context matches the child_id.
  // Returns INVALID_ARGUMENT error, if no context matches the parent_id.
  // Returns ALREADY_EXISTS error, if the same parent context already exists.
  absl::Status PutParentContexts(const PutParentContextsRequest& request,
                                 PutParentContextsResponse* response) override;

  // Gets all context that an artifact is attributed to.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetContextsByArtifact(
      const GetContextsByArtifactRequest& request,
      GetContextsByArtifactResponse* response) override;

  // Gets all context that an execution is associated with.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetContextsByExecution(
      const GetContextsByExecutionRequest& request,
      GetContextsByExecutionResponse* response) override;

  // Gets all direct artifacts using list options that a context attributes to.
  // If option is not set in the request, then all artifacts are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetArtifactsByContext(
      const GetArtifactsByContextRequest& request,
      GetArtifactsByContextResponse* response) override;

  // Gets direct executions using list options that a context associates with.
  // If option is not set in the request, then all executions are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetExecutionsByContext(
      const GetExecutionsByContextRequest& request,
      GetExecutionsByContextResponse* response) override;

  // Gets all parent contexts of a context.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetParentContextsByContext(
      const GetParentContextsByContextRequest& request,
      GetParentContextsByContextResponse* response) override;

  // Gets all children contexts of a context.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetChildrenContextsByContext(
      const GetChildrenContextsByContextRequest& request,
      GetChildrenContextsByContextResponse* response) override;

  // Gets all parent contexts of a list of contexts.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetParentContextsByContexts(
      const GetParentContextsByContextsRequest& request,
      GetParentContextsByContextsResponse* response) override;

  // Gets all children contexts of a list of contexts.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status GetChildrenContextsByContexts(
      const GetChildrenContextsByContextsRequest& request,
      GetChildrenContextsByContextsResponse* response) override;


  // TODO(b/283852485): Deprecate GetLineageGraph API after migration to
  // GetLineageSubgraph API.
  // The method is deprecated. Please use `GetLineageSubgraph` instead.
  absl::Status GetLineageGraph(const GetLineageGraphRequest& request,
                               GetLineageGraphResponse* response) override;

  // A method to perform lineage graph tracing from a list of interested
  // nodes. Please refer to metadata_store_service.proto for details.
  // Returns INVALID_ARGUMENT error, if `starting_nodes` is not specified in
  // the request.
  // Returns INVALID_ARGUMENT error, if `starting_nodes.filter_query` is
  // unspecified or invalid in the request.
  // Returns detailed INTERNAL error, if the operation fails.
  absl::Status GetLineageSubgraph(
      const GetLineageSubgraphRequest& request,
      GetLineageSubgraphResponse* response) override;



 private:
  // To construct the object, see Create(...).
  MetadataStore(std::unique_ptr<MetadataSource> metadata_source,
                std::unique_ptr<MetadataAccessObject> metadata_access_object,
                std::unique_ptr<TransactionExecutor> transaction_executor);

  std::unique_ptr<MetadataSource> metadata_source_;
  std::unique_ptr<MetadataAccessObject> metadata_access_object_;
  std::unique_ptr<TransactionExecutor> transaction_executor_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_METADATA_STORE_H_
