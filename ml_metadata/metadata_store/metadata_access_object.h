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
#ifndef ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_H_
#define ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_H_

#include <memory>
#include <vector>

#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// Data access object (DAO) for the domain entities (Type, Artifact, Execution,
// Event) defined in metadata_store.proto. It provides a list of query methods
// to store, update, and read entities. It takes a MetadataSourceQueryConfig
// which specifies query instructions on the given MetadataSource.
// Each method is a list of queries which needs to be run within a transaction,
// and the caller can use the methods provided and compose larger transactions
// externally. The caller is responsible to commit or rollback depending on the
// return status of each method: if it succeeds all changes should be committed
// in a transaction, if fails, then there should be no change in the underline
// MetadataSource. It is thread-unsafe.
//
// Usage example:
//
//    SomeConcreteMetadataSource src;
//    MetadataSourceQueryConfig config;
//    std::unique_ptr<MetadataAccessObject> mao;
//    TF_CHECK_OK(CreateMetadataAccessObject(config, &src, &mao));
//
//    if (mao->SomeCRUDMethod(...).ok())
//      TF_CHECK_OK(src.Commit()); // or do more queries
//    else
//      TF_CHECK_OK(src.Rollback());
//
class MetadataAccessObject {
 public:
  virtual ~MetadataAccessObject() = default;

  // default & copy constructors are disallowed.
  MetadataAccessObject() = default;
  MetadataAccessObject(const MetadataAccessObject&) = delete;
  MetadataAccessObject& operator=(const MetadataAccessObject&) = delete;

  // Initializes the metadata source and creates schema. Any existing data in
  // the MetadataSource is dropped.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status InitMetadataSource() = 0;

  // Initializes the metadata source and creates schema.
  // Returns OK and does nothing, if all required schema exist.
  // Returns OK and creates schema, if no schema exists yet.
  // Returns DATA_LOSS error, if the MLMDENv has more than one schema version.
  // Returns ABORTED error, if any required schema is missing.
  // Returns FAILED_PRECONDITION error, if library and db have incompatible
  //   schema versions, and upgrade migrations are not enabled.
  // Returns detailed INTERNAL error, if create schema query execution fails.
  virtual tensorflow::Status InitMetadataSourceIfNotExists(
      bool enable_upgrade_migration = false) = 0;

  // Downgrades the schema to `to_schema_version` in the given metadata source.
  // Returns INVALID_ARGUMENT, if `to_schema_version` is less than 0, or newer
  //   than the library version.
  // Returns FAILED_PRECONDITION, if db schema version is newer than the
  //   library version.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status DowngradeMetadataSource(
      int64 to_schema_version) = 0;

  // Creates a type, returns the assigned type id. A type is one of
  // {ArtifactType, ExecutionType, ContextType}. The id field of the given type
  // is ignored.
  // Returns INVALID_ARGUMENT error, if name field is not given.
  // Returns INVALID_ARGUMENT error, if any property type is unknown.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status CreateType(const ArtifactType& type,
                                        int64* type_id) = 0;
  virtual tensorflow::Status CreateType(const ExecutionType& type,
                                        int64* type_id) = 0;
  virtual tensorflow::Status CreateType(const ContextType& type,
                                        int64* type_id) = 0;

  // Updates an existing type. A type is one of {ArtifactType, ExecutionType,
  // ContextType}. The update should be backward compatible, i.e., existing
  // properties should not be modified, only new properties can be added.
  // Returns INVALID_ARGUMENT error if name field is not given.
  // Returns INVALID_ARGUMENT error, if id field is given but differs from the
  //   the stored type with the same type name.
  // Returns INVALID_ARGUMENT error, if any property type is unknown.
  // Returns ALREADY_EXISTS error, if any property type is different.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status UpdateType(const ArtifactType& type) = 0;
  virtual tensorflow::Status UpdateType(const ExecutionType& type) = 0;
  virtual tensorflow::Status UpdateType(const ContextType& type) = 0;

  // Queries a type by an id. A type is one of
  // {ArtifactType, ExecutionType, ContextType}
  // Returns NOT_FOUND error, if the given type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindTypeById(int64 type_id,
                                          ArtifactType* artifact_type) = 0;
  virtual tensorflow::Status FindTypeById(int64 type_id,
                                          ExecutionType* execution_type) = 0;
  virtual tensorflow::Status FindTypeById(int64 type_id,
                                          ContextType* context_type) = 0;

  // Queries a type by its name. A type is one of
  // {ArtifactType, ExecutionType, ContextType}
  // Returns NOT_FOUND error, if the given name cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindTypeByName(absl::string_view name,
                                            ArtifactType* artifact_type) = 0;
  virtual tensorflow::Status FindTypeByName(absl::string_view name,
                                            ExecutionType* execution_type) = 0;
  virtual tensorflow::Status FindTypeByName(absl::string_view name,
                                            ContextType* context_type) = 0;

  // Returns a list of all known type instances. A type is one of
  // {ArtifactType, ExecutionType, ContextType}
  // Returns NOT_FOUND error, if no types can be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindTypes(
      std::vector<ArtifactType>* artifact_types) = 0;
  virtual tensorflow::Status FindTypes(
      std::vector<ExecutionType>* execution_types) = 0;
  virtual tensorflow::Status FindTypes(
      std::vector<ContextType>* context_types) = 0;

  // Creates an artifact, returns the assigned artifact id. The id field of the
  // artifact is ignored.
  // Returns INVALID_ARGUMENT error, if the ArtifactType is not given.
  // Returns NOT_FOUND error, if the ArtifactType cannot be found.
  // Returns INVALID_ARGUMENT error, if the artifact contains any property
  //  undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the artifact type.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status CreateArtifact(const Artifact& artifact,
                                            int64* artifact_id) = 0;

  // Queries an artifact by an id.
  // Returns NOT_FOUND error, if the given artifact_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindArtifactById(int64 artifact_id,
                                              Artifact* artifact) = 0;

  // Queries artifacts stored in the metadata source
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindArtifacts(
      std::vector<Artifact>* artifacts) = 0;

  // Queries artifacts stored in the metadata source using `options`.
  // `options` is the ListOperationOptions proto message defined
  // in metadata_store.
  // If successfull:
  // 1. `artifacts` is updated with result set of size determined by
  //    max_result_size set in `options`.
  // 2. `next_page_token` is populated with information necessary to fetch next
  //    page of results.
  // RETURNS INVALID_ARGUMENT if the `options` is invalid with one of
  //    the cases:
  // 1. order_by_field is not set or has an unspecified field.
  // 2. Direction of ordering is not specified for the order_by_field.
  // 3. next_page_token cannot be decoded.
  virtual tensorflow::Status ListArtifacts(const ListOperationOptions& options,
                                           std::vector<Artifact>* artifacts,
                                           std::string* next_page_token) = 0;

  // Queries executions stored in the metadata source using `options`.
  // `options` is the ListOperationOptions proto message defined
  // in metadata_store.
  // If successfull:
  // 1. `executions` is updated with result set of size determined by
  //    max_result_size set in `options`.
  // 2. `next_page_token` is populated with information necessary to fetch next
  //    page of results.
  // RETURNS INVALID_ARGUMENT if the `options` is invalid with one of
  //    the cases:
  // 1. order_by_field is not set or has an unspecified field.
  // 2. Direction of ordering is not specified for the order_by_field.
  // 3. next_page_token cannot be decoded.
  virtual tensorflow::Status ListExecutions(const ListOperationOptions& options,
                                            std::vector<Execution>* executions,
                                            std::string* next_page_token) = 0;

  // Queries contexts stored in the metadata source using `options`.
  // `options` is the ListOperationOptions proto message defined
  // in metadata_store.
  // If successfull:
  // 1. `contexts` is updated with result set of size determined by
  //    max_result_size set in `options`.
  // 2. `next_page_token` is populated with information necessary to fetch next
  //    page of results.
  // RETURNS INVALID_ARGUMENT if the `options` is invalid with one of
  //    the cases:
  // 1. order_by_field is not set or has an unspecified field.
  // 2. Direction of ordering is not specified for the order_by_field.
  // 3. next_page_token cannot be decoded.
  virtual tensorflow::Status ListContexts(const ListOperationOptions& options,
                                          std::vector<Context>* contexts,
                                          std::string* next_page_token) = 0;

  // Queries an artifact by its type_id and name.
  // Returns NOT_FOUND error, if no artifact can be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindArtifactByTypeIdAndArtifactName(
      int64 artifact_type_id, absl::string_view name, Artifact* artifact) = 0;

  // Queries artifacts by a given type_id.
  // Returns NOT_FOUND error, if the given artifact_type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindArtifactsByTypeId(
      int64 artifact_type_id, std::vector<Artifact>* artifacts) = 0;

  // Queries artifacts by a given uri with exact match.
  // Returns NOT_FOUND error, if the given uri cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindArtifactsByURI(
      absl::string_view uri, std::vector<Artifact>* artifacts) = 0;

  // Updates an artifact.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no artifact is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ArtifactType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status UpdateArtifact(const Artifact& artifact) = 0;

  // Creates an execution, returns the assigned execution id. The id field of
  // the execution is ignored.
  // Returns INVALID_ARGUMENT error, if the ExecutionType is not given.
  // Returns NOT_FOUND error, if the ExecutionType cannot be found.
  // Returns INVALID_ARGUMENT error, if the execution contains any property
  //   undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the ExecutionType.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status CreateExecution(const Execution& execution,
                                             int64* execution_id) = 0;

  // Queries an entity by an id.
  // Returns NOT_FOUND error, if the given execution_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindExecutionById(int64 execution_id,
                                               Execution* execution) = 0;

  // Queries executions stored in the metadata source
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindExecutions(
      std::vector<Execution>* executions) = 0;

  // Queries an execution by its type_id and name.
  // Returns NOT_FOUND error, if no execution can be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindExecutionByTypeIdAndExecutionName(
      int64 execution_type_id, absl::string_view name,
      Execution* execution) = 0;

  // Queries executions by a given type_id.
  // Returns NOT_FOUND error, if the given execution_type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindExecutionsByTypeId(
      int64 execution_type_id, std::vector<Execution>* executions) = 0;

  // Updates an execution.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no execution is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ExecutionType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status UpdateExecution(const Execution& execution) = 0;

  // Creates a context, returns the assigned context id. The id field of the
  // context is ignored. The name field of the context must not be empty and it
  // should be unique in the same ContextType.
  // Returns INVALID_ARGUMENT error, if the ContextType is not given.
  // Returns NOT_FOUND error, if the ContextType cannot be found.
  // Returns INVALID_ARGUMENT error, if the context name is empty.
  // Returns INVALID_ARGUMENT error, if the context contains any property
  //  undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the context type.
  // Returns ALREADY_EXISTS error, if the ContextType has context with the name.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status CreateContext(const Context& context,
                                           int64* context_id) = 0;

  // Queries a context by an id.
  // Returns NOT_FOUND error, if the given context_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindContextById(int64 context_id,
                                             Context* context) = 0;

  // Queries contexts stored in the metadata source
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindContexts(std::vector<Context>* contexts) = 0;

  // Queries contexts by a given type_id.
  // Returns NOT_FOUND error, if no context can be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindContextsByTypeId(
      int64 context_type_id, std::vector<Context>* contexts) = 0;

  // Queries a context by a type_id and a context name.
  // Returns NOT_FOUND error, if no context can be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status FindContextByTypeIdAndContextName(
      int64 type_id, absl::string_view name, Context* context) = 0;

  // Updates a context.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no context is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ContextType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status UpdateContext(const Context& context) = 0;

  // Creates an event, returns the assigned event id. If the event occurrence
  // time is not given, the insertion time is used.
  // TODO(huimiao) Allow to have a unknown event time.
  // Returns INVALID_ARGUMENT error, if no artifact matches the artifact_id.
  // Returns INVALID_ARGUMENT error, if no execution matches the execution_id.
  // Returns INVALID_ARGUMENT error, if the type field is UNKNOWN.
  virtual tensorflow::Status CreateEvent(const Event& event,
                                         int64* event_id) = 0;

  // Queries the events associated with a collection of artifact_ids.
  // Returns INVALID_ARGUMENT error, if the `events` is null.
  virtual tensorflow::Status FindEventsByArtifacts(
      const std::vector<int64>& artifact_ids, std::vector<Event>* events) = 0;

  // Queries the events associated with a collection of execution_ids.
  // Returns INVALID_ARGUMENT error, if the `events` is null.
  virtual tensorflow::Status FindEventsByExecutions(
      const std::vector<int64>& execution_ids, std::vector<Event>* events) = 0;

  // Creates an association, returns the assigned association id.
  // Returns INVALID_ARGUMENT error, if no context matches the context_id.
  // Returns INVALID_ARGUMENT error, if no execution matches the execution_id.
  // Returns INTERNAL error, if the same association already exists.
  virtual tensorflow::Status CreateAssociation(const Association& association,
                                               int64* association_id) = 0;

  // Queries the contexts that an execution_id is associated with.
  // Returns INVALID_ARGUMENT error, if the `contexts` is null.
  virtual tensorflow::Status FindContextsByExecution(
      int64 execution_id, std::vector<Context>* contexts) = 0;

  // Queries the executions associated with a context_id.
  // Returns INVALID_ARGUMENT error, if the `executions` is null.
  virtual tensorflow::Status FindExecutionsByContext(
      int64 context_id, std::vector<Execution>* executions) = 0;

  // Creates an attribution, returns the assigned attribution id.
  // Returns INVALID_ARGUMENT error, if no context matches the context_id.
  // Returns INVALID_ARGUMENT error, if no artifact matches the artifact_id.
  // Returns AlreadyExists error, if the same attribution already exists.
  virtual tensorflow::Status CreateAttribution(const Attribution& attribution,
                                               int64* attribution_id) = 0;

  // Queries the contexts that an artifact_id is attributed to.
  // Returns INVALID_ARGUMENT error, if the `contexts` is null.
  virtual tensorflow::Status FindContextsByArtifact(
      int64 artifact_id, std::vector<Context>* contexts) = 0;

  // Queries the artifacts attributed to a context_id.
  // Returns INVALID_ARGUMENT error, if the `artifacts` is null.
  virtual tensorflow::Status FindArtifactsByContext(
      int64 context_id, std::vector<Artifact>* artifacts) = 0;

  // Resolves the schema version stored in the metadata source. The `db_version`
  // is set to 0, if it is a 0.13.2 release pre-existing database.
  // Returns DATA_LOSS error, if schema version info table exists but its value
  //   cannot be resolved from the database.
  // Returns NOT_FOUND error, if the database is empty.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status GetSchemaVersion(int64* db_version) = 0;

  // The version of the current query config or source. Increase the version by
  // 1 in any CL that includes physical schema changes and provides a migration
  // function that uses a list migration queries. The database stores it to
  // indicate the current database version. When metadata source is connected to
  // the database it can compare the given library `schema_version` in the query
  // config with the `schema_version` stored in the database, and migrate the
  // database if needed.
  virtual int64 GetLibraryVersion() = 0;

};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_H_
