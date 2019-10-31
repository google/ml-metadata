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
//    TF_CHECK_OK(MetadataAccessObject::Create(config, &src, &mao));
//
//    if (mao->SomeCRUDMethod(...).ok())
//      TF_CHECK_OK(src.Commit()); // or do more queries
//    else
//      TF_CHECK_OK(src.Rollback());
//
// TODO(huimiao) Refactor it to be an abstract interface when the CRUD methods
// are more stable. Having an abstract interface allows
//   a) alternative RDBMS schema designs
//   b) non-SQL backend
//   c) easier to test
class MetadataAccessObject {
 public:
  // Factory method, if the return value is ok, 'result' is populated with an
  // object that can be used to access metadata with the given config and
  // metadata_source. The caller is responsible to own a MetadataSource, and the
  // MetadataAccessObject connects and execute queries with the MetadataSource.
  // Returns INVALID_ARGUMENT error, if query_config is not valid.
  // Returns detailed INTERNAL error, if the MetadataSource cannot be connected.
  static tensorflow::Status Create(
      const MetadataSourceQueryConfig& query_config,
      MetadataSource* metadata_source,
      std::unique_ptr<MetadataAccessObject>* result);

  ~MetadataAccessObject() = default;

  // default & copy constructors are disallowed.
  MetadataAccessObject() = delete;
  MetadataAccessObject(const MetadataAccessObject&) = delete;
  MetadataAccessObject& operator=(const MetadataAccessObject&) = delete;

  // Initializes the metadata source and creates schema. Any existing data in
  // the MetadataSource is dropped.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status InitMetadataSource();

  // Initializes the metadata source and creates schema.
  // Returns OK and does nothing, if all required schema exist.
  // Returns OK and creates schema, if no schema exists yet.
  // Returns DATA_LOSS error, if any required schema is missing.
  // Returns FAILED_PRECONDITION error, if library and db have incompatible
  //   schema versions, and upgrade migrations are disallowed.
  // Returns detailed INTERNAL error, if create schema query execution fails.
  tensorflow::Status InitMetadataSourceIfNotExists(
      bool disable_upgrade_migration = false);

  // Downgrades the schema to `to_schema_version` in the given metadata source.
  // Returns INVALID_ARGUMENT, if `to_schema_version` is less than 0, or newer
  //   than the library version.
  // Returns FAILED_PRECONDITION, if db schema version is newer than the
  //   library version.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status DowngradeMetadataSource(int64 to_schema_version);

  // Creates a type, returns the assigned type id. A type is one of
  // {ArtifactType, ExecutionType, ContextType}. The id field of the given type
  // is ignored.
  // Returns INVALID_ARGUMENT error, if name field is not given.
  // Returns INVALID_ARGUMENT error, if any property type is unknown.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status CreateType(const ArtifactType& type, int64* type_id);
  tensorflow::Status CreateType(const ExecutionType& type, int64* type_id);
  tensorflow::Status CreateType(const ContextType& type, int64* type_id);

  // Updates an existing type. A type is one of {ArtifactType, ExecutionType,
  // ContextType}. The update should be backward compatible, i.e., existing
  // properties should not be modified, only new properties can be added.
  // Returns INVALID_ARGUMENT error if name field is not given.
  // Returns INVALID_ARGUMENT error, if id field is given but differs from the
  //   the stored type with the same type name.
  // Returns INVALID_ARGUMENT error, if any property type is unknown.
  // Returns ALREADY_EXISTS error, if any property type is different.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status UpdateType(const ArtifactType& type);
  tensorflow::Status UpdateType(const ExecutionType& type);
  tensorflow::Status UpdateType(const ContextType& type);

  // Queries a type by an id. A type is one of
  // {ArtifactType, ExecutionType, ContextType}
  // Returns NOT_FOUND error, if the given type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindTypeById(int64 type_id, ArtifactType* artifact_type);
  tensorflow::Status FindTypeById(int64 type_id, ExecutionType* execution_type);
  tensorflow::Status FindTypeById(int64 type_id, ContextType* context_type);

  // Queries a type by its name. A type is one of
  // {ArtifactType, ExecutionType, ContextType}
  // Returns NOT_FOUND error, if the given name cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindTypeByName(absl::string_view name,
                                    ArtifactType* artifact_type);
  tensorflow::Status FindTypeByName(absl::string_view name,
                                    ExecutionType* execution_type);
  tensorflow::Status FindTypeByName(absl::string_view name,
                                    ContextType* context_type);

  // Returns a list of all known type instances. A type is one of
  // {ArtifactType, ExecutionType, ContextType}
  // Returns NOT_FOUND error, if no types can be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindTypes(std::vector<ArtifactType>* artifact_types);
  tensorflow::Status FindTypes(std::vector<ExecutionType>* execution_types);
  tensorflow::Status FindTypes(std::vector<ContextType>* context_types);

  // Creates an artifact, returns the assigned artifact id. The id field of the
  // artifact is ignored.
  // Returns INVALID_ARGUMENT error, if the ArtifactType is not given.
  // Returns NOT_FOUND error, if the ArtifactType cannot be found.
  // Returns INVALID_ARGUMENT error, if the artifact contains any property
  //  undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the artifact type.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status CreateArtifact(const Artifact& artifact,
                                    int64* artifact_id);

  // Queries an artifact by an id.
  // Returns NOT_FOUND error, if the given artifact_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindArtifactById(int64 artifact_id, Artifact* artifact);

  // Queries artifacts stored in the metadata source
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindArtifacts(std::vector<Artifact>* artifacts);

  // Queries artifacts by a given type_id.
  // Returns NOT_FOUND error, if the given artifact_type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindArtifactsByTypeId(int64 artifact_type_id,
                                           std::vector<Artifact>* artifacts);

  // Queries artifacts by a given uri with exact match.
  // Returns NOT_FOUND error, if the given uri cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindArtifactsByURI(absl::string_view uri,
                                        std::vector<Artifact>* artifacts);

  // Updates an artifact.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no artifact is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ArtifactType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status UpdateArtifact(const Artifact& artifact);

  // Creates an execution, returns the assigned execution id. The id field of
  // the execution is ignored.
  // Returns INVALID_ARGUMENT error, if the ExecutionType is not given.
  // Returns NOT_FOUND error, if the ExecutionType cannot be found.
  // Returns INVALID_ARGUMENT error, if the execution contains any property
  //   undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the ExecutionType.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status CreateExecution(const Execution& execution,
                                     int64* execution_id);

  // Queries an entity by an id.
  // Returns NOT_FOUND error, if the given execution_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindExecutionById(int64 execution_id,
                                       Execution* execution);

  // Queries executions stored in the metadata source
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindExecutions(std::vector<Execution>* executions);

  // Queries executions by a given type_id.
  // Returns NOT_FOUND error, if the given execution_type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindExecutionsByTypeId(int64 execution_type_id,
                                            std::vector<Execution>* executions);

  // Updates an execution.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no execution is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ExecutionType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status UpdateExecution(const Execution& execution);

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
  tensorflow::Status CreateContext(const Context& context, int64* context_id);

  // Queries a context by an id.
  // Returns NOT_FOUND error, if the given context_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindContextById(int64 context_id, Context* context);

  // Queries contexts stored in the metadata source
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindContexts(std::vector<Context>* contexts);

  // Queries contexts by a given type_id.
  // Returns NOT_FOUND error, if the given context_type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindContextsByTypeId(int64 context_type_id,
                                          std::vector<Context>* contexts);

  // Updates a context.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no context is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ContextType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status UpdateContext(const Context& context);

  // Creates an event, returns the assigned event id. If the event occurrence
  // time is not given, the insertion time is used.
  // TODO(huimiao) Allow to have a unknown event time.
  // Returns INVALID_ARGUMENT error, if no artifact matches the artifact_id.
  // Returns INVALID_ARGUMENT error, if no execution matches the execution_id.
  // Returns INVALID_ARGUMENT error, if the type field is UNKNOWN.
  tensorflow::Status CreateEvent(const Event& event, int64* event_id);

  // Queries the events associated with an artifact_id.
  // Returns INVALID_ARGUMENT error, if the `events` is null.
  // Returns NOT_FOUND error, if there are no events found with the `artifact`.
  tensorflow::Status FindEventsByArtifact(int64 artifact_id,
                                          std::vector<Event>* events);

  // Queries the events associated with an execution_id.
  // Returns INVALID_ARGUMENT error, if the `events` is null.
  // Returns NOT_FOUND error, if there are no events found with the `execution`.
  tensorflow::Status FindEventsByExecution(int64 execution_id,
                                           std::vector<Event>* events);

  // Creates an association, returns the assigned association id.
  // Returns INVALID_ARGUMENT error, if no context matches the context_id.
  // Returns INVALID_ARGUMENT error, if no execution matches the execution_id.
  // Returns INTERNAL error, if the same association already exists.
  tensorflow::Status CreateAssociation(const Association& association,
                                       int64* association_id);

  // Queries the contexts that an execution_id is associated with.
  // Returns INVALID_ARGUMENT error, if the `contexts` is null.
  tensorflow::Status FindContextsByExecution(int64 execution_id,
                                             std::vector<Context>* contexts);

  // Queries the executions associated with a context_id.
  // Returns INVALID_ARGUMENT error, if the `executions` is null.
  tensorflow::Status FindExecutionsByContext(
      int64 context_id, std::vector<Execution>* executions);

  // Creates an attribution, returns the assigned attribution id.
  // Returns INVALID_ARGUMENT error, if no context matches the context_id.
  // Returns INVALID_ARGUMENT error, if no artifact matches the artifact_id.
  // Returns INTERNAL error, if the same attribution already exists.
  tensorflow::Status CreateAttribution(const Attribution& attribution,
                                       int64* attribution_id);

  // Queries the contexts that an artifact_id is attributed to.
  // Returns INVALID_ARGUMENT error, if the `contexts` is null.
  tensorflow::Status FindContextsByArtifact(int64 artifact_id,
                                            std::vector<Context>* contexts);

  // Queries the artifacts attributed to a context_id.
  // Returns INVALID_ARGUMENT error, if the `artifacts` is null.
  tensorflow::Status FindArtifactsByContext(int64 context_id,
                                            std::vector<Artifact>* artifacts);

  // Resolves the schema version stored in the metadata source. The `db_version`
  // is set to 0, if it is a 0.13.2 release pre-existing database.
  // Returns DATA_LOSS error, if schema version info table exists but its value
  //   cannot be resolved from the database.
  // Returns NOT_FOUND error, if the database is empty.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status GetSchemaVersion(int64* db_version);

  MetadataSource* metadata_source() { return metadata_source_; }

  MetadataSourceQueryConfig query_config() { return query_config_; }

 private:
  // Constructs a MetadataAccessObject with a query config and a connected
  // MetadataSource.
  explicit MetadataAccessObject(const MetadataSourceQueryConfig& query_config,
                                MetadataSource* connected_metadata_source);

  // Upgrades the database schema version (db_v) to align with the library
  // schema version (lib_v). It retrieves db_v from the metadata source and
  // compares it with the lib_v in the given query_config, and runs migration
  // queries if db_v < lib_v. If `disable_migration`, it only compares the
  // db_v with lib_v and does not change the db schema.
  // Returns FAILED_PRECONDITION error, if db_v > lib_v for the case that the
  //   user use a database produced by a newer version of the library. In that
  //   case, downgrading the database may result in data loss. Often upgrading
  //   the library is required.
  // Returns FAILED_PRECONDITION error, if db_v < lib_v and `disable_migration`
  //   is set to true.
  // Returns DATA_LOSS error, if schema version table exists but no value found.
  // Returns DATA_LOSS error, if the database is not a 0.13.2 release database
  //   and the schema version cannot be resolved.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status UpgradeMetadataSourceIfOutOfDate(bool disable_migration);

  const MetadataSourceQueryConfig query_config_;

  MetadataSource* const metadata_source_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_H_
