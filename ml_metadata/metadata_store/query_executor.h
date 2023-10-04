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
#ifndef ML_METADATA_METADATA_STORE_QUERY_EXECUTOR_H_
#define ML_METADATA_METADATA_STORE_QUERY_EXECUTOR_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {

// A class wrapping a low-level interface to a database.
// This contains both the queries and the method for executing them.
// Most methods correspond to one or two queries, with a few exceptions
// (such as InitMetadataSource).
//
// IMPORTANT NOTE: All Select{X}PropertyBy{X}Id methods return a RecordSet for
// the properties of the input {X} type of node (X in {Artifact, Context,
// Execution}) and use the same convention:
// - Column 0: int: node id
// - Column 1: string: property name
// - Column 2: boolean: true if this is a custom property
// - Column 3: int: property value or NULL
// - Column 4: double: property value or NULL
// - Column 5: string: property value or NULL
//
// Some methods might add additional columns
class QueryExecutor {
 public:
  // By default, for any empty db, the head schema should be used to init new
  // db instances. Giving an optional `query_schema_version` allows the query
  // executor to work with an existing db with an earlier schema version other
  // than the current library version. The earlier `query_schema_version` is
  // useful for multi-tenant applications to have better availability when
  // configured with a set of existing backends with different schema versions.
  explicit QueryExecutor(
      std::optional<int64_t> query_schema_version = absl::nullopt);
  virtual ~QueryExecutor() = default;

  // default & copy constructors are disallowed.
  QueryExecutor(const QueryExecutor&) = delete;
  QueryExecutor& operator=(const QueryExecutor&) = delete;

  // Initializes the metadata source and creates schema. Any existing data in
  // the MetadataSource is dropped.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status InitMetadataSource() = 0;

  // Initializes the metadata source and creates schema if not exist.
  // Returns OK and does nothing, if all required schema exist.
  // Returns OK and creates schema, if no schema exists yet.
  // Returns DATA_LOSS error, if the MLMDENv has more than one schema version.
  // Returns ABORTED error, if any required schema is missing.
  // Returns FAILED_PRECONDITION error, if library and db have incompatible
  //   schema versions, and upgrade migrations are not enabled.
  //
  // When |query_schema_version_| is set:
  // Returns OK and does nothing, if the |query_schema_version_| aligns with
  //   the db schema version in the metadata source.
  // Returns FAILED_PRECONDITION error, if the given db is empty or at another
  //   schema version.
  // Returns detailed INTERNAL error, if create schema query execution fails.
  virtual absl::Status InitMetadataSourceIfNotExists(
      bool enable_upgrade_migration = false) = 0;

  // Initializes the metadata source without checking schema.
  // It assumes the schema is already in place and up-to-date.
  // Creates a new store if enable_new_store_creation is set to true and
  // corresponding store is not created yet.
  // Returns OK if the init succeeds.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status InitMetadataSourceLight(
      bool enable_new_store_creation = false) = 0;

  // Deletes the metadata source.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteMetadataSource() = 0;

  // Upgrades the database schema version (db_v) to align with the library
  // schema version (lib_v). It retrieves db_v from the metadata source and
  // compares it with the lib_v in the given query_config, and runs migration
  // queries if db_v < lib_v.
  // Returns FAILED_PRECONDITION error, if db_v > lib_v for the case that the
  //   user use a database produced by a newer version of the library. In that
  //   case, downgrading the database may result in data loss. Often upgrading
  //   the library is required.
  // Returns DATA_LOSS error, if schema version table exists but no value found.
  // Returns DATA_LOSS error, if the database is not a 0.13.2 release database
  //   and the schema version cannot be resolved.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpgradeMetadataSourceIfOutOfDate(
      bool enable_migration) = 0;

  // Downgrades the schema to `to_schema_version` in the given metadata source.
  // Returns INVALID_ARGUMENT, if `to_schema_version` is less than 0, or newer
  //   than the library version.
  // Returns FAILED_PRECONDITION, if db schema version is newer than the
  //   library version.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DowngradeMetadataSource(int64_t to_schema_version) = 0;

  // Resolves the schema version stored in the metadata source. The `db_version`
  // is set to 0, if it is a 0.13.2 release pre-existing database.
  // Returns DATA_LOSS error, if schema version info table exists but there is
  // more than one value in the database.
  // Returns ABORT error, if schema version info table exists but there is
  // more than one value in the database.
  // Returns NOT_FOUND error, if the database is empty.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status GetSchemaVersion(int64_t* db_version) = 0;

  // The version of the current query config or source. Increase the version by
  // 1 in any CL that includes physical schema changes and provides a migration
  // function that uses a list migration queries. The database stores it to
  // indicate the current database version. When metadata source creates, it
  // compares the given local `schema_version` in query config with the
  // `schema_version` stored in the database, and migrate the database if
  // needed.
  virtual int64_t GetLibraryVersion() = 0;

  // Each of the following methods roughly corresponds to a query (or two).
  virtual absl::Status CheckTypeTable() = 0;

  // Inserts a type (ArtifactType/ExecutionType/ContextType).
  //
  // A type has a name and a set of strong typed properties describing the
  // schema of any stored instance associated with that type.
  // A type can be evolved in multiple ways:
  // a) it can be updated in-place by adding more properties, but not remove
  //    or change value type of registered properties. The in-place updates
  //    remain backward-compatible for all stored instances of that type.
  // b) it can be annotated with a different version for non-backward
  //    compatible changes, e.g., deprecate properties, re-purpose property
  //    name, change property value types.
  //
  // `name` is mandatory for the type. If `version` is not given, `name` is
  //    unique among stored types. If `version` is given, a type can have
  //    multiple `versions` with the same `name`.
  // `version` is an optional field to annotate the version for the type. A
  //    (`name`, `version`) tuple has its own id and may not be compatible with
  //    other versions of the same `name`.
  // `description` is an optional field to capture auxiliary type information.
  //
  // `type_id` is the output ID of the artifact type.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status InsertArtifactType(
      const std::string& name, std::optional<absl::string_view> version,
      std::optional<absl::string_view> description,
      std::optional<absl::string_view> external_id, int64_t* type_id) = 0;

  // Inserts an ExecutionType into the database.
  // `input_type` is an optional field to describe the input artifact types.
  // `output_type` is an optional field to describe the output artifact types.
  // `type_id` is the ID of the execution type.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status InsertExecutionType(
      const std::string& name, std::optional<absl::string_view> version,
      std::optional<absl::string_view> description,
      const ArtifactStructType* input_type,
      const ArtifactStructType* output_type,
      std::optional<absl::string_view> external_id, int64_t* type_id) = 0;

  // Inserts a ContextType into the database.
  // `type_id` is the ID of the context type.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status InsertContextType(
      const std::string& name, std::optional<absl::string_view> version,
      std::optional<absl::string_view> description,
      std::optional<absl::string_view> external_id, int64_t* type_id) = 0;

  // Retrieves types from the database by their ids. Not found ids are
  // skipped.
  // Returned messages can be converted to ArtifactType, ContextType, or
  // ExecutionType.
  virtual absl::Status SelectTypesByID(absl::Span<const int64_t> type_ids,
                                       TypeKind type_kind,
                                       RecordSet* record_set) = 0;

  // Gets types from the database by their external_ids. Not found ids are
  // skipped.
  // Returned messages can be converted to ArtifactType, ContextType, or
  // ExecutionType.
  virtual absl::Status SelectTypesByExternalIds(
      absl::Span<absl::string_view> external_ids, TypeKind type_kind,
      RecordSet* record_set) = 0;

  // Gets a type by its type id.
  // Returns a message that can be converted to an ArtifactType,
  // ContextType, or ExecutionType.
  // TODO(b/171597866) Improve document and describe the returned `record_set`.
  // for the query executor APIs.
  virtual absl::Status SelectTypeByID(int64_t type_id, TypeKind type_kind,
                                      RecordSet* record_set) = 0;

  // Gets a type by its type name and an optional version. If version is
  // not given or the version is an empty string, (type_name, version = NULL)
  // is used to retrieve types.
  // Returns a message that can be converted to an ArtifactType,
  // ContextType, or ExecutionType.
  virtual absl::Status SelectTypeByNameAndVersion(
      absl::string_view type_name,
      std::optional<absl::string_view> type_version, TypeKind type_kind,
      RecordSet* record_set) = 0;

  // Gets types from the database by their name and version pairs. Not found
  // pairs are skipped. If the version of a type is not available, then an empty
  // string should be used. Internally (type_name, version = NULL) is used to
  // retrieve types when the version is an empty string.
  // Note: The returned types does not guaranteed to have the same order as
  // `names_and_versions`.
  // Returned messages can be converted to ArtifactType, ContextType, or
  // ExecutionType.
  virtual absl::Status SelectTypesByNamesAndVersions(
      absl::Span<std::pair<std::string, std::string>> names_and_versions,
      TypeKind type_kind, RecordSet* record_set) = 0;

  // Gets all type instances.
  // Returns a message that can be converted to an ArtifactType,
  // ContextType, or ExecutionType.
  virtual absl::Status SelectAllTypes(TypeKind type_kind,
                                      RecordSet* record_set) = 0;

  // Updates a type's `external_id` in the database.
  virtual absl::Status UpdateTypeExternalIdDirect(
      int64_t type_id, std::optional<absl::string_view> external_id) = 0;

  // Checks the existence of the TypeProperty table.
  virtual absl::Status CheckTypePropertyTable() = 0;

  // Inserts a property of a type into the database.
  virtual absl::Status InsertTypeProperty(int64_t type_id,
                                          absl::string_view property_name,
                                          PropertyType property_type) = 0;

  // Gets properties of types from the database for each type_id in
  // `type_ids`.
  // Returns a list of properties (type_id, name, data_type).
  virtual absl::Status SelectPropertiesByTypeID(
      absl::Span<const int64_t> type_ids, RecordSet* record_set) = 0;

  // Checks the existence of the ParentType table.
  virtual absl::Status CheckParentTypeTable() = 0;

  // Inserts a parent type record.
  // Returns OK if the insertion succeeds.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status InsertParentType(int64_t type_id,
                                        int64_t parent_type_id) = 0;

  // Deletes a parent type record.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteParentType(int64_t type_id,
                                        int64_t parent_type_id) = 0;

  // Returns parent types for the type id in `type_ids`. Each record has:
  // Column 0: int: type_id (= type_id in `type_ids`)
  // Column 1: int: parent_type_id
  virtual absl::Status SelectParentTypesByTypeID(
      absl::Span<const int64_t> type_ids, RecordSet* record_set) = 0;

  // Checks the existence of the Artifact table.
  virtual absl::Status CheckArtifactTable() = 0;

  // Inserts an artifact into the database.
  virtual absl::Status InsertArtifact(
      int64_t type_id, const std::string& artifact_uri,
      const std::optional<Artifact::State>& state,
      const std::optional<std::string>& name,
      std::optional<absl::string_view> external_id, absl::Time create_time,
      absl::Time update_time, int64_t* artifact_id) = 0;

  // Gets artifacts from the database by their ids. Not found ids are
  // skipped. For each matched artifact, returns a row that contains the
  // following columns (order not important):
  // - int: id
  // - int: type_id
  // - string: uri
  // - int: state
  // - string: name
  // - int: create time (since epoch)
  // - int: last update time (since epoch)
  // - string: type (which refers to type name)
  // - string: type_version
  // - string: type_description
  // - string: type_external_id
  virtual absl::Status SelectArtifactsByID(absl::Span<const int64_t> ids,
                                           RecordSet* record_set) = 0;

  // Gets artifacts from the database by their external_ids. Not found
  // external_ids are skipped.
  virtual absl::Status SelectArtifactsByExternalIds(
      absl::Span<absl::string_view> external_ids, RecordSet* record_set) = 0;

  // Gets an artifact from the Artifact table by its type_id and name.
  // Returns the artifact ID.
  virtual absl::Status SelectArtifactByTypeIDAndArtifactName(
      int64_t artifact_type_id, absl::string_view name,
      RecordSet* record_set) = 0;

  // Gets artifacts from the Artifact table by their type_id.
  // Returns a list of artifact IDs.
  virtual absl::Status SelectArtifactsByTypeID(int64_t artifact_type_id,
                                               RecordSet* record_set) = 0;

  // Gets an artifact from the database by its uri.
  // Returns a list of artifact IDs.
  virtual absl::Status SelectArtifactsByURI(absl::string_view uri,
                                            RecordSet* record_set) = 0;

  // Updates an artifact in the database.
  virtual absl::Status UpdateArtifactDirect(
      int64_t artifact_id, int64_t type_id, const std::string& uri,
      const std::optional<Artifact::State>& state,
      std::optional<absl::string_view> external_id, absl::Time update_time) = 0;

  // Checks the existence of the ArtifactProperty table.
  virtual absl::Status CheckArtifactPropertyTable() = 0;

  // Insert a property of an artifact into the database.
  virtual absl::Status InsertArtifactProperty(
      int64_t artifact_id, absl::string_view artifact_property_name,
      bool is_custom_property, const Value& property_value) = 0;

  // Gets properties of an artifact from the database by the
  // artifact id. Upon return, each property is mapped to a row in 'record_set'
  // using the convention spelled out in the class docstring.
  virtual absl::Status SelectArtifactPropertyByArtifactID(
      absl::Span<const int64_t> artifact_ids, RecordSet* record_set) = 0;

  // Updates a property of an artifact in the database.
  virtual absl::Status UpdateArtifactProperty(int64_t artifact_id,
                                              absl::string_view property_name,
                                              const Value& property_value) = 0;

  // Deletes a property of an artifact.
  virtual absl::Status DeleteArtifactProperty(
      int64_t artifact_id, absl::string_view property_name) = 0;

  // Checks the existence of the Execution table.
  virtual absl::Status CheckExecutionTable() = 0;

  // Inserts an execution into the database.
  virtual absl::Status InsertExecution(
      int64_t type_id, const std::optional<Execution::State>& last_known_state,
      const std::optional<std::string>& name,
      std::optional<absl::string_view> external_id, absl::Time create_time,
      absl::Time update_time, int64_t* execution_id) = 0;

  // Gets Executions based on the given ids. Not found ids are skipped.
  // For each matched execution, returns a row that contains the following
  // columns (order not important):
  // - id
  // - type_id
  // - last_known_state
  // - name
  // - create_time_since_epoch
  // - last_update_time_since_epoch
  // - string: type (which refers to type name)
  // - string: type_version
  // - string: type_description
  // - string: type_external_id
  virtual absl::Status SelectExecutionsByID(
      absl::Span<const int64_t> execution_ids, RecordSet* record_set) = 0;

  // Gets executions based on the given external_ids. Not found
  // external_ids are skipped.
  virtual absl::Status SelectExecutionsByExternalIds(
      absl::Span<absl::string_view> external_ids, RecordSet* record_set) = 0;

  // Gets an execution from the database by its type_id and name.
  virtual absl::Status SelectExecutionByTypeIDAndExecutionName(
      int64_t execution_type_id, absl::string_view name,
      RecordSet* record_set) = 0;

  // Gets an execution from the database by its type_id.
  virtual absl::Status SelectExecutionsByTypeID(int64_t execution_type_id,
                                                RecordSet* record_set) = 0;

  // Updates an execution in the database.
  virtual absl::Status UpdateExecutionDirect(
      int64_t execution_id, int64_t type_id,
      const std::optional<Execution::State>& last_known_state,
      std::optional<absl::string_view> external_id, absl::Time update_time) = 0;

  // Checks the existence of the ExecutionProperty table.
  virtual absl::Status CheckExecutionPropertyTable() = 0;

  // Insert a property of an execution from the database.
  virtual absl::Status InsertExecutionProperty(int64_t execution_id,
                                               absl::string_view name,
                                               bool is_custom_property,
                                               const Value& value) = 0;

  // Gets properties of executions matching the given 'ids'.
  // Upon return, each property is mapped to a row in 'record_set'
  // using the convention spelled out in the class docstring.
  virtual absl::Status SelectExecutionPropertyByExecutionID(
      absl::Span<const int64_t> execution_ids, RecordSet* record_set) = 0;

  // Updates a property of an execution from the database.
  virtual absl::Status UpdateExecutionProperty(int64_t execution_id,
                                               absl::string_view name,
                                               const Value& value) = 0;

  // Deletes a property of an execution.
  virtual absl::Status DeleteExecutionProperty(int64_t execution_id,
                                               absl::string_view name) = 0;

  // Checks the existence of the Context table.
  virtual absl::Status CheckContextTable() = 0;

  // Inserts a context into the database.
  virtual absl::Status InsertContext(
      int64_t type_id, const std::string& name,
      std::optional<absl::string_view> external_id,
      const absl::Time create_time, const absl::Time update_time,
      int64_t* context_id) = 0;

  // Gets contexts from the database by their ids. For each context,
  // returns a row that contains the following columns (order not important):
  // - int: id
  // - int: type_id
  // - string: name
  // - int: create time (since epoch)
  // - int: last update time (since epoch)
  // - string: type (which refers to type name)
  // - string: type_version
  // - string: type_description
  // - string: type_external_id
  virtual absl::Status SelectContextsByID(absl::Span<const int64_t> context_ids,
                                          RecordSet* record_set) = 0;

  // Gets contexts from the database by their external_ids. Not found
  // external_ids are skipped.
  virtual absl::Status SelectContextsByExternalIds(
      absl::Span<absl::string_view> external_ids, RecordSet* record_set) = 0;

  // Returns ids of contexts matching the given context_type_id.
  virtual absl::Status SelectContextsByTypeID(int64_t context_type_id,
                                              RecordSet* record_set) = 0;

  // Returns ids of contexts matching the given context_type_id and name.
  virtual absl::Status SelectContextByTypeIDAndContextName(
      int64_t context_type_id, absl::string_view name,
      RecordSet* record_set) = 0;

  // Updates a context in the Context table.
  virtual absl::Status UpdateContextDirect(
      int64_t existing_context_id, int64_t type_id,
      const std::string& context_name,
      std::optional<absl::string_view> external_id,
      const absl::Time update_time) = 0;

  // Checks the existence of the ContextProperty table.
  virtual absl::Status CheckContextPropertyTable() = 0;

  // Insert a property of a context into the database.
  virtual absl::Status InsertContextProperty(int64_t context_id,
                                             absl::string_view name,
                                             bool custom_property,
                                             const Value& value) = 0;

  // Gets properties of contexts from the database by the
  // given context ids.
  virtual absl::Status SelectContextPropertyByContextID(
      absl::Span<const int64_t> context_id, RecordSet* record_set) = 0;

  // Updates a property of a context in the database.
  virtual absl::Status UpdateContextProperty(int64_t context_id,
                                             absl::string_view property_name,
                                             const Value& property_value) = 0;

  // Deletes a property of a context.
  virtual absl::Status DeleteContextProperty(
      const int64_t context_id, absl::string_view property_name) = 0;

  // Checks the existence of the Event table.
  virtual absl::Status CheckEventTable() = 0;

  // Inserts an event into the database.
  virtual absl::Status InsertEvent(int64_t artifact_id, int64_t execution_id,
                                   int event_type,
                                   int64_t event_time_milliseconds,
                                   int64_t* event_id) = 0;

  // Gets events from the Event table by a collection of artifact ids.
  virtual absl::Status SelectEventByArtifactIDs(
      absl::Span<const int64_t> artifact_ids, RecordSet* event_record_set) = 0;

  // Gets events from the Event table by a collection of execution ids.
  virtual absl::Status SelectEventByExecutionIDs(
      absl::Span<const int64_t> execution_ids, RecordSet* event_record_set) = 0;

  // Checks the existence of the EventPath table.
  virtual absl::Status CheckEventPathTable() = 0;

  // Inserts a path step into the EventPath table.
  virtual absl::Status InsertEventPath(int64_t event_id,
                                       const Event::Path::Step& step) = 0;

  // Gets paths from the database by a collection of event ids.
  virtual absl::Status SelectEventPathByEventIDs(
      absl::Span<const int64_t> event_ids, RecordSet* record_set) = 0;

  // Checks the existence of the Association table.
  virtual absl::Status CheckAssociationTable() = 0;

  // Inserts an association into the database.
  virtual absl::Status InsertAssociation(int64_t context_id,
                                         int64_t execution_id,
                                         int64_t* association_id) = 0;

  // Returns association triplets for the given context id. Each triplet has:
  // Column 0: int: attribution id
  // Column 1: int: context id
  // Column 2: int: execution id
  virtual absl::Status SelectAssociationByContextIDs(
      absl::Span<const int64_t> context_id, RecordSet* record_set) = 0;

  // Returns association triplets for the given execution ids.
  // Each triplet has:
  // Column 0: int: attribution id
  // Column 1: int: context id
  // Column 2: int: execution id
  virtual absl::Status SelectAssociationsByExecutionIds(
      absl::Span<const int64_t> execution_ids, RecordSet* record_set) = 0;

  // Checks the existence of the Attribution table.
  virtual absl::Status CheckAttributionTable() = 0;

  // Inserts an attribution into the database.
  virtual absl::Status InsertAttributionDirect(int64_t context_id,
                                               int64_t artifact_id,
                                               int64_t* attribution_id) = 0;

  // Returns attribution triplets for the given context id. Each triplet has:
  // Column 0: int: attribution id
  // Column 1: int: context id
  // Column 2: int: artifact id
  virtual absl::Status SelectAttributionByContextID(int64_t context_id,
                                                    RecordSet* record_set) = 0;

  // Returns attribution triplets for the given artifact ids. Each triplet has:
  // Column 0: int: attribution id
  // Column 1: int: context id
  // Column 2: int: artifact id
  virtual absl::Status SelectAttributionsByArtifactIds(
      absl::Span<const int64_t> artifact_ids, RecordSet* record_set) = 0;

  // Checks the existence of the ParentContext table.
  virtual absl::Status CheckParentContextTable() = 0;

  // Inserts a parent context.
  // Returns OK if the insertion succeeds.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status InsertParentContext(int64_t parent_id,
                                           int64_t child_id) = 0;

  // Returns parent contexts for the given context id. Each record has:
  // Column 0: int: context id (= context_id)
  // Column 1: int: parent context id
  virtual absl::Status SelectParentContextsByContextID(
      int64_t context_id, RecordSet* record_set) = 0;

  // Returns child contexts for the given context id. Each record has:
  // Column 0: int: context id
  // Column 1: int: parent context id (= context_id)
  virtual absl::Status SelectChildContextsByContextID(
      int64_t context_id, RecordSet* record_set) = 0;

  // Returns parent contexts for the given context ids. Each record has:
  // Column 0: int: context id (IN context_ids)
  // Column 1: int: parent context id
  virtual absl::Status SelectParentContextsByContextIDs(
      absl::Span<const int64_t> context_ids, RecordSet* record_set) = 0;

  // Returns child contexts for the given parent context ids. Each record has:
  // Column 0: int: context id
  // Column 1: int: parent context id (IN context_ids)
  virtual absl::Status SelectChildContextsByContextIDs(
      absl::Span<const int64_t> context_ids, RecordSet* record_set) = 0;

  // Checks the MLMDEnv table and query the schema version.
  // At MLMD release v0.13.2, by default it is v0.
  virtual absl::Status CheckMLMDEnvTable() = 0;

  // Insert schema_version.
  virtual absl::Status InsertSchemaVersion(int64_t schema_version) = 0;

  // Update schema_version
  virtual absl::Status UpdateSchemaVersion(int64_t schema_version) = 0;

  // Check the database is a valid database produced by 0.13.2 MLMD release.
  // The schema version and migration are introduced after that release.
  virtual absl::Status CheckTablesIn_V0_13_2() = 0;

  // Note: these are not reflected in the original queries.
  // Select all artifact IDs.
  // Returns a list of IDs.
  virtual absl::Status SelectAllArtifactIDs(RecordSet* set) = 0;

  // Select all execution IDs.
  // Returns a list of IDs.
  virtual absl::Status SelectAllExecutionIDs(RecordSet* set) = 0;

  // Select all context IDs.
  // Returns a list of IDs.
  virtual absl::Status SelectAllContextIDs(RecordSet* set) = 0;

  // List Artifact IDs using `options`. If `candidate_ids` is provided, then
  // returned result is only built using ids in the `candidate_ids`, when
  // nullopt, all stored artifacts are considered as candidates. On success
  // `record_set` is updated with artifact IDs based on `options`
  virtual absl::Status ListArtifactIDsUsingOptions(
      const ListOperationOptions& options,
      std::optional<absl::Span<const int64_t>> candidate_ids,
      RecordSet* record_set) = 0;

  // List Execution IDs using `options`. If `candidate_ids` is provided, then
  // returned result is only built using ids in the `candidate_ids`, when
  // nullopt, all stored executions are considered as candidates. On success
  // `record_set` is updated with execution IDs based on `options`.
  virtual absl::Status ListExecutionIDsUsingOptions(
      const ListOperationOptions& options,
      std::optional<absl::Span<const int64_t>> candidate_ids,
      RecordSet* record_set) = 0;

  // List Context IDs using `options`. If `candidate_ids` is provided, then
  // returned result is only built using ids in the `candidate_ids`, when
  // nullopt, all stored contexts are considered as candidates. On success
  // `record_set` is updated with context IDs based on `options`.
  virtual absl::Status ListContextIDsUsingOptions(
      const ListOperationOptions& options,
      std::optional<absl::Span<const int64_t>> candidate_ids,
      RecordSet* record_set) = 0;


  // Deletes a list of artifacts by id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteArtifactsById(
      absl::Span<const int64_t> artifact_ids) = 0;

  // Deletes a list of contexts by id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteContextsById(
      absl::Span<const int64_t> context_ids) = 0;

  // Deletes a list of executions by id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteExecutionsById(
      absl::Span<const int64_t> execution_ids) = 0;

  // Deletes the events corresponding to the |artifact_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteEventsByArtifactsId(
      absl::Span<const int64_t> artifact_ids) = 0;

  // Deletes the events corresponding to the |execution_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteEventsByExecutionsId(
      absl::Span<const int64_t> execution_ids) = 0;

  // Deletes the associations corresponding to the |context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteAssociationsByContextsId(
      absl::Span<const int64_t> context_ids) = 0;

  // Deletes the associations corresponding to the |execution_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteAssociationsByExecutionsId(
      absl::Span<const int64_t> execution_ids) = 0;

  // Deletes the attributions corresponding to the |context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteAttributionsByContextsId(
      absl::Span<const int64_t> context_ids) = 0;

  // Deletes the attributions corresponding to the |artifact_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteAttributionsByArtifactsId(
      absl::Span<const int64_t> artifact_ids) = 0;

  // Deletes the parent contexts corresponding to the |parent_context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteParentContextsByParentIds(
      absl::Span<const int64_t> parent_context_ids) = 0;

  // Deletes the parent contexts corresponding to the |child_context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteParentContextsByChildIds(
      absl::Span<const int64_t> child_context_ids) = 0;

  // Deletes the parent contexts corresponding to the |parent_context_id|
  // and |child_context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteParentContextsByParentIdAndChildIds(
      int64_t parent_context_id,
      absl::Span<const int64_t> child_context_ids) = 0;

  // Utility methods which may be used to websafe encode bytes specific to a
  // metadata source. For example, the MySQL and SQLite3 metadata sources do not
  // handle serialized protocol buffer bytes, but can handle base64 encoded
  // bytes.
  // EncodeBytes defaults to the trivial encoding enc(x) = x
  virtual std::string EncodeBytes(absl::string_view value) const = 0;

  // Decodes value and writes the result to dest.
  // The default implementation is the trivial decoding dec(x) = x
  // Returns OkStatus on success, otherwise an informative error.
  virtual absl::Status DecodeBytes(
    absl::string_view value, string& dest) const = 0;

 protected:
  // Uses the method to document the min schema version of an API explicitly.
  // Returns FailedPrecondition, if the |query_schema_version_| is less than the
  //   mininum schema version that the API is expected to work with.
  absl::Status VerifyCurrentQueryVersionIsAtLeast(
      int64_t min_schema_version) const;

  // If |query_schema_version_| is given, then the query executor is expected to
  // work with an existing db with an earlier schema version (=
  // query_schema_version_). Returns FailedPrecondition, if the db is empty or
  // the db is initialized
  //   with a schema_version != query_schema_version_.
  absl::Status CheckSchemaVersionAlignsWithQueryVersion();

  // Uses the method to document the query branches for earlier schema for
  // ease of cleanup after the temporary branches after the migration.
  // Returns true if |query_schema_version_| = `schema_version`.
  bool IsQuerySchemaVersionEquals(int64_t schema_version) const;

  // Access the query_schema_version_ if any.
  std::optional<int64_t> query_schema_version() const {
    return query_schema_version_;
  }

 private:
  // By default, the query executor assumes that the db schema version aligns
  // with the library version. If set, the query executor is switched to use
  // the queries to talk to an earlier schema_version = query_schema_version_.
  std::optional<int64_t> query_schema_version_ = absl::nullopt;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_QUERY_EXECUTOR_H_
