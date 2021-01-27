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

#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/status.h"

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
      absl::optional<int64> query_schema_version = absl::nullopt);
  virtual ~QueryExecutor() = default;

  // default & copy constructors are disallowed.
  QueryExecutor(const QueryExecutor&) = delete;
  QueryExecutor& operator=(const QueryExecutor&) = delete;

  // Initializes the metadata source and creates schema. Any existing data in
  // the MetadataSource is dropped.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status InitMetadataSource() = 0;

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
  virtual tensorflow::Status InitMetadataSourceIfNotExists(
      bool enable_upgrade_migration = false) = 0;

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
  virtual tensorflow::Status UpgradeMetadataSourceIfOutOfDate(
      bool enable_migration) = 0;

  // Downgrades the schema to `to_schema_version` in the given metadata source.
  // Returns INVALID_ARGUMENT, if `to_schema_version` is less than 0, or newer
  //   than the library version.
  // Returns FAILED_PRECONDITION, if db schema version is newer than the
  //   library version.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status DowngradeMetadataSource(
      int64 to_schema_version) = 0;

  // Resolves the schema version stored in the metadata source. The `db_version`
  // is set to 0, if it is a 0.13.2 release pre-existing database.
  // Returns DATA_LOSS error, if schema version info table exists but there is
  // more than one value in the database.
  // Returns ABORT error, if schema version info table exists but there is
  // more than one value in the database.
  // Returns NOT_FOUND error, if the database is empty.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status GetSchemaVersion(int64* db_version) = 0;

  // The version of the current query config or source. Increase the version by
  // 1 in any CL that includes physical schema changes and provides a migration
  // function that uses a list migration queries. The database stores it to
  // indicate the current database version. When metadata source creates, it
  // compares the given local `schema_version` in query config with the
  // `schema_version` stored in the database, and migrate the database if
  // needed.
  virtual int64 GetLibraryVersion() = 0;

  // Each of the following methods roughly corresponds to a query (or two).
  virtual tensorflow::Status CheckTypeTable() = 0;

  // Inserts an artifact type into the database.
  // name is the name of the type.
  // type_id is the ID of the artifact type,
  virtual tensorflow::Status InsertArtifactType(const std::string& name,
                                                int64* type_id) = 0;

  // Inserts an execution type into the database.
  // type_name is the name of the type.
  // if has_input_type is true, input_type must be a valid protocol buffer.
  // if has_output_type is true, output_type must be a valid protocol buffer.
  // type_id is the resulting type of the execution.
  virtual tensorflow::Status InsertExecutionType(
      const std::string& type_name, bool has_input_type,
      const google::protobuf::Message& input_type, bool has_output_type,
      const google::protobuf::Message& output_type, int64* type_id) = 0;

  // Inserts a context type into the database.
  // type_name is the name of the type.
  // type_id is the ID of the context type.
  virtual tensorflow::Status InsertContextType(const std::string& type_name,
                                               int64* type_id) = 0;

  // Queries a type by its type id.
  // Returns a message that can be converted to an ArtifactType,
  // ContextType, or ExecutionType.
  virtual tensorflow::Status SelectTypeByID(int64 type_id, TypeKind type_kind,
                                            RecordSet* record_set) = 0;

  // Queries a type by its type name.
  // Returns a message that can be converted to an ArtifactType,
  // ContextType, or ExecutionType.
  virtual tensorflow::Status SelectTypeByName(const absl::string_view type_name,
                                              TypeKind type_kind,
                                              RecordSet* record_set) = 0;

  // Queries for all type instances.
  // Returns a message that can be converted to an ArtifactType,
  // ContextType, or ExecutionType.
  virtual tensorflow::Status SelectAllTypes(TypeKind type_kind,
                                            RecordSet* record_set) = 0;

  // Checks the existence of the TypeProperty table.
  virtual tensorflow::Status CheckTypePropertyTable() = 0;

  // Inserts a property of a type into the database.
  virtual tensorflow::Status InsertTypeProperty(
      int64 type_id, const absl::string_view property_name,
      PropertyType property_type) = 0;

  // Queries properties of a type from the database by the type_id
  // Returns a list of properties (name, data_type).
  virtual tensorflow::Status SelectPropertyByTypeID(int64 type_id,
                                                    RecordSet* record_set) = 0;

  // Checks the existence of the ParentType table.
  virtual tensorflow::Status CheckParentTypeTable() = 0;

  // Inserts a parent type record.
  // Returns OK if the insertion succeeds.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status InsertParentType(int64 type_id,
                                              int64 parent_type_id) = 0;

  // Returns parent types for the type id. Each record has:
  // Column 0: int: type_id (= type_id)
  // Column 1: int: parent_type_id
  virtual tensorflow::Status SelectParentTypesByTypeID(
      int64 type_id, RecordSet* record_set) = 0;

  // Checks the existence of the Artifact table.
  virtual tensorflow::Status CheckArtifactTable() = 0;

  // Inserts an artifact into the database.
  virtual tensorflow::Status InsertArtifact(
      int64 type_id, const std::string& artifact_uri,
      const absl::optional<Artifact::State>& state,
      const absl::optional<std::string>& name, absl::Time create_time,
      absl::Time update_time, int64* artifact_id) = 0;

  // Retrieves artifacts from the database by their ids. Not found ids are
  // skipped. For each matched artifact, returns a row that contains the
  // following columns (order not important):
  // - int: id
  // - int: type_id
  // - string: uri
  // - int: state
  // - string: name
  // - int: create time (since epoch)
  // - int: last update time (since epoch)
  virtual tensorflow::Status SelectArtifactsByID(absl::Span<const int64> ids,
                                                 RecordSet* record_set) = 0;
  // Queries an artifact from the Artifact table by its type_id and name.
  // Returns the artifact ID.
  virtual tensorflow::Status SelectArtifactByTypeIDAndArtifactName(
      int64 artifact_type_id, const absl::string_view name,
      RecordSet* record_set) = 0;

  // Queries artifacts from the Artifact table by their type_id.
  // Returns a list of artifact IDs.
  virtual tensorflow::Status SelectArtifactsByTypeID(int64 artifact_type_id,
                                                     RecordSet* record_set) = 0;

  // Queries an artifact from the database by its uri.
  // Returns a list of artifact IDs.
  virtual tensorflow::Status SelectArtifactsByURI(const absl::string_view uri,
                                                  RecordSet* record_set) = 0;

  // Updates an artifact in the database.
  virtual tensorflow::Status UpdateArtifactDirect(
      int64 artifact_id, int64 type_id, const std::string& uri,
      const absl::optional<Artifact::State>& state, absl::Time update_time) = 0;

  // Checks the existence of the ArtifactProperty table.
  virtual tensorflow::Status CheckArtifactPropertyTable() = 0;

  // Insert a property of an artifact into the database.
  virtual tensorflow::Status InsertArtifactProperty(
      int64 artifact_id, absl::string_view artifact_property_name,
      bool is_custom_property, const Value& property_value) = 0;

  // Queries properties of an artifact from the database by the
  // artifact id. Upon return, each property is mapped to a row in 'record_set'
  // using the convention spelled out in the class docstring.
  virtual tensorflow::Status SelectArtifactPropertyByArtifactID(
      absl::Span<const int64> artifact_ids, RecordSet* record_set) = 0;

  // Updates a property of an artifact in the database.
  virtual tensorflow::Status UpdateArtifactProperty(
      int64 artifact_id, const absl::string_view property_name,
      const Value& property_value) = 0;

  // Deletes a property of an artifact.
  virtual tensorflow::Status DeleteArtifactProperty(
      int64 artifact_id, const absl::string_view property_name) = 0;

  // Checks the existence of the Execution table.
  virtual tensorflow::Status CheckExecutionTable() = 0;

  // Inserts an execution into the database.
  virtual tensorflow::Status InsertExecution(
      int64 type_id, const absl::optional<Execution::State>& last_known_state,
      const absl::optional<std::string>& name, absl::Time create_time,
      absl::Time update_time, int64* execution_id) = 0;

  // Retrieves Executions based on the given ids. Not found ids are skipped.
  // For each matched execution, returns a row that contains the following
  // columns (order not important):
  // - id
  // - type_id
  // - last_known_state
  // - name
  // - create_time_since_epoch
  // - last_update_time_since_epoch
  virtual tensorflow::Status SelectExecutionsByID(
      absl::Span<const int64> execution_ids, RecordSet* record_set) = 0;

  // Queries an execution from the database by its type_id and name.
  virtual tensorflow::Status SelectExecutionByTypeIDAndExecutionName(
      int64 execution_type_id, const absl::string_view name,
      RecordSet* record_set) = 0;

  // Queries an execution from the database by its type_id.
  virtual tensorflow::Status SelectExecutionsByTypeID(
      int64 execution_type_id, RecordSet* record_set) = 0;

  // Updates an execution in the database.
  virtual tensorflow::Status UpdateExecutionDirect(
      int64 execution_id, int64 type_id,
      const absl::optional<Execution::State>& last_known_state,
      absl::Time update_time) = 0;

  // Checks the existence of the ExecutionProperty table.
  virtual tensorflow::Status CheckExecutionPropertyTable() = 0;

  // Insert a property of an execution from the database.
  virtual tensorflow::Status InsertExecutionProperty(
      int64 execution_id, const absl::string_view name, bool is_custom_property,
      const Value& value) = 0;

  // Queries properties of executions matching the given 'ids'.
  // Upon return, each property is mapped to a row in 'record_set'
  // using the convention spelled out in the class docstring.
  virtual tensorflow::Status SelectExecutionPropertyByExecutionID(
      absl::Span<const int64> execution_ids, RecordSet* record_set) = 0;

  // Updates a property of an execution from the database.
  virtual tensorflow::Status UpdateExecutionProperty(
      int64 execution_id, const absl::string_view name, const Value& value) = 0;

  // Deletes a property of an execution.
  virtual tensorflow::Status DeleteExecutionProperty(
      int64 execution_id, const absl::string_view name) = 0;

  // Checks the existence of the Context table.
  virtual tensorflow::Status CheckContextTable() = 0;

  // Inserts a context into the database.
  virtual tensorflow::Status InsertContext(int64 type_id,
                                           const std::string& name,
                                           const absl::Time create_time,
                                           const absl::Time update_time,
                                           int64* context_id) = 0;

  // Retrieves contexts from the database by their ids. For each context,
  // returns a row that contains the following columns (order not important):
  // - int: id
  // - int: type_id
  // - string: name
  // - int: create time (since epoch)
  // - int: last update time (since epoch)
  virtual tensorflow::Status SelectContextsByID(
      absl::Span<const int64> context_ids, RecordSet* record_set) = 0;

  // Returns ids of contexts matching the given context_type_id.
  virtual tensorflow::Status SelectContextsByTypeID(int64 context_type_id,
                                                    RecordSet* record_set) = 0;

  // Returns ids of contexts matching the given context_type_id and name.
  virtual tensorflow::Status SelectContextByTypeIDAndContextName(
      int64 context_type_id, const absl::string_view name,
      RecordSet* record_set) = 0;

  // Updates a context in the Context table.
  virtual tensorflow::Status UpdateContextDirect(
      int64 existing_context_id, int64 type_id, const std::string& context_name,
      const absl::Time update_time) = 0;

  // Checks the existence of the ContextProperty table.
  virtual tensorflow::Status CheckContextPropertyTable() = 0;

  // Insert a property of a context into the database.
  virtual tensorflow::Status InsertContextProperty(int64 context_id,
                                                   const absl::string_view name,
                                                   bool custom_property,
                                                   const Value& value) = 0;

  // Queries properties of contexts from the database by the
  // given context ids.
  virtual tensorflow::Status SelectContextPropertyByContextID(
      absl::Span<const int64> context_id, RecordSet* record_set) = 0;

  // Updates a property of a context in the database.
  virtual tensorflow::Status UpdateContextProperty(
      int64 context_id, const absl::string_view property_name,
      const Value& property_value) = 0;

  // Deletes a property of a context.
  virtual tensorflow::Status DeleteContextProperty(
      const int64 context_id, const absl::string_view property_name) = 0;

  // Checks the existence of the Event table.
  virtual tensorflow::Status CheckEventTable() = 0;

  // Inserts an event into the database.
  virtual tensorflow::Status InsertEvent(int64 artifact_id, int64 execution_id,
                                         int event_type,
                                         int64 event_time_milliseconds,
                                         int64* event_id) = 0;

  // Queries events from the Event table by a collection of artifact ids.
  virtual tensorflow::Status SelectEventByArtifactIDs(
      absl::Span<const int64> artifact_ids, RecordSet* event_record_set) = 0;

  // Queries events from the Event table by a collection of execution ids.
  virtual tensorflow::Status SelectEventByExecutionIDs(
      absl::Span<const int64> execution_ids, RecordSet* event_record_set) = 0;

  // Checks the existence of the EventPath table.
  virtual tensorflow::Status CheckEventPathTable() = 0;

  // Inserts a path step into the EventPath table.
  virtual tensorflow::Status InsertEventPath(int64 event_id,
                                             const Event::Path::Step& step) = 0;

  // Queries paths from the database by a collection of event ids.
  virtual tensorflow::Status SelectEventPathByEventIDs(
      absl::Span<const int64> event_ids, RecordSet* record_set) = 0;

  // Checks the existence of the Association table.
  virtual tensorflow::Status CheckAssociationTable() = 0;

  // Inserts an association into the database.
  virtual tensorflow::Status InsertAssociation(int64 context_id,
                                               int64 execution_id,
                                               int64* association_id) = 0;

  // Returns association triplets for the given context id. Each triplet has:
  // Column 0: int: attribution id
  // Column 1: int: context id
  // Column 2: int: execution id
  virtual tensorflow::Status SelectAssociationByContextID(
      int64 context_id, RecordSet* record_set) = 0;

  // Returns association triplets for the given context id. Each triplet has:
  // Column 0: int: attribution id
  // Column 1: int: context id
  // Column 2: int: execution id
  virtual tensorflow::Status SelectAssociationByExecutionID(
      int64 execution_id, RecordSet* record_set) = 0;

  // Checks the existence of the Attribution table.
  virtual tensorflow::Status CheckAttributionTable() = 0;

  // Inserts an attribution into the database.
  virtual tensorflow::Status InsertAttributionDirect(int64 context_id,
                                                     int64 artifact_id,
                                                     int64* attribution_id) = 0;

  // Returns attribution triplets for the given context id. Each triplet has:
  // Column 0: int: attribution id
  // Column 1: int: context id
  // Column 2: int: artifact id
  virtual tensorflow::Status SelectAttributionByContextID(
      int64 context_id, RecordSet* record_set) = 0;

  // Returns attribution triplets for the given artifact id. Each triplet has:
  // Column 0: int: attribution id
  // Column 1: int: context id
  // Column 2: int: artifact id
  virtual tensorflow::Status SelectAttributionByArtifactID(
      int64 artifact_id, RecordSet* record_set) = 0;

  // Checks the existence of the ParentContext table.
  virtual tensorflow::Status CheckParentContextTable() = 0;

  // Inserts a parent context.
  // Returns OK if the insertion succeeds.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual tensorflow::Status InsertParentContext(int64 parent_id,
                                                 int64 child_id) = 0;

  // Returns parent contexts for the given context id. Each record has:
  // Column 0: int: context id (= context_id)
  // Column 1: int: parent context id
  virtual tensorflow::Status SelectParentContextsByContextID(
      int64 context_id, RecordSet* record_set) = 0;

  // Returns child contexts for the given context id. Each record has:
  // Column 0: int: context id
  // Column 1: int: parent context id (= context_id)
  virtual tensorflow::Status SelectChildContextsByContextID(
      int64 context_id, RecordSet* record_set) = 0;

  // Checks the MLMDEnv table and query the schema version.
  // At MLMD release v0.13.2, by default it is v0.
  virtual tensorflow::Status CheckMLMDEnvTable() = 0;

  // Insert schema_version.
  virtual tensorflow::Status InsertSchemaVersion(int64 schema_version) = 0;

  // Update schema_version
  virtual tensorflow::Status UpdateSchemaVersion(int64 schema_version) = 0;

  // Check the database is a valid database produced by 0.13.2 MLMD release.
  // The schema version and migration are introduced after that release.
  virtual tensorflow::Status CheckTablesIn_V0_13_2() = 0;

  // Note: these are not reflected in the original queries.
  // Select all artifact IDs.
  // Returns a list of IDs.
  virtual tensorflow::Status SelectAllArtifactIDs(RecordSet* set) = 0;

  // Select all execution IDs.
  // Returns a list of IDs.
  virtual tensorflow::Status SelectAllExecutionIDs(RecordSet* set) = 0;

  // Select all context IDs.
  // Returns a list of IDs.
  virtual tensorflow::Status SelectAllContextIDs(RecordSet* set) = 0;

  // List Artifact IDs using `options`. If `candidate_ids` is not empty, then
  // returned result is only built using ids in the `candidate_ids`. On success
  // `record_set` is updated with artifact IDs based on `options`
  virtual tensorflow::Status ListArtifactIDsUsingOptions(
      const ListOperationOptions& options,
      const absl::Span<const int64> candidate_ids, RecordSet* record_set) = 0;

  // List Execution IDs using `options`. If `candidate_ids` is not empty, then
  // returned result is only built using ids in the `candidate_ids`. On success
  // `set` is updated with execution IDs based on `options` and
  // `next_page_token` is updated with information for the caller to use for
  // next page of results.
  virtual tensorflow::Status ListExecutionIDsUsingOptions(
      const ListOperationOptions& options,
      const absl::Span<const int64> candidate_ids, RecordSet* record_set) = 0;

  // List Context IDs using `options`. If `candidate_ids` is not empty, then
  // returned result is only built using ids in the `candidate_ids`. On success
  // `set` is updated with context IDs based on `options` and `next_page_token`
  // is updated with information for the caller to use for next page of results.
  virtual tensorflow::Status ListContextIDsUsingOptions(
      const ListOperationOptions& options,
      const absl::Span<const int64> candidate_ids, RecordSet* record_set) = 0;


 protected:
  // Uses the method to document the min schema version of an API explicitly.
  // Returns FailedPrecondition, if the |query_schema_version_| is less than the
  //   mininum schema version that the API is expected to work with.
  tensorflow::Status VerifyCurrentQueryVersionIsAtLeast(
      int64 min_schema_version) const;

  // If |query_schema_version_| is given, then the query executor is expected to
  // work with an existing db with an earlier schema version (=
  // query_schema_version_). Returns FailedPrecondition, if the db is empty or
  // the db is initialized
  //   with a schema_version != query_schema_version_.
  tensorflow::Status CheckSchemaVersionAlignsWithQueryVersion();

  // Access the query_schema_version_ if any.
  absl::optional<int64> query_schema_version() const {
    return query_schema_version_;
  }

 private:
  // By default, the query executor assumes that the db schema version aligns
  // with the library version. If set, the query executor is switched to use
  // the queries to talk to an earlier schema_version = query_schema_version_.
  absl::optional<int64> query_schema_version_ = absl::nullopt;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_QUERY_EXECUTOR_H_
