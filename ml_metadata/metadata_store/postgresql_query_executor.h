/* Copyright 2023 Google LLC

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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_POSTGRESQL_QUERY_EXECUTOR_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_POSTGRESQL_QUERY_EXECUTOR_H_

#include <memory>
#include <string>
#include <vector>

#include <glog/logging.h>
#include "google/protobuf/text_format.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/query_config_executor.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {

// A PostgreSQL version of the QueryExecutor. The text of most queries are
// encoded in MetadataSourceQueryConfig. This class binds the relevant arguments
// for each query using the Bind() methods. See notes on constructor for various
// ways to construct this object.
class PostgreSQLQueryExecutor : public QueryExecutor {
 public:
  // Note that the query config and the MetadataSource must be compatible.
  // MetadataSourceQueryConfigs can be created using the methods in
  // ml_metadata/util/metadata_source_query_config.h.
  // For example:
  //    util::GetPostgreSQLMetadataSourceQueryConfig().
  //
  // The MetadataSource is not owned by this object, and must outlast it.
  PostgreSQLQueryExecutor(const MetadataSourceQueryConfig& query_config,
                          MetadataSource* source)
      : query_config_(query_config), metadata_source_(source) {}

  // A `query_version` can be passed to the PostgreSQLQueryExecutor to work with
  // an existing db with an earlier schema version.
  PostgreSQLQueryExecutor(const MetadataSourceQueryConfig& query_config,
                          MetadataSource* source, int64_t query_version);

  // default & copy constructors are disallowed.
  PostgreSQLQueryExecutor() = delete;
  PostgreSQLQueryExecutor(const PostgreSQLQueryExecutor&) = delete;
  PostgreSQLQueryExecutor& operator=(const PostgreSQLQueryExecutor&) = delete;

  virtual ~PostgreSQLQueryExecutor() = default;

  absl::Status InitMetadataSource() final;

  absl::Status InitMetadataSourceIfNotExists(
      bool enable_upgrade_migration) final;

  absl::Status InitMetadataSourceLight(bool enable_new_store_creation) final {
    return absl::UnimplementedError(
        "InitMetadataSourceLight not supported for PostgreSQLQueryExecutor");
  }

  absl::Status DeleteMetadataSource() final {
    return absl::UnimplementedError(
        "DeleteMetadataSource not supported for PostgreSQLQueryExecutor");
  }

  absl::Status GetSchemaVersion(int64_t* db_version) final;

  absl::Status CheckTableResult(
      const MetadataSourceQueryConfig::TemplateQuery query);

  absl::Status CheckTypeTable() final;

  absl::Status InsertArtifactType(const std::string& name,
                                  std::optional<absl::string_view> version,
                                  std::optional<absl::string_view> description,
                                  std::optional<absl::string_view> external_id,
                                  int64_t* type_id) final;

  absl::Status InsertExecutionType(const std::string& name,
                                   std::optional<absl::string_view> version,
                                   std::optional<absl::string_view> description,
                                   const ArtifactStructType* input_type,
                                   const ArtifactStructType* output_type,
                                   std::optional<absl::string_view> external_id,
                                   int64_t* type_id) final;

  absl::Status InsertContextType(const std::string& name,
                                 std::optional<absl::string_view> version,
                                 std::optional<absl::string_view> description,
                                 std::optional<absl::string_view> external_id,
                                 int64_t* type_id) final;

  absl::Status SelectTypesByID(absl::Span<const int64_t> type_ids,
                               TypeKind type_kind, RecordSet* record_set) final;

  absl::Status SelectTypesByExternalIds(
      absl::Span<absl::string_view> external_ids, TypeKind type_kind,
      RecordSet* record_set) final;

  absl::Status SelectTypeByID(int64_t type_id, TypeKind type_kind,
                              RecordSet* record_set) final;

  absl::Status SelectTypeByNameAndVersion(
      absl::string_view type_name,
      std::optional<absl::string_view> type_version, TypeKind type_kind,
      RecordSet* record_set) final;

  absl::Status SelectTypesByNamesAndVersions(
      absl::Span<std::pair<std::string, std::string>> names_and_versions,
      TypeKind type_kind, RecordSet* record_set) final;

  absl::Status SelectAllTypes(TypeKind type_kind, RecordSet* record_set) final;

  absl::Status UpdateTypeExternalIdDirect(
      int64_t type_id, std::optional<absl::string_view> external_id) final {
    MLMD_RETURN_IF_ERROR(
        VerifyCurrentQueryVersionIsAtLeast(kSchemaVersionNine));
    return ExecuteQuery(query_config_.update_type(),
                        {Bind(type_id), Bind(external_id)});
  }

  absl::Status CheckTypePropertyTable() final;

  absl::Status InsertTypeProperty(int64_t type_id,
                                  absl::string_view property_name,
                                  PropertyType property_type) final {
    return ExecuteQuery(
        query_config_.insert_type_property(),
        {Bind(type_id), Bind(property_name), Bind(property_type)});
  }

  absl::Status SelectPropertiesByTypeID(absl::Span<const int64_t> type_ids,
                                        RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_properties_by_type_id(),
                        {Bind(type_ids)}, record_set);
  }

  absl::Status CheckParentTypeTable() final;

  absl::Status InsertParentType(int64_t type_id, int64_t parent_type_id) final;

  absl::Status DeleteParentType(int64_t type_id, int64_t parent_type_id) final;

  absl::Status SelectParentTypesByTypeID(absl::Span<const int64_t> type_ids,
                                         RecordSet* record_set) final;

  // Gets the last inserted id.
  absl::Status SelectLastInsertID(int64_t* id);

  absl::Status CheckArtifactTable() final;

  absl::Status InsertArtifact(int64_t type_id, const std::string& artifact_uri,
                              const std::optional<Artifact::State>& state,
                              const std::optional<std::string>& name,
                              std::optional<absl::string_view> external_id,
                              const absl::Time create_time,
                              const absl::Time update_time,
                              int64_t* artifact_id) final {
    return ExecuteQuerySelectLastInsertID(
        query_config_.insert_artifact(),
        {Bind(type_id), Bind(artifact_uri), Bind(state), Bind(name),
         Bind(external_id), Bind(absl::ToUnixMillis(create_time)),
         Bind(absl::ToUnixMillis(update_time))},
        artifact_id);
  }

  absl::Status SelectArtifactsByID(absl::Span<const int64_t> artifact_ids,
                                   RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_artifact_by_id(),
                        {Bind(artifact_ids)}, record_set);
  }

  absl::Status SelectArtifactsByExternalIds(
      absl::Span<absl::string_view> external_ids, RecordSet* record_set) {
    MLMD_RETURN_IF_ERROR(
        VerifyCurrentQueryVersionIsAtLeast(kSchemaVersionNine));
    return ExecuteQuery(query_config_.select_artifacts_by_external_ids(),
                        {Bind(external_ids)}, record_set);
  }

  absl::Status SelectArtifactByTypeIDAndArtifactName(
      int64_t artifact_type_id, absl::string_view name,
      RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_artifact_by_type_id_and_name(),
                        {Bind(artifact_type_id), Bind(name)}, record_set);
  }

  absl::Status SelectArtifactsByTypeID(int64_t artifact_type_id,
                                       RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_artifacts_by_type_id(),
                        {Bind(artifact_type_id)}, record_set);
  }

  absl::Status SelectArtifactsByURI(absl::string_view uri,
                                    RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_artifacts_by_uri(), {Bind(uri)},
                        record_set);
  }

  absl::Status UpdateArtifactDirect(
      int64_t artifact_id, int64_t type_id, const std::string& uri,
      const std::optional<Artifact::State>& state,
      std::optional<absl::string_view> external_id,
      const absl::Time update_time) final {
    return ExecuteQuery(
        query_config_.update_artifact(),
        {Bind(artifact_id), Bind(type_id), Bind(uri), Bind(state),
         Bind(external_id), Bind(absl::ToUnixMillis(update_time))});
  }

  absl::Status CheckArtifactPropertyTable() final;

  absl::Status InsertArtifactProperty(int64_t artifact_id,
                                      absl::string_view artifact_property_name,
                                      bool is_custom_property,
                                      const Value& property_value) final {
    return ExecuteQuery(query_config_.insert_artifact_property(),
                        {BindDataType(property_value), Bind(artifact_id),
                         Bind(artifact_property_name), Bind(is_custom_property),
                         BindValue(property_value)});
  }

  absl::Status SelectArtifactPropertyByArtifactID(
      absl::Span<const int64_t> artifact_ids, RecordSet* record_set) final {
    MetadataSourceQueryConfig::TemplateQuery
        select_artifact_property_by_artifact_id;
    select_artifact_property_by_artifact_id =
        query_config_.select_artifact_property_by_artifact_id();
    return ExecuteQuery(select_artifact_property_by_artifact_id,
                        {Bind(artifact_ids)}, record_set);
  }

  absl::Status UpdateArtifactProperty(int64_t artifact_id,
                                      absl::string_view property_name,
                                      const Value& property_value) final {
    return ExecuteQuery(
        query_config_.update_artifact_property(),
        {BindDataType(property_value), BindValue(property_value),
         Bind(artifact_id), Bind(property_name)});
  }

  absl::Status DeleteArtifactProperty(int64_t artifact_id,
                                      absl::string_view property_name) final {
    return ExecuteQuery(query_config_.delete_artifact_property(),
                        {Bind(artifact_id), Bind(property_name)});
  }

  absl::Status CheckExecutionTable() final;

  absl::Status InsertExecution(
      int64_t type_id, const std::optional<Execution::State>& last_known_state,
      const std::optional<std::string>& name,
      std::optional<absl::string_view> external_id,
      const absl::Time create_time, const absl::Time update_time,
      int64_t* execution_id) final {
    return ExecuteQuerySelectLastInsertID(
        query_config_.insert_execution(),
        {Bind(type_id), Bind(last_known_state), Bind(name), Bind(external_id),
         Bind(absl::ToUnixMillis(create_time)),
         Bind(absl::ToUnixMillis(update_time))},
        execution_id);
  }

  absl::Status SelectExecutionsByID(absl::Span<const int64_t> execution_ids,
                                    RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_execution_by_id(),
                        {Bind(execution_ids)}, record_set);
  }

  absl::Status SelectExecutionsByExternalIds(absl::Span<absl::string_view> ids,
                                             RecordSet* record_set) final {
    MLMD_RETURN_IF_ERROR(
        VerifyCurrentQueryVersionIsAtLeast(kSchemaVersionNine));
    return ExecuteQuery(query_config_.select_executions_by_external_ids(),
                        {Bind(ids)}, record_set);
  }

  absl::Status SelectExecutionByTypeIDAndExecutionName(
      int64_t execution_type_id, absl::string_view name,
      RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_execution_by_type_id_and_name(),
                        {Bind(execution_type_id), Bind(name)}, record_set);
  }

  absl::Status SelectExecutionsByTypeID(int64_t execution_type_id,
                                        RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_executions_by_type_id(),
                        {Bind(execution_type_id)}, record_set);
  }

  absl::Status UpdateExecutionDirect(
      int64_t execution_id, int64_t type_id,
      const std::optional<Execution::State>& last_known_state,
      std::optional<absl::string_view> external_id,
      const absl::Time update_time) final {
    return ExecuteQuery(
        query_config_.update_execution(),
        {Bind(execution_id), Bind(type_id), Bind(last_known_state),
         Bind(external_id), Bind(absl::ToUnixMillis(update_time))});
  }

  absl::Status CheckExecutionPropertyTable() final;

  absl::Status InsertExecutionProperty(int64_t execution_id,
                                       absl::string_view name,
                                       bool is_custom_property,
                                       const Value& value) final {
    return ExecuteQuery(query_config_.insert_execution_property(),
                        {BindDataType(value), Bind(execution_id), Bind(name),
                         Bind(is_custom_property), BindValue(value)});
  }

  absl::Status SelectExecutionPropertyByExecutionID(
      absl::Span<const int64_t> execution_ids, RecordSet* record_set) final {
    MetadataSourceQueryConfig::TemplateQuery
        select_execution_property_by_execution_id;
    select_execution_property_by_execution_id =
        query_config_.select_execution_property_by_execution_id();
    return ExecuteQuery(select_execution_property_by_execution_id,
                        {Bind(execution_ids)}, record_set);
  }

  absl::Status UpdateExecutionProperty(int64_t execution_id,
                                       absl::string_view name,
                                       const Value& value) final {
    return ExecuteQuery(query_config_.update_execution_property(),
                        {BindDataType(value), BindValue(value),
                         Bind(execution_id), Bind(name)});
  }

  absl::Status DeleteExecutionProperty(int64_t execution_id,
                                       absl::string_view name) final {
    return ExecuteQuery(query_config_.delete_execution_property(),
                        {Bind(execution_id), Bind(name)});
  }

  absl::Status CheckContextTable() final;

  absl::Status InsertContext(int64_t type_id, const std::string& name,
                             std::optional<absl::string_view> external_id,
                             const absl::Time create_time,
                             const absl::Time update_time,
                             int64_t* context_id) final {
    return ExecuteQuerySelectLastInsertID(
        query_config_.insert_context(),
        {Bind(type_id), Bind(name), Bind(external_id),
         Bind(absl::ToUnixMillis(create_time)),
         Bind(absl::ToUnixMillis(update_time))},
        context_id);
  }

  absl::Status SelectContextsByID(absl::Span<const int64_t> context_ids,
                                  RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_context_by_id(),
                        {Bind(context_ids)}, record_set);
  }

  absl::Status SelectContextsByExternalIds(
      absl::Span<absl::string_view> external_ids, RecordSet* record_set) final {
    MLMD_RETURN_IF_ERROR(
        VerifyCurrentQueryVersionIsAtLeast(kSchemaVersionNine));
    return ExecuteQuery(query_config_.select_contexts_by_external_ids(),
                        {Bind(external_ids)}, record_set);
  }

  absl::Status SelectContextsByTypeID(int64_t context_type_id,
                                      RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_contexts_by_type_id(),
                        {Bind(context_type_id)}, record_set);
  }

  absl::Status SelectContextByTypeIDAndContextName(
      int64_t context_type_id, absl::string_view name,
      RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_context_by_type_id_and_name(),
                        {Bind(context_type_id), Bind(name)}, record_set);
  }

  absl::Status UpdateContextDirect(int64_t existing_context_id, int64_t type_id,
                                   const std::string& context_name,
                                   std::optional<absl::string_view> external_id,
                                   const absl::Time update_time) final {
    return ExecuteQuery(
        query_config_.update_context(),
        {Bind(existing_context_id), Bind(type_id), Bind(context_name),
         Bind(external_id), Bind(absl::ToUnixMillis(update_time))});
  }

  absl::Status CheckContextPropertyTable() final;

  absl::Status InsertContextProperty(int64_t context_id, absl::string_view name,
                                     bool custom_property,
                                     const Value& value) final {
    return ExecuteQuery(query_config_.insert_context_property(),
                        {BindDataType(value), Bind(context_id), Bind(name),
                         Bind(custom_property), BindValue(value)});
  }

  absl::Status SelectContextPropertyByContextID(
      absl::Span<const int64_t> context_ids, RecordSet* record_set) final {
    // TODO(b/257334039): Cleanup the fat-client after fully migrated to V10+.
    MetadataSourceQueryConfig::TemplateQuery
        select_context_property_by_context_id;

    select_context_property_by_context_id =
        query_config_.select_context_property_by_context_id();
    return ExecuteQuery(select_context_property_by_context_id,
                        {Bind(context_ids)}, record_set);
  }

  absl::Status UpdateContextProperty(int64_t context_id,
                                     absl::string_view property_name,
                                     const Value& property_value) final {
    return ExecuteQuery(
        query_config_.update_context_property(),
        {BindDataType(property_value), BindValue(property_value),
         Bind(context_id), Bind(property_name)});
  }

  absl::Status DeleteContextProperty(const int64_t context_id,
                                     absl::string_view property_name) final {
    return ExecuteQuery(query_config_.delete_context_property(),
                        {Bind(context_id), Bind(property_name)});
  }

  absl::Status CheckEventTable() final;

  absl::Status InsertEvent(int64_t artifact_id, int64_t execution_id,
                           int event_type, int64_t event_time_milliseconds,
                           int64_t* event_id) final {
    return ExecuteQuerySelectLastInsertID(
        query_config_.insert_event(),
        {Bind(artifact_id), Bind(execution_id), Bind(event_type),
         Bind(event_time_milliseconds)},
        event_id);
  }

  absl::Status SelectEventByArtifactIDs(absl::Span<const int64_t> artifact_ids,
                                        RecordSet* event_record_set) final {
    return ExecuteQuery(query_config_.select_event_by_artifact_ids(),
                        {Bind(artifact_ids)}, event_record_set);
  }

  absl::Status SelectEventByExecutionIDs(
      absl::Span<const int64_t> execution_ids,
      RecordSet* event_record_set) final {
    return ExecuteQuery(query_config_.select_event_by_execution_ids(),
                        {Bind(execution_ids)}, event_record_set);
  }

  absl::Status CheckEventPathTable() final;

  absl::Status InsertEventPath(int64_t event_id,
                               const Event::Path::Step& step) final;

  absl::Status SelectEventPathByEventIDs(absl::Span<const int64_t> event_ids,
                                         RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_event_path_by_event_ids(),
                        {Bind(event_ids)}, record_set);
  }

  absl::Status CheckAssociationTable() final;

  absl::Status InsertAssociation(int64_t context_id, int64_t execution_id,
                                 int64_t* association_id) final;

  absl::Status SelectAssociationByContextIDs(
      absl::Span<const int64_t> context_id, RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_association_by_context_id(),
                        {Bind(context_id)}, record_set);
  }

  absl::Status SelectAssociationsByExecutionIds(
      absl::Span<const int64_t> execution_ids, RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_associations_by_execution_ids(),
                        {Bind(execution_ids)}, record_set);
  }

  absl::Status CheckAttributionTable() final;

  absl::Status InsertAttributionDirect(int64_t context_id, int64_t artifact_id,
                                       int64_t* attribution_id) final;

  absl::Status SelectAttributionByContextID(int64_t context_id,
                                            RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_attribution_by_context_id(),
                        {Bind(context_id)}, record_set);
  }

  absl::Status SelectAttributionsByArtifactIds(
      absl::Span<const int64_t> artifact_ids, RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_attributions_by_artifact_ids(),
                        {Bind(artifact_ids)}, record_set);
  }

  absl::Status CheckParentContextTable() final;

  absl::Status InsertParentContext(int64_t parent_id, int64_t child_id) final;

  absl::Status SelectParentContextsByContextIDs(
      absl::Span<const int64_t> context_id, RecordSet* record_set) final;

  absl::Status SelectChildContextsByContextIDs(
      absl::Span<const int64_t> context_id, RecordSet* record_set) final;

  absl::Status SelectParentContextsByContextID(int64_t context_id,
                                               RecordSet* record_set) final;

  absl::Status SelectChildContextsByContextID(int64_t context_id,
                                              RecordSet* record_set) final;

  absl::Status CheckMLMDEnvTable() final;

  // Insert the schema version.
  absl::Status InsertSchemaVersion(int64_t schema_version) final {
    return ExecuteQuery(query_config_.insert_schema_version(),
                        {Bind(schema_version)});
  }

  // Update the schema version.
  absl::Status UpdateSchemaVersion(int64_t schema_version) final {
    return ExecuteQuery(query_config_.update_schema_version(),
                        {Bind(schema_version)});
  }

  absl::Status CheckTablesIn_V0_13_2() final;

  absl::Status SelectAllArtifactIDs(RecordSet* set) {
    return ExecuteQuery("select id from Artifact;", set);
  }

  absl::Status SelectAllExecutionIDs(RecordSet* set) {
    return ExecuteQuery("select id from Execution;", set);
  }

  absl::Status SelectAllContextIDs(RecordSet* set) {
    return ExecuteQuery("select id from Context;", set);
  }

  int64_t GetLibraryVersion() final {
    CHECK_GT(query_config_.schema_version(), 0);
    return query_config_.schema_version();
  }

  absl::Status DowngradeMetadataSource(const int64_t to_schema_version) final;

  absl::Status ListArtifactIDsUsingOptions(
      const ListOperationOptions& options,
      std::optional<absl::Span<const int64_t>> candidate_ids,
      RecordSet* record_set) final;

  absl::Status ListExecutionIDsUsingOptions(
      const ListOperationOptions& options,
      std::optional<absl::Span<const int64_t>> candidate_ids,
      RecordSet* record_set) final;

  absl::Status ListContextIDsUsingOptions(
      const ListOperationOptions& options,
      std::optional<absl::Span<const int64_t>> candidate_ids,
      RecordSet* record_set) final;


  absl::Status DeleteArtifactsById(
      absl::Span<const int64_t> artifact_ids) final;

  absl::Status DeleteContextsById(absl::Span<const int64_t> context_ids) final;

  absl::Status DeleteExecutionsById(
      absl::Span<const int64_t> execution_ids) final;

  absl::Status DeleteEventsByArtifactsId(
      absl::Span<const int64_t> artifact_ids) final;

  absl::Status DeleteEventsByExecutionsId(
      absl::Span<const int64_t> execution_ids) final;

  absl::Status DeleteAssociationsByContextsId(
      absl::Span<const int64_t> context_ids) final;

  absl::Status DeleteAssociationsByExecutionsId(
      absl::Span<const int64_t> execution_ids) final;

  absl::Status DeleteAttributionsByContextsId(
      absl::Span<const int64_t> context_ids) final;

  absl::Status DeleteAttributionsByArtifactsId(
      absl::Span<const int64_t> artifact_ids) final;

  absl::Status DeleteParentContextsByParentIds(
      absl::Span<const int64_t> parent_context_ids) final;

  absl::Status DeleteParentContextsByChildIds(
      absl::Span<const int64_t> child_context_ids) final;

  absl::Status DeleteParentContextsByParentIdAndChildIds(
      int64_t parent_context_id,
      absl::Span<const int64_t> child_context_ids) final;

 private:
  // Utility method to bind an nullable value.
  template <typename T>
  std::string Bind(const std::optional<T>& v) {
    return v ? Bind(v.value()) : "NULL";
  }

  // Utility method to bind an string_view value to a SQL clause.
  std::string Bind(absl::string_view value);

  // Utility method to bind an string_view value to a SQL clause.
  std::string Bind(const char* value);

  // Utility method to bind an int value to a SQL clause.
  std::string Bind(int value);

  // Utility method to bind an int64_t value to a SQL clause.
  std::string Bind(int64_t value);

  // Utility method to bind a boolean value to a SQL clause.
  std::string Bind(bool value);

  // Utility method to bind a double value to a SQL clause.
  std::string Bind(const double value);

  // Utility method to bind a proto value to a SQL clause.
  std::string Bind(const google::protobuf::Any& value);

  // Utility method to bind an PropertyType enum value to a SQL clause.
  // PropertyType is an enum (integer), EscapeString is not applicable.
  std::string Bind(const PropertyType value);

  // Utility method to bind an Event::Type enum value to a SQL clause.
  // Event::Type is an enum (integer), EscapeString is not applicable.
  std::string Bind(const Event::Type value);

  // Utility methods to bind the value to a SQL clause.
  std::string BindValue(const Value& value);
  std::string BindDataType(const Value& value);
  std::string Bind(const ArtifactStructType* message);

  // Utility method to bind an TypeKind to a SQL clause.
  // TypeKind is an enum (integer), EscapeString is not applicable.
  std::string Bind(TypeKind value);

  // Utility methods to bind Artifact::State/Execution::State to SQL clause.
  std::string Bind(Artifact::State value);
  std::string Bind(Execution::State value);

  // Utility method to bind an in64 vector to a string joined with "," that can
  // fit into SQL IN(...) clause.
  std::string Bind(absl::Span<const int64_t> value);

  // Utility method to bind a string_view vector to a string joined with ","
  // that can fit into SQL IN(...) clause.
  std::string Bind(absl::Span<absl::string_view> value);

  // Utility method to bind a pair<string_view, string_view> vector to a string
  // joined with "," that can fit into SQL IN(...) clause.
  std::string Bind(
      absl::Span<std::pair<absl::string_view, absl::string_view>> value);

  // Executes a template query. All strings in parameters should already be
  // in a format appropriate for the SQL variant being used (at this point,
  // they are just inserted).
  // Results consist of zero or more rows represented in RecordSet.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  absl::Status ExecuteQuery(
      const MetadataSourceQueryConfig::TemplateQuery& template_query,
      absl::Span<const std::string> parameters, RecordSet* record_set);

  // Executes a template query and ignore the result.
  // All strings in parameters should already be in a format appropriate for the
  // SQL variant being used (at this point, they are just inserted).
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  absl::Status ExecuteQuery(
      const MetadataSourceQueryConfig::TemplateQuery& template_query,
      absl::Span<const std::string> parameters) {
    RecordSet record_set;
    return ExecuteQuery(template_query, parameters, &record_set);
  }

  // Executes a template query without arguments and ignore the result.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  absl::Status ExecuteQuery(
      const MetadataSourceQueryConfig::TemplateQuery& query) {
    return ExecuteQuery(query, {});
  }

  // Executes a template query without arguments and ignore the result.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  // Returns INTERNAL error, if it cannot find the last insert ID.
  absl::Status ExecuteQuerySelectLastInsertID(
      const MetadataSourceQueryConfig::TemplateQuery& query,
      absl::Span<const std::string> arguments, int64_t* last_insert_id) {
    MLMD_RETURN_IF_ERROR(ExecuteQuery(query, arguments));
    return SelectLastInsertID(last_insert_id);
  }

  // Executes a query without arguments.
  // Results consist of zero or more rows represented in RecordSet.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  absl::Status ExecuteQuery(const std::string& query, RecordSet* record_set);

  // Executes a query without arguments and ignore the result.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  // Returns INTERNAL error, if it cannot find the last insert ID.
  absl::Status ExecuteQuery(const std::string& query);

  // Tests if the database version is compatible with the library version.
  // The database version and library version must be from the current
  // database.
  //
  // Returns OK.
  absl::Status IsCompatible(int64_t db_version, int64_t lib_version,
                            bool* is_compatible);

  // Upgrades the database schema version (db_v) to align with the library
  // schema version (lib_v). It retrieves db_v from the metadata source and
  // compares it with the lib_v in the given query_config, and runs
  // migration queries if db_v < lib_v. Returns FAILED_PRECONDITION error,
  // if db_v > lib_v for the case that the
  //   user use a database produced by a newer version of the library. In
  //   that case, downgrading the database may result in data loss. Often
  //   upgrading the library is required.
  // Returns DATA_LOSS error, if schema version table exists but no value
  // found. Returns DATA_LOSS error, if the database is not a 0.13.2 release
  // database
  //   and the schema version cannot be resolved.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status UpgradeMetadataSourceIfOutOfDate(bool enable_migration);

  // Lists Node IDs using `options` and `candidate_ids`. Template parameter
  // `Node` specifies the table to use for listing. If `candidate_ids` is not
  // empty then result set is constructed using only ids specified in
  // `candidate_ids`.
  // On success `record_set` is updated with Node IDs.
  // The `filter_query` field is supported for Artifacts.
  // Returns INVALID_ARGUMENT errors if the query specified is invalid.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Node>
  absl::Status ListNodeIDsUsingOptions(
      const ListOperationOptions& options,
      std::optional<absl::Span<const int64_t>> candidate_ids,
      RecordSet* record_set);

  MetadataSourceQueryConfig query_config_;

  // This object does not own the MetadataSource.
  MetadataSource* metadata_source_;

  // Delegates EncodeBytes to metadata_source_
  // Encodes value and returns the result as a string
  std::string EncodeBytes(absl::string_view value) const {
    return metadata_source_->EncodeBytes(value);
  }

  // Delegates DecodeBytes to metadata_source_
  // Decodes value and writes the result to dest
  // Returns OkStatus if the process succeeded, otherwise an informative error
  absl::Status DecodeBytes(absl::string_view value, std::string& dest) const {
    absl::StatusOr<std::string> decoded_or =
        metadata_source_->DecodeBytes(value);
    if (!decoded_or.ok()) return decoded_or.status();
    dest = decoded_or.value();
    return absl::OkStatus();
  }
};

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_POSTGRESQL_QUERY_EXECUTOR_H_
