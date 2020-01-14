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
#ifndef ML_METADATA_METADATA_STORE_QUERY_CONFIG_EXECUTOR_H_
#define ML_METADATA_METADATA_STORE_QUERY_CONFIG_EXECUTOR_H_

#include <memory>
#include <vector>

#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// A SQL version of the QueryExecutor. The text of most queries are
// encoded in MetadataSourceQueryConfig. This class binds the relevant arguments
// for each query using the Bind() methods. See notes on constructor for various
// ways to construct this object.
class QueryConfigExecutor : public QueryExecutor {
 public:
  // Note that the query config and the MetadataSource must be compatible.
  // MetadataSourceQueryConfigs can be created using the methods in
  // ml_metadata/util/metadata_source_query_config.h.
  // For example:
  // 1. If you use MySqlMetadataSource, use
  //    util::GetMySqlMetadataSourceQueryConfig().
  // 2. If you use SqliteMetadataSource, use
  //    util::GetSqliteMetadataSourceQueryConfig().
  //
  // The MetadataSource is not owned by this object, and must outlast it.
  QueryConfigExecutor(const MetadataSourceQueryConfig& query_config,
                      MetadataSource* source)
      : query_config_(query_config), metadata_source_(source) {}

  // default & copy constructors are disallowed.
  QueryConfigExecutor() = delete;
  QueryConfigExecutor(const QueryConfigExecutor&) = delete;
  QueryConfigExecutor& operator=(const QueryConfigExecutor&) = delete;

  virtual ~QueryConfigExecutor() = default;

  tensorflow::Status InitMetadataSource() final;

  tensorflow::Status InitMetadataSourceIfNotExists(
      const bool enable_upgrade_migration) final;

  tensorflow::Status GetSchemaVersion(int64* db_version) final;

  tensorflow::Status CheckTypeTable() final {
    return ExecuteQuery(query_config_.check_type_table());
  }

  tensorflow::Status InsertArtifactType(const string& name,
                                        int64* artifact_type_id) final {
    return ExecuteQuerySelectLastInsertID(query_config_.insert_artifact_type(),
                                          {Bind(name)}, artifact_type_id);
  }

  tensorflow::Status InsertExecutionType(const string& type_name,
                                         bool has_input_type,
                                         const google::protobuf::Message& input_type,
                                         bool has_output_type,
                                         const google::protobuf::Message& output_type,
                                         int64* execution_type_id) final;

  tensorflow::Status InsertContextType(const string& type_name,
                                       int64* context_id) final {
    return ExecuteQuerySelectLastInsertID(query_config_.insert_context_type(),
                                          {Bind(type_name)}, context_id);
  }

  tensorflow::Status SelectTypeByID(int64 type_id, TypeKind type_kind,
                                    RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_type_by_id(),
                        {Bind(type_id), Bind(type_kind)}, record_set);
  }

  tensorflow::Status SelectTypeByName(const absl::string_view type_name,
                                      TypeKind type_kind,
                                      RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_type_by_name(),
                        {Bind(type_name), Bind(type_kind)}, record_set);
  }

  tensorflow::Status SelectAllTypes(TypeKind type_kind,
                                    RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_all_types(), {Bind(type_kind)},
                        record_set);
  }

  tensorflow::Status CheckTypePropertyTable() final {
    return ExecuteQuery(query_config_.check_type_property_table());
  }

  tensorflow::Status InsertTypeProperty(int64 type_id,
                                        const absl::string_view property_name,
                                        PropertyType property_type) final {
    return ExecuteQuery(
        query_config_.insert_type_property(),
        {Bind(type_id), Bind(property_name), Bind(property_type)});
  }

  tensorflow::Status SelectPropertyByTypeID(int64 type_id,
                                            RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_property_by_type_id(),
                        {Bind(type_id)}, record_set);
  }

  // Queries the last inserted id.
  tensorflow::Status SelectLastInsertID(int64* id);

  tensorflow::Status CheckArtifactTable() final {
    return ExecuteQuery(query_config_.check_artifact_table());
  }

  tensorflow::Status InsertArtifact(int64 type_id, const string& artifact_uri,
                                    int64* artifact_id) final {
    return ExecuteQuerySelectLastInsertID(query_config_.insert_artifact(),
                                          {Bind(type_id), Bind(artifact_uri)},
                                          artifact_id);
  }

  tensorflow::Status SelectArtifactByID(int64 artifact_id,
                                        RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_artifact_by_id(),
                        {Bind(artifact_id)}, record_set);
  }

  tensorflow::Status SelectArtifactsByTypeID(int64 artifact_type_id,
                                             RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_artifacts_by_type_id(),
                        {Bind(artifact_type_id)}, record_set);
  }

  tensorflow::Status SelectArtifactsByURI(const absl::string_view uri,
                                          RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_artifacts_by_uri(), {Bind(uri)},
                        record_set);
  }

  tensorflow::Status UpdateArtifactDirect(int64 artifact_id, int64 type_id,
                                          const string& uri) final {
    return ExecuteQuery(query_config_.update_artifact(),
                        {Bind(artifact_id), Bind(type_id), Bind(uri)});
  }

  tensorflow::Status CheckArtifactPropertyTable() final {
    return ExecuteQuery(query_config_.check_artifact_property_table());
  }

  tensorflow::Status InsertArtifactProperty(
      int64 artifact_id, absl::string_view artifact_property_name,
      bool is_custom_property, const Value& property_value) final {
    return ExecuteQuery(query_config_.insert_artifact_property(),
                        {BindDataType(property_value), Bind(artifact_id),
                         Bind(artifact_property_name), Bind(is_custom_property),
                         BindValue(property_value)});
  }

  tensorflow::Status SelectArtifactPropertyByArtifactID(
      int64 artifact_id, RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_artifact_property_by_artifact_id(),
                        {Bind(artifact_id)}, record_set);
  }

  tensorflow::Status UpdateArtifactProperty(
      int64 artifact_id, const absl::string_view property_name,
      const Value& property_value) final {
    return ExecuteQuery(
        query_config_.update_artifact_property(),
        {BindDataType(property_value), BindValue(property_value),
         Bind(artifact_id), Bind(property_name)});
  }

  tensorflow::Status DeleteArtifactProperty(
      int64 artifact_id, const absl::string_view property_name) final {
    return ExecuteQuery(query_config_.delete_artifact_property(),
                        {Bind(artifact_id), Bind(property_name)});
  }

  tensorflow::Status CheckExecutionTable() final {
    return ExecuteQuery(query_config_.check_execution_table());
  }

  tensorflow::Status InsertExecution(int64 type_id, int64* execution_id) final {
    return ExecuteQuerySelectLastInsertID(query_config_.insert_execution(),
                                          {Bind(type_id)}, execution_id);
  }

  tensorflow::Status SelectExecutionByID(int64 execution_id,
                                         RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_execution_by_id(),
                        {Bind(execution_id)}, record_set);
  }

  tensorflow::Status SelectExecutionsByTypeID(int64 execution_type_id,
                                              RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_executions_by_type_id(),
                        {Bind(execution_type_id)}, record_set);
  }

  tensorflow::Status UpdateExecutionDirect(int64 execution_id,
                                           int64 type_id) final {
    return ExecuteQuery(query_config_.update_execution(),
                        {Bind(execution_id), Bind(type_id)});
  }

  tensorflow::Status CheckExecutionPropertyTable() final {
    return ExecuteQuery(query_config_.check_execution_property_table());
  }

  tensorflow::Status InsertExecutionProperty(int64 execution_id,
                                             const absl::string_view name,
                                             bool is_custom_property,
                                             const Value& value) final {
    return ExecuteQuery(query_config_.insert_execution_property(),
                        {BindDataType(value), Bind(execution_id), Bind(name),
                         Bind(is_custom_property), BindValue(value)});
  }

  tensorflow::Status SelectExecutionPropertyByExecutionID(
      int64 execution_id, RecordSet* record_set) final {
    return ExecuteQuery(
        query_config_.select_execution_property_by_execution_id(),
        {Bind(execution_id)}, record_set);
  }

  tensorflow::Status UpdateExecutionProperty(int64 execution_id,
                                             const absl::string_view name,
                                             const Value& value) final {
    return ExecuteQuery(query_config_.update_execution_property(),
                        {BindDataType(value), BindValue(value),
                         Bind(execution_id), Bind(name)});
  }

  tensorflow::Status DeleteExecutionProperty(
      int64 execution_id, const absl::string_view name) final {
    return ExecuteQuery(query_config_.delete_execution_property(),
                        {Bind(execution_id), Bind(name)});
  }

  tensorflow::Status CheckContextTable() final {
    return ExecuteQuery(query_config_.check_context_table());
  }

  tensorflow::Status InsertContext(int64 type_id, const string& name,
                                   int64* context_id) final {
    return ExecuteQuerySelectLastInsertID(query_config_.insert_context(),
                                          {Bind(type_id), Bind(name)},
                                          context_id);
  }

  tensorflow::Status SelectContextByID(int64 context_id,
                                       RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_context_by_id(),
                        {Bind(context_id)}, record_set);
  }

  tensorflow::Status SelectContextsByTypeID(int64 context_type_id,
                                            RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_contexts_by_type_id(),
                        {Bind(context_type_id)}, record_set);
  }


  tensorflow::Status SelectContextByTypeIDAndName(
      int64 context_type_id,
      const absl::string_view name,
      RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_context_by_type_id_and_name(),
                        {Bind(context_type_id), Bind(name)}, record_set);
  }

  tensorflow::Status UpdateContextDirect(int64 existing_context_id,
                                         int64 type_id,
                                         const string& context_name) final {
    return ExecuteQuery(
        query_config_.update_context(),
        {Bind(existing_context_id), Bind(type_id), Bind(context_name)});
  }

  tensorflow::Status CheckContextPropertyTable() final {
    return ExecuteQuery(query_config_.check_context_property_table());
  }

  tensorflow::Status InsertContextProperty(int64 context_id,
                                           const absl::string_view name,
                                           bool custom_property,
                                           const Value& value) final {
    return ExecuteQuery(query_config_.insert_context_property(),
                        {BindDataType(value), Bind(context_id), Bind(name),
                         Bind(custom_property), BindValue(value)});
  }

  tensorflow::Status SelectContextPropertyByContextID(
      int64 context_id, RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_context_property_by_context_id(),
                        {Bind(context_id)}, record_set);
  }

  tensorflow::Status UpdateContextProperty(
      int64 context_id, const absl::string_view property_name,
      const Value& property_value) final {
    return ExecuteQuery(
        query_config_.update_context_property(),
        {BindDataType(property_value), BindValue(property_value),
         Bind(context_id), Bind(property_name)});
  }

  tensorflow::Status DeleteContextProperty(
      const int64 context_id, const absl::string_view property_name) final {
    return ExecuteQuery(query_config_.delete_context_property(),
                        {Bind(context_id), Bind(property_name)});
  }

  tensorflow::Status CheckEventTable() final {
    return ExecuteQuery(query_config_.check_event_table());
  }

  tensorflow::Status InsertEvent(int64 artifact_id, int64 execution_id,
                                 int event_type, int64 event_time_milliseconds,
                                 int64* event_id) final {
    return ExecuteQuerySelectLastInsertID(
        query_config_.insert_event(),
        {Bind(artifact_id), Bind(execution_id), Bind(event_type),
         Bind(event_time_milliseconds)},
        event_id);
  }

  tensorflow::Status SelectEventByArtifactID(
      int64 artifact_id, RecordSet* event_record_set) final {
    return ExecuteQuery(query_config_.select_event_by_artifact_id(),
                        {Bind(artifact_id)}, event_record_set);
  }

  tensorflow::Status SelectEventByExecutionID(
      int64 execution_id, RecordSet* event_record_set) final {
    return ExecuteQuery(query_config_.select_event_by_execution_id(),
                        {Bind(execution_id)}, event_record_set);
  }

  tensorflow::Status CheckEventPathTable() final {
    return ExecuteQuery(query_config_.check_event_path_table());
  }

  tensorflow::Status InsertEventPath(int64 event_id,
                                     const Event::Path::Step& step) final;

  tensorflow::Status SelectEventPathByEventID(int64 event_id,
                                              RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_event_path_by_event_id(),
                        {Bind(event_id)}, record_set);
  }

  tensorflow::Status CheckAssociationTable() final {
    return ExecuteQuery(query_config_.check_association_table());
  }

  tensorflow::Status InsertAssociation(int64 context_id, int64 execution_id,
                                       int64* association_id) final {
    return ExecuteQuerySelectLastInsertID(
        query_config_.insert_association(),
        {Bind(context_id), Bind(execution_id)}, association_id);
  }

  tensorflow::Status SelectAssociationByContextID(int64 context_id,
                                                  RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_association_by_context_id(),
                        {Bind(context_id)}, record_set);
  }

  tensorflow::Status SelectAssociationByExecutionID(
      int64 execution_id, RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_association_by_execution_id(),
                        {Bind(execution_id)}, record_set);
  }

  tensorflow::Status CheckAttributionTable() final {
    return ExecuteQuery(query_config_.check_attribution_table());
  }

  tensorflow::Status InsertAttributionDirect(int64 context_id,
                                             int64 artifact_id,
                                             int64* attribution_id) final {
    return ExecuteQuerySelectLastInsertID(query_config_.insert_attribution(),
                                          {Bind(context_id), Bind(artifact_id)},
                                          attribution_id);
  }

  tensorflow::Status SelectAttributionByContextID(int64 context_id,
                                                  RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_attribution_by_context_id(),
                        {Bind(context_id)}, record_set);
  }

  tensorflow::Status SelectAttributionByArtifactID(
      int64 artifact_id, RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_attribution_by_artifact_id(),
                        {Bind(artifact_id)}, record_set);
  }

  tensorflow::Status CheckMLMDEnvTable() final {
    return ExecuteQuery(query_config_.check_mlmd_env_table());
  }

  tensorflow::Status InsertSchemaVersion(int64 schema_version) final {
    return ExecuteQuery(query_config_.insert_schema_version(),
                        {Bind(schema_version)});
  }

  tensorflow::Status UpdateSchemaVersion(int64 schema_version) final {
    return ExecuteQuery(query_config_.update_schema_version(),
                        {Bind(schema_version)});
  }

  tensorflow::Status CheckTablesIn_V0_13_2() final;

  tensorflow::Status SelectAllArtifactIDs(RecordSet* set) final {
    return ExecuteQuery("select `id` from `Artifact`;", set);
  }

  tensorflow::Status SelectAllExecutionIDs(RecordSet* set) final {
    return ExecuteQuery("select `id` from `Execution`;", set);
  }

  tensorflow::Status SelectAllContextIDs(RecordSet* set) final {
    return ExecuteQuery("select `id` from `Context`;", set);
  }

  tensorflow::Status GetLibraryVersion(int64* library_version) final {
    CHECK_GT(query_config_.schema_version(), 0);
    *library_version = query_config_.schema_version();
    return tensorflow::Status::OK();
  }

  MetadataSource* metadata_source() final { return metadata_source_; }

  tensorflow::Status DowngradeMetadataSource(
      const int64 to_schema_version) final;

 private:
  // Utility method to bind an string_view value to a SQL clause.
  string Bind(absl::string_view value);

  // Utility method to bind an string_view value to a SQL clause.
  string Bind(const char* value);

  // Utility method to bind an int value to a SQL clause.
  string Bind(int value);

  // Utility method to bind an int64 value to a SQL clause.
  string Bind(int64 value);

  // Utility method to bind a boolean value to a SQL clause.
  string Bind(bool value);

  // Utility method to bind an double value to a SQL clause.
  string Bind(const double value);

  // Utility method to bind an PropertyType enum value to a SQL clause.
  // PropertyType is an enum (integer), EscapeString is not applicable.
  string Bind(const PropertyType value);

  // Utility method to bind an Event::Type enum value to a SQL clause.
  // Event::Type is an enum (integer), EscapeString is not applicable.
  string Bind(const Event::Type value);

  // Bind the value to a SQL clause.
  string BindValue(const Value& value);
  string BindDataType(const Value& value);
  string Bind(bool exists, const google::protobuf::Message& message);
  // Utility method to bind an TypeKind to a SQL clause.
  // TypeKind is an enum (integer), EscapeString is not applicable.
  string Bind(TypeKind value);

  #if (!defined(__APPLE__) && !defined(_WIN32))
  string Bind(const google::protobuf::int64 value);
  #endif

  // Execute a template query. All strings in parameters should already be
  // in a format appropriate for the SQL variant being used (at this point,
  // they are just inserted).
  // Results consist of zero or more rows represented in RecordSet.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  tensorflow::Status ExecuteQuery(
      const MetadataSourceQueryConfig::TemplateQuery& template_query,
      const std::vector<string>& parameters, RecordSet* record_set);

  // Execute a template query and ignore the result.
  // All strings in parameters should already be in a format appropriate for the
  // SQL variant being used (at this point, they are just inserted).
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  tensorflow::Status ExecuteQuery(
      const MetadataSourceQueryConfig::TemplateQuery& template_query,
      const std::vector<string>& parameters) {
    RecordSet record_set;
    return ExecuteQuery(template_query, parameters, &record_set);
  }

  // Execute a template query without arguments and ignore the result.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  tensorflow::Status ExecuteQuery(
      const MetadataSourceQueryConfig::TemplateQuery& query) {
    return ExecuteQuery(query, {});
  }

  // Execute a template query without arguments and ignore the result.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  // Returns INTERNAL error, if it cannot find the last insert ID.
  tensorflow::Status ExecuteQuerySelectLastInsertID(
      const MetadataSourceQueryConfig::TemplateQuery& query,
      const std::vector<string>& arguments, int64* last_insert_id) {
    TF_RETURN_IF_ERROR(ExecuteQuery(query, arguments));
    return SelectLastInsertID(last_insert_id);
  }

  // Execute a query without arguments.
  // Results consist of zero or more rows represented in RecordSet.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  tensorflow::Status ExecuteQuery(const string& query, RecordSet* record_set);

  // Execute a query without arguments and ignore the result.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  // Returns INTERNAL error, if it cannot find the last insert ID.
  tensorflow::Status ExecuteQuery(const string& query);

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
  // TODO(martinz): consider promoting to MetadataAccessObject.
  tensorflow::Status UpgradeMetadataSourceIfOutOfDate(bool enable_migration);

  MetadataSourceQueryConfig query_config_;

  // This object does not own the MetadataSource.
  MetadataSource* metadata_source_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_QUERY_CONFIG_EXECUTOR_H_
