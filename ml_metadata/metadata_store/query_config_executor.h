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
#include <string>
#include <vector>

#include <glog/logging.h>
#include "google/protobuf/text_format.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {

constexpr int kSchemaVersionNine = 9;
constexpr int kSchemaVersionTen = 10;

// Prepares a template query used for earlier query schema version.
inline absl::Status GetTemplateQueryOrDie(
    const std::string& query,
    MetadataSourceQueryConfig::TemplateQuery& output) {
  if (!google::protobuf::TextFormat::ParseFromString(query, &output)) {
    return absl::InternalError(absl::StrCat(
        "query: `", query, "`, cannot be parsed to a TemplateQuery."));
  }
  return absl::OkStatus();
}

// TODO(b/248836219): Cleanup the fat-client after fully migrated to V9+.
// A set of utilities that is used in MLMD fat client where it is possible that
// the lib is used to query different sources that are at different schema
// versions (vj < vi = head).
namespace entity_query {
namespace v7_and_v8 {

static constexpr char kInsertArtifact[] = R"pb(
  query: " INSERT INTO `Artifact`( "
         "   `type_id`, `uri`, `state`, `name`, `create_time_since_epoch`, "
         "   `last_update_time_since_epoch` "
         ") VALUES($0, $1, $2, $3, $4, $5);"
  parameter_num: 6
)pb";

static constexpr char kSelectArtifactByIdForMySQL[] = R"pb(
  query: " SELECT `id`, `type_id`, `uri`, `state`, `name`, "
         "        `create_time_since_epoch`, `last_update_time_since_epoch` "
         " FROM `Artifact` "
         " WHERE id IN ($0) LOCK IN SHARE MODE;  "
  parameter_num: 1
)pb";

static constexpr char kSelectArtifactByIdForSQLite[] = R"pb(
  query: " SELECT `id`, `type_id`, `uri`, `state`, `name`, "
         "        `create_time_since_epoch`, `last_update_time_since_epoch` "
         " FROM `Artifact` "
         " WHERE id IN ($0);  "
  parameter_num: 1
)pb";

static constexpr char kUpdateArtifact[] = R"pb(
  query: " UPDATE `Artifact` "
         " SET `type_id` = $1, `uri` = $2, `state` = $3, "
         "     `last_update_time_since_epoch` = $4 "
         " WHERE id = $0;"
  parameter_num: 5
)pb";

static constexpr char kInsertExecution[] = R"pb(
  query: " INSERT INTO `Execution`( "
         "   `type_id`, `last_known_state`, `name`, "
         "   `create_time_since_epoch`, `last_update_time_since_epoch` "
         ") VALUES($0, $1, $2, $3, $4);"
  parameter_num: 5
)pb";

static constexpr char kSelectExecutionByIdForMySQL[] = R"pb(
  query: " SELECT `id`, `type_id`, `last_known_state`, `name`, "
         "        `create_time_since_epoch`, `last_update_time_since_epoch` "
         " FROM `Execution` "
         " WHERE id IN ($0) LOCK IN SHARE MODE; "
  parameter_num: 1
)pb";

static constexpr char kSelectExecutionByIdForSQLite[] = R"pb(
  query: " SELECT `id`, `type_id`, `last_known_state`, `name`, "
         "        `create_time_since_epoch`, `last_update_time_since_epoch` "
         " FROM `Execution` "
         " WHERE id IN ($0); "
  parameter_num: 1
)pb";

static constexpr char kUpdateExecution[] = R"pb(
  query: " UPDATE `Execution` "
         " SET `type_id` = $1, `last_known_state` = $2, "
         "     `last_update_time_since_epoch` = $3 "
         " WHERE id = $0;"
  parameter_num: 4
)pb";

static constexpr char kInsertContext[] = R"pb(
  query: " INSERT INTO `Context`( "
         "   `type_id`, `name`, "
         "   `create_time_since_epoch`, `last_update_time_since_epoch` "
         ") VALUES($0, $1, $2, $3);"
  parameter_num: 4
)pb";

static constexpr char kSelectContextByIdForMySQL[] = R"pb(
  query: " SELECT `id`, `type_id`, `name`, `create_time_since_epoch`, "
         "        `last_update_time_since_epoch`"
         " from `Context` WHERE id IN ($0) LOCK IN SHARE MODE; "
  parameter_num: 1
)pb";

static constexpr char kSelectContextByIdForSQLite[] = R"pb(
  query: " SELECT `id`, `type_id`, `name`, `create_time_since_epoch`, "
         "        `last_update_time_since_epoch`"
         " from `Context` WHERE id IN ($0); "
  parameter_num: 1
)pb";

static constexpr char kUpdateContext[] = R"pb(
  query: " UPDATE `Context` "
         " SET `type_id` = $1, `name` = $2, "
         "     `last_update_time_since_epoch` = $3 "
         " WHERE id = $0;"
  parameter_num: 4
)pb";

static constexpr char kInsertArtifactType[] = R"pb(
  query: " INSERT INTO `Type`( "
         "   `name`, `type_kind`, `version`, `description` "
         ") VALUES($0, 1, $1, $2);"
  parameter_num: 3
)pb";

static constexpr char kInsertExecutionType[] = R"pb(
  query: " INSERT INTO `Type`( "
         "   `name`, `type_kind`, `version`, `description`, "
         "   `input_type`, `output_type` "
         ") VALUES($0, 0, $1, $2, $3, $4);"
  parameter_num: 5
)pb";

static constexpr char kInsertContextType[] = R"pb(
  query: " INSERT INTO `Type`( "
         "   `name`, `type_kind`, `version`, `description` "
         ") VALUES($0, 2, $1, $2);"
  parameter_num: 3
)pb";

static constexpr char kSelectTypesById[] = R"pb(
  query: " SELECT `id`, `name`, `version`, `description` "
         " FROM `Type` "
         " WHERE id IN ($0) and type_kind = $1; "
  parameter_num: 2
)pb";

static constexpr char kSelectTypeById[] = R"pb(
  query: " SELECT `id`, `name`, `version`, `description`, "
         "        `input_type`, `output_type` FROM `Type` "
         " WHERE id = $0 and type_kind = $1; "
  parameter_num: 2
)pb";

static constexpr char kSelectTypeByNameAndVersion[] = R"pb(
  query: " SELECT `id`, `name`, `version`, `description`, "
         "        `input_type`, `output_type` FROM `Type` "
         " WHERE name = $0 AND version = $1 AND type_kind = $2; "
  parameter_num: 3
)pb";

static constexpr char kSelectTypeByName[] = R"pb(
  query: " SELECT `id`, `name`, `version`, `description`, "
         "        `input_type`, `output_type` FROM `Type` "
         " WHERE name = $0 AND version IS NULL AND type_kind = $1; "
  parameter_num: 2
)pb";

}  // namespace v7_and_v8
}  // namespace entity_query

namespace property_query {
namespace v7_v8_v9 {

// BEGIN ArtifactProperty queries

static constexpr char kCheckArtifactPropertyTable[] = R"pb(
  query: " SELECT `artifact_id`, `name`, `is_custom_property`, "
         "        `int_value`, `double_value`, `string_value`, `byte_value` "
         " FROM `ArtifactProperty` LIMIT 1; "
)pb";

static constexpr char kSelectArtifactPropertyByArtifactId[] = R"pb(
  query: " SELECT `artifact_id` as `id`, `name` as `key`, "
         "        `is_custom_property`, "
         "        `int_value`, `double_value`, `string_value` "
         " from `ArtifactProperty` "
         " WHERE `artifact_id` IN ($0); "
  parameter_num: 1
)pb";

// END ArtifactProperty queries

// BEGIN ExecutionProperty queries

static constexpr char kCheckExecutionPropertyTable[] = R"pb(
  query: " SELECT `execution_id`, `name`, `is_custom_property`, "
         "        `int_value`, `double_value`, `string_value`, `byte_value` "
         " FROM `ExecutionProperty` LIMIT 1; "
)pb";

static constexpr char kSelectExecutionPropertyByExecutionId[] = R"pb(
  query: " SELECT `execution_id` as `id`, `name` as `key`, "
         "        `is_custom_property`, "
         "        `int_value`, `double_value`, `string_value` "
         " from `ExecutionProperty` "
         " WHERE `execution_id` IN ($0); "
  parameter_num: 1
)pb";

// END ExecutionProperty queries

// BEGIN ContextProperty queries

static constexpr char kCheckContextPropertyTable[] = R"pb(
  query: " SELECT `context_id`, `name`, `is_custom_property`, "
         "        `int_value`, `double_value`, `string_value`, `byte_value` "
         " FROM `ContextProperty` LIMIT 1; "
)pb";

static constexpr char kSelectContextPropertyByContextId[] = R"pb(
  query: " SELECT `context_id` as `id`, `name` as `key`, "
         "        `is_custom_property`, "
         "        `int_value`, `double_value`, `string_value` "
         " from `ContextProperty` "
         " WHERE `context_id` IN ($0); "
  parameter_num: 1
)pb";

// END ContextProperty queries

}  // namespace v7_v8_v9
}  // namespace property_query

// A SQL version of the QueryExecutor. The text of most queries are
// encoded in MetadataSourceQueryConfig. This class binds the relevant arguments
// for each query using the Bind() methods. See notes on constructor for various
// ways to construct this object.
//
// QueryConfigExecutor delegates two QueryExecutor methods:
//
// 1. std::string EncodeBytes(absl::string_view value, std::string& dest) const
// 2. absl::Status DecodeBytes(absl::string_view value, std::string& dest) const
//
// to the MetadataSource equivalent
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

  // A `query_version` can be passed to the QueryConfigExecutor to work with
  // an existing db with an earlier schema version.
  QueryConfigExecutor(const MetadataSourceQueryConfig& query_config,
                      MetadataSource* source, int64 query_version);

  // default & copy constructors are disallowed.
  QueryConfigExecutor() = delete;
  QueryConfigExecutor(const QueryConfigExecutor&) = delete;
  QueryConfigExecutor& operator=(const QueryConfigExecutor&) = delete;

  virtual ~QueryConfigExecutor() = default;

  absl::Status InitMetadataSource() final;

  absl::Status InitMetadataSourceIfNotExists(
      bool enable_upgrade_migration) final;

  absl::Status InitMetadataSourceLight(bool enable_new_store_creation) final {
    return absl::UnimplementedError(
        "InitMetadataSourceLight not supported for QueryConfigExecutor");
  }

  absl::Status DeleteMetadataSource() final {
    return absl::UnimplementedError(
        "DeleteMetadataSource not supported for QueryConfigExecutor");
  }

  absl::Status GetSchemaVersion(int64* db_version) final;

  absl::Status CheckTypeTable() final {
    return ExecuteQuery(query_config_.check_type_table());
  }

  absl::Status InsertArtifactType(const std::string& name,
                                  absl::optional<absl::string_view> version,
                                  absl::optional<absl::string_view> description,
                                  absl::optional<absl::string_view> external_id,
                                  int64* type_id) final;

  absl::Status InsertExecutionType(
      const std::string& name, absl::optional<absl::string_view> version,
      absl::optional<absl::string_view> description,
      const ArtifactStructType* input_type,
      const ArtifactStructType* output_type,
      absl::optional<absl::string_view> external_id, int64* type_id) final;

  absl::Status InsertContextType(const std::string& name,
                                 absl::optional<absl::string_view> version,
                                 absl::optional<absl::string_view> description,
                                 absl::optional<absl::string_view> external_id,
                                 int64* type_id) final;

  absl::Status SelectTypesByID(absl::Span<const int64> type_ids,
                               TypeKind type_kind, RecordSet* record_set) final;

  absl::Status SelectTypesByExternalIds(
      absl::Span<absl::string_view> external_ids, TypeKind type_kind,
      RecordSet* record_set) final;

  absl::Status SelectTypeByID(int64 type_id, TypeKind type_kind,
                              RecordSet* record_set) final;

  absl::Status SelectTypeByNameAndVersion(
      absl::string_view type_name,
      absl::optional<absl::string_view> type_version, TypeKind type_kind,
      RecordSet* record_set) final;

  absl::Status SelectTypesByNamesAndVersions(
      absl::Span<std::pair<std::string, std::string>> names_and_versions,
      TypeKind type_kind, RecordSet* record_set) final;

  absl::Status SelectAllTypes(TypeKind type_kind, RecordSet* record_set) final;

  absl::Status UpdateTypeExternalIdDirect(
      int64 type_id, absl::optional<absl::string_view> external_id) final {
    MLMD_RETURN_IF_ERROR(
        VerifyCurrentQueryVersionIsAtLeast(kSchemaVersionNine));
    return ExecuteQuery(query_config_.update_type(),
                        {Bind(type_id), Bind(external_id)});
  }

  absl::Status CheckTypePropertyTable() final {
    return ExecuteQuery(query_config_.check_type_property_table());
  }

  absl::Status InsertTypeProperty(int64 type_id,
                                  absl::string_view property_name,
                                  PropertyType property_type) final {
    return ExecuteQuery(
        query_config_.insert_type_property(),
        {Bind(type_id), Bind(property_name), Bind(property_type)});
  }

  absl::Status SelectPropertiesByTypeID(absl::Span<const int64> type_ids,
                                        RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_properties_by_type_id(),
                        {Bind(type_ids)}, record_set);
  }

  absl::Status CheckParentTypeTable() final;

  absl::Status InsertParentType(int64 type_id, int64 parent_type_id) final;

  absl::Status DeleteParentType(int64 type_id, int64 parent_type_id) final;

  absl::Status SelectParentTypesByTypeID(absl::Span<const int64> type_ids,
                                         RecordSet* record_set) final;

  // Gets the last inserted id.
  absl::Status SelectLastInsertID(int64* id);

  absl::Status CheckArtifactTable() final {
    return ExecuteQuery(query_config_.check_artifact_table());
  }

  absl::Status InsertArtifact(int64 type_id, const std::string& artifact_uri,
                              const absl::optional<Artifact::State>& state,
                              const absl::optional<std::string>& name,
                              absl::optional<absl::string_view> external_id,
                              const absl::Time create_time,
                              const absl::Time update_time,
                              int64* artifact_id) final {
    // TODO(b/248836219): Cleanup the fat-client after fully migrated to V9+.
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionNine) {
      MetadataSourceQueryConfig::TemplateQuery insert_artifact;
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          entity_query::v7_and_v8::kInsertArtifact, insert_artifact));
      return ExecuteQuerySelectLastInsertID(
          insert_artifact,
          {Bind(type_id), Bind(artifact_uri), Bind(state), Bind(name),
           Bind(absl::ToUnixMillis(create_time)),
           Bind(absl::ToUnixMillis(update_time))},
          artifact_id);
    }

    return ExecuteQuerySelectLastInsertID(
        query_config_.insert_artifact(),
        {Bind(type_id), Bind(artifact_uri), Bind(state), Bind(name),
         Bind(external_id), Bind(absl::ToUnixMillis(create_time)),
         Bind(absl::ToUnixMillis(update_time))},
        artifact_id);
  }

  absl::Status SelectArtifactsByID(absl::Span<const int64> artifact_ids,
                                   RecordSet* record_set) final {
    // TODO(b/248836219): Cleanup the fat-client after fully migrated to V9+.
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionNine) {
      MetadataSourceQueryConfig::TemplateQuery select_artifact_by_id;
      if (query_config_.metadata_source_type() ==
          MetadataSourceType::MYSQL_METADATA_SOURCE) {
        MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
            entity_query::v7_and_v8::kSelectArtifactByIdForMySQL,
            select_artifact_by_id));
      } else {
        MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
            entity_query::v7_and_v8::kSelectArtifactByIdForSQLite,
            select_artifact_by_id));
      }
      return ExecuteQuery(select_artifact_by_id, {Bind(artifact_ids)},
                          record_set);
    }
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
      int64 artifact_type_id, absl::string_view name,
      RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_artifact_by_type_id_and_name(),
                        {Bind(artifact_type_id), Bind(name)}, record_set);
  }

  absl::Status SelectArtifactsByTypeID(int64 artifact_type_id,
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
      int64 artifact_id, int64 type_id, const std::string& uri,
      const absl::optional<Artifact::State>& state,
      absl::optional<absl::string_view> external_id,
      const absl::Time update_time) final {
    // TODO(b/248836219): Cleanup the fat-client after fully migrated to V9+.
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionNine) {
      MetadataSourceQueryConfig::TemplateQuery update_artifact;
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          entity_query::v7_and_v8::kUpdateArtifact, update_artifact));
      return ExecuteQuery(update_artifact,
                          {Bind(artifact_id), Bind(type_id), Bind(uri),
                           Bind(state), Bind(absl::ToUnixMillis(update_time))});
    }
    return ExecuteQuery(
        query_config_.update_artifact(),
        {Bind(artifact_id), Bind(type_id), Bind(uri), Bind(state),
         Bind(external_id), Bind(absl::ToUnixMillis(update_time))});
  }

  absl::Status CheckArtifactPropertyTable() final {
    // TODO(b/257334039): Cleanup the fat-client after fully migrated to V10+.
    MetadataSourceQueryConfig::TemplateQuery check_artifact_property_table;
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionTen) {
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          property_query::v7_v8_v9::kCheckArtifactPropertyTable,
          check_artifact_property_table));
    } else {
      check_artifact_property_table =
          query_config_.check_artifact_property_table();
    }
    return ExecuteQuery(check_artifact_property_table);
  }

  absl::Status InsertArtifactProperty(int64 artifact_id,
                                      absl::string_view artifact_property_name,
                                      bool is_custom_property,
                                      const Value& property_value) final {
    return ExecuteQuery(query_config_.insert_artifact_property(),
                        {BindDataType(property_value), Bind(artifact_id),
                         Bind(artifact_property_name), Bind(is_custom_property),
                         BindValue(property_value)});
  }

  absl::Status SelectArtifactPropertyByArtifactID(
      absl::Span<const int64> artifact_ids, RecordSet* record_set) final {
    // TODO(b/257334039): Cleanup the fat-client after fully migrated to V10+.
    MetadataSourceQueryConfig::TemplateQuery
        select_artifact_property_by_artifact_id;
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionTen) {
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          property_query::v7_v8_v9::kSelectArtifactPropertyByArtifactId,
          select_artifact_property_by_artifact_id));
    } else {
      select_artifact_property_by_artifact_id =
          query_config_.select_artifact_property_by_artifact_id();
    }
    return ExecuteQuery(select_artifact_property_by_artifact_id,
                        {Bind(artifact_ids)}, record_set);
  }

  absl::Status UpdateArtifactProperty(int64 artifact_id,
                                      absl::string_view property_name,
                                      const Value& property_value) final {
    return ExecuteQuery(
        query_config_.update_artifact_property(),
        {BindDataType(property_value), BindValue(property_value),
         Bind(artifact_id), Bind(property_name)});
  }

  absl::Status DeleteArtifactProperty(int64 artifact_id,
                                      absl::string_view property_name) final {
    return ExecuteQuery(query_config_.delete_artifact_property(),
                        {Bind(artifact_id), Bind(property_name)});
  }

  absl::Status CheckExecutionTable() final {
    return ExecuteQuery(query_config_.check_execution_table());
  }

  absl::Status InsertExecution(
      int64 type_id, const absl::optional<Execution::State>& last_known_state,
      const absl::optional<std::string>& name,
      absl::optional<absl::string_view> external_id,
      const absl::Time create_time, const absl::Time update_time,
      int64* execution_id) final {
    // TODO(b/248836219): Cleanup the fat-client after fully migrated to V9+.
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionNine) {
      MetadataSourceQueryConfig::TemplateQuery insert_execution;
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          entity_query::v7_and_v8::kInsertExecution, insert_execution));
      return ExecuteQuerySelectLastInsertID(
          insert_execution,
          {Bind(type_id), Bind(last_known_state), Bind(name),
           Bind(absl::ToUnixMillis(create_time)),
           Bind(absl::ToUnixMillis(update_time))},
          execution_id);
    }
    return ExecuteQuerySelectLastInsertID(
        query_config_.insert_execution(),
        {Bind(type_id), Bind(last_known_state), Bind(name), Bind(external_id),
         Bind(absl::ToUnixMillis(create_time)),
         Bind(absl::ToUnixMillis(update_time))},
        execution_id);
  }

  absl::Status SelectExecutionsByID(absl::Span<const int64> execution_ids,
                                    RecordSet* record_set) final {
    // TODO(b/248836219): Cleanup the fat-client after fully migrated to V9+.
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionNine) {
      MetadataSourceQueryConfig::TemplateQuery select_execution_by_id;
      if (query_config_.metadata_source_type() ==
          MetadataSourceType::MYSQL_METADATA_SOURCE) {
        MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
            entity_query::v7_and_v8::kSelectExecutionByIdForMySQL,
            select_execution_by_id));

      } else {
        MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
            entity_query::v7_and_v8::kSelectExecutionByIdForSQLite,
            select_execution_by_id));
      }
      return ExecuteQuery(select_execution_by_id, {Bind(execution_ids)},
                          record_set);
    }
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
      int64 execution_type_id, absl::string_view name,
      RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_execution_by_type_id_and_name(),
                        {Bind(execution_type_id), Bind(name)}, record_set);
  }

  absl::Status SelectExecutionsByTypeID(int64 execution_type_id,
                                        RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_executions_by_type_id(),
                        {Bind(execution_type_id)}, record_set);
  }

  absl::Status UpdateExecutionDirect(
      int64 execution_id, int64 type_id,
      const absl::optional<Execution::State>& last_known_state,
      absl::optional<absl::string_view> external_id,
      const absl::Time update_time) final {
    // TODO(b/248836219): Cleanup the fat-client after fully migrated to V9+.
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionNine) {
      MetadataSourceQueryConfig::TemplateQuery update_execution;
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          entity_query::v7_and_v8::kUpdateExecution, update_execution));
      absl::Status s = ExecuteQuery(
          update_execution,
          {Bind(execution_id), Bind(type_id), Bind(last_known_state),
           Bind(absl::ToUnixMillis(update_time))});
      return s;
    }
    return ExecuteQuery(
        query_config_.update_execution(),
        {Bind(execution_id), Bind(type_id), Bind(last_known_state),
         Bind(external_id), Bind(absl::ToUnixMillis(update_time))});
  }

  absl::Status CheckExecutionPropertyTable() final {
    // TODO(b/257334039): Cleanup the fat-client after fully migrated to V10+.
    MetadataSourceQueryConfig::TemplateQuery check_execution_property_table;
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionTen) {
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          property_query::v7_v8_v9::kCheckExecutionPropertyTable,
          check_execution_property_table));
    } else {
      check_execution_property_table =
          query_config_.check_execution_property_table();
    }
    return ExecuteQuery(check_execution_property_table);
  }

  absl::Status InsertExecutionProperty(int64 execution_id,
                                       absl::string_view name,
                                       bool is_custom_property,
                                       const Value& value) final {
    return ExecuteQuery(query_config_.insert_execution_property(),
                        {BindDataType(value), Bind(execution_id), Bind(name),
                         Bind(is_custom_property), BindValue(value)});
  }

  absl::Status SelectExecutionPropertyByExecutionID(
      absl::Span<const int64> execution_ids, RecordSet* record_set) final {
    // TODO(b/257334039): Cleanup the fat-client after fully migrated to V10+.
    MetadataSourceQueryConfig::TemplateQuery
        select_execution_property_by_execution_id;
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionTen) {
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          property_query::v7_v8_v9::kSelectExecutionPropertyByExecutionId,
          select_execution_property_by_execution_id));
    } else {
      select_execution_property_by_execution_id =
          query_config_.select_execution_property_by_execution_id();
    }
    return ExecuteQuery(select_execution_property_by_execution_id,
                        {Bind(execution_ids)}, record_set);
  }

  absl::Status UpdateExecutionProperty(int64 execution_id,
                                       absl::string_view name,
                                       const Value& value) final {
    return ExecuteQuery(query_config_.update_execution_property(),
                        {BindDataType(value), BindValue(value),
                         Bind(execution_id), Bind(name)});
  }

  absl::Status DeleteExecutionProperty(int64 execution_id,
                                       absl::string_view name) final {
    return ExecuteQuery(query_config_.delete_execution_property(),
                        {Bind(execution_id), Bind(name)});
  }

  absl::Status CheckContextTable() final {
    return ExecuteQuery(query_config_.check_context_table());
  }

  absl::Status InsertContext(int64 type_id, const std::string& name,
                             absl::optional<absl::string_view> external_id,
                             const absl::Time create_time,
                             const absl::Time update_time,
                             int64* context_id) final {
    // TODO(b/248836219): Cleanup the fat-client after fully migrated to V9+.
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionNine) {
      MetadataSourceQueryConfig::TemplateQuery insert_context;
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          entity_query::v7_and_v8::kInsertContext, insert_context));
      return ExecuteQuerySelectLastInsertID(
          insert_context,
          {Bind(type_id), Bind(name), Bind(absl::ToUnixMillis(create_time)),
           Bind(absl::ToUnixMillis(update_time))},
          context_id);
    }
    return ExecuteQuerySelectLastInsertID(
        query_config_.insert_context(),
        {Bind(type_id), Bind(name), Bind(external_id),
         Bind(absl::ToUnixMillis(create_time)),
         Bind(absl::ToUnixMillis(update_time))},
        context_id);
  }

  absl::Status SelectContextsByID(absl::Span<const int64> context_ids,
                                  RecordSet* record_set) final {
    // TODO(b/248836219): Cleanup the fat-client after fully migrated to V9+.
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionNine) {
      MetadataSourceQueryConfig::TemplateQuery select_context_by_id;
      if (query_config_.metadata_source_type() ==
          MetadataSourceType::MYSQL_METADATA_SOURCE) {
        MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
            entity_query::v7_and_v8::kSelectContextByIdForMySQL,
            select_context_by_id));
      } else {
        MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
            entity_query::v7_and_v8::kSelectContextByIdForSQLite,
            select_context_by_id));
      }
      return ExecuteQuery(select_context_by_id, {Bind(context_ids)},
                          record_set);
    }
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

  absl::Status SelectContextsByTypeID(int64 context_type_id,
                                      RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_contexts_by_type_id(),
                        {Bind(context_type_id)}, record_set);
  }

  absl::Status SelectContextByTypeIDAndContextName(
      int64 context_type_id, absl::string_view name,
      RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_context_by_type_id_and_name(),
                        {Bind(context_type_id), Bind(name)}, record_set);
  }

  absl::Status UpdateContextDirect(
      int64 existing_context_id, int64 type_id, const std::string& context_name,
      absl::optional<absl::string_view> external_id,
      const absl::Time update_time) final {
    // TODO(b/248836219): Cleanup the fat-client after fully migrated to V9+.
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionNine) {
      MetadataSourceQueryConfig::TemplateQuery update_context;
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          entity_query::v7_and_v8::kUpdateContext, update_context));
      return ExecuteQuery(
          update_context,
          {Bind(existing_context_id), Bind(type_id), Bind(context_name),
           Bind(absl::ToUnixMillis(update_time))});
    }
    return ExecuteQuery(
        query_config_.update_context(),
        {Bind(existing_context_id), Bind(type_id), Bind(context_name),
         Bind(external_id), Bind(absl::ToUnixMillis(update_time))});
  }

  absl::Status CheckContextPropertyTable() final {
    // TODO(b/257334039): Cleanup the fat-client after fully migrated to V10+.
    MetadataSourceQueryConfig::TemplateQuery check_context_property_table;
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionTen) {
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          property_query::v7_v8_v9::kCheckContextPropertyTable,
          check_context_property_table));
    } else {
      check_context_property_table =
          query_config_.check_context_property_table();
    }
    return ExecuteQuery(check_context_property_table);
  }

  absl::Status InsertContextProperty(int64 context_id, absl::string_view name,
                                     bool custom_property,
                                     const Value& value) final {
    return ExecuteQuery(query_config_.insert_context_property(),
                        {BindDataType(value), Bind(context_id), Bind(name),
                         Bind(custom_property), BindValue(value)});
  }

  absl::Status SelectContextPropertyByContextID(
      absl::Span<const int64> context_ids, RecordSet* record_set) final {
    // TODO(b/257334039): Cleanup the fat-client after fully migrated to V10+.
    MetadataSourceQueryConfig::TemplateQuery
        select_context_property_by_context_id;
    if (query_schema_version().has_value() &&
        query_schema_version().value() < kSchemaVersionTen) {
      MLMD_RETURN_IF_ERROR(GetTemplateQueryOrDie(
          property_query::v7_v8_v9::kSelectContextPropertyByContextId,
          select_context_property_by_context_id));
    } else {
      select_context_property_by_context_id =
          query_config_.select_context_property_by_context_id();
    }
    return ExecuteQuery(select_context_property_by_context_id,
                        {Bind(context_ids)}, record_set);
  }

  absl::Status UpdateContextProperty(int64 context_id,
                                     absl::string_view property_name,
                                     const Value& property_value) final {
    return ExecuteQuery(
        query_config_.update_context_property(),
        {BindDataType(property_value), BindValue(property_value),
         Bind(context_id), Bind(property_name)});
  }

  absl::Status DeleteContextProperty(const int64 context_id,
                                     absl::string_view property_name) final {
    return ExecuteQuery(query_config_.delete_context_property(),
                        {Bind(context_id), Bind(property_name)});
  }

  absl::Status CheckEventTable() final {
    return ExecuteQuery(query_config_.check_event_table());
  }

  absl::Status InsertEvent(int64 artifact_id, int64 execution_id,
                           int event_type, int64 event_time_milliseconds,
                           int64* event_id) final {
    return ExecuteQuerySelectLastInsertID(
        query_config_.insert_event(),
        {Bind(artifact_id), Bind(execution_id), Bind(event_type),
         Bind(event_time_milliseconds)},
        event_id);
  }

  absl::Status SelectEventByArtifactIDs(absl::Span<const int64> artifact_ids,
                                        RecordSet* event_record_set) final {
    return ExecuteQuery(query_config_.select_event_by_artifact_ids(),
                        {Bind(artifact_ids)}, event_record_set);
  }

  absl::Status SelectEventByExecutionIDs(absl::Span<const int64> execution_ids,
                                         RecordSet* event_record_set) final {
    return ExecuteQuery(query_config_.select_event_by_execution_ids(),
                        {Bind(execution_ids)}, event_record_set);
  }

  absl::Status CheckEventPathTable() final {
    return ExecuteQuery(query_config_.check_event_path_table());
  }

  absl::Status InsertEventPath(int64 event_id,
                               const Event::Path::Step& step) final;

  absl::Status SelectEventPathByEventIDs(absl::Span<const int64> event_ids,
                                         RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_event_path_by_event_ids(),
                        {Bind(event_ids)}, record_set);
  }

  absl::Status CheckAssociationTable() final {
    return ExecuteQuery(query_config_.check_association_table());
  }

  absl::Status InsertAssociation(int64 context_id, int64 execution_id,
                                 int64* association_id) final {
    return ExecuteQuerySelectLastInsertID(
        query_config_.insert_association(),
        {Bind(context_id), Bind(execution_id)}, association_id);
  }

  absl::Status SelectAssociationByContextIDs(absl::Span<const int64> context_id,
                                             RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_association_by_context_id(),
                        {Bind(context_id)}, record_set);
  }

  absl::Status SelectAssociationByExecutionID(int64 execution_id,
                                              RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_association_by_execution_id(),
                        {Bind(execution_id)}, record_set);
  }

  absl::Status CheckAttributionTable() final {
    return ExecuteQuery(query_config_.check_attribution_table());
  }

  absl::Status InsertAttributionDirect(int64 context_id, int64 artifact_id,
                                       int64* attribution_id) final {
    return ExecuteQuerySelectLastInsertID(query_config_.insert_attribution(),
                                          {Bind(context_id), Bind(artifact_id)},
                                          attribution_id);
  }

  absl::Status SelectAttributionByContextID(int64 context_id,
                                            RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_attribution_by_context_id(),
                        {Bind(context_id)}, record_set);
  }

  absl::Status SelectAttributionByArtifactID(int64 artifact_id,
                                             RecordSet* record_set) final {
    return ExecuteQuery(query_config_.select_attribution_by_artifact_id(),
                        {Bind(artifact_id)}, record_set);
  }

  absl::Status CheckParentContextTable() final;

  absl::Status InsertParentContext(int64 parent_id, int64 child_id) final;

  absl::Status SelectParentContextsByContextID(int64 context_id,
                                               RecordSet* record_set) final;

  absl::Status SelectChildContextsByContextID(int64 context_id,
                                              RecordSet* record_set) final;

  absl::Status CheckMLMDEnvTable() final {
    return ExecuteQuery(query_config_.check_mlmd_env_table());
  }

  // Insert the schema version.
  absl::Status InsertSchemaVersion(int64 schema_version) final {
    return ExecuteQuery(query_config_.insert_schema_version(),
                        {Bind(schema_version)});
  }

  // Update the schema version.
  absl::Status UpdateSchemaVersion(int64 schema_version) final {
    return ExecuteQuery(query_config_.update_schema_version(),
                        {Bind(schema_version)});
  }

  absl::Status CheckTablesIn_V0_13_2() final;

  absl::Status SelectAllArtifactIDs(RecordSet* set) final {
    return ExecuteQuery("select `id` from `Artifact`;", set);
  }

  absl::Status SelectAllExecutionIDs(RecordSet* set) final {
    return ExecuteQuery("select `id` from `Execution`;", set);
  }

  absl::Status SelectAllContextIDs(RecordSet* set) final {
    return ExecuteQuery("select `id` from `Context`;", set);
  }

  int64 GetLibraryVersion() final {
    CHECK_GT(query_config_.schema_version(), 0);
    return query_config_.schema_version();
  }

  absl::Status DowngradeMetadataSource(const int64 to_schema_version) final;

  absl::Status ListArtifactIDsUsingOptions(
      const ListOperationOptions& options,
      absl::optional<absl::Span<const int64>> candidate_ids,
      RecordSet* record_set) final;

  absl::Status ListExecutionIDsUsingOptions(
      const ListOperationOptions& options,
      absl::optional<absl::Span<const int64>> candidate_ids,
      RecordSet* record_set) final;

  absl::Status ListContextIDsUsingOptions(
      const ListOperationOptions& options,
      absl::optional<absl::Span<const int64>> candidate_ids,
      RecordSet* record_set) final;


  absl::Status DeleteArtifactsById(absl::Span<const int64> artifact_ids) final;

  absl::Status DeleteContextsById(absl::Span<const int64> context_ids) final;

  absl::Status DeleteExecutionsById(
      absl::Span<const int64> execution_ids) final;

  absl::Status DeleteEventsByArtifactsId(
      absl::Span<const int64> artifact_ids) final;

  absl::Status DeleteEventsByExecutionsId(
      absl::Span<const int64> execution_ids) final;

  absl::Status DeleteAssociationsByContextsId(
      absl::Span<const int64> context_ids) final;

  absl::Status DeleteAssociationsByExecutionsId(
      absl::Span<const int64> execution_ids) final;

  absl::Status DeleteAttributionsByContextsId(
      absl::Span<const int64> context_ids) final;

  absl::Status DeleteAttributionsByArtifactsId(
      absl::Span<const int64> artifact_ids) final;

  absl::Status DeleteParentContextsByParentIds(
      absl::Span<const int64> parent_context_ids) final;

  absl::Status DeleteParentContextsByChildIds(
      absl::Span<const int64> child_context_ids) final;

  absl::Status DeleteParentContextsByParentIdAndChildIds(
      int64 parent_context_id, absl::Span<const int64> child_context_ids) final;

 private:
  // Utility method to bind an nullable value.
  template <typename T>
  std::string Bind(const absl::optional<T>& v) {
    return v ? Bind(v.value()) : "NULL";
  }

  // Utility method to bind an string_view value to a SQL clause.
  std::string Bind(absl::string_view value);

  // Utility method to bind an string_view value to a SQL clause.
  std::string Bind(const char* value);

  // Utility method to bind an int value to a SQL clause.
  std::string Bind(int value);

  // Utility method to bind an int64 value to a SQL clause.
  std::string Bind(int64 value);

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
  std::string Bind(absl::Span<const int64> value);

  // Utility method to bind a string_view vector to a string joined with ","
  // that can fit into SQL IN(...) clause.
  std::string Bind(absl::Span<absl::string_view> value);

  // Utility method to bind a pair<string_view, string_view> vector to a string
  // joined with "," that can fit into SQL IN(...) clause.
  std::string Bind(
      absl::Span<std::pair<absl::string_view, absl::string_view>> value);

  #if (!defined(__APPLE__) && !defined(_WIN32))
  std::string Bind(const google::protobuf::int64 value);
  #endif

  // Execute a template query. All strings in parameters should already be
  // in a format appropriate for the SQL variant being used (at this point,
  // they are just inserted).
  // Results consist of zero or more rows represented in RecordSet.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  absl::Status ExecuteQuery(
      const MetadataSourceQueryConfig::TemplateQuery& template_query,
      absl::Span<const std::string> parameters, RecordSet* record_set);

  // Execute a template query and ignore the result.
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

  // Execute a template query without arguments and ignore the result.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  absl::Status ExecuteQuery(
      const MetadataSourceQueryConfig::TemplateQuery& query) {
    return ExecuteQuery(query, {});
  }

  // Execute a template query without arguments and ignore the result.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  // Returns INTERNAL error, if it cannot find the last insert ID.
  absl::Status ExecuteQuerySelectLastInsertID(
      const MetadataSourceQueryConfig::TemplateQuery& query,
      absl::Span<const std::string> arguments, int64* last_insert_id) {
    MLMD_RETURN_IF_ERROR(ExecuteQuery(query, arguments));
    return SelectLastInsertID(last_insert_id);
  }

  // Execute a query without arguments.
  // Results consist of zero or more rows represented in RecordSet.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  absl::Status ExecuteQuery(const std::string& query, RecordSet* record_set);

  // Execute a query without arguments and ignore the result.
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
  absl::Status IsCompatible(int64 db_version, int64 lib_version,
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
  // TODO(martinz): consider promoting to MetadataAccessObject.
  absl::Status UpgradeMetadataSourceIfOutOfDate(bool enable_migration);

  // List Node IDs using `options` and `candidate_ids`. Template parameter
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
      absl::optional<absl::Span<const int64>> candidate_ids,
      RecordSet* record_set);

  MetadataSourceQueryConfig query_config_;

  // This object does not own the MetadataSource.
  MetadataSource* metadata_source_;

  // Delegate EncodeBytes to metadata_source_
  // Encodes value and returns the result as a string
  std::string EncodeBytes(absl::string_view value) const {
    return metadata_source_->EncodeBytes(value);
  }

  // Delegate DecodeBytes to metadata_source_
  // Decodes value and writes the result to dest
  // Returns OkStatus if the process succeeded, otherwise an informative error
  absl::Status DecodeBytes(absl::string_view value, std::string& dest) const {
    absl::StatusOr<std::string> decoded_or
      = metadata_source_->DecodeBytes(value);
    if (!decoded_or.ok()) return decoded_or.status();
    dest = decoded_or.value();
    return absl::OkStatus();
  }
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_QUERY_CONFIG_EXECUTOR_H_
