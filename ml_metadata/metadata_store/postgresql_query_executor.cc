
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
#include "ml_metadata/metadata_store/postgresql_query_executor.h"

#include <optional>
#include <string>
#include <vector>

#include "google/protobuf/struct.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/json_util.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "ml_metadata/metadata_store/list_operation_query_helper.h"
#include "ml_metadata/metadata_store/query_config_executor.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/query/filter_query_ast_resolver.h"
#include "ml_metadata/query/filter_query_builder.h"
#include "ml_metadata/util/return_utils.h"
#include "ml_metadata/util/struct_utils.h"

namespace ml_metadata {

PostgreSQLQueryExecutor::PostgreSQLQueryExecutor(
    const MetadataSourceQueryConfig& query_config, MetadataSource* source,
    int64_t query_version)
    : QueryExecutor(query_version),
      query_config_(query_config),
      metadata_source_(source) {}

absl::Status PostgreSQLQueryExecutor::InsertAttributionDirect(
    int64_t context_id, int64_t artifact_id, int64_t* attribution_id) {
  // Early return if duplicated entry exists to avoid transaction failure.
  RecordSet record_set;
  constexpr absl::string_view check_exist_query_str =
      R"pb(
    query: " SELECT count(*) FROM Attribution "
           "  WHERE context_id  = $0 "
           "    AND artifact_id = $1; "
    parameter_num: 2
      )pb";
  MetadataSourceQueryConfig::TemplateQuery check_exist_query;
  MLMD_RETURN_IF_ERROR(
      GetTemplateQueryOrDie(check_exist_query_str.data(), check_exist_query));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(
      check_exist_query, {Bind(context_id), Bind(artifact_id)}, &record_set));
  int64_t stored_attribution_cnt;
  if (record_set.records_size() != 1 ||
      record_set.records(0).values_size() != 1 ||
      !absl::SimpleAtoi(record_set.records(0).values(0),
                        &stored_attribution_cnt)) {
    return absl::DataLossError(
        absl::StrCat("Expect Attribution primary key check query to return "
                     "single count with one int value, result is different.",
                     record_set.DebugString()));
  }
  if (stored_attribution_cnt > 0) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Duplicate Association exists with input:", " context_id: ", context_id,
        " artifact_id: ", artifact_id));
  }
  return ExecuteQuerySelectLastInsertID(query_config_.insert_attribution(),
                                        {Bind(context_id), Bind(artifact_id)},
                                        attribution_id);
}

absl::Status PostgreSQLQueryExecutor::InsertAssociation(
    int64_t context_id, int64_t execution_id, int64_t* association_id) {
  // Early return if duplicated entry exists to avoid transaction failure.
  RecordSet record_set;
  constexpr absl::string_view check_exist_query_str =
      R"pb(
    query: " SELECT count(*) FROM Association "
           "  WHERE context_id  = $0 "
           "    AND execution_id = $1; "
    parameter_num: 2
      )pb";
  MetadataSourceQueryConfig::TemplateQuery check_exist_query;
  MLMD_RETURN_IF_ERROR(
      GetTemplateQueryOrDie(check_exist_query_str.data(), check_exist_query));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(
      check_exist_query, {Bind(context_id), Bind(execution_id)}, &record_set));
  int64_t stored_association_cnt;
  if (record_set.records_size() != 1 ||
      record_set.records(0).values_size() != 1 ||
      !absl::SimpleAtoi(record_set.records(0).values(0),
                        &stored_association_cnt)) {
    return absl::DataLossError(
        absl::StrCat("Expect Association primary key check query to return "
                     "single count with one int value, result is different.",
                     record_set.DebugString()));
  }
  if (stored_association_cnt > 0) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Duplicate Association exists with input:", " context_id: ", context_id,
        " execution_id: ", execution_id));
  }
  return ExecuteQuerySelectLastInsertID(query_config_.insert_association(),
                                        {Bind(context_id), Bind(execution_id)},
                                        association_id);
}

absl::Status PostgreSQLQueryExecutor::InsertParentType(int64_t type_id,
                                                       int64_t parent_type_id) {
  // Early return if duplicated entry exists to avoid transaction failure.
  RecordSet record_set;
  constexpr absl::string_view check_exist_query_str =
      R"pb(
    query: " SELECT count(*) FROM ParentType "
           "  WHERE type_id = $0 "
           "    AND parent_type_id = $1;"
    parameter_num: 2
      )pb";
  MetadataSourceQueryConfig::TemplateQuery check_exist_query;
  MLMD_RETURN_IF_ERROR(
      GetTemplateQueryOrDie(check_exist_query_str.data(), check_exist_query));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(
      check_exist_query, {Bind(type_id), Bind(parent_type_id)}, &record_set));
  int64_t stored_parent_cnt;
  if (record_set.records_size() != 1 ||
      record_set.records(0).values_size() != 1 ||
      !absl::SimpleAtoi(record_set.records(0).values(0), &stored_parent_cnt)) {
    return absl::DataLossError(
        absl::StrCat("Expect ParentType primary key check query to return "
                     "single count with one int value, result is different.",
                     record_set.DebugString()));
  }
  if (stored_parent_cnt > 0) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Duplicate parent type exists with input:", " parent_type_id: ",
        parent_type_id, " type_id: ", type_id));
  }
  return ExecuteQuery(query_config_.insert_parent_type(),
                      {Bind(type_id), Bind(parent_type_id)});
}

absl::Status PostgreSQLQueryExecutor::DeleteParentType(int64_t type_id,
                                                       int64_t parent_type_id) {
  return ExecuteQuery(query_config_.delete_parent_type(),
                      {Bind(type_id), Bind(parent_type_id)});
}
absl::Status PostgreSQLQueryExecutor::SelectParentTypesByTypeID(
    absl::Span<const int64_t> type_ids, RecordSet* record_set) {
  return ExecuteQuery(query_config_.select_parent_type_by_type_id(),
                      {Bind(type_ids)}, record_set);
}
absl::Status PostgreSQLQueryExecutor::InsertEventPath(
    int64_t event_id, const Event::Path::Step& step) {
  // Inserts a path into the EventPath table. It has 4 parameters
  // $0 is the event_id
  // $1 is the step value case, either index or key
  // $2 is the is_index_step indicates the step value case
  // $3 is the value of the step
  if (step.has_index()) {
    return ExecuteQuery(
        query_config_.insert_event_path(),
        {Bind(event_id), "step_index", Bind(true), Bind(step.index())});
  } else if (step.has_key()) {
    return ExecuteQuery(
        query_config_.insert_event_path(),
        {Bind(event_id), "step_key", Bind(false), Bind(step.key())});
  }
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::InsertParentContext(int64_t parent_id,
                                                          int64_t child_id) {
  return ExecuteQuery(query_config_.insert_parent_context(),
                      {Bind(child_id), Bind(parent_id)});
}

absl::Status PostgreSQLQueryExecutor::SelectParentContextsByContextIDs(
    absl::Span<const int64_t> context_ids, RecordSet* record_set) {
  return ExecuteQuery(query_config_.select_parent_contexts_by_context_ids(),
                      {Bind(context_ids)}, record_set);
}

absl::Status PostgreSQLQueryExecutor::SelectChildContextsByContextIDs(
    absl::Span<const int64_t> context_ids, RecordSet* record_set) {
  return ExecuteQuery(
      query_config_.select_parent_contexts_by_parent_context_ids(),
      {Bind(context_ids)}, record_set);
}

absl::Status PostgreSQLQueryExecutor::SelectParentContextsByContextID(
    int64_t context_id, RecordSet* record_set) {
  return SelectParentContextsByContextIDs({context_id}, record_set);
}
absl::Status PostgreSQLQueryExecutor::SelectChildContextsByContextID(
    int64_t context_id, RecordSet* record_set) {
  return SelectChildContextsByContextIDs({context_id}, record_set);
}

absl::Status PostgreSQLQueryExecutor::GetSchemaVersion(int64_t* db_version) {
  absl::Status check_mlmd_env_table_status = CheckMLMDEnvTable();
  if (check_mlmd_env_table_status.ok()) {
    RecordSet mlmd_schema_record_set;
    absl::Status schema_check_status = ExecuteQuery(
        query_config_.check_mlmd_env_table(), {}, &mlmd_schema_record_set);
    MLMD_RETURN_IF_ERROR(schema_check_status);
    if (mlmd_schema_record_set.records_size() == 0) {
      return absl::DataLossError(absl::StrCat(
          "In the given db, there is no MLMDEnv version exist, this is "
          "unexpected."));
    }
    if (mlmd_schema_record_set.records_size() > 1) {
      return absl::DataLossError(absl::StrCat(
          "In the given db, there are multiple MLMDEnv versions exist, this is "
          "unexpected. Result detail: ",
          mlmd_schema_record_set.DebugString()));
    }
    CHECK(absl::SimpleAtoi(mlmd_schema_record_set.records(0).values(0),
                           db_version));
    return absl::OkStatus();
  }
  RecordSet record_set;
  // if MLMDEnv does not exist, it may be the v0.13.2 release or an empty db.
  absl::Status maybe_v0_13_2_status =
      ExecuteQuery(query_config_.check_tables_in_v0_13_2(), {}, &record_set);
  if (maybe_v0_13_2_status.ok() && record_set.records_size() == 1) {
    const RecordSet::Record& record = record_set.records(0);
    int64_t v0_table_count;
    if (!absl::SimpleAtoi(record.values(0), &v0_table_count)) {
      return absl::InternalError("Could not parse v0 table count to integer");
    }
    // v0 schema has 8 tables.
    if (v0_table_count == 8) {
      *db_version = 0;
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError("it looks an empty db is given.");
}
absl::Status PostgreSQLQueryExecutor::UpgradeMetadataSourceIfOutOfDate(
    bool enable_migration) {
  int64_t db_version = 0;
  absl::Status get_schema_version_status = GetSchemaVersion(&db_version);
  int64_t lib_version = GetLibraryVersion();
  if (absl::IsNotFound(get_schema_version_status)) {
    db_version = lib_version;
  } else {
    MLMD_RETURN_IF_ERROR(get_schema_version_status);
  }
  bool is_compatible = false;
  MLMD_RETURN_IF_ERROR(IsCompatible(db_version, lib_version, &is_compatible));
  if (is_compatible) {
    return absl::OkStatus();
  }
  if (db_version > lib_version) {
    return absl::FailedPreconditionError(absl::StrCat(
        "MLMD database version ", db_version,
        " is greater than library version ", lib_version,
        ". Please upgrade the library to use the given database in order to "
        "prevent potential data loss. If data loss is acceptable, please"
        " downgrade the database using a newer version of library."));
  }
  // returns error if upgrade is explicitly disabled, as we are missing schema
  // and cannot continue with this library version.
  if (db_version < lib_version && !enable_migration) {
    return absl::FailedPreconditionError(absl::StrCat(
        "MLMD database version ", db_version, " is older than library version ",
        lib_version,
        ". Schema migration is disabled. Please upgrade the database then use"
        " the library version; or switch to a older library version to use the"
        " current database. For more details, please refer to ml-metadata"
        " g3doc/third_party/ml_metadata/g3doc/"
        "get_started.md#upgrade-the-database-schema"));
  }
  // migrate db_version to lib version
  const auto& migration_schemes = query_config_.migration_schemes();
  while (db_version < lib_version) {
    const int64_t to_version = db_version + 1;
    if (migration_schemes.find(to_version) == migration_schemes.end()) {
      return absl::InternalError(absl::StrCat(
          "Cannot find migration_schemes to version ", to_version));
    }
    for (const MetadataSourceQueryConfig::TemplateQuery& upgrade_query :
         migration_schemes.at(to_version).upgrade_queries()) {
      MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
          ExecuteQuery(upgrade_query.query()),
          absl::StrCat("Upgrade query failed: ", upgrade_query.query()));
    }
    MLMD_RETURN_WITH_CONTEXT_IF_ERROR(UpdateSchemaVersion(to_version),
                                      "Failed to update schema.");
    db_version = to_version;
  }
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::SelectLastInsertID(
    int64_t* last_insert_id) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.select_last_insert_id(), {}, &record_set));
  if (record_set.records_size() == 0) {
    return absl::InternalError("Could not find last insert ID: no record");
  }
  const RecordSet::Record& record = record_set.records(0);
  if (record.values_size() == 0) {
    return absl::InternalError("Could not find last insert ID: missing value");
  }
  if (!absl::SimpleAtoi(record.values(0), last_insert_id)) {
    return absl::InternalError("Could not parse last insert ID as string");
  }
  return absl::OkStatus();
}
absl::Status ml_metadata::PostgreSQLQueryExecutor::CheckTablesIn_V0_13_2() {
  return ExecuteQuery(query_config_.check_tables_in_v0_13_2());
}
absl::Status PostgreSQLQueryExecutor::DowngradeMetadataSource(
    const int64_t to_schema_version) {
  const int64_t lib_version = query_config_.schema_version();
  if (to_schema_version < 0 || to_schema_version > lib_version) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MLMD cannot be downgraded to schema_version: ", to_schema_version,
        ". The target version should be greater or equal to 0, and the current"
        " library version: ",
        lib_version, " needs to be greater than the target version."));
  }
  int64_t db_version = 0;
  absl::Status get_schema_version_status = GetSchemaVersion(&db_version);
  // if it is an empty database, then we skip downgrade and returns.
  if (absl::IsNotFound(get_schema_version_status)) {
    return absl::InvalidArgumentError(
        "Empty database is given. Downgrade operation is not needed.");
  }
  MLMD_RETURN_IF_ERROR(get_schema_version_status);
  if (db_version > lib_version) {
    return absl::FailedPreconditionError(
        absl::StrCat("MLMD database version ", db_version,
                     " is greater than library version ", lib_version,
                     ". The current library does not know how to downgrade it. "
                     "Please upgrade the library to downgrade the schema."));
  }
  // perform downgrade
  const auto& migration_schemes = query_config_.migration_schemes();
  while (db_version > to_schema_version) {
    const int64_t to_version = db_version - 1;
    if (migration_schemes.find(to_version) == migration_schemes.end()) {
      return absl::InternalError(absl::StrCat(
          "Cannot find migration_schemes to version ", to_version));
    }
    for (const MetadataSourceQueryConfig::TemplateQuery& downgrade_query :
         migration_schemes.at(to_version).downgrade_queries()) {
      MLMD_RETURN_WITH_CONTEXT_IF_ERROR(ExecuteQuery(downgrade_query),
                                        "Failed to migrate existing db; the "
                                        "migration transaction rolls back.");
    }
    // at version 0, v0.13.2, there is no schema version information.
    if (to_version > 0) {
      MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
          UpdateSchemaVersion(to_version),
          "Failed to migrate existing db; the migration transaction rolls "
          "back.");
    }
    db_version = to_version;
  }
  return absl::OkStatus();
}
std::string PostgreSQLQueryExecutor::Bind(const char* value) {
  return absl::StrCat("'", metadata_source_->EscapeString(value), "'");
}
std::string PostgreSQLQueryExecutor::Bind(absl::string_view value) {
  return absl::StrCat("'", metadata_source_->EscapeString(value), "'");
}
std::string PostgreSQLQueryExecutor::Bind(int value) {
  return std::to_string(value);
}
std::string PostgreSQLQueryExecutor::Bind(int64_t value) {
  return std::to_string(value);
}
std::string PostgreSQLQueryExecutor::Bind(double value) {
  return std::to_string(value);
}
std::string PostgreSQLQueryExecutor::Bind(const google::protobuf::Any& value) {
  return absl::StrCat(
      "decode('",
      metadata_source_->EscapeString(
          metadata_source_->EncodeBytes(value.SerializeAsString())),
      "', 'base64')");
}
std::string PostgreSQLQueryExecutor::Bind(bool value) {
  return value ? "TRUE" : "FALSE";
}
// Utility method to bind an Event::Type enum value to a SQL clause.
// Event::Type is an enum (integer), EscapeString is not applicable.
std::string PostgreSQLQueryExecutor::Bind(const Event::Type value) {
  return std::to_string(value);
}
std::string PostgreSQLQueryExecutor::Bind(PropertyType value) {
  return std::to_string((int)value);
}
std::string PostgreSQLQueryExecutor::Bind(TypeKind value) {
  return std::to_string((int)value);
}
std::string PostgreSQLQueryExecutor::Bind(Artifact::State value) {
  return std::to_string((int)value);
}
std::string PostgreSQLQueryExecutor::Bind(Execution::State value) {
  return std::to_string((int)value);
}
std::string PostgreSQLQueryExecutor::Bind(absl::Span<const int64_t> value) {
  return absl::StrJoin(value, ", ");
}
std::string PostgreSQLQueryExecutor::Bind(absl::Span<absl::string_view> value) {
  std::vector<std::string> escape_string_value;
  escape_string_value.reserve(value.size());
  for (const auto& v : value) {
    escape_string_value.push_back(Bind(v));
  }
  return absl::StrJoin(escape_string_value, ", ");
}
std::string PostgreSQLQueryExecutor::Bind(
    absl::Span<std::pair<absl::string_view, absl::string_view>> value) {
  std::vector<std::string> escape_string_value;
  escape_string_value.reserve(value.size());
  for (const auto& v : value) {
    escape_string_value.push_back(
        absl::StrCat("(", Bind(v.first), ",", Bind(v.second), ")"));
  }
  return absl::StrJoin(escape_string_value, ", ");
}
std::string PostgreSQLQueryExecutor::BindValue(const Value& value) {
  switch (value.value_case()) {
    case PropertyType::INT:
      return Bind(value.int_value());
    case PropertyType::DOUBLE:
      return Bind(value.double_value());
    case PropertyType::STRING:
      return Bind(value.string_value());
    case PropertyType::STRUCT:
      return Bind(StructToString(value.struct_value()));
    case PropertyType::PROTO:
      return Bind(value.proto_value());
    case PropertyType::BOOLEAN:
      return Bind(value.bool_value());
    default:
      LOG(FATAL) << "Unknown registered property type: " << value.value_case()
                 << "This is an internal error: properties should have been "
                    "checked before"
                    " they got here";
  }
}
std::string PostgreSQLQueryExecutor::BindDataType(const Value& value) {
  switch (value.value_case()) {
    case PropertyType::INT: {
      return "int_value";
      break;
    }
    case PropertyType::DOUBLE: {
      return "double_value";
      break;
    }
    case PropertyType::STRING:
    case PropertyType::STRUCT: {
      return "string_value";
      break;
    }
    case PropertyType::PROTO: {
      return "proto_value";
      break;
    }
    case PropertyType::BOOLEAN: {
      return "bool_value";
      break;
    }
    default: {
      LOG(FATAL) << "Unexpected oneof: " << value.DebugString();
    }
  }
}
std::string PostgreSQLQueryExecutor::Bind(const ArtifactStructType* message) {
  if (message) {
    std::string json_output;
    CHECK(::google::protobuf::util::MessageToJsonString(*message, &json_output).ok())
        << "Could not write proto to JSON: " << message->DebugString();
    return Bind(json_output);
  } else {
    return "null";
  }
}

absl::Status PostgreSQLQueryExecutor::ExecuteQuery(const std::string& query) {
  RecordSet record_set;
  return metadata_source_->ExecuteQuery(query, &record_set);
}
absl::Status PostgreSQLQueryExecutor::ExecuteQuery(const std::string& query,
                                                   RecordSet* record_set) {
  return metadata_source_->ExecuteQuery(query, record_set);
}
absl::Status PostgreSQLQueryExecutor::ExecuteQuery(
    const MetadataSourceQueryConfig::TemplateQuery& template_query,
    absl::Span<const std::string> parameters, RecordSet* record_set) {
  if (parameters.size() > 10) {
    return absl::InvalidArgumentError(
        "Template query has too many parameters (at most 10 is supported).");
  }
  if (template_query.parameter_num() != parameters.size()) {
    LOG(FATAL) << "Template query parameter_num ("
               << template_query.parameter_num()
               << ") does not match with given "
               << "parameters size (" << parameters.size()
               << "): " << template_query.DebugString();
  }
  std::vector<std::pair<const std::string, const std::string>> replacements;
  replacements.reserve(parameters.size());
  for (int i = 0; i < parameters.size(); i++) {
    replacements.push_back({absl::StrCat("$", i), parameters[i]});
  }
  return metadata_source_->ExecuteQuery(
      absl::StrReplaceAll(template_query.query(), replacements), record_set);
}
absl::Status PostgreSQLQueryExecutor::IsCompatible(int64_t db_version,
                                                   int64_t lib_version,
                                                   bool* is_compatible) {
  // Currently, we don't support a database version that is older than the
  // library version. Revisit this if a more sophisticated rule is required.
  *is_compatible = (db_version == lib_version);
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::InitMetadataSource() {
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_type_table()));
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.create_type_property_table()));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_parent_type_table()));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_artifact_table()));
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.create_artifact_property_table()));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_execution_table()));
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.create_execution_property_table()));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_event_table()));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_event_path_table()));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_mlmd_env_table()));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_context_table()));
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.create_context_property_table()));
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.create_parent_context_table()));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_association_table()));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_attribution_table()));
  for (const MetadataSourceQueryConfig::TemplateQuery& index_query :
       query_config_.secondary_indices()) {
    const absl::Status status = ExecuteQuery(index_query);
    // For databases (e.g., MySQL), idempotency of indexing creation is not
    // supported well. We handle it here and covered by the `InitForReset` test.
    if (!status.ok() && absl::StrContains(std::string(status.message()),
                                          "Duplicate key name")) {
      continue;
    }
    MLMD_RETURN_IF_ERROR(status);
  }
  int64_t library_version = GetLibraryVersion();
  absl::Status insert_schema_version_status =
      InsertSchemaVersion(library_version);
  if (!insert_schema_version_status.ok()) {
    int64_t db_version = -1;
    MLMD_RETURN_IF_ERROR(GetSchemaVersion(&db_version));
    if (db_version != library_version) {
      return absl::DataLossError(absl::StrCat(
          "The database cannot be initialized with the schema_version in the "
          "current library. Current library version: ",
          library_version, ", the db version on record is: ", db_version,
          ". It may result from a data race condition caused by other "
          "concurrent MLMD's migration procedures."));
    }
  }
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::CheckTableResult(
    const MetadataSourceQueryConfig::TemplateQuery query) {
  RecordSet record_set;
  const absl::Status status = ExecuteQuery(query, {}, &record_set);
  if (!status.ok()) {
    return status;
  }
  // For any DB type that fails the whole transaction when one query has failed,
  // MLMD expects the check table query to return a one row with column name
  // `table_exists`. In that case, the check table query will not fail if the
  // table doesn't exist, but return a boolean value to indicate whether the
  // table exists.
  if (record_set.records_size() == 1 && record_set.column_names_size() == 1 &&
      record_set.column_names(0) == "table_exists") {
    // Default: If value is 1, that means the schema exists, otherwise it is 0.
    // Postgresql: If the expected schema exists, result is `t`, otherwise `f`.
    // Therefore Postgresql result needs to be converted to 1 or 0.
    bool table_exists_value = record_set.records(0).values(0) == "1";
    if (table_exists_value) {
      return absl::OkStatus();
    } else {
      return absl::NotFoundError("Desired table and columns don't exist.");
    }
  }
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::CheckTypeTable() {
  return CheckTableResult(query_config_.check_type_table());
}
absl::Status PostgreSQLQueryExecutor::CheckParentTypeTable() {
  return CheckTableResult(query_config_.check_parent_type_table());
}
absl::Status PostgreSQLQueryExecutor::CheckTypePropertyTable() {
  return CheckTableResult(query_config_.check_type_property_table());
}
absl::Status PostgreSQLQueryExecutor::CheckArtifactTable() {
  return CheckTableResult(query_config_.check_artifact_table());
}
absl::Status PostgreSQLQueryExecutor::CheckArtifactPropertyTable() {
  // TODO(b/257334039): Cleanup the fat-client after fully migrated to V10+.
  MetadataSourceQueryConfig::TemplateQuery check_artifact_property_table;
  check_artifact_property_table = query_config_.check_artifact_property_table();
  return CheckTableResult(check_artifact_property_table);
}
absl::Status PostgreSQLQueryExecutor::CheckExecutionTable() {
  return CheckTableResult(query_config_.check_execution_table());
}
absl::Status PostgreSQLQueryExecutor::CheckExecutionPropertyTable() {
  // TODO(b/257334039): Cleanup the fat-client after fully migrated to V10+.
  MetadataSourceQueryConfig::TemplateQuery check_execution_property_table;
  check_execution_property_table =
      query_config_.check_execution_property_table();
  return CheckTableResult(check_execution_property_table);
}
absl::Status PostgreSQLQueryExecutor::CheckEventTable() {
  return CheckTableResult(query_config_.check_event_table());
}
absl::Status PostgreSQLQueryExecutor::CheckEventPathTable() {
  return CheckTableResult(query_config_.check_event_path_table());
}
absl::Status PostgreSQLQueryExecutor::CheckMLMDEnvTable() {
  return CheckTableResult(query_config_.check_mlmd_env_table_existence());
}
absl::Status PostgreSQLQueryExecutor::CheckContextTable() {
  return CheckTableResult(query_config_.check_context_table());
}
absl::Status PostgreSQLQueryExecutor::CheckParentContextTable() {
  return CheckTableResult(query_config_.check_parent_context_table());
}
absl::Status PostgreSQLQueryExecutor::CheckContextPropertyTable() {
  // TODO(b/257334039): Cleanup the fat-client after fully migrated to V10+.
  MetadataSourceQueryConfig::TemplateQuery check_context_property_table;
  check_context_property_table = query_config_.check_context_property_table();
  return CheckTableResult(check_context_property_table);
}
absl::Status PostgreSQLQueryExecutor::CheckAssociationTable() {
  return CheckTableResult(query_config_.check_association_table());
}
absl::Status PostgreSQLQueryExecutor::CheckAttributionTable() {
  return CheckTableResult(query_config_.check_attribution_table());
}
absl::Status PostgreSQLQueryExecutor::InitMetadataSourceIfNotExists(
    const bool enable_upgrade_migration) {
  // If |query_schema_version_| is given, then the query executor is expected to
  // work with an existing db with an earlier schema version equals to that.
  if (query_schema_version()) {
    return CheckSchemaVersionAlignsWithQueryVersion();
  }
  // When working at head, we reuse existing db or create a new db.
  // check db version, and make it to align with the lib version.
  MLMD_RETURN_IF_ERROR(
      UpgradeMetadataSourceIfOutOfDate(enable_upgrade_migration));
  // if lib and db versions align, we check the required tables for the lib.
  std::vector<std::pair<absl::Status, std::string>> checks;
  checks.push_back({CheckTypeTable(), "type_table"});
  checks.push_back({CheckParentTypeTable(), "parent_type_table"});
  checks.push_back({CheckTypePropertyTable(), "type_property_table"});
  checks.push_back({CheckArtifactTable(), "artifact_table"});
  checks.push_back({CheckArtifactPropertyTable(), "artifact_property_table"});
  checks.push_back({CheckExecutionTable(), "execution_table"});
  checks.push_back({CheckExecutionPropertyTable(), "execution_property_table"});
  checks.push_back({CheckEventTable(), "event_table"});
  checks.push_back({CheckEventPathTable(), "event_path_table"});
  checks.push_back({CheckMLMDEnvTable(), "mlmd_env_table"});
  checks.push_back({CheckContextTable(), "context_table"});
  checks.push_back({CheckParentContextTable(), "parent_context_table"});
  checks.push_back({CheckContextPropertyTable(), "context_property_table"});
  checks.push_back({CheckAssociationTable(), "association_table"});
  checks.push_back({CheckAttributionTable(), "attribution_table"});
  std::vector<std::string> missing_schema_error_messages;
  std::vector<std::string> successful_checks;
  std::vector<std::string> failing_checks;
  for (const auto& check_pair : checks) {
    const absl::Status& check = check_pair.first;
    const std::string& name = check_pair.second;
    if (!check.ok()) {
      missing_schema_error_messages.push_back(check.ToString());
      failing_checks.push_back(name);
    } else {
      successful_checks.push_back(name);
    }
  }
  // all table required by the current lib version exists
  if (missing_schema_error_messages.empty()) return absl::OkStatus();
  // some table exists, but not all.
  if (checks.size() != missing_schema_error_messages.size()) {
    return absl::AbortedError(absl::StrCat(
        "There are a subset of tables in MLMD instance. This may be due to "
        "concurrent connection to the empty database. "
        "Please retry the connection. checks: ",
        checks.size(), " errors: ", missing_schema_error_messages.size(),
        ", present tables: ", absl::StrJoin(successful_checks, ", "),
        ", missing tables: ", absl::StrJoin(failing_checks, ", "),
        " Errors: ", absl::StrJoin(missing_schema_error_messages, "\n")));
  }
  // no table exists, then init the MetadataSource
  return InitMetadataSource();
}
absl::Status PostgreSQLQueryExecutor::InsertArtifactType(
    const std::string& name, std::optional<absl::string_view> version,
    std::optional<absl::string_view> description,
    std::optional<absl::string_view> external_id, int64_t* type_id) {
  if (external_id.has_value()) {
    RecordSet record;
    MLMD_RETURN_IF_ERROR(
        ExecuteQuery(query_config_.select_types_by_external_ids(),
                     {Bind({external_id.value()}), Bind(1)}, &record));
    if (record.records_size() > 0) {
      return absl::AlreadyExistsError(absl::StrCat(
          "Conflict of external_id: ", external_id.value(),
          " Found already existing Artifact type with the same external_id: ",
          record.DebugString()));
    }
  }
  return ExecuteQuerySelectLastInsertID(
      query_config_.insert_artifact_type(),
      {Bind(name), Bind(version), Bind(description), Bind(external_id)},
      type_id);
}
absl::Status PostgreSQLQueryExecutor::InsertExecutionType(
    const std::string& name, std::optional<absl::string_view> version,
    std::optional<absl::string_view> description,
    const ArtifactStructType* input_type, const ArtifactStructType* output_type,
    std::optional<absl::string_view> external_id, int64_t* type_id) {
  if (external_id.has_value()) {
    RecordSet record;
    MLMD_RETURN_IF_ERROR(
        ExecuteQuery(query_config_.select_types_by_external_ids(),
                     {Bind({external_id.value()}), Bind(0)}, &record));
    if (record.records_size() > 0) {
      return absl::AlreadyExistsError(absl::StrCat(
          "Conflict of external_id: ", external_id.value(),
          " Found already existing Execution type with the same external_id: ",
          record.DebugString()));
    }
  }
  return ExecuteQuerySelectLastInsertID(
      query_config_.insert_execution_type(),
      {Bind(name), Bind(version), Bind(description), Bind(input_type),
       Bind(output_type), Bind(external_id)},
      type_id);
}
absl::Status PostgreSQLQueryExecutor::InsertContextType(
    const std::string& name, std::optional<absl::string_view> version,
    std::optional<absl::string_view> description,
    std::optional<absl::string_view> external_id, int64_t* type_id) {
  if (external_id.has_value()) {
    RecordSet record;
    MLMD_RETURN_IF_ERROR(
        ExecuteQuery(query_config_.select_types_by_external_ids(),
                     {Bind({external_id.value()}), Bind(2)}, &record));
    if (record.records_size() > 0) {
      return absl::AlreadyExistsError(absl::StrCat(
          "Conflict of external_id: ", external_id.value(),
          " Found already existing Context type with the same external_id: ",
          record.DebugString()));
    }
  }
  return ExecuteQuerySelectLastInsertID(
      query_config_.insert_context_type(),
      {Bind(name), Bind(version), Bind(description), Bind(external_id)},
      type_id);
}
absl::Status PostgreSQLQueryExecutor::SelectTypesByID(
    absl::Span<const int64_t> type_ids, TypeKind type_kind,
    RecordSet* record_set) {
  return ExecuteQuery(query_config_.select_types_by_id(),
                      {Bind(type_ids), Bind(type_kind)}, record_set);
}
absl::Status PostgreSQLQueryExecutor::SelectTypeByID(int64_t type_id,
                                                     TypeKind type_kind,
                                                     RecordSet* record_set) {
  return ExecuteQuery(query_config_.select_type_by_id(),
                      {Bind(type_id), Bind(type_kind)}, record_set);
}
absl::Status PostgreSQLQueryExecutor::SelectTypesByExternalIds(
    const absl::Span<absl::string_view> external_ids, TypeKind type_kind,
    RecordSet* record_set) {
  MLMD_RETURN_IF_ERROR(VerifyCurrentQueryVersionIsAtLeast(kSchemaVersionNine));
  return ExecuteQuery(query_config_.select_types_by_external_ids(),
                      {Bind(external_ids), Bind(type_kind)}, record_set);
}
absl::Status PostgreSQLQueryExecutor::SelectTypeByNameAndVersion(
    absl::string_view type_name, std::optional<absl::string_view> type_version,
    TypeKind type_kind, RecordSet* record_set) {
  if (type_version && !type_version->empty()) {
    return ExecuteQuery(query_config_.select_type_by_name_and_version(),
                        {Bind(type_name), Bind(*type_version), Bind(type_kind)},
                        record_set);
  } else {
    return ExecuteQuery(query_config_.select_type_by_name(),
                        {Bind(type_name), Bind(type_kind)}, record_set);
  }
}
absl::Status PostgreSQLQueryExecutor::SelectTypesByNamesAndVersions(
    absl::Span<std::pair<std::string, std::string>> names_and_versions,
    TypeKind type_kind, RecordSet* record_set) {
  auto partition = std::partition(
      names_and_versions.begin(), names_and_versions.end(),
      [](std::pair<absl::string_view, absl::string_view> name_and_version) {
        return !name_and_version.second.empty();
      });
  // find types with both name and version
  if (names_and_versions.begin() != partition) {
    std::vector<std::pair<absl::string_view, absl::string_view>>
        names_with_versions = {names_and_versions.begin(), partition};
    MLMD_RETURN_IF_ERROR(ExecuteQuery(
        query_config_.select_types_by_names_and_versions(),
        {Bind(absl::MakeSpan(names_with_versions)), Bind(type_kind)},
        record_set));
  }
  // find types with names only
  if (partition != names_and_versions.end()) {
    std::vector<absl::string_view> names;
    std::transform(
        partition, names_and_versions.end(), std::back_inserter(names),
        [](const auto& pair) { return absl::string_view(pair.first); });
    RecordSet record_set_for_types_with_names_only;
    MLMD_RETURN_IF_ERROR(
        ExecuteQuery(query_config_.select_types_by_names(),
                     {Bind(absl::MakeSpan(names)), Bind(type_kind)},
                     &record_set_for_types_with_names_only));
    if (record_set->records_size() > 0) {
      record_set->mutable_records()->MergeFrom(
          record_set_for_types_with_names_only.records());
    } else {
      *record_set = std::move(record_set_for_types_with_names_only);
    }
  }
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::SelectAllTypes(TypeKind type_kind,
                                                     RecordSet* record_set) {
  return ExecuteQuery(query_config_.select_all_types(), {Bind(type_kind)},
                      record_set);
}
template <typename Node>
absl::Status PostgreSQLQueryExecutor::ListNodeIDsUsingOptions(
    const ListOperationOptions& options,
    std::optional<absl::Span<const int64_t>> candidate_ids,
    RecordSet* record_set) {
  // Skip query if candidate_ids are set with an empty collection.
  if (candidate_ids && candidate_ids->empty()) {
    return absl::OkStatus();
  }
  std::string sql_query;
  int64_t query_version = query_schema_version().has_value()
                              ? query_schema_version().value()
                              : kSchemaVersionTen;
  std::optional<absl::string_view> node_table_alias;
  if (std::is_same<Node, Artifact>::value) {
    sql_query = "SELECT id FROM Artifact WHERE";
  } else if (std::is_same<Node, Execution>::value) {
    sql_query = "SELECT id FROM Execution WHERE";
  } else if (std::is_same<Node, Context>::value) {
    sql_query = "SELECT id FROM Context WHERE";
  } else {
    return absl::InvalidArgumentError(
        "Invalid Node passed to ListNodeIDsUsingOptions");
  }

  if (options.has_filter_query() && !options.filter_query().empty()) {
    // DEPRECATED: ZetaSQL-based filter_query is deprecated and will be removed
    // in version 1.18.0. This feature depends on ZetaSQL which is being phased
    // out. Please migrate to alternative filtering approaches.
    LOG(WARNING) << "DEPRECATION WARNING: ZetaSQL-based filter_query is "
                 << "deprecated and will be removed in version 1.18.0. "
                 << "This feature depends on ZetaSQL which is being removed.";
    node_table_alias = ml_metadata::FilterQueryBuilder<Node>::kBaseTableAlias;
    ml_metadata::FilterQueryAstResolver<Node> ast_resolver(
        options.filter_query());
    const absl::Status ast_gen_status = ast_resolver.Resolve();
    if (!ast_gen_status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid `filter_query`: ", ast_gen_status.message()));
    }
    // Generate SQL
    ml_metadata::FilterQueryBuilder<Node> query_builder;
    const absl::Status sql_gen_status =
        ast_resolver.GetAst()->Accept(&query_builder);
    if (!sql_gen_status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to construct valid SQL from `filter_query`: ",
                       sql_gen_status.message()));
    }
    sql_query = absl::Substitute(
        "SELECT distinct $0.id, $0.create_time_since_epoch FROM $1 WHERE $2 "
        "AND ",
        *node_table_alias,
        // TODO(b/257334039): remove query_version-conditional logic
        query_builder.GetFromClause(query_version),
        query_builder.GetWhereClause());
  }

  if (candidate_ids) {
    if (node_table_alias.has_value() && !node_table_alias->empty()) {
      absl::SubstituteAndAppend(&sql_query, " $0.id", *node_table_alias);
      absl::SubstituteAndAppend(&sql_query, " IN ($0) AND ",
                                Bind(*candidate_ids));
    } else {
      absl::SubstituteAndAppend(&sql_query, " id IN ($0) AND ",
                                Bind(*candidate_ids));
    }
  }
  MLMD_RETURN_IF_ERROR(
      AppendOrderingThresholdClause(options, node_table_alias, sql_query));
  MLMD_RETURN_IF_ERROR(
      AppendOrderByClause(options, node_table_alias, sql_query));
  MLMD_RETURN_IF_ERROR(AppendLimitClause(options, sql_query));
  std::string sql_escaped =
      absl::StrReplaceAll(sql_query, {{"\"", "'"}, {"`", ""}});
  return ExecuteQuery(sql_escaped, record_set);
}
absl::Status PostgreSQLQueryExecutor::ListArtifactIDsUsingOptions(
    const ListOperationOptions& options,
    std::optional<absl::Span<const int64_t>> candidate_ids,
    RecordSet* record_set) {
  return ListNodeIDsUsingOptions<Artifact>(options, candidate_ids, record_set);
}
absl::Status PostgreSQLQueryExecutor::ListExecutionIDsUsingOptions(
    const ListOperationOptions& options,
    std::optional<absl::Span<const int64_t>> candidate_ids,
    RecordSet* record_set) {
  return ListNodeIDsUsingOptions<Execution>(options, candidate_ids, record_set);
}
absl::Status PostgreSQLQueryExecutor::ListContextIDsUsingOptions(
    const ListOperationOptions& options,
    std::optional<absl::Span<const int64_t>> candidate_ids,
    RecordSet* record_set) {
  return ListNodeIDsUsingOptions<Context>(options, candidate_ids, record_set);
}
absl::Status PostgreSQLQueryExecutor::DeleteExecutionsById(
    absl::Span<const int64_t> execution_ids) {
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.delete_executions_by_id(),
                                    {Bind(execution_ids)}));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(
      query_config_.delete_executions_properties_by_executions_id(),
      {Bind(execution_ids)}));
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::DeleteArtifactsById(
    absl::Span<const int64_t> artifact_ids) {
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.delete_artifacts_by_id(),
                                    {Bind(artifact_ids)}));
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.delete_artifacts_properties_by_artifacts_id(),
                   {Bind(artifact_ids)}));
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::DeleteContextsById(
    absl::Span<const int64_t> context_ids) {
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.delete_contexts_by_id(), {Bind(context_ids)}));
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.delete_contexts_properties_by_contexts_id(),
                   {Bind(context_ids)}));
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::DeleteEventsByArtifactsId(
    absl::Span<const int64_t> artifact_ids) {
  MLMD_RETURN_IF_ERROR(ExecuteQuery(
      query_config_.delete_events_by_artifacts_id(), {Bind(artifact_ids)}));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.delete_event_paths()));
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::DeleteEventsByExecutionsId(
    absl::Span<const int64_t> execution_ids) {
  MLMD_RETURN_IF_ERROR(ExecuteQuery(
      query_config_.delete_events_by_executions_id(), {Bind(execution_ids)}));
  MLMD_RETURN_IF_ERROR(ExecuteQuery(query_config_.delete_event_paths()));
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::DeleteAttributionsByContextsId(
    absl::Span<const int64_t> context_ids) {
  MLMD_RETURN_IF_ERROR(ExecuteQuery(
      query_config_.delete_attributions_by_contexts_id(), {Bind(context_ids)}));
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::DeleteAttributionsByArtifactsId(
    absl::Span<const int64_t> artifact_ids) {
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.delete_attributions_by_artifacts_id(),
                   {Bind(artifact_ids)}));
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::DeleteAssociationsByContextsId(
    absl::Span<const int64_t> context_ids) {
  MLMD_RETURN_IF_ERROR(ExecuteQuery(
      query_config_.delete_associations_by_contexts_id(), {Bind(context_ids)}));
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::DeleteAssociationsByExecutionsId(
    absl::Span<const int64_t> execution_ids) {
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.delete_associations_by_executions_id(),
                   {Bind(execution_ids)}));
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::DeleteParentContextsByParentIds(
    absl::Span<const int64_t> parent_context_ids) {
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.delete_parent_contexts_by_parent_ids(),
                   {Bind(parent_context_ids)}));
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::DeleteParentContextsByChildIds(
    absl::Span<const int64_t> child_context_ids) {
  MLMD_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.delete_parent_contexts_by_child_ids(),
                   {Bind(child_context_ids)}));
  return absl::OkStatus();
}
absl::Status PostgreSQLQueryExecutor::DeleteParentContextsByParentIdAndChildIds(
    int64_t parent_context_id, absl::Span<const int64_t> child_context_ids) {
  MLMD_RETURN_IF_ERROR(ExecuteQuery(
      query_config_.delete_parent_contexts_by_parent_id_and_child_ids(),
      {Bind(parent_context_id), Bind(child_context_ids)}));
  return absl::OkStatus();
}
}  // namespace ml_metadata
