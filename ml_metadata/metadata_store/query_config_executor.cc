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
#include "ml_metadata/metadata_store/query_config_executor.h"

#include <string>
#include <vector>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/util/json_util.h"
#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_metadata/metadata_store/list_operation_query_helper.h"
#include "ml_metadata/metadata_store/list_operation_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
namespace ml_metadata {

tensorflow::Status QueryConfigExecutor::InsertEventPath(
    int64 event_id, const Event::Path::Step& step) {
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
  return tensorflow::Status::OK();
}

tensorflow::Status QueryConfigExecutor::GetSchemaVersion(int64* db_version) {
  RecordSet record_set;
  tensorflow::Status maybe_schema_version_status =
      ExecuteQuery(query_config_.check_mlmd_env_table(), {}, &record_set);
  if (maybe_schema_version_status.ok()) {
    if (record_set.records_size() == 0) {
      return tensorflow::errors::Aborted(
          "In the given db, MLMDEnv table exists but no schema_version can be "
          "found. This may be due to concurrent connection to the empty "
          "database. Please retry connection.");

    } else if (record_set.records_size() > 1) {
      return tensorflow::errors::DataLoss(
          "In the given db, MLMDEnv table exists but schema_version cannot be "
          "resolved due to there being more than one rows with the schema "
          "version. Expecting a single row: ",
          record_set.DebugString());
    }
    CHECK(absl::SimpleAtoi(record_set.records(0).values(0), db_version));
    return tensorflow::Status::OK();
  }
  // if MLMDEnv does not exist, it may be the v0.13.2 release or an empty db.
  tensorflow::Status maybe_v0_13_2_status =
      ExecuteQuery(query_config_.check_tables_in_v0_13_2(), {}, &record_set);
  if (maybe_v0_13_2_status.ok()) {
    *db_version = 0;
    return tensorflow::Status::OK();
  }
  return tensorflow::errors::NotFound("it looks an empty db is given.");
}

tensorflow::Status QueryConfigExecutor::UpgradeMetadataSourceIfOutOfDate(
    bool enable_migration) {
  int64 db_version = 0;
  tensorflow::Status get_schema_version_status = GetSchemaVersion(&db_version);
  int64 lib_version = GetLibraryVersion();
  if (tensorflow::errors::IsNotFound(get_schema_version_status)) {
    db_version = lib_version;
  } else {
    TF_RETURN_IF_ERROR(get_schema_version_status);
  }

  bool is_compatible = false;
  TF_RETURN_IF_ERROR(IsCompatible(db_version, lib_version, &is_compatible));
  if (is_compatible) {
    return tensorflow::Status::OK();
  }
  if (db_version > lib_version) {
    return tensorflow::errors::FailedPrecondition(
        "MLMD database version ", db_version,
        " is greater than library version ", lib_version,
        ". Please upgrade the library to use the given database in order to "
        "prevent potential data loss. If data loss is acceptable, please"
        " downgrade the database using a newer version of library.");
  }
  // returns error if upgrade is explicitly disabled, as we are missing schema
  // and cannot continue with this library version.
  if (db_version < lib_version && !enable_migration) {
    return tensorflow::errors::FailedPrecondition(
        "MLMD database version ", db_version, " is older than library version ",
        lib_version,
        ". Schema migration is disabled. Please upgrade the database then use"
        " the library version; or switch to a older library version to use the"
        " current database.");
  }

  // migrate db_version to lib version
  const auto& migration_schemes = query_config_.migration_schemes();
  while (db_version < lib_version) {
    const int64 to_version = db_version + 1;
    if (migration_schemes.find(to_version) == migration_schemes.end()) {
      return tensorflow::errors::Internal(
          "Cannot find migration_schemes to version ", to_version);
    }
    for (const MetadataSourceQueryConfig::TemplateQuery& upgrade_query :
         migration_schemes.at(to_version).upgrade_queries()) {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          ExecuteQuery(upgrade_query.query()),
          absl::StrCat("Upgrade query failed: ", upgrade_query.query()));
    }
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        UpdateSchemaVersion(to_version), "Failed to update schema.");
    db_version = to_version;
  }
  return tensorflow::Status::OK();
}

tensorflow::Status QueryConfigExecutor::SelectLastInsertID(
    int64* last_insert_id) {
  RecordSet record_set;
  TF_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.select_last_insert_id(), {}, &record_set));
  if (record_set.records_size() == 0) {
    return tensorflow::errors::Internal(
        "Could not find last insert ID: no record");
  }
  const RecordSet::Record& record = record_set.records(0);
  if (record.values_size() == 0) {
    return tensorflow::errors::Internal(
        "Could not find last insert ID: missing value");
  }
  if (!absl::SimpleAtoi(record.values(0), last_insert_id)) {
    return tensorflow::errors::Internal(
        "Could not parse last insert ID as string");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ml_metadata::QueryConfigExecutor::CheckTablesIn_V0_13_2() {
  return ExecuteQuery(query_config_.check_tables_in_v0_13_2());
}

tensorflow::Status QueryConfigExecutor::DowngradeMetadataSource(
    const int64 to_schema_version) {
  const int64 lib_version = query_config_.schema_version();
  if (to_schema_version < 0 || to_schema_version > lib_version) {
    return tensorflow::errors::InvalidArgument(
        "MLMD cannot be downgraded to schema_version: ", to_schema_version,
        ". The target version should be greater or equal to 0, and the current"
        " library version: ",
        lib_version, " needs to be greater than the target version.");
  }
  int64 db_version = 0;
  tensorflow::Status get_schema_version_status = GetSchemaVersion(&db_version);
  // if it is an empty database, then we skip downgrade and returns.
  if (tensorflow::errors::IsNotFound(get_schema_version_status)) {
    return tensorflow::errors::InvalidArgument(
        "Empty database is given. Downgrade operation is not needed.");
  }
  TF_RETURN_IF_ERROR(get_schema_version_status);
  if (db_version > lib_version) {
    return tensorflow::errors::FailedPrecondition(
        "MLMD database version ", db_version,
        " is greater than library version ", lib_version,
        ". The current library does not know how to downgrade it. "
        "Please upgrade the library to downgrade the schema.");
  }
  // perform downgrade
  const auto& migration_schemes = query_config_.migration_schemes();
  while (db_version > to_schema_version) {
    const int64 to_version = db_version - 1;
    if (migration_schemes.find(to_version) == migration_schemes.end()) {
      return tensorflow::errors::Internal(
          "Cannot find migration_schemes to version ", to_version);
    }
    for (const MetadataSourceQueryConfig::TemplateQuery& downgrade_query :
         migration_schemes.at(to_version).downgrade_queries()) {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          ExecuteQuery(downgrade_query),
          "Failed to migrate existing db; the "
          "migration transaction rolls back.");
    }
    // at version 0, v0.13.2, there is no schema version information.
    if (to_version > 0) {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(UpdateSchemaVersion(to_version),
          "Failed to migrate existing db; the migration transaction rolls "
          "back.");
    }
    db_version = to_version;
  }
  return tensorflow::Status::OK();
}

std::string QueryConfigExecutor::Bind(const char* value) {
  return absl::StrCat("'", metadata_source_->EscapeString(value), "'");
}

std::string QueryConfigExecutor::Bind(absl::string_view value) {
  return absl::StrCat("'", metadata_source_->EscapeString(value), "'");
}

std::string QueryConfigExecutor::Bind(int value) {
  return std::to_string(value);
}

std::string QueryConfigExecutor::Bind(int64 value) {
  return std::to_string(value);
}

std::string QueryConfigExecutor::Bind(double value) {
  return std::to_string(value);
}

std::string QueryConfigExecutor::Bind(bool value) { return value ? "1" : "0"; }

// Utility method to bind an Event::Type enum value to a SQL clause.
// Event::Type is an enum (integer), EscapeString is not applicable.
std::string QueryConfigExecutor::Bind(const Event::Type value) {
  return std::to_string(value);
}

std::string QueryConfigExecutor::Bind(PropertyType value) {
  return std::to_string((int)value);
}

std::string QueryConfigExecutor::Bind(TypeKind value) {
  return std::to_string((int)value);
}

std::string QueryConfigExecutor::Bind(Artifact::State value) {
  return std::to_string((int)value);
}

std::string QueryConfigExecutor::Bind(Execution::State value) {
  return std::to_string((int)value);
}

std::string QueryConfigExecutor::Bind(const std::vector<int64>& value) {
  return absl::StrJoin(value, ", ");
}

std::string QueryConfigExecutor::BindValue(const Value& value) {
  switch (value.value_case()) {
    case PropertyType::INT:
      return Bind(value.int_value());
    case PropertyType::DOUBLE:
      return Bind(value.double_value());
    case PropertyType::STRING:
      return Bind(value.string_value());
    default:
      LOG(FATAL) << "Unknown registered property type: " << value.value_case()
                 << "This is an internal error: properties should have been "
                    "checked before"
                    " they got here";
  }
}

std::string QueryConfigExecutor::BindDataType(const Value& value) {
  switch (value.value_case()) {
    case PropertyType::INT: {
      return "int_value";
      break;
    }
    case PropertyType::DOUBLE: {
      return "double_value";
      break;
    }
    case PropertyType::STRING: {
      return "string_value";
      break;
    }
    default: {
      LOG(FATAL) << "Unexpected oneof: " << value.DebugString();
    }
  }
}

std::string QueryConfigExecutor::Bind(bool exists,
                                      const google::protobuf::Message& message) {
  if (exists) {
    std::string json_output;
    CHECK(::google::protobuf::util::MessageToJsonString(message, &json_output).ok())
        << "Could not write proto to JSON: " << message.DebugString();
    return Bind(json_output);
  } else {
    return "null";
  }
}

#if (!defined(__APPLE__) && !defined(_WIN32))
string QueryConfigExecutor::Bind(
    const google::protobuf::int64 value) {
  return std::to_string(value);
}
#endif

tensorflow::Status QueryConfigExecutor::ExecuteQuery(const std::string& query) {
  RecordSet record_set;
  return metadata_source_->ExecuteQuery(query, &record_set);
}

tensorflow::Status QueryConfigExecutor::ExecuteQuery(const std::string& query,
                                                     RecordSet* record_set) {
  return metadata_source_->ExecuteQuery(query, record_set);
}

tensorflow::Status QueryConfigExecutor::ExecuteQuery(
    const MetadataSourceQueryConfig::TemplateQuery& template_query,
    const std::vector<std::string>& parameters, RecordSet* record_set) {
  if (parameters.size() > 10) {
    return tensorflow::errors::InvalidArgument(
        "Template query has too many parameters (at most 10 is supported).");
  }
  if (template_query.parameter_num() != parameters.size()) {
    LOG(FATAL) << "Template query parameter_num does not match with given "
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

tensorflow::Status QueryConfigExecutor::IsCompatible(int64 db_version,
                                                     int64 lib_version,
                                                     bool* is_compatible) {
  // Currently, we don't support a database version that is older than the
  // library version. Revisit this if a more sophisticated rule is required.
  *is_compatible = (db_version == lib_version);
  return tensorflow::Status::OK();
}

tensorflow::Status QueryConfigExecutor::InitMetadataSource() {
  TF_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_type_table()));
  TF_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_type_property_table()));
  TF_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_artifact_table()));
  TF_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.create_artifact_property_table()));
  TF_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_execution_table()));
  TF_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.create_execution_property_table()));
  TF_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_event_table()));
  TF_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_event_path_table()));
  TF_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_mlmd_env_table()));
  TF_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_context_table()));
  TF_RETURN_IF_ERROR(
      ExecuteQuery(query_config_.create_context_property_table()));
  TF_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_association_table()));
  TF_RETURN_IF_ERROR(ExecuteQuery(query_config_.create_attribution_table()));

  int64 library_version = GetLibraryVersion();
  tensorflow::Status insert_schema_version_status =
      InsertSchemaVersion(library_version);
  if (!insert_schema_version_status.ok()) {
    int64 db_version = -1;
    TF_RETURN_IF_ERROR(GetSchemaVersion(&db_version));
    if (db_version != library_version) {
      return tensorflow::errors::DataLoss(
          "The database cannot be initialized with the schema_version in the "
          "current library. Current library version: ",
          library_version, ", the db version on record is: ", db_version,
          ". It may result from a data race condition caused by other "
          "concurrent MLMD's migration procedures.");
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status QueryConfigExecutor::InitMetadataSourceIfNotExists(
    const bool enable_upgrade_migration) {
  // check db version, and make it to align with the lib version.
  TF_RETURN_IF_ERROR(
      UpgradeMetadataSourceIfOutOfDate(enable_upgrade_migration));
  // if lib and db versions align, we check the required tables for the lib.
  std::vector<std::pair<tensorflow::Status, std::string>> checks;
  checks.push_back({CheckTypeTable(), "type_table"});
  checks.push_back({CheckTypePropertyTable(), "type_property_table"});
  checks.push_back({CheckArtifactTable(), "artifact_table"});
  checks.push_back({CheckArtifactPropertyTable(), "artifact_property_table"});
  checks.push_back({CheckExecutionTable(), "execution_table"});
  checks.push_back({CheckExecutionPropertyTable(), "execution_property_table"});
  checks.push_back({CheckEventTable(), "event_table"});
  checks.push_back({CheckEventPathTable(), "event_path_table"});
  checks.push_back({CheckMLMDEnvTable(), "mlmd_env_table"});
  checks.push_back({CheckContextTable(), "context_table"});
  checks.push_back({CheckContextPropertyTable(), "context_property_table"});
  checks.push_back({CheckAssociationTable(), "check_association_table"});
  checks.push_back({CheckAttributionTable(), "check_attribution_table"});
  std::vector<std::string> missing_schema_error_messages;
  std::vector<std::string> successful_checks;
  std::vector<std::string> failing_checks;
  for (const auto& check_pair : checks) {
    const tensorflow::Status &check = check_pair.first;
    const std::string &name = check_pair.second;
    if (!check.ok()) {
      missing_schema_error_messages.push_back(check.error_message());
      failing_checks.push_back(name);
    } else {
      successful_checks.push_back(name);
    }
  }

  // all table required by the current lib version exists
  if (missing_schema_error_messages.empty()) return tensorflow::Status::OK();

  // some table exists, but not all.
  if (checks.size() != missing_schema_error_messages.size()) {
    return tensorflow::errors::Aborted(
        "There are a subset of tables in MLMD instance. This may be due to "
        "concurrent connection to the empty database. "
        "Please retry the connection. checks: ", checks.size(),
        " errors: ", missing_schema_error_messages.size(),
        ", present tables: ", absl::StrJoin(successful_checks, ", "),
        ", missing tables: ", absl::StrJoin(failing_checks, ", "),
        " Errors: ",
        absl::StrJoin(missing_schema_error_messages, "\n"));
  }

  // no table exists, then init the MetadataSource
  return InitMetadataSource();
}

tensorflow::Status QueryConfigExecutor::InsertExecutionType(
    const std::string& type_name, bool has_input_type,
    const google::protobuf::Message& input_type, bool has_output_type,
    const google::protobuf::Message& output_type, int64* execution_type_id) {
  return ExecuteQuerySelectLastInsertID(
      query_config_.insert_execution_type(),
      {Bind(type_name), Bind(has_input_type, input_type),
       Bind(has_output_type, output_type)},
      execution_type_id);
}

template <typename Node>
tensorflow::Status QueryConfigExecutor::ListNodeIDsUsingOptions(
    const ListOperationOptions& options, RecordSet* record_set) {
  int64 id_offset, field_offset;
  if (!options.next_page_token().empty()) {
    ListOperationNextPageToken next_page_token;
    TF_RETURN_IF_ERROR(DecodeListOperationNextPageToken(
        options.next_page_token(), next_page_token));
    TF_RETURN_IF_ERROR(ValidateListOperationOptionsAreIdentical(
        next_page_token.set_options(), options));
    id_offset = next_page_token.id_offset();
    field_offset = next_page_token.field_offset();
  } else {
    SetListOperationInitialValues(options, field_offset, id_offset);
  }

  std::string sql_query;
  if (std::is_same<Node, Artifact>::value) {
    sql_query = "SELECT `id` FROM `Artifact` WHERE";
  } else if (std::is_same<Node, Execution>::value) {
    sql_query = "SELECT `id` FROM `Execution` WHERE";
  } else if (std::is_same<Node, Context>::value) {
    sql_query = "select `id` FROM `Context` WHERE";
  } else {
    return tensorflow::errors::InvalidArgument(
        "Invalid Node passed to ListNodeIDsUsingOptions");
  }
  TF_RETURN_IF_ERROR(AppendOrderingThresholdClause(options, id_offset,
                                                   field_offset, sql_query));
  TF_RETURN_IF_ERROR(AppendOrderByClause(options, sql_query));
  TF_RETURN_IF_ERROR(AppendLimitClause(options, sql_query));
  return ExecuteQuery(sql_query, record_set);
}

tensorflow::Status QueryConfigExecutor::ListArtifactIDsUsingOptions(
    const ListOperationOptions& options, RecordSet* record_set) {
  return ListNodeIDsUsingOptions<Artifact>(options, record_set);
}

tensorflow::Status QueryConfigExecutor::ListExecutionIDsUsingOptions(
    const ListOperationOptions& options, RecordSet* record_set) {
  return ListNodeIDsUsingOptions<Execution>(options, record_set);
}

tensorflow::Status QueryConfigExecutor::ListContextIDsUsingOptions(
    const ListOperationOptions& options, RecordSet* record_set) {
  return ListNodeIDsUsingOptions<Context>(options, record_set);
}


}  // namespace ml_metadata
