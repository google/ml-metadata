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
#include "ml_metadata/util/metadata_source_query_config.h"

#include "absl/strings/str_cat.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace ml_metadata {
namespace util {
namespace {

// A set of common template queries used by the MetadataAccessObject for SQLite
// based MetadataSource.
// no-lint to support vc (C2026) 16380 max length for char[].
const std::string kBaseQueryConfig = absl::StrCat( // NOLINT
R"pb(
  schema_version: 5
  drop_type_table { query: " DROP TABLE IF EXISTS `Type`; " }
  create_type_table {
    query: " CREATE TABLE IF NOT EXISTS `Type` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `type_kind` TINYINT(1) NOT NULL, "
           "   `input_type` TEXT, "
           "   `output_type` TEXT"
           " ); "
  }
  check_type_table {
    query: " SELECT "
           "`id`, `name`, `type_kind`, `input_type`, `output_type` "
           " FROM `Type` LIMIT 1; "
  }

  insert_artifact_type {
    query: " INSERT INTO `Type`( "
           "   `name`, `type_kind` "
           ") VALUES($0, 1);"
    parameter_num: 1
  }

  insert_execution_type {
    query: " INSERT INTO `Type`( "
           "   `name`, `type_kind`, `input_type`,  `output_type` "
           ") VALUES($0, 0, $1, $2);"
    parameter_num: 3
  }

  insert_context_type {
    query: " INSERT INTO `Type`( "
           "   `name`, `type_kind` "
           ") VALUES($0, 2);"
    parameter_num: 1
  }

  select_type_by_id {
    query: " SELECT `id`, `name`, `input_type`, `output_type` "
           " from `Type` "
           " WHERE id = $0 and type_kind = $1; "
    parameter_num: 2
  }

  select_type_by_name {
    query: " SELECT `id`, `name`, `input_type`, `output_type` "
           " from `Type` "
           " WHERE name = $0 and type_kind = $1; "
    parameter_num: 2
  }

  select_all_types {
    query: " SELECT `id`, `name`, `input_type`, `output_type` "
           " from `Type` "
           " WHERE type_kind = $0; "
    parameter_num: 1
  }
  drop_type_property_table { query: " DROP TABLE IF EXISTS `TypeProperty`; " }
  create_type_property_table {
    query: " CREATE TABLE IF NOT EXISTS `TypeProperty` ( "
           "   `type_id` INT NOT NULL, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `data_type` INT NULL, "
           " PRIMARY KEY (`type_id`, `name`)); "
  }
  check_type_property_table {
    query: " SELECT `type_id`, `name`, `data_type` "
           " FROM `TypeProperty` LIMIT 1; "
  }
  insert_type_property {
    query: " INSERT INTO `TypeProperty`( "
           "   `type_id`, `name`, `data_type` "
           ") VALUES($0, $1, $2);"
    parameter_num: 3
  }
  select_property_by_type_id {
    query: " SELECT `name` as `key`, `data_type` as `value` "
           " from `TypeProperty` "
           " WHERE `type_id` = $0; "
    parameter_num: 1
  }
  select_last_insert_id { query: " SELECT last_insert_rowid(); " }
)pb",
R"pb(
  drop_artifact_table { query: " DROP TABLE IF EXISTS `Artifact`; " }
  create_artifact_table {
    query: " CREATE TABLE IF NOT EXISTS `Artifact` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `uri` TEXT, "
           "   `state` INT, "
           "   `name` VARCHAR(255), "
           "   `create_time_since_epoch` INT NOT NULL DEFAULT 0, "
           "   `last_update_time_since_epoch` INT NOT NULL DEFAULT 0, "
           "   UNIQUE(`type_id`, `name`) "
           " ); "
  }
  check_artifact_table {
    query: " SELECT `id`, `type_id`, `uri`, `state`, `name`, "
           "        `create_time_since_epoch`, `last_update_time_since_epoch` "
           " FROM `Artifact` LIMIT 1; "
  }
  insert_artifact {
    query: " INSERT INTO `Artifact`( "
           "   `type_id`, `uri`, `state`, `name`, `create_time_since_epoch`, "
           "   `last_update_time_since_epoch` "
           ") VALUES($0, $1, $2, $3, $4, $5);"
    parameter_num: 6
  }
  select_artifact_by_id {
    query: " SELECT `type_id`, `uri`, `state`, `name`, "
           "        `create_time_since_epoch`, `last_update_time_since_epoch` "
           " from `Artifact` "
           " WHERE id = $0; "
    parameter_num: 1
  }
  select_artifact_by_type_id_and_name {
    query: " SELECT `id` from `Artifact` WHERE `type_id` = $0 and `name` = $1; "
    parameter_num: 2
  }
  select_artifacts_by_type_id {
    query: " SELECT `id` from `Artifact` WHERE `type_id` = $0; "
    parameter_num: 1
  }
  select_artifacts_by_uri {
    query: " SELECT `id` from `Artifact` WHERE `uri` = $0; "
    parameter_num: 1
  }
  update_artifact {
    query: " UPDATE `Artifact` "
           " SET `type_id` = $1, `uri` = $2, `state` = $3, "
           "     `last_update_time_since_epoch` = $4 "
           " WHERE id = $0;"
    parameter_num: 5
  }
  drop_artifact_property_table {
    query: " DROP TABLE IF EXISTS `ArtifactProperty`; "
  }
  create_artifact_property_table {
    query: " CREATE TABLE IF NOT EXISTS `ArtifactProperty` ( "
           "   `artifact_id` INT NOT NULL, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `is_custom_property` TINYINT(1) NOT NULL, "
           "   `int_value` INT, "
           "   `double_value` DOUBLE, "
           "   `string_value` TEXT, "
           " PRIMARY KEY (`artifact_id`, `name`, `is_custom_property`)); "
  }
  check_artifact_property_table {
    query: " SELECT `artifact_id`, `name`, `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value` "
           " FROM `ArtifactProperty` LIMIT 1; "
  }
  insert_artifact_property {
    query: " INSERT INTO `ArtifactProperty`( "
           "   `artifact_id`, `name`, `is_custom_property`, `$0` "
           ") VALUES($1, $2, $3, $4);"
    parameter_num: 5
  }
  select_artifact_property_by_artifact_id {
    query: " SELECT `name` as `key`, `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value` "
           " from `ArtifactProperty` "
           " WHERE `artifact_id` = $0; "
    parameter_num: 1
  }
  update_artifact_property {
    query: " UPDATE `ArtifactProperty` "
           " SET `$0` = $1 "
           " WHERE `artifact_id` = $2 and `name` = $3;"
    parameter_num: 4
  }
  delete_artifact_property {
    query: " DELETE FROM `ArtifactProperty` "
           " WHERE `artifact_id` = $0 and `name` = $1;"
    parameter_num: 2
  }
)pb",
R"pb(
  drop_execution_table { query: " DROP TABLE IF EXISTS `Execution`; " }
  create_execution_table {
    query: " CREATE TABLE IF NOT EXISTS `Execution` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `last_known_state` INT, "
           "   `name` VARCHAR(255), "
           "   `create_time_since_epoch` INT NOT NULL DEFAULT 0, "
           "   `last_update_time_since_epoch` INT NOT NULL DEFAULT 0, "
           "   UNIQUE(`type_id`, `name`) "
           " ); "
  }
  check_execution_table {
    query: " SELECT `id`, `type_id`, `last_known_state`, `name`, "
           "        `create_time_since_epoch`, `last_update_time_since_epoch` "
           " FROM `Execution` LIMIT 1; "
  }
  insert_execution {
    query: " INSERT INTO `Execution`( "
           "   `type_id`, `last_known_state`, `name`, "
           "   `create_time_since_epoch`, `last_update_time_since_epoch` "
           ") VALUES($0, $1, $2, $3, $4);"
    parameter_num: 5
  }
  select_execution_by_id {
    query: " SELECT `type_id`, `last_known_state`, `name`, "
           "        `create_time_since_epoch`, `last_update_time_since_epoch` "
           " from `Execution` "
           " WHERE id = $0; "
    parameter_num: 1
  }
  select_execution_by_type_id_and_name {
    query: " SELECT `id` from `Execution` WHERE `type_id` = $0 and `name` = $1; "
    parameter_num: 2
  }
  select_executions_by_type_id {
    query: " SELECT `id` from `Execution` WHERE `type_id` = $0; "
    parameter_num: 1
  }
  update_execution {
    query: " UPDATE `Execution` "
           " SET `type_id` = $1, `last_known_state` = $2, "
           "     `last_update_time_since_epoch` = $3 "
           " WHERE id = $0;"
    parameter_num: 4
  }
  drop_execution_property_table {
    query: " DROP TABLE IF EXISTS `ExecutionProperty`; "
  }
  create_execution_property_table {
    query: " CREATE TABLE IF NOT EXISTS `ExecutionProperty` ( "
           "   `execution_id` INT NOT NULL, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `is_custom_property` TINYINT(1) NOT NULL, "
           "   `int_value` INT, "
           "   `double_value` DOUBLE, "
           "   `string_value` TEXT, "
           " PRIMARY KEY (`execution_id`, `name`, `is_custom_property`)); "
  }
  check_execution_property_table {
    query: " SELECT `execution_id`, `name`, `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value` "
           " FROM `ExecutionProperty` LIMIT 1; "
  }
  insert_execution_property {
    query: " INSERT INTO `ExecutionProperty`( "
           "   `execution_id`, `name`, `is_custom_property`, `$0` "
           ") VALUES($1, $2, $3, $4);"
    parameter_num: 5
  }
  select_execution_property_by_execution_id {
    query: " SELECT `name` as `key`, `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value` "
           " from `ExecutionProperty` "
           " WHERE `execution_id` = $0; "
    parameter_num: 1
  }
  update_execution_property {
    query: " UPDATE `ExecutionProperty` "
           " SET `$0` = $1 "
           " WHERE `execution_id` = $2 and `name` = $3;"
    parameter_num: 4
  }
  delete_execution_property {
    query: " DELETE FROM `ExecutionProperty` "
           " WHERE `execution_id` = $0 and `name` = $1;"
    parameter_num: 2
  }
)pb",
R"pb(
  drop_context_table { query: " DROP TABLE IF EXISTS `Context`; " }
  create_context_table {
    query: " CREATE TABLE IF NOT EXISTS `Context` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `create_time_since_epoch` INT NOT NULL DEFAULT 0, "
           "   `last_update_time_since_epoch` INT NOT NULL DEFAULT 0, "
           "   UNIQUE(`type_id`, `name`) "
           " ); "
  }
  check_context_table {
    query: " SELECT `id`, `type_id`, `name`, "
           "        `create_time_since_epoch`, `last_update_time_since_epoch` "
           " FROM `Context` LIMIT 1; "
  }
  insert_context {
    query: " INSERT INTO `Context`( "
           "   `type_id`, `name`, "
           "   `create_time_since_epoch`, `last_update_time_since_epoch` "
           ") VALUES($0, $1, $2, $3);"
    parameter_num: 4
  }
  select_context_by_id {
    query: " SELECT `type_id`, `name`, `create_time_since_epoch`, "
           "        `last_update_time_since_epoch` "
           " from `Context` WHERE id = $0; "
    parameter_num: 1
  }
  select_contexts_by_type_id {
    query: " SELECT `id` from `Context` WHERE `type_id` = $0; "
    parameter_num: 1
  }
  select_context_by_type_id_and_name {
    query: " SELECT `id` from `Context` WHERE `type_id` = $0 and `name` = $1; "
    parameter_num: 2
  }
  update_context {
    query: " UPDATE `Context` "
           " SET `type_id` = $1, `name` = $2, "
           "     `last_update_time_since_epoch` = $3 "
           " WHERE id = $0;"
    parameter_num: 4
  }
  drop_context_property_table {
    query: " DROP TABLE IF EXISTS `ContextProperty`; "
  }
  create_context_property_table {
    query: " CREATE TABLE IF NOT EXISTS `ContextProperty` ( "
           "   `context_id` INT NOT NULL, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `is_custom_property` TINYINT(1) NOT NULL, "
           "   `int_value` INT, "
           "   `double_value` DOUBLE, "
           "   `string_value` TEXT, "
           " PRIMARY KEY (`context_id`, `name`, `is_custom_property`)); "
  }
  check_context_property_table {
    query: " SELECT `context_id`, `name`, `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value` "
           " FROM `ContextProperty` LIMIT 1; "
  }
  insert_context_property {
    query: " INSERT INTO `ContextProperty`( "
           "   `context_id`, `name`, `is_custom_property`, `$0` "
           ") VALUES($1, $2, $3, $4);"
    parameter_num: 5
  }
  select_context_property_by_context_id {
    query: " SELECT `name` as `key`, `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value` "
           " from `ContextProperty` "
           " WHERE `context_id` = $0; "
    parameter_num: 1
  }
  update_context_property {
    query: " UPDATE `ContextProperty` "
           " SET `$0` = $1 "
           " WHERE `context_id` = $2 and `name` = $3;"
    parameter_num: 4
  }
  delete_context_property {
    query: " DELETE FROM `ContextProperty` "
           " WHERE `context_id` = $0 and `name` = $1;"
    parameter_num: 2
  }
)pb",
R"pb(
  drop_event_table { query: " DROP TABLE IF EXISTS `Event`; " }
  create_event_table {
    query: " CREATE TABLE IF NOT EXISTS `Event` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `artifact_id` INT NOT NULL, "
           "   `execution_id` INT NOT NULL, "
           "   `type` INT NOT NULL, "
           "   `milliseconds_since_epoch` INT "
           " ); "
  }
  check_event_table {
    query: " SELECT `id`, `artifact_id`, `execution_id`, "
           "        `type`, `milliseconds_since_epoch` "
           " FROM `Event` LIMIT 1; "
  }
  insert_event {
    query: " INSERT INTO `Event`( "
           "   `artifact_id`, `execution_id`, `type`, "
           "   `milliseconds_since_epoch` "
           ") VALUES($0, $1, $2, $3);"
    parameter_num: 4
  }
  select_event_by_artifact_ids {
    query: " SELECT `id`, `artifact_id`, `execution_id`, "
           "        `type`, `milliseconds_since_epoch` "
           " from `Event` "
           " WHERE `artifact_id` IN ($0); "
    parameter_num: 1
  }
  select_event_by_execution_ids {
    query: " SELECT `id`, `artifact_id`, `execution_id`, "
           "        `type`, `milliseconds_since_epoch` "
           " from `Event` "
           " WHERE `execution_id` IN ($0); "
    parameter_num: 1
  }
  drop_event_path_table { query: " DROP TABLE IF EXISTS `EventPath`; " }
  create_event_path_table {
    query: " CREATE TABLE IF NOT EXISTS `EventPath` ( "
           "   `event_id` INT NOT NULL, "
           "   `is_index_step` TINYINT(1) NOT NULL, "
           "   `step_index` INT, "
           "   `step_key` TEXT "
           " ); "
  }
  check_event_path_table {
    query: " SELECT `event_id`, `is_index_step`, `step_index`, `step_key` "
           " FROM `EventPath` LIMIT 1; "
  }
  insert_event_path {
    query: " INSERT INTO `EventPath`( "
           "   `event_id`, `is_index_step`, `$1` "
           ") VALUES($0, $2, $3);"
    parameter_num: 4
  }
  select_event_path_by_event_ids {
    query: " SELECT `event_id`, `is_index_step`, `step_index`, `step_key` "
           " from `EventPath` "
           " WHERE `event_id` IN ($0); "
    parameter_num: 1
  }
)pb",
R"pb(
  drop_association_table { query: " DROP TABLE IF EXISTS `Association`; " }
  create_association_table {
    query: " CREATE TABLE IF NOT EXISTS `Association` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `context_id` INT NOT NULL, "
           "   `execution_id` INT NOT NULL, "
           "   UNIQUE(`context_id`, `execution_id`) "
           " ); "
  }
  check_association_table {
    query: " SELECT `id`, `context_id`, `execution_id` "
           " FROM `Association` LIMIT 1; "
  }
  insert_association {
    query: " INSERT INTO `Association`( "
           "   `context_id`, `execution_id` "
           ") VALUES($0, $1);"
    parameter_num: 2
  }
  select_association_by_context_id {
    query: " SELECT `id`, `context_id`, `execution_id` "
           " from `Association` "
           " WHERE `context_id` = $0; "
    parameter_num: 1
  }
  select_association_by_execution_id {
    query: " SELECT `id`, `context_id`, `execution_id` "
           " from `Association` "
           " WHERE `execution_id` = $0; "
    parameter_num: 1
  }
  drop_attribution_table { query: " DROP TABLE IF EXISTS `Attribution`; " }
  create_attribution_table {
    query: " CREATE TABLE IF NOT EXISTS `Attribution` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `context_id` INT NOT NULL, "
           "   `artifact_id` INT NOT NULL, "
           "   UNIQUE(`context_id`, `artifact_id`) "
           " ); "
  }
  check_attribution_table {
    query: " SELECT `id`, `context_id`, `artifact_id` "
           " FROM `Attribution` LIMIT 1; "
  }
  insert_attribution {
    query: " INSERT INTO `Attribution`( "
           "   `context_id`, `artifact_id` "
           ") VALUES($0, $1);"
    parameter_num: 2
  }
  select_attribution_by_context_id {
    query: " SELECT `id`, `context_id`, `artifact_id` "
           " from `Attribution` "
           " WHERE `context_id` = $0; "
    parameter_num: 1
  }
  select_attribution_by_artifact_id {
    query: " SELECT `id`, `context_id`, `artifact_id` "
           " from `Attribution` "
           " WHERE `artifact_id` = $0; "
    parameter_num: 1
  }
  drop_mlmd_env_table { query: " DROP TABLE IF EXISTS `MLMDEnv`; " }
  create_mlmd_env_table {
    query: " CREATE TABLE IF NOT EXISTS `MLMDEnv` ( "
           "   `schema_version` INTEGER PRIMARY KEY "
           " ); "
  }
  check_mlmd_env_table { query: " SELECT `schema_version` FROM `MLMDEnv`; " }
  insert_schema_version {
    query: " INSERT INTO `MLMDEnv`(`schema_version`) VALUES($0); "
    parameter_num: 1
  }
  update_schema_version {
    query: " UPDATE `MLMDEnv` SET `schema_version` = $0; "
    parameter_num: 1
  }
  check_tables_in_v0_13_2 {
    query: " SELECT `Type`.`is_artifact_type` from "
           " `Artifact`, `Event`, `Execution`, `Type`, `ArtifactProperty`, "
           " `EventPath`, `ExecutionProperty`, `TypeProperty` LIMIT 1; "
  }
)pb");

// no-lint to support vc (C2026) 16380 max length for char[].
const std::string kSQLiteMetadataSourceQueryConfig = absl::StrCat( // NOLINT
R"pb(
  metadata_source_type: SQLITE_METADATA_SOURCE
  # downgrade to 0.13.2 (i.e., v0), and drop the MLMDEnv table.
  migration_schemes {
    key: 0
    value: {
      # downgrade queries from version 1
      downgrade_queries { query: " DROP TABLE IF EXISTS `MLMDEnv`; " }
      # check the tables are deleted properly
      downgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `tbl_name` = 'MLMDEnv'; "
        }
      }
    }
  }
)pb",
R"pb(
  # From 0.13.2 to v1, it creates a new MLMDEnv table to track schema_version.
  migration_schemes {
    key: 1
    value: {
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `MLMDEnv` ( "
               "   `schema_version` INTEGER PRIMARY KEY "
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO `MLMDEnv`(`schema_version`) VALUES(0); "
      }
      # v0.13.2 release
      upgrade_verification {
        # reproduce the v0.13.2 release table setup
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `Type` ( "
                 "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "   `name` VARCHAR(255) NOT NULL, "
                 "   `is_artifact_type` TINYINT(1) NOT NULL, "
                 "   `input_type` TEXT, "
                 "   `output_type` TEXT "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `TypeProperty` ( "
                 "   `type_id` INT NOT NULL, "
                 "   `name` VARCHAR(255) NOT NULL, "
                 "   `data_type` INT NULL, "
                 " PRIMARY KEY (`type_id`, `name`)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `Artifact` ( "
                 "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "   `type_id` INT NOT NULL, "
                 "   `uri` TEXT "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `ArtifactProperty` ( "
                 "   `artifact_id` INT NOT NULL, "
                 "   `name` VARCHAR(255) NOT NULL, "
                 "   `is_custom_property` TINYINT(1) NOT NULL, "
                 "   `int_value` INT, "
                 "   `double_value` DOUBLE, "
                 "   `string_value` TEXT, "
                 " PRIMARY KEY (`artifact_id`, `name`, `is_custom_property`)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `Execution` ( "
                 "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "   `type_id` INT NOT NULL "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `ExecutionProperty` ( "
                 "   `execution_id` INT NOT NULL, "
                 "   `name` VARCHAR(255) NOT NULL, "
                 "   `is_custom_property` TINYINT(1) NOT NULL, "
                 "   `int_value` INT, "
                 "   `double_value` DOUBLE, "
                 "   `string_value` TEXT, "
                 " PRIMARY KEY (`execution_id`, `name`, `is_custom_property`)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `Event` ( "
                 "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "   `artifact_id` INT NOT NULL, "
                 "   `execution_id` INT NOT NULL, "
                 "   `type` INT NOT NULL, "
                 "   `milliseconds_since_epoch` INT "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `EventPath` ( "
                 "   `event_id` INT NOT NULL, "
                 "   `is_index_step` TINYINT(1) NOT NULL, "
                 "   `step_index` INT, "
                 "   `step_key` TEXT "
                 " ); "
        }
        # check the new table has 1 row
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `MLMDEnv`; "
        }
      }
      # downgrade queries from version 2, drop all ContextTypes and rename the
      # the `type_kind` back to `is_artifact_type` column.
      downgrade_queries { query: " DELETE FROM `Type` WHERE `type_kind` = 2; " }
      downgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `TypeTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `is_artifact_type` TINYINT(1) NOT NULL, "
               "   `input_type` TEXT, "
               "   `output_type` TEXT"
               " ); "
      }
      downgrade_queries {
        query: " INSERT INTO `TypeTemp` SELECT * FROM `Type`; "
      }
      downgrade_queries { query: " DROP TABLE `Type`; " }
      downgrade_queries { query: " ALTER TABLE `TypeTemp` rename to `Type`; " }
      # check the tables are deleted properly
      downgrade_verification {
        # populate the `Type` table with context types.
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`name`, `type_kind`, `input_type`, `output_type`) "
                 " VALUES ('execution_type', 0, 'input', 'output'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`name`, `type_kind`, `input_type`, `output_type`) "
                 " VALUES ('artifact_type', 1, 'input', 'output'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`name`, `type_kind`, `input_type`, `output_type`) "
                 " VALUES ('context_type', 2, 'input', 'output'); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `Type` "
                 " WHERE `is_artifact_type` = 2; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` "
                 " WHERE `is_artifact_type` = 1 AND `name` = 'artifact_type'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` "
                 " WHERE `is_artifact_type` = 0 AND `name` = 'execution_type'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v2, to support context type, and we renamed `is_artifact_type` column in
  # `Type` table.
  migration_schemes {
    key: 2
    value: {
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `TypeTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `type_kind` TINYINT(1) NOT NULL, "
               "   `input_type` TEXT, "
               "   `output_type` TEXT"
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO `TypeTemp` SELECT * FROM `Type`; "
      }
      upgrade_queries { query: " DROP TABLE `Type`; " }
      upgrade_queries { query: " ALTER TABLE `TypeTemp` rename to `Type`; " }
      upgrade_verification {
        # populate one ArtifactType and one ExecutionType.
        previous_version_setup_queries {
          query: " INSERT INTO `Type` (`name`, `is_artifact_type`) VALUES "
                 " ('artifact_type', 1); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`name`, `is_artifact_type`, `input_type`, `output_type`) "
                 " VALUES ('execution_type', 0, 'input', 'output'); "
        }
        # check after migration, the existing types are the same including id.
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` WHERE "
                 " `id` = 1 AND `type_kind` = 1 AND `name` = 'artifact_type'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` WHERE "
                 " `id` = 2 AND `type_kind` = 0 AND `name` = 'execution_type' "
                 " AND `input_type` = 'input' AND `output_type` = 'output'; "
        }
      }
      # downgrade queries from version 3
      downgrade_queries { query: " DROP TABLE IF EXISTS `Context`; " }
      downgrade_queries { query: " DROP TABLE IF EXISTS `ContextProperty`; " }
      # check the tables are deleted properly
      downgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `tbl_name` = 'Context'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `tbl_name` = 'ContextProperty'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v3, to support context, we added two tables `Context` and
  # `ContextProperty`, and made no change to other existing records.
  migration_schemes {
    key: 3
    value: {
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `Context` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `type_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   UNIQUE(`type_id`, `name`) "
               " ); "
      }
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ContextProperty` ( "
               "   `context_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `is_custom_property` TINYINT(1) NOT NULL, "
               "   `int_value` INT, "
               "   `double_value` DOUBLE, "
               "   `string_value` TEXT, "
               " PRIMARY KEY (`context_id`, `name`, `is_custom_property`)); "
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `id`, `type_id`, `name` FROM `Context` "
                 " ); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `context_id`, `name`, `is_custom_property`, "
                 "          `int_value`, `double_value`, `string_value` "
                 "    FROM `ContextProperty` "
                 " ); "
        }
      }
      # downgrade queries from version 4
      downgrade_queries { query: " DROP TABLE IF EXISTS `Association`; " }
      downgrade_queries { query: " DROP TABLE IF EXISTS `Attribution`; " }
      # check the tables are deleted properly
      downgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `tbl_name` = 'Association'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `tbl_name` = 'Attribution'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v4, to support context-execution association and context-artifact
  # attribution, we added two tables `Association` and `Attribution` and made
  # no change to other existing records.
  migration_schemes {
    key: 4
    value: {
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `Association` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `context_id` INT NOT NULL, "
               "   `execution_id` INT NOT NULL, "
               "   UNIQUE(`context_id`, `execution_id`) "
               " ); "
      }
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `Attribution` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `context_id` INT NOT NULL, "
               "   `artifact_id` INT NOT NULL, "
               "   UNIQUE(`context_id`, `artifact_id`) "
               " ); "
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `id`, `context_id`, `execution_id` "
                 "   FROM `Association` "
                 " ); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `id`, `context_id`, `artifact_id` "
                 "   FROM `Attribution` "
                 " ); "
        }
      }
      # downgrade queries from version 5
      downgrade_queries {
        query: " CREATE TABLE `ArtifactTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `type_id` INT NOT NULL, "
               "   `uri` TEXT "
               " ); "
      }
      downgrade_queries {
        query: " INSERT INTO `ArtifactTemp` "
               " SELECT `id`, `type_id`, `uri` FROM `Artifact`; "
      }
      downgrade_queries { query: " DROP TABLE `Artifact`; " }
      downgrade_queries {
        query: " ALTER TABLE `ArtifactTemp` RENAME TO `Artifact`; "
      }
      downgrade_queries {
        query: " CREATE TABLE `ExecutionTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `type_id` INT NOT NULL "
               " ); "
      }
      downgrade_queries {
        query: " INSERT INTO `ExecutionTemp` "
               " SELECT `id`, `type_id` FROM `Execution`; "
      }
      downgrade_queries { query: " DROP TABLE `Execution`; " }
      downgrade_queries {
        query: " ALTER TABLE `ExecutionTemp` RENAME TO `Execution`; "
      }
      downgrade_queries {
        query: " CREATE TABLE `ContextTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `type_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   UNIQUE(`type_id`, `name`) "
               " ); "
      }
      downgrade_queries {
        query: " INSERT INTO `ContextTemp` "
               " SELECT `id`, `type_id`, `name` FROM `Context`; "
      }
      downgrade_queries { query: " DROP TABLE `Context`; " }
      downgrade_queries {
        query: " ALTER TABLE `ContextTemp` RENAME TO `Context`; "
      }
      # verify if the downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Artifact`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Artifact` "
                 " (`id`, `type_id`, `uri`, `state`, `name`, "
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 'uri1', 1, NULL, 0, 1); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Artifact` "
                 " (`id`, `type_id`, `uri`, `state`, `name`, "
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (2, 3, 'uri2', NULL, 'name2', 1, 0); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Execution`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Execution` "
                 " (`id`, `type_id`, `last_known_state`, `name`, "
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 1, NULL, 0, 1); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Context`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Context` "
                 " (`id`, `type_id`, `name`, "
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 'name1', 1, 0); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 2 FROM `Artifact`; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Artifact` "
                 "   WHERE `id` = 1 and `type_id` = 2 and `uri` = 'uri1' "
                 " ); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Artifact` "
                 "   WHERE `id` = 2 and `type_id` = 3 and `uri` = 'uri2' "
                 " ); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Execution`; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Execution` "
                 "   WHERE `id` = 1 and `type_id` = 2 "
                 " ); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Context` "
                 "   WHERE `id` = 1 and `type_id` = 2 "
                 " ); "
        }
      }
    }
  }
)pb",
R"pb(
  # In v5, to support MLMD based orchestration better, we added state, time-
  # stamps, as well as user generated unique name per type to Artifact,
  # Execution and Context.
  migration_schemes {
    key: 5
    value: {
      # upgrade Artifact table
      upgrade_queries {
        query: " CREATE TABLE `ArtifactTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `type_id` INT NOT NULL, "
               "   `uri` TEXT, "
               "   `state` INT, "
               "   `name` VARCHAR(255), "
               "   `create_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   `last_update_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   UNIQUE(`type_id`, `name`) "
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO `ArtifactTemp` (`id`, `type_id`, `uri`) "
               " SELECT * FROM `Artifact`; "
      }
      upgrade_queries { query: " DROP TABLE `Artifact`; " }
      upgrade_queries {
        query: " ALTER TABLE `ArtifactTemp` RENAME TO `Artifact`; "
      }
      # upgrade Execution table
      upgrade_queries {
        query: " CREATE TABLE `ExecutionTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `type_id` INT NOT NULL, "
               "   `last_known_state` INT, "
               "   `name` VARCHAR(255), "
               "   `create_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   `last_update_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   UNIQUE(`type_id`, `name`) "
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO `ExecutionTemp` (`id`, `type_id`) "
               " SELECT * FROM `Execution`; "
      }
      upgrade_queries { query: " DROP TABLE `Execution`; " }
      upgrade_queries {
        query: " ALTER TABLE `ExecutionTemp` RENAME TO `Execution`; "
      }
      # upgrade Context table
      upgrade_queries {
        query: " ALTER TABLE `Context` "
               " ADD COLUMN `create_time_since_epoch` INT NOT NULL DEFAULT 0; "
      }
      upgrade_queries {
        query: " ALTER TABLE `Context` "
               " ADD COLUMN "
               "     `last_update_time_since_epoch` INT NOT NULL DEFAULT 0; "
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Artifact`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Artifact` "
                 " (`id`, `type_id`, `uri`) VALUES (1, 2, 'uri1'); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Execution`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Execution` "
                 " (`id`, `type_id`) VALUES (1, 3); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE `Context` ( "
                 "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "   `type_id` INT NOT NULL, "
                 "   `name` VARCHAR(255) NOT NULL, "
                 "   UNIQUE(`type_id`, `name`) "
                 " ); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Context` "
                 " (`id`, `type_id`, `name`) VALUES (1, 2, 'name1'); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT `id`, `type_id`, `uri`, `state`, `name`, "
                 "          `create_time_since_epoch`, "
                 "          `last_update_time_since_epoch` "
                 "   FROM `Artifact` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND `uri` = 'uri1' AND "
                 "         `create_time_since_epoch` = 0 AND "
                 "         `last_update_time_since_epoch` = 0 "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT `id`, `type_id`, `last_known_state`, `name`, "
                 "          `create_time_since_epoch`, "
                 "          `last_update_time_since_epoch` "
                 "   FROM `Execution` "
                 "   WHERE `id` = 1 AND `type_id` = 3 AND "
                 "         `create_time_since_epoch` = 0 AND "
                 "         `last_update_time_since_epoch` = 0 "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT `id`, `type_id`, `name`, "
                 "          `create_time_since_epoch`, "
                 "          `last_update_time_since_epoch` "
                 "   FROM `Context` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND `name` = 'name1' AND "
                 "         `create_time_since_epoch` = 0 AND "
                 "         `last_update_time_since_epoch` = 0 "
                 " ) as T1; "
        }
      }
    }
  }
)pb");

// Template queries for MySQLMetadataSources.
// no-lint to support vc (C2026) 16380 max length for char[].
const std::string kMySQLMetadataSourceQueryConfig = absl::StrCat( // NOLINT
R"pb(
  metadata_source_type: MYSQL_METADATA_SOURCE
  select_last_insert_id { query: " SELECT last_insert_id(); " }
  create_type_table {
    query: " CREATE TABLE IF NOT EXISTS `Type` ( "
           "   `id` INT PRIMARY KEY AUTO_INCREMENT, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `type_kind` TINYINT(1) NOT NULL, "
           "   `input_type` TEXT, "
           "   `output_type` TEXT"
           " ); "
  }
  create_artifact_table {
    query: " CREATE TABLE IF NOT EXISTS `Artifact` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `uri` TEXT, "
           "   `state` INT, "
           "   `name` VARCHAR(255), "
           "   `create_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   `last_update_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   CONSTRAINT UniqueArtifactTypeName UNIQUE(`type_id`, `name`) "
           " ); "
  }
  create_execution_table {
    query: " CREATE TABLE IF NOT EXISTS `Execution` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `last_known_state` INT, "
           "   `name` VARCHAR(255), "
           "   `create_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   `last_update_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   CONSTRAINT UniqueExecutionTypeName UNIQUE(`type_id`, `name`) "
           " ); "
  }
  create_context_table {
    query: " CREATE TABLE IF NOT EXISTS `Context` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `create_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   `last_update_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   UNIQUE(`type_id`, `name`) "
           " ); "
  }
  create_event_table {
    query: " CREATE TABLE IF NOT EXISTS `Event` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `artifact_id` INT NOT NULL, "
           "   `execution_id` INT NOT NULL, "
           "   `type` INT NOT NULL, "
           "   `milliseconds_since_epoch` BIGINT "
           " ); "
  }
  create_association_table {
    query: " CREATE TABLE IF NOT EXISTS `Association` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `context_id` INT NOT NULL, "
           "   `execution_id` INT NOT NULL, "
           "   UNIQUE(`context_id`, `execution_id`) "
           " ); "
  }
  create_attribution_table {
    query: " CREATE TABLE IF NOT EXISTS `Attribution` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `context_id` INT NOT NULL, "
           "   `artifact_id` INT NOT NULL, "
           "   UNIQUE(`context_id`, `artifact_id`) "
           " ); "
  }
  # downgrade to 0.13.2 (i.e., v0), and drops the MLMDEnv table.
  migration_schemes {
    key: 0
    value: {
      # downgrade queries from version 1
      downgrade_queries { query: " DROP TABLE IF EXISTS `MLMDEnv`; " }
      # check the tables are deleted properly
      downgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`tables` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'MLMDEnv'; "
        }
      }
    }
  }
)pb",
R"pb(
  migration_schemes {
    key: 1
    value: {
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `MLMDEnv` ( "
               "   `schema_version` INTEGER PRIMARY KEY "
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO `MLMDEnv`(`schema_version`) VALUES(0); "
      }
      # v0.13.2 release
      upgrade_verification {
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `Type` ( "
                 "   `id` INT PRIMARY KEY AUTO_INCREMENT, "
                 "   `name` VARCHAR(255) NOT NULL, "
                 "   `is_artifact_type` TINYINT(1) NOT NULL, "
                 "   `input_type` TEXT, "
                 "   `output_type` TEXT"
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `TypeProperty` ( "
                 "   `type_id` INT NOT NULL, "
                 "   `name` VARCHAR(255) NOT NULL, "
                 "   `data_type` INT NULL, "
                 " PRIMARY KEY (`type_id`, `name`)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `Artifact` ( "
                 "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
                 "   `type_id` INT NOT NULL, "
                 "   `uri` TEXT "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `ArtifactProperty` ( "
                 "   `artifact_id` INT NOT NULL, "
                 "   `name` VARCHAR(255) NOT NULL, "
                 "   `is_custom_property` TINYINT(1) NOT NULL, "
                 "   `int_value` INT, "
                 "   `double_value` DOUBLE, "
                 "   `string_value` TEXT, "
                 " PRIMARY KEY (`artifact_id`, `name`, `is_custom_property`)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `Execution` ( "
                 "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
                 "   `type_id` INT NOT NULL "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `ExecutionProperty` ( "
                 "   `execution_id` INT NOT NULL, "
                 "   `name` VARCHAR(255) NOT NULL, "
                 "   `is_custom_property` TINYINT(1) NOT NULL, "
                 "   `int_value` INT, "
                 "   `double_value` DOUBLE, "
                 "   `string_value` TEXT, "
                 " PRIMARY KEY (`execution_id`, `name`, `is_custom_property`)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `Event` ( "
                 "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
                 "   `artifact_id` INT NOT NULL, "
                 "   `execution_id` INT NOT NULL, "
                 "   `type` INT NOT NULL, "
                 "   `milliseconds_since_epoch` BIGINT "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `EventPath` ( "
                 "   `event_id` INT NOT NULL, "
                 "   `is_index_step` TINYINT(1) NOT NULL, "
                 "   `step_index` INT, "
                 "   `step_key` TEXT "
                 " ); "
        }
        # check the new table has 1 row
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `MLMDEnv`; "
        }
      }
      # downgrade queries from version 2, drop all ContextTypes and rename the
      # the `type_kind` back to `is_artifact_type` column.
      downgrade_queries { query: " DELETE FROM `Type` WHERE `type_kind` = 2; " }
      downgrade_queries {
        query: " ALTER TABLE `Type` CHANGE COLUMN "
               " `type_kind` `is_artifact_type` TINYINT(1) NOT NULL; "
      }
      # check the tables are deleted properly
      downgrade_verification {
        # populate the `Type` table with context types.
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`name`, `type_kind`, `input_type`, `output_type`) "
                 " VALUES ('execution_type', 0, 'input', 'output'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`name`, `type_kind`, `input_type`, `output_type`) "
                 " VALUES ('artifact_type', 1, 'input', 'output'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`name`, `type_kind`, `input_type`, `output_type`) "
                 " VALUES ('context_type', 2, 'input', 'output'); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `Type` "
                 " WHERE `is_artifact_type` = 2; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` "
                 " WHERE `is_artifact_type` = 1 AND `name` = 'artifact_type'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` "
                 " WHERE `is_artifact_type` = 0 AND `name` = 'execution_type'; "
        }
      }
    }
  }
)pb",
R"pb(
  migration_schemes {
    key: 2
    value: {
      upgrade_queries {
        query: " ALTER TABLE `Type` CHANGE COLUMN "
               " `is_artifact_type` `type_kind` TINYINT(1) NOT NULL; "
      }
      upgrade_verification {
        # populate one ArtifactType and one ExecutionType.
        previous_version_setup_queries {
          query: " INSERT INTO `Type` (`name`, `is_artifact_type`) VALUES "
                 " ('artifact_type', 1); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`name`, `is_artifact_type`, `input_type`, `output_type`) "
                 " VALUES ('execution_type', 0, 'input', 'output'); "
        }
        # check after migration, the existing types are the same including id.
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` WHERE "
                 " `id` = 1 AND `type_kind` = 1 AND `name` = 'artifact_type'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` WHERE "
                 " `id` = 2 AND `type_kind` = 0 AND `name` = 'execution_type' "
                 " AND `input_type` = 'input' AND `output_type` = 'output'; "
        }
      }
      # downgrade queries from version 3
      downgrade_queries { query: " DROP TABLE IF EXISTS `Context`; " }
      downgrade_queries { query: " DROP TABLE IF EXISTS `ContextProperty`; " }
      # check the tables are deleted properly
      downgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`tables` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Context'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`tables` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'ContextProperty'; "
        }
      }
    }
  }
)pb",
R"pb(
  migration_schemes {
    key: 3
    value: {
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `Context` ( "
               "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
               "   `type_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   UNIQUE(`type_id`, `name`) "
               " ); "
      }
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ContextProperty` ( "
               "   `context_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `is_custom_property` TINYINT(1) NOT NULL, "
               "   `int_value` INT, "
               "   `double_value` DOUBLE, "
               "   `string_value` TEXT, "
               " PRIMARY KEY (`context_id`, `name`, `is_custom_property`)); "
      }
      upgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `id`, `type_id`, `name` FROM `Context` "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `context_id`, `name`, `is_custom_property`, "
                 "          `int_value`, `double_value`, `string_value` "
                 "    FROM `ContextProperty` "
                 " ) as T2; "
        }
      }
      # downgrade queries from version 4
      downgrade_queries { query: " DROP TABLE IF EXISTS `Association`; " }
      downgrade_queries { query: " DROP TABLE IF EXISTS `Attribution`; " }
      # check the tables are deleted properly
      downgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`tables` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Association'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`tables` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Attribution'; "
        }
      }
    }
  }
)pb",
R"pb(
  migration_schemes {
    key: 4
    value: {
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `Association` ( "
               "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
               "   `context_id` INT NOT NULL, "
               "   `execution_id` INT NOT NULL, "
               "   UNIQUE(`context_id`, `execution_id`) "
               " ); "
      }
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `Attribution` ( "
               "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
               "   `context_id` INT NOT NULL, "
               "   `artifact_id` INT NOT NULL, "
               "   UNIQUE(`context_id`, `artifact_id`) "
               " ); "
      }
      upgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `id`, `context_id`, `execution_id` "
                 "   FROM `Association` "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `id`, `context_id`, `artifact_id` "
                 "   FROM `Attribution` "
                 " ) as T1; "
        }
      }
      # downgrade queries from version 5
      downgrade_queries {
        query: " ALTER TABLE `Artifact` "
               " DROP INDEX UniqueArtifactTypeName; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Artifact` "
               " DROP COLUMN `state`, "
               " DROP COLUMN `name`, "
               " DROP COLUMN `create_time_since_epoch`, "
               " DROP COLUMN `last_update_time_since_epoch`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Execution` "
               " DROP INDEX UniqueExecutionTypeName; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Execution` "
               " DROP COLUMN `last_known_state`, "
               " DROP COLUMN `name`, "
               " DROP COLUMN `create_time_since_epoch`, "
               " DROP COLUMN `last_update_time_since_epoch`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Context` "
               " DROP COLUMN `create_time_since_epoch`, "
               " DROP COLUMN `last_update_time_since_epoch`; "
      }
      # verify if the downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Artifact`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Artifact` "
                 " (`id`, `type_id`, `uri`, `state`, `name`, "
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 'uri1', 1, NULL, 0, 1); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Artifact` "
                 " (`id`, `type_id`, `uri`, `state`, `name`, "
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (2, 3, 'uri2', NULL, 'name2', 1, 0); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Execution`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Execution` "
                 " (`id`, `type_id`, `last_known_state`, `name`, "
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 1, NULL, 0, 1); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Context`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Context` "
                 " (`id`, `type_id`, `name`, "
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 'name1', 1, 0); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 2 FROM `Artifact`; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Artifact` "
                 "   WHERE `id` = 1 and `type_id` = 2 and `uri` = 'uri1' "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Artifact` "
                 "   WHERE `id` = 2 and `type_id` = 3 and `uri` = 'uri2' "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Execution`; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Execution` "
                 "   WHERE `id` = 1 and `type_id` = 2 "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Context` "
                 "   WHERE `id` = 1 and `type_id` = 2 "
                 " ) as T1; "
        }
      }
    }
  }
)pb",
R"pb(
  migration_schemes {
    key: 5
    value: {
      # upgrade Artifact table
      upgrade_queries {
        query: " ALTER TABLE `Artifact` ADD ( "
               "   `state` INT, "
               "   `name` VARCHAR(255), "
               "   `create_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
               "   `last_update_time_since_epoch` BIGINT NOT NULL DEFAULT 0 "
               " ), "
               " ADD CONSTRAINT UniqueArtifactTypeName "
               " UNIQUE(`type_id`, `name`); "
      }
      # upgrade Execution table
      upgrade_queries {
        query: " ALTER TABLE `Execution` ADD ( "
               "   `last_known_state` INT, "
               "   `name` VARCHAR(255), "
               "   `create_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
               "   `last_update_time_since_epoch` BIGINT NOT NULL DEFAULT 0 "
               " ), "
               " ADD CONSTRAINT UniqueExecutionTypeName "
               " UNIQUE(`type_id`, `name`); "
      }
      # upgrade Context table
      upgrade_queries {
        query: " ALTER TABLE `Context` ADD ( "
               "   `create_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
               "   `last_update_time_since_epoch` BIGINT NOT NULL DEFAULT 0 "
               " ) "
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Artifact`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Artifact` "
                 " (`id`, `type_id`, `uri`) VALUES (1, 2, 'uri1'); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Execution`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Execution` "
                 " (`id`, `type_id`) VALUES (1, 3); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS `Context` ( "
                 "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
                 "   `type_id` INT NOT NULL, "
                 "   `name` VARCHAR(255) NOT NULL, "
                 "   UNIQUE(`type_id`, `name`) "
                 " ); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Context` "
                 " (`id`, `type_id`, `name`) VALUES (1, 2, 'name1'); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT `id`, `type_id`, `uri`, `state`, `name`, "
                 "          `create_time_since_epoch`, "
                 "          `last_update_time_since_epoch` "
                 "   FROM `Artifact` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND `uri` = 'uri1' AND "
                 "         `create_time_since_epoch` = 0 AND "
                 "         `last_update_time_since_epoch` = 0 "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT `id`, `type_id`, `last_known_state`, `name`, "
                 "          `create_time_since_epoch`, "
                 "          `last_update_time_since_epoch` "
                 "   FROM `Execution` "
                 "   WHERE `id` = 1 AND `type_id` = 3 AND "
                 "         `create_time_since_epoch` = 0 AND "
                 "         `last_update_time_since_epoch` = 0 "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT `id`, `type_id`, `name`, "
                 "          `create_time_since_epoch`, "
                 "          `last_update_time_since_epoch` "
                 "   FROM `Context` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND `name` = 'name1' AND "
                 "         `create_time_since_epoch` = 0 AND "
                 "         `last_update_time_since_epoch` = 0 "
                 " ) as T1; "
        }
      }
    }
  }
)pb");

}  // namespace

// The `MetadataSourceQueryConfig` protobuf messages are merged to the query
// config with `MergeFrom`.
// Note: Singular fields overwrite the `kBaseQueryConfig` message. Repeated
// fields by default are concatenated to it and should be used with caution.
MetadataSourceQueryConfig GetMySqlMetadataSourceQueryConfig() {
  MetadataSourceQueryConfig config;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(kBaseQueryConfig,
                                                          &config));
  MetadataSourceQueryConfig mysql_config;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      kMySQLMetadataSourceQueryConfig, &mysql_config));
  config.MergeFrom(mysql_config);
  return config;
}

MetadataSourceQueryConfig GetSqliteMetadataSourceQueryConfig() {
  MetadataSourceQueryConfig config;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(kBaseQueryConfig,
                                                          &config));
  MetadataSourceQueryConfig sqlite_config;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      kSQLiteMetadataSourceQueryConfig, &sqlite_config));
  config.MergeFrom(sqlite_config);
  return config;
}


}  // namespace util
}  // namespace ml_metadata
