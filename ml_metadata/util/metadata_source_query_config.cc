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

#include <glog/logging.h>
#include "google/protobuf/text_format.h"
#include "absl/strings/str_cat.h"
#include "ml_metadata/proto/metadata_source.pb.h"

namespace ml_metadata {
namespace util {
namespace {

// clang-format off

// A set of common template queries used by the MetadataAccessObject for SQLite
// based MetadataSource.
// no-lint to support vc (C2026) 16380 max length for char[].
// TODO(b/257370493) Use ALTER TABLE instead of copying most of the data inside
// a datastore as current approach for schema upgrade/downgrade.
const std::string kBaseQueryConfig = absl::StrCat(  // NOLINT
R"pb(
  schema_version: 10
  drop_type_table { query: " DROP TABLE IF EXISTS `Type`; " }
  create_type_table {
    query: " CREATE TABLE IF NOT EXISTS `Type` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `version` VARCHAR(255), "
           "   `type_kind` TINYINT(1) NOT NULL, "
           "   `description` TEXT, "
           "   `input_type` TEXT, "
           "   `output_type` TEXT, "
           "   `external_id` VARCHAR(255) UNIQUE"
           " ); "
  }
  check_type_table {
    query: " SELECT `id`, `name`, `version`, `type_kind`, `description`, "
           "        `input_type`, `output_type` "
           " FROM `Type` LIMIT 1; "
  }
  insert_artifact_type {
    query: " INSERT INTO `Type`( "
           "   `name`, `type_kind`, `version`, `description`, `external_id` "
           ") VALUES($0, 1, $1, $2, $3);"
    parameter_num: 4
  }
  insert_execution_type {
    query: " INSERT INTO `Type`( "
           "   `name`, `type_kind`, `version`, `description`, "
           "   `input_type`, `output_type`, `external_id`  "
           ") VALUES($0, 0, $1, $2, $3, $4, $5);"
    parameter_num: 6
  }
  insert_context_type {
    query: " INSERT INTO `Type`( "
           "   `name`, `type_kind`, `version`, `description`, `external_id` "
           ") VALUES($0, 2, $1, $2, $3);"
    parameter_num: 4
  }
  select_types_by_id {
    query: " SELECT `id`, `name`, `version`, `description`, `external_id` "
           " FROM `Type` "
           " WHERE id IN ($0) and type_kind = $1; "
    parameter_num: 2
  }
  select_type_by_id {
    query: " SELECT `id`, `name`, `version`, `description`, `external_id`, "
           "        `input_type`, `output_type` FROM `Type` "
           " WHERE id = $0 and type_kind = $1; "
    parameter_num: 2
  }
  select_type_by_name {
    query: " SELECT `id`, `name`, `version`, `description`, `external_id`, "
           "        `input_type`, `output_type` FROM `Type` "
           " WHERE name = $0 AND version IS NULL AND type_kind = $1; "
    parameter_num: 2
  }
  select_type_by_name_and_version {
    query: " SELECT `id`, `name`, `version`, `description`, `external_id`, "
           "        `input_type`, `output_type` FROM `Type` "
           " WHERE name = $0 AND version = $1 AND type_kind = $2; "
    parameter_num: 3
  }
  select_types_by_external_ids {
    query: " SELECT `id`, `name`, `version`, `description`, `external_id` "
           " FROM `Type` "
           " WHERE external_id IN ($0) and type_kind = $1; "
    parameter_num: 2
  }
  select_types_by_names {
    query: " SELECT `id`, `name`, `version`, `description`, "
           "        `input_type`, `output_type` FROM `Type` "
           " WHERE name IN ($0) AND version IS NULL AND type_kind = $1; "
    parameter_num: 2
  }
  select_types_by_names_and_versions {
    query: " SELECT `id`, `name`, `version`, `description`, "
           "        `input_type`, `output_type` FROM `Type` "
           " WHERE (name, version) IN ($0) AND type_kind = $1; "
    parameter_num: 2
  }
  select_all_types {
    query: " SELECT `id`, `name`, `version`, `description`, "
           "        `input_type`, `output_type` FROM `Type` "
           " WHERE type_kind = $0; "
    parameter_num: 1
  }
  update_type {
    query: " UPDATE `Type` "
           " SET `external_id` = $1 "
           " WHERE id = $0;"
    parameter_num: 2
  }
  drop_parent_type_table { query: " DROP TABLE IF EXISTS `ParentType`; " }
  create_parent_type_table {
    query: " CREATE TABLE IF NOT EXISTS `ParentType` ( "
           "   `type_id` INT NOT NULL, "
           "   `parent_type_id` INT NOT NULL, "
           " PRIMARY KEY (`type_id`, `parent_type_id`)); "
  }
  check_parent_type_table {
    query: " SELECT `type_id`, `parent_type_id` "
           " FROM `ParentType` LIMIT 1; "
  }
  insert_parent_type {
    query: " INSERT INTO `ParentType`(`type_id`, `parent_type_id`) "
           " VALUES($0, $1);"
    parameter_num: 2
  }
  delete_parent_type {
    query: " DELETE FROM `ParentType` "
           " WHERE `type_id` = $0 AND `parent_type_id` = $1;"
    parameter_num: 2
  }
  select_parent_type_by_type_id {
    query: " SELECT `type_id`, `parent_type_id` "
           " FROM `ParentType` WHERE `type_id` IN ($0); "
    parameter_num: 1
  }
  select_parent_contexts_by_context_ids {
    query: " SELECT `context_id`, `parent_context_id` From `ParentContext` "
           " WHERE `context_id` IN ($0); "
    parameter_num: 1
  }
  select_parent_contexts_by_parent_context_ids {
    query: " SELECT `context_id`, `parent_context_id` From `ParentContext` "
           " WHERE `parent_context_id` IN ($0); "
    parameter_num: 1
  }
  drop_type_property_table {
    query: " DROP TABLE IF EXISTS `TypeProperty`; "
  }
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
  select_properties_by_type_id {
    query: " SELECT `type_id`, `name` as `key`, `data_type` as `value` "
           " from `TypeProperty` WHERE `type_id` IN ($0); "
    parameter_num: 1
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
           "   `external_id` VARCHAR(255) UNIQUE, "
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
           "   `type_id`, `uri`, `state`, `name`, `external_id`, "
           "   `create_time_since_epoch`, `last_update_time_since_epoch` "
           ") VALUES($0, $1, $2, $3, $4, $5, $6);"
    parameter_num: 7
  }
  select_artifact_by_id {
    query: " SELECT A.id, A.type_id, A.uri, A.state, A.name, "
           "        A.external_id, A.create_time_since_epoch, "
           "        A.last_update_time_since_epoch, "
           "        T.name AS `type`, T.version AS type_version, "
           "        T.description AS type_description, "
           "        T.external_id AS type_external_id "
           " FROM `Artifact` AS A "
           " LEFT JOIN `Type` AS T "
           "   ON (T.id = A.type_id) "
           " WHERE A.id IN ($0); "
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
  select_artifacts_by_external_ids {
    query: " SELECT `id` from `Artifact` WHERE `external_id` IN ($0); "
    parameter_num: 1
  }
  update_artifact {
    query: " UPDATE `Artifact` "
           " SET `type_id` = $1, `uri` = $2, `state` = $3, `external_id` = $4, "
           "     `last_update_time_since_epoch` = $5 "
           " WHERE id = $0;"
    parameter_num: 6
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
           "   `byte_value` BLOB, "
           "   `proto_value` BLOB, "
           "   `bool_value` BOOLEAN, "
           " PRIMARY KEY (`artifact_id`, `name`, `is_custom_property`)); "
  }
  check_artifact_property_table {
    query: " SELECT `artifact_id`, `name`, `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value`, `byte_value`, "
           "        `proto_value`, `bool_value` "
           " FROM `ArtifactProperty` LIMIT 1; "
  }
  insert_artifact_property {
    query: " INSERT INTO `ArtifactProperty`( "
           "   `artifact_id`, `name`, `is_custom_property`, `$0` "
           ") VALUES($1, $2, $3, $4);"
    parameter_num: 5
  }
  select_artifact_property_by_artifact_id {
    query: " SELECT `artifact_id` as `id`, `name` as `key`, "
           "        `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value`, `proto_value`,"
           "        `bool_value` "
           " from `ArtifactProperty` "
           " WHERE `artifact_id` IN ($0); "
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
  delete_artifacts_by_id {
    query: "DELETE FROM `Artifact` WHERE `id` IN ($0); "
    parameter_num: 1
  }
  delete_artifacts_properties_by_artifacts_id {
    query: "DELETE FROM `ArtifactProperty` WHERE `artifact_id` IN ($0); "
    parameter_num: 1
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
           "   `external_id` VARCHAR(255) UNIQUE, "
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
           "   `type_id`, `last_known_state`, `name`, `external_id`, "
           "   `create_time_since_epoch`, `last_update_time_since_epoch` "
           ") VALUES($0, $1, $2, $3, $4, $5);"
    parameter_num: 6
  }
  select_execution_by_id {
    query: " SELECT E.id, E.type_id, E.last_known_state, E.name, "
          "         E.external_id, E.create_time_since_epoch, "
          "         E.last_update_time_since_epoch, "
          "         T.name AS `type`, T.version AS type_version, "
          "         T.description AS type_description, "
          "         T.external_id AS type_external_id "
          " FROM `Execution` AS E "
          " LEFT JOIN `Type` AS T "
          "   ON (T.id = E.type_id) "
          " WHERE E.id IN ($0); "
    parameter_num: 1
  }
  select_execution_by_type_id_and_name {
    query: " SELECT `id` from `Execution` WHERE `type_id` = $0 and `name` = $1;"
    parameter_num: 2
  }
  select_executions_by_type_id {
    query: " SELECT `id` from `Execution` WHERE `type_id` = $0; "
    parameter_num: 1
  }
  select_executions_by_external_ids {
    query: " SELECT `id` from `Execution` WHERE `external_id` IN ($0);"
    parameter_num: 1
  }
  update_execution {
    query: " UPDATE `Execution` "
           " SET `type_id` = $1, `last_known_state` = $2, "
           "     `external_id` = $3, "
           "     `last_update_time_since_epoch` = $4 "
           " WHERE id = $0;"
    parameter_num: 5
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
           "   `byte_value` BLOB, "
           "   `proto_value` BLOB, "
           "   `bool_value` BOOLEAN, "
           " PRIMARY KEY (`execution_id`, `name`, `is_custom_property`)); "
  }
  check_execution_property_table {
    query: " SELECT `execution_id`, `name`, `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value`, `byte_value`, "
           "        `proto_value`, `bool_value` "
           " FROM `ExecutionProperty` LIMIT 1; "
  }
  insert_execution_property {
    query: " INSERT INTO `ExecutionProperty`( "
           "   `execution_id`, `name`, `is_custom_property`, `$0` "
           ") VALUES($1, $2, $3, $4);"
    parameter_num: 5
  }
  select_execution_property_by_execution_id {
    query: " SELECT `execution_id` as `id`, `name` as `key`, "
           "        `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value`, `proto_value`,"
           "        `bool_value` "
           " from `ExecutionProperty` "
           " WHERE `execution_id` IN ($0); "
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
  delete_executions_by_id {
    query: "DELETE FROM `Execution` WHERE `id` IN ($0); "
    parameter_num: 1
  }
  delete_executions_properties_by_executions_id {
    query: "DELETE FROM `ExecutionProperty` WHERE `execution_id` IN ($0); "
    parameter_num: 1
  }
)pb",
R"pb(
  drop_context_table { query: " DROP TABLE IF EXISTS `Context`; " }
  create_context_table {
    query: " CREATE TABLE IF NOT EXISTS `Context` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `external_id` VARCHAR(255) UNIQUE, "
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
           "   `type_id`, `name`, `external_id`, "
           "   `create_time_since_epoch`, `last_update_time_since_epoch` "
           ") VALUES($0, $1, $2, $3, $4);"
    parameter_num: 5
  }
  select_context_by_id {
    query: " SELECT C.id, C.type_id, C.name, C.external_id, "
           "        C.create_time_since_epoch, C.last_update_time_since_epoch, "
           "        T.name AS `type`, T.version AS type_version, "
           "        T.description AS type_description, "
           "        T.external_id AS type_external_id "
           " FROM `Context` AS C "
           " LEFT JOIN `Type` AS T ON (T.id = C.type_id) "
           " WHERE C.id IN ($0); "
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
  select_contexts_by_external_ids {
    query: " SELECT `id` from `Context` WHERE `external_id` IN ($0); "
    parameter_num: 1
  }
  update_context {
    query: " UPDATE `Context` "
           " SET `type_id` = $1, `name` = $2, `external_id` = $3, "
           "     `last_update_time_since_epoch` = $4 "
           " WHERE id = $0;"
    parameter_num: 5
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
           "   `byte_value` BLOB, "
           "   `proto_value` BLOB, "
           "   `bool_value` BOOLEAN, "
           " PRIMARY KEY (`context_id`, `name`, `is_custom_property`)); "
  }
  check_context_property_table {
    query: " SELECT `context_id`, `name`, `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value`, `byte_value`, "
           "        `proto_value`, `bool_value` "
           " FROM `ContextProperty` LIMIT 1; "
  }
  insert_context_property {
    query: " INSERT INTO `ContextProperty`( "
           "   `context_id`, `name`, `is_custom_property`, `$0` "
           ") VALUES($1, $2, $3, $4);"
    parameter_num: 5
  }
  select_context_property_by_context_id {
    query: " SELECT `context_id` as `id`, `name` as `key`, "
           "        `is_custom_property`, "
           "        `int_value`, `double_value`, `string_value`, `proto_value`,"
           "        `bool_value` "
           " from `ContextProperty` "
           " WHERE `context_id` IN ($0); "
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
  drop_parent_context_table {
    query: " DROP TABLE IF EXISTS `ParentContext`;"
  }
  create_parent_context_table {
    query: " CREATE TABLE IF NOT EXISTS `ParentContext` ( "
           "   `context_id` INT NOT NULL, "
           "   `parent_context_id` INT NOT NULL, "
           " PRIMARY KEY (`context_id`, `parent_context_id`)); "
  }
  check_parent_context_table {
    query: " SELECT `context_id`, `parent_context_id` "
           " FROM `ParentContext` LIMIT 1; "
  }
  insert_parent_context {
    query: " INSERT INTO `ParentContext`( "
           "   `context_id`, `parent_context_id` "
           ") VALUES($0, $1);"
    parameter_num: 2
  }
  select_parent_context_by_context_id {
    query: " SELECT `context_id`, `parent_context_id` From `ParentContext` "
           " WHERE `context_id` = $0; "
    parameter_num: 1
  }
  select_parent_context_by_parent_context_id {
    query: " SELECT `context_id`, `parent_context_id` From `ParentContext` "
           " WHERE `parent_context_id` = $0; "
    parameter_num: 1
  }
  delete_contexts_by_id {
    query: "DELETE FROM `Context` WHERE `id` IN ($0); "
    parameter_num: 1
  }
  delete_contexts_properties_by_contexts_id {
    query: "DELETE FROM `ContextProperty` WHERE `context_id` IN ($0); "
    parameter_num: 1
  }
  delete_parent_contexts_by_parent_ids {
    query: "DELETE FROM `ParentContext` WHERE `parent_context_id` IN ($0); "
    parameter_num: 1
  }
  delete_parent_contexts_by_child_ids {
    query: "DELETE FROM `ParentContext` WHERE `context_id` IN ($0); "
    parameter_num: 1
  }
  delete_parent_contexts_by_parent_id_and_child_ids {
    query: "DELETE FROM `ParentContext` "
           "WHERE `parent_context_id` = $0 AND `context_id` IN ($1); "
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
           "   `milliseconds_since_epoch` INT, "
           "   UNIQUE(`artifact_id`, `execution_id`, `type`) "
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
  delete_events_by_artifacts_id {
    query: "DELETE FROM `Event` WHERE `artifact_id` IN ($0); "
    parameter_num: 1
  }
  delete_events_by_executions_id {
    query: "DELETE FROM `Event` WHERE `execution_id` IN ($0); "
    parameter_num: 1
  }
  delete_event_paths {
    query: "DELETE FROM `EventPath` WHERE `event_id` NOT IN "
           " (SELECT `id` FROM `Event`); "
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
           " WHERE `context_id` IN ($0); "
    parameter_num: 1
  }
  select_associations_by_execution_ids {
    query: " SELECT `id`, `context_id`, `execution_id` "
           " FROM `Association` "
           " WHERE `execution_id` IN ($0); "
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
  select_attributions_by_artifact_ids {
    query: " SELECT `id`, `context_id`, `artifact_id` "
           " FROM `Attribution` "
           " WHERE `artifact_id` IN ($0); "
    parameter_num: 1
  }
  drop_mlmd_env_table { query: " DROP TABLE IF EXISTS `MLMDEnv`; " }
  create_mlmd_env_table {
    query: " CREATE TABLE IF NOT EXISTS `MLMDEnv` ( "
           "   `schema_version` INTEGER PRIMARY KEY "
           " ); "
  }
  check_mlmd_env_table_existence {
    query: " SELECT ("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'mlmdenv'"
           "      AND column_name IN ('schema_version')"
           "   ) = 1"
           " ) AS table_exists;"
  }
  check_mlmd_env_table {
    query: " SELECT `schema_version` FROM `MLMDEnv`; "
  }
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
  delete_associations_by_contexts_id {
    query: "DELETE FROM `Association` WHERE `context_id` IN ($0); "
    parameter_num: 1
  }
  delete_associations_by_executions_id {
    query: "DELETE FROM `Association` WHERE `execution_id` IN ($0); "
    parameter_num: 1
  }
  delete_attributions_by_contexts_id {
    query: "DELETE FROM `Attribution` WHERE `context_id` IN ($0); "
    parameter_num: 1
  }
  delete_attributions_by_artifacts_id {
    query: "DELETE FROM `Attribution` WHERE `artifact_id` IN ($0); "
    parameter_num: 1
  }
)pb");

// no-lint to support vc (C2026) 16380 max length for char[].
const std::string kSQLiteMetadataSourceQueryConfig = absl::StrCat(  // NOLINT
R"pb(
  metadata_source_type: SQLITE_METADATA_SOURCE
  check_mlmd_env_table_existence {
    query: " SELECT ("
           "   SELECT COUNT(*)"
           "   FROM   sqlite_master"
           "   WHERE  type='table'"
           "      AND name = 'mlmdenv'"
           "      AND sql LIKE '%schema_version%'"
           "   ) = 1"
           " ) AS table_exists;"
  }
  # secondary indices in the current schema.
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_artifact_uri` "
           " ON `Artifact`(`uri`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "   `idx_artifact_create_time_since_epoch` "
           " ON `Artifact`(`create_time_since_epoch`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "   `idx_artifact_last_update_time_since_epoch` "
           " ON `Artifact`(`last_update_time_since_epoch`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_event_execution_id` "
           " ON `Event`(`execution_id`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_parentcontext_parent_context_id` "
           " ON `ParentContext`(`parent_context_id`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_type_name` "
           " ON `Type`(`name`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "   `idx_execution_create_time_since_epoch` "
           " ON `Execution`(`create_time_since_epoch`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "   `idx_execution_last_update_time_since_epoch` "
           " ON `Execution`(`last_update_time_since_epoch`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "   `idx_context_create_time_since_epoch` "
           " ON `Context`(`create_time_since_epoch`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "   `idx_context_last_update_time_since_epoch` "
           " ON `Context`(`last_update_time_since_epoch`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_eventpath_event_id` "
           " ON `EventPath`(`event_id`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_artifact_property_int` "
           " ON `ArtifactProperty`(`name`, `is_custom_property`, `int_value`) "
           " WHERE `int_value` IS NOT NULL; "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_artifact_property_double` "
           " ON `ArtifactProperty`(`name`, `is_custom_property`, `double_value`) "
           " WHERE `double_value` IS NOT NULL; "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_artifact_property_string` "
           " ON `ArtifactProperty`(`name`, `is_custom_property`, `string_value`) "
           " WHERE `string_value` IS NOT NULL; "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_execution_property_int` "
           " ON `ExecutionProperty`(`name`, `is_custom_property`, `int_value`) "
           " WHERE `int_value` IS NOT NULL; "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_execution_property_double` "
           " ON `ExecutionProperty`(`name`, `is_custom_property`, `double_value`) "
           " WHERE `double_value` IS NOT NULL; "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_execution_property_string` "
           " ON `ExecutionProperty`(`name`, `is_custom_property`, `string_value`) "
           " WHERE `string_value` IS NOT NULL; "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_context_property_int` "
           " ON `ContextProperty`(`name`, `is_custom_property`, `int_value`) "
           " WHERE `int_value` IS NOT NULL; "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_context_property_double` "
           " ON `ContextProperty`(`name`, `is_custom_property`, `double_value`) "
           " WHERE `double_value` IS NOT NULL; "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_context_property_string` "
           " ON `ContextProperty`(`name`, `is_custom_property`, `string_value`) "
           " WHERE `string_value` IS NOT NULL; "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_type_external_id` "
           " ON `Type`(`external_id`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_artifact_external_id` "
           " ON `Artifact`(`external_id`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_execution_external_id` "
           " ON `Execution`(`external_id`); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS `idx_context_external_id` "
           " ON `Context`(`external_id`); "
  }
)pb",
R"pb(
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
  # From 0.13.2 to v1, it creates a new MLMDEnv table to track
  # schema_version.
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
      # downgrade queries from version 2, drop all ContextTypes and rename
      # the `type_kind` back to `is_artifact_type` column.
      downgrade_queries {
        query: " DELETE FROM `Type` WHERE `type_kind` = 2; "
      }
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
      downgrade_queries {
        query: " ALTER TABLE `TypeTemp` rename to `Type`; "
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
  # In v2, to support context type, and we renamed `is_artifact_type` column
  # in `Type` table.
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
      upgrade_queries {
        query: " ALTER TABLE `TypeTemp` rename to `Type`; "
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
        # check after migration, the existing types are the same including
        # id.
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
      downgrade_queries {
        query: " DROP TABLE IF EXISTS `ContextProperty`; "
      }
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
  # attribution, we added two tables `Association` and `Attribution` and
  # made no change to other existing records.
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
      # downgrade queries from version 6
      downgrade_queries { query: " DROP TABLE `ParentType`; " }
      downgrade_queries { query: " DROP TABLE `ParentContext`; " }
      downgrade_queries {
        query: " CREATE TABLE `TypeTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `type_kind` TINYINT(1) NOT NULL, "
               "   `input_type` TEXT, "
               "   `output_type` TEXT"
               " ); "
      }
      downgrade_queries {
        query: " INSERT INTO `TypeTemp` "
               " SELECT `id`, `name`, `type_kind`, `input_type`, `output_type`"
               " FROM `Type`; "
      }
      downgrade_queries { query: " DROP TABLE `Type`; " }
      downgrade_queries {
        query: " ALTER TABLE `TypeTemp` RENAME TO `Type`; "
      }
      downgrade_queries { query: " DROP INDEX `idx_artifact_uri`; " }
      downgrade_queries {
        query: " DROP INDEX`idx_artifact_create_time_since_epoch`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_artifact_last_update_time_since_epoch`; "
      }
      downgrade_queries { query: " DROP INDEX `idx_event_artifact_id`; " }
      downgrade_queries { query: " DROP INDEX `idx_event_execution_id`; " }
      downgrade_queries {
        query: " DROP INDEX `idx_execution_create_time_since_epoch`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_execution_last_update_time_since_epoch`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_context_create_time_since_epoch`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_context_last_update_time_since_epoch`; "
      }
      # verify if the downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Type`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`id`, `name`, `version`, `type_kind`, "
                 "  `description`, `input_type`, `output_type`) "
                 " VALUES (1, 't1', 'v1', 1, 'desc1', 'input1', 'output1'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`id`, `name`, `version`, `type_kind`, "
                 "  `description`, `input_type`, `output_type`) "
                 " VALUES (2, 't2', 'v2', 2, 'desc2', 'input2', 'output2'); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 2 FROM `Type`; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Type` "
                 "   WHERE `id` = 1 AND `name` = 't1' AND type_kind = 1 "
                 "   AND `input_type` = 'input1' AND `output_type` = 'output1'"
                 " ); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Artifact' "
                 "       AND `name` LIKE 'idx_artifact_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Event' "
                 "       AND `name` LIKE 'idx_event_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `tbl_name` = 'ParentType'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `tbl_name` = 'ParentContext'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'ParentContext' "
                 "       AND `name` LIKE 'idx_parentcontext_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Type' "
                 "       AND `name` LIKE 'idx_type_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Execution' "
                 "       AND `name` LIKE 'idx_execution_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Context' "
                 "       AND `name` LIKE 'idx_context_%'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v6, to support parental type and parental context, we added two
  # tables `ParentType` and `ParentContext`. In addition, we added `version`
  # and `description` in the `Type` table for improving type registrations.
  # We introduce indices on Type.name, Artifact.uri, Event's artifact_id and
  # execution_id, and create_time_since_epoch, last_update_time_since_epoch
  # for all nodes.
  migration_schemes {
    key: 6
    value: {
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ParentType` ( "
               "   `type_id` INT NOT NULL, "
               "   `parent_type_id` INT NOT NULL, "
               " PRIMARY KEY (`type_id`, `parent_type_id`)); "
      }
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ParentContext` ( "
               "   `context_id` INT NOT NULL, "
               "   `parent_context_id` INT NOT NULL, "
               " PRIMARY KEY (`context_id`, `parent_context_id`)); "
      }
      # upgrade Type table
      upgrade_queries {
        query: " CREATE TABLE `TypeTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `version` VARCHAR(255), "
               "   `type_kind` TINYINT(1) NOT NULL, "
               "   `description` TEXT, "
               "   `input_type` TEXT, "
               "   `output_type` TEXT"
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO `TypeTemp` "
               " (`id`, `name`, `type_kind`, `input_type`, `output_type`) "
               " SELECT * FROM `Type`; "
      }
      upgrade_queries { query: " DROP TABLE `Type`; " }
      upgrade_queries {
        query: " ALTER TABLE `TypeTemp` rename to `Type`; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_artifact_uri` "
               " ON `Artifact`(`uri`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_artifact_create_time_since_epoch` "
               " ON `Artifact`(`create_time_since_epoch`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_artifact_last_update_time_since_epoch` "
               " ON `Artifact`(`last_update_time_since_epoch`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_event_artifact_id` "
               " ON `Event`(`artifact_id`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_event_execution_id` "
               " ON `Event`(`execution_id`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               " `idx_parentcontext_parent_context_id` "
               " ON `ParentContext`(`parent_context_id`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_type_name` "
               " ON `Type`(`name`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_execution_create_time_since_epoch` "
               " ON `Execution`(`create_time_since_epoch`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_execution_last_update_time_since_epoch` "
               " ON `Execution`(`last_update_time_since_epoch`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_context_create_time_since_epoch` "
               " ON `Context`(`create_time_since_epoch`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_context_last_update_time_since_epoch` "
               " ON `Context`(`last_update_time_since_epoch`); "
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        # check existing rows in previous Type table are migrated properly.
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` WHERE "
                 " `id` = 1 AND `type_kind` = 1 AND `name` = 'artifact_type'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` WHERE "
                 " `id` = 2 AND `type_kind` = 0 AND `name` = 'execution_type' "
                 " AND `input_type` = 'input' AND `output_type` = 'output'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `type_id`, `parent_type_id` "
                 "   FROM `ParentType` "
                 " ); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `context_id`, `parent_context_id` "
                 "   FROM `ParentContext` "
                 " ); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Artifact' "
                 "       AND `name` = 'idx_artifact_uri'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Artifact' "
                 "       AND `name` = 'idx_artifact_create_time_since_epoch'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Artifact' AND "
                 "       `name` = 'idx_artifact_last_update_time_since_epoch'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Event' "
                 "       AND `name` = 'idx_event_artifact_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Event' "
                 "       AND `name` = 'idx_event_execution_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'ParentContext' "
                 "       AND `name` = 'idx_parentcontext_parent_context_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Type' "
                 "       AND `name` = 'idx_type_name'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Execution' "
                 "       AND `name` = 'idx_execution_create_time_since_epoch';"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Execution' AND "
                 "       `name` = 'idx_execution_last_update_time_since_epoch';"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Context' "
                 "       AND `name` = 'idx_context_create_time_since_epoch';"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Context' AND "
                 "       `name` = 'idx_context_last_update_time_since_epoch';"
        }
      }
      # downgrade queries from version 7
      downgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ArtifactPropertyTemp` ( "
               "   `artifact_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `is_custom_property` TINYINT(1) NOT NULL, "
               "   `int_value` INT, "
               "   `double_value` DOUBLE, "
               "   `string_value` TEXT, "
               " PRIMARY KEY (`artifact_id`, `name`, `is_custom_property`)); "
      }
      downgrade_queries {
        query: " INSERT INTO `ArtifactPropertyTemp`  "
               " SELECT `artifact_id`, `name`,  `is_custom_property`, "
               "        `int_value`, `double_value`, `string_value` "
               " FROM `ArtifactProperty`; "
      }
      downgrade_queries { query: " DROP TABLE `ArtifactProperty`; " }
      downgrade_queries {
        query: " ALTER TABLE `ArtifactPropertyTemp` "
               "  RENAME TO `ArtifactProperty`; "
      }
      downgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ExecutionPropertyTemp` ( "
               "   `execution_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `is_custom_property` TINYINT(1) NOT NULL, "
               "   `int_value` INT, "
               "   `double_value` DOUBLE, "
               "   `string_value` TEXT, "
               " PRIMARY KEY (`execution_id`, `name`, `is_custom_property`)); "
      }
      downgrade_queries {
        query: " INSERT INTO `ExecutionPropertyTemp` "
               " SELECT `execution_id`, `name`,  `is_custom_property`, "
               "     `int_value`, `double_value`, `string_value` "
               " FROM `ExecutionProperty`; "
      }
      downgrade_queries { query: " DROP TABLE `ExecutionProperty`; " }
      downgrade_queries {
        query: " ALTER TABLE `ExecutionPropertyTemp` "
               "  RENAME TO `ExecutionProperty`; "
      }
      downgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ContextPropertyTemp` ( "
               "   `context_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `is_custom_property` TINYINT(1) NOT NULL, "
               "   `int_value` INT, "
               "   `double_value` DOUBLE, "
               "   `string_value` TEXT, "
               " PRIMARY KEY (`context_id`, `name`, `is_custom_property`)); "
      }
      downgrade_queries {
        query: " INSERT INTO `ContextPropertyTemp` "
               " SELECT `context_id`, `name`,  `is_custom_property`, "
               "        `int_value`, `double_value`, `string_value` "
               " FROM `ContextProperty`; "
      }
      downgrade_queries { query: " DROP TABLE `ContextProperty`; " }
      downgrade_queries {
        query: " ALTER TABLE `ContextPropertyTemp` "
               "  RENAME TO `ContextProperty`; "
      }
      downgrade_queries { query: " DROP INDEX `idx_eventpath_event_id`; " }
      # verify if the downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries {
          query: "DELETE FROM `ArtifactProperty`;"
        }
        previous_version_setup_queries {
          query: "DELETE FROM `ExecutionProperty`;"
        }
        previous_version_setup_queries {
          query: "DELETE FROM `ContextProperty`;"
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ArtifactProperty` (`artifact_id`, "
                 "     `is_custom_property`, `name`, `string_value`) "
                 " VALUES (1, 0, 'p1', 'abc'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ExecutionProperty` (`execution_id`, "
                 "     `is_custom_property`, `name`, `int_value`) "
                 " VALUES (1, 1, 'p1', 1); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ContextProperty` (`context_id`, "
                 "     `is_custom_property`, `name`, `double_value`) "
                 " VALUES (1, 0, 'p1', 1.0); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        PRAGMA_TABLE_INFO('ArtifactProperty') "
                 " WHERE `name` = 'byte_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        PRAGMA_TABLE_INFO('ExecutionProperty') "
                 " WHERE `name` = 'byte_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        PRAGMA_TABLE_INFO('ContextProperty') "
                 " WHERE `name` = 'byte_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'EventPath' "
                 "       AND `name` LIKE 'idx_eventpath_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ArtifactProperty` "
                 " WHERE `artifact_id` = 1 AND `is_custom_property` = 0 AND "
                 "       `name` = 'p1' AND `string_value` = 'abc'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ExecutionProperty` "
                 " WHERE `execution_id` = 1 AND `is_custom_property` = 1 AND "
                 "        `name` = 'p1' AND `int_value` = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ContextProperty` "
                 " WHERE `context_id` = 1  AND `is_custom_property` = 0 AND "
                 "        `name` = 'p1' AND `double_value` = 1.0; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v7, we added byte_value for property tables for better storing binary
  # property values. In addition, we added index for `EventPath` to improve
  # Event reads.
  migration_schemes {
    key: 7
    value: {
      upgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               " ADD COLUMN `byte_value` BLOB; "
      }
      upgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               " ADD COLUMN `byte_value` BLOB; "
      }
      upgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               " ADD COLUMN `byte_value` BLOB; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_eventpath_event_id` "
               " ON `EventPath`(`event_id`); "
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        # check existing rows in previous Type table are migrated properly.
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ArtifactProperty` WHERE "
                 " `byte_value` IS NOT NULL; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ExecutionProperty` WHERE "
                 " `byte_value` IS NOT NULL; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ContextProperty` WHERE "
                 " `byte_value` IS NOT NULL; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'EventPath' AND "
                 "       `name` = 'idx_eventpath_event_id';"
        }
      }
      db_verification { total_num_indexes: 23 total_num_tables: 15 }
      # Downgrade from v8.
      downgrade_queries {
        query: " CREATE TABLE `EventTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `artifact_id` INT NOT NULL, "
               "   `execution_id` INT NOT NULL, "
               "   `type` INT NOT NULL, "
               "   `milliseconds_since_epoch` INT "
               " ); "
      }
      downgrade_queries {
        query: " INSERT INTO `EventTemp` "
               " (`id`, `artifact_id`, `execution_id`, `type`, "
               " `milliseconds_since_epoch`) "
               " SELECT * FROM `Event`; "
      }
      downgrade_queries { query: " DROP TABLE `Event`; " }
      downgrade_queries {
        query: " ALTER TABLE `EventTemp` RENAME TO `Event`; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_event_artifact_id` "
               " ON `Event`(`artifact_id`); "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_event_execution_id` "
               " ON `Event`(`execution_id`); "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_artifact_property_int`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_artifact_property_double`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_artifact_property_string`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_execution_property_int`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_execution_property_double`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_execution_property_string`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_context_property_int`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_context_property_double`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_context_property_string`; "
      }
      downgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Event`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Event` "
                 " (`id`, `artifact_id`, `execution_id`, `type`, "
                 " `milliseconds_since_epoch`) "
                 " VALUES (1, 1, 1, 1, 1); "
        }
        previous_version_setup_queries { query: "DELETE FROM `EventPath`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `EventPath` "
                 " (`event_id`, `is_index_step`, `step_index`, `step_key`) "
                 " VALUES (1, 1, 1, 'a'); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Event` "
                 " WHERE `artifact_id` = 1 AND `execution_id` = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `EventPath` "
                 " WHERE `event_id` = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Event' AND "
                 "       `name` = 'idx_event_artifact_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Event' AND "
                 "       `name` = 'idx_event_execution_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'EventPath' AND "
                 "       `name` = 'idx_eventpath_event_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'ArtifactProperty' "
                 "       AND `name` LIKE 'idx_artifact_property_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'ExecutionProperty' "
                 "       AND `name` LIKE 'idx_execution_property_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'ContextProperty' "
                 "       AND `name` LIKE 'idx_context_property_%'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v8, we added index for `ArtifactProperty`, `ExecutionProperty`,
  # `ContextProperty` to improve property queries on name, and unique
  # constraint on Event table for (`artifact_id`, `execution_id`, `type`).
  migration_schemes {
    key: 8
    value: {
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_artifact_property_int` "
               " ON `ArtifactProperty`(`name`, `is_custom_property`, "
               " `int_value`) "
               " WHERE `int_value` IS NOT NULL; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_artifact_property_double` "
               " ON `ArtifactProperty`(`name`, `is_custom_property`, "
               " `double_value`) "
               " WHERE `double_value` IS NOT NULL; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_artifact_property_string` "
               " ON `ArtifactProperty`(`name`, `is_custom_property`, "
               " `string_value`) "
               " WHERE `string_value` IS NOT NULL; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_execution_property_int` "
               " ON `ExecutionProperty`(`name`, `is_custom_property`, "
               " `int_value`) "
               " WHERE `int_value` IS NOT NULL; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_execution_property_double` "
               " ON `ExecutionProperty`(`name`, `is_custom_property`, "
               " `double_value`) "
               " WHERE `double_value` IS NOT NULL; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_execution_property_string` "
               " ON `ExecutionProperty`(`name`, `is_custom_property`, "
               " `string_value`) "
               " WHERE `string_value` IS NOT NULL; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_context_property_int` "
               " ON `ContextProperty`(`name`, `is_custom_property`, "
               " `int_value`) "
               " WHERE `int_value` IS NOT NULL; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_context_property_double` "
               " ON `ContextProperty`(`name`, `is_custom_property`, "
               " `double_value`) "
               " WHERE `double_value` IS NOT NULL; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_context_property_string` "
               " ON `ContextProperty`(`name`, `is_custom_property`, "
               " `string_value`) "
               " WHERE `string_value` IS NOT NULL; "
      }
      upgrade_queries {
        query: " CREATE TABLE `EventTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `artifact_id` INT NOT NULL, "
               "   `execution_id` INT NOT NULL, "
               "   `type` INT NOT NULL, "
               "   `milliseconds_since_epoch` INT, "
               "   UNIQUE(`artifact_id`, `execution_id`, `type`) "
               " ); "
      }
      upgrade_queries {
        query: " INSERT OR IGNORE INTO `EventTemp` "
               " (`id`, `artifact_id`, `execution_id`, `type`, "
               " `milliseconds_since_epoch`) "
               " SELECT * FROM `Event` ORDER BY `id` desc; "
      }
      upgrade_queries { query: " DROP TABLE `Event`; " }
      upgrade_queries {
        query: " ALTER TABLE `EventTemp` RENAME TO `Event`; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_event_execution_id` "
               " ON `Event`(`execution_id`); "
      }
      upgrade_queries {
        query: " DELETE FROM `EventPath` "
               "   WHERE event_id not in ( SELECT `id` from Event ) "
      }
      upgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Event`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Event` "
                 " (`id`, `artifact_id`, `execution_id`, `type`, "
                 " `milliseconds_since_epoch`) "
                 " VALUES (1, 1, 1, 1, 1); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Event` "
                 " (`id`, `artifact_id`, `execution_id`, `type`, "
                 " `milliseconds_since_epoch`) "
                 " VALUES (2, 1, 1, 1, 2); "
        }
        previous_version_setup_queries { query: "DELETE FROM `EventPath`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `EventPath` "
                 " (`event_id`, `is_index_step`, `step_index`, `step_key`) "
                 " VALUES (1, 1, 1, 'a'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `EventPath` "
                 " (`event_id`, `is_index_step`, `step_index`, `step_key`) "
                 " VALUES (2, 1, 1, 'b'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `EventPath` "
                 " (`event_id`, `is_index_step`, `step_index`, `step_key`) "
                 " VALUES (2, 1, 2, 'c'); "
        }
        # check event table unique constraint is applied and event path
        # records are deleted.
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Event` "
                 " WHERE `artifact_id` = 1 AND `execution_id` = 1 "
                 "     AND `type` = 1;"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Event` "
                 "   WHERE `id` = 2 AND `artifact_id` = 1 AND  "
                 "       `execution_id` = 1 AND `type` = 1 "
                 " ); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 2 FROM `EventPath` "
                 " WHERE `event_id` = 2; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `EventPath` "
                 " WHERE `event_id` = 2 AND `step_key` = 'b'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `EventPath` "
                 " WHERE `event_id` = 2 AND `step_key` = 'c'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `EventPath` "
                 " WHERE `event_id` = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 2 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Event'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Event' AND "
                 "       `name` = 'idx_event_artifact_id';"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'Event' AND "
                 "       `name` = 'idx_event_execution_id';"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'EventPath' AND "
                 "       `name` = 'idx_eventpath_event_id';"
        }
        # check indexes are added.
        post_migration_verification_queries {
          query: " SELECT count(*) = 3 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'ArtifactProperty' "
                 "       AND `name` LIKE 'idx_artifact_property_%';"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 3 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'ExecutionProperty' "
                 "       AND `name` LIKE 'idx_execution_property_%';"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 3 FROM `sqlite_master` "
                 " WHERE `type` = 'index' AND `tbl_name` = 'ContextProperty' "
                 "       AND `name` LIKE 'idx_context_property_%';"
        }
      }
      db_verification { total_num_indexes: 32 total_num_tables: 15 }
)pb",
R"pb(
  # downgrade queries from version 9
      downgrade_queries { query: " DROP INDEX `idx_type_external_id`; " }
      downgrade_queries {
        query: " DROP INDEX `idx_artifact_external_id`; "
      }
      downgrade_queries {
        query: " DROP INDEX `idx_execution_external_id`; "
      }
      downgrade_queries { query: " DROP INDEX `idx_context_external_id`; " }
      downgrade_queries {
        query: " CREATE TABLE `TypeTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `version` VARCHAR(255), "
               "   `type_kind` TINYINT(1) NOT NULL, "
               "   `description` TEXT, "
               "   `input_type` TEXT, "
               "   `output_type` TEXT"
               " ); "
      }
      downgrade_queries {
        query: " INSERT INTO `TypeTemp` "
               " SELECT `id`, `name`, `version`, `type_kind`, `description`,"
               "        `input_type`, `output_type` "
               " FROM `Type`; "
      }
      downgrade_queries { query: " DROP TABLE `Type`; " }
      downgrade_queries {
        query: " ALTER TABLE `TypeTemp` rename to `Type`; "
      }
      downgrade_queries {
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
      downgrade_queries {
        query: " INSERT INTO `ArtifactTemp` "
               " SELECT `id`, `type_id`, `uri`, `state`, `name`, "
               "        `create_time_since_epoch`, "
               "        `last_update_time_since_epoch` "
               "FROM `Artifact`; "
      }
      downgrade_queries { query: " DROP TABLE `Artifact`; " }
      downgrade_queries {
        query: " ALTER TABLE `ArtifactTemp` RENAME TO `Artifact`; "
      }
      downgrade_queries {
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
      downgrade_queries {
        query: " INSERT INTO `ExecutionTemp` "
               " SELECT `id`, `type_id`, `last_known_state`, `name`, "
               "        `create_time_since_epoch`, "
               "        `last_update_time_since_epoch` "
               " FROM `Execution`; "
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
               "   `create_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   `last_update_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   UNIQUE(`type_id`, `name`) "
               " ); "
      }
      downgrade_queries {
        query: " INSERT INTO `ContextTemp` "
               " SELECT `id`, `type_id`, `name`, "
               "        `create_time_since_epoch`, "
               "        `last_update_time_since_epoch` "
               " FROM `Context`; "
      }
      downgrade_queries { query: " DROP TABLE `Context`; " }
      downgrade_queries {
        query: " ALTER TABLE `ContextTemp` RENAME TO `Context`; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_artifact_uri` "
               " ON `Artifact`(`uri`); "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_artifact_create_time_since_epoch` "
               " ON `Artifact`(`create_time_since_epoch`); "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_artifact_last_update_time_since_epoch` "
               " ON `Artifact`(`last_update_time_since_epoch`); "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_type_name` "
               " ON `Type`(`name`); "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_execution_create_time_since_epoch` "
               " ON `Execution`(`create_time_since_epoch`); "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_execution_last_update_time_since_epoch` "
               " ON `Execution`(`last_update_time_since_epoch`); "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_context_create_time_since_epoch` "
               " ON `Context`(`create_time_since_epoch`); "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_context_last_update_time_since_epoch` "
               " ON `Context`(`last_update_time_since_epoch`); "
      }
      # verify if the downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Type`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`id`, `name`, `version`, `type_kind`, "
                 "  `description`, `input_type`, `output_type`, `external_id`) "
                 " VALUES (1, 't1', 'v1', 1, 'desc1', 'input1', 'output1', "
                 "           'type_1'); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Artifact`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Artifact` "
                 " (`id`, `type_id`, `uri`, `state`, `name`, `external_id`,"
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 'uri1', 1, NULL, 'artifact_1', 0, 1); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Execution`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Execution` "
                 " (`id`, `type_id`, `last_known_state`, `name`, `external_id`,"
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 1, NULL, 'execution_1', 0, 1); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Context`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Context` "
                 " (`id`, `type_id`, `name`, `external_id`,"
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 'name1', 'context_1', 1, 0); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type`; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Type` "
                 "   WHERE `id` = 1 AND `name` = 't1' AND type_kind = 1 "
                 "   AND `input_type` = 'input1' AND `output_type` = 'output1'"
                 " ); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Artifact`; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Artifact` "
                 "   WHERE `id` = 1 and `type_id` = 2 and `uri` = 'uri1' "
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
          query: " SELECT COUNT(*) = 1 FROM `Context`; "
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
  # In v9, to store the ids that come from the clients' system (like Vertex
  # Metadata), we added a new column `external_id` in the `Type` \
  # `Artifacrt` \ `Execution` \ `Context` tables. We introduce unique and
  # null-filtered indices on Type.external_id, Artifact.external_id,
  # Execution's external_id and Context's external_id.
  migration_schemes {
    key: 9
    value: {
      upgrade_queries {
        query: " CREATE TABLE `TypeTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `version` VARCHAR(255), "
               "   `type_kind` TINYINT(1) NOT NULL, "
               "   `description` TEXT, "
               "   `input_type` TEXT, "
               "   `output_type` TEXT, "
               "   `external_id` VARCHAR(255) UNIQUE"
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO `TypeTemp` (`id`, `name`, `version`, `type_kind`, "
               "        `description`, `input_type`, `output_type`) "
               " SELECT `id`, `name`, `version`, `type_kind`, `description`,"
               "        `input_type`, `output_type` "
               " FROM `Type`; "
      }
      upgrade_queries { query: " DROP TABLE `Type`; " }
      upgrade_queries {
        query: " ALTER TABLE `TypeTemp` rename to `Type`; "
      }
      upgrade_queries {
        query: " CREATE TABLE `ArtifactTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `type_id` INT NOT NULL, "
               "   `uri` TEXT, "
               "   `state` INT, "
               "   `name` VARCHAR(255), "
               "   `external_id` VARCHAR(255) UNIQUE, "
               "   `create_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   `last_update_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   UNIQUE(`type_id`, `name`) "
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO `ArtifactTemp` (`id`, `type_id`, `uri`, `state`, "
               "        `name`, `create_time_since_epoch`, "
               "        `last_update_time_since_epoch`) "
               " SELECT `id`, `type_id`, `uri`, `state`, `name`, "
               "        `create_time_since_epoch`, "
               "        `last_update_time_since_epoch` "
               "FROM `Artifact`; "
      }
      upgrade_queries { query: " DROP TABLE `Artifact`; " }
      upgrade_queries {
        query: " ALTER TABLE `ArtifactTemp` RENAME TO `Artifact`; "
      }
      upgrade_queries {
        query: " CREATE TABLE `ExecutionTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `type_id` INT NOT NULL, "
               "   `last_known_state` INT, "
               "   `name` VARCHAR(255), "
               "   `external_id` VARCHAR(255) UNIQUE, "
               "   `create_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   `last_update_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   UNIQUE(`type_id`, `name`) "
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO `ExecutionTemp` (`id`, `type_id`, "
               "        `last_known_state`, `name`, "
               "        `create_time_since_epoch`, "
               "        `last_update_time_since_epoch`) "
               " SELECT `id`, `type_id`, `last_known_state`, `name`, "
               "        `create_time_since_epoch`, "
               "        `last_update_time_since_epoch` "
               " FROM `Execution`; "
      }
      upgrade_queries { query: " DROP TABLE `Execution`; " }
      upgrade_queries {
        query: " ALTER TABLE `ExecutionTemp` RENAME TO `Execution`; "
      }
      upgrade_queries {
        query: " CREATE TABLE `ContextTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
               "   `type_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `external_id` VARCHAR(255) UNIQUE, "
               "   `create_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   `last_update_time_since_epoch` INT NOT NULL DEFAULT 0, "
               "   UNIQUE(`type_id`, `name`) "
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO `ContextTemp` (`id`, `type_id`, `name`, "
               "        `create_time_since_epoch`, "
               "        `last_update_time_since_epoch`) "
               " SELECT `id`, `type_id`, `name`, "
               "        `create_time_since_epoch`, "
               "        `last_update_time_since_epoch` "
               " FROM `Context`; "
      }
      upgrade_queries { query: " DROP TABLE `Context`; " }
      upgrade_queries {
        query: " ALTER TABLE `ContextTemp` RENAME TO `Context`; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_artifact_uri` "
               " ON `Artifact`(`uri`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_artifact_create_time_since_epoch` "
               " ON `Artifact`(`create_time_since_epoch`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_artifact_last_update_time_since_epoch` "
               " ON `Artifact`(`last_update_time_since_epoch`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_type_name` "
               " ON `Type`(`name`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_execution_create_time_since_epoch` "
               " ON `Execution`(`create_time_since_epoch`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_execution_last_update_time_since_epoch` "
               " ON `Execution`(`last_update_time_since_epoch`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_context_create_time_since_epoch` "
               " ON `Context`(`create_time_since_epoch`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   `idx_context_last_update_time_since_epoch` "
               " ON `Context`(`last_update_time_since_epoch`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_type_external_id` "
               " ON `Type`(`external_id`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_artifact_external_id` "
               " ON `Artifact`(`external_id`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_execution_external_id` "
               " ON `Execution`(`external_id`); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_context_external_id` "
               " ON `Context`(`external_id`); "
      }
      # check the expected table columns are created properly.
      # table type is using the old schema for upgrade verification, which
      # contains `is_artifact_type` column
      upgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Type`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` (`name`, `is_artifact_type`) VALUES "
                 " ('artifact_type', 1); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Artifact`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Artifact` "
                 " (`id`, `type_id`) "
                 " VALUES (1, 2); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Execution`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Execution` "
                 " (`id`, `type_id`) "
                 " VALUES (1, 2); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Context`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Context` "
                 " (`id`, `type_id`, `name`) "
                 " VALUES (1, 2, 'name1'); "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM `Type`; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM `Type` "
                 "   WHERE `name` = 'artifact_type' AND "
                 "         `external_id` IS NULL "
                 " ) AS T1; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM `Artifact`; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM `Artifact` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND "
                 "         `external_id` IS NULL "
                 " ) AS T1; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM `Execution`; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM `Execution` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND "
                 "          `external_id` IS NULL "
                 " ) AS T1; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM `Context`; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM `Context` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND `name` = 'name1' AND "
                 "         `external_id` IS NULL "
                 " ) as T1; "
        }
      }
      db_verification { total_num_indexes: 40 total_num_tables: 15 }
      # downgrade queries from version 10
      downgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ArtifactPropertyTemp` ( "
               "   `artifact_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `is_custom_property` TINYINT(1) NOT NULL, "
               "   `int_value` INT, "
               "   `double_value` DOUBLE, "
               "   `string_value` TEXT, "
               "   `byte_value` BLOB, "
               " PRIMARY KEY (`artifact_id`, `name`, `is_custom_property`)); "
      }
      downgrade_queries {
        query: " INSERT INTO `ArtifactPropertyTemp`  "
               " SELECT `artifact_id`, `name`,  `is_custom_property`, "
               "        `int_value`, `double_value`, `string_value`, "
               "        `byte_value` "
               " FROM `ArtifactProperty`; "
      }
      downgrade_queries { query: " DROP TABLE `ArtifactProperty`; " }
      downgrade_queries {
        query: " ALTER TABLE `ArtifactPropertyTemp` "
               "  RENAME TO `ArtifactProperty`; "
      }
      downgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ExecutionPropertyTemp` ( "
               "   `execution_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `is_custom_property` TINYINT(1) NOT NULL, "
               "   `int_value` INT, "
               "   `double_value` DOUBLE, "
               "   `string_value` TEXT, "
               "   `byte_value` BLOB, "
               " PRIMARY KEY (`execution_id`, `name`, `is_custom_property`)); "
      }
      downgrade_queries {
        query: " INSERT INTO `ExecutionPropertyTemp` "
               " SELECT `execution_id`, `name`,  `is_custom_property`, "
               "     `int_value`, `double_value`, `string_value`, "
               "     `byte_value` "
               " FROM `ExecutionProperty`; "
      }
      downgrade_queries { query: " DROP TABLE `ExecutionProperty`; " }
      downgrade_queries {
        query: " ALTER TABLE `ExecutionPropertyTemp` "
               "  RENAME TO `ExecutionProperty`; "
      }
      downgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ContextPropertyTemp` ( "
               "   `context_id` INT NOT NULL, "
               "   `name` VARCHAR(255) NOT NULL, "
               "   `is_custom_property` TINYINT(1) NOT NULL, "
               "   `int_value` INT, "
               "   `double_value` DOUBLE, "
               "   `string_value` TEXT, "
               "   `byte_value` BLOB, "
               " PRIMARY KEY (`context_id`, `name`, `is_custom_property`)); "
      }
      downgrade_queries {
        query: " INSERT INTO `ContextPropertyTemp` "
               " SELECT `context_id`, `name`,  `is_custom_property`, "
               "        `int_value`, `double_value`, `string_value`, "
               "        `byte_value` "
               " FROM `ContextProperty`; "
      }
      downgrade_queries { query: " DROP TABLE `ContextProperty`; " }
      downgrade_queries {
        query: " ALTER TABLE `ContextPropertyTemp` "
               "  RENAME TO `ContextProperty`; "
      }
      # recreate the indices that were dropped along with the old tables
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_artifact_property_int` "
               " ON `ArtifactProperty`(`name`, `is_custom_property`, "
               " `int_value`) "
               " WHERE `int_value` IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_artifact_property_double` "
               " ON `ArtifactProperty`(`name`, `is_custom_property`, "
               " `double_value`) "
               " WHERE `double_value` IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_artifact_property_string` "
               " ON `ArtifactProperty`(`name`, `is_custom_property`, "
               " `string_value`) "
               " WHERE `string_value` IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_execution_property_int` "
               " ON `ExecutionProperty`(`name`, `is_custom_property`, "
               " `int_value`) "
               " WHERE `int_value` IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_execution_property_double` "
               " ON `ExecutionProperty`(`name`, `is_custom_property`, "
               " `double_value`) "
               " WHERE `double_value` IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_execution_property_string` "
               " ON `ExecutionProperty`(`name`, `is_custom_property`, "
               " `string_value`) "
               " WHERE `string_value` IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_context_property_int` "
               " ON `ContextProperty`(`name`, `is_custom_property`, "
               " `int_value`) "
               " WHERE `int_value` IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_context_property_double` "
               " ON `ContextProperty`(`name`, `is_custom_property`, "
               " `double_value`) "
               " WHERE `double_value` IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS `idx_context_property_string` "
               " ON `ContextProperty`(`name`, `is_custom_property`, "
               " `string_value`) "
               " WHERE `string_value` IS NOT NULL; "
      }
)pb",
R"pb(
      # verify that downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries {
          query: "DELETE FROM `ArtifactProperty`;"
        }
        previous_version_setup_queries {
          query: "DELETE FROM `ExecutionProperty`;"
        }
        previous_version_setup_queries {
          query: "DELETE FROM `ContextProperty`;"
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ArtifactProperty` (`artifact_id`, "
                 "     `is_custom_property`, `name`, `string_value`) "
                 " VALUES (1, 0, 'p1', 'abc'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ExecutionProperty` (`execution_id`, "
                 "     `is_custom_property`, `name`, `int_value`) "
                 " VALUES (1, 1, 'p1', 1); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ContextProperty` (`context_id`, "
                 "     `is_custom_property`, `name`, `double_value`) "
                 " VALUES (1, 0, 'p1', 1.0); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        PRAGMA_TABLE_INFO('ArtifactProperty') "
                 " WHERE `name` = 'proto_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        PRAGMA_TABLE_INFO('ArtifactProperty') "
                 " WHERE `name` = 'bool_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        PRAGMA_TABLE_INFO('ExecutionProperty') "
                 " WHERE `name` = 'proto_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        PRAGMA_TABLE_INFO('ExecutionProperty') "
                 " WHERE `name` = 'bool_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        PRAGMA_TABLE_INFO('ContextProperty') "
                 " WHERE `name` = 'proto_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        PRAGMA_TABLE_INFO('ContextProperty') "
                 " WHERE `name` = 'bool_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ArtifactProperty` "
                 " WHERE `artifact_id` = 1 AND `is_custom_property` = 0 AND "
                 "       `name` = 'p1' AND `string_value` = 'abc'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ExecutionProperty` "
                 " WHERE `execution_id` = 1 AND `is_custom_property` = 1 AND "
                 "        `name` = 'p1' AND `int_value` = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ContextProperty` "
                 " WHERE `context_id` = 1  AND `is_custom_property` = 0 AND "
                 "        `name` = 'p1' AND `double_value` = 1.0; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v10, we added proto_value and bool_value columns to {X}Property tables
  migration_schemes {
    key: 10
    value: {
      upgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               " ADD COLUMN `proto_value` BLOB; "
      }
      upgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               " ADD COLUMN `bool_value` BOOLEAN; "
      }
      upgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               " ADD COLUMN `proto_value` BLOB; "
      }
      upgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               " ADD COLUMN `bool_value` BOOLEAN; "
      }
      upgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               " ADD COLUMN `proto_value` BLOB;"
      }
      upgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               " ADD COLUMN `bool_value` BOOLEAN; "
      }
      db_verification { total_num_indexes: 40 total_num_tables: 15 }
    }
  }
)pb");

// Template queries for MySQLMetadataSources.
// no-lint to support vc (C2026) 16380 max length for char[].
const std::string kMySQLMetadataSourceQueryConfig = absl::StrCat(  // NOLINT
R"pb(
  metadata_source_type: MYSQL_METADATA_SOURCE
  select_last_insert_id { query: " SELECT last_insert_id(); " }
  select_type_by_name {
    query: " SELECT `id`, `name`, `version`, `description`, "
           "        `input_type`, `output_type` FROM `Type` "
           " WHERE name = $0 AND version IS NULL AND type_kind = $1 "
           " LOCK IN SHARE MODE; "
    parameter_num: 2
  }
  select_type_by_name_and_version {
    query: " SELECT `id`, `name`, `version`, `description`, "
           "        `input_type`, `output_type` FROM `Type` "
           " WHERE name = $0 AND version = $1 AND type_kind = $2 "
           " LOCK IN SHARE MODE; "
    parameter_num: 3
  }
  select_types_by_names {
    query: " SELECT `id`, `name`, `version`, `description`, "
           "        `input_type`, `output_type` FROM `Type` "
           " WHERE name IN ($0) AND version IS NULL AND type_kind = $1 "
           " LOCK IN SHARE MODE; "
    parameter_num: 2
    }
    select_types_by_names_and_versions {
    query: " SELECT `id`, `name`, `version`, `description`, "
           "        `input_type`, `output_type` FROM `Type` "
           " WHERE (name, version) IN ($0) AND type_kind = $1 "
           " LOCK IN SHARE MODE; "
    parameter_num: 2
    }
  select_context_by_id {
    query: " SELECT C.id, C.type_id, C.name, C.external_id, "
           "        C.create_time_since_epoch, C.last_update_time_since_epoch, "
           "        T.name AS `type`, T.version AS type_version, "
           "        T.description AS type_description, "
           "        T.external_id AS type_external_id "
           " FROM `Context` AS C "
           " LEFT JOIN `Type` AS T ON (T.id = C.type_id) "
           " WHERE C.id IN ($0) LOCK IN SHARE MODE; "
    parameter_num: 1
  }
  select_execution_by_id {
    query: " SELECT E.id, E.type_id, E.last_known_state, E.name, "
           "        E.external_id, E.create_time_since_epoch, "
           "        E.last_update_time_since_epoch, "
           "        T.name AS `type`, T.version AS type_version, "
           "        T.description AS type_description, "
           "        T.external_id AS type_external_id "
           " FROM `Execution` AS E "
           " LEFT JOIN `Type` AS T "
           "   ON (T.id = E.type_id) "
           " WHERE E.id IN ($0) LOCK IN SHARE MODE; "
    parameter_num: 1
  }
  select_artifact_by_id {
    query: " SELECT A.id, A.type_id, A.uri, A.state, A.name, "
           "        A.external_id, A.create_time_since_epoch, "
           "        A.last_update_time_since_epoch, "
           "        T.name AS `type`, T.version AS type_version, "
           "        T.description AS type_description, "
           "        T.external_id AS type_external_id "
           " FROM `Artifact` AS A "
           " LEFT JOIN `Type` AS T "
           "   ON (T.id = A.type_id) "
           " WHERE A.id IN ($0) LOCK IN SHARE MODE; "
    parameter_num: 1
  }
  select_parent_type_by_type_id {
    query: " SELECT `type_id`, `parent_type_id` "
           " FROM `ParentType` WHERE type_id IN ($0) "
           " LOCK IN SHARE MODE; "
    parameter_num: 1
  }
  create_type_table {
    query: " CREATE TABLE IF NOT EXISTS `Type` ( "
           "   `id` INT PRIMARY KEY AUTO_INCREMENT, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `version` VARCHAR(255), "
           "   `type_kind` TINYINT(1) NOT NULL, "
           "   `description` TEXT, "
           "   `input_type` TEXT, "
           "   `output_type` TEXT, "
           "   `external_id` VARCHAR(255) UNIQUE "
           " ); "
  }
  create_artifact_table {
    query: " CREATE TABLE IF NOT EXISTS `Artifact` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `uri` TEXT, "
           "   `state` INT, "
           "   `name` VARCHAR(255), "
           "   `external_id` VARCHAR(255) UNIQUE, "
           "   `create_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   `last_update_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   CONSTRAINT UniqueArtifactTypeName UNIQUE(`type_id`, `name`) "
           " ); "
  }
  create_artifact_property_table {
    query: " CREATE TABLE IF NOT EXISTS `ArtifactProperty` ( "
           "   `artifact_id` INT NOT NULL, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `is_custom_property` TINYINT(1) NOT NULL, "
           "   `int_value` INT, "
           "   `double_value` DOUBLE, "
           "   `string_value` MEDIUMTEXT, "
           "   `byte_value` MEDIUMBLOB, "
           "   `proto_value` MEDIUMBLOB, "
           "   `bool_value` BOOLEAN, "
           " PRIMARY KEY (`artifact_id`, `name`, `is_custom_property`)); "
  }
  create_execution_table {
    query: " CREATE TABLE IF NOT EXISTS `Execution` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `last_known_state` INT, "
           "   `name` VARCHAR(255), "
           "   `external_id` VARCHAR(255) UNIQUE, "
           "   `create_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   `last_update_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   CONSTRAINT UniqueExecutionTypeName UNIQUE(`type_id`, `name`) "
           " ); "
  }
  create_execution_property_table {
    query: " CREATE TABLE IF NOT EXISTS `ExecutionProperty` ( "
           "   `execution_id` INT NOT NULL, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `is_custom_property` TINYINT(1) NOT NULL, "
           "   `int_value` INT, "
           "   `double_value` DOUBLE, "
           "   `string_value` MEDIUMTEXT, "
           "   `byte_value` MEDIUMBLOB, "
           "   `proto_value` MEDIUMBLOB, "
           "   `bool_value` BOOLEAN, "
           " PRIMARY KEY (`execution_id`, `name`, `is_custom_property`)); "
  }
  create_context_table {
    query: " CREATE TABLE IF NOT EXISTS `Context` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `external_id` VARCHAR(255) UNIQUE, "
           "   `create_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   `last_update_time_since_epoch` BIGINT NOT NULL DEFAULT 0, "
           "   UNIQUE(`type_id`, `name`) "
           " ); "
  }
  create_context_property_table {
    query: " CREATE TABLE IF NOT EXISTS `ContextProperty` ( "
           "   `context_id` INT NOT NULL, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `is_custom_property` TINYINT(1) NOT NULL, "
           "   `int_value` INT, "
           "   `double_value` DOUBLE, "
           "   `string_value` MEDIUMTEXT, "
           "   `byte_value` MEDIUMBLOB, "
           "   `proto_value` MEDIUMBLOB, "
           "   `bool_value` BOOLEAN, "
           " PRIMARY KEY (`context_id`, `name`, `is_custom_property`)); "
  }
  create_event_table {
    query: " CREATE TABLE IF NOT EXISTS `Event` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `artifact_id` INT NOT NULL, "
           "   `execution_id` INT NOT NULL, "
           "   `type` INT NOT NULL, "
           "   `milliseconds_since_epoch` BIGINT, "
           "   CONSTRAINT UniqueEvent UNIQUE( "
           "     `artifact_id`, `execution_id`, `type`) "
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
  # secondary indices in the current schema.
  secondary_indices {
    # MySQL does not support arbitrary length string index. Only prefix
    # index
    # is supported. Max size for 5.6/5.7 is 255 char for utf8 charset.
    query: " ALTER TABLE `Artifact` "
           "  ADD INDEX `idx_artifact_uri`(`uri`(255)), "
           "  ADD INDEX `idx_artifact_create_time_since_epoch` "
           "             (`create_time_since_epoch`), "
           "  ADD INDEX `idx_artifact_last_update_time_since_epoch` "
           "             (`last_update_time_since_epoch`); "
  }
  secondary_indices {
    query: " ALTER TABLE `Event` "
           " ADD INDEX `idx_event_execution_id` (`execution_id`); "
  }
  secondary_indices {
    query: " ALTER TABLE `ParentContext` "
           " ADD INDEX "
           "   `idx_parentcontext_parent_context_id` (`parent_context_id`); "
  }
  secondary_indices {
    query: " ALTER TABLE `Type` "
           " ADD INDEX `idx_type_name` (`name`); "
  }
  secondary_indices {
    query: " ALTER TABLE `Execution` "
           "  ADD INDEX `idx_execution_create_time_since_epoch` "
           "             (`create_time_since_epoch`), "
           "  ADD INDEX `idx_execution_last_update_time_since_epoch` "
           "             (`last_update_time_since_epoch`); "
  }
  secondary_indices {
    query: " ALTER TABLE `Context` "
           "  ADD INDEX `idx_context_create_time_since_epoch` "
           "             (`create_time_since_epoch`), "
           "  ADD INDEX `idx_context_last_update_time_since_epoch` "
           "             (`last_update_time_since_epoch`); "
  }
  secondary_indices {
    query: " ALTER TABLE `EventPath` "
           "  ADD INDEX `idx_eventpath_event_id`(`event_id`); "
  }
  secondary_indices {
    query: " ALTER TABLE `ArtifactProperty` "
           "  ADD INDEX `idx_artifact_property_int`( "
           "    `name`, `is_custom_property`, `int_value`); "
  }
  secondary_indices {
    query: " ALTER TABLE `ArtifactProperty` "
           "  ADD INDEX `idx_artifact_property_double`( "
           "    `name`, `is_custom_property`, `double_value`); "
  }
  secondary_indices {
    query: " ALTER TABLE `ArtifactProperty` "
           "  ADD INDEX `idx_artifact_property_string`( "
           "    `name`, `is_custom_property`, `string_value`(255)); "
  }
  secondary_indices {
    query: " ALTER TABLE `ExecutionProperty` "
           "  ADD INDEX `idx_execution_property_int`( "
           "    `name`, `is_custom_property`, `int_value`); "
  }
  secondary_indices {
    query: " ALTER TABLE `ExecutionProperty` "
           "  ADD INDEX `idx_execution_property_double`( "
           "    `name`, `is_custom_property`, `double_value`); "
  }
  secondary_indices {
    query: " ALTER TABLE `ExecutionProperty` "
           "  ADD INDEX `idx_execution_property_string`( "
           "    `name`, `is_custom_property`, `string_value`(255)); "
  }
  secondary_indices {
    query: " ALTER TABLE `ContextProperty` "
           "  ADD INDEX `idx_context_property_int`( "
           "    `name`, `is_custom_property`, `int_value`); "
  }
  secondary_indices {
    query: " ALTER TABLE `ContextProperty` "
           "  ADD INDEX `idx_context_property_double`( "
           "    `name`, `is_custom_property`, `double_value`); "
  }
  secondary_indices {
    query: " ALTER TABLE `ContextProperty` "
           "  ADD INDEX `idx_context_property_string`( "
           "    `name`, `is_custom_property`, `string_value`(255)); "
  }
  secondary_indices {
    query: " ALTER TABLE `Type` "
           " ADD INDEX `idx_type_external_id` (`external_id`);"
  }
  secondary_indices {
    query: " ALTER TABLE `Artifact` "
           " ADD INDEX `idx_artifact_external_id` (`external_id`);"
  }
  secondary_indices {
    query: " ALTER TABLE `Execution` "
           " ADD INDEX `idx_execution_external_id` (`external_id`);"
  }
  secondary_indices {
    query: " ALTER TABLE `Context` "
           " ADD INDEX `idx_context_external_id` (`external_id`);"
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
      # downgrade queries from version 2, drop all ContextTypes and rename
      # the `type_kind` back to `is_artifact_type` column.
      downgrade_queries {
        query: " DELETE FROM `Type` WHERE `type_kind` = 2; "
      }
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
        # check after migration, the existing types are the same including
        # id.
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
      downgrade_queries {
        query: " DROP TABLE IF EXISTS `ContextProperty`; "
      }
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
      # downgrade queries from version 6
      downgrade_queries { query: " DROP TABLE `ParentType`; " }
      downgrade_queries { query: " DROP TABLE `ParentContext`; " }
      downgrade_queries {
        query: " ALTER TABLE `Type` "
               " DROP COLUMN `version`, "
               " DROP COLUMN `description`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Artifact` "
               " DROP INDEX `idx_artifact_uri`, "
               " DROP INDEX `idx_artifact_create_time_since_epoch`, "
               " DROP INDEX `idx_artifact_last_update_time_since_epoch`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Event` "
               " DROP INDEX `idx_event_artifact_id`, "
               " DROP INDEX `idx_event_execution_id`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Type` "
               " DROP INDEX `idx_type_name`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Execution` "
               " DROP INDEX `idx_execution_create_time_since_epoch`, "
               " DROP INDEX `idx_execution_last_update_time_since_epoch`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Context` "
               " DROP INDEX `idx_context_create_time_since_epoch`, "
               " DROP INDEX `idx_context_last_update_time_since_epoch`; "
      }
      # verify if the downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Type`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`id`, `name`, `version`, `type_kind`, "
                 "  `description`, `input_type`, `output_type`) "
                 " VALUES (1, 't1', 'v1', 1, 'desc1', 'input1', 'output1'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`id`, `name`, `version`, `type_kind`, "
                 "  `description`, `input_type`, `output_type`) "
                 " VALUES (2, 't2', 'v2', 2, 'desc2', 'input2', 'output2'); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 2 FROM `Type`; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Type` "
                 "   WHERE `id` = 1 AND `name` = 't1' AND type_kind = 1 "
                 "   AND `input_type` = 'input1' AND `output_type` = 'output1'"
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`tables` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'ParentType'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`tables` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'ParentContext'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'Artifact' AND "
                 "       `index_name` LIKE 'idx_artifact_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'Event' AND "
                 "       `index_name` LIKE 'idx_event_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'ParentContext' AND "
                 "       `index_name` LIKE 'idx_parentcontext_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'Type' AND "
                 "       `index_name` LIKE 'idx_type_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'Execution' AND "
                 "       `index_name` LIKE 'idx_execution_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'Context' AND "
                 "       `index_name` LIKE 'idx_context_%'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v6, to support parental type and parental context, we added two
  # tables `ParentType` and `ParentContext`. In addition, we added `version`
  # and `description` in the `Type` table for improving type registrations.
  # We introduce indices on Type.name, Artifact.uri, Event's artifact_id and
  # execution_id, and create_time_since_epoch, last_update_time_since_epoch
  # for all nodes.
  migration_schemes {
    key: 6
    value: {
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ParentType` ( "
               "   `type_id` INT NOT NULL, "
               "   `parent_type_id` INT NOT NULL, "
               " PRIMARY KEY (`type_id`, `parent_type_id`)); "
      }
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS `ParentContext` ( "
               "   `context_id` INT NOT NULL, "
               "   `parent_context_id` INT NOT NULL, "
               " PRIMARY KEY (`context_id`, `parent_context_id`)); "
      }
      upgrade_queries {
        query: " ALTER TABLE `Type` ADD ( "
               "   `version` VARCHAR(255), "
               "   `description` TEXT "
               " ); "
      }
      upgrade_queries {
        query: " ALTER TABLE `Artifact` "
               " ADD INDEX `idx_artifact_uri`(`uri`(255)), "
               "  ADD INDEX `idx_artifact_create_time_since_epoch` "
               "             (`create_time_since_epoch`), "
               "  ADD INDEX `idx_artifact_last_update_time_since_epoch` "
               "             (`last_update_time_since_epoch`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `Event` "
               " ADD INDEX `idx_event_artifact_id` (`artifact_id`), "
               " ADD INDEX `idx_event_execution_id` (`execution_id`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `ParentContext` "
               " ADD INDEX "
               "  `idx_parentcontext_parent_context_id` (`parent_context_id`);"
      }
      upgrade_queries {
        query: " ALTER TABLE `Type` "
               " ADD INDEX `idx_type_name` (`name`);"
      }
      upgrade_queries {
        query: " ALTER TABLE `Execution` "
               "  ADD INDEX `idx_execution_create_time_since_epoch` "
               "             (`create_time_since_epoch`), "
               "  ADD INDEX `idx_execution_last_update_time_since_epoch` "
               "             (`last_update_time_since_epoch`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `Context` "
               "  ADD INDEX `idx_context_create_time_since_epoch` "
               "             (`create_time_since_epoch`), "
               "  ADD INDEX `idx_context_last_update_time_since_epoch` "
               "             (`last_update_time_since_epoch`); "
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        # check existing rows in previous Type table are migrated properly.
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` WHERE "
                 " `id` = 1 AND `type_kind` = 1 AND `name` = 'artifact_type'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type` WHERE "
                 " `id` = 2 AND `type_kind` = 0 AND `name` = 'execution_type' "
                 " AND `input_type` = 'input' AND `output_type` = 'output'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `type_id`, `parent_type_id` "
                 "   FROM `ParentType` "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT `context_id`, `parent_context_id` "
                 "   FROM `ParentContext` "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Artifact' AND "
                 "       `index_name` = 'idx_artifact_uri'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Artifact' AND "
                 "       `index_name` = 'idx_artifact_create_time_since_epoch';"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Artifact' AND `index_name` = "
                 "       'idx_artifact_last_update_time_since_epoch';"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Event' AND "
                 "       `index_name` = 'idx_event_artifact_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Event' AND "
                 "       `index_name` = 'idx_event_execution_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'ParentContext' AND "
                 "       `index_name` = 'idx_parentcontext_parent_context_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Type' AND "
                 "       `index_name` = 'idx_type_name'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Execution' AND `index_name` = "
                 "       'idx_execution_create_time_since_epoch'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Execution' AND `index_name` = "
                 "       'idx_execution_last_update_time_since_epoch'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Context' AND "
                 "       `index_name` = 'idx_context_create_time_since_epoch'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Context' AND `index_name` = "
                 "       'idx_context_last_update_time_since_epoch'; "
        }
      }
      # downgrade queries from version 7
      # Note v7 for MySQL used mediumtext for string_value, when downgrade
      # the long text will be truncated to 65536 chars.
      downgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` DROP COLUMN `byte_value`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` DROP COLUMN `byte_value`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ContextProperty` DROP COLUMN `byte_value`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `EventPath` DROP INDEX `idx_eventpath_event_id`; "
      }
      downgrade_queries {
        query: " UPDATE `ArtifactProperty` "
               " SET `string_value` = SUBSTRING(`string_value`, 1, 65535); "
      }
      downgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               " MODIFY COLUMN `string_value` TEXT; "
      }
      downgrade_queries {
        query: " UPDATE `ExecutionProperty` "
               " SET `string_value` = SUBSTRING(`string_value`, 1, 65535); "
      }
      downgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               " MODIFY COLUMN `string_value` TEXT; "
      }
      downgrade_queries {
        query: " UPDATE `ContextProperty` "
               " SET `string_value` = SUBSTRING(`string_value`, 1, 65535); "
      }
      downgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               " MODIFY COLUMN `string_value` TEXT; "
      }
      # verify if the downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries {
          query: "DELETE FROM `ArtifactProperty`;"
        }
        previous_version_setup_queries {
          query: "DELETE FROM `ExecutionProperty`;"
        }
        previous_version_setup_queries {
          query: "DELETE FROM `ContextProperty`;"
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ArtifactProperty` "
                 " (`artifact_id`, `name`, `string_value`) "
                 " VALUES (1, 'p1', CONCAT('_prefix_', REPEAT('a', 160000))), "
                 "        (1, 'p2', 'abc'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ExecutionProperty` "
                 " (`execution_id`, `name`, `string_value`) "
                 " VALUES (1, 'p1', CONCAT('_prefix_', REPEAT('e', 160000))), "
                 "        (1, 'p2', 'abc'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ContextProperty` "
                 " (`context_id`, `name`, `string_value`) "
                 " VALUES (1, 'p1', CONCAT('_prefix_', REPEAT('c', 160000))), "
                 "        (1, 'p2', 'abc'); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'EventPath' AND "
                 "       `index_name` = 'idx_eventpath_event_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 3 FROM `information_schema`.`columns` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` IN ('ArtifactProperty', "
                 "           'ExecutionProperty', 'ContextProperty') AND "
                 "       `column_name` = 'string_value' AND "
                 "       `data_type` = 'text'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`columns` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` IN ('ArtifactProperty', "
                 "           'ExecutionProperty', 'ContextProperty') AND "
                 "       `column_name` = 'byte_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ArtifactProperty` "
                 " WHERE `artifact_id` = 1 AND `name` = 'p1' AND "
                 "   `string_value` = CONCAT('_prefix_', REPEAT('a', 65527)); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ArtifactProperty` "
                 " WHERE `artifact_id` = 1 AND `name` = 'p2' AND "
                 "       `string_value` = 'abc'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ExecutionProperty` "
                 " WHERE `execution_id` = 1 AND `name` = 'p1' AND "
                 "   `string_value` = CONCAT('_prefix_', REPEAT('e', 65527)); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ExecutionProperty` "
                 " WHERE `execution_id` = 1 AND `name` = 'p2' AND "
                 "       `string_value` = 'abc'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ContextProperty` "
                 " WHERE `context_id` = 1 AND `name` = 'p1' AND "
                 "   `string_value` = CONCAT('_prefix_', REPEAT('c', 65527)); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `ContextProperty` "
                 " WHERE `context_id` = 1 AND `name` = 'p2' AND "
                 "       `string_value` = 'abc'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v7, we added byte_value for property tables for better storing binary
  # property values. For MySQL, we extends string_value column to be
  # MEDIUMTEXT in order to persist string value with size upto 16MB. In
  # addition, we added index for `EventPath` to improve Event reads.
  migration_schemes {
    key: 7
    value: {
      upgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               " MODIFY COLUMN `string_value` MEDIUMTEXT, "
               " ADD COLUMN `byte_value` MEDIUMBLOB; "
      }
      upgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               " MODIFY COLUMN `string_value` MEDIUMTEXT, "
               " ADD COLUMN `byte_value` MEDIUMBLOB; "
      }
      upgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               " MODIFY COLUMN `string_value` MEDIUMTEXT, "
               " ADD COLUMN `byte_value` MEDIUMBLOB; "
      }
      upgrade_queries {
        query: " ALTER TABLE `EventPath` "
               " ADD INDEX `idx_eventpath_event_id` (`event_id`); "
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        # check existing rows in previous Type table are migrated properly.
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ArtifactProperty` WHERE "
                 " `byte_value` IS NOT NULL; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ExecutionProperty` WHERE "
                 " `byte_value` IS NOT NULL; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ContextProperty` WHERE "
                 " `byte_value` IS NOT NULL; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'EventPath' AND "
                 "       `index_name` = 'idx_eventpath_event_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 3 FROM `information_schema`.`columns` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` IN ('ArtifactProperty', "
                 "           'ExecutionProperty', 'ContextProperty') AND "
                 "       `column_name` = 'string_value' AND "
                 "       `data_type` = 'mediumtext'; "
        }
      }
      db_verification { total_num_indexes: 45 total_num_tables: 15 }
      # downgrade queries from version 8
      downgrade_queries {
        query: " ALTER TABLE `Event` "
               " DROP INDEX UniqueEvent; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Event` "
               " ADD INDEX `idx_event_artifact_id` (`artifact_id`); "
      }
      downgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               " DROP INDEX `idx_artifact_property_int`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               " DROP INDEX `idx_artifact_property_double`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               " DROP INDEX `idx_artifact_property_string`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               " DROP INDEX `idx_execution_property_int`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               " DROP INDEX `idx_execution_property_double`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               " DROP INDEX `idx_execution_property_string`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               " DROP INDEX `idx_context_property_int`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               " DROP INDEX `idx_context_property_double`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               " DROP INDEX `idx_context_property_string`; "
      }
      downgrade_verification {
        previous_version_setup_queries {
          query: " INSERT INTO `Event` "
                 " (`id`, `artifact_id`, `execution_id`, `type`, "
                 " `milliseconds_since_epoch`) "
                 " VALUES (1, 1, 1, 1, 1); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `EventPath` "
                 " (`event_id`, `is_index_step`, `step_index`, `step_key`) "
                 " VALUES (1, 1, 1, 'a'); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Event` "
                 " WHERE `artifact_id` = 1 AND `execution_id` = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `EventPath` "
                 " WHERE `event_id` = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'Event' AND "
                 "       `index_name` = 'UniqueEvent'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'Event' AND "
                 "       `index_name` = 'idx_event_artifact_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'Event' AND "
                 "       `index_name` = 'idx_event_execution_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'EventPath' AND "
                 "       `index_name` = 'idx_eventpath_event_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'ArtifactProperty' AND "
                 "       `index_name` LIKE 'idx_artifact_property_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'ExecutionProperty' AND "
                 "       `index_name` LIKE 'idx_execution_property_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) AND "
                 "       `table_name` = 'ContextProperty' AND "
                 "       `index_name` LIKE 'idx_context_property_%'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v8, we added index for `ArtifactProperty`, `ExecutionProperty`,
  # `ContextProperty` to improve property queries on name.
  migration_schemes {
    key: 8
    value: {
      upgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               "  ADD INDEX `idx_artifact_property_int`( "
               "    `name`, `is_custom_property`, `int_value`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               "  ADD INDEX `idx_artifact_property_double`( "
               "    `name`, `is_custom_property`, `double_value`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               "  ADD INDEX `idx_artifact_property_string`( "
               "    `name`, `is_custom_property`, `string_value`(255)); "
      }
      upgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               "  ADD INDEX `idx_execution_property_int`( "
               "    `name`, `is_custom_property`, `int_value`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               "  ADD INDEX `idx_execution_property_double`( "
               "    `name`, `is_custom_property`, `double_value`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               "  ADD INDEX `idx_execution_property_string`( "
               "    `name`, `is_custom_property`, `string_value`(255)); "
      }
      upgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               "  ADD INDEX `idx_context_property_int`( "
               "    `name`, `is_custom_property`, `int_value`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               "  ADD INDEX `idx_context_property_double`( "
               "    `name`, `is_custom_property`, `double_value`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               "  ADD INDEX `idx_context_property_string`( "
               "    `name`, `is_custom_property`, `string_value`(255)); "
      }
      upgrade_queries {
        query: " CREATE TABLE `EventTemp` ( "
               "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
               "   `artifact_id` INT NOT NULL, "
               "   `execution_id` INT NOT NULL, "
               "   `type` INT NOT NULL, "
               "   `milliseconds_since_epoch` BIGINT, "
               "   CONSTRAINT UniqueEvent UNIQUE(`artifact_id`, `execution_id`, `type`) "
               " ); "
      }
      upgrade_queries {
        query: " INSERT IGNORE INTO `EventTemp` "
               " (`id`, `artifact_id`, `execution_id`, `type`, "
               " `milliseconds_since_epoch`) "
               " SELECT * FROM `Event` ORDER BY `id` desc; "
      }
      upgrade_queries { query: " DROP TABLE `Event`; " }
      upgrade_queries {
        query: " ALTER TABLE `EventTemp` RENAME TO `Event`, "
               " ADD INDEX `idx_event_execution_id` (`execution_id`) ; "
      }
      upgrade_queries {
        query: " DELETE FROM `EventPath` "
               "   WHERE event_id not in ( SELECT `id` from Event ) "
      }
      # check the expected indexes are created properly.
      upgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Event`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Event` "
                 " (`id`, `artifact_id`, `execution_id`, `type`, "
                 " `milliseconds_since_epoch`) "
                 " VALUES (1, 1, 1, 1, 1); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `Event` "
                 " (`id`, `artifact_id`, `execution_id`, `type`, "
                 " `milliseconds_since_epoch`) "
                 " VALUES (2, 1, 1, 1, 2); "
        }
        previous_version_setup_queries { query: "DELETE FROM `EventPath`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `EventPath` "
                 " (`event_id`, `is_index_step`, `step_index`, `step_key`) "
                 " VALUES (1, 1, 1, 'a'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `EventPath` "
                 " (`event_id`, `is_index_step`, `step_index`, `step_key`) "
                 " VALUES (2, 1, 1, 'b'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `EventPath` "
                 " (`event_id`, `is_index_step`, `step_index`, `step_key`) "
                 " VALUES (2, 1, 2, 'c'); "
        }
        # check event table unique constraint is applied and event path
        # records are deleted.
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Event` "
                 " WHERE `artifact_id` = 1 AND `execution_id` = 1 "
                 "     AND `type` = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Event` "
                 "   WHERE `id` = 2 AND `artifact_id` = 1 AND  "
                 "       `execution_id` = 1 AND `type` = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 2 FROM `EventPath` "
                 " WHERE `event_id` = 2; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `EventPath` "
                 " WHERE `event_id` = 2 AND `step_key` = 'b'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `EventPath` "
                 " WHERE `event_id` = 2 AND `step_key` = 'c'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `EventPath` "
                 " WHERE `event_id` = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 3 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Event' AND "
                 "       `index_name` = 'UniqueEvent'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'Event' AND "
                 "       `index_name` LIKE 'idx_event_execution_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'EventPath' AND "
                 "       `index_name` LIKE 'idx_eventpath_event_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 9 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'ArtifactProperty' AND "
                 "       `index_name` LIKE 'idx_artifact_property_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 9 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'ExecutionProperty' AND "
                 "       `index_name` LIKE 'idx_execution_property_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 9 FROM `information_schema`.`statistics` "
                 " WHERE `table_schema` = (SELECT DATABASE()) and "
                 "       `table_name` = 'ContextProperty' AND "
                 "       `index_name` LIKE 'idx_context_property_%'; "
        }
      }
      db_verification { total_num_indexes: 74 total_num_tables: 15 }
      # downgrade queries from version 9
      downgrade_queries {
        query: " ALTER TABLE `Type` DROP INDEX `idx_type_external_id`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Artifact` DROP INDEX `idx_artifact_external_id`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Execution` DROP INDEX `idx_execution_external_id`;"
      }
      downgrade_queries {
        query: " ALTER TABLE `Context` DROP INDEX `idx_context_external_id`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Type` DROP COLUMN `external_id`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Artifact` DROP COLUMN `external_id`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Execution` DROP COLUMN `external_id`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `Context` DROP COLUMN `external_id`; "
      }
      # verify if the downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Type`; " }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`id`, `name`, `type_kind`, "
                 "  `description`, `input_type`, `output_type`, `external_id`) "
                 " VALUES (1, 't1', 1, 'desc1', 'input1', 'output1', "
                 "         'type_1'); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Artifact`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Artifact` "
                 " (`id`, `type_id`, `uri`, `state`, `name`, `external_id`, "
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 'uri1', 1, NULL, 'artifact_1', 0, 1); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Execution`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Execution` "
                 " (`id`, `type_id`, `last_known_state`, `name`, `external_id`,"
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 1, NULL, 'execution_1', 0, 1); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Context`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Context` "
                 " (`id`, `type_id`, `name`, `external_id`, "
                 "  `create_time_since_epoch`, `last_update_time_since_epoch`) "
                 " VALUES (1, 2, 'name1', 'context_1', 1, 0); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type`; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Type` "
                 "   WHERE `id` = 1 AND `name` = 't1' AND type_kind = 1 AND "
                 "         `input_type` = 'input1' AND "
                 "         `output_type` = 'output1') AS T1;  "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM `Artifact`; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM `Artifact` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND `uri` = 'uri1' AND "
                 "         `state` = 1 "
                 " ) AS T1; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM `Execution`; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM `Execution` "
                 "   WHERE `id` = 1 AND `type_id` = 2 "
                 " ) AS T1; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM `Context`; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM `Context` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND `name` = 'name1' "
                 " ) as T1; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v9, to store the ids that come from the clients' system (like Vertex
  # Metadata), we added a new column `external_id` in the `Type`,
  # `Artifact`, `Execution`, `Context` tables. We introduce unique and
  # null-filtered indices on Type.external_id, Artifact.external_id,
  # Execution's external_id and Context's external_id.
  migration_schemes {
    key: 9
    value: {
      upgrade_queries {
        query: " ALTER TABLE `Type` ADD ( "
               "   `external_id` VARCHAR(255) "
               " ), "
               " ADD CONSTRAINT UniqueTypeExternalId "
               " UNIQUE(`external_id`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `Artifact` ADD ( "
               "   `external_id` VARCHAR(255) "
               " ), "
               " ADD CONSTRAINT UniqueArtifactExternalId "
               " UNIQUE(`external_id`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `Execution` ADD ( "
               "   `external_id` VARCHAR(255) "
               " ), "
               " ADD CONSTRAINT UniqueExecutionExternalId "
               " UNIQUE(`external_id`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `Context` ADD ( "
               "   `external_id` VARCHAR(255) "
               " ), "
               " ADD CONSTRAINT UniqueContextExternalId "
               " UNIQUE(`external_id`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `Type` "
               " ADD INDEX `idx_type_external_id` (`external_id`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `Artifact` "
               " ADD INDEX `idx_artifact_external_id` (`external_id`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `Execution` "
               " ADD INDEX `idx_execution_external_id` (`external_id`); "
      }
      upgrade_queries {
        query: " ALTER TABLE `Context` "
               " ADD INDEX `idx_context_external_id` (`external_id`); "
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM `Type`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Type` "
                 " (`id`, `name`) "
                 " VALUES (1, 't1'); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Artifact`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Artifact` "
                 " (`id`, `type_id`) "
                 " VALUES (1, 2); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Execution`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Execution` "
                 " (`id`, `type_id`) "
                 " VALUES (1, 2); "
        }
        previous_version_setup_queries { query: "DELETE FROM `Context`;" }
        previous_version_setup_queries {
          query: " INSERT INTO `Context` "
                 " (`id`, `type_id`, `name`) "
                 " VALUES (1, 2, 'name1'); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM `Type`; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM `Type` "
                 "   WHERE `id` = 1 AND `name` = 't1' AND "
                 "         `external_id` IS NULL "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM `Artifact`; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM `Artifact` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND "
                 "         `external_id` IS NULL "
                 " ) AS T1; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM `Execution`; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM `Execution` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND "
                 "         `external_id` IS NULL "
                 " ) AS T1; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM `Context`; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM `Context` "
                 "   WHERE `id` = 1 AND `type_id` = 2 AND `name` = 'name1' AND "
                 "         `external_id` IS NULL "
                 " ) as T1; "
        }
      }
      db_verification { total_num_indexes: 82 total_num_tables: 15 }
      # downgrade queries from version 10
      downgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               " DROP COLUMN `proto_value`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               " DROP COLUMN `bool_value`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               " DROP COLUMN `proto_value`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               " DROP COLUMN `bool_value`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               " DROP COLUMN `proto_value`; "
      }
      downgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               " DROP COLUMN `bool_value`; "
      }
      downgrade_verification {
        previous_version_setup_queries {
          query: " INSERT INTO `Artifact` "
                 " (`id`, `name`) "
                 " VALUES (1, 'artifact-name'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ArtifactProperty` (`artifact_id`, "
                 "     `name`, `is_custom_property`, `string_value`) "
                 " VALUES (1, 'p-0', false, 'string_property_value'); "
        }

        previous_version_setup_queries {
          query: " INSERT INTO `Execution` "
                 " (`id`, `name`) "
                 " VALUES (1, 'execution-name'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ExecutionProperty` (`execution_id`, "
                 "     `name`, `is_custom_property`, `string_value`) "
                 " VALUES (1, 'p-0', false, 'string_property_value'); "
        }

        previous_version_setup_queries {
          query: " INSERT INTO `Context` "
                 " (`id`, `name`) "
                 " VALUES (1, 'context-name'); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO `ContextProperty` (`context_id`, "
                 "     `name`, `is_custom_property`, `string_value`) "
                 " VALUES (1, 'p-0', false, 'string_property_value'); "
        }

        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 "
                 " FROM ArtifactProperty AS AP "
                 " WHERE AP.artifact_id = 1 "
                 "   AND AP.name = 'p-0' "
                 "   AND AP.is_custom_property = false "
                 "   AND AP.string_value = 'string_property_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 "
                 " FROM ExecutionProperty AS EP "
                 " WHERE EP.execution_id = 1 "
                 "   AND EP.name = 'p-0' "
                 "   AND EP.is_custom_property = false "
                 "   AND EP.string_value = 'string_property_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 "
                 " FROM ContextProperty AS CP "
                 " WHERE CP.context_id = 1 "
                 "   AND CP.name = 'p-0' "
                 "   AND CP.is_custom_property = false "
                 "   AND CP.string_value = 'string_property_value'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v10, we added proto_value and bool_value for property tables.
  migration_schemes {
    key: 10
    value: {
      upgrade_queries {
        query: " ALTER TABLE `ArtifactProperty` "
               " ADD COLUMN `proto_value` MEDIUMBLOB, "
               " ADD COLUMN `bool_value` BOOLEAN; "
      }
      upgrade_queries {
        query: " ALTER TABLE `ExecutionProperty` "
               " ADD COLUMN `proto_value` MEDIUMBLOB, "
               " ADD COLUMN `bool_value` BOOLEAN; "
      }
      upgrade_queries {
        query: " ALTER TABLE `ContextProperty` "
               " ADD COLUMN `proto_value` MEDIUMBLOB, "
               " ADD COLUMN `bool_value` BOOLEAN; "
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        # check existing rows in previous Type table are migrated properly.
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ArtifactProperty` WHERE "
                 " `proto_value` IS NOT NULL; "
        } post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ArtifactProperty` WHERE "
                 " `bool_value` IS NOT NULL; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ExecutionProperty` WHERE "
                 " `proto_value` IS NOT NULL; "
        } post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ExecutionProperty` WHERE "
                 " `bool_value` IS NOT NULL; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ContextProperty` WHERE "
                 " `proto_value` IS NOT NULL; "
        } post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM `ContextProperty` WHERE "
                 " `bool_value` IS NOT NULL; "
        }
      }
      db_verification { total_num_indexes: 82 total_num_tables: 15 }
    }
  }
)pb");

const std::string kPostgreSQLMetadataSourceQueryConfig = absl::StrCat(  // NOLINT
R"pb(
  metadata_source_type: POSTGRESQL_METADATA_SOURCE
  drop_type_table { query: " DROP TABLE IF EXISTS Type; " }
  create_type_table {
    query: " CREATE TABLE IF NOT EXISTS Type( "
           "   id SERIAL PRIMARY KEY, "
           "   name VARCHAR(255) NOT NULL, "
           "   version VARCHAR(255), "
           "   type_kind SMALLINT NOT NULL, "
           "   description TEXT, "
           "   input_type TEXT, "
           "   output_type TEXT, "
           "   external_id VARCHAR(255) UNIQUE"
           " ); "
  }
  check_type_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'type'"
           "      AND column_name IN ('id', 'name', 'version', 'type_kind', "
           "                      'description', 'input_type', 'output_type')"
           "   ) = 7"
           " )::int AS table_exists;"
  }
  # After insertion, return id for the new inserted Type.
  insert_artifact_type {
    query: " INSERT INTO Type( "
           "   name, type_kind, version, description, external_id "
           ") VALUES($0, 1, $1, $2, $3)"
    parameter_num: 4
  }
  # After insertion, return id for the new inserted Type.
  insert_execution_type {
    query: " INSERT INTO Type( "
           "   name, type_kind, version, description, "
           "   input_type, output_type, external_id "
           ") VALUES($0, 0, $1, $2, $3, $4, $5)"
    parameter_num: 6
  }
  # After insertion, return id for the new inserted Type.
  insert_context_type {
    query: " INSERT INTO Type( "
           "   name, type_kind, version, description, external_id "
           ") VALUES($0, 2, $1, $2, $3)"
    parameter_num: 4
  }
  select_types_by_id {
    query: " SELECT id, name, version, description, external_id "
           " FROM Type "
           " WHERE id IN ($0) and type_kind = $1; "
    parameter_num: 2
  }
  select_type_by_id {
    query: " SELECT id, name, version, description, external_id, "
           "        input_type, output_type"
           " FROM Type "
           " WHERE id = $0 and type_kind = $1; "
    parameter_num: 2
  }
  select_type_by_name {
    query: " SELECT id, name, version, description,"
           "        external_id, input_type, output_type "
           " FROM Type "
           " WHERE name = $0 AND version IS NULL AND type_kind = $1; "
    parameter_num: 2
  }
  select_type_by_name_and_version {
    query: " SELECT id, name, version, description, external_id, "
           "        input_type, output_type FROM Type "
           " WHERE name = $0 AND version = $1 AND type_kind = $2; "
    parameter_num: 3
  }
  select_types_by_external_ids {
    query: " SELECT id, name, version, description, external_id "
           " FROM Type "
           " WHERE external_id IN ($0) and type_kind = $1; "
    parameter_num: 2
  }
  select_types_by_names {
    query: " SELECT id, name, version, description, "
           "        input_type, output_type"
           " FROM Type "
           " WHERE name IN ($0) AND version IS NULL AND type_kind = $1; "
    parameter_num: 2
  }
  select_types_by_names_and_versions {
    query: " SELECT id, name, version, description, "
           "        input_type, output_type"
           " FROM Type "
           " WHERE (name, version) IN ($0) AND type_kind = $1; "
    parameter_num: 2
  }
  select_all_types {
    query: " SELECT id, name, version, description, "
           "        input_type, output_type FROM Type "
           " WHERE type_kind = $0; "
    parameter_num: 1
  }
  update_type {
    query: " UPDATE Type "
           " SET external_id = $1 "
           " WHERE id = $0;"
    parameter_num: 2
  }
  drop_parent_type_table { query: " DROP TABLE IF EXISTS ParentType; " }
  create_parent_type_table {
    query: " CREATE TABLE IF NOT EXISTS ParentType ( "
           "   type_id INT NOT NULL, "
           "   parent_type_id INT NOT NULL, "
           " PRIMARY KEY (type_id, parent_type_id)); "
  }
  check_parent_type_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'parenttype'"
           "      AND column_name IN ('type_id', 'parent_type_id')"
           "   ) = 2"
           " )::int AS table_exists;"
  }
  insert_parent_type {
    query: " INSERT INTO ParentType(type_id, parent_type_id) "
           " VALUES($0, $1);"
    parameter_num: 2
  }
  delete_parent_type {
    query: " DELETE FROM ParentType "
           " WHERE type_id = $0 AND parent_type_id = $1;"
    parameter_num: 2
  }
  select_parent_type_by_type_id {
    query: " SELECT type_id, parent_type_id "
           " FROM ParentType WHERE type_id IN ($0); "
    parameter_num: 1
  }
  select_parent_contexts_by_context_ids {
    query: " SELECT context_id, parent_context_id From ParentContext "
           " WHERE context_id IN ($0); "
    parameter_num: 1
  }
  select_parent_contexts_by_parent_context_ids {
    query: " SELECT context_id, parent_context_id From ParentContext "
           " WHERE parent_context_id IN ($0); "
    parameter_num: 1
  }
  drop_type_property_table {
    query: " DROP TABLE IF EXISTS TypeProperty; "
  }
  create_type_property_table {
    query: " CREATE TABLE IF NOT EXISTS TypeProperty ( "
           "   type_id INT NOT NULL, "
           "   name VARCHAR(255) NOT NULL, "
           "   data_type INT NULL, "
           " PRIMARY KEY (type_id, name)); "
  }
  check_type_property_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'typeproperty'"
           "      AND column_name IN ('type_id', 'name', 'data_type')"
           "   ) = 3"
           " )::int AS table_exists;"
  }
  # After insertion, return id for the new inserted TypeProperty.
  insert_type_property {
    query: " INSERT INTO TypeProperty( "
           "   type_id, name, data_type "
           ") VALUES($0, $1, $2)"
    parameter_num: 3
  }
  select_properties_by_type_id {
    query: " SELECT type_id, name as key, data_type as value "
           " FROM TypeProperty WHERE type_id IN ($0); "
    parameter_num: 1
  }
  select_property_by_type_id {
    query: " SELECT name as key, data_type as value "
           " FROM TypeProperty "
           " WHERE type_id = $0; "
    parameter_num: 1
  }
  select_last_insert_id { query: " SELECT LASTVAL(); " }
)pb",
R"pb(
  drop_artifact_table { query: " DROP TABLE IF EXISTS Artifact; " }
  create_artifact_table {
    query: " CREATE TABLE IF NOT EXISTS Artifact ( "
           "   id SERIAL PRIMARY KEY, "
           "   type_id INT NOT NULL, "
           "   uri TEXT, "
           "   state INT, "
           "   name VARCHAR(255), "
           "   external_id VARCHAR(255) UNIQUE, "
           "   create_time_since_epoch BIGINT NOT NULL DEFAULT 0, "
           "   last_update_time_since_epoch BIGINT NOT NULL DEFAULT 0, "
           "   CONSTRAINT UniqueArtifactTypeName UNIQUE(type_id, name) "
           " ); "
  }
  check_artifact_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'artifact'"
           "      AND column_name IN ('id', 'type_id', 'uri',"
           "                'state', 'name', 'create_time_since_epoch',"
           "                'last_update_time_since_epoch')"
           "   ) = 7"
           " )::int AS table_exists;"
  }
  # After insertion, return id for the new inserted Artifact.
  insert_artifact {
    query: " INSERT INTO Artifact( "
           "   type_id, uri, state, name, external_id, "
           "   create_time_since_epoch, last_update_time_since_epoch "
           ") VALUES($0, $1, $2, $3, $4, $5, $6)"
    parameter_num: 7
  }
  select_artifact_by_id {
    query: " SELECT A.id, A.type_id, A.uri, A.state, A.name, "
           " A.external_id, A.create_time_since_epoch, "
           " A.last_update_time_since_epoch, "
           " T.name AS type, T.version AS type_version, "
           " T.description AS type_description, "
           " T.external_id AS type_external_id "
           " FROM Artifact AS A "
           " LEFT JOIN Type AS T "
           "    ON (T.id = A.type_id) "
           " WHERE A.id IN ($0); "
    parameter_num: 1
  }
  select_artifact_by_type_id_and_name {
    query: " SELECT id FROM Artifact WHERE type_id = $0 and name = $1; "
    parameter_num: 2
  }
  select_artifacts_by_type_id {
    query: " SELECT id FROM Artifact WHERE type_id = $0; "
    parameter_num: 1
  }
  select_artifacts_by_uri {
    query: " SELECT id FROM Artifact WHERE uri = $0; "
    parameter_num: 1
  }
  select_artifacts_by_external_ids {
    query: " SELECT id FROM Artifact WHERE external_id IN ($0); "
    parameter_num: 1
  }
  update_artifact {
    query: " UPDATE Artifact "
           " SET type_id = $1, uri = $2, state = $3, external_id = $4, "
           "     last_update_time_since_epoch = $5 "
           " WHERE id = $0;"
    parameter_num: 6
  }
  drop_artifact_property_table {
    query: " DROP TABLE IF EXISTS ArtifactProperty; "
  }
  create_artifact_property_table {
    query: " CREATE TABLE IF NOT EXISTS ArtifactProperty ( "
           "   artifact_id INT NOT NULL, "
           "   name VARCHAR(255) NOT NULL, "
           "   is_custom_property BOOLEAN NOT NULL, "
           "   int_value INT, "
           "   double_value DOUBLE PRECISION, "
           "   string_value TEXT, "
           "   byte_value BYTEA, "
           "   proto_value BYTEA, "
           "   bool_value BOOLEAN, "
           " PRIMARY KEY (artifact_id, name, is_custom_property)); "
  }
  check_artifact_property_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'artifactproperty'"
           "      AND column_name IN ('artifact_id', 'name', "
           "        'is_custom_property', 'int_value', 'double_value',"
           "        'string_value', 'byte_value', 'proto_value', 'bool_value')"
           "   ) = 9"
           " )::int AS table_exists;"
  }
  # After insertion, return id for the new inserted ArtifactProperty.
  insert_artifact_property {
    query: " INSERT INTO ArtifactProperty( "
           "   artifact_id, name, is_custom_property, $0 "
           ") VALUES($1, $2, $3, $4)"
    parameter_num: 5
  }
  select_artifact_property_by_artifact_id {
    query: " SELECT artifact_id as id, name as key, "
           "        is_custom_property, "
           "        int_value, double_value, string_value, "
           "        encode(proto_value, 'base64'), bool_value "
           " FROM ArtifactProperty "
           " WHERE artifact_id IN ($0); "
    parameter_num: 1
  }
  update_artifact_property {
    query: " UPDATE ArtifactProperty "
           " SET $0 = $1 "
           " WHERE artifact_id = $2 and name = $3;"
    parameter_num: 4
  }
  delete_artifact_property {
    query: " DELETE FROM ArtifactProperty "
           " WHERE artifact_id = $0 and name = $1;"
    parameter_num: 2
  }
  delete_artifacts_by_id {
    query: "DELETE FROM Artifact WHERE id IN ($0); "
    parameter_num: 1
  }
  delete_artifacts_properties_by_artifacts_id {
    query: "DELETE FROM ArtifactProperty WHERE artifact_id IN ($0); "
    parameter_num: 1
  }
)pb",
R"pb(
  drop_execution_table { query: " DROP TABLE IF EXISTS Execution; " }
  create_execution_table {
    query: " CREATE TABLE IF NOT EXISTS Execution ( "
           "   id SERIAL PRIMARY KEY, "
           "   type_id INT NOT NULL, "
           "   last_known_state INT, "
           "   name VARCHAR(255), "
           "   external_id VARCHAR(255) UNIQUE, "
           "   create_time_since_epoch BIGINT NOT NULL DEFAULT 0, "
           "   last_update_time_since_epoch BIGINT NOT NULL DEFAULT 0, "
           "   CONSTRAINT UniqueExecutionTypeName UNIQUE(type_id, name) "
           " ); "
  }
  check_execution_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'execution'"
           "      AND column_name IN ('id', 'type_id', "
           "        'last_known_state', 'name', 'create_time_since_epoch',"
           "        'last_update_time_since_epoch')"
           "   ) = 6"
           " )::int AS table_exists;"
  }
  insert_execution {
    query: " INSERT INTO Execution( "
           "   type_id, last_known_state, name, external_id, "
           "   create_time_since_epoch, last_update_time_since_epoch "
           ") VALUES($0, $1, $2, $3, $4, $5)"
    parameter_num: 6
  }
  select_execution_by_id {
    query: " SELECT E.id, E.type_id, E.last_known_state, E.name, "
           "        E.external_id, E.create_time_since_epoch, "
           "        E.last_update_time_since_epoch, T.name AS type, "
           "        T.version AS type_version, "
           "        T.description AS type_description, "
           "        T.external_id AS type_external_id "
           " FROM Execution AS E "
           " LEFT JOIN Type AS T "
           "        ON (T.id = E.type_id) "
           " WHERE E.id IN ($0);"
    parameter_num: 1
  }
  select_execution_by_type_id_and_name {
    query: " SELECT id FROM Execution WHERE type_id = $0 and name = $1;"
    parameter_num: 2
  }
  select_executions_by_type_id {
    query: " SELECT id FROM Execution WHERE type_id = $0; "
    parameter_num: 1
  }
  select_executions_by_external_ids {
    query: " SELECT id FROM Execution WHERE external_id IN ($0);"
    parameter_num: 1
  }
  update_execution {
    query: " UPDATE Execution "
           " SET type_id = $1, last_known_state = $2, "
           "     external_id = $3, "
           "     last_update_time_since_epoch = $4 "
           " WHERE id = $0;"
    parameter_num: 5
  }
  drop_execution_property_table {
    query: " DROP TABLE IF EXISTS ExecutionProperty; "
  }
  create_execution_property_table {
    query: " CREATE TABLE IF NOT EXISTS ExecutionProperty ( "
           "   execution_id INT NOT NULL, "
           "   name VARCHAR(255) NOT NULL, "
           "   is_custom_property BOOLEAN NOT NULL, "
           "   int_value INT, "
           "   double_value DOUBLE PRECISION, "
           "   string_value TEXT, "
           "   byte_value BYTEA, "
           "   proto_value BYTEA, "
           "   bool_value BOOLEAN, "
           " PRIMARY KEY (execution_id, name, is_custom_property)); "
  }
  check_execution_property_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'executionproperty'"
           "      AND column_name IN ('execution_id', 'name', "
           "        'is_custom_property', 'int_value', 'double_value',"
           "        'string_value', 'byte_value', 'proto_value', 'bool_value')"
           "   ) = 9"
           " )::int AS table_exists;"
  }
  insert_execution_property {
    query: " INSERT INTO ExecutionProperty( "
           "   execution_id, name, is_custom_property, $0 "
           ") VALUES($1, $2, $3, $4)"
    parameter_num: 5
  }
  select_execution_property_by_execution_id {
    query: " SELECT execution_id as id, name as key, "
           "        is_custom_property, "
           "        int_value, double_value, string_value, "
           "        encode(proto_value, 'base64'),"
           "        bool_value "
           " FROM ExecutionProperty "
           " WHERE execution_id IN ($0); "
    parameter_num: 1
  }
  update_execution_property {
    query: " UPDATE ExecutionProperty "
           " SET $0 = $1 "
           " WHERE execution_id = $2 and name = $3;"
    parameter_num: 4
  }
  delete_execution_property {
    query: " DELETE FROM ExecutionProperty "
           " WHERE execution_id = $0 and name = $1;"
    parameter_num: 2
  }
  delete_executions_by_id {
    query: " DELETE FROM Execution WHERE id IN ($0); "
    parameter_num: 1
  }
  delete_executions_properties_by_executions_id {
    query: " DELETE FROM ExecutionProperty WHERE execution_id IN ($0); "
    parameter_num: 1
  }
)pb",
R"pb(
  drop_context_table { query: " DROP TABLE IF EXISTS Context; " }
  create_context_table {
    query: " CREATE TABLE IF NOT EXISTS Context ( "
           "   id SERIAL PRIMARY KEY, "
           "   type_id INT NOT NULL, "
           "   name VARCHAR(255) NOT NULL, "
           "   external_id VARCHAR(255) UNIQUE, "
           "   create_time_since_epoch BIGINT NOT NULL DEFAULT 0, "
           "   last_update_time_since_epoch BIGINT NOT NULL DEFAULT 0, "
           "   UNIQUE(type_id, name) "
           " ); "
  }
  check_context_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'context'"
           "      AND column_name IN ('id', 'type_id', "
           "        'name', 'create_time_since_epoch', "
           "        'last_update_time_since_epoch')"
           "   ) = 5"
           " )::int AS table_exists;"
  }
  insert_context {
    query: " INSERT INTO Context( "
           "   type_id, name, external_id, "
           "   create_time_since_epoch, last_update_time_since_epoch "
           ") VALUES($0, $1, $2, $3, $4);"
    parameter_num: 5
  }
  select_context_by_id {
    query: " SELECT C.id, C.type_id, C.name, C.external_id, "
           " C.create_time_since_epoch, C.last_update_time_since_epoch, "
           " T.name AS type, T.version AS type_version, "
           " T.description AS type_description, "
           " T.external_id AS type_external_id "
           " FROM Context AS C "
           " LEFT JOIN Type AS T ON (T.id = C.type_id) "
           " WHERE C.id IN ($0); "
    parameter_num: 1
  }
  select_contexts_by_type_id {
    query: " SELECT id FROM Context WHERE type_id = $0; "
    parameter_num: 1
  }
  select_context_by_type_id_and_name {
    query: " SELECT id FROM Context WHERE type_id = $0 and name = $1; "
    parameter_num: 2
  }
  select_contexts_by_external_ids {
    query: " SELECT id FROM Context WHERE external_id IN ($0); "
    parameter_num: 1
  }
  update_context {
    query: " UPDATE Context "
           " SET type_id = $1, name = $2, external_id = $3, "
           "     last_update_time_since_epoch = $4 "
           " WHERE id = $0;"
    parameter_num: 5
  }
  drop_context_property_table {
    query: " DROP TABLE IF EXISTS ContextProperty; "
  }
  create_context_property_table {
    query: " CREATE TABLE IF NOT EXISTS ContextProperty ( "
           "   context_id INT NOT NULL, "
           "   name VARCHAR(255) NOT NULL, "
           "   is_custom_property BOOLEAN NOT NULL, "
           "   int_value INT, "
           "   double_value DOUBLE PRECISION, "
           "   string_value TEXT, "
           "   byte_value BYTEA, "
           "   proto_value BYTEA, "
           "   bool_value BOOLEAN, "
           " PRIMARY KEY (context_id, name, is_custom_property)); "
  }
  check_context_property_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'contextproperty'"
           "      AND column_name IN ('context_id', 'name', "
           "        'is_custom_property', 'int_value', 'double_value',"
           "        'string_value', 'byte_value', 'proto_value', 'bool_value')"
           "   ) = 9"
           " )::int AS table_exists;"
  }
  insert_context_property {
    query: " INSERT INTO ContextProperty( "
           "   context_id, name, is_custom_property, $0 "
           ") VALUES($1, $2, $3, $4)"
    parameter_num: 5
  }
  select_context_property_by_context_id {
    query: " SELECT context_id as id, name as key, "
           "        is_custom_property, "
           "        int_value, double_value, string_value, "
           "        encode(proto_value, 'base64'),"
           "        bool_value "
           " FROM ContextProperty "
           " WHERE context_id IN ($0); "
    parameter_num: 1
  }
  update_context_property {
    query: " UPDATE ContextProperty "
           " SET $0 = $1 "
           " WHERE context_id = $2 and name = $3;"
    parameter_num: 4
  }
  delete_context_property {
    query: " DELETE FROM ContextProperty "
           " WHERE context_id = $0 and name = $1;"
    parameter_num: 2
  }
  drop_parent_context_table {
    query: " DROP TABLE IF EXISTS ParentContext;"
  }
  create_parent_context_table {
    query: " CREATE TABLE IF NOT EXISTS ParentContext ( "
           "   context_id INT NOT NULL, "
           "   parent_context_id INT NOT NULL, "
           " PRIMARY KEY (context_id, parent_context_id)); "
  }
  check_parent_context_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'parentcontext'"
           "      AND column_name IN ('context_id', 'parent_context_id')"
           "   ) = 2"
           " )::int AS table_exists;"
  }
  insert_parent_context {
    query: " INSERT INTO ParentContext( "
           "   context_id, parent_context_id "
           ") VALUES($0, $1)"
    parameter_num: 2
  }
  select_parent_context_by_context_id {
    query: " SELECT context_id, parent_context_id From ParentContext "
           " WHERE context_id = $0; "
    parameter_num: 1
  }
  select_parent_context_by_parent_context_id {
    query: " SELECT context_id, parent_context_id From ParentContext "
           " WHERE parent_context_id = $0; "
    parameter_num: 1
  }
  delete_contexts_by_id {
    query: " DELETE FROM Context WHERE id IN ($0); "
    parameter_num: 1
  }
  delete_contexts_properties_by_contexts_id {
    query: " DELETE FROM ContextProperty WHERE context_id IN ($0); "
    parameter_num: 1
  }
  delete_parent_contexts_by_parent_ids {
    query: " DELETE FROM ParentContext WHERE parent_context_id IN ($0); "
    parameter_num: 1
  }
  delete_parent_contexts_by_child_ids {
    query: " DELETE FROM ParentContext WHERE context_id IN ($0); "
    parameter_num: 1
  }
  delete_parent_contexts_by_parent_id_and_child_ids {
    query: " DELETE FROM ParentContext "
           " WHERE parent_context_id = $0 AND context_id IN ($1); "
    parameter_num: 2
  }
)pb",
R"pb(
  drop_event_table { query: " DROP TABLE IF EXISTS Event; " }
  create_event_table {
    query: " CREATE TABLE IF NOT EXISTS Event ( "
           "   id SERIAL PRIMARY KEY, "
           "   artifact_id INT NOT NULL, "
           "   execution_id INT NOT NULL, "
           "   type INT NOT NULL, "
           "   milliseconds_since_epoch BIGINT, "
           "   CONSTRAINT UniqueEvent UNIQUE( "
           "     artifact_id, execution_id, type) "
           " ); "
  }
  check_event_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'event'"
           "      AND column_name IN ('id', 'artifact_id', "
           "        'execution_id', 'type', 'milliseconds_since_epoch')"
           "   ) = 5"
           " )::int AS table_exists;"
  }
  insert_event {
    query: " INSERT INTO Event( "
           "   artifact_id, execution_id, type, "
           "   milliseconds_since_epoch "
           ") VALUES($0, $1, $2, $3);"
    parameter_num: 4
  }
  select_event_by_artifact_ids {
    query: " SELECT id, artifact_id, execution_id, "
           "        type, milliseconds_since_epoch "
           " FROM Event "
           " WHERE artifact_id IN ($0); "
    parameter_num: 1
  }
  select_event_by_execution_ids {
    query: " SELECT id, artifact_id, execution_id, "
           "        type, milliseconds_since_epoch "
           " FROM Event "
           " WHERE execution_id IN ($0); "
    parameter_num: 1
  }
  drop_event_path_table { query: " DROP TABLE IF EXISTS EventPath; " }
  create_event_path_table {
    query: " CREATE TABLE IF NOT EXISTS EventPath ( "
           "   event_id INT NOT NULL, "
           "   is_index_step BOOLEAN NOT NULL, "
           "   step_index INT, "
           "   step_key TEXT "
           " ); "
  }
  check_event_path_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'eventpath'"
           "      AND column_name IN ('event_id', 'is_index_step', "
           "        'step_index', 'step_key')"
           "   ) = 4"
           " )::int AS table_exists;"
  }
  insert_event_path {
    query: " INSERT INTO EventPath( "
           "   event_id, is_index_step, $1 "
           ") VALUES($0, $2, $3);"
    parameter_num: 4
  }
  select_event_path_by_event_ids {
    query: " SELECT event_id, is_index_step, step_index, step_key "
           " FROM EventPath "
           " WHERE event_id IN ($0); "
    parameter_num: 1
  }
  delete_events_by_artifacts_id {
    query: " DELETE FROM Event WHERE artifact_id IN ($0); "
    parameter_num: 1
  }
  delete_events_by_executions_id {
    query: " DELETE FROM Event WHERE execution_id IN ($0); "
    parameter_num: 1
  }
  delete_event_paths {
    query: " DELETE FROM EventPath WHERE event_id NOT IN "
           " (SELECT id FROM Event); "
  }
)pb",
R"pb(
  drop_association_table { query: " DROP TABLE IF EXISTS Association; " }
  create_association_table {
    query: " CREATE TABLE IF NOT EXISTS Association ( "
           "   id SERIAL PRIMARY KEY, "
           "   context_id INT NOT NULL, "
           "   execution_id INT NOT NULL, "
           "   UNIQUE(context_id, execution_id) "
           " ); "
  }
  check_association_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'association'"
           "      AND column_name IN ('id', 'context_id', 'execution_id')"
           "   ) = 3"
           " )::int AS table_exists;"
  }
  insert_association {
    query: " INSERT INTO Association( "
           "   context_id, execution_id "
           ") VALUES($0, $1)"
    parameter_num: 2
  }
  select_association_by_context_id {
    query: " SELECT id, context_id, execution_id "
           " FROM Association "
           " WHERE context_id IN ($0); "
    parameter_num: 1
  }
  select_associations_by_execution_ids {
    query: " SELECT id, context_id, execution_id "
           " FROM Association "
           " WHERE execution_id IN ($0); "
    parameter_num: 1
  }
  drop_attribution_table { query: " DROP TABLE IF EXISTS Attribution; " }
  create_attribution_table {
    query: " CREATE TABLE IF NOT EXISTS Attribution ( "
           "   id SERIAL PRIMARY KEY, "
           "   context_id INT NOT NULL, "
           "   artifact_id INT NOT NULL, "
           "   UNIQUE(context_id, artifact_id) "
           " ); "
  }
  check_attribution_table {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'attribution'"
           "      AND column_name IN ('id', 'context_id', 'artifact_id')"
           "   ) = 3"
           " )::int AS table_exists;"
  }
  insert_attribution {
    query: " INSERT INTO Attribution( "
           "   context_id, artifact_id "
           ") VALUES($0, $1)"
    parameter_num: 2
  }
  select_attribution_by_context_id {
    query: " SELECT id, context_id, artifact_id "
           " FROM Attribution "
           " WHERE context_id = $0; "
    parameter_num: 1
  }
  select_attributions_by_artifact_ids {
    query: " SELECT id, context_id, artifact_id "
           " FROM Attribution "
           " WHERE artifact_id IN ($0); "
    parameter_num: 1
  }
  drop_mlmd_env_table { query: " DROP TABLE IF EXISTS MLMDEnv; " }
  create_mlmd_env_table {
    query: " CREATE TABLE IF NOT EXISTS MLMDEnv ( "
           "   schema_version INT PRIMARY KEY "
           " ); "
  }
  check_mlmd_env_table_existence {
    query: " SELECT (("
           "   SELECT COUNT(*)"
           "   FROM   information_schema.columns"
           "   WHERE  table_name = 'mlmdenv'"
           "      AND column_name IN ('schema_version')"
           "   ) = 1"
           " )::int AS table_exists;"
  }
  check_mlmd_env_table {
    query: "SELECT schema_version FROM MLMDEnv; "
  }
  # To avoid multiple rows in MLMDEnv, truncate table first.
  insert_schema_version {
    query: " TRUNCATE TABLE MLMDEnv; "
           " INSERT INTO MLMDEnv(schema_version) VALUES($0);"
    parameter_num: 1
  }
  # To avoid multiple rows in MLMDEnv, truncate table first.
  # Instead of UPDATE, use INSERT to update the schema version.
  update_schema_version {
    query: "TRUNCATE TABLE MLMDEnv;"
          " INSERT INTO MLMDEnv(schema_version) VALUES($0); "
    parameter_num: 1
  }
  check_tables_in_v0_13_2 {
    query: " SELECT COUNT(*) FROM information_schema.tables "
           " WHERE table_name "
           " IN ('Artifact', 'Event', 'Execution', 'Type', "
           " 'ArtifactProperty', 'EventPath', 'ExecutionProperty', "
           " 'TypeProperty');"
  }
  delete_associations_by_contexts_id {
    query: "DELETE FROM Association WHERE context_id IN ($0); "
    parameter_num: 1
  }
  delete_associations_by_executions_id {
    query: "DELETE FROM Association WHERE execution_id IN ($0); "
    parameter_num: 1
  }
  delete_attributions_by_contexts_id {
    query: "DELETE FROM Attribution WHERE context_id IN ($0); "
    parameter_num: 1
  }
  delete_attributions_by_artifacts_id {
    query: "DELETE FROM Attribution WHERE artifact_id IN ($0); "
    parameter_num: 1
  }
)pb",
R"pb(
  # secondary indices in the current schema.
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS idx_artifact_uri ON Artifact (uri); "
           " CREATE INDEX IF NOT EXISTS "
           "  idx_artifact_create_time_since_epoch "
           "  ON Artifact (create_time_since_epoch); "
           " CREATE INDEX IF NOT EXISTS "
           "  idx_artifact_last_update_time_since_epoch "
           "  ON Artifact (last_update_time_since_epoch); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_event_execution_id "
           "  ON Event (execution_id); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_parentcontext_parent_context_id "
           "  ON ParentContext (parent_context_id); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS idx_type_name ON Type (name); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_execution_create_time_since_epoch "
           "  ON Execution (create_time_since_epoch); "
           " CREATE INDEX IF NOT EXISTS "
           "  idx_execution_last_update_time_since_epoch "
           "  ON Execution (last_update_time_since_epoch); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_context_create_time_since_epoch "
           "  ON Context (create_time_since_epoch); "
           " CREATE INDEX IF NOT EXISTS "
           "  idx_context_last_update_time_since_epoch "
           "  ON Context (last_update_time_since_epoch); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_eventpath_event_id "
           "  ON EventPath (event_id); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_artifact_property_int "
           "  ON ArtifactProperty (name, is_custom_property, int_value); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_artifact_property_double "
           "  ON ArtifactProperty (name, is_custom_property, double_value); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_artifact_property_string "
           "  ON ArtifactProperty (name, is_custom_property, string_value); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_execution_property_int "
           "  ON ExecutionProperty (name, is_custom_property, int_value); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_execution_property_double "
           "  ON ExecutionProperty (name, is_custom_property, double_value); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_execution_property_string "
           "  ON ExecutionProperty (name, is_custom_property, string_value); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_context_property_int "
           "  ON ContextProperty (name, is_custom_property, int_value); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_context_property_double "
           "  ON ContextProperty (name, is_custom_property, double_value); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS "
           "  idx_context_property_string "
           "  ON ContextProperty (name, is_custom_property, string_value); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS idx_type_external_id "
           " ON Type(external_id); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS idx_artifact_external_id "
           " ON Artifact(external_id); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS idx_execution_external_id "
           " ON Execution(external_id); "
  }
  secondary_indices {
    query: " CREATE INDEX IF NOT EXISTS idx_context_external_id "
           " ON Context(external_id); "
  }
)pb",
R"pb(
  # PostgreSQL doesn't have migration schema for version 0, because MLMDEnv
  # table doesn't exist in this version 0.13.2. It will cause failure in
  # verifySchema because MLMD will suspect that it is an empty DB. It is okay
  # to omit version 0 for migration validation because PostgreSQL was added in
  # later schema.
)pb",
R"pb(
  migration_schemes {
    key: 1
    value: {
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS MLMDEnv ( "
               "   schema_version INT PRIMARY KEY "
               " ); "
      }
      # v0.13.2 release
      upgrade_verification {
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS Type ( "
                 "   id SERIAL PRIMARY KEY, "
                 "   name VARCHAR(255) NOT NULL, "
                 "   is_artifact_type SMALLINT NOT NULL, "
                 "   input_type TEXT, "
                 "   output_type TEXT"
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS TypeProperty ( "
                 "   type_id INT NOT NULL, "
                 "   name VARCHAR(255) NOT NULL, "
                 "   data_type INT NULL, "
                 " PRIMARY KEY (type_id, name)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS Artifact ( "
                 "   id SERIAL PRIMARY KEY, "
                 "   type_id INT NOT NULL, "
                 "   uri TEXT "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS ArtifactProperty ( "
                 "   artifact_id INT NOT NULL, "
                 "   name VARCHAR(255) NOT NULL, "
                 "   is_custom_property SMALLINT NOT NULL, "
                 "   int_value INT, "
                 "   double_value DOUBLE PRECISION, "
                 "   string_value TEXT, "
                 " PRIMARY KEY (artifact_id, name, is_custom_property)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS Execution ( "
                 "   id SERIAL PRIMARY KEY, "
                 "   type_id INT NOT NULL "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS ExecutionProperty ( "
                 "   execution_id INT NOT NULL, "
                 "   name VARCHAR(255) NOT NULL, "
                 "   is_custom_property SMALLINT NOT NULL, "
                 "   int_value INT, "
                 "   double_value DOUBLE PRECISION, "
                 "   string_value TEXT, "
                 " PRIMARY KEY (execution_id, name, is_custom_property)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS Event ( "
                 "   id SERIAL PRIMARY KEY, "
                 "   artifact_id INT NOT NULL, "
                 "   execution_id INT NOT NULL, "
                 "   type INT NOT NULL, "
                 "   milliseconds_since_epoch BIGINT "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS EventPath ( "
                 "   event_id INT NOT NULL, "
                 "   is_index_step SMALLINT NOT NULL, "
                 "   step_index INT, "
                 "   step_key TEXT "
                 " ); "
        }
        # check the new table has 1 row
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM MLMDEnv; "
        }
      }
      # downgrade queries from version 2, drop all ContextTypes and rename
      # the type_kind back to is_artifact_type column.
      downgrade_queries {
        query: " DELETE FROM Type WHERE type_kind = 2; "
      }
      downgrade_queries {
        query: " ALTER TABLE Type "
               " RENAME COLUMN type_kind TO is_artifact_type;"
      }
      # check the tables are deleted properly
      downgrade_verification {
        previous_version_setup_queries {
          # To prevent ID serial increment to go out-of-sync,
          # reset the id sequence at: https://stackoverflow.com/q/4448340
          query: "SELECT SETVAL((SELECT PG_GET_SERIAL_SEQUENCE('Type', 'id')),"
                 "      (SELECT (MAX(id) + 1) FROM Type), FALSE);"
        }
        # populate the Type table with context types.
        previous_version_setup_queries {
          query: " INSERT INTO Type "
                 " (name, type_kind, input_type, output_type) "
                 " VALUES ('execution_type', 0, 'input', 'output')"
        }
        previous_version_setup_queries {
          query: " INSERT INTO Type "
                 " (name, type_kind, input_type, output_type) "
                 " VALUES ('artifact_type', 1, 'input', 'output')"
        }
        previous_version_setup_queries {
          query: " INSERT INTO Type "
                 " (name, type_kind, input_type, output_type) "
                 " VALUES ('context_type', 2, 'input', 'output')"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM Type "
                 " WHERE is_artifact_type = 2; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM Type "
                 " WHERE is_artifact_type = 1 AND name = 'artifact_type'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM Type "
                 " WHERE is_artifact_type = 0 AND name = 'execution_type'; "
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
        query: " ALTER TABLE Type "
               " RENAME COLUMN is_artifact_type TO type_kind;"
      }
      upgrade_verification {
        # FROMV1: BEGIN
        # Copy V0 to V1 upgrade logic
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS MLMDEnv ( "
                "   schema_version INT PRIMARY KEY "
                " ); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO MLMDEnv(schema_version) VALUES(0)"
                 " ON CONFLICT DO NOTHING; "
        }
        # Skip V0 setup logic to avoid transaction failure
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS Type ( "
                 "   id SERIAL PRIMARY KEY, "
                 "   name VARCHAR(255) NOT NULL, "
                 "   is_artifact_type SMALLINT NOT NULL, "
                 "   input_type TEXT, "
                 "   output_type TEXT"
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS TypeProperty ( "
                 "   type_id INT NOT NULL, "
                 "   name VARCHAR(255) NOT NULL, "
                 "   data_type INT NULL, "
                 " PRIMARY KEY (type_id, name)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS Artifact ( "
                 "   id SERIAL PRIMARY KEY, "
                 "   type_id INT NOT NULL, "
                 "   uri TEXT "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS ArtifactProperty ( "
                 "   artifact_id INT NOT NULL, "
                 "   name VARCHAR(255) NOT NULL, "
                 "   is_custom_property SMALLINT NOT NULL, "
                 "   int_value INT, "
                 "   double_value DOUBLE PRECISION, "
                 "   string_value TEXT, "
                 " PRIMARY KEY (artifact_id, name, is_custom_property)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS Execution ( "
                 "   id SERIAL PRIMARY KEY, "
                 "   type_id INT NOT NULL "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS ExecutionProperty ( "
                 "   execution_id INT NOT NULL, "
                 "   name VARCHAR(255) NOT NULL, "
                 "   is_custom_property SMALLINT NOT NULL, "
                 "   int_value INT, "
                 "   double_value DOUBLE PRECISION, "
                 "   string_value TEXT, "
                 " PRIMARY KEY (execution_id, name, is_custom_property)); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS Event ( "
                 "   id SERIAL PRIMARY KEY, "
                 "   artifact_id INT NOT NULL, "
                 "   execution_id INT NOT NULL, "
                 "   type INT NOT NULL, "
                 "   milliseconds_since_epoch BIGINT "
                 " ); "
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS EventPath ( "
                 "   event_id INT NOT NULL, "
                 "   is_index_step SMALLINT NOT NULL, "
                 "   step_index INT, "
                 "   step_key TEXT "
                 " ); "
        }
        # FROMV1: END
        # populate one ArtifactType and one ExecutionType.
        previous_version_setup_queries {
          query: " INSERT INTO Type (name, is_artifact_type) VALUES "
                 " ('artifact_type', 1);"
        }
        previous_version_setup_queries {
          # is_artifact_type* to type_kind
          query: " INSERT INTO Type "
                 " (name, is_artifact_type, input_type, output_type) "
                 " VALUES ('execution_type', 0, 'input', 'output');"
        }
        # check after migration, the existing types are the same including
        # id.
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM Type WHERE "
                 " id = 1 AND type_kind = 1 AND name = 'artifact_type'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM Type WHERE "
                 " id = 2 AND type_kind = 0 AND name = 'execution_type' "
                 " AND input_type = 'input' AND output_type = 'output'; "
        }
      }
      # downgrade queries from version 3
      downgrade_queries { query: " DROP TABLE IF EXISTS Context; " }
      downgrade_queries {
        query: " DROP TABLE IF EXISTS ContextProperty; "
      }
      # check the tables are deleted properly
      downgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM information_schema.tables "
                 " WHERE table_schema = 'public' and "
                 "       table_name = 'Context'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM information_schema.tables "
                 " WHERE table_schema = 'public' and "
                 "       table_name = 'ContextProperty'; "
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
        query: " CREATE TABLE IF NOT EXISTS Context ( "
               "   id SERIAL PRIMARY KEY, "
               "   type_id INT NOT NULL, "
               "   name VARCHAR(255) NOT NULL, "
               "   UNIQUE(type_id, name) "
               " ); "
      }
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS ContextProperty ( "
               "   context_id INT NOT NULL, "
               "   name VARCHAR(255) NOT NULL, "
               "   is_custom_property SMALLINT NOT NULL, "
               "   int_value INT, "
               "   double_value DOUBLE PRECISION, "
               "   string_value TEXT, "
               " PRIMARY KEY (context_id, name, is_custom_property)); "
      }
      upgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT id, type_id, name FROM Context "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT context_id, name, is_custom_property, "
                 "          int_value, double_value, string_value "
                 "    FROM ContextProperty "
                 " ) as T2; "
        }
      }
      # downgrade queries from version 4
      downgrade_queries { query: " DROP TABLE IF EXISTS Association; " }
      downgrade_queries { query: " DROP TABLE IF EXISTS Attribution; " }
      # check the tables are deleted properly
      downgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM information_schema.tables "
                 " WHERE table_schema = 'public' and "
                 "       table_name = 'Association'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM information_schema.tables "
                 " WHERE table_schema = 'public' and "
                 "       table_name = 'Attribution'; "
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
        query: " CREATE TABLE IF NOT EXISTS Association ( "
               "   id SERIAL PRIMARY KEY, "
               "   context_id INT NOT NULL, "
               "   execution_id INT NOT NULL, "
               "   UNIQUE(context_id, execution_id) "
               " ); "
      }
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS Attribution ( "
               "   id SERIAL PRIMARY KEY, "
               "   context_id INT NOT NULL, "
               "   artifact_id INT NOT NULL, "
               "   UNIQUE(context_id, artifact_id) "
               " ); "
      }
      upgrade_verification {
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT id, context_id, execution_id "
                 "   FROM Association "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT id, context_id, artifact_id "
                 "   FROM Attribution "
                 " ) as T1; "
        }
      }
      # downgrade queries from version 5
      downgrade_queries {
        query: " ALTER TABLE Artifact DROP CONSTRAINT UniqueArtifactTypeName; "
      }
      downgrade_queries {
        query: " ALTER TABLE Artifact "
               " DROP COLUMN state, "
               " DROP COLUMN name, "
               " DROP COLUMN create_time_since_epoch, "
               " DROP COLUMN last_update_time_since_epoch; "
      }
      downgrade_queries {
        query: " ALTER TABLE Execution DROP CONSTRAINT UniqueExecutionTypeName; "
      }
      downgrade_queries {
        query: " ALTER TABLE Execution "
               " DROP COLUMN last_known_state, "
               " DROP COLUMN name, "
               " DROP COLUMN create_time_since_epoch, "
               " DROP COLUMN last_update_time_since_epoch; "
      }
      downgrade_queries {
        query: " ALTER TABLE Context "
               " DROP COLUMN create_time_since_epoch, "
               " DROP COLUMN last_update_time_since_epoch; "
      }
      # verify if the downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM Artifact;" }
        previous_version_setup_queries {
          query: " INSERT INTO Artifact "
                 " (id, type_id, uri, state, name, "
                 "  create_time_since_epoch, last_update_time_since_epoch) "
                 " VALUES (1, 2, 'uri1', 1, NULL, 0, 1)"
                 " ON CONFLICT DO NOTHING;"
        }
        previous_version_setup_queries {
          query: " INSERT INTO Artifact "
                 " (id, type_id, uri, state, name, "
                 "  create_time_since_epoch, last_update_time_since_epoch) "
                 " VALUES (2, 3, 'uri2', NULL, 'name2', 1, 0)"
                 " ON CONFLICT DO NOTHING;"
        }
        previous_version_setup_queries { query: "DELETE FROM Execution;" }
        previous_version_setup_queries {
          query: " INSERT INTO Execution "
                 " (id, type_id, last_known_state, name, "
                 "  create_time_since_epoch, last_update_time_since_epoch) "
                 " VALUES (1, 2, 1, NULL, 0, 1)"
                 " ON CONFLICT DO NOTHING;"
        }
        previous_version_setup_queries { query: "DELETE FROM Context;" }
        previous_version_setup_queries {
          query: " INSERT INTO Context "
                 " (id, type_id, name, "
                 "  create_time_since_epoch, last_update_time_since_epoch) "
                 " VALUES (1, 2, 'name1', 1, 0)"
                 " ON CONFLICT DO NOTHING;"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 2 FROM Artifact; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM Artifact "
                 "   WHERE id = 1 and type_id = 2 and uri = 'uri1' "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM Artifact "
                 "   WHERE id = 2 and type_id = 3 and uri = 'uri2' "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM Execution; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM Execution "
                 "   WHERE id = 1 and type_id = 2 "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM Context "
                 "   WHERE id = 1 and type_id = 2 "
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
        query: "ALTER TABLE Artifact "
                "ADD COLUMN state INT,"
                "ADD COLUMN name VARCHAR(255),"
                "ADD COLUMN create_time_since_epoch BIGINT NOT NULL DEFAULT 0,"
                "ADD COLUMN "
                " last_update_time_since_epoch BIGINT NOT NULL DEFAULT 0,"
                "ADD CONSTRAINT UniqueArtifactTypeName UNIQUE (type_id, name);"
      }
      # upgrade Execution table
      upgrade_queries {
        query: "ALTER TABLE Execution "
            "ADD COLUMN last_known_state INT,"
            "ADD COLUMN name VARCHAR(255),"
            "ADD COLUMN create_time_since_epoch BIGINT NOT NULL DEFAULT 0,"
            "ADD COLUMN last_update_time_since_epoch BIGINT NOT NULL DEFAULT 0,"
            "ADD CONSTRAINT UniqueExecutionTypeName UNIQUE (type_id, name);"
      }
      # upgrade Context table
      upgrade_queries {
        query: "ALTER TABLE Context "
            "ADD COLUMN create_time_since_epoch BIGINT NOT NULL DEFAULT 0,"
            "ADD COLUMN last_update_time_since_epoch BIGINT NOT NULL DEFAULT 0;"
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM Artifact;" }
        previous_version_setup_queries {
          query: " INSERT INTO Artifact "
                 " (id, type_id, uri) VALUES (1, 2, 'uri1');"
        }
        previous_version_setup_queries { query: "DELETE FROM Execution;" }
        previous_version_setup_queries {
          query: " INSERT INTO Execution "
                 " (id, type_id) VALUES (1, 3);"
        }
        previous_version_setup_queries {
          query: " CREATE TABLE IF NOT EXISTS Context ( "
                 "   id SERIAL PRIMARY KEY, "
                 "   type_id INT NOT NULL, "
                 "   name VARCHAR(255) NOT NULL, "
                 "   UNIQUE(type_id, name) "
                 " ); "
        }
        previous_version_setup_queries {
          query: " INSERT INTO Context "
                 " (id, type_id, name) VALUES (1, 2, 'name1');"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT id, type_id, uri, state, name, "
                 "          create_time_since_epoch, "
                 "          last_update_time_since_epoch "
                 "   FROM Artifact "
                 "   WHERE id = 1 AND type_id = 2 AND uri = 'uri1' AND "
                 "         create_time_since_epoch = 0 AND "
                 "         last_update_time_since_epoch = 0 "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT id, type_id, last_known_state, name, "
                 "          create_time_since_epoch, "
                 "          last_update_time_since_epoch "
                 "   FROM Execution "
                 "   WHERE id = 1 AND type_id = 3 AND "
                 "         create_time_since_epoch = 0 AND "
                 "         last_update_time_since_epoch = 0 "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT id, type_id, name, "
                 "          create_time_since_epoch, "
                 "          last_update_time_since_epoch "
                 "   FROM Context "
                 "   WHERE id = 1 AND type_id = 2 AND name = 'name1' AND "
                 "         create_time_since_epoch = 0 AND "
                 "         last_update_time_since_epoch = 0 "
                 " ) as T1; "
        }
      }
      # downgrade queries from version 6
      downgrade_queries { query: " DROP TABLE ParentType; " }
      downgrade_queries { query: " DROP TABLE ParentContext; " }
      downgrade_queries {
        query: " ALTER TABLE Type "
               " DROP COLUMN version, "
               " DROP COLUMN description; "
      }
      downgrade_queries {
        query: # " ALTER TABLE Artifact "
               " DROP INDEX idx_artifact_uri, "
               " idx_artifact_create_time_since_epoch, "
               " idx_artifact_last_update_time_since_epoch; "
      }
      downgrade_queries {
        query: # " ALTER TABLE Event "
               " DROP INDEX idx_event_artifact_id, "
               " idx_event_execution_id; "
      }
      downgrade_queries {
        query: # " ALTER TABLE Type "
               " DROP INDEX idx_type_name; "
      }
      downgrade_queries {
        query: # " ALTER TABLE Execution "
               " DROP INDEX idx_execution_create_time_since_epoch, "
               " idx_execution_last_update_time_since_epoch; "
      }
      downgrade_queries {
        query: # " ALTER TABLE Context "
               " DROP INDEX idx_context_create_time_since_epoch, "
               " idx_context_last_update_time_since_epoch; "
      }
      # verify if the downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM Type;" }
        previous_version_setup_queries {
          query: " INSERT INTO Type "
                 " (id, name, version, type_kind, "
                 "  description, input_type, output_type) "
                 " VALUES (1, 't1', 'v1', 1, 'desc1', 'input1', 'output1')"
                 " ON CONFLICT DO NOTHING;"
        }
        previous_version_setup_queries {
          query: " INSERT INTO Type "
                 " (id, name, version, type_kind, "
                 "  description, input_type, output_type) "
                 " VALUES (2, 't2', 'v2', 2, 'desc2', 'input2', 'output2')"
                 " ON CONFLICT DO NOTHING;"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 2 FROM Type; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ( "
                 "   SELECT * FROM Type "
                 "   WHERE id = 1 AND name = 't1' AND type_kind = 1 "
                 "   AND input_type = 'input1' AND output_type = 'output1'"
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM information_schema.tables "
                 " WHERE table_schema = 'public' and "
                 "       table_name = 'ParentType'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM information_schema.tables "
                 " WHERE table_schema = 'public' and "
                 "       table_name = 'ParentContext'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM pg_indexes"
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Artifact' AND "
                 "       indexname LIKE 'idx_artifact_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Event' AND "
                 "       indexname LIKE 'idx_event_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'ParentContext' AND "
                 "       indexname LIKE 'idx_parentcontext_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Type' AND "
                 "       indexname LIKE 'idx_type_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Execution' AND "
                 "       indexname LIKE 'idx_execution_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Context' AND "
                 "       indexname LIKE 'idx_context_%'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v6, to support parental type and parental context, we added two
  # tables ParentType and ParentContext. In addition, we added version
  # and description in the Type table for improving type registrations.
  # We introduce indices on Type.name, Artifact.uri, Event's artifact_id and
  # execution_id, and create_time_since_epoch, last_update_time_since_epoch
  # for all nodes.
  migration_schemes {
    key: 6
    value: {
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS ParentType ( "
               "   type_id INT NOT NULL, "
               "   parent_type_id INT NOT NULL, "
               " PRIMARY KEY (type_id, parent_type_id)); "
      }
      upgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS ParentContext ( "
               "   context_id INT NOT NULL, "
               "   parent_context_id INT NOT NULL, "
               " PRIMARY KEY (context_id, parent_context_id)); "
      }
      upgrade_queries {
        query: "ALTER TABLE Type "
                "ADD COLUMN version VARCHAR(255),"
                "ADD COLUMN description TEXT;"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_artifact_uri "
               "  ON Artifact (uri);" # remove (uri(255))
               " CREATE INDEX idx_artifact_create_time_since_epoch "
               "  ON Artifact (create_time_since_epoch);"
               " CREATE INDEX idx_artifact_last_update_time_since_epoch "
               "  ON Artifact (last_update_time_since_epoch);"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_event_artifact_id ON Event (artifact_id);"
               " CREATE INDEX idx_event_execution_id ON Event (execution_id);"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_parentcontext_parent_context_id "
               "  ON ParentContext (parent_context_id);"
      }
      upgrade_queries {
        query: "CREATE INDEX idx_type_name ON Type (name);"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_execution_create_time_since_epoch "
               "  ON Execution (create_time_since_epoch);"
               " CREATE INDEX idx_execution_last_update_time_since_epoch "
               "  ON Execution (last_update_time_since_epoch);"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_context_create_time_since_epoch "
               "  ON Context (create_time_since_epoch);"
               " CREATE INDEX idx_context_last_update_time_since_epoch "
               "  ON Context (last_update_time_since_epoch);"
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        # check existing rows in previous Type table are migrated properly.
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM Type WHERE "
                 " id = 1 AND type_kind = 1 AND name = 'artifact_type'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM Type WHERE "
                 " id = 2 AND type_kind = 0 AND name = 'execution_type' "
                 " AND input_type = 'input' AND output_type = 'output'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT type_id, parent_type_id "
                 "   FROM ParentType "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ( "
                 "   SELECT context_id, parent_context_id "
                 "   FROM ParentContext "
                 " ) as T1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Artifact' AND "
                 "       indexname = 'idx_artifact_uri'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Artifact' AND "
                 "       indexname = 'idx_artifact_create_time_since_epoch';"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Artifact' AND indexname = "
                 "       'idx_artifact_last_update_time_since_epoch';"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Event' AND "
                 "       indexname = 'idx_event_artifact_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Event' AND "
                 "       indexname = 'idx_event_execution_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'ParentContext' AND "
                 "       indexname = 'idx_parentcontext_parent_context_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Type' AND "
                 "       indexname = 'idx_type_name'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Execution' AND indexname = "
                 "       'idx_execution_create_time_since_epoch'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Execution' AND indexname = "
                 "       'idx_execution_last_update_time_since_epoch'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Context' AND "
                 "       indexname = 'idx_context_create_time_since_epoch'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Context' AND indexname = "
                 "       'idx_context_last_update_time_since_epoch'; "
        }
      }
      # downgrade queries from version 7
      # Note v7 for MySQL used mediumtext for string_value, when downgrade
      # the long text will be truncated to 65536 chars.
      downgrade_queries {
        query: " ALTER TABLE ArtifactProperty DROP COLUMN byte_value; "
      }
      downgrade_queries {
        query: " ALTER TABLE ExecutionProperty DROP COLUMN byte_value; "
      }
      downgrade_queries {
        query: " ALTER TABLE ContextProperty DROP COLUMN byte_value; "
      }
      downgrade_queries {
        query: #" ALTER TABLE EventPath"
               " DROP INDEX idx_eventpath_event_id; "
      }
      downgrade_queries {
        query: " UPDATE ArtifactProperty "
               " SET string_value = SUBSTRING(string_value, 1, 65535); "
      }
      downgrade_queries {
        query: " ALTER TABLE ArtifactProperty "
               " ALTER COLUMN string_value TYPE TEXT; "
      }
      downgrade_queries {
        query: " UPDATE ExecutionProperty "
               " SET string_value = SUBSTRING(string_value, 1, 65535); "
      }
      downgrade_queries {
        query: " ALTER TABLE ExecutionProperty "
               " ALTER COLUMN string_value TYPE TEXT; "
      }
      downgrade_queries {
        query: " UPDATE ContextProperty "
               " SET string_value = SUBSTRING(string_value, 1, 65535); "
      }
      downgrade_queries {
        query: " ALTER TABLE ContextProperty "
               " ALTER COLUMN string_value TYPE TEXT; "
      }
      # verify if the downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries {
          query: "DELETE FROM ArtifactProperty;"
        }
        previous_version_setup_queries {
          query: "DELETE FROM ExecutionProperty;"
        }
        previous_version_setup_queries {
          query: "DELETE FROM ContextProperty;"
        }
        previous_version_setup_queries {
          query: " INSERT INTO ArtifactProperty "
                 " (artifact_id, name, string_value, is_custom_property) "
                 " VALUES "
                 "  (1, 'p1', CONCAT('_prefix_', REPEAT('a', 160000)), false), "
                 "  (1, 'p2', 'abc', false)"
        }
        previous_version_setup_queries {
          query: " INSERT INTO ExecutionProperty "
                 " (execution_id, name, string_value, is_custom_property) "
                 " VALUES "
                 "  (1, 'p1', CONCAT('_prefix_', REPEAT('e', 160000)), false), "
                 "  (1, 'p2', 'abc', false)"
        }
        previous_version_setup_queries {
          query: " INSERT INTO ContextProperty "
                 " (context_id, name, string_value, is_custom_property) "
                 " VALUES "
                 "  (1, 'p1', CONCAT('_prefix_', REPEAT('c', 160000)), false), "
                 "  (1, 'p2', 'abc', false)"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'EventPath' AND "
                 "       indexname = 'idx_eventpath_event_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 3 FROM information_schema.columns "
                 " WHERE table_schema = 'public' AND "
                 "       column_name = 'string_value' AND "
                 "       data_type = 'text'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM information_schema.columns "
                 " WHERE table_schema = 'public' AND "
                 "       table_name IN ('ArtifactProperty', "
                 "           'ExecutionProperty', 'ContextProperty') AND "
                 "       column_name = 'byte_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ArtifactProperty "
                 " WHERE artifact_id = 1 AND name = 'p1' AND "
                 "   string_value = CONCAT('_prefix_', REPEAT('a', 65527)); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ArtifactProperty "
                 " WHERE artifact_id = 1 AND name = 'p2' AND "
                 "       string_value = 'abc'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ExecutionProperty "
                 " WHERE execution_id = 1 AND name = 'p1' AND "
                 "   string_value = CONCAT('_prefix_', REPEAT('e', 65527)); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ExecutionProperty "
                 " WHERE execution_id = 1 AND name = 'p2' AND "
                 "       string_value = 'abc'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ContextProperty "
                 " WHERE context_id = 1 AND name = 'p1' AND "
                 "   string_value = CONCAT('_prefix_', REPEAT('c', 65527)); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ContextProperty "
                 " WHERE context_id = 1 AND name = 'p2' AND "
                 "       string_value = 'abc'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v7, we added byte_value for property tables for better storing binary
  # property values. For MySQL, we extends string_value column to be
  # MEDIUMTEXT in order to persist string value with size upto 16MB. In
  # addition, we added index for EventPath to improve Event reads.
  migration_schemes {
    key: 7
    value: {
      upgrade_queries {
        query: " ALTER TABLE ArtifactProperty "
               " ALTER COLUMN string_value TYPE TEXT, "
               " ADD COLUMN byte_value BYTEA; "
      }
      upgrade_queries {
        query: " ALTER TABLE ExecutionProperty "
               " ALTER COLUMN string_value TYPE TEXT, "
               " ADD COLUMN byte_value BYTEA; "
      }
      upgrade_queries {
        query: " ALTER TABLE ContextProperty "
               " ALTER COLUMN string_value TYPE TEXT, "
               " ADD COLUMN byte_value BYTEA; "
      }
      upgrade_queries {
        query: "CREATE INDEX idx_eventpath_event_id ON EventPath (event_id);"
      }
      # check the expected table columns are created properly.
      upgrade_verification {
        # check existing rows in previous Type table are migrated properly.
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ArtifactProperty WHERE "
                 " byte_value IS NOT NULL; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ExecutionProperty WHERE "
                 " byte_value IS NOT NULL; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM ContextProperty WHERE "
                 " byte_value IS NOT NULL; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'EventPath' AND "
                 "       indexname = 'idx_eventpath_event_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 3 FROM information_schema.columns "
                 " WHERE table_schema = 'public' AND "
                 "       table_name IN ('ArtifactProperty', "
                 "           'ExecutionProperty', 'ContextProperty') AND "
                 "       column_name = 'string_value' AND "
                 "       data_type = 'mediumtext'; "
        }
      }
      db_verification { total_num_indexes: 39 total_num_tables: 15 }
      # downgrade queries from version 8
      downgrade_queries {
        query: " ALTER TABLE Event DROP CONSTRAINT UniqueEvent; "
      }
      downgrade_queries {
        query: "CREATE INDEX idx_event_artifact_id ON Event (artifact_id);"
      }
      downgrade_queries {
        # " ALTER TABLE ArtifactProperty "
        query: " DROP INDEX idx_artifact_property_int; "
      }
      downgrade_queries {
        # " ALTER TABLE ArtifactProperty "
        query: " DROP INDEX idx_artifact_property_double; "
      }
      downgrade_queries {
        # " ALTER TABLE ArtifactProperty "
        query: " DROP INDEX idx_artifact_property_string; "
      }
      downgrade_queries {
        # " ALTER TABLE ExecutionProperty "
        query: " DROP INDEX idx_execution_property_int; "
      }
      downgrade_queries {
        # " ALTER TABLE ExecutionProperty "
        query: " DROP INDEX idx_execution_property_double; "
      }
      downgrade_queries {
        # " ALTER TABLE ExecutionProperty "
        query: " DROP INDEX idx_execution_property_string; "
      }
      downgrade_queries {
        # " ALTER TABLE ContextProperty "
        query: " DROP INDEX idx_context_property_int; "
      }
      downgrade_queries {
        #" ALTER TABLE ContextProperty "
        query: " DROP INDEX idx_context_property_double; "
      }
      downgrade_queries {
        # " ALTER TABLE ContextProperty "
        query: " DROP INDEX idx_context_property_string; "
      }
      downgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM Event;" }
        previous_version_setup_queries {
          query: " INSERT INTO Event "
                 " (id, artifact_id, execution_id, type, "
                 " milliseconds_since_epoch) "
                 " VALUES (1, 1, 1, 1, 1); "
        }
        previous_version_setup_queries { query: "DELETE FROM EventPath;" }
        previous_version_setup_queries {
          query: " INSERT INTO EventPath "
                 " (event_id, is_index_step, step_index, step_key) "
                 " VALUES (1, true, 1, 'a'); "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM Event "
                 " WHERE artifact_id = 1 AND execution_id = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM EventPath "
                 " WHERE event_id = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE "
                 "       schemaname='public' AND "
                 "       indexname = 'idx_event_artifact_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       indexname = 'idx_event_execution_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       indexname = 'idx_eventpath_event_id'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       indexname LIKE 'idx_artifact_property_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       indexname LIKE 'idx_execution_property_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       indexname LIKE 'idx_context_property_%'; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v8, we added index for ArtifactProperty, ExecutionProperty,
  # ContextProperty to improve property queries on name.
  migration_schemes {
    key: 8
    value: {
      upgrade_queries {
        query: " CREATE INDEX idx_artifact_property_int "
               "  ON ArtifactProperty (name, is_custom_property, int_value);"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_artifact_property_double "
               "  ON ArtifactProperty (name, is_custom_property, double_value);"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_artifact_property_string "
               "  ON ArtifactProperty (name, is_custom_property, string_value);"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_execution_property_int "
               "  ON ExecutionProperty (name, is_custom_property, int_value);"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_execution_property_double "
               "  ON ExecutionProperty "
               "   (name, is_custom_property, double_value);"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_execution_property_string "
               "  ON ExecutionProperty "
               "   (name, is_custom_property, string_value);" 
      }
      upgrade_queries {
        query: " CREATE INDEX idx_context_property_int "
               "  ON ContextProperty (name, is_custom_property, int_value);"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_context_property_double "
               "  ON ContextProperty (name, is_custom_property, double_value);"
      }
      upgrade_queries {
        query: " CREATE INDEX idx_context_property_string "
               "  ON ContextProperty (name, is_custom_property, string_value);"
      }
      upgrade_queries {
        query: " CREATE TABLE EventTemp ( "
               "   id SERIAL PRIMARY KEY, "
               "   artifact_id INT NOT NULL, "
               "   execution_id INT NOT NULL, "
               "   type INT NOT NULL, "
               "   milliseconds_since_epoch BIGINT, "
               "   CONSTRAINT UniqueEvent "
               "     UNIQUE(artifact_id, execution_id, type) "
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO EventTemp "
               " (id, artifact_id, execution_id, type, "
               " milliseconds_since_epoch) "
               " SELECT * FROM Event ORDER BY id desc"
               " ON CONFLICT DO NOTHING"
      }
      upgrade_queries { query: " DROP TABLE Event; " }
      upgrade_queries {
        query: " ALTER TABLE EventTemp RENAME TO Event; "
      }
      upgrade_queries {
        query: "CREATE INDEX idx_event_execution_id ON Event (execution_id);"
      }
      upgrade_queries {
        query: " DELETE FROM EventPath "
               "   WHERE event_id not in ( SELECT id from Event ) "
      }
      # check the expected indexes are created properly.
      upgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM Event;" }
        previous_version_setup_queries {
          query: " INSERT INTO Event "
                 " (id, artifact_id, execution_id, type, "
                 " milliseconds_since_epoch) "
                 " VALUES (1, 1, 1, 1, 1)"
        }
        previous_version_setup_queries {
          query: " INSERT INTO Event "
                 " (id, artifact_id, execution_id, type, "
                 " milliseconds_since_epoch) "
                 " VALUES (2, 1, 1, 1, 2)"
        }
        previous_version_setup_queries { query: "DELETE FROM EventPath;" }
        previous_version_setup_queries {
          query: " INSERT INTO EventPath "
                 " (event_id, is_index_step, step_index, step_key) "
                 " VALUES (1, 1, 1, 'a');"
        }
        previous_version_setup_queries {
          query: " INSERT INTO EventPath "
                 " (event_id, is_index_step, step_index, step_key) "
                 " VALUES (2, 1, 1, 'b');"
        }
        previous_version_setup_queries {
          query: " INSERT INTO EventPath "
                 " (event_id, is_index_step, step_index, step_key) "
                 " VALUES (2, 1, 2, 'c');"
        }
        # check event table unique constraint is applied and event path
        # records are deleted.
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM Event "
                 " WHERE artifact_id = 1 AND execution_id = 1 "
                 "     AND type = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM Event "
                 "   WHERE id = 2 AND artifact_id = 1 AND  "
                 "       execution_id = 1 AND type = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 2 FROM EventPath "
                 " WHERE event_id = 2; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM EventPath "
                 " WHERE event_id = 2 AND step_key = 'b'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM EventPath "
                 " WHERE event_id = 2 AND step_key = 'c'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM EventPath "
                 " WHERE event_id = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 "
                 " FROM information_schema.table_constraints"
                 " WHERE table_schema='public' AND "
                 "       table_name = 'Event' AND "
                 "       constraint_name = 'UniqueEvent'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'Event' AND "
                 "       indexname LIKE 'idx_event_execution_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'EventPath' AND "
                 "       indexname LIKE 'idx_eventpath_event_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 9 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'ArtifactProperty' AND "
                 "       indexname LIKE 'idx_artifact_property_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 9 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'ExecutionProperty' AND "
                 "       indexname LIKE 'idx_execution_property_%'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 9 FROM pg_indexes "
                 " WHERE schemaname='public' AND "
                 "       tablename = 'ContextProperty' AND "
                 "       indexname LIKE 'idx_context_property_%'; "
        }
      }
      db_verification { total_num_indexes: 74 total_num_tables: 15 }
    }
  }
)pb",

R"pb(
  # In v9, to store the ids that come from the clients' system (like Vertex
  # Metadata), we added a new column external_id in the Type \
  # Artifacrt \ Execution \ Context tables. We introduce unique and
  # null-filtered indices on Type.external_id, Artifact.external_id,
  # Execution's external_id and Context's external_id.
  migration_schemes {
    key: 9
    value: {
      upgrade_queries {
        query: " CREATE TABLE TypeTemp ( "
               "   id SERIAL PRIMARY KEY, "
               "   name VARCHAR(255) NOT NULL, "
               "   version VARCHAR(255), "
               "   type_kind SMALLINT NOT NULL, "
               "   description TEXT, "
               "   input_type TEXT, "
               "   output_type TEXT, "
               "   external_id VARCHAR(255) UNIQUE"
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO TypeTemp (id, name, version, type_kind, "
               "        description, input_type, output_type) "
               " SELECT id, name, version, type_kind, description,"
               "        input_type, output_type "
               " FROM Type"
               " ON CONFLICT DO NOTHING;"
      }
      upgrade_queries { query: " DROP TABLE Type; " }
      upgrade_queries {
        query: " ALTER TABLE TypeTemp rename to Type; "
      }
      upgrade_queries {
        query: " CREATE TABLE ArtifactTemp ( "
               "   id SERIAL PRIMARY KEY, "
               "   type_id INT NOT NULL, "
               "   uri TEXT, "
               "   state INT, "
               "   name VARCHAR(255), "
               "   external_id VARCHAR(255) UNIQUE, "
               "   create_time_since_epoch INT NOT NULL DEFAULT 0, "
               "   last_update_time_since_epoch INT NOT NULL DEFAULT 0, "
               "   UNIQUE(type_id, name) "
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO ArtifactTemp (id, type_id, uri, state, "
               "        name, create_time_since_epoch, "
               "        last_update_time_since_epoch) "
               " SELECT id, type_id, uri, state, name, "
               "        create_time_since_epoch, "
               "        last_update_time_since_epoch "
               "FROM Artifact"
               " ON CONFLICT DO NOTHING;"
      }
      upgrade_queries { query: " DROP TABLE Artifact; " }
      upgrade_queries {
        query: " ALTER TABLE ArtifactTemp RENAME TO Artifact; "
      }
      upgrade_queries {
        query: " CREATE TABLE ExecutionTemp ( "
               "   id SERIAL PRIMARY KEY, "
               "   type_id INT NOT NULL, "
               "   last_known_state INT, "
               "   name VARCHAR(255), "
               "   external_id VARCHAR(255) UNIQUE, "
               "   create_time_since_epoch INT NOT NULL DEFAULT 0, "
               "   last_update_time_since_epoch INT NOT NULL DEFAULT 0, "
               "   UNIQUE(type_id, name) "
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO ExecutionTemp (id, type_id, "
               "        last_known_state, name, "
               "        create_time_since_epoch, "
               "        last_update_time_since_epoch) "
               " SELECT id, type_id, last_known_state, name, "
               "        create_time_since_epoch, "
               "        last_update_time_since_epoch "
               " FROM Execution"
               " ON CONFLICT DO NOTHING;"
      }
      upgrade_queries { query: " DROP TABLE Execution; " }
      upgrade_queries {
        query: " ALTER TABLE ExecutionTemp RENAME TO Execution; "
      }
      upgrade_queries {
        query: " CREATE TABLE ContextTemp ( "
               "   id SERIAL PRIMARY KEY, "
               "   type_id INT NOT NULL, "
               "   name VARCHAR(255) NOT NULL, "
               "   external_id VARCHAR(255) UNIQUE, "
               "   create_time_since_epoch INT NOT NULL DEFAULT 0, "
               "   last_update_time_since_epoch INT NOT NULL DEFAULT 0, "
               "   UNIQUE(type_id, name) "
               " ); "
      }
      upgrade_queries {
        query: " INSERT INTO ContextTemp (id, type_id, name, "
               "        create_time_since_epoch, "
               "        last_update_time_since_epoch) "
               " SELECT id, type_id, name, "
               "        create_time_since_epoch, "
               "        last_update_time_since_epoch "
               " FROM Context"
               " ON CONFLICT DO NOTHING;"
      }
      upgrade_queries { query: " DROP TABLE Context; " }
      upgrade_queries {
        query: " ALTER TABLE ContextTemp RENAME TO Context; "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_artifact_uri "
               " ON Artifact(uri); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   idx_artifact_create_time_since_epoch "
               " ON Artifact(create_time_since_epoch); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   idx_artifact_last_update_time_since_epoch "
               " ON Artifact(last_update_time_since_epoch); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_type_name "
               " ON Type(name); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   idx_execution_create_time_since_epoch "
               " ON Execution(create_time_since_epoch); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   idx_execution_last_update_time_since_epoch "
               " ON Execution(last_update_time_since_epoch); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   idx_context_create_time_since_epoch "
               " ON Context(create_time_since_epoch); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS "
               "   idx_context_last_update_time_since_epoch "
               " ON Context(last_update_time_since_epoch); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_type_external_id "
               " ON Type(external_id); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_artifact_external_id "
               " ON Artifact(external_id); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_execution_external_id "
               " ON Execution(external_id); "
      }
      upgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_context_external_id "
               " ON Context(external_id); "
      }
      # check the expected table columns are created properly.
      # table type is using the old schema for upgrade verification, which
      # contains is_artifact_type column
      upgrade_verification {
        previous_version_setup_queries { query: "DELETE FROM Type;" }
        previous_version_setup_queries {
          query: " INSERT INTO Type (name, is_artifact_type) VALUES "
                 " ('artifact_type', 1)"
                 " ON CONFLICT DO NOTHING;"
        }
        previous_version_setup_queries { query: "DELETE FROM Artifact;" }
        previous_version_setup_queries {
          query: " INSERT INTO Artifact "
                 " (id, type_id) "
                 " VALUES (1, 2)"
                 " ON CONFLICT DO NOTHING;"
        }
        previous_version_setup_queries { query: "DELETE FROM Execution;" }
        previous_version_setup_queries {
          query: " INSERT INTO Execution "
                 " (id, type_id) "
                 " VALUES (1, 2)"
                 " ON CONFLICT DO NOTHING;"
        }
        previous_version_setup_queries { query: "DELETE FROM Context;" }
        previous_version_setup_queries {
          query: " INSERT INTO Context "
                 " (id, type_id, name) "
                 " VALUES (1, 2, 'name1');"
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM Type; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM Type "
                 "   WHERE name = 'artifact_type' AND "
                 "         external_id IS NULL "
                 " ) AS T1; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM Artifact; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM Artifact "
                 "   WHERE id = 1 AND type_id = 2 AND "
                 "         external_id IS NULL "
                 " ) AS T1; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM Execution; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM Execution "
                 "   WHERE id = 1 AND type_id = 2 AND "
                 "          external_id IS NULL "
                 " ) AS T1; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM Context; "
        }
        post_migration_verification_queries {
          query: " SELECT COUNT(*) = 1 FROM ( "
                 "   SELECT * FROM Context "
                 "   WHERE id = 1 AND type_id = 2 AND name = 'name1' AND "
                 "         external_id IS NULL "
                 " ) as T1; "
        }
      }
      db_verification { total_num_indexes: 48 total_num_tables: 15 }
      # downgrade queries from version 10
      downgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS ArtifactPropertyTemp ( "
               "   artifact_id INT NOT NULL, "
               "   name VARCHAR(255) NOT NULL, "
               "   is_custom_property BOOLEAN NOT NULL, "
               "   int_value INT, "
               "   double_value DOUBLE PRECISION, "
               "   string_value TEXT, "
               "   byte_value BYTEA, "
               " PRIMARY KEY (artifact_id, name, is_custom_property)); "
      }
      downgrade_queries {
        query: " INSERT INTO ArtifactPropertyTemp  "
               " SELECT artifact_id, name,  is_custom_property, "
               "        int_value, double_value, string_value, "
               "        byte_value "
               " FROM ArtifactProperty; "
      }
      downgrade_queries { query: " DROP TABLE ArtifactProperty; " }
      downgrade_queries {
        query: " ALTER TABLE ArtifactPropertyTemp "
               "  RENAME TO ArtifactProperty; "
      }
      downgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS ExecutionPropertyTemp ( "
               "   execution_id INT NOT NULL, "
               "   name VARCHAR(255) NOT NULL, "
               "   is_custom_property BOOLEAN NOT NULL, "
               "   int_value INT, "
               "   double_value DOUBLE PRECISION, "
               "   string_value TEXT, "
               "   byte_value BYTEA, "
               " PRIMARY KEY (execution_id, name, is_custom_property)); "
      }
      downgrade_queries {
        query: " INSERT INTO ExecutionPropertyTemp "
               " SELECT execution_id, name,  is_custom_property, "
               "     int_value, double_value, string_value, "
               "     byte_value "
               " FROM ExecutionProperty; "
      }
      downgrade_queries { query: " DROP TABLE ExecutionProperty; " }
      downgrade_queries {
        query: " ALTER TABLE ExecutionPropertyTemp "
               "  RENAME TO ExecutionProperty; "
      }
      downgrade_queries {
        query: " CREATE TABLE IF NOT EXISTS ContextPropertyTemp ( "
               "   context_id INT NOT NULL, "
               "   name VARCHAR(255) NOT NULL, "
               "   is_custom_property BOOLEAN NOT NULL, "
               "   int_value INT, "
               "   double_value DOUBLE PRECISION, "
               "   string_value TEXT, "
               "   byte_value BYTEA, "
               " PRIMARY KEY (context_id, name, is_custom_property)); "
      }
      downgrade_queries {
        query: " INSERT INTO ContextPropertyTemp "
               " SELECT context_id, name,  is_custom_property, "
               "        int_value, double_value, string_value, "
               "        byte_value "
               " FROM ContextProperty; "
      }
      downgrade_queries { query: " DROP TABLE ContextProperty; " }
      downgrade_queries {
        query: " ALTER TABLE ContextPropertyTemp "
               "  RENAME TO ContextProperty; "
      }
      # recreate the indices that were dropped along with the old tables
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_artifact_property_int "
               " ON ArtifactProperty(name, is_custom_property, "
               " int_value) "
               " WHERE int_value IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_artifact_property_double "
               " ON ArtifactProperty(name, is_custom_property, "
               " double_value) "
               " WHERE double_value IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_artifact_property_string "
               " ON ArtifactProperty(name, is_custom_property, "
               " string_value) "
               " WHERE string_value IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_execution_property_int "
               " ON ExecutionProperty(name, is_custom_property, "
               " int_value) "
               " WHERE int_value IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_execution_property_double "
               " ON ExecutionProperty(name, is_custom_property, "
               " double_value) "
               " WHERE double_value IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_execution_property_string "
               " ON ExecutionProperty(name, is_custom_property, "
               " string_value) "
               " WHERE string_value IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_context_property_int "
               " ON ContextProperty(name, is_custom_property, "
               " int_value) "
               " WHERE int_value IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_context_property_double "
               " ON ContextProperty(name, is_custom_property, "
               " double_value) "
               " WHERE double_value IS NOT NULL; "
      }
      downgrade_queries {
        query: " CREATE INDEX IF NOT EXISTS idx_context_property_string "
               " ON ContextProperty(name, is_custom_property, "
               " string_value) "
               " WHERE string_value IS NOT NULL; "
      }
)pb",
R"pb(
      # verify that downgrading keeps the existing columns
      downgrade_verification {
        previous_version_setup_queries {
          query: "DELETE FROM ArtifactProperty;"
        }
        previous_version_setup_queries {
          query: "DELETE FROM ExecutionProperty;"
        }
        previous_version_setup_queries {
          query: "DELETE FROM ContextProperty;"
        }
        previous_version_setup_queries {
          query: " INSERT INTO ArtifactProperty (artifact_id, "
                 "     is_custom_property, name, string_value) "
                 " VALUES (1, false, 'p1', 'abc')"
        }
        previous_version_setup_queries {
          query: " INSERT INTO ExecutionProperty (execution_id, "
                 "     is_custom_property, name, int_value) "
                 " VALUES (1, true, 'p1', 1)"
        }
        previous_version_setup_queries {
          query: " INSERT INTO ContextProperty (context_id, "
                 "     is_custom_property, name, double_value) "
                 " VALUES (1, false, 'p1', 1.0)"
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        information_schema.columns "
                 " WHERE table_name = 'ArtifactProperty'"
                 "   AND column_name = 'proto_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        information_schema.columns "
                 " WHERE table_name = 'ArtifactProperty'"
                 "   AND column_name = 'bool_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        information_schema.columns "
                 " WHERE table_name = 'ExecutionProperty'"
                 "   AND column_name = 'proto_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        information_schema.columns "
                 " WHERE table_name = 'ExecutionProperty'"
                 "   AND column_name = 'bool_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        information_schema.columns "
                 " WHERE table_name = 'ContextProperty'"
                 "   AND column_name = 'proto_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 0 FROM "
                 "        information_schema.columns "
                 " WHERE table_name = 'ContextProperty'"
                 "   AND column_name = 'bool_value'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ArtifactProperty "
                 " WHERE artifact_id = 1 AND is_custom_property = false AND "
                 "       name = 'p1' AND string_value = 'abc'; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ExecutionProperty "
                 " WHERE execution_id = 1 AND is_custom_property = true AND "
                 "        name = 'p1' AND int_value = 1; "
        }
        post_migration_verification_queries {
          query: " SELECT count(*) = 1 FROM ContextProperty "
                 " WHERE context_id = 1  AND is_custom_property = false AND "
                 "        name = 'p1' AND double_value = 1.0; "
        }
      }
    }
  }
)pb",
R"pb(
  # In v10, we added proto_value and bool_value columns to {X}Property tables
  migration_schemes {
    key: 10
    value: {
      upgrade_queries {
        query: " ALTER TABLE ArtifactProperty "
               " ADD COLUMN proto_value BYTEA; "
      }
      upgrade_queries {
        query: " ALTER TABLE ArtifactProperty "
               " ADD COLUMN bool_value BOOLEAN; "
      }
      upgrade_queries {
        query: " ALTER TABLE ExecutionProperty "
               " ADD COLUMN proto_value BYTEA; "
      }
      upgrade_queries {
        query: " ALTER TABLE ExecutionProperty "
               " ADD COLUMN bool_value BOOLEAN; "
      }
      upgrade_queries {
        query: " ALTER TABLE ContextProperty "
               " ADD COLUMN proto_value BYTEA;"
      }
      upgrade_queries {
        query: " ALTER TABLE ContextProperty "
               " ADD COLUMN bool_value BOOLEAN; "
      }
      db_verification { total_num_indexes: 48 total_num_tables: 15 }
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
  CHECK(google::protobuf::TextFormat::ParseFromString(kBaseQueryConfig, &config));
  MetadataSourceQueryConfig mysql_config;
  CHECK(google::protobuf::TextFormat::ParseFromString(kMySQLMetadataSourceQueryConfig,
                                            &mysql_config));
  config.MergeFrom(mysql_config);
  return config;
}

MetadataSourceQueryConfig GetSqliteMetadataSourceQueryConfig() {
  MetadataSourceQueryConfig config;
  CHECK(google::protobuf::TextFormat::ParseFromString(kBaseQueryConfig, &config));
  MetadataSourceQueryConfig sqlite_config;
  CHECK(google::protobuf::TextFormat::ParseFromString(kSQLiteMetadataSourceQueryConfig,
                                            &sqlite_config));
  config.MergeFrom(sqlite_config);
  return config;
}

MetadataSourceQueryConfig GetPostgreSQLMetadataSourceQueryConfig() {
  MetadataSourceQueryConfig config;
  CHECK(google::protobuf::TextFormat::ParseFromString(kBaseQueryConfig, &config));
  MetadataSourceQueryConfig postgres_config;
  CHECK(google::protobuf::TextFormat::ParseFromString(
    kPostgreSQLMetadataSourceQueryConfig, &postgres_config));
  config.MergeFrom(postgres_config);
  return config;
}


}  // namespace util
}  // namespace ml_metadata
