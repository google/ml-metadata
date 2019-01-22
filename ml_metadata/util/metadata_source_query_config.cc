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

#include "ml_metadata/proto/metadata_source.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace ml_metadata {
namespace util {
namespace {

// A set of common template queries used by the MetadataAccessObject for SQLite
// based MetadataSource.
constexpr char kBaseQueryConfig[] = R"pb(
  drop_type_table { query: " DROP TABLE IF EXISTS `Type`; " }
  create_type_table {
    query: " CREATE TABLE IF NOT EXISTS `Type` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `is_artifact_type` TINYINT(1) NOT NULL "
           " ); "
  }
  check_type_table {
    query: " SELECT `id`, `name`, `is_artifact_type` "
           " FROM `Type` LIMIT 1; "
  }
  insert_type {
    query: " INSERT INTO `Type`( "
           "   `name`, `is_artifact_type` "
           ") VALUES($0, $1);"
    parameter_num: 2
  }
  select_type_by_id {
    query: " SELECT `id`, `name` "
           " from `Type` "
           " WHERE id = $0 and is_artifact_type = $1; "
    parameter_num: 2
  }
  select_type_by_name {
    query: " SELECT `id`, `name` "
           " from `Type` "
           " WHERE name = $0 and is_artifact_type = $1; "
    parameter_num: 2
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
  drop_artifact_table { query: " DROP TABLE IF EXISTS `Artifact`; " }
  create_artifact_table {
    query: " CREATE TABLE IF NOT EXISTS `Artifact` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `uri` TEXT "
           " ); "
  }
  check_artifact_table {
    query: " SELECT `id`, `type_id`, `uri` "
           " FROM `Artifact` LIMIT 1; "
  }
  insert_artifact {
    query: " INSERT INTO `Artifact`( "
           "   `type_id`, `uri` "
           ") VALUES($0, $1);"
    parameter_num: 2
  }
  select_artifact_by_id {
    query: " SELECT `type_id`, `uri` "
           " from `Artifact` "
           " WHERE id = $0; "
    parameter_num: 1
  }
  update_artifact {
    query: " UPDATE `Artifact` "
           " SET `type_id` = $1, `uri` = $2"
           " WHERE id = $0;"
    parameter_num: 3
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
  drop_execution_table { query: " DROP TABLE IF EXISTS `Execution`; " }
  create_execution_table {
    query: " CREATE TABLE IF NOT EXISTS `Execution` ( "
           "   `id` INTEGER PRIMARY KEY AUTOINCREMENT, "
           "   `type_id` INT NOT NULL "
           " ); "
  }
  check_execution_table {
    query: " SELECT `id`, `type_id` "
           " FROM `Execution` LIMIT 1; "
  }
  insert_execution {
    query: " INSERT INTO `Execution`( "
           "   `type_id` "
           ") VALUES($0);"
    parameter_num: 1
  }
  select_execution_by_id {
    query: " SELECT `type_id` "
           " from `Execution` "
           " WHERE id = $0; "
    parameter_num: 1
  }
  update_execution {
    query: " UPDATE `Execution` "
           " SET `type_id` = $1 "
           " WHERE id = $0;"
    parameter_num: 2
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
  select_event_by_artifact_id {
    query: " SELECT `id`, `artifact_id`, `execution_id`, "
           "        `type`, `milliseconds_since_epoch` "
           " from `Event` "
           " WHERE `artifact_id` = $0; "
    parameter_num: 1
  }
  select_event_by_execution_id {
    query: " SELECT `id`, `artifact_id`, `execution_id`, "
           "        `type`, `milliseconds_since_epoch` "
           " from `Event` "
           " WHERE `execution_id` = $0; "
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
  select_event_path_by_event_id {
    query: " SELECT `event_id`, `is_index_step`, `step_index`, `step_key` "
           " from `EventPath` "
           " WHERE `event_id` = $0; "
    parameter_num: 1
  }
)pb";

constexpr char kFakeMetadataSourceQueryConfig[] = R"pb(
  metadata_source_type: FAKE_METADATA_SOURCE
)pb";

constexpr char kSQLiteMetadataSourceQueryConfig[] = R"pb(
  metadata_source_type: SQLITE_METADATA_SOURCE
)pb";

// Template queries for  MySQLMetadataSources.
constexpr char kMySQLMetadataSourceQueryConfig[] = R"pb(
  metadata_source_type: MYSQL_METADATA_SOURCE
  select_last_insert_id { query: " SELECT last_insert_id(); " }
  create_type_table {
    query: " CREATE TABLE IF NOT EXISTS `Type` ( "
           "   `id` INT PRIMARY KEY AUTO_INCREMENT, "
           "   `name` VARCHAR(255) NOT NULL, "
           "   `is_artifact_type` TINYINT(1) NOT NULL "
           " ); "
  }
  create_artifact_table {
    query: " CREATE TABLE IF NOT EXISTS `Artifact` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `type_id` INT NOT NULL, "
           "   `uri` TEXT "
           " ); "
  }
  create_execution_table {
    query: " CREATE TABLE IF NOT EXISTS `Execution` ( "
           "   `id` INTEGER PRIMARY KEY AUTO_INCREMENT, "
           "   `type_id` INT NOT NULL "
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
)pb";

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

MetadataSourceQueryConfig GetFakeMetadataSourceQueryConfig() {
  MetadataSourceQueryConfig config;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(kBaseQueryConfig,
                                                          &config));
  MetadataSourceQueryConfig fakedb_config;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      kFakeMetadataSourceQueryConfig, &fakedb_config));
  config.MergeFrom(fakedb_config);
  return config;
}

}  // namespace util
}  // namespace ml_metadata
