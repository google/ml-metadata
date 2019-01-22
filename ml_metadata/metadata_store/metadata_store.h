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
#ifndef ML_METADATA_METADATA_STORE_METADATA_STORE_H_
#define ML_METADATA_METADATA_STORE_METADATA_STORE_H_

#include <memory>

#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// A metadata store.
// Implements the API specified in MetadataStoreService.
// Each method is an atomic operation.
class MetadataStore {
 public:
  // Factory method, if the return value is ok, 'result' is populated with an
  // object that can be used to access metadata with the given config and
  // metadata_source.
  // Returns INVALID_ARGUMENT error, if query_config is not valid.
  // Returns detailed INTERNAL error, if the MetadataSource cannot be connected.
  // Creates a MetadataStore in result.
  // result is owned by the caller.
  // metadata_source is owned by result.
  static tensorflow::Status Create(
      const MetadataSourceQueryConfig& query_config,
      std::unique_ptr<MetadataSource> metadata_source,
      std::unique_ptr<MetadataStore>* result);

  // Initializes the metadata source and creates schema. Any existing data in
  // the metadata is dropped.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status InitMetadataStore();

  // Initializes the metadata source and creates schema if it does not exist.
  // Returns OK and does nothing, if all required schema exist.
  // Returns OK and creates schema, if no schema exists yet.
  // Returns DATA_LOSS error, if any required schema is missing.
  // Returns detailed INTERNAL error, if create schema query execution fails.
  tensorflow::Status InitMetadataStoreIfNotExists();

  // Inserts or updates an artifact type.
  //
  // If no artifact type exists in the database with the given name, it creates
  // a new artifact type and returns the type_id.
  //
  // If an artifact type with the same name already exists (let's call it
  // old_artifact_type),
  //   If there is a property where artifact_type and old_artifact_type
  //     have different types, or artifact_type and old_artifact_type
  //     have different properties, it fails and returns ALREADY_EXISTS.
  //
  // Otherwise, it returns the type_id of old_artifact_type.
  // For the fields of PutArtifactTypeRequest:
  //   artifact_type: the type to add: type_id must not be present.
  //   can_add_fields must be false (otherwise returns UNIMPLEMENTED).
  //   all_fields_match must be true (otherwise returns UNIMPLEMENTED).
  //   can_delete_fields must be false (otherwise returns UNIMPLEMENTED).
  tensorflow::Status PutArtifactType(const PutArtifactTypeRequest& request,
                                     PutArtifactTypeResponse* response);

  // Inserts or updates an execution type.
  //
  // If no execution type exists in the database with the given name, it creates
  // a new execution type and returns the type_id.
  //
  // If an execution type with the same name already exists (let's call it
  // old_execution_type), it compares the given properties in execution_type
  // with the properties in old_execution_type. If there is a property where
  // execution_type and old_execution_type have different types, or
  // execution_type and old_execution_type have different properties, it fails
  // and returns ALREADY_EXISTS.
  // Otherwise, it returns the type_id of old_execution_type.
  //
  // For the fields of PutExecutionTypeRequest:
  //   execution_type: the type to add: type_id must not be present.
  //   can_add_fields must be false (otherwise returns UNIMPLEMENTED).
  //   all_fields_match must be true  (otherwise returns UNIMPLEMENTED).
  //   can_delete_fields must be false  (otherwise returns UNIMPLEMENTED).
  tensorflow::Status PutExecutionType(const PutExecutionTypeRequest& request,
                                      PutExecutionTypeResponse* response);

  // Gets an execution type by name.
  // If no type exists, returns a NOT_FOUND error.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status GetExecutionType(const GetExecutionTypeRequest& request,
                                      GetExecutionTypeResponse* response);

  // Gets an artifact type by name.
  // If no type exists, returns a NOT_FOUND error.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status GetArtifactType(const GetArtifactTypeRequest& request,
                                     GetArtifactTypeResponse* response);

  // Inserts and updates artifacts in request into the database.
  // If artifact_id is specified, an existing artifact is updated.
  // If artifact_id is not specified, a new artifact is created.
  //
  // response is a list of artifact ids index-aligned with the input.
  // Returns INVALID_ARGUMENT error, if no artifact is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ArtifactType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status PutArtifacts(const PutArtifactsRequest& request,
                                  PutArtifactsResponse* response);

  // Inserts and updates executions in the request into the database.
  // If execution_id is specified, an existing execution is updated.
  // If execution_id is not specified, a new execution is created.
  //
  // Returns a list of execution ids index-aligned with the input.
  // Returns INVALID_ARGUMENT error, if no execution is found with a given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ExecutionType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status PutExecutions(const PutExecutionsRequest& request,
                                   PutExecutionsResponse* response);

  // Gets a list of artifacts by ID.
  // If no artifact with an ID exists, the artifact is skipped.
  // Sets the error field if any other internal errors are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status GetArtifactsByID(const GetArtifactsByIDRequest& request,
                                      GetArtifactsByIDResponse* response);

  // Gets a list of executions by ID.
  // If no execution with an ID exists, the execution is skipped.
  // Sets the error field if any other internal errors are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status GetExecutionsByID(const GetExecutionsByIDRequest& request,
                                       GetExecutionsByIDResponse* response);

  // Gets a list of artifact types by ID.
  // If no artifact types with an ID exists, the artifact type is skipped.
  // Sets the error field if any other internal errors are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status GetArtifactTypesByID(
      const GetArtifactTypesByIDRequest& request,
      GetArtifactTypesByIDResponse* response);

  // Gets a list of execution types by ID.
  // If no artifact types with an ID exists, the artifact type is skipped.
  // Sets the error field if any other internal errors are returned.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status GetExecutionTypesByID(
      const GetExecutionTypesByIDRequest& request,
      GetExecutionTypesByIDResponse* response);

  // Inserts events into the database.
  //
  // The execution_id and artifact_id must already exist.
  // Once created, events cannot be modified.
  // If timestamp is not set, it will be set to the current time.
  // request.events is a list of events to insert or update.
  // response is empty.
  // Returns INVALID_ARGUMENT error, if no artifact matches the artifact_id.
  // Returns INVALID_ARGUMENT error, if no execution matches the execution_id.
  // Returns INVALID_ARGUMENT error, if the type field is UNKNOWN.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status PutEvents(const PutEventsRequest& request,
                               PutEventsResponse* response);

  // Gets all events with matching execution ids.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status GetEventsByExecutionIDs(
      const GetEventsByExecutionIDsRequest& request,
      GetEventsByExecutionIDsResponse* response);

  // Gets all events with matching artifact ids.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status GetEventsByArtifactIDs(
      const GetEventsByArtifactIDsRequest& request,
      GetEventsByArtifactIDsResponse* response);

  // Gets all artifacts.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status GetArtifacts(const GetArtifactsRequest& request,
                                  GetArtifactsResponse* response);

  // Gets all executions.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status GetExecutions(const GetExecutionsRequest& request,
                                   GetExecutionsResponse* response);

 private:
  // To construct the object, see Create(...).
  MetadataStore(std::unique_ptr<MetadataSource> metadata_source,
                std::unique_ptr<MetadataAccessObject> metadata_access_object);

  // If an execution type with the same name already exists (let's call it
  // old_execution_type), it compares the given properties in execution_type
  // with the properties in old_execution_type. If there is a property where
  // execution_type and old_execution_type have different types, or
  // execution_type and old_execution_type have different properties, it fails
  // and returns ALREADY_EXISTS.
  // Otherwise, it returns the type_id of old_execution_type.
  // Insert a new execution type.
  // Returns INVALID_ARGUMENT error, if name field in request.execution_type
  //     is not given.
  // Returns INVALID_ARGUMENT error, if any property type in
  //     request.execution_type is unknown.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status InsertExecutionType(const PutExecutionTypeRequest& request,
                                         PutExecutionTypeResponse* response);

  std::unique_ptr<MetadataSource> metadata_source_;
  std::unique_ptr<MetadataAccessObject> metadata_access_object_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_METADATA_STORE_H_
