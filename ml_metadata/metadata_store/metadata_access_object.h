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
#ifndef ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_H_
#define ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_H_

#include <memory>
#include <vector>

#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// Data access object (DAO) for the domain entities (Type, Artifact, Execution,
// Event) defined in metadata_store.proto. It provides a list of query methods
// to store, update, and read entities. It takes a MetadataSourceQueryConfig
// which specifies query instructions on the given MetadataSource.
// Each method is a list of queries which needs to be run within a transaction,
// and the caller can use the methods provided and compose larger transactions
// externally. The caller is responsible to commit or rollback depending on the
// return status of each method: if it succeeds all changes should be committed
// in a transaction, if fails, then there should be no change in the underline
// MetadataSource. It is thread-unsafe.
//
// Usage example:
//
//    SomeConcreteMetadataSource src;
//    MetadataSourceQueryConfig config;
//    std::unique_ptr<MetadataAccessObject> mao;
//    TF_CHECK_OK(MetadataAccessObject::Create(config, &src, &mao));
//
//    if (mao->SomeCRUDMethod(...).ok())
//      TF_CHECK_OK(src.Commit()); // or do more queries
//    else
//      TF_CHECK_OK(src.Rollback());
//
class MetadataAccessObject {
 public:
  // Factory method, if the return value is ok, 'result' is populated with an
  // object that can be used to access metadata with the given config and
  // metadata_source. The caller is responsible to own a MetadataSource, and the
  // MetadataAccessObject connects and execute queries with the MetadataSource.
  // Returns INVALID_ARGUMENT error, if query_config is not valid.
  // Returns detailed INTERNAL error, if the MetadataSource cannot be connected.
  static tensorflow::Status Create(
      const MetadataSourceQueryConfig& query_config,
      MetadataSource* metadata_source,
      std::unique_ptr<MetadataAccessObject>* result);

  ~MetadataAccessObject() = default;

  // default & copy constructors are disallowed.
  MetadataAccessObject() = delete;
  MetadataAccessObject(const MetadataAccessObject&) = delete;
  MetadataAccessObject& operator=(const MetadataAccessObject&) = delete;

  // Initializes the metadata source and creates schema. Any existing data in
  // the MetadataSource is dropped.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status InitMetadataSource();

  // Initializes the metadata source and creates schema.
  // Returns OK and does nothing, if all required schema exist.
  // Returns OK and creates schema, if no schema exists yet.
  // Returns DATA_LOSS error, if any required schema is missing.
  // Returns detailed INTERNAL error, if create schema query execution fails.
  tensorflow::Status InitMetadataSourceIfNotExists();

  // Creates a type, returns the assigned type id. A type is an ArtifactType or
  // an ExecutionType. The id field of the given type is ignored.
  tensorflow::Status CreateType(const ArtifactType& type, int64* type_id);
  tensorflow::Status CreateType(const ExecutionType& type, int64* type_id);

  // Queries a type by an id. A type is an ArtifactType or an ExecutionType.
  // Returns NOT_FOUND error, if the given type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindTypeById(int64 type_id, ArtifactType* artifact_type);
  tensorflow::Status FindTypeById(int64 type_id, ExecutionType* execution_type);

  // Queries a type by its name. A type is an ArtifactType or an ExecutionType.
  // Returns NOT_FOUND error, if the given name cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindTypeByName(absl::string_view name,
                                    ArtifactType* artifact_type);
  tensorflow::Status FindTypeByName(absl::string_view name,
                                    ExecutionType* execution_type);

  // Creates an artifact, returns the assigned artifact id. The id field of the
  // artifact is ignored.
  // Returns INVALID_ARGUMENT error, if the ArtifactType is not given.
  // Returns NOT_FOUND error, if the ArtifactType cannot be found.
  // Returns INVALID_ARGUMENT error, if the artifact contains any property
  //  undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the artifact type.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status CreateArtifact(const Artifact& artifact,
                                    int64* artifact_id);

  // Queries an artifact by an id.
  // Returns NOT_FOUND error, if the given artifact_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindArtifactById(int64 artifact_id, Artifact* artifact);

  // Queries artifacts stored in the metadata source
  tensorflow::Status FindArtifacts(std::vector<Artifact>* artifacts);

  // Updates an artifact.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no artifact is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ArtifactType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status UpdateArtifact(const Artifact& artifact);

  // Creates an execution, returns the assigned execution id. The id field of
  // the execution is ignored.
  // Returns INVALID_ARGUMENT error, if the ExecutionType is not given.
  // Returns NOT_FOUND error, if the ExecutionType cannot be found.
  // Returns INVALID_ARGUMENT error, if the execution contains any property
  //   undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the ExecutionType.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status CreateExecution(const Execution& execution,
                                     int64* execution_id);

  // Queries an entity by an id.
  // Returns NOT_FOUND error, if the given execution_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindExecutionById(int64 execution_id,
                                       Execution* execution);

  // Queries executions stored in the metadata source
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status FindExecutions(std::vector<Execution>* executions);

  // Updates an execution.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no execution is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ExecutionType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status UpdateExecution(const Execution& execution);

  // Creates an event, returns the assigned event id. If the event occurrence
  // time is not given, the insertion time is used.
  tensorflow::Status CreateEvent(const Event& event, int64* event_id);

  // Queries the events associated with an artifact_id.
  // Returns INVALID_ARGUMENT error, if the `events` is null.
  // Returns NOT_FOUND error, if there are no events found with the `artifact`.
  tensorflow::Status FindEventsByArtifact(int64 artifact_id,
                                          std::vector<Event>* events);

  // Queries the events associated with an execution_id.
  // Returns INVALID_ARGUMENT error, if the `events` is null.
  // Returns NOT_FOUND error, if there are no events found with the `execution`.
  tensorflow::Status FindEventsByExecution(int64 execution_id,
                                           std::vector<Event>* events);

  tensorflow::Status UpdateType(const ArtifactType& type);

  MetadataSource* metadata_source() { return metadata_source_; }

 private:
  // Constructs a MetadataAccessObject with a query config and a connected
  // MetadataSource.
  explicit MetadataAccessObject(const MetadataSourceQueryConfig& query_config,
                                MetadataSource* connected_metadata_source);

  const MetadataSourceQueryConfig query_config_;

  MetadataSource* const metadata_source_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_H_
