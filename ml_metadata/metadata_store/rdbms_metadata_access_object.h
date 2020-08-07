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
#ifndef ML_METADATA_METADATA_STORE_RDBMS_METADATA_ACCESS_OBJECT_H_
#define ML_METADATA_METADATA_STORE_RDBMS_METADATA_ACCESS_OBJECT_H_

#include <memory>
#include <vector>

#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// An implementation of MetadataAccessObject for a typical relational
// database. The basic assumption is that the database has a schema similar
// to the schema of the SQLite database, and that an API close to SQL queries
// is applicable.
//
// This class contains a QueryExecutor object, that handles direct access to the
// database. The intent is that this class should not be subclassed: instead,
// a new subclass of QueryExecutor should be created.
class RDBMSMetadataAccessObject : public MetadataAccessObject {
 public:
  virtual ~RDBMSMetadataAccessObject() = default;

  // default & copy constructors are disallowed.
  RDBMSMetadataAccessObject(std::unique_ptr<QueryExecutor> executor)
      : executor_(std::move(executor)) {}

  // default & copy constructors are disallowed.
  RDBMSMetadataAccessObject() = delete;
  RDBMSMetadataAccessObject(const RDBMSMetadataAccessObject&) = delete;
  RDBMSMetadataAccessObject& operator=(const RDBMSMetadataAccessObject&) =
      delete;

  // Initializes the metadata source and creates schema. Any existing data in
  // the MetadataSource is dropped.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status InitMetadataSource() final {
    return executor_->InitMetadataSource();
  }

  // Initializes the metadata source and creates schema.
  // Returns OK and does nothing, if all required schema exist.
  // Returns OK and creates schema, if no schema exists yet.
  // Returns DATA_LOSS error, if the MLMDENv has more than one schema version.
  // Returns ABORTED error, if any required schema is missing.
  // Returns FAILED_PRECONDITION error, if library and db have incompatible
  //   schema versions, and upgrade migrations are not enabled.
  // Returns detailed INTERNAL error, if create schema query execution fails.
  tensorflow::Status InitMetadataSourceIfNotExists(
      bool enable_upgrade_migration = false) final {
    return executor_->InitMetadataSourceIfNotExists(enable_upgrade_migration);
  }

  // Downgrades the schema to `to_schema_version` in the given metadata source.
  // Returns INVALID_ARGUMENT, if `to_schema_version` is less than 0, or newer
  //   than the library version.
  // Returns FAILED_PRECONDITION, if db schema version is newer than the
  //   library version.
  // Returns detailed INTERNAL error, if query execution fails.
  tensorflow::Status DowngradeMetadataSource(int64 to_schema_version) final {
    return executor_->DowngradeMetadataSource(to_schema_version);
  }

  tensorflow::Status CreateType(const ArtifactType& type, int64* type_id) final;
  tensorflow::Status CreateType(const ExecutionType& type,
                                int64* type_id) final;
  tensorflow::Status CreateType(const ContextType& type, int64* type_id) final;

  tensorflow::Status UpdateType(const ArtifactType& type) final;
  tensorflow::Status UpdateType(const ExecutionType& type) final;
  tensorflow::Status UpdateType(const ContextType& type) final;

  tensorflow::Status FindTypeById(int64 type_id,
                                  ArtifactType* artifact_type) final;
  tensorflow::Status FindTypeById(int64 type_id,
                                  ExecutionType* execution_type) final;
  tensorflow::Status FindTypeById(int64 type_id,
                                  ContextType* context_type) final;

  tensorflow::Status FindTypeByName(absl::string_view name,
                                    ArtifactType* artifact_type) final;
  tensorflow::Status FindTypeByName(absl::string_view name,
                                    ExecutionType* execution_type) final;
  tensorflow::Status FindTypeByName(absl::string_view name,
                                    ContextType* context_type) final;

  tensorflow::Status FindTypes(std::vector<ArtifactType>* artifact_types) final;
  tensorflow::Status FindTypes(
      std::vector<ExecutionType>* execution_types) final;
  tensorflow::Status FindTypes(std::vector<ContextType>* context_types) final;

  tensorflow::Status CreateArtifact(const Artifact& artifact,
                                    int64* artifact_id) final;

  tensorflow::Status FindArtifactById(int64 artifact_id,
                                      Artifact* artifact) final;

  tensorflow::Status FindArtifacts(std::vector<Artifact>* artifacts) final;

  tensorflow::Status ListArtifacts(const ListOperationOptions& options,
                                   std::vector<Artifact>* artifacts,
                                   std::string* next_page_token) final;

  tensorflow::Status ListExecutions(const ListOperationOptions& options,
                                    std::vector<Execution>* executions,
                                    std::string* next_page_token) final;

  tensorflow::Status ListContexts(const ListOperationOptions& options,
                                  std::vector<Context>* contexts,
                                  std::string* next_page_token) final;

  tensorflow::Status FindArtifactsByTypeId(
      int64 artifact_type_id, std::vector<Artifact>* artifacts) final;

  tensorflow::Status FindArtifactByTypeIdAndArtifactName(
      int64 type_id, absl::string_view name, Artifact* artifact) final;

  tensorflow::Status FindArtifactsByURI(absl::string_view uri,
                                        std::vector<Artifact>* artifacts) final;

  tensorflow::Status UpdateArtifact(const Artifact& artifact) final;

  tensorflow::Status CreateExecution(const Execution& execution,
                                     int64* execution_id) final;

  tensorflow::Status FindExecutionById(int64 execution_id,
                                       Execution* execution) final;

  tensorflow::Status FindExecutions(std::vector<Execution>* executions) final;

  tensorflow::Status FindExecutionByTypeIdAndExecutionName(
      int64 type_id, absl::string_view name, Execution* execution) final;

  tensorflow::Status FindExecutionsByTypeId(
      int64 execution_type_id, std::vector<Execution>* executions) final;

  tensorflow::Status UpdateExecution(const Execution& execution) final;

  tensorflow::Status CreateContext(const Context& context,
                                   int64* context_id) final;

  tensorflow::Status FindContextById(int64 context_id, Context* context) final;

  tensorflow::Status FindContexts(std::vector<Context>* contexts) final;

  tensorflow::Status FindContextsByTypeId(int64 context_type_id,
                                          std::vector<Context>* contexts) final;

  tensorflow::Status FindContextByTypeIdAndContextName(int64 type_id,
                                                       absl::string_view name,
                                                       Context* context) final;

  tensorflow::Status UpdateContext(const Context& context) final;

  tensorflow::Status CreateEvent(const Event& event, int64* event_id) final;

  tensorflow::Status FindEventsByArtifacts(
      const std::vector<int64>& artifact_ids, std::vector<Event>* events) final;

  tensorflow::Status FindEventsByExecutions(
      const std::vector<int64>& execution_ids,
      std::vector<Event>* events) final;

  tensorflow::Status CreateAssociation(const Association& association,
                                       int64* association_id) final;

  tensorflow::Status FindContextsByExecution(
      int64 execution_id, std::vector<Context>* contexts) final;

  tensorflow::Status FindExecutionsByContext(
      int64 context_id, std::vector<Execution>* executions) final;

  tensorflow::Status CreateAttribution(const Attribution& attribution,
                                       int64* attribution_id) final;

  tensorflow::Status FindContextsByArtifact(
      int64 artifact_id, std::vector<Context>* contexts) final;

  tensorflow::Status FindArtifactsByContext(
      int64 context_id, std::vector<Artifact>* artifacts) final;

  tensorflow::Status GetSchemaVersion(int64* db_version) final {
    return executor_->GetSchemaVersion(db_version);
  }

  int64 GetLibraryVersion() final { return executor_->GetLibraryVersion(); }


 private:
  ///////// These methods are implementations details //////////////////////////

  // Creates an Artifact (without properties).
  tensorflow::Status CreateBasicNode(const Artifact& artifact, int64* node_id);

  // Creates an Execution (without properties).
  tensorflow::Status CreateBasicNode(const Execution& execution,
                                     int64* node_id);
  // Creates a Context (without properties).
  tensorflow::Status CreateBasicNode(const Context& context, int64* node_id);

  // Lookup Artifact by id.
  tensorflow::Status NodeLookups(const Artifact& artifact, RecordSet* header,
                                 RecordSet* properties);

  // Generates a select queries for an Execution by id.
  tensorflow::Status NodeLookups(const Execution& execution, RecordSet* header,
                                 RecordSet* properties);

  // Lookup Context by id.
  tensorflow::Status NodeLookups(const Context& context, RecordSet* header,
                                 RecordSet* properties);

  // Update an Artifact's type_id and URI.
  tensorflow::Status RunNodeUpdate(const Artifact& artifact);

  // Update an Execution's type_id.
  tensorflow::Status RunNodeUpdate(const Execution& execution);

  // Update a Context's type id and name.
  tensorflow::Status RunNodeUpdate(const Context& context);

  // Runs a property insertion query for a NodeType.
  template <typename NodeType>
  tensorflow::Status InsertProperty(const int64 node_id,
                                    const absl::string_view name,
                                    const bool is_custom_property,
                                    const Value& value);

  // Generates a property update query for a NodeType.
  template <typename NodeType>
  tensorflow::Status UpdateProperty(const int64 node_id,
                                    const absl::string_view name,
                                    const Value& value);

  // Generates a property deletion query for a NodeType.
  template <typename NodeType>
  tensorflow::Status DeleteProperty(const int64 node_id,
                                    const absl::string_view name);

  // Generates a list of queries for the `curr_properties` (C) based on the
  // given `prev_properties` (P). A property definition is a 2-tuple (name,
  // value_type). a) any property in the intersection of C and P, a update query
  // is generated. b) any property in C \ P, insert query is generated. c) any
  // property in P \ C, delete query is generated. The queries are composed from
  // corresponding template queries with the given `NodeType` (which is one of
  // {`ArtifactType`, `ExecutionType`, `ContextType`} and the
  // `is_custom_property` (which indicates the space of the given properties.
  template <typename NodeType>
  tensorflow::Status ModifyProperties(
      const google::protobuf::Map<std::string, Value>& curr_properties,
      const google::protobuf::Map<std::string, Value>& prev_properties,
      const int64 node_id, const bool is_custom_property);

  // Creates a query to insert an artifact type.
  tensorflow::Status InsertTypeID(const ArtifactType& type, int64* type_id);

  // Creates a query to insert an execution type.
  tensorflow::Status InsertTypeID(const ExecutionType& type, int64* type_id);

  // Creates a query to insert a context type.
  tensorflow::Status InsertTypeID(const ContextType& type, int64* type_id);

  // Creates a `Type` where acceptable ones are in {ArtifactType, ExecutionType,
  // ContextType}.
  // Returns INVALID_ARGUMENT error, if name field is not given.
  // Returns INVALID_ARGUMENT error, if any property type is unknown.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Type>
  tensorflow::Status CreateTypeImpl(const Type& type, int64* type_id);

  // Generates a query to find type by id
  tensorflow::Status RunFindTypeByID(const int64 condition,
                                     const TypeKind type_kind,
                                     RecordSet* record_set);

  // Generates a query to find type by name
  tensorflow::Status RunFindTypeByID(absl::string_view condition,
                                     const TypeKind type_kind,
                                     RecordSet* record_set);

  // Generates a query to find all type instances.
  tensorflow::Status GenerateFindAllTypeInstancesQuery(const TypeKind type_kind,
                                                       RecordSet* record_set);

  // FindType takes a result of a query for types, and populates additional
  // information such as properties, and returns it in `types`.
  template <typename MessageType>
  tensorflow::Status FindTypesFromRecordSet(const RecordSet& type_record_set,
                                            std::vector<MessageType>* types);

  // Finds a type by query conditions. Acceptable types are {ArtifactType,
  // ExecutionType, ContextType} (`MessageType`). The types can be queried by
  // two kinds of query conditions, which are type id (int64) or type name
  // (string_view). Returns NOT_FOUND error, if the given type_id cannot be
  // found. Returns detailed INTERNAL error, if query execution fails.
  template <typename QueryCondition, typename MessageType>
  tensorflow::Status FindTypeImpl(const QueryCondition condition,
                                  MessageType* type);

  // Finds all type instances of the type `MessageType`.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename MessageType>
  tensorflow::Status FindAllTypeInstancesImpl(std::vector<MessageType>* types);

  // Updates an existing type. A type is one of {ArtifactType, ExecutionType,
  // ContextType}
  // Returns INVALID_ARGUMENT error, if name field is not given.
  // Returns INVALID_ARGUMENT error, if id field is given and is different.
  // Returns INVALID_ARGUMENT error, if any property type is unknown.
  // Returns ALREADY_EXISTS error, if any property type is different.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Type>
  tensorflow::Status UpdateTypeImpl(const Type& type);

  // Creates an `Node`, which is one of {`Artifact`, `Execution`, `Context`},
  // then returns the assigned node id. The node's id field is ignored. The node
  // should have a `NodeType`, which is one of {`ArtifactType`, `ExecutionType`,
  // `ContextType`}.
  // Returns INVALID_ARGUMENT error, if the node does not align with its type.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Node, typename NodeType>
  tensorflow::Status CreateNodeImpl(const Node& node, int64* node_id);

  // Queries a `Node` which is one of {`Artifact`, `Execution`, `Context`} by
  // an id.
  // Returns NOT_FOUND error, if the given id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Node>
  tensorflow::Status FindNodeImpl(const int64 node_id, Node* node);

  // Find nodes by ID, where the IDs are encoded in a record set.
  template <typename Node>
  tensorflow::Status FindManyNodesImpl(const RecordSet& record_set,
                                       std::vector<Node>* nodes);

  // Updates a `Node` which is one of {`Artifact`, `Execution`, `Context`}.
  // Returns INVALID_ARGUMENT error, if the node cannot be found
  // Returns INVALID_ARGUMENT error, if the node does not match with its type
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Node, typename NodeType>
  tensorflow::Status UpdateNodeImpl(const Node& node);

  // Takes a record set that has one record per event and for each record:
  //   parses it into an Event object
  //   gets the path of the event from the database
  // Returns INVALID_ARGUMENT error, if the `events` is null.
  tensorflow::Status FindEventsFromRecordSet(const RecordSet& event_record_set,
                                             std::vector<Event>* events);

  // Queries `contexts` related to a `Node` (either `Artifact` or `Execution`)
  // by the node id.
  // Returns INVALID_ARGUMENT error, if the `contexts` is null.
  template <typename Node>
  tensorflow::Status FindContextsByNodeImpl(const int64 node_id,
                                            std::vector<Context>* contexts);

  // Queries nodes related to a context. Node is either `Artifact` or
  // `Execution`. Returns INVALID_ARGUMENT error, if the `nodes` is null.
  template <typename Node>
  tensorflow::Status FindNodesByContextImpl(const int64 context_id,
                                            std::vector<Node>* nodes);

  // Queries nodes stored in the metadata source using `options`.
  // `options` is the ListOperationOptions proto message defined
  // in metadata_store.
  // If successfull:
  // 1. `nodes` is updated with result set of size determined by
  //    max_result_size set in `options`.
  // 2. `next_page_token` is populated with information necessary to fetch next
  //    page of results.
  // RETURNS INVALID_ARGUMENT if the `options` is invalid with one of
  //    the cases:
  // 1. order_by_field is not set or has an unspecified field.
  // 2. Direction of ordering is not specified for the order_by_field.
  // 3. next_page_token cannot be decoded.
  template <typename Node>
  tensorflow::Status ListNodes(const ListOperationOptions& options,
                               std::vector<Node>* nodes,
                               std::string* next_page_token);

  std::unique_ptr<QueryExecutor> executor_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_RDBMS_METADATA_ACCESS_OBJECT_H_
