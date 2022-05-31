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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"

namespace ml_metadata {

// Declare a parameterized abstract test fixture to run tests on private methods
// of RDBMSMetadataAccessObject created with different MetadataSource types.
class RDBMSMetadataAccessObjectTest;

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
  absl::Status InitMetadataSource() final {
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
  absl::Status InitMetadataSourceIfNotExists(
      bool enable_upgrade_migration = false) final {
    return executor_->InitMetadataSourceIfNotExists(enable_upgrade_migration);
  }


  // Deletes the metadata source. All the metadata and other associated data
  // will be deleted. The metadata source can no longer be queried before
  // calling InitMetadataSourceIfNotExists again.
  // Returns detailed INTERNAL error, if update execution fails.
  absl::Status DeleteMetadataSource() final {
    return executor_->DeleteMetadataSource();
  }

  // Downgrades the schema to `to_schema_version` in the given metadata source.
  // Returns INVALID_ARGUMENT, if `to_schema_version` is less than 0, or newer
  //   than the library version.
  // Returns FAILED_PRECONDITION, if db schema version is newer than the
  //   library version.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DowngradeMetadataSource(int64 to_schema_version) final {
    return executor_->DowngradeMetadataSource(to_schema_version);
  }

  absl::Status CreateType(const ArtifactType& type, int64* type_id) final;
  absl::Status CreateType(const ExecutionType& type, int64* type_id) final;
  absl::Status CreateType(const ContextType& type, int64* type_id) final;

  absl::Status UpdateType(const ArtifactType& type) final;
  absl::Status UpdateType(const ExecutionType& type) final;
  absl::Status UpdateType(const ContextType& type) final;

  absl::Status FindTypeById(int64 type_id, ArtifactType* artifact_type) final;
  absl::Status FindTypeById(int64 type_id, ExecutionType* execution_type) final;
  absl::Status FindTypeById(int64 type_id, ContextType* context_type) final;

  absl::Status FindTypeByNameAndVersion(
      absl::string_view name, absl::optional<absl::string_view> version,
      ArtifactType* artifact_type) final;
  absl::Status FindTypeByNameAndVersion(
      absl::string_view name, absl::optional<absl::string_view> version,
      ExecutionType* execution_type) final;
  absl::Status FindTypeByNameAndVersion(
      absl::string_view name, absl::optional<absl::string_view> version,
      ContextType* context_type) final;

  absl::Status FindTypeIdByNameAndVersion(
      absl::string_view name, absl::optional<absl::string_view> version,
      TypeKind type_kind, int64* type_id) final;

  absl::Status FindTypes(std::vector<ArtifactType>* artifact_types) final;
  absl::Status FindTypes(std::vector<ExecutionType>* execution_types) final;
  absl::Status FindTypes(std::vector<ContextType>* context_types) final;

  absl::Status CreateParentTypeInheritanceLink(
      const ArtifactType& type, const ArtifactType& parent_type) final;
  absl::Status CreateParentTypeInheritanceLink(
      const ExecutionType& type, const ExecutionType& parent_type) final;
  absl::Status CreateParentTypeInheritanceLink(
      const ContextType& type, const ContextType& parent_type) final;

  absl::Status FindParentTypesByTypeId(
      const absl::Span<const int64> type_ids,
      absl::flat_hash_map<int64, ArtifactType>& output_parent_types) final;
  absl::Status FindParentTypesByTypeId(
      const absl::Span<const int64> type_ids,
      absl::flat_hash_map<int64, ExecutionType>& output_parent_types) final;
  absl::Status FindParentTypesByTypeId(
      const absl::Span<const int64> type_ids,
      absl::flat_hash_map<int64, ContextType>& output_parent_types) final;

  absl::Status CreateArtifact(const Artifact& artifact,
                              int64* artifact_id) final;

  absl::Status FindArtifactsById(absl::Span<const int64> artifact_ids,
                                 std::vector<Artifact>* artifacts) final;

  absl::Status FindArtifacts(std::vector<Artifact>* artifacts) final;

  absl::Status ListArtifacts(const ListOperationOptions& options,
                             std::vector<Artifact>* artifacts,
                             std::string* next_page_token) final;

  absl::Status ListExecutions(const ListOperationOptions& options,
                              std::vector<Execution>* executions,
                              std::string* next_page_token) final;

  absl::Status ListContexts(const ListOperationOptions& options,
                            std::vector<Context>* contexts,
                            std::string* next_page_token) final;

  absl::Status FindArtifactsByTypeId(
      int64 artifact_type_id, absl::optional<ListOperationOptions> list_options,
      std::vector<Artifact>* artifacts, std::string* next_page_token) final;

  absl::Status FindArtifactByTypeIdAndArtifactName(int64 type_id,
                                                   absl::string_view name,
                                                   Artifact* artifact) final;

  absl::Status FindArtifactsByURI(absl::string_view uri,
                                  std::vector<Artifact>* artifacts) final;

  absl::Status UpdateArtifact(const Artifact& artifact) final;

  absl::Status CreateExecution(const Execution& execution,
                               int64* execution_id) final;

  absl::Status FindExecutionsById(absl::Span<const int64> execution_ids,
                                  std::vector<Execution>* executions) final;

  absl::Status FindExecutions(std::vector<Execution>* executions) final;

  absl::Status FindExecutionByTypeIdAndExecutionName(
      int64 type_id, absl::string_view name, Execution* execution) final;

  absl::Status FindExecutionsByTypeId(
      int64 execution_type_id,
      absl::optional<ListOperationOptions> list_options,
      std::vector<Execution>* executions, std::string* next_page_token) final;

  absl::Status UpdateExecution(const Execution& execution) final;

  absl::Status CreateContext(const Context& context, int64* context_id) final;

  absl::Status FindContextsById(absl::Span<const int64> context_ids,
                                std::vector<Context>* contexts) final;

  absl::Status FindContexts(std::vector<Context>* contexts) final;

  absl::Status FindContextsByTypeId(
      int64 type_id, absl::optional<ListOperationOptions> list_options,
      std::vector<Context>* contexts, std::string* next_page_token) final;

  absl::Status FindContextByTypeIdAndContextName(int64 type_id,
                                                 absl::string_view name,
                                                 bool id_only,
                                                 Context* context) final;

  absl::Status UpdateContext(const Context& context) final;

  absl::Status CreateEvent(const Event& event, int64* event_id) final;

  absl::Status CreateEvent(const Event& event, const bool is_already_validated,
                           int64* event_id) final;

  absl::Status FindEventsByArtifacts(const std::vector<int64>& artifact_ids,
                                     std::vector<Event>* events) final;

  absl::Status FindEventsByExecutions(const std::vector<int64>& execution_ids,
                                      std::vector<Event>* events) final;

  absl::Status CreateAssociation(const Association& association,
                                 int64* association_id) final;

  absl::Status CreateAssociation(const Association& association,
                                 bool is_already_validated,
                                 int64* association_id) final;


  absl::Status FindContextsByExecution(int64 execution_id,
                                       std::vector<Context>* contexts) final;

  absl::Status FindExecutionsByContext(
      int64 context_id, std::vector<Execution>* executions) final;

  absl::Status FindExecutionsByContext(
      int64 context_id, absl::optional<ListOperationOptions> list_options,
      std::vector<Execution>* executions, std::string* next_page_token) final;

  absl::Status CreateAttribution(const Attribution& attribution,
                                 int64* attribution_id) final;

  absl::Status CreateAttribution(const Attribution& attribution,
                                 bool is_already_validated,
                                 int64* attribution_id) final;

  absl::Status FindContextsByArtifact(int64 artifact_id,
                                      std::vector<Context>* contexts) final;

  absl::Status FindArtifactsByContext(int64 context_id,
                                      std::vector<Artifact>* artifacts) final;

  absl::Status FindArtifactsByContext(
      int64 context_id, absl::optional<ListOperationOptions> list_options,
      std::vector<Artifact>* artifacts, std::string* next_page_token) final;

  absl::Status CreateParentContext(const ParentContext& parent_context) final;

  absl::Status FindParentContextsByContextId(
      int64 context_id, std::vector<Context>* contexts) final;

  absl::Status FindChildContextsByContextId(
      int64 context_id, std::vector<Context>* contexts) final;

  absl::Status GetSchemaVersion(int64* db_version) final {
    return executor_->GetSchemaVersion(db_version);
  }

  int64 GetLibraryVersion() final { return executor_->GetLibraryVersion(); }


  // The method is currently used for accessing MLMD lineage.
  // TODO(b/178491112) Support Execution typed query_nodes.
  // TODO(b/178491112) Returns contexts in the returned subgraphs.
  absl::Status QueryLineageGraph(
      const std::vector<Artifact>& query_nodes, int64 max_num_hops,
      absl::optional<int64> max_nodes,
      absl::optional<std::string> boundary_artifacts,
      absl::optional<std::string> boundary_executions,
      LineageGraph& subgraph) final;


  // Deletes a list of artifacts by id.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteArtifactsById(
      absl::Span<const int64> artifact_ids) final;

  // Deletes a list of executions by id.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteExecutionsById(
      absl::Span<const int64> execution_ids) final;

  // Deletes a list of contexts by id.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteContextsById(
      absl::Span<const int64> context_ids) final;

  // Deletes the events corresponding to the |artifact_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteEventsByArtifactsId(
      absl::Span<const int64> artifact_ids) final;

  // Deletes the events corresponding to the |execution_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteEventsByExecutionsId(
      absl::Span<const int64> execution_ids) final;

  // Deletes the associations corresponding to the |context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteAssociationsByContextsId(
      absl::Span<const int64> context_ids) final;

  // Deletes the associations corresponding to the |execution_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteAssociationsByExecutionsId(
      absl::Span<const int64> execution_ids) final;

  // Deletes the attributions corresponding to the |context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteAttributionsByContextsId(
      absl::Span<const int64> context_ids) final;

  // Deletes the attributions corresponding to the |artifact_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteAttributionsByArtifactsId(
      absl::Span<const int64> artifact_ids) final;

  // Deletes the parent contexts corresponding to the |parent_context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteParentContextsByParentIds(
      absl::Span<const int64> parent_context_ids) final;

  // Deletes the parent contexts corresponding to the |child_context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteParentContextsByChildIds(
      absl::Span<const int64> child_context_ids) final;

  // Deletes the parent type link |type_id, parent_type_id|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteParentTypeInheritanceLink(
      int64 type_id, int64 parent_type_id) final;

 private:
  ///////// These methods are implementations details //////////////////////////

  // Creates an Artifact (without properties).
  absl::Status CreateBasicNode(const Artifact& artifact, int64* node_id);

  // Creates an Execution (without properties).
  absl::Status CreateBasicNode(const Execution& execution, int64* node_id);
  // Creates a Context (without properties).
  absl::Status CreateBasicNode(const Context& context, int64* node_id);

  // Retrieves nodes (and their properties) based on the provided 'ids'.
  // 'header' contains the non-property information, and 'properties' contains
  // information about properties. The node id is present in both record sets
  // and can be used to join the information. The 'properties' are returned
  // using the same convention as
  // QueryExecutor::Select{Node}PropertyBy{Node}ID().
  template <typename T>
  absl::Status RetrieveNodesById(
      absl::Span<const int64> id, RecordSet* header, RecordSet* properties,
      T* tag = nullptr /* used only for the template */);

  // Update an Artifact's type_id and URI.
  absl::Status RunNodeUpdate(const Artifact& artifact);

  // Update an Execution's type_id.
  absl::Status RunNodeUpdate(const Execution& execution);

  // Update a Context's type id and name.
  absl::Status RunNodeUpdate(const Context& context);

  // Runs a property insertion query for a NodeType.
  template <typename NodeType>
  absl::Status InsertProperty(const int64 node_id, const absl::string_view name,
                              const bool is_custom_property,
                              const Value& value);

  // Generates a property update query for a NodeType.
  template <typename NodeType>
  absl::Status UpdateProperty(const int64 node_id, const absl::string_view name,
                              const Value& value);

  // Generates a property deletion query for a NodeType.
  template <typename NodeType>
  absl::Status DeleteProperty(const int64 node_id,
                              const absl::string_view name);

  // Generates a list of queries for the `curr_properties` (C) based on the
  // given `prev_properties` (P). A property definition is a 2-tuple (name,
  // value_type). a) any property in the intersection of C and P, a update query
  // is generated. b) any property in C \ P, insert query is generated. c) any
  // property in P \ C, delete query is generated. The queries are composed from
  // corresponding template queries with the given `NodeType` (which is one of
  // {`ArtifactType`, `ExecutionType`, `ContextType`} and the
  // `is_custom_property` (which indicates the space of the given properties.
  // Returns `output_num_changed_properties` which equals to the number of
  // properties are changed (deleted, updated or inserted).
  template <typename NodeType>
  absl::Status ModifyProperties(
      const google::protobuf::Map<std::string, Value>& curr_properties,
      const google::protobuf::Map<std::string, Value>& prev_properties,
      const int64 node_id, const bool is_custom_property,
      int& output_num_changed_properties);

  // Creates a query to insert an artifact type.
  absl::Status InsertTypeID(const ArtifactType& type, int64* type_id);

  // Creates a query to insert an execution type.
  absl::Status InsertTypeID(const ExecutionType& type, int64* type_id);

  // Creates a query to insert a context type.
  absl::Status InsertTypeID(const ContextType& type, int64* type_id);

  // Creates a `Type` where acceptable ones are in {ArtifactType, ExecutionType,
  // ContextType}.
  // Returns INVALID_ARGUMENT error, if name field is not given.
  // Returns INVALID_ARGUMENT error, if any property type is unknown.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Type>
  absl::Status CreateTypeImpl(const Type& type, int64* type_id);

  // Generates a query to find all type instances.
  absl::Status GenerateFindAllTypeInstancesQuery(const TypeKind type_kind,
                                                 RecordSet* record_set);

  // FindType takes a result of a query for types, and populates additional
  // information such as properties, and returns it in `types`.
  // If `get_properties` equals false, skip the query that retrieves properties
  // from property table.
  template <typename MessageType>
  absl::Status FindTypesFromRecordSet(const RecordSet& type_record_set,
                                      std::vector<MessageType>* types,
                                      bool get_properties = true);

  // Finds types by the given `type_ids`. Acceptable types are {ArtifactType,
  // ExecutionType, ContextType} (`MessageType`).
  // Returns INVALID_ARGUMENT if `type_ids` is empty or `types` is not empty.
  // Returns detailed INTERNAL error if query execution fails.
  // If any ids are not found then returns NOT_FOUND error.
  template <typename MessageType>
  absl::Status FindTypesImpl(absl::Span<const int64> type_ids,
                             std::vector<MessageType>& types);

  // Finds a type by its type_id. Acceptable types are {ArtifactType,
  // ExecutionType, ContextType} (`MessageType`).
  // Returns NOT_FOUND error, if the given type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename MessageType>
  absl::Status FindTypeImpl(int64 type_id, MessageType* type);

  // Finds a type by its name and an optional version.
  // Acceptable types are {ArtifactType,
  // ExecutionType, ContextType} (`MessageType`).
  // Returns NOT_FOUND error, if the given name and version cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename MessageType>
  absl::Status FindTypeImpl(absl::string_view name,
                            absl::optional<absl::string_view> version,
                            MessageType* type);

  // Finds all type instances of the type `MessageType`.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename MessageType>
  absl::Status FindAllTypeInstancesImpl(std::vector<MessageType>* types);

  // Updates an existing type. A type is one of {ArtifactType, ExecutionType,
  // ContextType}
  // Returns INVALID_ARGUMENT error, if name field is not given.
  // Returns INVALID_ARGUMENT error, if id field is given and is different.
  // Returns INVALID_ARGUMENT error, if any property type is unknown.
  // Returns ALREADY_EXISTS error, if any property type is different.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Type>
  absl::Status UpdateTypeImpl(const Type& type);

  // Queries the parent type of each type_id in `type_ids`. Currently only
  // single inheritance (one parent type per type_id) is supported.
  // The prerequisite is that all the types with `type_ids` already exist in db.
  // Returns INVALID_ARGUMENT error, if the given `type_ids` is empty, or
  // `output_parent_types` is not empty.
  template <typename Type>
  absl::Status FindParentTypesByTypeIdImpl(
      const absl::Span<const int64> type_ids,
      absl::flat_hash_map<int64, Type>& output_parent_types);

  // Creates an `Node`, which is one of {`Artifact`, `Execution`, `Context`},
  // then returns the assigned node id. The node's id field is ignored. The node
  // should have a `NodeType`, which is one of {`ArtifactType`, `ExecutionType`,
  // `ContextType`}.
  // Returns INVALID_ARGUMENT error, if the node does not align with its type.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Node, typename NodeType>
  absl::Status CreateNodeImpl(const Node& node, int64* node_id);

  // Queries a `Node` which is one of {`Artifact`, `Execution`, `Context`} by
  // an id.
  // Returns NOT_FOUND error, if the given id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Node>
  absl::Status FindNodeImpl(const int64 node_id, Node* node);

  // Retrieves a set of `Node` which is one of {`Artifact`, `Execution`,
  // `Context`} by the given 'ids'.
  // 'skipped_ids_ok' controls the return error value if any of the ids are not
  // found.
  // Returns INVALID_ARGUMENT if node_ids is empty or nodes is not empty.
  // Returns detailed INTERNAL error if query execution fails.
  // If any ids are not found then returns NOT_FOUND if skipped_ids_ok is true,
  // otherwise INTERNAL error.
  template <typename Node>
  absl::Status FindNodesImpl(absl::Span<const int64> node_ids,
                             bool skipped_ids_ok, std::vector<Node>& nodes);

  // Updates a `Node` which is one of {`Artifact`, `Execution`, `Context`}.
  // Returns INVALID_ARGUMENT error, if the node cannot be found
  // Returns INVALID_ARGUMENT error, if the node does not match with its type
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Node, typename NodeType>
  absl::Status UpdateNodeImpl(const Node& node);

  // Takes a record set that has one record per event and for each record:
  //   parses it into an Event object
  //   gets the path of the event from the database
  // Returns INVALID_ARGUMENT error, if the `events` is null.
  absl::Status FindEventsFromRecordSet(const RecordSet& event_record_set,
                                       std::vector<Event>* events);

  // Takes a record set that has one record per association and parses it into
  // an Association object for each record.
  // Returns INVALID_ARGUMENT error, if the `associations` is null.
  absl::Status FindAssociationsFromRecordSet(
      const RecordSet& association_record_set,
      std::vector<Association>* associations);

  // Retrieves the ids of the nodes based on 'options' and `candidate_ids`.
  // If `candidate_ids` is provided, then only the nodes with those ids are
  // considered when applying list options; when nullopt, all stored nodes are
  // considered as candidates.
  // The returned record_set
  // has a single row per id, with the corresponding value.
  template <typename Node>
  absl::Status ListNodeIds(
      const ListOperationOptions& options,
      absl::optional<absl::Span<const int64>> candidate_ids,
      RecordSet* record_set,
      Node* tag = nullptr /* used only for template instantiation*/);

  // Queries nodes stored in the metadata source using `options`.
  // `options` is the ListOperationOptions proto message defined
  // in metadata_store.
  // If `candidate_ids` is provided, then only the nodes with those ids are
  // considered when applying list options; when nullopt, all stored nodes are
  // considered as candidates.
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
  absl::Status ListNodes(const ListOperationOptions& options,
                         absl::optional<absl::Span<const int64>> candidate_ids,
                         std::vector<Node>* nodes,
                         std::string* next_page_token);

  // Traverse a ParentContext relation to look for parent or child context.
  enum class ParentContextTraverseDirection { kParent, kChild };

  // Queries the ParentContext with a context_id and returns a list of Context.
  // If direction is kParent, then context_id is used to look for its parents.
  // If direction is kChild, then context_id is used to look for its children.
  absl::Status FindLinkedContextsImpl(int64 context_id,
                                      ParentContextTraverseDirection direction,
                                      std::vector<Context>& output_contexts);

  // The utilities to expand lineage `subgraph` within one hop from artifacts.
  // For the `input_artifacts`, their neighborhood executions that do not
  // satisfy `boundary_condition` are visited and output as `output_executions`.
  // The `output_executions` and the events between `input_artifacts` are added
  // to the `subgraph`. The `visited_execution_ids` captures the already
  // visited executions in previous traversal, while the `visited_artifact_ids`
  // maintains previously visited and the newly visited `input_artifacts`.
  absl::Status ExpandLineageGraphImpl(
      const std::vector<Artifact>& input_artifacts, int64 max_nodes,
      absl::optional<std::string> boundary_condition,
      const absl::flat_hash_set<int64>& visited_execution_ids,
      absl::flat_hash_set<int64>& visited_artifact_ids,
      std::vector<Execution>& output_executions, LineageGraph& subgraph);

  // The utilities to expand lineage `subgraph` within one hop from executions.
  // For the `input_executions`, their neighborhood artifacts that do not
  // satisfy `boundary_condition` are visited and output as `output_artifacts`.
  // The `output_artifacts` and the events between `input_executions` are added
  // to the `subgraph`. The `visited_artifact_ids` captures the already
  // visited artifacts in previous traversal, while the `visited_execution_ids`
  // maintains previously visited and the newly visited `input_executions`.
  absl::Status ExpandLineageGraphImpl(
      const std::vector<Execution>& input_executions, int64 max_nodes,
      absl::optional<std::string> boundary_condition,
      const absl::flat_hash_set<int64>& visited_artifact_ids,
      absl::flat_hash_set<int64>& visited_execution_ids,
      std::vector<Artifact>& output_artifacts, LineageGraph& subgraph);

  // Given `boundary_condition`, the utility method keeps nodes that satisfy
  // the `boundary_condition`, and removes any nodes that do not satisfy the
  // `boundary_condition` from `unvisited_node_ids`.
  template <typename Node>
  absl::Status SkipBoundaryNodesImpl(
      absl::optional<std::string> boundary_condition,
      absl::flat_hash_set<int64>& unvisited_node_ids);

  std::unique_ptr<QueryExecutor> executor_;

  friend RDBMSMetadataAccessObjectTest;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_RDBMS_METADATA_ACCESS_OBJECT_H_
