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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/field_mask.pb.h"
#include "google/protobuf/struct.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/util/field_mask_utils.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {

// Validates properties in a `Node` with the properties defined in a `Type`.
// `Node` is one of {`Artifact`, `Execution`, `Context`}. `Type` is one of
// {`ArtifactType`, `ExecutionType`, `ContextType`}.
// Returns INVALID_ARGUMENT error, if there is unknown or mismatched property
// w.r.t. its definition.
// TODO(b/197686185): Move this helper function to  metadata_store.cc once MAO
// no longer needs it.
template <typename Node, typename Type>
absl::Status ValidatePropertiesWithType(
    const Node& node, const Type& type,
    const google::protobuf::FieldMask& mask = {}) {
  const google::protobuf::Map<std::string, PropertyType>& type_properties =
      type.properties();

  bool check_masked_properties = true;
  absl::flat_hash_set<absl::string_view> property_names_in_mask;
  if (!mask.paths().empty()) {
    absl::StatusOr<absl::flat_hash_set<absl::string_view>>
        property_names_or_internal_error =
            GetPropertyNamesFromMask(mask, /*is_custom_properties=*/false);
    if (property_names_or_internal_error.status().code() ==
            absl::StatusCode::kInternal &&
        absl::StrContains(property_names_or_internal_error.status().message(),
                          "no property name is specified")) {
      check_masked_properties = false;
    } else {
      MLMD_RETURN_IF_ERROR(property_names_or_internal_error.status());
      property_names_in_mask = property_names_or_internal_error.value();
    }
  } else {
    check_masked_properties = false;
  }

  for (const auto& p : node.properties()) {
    const std::string& property_name = p.first;
    // If mask is specified, only check masked property names.
    if (check_masked_properties && !property_names_in_mask.count(property_name))
      continue;

    const Value& property_value = p.second;
    if (type_properties.find(property_name) == type_properties.end())
      return absl::InvalidArgumentError(
          absl::StrCat("Found unknown property: ", property_name));
    bool is_type_match = false;
    switch (type_properties.at(property_name)) {
      case PropertyType::INT: {
        is_type_match = property_value.has_int_value();
        break;
      }
      case PropertyType::DOUBLE: {
        is_type_match = property_value.has_double_value();
        break;
      }
      case PropertyType::STRING: {
        is_type_match = property_value.has_string_value();
        break;
      }
      case PropertyType::STRUCT: {
        is_type_match = property_value.has_struct_value();
        break;
      }
      case PropertyType::PROTO: {
        is_type_match = property_value.has_proto_value();
        break;
      }
      case PropertyType::BOOLEAN: {
        is_type_match = property_value.has_bool_value();
        break;
      }
      default: {
        return absl::InternalError(absl::StrCat(
            "Unknown registered property type: ", type.DebugString()));
      }
    }
    if (!is_type_match)
      return absl::InvalidArgumentError(
          absl::StrCat("Found unmatched property type: ", property_name));
  }
  return absl::OkStatus();
}

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
  absl::Status DowngradeMetadataSource(int64_t to_schema_version) final {
    return executor_->DowngradeMetadataSource(to_schema_version);
  }

  absl::Status CreateType(const ArtifactType& type, int64_t* type_id) final;
  absl::Status CreateType(const ExecutionType& type, int64_t* type_id) final;
  absl::Status CreateType(const ContextType& type, int64_t* type_id) final;

  absl::Status UpdateType(const ArtifactType& type) final;
  absl::Status UpdateType(const ExecutionType& type) final;
  absl::Status UpdateType(const ContextType& type) final;

  absl::Status FindTypeById(int64_t type_id, ArtifactType* artifact_type) final;
  absl::Status FindTypeById(int64_t type_id,
                            ExecutionType* execution_type) final;
  absl::Status FindTypeById(int64_t type_id, ContextType* context_type) final;

  absl::Status FindTypesByIds(absl::Span<const int64_t> type_ids,
                              std::vector<ArtifactType>& artifact_types) final;
  absl::Status FindTypesByIds(
      absl::Span<const int64_t> type_ids,
      std::vector<ExecutionType>& execution_types) final;
  absl::Status FindTypesByIds(absl::Span<const int64_t> type_ids,
                              std::vector<ContextType>& context_types) final;

  absl::Status FindTypesByExternalIds(
      absl::Span<absl::string_view> external_ids,
      std::vector<ArtifactType>& artifact_types) final;
  absl::Status FindTypesByExternalIds(
      absl::Span<absl::string_view> external_ids,
      std::vector<ExecutionType>& execution_types) final;
  absl::Status FindTypesByExternalIds(
      absl::Span<absl::string_view> external_ids,
      std::vector<ContextType>& context_types) final;

  absl::Status FindTypeByNameAndVersion(
      absl::string_view name, std::optional<absl::string_view> version,
      ArtifactType* artifact_type) final;
  absl::Status FindTypeByNameAndVersion(
      absl::string_view name, std::optional<absl::string_view> version,
      ExecutionType* execution_type) final;
  absl::Status FindTypeByNameAndVersion(
      absl::string_view name, std::optional<absl::string_view> version,
      ContextType* context_type) final;

  absl::Status FindTypeIdByNameAndVersion(
      absl::string_view name, std::optional<absl::string_view> version,
      TypeKind type_kind, int64_t* type_id) final;

  absl::Status FindTypesByNamesAndVersions(
      absl::Span<std::pair<std::string, std::string>> names_and_versions,
      std::vector<ArtifactType>& artifact_types) final;
  absl::Status FindTypesByNamesAndVersions(
      absl::Span<std::pair<std::string, std::string>> names_and_versions,
      std::vector<ExecutionType>& execution_types) final;
  absl::Status FindTypesByNamesAndVersions(
      absl::Span<std::pair<std::string, std::string>> names_and_versions,
      std::vector<ContextType>& context_types) final;

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
      absl::Span<const int64_t> type_ids,
      absl::flat_hash_map<int64_t, ArtifactType>& output_parent_types) final;
  absl::Status FindParentTypesByTypeId(
      absl::Span<const int64_t> type_ids,
      absl::flat_hash_map<int64_t, ExecutionType>& output_parent_types) final;
  absl::Status FindParentTypesByTypeId(
      absl::Span<const int64_t> type_ids,
      absl::flat_hash_map<int64_t, ContextType>& output_parent_types) final;

  absl::Status CreateArtifact(const Artifact& artifact,
                              bool skip_type_and_property_validation,
                              int64_t* artifact_id) final;

  absl::Status CreateArtifact(const Artifact& artifact,
                              bool skip_type_and_property_validation,
                              absl::Time create_timestamp,
                              int64_t* artifact_id) final;

  absl::Status CreateArtifact(const Artifact& artifact,
                              int64_t* artifact_id) final;

  absl::Status FindArtifactsById(absl::Span<const int64_t> artifact_ids,
                                 std::vector<Artifact>* artifacts) final;

  absl::Status FindArtifactsById(
      absl::Span<const int64_t> artifact_ids, std::vector<Artifact>& artifacts,
      std::vector<ArtifactType>& artifact_types) final;

  absl::Status FindArtifactsByExternalIds(
      absl::Span<absl::string_view> external_ids,
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
      int64_t artifact_type_id,
      std::optional<ListOperationOptions> list_options,
      std::vector<Artifact>* artifacts, std::string* next_page_token) final;

  absl::Status FindArtifactByTypeIdAndArtifactName(int64_t type_id,
                                                   absl::string_view name,
                                                   Artifact* artifact) final;

  absl::Status FindArtifactsByURI(absl::string_view uri,
                                  std::vector<Artifact>* artifacts) final;

  absl::Status UpdateArtifact(const Artifact& artifact) final;

  absl::Status UpdateArtifact(const Artifact& artifact,
                              const google::protobuf::FieldMask& mask) final;

  absl::Status UpdateArtifact(const Artifact& artifact,
                              absl::Time update_timestamp,
                              bool force_update_time) final;

  absl::Status UpdateArtifact(const Artifact& artifact,
                              absl::Time update_timestamp,
                              bool force_update_time,
                              const google::protobuf::FieldMask& mask) final;

  absl::Status CreateExecution(const Execution& execution,
                               bool skip_type_and_property_validation,
                               int64_t* execution_id) final;

  absl::Status CreateExecution(const Execution& execution,
                               bool skip_type_and_property_validation,
                               absl::Time create_timestamp,
                               int64_t* execution_id) final;

  absl::Status CreateExecution(const Execution& execution,
                               int64_t* execution_id) final;

  absl::Status FindExecutionsById(absl::Span<const int64_t> execution_ids,
                                  std::vector<Execution>* executions) final;

  absl::Status FindExecutionsByExternalIds(
      absl::Span<absl::string_view> external_ids,
      std::vector<Execution>* executions) final;

  absl::Status FindExecutions(std::vector<Execution>* executions) final;

  absl::Status FindExecutionByTypeIdAndExecutionName(
      int64_t type_id, absl::string_view name, Execution* execution) final;

  absl::Status FindExecutionsByTypeId(
      int64_t execution_type_id,
      std::optional<ListOperationOptions> list_options,
      std::vector<Execution>* executions, std::string* next_page_token) final;

  absl::Status UpdateExecution(const Execution& execution) final;

  absl::Status UpdateExecution(const Execution& execution,
                               absl::Time update_timestamp,
                               bool force_update_time) final;

  absl::Status UpdateExecution(const Execution& execution,
                               bool force_update_time,
                               const google::protobuf::FieldMask& mask) final;

  absl::Status UpdateExecution(const Execution& execution,
                               absl::Time update_timestamp,
                               bool force_update_time,
                               const google::protobuf::FieldMask& mask) final;

  absl::Status CreateContext(const Context& context,
                             bool skip_type_and_property_validation,
                             int64_t* context_id) final;

  absl::Status CreateContext(const Context& context,
                             bool skip_type_and_property_validation,
                             absl::Time create_timestamp,
                             int64_t* context_id) final;

  absl::Status CreateContext(const Context& context, int64_t* context_id) final;

  absl::Status FindContextsById(absl::Span<const int64_t> context_ids,
                                std::vector<Context>* contexts) final;

  absl::Status FindContextsByExternalIds(
      absl::Span<absl::string_view> external_ids,
      std::vector<Context>* contexts) final;

  absl::Status FindContexts(std::vector<Context>* contexts) final;

  absl::Status FindContextsByTypeId(
      int64_t type_id, std::optional<ListOperationOptions> list_options,
      std::vector<Context>* contexts, std::string* next_page_token) final;

  absl::Status FindContextByTypeIdAndContextName(int64_t type_id,
                                                 absl::string_view name,
                                                 bool id_only,
                                                 Context* context) final;

  absl::Status UpdateContext(const Context& context) final;

  absl::Status UpdateContext(const Context& context,
                             absl::Time update_timestamp,
                             bool force_update_time) final;

  absl::Status UpdateContext(const Context& context,
                             const google::protobuf::FieldMask& mask) final;

  absl::Status UpdateContext(const Context& context,
                             absl::Time update_timestamp,
                             bool force_update_time,
                             const google::protobuf::FieldMask& mask) final;

  absl::Status CreateEvent(const Event& event, int64_t* event_id) final;

  absl::Status CreateEvent(const Event& event, bool is_already_validated,
                           int64_t* event_id) final;

  absl::Status FindEventsByArtifacts(absl::Span<const int64_t> artifact_ids,
                                     std::vector<Event>* events) final;

  absl::Status FindEventsByExecutions(absl::Span<const int64_t> execution_ids,
                                      std::vector<Event>* events) final;

  absl::Status CreateAssociation(const Association& association,
                                 int64_t* association_id) final;

  absl::Status CreateAssociation(const Association& association,
                                 bool is_already_validated,
                                 int64_t* association_id) final;

  absl::Status FindAssociationsByContexts(
      absl::Span<const int64_t> context_ids,
      std::vector<Association>* associations) final;

  absl::Status FindAssociationsByExecutions(
      absl::Span<const int64_t> execution_ids,
      std::vector<Association>* associations) final;

  absl::Status FindAttributionsByArtifacts(
      absl::Span<const int64_t> artifact_ids,
      std::vector<Attribution>* attributions) final;

  absl::Status FindContextsByExecution(int64_t execution_id,
                                       std::vector<Context>* contexts) final;

  absl::Status FindExecutionsByContext(
      int64_t context_id, std::vector<Execution>* executions) final;

  absl::Status FindExecutionsByContext(
      int64_t context_id, std::optional<ListOperationOptions> list_options,
      std::vector<Execution>* executions, std::string* next_page_token) final;

  absl::Status CreateAttribution(const Attribution& attribution,
                                 int64_t* attribution_id) final;

  absl::Status CreateAttribution(const Attribution& attribution,
                                 bool is_already_validated,
                                 int64_t* attribution_id) final;

  absl::Status FindContextsByArtifact(int64_t artifact_id,
                                      std::vector<Context>* contexts) final;

  absl::Status FindArtifactsByContext(int64_t context_id,
                                      std::vector<Artifact>* artifacts) final;

  absl::Status FindArtifactsByContext(
      int64_t context_id, std::optional<ListOperationOptions> list_options,
      std::vector<Artifact>* artifacts, std::string* next_page_token) final;

  absl::Status CreateParentContext(const ParentContext& parent_context) final;

  absl::Status FindParentContextsByContextId(
      int64_t context_id, std::vector<Context>* contexts) final;

  absl::Status FindChildContextsByContextId(
      int64_t context_id, std::vector<Context>* contexts) final;

  absl::Status FindParentContextsByContextIds(
      absl::Span<const int64_t> context_ids,
      absl::node_hash_map<int64_t, std::vector<Context>>& contexts) final;

  absl::Status FindChildContextsByContextIds(
      absl::Span<const int64_t> context_ids,
      absl::node_hash_map<int64_t, std::vector<Context>>& contexts) final;

  absl::Status GetSchemaVersion(int64_t* db_version) final {
    return executor_->GetSchemaVersion(db_version);
  }

  int64_t GetLibraryVersion() final { return executor_->GetLibraryVersion(); }


  // The method is currently used for accessing MLMD lineage.
  // TODO(b/178491112) Support Execution typed query_nodes.
  // TODO(b/178491112) Returns contexts in the returned subgraphs.
  // TODO(b/283852485): Deprecate GetLineageGraph API after migration to
  // GetLineageSubgraph API.
  absl::Status QueryLineageGraph(const std::vector<Artifact>& query_nodes,
                                 int64_t max_num_hops,
                                 std::optional<int64_t> max_nodes,
                                 std::optional<std::string> boundary_artifacts,
                                 std::optional<std::string> boundary_executions,
                                 LineageGraph& subgraph) final;

  // TODO(b/283852485): migrate from QueryLineageGraph to QueryLineageSubgraph.
  // Given the `lineage_subgraph_query_options`, performs a constrained BFS
  // on the lineage graph and returns a graph including the reached edges and
  // nodes with only `id` field.
  // The constraints include:
  // a) `max_num_hops`: it stops traversal at nodes that are at `max_num_hops`
  //   away from the starting nodes.
  // b) `direction`: it performs either a single-directional graph traversal or
  //   bidirectional graph traversal based on `direction`.
  // `read_mask` contains user specified paths of fields that should be included
  // in the output `subgraph`.
  //   If 'artifacts', 'executions', or 'contexts' is specified in `read_mask`,
  //     the dehydrated nodes will be included.
  //   If 'artifact_types', 'execution_types', or 'context_types' is specified
  //     in `read_mask`, all the node types will be included.
  //   If 'events', 'associations', or 'attributions' is specified in
  //     `read_mask`, the corresponding edges will be included.
  //   Note: `read_mask` is a mask on fields from `LineageGraph`. Any other
  //   field path such as artifact.id, execution.name will not be supported.
  // Returns INVALID_ARGUMENT error, if no paths are specified in `read_mask`.
  // Returns INVALID_ARGUMENT error, if `starting_nodes` is not specified in
  // `lineage_subgraph_query_options`.
  // Returns INVALID_ARGUMENT error, if `starting_nodes.filter_query` is
  // unspecified or invalid in `lineage_subgraph_query_options`.
  // Returns detailed INTERNAL error, if the operation fails.
  absl::Status QueryLineageSubgraph(
      const LineageSubgraphQueryOptions& options,
      const google::protobuf::FieldMask& read_mask,
      LineageGraph& subgraph) final;


  // Deletes a list of artifacts by id.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteArtifactsById(
      absl::Span<const int64_t> artifact_ids) final;

  // Deletes a list of executions by id.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteExecutionsById(
      absl::Span<const int64_t> execution_ids) final;

  // Deletes a list of contexts by id.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteContextsById(absl::Span<const int64_t> context_ids) final;

  // Deletes the events corresponding to the |artifact_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteEventsByArtifactsId(
      absl::Span<const int64_t> artifact_ids) final;

  // Deletes the events corresponding to the |execution_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteEventsByExecutionsId(
      absl::Span<const int64_t> execution_ids) final;

  // Deletes the associations corresponding to the |context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteAssociationsByContextsId(
      absl::Span<const int64_t> context_ids) final;

  // Deletes the associations corresponding to the |execution_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteAssociationsByExecutionsId(
      absl::Span<const int64_t> execution_ids) final;

  // Deletes the attributions corresponding to the |context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteAttributionsByContextsId(
      absl::Span<const int64_t> context_ids) final;

  // Deletes the attributions corresponding to the |artifact_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteAttributionsByArtifactsId(
      absl::Span<const int64_t> artifact_ids) final;

  // Deletes the parent contexts corresponding to the |parent_context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteParentContextsByParentIds(
      absl::Span<const int64_t> parent_context_ids) final;

  // Deletes the parent contexts corresponding to the |child_context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteParentContextsByChildIds(
      absl::Span<const int64_t> child_context_ids) final;

  // Deletes the parent contexts corresponding to the |parent_context_id|
  // and |child_context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteParentContextsByParentIdAndChildIds(
      int64_t parent_context_id,
      absl::Span<const int64_t> child_context_ids) final;

  // Deletes the parent type link |type_id, parent_type_id|.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::Status DeleteParentTypeInheritanceLink(int64_t type_id,
                                               int64_t parent_type_id) final;

 private:
  ///////// These methods are implementations details //////////////////////////

  // Creates an Artifact (without properties).
  absl::Status CreateBasicNode(const Artifact& artifact,
                               absl::Time create_timestamp,
                               int64_t* node_id);

  // Creates an Execution (without properties).
  absl::Status CreateBasicNode(const Execution& execution,
                               absl::Time create_timestamp,
                               int64_t* node_id);
  // Creates a Context (without properties).
  absl::Status CreateBasicNode(const Context& context,
                               absl::Time create_timestamp,
                               int64_t* node_id);

  // Gets nodes (and their properties) based on the provided 'ids'.
  // 'header' contains the non-property information, and 'properties' contains
  // information about properties. The node id is present in both record sets
  // and can be used to join the information. The 'properties' are returned
  // using the same convention as
  // QueryExecutor::Select{Node}PropertyBy{Node}ID().
  template <typename T>
  absl::Status RetrieveNodesById(
      absl::Span<const int64_t> id, RecordSet* header, RecordSet* properties);

  // Update a Node's assets based on the field mask.
  // If `mask` is empty, update `stored_node` as a whole.
  // If `mask` is not empty, only update fields specified in `mask`.
  template <typename Node>
  absl::Status RunMaskedNodeUpdate(
      const Node& node, Node& stored_node, absl::Time update_timestamp,
      const google::protobuf::FieldMask& mask = {});

  // Update an Artifact's type_id and URI.
  absl::Status RunNodeUpdate(const Artifact& artifact,
                             absl::Time update_timestamp);

  // Update an Execution's type_id.
  absl::Status RunNodeUpdate(const Execution& execution,
                             absl::Time update_timestamp);

  // Update a Context's type id and name.
  absl::Status RunNodeUpdate(const Context& context,
                             absl::Time update_timestamp);

  // Runs a property insertion query for a NodeType.
  template <typename NodeType>
  absl::Status InsertProperty(int64_t node_id, absl::string_view name,
                              bool is_custom_property,
                              const Value& value);

  // Generates a property update query for a NodeType.
  template <typename NodeType>
  absl::Status UpdateProperty(int64_t node_id, absl::string_view name,
                              const Value& value);

  // Generates a property deletion query for a NodeType.
  template <typename NodeType>
  absl::Status DeleteProperty(int64_t node_id, absl::string_view name);

  // Generates a list of queries for the `curr_properties` (C) based on the
  // given `prev_properties` (P) only for properties associated with names in
  // `mask`(M). A property definition is a 2-tuple (name, value_type).
  // a) any property in the intersection of M, C and P, an update query is
  // generated.
  // b) any property in the intersection of M and (C \ P), insert query is
  // generated.
  // c) any property in the intersection of M and (P \ C), delete query is
  // generated.
  // The queries are composed from corresponding template
  // queries with the given `NodeType` (which is one of {`ArtifactType`,
  // `ExecutionType`, `ContextType`} and the `is_custom_property` (which
  // indicates the space of the given properties. Returns
  // `output_num_changed_properties` which equals to the number of properties
  // are changed (deleted, updated or inserted).
  template <typename NodeType>
  absl::StatusOr<int64_t> ModifyProperties(
      const google::protobuf::Map<std::string, Value>& curr_properties,
      const google::protobuf::Map<std::string, Value>& prev_properties,
      int64_t node_id, bool is_custom_property,
      const google::protobuf::FieldMask& mask = {});

  // Creates a query to insert an artifact type.
  absl::Status InsertTypeID(const ArtifactType& type, int64_t* type_id);

  // Creates a query to insert an execution type.
  absl::Status InsertTypeID(const ExecutionType& type, int64_t* type_id);

  // Creates a query to insert a context type.
  absl::Status InsertTypeID(const ContextType& type, int64_t* type_id);

  // Creates a `Type` where acceptable ones are in {ArtifactType, ExecutionType,
  // ContextType}.
  // Returns INVALID_ARGUMENT error, if name field is not given.
  // Returns INVALID_ARGUMENT error, if any property type is unknown.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Type>
  absl::Status CreateTypeImpl(const Type& type, int64_t* type_id);

  // Generates a query to find all type instances.
  absl::Status GenerateFindAllTypeInstancesQuery(TypeKind type_kind,
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
  // `get_properties` flag is used to control whether the returned `types`
  // should contain any properties.
  // Returns INVALID_ARGUMENT if `type_ids` is empty or `types` is not empty.
  // Returns detailed INTERNAL error if query execution fails.
  // If any ids are not found then returns NOT_FOUND error.
  template <typename MessageType>
  absl::Status FindTypesImpl(absl::Span<const int64_t> type_ids,
                             bool get_properties,
                             std::vector<MessageType>& types);

  // Finds types by the given `external_ids`.
  // Acceptable types are {ArtifactType, ExecutionType, ContextType}
  // (`MessageType`). `get_properties` flag is used to control whether the
  // returned `types` should contain any properties.
  // Returns whatever found when a part of `external_ids` is non-existing.
  // Returns NOT_FOUND error if all the given `external_ids` are not found.
  // Returns INVALID_ARGUMENT if `external_ids` is empty or `types` is not
  // empty.
  // Returns detailed INTERNAL error if query execution fails.
  template <typename MessageType>
  absl::Status FindTypesByExternalIdsImpl(
      absl::Span<absl::string_view> external_ids, bool get_properties,
      std::vector<MessageType>& types);

  // Finds a type by its type_id. Acceptable types are {ArtifactType,
  // ExecutionType, ContextType} (`MessageType`).
  // Returns NOT_FOUND error, if the given type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename MessageType>
  absl::Status FindTypeImpl(int64_t type_id, MessageType* type);

  // Finds a type by its name and an optional version.
  // Acceptable types are {ArtifactType,
  // ExecutionType, ContextType} (`MessageType`).
  // Returns NOT_FOUND error, if the given name and version cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename MessageType>
  absl::Status FindTypeImpl(absl::string_view name,
                            std::optional<absl::string_view> version,
                            MessageType* type);

  // Finds types by the given `names_and_versions`.
  // Acceptable types are {ArtifactType, ExecutionType, ContextType}
  // (`MessageType`).
  // Returns whatever found when a part of `names_and_versions` is non-existing.
  // Returns INVALID_ARGUMENT if `types` is not empty.
  // Returns detailed INTERNAL error if query execution fails.
  template <typename MessageType>
  absl::Status FindTypesImpl(
      absl::Span<std::pair<std::string, std::string>> names_and_versions,
      std::vector<MessageType>& types);

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

  // Gets the parent type of each type_id in `type_ids`. Currently only
  // single inheritance (one parent type per type_id) is supported.
  // The prerequisite is that all the types with `type_ids` already exist in db.
  // Returns INVALID_ARGUMENT error, if the given `type_ids` is empty, or
  // `output_parent_types` is not empty.
  template <typename Type>
  absl::Status FindParentTypesByTypeIdImpl(
      absl::Span<const int64_t> type_ids,
      absl::flat_hash_map<int64_t, Type>& output_parent_types);

  // Creates an `Node`, which is one of {`Artifact`, `Execution`, `Context`},
  // then returns the assigned node id. The node's id field is ignored. The node
  // should have a `NodeType`, which is one of {`ArtifactType`, `ExecutionType`,
  // `ContextType`}.
  // If `skip_type_verfication` is set to be true, the `FindTypeImpl()` and
  // `ValidatePropertiesWithType()` are skipped.
  // `create_timestamp` should be used as the create time of the Node.
  // Returns INVALID_ARGUMENT error, if the node does not align with its type.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Node, typename NodeType>
  absl::Status CreateNodeImpl(const Node& node,
                              bool skip_type_and_property_validation,
                              absl::Time create_timestamp,
                              int64_t* node_id);

  // Gets a `Node` which is one of {`Artifact`, `Execution`, `Context`} by
  // an id.
  // Returns NOT_FOUND error, if the given id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  template <typename Node>
  absl::Status FindNodeImpl(int64_t node_id, Node* node);

  // Gets a set of `Node` which is one of {`Artifact`, `Execution`,
  // `Context`} by the given 'ids'.
  // 'skipped_ids_ok' controls the return error value if any of the ids are not
  // found.
  // Returns INVALID_ARGUMENT if node_ids is empty or nodes is not empty.
  // Returns detailed INTERNAL error if query execution fails.
  // If any ids are not found then returns NOT_FOUND if skipped_ids_ok is true,
  // otherwise INTERNAL error.
  template <typename Node>
  absl::Status FindNodesImpl(absl::Span<const int64_t> node_ids,
                             bool skipped_ids_ok, std::vector<Node>& nodes);

  // Gets a set of `Node` which is one of {`Artifact`, `Execution`,
  // `Context`} by the given 'node_ids' and their node types, which
  // can be matched by type_ids.
  // Returns INVALID_ARGUMENT if node_ids is empty or nodes is not empty.
  // Returns NOT_FOUND error if any of the given `node_ids` is not found.
  // Returns detailed INTERNAL error if query execution fails.
  template <typename Node, typename NodeType>
  absl::Status FindNodesWithTypesImpl(absl::Span<const int64_t> node_ids,
                                      std::vector<Node>& nodes,
                                      std::vector<NodeType>& node_types);

  // Updates with masking for a `Node` being one of {`Artifact`, `Execution`,
  // `Context`}.
  // `update_timestamp` should be used as the update time of the Node.
  // When `force_update_time` is set to true, `last_update_time_since_epoch` is
  // updated even if input node is the same as stored node.
  // If `mask` is empty, update the `node` as a whole, otherwise, perform masked
  // update on the `node`.
  // Returns INVALID_ARGUMENT error, if the node cannot be
  // found Returns INVALID_ARGUMENT error, if the node does not match with its
  // type Returns detailed INTERNAL error, if query execution fails.
  template <typename Node, typename NodeType>
  absl::Status UpdateNodeImpl(const Node& node,
                              absl::Time update_timestamp,
                              bool force_update_time,
                              const google::protobuf::FieldMask& mask = {});

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

  // Takes a record set that has one record per attribution and parses it into
  // an Attribution object for each record.
  // Returns INVALID_ARGUMENT error, if the `attributions` is null.
  absl::Status FindAttributionsFromRecordSet(
      const RecordSet& attribution_record_set,
      std::vector<Attribution>* attributions);

  // Gets the ids of the nodes based on 'options' and `candidate_ids`.
  // If `candidate_ids` is provided, then only the nodes with those ids are
  // considered when applying list options; when nullopt, all stored nodes are
  // considered as candidates.
  // The returned record_set
  // has a single row per id, with the corresponding value.
  template <typename Node>
  absl::Status ListNodeIds(
      const ListOperationOptions& options,
      std::optional<absl::Span<const int64_t>> candidate_ids,
      RecordSet* record_set);

  // Gets nodes stored in the metadata source using `options`.
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
                         std::optional<absl::Span<const int64_t>> candidate_ids,
                         std::vector<Node>* nodes,
                         std::string* next_page_token);

  // Traverse a ParentContext relation to look for parent or child context.
  enum class ParentContextTraverseDirection { kParent, kChild };

  // Gets the ParentContext with a context_id and returns a list of Context.
  // If direction is kParent, then context_id is used to look for its parents.
  // If direction is kChild, then context_id is used to look for its children.
  absl::Status FindLinkedContextsImpl(int64_t context_id,
                                      ParentContextTraverseDirection direction,
                                      std::vector<Context>& output_contexts);

  // Gets the ParentContext with a context_ids list and returns a map of
  // <context_id, linked_contexts> for each context_id in context_ids. If
  // direction is kParent, then context_id is used to look for its parents. If
  // direction is kChild, then context_id is used to look for its children.
  absl::Status FindLinkedContextsMapImpl(
      absl::Span<const int64_t> context_ids,
      ParentContextTraverseDirection direction,
      absl::node_hash_map<int64_t, std::vector<Context>>& output_contexts);

  // TODO(b/283852485): Deprecate GetLineageGraph API after migration to
  // GetLineageSubgraph API.
  // The utilities to expand lineage `subgraph` within one hop from artifacts.
  // For the `input_artifacts`, their neighborhood executions that do not
  // satisfy `boundary_condition` are visited and output as `output_executions`.
  // The `output_executions` and the events between `input_artifacts` are added
  // to the `subgraph`. The `visited_execution_ids` captures the already
  // visited executions in previous traversal, while the `visited_artifact_ids`
  // maintains previously visited and the newly visited `input_artifacts`.
  absl::Status ExpandLineageGraphImpl(
      const std::vector<Artifact>& input_artifacts, int64_t max_nodes,
      std::optional<std::string> boundary_condition,
      const absl::flat_hash_set<int64_t>& visited_execution_ids,
      absl::flat_hash_set<int64_t>& visited_artifact_ids,
      std::vector<Execution>& output_executions, LineageGraph& subgraph);

  // TODO(b/283852485): Deprecate GetLineageGraph API after migration to
  // GetLineageSubgraph API.
  // The utilities to expand lineage `subgraph` within one hop from executions.
  // For the `input_executions`, their neighborhood artifacts that do not
  // satisfy `boundary_condition` are visited and output as `output_artifacts`.
  // The `output_artifacts` and the events between `input_executions` are added
  // to the `subgraph`. The `visited_artifact_ids` captures the already
  // visited artifacts in previous traversal, while the `visited_execution_ids`
  // maintains previously visited and the newly visited `input_executions`.
  absl::Status ExpandLineageGraphImpl(
      const std::vector<Execution>& input_executions, int64_t max_nodes,
      std::optional<std::string> boundary_condition,
      const absl::flat_hash_set<int64_t>& visited_artifact_ids,
      absl::flat_hash_set<int64_t>& visited_execution_ids,
      std::vector<Artifact>& output_artifacts, LineageGraph& subgraph);

  // Expands the lineage subgraph under constraint within one hop from input
  // nodes.
  // When expanding from artifacts to executions, it treats artifacts as input
  //   nodes and executions as output nodes.
  // When expanding from executions to artifacts, it treats executions as input
  //   nodes and artifacts as output nodes.
  // When expanding towards upstream, it only looks at upstream hops such as
  //   execution -> input_event -> artifact or
  //   artifact -> output_event -> execution.
  // When expanding towards downstream, it only looks at downstream hops such as
  //   execution -> output_event -> artifact or
  //   artifact -> input_event -> execution.
  // When expanding bidirectionally, it looks at both upstream hops and
  //   downsteam hops.
  // If `ending_nodes` is set in `options`, do not expand from those ending
  // nodes.
  // Adds events between input nodes and output nodes to `output_events`.
  // Returns ids of output nodes that are one hop away from input nodes if
  // expanding the lineage subgraph succeeds.
  // Returns an empty list if no events are found for given input nodes.
  // Returns detailed INTERNAL error, if expanding the lineage subgraph fails.
  absl::StatusOr<std::vector<int64_t>> ExpandLineageSubgraphImpl(
      bool expand_from_artifacts, const LineageSubgraphQueryOptions& options,
      absl::Span<const int64_t> input_node_ids,
      absl::flat_hash_set<int64_t>& visited_output_node_ids,
      absl::flat_hash_set<int64_t>& output_ending_node_ids,
      std::vector<Event>& output_events);

  // Given `node_filter`, keeps nodes that satisfy the `node_filter`, and
  // removes any nodes that do not satisfy the `node_filter` from
  // `boundary_node_ids`.
  // Returns OK status if `node_filter` is not specified or filtering boundary
  // nodes succeeds.
  // Returns detailed INTERNAL error, if filtering boundary nodes fails.
  template <typename Node>
  absl::Status FilterBoundaryNodesImpl(
      std::optional<absl::string_view> node_filter,
      absl::flat_hash_set<int64_t>& boundary_node_ids);

  // Given a list of node ids, finds nodes that satisfy the `filter_query` in
  // `ending_nodes`.
  // Returns a list of ending node ids if executing the filter query succeeds.
  // Returns an empty list if no `filter_query` is specified in `ending_nodes`.
  // Returns detailed INTERNAL error, if executing the filter query fails.
  template <typename Node>
  absl::StatusOr<absl::flat_hash_set<int64_t>> FindEndingNodeIdsIfExists(
      const LineageSubgraphQueryOptions::EndingNodes ending_nodes,
      const absl::flat_hash_set<int64_t>& unvisited_node_ids);

  // Find Contexts based on the given artifact_ids and execution_ids.
  // Returns a list of found Contexts if succeeds.
  // Returns detailed INTERNAL error, if query execution fails.
  absl::StatusOr<std::vector<int64_t>> FindContextIdsByArtifactsAndExecutions(
      absl::Span<const int64_t> artifact_ids,
      absl::Span<const int64_t> execution_ids);

  std::unique_ptr<QueryExecutor> executor_;

  friend RDBMSMetadataAccessObjectTest;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_RDBMS_METADATA_ACCESS_OBJECT_H_
