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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/field_mask.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/proto/metadata_store.pb.h"

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
//    CHECK_EQ(absl::OkStatus(), CreateMetadataAccessObject(config, &src,
//    &mao));
//
//    if (mao->SomeCRUDMethod(...).ok())
//      CHECK_EQ(absl::OkStatus(), src.Commit()); // or do more queries
//    else
//      CHECK_EQ(absl::OkStatus(), src.Rollback());
//
class MetadataAccessObject {
 public:
  virtual ~MetadataAccessObject() = default;

  // default & copy constructors are disallowed.
  MetadataAccessObject() = default;
  MetadataAccessObject(const MetadataAccessObject&) = delete;
  MetadataAccessObject& operator=(const MetadataAccessObject&) = delete;

  // Initializes the metadata source and creates schema. Any existing data in
  // the MetadataSource is dropped.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status InitMetadataSource() = 0;

  // Initializes the metadata source and creates schema.
  // Changes not in effect until transaction is COMMITTED.
  // Returns OK and does nothing, if all required schema exist.
  // Returns OK and creates schema, if no schema exists yet.
  // Returns DATA_LOSS error, if the MLMDENv has more than one schema version.
  // Returns ABORTED error, if any required schema is missing.
  // Returns FAILED_PRECONDITION error, if library and db have incompatible
  //   schema versions, and upgrade migrations are not enabled.
  // Returns detailed INTERNAL error, if create schema query execution fails.
  virtual absl::Status InitMetadataSourceIfNotExists(
      bool enable_upgrade_migration = false) = 0;


  // Deletes the metadata source.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteMetadataSource() = 0;

  // Downgrades the schema to `to_schema_version` in the given metadata source.
  // Returns INVALID_ARGUMENT, if `to_schema_version` is less than 0, or newer
  //   than the library version.
  // Returns FAILED_PRECONDITION, if db schema version is newer than the
  //   library version.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DowngradeMetadataSource(int64_t to_schema_version) = 0;

  // Creates a type, returns the assigned type id. A type is one of
  // {ArtifactType, ExecutionType, ContextType}. The id field of the given type
  // is ignored.
  // Returns INVALID_ARGUMENT error, if name field is not given.
  // Returns INVALID_ARGUMENT error, if any property type is unknown.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status CreateType(const ArtifactType& type,
                                  int64_t* type_id) = 0;
  virtual absl::Status CreateType(const ExecutionType& type,
                                  int64_t* type_id) = 0;
  virtual absl::Status CreateType(const ContextType& type,
                                  int64_t* type_id) = 0;

  // Updates an existing type. A type is one of {ArtifactType, ExecutionType,
  // ContextType}. The update should be backward compatible, i.e., existing
  // properties should not be modified, only new properties can be added.
  // Returns INVALID_ARGUMENT error if name field is not given.
  // Returns INVALID_ARGUMENT error, if id field is given but differs from the
  //   the stored type with the same type name.
  // Returns INVALID_ARGUMENT error, if any property type is unknown.
  // Returns ALREADY_EXISTS error, if any property type is different.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateType(const ArtifactType& type) = 0;
  virtual absl::Status UpdateType(const ExecutionType& type) = 0;
  virtual absl::Status UpdateType(const ContextType& type) = 0;

  // Gets a type by an id. A type is one of
  // {ArtifactType, ExecutionType, ContextType}
  // Returns NOT_FOUND error, if the given type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindTypeById(int64_t type_id,
                                    ArtifactType* artifact_type) = 0;
  virtual absl::Status FindTypeById(int64_t type_id,
                                    ExecutionType* execution_type) = 0;
  virtual absl::Status FindTypeById(int64_t type_id,
                                    ContextType* context_type) = 0;

  // Gets a list of types using their ids in `type_ids` and appends the types
  // to `artifact_types`/`execution_type`/`context_type`.
  // Returns absl::OkStatus() if every type is fetched by the according id
  // successfully.
  // Note: The `types` has been deduped.
  // Returns INVALID_ARGUMENT if `type_ids` is empty or
  // `artifact_types`/`execution_type`/`context_type` is not empty.
  // Returns NOT_FOUND error if any of the given `type_ids` cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindTypesByIds(
      absl::Span<const int64_t> type_ids,
      std::vector<ArtifactType>& artifact_types) = 0;
  virtual absl::Status FindTypesByIds(
      absl::Span<const int64_t> type_ids,
      std::vector<ExecutionType>& execution_types) = 0;
  virtual absl::Status FindTypesByIds(
      absl::Span<const int64_t> type_ids,
      std::vector<ContextType>& context_types) = 0;

  // Gets a list of types using their external ids in `external_ids` and
  // appends the types to `artifact_types`/`execution_types`/`context_types`.
  // Returns absl::OkStatus() if types are fetched successfully.
  // Note: The `types` has been deduped.
  // Returns whatever found when a part of `external_ids` is non-existing.
  // Returns NOT_FOUND error if all the given `external_ids` are not found.
  // Returns INVALID_ARGUMENT if `external_ids` is empty or
  // `artifact_types`/`execution_types`/`context_types` is not empty.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindTypesByExternalIds(
      absl::Span<absl::string_view> external_ids,
      std::vector<ArtifactType>& artifact_types) = 0;
  virtual absl::Status FindTypesByExternalIds(
      absl::Span<absl::string_view> external_ids,
      std::vector<ExecutionType>& execution_types) = 0;
  virtual absl::Status FindTypesByExternalIds(
      absl::Span<absl::string_view> external_ids,
      std::vector<ContextType>& context_types) = 0;

  // Gets a type by its name and version. A type is one of
  // {ArtifactType, ExecutionType, ContextType}. The type version is optional.
  // If not given or the version is an empty string, the type with
  // (name, version = NULL) is returned.
  // Returns NOT_FOUND error, if the given name, version cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindTypeByNameAndVersion(
      absl::string_view name, std::optional<absl::string_view> version,
      ArtifactType* artifact_type) = 0;
  virtual absl::Status FindTypeByNameAndVersion(
      absl::string_view name, std::optional<absl::string_view> version,
      ExecutionType* execution_type) = 0;
  virtual absl::Status FindTypeByNameAndVersion(
      absl::string_view name, std::optional<absl::string_view> version,
      ContextType* context_type) = 0;

  virtual absl::Status FindTypeIdByNameAndVersion(
      absl::string_view name, std::optional<absl::string_view> version,
      TypeKind type_kind, int64_t* type_id) = 0;

  // Gets a list of types using their name and version pairs in
  // `names_and_versions` and appends the types to `artifact_types`/
  // `execution_types`/`context_types`.
  // This method assumes the version is set to an empty string if it is not
  // available. In this case, the type with (name, version = NULL) is fetched.
  // Note: The returned types does not guaranteed to have the same order as
  // `names_and_versions`.
  // Returns absl::OkStatus() if types are fetched successfully.
  // Returns whatever found when a part of `names_and_versions` is non-existing.
  // Returns INVALID_ARGUMENT if `names_and_versions` is empty or
  // `artifact_types`/`execution_types`/`context_types` is not empty.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindTypesByNamesAndVersions(
      absl::Span<std::pair<std::string, std::string>> names_and_versions,
      std::vector<ArtifactType>& artifact_types) = 0;
  virtual absl::Status FindTypesByNamesAndVersions(
      absl::Span<std::pair<std::string, std::string>> names_and_versions,
      std::vector<ExecutionType>& execution_types) = 0;
  virtual absl::Status FindTypesByNamesAndVersions(
      absl::Span<std::pair<std::string, std::string>> names_and_versions,
      std::vector<ContextType>& context_types) = 0;

  // Returns a list of all known type instances. A type is one of
  // {ArtifactType, ExecutionType, ContextType}
  // Returns NOT_FOUND error, if no types can be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindTypes(std::vector<ArtifactType>* artifact_types) = 0;
  virtual absl::Status FindTypes(
      std::vector<ExecutionType>* execution_types) = 0;
  virtual absl::Status FindTypes(std::vector<ContextType>* context_types) = 0;

  // Creates a parent type, returns OK if successful.
  // Returns INVALID_ARGUMENT error, if type or parent_type does not have id.
  // Returns INVALID_ARGUMENT error, if type and parent_type introduces cycle.
  // Returns ALREADY_EXISTS error, if the same ParentType record already exists.
  virtual absl::Status CreateParentTypeInheritanceLink(
      const ArtifactType& type, const ArtifactType& parent_type) = 0;
  virtual absl::Status CreateParentTypeInheritanceLink(
      const ExecutionType& type, const ExecutionType& parent_type) = 0;
  virtual absl::Status CreateParentTypeInheritanceLink(
      const ContextType& type, const ContextType& parent_type) = 0;

  // Deletes a parent type, returns OK if successful.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteParentTypeInheritanceLink(
      int64_t type_id, int64_t parent_type_id) = 0;

  // Gets the parent type of each type_id in `type_ids`. Currently only
  // single inheritance (one parent type per type_id) is supported.
  // The prerequisite is that all the types with `type_ids` already exist in db.
  // Returns INVALID_ARGUMENT error, if the given `type_ids` is empty, or
  // `output_parent_types` is not empty.
  virtual absl::Status FindParentTypesByTypeId(
      absl::Span<const int64_t> type_ids,
      absl::flat_hash_map<int64_t, ArtifactType>& output_parent_types) = 0;
  virtual absl::Status FindParentTypesByTypeId(
      absl::Span<const int64_t> type_ids,
      absl::flat_hash_map<int64_t, ExecutionType>& output_parent_types) = 0;
  virtual absl::Status FindParentTypesByTypeId(
      absl::Span<const int64_t> type_ids,
      absl::flat_hash_map<int64_t, ContextType>& output_parent_types) = 0;

  // Creates an artifact, returns the assigned artifact id. The id field of the
  // artifact is ignored.
  // `skip_type_and_property_validation` is set to be true if the `artifact`'s
  // type and properties have been validated.
  // The `create_time_since_epoch` and `last_update_time_since_epoch` fields are
  // determined under the hood with absl::Now().
  // Returns INVALID_ARGUMENT error, if the ArtifactType is not given.
  // Returns NOT_FOUND error, if the ArtifactType cannot be found.
  // Returns INVALID_ARGUMENT error, if the artifact contains any property
  //  undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the artifact type.
  // Returns ALREADY_EXISTS error, if the ArtifactType has artifact with the
  // same name.
  // Returns ALREADY_EXISTS error, if there is artifact with the same
  // external_id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status CreateArtifact(const Artifact& artifact,
                                      bool skip_type_and_property_validation,
                                      int64_t* artifact_id) = 0;

  // Creates an artifact, returns the assigned artifact id. The id field of the
  // artifact is ignored.
  // `skip_type_and_property_validation` is set to be true if the `artifact`'s
  // type and properties have been validated.
  // `create_timestamp` is used as the value of Artifact.create_time_since_epoch
  // and Artifact.last_update_time_since_epoch.
  // Returns INVALID_ARGUMENT error, if the ArtifactType is not given.
  // Returns NOT_FOUND error, if the ArtifactType cannot be found.
  // Returns INVALID_ARGUMENT error, if the artifact contains any property
  //  undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the artifact type.
  // Returns ALREADY_EXISTS error, if the ArtifactType has artifact with the
  // same name.
  // Returns ALREADY_EXISTS error, if there is artifact with the same
  // external_id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status CreateArtifact(const Artifact& artifact,
                                      bool skip_type_and_property_validation,
                                      absl::Time create_timestamp,
                                      int64_t* artifact_id) = 0;

  // Creates an artifact, returns the assigned artifact id. The id field of the
  // artifact is ignored.
  // The `create_time_since_epoch` and `last_update_time_since_epoch` fields are
  // determined under the hood with absl::Now().
  // Please refer to the docstring for CreateArtifact() with the
  // `skip_type_and_property_validation` flag for more details. This method
  // assumes the `artifact`'s type/property has not been validated yet and
  // sets `skip_type_and_property_validation` to false.
  virtual absl::Status CreateArtifact(const Artifact& artifact,
                                      int64_t* artifact_id) = 0;

  // Gets Artifacts matching the given 'artifact_ids'.
  // Returns NOT_FOUND error, if any of the given artifact_ids are not found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindArtifactsById(absl::Span<const int64_t> artifact_ids,
                                         std::vector<Artifact>* artifact) = 0;

  // Gets a set of Artifacts by the given ids and their artifact types, which
  // can be matched by type_ids.
  // Returns NOT_FOUND error, if any of the given artifact_ids is not found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindArtifactsById(
      absl::Span<const int64_t> artifact_ids, std::vector<Artifact>& artifacts,
      std::vector<ArtifactType>& artifact_types) = 0;

  // Gets Artifacts matching the given 'external_ids'.
  // |external_ids| is a list of non-null strings for the given external ids.
  // Returns whatever found when a part of |external_ids| is non-existing.
  // Returns NOT_FOUND error, if all the given external_ids are not found.
  // Returns INVALID_ARGUMENT error, if any of the |external_ids| is empty.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindArtifactsByExternalIds(
      absl::Span<absl::string_view> external_ids,
      std::vector<Artifact>* artifacts) = 0;

  // Gets artifacts stored in the metadata source
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindArtifacts(std::vector<Artifact>* artifacts) = 0;

  // Gets artifacts stored in the metadata source using `options`.
  // `options` is the ListOperationOptions proto message defined
  // in metadata_store.
  // If successfull:
  // 1. `artifacts` is updated with result set of size determined by
  //    max_result_size set in `options`.
  // 2. `next_page_token` is populated with information necessary to fetch next
  //    page of results.
  // An empty result set is returned if no artifacts are retrieved.
  // RETURNS INVALID_ARGUMENT if the `options` is invalid with one of
  //    the cases:
  // 1. order_by_field is not set or has an unspecified field.
  // 2. Direction of ordering is not specified for the order_by_field.
  // 3. next_page_token cannot be decoded.
  virtual absl::Status ListArtifacts(const ListOperationOptions& options,
                                     std::vector<Artifact>* artifacts,
                                     std::string* next_page_token) = 0;

  // Gets executions stored in the metadata source using `options`.
  // `options` is the ListOperationOptions proto message defined
  // in metadata_store.
  // If successfull:
  // 1. `executions` is updated with result set of size determined by
  //    max_result_size set in `options`.
  // 2. `next_page_token` is populated with information necessary to fetch next
  //    page of results.
  // An empty result set is returned if no executions are retrieved.
  // RETURNS INVALID_ARGUMENT if the `options` is invalid with one of
  //    the cases:
  // 1. order_by_field is not set or has an unspecified field.
  // 2. Direction of ordering is not specified for the order_by_field.
  // 3. next_page_token cannot be decoded.
  virtual absl::Status ListExecutions(const ListOperationOptions& options,
                                      std::vector<Execution>* executions,
                                      std::string* next_page_token) = 0;

  // Gets contexts stored in the metadata source using `options`.
  // `options` is the ListOperationOptions proto message defined
  // in metadata_store.
  // If successfull:
  // 1. `contexts` is updated with result set of size determined by
  //    max_result_size set in `options`.
  // 2. `next_page_token` is populated with information necessary to fetch next
  //    page of results.
  // An empty result set is returned if no contexts are retrieved.
  // RETURNS INVALID_ARGUMENT if the `options` is invalid with one of
  //    the cases:
  // 1. order_by_field is not set or has an unspecified field.
  // 2. Direction of ordering is not specified for the order_by_field.
  // 3. next_page_token cannot be decoded.
  virtual absl::Status ListContexts(const ListOperationOptions& options,
                                    std::vector<Context>* contexts,
                                    std::string* next_page_token) = 0;

  // Gets an artifact by its type_id and name.
  // Returns NOT_FOUND error, if no artifact can be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindArtifactByTypeIdAndArtifactName(
      int64_t artifact_type_id, absl::string_view name, Artifact* artifact) = 0;

  // Gets artifacts by a given type_id.
  // Returns NOT_FOUND error, if the given artifact_type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindArtifactsByTypeId(
      int64_t artifact_type_id,
      std::optional<ListOperationOptions> list_options,
      std::vector<Artifact>* artifacts, std::string* next_page_token) = 0;

  // Gets artifacts by a given uri with exact match.
  // Returns NOT_FOUND error, if the given uri cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindArtifactsByURI(absl::string_view uri,
                                          std::vector<Artifact>* artifacts) = 0;

  // Updates an artifact.
  // The `last_update_time_since_epoch` field is determined under the hood
  //  and set to absl::Now().
  // If input artifact is the same as stored artifact, skip update operation and
  // return OK status.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no artifact is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ArtifactType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateArtifact(const Artifact& artifact) = 0;

  // Updates an artifact under masking.
  // If `mask` is empty, update `stored_node` as a whole.
  // If `mask` is not empty, only update fields specified in `mask`.
  // The `last_update_time_since_epoch` field is determined under the hood
  //  and set to absl::Now().
  // If input artifact is the same as stored artifact, skip update operation and
  // return OK status.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no artifact is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ArtifactType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateArtifact(
      const Artifact& artifact, const google::protobuf::FieldMask& mask) = 0;

  // Updates an artifact.
  // `update_timestamp` is used as the value of
  // Artifact.last_update_time_since_epoch.
  // When `force_update_time` is set to true, `last_update_time_since_epoch` is
  // updated even if input artifact is the same as stored artifact.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no artifact is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ArtifactType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateArtifact(const Artifact& artifact,
                                      absl::Time update_timestamp,
                                      bool force_update_time) = 0;

  // Updates an artifact under masking.
  // If `mask` is empty, update `stored_node` as a whole.
  // If `mask` is not empty, only update fields specified in `mask`.
  // `update_timestamp` is used as the value of
  // Artifact.last_update_time_since_epoch.
  // When `force_update_time` is set to true, `last_update_time_since_epoch` is
  // updated even if input artifact is the same as stored artifact.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no artifact is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ArtifactType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateArtifact(
      const Artifact& artifact, absl::Time update_timestamp,
      bool force_update_time, const google::protobuf::FieldMask& mask) = 0;

  // Creates an execution, returns the assigned execution id. The id field of
  // the execution is ignored.
  // `skip_type_and_property_validation` is set to be true if the `execution`'s
  // type and properties have been validated.
  // The `create_time_since_epoch` and `last_update_time_since_epoch` fields are
  // determined under the hood with absl::Now().
  // Returns INVALID_ARGUMENT error, if the ExecutionType is not given.
  // Returns NOT_FOUND error, if the ExecutionType cannot be found.
  // Returns INVALID_ARGUMENT error, if the execution contains any property
  //   undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the ExecutionType.
  // Returns ALREADY_EXISTS error, if the ExecutionType has execution with the
  // same name.
  // Returns ALREADY_EXISTS error, if there is execution with the
  // same external_id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status CreateExecution(const Execution& execution,
                                       bool skip_type_and_property_validation,
                                       int64_t* execution_id) = 0;

  // Creates an execution, returns the assigned execution id. The id field of
  // the execution is ignored.
  // `skip_type_and_property_validation` is set to be true if the `execution`'s
  // type and properties have been validated.
  // `create_timestamp` is used as the value of
  // Execution.create_time_since_epoch and
  // Execution.last_update_time_since_epoch.
  // Returns INVALID_ARGUMENT error, if the ExecutionType is not given.
  // Returns NOT_FOUND error, if the ExecutionType cannot be found.
  // Returns INVALID_ARGUMENT error, if the execution contains any property
  //   undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the ExecutionType.
  // Returns ALREADY_EXISTS error, if the ExecutionType has execution with the
  // same name.
  // Returns ALREADY_EXISTS error, if there is execution with the
  // same external_id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status CreateExecution(const Execution& execution,
                                       bool skip_type_and_property_validation,
                                       absl::Time create_timestamp,
                                       int64_t* execution_id) = 0;

  // Creates an execution, returns the assigned execution id. The id field of
  // the execution is ignored.
  // The `create_time_since_epoch` and `last_update_time_since_epoch` fields are
  // determined under the hood with absl::Now().
  // Please refer to the docstring for CreateExecution() with the
  // `skip_type_and_property_validation` flag for more details. This method
  // assumes the `execution`'s type/property has not been validated yet and
  // by setting `skip_type_and_property_validation` to false.
  virtual absl::Status CreateExecution(const Execution& execution,
                                       int64_t* execution_id) = 0;

  // Gets executions matching the given 'ids'.
  // Returns NOT_FOUND error, if any of the given ids are not found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindExecutionsById(
      absl::Span<const int64_t> execution_ids,
      std::vector<Execution>* executions) = 0;

  // Gets executions matching the given 'external_ids'.
  // |external_ids| is a list of non-null strings for the given external ids.
  // Returns whatever found when a part of |external_ids| is non-existing.
  // Returns NOT_FOUND error, if all the given external_ids are not found.
  // Returns INVALID_ARGUMENT error, if any of the |external_ids| is empty.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindExecutionsByExternalIds(
      absl::Span<absl::string_view> external_ids,
      std::vector<Execution>* executions) = 0;

  // Gets executions stored in the metadata source
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindExecutions(std::vector<Execution>* executions) = 0;

  // Gets an execution by its type_id and name.
  // Returns NOT_FOUND error, if no execution can be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindExecutionByTypeIdAndExecutionName(
      int64_t execution_type_id, absl::string_view name,
      Execution* execution) = 0;

  // Gets executions by a given type_id.
  // Returns NOT_FOUND error, if the given execution_type_id cannot be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindExecutionsByTypeId(
      int64_t execution_type_id,
      std::optional<ListOperationOptions> list_options,
      std::vector<Execution>* executions, std::string* next_page_token) = 0;

  // Updates an execution.
  // The `last_update_time_since_epoch` field is determined under the hood
  //  and set to absl::Now().
  // If input execution is the same as stored execution, skip update operation
  // and return OK status.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no execution is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ExecutionType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateExecution(const Execution& execution) = 0;

  // Updates an execution.
  // `update_timestamp` is used as the value of
  // Execution.last_update_time_since_epoch.
  // When `force_update_time` is set to true, `last_update_time_since_epoch` is
  // updated even if input execution is the same as stored execution.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no execution is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ExecutionType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateExecution(const Execution& execution,
                                       absl::Time update_timestamp,
                                       bool force_update_time) = 0;

  // Updates an execution under masking.
  // If `mask` is empty, update `stored_node` as a whole.
  // If `mask` is not empty, only update fields specified in `mask`.
  // When `force_update_time` is set to true, `last_update_time_since_epoch` is
  // updated even if input execution is the same as stored execution.
  // The `last_update_time_since_epoch` field is determined under the hood
  //  and set to absl::Now().
  // If input execution is the same as stored execution, skip update operation
  // and return OK status.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no execution is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ExecutionType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateExecution(
      const Execution& execution, bool force_update_time,
      const google::protobuf::FieldMask& mask) = 0;

  // Updates an execution under masking.
  // If `mask` is empty, update `stored_node` as a whole.
  // If `mask` is not empty, only update fields specified in `mask`.
  // `update_timestamp` is used as the value of
  // Execution.last_update_time_since_epoch.
  // When `force_update_time` is set to true, `last_update_time_since_epoch` is
  // updated even if input execution is the same as stored execution.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no execution is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ExecutionType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateExecution(
      const Execution& execution, absl::Time update_timestamp,
      bool force_update_time, const google::protobuf::FieldMask& mask) = 0;

  // Creates a context, returns the assigned context id. The id field of the
  // context is ignored. The name field of the context must not be empty and it
  // should be unique in the same ContextType.
  // `skip_type_and_property_validation` is set to be true if the `context`'s
  // type and properties have been validated.
  // The `create_time_since_epoch` and `last_update_time_since_epoch` fields are
  // determined under the hood with absl::Now().
  // Returns INVALID_ARGUMENT error, if the ContextType is not given.
  // Returns NOT_FOUND error, if the ContextType cannot be found.
  // Returns INVALID_ARGUMENT error, if the context name is empty.
  // Returns INVALID_ARGUMENT error, if the context contains any property
  //  undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the context type.
  // Returns ALREADY_EXISTS error, if the ContextType has context with the name.
  // Returns ALREADY_EXISTS error, if there is context with the
  // same external_id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status CreateContext(const Context& context,
                                     bool skip_type_and_property_validation,
                                     int64_t* context_id) = 0;

  // Creates a context, returns the assigned context id. The id field of the
  // context is ignored. The name field of the context must not be empty and it
  // should be unique in the same ContextType.
  // `skip_type_and_property_validation` is set to be true if the `context`'s
  // type and properties have been validated.
  // `create_timestamp` is used as the value of Context.create_time_since_epoch
  // and Context.last_update_time_since_epoch.
  // Returns INVALID_ARGUMENT error, if the ContextType is not given.
  // Returns NOT_FOUND error, if the ContextType cannot be found.
  // Returns INVALID_ARGUMENT error, if the context name is empty.
  // Returns INVALID_ARGUMENT error, if the context contains any property
  //  undefined in the type.
  // Returns INVALID_ARGUMENT error, if given value of a property does not match
  //   with its data type definition in the context type.
  // Returns ALREADY_EXISTS error, if the ContextType has context with the name.
  // Returns ALREADY_EXISTS error, if there is context with the
  // same external_id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status CreateContext(const Context& context,
                                     bool skip_type_and_property_validation,
                                     absl::Time create_timestamp,
                                     int64_t* context_id) = 0;

  // Creates a context, returns the assigned context id. The id field of the
  // context is ignored. The name field of the context must not be empty and it
  // should be unique in the same ContextType.
  // The `create_time_since_epoch` and `last_update_time_since_epoch` fields are
  // determined under the hood with absl::Now().
  // Please refer to the docstring for CreateContext() with the
  // `skip_type_and_property_validation` flag for more details. This method
  // assumes the `context`'s type/property has not been validated yet and
  // by setting `skip_type_and_property_validation` to false.
  virtual absl::Status CreateContext(const Context& context,
                                     int64_t* context_id) = 0;

  // Gets contexts matching a collection of ids.
  // Returns NOT_FOUND if any of the given ids are not found.
  // Returns detailed INTERNAL error if query execution fails.
  virtual absl::Status FindContextsById(absl::Span<const int64_t> context_ids,
                                        std::vector<Context>* context) = 0;

  // Gets contexts matching a collection of external_ids.
  // |external_ids| is a list of non-null strings for the given external ids.
  // Returns whatever found when a part of |external_ids| is non-existing.
  // Returns NOT_FOUND error, if all the given external_ids are not found.
  // Returns INVALID_ARGUMENT error, if any of the |external_ids| is empty.
  // Returns detailed INTERNAL error if query execution fails.
  virtual absl::Status FindContextsByExternalIds(
      absl::Span<absl::string_view> external_ids,
      std::vector<Context>* contexts) = 0;

  // Gets contexts stored in the metadata source
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindContexts(std::vector<Context>* contexts) = 0;

  // Gets contexts by a given type_id.
  // Returns NOT_FOUND error, if no context can be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindContextsByTypeId(
      int64_t type_id, std::optional<ListOperationOptions> list_options,
      std::vector<Context>* contexts, std::string* next_page_token) = 0;

  // Gets a context by a type_id and a context name. If id_only is true, the
  // returned context will contain only the id field.
  // Returns NOT_FOUND error, if no context can be found.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindContextByTypeIdAndContextName(int64_t type_id,
                                                         absl::string_view name,
                                                         bool id_only,
                                                         Context* context) = 0;

  // Updates a context.
  // The `last_update_time_since_epoch` field is determined under the hood
  //  and set to absl::Now().
  // If input context is the same as stored context, skip update operation and
  // return OK status.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no context is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ContextType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateContext(const Context& context) = 0;

  // Updates a context.
  // `update_timestamp` is used as the value of
  // Context.last_update_time_since_epoch.
  // When `force_update_time` is set to true, `last_update_time_since_epoch` is
  // updated even if input context is the same as stored context.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no context is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ContextType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateContext(const Context& context,
                                     absl::Time update_timestamp,
                                     bool force_update_time) = 0;

  // Updates a context under masking.
  // If `mask` is empty, update `stored_node` as a whole.
  // If `mask` is not empty, only update fields specified in `mask`.
  // The `last_update_time_since_epoch` field is determined under the hood
  //  and set to absl::Now().
  // If input context is the same as stored context, skip update operation and
  // return OK status.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no context is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ContextType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateContext(
      const Context& context, const google::protobuf::FieldMask& mask) = 0;

  // Updates a context under masking.
  // If `mask` is empty, update `stored_node` as a whole.
  // If `mask` is not empty, only update fields specified in `mask`.
  // `update_timestamp` is used as the value of
  // Context.last_update_time_since_epoch.
  // When `force_update_time` is set to true, `last_update_time_since_epoch` is
  // updated even if input context is the same as stored context.
  // Returns INVALID_ARGUMENT error, if the id field is not given.
  // Returns INVALID_ARGUMENT error, if no context is found with the given id.
  // Returns INVALID_ARGUMENT error, if type_id is given and is different from
  // the one stored.
  // Returns INVALID_ARGUMENT error, if given property names and types do not
  // align with the ContextType on file.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status UpdateContext(
      const Context& context, absl::Time update_timestamp,
      bool force_update_time, const google::protobuf::FieldMask& mask) = 0;

  // Creates an event, and returns the assigned event id. Please refer to the
  // docstring for CreateEvent() with the `is_already_validated` flag for more
  // details. This method assumes the event has not been validated yet and sets
  // `is_already_validated` to false.
  virtual absl::Status CreateEvent(const Event& event, int64_t* event_id) = 0;

  // Creates an event, returns the assigned event id. If the event occurrence
  // time is not given, the insertion time is used. If the event's execution
  // and artifact are known to exist in the database, set is_already_validated
  // to true to avoid redundant reads to the database for checking.
  // TODO(huimiao) Allow to have a unknown event time.
  // Returns INVALID_ARGUMENT error, if no artifact matches the artifact_id.
  // Returns INVALID_ARGUMENT error, if no execution matches the execution_id.
  // Returns INVALID_ARGUMENT error, if the type field is UNKNOWN.
  // Returns ALREADY_EXIST error, if duplicated event is found.
  // TODO(b/197686185): Deprecate this method once foreign keys schema is
  // implemented.
  virtual absl::Status CreateEvent(const Event& event,
                                   bool is_already_validated,
                                   int64_t* event_id) = 0;

  // Gets the events associated with a collection of artifact_ids.
  // Returns NOT_FOUND error, if no `events` can be found.
  // Returns INVALID_ARGUMENT error, if the `events` is null.
  virtual absl::Status FindEventsByArtifacts(
      absl::Span<const int64_t> artifact_ids, std::vector<Event>* events) = 0;

  // Gets the events associated with a collection of execution_ids.
  // Returns NOT_FOUND error, if no `events` can be found.
  // Returns INVALID_ARGUMENT error, if the `events` is null.
  virtual absl::Status FindEventsByExecutions(
      absl::Span<const int64_t> execution_ids, std::vector<Event>* events) = 0;

  // Creates an association, and returns the assigned association id.
  // Please refer to the docstring for CreateAssociation() with the
  // `is_already_validated` flag for more details. This method assumes the
  // association has not been validated yet and sets `is_already_validated` to
  // false.
  virtual absl::Status CreateAssociation(const Association& association,
                                         int64_t* association_id) = 0;

  // Creates an association, returns the assigned association id.
  // If the association's context and execution are known to exist in the
  // database, set `is_already_validated` to true to avoid redundant reads to
  // the database for checking.
  //
  // Returns INVALID_ARGUMENT error, if no context matches the context_id
  //   and `is_already_validated=false`.
  // Returns INVALID_ARGUMENT error, if no execution matches the execution_id.
  //   and `is_already_validated=false`.
  // Returns ALREADY_EXISTS error, if the same association already exists.
  // TODO(b/197686185): Deprecate this method once foreign keys schema is
  // implemented.
  virtual absl::Status CreateAssociation(const Association& association,
                                         bool is_already_validated,
                                         int64_t* association_id) = 0;

  // Gets the associations that context_ids are associated with.
  // Returns INVALID_ARGUMENT error, if the `associations` is null.
  // Returns NOT_FOUND error, if no associations are found.
  // Returns detailed INTERNAL error, if query execution fails.
  // TODO(b/203114828): Support a generic FindAssociations in MetadataStore for
  // executions.
  virtual absl::Status FindAssociationsByContexts(
      absl::Span<const int64_t> context_ids,
      std::vector<Association>* associations) = 0;

  // Gets the associations that `execution_ids` are associated with.
  // Returns an empty vector if no associations are found.
  // Returns INVALID_ARGUMENT error, if the `associations` is null.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindAssociationsByExecutions(
      absl::Span<const int64_t> execution_ids,
      std::vector<Association>* associations) = 0;

  // Gets the attributions that `artifact_ids` are attributed to.
  // Returns an empty vector if no attributions are found.
  // Returns INVALID_ARGUMENT error, if the `attributions` is null.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status FindAttributionsByArtifacts(
      absl::Span<const int64_t> artifact_ids,
      std::vector<Attribution>* attributions) = 0;

  // Gets the contexts that an execution_id is associated with.
  // Returns INVALID_ARGUMENT error, if the `contexts` is null.
  virtual absl::Status FindContextsByExecution(
      int64_t execution_id, std::vector<Context>* contexts) = 0;

  // Gets the executions associated with a context_id.
  // Returns INVALID_ARGUMENT error, if the `executions` is null.
  virtual absl::Status FindExecutionsByContext(
      int64_t context_id, std::vector<Execution>* executions) = 0;

  // Gets the executions associated with a context_id.
  // Returns INVALID_ARGUMENT error, if the `executions` is null.
  virtual absl::Status FindExecutionsByContext(
      int64_t context_id, std::optional<ListOperationOptions> list_options,
      std::vector<Execution>* executions, std::string* next_page_token) = 0;

  // Creates an attribution, and returns the assigned attribution id.
  // Please refer to the docstring for CreateAttribution() with the
  // `is_already_validated` flag for more details. This method assumes the
  // attribution has not been validated yet and sets `is_already_validated` to
  // false.
  virtual absl::Status CreateAttribution(const Attribution& attribution,
                                         int64_t* attribution_id) = 0;

  // Creates an attribution, returns the assigned attribution id.
  // If the attribution's context and artifact are known to exist in the
  // database, set `is_already_validated` to true to avoid redundant reads to
  // the database for checking.
  //
  // Returns INVALID_ARGUMENT error, if no context matches the context_id
  //   and `is_already_validated=false`.
  // Returns INVALID_ARGUMENT error, if no artifact matches the artifact_id.
  //   and `is_already_validated=false`.
  // Returns ALREADY_EXISTS error, if the same attribution already exists.
  // TODO(b/197686185): Deprecate this method once foreign keys schema is
  // implemented.
  virtual absl::Status CreateAttribution(const Attribution& attribution,
                                         bool is_already_validated,
                                         int64_t* attribution_id) = 0;

  // Gets the contexts that an artifact_id is attributed to.
  // Returns INVALID_ARGUMENT error, if the `contexts` is null.
  virtual absl::Status FindContextsByArtifact(
      int64_t artifact_id, std::vector<Context>* contexts) = 0;

  // Gets the artifacts attributed to a context_id.
  // Returns INVALID_ARGUMENT error, if the `artifacts` is null.
  virtual absl::Status FindArtifactsByContext(
      int64_t context_id, std::vector<Artifact>* artifacts) = 0;

  // Gets the artifacts attributed to a context_id.
  // If `list_options` is specified then results are paginated based on the
  // fields set in `list_options`.
  // Returns INVALID_ARGUMENT error, if the `artifacts` is null.
  virtual absl::Status FindArtifactsByContext(
      int64_t context_id, std::optional<ListOperationOptions> list_options,
      std::vector<Artifact>* artifacts, std::string* next_page_token) = 0;

  // Creates a parent context, returns OK if succeeds.
  // Returns INVALID_ARGUMENT error, if no context matches the child_id.
  // Returns INVALID_ARGUMENT error, if no context matches the parent_id.
  // Returns INVALID_ARGUMENT error, if child context and parent context
  // introduces cycle.
  // Returns ALREADY_EXISTS error, if the same parent context already exists.
  virtual absl::Status CreateParentContext(
      const ParentContext& parent_context) = 0;

  // Gets the parent-contexts of a context_id.
  // Returns INVALID_ARGUMENT error, if the `contexts` is null.
  virtual absl::Status FindParentContextsByContextId(
      int64_t context_id, std::vector<Context>* contexts) = 0;

  // Gets the child-contexts of a context_id.
  // Returns INVALID_ARGUMENT error, if the `contexts` is null.
  virtual absl::Status FindChildContextsByContextId(
      int64_t context_id, std::vector<Context>* contexts) = 0;

  // Gets the parent-contexts of child context_ids list.
  // Returns INVALID_ARGUMENT error, if `context_ids` is empty.
  virtual absl::Status FindParentContextsByContextIds(
      absl::Span<const int64_t> context_ids,
      absl::node_hash_map<int64_t, std::vector<Context>>& contexts) = 0;

  // Gets the child-contexts of parent context_ids list.
  // Returns INVALID_ARGUMENT error, if `context_ids` is empty.
  virtual absl::Status FindChildContextsByContextIds(
      absl::Span<const int64_t> context_ids,
      absl::node_hash_map<int64_t, std::vector<Context>>& contexts) = 0;

  // Resolves the schema version stored in the metadata source. The `db_version`
  // is set to 0, if it is a 0.13.2 release pre-existing database.
  // Returns DATA_LOSS error, if schema version info table exists but its value
  //   cannot be resolved from the database.
  // Returns NOT_FOUND error, if the database is empty.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status GetSchemaVersion(int64_t* db_version) = 0;

  // The version of the current query config or source. Increase the version by
  // 1 in any CL that includes physical schema changes and provides a migration
  // function that uses a list migration queries. The database stores it to
  // indicate the current database version. When metadata source is connected to
  // the database it can compare the given library `schema_version` in the query
  // config with the `schema_version` stored in the database, and migrate the
  // database if needed.
  virtual int64_t GetLibraryVersion() = 0;


  // Dependent of `CloudMetadataStore::GetLineageSubgraphByArtifact`.
  // Giving a set of `query_nodes` and a set of boundary constraints. The method
  // performs constrained transitive closure and returns a subgraph including
  // the reached nodes and edges. The boundary conditions include
  // a) number of hops: it stops traversal at nodes that at max_num_hops away
  //    from the `query_nodes`.
  // b) boundary nodes: it stops traversal at the nodes that satisfies
  //    `boundary_artifacts` or `boundary_executions`.
  // c) number of total nodes: it stops traversal once total nodes meets
  // max_nodes. No limits on total nodes if max_nodes is not set.
  virtual absl::Status QueryLineageGraph(
      const std::vector<Artifact>& query_nodes, int64_t max_num_hops,
      std::optional<int64_t> max_nodes,
      std::optional<std::string> boundary_artifacts,
      std::optional<std::string> boundary_executions,
      LineageGraph& subgraph) = 0;

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
  // Returns INVALID_ARGUMENT error, if no paths are specified in `read_mask`.
  // Returns INVALID_ARGUMENT error, if `starting_nodes` is not specified in
  // `lineage_subgraph_query_options`.
  // Returns INVALID_ARGUMENT error, if `starting_nodes.filter_query` is
  // unspecified or invalid in `lineage_subgraph_query_options`.
  // Returns detailed INTERNAL error, if the operation fails.
  virtual absl::Status QueryLineageSubgraph(
      const LineageSubgraphQueryOptions& options,
      const google::protobuf::FieldMask& read_mask, LineageGraph& subgraph) = 0;


  // Deletes a list of artifacts by id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteArtifactsById(
      absl::Span<const int64_t> artifact_ids) = 0;

  // Deletes a list of executions by id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteExecutionsById(
      absl::Span<const int64_t> execution_ids) = 0;

  // Deletes a list of contexts by id.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteContextsById(
      absl::Span<const int64_t> context_ids) = 0;

  // Deletes the events corresponding to the |artifact_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteEventsByArtifactsId(
      absl::Span<const int64_t> artifact_ids) = 0;

  // Deletes the events corresponding to the |execution_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteEventsByExecutionsId(
      absl::Span<const int64_t> execution_ids) = 0;

  // Deletes the associations corresponding to the |context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteAssociationsByContextsId(
      absl::Span<const int64_t> context_ids) = 0;

  // Deletes the associations corresponding to the |execution_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteAssociationsByExecutionsId(
      absl::Span<const int64_t> execution_ids) = 0;

  // Deletes the attributions corresponding to the |context_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteAttributionsByContextsId(
      absl::Span<const int64_t> context_ids) = 0;

  // Deletes the attributions corresponding to the |artifact_ids|.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteAttributionsByArtifactsId(
      absl::Span<const int64_t> artifact_ids) = 0;

  // Deletes the parent contexts corresponding to the |parent_context_ids|.
  virtual absl::Status DeleteParentContextsByParentIds(
      absl::Span<const int64_t> parent_context_ids) = 0;

  // Deletes the parent contexts corresponding to the |child_context_ids|.
  virtual absl::Status DeleteParentContextsByChildIds(
      absl::Span<const int64_t> child_context_ids) = 0;

  // Deletes the parent contexts corresponding to the |parent_context_id|
  // and |child_context_ids|.
  // Nothing will be deleted if |child_context_ids| is empty.
  // Returns detailed INTERNAL error, if query execution fails.
  virtual absl::Status DeleteParentContextsByParentIdAndChildIds(
      int64_t parent_context_id,
      absl::Span<const int64_t> child_context_ids) = 0;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_METADATA_ACCESS_OBJECT_H_
