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
#include "ml_metadata/metadata_store/metadata_store.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include "google/protobuf/field_mask.pb.h"
#include "google/protobuf/util/message_differencer.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/metadata_access_object.h"
#include "ml_metadata/metadata_store/metadata_access_object_factory.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/metadata_store/rdbms_metadata_access_object.h"
#include "ml_metadata/metadata_store/simple_types_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/simple_types/proto/simple_types.pb.h"
#include "ml_metadata/simple_types/simple_types_constants.h"
#include "ml_metadata/util/return_utils.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "google/protobuf/unknown_field_set.h"

namespace ml_metadata {
namespace {
using std::unique_ptr;

// Checks if the `stored_type` and `other_type` have the same names.
// In addition, it checks whether the types are inconsistent:
// a) `stored_type` and `other_type` have conflicting property value type
// b) `can_omit_fields` is false, while `stored_type`/`other_type` is not empty.
// c) `can_add_fields` is false, while `other_type`/`store_type` is not empty.
// Returns OK if the types are consistent and an output_type that contains the
// union of the properties in stored_type and other_type.
// Returns FAILED_PRECONDITION, if the types are inconsistent.
template <typename T>
absl::Status CheckFieldsConsistent(const T& stored_type, const T& other_type,
                                   bool can_add_fields, bool can_omit_fields,
                                   T& output_type) {
  if (stored_type.name() != other_type.name()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Conflicting type name found in stored and given types: "
        "stored type: ",
        stored_type.DebugString(), "; given type: ", other_type.DebugString()));
  }
  // Make sure every property in stored_type matches with the one in other_type
  // unless can_omit_fields is set to true.
  int omitted_fields_count = 0;
  for (const auto& pair : stored_type.properties()) {
    const std::string& key = pair.first;
    const PropertyType value = pair.second;
    const auto other_iter = other_type.properties().find(key);
    if (other_iter == other_type.properties().end()) {
      omitted_fields_count++;
    } else if (other_iter->second != value) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Conflicting property value type found in stored and given types: "
          "stored_type: ",
          stored_type.DebugString(),
          "; other_type: ", other_type.DebugString()));
    }
    if (omitted_fields_count > 0 && !can_omit_fields) {
      return absl::FailedPreconditionError(absl::StrCat(
          "can_omit_fields is false while stored type has more properties: "
          "stored type: ",
          stored_type.DebugString(),
          "; given type: ", other_type.DebugString()));
    }
  }
  if (stored_type.properties_size() - omitted_fields_count ==
      other_type.properties_size()) {
    output_type = stored_type;
    if (stored_type.external_id() != other_type.external_id()) {
      output_type.set_external_id(other_type.external_id());
    }
    return absl::OkStatus();
  }
  if (!can_add_fields) {
    return absl::FailedPreconditionError(absl::StrCat(
        "can_add_fields is false while the given type has more properties: "
        "stored_type: ",
        stored_type.DebugString(), "; other_type: ", other_type.DebugString()));
  }
  // add new properties to output_types if can_add_fields is true.
  output_type = stored_type;
  for (const auto& pair : other_type.properties()) {
    const std::string& property_name = pair.first;
    const PropertyType value = pair.second;
    if (stored_type.properties().find(property_name) ==
        stored_type.properties().end()) {
      (*output_type.mutable_properties())[property_name] = value;
    }
  }
  if (stored_type.external_id() != other_type.external_id()) {
    output_type.set_external_id(other_type.external_id());
  }
  return absl::OkStatus();
}

// Creates a type inheritance link between 'type'.base_type from request and
// type with 'type_id'.
//
// a) If 'type'.base_type = NULL in request 'type', no-op;
// b) If 'type'.base_type = UNSET in request 'type', error out as type deletion
//    is not supported yet;
// c) If more than 1 parent types are found for 'type_id', return error;
// d) If 'type'.base_type is set,
//    1) If no parent type is found for 'type_id', create a new parent
//       inheritance link;
//    2) If 1 parent type is found for 'type_id' and it is not equal to
//       'type'.base_type, error out as type update is not supported yet;
//    3) If 1 parent type is found for 'type_id' and it is equal to
//       'type'.base_type, no-op.
// TODO(b/195375645): support parent type update and deletion
template <typename T>
absl::Status UpsertTypeInheritanceLink(
    const T& type, int64_t type_id,
    MetadataAccessObject* metadata_access_object) {
  if (!type.has_base_type()) return absl::OkStatus();

  SystemTypeExtension extension;
  MLMD_RETURN_IF_ERROR(GetSystemTypeExtension(type, extension));
  if (IsUnsetBaseType(extension)) {
    return absl::UnimplementedError("base_type deletion is not supported yet");
  }
  absl::flat_hash_map<int64_t, T> output_parent_types;
  MLMD_RETURN_IF_ERROR(metadata_access_object->FindParentTypesByTypeId(
      {type_id}, output_parent_types));

  const bool no_parent_type = !output_parent_types.contains(type_id);
  if (no_parent_type) {
    T type_with_id = type;
    type_with_id.set_id(type_id);
    T base_type;
    MLMD_RETURN_IF_ERROR(metadata_access_object->FindTypeByNameAndVersion(
        extension.type_name(), /*version=*/absl::nullopt, &base_type));
    return metadata_access_object->CreateParentTypeInheritanceLink(type_with_id,
                                                                   base_type);
  }

  if (output_parent_types[type_id].name() != extension.type_name()) {
    return absl::UnimplementedError("base_type update is not supported yet");
  }
  return absl::OkStatus();
}

// For each type, if there is no type having the same name and version, then
// inserts a new type. If a type with the same name and version already exists
// (let's call it `old_type`), it checks the consistency of `type` and
// `old_type` as described in CheckFieldsConsistent according to
// can_add_fields and can_omit_fields.
// It returns ALREADY_EXISTS if any `type` insides `types`:
//  a) any property in `type` has different value from the one in `old_type`
//  b) can_add_fields = false, `type` has more properties than `old_type`
//  c) can_omit_fields = false, `type` has less properties than `old_type`
// If `type` is a valid update, then new fields in `type` are added.
// Returns INVALID_ARGUMENT error, if name field in `type` is not given.
// Returns INVALID_ARGUMENT error, if any property type in `type` is unknown.
// Returns detailed INTERNAL error, if query execution fails.
template <typename T>
absl::Status UpsertTypes(const google::protobuf::RepeatedPtrField<T>& types,
                         const bool can_add_fields, const bool can_omit_fields,
                         MetadataAccessObject* metadata_access_object,
                         std::vector<int64_t>& type_ids) {
  if (types.empty()) return absl::OkStatus();
  std::vector<T> stored_types;
  std::vector<std::pair<std::string, std::string>> names_and_versions;
  absl::c_transform(types, std::back_inserter(names_and_versions),
                    [](const T& type) {
                      return std::make_pair(type.name(), type.version());
                    });
  MLMD_RETURN_IF_ERROR(metadata_access_object->FindTypesByNamesAndVersions(
      absl::MakeSpan(names_and_versions), stored_types));

  absl::flat_hash_map<
      std::pair<absl::string_view, absl::string_view>, T>
      name_and_version_to_stored_type;
  for (const T& type : stored_types) {
    name_and_version_to_stored_type.insert(
        {{type.name(), type.version()}, type});
  }

  for (const T& type : types) {
    const std::pair<absl::string_view, absl::string_view> key{
        type.name(), type.version()};
    int64_t type_id;
    const auto iter = name_and_version_to_stored_type.find(key);
    if (iter == name_and_version_to_stored_type.end()) {
      // if not found, then it creates a type. `can_add_fields` is ignored.
      MLMD_RETURN_IF_ERROR(
          metadata_access_object->CreateType(type, &type_id));
      T stored_type(type);
      stored_type.set_id(type_id);
      name_and_version_to_stored_type.insert({key, stored_type});
    } else {
      // otherwise it checks if fields are consistent.
      T stored_type = iter->second;
      type_id = stored_type.id();
      T output_type;
      const absl::Status check_status = CheckFieldsConsistent(
          stored_type, type, can_add_fields, can_omit_fields, output_type);
      if (!check_status.ok()) {
        return absl::AlreadyExistsError(
            absl::StrCat("Type already exists with different properties: ",
                          std::string(check_status.message())));
      }
      // only update if the type is different from the stored type.
      if (!google::protobuf::util::MessageDifferencer::Equals(output_type,
                                                    stored_type)) {
        MLMD_RETURN_IF_ERROR(metadata_access_object->UpdateType(output_type));
      }
    }
    MLMD_RETURN_IF_ERROR(
        UpsertTypeInheritanceLink(type, type_id, metadata_access_object));
    type_ids.push_back(type_id);
  }
  return absl::OkStatus();
}

// Inserts or updates all the types in the argument list. 'can_add_fields' and
// 'can_omit_fields' are both enabled. Type ids are inserted into the
// PutTypesResponse 'response'.
absl::Status UpsertTypes(
    const google::protobuf::RepeatedPtrField<ArtifactType>& artifact_types,
    const google::protobuf::RepeatedPtrField<ExecutionType>& execution_types,
    const google::protobuf::RepeatedPtrField<ContextType>& context_types,
    const bool can_add_fields, const bool can_omit_fields,
    MetadataAccessObject* metadata_access_object, PutTypesResponse* response) {
  std::vector<int64_t> type_ids;
  MLMD_RETURN_IF_ERROR(UpsertTypes(artifact_types, can_add_fields,
                                   can_omit_fields, metadata_access_object,
                                   type_ids));
  response->mutable_artifact_type_ids()->Add(type_ids.begin(), type_ids.end());
  type_ids.clear();
  MLMD_RETURN_IF_ERROR(UpsertTypes(execution_types, can_add_fields,
                                   can_omit_fields, metadata_access_object,
                                   type_ids));
  response->mutable_execution_type_ids()->Add(type_ids.begin(), type_ids.end());
  type_ids.clear();
  MLMD_RETURN_IF_ERROR(UpsertTypes(context_types, can_add_fields,
                                   can_omit_fields, metadata_access_object,
                                   type_ids));
  response->mutable_context_type_ids()->Add(type_ids.begin(), type_ids.end());
  return absl::OkStatus();
}

// Loads SimpleTypes proto from string and updates or inserts it into database.
absl::Status UpsertSimpleTypes(MetadataAccessObject* metadata_access_object) {
  SimpleTypes simple_types;
  PutTypesResponse response;
  MLMD_RETURN_IF_ERROR(LoadSimpleTypes(simple_types));
  return UpsertTypes(
      simple_types.artifact_types(), simple_types.execution_types(),
      simple_types.context_types(), /*can_add_fields=*/true,
      /*can_omit_fields=*/true, metadata_access_object, &response);
}

// Updates or inserts an artifact.
// If the artifact.id is given, it updates the stored artifact, otherwise,
// it creates a new artifact.
// While creating a new artifact, `mask` will be ignored.
// While updating an existing artifact, the update can be performed under
// masking.
// If artifact.id is given and `mask` is empty, it updates `stored_node` as a
// whole.
// If artifact.id is given and `mask` is not empty, it only updates
// fields specified in `mask`.
// `skip_type_and_property_validation` is set to be true if the `artifact`'s
// type/property has been validated.
// If reuse_artifact_if_already_exist_by_external_id=true, it will first query
// with artifact.external_id to see if there is existing artifact. If there is
// existing artifact, repopulate artifact.id as if it's provided to perform an
// update. If there is no existing artifact, continue to insert.
absl::Status UpsertArtifact(const Artifact& artifact,
                            MetadataAccessObject* metadata_access_object,
                            bool skip_type_and_property_validation,
                            const google::protobuf::FieldMask& mask,
                            bool reuse_artifact_if_already_exist_by_external_id,
                            int64_t* artifact_id) {
  CHECK(artifact_id) << "artifact_id should not be null";

  Artifact artifact_copy_to_be_upserted(artifact);
  // Try to reuse existing artifact if the options is set to true and has
  // non-empty external_id by querying and populating
  // artifact_copy_to_be_upserted.id.
  if (reuse_artifact_if_already_exist_by_external_id && !artifact.has_id() &&
      artifact.has_external_id() && !artifact.external_id().empty()) {
    std::vector<absl::string_view> artifact_external_ids = {
        artifact.external_id()};
    std::vector<Artifact> artifacts;
    const absl::Status status =
        metadata_access_object->FindArtifactsByExternalIds(
            absl::MakeSpan(artifact_external_ids), &artifacts);
    if (!absl::IsNotFound(status)) {
      MLMD_RETURN_IF_ERROR(status);
      // Found the artifact by external_id. Use it as artifact_id to return.
      artifact_copy_to_be_upserted.set_id(artifacts[0].id());
    }
  }

  if (artifact_copy_to_be_upserted.has_id()) {
    MLMD_RETURN_IF_ERROR(metadata_access_object->UpdateArtifact(
        artifact_copy_to_be_upserted, mask));
    *artifact_id = artifact_copy_to_be_upserted.id();
  } else {
    MLMD_RETURN_IF_ERROR(metadata_access_object->CreateArtifact(
        artifact_copy_to_be_upserted, skip_type_and_property_validation,
        artifact_id));
  }
  return absl::OkStatus();
}

// Updates or inserts an execution.
// If the execution.id is given, it updates the stored execution,
// otherwise, it creates a new execution.
// While creating a new execution, `mask` will be ignored.
// While updating an existing execution, the update can be performed under
// masking.
// If execution.id is given and `mask` is empty, it updates `stored_node` as a
// whole.
// If execution.id is given and `mask` is not empty, it only updates
// fields specified in `mask`.
// When `skip_type_and_property_validation` is set to true, skip the validations
// of `execution` type and properties.
// When `force_update_time` is set to true, `last_update_time_since_epoch` is
// updated even if input execution is the same as stored execution.
absl::Status UpsertExecution(const Execution& execution,
                             MetadataAccessObject* metadata_access_object,
                             const bool skip_type_and_property_validation,
                             const bool force_update_time,
                             const google::protobuf::FieldMask& mask,
                             int64_t* execution_id) {
  CHECK(execution_id) << "execution_id should not be null";
  if (execution.has_id()) {
    MLMD_RETURN_IF_ERROR(metadata_access_object->UpdateExecution(
        execution, force_update_time, mask));
    *execution_id = execution.id();
  } else {
    MLMD_RETURN_IF_ERROR(metadata_access_object->CreateExecution(
        execution, skip_type_and_property_validation, execution_id));
  }
  return absl::OkStatus();
}

// Updates or inserts a context.
// If the context.id is given, it updates the stored context,
// otherwise, it creates a new context.
// While creating a new context, `mask` will be ignored.
// While updating an existing context, the update can be performed under
// masking.
// If context.id is given and `mask` is empty, it updates `stored_node` as a
// whole.
// If context.id is given and `mask` is not empty, it only updates
// fields specified in `mask`.
// `skip_type_and_property_validation` is set to be true if the `context`'s
// type/property has been validated.
absl::Status UpsertContext(const Context& context,
                           MetadataAccessObject* metadata_access_object,
                           const bool skip_type_and_property_validation,
                           const google::protobuf::FieldMask& mask,
                           int64_t* context_id) {
  CHECK(context_id) << "context_id should not be null";
  if (context.has_id()) {
    MLMD_RETURN_IF_ERROR(metadata_access_object->UpdateContext(context, mask));
    *context_id = context.id();
  } else {
    MLMD_RETURN_IF_ERROR(metadata_access_object->CreateContext(
        context, skip_type_and_property_validation, context_id));
  }
  return absl::OkStatus();
}

// Updates, inserts, or finds context. If `reuse_context_if_already_exist`, it
// tries to find the existing context before trying to upsert the context.
// `skip_type_and_property_validation` is set to be true if the `context`'s
// type/property has been validated.
absl::Status UpsertContextWithOptions(
    const Context& context, MetadataAccessObject* metadata_access_object,
    bool reuse_context_if_already_exist,
    const bool skip_type_and_property_validation, int64_t* context_id) {
  CHECK(context_id) << "context_id should not be null";

  if (!context.has_type_id()) {
    return absl::InvalidArgumentError(
      absl::StrCat("Context is missing a type_id: ", context.DebugString())
    );
  }
  if (!context.has_name()) {
    return absl::InvalidArgumentError(
      absl::StrCat("Context is missing a name: ", context.DebugString())
    );
  }

  // Try to reuse existing context if the options is set.
  if (reuse_context_if_already_exist && !context.has_id()) {
    Context id_only_context;
    const absl::Status status =
        metadata_access_object->FindContextByTypeIdAndContextName(
            context.type_id(), context.name(), /*id_only=*/true,
            &id_only_context);
    if (!absl::IsNotFound(status)) {
      MLMD_RETURN_IF_ERROR(status);
      *context_id = id_only_context.id();
    }
  }
  if (*context_id == -1) {
    const absl::Status status = UpsertContext(
        context, metadata_access_object, skip_type_and_property_validation,
        google::protobuf::FieldMask(), context_id);
    // When `reuse_context_if_already_exist`, there are concurrent timelines
    // to create the same new context. If use the option, let client side
    // to retry the failed transaction safely.
    if (reuse_context_if_already_exist && absl::IsAlreadyExists(status)) {
      return absl::AbortedError(absl::StrCat(
          "Concurrent creation of the same context at the first time. "
          "Retry the transaction to reuse the context: ",
          context.DebugString()));
    }
    MLMD_RETURN_IF_ERROR(status);
  }
  return absl::OkStatus();
}

// Inserts an association. If the association already exists it returns OK.
// TODO(b/197686185): Remove `is_already_validated` parameter once foreign key
// schema is implemented.
absl::Status InsertAssociationIfNotExist(
    int64_t context_id, int64_t execution_id, bool is_already_validated,
    MetadataAccessObject* metadata_access_object) {
  Association association;
  association.set_execution_id(execution_id);
  association.set_context_id(context_id);
  int64_t dummy_association_id;
  absl::Status status = metadata_access_object->CreateAssociation(
      association, /*is_already_validated=*/is_already_validated,
      &dummy_association_id);
  if (!status.ok() && !absl::IsAlreadyExists(status)) {
    return status;
  }
  return absl::OkStatus();
}

// Inserts an attribution. If the attribution already exists it returns OK.
// TODO(b/197686185): Remove `is_already_validated` parameter once foreign key
// schema is implemented.
absl::Status InsertAttributionIfNotExist(
    int64_t context_id, int64_t artifact_id, bool is_already_validated,
    MetadataAccessObject* metadata_access_object) {
  Attribution attribution;
  attribution.set_artifact_id(artifact_id);
  attribution.set_context_id(context_id);
  int64_t dummy_attribution_id;
  absl::Status status = metadata_access_object->CreateAttribution(
      attribution, /*is_already_validated=*/is_already_validated,
      &dummy_attribution_id);
  if (!status.ok() && !absl::IsAlreadyExists(status)) {
    return status;
  }
  return absl::OkStatus();
}

// Inserts an Event in a provided ArtifactAndEvent.
// If there is no Event in ArtifactAndEvent, return OkStatus().
// If an Event is provided in ArtifactAndEvent, will do some validation check
// and then create a Event.
absl::Status InsertEvent(
    const PutExecutionRequest::ArtifactAndEvent& artifact_and_event,
    const int64_t execution_id, MetadataAccessObject* metadata_access_object) {
  if (!artifact_and_event.has_event()) {
    return absl::OkStatus();
  }

  // Validate execution and event.
  Event event(artifact_and_event.event());
  if (event.has_execution_id() && event.execution_id() != execution_id) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Request's event.execution_id does not match with the given "
        "execution: "));
  }
  event.set_execution_id(execution_id);

  // Validate artifact and event.
  if (artifact_and_event.has_artifact()) {
    if (!artifact_and_event.artifact().has_id()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Given artifact does not have an id: ",
                       artifact_and_event.DebugString()));
    }

    if (event.has_artifact_id() &&
        event.artifact_id() != artifact_and_event.artifact().id()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Given event.artifact_id is not aligned with the artifact: ",
          artifact_and_event.DebugString()));
    }

    event.set_artifact_id(artifact_and_event.artifact().id());
  } else if (!event.has_artifact_id()) {
    return absl::InvalidArgumentError(
        absl::StrCat("If no artifact is present, given event must have an "
                     "artifact_id: ",
                     artifact_and_event.DebugString()));
  }

  // Create an Event.
  int64_t dummy_event_id = -1;
  MLMD_RETURN_IF_ERROR(metadata_access_object->CreateEvent(
      event,
      /*is_already_validated=*/true, &dummy_event_id));

  return absl::OkStatus();
}

// Gets the <external_id, id> mapping from input artifacts by querying the db.
absl::Status GetExternalIdToIdMapping(
    absl::Span<const Artifact> artifacts,
    MetadataAccessObject* metadata_access_object,
    absl::flat_hash_map<std::string, int64_t>& output_external_id_to_id_map) {
  std::vector<absl::string_view> external_ids;
  for (const Artifact& artifact : artifacts) {
    if (artifact.has_external_id() && !artifact.external_id().empty()) {
      external_ids.push_back(absl::string_view(artifact.external_id()));
    }
  }
  std::vector<Artifact> artifacts_to_reuse;
  absl::Status status = metadata_access_object->FindArtifactsByExternalIds(
      absl::MakeSpan(external_ids), &artifacts_to_reuse);
  if (!(status.ok() || absl::IsNotFound(status))) {
    return status;
  }
  for (const Artifact& artifact : artifacts_to_reuse) {
    output_external_id_to_id_map.insert(
        {artifact.external_id(), artifact.id()});
  }
  return absl::OkStatus();
}

// A util to handle type_version in type read/write API requests.
template <typename T>
std::optional<std::string> GetRequestTypeVersion(const T& type_request) {
  return type_request.has_type_version() && !type_request.type_version().empty()
             ? absl::make_optional(type_request.type_version())
             : absl::nullopt;
}

// Sets base_type field in `type` with its parent type queried from ParentType
// table.
// Returns FAILED_PRECONDITION if there are more than 1 system type.
// TODO(b/153373285): consider moving it to FindTypesFromRecordSet at MAO layer
template <typename T, typename ST>
absl::Status SetBaseType(absl::Span<T* const> types,
                         MetadataAccessObject* metadata_access_object) {
  if (types.empty()) return absl::OkStatus();
  absl::flat_hash_map<int64_t, T> output_parent_types;
  std::vector<int64_t> type_ids;
  absl::c_transform(types, std::back_inserter(type_ids),
                    [](const T* type) { return type->id(); });
  MLMD_RETURN_IF_ERROR(metadata_access_object->FindParentTypesByTypeId(
      type_ids, output_parent_types));

  for (T* type : types) {
    if (output_parent_types.find(type->id()) == output_parent_types.end())
      continue;
    auto parent_type = output_parent_types[type->id()];
    SystemTypeExtension extension;
    extension.set_type_name(parent_type.name());
    ST type_enum;
    MLMD_RETURN_IF_ERROR(GetSystemTypeEnum(extension, type_enum));
    type->set_base_type(type_enum);
  }
  return absl::OkStatus();
}

// Verifies that the input PutLineageSubgraphRequest has valid EventEdges.
absl::Status CheckEventEdges(const PutLineageSubgraphRequest& request) {
  for (const auto& event_edge : request.event_edges()) {
    if (!event_edge.has_event()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "No event found for event_edge: ", request.DebugString()));
    }
    if (!event_edge.has_execution_index() &&
        !event_edge.event().has_execution_id()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Event_edge is missing an execution. Could not "
                       "find either an execution_index or an "
                       "event.execution_id: ",
                       event_edge.DebugString()));
    }
    if (event_edge.has_execution_index() &&
        (event_edge.execution_index() < 0 ||
         event_edge.execution_index() >= request.executions_size())) {
      return absl::OutOfRangeError(absl::StrCat(
          "The event_edge has an execution_index ",
          event_edge.execution_index(),
          " that is outside the bounds of the request's executions list, [0, ",
          request.executions_size(),
          "). Event_edge: ", event_edge.DebugString()));
    }
    if (event_edge.has_execution_index() &&
        event_edge.event().has_execution_id()) {
      const Execution& execution =
          request.executions(event_edge.execution_index());

      if (!execution.has_id()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Missing execution ID. The referenced execution at "
            "execution_index ",
            event_edge.execution_index(),
            " does not have an ID even though the event_edge specifies an "
            "event.execution_id of ",
            event_edge.event().execution_id(),
            ". Execution: ", execution.DebugString(),
            ", Event_edge: ", event_edge.DebugString()));
      }

      if (execution.id() != event_edge.event().execution_id()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Execution ID mismatch. The referenced execution at "
            "execution_index ",
            event_edge.execution_index(), " has an ID of ", execution.id(),
            " that does not match the event_edge's event.execution_id of ",
            event_edge.event().execution_id(),
            ". Artifact: ", execution.DebugString(),
            ", Event_edge: ", event_edge.DebugString()));
      }
    }
    if (!event_edge.has_artifact_index() &&
        !event_edge.event().has_artifact_id()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Event_edge is missing an artifact. Could not "
                       "find either an artifact_index or an "
                       "event.artifact_id: ",
                       event_edge.DebugString()));
    }
    if (event_edge.has_artifact_index() &&
        (event_edge.artifact_index() < 0 ||
         event_edge.artifact_index() >= request.artifacts_size())) {
      return absl::OutOfRangeError(absl::StrCat(
          "The event_edge has an artifact_index ", event_edge.artifact_index(),
          " that is outside the bounds of the request's artifacts list, [0, ",
          request.artifacts_size(),
          "). Event_edge: ", event_edge.DebugString()));
    }
    if (event_edge.has_artifact_index() &&
        event_edge.event().has_artifact_id()) {
      const Artifact& artifact = request.artifacts(event_edge.artifact_index());

      if (!artifact.has_id()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Missing artifact ID. The referenced artifact at artifact_index ",
            event_edge.artifact_index(),
            " does not have an ID even though the event_edge specifies an "
            "event.artifact_id of ",
            event_edge.event().artifact_id(),
            ". Artifact: ", artifact.DebugString(),
            ", Event_edge: ", event_edge.DebugString()));
      }

      if (artifact.id() != event_edge.event().artifact_id()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Artifact ID mismatch. The referenced artifact at artifact_index ",
            event_edge.artifact_index(), " has an ID of ", artifact.id(),
            " that does not match the event_edge's event.artifact_id of ",
            event_edge.event().artifact_id(),
            ". Artifact: ", artifact.DebugString(),
            ", Event_edge: ", event_edge.DebugString()));
      }
    }
  }

  return absl::OkStatus();
}

// Validates the properties of all `nodes` with corresponding `types`.
template <typename Type, typename Node>
absl::Status ValidateNodesPropertyWithTypes(
    const std::vector<Type>& types,
    const google::protobuf::RepeatedPtrField<Node>& nodes) {
  absl::flat_hash_map<int64_t, Type> type_id_to_type;
  for (const Type& type : types) {
    type_id_to_type.insert({type.id(), type});
  }

  for (const Node& node : nodes) {
    MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
        ml_metadata::ValidatePropertiesWithType(
            node, type_id_to_type[node.type_id()]),
        "Cannot validate properties of ", node.ShortDebugString());
  }

  return absl::OkStatus();
}

// Batch validates type existance and property matching for `nodes`.
template <typename Type, typename Node>
absl::Status BatchTypeAndPropertyValidation(
    const google::protobuf::RepeatedPtrField<Node>& nodes,
    MetadataAccessObject* metadata_access_object) {
  if (nodes.empty()) {
    return absl::OkStatus();
  }

  std::vector<int64_t> type_ids;
  std::vector<Type> types;
  for (const auto& node : nodes) {
    if (!node.has_type_id()) {
      return absl::InvalidArgumentError("Type id is missing.");
    }
    type_ids.push_back(node.type_id());
  }

  // Validates types.
  MLMD_RETURN_IF_ERROR(metadata_access_object->FindTypesByIds(type_ids, types));
  // Validates properties.
  MLMD_RETURN_IF_ERROR(ValidateNodesPropertyWithTypes(types, nodes));
  return absl::OkStatus();
}

}  // namespace

absl::Status MetadataStore::InitMetadataStore() {
  TransactionOptions options;
  options.set_tag("InitMetadataStore");
  MLMD_RETURN_IF_ERROR(transaction_executor_->Execute(
      [this]() -> absl::Status {
        return metadata_access_object_->InitMetadataSource();
      },
      options));
  options.set_tag("InitMetadataStore_UpsertSimpleTypes");
  return transaction_executor_->Execute(
      [this]() -> absl::Status {
        return UpsertSimpleTypes(metadata_access_object_.get());
      },
      options);
}

// TODO(b/187357155): duplicated results when inserting simple types
// concurrently
absl::Status MetadataStore::InitMetadataStoreIfNotExists(
    const bool enable_upgrade_migration) {
  TransactionOptions options;
  options.set_tag("InitMetadataStoreIfNotExists");
  MLMD_RETURN_IF_ERROR(transaction_executor_->Execute(
      [this, &enable_upgrade_migration]() -> absl::Status {
        return metadata_access_object_->InitMetadataSourceIfNotExists(
            enable_upgrade_migration);
      },
      options));
  options.set_tag("InitMetadataStoreIfNotExists_UpsertSimpleTypes");
  return transaction_executor_->Execute(
      [this]() -> absl::Status {
        return UpsertSimpleTypes(metadata_access_object_.get());
      },
      options);
}



absl::Status MetadataStore::PutTypes(const PutTypesRequest& request,
                                     PutTypesResponse* response) {
  if (!request.all_fields_match()) {
    return absl::UnimplementedError("Must match all fields.");
  }
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        return UpsertTypes(request.artifact_types(), request.execution_types(),
                           request.context_types(), request.can_add_fields(),
                           request.can_omit_fields(),
                           metadata_access_object_.get(), response);
      },
      request.transaction_options());
}

absl::Status MetadataStore::PutArtifactType(
    const PutArtifactTypeRequest& request, PutArtifactTypeResponse* response) {
  if (!request.all_fields_match()) {
    return absl::UnimplementedError("Must match all fields.");
  }
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<ArtifactType> types = {request.artifact_type()};
        std::vector<int64_t> type_ids;
        MLMD_RETURN_IF_ERROR(UpsertTypes<ArtifactType>(
            {types.begin(), types.end()}, request.can_add_fields(),
            request.can_omit_fields(), metadata_access_object_.get(),
            type_ids));
        response->set_type_id(type_ids.front());
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::PutExecutionType(
    const PutExecutionTypeRequest& request,
    PutExecutionTypeResponse* response) {
  if (!request.all_fields_match()) {
    return absl::UnimplementedError("Must match all fields.");
  }
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<ExecutionType> types = {request.execution_type()};
        std::vector<int64_t> type_ids;
        MLMD_RETURN_IF_ERROR(UpsertTypes<ExecutionType>(
            {types.begin(), types.end()}, request.can_add_fields(),
            request.can_omit_fields(), metadata_access_object_.get(),
            type_ids));
        response->set_type_id(type_ids.front());
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::PutContextType(const PutContextTypeRequest& request,
                                           PutContextTypeResponse* response) {
  if (!request.all_fields_match()) {
    return absl::UnimplementedError("Must match all fields.");
  }
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<ContextType> types = {request.context_type()};
        std::vector<int64_t> type_ids;
        MLMD_RETURN_IF_ERROR(UpsertTypes<ContextType>(
            {types.begin(), types.end()}, request.can_add_fields(),
            request.can_omit_fields(), metadata_access_object_.get(),
            type_ids));
        response->set_type_id(type_ids.front());
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetArtifactType(
    const GetArtifactTypeRequest& request, GetArtifactTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        ArtifactType type;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindTypeByNameAndVersion(
            request.type_name(), GetRequestTypeVersion(request), &type));
        std::vector<ArtifactType*> types({&type});
        MLMD_RETURN_IF_ERROR(
            SetBaseType<ArtifactType, ArtifactType::SystemDefinedBaseType>(
                absl::MakeSpan(types), metadata_access_object_.get()));
        *response->mutable_artifact_type() = type;
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetExecutionType(
    const GetExecutionTypeRequest& request,
    GetExecutionTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        ExecutionType type;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindTypeByNameAndVersion(
            request.type_name(), GetRequestTypeVersion(request), &type));
        std::vector<ExecutionType*> types({&type});
        MLMD_RETURN_IF_ERROR(
            SetBaseType<ExecutionType, ExecutionType::SystemDefinedBaseType>(
                absl::MakeSpan(types), metadata_access_object_.get()));
        *response->mutable_execution_type() = type;
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetContextType(const GetContextTypeRequest& request,
                                           GetContextTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        return metadata_access_object_->FindTypeByNameAndVersion(
            request.type_name(), GetRequestTypeVersion(request),
            response->mutable_context_type());
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetArtifactTypesByID(
    const GetArtifactTypesByIDRequest& request,
    GetArtifactTypesByIDResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const int64_t type_id : request.type_ids()) {
          ArtifactType artifact_type;
          // TODO(b/218884256): replace FindTypeById with FindTypesById.
          const absl::Status status =
              metadata_access_object_->FindTypeById(type_id, &artifact_type);
          if (status.ok()) {
            *response->mutable_artifact_types()->Add() = artifact_type;
          } else if (!absl::IsNotFound(status)) {
            return status;
          }
        }
        MLMD_RETURN_IF_ERROR(
            SetBaseType<ArtifactType, ArtifactType::SystemDefinedBaseType>(
                absl::MakeSpan(const_cast<ArtifactType* const*>(
                                   response->mutable_artifact_types()->data()),
                               response->artifact_types_size()),
                metadata_access_object_.get()));
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetExecutionTypesByID(
    const GetExecutionTypesByIDRequest& request,
    GetExecutionTypesByIDResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const int64_t type_id : request.type_ids()) {
          ExecutionType execution_type;
          const absl::Status status =
              metadata_access_object_->FindTypeById(type_id, &execution_type);
          if (status.ok()) {
            *response->mutable_execution_types()->Add() = execution_type;
          } else if (!absl::IsNotFound(status)) {
            return status;
          }
        }
        MLMD_RETURN_IF_ERROR(
            SetBaseType<ExecutionType, ExecutionType::SystemDefinedBaseType>(
                absl::MakeSpan(const_cast<ExecutionType* const*>(
                                   response->mutable_execution_types()->data()),
                               response->execution_types_size()),
                metadata_access_object_.get()));
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetContextTypesByID(
    const GetContextTypesByIDRequest& request,
    GetContextTypesByIDResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const int64_t type_id : request.type_ids()) {
          ContextType context_type;
          const absl::Status status =
              metadata_access_object_->FindTypeById(type_id, &context_type);
          if (status.ok()) {
            *response->mutable_context_types()->Add() = context_type;
          } else if (!absl::IsNotFound(status)) {
            return status;
          }
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetArtifactTypesByExternalIds(
    const GetArtifactTypesByExternalIdsRequest& request,
    GetArtifactTypesByExternalIdsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<absl::string_view> external_ids;
        std::transform(request.external_ids().begin(),
                       request.external_ids().end(),
                       std::back_inserter(external_ids),
                       [](const std::string& external_id) {
                         return absl::string_view(external_id);
                       });
        std::vector<ArtifactType> artifact_types;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindTypesByExternalIds(
            absl::MakeSpan(external_ids), artifact_types));
        for (const ArtifactType& artifact_type : artifact_types) {
          *response->mutable_artifact_types()->Add() = artifact_type;
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetExecutionTypesByExternalIds(
    const GetExecutionTypesByExternalIdsRequest& request,
    GetExecutionTypesByExternalIdsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<absl::string_view> external_ids;
        std::transform(request.external_ids().begin(),
                       request.external_ids().end(),
                       std::back_inserter(external_ids),
                       [](const std::string& external_id) {
                         return absl::string_view(external_id);
                       });
        std::vector<ExecutionType> execution_types;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindTypesByExternalIds(
            absl::MakeSpan(external_ids), execution_types));
        for (const ExecutionType& execution_type : execution_types) {
          *response->mutable_execution_types()->Add() = execution_type;
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetContextTypesByExternalIds(
    const GetContextTypesByExternalIdsRequest& request,
    GetContextTypesByExternalIdsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<absl::string_view> external_ids;
        std::transform(request.external_ids().begin(),
                       request.external_ids().end(),
                       std::back_inserter(external_ids),
                       [](const std::string& external_id) {
                         return absl::string_view(external_id);
                       });
        std::vector<ContextType> context_types;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindTypesByExternalIds(
            absl::MakeSpan(external_ids), context_types));
        for (const ContextType& context_type : context_types) {
          *response->mutable_context_types()->Add() = context_type;
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetArtifactsByID(
    const GetArtifactsByIDRequest& request,
    GetArtifactsByIDResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Artifact> artifacts;
        std::vector<ArtifactType> artifact_types;
        const std::vector<int64_t> ids(request.artifact_ids().begin(),
                                       request.artifact_ids().end());
        const absl::Status status =
            request.populate_artifact_types()
                ? metadata_access_object_->FindArtifactsById(ids, artifacts,
                                                             artifact_types)
                : metadata_access_object_->FindArtifactsById(ids, &artifacts);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        absl::c_copy(artifacts, google::protobuf::RepeatedPtrFieldBackInserter(
                                    response->mutable_artifacts()));
        if (request.populate_artifact_types()) {
          absl::c_copy(artifact_types, google::protobuf::RepeatedFieldBackInserter(
                                           response->mutable_artifact_types()));
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetExecutionsByID(
    const GetExecutionsByIDRequest& request,
    GetExecutionsByIDResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Execution> executions;
        const std::vector<int64_t> ids(request.execution_ids().begin(),
                                       request.execution_ids().end());
        const absl::Status status =
            metadata_access_object_->FindExecutionsById(ids, &executions);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        absl::c_copy(executions, google::protobuf::RepeatedPtrFieldBackInserter(
                                     response->mutable_executions()));
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetContextsByID(
    const GetContextsByIDRequest& request, GetContextsByIDResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> contexts;
        const std::vector<int64_t> ids(request.context_ids().begin(),
                                       request.context_ids().end());
        const absl::Status status =
            metadata_access_object_->FindContextsById(ids, &contexts);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        absl::c_copy(contexts, google::protobuf::RepeatedFieldBackInserter(
                                   response->mutable_contexts()));
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::PutArtifacts(const PutArtifactsRequest& request,
                                         PutArtifactsResponse* response) {
  return transaction_executor_->Execute([this, &request,
                                         &response]() -> absl::Status {
    response->Clear();
    for (const Artifact& artifact : request.artifacts()) {
      int64_t artifact_id = -1;
      // Verify the latest_updated_time before upserting the artifact.
      if (artifact.has_id() &&
          request.options().abort_if_latest_updated_time_changed()) {
        Artifact existing_artifact;
        absl::Status status;
        {
          std::vector<Artifact> artifacts;
          status = metadata_access_object_->FindArtifactsById({artifact.id()},
                                                              &artifacts);
          if (status.ok()) {
            existing_artifact = artifacts.at(0);
          }
        }
        if (!absl::IsNotFound(status)) {
          MLMD_RETURN_IF_ERROR(status);
          if (artifact.last_update_time_since_epoch() !=
              existing_artifact.last_update_time_since_epoch()) {
            return absl::FailedPreconditionError(absl::StrCat(
                "`abort_if_latest_updated_time_changed` is set, and the stored "
                "artifact with id = ",
                artifact.id(),
                " has a different last_update_time_since_epoch: ",
                existing_artifact.last_update_time_since_epoch(),
                " from the one in the given artifact: ",
                artifact.last_update_time_since_epoch()));
          }
          // If set the option and all check succeeds, we make sure the
          // timestamp after the update increases.
          absl::SleepFor(absl::Milliseconds(1));
        }
      }
      MLMD_RETURN_IF_ERROR(UpsertArtifact(
          artifact, metadata_access_object_.get(),
          /*skip_type_and_property_validation=*/false, request.update_mask(),
          /*reuse_artifact_if_already_exist_by_external_id=*/false,
          &artifact_id));
      response->add_artifact_ids(artifact_id);
    }
    return absl::OkStatus();
  },
  request.transaction_options());
}

absl::Status MetadataStore::PutExecutions(const PutExecutionsRequest& request,
                                          PutExecutionsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const Execution& execution : request.executions()) {
          int64_t execution_id = -1;
          MLMD_RETURN_IF_ERROR(
              UpsertExecution(execution, metadata_access_object_.get(),
                              /*skip_type_and_property_validation=*/false,
                              /*force_update_time=*/false,
                              request.update_mask(), &execution_id));
          response->add_execution_ids(execution_id);
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::PutContexts(const PutContextsRequest& request,
                                        PutContextsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const Context& context : request.contexts()) {
          int64_t context_id = -1;
          MLMD_RETURN_IF_ERROR(
              UpsertContext(context, metadata_access_object_.get(),
                            /*skip_type_and_property_validation=*/false,
                            request.update_mask(), &context_id));
          response->add_context_ids(context_id);
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::Create(
    const MetadataSourceQueryConfig& query_config,
    const MigrationOptions& migration_options,
    unique_ptr<MetadataSource> metadata_source,
    unique_ptr<TransactionExecutor> transaction_executor,
    unique_ptr<MetadataStore>* result) {
  unique_ptr<MetadataAccessObject> metadata_access_object;
  MLMD_RETURN_IF_ERROR(CreateMetadataAccessObject(
      query_config, metadata_source.get(), &metadata_access_object));
  // if downgrade migration is specified
  if (migration_options.downgrade_to_schema_version() >= 0) {
    MLMD_RETURN_IF_ERROR(transaction_executor->Execute(
        [&migration_options, &metadata_access_object]() -> absl::Status {
          return metadata_access_object->DowngradeMetadataSource(
              migration_options.downgrade_to_schema_version());
        }));
    return absl::CancelledError(absl::StrCat(
        "Downgrade migration was performed. Connection to the downgraded "
        "database is Cancelled. Now the database is at schema version ",
        migration_options.downgrade_to_schema_version(),
        ". Please refer to the migration guide and use lower version of the "
        "library to connect to the metadata store."));
  }
  *result = absl::WrapUnique(new MetadataStore(
      std::move(metadata_source), std::move(metadata_access_object),
      std::move(transaction_executor)));
  return absl::OkStatus();
}

absl::Status MetadataStore::PutEvents(const PutEventsRequest& request,
                                      PutEventsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const Event& event : request.events()) {
          int64_t dummy_event_id = -1;
          MLMD_RETURN_IF_ERROR(
              metadata_access_object_->CreateEvent(event, &dummy_event_id));
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::PutExecution(const PutExecutionRequest& request,
                                         PutExecutionResponse* response) {
  return transaction_executor_->Execute([this, &request,
                                         &response]() -> absl::Status {
    response->Clear();
    if (!request.has_execution()) {
      return absl::InvalidArgumentError(
          absl::StrCat("No execution is found: ", request.DebugString()));
    }

    std::vector<PutExecutionRequest::ArtifactAndEvent> artifact_event_pairs(
        request.artifact_event_pairs().begin(),
        request.artifact_event_pairs().end());

    // 1. Upsert Artifacts.
    for (PutExecutionRequest::ArtifactAndEvent& artifact_and_event :
         artifact_event_pairs) {
      if (!artifact_and_event.has_artifact()) continue;

      int64_t artifact_id = -1;
      MLMD_RETURN_IF_ERROR(UpsertArtifact(
          artifact_and_event.artifact(), metadata_access_object_.get(),
          /*skip_type_and_property_validation=*/false,
          google::protobuf::FieldMask(),
          request.options().reuse_artifact_if_already_exist_by_external_id(),
          &artifact_id));
      artifact_and_event.mutable_artifact()->set_id(artifact_id);
    }

    // 2. Upsert Execution.
    int64_t execution_id = -1;
    MLMD_RETURN_IF_ERROR(
        UpsertExecution(request.execution(), metadata_access_object_.get(),
                        /*skip_type_and_property_validation=*/false,
                        request.options().force_update_time(),
                        google::protobuf::FieldMask(), &execution_id));
    response->set_execution_id(execution_id);

    // 3. Insert events.
    for (const PutExecutionRequest::ArtifactAndEvent& artifact_and_event :
         artifact_event_pairs) {
      MLMD_RETURN_IF_ERROR(InsertEvent(artifact_and_event, execution_id,
                                       metadata_access_object_.get()));

      if (artifact_and_event.has_artifact()) {
        response->add_artifact_ids(artifact_and_event.artifact().id());
      } else if (artifact_and_event.has_event()) {
        response->add_artifact_ids(artifact_and_event.event().artifact_id());
      } else {
        // It is valid to have empty artifact and event pair, i.e. both artifact
        // and event are missing. In such a case, we return -1.
        response->add_artifact_ids(-1);
      }
    }

    // 4. Upsert contexts and insert associations and attributions.
    absl::flat_hash_set<int64_t> artifact_ids(response->artifact_ids().begin(),
                                              response->artifact_ids().end());
    for (const Context& context : request.contexts()) {
      int64_t context_id = -1;

      if (context.has_id() && request.options().force_reuse_context()) {
        context_id = context.id();
        std::vector<Context> contexts;
        const absl::Status status =
            metadata_access_object_->FindContextsById({context_id}, &contexts);
        MLMD_RETURN_IF_ERROR(status);
        if ((contexts.size() != 1) || (contexts[0].id() != context_id)) {
          return absl::NotFoundError(absl::StrCat(
              "Context with ID ", context_id, " was not found in MLMD"));
        }
      } else {
        const absl::Status status = UpsertContextWithOptions(
            context, metadata_access_object_.get(),
            request.options().reuse_context_if_already_exist(),
            /*skip_type_and_property_validation=*/false, &context_id);
        MLMD_RETURN_IF_ERROR(status);
      }

      response->add_context_ids(context_id);
      MLMD_RETURN_IF_ERROR(InsertAssociationIfNotExist(
          context_id, response->execution_id(), /*is_already_validated=*/true,
          metadata_access_object_.get()));
      for (const int64_t artifact_id : artifact_ids) {
        MLMD_RETURN_IF_ERROR(InsertAttributionIfNotExist(
            context_id, artifact_id, /*is_already_validated=*/true,
            metadata_access_object_.get()));
      }
    }
    return absl::OkStatus();
  },
  request.transaction_options());
}

// TODO(b/217390865): Optimize upsert calls once b/197686185 is complete.
absl::Status MetadataStore::PutLineageSubgraph(
    const PutLineageSubgraphRequest& request,
    PutLineageSubgraphResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();

        MLMD_RETURN_IF_ERROR(CheckEventEdges(request));
        MLMD_RETURN_IF_ERROR(
            BatchTypeAndPropertyValidation<ArtifactType, Artifact>(
                request.artifacts(), metadata_access_object_.get()));
        MLMD_RETURN_IF_ERROR(
            BatchTypeAndPropertyValidation<ExecutionType, Execution>(
                request.executions(), metadata_access_object_.get()));
        MLMD_RETURN_IF_ERROR(
            BatchTypeAndPropertyValidation<ContextType, Context>(
                request.contexts(), metadata_access_object_.get()));

        // 1. Upsert contexts.
        for (const Context& context : request.contexts()) {
          int64_t context_id = -1;
          absl::Status status = UpsertContextWithOptions(
              context, metadata_access_object_.get(),
              request.options().reuse_context_if_already_exist(),
              /*skip_type_and_property_validation=*/true, &context_id);
          MLMD_RETURN_IF_ERROR(status);
          response->add_context_ids(context_id);
        }

        // 2. Upsert artifacts.
        // Select the list of external_ids from Artifacts.
        // Search within the db to create a mapping from external_id to id.
        absl::flat_hash_map<std::string, int64_t> external_id_to_id_map;
        if (request.options()
                .reuse_artifact_if_already_exist_by_external_id()) {
          std::vector<Artifact> artifacts_to_build_mapping(
              request.artifacts().begin(), request.artifacts().end());
          MLMD_RETURN_IF_ERROR(GetExternalIdToIdMapping(
              absl::MakeSpan(artifacts_to_build_mapping),
              metadata_access_object_.get(), external_id_to_id_map));
        }
        for (const Artifact& artifact : request.artifacts()) {
          Artifact artifact_copy;
          artifact_copy.CopyFrom(artifact);
          if (request.options()
                  .reuse_artifact_if_already_exist_by_external_id() &&
              artifact.has_external_id() && !artifact.external_id().empty() &&
              !artifact.has_id() &&
              external_id_to_id_map.contains(artifact.external_id())) {
            artifact_copy.set_id(
                external_id_to_id_map.find(artifact.external_id())->second);
          }
          int64_t artifact_id = -1;
          MLMD_RETURN_IF_ERROR(UpsertArtifact(
              artifact_copy, metadata_access_object_.get(),
              /*skip_type_and_property_validation=*/true,
              google::protobuf::FieldMask(),
              /*reuse_artifact_if_already_exist_by_external_id=*/false,
              &artifact_id));
          response->add_artifact_ids(artifact_id);
        }

        // 3. Upsert executions.
        for (const Execution& execution : request.executions()) {
          int64_t execution_id = -1;
          MLMD_RETURN_IF_ERROR(
              UpsertExecution(execution, metadata_access_object_.get(),
                              /*skip_type_and_property_validation=*/true,
                              /*force_update_time=*/false,
                              google::protobuf::FieldMask(), &execution_id));
          response->add_execution_ids(execution_id);
        }

        // 4. Create associations and attributions.
        absl::flat_hash_set<int64_t> artifact_ids(
            response->artifact_ids().begin(), response->artifact_ids().end());
        absl::flat_hash_set<int64_t> context_ids(
            response->context_ids().begin(), response->context_ids().end());
        absl::flat_hash_set<int64_t> execution_ids(
            response->execution_ids().begin(), response->execution_ids().end());

        for (const int64_t context_id : context_ids) {
          for (const int64_t execution_id : execution_ids) {
            MLMD_RETURN_IF_ERROR(InsertAssociationIfNotExist(
                context_id, execution_id, /*is_already_validated=*/true,
                metadata_access_object_.get()));
          }

          for (const int64_t artifact_id : artifact_ids) {
            MLMD_RETURN_IF_ERROR(InsertAttributionIfNotExist(
                context_id, artifact_id, /*is_already_validated=*/true,
                metadata_access_object_.get()));
          }
        }

        // 5. Add events with the upserted executions and artifacts.
        for (const PutLineageSubgraphRequest::EventEdge& event_edge :
             request.event_edges()) {
          Event event = event_edge.event();

          // Provided execution ID and artifact ID are verified to be valid in
          // CheckEventEdges(). If not provided, use the ID of the created
          // execution or artifact.
          if (!event.has_execution_id()) {
            event.set_execution_id(
                response->execution_ids(event_edge.execution_index()));
          }
          if (!event.has_artifact_id()) {
            event.set_artifact_id(
                response->artifact_ids(event_edge.artifact_index()));
          }

          int64_t dummy_event_id = -1;
          MLMD_RETURN_IF_ERROR(metadata_access_object_->CreateEvent(
              event, /*is_already_validated=*/true, &dummy_event_id));
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}


absl::Status MetadataStore::GetEventsByExecutionIDs(
    const GetEventsByExecutionIDsRequest& request,
    GetEventsByExecutionIDsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Event> events;
        const absl::Status status =
            metadata_access_object_->FindEventsByExecutions(
                std::vector<int64_t>(request.execution_ids().begin(),
                                     request.execution_ids().end()),
                &events);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const Event& event : events) {
          *response->mutable_events()->Add() = event;
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetEventsByArtifactIDs(
    const GetEventsByArtifactIDsRequest& request,
    GetEventsByArtifactIDsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Event> events;
        const absl::Status status =
            metadata_access_object_->FindEventsByArtifacts(
                std::vector<int64_t>(request.artifact_ids().begin(),
                                     request.artifact_ids().end()),
                &events);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const Event& event : events) {
          *response->mutable_events()->Add() = event;
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetExecutions(const GetExecutionsRequest& request,
                                          GetExecutionsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Execution> executions;
        absl::Status status;
        std::string next_page_token;
        if (request.has_options()) {
          status = metadata_access_object_->ListExecutions(
              request.options(), &executions, &next_page_token);
        } else {
          status = metadata_access_object_->FindExecutions(&executions);
        }

        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }

        for (const Execution& execution : executions) {
          *response->mutable_executions()->Add() = execution;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetArtifacts(const GetArtifactsRequest& request,
                                         GetArtifactsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Artifact> artifacts;
        absl::Status status;
        std::string next_page_token;
        if (request.has_options()) {
          status = metadata_access_object_->ListArtifacts(
              request.options(), &artifacts, &next_page_token);
        } else {
          status = metadata_access_object_->FindArtifacts(&artifacts);
        }

        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }

        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }

        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetContexts(const GetContextsRequest& request,
                                        GetContextsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> contexts;
        absl::Status status;
        std::string next_page_token;
        if (request.has_options()) {
          status = metadata_access_object_->ListContexts(
              request.options(), &contexts, &next_page_token);
        } else {
          status = metadata_access_object_->FindContexts(&contexts);
        }

        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }

        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }

        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetArtifactTypes(
    const GetArtifactTypesRequest& request,
    GetArtifactTypesResponse* response) {
  return transaction_executor_->Execute(
      [this, &response]() -> absl::Status {
        response->Clear();
        std::vector<ArtifactType> artifact_types;
        const absl::Status status =
            metadata_access_object_->FindTypes(&artifact_types);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        absl::c_copy_if(
            artifact_types,
            google::protobuf::RepeatedFieldBackInserter(
                response->mutable_artifact_types()),
            [](const ArtifactType& type) {
              // Simple types will not be returned by Get*Types APIs
              // because they are invisible to users.
              return std::find(kSimpleTypeNames.begin(), kSimpleTypeNames.end(),
                               type.name()) == kSimpleTypeNames.end();
            });
        MLMD_RETURN_IF_ERROR(
            SetBaseType<ArtifactType, ArtifactType::SystemDefinedBaseType>(
                absl::MakeSpan(const_cast<ArtifactType* const*>(
                                   response->mutable_artifact_types()->data()),
                               response->artifact_types_size()),
                metadata_access_object_.get()));
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetExecutionTypes(
    const GetExecutionTypesRequest& request,
    GetExecutionTypesResponse* response) {
  return transaction_executor_->Execute(
      [this, &response]() -> absl::Status {
        response->Clear();
        std::vector<ExecutionType> execution_types;
        const absl::Status status =
            metadata_access_object_->FindTypes(&execution_types);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        absl::c_copy_if(
            execution_types,
            google::protobuf::RepeatedFieldBackInserter(
                response->mutable_execution_types()),
            [](const ExecutionType& type) {
              // Simple types will not be returned by Get*Types APIs
              // because they are invisible to users.
              return std::find(kSimpleTypeNames.begin(), kSimpleTypeNames.end(),
                               type.name()) == kSimpleTypeNames.end();
            });
        MLMD_RETURN_IF_ERROR(
            SetBaseType<ExecutionType, ExecutionType::SystemDefinedBaseType>(
                absl::MakeSpan(const_cast<ExecutionType* const*>(
                                   response->mutable_execution_types()->data()),
                               response->execution_types_size()),
                metadata_access_object_.get()));
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetContextTypes(
    const GetContextTypesRequest& request, GetContextTypesResponse* response) {
  return transaction_executor_->Execute(
      [this, &response]() -> absl::Status {
        response->Clear();
        std::vector<ContextType> context_types;
        const absl::Status status =
            metadata_access_object_->FindTypes(&context_types);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const ContextType& context_type : context_types) {
          *response->mutable_context_types()->Add() = context_type;
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetArtifactsByURI(
    const GetArtifactsByURIRequest& request,
    GetArtifactsByURIResponse* response) {
  // Validate if there's already deprecated optional string uri = 1 field.
  const google::protobuf::UnknownFieldSet& unknown_field_set =
      request.GetReflection()->GetUnknownFields(request);
  for (int i = 0; i < unknown_field_set.field_count(); i++) {
    const google::protobuf::UnknownField& unknown_field = unknown_field_set.field(i);
    if (unknown_field.number() == 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The request contains deprecated field `uri`. Please upgrade the "
          "client library version above 0.21.0. GetArtifactsByURIRequest: ",
          request.DebugString()));
    }
  }
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        absl::flat_hash_set<std::string> uris(request.uris().begin(),
                                              request.uris().end());
        for (const std::string& uri : uris) {
          std::vector<Artifact> artifacts;
          const absl::Status status =
              metadata_access_object_->FindArtifactsByURI(uri, &artifacts);
          if (!status.ok() && !absl::IsNotFound(status)) {
            // If any none NotFound error returned, we do early stopping as
            // the query execution has internal db errors.
            return status;
          }
          for (const Artifact& artifact : artifacts) {
            *response->mutable_artifacts()->Add() = artifact;
          }
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetArtifactsByType(
    const GetArtifactsByTypeRequest& request,
    GetArtifactsByTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        int64_t artifact_type_id;
        absl::Status status =
            metadata_access_object_->FindTypeIdByNameAndVersion(
                request.type_name(), GetRequestTypeVersion(request),
                TypeKind::ARTIFACT_TYPE, &artifact_type_id);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        std::vector<Artifact> artifacts;
        std::string next_page_token;
        status = metadata_access_object_->FindArtifactsByTypeId(
            artifact_type_id,
            (request.has_options() ? absl::make_optional(request.options())
                                   : absl::nullopt),
            &artifacts, &next_page_token);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }
        if (request.has_options()) {
          response->set_next_page_token(next_page_token);
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetArtifactByTypeAndName(
    const GetArtifactByTypeAndNameRequest& request,
    GetArtifactByTypeAndNameResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        int64_t artifact_type_id;
        absl::Status status =
            metadata_access_object_->FindTypeIdByNameAndVersion(
                request.type_name(), GetRequestTypeVersion(request),
                TypeKind::ARTIFACT_TYPE, &artifact_type_id);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        Artifact artifact;
        status = metadata_access_object_->FindArtifactByTypeIdAndArtifactName(
            artifact_type_id, request.artifact_name(), &artifact);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        *response->mutable_artifact() = artifact;
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetArtifactsByExternalIds(
    const GetArtifactsByExternalIdsRequest& request,
    GetArtifactsByExternalIdsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<absl::string_view> external_ids;
        std::transform(request.external_ids().begin(),
                       request.external_ids().end(),
                       std::back_inserter(external_ids),
                       [](const std::string& external_id) {
                         return absl::string_view(external_id);
                       });
        std::vector<Artifact> artifacts;
        MLMD_RETURN_IF_ERROR(
            metadata_access_object_->FindArtifactsByExternalIds(
                absl::MakeSpan(external_ids), &artifacts));
        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetExecutionsByType(
    const GetExecutionsByTypeRequest& request,
    GetExecutionsByTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        int64_t execution_type_id;
        absl::Status status =
            metadata_access_object_->FindTypeIdByNameAndVersion(
                request.type_name(), GetRequestTypeVersion(request),
                TypeKind::EXECUTION_TYPE, &execution_type_id);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        std::vector<Execution> executions;
        std::string next_page_token;
        status = metadata_access_object_->FindExecutionsByTypeId(
            execution_type_id,
            (request.has_options() ? absl::make_optional(request.options())
                                   : absl::nullopt),
            &executions, &next_page_token);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        for (const Execution& execution : executions) {
          *response->mutable_executions()->Add() = execution;
        }
        if (request.has_options()) {
          response->set_next_page_token(next_page_token);
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetExecutionByTypeAndName(
    const GetExecutionByTypeAndNameRequest& request,
    GetExecutionByTypeAndNameResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        int64_t execution_type_id;
        absl::Status status =
            metadata_access_object_->FindTypeIdByNameAndVersion(
                request.type_name(), GetRequestTypeVersion(request),
                TypeKind::EXECUTION_TYPE, &execution_type_id);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        Execution execution;
        status = metadata_access_object_->FindExecutionByTypeIdAndExecutionName(
            execution_type_id, request.execution_name(), &execution);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        *response->mutable_execution() = execution;
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetExecutionsByExternalIds(
    const GetExecutionsByExternalIdsRequest& request,
    GetExecutionsByExternalIdsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<absl::string_view> external_ids;
        std::transform(request.external_ids().begin(),
                       request.external_ids().end(),
                       std::back_inserter(external_ids),
                       [](const std::string& external_id) {
                         return absl::string_view(external_id);
                       });
        std::vector<Execution> executions;
        MLMD_RETURN_IF_ERROR(
            metadata_access_object_->FindExecutionsByExternalIds(
                absl::MakeSpan(external_ids), &executions));
        for (const Execution& execution : executions) {
          *response->mutable_executions()->Add() = execution;
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetContextsByType(
    const GetContextsByTypeRequest& request,
    GetContextsByTypeResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        int64_t context_type_id;
        {
          absl::Status status =
              metadata_access_object_->FindTypeIdByNameAndVersion(
                  request.type_name(), GetRequestTypeVersion(request),
                  TypeKind::CONTEXT_TYPE, &context_type_id);
          if (absl::IsNotFound(status)) {
            return absl::OkStatus();
          } else if (!status.ok()) {
            return status;
          }
        }
        std::vector<Context> contexts;
        std::string next_page_token;
        {
          absl::Status status;
          status = metadata_access_object_->FindContextsByTypeId(
              context_type_id,
              (request.has_options() ? absl::make_optional(request.options())
                                     : absl::nullopt),
              &contexts, &next_page_token);
          if (absl::IsNotFound(status)) {
            return absl::OkStatus();
          } else if (!status.ok()) {
            return status;
          }
        }
        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }
        if (request.has_options()) {
          response->set_next_page_token(next_page_token);
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetContextByTypeAndName(
    const GetContextByTypeAndNameRequest& request,
    GetContextByTypeAndNameResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        int64_t context_type_id;
        absl::Status status =
            metadata_access_object_->FindTypeIdByNameAndVersion(
                request.type_name(), GetRequestTypeVersion(request),
                TypeKind::CONTEXT_TYPE, &context_type_id);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        Context context;
        status = metadata_access_object_->FindContextByTypeIdAndContextName(
            context_type_id, request.context_name(), /*id_only=*/false,
            &context);
        if (absl::IsNotFound(status)) {
          return absl::OkStatus();
        } else if (!status.ok()) {
          return status;
        }
        *response->mutable_context() = context;
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetContextsByExternalIds(
    const GetContextsByExternalIdsRequest& request,
    GetContextsByExternalIdsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<absl::string_view> external_ids;
        std::transform(request.external_ids().begin(),
                       request.external_ids().end(),
                       std::back_inserter(external_ids),
                       [](const std::string& external_id) {
                         return absl::string_view(external_id);
                       });
        std::vector<Context> contexts;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindContextsByExternalIds(
            absl::MakeSpan(external_ids), &contexts));
        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::PutAttributionsAndAssociations(
    const PutAttributionsAndAssociationsRequest& request,
    PutAttributionsAndAssociationsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const Attribution& attribution : request.attributions()) {
          MLMD_RETURN_IF_ERROR(InsertAttributionIfNotExist(
              attribution.context_id(), attribution.artifact_id(),
              /*is_already_validated=*/false, metadata_access_object_.get()));
        }
        for (const Association& association : request.associations()) {
          MLMD_RETURN_IF_ERROR(InsertAssociationIfNotExist(
              association.context_id(), association.execution_id(),
              /*is_already_validated=*/false, metadata_access_object_.get()));
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::PutParentContexts(
    const PutParentContextsRequest& request,
    PutParentContextsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        for (const ParentContext& parent_context : request.parent_contexts()) {
          MLMD_RETURN_IF_ERROR(
              metadata_access_object_->CreateParentContext(parent_context));
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetContextsByArtifact(
    const GetContextsByArtifactRequest& request,
    GetContextsByArtifactResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> contexts;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindContextsByArtifact(
            request.artifact_id(), &contexts));
        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetContextsByExecution(
    const GetContextsByExecutionRequest& request,
    GetContextsByExecutionResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> contexts;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindContextsByExecution(
            request.execution_id(), &contexts));
        for (const Context& context : contexts) {
          *response->mutable_contexts()->Add() = context;
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetArtifactsByContext(
    const GetArtifactsByContextRequest& request,
    GetArtifactsByContextResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Artifact> artifacts;
        std::string next_page_token;
        auto list_options = request.has_options()
                                ? absl::make_optional(request.options())
                                : absl::nullopt;
        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindArtifactsByContext(
            request.context_id(), list_options, &artifacts, &next_page_token));

        for (const Artifact& artifact : artifacts) {
          *response->mutable_artifacts()->Add() = artifact;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }

        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetExecutionsByContext(
    const GetExecutionsByContextRequest& request,
    GetExecutionsByContextResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Execution> executions;
        std::string next_page_token;
        auto list_options = request.has_options()
                                ? absl::make_optional(request.options())
                                : absl::nullopt;

        MLMD_RETURN_IF_ERROR(metadata_access_object_->FindExecutionsByContext(
            request.context_id(), list_options, &executions, &next_page_token));

        for (const Execution& execution : executions) {
          *response->mutable_executions()->Add() = execution;
        }

        if (!next_page_token.empty()) {
          response->set_next_page_token(next_page_token);
        }

        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetParentContextsByContext(
    const GetParentContextsByContextRequest& request,
    GetParentContextsByContextResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> parent_contexts;
        const absl::Status status =
            metadata_access_object_->FindParentContextsByContextId(
                request.context_id(), &parent_contexts);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        absl::c_copy(parent_contexts, google::protobuf::RepeatedPtrFieldBackInserter(
                                          response->mutable_contexts()));
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetChildrenContextsByContext(
    const GetChildrenContextsByContextRequest& request,
    GetChildrenContextsByContextResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<Context> child_contexts;
        const absl::Status status =
            metadata_access_object_->FindChildContextsByContextId(
                request.context_id(), &child_contexts);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        absl::c_copy(child_contexts, google::protobuf::RepeatedPtrFieldBackInserter(
                                         response->mutable_contexts()));
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetParentContextsByContexts(
    const GetParentContextsByContextsRequest& request,
    GetParentContextsByContextsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<int64_t> context_ids;
        std::copy(request.context_ids().begin(), request.context_ids().end(),
                  std::back_inserter(context_ids));
        absl::node_hash_map<int64_t, std::vector<Context>> parent_contexts;
        const absl::Status status =
            metadata_access_object_->FindParentContextsByContextIds(
                context_ids, parent_contexts);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        for (auto& entry : parent_contexts) {
          absl::c_move(entry.second,
                       google::protobuf::RepeatedPtrFieldBackInserter(
                           (*response->mutable_contexts())[entry.first]
                               .mutable_parent_contexts()));
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}

absl::Status MetadataStore::GetChildrenContextsByContexts(
    const GetChildrenContextsByContextsRequest& request,
    GetChildrenContextsByContextsResponse* response) {
  return transaction_executor_->Execute(
      [this, &request, &response]() -> absl::Status {
        response->Clear();
        std::vector<int64_t> context_ids;
        std::copy(request.context_ids().begin(), request.context_ids().end(),
                  std::back_inserter(context_ids));
        absl::node_hash_map<int64_t, std::vector<Context>> child_contexts;
        const absl::Status status =
            metadata_access_object_->FindChildContextsByContextIds(
                context_ids, child_contexts);
        if (!status.ok() && !absl::IsNotFound(status)) {
          return status;
        }
        for (auto& entry : child_contexts) {
          absl::c_move(entry.second,
                       google::protobuf::RepeatedPtrFieldBackInserter(
                           (*response->mutable_contexts())[entry.first]
                               .mutable_children_contexts()));
        }
        return absl::OkStatus();
      },
      request.transaction_options());
}


absl::Status MetadataStore::GetLineageGraph(
    const GetLineageGraphRequest& request, GetLineageGraphResponse* response) {
  return absl::UnimplementedError("GetLineageGraph is not implemented.");
}

absl::Status MetadataStore::GetLineageSubgraph(
    const GetLineageSubgraphRequest& request,
    GetLineageSubgraphResponse* response) {
  // If no mask path is provided, add all paths to retrieve all.
  google::protobuf::FieldMask read_mask;
  if (request.read_mask().paths().empty()) {
    for (int64_t index = 0; index < LineageGraph::descriptor()->field_count();
         index++) {
      read_mask.add_paths(LineageGraph::descriptor()->field(index)->name());
    }
  } else {
    read_mask = request.read_mask();
  }
  return transaction_executor_->Execute(
      [&]() -> absl::Status {
        response->Clear();
        return metadata_access_object_->QueryLineageSubgraph(
            request.lineage_subgraph_query_options(), read_mask,
            *response->mutable_lineage_subgraph());
      },
      request.transaction_options());
}



MetadataStore::MetadataStore(
    std::unique_ptr<MetadataSource> metadata_source,
    std::unique_ptr<MetadataAccessObject> metadata_access_object,
    std::unique_ptr<TransactionExecutor> transaction_executor)
    : metadata_source_(std::move(metadata_source)),
      metadata_access_object_(std::move(metadata_access_object)),
      transaction_executor_(std::move(transaction_executor)) {}

}  // namespace ml_metadata
