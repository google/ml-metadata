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
#include "ml_metadata/metadata_store/rdbms_metadata_access_object.h"


#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include "google/protobuf/field_mask.pb.h"
#include "google/protobuf/struct.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/message_differencer.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/metadata_store/list_operation_util.h"
#include "ml_metadata/metadata_store/query_executor.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/simple_types/simple_types_constants.h"
#include "ml_metadata/util/field_mask_utils.h"
#include "ml_metadata/util/record_parsing_utils.h"
#include "ml_metadata/util/return_utils.h"
#include "ml_metadata/util/struct_utils.h"
#include "google/protobuf/util/field_mask_util.h"

namespace ml_metadata {
namespace {

TypeKind ResolveTypeKind(const ArtifactType* const type) {
  return TypeKind::ARTIFACT_TYPE;
}

TypeKind ResolveTypeKind(const ExecutionType* const type) {
  return TypeKind::EXECUTION_TYPE;
}

TypeKind ResolveTypeKind(const ContextType* const type) {
  return TypeKind::CONTEXT_TYPE;
}

// Converts a record set that contains an id column at position per record to a
// vector.
std::vector<int64_t> ConvertToIds(const RecordSet& record_set,
                                  int position = 0) {
  std::vector<int64_t> result;
  result.reserve(record_set.records().size());
  for (const RecordSet::Record& record : record_set.records()) {
    int64_t id;
    CHECK(absl::SimpleAtoi(record.values(position), &id));
    result.push_back(id);
  }
  return result;
}

// Dedups an id list.
std::vector<int64_t> DedupIds(absl::Span<const int64_t> ids) {
  std::vector<int64_t> result;
  absl::flat_hash_set<int64_t> deduped_set(ids.begin(), ids.end());
  absl::c_move(deduped_set, std::back_inserter(result));
  return result;
}

// Extracts 2 vectors of ids and corresponding parent ids from parent_{}
// records.
void ConvertToIdAndParentIds(const RecordSet& record_set,
                             std::vector<int64_t>& ids,
                             std::vector<int64_t>& parent_ids) {
  const std::vector<int64_t> converted_ids = ConvertToIds(record_set);
  absl::c_copy(converted_ids, std::back_inserter(ids));
  const std::vector<int64_t> converted_parent_ids =
      ConvertToIds(record_set, /*position=*/1);
  absl::c_copy(converted_parent_ids, std::back_inserter(parent_ids));
}

// Extracts a vector of parent type ids from the parent_type triplets.
std::vector<int64_t> ParentTypesToParentTypeIds(const RecordSet& record_set) {
  return ConvertToIds(record_set, /*position=*/1);
}

// Extracts a vector of context ids from attribution triplets.
std::vector<int64_t> AttributionsToContextIds(const RecordSet& record_set) {
  return ConvertToIds(record_set, /*position=*/1);
}

// Extracts a vector of context ids from attribution triplets.
std::vector<int64_t> AttributionsToArtifactIds(const RecordSet& record_set) {
  return ConvertToIds(record_set, /*position=*/2);
}

// Extracts a vector of context ids from association triplets.
std::vector<int64_t> AssociationsToContextIds(const RecordSet& record_set) {
  return ConvertToIds(record_set, /*position=*/1);
}

// Extracts a vector of execution ids from association triplets.
std::vector<int64_t> AssociationsToExecutionIds(const RecordSet& record_set) {
  return ConvertToIds(record_set, /*position=*/2);
}

// Extracts a vector of parent context ids from parent context triplets.
// If is_parent is true, then parent_context_ids are returned.
// If is_parent is false, then context_ids for children are returned.
std::vector<int64_t> ParentContextsToContextIds(const RecordSet& record_set,
                                                bool is_parent) {
  const int position = is_parent ? 1 : 0;
  return ConvertToIds(record_set, position);
}

// Extracts a map of <key_id, vector<value_id_subset>> for each key_id in the
// key_ids list.
absl::flat_hash_map<int64_t, std::vector<int64_t>> MapKeyIdToValueIds(
    absl::Span<const int64_t> key_ids, absl::Span<const int64_t> value_ids) {
  CHECK_EQ(key_ids.size(), value_ids.size());
  absl::flat_hash_map<int64_t, std::vector<int64_t>> result;
  for (int i = 0; i < key_ids.size(); ++i) {
    if (!result.contains(key_ids[i])) {
      result.insert({key_ids[i], {value_ids[i]}});
    } else {
      result[key_ids[i]].push_back(value_ids[i]);
    }
  }
  return result;
}

// Parses and converts a string value to a specific field in a message.
// If the given string `value` is NULL (encoded as kMetadataSourceNull), then
// leave the field unset.
// The field should be a scalar field. The field type must be one of {string,
// int64_t, bool, enum, message}.
absl::Status ParseValueToField(const google::protobuf::FieldDescriptor* field_descriptor,
                               absl::string_view value,
                               google::protobuf::Message* message) {
  if (value == kMetadataSourceNull) {
    return absl::OkStatus();
  }
  const google::protobuf::Reflection* reflection = message->GetReflection();
  switch (field_descriptor->cpp_type()) {
    case google::protobuf::FieldDescriptor::CppType::CPPTYPE_STRING: {
      if (field_descriptor->is_repeated())
        reflection->AddString(message, field_descriptor, std::string(value));
      else
        reflection->SetString(message, field_descriptor, std::string(value));
      break;
    }
    case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT64: {
      int64_t int64_value;
      CHECK(absl::SimpleAtoi(value, &int64_value));
      if (field_descriptor->is_repeated())
        reflection->AddInt64(message, field_descriptor, int64_value);
      else
        reflection->SetInt64(message, field_descriptor, int64_value);
      break;
    }
    case google::protobuf::FieldDescriptor::CppType::CPPTYPE_BOOL: {
      bool bool_value;
      CHECK(absl::SimpleAtob(value, &bool_value));
      if (field_descriptor->is_repeated())
        reflection->AddBool(message, field_descriptor, bool_value);
      else
        reflection->SetBool(message, field_descriptor, bool_value);
      break;
    }
    case google::protobuf::FieldDescriptor::CppType::CPPTYPE_ENUM: {
      int enum_value;
      CHECK(absl::SimpleAtoi(value, &enum_value));
      if (field_descriptor->is_repeated())
        reflection->AddEnumValue(message, field_descriptor, enum_value);
      else
        reflection->SetEnumValue(message, field_descriptor, enum_value);
      break;
    }
    case google::protobuf::FieldDescriptor::CppType::CPPTYPE_MESSAGE: {
      CHECK(!field_descriptor->is_repeated())
          << "Cannot handle a repeated message";
      if (!value.empty()) {
        ::google::protobuf::Message* sub_message =
            reflection->MutableMessage(message, field_descriptor);
        if (!::google::protobuf::util::JsonStringToMessage(
                 std::string(value.begin(), value.size()), sub_message)
                 .ok()) {
          return absl::InternalError(
              ::absl::StrCat("Failed to parse proto: ", value));
        }
      }
      break;
    }
    default: {
      return absl::InternalError(absl::StrCat("Unsupported field type: ",
                                              field_descriptor->cpp_type()));
    }
  }
  return absl::OkStatus();
}

// Converts a RecordSet in the query result to a MessageType. In the record at
// the `record_index`, its value of each column is assigned to a message field
// with the same field name as the column name.
template <typename MessageType>
absl::Status ParseRecordSetToMessage(const RecordSet& record_set,
                                     MessageType* message,
                                     int record_index = 0) {
  CHECK_LT(record_index, record_set.records_size());
  const google::protobuf::Descriptor* descriptor = message->descriptor();
  for (int i = 0; i < record_set.column_names_size(); i++) {
    const std::string& column_name = record_set.column_names(i);
    const google::protobuf::FieldDescriptor* field_descriptor =
        descriptor->FindFieldByName(column_name);
    if (field_descriptor != nullptr) {
      const std::string& value = record_set.records(record_index).values(i);
      MLMD_RETURN_IF_ERROR(ParseValueToField(field_descriptor, value, message));
    }
  }
  return absl::OkStatus();
}


// Converts a list of Records containing key-value pairs to a proto Map.
// The field_name is the map field in the MessageType. The method fills the
// message's map field with field_name using the given records.
template <typename MessageType>
absl::Status ParseRecordsToMapField(
    const std::vector<RecordSet::Record>& records,
    const std::string& field_name, MessageType* message) {
  const google::protobuf::Descriptor* descriptor = message->descriptor();
  const google::protobuf::Reflection* reflection = message->GetReflection();
  const google::protobuf::FieldDescriptor* map_field_descriptor =
      descriptor->FindFieldByName(field_name);
  if (map_field_descriptor == nullptr || !map_field_descriptor->is_map()) {
    return absl::InternalError(
        absl::StrCat("Cannot find map field with field name: ", field_name));
  }

  const google::protobuf::FieldDescriptor* key_descriptor =
      map_field_descriptor->message_type()->FindFieldByName("key");
  const google::protobuf::FieldDescriptor* value_descriptor =
      map_field_descriptor->message_type()->FindFieldByName("value");

  for (const RecordSet::Record& record : records) {
    google::protobuf::Message* map_field_message =
        reflection->AddMessage(message, map_field_descriptor);
    const std::string& key = record.values(1);
    const std::string& value = record.values(2);
    MLMD_RETURN_IF_ERROR(
        ParseValueToField(key_descriptor, key, map_field_message));
    MLMD_RETURN_IF_ERROR(
        ParseValueToField(value_descriptor, value, map_field_message));
  }

  return absl::OkStatus();
}

// Populates 'node' properties from the rows in 'record'. The assumption is that
// properties are encoded using the convention in
// QueryExecutor::Get{X}PropertyBy{X}Id() where X in {Artifact, Execution,
// Context}.
template <typename Node>
absl::Status PopulateNodeProperties(const RecordSet::Record& record,
                                    const QueryExecutor& executor, Node& node) {
  // Populate the property of the node.
  const std::string& property_name = record.values(1);
  bool is_custom_property;
  CHECK(absl::SimpleAtob(record.values(2), &is_custom_property));
  auto& property_value =
      (is_custom_property ? (*node.mutable_custom_properties())[property_name]
                          : (*node.mutable_properties())[property_name]);
  if (record.values(3) != kMetadataSourceNull) {
    int64_t int_value;
    CHECK(absl::SimpleAtoi(record.values(3), &int_value));
    property_value.set_int_value(int_value);
  } else if (record.values(4) != kMetadataSourceNull) {
    double double_value;
    CHECK(absl::SimpleAtod(record.values(4), &double_value));
    property_value.set_double_value(double_value);
  } else if (record.values(5) != kMetadataSourceNull) {
    const std::string& string_value = record.values(5);
    if (IsStructSerializedString(string_value)) {
      MLMD_RETURN_IF_ERROR(
          StringToStruct(string_value, *property_value.mutable_struct_value()));
    } else {
      property_value.set_string_value(string_value);
    }
  } else if (record.values(6) != kMetadataSourceNull) {
    std::string to_parse;
    MLMD_RETURN_IF_ERROR(executor.DecodeBytes(record.values(6), to_parse));
    google::protobuf::Any* proto_value = property_value.mutable_proto_value();
    proto_value->ParseFromString(to_parse);
    if (proto_value->type_url().empty()) {
      return absl::InternalError(absl::StrCat(
          "Retrieved proto_value should have a nonempty type_url. Got: ",
          proto_value->DebugString()));
    }
  } else if (record.values(7) != kMetadataSourceNull) {
    bool bool_value;
    CHECK(absl::SimpleAtob(record.values(7), &bool_value));
    property_value.set_bool_value(bool_value);
  } else {
    return absl::InternalError(
        "Attempt to populate property with null value "
        "in every known *_value column.");
  }

  return absl::OkStatus();
}

// Populates type properties for types in `node_types`
// Returns detailed INTERNAl error if execution fails.
template <typename NodeType>
absl::Status PopulateTypeProperties(QueryExecutor& executor,
                                    std::vector<NodeType>& node_types) {
  if (node_types.empty()) {
    return absl::OkStatus();
  }
  RecordSet property_record_set;
  std::vector<int64_t> type_ids;
  absl::c_transform(node_types, std::back_inserter(type_ids),
                    [](const NodeType& type) { return type.id(); });
  MLMD_RETURN_IF_ERROR(
      executor.SelectPropertiesByTypeID(type_ids, &property_record_set));
  // Builds a map between type.id and all its properties.
  absl::flat_hash_map<int64_t, std::vector<RecordSet::Record>>
      type_id_to_records;
  for (const RecordSet::Record& record : property_record_set.records()) {
    int64_t type_id;
    CHECK(absl::SimpleAtoi(record.values(0), &type_id));
    if (type_id_to_records.contains(type_id)) {
      type_id_to_records[type_id].push_back(record);
    } else {
      type_id_to_records.insert({type_id, {record}});
    }
  }
  // Builds a map between type.id and its position in `types` vector.
  absl::flat_hash_map<int64_t, std::vector<int64_t>> type_id_to_positions;
  for (int i = 0; i < node_types.size(); ++i) {
    if (type_id_to_positions.contains(node_types[i].id())) {
      type_id_to_positions[node_types[i].id()].push_back(i);
    } else {
      type_id_to_positions.insert({node_types[i].id(), {i}});
    }
  }
  // Populates `properties` field.
  for (auto i = type_id_to_records.begin(); i != type_id_to_records.end();
       ++i) {
    for (int64_t position : type_id_to_positions[i->first]) {
      MLMD_RETURN_IF_ERROR(ParseRecordsToMapField(i->second, "properties",
                                                  &node_types[position]));
    }
  }
  return absl::OkStatus();
}

// Casts lower level db error messages for violating primary key constraint or
// unique constraint. Returns true when the status error message indicates such
// unique constraint is violated.
bool IsUniqueConstraintViolated(const absl::Status status) {
  return absl::IsInternal(status) &&
         (absl::StrContains(std::string(status.message()), "Duplicate") ||
          absl::StrContains(std::string(status.message()), "duplicate") ||
          absl::StrContains(std::string(status.message()), "UNIQUE"));
}

// A util to handle `version` in ArtifactType/ExecutionType/ContextType protos.
template <typename T>
std::optional<std::string> GetTypeVersion(const T& type_message) {
  return type_message.has_version() && !type_message.version().empty()
             ? absl::make_optional(type_message.version())
             : absl::nullopt;
}

// Template function for adding ancestors for ArtifactType, ExecutionType and
// ContextType cases in CheckCyClicDependency().
template <typename T>
absl::Status AddAncestors(int64_t ancestor_id,
                          std::vector<int64_t>& ancestor_ids,
                          std::unique_ptr<QueryExecutor>& executor) {
  // Checks whether type T is one of the expected node type.
  // If not, returns Internal Error.
  if (!(std::is_same<T, ArtifactType>::value ||
        std::is_same<T, ExecutionType>::value ||
        std::is_same<T, ContextType>::value)) {
    return absl::InternalError("Unexpected node type!");
  }
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor->SelectParentTypesByTypeID(
      absl::Span<const int64_t>({ancestor_id}), &record_set));
  for (const int64_t parent_id : ParentTypesToParentTypeIds(record_set)) {
    ancestor_ids.push_back(parent_id);
  }
  return absl::OkStatus();
}

// Template function for adding ancestors for ParentContext case in
// CheckCyClicDependency().
template <>
absl::Status AddAncestors<ParentContext>(
    int64_t ancestor_id, std::vector<int64_t>& ancestor_ids,
    std::unique_ptr<QueryExecutor>& executor) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor->SelectParentContextsByContextID(ancestor_id, &record_set));
  for (const int64_t parent_id :
       ParentContextsToContextIds(record_set, /*is_parent=*/true)) {
    ancestor_ids.push_back(parent_id);
  }
  return absl::OkStatus();
}

// Check whether there is a cyclic dependency. We do a DFS traversal from
// root node's id(`parent_id`) and it introduces a cycle if any ancestors' id
// is `child_id`. It assumes that existing inheritance are acyclic.
template <typename T>
absl::Status CheckCyClicDependency(int64_t child_id, int64_t parent_id,
                                   std::unique_ptr<QueryExecutor>& executor) {
  std::vector<int64_t> ancestor_ids = {parent_id};
  absl::flat_hash_set<int64_t> visited_ancestors_ids;
  while (!ancestor_ids.empty()) {
    const int64_t ancestor_id = ancestor_ids.back();
    if (ancestor_id == child_id) {
      return absl::InvalidArgumentError(
          "There is a cycle detected of the given relationship.");
    }
    ancestor_ids.pop_back();
    if (visited_ancestors_ids.contains(ancestor_id)) {
      continue;
    }
    visited_ancestors_ids.insert(ancestor_id);
    MLMD_RETURN_IF_ERROR(AddAncestors<T>(ancestor_id, ancestor_ids, executor));
  }
  return absl::OkStatus();
}

std::vector<Event> FilterEventsByDirectionAndEventType(
    absl::Span<const Event> events, const bool is_from_artifact,
    LineageSubgraphQueryOptions::Direction direction) {
  std::vector<Event> filtered_events;
  if (is_from_artifact) {
    switch (direction) {
      case LineageSubgraphQueryOptions::UPSTREAM:
        absl::c_copy_if(events, std::back_inserter(filtered_events),
                        [](const Event& event) {
                          return event.type() == Event::DECLARED_OUTPUT ||
                                 event.type() == Event::INTERNAL_OUTPUT ||
                                 event.type() == Event::PENDING_OUTPUT ||
                                 event.type() == Event::OUTPUT;
                        });
        break;
      case LineageSubgraphQueryOptions::DOWNSTREAM:
        absl::c_copy_if(events, std::back_inserter(filtered_events),
                        [](const Event& event) {
                          return event.type() == Event::DECLARED_INPUT ||
                                 event.type() == Event::INTERNAL_INPUT ||
                                 event.type() == Event::INPUT;
                        });
        break;
      default:
        absl::c_copy(events, std::back_inserter(filtered_events));
        break;
    }
  } else {
    switch (direction) {
      case LineageSubgraphQueryOptions::UPSTREAM:
        absl::c_copy_if(events, std::back_inserter(filtered_events),
                        [](const Event& event) {
                          return event.type() == Event::DECLARED_INPUT ||
                                 event.type() == Event::INTERNAL_INPUT ||
                                 event.type() == Event::INPUT;
                        });
        break;
      case LineageSubgraphQueryOptions::DOWNSTREAM:
        absl::c_copy_if(events, std::back_inserter(filtered_events),
                        [](const Event& event) {
                          return event.type() == Event::DECLARED_OUTPUT ||
                                 event.type() == Event::INTERNAL_OUTPUT ||
                                 event.type() == Event::PENDING_OUTPUT ||
                                 event.type() == Event::OUTPUT;
                        });
        break;
      default:
        absl::c_copy(events, std::back_inserter(filtered_events));
        break;
    }
  }
  return filtered_events;
}
}  // namespace


// Creates an Artifact (without properties).
absl::Status RDBMSMetadataAccessObject::CreateBasicNode(
    const Artifact& artifact, const absl::Time create_timestamp,
    int64_t* node_id) {
  return executor_->InsertArtifact(
      artifact.type_id(), artifact.uri(),
      artifact.has_state() ? absl::make_optional(artifact.state())
                           : absl::nullopt,
      artifact.has_name() ? absl::make_optional(artifact.name())
                          : absl::nullopt,
      artifact.has_external_id() ? absl::make_optional(artifact.external_id())
                                 : absl::nullopt,
      create_timestamp, create_timestamp, node_id);
}

// Creates an Execution (without properties).
absl::Status RDBMSMetadataAccessObject::CreateBasicNode(
    const Execution& execution, const absl::Time create_timestamp,
    int64_t* node_id) {
  return executor_->InsertExecution(
      execution.type_id(),
      execution.has_last_known_state()
          ? absl::make_optional(execution.last_known_state())
          : absl::nullopt,
      execution.has_name() ? absl::make_optional(execution.name())
                           : absl::nullopt,
      execution.has_external_id() ? absl::make_optional(execution.external_id())
                                  : absl::nullopt,
      create_timestamp, create_timestamp, node_id);
}

// Creates a Context (without properties).
absl::Status RDBMSMetadataAccessObject::CreateBasicNode(
    const Context& context, const absl::Time create_timestamp,
    int64_t* node_id) {
  if (!context.has_name() || context.name().empty()) {
    return absl::InvalidArgumentError("Context name should not be empty");
  }
  return executor_->InsertContext(
      context.type_id(), context.name(),
      context.has_external_id() ? absl::make_optional(context.external_id())
                                : absl::nullopt,
      create_timestamp, create_timestamp, node_id);
}

template <>
absl::Status RDBMSMetadataAccessObject::RetrieveNodesById<Context>(
    absl::Span<const int64_t> ids, RecordSet* header, RecordSet* properties) {
  MLMD_RETURN_IF_ERROR(executor_->SelectContextsByID(ids, header));
  if (!header->records().empty()) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectContextPropertyByContextID(ids, properties));
  }
  return absl::OkStatus();
}

template <>
absl::Status RDBMSMetadataAccessObject::RetrieveNodesById<Artifact>(
    absl::Span<const int64_t> ids, RecordSet* header, RecordSet* properties) {
  MLMD_RETURN_IF_ERROR(executor_->SelectArtifactsByID(ids, header));
  if (!header->records().empty()) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectArtifactPropertyByArtifactID(ids, properties));
  }
  return absl::OkStatus();
}

template <>
absl::Status RDBMSMetadataAccessObject::RetrieveNodesById<Execution>(
    absl::Span<const int64_t> ids, RecordSet* header, RecordSet* properties) {
  MLMD_RETURN_IF_ERROR(executor_->SelectExecutionsByID(ids, header));
  if (!header->records().empty()) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectExecutionPropertyByExecutionID(ids, properties));
  }
  return absl::OkStatus();
}

// Update a Node's assets based on the field mask.
// If `mask` is empty, update `stored_node` as a whole.
// If `mask` is not empty, only update fields specified in `mask`.
template <typename Node>
absl::Status RDBMSMetadataAccessObject::RunMaskedNodeUpdate(
    const Node& node, Node& stored_node, absl::Time update_timestamp,
    const google::protobuf::FieldMask& mask) {
  if (!mask.paths().empty()) {
    absl::StatusOr<google::protobuf::FieldMask> fields_mask_or =
        GetFieldsSubMaskFromMask(mask, node.GetDescriptor());
    MLMD_RETURN_IF_ERROR(fields_mask_or.status());
    // seperate fields_mask from mask
    google::protobuf::util::FieldMaskUtil::MergeOptions merge_options;
    google::protobuf::util::FieldMaskUtil::MergeMessageTo(node, fields_mask_or.value(),
                                                merge_options, &stored_node);
    MLMD_RETURN_IF_ERROR(RunNodeUpdate(stored_node, update_timestamp));
  } else {
    MLMD_RETURN_IF_ERROR(RunNodeUpdate(node, update_timestamp));
  }
  return absl::OkStatus();
}

// Update an Artifact's type_id, URI, external_id and last_update_time.
absl::Status RDBMSMetadataAccessObject::RunNodeUpdate(
    const Artifact& artifact, const absl::Time update_timestamp) {
  return executor_->UpdateArtifactDirect(
      artifact.id(), artifact.type_id(), artifact.uri(),
      artifact.has_state() ? absl::make_optional(artifact.state())
                           : absl::nullopt,
      artifact.has_external_id() ? absl::make_optional(artifact.external_id())
                                 : absl::nullopt,
      update_timestamp);
}

// Update an Execution's type_id, external_id and last_update_time.
absl::Status RDBMSMetadataAccessObject::RunNodeUpdate(
    const Execution& execution, const absl::Time update_timestamp) {
  return executor_->UpdateExecutionDirect(
      execution.id(), execution.type_id(),
      execution.has_last_known_state()
          ? absl::make_optional(execution.last_known_state())
          : absl::nullopt,
      execution.has_external_id() ? absl::make_optional(execution.external_id())
                                  : absl::nullopt,
      update_timestamp);
}

// Update a Context's type id, external_id and name.
absl::Status RDBMSMetadataAccessObject::RunNodeUpdate(
    const Context& context, const absl::Time update_timestamp) {
  if (!context.has_name() || context.name().empty()) {
    return absl::InvalidArgumentError("Context name should not be empty");
  }
  return executor_->UpdateContextDirect(
      context.id(), context.type_id(), context.name(),
      context.has_external_id() ? absl::make_optional(context.external_id())
                                : absl::nullopt,
      update_timestamp);
}

// Runs a property insertion query for a NodeType.
template <typename NodeType>
absl::Status RDBMSMetadataAccessObject::InsertProperty(
    const int64_t node_id, absl::string_view name,
    const bool is_custom_property, const Value& value) {
  NodeType node;
  const TypeKind type_kind = ResolveTypeKind(&node);
  MetadataSourceQueryConfig::TemplateQuery insert_property;
  switch (type_kind) {
    case TypeKind::ARTIFACT_TYPE:
      return executor_->InsertArtifactProperty(node_id, name,
                                               is_custom_property, value);
    case TypeKind::EXECUTION_TYPE:
      return executor_->InsertExecutionProperty(node_id, name,
                                                is_custom_property, value);

    case TypeKind::CONTEXT_TYPE:
      return executor_->InsertContextProperty(node_id, name, is_custom_property,
                                              value);
    default:
      return absl::InternalError(
          absl::StrCat("Unsupported TypeKind: ", type_kind));
  }
}

// Generates a property update query for a NodeType.
template <typename NodeType>
absl::Status RDBMSMetadataAccessObject::UpdateProperty(const int64_t node_id,
                                                       absl::string_view name,
                                                       const Value& value) {
  NodeType node;
  const TypeKind type_kind = ResolveTypeKind(&node);
  MetadataSourceQueryConfig::TemplateQuery update_property;
  switch (type_kind) {
    case TypeKind::ARTIFACT_TYPE:
      return executor_->UpdateArtifactProperty(node_id, name, value);
    case TypeKind::EXECUTION_TYPE:
      return executor_->UpdateExecutionProperty(node_id, name, value);
    case TypeKind::CONTEXT_TYPE:
      return executor_->UpdateContextProperty(node_id, name, value);
    default:
      return absl::InternalError(
          absl::StrCat("Unsupported TypeKind: ", type_kind));
  }
}

// Generates a property deletion query for a NodeType.
template <typename NodeType>
absl::Status RDBMSMetadataAccessObject::DeleteProperty(const int64_t node_id,
                                                       absl::string_view name) {
  NodeType type;
  const TypeKind type_kind = ResolveTypeKind(&type);
  switch (type_kind) {
    case TypeKind::ARTIFACT_TYPE:
      return executor_->DeleteArtifactProperty(node_id, name);
    case TypeKind::EXECUTION_TYPE:
      return executor_->DeleteExecutionProperty(node_id, name);
    case TypeKind::CONTEXT_TYPE:
      return executor_->DeleteContextProperty(node_id, name);
    default:
      return absl::InternalError("Unsupported TypeKind.");
  }
}

// Generates a list of queries for the `curr_properties` (C) based on the given
// `prev_properties` (P) only for properties associated with names in `mask`(M).
// A property definition is a 2-tuple (name, value_type).
// a) any property in the intersection of M, C and P, an update query is
// generated.
// b) any property in the intersection of M and (C \ P), insert query is
// generated.
// c) any property in the intersection of M and (P \ C), delete query is
// generated.
// The queries are composed from corresponding template
// queries with the given `NodeType` (which is one of {`ArtifactType`,
// `ExecutionType`, `ContextType`} and the `is_custom_property` (which indicates
// the space of the given properties. Returns `output_num_changed_properties`
// which equals to the number of properties are changed (deleted, updated or
// inserted).
template <typename NodeType>
absl::StatusOr<int64_t> RDBMSMetadataAccessObject::ModifyProperties(
    const google::protobuf::Map<std::string, Value>& curr_properties,
    const google::protobuf::Map<std::string, Value>& prev_properties,
    const int64_t node_id, const bool is_custom_property,
    const google::protobuf::FieldMask& mask) {
  int output_num_changed_properties = 0;
  absl::flat_hash_set<absl::string_view> property_names_in_mask;
  absl::StatusOr<absl::flat_hash_set<absl::string_view>>
      property_names_or_internal_error =
          GetPropertyNamesFromMaskOrUnionOfProperties(
              mask, is_custom_property, curr_properties, prev_properties);

  MLMD_RETURN_IF_ERROR(property_names_or_internal_error.status());
  property_names_in_mask = property_names_or_internal_error.value();

  for (absl::string_view name : property_names_in_mask) {
    const bool name_in_previous_properties =
        prev_properties.find(name.data()) != prev_properties.end();
    const bool name_in_current_properties =
        curr_properties.find(name.data()) != curr_properties.end();

    if (name_in_previous_properties) {
      if (!name_in_current_properties) {
        // Generate delete clauses for properties in P \ C
        MLMD_RETURN_IF_ERROR(DeleteProperty<NodeType>(node_id, name));
        output_num_changed_properties++;
      } else {
        // Generate update clauses for properties in the intersection P & C only
        // if the values associated with `name` are different.
        if (!google::protobuf::util::MessageDifferencer::Equals(
                prev_properties.at(name.data()),
                curr_properties.at(name.data()))) {
          if (is_custom_property) {
            MLMD_RETURN_IF_ERROR(DeleteProperty<NodeType>(node_id, name));
            MLMD_RETURN_IF_ERROR(
                InsertProperty<NodeType>(node_id, name, is_custom_property,
                                         curr_properties.at(name.data())));
          } else {
            MLMD_RETURN_IF_ERROR(UpdateProperty<NodeType>(
                node_id, name, curr_properties.at(name.data())));
          }
          output_num_changed_properties++;
        }
      }
    } else {
      if (name_in_current_properties) {
        // Generate insert clauses for properties in C \ P
        MLMD_RETURN_IF_ERROR(
            InsertProperty<NodeType>(node_id, name, is_custom_property,
                                     curr_properties.at(name.data())));
        output_num_changed_properties++;
      }
      // Else, do nothing because `name` does not exist in either
      // `prev_properties` nor `curr_properties`
    }
  }
  return output_num_changed_properties;
}

// Creates a query to insert an artifact type.
absl::Status RDBMSMetadataAccessObject::InsertTypeID(const ArtifactType& type,
                                                     int64_t* type_id) {
  return executor_->InsertArtifactType(
      type.name(), GetTypeVersion(type),
      type.has_description() ? absl::make_optional(type.description())
                             : absl::nullopt,
      type.has_external_id() ? absl::make_optional(type.external_id())
                             : absl::nullopt,
      type_id);
}

// Creates a query to insert an execution type.
absl::Status RDBMSMetadataAccessObject::InsertTypeID(const ExecutionType& type,
                                                     int64_t* type_id) {
  return executor_->InsertExecutionType(
      type.name(), GetTypeVersion(type),
      type.has_description() ? absl::make_optional(type.description())
                             : absl::nullopt,
      type.has_input_type() ? &type.input_type() : nullptr,
      type.has_output_type() ? &type.output_type() : nullptr,
      type.has_external_id() ? absl::make_optional(type.external_id())
                             : absl::nullopt,
      type_id);
}

// Creates a query to insert a context type.
absl::Status RDBMSMetadataAccessObject::InsertTypeID(const ContextType& type,
                                                     int64_t* type_id) {
  return executor_->InsertContextType(
      type.name(), GetTypeVersion(type),
      type.has_description() ? absl::make_optional(type.description())
                             : absl::nullopt,
      type.has_external_id() ? absl::make_optional(type.external_id())
                             : absl::nullopt,
      type_id);
}

// Creates a `Type` where acceptable ones are in {ArtifactType, ExecutionType,
// ContextType}.
// Returns INVALID_ARGUMENT error, if name field is not given.
// Returns INVALID_ARGUMENT error, if any property type is unknown.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Type>
absl::Status RDBMSMetadataAccessObject::CreateTypeImpl(const Type& type,
                                                       int64_t* type_id) {
  const std::string& type_name = type.name();
  const google::protobuf::Map<std::string, PropertyType>& type_properties =
      type.properties();

  // validate the given type
  if (type_name.empty()) {
    return absl::InvalidArgumentError("No type name is specified.");
  }

  // insert a type and get its given id
  MLMD_RETURN_IF_ERROR(InsertTypeID(type, type_id));

  // insert type properties and commit
  for (const auto& property : type_properties) {
    const std::string& property_name = property.first;
    const PropertyType property_type = property.second;
    if (property_type == PropertyType::UNKNOWN) {
      LOG(ERROR) << "Property " << property_name << "'s value type is UNKNOWN.";
      return absl::InvalidArgumentError(
          absl::StrCat("Property ", property_name, " is UNKNOWN."));
    }
    MLMD_RETURN_IF_ERROR(
        executor_->InsertTypeProperty(*type_id, property_name, property_type));
  }
  return absl::OkStatus();
}

// Generates a query to find all type instances.
absl::Status RDBMSMetadataAccessObject::GenerateFindAllTypeInstancesQuery(
    const TypeKind type_kind, RecordSet* record_set) {
  return executor_->SelectAllTypes(type_kind, record_set);
}

// FindType takes a result of a query for types, and populates additional
// information such as properties, and returns it in `types`.
template <typename MessageType>
absl::Status RDBMSMetadataAccessObject::FindTypesFromRecordSet(
    const RecordSet& type_record_set, std::vector<MessageType>* types,
    bool get_properties) {
  // Query type with the given condition
  const int num_records = type_record_set.records_size();
  if (num_records == 0) return absl::OkStatus();

  types->resize(num_records);
  for (int i = 0; i < num_records; ++i) {
    MLMD_RETURN_IF_ERROR(
        ParseRecordSetToMessage(type_record_set, &types->at(i), i));
  }
  if (get_properties) {
    MLMD_RETURN_IF_ERROR(PopulateTypeProperties(*executor_, *types));
  }

  return absl::OkStatus();
}

template <typename MessageType>
absl::Status RDBMSMetadataAccessObject::FindTypesImpl(
    absl::Span<const int64_t> type_ids, bool get_properties,
    std::vector<MessageType>& types) {
  if (type_ids.empty()) {
    return absl::InvalidArgumentError("ids cannot be empty");
  }
  if (!types.empty()) {
    return absl::InvalidArgumentError("types parameter is not empty");
  }
  MessageType dummy_type;
  const TypeKind type_kind = ResolveTypeKind(&dummy_type);

  absl::flat_hash_set<int64_t> deduped_id_set;
  for (const auto& id : type_ids) {
    deduped_id_set.insert(id);
  }
  std::vector<int64_t> deduped_ids;
  absl::c_transform(deduped_id_set, std::back_inserter(deduped_ids),
                    [](const int64_t id) { return id; });

  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectTypesByID(deduped_ids, type_kind, &record_set));
  MLMD_RETURN_IF_ERROR(
      FindTypesFromRecordSet(record_set, &types, get_properties));

  if (deduped_ids.size() != types.size()) {
    std::vector<int64_t> found_ids;
    absl::c_transform(types, std::back_inserter(found_ids),
                      [](const MessageType& type) { return type.id(); });
    return absl::NotFoundError(absl::StrCat(
        "Results missing for ids: {", absl::StrJoin(deduped_ids, ","),
        "}. Found results for {", absl::StrJoin(found_ids, ","), "}"));
  }
  return absl::OkStatus();
}

template <typename MessageType>
absl::Status RDBMSMetadataAccessObject::FindTypeImpl(int64_t type_id,
                                                     MessageType* type) {
  const TypeKind type_kind = ResolveTypeKind(type);
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectTypeByID(type_id, type_kind, &record_set));
  std::vector<MessageType> types;
  MLMD_RETURN_IF_ERROR(FindTypesFromRecordSet(record_set, &types));
  if (types.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No type found for query, type_id: ", type_id));
  }
  *type = std::move(types[0]);
  return absl::OkStatus();
}

template <typename MessageType>
absl::Status RDBMSMetadataAccessObject::FindTypesByExternalIdsImpl(
    absl::Span<absl::string_view> external_ids, const bool get_properties,
    std::vector<MessageType>& types) {
  if (external_ids.empty()) {
    return absl::OkStatus();
  }
  if (!types.empty()) {
    return absl::InvalidArgumentError("types parameter is not empty");
  }
  MessageType dummy_type;
  const TypeKind type_kind = ResolveTypeKind(&dummy_type);

  absl::flat_hash_set<absl::string_view> deduped_external_id_set;
  for (const auto& external_id : external_ids) {
    if (external_id.empty())
      return absl::InvalidArgumentError(
          "Invalid argument because at least one empty external_id exists.");
    deduped_external_id_set.insert(external_id);
  }
  std::vector<absl::string_view> deduped_external_ids;
  absl::c_transform(deduped_external_id_set,
                    std::back_inserter(deduped_external_ids),
                    [](absl::string_view external_id) { return external_id; });

  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectTypesByExternalIds(
      deduped_external_ids.empty()
          ? absl::Span<absl::string_view>()
          : absl::MakeSpan(&deduped_external_ids[0],
                           deduped_external_ids.size()),
      type_kind, &record_set));
  MLMD_RETURN_IF_ERROR(
      FindTypesFromRecordSet(record_set, &types, get_properties));

  if (types.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No types found for external_ids."));
  }
  return absl::OkStatus();
}

template <typename MessageType>
absl::Status RDBMSMetadataAccessObject::FindTypeImpl(
    absl::string_view name, std::optional<absl::string_view> version,
    MessageType* type) {
  const TypeKind type_kind = ResolveTypeKind(type);
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectTypeByNameAndVersion(
      name, version, type_kind, &record_set));
  std::vector<MessageType> types;
  MLMD_RETURN_IF_ERROR(FindTypesFromRecordSet(record_set, &types));
  if (types.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No type found for query, name: `", name, "`, version: `",
                     version ? *version : "nullopt", "`"));
  }
  *type = std::move(types[0]);
  return absl::OkStatus();
}

template <typename MessageType>
absl::Status RDBMSMetadataAccessObject::FindTypesImpl(
    absl::Span<std::pair<std::string, std::string>> names_and_versions,
    std::vector<MessageType>& types) {
  if (names_and_versions.empty()) {
    return absl::InvalidArgumentError("names_and_versions cannot be empty");
  }
  if (!types.empty()) {
    return absl::InvalidArgumentError("types parameter is not empty");
  }
  MessageType dummy_type;
  const TypeKind type_kind = ResolveTypeKind(&dummy_type);
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectTypesByNamesAndVersions(
      names_and_versions, type_kind, &record_set));

  return FindTypesFromRecordSet(record_set, &types);
}

// Finds all type instances of the type `MessageType`.
// Returns detailed INTERNAL error, if query execution fails.
template <typename MessageType>
absl::Status RDBMSMetadataAccessObject::FindAllTypeInstancesImpl(
    std::vector<MessageType>* types) {
  MessageType type;
  const TypeKind type_kind = ResolveTypeKind(&type);
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      GenerateFindAllTypeInstancesQuery(type_kind, &record_set));

  return FindTypesFromRecordSet(record_set, types);
}

// Updates an existing type. A type is one of {ArtifactType, ExecutionType,
// ContextType}
// Returns INVALID_ARGUMENT error, if name field is not given.
// Returns INVALID_ARGUMENT error, if id field is given and is different.
// Returns INVALID_ARGUMENT error, if any property type is unknown.
// Returns ALREADY_EXISTS error, if any property type is different.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Type>
absl::Status RDBMSMetadataAccessObject::UpdateTypeImpl(const Type& type) {
  if (!type.has_name()) {
    return absl::InvalidArgumentError("No type name is specified.");
  }
  // find the current stored type and validate the id.
  Type stored_type;
  MLMD_RETURN_IF_ERROR(FindTypeByNameAndVersion(
      type.name(), GetTypeVersion(type), &stored_type));
  if (type.has_id() && type.id() != stored_type.id()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Given type id is different from the existing type: ",
                     stored_type.DebugString()));
  }
  // update the list of type properties
  const google::protobuf::Map<std::string, PropertyType>& stored_properties =
      stored_type.properties();
  for (const auto& p : type.properties()) {
    const std::string& property_name = p.first;
    const PropertyType property_type = p.second;
    if (property_type == PropertyType::UNKNOWN) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Property:", property_name, " type should not be UNKNOWN."));
    }
    if (stored_properties.find(property_name) != stored_properties.end()) {
      // for stored properties, type should not be changed.
      if (stored_properties.at(property_name) != property_type) {
        return absl::AlreadyExistsError(
            absl::StrCat("Property:", property_name,
                         " type is different from the existing type: ",
                         stored_type.DebugString()));
      }
      continue;
    }
    MLMD_RETURN_IF_ERROR(executor_->InsertTypeProperty(
        stored_type.id(), property_name, property_type));
  }

  // Update the external_id.
  if (type.has_id() && type.has_external_id()) {
    MLMD_RETURN_IF_ERROR(
        executor_->UpdateTypeExternalIdDirect(type.id(), type.external_id()));
  }
  return absl::OkStatus();
}

template <typename Type>
absl::Status RDBMSMetadataAccessObject::FindParentTypesByTypeIdImpl(
    absl::Span<const int64_t> type_ids,
    absl::flat_hash_map<int64_t, Type>& output_parent_types) {
  if (type_ids.empty()) {
    return absl::InvalidArgumentError("type_ids cannot be empty");
  }
  if (!output_parent_types.empty()) {
    return absl::InvalidArgumentError("output_parent_types is not empty");
  }

  // Retrieve parent types based on `type_ids`.
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectParentTypesByTypeID(type_ids, &record_set));
  if (record_set.records_size() == 0) return absl::OkStatus();

  // `type_id` and `parent_type_id` have a 1:1 mapping based on the database
  // records.
  std::vector<int64_t> type_ids_with_parent, parent_type_ids;
  ConvertToIdAndParentIds(record_set, type_ids_with_parent, parent_type_ids);

  // Creates a {parent_id, parent_type} mapping.
  std::vector<Type> parent_types;
  MLMD_RETURN_IF_ERROR(
      FindTypesImpl(parent_type_ids, /*get_properties=*/false, parent_types));
  absl::flat_hash_map<int64_t, Type> parent_id_to_type;
  for (const auto& parent_type : parent_types) {
    parent_id_to_type.insert({parent_type.id(), parent_type});
  }

  // Returns a {type_id, parent_type} mapping. Each type_id in
  // `type_ids_with_parent` maps to its parent type.
  for (int i = 0; i < type_ids_with_parent.size(); ++i) {
    output_parent_types.insert(
        {type_ids_with_parent[i], parent_id_to_type[parent_type_ids[i]]});
  }
  return absl::OkStatus();
}

// Creates an `Node`, which is one of {`Artifact`, `Execution`, `Context`},
// then returns the assigned node id. The node's id field is ignored. The node
// should have a `NodeType`, which is one of {`ArtifactType`, `ExecutionType`,
// `ContextType`}.
// Returns INVALID_ARGUMENT error, if the node does not align with its type.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Node, typename NodeType>
absl::Status RDBMSMetadataAccessObject::CreateNodeImpl(
    const Node& node, const bool skip_type_and_property_validation,
    const absl::Time create_timestamp, int64_t* node_id) {
  // clear node id
  *node_id = 0;
  if (!skip_type_and_property_validation) {
    // validate type
    if (!node.has_type_id())
      return absl::InvalidArgumentError("Type id is missing.");
    const int64_t type_id = node.type_id();
    NodeType node_type;
    MLMD_RETURN_WITH_CONTEXT_IF_ERROR(FindTypeImpl(type_id, &node_type),
                                      "Cannot find type for ",
                                      node.ShortDebugString());

    // validate properties
    MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
        ValidatePropertiesWithType(node, node_type),
        "Cannot validate properties of ", node.ShortDebugString());
  }

  // insert a node and get the assigned id
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(
      CreateBasicNode(node, create_timestamp, node_id),
      "Cannot create node for ", node.ShortDebugString());

  // insert properties
  const google::protobuf::Map<std::string, Value> prev_properties;
  int num_changed_properties = 0;
  MLMD_ASSIGN_OR_RETURN(
      num_changed_properties,
      ModifyProperties<NodeType>(node.properties(), prev_properties, *node_id,
                                 /*is_custom_property=*/false));
  int num_changed_custom_properties = 0;
  MLMD_ASSIGN_OR_RETURN(num_changed_custom_properties,
                        ModifyProperties<NodeType>(
                            node.custom_properties(), prev_properties, *node_id,
                            /*is_custom_property=*/true));
  return absl::OkStatus();
}

template <typename Node>
absl::Status RDBMSMetadataAccessObject::FindNodesImpl(
    absl::Span<const int64_t> node_ids, const bool skipped_ids_ok,
    std::vector<Node>& nodes) {
  if (node_ids.empty()) {
    return absl::InvalidArgumentError("ids cannot be empty");
  }

  if (!nodes.empty()) {
    return absl::InvalidArgumentError("nodes parameter is not empty");
  }

  RecordSet node_record_set;
  RecordSet properties_record_set;

  MLMD_RETURN_IF_ERROR(RetrieveNodesById<Node>(node_ids, &node_record_set,
                                               &properties_record_set));

  MLMD_RETURN_IF_ERROR(ParseRecordSetToNodeArray(node_record_set, nodes));
  for (const auto& node : nodes) {
    if (!node.has_type_id()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "There exists nodes without an associated node type.",
          node.DebugString(),
          "Note that MLMD does not allow creating nodes without a node type."
          "Please check if there is something wrong with your pipeline."));
    }
  }
  // if there are properties associated with the nodes, parse the returned
  // values.
  if (!properties_record_set.records().empty()) {
    // First we build a hash map from node ids to Node messages, to
    // facilitate lookups.
    absl::flat_hash_map<int64_t, typename std::vector<Node>::iterator>
        node_by_id;
    for (auto i = nodes.begin(); i != nodes.end(); ++i) {
      node_by_id.insert({i->id(), i});
    }

    // previous metadata source versions have fewer property types
    CHECK_GE(properties_record_set.column_names_size(), 6);
    CHECK_LE(properties_record_set.column_names_size(), kPropertyRecordSetSize);
    for (const RecordSet::Record& record : properties_record_set.records()) {
      // Match the record against a node in the hash map.
      int64_t node_id;
      CHECK(absl::SimpleAtoi(record.values(0), &node_id));
      auto iter = node_by_id.find(node_id);
      CHECK(iter != node_by_id.end());
      Node& node = *iter->second;

      MLMD_RETURN_IF_ERROR(PopulateNodeProperties(record, *executor_, node));
    }
  }

  if (node_ids.size() != nodes.size()) {
    std::vector<int64_t> found_ids;
    absl::c_transform(nodes, std::back_inserter(found_ids),
                      [](const Node& node) { return node.id(); });

    const std::string message = absl::StrCat(
        "Results missing for ids: {", absl::StrJoin(node_ids, ","),
        "}. Found results for {", absl::StrJoin(found_ids, ","), "}");

    if (!skipped_ids_ok) {
      return absl::InternalError(message);
    } else {
      return absl::NotFoundError(message);
    }
  }
  return absl::OkStatus();
}

template <typename Node, typename NodeType>
absl::Status RDBMSMetadataAccessObject::FindNodesWithTypesImpl(
    absl::Span<const int64_t> node_ids, std::vector<Node>& nodes,
    std::vector<NodeType>& node_types) {
  if (node_ids.empty()) {
    return absl::InvalidArgumentError("ids cannot be empty");
  }

  if (!nodes.empty()) {
    return absl::InvalidArgumentError("nodes parameter is not empty");
  }

  RecordSet node_record_set;
  RecordSet properties_record_set;

  MLMD_RETURN_IF_ERROR(RetrieveNodesById<Node>(node_ids, &node_record_set,
                                               &properties_record_set));

  MLMD_RETURN_IF_ERROR(ParseRecordSetToNodeArray(node_record_set, nodes));
  for (const auto& node : nodes) {
    if (!node.has_type_id()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "There exists nodes without an associated node type.",
          node.DebugString(),
          "Note that MLMD does not allow creating nodes without a node type."
          "Please check if there is something wrong with your pipeline."));
    }
  }
  // if there are properties associated with the nodes, parse the returned
  // values.
  if (!properties_record_set.records().empty()) {
    // First we build a hash map from node ids to Node messages, to
    // facilitate lookups.
    absl::flat_hash_map<int64_t, typename std::vector<Node>::iterator>
        node_by_id;
    for (auto i = nodes.begin(); i != nodes.end(); ++i) {
      node_by_id.insert({i->id(), i});
    }

    // previous metadata source versions have fewer property types
    CHECK_GE(properties_record_set.column_names_size(), 6);
    CHECK_LE(properties_record_set.column_names_size(), kPropertyRecordSetSize);
    for (const RecordSet::Record& record : properties_record_set.records()) {
      // Match the record against a node in the hash map.
      int64_t node_id;
      CHECK(absl::SimpleAtoi(record.values(0), &node_id));
      auto iter = node_by_id.find(node_id);
      CHECK(iter != node_by_id.end());
      Node& node = *iter->second;

      MLMD_RETURN_IF_ERROR(PopulateNodeProperties(record, *executor_, node));
    }
  }
  absl::Status return_status = absl::OkStatus();
  if (node_ids.size() != nodes.size()) {
    std::vector<int64_t> found_ids;
    absl::c_transform(nodes, std::back_inserter(found_ids),
                      [](const Node& node) { return node.id(); });

    const std::string message = absl::StrCat(
        "Results missing for ids: {", absl::StrJoin(node_ids, ","),
        "}. Found results for {", absl::StrJoin(found_ids, ","), "}");
    return_status = absl::NotFoundError(message);
  }
  // Extract type fields from `node_record_set` and populate type properties.
  MLMD_RETURN_IF_ERROR(
      ParseNodeRecordSetToDedupedTypes(node_record_set, node_types));
  MLMD_RETURN_IF_ERROR(PopulateTypeProperties(*executor_, node_types));
  return return_status;
}

template <typename Node>
absl::Status RDBMSMetadataAccessObject::FindNodeImpl(const int64_t node_id,
                                                     Node* node) {
  std::vector<Node> nodes;
  MLMD_RETURN_IF_ERROR(
      FindNodesImpl({node_id}, /*skipped_ids_ok=*/true, nodes));
  *node = nodes.at(0);

  return absl::OkStatus();
}

// Updates a `Node` which is one of {`Artifact`, `Execution`, `Context`} under
// masking.
// `update_timestamp` should be used as the update time of the Node.
// If `mask` is empty, update the node as a whole.
// If `mask` is not empty, only update fields and properties specified in
// `mask`.
// Returns INVALID_ARGUMENT error, if the node cannot be found.
// Returns INVALID_ARGUMENT error, if the node does not match with its type.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Node, typename NodeType>
absl::Status RDBMSMetadataAccessObject::UpdateNodeImpl(
    const Node& node, const absl::Time update_timestamp, bool force_update_time,
    const google::protobuf::FieldMask& mask) {
  // validate node
  if (!node.has_id()) return absl::InvalidArgumentError("No id is given.");

  Node stored_node;
  absl::Status status = FindNodeImpl(node.id(), &stored_node);
  if (absl::IsNotFound(status)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot find the given id ", node.id()));
  }
  if (!status.ok()) return status;
  if (node.has_type_id() && node.type_id() != stored_node.type_id()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Given type_id ", node.type_id(),
        " is different from the one known before: ", stored_node.type_id()));
  }
  const int64_t type_id = stored_node.type_id();

  NodeType stored_type;
  MLMD_RETURN_IF_ERROR(FindTypeImpl(type_id, &stored_type));
  MLMD_RETURN_IF_ERROR(ValidatePropertiesWithType(node, stored_type, mask));

  // Update, insert, delete properties if changed.
  int num_changed_properties = 0;
  MLMD_ASSIGN_OR_RETURN(
      num_changed_properties,
      ModifyProperties<NodeType>(node.properties(), stored_node.properties(),
                                 node.id(),
                                 /*is_custom_property=*/false, mask));
  int num_changed_custom_properties = 0;
  MLMD_ASSIGN_OR_RETURN(
      num_changed_custom_properties,
      ModifyProperties<NodeType>(node.custom_properties(),
                                 stored_node.custom_properties(), node.id(),
                                 /*is_custom_property=*/true, mask));
  Node mutable_node = node;
  // This is added to prevent the case where no `type_id` is specified. The
  // `type_id` of stored_node will be updated to 0.
  if (!node.has_type_id()) mutable_node.set_type_id(stored_node.type_id());

  // If `force_update_time` is set to True. Always update node regardless of
  // whether input node is the same as stored node or not.
  if (force_update_time) {
    return RunMaskedNodeUpdate(mutable_node, stored_node, update_timestamp,
                               mask);
  }
  // Update node if attributes are different or properties are updated, so
  // that the last_update_time_since_epoch is updated properly.
  google::protobuf::util::MessageDifferencer diff;
  // Ignore `type_id` because if node's type is different from stored_node, an
  // invalid_argument error is already returned.
  diff.IgnoreField(Node::descriptor()->FindFieldByName("type_id"));
  diff.IgnoreField(Node::descriptor()->FindFieldByName("properties"));
  diff.IgnoreField(Node::descriptor()->FindFieldByName("custom_properties"));
  // create_time_since_epoch and last_update_time_since_epoch are output only
  // fields. Two nodes are treated as equal as long as other fields match.
  diff.IgnoreField(
      Node::descriptor()->FindFieldByName("create_time_since_epoch"));
  diff.IgnoreField(
      Node::descriptor()->FindFieldByName("last_update_time_since_epoch"));
  if (!diff.Compare(node, stored_node) ||
      num_changed_properties + num_changed_custom_properties > 0) {
    MLMD_RETURN_IF_ERROR(
        RunMaskedNodeUpdate(mutable_node, stored_node, update_timestamp, mask));
  }
  return absl::OkStatus();
}

// Takes a record set that has one record per event, parses them into Event
// objects, gets the paths for the events from the database using collected
// event ids, and assign paths to each corresponding event.
// Returns INVALID_ARGUMENT error, if the `events` is null.
absl::Status RDBMSMetadataAccessObject::FindEventsFromRecordSet(
    const RecordSet& event_record_set, std::vector<Event>* events) {
  if (events == nullptr)
    return absl::InvalidArgumentError("Given events is NULL.");

  events->reserve(event_record_set.records_size());
  MLMD_RETURN_IF_ERROR(ParseRecordSetToEdgeArray(event_record_set, *events));

  absl::flat_hash_map<int64_t, Event*> event_id_to_event_map;
  std::vector<int64_t> event_ids;
  event_ids.reserve(event_record_set.records_size());
  for (int i = 0; i < events->size(); ++i) {
    CHECK_LT(i, event_record_set.records_size());
    const RecordSet::Record& record = event_record_set.records()[i];
    int64_t event_id;
    CHECK(absl::SimpleAtoi(record.values(0), &event_id));
    event_id_to_event_map[event_id] = &(*events)[i];
    event_ids.push_back(event_id);
  }

  RecordSet path_record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectEventPathByEventIDs(event_ids, &path_record_set));
  for (const RecordSet::Record& record : path_record_set.records()) {
    int64_t event_id;
    CHECK(absl::SimpleAtoi(record.values(0), &event_id));
    auto iter = event_id_to_event_map.find(event_id);
    CHECK(iter != event_id_to_event_map.end());
    Event* event = iter->second;
    bool is_index_step;
    CHECK(absl::SimpleAtob(record.values(1), &is_index_step));
    if (is_index_step) {
      int64_t step_index;
      CHECK(absl::SimpleAtoi(record.values(2), &step_index));
      event->mutable_path()->add_steps()->set_index(step_index);
    } else {
      event->mutable_path()->add_steps()->set_key(record.values(3));
    }
  }
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::FindAssociationsFromRecordSet(
    const RecordSet& association_record_set,
    std::vector<Association>* associations) {
  if (associations == nullptr)
    return absl::InvalidArgumentError("Given associations is NULL.");

  associations->reserve(association_record_set.records_size());
  MLMD_RETURN_IF_ERROR(
      ParseRecordSetToEdgeArray(association_record_set, *associations));

  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::FindAttributionsFromRecordSet(
    const RecordSet& attribution_record_set,
    std::vector<Attribution>* attributions) {
  if (attributions == nullptr)
    return absl::InvalidArgumentError("Given attributions is NULL.");

  attributions->reserve(attribution_record_set.records_size());
  MLMD_RETURN_IF_ERROR(
      ParseRecordSetToEdgeArray(attribution_record_set, *attributions));

  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::CreateType(const ArtifactType& type,
                                                   int64_t* type_id) {
  return CreateTypeImpl(type, type_id);
}

absl::Status RDBMSMetadataAccessObject::CreateType(const ExecutionType& type,
                                                   int64_t* type_id) {
  return CreateTypeImpl(type, type_id);
}

absl::Status RDBMSMetadataAccessObject::CreateType(const ContextType& type,
                                                   int64_t* type_id) {
  return CreateTypeImpl(type, type_id);
}

absl::Status RDBMSMetadataAccessObject::FindTypeById(
    const int64_t type_id, ArtifactType* artifact_type) {
  return FindTypeImpl(type_id, artifact_type);
}

absl::Status RDBMSMetadataAccessObject::FindTypeById(
    const int64_t type_id, ExecutionType* execution_type) {
  return FindTypeImpl(type_id, execution_type);
}

absl::Status RDBMSMetadataAccessObject::FindTypes(
    std::vector<ArtifactType>* artifact_types) {
  return FindAllTypeInstancesImpl(artifact_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypeById(
    const int64_t type_id, ContextType* context_type) {
  return FindTypeImpl(type_id, context_type);
}

absl::Status RDBMSMetadataAccessObject::FindTypesByIds(
    absl::Span<const int64_t> type_ids,
    std::vector<ArtifactType>& artifact_types) {
  return FindTypesImpl(type_ids, /*get_properties=*/true, artifact_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypesByIds(
    absl::Span<const int64_t> type_ids,
    std::vector<ExecutionType>& execution_types) {
  return FindTypesImpl(type_ids, /*get_properties=*/true, execution_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypesByIds(
    absl::Span<const int64_t> type_ids,
    std::vector<ContextType>& context_types) {
  return FindTypesImpl(type_ids, /*get_properties=*/true, context_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypesByExternalIds(
    absl::Span<absl::string_view> external_ids,
    std::vector<ArtifactType>& artifact_types) {
  return FindTypesByExternalIdsImpl(external_ids, /*get_properties=*/true,
                                    artifact_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypesByExternalIds(
    absl::Span<absl::string_view> external_ids,
    std::vector<ExecutionType>& execution_types) {
  return FindTypesByExternalIdsImpl(external_ids, /*get_properties=*/true,
                                    execution_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypesByExternalIds(
    absl::Span<absl::string_view> external_ids,
    std::vector<ContextType>& context_types) {
  return FindTypesByExternalIdsImpl(external_ids, /*get_properties=*/true,
                                    context_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypes(
    std::vector<ExecutionType>* execution_types) {
  return FindAllTypeInstancesImpl(execution_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypes(
    std::vector<ContextType>* context_types) {
  return FindAllTypeInstancesImpl(context_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypeByNameAndVersion(
    absl::string_view name, std::optional<absl::string_view> version,
    ArtifactType* artifact_type) {
  return FindTypeImpl(name, version, artifact_type);
}

absl::Status RDBMSMetadataAccessObject::FindTypeByNameAndVersion(
    absl::string_view name, std::optional<absl::string_view> version,
    ExecutionType* execution_type) {
  return FindTypeImpl(name, version, execution_type);
}

absl::Status RDBMSMetadataAccessObject::FindTypeByNameAndVersion(
    absl::string_view name, std::optional<absl::string_view> version,
    ContextType* context_type) {
  return FindTypeImpl(name, version, context_type);
}

absl::Status RDBMSMetadataAccessObject::FindTypeIdByNameAndVersion(
    absl::string_view name, std::optional<absl::string_view> version,
    TypeKind type_kind, int64_t* type_id) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectTypeByNameAndVersion(
      name, version, type_kind, &record_set));
  if (record_set.records().empty()) {
    return absl::NotFoundError(
        absl::StrCat("No type found for query, name: `", name, "`, version: `",
                     version ? *version : "nullopt", "`"));
  }
  for (int i = 0; i < record_set.column_names_size(); i++) {
    if (record_set.column_names(i) == "id") {
      if (absl::SimpleAtoi(record_set.records(0).values(i), type_id)) {
        return absl::OkStatus();
      }
      return absl::InternalError(absl::StrCat(
          "Cannot parse RecordSet for type_id on type name: `", name,
          "`, version: `", version ? *version : "nullopt", "`"));
    }
  }
  return absl::NotFoundError(
      absl::StrCat("No type_id found from RecordSet for type name: `", name,
                   "`, version: `", version ? *version : "nullopt", "`"));
}

absl::Status RDBMSMetadataAccessObject::FindTypesByNamesAndVersions(
    absl::Span<std::pair<std::string, std::string>> names_and_versions,
    std::vector<ArtifactType>& artifact_types) {
  return FindTypesImpl(names_and_versions, artifact_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypesByNamesAndVersions(
    absl::Span<std::pair<std::string, std::string>> names_and_versions,
    std::vector<ExecutionType>& execution_types) {
  return FindTypesImpl(names_and_versions, execution_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypesByNamesAndVersions(
    absl::Span<std::pair<std::string, std::string>> names_and_versions,
    std::vector<ContextType>& context_types) {
  return FindTypesImpl(names_and_versions, context_types);
}

absl::Status RDBMSMetadataAccessObject::UpdateType(const ArtifactType& type) {
  return UpdateTypeImpl(type);
}

absl::Status RDBMSMetadataAccessObject::UpdateType(const ExecutionType& type) {
  return UpdateTypeImpl(type);
}

absl::Status RDBMSMetadataAccessObject::UpdateType(const ContextType& type) {
  return UpdateTypeImpl(type);
}

absl::Status RDBMSMetadataAccessObject::CreateParentTypeInheritanceLink(
    const ArtifactType& type, const ArtifactType& parent_type) {
  if (!type.has_id() || !parent_type.has_id()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing id in the given types: ", type.DebugString(),
                     parent_type.DebugString()));
  }
  MLMD_RETURN_IF_ERROR(CheckCyClicDependency<ArtifactType>(
      /*child_id=*/type.id(), /*parent_id=*/parent_type.id(), executor_));
  const absl::Status status =
      executor_->InsertParentType(type.id(), parent_type.id());
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError("The ParentType already exists.");
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::CreateParentTypeInheritanceLink(
    const ExecutionType& type, const ExecutionType& parent_type) {
  if (!type.has_id() || !parent_type.has_id()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing id in the given types: ", type.DebugString(),
                     parent_type.DebugString()));
  }
  MLMD_RETURN_IF_ERROR(CheckCyClicDependency<ExecutionType>(
      /*child_id=*/type.id(), /*parent_id=*/parent_type.id(), executor_));
  const absl::Status status =
      executor_->InsertParentType(type.id(), parent_type.id());
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError("The ParentType already exists.");
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::CreateParentTypeInheritanceLink(
    const ContextType& type, const ContextType& parent_type) {
  if (!type.has_id() || !parent_type.has_id()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing id in the given types: ", type.DebugString(),
                     parent_type.DebugString()));
  }
  MLMD_RETURN_IF_ERROR(CheckCyClicDependency<ContextType>(

      /*child_id=*/type.id(), /*parent_id=*/parent_type.id(), executor_));
  const absl::Status status =
      executor_->InsertParentType(type.id(), parent_type.id());
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError("The ParentType already exists.");
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::DeleteParentTypeInheritanceLink(
    int64_t type_id, int64_t parent_type_id) {
  return executor_->DeleteParentType(type_id, parent_type_id);
}

absl::Status RDBMSMetadataAccessObject::FindParentTypesByTypeId(
    absl::Span<const int64_t> type_ids,
    absl::flat_hash_map<int64_t, ArtifactType>& output_parent_types) {
  return FindParentTypesByTypeIdImpl(type_ids, output_parent_types);
}

absl::Status RDBMSMetadataAccessObject::FindParentTypesByTypeId(
    absl::Span<const int64_t> type_ids,
    absl::flat_hash_map<int64_t, ExecutionType>& output_parent_types) {
  return FindParentTypesByTypeIdImpl(type_ids, output_parent_types);
}

absl::Status RDBMSMetadataAccessObject::FindParentTypesByTypeId(
    absl::Span<const int64_t> type_ids,
    absl::flat_hash_map<int64_t, ContextType>& output_parent_types) {
  return FindParentTypesByTypeIdImpl(type_ids, output_parent_types);
}

absl::Status RDBMSMetadataAccessObject::CreateArtifact(
    const Artifact& artifact, const bool skip_type_and_property_validation,
    int64_t* artifact_id) {
  const absl::Status& status = CreateNodeImpl<Artifact, ArtifactType>(
      artifact, skip_type_and_property_validation, absl::Now(), artifact_id);
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Given node already exists: ", artifact.DebugString(),
                     status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::CreateArtifact(
    const Artifact& artifact, const bool skip_type_and_property_validation,
    const absl::Time create_timestamp, int64_t* artifact_id) {
  const absl::Status& status = CreateNodeImpl<Artifact, ArtifactType>(
      artifact, skip_type_and_property_validation, create_timestamp,
      artifact_id);
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Given node already exists: ", artifact.DebugString(),
                     status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::CreateArtifact(const Artifact& artifact,
                                                       int64_t* artifact_id) {
  return CreateArtifact(artifact, /*skip_type_and_property_validation=*/false,
                        artifact_id);
}

absl::Status RDBMSMetadataAccessObject::CreateExecution(
    const Execution& execution, const bool skip_type_and_property_validation,
    int64_t* execution_id) {
  const absl::Status& status = CreateNodeImpl<Execution, ExecutionType>(
      execution, skip_type_and_property_validation, absl::Now(), execution_id);
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Given node already exists: ", execution.DebugString(),
                     status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::CreateExecution(
    const Execution& execution, const bool skip_type_and_property_validation,
    const absl::Time create_timestamp, int64_t* execution_id) {
  const absl::Status& status = CreateNodeImpl<Execution, ExecutionType>(
      execution, skip_type_and_property_validation, create_timestamp,
      execution_id);
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Given node already exists: ", execution.DebugString(),
                     status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::CreateExecution(
    const Execution& execution, int64_t* execution_id) {
  return CreateExecution(execution, /*skip_type_and_property_validation=*/false,
                         execution_id);
}

absl::Status RDBMSMetadataAccessObject::CreateContext(
    const Context& context, const bool skip_type_and_property_validation,
    int64_t* context_id) {
  const absl::Status& status = CreateNodeImpl<Context, ContextType>(
      context, skip_type_and_property_validation, absl::Now(), context_id);
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Given node already exists: ", context.DebugString(),
                     status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::CreateContext(
    const Context& context, const bool skip_type_and_property_validation,
    const absl::Time create_timestamp, int64_t* context_id) {
  const absl::Status& status = CreateNodeImpl<Context, ContextType>(
      context, skip_type_and_property_validation, create_timestamp, context_id);
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Given node already exists: ", context.DebugString(),
                     status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::CreateContext(const Context& context,
                                                      int64_t* context_id) {
  return CreateContext(context, /*skip_type_and_property_validation=*/false,
                       context_id);
}

absl::Status RDBMSMetadataAccessObject::FindArtifactsById(
    absl::Span<const int64_t> artifact_ids, std::vector<Artifact>* artifacts) {
  if (artifact_ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(artifact_ids, /*skipped_ids_ok=*/true, *artifacts);
}

absl::Status RDBMSMetadataAccessObject::FindArtifactsById(
    absl::Span<const int64_t> artifact_ids, std::vector<Artifact>& artifacts,
    std::vector<ArtifactType>& artifact_types) {
  if (artifact_ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesWithTypesImpl(artifact_ids, artifacts, artifact_types);
}

absl::Status RDBMSMetadataAccessObject::FindExecutionsById(
    absl::Span<const int64_t> execution_ids,
    std::vector<Execution>* executions) {
  if (execution_ids.empty()) {
    return absl::OkStatus();
  }

  return FindNodesImpl(execution_ids, /*skipped_ids_ok=*/true, *executions);
}

absl::Status RDBMSMetadataAccessObject::FindContextsById(
    absl::Span<const int64_t> context_ids, std::vector<Context>* contexts) {
  if (context_ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(context_ids, /*skipped_ids_ok=*/true, *contexts);
}

absl::Status RDBMSMetadataAccessObject::FindArtifactsByExternalIds(
    absl::Span<absl::string_view> external_ids,
    std::vector<Artifact>* artifacts) {
  if (external_ids.empty()) {
    return absl::OkStatus();
  }
  for (absl::string_view external_id : external_ids) {
    if (external_id.empty()) {
      return absl::InvalidArgumentError(
          "Invalid argument because at least one empty external_id exists.");
    }
  }
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectArtifactsByExternalIds(external_ids, &record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No artifacts found for external_ids."));
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *artifacts);
}

absl::Status RDBMSMetadataAccessObject::FindExecutionsByExternalIds(
    absl::Span<absl::string_view> external_ids,
    std::vector<Execution>* executions) {
  if (external_ids.empty()) {
    return absl::OkStatus();
  }
  for (absl::string_view external_id : external_ids) {
    if (external_id.empty()) {
      return absl::InvalidArgumentError(
          "Invalid argument because at least one empty external_id exists.");
    }
  }
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectExecutionsByExternalIds(external_ids, &record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No executions found for external_ids."));
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *executions);
}

absl::Status RDBMSMetadataAccessObject::FindContextsByExternalIds(
    absl::Span<absl::string_view> external_ids,
    std::vector<Context>* contexts) {
  if (external_ids.empty()) {
    return absl::OkStatus();
  }
  for (absl::string_view external_id : external_ids) {
    if (external_id.empty()) {
      return absl::InvalidArgumentError(
          "Invalid argument because at least one empty external_id exists.");
    }
  }
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectContextsByExternalIds(external_ids, &record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No contexts found for external_ids."));
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *contexts);
}

absl::Status RDBMSMetadataAccessObject::UpdateArtifact(
    const Artifact& artifact) {
  return UpdateArtifact(artifact, /*update_timestamp=*/absl::Now(),
                        /*force_update_time=*/false,
                        google::protobuf::FieldMask());
}

absl::Status RDBMSMetadataAccessObject::UpdateArtifact(
    const Artifact& artifact, const google::protobuf::FieldMask& mask) {
  return UpdateArtifact(artifact, /*update_timestamp=*/absl::Now(),
                        /*force_update_time=*/false, mask);
}

absl::Status RDBMSMetadataAccessObject::UpdateExecution(
    const Execution& execution) {
  return UpdateExecution(execution, /*update_timestamp=*/absl::Now(),
                         /*force_update_time=*/false);
}

absl::Status RDBMSMetadataAccessObject::UpdateExecution(
    const Execution& execution, bool force_update_time,
    const google::protobuf::FieldMask& mask) {
  return UpdateExecution(execution, /*update_timestamp=*/absl::Now(),
                         force_update_time, mask);
}

absl::Status RDBMSMetadataAccessObject::UpdateContext(const Context& context) {
  return UpdateContext(context, /*update_timestamp=*/absl::Now(),
                       /*force_update_time=*/false);
}

absl::Status RDBMSMetadataAccessObject::UpdateContext(
    const Context& context, const google::protobuf::FieldMask& mask) {
  return UpdateContext(context, /*update_timestamp=*/absl::Now(),
                       /*force_update_time=*/false, mask);
}

absl::Status RDBMSMetadataAccessObject::UpdateArtifact(
    const Artifact& artifact, const absl::Time update_timestamp,
    bool force_update_time) {
  return UpdateNodeImpl<Artifact, ArtifactType>(artifact, update_timestamp,
                                                force_update_time,
                                                google::protobuf::FieldMask());
}

absl::Status RDBMSMetadataAccessObject::UpdateArtifact(
    const Artifact& artifact, const absl::Time update_timestamp,
    bool force_update_time, const google::protobuf::FieldMask& mask) {
  return UpdateNodeImpl<Artifact, ArtifactType>(artifact, update_timestamp,
                                                force_update_time, mask);
}

absl::Status RDBMSMetadataAccessObject::UpdateExecution(
    const Execution& execution, const absl::Time update_timestamp,
    bool force_update_time) {
  return UpdateNodeImpl<Execution, ExecutionType>(
      execution, update_timestamp, force_update_time,
      google::protobuf::FieldMask());
}

absl::Status RDBMSMetadataAccessObject::UpdateExecution(
    const Execution& execution, const absl::Time update_timestamp,
    bool force_update_time, const google::protobuf::FieldMask& mask) {
  return UpdateNodeImpl<Execution, ExecutionType>(execution, update_timestamp,
                                                  force_update_time, mask);
}

absl::Status RDBMSMetadataAccessObject::UpdateContext(
    const Context& context, const absl::Time update_timestamp,
    bool force_update_time) {
  return UpdateNodeImpl<Context, ContextType>(context, update_timestamp,
                                              force_update_time,
                                              google::protobuf::FieldMask());
}

absl::Status RDBMSMetadataAccessObject::UpdateContext(
    const Context& context, const absl::Time update_timestamp,
    bool force_update_time, const google::protobuf::FieldMask& mask) {
  return UpdateNodeImpl<Context, ContextType>(context, update_timestamp,
                                              force_update_time, mask);
}

absl::Status RDBMSMetadataAccessObject::CreateEvent(const Event& event,
                                                    int64_t* event_id) {
  return CreateEvent(event, /*is_already_validated=*/false, event_id);
}

absl::Status RDBMSMetadataAccessObject::CreateEvent(
    const Event& event, const bool is_already_validated, int64_t* event_id) {
  // validate the given event
  if (!event.has_artifact_id())
    return absl::InvalidArgumentError("No artifact id is specified.");
  if (!event.has_execution_id())
    return absl::InvalidArgumentError("No execution id is specified.");
  if (!event.has_type() || event.type() == Event::UNKNOWN)
    return absl::InvalidArgumentError("No event type is specified.");

  // check database for existing artifact and execution
  // skip check iff a transaction guarantees the artifact and execution exist,
  // i.e. `PutExecution()` and `PutLineageSubgraph()` in metadata_store.cc.
  // TODO(b/197686185): Remove validation after migrating to a schema with
  // foreign keys for artifact id and execution id
  if (!is_already_validated) {
    RecordSet artifacts;
    MLMD_RETURN_IF_ERROR(
        executor_->SelectArtifactsByID({event.artifact_id()}, &artifacts));
    RecordSet executions;
    MLMD_RETURN_IF_ERROR(
        executor_->SelectExecutionsByID({event.execution_id()}, &executions));
    RecordSet record_set;
    if (artifacts.records_size() == 0)
      return absl::InvalidArgumentError(
          absl::StrCat("No artifact with the given id ", event.artifact_id()));
    if (executions.records_size() == 0)
      return absl::InvalidArgumentError(absl::StrCat(
          "No execution with the given id ", event.execution_id()));
  }

  // insert an event and get its given id
  int64_t event_time = event.has_milliseconds_since_epoch()
                           ? event.milliseconds_since_epoch()
                           : absl::ToUnixMillis(absl::Now());

  const absl::Status status =
      executor_->InsertEvent(event.artifact_id(), event.execution_id(),
                             event.type(), event_time, event_id);
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Given event already exists: ", event.DebugString(),
                     status.ToString()));
  }
  // insert event paths
  for (const Event::Path::Step& step : event.path().steps()) {
    // step value oneof
    MLMD_RETURN_IF_ERROR(executor_->InsertEventPath(*event_id, step));
  }
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::FindEventsByArtifacts(
    absl::Span<const int64_t> artifact_ids, std::vector<Event>* events) {
  if (events == nullptr) {
    return absl::InvalidArgumentError("Given events is NULL.");
  }

  RecordSet event_record_set;
  if (!artifact_ids.empty()) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectEventByArtifactIDs(artifact_ids, &event_record_set));
  }

  if (event_record_set.records_size() == 0) {
    return absl::NotFoundError("Cannot find events by given artifact ids.");
  }
  return FindEventsFromRecordSet(event_record_set, events);
}

absl::Status RDBMSMetadataAccessObject::FindEventsByExecutions(
    absl::Span<const int64_t> execution_ids, std::vector<Event>* events) {
  if (events == nullptr) {
    return absl::InvalidArgumentError("Given events is NULL.");
  }

  RecordSet event_record_set;
  if (!execution_ids.empty()) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectEventByExecutionIDs(execution_ids, &event_record_set));
  }

  if (event_record_set.records_size() == 0) {
    return absl::NotFoundError("Cannot find events by given execution ids.");
  }
  return FindEventsFromRecordSet(event_record_set, events);
}

absl::Status RDBMSMetadataAccessObject::CreateAssociation(
    const Association& association, int64_t* association_id) {
  return CreateAssociation(association, /*is_already_validated=*/false,
                           association_id);
}

absl::Status RDBMSMetadataAccessObject::CreateAssociation(
    const Association& association, const bool is_already_validated,
    int64_t* association_id) {
  if (!association.has_context_id())
    return absl::InvalidArgumentError("No context id is specified.");
  if (!association.has_execution_id())
    return absl::InvalidArgumentError("No execution id is specified");

  // check database for existing context and execution
  // skip check iff a transaction guarantees the context and execution exist
  // i.e. `PutExecution()` and `PutLineageSubgraph()` in metadata_store.cc
  // TODO(b/197686185): Remove validation after migrating to a schema with
  // foreign keys for context id and execution id
  if (!is_already_validated) {
    RecordSet context_id_header;
    MLMD_RETURN_IF_ERROR(executor_->SelectContextsByID(
        {association.context_id()}, &context_id_header));
    if (context_id_header.records_size() == 0)
      return absl::InvalidArgumentError("Context id not found.");

    RecordSet execution_id_header;
    MLMD_RETURN_IF_ERROR(executor_->SelectExecutionsByID(
        {association.execution_id()}, &execution_id_header));
    if (execution_id_header.records_size() == 0)
      return absl::InvalidArgumentError("Execution id not found.");
  }

  absl::Status status = executor_->InsertAssociation(
      association.context_id(), association.execution_id(), association_id);

  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Given association already exists: ", association.DebugString(),
        status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::FindAssociationsByContexts(
    absl::Span<const int64_t> context_ids,
    std::vector<Association>* associations) {
  if (associations == nullptr) {
    return absl::InvalidArgumentError(
        "Given input associations vector is NULL.");
  }

  RecordSet association_record_set;
  if (!context_ids.empty()) {
    MLMD_RETURN_IF_ERROR(executor_->SelectAssociationByContextIDs(
        context_ids, &association_record_set));
  }

  if (association_record_set.records_size() == 0) {
    return absl::NotFoundError(
        "Cannot find associations by given context ids.");
  }
  return FindAssociationsFromRecordSet(association_record_set, associations);
}

absl::Status RDBMSMetadataAccessObject::FindAssociationsByExecutions(
    absl::Span<const int64_t> execution_ids,
    std::vector<Association>* associations) {
  if (associations == nullptr) {
    return absl::InvalidArgumentError(
        "Given input associations vector is NULL.");
  }
  RecordSet record_set;
  if (!execution_ids.empty()) {
    MLMD_RETURN_IF_ERROR(executor_->SelectAssociationsByExecutionIds(
        execution_ids, &record_set));
  }
  if (record_set.records_size() == 0) {
    return absl::OkStatus();
  }
  MLMD_RETURN_IF_ERROR(FindAssociationsFromRecordSet(record_set, associations));
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::FindAttributionsByArtifacts(
    absl::Span<const int64_t> artifact_ids,
    std::vector<Attribution>* attributions) {
  if (attributions == nullptr) {
    return absl::InvalidArgumentError(
        "Given input attributions vector is NULL.");
  }
  RecordSet record_set;
  if (!artifact_ids.empty()) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectAttributionsByArtifactIds(artifact_ids, &record_set));
  }
  if (record_set.records_size() == 0) {
    return absl::OkStatus();
  }
  MLMD_RETURN_IF_ERROR(FindAttributionsFromRecordSet(record_set, attributions));
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::FindContextsByExecution(
    int64_t execution_id, std::vector<Context>* contexts) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectAssociationsByExecutionIds({execution_id}, &record_set));
  const std::vector<int64_t> context_ids = AssociationsToContextIds(record_set);
  if (context_ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No contexts found for execution_id: ", execution_id));
  }
  return FindNodesImpl(context_ids, /*skipped_ids_ok=*/false, *contexts);
}

absl::Status RDBMSMetadataAccessObject::FindExecutionsByContext(
    int64_t context_id, std::vector<Execution>* executions) {
  std::string unused_next_page_toke;
  return FindExecutionsByContext(context_id, absl::nullopt, executions,
                                 &unused_next_page_toke);
}

absl::Status RDBMSMetadataAccessObject::FindExecutionsByContext(
    int64_t context_id, std::optional<ListOperationOptions> list_options,
    std::vector<Execution>* executions, std::string* next_page_token) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectAssociationByContextIDs({context_id}, &record_set));
  const std::vector<int64_t> ids = AssociationsToExecutionIds(record_set);
  if (ids.empty()) {
    return absl::OkStatus();
  }
  if (list_options.has_value()) {
    return ListNodes<Execution>(list_options.value(), ids, executions,
                                next_page_token);
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *executions);
}

absl::Status RDBMSMetadataAccessObject::CreateAttribution(
    const Attribution& attribution, int64_t* attribution_id) {
  return CreateAttribution(attribution,
                           /*is_already_validated=*/false, attribution_id);
}

absl::Status RDBMSMetadataAccessObject::CreateAttribution(
    const Attribution& attribution, const bool is_already_validated,
    int64_t* attribution_id) {
  if (!attribution.has_context_id())
    return absl::InvalidArgumentError("No context id is specified.");
  if (!attribution.has_artifact_id())
    return absl::InvalidArgumentError("No artifact id is specified");

  // check database for existing context and artifact
  // skip check iff a transaction guarantees the context and artifact exist,
  // i.e. `PutExecution()` and `PutLineageSubgraph()` in metadata_store.cc.
  // TODO(b/197686185): Remove validation after migrating to a schema with
  // foreign keys for context id and artifact id
  if (!is_already_validated) {
    RecordSet context_id_header;
    MLMD_RETURN_IF_ERROR(executor_->SelectContextsByID(
        {attribution.context_id()}, &context_id_header));
    if (context_id_header.records_size() == 0)
      return absl::InvalidArgumentError("Context id not found.");

    RecordSet artifact_id_header;
    MLMD_RETURN_IF_ERROR(executor_->SelectArtifactsByID(
        {attribution.artifact_id()}, &artifact_id_header));
    if (artifact_id_header.records_size() == 0)
      return absl::InvalidArgumentError("Artifact id not found.");
  }

  absl::Status status = executor_->InsertAttributionDirect(
      attribution.context_id(), attribution.artifact_id(), attribution_id);

  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Given attribution already exists: ", attribution.DebugString(),
        status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::FindContextsByArtifact(
    int64_t artifact_id, std::vector<Context>* contexts) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectAttributionsByArtifactIds({artifact_id}, &record_set));
  const std::vector<int64_t> context_ids = AttributionsToContextIds(record_set);
  if (context_ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No contexts found for artifact_id: ", artifact_id));
  }
  return FindNodesImpl(context_ids, /*skipped_ids_ok=*/false, *contexts);
}

absl::Status RDBMSMetadataAccessObject::FindArtifactsByContext(
    int64_t context_id, std::vector<Artifact>* artifacts) {
  std::string unused_next_page_token;
  return FindArtifactsByContext(context_id, absl::nullopt, artifacts,
                                &unused_next_page_token);
}

absl::Status RDBMSMetadataAccessObject::FindArtifactsByContext(
    int64_t context_id, std::optional<ListOperationOptions> list_options,
    std::vector<Artifact>* artifacts, std::string* next_page_token) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectAttributionByContextID(context_id, &record_set));
  const std::vector<int64_t> ids = AttributionsToArtifactIds(record_set);
  if (ids.empty()) {
    return absl::OkStatus();
  }
  if (list_options.has_value()) {
    return ListNodes<Artifact>(list_options.value(), ids, artifacts,
                               next_page_token);
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *artifacts);
}

absl::StatusOr<std::vector<int64_t>>
RDBMSMetadataAccessObject::FindContextIdsByArtifactsAndExecutions(
    absl::Span<const int64_t> artifact_ids,
    absl::Span<const int64_t> execution_ids) {
  absl::flat_hash_set<int64_t> deduped_context_ids;
  if (!artifact_ids.empty()) {
    RecordSet record_set;
    MLMD_RETURN_IF_ERROR(
        executor_->SelectAttributionsByArtifactIds(artifact_ids, &record_set));
    const std::vector<int64_t> context_ids =
        AttributionsToContextIds(record_set);
    absl::c_copy(context_ids,
                 std::inserter(deduped_context_ids, deduped_context_ids.end()));
  }
  if (!execution_ids.empty()) {
    RecordSet record_set;
    MLMD_RETURN_IF_ERROR(executor_->SelectAssociationsByExecutionIds(
        execution_ids, &record_set));
    const std::vector<int64_t> context_ids =
        AssociationsToContextIds(record_set);
    absl::c_copy(context_ids,
                 std::inserter(deduped_context_ids, deduped_context_ids.end()));
  }

  std::vector<int64_t> context_ids;
  absl::c_copy(deduped_context_ids, std::back_inserter(context_ids));
  return context_ids;
}

absl::Status RDBMSMetadataAccessObject::CreateParentContext(
    const ParentContext& parent_context) {
  if (!parent_context.has_parent_id() || !parent_context.has_child_id()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing parent / child id in the parent_context: ",
                     parent_context.DebugString()));
  }
  RecordSet contexts_id_header;
  MLMD_RETURN_IF_ERROR(executor_->SelectContextsByID(
      {parent_context.parent_id(), parent_context.child_id()},
      &contexts_id_header));
  if (contexts_id_header.records_size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Given parent / child id in the parent_context cannot be found: ",
        parent_context.DebugString()));
  }
  MLMD_RETURN_IF_ERROR(CheckCyClicDependency<ParentContext>(
      /*child_id=*/parent_context.child_id(),
      /*parent_id=*/parent_context.parent_id(), executor_));
  const absl::Status status = executor_->InsertParentContext(
      parent_context.parent_id(), parent_context.child_id());
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Given parent_context already exists: ", parent_context.DebugString(),
        status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::FindLinkedContextsImpl(
    int64_t context_id, ParentContextTraverseDirection direction,
    std::vector<Context>& output_contexts) {
  RecordSet record_set;
  if (direction == ParentContextTraverseDirection::kParent) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectParentContextsByContextID(context_id, &record_set));
  } else if (direction == ParentContextTraverseDirection::kChild) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectChildContextsByContextID(context_id, &record_set));
  } else {
    return absl::InternalError("Unexpected ParentContext direction");
  }
  const bool is_parent = direction == ParentContextTraverseDirection::kParent;
  const std::vector<int64_t> ids =
      ParentContextsToContextIds(record_set, is_parent);
  output_contexts.clear();
  if (ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, output_contexts);
}

absl::Status RDBMSMetadataAccessObject::FindLinkedContextsMapImpl(
    absl::Span<const int64_t> context_ids,
    ParentContextTraverseDirection direction,
    absl::node_hash_map<int64_t, std::vector<Context>>& output_contexts) {
  if (context_ids.empty()) {
    return absl::InvalidArgumentError("Given context_ids is empty.");
  }
  RecordSet record_set;
  if (direction == ParentContextTraverseDirection::kParent) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectParentContextsByContextIDs(context_ids, &record_set));
  } else if (direction == ParentContextTraverseDirection::kChild) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectChildContextsByContextIDs(context_ids, &record_set));
  } else {
    return absl::InternalError("Unexpected ParentContext direction");
  }

  // `ids` and `parent_ids` have a 1:1 mapping based on the database records.
  std::vector<int64_t> ids, parent_ids;
  ConvertToIdAndParentIds(record_set, ids, parent_ids);

  const bool is_parent = direction == ParentContextTraverseDirection::kParent;
  const absl::flat_hash_map<int64_t, std::vector<int64_t>>
      id_map_to_linked_ids = is_parent ? MapKeyIdToValueIds(ids, parent_ids)
                                       : MapKeyIdToValueIds(parent_ids, ids);
  if (id_map_to_linked_ids.empty()) {
    return absl::OkStatus();
  }

  std::vector<Context> linked_contexts;
  absl::Status result =
      FindNodesImpl(is_parent ? DedupIds(parent_ids) : DedupIds(ids),
                    /*skipped_ids_ok=*/false, linked_contexts);
  MLMD_RETURN_IF_ERROR(result);

  absl::flat_hash_map<int64_t, Context> id_map_to_linked_context;
  for (const Context& context : linked_contexts) {
    id_map_to_linked_context.insert(std::make_pair(context.id(), context));
  }

  output_contexts.clear();
  for (const auto& entry : id_map_to_linked_ids) {
    const int64_t id = entry.first;
    absl::Span<const int64_t> linked_ids = entry.second;
    std::vector<Context> contexts_per_id;
    contexts_per_id.reserve(linked_ids.size());
    for (const int64_t linked_id : linked_ids) {
      contexts_per_id.push_back(id_map_to_linked_context[linked_id]);
    }
    output_contexts.insert(std::make_pair(id, contexts_per_id));
  }
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::FindParentContextsByContextId(
    int64_t context_id, std::vector<Context>* contexts) {
  if (contexts == nullptr) {
    return absl::InvalidArgumentError("Given contexts is NULL.");
  }
  return FindLinkedContextsImpl(
      context_id, ParentContextTraverseDirection::kParent, *contexts);
}

absl::Status RDBMSMetadataAccessObject::FindChildContextsByContextId(
    int64_t context_id, std::vector<Context>* contexts) {
  if (contexts == nullptr) {
    return absl::InvalidArgumentError("Given contexts is NULL.");
  }
  return FindLinkedContextsImpl(
      context_id, ParentContextTraverseDirection::kChild, *contexts);
}

absl::Status RDBMSMetadataAccessObject::FindParentContextsByContextIds(
    absl::Span<const int64_t> context_ids,
    absl::node_hash_map<int64_t, std::vector<Context>>& contexts) {
  return FindLinkedContextsMapImpl(
      context_ids, ParentContextTraverseDirection::kParent, contexts);
}

absl::Status RDBMSMetadataAccessObject::FindChildContextsByContextIds(
    absl::Span<const int64_t> context_ids,
    absl::node_hash_map<int64_t, std::vector<Context>>& contexts) {
  return FindLinkedContextsMapImpl(
      context_ids, ParentContextTraverseDirection::kChild, contexts);
}

absl::Status RDBMSMetadataAccessObject::FindArtifacts(
    std::vector<Artifact>* artifacts) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectAllArtifactIDs(&record_set));
  std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *artifacts);
}

template <>
absl::Status RDBMSMetadataAccessObject::ListNodeIds<Artifact>(
    const ListOperationOptions& options,
    std::optional<absl::Span<const int64_t>> candidate_ids,
    RecordSet* record_set) {
  return executor_->ListArtifactIDsUsingOptions(options, candidate_ids,
                                                record_set);
}

template <>
absl::Status RDBMSMetadataAccessObject::ListNodeIds<Execution>(
    const ListOperationOptions& options,
    std::optional<absl::Span<const int64_t>> candidate_ids,
    RecordSet* record_set) {
  return executor_->ListExecutionIDsUsingOptions(options, candidate_ids,
                                                 record_set);
}

template <>
absl::Status RDBMSMetadataAccessObject::ListNodeIds<Context>(
    const ListOperationOptions& options,
    std::optional<absl::Span<const int64_t>> candidate_ids,
    RecordSet* record_set) {
  return executor_->ListContextIDsUsingOptions(options, candidate_ids,
                                               record_set);
}

template <typename Node>
absl::Status RDBMSMetadataAccessObject::ListNodes(
    const ListOperationOptions& options,
    std::optional<absl::Span<const int64_t>> candidate_ids,
    std::vector<Node>* nodes, std::string* next_page_token) {
  if (options.max_result_size() <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("max_result_size field value is required to be greater "
                     "than 0 and less than or equal to 100. Set value:",
                     options.max_result_size()));
  }
  if (!nodes->empty()) {
    return absl::InvalidArgumentError("nodes argument is not empty");
  }

  // Retrieving page of size 1 greater that max_result_size to detect if this
  // is the last page.
  ListOperationOptions updated_options = options;
  updated_options.set_max_result_size(options.max_result_size() + 1);
  // Retrieve ids based on the list options
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      ListNodeIds<Node>(updated_options, candidate_ids, &record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::OkStatus();
  }

  // Map node ids to positions
  absl::flat_hash_map<int64_t, size_t> position_by_id;
  for (int i = 0; i < ids.size(); ++i) {
    position_by_id[ids.at(i)] = i;
  }

  // Retrieve nodes
  MLMD_RETURN_IF_ERROR(FindNodesImpl(ids, /*skipped_ids_ok=*/false, *nodes));

  // Sort nodes in the right order
  absl::c_sort(*nodes, [&](const Node& a, const Node& b) {
    return position_by_id.at(a.id()) < position_by_id.at(b.id());
  });

  if (nodes->size() > options.max_result_size()) {
    // Removing the extra node retrieved for last page detection.
    nodes->pop_back();
    MLMD_RETURN_IF_ERROR(BuildListOperationNextPageToken<Node>(
        *nodes, options, next_page_token));
  } else {
    *next_page_token = "";
  }
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::ListArtifacts(
    const ListOperationOptions& options, std::vector<Artifact>* artifacts,
    std::string* next_page_token) {
  return ListNodes<Artifact>(options, absl::nullopt, artifacts,
                             next_page_token);
}

absl::Status RDBMSMetadataAccessObject::ListExecutions(
    const ListOperationOptions& options, std::vector<Execution>* executions,
    std::string* next_page_token) {
  return ListNodes<Execution>(options, absl::nullopt, executions,
                              next_page_token);
}

absl::Status RDBMSMetadataAccessObject::ListContexts(
    const ListOperationOptions& options, std::vector<Context>* contexts,
    std::string* next_page_token) {
  return ListNodes<Context>(options, absl::nullopt, contexts, next_page_token);
}

absl::Status RDBMSMetadataAccessObject::FindArtifactByTypeIdAndArtifactName(
    const int64_t type_id, absl::string_view name, Artifact* artifact) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectArtifactByTypeIDAndArtifactName(
      type_id, name, &record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "No artifacts found for type_id:", type_id, ", name:", name));
  }
  std::vector<Artifact> artifacts;
  MLMD_RETURN_IF_ERROR(FindNodesImpl(ids, /*skipped_ids_ok=*/false, artifacts));
  // By design, a <type_id, name> pair uniquely identifies an artifact.
  // Fails if multiple artifacts are found.
  CHECK_EQ(artifacts.size(), 1)
      << absl::StrCat("Found more than one artifact with type_id: ", type_id,
                      " and artifact name: ", name);
  *artifact = artifacts[0];
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::FindArtifactsByTypeId(
    const int64_t type_id, std::optional<ListOperationOptions> list_options,
    std::vector<Artifact>* artifacts, std::string* next_page_token) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectArtifactsByTypeID(type_id, &record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No artifacts found for type_id:", type_id));
  }
  if (list_options) {
    return ListNodes<Artifact>(list_options.value(), ids, artifacts,
                               next_page_token);
  } else {
    return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *artifacts);
  }
}

absl::Status RDBMSMetadataAccessObject::FindExecutions(
    std::vector<Execution>* executions) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectAllExecutionIDs(&record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *executions);
}

absl::Status RDBMSMetadataAccessObject::FindExecutionByTypeIdAndExecutionName(
    const int64_t type_id, absl::string_view name, Execution* execution) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectExecutionByTypeIDAndExecutionName(
      type_id, name, &record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "No executions found for type_id:", type_id, ", name:", name));
  }
  std::vector<Execution> executions;
  MLMD_RETURN_IF_ERROR(
      FindNodesImpl(ids, /*skipped_ids_ok=*/false, executions));
  // By design, a <type_id, name> pair uniquely identifies an execution.
  // Fails if multiple executions are found.
  CHECK_EQ(executions.size(), 1)
      << absl::StrCat("Found more than one execution with type_id: ", type_id,
                      " and execution name: ", name);
  *execution = executions[0];
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::FindExecutionsByTypeId(
    const int64_t type_id, std::optional<ListOperationOptions> list_options,
    std::vector<Execution>* executions, std::string* next_page_token) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectExecutionsByTypeID(type_id, &record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No executions found for type_id:", type_id));
  }
  if (list_options) {
    return ListNodes<Execution>(list_options.value(), ids, executions,
                                next_page_token);
  } else {
    return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *executions);
  }
}

absl::Status RDBMSMetadataAccessObject::FindContexts(
    std::vector<Context>* contexts) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectAllContextIDs(&record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *contexts);
}

absl::Status RDBMSMetadataAccessObject::FindContextsByTypeId(
    const int64_t type_id, std::optional<ListOperationOptions> list_options,
    std::vector<Context>* contexts, std::string* next_page_token) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectContextsByTypeID(type_id, &record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No contexts found with type_id: ", type_id));
  }

  if (list_options) {
    return ListNodes<Context>(list_options.value(), ids, contexts,
                              next_page_token);
  } else {
    return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *contexts);
  }
}

absl::Status RDBMSMetadataAccessObject::FindArtifactsByURI(
    absl::string_view uri, std::vector<Artifact>* artifacts) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectArtifactsByURI(uri, &record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No artifacts found for uri:", uri));
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *artifacts);
}

absl::Status RDBMSMetadataAccessObject::FindContextByTypeIdAndContextName(
    int64_t type_id, absl::string_view name, bool id_only, Context* context) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectContextByTypeIDAndContextName(
      type_id, name, &record_set));
  const std::vector<int64_t> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "No contexts found with type_id: ", type_id, ", name: ", name));
  }
  // By design, a <type_id, name> pair uniquely identifies a context.
  // Fails if multiple contexts are found.
  CHECK_EQ(ids.size(), 1) << absl::StrCat(
      "Found more than one contexts with type_id: ", type_id,
      " and context name: ", name);
  if (id_only) {
    context->set_id(ids[0]);
    return absl::OkStatus();
  }

  std::vector<Context> contexts;
  MLMD_RETURN_IF_ERROR(FindNodesImpl(ids, /*skipped_ids_ok=*/false, contexts));
  *context = contexts[0];
  return absl::OkStatus();
}


template <typename Node>
absl::Status RDBMSMetadataAccessObject::FilterBoundaryNodesImpl(
    std::optional<absl::string_view> node_filter,
    absl::flat_hash_set<int64_t>& boundary_node_ids) {
  if (!node_filter.has_value()) {
    return absl::OkStatus();
  }
  const std::vector<int64_t> candidate_ids(boundary_node_ids.begin(),
                                           boundary_node_ids.end());
  auto list_ids = absl::MakeConstSpan(candidate_ids);
  // Uses batched retrieval to bound query length and list query invariant.
  static constexpr int kBatchSize = 100;
  boundary_node_ids.clear();
  for (int i = 0; i * kBatchSize < candidate_ids.size(); i++) {
    ListOperationOptions boundary_options;
    boundary_options.set_max_result_size(kBatchSize);
    boundary_options.set_filter_query(node_filter.value().data());
    RecordSet record_set;
    MLMD_RETURN_IF_ERROR(ListNodeIds<Node>(
        boundary_options, list_ids.subspan(i * kBatchSize, kBatchSize),
        &record_set));
    for (int64_t keep_id : ConvertToIds(record_set)) {
      boundary_node_ids.insert(keep_id);
    }
  }

  return absl::OkStatus();
}

template <typename Node>
absl::StatusOr<absl::flat_hash_set<int64_t>>
RDBMSMetadataAccessObject::FindEndingNodeIdsIfExists(
    const LineageSubgraphQueryOptions::EndingNodes ending_nodes,
    const absl::flat_hash_set<int64_t>& unvisited_node_ids) {
  absl::flat_hash_set<int64_t> ending_node_ids;
  if (!ending_nodes.has_filter_query()) {
    return ending_node_ids;
  }
  const std::vector<int64_t> candidate_ids(unvisited_node_ids.begin(),
                                           unvisited_node_ids.end());
  auto list_ids = absl::MakeConstSpan(candidate_ids);
  // Uses batched retrieval to bound query length and list query invariant.
  int64_t batch_size = kDefaultMaxListOperationResultSize;
  for (int offset = 0; offset < candidate_ids.size(); offset += batch_size) {
    ListOperationOptions boundary_options;
    boundary_options.set_max_result_size(batch_size);
    boundary_options.set_filter_query(ending_nodes.filter_query().data());
    RecordSet record_set;
    MLMD_RETURN_IF_ERROR(ListNodeIds<Node>(
        boundary_options, list_ids.subspan(offset, batch_size), &record_set));
    for (int64_t ending_node_id : ConvertToIds(record_set)) {
      ending_node_ids.insert(ending_node_id);
    }
  }

  return ending_node_ids;
}

absl::Status RDBMSMetadataAccessObject::ExpandLineageGraphImpl(
    const std::vector<Artifact>& input_artifacts, int64_t max_nodes,
    std::optional<std::string> boundary_condition,
    const absl::flat_hash_set<int64_t>& visited_execution_ids,
    absl::flat_hash_set<int64_t>& visited_artifact_ids,
    std::vector<Execution>& output_executions, LineageGraph& subgraph) {
  if (max_nodes <= 0) {
    return absl::OkStatus();
  }
  std::vector<int64_t> input_artifact_ids(input_artifacts.size());
  for (int i = 0; i < input_artifacts.size(); i++) {
    input_artifact_ids[i] = input_artifacts[i].id();
    visited_artifact_ids.insert(input_artifact_ids[i]);
  }
  std::vector<Event> events;
  const auto status = FindEventsByArtifacts(input_artifact_ids, &events);
  // If no events are found for the given artifacts, directly return ok
  // status.
  if (absl::IsNotFound(status)) {
    return absl::OkStatus();
  }
  absl::flat_hash_set<int64_t> unvisited_execution_ids;
  for (const Event& event : events) {
    if (!visited_execution_ids.contains(event.execution_id())) {
      unvisited_execution_ids.insert(event.execution_id());
    }
  }
  MLMD_RETURN_IF_ERROR(FilterBoundaryNodesImpl<Execution>(
      boundary_condition, unvisited_execution_ids));

  // Randomly remove extra nodes if more than max_nodes executions are found.
  while (unvisited_execution_ids.size() > max_nodes) {
    unvisited_execution_ids.erase(unvisited_execution_ids.begin());
  }

  for (const Event& event : events) {
    if (unvisited_execution_ids.contains(event.execution_id())) {
      *subgraph.add_events() = event;
    }
  }
  const std::vector<int64_t> expand_execution_ids(
      unvisited_execution_ids.begin(), unvisited_execution_ids.end());
  output_executions.clear();
  MLMD_RETURN_IF_ERROR(
      FindExecutionsById(expand_execution_ids, &output_executions));
  absl::c_copy(output_executions, google::protobuf::RepeatedFieldBackInserter(
                                      subgraph.mutable_executions()));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int64_t>>
RDBMSMetadataAccessObject::ExpandLineageSubgraphImpl(
    const bool expand_from_artifacts,
    const LineageSubgraphQueryOptions& options,
    absl::Span<const int64_t> input_node_ids,
    absl::flat_hash_set<int64_t>& visited_output_node_ids,
    absl::flat_hash_set<int64_t>& output_ending_node_ids,
    std::vector<Event>& output_events) {
  // Step 1: filter events by direction.
  std::vector<Event> candidate_events;
  absl::Status status =
      expand_from_artifacts
          ? FindEventsByArtifacts(input_node_ids, &candidate_events)
          : FindEventsByExecutions(input_node_ids, &candidate_events);
  std::vector<Event> events = FilterEventsByDirectionAndEventType(
      candidate_events, /*is_from_artifact=*/expand_from_artifacts,
      options.direction());

  std::vector<int64_t> output_node_ids;
  // If no events are found for the given input nodes, directly return.
  if (absl::IsNotFound(status)) {
    return output_node_ids;
  }
  MLMD_RETURN_IF_ERROR(status);

  // Step 2: collect the new node IDs to visit from filtered events.
  absl::flat_hash_set<int64_t> unvisited_output_node_ids;
  for (const Event& event : events) {
    int64_t output_node_id =
        expand_from_artifacts ? event.execution_id() : event.artifact_id();
    // Filter nodes by visited node ids and ending node ids collected so far in
    // the previous graph expansions.
    if (!visited_output_node_ids.contains(output_node_id) &&
        !output_ending_node_ids.contains(output_node_id)) {
      unvisited_output_node_ids.insert(output_node_id);
    }
  }
  // Step 3: determine if node IDs are ending nodes and exclude them from
  // further expansion.
  absl::flat_hash_set<int64_t> ending_node_ids;
  MLMD_ASSIGN_OR_RETURN(
      ending_node_ids,
      expand_from_artifacts
          ? FindEndingNodeIdsIfExists<Execution>(options.ending_executions(),
                                                 unvisited_output_node_ids)
          : FindEndingNodeIdsIfExists<Artifact>(options.ending_artifacts(),
                                                unvisited_output_node_ids));
  absl::c_copy(ending_node_ids, std::inserter(output_ending_node_ids,
                                              output_ending_node_ids.end()));
  // Step 4: Filter unvisited_output_node_ids if ending nodes exist.
  absl::erase_if(unvisited_output_node_ids,
                 [&output_ending_node_ids](int64_t id) {
                   return output_ending_node_ids.contains(id);
                 });

  // Step 5: Filter events by visited nodes and ending nodes if possible.
  for (const Event& event : events) {
    int64_t output_node_id =
        expand_from_artifacts ? event.execution_id() : event.artifact_id();
    if (!visited_output_node_ids.contains(output_node_id) &&
        !output_ending_node_ids.contains(output_node_id)) {
      output_events.push_back(event);
    } else if (output_ending_node_ids.contains(output_node_id)) {
      if ((expand_from_artifacts &&
           options.ending_executions().include_ending_nodes()) ||
          (!expand_from_artifacts &&
           options.ending_artifacts().include_ending_nodes())) {
        output_events.push_back(event);
      }
    } else if (visited_output_node_ids.contains(output_node_id)) {
      if (options.direction() == LineageSubgraphQueryOptions::UPSTREAM ||
          options.direction() == LineageSubgraphQueryOptions::DOWNSTREAM) {
        // For directional bfs, we should still add the edge even a visited
        // output node is encountered. For example, there are two paths:
        //   a1 -> input_event1 -> e1 -> output_event1 -> a2 -> input_event2 ->
        //   e2 a1 -> input_event3 -> e2
        // Imagine when the downstream graph tracing starts from a1 with hops
        // = 3. In the last hop, although e2 is already visited, input_event2
        // still counts as a valid edge that should be included into
        // `output_events`. For bidirectional bfs, we skip adding those events
        // to avoid duplicate events which can only happen in this case.
        output_events.push_back(event);
      }
    }
  }

  absl::c_copy(unvisited_output_node_ids, std::back_inserter(output_node_ids));
  visited_output_node_ids.insert(output_node_ids.begin(),
                                 output_node_ids.end());
  return output_node_ids;
}

absl::Status RDBMSMetadataAccessObject::ExpandLineageGraphImpl(
    const std::vector<Execution>& input_executions, int64_t max_nodes,
    std::optional<std::string> boundary_condition,
    const absl::flat_hash_set<int64_t>& visited_artifact_ids,
    absl::flat_hash_set<int64_t>& visited_execution_ids,
    std::vector<Artifact>& output_artifacts, LineageGraph& subgraph) {
  if (max_nodes <= 0) {
    return absl::OkStatus();
  }
  std::vector<int64_t> input_execution_ids(input_executions.size());
  for (int i = 0; i < input_executions.size(); i++) {
    input_execution_ids[i] = input_executions[i].id();
    visited_execution_ids.insert(input_execution_ids[i]);
  }
  std::vector<Event> events;
  const auto status = FindEventsByExecutions(input_execution_ids, &events);
  // If no events are found for the given executions, directly return ok
  // status.
  if (absl::IsNotFound(status)) {
    return absl::OkStatus();
  }
  absl::flat_hash_set<int64_t> unvisited_artifact_ids;
  for (const Event& event : events) {
    if (!visited_artifact_ids.contains(event.artifact_id())) {
      unvisited_artifact_ids.insert(event.artifact_id());
    }
  }
  MLMD_RETURN_IF_ERROR(FilterBoundaryNodesImpl<Artifact>(
      boundary_condition, unvisited_artifact_ids));

  // Randomly remove extra nodes if more than max_nodes artifacts are found.
  while (unvisited_artifact_ids.size() > max_nodes) {
    unvisited_artifact_ids.erase(unvisited_artifact_ids.begin());
  }

  for (const Event& event : events) {
    if (unvisited_artifact_ids.contains(event.artifact_id())) {
      *subgraph.add_events() = event;
    }
  }
  const std::vector<int64_t> expand_artifact_ids(unvisited_artifact_ids.begin(),
                                                 unvisited_artifact_ids.end());
  output_artifacts.clear();
  MLMD_RETURN_IF_ERROR(
      FindArtifactsById(expand_artifact_ids, &output_artifacts));
  absl::c_copy(output_artifacts,
               google::protobuf::RepeatedFieldBackInserter(subgraph.mutable_artifacts()));
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::QueryLineageGraph(
    const std::vector<Artifact>& query_nodes, int64_t max_num_hops,
    std::optional<int64_t> max_nodes,
    std::optional<std::string> boundary_artifacts,
    std::optional<std::string> boundary_executions, LineageGraph& subgraph) {
  absl::c_copy(query_nodes,
               google::protobuf::RepeatedFieldBackInserter(subgraph.mutable_artifacts()));
  // Add nodes and edges
  absl::flat_hash_set<int64_t> visited_artifacts_ids;
  absl::flat_hash_set<int64_t> visited_executions_ids;
  int64_t curr_distance = 0;
  std::vector<Artifact> output_artifacts;
  std::vector<Execution> output_executions;
  // If max_nodes is not set, set nodes quota to max int64_t value to
  // effectively disable the nodes count limit of the lineage graph.
  int64_t nodes_quota;
  if (!max_nodes) {
    nodes_quota = std::numeric_limits<int64_t>::max();
  } else {
    nodes_quota = max_nodes.value() - query_nodes.size();
  }

  while (curr_distance < max_num_hops && nodes_quota > 0) {
    const bool is_traverse_from_artifact = (curr_distance % 2 == 0);
    if (is_traverse_from_artifact) {
      if (curr_distance == 0) {
        MLMD_RETURN_IF_ERROR(ExpandLineageGraphImpl(
            query_nodes, nodes_quota, boundary_executions,
            visited_executions_ids, visited_artifacts_ids, output_executions,
            subgraph));
      } else {
        MLMD_RETURN_IF_ERROR(ExpandLineageGraphImpl(
            output_artifacts, nodes_quota, boundary_executions,
            visited_executions_ids, visited_artifacts_ids, output_executions,
            subgraph));
      }
      if (output_executions.empty()) {
        break;
      }
      nodes_quota -= output_executions.size();
    } else {
      MLMD_RETURN_IF_ERROR(ExpandLineageGraphImpl(
          output_executions, nodes_quota, boundary_artifacts,
          visited_artifacts_ids, visited_executions_ids, output_artifacts,
          subgraph));
      if (output_artifacts.empty()) {
        break;
      }
      nodes_quota -= output_artifacts.size();
    }
    curr_distance++;
  }
  // Add node types.
  std::vector<ArtifactType> artifact_types;
  MLMD_RETURN_IF_ERROR(FindTypes(&artifact_types));
  for (const ArtifactType& artifact_type : artifact_types) {
    const bool is_simple_type =
        std::find(kSimpleTypeNames.begin(), kSimpleTypeNames.end(),
                  artifact_type.name()) != kSimpleTypeNames.end();
    if (!is_simple_type) {
      *subgraph.mutable_artifact_types()->Add() = artifact_type;
    }
  }
  std::vector<ExecutionType> execution_types;
  MLMD_RETURN_IF_ERROR(FindTypes(&execution_types));
  for (const ExecutionType& execution_type : execution_types) {
    const bool is_simple_type =
        std::find(kSimpleTypeNames.begin(), kSimpleTypeNames.end(),
                  execution_type.name()) != kSimpleTypeNames.end();
    if (!is_simple_type) {
      *subgraph.mutable_execution_types()->Add() = execution_type;
    }
  }
  std::vector<ContextType> context_types;
  MLMD_RETURN_IF_ERROR(FindTypes(&context_types));
  absl::c_copy(context_types, google::protobuf::RepeatedFieldBackInserter(
                                  subgraph.mutable_context_types()));
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::QueryLineageSubgraph(
    const LineageSubgraphQueryOptions& lineage_subgraph_query_options,
    const google::protobuf::FieldMask& read_mask, LineageGraph& subgraph) {
  if (read_mask.paths().empty()) {
    return absl::InvalidArgumentError(
        "Cannot execute QueryLineageSubgraph when `read_mask` is empty.");
  }
  // Perform precondition check for `starting_nodes` and `max_num_hops`.
  std::string starting_nodes_filter_query;
  bool is_from_artifacts;

  if (lineage_subgraph_query_options.has_starting_artifacts()) {
    starting_nodes_filter_query =
        lineage_subgraph_query_options.starting_artifacts().filter_query();
    is_from_artifacts = true;
  } else if (lineage_subgraph_query_options.has_starting_executions()) {
    starting_nodes_filter_query =
        lineage_subgraph_query_options.starting_executions().filter_query();
    is_from_artifacts = false;
  } else {
    return absl::InvalidArgumentError(
        "Missing arguments for listing starting nodes.");
  }

  if (starting_nodes_filter_query.empty()) {
    return absl::InvalidArgumentError(
        "Cannot list starting nodes if `filter_query` is unspecified.");
  }

  int64_t max_num_hops = kQueryLineageSubgraphMaxNumHops;
  if (!lineage_subgraph_query_options.has_max_num_hops()) {
    LOG(INFO) << "max_num_hops is not set. Using maximum value: "
              << max_num_hops << " to limit the steps of the traversal.";
  } else {
    max_num_hops = lineage_subgraph_query_options.max_num_hops();
    if (max_num_hops < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("max_num_hops cannot be negative: max_num_hops = ",
                       lineage_subgraph_query_options.max_num_hops()));
    }
    if (max_num_hops > kQueryLineageSubgraphMaxNumHops) {
      LOG(INFO) << "The value of max_num_hops, "
                << lineage_subgraph_query_options.max_num_hops()
                << ", is too large. Using the max allowed value, "
                << kQueryLineageSubgraphMaxNumHops
                << ", instead to limit the steps of the traversal.";
      max_num_hops = kQueryLineageSubgraphMaxNumHops;
    }
  }

  // Get ordered ids of starting nodes based on `starting_nodes_filter_query`.
  ListOperationOptions starting_nodes_options;
  starting_nodes_options.set_filter_query(starting_nodes_filter_query);
  // Set max_result_size to the maximum value of int32_t.
  // TODO: Remove the default value of 20 in `ListOperationOptions` proto.
  // TODO(b/283852485): Investigate whether to increase the 100 limitation of
  // starting nodes.
  starting_nodes_options.set_max_result_size(
      std::numeric_limits<int32_t>::max());

  RecordSet record_set;
  if (is_from_artifacts) {
    MLMD_RETURN_IF_ERROR(ListNodeIds<Artifact>(starting_nodes_options,
                                               absl::nullopt, &record_set));
  } else {
    MLMD_RETURN_IF_ERROR(ListNodeIds<Execution>(starting_nodes_options,
                                                absl::nullopt, &record_set));
  }
  if (record_set.records_size() == 0) {
    return absl::NotFoundError(
        "Cannot find qualified nodes to start the traversal.");
  }

  // Start expanding from the starting artifacts.
  std::vector<int64_t> output_artifact_ids;
  std::vector<int64_t> output_execution_ids;
  absl::flat_hash_set<int64_t> visited_artifacts_ids;
  absl::flat_hash_set<int64_t> visited_executions_ids;
  std::vector<Event> visited_events;
  absl::flat_hash_set<int64_t> ending_artifact_ids;
  absl::flat_hash_set<int64_t> ending_execution_ids;

  if (is_from_artifacts) {
    absl::c_copy(ConvertToIds(record_set),
                 std::back_inserter(output_artifact_ids));
    absl::c_copy(
        output_artifact_ids,
        std::inserter(visited_artifacts_ids, visited_artifacts_ids.end()));
    MLMD_ASSIGN_OR_RETURN(ending_artifact_ids,
                          FindEndingNodeIdsIfExists<Artifact>(
                              lineage_subgraph_query_options.ending_artifacts(),
                              visited_artifacts_ids));
    output_artifact_ids.erase(
        std::remove_if(
            output_artifact_ids.begin(), output_artifact_ids.end(),
            [&](int64_t id) { return ending_artifact_ids.contains(id); }),
        output_artifact_ids.end());

    absl::erase_if(visited_artifacts_ids, [&](int64_t artifact_id) {
      return ending_artifact_ids.contains(artifact_id);
    });

  } else {
    absl::c_copy(ConvertToIds(record_set),
                 std::back_inserter(output_execution_ids));
    absl::c_copy(
        output_execution_ids,
        std::inserter(visited_executions_ids, visited_executions_ids.end()));
    MLMD_ASSIGN_OR_RETURN(
        ending_execution_ids,
        FindEndingNodeIdsIfExists<Execution>(
            lineage_subgraph_query_options.ending_executions(),
            visited_executions_ids));
    output_execution_ids.erase(
        std::remove_if(
            output_execution_ids.begin(), output_execution_ids.end(),
            [&](int64_t id) { return ending_execution_ids.contains(id); }),
        output_execution_ids.end());

    absl::erase_if(visited_executions_ids, [&](int64_t execution_id) {
      return ending_execution_ids.contains(execution_id);
    });
  }

  int64_t curr_distance = 0;

  while (curr_distance < max_num_hops) {
    // If `is_from_artifacts` is true, expand from Artifacts to Executions if
    // `curr_distance` is even, vice versa.
    // If `is_from_artifacts` is false, expand from Executions to Artifacts if
    // `curr_distance` is even, vice versa.
    const bool expand_from_artifacts =
        (is_from_artifacts == (curr_distance % 2 == 0));
    if (expand_from_artifacts) {
      MLMD_ASSIGN_OR_RETURN(
          output_execution_ids,
          ExpandLineageSubgraphImpl(
              /*expand_from_artifacts=*/expand_from_artifacts,
              /*options=*/lineage_subgraph_query_options,
              /*input_node_ids=*/output_artifact_ids,
              /*visited_output_node_ids=*/visited_executions_ids,
              /*output_ending_node_ids=*/ending_execution_ids, visited_events));
      if (output_execution_ids.empty()) {
        break;
      }
    } else {
      MLMD_ASSIGN_OR_RETURN(
          output_artifact_ids,
          ExpandLineageSubgraphImpl(
              /*expand_from_artifacts=*/expand_from_artifacts,
              /*options=*/lineage_subgraph_query_options,
              /*input_node_ids=*/output_execution_ids,
              /*visited_output_node_ids=*/visited_artifacts_ids,
              /*output_ending_node_ids=*/ending_artifact_ids, visited_events));
      if (output_artifact_ids.empty()) {
        break;
      }
    }
    curr_distance++;
  }

  absl::flat_hash_set<std::string> field_mask_paths;
  absl::c_copy(read_mask.paths(),
               std::inserter(field_mask_paths, field_mask_paths.end()));
  std::vector<int64_t> artifact_ids(visited_artifacts_ids.begin(),
                                    visited_artifacts_ids.end());
  std::vector<int64_t> execution_ids(visited_executions_ids.begin(),
                                     visited_executions_ids.end());
  // Append ending nodes to return results if possible.
  if (lineage_subgraph_query_options.ending_artifacts()
          .include_ending_nodes()) {
    artifact_ids.insert(artifact_ids.end(), ending_artifact_ids.begin(),
                        ending_artifact_ids.end());
  }
  if (lineage_subgraph_query_options.ending_executions()
          .include_ending_nodes()) {
    execution_ids.insert(execution_ids.end(), ending_execution_ids.begin(),
                         ending_execution_ids.end());
  }
  if (field_mask_paths.contains("artifacts")) {
    for (int64_t artifact_id : artifact_ids) {
      subgraph.add_artifacts()->set_id(artifact_id);
    }
  }
  if (field_mask_paths.contains("executions")) {
    for (int64_t execution_id : execution_ids) {
      subgraph.add_executions()->set_id(execution_id);
    }
  }
  if (field_mask_paths.contains("contexts")) {
    std::vector<int64_t> context_ids;
    MLMD_ASSIGN_OR_RETURN(context_ids, FindContextIdsByArtifactsAndExecutions(
                                           artifact_ids, execution_ids));
    for (int64_t context_id : context_ids) {
      subgraph.add_contexts()->set_id(context_id);
    }
  }
  if (field_mask_paths.contains("events")) {
    absl::c_copy(visited_events,
                 google::protobuf::RepeatedFieldBackInserter(subgraph.mutable_events()));
  }
  if (field_mask_paths.contains("attributions")) {
    std::vector<Attribution> attributions;
    MLMD_RETURN_IF_ERROR(
        FindAttributionsByArtifacts(artifact_ids, &attributions));
    absl::c_copy(attributions, google::protobuf::RepeatedFieldBackInserter(
                                   subgraph.mutable_attributions()));
  }
  if (field_mask_paths.contains("associations")) {
    std::vector<Association> associations;
    MLMD_RETURN_IF_ERROR(
        FindAssociationsByExecutions(execution_ids, &associations));
    absl::c_copy(associations, google::protobuf::RepeatedFieldBackInserter(
                                   subgraph.mutable_associations()));
  }
  if (field_mask_paths.contains("artifact_types")) {
    std::vector<ArtifactType> artifact_types;
    MLMD_RETURN_IF_ERROR(FindTypes(&artifact_types));
    absl::c_copy(artifact_types, google::protobuf::RepeatedFieldBackInserter(
                                     subgraph.mutable_artifact_types()));
  }
  if (field_mask_paths.contains("execution_types")) {
    std::vector<ExecutionType> execution_types;
    MLMD_RETURN_IF_ERROR(FindTypes(&execution_types));
    absl::c_copy(execution_types, google::protobuf::RepeatedFieldBackInserter(
                                      subgraph.mutable_execution_types()));
  }
  if (field_mask_paths.contains("context_types")) {
    std::vector<ContextType> context_types;
    MLMD_RETURN_IF_ERROR(FindTypes(&context_types));
    absl::c_copy(context_types, google::protobuf::RepeatedFieldBackInserter(
                                    subgraph.mutable_context_types()));
  }
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::DeleteArtifactsById(
    absl::Span<const int64_t> artifact_ids) {
  if (artifact_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteArtifactsById(artifact_ids);
}

absl::Status RDBMSMetadataAccessObject::DeleteExecutionsById(
    absl::Span<const int64_t> execution_ids) {
  if (execution_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteExecutionsById(execution_ids);
}

absl::Status RDBMSMetadataAccessObject::DeleteContextsById(
    absl::Span<const int64_t> context_ids) {
  if (context_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteContextsById(context_ids);
}

absl::Status RDBMSMetadataAccessObject::DeleteEventsByArtifactsId(
    absl::Span<const int64_t> artifact_ids) {
  if (artifact_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteEventsByArtifactsId(artifact_ids);
}

absl::Status RDBMSMetadataAccessObject::DeleteEventsByExecutionsId(
    absl::Span<const int64_t> execution_ids) {
  if (execution_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteEventsByExecutionsId(execution_ids);
}

absl::Status RDBMSMetadataAccessObject::DeleteAssociationsByContextsId(
    absl::Span<const int64_t> context_ids) {
  if (context_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteAssociationsByContextsId(context_ids);
}

absl::Status RDBMSMetadataAccessObject::DeleteAssociationsByExecutionsId(
    absl::Span<const int64_t> execution_ids) {
  if (execution_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteAssociationsByExecutionsId(execution_ids);
}

absl::Status RDBMSMetadataAccessObject::DeleteAttributionsByContextsId(
    absl::Span<const int64_t> context_ids) {
  if (context_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteAttributionsByContextsId(context_ids);
}

absl::Status RDBMSMetadataAccessObject::DeleteAttributionsByArtifactsId(
    absl::Span<const int64_t> artifact_ids) {
  if (artifact_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteAttributionsByArtifactsId(artifact_ids);
}

absl::Status RDBMSMetadataAccessObject::DeleteParentContextsByParentIds(
    absl::Span<const int64_t> parent_context_ids) {
  if (parent_context_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteParentContextsByParentIds(parent_context_ids);
}

absl::Status RDBMSMetadataAccessObject::DeleteParentContextsByChildIds(
    absl::Span<const int64_t> child_context_ids) {
  if (child_context_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteParentContextsByChildIds(child_context_ids);
}

absl::Status
RDBMSMetadataAccessObject::DeleteParentContextsByParentIdAndChildIds(
    int64_t parent_context_id, absl::Span<const int64_t> child_context_ids) {
  if (child_context_ids.empty()) {
    return absl::OkStatus();
  }
  return executor_->DeleteParentContextsByParentIdAndChildIds(
      parent_context_id, child_context_ids);
}

}  // namespace ml_metadata
