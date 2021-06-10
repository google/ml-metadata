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
#ifndef _WIN32
#include "ml_metadata/metadata_store/rdbms_metadata_access_object.h"

#endif

#include <string>
#include <vector>

#include <glog/logging.h>
#include "google/protobuf/struct.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/message_differencer.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/strip.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
// clang-format off
#ifdef _WIN32
#include "ml_metadata/metadata_store/rdbms_metadata_access_object.h" // NOLINT
#endif
// clang-format on
#include "absl/types/optional.h"
#include "ml_metadata/metadata_store/list_operation_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/simple_types/simple_types_constants.h"
#include "ml_metadata/util/return_utils.h"
#include "ml_metadata/util/struct_utils.h"

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

// Populates 'node' properties from the rows in 'record'. The assumption is that
// properties are encoded using the convention in
// QueryExecutor::Get{X}PropertyBy{X}Id() where X in {Artifact, Execution,
// Context}.
template <typename Node>
absl::Status PopulateNodeProperties(const RecordSet::Record& record,
                                    Node& node) {
  // Populate the property of the node.
  const std::string& property_name = record.values(1);
  bool is_custom_property;
  CHECK(absl::SimpleAtob(record.values(2), &is_custom_property));
  auto& property_value =
      (is_custom_property ? (*node.mutable_custom_properties())[property_name]
                          : (*node.mutable_properties())[property_name]);
  if (record.values(3) != kMetadataSourceNull) {
    int64 int_value;
    CHECK(absl::SimpleAtoi(record.values(3), &int_value));
    property_value.set_int_value(int_value);
  } else if (record.values(4) != kMetadataSourceNull) {
    double double_value;
    CHECK(absl::SimpleAtod(record.values(4), &double_value));
    property_value.set_double_value(double_value);
  } else {
    const std::string& string_value = record.values(5);
    if (IsStructSerializedString(string_value)) {
      MLMD_RETURN_IF_ERROR(
          StringToStruct(string_value, *property_value.mutable_struct_value()));
    } else {
      property_value.set_string_value(string_value);
    }
  }

  return absl::OkStatus();
}

// Converts a record set that contains an id column at position per record to a
// vector.
std::vector<int64> ConvertToIds(const RecordSet& record_set, int position = 0) {
  std::vector<int64> result;
  result.reserve(record_set.records().size());
  for (const RecordSet::Record& record : record_set.records()) {
    int64 id;
    CHECK(absl::SimpleAtoi(record.values(position), &id));
    result.push_back(id);
  }
  return result;
}

// Extracts a vector of type ids from the parent_type triplets.
std::vector<int64> ParentTypesToParentTypeIds(const RecordSet& record_set) {
  return ConvertToIds(record_set, /*position=*/1);
}

// Extracts a vector of context ids from attribution triplets.
std::vector<int64> AttributionsToContextIds(const RecordSet& record_set) {
  return ConvertToIds(record_set, /*position=*/1);
}

// Extracts a vector of context ids from attribution triplets.
std::vector<int64> AttributionsToArtifactIds(const RecordSet& record_set) {
  return ConvertToIds(record_set, /*position=*/2);
}

// Extracts a vector of context ids from association triplets.
std::vector<int64> AssociationsToContextIds(const RecordSet& record_set) {
  return ConvertToIds(record_set, /*position=*/1);
}

// Extracts a vector of execution ids from association triplets.
std::vector<int64> AssociationsToExecutionIds(const RecordSet& record_set) {
  return ConvertToIds(record_set, /*position=*/2);
}

// Extracts a vector of parent context ids from parent context triplets.
// If is_parent is true, then parent_context_ids are returned.
// If is_parent is false, then context_ids for children are returned.
std::vector<int64> ParentContextsToContextIds(const RecordSet& record_set,
                                              bool is_parent) {
  const int position = is_parent ? 1 : 0;
  return ConvertToIds(record_set, position);
}

// Parses and converts a string value to a specific field in a message.
// If the given string `value` is NULL (encoded as kMetadataSourceNull), then
// leave the field unset.
// The field should be a scalar field. The field type must be one of {string,
// int64, bool, enum, message}.
absl::Status ParseValueToField(const google::protobuf::FieldDescriptor* field_descriptor,
                               const absl::string_view value,
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
      int64 int64_value;
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

// Converts a RecordSet in the query result to a MessageType array.
template <typename MessageType>
absl::Status ParseRecordSetToMessageArray(const RecordSet& record_set,
                                          std::vector<MessageType>* messages) {
  for (int i = 0; i < record_set.records_size(); i++) {
    messages->push_back(MessageType());
    MLMD_RETURN_IF_ERROR(
        ParseRecordSetToMessage(record_set, &messages->back(), i));
  }
  return absl::OkStatus();
}

// Converts a RecordSet containing key-value pairs to a proto Map.
// The field_name is the map field in the MessageType. The method fills the
// message's map field with field_name using the rows in the given record_set.
template <typename MessageType>
absl::Status ParseRecordSetToMapField(const RecordSet& record_set,
                                      const std::string& field_name,
                                      MessageType* message) {
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

  for (const RecordSet::Record& record : record_set.records()) {
    google::protobuf::Message* map_field_message =
        reflection->AddMessage(message, map_field_descriptor);
    const std::string& key = record.values(0);
    const std::string& value = record.values(1);
    MLMD_RETURN_IF_ERROR(
        ParseValueToField(key_descriptor, key, map_field_message));
    MLMD_RETURN_IF_ERROR(
        ParseValueToField(value_descriptor, value, map_field_message));
  }

  return absl::OkStatus();
}

// Validates properties in a `Node` with the properties defined in a `Type`.
// `Node` is one of {`Artifact`, `Execution`, `Context`}. `Type` is one of
// {`ArtifactType`, `ExecutionType`, `ContextType`}.
// Returns INVALID_ARGUMENT error, if there is unknown or mismatched property
// w.r.t. its definition.
template <typename Node, typename Type>
absl::Status ValidatePropertiesWithType(const Node& node, const Type& type) {
  const google::protobuf::Map<std::string, PropertyType>& type_properties =
      type.properties();
  for (const auto& p : node.properties()) {
    const std::string& property_name = p.first;
    const Value& property_value = p.second;
    // Note that this is a google::protobuf::Map, not a std::map.
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

// Casts lower level db error messages for violating primary key constraint or
// unique constraint. Returns true when the status error message indicates such
// unique constraint is violated.
bool IsUniqueConstraintViolated(const absl::Status status) {
  return absl::IsInternal(status) &&
         (absl::StrContains(std::string(status.message()), "Duplicate") ||
          absl::StrContains(std::string(status.message()), "UNIQUE"));
}

// A util to handle `version` in ArtifactType/ExecutionType/ContextType protos.
template <typename T>
absl::optional<std::string> GetTypeVersion(const T& type_message) {
  return type_message.has_version() && !type_message.version().empty()
             ? absl::make_optional(type_message.version())
             : absl::nullopt;
}

}  // namespace

// Creates an Artifact (without properties).
absl::Status RDBMSMetadataAccessObject::CreateBasicNode(
    const Artifact& artifact, int64* node_id) {
  const absl::Time now = absl::Now();
  return executor_->InsertArtifact(
      artifact.type_id(), artifact.uri(),
      artifact.has_state() ? absl::make_optional(artifact.state())
                           : absl::nullopt,
      artifact.has_name() ? absl::make_optional(artifact.name())
                          : absl::nullopt,
      now, now, node_id);
}

// Creates an Execution (without properties).
absl::Status RDBMSMetadataAccessObject::CreateBasicNode(
    const Execution& execution, int64* node_id) {
  const absl::Time now = absl::Now();
  return executor_->InsertExecution(
      execution.type_id(),
      execution.has_last_known_state()
          ? absl::make_optional(execution.last_known_state())
          : absl::nullopt,
      execution.has_name() ? absl::make_optional(execution.name())
                           : absl::nullopt,
      now, now, node_id);
}

// Creates a Context (without properties).
absl::Status RDBMSMetadataAccessObject::CreateBasicNode(const Context& context,
                                                        int64* node_id) {
  const absl::Time now = absl::Now();
  if (!context.has_name() || context.name().empty()) {
    return absl::InvalidArgumentError("Context name should not be empty");
  }
  return executor_->InsertContext(context.type_id(), context.name(), now, now,
                                  node_id);
}

template <>
absl::Status RDBMSMetadataAccessObject::RetrieveNodesById(
    const absl::Span<const int64> ids, RecordSet* header, RecordSet* properties,
    Context* tag) {
  MLMD_RETURN_IF_ERROR(executor_->SelectContextsByID(ids, header));
  if (!header->records().empty()) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectContextPropertyByContextID(ids, properties));
  }
  return absl::OkStatus();
}

template <>
absl::Status RDBMSMetadataAccessObject::RetrieveNodesById(
    const absl::Span<const int64> ids, RecordSet* header, RecordSet* properties,
    Artifact* tag) {
  MLMD_RETURN_IF_ERROR(executor_->SelectArtifactsByID(ids, header));
  if (!header->records().empty()) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectArtifactPropertyByArtifactID(ids, properties));
  }
  return absl::OkStatus();
}

template <>
absl::Status RDBMSMetadataAccessObject::RetrieveNodesById(
    const absl::Span<const int64> ids, RecordSet* header, RecordSet* properties,
    Execution* tag) {
  MLMD_RETURN_IF_ERROR(executor_->SelectExecutionsByID(ids, header));
  if (!header->records().empty()) {
    MLMD_RETURN_IF_ERROR(
        executor_->SelectExecutionPropertyByExecutionID(ids, properties));
  }
  return absl::OkStatus();
}

// Update an Artifact's type_id, URI and last_update_time.
absl::Status RDBMSMetadataAccessObject::RunNodeUpdate(
    const Artifact& artifact) {
  return executor_->UpdateArtifactDirect(
      artifact.id(), artifact.type_id(), artifact.uri(),
      artifact.has_state() ? absl::make_optional(artifact.state())
                           : absl::nullopt,
      absl::Now());
}

// Update an Execution's type_id and last_update_time.
absl::Status RDBMSMetadataAccessObject::RunNodeUpdate(
    const Execution& execution) {
  return executor_->UpdateExecutionDirect(
      execution.id(), execution.type_id(),
      execution.has_last_known_state()
          ? absl::make_optional(execution.last_known_state())
          : absl::nullopt,
      absl::Now());
}

// Update a Context's type id and name.
absl::Status RDBMSMetadataAccessObject::RunNodeUpdate(const Context& context) {
  if (!context.has_name() || context.name().empty()) {
    return absl::InvalidArgumentError("Context name should not be empty");
  }
  return executor_->UpdateContextDirect(context.id(), context.type_id(),
                                        context.name(), absl::Now());
}

// Runs a property insertion query for a NodeType.
template <typename NodeType>
absl::Status RDBMSMetadataAccessObject::InsertProperty(
    const int64 node_id, const absl::string_view name,
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
absl::Status RDBMSMetadataAccessObject::UpdateProperty(
    const int64 node_id, const absl::string_view name, const Value& value) {
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
absl::Status RDBMSMetadataAccessObject::DeleteProperty(
    const int64 node_id, const absl::string_view name) {
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
// `prev_properties` (P). A property definition is a 2-tuple (name, value_type).
// a) any property in the intersection of C and P, a update query is generated.
// b) any property in C \ P, insert query is generated.
// c) any property in P \ C, delete query is generated.
// The queries are composed from corresponding template queries with the given
// `NodeType` (which is one of {`ArtifactType`, `ExecutionType`, `ContextType`}
// and the `is_custom_property` (which indicates the space of the given
// properties.
// Returns `output_num_changed_properties` which equals to the number of
// properties are changed (deleted, updated or inserted).
template <typename NodeType>
absl::Status RDBMSMetadataAccessObject::ModifyProperties(
    const google::protobuf::Map<std::string, Value>& curr_properties,
    const google::protobuf::Map<std::string, Value>& prev_properties, const int64 node_id,
    const bool is_custom_property, int& output_num_changed_properties) {
  output_num_changed_properties = 0;
  // generates delete clauses for properties in P \ C
  for (const auto& p : prev_properties) {
    const std::string& name = p.first;
    const Value& value = p.second;
    // check the 2-tuple (name, value_type) in prev_properties
    if (curr_properties.find(name) != curr_properties.end() &&
        curr_properties.at(name).value_case() == value.value_case())
      continue;

    MLMD_RETURN_IF_ERROR(DeleteProperty<NodeType>(node_id, name));
    output_num_changed_properties++;
  }

  for (const auto& p : curr_properties) {
    const std::string& name = p.first;
    const Value& value = p.second;
    const auto prev_value_it = prev_properties.find(name);
    if (prev_value_it != prev_properties.end() &&
        prev_value_it->second.value_case() == p.second.value_case()) {
      if (!google::protobuf::util::MessageDifferencer::Equals(prev_value_it->second,
                                                    value)) {
        // generates update clauses for properties in the intersection P & C
        MLMD_RETURN_IF_ERROR(UpdateProperty<NodeType>(node_id, name, value));
        output_num_changed_properties++;
      }
    } else {
      // generate insert clauses for properties in C \ P
      MLMD_RETURN_IF_ERROR(
          InsertProperty<NodeType>(node_id, name, is_custom_property, value));
      output_num_changed_properties++;
    }
  }
  return absl::OkStatus();
}

// Creates a query to insert an artifact type.
absl::Status RDBMSMetadataAccessObject::InsertTypeID(const ArtifactType& type,
                                                     int64* type_id) {
  return executor_->InsertArtifactType(
      type.name(), GetTypeVersion(type),
      type.has_description() ? absl::make_optional(type.description())
                             : absl::nullopt,
      type_id);
}

// Creates a query to insert an execution type.
absl::Status RDBMSMetadataAccessObject::InsertTypeID(const ExecutionType& type,
                                                     int64* type_id) {
  return executor_->InsertExecutionType(
      type.name(), GetTypeVersion(type),
      type.has_description() ? absl::make_optional(type.description())
                             : absl::nullopt,
      type.has_input_type() ? &type.input_type() : nullptr,
      type.has_output_type() ? &type.output_type() : nullptr, type_id);
}

// Creates a query to insert a context type.
absl::Status RDBMSMetadataAccessObject::InsertTypeID(const ContextType& type,
                                                     int64* type_id) {
  return executor_->InsertContextType(
      type.name(), GetTypeVersion(type),
      type.has_description() ? absl::make_optional(type.description())
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
                                                       int64* type_id) {
  const std::string& type_name = type.name();
  const google::protobuf::Map<std::string, PropertyType>& type_properties =
      type.properties();

  // validate the given type
  if (type_name.empty())
    return absl::InvalidArgumentError("No type name is specified.");
  if (type_properties.empty())
    LOG(INFO) << "No property is defined for the Type";

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
    const RecordSet& type_record_set, std::vector<MessageType>* types) {
  // Query type with the given condition
  const int num_records = type_record_set.records_size();
  types->resize(num_records);
  for (int i = 0; i < num_records; ++i) {
    MLMD_RETURN_IF_ERROR(
        ParseRecordSetToMessage(type_record_set, &types->at(i), i));

    RecordSet property_record_set;
    MLMD_RETURN_IF_ERROR(executor_->SelectPropertyByTypeID(
        types->at(i).id(), &property_record_set));

    MLMD_RETURN_IF_ERROR(ParseRecordSetToMapField(property_record_set,
                                                  "properties", &types->at(i)));
  }

  return absl::OkStatus();
}

template <typename MessageType>
absl::Status RDBMSMetadataAccessObject::FindTypeImpl(int64 type_id,
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
absl::Status RDBMSMetadataAccessObject::FindTypeImpl(
    absl::string_view name, absl::optional<absl::string_view> version,
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
  // updates the list of type properties
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
  return absl::OkStatus();
}

template <typename Type>
absl::Status RDBMSMetadataAccessObject::CreateParentTypeImpl(
    const int64 type_id, const int64 parent_type_id) {
  // Check whether there is a cyclic dependency if we insert the tuple:
  // (type_id, parent_type_id). We do a DFS traversal from `parent_type_id` as
  // the root node and it introduces a cycle if any ancestors is `type_id`. It
  // assumes that existing parent types inheritance are acyclic.
  std::vector<int64> ancestor_ids = {parent_type_id};
  absl::flat_hash_set<int64> visited_ancestors_ids;
  while (!ancestor_ids.empty()) {
    const int64 ancestor_id = ancestor_ids.back();
    if (ancestor_id == type_id) {
      return absl::InvalidArgumentError(
          "There is a cycle detected of the given parent type.");
    }
    ancestor_ids.pop_back();
    if (visited_ancestors_ids.contains(ancestor_id)) {
      continue;
    }
    visited_ancestors_ids.insert(ancestor_id);
    RecordSet record_set;
    MLMD_RETURN_IF_ERROR(
        executor_->SelectParentTypesByTypeID(ancestor_id, &record_set));
    for (const int64 parent_id : ParentTypesToParentTypeIds(record_set)) {
      ancestor_ids.push_back(parent_id);
    }
  }
  const absl::Status status =
      executor_->InsertParentType(type_id, parent_type_id);
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError("The ParentType already exists.");
  }
  return status;
}

template <typename Type>
absl::Status RDBMSMetadataAccessObject::FindParentTypesByTypeIdImpl(
    int64 type_id, std::vector<Type>& output_parent_types) {
  // check the there's a Type instance with the given type_id
  Type type;
  MLMD_RETURN_IF_ERROR(FindTypeImpl(type_id, &type));
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectParentTypesByTypeID(type_id, &record_set));
  const std::vector<int64> ids = ParentTypesToParentTypeIds(record_set);
  const int output_size = output_parent_types.size();
  output_parent_types.resize(output_size + ids.size());
  // TODO(b/169075639) Use a single query to retrieve a list of types.
  for (int i = output_size; i < output_size + ids.size(); i++) {
    MLMD_RETURN_IF_ERROR(FindTypeImpl(ids[i], &output_parent_types.at(i)));
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
absl::Status RDBMSMetadataAccessObject::CreateNodeImpl(const Node& node,
                                                       int64* node_id) {
  // clear node id
  *node_id = 0;
  // validate type
  if (!node.has_type_id())
    return absl::InvalidArgumentError("Type id is missing.");
  const int64 type_id = node.type_id();
  NodeType node_type;
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(FindTypeImpl(type_id, &node_type),
                                    "Cannot find type for ",
                                    node.ShortDebugString());

  // validate properties
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(ValidatePropertiesWithType(node, node_type),
                                    "Cannot validate properties of ",
                                    node.ShortDebugString());

  // insert a node and get the assigned id
  MLMD_RETURN_WITH_CONTEXT_IF_ERROR(CreateBasicNode(node, node_id),
                                    "Cannot create node for ",
                                    node.ShortDebugString());

  // insert properties
  const google::protobuf::Map<std::string, Value> prev_properties;
  int num_changed_properties = 0;
  MLMD_RETURN_IF_ERROR(ModifyProperties<NodeType>(
      node.properties(), prev_properties, *node_id,
      /*is_custom_property=*/false, num_changed_properties));
  int num_changed_custom_properties = 0;
  MLMD_RETURN_IF_ERROR(ModifyProperties<NodeType>(
      node.custom_properties(), prev_properties, *node_id,
      /*is_custom_property=*/true, num_changed_custom_properties));
  return absl::OkStatus();
}

template <typename Node>
absl::Status RDBMSMetadataAccessObject::FindNodesImpl(
    const absl::Span<const int64> node_ids, const bool skipped_ids_ok,
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

  MLMD_RETURN_IF_ERROR(ParseRecordSetToMessageArray(node_record_set, &nodes));

  // if there are properties associated with the nodes, parse the returned
  // values.
  if (!properties_record_set.records().empty()) {
    // First we build a hash map from node ids to Node messages, to
    // facilitate lookups.
    absl::flat_hash_map<int64, typename std::vector<Node>::iterator> node_by_id;
    for (auto i = nodes.begin(); i != nodes.end(); ++i) {
      node_by_id.insert({i->id(), i});
    }

    CHECK_EQ(properties_record_set.column_names_size(), 6);
    for (const RecordSet::Record& record : properties_record_set.records()) {
      // Match the record against a node in the hash map.
      int64 node_id;
      CHECK(absl::SimpleAtoi(record.values(0), &node_id));
      auto iter = node_by_id.find(node_id);
      CHECK(iter != node_by_id.end());
      Node& node = *iter->second;

      MLMD_RETURN_IF_ERROR(PopulateNodeProperties(record, node));
    }
  }

  if (node_ids.size() != nodes.size()) {
    std::vector<int64> found_ids;
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

template <typename Node>
absl::Status RDBMSMetadataAccessObject::FindNodeImpl(const int64 node_id,
                                                     Node* node) {
  std::vector<Node> nodes;
  MLMD_RETURN_IF_ERROR(
      FindNodesImpl({node_id}, /*skipped_ids_ok=*/true, nodes));
  *node = nodes.at(0);

  return absl::OkStatus();
}

// Updates a `Node` which is one of {`Artifact`, `Execution`, `Context`}.
// Returns INVALID_ARGUMENT error, if the node cannot be found
// Returns INVALID_ARGUMENT error, if the node does not match with its type
// Returns detailed INTERNAL error, if query execution fails.
template <typename Node, typename NodeType>
absl::Status RDBMSMetadataAccessObject::UpdateNodeImpl(const Node& node) {
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
  const int64 type_id = stored_node.type_id();

  NodeType stored_type;
  MLMD_RETURN_IF_ERROR(FindTypeImpl(type_id, &stored_type));
  MLMD_RETURN_IF_ERROR(ValidatePropertiesWithType(node, stored_type));

  // Update, insert, delete properties if changed.
  int num_changed_properties = 0;
  MLMD_RETURN_IF_ERROR(ModifyProperties<NodeType>(
      node.properties(), stored_node.properties(), node.id(),
      /*is_custom_property=*/false, num_changed_properties));
  int num_changed_custom_properties = 0;
  MLMD_RETURN_IF_ERROR(ModifyProperties<NodeType>(
      node.custom_properties(), stored_node.custom_properties(), node.id(),
      /*is_custom_property=*/true, num_changed_custom_properties));
  // Update node if attributes are different or properties are updated, so that
  // the last_update_time_since_epoch is updated properly.
  google::protobuf::util::MessageDifferencer diff;
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
    MLMD_RETURN_IF_ERROR(RunNodeUpdate(node));
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
  MLMD_RETURN_IF_ERROR(ParseRecordSetToMessageArray(event_record_set, events));

  absl::flat_hash_map<int64, Event*> event_id_to_event_map;
  std::vector<int64> event_ids;
  event_ids.reserve(event_record_set.records_size());
  for (int i = 0; i < events->size(); ++i) {
    CHECK_LT(i, event_record_set.records_size());
    const RecordSet::Record& record = event_record_set.records()[i];
    int64 event_id;
    CHECK(absl::SimpleAtoi(record.values(0), &event_id));
    event_id_to_event_map[event_id] = &(*events)[i];
    event_ids.push_back(event_id);
  }

  RecordSet path_record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectEventPathByEventIDs(event_ids, &path_record_set));
  for (const RecordSet::Record& record : path_record_set.records()) {
    int64 event_id;
    CHECK(absl::SimpleAtoi(record.values(0), &event_id));
    auto iter = event_id_to_event_map.find(event_id);
    CHECK(iter != event_id_to_event_map.end());
    Event* event = iter->second;
    bool is_index_step;
    CHECK(absl::SimpleAtob(record.values(1), &is_index_step));
    if (is_index_step) {
      int64 step_index;
      CHECK(absl::SimpleAtoi(record.values(2), &step_index));
      event->mutable_path()->add_steps()->set_index(step_index);
    } else {
      event->mutable_path()->add_steps()->set_key(record.values(3));
    }
  }
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::CreateType(const ArtifactType& type,
                                                   int64* type_id) {
  return CreateTypeImpl(type, type_id);
}

absl::Status RDBMSMetadataAccessObject::CreateType(const ExecutionType& type,
                                                   int64* type_id) {
  return CreateTypeImpl(type, type_id);
}

absl::Status RDBMSMetadataAccessObject::CreateType(const ContextType& type,
                                                   int64* type_id) {
  return CreateTypeImpl(type, type_id);
}

absl::Status RDBMSMetadataAccessObject::FindTypeById(
    const int64 type_id, ArtifactType* artifact_type) {
  return FindTypeImpl(type_id, artifact_type);
}

absl::Status RDBMSMetadataAccessObject::FindTypeById(
    const int64 type_id, ExecutionType* execution_type) {
  return FindTypeImpl(type_id, execution_type);
}

absl::Status RDBMSMetadataAccessObject::FindTypes(
    std::vector<ArtifactType>* artifact_types) {
  return FindAllTypeInstancesImpl(artifact_types);
}

absl::Status RDBMSMetadataAccessObject::FindTypeById(
    const int64 type_id, ContextType* context_type) {
  return FindTypeImpl(type_id, context_type);
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
    absl::string_view name, absl::optional<absl::string_view> version,
    ArtifactType* artifact_type) {
  return FindTypeImpl(name, version, artifact_type);
}

absl::Status RDBMSMetadataAccessObject::FindTypeByNameAndVersion(
    absl::string_view name, absl::optional<absl::string_view> version,
    ExecutionType* execution_type) {
  return FindTypeImpl(name, version, execution_type);
}

absl::Status RDBMSMetadataAccessObject::FindTypeByNameAndVersion(
    absl::string_view name, absl::optional<absl::string_view> version,
    ContextType* context_type) {
  return FindTypeImpl(name, version, context_type);
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
  return CreateParentTypeImpl<ArtifactType>(type.id(), parent_type.id());
}

absl::Status RDBMSMetadataAccessObject::CreateParentTypeInheritanceLink(
    const ExecutionType& type, const ExecutionType& parent_type) {
  if (!type.has_id() || !parent_type.has_id()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing id in the given types: ", type.DebugString(),
                     parent_type.DebugString()));
  }
  return CreateParentTypeImpl<ExecutionType>(type.id(), parent_type.id());
}

absl::Status RDBMSMetadataAccessObject::CreateParentTypeInheritanceLink(
    const ContextType& type, const ContextType& parent_type) {
  if (!type.has_id() || !parent_type.has_id()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing id in the given types: ", type.DebugString(),
                     parent_type.DebugString()));
  }
  return CreateParentTypeImpl<ContextType>(type.id(), parent_type.id());
}

absl::Status RDBMSMetadataAccessObject::FindParentTypesByTypeId(
    int64 type_id, std::vector<ArtifactType>& output_parent_types) {
  return FindParentTypesByTypeIdImpl(type_id, output_parent_types);
}

absl::Status RDBMSMetadataAccessObject::FindParentTypesByTypeId(
    int64 type_id, std::vector<ExecutionType>& output_parent_types) {
  return FindParentTypesByTypeIdImpl(type_id, output_parent_types);
}

absl::Status RDBMSMetadataAccessObject::FindParentTypesByTypeId(
    int64 type_id, std::vector<ContextType>& output_parent_types) {
  return FindParentTypesByTypeIdImpl(type_id, output_parent_types);
}

absl::Status RDBMSMetadataAccessObject::CreateArtifact(const Artifact& artifact,
                                                       int64* artifact_id) {
  const absl::Status& status =
      CreateNodeImpl<Artifact, ArtifactType>(artifact, artifact_id);
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Given node already exists: ", artifact.DebugString(),
                     status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::CreateExecution(
    const Execution& execution, int64* execution_id) {
  const absl::Status& status =
      CreateNodeImpl<Execution, ExecutionType>(execution, execution_id);
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Given node already exists: ", execution.DebugString(),
                     status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::CreateContext(const Context& context,
                                                      int64* context_id) {
  const absl::Status& status =
      CreateNodeImpl<Context, ContextType>(context, context_id);
  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Given node already exists: ", context.DebugString(),
                     status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::FindArtifactsById(
    const absl::Span<const int64> artifact_ids,
    std::vector<Artifact>* artifacts) {
  if (artifact_ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(artifact_ids, /*skipped_ids_ok=*/true, *artifacts);
}

absl::Status RDBMSMetadataAccessObject::FindExecutionsById(
    const absl::Span<const int64> execution_ids,
    std::vector<Execution>* executions) {
  if (execution_ids.empty()) {
    return absl::OkStatus();
  }

  return FindNodesImpl(execution_ids, /*skipped_ids_ok=*/true, *executions);
}

absl::Status RDBMSMetadataAccessObject::FindContextsById(
    const absl::Span<const int64> context_ids, std::vector<Context>* contexts) {
  if (context_ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(context_ids, /*skipped_ids_ok=*/true, *contexts);
}

absl::Status RDBMSMetadataAccessObject::UpdateArtifact(
    const Artifact& artifact) {
  return UpdateNodeImpl<Artifact, ArtifactType>(artifact);
}

absl::Status RDBMSMetadataAccessObject::UpdateExecution(
    const Execution& execution) {
  return UpdateNodeImpl<Execution, ExecutionType>(execution);
}

absl::Status RDBMSMetadataAccessObject::UpdateContext(const Context& context) {
  return UpdateNodeImpl<Context, ContextType>(context);
}

absl::Status RDBMSMetadataAccessObject::CreateEvent(const Event& event,
                                                    int64* event_id) {
  // validate the given event
  if (!event.has_artifact_id())
    return absl::InvalidArgumentError("No artifact id is specified.");
  if (!event.has_execution_id())
    return absl::InvalidArgumentError("No execution id is specified.");
  if (!event.has_type() || event.type() == Event::UNKNOWN)
    return absl::InvalidArgumentError("No event type is specified.");
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
    return absl::InvalidArgumentError(
        absl::StrCat("No execution with the given id ", event.execution_id()));

  // insert an event and get its given id
  int64 event_time = event.has_milliseconds_since_epoch()
                         ? event.milliseconds_since_epoch()
                         : absl::ToUnixMillis(absl::Now());

  MLMD_RETURN_IF_ERROR(
      executor_->InsertEvent(event.artifact_id(), event.execution_id(),
                             event.type(), event_time, event_id));
  // insert event paths
  for (const Event::Path::Step& step : event.path().steps()) {
    // step value oneof
    MLMD_RETURN_IF_ERROR(executor_->InsertEventPath(*event_id, step));
  }
  return absl::OkStatus();
}

absl::Status RDBMSMetadataAccessObject::FindEventsByArtifacts(
    const std::vector<int64>& artifact_ids, std::vector<Event>* events) {
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
    const std::vector<int64>& execution_ids, std::vector<Event>* events) {
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
    const Association& association, int64* association_id) {
  if (!association.has_context_id())
    return absl::InvalidArgumentError("No context id is specified.");
  RecordSet context_id_header;
  MLMD_RETURN_IF_ERROR(executor_->SelectContextsByID({association.context_id()},
                                                     &context_id_header));
  if (context_id_header.records_size() == 0)
    return absl::InvalidArgumentError("Context id not found.");

  if (!association.has_execution_id())
    return absl::InvalidArgumentError("No execution id is specified");
  RecordSet execution_id_header;
  MLMD_RETURN_IF_ERROR(executor_->SelectExecutionsByID(
      {association.execution_id()}, &execution_id_header));
  if (execution_id_header.records_size() == 0)
    return absl::InvalidArgumentError("Execution id not found.");

  absl::Status status = executor_->InsertAssociation(
      association.context_id(), association.execution_id(), association_id);

  if (IsUniqueConstraintViolated(status)) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Given association already exists: ", association.DebugString(),
        status.ToString()));
  }
  return status;
}

absl::Status RDBMSMetadataAccessObject::FindContextsByExecution(
    int64 execution_id, std::vector<Context>* contexts) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectAssociationByExecutionID(execution_id, &record_set));
  const std::vector<int64> context_ids = AssociationsToContextIds(record_set);
  if (context_ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No contexts found for execution_id: ", execution_id));
  }
  return FindNodesImpl(context_ids, /*skipped_ids_ok=*/false, *contexts);
}

absl::Status RDBMSMetadataAccessObject::FindExecutionsByContext(
    int64 context_id, std::vector<Execution>* executions) {
  std::string unused_next_page_toke;
  return FindExecutionsByContext(context_id, absl::nullopt, executions,
                                 &unused_next_page_toke);
}

absl::Status RDBMSMetadataAccessObject::FindExecutionsByContext(
    int64 context_id, absl::optional<ListOperationOptions> list_options,
    std::vector<Execution>* executions, std::string* next_page_token) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectAssociationByContextID(context_id, &record_set));
  const std::vector<int64> ids = AssociationsToExecutionIds(record_set);
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
    const Attribution& attribution, int64* attribution_id) {
  if (!attribution.has_context_id())
    return absl::InvalidArgumentError("No context id is specified.");
  RecordSet context_id_header;
  MLMD_RETURN_IF_ERROR(executor_->SelectContextsByID({attribution.context_id()},
                                                     &context_id_header));
  if (context_id_header.records_size() == 0)
    return absl::InvalidArgumentError("Context id not found.");

  if (!attribution.has_artifact_id())
    return absl::InvalidArgumentError("No artifact id is specified");
  RecordSet artifact_id_header;
  MLMD_RETURN_IF_ERROR(executor_->SelectArtifactsByID(
      {attribution.artifact_id()}, &artifact_id_header));
  if (artifact_id_header.records_size() == 0)
    return absl::InvalidArgumentError("Artifact id not found.");

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
    int64 artifact_id, std::vector<Context>* contexts) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectAttributionByArtifactID(artifact_id, &record_set));
  const std::vector<int64> context_ids = AttributionsToContextIds(record_set);
  if (context_ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No contexts found for artifact_id: ", artifact_id));
  }
  return FindNodesImpl(context_ids, /*skipped_ids_ok=*/false, *contexts);
}

absl::Status RDBMSMetadataAccessObject::FindArtifactsByContext(
    int64 context_id, std::vector<Artifact>* artifacts) {
  std::string unused_next_page_token;
  return FindArtifactsByContext(context_id, absl::nullopt, artifacts,
                                &unused_next_page_token);
}

absl::Status RDBMSMetadataAccessObject::FindArtifactsByContext(
    int64 context_id, absl::optional<ListOperationOptions> list_options,
    std::vector<Artifact>* artifacts, std::string* next_page_token) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectAttributionByContextID(context_id, &record_set));
  const std::vector<int64> ids = AttributionsToArtifactIds(record_set);
  if (ids.empty()) {
    return absl::OkStatus();
  }
  if (list_options.has_value()) {
    return ListNodes<Artifact>(list_options.value(), ids, artifacts,
                               next_page_token);
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *artifacts);
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
    int64 context_id, ParentContextTraverseDirection direction,
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
  const std::vector<int64> ids =
      ParentContextsToContextIds(record_set, is_parent);
  output_contexts.clear();
  if (ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, output_contexts);
}

absl::Status RDBMSMetadataAccessObject::FindParentContextsByContextId(
    int64 context_id, std::vector<Context>* contexts) {
  if (contexts == nullptr) {
    return absl::InvalidArgumentError("Given contexts is NULL.");
  }
  return FindLinkedContextsImpl(
      context_id, ParentContextTraverseDirection::kParent, *contexts);
}

absl::Status RDBMSMetadataAccessObject::FindChildContextsByContextId(
    int64 context_id, std::vector<Context>* contexts) {
  if (contexts == nullptr) {
    return absl::InvalidArgumentError("Given contexts is NULL.");
  }
  return FindLinkedContextsImpl(
      context_id, ParentContextTraverseDirection::kChild, *contexts);
}

absl::Status RDBMSMetadataAccessObject::FindArtifacts(
    std::vector<Artifact>* artifacts) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectAllArtifactIDs(&record_set));
  std::vector<int64> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *artifacts);
}

template <>
absl::Status RDBMSMetadataAccessObject::ListNodeIds(
    const ListOperationOptions& options,
    absl::optional<absl::Span<const int64>> candidate_ids,
    RecordSet* record_set, Artifact* tag) {
  return executor_->ListArtifactIDsUsingOptions(options, candidate_ids,
                                                record_set);
}

template <>
absl::Status RDBMSMetadataAccessObject::ListNodeIds(
    const ListOperationOptions& options,
    absl::optional<absl::Span<const int64>> candidate_ids,
    RecordSet* record_set, Execution* tag) {
  return executor_->ListExecutionIDsUsingOptions(options, candidate_ids,
                                                 record_set);
}

template <>
absl::Status RDBMSMetadataAccessObject::ListNodeIds(
    const ListOperationOptions& options,
    absl::optional<absl::Span<const int64>> candidate_ids,
    RecordSet* record_set, Context* tag) {
  return executor_->ListContextIDsUsingOptions(options, candidate_ids,
                                               record_set);
}

template <typename Node>
absl::Status RDBMSMetadataAccessObject::ListNodes(
    const ListOperationOptions& options,
    absl::optional<absl::Span<const int64>> candidate_ids,
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
  const std::vector<int64> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::OkStatus();
  }

  // Map node ids to positions
  absl::flat_hash_map<int64, size_t> position_by_id;
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
    const int64 type_id, const absl::string_view name, Artifact* artifact) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectArtifactByTypeIDAndArtifactName(
      type_id, name, &record_set));
  const std::vector<int64> ids = ConvertToIds(record_set);
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
    const int64 type_id, std::vector<Artifact>* artifacts) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectArtifactsByTypeID(type_id, &record_set));
  const std::vector<int64> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No artifacts found for type_id:", type_id));
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *artifacts);
}

absl::Status RDBMSMetadataAccessObject::FindExecutions(
    std::vector<Execution>* executions) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectAllExecutionIDs(&record_set));
  const std::vector<int64> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *executions);
}

absl::Status RDBMSMetadataAccessObject::FindExecutionByTypeIdAndExecutionName(
    const int64 type_id, const absl::string_view name, Execution* execution) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectExecutionByTypeIDAndExecutionName(
      type_id, name, &record_set));
  const std::vector<int64> ids = ConvertToIds(record_set);
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
    const int64 type_id, std::vector<Execution>* executions) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(
      executor_->SelectExecutionsByTypeID(type_id, &record_set));
  const std::vector<int64> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No executions found for type_id:", type_id));
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *executions);
}

absl::Status RDBMSMetadataAccessObject::FindContexts(
    std::vector<Context>* contexts) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectAllContextIDs(&record_set));
  const std::vector<int64> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::OkStatus();
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *contexts);
}

absl::Status RDBMSMetadataAccessObject::FindContextsByTypeId(
    const int64 type_id, absl::optional<ListOperationOptions> list_options,
    std::vector<Context>* contexts, std::string* next_page_token) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectContextsByTypeID(type_id, &record_set));
  const std::vector<int64> ids = ConvertToIds(record_set);
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
    const absl::string_view uri, std::vector<Artifact>* artifacts) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectArtifactsByURI(uri, &record_set));
  const std::vector<int64> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No artifacts found for uri:", uri));
  }
  return FindNodesImpl(ids, /*skipped_ids_ok=*/false, *artifacts);
}

absl::Status RDBMSMetadataAccessObject::FindContextByTypeIdAndContextName(
    int64 type_id, absl::string_view name, Context* context) {
  RecordSet record_set;
  MLMD_RETURN_IF_ERROR(executor_->SelectContextByTypeIDAndContextName(
      type_id, name, &record_set));
  const std::vector<int64> ids = ConvertToIds(record_set);
  if (ids.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "No contexts found with type_id: ", type_id, ", name: ", name));
  }
  std::vector<Context> contexts;
  MLMD_RETURN_IF_ERROR(FindNodesImpl(ids, /*skipped_ids_ok=*/false, contexts));
  // By design, a <type_id, name> pair uniquely identifies a context.
  // Fails if multiple contexts are found.
  CHECK_EQ(contexts.size(), 1)
      << absl::StrCat("Found more than one contexts with type_id: ", type_id,
                      " and context name: ", name);
  *context = contexts[0];
  return absl::OkStatus();
}


}  // namespace ml_metadata
