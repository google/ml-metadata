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

#include "google/protobuf/descriptor.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/message_differencer.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
// clang-format off
#ifdef _WIN32
#include "ml_metadata/metadata_store/rdbms_metadata_access_object.h" // NOLINT
#endif
// clang-format on
#include "ml_metadata/metadata_store/list_operation_util.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

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

// Parses and converts a string value to a specific field in a message.
// If the given string `value` is NULL (encoded as kMetadataSourceNull), then
// leave the field unset.
// The field should be a scalar field. The field type must be one of {string,
// int64, bool, enum, message}.
tensorflow::Status ParseValueToField(
    const google::protobuf::FieldDescriptor* field_descriptor,
    const absl::string_view value, google::protobuf::Message* message) {
  if (value == kMetadataSourceNull) {
    return tensorflow::Status::OK();
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
          return tensorflow::errors::Internal(
              ::absl::StrCat("Failed to parse proto: ", value));
        }
      }
      break;
    }
    default: {
      return tensorflow::errors::Internal(absl::StrCat(
          "Unsupported field type: ", field_descriptor->cpp_type()));
    }
  }
  return tensorflow::Status::OK();
}

// Converts a RecordSet in the query result to a MessageType. In the record at
// the `record_index`, its value of each column is assigned to a message field
// with the same field name as the column name.
template <typename MessageType>
tensorflow::Status ParseRecordSetToMessage(const RecordSet& record_set,
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
      TF_RETURN_IF_ERROR(ParseValueToField(field_descriptor, value, message));
    }
  }
  return tensorflow::Status::OK();
}

// Converts a RecordSet in the query result to a MessageType array.
template <typename MessageType>
tensorflow::Status ParseRecordSetToMessageArray(
    const RecordSet& record_set, std::vector<MessageType>* messages) {
  for (int i = 0; i < record_set.records_size(); i++) {
    messages->push_back(MessageType());
    TF_RETURN_IF_ERROR(
        ParseRecordSetToMessage(record_set, &messages->back(), i));
  }
  return tensorflow::Status::OK();
}

// Converts a RecordSet containing key-value pairs to a proto Map.
// The field_name is the map field in the MessageType. The method fills the
// message's map field with field_name using the rows in the given record_set.
template <typename MessageType>
tensorflow::Status ParseRecordSetToMapField(const RecordSet& record_set,
                                            const std::string& field_name,
                                            MessageType* message) {
  const google::protobuf::Descriptor* descriptor = message->descriptor();
  const google::protobuf::Reflection* reflection = message->GetReflection();
  const google::protobuf::FieldDescriptor* map_field_descriptor =
      descriptor->FindFieldByName(field_name);
  if (map_field_descriptor == nullptr || !map_field_descriptor->is_map()) {
    return tensorflow::errors::Internal(
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
    TF_RETURN_IF_ERROR(
        ParseValueToField(key_descriptor, key, map_field_message));
    TF_RETURN_IF_ERROR(
        ParseValueToField(value_descriptor, value, map_field_message));
  }

  return tensorflow::Status::OK();
}

// Validates properties in a `Node` with the properties defined in a `Type`.
// `Node` is one of {`Artifact`, `Execution`, `Context`}. `Type` is one of
// {`ArtifactType`, `ExecutionType`, `ContextType`}.
// Returns INVALID_ARGUMENT error, if there is unknown or mismatched property
// w.r.t. its definition.
template <typename Node, typename Type>
tensorflow::Status ValidatePropertiesWithType(const Node& node,
                                              const Type& type) {
  const google::protobuf::Map<std::string, PropertyType>& type_properties =
      type.properties();
  for (const auto& p : node.properties()) {
    const std::string& property_name = p.first;
    const Value& property_value = p.second;
    // Note that this is a google::protobuf::Map, not a std::map.
    if (type_properties.find(property_name) == type_properties.end())
      return tensorflow::errors::InvalidArgument(
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
      default: {
        return tensorflow::errors::Internal(absl::StrCat(
            "Unknown registered property type: ", type.DebugString()));
      }
    }
    if (!is_type_match)
      return tensorflow::errors::InvalidArgument(
          absl::StrCat("Found unmatched property type: ", property_name));
  }
  return tensorflow::Status::OK();
}

}  // namespace

// Creates an Artifact (without properties).
tensorflow::Status RDBMSMetadataAccessObject::CreateBasicNode(
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
tensorflow::Status RDBMSMetadataAccessObject::CreateBasicNode(
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
tensorflow::Status RDBMSMetadataAccessObject::CreateBasicNode(
    const Context& context, int64* node_id) {
  const absl::Time now = absl::Now();
  if (!context.has_name() || context.name().empty()) {
    return tensorflow::errors::InvalidArgument(
        "Context name should not be empty");
  }
  return executor_->InsertContext(context.type_id(), context.name(), now, now,
                                  node_id);
}

// Lookup Artifact by id.
tensorflow::Status RDBMSMetadataAccessObject::NodeLookups(
    const Artifact& artifact, RecordSet* header, RecordSet* properties) {
  TF_RETURN_IF_ERROR(executor_->SelectArtifactByID(artifact.id(), header));
  TF_RETURN_IF_ERROR(
      executor_->SelectArtifactPropertyByArtifactID(artifact.id(), properties));
  return tensorflow::Status::OK();
}

// Generates a select queries for an Execution by id.
tensorflow::Status RDBMSMetadataAccessObject::NodeLookups(
    const Execution& execution, RecordSet* header, RecordSet* properties) {
  TF_RETURN_IF_ERROR(executor_->SelectExecutionByID(execution.id(), header));
  TF_RETURN_IF_ERROR(executor_->SelectExecutionPropertyByExecutionID(
      execution.id(), properties));
  return tensorflow::Status::OK();
}

// Lookup Context by id.
tensorflow::Status RDBMSMetadataAccessObject::NodeLookups(
    const Context& context, RecordSet* header, RecordSet* properties) {
  TF_RETURN_IF_ERROR(executor_->SelectContextByID(context.id(), header));
  TF_RETURN_IF_ERROR(
      executor_->SelectContextPropertyByContextID(context.id(), properties));
  return tensorflow::Status::OK();
}

// Update an Artifact's type_id, URI and last_update_time.
tensorflow::Status RDBMSMetadataAccessObject::RunNodeUpdate(
    const Artifact& artifact) {
  return executor_->UpdateArtifactDirect(
      artifact.id(), artifact.type_id(), artifact.uri(),
      artifact.has_state() ? absl::make_optional(artifact.state())
                           : absl::nullopt,
      absl::Now());
}

// Update an Execution's type_id and last_update_time.
tensorflow::Status RDBMSMetadataAccessObject::RunNodeUpdate(
    const Execution& execution) {
  return executor_->UpdateExecutionDirect(
      execution.id(), execution.type_id(),
      execution.has_last_known_state()
          ? absl::make_optional(execution.last_known_state())
          : absl::nullopt,
      absl::Now());
}

// Update a Context's type id and name.
tensorflow::Status RDBMSMetadataAccessObject::RunNodeUpdate(
    const Context& context) {
  if (!context.has_name() || context.name().empty()) {
    return tensorflow::errors::InvalidArgument(
        "Context name should not be empty");
  }
  return executor_->UpdateContextDirect(context.id(), context.type_id(),
                                        context.name(), absl::Now());
}

// Runs a property insertion query for a NodeType.
template <typename NodeType>
tensorflow::Status RDBMSMetadataAccessObject::InsertProperty(
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
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported TypeKind: ", type_kind));
  }
}

// Generates a property update query for a NodeType.
template <typename NodeType>
tensorflow::Status RDBMSMetadataAccessObject::UpdateProperty(
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
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported TypeKind: ", type_kind));
  }
}

// Generates a property deletion query for a NodeType.
template <typename NodeType>
tensorflow::Status RDBMSMetadataAccessObject::DeleteProperty(
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
      return tensorflow::errors::Internal("Unsupported TypeKind.");
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
template <typename NodeType>
tensorflow::Status RDBMSMetadataAccessObject::ModifyProperties(
    const google::protobuf::Map<std::string, Value>& curr_properties,
    const google::protobuf::Map<std::string, Value>& prev_properties, const int64 node_id,
    const bool is_custom_property) {
  // generates delete clauses for properties in P \ C
  for (const auto& p : prev_properties) {
    const std::string& name = p.first;
    const Value& value = p.second;
    // check the 2-tuple (name, value_type) in prev_properties
    if (curr_properties.find(name) != curr_properties.end() &&
        curr_properties.at(name).value_case() == value.value_case())
      continue;

    TF_RETURN_IF_ERROR(DeleteProperty<NodeType>(node_id, name));
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
        TF_RETURN_IF_ERROR(UpdateProperty<NodeType>(node_id, name, value));
      }
    } else {
      // generate insert clauses for properties in C \ P
      TF_RETURN_IF_ERROR(
          InsertProperty<NodeType>(node_id, name, is_custom_property, value));
    }
  }
  return tensorflow::Status::OK();
}

// Creates a query to insert an artifact type.
tensorflow::Status RDBMSMetadataAccessObject::InsertTypeID(
    const ArtifactType& type, int64* type_id) {
  return executor_->InsertArtifactType(type.name(), type_id);
}

// Creates a query to insert an execution type.
tensorflow::Status RDBMSMetadataAccessObject::InsertTypeID(
    const ExecutionType& type, int64* type_id) {
  return executor_->InsertExecutionType(
      type.name(), type.has_input_type(), type.input_type(),
      type.has_output_type(), type.output_type(), type_id);
}

// Creates a query to insert a context type.
tensorflow::Status RDBMSMetadataAccessObject::InsertTypeID(
    const ContextType& type, int64* type_id) {
  return executor_->InsertContextType(type.name(), type_id);
}

// Creates a `Type` where acceptable ones are in {ArtifactType, ExecutionType,
// ContextType}.
// Returns INVALID_ARGUMENT error, if name field is not given.
// Returns INVALID_ARGUMENT error, if any property type is unknown.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Type>
tensorflow::Status RDBMSMetadataAccessObject::CreateTypeImpl(const Type& type,
                                                             int64* type_id) {
  const std::string& type_name = type.name();
  const google::protobuf::Map<std::string, PropertyType>& type_properties =
      type.properties();

  // validate the given type
  if (type_name.empty())
    return tensorflow::errors::InvalidArgument("No type name is specified.");
  if (type_properties.empty())
    LOG(WARNING) << "No property is defined for the Type";

  // insert a type and get its given id
  TF_RETURN_IF_ERROR(InsertTypeID(type, type_id));

  // insert type properties and commit
  for (const auto& property : type_properties) {
    const std::string& property_name = property.first;
    const PropertyType property_type = property.second;
    if (property_type == PropertyType::UNKNOWN) {
      LOG(ERROR) << "Property " << property_name << "'s value type is UNKNOWN.";
      return tensorflow::errors::InvalidArgument(
          absl::StrCat("Property ", property_name, " is UNKNOWN."));
    }
    TF_RETURN_IF_ERROR(
        executor_->InsertTypeProperty(*type_id, property_name, property_type));
  }
  return tensorflow::Status::OK();
}

// Generates a query to find type by id
tensorflow::Status RDBMSMetadataAccessObject::RunFindTypeByID(
    const int64 condition, const TypeKind type_kind, RecordSet* record_set) {
  TF_RETURN_IF_ERROR(
      executor_->SelectTypeByID(condition, type_kind, record_set));
  return tensorflow::Status::OK();
}

// Generates a query to find type by name
tensorflow::Status RDBMSMetadataAccessObject::RunFindTypeByID(
    absl::string_view condition, const TypeKind type_kind,
    RecordSet* record_set) {
  TF_RETURN_IF_ERROR(
      executor_->SelectTypeByName(condition, type_kind, record_set));
  return tensorflow::Status::OK();
}

// Generates a query to find all type instances.
tensorflow::Status RDBMSMetadataAccessObject::GenerateFindAllTypeInstancesQuery(
    const TypeKind type_kind, RecordSet* record_set) {
  return executor_->SelectAllTypes(type_kind, record_set);
}

// FindType takes a result of a query for types, and populates additional
// information such as properties, and returns it in `types`.
template <typename MessageType>
tensorflow::Status RDBMSMetadataAccessObject::FindTypesFromRecordSet(
    const RecordSet& type_record_set, std::vector<MessageType>* types) {
  // Query type with the given condition
  const int num_records = type_record_set.records_size();
  types->resize(num_records);
  for (int i = 0; i < num_records; ++i) {
    TF_RETURN_IF_ERROR(
        ParseRecordSetToMessage(type_record_set, &types->at(i), i));

    RecordSet property_record_set;
    TF_RETURN_IF_ERROR(executor_->SelectPropertyByTypeID(types->at(i).id(),
                                                         &property_record_set));

    TF_RETURN_IF_ERROR(ParseRecordSetToMapField(property_record_set,
                                                "properties", &types->at(i)));
  }

  return tensorflow::Status::OK();
}

// Finds a type by query conditions. Acceptable types are {ArtifactType,
// ExecutionType, ContextType} (`MessageType`). The types can be queried by two
// kinds of query conditions, which are type id (int64) or type
// name (string_view).
// Returns NOT_FOUND error, if the given type_id cannot be found.
// Returns detailed INTERNAL error, if query execution fails.
template <typename QueryCondition, typename MessageType>
tensorflow::Status RDBMSMetadataAccessObject::FindTypeImpl(
    const QueryCondition condition, MessageType* type) {
  const TypeKind type_kind = ResolveTypeKind(type);
  RecordSet record_set;
  TF_RETURN_IF_ERROR(RunFindTypeByID(condition, type_kind, &record_set));
  std::vector<MessageType> types;
  TF_RETURN_IF_ERROR(FindTypesFromRecordSet(record_set, &types));

  if (types.empty()) {
    return tensorflow::errors::NotFound(
        absl::StrCat("No type found for query: ", condition));
  }
  *type = std::move(types[0]);
  return tensorflow::Status::OK();
}

// Finds all type instances of the type `MessageType`.
// Returns detailed INTERNAL error, if query execution fails.
template <typename MessageType>
tensorflow::Status RDBMSMetadataAccessObject::FindAllTypeInstancesImpl(
    std::vector<MessageType>* types) {
  MessageType type;
  const TypeKind type_kind = ResolveTypeKind(&type);
  RecordSet record_set;
  TF_RETURN_IF_ERROR(GenerateFindAllTypeInstancesQuery(type_kind, &record_set));

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
tensorflow::Status RDBMSMetadataAccessObject::UpdateTypeImpl(const Type& type) {
  if (!type.has_name()) {
    return tensorflow::errors::InvalidArgument("No type name is specified.");
  }
  // find the current stored type and validate the id.
  Type stored_type;
  TF_RETURN_IF_ERROR(FindTypeImpl(type.name(), &stored_type));
  if (type.has_id() && type.id() != stored_type.id()) {
    return tensorflow::errors::InvalidArgument(
        "Given type id is different from the existing type: ",
        stored_type.DebugString());
  }
  // updates the list of type properties
  const google::protobuf::Map<std::string, PropertyType>& stored_properties =
      stored_type.properties();
  for (const auto& p : type.properties()) {
    const std::string& property_name = p.first;
    const PropertyType property_type = p.second;
    if (property_type == PropertyType::UNKNOWN) {
      return tensorflow::errors::InvalidArgument(
          "Property:", property_name, " type should not be UNKNOWN.");
    }
    if (stored_properties.find(property_name) != stored_properties.end()) {
      // for stored properties, type should not be changed.
      if (stored_properties.at(property_name) != property_type) {
        return tensorflow::errors::AlreadyExists(
            "Property:", property_name,
            " type is different from the existing type: ",
            stored_type.DebugString());
      }
      continue;
    }
    TF_RETURN_IF_ERROR(executor_->InsertTypeProperty(
        stored_type.id(), property_name, property_type));
  }
  return tensorflow::Status::OK();
}

// Creates an `Node`, which is one of {`Artifact`, `Execution`, `Context`},
// then returns the assigned node id. The node's id field is ignored. The node
// should have a `NodeType`, which is one of {`ArtifactType`, `ExecutionType`,
// `ContextType`}.
// Returns INVALID_ARGUMENT error, if the node does not align with its type.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Node, typename NodeType>
tensorflow::Status RDBMSMetadataAccessObject::CreateNodeImpl(const Node& node,
                                                             int64* node_id) {
  // clear node id
  *node_id = 0;
  // validate type
  if (!node.has_type_id())
    return tensorflow::errors::InvalidArgument("Type id is missing.");
  const int64 type_id = node.type_id();
  NodeType node_type;
  TF_RETURN_IF_ERROR(FindTypeImpl(type_id, &node_type));

  // validate properties
  TF_RETURN_IF_ERROR(ValidatePropertiesWithType(node, node_type));

  // insert a node and get the assigned id
  TF_RETURN_IF_ERROR(CreateBasicNode(node, node_id));

  // insert properties
  const google::protobuf::Map<std::string, Value> prev_properties;
  TF_RETURN_IF_ERROR(ModifyProperties<NodeType>(node.properties(),
                                                prev_properties, *node_id,
                                                /*is_custom_property=*/false));
  TF_RETURN_IF_ERROR(ModifyProperties<NodeType>(node.custom_properties(),
                                                prev_properties, *node_id,
                                                /*is_custom_property=*/true));
  return tensorflow::Status::OK();
}

// Queries a `Node` which is one of {`Artifact`, `Execution`, `Context`} by
// an id.
// Returns NOT_FOUND error, if the given id cannot be found.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Node>
tensorflow::Status RDBMSMetadataAccessObject::FindNodeImpl(const int64 node_id,
                                                           Node* node) {
  node->set_id(node_id);
  RecordSet node_record_set;
  RecordSet properties_record_set;

  TF_RETURN_IF_ERROR(
      NodeLookups(*node, &node_record_set, &properties_record_set));

  if (node_record_set.records_size() == 0)
    return tensorflow::errors::NotFound(
        absl::StrCat("Cannot find record by given id ", node_id));

  TF_RETURN_IF_ERROR(ParseRecordSetToMessage(node_record_set, node));

  // it is ok that there is no property associated with a node
  if (properties_record_set.records_size() == 0)
    return tensorflow::Status::OK();
  // if there are properties associated with the node, parse the returned values
  CHECK_EQ(properties_record_set.column_names_size(), 5);
  for (const RecordSet::Record& record : properties_record_set.records()) {
    const std::string& property_name = record.values(0);
    bool is_custom_property;
    CHECK(absl::SimpleAtob(record.values(1), &is_custom_property));
    auto& property_value =
        (is_custom_property
             ? (*node->mutable_custom_properties())[property_name]
             : (*node->mutable_properties())[property_name]);
    if (record.values(2) != kMetadataSourceNull) {
      int64 int_value;
      CHECK(absl::SimpleAtoi(record.values(2), &int_value));
      property_value.set_int_value(int_value);
    } else if (record.values(3) != kMetadataSourceNull) {
      double double_value;
      CHECK(absl::SimpleAtod(record.values(3), &double_value));
      property_value.set_double_value(double_value);
    } else {
      const std::string& string_value = record.values(4);
      property_value.set_string_value(string_value);
    }
  }
  return tensorflow::Status::OK();
}

// Find nodes by ID, where the IDs are encoded in a record set.
template <typename Node>
tensorflow::Status RDBMSMetadataAccessObject::FindManyNodesImpl(
    const RecordSet& record_set, std::vector<Node>* nodes) {
  if (record_set.records_size() == 0)
    return tensorflow::errors::NotFound(absl::StrCat("Cannot find any record"));
  nodes->reserve(record_set.records_size());
  for (const RecordSet::Record& record : record_set.records()) {
    int64 node_id;
    CHECK(absl::SimpleAtoi(record.values(0), &node_id));
    nodes->push_back(Node());
    TF_RETURN_IF_ERROR(FindNodeImpl<Node>(node_id, &nodes->back()));
  }
  return tensorflow::Status::OK();
}

// Updates a `Node` which is one of {`Artifact`, `Execution`, `Context`}.
// Returns INVALID_ARGUMENT error, if the node cannot be found
// Returns INVALID_ARGUMENT error, if the node does not match with its type
// Returns detailed INTERNAL error, if query execution fails.
template <typename Node, typename NodeType>
tensorflow::Status RDBMSMetadataAccessObject::UpdateNodeImpl(const Node& node) {
  // validate node
  if (!node.has_id())
    return tensorflow::errors::InvalidArgument("No id is given.");

  Node stored_node;
  tensorflow::Status status = FindNodeImpl(node.id(), &stored_node);
  if (tensorflow::errors::IsNotFound(status)) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Cannot find the given id ", node.id()));
  }
  if (!status.ok()) return status;
  if (node.has_type_id() && node.type_id() != stored_node.type_id()) {
    return tensorflow::errors::InvalidArgument(absl::StrCat(
        "Given type_id ", node.type_id(),
        " is different from the one known before: ", stored_node.type_id()));
  }
  const int64 type_id = stored_node.type_id();

  NodeType stored_type;
  TF_RETURN_IF_ERROR(FindTypeImpl(type_id, &stored_type));
  TF_RETURN_IF_ERROR(ValidatePropertiesWithType(node, stored_type));

  // update nodes, and update, insert, delete properties
  google::protobuf::util::MessageDifferencer diff;
  diff.IgnoreField(Node::descriptor()->FindFieldByName("properties"));
  diff.IgnoreField(Node::descriptor()->FindFieldByName("custom_properties"));
  if (!diff.Compare(node, stored_node)) {
    TF_RETURN_IF_ERROR(RunNodeUpdate(node));
  }

  // modify properties
  TF_RETURN_IF_ERROR(ModifyProperties<NodeType>(
      node.properties(), stored_node.properties(), node.id(),
      /*is_custom_property=*/false));
  TF_RETURN_IF_ERROR(ModifyProperties<NodeType>(
      node.custom_properties(), stored_node.custom_properties(), node.id(),
      /*is_custom_property=*/true));
  return tensorflow::Status::OK();
}

// Takes a record set that has one record per event, parses them into Event
// objects, gets the paths for the events from the database using collected
// event ids, and assign paths to each corresponding event.
// Returns INVALID_ARGUMENT error, if the `events` is null.
tensorflow::Status RDBMSMetadataAccessObject::FindEventsFromRecordSet(
    const RecordSet& event_record_set, std::vector<Event>* events) {
  if (events == nullptr)
    return tensorflow::errors::InvalidArgument("Given events is NULL.");

  events->reserve(event_record_set.records_size());
  TF_RETURN_IF_ERROR(ParseRecordSetToMessageArray(event_record_set, events));

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
  TF_RETURN_IF_ERROR(
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
  return tensorflow::Status::OK();
}

// Queries `contexts` related to a `Node` (either `Artifact` or `Execution`)
// by the node id.
// Returns INVALID_ARGUMENT error, if the `contexts` is null.
template <typename Node>
tensorflow::Status RDBMSMetadataAccessObject::FindContextsByNodeImpl(
    const int64 node_id, std::vector<Context>* contexts) {
  if (contexts == nullptr)
    return tensorflow::errors::InvalidArgument("Given contexts is NULL.");

  constexpr bool is_artifact = std::is_same<Node, Artifact>::value;

  // find context ids with the given node id
  RecordSet node_ids;
  if (is_artifact) {
    TF_RETURN_IF_ERROR(
        executor_->SelectAttributionByArtifactID(node_id, &node_ids));

  } else {
    TF_RETURN_IF_ERROR(
        executor_->SelectAssociationByExecutionID(node_id, &node_ids));
  }

  contexts->clear();
  for (const RecordSet::Record& record : node_ids.records()) {
    contexts->push_back(Context());
    Context& curr_context = contexts->back();
    TF_RETURN_IF_ERROR(
        ParseValueToField(curr_context.descriptor()->FindFieldByName("id"),
                          record.values(1), &curr_context));
    TF_RETURN_IF_ERROR(FindNodeImpl(curr_context.id(), &curr_context));
  }
  return tensorflow::Status::OK();
}

// Queries nodes related to a context. Node is either `Artifact` or `Execution`.
// Returns INVALID_ARGUMENT error, if the `nodes` is null.
template <typename Node>
tensorflow::Status RDBMSMetadataAccessObject::FindNodesByContextImpl(
    const int64 context_id, std::vector<Node>* nodes) {
  if (nodes == nullptr)
    return tensorflow::errors::InvalidArgument("Given array is NULL.");

  constexpr bool is_artifact = std::is_same<Node, Artifact>::value;

  RecordSet record_set;
  // find node ids with the given context id
  if (is_artifact) {
    TF_RETURN_IF_ERROR(
        executor_->SelectAttributionByContextID(context_id, &record_set));
  } else {
    TF_RETURN_IF_ERROR(
        executor_->SelectAssociationByContextID(context_id, &record_set));
  }

  nodes->clear();
  for (const RecordSet::Record& record : record_set.records()) {
    nodes->push_back(Node());
    Node& curr_node = nodes->back();
    TF_RETURN_IF_ERROR(
        ParseValueToField(curr_node.descriptor()->FindFieldByName("id"),
                          record.values(2), &curr_node));
    TF_RETURN_IF_ERROR(FindNodeImpl(curr_node.id(), &curr_node));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status RDBMSMetadataAccessObject::CreateType(
    const ArtifactType& type, int64* type_id) {
  return CreateTypeImpl(type, type_id);
}

tensorflow::Status RDBMSMetadataAccessObject::CreateType(
    const ExecutionType& type, int64* type_id) {
  return CreateTypeImpl(type, type_id);
}

tensorflow::Status RDBMSMetadataAccessObject::CreateType(
    const ContextType& type, int64* type_id) {
  return CreateTypeImpl(type, type_id);
}

tensorflow::Status RDBMSMetadataAccessObject::FindTypeById(
    const int64 type_id, ArtifactType* artifact_type) {
  return FindTypeImpl(type_id, artifact_type);
}

tensorflow::Status RDBMSMetadataAccessObject::FindTypeById(
    const int64 type_id, ExecutionType* execution_type) {
  return FindTypeImpl(type_id, execution_type);
}

tensorflow::Status RDBMSMetadataAccessObject::FindTypes(
    std::vector<ArtifactType>* artifact_types) {
  return FindAllTypeInstancesImpl(artifact_types);
}

tensorflow::Status RDBMSMetadataAccessObject::FindTypeById(
    const int64 type_id, ContextType* context_type) {
  return FindTypeImpl(type_id, context_type);
}

tensorflow::Status RDBMSMetadataAccessObject::FindTypes(
    std::vector<ExecutionType>* execution_types) {
  return FindAllTypeInstancesImpl(execution_types);
}

tensorflow::Status RDBMSMetadataAccessObject::FindTypes(
    std::vector<ContextType>* context_types) {
  return FindAllTypeInstancesImpl(context_types);
}

tensorflow::Status RDBMSMetadataAccessObject::FindTypeByName(
    absl::string_view name, ArtifactType* artifact_type) {
  return FindTypeImpl(name, artifact_type);
}

tensorflow::Status RDBMSMetadataAccessObject::FindTypeByName(
    absl::string_view name, ExecutionType* execution_type) {
  return FindTypeImpl(name, execution_type);
}

tensorflow::Status RDBMSMetadataAccessObject::FindTypeByName(
    absl::string_view name, ContextType* context_type) {
  return FindTypeImpl(name, context_type);
}

tensorflow::Status RDBMSMetadataAccessObject::UpdateType(
    const ArtifactType& type) {
  return UpdateTypeImpl(type);
}

tensorflow::Status RDBMSMetadataAccessObject::UpdateType(
    const ExecutionType& type) {
  return UpdateTypeImpl(type);
}

tensorflow::Status RDBMSMetadataAccessObject::UpdateType(
    const ContextType& type) {
  return UpdateTypeImpl(type);
}

tensorflow::Status RDBMSMetadataAccessObject::CreateArtifact(
    const Artifact& artifact, int64* artifact_id) {
  return CreateNodeImpl<Artifact, ArtifactType>(artifact, artifact_id);
}

tensorflow::Status RDBMSMetadataAccessObject::CreateExecution(
    const Execution& execution, int64* execution_id) {
  return CreateNodeImpl<Execution, ExecutionType>(execution, execution_id);
}

tensorflow::Status RDBMSMetadataAccessObject::CreateContext(
    const Context& context, int64* context_id) {
  tensorflow::Status status =
      CreateNodeImpl<Context, ContextType>(context, context_id);
  if (absl::StrContains(status.error_message(), "Duplicate") ||
      absl::StrContains(status.error_message(), "UNIQUE")) {
    return tensorflow::errors::AlreadyExists(
        "Given node already exists: ", context.DebugString(), status);
  }
  return status;
}

tensorflow::Status RDBMSMetadataAccessObject::FindArtifactById(
    const int64 artifact_id, Artifact* artifact) {
  return FindNodeImpl(artifact_id, artifact);
}

tensorflow::Status RDBMSMetadataAccessObject::FindExecutionById(
    const int64 execution_id, Execution* execution) {
  return FindNodeImpl(execution_id, execution);
}

tensorflow::Status RDBMSMetadataAccessObject::FindContextById(
    const int64 context_id, Context* context) {
  return FindNodeImpl(context_id, context);
}

tensorflow::Status RDBMSMetadataAccessObject::UpdateArtifact(
    const Artifact& artifact) {
  return UpdateNodeImpl<Artifact, ArtifactType>(artifact);
}

tensorflow::Status RDBMSMetadataAccessObject::UpdateExecution(
    const Execution& execution) {
  return UpdateNodeImpl<Execution, ExecutionType>(execution);
}

tensorflow::Status RDBMSMetadataAccessObject::UpdateContext(
    const Context& context) {
  return UpdateNodeImpl<Context, ContextType>(context);
}

tensorflow::Status RDBMSMetadataAccessObject::CreateEvent(const Event& event,

                                                          int64* event_id) {
  // validate the given event
  if (!event.has_artifact_id())
    return tensorflow::errors::InvalidArgument("No artifact id is specified.");
  if (!event.has_execution_id())
    return tensorflow::errors::InvalidArgument("No execution id is specified.");
  if (!event.has_type() || event.type() == Event::UNKNOWN)
    return tensorflow::errors::InvalidArgument("No event type is specified.");
  RecordSet artifacts;
  TF_RETURN_IF_ERROR(
      executor_->SelectArtifactByID(event.artifact_id(), &artifacts));
  RecordSet executions;
  TF_RETURN_IF_ERROR(
      executor_->SelectExecutionByID(event.execution_id(), &executions));
  RecordSet record_set;
  if (artifacts.records_size() == 0)
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("No artifact with the given id ", event.artifact_id()));
  if (executions.records_size() == 0)
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("No execution with the given id ", event.execution_id()));

  // insert an event and get its given id
  int64 event_time = event.has_milliseconds_since_epoch()
                         ? event.milliseconds_since_epoch()
                         : absl::ToUnixMillis(absl::Now());

  TF_RETURN_IF_ERROR(executor_->InsertEvent(event.artifact_id(),
                                            event.execution_id(), event.type(),
                                            event_time, event_id));
  // insert event paths
  for (const Event::Path::Step& step : event.path().steps()) {
    // step value oneof
    TF_RETURN_IF_ERROR(executor_->InsertEventPath(*event_id, step));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status RDBMSMetadataAccessObject::FindEventsByArtifacts(
    const std::vector<int64>& artifact_ids, std::vector<Event>* events) {
  if (events == nullptr) {
    return tensorflow::errors::InvalidArgument("Given events is NULL.");
  }

  RecordSet event_record_set;
  if (!artifact_ids.empty()) {
    TF_RETURN_IF_ERROR(
        executor_->SelectEventByArtifactIDs(artifact_ids, &event_record_set));
  }

  if (event_record_set.records_size() == 0) {
    return tensorflow::errors::NotFound(
        "Cannot find events by given artifact ids.");
  }
  return FindEventsFromRecordSet(event_record_set, events);
}

tensorflow::Status RDBMSMetadataAccessObject::FindEventsByExecutions(
    const std::vector<int64>& execution_ids, std::vector<Event>* events) {
  if (events == nullptr) {
    return tensorflow::errors::InvalidArgument("Given events is NULL.");
  }

  RecordSet event_record_set;
  if (!execution_ids.empty()) {
    TF_RETURN_IF_ERROR(
        executor_->SelectEventByExecutionIDs(execution_ids, &event_record_set));
  }

  if (event_record_set.records_size() == 0) {
      return tensorflow::errors::NotFound(
          "Cannot find events by given execution ids.");
  }
  return FindEventsFromRecordSet(event_record_set, events);
}

tensorflow::Status RDBMSMetadataAccessObject::CreateAssociation(
    const Association& association, int64* association_id) {
  if (!association.has_context_id())
    return tensorflow::errors::InvalidArgument("No context id is specified.");
  RecordSet context_id_header;
  TF_RETURN_IF_ERROR(executor_->SelectContextByID(association.context_id(),
                                                  &context_id_header));
  if (context_id_header.records_size() == 0)
    return tensorflow::errors::InvalidArgument("Context id not found.");

  if (!association.has_execution_id())
    return tensorflow::errors::InvalidArgument("No execution id is specified");
  RecordSet execution_id_header;
  TF_RETURN_IF_ERROR(executor_->SelectExecutionByID(association.execution_id(),
                                                    &execution_id_header));
  if (execution_id_header.records_size() == 0)
    return tensorflow::errors::InvalidArgument("Execution id not found.");

  tensorflow::Status status = executor_->InsertAssociation(
      association.context_id(), association.execution_id(), association_id);

  if (absl::StrContains(status.error_message(), "Duplicate") ||
      absl::StrContains(status.error_message(), "UNIQUE")) {
    return tensorflow::errors::AlreadyExists(
        "Given association already exists: ", association.DebugString(),
        status);
  }
  return status;
}

tensorflow::Status RDBMSMetadataAccessObject::FindContextsByExecution(
    int64 execution_id, std::vector<Context>* contexts) {
  return FindContextsByNodeImpl<Execution>(execution_id, contexts);
}

tensorflow::Status RDBMSMetadataAccessObject::FindExecutionsByContext(
    int64 context_id, std::vector<Execution>* executions) {
  return FindNodesByContextImpl(context_id, executions);
}

tensorflow::Status RDBMSMetadataAccessObject::CreateAttribution(
    const Attribution& attribution, int64* attribution_id) {
  if (!attribution.has_context_id())
    return tensorflow::errors::InvalidArgument("No context id is specified.");
  RecordSet context_id_header;
  TF_RETURN_IF_ERROR(executor_->SelectContextByID(attribution.context_id(),
                                                  &context_id_header));
  if (context_id_header.records_size() == 0)
    return tensorflow::errors::InvalidArgument("Context id not found.");

  if (!attribution.has_artifact_id())
    return tensorflow::errors::InvalidArgument("No artifact id is specified");
  RecordSet artifact_id_header;
  TF_RETURN_IF_ERROR(executor_->SelectArtifactByID(attribution.artifact_id(),
                                                   &artifact_id_header));
  if (artifact_id_header.records_size() == 0)
    return tensorflow::errors::InvalidArgument("Artifact id not found.");

  tensorflow::Status status = executor_->InsertAttributionDirect(
      attribution.context_id(), attribution.artifact_id(), attribution_id);

  if (absl::StrContains(status.error_message(), "Duplicate") ||
      absl::StrContains(status.error_message(), "UNIQUE")) {
    return tensorflow::errors::AlreadyExists(
        "Given attribution already exists: ", attribution.DebugString(),
        status);
  }
  return status;
}

tensorflow::Status RDBMSMetadataAccessObject::FindContextsByArtifact(
    int64 artifact_id, std::vector<Context>* contexts) {
  return FindContextsByNodeImpl<Artifact>(artifact_id, contexts);
}

tensorflow::Status RDBMSMetadataAccessObject::FindArtifactsByContext(
    int64 context_id, std::vector<Artifact>* artifacts) {
  return FindNodesByContextImpl(context_id, artifacts);
}

tensorflow::Status RDBMSMetadataAccessObject::FindArtifacts(
    std::vector<Artifact>* artifacts) {
  RecordSet record_set;
  TF_RETURN_IF_ERROR(executor_->SelectAllArtifactIDs(&record_set));
  return FindManyNodesImpl(record_set, artifacts);
}

template <typename Node>
tensorflow::Status RDBMSMetadataAccessObject::ListNodes(
    const ListOperationOptions& options, std::vector<Node>* nodes,
    std::string* next_page_token) {
  if (options.max_result_size() <= 0) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("max_result_size field value is required to be greater "
                     "than 0 and less than or equal to 100. Set value:",
                     options.max_result_size()));
  }

  // Retrieving page of size 1 greater that max_result_size to detect if this
  // is the last page.
  ListOperationOptions updated_options;
  updated_options.CopyFrom(options);
  updated_options.set_max_result_size(options.max_result_size() + 1);

  RecordSet record_set;
  if (std::is_same<Node, Artifact>::value) {
    TF_RETURN_IF_ERROR(
        executor_->ListArtifactIDsUsingOptions(updated_options, &record_set));
  } else if (std::is_same<Node, Execution>::value) {
    TF_RETURN_IF_ERROR(
        executor_->ListExecutionIDsUsingOptions(updated_options, &record_set));
  } else if (std::is_same<Node, Context>::value) {
    TF_RETURN_IF_ERROR(
        executor_->ListContextIDsUsingOptions(updated_options, &record_set));
  } else {
    return tensorflow::errors::InvalidArgument(
        "Invalid Node passed to ListNodes");
  }

  TF_RETURN_IF_ERROR(FindManyNodesImpl(record_set, nodes));

  if (nodes->size() > options.max_result_size()) {
    // Removing the extra node retrieved for last page detection.
    nodes->pop_back();
    Node last_node = nodes->back();
    TF_RETURN_IF_ERROR(BuildListOperationNextPageToken<Node>(last_node, options,
                                                             next_page_token));
  } else {
    *next_page_token = "";
  }
  return tensorflow::Status::OK();
}

tensorflow::Status RDBMSMetadataAccessObject::ListArtifacts(
    const ListOperationOptions& options, std::vector<Artifact>* artifacts,
    std::string* next_page_token) {
  return ListNodes<Artifact>(options, artifacts, next_page_token);
}

tensorflow::Status RDBMSMetadataAccessObject::ListExecutions(
    const ListOperationOptions& options, std::vector<Execution>* executions,
    std::string* next_page_token) {
  return ListNodes<Execution>(options, executions, next_page_token);
}

tensorflow::Status RDBMSMetadataAccessObject::ListContexts(
    const ListOperationOptions& options, std::vector<Context>* contexts,
    std::string* next_page_token) {
  return ListNodes<Context>(options, contexts, next_page_token);
}

tensorflow::Status
RDBMSMetadataAccessObject::FindArtifactByTypeIdAndArtifactName(
    const int64 type_id, const absl::string_view name, Artifact* artifact) {
  RecordSet record_set;
  TF_RETURN_IF_ERROR(executor_->SelectArtifactByTypeIDAndArtifactName(
      type_id, name, &record_set));
  std::vector<Artifact> artifacts;
  TF_RETURN_IF_ERROR(FindManyNodesImpl(record_set, &artifacts));
  // By design, a <type_id, name> pair uniquely identifies an artifact.
  // Fails if multiple artifacts are found.
  // Returns ok status and updates the input artifact if one artifact is
  // found. Returns ok status and does nothing if no artifact is found.
  CHECK(artifacts.size() <= 1) << absl::StrCat(
      "Found more than one artifact with type_id: ", std::to_string(type_id),
      " and artifact name: ", name);
  if (artifacts.size() == 1) {
    *artifact = artifacts[0];
  }
  return tensorflow::Status::OK();
}

tensorflow::Status RDBMSMetadataAccessObject::FindArtifactsByTypeId(
    const int64 type_id, std::vector<Artifact>* artifacts) {
  RecordSet record_set;
  TF_RETURN_IF_ERROR(executor_->SelectArtifactsByTypeID(type_id, &record_set));
  return FindManyNodesImpl(record_set, artifacts);
}

tensorflow::Status RDBMSMetadataAccessObject::FindExecutions(
    std::vector<Execution>* executions) {
  RecordSet record_set;
  TF_RETURN_IF_ERROR(executor_->SelectAllExecutionIDs(&record_set));
  return FindManyNodesImpl(record_set, executions);
}

tensorflow::Status
RDBMSMetadataAccessObject::FindExecutionByTypeIdAndExecutionName(
    const int64 type_id, const absl::string_view name, Execution* execution) {
  RecordSet record_set;
  TF_RETURN_IF_ERROR(executor_->SelectExecutionByTypeIDAndExecutionName(
      type_id, name, &record_set));
  std::vector<Execution> executions;
  TF_RETURN_IF_ERROR(FindManyNodesImpl(record_set, &executions));
  // By design, a <type_id, name> pair uniquely identifies an execution.
  // Fails if multiple executions are found.
  // Returns ok status and updates the input execution if one execution is
  // found. Returns ok status and does nothing if no execution is found.
  CHECK(executions.size() <= 1) << absl::StrCat(
      "Found more than one execution with type_id: ", std::to_string(type_id),
      " and execution name: ", name);
  if (executions.size() == 1) {
    *execution = executions[0];
  }
  return tensorflow::Status::OK();
}

tensorflow::Status RDBMSMetadataAccessObject::FindExecutionsByTypeId(
    const int64 type_id, std::vector<Execution>* executions) {
  RecordSet record_set;
  TF_RETURN_IF_ERROR(executor_->SelectExecutionsByTypeID(type_id, &record_set));
  return FindManyNodesImpl(record_set, executions);
}

tensorflow::Status RDBMSMetadataAccessObject::FindContexts(
    std::vector<Context>* contexts) {
  RecordSet record_set;
  TF_RETURN_IF_ERROR(executor_->SelectAllContextIDs(&record_set));
  return FindManyNodesImpl(record_set, contexts);
}

tensorflow::Status RDBMSMetadataAccessObject::FindContextsByTypeId(
    const int64 type_id, std::vector<Context>* contexts) {
  RecordSet record_set;
  TF_RETURN_IF_ERROR(executor_->SelectContextsByTypeID(type_id, &record_set));
  return FindManyNodesImpl(record_set, contexts);
}

tensorflow::Status RDBMSMetadataAccessObject::FindArtifactsByURI(
    const absl::string_view uri, std::vector<Artifact>* artifacts) {
  RecordSet record_set;
  TF_RETURN_IF_ERROR(executor_->SelectArtifactsByURI(uri, &record_set));
  return FindManyNodesImpl(record_set, artifacts);
}

tensorflow::Status RDBMSMetadataAccessObject::FindContextByTypeIdAndContextName(
    int64 type_id, absl::string_view name, Context* context) {
  RecordSet record_set;
  TF_RETURN_IF_ERROR(executor_->SelectContextByTypeIDAndContextName(
      type_id, name, &record_set));
  std::vector<Context> contexts;
  TF_RETURN_IF_ERROR(FindManyNodesImpl(record_set, &contexts));
  // By design, a <type_id, name> pair uniquely identifies a context.
  // Fails if multiple contexts are found.
  // Returns ok status and updates the input context if one context is found.
  // Returns ok status and does nothing if no context is found.
  CHECK(contexts.size() <= 1) << absl::StrCat(
      "Found more than one contexts with type_id: ", std::to_string(type_id),
      " and context name: ", name);
  if (contexts.size() == 1) {
    *context = contexts[0];
  }
  return tensorflow::Status::OK();
}


}  // namespace ml_metadata

