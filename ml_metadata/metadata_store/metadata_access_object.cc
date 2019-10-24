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
#include "ml_metadata/metadata_store/metadata_access_object.h"
#endif

#include <string>
#include <vector>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/util/json_util.h"
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
#include "ml_metadata/metadata_store/metadata_access_object.h" // NOLINT
#endif
// clang-format on
#include "ml_metadata/proto/metadata_source.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {
namespace {

using Query = std::string;

// The node type_kind enum values used for internal storage. The enum value
// should not be modified, in order to be backward compatible with stored types.
// LINT.IfChange
enum class TypeKind { EXECUTION_TYPE = 0, ARTIFACT_TYPE = 1, CONTEXT_TYPE = 2 };
// LINT.ThenChange(../util/metadata_source_query_config.cc)

TypeKind ResolveTypeKind(const ArtifactType* const type) {
  return TypeKind::ARTIFACT_TYPE;
}

TypeKind ResolveTypeKind(const ExecutionType* const type) {
  return TypeKind::EXECUTION_TYPE;
}

TypeKind ResolveTypeKind(const ContextType* const type) {
  return TypeKind::CONTEXT_TYPE;
}

// Executes multiple queries within a transaction, either using an opened one or
// start a new one. If failed, this method returns the detailed error. This
// method does not commit and allows executing more queries within the same
// transaction. It returns `results`, an array of RecordSet corresponding to
// each query in the `queries`. If `results` is nullptr, then all query results
// if any are ignored.
tensorflow::Status ExecuteMultiQuery(
    const std::vector<Query>& queries, MetadataSource* metadata_source,
    std::vector<RecordSet>* results = nullptr) {
  for (const Query& q : queries) {
    RecordSet record_set;
    TF_RETURN_IF_ERROR(metadata_source->ExecuteQuery(q, &record_set));
    if (results != nullptr) results->push_back(record_set);
  }
  return tensorflow::Status::OK();
}

// Fills a template query with the given parameters.
tensorflow::Status ComposeParameterizedQuery(
    const MetadataSourceQueryConfig::TemplateQuery& template_query,
    const std::vector<string>& parameters, Query* result) {
  if (parameters.empty()) {
    return tensorflow::errors::InvalidArgument(
        "Template query has no parameters (at least 1 is required).");
  } else if (parameters.size() > 10) {
    return tensorflow::errors::InvalidArgument(
        "Template query has too many parameters (at most 10 is supported).");
  }
  if (template_query.parameter_num() != parameters.size()) {
    LOG(FATAL) << "Template query parameter_num does not match with given "
               << "parameters size (" << parameters.size()
               << "): " << template_query.DebugString();
  }
  std::vector<std::pair<const string, const string>> replacements;
  replacements.reserve(parameters.size());
  for (int i = 0; i < parameters.size(); i++) {
    replacements.push_back({absl::StrCat("$", i), parameters[i]});
  }
  *result = absl::StrReplaceAll(template_query.query(), replacements);
  return tensorflow::Status::OK();
}

// Parses and converts a string value to a specific field in a message.
// The field should be a scalar field. The field type can be one of {string,
// int64, bool, enum}.
tensorflow::Status ParseValueToField(
    const google::protobuf::FieldDescriptor* field_descriptor,
    const absl::string_view value, google::protobuf::Message* message) {
  const google::protobuf::Reflection* reflection = message->GetReflection();
  switch (field_descriptor->cpp_type()) {
    case google::protobuf::FieldDescriptor::CppType::CPPTYPE_STRING: {
      if (field_descriptor->is_repeated())
        reflection->AddString(message, field_descriptor, string(value));
      else
        reflection->SetString(message, field_descriptor, string(value));
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
                 string(value.begin(), value.size()), sub_message)
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
// with the same field name with the column name.
template <typename MessageType>
tensorflow::Status ParseRecordSetToMessage(const RecordSet& record_set,
                                           MessageType* message,
                                           int record_index = 0) {
  CHECK_LT(record_index, record_set.records_size());
  const google::protobuf::Descriptor* descriptor = message->descriptor();
  for (int i = 0; i < record_set.column_names_size(); i++) {
    const string& column_name = record_set.column_names(i);
    const google::protobuf::FieldDescriptor* field_descriptor =
        descriptor->FindFieldByName(column_name);
    if (field_descriptor != nullptr) {
      const string& value = record_set.records(record_index).values(i);
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
                                            const string& field_name,
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
    const string& key = record.values(0);
    const string& value = record.values(1);
    TF_RETURN_IF_ERROR(
        ParseValueToField(key_descriptor, key, map_field_message));
    TF_RETURN_IF_ERROR(
        ParseValueToField(value_descriptor, value, map_field_message));
  }

  return tensorflow::Status::OK();
}

#if (!defined(__APPLE__) && !defined(_WIN32))
string Bind(const google::protobuf::int64 value) {
  return std::to_string(value);
}
#endif

// Utility method to bind a boolean value to a SQL clause.
string Bind(const bool value) { return value ? "1" : "0"; }

// Utility method to bind an double value to a SQL clause.
string Bind(const double value) { return std::to_string(value); }

// Utility method to bind an int value to a SQL clause.
string Bind(const TypeKind value) { return std::to_string((int)value); }

// Utility method to bind an int64 value to a SQL clause.
string Bind(const int64 value) { return std::to_string(value); }

// Utility method to bind an Event::Type enum value to a SQL clause.
// Event::Type is an enum (integer), EscapeString is not applicable.
string Bind(const Event::Type type) { return std::to_string(type); }

// Utility method to bind an PropertyType enum value to a SQL clause.
// PropertyType is an enum (integer), EscapeString is not applicable.
string Bind(const PropertyType type) { return std::to_string(type); }

// Utility method to bind a string to a SQL clause. The given string is modified
// by escaping metadata_source specific characters.
string Bind(const MetadataSource* metadata_source, absl::string_view value) {
  return absl::StrCat("'", metadata_source->EscapeString(value), "'");
}

// Utility method to bind a proto message to a text field.
// The proto message is encoded as JSON (which is unicode).
// If present is false, this indicates a missing field, and is encoded as
// null.
string Bind(const MetadataSource* metadata_source, bool present,
            const google::protobuf::Message& message) {
  if (present) {
    string json_output;
    CHECK(::google::protobuf::util::MessageToJsonString(message, &json_output).ok())
        << "Could not write proto to JSON: " << message.DebugString();
    return Bind(metadata_source, json_output);
  } else {
    return "null";
  }
}

// Validates properties in a `Node` with the properties defined in a `Type`.
// `Node` is one of {`Artifact`, `Execution`, `Context`}. `Type` is one of
// {`ArtifactType`, `ExecutionType`, `ContextType`}.
// Returns INVALID_ARGUMENT error, if there is unknown or mismatched property
// w.r.t. its definition.
template <typename Node, typename Type>
tensorflow::Status ValidatePropertiesWithType(const Node& node,
                                              const Type& type) {
  const google::protobuf::Map<string, PropertyType>& type_properties = type.properties();
  for (const auto& p : node.properties()) {
    const string& property_name = p.first;
    const Value& property_value = p.second;
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

// Generates an insert query for Artifact.
tensorflow::Status GenerateNodeCreationQuery(
    const Artifact& artifact, const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  return ComposeParameterizedQuery(
      query_config.insert_artifact(),
      {Bind(artifact.type_id()), Bind(metadata_source, artifact.uri())}, query);
}

// Generates an insert query for Execution.
tensorflow::Status GenerateNodeCreationQuery(
    const Execution& execution, const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  return ComposeParameterizedQuery(query_config.insert_execution(),
                                   {Bind(execution.type_id())}, query);
}

// Generates an insert query for Context.
tensorflow::Status GenerateNodeCreationQuery(
    const Context& context, const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  if (!context.has_name() || context.name().empty()) {
    return tensorflow::errors::InvalidArgument(
        "Context name should not be empty");
  }
  return ComposeParameterizedQuery(
      query_config.insert_context(),
      {Bind(context.type_id()), Bind(metadata_source, context.name())}, query);
}

// Generates a select queries for an Artifact by id.
tensorflow::Status GenerateNodeLookupQueries(
    const Artifact& artifact, const MetadataSourceQueryConfig& query_config,
    Query* find_node_query, Query* find_property_query) {
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(query_config.select_artifact_by_id(),
                                {Bind(artifact.id())}, find_node_query));
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config.select_artifact_property_by_artifact_id(),
      {Bind(artifact.id())}, find_property_query));
  return tensorflow::Status::OK();
}

// Generates a select queries for an Execution by id.
tensorflow::Status GenerateNodeLookupQueries(
    const Execution& execution, const MetadataSourceQueryConfig& query_config,
    Query* find_node_query, Query* find_property_query) {
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(query_config.select_execution_by_id(),
                                {Bind(execution.id())}, find_node_query));
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config.select_execution_property_by_execution_id(),
      {Bind(execution.id())}, find_property_query));
  return tensorflow::Status::OK();
}

// Generates a select queries for a Context by id.
tensorflow::Status GenerateNodeLookupQueries(
    const Context& context, const MetadataSourceQueryConfig& query_config,
    Query* find_node_query, Query* find_property_query) {
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(query_config.select_context_by_id(),
                                {Bind(context.id())}, find_node_query));
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config.select_context_property_by_context_id(),
      {Bind(context.id())}, find_property_query));
  return tensorflow::Status::OK();
}

// Generates an update query for Artifact.
tensorflow::Status GenerateNodeUpdateQuery(
    const Artifact& artifact, const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  return ComposeParameterizedQuery(
      query_config.update_artifact(),
      {Bind(artifact.id()), Bind(artifact.type_id()),
       Bind(metadata_source, artifact.uri())},
      query);
}

// Generates an update query for Execution.
tensorflow::Status GenerateNodeUpdateQuery(
    const Execution& execution, const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  return ComposeParameterizedQuery(
      query_config.update_execution(),
      {Bind(execution.id()), Bind(execution.type_id())}, query);
}

// Generates an update query for Context.
tensorflow::Status GenerateNodeUpdateQuery(
    const Context& context, const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  if (!context.has_name() || context.name().empty()) {
    return tensorflow::errors::InvalidArgument(
        "Context name should not be empty");
  }
  return ComposeParameterizedQuery(query_config.update_context(),
                                   {Bind(context.id()), Bind(context.type_id()),
                                    Bind(metadata_source, context.name())},
                                   query);
}

// Generates a property insertion query for a NodeType.
template <typename NodeType>
tensorflow::Status GeneratePropertyInsertionQuery(
    const int64 node_id, const absl::string_view name,
    const bool is_custom_property, const string& binded_value_type,
    const string& binded_property_value,
    const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  NodeType node;
  const TypeKind type_kind = ResolveTypeKind(&node);
  MetadataSourceQueryConfig::TemplateQuery insert_property;
  switch (type_kind) {
    case TypeKind::ARTIFACT_TYPE: {
      insert_property = query_config.insert_artifact_property();
      break;
    }
    case TypeKind::EXECUTION_TYPE: {
      insert_property = query_config.insert_execution_property();
      break;
    }
    case TypeKind::CONTEXT_TYPE: {
      insert_property = query_config.insert_context_property();
      break;
    }
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported TypeKind: ", type_kind));
  }
  return ComposeParameterizedQuery(
      insert_property,
      {binded_value_type, Bind(node_id), Bind(metadata_source, name),
       Bind(is_custom_property), binded_property_value},
      query);
}

// Generates a property update query for a NodeType.
template <typename NodeType>
tensorflow::Status GeneratePropertyUpdateQuery(
    const int64 node_id, const absl::string_view name,
    const string& binded_value_type, const string& binded_property_value,
    const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  NodeType node;
  const TypeKind type_kind = ResolveTypeKind(&node);
  MetadataSourceQueryConfig::TemplateQuery update_property;
  switch (type_kind) {
    case TypeKind::ARTIFACT_TYPE: {
      update_property = query_config.update_artifact_property();
      break;
    }
    case TypeKind::EXECUTION_TYPE: {
      update_property = query_config.update_execution_property();
      break;
    }
    case TypeKind::CONTEXT_TYPE: {
      update_property = query_config.update_context_property();
      break;
    }
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported TypeKind: ", type_kind));
  }
  return ComposeParameterizedQuery(update_property,
                                   {binded_value_type, binded_property_value,
                                    Bind(node_id), Bind(metadata_source, name)},
                                   query);
}

// Generates a property deletion query for a NodeType.
template <typename NodeType>
tensorflow::Status GeneratePropertyDeletionQuery(
    const int64 node_id, const absl::string_view name,
    const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  NodeType type;
  const TypeKind type_kind = ResolveTypeKind(&type);
  MetadataSourceQueryConfig::TemplateQuery delete_property;
  switch (type_kind) {
    case TypeKind::ARTIFACT_TYPE: {
      delete_property = query_config.delete_artifact_property();
      break;
    }
    case TypeKind::EXECUTION_TYPE: {
      delete_property = query_config.delete_execution_property();
      break;
    }
    case TypeKind::CONTEXT_TYPE: {
      delete_property = query_config.delete_context_property();
      break;
    }
    default:
      return tensorflow::errors::Internal("Unsupported TypeKind.");
  }
  return ComposeParameterizedQuery(
      delete_property, {Bind(node_id), Bind(metadata_source, name)}, query);
}

// A utility method to derive the `property_type` and `property_value` in the
// metadata source from a property's value protobuf `value_message`.
tensorflow::Status GetPropertyTypeAndValue(
    const Value& value_message, const MetadataSource* metadata_source,
    string* property_type, string* property_value) {
  switch (value_message.value_case()) {
    case PropertyType::INT: {
      *property_type = "int_value";
      *property_value = Bind(value_message.int_value());
      break;
    }
    case PropertyType::DOUBLE: {
      *property_type = "double_value";
      *property_value = Bind(value_message.double_value());
      break;
    }
    case PropertyType::STRING: {
      *property_type = "string_value";
      *property_value = Bind(metadata_source, value_message.string_value());
      break;
    }
    default: {
      return tensorflow::errors::Internal(
          absl::StrCat("Unexpected oneof: ", value_message.DebugString()));
    }
  }
  return tensorflow::Status::OK();
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
tensorflow::Status GeneratePropertiesModificationQueries(
    const google::protobuf::Map<string, Value>& curr_properties,
    const google::protobuf::Map<string, Value>& prev_properties, const int64 node_id,
    const bool is_custom_property,
    const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, std::vector<Query>* queries) {
  // generates delete clauses for properties in P \ C
  for (const auto& p : prev_properties) {
    const string& name = p.first;
    const Value& value = p.second;
    // check the 2-tuple (name, value_type) in prev_properties
    if (curr_properties.find(name) != curr_properties.end() &&
        curr_properties.at(name).value_case() == value.value_case())
      continue;

    Query delete_query;
    TF_RETURN_IF_ERROR(GeneratePropertyDeletionQuery<NodeType>(
        node_id, name, query_config, metadata_source, &delete_query));
    queries->push_back(delete_query);
  }

  for (const auto& p : curr_properties) {
    const string& name = p.first;
    const Value& value = p.second;
    string value_type;
    string property_value;
    TF_RETURN_IF_ERROR(GetPropertyTypeAndValue(value, metadata_source,
                                               &value_type, &property_value));
    if (prev_properties.find(name) != prev_properties.end() &&
        prev_properties.at(name).value_case() == p.second.value_case()) {
      // generates update clauses for properties in the intersection P & C
      Query update_query;
      TF_RETURN_IF_ERROR(GeneratePropertyUpdateQuery<NodeType>(
          node_id, name, value_type, property_value, query_config,
          metadata_source, &update_query));
      queries->push_back(update_query);
    } else {
      // generate insert clauses for properties in C \ P
      Query insert_query;
      TF_RETURN_IF_ERROR(GeneratePropertyInsertionQuery<NodeType>(
          node_id, name, is_custom_property, value_type, property_value,
          query_config, metadata_source, &insert_query));
      queries->push_back(insert_query);
    }
  }
  return tensorflow::Status::OK();
}

// Creates a query to insert an artifact type.
tensorflow::Status ComposeInsertTypeQueryImpl(
    const ArtifactType& type, const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, Query* insert_type) {
  return ComposeParameterizedQuery(query_config.insert_artifact_type(),
                                   {Bind(metadata_source, type.name())},
                                   insert_type);
}

// Creates a query to insert an execution type.
tensorflow::Status ComposeInsertTypeQueryImpl(
    const ExecutionType& type, const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, Query* insert_type) {
  return ComposeParameterizedQuery(
      query_config.insert_execution_type(),
      {Bind(metadata_source, type.name()),
       Bind(metadata_source, type.has_input_type(), type.input_type()),
       Bind(metadata_source, type.has_output_type(), type.output_type())},
      insert_type);
}

// Creates a query to insert a context type.
tensorflow::Status ComposeInsertTypeQueryImpl(
    const ContextType& type, const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, Query* insert_type) {
  return ComposeParameterizedQuery(query_config.insert_context_type(),
                                   {Bind(metadata_source, type.name())},
                                   insert_type);
}

// Creates a `Type` where acceptable ones are in {ArtifactType, ExecutionType,
// ContextType}.
// Returns INVALID_ARGUMENT error, if name field is not given.
// Returns INVALID_ARGUMENT error, if any property type is unknown.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Type>
tensorflow::Status CreateTypeImpl(const Type& type,
                                  const MetadataSourceQueryConfig& query_config,
                                  MetadataSource* metadata_source,
                                  int64* type_id) {
  const string& type_name = type.name();
  const auto& type_properties = type.properties();

  // validate the given type
  if (type_name.empty())
    return tensorflow::errors::InvalidArgument("No type name is specified.");
  if (type_properties.empty())
    LOG(WARNING) << "No property is defined for the Type";

  // insert a type and get its given id
  Query insert_type;
  TF_RETURN_IF_ERROR(ComposeInsertTypeQueryImpl(type, query_config,
                                                metadata_source, &insert_type));

  const Query& last_type_id = query_config.select_last_insert_id().query();
  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(ExecuteMultiQuery({insert_type, last_type_id},
                                       metadata_source, &record_sets));
  CHECK(absl::SimpleAtoi(record_sets.back().records(0).values(0), type_id));

  // insert type properties and commit
  std::vector<Query> insert_property_queries;
  for (const auto& property : type_properties) {
    const string& property_name = Bind(metadata_source, property.first);
    const PropertyType property_type = property.second;
    if (property_type == PropertyType::UNKNOWN) {
      LOG(ERROR) << "Property " << property_name << "'s value type is UNKNOWN.";
      return tensorflow::errors::InvalidArgument(
          absl::StrCat("Property ", property_name, " is UNKNOWN."));
    }
    Query insert_property;
    TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
        query_config.insert_type_property(),
        {Bind(*type_id), property_name, Bind(property_type)},
        &insert_property));
    insert_property_queries.push_back(insert_property);
  }
  return ExecuteMultiQuery(insert_property_queries, metadata_source);
}

// Generates a query to find type by id
tensorflow::Status GenerateFindTypeQuery(
    const int64 condition, const MetadataSourceQueryConfig& query_config,
    const TypeKind type_kind, const MetadataSource* metadata_source,
    Query* query_type) {
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config.select_type_by_id(), {Bind(condition), Bind(type_kind)},
      query_type));
  return tensorflow::Status::OK();
}

// Generates a query to find type by name
tensorflow::Status GenerateFindTypeQuery(
    absl::string_view condition, const MetadataSourceQueryConfig& query_config,
    const TypeKind type_kind, const MetadataSource* metadata_source,
    Query* query_type) {
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config.select_type_by_name(),
      {Bind(metadata_source, condition), Bind(type_kind)}, query_type));
  return tensorflow::Status::OK();
}

// Generates a query to find all type instances.
tensorflow::Status GenerateFindAllTypeInstancesQuery(
    const MetadataSourceQueryConfig& query_config, const TypeKind type_kind,
    const MetadataSource* metadata_source, Query* query_type) {
  return ComposeParameterizedQuery(query_config.select_all_types(),
                                   {Bind(type_kind)}, query_type);
}

// FindType executes `query` to obtain a list of types of the type `MessageType`
// and returns it in `types`.
template <typename MessageType>
tensorflow::Status FindTypes(const Query& query,
                             const MetadataSourceQueryConfig& query_config,
                             MetadataSource* metadata_source,
                             std::vector<MessageType>* types) {
  // Query type with the given condition
  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(ExecuteMultiQuery({query}, metadata_source, &record_sets));

  if (record_sets.front().records().empty()) {
    return tensorflow::errors::NotFound(
        absl::StrCat("Cannot find type: ", query));
  }

  const RecordSet& type_record_set = record_sets[0];
  const int num_records = type_record_set.records_size();
  types->resize(num_records);
  for (int i = 0; i < num_records; ++i) {
    TF_RETURN_IF_ERROR(
        ParseRecordSetToMessage(type_record_set, &types->at(i), i));

    Query query_property;
    TF_RETURN_IF_ERROR(
        ComposeParameterizedQuery(query_config.select_property_by_type_id(),
                                  {Bind(types->at(i).id())}, &query_property));

    std::vector<RecordSet> property_record_sets;
    TF_RETURN_IF_ERROR(ExecuteMultiQuery({query_property}, metadata_source,
                                         &property_record_sets));

    TF_RETURN_IF_ERROR(ParseRecordSetToMapField(property_record_sets[0],
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
tensorflow::Status FindTypeImpl(const QueryCondition condition,
                                const MetadataSourceQueryConfig& query_config,
                                MetadataSource* metadata_source,
                                MessageType* type) {
  const TypeKind type_kind = ResolveTypeKind(type);
  Query query;
  TF_RETURN_IF_ERROR(GenerateFindTypeQuery(condition, query_config, type_kind,
                                           metadata_source, &query));
  std::vector<MessageType> types;
  TF_RETURN_IF_ERROR(FindTypes(query, query_config, metadata_source, &types));

  if (types.empty()) {
    return tensorflow::errors::NotFound(
        absl::StrCat("No type found for query: ", query));
  }
  *type = std::move(types[0]);
  return tensorflow::Status::OK();
}

// Finds all type instances of the type `MessageType`.
// Returns detailed INTERNAL error, if query execution fails.
template <typename MessageType>
tensorflow::Status FindAllTypeInstancesImpl(
    const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::vector<MessageType>* types) {
  MessageType type;
  const TypeKind type_kind = ResolveTypeKind(&type);
  Query query;
  TF_RETURN_IF_ERROR(GenerateFindAllTypeInstancesQuery(
      query_config, type_kind, metadata_source, &query));

  return FindTypes(query, query_config, metadata_source, types);
}

// Updates an existing type. A type is one of {ArtifactType, ExecutionType,
// ContextType}
// Returns INVALID_ARGUMENT error, if name field is not given.
// Returns INVALID_ARGUMENT error, if id field is given and is different.
// Returns INVALID_ARGUMENT error, if any property type is unknown.
// Returns ALREADY_EXISTS error, if any property type is different.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Type>
tensorflow::Status UpdateTypeImpl(const Type& type,
                                  const MetadataSourceQueryConfig& query_config,
                                  MetadataSource* metadata_source) {
  if (!type.has_name()) {
    return tensorflow::errors::InvalidArgument("No type name is specified.");
  }
  // find the current stored type and validate the id.
  Type stored_type;
  TF_RETURN_IF_ERROR(
      FindTypeImpl(type.name(), query_config, metadata_source, &stored_type));
  if (type.has_id() && type.id() != stored_type.id()) {
    return tensorflow::errors::InvalidArgument(
        "Given type id is different from the existing type: ",
        stored_type.DebugString());
  }
  // updates the list of type properties
  std::vector<Query> insert_property_queries;
  const google::protobuf::Map<string, PropertyType>& stored_properties =
      stored_type.properties();
  for (const auto& p : type.properties()) {
    const string& property_name = p.first;
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
    Query insert_property_query;
    TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
        query_config.insert_type_property(),
        {Bind(stored_type.id()), Bind(metadata_source, property_name),
         Bind(property_type)},
        &insert_property_query));
    insert_property_queries.push_back(insert_property_query);
  }
  return ExecuteMultiQuery(insert_property_queries, metadata_source);
}

// Creates an `Node`, which is one of {`Artifact`, `Execution`, `Context`},
// then returns the assigned node id. The node's id field is ignored. The node
// should have a `NodeType`, which is one of {`ArtifactType`, `ExecutionType`,
// `ContextType`}.
// Returns INVALID_ARGUMENT error, if the node does not align with its type.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Node, typename NodeType>
tensorflow::Status CreateNodeImpl(const Node& node,
                                  const MetadataSourceQueryConfig& query_config,
                                  MetadataSource* metadata_source,
                                  int64* node_id) {
  // validate type
  if (!node.has_type_id())
    return tensorflow::errors::InvalidArgument("Type id is missing.");
  const int64 type_id = node.type_id();
  NodeType node_type;
  TF_RETURN_IF_ERROR(
      FindTypeImpl(type_id, query_config, metadata_source, &node_type));

  // validate properties
  TF_RETURN_IF_ERROR(ValidatePropertiesWithType(node, node_type));

  // insert a node and get the assigned id
  Query insert_node;
  TF_RETURN_IF_ERROR(GenerateNodeCreationQuery(node, query_config,
                                               metadata_source, &insert_node));
  const Query& last_node_id = query_config.select_last_insert_id().query();
  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(ExecuteMultiQuery({insert_node, last_node_id},
                                       metadata_source, &record_sets));
  CHECK(absl::SimpleAtoi(record_sets.back().records(0).values(0), node_id));

  // insert properties
  std::vector<Query> insert_node_property_queries;
  const google::protobuf::Map<string, Value> prev_properties;
  TF_RETURN_IF_ERROR(GeneratePropertiesModificationQueries<NodeType>(
      node.properties(), prev_properties, *node_id,
      /*is_custom_property=*/false, query_config, metadata_source,
      &insert_node_property_queries));
  TF_RETURN_IF_ERROR(GeneratePropertiesModificationQueries<NodeType>(
      node.custom_properties(), prev_properties, *node_id,
      /*is_custom_property=*/true, query_config, metadata_source,
      &insert_node_property_queries));
  return ExecuteMultiQuery(insert_node_property_queries, metadata_source);
}

// Queries a `Node` which is one of {`Artifact`, `Execution`, `Context`} by
// an id.
// Returns NOT_FOUND error, if the given id cannot be found.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Node>
tensorflow::Status FindNodeImpl(const int64 node_id,
                                const MetadataSourceQueryConfig& query_config,
                                MetadataSource* metadata_source, Node* node) {
  node->set_id(node_id);
  Query find_node_query, find_property_query;
  TF_RETURN_IF_ERROR(GenerateNodeLookupQueries(
      *node, query_config, &find_node_query, &find_property_query));

  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(ExecuteMultiQuery({find_node_query, find_property_query},
                                       metadata_source, &record_sets));

  if (record_sets.front().records_size() == 0)
    return tensorflow::errors::NotFound(
        absl::StrCat("Cannot find record by given id ", node_id));

  const RecordSet& node_record_set = record_sets[0];
  TF_RETURN_IF_ERROR(ParseRecordSetToMessage(node_record_set, node));

  const RecordSet& properties_record_set = record_sets[1];
  // it is ok that there is no property associated with a node
  if (properties_record_set.records_size() == 0)
    return tensorflow::Status::OK();
  // if there are properties associated with the node, parse the returned values
  CHECK_EQ(properties_record_set.column_names_size(), 5);
  for (const RecordSet::Record& record : properties_record_set.records()) {
    const string& property_name = record.values(0);
    bool is_custom_property;
    CHECK(absl::SimpleAtob(record.values(1), &is_custom_property));
    auto& property_value =
        (is_custom_property
             ? (*node->mutable_custom_properties())[property_name]
             : (*node->mutable_properties())[property_name]);
    if (!record.values(2).empty()) {
      int64 int_value;
      CHECK(absl::SimpleAtoi(record.values(2), &int_value));
      property_value.set_int_value(int_value);
    } else if (!record.values(3).empty()) {
      double double_value;
      CHECK(absl::SimpleAtod(record.values(3), &double_value));
      property_value.set_double_value(double_value);
    } else {
      const string& string_value = record.values(4);
      property_value.set_string_value(string_value);
    }
  }
  return tensorflow::Status::OK();
}

// Queries `Node`(s) which is one of {`Artifact`, `Execution`, `Context`} whose
// id is defined by the `find_node_ids_query`.
// Returns NOT_FOUND error, if the given id cannot be found.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Node>
tensorflow::Status FindNodeByIdsQueryImpl(
    const Query& find_node_ids_query,
    const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::vector<Node>* nodes) {
  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(
      ExecuteMultiQuery({find_node_ids_query}, metadata_source, &record_sets));

  if (record_sets[0].records_size() == 0)
    return tensorflow::errors::NotFound(absl::StrCat("Cannot find any record"));

  nodes->reserve(record_sets[0].records_size());
  for (const RecordSet::Record& record : record_sets[0].records()) {
    int64 node_id;
    CHECK(absl::SimpleAtoi(record.values(0), &node_id));
    nodes->push_back(Node());
    TF_RETURN_IF_ERROR(FindNodeImpl<Node>(node_id, query_config,
                                          metadata_source, &nodes->back()));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status FindAllNodesImpl(
    const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::vector<Artifact>* nodes) {
  Query find_node_ids = "select `id` from `Artifact`;";
  return FindNodeByIdsQueryImpl(find_node_ids, query_config, metadata_source,
                                nodes);
}

tensorflow::Status FindAllNodesImpl(
    const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::vector<Execution>* nodes) {
  Query find_node_ids = "select `id` from `Execution`;";
  return FindNodeByIdsQueryImpl(find_node_ids, query_config, metadata_source,
                                nodes);
}

tensorflow::Status FindAllNodesImpl(
    const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::vector<Context>* nodes) {
  Query find_node_ids = "select `id` from `Context`;";
  return FindNodeByIdsQueryImpl(find_node_ids, query_config, metadata_source,
                                nodes);
}

tensorflow::Status FindNodesByTypeIdImpl(
    const int64 type_id, const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::vector<Artifact>* nodes) {
  Query find_node_ids_query;
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(query_config.select_artifacts_by_type_id(),
                                {Bind(type_id)}, &find_node_ids_query));
  return FindNodeByIdsQueryImpl(find_node_ids_query, query_config,
                                metadata_source, nodes);
}

tensorflow::Status FindNodesByTypeIdImpl(
    const int64 type_id, const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::vector<Execution>* nodes) {
  Query find_node_ids_query;
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(query_config.select_executions_by_type_id(),
                                {Bind(type_id)}, &find_node_ids_query));
  return FindNodeByIdsQueryImpl(find_node_ids_query, query_config,
                                metadata_source, nodes);
}

tensorflow::Status FindNodesByTypeIdImpl(
    const int64 type_id, const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::vector<Context>* nodes) {
  Query find_node_ids_query;
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(query_config.select_contexts_by_type_id(),
                                {Bind(type_id)}, &find_node_ids_query));
  return FindNodeByIdsQueryImpl(find_node_ids_query, query_config,
                                metadata_source, nodes);
}

// Updates a `Node` which is one of {`Artifact`, `Execution`, `Context`}.
// Returns INVALID_ARGUMENT error, if the node cannot be found
// Returns INVALID_ARGUMENT error, if the node does not match with its type
// Returns detailed INTERNAL error, if query execution fails.
template <typename Node, typename NodeType>
tensorflow::Status UpdateNodeImpl(const Node& node,
                                  const MetadataSourceQueryConfig& query_config,
                                  MetadataSource* metadata_source) {
  // validate node
  if (!node.has_id())
    return tensorflow::errors::InvalidArgument("No id is given.");

  Node stored_node;
  tensorflow::Status status =
      FindNodeImpl(node.id(), query_config, metadata_source, &stored_node);
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
  TF_RETURN_IF_ERROR(
      FindTypeImpl(type_id, query_config, metadata_source, &stored_type));
  TF_RETURN_IF_ERROR(ValidatePropertiesWithType(node, stored_type));

  // update artifacts, and update, insert, delete properties
  std::vector<Query> update_node_queries;
  Query update_node;
  TF_RETURN_IF_ERROR(GenerateNodeUpdateQuery(node, query_config,
                                             metadata_source, &update_node));
  update_node_queries.push_back(update_node);

  // modify properties
  TF_RETURN_IF_ERROR(GeneratePropertiesModificationQueries<NodeType>(
      node.properties(), stored_node.properties(), node.id(),
      /*is_custom_property=*/false, query_config, metadata_source,
      &update_node_queries));
  TF_RETURN_IF_ERROR(GeneratePropertiesModificationQueries<NodeType>(
      node.custom_properties(), stored_node.custom_properties(), node.id(),
      /*is_custom_property=*/true, query_config, metadata_source,
      &update_node_queries));
  return ExecuteMultiQuery(update_node_queries, metadata_source);
}

// Queries `events` associated with a `Node` (either `Artifact` or `Execution`)
// by the node id.
// Returns INVALID_ARGUMENT error, if the `events` is null.
// Returns NOT_FOUND error, if no events are found given the node id.
template <typename Node>
tensorflow::Status FindEventsByNodeImpl(
    const int64 node_id, const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::vector<Event>* events) {
  if (events == nullptr)
    return tensorflow::errors::InvalidArgument("Given events is NULL.");

  constexpr bool is_artifact = std::is_same<Node, Artifact>::value;

  // find event with the given id
  Query find_event_query;
  const MetadataSourceQueryConfig::TemplateQuery& find_event =
      is_artifact ? query_config.select_event_by_artifact_id()
                  : query_config.select_event_by_execution_id();
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(find_event, {Bind(node_id)},
                                               &find_event_query));

  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(
      ExecuteMultiQuery({find_event_query}, metadata_source, &record_sets));

  if (record_sets.front().records_size() == 0) {
    return tensorflow::errors::NotFound(
        absl::StrCat("Cannot find events by given id ", node_id));
  }
  events->reserve(record_sets.front().records_size());
  TF_RETURN_IF_ERROR(ParseRecordSetToMessageArray(record_sets.front(), events));

  if (events->empty()) return tensorflow::Status::OK();

  // get event paths
  std::vector<Query> find_event_path_queries;
  for (const RecordSet::Record& record : record_sets[0].records()) {
    int64 event_id;
    CHECK(absl::SimpleAtoi(record.values(0), &event_id));
    Query find_event_path_query;
    TF_RETURN_IF_ERROR(
        ComposeParameterizedQuery(query_config.select_event_path_by_event_id(),
                                  {Bind(event_id)}, &find_event_path_query));
    find_event_path_queries.push_back(find_event_path_query);
  }
  std::vector<RecordSet> path_record_sets;
  TF_RETURN_IF_ERROR(ExecuteMultiQuery(find_event_path_queries, metadata_source,
                                       &path_record_sets));
  // parse results
  for (int i = 0; i < path_record_sets.size(); i++) {
    if (path_record_sets[i].records_size() == 0) continue;
    Event& event = (*events)[i];
    for (const RecordSet::Record& record : path_record_sets[i].records()) {
      bool is_index_step;
      CHECK(absl::SimpleAtob(record.values(1), &is_index_step));
      if (is_index_step) {
        int64 step_index;
        CHECK(absl::SimpleAtoi(record.values(2), &step_index));
        event.mutable_path()->add_steps()->set_index(step_index);
      } else {
        event.mutable_path()->add_steps()->set_key(record.values(3));
      }
    }
  }
  return tensorflow::Status::OK();
}

// Generates the execution node query given an association edge.
tensorflow::Status GenerateNodeSelectionQuery(
    const Association& association,
    const MetadataSourceQueryConfig& query_config, Query* query) {
  if (!association.has_execution_id())
    return tensorflow::errors::InvalidArgument("No execution id is specified.");
  return ComposeParameterizedQuery(query_config.select_execution_by_id(),
                                   {Bind(association.execution_id())}, query);
}

// Generates the artifact node query given an attribution edge.
tensorflow::Status GenerateNodeSelectionQuery(
    const Attribution& attribution,
    const MetadataSourceQueryConfig& query_config, Query* query) {
  if (!attribution.has_artifact_id())
    return tensorflow::errors::InvalidArgument("No artifact id is specified.");
  return ComposeParameterizedQuery(query_config.select_artifact_by_id(),
                                   {Bind(attribution.artifact_id())}, query);
}

// Generates the context edge insertion query.
tensorflow::Status GenerateContextEdgeInsertionQuery(
    const Association& association,
    const MetadataSourceQueryConfig& query_config, Query* query) {
  return ComposeParameterizedQuery(
      query_config.insert_association(),
      {Bind(association.context_id()), Bind(association.execution_id())},
      query);
}

// Generates the context edge insertion query.
tensorflow::Status GenerateContextEdgeInsertionQuery(
    const Attribution& attribution,
    const MetadataSourceQueryConfig& query_config, Query* query) {
  return ComposeParameterizedQuery(
      query_config.insert_attribution(),
      {Bind(attribution.context_id()), Bind(attribution.artifact_id())}, query);
}

// Creates an `edge` and returns it id. The Edge is either `Association` between
// context and execution, or `Attribution` between context and artifact.
// Returns INVALID_ARGUMENT error, if any node in the edge cannot be matched.
// Returns INTERNAL error, if the same edge already exists.
template <typename Edge>
tensorflow::Status CreateContextEdgeImpl(
    const Edge& edge, const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, int64* edge_id) {
  if (!edge.has_context_id())
    return tensorflow::errors::InvalidArgument("No context id is specified.");
  Query find_context;
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(query_config.select_context_by_id(),
                                {Bind(edge.context_id())}, &find_context));
  Query find_node;
  TF_RETURN_IF_ERROR(
      GenerateNodeSelectionQuery(edge, query_config, &find_node));

  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(ExecuteMultiQuery({find_context, find_node},
                                       metadata_source, &record_sets));
  if (record_sets[0].records_size() == 0 ||
      record_sets[1].records_size() == 0) {
    return tensorflow::errors::InvalidArgument(
        "No node found with the given id ", edge.DebugString());
  }

  Query insert_edge;
  TF_RETURN_IF_ERROR(
      GenerateContextEdgeInsertionQuery(edge, query_config, &insert_edge));
  const Query& last_edge_id = query_config.select_last_insert_id().query();
  record_sets.clear();
  tensorflow::Status status = ExecuteMultiQuery({insert_edge, last_edge_id},
                                                metadata_source, &record_sets);
  if (absl::StrContains(status.error_message(), "Duplicate") ||
      absl::StrContains(status.error_message(), "UNIQUE")) {
    return tensorflow::errors::AlreadyExists(
        "Given relationship already exists: ", edge.DebugString(), status);
  }
  TF_RETURN_IF_ERROR(status);
  CHECK(absl::SimpleAtoi(record_sets.back().records(0).values(0), edge_id));
  return tensorflow::Status::OK();
}

// Queries `contexts` related to a `Node` (either `Artifact` or `Execution`)
// by the node id.
// Returns INVALID_ARGUMENT error, if the `contexts` is null.
template <typename Node>
tensorflow::Status FindContextsByNodeImpl(
    const int64 node_id, const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::vector<Context>* contexts) {
  if (contexts == nullptr)
    return tensorflow::errors::InvalidArgument("Given contexts is NULL.");

  constexpr bool is_artifact = std::is_same<Node, Artifact>::value;

  // find context ids with the given node id
  Query find_context_ids_query;
  const MetadataSourceQueryConfig::TemplateQuery& find_context_ids =
      is_artifact ? query_config.select_attribution_by_artifact_id()
                  : query_config.select_association_by_execution_id();
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      find_context_ids, {Bind(node_id)}, &find_context_ids_query));

  // get the context and its properties.
  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(ExecuteMultiQuery({find_context_ids_query},
                                       metadata_source, &record_sets));
  contexts->clear();
  for (const RecordSet::Record& record : record_sets.front().records()) {
    contexts->push_back(Context());
    Context& curr_context = contexts->back();
    TF_RETURN_IF_ERROR(
        ParseValueToField(curr_context.descriptor()->FindFieldByName("id"),
                          record.values(1), &curr_context));
    TF_RETURN_IF_ERROR(FindNodeImpl(curr_context.id(), query_config,
                                    metadata_source, &curr_context));
  }
  return tensorflow::Status::OK();
}

// Queries nodes related to a context. Node is either `Artifact` or `Execution`.
// Returns INVALID_ARGUMENT error, if the `nodes` is null.
template <typename Node>
tensorflow::Status FindNodesByContextImpl(
    const int64 context_id, const MetadataSourceQueryConfig& query_config,
    MetadataSource* metadata_source, std::vector<Node>* nodes) {
  if (nodes == nullptr)
    return tensorflow::errors::InvalidArgument("Given array is NULL.");

  constexpr bool is_artifact = std::is_same<Node, Artifact>::value;

  // find node ids with the given context id
  Query find_node_ids_query;
  const MetadataSourceQueryConfig::TemplateQuery& find_node_ids =
      is_artifact ? query_config.select_attribution_by_context_id()
                  : query_config.select_association_by_context_id();
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      find_node_ids, {Bind(context_id)}, &find_node_ids_query));

  // get the context and its properties.
  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(
      ExecuteMultiQuery({find_node_ids_query}, metadata_source, &record_sets));
  nodes->clear();
  for (const RecordSet::Record& record : record_sets.front().records()) {
    nodes->push_back(Node());
    Node& curr_node = nodes->back();
    TF_RETURN_IF_ERROR(
        ParseValueToField(curr_node.descriptor()->FindFieldByName("id"),
                          record.values(2), &curr_node));
    TF_RETURN_IF_ERROR(FindNodeImpl(curr_node.id(), query_config,
                                    metadata_source, &curr_node));
  }
  return tensorflow::Status::OK();
}

}  // namespace

tensorflow::Status MetadataAccessObject::Create(
    const MetadataSourceQueryConfig& query_config,
    MetadataSource* const metadata_source,
    std::unique_ptr<MetadataAccessObject>* result) {
  // validates query config
  if (query_config.metadata_source_type() == UNKNOWN_METADATA_SOURCE)
    return tensorflow::errors::InvalidArgument(
        "Metadata source type is not specified.");
  if (!metadata_source->is_connected())
    TF_RETURN_IF_ERROR(metadata_source->Connect());
  *result =
      absl::WrapUnique(new MetadataAccessObject(query_config, metadata_source));
  return tensorflow::Status::OK();
}

MetadataAccessObject::MetadataAccessObject(
    const MetadataSourceQueryConfig& query_config,
    MetadataSource* const connected_metadata_source)
    : query_config_(query_config),
      metadata_source_(connected_metadata_source) {}

tensorflow::Status MetadataAccessObject::InitMetadataSource() {
  const Query& drop_type_table = query_config_.drop_type_table().query();
  const Query& create_type_table = query_config_.create_type_table().query();
  const Query& drop_properties_table =
      query_config_.drop_type_property_table().query();
  const Query& create_properties_table =
      query_config_.create_type_property_table().query();
  const Query& drop_artifact_table =
      query_config_.drop_artifact_table().query();
  const Query& create_artifact_table =
      query_config_.create_artifact_table().query();
  const Query& drop_artifact_property_table =
      query_config_.drop_artifact_property_table().query();
  const Query& create_artifact_property_table =
      query_config_.create_artifact_property_table().query();
  const Query& drop_execution_table =
      query_config_.drop_execution_table().query();
  const Query& create_execution_table =
      query_config_.create_execution_table().query();
  const Query& drop_execution_property_table =
      query_config_.drop_execution_property_table().query();
  const Query& create_execution_property_table =
      query_config_.create_execution_property_table().query();
  const Query& drop_event_table = query_config_.drop_event_table().query();
  const Query& create_event_table = query_config_.create_event_table().query();
  const Query& drop_event_path_table =
      query_config_.drop_event_path_table().query();
  const Query& create_event_path_table =
      query_config_.create_event_path_table().query();
  const Query& drop_mlmd_env_table =
      query_config_.drop_mlmd_env_table().query();
  const Query& create_mlmd_env_table =
      query_config_.create_mlmd_env_table().query();
  const Query& drop_context_table = query_config_.drop_context_table().query();
  const Query& create_context_table =
      query_config_.create_context_table().query();
  const Query& drop_context_property_table =
      query_config_.drop_context_property_table().query();
  const Query& create_context_property_table =
      query_config_.create_context_property_table().query();
  const Query& drop_association_table =
      query_config_.drop_association_table().query();
  const Query& create_association_table =
      query_config_.create_association_table().query();
  const Query& drop_attribution_table =
      query_config_.drop_attribution_table().query();
  const Query& create_attribution_table =
      query_config_.create_attribution_table().query();
  // check error, if it happens, it is an internal development error.
  CHECK_GT(query_config_.schema_version(), 0);
  Query insert_schema_version;
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config_.insert_schema_version(),
      {Bind(query_config_.schema_version())}, &insert_schema_version));

  return ExecuteMultiQuery({drop_type_table,
                            create_type_table,
                            drop_properties_table,
                            create_properties_table,
                            drop_artifact_table,
                            create_artifact_table,
                            drop_artifact_property_table,
                            create_artifact_property_table,
                            drop_execution_table,
                            create_execution_table,
                            drop_execution_property_table,
                            create_execution_property_table,
                            drop_event_table,
                            create_event_table,
                            drop_event_path_table,
                            create_event_path_table,
                            drop_mlmd_env_table,
                            create_mlmd_env_table,
                            drop_context_table,
                            create_context_table,
                            drop_context_property_table,
                            create_context_property_table,
                            drop_association_table,
                            create_association_table,
                            drop_attribution_table,
                            create_attribution_table,
                            insert_schema_version},
                           metadata_source_);
}

// After 0.13.2 release, MLMD starts to have schema_version. The library always
// populates the MLMDEnv table and sets the schema_version when creating a new
// database. This method checks schema_version first, then if it exists,
// return it as `db_version`. If error occurs when reading `schema_version` from
// the MLMDEnv table, the database is either
// a) an empty database.
// b) an pre-existing database populated by 0.13.2 release.
// For a), it returns NotFound.
// For b), it set db_version as 0.
tensorflow::Status MetadataAccessObject::GetSchemaVersion(int64* db_version) {
  RecordSet record_set;
  const Query& select_schema_version =
      query_config_.check_mlmd_env_table().query();
  tensorflow::Status maybe_schema_version_status =
      metadata_source_->ExecuteQuery(select_schema_version, &record_set);
  if (maybe_schema_version_status.ok()) {
    if (record_set.records_size() != 1) {
      return tensorflow::errors::DataLoss(
          "In the given db, MLMDEnv table exists but schema_version cannot be "
          "resolved due to there being zero or more than one rows with the "
          "schema version. Expecting a single row.");
    }
    CHECK(absl::SimpleAtoi(record_set.records(0).values(0), db_version));
    return tensorflow::Status::OK();
  }
  // if MLMDEnv does not exist, it may be the v0.13.2 release or an empty db.
  const Query& check_tables_in_v0_13_2 =
      query_config_.check_tables_in_v0_13_2().query();
  tensorflow::Status maybe_v0_13_2_status =
      metadata_source_->ExecuteQuery(check_tables_in_v0_13_2, &record_set);
  if (maybe_v0_13_2_status.ok()) {
    *db_version = 0;
    return tensorflow::Status::OK();
  }
  return tensorflow::errors::NotFound("it looks an empty db is given.");
}

tensorflow::Status MetadataAccessObject::UpgradeMetadataSourceIfOutOfDate() {
  const int64 lib_version = query_config_.schema_version();
  int64 db_version = 0;
  tensorflow::Status get_schema_version_status = GetSchemaVersion(&db_version);
  // if it is an empty database, then we skip migration and create tables.
  if (tensorflow::errors::IsNotFound(get_schema_version_status)) {
    db_version = lib_version;
  } else {
    TF_RETURN_IF_ERROR(get_schema_version_status);
  }
  // we don't support downgrade a live database.
  if (db_version > lib_version) {
    return tensorflow::errors::FailedPrecondition(
        "MLMD database version ", db_version,
        " is greater than library version ", lib_version,
        ". Please upgrade the library to use the given database in order to "
        "prevent potential data loss.");
  }
  // migrate db_version to lib version
  const auto& migration_schemes = query_config_.migration_schemes();
  while (db_version < lib_version) {
    const int64 to_version = db_version + 1;
    if (migration_schemes.find(to_version) == migration_schemes.end()) {
      return tensorflow::errors::Internal(
          "Cannot find migration_schemes to version ", to_version);
    }
    std::vector<Query> upgrade_queries;
    for (const MetadataSourceQueryConfig::TemplateQuery& upgrade_query :
         migration_schemes.at(to_version).upgrade_queries()) {
      upgrade_queries.push_back(upgrade_query.query());
    }
    Query update_schema_version;
    TF_RETURN_IF_ERROR(
        ComposeParameterizedQuery(query_config_.update_schema_version(),
                                  {Bind(to_version)}, &update_schema_version));
    upgrade_queries.push_back(update_schema_version);
    std::vector<RecordSet> dummy_record_sets;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        ExecuteMultiQuery(upgrade_queries, metadata_source_,
                          &dummy_record_sets),
        "Failed to migrate existing db; the migration transaction rolls back.");
    db_version = to_version;
  }
  return tensorflow::Status::OK();
}

tensorflow::Status MetadataAccessObject::InitMetadataSourceIfNotExists() {
  // check db version, and make it to align with the lib version.
  TF_RETURN_IF_ERROR(UpgradeMetadataSourceIfOutOfDate());

  // if lib and db versions align, we check the required tables for the lib.
  const Query& check_type_table = query_config_.check_type_table().query();
  const Query& check_properties_table =
      query_config_.check_type_property_table().query();
  const Query& check_artifact_table =
      query_config_.check_artifact_table().query();
  const Query& check_artifact_property_table =
      query_config_.check_artifact_property_table().query();
  const Query& check_execution_table =
      query_config_.check_execution_table().query();
  const Query& check_execution_property_table =
      query_config_.check_execution_property_table().query();
  const Query& check_event_table = query_config_.check_event_table().query();
  const Query& check_event_path_table =
      query_config_.check_event_path_table().query();
  const Query& check_mlmd_env_table =
      query_config_.check_mlmd_env_table().query();
  const Query& check_context_table =
      query_config_.check_context_table().query();
  const Query& check_context_property_table =
      query_config_.check_context_property_table().query();
  const Query& check_association_table =
      query_config_.check_association_table().query();
  const Query& check_attribution_table =
      query_config_.check_attribution_table().query();

  std::vector<Query> schema_check_queries = {check_type_table,
                                             check_properties_table,
                                             check_artifact_table,
                                             check_artifact_property_table,
                                             check_execution_table,
                                             check_execution_property_table,
                                             check_event_table,
                                             check_event_path_table,
                                             check_mlmd_env_table,
                                             check_context_table,
                                             check_context_property_table,
                                             check_association_table,
                                             check_attribution_table};

  std::vector<string> missing_schema_error_messages;
  for (const Query& query : schema_check_queries) {
    RecordSet record_set;
    tensorflow::Status s = metadata_source_->ExecuteQuery(query, &record_set);
    if (!s.ok()) missing_schema_error_messages.push_back(s.error_message());
  }

  // all table required by the current lib version exists
  if (missing_schema_error_messages.empty()) return tensorflow::Status::OK();

  // some table exists, but not all.
  if (schema_check_queries.size() != missing_schema_error_messages.size()) {
    return tensorflow::errors::DataLoss(
        absl::StrJoin(missing_schema_error_messages, "\n"));
  }

  // no table exists, then init the MetadataSource
  return InitMetadataSource();
}

tensorflow::Status MetadataAccessObject::CreateType(const ArtifactType& type,
                                                    int64* type_id) {
  return CreateTypeImpl(type, query_config_, metadata_source_, type_id);
}

tensorflow::Status MetadataAccessObject::CreateType(const ExecutionType& type,
                                                    int64* type_id) {
  return CreateTypeImpl(type, query_config_, metadata_source_, type_id);
}

tensorflow::Status MetadataAccessObject::CreateType(const ContextType& type,
                                                    int64* type_id) {
  return CreateTypeImpl(type, query_config_, metadata_source_, type_id);
}

tensorflow::Status MetadataAccessObject::FindTypeById(
    const int64 type_id, ArtifactType* artifact_type) {
  return FindTypeImpl(type_id, query_config_, metadata_source_, artifact_type);
}

tensorflow::Status MetadataAccessObject::FindTypeById(
    const int64 type_id, ExecutionType* execution_type) {
  return FindTypeImpl(type_id, query_config_, metadata_source_, execution_type);
}

tensorflow::Status MetadataAccessObject::FindTypes(
    std::vector<ArtifactType>* artifact_types) {
  return FindAllTypeInstancesImpl(query_config_, metadata_source_,
                                  artifact_types);
}

tensorflow::Status MetadataAccessObject::FindTypeById(
    const int64 type_id, ContextType* context_type) {
  return FindTypeImpl(type_id, query_config_, metadata_source_, context_type);
}

tensorflow::Status MetadataAccessObject::FindTypes(
    std::vector<ExecutionType>* execution_types) {
  return FindAllTypeInstancesImpl(query_config_, metadata_source_,
                                  execution_types);
}

tensorflow::Status MetadataAccessObject::FindTypeByName(
    absl::string_view name, ArtifactType* artifact_type) {
  return FindTypeImpl(name, query_config_, metadata_source_, artifact_type);
}

tensorflow::Status MetadataAccessObject::FindTypeByName(
    absl::string_view name, ExecutionType* execution_type) {
  return FindTypeImpl(name, query_config_, metadata_source_, execution_type);
}

tensorflow::Status MetadataAccessObject::FindTypeByName(
    absl::string_view name, ContextType* context_type) {
  return FindTypeImpl(name, query_config_, metadata_source_, context_type);
}

tensorflow::Status MetadataAccessObject::UpdateType(const ArtifactType& type) {
  return UpdateTypeImpl(type, query_config_, metadata_source_);
}

tensorflow::Status MetadataAccessObject::UpdateType(const ExecutionType& type) {
  return UpdateTypeImpl(type, query_config_, metadata_source_);
}

tensorflow::Status MetadataAccessObject::UpdateType(const ContextType& type) {
  return UpdateTypeImpl(type, query_config_, metadata_source_);
}

tensorflow::Status MetadataAccessObject::CreateArtifact(
    const Artifact& artifact, int64* artifact_id) {
  return CreateNodeImpl<Artifact, ArtifactType>(artifact, query_config_,
                                                metadata_source_, artifact_id);
}

tensorflow::Status MetadataAccessObject::CreateExecution(
    const Execution& execution, int64* execution_id) {
  return CreateNodeImpl<Execution, ExecutionType>(
      execution, query_config_, metadata_source_, execution_id);
}

tensorflow::Status MetadataAccessObject::CreateContext(const Context& context,
                                                       int64* context_id) {
  tensorflow::Status status = CreateNodeImpl<Context, ContextType>(
      context, query_config_, metadata_source_, context_id);
  if (absl::StrContains(status.error_message(), "Duplicate") ||
      absl::StrContains(status.error_message(), "UNIQUE")) {
    return tensorflow::errors::AlreadyExists(
        "Given node already exists: ", context.DebugString(), status);
  }
  return status;
}

tensorflow::Status MetadataAccessObject::FindArtifactById(
    const int64 artifact_id, Artifact* artifact) {
  return FindNodeImpl(artifact_id, query_config_, metadata_source_, artifact);
}

tensorflow::Status MetadataAccessObject::FindExecutionById(
    const int64 execution_id, Execution* execution) {
  return FindNodeImpl(execution_id, query_config_, metadata_source_, execution);
}

tensorflow::Status MetadataAccessObject::FindContextById(const int64 context_id,
                                                         Context* context) {
  return FindNodeImpl(context_id, query_config_, metadata_source_, context);
}

tensorflow::Status MetadataAccessObject::UpdateArtifact(
    const Artifact& artifact) {
  return UpdateNodeImpl<Artifact, ArtifactType>(artifact, query_config_,
                                                metadata_source_);
}

tensorflow::Status MetadataAccessObject::UpdateExecution(
    const Execution& execution) {
  return UpdateNodeImpl<Execution, ExecutionType>(execution, query_config_,
                                                  metadata_source_);
}

tensorflow::Status MetadataAccessObject::UpdateContext(const Context& context) {
  return UpdateNodeImpl<Context, ContextType>(context, query_config_,
                                              metadata_source_);
}

tensorflow::Status MetadataAccessObject::CreateEvent(const Event& event,
                                                     int64* event_id) {
  // validate the given event
  if (!event.has_artifact_id())
    return tensorflow::errors::InvalidArgument("No artifact id is specified.");
  if (!event.has_execution_id())
    return tensorflow::errors::InvalidArgument("No execution id is specified.");
  if (!event.has_type() || event.type() == Event::UNKNOWN)
    return tensorflow::errors::InvalidArgument("No event type is specified.");
  Query find_artifact, find_execution;
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(query_config_.select_artifact_by_id(),
                                {Bind(event.artifact_id())}, &find_artifact));
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(query_config_.select_execution_by_id(),
                                {Bind(event.execution_id())}, &find_execution));
  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(ExecuteMultiQuery({find_artifact, find_execution},
                                       metadata_source_, &record_sets));
  if (record_sets[0].records_size() == 0)
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("No artifact with the given id ", event.artifact_id()));
  if (record_sets[1].records_size() == 0)
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("No execution with the given id ", event.execution_id()));

  // insert an event and get its given id
  Query insert_event;
  int64 event_time = event.has_milliseconds_since_epoch()
                         ? event.milliseconds_since_epoch()
                         : absl::ToUnixMillis(absl::Now());
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config_.insert_event(),
      {Bind(event.artifact_id()), Bind(event.execution_id()),
       Bind(event.type()), Bind(event_time)},
      &insert_event));
  const Query& last_event_id = query_config_.select_last_insert_id().query();
  record_sets.clear();
  TF_RETURN_IF_ERROR(ExecuteMultiQuery({insert_event, last_event_id},
                                       metadata_source_, &record_sets));
  CHECK(absl::SimpleAtoi(record_sets.back().records(0).values(0), event_id));

  if (!event.has_path() || event.path().steps_size() == 0)
    return tensorflow::Status::OK();

  // insert event paths
  std::vector<Query> insert_event_paths;
  insert_event_paths.reserve(event.path().steps_size());
  for (const Event::Path::Step& step : event.path().steps()) {
    Query insert_event_path;
    const bool is_index_step = step.has_index();
    // step value oneof
    switch (step.value_case()) {
      case Event::Path::Step::kIndex: {
        TF_RETURN_IF_ERROR(
            ComposeParameterizedQuery(query_config_.insert_event_path(),
                                      {Bind(*event_id), "step_index",
                                       Bind(is_index_step), Bind(step.index())},
                                      &insert_event_path));
        break;
      }
      case Event::Path::Step::kKey: {
        TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
            query_config_.insert_event_path(),
            {Bind(*event_id), "step_key", Bind(is_index_step),
             Bind(metadata_source_, step.key())},
            &insert_event_path));
        break;
      }
      default:
        LOG(FATAL) << "Unknown Event::Path::Step value case.";
    }
    insert_event_paths.push_back(insert_event_path);
  }
  return ExecuteMultiQuery(insert_event_paths, metadata_source_);
}

tensorflow::Status MetadataAccessObject::FindEventsByArtifact(
    const int64 artifact_id, std::vector<Event>* events) {
  return FindEventsByNodeImpl<Artifact>(artifact_id, query_config_,
                                        metadata_source_, events);
}

tensorflow::Status MetadataAccessObject::FindEventsByExecution(
    const int64 execution_id, std::vector<Event>* events) {
  return FindEventsByNodeImpl<Execution>(execution_id, query_config_,
                                         metadata_source_, events);
}

tensorflow::Status MetadataAccessObject::CreateAssociation(
    const Association& association, int64* association_id) {
  return CreateContextEdgeImpl(association, query_config_, metadata_source_,
                               association_id);
}

tensorflow::Status MetadataAccessObject::FindContextsByExecution(
    int64 execution_id, std::vector<Context>* contexts) {
  return FindContextsByNodeImpl<Execution>(execution_id, query_config_,
                                           metadata_source_, contexts);
}

tensorflow::Status MetadataAccessObject::FindExecutionsByContext(
    int64 context_id, std::vector<Execution>* executions) {
  return FindNodesByContextImpl(context_id, query_config_, metadata_source_,
                                executions);
}

tensorflow::Status MetadataAccessObject::CreateAttribution(
    const Attribution& attribution, int64* attribution_id) {
  return CreateContextEdgeImpl(attribution, query_config_, metadata_source_,
                               attribution_id);
}

tensorflow::Status MetadataAccessObject::FindContextsByArtifact(
    int64 artifact_id, std::vector<Context>* contexts) {
  return FindContextsByNodeImpl<Artifact>(artifact_id, query_config_,
                                          metadata_source_, contexts);
}

tensorflow::Status MetadataAccessObject::FindArtifactsByContext(
    int64 context_id, std::vector<Artifact>* artifacts) {
  return FindNodesByContextImpl(context_id, query_config_, metadata_source_,
                                artifacts);
}

tensorflow::Status MetadataAccessObject::FindArtifacts(
    std::vector<Artifact>* artifacts) {
  return FindAllNodesImpl(query_config_, metadata_source_, artifacts);
}

tensorflow::Status MetadataAccessObject::FindArtifactsByTypeId(
    const int64 type_id, std::vector<Artifact>* artifacts) {
  return FindNodesByTypeIdImpl(type_id, query_config_, metadata_source_,
                               artifacts);
}

tensorflow::Status MetadataAccessObject::FindExecutions(
    std::vector<Execution>* executions) {
  return FindAllNodesImpl(query_config_, metadata_source_, executions);
}

tensorflow::Status MetadataAccessObject::FindExecutionsByTypeId(
    const int64 type_id, std::vector<Execution>* executions) {
  return FindNodesByTypeIdImpl(type_id, query_config_, metadata_source_,
                               executions);
}

tensorflow::Status MetadataAccessObject::FindContexts(
    std::vector<Context>* contexts) {
  return FindAllNodesImpl(query_config_, metadata_source_, contexts);
}

tensorflow::Status MetadataAccessObject::FindContextsByTypeId(
    const int64 type_id, std::vector<Context>* contexts) {
  return FindNodesByTypeIdImpl(type_id, query_config_, metadata_source_,
                               contexts);
}

tensorflow::Status MetadataAccessObject::FindArtifactsByURI(
    const absl::string_view uri, std::vector<Artifact>* artifacts) {
  Query find_node_ids_query;
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config_.select_artifacts_by_uri(), {Bind(metadata_source_, uri)},
      &find_node_ids_query));
  return FindNodeByIdsQueryImpl(find_node_ids_query, query_config_,
                                metadata_source_, artifacts);
}

}  // namespace ml_metadata
