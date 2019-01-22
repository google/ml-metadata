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
#include "ml_metadata/metadata_store/metadata_access_object.h"

#include <string>
#include <vector>

#include "google/protobuf/descriptor.h"
#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {
namespace {

using Query = std::string;

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

string Bind(const google::protobuf::int64 value) {
  return std::to_string(value);
}

// Utility method to bind a boolean value to a SQL clause.
string Bind(const bool value) { return value ? "1" : "0"; }

// Utility method to bind an double value to a SQL clause.
string Bind(const double value) { return std::to_string(value); }

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

// Validates properties in a `Node` with the properties defined in a `Type`.
// `Node` is either Artifact or Execution. `Type` is either ArtifactType or
// ExecutionType.
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
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config.insert_artifact(),
      {Bind(artifact.type_id()), Bind(metadata_source, artifact.uri())},
      query));
  return tensorflow::Status::OK();
}

// Generates an insert query for Execution.
tensorflow::Status GenerateNodeCreationQuery(
    const Execution& execution, const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config.insert_execution(), {Bind(execution.type_id())}, query));
  return tensorflow::Status::OK();
}

// Generates an update query for Artifact.
tensorflow::Status GenerateNodeUpdateQuery(
    const Artifact& artifact, const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(query_config.update_artifact(),
                                {Bind(artifact.id()), Bind(artifact.type_id()),
                                 Bind(metadata_source, artifact.uri())},
                                query));
  return tensorflow::Status::OK();
}

// Generates an update query for Execution.
tensorflow::Status GenerateNodeUpdateQuery(
    const Execution& execution, const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, Query* query) {
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config.update_execution(),
      {Bind(execution.id()), Bind(execution.type_id())}, query));
  return tensorflow::Status::OK();
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
// `Node` (which is either Artifact or Execution) and the `is_custom_property`
// (which indicates the space of the given properties in the `node`).
template <typename Node>
tensorflow::Status GeneratePropertiesModificationQueries(
    const google::protobuf::Map<string, Value>& curr_properties,
    const google::protobuf::Map<string, Value>& prev_properties, const int64 node_id,
    const bool is_custom_property,
    const MetadataSourceQueryConfig& query_config,
    const MetadataSource* metadata_source, std::vector<Query>* queries) {
  static_assert(std::is_same<Node, Artifact>::value ||
                    std::is_same<Node, Execution>::value,
                "Unsupported template instantiation");
  constexpr bool is_artifact = std::is_same<Node, Artifact>::value;
  // generates delete clauses for properties in P \ C
  for (const auto& p : prev_properties) {
    const string& name = p.first;
    const Value& value = p.second;
    // check the 2-tuple (name, value_type) in prev_properties
    if (curr_properties.find(name) != curr_properties.end() &&
        curr_properties.at(name).value_case() == value.value_case())
      continue;

    Query delete_query;
    const MetadataSourceQueryConfig::TemplateQuery& delete_property =
        is_artifact ? query_config.delete_artifact_property()
                    : query_config.delete_execution_property();
    TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
        delete_property, {Bind(node_id), Bind(metadata_source, name)},
        &delete_query));
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
      const MetadataSourceQueryConfig::TemplateQuery& update_property =
          is_artifact ? query_config.update_artifact_property()
                      : query_config.update_execution_property();
      TF_RETURN_IF_ERROR(
          ComposeParameterizedQuery(update_property,
                                    {value_type, property_value, Bind(node_id),
                                     Bind(metadata_source, name)},
                                    &update_query));
      queries->push_back(update_query);
    } else {
      // generate insert clauses for properties in C \ P
      Query insert_query;
      const MetadataSourceQueryConfig::TemplateQuery& insert_property =
          is_artifact ? query_config.insert_artifact_property()
                      : query_config.insert_execution_property();
      TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
          insert_property,
          {value_type, Bind(node_id), Bind(metadata_source, name),
           Bind(is_custom_property), property_value},
          &insert_query));
      queries->push_back(insert_query);
    }
  }
  return tensorflow::Status::OK();
}

// Creates a `Type` where acceptable types are ArtifactType and ExecutionType.
// Returns INVALID_ARGUMENT error, if name field is not given.
// Returns INVALID_ARGUMENT error, if any property type is unknown.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Type>
tensorflow::Status CreateTypeImpl(const Type& type,
                                  const MetadataSourceQueryConfig& query_config,
                                  MetadataSource* metadata_source,
                                  int64* type_id) {
  constexpr bool is_artifact_type = std::is_same<Type, ArtifactType>::value;
  const string& type_name = type.name();
  const auto& type_properties = type.properties();

  // validate the given type
  if (type_name.empty())
    return tensorflow::errors::InvalidArgument("No type name is specified.");
  if (type_properties.empty())
    LOG(WARNING) << "No property is defined for the Type";

  // insert a type and get its given id
  Query insert_type;
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config.insert_type(),
      {Bind(metadata_source, type_name), Bind(is_artifact_type)},
      &insert_type));

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
    const bool is_artifact_type, const MetadataSource* metadata_source,
    Query* query_type) {
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config.select_type_by_id(),
      {Bind(condition), Bind(is_artifact_type)}, query_type));
  return tensorflow::Status::OK();
}

// Generates a query to find type by name
tensorflow::Status GenerateFindTypeQuery(
    absl::string_view condition, const MetadataSourceQueryConfig& query_config,
    const bool is_artifact_type, const MetadataSource* metadata_source,
    Query* query_type) {
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
      query_config.select_type_by_name(),
      {Bind(metadata_source, condition), Bind(is_artifact_type)}, query_type));
  return tensorflow::Status::OK();
}

// Finds a type by query conditions. Acceptable types are ArtifactType and
// ExecutionType (`MessageType`). The types can be queried by two kinds of query
// conditions, which are type id (int64) or type name (string_view).
// Returns NOT_FOUND error, if the given type_id cannot be found.
// Returns detailed INTERNAL error, if query execution fails.
template <typename QueryCondition, typename MessageType>
tensorflow::Status FindTypeImpl(const QueryCondition condition,
                                const MetadataSourceQueryConfig& query_config,
                                MetadataSource* metadata_source,
                                MessageType* type) {
  // constexpr bool is_query_by_id = std::is_same<QueryCondition, int64>::value;
  constexpr bool is_artifact_type =
      std::is_same<MessageType, ArtifactType>::value;

  Query query_type;
  TF_RETURN_IF_ERROR(GenerateFindTypeQuery(
      condition, query_config, is_artifact_type, metadata_source, &query_type));

  // Query type with the given condition
  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(
      ExecuteMultiQuery({query_type}, metadata_source, &record_sets));
  if (record_sets.front().records().empty())
    return tensorflow::errors::NotFound(
        absl::StrCat("Cannot find type: ", query_type));
  const RecordSet& type_record_set = record_sets[0];
  TF_RETURN_IF_ERROR(ParseRecordSetToMessage(type_record_set, type));

  // Query type properties
  Query query_property;
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(query_config.select_property_by_type_id(),
                                {Bind(type->id())}, &query_property));
  TF_RETURN_IF_ERROR(
      ExecuteMultiQuery({query_property}, metadata_source, &record_sets));
  const RecordSet& properties_record_set = record_sets[1];
  TF_RETURN_IF_ERROR(
      ParseRecordSetToMapField(properties_record_set, "properties", type));
  return tensorflow::Status::OK();
}

// Creates an `Node` (either `Artifact` or `Execution`), returns the assigned
// node id. The node's id field is ignored. The node should have a `NodeType`.
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
  TF_RETURN_IF_ERROR(GeneratePropertiesModificationQueries<Node>(
      node.properties(), prev_properties, *node_id,
      /*is_custom_property=*/false, query_config, metadata_source,
      &insert_node_property_queries));
  TF_RETURN_IF_ERROR(GeneratePropertiesModificationQueries<Node>(
      node.custom_properties(), prev_properties, *node_id,
      /*is_custom_property=*/true, query_config, metadata_source,
      &insert_node_property_queries));
  return ExecuteMultiQuery(insert_node_property_queries, metadata_source);
}

// Queries a `Node` (either `Artifact` or `Execution`) by an id.
// Returns NOT_FOUND error, if the given id cannot be found.
// Returns detailed INTERNAL error, if query execution fails.
template <typename Node>
tensorflow::Status FindNodeImpl(const int64 node_id,
                                const MetadataSourceQueryConfig& query_config,
                                MetadataSource* metadata_source, Node* node) {
  constexpr bool is_artifact = std::is_same<Node, Artifact>::value;
  Query find_node_query;
  const MetadataSourceQueryConfig::TemplateQuery& find_node =
      is_artifact ? query_config.select_artifact_by_id()
                  : query_config.select_execution_by_id();
  TF_RETURN_IF_ERROR(
      ComposeParameterizedQuery(find_node, {Bind(node_id)}, &find_node_query));

  Query find_property_query;
  const MetadataSourceQueryConfig::TemplateQuery& find_property =
      is_artifact ? query_config.select_artifact_property_by_artifact_id()
                  : query_config.select_execution_property_by_execution_id();
  TF_RETURN_IF_ERROR(ComposeParameterizedQuery(find_property, {Bind(node_id)},
                                               &find_property_query));

  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(ExecuteMultiQuery({find_node_query, find_property_query},
                                       metadata_source, &record_sets));

  if (record_sets.front().records_size() == 0)
    return tensorflow::errors::NotFound(
        absl::StrCat("Cannot find record by given id ", node_id));

  node->set_id(node_id);
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

template <typename Node>
tensorflow::Status FindNodeImpl(const MetadataSourceQueryConfig& query_config,
                                MetadataSource* metadata_source,
                                std::vector<Node>* nodes) {
  constexpr bool is_artifact = std::is_same<Node, Artifact>::value;
  constexpr absl::string_view find_artifacts = "select `id` from `Artifact`;";
  constexpr absl::string_view find_executions = "select `id` from `Execution`;";
  Query find_node_ids = Query(is_artifact ? find_artifacts : find_executions);
  std::vector<RecordSet> record_sets;
  TF_RETURN_IF_ERROR(
      ExecuteMultiQuery({find_node_ids}, metadata_source, &record_sets));

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

// Updates a `Node` (either Artifact or Execution).
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
  TF_RETURN_IF_ERROR(GeneratePropertiesModificationQueries<Node>(
      node.properties(), stored_node.properties(), node.id(),
      /*is_custom_property=*/false, query_config, metadata_source,
      &update_node_queries));
  TF_RETURN_IF_ERROR(GeneratePropertiesModificationQueries<Node>(
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

  return ExecuteMultiQuery(
      {drop_type_table, create_type_table, drop_properties_table,
       create_properties_table, drop_artifact_table, create_artifact_table,
       drop_artifact_property_table, create_artifact_property_table,
       drop_execution_table, create_execution_table,
       drop_execution_property_table, create_execution_property_table,
       drop_event_table, create_event_table, drop_event_path_table,
       create_event_path_table},
      metadata_source_);
}

tensorflow::Status MetadataAccessObject::InitMetadataSourceIfNotExists() {
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

  std::vector<Query> schema_check_queries = {
      check_type_table,      check_properties_table,
      check_artifact_table,  check_artifact_property_table,
      check_execution_table, check_execution_property_table,
      check_event_table,     check_event_path_table};

  std::vector<string> missing_schema_error_messages;
  for (const Query& query : schema_check_queries) {
    RecordSet record_set;
    tensorflow::Status s = metadata_source_->ExecuteQuery(query, &record_set);
    if (!s.ok()) missing_schema_error_messages.push_back(s.error_message());
  }

  // all table exists
  if (missing_schema_error_messages.empty()) return tensorflow::Status::OK();

  // some table exists, but not all
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

tensorflow::Status MetadataAccessObject::FindTypeById(
    const int64 type_id, ArtifactType* artifact_type) {
  return FindTypeImpl(type_id, query_config_, metadata_source_, artifact_type);
}

tensorflow::Status MetadataAccessObject::FindTypeById(
    const int64 type_id, ExecutionType* execution_type) {
  return FindTypeImpl(type_id, query_config_, metadata_source_, execution_type);
}

tensorflow::Status MetadataAccessObject::FindTypeByName(
    absl::string_view name, ArtifactType* artifact_type) {
  return FindTypeImpl(name, query_config_, metadata_source_, artifact_type);
}

tensorflow::Status MetadataAccessObject::FindTypeByName(
    absl::string_view name, ExecutionType* execution_type) {
  return FindTypeImpl(name, query_config_, metadata_source_, execution_type);
}

tensorflow::Status MetadataAccessObject::UpdateType(const ArtifactType& type) {
  return tensorflow::errors::Unimplemented("Not implemented yet.");
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

tensorflow::Status MetadataAccessObject::FindArtifactById(
    const int64 artifact_id, Artifact* artifact) {
  return FindNodeImpl(artifact_id, query_config_, metadata_source_, artifact);
}

tensorflow::Status MetadataAccessObject::FindExecutionById(
    const int64 execution_id, Execution* execution) {
  return FindNodeImpl(execution_id, query_config_, metadata_source_, execution);
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
                                      {Bind(event_id), "step_index",
                                       Bind(is_index_step), Bind(step.index())},
                                      &insert_event_path));
        break;
      }
      case Event::Path::Step::kKey: {
        TF_RETURN_IF_ERROR(ComposeParameterizedQuery(
            query_config_.insert_event_path(),
            {Bind(event_id), "step_key", Bind(is_index_step),
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

tensorflow::Status MetadataAccessObject::FindArtifacts(
    std::vector<Artifact>* artifacts) {
  return FindNodeImpl(query_config_, metadata_source_, artifacts);
}

tensorflow::Status MetadataAccessObject::FindExecutions(
    std::vector<Execution>* executions) {
  return FindNodeImpl(query_config_, metadata_source_, executions);
}

}  // namespace ml_metadata
