/* Copyright 2023 Google LLC

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
#include "ml_metadata/util/record_parsing_utils.h"
#include <cstdint>
#include <string>
#include <vector>

#include <glog/logging.h>
#include "google/protobuf/util/json_util.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {
namespace {

constexpr absl::string_view kTypeTablePrefix = "type_";
constexpr absl::string_view kTypeNameAliasInNodeRecordSet = "type";
constexpr absl::string_view kTypeName = "name";

// Converts a RecordSet of Artifact/Execution/Context in the query result to
// a ArtifactType/ExecutionType/ContextType message.
template <typename MessageType>
absl::Status ParseNodeRecordSetToTypeMessage(const RecordSet& record_set,
                                             const int record_index,
                                             MessageType& output_message,
                                             const CustomColumnParser& parser) {
  CHECK_LT(record_index, record_set.records_size());
  const google::protobuf::Descriptor* descriptor = output_message.descriptor();
  for (int i = 0; i < record_set.column_names_size(); i++) {
    // Get field name of Type proto from column name.
    absl::string_view field_name;
    if (absl::StartsWith(record_set.column_names(i), kTypeTablePrefix)) {
      field_name =
          absl::StripPrefix(record_set.column_names(i), kTypeTablePrefix);
    } else if (record_set.column_names(i) == kTypeNameAliasInNodeRecordSet) {
      field_name = kTypeName;
    } else {
      continue;
    }
    const google::protobuf::FieldDescriptor* field_descriptor =
        descriptor->FindFieldByName(std::string(field_name));
    const std::string& value = record_set.records(record_index).values(i);
    if (field_descriptor != nullptr) {
      MLMD_RETURN_IF_ERROR(
          ParseValueToField(field_descriptor, value, output_message));
    } else {
      MLMD_RETURN_IF_ERROR(
          parser.ParseIntoMessage(field_name, value, &output_message));
    }
  }
  return absl::OkStatus();
}

template <typename MessageType>
absl::Status ParseNodeRecordSetToDedupedTypeMessageArray(
    const RecordSet& record_set, std::vector<MessageType>& output_messages,
    const CustomColumnParser& parser) {
  absl::flat_hash_set<int64_t> type_ids;
  for (int i = 0; i < record_set.records_size(); i++) {
    MessageType message;
    MLMD_RETURN_IF_ERROR(
        ParseNodeRecordSetToTypeMessage(record_set, i, message, parser));
    if (!type_ids.contains(message.id())) {
      output_messages.push_back(message);
      type_ids.insert(message.id());
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ParseRecordSetToNodeArray(const RecordSet& record_set,
                                       std::vector<Artifact>& output_artifacts,
                                       const CustomColumnParser& parser) {
  return ParseRecordSetToMessageArray(record_set, output_artifacts, parser);
}

absl::Status ParseRecordSetToNodeArray(
    const RecordSet& record_set, std::vector<Execution>& output_executions,
    const CustomColumnParser& parser) {
  return ParseRecordSetToMessageArray(record_set, output_executions, parser);
}

absl::Status ParseRecordSetToNodeArray(const RecordSet& record_set,
                                       std::vector<Context>& output_contexts,
                                       const CustomColumnParser& parser) {
  return ParseRecordSetToMessageArray(record_set, output_contexts, parser);
}

absl::Status ParseRecordSetToEdgeArray(const RecordSet& record_set,
                                       std::vector<Event>& output_events,
                                       const CustomColumnParser& parser) {
  return ParseRecordSetToMessageArray(record_set, output_events, parser);
}

absl::Status ParseRecordSetToEdgeArray(
    const RecordSet& record_set, std::vector<Association>& output_associations,
    const CustomColumnParser& parser) {
  return ParseRecordSetToMessageArray(record_set, output_associations, parser);
}

absl::Status ParseRecordSetToEdgeArray(
    const RecordSet& record_set, std::vector<Attribution>& output_attributions,
    const CustomColumnParser& parser) {
  return ParseRecordSetToMessageArray(record_set, output_attributions, parser);
}

absl::Status ParseRecordSetToEdgeArray(
    const RecordSet& record_set,
    std::vector<ParentContext>& output_parent_contexts,
    const CustomColumnParser& parser) {
  return ParseRecordSetToMessageArray(record_set, output_parent_contexts,
                                      parser);
}

absl::Status ParseNodeRecordSetToDedupedTypes(
    const RecordSet& node_record_set,
    std::vector<ArtifactType>& output_artifact_types,
    const CustomColumnParser& parser) {
  return ParseNodeRecordSetToDedupedTypeMessageArray(
      node_record_set, output_artifact_types, parser);
}

absl::Status ParseNodeRecordSetToDedupedTypes(
    const RecordSet& node_record_set,
    std::vector<ExecutionType>& output_execution_types,
    const CustomColumnParser& parser) {
  return ParseNodeRecordSetToDedupedTypeMessageArray(
      node_record_set, output_execution_types, parser);
}

absl::Status ParseNodeRecordSetToDedupedTypes(
    const RecordSet& node_record_set,
    std::vector<ContextType>& output_context_types,
    const CustomColumnParser& parser) {
  return ParseNodeRecordSetToDedupedTypeMessageArray(
      node_record_set, output_context_types, parser);
}

absl::Status ParseValueToField(const google::protobuf::FieldDescriptor* field_descriptor,
                               absl::string_view value,
                               google::protobuf::Message& output_message) {
  if (value == kMetadataSourceNull) {
    return absl::OkStatus();
  }
  const google::protobuf::Reflection* reflection = output_message.GetReflection();
  switch (field_descriptor->cpp_type()) {
    case google::protobuf::FieldDescriptor::CppType::CPPTYPE_STRING: {
      if (field_descriptor->is_repeated())
        reflection->AddString(&output_message, field_descriptor,
                              std::string(value));
      else
        reflection->SetString(&output_message, field_descriptor,
                              std::string(value));
      break;
    }
    case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT64: {
      int64_t int64_value;
      CHECK(absl::SimpleAtoi(value, &int64_value));
      if (field_descriptor->is_repeated())
        reflection->AddInt64(&output_message, field_descriptor, int64_value);
      else
        reflection->SetInt64(&output_message, field_descriptor, int64_value);
      break;
    }
    case google::protobuf::FieldDescriptor::CppType::CPPTYPE_BOOL: {
      bool bool_value;
      CHECK(absl::SimpleAtob(value, &bool_value));
      if (field_descriptor->is_repeated())
        reflection->AddBool(&output_message, field_descriptor, bool_value);
      else
        reflection->SetBool(&output_message, field_descriptor, bool_value);
      break;
    }
    case google::protobuf::FieldDescriptor::CppType::CPPTYPE_ENUM: {
      int enum_value;
      CHECK(absl::SimpleAtoi(value, &enum_value));
      if (field_descriptor->is_repeated())
        reflection->AddEnumValue(&output_message, field_descriptor, enum_value);
      else
        reflection->SetEnumValue(&output_message, field_descriptor, enum_value);
      break;
    }
    case google::protobuf::FieldDescriptor::CppType::CPPTYPE_MESSAGE: {
      CHECK(!field_descriptor->is_repeated())
          << "Cannot handle a repeated message";
      if (!value.empty()) {
        google::protobuf::Message* sub_message =
            reflection->MutableMessage(&output_message, field_descriptor);
        if (!google::protobuf::util::JsonStringToMessage(
                 std::string(value.begin(), value.size()), sub_message)
                 .ok()) {
          return absl::InternalError(
              absl::StrCat("Failed to parse proto: ", value));
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
}  // namespace ml_metadata
