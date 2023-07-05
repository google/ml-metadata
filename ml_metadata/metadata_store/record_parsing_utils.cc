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
#include "ml_metadata/metadata_store/record_parsing_utils.h"

#include <glog/logging.h>
#include "google/protobuf/util/json_util.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ml_metadata/metadata_store/constants.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {
namespace {

// Parses and converts a string value to a specific field in a message.
// If the given string `value` is NULL (encoded as kMetadataSourceNull), then
// leave the field unset.
// The field should be a scalar field. The field type must be one of {string,
// int64_t, bool, enum, message}.
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

// Converts a RecordSet in the query result to a MessageType. In the record at
// the `record_index`, its value of each column is assigned to a message field
// with the same field name as the column name.
template <typename MessageType>
absl::Status ParseRecordSetToMessage(const RecordSet& record_set,
                                     const int record_index,
                                     MessageType& output_message,
                                     const CustomColumnParser& parser) {
  CHECK_LT(record_index, record_set.records_size());
  const google::protobuf::Descriptor* descriptor = output_message.descriptor();
  for (int i = 0; i < record_set.column_names_size(); i++) {
    const std::string& column_name = record_set.column_names(i);
    const google::protobuf::FieldDescriptor* field_descriptor =
        descriptor->FindFieldByName(column_name);
    const std::string& value = record_set.records(record_index).values(i);
    if (field_descriptor != nullptr) {
      MLMD_RETURN_IF_ERROR(
          ParseValueToField(field_descriptor, value, output_message));
    } else {
      MLMD_RETURN_IF_ERROR(
          parser.ParseIntoMessage(column_name, value, &output_message));
    }
  }
  return absl::OkStatus();
}

template <typename MessageType>
absl::Status ParseRecordSetToMessageArray(
    const RecordSet& record_set, std::vector<MessageType>& output_messages,
    const CustomColumnParser& parser) {
  for (int i = 0; i < record_set.records_size(); i++) {
    output_messages.push_back(MessageType());
    MLMD_RETURN_IF_ERROR(
        ParseRecordSetToMessage(record_set, i, output_messages.back(), parser));
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
    const RecordSet& record_set,
    std::vector<Association>& output_associations) {
  return ParseRecordSetToMessageArray(record_set, output_associations,
                                      CustomColumnParser());
}

}  // namespace ml_metadata
