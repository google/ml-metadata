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
#ifndef THIRD_PARTY_ML_METADATA_UTIL_RECORD_PARSING_UTILS_H_
#define THIRD_PARTY_ML_METADATA_UTIL_RECORD_PARSING_UTILS_H_

#include <vector>

#include <glog/logging.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/util/return_utils.h"

namespace ml_metadata {

// Customized column parser. Different database types can implement its own
// parser to parse customized data into fields like `system_metadada`.
class CustomColumnParser {
 public:
  CustomColumnParser() = default;
  virtual ~CustomColumnParser() = default;

  virtual absl::Status ParseIntoMessage(absl::string_view column_name,
                                        absl::string_view value,
                                        google::protobuf::Message* message) const {
    return absl::OkStatus();
  }
};

// Converts `record_set` to an Artifact array.
// Returns OK and the parsed result is outputted by `output_artifacts`.
// Returns error when internal error happens.
absl::Status ParseRecordSetToNodeArray(
    const RecordSet& record_set, std::vector<Artifact>& output_artifacts,
    const CustomColumnParser& parser = CustomColumnParser());

// Converts `record_set` to an Execution array.
// Returns OK and the parsed result is outputted by `output_executions`.
// Returns error when internal error happens.
absl::Status ParseRecordSetToNodeArray(
    const RecordSet& record_set, std::vector<Execution>& output_executions,
    const CustomColumnParser& parser = CustomColumnParser());

// Converts `record_set` to a Context array.
// Returns OK and the parsed result is outputted by `output_contexts`.
// Returns error when internal error happens.
absl::Status ParseRecordSetToNodeArray(
    const RecordSet& record_set, std::vector<Context>& output_contexts,
    const CustomColumnParser& parser = CustomColumnParser());

// Converts `record_set` to an Event array.
// Returns OK and the parsed result is outputted by `output_events`.
// Returns error when internal error happens.
absl::Status ParseRecordSetToEdgeArray(
    const RecordSet& record_set, std::vector<Event>& output_events,
    const CustomColumnParser& parser = CustomColumnParser());

// Converts `record_set` to an Association array.
// Returns OK and the parsed result is outputted by `output_associations`.
// Returns error when internal error happens.
absl::Status ParseRecordSetToEdgeArray(
    const RecordSet& record_set, std::vector<Association>& output_associations,
    const CustomColumnParser& parser = CustomColumnParser());

// Converts `record_set` to an Attribution array.
// Returns OK and the parsed result is outputted by `output_attributions`.
// Returns error when internal error happens.
absl::Status ParseRecordSetToEdgeArray(
    const RecordSet& record_set, std::vector<Attribution>& output_attributions,
    const CustomColumnParser& parser = CustomColumnParser());

// Converts `record_set` to a ParentContext array.
// Returns OK and the parsed result is outputted by `output_parent_contexts`.
// Returns error when internal error happens.
absl::Status ParseRecordSetToEdgeArray(
    const RecordSet& record_set,
    std::vector<ParentContext>& output_parent_contexts,
    const CustomColumnParser& parser = CustomColumnParser());

// Extracts ArtifactType information from `node_record_set` to a deduped
// ArtifactType array.
// Returns OK and the parsed result is outputted by `output_artifact_types`.
// Returns ERROR when internal error happens.
absl::Status ParseNodeRecordSetToDedupedTypes(
    const RecordSet& node_record_set,
    std::vector<ArtifactType>& output_artifact_types,
    const CustomColumnParser& parser = CustomColumnParser());

// Extracts ExecutionType information from `node_record_set` to a deduped
// ExecutionType array.
// Returns OK and the parsed result is outputted by `output_execution_types`.
// Returns ERROR when internal error happens.
absl::Status ParseNodeRecordSetToDedupedTypes(
    const RecordSet& node_record_set,
    std::vector<ExecutionType>& output_execution_types,
    const CustomColumnParser& parser = CustomColumnParser());

// Extracts ContextType information from `node_record_set` to a deduped
// ContextType array.
// Returns OK and the parsed result is outputted by `output_context_types`.
// Returns ERROR when internal error happens.
absl::Status ParseNodeRecordSetToDedupedTypes(
    const RecordSet& node_record_set,
    std::vector<ContextType>& output_context_types,
    const CustomColumnParser& parser = CustomColumnParser());

// -----------------------------------------------------------------------------
// Implementation details follow. Parsing util users need not look further.

// Parses and converts a string value to a specific field in a message.
// If the given string `value` is NULL (encoded as kMetadataSourceNull), then
// leave the field unset.
// The field should be a scalar field. The field type must be one of {string,
// int64_t, bool, enum, message}.
absl::Status ParseValueToField(const google::protobuf::FieldDescriptor* field_descriptor,
                               absl::string_view value,
                               google::protobuf::Message& output_message);

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

// Extracts MessageType information from  `node_record_set`  to a deduped
// MessageType array.
// Returns OK and the parsed result is outputted by `output_messages`.
// Returns ERROR when internal error happens.
template <typename MessageType>
absl::Status ParseRecordSetToMessageArray(
    const RecordSet& record_set, std::vector<MessageType>& output_messages,
    const CustomColumnParser& parser = CustomColumnParser()) {
  for (int i = 0; i < record_set.records_size(); i++) {
    output_messages.push_back(MessageType());
    MLMD_RETURN_IF_ERROR(
        ParseRecordSetToMessage(record_set, i, output_messages.back(), parser));
  }
  return absl::OkStatus();
}
}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_UTIL_RECORD_PARSING_UTILS_H_
