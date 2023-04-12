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
#ifndef THIRD_PARTY_ML_METADATA_METADATA_STORE_RECORD_PARSING_UTILS_H_
#define THIRD_PARTY_ML_METADATA_METADATA_STORE_RECORD_PARSING_UTILS_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"

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
// Returns OK and the parsed result is outputted by `output_artifacts`.
// Returns error when internal error happens.
absl::Status ParseRecordSetToNodeArray(
    const RecordSet& record_set, std::vector<Execution>& output_executions,
    const CustomColumnParser& parser = CustomColumnParser());

// Converts `record_set` to a Context array.
// Returns OK and the parsed result is outputted by `output_artifacts`.
// Returns error when internal error happens.
absl::Status ParseRecordSetToNodeArray(
    const RecordSet& record_set, std::vector<Context>& output_contexts,
    const CustomColumnParser& parser = CustomColumnParser());

// Converts `record_set` to an Event array.
// Returns OK and the parsed result is outputted by `output_artifacts`.
// Returns error when internal error happens.
absl::Status ParseRecordSetToEdgeArray(
    const RecordSet& record_set, std::vector<Event>& output_events,
    const CustomColumnParser& parser = CustomColumnParser());

// Converts `record_set` to an Association array.
// Returns OK and the parsed result is outputted by `output_artifacts`.
// Returns error when internal error happens.
absl::Status ParseRecordSetToEdgeArray(
    const RecordSet& record_set, std::vector<Association>& output_associations);

}  // namespace ml_metadata

#endif  // THIRD_PARTY_ML_METADATA_METADATA_STORE_RECORD_PARSING_UTILS_H_
