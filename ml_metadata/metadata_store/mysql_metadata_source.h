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
#ifndef ML_METADATA_METADATA_STORE_MYSQL_METADATA_SOURCE_H_
#define ML_METADATA_METADATA_STORE_MYSQL_METADATA_SOURCE_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_source.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "mysql.h"

namespace ml_metadata {

// A MetadataSource based on a MYSQL backend.
// This class is thread-unsafe.
class MySqlMetadataSource : public MetadataSource {
 public:
  // Initializes the MySqlMetadataSource with given config.
  // Check-fails if config is invalid.
  explicit MySqlMetadataSource(const MySQLDatabaseConfig& config);

  // Disallow copy and assign.
  MySqlMetadataSource(const MySqlMetadataSource&) = delete;
  MySqlMetadataSource& operator=(const MySqlMetadataSource&) = delete;

  ~MySqlMetadataSource() override;

  // Escape strings with backslashes for special characters for mysql. The
  // implementation uses mysql_real_escape_string in MySql C API. It aborts if
  // the metadata source is not connected.
  std::string EscapeString(absl::string_view value) const final;

  // SQL sources use base64 encoding.
  std::string EncodeBytes(absl::string_view value) const final;

  // SQL sources use base64 decoding. Returns absl::Status if decoding failed.
  absl::StatusOr<std::string> DecodeBytes(absl::string_view value) const final;

 private:
  // Connects to the MYSQL backend specified in options_.
  // Returns an INTERNAL error upon any errors from the MYSQL backend.
  absl::Status ConnectImpl() final;

  // Closes the existing open connection to the MYSQL backend.
  // Any existing MYSQL_RES in `result_set_` is also cleaned up.
  absl::Status CloseImpl() final;

  // Opens a transaction.
  absl::Status BeginImpl() final;


  // Executes a SQL statement and returns the rows if any.
  // Returns an INTERNAL error upon any errors from the MYSQL backend.
  absl::Status ExecuteQueryImpl(const std::string& query,
                                RecordSet* results) final;

  // Commits the currently open transaction.
  absl::Status CommitImpl() final;

  // Rollbacks the currently open transaction.
  absl::Status RollbackImpl() final;

  // Returns an error if the default storage engine doesn't support transaction
  // or OK otherwise.
  absl::Status CheckTransactionSupport();

  // Runs the given query and stores the MYSQL_RES in result_set_.
  // Any existing MYSQL_RES in `result_set_` is cleaned up prior to issuing
  // the given query.
  // Returns an INTERNAL error upon any errors from the MYSQL backend.
  absl::Status RunQuery(const std::string& query);

  // Checks whether incoming query is a `USE <database>` query. And if so,
  // update database_name_ field.
  void MaybeUpdateDatabaseNameFromQuery(const std::string& query);

  // Discards any existing MYSQL_RES in `result_set_`.
  void DiscardResultSet();

  // Converts the MYSQL_RES in `result_set_` to `record_set_out`.
  absl::Status ConvertMySqlRowSetToRecordSet(RecordSet* record_set_out);

  // The handler for the connection to the MYSQL backend.
  // Initialized in ConnectImpl().
  MYSQL* db_ = nullptr;

  // The ResultSet from the previously executed query in RunQuery.
  MYSQL_RES* result_set_ = nullptr;

  // Config to connect to the MYSQL backend.
  const MySQLDatabaseConfig config_;

  // database_name_ stores the lasted database that the MetadataSoure has been
  // connected to through USE query. database_name_ may contains `;` at the end
  // which is not included in mysql db, and the state is used for concatenating
  // `USE` to form a valid query.
  std::string database_name_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_MYSQL_METADATA_SOURCE_H_
