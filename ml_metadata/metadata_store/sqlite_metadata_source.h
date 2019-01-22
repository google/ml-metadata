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
#ifndef ML_METADATA_METADATA_STORE_SQLITE_METADATA_SOURCE_H_
#define ML_METADATA_METADATA_STORE_SQLITE_METADATA_SOURCE_H_

#include "ml_metadata/metadata_store/metadata_source.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "sqlite3.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// A MetadataSource based on Sqlite3. By default it uses a in memory Sqlite3
// database, and destroys it when the metadata source is destructed. It can be
// configured via a SqliteMetadataSourceConfig to use physical Sqlite3 and open
// it in read only, read and write, and create if not exists modes.
// This class is thread-unsafe. Multiple objects can be created by using the
// same SqliteMetadataSourceConfig to use the same Sqlite3 database.
class SqliteMetadataSource : public MetadataSource {
 public:
  explicit SqliteMetadataSource(const SqliteMetadataSourceConfig& config);
  ~SqliteMetadataSource() override;

  // Disallow copy and assign.
  SqliteMetadataSource(const SqliteMetadataSource&) = delete;
  SqliteMetadataSource& operator=(const SqliteMetadataSource&) = delete;

  // Escape strings having single quotes using built-in printf in Sqlite3 C API.
  string EscapeString(absl::string_view value) const final;

 private:
  // Creates an in memory db.
  // If error happens, Returns INTERNAL error.
  tensorflow::Status ConnectImpl() final;

  // Closes in memory db. All data stored will be cleaned up.
  tensorflow::Status CloseImpl() final;

  // Executes a SQL statement and returns the rows if any.
  tensorflow::Status ExecuteQueryImpl(const string& query,
                                      RecordSet* results) final;

  // Commits a transaction.
  tensorflow::Status CommitImpl() final;

  // Rollbacks a transaction
  tensorflow::Status RollbackImpl() final;

  // Begins a transaction
  tensorflow::Status BeginImpl() final;

  // Util methods to execute query.
  tensorflow::Status RunStatement(const string& query, RecordSet* results);

  // The sqlite3 handle to a database.
  sqlite3* db_ = nullptr;

  // A config including connection parameters.
  SqliteMetadataSourceConfig config_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_SQLITE_METADATA_SOURCE_H_
