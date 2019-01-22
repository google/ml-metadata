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
#ifndef ML_METADATA_METADATA_STORE_METADATA_SOURCE_H_
#define ML_METADATA_METADATA_STORE_METADATA_SOURCE_H_

#include <memory>
#include <string>

#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_source.pb.h"

#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// The base class for all metadata data sources. It provides an interface used
// by MetadataAccessObject. Each concrete MetadataSource provides a physical
// backend to persist and query metadata. An implementation of MetadataSource
// must implement transactions, and MetadataSource queries must be executed
// within transactions.
//
// Usage example:
//
//    SomeConcreteMetadataSource src;
//    TF_CHECK_OK(src.Connect());
//    TF_CHECK_OK(src.Begin());
//    TF_CHECK_OK(src.ExecuteQuery("create table foo(bar int)", nullptr));
//    TF_CHECK_OK(src.ExecuteQuery("insert into foo values (1)", nullptr));
//    RecordSet results;
//    TF_CHECK_OK(src.ExecuteQuery("select * from foo", &results));
//    for (const string c_name: results.column_names()) {
//      // process column  name
//    }
//    for (const Record& row: results.records) {
//       // process row values
//    }
//    TF_CHECK_OK(src.Commit());
//    TF_CHECK_OK(src.Close());
//
// In order to avoid unpaired Begin() and Commit(), the user can access
// the MetadataSource using ScopedTransaction below.
class MetadataSource {
 public:
  MetadataSource() = default;
  // Releases opened resources if any during destruction.
  virtual ~MetadataSource() = default;

  // Disallows copy.
  MetadataSource(const MetadataSource&) = delete;
  MetadataSource& operator=(const MetadataSource&) = delete;

  // Establishes connection to the physical data source. This method should be
  // called before other methods to store or query metadata from the datasource.
  // Returns FAILED_PRECONDITION error, if calls Connect again without Close.
  tensorflow::Status Connect();

  // Closes any opened connections, and release any resource. After closing a
  // connection, new connections can be opened again.
  // Returns FAILED_PRECONDITION error, if calls Close without a connection.
  tensorflow::Status Close();

  // Runs DDL and DML query on data source. If the data source supports
  // transactions, each query is executed within one transaction by default.
  // A more complicated transaction across multiple queries can be supported by
  // setting is_auto_comit to false; then the caller is responsible to
  // call Commit and Rollback respectively after multiple ExecuteQuery.
  //
  // Results are consist of zero or more rows represented in RecordSet.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns detailed INTERNAL error, if query execution fails.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  tensorflow::Status ExecuteQuery(const string& query, RecordSet* results);

  // Begins (opens) a transaction.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns FAILED_PRECONDITION error, if a transaction has already begun.
  tensorflow::Status Begin();

  // Commits a transaction.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  tensorflow::Status Commit();

  // Rolls back a transaction. Undoes all uncommitted updates queries, i.e., all
  // DML queries using insert, update, delete are discarded.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  tensorflow::Status Rollback();

  // Utility method to escape characters specific to the metadata source. The
  // returned string is used to bind text parameters for query composition. The
  // escaping characters and method depends on the metadata source backend.
  virtual string EscapeString(absl::string_view value) const = 0;

  bool is_connected() const { return is_connected_; }

 private:
  // Implementation of connecting to a backend.
  virtual tensorflow::Status ConnectImpl() = 0;

  // Implementation of closing the current connection.
  virtual tensorflow::Status CloseImpl() = 0;

  // Implementation of executing queries.
  virtual tensorflow::Status ExecuteQueryImpl(const string& query,
                                              RecordSet* results) = 0;

  // Implementation of opening a transaction.
  virtual tensorflow::Status BeginImpl() = 0;

  // Implementation of a transaction commit.
  virtual tensorflow::Status CommitImpl() = 0;

  // Implementation of a transaction rollback.
  virtual tensorflow::Status RollbackImpl() = 0;

  bool is_connected_ = false;
  bool transaction_open_ = false;
};

// A scoped transaction. When it is destroyed, if Commit has not been called,
// the destructor rolls back the transaction. Commit() should be called exactly
// once.
// Usage:
// MetadataSource my_source = ...;
// {
//   ScopedTransaction transaction(&my_source);
//   my_source.ExecuteQuery(...)
//   if (...) {
//     return;  // First transaction rolled back.
//   }
//   my_source.ExecuteQuery(...);
//   transaction.Commit(); // If commit fails, rollback occurs.
// }
class ScopedTransaction {
 public:
  // MetadataSource should outlast the scoped transaction, and should be
  // connected before the scoped transaction is created.
  // During the lifetime of the scoped transaction object, the user should
  // limit calls made directly on the MetadataSource to executions.
  // The user should not:
  //   1. Commit() the metadata_source.
  //   2. Rollback() the metadata_source.
  //   3. Close() the metadata_source.
  ScopedTransaction(MetadataSource* metadata_source);

  // If the transaction is not committed, it is rolled back.
  ~ScopedTransaction();

  // Commit the transaction.
  // If there is a failure during the commit, the commit_ flag is not
  // set, resulting in a Rollback().
  // Should be called no more than once on a transaction.
  tensorflow::Status Commit();

 private:
  // True iff the transaction has been committed.
  bool committed_;
  // Does not own the metadata_source_.
  // Used for beginning, rolling back, and committing a transaction.
  MetadataSource* metadata_source_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_METADATA_SOURCE_H_
