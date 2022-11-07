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

#include <functional>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_source.pb.h"

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
  absl::Status Connect();

  // Closes any opened connections, and release any resource. After closing a
  // connection, new connections can be opened again.
  // Returns FAILED_PRECONDITION error, if calls Close without a connection.
  absl::Status Close();

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
  absl::Status ExecuteQuery(const std::string& query, RecordSet* results);

  // Begins (opens) a transaction.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns FAILED_PRECONDITION error, if a transaction has already begun.
  absl::Status Begin();


  // Commits a transaction.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  // Returns ABORTED error, if there is a data race detected at commit time.
  // The caller can rollback the transaction, and retry the transaction again.
  absl::Status Commit();

  // Rolls back a transaction. Undoes all uncommitted updates queries, i.e., all
  // DML queries using insert, update, delete are discarded.
  // Returns FAILED_PRECONDITION error, if Connection() is not opened.
  // Returns FAILED_PRECONDITION error, if a transaction has not begun.
  absl::Status Rollback();

  // Utility method to escape characters specific to the metadata source. The
  // returned string is used to bind text parameters for query composition. The
  // escaping characters and method depends on the metadata source backend.
  virtual std::string EscapeString(absl::string_view value) const = 0;

  // Called by QueryExecutor:EncodeBytes to use a byte encoding routine that
  // depends on the MetadataSource. Most MetadataSources would implement this as
  // std::string that simply wraps and returns the incoming `value` string_view,
  // but some SQL MetadataSources (e.g. MySQL and SQLite3) define a nontrivial
  // base64 encoding and a corresponding decoding.
  //
  // In general, for a given MetadataSource ms and any absl::string_view sv,
  //
  //   ms.DecodeBytes(ms.EncodeBytes(sv)) == std::string(sv)
  //
  virtual std::string EncodeBytes(absl::string_view value) const = 0;
  // Called by QueryExecutor:DecodeBytes to use a byte decoding routine that
  // depends on the MetadataSource. Most MetadataSources would implement this as
  // a StatusOr<std::string> that wraps the incoming value string_view, but some
  // SQL MetadataSources (e.g. MySQL and SQLite3) would attempt to decode from a
  // base64 scheme defined in EncodeBytes, and return either the decoded string
  // or an informative absl::Status indicating why decoding failed
  // (most likely absl::InvalidArgument).
  //
  // In general, for a given MetadataSource ms and any absl::string_view sv,
  //
  //   ms.DecodeBytes(ms.EncodeBytes(sv)) == std::string(sv)
  //
  virtual absl::StatusOr<std::string> DecodeBytes(
    absl::string_view value) const = 0;

  bool is_connected() const { return is_connected_; }

 protected:
  bool transaction_open() const { return transaction_open_; }

  void set_transaction_open(bool transaction_open) {
    transaction_open_ = transaction_open;
  }

 private:
  // Implementation of connecting to a backend.
  virtual absl::Status ConnectImpl() = 0;

  // Implementation of closing the current connection.
  virtual absl::Status CloseImpl() = 0;

  // Implementation of executing queries.
  virtual absl::Status ExecuteQueryImpl(const std::string& query,
                                        RecordSet* results) = 0;

  // Implementation of opening a transaction.
  virtual absl::Status BeginImpl() = 0;


  // Implementation of a transaction commit.
  virtual absl::Status CommitImpl() = 0;

  // Implementation of a transaction rollback.
  virtual absl::Status RollbackImpl() = 0;

  bool is_connected_ = false;
  bool transaction_open_ = false;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_METADATA_SOURCE_H_
