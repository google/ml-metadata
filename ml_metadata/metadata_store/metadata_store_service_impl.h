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
#ifndef ML_METADATA_METADATA_STORE_METADATA_STORE_SERVICE_IMPL_H_
#define ML_METADATA_METADATA_STORE_METADATA_STORE_SERVICE_IMPL_H_

#include "absl/synchronization/mutex.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/proto/metadata_store_service.grpc.pb.h"

namespace ml_metadata {

// A metadata store gRPC server that implements MetadataStoreService defined in
// proto/metadata_store_service.proto. It is thread-safe.
// Note, concurrent call to methods in different threads are sequential.
class MetadataStoreServiceImpl final
    : public MetadataStoreService::Service {
 public:
  explicit MetadataStoreServiceImpl(
      std::unique_ptr<MetadataStore> metadata_store);

  // default & copy constructors are disallowed.
  MetadataStoreServiceImpl() = delete;
  MetadataStoreServiceImpl(const MetadataStoreServiceImpl&) = delete;
  MetadataStoreServiceImpl& operator=(const MetadataStoreServiceImpl&) = delete;

  ::grpc::Status PutArtifactType(
      ::grpc::ServerContext* context,
      const ::ml_metadata::PutArtifactTypeRequest* request,
      ::ml_metadata::PutArtifactTypeResponse* response) override
      LOCKS_EXCLUDED(lock_);

  ::grpc::Status GetArtifactType(
      ::grpc::ServerContext* context,
      const ::ml_metadata::GetArtifactTypeRequest* request,
      ::ml_metadata::GetArtifactTypeResponse* response) override
      LOCKS_EXCLUDED(lock_);

  ::grpc::Status PutExecutionType(
      ::grpc::ServerContext* context,
      const ::ml_metadata::PutExecutionTypeRequest* request,
      ::ml_metadata::PutExecutionTypeResponse* response) override
      LOCKS_EXCLUDED(lock_);

  ::grpc::Status GetExecutionType(
      ::grpc::ServerContext* context,
      const ::ml_metadata::GetExecutionTypeRequest* request,
      ::ml_metadata::GetExecutionTypeResponse* response) override
      LOCKS_EXCLUDED(lock_);

  ::grpc::Status PutArtifacts(::grpc::ServerContext* context,
                              const ::ml_metadata::PutArtifactsRequest* request,
                              ::ml_metadata::PutArtifactsResponse* response)
      override LOCKS_EXCLUDED(lock_);

  ::grpc::Status PutExecutions(
      ::grpc::ServerContext* context,
      const ::ml_metadata::PutExecutionsRequest* request,
      ::ml_metadata::PutExecutionsResponse* response) override
      LOCKS_EXCLUDED(lock_);

  ::grpc::Status GetArtifactsByID(
      ::grpc::ServerContext* context,
      const ::ml_metadata::GetArtifactsByIDRequest* request,
      ::ml_metadata::GetArtifactsByIDResponse* response) override
      LOCKS_EXCLUDED(lock_);

  ::grpc::Status GetExecutionsByID(
      ::grpc::ServerContext* context,
      const ::ml_metadata::GetExecutionsByIDRequest* request,
      ::ml_metadata::GetExecutionsByIDResponse* response) override
      LOCKS_EXCLUDED(lock_);

  ::grpc::Status PutEvents(::grpc::ServerContext* context,
                           const ::ml_metadata::PutEventsRequest* request,
                           ::ml_metadata::PutEventsResponse* response) override
      LOCKS_EXCLUDED(lock_);

  ::grpc::Status GetEventsByArtifactIDs(
      ::grpc::ServerContext* context,
      const ::ml_metadata::GetEventsByArtifactIDsRequest* request,
      ::ml_metadata::GetEventsByArtifactIDsResponse* response) override
      LOCKS_EXCLUDED(lock_);

  ::grpc::Status GetEventsByExecutionIDs(
      ::grpc::ServerContext* context,
      const ::ml_metadata::GetEventsByExecutionIDsRequest* request,
      ::ml_metadata::GetEventsByExecutionIDsResponse* response) override
      LOCKS_EXCLUDED(lock_);

  ::grpc::Status GetArtifacts(::grpc::ServerContext* context,
                              const ::ml_metadata::GetArtifactsRequest* request,
                              ::ml_metadata::GetArtifactsResponse* response)
      override LOCKS_EXCLUDED(lock_);

  ::grpc::Status GetExecutions(
      ::grpc::ServerContext* context,
      const ::ml_metadata::GetExecutionsRequest* request,
      ::ml_metadata::GetExecutionsResponse* response) override
      LOCKS_EXCLUDED(lock_);

 private:
  absl::Mutex lock_;
  std::unique_ptr<MetadataStore> metadata_store_ GUARDED_BY(lock_);
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_METADATA_STORE_SERVICE_IMPL_H_
