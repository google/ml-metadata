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
#include "ml_metadata/metadata_store/metadata_store_service_impl.h"

#include "grpcpp/support/status_code_enum.h"
#include "absl/synchronization/mutex.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ml_metadata {
namespace {

// Converts from tensorflow Status to GRPC Status.
::grpc::Status ToGRPCStatus(const ::tensorflow::Status& status) {
  // Note: the tensorflow and grpc status codes align with each other.
  return ::grpc::Status(static_cast<::grpc::StatusCode>(status.code()),
                        status.error_message());
}

}  // namespace

MetadataStoreServiceImpl::MetadataStoreServiceImpl(
    std::unique_ptr<MetadataStore> metadata_store)
    : metadata_store_(std::move(metadata_store)) {
  CHECK(metadata_store_ != nullptr);
  TF_CHECK_OK(metadata_store_->InitMetadataStoreIfNotExists());
}

::grpc::Status MetadataStoreServiceImpl::PutArtifactType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutArtifactTypeRequest* request,
    ::ml_metadata::PutArtifactTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutArtifactType(*request, response));
  if (!status.ok())
    LOG(WARNING) << "PutArtifactType failed: " << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetArtifactTypeRequest* request,
    ::ml_metadata::GetArtifactTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetArtifactType(*request, response));
  if (!status.ok())
    LOG(WARNING) << "GetArtifactType failed: " << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutExecutionType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutExecutionTypeRequest* request,
    ::ml_metadata::PutExecutionTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutExecutionType(*request, response));
  if (!status.ok())
    LOG(WARNING) << "PutExecutionType failed: " << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetExecutionTypeRequest* request,
    ::ml_metadata::GetExecutionTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetExecutionType(*request, response));
  if (!status.ok())
    LOG(WARNING) << "GetExecutionType failed: " << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutArtifacts(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutArtifactsRequest* request,
    ::ml_metadata::PutArtifactsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutArtifacts(*request, response));
  if (!status.ok())
    LOG(WARNING) << "PutArtifacts failed: " << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutExecutions(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutExecutionsRequest* request,
    ::ml_metadata::PutExecutionsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutExecutions(*request, response));
  if (!status.ok())
    LOG(WARNING) << "PutExecutions failed: " << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactsByID(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetArtifactsByIDRequest* request,
    ::ml_metadata::GetArtifactsByIDResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetArtifactsByID(*request, response));
  if (!status.ok())
    LOG(WARNING) << "GetArtifactsByID failed: " << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionsByID(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetExecutionsByIDRequest* request,
    ::ml_metadata::GetExecutionsByIDResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetExecutionsByID(*request, response));
  if (!status.ok())
    LOG(WARNING) << "GetExecutionsByID failed: " << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutEvents(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutEventsRequest* request,
    ::ml_metadata::PutEventsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutEvents(*request, response));
  if (!status.ok())
    LOG(WARNING) << "PutEvents failed: " << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetEventsByArtifactIDs(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetEventsByArtifactIDsRequest* request,
    ::ml_metadata::GetEventsByArtifactIDsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetEventsByArtifactIDs(*request, response));
  if (!status.ok())
    LOG(WARNING) << "GetEventsByArtifactIDs failed: " << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetEventsByExecutionIDs(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetEventsByExecutionIDsRequest* request,
    ::ml_metadata::GetEventsByExecutionIDsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status = ToGRPCStatus(
      metadata_store_->GetEventsByExecutionIDs(*request, response));
  if (!status.ok())
    LOG(WARNING) << "GetEventsByExecutionIDs failed: "
                 << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifacts(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetArtifactsRequest* request,
    ::ml_metadata::GetArtifactsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetArtifacts(*request, response));
  if (!status.ok())
    LOG(WARNING) << "GetArtifacts failed: " << status.error_message();
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutions(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetExecutionsRequest* request,
    ::ml_metadata::GetExecutionsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetExecutions(*request, response));
  if (!status.ok())
    LOG(WARNING) << "GetExecutions failed: " << status.error_message();
  return status;
}

}  // namespace ml_metadata
