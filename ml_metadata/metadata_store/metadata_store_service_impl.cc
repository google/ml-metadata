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
  if (!status.ok()) {
    LOG(WARNING) << "PutArtifactType failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetArtifactTypeRequest* request,
    ::ml_metadata::GetArtifactTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetArtifactType(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetArtifactType failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactTypesByID(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetArtifactTypesByIDRequest* request,
    ::ml_metadata::GetArtifactTypesByIDResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetArtifactTypesByID(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetArtifactTypesByID failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactTypes(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetArtifactTypesRequest* request,
    ::ml_metadata::GetArtifactTypesResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetArtifactTypes(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetArtifactTypes failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutExecutionType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutExecutionTypeRequest* request,
    ::ml_metadata::PutExecutionTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutExecutionType(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "PutExecutionType failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetExecutionTypeRequest* request,
    ::ml_metadata::GetExecutionTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetExecutionType(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetExecutionType failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionTypesByID(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetExecutionTypesByIDRequest* request,
    ::ml_metadata::GetExecutionTypesByIDResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetExecutionTypesByID(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetExecutionTypesByID failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionTypes(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetExecutionTypesRequest* request,
    ::ml_metadata::GetExecutionTypesResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetExecutionTypes(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetExecutionTypesByID failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutContextType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutContextTypeRequest* request,
    ::ml_metadata::PutContextTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutContextType(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "PutContextType failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetContextTypeRequest* request,
    ::ml_metadata::GetContextTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetContextType(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetContextType failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextTypesByID(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetContextTypesByIDRequest* request,
    ::ml_metadata::GetContextTypesByIDResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetContextTypesByID(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetContextTypesByID failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextTypes(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetContextTypesRequest* request,
    ::ml_metadata::GetContextTypesResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetContextTypes(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetContextTypes failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutArtifacts(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutArtifactsRequest* request,
    ::ml_metadata::PutArtifactsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutArtifacts(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "PutArtifacts failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutExecutions(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutExecutionsRequest* request,
    ::ml_metadata::PutExecutionsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutExecutions(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "PutExecutions failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactsByID(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetArtifactsByIDRequest* request,
    ::ml_metadata::GetArtifactsByIDResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetArtifactsByID(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetArtifactsByID failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionsByID(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetExecutionsByIDRequest* request,
    ::ml_metadata::GetExecutionsByIDResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetExecutionsByID(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetExecutionsByID failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutEvents(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutEventsRequest* request,
    ::ml_metadata::PutEventsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutEvents(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "PutEvents failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutExecution(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutExecutionRequest* request,
    ::ml_metadata::PutExecutionResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutExecution(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "PutExecution failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetEventsByArtifactIDs(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetEventsByArtifactIDsRequest* request,
    ::ml_metadata::GetEventsByArtifactIDsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetEventsByArtifactIDs(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetEventsByArtifactIDs failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetEventsByExecutionIDs(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetEventsByExecutionIDsRequest* request,
    ::ml_metadata::GetEventsByExecutionIDsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status = ToGRPCStatus(
      metadata_store_->GetEventsByExecutionIDs(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetEventsByExecutionIDs failed: "
                 << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifacts(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetArtifactsRequest* request,
    ::ml_metadata::GetArtifactsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetArtifacts(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetArtifacts failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactsByType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetArtifactsByTypeRequest* request,
    ::ml_metadata::GetArtifactsByTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetArtifactsByType(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetArtifactsByType failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactByTypeAndName(
      ::grpc::ServerContext* context,
      const ::ml_metadata::GetArtifactByTypeAndNameRequest* request,
      ::ml_metadata::GetArtifactByTypeAndNameResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status = ToGRPCStatus(
      metadata_store_->GetArtifactByTypeAndName(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetArtifactByTypeAndName failed: "
        << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactsByURI(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetArtifactsByURIRequest* request,
    ::ml_metadata::GetArtifactsByURIResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetArtifactsByURI(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetArtifactsByURI failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutions(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetExecutionsRequest* request,
    ::ml_metadata::GetExecutionsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetExecutions(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetExecutions failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionsByType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetExecutionsByTypeRequest* request,
    ::ml_metadata::GetExecutionsByTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetExecutionsByType(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetExecutionsByType failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionByTypeAndName(
      ::grpc::ServerContext* context,
      const ::ml_metadata::GetExecutionByTypeAndNameRequest* request,
      ::ml_metadata::GetExecutionByTypeAndNameResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status = ToGRPCStatus(
      metadata_store_->GetExecutionByTypeAndName(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetExecutionByTypeAndName failed: "
        << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutContexts(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutContextsRequest* request,
    ::ml_metadata::PutContextsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->PutContexts(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "PutContexts failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextsByID(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetContextsByIDRequest* request,
    ::ml_metadata::GetContextsByIDResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetContextsByID(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetContextsByID failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetContexts(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetContextsRequest* request,
    ::ml_metadata::GetContextsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetContexts(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetContexts failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextsByType(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetContextsByTypeRequest* request,
    ::ml_metadata::GetContextsByTypeResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetContextsByType(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetContextsByType failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextByTypeAndName(
      ::grpc::ServerContext* context,
      const ::ml_metadata::GetContextByTypeAndNameRequest* request,
      ::ml_metadata::GetContextByTypeAndNameResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status = ToGRPCStatus(
      metadata_store_->GetContextByTypeAndName(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetContextByTypeAndName failed: "
        << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::PutAttributionsAndAssociations(
    ::grpc::ServerContext* context,
    const ::ml_metadata::PutAttributionsAndAssociationsRequest* request,
    ::ml_metadata::PutAttributionsAndAssociationsResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status = ToGRPCStatus(
      metadata_store_->PutAttributionsAndAssociations(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "PutAttributionsAndAssociations failed: "
                 << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextsByArtifact(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetContextsByArtifactRequest* request,
    ::ml_metadata::GetContextsByArtifactResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetContextsByArtifact(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetContextsByArtifact failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextsByExecution(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetContextsByExecutionRequest* request,
    ::ml_metadata::GetContextsByExecutionResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetContextsByExecution(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetContextsByExecution failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactsByContext(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetArtifactsByContextRequest* request,
    ::ml_metadata::GetArtifactsByContextResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetArtifactsByContext(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetArtifactsByContext failed: " << status.error_message();
  }
  return status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionsByContext(
    ::grpc::ServerContext* context,
    const ::ml_metadata::GetExecutionsByContextRequest* request,
    ::ml_metadata::GetExecutionsByContextResponse* response) {
  absl::WriterMutexLock l(&lock_);
  const ::grpc::Status status =
      ToGRPCStatus(metadata_store_->GetExecutionsByContext(*request, response));
  if (!status.ok()) {
    LOG(WARNING) << "GetExecutionsByContext failed: " << status.error_message();
  }
  return status;
}

}  // namespace ml_metadata
