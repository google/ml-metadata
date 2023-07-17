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

#include <glog/logging.h>
#include "absl/status/status.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"

namespace ml_metadata {
namespace {

// Converts from absl Status to GRPC Status.
::grpc::Status ToGRPCStatus(const ::absl::Status& status) {
  // Note: the absl and grpc status codes align with each other.
  return ::grpc::Status(static_cast<::grpc::StatusCode>(status.code()),
                        std::string(status.message()));
}

// Creates a store on demand. The store created does not handle migration.
::grpc::Status ConnectMetadataStore(
    const ConnectionConfig& connection_config,
    std::unique_ptr<MetadataStore>* metadata_store) {
  return ToGRPCStatus(CreateMetadataStore(connection_config, metadata_store));
}

}  // namespace

MetadataStoreServiceImpl::MetadataStoreServiceImpl(
    const ConnectionConfig& connection_config)
    : connection_config_(connection_config) {}

::grpc::Status MetadataStoreServiceImpl::PutArtifactType(
    ::grpc::ServerContext* context, const PutArtifactTypeRequest* request,
    PutArtifactTypeResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->PutArtifactType(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutArtifactType failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactType(
    ::grpc::ServerContext* context, const GetArtifactTypeRequest* request,
    GetArtifactTypeResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetArtifactType(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetArtifactType failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactTypesByID(
    ::grpc::ServerContext* context, const GetArtifactTypesByIDRequest* request,
    GetArtifactTypesByIDResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetArtifactTypesByID(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetArtifactTypesByID failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactTypes(
    ::grpc::ServerContext* context, const GetArtifactTypesRequest* request,
    GetArtifactTypesResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetArtifactTypes(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetArtifactTypes failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::PutExecutionType(
    ::grpc::ServerContext* context, const PutExecutionTypeRequest* request,
    PutExecutionTypeResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->PutExecutionType(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutExecutionType failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionType(
    ::grpc::ServerContext* context, const GetExecutionTypeRequest* request,
    GetExecutionTypeResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetExecutionType(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetExecutionType failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionTypesByID(
    ::grpc::ServerContext* context, const GetExecutionTypesByIDRequest* request,
    GetExecutionTypesByIDResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetExecutionTypesByID(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetExecutionTypesByID failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionTypes(
    ::grpc::ServerContext* context, const GetExecutionTypesRequest* request,
    GetExecutionTypesResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetExecutionTypes(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetExecutionTypesByID failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::PutContextType(
    ::grpc::ServerContext* context, const PutContextTypeRequest* request,
    PutContextTypeResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->PutContextType(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutContextType failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextType(
    ::grpc::ServerContext* context, const GetContextTypeRequest* request,
    GetContextTypeResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetContextType(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetContextType failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextTypesByID(
    ::grpc::ServerContext* context, const GetContextTypesByIDRequest* request,
    GetContextTypesByIDResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetContextTypesByID(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetContextTypesByID failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextTypes(
    ::grpc::ServerContext* context, const GetContextTypesRequest* request,
    GetContextTypesResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetContextTypes(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetContextTypes failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::PutArtifacts(
    ::grpc::ServerContext* context, const PutArtifactsRequest* request,
    PutArtifactsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->PutArtifacts(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutArtifacts failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::PutExecutions(
    ::grpc::ServerContext* context, const PutExecutionsRequest* request,
    PutExecutionsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->PutExecutions(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutExecutions failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::PutTypes(
    ::grpc::ServerContext* context, const PutTypesRequest* request,
    PutTypesResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->PutTypes(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutTypes failed: " << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactsByID(
    ::grpc::ServerContext* context, const GetArtifactsByIDRequest* request,
    GetArtifactsByIDResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetArtifactsByID(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetArtifactsByID failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionsByID(
    ::grpc::ServerContext* context, const GetExecutionsByIDRequest* request,
    GetExecutionsByIDResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetExecutionsByID(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetExecutionsByID failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::PutEvents(
    ::grpc::ServerContext* context, const PutEventsRequest* request,
    PutEventsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->PutEvents(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutEvents failed: " << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::PutExecution(
    ::grpc::ServerContext* context, const PutExecutionRequest* request,
    PutExecutionResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->PutExecution(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutExecution failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetEventsByArtifactIDs(
    ::grpc::ServerContext* context,
    const GetEventsByArtifactIDsRequest* request,
    GetEventsByArtifactIDsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetEventsByArtifactIDs(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetEventsByArtifactIDs failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetEventsByExecutionIDs(
    ::grpc::ServerContext* context,
    const GetEventsByExecutionIDsRequest* request,
    GetEventsByExecutionIDsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetEventsByExecutionIDs(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetEventsByExecutionIDs failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifacts(
    ::grpc::ServerContext* context, const GetArtifactsRequest* request,
    GetArtifactsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetArtifacts(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetArtifacts failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactsByType(
    ::grpc::ServerContext* context, const GetArtifactsByTypeRequest* request,
    GetArtifactsByTypeResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetArtifactsByType(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetArtifactsByType failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactByTypeAndName(
    ::grpc::ServerContext* context,
    const GetArtifactByTypeAndNameRequest* request,
    GetArtifactByTypeAndNameResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->GetArtifactByTypeAndName(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetArtifactByTypeAndName failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactsByURI(
    ::grpc::ServerContext* context, const GetArtifactsByURIRequest* request,
    GetArtifactsByURIResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetArtifactsByURI(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetArtifactsByURI failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactsByExternalIds(
    ::grpc::ServerContext* context,
    const GetArtifactsByExternalIdsRequest* request,
    GetArtifactsByExternalIdsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->GetArtifactsByExternalIds(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetArtifactsByExternalIds failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionsByExternalIds(
    ::grpc::ServerContext* context,
    const GetExecutionsByExternalIdsRequest* request,
    GetExecutionsByExternalIdsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->GetExecutionsByExternalIds(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetExecutionsByExternalIds failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextsByExternalIds(
    ::grpc::ServerContext* context,
    const GetContextsByExternalIdsRequest* request,
    GetContextsByExternalIdsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->GetContextsByExternalIds(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetContextsByExternalIds failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactTypesByExternalIds(
    ::grpc::ServerContext* context,
    const GetArtifactTypesByExternalIdsRequest* request,
    GetArtifactTypesByExternalIdsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->GetArtifactTypesByExternalIds(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetArtifactTypesByExternalIds failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionTypesByExternalIds(
    ::grpc::ServerContext* context,
    const GetExecutionTypesByExternalIdsRequest* request,
    GetExecutionTypesByExternalIdsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->GetExecutionTypesByExternalIds(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetExecutionTypesByExternalIds failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextTypesByExternalIds(
    ::grpc::ServerContext* context,
    const GetContextTypesByExternalIdsRequest* request,
    GetContextTypesByExternalIdsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->GetContextTypesByExternalIds(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetContextTypesByExternalIds failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutions(
    ::grpc::ServerContext* context, const GetExecutionsRequest* request,
    GetExecutionsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetExecutions(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetExecutions failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionsByType(
    ::grpc::ServerContext* context, const GetExecutionsByTypeRequest* request,
    GetExecutionsByTypeResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetExecutionsByType(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetExecutionsByType failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionByTypeAndName(
    ::grpc::ServerContext* context,
    const GetExecutionByTypeAndNameRequest* request,
    GetExecutionByTypeAndNameResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->GetExecutionByTypeAndName(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetExecutionByTypeAndName failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::PutContexts(
    ::grpc::ServerContext* context, const PutContextsRequest* request,
    PutContextsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->PutContexts(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutContexts failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextsByID(
    ::grpc::ServerContext* context, const GetContextsByIDRequest* request,
    GetContextsByIDResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetContextsByID(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetContextsByID failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetContexts(
    ::grpc::ServerContext* context, const GetContextsRequest* request,
    GetContextsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetContexts(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetContexts failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextsByType(
    ::grpc::ServerContext* context, const GetContextsByTypeRequest* request,
    GetContextsByTypeResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetContextsByType(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetContextsByType failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextByTypeAndName(
    ::grpc::ServerContext* context,
    const GetContextByTypeAndNameRequest* request,
    GetContextByTypeAndNameResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetContextByTypeAndName(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetContextByTypeAndName failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::PutAttributionsAndAssociations(
    ::grpc::ServerContext* context,
    const PutAttributionsAndAssociationsRequest* request,
    PutAttributionsAndAssociationsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->PutAttributionsAndAssociations(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutAttributionsAndAssociations failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::PutParentContexts(
    ::grpc::ServerContext* context, const PutParentContextsRequest* request,
    PutParentContextsResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->PutParentContexts(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutParentContexts failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextsByArtifact(
    ::grpc::ServerContext* context, const GetContextsByArtifactRequest* request,
    GetContextsByArtifactResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetContextsByArtifact(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetContextsByArtifact failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetContextsByExecution(
    ::grpc::ServerContext* context,
    const GetContextsByExecutionRequest* request,
    GetContextsByExecutionResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetContextsByExecution(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetContextsByExecution failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetArtifactsByContext(
    ::grpc::ServerContext* context, const GetArtifactsByContextRequest* request,
    GetArtifactsByContextResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetArtifactsByContext(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetArtifactsByContext failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetExecutionsByContext(
    ::grpc::ServerContext* context,
    const GetExecutionsByContextRequest* request,
    GetExecutionsByContextResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetExecutionsByContext(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetExecutionsByContext failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetParentContextsByContext(
    ::grpc::ServerContext* context,
    const GetParentContextsByContextRequest* request,
    GetParentContextsByContextResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->GetParentContextsByContext(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetParentContextsByContext failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetChildrenContextsByContext(
    ::grpc::ServerContext* context,
    const GetChildrenContextsByContextRequest* request,
    GetChildrenContextsByContextResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->GetChildrenContextsByContext(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetChildrenContextsByContext failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::PutLineageSubgraph(
    ::grpc::ServerContext* context, const PutLineageSubgraphRequest* request,
    PutLineageSubgraphResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status = ToGRPCStatus(
      metadata_store->PutLineageSubgraph(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "PutLineageSubgraph failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}

::grpc::Status MetadataStoreServiceImpl::GetLineageSubgraph(
    ::grpc::ServerContext* context, const GetLineageSubgraphRequest* request,
    GetLineageSubgraphResponse* response) {
  std::unique_ptr<MetadataStore> metadata_store;
  const ::grpc::Status connection_status =
      ConnectMetadataStore(connection_config_, &metadata_store);
  if (!connection_status.ok()) {
    LOG(WARNING) << "Failed to connect to the database: "
                 << connection_status.error_message();
    return connection_status;
  }
  const ::grpc::Status transaction_status =
      ToGRPCStatus(metadata_store->GetLineageSubgraph(*request, response));
  if (!transaction_status.ok()) {
    LOG(WARNING) << "GetLineageSubgraph failed: "
                 << transaction_status.error_message();
  }
  return transaction_status;
}
}  // namespace ml_metadata
