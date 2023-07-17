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

#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.grpc.pb.h"

namespace ml_metadata {

// A metadata store gRPC server that implements MetadataStoreService defined in
// proto/metadata_store_service.proto. It is thread-safe.
// Note, concurrent call to methods in different threads are sequential.
// TODO(b/123039345) Add a connection pool in metadata_store or grpc service.
class MetadataStoreServiceImpl final
    : public MetadataStoreService::Service {
 public:
  explicit MetadataStoreServiceImpl(const ConnectionConfig& connection_config);

  // default & copy constructors are disallowed.
  MetadataStoreServiceImpl() = delete;
  MetadataStoreServiceImpl(const MetadataStoreServiceImpl&) = delete;
  MetadataStoreServiceImpl& operator=(const MetadataStoreServiceImpl&) = delete;

  ::grpc::Status PutArtifactType(::grpc::ServerContext* context,
                                 const PutArtifactTypeRequest* request,
                                 PutArtifactTypeResponse* response) override;

  ::grpc::Status GetArtifactType(::grpc::ServerContext* context,
                                 const GetArtifactTypeRequest* request,
                                 GetArtifactTypeResponse* response) override;

  ::grpc::Status GetArtifactTypesByID(
      ::grpc::ServerContext* context,
      const GetArtifactTypesByIDRequest* request,
      GetArtifactTypesByIDResponse* response) override;

  ::grpc::Status GetArtifactTypes(::grpc::ServerContext* context,
                                  const GetArtifactTypesRequest* request,
                                  GetArtifactTypesResponse* response) override;

  ::grpc::Status PutExecutionType(::grpc::ServerContext* context,
                                  const PutExecutionTypeRequest* request,
                                  PutExecutionTypeResponse* response) override;

  ::grpc::Status GetExecutionType(::grpc::ServerContext* context,
                                  const GetExecutionTypeRequest* request,
                                  GetExecutionTypeResponse* response) override;

  ::grpc::Status GetExecutionTypesByID(
      ::grpc::ServerContext* context,
      const GetExecutionTypesByIDRequest* request,
      GetExecutionTypesByIDResponse* response) override;

  ::grpc::Status GetExecutionTypes(
      ::grpc::ServerContext* context, const GetExecutionTypesRequest* request,
      GetExecutionTypesResponse* response) override;

  ::grpc::Status PutContextType(::grpc::ServerContext* context,
                                const PutContextTypeRequest* request,
                                PutContextTypeResponse* response) override;

  ::grpc::Status GetContextType(::grpc::ServerContext* context,
                                const GetContextTypeRequest* request,
                                GetContextTypeResponse* response) override;

  ::grpc::Status GetContextTypesByID(
      ::grpc::ServerContext* context, const GetContextTypesByIDRequest* request,
      GetContextTypesByIDResponse* response) override;

  ::grpc::Status GetContextTypes(::grpc::ServerContext* context,
                                 const GetContextTypesRequest* request,
                                 GetContextTypesResponse* response) override;

  ::grpc::Status PutArtifacts(::grpc::ServerContext* context,
                              const PutArtifactsRequest* request,
                              PutArtifactsResponse* response) override;

  ::grpc::Status PutExecutions(::grpc::ServerContext* context,
                               const PutExecutionsRequest* request,
                               PutExecutionsResponse* response) override;

  ::grpc::Status GetArtifactsByID(::grpc::ServerContext* context,
                                  const GetArtifactsByIDRequest* request,
                                  GetArtifactsByIDResponse* response) override;

  ::grpc::Status GetExecutionsByID(
      ::grpc::ServerContext* context, const GetExecutionsByIDRequest* request,
      GetExecutionsByIDResponse* response) override;

  ::grpc::Status PutEvents(::grpc::ServerContext* context,
                           const PutEventsRequest* request,
                           PutEventsResponse* response) override;

  ::grpc::Status PutExecution(::grpc::ServerContext* context,
                              const PutExecutionRequest* request,
                              PutExecutionResponse* response) override;

  ::grpc::Status PutTypes(::grpc::ServerContext* context,
                          const PutTypesRequest* request,
                          PutTypesResponse* response) override;

  ::grpc::Status GetEventsByArtifactIDs(
      ::grpc::ServerContext* context,
      const GetEventsByArtifactIDsRequest* request,
      GetEventsByArtifactIDsResponse* response) override;

  ::grpc::Status GetEventsByExecutionIDs(
      ::grpc::ServerContext* context,
      const GetEventsByExecutionIDsRequest* request,
      GetEventsByExecutionIDsResponse* response) override;

  ::grpc::Status GetArtifacts(::grpc::ServerContext* context,
                              const GetArtifactsRequest* request,
                              GetArtifactsResponse* response) override;

  ::grpc::Status GetArtifactsByExternalIds(
      ::grpc::ServerContext* context,
      const GetArtifactsByExternalIdsRequest* request,
      GetArtifactsByExternalIdsResponse* response) override;

  ::grpc::Status GetExecutionsByExternalIds(
      ::grpc::ServerContext* context,
      const GetExecutionsByExternalIdsRequest* request,
      GetExecutionsByExternalIdsResponse* response) override;

  ::grpc::Status GetContextsByExternalIds(
      ::grpc::ServerContext* context,
      const GetContextsByExternalIdsRequest* request,
      GetContextsByExternalIdsResponse* response) override;

  ::grpc::Status GetArtifactTypesByExternalIds(
      ::grpc::ServerContext* context,
      const GetArtifactTypesByExternalIdsRequest* request,
      GetArtifactTypesByExternalIdsResponse* response) override;

  ::grpc::Status GetExecutionTypesByExternalIds(
      ::grpc::ServerContext* context,
      const GetExecutionTypesByExternalIdsRequest* request,
      GetExecutionTypesByExternalIdsResponse* response) override;

  ::grpc::Status GetContextTypesByExternalIds(
      ::grpc::ServerContext* context,
      const GetContextTypesByExternalIdsRequest* request,
      GetContextTypesByExternalIdsResponse* response) override;

  ::grpc::Status GetArtifactsByType(
      ::grpc::ServerContext* context, const GetArtifactsByTypeRequest* request,
      GetArtifactsByTypeResponse* response) override;

  ::grpc::Status GetArtifactByTypeAndName(
      ::grpc::ServerContext* context,
      const GetArtifactByTypeAndNameRequest* request,
      GetArtifactByTypeAndNameResponse* response) override;

  ::grpc::Status GetArtifactsByURI(
      ::grpc::ServerContext* context, const GetArtifactsByURIRequest* request,
      GetArtifactsByURIResponse* response) override;

  ::grpc::Status GetExecutions(::grpc::ServerContext* context,
                               const GetExecutionsRequest* request,
                               GetExecutionsResponse* response) override;

  ::grpc::Status GetExecutionsByType(
      ::grpc::ServerContext* context, const GetExecutionsByTypeRequest* request,
      GetExecutionsByTypeResponse* response) override;

  ::grpc::Status GetExecutionByTypeAndName(
      ::grpc::ServerContext* context,
      const GetExecutionByTypeAndNameRequest* request,
      GetExecutionByTypeAndNameResponse* response) override;

  ::grpc::Status PutContexts(::grpc::ServerContext* context,
                             const PutContextsRequest* request,
                             PutContextsResponse* response) override;

  ::grpc::Status GetContextsByID(::grpc::ServerContext* context,
                                 const GetContextsByIDRequest* request,
                                 GetContextsByIDResponse* response) override;

  ::grpc::Status GetContexts(::grpc::ServerContext* context,
                             const GetContextsRequest* request,
                             GetContextsResponse* response) override;

  ::grpc::Status GetContextsByType(
      ::grpc::ServerContext* context, const GetContextsByTypeRequest* request,
      GetContextsByTypeResponse* response) override;

  ::grpc::Status GetContextByTypeAndName(
      ::grpc::ServerContext* context,
      const GetContextByTypeAndNameRequest* request,
      GetContextByTypeAndNameResponse* response) override;

  ::grpc::Status PutAttributionsAndAssociations(
      ::grpc::ServerContext* context,
      const PutAttributionsAndAssociationsRequest* request,
      PutAttributionsAndAssociationsResponse* response) override;

  ::grpc::Status PutParentContexts(
      ::grpc::ServerContext* context, const PutParentContextsRequest* request,
      PutParentContextsResponse* response) override;

  ::grpc::Status GetContextsByArtifact(
      ::grpc::ServerContext* context,
      const GetContextsByArtifactRequest* request,
      GetContextsByArtifactResponse* response) override;

  ::grpc::Status GetContextsByExecution(
      ::grpc::ServerContext* context,
      const GetContextsByExecutionRequest* request,
      GetContextsByExecutionResponse* response) override;

  ::grpc::Status GetArtifactsByContext(
      ::grpc::ServerContext* context,
      const GetArtifactsByContextRequest* request,
      GetArtifactsByContextResponse* response) override;

  ::grpc::Status GetExecutionsByContext(
      ::grpc::ServerContext* context,
      const GetExecutionsByContextRequest* request,
      GetExecutionsByContextResponse* response) override;

  ::grpc::Status GetParentContextsByContext(
      ::grpc::ServerContext* context,
      const GetParentContextsByContextRequest* request,
      GetParentContextsByContextResponse* response) override;

  ::grpc::Status GetChildrenContextsByContext(
      ::grpc::ServerContext* context,
      const GetChildrenContextsByContextRequest* request,
      GetChildrenContextsByContextResponse* response) override;

  ::grpc::Status PutLineageSubgraph(
      ::grpc::ServerContext* context, const PutLineageSubgraphRequest* request,
      PutLineageSubgraphResponse* response) override;

  ::grpc::Status GetLineageSubgraph(
      ::grpc::ServerContext* context, const GetLineageSubgraphRequest* request,
      GetLineageSubgraphResponse* response) override;

 private:
  const ConnectionConfig connection_config_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_METADATA_STORE_SERVICE_IMPL_H_
