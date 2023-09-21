/* Copyright 2020 Google LLC

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
#ifndef ML_METADATA_METADATA_STORE_METADATA_STORE_SERVICE_INTERFACE_H_
#define ML_METADATA_METADATA_STORE_METADATA_STORE_SERVICE_INTERFACE_H_

#include "absl/status/status.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"

namespace ml_metadata {

// An interface for calling MetadataStoreService methods. This interface hides
// the details of the backend (in-process MetadataStore, gRPC client etc).
// Implementations must ensure that each method is an atomic operation.
class MetadataStoreServiceInterface {
 public:
  virtual ~MetadataStoreServiceInterface() {}

#define METADATA_STORE_SERVICE_INTERFACE_DECLARE(method)      \
  virtual absl::Status method(const method##Request& request, \
                              method##Response* response) {   \
    return absl::UnimplementedError(#method);                 \
  }

  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutArtifacts)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutArtifactType)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutExecutions)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutExecutionType)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutEvents)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutExecution)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutTypes)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutContextType)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutContexts)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutAttributionsAndAssociations)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutParentContexts)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(PutLineageSubgraph)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetArtifactType)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetArtifactTypesByID)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetArtifactTypes)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetArtifactTypesByExternalIds)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetExecutionType)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetExecutionTypesByID)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetExecutionTypes)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetExecutionTypesByExternalIds)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetContextType)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetContextTypesByID)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetContextTypes)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetContextTypesByExternalIds)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetArtifacts)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetExecutions)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetContexts)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetArtifactsByID)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetExecutionsByID)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetContextsByID)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetArtifactsByType)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetArtifactByTypeAndName)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetArtifactsByExternalIds)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetExecutionsByType)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetExecutionByTypeAndName)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetExecutionsByExternalIds)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetContextsByType)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetContextByTypeAndName)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetContextsByExternalIds)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetArtifactsByURI)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetEventsByExecutionIDs)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetEventsByArtifactIDs)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetContextsByArtifact)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetContextsByExecution)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetParentContextsByContext)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetChildrenContextsByContext)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetParentContextsByContexts)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetChildrenContextsByContexts)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetArtifactsByContext)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetExecutionsByContext)
  // TODO(b/283852485): delete interface later on once ensure no incoming
  // traffic.
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetLineageGraph)
  METADATA_STORE_SERVICE_INTERFACE_DECLARE(GetLineageSubgraph)

#undef METADATA_STORE_SERVICE_INTERFACE_DECLARE
};

}  // namespace ml_metadata

#endif  // ML_METADATA_METADATA_STORE_METADATA_STORE_SERVICE_INTERFACE_H_
