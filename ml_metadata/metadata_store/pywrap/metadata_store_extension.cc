/* Copyright 2021 Google LLC

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

#include <memory>

#include "absl/status/status.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/simple_types_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/simple_types/proto/simple_types.pb.h"
#include "include/pybind11/pybind11.h"

namespace {
namespace py = pybind11;

// Creates a MetadataStore object and returns as a unique pointer.
// Returns python RuntimeError if any error occur during creation.
std::unique_ptr<ml_metadata::MetadataStore> CreateMetadataStore(
    const std::string& connection_config,
    const std::string& migration_options) {
  ml_metadata::ConnectionConfig proto_connection_config;
  if (!proto_connection_config.ParseFromString(connection_config)) {
    throw std::runtime_error("Could not parse proto.");
  }
  ml_metadata::MigrationOptions proto_migration_options;
  if (!proto_migration_options.ParseFromString(migration_options)) {
    throw std::runtime_error("Could not parse proto.");
  }
  std::unique_ptr<ml_metadata::MetadataStore> metadata_store;
  absl::Status creation_status = ml_metadata::CreateMetadataStore(
      proto_connection_config, proto_migration_options, &metadata_store);
  if (!creation_status.ok()) {
    throw std::runtime_error(std::string(creation_status.message()));
  }
  return metadata_store;
}

// Loads simple types from
// third_party/ml_metadata/simple_types/simple_types_constants.cc into a
// serialized SimpleTypes proto.
//
// The returned tupe consists of the serialized proto, a strong-typed error with
// error message and the canonical error code.
py::tuple LoadSimpleTypes() {
  ml_metadata::SimpleTypes simple_types;
  absl::Status status = ml_metadata::LoadSimpleTypes(simple_types);
  std::string simple_types_serialized;
  simple_types.SerializeToString(&simple_types_serialized);
  return py::make_tuple(py::bytes(simple_types_serialized),
                        py::bytes(std::string(status.message())),
                        py::int_((int)status.code()));
}

// A MetadataStore method returns a tuple to the python metadata_store.py.
// The tuple is consist of serialized method response, and a strong typed
// error with error_message and canonical error code.
py::tuple ConvertAccessMetadataStoreResultToPyTuple(
    const std::string& serialized_proto_message, const absl::Status& status) {
  return py::make_tuple(py::bytes(serialized_proto_message),
                        py::bytes(std::string(status.message())),
                        py::int_((int)status.code()));
}

// Utility method to dispatch python method calls. The `request` is parsed and
// passed to the `method` of MetadataStore. It returns the `response` and
// strong typed errors if any.
template <typename InputProto, typename OutputProto>
py::tuple AccessMetadataStore(
    ml_metadata::MetadataStore* metadata_store, const std::string& request,
    absl::Status (ml_metadata::MetadataStore::*method)(const InputProto&,
                                                       OutputProto*)) {
  InputProto proto_request;
  if (!proto_request.ParseFromString(request)) {
    return ConvertAccessMetadataStoreResultToPyTuple(
        /*serialized_proto_message=*/"",
        absl::InvalidArgumentError("Could not parse proto"));
  }
  OutputProto proto_response;
  absl::Status call_status =
      ((*metadata_store).*method)(proto_request, &proto_response);
  std::string response;
  proto_response.SerializeToString(&response);
  return ConvertAccessMetadataStoreResultToPyTuple(response, call_status);
}

// A macro to define pybind module methods.
#define METADATA_STORE_METHOD_PYBIND11_DECLARE(method)            \
  m.def(#method,                                                  \
      [](ml_metadata::MetadataStore& metadata_store,              \
         const std::string& request) -> py::tuple {               \
        return AccessMetadataStore(                               \
            &metadata_store, request,                             \
            &ml_metadata::MetadataStore::method);                 \
      });

PYBIND11_MODULE(metadata_store_extension, main_module) {
  auto m = main_module.def_submodule("metadata_store");
  m.doc() = "MLMD MetadataStore API pybind11 extension module.";
  py::class_<ml_metadata::MetadataStore>(m, "MetadataStore");
  m.def("CreateMetadataStore", &CreateMetadataStore, "Create MetadataStore.");
  m.def("LoadSimpleTypes", &LoadSimpleTypes, "Load MLMD Simple Types.");
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutArtifactType)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutArtifacts)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutExecutions)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutExecutionType)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutEvents)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutExecution)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutTypes)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutContextType)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutContexts)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutAttributionsAndAssociations)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutParentContexts)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(PutLineageSubgraph)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetArtifactType)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetArtifactTypesByID)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetArtifactTypes)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetArtifactTypesByExternalIds)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetExecutionType)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetExecutionTypesByID)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetExecutionTypes)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetExecutionTypesByExternalIds)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetContextType)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetContextTypesByID)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetContextTypes)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetContextTypesByExternalIds)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetArtifacts)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetExecutions)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetContexts)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetArtifactsByID)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetExecutionsByID)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetContextsByID)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetArtifactsByType)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetArtifactByTypeAndName)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetExecutionsByType)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetExecutionByTypeAndName)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetContextsByType)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetContextByTypeAndName)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetArtifactsByURI)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetArtifactsByExternalIds)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetExecutionsByExternalIds)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetContextsByExternalIds)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetEventsByExecutionIDs)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetEventsByArtifactIDs)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetContextsByArtifact)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetContextsByExecution)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetArtifactsByContext)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetExecutionsByContext)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetParentContextsByContext)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetChildrenContextsByContext)
  METADATA_STORE_METHOD_PYBIND11_DECLARE(GetLineageSubgraph)
}

}  // namespace
