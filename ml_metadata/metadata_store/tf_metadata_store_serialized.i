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


%{
// Do not call these methods directly. Prefer metadata_store.py.
// For tests, see metadata_store_test.py
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "tensorflow/core/lib/core/errors.h"

#ifdef HAS_GLOBAL_STRING
  using ::string;
#else
  using std::string;
#endif

PyObject* ConvertToPythonString(const string& input_str) {
  return PyBytes_FromStringAndSize(input_str.data(), input_str.size());
}

template<typename ProtoType>
tensorflow::Status ParseProto(const string& input, ProtoType* proto) {
  if (proto->ParseFromString(input)) {
    return tensorflow::Status::OK();
  }
  return tensorflow::errors::InvalidArgument(
      "Could not parse proto");
}



%}

%{

ml_metadata::MetadataStore* CreateMetadataStore(const string& connection_config) {
  ml_metadata::ConnectionConfig proto_connection_config;
  const tensorflow::Status parse_proto_result =
      ParseProto(connection_config, &proto_connection_config);
  if (!parse_proto_result.ok()) {
    PyErr_SetString(PyExc_RuntimeError, parse_proto_result.error_message().c_str());
    return NULL;
  }


  std::unique_ptr<ml_metadata::MetadataStore> metadata_store;
  tensorflow::Status status = ml_metadata::CreateMetadataStore(
      proto_connection_config,
      &metadata_store);
  if (!status.ok()) {
    PyErr_SetString(PyExc_RuntimeError, status.error_message().c_str());
    return NULL;
  }
  return metadata_store.release();
}

void DestroyMetadataStore(ml_metadata::MetadataStore* metadata_store) {
  if (metadata_store != nullptr) {
    delete metadata_store;
  }
}

// Returns a native Python tuple:
// (serialized_proto_message, status message, status code),
// that can be deleted in Python safely.
PyObject* ConvertAccessMetadataStoreResultToPyTuple(
    const string& serialized_proto_message, const tensorflow::Status& status) {
  return PyTuple_Pack(3, ConvertToPythonString(serialized_proto_message),
  ConvertToPythonString(status.error_message()), PyInt_FromLong(status.code()));
}

// Given a method for MetadataStore of the form:
// tensorflow::Status my_method(const InputProto& input, OutputProto* output);
// this method will deserialize the request to an object of type InputProto,
// and serialize the result to a python string object. If there is an error,
// out_status will be set.
template<typename InputProto, typename OutputProto>
PyObject* AccessMetadataStore(
    ml_metadata::MetadataStore* metadata_store,
    const string& request,
    tensorflow::Status(ml_metadata::MetadataStore::*method)(const InputProto&,
        OutputProto*)) {
  InputProto proto_request;
  tensorflow::Status parse_result = ParseProto(request, &proto_request);
  if (!parse_result.ok()) {
    return ConvertAccessMetadataStoreResultToPyTuple(
        /* serialized_proto_message */ "",
        parse_result);
  }

  OutputProto proto_response;

  tensorflow::Status status = ((*metadata_store).*method)(proto_request,
                                                          &proto_response);
  string response;
  proto_response.SerializeToString(&response);
  return ConvertAccessMetadataStoreResultToPyTuple(response, status);
}

PyObject* GetArtifactType(ml_metadata::MetadataStore* metadata_store,
                          const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetArtifactType);
}

PyObject* PutArtifacts(ml_metadata::MetadataStore* metadata_store,
                       const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutArtifacts);
}

PyObject* GetArtifactsByID(ml_metadata::MetadataStore* metadata_store,
                           const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetArtifactsByID);
}

PyObject* GetArtifactsByType(ml_metadata::MetadataStore* metadata_store,
                             const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetArtifactsByType);
}

PyObject* GetArtifactsByURI(ml_metadata::MetadataStore* metadata_store,
                             const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetArtifactsByURI);
}

PyObject* PutArtifactType(ml_metadata::MetadataStore* metadata_store,
                          const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutArtifactType);
}

PyObject* GetExecutionType(ml_metadata::MetadataStore* metadata_store,
                           const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetExecutionType);
}

PyObject* GetExecutionTypes(ml_metadata::MetadataStore* metadata_store,
                            const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetExecutionTypes);
}

PyObject* GetArtifactTypes(ml_metadata::MetadataStore* metadata_store,
                           const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetArtifactTypes);
}

PyObject* PutExecutions(ml_metadata::MetadataStore* metadata_store,
                        const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutExecutions);
}

PyObject* GetExecutionsByID(ml_metadata::MetadataStore* metadata_store,
                            const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetExecutionsByID);
}

PyObject* GetExecutionsByType(ml_metadata::MetadataStore* metadata_store,
                              const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetExecutionsByType);
}

PyObject* PutExecutionType(ml_metadata::MetadataStore* metadata_store,
                           const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutExecutionType);
}

PyObject* PutEvents(ml_metadata::MetadataStore* metadata_store,
                    const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutEvents);
}

PyObject* PutExecution(ml_metadata::MetadataStore* metadata_store,
    const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutExecution);
}

PyObject* GetEventsByExecutionIDs(ml_metadata::MetadataStore* metadata_store,
                                  const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetEventsByExecutionIDs);
}

PyObject* GetArtifactTypesByID(ml_metadata::MetadataStore* metadata_store,
                               const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetArtifactTypesByID);
}

PyObject* GetExecutionTypesByID(ml_metadata::MetadataStore* metadata_store,
                                const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetExecutionTypesByID);
}

PyObject* GetEventsByArtifactIDs(ml_metadata::MetadataStore* metadata_store,
                                 const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetEventsByArtifactIDs);
}

PyObject* GetArtifacts(ml_metadata::MetadataStore* metadata_store,
                       const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetArtifacts);
}

PyObject* GetExecutions(ml_metadata::MetadataStore* metadata_store,
                        const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetExecutions);
}

PyObject* PutContextType(ml_metadata::MetadataStore* metadata_store,
                         const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutContextType);
}

PyObject* GetContextType(ml_metadata::MetadataStore* metadata_store,
                         const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetContextType);
}

PyObject* GetContextTypesByID(ml_metadata::MetadataStore* metadata_store,
                              const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetContextTypesByID);
}

PyObject* PutContexts(ml_metadata::MetadataStore* metadata_store,
                      const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutContexts);
}

PyObject* GetContextsByID(ml_metadata::MetadataStore* metadata_store,
                          const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetContextsByID);
}

PyObject* GetContexts(ml_metadata::MetadataStore* metadata_store,
                      const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetContexts);
}

PyObject* GetContextsByType(ml_metadata::MetadataStore* metadata_store,
                            const string& request) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetContextsByType);
}

%}


// Typemap to convert an input argument from Python object to C++ string.
%typemap(in) const string& (string temp) {
  char *buf;
  Py_ssize_t len;
  if (PyBytes_AsStringAndSize($input, &buf, &len) == -1) SWIG_fail;
  temp.assign(buf, len);
  $1 = &temp;
}



%newobject CreateMetadataStore;
%delobject DestroyMetadataStore;

ml_metadata::MetadataStore* CreateMetadataStore(const string& connection_config);

void DestroyMetadataStore(ml_metadata::MetadataStore* metadata_store);


PyObject* PutArtifactType(ml_metadata::MetadataStore* metadata_store,
                          const string& request);

PyObject* PutArtifacts(ml_metadata::MetadataStore* metadata_store,
                       const string& request);

PyObject* GetArtifactType(ml_metadata::MetadataStore* metadata_store,
                          const string& request);
PyObject* GetArtifactTypes(ml_metadata::MetadataStore* metadata_store,
                           const string& request);
PyObject* GetExecutionTypes(ml_metadata::MetadataStore* metadata_store,
                           const string& request);

PyObject* GetArtifactsByID(ml_metadata::MetadataStore* metadata_store,
                           const string& request);

PyObject* GetArtifactsByType(ml_metadata::MetadataStore* metadata_store,
                             const string& request);

PyObject* GetArtifactsByURI(ml_metadata::MetadataStore* metadata_store,
                            const string& request);

PyObject* PutExecutionType(ml_metadata::MetadataStore* metadata_store,
                           const string& request);

PyObject* PutExecutions(ml_metadata::MetadataStore* metadata_store,
                        const string& request);

PyObject* GetExecutionType(ml_metadata::MetadataStore* metadata_store,
                           const string& request);

PyObject* GetExecutionsByID(ml_metadata::MetadataStore* metadata_store,
                            const string& request);

PyObject* GetExecutionsByType(ml_metadata::MetadataStore* metadata_store,
                              const string& request);

PyObject* PutContextType(ml_metadata::MetadataStore* metadata_store,
                         const string& request);

PyObject* GetContextType(ml_metadata::MetadataStore* metadata_store,
                         const string& request);

PyObject* GetArtifactTypesByID(ml_metadata::MetadataStore* metadata_store,
    const string& request);

PyObject* GetExecutionTypesByID(ml_metadata::MetadataStore* metadata_store,
    const string& request);

PyObject* GetContextTypesByID(ml_metadata::MetadataStore* metadata_store,
    const string& request);

PyObject* PutEvents(ml_metadata::MetadataStore* metadata_store,
                    const string& request);

PyObject* PutExecution(ml_metadata::MetadataStore* metadata_store,
                       const string& request);

PyObject* GetEventsByExecutionIDs(ml_metadata::MetadataStore* metadata_store,
                                  const string& request);

PyObject* GetEventsByArtifactIDs(ml_metadata::MetadataStore* metadata_store,
    const string& request);

PyObject* GetArtifacts(ml_metadata::MetadataStore* metadata_store,
    const string& request);

PyObject* GetExecutions(ml_metadata::MetadataStore* metadata_store,
    const string& request);

PyObject* PutContexts(ml_metadata::MetadataStore* metadata_store,
                      const string& request);

PyObject* GetContextsByID(ml_metadata::MetadataStore* metadata_store,
                          const string& request);

PyObject* GetContexts(ml_metadata::MetadataStore* metadata_store,
                      const string& request);

PyObject* GetContextsByType(ml_metadata::MetadataStore* metadata_store,
                            const string& request);
