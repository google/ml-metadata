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
#include "tensorflow/c/tf_status_helper.h"
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
bool ParseProto(const string& input, ProtoType* proto, TF_Status* out_status) {
  if (proto->ParseFromString(input)) {
    return true;
  }
  const tensorflow::Status status = tensorflow::errors::InvalidArgument(
      "Could not parse proto");
  Set_TF_Status_from_Status(out_status, status);
  return false;
}

%}

%{

ml_metadata::MetadataStore* CreateMetadataStore(const string& connection_config,
                                    TF_Status* out_status) {
  ml_metadata::ConnectionConfig proto_connection_config;
  if (!ParseProto(connection_config, &proto_connection_config, out_status)) {
    return nullptr;
  }

  std::unique_ptr<ml_metadata::MetadataStore> metadata_store;
  tensorflow::Status status = ml_metadata::CreateMetadataStore(
      proto_connection_config,
      &metadata_store);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
  // If the status fails, this will be a nullptr.
  return metadata_store.release();
}

void DestroyMetadataStore(ml_metadata::MetadataStore* metadata_store) {
  if (metadata_store != nullptr) {
    delete metadata_store;
  }
}

// Given a method for MetadataStore of the form:
// tensorflow::Status my_method(const InputProto& input, OutputProto* output);
// this method will deserialize the request to an object of type InputProto,
// and serialize the result to a python string object. If there is an error,
// out_status will be set.
template<typename InputProto, typename OutputProto>
PyObject* AccessMetadataStore(ml_metadata::MetadataStore* metadata_store,
    const string& request, tensorflow::Status(ml_metadata::MetadataStore::*method)(const InputProto&, OutputProto*), 
    TF_Status *out_status) {
  InputProto proto_request;
  if (!ParseProto(request, &proto_request, out_status)) {
    string response;
    return ConvertToPythonString(response);
  }

  OutputProto proto_response;

  tensorflow::Status status = ((*metadata_store).*method)(proto_request,
                                                          &proto_response);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }
  string response;
  proto_response.SerializeToString(&response);
  return ConvertToPythonString(response);
}

PyObject* GetArtifactType(ml_metadata::MetadataStore* metadata_store,
                          const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetArtifactType, out_status);
}

PyObject* PutArtifacts(ml_metadata::MetadataStore* metadata_store,
                       const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutArtifacts, out_status);
}

PyObject* GetArtifactsByID(ml_metadata::MetadataStore* metadata_store,
                           const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetArtifactsByID, out_status);
}

PyObject* PutArtifactType(ml_metadata::MetadataStore* metadata_store,
                          const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutArtifactType, out_status);
}

PyObject* GetExecutionType(ml_metadata::MetadataStore* metadata_store,
                           const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetExecutionType, out_status);
}

PyObject* PutExecutions(ml_metadata::MetadataStore* metadata_store,
                        const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutExecutions, out_status);
}

PyObject* GetExecutionsByID(ml_metadata::MetadataStore* metadata_store,
                            const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetExecutionsByID, out_status);
}

PyObject* PutExecutionType(ml_metadata::MetadataStore* metadata_store,
                           const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutExecutionType, out_status);
}

PyObject* PutEvents(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::PutEvents, out_status);
}

PyObject* GetEventsByExecutionIDs(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetEventsByExecutionIDs, out_status);
}

PyObject* GetArtifactTypesByID(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetArtifactTypesByID, out_status);
}

PyObject* GetExecutionTypesByID(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetExecutionTypesByID, out_status);
}

PyObject* GetEventsByArtifactIDs(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetEventsByArtifactIDs, out_status);
}

PyObject* GetArtifacts(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetArtifacts, out_status);
}

PyObject* GetExecutions(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status) {
  return AccessMetadataStore(metadata_store, request,
      &ml_metadata::MetadataStore::GetExecutions, out_status);
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


%delobject DestroyMetadataStore;

// Indicates that the CreateMetadataStore2 creates a new object, and that
// python is responsible for destroying it.
%newobject CreateMetadataStore;

ml_metadata::MetadataStore* CreateMetadataStore(const string& connection_config,
                                                TF_Status* out_status);

void DestroyMetadataStore(ml_metadata::MetadataStore* metadata_store);


PyObject* PutArtifactType(ml_metadata::MetadataStore* metadata_store,
                          const string& request, TF_Status* out_status);

PyObject* PutArtifacts(ml_metadata::MetadataStore* metadata_store,
                       const string& request, TF_Status* out_status);

PyObject* GetArtifactType(ml_metadata::MetadataStore* metadata_store,
                          const string& request, TF_Status* out_status);

PyObject* GetArtifactsByID(ml_metadata::MetadataStore* metadata_store,
                           const string& request, TF_Status* out_status);

PyObject* PutExecutionType(ml_metadata::MetadataStore* metadata_store,
                           const string& request, TF_Status* out_status);

PyObject* PutExecutions(ml_metadata::MetadataStore* metadata_store,
                        const string& request, TF_Status* out_status);

PyObject* GetExecutionType(ml_metadata::MetadataStore* metadata_store,
                           const string& request, TF_Status* out_status);

PyObject* GetExecutionsByID(ml_metadata::MetadataStore* metadata_store,
                            const string& request, TF_Status* out_status);


PyObject* GetArtifactTypesByID(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status);

PyObject* GetExecutionTypesByID(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status);

PyObject* PutEvents(ml_metadata::MetadataStore* metadata_store,
                    const string& request, TF_Status* out_status);

PyObject* GetEventsByExecutionIDs(ml_metadata::MetadataStore* metadata_store,
                                  const string& request, TF_Status* out_status);

PyObject* GetEventsByArtifactIDs(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status);

PyObject* GetArtifacts(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status);

PyObject* GetExecutions(ml_metadata::MetadataStore* metadata_store,
    const string& request, TF_Status* out_status);


