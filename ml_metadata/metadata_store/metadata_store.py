# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A python API to the metadata store.

Provides access to a SQLite3 or a MySQL backend. Artifact types and execution
types can be created on the fly.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function


from ml_metadata.metadata_store import pywrap_metadata_store_serialized as metadata_store_serialized
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from tensorflow.python.framework import errors


class MetadataStore(object):
  """A store for the artifact metadata."""

  def __init__(self, config):
    with errors.raise_exception_on_not_ok_status() as status:
      # MetadataStore will contain a MetadataStorePtr, which may be nullptr
      # if the store was not correctly initialized.
      self._metadata_store = metadata_store_serialized.CreateMetadataStore(
          config.SerializeToString(), status)

  def __del__(self):
    metadata_store_serialized.DestroyMetadataStore(self._metadata_store)

  def _swig_call(self, method, request, response):
    """Calls method, serializing and deserializing inputs and outputs.

    Note that this does not check the types of request and response.

    This can throw a variety of Python errors, based upon the underlying
    tensorflow error returned in MetadataStore.
    See _CODE_TO_EXCEPTION_CLASS in tensorflow/python/framework/errors_impl.py
    for the mapping.

    Args:
      method: the method to call in SWIG.
      request: a protobuf message, serialized and sent to the method.
      response: a protobuf message, filled from the return value of the method.

    Raises:
      Error: whatever tensorflow error is returned by the method.
    """
    with errors.raise_exception_on_not_ok_status() as status:
      response_str = method(self._metadata_store, request.SerializeToString(),
                            status)
    response.ParseFromString(response_str)

  def put_artifacts(
      self, artifacts):
    """Inserts or updates artifacts in the database.

    If an artifact_id is specified for an artifact, it is an update.
    If an artifact_id is unspecified, it will insert a new artifact.
    For new artifacts, type must be specified.
    For old artifacts, type must be unchanged or unspecified.

    Args:
      artifacts: A list of artifacts to insert or update.

    Returns:
      A list of artifact ids index-aligned with the input.
    """
    request = metadata_store_service_pb2.PutArtifactsRequest()
    for x in artifacts:
      request.artifacts.add().CopyFrom(x)
    response = metadata_store_service_pb2.PutArtifactsResponse()
    self._swig_call(metadata_store_serialized.PutArtifacts, request, response)
    result = []
    for x in response.artifact_ids:
      result.append(x)
    return result

  def put_artifact_type(self,
                        artifact_type,
                        can_add_fields = False,
                        can_delete_fields = False,
                        all_fields_match = True):
    """Inserts or updates an artifact type.

    If no artifact type exists in the database with the given name, it creates
    a new artifact type (and a database).

    If an artifact type with the same name already exists (let's call it
    old_artifact_type), then the impact depends upon the other options.

    If artifact_type == old_artifact_type, then nothing happens.

    Otherwise, if there is a field where artifact_type and old_artifact_type
    have different types, then it fails.

    Otherwise, if can_add_fields is False and artifact_type has a field
    old_artifact_type is missing, then it fails.

    Otherwise, if all_fields_match is True and old_artifact_type has a field
    artifact_type is missing, then it fails.

    Otherwise, if can_delete_fields is True and old_artifact_type has a field
    artifact_type is missing, then it deletes that field.

    Otherwise, it does nothing.

    Args:
      artifact_type: the type to add or update.
      can_add_fields: if true, you can add fields with this operation. If false,
        then if there are more fields in artifact_type than in the database, the
        call fails.
      can_delete_fields: if true, you can remove fields with this operation. If
        false, then if there are more fields in the current type, they are not
        removed.
      all_fields_match: if true, all fields must match, and the method fails if
        they are not the same.

    Returns:
      the type_id of the response.

    Raises:
      InvalidArgumentError: If a constraint is violated.
    """
    request = metadata_store_service_pb2.PutArtifactTypeRequest()
    request.can_add_fields = can_add_fields
    request.can_delete_fields = can_delete_fields
    request.all_fields_match = all_fields_match
    request.artifact_type.CopyFrom(artifact_type)
    response = metadata_store_service_pb2.PutArtifactTypeResponse()
    self._swig_call(metadata_store_serialized.PutArtifactType, request,
                    response)
    return response.type_id

  def create_artifact_with_type(
      self, artifact,
      artifact_type):
    """Creates an artifact with a type.

    This first gets the type (or creates it if it does not exist), and then
    puts the artifact into the database with that type.

    The type_id should not be specified in the artifact (it is ignored).

    Note that this is not a transaction!
    1. First, the type is created as a transaction.
    2. Then the artifact is created as a transaction.

    Args:
      artifact: the artifact to create (no id or type_id)
      artifact_type: the type of the new artifact (no id)

    Returns:
      the artifact ID of the resulting type.

    Raises:
      InvalidArgument: if the type is not the same as one with the same name
        already in the database.
    """
    type_id = self.put_artifact_type(artifact_type)
    artifact_copy = metadata_store_pb2.Artifact()
    artifact_copy.CopyFrom(artifact)
    artifact_copy.type_id = type_id
    [artifact_id] = self.put_artifacts([artifact_copy])
    return artifact_id

  def put_executions(
      self, executions):
    """Inserts or updates executions in the database.

    If an execution_id is specified for an execution, it is an update.
    If an execution_id is unspecified, it will insert a new execution.
    For new executions, type must be specified.
    For old executions, type must be unchanged or unspecified.

    Args:
      executions: A list of executions to insert or update.

    Returns:
      A list of execution ids index-aligned with the input.
    """
    request = metadata_store_service_pb2.PutExecutionsRequest()
    for x in executions:
      request.executions.add().CopyFrom(x)
    response = metadata_store_service_pb2.PutExecutionsResponse()
    self._swig_call(metadata_store_serialized.PutExecutions, request, response)
    result = []
    for x in response.execution_ids:
      result.append(x)
    return result

  def put_execution_type(self,
                         execution_type,
                         can_add_fields = False,
                         can_delete_fields = False,
                         all_fields_match = True):
    """Inserts or updates an execution type.

    If no execution type exists in the database with the given name, it creates
    a new execution type (and a database).

    If an execution type with the same name already exists (let's call it
    old_execution_type), then the impact depends upon the other options.

    If execution_type == old_execution_type, then nothing happens.

    Otherwise, if there is a field where execution_type and old_execution_type
    have different types, then it fails.

    Otherwise, if can_add_fields is False and execution_type has a field
    old_execution_type is missing, then it fails.

    Otherwise, if all_fields_match is True and old_execution_type has a field
    execution_type is missing, then it fails.

    Otherwise, if can_delete_fields is True and old_execution_type has a field
    execution_type is missing, then it deletes that field.

    Otherwise, it does nothing.
    Args:
      execution_type: the type to add or update.
      can_add_fields: if true, you can add fields with this operation. If false,
        then if there are more fields in execution_type than in the database,
        the call fails.
      can_delete_fields: if true, you can remove fields with this operation. If
        false, then if there are more fields.
      all_fields_match: if true, all fields must match, and the method fails if
        they are not the same.

    Returns:
      the type id of the type.
    Raises:
      ValueError: If a constraint is violated.
    """
    request = metadata_store_service_pb2.PutExecutionTypeRequest()
    request.can_add_fields = can_add_fields
    request.can_delete_fields = can_delete_fields
    request.all_fields_match = all_fields_match
    request.execution_type.CopyFrom(execution_type)
    response = metadata_store_service_pb2.PutExecutionTypeResponse()
    self._swig_call(metadata_store_serialized.PutExecutionType, request,
                    response)
    return response.type_id

  def put_events(self, events):
    """Inserts events in the database.

    The execution_id and artifact_id must already exist.
    Once created, events cannot be modified.


    Args:
      events: A list of events to insert or update.
    """
    request = metadata_store_service_pb2.PutEventsRequest()
    for x in events:
      request.events.add().CopyFrom(x)
    response = metadata_store_service_pb2.PutEventsResponse()

    self._swig_call(metadata_store_serialized.PutEvents, request, response)

  def get_artifacts_by_type(
      self, type_name):
    """Gets all the artifacts of a given type."""
    raise NotImplementedError()

  def get_artifacts_by_id(
      self, artifact_ids):
    """Gets all artifacts with matching ids.

    The result is not index-aligned: if an id is not found, it is not returned.

    Args:
      artifact_ids: A list of artifact ids to retrieve.

    Returns:
      Artifacts with matching ids.
    """
    request = metadata_store_service_pb2.GetArtifactsByIDRequest()
    for x in artifact_ids:
      request.artifact_ids.append(x)
    response = metadata_store_service_pb2.GetArtifactsByIDResponse()
    self._swig_call(metadata_store_serialized.GetArtifactsByID, request,
                    response)
    result = []
    for x in response.artifacts:
      result.append(x)
    return result

  def get_artifact_type(
      self, type_name):
    """Gets an artifact type by name.

    Args:
     type_name: the type with that name.

    Returns:
     The type with name type_name.

    Raises:
    tensorflow.errors.NotFoundError: if no type exists
    tensorflow.errors.InternalError: if query execution fails
    """
    request = metadata_store_service_pb2.GetArtifactTypeRequest()
    request.type_name = type_name
    response = metadata_store_service_pb2.GetArtifactTypeResponse()
    self._swig_call(metadata_store_serialized.GetArtifactType, request,
                    response)
    return response.artifact_type

  def get_execution_type(
      self, type_name):
    """Gets an execution type, or None if it does not exist."""
    request = metadata_store_service_pb2.GetExecutionTypeRequest()
    request.type_name = type_name
    response = metadata_store_service_pb2.GetExecutionTypeResponse()
    self._swig_call(metadata_store_serialized.GetExecutionType, request,
                    response)
    return response.execution_type

  def get_executions_by_type(
      self, type_name):
    """Gets all the executions of a given type."""
    raise NotImplementedError()

  def get_executions_by_id(
      self, execution_ids):
    """Gets all executions with matching ids.

    The result is not index-aligned: if an id is not found, it is not returned.

    Args:
      execution_ids: A list of execution ids to retrieve.

    Returns:
      Executions with matching ids.
    """
    request = metadata_store_service_pb2.GetExecutionsByIDRequest()
    for x in execution_ids:
      request.execution_ids.append(x)
    response = metadata_store_service_pb2.GetExecutionsByIDResponse()
    self._swig_call(metadata_store_serialized.GetExecutionsByID, request,
                    response)
    result = []
    for x in response.executions:
      result.append(x)
    return result

  def get_executions(self):
    """Gets all executions.

    Returns:
      A list of all executions.

    Raises:
      InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetExecutionsRequest()
    response = metadata_store_service_pb2.GetExecutionsResponse()
    self._swig_call(metadata_store_serialized.GetExecutions, request, response)
    result = []
    for x in response.executions:
      result.append(x)
    return result

  def get_artifacts(self):
    """Gets all artifacts.

    Returns:
      A list of all artifacts.

    Raises:
      InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetArtifactsRequest()
    response = metadata_store_service_pb2.GetArtifactsResponse()
    self._swig_call(metadata_store_serialized.GetArtifacts, request, response)
    result = []
    for x in response.artifacts:
      result.append(x)
    return result

  def get_artifact_types_by_id(
      self, type_ids):
    """Gets types by ID.

    Args:
      type_ids: a sequence of artifact type IDs.

    Returns:
      A list of artifact types.

    Raises:
      InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetArtifactTypesByIDRequest()
    response = metadata_store_service_pb2.GetArtifactTypesByIDResponse()
    for x in type_ids:
      request.type_ids.append(x)
    self._swig_call(metadata_store_serialized.GetArtifactTypesByID, request,
                    response)
    result = []
    for x in response.artifact_types:
      result.append(x)
    return result

  def get_execution_types_by_id(
      self, type_ids):
    """Gets types by ID.

    Args:
      type_ids: a sequence of artifact type IDs.

    Returns:
      A list of execution types.

    Args:
      type_ids: ids to look for.

    Raises:
      InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetExecutionTypesByIDRequest()
    response = metadata_store_service_pb2.GetExecutionTypesByIDResponse()
    for x in type_ids:
      request.type_ids.append(x)
    self._swig_call(metadata_store_serialized.GetExecutionTypesByID, request,
                    response)
    result = []
    for x in response.execution_types:
      result.append(x)
    return result

  def get_events_by_execution_ids(
      self, execution_ids):
    """Gets all events with matching execution ids.

    Args:
      execution_ids: a list of execution ids.

    Returns:
      Events with the execution IDs given.

    Raises:
      InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetEventsByExecutionIDsRequest()
    for x in execution_ids:
      request.execution_ids.append(x)
    response = metadata_store_service_pb2.GetEventsByExecutionIDsResponse()
    self._swig_call(metadata_store_serialized.GetEventsByExecutionIDs, request,
                    response)
    result = []
    for x in response.events:
      result.append(x)
    return result

  def get_events_by_artifact_ids(
      self, artifact_ids):
    """Gets all events with matching artifact ids.

    Args:
      artifact_ids: a list of artifact ids.

    Returns:
      Events with the execution IDs given.

    Raises:
      InternalError: if query execution fails.
    """

    request = metadata_store_service_pb2.GetEventsByArtifactIDsRequest()
    for x in artifact_ids:
      request.artifact_ids.append(x)
    response = metadata_store_service_pb2.GetEventsByArtifactIDsResponse()
    self._swig_call(metadata_store_serialized.GetEventsByArtifactIDs, request,
                    response)
    result = []
    for x in response.events:
      result.append(x)
    return result

  def make_artifact_live(self, artifact_id):
    """Changes the state of each artifact to LIVE.

    The artifact state must be NEW or CREATABLE.

    Args:
      artifact_id: the ID of the artifact.
    """
    raise NotImplementedError()

  def complete_execution(self, execution_id,
                         artifact_ids):
    """Changes the state of an execution to COMPLETE and the artifacts to LIVE.

    The execution state must be NEW or RUNNING.
    The artifacts must be NEW or CREATABLE.

    Args:
      execution_id: the execution to change to COMPLETE.
      artifact_ids: the artifacts to change to LIVE.
    """
    raise NotImplementedError()
