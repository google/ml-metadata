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

from absl import logging
import grpc
from typing import List, Optional, Sequence, Text, Tuple, Union

from ml_metadata.metadata_store import pywrap_tf_metadata_store_serialized as metadata_store_serialized
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from ml_metadata.proto import metadata_store_service_pb2_grpc
from tensorflow.python.framework import errors


class MetadataStore(object):
  """A store for the artifact metadata."""

  def __init__(self,
               config: Union[metadata_store_pb2.ConnectionConfig,
                             metadata_store_pb2.MetadataStoreClientConfig],
               enable_upgrade_migration: bool = False):
    """Initialize the MetadataStore.

    MetadataStore can directly connect to either the metadata database or
    the metadata store server.

    Args:
      config: metadata_store_pb2.ConnectionConfig or
        metadata_store_pb2.MetadataStoreClientConfig. Configuration to
        connect to the database or the metadata store server.
      enable_upgrade_migration: if set to True, the library upgrades the db
        schema and migrates all data if it connects to an old version backend.
        It is ignored when using GRPC client connection config.
    """
    if isinstance(config, metadata_store_pb2.ConnectionConfig):
      self._using_db_connection = True
      migration_options = metadata_store_pb2.MigrationOptions()
      migration_options.enable_upgrade_migration = enable_upgrade_migration
      self._metadata_store = metadata_store_serialized.CreateMetadataStore(
          config.SerializeToString(), migration_options.SerializeToString())
      logging.log(logging.INFO, 'MetadataStore with DB connection initialized')
      return
    if not isinstance(config, metadata_store_pb2.MetadataStoreClientConfig):
      raise ValueError('MetadataStore is expecting either'
                       'metadata_store_pb2.ConnectionConfig or'
                       'metadata_store_pb2.MetadataStoreClientConfig')
    self._using_db_connection = False
    if enable_upgrade_migration:
      raise ValueError('Upgrade migration is not allowed when using gRPC '
                       'connection client. Upgrade needs to be performed on '
                       'the server side.')
    target = ':'.join([config.host, str(config.port)])
    channel = self._get_channel(config, target)
    self._metadata_store_stub = (metadata_store_service_pb2_grpc.
                                 MetadataStoreServiceStub(channel))
    logging.log(logging.INFO, 'MetadataStore with gRPC connection initialized')

  def _get_channel(self, config: metadata_store_pb2.MetadataStoreClientConfig,
                   target: Text):
    """Configures the channel, which could be secure or insecure.

    It returns a channel that can be specified to be secure or insecure,
    depending on whether ssl_config is specified in the config.

    Args:
      config: metadata_store_pb2.MetadataStoreClientConfig.
      target: target host with port.

    Returns:
      an initialized gRPC channel.
    """
    if not config.HasField('ssl_config'):
      return grpc.insecure_channel(target)

    root_certificates = None
    private_key = None
    certificate_chain = None
    if config.ssl_config.HasField('custom_ca'):
      root_certificates = bytes(
          str(config.ssl_config.custom_ca).encode('ascii'))
    if config.ssl_config.HasField('client_key'):
      private_key = bytes(str(config.ssl_config.client_key).encode('ascii'))
    if config.ssl_config.HasField('server_cert'):
      certificate_chain = bytes(
          str(config.ssl_config.server_cert).encode('ascii'))
    credentials = grpc.ssl_channel_credentials(root_certificates, private_key,
                                               certificate_chain)
    return grpc.secure_channel(target, credentials)

  def __del__(self):
    if self._using_db_connection:
      metadata_store_serialized.DestroyMetadataStore(self._metadata_store)

  def _call(self, method_name, request, response) -> None:
    """Calls method using SWIG or gRPC.

    Args:
      method_name: the method to call in SWIG or gRPC.
      request: a protobuf message, serialized and sent to the method.
      response: a protobuf message, filled from the return value of the method.
    """
    if self._using_db_connection:
      swig_method = getattr(metadata_store_serialized, method_name)
      self._swig_call(swig_method, request, response)
    else:
      # TODO(b/144590158): Add E2E tests to verify grpc support.
      grpc_method = getattr(self._metadata_store_stub, method_name)
      try:
        response.CopyFrom(grpc_method(request))
      except grpc.RpcError as e:
        # RpcError code uses a tuple to specify error code and short
        # description.
        # https://grpc.github.io/grpc/python/_modules/grpc.html#StatusCode
        raise _make_exception(e.details(), e.code().value[0])

  def _swig_call(self, method, request, response) -> None:
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
    [response_str, error_message, status_code] = method(
        self._metadata_store, request.SerializeToString())
    if status_code != 0:
      raise _make_exception(error_message, status_code)
    response.ParseFromString(response_str)

  def put_artifacts(
      self, artifacts: Sequence[metadata_store_pb2.Artifact]) -> List[int]:
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

    self._call('PutArtifacts', request, response)
    result = []
    for x in response.artifact_ids:
      result.append(x)
    return result

  def put_artifact_type(self,
                        artifact_type: metadata_store_pb2.ArtifactType,
                        can_add_fields: bool = False,
                        can_delete_fields: bool = False,
                        all_fields_match: bool = True) -> int:
    """Inserts or updates an artifact type.

    Similar to put execution/context type, if no artifact type exists in the
    database with the given name, it creates a new artifact type (and a
    database).

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

    self._call('PutArtifactType', request, response)
    return response.type_id

  def create_artifact_with_type(
      self, artifact: metadata_store_pb2.Artifact,
      artifact_type: metadata_store_pb2.ArtifactType) -> int:
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
    return self.put_artifacts([artifact_copy])[0]

  def put_executions(
      self, executions: Sequence[metadata_store_pb2.Execution]) -> List[int]:
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

    self._call('PutExecutions', request, response)
    result = []
    for x in response.execution_ids:
      result.append(x)
    return result

  def put_execution_type(self,
                         execution_type: metadata_store_pb2.ExecutionType,
                         can_add_fields: bool = False,
                         can_delete_fields: bool = False,
                         all_fields_match: bool = True) -> int:
    """Inserts or updates an execution type.

    Similar to put artifact/context type, if no execution type exists in the
    database with the given name, it creates a new execution type (and a
    database).

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

    self._call('PutExecutionType', request, response)
    return response.type_id

  def put_contexts(self,
                   contexts: Sequence[metadata_store_pb2.Context]) -> List[int]:
    """Inserts or updates contexts in the database.

    If an context_id is specified for an context, it is an update.
    If an context_id is unspecified, it will insert a new context.
    For new contexts, type must be specified.
    For old contexts, type must be unchanged or unspecified.
    The name of a context cannot be empty, and it should be unique among
    contexts of the same ContextType.

    Args:
      contexts: A list of contexts to insert or update.

    Returns:
      A list of context ids index-aligned with the input.
    """
    request = metadata_store_service_pb2.PutContextsRequest()
    for x in contexts:
      request.contexts.add().CopyFrom(x)
    response = metadata_store_service_pb2.PutContextsResponse()

    self._call('PutContexts', request, response)
    result = []
    for x in response.context_ids:
      result.append(x)
    return result

  def put_context_type(self,
                       context_type: metadata_store_pb2.ContextType,
                       can_add_fields: bool = False,
                       can_delete_fields: bool = False,
                       all_fields_match: bool = True) -> int:
    """Inserts or updates a context type.

    Similar to put artifact/execution type, if no context type exists in the
    database with the given name, it creates a new context type (and a
    database).

    If a context type with the same name already exists (let's call it
    old_context_type), then the impact depends upon the other options.

    If context_type == old_context_type, then nothing happens.

    Otherwise, if there is a field where context_type and old_context_type
    have different types, then it fails.

    Otherwise, if can_add_fields is False and context_type has a field
    old_context_type is missing, then it fails.

    Otherwise, if all_fields_match is True and old_context_type has a field
    context_type is missing, then it fails.

    Otherwise, if can_delete_fields is True and old_context_type has a field
    context_type is missing, then it deletes that field.

    Otherwise, it does nothing.

    Args:
      context_type: the type to add or update.
      can_add_fields: if true, you can add fields with this operation. If false,
        then if there are more fields in context_type than in the database, the
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
    request = metadata_store_service_pb2.PutContextTypeRequest()
    request.can_add_fields = can_add_fields
    request.can_delete_fields = can_delete_fields
    request.all_fields_match = all_fields_match
    request.context_type.CopyFrom(context_type)
    response = metadata_store_service_pb2.PutContextTypeResponse()
    self._call('PutContextType', request, response)
    return response.type_id

  def put_events(self, events: Sequence[metadata_store_pb2.Event]) -> None:
    """Inserts events in the database.

    The execution_id and artifact_id must already exist.
    Once created, events cannot be modified.

    Args:
      events: A list of events to insert.
    """
    request = metadata_store_service_pb2.PutEventsRequest()
    for x in events:
      request.events.add().CopyFrom(x)
    response = metadata_store_service_pb2.PutEventsResponse()

    self._call('PutEvents', request, response)

  def put_execution(
      self, execution: metadata_store_pb2.Execution,
      artifact_and_events: Sequence[Tuple[metadata_store_pb2.Artifact,
                                          Optional[metadata_store_pb2.Event]]],
      contexts: Optional[Sequence[metadata_store_pb2.Context]]
  ) -> Tuple[int, List[int], List[int]]:
    """Inserts or updates an Execution with artifacts, events and contexts.

    If an execution_id, artifact_id or context_id is specified, it is an update,
    otherwise it does an insertion.

    Args:
      execution: The execution to be created or updated.
      artifact_and_events: a pair of Artifact and Event that the execution uses
        or generates. The event's execution id or artifact id can be empty, as
        the artifact or execution may not be stored beforehand. If given, the
        ids must match with the paired Artifact and the input execution.
      contexts: The Contexts that the execution should be assotiated with and
        the artifacts should be attributed to.

    Returns:
      the execution id, the list of artifact's id, and the list of context's id.
    """
    request = metadata_store_service_pb2.PutExecutionRequest()
    request.execution.CopyFrom(execution)
    # Add artifact_and_event pairs to the request.
    for pair in artifact_and_events:
      artifact_and_event = request.artifact_event_pairs.add()
      artifact_and_event.artifact.CopyFrom(pair[0])
      if len(pair) == 2 and pair[1] is not None:
        artifact_and_event.event.CopyFrom(pair[1])
    # Add contexts to the request.
    for context in contexts:
      context_to_add = request.contexts.add()
      context_to_add.CopyFrom(context)
    response = metadata_store_service_pb2.PutExecutionResponse()

    self._call('PutExecution', request, response)
    artifact_ids = [x for x in response.artifact_ids]
    context_ids = [x for x in response.context_ids]
    return response.execution_id, artifact_ids, context_ids

  def get_artifacts_by_type(
      self, type_name: Text) -> List[metadata_store_pb2.Artifact]:
    """Gets all the artifacts of a given type."""
    request = metadata_store_service_pb2.GetArtifactsByTypeRequest()
    request.type_name = type_name
    response = metadata_store_service_pb2.GetArtifactsByTypeResponse()

    self._call('GetArtifactsByType', request, response)
    result = []
    for x in response.artifacts:
      result.append(x)
    return result

  def get_artifacts_by_uri(self,
                           uri: Text) -> List[metadata_store_pb2.Artifact]:
    """Gets all the artifacts of a given uri."""
    request = metadata_store_service_pb2.GetArtifactsByURIRequest()
    request.uri = uri
    response = metadata_store_service_pb2.GetArtifactsByURIResponse()

    self._call('GetArtifactsByURI', request, response)
    result = []
    for x in response.artifacts:
      result.append(x)
    return result

  def get_artifacts_by_id(
      self, artifact_ids: Sequence[int]) -> List[metadata_store_pb2.Artifact]:
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

    self._call('GetArtifactsByID', request, response)
    result = []
    for x in response.artifacts:
      result.append(x)
    return result

  def get_artifact_type(
      self, type_name: Text) -> Optional[metadata_store_pb2.ArtifactType]:
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

    self._call('GetArtifactType', request, response)
    return response.artifact_type

  def get_artifact_types(self) -> List[metadata_store_pb2.ArtifactType]:
    """Gets all artifact types.

    Returns:
     A list of all known ArtifactTypes.

    Raises:
    tensorflow.errors.InternalError: if query execution fails
    """
    request = metadata_store_service_pb2.GetArtifactTypesRequest()
    response = metadata_store_service_pb2.GetArtifactTypesResponse()

    self._call('GetArtifactTypes', request, response)
    result = []
    for x in response.artifact_types:
      result.append(x)
    return result

  def get_execution_type(
      self, type_name: Text) -> Optional[metadata_store_pb2.ExecutionType]:
    """Gets an execution type by name.

    Args:
     type_name: the type with that name.

    Returns:
     The type with name type_name.

    Raises:
    tensorflow.errors.NotFoundError: if no type exists
    tensorflow.errors.InternalError: if query execution fails
    """
    request = metadata_store_service_pb2.GetExecutionTypeRequest()
    request.type_name = type_name
    response = metadata_store_service_pb2.GetExecutionTypeResponse()

    self._call('GetExecutionType', request, response)
    return response.execution_type

  def get_execution_types(self) -> List[metadata_store_pb2.ExecutionType]:
    """Gets all execution types.

    Returns:
     A list of all known ExecutionTypes.

    Raises:
    tensorflow.errors.InternalError: if query execution fails
    """
    request = metadata_store_service_pb2.GetExecutionTypesRequest()
    response = metadata_store_service_pb2.GetExecutionTypesResponse()

    self._call('GetExecutionTypes', request, response)
    result = []
    for x in response.execution_types:
      result.append(x)
    return result

  def get_context_type(
      self, type_name: Text) -> Optional[metadata_store_pb2.ContextType]:
    """Gets a context type by name.

    Args:
     type_name: the type with that name.

    Returns:
     The type with name type_name.

    Raises:
    tensorflow.errors.NotFoundError: if no type exists
    tensorflow.errors.InternalError: if query execution fails
    """
    request = metadata_store_service_pb2.GetContextTypeRequest()
    request.type_name = type_name
    response = metadata_store_service_pb2.GetContextTypeResponse()

    self._call('GetContextType', request, response)
    return response.context_type

  def get_context_types(self) -> List[metadata_store_pb2.ContextType]:
    """Gets all context types.

    Returns:
     A list of all known ContextTypes.

    Raises:
    tensorflow.errors.InternalError: if query execution fails
    """
    request = metadata_store_service_pb2.GetContextTypesRequest()
    response = metadata_store_service_pb2.GetContextTypesResponse()

    self._call('GetContextTypes', request, response)
    result = []
    for x in response.context_types:
      result.append(x)
    return result

  def get_executions_by_type(
      self, type_name: Text) -> List[metadata_store_pb2.Execution]:
    """Gets all the executions of a given type."""
    request = metadata_store_service_pb2.GetExecutionsByTypeRequest()
    request.type_name = type_name
    response = metadata_store_service_pb2.GetExecutionsByTypeResponse()

    self._call('GetExecutionsByType', request, response)
    result = []
    for x in response.executions:
      result.append(x)
    return result

  def get_executions_by_id(
      self, execution_ids: Sequence[int]) -> List[metadata_store_pb2.Execution]:
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

    self._call('GetExecutionsByID', request, response)
    result = []
    for x in response.executions:
      result.append(x)
    return result

  def get_executions(self) -> List[metadata_store_pb2.Execution]:
    """Gets all executions.

    Returns:
      A list of all executions.

    Raises:
      tensorflow.errors.NotFoundError: if no execution exists
      InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetExecutionsRequest()
    response = metadata_store_service_pb2.GetExecutionsResponse()

    self._call('GetExecutions', request, response)
    result = []
    for x in response.executions:
      result.append(x)
    return result

  def get_artifacts(self) -> List[metadata_store_pb2.Artifact]:
    """Gets all artifacts.

    Returns:
      A list of all artifacts.

    Raises:
      tensorflow.errors.NotFoundError: if no artifact exists
      InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetArtifactsRequest()
    response = metadata_store_service_pb2.GetArtifactsResponse()

    self._call('GetArtifacts', request, response)
    result = []
    for x in response.artifacts:
      result.append(x)
    return result

  def get_contexts(self) -> List[metadata_store_pb2.Context]:
    """Gets all contexts.

    Returns:
      A list of all contexts.

    Raises:
      tensorflow.errors.NotFoundError: if no contexts exists
      InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetContextsRequest()
    response = metadata_store_service_pb2.GetContextsResponse()

    self._call('GetContexts', request, response)
    result = []
    for x in response.contexts:
      result.append(x)
    return result

  def get_contexts_by_id(
      self, context_ids: Sequence[int]) -> List[metadata_store_pb2.Context]:
    """Gets all contexts with matching ids.

    The result is not index-aligned: if an id is not found, it is not returned.

    Args:
      context_ids: A list of context ids to retrieve.

    Returns:
      Contexts with matching ids.
    """
    request = metadata_store_service_pb2.GetContextsByIDRequest()
    for x in context_ids:
      request.context_ids.append(x)
    response = metadata_store_service_pb2.GetContextsByIDResponse()

    self._call('GetContextsByID', request, response)
    result = []
    for x in response.contexts:
      result.append(x)
    return result

  def get_contexts_by_type(self,
                           type_name: Text) -> List[metadata_store_pb2.Context]:
    """Gets all the contexts of a given type.

    Args:
      type_name: The context type name to look for.

    Returns:
      Contexts that matches the context type name.
    """
    request = metadata_store_service_pb2.GetContextsByTypeRequest()
    request.type_name = type_name
    response = metadata_store_service_pb2.GetContextsByTypeResponse()

    self._call('GetContextsByType', request, response)
    result = []
    for x in response.contexts:
      result.append(x)
    return result

  def get_context_by_type_and_name(
      self, type_name: Text,
      context_name: Text) -> Optional[metadata_store_pb2.Context]:
    """Get the context of the given type and context name.

    The API fails if more than one contexts are found.

    Args:
      type_name: The context type name to look for.
      context_name: The context name to look for.

    Returns:
      The Context matching the type and context name.
      None if no matched Context found.
    """
    # TODO(b/139092990): Change get logic to the new C++ API once implemented.
    request = metadata_store_service_pb2.GetContextsByTypeRequest()
    request.type_name = type_name
    response = metadata_store_service_pb2.GetContextsByTypeResponse()

    self._call('GetContextsByType', request, response)
    result = [c for c in response.contexts
              if c.HasField('name') and c.name == context_name]

    assert len(result) <= 1, 'Found more than one contexts with input.'
    if not result:
      return None
    return result[0]

  def get_artifact_types_by_id(
      self, type_ids: Sequence[int]) -> List[metadata_store_pb2.ArtifactType]:
    """Gets artifact types by ID.

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

    self._call('GetArtifactTypesByID', request, response)
    result = []
    for x in response.artifact_types:
      result.append(x)
    return result

  def get_execution_types_by_id(
      self, type_ids: Sequence[int]) -> List[metadata_store_pb2.ExecutionType]:
    """Gets execution types by ID.

    Args:
      type_ids: a sequence of execution type IDs.

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

    self._call('GetExecutionTypesByID', request, response)
    result = []
    for x in response.execution_types:
      result.append(x)
    return result

  def get_context_types_by_id(
      self, type_ids: Sequence[int]) -> List[metadata_store_pb2.ContextType]:
    """Gets context types by ID.

    Args:
      type_ids: a sequence of context type IDs.

    Returns:
      A list of context types.

    Args:
      type_ids: ids to look for.

    Raises:
      InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetContextTypesByIDRequest()
    response = metadata_store_service_pb2.GetContextTypesByIDResponse()
    for x in type_ids:
      request.type_ids.append(x)

    self._call('GetContextTypesByID', request, response)
    result = []
    for x in response.context_types:
      result.append(x)
    return result

  def put_attributions_and_associations(
      self, attributions: Sequence[metadata_store_pb2.Attribution],
      associations: Sequence[metadata_store_pb2.Association]) -> None:
    """Inserts attribution and association relationships in the database.

    The context_id, artifact_id, and execution_id must already exist.
    If the relationship exists, this call does nothing. Once added, the
    relationships cannot be modified.

    Args:
      attributions: A list of attributions to insert.
      associations: A list of associations to insert.
    """
    request = metadata_store_service_pb2.PutAttributionsAndAssociationsRequest()
    for x in attributions:
      request.attributions.add().CopyFrom(x)
    for x in associations:
      request.associations.add().CopyFrom(x)
    response = metadata_store_service_pb2.PutAttributionsAndAssociationsResponse(
    )
    self._call('PutAttributionsAndAssociations', request, response)

  def get_contexts_by_artifact(
      self, artifact_id: int) -> List[metadata_store_pb2.Context]:
    """Gets all context that an artifact is attributed to.

    Args:
      artifact_id: The id of the querying artifact

    Returns:
      Contexts that the artifact is attributed to.
    """
    request = metadata_store_service_pb2.GetContextsByArtifactRequest()
    request.artifact_id = artifact_id
    response = metadata_store_service_pb2.GetContextsByArtifactResponse()

    self._call('GetContextsByArtifact', request, response)
    result = []
    for x in response.contexts:
      result.append(x)
    return result

  def get_contexts_by_execution(
      self, execution_id: int) -> List[metadata_store_pb2.Context]:
    """Gets all context that an execution is associated with.

    Args:
      execution_id: The id of the querying execution

    Returns:
      Contexts that the execution is associated with.
    """
    request = metadata_store_service_pb2.GetContextsByExecutionRequest()
    request.execution_id = execution_id
    response = metadata_store_service_pb2.GetContextsByExecutionResponse()

    self._call('GetContextsByExecution', request, response)
    result = []
    for x in response.contexts:
      result.append(x)
    return result

  def get_artifacts_by_context(
      self, context_id: int) -> List[metadata_store_pb2.Artifact]:
    """Gets all direct artifacts that a context attributes to.

    Args:
      context_id: The id of the querying context

    Returns:
      Artifacts attributing to the context.
    """
    request = metadata_store_service_pb2.GetArtifactsByContextRequest()
    request.context_id = context_id
    response = metadata_store_service_pb2.GetArtifactsByContextResponse()

    self._call('GetArtifactsByContext', request, response)
    result = []
    for x in response.artifacts:
      result.append(x)
    return result

  def get_executions_by_context(
      self, context_id: int) -> List[metadata_store_pb2.Execution]:
    """Gets all direct executions that a context associates with.

    Args:
      context_id: The id of the querying context

    Returns:
      Executions associating with the context.
    """
    request = metadata_store_service_pb2.GetExecutionsByContextRequest()
    request.context_id = context_id
    response = metadata_store_service_pb2.GetExecutionsByContextResponse()

    self._call('GetExecutionsByContext', request, response)
    result = []
    for x in response.executions:
      result.append(x)
    return result

  def get_events_by_execution_ids(
      self, execution_ids: Sequence[int]) -> List[metadata_store_pb2.Event]:
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

    self._call('GetEventsByExecutionIDs', request, response)
    result = []
    for x in response.events:
      result.append(x)
    return result

  def get_events_by_artifact_ids(
      self, artifact_ids: Sequence[int]) -> List[metadata_store_pb2.Event]:
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

    self._call('GetEventsByArtifactIDs', request, response)
    result = []
    for x in response.events:
      result.append(x)
    return result


def downgrade_schema(config: metadata_store_pb2.ConnectionConfig,
                     downgrade_to_schema_version: int) -> None:
  """Downgrades the db specified in the connection config to a schema version.

  If `downgrade_to_schema_version` is greater or equals to zero and less than
  the current library's schema version, it runs a downgrade transaction to
  revert the db schema and migrate the data. The update is transactional, and
  any failure will cause a full rollback of the downgrade. Once the downgrade
  is done, the user needs to use the older version of the library to connect to
  the database.

  Args:
    config: a metadata_store_pb2.ConnectionConfig having the connection params
    downgrade_to_schema_version: downgrades the given database to a specific
      version. For v0.13.2 release, the schema_version is 0. For 0.14.0 and
      0.15.0 release, the schema_version is 4. More details are described in
      g3doc/get_start.md#upgrade-mlmd-library

  Raises:
    InvalidArgumentError: if the `downgrade_to_schema_version` is not given or
      it is negative or greater than the library version.
    RuntimeError: if the downgrade is not finished, return detailed error.
  """
  if downgrade_to_schema_version < 0:
    raise _make_exception('downgrade_to_schema_version not specified',
                          errors.INVALID_ARGUMENT)

  try:
    migration_options = metadata_store_pb2.MigrationOptions()
    migration_options.downgrade_to_schema_version = downgrade_to_schema_version
    metadata_store_serialized.CreateMetadataStore(
        config.SerializeToString(), migration_options.SerializeToString())
  except RuntimeError as e:
    if str(e).startswith('MLMD cannot be downgraded to schema_version'):
      raise _make_exception(str(e), errors.INVALID_ARGUMENT)
    if not str(e).startswith('Downgrade migration was performed.'):
      raise e
    # downgrade is done.
    logging.log(logging.INFO, str(e))


# See  _make_specific_exception in tensorflow.python.framework.errors
def _make_exception(message, error_code):
  try:
    exc_type = errors.exception_type_from_error_code(error_code)
    return exc_type(None, None, message)
  except KeyError:
    return errors.UnknownError(None, None, message, error_code)
