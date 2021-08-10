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

import enum
import random
import time
from typing import Iterable, List, Optional, Sequence, Text, Tuple, Union

from absl import logging
import attr
import grpc

from ml_metadata import errors
from ml_metadata import proto
from ml_metadata.metadata_store.pywrap.metadata_store_extension import metadata_store as metadata_store_serialized
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from ml_metadata.proto import metadata_store_service_pb2_grpc


@enum.unique
class OrderByField(enum.Enum):
  """Defines the available fields to order results in ListOperations."""

  CREATE_TIME = (
      metadata_store_pb2.ListOperationOptions.OrderByField.Field.CREATE_TIME)
  UPDATE_TIME = (
      metadata_store_pb2.ListOperationOptions.OrderByField.Field
      .LAST_UPDATE_TIME)
  ID = metadata_store_pb2.ListOperationOptions.OrderByField.Field.ID


@attr.s(auto_attribs=True)
class ListOptions(object):
  """Defines the available options when listing nodes.

  Attributes:
    limit: The maximum size of the result. If a value is not specified then all
      artifacts are returned.
    order_by: The field to order the results. If the field is not provided,
      then the order is up to the database backend implementation.
    is_asc: Specifies `order_by` is ascending or descending. If `order_by` is
      not given, the field is ignored. If `order_by` is set, then by default
      descending order is used.
    filter_query: An optional boolean expression in SQL syntax to specify
      conditions on nodes' attributes and 1-hop neighborhood. See
      https://github.com/google/ml-metadata/blob/master/ml_metadata/proto/metadata_store.proto#L705-L783
      for the query capabilities and syntax.
      Note in Windows, this is only supported when using grpc config via an mlmd
      server instance.
  """

  limit: Optional[int] = None
  order_by: Optional[OrderByField] = None
  is_asc: bool = False
  filter_query: Optional[str] = None


class MetadataStore(object):
  """A store for the metadata."""

  def __init__(self,
               config: Union[proto.ConnectionConfig,
                             proto.MetadataStoreClientConfig],
               enable_upgrade_migration: bool = False):
    """Initialize the MetadataStore.

    MetadataStore can directly connect to either the metadata database or
    the MLMD MetadataStore gRPC server.

    Args:
      config: `proto.ConnectionConfig` or `proto.MetadataStoreClientConfig`.
        Configuration to connect to the database or the metadata store server.
      enable_upgrade_migration: if set to True, the library upgrades the db
        schema and migrates all data if it connects to an old version backend.
        It is ignored when using gRPC `proto.MetadataStoreClientConfig`.
    """
    self._max_num_retries = 5
    if isinstance(config, proto.ConnectionConfig):
      self._using_db_connection = True
      migration_options = metadata_store_pb2.MigrationOptions()
      migration_options.enable_upgrade_migration = enable_upgrade_migration
      self._metadata_store = metadata_store_serialized.CreateMetadataStore(
          config.SerializeToString(), migration_options.SerializeToString())
      logging.log(logging.INFO, 'MetadataStore with DB connection initialized')
      logging.log(logging.DEBUG, 'ConnectionConfig: %s', config)
      if config.HasField('retry_options'):
        self._max_num_retries = config.retry_options.max_num_retries
        logging.log(logging.INFO,
                    'retry options is overwritten: max_num_retries = %d',
                    self._max_num_retries)
      return
    if not isinstance(config, proto.MetadataStoreClientConfig):
      raise ValueError('MetadataStore is expecting either '
                       'proto.ConnectionConfig or '
                       'proto.MetadataStoreClientConfig')
    self._grpc_timeout_sec = None
    self._using_db_connection = False
    if enable_upgrade_migration:
      raise ValueError('Upgrade migration is not allowed when using gRPC '
                       'connection client. Upgrade needs to be performed on '
                       'the server side.')
    channel = self._get_channel(config)
    self._metadata_store_stub = (metadata_store_service_pb2_grpc.
                                 MetadataStoreServiceStub(channel))
    logging.log(logging.INFO, 'MetadataStore with gRPC connection initialized')
    logging.log(logging.DEBUG, 'ConnectionConfig: %s', config)

  def _get_channel(self, config: proto.MetadataStoreClientConfig):
    """Configures the channel, which could be secure or insecure.

    It returns a channel that can be specified to be secure or insecure,
    depending on whether ssl_config is specified in the config.

    Args:
      config: proto.MetadataStoreClientConfig.

    Returns:
      an initialized gRPC channel.
    """
    target = ':'.join([config.host, str(config.port)])

    if config.HasField('client_timeout_sec'):
      self._grpc_timeout_sec = config.client_timeout_sec

    options = []
    if config.HasField('channel_arguments'):
      if config.channel_arguments.HasField('max_receive_message_length'):
        options.append(('grpc.max_receive_message_length',
                        config.channel_arguments.max_receive_message_length))
      if config.channel_arguments.HasField('http2_max_ping_strikes'):
        options.append(('grpc.http2.max_ping_strikes',
                        config.channel_arguments.http2_max_ping_strikes))

    if not config.HasField('ssl_config'):
      return grpc.insecure_channel(target, options=options)

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
    return grpc.secure_channel(target, credentials, options=options)

  def _call(self, method_name, request, response):
    """Calls method with retry when Aborted error is returned.

    Args:
      method_name: the method to call.
      request: the request protobuf message.
      response: the response protobuf message.

    Returns:
      Detailed errors if the method is failed.
    """
    num_retries = self._max_num_retries
    avg_delay_sec = 2
    while True:
      try:
        return self._call_method(method_name, request, response)
      except errors.AbortedError:
        num_retries -= 1
        if num_retries == 0:
          logging.log(logging.ERROR, '%s failed after retrying %d times.',
                      method_name, self._max_num_retries)
          raise
        wait_seconds = random.expovariate(1.0 / avg_delay_sec)
        logging.log(logging.INFO, 'mlmd client retry in %f secs', wait_seconds)
        time.sleep(wait_seconds)

  def _call_method(self, method_name, request, response) -> None:
    """Calls method using wrapped C++ library or gRPC.

    Args:
      method_name: the method to call in wrapped C++ library or gRPC.
      request: a protobuf message, serialized and sent to the method.
      response: a protobuf message, filled from the return value of the method.
    """
    if self._using_db_connection:
      cc_method = getattr(metadata_store_serialized, method_name)
      self._pywrap_cc_call(cc_method, request, response)
    else:
      grpc_method = getattr(self._metadata_store_stub, method_name)
      try:
        response.CopyFrom(grpc_method(request, timeout=self._grpc_timeout_sec))
      except grpc.RpcError as e:
        # RpcError code uses a tuple to specify error code and short
        # description.
        # https://grpc.github.io/grpc/python/_modules/grpc.html#StatusCode
        raise _make_exception(e.details(), e.code().value[0])  # pytype: disable=attribute-error

  def _pywrap_cc_call(self, method, request, response) -> None:
    """Calls method, serializing and deserializing inputs and outputs.

    Note that this does not check the types of request and response.

    This can throw a variety of Python errors, based upon the underlying
    errors returned in MetadataStore. See _CODE_TO_EXCEPTION_CLASS in
    ml_metadata/errors.py for the mapping.

    Args:
      method: the method to call exposed in the pybind11 module.
      request: a protobuf message, serialized and sent to the method.
      response: a protobuf message, filled from the return value of the method.

    Raises:
      Error: ml_metadata error returned by the method.
    """
    [response_str, error_message, status_code] = method(
        self._metadata_store, request.SerializeToString())
    if status_code != 0:
      raise _make_exception(error_message.decode('utf-8'), status_code)
    response.ParseFromString(response_str)

  def put_artifacts(self, artifacts: Sequence[proto.Artifact]) -> List[int]:
    """Inserts or updates artifacts in the database.

    If an artifact id is specified for an artifact, it is an update.
    If an artifact id is unspecified, it will insert a new artifact.
    For new artifacts, type must be specified.
    For old artifacts, type must be unchanged or unspecified.
    When the name of an artifact is given, it should be unique among artifacts
    of the same ArtifactType.

    Args:
      artifacts: A list of artifacts to insert or update.

    Returns:
      A list of artifact ids index-aligned with the input.

    Raises:
      errors.AlreadyExistsError: If artifact's name is specified and it is
        already used by stored artifacts of that ArtifactType.
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
                        artifact_type: proto.ArtifactType,
                        can_add_fields: bool = False,
                        can_omit_fields: bool = False) -> int:
    """Inserts or updates an artifact type.

    A type has a set of strong typed properties describing the schema of any
    stored instance associated with that type. A type is identified by a name
    and an optional version.

    Type Creation:
    If no type exists in the database with the given identifier
    (name, version), it creates a new type and returns the type_id.

    Type Evolution:
    If the request type with the same (name, version) already exists
    (let's call it stored_type), the method enforces the stored_type can be
    updated only when the request type is backward compatible for the already
    stored instances.

    Backwards compatibility is violated iff:

      1. there is a property where the request type and stored_type have
         different value type (e.g., int vs. string)
      2. `can_add_fields = false` and the request type has a new property that
         is not stored.
      3. `can_omit_fields = false` and stored_type has an existing property
         that is not provided in the request type.

    If non-backward type change is required in the application, e.g.,
    deprecate properties, re-purpose property name, change value types,
    a new type can be created with a different (name, version) identifier.
    Note the type version is optional, and a version value with empty string
    is treated as unset.

    Args:
      artifact_type: the request type to be inserted or updated.
      can_add_fields:
        when true, new properties can be added;
        when false, returns ALREADY_EXISTS if the request type has properties
        that are not in stored_type.
      can_omit_fields:
        when true, stored properties can be omitted in the request type;
        when false, returns ALREADY_EXISTS if the stored_type has properties
        not in the request type.

    Returns:
      the type_id of the response.

    Raises:
      errors.AlreadyExistsError: If the type is not backward compatible.
      errors.InvalidArgumentError: If the request type has no name, or any
        property value type is unknown.
    """
    request = metadata_store_service_pb2.PutArtifactTypeRequest(
        can_add_fields=can_add_fields,
        can_omit_fields=can_omit_fields,
        artifact_type=artifact_type)
    response = metadata_store_service_pb2.PutArtifactTypeResponse()
    self._call('PutArtifactType', request, response)
    return response.type_id

  def put_executions(self, executions: Sequence[proto.Execution]) -> List[int]:
    """Inserts or updates executions in the database.

    If an execution id is specified for an execution, it is an update.
    If an execution id is unspecified, it will insert a new execution.
    For new executions, type must be specified.
    For old executions, type must be unchanged or unspecified.
    When the name of an execution is given, it should be unique among
    executions of the same ExecutionType.

    Args:
      executions: A list of executions to insert or update.

    Returns:
      A list of execution ids index-aligned with the input.

    Raises:
      errors.AlreadyExistsError: If execution's name is specified and it is
        already used by stored executions of that ExecutionType.
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
                         execution_type: proto.ExecutionType,
                         can_add_fields: bool = False,
                         can_omit_fields: bool = False) -> int:
    """Inserts or updates an execution type.

    A type has a set of strong typed properties describing the schema of any
    stored instance associated with that type. A type is identified by a name
    and an optional version.

    Type Creation:
    If no type exists in the database with the given identifier
    (name, version), it creates a new type and returns the type_id.

    Type Evolution:
    If the request type with the same (name, version) already exists
    (let's call it stored_type), the method enforces the stored_type can be
    updated only when the request type is backward compatible for the already
    stored instances.

    Backwards compatibility is violated iff:

      1. there is a property where the request type and stored_type have
         different value type (e.g., int vs. string)
      2. `can_add_fields = false` and the request type has a new property that
         is not stored.
      3. `can_omit_fields = false` and stored_type has an existing property
         that is not provided in the request type.

    If non-backward type change is required in the application, e.g.,
    deprecate properties, re-purpose property name, change value types,
    a new type can be created with a different (name, version) identifier.
    Note the type version is optional, and a version value with empty string
    is treated as unset.

    Args:
      execution_type: the request type to be inserted or updated.
      can_add_fields:
        when true, new properties can be added;
        when false, returns ALREADY_EXISTS if the request type has properties
        that are not in stored_type.
      can_omit_fields:
        when true, stored properties can be omitted in the request type;
        when false, returns ALREADY_EXISTS if the stored_type has properties
        not in the request type.

    Returns:
      the type_id of the response.

    Raises:
      errors.AlreadyExistsError: If the type is not backward compatible.
      errors.InvalidArgumentError: If the request type has no name, or any
        property value type is unknown.
    """
    request = metadata_store_service_pb2.PutExecutionTypeRequest(
        can_add_fields=can_add_fields,
        can_omit_fields=can_omit_fields,
        execution_type=execution_type)
    response = metadata_store_service_pb2.PutExecutionTypeResponse()
    self._call('PutExecutionType', request, response)
    return response.type_id

  def put_contexts(self, contexts: Sequence[proto.Context]) -> List[int]:
    """Inserts or updates contexts in the database.

    If an context id is specified for an context, it is an update.
    If an context id is unspecified, it will insert a new context.
    For new contexts, type must be specified.
    For old contexts, type must be unchanged or unspecified.
    The name of a context cannot be empty, and it should be unique among
    contexts of the same ContextType.

    Args:
      contexts: A list of contexts to insert or update.

    Returns:
      A list of context ids index-aligned with the input.

    Raises:
      errors.InvalidArgumentError: If name of the new contexts are empty.
      errors.AlreadyExistsError: If name of the new contexts already used by
        stored contexts of that ContextType.
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
                       context_type: proto.ContextType,
                       can_add_fields: bool = False,
                       can_omit_fields: bool = False) -> int:
    """Inserts or updates a context type.

    A type has a set of strong typed properties describing the schema of any
    stored instance associated with that type. A type is identified by a name
    and an optional version.

    Type Creation:
    If no type exists in the database with the given identifier
    (name, version), it creates a new type and returns the type_id.

    Type Evolution:
    If the request type with the same (name, version) already exists
    (let's call it stored_type), the method enforces the stored_type can be
    updated only when the request type is backward compatible for the already
    stored instances.

    Backwards compatibility is violated iff:

      1. there is a property where the request type and stored_type have
         different value type (e.g., int vs. string)
      2. `can_add_fields = false` and the request type has a new property that
         is not stored.
      3. `can_omit_fields = false` and stored_type has an existing property
         that is not provided in the request type.

    If non-backward type change is required in the application, e.g.,
    deprecate properties, re-purpose property name, change value types,
    a new type can be created with a different (name, version) identifier.
    Note the type version is optional, and a version value with empty string
    is treated as unset.

    Args:
      context_type: the request type to be inserted or updated.
      can_add_fields:
        when true, new properties can be added;
        when false, returns ALREADY_EXISTS if the request type has properties
        that are not in stored_type.
      can_omit_fields:
        when true, stored properties can be omitted in the request type;
        when false, returns ALREADY_EXISTS if the stored_type has properties
        not in the request type.

    Returns:
      the type_id of the response.

    Raises:
      errors.AlreadyExistsError: If the type is not backward compatible.
      errors.InvalidArgumentError: If the request type has no name, or any
        property value type is unknown.
    """
    request = metadata_store_service_pb2.PutContextTypeRequest(
        can_add_fields=can_add_fields,
        can_omit_fields=can_omit_fields,
        context_type=context_type)
    response = metadata_store_service_pb2.PutContextTypeResponse()
    self._call('PutContextType', request, response)
    return response.type_id

  def put_events(self, events: Sequence[proto.Event]) -> None:
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
      self,
      execution: proto.Execution,
      artifact_and_events: Sequence[Tuple[proto.Artifact,
                                          Optional[proto.Event]]],
      contexts: Optional[Sequence[proto.Context]],
      reuse_context_if_already_exist: bool = False
  ) -> Tuple[int, List[int], List[int]]:
    """Inserts or updates an Execution with artifacts, events and contexts.

    In contrast with other put methods, the method update an
    execution atomically with its input/output artifacts and events and adds
    attributions and associations to related contexts.

    If an execution_id, artifact_id or context_id is specified, it is an update,
    otherwise it does an insertion.

    Args:
      execution: The execution to be created or updated.
      artifact_and_events: a pair of Artifact and Event that the execution uses
        or generates. The event's execution id or artifact id can be empty, as
        the artifact or execution may not be stored beforehand. If given, the
        ids must match with the paired Artifact and the input execution.
      contexts: The Contexts that the execution should be associated with and
        the artifacts should be attributed to.
      reuse_context_if_already_exist: when there's a race to publish executions
        with a new context (no id) with the same context.name, by default there
        will be one writer succeeds and the rest of the writers fail with
        AlreadyExists errors. If set is to True, failed writers will reuse the
        stored context.

    Returns:
      the execution id, the list of artifact's id, and the list of context's id.

    Raises:
      errors.InvalidArgumentError: If the id of the input nodes do not align
        with the store. Please refer to InvalidArgument errors in other put
        methods.
      errors.AlreadyExistsError: If the new nodes to be created is already
        exists. Please refer to AlreadyExists errors in other put methods.
    """
    request = metadata_store_service_pb2.PutExecutionRequest(
        execution=execution,
        contexts=(context for context in contexts),
        options=metadata_store_service_pb2.PutExecutionRequest.Options(
            reuse_context_if_already_exist=reuse_context_if_already_exist))
    # Add artifact_and_event pairs to the request.
    for pair in artifact_and_events:
      if pair:
        request.artifact_event_pairs.add(
            artifact=pair[0], event=pair[1] if len(pair) == 2 else None)
    response = metadata_store_service_pb2.PutExecutionResponse()
    self._call('PutExecution', request, response)
    artifact_ids = [x for x in response.artifact_ids]
    context_ids = [x for x in response.context_ids]
    return response.execution_id, artifact_ids, context_ids

  def get_artifacts_by_type(
      self,
      type_name: Text,
      type_version: Optional[Text] = None) -> List[proto.Artifact]:
    """Gets all the artifacts of a given type.

    Args:
      type_name: The artifact type name to look for.
      type_version: An optional artifact type version. If not given, then only
        the type_name are used to look for the artifacts with default version.

    Returns:
      Artifacts that matches the type.
    """
    request = metadata_store_service_pb2.GetArtifactsByTypeRequest()
    request.type_name = type_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetArtifactsByTypeResponse()

    self._call('GetArtifactsByType', request, response)
    result = []
    for x in response.artifacts:
      result.append(x)
    return result

  def get_artifact_by_type_and_name(
      self,
      type_name: Text,
      artifact_name: Text,
      type_version: Optional[Text] = None) -> Optional[proto.Artifact]:
    """Get the artifact of the given type and name.

    The API fails if more than one artifact is found.

    Args:
      type_name: The artifact type name to look for.
      artifact_name: The artifact name to look for.
      type_version: An optional artifact type version. If not given, then only
        the type_name and artifact_name are used to look for the artifact with
        default version.

    Returns:
      The Artifact matching the type and name.
      None if no matched Artifact was found.
    """
    request = metadata_store_service_pb2.GetArtifactByTypeAndNameRequest()
    request.type_name = type_name
    request.artifact_name = artifact_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetArtifactByTypeAndNameResponse()

    self._call('GetArtifactByTypeAndName', request, response)
    if not response.HasField('artifact'):
      return None
    return response.artifact

  def get_artifacts_by_uri(self, uri: Text) -> List[proto.Artifact]:
    """Gets all the artifacts of a given uri.

    Args:
      uri: The artifact uri to look for.

    Returns:
      The Artifacts matching the uri.
    """
    request = metadata_store_service_pb2.GetArtifactsByURIRequest()
    request.uris.append(uri)
    response = metadata_store_service_pb2.GetArtifactsByURIResponse()

    self._call('GetArtifactsByURI', request, response)
    result = []
    for x in response.artifacts:
      result.append(x)
    return result

  def get_artifacts_by_id(self,
                          artifact_ids: Iterable[int]) -> List[proto.Artifact]:
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
      self,
      type_name: Text,
      type_version: Optional[Text] = None) -> proto.ArtifactType:
    """Gets an artifact type by name and version.

    Args:
      type_name: the type with that name.
      type_version: an optional version of the type, if not given, then only
        the type_name is used to look for types with no versions.

    Returns:
      The type with name type_name and version type version.

    Raises:
      errors.NotFoundError: if no type exists.
      errors.InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetArtifactTypeRequest()
    request.type_name = type_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetArtifactTypeResponse()

    self._call('GetArtifactType', request, response)
    return response.artifact_type

  def get_artifact_types(self) -> List[proto.ArtifactType]:
    """Gets all artifact types.

    Returns:
      A list of all known ArtifactTypes.

    Raises:
      errors.InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetArtifactTypesRequest()
    response = metadata_store_service_pb2.GetArtifactTypesResponse()

    self._call('GetArtifactTypes', request, response)
    result = []
    for x in response.artifact_types:
      result.append(x)
    return result

  def get_execution_type(
      self,
      type_name: Text,
      type_version: Optional[Text] = None) -> proto.ExecutionType:
    """Gets an execution type by name and version.

    Args:
      type_name: the type with that name.
      type_version: an optional version of the type, if not given, then only
        the type_name is used to look for types with no versions.

    Returns:
      The type with name type_name and version type_version.

    Raises:
      errors.NotFoundError: if no type exists.
      errors.InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetExecutionTypeRequest()
    request.type_name = type_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetExecutionTypeResponse()

    self._call('GetExecutionType', request, response)
    return response.execution_type

  def get_execution_types(self) -> List[proto.ExecutionType]:
    """Gets all execution types.

    Returns:
      A list of all known ExecutionTypes.

    Raises:
      errors.InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetExecutionTypesRequest()
    response = metadata_store_service_pb2.GetExecutionTypesResponse()

    self._call('GetExecutionTypes', request, response)
    result = []
    for x in response.execution_types:
      result.append(x)
    return result

  def get_context_type(
      self,
      type_name: Text,
      type_version: Optional[Text] = None) -> proto.ContextType:
    """Gets a context type by name and version.

    Args:
      type_name: the type with that name.
      type_version: an optional version of the type, if not given, then only
        the type_name is used to look for types with no versions.

    Returns:
      The type with name type_name and version type_version.

    Raises:
      errors.NotFoundError: if no type exists.
      errors.InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetContextTypeRequest()
    request.type_name = type_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetContextTypeResponse()

    self._call('GetContextType', request, response)
    return response.context_type

  def get_context_types(self) -> List[proto.ContextType]:
    """Gets all context types.

    Returns:
      A list of all known ContextTypes.

    Raises:
      errors.InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetContextTypesRequest()
    response = metadata_store_service_pb2.GetContextTypesResponse()

    self._call('GetContextTypes', request, response)
    result = []
    for x in response.context_types:
      result.append(x)
    return result

  def get_executions_by_type(
      self,
      type_name: Text,
      type_version: Optional[Text] = None) -> List[proto.Execution]:
    """Gets all the executions of a given type.

    Args:
      type_name: The execution type name to look for.
      type_version: An optional execution type version. If not given, then only
        the type_name are used to look for the executions with default version.

    Returns:
      Executions that matches the type.
    """
    request = metadata_store_service_pb2.GetExecutionsByTypeRequest()
    request.type_name = type_name
    response = metadata_store_service_pb2.GetExecutionsByTypeResponse()
    if type_version:
      request.type_version = type_version
    self._call('GetExecutionsByType', request, response)
    result = []
    for x in response.executions:
      result.append(x)
    return result

  def get_execution_by_type_and_name(
      self,
      type_name: Text,
      execution_name: Text,
      type_version: Optional[Text] = None) -> Optional[proto.Execution]:
    """Get the execution of the given type and name.

    The API fails if more than one execution is found.

    Args:
      type_name: The execution type name to look for.
      execution_name: The execution name to look for.
      type_version: An optional execution type version. If not given, then only
        the type_name and execution_name are used to look for the execution with
        default version.

    Returns:
      The Execution matching the type and name.
      None if no matched Execution found.
    """
    request = metadata_store_service_pb2.GetExecutionByTypeAndNameRequest()
    request.type_name = type_name
    request.execution_name = execution_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetExecutionByTypeAndNameResponse()

    self._call('GetExecutionByTypeAndName', request, response)
    if not response.HasField('execution'):
      return None
    return response.execution

  def get_executions_by_id(
      self, execution_ids: Iterable[int]) -> List[proto.Execution]:
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

  def get_executions(
      self,
      list_options: Optional[ListOptions] = None) -> List[proto.Execution]:
    """Gets executions.

    Args:
      list_options: A set of options to specify the conditions, limit the
        size and adjust order of the returned executions.

    Returns:
      A list of executions.

    Raises:
      errors.InternalError: if query execution fails.
      errors.InvalidArgument: if list_options is invalid.
    """
    if list_options:
      if list_options.limit and list_options.limit < 1:
        raise _make_exception(
            'Invalid list_options.limit value passed. list_options.limit '
            'is expected to be greater than 1', errors.INVALID_ARGUMENT)

    request = metadata_store_service_pb2.GetExecutionsRequest()
    return_size = None
    if list_options:
      request.options.max_result_size = 100
      request.options.order_by_field.is_asc = list_options.is_asc
      if list_options.limit:
        return_size = list_options.limit
      if list_options.order_by:
        request.options.order_by_field.field = list_options.order_by.value
      if list_options.filter_query:
        request.options.filter_query = list_options.filter_query  # windows
        # windows 1
        # windows 2
        # windows 3
        # windows 4
        # windows 5
        # windows 6

    result = []
    while True:
      response = metadata_store_service_pb2.GetExecutionsResponse()
      # Updating request max_result_size option to optimize and avoid
      # discarding returned results.
      if return_size and return_size < 100:
        request.options.max_result_size = return_size

      self._call('GetExecutions', request, response)
      for x in response.executions:
        result.append(x)

      if return_size:
        return_size = return_size - len(response.executions)
        if return_size <= 0:
          break

      if not response.HasField('next_page_token'):
        break

      request.options.next_page_token = response.next_page_token

    return result

  def get_artifacts(self,
                    list_options: Optional[ListOptions] = None
                   ) -> List[proto.Artifact]:
    """Gets artifacts.

    Args:
      list_options: A set of options to specify the conditions, limit the
        size and adjust order of the returned artifacts.

    Returns:
      A list of artifacts.

    Raises:
      errors.InternalError: if query execution fails.
      errors.InvalidArgument: if list_options is invalid.
    """

    if list_options:
      if list_options.limit and list_options.limit < 1:
        raise _make_exception(
            'Invalid list_options.limit value passed. list_options.limit is '
            'expected to be greater than 1', errors.INVALID_ARGUMENT)

    request = metadata_store_service_pb2.GetArtifactsRequest()
    return_size = None
    if list_options:
      request.options.max_result_size = 100
      request.options.order_by_field.is_asc = list_options.is_asc
      if list_options.limit:
        return_size = list_options.limit
      if list_options.order_by:
        request.options.order_by_field.field = list_options.order_by.value
      if list_options.filter_query:
        request.options.filter_query = list_options.filter_query  # windows
        # windows 1
        # windows 2
        # windows 3
        # windows 4
        # windows 5
        # windows 6

    result = []
    while True:
      response = metadata_store_service_pb2.GetArtifactsResponse()
      # Updating request max_result_size option to optimize and avoid
      # discarding returned results.
      if return_size and return_size < 100:
        request.options.max_result_size = return_size

      self._call('GetArtifacts', request, response)
      for x in response.artifacts:
        result.append(x)

      if return_size:
        return_size = return_size - len(response.artifacts)
        if return_size <= 0:
          break

      if not response.HasField('next_page_token'):
        break

      request.options.next_page_token = response.next_page_token

    return result

  def get_contexts(self,
                   list_options: Optional[ListOptions] = None
                  ) -> List[proto.Context]:
    """Gets contexts.

    Args:
      list_options: A set of options to specify the conditions, limit the
        size and adjust order of the returned contexts.

    Returns:
      A list of contexts.

    Raises:
      errors.InternalError: if query execution fails.
      errors.InvalidArgument: if list_options is invalid.
    """
    if list_options:
      if list_options.limit and list_options.limit < 1:
        raise _make_exception(
            'Invalid list_options.limit value passed. list_options.limit '
            'is expected to be greater than 1', errors.INVALID_ARGUMENT)

    request = metadata_store_service_pb2.GetContextsRequest()
    return_size = None
    if list_options:
      request.options.max_result_size = 100
      request.options.order_by_field.is_asc = list_options.is_asc
      if list_options.limit:
        return_size = list_options.limit
      if list_options.order_by:
        request.options.order_by_field.field = list_options.order_by.value
      if list_options.filter_query:
        request.options.filter_query = list_options.filter_query  # windows
        # windows 1
        # windows 2
        # windows 3
        # windows 4
        # windows 5
        # windows 6

    result = []
    while True:
      response = metadata_store_service_pb2.GetContextsResponse()
      # Updating request max_result_size option to optimize and avoid
      # discarding returned results.
      if return_size and return_size < 100:
        request.options.max_result_size = return_size

      self._call('GetContexts', request, response)
      for x in response.contexts:
        result.append(x)

      if return_size:
        return_size = return_size - len(response.contexts)
        if return_size <= 0:
          break

      if not response.HasField('next_page_token'):
        break

      request.options.next_page_token = response.next_page_token

    return result

  def get_contexts_by_id(self,
                         context_ids: Iterable[int]) -> List[proto.Context]:
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

  def get_contexts_by_type(
      self,
      type_name: Text,
      type_version: Optional[Text] = None) -> List[proto.Context]:
    """Gets all the contexts of a given type.

    Args:
      type_name: The context type name to look for.
      type_version: An optional context type version. If not given, then only
        the type_name are used to look for the contexts with default version.

    Returns:
      Contexts that matches the type.
    """
    request = metadata_store_service_pb2.GetContextsByTypeRequest()
    request.type_name = type_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetContextsByTypeResponse()

    self._call('GetContextsByType', request, response)
    result = []
    for x in response.contexts:
      result.append(x)
    return result

  def get_context_by_type_and_name(
      self,
      type_name: Text,
      context_name: Text,
      type_version: Optional[Text] = None) -> Optional[proto.Context]:
    """Get the context of the given type and context name.

    The API fails if more than one contexts are found.

    Args:
      type_name: The context type name to look for.
      context_name: The context name to look for.
      type_version: An optional context type version. If not given, then only
        the type_name and context_name are used to look for the context with
        default version.

    Returns:
      The Context matching the type and context name.
      None if no matched Context found.
    """
    request = metadata_store_service_pb2.GetContextByTypeAndNameRequest()
    request.type_name = type_name
    request.context_name = context_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetContextByTypeAndNameResponse()

    self._call('GetContextByTypeAndName', request, response)
    if not response.HasField('context'):
      return None
    return response.context

  def get_artifact_types_by_id(
      self, type_ids: Iterable[int]) -> List[proto.ArtifactType]:
    """Gets artifact types by ID.

    Args:
      type_ids: a sequence of artifact type IDs.

    Returns:
      A list of artifact types.

    Raises:
      errors.InternalError: if query execution fails.
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
      self, type_ids: Iterable[int]) -> List[proto.ExecutionType]:
    """Gets execution types by ID.

    Args:
      type_ids: a sequence of execution type IDs.

    Returns:
      A list of execution types.

    Args:
      type_ids: ids to look for.

    Raises:
      errors.InternalError: if query execution fails.
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
      self, type_ids: Iterable[int]) -> List[proto.ContextType]:
    """Gets context types by ID.

    Args:
      type_ids: a sequence of context type IDs.

    Returns:
      A list of context types.

    Args:
      type_ids: ids to look for.

    Raises:
      errors.InternalError: if query execution fails.
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
      self, attributions: Sequence[proto.Attribution],
      associations: Sequence[proto.Association]) -> None:
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

  def put_parent_contexts(
      self, parent_contexts: Sequence[proto.ParentContext]) -> None:
    """Inserts parent contexts in the database.

    The `child_id` and `parent_id` in every parent context must already exist.

    Args:
      parent_contexts: A list of parent contexts to insert.

    Raises:
      errors.InvalidArgumentError: if no context matches the `child_id` or no
        context matches the `parent_id` in any parent context.
      errors.AlreadyExistsError: if the same parent context already exists.
    """
    request = metadata_store_service_pb2.PutParentContextsRequest()
    for x in parent_contexts:
      request.parent_contexts.add().CopyFrom(x)
    response = metadata_store_service_pb2.PutParentContextsResponse()
    self._call('PutParentContexts', request, response)

  def get_contexts_by_artifact(self, artifact_id: int) -> List[proto.Context]:
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

  def get_contexts_by_execution(self, execution_id: int) -> List[proto.Context]:
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

  def get_artifacts_by_context(self, context_id: int) -> List[proto.Artifact]:
    """Gets all direct artifacts that a context attributes to.

    Args:
      context_id: The id of the querying context

    Returns:
      Artifacts attributing to the context.
    """
    request = metadata_store_service_pb2.GetArtifactsByContextRequest()
    request.context_id = context_id
    request.options.max_result_size = 100
    request.options.order_by_field.field = (
        metadata_store_pb2.ListOperationOptions.OrderByField.Field.CREATE_TIME)
    request.options.order_by_field.is_asc = False

    result = []
    while True:
      response = metadata_store_service_pb2.GetArtifactsByContextResponse()
      self._call('GetArtifactsByContext', request, response)
      for x in response.artifacts:
        result.append(x)

      if not response.next_page_token:
        break

      request.options.next_page_token = response.next_page_token

    return result

  def get_executions_by_context(self, context_id: int) -> List[proto.Execution]:
    """Gets all direct executions that a context associates with.

    Args:
      context_id: The id of the querying context

    Returns:
      Executions associating with the context.
    """
    request = metadata_store_service_pb2.GetExecutionsByContextRequest()
    request.context_id = context_id
    request.options.max_result_size = 100
    request.options.order_by_field.field = (
        metadata_store_pb2.ListOperationOptions.OrderByField.Field.CREATE_TIME)
    request.options.order_by_field.is_asc = False

    result = []
    while True:
      response = metadata_store_service_pb2.GetExecutionsByContextResponse()
      self._call('GetExecutionsByContext', request, response)
      for x in response.executions:
        result.append(x)

      if not response.next_page_token:
        break

      request.options.next_page_token = response.next_page_token

    return result

  def get_events_by_execution_ids(
      self, execution_ids: Iterable[int]) -> List[proto.Event]:
    """Gets all events with matching execution ids.

    Args:
      execution_ids: a list of execution ids.

    Returns:
      Events with the execution IDs given.

    Raises:
      errors.InternalError: if query execution fails.
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
      self, artifact_ids: Iterable[int]) -> List[proto.Event]:
    """Gets all events with matching artifact ids.

    Args:
      artifact_ids: a list of artifact ids.

    Returns:
      Events with the execution IDs given.

    Raises:
      errors.InternalError: if query execution fails.
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

  def get_parent_contexts_by_context(self,
                                     context_id: int) -> List[proto.Context]:
    """Gets all parent contexts of a context.

    Args:
      context_id: The id of the querying context.

    Returns:
      Parent contexts of the querying context.

    Raises:
      errors.InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetParentContextsByContextRequest()
    request.context_id = context_id
    response = metadata_store_service_pb2.GetParentContextsByContextResponse()
    self._call('GetParentContextsByContext', request, response)
    result = []
    for x in response.contexts:
      result.append(x)
    return result

  def get_children_contexts_by_context(self,
                                       context_id: int) -> List[proto.Context]:
    """Gets all children contexts of a context.

    Args:
      context_id: The id of the querying context.

    Returns:
      Children contexts of the querying context.

    Raises:
      errors.InternalError: if query execution fails.
    """
    request = metadata_store_service_pb2.GetChildrenContextsByContextRequest()
    request.context_id = context_id
    response = metadata_store_service_pb2.GetChildrenContextsByContextResponse()
    self._call('GetChildrenContextsByContext', request, response)
    result = []
    for x in response.contexts:
      result.append(x)
    return result


def downgrade_schema(config: proto.ConnectionConfig,
                     downgrade_to_schema_version: int) -> None:
  """Downgrades the db specified in the connection config to a schema version.

  If `downgrade_to_schema_version` is greater or equals to zero and less than
  the current library's schema version, it runs a downgrade transaction to
  revert the db schema and migrate the data. The update is transactional, and
  any failure will cause a full rollback of the downgrade. Once the downgrade
  is done, the user needs to use the older version of the library to connect to
  the database.

  Args:
    config: a `proto.ConnectionConfig` having the connection params.
    downgrade_to_schema_version: downgrades the given database to a specific
      version. For v0.13.2 release, the schema_version is 0. For 0.14.0 and
      0.15.0 release, the schema_version is 4. More details are described in
      g3doc/get_start.md#upgrade-mlmd-library

  Raises:
    errors.InvalidArgumentError: if the `downgrade_to_schema_version` is not
      given or it is negative or greater than the library version.
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


def _make_exception(msg, error_code):
  """Makes an exception with MLMD error code.

  Args:
    msg: Error message.
    error_code: MLMD error code.

  Returns:
    An exception.
  """

  try:
    exc_type = errors.exception_type_from_error_code(error_code)
    # log internal backend engine errors only.
    if error_code == errors.INTERNAL:
      logging.log(logging.WARNING, 'mlmd client %s: %s', exc_type.__name__, msg)
    return exc_type(msg)
  except KeyError:
    return errors.UnknownError(msg)
