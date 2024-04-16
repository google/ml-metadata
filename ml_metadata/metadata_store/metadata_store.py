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
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from absl import logging
import attr
import grpc

from ml_metadata import errors
from ml_metadata import proto
from ml_metadata.metadata_store.pywrap.metadata_store_extension import metadata_store as metadata_store_serialized
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from ml_metadata.proto import metadata_store_service_pb2_grpc


# Max number of results for one call.
MAX_NUM_RESULT = 100

# Supported field mask paths in LineageGraph message for get_lineage_subgraph().
_ARTIFACTS_FIELD_MASK_PATH = 'artifacts'
_ARTIFACT_TYPES_FIELD_MASK_PATH = 'artifact_types'
_EXECUTIONS_FIELD_MASK_PATH = 'executions'
_EXECUTION_TYPES_FIELD_MASK_PATH = 'execution_types'
_CONTEXTS_FIELD_MASK_PATH = 'contexts'
_CONTEXT_TYPES_FIELD_MASK_PATH = 'context_types'
_EVENTS_FIELD_MASK_PATH = 'events'
_ASSOCIATIONS_FIELD_MASK_PATH = 'associations'
_ATTRIBUTIONS_FIELD_MASK_PATH = 'attributions'


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
    order_by: The field to order the results. If the field is not provided, then
      the order is up to the database backend implementation.
    is_asc: Specifies `order_by` is ascending or descending. If `order_by` is
      not given, the field is ignored. If `order_by` is set, then by default
      ascending order is used for performance benefit.
    filter_query: An optional boolean expression in SQL syntax to specify
      conditions on node attributes and directly connected assets. See
      https://github.com/google/ml-metadata/blob/master/ml_metadata/proto/metadata_store.proto#L705-L783 for the query capabilities and syntax.
  """

  limit: Optional[int] = None
  order_by: Optional[OrderByField] = None
  is_asc: bool = True
  filter_query: Optional[str] = None


class ExtraOptions(object):
  """Dummy Extra options to align with internal MetadataStore."""

  def __init__(self, euc=None):
    """Initialize ExtraOptions."""


class MetadataStore(object):
  """A store for the metadata."""

  def __init__(self, config, enable_upgrade_migration: bool = False):
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
    self._config = config
    self._max_num_retries = 5
    self._service_client_wrapper = None
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
    self._metadata_store_stub = (
        metadata_store_service_pb2_grpc.MetadataStoreServiceStub(channel))
    logging.log(logging.INFO, 'MetadataStore with gRPC connection initialized')
    logging.log(logging.DEBUG, 'ConnectionConfig: %s', config)

  @property
  def pipeline_asset(self) -> None:
    """Returns None as the pipeline asset."""
    logging.log(
        logging.WARNING, 'Getting pipeline asset from an ossed metadata store.'
    )
    return None

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

  def _call(self, method_name, request, response, extra_options=None):
    """Calls method with retry when Aborted error is returned.

    Args:
      method_name: the method to call.
      request: the request protobuf message.
      response: the response protobuf message.
      extra_options: ExtraOptions instance.

    Returns:
      Detailed errors if the method is failed.
    """
    del extra_options
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
        raise errors.make_exception(e.details(), e.code().value[0]) from e  # pytype: disable=attribute-error

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
    [response_str, error_message,
     status_code] = method(self._metadata_store, request.SerializeToString())
    if status_code != 0:
      raise errors.make_exception(error_message.decode('utf-8'), status_code)
    response.ParseFromString(response_str)

  def put_artifacts(
      self,
      artifacts: Sequence[proto.Artifact],
      field_mask_paths: Optional[Sequence[str]] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[int]:
    """Inserts or updates artifacts in the database.

    If an artifact id is specified for an artifact, it is an update.
    If an artifact id is unspecified, it will insert a new artifact.
    For new artifacts, type must be specified.
    For old artifacts, type must be unchanged or unspecified.
    When the name of an artifact is given, it should be unique among artifacts
    of the same ArtifactType.

    It is not guaranteed that the created or updated artifacts will share the
    same `create_time_since_epoch` or `last_update_time_since_epoch` timestamps.

    If `field_mask_paths` is specified and non-empty:
      1. while updating an existing artifact, it only updates fields specified
         in `field_mask_paths`.
      2. while inserting a new artifact, `field_mask_paths` will be ignored.
      3. otherwise, `field_mask_paths` will be applied to all `artifacts`.
    If `field_mask_paths` is unspecified or is empty, it updates the artifact
    as a whole.

    Args:
      artifacts: A list of artifacts to insert or update.
      field_mask_paths: A list of field mask paths for masked update.
      extra_options: ExtraOptions instance.

    Returns:
      A list of artifact ids index-aligned with the input.

    Raises:
      errors.AlreadyExistsError: If artifact's name is specified and it is
        already used by stored artifacts of that ArtifactType.
    """
    del extra_options
    request = metadata_store_service_pb2.PutArtifactsRequest()
    for x in artifacts:
      request.artifacts.add().CopyFrom(x)

    if field_mask_paths:
      for path in field_mask_paths:
        request.update_mask.paths.append(path)
    response = metadata_store_service_pb2.PutArtifactsResponse()

    self._call('PutArtifacts', request, response)
    return list(response.artifact_ids)

  def put_artifact_type(
      self,
      artifact_type: proto.ArtifactType,
      can_add_fields: bool = False,
      can_omit_fields: bool = False,
      extra_options: Optional[ExtraOptions] = None,
  ) -> int:
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
      can_add_fields: when true, new properties can be added; when false,
        returns ALREADY_EXISTS if the request type has properties that are not
        in stored_type.
      can_omit_fields: when true, stored properties can be omitted in the
        request type; when false, returns ALREADY_EXISTS if the stored_type has
        properties not in the request type.
      extra_options: ExtraOptions instance.

    Returns:
      the type_id of the response.

    Raises:
      errors.AlreadyExistsError: If the type is not backward compatible.
      errors.InvalidArgumentError: If the request type has no name, or any
        property value type is unknown.
    """
    del extra_options
    request = metadata_store_service_pb2.PutArtifactTypeRequest(
        can_add_fields=can_add_fields,
        can_omit_fields=can_omit_fields,
        artifact_type=artifact_type)
    response = metadata_store_service_pb2.PutArtifactTypeResponse()
    self._call('PutArtifactType', request, response)
    return response.type_id

  def put_executions(
      self,
      executions: Sequence[proto.Execution],
      field_mask_paths: Optional[Sequence[str]] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[int]:
    """Inserts or updates executions in the database.

    If an execution id is specified for an execution, it is an update.
    If an execution id is unspecified, it will insert a new execution.
    For new executions, type must be specified.
    For old executions, type must be unchanged or unspecified.
    When the name of an execution is given, it should be unique among
    executions of the same ExecutionType.

    It is not guaranteed that the created or updated executions will share the
    same `create_time_since_epoch` or `last_update_time_since_epoch` timestamps.

    If `field_mask_paths` is specified and non-empty:
      1. while updating an existing execution, it only updates fields specified
         in `field_mask_paths`.
      2. while inserting a new execution, `field_mask_paths` will be ignored.
      3. otherwise, `field_mask_paths` will be applied to all `executions`.
    If `field_mask_paths` is unspecified or is empty, it updates the execution
    as a whole.

    Args:
      executions: A list of executions to insert or update.
      field_mask_paths: A list of field mask paths for masked update.
      extra_options: ExtraOptions instance.

    Returns:
      A list of execution ids index-aligned with the input.

    Raises:
      errors.AlreadyExistsError: If execution's name is specified and it is
        already used by stored executions of that ExecutionType.
    """
    del extra_options
    request = metadata_store_service_pb2.PutExecutionsRequest()
    for x in executions:
      request.executions.add().CopyFrom(x)

    if field_mask_paths:
      for path in field_mask_paths:
        request.update_mask.paths.append(path)
    response = metadata_store_service_pb2.PutExecutionsResponse()

    self._call('PutExecutions', request, response)
    return list(response.execution_ids)

  def put_execution_type(
      self,
      execution_type: proto.ExecutionType,
      can_add_fields: bool = False,
      can_omit_fields: bool = False,
      extra_options: Optional[ExtraOptions] = None,
  ) -> int:
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
      can_add_fields: when true, new properties can be added; when false,
        returns ALREADY_EXISTS if the request type has properties that are not
        in stored_type.
      can_omit_fields: when true, stored properties can be omitted in the
        request type; when false, returns ALREADY_EXISTS if the stored_type has
        properties not in the request type.
      extra_options: ExtraOptions instance.

    Returns:
      the type_id of the response.

    Raises:
      errors.AlreadyExistsError: If the type is not backward compatible.
      errors.InvalidArgumentError: If the request type has no name, or any
        property value type is unknown.
    """
    del extra_options
    request = metadata_store_service_pb2.PutExecutionTypeRequest(
        can_add_fields=can_add_fields,
        can_omit_fields=can_omit_fields,
        execution_type=execution_type)
    response = metadata_store_service_pb2.PutExecutionTypeResponse()
    self._call('PutExecutionType', request, response)
    return response.type_id

  def put_contexts(
      self,
      contexts: Sequence[proto.Context],
      field_mask_paths: Optional[Sequence[str]] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[int]:
    """Inserts or updates contexts in the database.

    If an context id is specified for an context, it is an update.
    If an context id is unspecified, it will insert a new context.
    For new contexts, type must be specified.
    For old contexts, type must be unchanged or unspecified.
    The name of a context cannot be empty, and it should be unique among
    contexts of the same ContextType.

    It is not guaranteed that the created or updated contexts will share the
    same `create_time_since_epoch` or `last_update_time_since_epoch` timestamps.

    If `field_mask_paths` is specified and non-empty:
      1. while updating an existing context, it only updates fields specified
         in `field_mask_paths`.
      2. while inserting a new context, `field_mask_paths` will be ignored.
      3. otherwise, `field_mask_paths` will be applied to all `contexts`.
    If `field_mask_paths` is unspecified or is empty, it updates the context
    as a whole.

    Args:
      contexts: A list of contexts to insert or update.
      field_mask_paths: A list of field mask paths for masked update.
      extra_options: ExtraOptions instance.

    Returns:
      A list of context ids index-aligned with the input.

    Raises:
      errors.InvalidArgumentError: If name of the new contexts are empty.
      errors.AlreadyExistsError: If name of the new contexts already used by
        stored contexts of that ContextType.
    """
    del extra_options
    request = metadata_store_service_pb2.PutContextsRequest()
    for x in contexts:
      request.contexts.add().CopyFrom(x)

    if field_mask_paths:
      for path in field_mask_paths:
        request.update_mask.paths.append(path)
    response = metadata_store_service_pb2.PutContextsResponse()

    self._call('PutContexts', request, response)
    return list(response.context_ids)

  def put_context_type(
      self,
      context_type: proto.ContextType,
      can_add_fields: bool = False,
      can_omit_fields: bool = False,
      extra_options: Optional[ExtraOptions] = None,
  ) -> int:
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
      can_add_fields: when true, new properties can be added; when false,
        returns ALREADY_EXISTS if the request type has properties that are not
        in stored_type.
      can_omit_fields: when true, stored properties can be omitted in the
        request type; when false, returns ALREADY_EXISTS if the stored_type has
        properties not in the request type.
      extra_options: ExtraOptions instance.

    Returns:
      the type_id of the response.

    Raises:
      errors.AlreadyExistsError: If the type is not backward compatible.
      errors.InvalidArgumentError: If the request type has no name, or any
        property value type is unknown.
    """
    del extra_options
    request = metadata_store_service_pb2.PutContextTypeRequest(
        can_add_fields=can_add_fields,
        can_omit_fields=can_omit_fields,
        context_type=context_type)
    response = metadata_store_service_pb2.PutContextTypeResponse()
    self._call('PutContextType', request, response)
    return response.type_id

  def put_events(
      self,
      events: Sequence[proto.Event],
      extra_options: Optional[ExtraOptions] = None,
  ) -> None:
    """Inserts events in the database.

    The execution_id and artifact_id must already exist.
    Once created, events cannot be modified.

    It is not guaranteed that the created or updated events will share the
    same `milliseconds_since_epoch` timestamps.

    Args:
      events: A list of events to insert.
      extra_options: ExtraOptions instance.
    """
    del extra_options
    request = metadata_store_service_pb2.PutEventsRequest()
    for x in events:
      request.events.add().CopyFrom(x)
    response = metadata_store_service_pb2.PutEventsResponse()

    self._call('PutEvents', request, response)

  def put_execution(
      self,
      execution: proto.Execution,
      artifact_and_events: Sequence[
          Tuple[proto.Artifact, Optional[proto.Event]]
      ],
      contexts: Optional[Sequence[proto.Context]],
      reuse_context_if_already_exist: bool = False,
      reuse_artifact_if_already_exist_by_external_id: bool = False,
      force_reuse_context: bool = False,
      force_update_time: bool = False,
      extra_options: Optional[ExtraOptions] = None,
  ) -> Tuple[int, List[int], List[int]]:
    """Inserts or updates an Execution with artifacts, events and contexts.

    In contrast with other put methods, the method update an
    execution atomically with its input/output artifacts and events and adds
    attributions and associations to related contexts.

    If an execution_id, artifact_id or context_id is specified, it is an update,
    otherwise it does an insertion.

    It is not guaranteed that the created or updated executions, artifacts,
    contexts and events will share the same `create_time_since_epoch`,
    `last_update_time_since_epoch`, or `milliseconds_since_epoch` timestamps.

    Args:
      execution: The execution to be created or updated.
      artifact_and_events: a pair of Artifact and Event that the execution uses
        or generates. The event's execution id or artifact id can be empty, as
        the artifact or execution may not be stored beforehand. If given, the
        ids must match with the paired Artifact and the input execution.
      contexts: The Contexts that the execution should be associated with and
        the artifacts should be attributed to.
      reuse_context_if_already_exist: When there's a race to publish executions
        with a new context (no id) with the same context.name, by default there
        will be one writer succeeds and the rest of the writers fail with
        AlreadyExists errors. If set is to True, failed writers will reuse the
        stored context.
      reuse_artifact_if_already_exist_by_external_id: When there's a race to
        publish executions with a new artifact with the same
        artifact.external_id, by default there'll be one writer succeeds and the
        rest of the writers returning AlreadyExists errors. If set to true and
        an Artifact has non-empty external_id, the API will reuse the stored
        artifact in the transaction and perform an update. Otherwise, it will
        fall back to relying on `id` field to decide if it's update (if `id`
        exists) or insert (if `id` is empty).
      force_reuse_context: If True, for contexts with a context.id, the stored
        context will NOT be updated. For such contexts,  we will only look at
        the context.id to associate the context with the execution.
      force_update_time: If it is true, `last_update_time_since_epoch` is
        updated even if input execution is the same as stored execution.
      extra_options: ExtraOptions instance.

    Returns:
      the execution id, the list of artifact's id, and the list of context's id.

    Raises:
      errors.InvalidArgumentError: If the id of the input nodes do not align
        with the store. Please refer to InvalidArgument errors in other put
        methods.
      errors.AlreadyExistsError: If the new nodes to be created is already
        exists. Please refer to AlreadyExists errors in other put methods.
    """
    del extra_options
    request = metadata_store_service_pb2.PutExecutionRequest(
        execution=execution,
        contexts=contexts,
        options=metadata_store_service_pb2.PutExecutionRequest.Options(
            reuse_context_if_already_exist=reuse_context_if_already_exist,
            reuse_artifact_if_already_exist_by_external_id=(
                reuse_artifact_if_already_exist_by_external_id
            ),
            force_reuse_context=force_reuse_context,
            force_update_time=force_update_time,
        ),
    )
    # Add artifact_and_event pairs to the request.
    for pair in artifact_and_events:
      if pair:
        request.artifact_event_pairs.add(
            artifact=pair[0], event=pair[1] if len(pair) == 2 else None)
    response = metadata_store_service_pb2.PutExecutionResponse()
    self._call('PutExecution', request, response)
    artifact_ids = list(response.artifact_ids)
    context_ids = list(response.context_ids)
    return response.execution_id, artifact_ids, context_ids

  def put_lineage_subgraph(
      self,
      executions: Sequence[proto.Execution],
      artifacts: Sequence[proto.Artifact],
      contexts: Sequence[proto.Context],
      event_edges: Sequence[Tuple[Optional[int], Optional[int], proto.Event]],
      reuse_context_if_already_exist: bool = False,
      reuse_artifact_if_already_exist_by_external_id: bool = False,
      extra_options: Optional[ExtraOptions] = None,
  ) -> Tuple[List[int], List[int], List[int]]:
    """Inserts a collection of executions, artifacts, contexts, and events.

    This method atomically inserts or updates all specified executions,
    artifacts, and events and adds attributions and associations to related
    contexts.

    It is not guaranteed that the created or updated executions, artifacts,
    contexts and events will share the same `create_time_since_epoch`,
    `last_update_time_since_epoch`, or `milliseconds_since_epoch` timestamps.

    Args:
      executions: List of executions to be created or updated.
      artifacts: List of artifacts to be created or updated.
      contexts: List of contexts to be created or reused. Contexts will be
        associated with the inserted executions and attributed to the inserted
        artifacts.
      event_edges: List of event edges in the subgraph to be inserted. Event
        edges are defined as an optional execution_index, an optional
        artifact_index, and a required event. Event edges must have an
        execution_index and/or an event.execution_id. Execution_index
        corresponds to an execution in the executions list at the specified
        index. If both execution_index and event.execution_id are provided, the
        execution ids of the execution and the event must match. The same rules
        apply to artifact_index and event.artifact_id.
      reuse_context_if_already_exist: When there's a race to publish executions
        with a new context (no id) with the same context.name, by default there
        will be one writer that succeeds and the rest of the writers will fail
        with AlreadyExists errors. If set to True, failed writers will reuse the
        stored context.
      reuse_artifact_if_already_exist_by_external_id: When there's a race to
        publish executions with a new artifact with the same
        artifact.external_id, by default there'll be one writer succeeds and the
        rest of the writers returning AlreadyExists errors. If set to true and
        an Artifact has non-empty external_id, the API will reuse the stored
        artifact in the transaction and perform an update. Otherwise, it will
        fall back to relying on `id` field to decide if it's update (if `id`
        exists) or insert (if `id` is empty).
      extra_options: ExtraOptions instance.

    Returns:
      The lists of execution ids, artifact ids, and context ids index aligned
      to the input executions, artifacts, and contexts.

    Raises:
      errors.InvalidArgumentError: If the id of the input nodes do not align
        with the store. Please refer to InvalidArgument errors in other put
        methods.
      errors.AlreadyExistsError: If the new nodes to be created already exist.
        Please refer to AlreadyExists errors in other put methods.
      errors.OutOfRangeError: If event_edge indices do not correspond to
        existing indices in the input lists of executions and artifacts.
    """
    del extra_options
    request = metadata_store_service_pb2.PutLineageSubgraphRequest(
        executions=executions,
        artifacts=artifacts,
        contexts=contexts,
        options=metadata_store_service_pb2.PutLineageSubgraphRequest.Options(
            reuse_context_if_already_exist=reuse_context_if_already_exist,
            reuse_artifact_if_already_exist_by_external_id=(
                reuse_artifact_if_already_exist_by_external_id)))

    # Add event edges to the request
    for execution_index, artifact_index, event in event_edges:
      request.event_edges.add(
          execution_index=execution_index,
          artifact_index=artifact_index,
          event=event)

    response = metadata_store_service_pb2.PutLineageSubgraphResponse()
    self._call('PutLineageSubgraph', request, response)
    execution_ids = list(response.execution_ids)
    artifact_ids = list(response.artifact_ids)
    context_ids = list(response.context_ids)
    return execution_ids, artifact_ids, context_ids

  def get_lineage_subgraph(
      self,
      query_options: metadata_store_pb2.LineageSubgraphQueryOptions,
      field_mask_paths: Optional[Sequence[str]] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> metadata_store_pb2.LineageGraph:
    """Gets lineage graph including fields specified in a field mask.

    Args:
      query_options: metadata_store_pb2.LineageSubgraphQueryOptions object. It
        allows users to specify query options for lineage graph tracing from a
        list of interested nodes (limited to 100). Please refer to
        LineageSubgraphQueryOptions for more details.
      field_mask_paths: a list of user specified paths of fields that should be
        included in the returned lineage graph.
        If `field_mask_paths` is specified and non-empty:
          1. If 'artifacts', 'executions', or 'contexts' is specified in
          `read_mask`, the nodes with details will be included.
          2. If 'artifact_types', 'execution_types', or 'context_types' is
          specified in `read_mask`, all the node types with matched `type_id`
          in nodes in the returned graph will be included.
          3. If 'events', 'associations', or 'attributions' is specified in
          `read_mask`, the corresponding edges will be included.
        If `field_mask_paths` is unspecified or is empty, it will return all the
        fields in the returned graph.
      extra_options: ExtraOptions instance.

    Returns:
      metadata_store_pb2.LineageGraph object that contains the lineage graph.
    """
    del extra_options
    request = metadata_store_service_pb2.GetLineageSubgraphRequest(
        lineage_subgraph_query_options=query_options
    )
    if not field_mask_paths:
      field_mask_paths = [
          field.name
          for field in metadata_store_pb2.LineageGraph.DESCRIPTOR.fields
      ]
    # Do not get types from GetLineageSubgraph API, but send extra RPCs after
    # retrieving node details.
    request.read_mask.paths.extend(
        path for path in field_mask_paths if not path.endswith('_types')
    )
    response = metadata_store_service_pb2.GetLineageSubgraphResponse()
    self._call('GetLineageSubgraph', request, response)
    skeleton = response.lineage_subgraph

    lineage_subgraph = metadata_store_pb2.LineageGraph()
    if (
        _ARTIFACTS_FIELD_MASK_PATH in field_mask_paths
        or _ARTIFACT_TYPES_FIELD_MASK_PATH in field_mask_paths
    ):
      artifacts, artifact_types = self.get_artifacts_and_types_by_artifact_ids(
          artifact.id for artifact in skeleton.artifacts
      )
      if _ARTIFACTS_FIELD_MASK_PATH in field_mask_paths:
        lineage_subgraph.artifacts.extend(artifacts)
      if _ARTIFACT_TYPES_FIELD_MASK_PATH in field_mask_paths:
        lineage_subgraph.artifact_types.extend(artifact_types)

    # TODO(b/289277521): Use 1 rpc to get both executions and execution types.
    if (
        _EXECUTIONS_FIELD_MASK_PATH in field_mask_paths
        or _EXECUTION_TYPES_FIELD_MASK_PATH in field_mask_paths
    ):
      executions = self.get_executions_by_id(
          execution.id for execution in skeleton.executions
      )
      if _EXECUTIONS_FIELD_MASK_PATH in field_mask_paths:
        lineage_subgraph.executions.extend(executions)
      if _EXECUTION_TYPES_FIELD_MASK_PATH in field_mask_paths:
        execution_types = self.get_execution_types_by_id(
            set(execution.type_id for execution in executions)
        )
        lineage_subgraph.execution_types.extend(execution_types)

    # TODO(b/289277521): Use 1 rpc to get both contexts and context types.
    if (
        _CONTEXTS_FIELD_MASK_PATH in field_mask_paths
        or _CONTEXT_TYPES_FIELD_MASK_PATH in field_mask_paths
    ):
      contexts = self.get_contexts_by_id(
          context.id for context in skeleton.contexts
      )
      if _CONTEXTS_FIELD_MASK_PATH in field_mask_paths:
        lineage_subgraph.contexts.extend(contexts)
      if _CONTEXT_TYPES_FIELD_MASK_PATH in field_mask_paths:
        context_types = self.get_context_types_by_id(
            set(context.type_id for context in contexts)
        )
        lineage_subgraph.context_types.extend(context_types)

    if _EVENTS_FIELD_MASK_PATH in field_mask_paths:
      lineage_subgraph.events.extend(skeleton.events)

    if _ASSOCIATIONS_FIELD_MASK_PATH in field_mask_paths:
      lineage_subgraph.associations.extend(skeleton.associations)

    if _ATTRIBUTIONS_FIELD_MASK_PATH in field_mask_paths:
      lineage_subgraph.attributions.extend(skeleton.attributions)

    return lineage_subgraph

  def get_artifacts_by_type(
      self,
      type_name: str,
      type_version: Optional[str] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Artifact]:
    """Gets all the artifacts of a given type.

    Args:
      type_name: The artifact type name to look for.
      type_version: An optional artifact type version. If not given, then only
        the type_name are used to look for the artifacts with default version.
      extra_options: ExtraOptions instance.

    Returns:
      Artifacts that matches the type.
    """
    del extra_options
    request = metadata_store_service_pb2.GetArtifactsByTypeRequest()
    request.type_name = type_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetArtifactsByTypeResponse()

    self._call('GetArtifactsByType', request, response)
    return list(response.artifacts)

  def get_artifact_by_type_and_name(
      self,
      type_name: str,
      artifact_name: str,
      type_version: Optional[str] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> Optional[proto.Artifact]:
    """Get the artifact of the given type and name.

    The API fails if more than one artifact is found.

    Args:
      type_name: The artifact type name to look for.
      artifact_name: The artifact name to look for.
      type_version: An optional artifact type version. If not given, then only
        the type_name and artifact_name are used to look for the artifact with
        default version.
      extra_options: ExtraOptions instance.

    Returns:
      The Artifact matching the type and name.
      None if no matched Artifact was found.
    """
    del extra_options
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

  def get_artifacts_by_uri(
      self, uri: str, extra_options: Optional[ExtraOptions] = None
  ) -> List[proto.Artifact]:
    """Gets all the artifacts of a given uri.

    Args:
      uri: The artifact uri to look for.
      extra_options: ExtraOptions instance.

    Returns:
      The Artifacts matching the uri.
    """
    del extra_options
    request = metadata_store_service_pb2.GetArtifactsByURIRequest()
    request.uris.append(uri)
    response = metadata_store_service_pb2.GetArtifactsByURIResponse()

    self._call('GetArtifactsByURI', request, response)
    return list(response.artifacts)

  def get_artifacts_by_id(
      self,
      artifact_ids: Iterable[int],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Artifact]:
    """Gets all artifacts with matching ids.

    The result is not index-aligned: if an id is not found, it is not returned.

    Args:
      artifact_ids: A list of artifact ids to retrieve.
      extra_options: ExtraOptions instance.

    Returns:
      Artifacts with matching ids.
    """
    del extra_options
    request = metadata_store_service_pb2.GetArtifactsByIDRequest()
    for x in artifact_ids:
      request.artifact_ids.append(x)
    response = metadata_store_service_pb2.GetArtifactsByIDResponse()

    self._call('GetArtifactsByID', request, response)
    return list(response.artifacts)

  def get_artifacts_and_types_by_artifact_ids(
      self,
      artifact_ids: Iterable[int],
      extra_options: Optional[ExtraOptions] = None,
  ) -> Tuple[List[proto.Artifact], List[proto.ArtifactType]]:
    """Gets all artifacts with matching ids and populates types.

    The result is not index-aligned: if an id is not found, it is not returned.

    Args:
      artifact_ids: A list of artifact ids to retrieve.
      extra_options: ExtraOptions instance.

    Returns:
      Artifacts with matching ids and ArtifactTypes which can be matched by
      type_ids from Artifacts. Each ArtifactType contains id, name,
      properties and custom_properties fields.
    """
    del extra_options
    request = metadata_store_service_pb2.GetArtifactsByIDRequest(
        artifact_ids=artifact_ids, populate_artifact_types=True
    )
    response = metadata_store_service_pb2.GetArtifactsByIDResponse()

    self._call('GetArtifactsByID', request, response)
    return list(response.artifacts), list(response.artifact_types)

  def get_artifacts_by_external_ids(
      self,
      external_ids: Iterable[str],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Artifact]:
    """Gets all artifacts with matching external ids.

    Args:
      external_ids: A list of external_ids for retrieving the Artifacts.
      extra_options: ExtraOptions instance.

    Returns:
      Artifacts with matching external_ids.
    """
    del extra_options
    request = metadata_store_service_pb2.GetArtifactsByExternalIdsRequest(
        external_ids=external_ids)
    response = metadata_store_service_pb2.GetArtifactsByExternalIdsResponse()

    self._call('GetArtifactsByExternalIds', request, response)
    return response.artifacts[:]

  def get_artifact_type(
      self,
      type_name: str,
      type_version: Optional[str] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> proto.ArtifactType:
    """Gets an artifact type by name and version.

    Args:
      type_name: the type with that name.
      type_version: an optional version of the type, if not given, then only the
        type_name is used to look for types with no versions.
      extra_options: ExtraOptions instance.

    Returns:
      The type with name type_name and version type version.

    Raises:
      errors.NotFoundError: if no type exists.
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetArtifactTypeRequest()
    request.type_name = type_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetArtifactTypeResponse()

    self._call('GetArtifactType', request, response)
    return response.artifact_type

  def get_artifact_types(
      self, extra_options: Optional[ExtraOptions] = None
  ) -> List[proto.ArtifactType]:
    """Gets all artifact types.

    Args:
      extra_options: ExtraOptions instance.

    Returns:
      A list of all known ArtifactTypes.

    Raises:
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetArtifactTypesRequest()
    response = metadata_store_service_pb2.GetArtifactTypesResponse()

    self._call('GetArtifactTypes', request, response)
    return list(response.artifact_types)

  def get_artifact_types_by_external_ids(
      self,
      external_ids: Iterable[str],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.ArtifactType]:
    """Gets all artifact types with matching external ids.

    Args:
      external_ids: A list of external_ids for retrieving the ArtifactTypes.
      extra_options: ExtraOptions instance.

    Returns:
      ArtifactTypes with matching external_ids.
    """
    del extra_options
    request = metadata_store_service_pb2.GetArtifactTypesByExternalIdsRequest(
        external_ids=external_ids)
    response = (
        metadata_store_service_pb2.GetArtifactTypesByExternalIdsResponse())

    self._call('GetArtifactTypesByExternalIds', request, response)
    return response.artifact_types[:]

  def get_execution_type(
      self,
      type_name: str,
      type_version: Optional[str] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> proto.ExecutionType:
    """Gets an execution type by name and version.

    Args:
      type_name: the type with that name.
      type_version: an optional version of the type, if not given, then only the
        type_name is used to look for types with no versions.
      extra_options: ExtraOptions instance.

    Returns:
      The type with name type_name and version type_version.

    Raises:
      errors.NotFoundError: if no type exists.
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetExecutionTypeRequest()
    request.type_name = type_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetExecutionTypeResponse()

    self._call('GetExecutionType', request, response)
    return response.execution_type

  def get_execution_types(
      self, extra_options: Optional[ExtraOptions] = None
  ) -> List[proto.ExecutionType]:
    """Gets all execution types.

    Args:
      extra_options: ExtraOptions instance.

    Returns:
      A list of all known ExecutionTypes.

    Raises:
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetExecutionTypesRequest()
    response = metadata_store_service_pb2.GetExecutionTypesResponse()

    self._call('GetExecutionTypes', request, response)
    return list(response.execution_types)

  def get_execution_types_by_external_ids(
      self,
      external_ids: Iterable[str],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.ExecutionType]:
    """Gets all execution types with matching external ids.

    Args:
      external_ids: A list of external_ids for retrieving the ExecutionTypes.
      extra_options: ExtraOptions instance.

    Returns:
      ExecutionTypes with matching external_ids.
    """
    del extra_options
    request = metadata_store_service_pb2.GetExecutionTypesByExternalIdsRequest(
        external_ids=external_ids)
    response = (
        metadata_store_service_pb2.GetExecutionTypesByExternalIdsResponse())

    self._call('GetExecutionTypesByExternalIds', request, response)
    return response.execution_types[:]

  def get_context_type(
      self,
      type_name: str,
      type_version: Optional[str] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> proto.ContextType:
    """Gets a context type by name and version.

    Args:
      type_name: the type with that name.
      type_version: an optional version of the type, if not given, then only the
        type_name is used to look for types with no versions.
      extra_options: ExtraOptions instance.

    Returns:
      The type with name type_name and version type_version.

    Raises:
      errors.NotFoundError: if no type exists.
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetContextTypeRequest()
    request.type_name = type_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetContextTypeResponse()

    self._call('GetContextType', request, response)
    return response.context_type

  def get_context_types(
      self, extra_options: Optional[ExtraOptions] = None
  ) -> List[proto.ContextType]:
    """Gets all context types.

    Args:
      extra_options: ExtraOptions instance.

    Returns:
      A list of all known ContextTypes.

    Raises:
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetContextTypesRequest()
    response = metadata_store_service_pb2.GetContextTypesResponse()

    self._call('GetContextTypes', request, response)
    return list(response.context_types)

  def get_context_types_by_external_ids(
      self,
      external_ids: Iterable[str],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.ContextType]:
    """Gets all context types with matching external ids.

    Args:
      external_ids: A list of external_ids for retrieving the ContextTypes.
      extra_options: ExtraOptions instance.

    Returns:
      ContextTypes with matching external_ids.
    """
    del extra_options
    request = metadata_store_service_pb2.GetContextTypesByExternalIdsRequest(
        external_ids=external_ids)
    response = metadata_store_service_pb2.GetContextTypesByExternalIdsResponse()

    self._call('GetContextTypesByExternalIds', request, response)
    return response.context_types[:]

  def get_executions_by_type(
      self,
      type_name: str,
      type_version: Optional[str] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Execution]:
    """Gets all the executions of a given type.

    Args:
      type_name: The execution type name to look for.
      type_version: An optional execution type version. If not given, then only
        the type_name are used to look for the executions with default version.
      extra_options: ExtraOptions instance.

    Returns:
      Executions that matches the type.
    """
    del extra_options
    request = metadata_store_service_pb2.GetExecutionsByTypeRequest()
    request.type_name = type_name
    response = metadata_store_service_pb2.GetExecutionsByTypeResponse()
    if type_version:
      request.type_version = type_version
    self._call('GetExecutionsByType', request, response)
    return list(response.executions)

  def get_execution_by_type_and_name(
      self,
      type_name: str,
      execution_name: str,
      type_version: Optional[str] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> Optional[proto.Execution]:
    """Get the execution of the given type and name.

    The API fails if more than one execution is found.

    Args:
      type_name: The execution type name to look for.
      execution_name: The execution name to look for.
      type_version: An optional execution type version. If not given, then only
        the type_name and execution_name are used to look for the execution with
        default version.
      extra_options: ExtraOptions instance.

    Returns:
      The Execution matching the type and name.
      None if no matched Execution found.
    """
    del extra_options
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
      self,
      execution_ids: Iterable[int],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Execution]:
    """Gets all executions with matching ids.

    The result is not index-aligned: if an id is not found, it is not returned.

    Args:
      execution_ids: A list of execution ids to retrieve.
      extra_options: ExtraOptions instance.

    Returns:
      Executions with matching ids.
    """
    del extra_options
    request = metadata_store_service_pb2.GetExecutionsByIDRequest()
    for x in execution_ids:
      request.execution_ids.append(x)
    response = metadata_store_service_pb2.GetExecutionsByIDResponse()

    self._call('GetExecutionsByID', request, response)
    return list(response.executions)

  def get_executions_by_external_ids(
      self,
      external_ids: Iterable[str],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Execution]:
    """Gets all executions with matching external ids.

    Args:
      external_ids: A list of external_ids for retrieving the Executions.
      extra_options: ExtraOptions instance.

    Returns:
      Executions with matching external_ids.
    """
    del extra_options
    request = metadata_store_service_pb2.GetExecutionsByExternalIdsRequest(
        external_ids=external_ids)
    response = metadata_store_service_pb2.GetExecutionsByExternalIdsResponse()

    self._call('GetExecutionsByExternalIds', request, response)
    return response.executions[:]

  def get_executions(
      self,
      list_options: Optional[ListOptions] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Execution]:
    """Gets executions.

    Args:
      list_options: A set of options to specify the conditions, limit the size
        and adjust order of the returned executions.
      extra_options: ExtraOptions instance.

    Returns:
      A list of executions.

    Raises:
      errors.InternalError: if query execution fails.
      errors.InvalidArgument: if list_options is invalid.
    """
    del extra_options
    request = metadata_store_service_pb2.GetExecutionsRequest()
    return self._call_method_with_list_options('GetExecutions', 'executions',
                                               request, list_options)

  def _call_method_with_list_options(
      self,
      method_name: str,
      entity_field_name: str,
      request_without_list_options: Any,
      list_options: Optional[ListOptions] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[Any]:
    """Apply for loop for functions with list_options in request.

    Args:
      method_name: MLMD method name.
      entity_field_name: Field name to look for in response.
      request_without_list_options: A MLMD API request without setting options.
      list_options: optional list options.
      extra_options: ExtraOptions instance.

    Returns:
      A list of entities.
    """
    del extra_options
    if list_options is not None:
      if list_options.limit and list_options.limit < 1:
        raise errors.make_exception(
            (
                'Invalid list_options.limit value passed. '
                'list_options.limit is expected to be greater than 1'
            ),
            errors.INVALID_ARGUMENT,
        )
    request = request_without_list_options
    return_size = None
    if list_options is not None:
      request.options.max_result_size = MAX_NUM_RESULT
      request.options.order_by_field.is_asc = list_options.is_asc
      if list_options.limit:
        return_size = list_options.limit
      if list_options.order_by:
        request.options.order_by_field.field = list_options.order_by.value
      if list_options.filter_query:
        request.options.filter_query = list_options.filter_query

    result = []
    while True:
      response = getattr(metadata_store_service_pb2, method_name + 'Response')()
      # Updating request max_result_size option to optimize and avoid
      # discarding returned results.
      if return_size and return_size < MAX_NUM_RESULT:
        request.options.max_result_size = return_size

      self._call(method_name, request, response)
      entities = getattr(response, entity_field_name)
      for x in entities:
        result.append(x)

      if return_size:
        return_size = return_size - len(entities)
        if return_size <= 0:
          break

      if (not response.HasField('next_page_token') or
          not response.next_page_token):
        break

      request.options.next_page_token = response.next_page_token
    return result

  def get_artifacts(
      self,
      list_options: Optional[ListOptions] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Artifact]:
    """Gets artifacts.

    Args:
      list_options: A set of options to specify the conditions, limit the size
        and adjust order of the returned artifacts.
      extra_options: ExtraOptions instance.

    Returns:
      A list of artifacts.

    Raises:
      errors.InternalError: if query execution fails.
      errors.InvalidArgument: if list_options is invalid.
    """
    del extra_options
    request = metadata_store_service_pb2.GetArtifactsRequest()
    return self._call_method_with_list_options('GetArtifacts', 'artifacts',
                                               request, list_options)

  def get_contexts(
      self,
      list_options: Optional[ListOptions] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Context]:
    """Gets contexts.

    Args:
      list_options: A set of options to specify the conditions, limit the size
        and adjust order of the returned contexts.
      extra_options: ExtraOptions instance.

    Returns:
      A list of contexts.

    Raises:
      errors.InternalError: if query execution fails.
      errors.InvalidArgument: if list_options is invalid.
    """
    del extra_options
    request = metadata_store_service_pb2.GetContextsRequest()
    return self._call_method_with_list_options('GetContexts', 'contexts',
                                               request, list_options)

  def get_contexts_by_id(
      self,
      context_ids: Iterable[int],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Context]:
    """Gets all contexts with matching ids.

    The result is not index-aligned: if an id is not found, it is not returned.

    Args:
      context_ids: A list of context ids to retrieve.
      extra_options: ExtraOptions instance.

    Returns:
      Contexts with matching ids.
    """
    del extra_options
    request = metadata_store_service_pb2.GetContextsByIDRequest()
    for x in context_ids:
      request.context_ids.append(x)
    response = metadata_store_service_pb2.GetContextsByIDResponse()

    self._call('GetContextsByID', request, response)
    return list(response.contexts)

  def get_contexts_by_type(
      self,
      type_name: str,
      type_version: Optional[str] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Context]:
    """Gets all the contexts of a given type.

    Args:
      type_name: The context type name to look for.
      type_version: An optional context type version. If not given, then only
        the type_name are used to look for the contexts with default version.
      extra_options: ExtraOptions instance.

    Returns:
      Contexts that matches the type.
    """
    del extra_options
    request = metadata_store_service_pb2.GetContextsByTypeRequest()
    request.type_name = type_name
    if type_version:
      request.type_version = type_version
    response = metadata_store_service_pb2.GetContextsByTypeResponse()

    self._call('GetContextsByType', request, response)
    return list(response.contexts)

  def get_context_by_type_and_name(
      self,
      type_name: str,
      context_name: str,
      type_version: Optional[str] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> Optional[proto.Context]:
    """Get the context of the given type and context name.

    The API fails if more than one contexts are found.

    Args:
      type_name: The context type name to look for.
      context_name: The context name to look for.
      type_version: An optional context type version. If not given, then only
        the type_name and context_name are used to look for the context with
        default version.
      extra_options: ExtraOptions instance.

    Returns:
      The Context matching the type and context name.
      None if no matched Context found.
    """
    del extra_options
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

  def get_contexts_by_external_ids(
      self,
      external_ids: Iterable[str],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Context]:
    """Gets all contexts with matching external ids.

    Args:
      external_ids: A list of external_ids for retrieving the Contexts.
      extra_options: ExtraOptions instance.

    Returns:
      Contexts with matching external_ids.
    """
    del extra_options
    request = metadata_store_service_pb2.GetContextsByExternalIdsRequest(
        external_ids=external_ids)
    response = metadata_store_service_pb2.GetContextsByExternalIdsResponse()

    self._call('GetContextsByExternalIds', request, response)
    return response.contexts[:]

  def get_artifact_types_by_id(
      self,
      type_ids: Iterable[int],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.ArtifactType]:
    """Gets artifact types by ID.

    Args:
      type_ids: a sequence of artifact type IDs.
      extra_options: ExtraOptions instance.

    Returns:
      A list of artifact types.

    Raises:
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetArtifactTypesByIDRequest()
    response = metadata_store_service_pb2.GetArtifactTypesByIDResponse()
    for x in type_ids:
      request.type_ids.append(x)

    self._call('GetArtifactTypesByID', request, response)
    return list(response.artifact_types)

  def get_execution_types_by_id(
      self,
      type_ids: Iterable[int],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.ExecutionType]:
    """Gets execution types by ID.

    Args:
      type_ids: a sequence of execution type IDs.
      extra_options: ExtraOptions instance.

    Returns:
      A list of execution types.

    Raises:
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetExecutionTypesByIDRequest()
    response = metadata_store_service_pb2.GetExecutionTypesByIDResponse()
    for x in type_ids:
      request.type_ids.append(x)

    self._call('GetExecutionTypesByID', request, response)
    return list(response.execution_types)

  def get_context_types_by_id(
      self,
      type_ids: Iterable[int],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.ContextType]:
    """Gets context types by ID.

    Args:
      type_ids: a sequence of context type IDs.
      extra_options: ExtraOptions instance.

    Returns:
      A list of context types.

    Raises:
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetContextTypesByIDRequest()
    response = metadata_store_service_pb2.GetContextTypesByIDResponse()
    for x in type_ids:
      request.type_ids.append(x)

    self._call('GetContextTypesByID', request, response)
    return list(response.context_types)

  def put_attributions_and_associations(
      self,
      attributions: Sequence[proto.Attribution],
      associations: Sequence[proto.Association],
      extra_options: Optional[ExtraOptions] = None,
  ) -> None:
    """Inserts attribution and association relationships in the database.

    The context_id, artifact_id, and execution_id must already exist.
    If the relationship exists, this call does nothing. Once added, the
    relationships cannot be modified.

    Args:
      attributions: A list of attributions to insert.
      associations: A list of associations to insert.
      extra_options: ExtraOptions instance.
    """
    del extra_options
    request = metadata_store_service_pb2.PutAttributionsAndAssociationsRequest()
    for x in attributions:
      request.attributions.add().CopyFrom(x)
    for x in associations:
      request.associations.add().CopyFrom(x)
    response = metadata_store_service_pb2.PutAttributionsAndAssociationsResponse(
    )
    self._call('PutAttributionsAndAssociations', request, response)


  def put_parent_contexts(
      self,
      parent_contexts: Sequence[proto.ParentContext],
      extra_options: Optional[ExtraOptions] = None,
  ) -> None:
    """Inserts parent contexts in the database.

    The `child_id` and `parent_id` in every parent context must already exist.

    Args:
      parent_contexts: A list of parent contexts to insert.
      extra_options: ExtraOptions instance.

    Raises:
      errors.InvalidArgumentError: if no context matches the `child_id` or no
        context matches the `parent_id` in any parent context.
      errors.AlreadyExistsError: if the same parent context already exists.
    """
    del extra_options
    request = metadata_store_service_pb2.PutParentContextsRequest()
    for x in parent_contexts:
      request.parent_contexts.add().CopyFrom(x)
    response = metadata_store_service_pb2.PutParentContextsResponse()
    self._call('PutParentContexts', request, response)

  def get_contexts_by_artifact(
      self, artifact_id: int, extra_options: Optional[ExtraOptions] = None
  ) -> List[proto.Context]:
    """Gets all context that an artifact is attributed to.

    Args:
      artifact_id: The id of the querying artifact
      extra_options: ExtraOptions instance.

    Returns:
      Contexts that the artifact is attributed to.
    """
    del extra_options
    request = metadata_store_service_pb2.GetContextsByArtifactRequest()
    request.artifact_id = artifact_id
    response = metadata_store_service_pb2.GetContextsByArtifactResponse()

    self._call('GetContextsByArtifact', request, response)
    return list(response.contexts)

  def get_contexts_by_execution(
      self, execution_id: int, extra_options: Optional[ExtraOptions] = None
  ) -> List[proto.Context]:
    """Gets all context that an execution is associated with.

    Args:
      execution_id: The id of the querying execution
      extra_options: ExtraOptions instance.

    Returns:
      Contexts that the execution is associated with.
    """
    del extra_options
    request = metadata_store_service_pb2.GetContextsByExecutionRequest()
    request.execution_id = execution_id
    response = metadata_store_service_pb2.GetContextsByExecutionResponse()

    self._call('GetContextsByExecution', request, response)
    return list(response.contexts)

  def get_artifacts_by_context(
      self,
      context_id: int,
      list_options: Optional[ListOptions] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Artifact]:
    """Gets all direct artifacts that are attributed to the given context.

    Args:
      context_id: The id of the querying context
      list_options: A set of options to specify the conditions, limit the size
        and adjust order of the returned executions.
      extra_options: ExtraOptions instance.

    Returns:
      Artifacts attributing to the context.
    """
    del extra_options
    request = metadata_store_service_pb2.GetArtifactsByContextRequest()
    request.context_id = context_id
    return self._call_method_with_list_options('GetArtifactsByContext',
                                               'artifacts', request,
                                               list_options)

  def get_executions_by_context(
      self,
      context_id: int,
      list_options: Optional[ListOptions] = None,
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Execution]:
    """Gets all direct executions that a context associates with.

    Args:
      context_id: The id of the querying context
      list_options: A set of options to specify the conditions, limit the size
        and adjust order of the returned executions.
      extra_options: ExtraOptions instance.

    Returns:
      Executions associating with the context.
    """
    del extra_options
    if list_options is None:
      # Default order is CREATE_TIME DESC.
      list_options = ListOptions(
          order_by=OrderByField.CREATE_TIME, is_asc=False)

    request = metadata_store_service_pb2.GetExecutionsByContextRequest()
    request.context_id = context_id
    return self._call_method_with_list_options('GetExecutionsByContext',
                                               'executions', request,
                                               list_options)

  def get_events_by_execution_ids(
      self,
      execution_ids: Iterable[int],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Event]:
    """Gets all events with matching execution ids.

    Args:
      execution_ids: a list of execution ids.
      extra_options: ExtraOptions instance.

    Returns:
      Events with the execution IDs given.

    Raises:
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetEventsByExecutionIDsRequest()
    for x in execution_ids:
      request.execution_ids.append(x)
    response = metadata_store_service_pb2.GetEventsByExecutionIDsResponse()

    self._call('GetEventsByExecutionIDs', request, response)
    return list(response.events)

  def get_events_by_artifact_ids(
      self,
      artifact_ids: Iterable[int],
      extra_options: Optional[ExtraOptions] = None,
  ) -> List[proto.Event]:
    """Gets all events with matching artifact ids.

    Args:
      artifact_ids: a list of artifact ids.
      extra_options: ExtraOptions instance.

    Returns:
      Events with the execution IDs given.

    Raises:
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetEventsByArtifactIDsRequest()
    for x in artifact_ids:
      request.artifact_ids.append(x)
    response = metadata_store_service_pb2.GetEventsByArtifactIDsResponse()

    self._call('GetEventsByArtifactIDs', request, response)
    return list(response.events)

  def get_parent_contexts_by_context(
      self, context_id: int, extra_options: Optional[ExtraOptions] = None
  ) -> List[proto.Context]:
    """Gets all parent contexts of a context.

    Args:
      context_id: The id of the querying context.
      extra_options: ExtraOptions instance.

    Returns:
      Parent contexts of the querying context.

    Raises:
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetParentContextsByContextRequest()
    request.context_id = context_id
    response = metadata_store_service_pb2.GetParentContextsByContextResponse()
    self._call('GetParentContextsByContext', request, response)
    return list(response.contexts)

  def get_children_contexts_by_context(
      self, context_id: int, extra_options: Optional[ExtraOptions] = None
  ) -> List[proto.Context]:
    """Gets all children contexts of a context.

    Args:
      context_id: The id of the querying context.
      extra_options: ExtraOptions instance.

    Returns:
      Children contexts of the querying context.

    Raises:
      errors.InternalError: if query execution fails.
    """
    del extra_options
    request = metadata_store_service_pb2.GetChildrenContextsByContextRequest()
    request.context_id = context_id
    response = metadata_store_service_pb2.GetChildrenContextsByContextResponse()
    self._call('GetChildrenContextsByContext', request, response)
    return list(response.contexts)


def downgrade_schema(
    config: proto.ConnectionConfig, downgrade_to_schema_version: int
) -> None:
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
    raise errors.make_exception(
        'downgrade_to_schema_version not specified', errors.INVALID_ARGUMENT
    )

  try:
    migration_options = metadata_store_pb2.MigrationOptions()
    migration_options.downgrade_to_schema_version = downgrade_to_schema_version
    metadata_store_serialized.CreateMetadataStore(
        config.SerializeToString(), migration_options.SerializeToString())
  except RuntimeError as e:
    if str(e).startswith('MLMD cannot be downgraded to schema_version'):
      raise errors.make_exception(str(e), errors.INVALID_ARGUMENT) from e
    if not str(e).startswith('Downgrade migration was performed.'):
      raise e
    # downgrade is done.
    logging.log(logging.INFO, str(e))
