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
"""Tests for ml_metadata.metadata_store.metadata_store."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import uuid

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from ml_metadata import errors
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

FLAGS = flags.FLAGS

# TODO(b/145819288) to add SSL related configurations.
flags.DEFINE_boolean(
    "use_grpc_backend", False,
    "Set this to true to use gRPC instead of sqlLite backend.")
flags.DEFINE_string(
    "grpc_host", None,
    "The gRPC host name to use when use_grpc_backed is set to 'True'")
flags.DEFINE_integer(
    "grpc_port", 0,
    "The gRPC port number to use when use_grpc_backed is set to 'True'")


def _get_metadata_store(grpc_max_receive_message_length=None,
                        grpc_client_timeout_sec=None):
  if FLAGS.use_grpc_backend:
    grpc_connection_config = metadata_store_pb2.MetadataStoreClientConfig()
    if grpc_max_receive_message_length:
      (grpc_connection_config.channel_arguments.max_receive_message_length
      ) = grpc_max_receive_message_length
    if grpc_client_timeout_sec is not None:
      grpc_connection_config.client_timeout_sec = grpc_client_timeout_sec
    if FLAGS.grpc_host is None:
      raise ValueError("grpc_host argument not set.")
    grpc_connection_config.host = FLAGS.grpc_host
    if not FLAGS.grpc_port:
      raise ValueError("grpc_port argument not set.")
    grpc_connection_config.port = FLAGS.grpc_port
    return metadata_store.MetadataStore(grpc_connection_config)

  connection_config = metadata_store_pb2.ConnectionConfig()
  connection_config.sqlite.SetInParent()
  return metadata_store.MetadataStore(connection_config)


def _create_example_artifact_type(type_name):
  artifact_type = metadata_store_pb2.ArtifactType()
  artifact_type.name = type_name
  artifact_type.properties["foo"] = metadata_store_pb2.INT
  artifact_type.properties["bar"] = metadata_store_pb2.STRING
  artifact_type.properties["baz"] = metadata_store_pb2.DOUBLE
  return artifact_type


def _create_example_artifact_type_2(type_name):
  artifact_type = metadata_store_pb2.ArtifactType()
  artifact_type.name = type_name
  artifact_type.properties["foo"] = metadata_store_pb2.INT
  artifact_type.properties["bar"] = metadata_store_pb2.STRING
  artifact_type.properties["baz"] = metadata_store_pb2.DOUBLE
  return artifact_type


def _create_example_execution_type(type_name):
  execution_type = metadata_store_pb2.ExecutionType()
  execution_type.name = type_name
  execution_type.properties["foo"] = metadata_store_pb2.INT
  execution_type.properties["bar"] = metadata_store_pb2.STRING
  return execution_type


def _create_example_execution_type_2(type_name):
  execution_type = metadata_store_pb2.ExecutionType()
  execution_type.name = type_name
  execution_type.properties["foo"] = metadata_store_pb2.INT
  execution_type.properties["bar"] = metadata_store_pb2.STRING
  return execution_type


def _create_example_context_type(type_name):
  context_type = metadata_store_pb2.ContextType()
  context_type.name = type_name
  context_type.properties["foo"] = metadata_store_pb2.INT
  context_type.properties["bar"] = metadata_store_pb2.STRING
  context_type.properties["baz"] = metadata_store_pb2.DOUBLE
  return context_type


def _create_example_context_type_2(type_name):
  context_type = metadata_store_pb2.ContextType()
  context_type.name = type_name
  context_type.properties["foo"] = metadata_store_pb2.INT
  context_type.properties["bar"] = metadata_store_pb2.STRING
  return context_type


class MetadataStoreTest(parameterized.TestCase):

  def _get_test_type_name(self):
    return "test_type_{}".format(uuid.uuid4())

  def _get_test_db_name(self):
    return "test_mlmd_{}.db".format(uuid.uuid4())

  def test_unset_connection_config(self):
    connection_config = metadata_store_pb2.ConnectionConfig()
    for _ in range(3):
      with self.assertRaises(RuntimeError):
        metadata_store.MetadataStore(connection_config)

  def test_connection_config_with_retry_options(self):
    # both client and grpc modes have none-zero setting by default.
    store = _get_metadata_store()
    self.assertGreater(store._max_num_retries, 0)
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    want_num_retries = 100
    connection_config.retry_options.max_num_retries = want_num_retries
    store = metadata_store.MetadataStore(connection_config)
    self.assertEqual(store._max_num_retries, want_num_retries)

  def test_connection_config_with_grpc_max_receive_message_length(self):
    # The test is irrelevant when not using grpc connection.
    if not FLAGS.use_grpc_backend:
      return
    # Set max_receive_message_length to 100. The returned artifact type is
    # less than 100 bytes.
    artifact_type_name = self._get_test_type_name()
    artifact_type = _create_example_artifact_type(artifact_type_name)
    self.assertLess(artifact_type.ByteSize(), 100)
    store = _get_metadata_store(grpc_max_receive_message_length=100)
    store.put_artifact_type(artifact_type)
    _ = store.get_artifact_types()

  def test_connection_config_with_grpc_max_receive_message_length_errors(self):
    # The test is irrelevant when not using grpc connection.
    if not FLAGS.use_grpc_backend:
      return
    # Set max_receive_message_length to 1. The client should raise
    # ResourceExhaustedError as the returned artifact type is more than 1 byte.
    store = _get_metadata_store(grpc_max_receive_message_length=1)
    artifact_type_name = self._get_test_type_name()
    artifact_type = _create_example_artifact_type(artifact_type_name)
    with self.assertRaises(errors.ResourceExhaustedError):
      store.put_artifact_type(artifact_type)
      _ = store.get_artifact_types()

  def test_connection_config_with_grpc_client_timeout_sec_errors(self):
    # The test is irrelevant when not using grpc connection.
    if not FLAGS.use_grpc_backend:
      return

    # Set timeout=0 to make sure a rpc call will fail.
    store = _get_metadata_store(grpc_client_timeout_sec=0)
    with self.assertRaises(errors.DeadlineExceededError):
      _ = store.get_artifact_types()

  def test_put_artifact_type_get_artifact_type(self):
    store = _get_metadata_store()
    artifact_type_name = self._get_test_type_name()
    artifact_type = _create_example_artifact_type(artifact_type_name)

    type_id = store.put_artifact_type(artifact_type)
    artifact_type_result = store.get_artifact_type(artifact_type_name)
    self.assertEqual(artifact_type_result.id, type_id)
    self.assertEqual(artifact_type_result.name, artifact_type_name)
    self.assertEqual(artifact_type_result.properties["foo"],
                     metadata_store_pb2.INT)
    self.assertEqual(artifact_type_result.properties["bar"],
                     metadata_store_pb2.STRING)
    self.assertEqual(artifact_type_result.properties["baz"],
                     metadata_store_pb2.DOUBLE)

  def test_put_artifact_type_with_update_get_artifact_type(self):
    store = _get_metadata_store()
    artifact_type_name = self._get_test_type_name()
    artifact_type = _create_example_artifact_type(artifact_type_name)
    type_id = store.put_artifact_type(artifact_type)

    artifact_type.properties["new_property"] = metadata_store_pb2.INT
    store.put_artifact_type(artifact_type, can_add_fields=True)

    artifact_type_result = store.get_artifact_type(artifact_type_name)
    self.assertEqual(artifact_type_result.id, type_id)
    self.assertEqual(artifact_type_result.name, artifact_type_name)
    self.assertEqual(artifact_type_result.properties["foo"],
                     metadata_store_pb2.INT)
    self.assertEqual(artifact_type_result.properties["bar"],
                     metadata_store_pb2.STRING)
    self.assertEqual(artifact_type.properties["baz"], metadata_store_pb2.DOUBLE)
    self.assertEqual(artifact_type.properties["new_property"],
                     metadata_store_pb2.INT)

  @parameterized.parameters(
      (metadata_store_pb2.ArtifactType(), _create_example_artifact_type,
       metadata_store.MetadataStore.put_artifact_type,
       metadata_store.MetadataStore.get_artifact_type),
      (metadata_store_pb2.ExecutionType(), _create_example_execution_type,
       metadata_store.MetadataStore.put_execution_type,
       metadata_store.MetadataStore.get_execution_type),
      (metadata_store_pb2.ContextType(), _create_example_context_type,
       metadata_store.MetadataStore.put_context_type,
       metadata_store.MetadataStore.get_context_type))
  def test_put_type_with_omitted_fields_get_type(self, stored_type,
                                                 create_type_fn, put_type_fn,
                                                 get_type_fn):
    store = _get_metadata_store()
    type_name = self._get_test_type_name()
    base_type = create_type_fn(type_name)
    # store a type by adding more properties
    stored_type.CopyFrom(base_type)
    stored_type.properties["p1"] = metadata_store_pb2.INT
    type_id = put_type_fn(store, stored_type)
    # put a type with missing properties
    with self.assertRaises(errors.AlreadyExistsError):
      put_type_fn(store, base_type)
    # when set can_omit_fields, the upsert is ok
    put_type_fn(store, base_type, can_omit_fields=True)
    # verify the stored type remains the same.
    got_type = get_type_fn(store, type_name)
    self.assertEqual(got_type.id, type_id)
    self.assertEqual(got_type.name, type_name)
    self.assertEqual(got_type.properties, stored_type.properties)

  @parameterized.parameters(
      (metadata_store_pb2.ArtifactType(), _create_example_artifact_type,
       metadata_store.MetadataStore.put_artifact_type,
       metadata_store.MetadataStore.get_artifact_type),
      (metadata_store_pb2.ExecutionType(), _create_example_execution_type,
       metadata_store.MetadataStore.put_execution_type,
       metadata_store.MetadataStore.get_execution_type),
      (metadata_store_pb2.ContextType(), _create_example_context_type,
       metadata_store.MetadataStore.put_context_type,
       metadata_store.MetadataStore.get_context_type))
  def test_put_artifact_type_with_omitted_fields_and_add_fields(
      self, stored_type, create_type_fn, put_type_fn, get_type_fn):
    store = _get_metadata_store()
    type_name = self._get_test_type_name()
    base_type = create_type_fn(type_name)
    # store a type by adding more properties
    stored_type.CopyFrom(base_type)
    stored_type.properties["p1"] = metadata_store_pb2.INT
    type_id = put_type_fn(store, stored_type)
    # put a type with missing properties and an additional property
    base_type.properties["p2"] = metadata_store_pb2.DOUBLE
    # base_type with missing properties cannot be updated
    with self.assertRaises(errors.AlreadyExistsError):
      put_type_fn(store, base_type)
    # base_type with new properties cannot be updated
    with self.assertRaises(errors.AlreadyExistsError):
      put_type_fn(store, base_type, can_omit_fields=True)
    # if both can_add_fields, and can_omit_fields are set, then it succeeds
    put_type_fn(store, base_type, can_add_fields=True, can_omit_fields=True)
    got_type = get_type_fn(store, type_name)
    want_type = stored_type
    want_type.properties["p2"] = metadata_store_pb2.DOUBLE
    self.assertEqual(got_type.id, type_id)
    self.assertEqual(got_type.name, type_name)
    self.assertEqual(got_type.properties, want_type.properties)

  def test_get_artifact_types(self):
    store = _get_metadata_store()
    artifact_type_1 = _create_example_artifact_type(self._get_test_type_name())
    artifact_type_2 = _create_example_artifact_type_2(
        self._get_test_type_name())

    type_id_1 = store.put_artifact_type(artifact_type_1)
    artifact_type_1.id = type_id_1
    type_id_2 = store.put_artifact_type(artifact_type_2)
    artifact_type_2.id = type_id_2

    got_types = store.get_artifact_types()
    got_types = [t for t in got_types if t.id == type_id_1 or t.id == type_id_2]
    got_types.sort(key=lambda x: x.id)
    self.assertListEqual([artifact_type_1, artifact_type_2], got_types)

  def test_get_execution_types(self):
    store = _get_metadata_store()
    execution_type_1 = _create_example_execution_type(
        self._get_test_type_name())
    execution_type_2 = _create_example_execution_type_2(
        self._get_test_type_name())

    type_id_1 = store.put_execution_type(execution_type_1)
    execution_type_1.id = type_id_1
    type_id_2 = store.put_execution_type(execution_type_2)
    execution_type_2.id = type_id_2

    got_types = store.get_execution_types()
    got_types = [t for t in got_types if t.id == type_id_1 or t.id == type_id_2]
    got_types.sort(key=lambda x: x.id)
    self.assertListEqual([execution_type_1, execution_type_2], got_types)

  def test_get_context_types(self):
    if FLAGS.use_grpc_backend:
      return
    store = _get_metadata_store()
    context_type_1 = _create_example_context_type(self._get_test_type_name())
    context_type_2 = _create_example_context_type_2(self._get_test_type_name())

    type_id_1 = store.put_context_type(context_type_1)
    context_type_1.id = type_id_1
    type_id_2 = store.put_context_type(context_type_2)
    context_type_2.id = type_id_2

    got_types = store.get_context_types()
    got_types = [t for t in got_types if t.id == type_id_1 or t.id == type_id_2]
    got_types.sort(key=lambda x: x.id)
    self.assertListEqual([context_type_1, context_type_2], got_types)

  def test_put_artifacts_get_artifacts_by_id(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type(self._get_test_type_name())
    type_id = store.put_artifact_type(artifact_type)
    artifact = metadata_store_pb2.Artifact()
    artifact.type_id = type_id
    artifact.properties["foo"].int_value = 3
    artifact.properties["bar"].string_value = "Hello"
    [artifact_id] = store.put_artifacts([artifact])
    [artifact_result] = store.get_artifacts_by_id([artifact_id])
    self.assertEqual(artifact_result.properties["bar"].string_value, "Hello")
    self.assertEqual(artifact_result.properties["foo"].int_value, 3)

  def test_put_artifacts_get_artifacts_by_id_with_set(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type(self._get_test_type_name())
    type_id = store.put_artifact_type(artifact_type)
    artifact = metadata_store_pb2.Artifact()
    artifact.type_id = type_id
    [artifact_id] = store.put_artifacts([artifact])
    [artifact_result] = store.get_artifacts_by_id({artifact_id})
    self.assertEqual(artifact_result.type_id, artifact.type_id)

  def test_put_artifacts_get_artifacts(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type(self._get_test_type_name())
    type_id = store.put_artifact_type(artifact_type)
    artifact_0 = metadata_store_pb2.Artifact()
    artifact_0.type_id = type_id
    artifact_0.properties["foo"].int_value = 3
    artifact_0.properties["bar"].string_value = "Hello"
    artifact_1 = metadata_store_pb2.Artifact()
    artifact_1.type_id = type_id

    existing_artifacts_count = 0
    try:
      existing_artifacts_count = len(store.get_artifacts())
    except errors.NotFoundError:
      existing_artifacts_count = 0

    [artifact_id_0,
     artifact_id_1] = store.put_artifacts([artifact_0, artifact_1])
    artifact_result = store.get_artifacts()
    new_artifacts_count = len(artifact_result)
    artifact_result = [
        a for a in artifact_result
        if a.id == artifact_id_0 or a.id == artifact_id_1
    ]

    if artifact_result[0].id == artifact_id_0:
      [artifact_result_0, artifact_result_1] = artifact_result
    else:
      [artifact_result_1, artifact_result_0] = artifact_result
    self.assertEqual(existing_artifacts_count + 2, new_artifacts_count)
    self.assertEqual(artifact_result_0.id, artifact_id_0)
    self.assertEqual(artifact_result_0.properties["bar"].string_value, "Hello")
    self.assertEqual(artifact_result_0.properties["foo"].int_value, 3)
    self.assertEqual(artifact_result_1.id, artifact_id_1)

  def test_put_artifacts_get_artifacts_by_type(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type(self._get_test_type_name())
    type_id = store.put_artifact_type(artifact_type)
    artifact_type_2 = _create_example_artifact_type(self._get_test_type_name())
    type_id_2 = store.put_artifact_type(artifact_type_2)
    artifact_0 = metadata_store_pb2.Artifact()
    artifact_0.type_id = type_id
    artifact_0.properties["foo"].int_value = 3
    artifact_0.properties["bar"].string_value = "Hello"
    artifact_1 = metadata_store_pb2.Artifact()
    artifact_1.type_id = type_id_2

    [_, artifact_id_1] = store.put_artifacts([artifact_0, artifact_1])
    artifact_result = store.get_artifacts_by_type(artifact_type_2.name)
    self.assertLen(artifact_result, 1)
    self.assertEqual(artifact_result[0].id, artifact_id_1)

  def test_put_artifacts_get_artifact_by_type_and_name(self):
    # Prepare test data.
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type(self._get_test_type_name())
    type_id = store.put_artifact_type(artifact_type)
    artifact = metadata_store_pb2.Artifact()
    artifact.type_id = type_id
    artifact.name = self._get_test_type_name()
    [artifact_id] = store.put_artifacts([artifact])

    # Test Artifact found case.
    got_artifact = store.get_artifact_by_type_and_name(artifact_type.name,
                                                       artifact.name)
    self.assertEqual(got_artifact.id, artifact_id)
    self.assertEqual(got_artifact.type_id, type_id)
    self.assertEqual(got_artifact.name, artifact.name)

    # Test Artifact not found cases.
    empty_artifact = store.get_artifact_by_type_and_name(
        "random_name", artifact.name)
    self.assertIsNone(empty_artifact)
    empty_artifact = store.get_artifact_by_type_and_name(
        artifact_type.name, "random_name")
    self.assertIsNone(empty_artifact)
    empty_artifact = store.get_artifact_by_type_and_name(
        "random_name", "random_name")
    self.assertIsNone(empty_artifact)

  def test_put_artifacts_get_artifacts_by_uri(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type(self._get_test_type_name())
    type_id = store.put_artifact_type(artifact_type)
    want_artifact = metadata_store_pb2.Artifact()
    want_artifact.type_id = type_id
    want_artifact.uri = "test_uri"
    other_artifact = metadata_store_pb2.Artifact()
    other_artifact.uri = "other_uri"
    other_artifact.type_id = type_id

    [want_artifact_id, _] = store.put_artifacts([want_artifact, other_artifact])
    artifact_result = store.get_artifacts_by_uri(want_artifact.uri)
    self.assertLen(artifact_result, 1)
    self.assertEqual(artifact_result[0].id, want_artifact_id)

  def test_puts_artifacts_duplicated_name_with_the_same_type(self):
    store = _get_metadata_store()
    with self.assertRaises(errors.AlreadyExistsError):
      artifact_type = _create_example_artifact_type(self._get_test_type_name())
      type_id = store.put_artifact_type(artifact_type)
      artifact_0 = metadata_store_pb2.Artifact()
      artifact_0.type_id = type_id
      artifact_0.name = "the_same_name"
      artifact_1 = metadata_store_pb2.Artifact()
      artifact_1.type_id = type_id
      artifact_1.name = "the_same_name"
      store.put_artifacts([artifact_0, artifact_1])

  def test_put_executions_get_executions_by_type(self):
    store = _get_metadata_store()
    execution_type = _create_example_execution_type(self._get_test_type_name())
    type_id = store.put_execution_type(execution_type)
    execution_type_2 = _create_example_execution_type_2(
        self._get_test_type_name())
    type_id_2 = store.put_execution_type(execution_type_2)
    execution_0 = metadata_store_pb2.Execution()
    execution_0.type_id = type_id
    execution_0.properties["foo"].int_value = 3
    execution_0.properties["bar"].string_value = "Hello"
    execution_1 = metadata_store_pb2.Execution()
    execution_1.type_id = type_id_2

    [_, execution_id_1] = store.put_executions([execution_0, execution_1])
    execution_result = store.get_executions_by_type(execution_type_2.name)
    self.assertLen(execution_result, 1)
    self.assertEqual(execution_result[0].id, execution_id_1)

  def test_put_executions_get_execution_by_type_and_name(self):
    # Prepare test data.
    store = _get_metadata_store()
    execution_type = _create_example_execution_type(self._get_test_type_name())
    type_id = store.put_execution_type(execution_type)
    execution = metadata_store_pb2.Execution()
    execution.type_id = type_id
    execution.name = self._get_test_type_name()
    [execution_id] = store.put_executions([execution])

    # Test Execution found case.
    got_execution = store.get_execution_by_type_and_name(
        execution_type.name, execution.name)
    self.assertEqual(got_execution.id, execution_id)
    self.assertEqual(got_execution.type_id, type_id)
    self.assertEqual(got_execution.name, execution.name)

    # Test Execution not found cases.
    empty_execution = store.get_execution_by_type_and_name(
        "random_name", execution.name)
    self.assertIsNone(empty_execution)
    empty_execution = store.get_execution_by_type_and_name(
        execution_type.name, "random_name")
    self.assertIsNone(empty_execution)
    empty_execution = store.get_execution_by_type_and_name(
        "random_name", "random_name")
    self.assertIsNone(empty_execution)

  def test_update_artifact_get_artifact(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type(self._get_test_type_name())
    type_id = store.put_artifact_type(artifact_type)
    artifact = metadata_store_pb2.Artifact()
    artifact.type_id = type_id
    artifact.properties["bar"].string_value = "Hello"

    [artifact_id] = store.put_artifacts([artifact])
    artifact_2 = metadata_store_pb2.Artifact()
    artifact_2.CopyFrom(artifact)
    artifact_2.id = artifact_id
    artifact_2.properties["foo"].int_value = artifact_id
    artifact_2.properties["bar"].string_value = "Goodbye"
    [artifact_id_2] = store.put_artifacts([artifact_2])
    self.assertEqual(artifact_id, artifact_id_2)

    [artifact_result] = store.get_artifacts_by_id([artifact_id])
    self.assertEqual(artifact_result.properties["bar"].string_value, "Goodbye")
    self.assertEqual(artifact_result.properties["foo"].int_value, artifact_id)

  def test_create_artifact_with_type_get_artifacts_by_id(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type(self._get_test_type_name())
    artifact = metadata_store_pb2.Artifact()
    artifact.properties["foo"].int_value = 3
    artifact.properties["bar"].string_value = "Hello"
    artifact_id = store.create_artifact_with_type(artifact, artifact_type)
    [artifact_result] = store.get_artifacts_by_id([artifact_id])
    self.assertEqual(artifact_result.properties["bar"].string_value, "Hello")
    self.assertEqual(artifact_result.properties["foo"].int_value, 3)

  def test_put_execution_type_get_execution_type(self):
    store = _get_metadata_store()
    execution_type_name = self._get_test_type_name()
    execution_type = metadata_store_pb2.ExecutionType()
    execution_type.name = execution_type_name
    execution_type.properties["foo"] = metadata_store_pb2.INT
    execution_type.properties["bar"] = metadata_store_pb2.STRING
    type_id = store.put_execution_type(execution_type)
    execution_type_result = store.get_execution_type(execution_type_name)
    self.assertEqual(execution_type_result.id, type_id)
    self.assertEqual(execution_type_result.name, execution_type_name)

  def test_put_execution_type_with_update_get_execution_type(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType()
    execution_type_name = self._get_test_type_name()
    execution_type.name = execution_type_name
    execution_type.properties["foo"] = metadata_store_pb2.DOUBLE
    type_id = store.put_execution_type(execution_type)

    want_execution_type = metadata_store_pb2.ExecutionType()
    want_execution_type.id = type_id
    want_execution_type.name = execution_type_name
    want_execution_type.properties["foo"] = metadata_store_pb2.DOUBLE
    want_execution_type.properties["new_property"] = metadata_store_pb2.INT
    store.put_execution_type(want_execution_type, can_add_fields=True)

    got_execution_type = store.get_execution_type(execution_type_name)
    self.assertEqual(got_execution_type.id, type_id)
    self.assertEqual(got_execution_type.name, execution_type_name)
    self.assertEqual(got_execution_type.properties["foo"],
                     metadata_store_pb2.DOUBLE)
    self.assertEqual(got_execution_type.properties["new_property"],
                     metadata_store_pb2.INT)

  def test_put_executions_get_executions_by_id(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType()
    execution_type.name = self._get_test_type_name()
    execution_type.properties["foo"] = metadata_store_pb2.INT
    execution_type.properties["bar"] = metadata_store_pb2.STRING
    type_id = store.put_execution_type(execution_type)
    execution = metadata_store_pb2.Execution()
    execution.type_id = type_id
    execution.properties["foo"].int_value = 3
    execution.properties["bar"].string_value = "Hello"
    [execution_id] = store.put_executions([execution])
    [execution_result] = store.get_executions_by_id([execution_id])
    self.assertEqual(execution_result.properties["bar"].string_value, "Hello")
    self.assertEqual(execution_result.properties["foo"].int_value, 3)

  def test_put_executions_get_executions(self):
    store = _get_metadata_store()
    execution_type = _create_example_execution_type(self._get_test_type_name())
    type_id = store.put_execution_type(execution_type)
    execution_0 = metadata_store_pb2.Execution()
    execution_0.type_id = type_id
    execution_0.properties["foo"].int_value = 3
    execution_0.properties["bar"].string_value = "Hello"
    execution_1 = metadata_store_pb2.Execution()
    execution_1.type_id = type_id
    execution_1.properties["foo"].int_value = -9
    execution_1.properties["bar"].string_value = "Goodbye"

    existing_executions_count = 0
    try:
      existing_executions_count = len(store.get_executions())
    except errors.NotFoundError:
      existing_executions_count = 0

    [execution_id_0,
     execution_id_1] = store.put_executions([execution_0, execution_1])
    execution_result = store.get_executions()
    new_executions_count = len(execution_result)
    execution_result = [
        e for e in execution_result
        if e.id == execution_id_0 or e.id == execution_id_1
    ]

    self.assertLen(execution_result, 2)
    self.assertEqual(existing_executions_count + 2, new_executions_count)
    # Normalize the order of the results.
    if execution_result[0].id == execution_id_0:
      [execution_result_0, execution_result_1] = execution_result
    else:
      [execution_result_1, execution_result_0] = execution_result

    self.assertEqual(execution_result_0.id, execution_id_0)
    self.assertEqual(execution_result_0.properties["bar"].string_value, "Hello")
    self.assertEqual(execution_result_0.properties["foo"].int_value, 3)
    self.assertEqual(execution_result_1.id, execution_id_1)
    self.assertEqual(execution_result_1.properties["bar"].string_value,
                     "Goodbye")
    self.assertEqual(execution_result_1.properties["foo"].int_value, -9)

  def test_puts_executions_duplicated_name_with_the_same_type(self):
    store = _get_metadata_store()
    with self.assertRaises(errors.AlreadyExistsError):
      execution_type = _create_example_execution_type(
          self._get_test_type_name())
      type_id = store.put_execution_type(execution_type)
      execution_0 = metadata_store_pb2.Execution()
      execution_0.type_id = type_id
      execution_0.name = "the_same_name"
      execution_1 = metadata_store_pb2.Execution()
      execution_1.type_id = type_id
      execution_1.name = "the_same_name"
      store.put_executions([execution_0, execution_1])

  def test_update_execution_get_execution(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType()
    execution_type.name = self._get_test_type_name()
    execution_type.properties["foo"] = metadata_store_pb2.INT
    execution_type.properties["bar"] = metadata_store_pb2.STRING
    type_id = store.put_execution_type(execution_type)
    execution = metadata_store_pb2.Execution()
    execution.type_id = type_id
    execution.properties["bar"].string_value = "Hello"

    [execution_id] = store.put_executions([execution])
    execution_2 = metadata_store_pb2.Execution()
    execution_2.id = execution_id
    execution_2.type_id = type_id
    execution_2.properties["foo"].int_value = 12
    execution_2.properties["bar"].string_value = "Goodbye"
    [execution_id_2] = store.put_executions([execution_2])
    self.assertEqual(execution_id, execution_id_2)

    [execution_result] = store.get_executions_by_id([execution_id])
    self.assertEqual(execution_result.properties["bar"].string_value, "Goodbye")
    self.assertEqual(execution_result.properties["foo"].int_value, 12)

  def test_put_events_get_events(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType()
    execution_type.name = self._get_test_type_name()
    execution_type_id = store.put_execution_type(execution_type)
    execution = metadata_store_pb2.Execution()
    execution.type_id = execution_type_id
    [execution_id] = store.put_executions([execution])
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = self._get_test_type_name()
    artifact_type_id = store.put_artifact_type(artifact_type)
    artifact = metadata_store_pb2.Artifact()
    artifact.type_id = artifact_type_id
    [artifact_id] = store.put_artifacts([artifact])

    event = metadata_store_pb2.Event()
    event.type = metadata_store_pb2.Event.DECLARED_OUTPUT
    event.artifact_id = artifact_id
    event.execution_id = execution_id
    store.put_events([event])
    [event_result] = store.get_events_by_artifact_ids([artifact_id])
    self.assertEqual(event_result.artifact_id, artifact_id)
    self.assertEqual(event_result.execution_id, execution_id)
    self.assertEqual(event_result.type,
                     metadata_store_pb2.Event.DECLARED_OUTPUT)

    [event_result_2] = store.get_events_by_execution_ids([execution_id])
    self.assertEqual(event_result_2.artifact_id, artifact_id)
    self.assertEqual(event_result_2.execution_id, execution_id)
    self.assertEqual(event_result_2.type,
                     metadata_store_pb2.Event.DECLARED_OUTPUT)

  def test_get_executions_by_id_empty(self):
    store = _get_metadata_store()
    result = store.get_executions_by_id({})
    self.assertEmpty(result)

  def test_get_artifact_type_fails(self):
    store = _get_metadata_store()
    with self.assertRaises(errors.NotFoundError):
      store.get_artifact_type("not_found_type")

  def test_put_events_no_artifact_id(self):
    # No execution_id throws the same error type, so we just test this.
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType()
    execution_type.name = self._get_test_type_name()
    execution_type_id = store.put_execution_type(execution_type)
    execution = metadata_store_pb2.Execution()
    execution.type_id = execution_type_id
    [execution_id] = store.put_executions([execution])

    event = metadata_store_pb2.Event()
    event.type = metadata_store_pb2.Event.DECLARED_OUTPUT
    event.execution_id = execution_id
    with self.assertRaises(errors.InvalidArgumentError):
      store.put_events([event])

  def test_put_events_with_paths(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType()
    execution_type.name = self._get_test_type_name()
    execution_type_id = store.put_execution_type(execution_type)
    execution = metadata_store_pb2.Execution()
    execution.type_id = execution_type_id
    [execution_id] = store.put_executions([execution])
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = self._get_test_type_name()
    artifact_type_id = store.put_artifact_type(artifact_type)
    artifact_0 = metadata_store_pb2.Artifact()
    artifact_0.type_id = artifact_type_id
    artifact_1 = metadata_store_pb2.Artifact()
    artifact_1.type_id = artifact_type_id
    [artifact_id_0,
     artifact_id_1] = store.put_artifacts([artifact_0, artifact_1])

    event_0 = metadata_store_pb2.Event()
    event_0.type = metadata_store_pb2.Event.DECLARED_INPUT
    event_0.artifact_id = artifact_id_0
    event_0.execution_id = execution_id
    event_0.path.steps.add().key = "ggg"

    event_1 = metadata_store_pb2.Event()
    event_1.type = metadata_store_pb2.Event.DECLARED_INPUT
    event_1.artifact_id = artifact_id_1
    event_1.execution_id = execution_id
    event_1.path.steps.add().key = "fff"

    store.put_events([event_0, event_1])
    [event_result_0,
     event_result_1] = store.get_events_by_execution_ids([execution_id])
    self.assertLen(event_result_0.path.steps, 1)
    self.assertEqual(event_result_0.path.steps[0].key, "ggg")
    self.assertLen(event_result_1.path.steps, 1)
    self.assertEqual(event_result_1.path.steps[0].key, "fff")

  def test_put_events_with_paths_same_artifact(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType()
    execution_type.name = self._get_test_type_name()
    execution_type_id = store.put_execution_type(execution_type)
    execution_0 = metadata_store_pb2.Execution()
    execution_0.type_id = execution_type_id
    execution_1 = metadata_store_pb2.Execution()
    execution_1.type_id = execution_type_id
    [execution_id_0,
     execution_id_1] = store.put_executions([execution_0, execution_1])
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = self._get_test_type_name()
    artifact_type_id = store.put_artifact_type(artifact_type)
    artifact = metadata_store_pb2.Artifact()
    artifact.type_id = artifact_type_id
    [artifact_id] = store.put_artifacts([artifact])

    event_0 = metadata_store_pb2.Event()
    event_0.type = metadata_store_pb2.Event.DECLARED_INPUT
    event_0.artifact_id = artifact_id
    event_0.execution_id = execution_id_0
    event_0.path.steps.add().key = "ggg"

    event_1 = metadata_store_pb2.Event()
    event_1.type = metadata_store_pb2.Event.DECLARED_INPUT
    event_1.artifact_id = artifact_id
    event_1.execution_id = execution_id_1
    event_1.path.steps.add().key = "fff"

    store.put_events([event_0, event_1])
    [event_result_0,
     event_result_1] = store.get_events_by_artifact_ids([artifact_id])
    self.assertLen(event_result_0.path.steps, 1)
    self.assertEqual(event_result_0.path.steps[0].key, "ggg")
    self.assertLen(event_result_1.path.steps, 1)
    self.assertEqual(event_result_1.path.steps[0].key, "fff")

  def test_put_execution_without_context(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType(
        name=self._get_test_type_name())
    execution_type_id = store.put_execution_type(execution_type)
    execution = metadata_store_pb2.Execution(type_id=execution_type_id)

    artifact_type = metadata_store_pb2.ArtifactType(
        name=self._get_test_type_name())
    artifact_type_id = store.put_artifact_type(artifact_type)
    input_artifact = metadata_store_pb2.Artifact(type_id=artifact_type_id)
    output_artifact = metadata_store_pb2.Artifact(type_id=artifact_type_id)
    output_event = metadata_store_pb2.Event(
        type=metadata_store_pb2.Event.DECLARED_INPUT)

    execution_id, artifact_ids, context_ids = store.put_execution(
        execution, [[input_artifact], [output_artifact, output_event]], [])
    self.assertLen(artifact_ids, 2)
    events = store.get_events_by_execution_ids([execution_id])
    self.assertLen(events, 1)
    self.assertEmpty(context_ids)

  def test_put_execution_with_context(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType(
        name=self._get_test_type_name())
    execution_type_id = store.put_execution_type(execution_type)
    execution = metadata_store_pb2.Execution(type_id=execution_type_id)

    artifact_type = metadata_store_pb2.ArtifactType(
        name=self._get_test_type_name())
    artifact_type_id = store.put_artifact_type(artifact_type)
    input_artifact = metadata_store_pb2.Artifact(type_id=artifact_type_id)
    output_artifact = metadata_store_pb2.Artifact(type_id=artifact_type_id)
    output_event = metadata_store_pb2.Event(
        type=metadata_store_pb2.Event.DECLARED_INPUT)

    context_type = metadata_store_pb2.ContextType(
        name=self._get_test_type_name())
    context_type_id = store.put_context_type(context_type)
    context = metadata_store_pb2.Context(
        type_id=context_type_id, name=self._get_test_type_name())

    execution_id, artifact_ids, context_ids = store.put_execution(
        execution, [[input_artifact], [output_artifact, output_event]],
        [context])

    # Test artifacts & events are correctly inserted.
    self.assertLen(artifact_ids, 2)
    events = store.get_events_by_execution_ids([execution_id])
    self.assertLen(events, 1)

    # Test the context is correctly inserted.
    got_contexts = store.get_contexts_by_id(context_ids)
    self.assertLen(context_ids, 1)
    self.assertLen(got_contexts, 1)

    # Test the association link between execution and the context is correct.
    contexts_by_execution_id = store.get_contexts_by_execution(execution_id)
    self.assertLen(contexts_by_execution_id, 1)
    self.assertEqual(contexts_by_execution_id[0].name, context.name)
    self.assertEqual(contexts_by_execution_id[0].type_id, context_type_id)
    executions_by_context = store.get_executions_by_context(context_ids[0])
    self.assertLen(executions_by_context, 1)

    # Test the attribution links between artifacts and the context are correct.
    contexts_by_artifact_id = store.get_contexts_by_artifact(artifact_ids[0])
    self.assertLen(contexts_by_artifact_id, 1)
    self.assertEqual(contexts_by_artifact_id[0].name, context.name)
    self.assertEqual(contexts_by_artifact_id[0].type_id, context_type_id)
    artifacts_by_context = store.get_artifacts_by_context(context_ids[0])
    self.assertLen(artifacts_by_context, 2)

  def test_get_executions_by_context_with_pagination(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType(
        name=self._get_test_type_name())
    execution_type_id = store.put_execution_type(execution_type)

    context_type = metadata_store_pb2.ContextType(
        name=self._get_test_type_name())
    context_type_id = store.put_context_type(context_type)
    context = metadata_store_pb2.Context(
        type_id=context_type_id, name=self._get_test_type_name())
    context_ids = store.put_contexts([context])
    context_id = context_ids[0]

    executions = []
    count = 0
    while count < 102:
      execution = metadata_store_pb2.Execution(type_id=execution_type_id)
      executions.append(execution)
      count += 1

    execution_ids = store.put_executions(executions)

    associations = []
    count = 0
    for execution_id in execution_ids:
      association = metadata_store_pb2.Association(
          context_id=context_id, execution_id=execution_id)
      associations.append(association)

    store.put_attributions_and_associations([], associations)

    got_executions = store.get_executions_by_context(context_id)
    reverse_index = len(execution_ids) - 1
    for got_execution in got_executions:
      self.assertEqual(got_execution.id, execution_ids[reverse_index])
      reverse_index -= 1

    self.assertEqual(reverse_index, -1)

  def test_get_artifacts_by_context_with_pagination(self):
    store = _get_metadata_store()
    artifact_type = metadata_store_pb2.ArtifactType(
        name=self._get_test_type_name())
    artifact_type_id = store.put_artifact_type(artifact_type)

    context_type = metadata_store_pb2.ContextType(
        name=self._get_test_type_name())
    context_type_id = store.put_context_type(context_type)
    context = metadata_store_pb2.Context(
        type_id=context_type_id, name=self._get_test_type_name())
    context_ids = store.put_contexts([context])
    context_id = context_ids[0]

    artifacts = []
    count = 0
    while count < 102:
      artifact = metadata_store_pb2.Artifact(type_id=artifact_type_id)
      artifacts.append(artifact)
      count += 1

    artifact_ids = store.put_artifacts(artifacts)

    attributions = []
    count = 0
    for artifact_id in artifact_ids:
      attribution = metadata_store_pb2.Attribution(
          context_id=context_id, artifact_id=artifact_id)
      attributions.append(attribution)

    store.put_attributions_and_associations(attributions, [])

    got_artifacts = store.get_artifacts_by_context(context_id)
    reverse_index = len(artifact_ids) - 1
    for got_artifact in got_artifacts:
      self.assertEqual(got_artifact.id, artifact_ids[reverse_index])
      reverse_index -= 1

    self.assertEqual(reverse_index, -1)

  def test_put_execution_with_reuse_context_if_already_exist(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType(
        name=self._get_test_type_name())
    execution_type_id = store.put_execution_type(execution_type)
    execution = metadata_store_pb2.Execution(type_id=execution_type_id)

    context_type = metadata_store_pb2.ContextType(
        name=self._get_test_type_name())
    context_type_id = store.put_context_type(context_type)
    context = metadata_store_pb2.Context(
        type_id=context_type_id, name=self._get_test_type_name())

    # mimic a race with calls to create new context with the same name
    execution_id, _, context_ids = store.put_execution(
        execution=execution, artifact_and_events=[], contexts=[context])

    # the second call fails due to the context of the same name AlreadyExists
    with self.assertRaises(errors.AlreadyExistsError):
      store.put_execution(
          execution=execution, artifact_and_events=[], contexts=[context])

    # if set reuse_context_if_already_exist, the same call succeeds
    # context ids are the same, and the execution ids are different.
    execution_id2, _, context_ids2 = store.put_execution(
        execution=execution,
        artifact_and_events=[],
        contexts=[context],
        reuse_context_if_already_exist=True)
    self.assertEqual(context_ids, context_ids2)
    self.assertNotEqual(execution_id, execution_id2)
    self.assertLen(store.get_executions_by_context(context_ids[0]), 2)

  def test_put_execution_with_invalid_argument_errors(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType(
        name=self._get_test_type_name())
    execution_type_id = store.put_execution_type(execution_type)
    not_exist_id = 1000
    execution = metadata_store_pb2.Execution(
        id=not_exist_id, type_id=execution_type_id)

    with self.assertRaises(errors.InvalidArgumentError):
      store.put_execution(
          execution=execution, artifact_and_events=[], contexts=[])

  def test_put_context_type_get_context_type(self):
    store = _get_metadata_store()
    context_type_name = self._get_test_type_name()
    context_type = _create_example_context_type(context_type_name)

    type_id = store.put_context_type(context_type)
    context_type_result = store.get_context_type(context_type_name)
    self.assertEqual(context_type_result.id, type_id)
    self.assertEqual(context_type_result.name, context_type_name)

    context_types_by_id_results = store.get_context_types_by_id([type_id])
    self.assertLen(context_types_by_id_results, 1)
    self.assertEqual(context_types_by_id_results[0].id, type_id)
    self.assertEqual(context_types_by_id_results[0].name, context_type_name)

  def test_put_context_type_with_update_get_context_type(self):
    store = _get_metadata_store()
    context_type = metadata_store_pb2.ContextType()
    context_type_name = self._get_test_type_name()
    context_type.name = context_type_name
    context_type.properties["foo"] = metadata_store_pb2.INT
    type_id = store.put_context_type(context_type)

    want_context_type = metadata_store_pb2.ContextType()
    want_context_type.name = context_type_name
    want_context_type.properties["foo"] = metadata_store_pb2.INT
    want_context_type.properties["new_property"] = metadata_store_pb2.STRING
    store.put_context_type(want_context_type, can_add_fields=True)

    got_context_type = store.get_context_type(context_type_name)
    self.assertEqual(got_context_type.id, type_id)
    self.assertEqual(got_context_type.name, context_type_name)
    self.assertEqual(got_context_type.properties["foo"], metadata_store_pb2.INT)
    self.assertEqual(got_context_type.properties["new_property"],
                     metadata_store_pb2.STRING)

  def test_put_contexts_get_contexts_by_id(self):
    store = _get_metadata_store()
    context_type = _create_example_context_type(self._get_test_type_name())
    type_id = store.put_context_type(context_type)
    context = metadata_store_pb2.Context()
    context.type_id = type_id
    context.name = self._get_test_type_name()
    context.properties["foo"].int_value = 3
    context.custom_properties["abc"].string_value = "s"
    [context_id] = store.put_contexts([context])
    [context_result] = store.get_contexts_by_id([context_id])
    self.assertEqual(context_result.name, context.name)
    self.assertEqual(context_result.properties["foo"].int_value,
                     context.properties["foo"].int_value)
    self.assertEqual(context_result.custom_properties["abc"].string_value,
                     context.custom_properties["abc"].string_value)

  def test_put_contexts_get_contexts(self):
    store = _get_metadata_store()
    context_type = _create_example_context_type(self._get_test_type_name())
    type_id = store.put_context_type(context_type)
    context_0 = metadata_store_pb2.Context()
    context_0.type_id = type_id
    context_0_name = self._get_test_type_name()
    context_0.name = context_0_name
    context_0.properties["bar"].string_value = "Hello"
    context_1 = metadata_store_pb2.Context()
    context_1_name = self._get_test_type_name()
    context_1.name = context_1_name
    context_1.type_id = type_id
    context_1.properties["foo"].int_value = -9

    existing_contexts_count = 0
    try:
      existing_contexts_count = len(store.get_contexts())
    except errors.NotFoundError:
      existing_contexts_count = 0
    [context_id_0, context_id_1] = store.put_contexts([context_0, context_1])

    context_result = store.get_contexts()
    new_contexts_count = len(context_result)
    context_result = [
        c for c in context_result
        if c.id == context_id_0 or c.id == context_id_1
    ]

    self.assertEqual(existing_contexts_count + 2, new_contexts_count)
    # Normalize the order of the results.
    if context_result[0].id == context_id_0:
      [context_result_0, context_result_1] = context_result
    else:
      [context_result_1, context_result_0] = context_result

    self.assertEqual(context_result_0.name, context_0_name)
    self.assertEqual(context_result_0.properties["bar"].string_value, "Hello")
    self.assertEqual(context_result_1.name, context_1_name)
    self.assertEqual(context_result_1.properties["foo"].int_value, -9)

  def test_put_contexts_get_context_by_type_and_name(self):
    # Prepare test data.
    store = _get_metadata_store()
    context_type = _create_example_context_type(self._get_test_type_name())
    type_id = store.put_context_type(context_type)
    context = metadata_store_pb2.Context()
    context.type_id = type_id
    context.name = self._get_test_type_name()
    [context_id] = store.put_contexts([context])

    # Test Context found case.
    got_context = store.get_context_by_type_and_name(
        context_type.name, context.name)
    self.assertEqual(got_context.id, context_id)
    self.assertEqual(got_context.type_id, type_id)
    self.assertEqual(got_context.name, context.name)

    # Test Context not found cases.
    empty_context = store.get_context_by_type_and_name("random_name",
                                                       context.name)
    self.assertIsNone(empty_context)
    empty_context = store.get_context_by_type_and_name(context_type.name,
                                                       "random_name")
    self.assertIsNone(empty_context)
    empty_context = store.get_context_by_type_and_name("random_name",
                                                       "random_name")
    self.assertIsNone(empty_context)

  def test_put_contexts_get_contexts_by_type(self):
    store = _get_metadata_store()
    context_type = _create_example_context_type(self._get_test_type_name())
    type_id = store.put_context_type(context_type)
    context_type_2 = _create_example_context_type_2(self._get_test_type_name())
    type_id_2 = store.put_context_type(context_type_2)
    context_0 = metadata_store_pb2.Context()
    context_0.type_id = type_id
    context_0.name = self._get_test_type_name()
    context_1 = metadata_store_pb2.Context()
    context_1.type_id = type_id_2
    context_1.name = self._get_test_type_name()

    [_, context_id_1] = store.put_contexts([context_0, context_1])
    context_result = store.get_contexts_by_type(context_type_2.name)
    self.assertLen(context_result, 1)
    self.assertEqual(context_result[0].id, context_id_1)

  def test_puts_contexts_empty_name(self):
    store = _get_metadata_store()
    with self.assertRaises(errors.InvalidArgumentError):
      context_type = _create_example_context_type(self._get_test_type_name())
      type_id = store.put_context_type(context_type)
      context_0 = metadata_store_pb2.Context()
      context_0.type_id = type_id
      store.put_contexts([context_0])

  def test_puts_contexts_duplicated_name_with_the_same_type(self):
    store = _get_metadata_store()
    with self.assertRaises(errors.AlreadyExistsError):
      context_type = _create_example_context_type(self._get_test_type_name())
      type_id = store.put_context_type(context_type)
      context_0 = metadata_store_pb2.Context()
      context_0.type_id = type_id
      context_0.name = "the_same_name"
      context_1 = metadata_store_pb2.Context()
      context_1.type_id = type_id
      context_1.name = "the_same_name"
      store.put_contexts([context_0, context_1])

  def test_update_context_get_context(self):
    store = _get_metadata_store()
    context_type = _create_example_context_type(self._get_test_type_name())
    type_id = store.put_context_type(context_type)
    context = metadata_store_pb2.Context()
    context.type_id = type_id
    context.name = self._get_test_type_name()
    context.properties["bar"].string_value = "Hello"
    [context_id] = store.put_contexts([context])

    context_2 = metadata_store_pb2.Context()
    context_2.id = context_id
    context_2.name = self._get_test_type_name()
    context_2.type_id = type_id
    context_2.properties["foo"].int_value = 12
    context_2.properties["bar"].string_value = "Goodbye"
    [context_id_2] = store.put_contexts([context_2])
    self.assertEqual(context_id, context_id_2)

    [context_result] = store.get_contexts_by_id([context_id])
    self.assertEqual(context_result.name, context_2.name)
    self.assertEqual(context_result.properties["bar"].string_value, "Goodbye")
    self.assertEqual(context_result.properties["foo"].int_value, 12)

  def test_put_and_use_attributions_and_associations(self):
    store = _get_metadata_store()
    context_type = _create_example_context_type(self._get_test_type_name())
    context_type_id = store.put_context_type(context_type)
    want_context = metadata_store_pb2.Context()
    want_context.type_id = context_type_id
    want_context.name = self._get_test_type_name()
    [context_id] = store.put_contexts([want_context])
    want_context.id = context_id

    execution_type = _create_example_execution_type(self._get_test_type_name())
    execution_type_id = store.put_execution_type(execution_type)
    want_execution = metadata_store_pb2.Execution()
    want_execution.type_id = execution_type_id
    want_execution.properties["foo"].int_value = 3
    [execution_id] = store.put_executions([want_execution])
    want_execution.id = execution_id

    artifact_type = _create_example_artifact_type(self._get_test_type_name())
    artifact_type_id = store.put_artifact_type(artifact_type)
    want_artifact = metadata_store_pb2.Artifact()
    want_artifact.type_id = artifact_type_id
    want_artifact.uri = "testuri"
    [artifact_id] = store.put_artifacts([want_artifact])
    want_artifact.id = artifact_id

    # insert attribution and association and test querying the relationship
    attribution = metadata_store_pb2.Attribution()
    attribution.artifact_id = want_artifact.id
    attribution.context_id = want_context.id
    association = metadata_store_pb2.Association()
    association.execution_id = want_execution.id
    association.context_id = want_context.id
    store.put_attributions_and_associations([attribution], [association])

    # test querying the relationship
    got_contexts = store.get_contexts_by_artifact(want_artifact.id)
    self.assertLen(got_contexts, 1)
    self.assertEqual(got_contexts[0].id, want_context.id)
    self.assertEqual(got_contexts[0].name, want_context.name)
    got_arifacts = store.get_artifacts_by_context(want_context.id)
    self.assertLen(got_arifacts, 1)
    self.assertEqual(got_arifacts[0].uri, want_artifact.uri)
    got_executions = store.get_executions_by_context(want_context.id)
    self.assertLen(got_executions, 1)
    self.assertEqual(got_executions[0].properties["foo"],
                     want_execution.properties["foo"])
    got_contexts = store.get_contexts_by_execution(want_execution.id)
    self.assertLen(got_contexts, 1)
    self.assertEqual(got_contexts[0].id, want_context.id)
    self.assertEqual(got_contexts[0].name, want_context.name)

  def test_put_duplicated_attributions_and_empty_associations(self):
    store = _get_metadata_store()
    context_type = _create_example_context_type(self._get_test_type_name())
    context_type_id = store.put_context_type(context_type)
    want_context = metadata_store_pb2.Context()
    want_context.type_id = context_type_id
    want_context.name = self._get_test_type_name()
    [context_id] = store.put_contexts([want_context])
    want_context.id = context_id

    artifact_type = _create_example_artifact_type(self._get_test_type_name())
    artifact_type_id = store.put_artifact_type(artifact_type)
    want_artifact = metadata_store_pb2.Artifact()
    want_artifact.type_id = artifact_type_id
    want_artifact.uri = "testuri"
    [artifact_id] = store.put_artifacts([want_artifact])
    want_artifact.id = artifact_id

    attribution = metadata_store_pb2.Attribution()
    attribution.artifact_id = want_artifact.id
    attribution.context_id = want_context.id
    store.put_attributions_and_associations([attribution, attribution], [])

    got_contexts = store.get_contexts_by_artifact(want_artifact.id)
    self.assertLen(got_contexts, 1)
    self.assertEqual(got_contexts[0].id, want_context.id)
    self.assertEqual(got_contexts[0].name, want_context.name)
    got_arifacts = store.get_artifacts_by_context(want_context.id)
    self.assertLen(got_arifacts, 1)
    self.assertEqual(got_arifacts[0].uri, want_artifact.uri)
    self.assertEmpty(store.get_executions_by_context(want_context.id))

  def test_downgrade_metadata_store(self):
    # create a metadata store and init to the current library version
    db_file = os.path.join(absltest.get_default_test_tmpdir(),
                           self._get_test_db_name())
    if os.path.exists(db_file):
      os.remove(db_file)
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.filename_uri = db_file
    metadata_store.MetadataStore(connection_config)

    # wrong downgrade_to_schema_version
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "downgrade_to_schema_version not specified"):
      metadata_store.downgrade_schema(connection_config, -1)

    # invalid argument for the downgrade_to_schema_version
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "MLMD cannot be downgraded to schema_version"):
      downgrade_to_version = 999999
      metadata_store.downgrade_schema(connection_config, downgrade_to_version)

    # downgrade the metadata store to v0.13.2 where schema version is 0
    metadata_store.downgrade_schema(
        connection_config, downgrade_to_schema_version=0)
    os.remove(db_file)

  def test_enable_metadata_store_upgrade_migration(self):
    # create a metadata store and downgrade to version 0
    db_file = os.path.join(absltest.get_default_test_tmpdir(),
                           self._get_test_db_name())
    if os.path.exists(db_file):
      os.remove(db_file)
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.filename_uri = db_file
    metadata_store.MetadataStore(connection_config)
    metadata_store.downgrade_schema(connection_config, 0)

    upgrade_conn_config = metadata_store_pb2.ConnectionConfig()
    upgrade_conn_config.sqlite.filename_uri = db_file
    with self.assertRaisesRegex(RuntimeError, "chema migration is disabled"):
      # if disabled then the store cannot be used.
      metadata_store.MetadataStore(upgrade_conn_config)

    # if enable, then the store can be created
    metadata_store.MetadataStore(
        upgrade_conn_config, enable_upgrade_migration=True)
    os.remove(db_file)

  def test_put_invalid_artifact(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type(self._get_test_type_name())
    artifact_type_id = store.put_artifact_type(artifact_type)
    artifact = metadata_store_pb2.Artifact()
    artifact.type_id = artifact_type_id
    artifact.uri = "testuri"
    # Create the Value message for "foo" but don't populate its value.
    artifact.properties["foo"]  # pylint: disable=pointless-statement
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Found unmatched property type: foo"):
      store.put_artifacts([artifact])

if __name__ == "__main__":
  absltest.main()
