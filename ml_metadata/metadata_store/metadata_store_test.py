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

from absl.testing import absltest
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2
from tensorflow.python.framework import errors


def _get_metadata_store():
  connection_config = metadata_store_pb2.ConnectionConfig()
  connection_config.sqlite.SetInParent()
  return metadata_store.MetadataStore(connection_config)


def _create_example_artifact_type():
  artifact_type = metadata_store_pb2.ArtifactType()
  artifact_type.name = "test_type_2"
  artifact_type.properties["foo"] = metadata_store_pb2.INT
  artifact_type.properties["bar"] = metadata_store_pb2.STRING
  artifact_type.properties["baz"] = metadata_store_pb2.DOUBLE
  return artifact_type


def _create_example_execution_type():
  execution_type = metadata_store_pb2.ExecutionType()
  execution_type.name = "test_type_2"
  execution_type.properties["foo"] = metadata_store_pb2.INT
  execution_type.properties["bar"] = metadata_store_pb2.STRING
  return execution_type




class MetadataStoreTest(absltest.TestCase):

  def test_put_artifact_type_get_artifact_type(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type()

    type_id = store.put_artifact_type(artifact_type)
    artifact_type_result = store.get_artifact_type("test_type_2")
    self.assertEqual(artifact_type_result.id, type_id)
    self.assertEqual(artifact_type_result.name, "test_type_2")
    self.assertEqual(artifact_type_result.properties["foo"],
                     metadata_store_pb2.INT)
    self.assertEqual(artifact_type_result.properties["bar"],
                     metadata_store_pb2.STRING)
    self.assertEqual(artifact_type.properties["baz"], metadata_store_pb2.DOUBLE)

  def test_put_artifacts_get_artifacts_by_id(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type()
    type_id = store.put_artifact_type(artifact_type)
    artifact = metadata_store_pb2.Artifact()
    artifact.type_id = type_id
    artifact.properties["foo"].int_value = 3
    artifact.properties["bar"].string_value = "Hello"
    [artifact_id] = store.put_artifacts([artifact])
    [artifact_result] = store.get_artifacts_by_id([artifact_id])
    self.assertEqual(artifact_result.properties["bar"].string_value, "Hello")
    self.assertEqual(artifact_result.properties["foo"].int_value, 3)

  def test_put_artifacts_get_artifacts(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type()
    type_id = store.put_artifact_type(artifact_type)
    artifact_0 = metadata_store_pb2.Artifact()
    artifact_0.type_id = type_id
    artifact_0.properties["foo"].int_value = 3
    artifact_0.properties["bar"].string_value = "Hello"
    artifact_1 = metadata_store_pb2.Artifact()
    artifact_1.type_id = type_id

    [artifact_id_0,
     artifact_id_1] = store.put_artifacts([artifact_0, artifact_1])
    artifact_result = store.get_artifacts()
    if artifact_result[0].id == artifact_id_0:
      [artifact_result_0, artifact_result_1] = artifact_result
    else:
      [artifact_result_1, artifact_result_0] = artifact_result
    self.assertEqual(artifact_result_0.id, artifact_id_0)
    self.assertEqual(artifact_result_0.properties["bar"].string_value, "Hello")
    self.assertEqual(artifact_result_0.properties["foo"].int_value, 3)
    self.assertEqual(artifact_result_1.id, artifact_id_1)

  def test_update_artifact_get_artifact(self):
    store = _get_metadata_store()
    artifact_type = _create_example_artifact_type()
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
    artifact_type = _create_example_artifact_type()
    artifact = metadata_store_pb2.Artifact()
    artifact.properties["foo"].int_value = 3
    artifact.properties["bar"].string_value = "Hello"
    artifact_id = store.create_artifact_with_type(artifact, artifact_type)
    [artifact_result] = store.get_artifacts_by_id([artifact_id])
    self.assertEqual(artifact_result.properties["bar"].string_value, "Hello")
    self.assertEqual(artifact_result.properties["foo"].int_value, 3)

  def test_put_execution_type_get_execution_type(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType()
    execution_type.name = "test_type_2"
    execution_type.properties["foo"] = metadata_store_pb2.INT
    execution_type.properties["bar"] = metadata_store_pb2.STRING
    type_id = store.put_execution_type(execution_type)
    execution_type_result = store.get_execution_type("test_type_2")
    self.assertEqual(execution_type_result.id, type_id)
    self.assertEqual(execution_type_result.name, "test_type_2")

  def test_put_executions_get_executions_by_id(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType()
    execution_type.name = "test_type_2"
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
    execution_type = _create_example_execution_type()
    type_id = store.put_execution_type(execution_type)
    execution_0 = metadata_store_pb2.Execution()
    execution_0.type_id = type_id
    execution_0.properties["foo"].int_value = 3
    execution_0.properties["bar"].string_value = "Hello"
    execution_1 = metadata_store_pb2.Execution()
    execution_1.type_id = type_id
    execution_1.properties["foo"].int_value = -9
    execution_1.properties["bar"].string_value = "Goodbye"

    [execution_id_0,
     execution_id_1] = store.put_executions([execution_0, execution_1])

    execution_result = store.get_executions()
    self.assertLen(execution_result, 2)
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

  def test_update_execution_get_execution(self):
    store = _get_metadata_store()
    execution_type = metadata_store_pb2.ExecutionType()
    execution_type.name = "test_type_2"
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
    execution_type.name = "execution_type"
    execution_type_id = store.put_execution_type(execution_type)
    execution = metadata_store_pb2.Execution()
    execution.type_id = execution_type_id
    [execution_id] = store.put_executions([execution])
    artifact_type = metadata_store_pb2.ArtifactType()
    artifact_type.name = "artifact_type"
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
    """See b/122594744."""
    store = _get_metadata_store()
    result = store.get_executions_by_id({})
    self.assertEmpty(result)


if __name__ == "__main__":
  absltest.main()
