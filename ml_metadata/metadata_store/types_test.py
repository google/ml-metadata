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
"""Tests for ml_metadata.metadata_store.types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl.testing import absltest

import unittest
from ml_metadata.metadata_store import metadata_store
from ml_metadata.metadata_store import types
from ml_metadata.proto import metadata_store_pb2


def _create_metadata_store():
  """Creates a new metadata store."""
  # Need to clear the registered types if you are connecting to a new database.
  types.clear_registered_types()
  connection_config = metadata_store_pb2.ConnectionConfig()
  connection_config.sqlite.SetInParent()
  return metadata_store.MetadataStore(connection_config)


def _create_example_artifact_type():
  return types.create_artifact_type(
      "test_type",
      foo=metadata_store_pb2.INT,
      bar=metadata_store_pb2.STRING,
      baz=metadata_store_pb2.DOUBLE)


def _create_example_artifact_type_2():
  return types.create_artifact_type(
      "test_type_2", foo=metadata_store_pb2.INT, bar=metadata_store_pb2.STRING)


def _create_example_artifact_proto():
  artifact = metadata_store_pb2.Artifact()
  artifact.properties["foo"].int_value = 3
  artifact.properties["bar"].string_value = "hello"
  artifact.properties["baz"].double_value = 1.25
  return artifact


def _create_example_artifact():
  return types.Artifact(_create_example_artifact_proto(),
                        _create_example_artifact_type())


def _create_schema_type():
  return types.create_artifact_type("Schema", version=metadata_store_pb2.INT)


def _create_data_type():
  return types.create_artifact_type(
      "Data",
      span=metadata_store_pb2.INT,
      split=metadata_store_pb2.STRING,
      version=metadata_store_pb2.INT)


def _create_data_artifact():
  return types.Artifact.create(
      _create_data_type(), uri="http://abc", span=3, split="TRAIN", version=1)


def _create_schema_artifact():
  return types.Artifact.create(
      _create_schema_type(), uri="http://xyz", version=3)


def _create_stats_artifact_type():
  return types.create_artifact_type("Stats")


def _create_stats_artifact():
  return types.Artifact.create(_create_stats_artifact_type())


def _create_stats_gen_execution_type():
  return types.ExecutionType.create(
      name="stats_gen",
      properties={
          "foo": metadata_store_pb2.INT,
          "bar": metadata_store_pb2.STRING
      },
      input_type={
          "data": _create_data_type(),
          "schema": _create_schema_type()
      },
      output_type=_create_stats_artifact_type())


def _create_transform_data_execution_type():
  return types.ExecutionType.create(
      name="transform_data",
      properties={
          "foo": metadata_store_pb2.INT,
          "bar": metadata_store_pb2.STRING
      },
      input_type=_create_data_type(),
      output_type=_create_data_type())


def _create_stats_gen_execution():
  return types.Execution.create(
      _create_stats_gen_execution_type(), {
          "data": _create_data_artifact(),
          "schema": _create_schema_artifact()
      },
      _create_stats_artifact(),
      foo=3,
      bar="baz")


class ArtifactsTest(absltest.TestCase):

  def test_init_artifact(self):
    artifact_type = _create_example_artifact_type()
    artifact = types.Artifact(_create_example_artifact_proto(), artifact_type)
    self.assertTrue(artifact.is_instance_of_type(artifact_type))

  def test_init_artifact_error_inconsistent_type(self):
    """The artifact does not match the artifact type."""
    artifact_proto = _create_example_artifact_proto()
    artifact_type = _create_example_artifact_type_2()
    with self.assertRaises(ValueError):
      types.Artifact(artifact_proto, artifact_type)

  def test_id(self):
    artifact_proto = _create_example_artifact_proto()
    artifact_proto.id = 12
    artifact_type = _create_example_artifact_type()
    artifact = types.Artifact(artifact_proto, artifact_type)
    self.assertEqual(12, artifact.id)

  def test_get_attribute(self):
    artifact = _create_example_artifact()
    want_artifact = _create_example_artifact_proto()
    self.assertEqual(want_artifact.properties["foo"].int_value, artifact.foo)
    self.assertEqual(want_artifact.properties["bar"].string_value, artifact.bar)
    self.assertEqual(want_artifact.properties["baz"].double_value, artifact.baz)

  def test_get_property(self):
    artifact = _create_example_artifact()
    want_artifact = _create_example_artifact_proto()
    self.assertEqual(want_artifact.properties["foo"].int_value,
                     artifact.get_property("foo"))
    self.assertEqual(want_artifact.properties["bar"].string_value,
                     artifact.get_property("bar"))
    self.assertEqual(want_artifact.properties["baz"].double_value,
                     artifact.get_property("baz"))

  def test_is_instance_of_type(self):
    artifact = _create_example_artifact()
    self.assertFalse(
        artifact.is_instance_of_type(_create_example_artifact_type_2()))
    self.assertTrue(
        artifact.is_instance_of_type(_create_example_artifact_type()))

  def test_get_all_registered_artifact_types(self):
    example_type = _create_example_artifact_type()
    types.clear_registered_types()
    types.Artifact(_create_example_artifact_proto(), example_type)
    [one_type] = types.get_all_registered_artifact_types()
    self.assertEqual(example_type.name, one_type.name)

  def test_get_all_registered_artifact_types_is_instance(self):
    """This checks what types get registered when we run is_instance."""
    artifact = _create_example_artifact()
    type_2 = _create_example_artifact_type_2()
    types.clear_registered_types()
    types.is_instance(artifact, type_2)
    [one_type] = types.get_all_registered_artifact_types()
    self.assertEqual(type_2.name, one_type.name)

  def test_get_all_registered_artifact_types_create_artifact_type(self):
    """Creating two types with the same name and different properties is bad."""
    types.clear_registered_types()
    example_type = _create_example_artifact_type()
    [one_type] = types.get_all_registered_artifact_types()
    self.assertEqual(example_type.name, one_type.name)

  def test_get_all_registered_artifact_types_equal_types(self):
    """Creating the same type twice is fine."""
    types.clear_registered_types()
    example_type = _create_example_artifact_type()
    _create_example_artifact_type()
    [one_type] = types.get_all_registered_artifact_types()
    self.assertEqual(example_type.name, one_type.name)

  def test_get_all_registered_artifact_types_unequal_types(self):
    """Creating two types with the same name and different properties is bad."""
    types.clear_registered_types()
    types.create_artifact_type(
        "test_type",
        foo=metadata_store_pb2.INT,
        bar=metadata_store_pb2.STRING,
        baz=metadata_store_pb2.DOUBLE)
    with self.assertRaises(ValueError):
      types.create_artifact_type(
          "test_type",
          bar=metadata_store_pb2.STRING,
          baz=metadata_store_pb2.DOUBLE)

  def test_save_and_find_by_id(self):
    store = _create_metadata_store()
    artifact = _create_example_artifact()
    artifact.save(store)
    id_result = artifact.id
    result_artifact_2 = types.Artifact.find_by_id(store, id_result)
    self.assertEqual(result_artifact_2.id, id_result)
    [result_artifact_3] = types.Artifact.find_by_ids(store, [id_result])
    self.assertEqual(result_artifact_3.id, id_result)

  def test_save_and_find_by_ids_with_2(self):
    store = _create_metadata_store()
    artifact = _create_example_artifact()
    artifact.save(store)
    artifact_2 = _create_example_artifact()
    artifact_2.save(store)

    id_result = artifact.id
    id_result_2 = artifact_2.id
    [result_artifact, result_artifact_2] = types.Artifact.find_by_ids(
        store, [id_result, id_result_2])
    self.assertEqual(result_artifact.id, id_result)
    self.assertEqual(result_artifact_2.id, id_result_2)


class IsInstanceTest(unittest.TestCase):

  def test_simple_instance(self):
    artifact = _create_example_artifact()
    self.assertTrue(
        types.is_instance(artifact, _create_example_artifact_type()))
    self.assertFalse(
        types.is_instance(artifact, _create_example_artifact_type_2()))

  def test_dict_structure(self):
    artifact_struct_type = {
        "data": _create_data_type(),
        "schema": _create_schema_type()
    }
    artifact_struct = {
        "data": _create_data_artifact(),
        "schema": _create_schema_artifact()
    }
    artifact_struct_bad = {
        "data2": _create_data_artifact(),
        "schema": _create_schema_artifact()
    }

    self.assertTrue(types.is_instance(artifact_struct, artifact_struct_type))
    self.assertFalse(
        types.is_instance(artifact_struct_bad, artifact_struct_type))

  def test_list_structure(self):
    artifact_list = [_create_data_artifact(), _create_data_artifact()]
    bad_artifact_list = [_create_data_artifact(), _create_schema_artifact()]

    self.assertTrue(
        types.is_instance(artifact_list, types.list_of(_create_data_type())))
    self.assertFalse(
        types.is_instance(bad_artifact_list,
                          types.list_of(_create_data_type())))

  def test_tuple_structure(self):
    artifact_struct_type = [_create_data_type(), _create_schema_type()]
    artifact_struct = [_create_data_artifact(), _create_schema_artifact()]
    bad_artifact_struct = [_create_schema_artifact(), _create_schema_artifact()]

    self.assertTrue(types.is_instance(artifact_struct, artifact_struct_type))
    self.assertFalse(
        types.is_instance(bad_artifact_struct, artifact_struct_type))

  def test_union_of(self):
    artifact_struct_type = types.union_of(_create_data_type(),
                                          _create_schema_type())
    artifact_struct = _create_data_artifact()
    bad_artifact_struct = None

    self.assertTrue(types.is_instance(artifact_struct, artifact_struct_type))
    self.assertFalse(
        types.is_instance(bad_artifact_struct, artifact_struct_type))

  def test_json_to_artifact_struct_and_back(self):
    original_struct = {
        "data": [_create_data_artifact(),
                 _create_data_artifact()],
        "schema": _create_schema_artifact()
    }
    serialized_a = types.create_json(original_struct)
    second_struct = types.create_artifact_struct_from_json(serialized_a)

    serialized_b = types.create_json(second_struct)
    parsed_json_a = json.loads(serialized_a)
    parsed_json_b = json.loads(serialized_b)
    self.assertEqual(parsed_json_a, parsed_json_b)

  def test_json_to_artifact_struct_and_back_with_custom(self):
    original_struct = {
        "data": [_create_data_artifact(),
                 _create_data_artifact()],
        "schema": _create_schema_artifact()
    }
    original_struct["data"][0].set_custom_property("foo", "bar")
    serialized_a = types.create_json(original_struct)
    second_struct = types.create_artifact_struct_from_json(serialized_a)
    self.assertEqual("bar", second_struct["data"][0].get_custom_property("foo"))


class ExecutionTypeTest(absltest.TestCase):

  def test_create_init_execution(self):
    execution_type = _create_stats_gen_execution_type()
    self.assertEqual("stats_gen", execution_type.type.name)

  def test_save_and_find_by_id(self):
    store = _create_metadata_store()
    execution_type = _create_stats_gen_execution_type()
    execution_type.save(store)

    execution_type_id = execution_type.type.id
    execution_type_result = types.ExecutionType.find_by_id(
        store, execution_type_id)
    self.assertEqual(execution_type.type.name, execution_type_result.type.name)

  def test_save_and_find_by_ids(self):
    store = _create_metadata_store()
    execution_type = _create_stats_gen_execution_type()
    execution_type.save(store)
    execution_type_2 = _create_transform_data_execution_type()
    execution_type_2.save(store)

    execution_type_id = execution_type.type.id
    execution_type_id_2 = execution_type_2.type.id
    [execution_type_result,
     execution_type_result_2] = types.ExecutionType.find_by_ids(
         store, [execution_type_id, execution_type_id_2])
    self.assertEqual(execution_type.type.name, execution_type_result.type.name)
    self.assertEqual(execution_type_2.type.name,
                     execution_type_result_2.type.name)


class ExecutionsTest(absltest.TestCase):

  def test_create_init_execution(self):
    execution_type = _create_stats_gen_execution_type()
    execution = _create_stats_gen_execution()
    self.assertTrue(execution.is_instance_of_type(execution_type))

  def test_init_execution_error_inconsistent_type(self):
    """The artifact does not match the artifact type."""
    execution_type = _create_stats_gen_execution_type()
    with self.assertRaises(ValueError):
      types.Execution.create(execution_type, None, None, no_field=6)

  def test_id(self):
    execution_proto = metadata_store_pb2.Execution()
    execution_proto.id = 12
    execution = types.Execution(execution_proto,
                                _create_stats_gen_execution_type(), {}, None)
    self.assertEqual(12, execution.id)

  def test_get_attribute(self):
    execution = _create_stats_gen_execution()
    self.assertEqual(3, execution.foo)
    self.assertEqual("baz", execution.bar)

  def test_get_property(self):
    execution = _create_stats_gen_execution()
    self.assertEqual(3, execution.get_property("foo"))
    self.assertEqual("baz", execution.get_property("bar"))

  def test_is_instance_of_type(self):
    execution = _create_stats_gen_execution()
    self.assertFalse(
        execution.is_instance_of_type(_create_transform_data_execution_type()))
    self.assertTrue(
        execution.is_instance_of_type(_create_stats_gen_execution_type()))

  def test_get_all_registered_execution_types(self):
    types.clear_registered_types()
    execution_type = _create_stats_gen_execution_type()
    [one_type] = types.get_all_registered_execution_types()
    self.assertEqual(execution_type.type.name, one_type.name)

  def test_get_all_registered_execution_types_equal_types(self):
    """Creating the same type twice is fine."""
    types.clear_registered_types()
    _create_stats_gen_execution_type()
    _create_stats_gen_execution_type()
    [one_type] = types.get_all_registered_execution_types()
    self.assertEqual("stats_gen", one_type.name)

  def test_get_all_registered_execution_types_unequal_types(self):
    """Creating two types with the same name and different properties is bad."""
    types.clear_registered_types()
    _create_stats_gen_execution_type()
    with self.assertRaises(ValueError):
      # Removed property bar, but kept the same name.
      types.ExecutionType.create(
          name="stats_gen",
          properties={
              "foo": metadata_store_pb2.INT,
          },
          input_type={
              "data": _create_data_type(),
              "schema": _create_schema_type()
          },
          output_type=_create_stats_artifact_type())

  def test_is_input_consistent(self):
    # data_2 should be data, which causes an inconsistency.
    invalid_input = types.Execution.create(
        _create_stats_gen_execution_type(), {
            "data_2": _create_data_artifact(),
            "schema": _create_schema_artifact()
        },
        _create_stats_artifact(),
        foo=3,
        bar="baz")
    self.assertFalse(invalid_input.is_input_consistent())
    self.assertTrue(_create_stats_gen_execution().is_input_consistent())

  def test_is_output_consistent(self):
    # Output of stats gen is just a stats proto.
    invalid_output = types.Execution.create(
        _create_stats_gen_execution_type(), {
            "data": _create_data_artifact(),
            "schema": _create_schema_artifact()
        },
        _create_data_artifact(),
        foo=3,
        bar="baz")
    self.assertFalse(invalid_output.is_output_consistent())
    self.assertTrue(_create_stats_gen_execution().is_output_consistent())

  def test_is_consistent(self):
    invalid_output = types.Execution.create(
        _create_stats_gen_execution_type(), {
            "data": _create_data_artifact(),
            "schema": _create_schema_artifact()
        },
        _create_data_artifact(),
        foo=3,
        bar="baz")
    invalid_input = types.Execution.create(
        _create_stats_gen_execution_type(), {
            "data_2": _create_data_artifact(),
            "schema": _create_schema_artifact()
        },
        _create_stats_artifact(),
        foo=3,
        bar="baz")
    self.assertFalse(invalid_input.is_consistent())
    self.assertFalse(invalid_output.is_consistent())
    self.assertTrue(_create_stats_gen_execution().is_consistent())

  def test_save_and_find_by_id(self):
    store = _create_metadata_store()
    stats_gen_execution = _create_stats_gen_execution()
    stats_gen_execution.save(store)
    execution_id = stats_gen_execution.id
    execution_result = types.Execution.find_by_id(store, execution_id)
    self.assertTrue(execution_result.id, stats_gen_execution.id)
    self.assertEqual(3, execution_result.input_struct["schema"].version)
    self.assertEqual("Stats", execution_result.output_struct.type.name)

  def test_save_execution_and_find_by_id(self):
    store = _create_metadata_store()
    stats_gen_execution = _create_stats_gen_execution()
    stats_gen_execution.save_execution(store)
    execution_id = stats_gen_execution.id
    execution_result = types.Execution.find_by_id(store, execution_id)
    self.assertTrue(execution_result.id, stats_gen_execution.id)
    self.assertEqual(None, execution_result.input_struct)
    self.assertEqual(None, execution_result.output_struct)

  def test_save_output_and_find_by_id(self):
    store = _create_metadata_store()
    stats_gen_execution = _create_stats_gen_execution()
    stats_gen_execution.save_execution(store)
    stats_gen_execution.save_output(store)
    execution_id = stats_gen_execution.id
    execution_result = types.Execution.find_by_id(store, execution_id)
    self.assertTrue(execution_result.id, stats_gen_execution.id)
    self.assertEqual(None, execution_result.input_struct)
    self.assertEqual("Stats", execution_result.output_struct.type.name)

  def test_save_output_twice(self):
    """Saving the output twice, even if it is the same, is disallowed."""
    store = _create_metadata_store()
    stats_gen_execution = _create_stats_gen_execution()
    stats_gen_execution.save_execution(store)
    stats_gen_execution.save_output(store)
    with self.assertRaises(ValueError):
      stats_gen_execution.save_output(store)

  def test_save_input_and_find_by_id(self):
    store = _create_metadata_store()
    stats_gen_execution = _create_stats_gen_execution()
    stats_gen_execution.save_execution(store)
    stats_gen_execution.save_input(store)
    execution_id = stats_gen_execution.id
    execution_result = types.Execution.find_by_id(store, execution_id)
    self.assertTrue(execution_result.id, stats_gen_execution.id)
    self.assertEqual(3, execution_result.input_struct["schema"].version)
    self.assertEqual(None, execution_result.output_struct)

  def test_save_input_twice(self):
    """Saving the input twice, even if it is the same, is disallowed."""
    store = _create_metadata_store()
    stats_gen_execution = _create_stats_gen_execution()
    stats_gen_execution.save_execution(store)
    stats_gen_execution.save_input(store)
    with self.assertRaises(ValueError):
      stats_gen_execution.save_input(store)

  def test_save_output_later(self):
    """Save execution and input, then save output.

    This is a standard pattern.
    """
    store = _create_metadata_store()
    stats_gen_execution = _create_stats_gen_execution()
    stats_gen_execution.save_execution(store)
    stats_gen_execution.save_input(store)
    execution_id = stats_gen_execution.id
    execution_result = types.Execution.find_by_id(store, execution_id)
    self.assertTrue(execution_result.id, stats_gen_execution.id)
    execution_result.output_struct = _create_stats_artifact()
    execution_result.save_output(store)
    execution_result_2 = types.Execution.find_by_id(store, execution_id)
    self.assertTrue(execution_result_2.id, stats_gen_execution.id)
    self.assertEqual(3, execution_result_2.input_struct["schema"].version)
    self.assertEqual("Stats", execution_result_2.output_struct.type.name)


if __name__ == "__main__":
  absltest.main()
