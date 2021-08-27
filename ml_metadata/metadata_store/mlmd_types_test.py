# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for mlmd_types."""

from absl.testing import absltest

from ml_metadata.metadata_store import mlmd_types
from ml_metadata.proto import metadata_store_pb2


def _get_artifact_type_name(
    system_type: metadata_store_pb2.ArtifactType.SystemDefinedBaseType):
  extensions = metadata_store_pb2.ArtifactType.SystemDefinedBaseType.DESCRIPTOR.values_by_number[
      system_type].GetOptions().Extensions
  return extensions[metadata_store_pb2.system_type_extension].type_name


def _get_execution_type_name(
    system_type: metadata_store_pb2.ExecutionType.SystemDefinedBaseType):
  extensions = metadata_store_pb2.ExecutionType.SystemDefinedBaseType.DESCRIPTOR.values_by_number[
      system_type].GetOptions().Extensions
  return extensions[metadata_store_pb2.system_type_extension].type_name


class MlmdTypesTest(absltest.TestCase):

  def testSystemArtifactTypes(self):
    self.assertEqual(
        _get_artifact_type_name(metadata_store_pb2.ArtifactType.DATASET),
        mlmd_types.Dataset().name)
    self.assertEqual(
        _get_artifact_type_name(metadata_store_pb2.ArtifactType.MODEL),
        mlmd_types.Model().name)
    self.assertEqual(
        _get_artifact_type_name(metadata_store_pb2.ArtifactType.METRICS),
        mlmd_types.Metrics().name)
    self.assertEqual(
        _get_artifact_type_name(metadata_store_pb2.ArtifactType.STATISTICS),
        mlmd_types.Statistics().name)

  def testSystemExecutionTypes(self):
    self.assertEqual(
        _get_execution_type_name(metadata_store_pb2.ExecutionType.TRAIN),
        mlmd_types.Train().name)
    self.assertEqual(
        _get_execution_type_name(metadata_store_pb2.ExecutionType.TRANSFORM),
        mlmd_types.Transform().name)
    self.assertEqual(
        _get_execution_type_name(metadata_store_pb2.ExecutionType.PROCESS),
        mlmd_types.Process().name)
    self.assertEqual(
        _get_execution_type_name(metadata_store_pb2.ExecutionType.EVALUATE),
        mlmd_types.Evaluate().name)
    self.assertEqual(
        _get_execution_type_name(metadata_store_pb2.ExecutionType.DEPLOY),
        mlmd_types.Deploy().name)


if __name__ == "__main__":
  absltest.main()
