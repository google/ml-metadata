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
"""A set of MLMD simple types."""

import abc
from typing import Union

from ml_metadata import errors
from ml_metadata.metadata_store.pywrap.metadata_store_extension import metadata_store as metadata_store_serialized
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.simple_types.proto import simple_types_pb2


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
    return exc_type(msg)
  except KeyError:
    return errors.UnknownError(msg)


def _get_type_proto(
    type_name: str, types: simple_types_pb2.SimpleTypes
) -> Union[metadata_store_pb2.ArtifactType, metadata_store_pb2.ExecutionType]:
  for artifact_type in types.artifact_types:
    if artifact_type.name == type_name:
      return artifact_type
  for execution_type in types.execution_types:
    if execution_type.name == type_name:
      return execution_type
  raise _make_exception('Type name: {} is not found'.format(type_name),
                        errors.NOT_FOUND)


class _SystemType(abc.ABC):
  """An abstract simple type."""

  def __init__(self, type_name: str):
    """Creates a type instance from system constants.

    Constructs an abstract system type based on 'type_name'. It loads simple
    type proto messages from simple_types_constants.cc and sets '_type'
    attribute based on 'type_name' parameter.

    Args:
      type_name: name of the desired system type.

    Raises:
      NOT_FOUND: if 'type_name' is not found in the pre-loaded simple type list;
      It also raises the corresponding error from wrapped LoadSimpleTypes util
      method.
    """
    [types_str, error_message,
     status_code] = metadata_store_serialized.LoadSimpleTypes()
    if status_code:
      raise _make_exception(error_message.decode('utf-8'), status_code)
    types = simple_types_pb2.SimpleTypes()
    types.ParseFromString(types_str)
    self._type = _get_type_proto(type_name, types)

  @property
  def name(self) -> str:
    return self._type.name


class ArtifactType(_SystemType):
  """An abstract simple artifact type for annotating artifact types."""

  def __init__(
      self, system_type: metadata_store_pb2.ArtifactType.SystemDefinedBaseType):
    self._system_type = system_type
    extensions = metadata_store_pb2.ArtifactType.SystemDefinedBaseType.DESCRIPTOR.values_by_number[
        system_type].GetOptions().Extensions
    type_name = extensions[metadata_store_pb2.system_type_extension].type_name
    super().__init__(type_name)

  @property
  def system_type(
      self) -> metadata_store_pb2.ArtifactType.SystemDefinedBaseType:
    return self._system_type


class ExecutionType(_SystemType):
  """An abstract simple execution type for annotating execution types."""

  def __init__(
      self,
      system_type: metadata_store_pb2.ExecutionType.SystemDefinedBaseType):
    self._system_type = system_type
    extensions = metadata_store_pb2.ExecutionType.SystemDefinedBaseType.DESCRIPTOR.values_by_number[
        system_type].GetOptions().Extensions
    type_name = extensions[metadata_store_pb2.system_type_extension].type_name
    super().__init__(type_name)

  @property
  def system_type(
      self) -> metadata_store_pb2.ExecutionType.SystemDefinedBaseType:
    return self._system_type


# A list of system pre-defined artifact types.
class Dataset(ArtifactType):
  """Dataset is a system pre-defined artifact type."""

  def __init__(self):
    super().__init__(metadata_store_pb2.ArtifactType.DATASET)


class Model(ArtifactType):
  """Model is a system pre-defined artifact type."""

  def __init__(self):
    super().__init__(metadata_store_pb2.ArtifactType.MODEL)


class Statistics(ArtifactType):
  """Statistics is a system pre-defined artifact type."""

  def __init__(self):
    super().__init__(metadata_store_pb2.ArtifactType.STATISTICS)


class Metrics(ArtifactType):
  """Metrics is a system pre-defined artifact type."""

  def __init__(self):
    super().__init__(metadata_store_pb2.ArtifactType.METRICS)


# A list of system pre-defined execution types.
class Train(ExecutionType):
  """Train is a system pre-defined execution type."""

  def __init__(self):
    super().__init__(metadata_store_pb2.ExecutionType.TRAIN)


class Transform(ExecutionType):
  """Transform is a system pre-defined execution type."""

  def __init__(self):
    super().__init__(metadata_store_pb2.ExecutionType.TRANSFORM)


class Process(ExecutionType):
  """Process is a system pre-defined execution type."""

  def __init__(self):
    super().__init__(metadata_store_pb2.ExecutionType.PROCESS)


class Evaluate(ExecutionType):
  """Evaluate is a system pre-defined execution type."""

  def __init__(self):
    super().__init__(metadata_store_pb2.ExecutionType.EVALUATE)


class Deploy(ExecutionType):
  """Deploy is a system pre-defined execution type."""

  def __init__(self):
    super().__init__(metadata_store_pb2.ExecutionType.DEPLOY)
