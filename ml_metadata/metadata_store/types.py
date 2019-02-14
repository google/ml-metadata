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
"""Objects for the high-level API.

Artifacts, Executions, ExecutionTypes, ArtifactStructs, and ArtifactStructTypes.

# Some basic artifact types for schema, data, and stats.
schema_type = types.create_artifact_type(
    "Schema",
    version = metadata_store_pb2.INT)
data_type = types.create_artifact_type(
    "Data",
    span=metadata_store_pb2.INT,
    split=metadata_store_pb2.STRING,
    version=metadata_store_pb2.INT)

# Creates a type with no properties.
stats_artifact_type = types.create_artifact_type("Statistics")

# Create a schema and data artifacts.
schema_obj = types.Artifact.create(schema_type,
                                   uri="http://xyz",
                                   version= 3)

data_artifact = types.Artifact.create(data_type,
  uri="http://abc",
  span= 3,
  split= "TRAIN",
  version= 1
)

# Create an ArtifactStruct that can be input to ArtifactStruct.
input_to_stats = {"schema":schema_obj, "data":data_artifact}


# Test if input_to_stats has the right type.
assert(types.is_instance(input_to_stats,
                         {"schema":schema_type, "data": data_type}))

# Create the execution type. Note that the input type and output type are
# specified.
stats_execution_type = types.Execution.create(
  name="Stats",                                    # Name of type
  properties={"mpm_version": metadata_store_pb2.INT},    # Properties of
  execution
  input_type={"schema":schema_type, "data": data_type},  # Input types
  output_type=stats_artifact_type                         # Output types
)


stats_execution = types.Execution.create(
  stats_execution_type,
  input_to_stats,
  None
  mpm_version=7)

Struct types are logically sets of structures. Basic concepts, such as
artifact types, homogeneous lists, dictionaries, the universe of all structures,
the empty set, intersection, "not", and union, can be represented.

Any ArtifactType or ExecutionType protos referred to are "registered" locally.
Specifically, this means that later they can be checked or added to the
database, without directly accessing the database while typechecking. If there
is a local conflict, e.g. two types with the same name are not equal, or they
both have IDs, but those IDs are different, then a ValueError is immediately
thrown.

This functionality can be paused, restarted, and the registry can be cleared.

Continuing the example from above:

# This will return the schema, data, and stats artifact types from above.
registered_artifact_types = types.get_all_registered_artifact_types()

# This will return the stats execution type from above.
registered_execution_types = types.get_all_registered_execution_types()

# This clears the registry.
types.clear_registered_types()

# These will now return empty lists.
types.get_all_registered_artifact_types()
types.get_all_registered_execution_types()

data_artifact_2 = types.Artifact.create(data_type,
  "uri"="http://abc2",
  "span"= 4,
  "split"= "TRAIN",
  "version"= 1
)

# Now data_type is registered again.

assert(not types.is_instance(input_to_stats_2, schema_type))

# Now schema_type is registered again.

types.stop_registering_types()

# Now, there will be no side-effects.
assert(not types.is_instance(input_to_stats_2, stats_artifact_type))

# stats_artifact_type will not be added to the registry.

types.start_registering_types()
# Now, we register types again.

assert(not types.is_instance(input_to_stats_2, stats_artifact_type))
# stats_artifact_type is added to the registry.


# This will return the schema, data, and stats artifact types from above.
types.get_all_registered_artifact_types()


The core of the type system is ArtifactStructType. ArtifactStructType has an
is_type_of method. Conceptually, one can think of ArtifactStructType X as a set
of ArtifactStruct objects Y where X.is_type_of(Y)==True.

create_artifact_struct_type() takes an object that can be "coerced" or "cast"
into an ArtifactStructType, called CoercableToType, and makes an
ArtifactStructType. Most methods take CoercableToType, but immediately cast it
to ArtifactStructType.

Examples of CoercableToType are:
1. metadata_store_pb2.ArtifactType: represents all artifact structs which are
   an artifact of that type.
2. ArtifactStructType itself: is passed through create_artifact_struct_type()
   unchanged.
3. {"foo":type_a, "bar":type_b}: represents a struct that is a dictionary. For
   more detail, see DictArtifactStructType below.
4. [type_a, type_b, type_c] represents a tuple, i.e. a list. For example,
   an ArtifactStruct [x, y, z] would satisfy the above tuple type iff x was of
   type_a, y was of type_b, and z was of type_c.

Other ways to create a type are:

none(): a type that only accepts None.
any_type(): a type that accepts anything.
list_of(type_a): a type that expects a list of type_a
union_of(type_a, type_b): a type that expects either type_a or type_b.
intersection_of(type_a, type_b): a type that an artifact that is both
type_a and type_b.
optional(type_a): a type that accepts artifacts of type_a or None.

In the future, we probably want to be able to check subtypes of artifact
structs.
TODO(b/124072881): move a lot of this logic to be in C++, so it can be shared
across languages.




"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import copy
import json
import threading
import six
import typing

from typing import Optional, Text, Dict
from ml_metadata.proto import metadata_store_pb2


def create_artifact_type(name: Text,
                         **kwargs) -> metadata_store_pb2.ArtifactType:
  """Create an ArtifactType without putting it in the database.

  Registers any ArtifactTypes locally for checking later.

  Example:
  create_artifact_type("Schema", version=metadata_store_pb2.INT)
  Args:
    name: the name of the artifact type.
    **kwargs: properties of new type.

  Returns:
    a new ArtifactType proto.
  """
  result = metadata_store_pb2.ArtifactType()
  result.name = name
  for k, v in kwargs.items():
    result.properties[k] = v
  _registered_types.register_artifact_type_as_used(result)
  return result


def _properties_are_equal(properties_a, properties_b) -> bool:
  """Test that the keys and values of these maps are equal."""
  for k, v in properties_a.items():
    if k not in properties_b.keys():
      return False
    if v != properties_b[k]:
      return False
  for k, _ in properties_b.items():
    if k not in properties_a.keys():
      return False
  return True


def _types_are_equal(
    a: typing.Union[metadata_store_pb2.ArtifactType, metadata_store_pb2
                    .ExecutionType],
    b: typing.Union[metadata_store_pb2.ArtifactType, metadata_store_pb2
                    .ExecutionType]) -> bool:
  """Returns true if a and b have equal names and properties.

  Also checks that both are the same kind of type (Execution or Artifact),
  and if both have an ID, then they are the same.

  Args:
    a: an artifact or execution type.
    b: an artifact or execution type.

  Returns:
    True if all all the constraints above are satisfied.
  """
  if type(a) != type(b):  # pylint: disable=unidiomatic-typecheck
    # This checks that they are both ExecutionTypes or ArtifactTypes.
    return False
  if a.name != b.name:
    return False

  if a.HasField("id") and b.HasField("id") and a.id != b.id:
    return False

  return _properties_are_equal(a.properties, b.properties)


# Mapping from Value Enum to the field in PropertyValue.
_appropriate_values = {
    metadata_store_pb2.UNKNOWN: "UNKNOWN",
    metadata_store_pb2.INT: "int_value",
    metadata_store_pb2.DOUBLE: "double_value",
    metadata_store_pb2.STRING: "string_value"
}


def _value_is_consistent(
    value: metadata_store_pb2.Value,
    primitive_type: metadata_store_pb2.PropertyType) -> bool:
  """Tests if a value is consistent with the type of that property."""
  appropriate_value = _appropriate_values[primitive_type]
  return appropriate_value == value.WhichOneof("value")


def _get_primitive(value: metadata_store_pb2.Value,
                   primitive_type: metadata_store_pb2.PropertyType
                  ) -> typing.Union[int, Text, float, None]:
  """Gets the primitive (float, string, int, or None) from value."""
  if primitive_type == metadata_store_pb2.INT:
    return value.int_value
  if primitive_type == metadata_store_pb2.STRING:
    return value.string_value
  if primitive_type == metadata_store_pb2.DOUBLE:
    return value.double_value
  return None


def _get_custom_primitive(
    value: metadata_store_pb2.Value) -> typing.Union[int, Text, float, None]:
  """Gets the primitive (float, string, int, or None) from value."""
  if value.HasField("int_value"):
    return value.int_value
  if value.HasField("string_value"):
    return value.string_value
  if value.HasField("double_value"):
    return value.double_value
  return None


def _set_value(result: metadata_store_pb2.Value, primitive,
               primitive_type: metadata_store_pb2.PropertyType) -> None:
  """Sets a value according to a property type."""
  if primitive_type == metadata_store_pb2.INT:
    result.int_value = primitive
  elif primitive_type == metadata_store_pb2.STRING:
    result.string_value = primitive
  elif primitive_type == metadata_store_pb2.DOUBLE:
    result.double_value = primitive
  else:
    raise ValueError("unknown property type")


def _set_custom_value(result: metadata_store_pb2.Value,
                      primitive: typing.Union[int, float, Text]) -> None:
  """Sets a value according to a property type."""
  if isinstance(primitive, int):
    result.int_value = primitive
  elif isinstance(primitive, float):
    result.double_value = primitive
  elif isinstance(primitive, (six.string_types, six.text_type)):
    result.string_value = primitive
  else:
    raise ValueError("unknown property type")


class _NodeAndType(object):
  """A base type for Artifact and Execution."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def _get_node(
      self
  ) -> typing.Union[metadata_store_pb2.Artifact, metadata_store_pb2.Execution]:
    """Gets the underlying proto."""

  @abc.abstractmethod
  def _get_type(
      self) -> typing.Union[metadata_store_pb2.ArtifactType, metadata_store_pb2
                            .ExecutionType]:
    """Gets the underlying type proto."""

  def _is_consistent(self) -> bool:
    """Test if the type and the node are consistent."""
    node = self._get_node()
    node_type = self._get_type()
    if node.HasField("type_id") and node_type.HasField("id"):
      if node.type_id != node_type.id:
        return False

    # All fields are optional, by definition.
    for k, v in node.properties.items():
      if k not in node_type.properties:
        return False
      if not _value_is_consistent(v, node_type.properties[k]):
        return False
    return True

  @property
  def id(self) -> int:
    return self._get_node().id

  def has_id(self) -> bool:
    return self._get_node().HasField("id")

  def __getattr__(self, attr):
    """Gets a property of the underlying node and unwraps it."""
    return self.get_property(attr)

  def get_property(self, attr) -> typing.Union[int, float, Text, None]:
    """Gets a property of the underlying node and unwraps it."""
    property_type = self._get_type().properties.get(attr)
    if property_type is None:
      raise ValueError("Type {} has no property {}.".format(
          self._get_type().name, attr))
    property_value = self._get_node().properties.get(attr)
    if property_value is None:
      return None
    return _get_primitive(property_value, property_type)

  def get_custom_property(self, attr) -> typing.Union[int, float, Text, None]:
    """Gets a custom property of the artifact."""
    property_value = self.artifact.custom_properties.get(attr)
    if property_value is None:
      return None
    return _get_custom_primitive(property_value)

  def set_custom_property(self, attr,
                          value: typing.Union[int, float, Text, None]):
    """Sets a property of the underlying artifact (wrapping the value)."""
    value_container = self.artifact.custom_properties[attr]
    _set_custom_value(value_container, value)

  def set_property(self, attr, value: typing.Union[int, float, Text, None]):
    """Sets a property of the underlying artifact (wrapping the value)."""
    property_type = self._get_type().properties.get(attr)
    if property_type is None:
      raise ValueError("Type {} has no property {}.".format(
          self._get_type().name, attr))
    value_container = self._get_node().properties[attr]
    _set_value(value_container, value, property_type)

  def __setattr__(self, attr, value):
    """Sets a property of the underlying artifact (wrapping the value)."""
    self.set_property(attr, value)


class Artifact(_NodeAndType):
  """A class representing an artifact and its associated type.

  Note: an artifact type can be shared between different Artifact
  objects.
  """

  def __init__(self, artifact: metadata_store_pb2.Artifact,
               artifact_type: metadata_store_pb2.ArtifactType):
    """Registers the type locally, and checks for internal consistency."""
    # self._artifact = artifact does not work, because we have a specialized
    # __setattr__ method.
    self.__dict__["_artifact"] = artifact
    self.__dict__["_type"] = artifact_type
    if not self._is_consistent():
      raise ValueError("Type is not internally consistent")
    _registered_types.register_artifact_type_as_used(artifact_type)

  @classmethod
  def create(cls, artifact_type: metadata_store_pb2.ArtifactType,
             **kwargs) -> "Artifact":
    """Creates an Artifact without committing it to the database."""
    result = Artifact(metadata_store_pb2.Artifact(), artifact_type)
    for k, v in kwargs.items():
      result.set_property(k, v)
    return result

  @classmethod
  def from_json(cls, from_json) -> "Artifact":
    """Constructs an Artifact object from parsed JSON."""
    artifact_type = _to_artifact_type(from_json["type"])
    artifact = metadata_store_pb2.Artifact()
    artifact.type_id = artifact_type.id
    result = Artifact(artifact, artifact_type)
    if "id" in from_json:
      result.id = from_json["id"]
    if "uri" in from_json:
      result.uri = from_json["uri"]
    if "properties" in from_json:
      for k, v in from_json["properties"].items():
        result.set_property(k, v)
    return result

  def _get_node(self) -> metadata_store_pb2.Artifact:
    return self._artifact

  def _get_type(self) -> metadata_store_pb2.ArtifactType:
    return self._type

  @property
  def artifact(self) -> metadata_store_pb2.Artifact:
    return self._artifact

  @property
  def type(self) -> metadata_store_pb2.ArtifactType:
    return self._type

  def is_instance_of_type(
      self, artifact_type: metadata_store_pb2.ArtifactType) -> bool:
    _registered_types.register_artifact_type_as_used(artifact_type)
    return _types_are_equal(artifact_type, self.type)

  @property
  def uri(self) -> Text:
    return self.artifact.uri

  def has_uri(self) -> bool:
    return self.artifact.HasField("uri")

  def set_property(self, attr, value: typing.Union[int, float, Text, None]):
    """Sets a property of the underlying artifact (wrapping the value)."""
    if attr == "uri":
      self.artifact.uri = value
    else:
      super(Artifact, self).set_property(attr, value)

  def _create_artifact_type_pre_json(self):
    """Create a dictionary that can be serialized to the JSON format."""
    result = {}
    if self.type.HasField("id"):
      result["id"] = self._type.id
    if self.type.HasField("name"):
      result["name"] = self._type.name
    if self.type.properties:
      properties = {}
      for k, v in self.type.properties.items():
        properties[k] = _property_type_to_text(v)
      result["properties"] = properties
    return result

  def _create_pre_json(self):
    """Create a dictionary that can be serialized to the JSON format."""
    result = {}
    result["type"] = self._create_artifact_type_pre_json()
    if self.has_id():
      result["id"] = self.id
    if self.has_uri():
      result["uri"] = self.uri
    if self.artifact.properties:
      properties = {}
      for k in self.artifact.properties.keys():
        properties[k] = self.get_property(k)
      result["properties"] = properties
    return result

  def __str__(self) -> str:  # pylint: disable=g-ambiguous-str-annotation
    return json.dumps(self._create_pre_json())


def _text_to_property_type(text: Text) -> metadata_store_pb2.PropertyType:
  """Converts JSON text to a PropertyType."""
  return {
      u"INT": metadata_store_pb2.INT,
      u"STRING": metadata_store_pb2.STRING,
      u"DOUBLE": metadata_store_pb2.DOUBLE
  }[text]


def _property_type_to_text(
    property_type: metadata_store_pb2.PropertyType) -> Text:
  """Converts PropertyType to a text for JSON."""
  return {
      metadata_store_pb2.INT: u"INT",
      metadata_store_pb2.STRING: u"STRING",
      metadata_store_pb2.DOUBLE: u"DOUBLE"
  }[property_type]


def _to_artifact_type(
    from_json: Dict[Text, typing.Any]) -> metadata_store_pb2.ArtifactType:
  """Converts parsed JSON to an artifact type."""
  result = metadata_store_pb2.ArtifactType()
  if "id" in from_json:
    result.id = from_json["id"]
  if "name" in from_json:
    result.name = from_json["name"]
  if "properties" in from_json:
    for k, v in from_json["properties"].items():
      result.properties[k] = _text_to_property_type(v)
  return result


def _is_serialized_artifact_type(from_json) -> bool:
  """Returns true if from_json["name"] is present."""
  return (isinstance(from_json, dict) and "name" in from_json and
          isinstance(from_json["name"], six.text_type))


def _is_serialized_artifact(from_json) -> bool:
  """Returns true if from_json["type"]["name"] is present."""
  return (isinstance(from_json, dict) and "type" in from_json and
          _is_serialized_artifact_type(from_json["type"]))


# Conceptually,
# ArtifactStruct = typing.Union[Dict[Text, ArtifactStruct],
#                        typing.List[ArtifactStruct],
#                        Artifact, None]
# However, pytype does not allow infinite recursion.
# Thus, we go three levels deep.
# NOTE: exporting this may be nasty. We may wish to make this GOOGLE_INTERNAL.
_ArtifactStruct0 = Optional[Artifact]
_ArtifactStruct1 = typing.Union[Dict[Text, _ArtifactStruct0], typing
                                .List[_ArtifactStruct0], _ArtifactStruct0]
_ArtifactStruct2 = typing.Union[Dict[Text, _ArtifactStruct1], typing
                                .List[_ArtifactStruct1], _ArtifactStruct1]
_ArtifactStruct3 = typing.Union[Dict[Text, _ArtifactStruct2], typing
                                .List[_ArtifactStruct2], _ArtifactStruct2]
ArtifactStruct = _ArtifactStruct3  # pylint: disable=invalid-name


def _create_artifact_struct_from_json_helper(from_json) -> ArtifactStruct:
  """Given parsed JSON, create an ArtifactStruct."""
  if _is_serialized_artifact(from_json):
    return Artifact.from_json(from_json)
  elif isinstance(from_json, dict):
    return {
        k: _create_artifact_struct_from_json_helper(v)
        for k, v in from_json.items()
    }
  elif isinstance(from_json, list):
    return [_create_artifact_struct_from_json_helper(x) for x in from_json]
  else:
    raise ValueError("Not a serialized ArtifactStruct")


def create_artifact_struct_from_json(json_text: Text) -> ArtifactStruct:
  """Given serialized JSON, create an ArtifactStruct."""
  return _create_artifact_struct_from_json_helper(json.loads(json_text))


def _create_pre_json(artifact_struct: ArtifactStruct):
  """Given an ArtifactStruct, create a structure that can be dumped to JSON."""
  if artifact_struct is None:
    raise NotImplementedError("Cannot serialize None yet")
  elif isinstance(artifact_struct, dict):
    return {k: _create_pre_json(v) for k, v in artifact_struct.items()}
  elif isinstance(artifact_struct, list):
    return [_create_pre_json(x) for x in artifact_struct]
  elif isinstance(artifact_struct, Artifact):
    return artifact_struct._create_pre_json()  # pylint:disable=protected-access
  else:
    raise NotImplementedError("Not an ArtifactStruct")


def create_json(artifact_struct: ArtifactStruct) -> Text:
  """Given an ArtifactStruct, create JSON text."""
  return json.dumps(_create_pre_json(artifact_struct))


class ArtifactStructType(object):
  """The base type for all type structures.

  At present, there is one method, is_instance. This is a useful place to
  begin, as it grounds the meaning of each type.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def is_type_of(self, struct: ArtifactStruct) -> bool:
    """Returns True iff struct is an instance of this type."""


def optional(struct_type: ArtifactStructType):
  """Returns an optional type."""
  return _UnionArtifactStructType([struct_type, _NoneArtifactStructType()])


def none() -> ArtifactStructType:
  """Returns a None type.

  A none type is satisfied only by the None object.

  Returns:
    An ArtifactStructType representing the none type.
  """
  return _NoneArtifactStructType()


def any_type() -> ArtifactStructType:
  """Returns an Any type.

  any_type().is_type_of(x) always returns true.

  Returns:
    an Any type.
  """
  return _AnyArtifactStructType()


class _NoneArtifactStructType(ArtifactStructType):
  """Type containing the None object."""

  def is_type_of(self, struct: ArtifactStruct) -> bool:
    return struct is None

  def __str__(self) -> str:  # pylint: disable=g-ambiguous-str-annotation
    return "None"


class _AnyArtifactStructType(ArtifactStructType):
  """Type containing all structs (top type)."""

  def is_type_of(self, struct: ArtifactStruct) -> bool:
    return True

  def __str__(self) -> str:  # pylint: disable=g-ambiguous-str-annotation
    return "Any"


class _UnionArtifactStructType(ArtifactStructType):
  """Union of multiple types."""

  def __init__(self, candidates: typing.List[ArtifactStructType]):
    self._candidates = candidates

  def is_type_of(self, struct: ArtifactStruct) -> bool:
    for candidate in self._candidates:
      if candidate.is_type_of(struct):
        return True
    return False

  def __str__(self) -> str:  # pylint: disable=g-ambiguous-str-annotation
    return "Union[{}]".format(", ".join([str(x) for x in self._candidates]))


class _IntersectionArtifactStructType(ArtifactStructType):
  """An intersection of types.

  This type is useful when:
  my_struct = ... # type = A
  if is_instance(my_struct, B):
    # my_struct type is Intersection(A, B)
    ...
  else:
    # my_struct type is Intersection(A,Not(B))
    ...

  """

  def __init__(self, candidates: typing.List[ArtifactStructType]):
    self._candidates = candidates

  def is_type_of(self, struct: ArtifactStruct) -> bool:
    for candidate in self._candidates:
      if not candidate.is_type_of(struct):
        return False
    return True

  def __str__(self) -> str:  # pylint: disable=g-ambiguous-str-annotation
    return "Intersection[{}]".format(", ".join(
        [str(x) for x in self._candidates]))


class _SimpleArtifactStructType(ArtifactStructType):
  """A type based on an ArtifactType."""

  def __init__(self, artifact_type: metadata_store_pb2.ArtifactType):
    """Initializes the type and registers the underlying proto as used."""
    self._type = artifact_type
    _registered_types.register_artifact_type_as_used(artifact_type)

  def is_type_of(self, struct: ArtifactStruct) -> bool:
    if not isinstance(struct, Artifact):
      return False
    return struct.is_instance_of_type(self._type)

  def __str__(self) -> str:  # pylint: disable=g-ambiguous-str-annotation
    return str(self._type)


class _ListArtifactStructType(ArtifactStructType):
  """A homogeneous list."""

  def __init__(self, element_type: ArtifactStructType):
    self._element_type = element_type

  def is_type_of(self, struct: ArtifactStruct) -> bool:
    if not isinstance(struct, list):
      return False
    for x in struct:
      if not self._element_type.is_type_of(x):
        return False
    return True

  def __str__(self) -> str:  # pylint: disable=g-ambiguous-str-annotation
    return "List[{}]".format(str(self._element_type))


class _TupleArtifactStructType(ArtifactStructType):
  """A heterogeneous list of fixed length."""

  def __init__(self, element_types: typing.List[ArtifactStructType]):
    self._element_types = element_types

  def is_type_of(self, struct: ArtifactStruct):
    if not isinstance(struct, list):
      return False
    if len(struct) != len(self._element_types):
      return False
    for struct_element, type_element in zip(struct, self._element_types):
      if not type_element.is_type_of(struct_element):
        return False
    return True

  def __str__(self) -> str:  # pylint: disable=g-ambiguous-str-annotation
    return "Tuple[{}]".format(", ".join([str(x) for x in self._element_types]))


class DictArtifactStructType(ArtifactStructType):
  """A type based on a dictionary of ArtifactTypes."""

  def __init__(self,
               dict_type: Dict[Text, ArtifactStructType],
               none_type_not_required=True,
               extra_keys_allowed=False):
    """Creates a dictionary type.

    Args:
      dict_type: the types of the keys.
      none_type_not_required: if a key type can be None, the key is optional.
      extra_keys_allowed: can have artifact structs with unregistered keys.
    """
    self._dict_type = dict_type
    self._none_type_not_required = none_type_not_required
    self._extra_keys_allowed = extra_keys_allowed

  def is_type_of(self, struct: ArtifactStruct):
    if not isinstance(struct, dict):
      return False
    # Check that required keys are present.
    for k, v in self._dict_type.items():
      if k not in struct:
        if self._none_type_not_required:
          # Check if key allowed to be missing (can be None).
          return v.is_type_of(None)
        return False
      if not v.is_type_of(struct[k]):
        return False
    # Check that there are no extra keys.
    if not self._extra_keys_allowed:
      # If you are not allowed to have an extra key, you also can't have a key
      # present with value None.
      for k in struct.keys():
        if k not in self._dict_type.keys():
          return False
    return True

  def __str__(self) -> str:  # pylint: disable=g-ambiguous-str-annotation
    return "Dict[{}]".format(", ".join(
        ["{}:{}".format(k, v) for k, v in self._dict_type.items()]))


# CoercableToType represents objects that can be cast to an ArtifactStructType.
# create_artifact_struct_type() converts CoercableToType to an
# ArtifactStructType.
# For example:
# schema_type = types.create_artifact_type("Schema",
#     version=metadata_store_pb2.INT)
# my_schema_type = create_artifact_struct_type(schema_type)
# schema_type is a metadata_store_pb2.ArtifactType, whereas
# my_schema_type is an ArtifactStructType (a _SimpleArtifactStructType).
# dict_type = create_artifact_struct_type({"schema":schema_type})
# {"schema":schema_type} is CoercableToType, whereas
# dict_type is an ArtifactStructType (a DictArtifactStructType).
#
# Conceptually:
# CoercableToType = Union[Dict[Text, CoercableToType],
#                         typing.List[CoercableToType],
#     None, ArtifactType, ArtifactStructType]
_CoercableToType0 = typing.Union[metadata_store_pb2
                                 .ArtifactType, ArtifactStructType, None]
_CoercableToType1 = typing.Union[Dict[Text, _CoercableToType0], typing
                                 .List[_CoercableToType0], _CoercableToType0]
_CoercableToType2 = typing.Union[Dict[Text, _CoercableToType1], typing
                                 .List[_CoercableToType1], _CoercableToType1]
_CoercableToType3 = typing.Union[Dict[Text, _CoercableToType2], typing
                                 .List[_CoercableToType2], _CoercableToType2]
CoercableToType = _CoercableToType3  # pylint: disable=invalid-name


def create_artifact_struct_type(
    coercable_to_type: CoercableToType) -> ArtifactStructType:
  """Coerces (i.e.

  casts) an object to an ArtiractStructType.

  As a side effect, registers all types in ArtifactStructType
  (if registration is not stopped with stop_registering_types()).

  Args:
    coercable_to_type: a structure or type that can be cast to an
      ArtifactStructType.

  Returns:
    The object as an ArtifactStructType.
  """
  if coercable_to_type is None:
    return _NoneArtifactStructType()
  if isinstance(coercable_to_type, ArtifactStructType):
    return coercable_to_type
  if isinstance(coercable_to_type, metadata_store_pb2.ArtifactType):
    return _SimpleArtifactStructType(coercable_to_type)
  if isinstance(coercable_to_type, dict):
    return DictArtifactStructType({
        k: create_artifact_struct_type(v) for k, v in coercable_to_type.items()
    })
  if isinstance(coercable_to_type, list):
    return _TupleArtifactStructType(
        [create_artifact_struct_type(v) for v in coercable_to_type])
  raise ValueError("Cannot create a type: {}".format(str(coercable_to_type)))


def list_of(element_type: CoercableToType) -> ArtifactStructType:
  """Returns the type of a homogeneous list of artifacts."""
  return _ListArtifactStructType(create_artifact_struct_type(element_type))


def union_of(*args) -> ArtifactStructType:
  """Returns a union of types."""
  return _UnionArtifactStructType(
      [create_artifact_struct_type(x) for x in args])


def intersection_of(*args) -> ArtifactStructType:
  """Returns an intersection of types."""
  return _IntersectionArtifactStructType(
      [create_artifact_struct_type(x) for x in args])


def is_instance(struct: ArtifactStruct, coercable_to_type: CoercableToType):
  """Tests if struct is an instance of coercable_to_type."""
  return create_artifact_struct_type(coercable_to_type).is_type_of(struct)


class ExecutionType(object):
  """Combining the basic ExecutionType with input and output types."""

  def __init__(self, execution_type: metadata_store_pb2.ExecutionType,
               input_type: ArtifactStructType, output_type: ArtifactStructType):
    _registered_types.register_execution_type_as_used(execution_type)
    self.type = execution_type
    self.input_type = input_type
    self.output_type = output_type

  @classmethod
  def create(cls,
             name: typing.Optional[Text] = None,
             properties: typing.Optional[Dict[Text, typing.Any]] = None,
             input_type: CoercableToType = None,
             output_type: CoercableToType = None) -> "ExecutionType":
    """Creates an execution type.

    Args:
      name: the name of the ExecutionType
      properties: a dictionary of properties to metadata_store_pb2.PropertyType
      input_type: the input artifact type of the execution.
      output_type: the output artifact type of the execution.

    Returns:
      an ExecutionType proto.
    """
    if name is None:
      raise ValueError("name is required for create_execution_type()")
    if properties is None:
      raise ValueError("properties is required for create_execution_type()")
    return ExecutionType(
        _create_execution_type_proto(name, properties),
        create_artifact_struct_type(input_type),
        create_artifact_struct_type(output_type))

  def _is_equal(self, execution_type: "ExecutionType") -> bool:
    """For now, just checks name equality."""
    # TODO(martinz): Decide what should be checked here with regard to input
    # structs and output structs.
    return _types_are_equal(self.type, execution_type.type)


def _create_execution_type_proto(name: Text, properties: Dict[Text, typing.Any]
                                ) -> metadata_store_pb2.ExecutionType:
  """Creates an ExecutionType without putting it in the database.

  Args:
    name: the name of the ExecutionType
    properties: a dictionary of properties to metadata_store_pb2.PropertyType

  Returns:
    an ExecutionType proto.
  """
  result = metadata_store_pb2.ExecutionType()
  result.name = name
  for k, v in properties.items():
    result.properties[k] = v
  return result


class Execution(_NodeAndType):
  """A representation of an execution, its type, inputs, and outputs."""

  def __init__(self, execution: metadata_store_pb2.Execution,
               execution_type: ExecutionType, input_struct: ArtifactStruct,
               output_struct: ArtifactStruct):
    self.__dict__["execution"] = execution
    self.__dict__["type"] = execution_type
    self.__dict__["input_struct"] = input_struct
    self.__dict__["output_struct"] = output_struct
    if not self._is_consistent():
      raise ValueError("Execution properties are not internally consistent")

  @classmethod
  def create(cls, execution_type: ExecutionType, input_struct: ArtifactStruct,
             output_struct: ArtifactStruct, **kwargs) -> "Execution":
    result = Execution(metadata_store_pb2.Execution(), execution_type,
                       input_struct, output_struct)
    for k, v in kwargs.items():
      result.set_property(k, v)
    return result

  def is_input_consistent(self) -> bool:
    return self.type.input_type.is_type_of(self.input_struct)

  def is_output_consistent(self) -> bool:
    return self.type.output_type.is_type_of(self.output_struct)

  def is_consistent(self) -> bool:
    return (self._is_consistent() and self.is_output_consistent() and
            self.is_input_consistent())

  def _get_node(self) -> metadata_store_pb2.Execution:
    return self.execution

  def _get_type(self) -> metadata_store_pb2.ExecutionType:
    return self.type.type

  def is_instance_of_type(self, execution_type: ExecutionType) -> bool:
    return self.type._is_equal(execution_type)  # pylint: disable=protected-access


def _register_type_as_used(registered_types, node_type) -> None:
  """Common implementation for registering a type."""
  if not node_type.HasField("name"):
    raise ValueError("ArtifactType must have a name.")
  if node_type.name in registered_types:
    current_type = registered_types[node_type.name]
    if not _types_are_equal(node_type, current_type):
      raise ValueError("Types not equal:{} vs {}".format(
          node_type, registered_types[node_type.name]))
    if current_type.HasField("id") or not node_type.HasField("id"):
      # Don't replace the current_type with node_type if current_type
      # has an ID or node_type does not.
      return
  # Make a copy of any types so that the dictionary has the only copy.
  type_to_register = copy.deepcopy(node_type)
  registered_types[type_to_register.name] = type_to_register


class _RegisteredTypes(object):
  """Dictionary of all registered types by name.

  This class is thread-safe, and provides locking for all methods.
  All objects inside the class (including in the dictionaries are owned
  by the class, and never shared elsewhere.
  """

  def __init__(self):
    self._registered_artifact_types = {
    }  # type: Dict[Text, metadata_store_pb2.ArtifactType]
    self._registered_execution_types = {
    }  # type: Dict[Text, metadata_store_pb2.ExecutionType]
    self._register_active = True
    self._lock = threading.Lock()

  def stop_registering_types(self):
    with self._lock:
      self._register_active = False

  def start_registering_types(self):
    with self._lock:
      self._register_active = True

  def register_artifact_type_as_used(
      self, artifact_type: metadata_store_pb2.ArtifactType) -> None:
    """Registers an artifact type as used.

    See get_all_registered_artifact_types.

    This is called locally whenever a raw ArtifactType is used in this library.

    Args:
      artifact_type: the artifact type that was used.

    Raises:
      ValueError: if two types are locally inconsistent.
    """
    with self._lock:
      if self._register_active:
        _register_type_as_used(self._registered_artifact_types, artifact_type)

  def register_execution_type_as_used(
      self, execution_type: metadata_store_pb2.ExecutionType) -> None:
    """Registers an execution type as used.

    See get_all_registered_execution_types.

    This is called locally whenever a raw ExecutionType is used in this module.

    Args:
      execution_type: the execution type that was used.

    Raises:
      ValueError: if two types are locally inconsistent (same name but not
      equal).
    """
    with self._lock:
      if self._register_active:
        _register_type_as_used(self._registered_execution_types, execution_type)

  def get_all_registered_artifact_types(
      self) -> typing.Sequence[metadata_store_pb2.ArtifactType]:
    """Get all artifact types used in calls in this package."""
    with self._lock:
      type_copies = copy.deepcopy(
          list(self._registered_artifact_types.values()))
    return type_copies

  def get_all_registered_execution_types(
      self) -> typing.Sequence[metadata_store_pb2.ExecutionType]:
    """Get all execution types used in calls in this package."""
    with self._lock:
      type_copies = copy.deepcopy(
          list(self._registered_execution_types.values()))
    return type_copies

  def clear_registered_types(self):
    """For testing only. In general, there should be no need to clear types."""
    with self._lock:
      self._registered_artifact_types.clear()
      self._registered_execution_types.clear()


_registered_types = _RegisteredTypes()


def stop_registering_types():
  """Stop registering types as a side-effect of methods in this module."""
  _registered_types.stop_registering_types()


def start_registering_types():
  """Start registering types as a side-effect of methods in this module."""
  _registered_types.start_registering_types()


def get_all_registered_artifact_types(
) -> typing.Sequence[metadata_store_pb2.ArtifactType]:
  """Get all artifact types used in calls in this package."""
  return _registered_types.get_all_registered_artifact_types()


def get_all_registered_execution_types(
) -> typing.Sequence[metadata_store_pb2.ExecutionType]:
  """Get all execution types used in calls in this package."""
  return _registered_types.get_all_registered_execution_types()


def clear_registered_types():
  """For testing only. In general, there should be no need to clear types."""
  _registered_types.clear_registered_types()
