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
"""ML Metadata proto module."""

# Connection configurations for different deployment.
from ml_metadata.proto import (
    metadata_store_service_pb2,
    metadata_store_service_pb2_grpc,
)

# ML Metadata core data model concepts.
from ml_metadata.proto.metadata_store_pb2 import (
    Artifact,
    ArtifactType,
    Association,
    Attribution,
    ConnectionConfig,
    Context,
    ContextType,
    Event,
    Execution,
    ExecutionType,
    FakeDatabaseConfig,
    MetadataStoreClientConfig,
    MySQLDatabaseConfig,
    ParentContext,
    SqliteMetadataSourceConfig,
)

del metadata_store_service_pb2
del metadata_store_service_pb2_grpc

Artifact.__doc__ = """
An artifact represents an input or an output of individual steps in a ML
workflow, e.g., a trained model, an input dataset, and evaluation metrics.
"""

Execution.__doc__ = """
An execution describes a component run or a step in an ML workflow along with
its runtime parameters, e.g., a Trainer run, a data transformation step.
"""

Context.__doc__ = """
A context defines a group of artifacts and/or executions. It captures the
commonalities of the grouped entities. For example, a project context that
contains all artifacts and executions of a ML workflow, an experiment that
includes the modified trainers and evaluation metrics.
"""

Event.__doc__ = """
An event records the relationship between artifacts and executions. When an
execution happens, events record every artifact that was used by the execution,
and every artifact that was produced. These records allow for lineage tracking
throughout a ML workflow.
"""

Attribution.__doc__ = """
An attribution represents the relationship between artifacts and contexts.
"""

Association.__doc__ = """
An association represents the relationship between executions and contexts.
"""

ParentContext.__doc__ = """
A parental context represents the relationship between contexts.
"""

ArtifactType.__doc__ = """
A user defined type about a collection of artifacts and their properties that
are stored in the metadata store.
"""

ExecutionType.__doc__ = """
A user defined type about a collection of executions and their properties that
are stored in the metadata store.
"""

ContextType.__doc__ = """
A user defined type about a collection of contexts and their properties that
are stored in the metadata store.
"""

ConnectionConfig.__doc__ = """
A connection configuration specifying the persistent backend to be used with
MLMD.
"""

FakeDatabaseConfig.__doc__ = """
An in-memory database configuration for testing purpose.
"""

MySQLDatabaseConfig.__doc__ = """
A connection configuration to use a MySQL db instance as a MLMD backend.
"""

SqliteMetadataSourceConfig.__doc__ = """
A connection configuration to use a Sqlite db file as a MLMD backend.
"""

MetadataStoreClientConfig.__doc__ = """
A connection configuration to use a MLMD server as the persistent backend.
"""

__all__ = [
    "ConnectionConfig",
    "MetadataStoreClientConfig",
    "Artifact",
    "Execution",
    "Context",
    "Event",
    "Attribution",
    "Association",
    "ParentContext",
    "ArtifactType",
    "ExecutionType",
    "ContextType",
    "FakeDatabaseConfig",
    "MySQLDatabaseConfig",
    "SqliteMetadataSourceConfig",
]
