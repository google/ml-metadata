# mlmd.proto

ML Metadata proto module.

## Classes

[`class Artifact`][ml_metadata.proto.Artifact]: An artifact represents an input or an output of individual steps in a ML workflow, e.g., a trained model, an input dataset, and evaluation metrics.

[`class ArtifactType`][ml_metadata.proto.ArtifactType]: A user defined type about a collection of artifacts and their properties that are stored in the metadata store.

[`class Association`][ml_metadata.proto.Association]: An association represents the relationship between executions and contexts.

[`class Attribution`][ml_metadata.proto.Attribution]: An attribution represents the relationship between artifacts and contexts.

[`class ConnectionConfig`][ml_metadata.proto.ConnectionConfig]: A connection configuration specifying the persistent backend to be used with MLMD.

[`class Context`][ml_metadata.proto.Context]: A context defines a group of artifacts and/or executions.

[`class ContextType`][ml_metadata.proto.ContextType]: A user defined type about a collection of contexts and their properties that are stored in the metadata store.

[`class Event`][ml_metadata.proto.Event]: An event records the relationship between artifacts and executions.

[`class Execution`][ml_metadata.proto.Execution]: An execution describes a component run or a step in an ML workflow along with its runtime parameters, e.g., a Trainer run, a data transformation step.

[`class ExecutionType`][ml_metadata.proto.ExecutionType]: A user defined type about a collection of executions and their properties that are stored in the metadata store.

[`class FakeDatabaseConfig`][ml_metadata.proto.FakeDatabaseConfig]: An in-memory database configuration for testing purpose.

[`class MetadataStoreClientConfig`][ml_metadata.proto.MetadataStoreClientConfig]: A connection configuration to use a MLMD server as the persistent backend.

[`class MySQLDatabaseConfig`][ml_metadata.proto.MySQLDatabaseConfig]: A connection configuration to use a MySQL db instance as a MLMD backend.

[`class ParentContext`][ml_metadata.proto.ParentContext]: A parental context represents the relationship between contexts.

[`class SqliteMetadataSourceConfig`][ml_metadata.proto.SqliteMetadataSourceConfig]: A connection configuration to use a Sqlite db file as a MLMD backend.
