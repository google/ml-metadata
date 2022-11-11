# Version 1.11.0

## Major Features and Improvements

*   Introduce methods to Create and Update Artifacts, Executions and Contexts
    with custom create and update timestamp.
*   Introduce option to always update node's `last_update_time_since_epoch` even
    if update request matches stored node. ## Bug Fixes and Other Changes
*   Filter support for list Context with Artifact alias `artifacts_0` and
    Execution alias `executions_0`.
*   Enclose `FilterQueryBuilder::GetWhereClause()` return value in parentheses
    to ensure filter query will be evaluated in the correct order.
*   Upgrade SQLite version to 3.39.2 to support more advanced SQL statements,
    e.g. using tuples with IN operator.
*   Adds `external_id` for Type, Artifact, Execution and Context to store unique
    string ids from other systems.
*   Implements a fat client that supports v7, v8 and v9 schema for MLMD.
*   Upgrades MLMD schema version to 10.
    -   Add `proto_value` and `bool_value` columns for `ArtifactProperty`,
        `ExecutionProperty`, `ContextProperty`. The `proto_value` columns store
        protocol buffer types (https://developers.google.com/protocol-buffers)
*   Implement `UpsertTypes()` with batch queries in metadata store.

## Breaking Changes

*   N/A


## Deprecations

*   N/A

