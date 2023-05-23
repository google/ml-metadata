# Current Version (not yet released; still in development)

## Major Features and Improvements

*   Support PostgreSQL database type.

## Breaking Changes

## Deprecations

## Bug Fixed and Other Changes

*   Bumped minimum bazel version to 5.3.0.
*   Upgrade Microsoft Visual Studio (MSVC) version to 2017.

# Version 1.13.1

## Major Features and Improvements

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.13.0

## Major Features and Improvements

*   Fix filter query parsing for QueryLineageGraph when using queries with
    aliases.
*   Depends on `numpy~=1.22.0`.

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.7 support.

# Version 1.12.0

## Major Features and Improvements

*   N/A

## Breaking Changes

*   N/A
## Deprecations

*   N/A

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

# Version 1.10.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Adds utils in shared C++ lib to delete ParentContexts by parent_context_id
    and child_context_ids.
*   Depends on `attrs>=20.3,<22`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.9.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.8.0

## Major Features and Improvements

*   From this version we will be releasing python 3.9 wheels.

## Bug Fixes and Other Changes

*   Updates Zlib to 1.2.12.
*   Adds .bazelversion to support Bazelisk.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.7.0

## Major Features and Improvements

*   Upgrade DB schema to V8, including the following changes:
    -   Add constraint on Event table on (artifact_id, execution_id, type). If
        duplicated events are created, already exists error will be returned.
    -   Add index for properties (for both property name and value).

## Bug Fixes and Other Changes

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.6.0

## Major Features and Improvements

*   Apply max_node_size to GetLineageGraphRequest for returned nodes.
*   Skip querying type properties when not needed.

## Bug Fixes and Other Changes

*   Added type information file py.typed according to PEP-561.
*   Depends on `absl-py>=0.9,<2.0.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.5.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Remove INFO log when a type is has empty property definition and not a
    simple type.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.4.0

## Major Features and Improvements

*   Introduce MySQLSourceErrorInfo message to metadata_source proto to support
    adding additional error information. MySQL implementation adds MySQL error
    numbers as part MySQLSourceErrorInfo payload message.
*   Open source GetLineageGraph() to MLMD OSS. MLMD OSS users now can use
    GetLineageGraph() to access MLMD lineage.

## Bug Fixes and Other Changes

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.6 support.

# Version 1.3.0

## Major Features and Improvements

*   Introduces `base_type` field to ArtifactType, ExecutionType and ConteextType
    messages in metadata_store.proto.
*   Modifies PutArtifactType, PutExecutionType and PutTypes APIs to support
    creation of type inheritance link; modifies GetArtifactType(s) and
    GetExecutionType(s) APIs to return `base_type` field values as well.

## Bug Fixes and Other Changes

*   Adds support for PutTypes in the gRPC server.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.2.0

## Major Features and Improvements

*   Open sources declarative nodes filtering with zetaSQL. It is currently
    supported on Linux and MacOS 10.14+.
*   Extends get_artifacts, get_executions, get_contexts APIs with filtering
    capabilities on properties and 1-hop neighborhood nodes.
*   Supports configure GRPC options `grpc.http2.max_ping_strikes` from the
    python client.

## Bug Fixes and Other Changes

*   Introduces `database_name_` field for MySQL MetadataSource implementation to
    enable MySQLMetadataSource to switch to different databases within the same
    SQL server after connection or during reconnection.

## Breaking Changes

*   The minimum required OS version for the macOS is 10.14 now.
*   Bumped the minimum bazel version required to build `ml_metadata` to 3.7.2.

## Deprecations

*   N/A

# Version 1.1.0

## Major Features and Improvements

*   Introduced `skip_db_creation` for MySQL backend. It is useful when db
    creation is handled by an admin process, while the lib users should not
    issue db creation clauses.
*   Adds utils in shared C++ lib to delete Contexts, Artifacts, Executions, and
    their corresponding edges (e.g., Associations, Attributions,
    ParentContexts).
*   Removes TensorFlow dependency in MLMD OSS C++ and TensorFlow bazel WORKSPACE
    dependency in MLMD OSS bazel WORKSPACE.
*   Upgrades MLMD schema version to 7.
    -   Add `byte_value` column for `ArtifactProperty`, `ExecutionProperty`,
        `ContextProperty` for better storing binary property values.
    -   For MySQL backend, modify `string_value` to use `MEDIUMTEXT` instead of
        `TEXT` to persist property value upto 16MB.
    -   Add index `EventPath`.`idx_eventpath_event_id` for Event retrieval
        queries.

## Bug Fixes and Other Changes

*   Depends on `protobuf>=3.13,<4`.
*   Fixes a bug of increasing size of `next_page_token` over pagination calls.
*   Returns InvalidArgumentError for PutParentContexts when creating a cyclic
    parent context.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.0.0

## Major Features and Improvements

*   Implements ParentContext creation and retrieval in MLMD python and gRPC
    APIs.

## Bug Fixes and Other Changes

*   When `reuse_context_if_already_exist` is used for PutExecution, returns
    Aborted instead of AlreadyExists to improve the API contract when there are
    data races to create the same Context for the first time.

## Breaking Changes

*   Predefines a list of types (MLMD SimpleTypes) when connecting a MLMD
    instance. If the types already exist, it is a no-op, otherwise, it creates
    the types. This change may break tests which assume the number of types
    exist in a MLMD instance.

## Deprecations

*   N/A

# Release 0.30.0

## Major Features and Improvements

*   Supports versioned type creation and listing in python APIs for non-backward
    compatible type evolutions.
*   Supports listing nodes by type with version in MLMD APIs.

## Bug Fixes and Other Changes

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   Drops `create_artifact_with_type` from MetadataStore python class.

# Release 0.29.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Uses Pybind11 instead of SWIG to wrap C++ libraries for MLMD python client.
*   Depends on `absl-py>=0.9,<0.13`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Release 0.28.0

## Major Features and Improvements

*   Extends type identifier to be (name, version). The version is optional and
    backward compatible with existing usage.
*   Supports versioned creation and listing of types in gRPC for non-backward
    compatible type evolutions.

## Bug Fixes and Other Changes

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Release 0.27.0

## Major Features and Improvements

*   Introduce `google.protobuf.Struct` as an additional `Value` type for storing
    complex, JSON-objects in MLMD.

## Bug Fixes and Other Changes

*   MLMD wheel for MacOS is now built with minimum OS version 10.9. This will
    improve compatibility for the older MacOS versions.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Release 0.26.0

## Major Features and Improvements

*   Exposes limit and order_by parameters to get_artifacts, get_executions and
    get_contexts API in python MetadataStore client to allow users to specify
    the maximum results to retrieve and the field the results to be ordered by.

## Bug Fixes and Other Changes

*   Adds `attrs` as py client dependency.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Release 0.25.1

## Major Features and Improvements

*   Adds `reuse_context_if_already_exist` option to `put_execution` python API
    to better support concurrent execution publishing with the same new context.
*   Supports pagination and ordering options in GetExecutionsByContext and
    GetArtifactsByContext APIs.

## Bug Fixes and Other Changes

*   N/A

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Release 0.25.0

## Major Features and Improvements

*   Supports MetadataStoreClientConfig options `client_timeout_sec` from the
    python client. The grpc APIs would return DeadlineExceededError when server
    does not respond within `client_timeout_ms`.
*   From this release MLMD will also be hosting nightly packages on
    https://pypi-nightly.tensorflow.org. To install the nightly package use the
    following command:

    ```
    pip install --extra-index-url https://pypi-nightly.tensorflow.org/simple ml-metadata
    ```

    Note: These nightly packages are unstable and breakages are likely to
    happen. The fix could often take a week or more depending on the complexity
    involved for the wheels to be available on the PyPI cloud service. You can
    always use the stable version of MLMD available on PyPI by running the
    command `pip install ml-metadata` .

*   Upgrades MLMD schema version to 6.

    -   Add `ParentType` table for supporting type inheritance.
    -   Add `Type`.`version` column for Type evolution development.
    -   Add `Type`.`idx_type_name` index for type lookup APIs.
    -   Add `Type`.`description` column for capturing static information about
        Type.
    -   Add `ParentContext` table for supporting context parental relationship.
    -   Add `Artifact`.`idx_artifact_uri` for filtering artifacts by uri.
    -   Add `Event`.`idx_event_artifact_id` and `idx_event_execution_id` for
        lineage traversal APIs.
    -   Add indices on `create_time_since_epoch`, `last_update_time_since_epoch`
        for `Artifact`, `Execution` and `Context` for sorted listing queries.

*   Allows omitting stored properties when using `put_artifact_type`,
    `put_execution_type`, `put_context_type`, to help writing forward
    compatibility MLMD type registration calls.

## Bug Fixes and Other Changes

*   Optimizes GetContext*/GetArtifact*/GetExecution* and corresponding List*
    calls to reduce number of backend queries.
*   Documentation fixes for QueryExecutor methods.

## Breaking Changes

*   N/A

## Deprecations

*   Deprecates `all_fields_match` and `can_delete_fields` from python APIs
    `put_artifact_type`, `put_execution_type`, `put_context_type`. In previous
    releases these parameters can only be set with default values, otherwise
    Unimplemented error returns. This change should be no-op for all existing
    users.

# Release 0.24.0

## Major Features and Improvements

*   Improves building wheels from source with `setup.py`.
*   Supports configure GRPC options `max_receive_message_length` from the python
    client.
*   Adds python 3.8 support.

## Bug Fixes and Other Changes

*   Adds `grpcio` as py client dependency.
*   Replaces the C++ MOCK_METHOD`<n>` family of macros with the new MOCK_METHOD
*   Updates node's `last_update_time_since_epoch` when changing
    (custom)properties.
*   Disables incompatible Golang BUILD targets temporarily.
*   Support GetArtifactByTypeAndName, GetExecutionByTypeAndName,
    GetContextByTypeAndName Go API

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated py3.5 support

# Release 0.23.0

## Major Features and Improvements

*   GetArtifacts, GetExecutions and GetContexts now supports pagination and
    ordering results by ID, Create time and last update time fields.

## Bug Fixes and Other Changes

*   Python MetadataStore now exposes get_artifact_by_type_and_name and
    get_execution_by_type_and_name methods.
*   Improves query performance of get_events_by_execution_ids and
    get_events_by_artifact_ids by combing multiple queries.
*   Drops python dependency on tensorflow to make ml-metadata be friendly with
    non-TFX use cases.

## Breaking Changes

*   Python MetadataStore APIs return mlmd errors instead of tensorflow errors.

## Deprecations

*   Note: We plan to remove Python 3.5 support after this release.

# Release 0.22.1

## Major Features and Improvements

*   Uses metadata_store per request for grpc server to improve scalability.

## Bug Fixes and Other Changes

*   Uses Iterable[int] instead Sequence[int] for listing APIs accepting ids.
*   Depends on `tensorflow>=1.15,!=2.0.*,<3`

## Breaking changes

## Deprecations

*   Drops Python 2 support and stops releasing py2 wheels.

# Release 0.22.0

## Major Features and Improvements

*   Upgrades MLMD schema version to 5.
    -   Added state columns to persistent Artifact.state,
        Execution.last_known_state
    -   Added user-given unique name per type column to Artifact and Execution.
    -   Added create_time_since_epoch, last_update_time_since_epoch to all
        Nodes.
*   Provides GetArtifactByTypeAndName and GetExecutionByTypeAndName API to get
    artifact/execution by type and name.
*   Refactors transaction executions using TransactionExecutor class.
*   Supports storing/retrieving Artifact.state and Execution.last_known_state.

## Bug Fixes and Other Changes

*   Returns explicit InvalidArgument for get_artifacts_by_uri from 0.15.x
    clients when using 0.21.0+ server.

## Breaking changes

## Deprecations

# Release 0.21.2

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Updates logging level for python mlmd client.

## Breaking changes

## Deprecations

# Release 0.21.1

## Major Features and Improvements

*   Refactoring MetadataAccessObject to allow for more flexibility.
*   Release a script to generate Python API documentation.

## Bug Fixes and Other Changes

*   GetArtifacts/Executions/Contexts returns OK instead of NotFound to align
    with other listing APIs.
*   Handles mysql stale connection error 2006 by client-side reconnection.
*   Handles mysql innodb deadlock error (1213) and lock timeout (1205) via
    client-side retry.
*   Avoids update node or properties without changes.

## Breaking changes

## Deprecations

# Release 0.21.0

## Major Features and Improvements

*   Adding artifact states.
*   Supporting connection retries to gRPC server.
*   Allowing the Python API put_execution to update or insert related contexts.
*   Adding a new execution state: CANCELED. This indicates an execution being
    canceled.
*   Adding two event types: INTERNAL_INPUT and INTERNAL_OUTPUT indended to be
    used by mlmd powered systems (e.g., orchestrator).
*   Add support to pass migration options as command line parameters to the MLMD
    gRPC server.
*   Adding a new Python API get_context_by_type_and_name to allow querying a
    context by its type and context name at the same time.

## Bug Fixes and Other Changes

*   Refactoring MetadataAccessObject to allow for more flexibility.

## Breaking Changes

*   The Python API put_execution will need an extra input argument to pass in
    contexts and return updated context_ids. Users using the old API could pass
    in None or an empty list as context and add another variable to hold the
    returned context_ids to migrate.

## Deprecations

# Release 0.15.2

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Passes bytes instead of string to grpc.ssl_channel_credentials.
*   Align GRPC python client stub error code with swig client error code.
*   Add verify_server_cert support to MySQL source SSL options.

## Breaking Changes

## Deprecations

# Release 0.15.1

## Major Features and Improvements

*   Add migration options to gRPC MetadataStoreServerConfig.
*   Disable auto schema migration by default during connection. The user needs
    to explicitly enable it when connecting an older database.
*   Support SSL options when using MySQL metadata source.

## Bug Fixes and Other Changes

*   Fixes MySQL errors with concurrent connection to an empty database. Now,
    MLMD returns Aborted when concurrent connection error happens and the caller
    can retry appropriately.

## Breaking Changes

## Deprecations

*   Deprecates proto field MigrationOptions.disable_upgrade_migration.
*   Deprecates `disable_upgrade_migration` in python MetadataStore constructor.

# Release 0.15.0

## Major Features and Improvements

*   Add Dockerfile for building a containerized version of the MLMD gRPC server.
*   Add support for connecting to a MYSQL Metadata Source via Unix sockets.
*   Add support to pass mysql connection configuration as command line
    parameters to the MLMD gRPC server.
*   Provides the ability for metadata_store.py to communicate with the MLMD gRPC
    server, as an alternative to connecting directly with a database.
*   Supports Sqlite for Windows and adds scripts to build wheels for python3 in
    Windows.
*   Provides GetContextTypes to list all Context Types.
*   MLMD ConnectionConfig provides an option to disable an automatic upgrade.
*   Supports downgrade of the database schema version to older versions.

## Bug Fixes and Other Changes

*   Depended on `tensorflow>=1.15,<3`
    *   Starting from 1.15, package `tensorflow` comes with GPU support. Users
        won't need to choose between `tensorflow` and `tensorflow-gpu`.
    *   Caveat: `tensorflow` 2.0.0 is an exception and does not have GPU
        support. If `tensorflow-gpu` 2.0.0 is installed before installing
        `ml_metadata`, it will be replaced with `tensorflow` 2.0.0. Re-install
        `tensorflow-gpu` 2.0.0 if needed.

## Breaking Changes

## Deprecations

# Release 0.14.0

## Major Features and Improvements

*   Add Context and ContextType to MLMD data model, which are used for capturing
    grouping concepts (e.g., Project, Pipeline, Owner, etc) of Artifacts and
    Executions.
*   Add CACHED state to Execution state enum to model an execution that is
    skipped due to cached results.
*   Add the ability to list all instances of ArtifactType and ExecutionType.
*   Support Type update and enforce backward compatibility.
*   Support atomic creation and publishing of an execution.
*   Support building a manylinux2010 compliant wheel in docker so that it works
    in other linux OS outside of Ubuntu.
*   Provide MLMD migration scheme to migrate out-of-date MLMD instance.
*   Support creating and querying ContextType.
*   Support creating, updating and querying Context instances.
*   Support grouping artifact and executions into contexts.
*   Support querying related artifacts and executions through contexts.

## Bug Fixes and Other Changes

## Breaking changes

## Deprecations

# Release 0.13.2

## Major Features and Improvements

*   Established ML Metadata as a standalone package.
*   Provides a way to store information about how each artifact (e.g. file) was
    generated.
*   Provides tools for determining provenance.

## Bug Fixes and Other Changes

## Breaking changes

## Deprecations
