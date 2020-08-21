# Current version (not yet released; still in development)

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Adds `grpcio` as py client dependency.
*   Improves building wheels from source with `setup.py`.

## Breaking Changes

## Deprecations

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

*   N/A

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
