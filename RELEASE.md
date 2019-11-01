# Current version (not yet released; still in development)

## Major Features and Improvements

## Bug Fixes and Other Changes

## Breaking Changes

## Deprecations

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
  * Starting from 1.15, package
    `tensorflow` comes with GPU support. Users won't need to choose between
    `tensorflow` and `tensorflow-gpu`.
  * Caveat: `tensorflow` 2.0.0 is an exception and does not have GPU
    support. If `tensorflow-gpu` 2.0.0 is installed before installing
    `ml_metadata`, it will be replaced with `tensorflow` 2.0.0.
    Re-install `tensorflow-gpu` 2.0.0 if needed.

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
