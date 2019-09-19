# Current version (not yet released; still in development)

## Major Features and Improvements

*   Add Dockerfile for building a containerized version of the MLMD gRPC server.
*   Add support for connecting to a MYSQL Metadata Source via Unix sockets.
*   Add support to pass mysql connection configuration as command line
    parameters to the MLMD gRPC server.
*   Provides the ability for metadata_store.py to communicate with the MLMD gRPC
    server, as an alternative to connecting directly with a database.


## Bug Fixes and Other Changes

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
