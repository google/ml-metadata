# Current version (not yet released; still in development)

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

## Bug Fixes and Other Changes

## Breaking changes

## Deprecations

# Version 0.13.2

## Major Features and Improvements

*   Established ML Metadata as a standalone package.
*   Provides a way to store information about how each artifact (e.g. file) was
    generated.
*   Provides tools for determining provenance.

## Bug Fixes and Other Changes

## Breaking changes

## Deprecations
