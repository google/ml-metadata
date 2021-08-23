# Version 1.3.0

## Major Features and Improvements

*   Introduces `base_type` field to ArtifactType, ExecutionType and ConteextType
    messages in metadata_store.proto.
*   Modifies PutArtifactType, PutExecutionType and PutTypes APIs to support
    creation of type inheritance link; modifies GetArtifactType(s) and
    GetExecutionType(s) APIs to return `base_type` field values as well.

## Bug Fixes and Other Changes

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A
