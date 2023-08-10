# Version 1.14.0

## Major Features and Improvements

*   Support PostgreSQL database type.
*   Support bool_value in (custom_)property filter queries.
*   Add masking support for Artifact / Execution / Context updates
*   Support using enum names in IN operator in filter queries.
*   Support populating ArtifactTypes for GetArtifactByID API.
*   Add GetLineageSubgraph API for efficient lineage tracing.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

## Bug Fixed and Other Changes

*   Bumped minimum bazel version to 5.3.0.
*   Upgrade Microsoft Visual Studio (MSVC) version to 2017.
*   Support filtering by parent/child context id in filter queries.
*   Add batch queries for retrieving attributions/associations by
    artifact/execution ids in query executor layer and deprecate the original
    SelectAttributionByArtifactID and SelectAssociationByExecutionID functions.

