# Version 1.15.0

## Major Features and Improvements

*   Add mlmd_resolver as a wrapper later upon metadata_store.
*   Extend GetLineageSubgraph API to support returning `associations` and
    `attributions`.

## Breaking Changes

*   N/A

## Deprecations

*   Deprecate GetLineageGraph API.
*   Deprecate OSS support on Windows OS platform.
*   Deprecated python 3.8 support.
*   Bumped minimum bazel version to 6.1.0.
*   Deprecate types.py support.

## Bug Fixed and Other Changes

*   Depends on `attrs>=20.3,<24`.
*   Depends on `protobuf>=4.25.2,<5` for Python 3.11 and on `protobuf>3.20.3,<5`
    for 3.9 and 3.10.

