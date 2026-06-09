# Version 1.21.0

## Major Features and Improvements

*   Added Python 3.12 and Python 3.13 support.

## Breaking Changes

*   Removed Python 3.9 support.
*   Removed ZetaSQL-based `filter_query` functionality. The `filter_query`
    parameter in `ListOptions` that relied on ZetaSQL for declarative filtering
    has been removed in version 1.21.0. Users should migrate to native lookup
    and lineage APIs (such as `get_artifacts_by_uri`, `get_artifacts_by_context`,
    and `get_lineage_subgraph`) and filter retrieved objects directly in Python.

## Deprecations

*   N/A

## Bug Fixed and Other Changes

*   Upgraded Bazel minimum required version to `7.7.0`.
*   Upgraded Protobuf C++ and Python dependencies to `v31.1` / `6.31.1` to align with TensorFlow 2.21.
*   Upgraded Abseil (`com_google_absl`) C++ libraries to stable LTS release `20250127`.
*   Upgraded gRPC (`com_github_grpc_grpc`) to stable release `v1.70.1`.
*   Upgraded `boringssl` dependency to stable commit `16c8d3db1af20fcc04b5190b25242aadcb1fbb30` to support OpenSSL 1.1.0+ APIs in modern gRPC versions.
*   Upgraded rules archives (`rules_swift`, `rules_go`, `bazel_gazelle`, `platforms`) to Bazel 7 compatible versions.
