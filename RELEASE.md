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

