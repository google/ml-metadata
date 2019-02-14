
# ML Metadata

*ML Metadata (MLMD)* is a library for recording and retrieving metadata
associated with ML developer and data scientist workflows.

Caution: ML Metadata may be backwards incompatible before version 1.0.

## Getting Started

For more background on MLMD and instructions on using it, see the
[getting started guide](https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md)

## Installing from PyPI

<!-- TODO: create PyPI repository -->
<!-- TODO: add instructions for installing from PyPI -->

## Installing from source

### 1. Prerequisites

#### Install Python

<!-- TODO: Add instructions for installing Python -->

#### Install Bazel

If Bazel is not installed on your system, install it now by following [these
directions](https://bazel.build/versions/master/docs/install.html).

NOTE: ML Metadata works only with bazel version 0.15.0. Higher bazel versions
are not guaranteed to compile ML Metadata correctly.

### 2. Clone ML Metadata repository

<!-- TODO: create ML Metadata repository -->
```shell
git clone https://github.com/google/ml-metadata
cd ml-metadata
```

Note that these instructions will install the latest master branch of
ML Metadata. If you want to install a specific branch (such as a release
branch), pass `-b <branchname>` to the `git clone` command.

### 3. Build the pip package

ML Metadata uses Bazel to build the pip package from source:

```shell
bazel run -c opt --define grpc_no_ares=true ml_metadata:build_pip_package
```

You can find the generated `.whl` file in the `dist` subdirectory.

### 4. Install the pip package

```shell
pip install dist/*.whl
```

## Supported platforms

ML Metadata works on Python 2.7 or Python 3.

ML Metadata is built and tested on the following 64-bit operating systems:

<!-- TODO:  * macOS 10.12.6 (Sierra) or later. -->
<!-- TODO:  * Ubuntu 14.04 or later. -->

## Dependencies

<!-- TODO: determine dependencies. -->

## Compatible versions

<!-- TODO: determine compatible versions. -->

## Questions

<!-- TODO: setup stackoverflow -->

