
# ML Metadata

[![Python](https://img.shields.io/badge/python%7C3.10%7C3.11%7C3.12%7C3.13-blue)](https://github.com/google/ml-metadata)
[![PyPI](https://badge.fury.io/py/ml-metadata.svg)](https://badge.fury.io/py/ml-metadata)

*ML Metadata (MLMD)* is a library for recording and retrieving metadata
associated with ML developer and data scientist workflows.

NOTE: ML Metadata may be backwards incompatible before version 1.0.

## Getting Started

For more background on MLMD and instructions on using it, see the
[getting started guide](https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md)

## Installing from PyPI

The recommended way to install ML Metadata is to use the
[PyPI package](https://pypi.org/project/ml-metadata/):

```bash
pip install ml-metadata
```

Then import the relevant packages:

```python
from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2
```

### Nightly Packages

ML Metadata (MLMD) also hosts nightly packages at
https://pypi-nightly.tensorflow.org on Google Cloud. To install the latest
nightly package, please use the following command:

```bash
pip install --extra-index-url https://pypi-nightly.tensorflow.org/simple ml-metadata
```

## Installing with Docker

This is the recommended way to build ML Metadata under Linux, and is
continuously tested at Google.

Please first install `docker` and `docker-compose` by following the directions:
[docker](https://docs.docker.com/install/);
[docker-compose](https://docs.docker.com/compose/install/).

Then, run the following at the project root:

```bash
DOCKER_SERVICE=manylinux-python${PY_VERSION}
sudo docker compose build ${DOCKER_SERVICE}
sudo docker compose run ${DOCKER_SERVICE}
```

where `PY_VERSION` is one of `{310, 311, 312, 313}`.

A wheel will be produced under `dist/`, and installed as follows:

```shell
pip install dist/*.whl
```

## Installing from source


### 1. Prerequisites

To compile and use ML Metadata, you need to set up some prerequisites.


#### Install Bazel

If Bazel is not installed on your system, install it now by following [these
directions](https://bazel.build/versions/master/docs/install.html).

#### Install cmake
If cmake is not installed on your system, install it now by following [these
directions](https://cmake.org/install/).

### 2. Clone ML Metadata repository

```shell
git clone https://github.com/google/ml-metadata
cd ml-metadata
```

Note that these instructions will install the latest master branch of ML
Metadata. If you want to install a specific branch (such as a release branch),
pass `-b <branchname>` to the `git clone` command.

### 3. Build the pip package

ML Metadata uses Bazel to build the pip package from source:

```shell
python setup.py bdist_wheel
```

You can find the generated `.whl` file in the `dist` subdirectory.

### 4. Install the pip package

```shell
pip install dist/*.whl
```

### 5.(Optional) Build the grpc server

ML Metadata uses Bazel to build the c++ binary from source:

```shell
bazel build -c opt --define grpc_no_ares=true  //ml_metadata/metadata_store:metadata_store_server
```

## Supported platforms

MLMD is built and tested on the following 64-bit operating systems:

*   macOS 10.14.6 (Mojave) or later.
*   Ubuntu 20.04 or later.
*   [DEPRECATED] Windows 10 or later. For a Windows-compatible library, please
    refer to MLMD 1.14.0 or earlier versions.

## Releasing Wheels to PyPI

### Setup (Required for both release methods)

Before releasing, you need to set up the PyPI environment and token once:

**Step 1: Create PyPI environment**

Create a new environment named `pypi` in the GitHub repository:
- Go to https://github.com/google/ml-metadata/settings/environments/new
- Name it `pypi`
- Click "Configure environment"

**Step 2: Add PYPI_TOKEN secret**

Add your PyPI token to the `pypi` environment:
- In the `pypi` environment settings, scroll to "Environment secrets"
- Click "Add secret"
- Name: `PYPI_TOKEN` (use this exact name)
- Value: Your PyPI API token
- Click "Add secret"

**Step 3: Commit and push your release branch**

Ensure your release branch has the correct version set in `ml_metadata/version.py`, then:

```bash
git add ml_metadata/version.py
git commit -m "Prepare release vX.Y.Z"
git push origin your-release-branch
```

### Part 1: Releasing via `workflow_dispatch`

This method allows you to manually trigger a release from any branch without creating a GitHub release.

**Steps** (after completing setup above):

1. Navigate to the GitHub Actions page: https://github.com/google/ml-metadata/actions
2. Find and select the `Build ml-metadata with Conda` workflow: https://github.com/google/ml-metadata/actions/workflows/conda-build.yml
3. Click the "Run workflow" dropdown button.
4. Select your release branch from the dropdown menu.
5. Click "Run workflow".

The workflow will build wheels for all supported Python versions and automatically upload them to PyPI if the token is configured correctly.

### Part 2: Releasing via GitHub Release

This method creates a formal GitHub release with a tag, which automatically triggers the build and upload workflow.

**Steps** (after completing setup above):

1. Go to the Releases tab: https://github.com/google/ml-metadata/releases
2. Click the `Draft new release` button (you'll be redirected to https://github.com/google/ml-metadata/releases/new)
3. Click the `Select tag` button and create a new tag for your release (e.g., `v1.18.0`)
4. Click the `Target` dropdown and select your release branch
5. Fill in the **Release title** and **Release notes** sections
6. Choose the release type:
   - Check `Set as a pre-release` if this is a beta/test release
   - Leave unchecked for `Set as the latest release` for stable releases
7. Click the `Publish release` button
8. Verify the workflow is running by going to the Actions tab: https://github.com/google/ml-metadata/actions/workflows/conda-build.yml

The `Build ml-metadata with Conda` workflow will automatically trigger and build/upload wheels to PyPI if the token is configured correctly.
