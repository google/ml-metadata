# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Package Setup script for ML Metadata."""

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True

# Get version from version module.
with open('ml_metadata/version.py') as fp:
  globals_dict = {}
  exec (fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

# TODO(b/121329572): Remove the following comment after we can guarantee the
# required versions of packages through kokoro release workflow.
# Note: In order for the README to be rendered correctly, make sure to have the
# following minimum required versions of the respective packages when building
# and uploading the zip/wheel package to PyPI:
# setuptools >= 38.6.0, wheel >= 0.31.0, twine >= 1.11.0

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

# TODO(martinz): this was copied from TFDV. Come up with more appropriate
# values.
setup(
    name='ml-metadata',
    version=__version__,
    author='Google LLC',
    author_email='tensorflow-extended-dev@googlegroups.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    namespace_packages=[],
    # Make sure to sync the versions of common dependencies (absl-py, numpy,
    # six, and protobuf) with TF.
    install_requires=[
        'absl-py>=0.1.6,<1',

        'protobuf>=3.7,<4',

        'six>=1.4.0,<2',

        # TODO(martinz): Add a method to check if we are using a
        # compatible TF version. If not, fail with a clear error.
        # TODO(martinz): Uncomment this once TF can automatically
        # select between CPU and GPU installation.
        # 'tensorflow>=1.13.1,<2',
    ],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*,<4',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.so']},
    zip_safe=False,
    distclass=BinaryDistribution,
    description='A library for maintaining metadata for artifacts.',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='machine learning metadata tfx',
    url='https://github.com/google/ml-metadata',
    download_url='https://github.com/google/ml-metadata/tags',
    requires=[])
