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

import os
import platform
import shutil
import subprocess

import setuptools
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution
# pylint: disable=g-bad-import-order
# It is recommended to import setuptools prior to importing distutils to avoid
# using legacy behavior from distutils.
# https://setuptools.readthedocs.io/en/latest/history.html#v48-0-0
from distutils.command import build
# pylint: enable=g-bad-import-order


class _BuildCommand(build.build):
  """Build everything that is needed to install.

  This overrides the original distutils "build" command to to run bazel_build
  command before any sub_commands.

  build command is also invoked from bdist_wheel and install command, therefore
  this implementation covers the following commands:
    - pip install . (which invokes bdist_wheel)
    - python setup.py install (which invokes install command)
    - python setup.py bdist_wheel (which invokes bdist_wheel command)
  """

  def _build_cc_extensions(self):
    return True

  # Add "bazel_build" command as the first sub_command of "build". Each
  # sub_command of "build" (e.g. "build_py", "build_ext", etc.) is executed
  # sequentially when running a "build" command, if the second item in the tuple
  # (predicate method) is evaluated to true.
  sub_commands = [
      ('bazel_build', _build_cc_extensions),
  ] + build.build.sub_commands


# MLMD is not a purelib. However because of the extension module is not built
# by setuptools, it will be incorrectly treated as a purelib. The following
# works around that bug.
class _InstallPlatlibCommand(install):

  def finalize_options(self):
    install.finalize_options(self)
    self.install_lib = self.install_platlib


class _BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def is_pure(self):
    return False

  def has_ext_modules(self):
    return True


class _BazelBuildCommand(setuptools.Command):
  """Build Bazel artifacts and move generated files."""

  def initialize_options(self):
    pass

  def finalize_options(self):
    self._bazel_cmd = shutil.which('bazel')
    if not self._bazel_cmd:
      raise RuntimeError(
          'Could not find "bazel" binary. Please visit '
          'https://docs.bazel.build/versions/master/install.html for '
          'installation instruction.')
    self._additional_build_options = []
    if platform.system() == 'Darwin':
      # This flag determines the platform qualifier of the macos wheel.
      if platform.machine() == 'arm64':
        self._additional_build_options = ['--macos_minimum_os=11.0',
                                          '--config=macos_arm64']
      else:
        self._additional_build_options = ['--macos_minimum_os=10.14']

      if 'ICONV_LIBRARIES' in os.environ:
        self._additional_build_options.append(
            '--action_env=CMAKE_ICONV_FLAG=-DICONV_LIBRARIES=' +
            os.environ['ICONV_LIBRARIES'])

  def run(self):
    subprocess.check_call(
        [self._bazel_cmd, 'run',
         '--compilation_mode', 'opt',
         '--define', 'grpc_no_ares=true',
         '--verbose_failures',
         *self._additional_build_options,
         '//ml_metadata:move_generated_files'],
        # Bazel should be invoked in a directory containing bazel WORKSPACE
        # file, which is the root directory.
        cwd=os.path.dirname(os.path.realpath(__file__)),)

    # explicitly call shutdown to free up resources.
    subprocess.check_call([self._bazel_cmd, "shutdown"])

# Get version from version module.
with open('ml_metadata/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setup(
    name='ml-metadata',
    version=__version__,
    author='Google LLC',
    author_email='tensorflow-extended-dev@googlegroups.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3 :: Only',
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
        'absl-py>=0.9,<2.0.0',
        'attrs>=20.3,<24',
        'grpcio>=1.8.6,<2',
        'protobuf>=3.13,<4',
        'six>=1.10,<2',
    ],
    python_requires='>=3.9,<4',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.so', '*.pyd']},
    zip_safe=False,
    distclass=_BinaryDistribution,
    description='A library for maintaining metadata for artifacts.',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='machine learning metadata tfx',
    url='https://github.com/google/ml-metadata',
    download_url='https://github.com/google/ml-metadata/tags',
    requires=[],
    cmdclass={
        'install': _InstallPlatlibCommand,
        'build': _BuildCommand,
        'bazel_build': _BazelBuildCommand,
    },
)
