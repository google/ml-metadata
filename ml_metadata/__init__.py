# Copyright 2019 Google LLC
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
"""Init module for ML Metadata."""


# pylint: disable=g-import-not-at-top
try:
  from ml_metadata import proto

  # Import metadata_store API.
  from ml_metadata.metadata_store import downgrade_schema
  from ml_metadata.metadata_store import ListOptions
  from ml_metadata.metadata_store import MetadataStore
  from ml_metadata.metadata_store import OrderByField

  # Import version string.
  from ml_metadata.version import __version__

except ImportError as err:
  import sys
  sys.stderr.write('Error importing: {}'.format(err))
# pylint: enable=g-import-not-at-top
