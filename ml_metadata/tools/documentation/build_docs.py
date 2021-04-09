# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Script to generate api_docs for MLMD.

The script needs to be run under Python3.

The doc generator can be installed with:

```
$> pip3 install git+https://github.com/tensorflow/docs
```

To run from it on the mlmd pip package:
```
python3 ml_metadata/tools/documentation/build_docs.py --output_dir=/tmp/mlmd
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import ml_metadata.metadata_store as mlmd
from ml_metadata.metadata_store import pywrap

flags.DEFINE_string('output_dir', '/tmp/mlmd_api', 'Where to output the docs')
flags.DEFINE_string(
    'code_url_prefix',
    'https://github.com/google/ml-metadata/tree/master/ml_metadata/metadata_store',
    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files')

flags.DEFINE_string('site_path', 'mlmd/api_docs/python',
                    'Path prefix in the _toc.yaml')

FLAGS = flags.FLAGS

suppress_docs_for = [
    absolute_import,
    division,
    print_function,
    pywrap,
]


def main(args):
  if args[1:]:
    raise ValueError('Unrecognized command line args', args[1:])

  for obj in suppress_docs_for:
    doc_controls.do_not_generate_docs(obj)

  doc_generator = generate_lib.DocGenerator(
      root_title='ML Metadata',
      py_modules=[('mlmd', mlmd)],
      base_dir=os.path.dirname(mlmd.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      private_map={},
      callbacks=[
          # This filters out objects not defined in the current module or its
          # sub-modules.
          public_api.local_definitions_filter
      ])

  doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
