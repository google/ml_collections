# Copyright 2024 The ML Collections Authors.
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

# pylint: disable=line-too-long
r"""Example of basic DEFINE_flag_dict usage.

To run this example with basic config file:
python define_config_dict_basic.py -- \
  --my_config=ml_collections/config_flags/examples/config.py
  \
  --my_config.field1=8 --my_config.nested.field=2.1 \
  --my_config.tuple='(1, 2, (1, 2))'

To run this example with parameterised config file:
python define_config_dict_basic.py -- \
  --my_config=ml_collections/config_flags/examples/parameterised_config.py:linear
  \
  --my_config.model_config.output_size=256'
"""
# pylint: enable=line-too-long

from absl import app

from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file('my_config')


def main(_):
  print(_CONFIG.value)


if __name__ == '__main__':
  app.run(main)
