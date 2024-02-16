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

"""Example of basic ConfigDict usage.

This example shows the most basic usage of ConfigDict, including type safety.
For examples of more features, see example_advanced.
"""

from absl import app
from ml_collections import config_dict


def main(_):
  cfg = config_dict.ConfigDict()
  cfg.float_field = 12.6
  cfg.integer_field = 123
  cfg.another_integer_field = 234
  cfg.nested = config_dict.ConfigDict()
  cfg.nested.string_field = 'tom'

  print(cfg.integer_field)  # Prints 123.
  print(cfg['integer_field'])  # Prints 123 as well.

  try:
    cfg.integer_field = 'tom'  # Raises TypeError as this field is an integer.
  except TypeError as e:
    print(e)

  cfg.float_field = 12  # Works: `int` types can be assigned to `float`.
  cfg.nested.string_field = u'bob'  # `String` fields can store Unicode strings.

  print(cfg)


if __name__ == '__main__':
  app.run(main)
