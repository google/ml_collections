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

"""Example of placeholder fields in a ConfigDict.

This example shows how ConfigDict placeholder fields work. For a more complete
example of ConfigDict features, see example_advanced.
"""

from absl import app
from ml_collections import config_dict


def main(_):
  placeholder = config_dict.FieldReference(0)
  cfg = config_dict.ConfigDict()
  cfg.placeholder = placeholder
  cfg.optional = config_dict.FieldReference(0, field_type=int)
  cfg.nested = config_dict.ConfigDict()
  cfg.nested.placeholder = placeholder

  try:
    cfg.optional = 'tom'  # Raises Type error as this field is an integer.
  except TypeError as e:
    print(e)

  cfg.optional = 1555  # Works fine.
  cfg.placeholder = 1  # Changes the value of both placeholder and
  # nested.placeholder fields.

  # Note that the indirection provided by FieldReferences will be lost if
  # accessed through a ConfigDict:
  placeholder = config_dict.FieldReference(0)
  cfg.field1 = placeholder
  cfg.field2 = placeholder  # This field will be tied to cfg.field1.
  cfg.field3 = cfg.field1  # This will just be an int field initialized to 0.

  print(cfg)


if __name__ == '__main__':
  app.run()
