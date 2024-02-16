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

"""Example of FieldReference usage.

This shows how to use FieldReferences for lazy computation.
"""

from absl import app
from ml_collections import config_dict


def lazy_computation():
  """Simple example of lazy computation with `configdict.FieldReference`."""
  ref = config_dict.FieldReference(1)
  print(ref.get())  # Prints 1

  add_ten = ref.get() + 10  # ref.get() is an integer and so is add_ten
  add_ten_lazy = ref + 10  # add_ten_lazy is a FieldReference - NOT an integer

  print(add_ten)  # Prints 11
  print(add_ten_lazy.get())  # Prints 11 because ref's value is 1

  # Addition is lazily computed for FieldReferences so changing ref will change
  # the value that is used to compute add_ten.
  ref.set(5)
  print(add_ten)  # Prints 11
  print(add_ten_lazy.get())  # Prints 15 because ref's value is 5


def change_lazy_computation():
  """Overriding lazily computed values."""
  config = config_dict.ConfigDict()
  config.reference = 1
  config.reference_0 = config.get_ref('reference') + 10
  config.reference_1 = config.get_ref('reference') + 20
  config.reference_1_0 = config.get_ref('reference_1') + 100

  print(config.reference)  # Prints 1.
  print(config.reference_0)  # Prints 11.
  print(config.reference_1)  # Prints 21.
  print(config.reference_1_0)  # Prints 121.

  config.reference_1 = 30

  print(config.reference)  # Prints 1 (unchanged).
  print(config.reference_0)  # Prints 11 (unchanged).
  print(config.reference_1)  # Prints 30.
  print(config.reference_1_0)  # Prints 130.


def create_cycle():
  """Creates a cycle within a ConfigDict."""
  config = config_dict.ConfigDict()
  config.integer_field = 1
  config.bigger_integer_field = config.get_ref('integer_field') + 10

  try:
    # Raises a MutabilityError because setting config.integer_field would
    # cause a cycle.
    config.integer_field = config.get_ref('bigger_integer_field') + 2
  except config_dict.MutabilityError as e:
    print(e)


def lazy_configdict():
  """Example usage of lazy computation with ConfigDict."""
  config = config_dict.ConfigDict()
  config.reference_field = config_dict.FieldReference(1)
  config.integer_field = 2
  config.float_field = 2.5

  # No lazy evaluatuations because we didn't use get_ref()
  config.no_lazy = config.integer_field * config.float_field

  # This will lazily evaluate ONLY config.integer_field
  config.lazy_integer = config.get_ref('integer_field') * config.float_field

  # This will lazily evaluate ONLY config.float_field
  config.lazy_float = config.integer_field * config.get_ref('float_field')

  # This will lazily evaluate BOTH config.integer_field and config.float_Field
  config.lazy_both = (config.get_ref('integer_field') *
                      config.get_ref('float_field'))

  config.integer_field = 3
  print(config.no_lazy)  # Prints 5.0 - It uses integer_field's original value

  print(config.lazy_integer)  # Prints 7.5

  config.float_field = 3.5
  print(config.lazy_float)  # Prints 7.0
  print(config.lazy_both)  # Prints 10.5


def lazy_configdict_advanced():
  """Advanced lazy computation with ConfigDict."""
  # FieldReferences can be used with ConfigDict as well
  config = config_dict.ConfigDict()
  config.float_field = 12.6
  config.integer_field = 123
  config.list_field = [0, 1, 2]

  config.float_multiply_field = config.get_ref('float_field') * 3
  print(config.float_multiply_field)  # Prints 37.8

  config.float_field = 10.0
  print(config.float_multiply_field)  # Prints 30.0

  config.longer_list_field = config.get_ref('list_field') + [3, 4, 5]
  print(config.longer_list_field)  # Prints [0, 1, 2, 3, 4, 5]

  config.list_field = [-1]
  print(config.longer_list_field)  # Prints [-1, 3, 4, 5]

  # Both operands can be references
  config.ref_subtraction = (
      config.get_ref('float_field') - config.get_ref('integer_field'))
  print(config.ref_subtraction)  # Prints -113.0

  config.integer_field = 10
  print(config.ref_subtraction)  # Prints 0.0


def main(argv=()):
  del argv  # Unused.
  lazy_computation()
  lazy_configdict()
  change_lazy_computation()
  create_cycle()
  lazy_configdict_advanced()


if __name__ == '__main__':
  app.run()
