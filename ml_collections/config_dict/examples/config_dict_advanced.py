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

"""Example of ConfigDict usage.

This example includes loading a ConfigDict in FLAGS, locking it, type
safety, iteration over fields, checking for a particular field, unpacking with
`**`, and loading dictionary from string representation.
"""

from absl import app
from ml_collections import config_flags
import yaml

_CONFIG = config_flags.DEFINE_config_file(
    'my_config',
    default='ml_collections/config_dict/examples/config.py')


def hello_function(string, **unused_kwargs):
  return 'Hello {}'.format(string)


def print_section(name):
  print()
  print()
  print('-' * len(name))
  print(name.upper())
  print('-' * len(name))
  print()


def main(_):
  # Config is already loaded in FLAGS.my_config due to the logic hidden
  # in app.run().
  config = _CONFIG.value

  print_section('Printing config.')
  print(config)

  # Config is of our type ConfigDict.
  print('Type of the config {}'.format(type(config)))

  # By default it is locked, thus you cannot add new fields.
  # This prevents you from misspelling your attribute name.
  print_section('Locking.')
  print('config.is_locked={}'.format(config.is_locked))
  try:
    config.object.new_field = -3
  except AttributeError as e:
    print(e)

  # There is also "did you mean" feature!
  try:
    config.object.floet = -3.
  except AttributeError as e:
    print(e)

  # However if you want to modify it you can always unlock.
  print_section('Unlocking.')
  with config.unlocked():
    config.object.new_field = -3
    print('config.object.new_field={}'.format(config.object.new_field))

  # By default config is also type-safe, so you cannot change the type of any
  # field.
  print_section('Type safety.')
  try:
    config.float = 'jerry'
  except TypeError as e:
    print(e)
  config.float = -1.2
  print('config.float={}'.format(config.float))

  # NoneType is ignored by type safety and can both override and be overridden.
  config.float = None
  config.float = -1.2

  # You can temporarly turn type safety off.
  with config.ignore_type():
    config.float = 'tom'
    print('config.float={}'.format(config.float))
    config.float = 2.3
    print('config.float={}'.format(config.float))

  # You can use ConfigDict as a regular dict in many typical use-cases:
  # Iteration over fields:
  print_section('Iteration over fields.')
  for field in config:
    print('config has field "{}"'.format(field))

  # Checking if it contains a particular field using the "in" command.
  print_section('Checking for a particular field.')
  for field in ('float', 'non_existing'):
    if field in config:
      print('"{}" is in config'.format(field))
    else:
      print('"{}" is not in config'.format(field))

  # Using ** unrolling to pass the config to a function as named arguments.
  print_section('Unpacking with **')
  print(hello_function(**config))

  # You can even load a dictionary (notice it is not ConfigDict anymore) from
  # a yaml string representation of ConfigDict.
  # Note: __repr__ (not __str__) is the recommended representation, as it
  # preserves FieldReferences and placeholders.
  print_section('Loading dictionary from string representation.')
  dictionary = yaml.load(repr(config), yaml.UnsafeLoader)
  print('dict["object_reference"]["dict"]["dict"]["float"]={}'.format(
      dictionary['object_reference']['dict']['dict']['float']))


if __name__ == '__main__':
  app.run(main)
