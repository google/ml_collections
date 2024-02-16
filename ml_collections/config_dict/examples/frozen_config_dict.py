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

"""Example of basic FrozenConfigDict usage.

This example shows the most basic usage of FrozenConfigDict, highlighting
the differences between FrozenConfigDict and ConfigDict and including
converting between the two.
"""

from absl import app
from ml_collections import config_dict


def print_section(name):
  print()
  print()
  print('-' * len(name))
  print(name.upper())
  print('-' * len(name))
  print()


def main(_):
  print_section('Attribute Types.')
  cfg = config_dict.ConfigDict()
  cfg.int = 1
  cfg.list = [1, 2, 3]
  cfg.tuple = (1, 2, 3)
  cfg.set = {1, 2, 3}
  cfg.frozenset = frozenset({1, 2, 3})
  cfg.dict = {
      'nested_int': 4,
      'nested_list': [4, 5, 6],
      'nested_tuple': ([4], 5, 6),
  }

  print('Types of cfg fields:')
  print('list: ', type(cfg.list))  # List
  print('set: ', type(cfg.set))  # Set
  print('nested_list: ', type(cfg.dict.nested_list))  # List
  print('nested_tuple[0]: ', type(cfg.dict.nested_tuple[0]))  # List

  frozen_cfg = config_dict.FrozenConfigDict(cfg)
  print('\nTypes of FrozenConfigDict(cfg) fields:')
  print('list: ', type(frozen_cfg.list))  # Tuple
  print('set: ', type(frozen_cfg.set))  # Frozenset
  print('nested_list: ', type(frozen_cfg.dict.nested_list))  # Tuple
  print('nested_tuple[0]: ', type(frozen_cfg.dict.nested_tuple[0]))  # Tuple

  cfg_from_frozen = config_dict.ConfigDict(frozen_cfg)
  print('\nTypes of ConfigDict(FrozenConfigDict(cfg)) fields:')
  print('list: ', type(cfg_from_frozen.list))  # List
  print('set: ', type(cfg_from_frozen.set))  # Set
  print('nested_list: ', type(cfg_from_frozen.dict.nested_list))  # List
  print('nested_tuple[0]: ', type(cfg_from_frozen.dict.nested_tuple[0]))  # List

  print('\nCan use FrozenConfigDict.as_configdict() to convert to ConfigDict:')
  print(cfg_from_frozen == frozen_cfg.as_configdict())  # True

  print_section('Immutability.')
  try:
    frozen_cfg.new_field = 1  # Raises AttributeError because of immutability.
  except AttributeError as e:
    print(e)

  print_section('"==" and eq_as_configdict().')
  # FrozenConfigDict.__eq__() is not type-invariant with respect to ConfigDict
  print(frozen_cfg == cfg)  # False
  # FrozenConfigDict.eq_as_configdict() is type-invariant with respect to
  # ConfigDict
  print(frozen_cfg.eq_as_configdict(cfg))  # True
  # .eq_as_congfigdict() is also a method of ConfigDict
  print(cfg.eq_as_configdict(frozen_cfg))  # True


if __name__ == '__main__':
  app.run(main)
