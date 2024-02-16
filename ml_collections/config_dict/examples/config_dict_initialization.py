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

"""Example of initialization features and gotchas in a ConfigDict.
"""

import copy

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

  inner_dict = {'list': [1, 2], 'tuple': (1, 2, [3, 4], (5, 6))}
  example_dict = {
      'string': 'tom',
      'int': 2,
      'list': [1, 2],
      'set': {1, 2},
      'tuple': (1, 2),
      'ref': config_dict.FieldReference({'int': 0}),
      'inner_dict_1': inner_dict,
      'inner_dict_2': inner_dict
  }

  print_section('Initializing on dictionary.')
  # ConfigDict can be initialized on example_dict
  example_cd = config_dict.ConfigDict(example_dict)

  # Dictionary fields are also converted to ConfigDict
  print(type(example_cd.inner_dict_1))

  # And the reference structure is preserved
  print(id(example_cd.inner_dict_1) == id(example_cd.inner_dict_2))

  print_section('Initializing on ConfigDict.')

  # ConfigDict can also be initialized on a ConfigDict
  example_cd_cd = config_dict.ConfigDict(example_cd)

  # Yielding the same result:
  print(example_cd == example_cd_cd)

  # Note that the memory addresses are different
  print(id(example_cd) == id(example_cd_cd))

  # The memory addresses of the attributes are not the same because of the
  # FieldReference, which gets removed on the second initialization
  list_to_ids = lambda x: [id(element) for element in x]
  print(
      set(list_to_ids(list(example_cd.values()))) == set(
          list_to_ids(list(example_cd_cd.values()))))

  print_section('Initializing on self-referencing dictionary.')

  # Initialization works on a self-referencing dict
  self_ref_dict = copy.deepcopy(example_dict)
  self_ref_dict['self'] = self_ref_dict
  self_ref_cd = config_dict.ConfigDict(self_ref_dict)

  # And the reference structure is replicated
  print(id(self_ref_cd) == id(self_ref_cd.self))

  print_section('Unexpected initialization behavior.')

  # ConfigDict initialization doesn't look inside lists, so doesn't convert a
  # dict in a list to ConfigDict
  dict_in_list_in_dict = {'list': [{'troublemaker': 0}]}
  dict_in_list_in_dict_cd = config_dict.ConfigDict(dict_in_list_in_dict)
  print(type(dict_in_list_in_dict_cd.list[0]))

  # This can cause the reference structure to not be replicated
  referred_dict = {'key': 'value'}
  bad_reference = {'referred_dict': referred_dict, 'list': [referred_dict]}
  bad_reference_cd = config_dict.ConfigDict(bad_reference)
  print(id(bad_reference_cd.referred_dict) == id(bad_reference_cd.list[0]))


if __name__ == '__main__':
  app.run()
