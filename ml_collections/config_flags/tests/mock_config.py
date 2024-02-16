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

"""Placeholder Config file."""

import copy

from ml_collections.config_flags.tests import spork


class TestConfig(object):
  """Default Test config value."""

  def __init__(self):
    self.integer = 23
    self.float = 2.34
    self.string = 'james'
    self.bool = True
    self.dict = {
        'integer': 1,
        'float': 3.14,
        'string': 'mark',
        'bool': False,
        'dict': {
            'float': 5.
        },
        'list': [1, 2, [3]]
    }
    self.list = [1, 2, [3]]
    self.tuple = (1, 2, (3,))
    self.tuple_with_spaces = (1, 2, (3,))
    self.enum = spork.SporkType.SPOON

  @property
  def readonly_field(self):
    return 42

  def __repr__(self):
    return str(self.__dict__)


def get_config():

  config = TestConfig()
  config.object = TestConfig()
  config.object_reference = config.object
  config.object_copy = copy.deepcopy(config.object)

  return config
