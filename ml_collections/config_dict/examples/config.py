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

"""Example of a config file using ConfigDict.

The idea of this configuration file is to show a typical use case of ConfigDict,
as well as its limitations. This also exemplifies a self-referencing ConfigDict.
"""

import copy
from ml_collections import config_dict


def _get_flat_config():
  """Helper to generate simple config without references."""

  # The suggested way to create a ConfigDict() is to call its constructor
  # and assign all relevant fields.
  config = config_dict.ConfigDict()

  # In order to add new attributes you can just use . notation, like with any
  # python object. They will be tracked by ConfigDict, and you get type checking
  # etc. for free.
  config.integer = 23
  config.float = 2.34
  config.string = 'james'
  config.bool = True

  # It is possible to assign dictionaries to ConfigDict and they will be
  # automatically and recursively wrapped with ConfigDict. However, make sure
  # that the dict you are assigning does not use internal references/cycles as
  # this is not supported. Instead, create the dicts explicitly as demonstrated
  # by get_config(). But note that this operation makes an element-by-element
  # copy of your original dict.

  # Also note that the recursive wrapping on input dictionaries with ConfigDict
  # does not extend through non-dictionary types (including basic Python types
  # and custom classes). This causes unexpected behavior most commonly if a
  # value is a list of dictionaries, so avoid giving ConfigDict such inputs.
  config.dict = {
      'integer': 1,
      'float': 3.14,
      'string': 'mark',
      'bool': False,
      'dict': {
          'float': 5
      }
  }
  return config


def get_config():
  """Returns a ConfigDict instance describing a complex config.

  Returns:
    A ConfigDict instance with the structure:

    ```
        CONFIG-+-- integer
               |-- float
               |-- string
               |-- bool
               |-- dict +-- integer
               |        |-- float
               |        |-- string
               |        |-- bool
               |        |-- dict +-- float
               |
               |-- object +-- integer
               |          |-- float
               |          |-- string
               |          |-- bool
               |          |-- dict +-- integer
               |                   |-- float
               |                   |-- string
               |                   |-- bool
               |                   |-- dict +-- float
               |
               |-- object_copy +-- integer
               |               |-- float
               |               |-- string
               |               |-- bool
               |               |-- dict +-- integer
               |                        |-- float
               |                        |-- string
               |                        |-- bool
               |                        |-- dict +-- float
               |
               |-- object_reference [reference pointing to CONFIG-+--object]
    ```
  """
  config = _get_flat_config()
  config.object = _get_flat_config()

  # References work just fine, so you will be able to override both
  # values at the same time. The rule is the same as for python objects,
  # everything that is mutable is passed as a reference, thus it will not work
  # with assigning integers or strings, but will work just fine with
  # ConfigDicts.
  # WARNING: Each time you assign a dictionary as a value it will create a new
  # instance of ConfigDict in memory, thus it will be a copy of the original
  # dict and not a reference to the original.
  config.object_reference = config.object

  # ConfigDict supports deepcopying.
  config.object_copy = copy.deepcopy(config.object)

  return config
