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

"""Config file where `get_config` takes a string argument."""

from ml_collections import config_dict


def get_config(config_string):
  """A config which takes an extra string argument."""
  possible_configs = {
      'type_a': config_dict.ConfigDict({
          'thing_a': 23,
          'thing_b': 42,
      }),
      'type_b': config_dict.ConfigDict({
          'thing_a': 19,
          'thing_c': 65,
      }),
  }
  return possible_configs[config_string]
