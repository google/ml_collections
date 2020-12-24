# Copyright 2021 The ML Collections Authors.
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

# Lint as: python 3
"""Defines a parameterized method which returns a config depending on input."""

import ml_collections


def get_config(config_string):
  """Return an instance of ConfigDict depending on `config_string`."""
  possible_structures = {
      'linear':
          ml_collections.ConfigDict({
              'model_constructor': 'snt.Linear',
              'model_config': ml_collections.ConfigDict({
                  'output_size': 42,
              })
          }),
      'lstm':
          ml_collections.ConfigDict({
              'model_constructor': 'snt.LSTM',
              'model_config': ml_collections.ConfigDict({
                  'hidden_size': 108,
              })
          })
  }

  return possible_structures[config_string]
