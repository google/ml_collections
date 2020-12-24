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

# Lint as: python3
"""Tests for ConfigDict examples.

Ensures that config_dict_basic, config_dict_initialization, config_dict_lock,
config_dict_placeholder, field_reference, frozen_config_dict run successfully.
"""

from absl.testing import absltest
from absl.testing import parameterized
from ml_collections.config_dict.examples import config_dict_advanced
from ml_collections.config_dict.examples import config_dict_basic
from ml_collections.config_dict.examples import config_dict_initialization
from ml_collections.config_dict.examples import config_dict_lock
from ml_collections.config_dict.examples import config_dict_placeholder
from ml_collections.config_dict.examples import field_reference
from ml_collections.config_dict.examples import frozen_config_dict


class ConfigDictExamplesTest(parameterized.TestCase):

  @parameterized.parameters(config_dict_advanced, config_dict_basic,
                            config_dict_initialization, config_dict_lock,
                            config_dict_placeholder, field_reference,
                            frozen_config_dict)
  def testScriptRuns(self, example_name):
    example_name.main(None)


if __name__ == '__main__':
  absltest.main()
