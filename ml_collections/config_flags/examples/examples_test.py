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
"""Tests for config_flags examples.

Ensures that from define_config_dict_basic, define_config_file_basic run
successfully.
"""

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from ml_collections.config_flags.examples import define_config_dict_basic
from ml_collections.config_flags.examples import define_config_file_basic

FLAGS = flags.FLAGS


class ConfigDictExamplesTest(absltest.TestCase):

  def test_define_config_dict_basic(self):
    define_config_dict_basic.main([])

  @flagsaver.flagsaver
  def test_define_config_file_basic(self):
    FLAGS.my_config = 'ml_collections/config_flags/examples/config.py'
    define_config_file_basic.main([])

  @flagsaver.flagsaver
  def test_define_config_file_parameterised(self):
    FLAGS.my_config = 'ml_collections/config_flags/examples/parameterised_config.py:linear'
    define_config_file_basic.main([])


if __name__ == '__main__':
  absltest.main()
