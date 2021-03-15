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

"""Tests for config_flags used in conjunction with DEFINE_config_dataclass."""

import shlex
import sys
from typing import Mapping, Optional, Sequence

from absl import flags
from absl.testing import absltest
import dataclasses
from ml_collections import config_flags


#####
# Simple dummy configuration classes.
@dataclasses.dataclass
class MyModelConfig:
  foo: int
  bar: Sequence[str]
  baz: Optional[Mapping[str, str]] = None


@dataclasses.dataclass
class MyConfig:
  my_model: MyModelConfig
  baseline_model: MyModelConfig


_CONFIG = MyConfig(
    my_model=MyModelConfig(
        foo=3,
        bar=['a', 'b'],
        baz={'foo': 'bar'},
    ),
    baseline_model=MyModelConfig(
        foo=55,
        bar=['c', 'd'],
    ),
)

# Define the flag.
_CONFIG_FLAG = config_flags.DEFINE_config_dataclass('config', _CONFIG)


class TypedConfigFlagsTest(absltest.TestCase):

  def test_instance(self):
    config = _CONFIG_FLAG.value
    self.assertIsInstance(config, MyConfig)
    self.assertEqual(config.my_model, _CONFIG.my_model)
    self.assertEqual(_CONFIG, config)

  def test_flag_overrides(self):

    # Set up some flag overrides.
    old_argv = list(sys.argv)
    sys.argv = shlex.split(
        './program foo.py --test_config.baseline_model.foo=99')
    flag_values = flags.FlagValues()

    # Define a config dataclass flag.
    test_config = config_flags.DEFINE_config_dataclass(
        'test_config', _CONFIG, flag_values=flag_values)

    # Inject the flag overrides.
    flag_values(sys.argv)
    sys.argv = old_argv

    # Did the value get overridden?
    self.assertEqual(test_config.value.baseline_model.foo, 99)

if __name__ == '__main__':
  absltest.main()
