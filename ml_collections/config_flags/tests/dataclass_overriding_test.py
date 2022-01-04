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

import dataclasses
import sys
from typing import Mapping, Optional, Sequence, Tuple

from absl import flags
from absl.testing import absltest
from ml_collections import config_flags


#####
# Simple dummy configuration classes.
@dataclasses.dataclass
class MyModelConfig:
  foo: int
  bar: Sequence[str]
  baz: Optional[Mapping[str, str]] = None
  buz: Optional[Mapping[Tuple[int, int], str]] = None


@dataclasses.dataclass
class MyConfig:
  my_model: MyModelConfig
  baseline_model: MyModelConfig


_CONFIG = MyConfig(
    my_model=MyModelConfig(
        foo=3,
        bar=['a', 'b'],
        baz={'foo.b': 'bar'},
        buz={(0, 0): 'ZeroZero', (0, 1): 'ZeroOne'}
    ),
    baseline_model=MyModelConfig(
        foo=55,
        bar=['c', 'd'],
    ),
)

_TEST_FLAG = config_flags.DEFINE_config_dataclass('test_flag', _CONFIG,
                                                  'MyConfig data')


def test_flags(default, *flag_args):
  flag_values = flags.FlagValues()
  # DEFINE_config_dataclass accesses sys.argv to build flag list!
  old_args = list(sys.argv)
  sys.argv[:] = ['', *['--test_config' + f for f in flag_args]]
  result = config_flags.DEFINE_config_dataclass(
      'test_config', default, flag_values=flag_values)
  _, *remaining = flag_values(sys.argv)
  sys.argv[:] = old_args
  if remaining:
    raise ValueError(f'{remaining}')
  # assert not remaining
  return result.value


class TypedConfigFlagsTest(absltest.TestCase):

  def test_types(self):
    self.assertIsInstance(_TEST_FLAG.value, MyConfig)
    self.assertEqual(_TEST_FLAG.value, _CONFIG)
    self.assertIsInstance(flags.FLAGS['test_flag'].value, MyConfig)
    self.assertIsInstance(flags.FLAGS.test_flag, MyConfig)
    self.assertEqual(flags.FLAGS['test_flag'].value, _CONFIG)
    self.assertEqual(flags.FLAGS.find_module_defining_flag('test_flag'),
                     __name__ if __name__ != '__main__' else sys.argv[0])

  def test_instance(self):
    config = test_flags(_CONFIG)
    self.assertIsInstance(config, MyConfig)
    self.assertEqual(config.my_model, _CONFIG.my_model)
    self.assertEqual(_CONFIG, config)

  def test_flag_config_dataclass(self):
    result = test_flags(_CONFIG, '.baseline_model.foo=10', '.my_model.foo=7')
    self.assertEqual(result.baseline_model.foo, 10)
    self.assertEqual(result.my_model.foo, 7)

  def test_flag_config_dataclass_string_dict(self):
    result = test_flags(_CONFIG, '.my_model.baz["foo.b"]=rab')
    self.assertEqual(result.my_model.baz['foo.b'], 'rab')

  def test_flag_config_dataclass_tuple_dict(self):
    result = test_flags(_CONFIG, '.my_model.buz[(0,1)]=hello')
    self.assertEqual(result.my_model.buz[(0, 1)], 'hello')


if __name__ == '__main__':
  absltest.main()
