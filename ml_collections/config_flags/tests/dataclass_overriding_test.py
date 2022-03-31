# Copyright 2022 The ML Collections Authors.
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
from ml_collections.config_flags import config_flags as config_flag_lib


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


def test_flags(default, *flag_args, parse_fn=None):
  flag_values = flags.FlagValues()
  # DEFINE_config_dataclass accesses sys.argv to build flag list!
  old_args = list(sys.argv)
  sys.argv[:] = ['', *['--test_config' + f for f in flag_args]]
  try:
    result = config_flags.DEFINE_config_dataclass(
        'test_config', default, flag_values=flag_values, parse_fn=parse_fn)
    _, *remaining = flag_values(sys.argv)
    if remaining:
      raise ValueError(f'{remaining}')
    # assert not remaining
    return result.value
  finally:
    sys.argv[:] = old_args


def parse_config_flag(value):
  return dataclasses.replace(
      _CONFIG, my_model=dataclasses.replace(_CONFIG.my_model, foo=int(value)))


class TypedConfigFlagsTest(absltest.TestCase):

  def test_types(self):
    self.assertIsInstance(_TEST_FLAG.value, MyConfig)
    self.assertEqual(_TEST_FLAG.value, _CONFIG)
    self.assertIsInstance(flags.FLAGS['test_flag'].value, MyConfig)
    self.assertIsInstance(flags.FLAGS.test_flag, MyConfig)
    self.assertEqual(flags.FLAGS['test_flag'].value, _CONFIG)
    module_name = __name__ if __name__ != '__main__' else sys.argv[0]
    self.assertEqual(
        flags.FLAGS.find_module_defining_flag('test_flag'), module_name)

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


class DataClassParseFnTest(absltest.TestCase):

  def test_parse_no_custom_value(self):
    result = test_flags(
        _CONFIG, '.baseline_model.foo=10', parse_fn=parse_config_flag)
    self.assertEqual(result.my_model.foo, 3)
    self.assertEqual(result.baseline_model.foo, 10)

  def test_parse_custom_value_applied(self):
    result = test_flags(
        _CONFIG, '=75', '.baseline_model.foo=10', parse_fn=parse_config_flag)
    self.assertEqual(result.my_model.foo, 75)
    self.assertEqual(result.baseline_model.foo, 10)

  def test_parse_out_of_order(self):
    with self.assertRaises(config_flag_lib.FlagOrderError):
      _ = test_flags(
          _CONFIG, '.baseline_model.foo=10', '=75', parse_fn=parse_config_flag)
    # Note: If this is ever supported, add verification that overrides are
    # applied correctly.

  def test_parse_assign_dataclass(self):
    flag_values = flags.FlagValues()

    def always_fail(v):
      raise ValueError()

    result = config_flags.DEFINE_config_dataclass(
        'test_config', _CONFIG, flag_values=flag_values, parse_fn=always_fail)
    flag_values(['program'])
    flag_values['test_config'].value = parse_config_flag('12')
    self.assertEqual(result.value.my_model.foo, 12)

  def test_parse_invalid_custom_value(self):
    with self.assertRaises(flags.IllegalFlagValueError):
      _ = test_flags(
          _CONFIG, '=?', '.baseline_model.foo=10', parse_fn=parse_config_flag)

  def test_parse_overrides_applied(self):
    result = test_flags(
        _CONFIG, '=34', '.my_model.foo=10', parse_fn=parse_config_flag)
    self.assertEqual(result.my_model.foo, 10)

if __name__ == '__main__':
  absltest.main()
