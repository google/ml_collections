# Copyright 2023 The ML Collections Authors.
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

import copy
import dataclasses
import functools
import sys
from typing import Mapping, Optional, Sequence, Tuple, Union
import unittest

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
  qux: Optional[int] = None
  bax: float = 1
  boj: Tuple[int, ...] = ()


class ParserForCustomConfig(flags.ArgumentParser):
  def __init__(self, delta=1):
    self.delta = delta

  def parse(self, value):
    if isinstance(value, CustomParserConfig):
      return value
    return CustomParserConfig(i=int(value), j=int(value) + self.delta)


@dataclasses.dataclass
@config_flags.register_flag_parser(parser=ParserForCustomConfig())
class CustomParserConfig():
  i: int
  j: int = 1


@dataclasses.dataclass
class MyConfig:
  my_model: MyModelConfig
  baseline_model: MyModelConfig
  custom: CustomParserConfig = dataclasses.field(
      default_factory=lambda: CustomParserConfig(0))


@dataclasses.dataclass
class SubConfig:
  model: Optional[MyModelConfig] = dataclasses.field(
      default_factory=lambda: MyModelConfig(foo=0, bar=['1']))


@dataclasses.dataclass
class ConfigWithOptionalNestedField:
  sub: Optional[SubConfig] = None
  non_optional: SubConfig = dataclasses.field(
      default_factory=SubConfig)

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
  cfg = _CONFIG
  return dataclasses.replace(
      cfg,
      my_model=dataclasses.replace(cfg.my_model, foo=int(value)))


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

  def test_flag_config_dataclass_optional(self):
    result = test_flags(_CONFIG, '.baseline_model.qux=10')
    self.assertEqual(result.baseline_model.qux, 10)
    self.assertIsInstance(result.baseline_model.qux, int)
    self.assertIsNone(result.my_model.qux)

  def test_custom_flag_parsing_shared_default(self):
    result = test_flags(_CONFIG, '.baseline_model.foo=324')
    result1 = test_flags(_CONFIG, '.baseline_model.foo=123')
    # Here we verify that despite using _CONFIG as shared default for
    # result and result1, the final values are not in fact shared.
    self.assertEqual(result.baseline_model.foo, 324)
    self.assertEqual(result1.baseline_model.foo, 123)
    self.assertEqual(_CONFIG.baseline_model.foo, 55)

  def test_custom_flag_parsing_parser_override(self):
    config_flags.register_flag_parser_for_type(
        CustomParserConfig, ParserForCustomConfig(2))
    result = test_flags(_CONFIG, '.custom=10')
    self.assertEqual(result.custom.i, 10)
    self.assertEqual(result.custom.j, 12)

    # Restore old parser.
    config_flags.register_flag_parser_for_type(
        CustomParserConfig, ParserForCustomConfig())

  @unittest.skipIf(sys.version_info[:2] < (3, 10), 'Need 3.10 to test | syntax')
  def test_pipe_syntax(self):
    @dataclasses.dataclass
    class PipeConfig:
      foo: int | None = None

    result = test_flags(PipeConfig(), '.foo=32')
    self.assertEqual(result.foo, 32)

  def test_custom_flag_parsing_override_work(self):
    # Overrides still work.
    result = test_flags(_CONFIG, '.custom.i=10')
    self.assertEqual(result.custom.i, 10)
    self.assertEqual(result.custom.j, 1)

  def test_optional_nested_fields(self):
    with self.assertRaises(ValueError):
      # Implicit creation not allowed.
      test_flags(ConfigWithOptionalNestedField(), '.sub.model.foo=12')

    # Explicit creation works.
    result = test_flags(ConfigWithOptionalNestedField(), '.sub=build',
                        '.sub.model.foo=12')
    self.assertEqual(result.sub.model.foo, 12)

    # Default initialization support.
    result = test_flags(ConfigWithOptionalNestedField(), '.sub=build')
    self.assertEqual(result.sub.model.foo, 0)

    # Using default value (None).
    result = test_flags(ConfigWithOptionalNestedField())
    self.assertIsNone(result.sub)

    with self.assertRaises(config_flag_lib.FlagOrderError):
      # Don't allow accidental overwrites.
      test_flags(ConfigWithOptionalNestedField(), '.sub.model.foo=12',
                 '.sub=build')

  def test_set_to_none_dataclass_fields(self):
    result = test_flags(ConfigWithOptionalNestedField(), '.sub=build',
                        '.sub.model=none')
    self.assertIsNone(result.sub.model, None)

    with self.assertRaises(KeyError):
      # Parent field is set to None (from not None default value),
      # so this is not a valid set of flags.
      test_flags(ConfigWithOptionalNestedField(),
                 '.sub=build', '.sub.model=none', '.sub.model.foo=12')

    with self.assertRaises(KeyError):
      # Parent field is explicitly set to None (with None default value),
      # so this is not a valid set of flags.
      test_flags(ConfigWithOptionalNestedField(),
                 '.sub=none', '.sub.model.foo=12')

  def test_set_none_non_optional_dataclass_fields(self):
    with self.assertRaises(flags.IllegalFlagValueError):
      # Field is not marked as optional so it can't be set to None.
      test_flags(ConfigWithOptionalNestedField(), '.non_optional=None')

  def test_no_default_initializer(self):
    with self.assertRaises(flags.IllegalFlagValueError):
      test_flags(ConfigWithOptionalNestedField(), '.sub=1', '.sub.model=1')

  def test_custom_flag_parser_invoked(self):
    # custom parser gets invoked
    result = test_flags(_CONFIG, '.custom=10')
    self.assertEqual(result.custom.i, 10)
    self.assertEqual(result.custom.j, 11)

  def test_custom_flag_parser_invoked_overrides_applied(self):
    result = test_flags(_CONFIG, '.custom=15', '.custom.i=11')
    # Override applied successfully
    self.assertEqual(result.custom.i, 11)
    self.assertEqual(result.custom.j, 16)

  def test_custom_flag_application_order(self):
    # Disallow for later value to override the earlier value.
    with self.assertRaises(config_flag_lib.FlagOrderError):
      test_flags(_CONFIG, '.custom.i=11', '.custom=15')

  def test_flag_config_dataclass_type_mismatch(self):
    result = test_flags(_CONFIG, '.my_model.bax=10')
    self.assertIsInstance(result.my_model.bax, float)
    # We can't do anything when the value isn't overridden.
    self.assertIsInstance(result.baseline_model.bax, int)
    self.assertRaises(
        flags.IllegalFlagValueError,
        functools.partial(test_flags, _CONFIG, '.my_model.bax=string'))

  def test_illegal_dataclass_field_type(self):

    @dataclasses.dataclass
    class Config:
      field: Union[int, float] = 3

    self.assertRaises(TypeError,
                      functools.partial(test_flags, Config(), '.field=1'))

  def test_spurious_dataclass_field(self):

    @dataclasses.dataclass
    class Config:
      field: int = 3
    cfg = Config()
    cfg.extra = 'test'

    self.assertRaises(KeyError, functools.partial(test_flags, cfg, '.extra=hi'))

  def test_nested_dataclass(self):

    @dataclasses.dataclass
    class Parent:
      field: int = 3

    @dataclasses.dataclass
    class Child(Parent):
      other: int = 4

    self.assertEqual(test_flags(Child(), '.field=1').field, 1)

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

  def test_flag_config_dataclass_typed_tuple(self):
    result = test_flags(_CONFIG, '.my_model.boj=(0, 1)')
    self.assertEqual(result.my_model.boj, (0, 1))


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
