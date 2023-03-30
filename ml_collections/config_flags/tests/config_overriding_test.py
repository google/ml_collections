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

"""Tests for ml_collection.config_flags."""

import copy
import enum
import shlex
import sys

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from ml_collections import config_dict
from ml_collections.config_flags import config_flags
from ml_collections.config_flags.tests import mock_config
from ml_collections.config_flags.tests import spork


_CHECK_TYPES = (int, str, float, bool)

_TEST_DIRECTORY = 'ml_collections/config_flags/tests'
_TEST_CONFIG_FILE = '{}/mock_config.py'.format(_TEST_DIRECTORY)
# Parameters to test that config loading and overriding works with both
# one and two dashes.
_DASH_PARAMETERS = (
    ('WithTwoDashesAndEqual', '--test_config={}'.format(_TEST_CONFIG_FILE)),
    ('WithTwoDashes', '--test_config {}'.format(_TEST_CONFIG_FILE)),
    ('WithOneDashAndEqual', '-test_config={}'.format(_TEST_CONFIG_FILE)),
    ('WithOneDash', '-test_config {}'.format(_TEST_CONFIG_FILE)))

_CONFIGDICT_CONFIG_FILE = '{}/configdict_config.py'.format(_TEST_DIRECTORY)
_IOERROR_CONFIG_FILE = '{}/ioerror_config.py'.format(_TEST_DIRECTORY)
_VALUEERROR_CONFIG_FILE = '{}/valueerror_config.py'.format(_TEST_DIRECTORY)
_TYPEERROR_CONFIG_FILE = '{}/typeerror_config.py'.format(_TEST_DIRECTORY)
_FIELDREFERENCE_CONFIG_FILE = '{}/fieldreference_config.py'.format(
    _TEST_DIRECTORY)
_PARAMETERISED_CONFIG_FILE = '{}/parameterised_config.py'.format(
    _TEST_DIRECTORY)


def _parse_flags(command,
                 default=None,
                 config=None,
                 lock_config=True,
                 required=False,
                 use_sys_argv_override=False):
  """Parses arguments simulating sys.argv or via sys_argv argument."""

  if config is not None and default is not None:
    raise ValueError('If config is supplied a default should not be.')

  # The module shlex is useful here because it splits the input similar to
  # sys.argv. For instance, string arguments are not split by space.
  argv = shlex.split(command)

  # Storing copy of the old sys.argv.
  old_argv = list(sys.argv)
  # Overwriting sys.argv, as sys has a global state it gets propagated.
  if not use_sys_argv_override:
    sys.argv = argv

  # Actual parsing.
  values = flags.FlagValues()
  if config is None:
    config_flags.DEFINE_config_file(
        'test_config',
        default=default,
        flag_values=values,
        lock_config=lock_config,
        sys_argv=(argv if use_sys_argv_override else None))
  else:
    config_flags.DEFINE_config_dict(
        'test_config',
        config=config,
        flag_values=values,
        lock_config=lock_config,
        sys_argv=(argv if use_sys_argv_override else None))

  if required:
    flags.mark_flag_as_required('test_config', flag_values=values)
  values(argv)

  # Going back to original values.
  if not use_sys_argv_override:
    sys.argv = old_argv

  return values


def _get_override_flags(overrides, override_format):
  return ' '.join([override_format.format(path, value)
                   for path, value in overrides.items()])


class _ConfigFlagTestCase(object):
  """Base class for tests with additional asserts for comparing configs."""

  def assert_subset_configs(self, config1, config2):
    """Checks if all attributes/values in config1 are present in config2."""

    if config1 is None:
      return

    if hasattr(config1, '__dict__'):
      keys = [key for key in config1.__dict__ if not key.startswith('_')]
      get_val = getattr
    elif hasattr(config1, 'keys') and callable(config1.keys):
      keys = [key for key in config1.keys() if not key.startswith('_')]
      get_val = lambda haystack, needle: haystack[needle]
    else:
      # This should not fail as it simply means we cannot iterate deeper.
      return

    for attribute in keys:
      if isinstance(get_val(config1, attribute), _CHECK_TYPES):
        self.assertEqual(get_val(config1, attribute),
                         get_val(config2, attribute))
      else:
        # Try to go deeper with comparison
        self.assert_subset_configs(get_val(config1, attribute),
                                   get_val(config2, attribute))

  def assert_equal_configs(self, config1, config2):
    """Checks if two configs are identical."""
    self.assert_subset_configs(config1, config2)
    self.assert_subset_configs(config2, config1)  # pylint: disable=arguments-out-of-order


class ConfigFileFlagTest(_ConfigFlagTestCase, parameterized.TestCase):
  """Tests config flags library."""

  @parameterized.named_parameters(*_DASH_PARAMETERS)
  def testLoading(self, config_flag):
    """Tests loading config from file."""

    values = _parse_flags('./program {}'.format(config_flag),
                          default='nonexisting.py')

    self.assertIn('test_config', values)

    self.assert_equal_configs(values.test_config,
                              mock_config.get_config())
    self.assertEqual(
        values['test_config'].serialize(),
        '--test_config=ml_collections/config_flags/tests/mock_config.py',
    )

  @parameterized.named_parameters(*_DASH_PARAMETERS)
  def testRequired(self, config_flag):
    """Tests making a config_file flag required."""
    with self.assertRaises(flags.IllegalFlagValueError):
      _parse_flags('./program ', required=True)

    values = _parse_flags('./program {}'.format(config_flag),
                          required=True)
    self.assertIn('test_config', values)

  def testDefaultLoading(self):
    """Tests loading config from file using default path."""

    for required in [True, False]:
      values = _parse_flags(
          './program', default=_TEST_CONFIG_FILE, required=required)

      self.assertIn('test_config', values)

      self.assert_equal_configs(values.test_config,
                                mock_config.get_config())
    self.assertEqual(
        values['test_config'].serialize(),
        f'--test_config={_TEST_CONFIG_FILE}',
    )

  def testLoadingNonExistingConfigLoading(self):
    """Tests whether loading non existing file raises an error."""

    nonexisting_file = 'nonexisting.py'

    # Test whether loading non existing files raises an Error with both
    # file loading formats i.e. with '--' and '-'.
    # The Error is not expected to be raised until the config dict actually
    # has one of its attributes accessed.
    values = _parse_flags(
        './program --test_config={}'.format(nonexisting_file),
        default=_TEST_CONFIG_FILE)
    with self.assertRaisesRegex(IOError, '.*{}.*'.format(nonexisting_file)):
      _ = values.test_config.a

    values = _parse_flags(
        './program -test_config {}'.format(nonexisting_file),
        default=_TEST_CONFIG_FILE)
    with self.assertRaisesRegex(IOError, '.*{}.*'.format(nonexisting_file)):
      _ = values.test_config.a

    values = _parse_flags(
        './program -test_config ""', default=_TEST_CONFIG_FILE)
    with self.assertRaisesRegex(IOError, 'empty string'):
      _ = values.test_config.a

  def testIOError(self):
    """Tests that IOErrors raised inside config files are reported correctly."""
    values = _parse_flags('./program --test_config={}'
                          .format(_IOERROR_CONFIG_FILE))
    with self.assertRaisesRegex(IOError, 'This is an IOError'):
      _ = values.test_config.a

  def testValueError(self):
    """Tests that ValueErrors raised when parsing config files are passed up."""
    allchars_regexp = r'[\s\S]*'
    error_regexp = allchars_regexp.join(['Error whilst parsing config file',
                                         'in get_config',
                                         'in value_error_function',
                                         'This is a ValueError'])
    with self.assertRaisesRegex(flags.IllegalFlagValueError, error_regexp):
      _ = _parse_flags('./program --test_config={}'
                       .format(_VALUEERROR_CONFIG_FILE))

  def testTypeError(self):
    """Tests that TypeErrors raised when parsing config files are passed up."""
    allchars_regexp = r'[\s\S]*'
    error_regexp = allchars_regexp.join(['Error whilst parsing config file',
                                         'in get_config',
                                         'in type_error_function',
                                         'This is a TypeError'])
    with self.assertRaisesRegex(flags.IllegalFlagValueError, error_regexp):
      _ = _parse_flags('./program --test_config={}'
                       .format(_TYPEERROR_CONFIG_FILE))

  # Note: While testing the overriding of parameters, we explicitly set
  # '!r' in the format string for the value. This ensures the 'repr()' is
  # called on the argument which basically means that for string arguments,
  # the quotes (' ') are left intact when we format the string.
  @parameterized.named_parameters(
      ('TwoDashConfigAndOverride',
       '--test_config={}'.format(_TEST_CONFIG_FILE), '--test_config.{}={!r}',
       False),
      ('TwoDashSpaceConfigAndOverride',
       '--test_config {}'.format(_TEST_CONFIG_FILE), '--test_config.{} {!r}',
       False),
      ('OneDashConfigAndOverride',
       '-test_config {}'.format(_TEST_CONFIG_FILE), '-test_config.{} {!r}',
       False),
      ('OneDashEqualConfigAndOverride',
       '-test_config={}'.format(_TEST_CONFIG_FILE), '-test_config.{}={!r}',
       False),
      ('OneDashConfigAndTwoDashOverride',
       '-test_config {}'.format(_TEST_CONFIG_FILE), '--test_config.{}={!r}',
       False),
      ('TwoDashConfigAndOneDashOverride',
       '--test_config={}'.format(_TEST_CONFIG_FILE), '-test_config.{} {!r}',
       False),
      ('TwoDashConfigAndOverrideAndSysArgvOverride',
       '--test_config={}'.format(_TEST_CONFIG_FILE), '--test_config.{}={!r}',
       True),
      ('TwoDashSpaceConfigAndOverrideAndSysArgvOverride',
       '--test_config {}'.format(_TEST_CONFIG_FILE), '--test_config.{} {!r}',
       True),
       )
  def testOverride(self, config_flag, override_format, use_sys_argv_override):
    """Tests overriding config values from command line."""
    overrides = {
        'integer': 1,
        'float': -3,
        'dict.float': 3,
        'object.integer': 12,
        'object.float': 123,
        'object.string': 'tom',
        'object.dict.integer': -2,
        'object.dict.float': 3.15,
        'object.dict.list[0]': 101,
        'object.dict.list[2][0]': 103,
        'object.list[0]': 101,
        'object.list[2][0]': 103,
        'object.tuple': '(1,2,(1,2))',
        'object.tuple_with_spaces': '(1, 2, (1, 2))',
        'object_reference.dict.string': 'marry',
        'object.dict.dict.float': 123,
        'object_copy.float': 111.111
    }

    override_flags = _get_override_flags(overrides, override_format)
    values = _parse_flags(
        './program {} {}'.format(config_flag, override_flags),
        use_sys_argv_override=use_sys_argv_override)

    test_config = mock_config.get_config()
    test_config.integer = overrides['integer']
    test_config.float = overrides['float']
    test_config.dict['float'] = overrides['dict.float']
    test_config.object.integer = overrides['object.integer']
    test_config.object.float = overrides['object.float']
    test_config.object.string = overrides['object.string']
    test_config.object.dict['integer'] = overrides['object.dict.integer']
    test_config.object.dict['float'] = overrides['object.dict.float']
    test_config.object.dict['list'][0] = overrides['object.dict.list[0]']
    test_config.object.dict['list'][2][0] = overrides['object.dict.list[2][0]']
    test_config.object.dict['list'][0] = overrides['object.list[0]']
    test_config.object.dict['list'][2][0] = overrides['object.list[2][0]']
    test_config.object.tuple = (1, 2, (1, 2))
    test_config.object.tuple_with_spaces = (1, 2, (1, 2))
    test_config.object_reference.dict['string'] = overrides[
        'object_reference.dict.string']
    test_config.object.dict['dict']['float'] = overrides[
        'object.dict.dict.float']
    test_config.object_copy.float = overrides['object_copy.float']

    self.assert_equal_configs(values.test_config, test_config)

  @parameterized.named_parameters(
      ('GlobalSysArgvParsing', False),
      ('SysArgvOverride', True))
  def testOverrideBoolean(self, use_sys_argv_override):
    """Tests overriding boolean config values from command line."""
    prefix = './program --test_config={}'.format(_TEST_CONFIG_FILE)

    # The default for dict.bool is False.
    values = _parse_flags(
        '{} --test_config.dict.bool'.format(prefix),
        use_sys_argv_override=use_sys_argv_override)
    self.assertTrue(values.test_config.dict['bool'])

    values = _parse_flags(
        '{} --test_config.dict.bool=true'.format(prefix),
        use_sys_argv_override=use_sys_argv_override)
    self.assertTrue(values.test_config.dict['bool'])

    # The default for object.bool is True.
    values = _parse_flags(
        '{} --test_config.object.bool=false'.format(prefix),
        use_sys_argv_override=use_sys_argv_override)
    self.assertFalse(values.test_config.object.bool)

    values = _parse_flags(
        '{} --notest_config.object.bool'.format(prefix),
        use_sys_argv_override=use_sys_argv_override)
    self.assertFalse(values.test_config.object.bool)

  def testOverrideEnum(self):
    """Tests overriding enumn config values from command line."""
    prefix = './program --test_config={}'.format(_TEST_CONFIG_FILE)

    # The default for dict.enum is SPOON.
    values = _parse_flags('{} --test_config.enum=FORK'.format(prefix))
    self.assertEqual(values.test_config.enum, spork.SporkType.FORK)
    # Flag is case-insensetive.
    values = _parse_flags('{} --test_config.enum=spork'.format(prefix))
    self.assertEqual(values.test_config.enum, spork.SporkType.SPORK)

  def testFieldReferenceOverride(self):
    """Tests whether types of FieldReference fields are valid."""
    overrides = {'ref_nodefault': 1, 'ref': 2}
    override_flags = _get_override_flags(overrides, '--test_config.{}={!r}')
    config_flag = '--test_config={}'.format(_FIELDREFERENCE_CONFIG_FILE)
    values = _parse_flags('./program {} {}'.format(config_flag, override_flags))
    cfg = values.test_config

    self.assertEqual(cfg.ref_nodefault, overrides['ref_nodefault'])
    self.assertEqual(cfg.ref, overrides['ref'])

  @parameterized.named_parameters(*_DASH_PARAMETERS)
  def testSetNotExistingKey(self, config_flag):
    """Tests setting value of not existing key."""

    with self.assertRaises(KeyError):
      _parse_flags('./program {} '
                   '--test_config.not_existing_key=1 '.format(config_flag))

  @parameterized.named_parameters(*_DASH_PARAMETERS)
  def testSetReadOnlyField(self, config_flag):
    """Tests setting value of key which is read only."""

    with self.assertRaises(AttributeError):
      _parse_flags('./program {} '
                   '--test_config.readonly_field=1 '.format(config_flag))

  @parameterized.named_parameters(*_DASH_PARAMETERS)
  def testNotSupportedOperation(self, config_flag):
    """Tests setting value to not supported type."""

    with self.assertRaises(config_flags.UnsupportedOperationError):
      _parse_flags('./program {} '
                   '--test_config.list=[1]'.format(config_flag))

  def testParserWrapping(self):
    """Tests callback based Parser wrapping."""

    parser = flags.IntegerParser()

    test_config = mock_config.get_config()
    overrides = {}

    config_field_flag = config_flags._ConfigFieldFlag(
        path='integer',
        config=test_config,
        override_values=overrides,
        parser=parser,
        serializer=flags.ArgumentSerializer(),
        name='integer',
        default=test_config.integer,
        help_string='')

    config_field_flag.parse('12321')
    self.assertEqual(test_config.integer, 12321)
    self.assertEqual(overrides, {'integer': 12321})

  def testTypes(self):
    """Tests whether various types of objects are valid."""

    parser = config_flags.ConfigFileFlagParser('test_config')
    self.assertEqual(parser.flag_type(), 'config object')

  @parameterized.named_parameters(
      ('WithTwoDashesAndEqual', '--test_config=config.py'),
      ('WithTwoDashes', '--test_config'),
      ('WithOneDashAndEqual', '-test_config=config.py'),
      ('WithOneDash', '-test_config'))
  def testConfigSpecified(self, config_argument):
    """Tests whether config is specified on the command line."""

    config_flag = config_flags._ConfigFlag(
        parser=flags.ArgumentParser(),
        serializer=None,
        name='test_config',
        default='defaultconfig.py',
        help_string=''
    )
    self.assertTrue(config_flag._IsConfigSpecified([config_argument]))
    self.assertFalse(config_flag._IsConfigSpecified(['']))

  def testFindConfigSpecified(self):
    """Tests whether config is specified on the command line."""

    config_flag = config_flags._ConfigFlag(
        parser=flags.ArgumentParser(),
        serializer=None,
        name='test_config',
        default='defaultconfig.py',
        help_string=''
    )
    self.assertEqual(config_flag._FindConfigSpecified(['']), -1)

    argv_length = 20
    for i in range(argv_length):
      # Generate list of '--test_config.i=0' args.
      argv = ['--test_config.{}=0'.format(arg) for arg in range(argv_length)]
      self.assertEqual(config_flag._FindConfigSpecified(argv), -1)

      # Override i-th arg with something specifying the value of 'test_config'.
      # After doing this, _FindConfigSpecified should return the value of i.
      argv[i] = '--test_config'
      self.assertEqual(config_flag._FindConfigSpecified(argv), i)
      argv[i] = '--test_config=config.py'
      self.assertEqual(config_flag._FindConfigSpecified(argv), i)
      argv[i] = '-test_config'
      self.assertEqual(config_flag._FindConfigSpecified(argv), i)
      argv[i] = '-test_config=config.py'
      self.assertEqual(config_flag._FindConfigSpecified(argv), i)

  def testLoadingLockedConfigDict(self):
    """Tests loading ConfigDict instance and that it is locked."""

    config_flag = '--test_config={}'.format(_CONFIGDICT_CONFIG_FILE)
    values = _parse_flags('./program {}'.format(config_flag),
                          lock_config=True)

    self.assertTrue(values.test_config.is_locked)
    self.assertTrue(values.test_config.nested_configdict.is_locked)

    values = _parse_flags('./program {}'.format(config_flag),
                          lock_config=False)

    self.assertFalse(values.test_config.is_locked)
    self.assertFalse(values.test_config.nested_configdict.is_locked)

  @parameterized.named_parameters(
      ('WithTwoDashesAndEqual', '--test_config={}'.format(_TEST_DIRECTORY)),
      ('WithTwoDashes', '--test_config {}'.format(_TEST_DIRECTORY)),
      ('WithOneDashAndEqual', '-test_config={}'.format(_TEST_DIRECTORY)),
      ('WithOneDash', '-test_config {}'.format(_TEST_DIRECTORY)))
  def testPriorityOfFieldLookup(self, config_flag):
    """Tests whether attributes have higher priority than key-based lookup."""

    values = _parse_flags('./program {}/mini_config.py'.format(config_flag),
                          lock_config=False)
    self.assertTrue(values.test_config.entry_with_collision)

  @parameterized.named_parameters(
      ('TypeAEnabled', ':type_a', {'thing_a': 23, 'thing_b': 42}, ['thing_c']),
      ('TypeASpecify', ':type_a --test_config.thing_a=24', {'thing_a': 24}, []),
      ('TypeBEnabled', ':type_b', {'thing_a': 19, 'thing_c': 65}, ['thing_b']))
  def testExtraConfigString(self, flag_override, should_exist, should_error):
    """Tests with the config_string argument is used properly."""
    values = _parse_flags(
        './program --test_config={}/parameterised_config.py{}'.format(
            _TEST_DIRECTORY, flag_override))
    # Ensure that the values exist in the ConfigDict, with desired values.
    for subfield_name, expected_value in should_exist.items():
      self.assertEqual(values.test_config[subfield_name], expected_value)

    # Ensure the values which should not be part of the ConfigDict are really
    # not there.
    for should_error_name in should_error:
      with self.assertRaisesRegex(KeyError, 'Did you mean'):
        _ = values.test_config[should_error_name]

  def testExtraConfigInvalidFlag(self):
    with self.assertRaisesRegex(AttributeError, 'not_valid_item'):
      _parse_flags(
          ('./program --test_config={}/parameterised_config.py:type_a '
           '--test_config.not_valid_item=42').format(_TEST_DIRECTORY))

  def testOverridingConfigDict(self):
    """Tests overriding of ConfigDict fields."""

    config_flag = '--test_config={}'.format(_CONFIGDICT_CONFIG_FILE)
    overrides = {
        'integer': 2,
        'reference': 2,
        'list[0]': 5,
        'nested_list[0][0]': 5,
        'nested_configdict.integer': 5,
        'unusable_config.dummy_attribute': 5
    }
    override_flags = _get_override_flags(overrides, '--test_config.{}={}')
    values = _parse_flags('./program {} {}'.format(config_flag, override_flags))
    self.assertEqual(values.test_config.integer, overrides['integer'])
    self.assertEqual(values.test_config.reference, overrides['reference'])
    self.assertEqual(values.test_config.list[0], overrides['list[0]'])
    self.assertEqual(values.test_config.nested_list[0][0],
                     overrides['nested_list[0][0]'])
    self.assertEqual(values.test_config.nested_configdict.integer,
                     overrides['nested_configdict.integer'])
    self.assertEqual(values.test_config.unusable_config.dummy_attribute,
                     overrides['unusable_config.dummy_attribute'])
    # Attribute error.
    overrides = {'nonexistent': 'value'}
    with self.assertRaises(AttributeError):
      override_flags = _get_override_flags(overrides, '--test_config.{}={}')
      _parse_flags('./program {} {}'.format(config_flag, override_flags))

    # "Did you mean" messages.
    overrides = {'integre': 2}
    with self.assertRaisesRegex(AttributeError, 'Did you.*integer.*'):
      override_flags = _get_override_flags(overrides, '--test_config.{}={}')
      _parse_flags('./program {} {}'.format(config_flag, override_flags))

    overrides = {'referecne': 2}
    with self.assertRaisesRegex(AttributeError, 'Did you.*reference.*'):
      override_flags = _get_override_flags(overrides, '--test_config.{}={}')
      _parse_flags('./program {} {}'.format(config_flag, override_flags))

  @parameterized.parameters(
      {'overrides': ('2,',), 'expected_tuple': (2,)},
      {'overrides': ('2,3',), 'expected_tuple': (2, 3)},
      {'overrides': ('(2,3)',), 'expected_tuple': (2, 3)},
      {'overrides': ('2',), 'expected_tuple': (2,)},
      {'overrides': ('2', '3',), 'expected_tuple': (2, 3)},
      {'overrides': ('a',), 'expected_tuple': ('a',)},
      {'overrides': ('"a"',), 'expected_tuple': ('a',)},
      {'overrides': ('"a",',), 'expected_tuple': ('a',)},
      {'overrides': ('("a",)',), 'expected_tuple': ('a',)},
      {'overrides': ('a', 'b',), 'expected_tuple': ('a', 'b')},
      {'overrides': ('("a","b")',), 'expected_tuple': ('a', 'b')},
  )
  def testOverridingMultiConfigDict(self, overrides, expected_tuple):
    config_flag = '--test_config={}'.format(_TEST_CONFIG_FILE)
    override_flags = ' '.join(
        [f'--test_config.tuple={shlex.quote(value)}' for value in overrides])
    values = _parse_flags('./program {} {}'.format(config_flag, override_flags))
    self.assertEqual(values.test_config.tuple, expected_tuple)

  # This test adds new flags, so use FlagSaver to make it hermetic.
  @flagsaver.flagsaver
  def testIsConfigFile(self):
    config_flags.DEFINE_config_file('is_a_config_flag')
    flags.DEFINE_integer('not_a_config_flag', -1, '')

    self.assertTrue(
        config_flags.is_config_flag(flags.FLAGS['is_a_config_flag']))
    self.assertFalse(
        config_flags.is_config_flag(flags.FLAGS['not_a_config_flag']))

  # This test adds new flags, so use FlagSaver to make it hermetic.
  @flagsaver.flagsaver
  def testModuleName(self):
    config_flags.DEFINE_config_file('flag')
    argv_0 = './program'
    _parse_flags(argv_0)
    self.assertIn(flags.FLAGS['flag'],
                  flags.FLAGS.flags_by_module_dict()[argv_0])

  def testFlagOrder(self):
    with self.assertRaisesWithLiteralMatch(
        config_flags.FlagOrderError,
        ('Found --test_config.int=1 in argv before a value for --test_config '
         'was specified')):
      _parse_flags('./program --test_config.int=1 '
                   '--test_config={}'.format(_TEST_CONFIG_FILE))

  @flagsaver.flagsaver
  def testOverrideValues(self):
    config_flags.DEFINE_config_file('config')
    with self.assertRaisesWithLiteralMatch(config_flags.UnparsedFlagError,
                                           'The flag has not been parsed yet'):
      flags.FLAGS['config'].override_values  # pylint: disable=pointless-statement

    original_float = -1.0
    original_dictfloat = -2.0
    config = config_dict.ConfigDict({
        'integer': -1,
        'float': original_float,
        'dict': {
            'float': original_dictfloat
        }
    })
    integer_override = 0
    dictfloat_override = 1.1
    values = _parse_flags('./program --test_config={} --test_config.integer={} '
                          '--test_config.dict.float={}'.format(
                              _TEST_CONFIG_FILE, integer_override,
                              dictfloat_override))

    config.update_from_flattened_dict(
        config_flags.get_override_values(values['test_config']))
    self.assertEqual(config['integer'], integer_override)
    self.assertEqual(config['float'], original_float)
    self.assertEqual(config['dict']['float'], dictfloat_override)

  @parameterized.named_parameters(
      ('ConfigFile1', _TEST_CONFIG_FILE),
      ('ConfigFile2', _CONFIGDICT_CONFIG_FILE),
      ('ParameterisedConfigFile', _PARAMETERISED_CONFIG_FILE + ':type_a'),
      )
  def testConfigPath(self, config_file):
    """Test access to saved config file path."""
    values = _parse_flags('./program --test_config={}'.format(config_file))
    self.assertEqual(config_flags.get_config_filename(values['test_config']),
                     config_file)


def _simple_config():
  config = config_dict.ConfigDict()
  config.foo = 3
  return config


class ConfigDictFlagTest(_ConfigFlagTestCase, parameterized.TestCase):
  """Tests DEFINE_config_dict.

  DEFINE_config_dict reuses a lot of code in DEFINE_config_file so the tests
  here are mostly sanity checks.
  """

  def testBasicUsage(self):
    values = _parse_flags('./program', config=_simple_config())
    self.assertIn('test_config', values)
    self.assert_equal_configs(_simple_config(), values.test_config)

  def testChangingLockedConfigRaisesAnError(self):
    values = _parse_flags(
        './program', config=_simple_config(), lock_config=True)
    with self.assertRaisesRegex(AttributeError, 'config is locked'):
      values.test_config.new_foo = 20

  def testChangingUnlockedConfig(self):
    values = _parse_flags(
        './program', config=_simple_config(), lock_config=False)
    values.test_config.new_foo = 20

  def testNonConfigDictAsConfig(self):
    non_config_dict = dict(a=1, b=2)
    with self.assertRaisesRegex(TypeError, 'should be a ConfigDict'):
      _parse_flags('./program', config=non_config_dict)

  @parameterized.named_parameters(
      ('GlobalSysArgvParsing', False),
      ('SysArgvOverride', True))
  def testOverridingAttribute(self, use_sys_argv_override):
    new_foo = 10
    values = _parse_flags(
        './program --test_config.foo={}'.format(new_foo),
        config=_simple_config(),
        use_sys_argv_override=use_sys_argv_override)
    self.assertNotEqual(new_foo, _simple_config().foo)
    self.assertEqual(new_foo, values.test_config.foo)

  def testOverridingMainConfigFlagRaisesAnError(self):
    with self.assertRaisesRegex(flags.IllegalFlagValueError,
                                'Overriding test_config is not allowed'):
      _parse_flags('./program --test_config=bad_input', config=_simple_config())

  def testOverridesSerialize(self):
    all_types_config = config_dict.ConfigDict()
    all_types_config.type_bool = False
    all_types_config.type_bytes = b'bytes'
    all_types_config.type_float = 1.0
    all_types_config.type_int = 1
    all_types_config.type_str = 'str'
    all_types_config.type_ustr = u'ustr'
    all_types_config.type_tuple = (False, b'bbytes', 1.0, 1, 'str', u'ustr',)
    # Change values via the command line:
    command_line = (
        './program'
        ' --test_config.type_bool=True'
        ' --test_config.type_float=10'
        ' --test_config.type_int=10'
        ' --test_config.type_str=str_commandline'
        ' --test_config.type_tuple="(\'tuple_str\', 10)"'
        )
    command_line += ' --test_config.type_ustr=ustr_commandline'
    values = _parse_flags(command_line, config=copy.copy(all_types_config))

    # Check we get the expected values (ie the ones defined above not the ones
    # defined in the config itself)
    get_parser = lambda name: values._flags()[name].parser.parse
    get_serializer = lambda name: values._flags()[name].serializer.serialize
    serialize_parse = (
        lambda name, value: get_parser(name)(get_serializer(name)(value)))

    with self.subTest('bool'):
      self.assertNotEqual(values.test_config.type_bool,
                          all_types_config['type_bool'])
      self.assertEqual(values.test_config.type_bool, True)
      self.assertEqual(values.test_config.type_bool,
                       serialize_parse('test_config.type_bool',
                                       values.test_config.type_bool))
    with self.subTest('float'):
      self.assertNotEqual(values.test_config.type_float,
                          all_types_config['type_float'])
      self.assertEqual(values.test_config.type_float, 10.)
      self.assertEqual(values.test_config.type_float,
                       serialize_parse('test_config.type_float',
                                       values.test_config.type_float))
    with self.subTest('int'):
      self.assertNotEqual(values.test_config.type_int,
                          all_types_config['type_int'])
      self.assertEqual(values.test_config.type_int, 10)
      self.assertEqual(values.test_config.type_int,
                       serialize_parse('test_config.type_int',
                                       values.test_config.type_int))
    with self.subTest('str'):
      self.assertNotEqual(values.test_config.type_str,
                          all_types_config['type_str'])
      self.assertEqual(values.test_config.type_str, 'str_commandline')
      self.assertEqual(values.test_config.type_str,
                       serialize_parse('test_config.type_str',
                                       values.test_config.type_str))

    with self.subTest('ustr'):
      self.assertNotEqual(values.test_config.type_ustr,
                          all_types_config['type_ustr'])
      self.assertEqual(values.test_config.type_ustr, u'ustr_commandline')
      self.assertEqual(
          values.test_config.type_ustr,
          serialize_parse('test_config.type_ustr',
                          values.test_config.type_ustr))
    with self.subTest('tuple'):
      self.assertNotEqual(values.test_config.type_tuple,
                          all_types_config['type_tuple'])
      self.assertEqual(values.test_config.type_tuple, ('tuple_str', 10))
      self.assertEqual(values.test_config.type_tuple,
                       serialize_parse('test_config.type_tuple',
                                       values.test_config.type_tuple))


def main():
  absltest.main()


if __name__ == '__main__':
  main()
