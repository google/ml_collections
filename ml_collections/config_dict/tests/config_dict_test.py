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
"""Tests for ml_collections.ConfigDict."""

import abc
from collections import abc as collections_abc
import functools
import json
import pickle
import sys

from absl.testing import absltest
from absl.testing import parameterized
import ml_collections
from ml_collections.config_dict import config_dict
import mock
import six
import yaml

_TEST_FIELD = {'int': 0}
_TEST_DICT = {
    'float': 2.34,
    'string': 'tom',
    'int': 2,
    'list': [1, 2],
    'dict': {
        'float': -1.23,
        'int': 23
    },
}


def _test_function():
  pass


# Having ABCMeta as a metaclass shouldn't break yaml serialization.
class _TestClass(six.with_metaclass(abc.ABCMeta, object)):

  def __init__(self):
    self.variable_1 = 1
    self.variable_2 = '2'


_test_object = _TestClass()


class _TestClassNoStr():
  pass


_TEST_DICT_BEST_EFFORT = dict(_TEST_DICT)
_TEST_DICT_BEST_EFFORT.update({
    'unserializable': _TestClass,
    'unserializable_no_str': _TestClassNoStr,
    'function': _test_function,
    'object': _test_object,
    'set': {1, 2, 3}
})

# This is how we expect the _TEST_DICT to look after we change the name float to
# double using the function configdict.recursive_rename
_TEST_DICT_CHANGE_FLOAT_NAME = {
    'double': 2.34,
    'string': 'tom',
    'int': 2,
    'list': [1, 2],
    'dict': {
        'double': -1.23,
        'int': 23
    },
}


def _get_test_dict():
  test_dict = dict(_TEST_DICT)
  field = ml_collections.FieldReference(_TEST_FIELD)
  test_dict['ref'] = field
  test_dict['ref2'] = field
  return test_dict


def _get_test_dict_best_effort():
  test_dict = dict(_TEST_DICT_BEST_EFFORT)
  field = ml_collections.FieldReference(_TEST_FIELD)
  test_dict['ref'] = field
  test_dict['ref2'] = field
  return test_dict


def _get_test_config_dict():
  return ml_collections.ConfigDict(_get_test_dict())


def _get_test_config_dict_best_effort():
  return ml_collections.ConfigDict(_get_test_dict_best_effort())


_JSON_TEST_DICT = ('{"dict": {"float": -1.23, "int": 23},'
                   ' "float": 2.34,'
                   ' "int": 2,'
                   ' "list": [1, 2],'
                   ' "ref": {"int": 0},'
                   ' "ref2": {"int": 0},'
                   ' "string": "tom"}')

if six.PY2:
  _DICT_TYPE = "!!python/name:__builtin__.dict ''"
  _UNSERIALIZABLE_MSG = "unserializable object of type: <type 'classobj'>"
else:
  _DICT_TYPE = "!!python/name:builtins.dict ''"
  _UNSERIALIZABLE_MSG = (
      "unserializable object: <class '__main__._TestClassNoStr'>")

_TYPES = {
    'dict_type': _DICT_TYPE,
    'configdict_type': '!!python/object:ml_collections.config_dict.config_dict'
                       '.ConfigDict',
    'fieldreference_type': '!!python/object:ml_collections.config_dict'
                           '.config_dict.FieldReference'
}

_JSON_BEST_EFFORT_TEST_DICT = (
    '{"dict": {"float": -1.23, "int": 23},'
    ' "float": 2.34,'
    ' "function": "function _test_function",'
    ' "int": 2,'
    ' "list": [1, 2],'
    ' "object": {"variable_1": 1, "variable_2": "2"},'
    ' "ref": {"int": 0},'
    ' "ref2": {"int": 0},'
    ' "set": [1, 2, 3],'
    ' "string": "tom",'
    ' "unserializable": "unserializable object: '
    '<class \'__main__._TestClass\'>",'
    ' "unserializable_no_str": "%s"}') % _UNSERIALIZABLE_MSG

_REPR_TEST_DICT = """
dict:
  float: -1.23
  int: 23
float: 2.34
int: 2
list:
- 1
- 2
ref: &id001 {fieldreference_type}
  _field_type: {dict_type}
  _ops: []
  _required: false
  _value:
    int: 0
ref2: *id001
string: tom
""".format(**_TYPES)

_STR_TEST_DICT = """
dict: {float: -1.23, int: 23}
float: 2.34
int: 2
list: [1, 2]
ref: &id001 {int: 0}
ref2: *id001
string: tom
"""

_STR_NESTED_TEST_DICT = """
dict: {float: -1.23, int: 23}
float: 2.34
int: 2
list: [1, 2]
nested_dict:
  float: -1.23
  int: 23
  nested_dict:
    float: -1.23
    int: 23
    non_nested_dict: {float: -1.23, int: 23}
  nested_list:
  - 1
  - 2
  - [3, 4, 5]
  - 6
ref: &id001 {int: 0}
ref2: *id001
string: tom
"""


class ConfigDictTest(parameterized.TestCase):
  """Tests ConfigDict in config flags library."""

  def assertEqualConfigs(self, cfg, dictionary):
    """Asserts recursive equality of config and a dictionary."""
    self.assertEqual(cfg.to_dict(), dictionary)

  def testCreating(self):
    """Tests basic config creation."""
    cfg = ml_collections.ConfigDict()
    cfg.field = 2.34
    self.assertEqual(cfg.field, 2.34)

  def testDir(self):
    """Test that dir() works correctly on config."""
    cfg = ml_collections.ConfigDict()
    cfg.field = 2.34
    self.assertIn('field', dir(cfg))
    self.assertIn('lock', dir(cfg))

  def testFromDictConstruction(self):
    """Tests creation of config from existing dictionary."""
    cfg = ml_collections.ConfigDict(_TEST_DICT)
    self.assertEqualConfigs(cfg, _TEST_DICT)

  def testOverridingValues(self):
    """Tests basic values overriding."""
    cfg = ml_collections.ConfigDict()
    cfg.field = 2.34
    self.assertEqual(cfg.field, 2.34)
    cfg.field = -2.34
    self.assertEqual(cfg.field, -2.34)

  def testDictAttributeTurnsIntoConfigDict(self):
    """Tests that dicts in a ConfigDict turn to ConfigDicts (recursively)."""
    cfg = ml_collections.ConfigDict(_TEST_DICT)
    # Test conversion to dict on creation.
    self.assertIsInstance(cfg.dict, ml_collections.ConfigDict)

    # Test conversion to dict on setting attribute.
    new_dict = {'inside_dict': {'inside_key': 0}}
    cfg.new_dict = new_dict
    self.assertIsInstance(cfg.new_dict, ml_collections.ConfigDict)
    self.assertIsInstance(cfg.new_dict.inside_dict, ml_collections.ConfigDict)
    self.assertEqual(cfg.new_dict.to_dict(), new_dict)

  def testOverrideExceptions(self):
    """Test the `int` and unicode-string exceptions to overriding.

    ConfigDict forces strong type-checking with two exceptions. The first is
    that `int` values can be stored to fields of type `float`. And secondly,
    all string types can be  stored in fields of type `str` or `unicode`.
    """
    cfg = ml_collections.ConfigDict()
    # Test that overriding 'float' fields with int works.
    cfg.float_field = 2.34
    cfg.float_field = 2
    self.assertEqual(cfg.float_field, 2.0)
    # Test that overriding with Unicode strings works.
    cfg.string_field = '42'
    cfg.string_field = u'42'
    self.assertEqual(cfg.string_field, '42')
    # Test that overriding a Unicode field with a `str` type works.
    cfg.unicode_string_field = u'42'
    cfg.unicode_string_field = '42'
    self.assertEqual(cfg.unicode_string_field, u'42')
    # Test that overriding a list with a tuple works.
    cfg.tuple_field = [1, 2, 3]
    cfg.tuple_field = (1, 2)
    self.assertEqual(cfg.tuple_field, [1, 2])
    # Test that overriding a tuple with a list works.
    cfg.list_field = [23, 42]
    cfg.list_field = (8, 9, 10)
    self.assertEqual(cfg.list_field, [8, 9, 10])
    # Test that int <-> long conversions work.
    int_value = 1
    # In Python 2, int(very large number) returns a long
    long_value = int(1e100)
    cfg.int_field = int_value
    cfg.int_field = long_value
    self.assertEqual(cfg.int_field, long_value)
    if sys.version_info.major == 2:
      expected = long
    else:
      expected = int
    self.assertIsInstance(cfg.int_field, expected)
    cfg.long_field = long_value
    cfg.long_field = int_value
    self.assertEqual(cfg.long_field, int_value)
    self.assertIsInstance(cfg.long_field, expected)

  def testOverrideCallable(self):
    """Test that overriding a callable with a callable works."""

    class SomeClass:

      def __init__(self, x, power=1):
        self.y = x**power

      def factory(self, x):
        return SomeClass(self.y + x)

    fn1 = SomeClass
    fn2 = lambda x: SomeClass(x, power=2)
    fn3 = functools.partial(SomeClass, power=3)
    fn4 = SomeClass(4.0).factory
    cfg = ml_collections.ConfigDict()
    for orig in [fn1, fn2, fn3, fn4]:
      for new in [fn1, fn2, fn3, fn4]:
        cfg.fn_field = orig
        cfg.fn_field = new
        self.assertEqual(cfg.fn_field, new)

  def testOverrideFieldReference(self):
    """Test overriding with FieldReference objects."""
    cfg = ml_collections.ConfigDict()
    cfg.field_1 = 'field_1'
    cfg.field_2 = 'field_2'
    # Override using a FieldReference.
    cfg.field_1 = ml_collections.FieldReference('override_1')
    # Override FieldReference field using another FieldReference.
    cfg.field_1 = ml_collections.FieldReference('override_2')
    # Override using empty FieldReference.
    cfg.field_2 = ml_collections.FieldReference(None, field_type=str)
    # Override FieldReference field using string.
    cfg.field_2 = 'field_2'

    # Check a TypeError is raised when using FieldReference's with wrong type.
    with self.assertRaises(TypeError):
      cfg.field_2 = ml_collections.FieldReference(1)

    with self.assertRaises(TypeError):
      cfg.field_2 = ml_collections.FieldReference(None, field_type=int)

  def testTypeSafe(self):
    """Tests type safe checking."""
    cfg = _get_test_config_dict()

    with self.assertRaisesRegex(TypeError, 'field \'float\''):
      cfg.float = 'tom'

    # Test that float cannot be assigned to int.
    with self.assertRaisesRegex(TypeError, 'field \'int\''):
      cfg.int = 12.8

    with self.assertRaisesRegex(TypeError, 'field \'string\''):
      cfg.string = -123

    with self.assertRaisesRegex(TypeError, 'field \'float\''):
      cfg.dict.float = 'string'

    # Ensure None is ignored by type safety
    cfg.string = None
    cfg.string = 'tom'

  def testIgnoreType(self):
    cfg = ml_collections.ConfigDict({
        'string': 'This is a string',
        'float': 3.0,
        'list': [ml_collections.ConfigDict({'float': 1.0})],
        'tuple': [ml_collections.ConfigDict({'float': 1.0})],
        'dict': {
            'float': 1.0
        }
    })

    with cfg.ignore_type():
      cfg.string = -123
      cfg.float = 'string'
      cfg.list[0].float = 'string'
      cfg.tuple[0].float = 'string'
      cfg.dict.float = 'string'

  def testTypeUnsafe(self):
    """Tests lack of type safe checking."""
    cfg = ml_collections.ConfigDict(_get_test_dict(), type_safe=False)
    cfg.float = 'tom'
    cfg.string = -123
    cfg.int = 12.8

  def testLocking(self):
    """Tests lock mechanism."""
    cfg = ml_collections.ConfigDict()
    cfg.field = 2
    cfg.dict_field = {'float': 1.23, 'integer': 3}
    cfg.ref = ml_collections.FieldReference(
        ml_collections.ConfigDict({'integer': 0}))
    cfg.lock()

    cfg.field = -4

    with self.assertRaises(AttributeError):
      cfg.new_field = 2

    with self.assertRaises(AttributeError):
      cfg.dict_field.new_field = -1.23

    with self.assertRaises(AttributeError):
      cfg.ref.new_field = 1

    with self.assertRaises(AttributeError):
      del cfg.field

  def testUnlocking(self):
    """Tests unlock mechanism."""
    cfg = ml_collections.ConfigDict()
    cfg.dict_field = {'float': 1.23, 'integer': 3}
    cfg.ref = ml_collections.FieldReference(
        ml_collections.ConfigDict({'integer': 0}))
    cfg.lock()
    with cfg.unlocked():
      cfg.new_field = 2
      cfg.dict_field.new_field = -1.23
      cfg.ref.new_field = 1

  def testGetMethod(self):
    """Tests get method."""
    cfg = _get_test_config_dict()
    self.assertEqual(cfg.get('float', -1), cfg.float)
    self.assertEqual(cfg.get('ref', -1), cfg.ref)
    self.assertEqual(cfg.get('another_key', -1), -1)
    self.assertIsNone(cfg.get('another_key'))

  def testItemsMethod(self):
    """Tests items method."""
    cfg = _get_test_config_dict()
    self.assertEqual(dict(**cfg), dict(cfg.items()))
    items = cfg.items()
    self.assertEqual(len(items), len(_get_test_dict()))
    for entry in _TEST_DICT.items():
      if isinstance(entry[1], dict):
        entry = (entry[0], ml_collections.ConfigDict(entry[1]))
      self.assertIn(entry, items)
    self.assertIn(('ref', cfg.ref), items)
    self.assertIn(('ref2', cfg.ref2), items)
    ind_ref = items.index(('ref', cfg.ref))
    ind_ref2 = items.index(('ref2', cfg.ref2))
    self.assertIs(items[ind_ref][1], items[ind_ref2][1])
    cfg = ml_collections.ConfigDict()
    self.assertEqual(dict(**cfg), dict(cfg.items()))
    # Test that items are sorted
    self.assertEqual(sorted(dict(**cfg).items()), cfg.items())

  def testGetItemRecursively(self):
    """Tests getting items recursively (e.g., config['a.b'])."""
    cfg = _get_test_config_dict()
    self.assertEqual(cfg['dict.float'], -1.23)
    self.assertEqual('%(dict.int)i' % cfg, '23')

  def testIterItemsMethod(self):
    """Tests iteritems method."""
    cfg = _get_test_config_dict()
    self.assertEqual(dict(**cfg), dict(cfg.iteritems()))
    cfg = ml_collections.ConfigDict()
    self.assertEqual(dict(**cfg), dict(cfg.iteritems()))

  def testIterKeysMethod(self):
    """Tests iterkeys method."""
    some_dict = {'x1': 32, 'x2': 5.2, 'x3': 'str'}
    cfg = ml_collections.ConfigDict(some_dict)
    self.assertEqual(set(six.iterkeys(some_dict)), set(six.iterkeys(cfg)))
    # Test that keys are sorted
    for k_ref, k in zip(sorted(six.iterkeys(cfg)), six.iterkeys(cfg)):
      self.assertEqual(k_ref, k)

  def testKeysMethod(self):
    """Tests keys method."""
    some_dict = {'x1': 32, 'x2': 5.2, 'x3': 'str'}
    cfg = ml_collections.ConfigDict(some_dict)
    self.assertEqual(set(some_dict.keys()), set(cfg.keys()))
    # Test that keys are sorted
    for k_ref, k in zip(sorted(cfg.keys()), cfg.keys()):
      self.assertEqual(k_ref, k)

  def testLenMethod(self):
    """Tests keys method."""
    some_dict = {'x1': 32, 'x2': 5.2, 'x3': 'str'}
    cfg = ml_collections.ConfigDict(some_dict)
    self.assertLen(cfg, len(some_dict))

  def testIterValuesMethod(self):
    """Tests itervalues method."""
    some_dict = {'x1': 32, 'x2': 5.2, 'x3': 'str'}
    cfg = ml_collections.ConfigDict(some_dict)
    self.assertEqual(set(six.itervalues(some_dict)), set(six.itervalues(cfg)))
    # Test that items are sorted
    for k_ref, v in zip(sorted(six.iterkeys(cfg)), six.itervalues(cfg)):
      self.assertEqual(cfg[k_ref], v)

  def testValuesMethod(self):
    """Tests values method."""
    some_dict = {'x1': 32, 'x2': 5.2, 'x3': 'str'}
    cfg = ml_collections.ConfigDict(some_dict)
    self.assertEqual(set(some_dict.values()), set(cfg.values()))
    # Test that items are sorted
    for k_ref, v in zip(sorted(cfg.keys()), cfg.values()):
      self.assertEqual(cfg[k_ref], v)

  def testIterValuesResolvesReferences(self):
    """Tests itervalues FieldReference resolution."""
    cfg = ml_collections.ConfigDict({'x1': 32, 'x2': 5.2, 'x3': 'str'})
    ref = ml_collections.FieldReference(0)
    cfg['x4'] = ref
    for v in cfg.itervalues():
      self.assertNotIsInstance(v, ml_collections.FieldReference)
    self.assertIn(ref, cfg.itervalues(preserve_field_references=True))

  def testValuesResolvesReferences(self):
    """Tests values FieldReference resolution."""
    cfg = ml_collections.ConfigDict({'x1': 32, 'x2': 5.2, 'x3': 'str'})
    ref = ml_collections.FieldReference(0)
    cfg['x4'] = ref
    for v in cfg.values():
      self.assertNotIsInstance(v, ml_collections.FieldReference)
    self.assertIn(ref, cfg.values(preserve_field_references=True))

  def testIterItemsResolvesReferences(self):
    """Tests iteritems FieldReference resolution."""
    cfg = ml_collections.ConfigDict({'x1': 32, 'x2': 5.2, 'x3': 'str'})
    ref = ml_collections.FieldReference(0)
    cfg['x4'] = ref
    for _, v in cfg.iteritems():
      self.assertNotIsInstance(v, ml_collections.FieldReference)
    self.assertIn(('x4', ref), cfg.iteritems(preserve_field_references=True))

  def testItemsResolvesReferences(self):
    """Tests items FieldReference resolution."""
    cfg = ml_collections.ConfigDict({'x1': 32, 'x2': 5.2, 'x3': 'str'})
    ref = ml_collections.FieldReference(0)
    cfg['x4'] = ref
    for _, v in cfg.items():
      self.assertNotIsInstance(v, ml_collections.FieldReference)
    self.assertIn(('x4', ref), cfg.items(preserve_field_references=True))

  def testEquals(self):
    """Tests __eq__ and __ne__ methods."""
    some_dict = {
        'float': 1.23,
        'integer': 3,
        'list': [1, 2],
        'dict': {
            'a': {},
            'b': 'string'
        }
    }
    cfg = ml_collections.ConfigDict(some_dict)
    cfg_other = ml_collections.ConfigDict(some_dict)
    self.assertEqual(cfg, cfg_other)
    self.assertEqual(ml_collections.ConfigDict(cfg), cfg_other)
    cfg_other.float = 3
    self.assertNotEqual(cfg, cfg_other)
    cfg_other.float = cfg.float
    cfg_other.list = ['a', 'b']
    self.assertNotEqual(cfg, cfg_other)
    cfg_other.list = cfg.list
    cfg_other.lock()
    self.assertNotEqual(cfg, cfg_other)
    cfg_other.unlock()
    cfg_other = ml_collections.ConfigDict(some_dict, type_safe=False)
    self.assertNotEqual(cfg, cfg_other)
    cfg = ml_collections.ConfigDict(some_dict)

    # References that have the same id should be equal (even if self-references)
    cfg_other = ml_collections.ConfigDict(some_dict)
    cfg_other.me = cfg
    cfg.me = cfg
    self.assertEqual(cfg, cfg_other)
    cfg = ml_collections.ConfigDict(some_dict)
    cfg.me = cfg
    self.assertEqual(cfg, cfg)

    # Self-references that do not have the same id loop infinitely
    cfg_other = ml_collections.ConfigDict(some_dict)
    cfg_other.me = cfg_other

    # Temporarily disable coverage trace while testing runtime is exceeded
    trace_func = sys.gettrace()
    sys.settrace(None)

    with self.assertRaises(RuntimeError):
      cfg == cfg_other  # pylint:disable=pointless-statement

    sys.settrace(trace_func)

  def testEqAsConfigDict(self):
    """Tests .eq_as_configdict() method."""
    cfg_1 = _get_test_config_dict()
    cfg_2 = _get_test_config_dict()
    cfg_2.added_field = 3.14159
    cfg_self_ref = _get_test_config_dict()
    cfg_self_ref.self_ref = cfg_self_ref
    frozen_cfg_1 = ml_collections.FrozenConfigDict(cfg_1)
    frozen_cfg_2 = ml_collections.FrozenConfigDict(cfg_2)

    self.assertTrue(cfg_1.eq_as_configdict(cfg_1))
    self.assertTrue(cfg_1.eq_as_configdict(frozen_cfg_1))
    self.assertTrue(frozen_cfg_1.eq_as_configdict(cfg_1))
    self.assertTrue(frozen_cfg_1.eq_as_configdict(frozen_cfg_1))

    self.assertFalse(cfg_1.eq_as_configdict(cfg_2))
    self.assertFalse(cfg_1.eq_as_configdict(frozen_cfg_2))
    self.assertFalse(frozen_cfg_1.eq_as_configdict(cfg_self_ref))
    self.assertFalse(frozen_cfg_1.eq_as_configdict(frozen_cfg_2))
    self.assertFalse(cfg_self_ref.eq_as_configdict(cfg_1))

  def testHash(self):
    some_dict = {'float': 1.23, 'integer': 3}
    cfg = ml_collections.ConfigDict(some_dict)
    with self.assertRaisesRegex(TypeError, 'unhashable type'):
      hash(cfg)

    # Ensure Python realizes ConfigDict is not hashable.
    self.assertNotIsInstance(cfg, collections_abc.Hashable)

  def testDidYouMeanFeature(self):
    """Tests 'did you mean' suggestions."""
    cfg = ml_collections.ConfigDict()
    cfg.learning_rate = 0.01
    cfg.lock()

    with self.assertRaisesRegex(AttributeError,
                                'Did you mean.*learning_rate.*'):
      _ = cfg.laerning_rate

    with cfg.unlocked():
      with self.assertRaisesRegex(AttributeError,
                                  'Did you mean.*learning_rate.*'):
        del cfg.laerning_rate

    with self.assertRaisesRegex(AttributeError,
                                'Did you mean.*learning_rate.*'):
      cfg.laerning_rate = 0.02

    self.assertEqual(cfg.learning_rate, 0.01)
    with self.assertRaises(AttributeError):
      _ = self.laerning_rate

  def testReferences(self):
    """Tests assigning references in the dict."""
    cfg = _get_test_config_dict()
    cfg.dict_ref = cfg.dict

    self.assertEqual(cfg.dict_ref, cfg.dict)

  def testPreserveReferences(self):
    """Tests that initializing with another ConfigDict preserves references."""
    cfg = _get_test_config_dict()
    # In the original, "ref" and "ref2" are the same FieldReference
    self.assertIs(cfg.get_ref('ref'), cfg.get_ref('ref2'))

    # Create a copy from the original
    cfg2 = ml_collections.ConfigDict(cfg)
    # If the refs had not been preserved, get_ref would create a new
    # reference for each call
    self.assertIs(cfg2.get_ref('ref'), cfg2.get_ref('ref2'))
    self.assertIs(cfg2.ref, cfg2.ref2)  # the values are also the same object

  def testUnpacking(self):
    """Tests ability to pass ConfigDict instance with ** operator."""
    cfg = ml_collections.ConfigDict()
    cfg.x = 2

    def foo(x):
      return x + 3

    self.assertEqual(foo(**cfg), 5)

  def testUnpackingWithFieldReference(self):
    """Tests ability to pass ConfigDict instance with ** operator."""
    cfg = ml_collections.ConfigDict()
    cfg.x = ml_collections.FieldReference(2)

    def foo(x):
      return x + 3

    self.assertEqual(foo(**cfg), 5)

  def testReadingIncorrectField(self):
    """Tests whether accessing non-existing fields raises an exception."""
    cfg = ml_collections.ConfigDict()
    with self.assertRaises(AttributeError):
      _ = cfg.non_existing_field

    with self.assertRaises(KeyError):
      _ = cfg['non_existing_field']

  def testIteration(self):
    """Tests whether one can iterate over ConfigDict."""
    cfg = ml_collections.ConfigDict()
    for i in range(10):
      cfg['field{}'.format(i)] = 'field{}'.format(i)

    for field in cfg:
      self.assertEqual(cfg[field], getattr(cfg, field))

  def testDeYaml(self):
    """Tests YAML deserialization."""
    cfg = _get_test_config_dict()
    deyamled = yaml.load(cfg.to_yaml(), yaml.UnsafeLoader)
    self.assertEqual(cfg, deyamled)

  def testJSONConversion(self):
    """Tests JSON serialization."""
    cfg = _get_test_config_dict()
    self.assertEqual(
        cfg.to_json(sort_keys=True).strip(), _JSON_TEST_DICT.strip())

    cfg = _get_test_config_dict_best_effort()
    with self.assertRaises(TypeError):
      cfg.to_json()

  def testJSONConversionCustomEncoder(self):
    """Tests JSON serialization with custom encoder."""
    cfg = _get_test_config_dict()
    encoder = json.JSONEncoder()
    mock_encoder_cls = mock.MagicMock()
    mock_encoder_cls.return_value = encoder
    with mock.patch.object(encoder, 'default') as mock_default:
      mock_default.return_value = ''
      cfg.to_json(json_encoder_cls=mock_encoder_cls)
      mock_default.assert_called()

  def testJSONConversionBestEffort(self):
    """Tests JSON serialization."""
    # Check that best effort option doesn't break default functionality
    cfg = _get_test_config_dict()
    self.assertEqual(
        cfg.to_json_best_effort(sort_keys=True).strip(),
        _JSON_TEST_DICT.strip())

    cfg_best_effort = _get_test_config_dict_best_effort()
    self.assertEqual(
        cfg_best_effort.to_json_best_effort(sort_keys=True).strip(),
        _JSON_BEST_EFFORT_TEST_DICT.strip())

  def testReprConversion(self):
    """Tests repr conversion."""
    cfg = _get_test_config_dict()
    self.assertEqual(repr(cfg).strip(), _REPR_TEST_DICT.strip())

  def testLoadFromRepr(self):
    cfg_dict = ml_collections.ConfigDict()
    field = ml_collections.FieldReference(1)
    cfg_dict.r1 = field
    cfg_dict.r2 = field

    cfg_load = yaml.load(repr(cfg_dict), yaml.UnsafeLoader)

    # Test FieldReferences are preserved
    cfg_load['r1'].set(2)
    self.assertEqual(cfg_load['r1'].get(), cfg_load['r2'].get())

  def testStrConversion(self):
    """Tests conversion to str."""
    cfg = _get_test_config_dict()
    # Verify srt(cfg) doesn't raise errors.
    _ = str(cfg)

    test_dict_2 = _get_test_dict()
    test_dict_2['nested_dict'] = {
        'float': -1.23,
        'int': 23,
        'nested_dict': {
            'float': -1.23,
            'int': 23,
            'non_nested_dict': {
                'float': -1.23,
                'int': 233,
            },
        },
        'nested_list': [1, 2, [3, 44, 5], 6],
    }
    cfg_2 = ml_collections.ConfigDict(test_dict_2)
    # Demonstrate that dot-access works.
    cfg_2.nested_dict.nested_dict.non_nested_dict.int = 23
    cfg_2.nested_dict.nested_list[2][1] = 4
    # Verify srt(cfg) doesn't raise errors.
    _ = str(cfg_2)

  def testDotInField(self):
    """Tests trying to create a dot containing field."""
    cfg = ml_collections.ConfigDict()

    with self.assertRaises(ValueError):
      cfg['invalid.name'] = 2.3

  def testToDictConversion(self):
    """Tests whether references are correctly handled when calling to_dict."""
    cfg = ml_collections.ConfigDict()

    field = ml_collections.FieldReference('a string')
    cfg.dict = {
        'float': 2.3,
        'integer': 1,
        'field_ref1': field,
        'field_ref2': field
    }
    cfg.ref = cfg.dict
    cfg.self_ref = cfg

    pure_dict = cfg.to_dict()
    self.assertEqual(type(pure_dict), dict)

    self.assertIs(pure_dict, pure_dict['self_ref'])
    self.assertIs(pure_dict['dict'], pure_dict['ref'])

    # Ensure ConfigDict has been converted to dict.
    self.assertEqual(type(pure_dict['dict']), dict)

    # Ensure FieldReferences are not preserved, by default.
    self.assertNotIsInstance(pure_dict['dict']['field_ref1'],
                             ml_collections.FieldReference)
    self.assertNotIsInstance(pure_dict['dict']['field_ref2'],
                             ml_collections.FieldReference)
    self.assertEqual(pure_dict['dict']['field_ref1'], field.get())
    self.assertEqual(pure_dict['dict']['field_ref2'], field.get())

    pure_dict_with_refs = cfg.to_dict(preserve_field_references=True)
    self.assertEqual(type(pure_dict_with_refs), dict)
    self.assertEqual(type(pure_dict_with_refs['dict']), dict)
    self.assertIsInstance(pure_dict_with_refs['dict']['field_ref1'],
                          ml_collections.FieldReference)
    self.assertIsInstance(pure_dict_with_refs['dict']['field_ref2'],
                          ml_collections.FieldReference)
    self.assertIs(pure_dict_with_refs['dict']['field_ref1'],
                  pure_dict_with_refs['dict']['field_ref2'])

    # Ensure FieldReferences in the dict are not the same as the FieldReferences
    # in the original ConfigDict.
    self.assertIsNot(pure_dict_with_refs['dict']['field_ref1'],
                     cfg.dict['field_ref1'])

  def testToDictTypeUnsafe(self):
    """Tests interaction between ignore_type() and to_dict()."""
    cfg = ml_collections.ConfigDict()
    cfg.string = ml_collections.FieldReference(None, field_type=str)

    with cfg.ignore_type():
      cfg.string = 1
    self.assertEqual(1, cfg.to_dict(preserve_field_references=True)['string'])

  def testCopyAndResolveReferences(self):
    """Tests the .copy_and_resolve_references() method."""
    cfg = ml_collections.ConfigDict()

    field = ml_collections.FieldReference('a string')
    int_field = ml_collections.FieldReference(5)
    cfg.dict = {
        'float': 2.3,
        'integer': 1,
        'field_ref1': field,
        'field_ref2': field,
        'field_ref_int1': int_field,
        'field_ref_int2': int_field + 5,
        'placeholder': config_dict.placeholder(str),
        'cfg': ml_collections.ConfigDict({
            'integer': 1,
            'int_field': int_field
        })
    }

    cfg.ref = cfg.dict
    cfg.self_ref = cfg

    cfg_resolved = cfg.copy_and_resolve_references()
    for field, value in [('float', 2.3), ('integer', 1),
                         ('field_ref1', 'a string'), ('field_ref2', 'a string'),
                         ('field_ref_int1', 5), ('field_ref_int2', 10),
                         ('placeholder', None)]:
      self.assertEqual(getattr(cfg_resolved.dict, field), value)
    for field, value in [('integer', 1), ('int_field', 5)]:
      self.assertEqual(getattr(cfg_resolved.dict.cfg, field), value)
    self.assertIs(cfg_resolved, cfg_resolved['self_ref'])
    self.assertIs(cfg_resolved['dict'], cfg_resolved['ref'])

  def testCopyAndResolveReferencesConfigTypes(self):
    """Tests that .copy_and_resolve_references() handles special types."""
    cfg_type_safe = ml_collections.ConfigDict()
    int_field = ml_collections.FieldReference(5)
    cfg_type_safe.field_ref1 = int_field
    cfg_type_safe.field_ref2 = int_field + 5

    cfg_type_safe.lock()
    cfg_type_safe_locked_resolved = cfg_type_safe.copy_and_resolve_references()
    self.assertTrue(cfg_type_safe_locked_resolved.is_locked)
    self.assertTrue(cfg_type_safe_locked_resolved.is_type_safe)

    cfg = ml_collections.ConfigDict(type_safe=False)
    cfg.field_ref1 = int_field
    cfg.field_ref2 = int_field + 5

    cfg_resolved = cfg.copy_and_resolve_references()
    self.assertFalse(cfg_resolved.is_locked)
    self.assertFalse(cfg_resolved.is_type_safe)

    cfg.lock()
    cfg_locked_resolved = cfg.copy_and_resolve_references()
    self.assertTrue(cfg_locked_resolved.is_locked)
    self.assertFalse(cfg_locked_resolved.is_type_safe)

    for resolved in [
        cfg_type_safe_locked_resolved, cfg_resolved, cfg_locked_resolved
    ]:
      self.assertEqual(resolved.field_ref1, 5)
      self.assertEqual(resolved.field_ref2, 10)

    frozen_cfg = ml_collections.FrozenConfigDict(cfg_type_safe)
    frozen_cfg_resolved = frozen_cfg.copy_and_resolve_references()

    for resolved in [frozen_cfg, frozen_cfg_resolved]:
      self.assertEqual(resolved.field_ref1, 5)
      self.assertEqual(resolved.field_ref2, 10)
      self.assertIsInstance(resolved, ml_collections.FrozenConfigDict)

  def testInitConfigDict(self):
    """Tests initializing a ConfigDict on a ConfigDict."""
    cfg = _get_test_config_dict()
    cfg_2 = ml_collections.ConfigDict(cfg)
    self.assertIsNot(cfg_2, cfg)
    self.assertIs(cfg_2.float, cfg.float)
    self.assertIs(cfg_2.dict, cfg.dict)

    # Ensure ConfigDict fields are initialized as is
    dict_with_cfg_field = {'cfg': cfg}
    cfg_3 = ml_collections.ConfigDict(dict_with_cfg_field)
    self.assertIs(cfg_3.cfg, cfg)

    # Now ensure it works with locking and type_safe
    cfg_4 = ml_collections.ConfigDict(cfg, type_safe=False)
    cfg_4.lock()
    self.assertEqual(cfg_4, ml_collections.ConfigDict(cfg_4))

  def testInitReferenceStructure(self):
    """Ensures initialization preserves reference structure."""
    x = [1, 2, 3]
    self_ref_dict = {
        'float': 2.34,
        'test_dict_1': _TEST_DICT,
        'test_dict_2': _TEST_DICT,
        'list': x
    }
    self_ref_dict['self'] = self_ref_dict
    self_ref_dict['self_fr'] = ml_collections.FieldReference(self_ref_dict)

    self_ref_cd = ml_collections.ConfigDict(self_ref_dict)
    self.assertIs(self_ref_cd.test_dict_1, self_ref_cd.test_dict_2)
    self.assertIs(self_ref_cd, self_ref_cd.self)
    self.assertIs(self_ref_cd, self_ref_cd.self_fr)
    self.assertIs(self_ref_cd.list, x)
    self.assertEqual(self_ref_cd, self_ref_cd.self)

    self_ref_cd.self.int = 1
    self.assertEqual(self_ref_cd.int, 1)

    self_ref_cd_2 = ml_collections.ConfigDict(self_ref_cd)
    self.assertIsNot(self_ref_cd_2, self_ref_cd)
    self.assertIs(self_ref_cd_2.self, self_ref_cd_2)
    self.assertIs(self_ref_cd_2.test_dict_1, self_ref_cd.test_dict_1)

  def testInitFieldReference(self):
    """Tests initialization with FieldReferences."""
    test_dict = dict(x=1, y=1)

    # Reference to a dict.
    reference = ml_collections.FieldReference(test_dict)
    cfg = ml_collections.ConfigDict()
    cfg.reference = reference
    self.assertIsInstance(cfg.reference, ml_collections.ConfigDict)
    self.assertEqual(test_dict['x'], cfg.reference.x)
    self.assertEqual(test_dict['y'], cfg.reference.y)

    # Reference to a ConfigDict.
    test_configdict = ml_collections.ConfigDict(test_dict)
    reference = ml_collections.FieldReference(test_configdict)
    cfg = ml_collections.ConfigDict()
    cfg.reference = reference

    test_configdict.x = 2
    self.assertEqual(test_configdict.x, cfg.reference.x)
    self.assertEqual(test_configdict.y, cfg.reference.y)

    # Reference to a reference.
    reference_int = ml_collections.FieldReference(0)
    reference = ml_collections.FieldReference(reference_int)
    cfg = ml_collections.ConfigDict()
    cfg.reference = reference

    reference_int.set(1)
    self.assertEqual(reference_int.get(), cfg.reference)

  def testDeletingFields(self):
    """Tests whether it is possible to delete fields."""
    cfg = ml_collections.ConfigDict()

    cfg.field1 = 123
    cfg.field2 = 123
    self.assertIn('field1', cfg)
    self.assertIn('field2', cfg)

    del cfg.field1

    self.assertNotIn('field1', cfg)
    self.assertIn('field2', cfg)

    del cfg.field2

    self.assertNotIn('field2', cfg)

    with self.assertRaises(AttributeError):
      del cfg.keys

    with self.assertRaises(KeyError):
      del cfg['keys']

  def testDeletingNestedFields(self):
    """Tests whether it is possible to delete nested fields."""

    cfg = ml_collections.ConfigDict({
        'a': {
            'aa': [1, 2],
        },
        'b': {
            'ba': {
                'baa': 2,
                'bab': 3,
            },
            'bb': {1, 2, 3},
        },
    })

    self.assertIn('a', cfg)
    self.assertIn('aa', cfg.a)
    self.assertIn('baa', cfg.b.ba)

    del cfg['a.aa']
    self.assertIn('a', cfg)
    self.assertNotIn('aa', cfg.a)

    del cfg['a']
    self.assertNotIn('a', cfg)

    del cfg['b.ba.baa']
    self.assertIn('ba', cfg.b)
    self.assertIn('bab', cfg.b.ba)
    self.assertNotIn('baa', cfg.b.ba)

    del cfg['b.ba']
    self.assertNotIn('ba', cfg.b)
    self.assertIn('bb', cfg.b)

    with self.assertRaises(AttributeError):
      del cfg.keys

    with self.assertRaises(KeyError):
      del cfg['keys']

  def testSetAttr(self):
    """Tests whether it is possible to override an attribute."""
    cfg = ml_collections.ConfigDict()
    with self.assertRaises(AttributeError):
      cfg.__setattr__('__class__', 'abc')

  def testPickling(self):
    """Tests whether ConfigDict can be pickled and unpickled."""
    cfg = _get_test_config_dict()
    cfg.lock()
    pickle_cfg = pickle.loads(pickle.dumps(cfg))
    self.assertTrue(pickle_cfg.is_locked)
    self.assertIsInstance(pickle_cfg, ml_collections.ConfigDict)
    self.assertEqual(str(cfg), str(pickle_cfg))

  def testPlaceholder(self):
    """Tests whether FieldReference works correctly as a placeholder."""
    cfg_element = ml_collections.FieldReference(0)
    cfg = ml_collections.ConfigDict({
        'element': cfg_element,
        'nested': {
            'element': cfg_element
        }
    })

    # Type mismatch.
    with self.assertRaises(TypeError):
      cfg.element = 'string'

    cfg.element = 1
    self.assertEqual(cfg.element, cfg.nested.element)

  def testOptional(self):
    """Tests whether FieldReference works correctly as an optional field."""
    # Type mismatch at construction.
    with self.assertRaises(TypeError):
      ml_collections.FieldReference(0, field_type=str)

    # None default and field_type.
    with self.assertRaises(ValueError):
      ml_collections.FieldReference(None)

    cfg = ml_collections.ConfigDict({
        'default': ml_collections.FieldReference(0),
    })

    cfg.default = 1
    self.assertEqual(cfg.default, 1)

  def testOptionalNoDefault(self):
    """Tests optional field with no default value."""
    cfg = ml_collections.ConfigDict({
        'nodefault': ml_collections.FieldReference(None, field_type=str),
    })

    # Type mismatch with field with no default value.
    with self.assertRaises(TypeError):
      cfg.nodefault = 1

    cfg.nodefault = 'string'
    self.assertEqual(cfg.nodefault, 'string')

  def testGetType(self):
    """Tests whether types are correct for FieldReference fields."""
    cfg = ml_collections.ConfigDict()
    cfg.integer = 123
    cfg.ref = ml_collections.FieldReference(123)
    cfg.ref_nodefault = ml_collections.FieldReference(None, field_type=int)

    self.assertEqual(cfg.get_type('integer'), int)
    self.assertEqual(cfg.get_type('ref'), int)
    self.assertEqual(cfg.get_type('ref_nodefault'), int)

    # Check errors in case of misspelled key.
    with self.assertRaisesRegex(AttributeError, 'Did you.*ref_nodefault.*'):
      cfg.get_type('ref_nodefualt')

    with self.assertRaisesRegex(AttributeError, 'Did you.*integer.*'):
      cfg.get_type('integre')


class ConfigDictUpdateTest(absltest.TestCase):

  def testUpdateSimple(self):
    """Tests updating from one ConfigDict to another."""
    first = ml_collections.ConfigDict()
    first.x = 5
    first.y = 'truman'
    first.q = 2.0
    second = ml_collections.ConfigDict()
    second.x = 9
    second.y = 'wilson'
    second.z = 'washington'
    first.update(second)

    self.assertEqual(first.x, 9)
    self.assertEqual(first.y, 'wilson')
    self.assertEqual(first.z, 'washington')
    self.assertEqual(first.q, 2.0)

  def testUpdateNothing(self):
    """Tests updating a ConfigDict with no arguments."""
    cfg = ml_collections.ConfigDict()
    cfg.x = 5
    cfg.y = 9
    cfg.update()
    self.assertLen(cfg, 2)
    self.assertEqual(cfg.x, 5)
    self.assertEqual(cfg.y, 9)

  def testUpdateFromDict(self):
    """Tests updating a ConfigDict from a dict."""
    cfg = ml_collections.ConfigDict()
    cfg.x = 5
    cfg.y = 9
    cfg.update({'x': 6, 'z': 2})
    self.assertEqual(cfg.x, 6)
    self.assertEqual(cfg.y, 9)
    self.assertEqual(cfg.z, 2)

  def testUpdateFromKwargs(self):
    """Tests updating a ConfigDict from kwargs."""
    cfg = ml_collections.ConfigDict()
    cfg.x = 5
    cfg.y = 9
    cfg.update(x=6, z=2)
    self.assertEqual(cfg.x, 6)
    self.assertEqual(cfg.y, 9)
    self.assertEqual(cfg.z, 2)

  def testUpdateFromDictAndKwargs(self):
    """Tests updating a ConfigDict from a dict and kwargs."""
    cfg = ml_collections.ConfigDict()
    cfg.x = 5
    cfg.y = 9
    cfg.update({'x': 4, 'z': 2}, x=6)
    self.assertEqual(cfg.x, 6)  # kwarg overrides value from dict
    self.assertEqual(cfg.y, 9)
    self.assertEqual(cfg.z, 2)

  def testUpdateFromMultipleDictTypeError(self):
    """Tests that updating a ConfigDict from two dicts raises a TypeError."""
    cfg = ml_collections.ConfigDict()
    cfg.x = 5
    cfg.y = 9
    with self.assertRaisesRegex(TypeError,
                                'update expected at most 1 arguments, got 2'):
      cfg.update({'x': 4}, {'z': 2})

  def testUpdateNested(self):
    """Tests updating a ConfigDict from a nested dict."""
    cfg = ml_collections.ConfigDict()
    cfg.subcfg = ml_collections.ConfigDict()
    cfg.p = 5
    cfg.q = 6
    cfg.subcfg.y = 9
    cfg.update({'p': 4, 'subcfg': {'y': 10, 'z': 5}})
    self.assertEqual(cfg.p, 4)
    self.assertEqual(cfg.q, 6)
    self.assertEqual(cfg.subcfg.y, 10)
    self.assertEqual(cfg.subcfg.z, 5)

  def _assert_associated(self, cfg1, cfg2, key):
    self.assertEqual(cfg1[key], cfg2[key])
    cfg1[key] = 1
    cfg2[key] = 2
    self.assertEqual(cfg1[key], 2)
    cfg1[key] = 3
    self.assertEqual(cfg2[key], 3)

  def testUpdateFieldReference(self):
    """Tests updating to/from FieldReference fields."""
    # Updating FieldReference...
    ref = ml_collections.FieldReference(1)
    cfg = ml_collections.ConfigDict(dict(a=ref, b=ref))
    # from value.
    cfg.update(ml_collections.ConfigDict(dict(a=2)))
    self.assertEqual(cfg.a, 2)
    self.assertEqual(cfg.b, 2)
    # from FieldReference.
    error_message = 'Cannot update a FieldReference from another FieldReference'
    with self.assertRaisesRegex(TypeError, error_message):
      cfg.update(
          ml_collections.ConfigDict(dict(a=ml_collections.FieldReference(2))))
    with self.assertRaisesRegex(TypeError, error_message):
      cfg.update(
          ml_collections.ConfigDict(dict(b=ml_collections.FieldReference(2))))

    # Updating empty ConfigDict with FieldReferences.
    ref = ml_collections.FieldReference(1)
    cfg_from = ml_collections.ConfigDict(dict(a=ref, b=ref))
    cfg = ml_collections.ConfigDict()
    cfg.update(cfg_from)
    self._assert_associated(cfg, cfg_from, 'a')
    self._assert_associated(cfg, cfg_from, 'b')

    # Updating values with FieldReferences.
    ref = ml_collections.FieldReference(1)
    cfg_from = ml_collections.ConfigDict(dict(a=ref, b=ref))
    cfg = ml_collections.ConfigDict(dict(a=2, b=3))
    cfg.update(cfg_from)
    self._assert_associated(cfg, cfg_from, 'a')
    self._assert_associated(cfg, cfg_from, 'b')

  def testUpdateFromFlattened(self):
    cfg = ml_collections.ConfigDict({'a': 1, 'b': {'c': {'d': 2}}})
    updates = {'a': 2, 'b.c.d': 3}
    cfg.update_from_flattened_dict(updates)
    self.assertEqual(cfg.a, 2)
    self.assertEqual(cfg.b.c.d, 3)

  def testUpdateFromFlattenedWithPrefix(self):
    cfg = ml_collections.ConfigDict({'a': 1, 'b': {'c': {'d': 2}}})
    updates = {'a': 2, 'b.c.d': 3}
    cfg.b.update_from_flattened_dict(updates, 'b.')
    self.assertEqual(cfg.a, 1)
    self.assertEqual(cfg.b.c.d, 3)

  def testUpdateFromFlattenedNotFound(self):
    cfg = ml_collections.ConfigDict({'a': 1, 'b': {'c': {'d': 2}}})
    updates = {'a': 2, 'b.d.e': 3}
    with self.assertRaisesRegex(
        KeyError, 'Key "b.d.e" cannot be set as "b.d" was not found.'):
      cfg.update_from_flattened_dict(updates)

  def testUpdateFromFlattenedWrongType(self):
    cfg = ml_collections.ConfigDict({'a': 1, 'b': {'c': {'d': 2}}})
    updates = {'a.b.c': 2}
    with self.assertRaisesRegex(
        KeyError, 'Key "a.b.c" cannot be updated as "a" is not a ConfigDict.'):
      cfg.update_from_flattened_dict(updates)

  def testUpdateFromFlattenedTupleListConversion(self):
    cfg = ml_collections.ConfigDict({
        'a': 1,
        'b': {
            'c': {
                'd': (1, 2, 3, 4, 5),
            }
        }
    })
    updates = {
        'b.c.d': [2, 4, 6, 8],
    }
    cfg.update_from_flattened_dict(updates)
    self.assertIsInstance(cfg.b.c.d, tuple)
    self.assertEqual(cfg.b.c.d, (2, 4, 6, 8))

  def testDecodeError(self):
    # ConfigDict containing two strings with incompatible encodings.
    cfg = ml_collections.ConfigDict({
        'dill': pickle.dumps(_test_function, protocol=pickle.HIGHEST_PROTOCOL),
        'unicode': u'unicode string'
    })

    expected_error = config_dict.JSONDecodeError if six.PY2 else TypeError
    with self.assertRaises(expected_error):
      cfg.to_json()

  def testConvertDict(self):
    """Test automatic conversion, or not, of dict to ConfigDict."""
    cfg = ml_collections.ConfigDict()
    cfg.a = dict(b=dict(c=0))
    self.assertIsInstance(cfg.a, ml_collections.ConfigDict)
    self.assertIsInstance(cfg.a.b, ml_collections.ConfigDict)

    cfg = ml_collections.ConfigDict(convert_dict=False)
    cfg.a = dict(b=dict(c=0))
    self.assertNotIsInstance(cfg.a, ml_collections.ConfigDict)
    self.assertIsInstance(cfg.a, dict)
    self.assertIsInstance(cfg.a['b'], dict)

  def testConvertDictInInitialValue(self):
    """Test automatic conversion, or not, of dict to ConfigDict."""
    initial_dict = dict(a=dict(b=dict(c=0)))
    cfg = ml_collections.ConfigDict(initial_dict)
    self.assertIsInstance(cfg.a, ml_collections.ConfigDict)
    self.assertIsInstance(cfg.a.b, ml_collections.ConfigDict)

    cfg = ml_collections.ConfigDict(initial_dict, convert_dict=False)
    self.assertNotIsInstance(cfg.a, ml_collections.ConfigDict)
    self.assertIsInstance(cfg.a, dict)
    self.assertIsInstance(cfg.a['b'], dict)

  def testConvertDictInCopyAndResolveReferences(self):
    """Test conversion, or not, of dict in copy and resolve references."""
    cfg = ml_collections.ConfigDict()
    cfg.a = dict(b=dict(c=0))
    copied_cfg = cfg.copy_and_resolve_references()
    self.assertIsInstance(copied_cfg.a, ml_collections.ConfigDict)
    self.assertIsInstance(copied_cfg.a.b, ml_collections.ConfigDict)

    cfg = ml_collections.ConfigDict(convert_dict=False)
    cfg.a = dict(b=dict(c=0))
    copied_cfg = cfg.copy_and_resolve_references()
    self.assertNotIsInstance(copied_cfg.a, ml_collections.ConfigDict)
    self.assertIsInstance(copied_cfg.a, dict)
    self.assertIsInstance(copied_cfg.a['b'], dict)

  def testConvertDictTypeCompat(self):
    """Test that automatic conversion to ConfigDict doesn't trigger type errors."""
    cfg = ml_collections.ConfigDict()
    cfg.a = {}
    self.assertIsInstance(cfg.a, ml_collections.ConfigDict)
    # This checks that dict to configdict casting doesn't produce type mismatch.
    cfg.a = {}

  def testYamlNoConvert(self):
    """Test deserialisation from YAML without convert dict.

    This checks backward compatibility of deserialisation.
    """
    cfg = ml_collections.ConfigDict(dict(a=1))
    self.assertTrue(yaml.load(cfg.to_yaml(), yaml.UnsafeLoader)._convert_dict)

  def testRecursiveRename(self):
    """Test recursive_rename.

    The dictionary should be the same but with the specified name changed.
    """
    cfg = ml_collections.ConfigDict(_TEST_DICT)
    new_cfg = config_dict.recursive_rename(cfg, 'float', 'double')
    # Check that the new config has float changed to double as we expect
    self.assertEqual(new_cfg.to_dict(), _TEST_DICT_CHANGE_FLOAT_NAME)
    # Check that the original config is unchanged
    self.assertEqual(cfg.to_dict(), _TEST_DICT)

  def testGetOnewayRef(self):
    cfg = config_dict.create(a=1)
    cfg.b = cfg.get_oneway_ref('a')

    cfg.a = 2
    self.assertEqual(2, cfg.b)

    cfg.b = 3
    self.assertEqual(2, cfg.a)
    self.assertEqual(3, cfg.b)


class CreateTest(absltest.TestCase):

  def testBasic(self):
    config = config_dict.create(a=1, b='b')
    dct = {'a': 1, 'b': 'b'}
    self.assertEqual(config.to_dict(), dct)

  def testNested(self):
    config = config_dict.create(
        data=config_dict.create(game='freeway'),
        model=config_dict.create(num_hidden=1000))

    dct = {'data': {'game': 'freeway'}, 'model': {'num_hidden': 1000}}
    self.assertEqual(config.to_dict(), dct)


class PlaceholderTest(absltest.TestCase):

  def testBasic(self):
    config = config_dict.create(a=1, b=config_dict.placeholder(int))
    self.assertEqual(config.to_dict(), {'a': 1, 'b': None})
    config.b = 5
    self.assertEqual(config.to_dict(), {'a': 1, 'b': 5})

  def testTypeChecking(self):
    config = config_dict.create(a=1, b=config_dict.placeholder(int))
    with self.assertRaises(TypeError):
      config.b = 'chutney'

  def testRequired(self):
    config = config_dict.create(a=config_dict.required_placeholder(int))
    ref = config.get_ref('a')
    with self.assertRaises(config_dict.RequiredValueError):
      config.a  # pylint: disable=pointless-statement
    with self.assertRaises(config_dict.RequiredValueError):
      config.to_dict()
    with self.assertRaises(config_dict.RequiredValueError):
      ref.get()

    config.a = 10
    self.assertEqual(config.to_dict(), {'a': 10})
    self.assertEqual(str(config), yaml.dump({'a': 10}))

    # Reset to None and check we still get an error.
    config.a = None
    with self.assertRaises(config_dict.RequiredValueError):
      config.a  # pylint: disable=pointless-statement

    # Set to a different value using the reference obtained calling get_ref().
    ref.set(5)
    self.assertEqual(config.to_dict(), {'a': 5})
    self.assertEqual(str(config), yaml.dump({'a': 5}))

    # dict placeholder.
    test_dict = {'field': 10}
    config = config_dict.create(
        a=config_dict.required_placeholder(dict),
        b=ml_collections.FieldReference(test_dict.copy()))
    # ConfigDict initialization converts dict to ConfigDict.
    self.assertEqual(test_dict, config.b.to_dict())
    config.a = test_dict
    self.assertEqual(test_dict, config.a)


class CycleTest(absltest.TestCase):

  def testCycle(self):
    config = config_dict.create(a=1)
    config.b = config.get_ref('a') + config.get_ref('a')
    self.assertFalse(config.get_ref('b').has_cycle())
    with self.assertRaises(config_dict.MutabilityError):
      config.a = config.get_ref('a')
    with self.assertRaises(config_dict.MutabilityError):
      config.a = config.get_ref('b')


if __name__ == '__main__':
  absltest.main()
