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
"""Tests for ml_collections.FrozenConfigDict."""

from collections import abc as collections_abc
import copy
import pickle

from absl.testing import absltest
import ml_collections

_TEST_DICT = {
    'int': 2,
    'list': [1, 2],
    'nested_list': [[1, [2]]],
    'set': {1, 2},
    'tuple': (1, 2),
    'frozenset': frozenset({1, 2}),
    'dict': {
        'float': -1.23,
        'list': [1, 2],
        'dict': {},
        'tuple_containing_list': (1, 2, (3, [4, 5], (6, 7))),
        'list_containing_tuple': [1, 2, [3, 4], (5, 6)],
    },
    'ref': ml_collections.FieldReference({'int': 0})
}


def _test_dict_deepcopy():
  return copy.deepcopy(_TEST_DICT)


def _test_configdict():
  return ml_collections.ConfigDict(_TEST_DICT)


def _test_frozenconfigdict():
  return ml_collections.FrozenConfigDict(_TEST_DICT)


class FrozenConfigDictTest(absltest.TestCase):
  """Tests FrozenConfigDict in config flags library."""

  def assertFrozenRaisesValueError(self, input_list):
    """Assert initialization on all elements of input_list raise ValueError."""
    for initial_dictionary in input_list:
      with self.assertRaises(ValueError):
        _ = ml_collections.FrozenConfigDict(initial_dictionary)

  def testBasicEquality(self):
    """Tests basic equality with different types of initialization."""
    fcd = _test_frozenconfigdict()
    fcd_cd = ml_collections.FrozenConfigDict(_test_configdict())
    fcd_fcd = ml_collections.FrozenConfigDict(fcd)
    self.assertEqual(fcd, fcd_cd)
    self.assertEqual(fcd, fcd_fcd)

  def testImmutability(self):
    """Tests immutability of frozen config."""
    fcd = _test_frozenconfigdict()
    self.assertEqual(fcd.list, tuple(_TEST_DICT['list']))
    self.assertEqual(fcd.tuple, _TEST_DICT['tuple'])
    self.assertEqual(fcd.set, frozenset(_TEST_DICT['set']))
    self.assertEqual(fcd.frozenset, _TEST_DICT['frozenset'])
    # Must manually check set to frozenset conversion, since Python == does not
    self.assertIsInstance(fcd.set, frozenset)

    self.assertEqual(fcd.dict.list, tuple(_TEST_DICT['dict']['list']))
    self.assertNotEqual(fcd.dict.tuple_containing_list,
                        _TEST_DICT['dict']['tuple_containing_list'])
    self.assertEqual(fcd.dict.tuple_containing_list[2][1],
                     tuple(_TEST_DICT['dict']['tuple_containing_list'][2][1]))
    self.assertIsInstance(fcd.dict, ml_collections.FrozenConfigDict)

    with self.assertRaises(AttributeError):
      fcd.newitem = 0
    with self.assertRaises(AttributeError):
      fcd.dict.int = 0
    with self.assertRaises(AttributeError):
      fcd['newitem'] = 0
    with self.assertRaises(AttributeError):
      del fcd.int
    with self.assertRaises(AttributeError):
      del fcd['int']

  def testLockAndFreeze(self):
    """Ensures .lock() and .freeze() raise errors."""
    fcd = _test_frozenconfigdict()

    self.assertFalse(fcd.is_locked)
    self.assertFalse(fcd.as_configdict().is_locked)

    with self.assertRaises(AttributeError):
      fcd.lock()
    with self.assertRaises(AttributeError):
      fcd.unlock()
    with self.assertRaises(AttributeError):
      fcd.freeze()
    with self.assertRaises(AttributeError):
      fcd.unfreeze()

  def testInitConfigDict(self):
    """Tests that ConfigDict initialization handles FrozenConfigDict.

    Initializing a ConfigDict on a dictionary with FrozenConfigDict values
    should unfreeze these values.
    """
    dict_without_fcd_node = _test_dict_deepcopy()
    dict_without_fcd_node.pop('ref')
    dict_with_fcd_node = copy.deepcopy(dict_without_fcd_node)
    dict_with_fcd_node['dict'] = ml_collections.FrozenConfigDict(
        dict_with_fcd_node['dict'])
    cd_without_fcd_node = ml_collections.ConfigDict(dict_without_fcd_node)
    cd_with_fcd_node = ml_collections.ConfigDict(dict_with_fcd_node)
    fcd_without_fcd_node = ml_collections.FrozenConfigDict(
        dict_without_fcd_node)
    fcd_with_fcd_node = ml_collections.FrozenConfigDict(dict_with_fcd_node)

    self.assertEqual(cd_without_fcd_node, cd_with_fcd_node)
    self.assertEqual(fcd_without_fcd_node, fcd_with_fcd_node)

  def testInitCopying(self):
    """Tests that initialization copies when and only when necessary.

    Ensures copying only occurs when converting mutable type to immutable type,
    regardless of whether the FrozenConfigDict is initialized by a dict or a
    FrozenConfigDict. Also ensures no copying occurs when converting from
    FrozenConfigDict back to ConfigDict.
    """
    fcd = _test_frozenconfigdict()

    # These should be uncopied when creating fcd
    fcd_unchanged_from_test_dict = [
        (_TEST_DICT['tuple'], fcd.tuple),
        (_TEST_DICT['frozenset'], fcd.frozenset),
        (_TEST_DICT['dict']['tuple_containing_list'][2][2],
         fcd.dict.tuple_containing_list[2][2]),
        (_TEST_DICT['dict']['list_containing_tuple'][3],
         fcd.dict.list_containing_tuple[3])
    ]

    # These should be copied when creating fcd
    fcd_different_from_test_dict = [
        (_TEST_DICT['list'], fcd.list),
        (_TEST_DICT['dict']['tuple_containing_list'][2][1],
         fcd.dict.tuple_containing_list[2][1])
    ]

    for (x, y) in fcd_unchanged_from_test_dict:
      self.assertEqual(id(x), id(y))
    for (x, y) in fcd_different_from_test_dict:
      self.assertNotEqual(id(x), id(y))

    # Also make sure that converting back to ConfigDict makes no copies
    self.assertEqual(
        id(_TEST_DICT['dict']['tuple_containing_list']),
        id(ml_collections.ConfigDict(fcd).dict.tuple_containing_list))

  def testAsConfigDict(self):
    """Tests that converting FrozenConfigDict to ConfigDict works correctly.

    In particular, ensures that FrozenConfigDict does the inverse of ConfigDict
    regarding type_safe, lock, and attribute mutability.
    """
    # First ensure conversion to ConfigDict works on empty FrozenConfigDict
    self.assertEqual(
        ml_collections.ConfigDict(ml_collections.FrozenConfigDict()),
        ml_collections.ConfigDict())

    cd = _test_configdict()
    cd_fcd_cd = ml_collections.ConfigDict(ml_collections.FrozenConfigDict(cd))
    self.assertEqual(cd, cd_fcd_cd)

    # Make sure locking is respected
    cd.lock()
    self.assertEqual(
        cd, ml_collections.ConfigDict(ml_collections.FrozenConfigDict(cd)))

    # Make sure type_safe is respected
    cd = ml_collections.ConfigDict(_TEST_DICT, type_safe=False)
    self.assertEqual(
        cd, ml_collections.ConfigDict(ml_collections.FrozenConfigDict(cd)))

  def testInitSelfReferencing(self):
    """Ensure initialization fails on self-referencing dicts."""
    self_ref = {}
    self_ref['self'] = self_ref
    parent_ref = {'dict': {}}
    parent_ref['dict']['parent'] = parent_ref
    tuple_parent_ref = {'dict': {}}
    tuple_parent_ref['dict']['tuple'] = (1, 2, tuple_parent_ref)
    attribute_cycle = {'dict': copy.deepcopy(self_ref)}

    self.assertFrozenRaisesValueError(
        [self_ref, parent_ref, tuple_parent_ref, attribute_cycle])

  def testInitCycles(self):
    """Ensure initialization fails if an attribute of input is cyclic."""
    inner_cyclic_list = [1, 2]
    cyclic_list = [3, inner_cyclic_list]
    inner_cyclic_list.append(cyclic_list)
    cyclic_tuple = tuple(cyclic_list)

    test_dict_cyclic_list = _test_dict_deepcopy()
    test_dict_cyclic_tuple = _test_dict_deepcopy()

    test_dict_cyclic_list['cyclic_list'] = cyclic_list
    test_dict_cyclic_tuple['dict']['cyclic_tuple'] = cyclic_tuple

    self.assertFrozenRaisesValueError(
        [test_dict_cyclic_list, test_dict_cyclic_tuple])

  def testInitDictInList(self):
    """Ensure initialization fails on dict and ConfigDict in lists/tuples."""
    list_containing_dict = {'list': [1, 2, 3, {'a': 4, 'b': 5}]}
    tuple_containing_dict = {'tuple': (1, 2, 3, {'a': 4, 'b': 5})}
    list_containing_cd = {'list': [1, 2, 3, _test_configdict()]}
    tuple_containing_cd = {'tuple': (1, 2, 3, _test_configdict())}
    fr_containing_list_containing_dict = {
        'fr': ml_collections.FieldReference([1, {
            'a': 2
        }])
    }

    self.assertFrozenRaisesValueError([
        list_containing_dict, tuple_containing_dict, list_containing_cd,
        tuple_containing_cd, fr_containing_list_containing_dict
    ])

  def testInitFieldReferenceInList(self):
    """Ensure initialization fails on FieldReferences in lists/tuples."""
    list_containing_fr = {'list': [1, 2, 3, ml_collections.FieldReference(4)]}
    tuple_containing_fr = {
        'tuple': (1, 2, 3, ml_collections.FieldReference('a'))
    }

    self.assertFrozenRaisesValueError([list_containing_fr, tuple_containing_fr])

  def testInitInvalidAttributeName(self):
    """Ensure initialization fails on attributes with invalid names."""
    dot_name = {'dot.name': None}
    immutable_name = {'__hash__': None}

    with self.assertRaises(ValueError):
      ml_collections.FrozenConfigDict(dot_name)

    with self.assertRaises(AttributeError):
      ml_collections.FrozenConfigDict(immutable_name)

  def testFieldReferenceResolved(self):
    """Tests that FieldReferences are resolved."""
    cfg = ml_collections.ConfigDict({'fr': ml_collections.FieldReference(1)})
    frozen_cfg = ml_collections.FrozenConfigDict(cfg)
    self.assertNotIsInstance(frozen_cfg._fields['fr'],
                             ml_collections.FieldReference)
    hash(frozen_cfg)  # with FieldReference resolved, frozen_cfg is hashable

  def testFieldReferenceCycle(self):
    """Tests that FieldReferences may not contain reference cycles."""
    frozenset_fr = {'frozenset': frozenset({1, 2})}
    frozenset_fr['fr'] = ml_collections.FieldReference(
        frozenset_fr['frozenset'])
    list_fr = {'list': [1, 2]}
    list_fr['fr'] = ml_collections.FieldReference(list_fr['list'])

    cyclic_fr = {'a': 1}
    cyclic_fr['fr'] = ml_collections.FieldReference(cyclic_fr)
    cyclic_fr_parent = {'dict': {}}
    cyclic_fr_parent['dict']['fr'] = ml_collections.FieldReference(
        cyclic_fr_parent)

    # FieldReference is allowed to point to non-cyclic objects:
    _ = ml_collections.FrozenConfigDict(frozenset_fr)
    _ = ml_collections.FrozenConfigDict(list_fr)
    # But not cycles:
    self.assertFrozenRaisesValueError([cyclic_fr, cyclic_fr_parent])

  def testDeepCopy(self):
    """Ensure deepcopy works and does not affect equality."""
    fcd = _test_frozenconfigdict()
    fcd_deepcopy = copy.deepcopy(fcd)
    self.assertEqual(fcd, fcd_deepcopy)

  def testEquals(self):
    """Tests that __eq__() respects hidden mutability."""
    fcd = _test_frozenconfigdict()

    # First, ensure __eq__() returns False when comparing to other types
    self.assertNotEqual(fcd, (1, 2))
    self.assertNotEqual(fcd, fcd.as_configdict())

    list_to_tuple = _test_dict_deepcopy()
    list_to_tuple['list'] = tuple(list_to_tuple['list'])
    fcd_list_to_tuple = ml_collections.FrozenConfigDict(list_to_tuple)

    set_to_frozenset = _test_dict_deepcopy()
    set_to_frozenset['set'] = frozenset(set_to_frozenset['set'])
    fcd_set_to_frozenset = ml_collections.FrozenConfigDict(set_to_frozenset)

    self.assertNotEqual(fcd, fcd_list_to_tuple)

    # Because set == frozenset in Python:
    self.assertEqual(fcd, fcd_set_to_frozenset)

    # Items are not affected by hidden mutability
    self.assertCountEqual(fcd.items(), fcd_list_to_tuple.items())
    self.assertCountEqual(fcd.items(), fcd_set_to_frozenset.items())

  def testEqualsAsConfigDict(self):
    """Tests that eq_as_configdict respects hidden mutability but not type."""
    fcd = _test_frozenconfigdict()

    # First, ensure eq_as_configdict() returns True with an equal ConfigDict but
    # False for other types.
    self.assertFalse(fcd.eq_as_configdict([1, 2]))
    self.assertTrue(fcd.eq_as_configdict(fcd.as_configdict()))
    empty_fcd = ml_collections.FrozenConfigDict()
    self.assertTrue(empty_fcd.eq_as_configdict(ml_collections.ConfigDict()))

    # Now, ensure it has the same immutability detection as __eq__().
    list_to_tuple = _test_dict_deepcopy()
    list_to_tuple['list'] = tuple(list_to_tuple['list'])
    fcd_list_to_tuple = ml_collections.FrozenConfigDict(list_to_tuple)

    set_to_frozenset = _test_dict_deepcopy()
    set_to_frozenset['set'] = frozenset(set_to_frozenset['set'])
    fcd_set_to_frozenset = ml_collections.FrozenConfigDict(set_to_frozenset)

    self.assertFalse(fcd.eq_as_configdict(fcd_list_to_tuple))
    # Because set == frozenset in Python:
    self.assertTrue(fcd.eq_as_configdict(fcd_set_to_frozenset))

  def testHash(self):
    """Ensures __hash__() respects hidden mutability."""
    list_to_tuple = _test_dict_deepcopy()
    list_to_tuple['list'] = tuple(list_to_tuple['list'])

    self.assertEqual(
        hash(_test_frozenconfigdict()),
        hash(ml_collections.FrozenConfigDict(_test_dict_deepcopy())))
    self.assertNotEqual(
        hash(_test_frozenconfigdict()),
        hash(ml_collections.FrozenConfigDict(list_to_tuple)))

    # Ensure Python realizes FrozenConfigDict is hashable
    self.assertIsInstance(_test_frozenconfigdict(), collections_abc.Hashable)

  def testUnhashableType(self):
    """Ensures __hash__() fails if FrozenConfigDict has unhashable value."""
    unhashable_fcd = ml_collections.FrozenConfigDict(
        {'unhashable': bytearray()})
    with self.assertRaises(TypeError):
      hash(unhashable_fcd)

  def testToDict(self):
    """Ensure to_dict() does not care about hidden mutability."""
    list_to_tuple = _test_dict_deepcopy()
    list_to_tuple['list'] = tuple(list_to_tuple['list'])

    self.assertEqual(_test_frozenconfigdict().to_dict(),
                     ml_collections.FrozenConfigDict(list_to_tuple).to_dict())

  def testPickle(self):
    """Make sure FrozenConfigDict can be dumped and loaded with pickle."""
    fcd = _test_frozenconfigdict()
    locked_fcd = ml_collections.FrozenConfigDict(_test_configdict().lock())

    unpickled_fcd = pickle.loads(pickle.dumps(fcd))
    unpickled_locked_fcd = pickle.loads(pickle.dumps(locked_fcd))

    self.assertEqual(fcd, unpickled_fcd)
    self.assertEqual(locked_fcd, unpickled_locked_fcd)


if __name__ == '__main__':
  absltest.main()
