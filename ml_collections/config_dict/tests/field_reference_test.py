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
"""Tests for ml_collections.FieldReference."""

import operator

from absl.testing import absltest
from absl.testing import parameterized
import ml_collections
from ml_collections.config_dict import config_dict


class FieldReferenceTest(parameterized.TestCase):

  def _test_binary_operator(self,
                            initial_value,
                            other_value,
                            op,
                            true_value,
                            new_initial_value,
                            new_true_value,
                            assert_fn=None):
    """Helper for testing binary operators.

    Generally speaking this checks that:
      1. `op(initial_value, other_value) COMP true_value`
      2. `op(new_initial_value, other_value) COMP new_true_value
    where `COMP` is the comparison function defined by `assert_fn`.

    Args:
      initial_value: Initial value for the `FieldReference`, this is the first
        argument for the binary operator.
      other_value: The second argument for the binary operator.
      op: The binary operator.
      true_value: The expected output of the binary operator.
      new_initial_value: The value that the `FieldReference` is changed to.
      new_true_value: The expected output of the binary operator after the
        `FieldReference` has changed.
      assert_fn: Function used to check the output values.
    """
    if assert_fn is None:
      assert_fn = self.assertEqual

    ref = ml_collections.FieldReference(initial_value)
    new_ref = op(ref, other_value)
    assert_fn(new_ref.get(), true_value)

    config = ml_collections.ConfigDict()
    config.a = initial_value
    config.b = other_value
    config.result = op(config.get_ref('a'), config.b)
    assert_fn(config.result, true_value)

    config.a = new_initial_value
    assert_fn(config.result, new_true_value)

  def _test_unary_operator(self,
                           initial_value,
                           op,
                           true_value,
                           new_initial_value,
                           new_true_value,
                           assert_fn=None):
    """Helper for testing unary operators.

    Generally speaking this checks that:
      1. `op(initial_value) COMP true_value`
      2. `op(new_initial_value) COMP new_true_value
    where `COMP` is the comparison function defined by `assert_fn`.

    Args:
      initial_value: Initial value for the `FieldReference`, this is the first
        argument for the unary operator.
      op: The unary operator.
      true_value: The expected output of the unary operator.
      new_initial_value: The value that the `FieldReference` is changed to.
      new_true_value: The expected output of the unary operator after the
        `FieldReference` has changed.
      assert_fn: Function used to check the output values.
    """
    if assert_fn is None:
      assert_fn = self.assertEqual

    ref = ml_collections.FieldReference(initial_value)
    new_ref = op(ref)
    assert_fn(new_ref.get(), true_value)

    config = ml_collections.ConfigDict()
    config.a = initial_value
    config.result = op(config.get_ref('a'))
    assert_fn(config.result, true_value)

    config.a = new_initial_value
    assert_fn(config.result, new_true_value)

  def testBasic(self):
    ref = ml_collections.FieldReference(1)
    self.assertEqual(ref.get(), 1)

  def testGetRef(self):
    config = ml_collections.ConfigDict()
    config.a = 1.
    config.b = config.get_ref('a') + 10
    config.c = config.get_ref('b') + 10
    self.assertEqual(config.c, 21.0)

  def testFunction(self):

    def fn(x):
      return x + 5

    config = ml_collections.ConfigDict()
    config.a = 1
    config.b = fn(config.get_ref('a'))
    config.c = fn(config.get_ref('b'))

    self.assertEqual(config.b, 6)
    self.assertEqual(config.c, 11)
    config.a = 2
    self.assertEqual(config.b, 7)
    self.assertEqual(config.c, 12)

  def testCycles(self):
    config = ml_collections.ConfigDict()
    config.a = 1.
    config.b = config.get_ref('a') + 10
    config.c = config.get_ref('b') + 10

    self.assertEqual(config.b, 11.0)
    self.assertEqual(config.c, 21.0)

    # Introduce a cycle
    with self.assertRaisesRegex(config_dict.MutabilityError, 'cycle'):
      config.a = config.get_ref('c') - 1.0

    # Introduce a cycle on second operand
    with self.assertRaisesRegex(config_dict.MutabilityError, 'cycle'):
      config.a = ml_collections.FieldReference(5.0) + config.get_ref('c')

    # We can create multiple FieldReferences that all point to the same object
    l = [0]
    config = ml_collections.ConfigDict()
    config.a = l
    config.b = l
    config.c = config.get_ref('a') + ['c']
    config.d = config.get_ref('b') + ['d']

    self.assertEqual(config.c, [0, 'c'])
    self.assertEqual(config.d, [0, 'd'])

    # Make sure nothing was mutated
    self.assertEqual(l, [0])
    self.assertEqual(config.c, [0, 'c'])

    config.a = [1]
    config.b = [2]
    self.assertEqual(l, [0])
    self.assertEqual(config.c, [1, 'c'])
    self.assertEqual(config.d, [2, 'd'])

  @parameterized.parameters(
      {
          'initial_value': 1,
          'other_value': 2,
          'true_value': 3,
          'new_initial_value': 10,
          'new_true_value': 12
      }, {
          'initial_value': 2.0,
          'other_value': 2.5,
          'true_value': 4.5,
          'new_initial_value': 3.7,
          'new_true_value': 6.2
      }, {
          'initial_value': 'hello, ',
          'other_value': 'world!',
          'true_value': 'hello, world!',
          'new_initial_value': 'foo, ',
          'new_true_value': 'foo, world!'
      }, {
          'initial_value': ['hello'],
          'other_value': ['world'],
          'true_value': ['hello', 'world'],
          'new_initial_value': ['foo'],
          'new_true_value': ['foo', 'world']
      }, {
          'initial_value': ml_collections.FieldReference(10),
          'other_value': ml_collections.FieldReference(5.0),
          'true_value': 15.0,
          'new_initial_value': 12,
          'new_true_value': 17.0
      }, {
          'initial_value': config_dict.placeholder(float),
          'other_value': 7.0,
          'true_value': None,
          'new_initial_value': 12,
          'new_true_value': 19.0
      }, {
          'initial_value': 5.0,
          'other_value': config_dict.placeholder(float),
          'true_value': None,
          'new_initial_value': 8.0,
          'new_true_value': None
      }, {
          'initial_value': config_dict.placeholder(str),
          'other_value': 'tail',
          'true_value': None,
          'new_initial_value': 'head',
          'new_true_value': 'headtail'
      })
  def testAdd(self, initial_value, other_value, true_value, new_initial_value,
              new_true_value):
    self._test_binary_operator(initial_value, other_value, operator.add,
                               true_value, new_initial_value, new_true_value)

  @parameterized.parameters(
      {
          'initial_value': 5,
          'other_value': 3,
          'true_value': 2,
          'new_initial_value': -1,
          'new_true_value': -4
      }, {
          'initial_value': 2.0,
          'other_value': 2.5,
          'true_value': -0.5,
          'new_initial_value': 12.3,
          'new_true_value': 9.8
      }, {
          'initial_value': set(['hello', 123, 4.5]),
          'other_value': set([123]),
          'true_value': set(['hello', 4.5]),
          'new_initial_value': set([123]),
          'new_true_value': set([])
      }, {
          'initial_value': ml_collections.FieldReference(10),
          'other_value': ml_collections.FieldReference(5.0),
          'true_value': 5.0,
          'new_initial_value': 12,
          'new_true_value': 7.0
      }, {
          'initial_value': config_dict.placeholder(float),
          'other_value': 7.0,
          'true_value': None,
          'new_initial_value': 12,
          'new_true_value': 5.0
      })
  def testSub(self, initial_value, other_value, true_value, new_initial_value,
              new_true_value):
    self._test_binary_operator(initial_value, other_value, operator.sub,
                               true_value, new_initial_value, new_true_value)

  @parameterized.parameters(
      {
          'initial_value': 1,
          'other_value': 2,
          'true_value': 2,
          'new_initial_value': 3,
          'new_true_value': 6
      }, {
          'initial_value': 2.0,
          'other_value': 2.5,
          'true_value': 5.0,
          'new_initial_value': 3.5,
          'new_true_value': 8.75
      }, {
          'initial_value': ['hello'],
          'other_value': 3,
          'true_value': ['hello', 'hello', 'hello'],
          'new_initial_value': ['foo'],
          'new_true_value': ['foo', 'foo', 'foo']
      }, {
          'initial_value': ml_collections.FieldReference(10),
          'other_value': ml_collections.FieldReference(5.0),
          'true_value': 50.0,
          'new_initial_value': 1,
          'new_true_value': 5.0
      }, {
          'initial_value': config_dict.placeholder(float),
          'other_value': 7.0,
          'true_value': None,
          'new_initial_value': 12,
          'new_true_value': 84.0
      })
  def testMul(self, initial_value, other_value, true_value, new_initial_value,
              new_true_value):
    self._test_binary_operator(initial_value, other_value, operator.mul,
                               true_value, new_initial_value, new_true_value)

  @parameterized.parameters(
      {
          'initial_value': 3,
          'other_value': 2,
          'true_value': 1.5,
          'new_initial_value': 10,
          'new_true_value': 5.0
      }, {
          'initial_value': 2.0,
          'other_value': 2.5,
          'true_value': 0.8,
          'new_initial_value': 6.3,
          'new_true_value': 2.52
      }, {
          'initial_value': ml_collections.FieldReference(10),
          'other_value': ml_collections.FieldReference(5.0),
          'true_value': 2.0,
          'new_initial_value': 13,
          'new_true_value': 2.6
      }, {
          'initial_value': config_dict.placeholder(float),
          'other_value': 7.0,
          'true_value': None,
          'new_initial_value': 17.5,
          'new_true_value': 2.5
      })
  def testTrueDiv(self, initial_value, other_value, true_value,
                  new_initial_value, new_true_value):
    self._test_binary_operator(initial_value, other_value, operator.truediv,
                               true_value, new_initial_value, new_true_value)

  @parameterized.parameters(
      {
          'initial_value': 3,
          'other_value': 2,
          'true_value': 1,
          'new_initial_value': 7,
          'new_true_value': 3
      }, {
          'initial_value': ml_collections.FieldReference(10),
          'other_value': ml_collections.FieldReference(5),
          'true_value': 2,
          'new_initial_value': 28,
          'new_true_value': 5
      }, {
          'initial_value': config_dict.placeholder(int),
          'other_value': 7,
          'true_value': None,
          'new_initial_value': 25,
          'new_true_value': 3
      })
  def testFloorDiv(self, initial_value, other_value, true_value,
                   new_initial_value, new_true_value):
    self._test_binary_operator(initial_value, other_value, operator.floordiv,
                               true_value, new_initial_value, new_true_value)

  @parameterized.parameters(
      {
          'initial_value': 3,
          'other_value': 2,
          'true_value': 9,
          'new_initial_value': 10,
          'new_true_value': 100
      }, {
          'initial_value': 2.7,
          'other_value': 3.2,
          'true_value': 24.0084457245,
          'new_initial_value': 6.5,
          'new_true_value': 399.321543621
      }, {
          'initial_value': ml_collections.FieldReference(10),
          'other_value': ml_collections.FieldReference(5),
          'true_value': 1e5,
          'new_initial_value': 2,
          'new_true_value': 32
      }, {
          'initial_value': config_dict.placeholder(float),
          'other_value': 3.0,
          'true_value': None,
          'new_initial_value': 7.0,
          'new_true_value': 343.0
      })
  def testPow(self, initial_value, other_value, true_value, new_initial_value,
              new_true_value):
    self._test_binary_operator(
        initial_value,
        other_value,
        operator.pow,
        true_value,
        new_initial_value,
        new_true_value,
        assert_fn=self.assertAlmostEqual)

  @parameterized.parameters(
      {
          'initial_value': 3,
          'other_value': 2,
          'true_value': 1,
          'new_initial_value': 10,
          'new_true_value': 0
      }, {
          'initial_value': 5.3,
          'other_value': 3.2,
          'true_value': 2.0999999999999996,
          'new_initial_value': 77,
          'new_true_value': 0.2
      }, {
          'initial_value': ml_collections.FieldReference(10),
          'other_value': ml_collections.FieldReference(5),
          'true_value': 0,
          'new_initial_value': 32,
          'new_true_value': 2
      }, {
          'initial_value': config_dict.placeholder(int),
          'other_value': 7,
          'true_value': None,
          'new_initial_value': 25,
          'new_true_value': 4
      })
  def testMod(self, initial_value, other_value, true_value, new_initial_value,
              new_true_value):
    self._test_binary_operator(
        initial_value,
        other_value,
        operator.mod,
        true_value,
        new_initial_value,
        new_true_value,
        assert_fn=self.assertAlmostEqual)

  @parameterized.parameters(
      {
          'initial_value': True,
          'other_value': True,
          'true_value': True,
          'new_initial_value': False,
          'new_true_value': False
      }, {
          'initial_value': ml_collections.FieldReference(False),
          'other_value': ml_collections.FieldReference(False),
          'true_value': False,
          'new_initial_value': True,
          'new_true_value': False
      }, {
          'initial_value': config_dict.placeholder(bool),
          'other_value': True,
          'true_value': None,
          'new_initial_value': False,
          'new_true_value': False
      })
  def testAnd(self, initial_value, other_value, true_value, new_initial_value,
              new_true_value):
    self._test_binary_operator(initial_value, other_value, operator.and_,
                               true_value, new_initial_value, new_true_value)

  @parameterized.parameters(
      {
          'initial_value': False,
          'other_value': False,
          'true_value': False,
          'new_initial_value': True,
          'new_true_value': True
      }, {
          'initial_value': ml_collections.FieldReference(True),
          'other_value': ml_collections.FieldReference(True),
          'true_value': True,
          'new_initial_value': False,
          'new_true_value': True
      }, {
          'initial_value': config_dict.placeholder(bool),
          'other_value': False,
          'true_value': None,
          'new_initial_value': True,
          'new_true_value': True
      })
  def testOr(self, initial_value, other_value, true_value, new_initial_value,
             new_true_value):
    self._test_binary_operator(initial_value, other_value, operator.or_,
                               true_value, new_initial_value, new_true_value)

  @parameterized.parameters(
      {
          'initial_value': False,
          'other_value': True,
          'true_value': True,
          'new_initial_value': True,
          'new_true_value': False
      }, {
          'initial_value': ml_collections.FieldReference(True),
          'other_value': ml_collections.FieldReference(True),
          'true_value': False,
          'new_initial_value': False,
          'new_true_value': True
      }, {
          'initial_value': config_dict.placeholder(bool),
          'other_value': True,
          'true_value': None,
          'new_initial_value': True,
          'new_true_value': False
      })
  def testXor(self, initial_value, other_value, true_value, new_initial_value,
              new_true_value):
    self._test_binary_operator(initial_value, other_value, operator.xor,
                               true_value, new_initial_value, new_true_value)

  @parameterized.parameters(
      {
          'initial_value': 3,
          'true_value': -3,
          'new_initial_value': -22,
          'new_true_value': 22
      }, {
          'initial_value': 15.3,
          'true_value': -15.3,
          'new_initial_value': -0.2,
          'new_true_value': 0.2
      }, {
          'initial_value': ml_collections.FieldReference(7),
          'true_value': ml_collections.FieldReference(-7),
          'new_initial_value': 123,
          'new_true_value': -123
      }, {
          'initial_value': config_dict.placeholder(int),
          'true_value': None,
          'new_initial_value': -6,
          'new_true_value': 6
      })
  def testNeg(self, initial_value, true_value, new_initial_value,
              new_true_value):
    self._test_unary_operator(initial_value, operator.neg, true_value,
                              new_initial_value, new_true_value)

  @parameterized.parameters(
      {
          'initial_value': config_dict.create(attribute=2),
          'true_value': 2,
          'new_initial_value': config_dict.create(attribute=3),
          'new_true_value': 3,
      },
      {
          'initial_value': config_dict.create(attribute={'a': 1}),
          'true_value': config_dict.create(a=1),
          'new_initial_value': config_dict.create(attribute={'b': 1}),
          'new_true_value': config_dict.create(b=1),
      },
      {
          'initial_value':
              ml_collections.FieldReference(config_dict.create(attribute=2)),
          'true_value':
              ml_collections.FieldReference(2),
          'new_initial_value':
              config_dict.create(attribute=3),
          'new_true_value':
              3,
      },
      {
          'initial_value': config_dict.placeholder(config_dict.ConfigDict),
          'true_value': None,
          'new_initial_value': config_dict.create(attribute=3),
          'new_true_value': 3,
      },
  )
  def testAttr(self, initial_value, true_value, new_initial_value,
               new_true_value):
    self._test_unary_operator(initial_value, lambda x: x.attr('attribute'),
                              true_value, new_initial_value, new_true_value)

  @parameterized.parameters(
      {
          'initial_value': 3,
          'true_value': 3,
          'new_initial_value': -101,
          'new_true_value': 101
      }, {
          'initial_value': -15.3,
          'true_value': 15.3,
          'new_initial_value': 7.3,
          'new_true_value': 7.3
      }, {
          'initial_value': ml_collections.FieldReference(-7),
          'true_value': ml_collections.FieldReference(7),
          'new_initial_value': 3,
          'new_true_value': 3
      }, {
          'initial_value': config_dict.placeholder(float),
          'true_value': None,
          'new_initial_value': -6.25,
          'new_true_value': 6.25
      })
  def testAbs(self, initial_value, true_value, new_initial_value,
              new_true_value):
    self._test_unary_operator(initial_value, operator.abs, true_value,
                              new_initial_value, new_true_value)

  def testToInt(self):
    self._test_unary_operator(25.3, lambda ref: ref.to_int(), 25, 27.9, 27)
    ref = ml_collections.FieldReference(64.7)
    ref = ref.to_int()
    self.assertEqual(ref.get(), 64)
    self.assertEqual(ref._field_type, int)

  def testToFloat(self):
    self._test_unary_operator(12, lambda ref: ref.to_float(), 12.0, 0, 0.0)

    ref = ml_collections.FieldReference(647)
    ref = ref.to_float()
    self.assertEqual(ref.get(), 647.0)
    self.assertEqual(ref._field_type, float)

  def testToString(self):
    self._test_unary_operator(12, lambda ref: ref.to_str(), '12', 0, '0')

    ref = ml_collections.FieldReference(647)
    ref = ref.to_str()
    self.assertEqual(ref.get(), '647')
    self.assertEqual(ref._field_type, str)

  def testSetValue(self):
    ref = ml_collections.FieldReference(1.0)
    other = ml_collections.FieldReference(3)
    ref_plus_other = ref + other

    self.assertEqual(ref_plus_other.get(), 4.0)

    ref.set(2.5)
    self.assertEqual(ref_plus_other.get(), 5.5)

    other.set(110)
    self.assertEqual(ref_plus_other.get(), 112.5)

    # Type checking
    with self.assertRaises(TypeError):
      other.set('this is a string')

    with self.assertRaises(TypeError):
      other.set(ml_collections.FieldReference('this is a string'))

    with self.assertRaises(TypeError):
      other.set(ml_collections.FieldReference(None, field_type=str))

  def testSetResult(self):
    ref = ml_collections.FieldReference(1.0)
    result = ref + 1.0
    second_result = result + 1.0

    self.assertEqual(ref.get(), 1.0)
    self.assertEqual(result.get(), 2.0)
    self.assertEqual(second_result.get(), 3.0)

    ref.set(2.0)
    self.assertEqual(ref.get(), 2.0)
    self.assertEqual(result.get(), 3.0)
    self.assertEqual(second_result.get(), 4.0)

    result.set(4.0)
    self.assertEqual(ref.get(), 2.0)
    self.assertEqual(result.get(), 4.0)
    self.assertEqual(second_result.get(), 5.0)

    # All references are broken at this point.
    ref.set(1.0)
    self.assertEqual(ref.get(), 1.0)
    self.assertEqual(result.get(), 4.0)
    self.assertEqual(second_result.get(), 5.0)

  def testTypeChecking(self):
    ref = ml_collections.FieldReference(1)
    string_ref = ml_collections.FieldReference('a')

    x = ref + string_ref
    with self.assertRaises(TypeError):
      x.get()

  def testNoType(self):
    self.assertRaisesRegex(TypeError, 'field_type should be a type.*',
                           ml_collections.FieldReference, None, 0)

  def testEqual(self):
    # Simple case
    ref1 = ml_collections.FieldReference(1)
    ref2 = ml_collections.FieldReference(1)
    ref3 = ml_collections.FieldReference(2)
    self.assertEqual(ref1, 1)
    self.assertEqual(ref1, ref1)
    self.assertEqual(ref1, ref2)
    self.assertNotEqual(ref1, 2)
    self.assertNotEqual(ref1, ref3)

    # ConfigDict inside FieldReference
    ref1 = ml_collections.FieldReference(ml_collections.ConfigDict({'a': 1}))
    ref2 = ml_collections.FieldReference(ml_collections.ConfigDict({'a': 1}))
    ref3 = ml_collections.FieldReference(ml_collections.ConfigDict({'a': 2}))
    self.assertEqual(ref1, ml_collections.ConfigDict({'a': 1}))
    self.assertEqual(ref1, ref1)
    self.assertEqual(ref1, ref2)
    self.assertNotEqual(ref1, ml_collections.ConfigDict({'a': 2}))
    self.assertNotEqual(ref1, ref3)

  def testLessEqual(self):
    # Simple case
    ref1 = ml_collections.FieldReference(1)
    ref2 = ml_collections.FieldReference(1)
    ref3 = ml_collections.FieldReference(2)
    self.assertLessEqual(ref1, 1)
    self.assertLessEqual(ref1, 2)
    self.assertLessEqual(0, ref1)
    self.assertLessEqual(1, ref1)
    self.assertGreater(ref1, 0)

    self.assertLessEqual(ref1, ref1)
    self.assertLessEqual(ref1, ref2)
    self.assertLessEqual(ref1, ref3)
    self.assertGreater(ref3, ref1)

  def testControlFlowError(self):
    ref1 = ml_collections.FieldReference(True)
    ref2 = ml_collections.FieldReference(False)

    with self.assertRaises(NotImplementedError):
      if ref1:
        pass
    with self.assertRaises(NotImplementedError):
      _ = ref1 and ref2
    with self.assertRaises(NotImplementedError):
      _ = ref1 or ref2
    with self.assertRaises(NotImplementedError):
      _ = not ref1


if __name__ == '__main__':
  absltest.main()
