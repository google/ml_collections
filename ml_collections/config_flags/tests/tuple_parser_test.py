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

"""Tests for ml_collection.config_flags.tuple_parser."""

from ml_collections.config_flags import tuple_parser
from absl.testing import absltest
from absl.testing import parameterized


class TupleParserTest(parameterized.TestCase):

  @parameterized.parameters(
      {'argument': 1, 'expected': (1,)},
      {'argument': (1, 2), 'expected': (1, 2)},
      {'argument': ('abc', 'def'), 'expected': ('abc', 'def')},
      {'argument': '1', 'expected': (1,)},
      {'argument': '"abc"', 'expected': ('abc',)},
      {'argument': '"abc",', 'expected': ('abc',)},
      {'argument': '1, "a"', 'expected': (1, 'a')},
      {'argument': '(1, "a")', 'expected': (1, 'a')},
      {'argument': '(1, "a", (2, 3))', 'expected': (1, 'a', (2, 3))},
      {'argument': ('abc*', 'def*'), 'expected': ('abc*', 'def*')},
      {'argument': '("abc*", "def*")', 'expected': ('abc*', 'def*')},
      {'argument': '("/abc",)', 'expected': ('/abc',)},
      {'argument': '("/abc*",)', 'expected': ('/abc*',)},
      {'argument': '("/abc/",)', 'expected': ('/abc/',)},
  )
  def test_tuple_parser_parse(self, argument, expected):
    parser = tuple_parser.TupleParser()
    self.assertEqual(parser.parse(argument), expected)

  @parameterized.parameters(
      {'argument': '1', 'expected': (1,)},
      {'argument': '"abc"', 'expected': ('abc',)},
      {'argument': '"abc",', 'expected': ('abc',)},
      {'argument': 'abc', 'expected': ('abc',)},
      {'argument': '1, "a"', 'expected': (1, 'a')},
      {'argument': '(1, "a")', 'expected': (1, 'a')},
      {'argument': '(1, "a", (2, 3))', 'expected': (1, 'a', (2, 3))},
      {'argument': '("abc*", "def*")', 'expected': ('abc*', 'def*')},
      {'argument': '"abc*", "def*"', 'expected': ('abc*', 'def*')},
      {'argument': '"abc*",', 'expected': ('abc*',)},
      {'argument': 'abc*', 'expected': ('abc*',)},
      {'argument': '"/abc",', 'expected': ('/abc',)},
      {'argument': '/abc*', 'expected': ('/abc*',)},
      {'argument': '/abc/', 'expected': ('/abc/',)},
  )
  def test_convert_str_to_tuple(self, argument, expected):
    self.assertEqual(tuple_parser._convert_str_to_tuple(argument), expected)

  @parameterized.parameters(
      'a b',
      'a,b*',
      '/a,b*',
      '/a,b',
      'a b',
  )
  def test_convert_str_to_tuple_bad_inputs(self, argument):
    with self.assertRaises(ValueError):
      tuple_parser._convert_str_to_tuple(argument)


if __name__ == '__main__':
  absltest.main()
