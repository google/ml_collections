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

"""Tests for config_path."""

from absl.testing import absltest
from absl.testing import parameterized
from ml_collections.config_flags import config_path
from ml_collections.config_flags.tests import fieldreference_config
from ml_collections.config_flags.tests import mock_config


class ConfigPathTest(parameterized.TestCase):

  def test_list_extra_index(self):
    """Tries to index a non-indexable list element."""

    test_config = mock_config.get_config()
    with self.assertRaises(IndexError):
      config_path.get_value('dict.list[0][0]', test_config)

  def test_list_out_of_range_get(self):
    """Tries to access out-of-range value in list."""

    test_config = mock_config.get_config()
    with self.assertRaises(IndexError):
      config_path.get_value('dict.list[2][1]', test_config)

  def test_list_out_of_range_set(self):
    """Tries to override out-of-range value in list."""

    test_config = mock_config.get_config()
    with self.assertRaises(IndexError):
      config_path.set_value('dict.list[2][1]', test_config, -1)

  def test_reading_non_existing_key(self):
    """Tests reading non existing key from config."""

    test_config = mock_config.get_config()
    with self.assertRaises(KeyError):
      config_path.set_value('dict.not_existing_key', test_config, 1)

  def test_reading_setting_existing_key_in_dict(self):
    """Tests setting non existing key from dict inside config."""

    test_config = mock_config.get_config()
    with self.assertRaises(KeyError):
      config_path.set_value('dict.not_existing_key.key', test_config, 1)

  def test_empty_key(self):
    """Tests calling an empty key update."""

    test_config = mock_config.get_config()
    with self.assertRaises(ValueError):
      config_path.set_value('', test_config, None)

  def test_field_reference_types(self):
    """Tests whether types of FieldReference fields are valid."""
    test_config = fieldreference_config.get_config()

    paths = ['ref_nodefault', 'ref']
    paths_types = [int, int]

    config_types = [config_path.get_type(path, test_config) for path in paths]
    self.assertEqual(paths_types, config_types)

  @parameterized.parameters(
      ('float', float),
      ('integer', int),
      ('string', str),
      ('bool', bool),
      ('dict', dict),
      ('dict.float', float),
      ('dict.list', list),
      ('list', list),
      ('list[0]', int),
      ('object.float', float),
      ('object.integer', int),
      ('object.string', str),
      ('object.bool', bool),
      ('object.dict', dict),
      ('object.dict.float', float),
      ('object.dict.list', list),
      ('object.list', list),
      ('object.list[0]', int),
      ('object.tuple', tuple),
      ('object_reference.float', float),
      ('object_reference.integer', int),
      ('object_reference.string', str),
      ('object_reference.bool', bool),
      ('object_reference.dict', dict),
      ('object_reference.dict.float', float),
      ('object_copy.float', float),
      ('object_copy.integer', int),
      ('object_copy.string', str),
      ('object_copy.bool', bool),
      ('object_copy.dict', dict),
      ('object_copy.dict.float', float),
  )
  def test_types(self, path, path_type):
    """Tests whether various types of objects are valid."""
    test_config = mock_config.get_config()
    self.assertEqual(path_type, config_path.get_type(path, test_config))

if __name__ == '__main__':
  absltest.main()
