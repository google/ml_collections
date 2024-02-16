# Copyright 2024 The ML Collections Authors.
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

"""ConfigDict config file."""

from ml_collections import config_dict


class UnusableConfig(object):
  """Test against code assuming the semantics of attributes (such as `lock`).

  This class is to test that the flags implementation does not assume the
  semantics of attributes. This is to avoid code such as:

  ```python
  if hasattr(obj, lock):
    obj.lock()
  ```

  which will fail if `obj` has an attribute `lock` that does not behave in the
  way we expect.

  This class only has unusable attributes. There are two
  exceptions for which this class behaves normally:
  * Python's special functions which start and end with a double underscore.
  * `valid_attribute`, an attribute used to test the class.

  For other attributes, both `hasttr(obj, attr)` and `callable(obj, attr)` will
  return True. Calling `obj.attr` will return a function which takes no
  arguments and raises an AttributeError when called. For example, the `lock`
  example above will raise an AttributeError. The only valid action on
  attributes is assignment, e.g.

  ```python
  obj = UnusableConfig()
  obj.attr = 1
  ```

  In which case the attribute will keep its assigned value and become usable.
  """

  def __init__(self):
    self._valid_attribute = 1

  def __getattr__(self, attribute):
    """Get an arbitrary attribute.

    Returns a function which takes no arguments and raises an AttributeError,
    except for Python special functions in which case an AttributeError is
    directly raised.

    Args:
      attribute: A string representing the attribute's name.
    Returns:
      A function which raises an AttributeError when called.
    Raises:
      AttributeError: when the attribute is a Python special function starting
          and ending with a double underscore.
    """
    if attribute.startswith("__") and attribute.endswith("__"):
      raise AttributeError("UnusableConfig does not contain entry {}.".
                           format(attribute))

    def raise_attribute_error_fun():
      raise AttributeError(
          "{} is not a usable attribute of UnusableConfig".format(
              attribute))

    return raise_attribute_error_fun

  @property
  def valid_attribute(self):
    return self._valid_attribute

  @valid_attribute.setter
  def valid_attribute(self, value):
    self._valid_attribute = value

def get_config():
  """Returns a ConfigDict. Used for tests."""
  cfg = config_dict.ConfigDict()
  cfg.integer = 1
  cfg.reference = config_dict.FieldReference(1)
  cfg.list = [1, 2, 3]
  cfg.nested_list = [[1, 2, 3]]
  cfg.nested_configdict = config_dict.ConfigDict()
  cfg.nested_configdict.integer = 1
  cfg.unusable_config = UnusableConfig()

  return cfg
