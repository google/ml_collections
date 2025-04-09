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

"""Classes for defining configurations of experiments and models.

This file defines the classes `ConfigDict` and `FrozenConfigDict`, which are
"dict-like" data structures with Lua-like access and some other nice features.
Together, they are supposed to be used as a main way of expressing
configurations of experiments and models.
"""

import abc
import collections
from collections import abc as collections_abc
import contextlib
import dataclasses
import difflib
import functools
import inspect
import json
import operator
import re
from typing import Any, Mapping, Optional, Tuple

from absl import logging
import yaml
from yaml import representer


# Workaround for https://github.com/yaml/pyyaml/issues/36. Classes that have
# `abc.ABCMeta` as a metaclass are incorrectly handled as objects. This results
# in the unbound `__reduce_ex__` method being called with the protocol version
# as its sole argument, resulting in a `TypeError`. A solution is to add a
# custom representer that represents `abc.ABCMeta` by name.
representer.Representer.add_representer(
    data_type=abc.ABCMeta,
    representer=representer.Representer.represent_name)


class RequiredValueError(Exception):
  pass


class MutabilityError(Exception):
  pass


class JSONDecodeError(Exception):
  pass

_NoneType = type(None)


def _is_callable_type(field_type):
  """Tries to ensure: `_is_callable_type(type(obj)) == callable(obj)`."""
  return any('__call__' in c.__dict__ for c in field_type.__mro__)


def _is_type_safety_violation(value, field_type):
  """Helper function for type safety exceptions.

  This function determines whether or not assigning a value to a field violates
  type safety.

  Args:
    value: The value to be assigned.
    field_type: Type of the field that we would like to assign value to.

  Returns:
    True if assigning value to field violates type safety, False otherwise.
  """
  # Allow None to override and be overridden by any type.
  if value is None or field_type == _NoneType:
    return False
  elif isinstance(value, field_type):
    return False
  else:
    # A callable can overridde a callable.
    return not (callable(value) and _is_callable_type(field_type))


def _safe_cast(value, field_type, type_safe=False):
  """Helper function to handle the exceptional type conversions.

  This function implements the following exceptions for type-checking rules:

  * An `int` will be converted to a `float` if overriding a `float` field.
  * Any string value can override a `str` or `unicode` field. The value is
  converted to `field_type`.
  * A `tuple` will be converted to a `list` if overriding a `list` field.
  * A `list` will be converted to a `tuple` if overriding `tuple` field.
  * Short and long integers are indistinguishable. The final value will always
  be a `long` if both types are present.

  Args:
    value: The value to be assigned.
    field_type: The type for the field that we would like to assign value to.
    type_safe: If True, the method will throw an error if the `value` is not of
        type `field_type` after safe type conversions.

  Returns:
    The converted type-safe version of the value if it is one of the cases
    described. Otherwise, return the value without conversion.

  Raises:
    TypeError: if types don't match  after safe type conversions.
  """
  original_value_type = type(value)

  # The int->float exception.
  if isinstance(value, int) and field_type is float:
    return float(value)

  # The unicode/string to string exception.
  if isinstance(value, str) and field_type is str:
    return field_type(value)

  # tuple<->list conversion. JSON serialization converts lists to tuples, so
  # we need this to avoid errors when overriding a list field with its
  # deserialized version. See b/34805906 for more details.
  if isinstance(value, tuple) and field_type is list:
    return list(value)
  if isinstance(value, list) and field_type is tuple:
    return tuple(value)

  if isinstance(value, int) and field_type is int:
    return value

  if type_safe and _is_type_safety_violation(value, field_type):
    raise TypeError(
        '{} is of original type {} and cannot be casted to type {}'
        .format(value, str(original_value_type), str(field_type)))
  return value


def _get_computed_value(value_or_fieldreference):
  if isinstance(value_or_fieldreference, FieldReference):
    return value_or_fieldreference.get()
  return value_or_fieldreference


def _parse_key(key: str) -> Tuple[str, Optional[int]]:
  """Parse a ConfigDict key into to it's initial part and index (if any)."""
  key = key.split('.')[0]
  index_match = re.match(r'(.*)\[([0-9]+)\]', key)
  if index_match:
    key = index_match.group(1)
    index = int(index_match.group(2))
  else:
    index = None
  return key, index


class _Op(collections.namedtuple('_Op', ['fn', 'args'])):
  """A named tuple representing a lazily computed op.

  The _Op named tuple has two fields:
    fn: The function to be applied.
    args: a tuple/list of arguments that are used with the op.
  """


@functools.total_ordering
class FieldReference:
  """Reference to a configuration element.

  Typed configuration element that can take a None default value. Example::

    from ml_collections import config_dict

    cfg_field = config_dict.FieldReference(0)
    cfg = config_dict.ConfigDict({
        'optional': configdict.FieldReference(None, field_type=str)
        'field': cfg_field,
        'nested': {'field': cfg_field}
    })

    with self.assertRaises(TypeError):
      cfg.optional = 10  # Raises an error because it's defined as an
                          # intfield.

    cfg.field = 1  # Changes the value of both cfg.field and cfg.nested.field.
    print(cfg)

  This class also supports lazy computation. Example::

    ref = config_dict.FieldReference(0)

    # Using ref in a standard operation returns another FieldReference. The new
    # reference ref_plus_ten will evaluate ref's value only when we call
    # ref_plus_ten.get()
    ref_plus_ten = ref + 10

    ref.set(3)  # change ref's value
    print(ref_plus_ten.get())  # Prints 13 because ref's value is 3

    ref.set(-2)  # change ref's value again
    print(ref_plus_ten.get())  # Prints 8 because ref's value is -2
  """

  def __init__(self, default, field_type=None, op=None, required=False):
    """Creates an instance of FieldReference.

    Args:
      default: Default value.
      field_type: Type for the values contained by the configuration element. If
          None the type will be inferred from the default value. This value is
          used as the second argument in calls to isinstance, so it has to
          follow that function's requirements (class, type or a tuple containing
          classes, types or tuples).
      op: An optional operation that is applied to the underlying value when
          `get()` is called.
      required: If True, the `get()` method will raise an error if the reference
           does not contain a value. This argument has no effect when a default
           value is provided. Setting this to True will raise an error if `op`
           is not None.

    Raises:
      TypeError: If field_type is not None and is different from the type of the
          default value.
      ValueError: If both the default value and field_type is None.
    """
    if field_type is None:
      if default is None:
        raise ValueError('Default value cannot be None if field_type is None')
      elif isinstance(default, FieldReference):
        field_type = default.get_type()
      else:
        field_type = type(default)
    else:
      # Early error when field_type doesn't a structure compatible with
      # isinstance (class, type or tuple containing classes, types or tuples.
      # The easiest way to check this is call isinstance and catch TypeError
      # exceptions.
      try:
        isinstance(None, field_type)
      except TypeError:
        raise TypeError('field_type should be a type, not {}'
                        .format(type(field_type)))

    self._field_type = field_type
    self.set(default)

    if required and op is not None:
      raise ValueError('Cannot set required to True if op is not None')
    self._required = required
    self._ops = [] if op is None else [op]

  def has_cycle(self, visited=None):
    """Finds cycles in the reference graph.

    Args:
      visited: Set containing the ids of all visited nodes in the graph. The
          default value is the empty set.

    Returns:
      True if there is a cycle in the reference graph.
    """
    visited = visited or set()

    if id(self) in visited:
      return True

    visited.add(id(self))

    # Verify the reference to the parent FieldReference doesn't introduce a
    # cycle.
    value = self._value
    if isinstance(value, FieldReference) and value.has_cycle(visited.copy()):
      return True

    # Verify references in the operator arguments don't introduce cycles.
    for op in self._ops:
      for arg in op.args:
        if isinstance(arg, FieldReference) and arg.has_cycle(visited.copy()):
          return True

    return False

  def set(self, value, type_safe=True):
    """Overwrites the value pointed by a FieldReference.

    Args:
      value: New value.
      type_safe: Check that old and new values are of the same type.

    Raises:
      TypeError: If type_safe is true and old and new values are not of the same
          type.
      MutabilityError: If a cycle is found in the reference graph.
    """
    # Disable ops.
    self._ops = []

    if value is None:
      self._value = None
    elif isinstance(value, FieldReference):
      if type_safe and not issubclass(value.get_type(), self.get_type()):
        raise TypeError('Reference is of type {} but should be of type {}'
                        .format(value.get_type(), self.get_type()))
      old_value = getattr(self, '_value', None)
      self._value = value
      if self.has_cycle():
        self._value = old_value
        raise MutabilityError('Found cycle in reference graph.')
    else:
      # TODO(sergomez): Update reference type.
      self._value = _safe_cast(value, self._field_type, type_safe)

  def empty(self):
    """Returns True if the reference points to a None value."""
    return self._value is None

  def get(self):
    """Gets the value of the `FieldReference` object.

    This will dereference `_pointer` and apply all ops to its value.

    Returns:
      The result of applying all ops to the dereferenced pointer.

    Raises:
      RequiredValueError: if `required` is True and the underlying value for the
          reference is False.
    """
    if self._required and self._value is None:
      raise RequiredValueError('None value found in required reference')

    value = _get_computed_value(self._value)
    for op in self._ops:
      # Dereference any FieldReference objects
      args = [_get_computed_value(arg) for arg in op.args]
      if value is None or None in args:
        value = None
        logging.debug('Cannot apply `%s` to `None`; skipping.', op)
      else:
        value = op.fn(value, *args)
      value = _get_computed_value(value)
    return value

  def get_type(self):
    return self._field_type

  def __eq__(self, other):
    if isinstance(other, FieldReference):
      return self.get() == other.get()
    else:
      return self.get() == other

  def __le__(self, other):
    if isinstance(other, FieldReference):
      return self.get() <= other.get()
    else:
      return self.get() <= other

  # Make FieldReference unhashable (as it's mutable).
  __hash__ = None

  def _apply_op(self, fn, *args):
    args = [_safe_cast(arg, self._field_type) for arg in args]
    return FieldReference(
        self,
        field_type=self._field_type,
        op=_Op(fn, args))

  def _apply_cast_op(self, field_type):
    """Apply a cast op that changes the field_type of this FieldReference.

    `_apply_op` assumes that the `field_type` does not change after the op is
    applied whereas `_apply_cast_op` generates a FieldReference with casted
    field_type.

    Since the signature is `fn(value, *args)` we need to ignore `value`
      which now contains a unused default value of field_type.

    Args:
      field_type: data type to cast to.

    Returns:
      A new FieldReference with of `field_type`.
    """
    return FieldReference(
        field_type(),  # Creates unused default value matching `field_type`.
        field_type=field_type,
        op=_Op(lambda _, val: field_type(val),  # `fn` ignores `field_type()`.
               [self]),
    )

  def identity(self):
    return self._apply_op(lambda x: x)

  def attr(self, attr_name):
    return self._apply_op(operator.attrgetter(attr_name))

  def __add__(self, other):
    return self._apply_op(operator.add, other)

  def __radd__(self, other):
    radd = functools.partial(operator.add, other)
    return self._apply_op(radd)

  def __sub__(self, other):
    return self._apply_op(operator.sub, other)

  def __rsub__(self, other):
    rsub = functools.partial(operator.sub, other)
    return self._apply_op(rsub)

  def __mul__(self, other):
    return self._apply_op(operator.mul, other)

  def __rmul__(self, other):
    rmul = functools.partial(operator.mul, other)
    return self._apply_op(rmul)

  def __div__(self, other):
    return self._apply_op(operator.truediv, other)

  def __rdiv__(self, other):
    rdiv = functools.partial(operator.truediv, other)
    return self._apply_op(rdiv)

  def __truediv__(self, other):
    return self._apply_op(operator.truediv, other)

  def __rtruediv__(self, other):
    rtruediv = functools.partial(operator.truediv, other)
    return self._apply_op(rtruediv)

  def __floordiv__(self, other):
    return self._apply_op(operator.floordiv, other)

  def __rfloordiv__(self, other):
    rfloordiv = functools.partial(operator.floordiv, other)
    return self._apply_op(rfloordiv)

  def __pow__(self, other):
    return self._apply_op(operator.pow, other)

  def __mod__(self, other):
    return self._apply_op(operator.mod, other)

  def __and__(self, other):
    return self._apply_op(operator.and_, other)

  def __or__(self, other):
    return self._apply_op(operator.or_, other)

  def __xor__(self, other):
    return self._apply_op(operator.xor, other)

  def __neg__(self):
    return self._apply_op(operator.neg)

  def __abs__(self):
    return self._apply_op(operator.abs)

  def to_int(self):
    return self._apply_cast_op(int)

  def to_float(self):
    return self._apply_cast_op(float)

  def to_str(self):
    return self._apply_cast_op(str)

  def __setstate__(self, state):
    self._value = state['_value']
    self._field_type = state['_field_type']
    self._ops = state['_ops']
    # TODO(sergomez): Remove default for _required (and potentially the whole
    #                 __setstate__ method) after June 2019.
    self._required = state.get('_required', False)

  def __nonzero__(self):
    raise NotImplementedError(
        'FieldReference cannot be used for control flow. For boolean '
        'operations use "&" (logical "and") or "|" (logical "or").')

  def __bool__(self):
    raise NotImplementedError(
        'FieldReference cannot be used for control flow. For boolean '
        'operations use "&" (logical "and") or "|" (logical "or").')


def _configdict_fill_seed(seed, initial_dictionary, visit_map=None):
  """Fills an empty ConfigDict without copying previously visited nodes.

  Turns seed (an empty ConfigDict) into a ConfigDict version of
  initial_dictionary. Avoids infinite looping on a self-referencing
  initial_dictionary because if a value of initial_dictionary has been
  previously visited, that value is not re-converted to a ConfigDict. If a
  FieldReference is encountered which contains a dict or FrozenConfigDict, its
  contents will be converted to ConfigDict.

  Note: As described in the __init__() documentation, this will not
  replicate the structure of initial_dictionary if it contains
  self-references within lists, tuples, or other types. There is no warning
  or error in this case.

  Args:
    seed: Empty ConfigDict, to be filled in.
    initial_dictionary: The template on which seed is built. May be of type
        dict, ConfigDict or FrozenConfigDict.
    visit_map: Dictionary from memory addresses to values, storing the
        ConfigDict versions of dictionary values. visit_map need not contain
        (id(initial_dictionary), seed) as a key/value pair.

  Raises:
    TypeError: If seed is not a ConfigDict.
    ValueError: If seed is not an empty ConfigDict.
  """
  # These should be impossible to raise, since the public call-site in
  # __init__() pass in valid input, as does this method recursively.
  assert isinstance(seed, ConfigDict)
  assert not seed

  visit_map = visit_map or {}
  visit_map[id(initial_dictionary)] = seed

  if isinstance(initial_dictionary, ConfigDict):
    iteritems = initial_dictionary.iteritems(preserve_field_references=True)
  else:
    iteritems = initial_dictionary.items()

  for key, value in iteritems:
    if id(value) in visit_map:
      value = visit_map[id(value)]

    elif (isinstance(value, FieldReference) and value.get_type() is dict
          and seed.convert_dict):
      # If the reference is empty, we don't have to do dict -> ConfigDict
      # conversion.
      # Calling get() on an empty required reference would raise an error so we
      # need a special case for this.
      if value.empty():
        pass
      elif id(value.get()) in visit_map:
        value.set(visit_map[id(value.get())], False)
      else:
        value_cd = ConfigDict(
            type_safe=seed.is_type_safe,
            allow_dotted_keys=seed.allow_dotted_keys,
        )
        _configdict_fill_seed(value_cd, value.get(), visit_map)
        value.set(value_cd, False)

    elif isinstance(value, dict) and seed.convert_dict:
      value_cd = ConfigDict(
          type_safe=seed.is_type_safe,
          allow_dotted_keys=seed.allow_dotted_keys,
      )
      _configdict_fill_seed(value_cd, value, visit_map)
      value = value_cd

    elif isinstance(value, FrozenConfigDict):
      value = ConfigDict(value, allow_dotted_keys=seed.allow_dotted_keys)

    seed.__setattr__(key, value)


class ConfigDict:
  # pylint: disable=line-too-long
  """Base class for configuration objects used in DeepMind.

  This is a container for configurations. It behaves similarly to Lua tables.
  Specifically:

  - it has dot-based access as well as dict-style key access,
  - it is type safe (once a value is set one cannot change its type).

  Typical usage example::

    from ml_collections import config_dict

    cfg = config_dict.ConfigDict()
    cfg.float_field = 12.6
    cfg.integer_field = 123
    cfg.another_integer_field = 234
    cfg.nested = config_dict.ConfigDict()
    cfg.nested.string_field = 'tom'

    print(cfg)

  Config dictionaries can also be used to pass named arguments to functions::

    from ml_collections import config_dict

    def print_point(x, y):
      print "({},{})".format(x, y)

    point = config_dict.ConfigDict()
    point.x = 1
    point.y = 2
    print_point(**point)

  Note that, depending on your use case, it may be easier to use the `create`
  function in this package to construct a `ConfigDict`:

    from ml_collections import config_dict
    point = config_dict.create(x=1, y=2)

  Differently from standard `dicts`, `ConfigDicts` also have the nice property
  that iterating over them is deterministic, in a fashion similar to
  `collections.OrderedDicts`.
  """
  # pylint: enable=line-too-long

  # Loosen the static type checking requirements.
  _HAS_DYNAMIC_ATTRIBUTES = True

  # For auto-complete
  _allow_dotted_keys: bool
  _sort_keys: bool

  def __init__(
      self,
      initial_dictionary: Optional[Mapping[str, Any]] = None,
      type_safe: bool = True,
      convert_dict: bool = True,
      *,
      allow_dotted_keys: bool = False,
      sort_keys: bool = True,
  ):
    """Creates an instance of ConfigDict.

    Warning: In most cases, this faithfully reproduces the reference structure
    of initial_dictionary, even if initial_dictionary is self-referencing.
    However, unexpected behavior occurs if self-references are contained within
    list, tuple, or custom types. For example::

      d = {}
      d['a'] = d
      d['b'] = [d]
      cd = ConfigDict(d)
      cd.a    # refers to cd, type ConfigDict. Expected behavior.
      cd.b    # refers to d, type dict. Unexpected behavior.

    Warning: FieldReference values may be changed. If initial_dictionary
    contains a FieldReference with a value of type dict or FrozenConfigDict,
    that value is converted to ConfigDict.

    Args:
      initial_dictionary: May be one of the following:  1) dict. In this case,
        all values of initial_dictionary that are dictionaries are also be
        converted to ConfigDict. However, dictionaries within values of non-dict
        type are untouched.  2) ConfigDict. In this case, all attributes are
        uncopied, and only the top-level object (self) is re-addressed. This is
        the same behavior as Python dict, list, and tuple.  3) FrozenConfigDict.
        In this case, initial_dictionary is converted to a ConfigDict version of
        the initial dictionary for the FrozenConfigDict (reversing any
        mutability changes FrozenConfigDict made).
      type_safe: If set to True, once an attribute value is assigned, its type
        cannot be overridden without .ignore_type() context manager (default:
        True).
      convert_dict: If set to True, all dict used as value in the ConfigDict
        will automatically be converted to ConfigDict (default: True).
      allow_dotted_keys: If set to True, keys can contain `.`. Integer and float
        keys are always allowed to have dots.
      sort_keys: If `True` (default), keys are sorted in alphabetical order.
    """

    if isinstance(initial_dictionary, FrozenConfigDict):
      initial_dictionary = initial_dictionary.as_configdict()

    super(ConfigDict, self).__setattr__('_fields', {})
    super(ConfigDict, self).__setattr__('_locked', False)
    super(ConfigDict, self).__setattr__('_type_safe', type_safe)
    super(ConfigDict, self).__setattr__('_convert_dict', convert_dict)
    super(ConfigDict, self).__setattr__('_allow_dotted_keys', allow_dotted_keys)
    super(ConfigDict, self).__setattr__('_sort_keys', sort_keys)

    if initial_dictionary is not None:
      _configdict_fill_seed(self, initial_dictionary)

    if isinstance(initial_dictionary, ConfigDict):
      super(ConfigDict, self).__setattr__('_locked',
                                          initial_dictionary.is_locked)
      super(ConfigDict, self).__setattr__('_type_safe',
                                          initial_dictionary.is_type_safe)

  @property
  def is_type_safe(self) -> bool:
    """Returns True if config dict is type safe."""
    return self._type_safe

  @property
  def allow_dotted_keys(self) -> bool:
    """Returns True if keys can contain `.`."""
    return self._allow_dotted_keys

  @property
  def convert_dict(self):
    """Returns True if it is converting dicts to ConfigDict automatically."""
    return self._convert_dict

  def lock(self) -> 'ConfigDict':
    """Locks object, preventing user from adding new fields.

    Returns:
      self
    """

    if self.is_locked:
      return self

    super(ConfigDict, self).__setattr__('_locked', True)
    for field in self._fields:
      element = self._fields[field]
      element = _get_computed_value(element)

      if isinstance(element, ConfigDict):
        element.lock()
    return self

  @property
  def is_locked(self) -> bool:
    """Returns True if object is locked."""
    return self._locked

  def unlock(self) -> 'ConfigDict':
    """Grants user the ability to add new fields to ConfigDict.

    In most cases, the unlocked() context manager should be preferred to the
    direct use of the unlock method.

    Returns:
      self
    """
    super(ConfigDict, self).__setattr__('_locked', False)
    for element in self._fields.values():
      element = _get_computed_value(element)

      if isinstance(element, ConfigDict) and element.is_locked:
        element.unlock()
    return self

  def get(self, key: str, default=None):
    """Returns value if key is present, or a user defined value otherwise."""
    try:
      return self[key]
    except KeyError:
      return default

  # TODO(sergomez): replace this with get_oneway_ref. The first step is to log
  #   usage patterns of this. How many users are overriding the value of the
  #   reference returned by this and expect the referenced field to change too?
  def get_ref(self, key):
    """Returns a FieldReference initialized on key's value."""
    field = self._fields[key]
    if field is None:
      raise ValueError('Cannot create reference to a field whose value is None')
    if not isinstance(field, FieldReference):
      field = FieldReference(field)
      with self.ignore_type():
        self[key] = field
    return field

  def get_oneway_ref(self, key):
    """Returns a one-way FieldReference.

    Example::

      cfg = config_dict.ConfigDict(dict(a=1))
      cfg.b = cfg.get_oneway_ref('a')

      cfg.a = 2
      print(cfg.b)  # 2

      cfg.b = 3
      print(cfg.a)  # 2 (would have been 3 if using get_ref())
      print(cfg.b)  # 3

    Args:
      key: Key for field we want to reference.
    """
    # Using the result of applying an operation on the reference means that
    # calling set() on this object won't propagate the new value up the
    # reference chain.
    return self.get_ref(key).identity()

  def items(self, preserve_field_references=False):
    """Returns list of dictionary key, value pairs, sorted by key.

    Args:
      preserve_field_references: (bool) Whether to preserve FieldReferences if
        the ConfigDict has them. By default, False: any FieldReferences will be
        resolved in the result.

    Returns:
      The key, value pairs in the config, sorted by key.
    """
    if preserve_field_references:
      return self._ordered_fields.items()
    else:
      return [(k, self[k]) for k in self._ordered_fields]

  @property
  def _ordered_fields(self):
    """Returns ordered dict shallow cast of _fields member."""
    if self._sort_keys:
      return collections.OrderedDict(sorted(self._fields.items()))
    else:
      return self._fields

  def iteritems(self, preserve_field_references=False):
    """Deterministically iterates over dictionary key, value pairs.

    Args:
      preserve_field_references: (bool) Whether to preserve FieldReferences if
        the ConfigDict has them. By default, False: any FieldReferences will be
        resolved in the result.
    Yields:
      The key, value pairs in the config, sorted by key.
    """
    for k in self._ordered_fields:
      if preserve_field_references:
        yield k, self._fields[k]
      else:
        yield k, self[k]

  def _ensure_mutability(self, attribute):
    if attribute in dir(super(ConfigDict, self)):
      raise KeyError('{} cannot be overridden.'.format(attribute))

  def __setattr__(self, attribute, value):
    try:
      self._ensure_mutability(attribute)
      self[attribute] = value
    except KeyError as e:
      raise AttributeError(e)

  def __delattr__(self, attribute):
    try:
      self._ensure_mutability(attribute)
      del self[attribute]
    except KeyError as e:
      raise AttributeError(e)

  def __getattr__(self, attribute):
    try:
      return self[attribute]
    except KeyError as e:
      raise AttributeError(e)

  def __setitem__(self, key, value):
    if (
        not self._allow_dotted_keys
        and not isinstance(key, (int, float))
        and '.' in key
    ):
      raise ValueError(
          'ConfigDict does not accept dots in field names (when'
          f' `allow_dotted_keys=False`), but the key {key} contains one.'
      )

    if self.is_locked and key not in self._fields:
      error_msg = ('Key "{}" does not exist and cannot be added since the '
                   'config is locked. Other fields present: "{}"')
      raise KeyError(
          self._generate_did_you_mean_message(
              key, error_msg.format(key, self._fields)
          )
      )

    if key in self._fields:
      field = self._fields[key]
      try:
        # Updating a FieldReference will update all it's copy.
        if isinstance(field, FieldReference):
          field.set(value, self._type_safe)
          return

        if not _should_skip_type_check(field, value):
          if isinstance(value, dict) and self._convert_dict:
            value = type(self)(value, type_safe=self._type_safe)
          value = _safe_cast(value, type(field), self._type_safe)
      except TypeError as e:
        raise TypeError(
            f"Could not override field '{key}' (reference). {e}"
        ) from None

    if self._convert_dict:
      if isinstance(value, dict):
        value = ConfigDict(value, self._type_safe)
      elif isinstance(value, FieldReference):
        # TODO(sergomez): We should consider using value.get_type().
        ref_type = _NoneType if value.empty() else type(value.get())
        if ref_type is dict or ref_type is FrozenConfigDict:
          value_cd = ConfigDict(value.get(), self._type_safe)
          value.set(value_cd, False)

    self._fields[key] = value

  def _generate_did_you_mean_message(self, request, message=''):
    if isinstance(request, (int, float)):
      return message
    # Keys can be integers,... so normalizing them to strings.
    keys = tuple(str(k) for k in self.keys())
    matches = difflib.get_close_matches(request, keys, 1, 0.75)
    if matches:
      if message:
        message += '\n'
      message += 'Did you mean "{}" instead of "{}"?'.format(matches[0],
                                                             request)
    return message

  def __delitem__(self, key: str):
    if self.is_locked:
      raise KeyError('This ConfigDict is locked, you have to unlock it before '
                     'trying to delete a field.')

    if (
        not self._allow_dotted_keys
        and not isinstance(key, (int, float))
        and '.' in key
    ):
      # As per the check in __setitem__ above, keys cannot contain dots.
      # Hence, we can use dots to do recursive calls.
      key, rest = key.split('.', 1)
      del self[key][rest]
      return

    try:
      del self._fields[key]
    except KeyError as e:
      raise KeyError(self._generate_did_you_mean_message(key, str(e)))

  def __getitem__(self, key: str):
    if (
        not self._allow_dotted_keys
        and not isinstance(key, (int, float))
        and '.' in key
    ):
      # As per the check in __setitem__ above, keys cannot contain dots.
      # Hence, we can use dots to do recursive calls.
      key, rest = key.split('.', 1)
      return self[key][rest]

    try:
      field = self._fields[key]
      if isinstance(field, FieldReference):
        return field.get()
      else:
        return field
    except KeyError as e:
      raise KeyError(self._generate_did_you_mean_message(key, str(e)))

  def __contains__(self, key: str):
    return key in self._fields

  def __repr__(self) -> str:
    # We want __repr__ to always run without throwing an exception,
    # even if the config dict is not YAML serialisable.
    try:
      return yaml.dump(self.to_dict(preserve_field_references=True),
                       default_flow_style=False)
    except Exception:  # pylint: disable=broad-except
      return repr(self.to_dict())

  def __str__(self) -> str:
    # We want __str__ to always run without throwing an exception,
    # even if the config dict is not YAML serialisable.
    try:
      return yaml.dump(self.to_dict())
    except Exception:  # pylint: disable=broad-except
      return str(self.to_dict())

  def keys(self):
    """Returns the sorted list of all the keys defined in a config."""
    return list(self._ordered_fields.keys())

  def iterkeys(self):
    """Deterministically iterates over dictionary keys, in sorted order."""
    return self._ordered_fields.keys()

  def values(self, preserve_field_references=False):
    """Returns the list of all values in a config, sorted by their keys.

    Args:
      preserve_field_references: (bool) Whether to preserve FieldReferences if
        the ConfigDict has them. By default, False: any FieldReferences will be
        resolved in the result.
    Returns:
      The values in the config, sorted by their corresponding keys.
    """
    if preserve_field_references:
      return list(self._ordered_fields.values())
    else:
      return [self[k] for k in self._ordered_fields]

  def itervalues(self, preserve_field_references=False):
    """Deterministically iterates over values in a config, sorted by their keys.

    Args:
      preserve_field_references: (bool) Whether to preserve FieldReferences if
        the ConfigDict has them. By default, False: any FieldReferences will be
        resolved in the result.
    Yields:
      The values in the config, sorted by their corresponding keys.
    """
    for k in self._ordered_fields:
      if preserve_field_references:
        yield self._fields[k]
      else:
        yield self[k]

  def __dir__(self):
    return self.keys() + dir(ConfigDict)

  def __len__(self):
    return self._ordered_fields.__len__()

  def __iter__(self):
    return self._ordered_fields.__iter__()

  def __eq__(self, other):
    """Override the default Equals behavior."""
    if isinstance(other, self.__class__):
      same = self._fields == other._fields
      same &= self._locked == other.is_locked
      same &= self._type_safe == other.is_type_safe
      return same
    return False

  def __ne__(self, other):
    """Define a non-equality test."""
    return not self.__eq__(other)

  def eq_as_configdict(self, other):
    """Type-invariant equals.

    This is like `__eq__`, except it does not distinguish `FrozenConfigDict`
    from `ConfigDict`. For example::

      cd = ConfigDict()
      fcd = FrozenConfigDict()
      fcd.eq_as_configdict(cd)  # Returns True

    Args:
      other: Object to compare self to.

    Returns:
      same: `True` if `self == other` after conversion to `ConfigDict`.
    """
    if isinstance(other, ConfigDict):
      return ConfigDict(self) == ConfigDict(other)
    else:
      return False

  # Make ConfigDict unhashable
  __hash__ = None

  def to_yaml(self, **kwargs):
    """Returns a YAML representation of the object.

    ConfigDict serializes types of fields as well as the values of fields
    themselves. Deserializing the YAML representation hence requires using
    YAML's UnsafeLoader:

    ```
    yaml.load(cfg.to_yaml(), Loader=yaml.UnsafeLoader)
    ```

    or equivalently:

    ```
    yaml.unsafe_load(cfg.to_yaml())
    ```

    Please see the PyYAML documentation and https://msg.pyyaml.org/load for more
    details on the consequences of this.

    Args:
      **kwargs: Keyword arguments for yaml.dump.

    Returns:
      YAML representation of the object.
    """
    return yaml.dump(self, **kwargs)

  def _json_dumps_wrapper(self, **kwargs):
    """Wrapper for json.dumps() method.

    Produces a more informative error message when there is a problem with
    string encodings in the ConfigDict.

    Args:
      **kwargs: Keyword arguments for json.dumps.

    Returns:
      JSON representation of the object.

    Raises:
      JSONDecodeError: If there is a problem with string encodings.
    """
    try:
      return json.dumps(self, **kwargs)
    except UnicodeDecodeError as error:
      # Re-raise exception with more informative error message.
      new_message = (
          'Decoding error. Make sure all strings in your ConfigDict use ASCII-'
          'compatible encodings. See '
          'https://docs.python.org/2.7/howto/unicode.html#the-unicode-type '
          'for details. Original error message: {}'.format(error))
      raise JSONDecodeError(new_message)

  def to_json(self, json_encoder_cls=None, **kwargs):
    """Returns a JSON representation of the object, fails if there is a cycle.

    Args:
      json_encoder_cls: An optional JSON encoder class to customize JSON
        serialization.
      **kwargs: Keyword arguments for json.dumps. They cannot contain "cls"
          as this method specifies it on its own.

    Returns:
      JSON representation of the object.

    Raises:
      TypeError: If self contains set, frozenset, custom type fields or any
          other objects that are not JSON serializable.
    """
    json_encoder_cls = json_encoder_cls or CustomJSONEncoder
    return self._json_dumps_wrapper(cls=json_encoder_cls, **kwargs)

  def to_json_best_effort(self, **kwargs):
    """Returns a best effort JSON representation of the object.

    Tries to serialize objects not inherently supported by JSON encoder. This
    may result in the configdict being partially serialized, skipping the
    unserializable bits. Ensures that no errors are thrown. Fails if there is a
    cycle.

    Args:
      **kwargs: Keyword arguments for json.dumps. They cannot contain "cls"
          as this method specifies it on its own.

    Returns:
      JSON representation of the object.
    """
    return self._json_dumps_wrapper(cls=_BestEffortCustomJSONEncoder, **kwargs)

  def to_dict(self, visit_map=None, preserve_field_references=False):
    """Converts ConfigDict to regular dict recursively with valid references.

    By default, the output dict will not contain FieldReferences, any present
    in the ConfigDict will be resolved. However, if `preserve_field_references`
    is True, the output dict will contain FieldReferences where the original
    ConfigDict has them. They will not be the same as the ConfigDict's, and
    their ops will be applied and dropped.

    Note: As with __eq__() and __init__(), this may not behave as expected on a
    ConfigDict with self-references contained in lists, tuples, or custom types.

    Args:
      visit_map: A mapping from object ids to their dict representation. Method
          is recursive in nature, and it will call ".to_dict(visit_map)" on each
          encountered object, unless it is already in visit_map.
      preserve_field_references: (bool) Whether the output dict should have
        FieldReferences if the ConfigDict has them. By default, False: any
        FieldReferences will be resolved and the result will go to the dict.
    Returns:
      Dictionary with the same values and references structure as a calling
          ConfigDict.
    """
    visit_map = visit_map or {}
    dict_self = {}
    visit_map[id(self)] = dict_self

    for key in self:
      if (isinstance(self._fields[key], FieldReference)
          and preserve_field_references):
        reference = self._fields[key]
        value = reference.get()
      else:
        value = self[key]
        reference = value

      if id(reference) in visit_map:
        dict_self[key] = visit_map[id(reference)]
      elif isinstance(value, ConfigDict):
        if isinstance(reference, FieldReference):
          # Create a new reference of type dict instead of ConfigDict.
          old_reference = reference
          reference = FieldReference({}, dict)
          visit_map[id(old_reference)] = reference
          reference.set(value.to_dict(visit_map, preserve_field_references))
        else:
          reference = value.to_dict(visit_map, preserve_field_references)
        dict_self[key] = reference
      else:
        if isinstance(reference, FieldReference):
          # Create a new reference to put in the new dict, which will be
          # reused whenever we find the same original reference.
          # Notice that ops are lost in the copy, but they are applied when
          # we do old_reference.get().
          old_reference = reference
          # Disable type safety since value in the field reference might have
          # been previously set with type safety disabled (e.g. ignore_type
          # context, as in b/119393923).
          reference = FieldReference(None, old_reference.get_type())
          reference.set(old_reference.get(), type_safe=False)
          visit_map[id(old_reference)] = reference
        dict_self[key] = reference

    return dict_self

  def copy_and_resolve_references(self, visit_map=None):
    """Returns a ConfigDict copy with FieldReferences replaced by values.

    If the object is a FrozenConfigDict, the copy returned is also a
    FrozenConfigDict. However, note that FrozenConfigDict should already have
    FieldReferences resolved to values, so this method effectively produces
    a deep copy.

    Note: As with __eq__() and __init__(), this may not behave as expected on a
    ConfigDict with self-references contained in lists, tuples, or custom types.

    Args:
      visit_map: A mapping from ConfigDict object ids to their copy. Method
          is recursive in nature, and it will call
          ".copy_and_resolve_references(visit_map)" on each encountered object,
          unless it is already in visit_map.

    Returns:
      ConfigDict copy with previous FieldReferences replaced by values.
    """
    visit_map = visit_map or {}
    config_dict_copy = self.__class__()
    super(ConfigDict, config_dict_copy).__setattr__('_convert_dict',
                                                    self.convert_dict)
    super(ConfigDict, config_dict_copy).__setattr__('_allow_dotted_keys',
                                                    self.allow_dotted_keys)

    visit_map[id(self)] = config_dict_copy

    for key, value in self._fields.items():
      if isinstance(value, FieldReference):
        value = value.get()

      if id(value) in visit_map:
        value = visit_map[id(value)]
      elif isinstance(value, ConfigDict):
        value = value.copy_and_resolve_references(visit_map)

      if isinstance(self, FrozenConfigDict):
        config_dict_copy._frozen_setattr(  # pylint:disable=protected-access
            key, value)
      else:
        config_dict_copy[key] = value

    super(ConfigDict, config_dict_copy).__setattr__(
        '_locked', self.is_locked)
    super(ConfigDict, config_dict_copy).__setattr__(
        '_type_safe', self.is_type_safe)
    return config_dict_copy

  def __setstate__(self, state):
    """Recreates ConfigDict from its dict representation."""
    self.__init__()
    super(ConfigDict, self).__setattr__('_type_safe', state['_type_safe'])
    super(ConfigDict, self).__setattr__('_convert_dict',
                                        state.get('_convert_dict', True))
    for field in state['_fields']:
      self[field] = state['_fields'][field]
    if state['_locked']:
      # Don't call self.lock() here as that recurses into its children. With
      # circular references that can lead to attempts to lock other instances
      # in the hierarchy before their __setstate__ method has been called.
      super(ConfigDict, self).__setattr__('_locked', True)

  @contextlib.contextmanager
  def unlocked(self):
    """Context manager which temporarily unlocks a ConfigDict."""
    was_locked = self._locked
    if was_locked: self.unlock()
    try:
      yield self
    finally:
      if was_locked: self.lock()

  @contextlib.contextmanager
  def ignore_type(self):
    """Context manager which temporarily turns off type safety recursively."""
    original_type_safety = self._type_safe

    managers = []
    visited = set()
    fields = list(self._fields.items())
    while fields:
      field = fields.pop()
      if id(field) in visited:
        continue

      visited.add(id(field))
      if isinstance(field, ConfigDict):
        managers.append(field.ignore_type)
      # Recursively add elements in nested collections.
      elif isinstance(field, collections_abc.Mapping):
        fields.extend(field.items())
      elif isinstance(field, (collections_abc.Sequence, collections_abc.Set)):
        fields.extend(field)

    super(ConfigDict, self).__setattr__('_type_safe', False)
    try:
      with contextlib.ExitStack() as stack:
        for manager in managers:
          stack.enter_context(manager())
        yield self
    finally:
      super(ConfigDict, self).__setattr__('_type_safe', original_type_safety)

  def get_type(self, key):
    """Returns type of the field associated with a key."""

    # We access the field outside of the if/else statement to raise in all cases
    # AttributeErrors (potentially including "did you mean" messages) for
    # non-existent keys.
    field = self.__getattr__(key)
    if isinstance(self._fields[key], FieldReference):
      return self._fields[key].get_type()
    else:
      return type(field)

  def update(self, *other, **kwargs):
    """Update values based on matching keys in another dict-like object.

    Mimics the built-in dict's update method: iterates over a given
    mapping object and adds/overwrites keys with the given mapping's
    values for those keys.

    Differs from dict.update in that it operates recursively on existing keys
    that are already a ConfigDict (i.e. calls their update() on the
    corresponding value from other), and respects the ConfigDict's
    type safety status.

    If keyword arguments are specified, the ConfigDict is updated with those
    key/value pairs.

    Args:
      *other: A (single) dict-like container, e.g. a dict or ConfigDict.
      **kwargs: Additional keyword arguments to update the ConfigDict.

    Raises:
      TypeError: if more than one value for `other` is specified.
    """
    if len(other) > 1:
      raise TypeError(
          'update expected at most 1 arguments, got {}'.format(len(other)))
    for other in other + (kwargs,):
      iteritems_kwargs = {}
      if isinstance(other, ConfigDict):
        iteritems_kwargs['preserve_field_references'] = True

      for key, value in other.items(**iteritems_kwargs):  # pytype: disable=wrong-keyword-args
        if key not in self:
          self[key] = value
        elif isinstance(self._fields[key], ConfigDict):
          self[key].update(other[key])
        elif (isinstance(self._fields[key], FieldReference) and
              isinstance(value, FieldReference)):
          # Updating FieldReferences from FieldReferences is not allowed.

          # One option could be to just grab the value from `other` and try to
          # update the reference in `self` using that. But that could result in
          # losing links between fields in `other`.

          # Example:
          #   other = ConfigDict(dict(a=1))
          #   other.b = other.get_ref('a')
          #   this = ConfigDict(dict(a=2))
          #   this.c = this.get_ref('a')
          #
          #   # Say we update `this` with `other`. The links between fields
          #   # in `other` could be lost in `this`.
          #   this.update(other)
          #
          #   # It is unclear what `this.b` should be when `this.a` is updated.
          #   this.a = 10
          #   # this.b?
          raise TypeError(
              'Cannot update a FieldReference from another '
              f'FieldReference: {key!r}'
          )
        else:
          self[key] = value

  def _update_value(self, key, index, value):
    if index is None:
      self[key] = value
    elif isinstance(self[key], list):
      self[key][index] = value
    elif isinstance(self[key], tuple):
      # Tuples are immutable, so convert to list, update and convert back.
      tuple_as_list = list(self[key])
      tuple_as_list[index] = value
      self[key] = tuple(tuple_as_list)

  def update_from_flattened_dict(self, flattened_dict, strip_prefix=''):
    """In-place updates values taken from a flattened dict.

    This allows a potentially nested source `ConfigDict` of the following form::

      cfg = ConfigDict({
          'a': 1,
          'b': {
              'c': {
                  'd': 2
              }
          }
      })

    to be updated from a dict containing paths navigating to child items, of the
    following form::

      updates = {
          'a': 2,
          'b.c.d': 3,
          'b.c.e': 4,
      }

    Note that update_from_flattened_dict will allow you add (not just update)
    leaf nodes - for example, 'b.c.e' above

    This filters `paths_dict` to only contain paths starting with
    `strip_prefix` and strips the prefix when applying the update.

    For example, consider we have the following values returned as flags::

      flags = {
          'flag1': x,
          'flag2': y,
          'config': 'some_file.py',
          'config.a.b': 1,
          'config.a.c': 2
      }

      config = ConfigDict({
          'a': {
              'b': 0,
              'c': 0
          }
      })

      config.update_from_flattened_dict(flags, 'config.')

    Then we will now have::

      config = ConfigDict({
          'a': {
              'b': 1,
              'c': 2
          }
      })

    Args:
      flattened_dict: A mapping (key path) -> value.
      strip_prefix: A prefix to be stripped from `path`. If specified, only
        paths matching `strip_prefix` will be processed.

    Raises:
      KeyError: if any of the key paths can't be found.
    """
    if strip_prefix:
      interesting_items = {
          key: value
          for key, value in flattened_dict.items()
          if key.startswith(strip_prefix)
      }
    else:
      interesting_items = flattened_dict

    # Keep track of any children that we want to update. Make sure that we
    # recurse into each one only once.
    children_to_update = {}

    for full_key, value in interesting_items.items():
      key = full_key[len(strip_prefix):] if strip_prefix else full_key

      # If the path is hierarchical, we'll need to tell the first component
      # to update itself.
      full_child = key.split('.')[0]

      # Check to see if we are trying to update a single element of a tuple/list
      #
      # TODO(kkg): The key/index parsing & handling logic below duplicates
      # similar logic in the config_flags/config_path module. Ideally we should
      # refactor the code to reuse the 'config_path' module here - but that is
      # likely a significant effort since that module already depends on this
      # leading to a circular dependency.
      child, index = _parse_key(full_child)

      if '.' not in key and index is None:
        # For a simple leaf node, just add the entry if it does not exist
        self[child] = value
        continue

      if not child in self:
        raise KeyError('Key "{}" cannot be set as "{}" was not found.'
                       .format(full_key, strip_prefix + child))

      if index is not None and not isinstance(self[child], (list, tuple)):
        raise KeyError('Key "{}" cannot be set as "{}" is not a tuple/list.'
                       .format(full_key, strip_prefix + child))

      if '.' in key:
        child_value = self[child] if index is None else self[child][index]
        if not isinstance(child_value, ConfigDict):
          raise KeyError(
              'Key "{}" cannot be updated as "{}" is not a ConfigDict ({}).'
              .format(full_key, strip_prefix + full_child, type(child_value))
          )

        children_to_update[full_child] = child_value
      else:
        self._update_value(child, index, value)

    for full_child, child_value in children_to_update.items():
      child_strip_prefix = f'{strip_prefix}{full_child}.'
      child_value.update_from_flattened_dict(
          interesting_items, child_strip_prefix
      )


def _should_skip_type_check(old_value, new_value) -> bool:
  """Returns True if the type check should be skipped."""

  if not isinstance(new_value, FieldReference):
    return False
  # Skip type checking if value is a FieldReference of the same type, or
  # FieldReference is generic type.
  if new_value.get_type() in (type(old_value), object):
    return True
  else:
    return False


def _frozenconfigdict_valid_input(obj, ancestor_list=None):
  """Raises error if obj is NOT a valid input for FrozenConfigDict.

  Args:
    obj: Object to check. In the first call (with ancestor_list = None), obj
        should be of type ConfigDict. During recursion, it may be any type
        except dict.
    ancestor_list: List of ancestors of obj in the attribute/element
        structure, used to detect reference cycles in obj.

  Raises:
    ValueError: If obj is an invalid input for FrozenConfigDict, i.e. if it
        contains a dict within a list/tuple or contains a reference cycle. Also
        if obj is a dict, which means it wasn't already converted to ConfigDict.
  """
  ancestor_list = ancestor_list or []

  # Dicts must be converted to ConfigDict before _frozenconfigdict_valid_input()
  assert not isinstance(obj, dict)

  if id(obj) in ancestor_list:
    raise ValueError('Bad FrozenConfigDict initialization: Cannot contain a '
                     'cycle in its reference structure.')
  ancestor_list.append(id(obj))

  if isinstance(obj, ConfigDict):
    for value in obj.values():
      _frozenconfigdict_valid_input(value, ancestor_list)
  elif isinstance(obj, FieldReference):
    _frozenconfigdict_valid_input(obj.get(), ancestor_list)
  elif isinstance(obj, (list, tuple)):
    for element in obj:
      if isinstance(element, (dict, ConfigDict, FieldReference)):
        raise ValueError('Bad FrozenConfigDict initialization: Cannot '
                         'contain a dict, ConfigDict, or FieldReference '
                         'within a list or tuple.')
      _frozenconfigdict_valid_input(element, ancestor_list)
  ancestor_list.pop()


def _tuple_to_immutable(value, visit_map):
  """Convert tuple to fully immutable tuple.

  Args:
    value: Tuple to be made fully immutable (including its elements).
    visit_map: As used elsewhere. See _frozenconfigdict_fill_seed()
        documentation. Must not contain id(value) as a key (if it does, an
        immutable version of value already exists).

  Returns:
    immutable_value: Immutable version of value, created with minimal
        copying (for example, if a value contains no mutable elements, it is
        returned untouched).
    same_value: Whether the same value was returned untouched, i.e. with the
        same memory address. Boolean.
    visit_map: Updated visit_map

  Raises:
    TypeError: If one of the following:
        1) value is not a tuple.
        2) value contains a dict, ConfigDict, or FieldReference. If it does,
           value is an invalid attribute of FrozenConfigDict, and this
           should have been caught in valid_input at initialization.
    ValueError: id(value) is in visit_map.
  """
  # Ensure there are no cycles
  assert id(value) not in visit_map

  value_copy = []
  same_value = True
  for element in value:
    # Sanity check: element cannot be dict, ConfigDict, or FieldReference
    assert not isinstance(element, (dict, ConfigDict, FieldReference))

    if isinstance(element, (list, tuple, set)):
      new_element, uncopied_element, visit_map = _convert_to_immutable(
          element, visit_map)
      same_value &= uncopied_element
      value_copy.append(new_element)
    else:
      value_copy.append(element)
  if same_value:
    return value, True, visit_map
  else:
    return tuple(value_copy), False, visit_map


def _convert_to_immutable(value, visit_map):
  """Convert Python built-in type to immutable, copying if necessary.

  Args:
    value: To be made immutable type (including its elements). Must have
        type list, tuple, or set.
    visit_map: As used elsewhere. See _frozenconfigdict_fill_seed()
        documentation.

  Returns:
    immutable_value: Immutable version of value, created with minimal
        copying.
    same_value: Whether the same value was returned untouched, i.e. with the
        same memory address. Boolean.
    visit_map: Updated visit_map.

  Raises:
    TypeError: If value is an invalid type (not a list, tuple, or set).
  """
  value_id = id(value)
  if value_id in visit_map:
    return visit_map[value_id], True, visit_map

  same_value = False
  if isinstance(value, set):
    immutable_value = frozenset(value)
  elif isinstance(value, tuple):
    immutable_value, same_value, visit_map = _tuple_to_immutable(
        value, visit_map)
  elif isinstance(value, list):
    immutable_value, _, visit_map = _tuple_to_immutable(tuple(value),
                                                        visit_map)
  else:
    # Type-check the input
    assert False
  visit_map[value_id] = immutable_value
  return immutable_value, same_value, visit_map


def _frozenconfigdict_fill_seed(seed, initial_configdict, visit_map=None):
  """Fills an empty FrozenConfigDict without copying previously visited nodes.

  Turns seed (an empty FrozenConfigDict) into a FrozenConfigDict version of
  initial_configdict. Avoids duplicating nodes of initial_configdict because if
  a value of initial_configdict has been previously visited, that value is not
  re-converted to a FrozenConfigDict. If a FieldReference is encountered which
  contains a dict, its contents will be converted to FrozenConfigDict.

  Note: As described in the __init__() documentation, this will not
  replicate the structure of initial_configdict if it contains
  self-references within lists, tuples, or other types. There is no warning
  or error in this case.

  Args:
    seed: Empty FrozenConfigDict, to be filled in.
    initial_configdict: The template on which seed is built. Must be of type
        ConfigDict.
    visit_map: Dictionary from memory addresses to values, storing the
        FrozenConfigDict versions of dictionary values. Lists which have
        been converted to tuples and sets to frozensets are also stored in
        visit_map to preserve the reference structure of initial_configdict.
        visit_map need not contain (id(initial_configdict), seed) as a key/value
        pair.

  Raises:
    ValueError: If one of the following, both of which can never happen in
        practice:
            1) seed is not an empty FrozenConfigDict.
            2) initial_configdict contains a FieldReference.
  """
  # These should be impossible to raise, since the public call-site in
  # __init__() pass in valid input, as does this method recursively.
  assert isinstance(seed, FrozenConfigDict)
  assert not seed

  # This is where we define self._configdict for the FrozenConfigDict() class.
  # It is defined here instead of in FrozenConfigDict.__init__() because we must
  # fill in an empty FrozenConfigDict but do not want to have an unexpected
  # signature for FrozenConfigDict.__init__() by passing it initial_configdict.
  object.__setattr__(seed, '_configdict', initial_configdict)

  visit_map = visit_map or {}
  visit_map[id(initial_configdict)] = seed

  for key, value in initial_configdict.items():
    # This should never be raised due to elimination of references by
    # ConfigDict's iteritems
    if isinstance(value, FieldReference):
      raise ValueError('Trying to initialize a FrozenConfigDict value with '
                       'a FieldReference. This should never happen, please '
                       'file a bug.')

    if id(value) in visit_map:
      value = visit_map[id(value)]
    elif (isinstance(value, ConfigDict) and
          not isinstance(value, FrozenConfigDict)):
      value_frozenconfigdict = FrozenConfigDict(type_safe=value.is_type_safe)
      _frozenconfigdict_fill_seed(value_frozenconfigdict, value, visit_map)
      value = value_frozenconfigdict

    seed._frozen_setattr(key, value,  # pylint:disable=protected-access
                         visit_map)


class FrozenConfigDict(ConfigDict):
  """Immutable and hashable type of ConfigDict.

  See ConfigDict() documentation above for details and usage.

  FrozenConfigDict is fully immutable. It contains no lists or sets (at
  initialization, lists and sets are converted to tuples and frozensets). The
  only potential sources of mutability are attributes with custom types, which
  are not touched.

  It is recommended to convert a ConfigDict to FrozenConfigDict after
  construction if possible.
  """

  def __init__(self, initial_dictionary=None, type_safe=True):
    """Creates an instance of FrozenConfigDict.

    Lists and sets are copied into tuples and frozensets. However, copying is
    kept to a minimum so tuples, frozensets, and other immutable types are not
    copied unless they contain mutable types.

    Prohibited initial_dictionary structures: initial_dictionary may not contain
    any lists or tuples with dictionary, ConfigDict, or FieldReference elements,
    or else an error is raised at initialization. It also may not contain loops
    in the reference structure, i.e. the reference structure must be a Directed
    Acyclic Graph. This includes loops in list-element and tuple-element
    references. initial_dictionary's reference structure need not be a tree.

    Warning: Unexpected behavior may occur with types other than Python's
    built-in types. See ConfigDict() documentation for details.

    Warning: As with ConfigDict, FieldReference values may be changed. If
    initial_dictionary contains a FieldReference with a value of type dict or
    ConfigDict, that value will be converted to FrozenConfigDict.

    Args:
      initial_dictionary: May be one of the following:

        1) dict. In this case all values of initial_dictionary that are
        dictionaries are also converted to FrozenConfigDict. If there are
        dictionaries contained in lists or tuples, an error is raised.

        2) ConfigDict. In this case all ConfigDict attributes are also
        converted to FrozenConfigDict.

        3) FrozenConfigDict. In this case all attributes are uncopied, and
        only the top-level object (self) is re-addressed.

      type_safe: See ConfigDict documentation. Note that this only matters
          if the FrozenConfigDict is converted to ConfigDict at some point.
    """

    super(FrozenConfigDict, self).__init__()

    initial_configdict = ConfigDict(initial_dictionary=initial_dictionary,
                                    type_safe=type_safe)

    _frozenconfigdict_valid_input(initial_configdict)
    # This will define the self._configdict attribute
    _frozenconfigdict_fill_seed(self, initial_configdict)

    object.__setattr__(self, '_locked', initial_configdict.is_locked)
    object.__setattr__(self, '_type_safe', initial_configdict.is_type_safe)

  def _frozen_setattr(self, key, value, visit_map=None):
    """Sets attribute, analogous to __setattr__().

    Args:
      key: Name of the attribute to set.
      value: Value of the attribute to set.
      visit_map: Dictionary from memory addresses to values, storing the
          FrozenConfigDict versions of value's elements. Lists which have been
          converted to tuples and sets to frozensets are also stored in
          visit_map.

    Returns:
      visit_map: Updated visit_map.

    Raises:
      ValueError: If there is a dot in key, or value contains dicts inside lists
          or tuples. Also if key is already an attribute, since redefining an
          attribute is prohibited for FrozenConfigDict.
      AttributeError: If key is protected (such as '_type_safe' and '_locked').
    """
    visit_map = visit_map or {}

    # These should always pass because of conversion to ConfigDict at
    # initialization
    self._ensure_mutability(key)
    assert '.' not in key

    if key in self._fields:
      raise ValueError('Cannot redefine attribute of FrozenConfigDict.')

    if isinstance(value, (list, tuple, set)):
      immutable_value, _, visit_map = _convert_to_immutable(value, visit_map)
      self._fields[key] = immutable_value
    else:
      self._fields[key] = value

    return visit_map

  def __setstate__(self, state):
    """Recreates FrozenConfigDict from its dict representation."""
    self.__init__(state['_configdict'])

  def __setattr__(self, attribute, value):
    raise AttributeError('FrozenConfigDict is immutable. Cannot call '
                         '__setattr__().')

  def __delattr__(self, attribute):
    raise AttributeError('FrozenConfigDict is immutable. Cannot call '
                         '__delattr__().')

  def __setitem__(self, attribute, value):
    raise AttributeError('FrozenConfigDict is immutable. Cannot call '
                         '__setitem__().')

  def __delitem__(self, attribute):
    raise AttributeError('FrozenConfigDict is immutable. Cannot call '
                         '__delitem__().')

  def lock(self):
    raise AttributeError('FrozenConfigDict is immutable. Cannot call lock().')

  def unlock(self):
    raise AttributeError('FrozenConfigDict is immutable. Cannot call unlock().')

  def __hash__(self):
    """Computes hash.

    The hash depends not only on the immutable aspects of the FrozenConfigDict,
    but on the types of the initial_dictionary at initialization (i.e. on the
    _configdict attribute). For example, in the following, hash(frozen_1) will
    not equal hash(frozen_2):
        d_1 = {'a': (1, )}
        d_2 = {'a': [1]}
        frozen_1 = FrozenConfigDict(d_1)
        frozen_2 = FrozenConfigDict(d_2)

    Note: This implementation relies on the particulars of the FrozenConfigDict
    structure. For example, the fact that lists and tuples cannot contain dicts
    or ConfigDicts is crucial, as is the fact that any loop in the reference
    structure is prohibited.

    Note: Due to hash randomization, this hash will likely differ in different
    Python sessions. For comparisons across sessions, please directly use
    equality of the serialization. For more, see
    https://bugs.python.org/issue13703

    Returns:
      frozenconfigdict_hash: The hash value.

    Raises:
      TypeError: self contains an unhashable type.
    """

    def value_hash(value):
      """Hashes a single value."""
      if isinstance(value, set):
        return hash((hash(frozenset(value)), 1))
      elif isinstance(value, (list, tuple)):
        value_hash_list = [isinstance(value, list)]
        for element in value:
          element_hash = value_hash(element)
          value_hash_list.append(element_hash)
        return hash(tuple(value_hash_list))
      elif isinstance(value, FieldReference):
        return value_hash(value.get())
      else:
        try:
          return hash(value)
        except TypeError:
          raise TypeError('FrozenConfigDict contains unhashable type '
                          '{}'.format(type(value)))

    fields_hash = 0
    for key, value in self._fields.items():
      if isinstance(value, FrozenConfigDict):
        fields_hash += hash((hash(key), hash(value)))
      else:
        # Use self._configdict value to ensure attending to mutability
        fields_hash += hash((hash(key),
                             value_hash(self._configdict._fields[key])))

    frozenconfigdict_hash = hash((fields_hash, self.is_locked,
                                  self.is_type_safe))
    return frozenconfigdict_hash

  def __eq__(self, other):
    """Override default Equals behavior.

    Like __hash__(), this pays attention to the type of initial_dictionary. See
    __hash__() documentation for details.

    Warning: This distinguishes FrozenConfigDict from ConfigDict. For example:
        cd = ConfigDict()
        fcd = FrozenConfigDict()
        fcd.__eq__(cd)  # Returns False

    Args:
      other: Object to compare self to.

    Returns:
      same: Boolean self == other.
    """
    if isinstance(other, FrozenConfigDict):
      return ConfigDict(self) == ConfigDict(other)
    else:
      return False

  def as_configdict(self):
    return self._configdict


class CustomJSONEncoder(json.JSONEncoder):
  """JSON encoder for ConfigDict and FieldReference.

  The encoder throws an exception for non-supported types.
  """

  def default(self, obj):
    if isinstance(obj, FieldReference):
      return obj.get()
    elif isinstance(obj, ConfigDict):
      return obj._fields
    elif isinstance(obj, type):
      return str(obj)
    else:
      raise TypeError('{} is not JSON serializable. Instead use '
                      'ConfigDict.to_json_best_effort()'.format(type(obj)))


class _BestEffortCustomJSONEncoder(CustomJSONEncoder):
  """Best effort JSON encoder for ConfigDict.

  The encoder tries to serialize non-supported types (doesn't throw exceptions).
  """

  def default(self, obj):
    try:
      return super(_BestEffortCustomJSONEncoder, self).default(obj)
    except TypeError:
      if isinstance(obj, set):
        return sorted(list(obj))
      elif inspect.isfunction(obj):
        return 'function {}'.format(obj.__name__)
      elif dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
      elif (hasattr(obj, '__dict__') and
            obj.__dict__ and
            not inspect.isclass(obj)):  # Instantiated object's variables
        return dict(obj.__dict__)
      elif hasattr(obj, '__str__'):
        return 'unserializable object: {}'.format(obj)
      else:
        return 'unserializable object of type: {}'.format(type(obj))


def create(**kwargs):
  """Creates a `ConfigDict` with the given named arguments as key-value pairs.

  This allows for simple dictionaries whose elements can be accessed directly
  using field access::

    from ml_collections import config_dict
    point = config_dict.create(x=1, y=2)
    print(point.x, point.y)

  This is particularly useful for compactly writing nested configurations::

    config = config_dict.create(
      data=config_dict.create(
        game='freeway',
        frame_size=100),
      model=config_dict.create(num_hidden=1000))

  The reason for the existence of this function is that it simplifies the
  code required for the majority of the use cases of `ConfigDict`, compared
  to using either `ConfigDict` or `namedtuple's`. Examples of such use cases
  include training script configuration, and returning multiple named values.

  Args:
    **kwargs: key-value pairs to be stored in the `ConfigDict`.

  Returns:
    A `ConfigDict` containing the key-value pairs in `kwargs`.
  """
  return ConfigDict(initial_dictionary=kwargs)


# TODO(sergomez): make placeholders required by default.
def placeholder(field_type, required=False):
  """Defines an entry in a ConfigDict that has no value yet.

  Example::

    config = configdict.create(
        batch_size = configdict.placeholder(int),
        frame_shape = configdict.placeholder(tf.TensorShape))

  Args:
    field_type: type of value.
    required: If True, the placeholder will raise an error on access if the
         underlying value hasn't been set.

  Returns:
    A `FieldReference` with value None and the given type.
  """
  return FieldReference(None, field_type=field_type, required=required)


def required_placeholder(field_type):
  """Defines an entry in a ConfigDict with unknown but required value.

  Example::

    config = configdict.create(
        batch_size = configdict.required_placeholder(int))

    try:
      print(config.batch_size)
    except RequiredValueError:
      pass

    config.batch_size = 10
    print(config.batch_size)  # 10

  Args:
    field_type: type of value.

  Returns:
    A `FieldReference` with value None and the given type.
  """
  return placeholder(field_type, required=True)


def recursive_rename(conf, old_name, new_name):
  """Returns copy of conf with old_name recursively replaced by new_name.

  This is not done in place, no changes are made to conf but a new ConfigDict is
  returned with the changes made. This is useful if the name of a parameter has
  been changed in code but you need to load an old config.

  Example usage:
    updated_conf = configdict.recursive_rename(conf, "config", "kwargs")

  Args:
    conf: a ConfigDict which needs updating
    old_name: the name used in the ConfigDict which is out of sync with the code
    new_name: the name used in the code

  Returns:
    A ConfigDict which is a copy of conf but with all instances of old_name
    replaced with new_name.
  """
  new_conf = ConfigDict()
  for name, c in conf.items():
    if isinstance(c, ConfigDict):
      new_c = recursive_rename(c, old_name, new_name)
    else:
      new_c = c
    if name == old_name:
      setattr(new_conf, new_name, new_c)
    else:
      setattr(new_conf, name, new_c)
  return new_conf
