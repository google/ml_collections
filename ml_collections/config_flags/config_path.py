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

"""Module for spliting flag prefixes."""

import ast
import dataclasses as dc
import functools
import types
import typing
from typing import Any, MutableSequence, Optional, Sequence, Tuple, Union, Type

from ml_collections import config_dict


NoneType = type(None)


_AST_SPLIT_CONFIG_PATH = {
    ast.Attribute: lambda n: (*_split_node(n.value), n.attr),
    ast.Index: lambda i: _split_node(i.value),
    ast.Name: lambda n: (n.id,),
    ast.Slice: lambda i: slice(*map(_split_node, (i.lower, i.upper, i.step))),
    ast.Subscript: lambda n: (*_split_node(n.value), _split_node(n.slice)),
    type(None): lambda n: None
}


def _split_node(node):
  return _AST_SPLIT_CONFIG_PATH.get(type(node), ast.literal_eval)(node)


def split(config_path: str) -> Tuple[Any]:
  """Returns config_path split into a tuple of parts.

  Example usage:
    >>> assert config_path.split('a.b.cc') == ('a', 'b', 'cc')
    >>> assert config_path.split('a.b["cc.d"]') == ('a', 'b', 'cc.d')
    >>> assert config_path.split('a.b[10]') == ('a', 'b', 10)
    >>> assert config_path.split('a[(1, 2)]') == ('a', (1, 2))
    >>> assert config_path.split('a[:]') == ('a', slice(None))

  Args:
    config_path: Input path to be split - see example usage.

  Returns:
    Tuple of config_path split into parts. Parts are attributes or subscripts.
    Attrributes are treated as strings and subscripts are parsed using
    ast.literal_eval. It is up to the caller to ensure all returned types are
    valid.

  Raises:
    ValueError: Failed to parse config_path.
  """
  try:
    node = ast.parse(config_path, mode='eval')
  except SyntaxError as e:
    raise ValueError(f'Could not parse {config_path!r}: {e!r}') from None
  if isinstance(node, ast.Expression):
    result = _split_node(node.body)
    if isinstance(result, tuple):
      return result
  raise ValueError(config_path)


def _get_item_or_attribute(config, field,
                           field_path: Optional[str] = None):
  """Returns attribute of member failing that the item."""
  if isinstance(field, str) and hasattr(config, field):
    return getattr(config, field)
  if hasattr(config, '__getitem__'):
    return config[field]
  if isinstance(field, int):
    raise IndexError(
        f'{type(config)} does not support integer indexing [{field}]]. '
        f'Attempting to lookup: {field_path}')
  raise KeyError(
      f'Attribute {type(config)}.{field} does not exist '
      'and the type does not support indexing. '
      f'Attempting to lookup: {field_path}')


def _get_holder_field(config_path: str, config: Any) -> Tuple[Any, str]:
  """Returns the last part config_path and config to allow assignment.

  Example usage:
    >>> config = {'a': {'b', {'c', 10}}}
    >>> holder, lastfield = _get_holder_field('a.b.c', config)
    >>> assert lastfield == 'c'
    >>> assert holder is config['a']['b']
    >>> assert holder[lastfield] == 10

  Args:
    config_path: Any string that `split` can process.
    config: A nested datastructure that can be accessed via
      _get_item_or_attribute

  Returns:
    The penultimate object when walking config with config_path. And the final
    part of the config path.

  Raises:
    IndexError: Integer field not found in nested structure.
    KeyError: Non-integer field not found in nested structure.
    ValueError: Empty/invalid config_path after parsing.
  """
  fields = split(config_path)
  if not fields:
    raise ValueError('Path cannot be empty')
  get_item = functools.partial(_get_item_or_attribute, field_path=config_path)
  holder = functools.reduce(get_item, fields[:-1], config)
  return holder, fields[-1]


def get_value(config_path: str, config: Any):
  """Gets value of a single field.

  Example usage:
    >>> config = {'a': {'b', {'c', 10}}}
    >>> assert config_path.get_value('a.b.c', config) == 10

  Args:
    config_path: Any string that `split` can process.
    config: A nested datastructure

  Returns:
    The last object when walking config with config_path.

  Raises:
    IndexError: Integer field not found in nested structure.
    KeyError: Non-integer field not found in nested structure.
    ValueError: Empty/invalid config_path after parsing.
  """
  get_item = functools.partial(_get_item_or_attribute, field_path=config_path)
  return functools.reduce(get_item, split(config_path), config)


def initialize_missing_parent_fields(
    config: Any, override: str,
    allowed_missing: Sequence[str]):
  """Adds some missing nested holder fields for a particular override.

  For example if override is 'config.a.b.c' and config.a is None, it
  will default initialize config.a, and if config.a.b is None will default
  initialize it as well. Only overrides present in allowed_missing will
  be initialized.

  Args:
    config: config object (typically dataclass)
    override: dot joined override name.
    allowed_missing: list of overrides that are allowed
    to be set. For example, if override is 'a.b.c.d',
    allowed_missing could be ['a.b.c', 'a', 'foo.bar'].

  Raises:
    ValueError: if parent field is not of dataclass type.
  """
  fields = split(override)
  # Collect the tree levels at which we are alloed to create override
  allowed_levels = {len(split(x)) for x in allowed_missing if
                    override.startswith(x + '.')}
  child = config
  for level, f in enumerate(fields[:-1], 1):
    parent = child
    child = _get_item_or_attribute(parent, f, override)
    if child is not None:
      continue
    # Field is not yet present, see if we should create it instead.
    field_type = get_type(f, parent)
    # Note: these two assertions below are mostly guard
    # rails to prevent behaviors that might be confusing/accidental.
    # Specifically we disallow implicit creation of parent fields,
    # creating non dataclass objects. They can be revisited
    # in the future.
    if not dc.is_dataclass(field_type):
      raise ValueError(
          f'Override {override} can not be applied because '
          f'field "{f}" is None, and its type "{field_type}" is not a '
          f'dataclass in the parent of type "{type(parent)}".')

    if level not in allowed_levels:
      raise ValueError(
          f'Flag {override} can not be applied because '
          f'field "{f}" is None by default and it is not explicitly '
          'provided in flags (it can be default intialized by '
          f'providing --<path-to-{f}>.{f}=build flag')
    try:
      child = field_type()
    except Exception as e:
      raise ValueError(
          f'Override {override} can not be applied because '
          f'field "{f}" of type {field_type} can not be default instantiated:'
          f'{e}') from e
    set_value(f, parent, child)


def get_origin(type_spec: type) -> Optional[type]:   # pylint: disable=g-bare-generic drop when 3.7 support is not needed
  """Call typing.get_origin, with a fallback for Python 3.7 and below."""
  if hasattr(typing, 'get_origin'):
    return typing.get_origin(type_spec)
  return getattr(type_spec, '__origin__', None)


def get_args(type_spec: type) -> Union[NoneType, Tuple[type, ...]]:  # pylint: disable=g-bare-generic drop when 3.7 support is not needed
  """Call typing.get_args, with fallback for Python 3.7 and below."""
  if hasattr(typing, 'get_args'):
    return typing.get_args(type_spec)
  return getattr(type_spec, '__args__', NoneType)


def _is_union_type(type_spec: type) -> bool:  # pylint: disable=g-bare-generic drop when 3.7 support is not needed
  """Cheeck if a type_spec is a Union type or not."""
  # UnionType was only introduced in python 3.10. We need getattr for
  # backward compatibility.
  return get_origin(type_spec) in [Union, getattr(types, 'UnionType', Union)]


def extract_type_from_optional(type_spec: type) -> Optional[type]:  # pylint: disable=g-bare-generic drop when 3.7 support is not needed
  """If type_spec is of type Optional[T], returns T object, otherwise None"""
  if not _is_union_type(type_spec):
    return None
  non_none = [t for t in get_args(type_spec) if t is not NoneType]
  if len(non_none) != 1:
    return None
  return non_none[0]


def normalize_type(type_spec: type) -> type:  # pylint: disable=g-bare-generic drop when 3.7 support is not needed
  """Normalizes a type object.

  Strips all None types from the type specification and returns the remaining
  single type. This is primarily useful for Optional type annotations in which
  case it will strip out the NoneType and return the inner type.

  Args:
    type_spec: The type to normalize.

  Raises:
    TypeError: If there is not exactly 1 non-None type in the union.
  Returns:
    The normalized type.
  """
  if _is_union_type(type_spec):
    subtype = extract_type_from_optional(type_spec)
    if subtype is None:
      raise TypeError(f'Unable to normalize ambiguous type: {type_spec}')
    return subtype

  return type_spec


def get_type(
    config_path: str,
    config: Any,
    normalize=True,
    default_type: Optional[Type[Any]] = None,
):
  """Gets type of field in config described by a config_path.

  Example usage:
    >>> config = {'a': {'b', {'c', 10}}}
    >>> assert config_path.get_type('a.b.c', config) is int

  Args:
    config_path: Any string that `split` can process.
    config: A nested datastructure
    normalize: whether to normalize the type (in particular strip Optional
      annotations on dataclass fields)
    default_type: If the `config_path` is not found and `default_type` is set,
      the `default_type` is returned.

  Returns:
    The type of last object when walking config with config_path.

  Raises:
    IndexError: Integer field not found in nested structure.
    KeyError: Non-integer field not found in nested structure.
    ValueError: Empty/invalid config_path after parsing.
    TypeError: Ambiguous type annotation on dataclass field.
  """
  holder, field = _get_holder_field(config_path, config)
  # Check if config is a DM collection and hence has attribute get_type()
  if isinstance(holder,
                (config_dict.ConfigDict, config_dict.FieldReference)):
    if default_type is not None and field not in holder:
      return default_type
    return holder.get_type(field)
  # For dataclasses we can just use the type annotation.
  elif dc.is_dataclass(holder):
    matches = [f.type for f in dc.fields(holder) if f.name == field]
    if not matches:
      raise KeyError(f'Field {field} not found on dataclass {type(holder)}')
    return normalize_type(matches[0]) if normalize else matches[0]
  else:
    return type(_get_item_or_attribute(holder, field, config_path))


def is_optional(config_path: str, config: Any) -> bool:
  raw_type = get_type(config_path, config, normalize=False)
  return extract_type_from_optional(raw_type) is not None


def set_value(
    config_path: str,
    config: Any,
    value: Any,
    *,
    accept_new_attributes: bool = False,
):
  """Sets value of field described by config_path.

  Example usage:
    >>> config = {'a': {'b', {'c', 10}}}
    >>> config_path.set_value('a.b.c', config, 20)
    >>> assert config['a']['b']['c'] == 20

  Args:
    config_path: Any string that `split` can process.
    config: A nested datastructure
    value: A value to assign to final field.
    accept_new_attributes: If `True`, the new config attributes can be added

  Raises:
    IndexError: Integer field not found in nested structure.
    KeyError: Non-integer field not found in nested structure.
    ValueError: Empty/invalid config_path after parsing.
  """
  holder, field = _get_holder_field(config_path, config)

  if isinstance(field, int) and isinstance(holder, MutableSequence):
    holder[field] = value
  elif hasattr(holder, '__setitem__') and (
      field in holder or accept_new_attributes
  ):
    holder[field] = value
  elif hasattr(holder, str(field)):
    setattr(holder, str(field), value)
  else:
    if isinstance(field, int):
      raise IndexError(
          f'{field} is not a valid index for {type(holder)} '
          f'(in: {config_path})')
    raise KeyError(f'{field} is not a valid key or attribute of {type(holder)} '
                   f'(in: {config_path})')
