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

"""Module for spliting flag prefixes."""

import ast
import dataclasses as dc
import functools
import typing
from typing import Any, MutableSequence, Tuple, Union

from ml_collections import config_dict


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
    raise ValueError(e)
  if isinstance(node, ast.Expression):
    result = _split_node(node.body)
    if isinstance(result, tuple):
      return result
  raise ValueError(config_path)


def _get_item_or_attribute(config, field):
  """Returns attribute of member failing that the item."""
  if isinstance(field, str) and hasattr(config, field):
    return getattr(config, field)
  if hasattr(config, '__getitem__'):
    return config[field]
  if isinstance(field, int):
    raise IndexError(f'{field}')
  raise KeyError(f'{field}')


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
  holder = functools.reduce(_get_item_or_attribute, fields[:-1], config)
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
  return functools.reduce(_get_item_or_attribute, split(config_path), config)


def normalize_type(type_spec: type) -> type:
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
  if hasattr(typing, 'get_origin'):
    if typing.get_origin(type_spec) == Union:
      non_none = [t for t in typing.get_args(type_spec) if t is not type(None)]
      if len(non_none) != 1:
        raise TypeError(f'Unable to normalize ambiguous type: {type_spec}')
      return non_none[0]
  # TODO(sergomez): Remove fallback when 3.7 support is no longer needed.
  else:
    if hasattr(type_spec, '__origin__') and type_spec.__origin__ is Union:
      non_none = [t for t in type_spec.__args__ if t is not type(None)]
      if len(non_none) != 1:
        raise TypeError(f'Unable to normalize ambiguous type: {type_spec}')
      return non_none[0]
  return type_spec


def get_type(config_path: str, config: Any):
  """Gets type of field in config described by a config_path.

  Example usage:
    >>> config = {'a': {'b', {'c', 10}}}
    >>> assert config_path.get_type('a.b.c', config) is int

  Args:
    config_path: Any string that `split` can process.
    config: A nested datastructure

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
    return holder.get_type(field)
  # For dataclasses we can just use the type annotation.
  elif dc.is_dataclass(holder):
    matches = [f.type for f in dc.fields(holder) if f.name == field]
    if not matches:
      raise KeyError(f'Field {field} not found on dataclass {type(holder)}')
    return normalize_type(matches[0])
  else:
    return type(_get_item_or_attribute(holder, field))


def set_value(config_path: str, config: Any, value: Any):
  """Sets value of field described by config_path.

  Example usage:
    >>> config = {'a': {'b', {'c', 10}}}
    >>> config_path.set_value('a.b.c', config, 20)
    >>> assert config['a']['b']['c'] == 20

  Args:
    config_path: Any string that `split` can process.
    config: A nested datastructure
    value: A value to assign to final field.

  Raises:
    IndexError: Integer field not found in nested structure.
    KeyError: Non-integer field not found in nested structure.
    ValueError: Empty/invalid config_path after parsing.
  """
  holder, field = _get_holder_field(config_path, config)

  if isinstance(field, int) and isinstance(holder, MutableSequence):
    holder[field] = value
  elif hasattr(holder, '__setitem__') and field in holder:
    holder[field] = value
  elif hasattr(holder, str(field)):
    setattr(holder, str(field), value)
  else:
    if isinstance(field, int):
      raise IndexError(f'{field}')
    raise KeyError(f'{field}')
