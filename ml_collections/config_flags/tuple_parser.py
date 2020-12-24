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

# Lint as: python 3
"""Custom parser to override tuples in the config dict."""
import ast

from absl import flags


class TupleParser(flags.ArgumentParser):
  """Parser for tuple arguments.

  Custom flag parser for Tuple objects that is based on the existing parsers
  in `absl.flags`. This parser can be used to read in and override tuple
  arguments. It outputs a `tuple` object from an existing `tuple` or `str`.
  The ony requirement is that the overriding parameter should be a `tuple`
  as well. The overriding parameter can have a different number of elements
  of different type than the original. For a detailed list of what `str`
  arguments are supported for overriding, look at `ast.literal_eval` from the
  Python Standard Library.
  """

  def parse(self, argument):
    """Returns a `tuple` representing the input `argument`.

    Args:
      argument: The argument to be parsed. Valid types are `tuple` and
        `str`. An empty `tuple` is returned for arguments `NoneType`.

    Returns:
      A `TupleType` representing the input argument as a `tuple`.

    Raises:
      `TypeError`: If the argument is not of type `tuple`, `str`, or
        `NoneType`.
      `ValueError`: If the string is not a well formed `tuple`.
    """
    if isinstance(argument, tuple):
      return argument
    elif isinstance(argument, str):
      return _convert_str_to_tuple(argument)
    elif argument is None:
      return ()
    else:
      msg = ('Could not parse argument {} of type '
             '{} for element of type `tuple`.'
            ).format(argument, type(argument))
      raise TypeError(msg)

  def flag_type(self):
    return 'tuple'


def _convert_str_to_tuple(string):
  """Function to convert a Python `str` object to a `tuple`.

  Args:
    string: The `str` to be converted.

  Returns:
    A `tuple` version of the string.

  Raises:
    ValueError: If the string is not a well formed `tuple`.
  """
  # literal_eval converts strings to int, tuple, list, float and dict,
  # booleans and None. It can also handle nested tuples.
  # It does not, however, handle elements of type set.
  try:
    value = ast.literal_eval(string)
  except ValueError:
    # A ValueError is raised by literal_eval if the string is not well
    # formed. Catch it and print out a more readable statement.
    msg = 'Argument {} does not evaluate to a `tuple` object.'.format(string)
    raise ValueError(msg)
  except SyntaxError:
    # The only other error that may be raised is a `SyntaxError` because
    # `literal_eval` calls the Python in-built `compile`. This error is
    # caused by parsing issues.
    msg = 'Error while parsing string: {}'.format(string)
    raise ValueError(msg)

  # Make sure we got a tuple. If not, its an error.
  if isinstance(value, tuple):
    return value
  else:
    raise ValueError('Expected a tuple argument, got {}'.format(type(value)))
