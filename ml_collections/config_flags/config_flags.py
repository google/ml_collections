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
"""Configuration commmand line parser."""

import errno
import imp
import os
import re
import sys
import traceback
from typing import Any, Dict, Generic, List, MutableMapping, Optional, Tuple, Type, TypeVar

from absl import flags
from absl import logging
import dataclasses
import ml_collections
from ml_collections.config_flags import tuple_parser
import six
from six import string_types

FLAGS = flags.FLAGS

# We need this to work both for Python 2 and 3.
# The initial code in Python 2 was:
# _FIELD_TYPE_TO_PARSER = {
#     types.IntType: flags.IntegerParser(),
#     types.FloatType: flags.FloatParser(),         # OK For Python 3
#     types.BooleanType: flags.BooleanParser(),     # OK For Python 3
#     types.StringType: flags.ArgumentParser(),
#     types.TupleType: tuple_parser.TupleParser(),  # OK For Python 3
# }
# The possible breaking changes are:
# - A Python 3 int could be a Python 2 long, which was not previously supported.
#   We then add support for long.
# - Only Python 2 str were supported (not unicode). Python 3 will behave the
#   same with the str semantic change.
_FIELD_TYPE_TO_PARSER = {
    float: flags.FloatParser(),
    bool: flags.BooleanParser(),
    # Implementing a custom parser to override `Tuple` arguments.
    tuple: tuple_parser.TupleParser(),
}
for t in six.integer_types:
  _FIELD_TYPE_TO_PARSER[t] = flags.IntegerParser()
for t in six.string_types:
  _FIELD_TYPE_TO_PARSER[t] = flags.ArgumentParser()
_FIELD_TYPE_TO_PARSER[str] = flags.ArgumentParser()
_FIELD_TYPE_TO_SERIALIZER = {
    t: flags.ArgumentSerializer()
    for t in _FIELD_TYPE_TO_PARSER
}


class UnsupportedOperationError(flags.Error):
  pass


class FlagOrderError(flags.Error):
  pass


class UnparsedFlagError(flags.Error):
  pass


def DEFINE_config_file(  # pylint: disable=g-bad-name
    name: str,
    default: Optional[str] = None,
    help_string: str = 'path to config file.',
    flag_values: flags.FlagValues = FLAGS,
    lock_config: bool = True,
    **kwargs) -> flags.FlagHolder:
  r"""Defines flag for `ConfigDict` files compatible with absl flags.

  The flag's value should be a path to a valid python file which contains a
  function called `get_config()` that returns a python object specifying
  a configuration. After the flag is parsed, `FLAGS.name` will contain
  a reference to this object, optionally with some values overridden.

  During flags parsing, every flag of form `--name.([a-zA-Z0-9]+\.?)+=value`
  and `-name.([a-zA-Z0-9]+\.?)+ value` will be treated as an override of a
  specific field in the config object returned by this flag. Field is
  essentially a dot delimited path inside the object where each path element
  has to be either an attribute or a key existing in the config object.
  For example `--my_config.field1.field2=val` means "assign value val
  to the attribute (or key) `field2` inside value of the attribute (or key)
  `field1` inside the value of `my_config` object". If there are both
  attribute and key-based access with the same name, attribute is preferred.

  Typical usage example:

  `script.py`::

    from absl import flags
    from ml_collections.config_flags import config_flags

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file('my_config')

    print(FLAGS.my_config)

  `config.py`::

    def get_config():
      return {
          'field1': 1,
          'field2': 'tom',
          'nested': {
              'field': 2.23,
          },
      }

  The following command::

    python script.py -- --my_config=config.py
                        --my_config.field1 8
                        --my_config.nested.field=2.1

  will print::

    {'field1': 8, 'field2': 'tom', 'nested': {'field': 2.1}}

  It is possible to parameterise the get_config function, allowing it to
  return a differently structured result for different occasions. This is
  particularly useful when setting up hyperparameter sweeps across various
  network architectures.

  `parameterised_config.py`::

    def get_config(config_string):
      possible_configs = {
          'mlp': {
              'constructor': 'snt.nets.MLP',
              'config': {
                  'output_sizes': (128, 128, 1),
              }
          },
          'lstm': {
              'constructor': 'snt.LSTM',
              'config': {
                  'hidden_size': 128,
                  'forget_bias': 1.0,
              }
          }
      }
      return possible_configs[config_string]

  If a colon is present in the command line override for the config file,
  everything to the right of the colon is passed into the get_config function.
  The following command lines will both function correctly::

    python script.py -- --my_config=parameterised_config.py:mlp
                        --my_config.config.output_sizes="(256,256,1)"


    python script.py -- --my_config=parameterised_config.py:lstm
                        --my_config.config.hidden_size=256

  The following will produce an error, as the hidden_size flag does not
  exist when the "mlp" config_string is provided::

    python script.py -- --my_config=parameterised_config.py:mlp
                        --my_config.config.hidden_size=256

  Args:
    name: Flag name, optionally including extra config after a colon.
    default: Default value of the flag (default: None).
    help_string: Help string to display when --helpfull is called.
        (default: "path to config file.")
    flag_values: FlagValues instance used for parsing.
        (default: absl.flags.FLAGS)
    lock_config: If set to True, loaded config will be locked through calling
        .lock() method on its instance (if it exists). (default: True)
    **kwargs: Optional keyword arguments passed to Flag constructor.

  Returns:
    a handle to defined flag.
  """
  parser = _ConfigFileParser(name=name, lock_config=lock_config)
  flag = _ConfigFlag(
      parser=parser,
      serializer=flags.ArgumentSerializer(),
      name=name,
      default=default,
      help_string=help_string,
      flag_values=flag_values,
      **kwargs)

  # Get the module name for the frame at depth 1 in the call stack.
  module_name = sys._getframe(1).f_globals.get('__name__', None)  # pylint: disable=protected-access
  module_name = sys.argv[0] if module_name == '__main__' else module_name
  return flags.DEFINE_flag(flag, flag_values, module_name=module_name)


def DEFINE_config_dict(  # pylint: disable=g-bad-name
    name: str,
    config: ml_collections.ConfigDict,
    help_string: str = 'ConfigDict instance.',
    flag_values: flags.FlagValues = FLAGS,
    lock_config: bool = True,
    **kwargs) -> flags.FlagHolder:
  """Defines flag for inline `ConfigDict's` compatible with absl flags.

  Similar to `DEFINE_config_file` except the flag's value should be a
  `ConfigDict` instead of a path to a file containing a `ConfigDict`. After the
  flag is parsed, `FLAGS.name` will contain a reference to the `ConfigDict`,
  optionally with some values overridden.

  Typical usage example:

  `script.py`::

    from absl import flags

    import ml_collections
    from ml_collections.config_flags import config_flags


    config = ml_collections.ConfigDict({
        'field1': 1,
        'field2': 'tom',
        'nested': {
            'field': 2.23,
        }
    })


    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_dict('my_config', config)
    ...

    print(FLAGS.my_config)

  The following command::

    python script.py -- --my_config.field1 8
                        --my_config.nested.field=2.1

  will print::

    field1: 8
    field2: tom
    nested: {field: 2.1}

  Args:
    name: Flag name.
    config: `ConfigDict` object.
    help_string: Help string to display when --helpfull is called.
        (default: "ConfigDict instance.")
    flag_values: FlagValues instance used for parsing.
        (default: absl.flags.FLAGS)
    lock_config: If set to True, loaded config will be locked through calling
        .lock() method on its instance (if it exists). (default: True)
    **kwargs: Optional keyword arguments passed to Flag constructor.

  Returns:
    a handle to defined flag.
  """
  if not isinstance(config, ml_collections.ConfigDict):
    raise TypeError('config should be a ConfigDict')
  parser = _InlineConfigParser(name=name, lock_config=lock_config)
  flag = _ConfigFlag(
      parser=parser,
      serializer=flags.ArgumentSerializer(),
      name=name,
      default=config,
      help_string=help_string,
      flag_values=flag_values,
      **kwargs)

  # Get the module name for the frame at depth 1 in the call stack.
  module_name = sys._getframe(1).f_globals.get('__name__', None)  # pylint: disable=protected-access
  module_name = sys.argv[0] if module_name == '__main__' else module_name
  return flags.DEFINE_flag(flag, flag_values, module_name=module_name)


# Note that we would add a bound to constrain this to be a dataclass, except
# that dataclasses don't have a specific base class, and structural typing for
# attributes is currently (2021Q1) not supported in pytype (b/150927776).
_T = TypeVar('_T')


class _TypedFlagHolder(flags.FlagHolder, Generic[_T]):
  """A typed wrapper for a FlagHolder."""

  def __init__(self, flag: flags.FlagHolder, ctor: Type[_T]):
    self._flag = flag
    self._ctor = ctor

  @property
  def value(self) -> _T:
    return self._ctor(**self._flag.value)

  @property
  def default(self) -> _T:
    return self._ctor(**self._flag.default)

  @property
  def name(self) -> str:
    return self._flag.name


def DEFINE_config_dataclass(  # pylint: disable=invalid-name
    name: str,
    config: _T,
    help_string: str = 'Configuration object. Must be a dataclass.',
    flag_values: flags.FlagValues = FLAGS,
    **kwargs,
) -> _TypedFlagHolder[_T]:
  """Defines a typed (dataclass) flag-overrideable configuration.

  Similar to `DEFINE_config_dict` except `config` should be a `dataclass`.

  Args:
    name: Flag name.
    config: A user-defined configuration object. Must be built via `dataclass`.
    help_string: Help string to display when --helpfull is called.
    flag_values: FlagValues instance used for parsing.
    **kwargs: Optional keyword arguments passed to Flag constructor.
  Returns:
    A handle to the defined flag.
  """

  if not dataclasses.is_dataclass(config):
    raise ValueError('Configuration object must be a `dataclass`.')

  # Convert to configdict *without* recursing into leaf node(s).
  # If our config contains dataclasses (or other types) as fields, we want to
  # preserve them; dataclasses.asdict recursively turns all fields into dicts.
  dictionary = {field.name: getattr(config, field.name)
                for field in dataclasses.fields(config)}
  config_dict = ml_collections.ConfigDict(initial_dictionary=dictionary)

  # Define the flag.
  config_flag = DEFINE_config_dict(
      name,
      config=config_dict,
      help_string=help_string,
      flag_values=flag_values,
      **kwargs,
  )

  return _TypedFlagHolder(flag=config_flag, ctor=config.__class__)


def get_config_filename(config_flag) -> str:  # pylint: disable=g-bad-name
  """Returns the path to the config file given the config flag.

  Args:
    config_flag: The flag instance obtained from FLAGS, e.g. FLAGS['config'].

  Returns:
    the path to the config file.
  """
  if not is_config_flag(config_flag):
    raise TypeError('expect a config flag, found {}'.format(type(config_flag)))
  return config_flag.config_filename


def get_override_values(config_flag) -> Dict[str, Any]:  # pylint: disable=g-bad-name
  """Returns a flat dict containing overridden values from the config flag.

  Args:
    config_flag: The flag instance obtained from FLAGS, e.g. FLAGS['config'].

  Returns:
    a flat dict containing overridden values from the config flag.
  """
  if not is_config_flag(config_flag):
    raise TypeError('expect a config flag, found {}'.format(type(config_flag)))
  return config_flag.override_values


class _IgnoreFileNotFoundAndCollectErrors:
  """Helps recording "file not found" exceptions when loading config.

  Usage:
    ignore_errors = _IgnoreFileNotFoundAndCollectErrors()
    with ignore_errors.Attempt('Loading from foo', 'bar.id'):
      ...
      return True  # successfully loaded from `foo`
    logging.error('Failed loading: {}'.format(ignore_errors.DescribeAttempts()))
  """

  def __init__(self):
    self._attempts = []  # type: List[Tuple[Tuple[str, str], IOError]]

  def Attempt(self, description, path):
    """Creates a context manager that routes exceptions to this class."""
    self._current_attempt = (description, path)
    ignore_errors = self

    class _ContextManager:

      def __enter__(self):
        return self

      def __exit__(self, exc_type, exc_value, unused_traceback):
        return ignore_errors.ProcessAttemptException(exc_type, exc_value)

    return _ContextManager()

  def ProcessAttemptException(self, exc_type, exc_value):
    expected_type = IOError if six.PY2 else FileNotFoundError  # pylint: disable=undefined-variable
    if exc_type is expected_type and exc_value.errno == errno.ENOENT:
      self._attempts.append((self._current_attempt, exc_value))
      # Returning a true value suppresses exceptions:
      # https://docs.python.org/2/reference/datamodel.html#object.__exit__
      return True

  def DescribeAttempts(self):
    return '\n'.join(
        '  Attempted [{}]:\n    {}\n      {}'.format(attempt[0], attempt[1], e)
        for attempt, e in self._attempts)


def _LoadConfigModule(name: str, path: str):
  """Loads a script from external file specified by path.

  Unprefixed path is looked for in the current working directory using
  regular file open operation. This should work with relative config paths.

  Args:
    name: Name of the new module.
    path: Path to the .py file containing the module.

  Returns:
    Module loaded from the given path.

  Raises:
    IOError: If the config file cannot be found.
  """
  if not path:
    raise IOError('Path to config file is an empty string.')

  ignoring_errors = _IgnoreFileNotFoundAndCollectErrors()

  # Works for relative paths.
  with ignoring_errors.Attempt('Relative path', path):
    config_module = imp.load_source(name, path)
    return config_module

  # Nothing worked. Log the paths that were attempted.
  raise IOError('Failed loading config file {}\n{}'.format(
      name, ignoring_errors.DescribeAttempts()))


class _ErrorConfig:
  """Dummy ConfigDict that raises an error on any attribute access."""

  def __init__(self, error):
    super(_ErrorConfig, self).__init__()
    super(_ErrorConfig, self).__setattr__('_error', error)

  def __getattr__(self, attr):
    self._ReportError()

  def __setattr__(self, attr, value):
    self._ReportError()

  def __delattr__(self, attr):
    self._ReportError()

  def __getitem__(self, key):
    self._ReportError()

  def __setitem__(self, key, value):
    self._ReportError()

  def __delitem__(self, key):
    self._ReportError()

  def _ReportError(self):
    raise IOError('Configuration is not available because of an earlier '
                  'failure to load: ' +
                  # 'message' is not available in Python 3.
                  getattr(self._error, 'message', str(self._error)))


def _LockConfig(config):
  """Calls config.lock() if config has a lock method."""
  if isinstance(config, _ErrorConfig):
    pass  # Attempting to access _ErrorConfig.lock will raise its error.
  elif getattr(config, 'lock', None) and callable(config.lock):
    config.lock()
  else:
    pass  # config.lock() does not have desired semantics, do nothing.


class _ConfigFileParser(flags.ArgumentParser):
  """Parser for config files."""

  def __init__(self, name, lock_config=True):
    self.name = name
    self._lock_config = lock_config

  def parse(self, path):
    """Loads a config module from `path` and returns the `get_config()` result.

    If a colon is present in `path`, everything to the right of the first colon
    is passed to `get_config` as an argument. This allows the structure of what
    is returned to be modified, which is useful when performing complex
    hyperparameter sweeps.

    Args:
      path: string, path pointing to the config file to execute. May also
          contain a config_string argument, e.g. be of the form
          "config.py:some_configuration".
    Returns:
      Result of calling `get_config` in the specified module.
    """
    # This will be a 2 element list iff extra configuration args are present.
    split_path = path.split(':', 1)
    try:
      config_module = _LoadConfigModule('{}_config'.format(self.name),
                                        split_path[0])
      config = config_module.get_config(*split_path[1:])
      if config is None:
        logging.warning(
            '%s:get_config() returned None, did you forget a return statement?',
            path)
    except IOError as e:
      # Don't raise the error unless/until the config is actually accessed.
      config = _ErrorConfig(e)
    # Third party flags library catches TypeError and ValueError and rethrows,
    # removing useful information unless it is added here (b/63877430):
    except (TypeError, ValueError) as e:
      error_trace = traceback.format_exc()
      raise type(e)('Error whilst parsing config file:\n\n' + error_trace)

    if self._lock_config:
      _LockConfig(config)

    return config

  def flag_type(self):
    return 'config object'


class _InlineConfigParser(flags.ArgumentParser):
  """Parser for a config defined inline (not from a file)."""

  def __init__(self, name, lock_config=True):
    self.name = name
    self._lock_config = lock_config

  def parse(self, config):
    if not isinstance(config, ml_collections.ConfigDict):
      raise TypeError('Overriding {} is not allowed.'.format(self.name))
    if self._lock_config:
      _LockConfig(config)
    return config

  def flag_type(self):
    return 'config object'


class _ConfigFlag(flags.Flag):
  """Flag definition for command-line overridable configs."""

  def __init__(self, flag_values=FLAGS, **kwargs):
    # Parent constructor can already call .Parse, thus additional fields
    # have to be set here.
    self.flag_values = flag_values
    super(_ConfigFlag, self).__init__(**kwargs)

  def _GetOverrides(self, argv):
    """Parses the command line arguments for the overrides."""
    overrides = []
    config_index = self._FindConfigSpecified(argv)
    for i, arg in enumerate(argv):
      if re.match(r'-{{1,2}}(no)?{}\.'.format(self.name), arg):
        if config_index > 0 and i < config_index:
          raise FlagOrderError('Found {} in argv before a value for --{} '
                               'was specified'.format(arg, self.name))
        arg_name = arg.split('=', 1)[0]
        overrides.append(arg_name.split('.', 1)[1])
    return overrides

  def _FindConfigSpecified(self, argv):
    """Finds element in argv specifying the value of the config flag.

    Args:
      argv: command line arguments as a list of strings.
    Returns:
      Index in argv if found and -1 otherwise.
    """
    for i, arg in enumerate(argv):
      # '-(-)config' followed by '=' or at the end of the string.
      if re.match(r'^-{{1,2}}{}(=|$)'.format(self.name), arg) is not None:
        return i
    return -1

  def _IsConfigSpecified(self, argv):
    """Returns `True` if the config file is specified on the command line."""
    return self._FindConfigSpecified(argv) >= 0

  def _set_default(self, default):
    if self._IsConfigSpecified(sys.argv):
      self.default = default
    else:
      super(_ConfigFlag, self)._set_default(default)  # pytype: disable=attribute-error
    self.default_as_str = "'{}'".format(default)

  def _parse(self, argument):
    # Parse config
    config = super(_ConfigFlag, self)._parse(argument)

    # Get list or overrides
    overrides = self._GetOverrides(sys.argv)
    # Attach types definitions
    overrides_types = GetTypes(overrides, config)

    # Iterate over overridden fields and create valid parsers
    self._override_values = {}
    for field_path, field_type in zip(overrides, overrides_types):
      field_help = 'An override of {}\'s field {}'.format(self.name, field_path)
      field_name = '{}.{}'.format(self.name, field_path)

      if field_type in _FIELD_TYPE_TO_PARSER:
        parser = _ConfigFieldParser(_FIELD_TYPE_TO_PARSER[field_type],
                                    field_path, config, self._override_values)
        flags.DEFINE(
            parser,
            field_name,
            GetValue(field_path, config),
            field_help,
            flag_values=self.flag_values,
            serializer=_FIELD_TYPE_TO_SERIALIZER[field_type])
        flag = self.flag_values._flags().get(field_name)  # pylint: disable=protected-access
        flag.boolean = field_type is bool
      else:
        raise UnsupportedOperationError(
            "Type {} of field {} is not supported for overriding. "
            "Currently supported types are: {}. (Note that tuples should "
            "be passed as a string on the command line: flag='(a, b, c)', "
            "rather than flag=(a, b, c).)".format(
                field_type, field_name, _FIELD_TYPE_TO_PARSER.keys()))

    self._config_filename = argument
    return config

  @property
  def config_filename(self):
    """Returns a path to a config file.

    Typical usage example:
    `script.py`:

    ```python
    ...
    from absl import flags
    from ml_collections.config_flags import config_flags

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
      name='my_config',
      default='ml_collections/config_flags/tests/configdict_config.py',
      help_string='config file')
    ...

    FLAGS['my_config'].config_filename

    will output
    'ml_collections/config_flags/tests/configdict_config.py'
    ```

    Returns:
      A path to a config file. For a parameterised get_config, the config
      filename with the provided parameterisation is returned.

    Raises:
      UnparsedFlagError: if the flag has not been parsed.

    """
    if not hasattr(self, '_config_filename'):
      raise UnparsedFlagError('The flag has not been parsed yet')
    return self._config_filename

  @property
  def override_values(self):
    """Returns a flat dictionary containing overridden values.

    Keys in the dictionary are dot-separated paths navigating to child items in
    the original configuration. For example, supppose that a `config` flag is
    defined and initialized to the following configuration:

    ```python
    {
        'a': 1,
        'nested': {
            'b': 2
        }
    }
    ```

    and the user overrides both values using command-line flags:

    ```
    --config.a=10 --config.nested.b=20
    ```

    Then `FLAGS['config'].override_values` will return:

    ```python
    {
        'a': 10,
        'nested.b': 20
    }
    ```

    The result can be passed to `ConfigDict.update_from_flattened_dict` to
    update the values in a configuration. Continuing with the example above:

    ```python
    import ml_collections
    config = ml_collections.ConfigDict{
        'a': 123,
        'nested': {
            'b': 456
        }
    }
    config.update_from_flattened_dict(FLAGS['config'].override_values)
    print(config.a)  # Prints `10`.
    print(config.nested.b)  # Prints `20`.
    ```

    Returns:
      Flat dictionary with overridden values.

    Raises:
      UnparsedFlagError: if the flag has not been parsed.
    """
    if not hasattr(self, '_override_values'):
      raise UnparsedFlagError('The flag has not been parsed yet')
    return self._override_values


def is_config_flag(flag):  # pylint: disable=g-bad-name
  """Returns True iff `flag` is an instance of `_ConfigFlag`.

  External users of the library may need to check if a flag is of this type
  or not, particularly because ConfigFlags should be parsed before any other
  flags. This function allows that test to be done without making the whole
  class public.

  Args:
    flag: Flag object.

  Returns:
    True iff `isinstance(flag, _ConfigFlag)` is true.
  """
  return isinstance(flag, _ConfigFlag)


class _ConfigFieldParser(flags.ArgumentParser):
  """Parser with config update after parsing.

  This class-based wrapper creates a new object, which uses
  existing parser to do actual parsing and attaches a single callback to
  SetValue afterwards, which is used to update a predefined path in
  the config object.
  """

  def __init__(
      self,
      parser: flags.ArgumentParser,
      path: str,
      config: ml_collections.ConfigDict,
      override_values: MutableMapping[str, Any]):
    """Creates new parser with callback, using existing one to perform parsing.

    Args:
      parser: ArgumentParser instance to wrap.
      path: Dot separated path in config to update with the result of
          parser.parse(...)
      config: Reference to the config object.
      override_values: Dictionary with override values. The 'parse' method will
          add the parsed value to this dictionary with key `path`.
    """
    self._parser = parser
    self._path = path
    self._config = config
    self._override_values = override_values

  def __getattr__(self, attr):
    return getattr(self._parser, attr)

  def parse(self, argument):  # pylint: disable=invalid-name
    value = self._parser.parse(argument)
    SetValue(self._path, self._config, value)
    self._override_values[self._path] = value
    return value

  def flag_type(self) -> str:
    return self._parser.flag_type()

  @property
  def syntactic_help(self) -> str:
    return self._parser.syntactic_help


def _ExtractIndicesFromStep(step):
  """Separates identifier from indexes. Example: id[1][10] -> 'id', [1, 10]."""
  # Map returns an iterable in Python 3, thus the additional `list`.
  return step.split('[', 1)[0], list(map(int, re.findall(r'\[(\d+)\]', step)))


def _AccessConfig(current, field, indices):
  """Returns member of the current config.

  The field in `current` specified by the `field` parameter is indexed using the
  values in the `indices` indices.

  Example:

  ```python
  current = {'first': [0], 'second': [[1]]}
  _AccessConfig(current, 'first', [0]) # Returns 0
  _AccessConfig(current, 'second', [0, 0]) # Returns 1

  ```

  Args:
    current: Current config.
    field: Field to access (string).
    indices: List of indices.

  Returns:
    Member of the config.

  Raises:
    IndexError: if indices are invalid.
    KeyError: if the field is not found.
  """
  if isinstance(field, string_types) and hasattr(current, field):
    current = getattr(current, field)
  elif hasattr(current, '__getitem__'):
    current = current[field]
  else:
    raise KeyError(field)

  for i, index in enumerate(indices):
    try:
      current = current[index]
    except TypeError:
      msg = 'Could not index config {}'.format(field)
      if i > 0:
        msg += '[{}]'.format(']['.join(map(str, indices[:i])))
      raise IndexError(msg)
    except IndexError:
      msg = 'Index [{}] not found in config {}'.format(
          ']['.join(map(str, indices[i:])), field)
      if i > 0:
        msg += '[{}]'.format(']['.join(map(str, indices[:i])))
      raise IndexError(msg)

  return current


def _TakeStep(current, step):
  field, indices = _ExtractIndicesFromStep(step)
  return _AccessConfig(current, field, indices)


def GetValue(path: str, config: ml_collections.ConfigDict):
  """Gets value of a single field."""
  current = config
  for step in path.split('.'):
    current = _TakeStep(current, step)
  return current


def GetType(path, config):
  """Gets type of field in config described by a dotted delimited path."""
  steps = path.split('.')
  current = config
  for step in steps[:-1]:
    current = _TakeStep(current, step)

  # We need to check if there is list-type indexing in the last step.
  field, indices = _ExtractIndicesFromStep(steps[-1])
  if indices:
    return type(_AccessConfig(current, field, indices))
  else:
    # Check if current is a DM collection and hence has attribute get_type()
    is_dm_collection = isinstance(current,
                                  ml_collections.ConfigDict) or isinstance(
                                      current, ml_collections.FieldReference)
    if is_dm_collection:
      return current.get_type(steps[-1])
    else:
      return type(_TakeStep(current, steps[-1]))


def GetTypes(paths, config):
  """Gets types of fields in config described by dotted delimited paths."""
  return [GetType(path, config) for path in paths]


def SetValue(path: str, config: ml_collections.ConfigDict, value):
  """Sets value of a single field."""
  current = config
  steps = path.split('.')

  if not steps or not path:
    raise ValueError('Path cannot be empty')

  for step in steps[:-1]:
    current = _TakeStep(current, step)

  field, indices = _ExtractIndicesFromStep(steps[-1])
  if indices:
    current = _AccessConfig(current, field, indices[:-1])
    try:
      current[indices[-1]] = value
    except:
      raise UnsupportedOperationError(
          'Config does not have a setter for {}'.format(path))
  else:
    if hasattr(current, '__setitem__') and field in current:
      current[field] = value
    elif hasattr(current, field):
      setattr(current, field, value)
    else:
      raise UnsupportedOperationError(
          'Config does not have a setter for {}'.format(path))
