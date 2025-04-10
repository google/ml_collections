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

"""Configuration commmand line parser."""

import ast
import copy
import dataclasses
import enum
import errno
import functools as ft
import importlib.machinery
import os
import re
import sys
import traceback
import types
from typing import Any, Callable, Dict, Generic, List, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar

from absl import flags
from absl import logging
from ml_collections import config_dict
from ml_collections.config_flags import config_path
from ml_collections.config_flags import tuple_parser

FLAGS = flags.FLAGS

# Forward for backwards compatibility.
GetValue = config_path.get_value
GetType = config_path.get_type
SetValue = config_path.set_value

# Any flags defined via any of the config_flags.DEFINE_config_*() functions
# should be attributed to the caller of that function rather than this module.
flags.disclaim_key_flags()


def _load_source(module_name: str, module_path: str) -> types.ModuleType:
  """Loads a Python module from its source file.

  Args:
    module_name: name of the module in sys.modules.
    module_path: path to the Python file containing the module.

  Returns:
    The loaded Python module.
  """
  loader = importlib.machinery.SourceFileLoader(module_name, module_path)
  return loader.load_module()


class _LiteralParser(flags.ArgumentParser):
  """Parse arbitrary built-in (`--cfg.val=1`, `--cfg.val="[1, 2, {}]"`,...)."""

  def parse(self, argument: str) -> Any:
    # _LiteralParser cannot know in advance what is the expected type.
    # The default value is never passed, as default is overwritten to `None`
    # bellow inside `_ConfigFlag._parse`.
    if not isinstance(argument, str):
      raise TypeError('argument should be a string')
    # Absl hardcode bool values as lower-case: `--cfg.my_bool`, so convert
    # them to Python built-in
    if argument in ('true', 'false'):
      argument = argument.capitalize()
    try:
      return ast.literal_eval(argument)
    except (SyntaxError, ValueError):
      # Otherwise, the flag is a string: `--cfg.value="my_string"`
      return argument

  def flag_type(self):
    return 'config_literal'


class _ConfigDictParser(flags.ArgumentParser):
  """Parser for ConfigDict values."""

  def parse(self, argument: str) -> config_dict.ConfigDict:
    try:
      value = ast.literal_eval(argument)
    except (SyntaxError, ValueError) as e:
      raise ValueError(
          f'Failed to parse {argument!r} as a ConfigDict: {e!r}'
      ) from None

    if not isinstance(value, dict):
      raise ValueError(
          f'Failed to parse {argument!r} as a ConfigDict: `{value!r}` is not a'
          ' dict.'
      )
    return config_dict.ConfigDict(value)


_FIELD_TYPE_TO_PARSER = {
    float: flags.FloatParser(),
    bool: flags.BooleanParser(),
    tuple: tuple_parser.TupleParser(),
    int: flags.IntegerParser(),
    str: flags.ArgumentParser(),
    config_dict.ConfigDict: _ConfigDictParser(),
    object: _LiteralParser(),
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
    accept_new_attributes: bool = False,
    sys_argv: Optional[List[str]] = None,
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

    from ml_collections import config_flags

    _CONFIG = config_flags.DEFINE_config_file('my_config')

    print(_CONFIG.value)

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
    help_string: Help string to display when --helpfull is called. (default:
      "path to config file.")
    flag_values: FlagValues instance used for parsing. (default:
      absl.flags.FLAGS)
    lock_config: If set to True, loaded config will be locked through calling
      .lock() method on its instance (if it exists). (default: True)
    accept_new_attributes: If `True`, accept to pass arbitrary attributes that
      are not originally defined in the `get_config()` dict.
      `accept_new_attributes` requires `lock_config=False`.
    sys_argv: If set, interprets this as the full list of args used in parsing.
      This is used to identify which overrides to define as flags. If not
      specified, uses the system sys.argv to figure it out.
    **kwargs: Optional keyword arguments passed to Flag constructor.

  Returns:
    a handle to defined flag.
  """
  if accept_new_attributes and lock_config:
    raise ValueError('`accept_new_attributes=True` requires lock_config=False')
  parser = ConfigFileFlagParser(name=name, lock_config=lock_config)
  serializer = flags.ArgumentSerializer()
  flag = _ConfigFlag(
      parser=parser,
      serializer=serializer,
      name=name,
      default=default,
      help_string=help_string,
      flag_values=flag_values,
      accept_new_attributes=accept_new_attributes,
      sys_argv=sys_argv,
      **kwargs)

  return flags.DEFINE_flag(flag, flag_values)


def DEFINE_config_dict(  # pylint: disable=g-bad-name
    name: str,
    config: config_dict.ConfigDict,
    help_string: str = 'ConfigDict instance.',
    flag_values: flags.FlagValues = FLAGS,
    lock_config: bool = True,
    accept_new_attributes: bool = False,
    sys_argv: Optional[List[str]] = None,
    **kwargs) -> flags.FlagHolder:
  """Defines flag for inline `ConfigDict's` compatible with absl flags.

  Similar to `DEFINE_config_file` except the flag's value should be a
  `ConfigDict` instead of a path to a file containing a `ConfigDict`. After the
  flag is parsed, `FLAGS.name` will contain a reference to the `ConfigDict`,
  optionally with some values overridden.

  Typical usage example:

  `script.py`::

    from ml_collections import config_dict
    from ml_collections import config_flags


    config = config_dict.ConfigDict({
        'field1': 1,
        'field2': 'tom',
        'nested': {
            'field': 2.23,
        }
    })


    _CONFIG = config_flags.DEFINE_config_dict('my_config', config)
    ...

    print(_CONFIG.value)

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
    accept_new_attributes: If `True`, accept to pass arbitrary attributes that
      are not originally defined in the `config` argument.
      `accept_new_attributes` requires `lock_config=False`.
    sys_argv: If set, interprets this as the full list of args used in parsing.
      This is used to identify which overrides to define as flags. If not
      specified, uses the system sys.argv to figure it out.
    **kwargs: Optional keyword arguments passed to Flag constructor.

  Returns:
    a handle to defined flag.
  """
  if not isinstance(config, config_dict.ConfigDict):
    raise TypeError('config should be a ConfigDict')
  if accept_new_attributes and lock_config:
    raise ValueError('`accept_new_attributes=True` requires lock_config=False')
  parser = _InlineConfigParser(name=name, lock_config=lock_config)
  flag = _ConfigFlag(
      parser=parser,
      serializer=flags.ArgumentSerializer(),
      name=name,
      default=config,
      help_string=help_string,
      flag_values=flag_values,
      accept_new_attributes=accept_new_attributes,
      sys_argv=sys_argv,
      **kwargs)

  # Get the module name for the frame at depth 1 in the call stack.
  module_name = sys._getframe(1).f_globals.get('__name__', None)  # pylint: disable=protected-access
  module_name = sys.argv[0] if module_name == '__main__' else module_name
  return flags.DEFINE_flag(flag, flag_values, module_name=module_name)


# Note that we would add a bound to constrain this to be a dataclass, except
# that dataclasses don't have a specific base class, and structural typing for
# attributes is currently (2021Q1) not supported in pytype (b/150927776).
_T = TypeVar('_T')


class  _DataclassParser(flags.ArgumentParser, Generic[_T]):
  """Parser for a config defined inline (not from a file)."""

  def __init__(self, name: str, dataclass_type: Type[_T],
               parse_fn: Optional[Callable[[Any], _T]] = None):
    self.name = name
    self.dataclass_type = dataclass_type
    self.parse_fn = parse_fn

  def _get_parse_fn(self):
    # NB: We are using lazy lookup in the parse function rather than
    # constructor, to give client's code a chance to create their parsers
    # and provide consistency with availability of parsers for nested fields
    # (which are also looked up during parsing).
    if self.parse_fn:
      return self.parse_fn
    if self.dataclass_type in _FIELD_TYPE_TO_PARSER:
      return _FIELD_TYPE_TO_PARSER[self.dataclass_type].parse

  def parse(self, config: Any) -> _T:
    # It is important to use deepcopy here, so if parser returns constants
    # they are not modified during the flag parsing.
    if isinstance(config, self.dataclass_type):
      return copy.deepcopy(config)
    parse_fn = self._get_parse_fn()
    if parse_fn:
      return copy.deepcopy(parse_fn(config))
    raise TypeError(
        'Overriding {} is not allowed: it has neither '
        'registered parser nor explicit parse_fn in the flag '
        'definition'.format(self.name)
    )

  def flag_type(self):
    return 'config_dataclass({})'.format(self.dataclass_type)


def DEFINE_config_dataclass(  # pylint: disable=invalid-name
    name: str,
    config: _T,
    help_string: str = 'Configuration object. Must be a dataclass.',
    flag_values: flags.FlagValues = FLAGS,
    sys_argv: Optional[List[str]] = None,
    parse_fn: Optional[Callable[[Any], _T]] = None,
    **kwargs,
) -> flags.FlagHolder[_T]:
  """Defines a typed (dataclass) flag-overrideable configuration.

  Similar to `DEFINE_config_dict` except `config` should be a `dataclass`.

  The config value can contain nested fields, including other dataclasses.
  If a field is of form  Optional[dataclass] with None as a default value,
  it can be explicitly initialized using special value `build`. E.g.
  For instance:

  ```
  @dc.dataclass
  class FancyLoss
    foo_scale: float = 0.1

  @dc.dataclass
  class Config:
    fancy_loss: Optional[FancyLoss] = None
  ```

  Then if `--config.fancy_loss=build  --config.fancy_loss.foo_scale=1` will
  instantiate and override foo_scale to 1.  Note: that the reverse
  order is not allowed:  `--config.fancy_loss.foo_scale=1
  --config.fancy_loss=build` will cause FlagOrderError.

  Optional dataclass fields can also be set to None using special `none` value.
  For instance:

  ```
  @dc.dataclass
  class FancyLossConfig
    foo_loss_scale: float = 0.1

  @dc.dataclass
  class Config:
    fancy_loss: Optional[FancyLossConfig] = FancyLossConfig()
  ```

  Then `--config.fancy_loss=none`, will set it to None.

  Implementation note: This flag will register all the needed nested flags
  dynamically based on sys.argv or sys_argv, in order to support
  free-text keyed flags such as `foo.bar['i']=1`. Because of how flag subsystem
  works this will happen either at flag-parsing time (during app.run),
  if there is a root level override, such as `--config=<..>` or during
  declaration otherwise (during this invocation).
  Parsing at declaration (e.g. if no root override) can cause problems
  with multiprocessing  since sys.argv is not yet populated at
  declaration time for spawn processes. To avoid this, pass custom sys_argv
  value instead if you want to use this library with multiprocessing.
  Also note, in the future we might consider to always do it at declaration
  time, as this cleans up the logic significantly.

  Args:
    name: Flag name.
    config: A user-defined configuration object. Must be built via `dataclass`.
    help_string: Help string to display when --helpfull is called.
    flag_values: FlagValues instance used for parsing.
    sys_argv: If set, interprets this as the full list of args used in parsing.
      This is used to identify which overrides to define as flags. If not
      specified, uses the system sys.argv to figure it out.
    parse_fn: Function that can parse provided flag value, when assigned via
      flag.value, or passed on command line. If not provided, but the class has
      registered parser register_flag_parser_for_type, the latter will be used.
      Otherwise only allows to assign instances of this class.
    **kwargs: Optional keyword arguments passed to Flag constructor.

  Returns:
    A handle to the defined flag.
  """

  if not dataclasses.is_dataclass(config):
    raise ValueError('Configuration object must be a `dataclass`.')
  # Define the flag.
  parser = _DataclassParser(name=name, dataclass_type=type(config),
                            parse_fn=parse_fn)
  flag = _ConfigFlag(
      flag_values=flag_values,
      parser=parser,
      serializer=flags.ArgumentSerializer(),
      name=name,
      default=config,
      help_string=help_string,
      sys_argv=sys_argv,
      **kwargs)

  return flags.DEFINE_flag(flag, flag_values)


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
    if exc_type is FileNotFoundError and exc_value.errno == errno.ENOENT:  # pytype: disable=attribute-error  # trace-all-classes
      self._attempts.append((self._current_attempt, exc_value))  # pytype: disable=container-type-mismatch  # trace-all-classes
      # Returning a true value suppresses exceptions:
      # https://docs.python.org/2/reference/datamodel.html#object.__exit__
      return True

  def DescribeAttempts(self):
    return '\n'.join(
        '  Attempted [{}]:\n    {}\n      {}'.format(attempt[0], attempt[1], e)
        for attempt, e in self._attempts)


def _MakeDefaultOrNone(kls, config, allow_none=True, field_path=''):
  if config in ['build', True]:
    try:
      return kls()
    except Exception as e:
      raise ValueError(
          f'Unable to create default instance for "{field_path}" '
          f'of type "{kls}": {e}') from e

  elif (config in ['0', 0, False] or config.lower() == 'none'):
    if not allow_none:
      raise ValueError(f'None is not allowed as value for "{field_path}", '
                       'as the dataclass field is not marked as optional.')
    return None
  raise ValueError(f'Unable to parse value "{config}" as instance of {kls}'
                   f'for {field_path} values allowed are [0/none, or 1]')


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
    config_module = _load_source(name, path)
    return config_module

  # Nothing worked. Log the paths that were attempted.
  raise IOError('Failed loading config file {}\n{}'.format(
      name, ignoring_errors.DescribeAttempts()))


class _ErrorConfig:
  """ConfigDict object that raises an error on any attribute access."""

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

  def _ReportError(self):  # pylint: disable=invalid-name
    raise IOError(
        'Configuration is not available because of an earlier failure to load:'
        f' {self._error}'
    ) from self._error


def _LockConfig(config):
  """Calls config.lock() if config has a lock method."""
  if isinstance(config, _ErrorConfig):
    pass  # Attempting to access _ErrorConfig.lock will raise its error.
  elif getattr(config, 'lock', None) and callable(config.lock):
    config.lock()
  else:
    pass  # config.lock() does not have desired semantics, do nothing.


class ConfigFileFlagParser(flags.ArgumentParser):
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


# Alias to an older name, for backwards compatibility.
_ConfigFileParser = ConfigFileFlagParser


class _InlineConfigParser(flags.ArgumentParser):
  """Parser for a config defined inline (not from a file)."""

  def __init__(self, name, lock_config=True):
    self.name = name
    self._lock_config = lock_config

  def parse(self, config):
    if not isinstance(config, config_dict.ConfigDict):
      raise TypeError('Overriding {} is not allowed.'.format(self.name))
    if self._lock_config:
      _LockConfig(config)
    return config

  def flag_type(self):
    return 'config object'


class _ConfigFlag(flags.Flag):
  """Flag definition for command-line overridable configs."""

  def __init__(
      self,
      flag_values=FLAGS,
      *,
      accept_new_attributes: bool = False,
      sys_argv=None,
      **kwargs,
  ):
    # Parent constructor can already call .Parse, thus additional fields
    # have to be set here.
    self.flag_values = flag_values
    self._accept_new_attributes = accept_new_attributes
    # Note, we don't replace sys_argv with sys.argv here if it's None because
    # in some obscure multiprocessing use cases, sys.argv may not be populated
    # until later and we need to look it up at parse time.
    self._sys_argv = sys_argv
    super(_ConfigFlag, self).__init__(**kwargs)

  def _GetArgv(self):
    """Lazily fetches sys.argv and expands any potential --flagfile=..."""
    argv = sys.argv if self._sys_argv is None else self._sys_argv
    argv = flags.FLAGS.read_flags_from_files(argv, force_gnu=False)
    return argv

  def _GetOverrides(self, argv):
    """Parses the command line arguments for the overrides."""
    # We use a dict to keep the order of the overrides.
    overrides = dict()
    config_index = self._FindConfigSpecified(argv)
    for i, arg in enumerate(argv):
      if re.match(r'-{{1,2}}(no)?{}\.'.format(self.name), arg):
        if config_index > 0 and i < config_index:
          raise FlagOrderError('Found {} in argv before a value for --{} '
                               'was specified'.format(arg, self.name))
        arg_name = arg.split('=', 1)[0]
        override_name = arg_name.split('.', 1)[1]
        # Reinsert override_name into overrides to keep the order of
        # the "last" flag.
        overrides[override_name] = overrides.pop(override_name, None)
    return list(overrides)

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
    if self._IsConfigSpecified(self._GetArgv()):
      self.default = default
    else:
      super(_ConfigFlag, self)._set_default(default)  # pytype: disable=attribute-error
    self.default_as_str = "'{}'".format(default)

  def _validate_overrides(self, config, overrides: Sequence[str]):
    # Verify that we don't provide --config.foo.bar=1 followed by override of
    # config.foo.
    for i, override_a in enumerate(overrides, 1):
      for override_b in overrides[i:]:
        # verify if override_b will overwrite override_a
        if override_a.startswith(override_b + '.'):
          raise FlagOrderError(
              f'Flag --{self.name}.{override_b} is provided after '
              f'--{self.name}.{override_a} and '
              'it will overwrite the value provided in '
              f'--{self.name}.{override_a}, '
              'which is probably not what you expect.')

  def _initialize_missing_parent_fields(self, config, overrides):
    for override in overrides:
      config_path.initialize_missing_parent_fields(
          config, override, overrides)

  def _parse(self, argument):
    # Parse config
    config = super(_ConfigFlag, self)._parse(argument)

    # Get list or overrides
    overrides = self._GetOverrides(self._GetArgv())
    # Iterate over overridden fields and create valid parsers
    self._override_values = {}
    self._initialize_missing_parent_fields(config, overrides)
    self._validate_overrides(config, overrides)

    if self._accept_new_attributes:
      # If user provide a new attribute, fallback to `object` to accept all
      # literal
      default_type = object
    else:
      default_type = None

    for field_path in overrides:
      field_type = config_path.get_type(
          field_path, config, default_type=default_type
      )
      field_type_origin = config_path.get_origin(field_type)
      field_help = 'An override of {}\'s field {}'.format(self.name, field_path)
      field_name = '{}.{}'.format(self.name, field_path)

      parser = None
      if field_type in _FIELD_TYPE_TO_PARSER:
        parser = _FIELD_TYPE_TO_PARSER[field_type]
      elif isinstance(field_type, type) and issubclass(
          field_type, config_dict.ConfigDict
      ):
        # Supports ConfigDict sub-classes.
        parser = _FIELD_TYPE_TO_PARSER[config_dict.ConfigDict]
      elif field_type_origin and field_type_origin in _FIELD_TYPE_TO_PARSER:
        parser = _FIELD_TYPE_TO_PARSER[field_type_origin]
      elif issubclass(field_type, enum.Enum):
        parser = flags.EnumClassParser(field_type, case_sensitive=False)
      elif dataclasses.is_dataclass(field_type):
        # For dataclasses-valued fields allow default instance creation.
        is_optional = config_path.is_optional(field_path, config)
        parser = _DataclassParser(
            name=field_path, dataclass_type=field_type,
            parse_fn=ft.partial(_MakeDefaultOrNone, field_type,
                                allow_none=is_optional, field_path=field_path))

      if parser:
        if not isinstance(parser, tuple_parser.TupleParser):
          if isinstance(parser, (_LiteralParser, _ConfigDictParser)):
            # We do not pass the default to `_ConfigFieldFlag`, otherwise
            # `_LiteralParser.parse(default)` is called with `default`,
            # which would try to parse string.
            # Setting the value to `None` never call `.parse`, so the
            # default value from the config is kept.
            # TODO(sandler): Investigate if value could be None for all parsers.
            default = None
          else:
            default = config_path.get_value(field_path, config)
          flag = _ConfigFieldFlag(
              path=field_path,
              config=config,
              override_values=self._override_values,
              parser=parser,
              serializer=flags.ArgumentSerializer(),
              name=field_name,
              default=default,
              accept_new_attributes=self._accept_new_attributes,
              help_string=field_help,
          )
          # Literal values support the `--my_bool` / `--nomy_bool` syntax
          flag.boolean = field_type is bool or isinstance(
              parser, _LiteralParser
          )
          flags.DEFINE_flag(flag=flag, flag_values=self.flag_values)
        elif field_name not in self.flag_values:
          # Overriding a tuple field. Define the flag only once -- it might
          # appear multiple times on the command-line (e.g.
          # `--config.flag a --config.flag b`) but defining the same flag
          # multiple times is an error. All arguments for the same flag are
          # passed to a single call of _ConfigFieldMultiFlag.parse.
          flag = _ConfigFieldMultiFlag(
              path=field_path,
              config=config,
              override_values=self._override_values,
              parser=parser,
              serializer=flags.ArgumentSerializer(),
              name=field_name,
              default=config_path.get_value(field_path, config),
              help_string=field_help,
          )
          flags.DEFINE_flag(flag=flag, flag_values=self.flag_values)
      else:
        raise UnsupportedOperationError(
            "Type {} of field {} is not supported for overriding. "
            "Currently supported types are: {}. (Note that tuples should "
            "be passed as a string on the command line, `--flag='(a, b, c)'`, "
            "or by repeated flags, `--flag=1 --flag=2 --flag=3`, rather than "
            "--flag=(a, b, c).)".format(
                field_type, field_name, _FIELD_TYPE_TO_PARSER.keys()))

    self._config_filename = argument
    return config

  def serialize(self):
    # Use the config filename instead of the dictionary when serializing.
    return self._serialize(self.config_filename)

  @property
  def config_filename(self):
    """Returns a path to a config file.

    Typical usage example:
    `script.py`:

    ```python
    ...
    from absl import flags
    from ml_collections import config_flags

    FLAGS = flags.FLAGS
    _CONFIG = config_flags.DEFINE_config_file(
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
    from ml_collections import config_dict
    config = config_dict.ConfigDict{
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


class _ConfigFieldFlag(flags.Flag):
  """Flag for updating a field in a ConfigDict."""

  def __init__(
      self,
      path: str,
      config: config_dict.ConfigDict,
      override_values: MutableMapping[str, Any],
      *,
      parser: flags.ArgumentParser,
      serializer: flags.ArgumentSerializer,
      name: str,
      default: Any,
      help_string: str,
      short_name: Optional[str] = None,
      boolean: bool = False,
      accept_new_attributes: bool = False,
  ):
    """Creates new flag with callback."""
    super().__init__(
        parser=parser,
        serializer=serializer,
        name=name,
        default=default,
        help_string=help_string,
        short_name=short_name,
        boolean=boolean)
    self._path = path
    self._config = config
    self._override_values = override_values
    self._accept_new_attributes = accept_new_attributes

  def parse(self, argument):
    super().parse(argument)
    # Callback to set value in ConfigDict.
    config_path.set_value(
        self._path, self._config, self.value,
        accept_new_attributes=self._accept_new_attributes,
    )
    self._override_values[self._path] = self.value


class _ConfigFieldMultiFlag(flags.MultiFlag):
  """Flag for updating a tuple field in a ConfigDict."""

  def __init__(
      self,
      path: str,
      config: config_dict.ConfigDict,
      override_values: MutableMapping[str, Any],
      *,
      parser: flags.ArgumentParser,
      serializer: flags.ArgumentSerializer,
      name: str,
      default: Any,
      help_string: str,
      short_name: Optional[str] = None,
      boolean: bool = False,
  ):
    """Creates new flag with callback."""
    super().__init__(
        parser=parser,
        serializer=serializer,
        name=name,
        default=default,
        help_string=help_string,
        short_name=short_name,
        boolean=boolean)
    self._path = path
    self._config = config
    self._override_values = override_values

  def parse(self, arguments):
    super().parse(arguments)
    # Callback to set value in ConfigDict.
    config_path.set_value(self._path, self._config, tuple(self.value))
    self._override_values[self._path] = tuple(self.value)

  def _parse(self, arguments):
    # MultiFlag passes each argument one-at-a-time to the parser.parse. Just
    # call Flag._parse (grandparent class) directly so all arguments are passed
    # to parser.parse in a single call.
    result = flags.Flag._parse(self, arguments)  # pylint: disable=protected-access
    return list(result)


def register_flag_parser_for_type(
    field_type: _T, parser: flags.ArgumentParser) -> _T:
  """Registers parser for a given type.

  See documentation for `register_flag_parser` for usage example.

  Args:
    field_type: field type to register
    parser: parser to use

  Returns:
    field_type unmodified.
  """
  _FIELD_TYPE_TO_PARSER[field_type] = parser
  return field_type


def register_flag_parser(*, parser: flags.ArgumentParser) -> Callable[[_T], _T]:
  """Creates a decorator to register parser on types.

  For example:

  ```
  class ParserForCustomConfig(flags.ArgumentParser):
  def parse(self, value):
    if isinstance(value, CustomConfig):
      return value
    return CustomConfig(i=int(value), j=int(value))


  @config_flags.register_flag_parser(parser=ParserForCustomConfig())
  @dataclasses.dataclass
  class CustomConfig:
    i: int = None
    j: int = None

  class MainConfig:
    sub: CustomConfig = CustomConfig()

  config_flags.DEFINE_config_dataclass(
    'cfg', MainConfig(), 'MyConfig data')
  ```

  will declare cfg flag, then passing `--cfg.sub=1`, will initialize
  both i and j fields to 1. The fields can still be set individually:
  `--cfg.sub=1 --cfg.sub.j=3` will set `i` to `1` and `j` to `3`.

  Args:
    parser: parser to use.

  Returns:
    Decorator to apply to types.
  """
  return ft.partial(register_flag_parser_for_type, parser=parser)
